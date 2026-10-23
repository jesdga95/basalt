// src/program/tokenizer/bpe.cpp - the byte-level BPE: table load, special-token split, piece memo, merge loop.
//
// A token is known by its raw bytes (the GPT-2 byte alphabet undone at load). A merge is an id pair with a rank and the
// id of the token it makes; a piece is merged by lowest rank first, the leftmost pair on a tie.
#include "basalt/program/tokenizer.hpp"

#include <algorithm>
#include <cstring>
#include <unordered_map>

#include "basalt/program/json_scan.hpp"
#include "unicode_flags.hpp"

namespace basalt::program {

namespace {

constexpr size_t kMemoMaxEntries = 200000;   // pieces remembered
constexpr size_t kMemoMaxPiece = 64;         // longer pieces are merged each time
constexpr size_t kScanMaxSymbols = 48;       // up to here the merge loop rescans; beyond, a heap picks the pair

// the byte-level spelling: a printable byte stands for itself, the others for U+0100 upward in byte order
struct ByteAlphabet {
    int16_t from_cp[324];
    ByteAlphabet() {
        for (auto& v : from_cp) v = -1;
        uint32_t next = 256;
        for (int b = 0; b < 256; ++b) {
            const bool plain = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            from_cp[plain ? (uint32_t) b : next++] = (int16_t) b;
        }
    }
};

// `key` spelled in the byte alphabet back to bytes; false if a character is outside the alphabet (the key then stands
// for its own UTF-8 bytes)
bool unspell(std::string_view key, std::string& out) {
    static const ByteAlphabet alpha;
    out.clear();
    bool pure = true;
    for (size_t i = 0; i < key.size();) {
        const uint32_t cp = json::next_cp(key, i);
        if (cp < 324 && alpha.from_cp[cp] >= 0) out += (char) alpha.from_cp[cp];
        else { pure = false; json::put_utf8(out, cp); }
    }
    return pure;
}

inline uint64_t hash_bytes(const char* p, size_t n) {
    uint64_t h = 0x9E3779B97F4A7C15ull ^ (n * 0xFF51AFD7ED558CCDull);
    while (n >= 8) {
        uint64_t w;
        std::memcpy(&w, p, 8);
        h = (h ^ w) * 0xFF51AFD7ED558CCDull;
        h ^= h >> 32;
        p += 8;
        n -= 8;
    }
    uint64_t w = 0;
    std::memcpy(&w, p, n);
    h = (h ^ w) * 0xC4CEB9FE1A85EC53ull;
    return h ^ (h >> 29);
}

inline uint64_t pair_key(int32_t left, int32_t right) { return ((uint64_t) (uint32_t) left << 32) | (uint32_t) right; }

inline size_t pair_slot(uint64_t key, uint64_t mask) { return (size_t) (((key * 0x9E3779B97F4A7C15ull) >> 24) & mask); }

}  // namespace

bool Tokenizer::load(std::string_view vocab_json, std::string_view merges_txt, std::string_view types_json,
                     std::string& err) {
    raw_.clear();
    merges_.clear();
    memo_.clear();
    memo_used_ = 0;
    memo_keys_.clear();
    memo_ids_.clear();
    std::fill(byte_id_, byte_id_ + 256, -1);
    (void) tokenizer::cpt_flag_table();   // built here, not on the first request

    std::vector<int8_t> types;
    {
        json::Scan sc{types_json};
        if (!sc.eat('[')) { err = "token_type.json: not an array"; return false; }
        if (!sc.eat(']')) {
            do {
                int64_t t = 0;
                if (!sc.integer(t)) { err = "token_type.json: bad entry"; return false; }
                types.push_back((int8_t) t);
            } while (sc.eat(','));
        }
    }

    // vocab.json: key text -> id; the raw bytes go to raw_[id]
    std::vector<std::pair<std::string, int64_t>> entries;
    int64_t max_id = -1;
    {
        json::Scan sc{vocab_json};
        if (!sc.eat('{')) { err = "vocab.json: not an object"; return false; }
        std::string key;
        if (!sc.eat('}')) {
            do {
                int64_t id = 0;
                if (!sc.str(key) || !sc.eat(':') || !sc.integer(id)) { err = "vocab.json: bad entry"; return false; }
                entries.emplace_back(key, id);
                max_id = std::max(max_id, id);
            } while (sc.eat(','));
            if (!sc.eat('}')) { err = "vocab.json: unterminated"; return false; }
        }
    }
    if (max_id < 0) { err = "vocab.json: empty"; return false; }
    raw_.assign((size_t) max_id + 1, std::string());
    std::vector<uint8_t> have((size_t) max_id + 1, 0);   // 1: an entry outside the byte alphabet, 2: a pure one
    std::unordered_map<std::string_view, int32_t> by_bytes;   // raw bytes -> id, load only
    by_bytes.reserve(entries.size() * 2);
    std::vector<Literal> literals_all;
    std::string bytes;
    for (auto& [key, id] : entries) {
        const bool pure = unspell(key, bytes);
        const int ty = (size_t) id < types.size() ? types[(size_t) id] : -1;
        if (ty == 3 || ty == 4) {
            bytes = key;   // an added token keeps its literal text
            literals_all.push_back({key, (int32_t) id, ty == 3});
        }
        raw_[(size_t) id] = bytes;
        have[(size_t) id] = pure ? 2 : 1;
    }
    // a pure token wins a bytes collision with an impure one
    for (size_t id = 0; id < raw_.size(); ++id) {
        if (!have[id]) continue;
        auto [it, fresh] = by_bytes.try_emplace(std::string_view(raw_[id]), (int32_t) id);
        if (!fresh && have[id] == 2 && have[(size_t) it->second] != 2) it->second = (int32_t) id;
    }
    for (int b = 0; b < 256; ++b) {
        const char c = (char) b;
        auto it = by_bytes.find(std::string_view(&c, 1));
        if (it != by_bytes.end()) byte_id_[b] = it->second;
    }

    // merges.txt -> open-addressed table; a repeated pair keeps its last rank
    size_t n_merges = 0;
    for (char c : merges_txt) n_merges += c == '\n';
    n_merges += merges_txt.empty() ? 0 : 1;
    size_t cap = 1024;
    while (cap < n_merges * 2) cap <<= 1;
    merges_.assign(cap, Merge());
    merge_mask_ = cap - 1;
    std::string l, r;
    uint32_t rank = 0;
    for (size_t pos = 0; pos < merges_txt.size(); ++rank) {
        size_t nl = merges_txt.find('\n', pos);
        if (nl == std::string_view::npos) nl = merges_txt.size();
        std::string_view line = merges_txt.substr(pos, nl - pos);
        pos = nl + 1;
        const size_t sp = line.find(' ');
        if (sp == std::string_view::npos || line.find(' ', sp + 1) != std::string_view::npos) {
            err = "merges.txt: line " + std::to_string(rank) + " is not a pair";
            return false;
        }
        unspell(line.substr(0, sp), l);
        unspell(line.substr(sp + 1), r);
        auto li = by_bytes.find(l), ri = by_bytes.find(r);
        if (li == by_bytes.end() || ri == by_bytes.end()) {
            err = "merges.txt: merge " + std::to_string(rank) + " names a token outside the vocabulary";
            return false;
        }
        l += r;
        auto mi = by_bytes.find(l);
        if (mi == by_bytes.end()) {
            err = "merges.txt: merge " + std::to_string(rank) + " makes a token outside the vocabulary";
            return false;
        }
        const uint64_t key = pair_key(li->second, ri->second);
        size_t s = pair_slot(key, merge_mask_);
        while (merges_[s].key != ~0ull && merges_[s].key != key) s = (s + 1) & merge_mask_;
        merges_[s] = {key, rank, mi->second};
    }

    // special-token literals: type 4 always, type 3 with parse_special
    std::vector<Literal> four, three_four;
    for (auto& lit : literals_all) {
        three_four.push_back(lit);
        if (!lit.control) four.push_back(lit);
    }
    build_literals(always_, four);
    build_literals(all_, three_four);
    return true;
}

void Tokenizer::build_literals(LiteralSet& set, const std::vector<Literal>& all) {
    set = LiteralSet();
    set.lits = all;
    std::stable_sort(set.lits.begin(), set.lits.end(),
                     [](const Literal& a, const Literal& b) { return a.text.size() > b.text.size(); });
    for (uint32_t i = 0; i < set.lits.size(); ++i) {
        if (set.lits[i].text.empty()) continue;
        const uint8_t f = (uint8_t) set.lits[i].text[0];
        if (!set.first[f]) { set.first[f] = true; ++set.distinct_first; set.single_first = f; }
        set.by_first[f].push_back(i);
    }
    if (set.distinct_first != 1) set.single_first = -1;
}

int32_t Tokenizer::find_merge(int32_t left, int32_t right, uint32_t& rank) const {
    const uint64_t key = pair_key(left, right);
    size_t s = pair_slot(key, merge_mask_);
    for (;;) {
        const Merge& m = merges_[s];
        if (m.key == key) { rank = m.rank; return m.result; }
        if (m.key == ~0ull) return -1;
        s = (s + 1) & merge_mask_;
    }
}

bool Tokenizer::merge_piece(const uint8_t* p, size_t n, std::vector<int32_t>& out, std::string& err) {
    sym_.resize(n);
    for (size_t i = 0; i < n; ++i) {
        const int32_t id = byte_id_[p[i]];
        if (id < 0) { err = "a byte has no token in the vocabulary"; return false; }
        sym_[i] = id;
    }
    size_t count = n;
    if (count == 1) { out.push_back(sym_[0]); return true; }
    if (count <= kScanMaxSymbols) {
        rank_.resize(count);
        const uint32_t kNone = 0xFFFFFFFFu;
        for (size_t i = 0; i + 1 < count; ++i) {
            uint32_t r;
            rank_[i] = find_merge(sym_[i], sym_[i + 1], r) >= 0 ? r : kNone;
        }
        for (;;) {
            size_t best = 0;
            uint32_t best_rank = kNone;
            for (size_t i = 0; i + 1 < count; ++i)
                if (rank_[i] < best_rank) { best_rank = rank_[i]; best = i; }
            if (best_rank == kNone) break;
            uint32_t r;
            sym_[best] = find_merge(sym_[best], sym_[best + 1], r);
            std::memmove(&sym_[best + 1], &sym_[best + 2], (count - best - 2) * sizeof(int32_t));
            std::memmove(&rank_[best], &rank_[best + 1], (count - best - 1) * sizeof(uint32_t));
            --count;
            if (best + 1 < count) rank_[best] = find_merge(sym_[best], sym_[best + 1], r) >= 0 ? r : kNone;
            else rank_[best] = kNone;
            if (best > 0) rank_[best - 1] = find_merge(sym_[best - 1], sym_[best], r) >= 0 ? r : kNone;
        }
        out.insert(out.end(), sym_.begin(), sym_.begin() + (std::ptrdiff_t) count);
        return true;
    }

    // a long piece: a linked list of symbols and a min-heap of (rank, position); a stale entry is skipped
    struct Item { uint32_t rank; uint32_t pos; int32_t left, right; };
    auto greater = [](const Item& a, const Item& b) { return a.rank != b.rank ? a.rank > b.rank : a.pos > b.pos; };
    std::vector<int32_t> next(n), prev(n);
    for (size_t i = 0; i < n; ++i) { next[i] = (int32_t) i + 1 < (int32_t) n ? (int32_t) i + 1 : -1; prev[i] = (int32_t) i - 1; }
    std::vector<Item> heap;
    heap.reserve(n * 2);
    for (size_t i = 0; i + 1 < n; ++i) {
        uint32_t r;
        if (find_merge(sym_[i], sym_[i + 1], r) >= 0) heap.push_back({r, (uint32_t) i, sym_[i], sym_[i + 1]});
    }
    std::make_heap(heap.begin(), heap.end(), greater);
    std::vector<uint8_t> dead(n, 0);
    while (!heap.empty()) {
        std::pop_heap(heap.begin(), heap.end(), greater);
        const Item it = heap.back();
        heap.pop_back();
        const int32_t i = (int32_t) it.pos, j = next[(size_t) i];
        if (dead[(size_t) i] || sym_[(size_t) i] != it.left || j < 0 || sym_[(size_t) j] != it.right) continue;
        uint32_t r;
        sym_[(size_t) i] = find_merge(it.left, it.right, r);
        dead[(size_t) j] = 1;
        const int32_t k = next[(size_t) j];
        next[(size_t) i] = k;
        if (k >= 0) {
            prev[(size_t) k] = i;
            if (find_merge(sym_[(size_t) i], sym_[(size_t) k], r) >= 0) { heap.push_back({r, (uint32_t) i, sym_[(size_t) i], sym_[(size_t) k]}); std::push_heap(heap.begin(), heap.end(), greater); }
        }
        const int32_t pv = prev[(size_t) i];
        if (pv >= 0 && find_merge(sym_[(size_t) pv], sym_[(size_t) i], r) >= 0) { heap.push_back({r, (uint32_t) pv, sym_[(size_t) pv], sym_[(size_t) i]}); std::push_heap(heap.begin(), heap.end(), greater); }
    }
    for (size_t i = 0; i < n; ++i)
        if (!dead[i]) out.push_back(sym_[i]);
    return true;
}

void Tokenizer::memo_grow() {
    const size_t cap = memo_.empty() ? 4096 : memo_.size() * 2;
    std::vector<MemoEntry> bigger(cap);
    for (const MemoEntry& e : memo_) {
        if (e.key_len == 0) continue;
        size_t s = (size_t) e.hash & (cap - 1);
        while (bigger[s].key_len != 0) s = (s + 1) & (cap - 1);
        bigger[s] = e;
    }
    memo_.swap(bigger);
}

bool Tokenizer::encode_plain(std::string_view text, std::vector<int32_t>& out, std::string& err) {
    tokenizer::pretok_qwen35(text.data(), text.size(), ends_);
    size_t begin = 0;
    for (const size_t end : ends_) {
        const char* p = text.data() + begin;
        const size_t n = end - begin;
        begin = end;
        const bool memoable = n <= kMemoMaxPiece;
        uint64_t h = 0;
        size_t slot = 0;
        if (memoable) {
            if (memo_.empty()) memo_grow();
            h = hash_bytes(p, n);
            slot = (size_t) h & (memo_.size() - 1);
            bool hit = false;
            while (memo_[slot].key_len != 0) {
                const MemoEntry& e = memo_[slot];
                if (e.hash == h && e.key_len == n && std::memcmp(memo_keys_.data() + e.key_off, p, n) == 0) {
                    out.insert(out.end(), memo_ids_.begin() + e.ids_off, memo_ids_.begin() + e.ids_off + e.ids_len);
                    hit = true;
                    break;
                }
                slot = (slot + 1) & (memo_.size() - 1);
            }
            if (hit) continue;
        }
        const size_t before = out.size();
        if (!merge_piece((const uint8_t*) p, n, out, err)) return false;
        if (memoable && memo_used_ < kMemoMaxEntries) {
            if ((memo_used_ + 1) * 2 > memo_.size()) {
                memo_grow();
                slot = (size_t) h & (memo_.size() - 1);
                while (memo_[slot].key_len != 0) slot = (slot + 1) & (memo_.size() - 1);
            }
            memo_[slot] = {h, (uint32_t) memo_keys_.size(), (uint32_t) n, (uint32_t) memo_ids_.size(), (uint32_t) (out.size() - before)};
            memo_keys_.append(p, n);
            memo_ids_.insert(memo_ids_.end(), out.begin() + (std::ptrdiff_t) before, out.end());
            ++memo_used_;
        }
    }
    return true;
}

bool Tokenizer::encode(std::string_view text, bool parse_special, const std::vector<Span>& plain,
                       std::vector<int32_t>& out, std::string& err) {
    out.clear();
    err.clear();
    if (!loaded()) { err = "the tokenizer is not loaded"; return false; }
    out.reserve(text.size() / 3 + 16);
    const LiteralSet& set = parse_special ? all_ : always_;
    size_t pos = 0, scan = 0;
    const size_t n = text.size();
    while (scan < n && !set.lits.empty()) {
        // the next position at which a literal starts
        size_t at = scan;
        const Literal* hit = nullptr;
        while (at < n) {
            if (set.single_first >= 0) {
                const void* f = std::memchr(text.data() + at, set.single_first, n - at);
                if (f == nullptr) { at = n; break; }
                at = (size_t) ((const char*) f - text.data());
            } else {
                while (at < n && !set.first[(uint8_t) text[at]]) ++at;
                if (at >= n) break;
            }
            for (uint32_t idx : set.by_first[(uint8_t) text[at]]) {
                const Literal& lit = set.lits[idx];
                if (lit.text.size() <= n - at && std::memcmp(text.data() + at, lit.text.data(), lit.text.size()) == 0) {
                    hit = &lit;
                    break;
                }
            }
            if (hit != nullptr) break;
            ++at;
        }
        if (hit == nullptr) break;
        const size_t end = at + hit->text.size();
        bool inside = false;
        for (const Span& s : plain) inside = inside || (s.begin <= at && at < s.end);
        scan = end;
        if (inside) continue;
        if (at > pos && !encode_plain(text.substr(pos, at - pos), out, err)) return false;
        out.push_back(hit->id);
        pos = end;
    }
    if (pos < n && !encode_plain(text.substr(pos), out, err)) return false;
    return true;
}

}  // namespace basalt::program
