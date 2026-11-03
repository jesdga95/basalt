// src/spec/suffix_drafter.cpp - see include/basalt/spec/suffix_drafter.hpp.
#include "basalt/spec/suffix_drafter.hpp"

#include <algorithm>

namespace basalt::spec {

namespace {
uint64_t mix(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    return x ^ (x >> 33);
}
}  // namespace

SuffixDrafter::SuffixDrafter(int min_match, int max_match, size_t capacity_tokens)
    : min_match_(std::max(3, min_match)), max_match_(std::max(min_match, max_match)) {
    size_t cap = 1;
    while (cap < capacity_tokens * 2) cap <<= 1;          // load factor <= 0.5 at the nominal capacity
    table_.assign(cap, Slot{});
    long_.assign(cap, Slot{});
    mask_ = cap - 1;
    hist_.reserve(capacity_tokens);
}

void SuffixDrafter::reset() {
    hist_.clear();
    for (const uint32_t d : dirty_) (d & kLongBit ? long_ : table_)[d & ~kLongBit] = Slot{};   // only what was filled
    dirty_.clear();
    last_match_ = 0;
}

uint64_t SuffixDrafter::key_at(size_t end) const {
    const uint64_t a = (uint32_t) hist_[end - 2], b = (uint32_t) hist_[end - 1], c = (uint32_t) hist_[end];
    return mix(a * 0x9E3779B97F4A7C15ull ^ mix(b + 0x632BE59BD9B4E019ull) ^ (c << 1)) | 1ull;   // never 0
}

uint64_t SuffixDrafter::long_key_at(size_t end) const {
    uint64_t h = 0x9E3779B97F4A7C15ull;
    for (int i = 0; i < LONG_ORDER; ++i) h = mix(h ^ (uint32_t) hist_[end - (size_t) i]) + (uint64_t) i;
    return h | 1ull;   // never 0
}

SuffixDrafter::Slot* SuffixDrafter::find_slot(std::vector<Slot>& table, uint64_t key, bool insert) {
    for (size_t i = key & mask_, probes = 0; probes <= mask_; i = (i + 1) & mask_, ++probes) {
        Slot& s = table[i];
        if (s.key == key) return &s;
        if (s.key == 0) {
            if (!insert) return nullptr;
            s.key = key;
            dirty_.push_back((uint32_t) i | (&table == &long_ ? kLongBit : 0u));
            return &s;
        }
    }
    return nullptr;                                         // table full: the history outgrew its capacity
}

void SuffixDrafter::append(const int32_t* tokens, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        hist_.push_back(tokens[i]);
        const size_t end = hist_.size() - 1;
        if (end < 2) continue;
        push(find_slot(table_, key_at(end), true), end);
        if (end >= (size_t) LONG_ORDER - 1) push(find_slot(long_, long_key_at(end), true), end);
    }
}

void SuffixDrafter::push(Slot* s, size_t end) {
    if (s == nullptr) return;
    for (int w = WAYS - 1; w > 0; --w) s->pos[w] = s->pos[w - 1];
    s->pos[0] = (uint32_t) end;
    if (s->n < WAYS) ++s->n;
}

int SuffixDrafter::propose(int max_k, int32_t* out) {
    last_match_ = 0;
    const size_t n = hist_.size();
    if (n < 4 || max_k <= 0) return 0;
    const size_t cur = n - 1;
    // The trigram slot holds only the newest repeats of a common trigram; the long slot keeps the repeats of the last
    // LONG_ORDER tokens, so an older source of a quoted stretch survives there. Longest match wins, the long slot's first.
    const Slot* slots[2] = {n >= (size_t) LONG_ORDER ? find_slot(long_, long_key_at(cur), false) : nullptr,
                            find_slot(table_, key_at(cur), false)};
    size_t best_end = 0;
    int best_len = 0;
    for (const Slot* s : slots) {
        if (s == nullptr) continue;
        for (int w = 0; w < s->n; ++w) {
            const size_t p = s->pos[w];
            if (p >= cur) continue;                         // the current suffix itself
            int len = 0;
            while (len < max_match_ && len <= (int) p && hist_[p - len] == hist_[cur - len]) ++len;
            if (len > best_len) { best_len = len; best_end = p; }   // most recent first, so ties keep the newer
        }
    }
    if (best_len < min_match_) return 0;
    last_match_ = best_len;
    int k = 0;
    // The continuation may run into the current suffix (periodic text); reading history up to `cur` is valid.
    for (size_t q = best_end + 1; q <= cur && k < max_k; ++q) out[k++] = hist_[q];
    return k;
}

const SuffixDrafter::Slot* SuffixDrafter::find_slot_ro(const std::vector<Slot>& table, uint64_t key) const {
    for (size_t i = key & mask_, probes = 0; probes <= mask_; i = (i + 1) & mask_, ++probes) {
        const Slot& s = table[i];
        if (s.key == key) return &s;
        if (s.key == 0) return nullptr;
    }
    return nullptr;
}

int SuffixDrafter::propose_after(const int32_t* extra, size_t extra_n, int max_k, int32_t* out, int* match_out) const {
    if (match_out) *match_out = 0;
    const size_t base_n = hist_.size(), tot = base_n + extra_n;
    if (tot < 4 || max_k <= 0) return 0;
    const size_t cur = tot - 1;
    const auto at = [&](size_t idx) -> int32_t {
        return idx < base_n ? hist_[idx] : extra[idx - base_n];
    };
    const uint64_t tri_key = [&] {
        const uint64_t a = (uint32_t) at(cur - 2), b = (uint32_t) at(cur - 1), c = (uint32_t) at(cur);
        return mix(a * 0x9E3779B97F4A7C15ull ^ mix(b + 0x632BE59BD9B4E019ull) ^ (c << 1)) | 1ull;
    }();
    uint64_t lkey = 0;
    if (tot >= (size_t) LONG_ORDER) {
        uint64_t h = 0x9E3779B97F4A7C15ull;
        for (int i = 0; i < LONG_ORDER; ++i) h = mix(h ^ (uint32_t) at(cur - (size_t) i)) + (uint64_t) i;
        lkey = h | 1ull;
    }
    const Slot* slots[2] = {tot >= (size_t) LONG_ORDER ? find_slot_ro(long_, lkey) : nullptr,
                            find_slot_ro(table_, tri_key)};
    size_t best_end = 0;
    int best_len = 0;
    for (size_t p = cur > 0 ? cur - 1 : 0; p >= base_n && p < cur; --p) {
        int len = 0;
        while (len < max_match_ && len <= (int) p && at(p - (size_t) len) == at(cur - (size_t) len)) ++len;
        if (len > best_len) { best_len = len; best_end = p; }
        if (p == 0) break;
    }
    for (const Slot* s : slots) {
        if (s == nullptr) continue;
        for (int w = 0; w < s->n; ++w) {
            const size_t p = s->pos[w];
            if (p >= cur) continue;
            int len = 0;
            while (len < max_match_ && len <= (int) p && at(p - (size_t) len) == at(cur - (size_t) len)) ++len;
            if (len > best_len) { best_len = len; best_end = p; }
        }
    }
    if (best_len < min_match_) return 0;
    if (match_out) *match_out = best_len;
    int k = 0;
    for (size_t q = best_end + 1; q <= cur && k < max_k; ++q) out[k++] = at(q);
    return k;
}

}  // namespace basalt::spec
