// tests/program/tokenizer_test.cpp - the C++ tokenizer: a synthetic vocabulary (always), and, given a pack's tokenizer
// directory and a golden file made by tokenizer_parity.py, exact ids against the reference tokenizer.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>

#include "basalt/program/tokenizer.hpp"

using basalt::program::Tokenizer;

namespace {

int failures = 0;

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++failures; } } while (0)

std::string read_all(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

// the byte alphabet spelling of a raw string, UTF-8
std::string spell(const std::string& raw) {
    std::string out;
    uint32_t next = 256;
    uint32_t map[256];
    for (int b = 0; b < 256; ++b) {
        const bool plain = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        map[b] = plain ? (uint32_t) b : next++;
    }
    for (unsigned char c : raw) {
        const uint32_t cp = map[c];
        if (cp < 0x80) out += (char) cp;
        else { out += (char) (0xC0 | (cp >> 6)); out += (char) (0x80 | (cp & 0x3F)); }
    }
    return out;
}

void synthetic() {
    // vocabulary: the 256 byte tokens, then merges a+b, ab+c, b+c, aa, then specials
    std::string vocab = "{", merges, types = "[";
    std::vector<std::string> toks;
    for (int b = 0; b < 256; ++b) toks.push_back(spell(std::string(1, (char) b)));
    const char* extra[] = {"ab", "abc", "bc", "aa", "aaa"};
    for (const char* e : extra) toks.push_back(e);
    toks.push_back("<|ctl|>");
    toks.push_back("<usr>");
    for (size_t i = 0; i < toks.size(); ++i) {
        std::string key;
        for (char c : toks[i]) { if (c == '"' || c == '\\') key += '\\'; key += c; }
        vocab += (i ? "," : "") + std::string("\"") + key + "\":" + std::to_string(i);
        types += (i ? "," : "") + std::string(i == 261 ? "3" : i == 262 ? "4" : "1");
    }
    vocab += "}";
    types += "]";
    merges = "b c\na b\nab c\na a\naa a";   // ranks: bc 0, ab 1, abc 2, aa 3, aaa 4
    Tokenizer t;
    std::string err;
    CHECK(t.load(vocab, merges, types, err));
    std::vector<int32_t> ids;
    CHECK(t.encode("abc", false, {}, ids, err));
    // lowest rank first: b+c (0) beats a+b (1), so a, bc
    CHECK((ids == std::vector<int32_t>{'a', 258}));
    CHECK(t.encode("aaa", false, {}, ids, err));
    CHECK((ids == std::vector<int32_t>{260}));
    CHECK(t.encode("aaaa", false, {}, ids, err));
    CHECK((ids == std::vector<int32_t>{259, 259}));
    // a long run takes the heap path and must equal the rescanning path piece by piece
    std::string longw(200, 'a');
    CHECK(t.encode(longw, false, {}, ids, err));
    size_t total = 0;
    for (int32_t id : ids) total += t.token_bytes(id).size();
    CHECK(total == 200);
    CHECK(t.encode("x<|ctl|>y<usr>", false, {}, ids, err));
    CHECK(ids.back() == 262 && std::find(ids.begin(), ids.end(), 261) == ids.end());
    CHECK(t.encode("x<|ctl|>y<usr>", true, {}, ids, err));
    CHECK((ids == std::vector<int32_t>{'x', 261, 'y', 262}));
    CHECK(t.encode("x<|ctl|>y", true, {{1, 8}}, ids, err));
    CHECK(std::find(ids.begin(), ids.end(), 261) == ids.end());
    CHECK(t.encode("", true, {}, ids, err) && ids.empty());
    // the same piece twice: the memo returns the same ids
    CHECK(t.encode("abcabc abcabc", false, {}, ids, err));
    CHECK(t.encode("abcabc abcabc", false, {}, ids, err));
    std::string bad;
    CHECK(!t.load("{\"a\":1", "", "[]", bad));
}

int parity(const std::string& dir, const std::string& golden) {
    Tokenizer t;
    std::string err;
    const auto t0 = std::chrono::steady_clock::now();
    if (!t.load(read_all(dir + "/vocab.json"), read_all(dir + "/merges.txt"), read_all(dir + "/token_type.json"), err)) {
        std::fprintf(stderr, "load: %s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr, "load %.0f ms, %zu tokens\n",
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), t.vocab_size());
    const std::string g = read_all(golden);
    size_t pos = 0, cases = 0, bad = 0;
    while (pos < g.size()) {
        const size_t nl = g.find('\n', pos);
        char name[512];
        int special = 0, nplain = 0;
        unsigned long nbytes = 0, nids = 0;
        if (std::sscanf(g.c_str() + pos, "CASE %511s %d %lu %lu %d", name, &special, &nbytes, &nids, &nplain) != 5) {
            std::fprintf(stderr, "golden: bad header at %zu\n", pos);
            return 1;
        }
        pos = nl + 1;
        const std::string text = g.substr(pos, nbytes);
        pos += nbytes + 1;
        const size_t sl = g.find('\n', pos);
        std::vector<Tokenizer::Span> plain;
        {
            std::string spans = g.substr(pos, sl - pos);
            size_t p = 0;
            while (p < spans.size()) {
                size_t a = 0, b = 0;
                int used = 0;
                if (std::sscanf(spans.c_str() + p, "%zu:%zu%n", &a, &b, &used) != 2) break;
                plain.push_back({a, b});
                p += (size_t) used;
                if (p < spans.size() && spans[p] == ',') ++p;
            }
        }
        pos = sl + 1;
        const size_t il = g.find('\n', pos);
        std::vector<int32_t> want;
        {
            const char* p = g.c_str() + pos;
            const char* end = g.c_str() + il;
            while (p < end) {
                char* e = nullptr;
                want.push_back((int32_t) std::strtol(p, &e, 10));
                p = e < end && *e == ',' ? e + 1 : e;
                if (e == p && p < end) break;
            }
        }
        pos = il + 1;
        std::vector<int32_t> got;
        ++cases;
        const bool ok = t.encode(text, special != 0, plain, got, err);
        std::string back;
        for (int32_t id : got) back += t.token_bytes(id);
        if (!ok || got != want || back != text) {
            ++bad;
            size_t d = 0;
            while (d < got.size() && d < want.size() && got[d] == want[d]) ++d;
            std::fprintf(stderr, "MISMATCH %s: got %zu ids, want %zu, first difference at %zu%s%s\n", name, got.size(),
                         want.size(), d, ok ? "" : ", error: ", ok ? "" : err.c_str());
            if (back != text) std::fprintf(stderr, "  round trip differs\n");
        }
    }
    std::fprintf(stderr, "%zu cases, %zu mismatches\n", cases, bad);
    return bad == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    synthetic();
    if (failures) return 1;
    if (argc >= 3) return parity(argv[1], argv[2]);
    std::printf("tokenizer_test: synthetic checks passed\n");
    return 0;
}
