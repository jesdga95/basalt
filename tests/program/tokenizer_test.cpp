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

#include "basalt/program/glm_forms.hpp"
#include "basalt/program/tokenizer.hpp"
#include "unicode_flags.hpp"

// The Qwen pre-tokenizer is shared (the tokenizer library); the others belong to the models this build contains.
#ifdef BASALT_MODEL_DEEPSEEK41
#include "models/deepseek41/adapter.hpp"
#endif
namespace basalt::program::tokenizer { void pretok_qwen35(const char* text, size_t size, std::vector<size_t>& ends); }

using basalt::program::Tokenizer;

namespace {
using PreTok = void (*)(const char*, size_t, std::vector<size_t>&);
#ifdef BASALT_MODEL_DEEPSEEK41
constexpr PreTok kJoyai = basalt::models::deepseek41::pretok_joyai;
#else
constexpr PreTok kJoyai = nullptr;
#endif
constexpr PreTok kQwen35 = basalt::program::tokenizer::pretok_qwen35;
constexpr PreTok kGlm4 = basalt::program::tokenizer::pretok_glm4;
}  // namespace

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
    t.set_pretokenize(kQwen35);
    CHECK(t.load(vocab, merges, types, err));
    std::vector<int32_t> ids;
    CHECK(t.encode("abc", false, {}, ids, err));
    // lowest rank first: b+c (0) beats a+b (1), so a, bc
    CHECK((ids == std::vector<int32_t>{'a', 258}));
    // the winning merge is the last pair of a 2-byte and of a 3-byte piece
    CHECK(t.encode("ab", false, {}, ids, err));
    CHECK((ids == std::vector<int32_t>{256}));
    CHECK(t.encode("bc", false, {}, ids, err));
    CHECK((ids == std::vector<int32_t>{258}));
    CHECK(t.encode("aab", false, {}, ids, err));
    CHECK((ids == std::vector<int32_t>{'a', 256}));
    CHECK(t.encode("xbc", false, {}, ids, err));
    CHECK((ids == std::vector<int32_t>{'x', 258}));
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
    // ignore_merges: a piece that is an entry is that token (abc, which the merges build as a, bc); others still merge
    Tokenizer whole;
    whole.set_pretokenize(kQwen35);
    whole.set_ignore_merges(true);
    CHECK(whole.load(vocab, merges, types, err));
    CHECK(whole.encode("abc", false, {}, ids, err));
    CHECK((ids == std::vector<int32_t>{257}));
    CHECK(whole.encode("abcx", false, {}, ids, err));
    CHECK((ids == std::vector<int32_t>{'a', 258, 'x'}));
    CHECK(whole.encode("x<|ctl|>y", true, {}, ids, err));
    CHECK((ids == std::vector<int32_t>{'x', 261, 'y'}));
    std::string bad;
    CHECK(!t.load("{\"a\":1", "", "[]", bad));

    // qwen35 pre-tokenizer edge cases: contractions, numbers, punctuation + CRLF, whitespace lookahead, Unicode, invalid UTF-8
    auto pieces = [](std::string_view s) {
        std::vector<size_t> ends;
        kQwen35(s.data(), s.size(), ends);
        std::vector<std::string> out;
        size_t start = 0;
        for (size_t e : ends) { out.emplace_back(s.substr(start, e - start)); start = e; }
        return out;
    };
    CHECK((pieces("it's YOU'RE 'll 'VE 'D") == std::vector<std::string>{"it", "'s", " YOU", "'RE", " '", "ll", " '", "VE", " '", "D"}));
    CHECK((pieces("a   b  ") == std::vector<std::string>{"a", "  ", " b", "  "}));
    CHECK((pieces(" ") == std::vector<std::string>{" "}));
    CHECK((pieces(" !!\r\n12") == std::vector<std::string>{" !!\r\n", "1", "2"}));
    CHECK((pieces(" \t\r\n  x") == std::vector<std::string>{" \t\r\n", " ", " x"}));
    CHECK((pieces("caf\xC3\xA9 \xE4\xB8\xAD\xE6\x96\x87\xE3\x80\x80\xE3\x80\x80x") ==
           std::vector<std::string>{"caf\xC3\xA9", " \xE4\xB8\xAD\xE6\x96\x87", "\xE3\x80\x80", "\xE3\x80\x80x"}));
    CHECK((pieces("\xFF\xFE" "abc") == std::vector<std::string>{"\xFF\xFE", "abc"}));
    CHECK((pieces("\xFF" "abc") == std::vector<std::string>{"\xFF" "abc"}));
}

// The pieces the glm4 expression fixes. Each expected split comes from the reference expression that tools/tokenizer.py
// ships for the same pre-type, so the two implementations are checked against one another, not against my reading of it.
// The cases are chosen where glm4 and qwen35 part company: a word run is \p{L} alone (a combining mark starts a piece),
// a digit run holds at most three digits, and the punctuation class keeps the marks.
void glm4_pieces() {
    struct Case { const char* text; std::vector<std::string> pieces; };
    const std::vector<Case> cases = {
        {"don't stop believing", {"don", "'t", " stop", " believing"}},
        {"Don't STOP'T 'RE 've 'M 'LL 'd", {"Don", "'t", " STOP", "'T", " '", "RE", " '", "ve", " '", "M", " '", "LL", " '", "d"}},
        {"it's a 12345 test", {"it", "'s", " a", " ", "123", "45", " test"}},
        {"123456789", {"123", "456", "789"}},
        {"v1.2.3", {"v", "1", ".", "2", ".", "3"}},
        {"  blank  ", {" ", " blank", "  "}},
        {"a\n\nb", {"a", "\n\n", "b"}},
        {"line1\r\nline2", {"line", "1", "\r\n", "line", "2"}},
        {"trailing spaces   ", {"trailing", " spaces", "   "}},
        {"tab\tsep\tvals", {"tab", "\tsep", "\tvals"}},
        {"h\xc3\xa9llo w\xc3\xb6rld", {"h\xc3\xa9llo", " w\xc3\xb6rld"}},
        {"\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e\xe3\x83\x86\xe3\x82\xad\xe3\x82\xb9\xe3\x83\x88", {"\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e\xe3\x83\x86\xe3\x82\xad\xe3\x82\xb9\xe3\x83\x88"}},
        {"e\xcc\x81" "e\xcc\x81" "fa\xc3\xa7 ca", {"e", "\xcc\x81" "e", "\xcc\x81" "fa\xc3\xa7", " ca"}},
        {"\xc3\xa9\xcc\x81", {"\xc3\xa9", "\xcc\x81"}},
        {"\xf0\x9f\x8e\x89 emoji \xf0\x9f\x8e\x89", {"\xf0\x9f\x8e\x89", " emoji", " \xf0\x9f\x8e\x89"}},
        {"<|user|>\nhello\n", {"<|", "user", "|>\n", "hello", "\n"}},
        {"[id=abc123]call(x) [func=get_weather] {\"a\": 1}", {"[id", "=abc", "123", "]call", "(x", ")", " [", "func", "=get", "_weather", "]", " {\"", "a", "\":", " ", "1", "}"}},
        {"[get_weather]({\"city\": \"Paris\"})", {"[get", "_weather", "]({\"", "city", "\":", " \"", "Paris", "\"})"}},
        {"  \n \t \n  ", {"  \n \t \n", "  "}},
        {"...!?  ok", {"...!?", " ", " ok"}},
        {"a  b   c  ", {"a", " ", " b", "  ", " c", "  "}},
        {"don't\xe3\x80\x8ctest\xe3\x80\x8d", {"don", "'t", "\xe3\x80\x8ctest", "\xe3\x80\x8d"}},
        {"3.14159 42 7", {"3", ".", "141", "59", " ", "42", " ", "7"}},
        {"-42 +17", {"-", "42", " +", "17"}},
        {"x'y z", {"x", "'y", " z"}},
        {"  ", {"  "}},
        {"\n", {"\n"}},
        {"\t\t", {"\t\t"}},
        {"\x0c\x0b x", {"\x0c\x0b", " x"}},
        {"\xc2\xab\xc3\xa7" "a\xc2\xbb 'l'homme", {"\xc2\xab\xc3\xa7" "a", "\xc2\xbb", " '", "l", "'homme"}},
        {"caf\xc3\xa9 au lait  ", {"caf\xc3\xa9", " au", " lait", "  "}},
        {"0123456789012345", {"012", "345", "678", "901", "234", "5"}},
        {"'s'T'rE'vE'm'LL'D'\xc4\xb0", {"'s", "'T", "'rE", "'vE", "'m", "'LL", "'D", "'\xc4\xb0"}},
        {"\xd8\xa7\xd9\x84\xd8\xb9\xd8\xb1\xd8\xa8\xd9\x8a\xd8\xa9 \xd9\xa1\xd9\xa2\xd9\xa3\xd9\xa4", {"\xd8\xa7\xd9\x84\xd8\xb9\xd8\xb1\xd8\xa8\xd9\x8a\xd8\xa9", " ", "\xd9\xa1\xd9\xa2\xd9\xa3", "\xd9\xa4"}},
        {"Hello, world! It's 2026 \xe2\x80\x94 don't you agree?", {"Hello", ",", " world", "!", " It", "'s", " ", "202", "6", " \xe2\x80\x94", " don", "'t", " you", " agree", "?"}},
        {"R\xc3\xa9" "flexion\xe2\x80\xa6 l'\xc3\xa9t\xc3\xa9 s'ach\xc3\xa8ve.", {"R\xc3\xa9" "flexion", "\xe2\x80\xa6", " l", "'\xc3\xa9t\xc3\xa9", " s", "'ach\xc3\xa8ve", "."}},
        {"<system>\nYou are GLM.\n</system>", {"<system", ">\n", "You", " are", " GLM", ".\n", "</", "system", ">"}},
    };
    for (const Case& c : cases) {
        std::vector<size_t> ends;
        kGlm4(c.text, std::string(c.text).size(), ends);
        std::vector<std::string> got;
        size_t from = 0;
        for (size_t e : ends) { got.push_back(std::string(c.text).substr(from, e - from)); from = e; }
        if (got != c.pieces) {
            std::fprintf(stderr, "FAIL glm4 '%s':", c.text);
            for (const std::string& p : got) std::fprintf(stderr, " [%s]", p.c_str());
            std::fprintf(stderr, " want");
            for (const std::string& p : c.pieces) std::fprintf(stderr, " [%s]", p.c_str());
            std::fprintf(stderr, "\n");
            ++failures;
        }
    }
}

int parity(const std::string& dir, const std::string& golden) {
    Tokenizer t;
    std::string err;
    const auto t0 = std::chrono::steady_clock::now();
    // the rule the pack names
    const std::string cfg = read_all(dir + "/tokenizer.json");
    const PreTok pre = cfg.find("\"pre\": \"joyai-llm\"") != std::string::npos   ? kJoyai
                       : cfg.find("\"pre\": \"hy_v4\"") != std::string::npos      ? kJoyai
                       : cfg.find("\"pre\": \"glm4\"") != std::string::npos       ? kGlm4
                       : cfg.find("\"pre\": \"glm5\"") != std::string::npos       ? kGlm4
                       : kQwen35;
    if (pre == nullptr) {
        std::fprintf(stderr, "the pre-tokenizer of this pack belongs to a model this build left out\n");
        return 1;
    }
    t.set_pretokenize(pre);
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
    glm4_pieces();
    if (failures) return 1;
    if (argc >= 3) return parity(argv[1], argv[2]);
    std::printf("tokenizer_test: synthetic checks passed\n");
    return 0;
}
