// tests/program/grammar_test.cpp - GBNF parse, accept/reject, UTF-8 across tokens, clone/rewind, lazy triggers, and
// the vocabulary mask against a token-by-token check. Ends with a timing line for allows() and fill_mask().
#include "tool_form_fixture.hpp"
#include "basalt/program/grammar/grammar.hpp"
#include "basalt/program/grammar/schema.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

using namespace basalt::program::grammar;

static int g_fail = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

static std::shared_ptr<const Grammar> parse(const std::string& g) {
    std::string err;
    auto p = Grammar::parse(g, err);
    if (!p) std::printf("parse error: %s\n", err.c_str());
    return p;
}

// A complete sentence of the grammar, fed byte by byte.
static bool full(const std::shared_ptr<const Grammar>& g, const std::string& text) {
    GrammarState s(g);
    for (char c : text) if (!s.accept(std::string_view(&c, 1))) return false;
    return s.accepting();
}
// Every byte is accepted (a prefix of some sentence).
static bool prefix(const std::shared_ptr<const Grammar>& g, const std::string& text) {
    GrammarState s(g);
    for (char c : text) if (!s.accept(std::string_view(&c, 1))) return false;
    return true;
}

static const char* kJson = R"(
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws
object ::= "{" ws ( string ":" ws value ("," ws string ":" ws value)* )? "}" ws
array  ::= "[" ws ( value ("," ws value)* )? "]" ws
string ::= "\"" ( [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}) )* "\"" ws
number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws
ws     ::= | " " | "\n" [ \t]{0,20}
)";

static void test_parse() {
    std::string err;
    CHECK(!Grammar::parse("root ::= a", err) && err.find("undefined") != std::string::npos);
    CHECK(!Grammar::parse("x ::= \"a\"", err) && err.find("root") != std::string::npos);
    CHECK(!Grammar::parse("root ::= root \"a\" | \"b\"", err) && err.find("left recursion") != std::string::npos);
    CHECK(!Grammar::parse("root ::= \"a", err));
    CHECK(!Grammar::parse("root ::= [a-", err));
    CHECK(!Grammar::parse("root ::= *", err));
    CHECK(!Grammar::parse("root ::= \"a\"{3", err));
    CHECK(Grammar::parse("# comment\nroot ::= \"a\" # tail\n", err));
}

static void test_basic() {
    auto g = parse("root ::= \"ab\" [0-9]+ (\"x\" | \"yz\")? \"!\"");
    CHECK(g);
    CHECK(full(g, "ab1!"));
    CHECK(full(g, "ab123x!"));
    CHECK(full(g, "ab9yz!"));
    CHECK(!full(g, "ab!"));
    CHECK(!prefix(g, "ac"));
    CHECK(prefix(g, "ab12y") && !full(g, "ab12y"));
    auto r = parse("root ::= \"a\"{2,3} \"b\"{2}");
    CHECK(full(r, "aabb") && full(r, "aaabb") && !full(r, "abb") && !prefix(r, "aaaa") && !full(r, "aab"));
    auto n = parse("root ::= [^a-c]+ .");
    CHECK(full(n, "xyzq") && !prefix(n, "a") && full(n, "dd\xC3\xA9"));
    auto esc = parse("root ::= \"\\x41\\u00e9\\n\"");
    CHECK(full(esc, "A\xC3\xA9\n"));
}

static void test_json() {
    auto g = parse(kJson);
    CHECK(g);
    CHECK(full(g, "{}"));
    CHECK(full(g, "{\"a\": 1, \"b\": [1, 2.5, -3e+2, true, null], \"c\": {\"d\": \"x\"}}"));
    CHECK(full(g, "{\"s\": \"a\\\"b\\n\\u00e9\"}"));
    CHECK(full(g, "{\"u\": \"\xC3\xA9\xE4\xB8\xAD\xF0\x9F\x98\x80\"}"));
    CHECK(full(g, "[[[[]]], [], {\"k\": []}]") == false);   // root is an object
    CHECK(full(g, "{\"k\": [[[[]]], [], {\"k\": []}]}"));
    CHECK(!prefix(g, "{\"a\": 01"));
    CHECK(!prefix(g, "{\"a\": True"));
    CHECK(!prefix(g, "{\"a\": \"x\ny\"}"));
    CHECK(!prefix(g, "{'a': 1}"));
    CHECK(prefix(g, "{\"a\": 1.") && !full(g, "{\"a\": 1."));
    CHECK(!full(g, "{\"a\": 1,}"));
}

static void test_utf8_tokens() {
    auto g = parse(kJson);
    GrammarState s(g);
    CHECK(s.accept("{\"k\": \""));
    // a three-byte character split over three tokens, and one that stops mid-way
    CHECK(s.allows("\xE4"));
    CHECK(s.accept("\xE4"));
    CHECK(s.accept("\xB8"));
    CHECK(!s.accepting());
    CHECK(s.accept("\xAD\"}"));
    CHECK(s.accepting());
    GrammarState t(g);
    CHECK(!t.allows("\xC3"));                 // not inside a string
    CHECK(t.accept("{\"") && !t.allows("\xC0\x80") && !t.allows("\xFF") && !t.allows("\x80"));
    CHECK(t.allows("\xC3\xA9") && t.allows("\xF0\x9F\x98"));
    CHECK(!t.allows("\xED\xA0\x80"));         // a surrogate
    // a class that excludes the range stops a partial early
    auto ascii = parse("root ::= [a-z]+");
    GrammarState a(ascii);
    CHECK(!a.allows("\xC3"));
    auto latin = parse("root ::= [\\u00e0-\\u00ff]+");
    GrammarState l(latin);
    CHECK(l.allows("\xC3") && !l.allows("\xE4"));
}

static void test_clone_rewind() {
    auto g = parse(kJson);
    GrammarState s(g);
    CHECK(s.accept("{\"a\": "));
    GrammarState mark = s.clone();
    CHECK(s.accept("[1, 2"));
    CHECK(!s.accept("}"));                    // a failed accept leaves the state as it was
    CHECK(s.accept("]}"));
    CHECK(s.accepting());
    CHECK(!mark.accepting());
    CHECK(mark.accept("7}"));
    CHECK(mark.accepting());
    s = mark.clone();
    CHECK(s.accepting());
    s.reset();
    CHECK(!s.accepting() && s.accept("{}") && s.accepting());
}

// A small vocabulary with multi-byte, partial and JSON-shaped pieces.
static Vocab test_vocab(int32_t& eos) {
    std::vector<std::string> v;
    for (int i = 0; i < 256; ++i) v.push_back(std::string(1, (char) i));
    for (const char* t : {"{\"", "\":", "\": ", "\",", "\"}", "true", "false", "null", "[]", "{}", ", ", "\\n", "\\u00",
                          "abc", " abc", "123", "-1", "0.5", "e+", "\xC3\xA9", "\xE4\xB8", "\xE4\xB8\xAD", "\xF0\x9F",
                          "\xF0\x9F\x98\x80", "}\n", "\n  ", "  ", "\"\"", "[1", "]]"})
        v.push_back(t);
    eos = (int32_t) v.size();
    v.push_back("");                           // the end token has no text
    v.push_back("");                           // a special token
    return Vocab(v, {eos});
}

static void test_mask() {
    auto g = parse(kJson);
    int32_t eos;
    Vocab voc = test_vocab(eos);
    const char* prefixes[] = {"", "{", "{\"a\"", "{\"a\": ", "{\"a\": 1", "{\"a\": \"x", "{\"a\": \"\xE4", "{\"a\": [1,",
                              "{\"a\": tr", "{\"a\": 1.", "{}", "{\"a\": \"x\\", "{\"a\": \"x\\u0"};
    for (const char* p : prefixes) {
        GrammarState s(g);
        CHECK(s.accept(p));
        std::vector<uint8_t> mask;
        s.fill_mask(voc, mask);
        std::vector<uint8_t> again;
        s.fill_mask(voc, again);                  // the second call comes from the cache
        CHECK(again == mask);
        s.fill_mask(voc, again, false);
        CHECK(again == mask);
        CHECK(mask.size() == voc.size());
        int bad = 0, allowed = 0;
        for (size_t t = 0; t < voc.size(); ++t) {
            bool want = (int32_t) t == eos ? s.accepting() : (!voc.bytes((int32_t) t).empty() && s.allows(voc.bytes((int32_t) t)));
            if ((mask[t] != 0) != want) ++bad;
            allowed += mask[t];
        }
        if (bad) std::printf("mask mismatch after '%s': %d\n", p, bad);
        CHECK(bad == 0);
        CHECK(allowed > 0);
    }
    GrammarState s(g);
    std::vector<uint8_t> mask;
    s.fill_mask(voc, mask);
    CHECK(mask['{'] && !mask['['] && !mask['a'] && !mask[(size_t) eos]);
    CHECK(s.accept("{}"));
    s.fill_mask(voc, mask);
    CHECK(mask[(size_t) eos] && mask[' ']);
}

static void test_lazy() {
    auto g = parse("root ::= \"<c>\" [a-z]+ \"</c>\"");
    Triggers tr;
    tr.words = {"<c>"};
    GrammarState s(g, tr);
    CHECK(!s.active() && s.accepting());
    CHECK(s.accept("Hello, 123 <b"));            // free text, not a trigger
    CHECK(!s.active());
    CHECK(s.allows("anything at all"));
    CHECK(s.accept("><"));                        // "<b><": no trigger yet
    CHECK(!s.active());
    CHECK(s.accept("c"));                         // "<" "c" is still short of "<c>"
    CHECK(!s.active());
    CHECK(s.accept(">"));                         // "<c>" completes across tokens
    CHECK(s.active() && !s.accepting());
    CHECK(!s.allows("1"));
    CHECK(s.accept("abc") && s.accept("</c>"));
    CHECK(!s.active() && s.accepting());          // complete: waiting for the next trigger
    CHECK(s.accept("more text <c>x</c>"));        // a whole call inside one token
    CHECK(!s.active());
    // a trigger followed by a violation in the same token is refused
    CHECK(!s.allows("<c>1"));
    CHECK(s.accept("<c>q") && s.active());
    // a trigger token
    Triggers tt;
    tt.tokens = {42};
    GrammarState k(g, tt);
    CHECK(k.accept("<c>", 7) == true && !k.active());   // not the trigger token: the text is just text
    CHECK(k.accept("<c>", 42) && k.active());
    int32_t eos;
    Vocab voc = test_vocab(eos);
    GrammarState f(g, tr);
    std::vector<uint8_t> mask;
    f.fill_mask(voc, mask);
    size_t sum = 0;
    for (uint8_t m : mask) sum += m;
    CHECK(sum == voc.size());                     // inactive: nothing is masked
}

// <[id]> / !<[id]> references: matched by token id, in the mask and in accept.
static void test_token_refs() {
    auto g = parse("root ::= <[300]> content <[301]>\ncontent ::= (!<[301]>)*");
    CHECK(g && g->elements().size() > 0);
    int32_t eos;
    std::vector<std::string> v;
    for (int i = 0; i < 256; ++i) v.push_back(std::string(1, (char) i));
    v.push_back("ab");          // 256
    while (v.size() < 300) v.push_back("x" + std::to_string(v.size()));
    v.push_back("");            // 300: a special token with no text
    v.push_back("");            // 301: another
    v.push_back("zz");          // 302
    eos = (int32_t) v.size();
    v.push_back("");
    Vocab voc(v, {eos});
    GrammarState s(g);
    std::vector<uint8_t> mask;
    s.fill_mask(voc, mask);
    size_t n = 0;
    for (uint8_t m : mask) n += m;
    CHECK(n == 1 && mask[300]);                     // only the opener, though it has no text
    CHECK(!s.accept("<[300]>", 1) && s.accept("", 300));
    s.fill_mask(voc, mask);
    CHECK(mask[301]);                               // the closer is allowed
    CHECK(mask[0] && mask[256] && mask[302] && !mask[(size_t) eos]);
    // every id: the mask agrees with accept (empty-text ids only count when a reference names them)
    for (size_t id = 0; id < voc.size(); ++id) {
        if ((int32_t) id == eos) continue;
        bool want = mask[id] != 0;
        bool got = s.allows(voc.bytes((int32_t) id), (int32_t) id);
        if (voc.bytes((int32_t) id).empty() && id != 301) continue;
        CHECK(want == got);
    }
    CHECK(s.accept("ab", 256) && s.accept("zz", 302) && s.accept("", 301));
    CHECK(s.accepting() && !s.accept("a", 5));
    // a token that is not the named id does not take a reference's place, whatever its text says
    GrammarState t(g);
    CHECK(!t.accept("<[300]>", 5) && !t.accept("a", 5));
    // <name> needs a resolver
    std::string err;
    CHECK(!Grammar::parse("root ::= <think> \"a\"", err) && !err.empty());
    ParseOptions opt;
    opt.token = [](std::string_view name, uint32_t& id) { if (name == "<think>") { id = 77; return true; } return false; };
    auto named = Grammar::parse("root ::= <think> \"a\"", err, opt);
    CHECK(named != nullptr);
    GrammarState u(named);
    CHECK(!u.accept("a", 5) && u.accept("<think>", 77) && u.accept("a") && u.accepting());
    CHECK(!Grammar::parse("root ::= <nope> \"a\"", err, opt));
}

// A regex trigger: the grammar starts at the first non-empty capture group (else the match), over tokens.
static void test_patterns() {
    auto g = parse("root ::= \"<c>\" [a-z]+ \"</c>\"");
    Triggers tr;
    tr.patterns = {"[\\s\\S]*?(<c>)"};
    GrammarState s(g, tr);
    CHECK(!s.active());
    CHECK(s.accept("say ", 1) && s.accept("<", 2) && !s.active());
    CHECK(s.accept("c>ab", 3) && s.active());     // the group starts at "<c>"; the text from there is replayed
    CHECK(!s.allows("1") && s.accept("c") && s.accept("</c>") && s.accepting() && !s.active());
    // no group: the match itself is where the grammar starts
    Triggers m;
    m.patterns = {"<c>"};
    GrammarState a(g, m);
    CHECK(a.accept("zz<c>q", 1) && a.active());
    // a whole-output pattern
    Triggers w;
    w.patterns = {"^hi (<c>)$"};
    GrammarState b(g, w);
    CHECK(b.accept("hi ", 1) && !b.active());
    CHECK(b.accept("<c>", 2) && b.active());
    // a bad pattern is refused up front
    Triggers bad;
    bad.patterns = {"(["};
    std::string err;
    CHECK(!bad.valid(err) && !err.empty());
    bool threw = false;
    try { GrammarState x(g, bad); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    // a token reference inside a replayed trigger sees the token ids
    auto r = parse("root ::= <[5]> \"x\"");
    Triggers rt;
    rt.patterns = {"(g)"};
    GrammarState c(r, rt);
    CHECK(!c.accept("g", 1) && !c.active());      // the trigger fires, the grammar wants token 5: refused, unchanged
    CHECK(c.accept("g", 5) && c.active() && c.accept("x") && c.accepting());
    // bounded pattern after a long inactive prefix (past tail trim) still matches across token boundaries,
    // preserves capture-group offsets and replayed token ids, and rolls back cleanly on failed fire
    auto mg = parse("root ::= \"<|message_model|>\" [a-z]+ \"<|end|>\"");
    Triggers mt;
    mt.patterns = {"<\\|message_model\\|>[^<]"};
    GrammarState ms(mg, mt);
    for (int i = 0; i < 2000; ++i) CHECK(ms.accept("hello world ", 10) && !ms.active());
    CHECK(ms.accept("<|message_model|>", 20) && !ms.active());
    CHECK(!ms.accept("1", 21) && !ms.active());   // fires on "<|message_model|>1", grammar rejects '1', rolls back
    CHECK(ms.accept("ab", 22) && ms.active());    // previous header piece survived rollback and matches now
    CHECK(ms.accept("c<|end|>", 23) && !ms.active() && ms.accepting());
    // capture group inside a bounded pattern after tail trim (lo > 0)
    Triggers cap;
    cap.patterns = {"pre_(<c>)"};
    GrammarState cs(g, cap);
    for (int i = 0; i < 400; ++i) CHECK(cs.accept("padding ", 1) && !cs.active());
    CHECK(cs.accept("pre_<", 2) && !cs.active());
    CHECK(cs.accept("c>xy", 3) && cs.active());
    CHECK(cs.accept("</c>", 4) && !cs.active());
    // token reference inside a bounded multi-token trigger after tail trim
    auto rg = parse("root ::= <[5]> <[6]> \"z\"");
    Triggers rgt;
    rgt.patterns = {"ab"};
    GrammarState rs(rg, rgt);
    for (int i = 0; i < 400; ++i) CHECK(rs.accept("----", 1) && !rs.active());
    CHECK(rs.accept("a", 5) && !rs.active());
    CHECK(!rs.accept("b", 9) && !rs.active());    // wrong id for 'b': rolls back keeping token 5's 'a'
    CHECK(rs.accept("b", 6) && rs.active() && rs.accept("z", 7) && !rs.active());
    // word-boundary pattern stays unbounded (falls back to full scan)
    auto cg = parse("root ::= \"call\" [a-z]+ \"!\"");
    Triggers wb;
    wb.patterns = {"\\bcall"};
    GrammarState ws(cg, wb);
    CHECK(ws.accept("xcall", 1) && !ws.active());
    CHECK(ws.accept(" callq", 2) && ws.active() && ws.accept("!", 3) && !ws.active());
}

// A byte sequence split across tokens: a prefix is fine while some completion can match.
static void test_partial_utf8() {
    auto g = parse("root ::= \"\xC3\xA9\" \"a\"");
    std::vector<std::string> v;
    for (int i = 0; i < 256; ++i) v.push_back(std::string(1, (char) i));
    v.push_back("\xC3\xA9");   // 256
    v.push_back("\xE2\x82");   // 257: a prefix of a euro sign
    v.push_back("\xC3\xA9" "a");
    Vocab voc(v, {});
    GrammarState s(g);
    std::vector<uint8_t> mask;
    s.fill_mask(voc, mask);
    CHECK(mask[0xC3] && mask[256] && mask[258] && !mask[0xA9] && !mask[0xE2] && !mask[257] && !mask['a']);
    CHECK(s.accept("\xC3") && !s.accepting());
    s.fill_mask(voc, mask);
    CHECK(mask[0xA9] && !mask[0xC3] && !mask['a']);
    CHECK(s.accept("\xA9") && s.accept("a") && s.accepting());
    // a class that takes any code point takes any lead byte; a negated class too
    auto any = parse("root ::= . [^a]");
    GrammarState t(any);
    t.fill_mask(voc, mask);
    CHECK(mask[0xE2] && mask[257] && mask['a'] && !mask[0xA9] && !mask[0xFF]);
    CHECK(t.accept("\xE2\x82") && t.accept("\xAC") && t.accept("\xF0\x9F") && t.accept("\x98\x80") && t.accepting());
    // the same stacks with a character half read give another mask, and the cached one for the whole state is not served
    GrammarState w(any);
    w.fill_mask(voc, mask);
    CHECK(mask['a'] && mask[0xE2] && !mask[0xA9]);
    CHECK(w.accept("\xE2"));
    w.fill_mask(voc, mask);
    CHECK(mask[0xA9] && !mask['a'] && !mask[0xE2]);
    // a continuation byte after a finished character, or a bad lead, is refused
    GrammarState u(any);
    CHECK(!u.accept("\x80") && !u.accept("\xC0\x80") && !u.accept("\xFF"));
    // partial UTF-8 prefixes whose only completions are surrogates (0xED 0xA0..0xBF) or > U+10FFFF (0xF4 0x90..0xBF)
    // must be rejected before the dead continuation byte so masked sampling cannot deadlock
    CHECK(!u.allows("\xED\xA0") && !u.allows("\xED\xBF") && u.allows("\xED\x9F"));
    CHECK(!u.allows("\xF4\x90") && !u.allows("\xF4\xBF") && u.allows("\xF4\x8F"));
    // negated character class excluding a single non-ASCII code point (or a sub-range): lead bytes that still have
    // valid unexcluded completions must be allowed across token boundaries and in fill_mask()
    auto not_e = parse("root ::= [^\\u00e9]");
    GrammarState ne(not_e);
    CHECK(ne.allows("\xC3\xA0") && !ne.allows("\xC3\xA9"));
    CHECK(ne.allows("\xC3"));
    ne.fill_mask(voc, mask);
    CHECK(mask[0xC3] && !mask[256]);
    CHECK(ne.accept("\xC3") && !ne.allows("\xA9") && ne.accept("\xA0") && ne.accepting());
    // negated class excluding C1 controls (0x80..0x9F): lead byte 0xC2 still allows 0xA0..0xBF
    auto not_c1 = parse("root ::= [^\\x7f-\\u009f]");
    GrammarState nc1(not_c1);
    CHECK(nc1.allows("\xC2") && nc1.accept("\xC2") && !nc1.allows("\x80") && nc1.accept("\xA0") && nc1.accepting());
    // negated class covering the entire range of lead byte 0xC3 (U+00C0..U+00FF) across two adjacent ranges must reject 0xC3
    auto not_c3 = parse("root ::= [^\\u00c0-\\u00df\\u00e0-\\u00ff]");
    GrammarState nc3(not_c3);
    CHECK(!nc3.allows("\xC3") && nc3.allows("\xC2"));
}

static double now_us() {
    using namespace std::chrono;
    return (double) duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count() / 1000.0;
}

// A synthetic vocabulary of the real size: byte tokens, words with and without a leading space, digits, punctuation
// runs and multi-byte characters.
static std::vector<std::string> bench_vocab(size_t n) {
    std::vector<std::string> v;
    for (int i = 0; i < 256; ++i) v.push_back(std::string(1, (char) i));
    uint64_t x = 88172645463325252ull;
    auto rnd = [&]() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
    const char* punct[] = {"\"", "\":", "\",", "{\"", "\"}", "},", "],", "\\n", "\\\"", "://", "=>", "();", "\n\n", "  ", "    "};
    while (v.size() < n) {
        std::string t;
        uint64_t k = rnd() % 100;
        if (k < 60) {
            if (rnd() % 2) t += ' ';
            size_t len = 2 + rnd() % 8;
            for (size_t i = 0; i < len; ++i) t += (char) ('a' + rnd() % 26);
        } else if (k < 70) {
            size_t len = 1 + rnd() % 4;
            for (size_t i = 0; i < len; ++i) t += (char) ('0' + rnd() % 10);
        } else if (k < 85) {
            t = punct[rnd() % 15];
        } else {
            size_t len = 1 + rnd() % 3;
            for (size_t i = 0; i < len; ++i) {
                uint32_t cp = 0x4E00 + (uint32_t) (rnd() % 20000);
                t += (char) (0xE0 | (cp >> 12));
                t += (char) (0x80 | ((cp >> 6) & 0x3F));
                t += (char) (0x80 | (cp & 0x3F));
            }
        }
        v.push_back(t);
    }
    return v;
}

// BASALT_GRAMMAR_VOCAB names a file of (u32 length, bytes) records, one per token id, for a run on the real vocabulary;
// BASALT_GRAMMAR_END is the end token's id.
static bool load_vocab(const char* path, std::vector<std::string>& out) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    uint32_t n;
    while (std::fread(&n, sizeof n, 1, f) == 1) {
        std::string t(n, '\0');
        if (n && std::fread(&t[0], 1, n, f) != n) break;
        out.push_back(std::move(t));
    }
    std::fclose(f);
    return !out.empty();
}

static void bench() {
    std::vector<std::string> tokens;
    const char* vpath = std::getenv("BASALT_GRAMMAR_VOCAB");
    const char* eos = std::getenv("BASALT_GRAMMAR_END");
    bool real = vpath && load_vocab(vpath, tokens);
    if (!real) tokens = bench_vocab(248320);
    const size_t N = tokens.size();
    std::printf("bench grammar: %s vocabulary, %zu tokens\n", real ? "real" : "synthetic", N);
    Vocab voc(std::move(tokens), {eos ? std::atoi(eos) : 0});
    auto g = parse(kJson);
    std::string err;
    auto tools = Grammar::parse(tool_call_gbnf(R"([{"name":"f","parameters":{"properties":{"text":{"type":"string"},"n":{"type":"integer"},"obj":{"type":"object","properties":{"k":{"type":"string"}}}},"required":["text"]}}])", {}, err), err);
    struct Case { const char* where; std::shared_ptr<const Grammar> g; const char* text; } cases[] = {
        {"json, start of an object", g, "{"},
        {"json, inside a string value", g, "{\"key\": \"some text"},
        {"json, inside a nested array of objects", g, "{\"a\": [{\"b\": [1, 2, {\"c\": \"x"},
        {"tool call, inside a string parameter", tools, "<tool_call>\n<function=f>\n<parameter=text>\nhello wor"},
        {"tool call, inside an integer parameter", tools, "<tool_call>\n<function=f>\n<parameter=text>\nhi\n</parameter>\n<parameter=n>\n12"},
        {"tool call, between parameters", tools, "<tool_call>\n<function=f>\n<parameter=text>\nhi\n</parameter>\n"},
    };
    for (const Case& c : cases) {
        GrammarState s(c.g);
        s.accept(c.text);
        const int reps = 2000;
        double t0 = now_us();
        int ok = 0;
        for (int i = 0; i < reps; ++i) ok += s.allows(" the") ? 1 : 0;
        double allows_us = (now_us() - t0) / reps;
        std::vector<uint8_t> mask;
        s.fill_mask(voc, mask, false);     // first call builds the memo tables
        double best = 1e18, cached = 1e18;
        size_t allowed = 0;
        for (int i = 0; i < 5; ++i) {
            double a = now_us();
            s.fill_mask(voc, mask, false);
            best = (std::min)(best, now_us() - a);
        }
        s.fill_mask(voc, mask);
        for (int i = 0; i < 5; ++i) {
            double a = now_us();
            s.fill_mask(voc, mask);
            cached = (std::min)(cached, now_us() - a);
        }
        for (uint8_t m : mask) allowed += m;
        std::printf("bench grammar: %-40s allows(1 token) %.2f us (%d/%d)  fill_mask(%zu tokens) %.2f ms, repeated position %.3f ms  allowed %zu\n",
                    c.where, allows_us, ok, reps, N, best / 1000.0, cached / 1000.0, allowed);
    }
}

int main() {
    set_test_tool_form();
    test_parse();
    test_basic();
    test_json();
    test_utf8_tokens();
    test_clone_rewind();
    test_mask();
    test_lazy();
    test_token_refs();
    test_patterns();
    test_partial_utf8();
    if (std::getenv("BASALT_GRAMMAR_BENCH")) bench();
    if (g_fail) { std::printf("grammar_test: %d failures\n", g_fail); return 1; }
    std::printf("grammar_test: ok\n");
    return 0;
}
