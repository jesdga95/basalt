// tests/program/grammar_guard_test.cpp - the pack vocabulary reader and the per-request window check (CPU only).
#include "tool_form_fixture.hpp"
#include "basalt/program/grammar/guard.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace basalt::program::grammar;

static int g_fail = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

// ids: 0 { 1 } 2 "a" 3 : 4 1 5 True 6 <think> 7 </think> 8 <|im_end|> 9 " x" 10 "\n" 11 <tool_call> 12 hello
static const char* kVocab =
    "{\"{\": 0, \"}\": 1, \"\\\"a\\\"\": 2, \":\": 3, \"1\": 4, \"True\": 5, \"<think>\": 6, \"</think>\": 7, "
    "\"<|im_end|>\": 8, \"\xC4\xA0x\": 9, \"\xC4\x8A\": 10, \"<tool_call>\": 11, \"hello\": 12}";
static const char* kTypes = "[1, 1, 1, 1, 1, 1, 4, 4, 3, 1, 1, 4, 1]";
static const char* kTools =
    R"({"kind":"tools","tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object",)"
    R"("properties":{"flag":{"type":"boolean"}},"required":["flag"]}}}],"tool_choice":"auto"})";

static PackVocab load() {
    PackVocab pv;
    std::string err;
    CHECK(vocab_from_pack(kVocab, kTypes, 14, {8}, pv, err));
    return pv;
}

// the refused row's draws: `tokens` are returned in turn (the last repeats); the masks they were drawn under are kept
struct Draws {
    std::vector<int32_t> tokens;
    std::vector<std::vector<float>> masks;
    std::vector<float> logits;
    bool fail = false;
    DrawFn draw() {
        return [this](const std::vector<float>& bias, int32_t& t) {
            masks.push_back(bias);
            t = tokens[std::min(masks.size(), tokens.size()) - 1];
            return !fail;
        };
    }
    LogitsFn read() {
        return [this](std::vector<float>& lg) { lg = logits; return true; };
    }
};

int main() {
    set_test_tool_form();
    const PackVocab pv = load();
    CHECK(pv.vocab->bytes(0) == "{");
    CHECK(pv.vocab->bytes(2) == "\"a\"");
    CHECK(pv.vocab->bytes(9) == " x");
    CHECK(pv.vocab->bytes(10) == "\n");
    CHECK(pv.vocab->bytes(8).empty());
    CHECK(pv.vocab->bytes(11) == "<tool_call>");
    CHECK(pv.vocab->bytes(13).empty());
    CHECK(pv.think_open == 6 && pv.think_close == 7);
    {
        PackVocab bad;
        std::string err;
        CHECK(!vocab_from_pack("[1]", kTypes, 14, {}, bad, err));
        CHECK(!vocab_from_pack(kVocab, "{}", 14, {}, bad, err));
    }

    const std::string jo = R"({"kind":"json_object"})";
    std::string err;
    {   // a valid object passes; a refused row is reported and its mask allows the object's opener only
        Guard g(pv);
        CHECK(g.begin(jo, nullptr, 0, err) && g.on());
        const int32_t good[] = {0, 2, 3, 4, 1};
        CHECK(g.scan(good, 5, 0) == -1);
    }
    {
        Guard g(pv);
        CHECK(g.begin(jo, nullptr, 0, err));
        const int32_t bad[] = {12};
        CHECK(g.scan(bad, 1, 0) == 0);
        std::vector<float> user(pv.vocab->size(), 0.0f);
        user[0] = 5.0f;
        Draws d{{0}, {}, {}};
        int32_t tok = -1;
        CHECK(g.redraw(user, d.draw(), d.read(), tok) == Redraw::drawn && tok == 0);
        CHECK(d.masks.size() == 1 && d.masks[0].size() == pv.vocab->size());
        CHECK(d.masks[0][0] == 5.0f && std::isinf(d.masks[0][12]) && std::isinf(d.masks[0][8]));
        const int32_t rest[] = {2, 3, 4, 1};
        CHECK(g.scan(rest, 4, 0) == -1);
        CHECK(g.misses() == 1 && g.on());
    }
    {   // an end token only where the object is complete
        Guard g(pv);
        CHECK(g.begin(jo, nullptr, 0, err));
        const int32_t early[] = {0, 8};
        CHECK(g.scan(early, 2, 0) == 1);
        Guard h(pv);
        CHECK(h.begin(jo, nullptr, 0, err));
        const int32_t done[] = {0, 1, 8};
        CHECK(h.scan(done, 3, 0) == -1);
    }
    {   // a prompt that ends inside a thinking block: free text until the block closes
        Guard g(pv);
        const int64_t prompt[] = {9, 10, 6, 10};
        CHECK(g.begin(jo, prompt, 4, err));
        const int32_t out[] = {12, 12, 7, 0, 1};
        CHECK(g.scan(out, 5, 0) == -1);
        Guard h(pv);
        const int64_t closed[] = {6, 10, 7, 10};
        CHECK(h.begin(jo, closed, 4, err));
        const int32_t first[] = {12};
        CHECK(h.scan(first, 1, 0) == 0);
    }
    {   // tools: free text passes, the opener starts the grammar, then the layout is enforced
        Guard g(pv);
        CHECK(g.begin(kTools, nullptr, 0, err) && g.on());
        const int32_t text[] = {12, 9, 12};
        CHECK(g.scan(text, 3, 0) == -1);
        const int32_t call[] = {11, 12};
        CHECK(g.scan(call, 2, 0) == 1);
        Draws d{{10}, {}, {}};
        int32_t tok = -1;
        CHECK(g.redraw({}, d.draw(), d.read(), tok) == Redraw::drawn);
        CHECK(d.masks[0][10] == 0.0f && std::isinf(d.masks[0][12]));
    }
    {   // tool_choice none: no grammar, the guard stays off
        Guard g(pv);
        CHECK(g.begin(R"({"kind":"tools","tools":[{"name":"f","parameters":{}}],"tool_choice":"none"})", nullptr, 0, err));
        CHECK(!g.on());
    }
    {   // a slot's guard is its own: two interleaved windows, one refused row each, the states stay apart
        Guard a(pv), b(pv);
        CHECK(a.begin(jo, nullptr, 0, err) && b.begin(jo, nullptr, 0, err));
        const int32_t wa[] = {0, 2, 12, 4};   // refused at row 2
        const int32_t wb[] = {0, 2, 3, 4};
        CHECK(a.scan(wa, 4, 0) == 2);
        CHECK(b.scan(wb, 4, 0) == -1);
        Draws d{{3}, {}, {}};
        int32_t tok = -1;
        CHECK(a.redraw({}, d.draw(), d.read(), tok) == Redraw::drawn && tok == 3);
        CHECK(d.masks[0][3] == 0.0f && std::isinf(d.masks[0][12]));
        const int32_t na[] = {4, 1};
        CHECK(a.scan(na, 2, 0) == -1);   // the corrected row 3 is what the slot continues from
        const int32_t nb[] = {1};
        CHECK(b.scan(nb, 1, 0) == -1);
        CHECK(a.misses() == 1 && b.misses() == 0);
    }
    {   // a draw refused once is drawn again under a freshly built mask; the guard stays on
        Guard g(pv);
        CHECK(g.begin(jo, nullptr, 0, err));
        const int32_t bad[] = {12};
        CHECK(g.scan(bad, 1, 0) == 0);
        Draws d{{12, 0}, {}, {}};
        int32_t tok = -1;
        CHECK(g.redraw({}, d.draw(), d.read(), tok) == Redraw::drawn && tok == 0);
        CHECK(d.masks.size() == 2 && d.masks[0] == d.masks[1] && g.on() && g.fallbacks() == 0);
        const int32_t rest[] = {1, 8};
        CHECK(g.scan(rest, 2, 0) == -1);   // the object is complete: enforcement went on
    }
    {   // refused twice: the highest logit among the tokens the grammar accepts, never a refused one
        Guard g(pv);
        CHECK(g.begin(jo, nullptr, 0, err));
        const int32_t bad[] = {12};
        CHECK(g.scan(bad, 1, 0) == 0);
        Draws d{{12}, {}, std::vector<float>(pv.vocab->size(), 0.0f)};
        d.logits[12] = 9.0f;   // refused, and the likeliest of all
        d.logits[2] = 4.0f;    // `"a"` opens no object either
        d.logits[0] = 1.0f;    // `{`: the only token the grammar accepts
        int32_t tok = -1;
        CHECK(g.redraw({}, d.draw(), d.read(), tok) == Redraw::fallback && tok == 0);
        CHECK(d.masks.size() == 2 && g.fallbacks() == 1 && g.on());
        const int32_t rest[] = {1, 8};
        CHECK(g.scan(rest, 2, 0) == -1);
    }
    {   // the fallback adds the request's own bias to the logits
        Guard g(pv);
        CHECK(g.begin(jo, nullptr, 0, err));
        const int32_t bad[] = {12};
        CHECK(g.scan(bad, 1, 0) == 0);
        std::vector<float> user(pv.vocab->size(), 0.0f);
        Draws d{{12}, {}, std::vector<float>(pv.vocab->size(), 0.0f)};
        int32_t tok = -1;
        CHECK(g.redraw(user, d.draw(), d.read(), tok) == Redraw::fallback && tok == 0);
    }
    {   // no token allowed: the request's bias rules out every one the grammar allows
        Guard g(pv);
        CHECK(g.begin(jo, nullptr, 0, err));
        const int32_t bad[] = {12};
        CHECK(g.scan(bad, 1, 0) == 0);
        std::vector<float> user(pv.vocab->size(), 0.0f);
        user[0] = -INFINITY;   // `{` is the only token the grammar allows
        Draws d{{12}, {}, std::vector<float>(pv.vocab->size(), 0.0f)};
        int32_t tok = -7;
        CHECK(g.redraw(user, d.draw(), d.read(), tok) == Redraw::dead && tok == -7);
        CHECK(d.masks.empty() && !g.on());
    }
    {   // a failed draw or logits read is a failure, not a verdict on the grammar
        Guard g(pv);
        CHECK(g.begin(jo, nullptr, 0, err));
        const int32_t bad[] = {12};
        CHECK(g.scan(bad, 1, 0) == 0);
        Draws d{{12}, {}, {}};
        d.fail = true;
        int32_t tok = -1;
        CHECK(g.redraw({}, d.draw(), d.read(), tok) == Redraw::failed && g.on());
        Draws e{{12}, {}, {}};
        CHECK(g.redraw({}, e.draw(), [](std::vector<float>&) { return false; }, tok) == Redraw::failed);
    }
    {   // a first window's rows before the generated ones are prompt tokens: not fed to the grammar
        Guard g(pv);
        CHECK(g.begin(jo, nullptr, 0, err));
        const int32_t rows[] = {12, 9, 0, 1};   // two prompt tokens the object grammar would refuse, then {}
        CHECK(g.scan(rows, 4, 2) == -1);
        Guard h(pv);
        CHECK(h.begin(jo, nullptr, 0, err));
        CHECK(h.scan(rows, 4, 0) == 0);          // fed from row 0 they are refused
        Guard k(pv);
        CHECK(k.begin(jo, nullptr, 0, err));
        const int32_t bad[] = {12, 9, 12};
        CHECK(k.scan(bad, 3, 2) == 2);           // the first generated row is the one reported
    }
    {   // the last think tag decides, however far back it is
        Guard g(pv);
        std::vector<int64_t> prompt = {6};
        for (int i = 0; i < 40; ++i) prompt.push_back(12);
        CHECK(g.begin(jo, prompt.data(), prompt.size(), err));
        const int32_t free_text[] = {12, 12, 7, 0, 1};
        CHECK(g.scan(free_text, 5, 0) == -1);   // still inside the block: free text until it closes
        Guard h(pv);
        std::vector<int64_t> closed = {6, 7};
        for (int i = 0; i < 40; ++i) closed.push_back(12);
        CHECK(h.begin(jo, closed.data(), closed.size(), err));
        const int32_t first[] = {12};
        CHECK(h.scan(first, 1, 0) == 0);        // the block closed long ago: enforced from the first token
    }
    {
        Guard g(pv);
        CHECK(!g.begin("{", nullptr, 0, err));
    }
    if (g_fail == 0) std::printf("grammar_guard_test: ok\n");
    return g_fail == 0 ? 0 : 1;
}
