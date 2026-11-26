// tests/models/glm5next/adapter_test.cpp - what the glm5next adapter answers, through the engine's own paths (CPU only).
//
// The answers a model gives about text are read by the start rather than by the load, so a wrong one changes behaviour
// silently instead of refusing the pack: the pre-tokenizer decides how a prompt is split, the token names become the
// defaults of --eos-ids and --turn-token, and the GPU modes decide what the launcher will even try. The one answer
// worth testing as behaviour rather than as a value is the tool-call form, because the grammar built from it is what
// constrains the model at inference: it has to accept a call the model really writes and refuse one with a marker
// missing.
//
// validate() and describe() are not exercised here: they read the GGUF's `glm5-next.*` keys, which only a pack
// converted from the release carries (the tiny fixture holds its geometry as JSON, which is what the pack reader
// uses, and opening its path as a GGUF fails on the magic). A converted pack keeps them in `meta.gguf_kv`, and
// `ModelFile::register_gguf_view` then serves them to `GgufFile(path)`; checking a real pack's is a §5.6 step.
#include "models/glm5next/adapter.hpp"
#include "models/glm5next/meta.hpp"

#include "basalt/program/glm_forms.hpp"
#include "basalt/program/grammar/guard.hpp"
#include "basalt/program/grammar/tool_form.hpp"

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

using namespace basalt;
using namespace basalt::models;
using namespace basalt::program;

namespace {

int g_failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++g_failures;
    }
}

// A vocabulary that spells a tool call the GLM way, one token per piece: `<tool_call>` NAME `<arg_key>` PARAM
// `</arg_key>` `<arg_value>` VALUE `</arg_value>` `</tool_call>`. Types: 4 = user-defined added token, 1 = text.
const char* kVocab = R"({
  "<tool_call>": 0, "get_weather": 1, "<arg_key>": 2, "city": 3, "</arg_key>": 4,
  "<arg_value>": 5, "Paris": 6, "</arg_value>": 7, "</tool_call>": 8, " ": 9
})";
const char* kTypes = "[4, 1, 4, 1, 4, 4, 1, 4, 4, 1]";
const char* kTools =
    R"({"kind":"tools","tools":[{"type":"function","function":{"name":"get_weather","parameters":{)"
    R"("type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]})";

grammar::PackVocab pack_vocab() {
    grammar::PackVocab pv;
    std::string err;
    if (!grammar::vocab_from_pack(kVocab, kTypes, 10, {}, pv, err)) {
        std::printf("FAIL: the test vocabulary was refused: %s\n", err.c_str());
        ++g_failures;
    }
    return pv;
}

// ids of a call, in the order the model writes them.
const int32_t kCall[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
// the same call with `</arg_key>` left out: the parameter's name runs into its value opener.
const int32_t kMissingClose[] = {0, 1, 2, 3, 5, 6, 7, 8};

void test_grammar(const model::ModelAdapter& a) {
    grammar::set_tool_form(a.tool_form());
    grammar::Guard g(pack_vocab());
    std::string err;
    if (!g.begin(kTools, nullptr, 0, err)) {
        std::printf("FAIL: the tool grammar was refused: %s\n", err.c_str());
        ++g_failures;
        return;
    }
    check(g.on(), "the grammar is on for a request with tools");
    check(g.scan(kCall, static_cast<int>(sizeof(kCall) / sizeof(kCall[0])), 0) == -1,
          "a tool call the model writes passes the grammar built from tool_form()");

    grammar::Guard bad(pack_vocab());
    check(bad.begin(kTools, nullptr, 0, err) &&
              bad.scan(kMissingClose, static_cast<int>(sizeof(kMissingClose) / sizeof(kMissingClose[0])), 0) != -1,
          "a call with a marker missing is refused");
}

}  // namespace

int main() {
    // make_adapter() must hand over an adapter: the engine registers what it returns without looking at it, so a null
    // here is a crash at every start, not a model that fails to load.
    std::unique_ptr<model::ModelAdapter> a = glm5next::make_adapter();
    check(a != nullptr, "make_adapter() returns an adapter");
    if (!a) return 1;
    check(std::strcmp(a->arch(), glm5next::kArch) == 0, "the adapter answers for the Basalt id");
    check(model::find_adapter(glm5next::kGgufArch) != nullptr,
          "and registers itself under the GGUF's own architecture name");
    model::register_adapter(std::move(a));
    const model::ModelAdapter& ad = *model::find_adapter(glm5next::kArch);

    // What the start reads: the split function, the tokens, and the cards the model asks for.
    check(ad.pretokenizer().split == &program::tokenizer::pretok_glm4, "the pre-tokenizer is the glm4 split");
    check(std::strcmp(ad.pretokenizer().name, "glm4") == 0, "named as the checkpoint names it");
    const model::TokenNames n = ad.token_names();
    check(n.eos.size() == 3 && n.eos[0] == "<|endoftext|>" && n.eos[1] == "<|user|>" && n.eos[2] == "<|observation|>",
          "the end tokens are the checkpoint's three eos ids");
    check(n.turn == "<|assistant|>" && n.reply_role.empty(), "the model's own turn token opens its reply");
    const program::GpuModes m = ad.gpu_modes();
    check(m.peer == 2 && m.split == 0 && m.tensor == 0, "the main card and one peer that holds experts");
    check(m.peer_ada, "an Ada card can be that peer, whose kernels -DPEER_ADA=ON builds for sm_89");

    test_grammar(ad);

    if (g_failures) {
        std::printf("ADAPTER: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("ADAPTER: ok\n");
    return 0;
}
