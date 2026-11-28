// tests/model/adapter_builtin_test.cpp - the program's own registration: every adapter the build holds, once each, DeepSeek's
// load error and the flags each model reports as lossy. Linking the models needs the CUDA runtime, so this runs where
// that is installed; no GPU work.
#include "basalt/model/adapter.hpp"
#include "basalt/model/flags.hpp"
#include "basalt/models/builtin.hpp"
#include "basalt/program/options.hpp"

#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

int main() {
    using namespace basalt;
    int failures = 0;
    const auto check = [&](bool ok, const char* what) {
        std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
        if (!ok) ++failures;
    };
    models::register_builtin_models();
    models::register_builtin_models();
    // BASALT_MODELS decides which adapters exist: each built model is registered once, each left out is refused by name
    size_t built = 0;
    const auto expect = [&](const char* arch, bool is_built) {
        built += is_built ? 1 : 0;
        check((model::find_adapter(arch) != nullptr) == is_built, is_built ? "a built model is registered" : "a model left out is not registered");
        if (!is_built) {
            const std::string why = model::unavailable_architecture(arch);
            check(why.find("left out") != std::string::npos && why.find("BASALT_MODELS=") != std::string::npos &&
                      why.find(std::string("-DBASALT_MODELS=\"")) != std::string::npos && why.find(arch) != std::string::npos,
                  "a model left out is refused with the option that adds it");
        }
    };
#ifdef BASALT_MODEL_QWEN4EXP
    expect("qwen4exp", true);
#else
    expect("qwen4exp", false);
#endif
#ifdef BASALT_MODEL_DEEPSEEK41
    expect("deepseek41", true);
#else
    expect("deepseek41", false);
#endif
#ifdef BASALT_MODEL_QWEN35
    expect("qwen35", true);
#else
    expect("qwen35", false);
#endif
#ifdef BASALT_MODEL_HY4FLASH
    expect("hy4flash", true);
    // and under the architecture its test pack (made from the AngelSlim GGUF) carries
    check(model::find_adapter("hy_v4_flash") != nullptr, "hy4flash also answers to hy_v4_flash");
    ++built;
#else
    expect("hy4flash", false);
#endif
#ifdef BASALT_MODEL_INKLING2
    expect("inkling2", true);
#else
    expect("inkling2", false);
#endif
#ifdef BASALT_MODEL_ERNIE55
    expect("ernie55", true);
#else
    expect("ernie55", false);
#endif
#ifdef BASALT_MODEL_GEMMA4
    expect("gemma4", true);
#else
    expect("gemma4", false);
#endif
    check(model::adapter_names().size() == built, "registering twice keeps one adapter per name");
    if (const model::ModelAdapter* ds = model::find_adapter("deepseek41")) {
        model::Started started;
        check(ds->start({}, started).status == model::Status::Error, "deepseek41 start without a path is an error");
        const auto o = ds->make_options();
        check(!ds->finish_flags(*o) && o->lossy.empty(), "deepseek41 with no flag is not lossy");
    }
    // the flags as the command line gives them, through the model's own parser and finish_flags
    const auto lossy_of = [](const model::ModelAdapter& a, std::vector<std::string> words) {
        const auto o = a.make_options();
        words.insert(words.begin(), "basalt");
        std::vector<char*> argv;
        for (auto& w : words) argv.push_back(w.data());
        for (int i = 1; i < (int) argv.size(); ++i) {
            int code = 0;
            model::FlagArgs args((int) argv.size(), argv.data(), i);
            if (!a.parse_flag(argv[i], args, *o, code) || code != 0) return std::vector<program::CommonOptions::LossyFlag>{{"?", argv[i]}};
        }
        (void) a.finish_flags(*o);
        return o->lossy;
    };
    if (const model::ModelAdapter* q = model::find_adapter("qwen4exp")) {
        check(lossy_of(*q, {}).empty(), "qwen4exp with no flag is not lossy");
        const auto one = lossy_of(*q, {"--route-tail-skip", "7"});
        check(one.size() == 1 && one[0].flag == "--route-tail-skip" &&
                  one[0].effect == "--route-tail-skip 7 drops low-ranked missed experts",
              "qwen4exp --route-tail-skip 7 is lossy, with its warning");
        const auto all = lossy_of(*q, {"--relaxed-drafting", "--route-prior", "1.5", "--route-tail-skip"});
        check(all.size() == 3 && all[0].flag == "--relaxed-drafting" && all[1].effect == "--route-prior 1.5 changes expert routing" &&
                  all[2].effect == "--route-tail-skip 9 drops low-ranked missed experts",
              "qwen4exp: every lossy flag, bare ones at their defaults, in the warning order");
    }
    // the routing flags of every mixture-of-experts model: the bare prior is the model's own; the tail skip needs the pack's
    // experts per token, so without a pack it is refused
    for (const auto& [arch, bare] : {std::pair<const char*, const char*>{"deepseek41", "0.1"}, {"hy4flash", "0.01"}}) {
        const model::ModelAdapter* m = model::find_adapter(arch);
        if (m == nullptr) continue;
        const auto p = lossy_of(*m, {"--route-prior"});
        check(p.size() == 1 && p[0].effect == std::string("--route-prior ") + bare + " changes expert routing", "the bare --route-prior is the model's");
        const auto o = m->make_options();
        std::vector<std::string> words = {"basalt", "--route-tail-skip", "3"};
        std::vector<char*> argv;
        for (auto& w : words) argv.push_back(w.data());
        int i = 1, code = 0;
        model::FlagArgs args((int) argv.size(), argv.data(), i);
        check(m->parse_flag(argv[1], args, *o, code) && code == 0 && o->route.tail_skip == 3, "--route-tail-skip 3 is taken");
        check(m->finish_flags(*o).value_or(0) == 2, "--route-tail-skip without the pack's experts per token is refused");
    }
    if (const model::ModelAdapter* d = model::find_adapter("qwen35")) {
        const auto o = d->make_options();
        std::vector<std::string> words = {"basalt", "--route-prior"};
        std::vector<char*> argv;
        for (auto& w : words) argv.push_back(w.data());
        int i = 1, code = 0;
        model::FlagArgs args((int) argv.size(), argv.data(), i);
        check(!d->parse_flag(argv[1], args, *o, code), "a dense model does not take the routing flags");
    }
    if (const model::ModelAdapter* e = model::find_adapter("ernie55")) {
        const auto tf = e->tool_form();
        check(tf.call_open == "<ernie:tool_call>\n" && tf.call_close == "\n</ernie:tool_call>" &&
                  tf.fn_open == "<invoke name=\"" && tf.fn_mid == "\">" && tf.fn_close == "</invoke>" &&
                  tf.param_open == "<parameter name=\"" && tf.param_mid_text == "\">" &&
                  tf.param_mid_value == "\">" && tf.param_close == "</parameter>" &&
                  tf.text_end == "</parameter>" && tf.calls_share_block && tf.trigger == "<ernie:tool_call>",
              "ernie55 tool_form matches compact <invoke><parameter> wire format");
        const auto rf = e->reply_form();
        check(rf.think_open == "<ernie:think>" && rf.think_close == "</ernie:think>" &&
                  rf.header.empty() && rf.answer_open.empty(),
              "ernie55 reply_form uses <ernie:think> / </ernie:think>");
    }
    return failures ? 1 : 0;
}
