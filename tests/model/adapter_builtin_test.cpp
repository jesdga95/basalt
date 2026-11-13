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
    check(model::adapter_names().size() == built, "registering twice keeps one adapter per name");
    if (const model::ModelAdapter* ds = model::find_adapter("deepseek41")) {
        std::unique_ptr<model::LoadedModel> loaded;
        check(ds->load({}, loaded).status == model::Status::Error, "deepseek41 load without a path is an error");
        const auto o = ds->make_options();
        check(!ds->finish_flags(*o) && o->lossy.empty(), "deepseek41 has no lossy flag");
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
    return failures ? 1 : 0;
}
