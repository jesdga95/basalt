// tests/model/adapter_builtin_test.cpp - the program's own registration: both real adapters, once each, and DeepSeek's
// load error. Linking the models needs the CUDA runtime, so this runs where that is installed; no GPU work.
#include "basalt/model/adapter.hpp"
#include "basalt/models/builtin.hpp"

#include <cstdio>
#include <memory>

int main() {
    using namespace basalt;
    int failures = 0;
    const auto check = [&](bool ok, const char* what) {
        std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
        if (!ok) ++failures;
    };
    models::register_builtin_models();
    models::register_builtin_models();
    check(model::find_adapter("qwen4exp") && model::find_adapter("deepseek41"), "both adapters registered");
    check(model::adapter_names().size() == 2, "registering twice keeps one adapter per name");
    if (const model::ModelAdapter* ds = model::find_adapter("deepseek41")) {
        std::unique_ptr<model::LoadedModel> loaded;
        check(ds->load({}, loaded).status == model::Status::Error, "deepseek41 load without a path is an error");
    }
    return failures ? 1 : 0;
}
