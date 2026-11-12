#include "startup.hpp"
#include "arguments.hpp"
#include "help.hpp"
#include "serve.hpp"

#include "basalt/core/model_file.hpp"
#include "basalt/log.hpp"
#include "basalt/model/adapter.hpp"
#include "basalt/platform/memory.hpp"
#include "basalt/platform/process.hpp"
#include "basalt/platform/system.hpp"
#include "basalt/program/grammar/tool_form.hpp"
#include "basalt/program/options.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace basalt::program::cli {

namespace {

// The architecture named in the file --model gives; empty without one. An exit code when it cannot be used.
std::optional<int> find_model(int argc, char** argv, const model::ModelAdapter*& adapter) {
    std::string path;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") return std::nullopt;
        if (a == "--model" && i + 1 < argc) path = argv[i + 1];
    }
    if (path.empty()) return std::nullopt;
    basalt::core::ModelFile file;
    std::string err;
    if (!file.open(path, err) || !file.load_meta(err)) {
        basalt::log::error("basalt", "--model: %s", err.c_str());
        return 2;
    }
    const basalt::core::Json* kv = file.meta().get("gguf_kv");
    const basalt::core::Json* arch = kv ? kv->get("general.architecture") : nullptr;
    const std::string name = arch && arch->is_string() ? arch->s : std::string();
    adapter = model::find_adapter(name);
    if (adapter == nullptr) {
        basalt::log::error("basalt", "--model: no model adapter for architecture '%s'", name.c_str());
        return 2;
    }
    return std::nullopt;
}

// What the flags left unset: each id is the pack's own token.
void pack_tokens(const model::ModelAdapter& adapter, CommonOptions& o) {
    basalt::core::ModelFile file;
    std::string err;
    std::vector<uint8_t> text;
    basalt::core::Json vocab;
    if (!file.open(o.model, err) || !file.load_meta(err) || !basalt::core::read_range(file.range("tokenizer/vocab.json"), text, err) ||
        !basalt::core::parse_json((const char*) text.data(), text.size(), vocab, err)) {
        basalt::log::warn("basalt", "the pack's vocabulary cannot be read (%s): no end-of-turn or turn token", err.c_str());
        return;
    }
    auto id_of = [&](const std::string& name) -> int64_t {
        const basalt::core::Json* v = vocab.get(name);
        return v != nullptr ? (int64_t) v->u64() : -1;
    };
    const model::TokenNames names = adapter.token_names();
    if (o.eos_ids.empty())
        for (const std::string& name : names.eos)
            if (const int64_t id = id_of(name); id >= 0) o.eos_ids.push_back(id);
    if (!o.turn_token_given) o.turn_token = id_of(names.turn);
    if (!o.image_pad_given) o.image_pad_id = id_of(names.image_pad);
    if (!names.video_pad.empty()) o.video_pad_id = id_of(names.video_pad);
}

}  // namespace

std::optional<int> start(int argc, char** argv, std::unique_ptr<CommonOptions>& out, const model::ModelAdapter*& adapter) {
    // **UNBUFFERED, BECAUSE THE INTERESTING OUTPUT IS THE OUTPUT BEFORE A CRASH.**  `stdout` redirected to a
    // pipe or a file is block-buffered, so a program that dies loses every line it had already printed - which
    // turns "it crashed at step 7" into "it crashed somewhere", and the difference is a debugging session.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc >= 2 && std::strcmp(argv[1], "serve") == 0) return serve_subcommand(argc, argv);
    // if memory runs out the kernel ends this process first, not the rest of the machine
    basalt::platform::prefer_oom_victim();
    // the CPU expert pool keeps its speed while the window is in the background
    if (!basalt::platform::keep_full_speed())
        basalt::log::warn("basalt", "full-speed scheduling refused: decode slows while the window is in the background");
    // Kernels load on first use: loading every kernel of the engine and of cuBLAS at context creation holds a few
    // hundred MiB of device memory for code that never runs.  The expert cache's sizing keeps back room for what loads
    // later (`kLazyKernelsMib`).  CUDA_MODULE_LOADING=EAGER restores the up-front load; batched serving sets it (below).
    const bool module_loading_set = std::getenv("CUDA_MODULE_LOADING") != nullptr;
    if (!module_loading_set) basalt::platform::set_env("CUDA_MODULE_LOADING", "LAZY");
    if (const auto code = find_model(argc, argv, adapter)) return *code;
    out = adapter != nullptr ? adapter->make_options() : std::make_unique<CommonOptions>();
    CommonOptions& o = *out;
    if (adapter != nullptr) {
        o.arch = adapter->arch();
        o.model_name = adapter->model_name();
        o.quality_flags = adapter->quality_flags();
        o.quality_window = adapter->quality_window();
    }
    if (const auto code = parse_options(argc, argv, o, adapter)) return *code;
    if (adapter != nullptr) {
        if (o.gpu_plan) o.plan_spec = adapter->planned_spec(o);
        pack_tokens(*adapter, o);
        // nothing would end a reply
        if (o.eos_ids.empty() && o.generates && !o.gpu_plan && (o.serve || (o.max_new > 0 && o.stop_eos))) {
            basalt::log::error("basalt", "this pack names no end-of-text token: give --eos-ids");
            return 2;
        }
        basalt::program::grammar::set_tool_form(adapter->tool_form());
    }
    // Batched serving reads a prompt on its own thread while verify windows run.  A lazy kernel load on that thread
    // synchronizes the context and can wait on a window kernel that waits on the load, so batched serving loads every
    // kernel up front.  An explicit CUDA_MODULE_LOADING wins.
    if (o.serve && o.batch >= 2 && !module_loading_set) basalt::platform::set_env("CUDA_MODULE_LOADING", "EAGER");
    return std::nullopt;
}

std::optional<int> prepare_model_process(const CommonOptions& o) {
    if (o.model.empty()) {
        basalt::log::error("basalt generate", "--model FILE.basalt is required");
        usage();
        return 2;
    }
    // a load that exhausts the RAM ends here instead of swapping the machine to a halt; stopped at READY
    if (o.serve) basalt::platform::start_memory_watchdog(1ull << 30);
    // the serve protocol's pipes carry bytes, so its newlines are the engine's own
    if (o.serve) basalt::platform::set_stdio_binary();
    return std::nullopt;
}

}  // namespace basalt::program::cli
