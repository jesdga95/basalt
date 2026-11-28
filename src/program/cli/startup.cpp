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

// Resolves the registered adapter for the pack at `path`. When `quiet` is true, errors leave `adapter` null without
// logging (used when `-h`/`--help` also appears in argv and may be a help request rather than a flag value).
std::optional<int> resolve_adapter(const std::string& path, const model::ModelAdapter*& adapter,
                                   std::string& model_name, std::string& community_quant, bool quiet = false) {
    basalt::core::ModelFile file;
    std::string err;
    if (!file.open(path, err) || !file.load_meta(err)) {
        if (quiet) return std::nullopt;
        basalt::log::error("basalt", "--model: %s", err.c_str());
        return 2;
    }
    const basalt::core::Json* kv = file.meta().get("gguf_kv");
    const basalt::core::Json* arch = kv ? kv->get("general.architecture") : nullptr;
    const std::string name = arch && arch->is_string() ? arch->s : std::string();
    adapter = model::find_adapter(name);
    if (adapter == nullptr) {
        if (quiet) return std::nullopt;
        basalt::log::error("basalt", "--model: %s", model::unavailable_architecture(name).c_str());
        return 2;
    }
    model_name = kv != nullptr ? adapter->model_name(*kv) : "";
    community_quant = adapter->community_quant(file.meta());
    return std::nullopt;
}

// The architecture named in the file --model gives; empty without one. An exit code when it cannot be used.
std::optional<int> find_model(int argc, char** argv, const model::ModelAdapter*& adapter,
                              std::string& model_name, std::string& community_quant) {
    std::string path;
    bool saw_help = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) {
            path = argv[++i];
        } else if (a == "--help" || a == "-h") {
            saw_help = true;
        }
    }
    if (path.empty()) return std::nullopt;
    return resolve_adapter(path, adapter, model_name, community_quant, saw_help);
}

// What the flags left unset: each id is the pack's own token.
void pack_tokens(const model::ModelAdapter& adapter, const basalt::core::ModelFile* file, const std::string& open_err,
                 CommonOptions& o) {
    std::string err = open_err;
    std::vector<uint8_t> text;
    basalt::core::Json vocab;
    if (file == nullptr || !basalt::core::read_range(file->range("tokenizer/vocab.json"), text, err) ||
        !basalt::core::parse_json((const char*) text.data(), text.size(), vocab, err)) {
        basalt::log::warn("basalt", "the pack's vocabulary cannot be read (%s): no end-of-turn or turn token", err.c_str());
        return;
    }
    // a name with one `*` matches the first token whose text has that start and end (a model whose special tokens carry a suffix
    // that differs between its files)
    auto id_of = [&](const std::string& name) -> int64_t {
        const size_t star = name.find('*');
        if (star == std::string::npos) {
            const basalt::core::Json* v = vocab.get(name);
            return v != nullptr ? (int64_t) v->u64() : -1;
        }
        const std::string head = name.substr(0, star), tail = name.substr(star + 1);
        for (const auto& [text, v] : vocab.members)
            if (text.size() >= head.size() + tail.size() && text.compare(0, head.size(), head) == 0 &&
                text.compare(text.size() - tail.size(), tail.size(), tail) == 0)
                return (int64_t) v.u64();
        return -1;
    };
    const model::TokenNames names = adapter.token_names();
    if (o.eos_ids.empty())
        for (const std::string& name : names.eos)
            if (const int64_t id = id_of(name); id >= 0) o.eos_ids.push_back(id);
    if (!o.turn_token_given) o.turn_token = id_of(names.turn);
    if (!names.reply_role.empty()) o.reply_role_token = id_of(names.reply_role);
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
    basalt::platform::install_crash_handler();
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
    std::string model_name, community_quant;
    if (const auto code = find_model(argc, argv, adapter, model_name, community_quant)) return *code;
    auto fill_adapter_meta = [&](CommonOptions& opts) {
        opts.arch = adapter->arch();
        opts.model_name = model_name;
        opts.community_kv = adapter->community_kv();
        opts.community_quant = community_quant;
        opts.quality_flags = adapter->quality_flags();
        opts.quality_window = adapter->quality_window();
    };
    out = adapter != nullptr ? adapter->make_options() : std::make_unique<CommonOptions>();
    if (adapter != nullptr) fill_adapter_meta(*out);
    if (const auto code = parse_options(argc, argv, *out, adapter)) return *code;
    // If another flag masked --model during the pre-scan, resolve now and re-parse into the adapter's options object.
    if (adapter == nullptr && !out->model.empty()) {
        if (const auto code = resolve_adapter(out->model, adapter, model_name, community_quant)) return *code;
        out = adapter->make_options();
        fill_adapter_meta(*out);
        if (const auto code = parse_options(argc, argv, *out, adapter)) return *code;
    }
    CommonOptions& o = *out;
    if (adapter != nullptr) {
        if (o.gpu_plan) o.plan_spec = adapter->planned_spec(o);
        basalt::core::ModelFile file;
        std::string ferr;
        const bool file_ok = !o.model.empty() && file.open(o.model, ferr) && file.load_meta(ferr);
        pack_tokens(*adapter, file_ok ? &file : nullptr, ferr, o);
        // nothing would end a reply
        if (o.eos_ids.empty() && o.generates && !o.gpu_plan && (o.serve || (o.max_new > 0 && o.stop_eos))) {
            basalt::log::error("basalt", "this pack names no end-of-text token: give --eos-ids");
            return 2;
        }
        basalt::program::grammar::set_tool_form(file_ok ? adapter->tool_form(file) : adapter->tool_form());
        basalt::program::grammar::set_reply_form(file_ok ? adapter->reply_form(file) : adapter->reply_form());
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
