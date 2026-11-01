#include "startup.hpp"
#include "arguments.hpp"
#include "help.hpp"
#include "serve.hpp"

#include "basalt/log.hpp"
#include "basalt/platform/memory.hpp"
#include "basalt/platform/process.hpp"
#include "basalt/platform/system.hpp"
#include "basalt/program/options.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace basalt::program::cli {

std::optional<int> start(int argc, char** argv, Options& o, std::string& ppl_base, std::string& ppl_out) {
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
    if (const auto code = parse_options(argc, argv, o, ppl_base, ppl_out)) return *code;
    // Batched serving reads a prompt on its own thread while verify windows run.  A lazy kernel load on that thread
    // synchronizes the context and can wait on a window kernel that waits on the load, so batched serving loads every
    // kernel up front.  An explicit CUDA_MODULE_LOADING wins.
    if (o.serve && o.batch >= 2 && !module_loading_set) basalt::platform::set_env("CUDA_MODULE_LOADING", "EAGER");
    return std::nullopt;
}

std::optional<int> prepare_model_process(const Options& o) {
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
