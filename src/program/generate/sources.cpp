#include "context.hpp"

namespace basalt::program::generate {

int sources(Context& context) {
    auto& o = *context.o;
    auto& expert_files = *context.expert_files;
    auto& err = *context.err;
    auto& g = *context.g;

    // ---- the CPU expert pool
    //
    // The experts are loaded into a RESIDENT ARENA by default.  The mmap path is kept behind `--mmap-experts`; it is
    // the low-RAM mode, not the fast one.
    //
    // `FileExpertSource` maps the pack's file, and mapped file pages are the first thing the OS reclaims; the engine's
    // rate then depends on whether the page cache happens to hold the experts file.  The arena is anonymous memory the
    // engine owns, so the pool's rate does not depend on the page cache.
    //
    // It is pinned (registered with CUDA) when the driver allows it, and the GPU's DMA out of it feeds the cache
    // fills and the PCIe share of the misses.  When `cudaHostRegister` fails the arena falls back to 4 KB anonymous
    // pages, reported by `note()`.  A stale CUDA error left set by a failed `cudaHostRegister` would be read later by
    // `gr_read`'s launch check: see the note in `pinned.cu`.
    basalt::core::FileExpertSource src;
    {   // the card, and whether this build has code for it (a binary built for other GPUs fails at its first kernel
        // otherwise, after the whole expert arena has loaded) - before the arena starts loading
        int dev = 0;
        cudaDeviceProp p{};
        const bool named = cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&p, dev) == cudaSuccess;
        if (!named) cudaGetLastError();
        const char* name = named && p.name[0] ? p.name : "(an unnamed GPU)";
        basalt::log::info("basalt generate", "GPU %d: %s, compute capability %d.%d", dev, name,
                     p.major, p.minor);
        {   // a build whose libcudart is older than its headers (a CUDA 13 kit with a dangling libcudart.so that
            // CMake resolved to the system's CUDA 12 one) reads cudaDeviceProp shifted - silently, and slowly
            int rt = 0;
            if (cudaRuntimeGetVersion(&rt) == cudaSuccess && rt / 1000 != CUDART_VERSION / 1000)
                basalt::log::warn("basalt generate", "this engine was compiled with CUDA %d.%d headers but "
                                     "loaded a CUDA %d.%d runtime (libcudart): GPU properties can read wrong and some "
                                     "kernels go unused. Rebuild it against one toolkit (cmake -DCUDAToolkit_ROOT=<the "
                                     "toolkit>, with its libcudart.so present)",
                             CUDART_VERSION / 1000, CUDART_VERSION % 1000 / 10, rt / 1000, rt % 1000 / 10);
        }
        const std::string e = basalt::core::device_code_error();
        if (!e.empty()) {
            basalt::log::error("basalt generate", "this engine has no code for %s (sm_%d%d): %s - rebuild it for this "
                                 "card (build it for this GPU's architecture)", name, p.major, p.minor, e.c_str());
            return 1;
        }
    }
    basalt::core::ArenaExpertSource arena_src;
    basalt::core::ExpertSource* srcp = nullptr;
    if (o.mmap_experts) {
        // The low-RAM mode: the experts come from the model file through the OS cache instead of a pinned copy in
        // RAM, for a machine whose GPU holds most of them but whose RAM cannot hold them all.
        if (!src.open(expert_files.experts, g.n_layers, g.n_expert, err)) {
            return fatal_start_error(err);
        }
        basalt::log::info("basalt generate", "experts via mmap (--mmap-experts)");
        srcp = &src;
    } else {
        const uint64_t pin_limit = 0;   // the whole arena is pinned
        if (!arena_src.open(expert_files, g.n_layers, g.n_expert, /*threads=*/6, err, pin_limit,
                            o.shared_expert_arena)) {
            return fatal_start_error(err);
        }
        if (!arena_src.ram_warning().empty())   // said before the load's numbers, which it explains
            basalt::log::warn("basalt generate", "%s", arena_src.ram_warning().c_str());
        basalt::log::info("basalt generate", "expert arena: %s", arena_src.note().c_str());
        basalt::log::info("basalt generate", "loaded %.2f GiB at %.2f GiB/s",
                     (double) basalt::kernels::cpu::expert_layout().total / (1024.0 * 1024 * 1024),
                     arena_src.load_gib_per_second());
        srcp = &arena_src;
    }
    // the pool is built once the start-up picks are known (below, with the resident RAM plan)
    std::optional<basalt::kernels::cpu::ExpertPool> pool_store;
    uint64_t ram_available_at_plan = 0;   // the available RAM the start-up plan was sized from
    uint64_t auto_resident_bytes = 0;   // the start-up pick for the RAM copy (used when no budget is given)

    context.src = &src;
    context.arena_src = &arena_src;
    context.srcp = &srcp;
    context.pool_store = &pool_store;
    context.ram_available_at_plan = &ram_available_at_plan;
    context.auto_resident_bytes = &auto_resident_bytes;
    return cache_size(context);
}

}
