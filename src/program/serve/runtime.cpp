// src/program/serve/runtime.cpp - lifecycle of the persistent engine.
#include "state.hpp"

namespace basalt::program::serve {

void Runtime::start_guard(uint64_t file_bytes, uint64_t prompt_file_bytes) {
    basalt::platform::stop_memory_watchdog();
    guard_floor = basalt::platform::ram_guard_floor(basalt::platform::ram_margin(), file_bytes, prompt_file_bytes);
    if (!o.no_ram_guard) basalt::platform::start_ram_guard(guard_floor);
    disk_floor = guard_floor + (1ull << 30);
    // what a stop's error compares against, to tell the engine's own growth from other programs'
    ready_avail = basalt::platform::available_memory();
    ready_rss = (uint64_t) basalt::platform::process_memory().rss_mib << 20;
    shed_drop = 0;
}

void Runtime::shed_if_low() {
    if (basalt::platform::ram_guard_level() == basalt::platform::RamLevel::Ok) return;
    const uint64_t drop = basalt::platform::ram_guard_drops();
    if (drop == shed_drop) return;
    shed_drop = drop;
    if (conversations) {
        const size_t parked = conversations->size();
        const size_t freed = conversations->clear();
        size_t to_disk = 0;
        for (auto& sc : conversations->take_spilled())
            if (disk.enabled() && !disk_has(sc.live.ids, sc.live.imgs, sc.checkpoints, sc.cvec) &&
                disk.put(std::move(sc), kDiskQueueCap))
                ++to_disk;
        if (freed > 0)
            basalt::log::info("basalt serve", "RAM guard: %zu parked conversations (%zu MiB) leave RAM, %zu of them "
                              "to disk", parked, freed >> 20, to_disk);
    }
    if (const size_t pooled = basalt::core::CheckpointBytes::allocator_type::release_pooled(); pooled > 0)
        basalt::log::info("basalt serve", "RAM guard: freed %zu MiB of checkpoint blocks kept for reuse",
                          pooled >> 20);
    const uint64_t before = basalt::platform::available_memory();
    basalt::platform::release_free_heap();
    const uint64_t after = basalt::platform::available_memory();
    basalt::log::info("basalt serve", "RAM guard: free heap returned to the OS, %lld MiB free now (%+lld MiB)",
                      (long long) (after >> 20), ((long long) after - (long long) before) / (1ll << 20));
}

int Runtime::execute() {
    if (const int code = init_before_input()) return code;
    if (const int code = init_protocol()) return code;
    if (const int code = announce()) return code;
    if (const int code = init_after_input()) return code;
    return run_requests();
}

int Runtime::run_requests() {
    for (;;) {
        const Flow flow = dispatch();
        if (flow == Flow::error) {
            if (basalt::core::gpu_left_running().load()) {   // the teardown would wait for the stuck window forever
                basalt::log::error("basalt serve", "the GPU did not finish a released window: ending without the CUDA teardown");
                std::fflush(stdout);
                std::_Exit(1);
            }
            return 1;
        }
        if (flow == Flow::quit) return shutdown();
    }
}

// Whether the disk tier needs no new copy of a conversation: it is short (below the disk's minimum, or below
// --conversation-cache-min-tokens when that is higher), or a stored prefix of it reaches within
// kDiskStaleTokens of the chain's deepest checkpoint (the point its next turn goes on from; a restore
// from that copy reads the rest again).  Each turn of a growing conversation is not written again in full.
bool Runtime::disk_has(const std::vector<int32_t>& ids, const std::vector<ImgKey>& imgs, const std::vector<ConvCheckpoint>& cks, bool cvec) {
    if ((int64_t) ids.size() < std::max<int64_t>(disk_min_tokens(), o.conversation_cache_min_tokens)) return true;
    int64_t deepest = 0;
    for (const ConvCheckpoint& c : cks) deepest = std::max<int64_t>(deepest, (int64_t) c.ids.size());
    if (deepest <= 0) deepest = (int64_t) ids.size();
    const auto m = disk.best(ids.data(), ids.size(), imgs, cvec);
    return m.tokens > 0 && m.tokens + kDiskStaleTokens >= deepest;
}

int Runtime::shutdown() {
    // the conversations this process holds go to the disk tier before it ends
    if (disk.enabled()) {
        const auto t0 = Clock::now();
        const size_t written = spill_to_disk(true, SIZE_MAX);
        const auto st = disk.stats();
        basalt::log::info("basalt serve", "conversation cache: %zu conversations written to disk at exit in %.0f ms; "
                          "on disk %zu (%.2f GiB); since the start written %zu (%.2f GiB, %.2f GiB with the shared "
                          "files they reuse), read back %zu (%.2f GiB), evicted %zu, dropped %zu", written,
                          std::chrono::duration<double, std::milli>(Clock::now() - t0).count(), st.files,
                          (double) st.bytes / 1073741824.0, st.written, (double) st.written_bytes / 1073741824.0,
                          (double) st.whole_bytes / 1073741824.0, st.restored, (double) st.read_bytes / 1073741824.0,
                          st.evicted, st.dropped);
    }
    at_exit();
    return 0;
}

}  // namespace basalt::program::serve
