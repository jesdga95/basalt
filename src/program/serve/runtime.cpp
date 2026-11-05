// src/program/serve/runtime.cpp - lifecycle of the persistent engine.
#include "state.hpp"

namespace basalt::program::serve {

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
