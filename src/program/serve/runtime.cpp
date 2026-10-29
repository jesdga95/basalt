// src/program/serve/runtime.cpp - lifecycle of the persistent engine.
#include "state.hpp"

namespace basalt::program::serve {

int Runtime::execute() {
    if (const int code = init_prompt()) return code;
    if (const int code = init_verify()) return code;
    if (const int code = init_conversations()) return code;
    if (const int code = init_tiers()) return code;
    if (const int code = init_protocol()) return code;
    if (const int code = announce()) return code;
    if (const int code = init_memory()) return code;
    if (const int code = capture_windows()) return code;
    if (const int code = init_batch()) return code;
    return run_requests();
}

int run(Context& context) {
    Runtime runtime(context);
    return runtime.execute();
}

int Runtime::run_requests() {
    for (;;) {
        const Flow flow = dispatch();
        if (flow == Flow::error) return 1;
        if (flow == Flow::quit) return shutdown();
    }
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
    save_profile("exit");   // QUIT, or the server closed stdin
    return 0;
}

}  // namespace basalt::program::serve
