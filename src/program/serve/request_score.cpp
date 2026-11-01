// ---- Jev: exact resident target scores and bounded suffix waves ----
#include "request.hpp"

namespace basalt::program::serve {

Flow Request::score() {
    if (cancelled || runtime_.halted()) {
        cancelled = true;
        finish = "cancel";
        return Flow::proceed;
    }
    const int32_t token = (int32_t) ids.back();
    int32_t ignored = 0;
    runtime_.drive.d.failed = false;
    if (!score_branches.empty()) {
        uint64_t bytes = 0;
        const bool ok = runtime_.ver.score_branches(score_branches, n - 1, runtime_.win_pool_fn,
                                                   runtime_.win_pool_user, bytes, (uint64_t) vram_floor_mib() << 20,
                                                   runtime_.err);
        const double native_ms = std::chrono::duration<double, std::milli>(Clock::now() - d0).count();
        if ((!ok && !runtime_.err.empty()) || runtime_.drive.d.failed) {
            std::printf("ERR Jev branch verify: %s\n", runtime_.err.c_str()); return Flow::error;
        }
        finish = "length";
        if (runtime_.halted()) { cancelled = true; finish = "cancel"; }
        else if (!ok) std::printf("JFALLBACK\n");
        else {
            std::printf("JWAVE %zu %llu\n", score_branches.size(), (unsigned long long) bytes);
            for (const auto& b : score_branches) {
                if (std::any_of(b.logits.begin(), b.logits.end(), [](float v) { return !std::isfinite(v); })) {
                    std::printf("ERR Jev: nonfinite branch logits\n"); return Flow::error;
                }
                std::printf("JLOGITS");
                for (float v : b.logits) std::printf(" %.9g", (double) v);
                std::printf("\n");
            }
        }
        std::printf("JTIME %.6f\n", native_ms);
        return Flow::proceed;
    }
    if (!runtime_.ver.run(1, &token, n - 1, runtime_.win_pool_fn, runtime_.win_pool_user, &ignored, runtime_.err) ||
        runtime_.drive.d.failed) {
        std::printf("ERR Jev verify: %s\n", runtime_.err.c_str());
        return Flow::error;
    }
    std::vector<float> logits((size_t) runtime_.n_vocab);
    if (!runtime_.ver.copy_score_logits(0, logits.data(), runtime_.err) ||
        std::any_of(score_ids.begin(), score_ids.end(), [&](int64_t id) { return !std::isfinite(logits[(size_t) id]); }) ||
        !runtime_.ver.commit(1, runtime_.err)) {
        std::printf("ERR Jev: target logits or commit failed\n");
        return Flow::error;
    }
    // The last token's recurrent/indexer/PLE state commits normally. Its MTP state is intentionally absent:
    // complete() keeps the pre-window checkpoints and marks this live tail unusable for continuation.
    consumed.push_back(token);
    p = n;
    finish = "length";
    const double native_ms = std::chrono::duration<double, std::milli>(Clock::now() - d0).count();
    if (!runtime_.halted()) {
        std::printf("JLOGITS");
        for (int64_t id : score_ids) std::printf(" %.9g", (double) logits[(size_t) id]);
        std::printf("\n");
    } else {
        cancelled = true;
        finish = "cancel";
    }
    std::printf("JTIME %.6f\n", native_ms);
    return Flow::proceed;
}

}  // namespace basalt::program::serve
// ---- end Jev ----
