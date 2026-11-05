// include/basalt/core/draft_cum.hpp - the cumulative draft-length rule.
//
// The MTP chain keeps draft j while the estimated chance that drafts 0..j are ALL accepted stays at or above the
// threshold: the product over i <= j of acc(p_i), where acc maps a draft probability to the observed acceptance rate of drafts in
// that probability bin.  A middling first draft is cheap to verify and still often right, so it is proposed; a long
// run of merely high probabilities stops once the joint chance decays.  The host (the window size, the draft loop)
// and the device (the conditional draft graph) evaluate the same products in the same order, so they agree exactly.
// Only what is proposed changes: acceptance and outputs stay exact.
#pragma once

#include "basalt/core/diag.hpp"

#include <cmath>

#if defined(__CUDACC__)
#define DRAFT_CUM_HD __host__ __device__
#else
#define DRAFT_CUM_HD
#endif

namespace basalt::core {

/// The acceptance rate of a sampled draft by its probability bin.
DRAFT_CUM_HD inline float draft_cum_acc(float p) {
    return p < 0.30f ? 0.12f : p < 0.50f ? 0.26f : p < 0.70f ? 0.43f : p < 0.80f ? 0.51f
         : p < 0.90f ? 0.59f : p < 0.95f ? 0.66f : p < 0.99f ? 0.79f : 0.96f;
}

/// Whether draft j stays in the window: prod_{i <= j} acc(probs[i]) >= th.
DRAFT_CUM_HD inline bool draft_keep(const float* probs, int j, float th) {
    float c = 1.0f;
    for (int i = 0; i <= j; ++i) c = c * draft_cum_acc(probs[i]);
    return c >= th;
}

/// The rule's base threshold: what a request starts from and the device graph's default.
constexpr float kDraftCumBase0 = 0.35f;

/// The CPU-served expert bytes per expert slot, in MB, set once when the expert tiers are final.  Each extra verified
/// token adds distinct experts the CPU must read from RAM, so a pack that leaves more bytes in RAM pays more per extra
/// token; the rule's threshold rises with it.  Fixed by the pack and its placement, never by timing.
inline float& draft_cum_cpu_mb() { static float v = 0.0f; return v; }
inline float draft_cum_scale() {
    constexpr float kLoMb = 1.1f, kHiMb = 3.5f, kMax = 1.5f;
    const float t = (draft_cum_cpu_mb() - kLoMb) / (kHiMb - kLoMb);
    return 1.0f + (kMax - 1.0f) * (t < 0.0f ? 0.0f : t > 1.0f ? 1.0f : t);
}
inline float draft_cum_base() { return kDraftCumBase0 * draft_cum_scale(); }

/// Per-request threshold.  The threshold falls from the base toward kLo as the
/// request's drafts are accepted more often than the bin table predicts: when they run well, longer windows pay.  The
/// observed and expected accepted counts decay per window; the ratio sets the threshold along a linear ramp.  The
/// window's value is fixed before its drafts are made and shared by the host rule and the device graph.
struct DraftCumAdapt {
    float base = draft_cum_base();   ///< the threshold while nothing is known
    double obs = 0.0, exp = 0.0;
    float cur = base;

    float th() const { return cur; }

    /// One verified window of the MTP chain: `n` drafts were offered with probabilities probs[0..n), `acc` accepted.
    void observe(const float* probs, int n, int acc) {
        if (n <= 0) return;
        double c = 1.0, e = 0.0;
        for (int j = 0; j < n; ++j) { c *= (double) draft_cum_acc(probs[j]); e += c; }
        observe_expected(e, acc);
    }
    /// The same for a drafter that states each draft's chance of acceptance itself.
    void observe_chances(const float* chances, int n, int acc) {
        if (n <= 0) return;
        double c = 1.0, e = 0.0;
        for (int j = 0; j < n; ++j) { c *= (double) chances[j]; e += c; }
        observe_expected(e, acc);
    }
    /// A window in which `e` drafts were expected accepted and `acc` were.
    void observe_expected(double e, int acc) {
        constexpr double kDecay = 0.97, kRlo = 0.9, kRhi = 1.1;
        const float kLo = 0.25f * draft_cum_scale();
        obs = obs * kDecay + (double) acc;
        exp = exp * kDecay + e;
        if (exp < 0.5) return;
        const double t = std::fmin(std::fmax((obs / exp - kRlo) / (kRhi - kRlo), 0.0), 1.0);
        cur = base + (float) t * (std::fmin(kLo, base) - base);
    }
};

}  // namespace basalt::core
