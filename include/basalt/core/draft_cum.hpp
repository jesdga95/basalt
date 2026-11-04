// include/basalt/core/draft_cum.hpp - the cumulative draft-length rule.
//
// The MTP chain keeps draft j while the estimated chance that drafts 0..j are ALL accepted stays at or above the
// threshold: the product over i <= j of acc(p_i), where acc maps a draft probability to the observed acceptance rate of drafts in
// that probability bin.  Every verified row adds the experts its token routes to, most of them computed by the CPU from
// RAM, so a row costs far more than the draft step that made it: a draft is worth a row only while that joint chance is
// at least one half.  The threshold is a constant, so the window width depends on the drafts alone, never on timing or
// on how earlier windows fared.  The host (the window size, the draft loop) and the device (the conditional draft
// graph) evaluate the same products in the same order, so they agree exactly.  Only what is proposed changes:
// acceptance and outputs stay exact.
#pragma once

#include "basalt/core/diag.hpp"

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

/// The rule's threshold.
constexpr float kDraftCumTh = 0.50f;

/// The threshold every window uses: kDraftCumTh, or V from --exp draft_th=V (V > 0) to tune the level.
inline float draft_cum_th() {
    const float v = (float) exp_flag("draft_th", 0.0);
    return v > 0.0f ? v : kDraftCumTh;
}

}  // namespace basalt::core
