// src/models/inkling2/kernels/window_plan.hpp - the plan a deep INT8 decode window of one slab count takes (attn.cu): the
// register-staged form with the most blocks a multiprocessor, the one with the second V tile on a tie, where that fits more
// blocks than the shared rings of the slab count; the rings when it does not, or when the register forms could not be
// probed (a call under capture before the first probe, or an optional probe that ran out of memory).
#pragma once

namespace basalt::models::inkling2 {

enum class WindowPlan { Rings, Registers, RegistersPingPong };

/// `probed`: the register forms' residencies are known; `plain`, `ping_pong`: their blocks a multiprocessor (0: does not
/// fit); `rings`: the most the shared rings of the slab count fit.
inline WindowPlan window_plan(bool probed, int plain, int ping_pong, int rings) {
    if (!probed) return WindowPlan::Rings;
    const int best = ping_pong >= plain ? ping_pong : plain;
    if (best <= rings) return WindowPlan::Rings;
    return ping_pong >= plain ? WindowPlan::RegistersPingPong : WindowPlan::Registers;
}

}  // namespace basalt::models::inkling2
