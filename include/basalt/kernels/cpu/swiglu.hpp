// include/basalt/kernels/cpu/swiglu.hpp - the SwiGLU of the CPU's expert rows: one expression for every row kernel.
#pragma once

#include <cmath>

namespace basalt::kernels::cpu {

/// Which SwiGLU a layer's experts take.  All off: silu(g) * u.
struct SwiGlu {
    float limit = 0.f;          ///< > 0: g capped above at it and u clamped to [-limit, limit] first (the limited SwiGLU)
    float gate_softcap = 0.f;   ///< > 0: the softcapped SwiGLU instead, as the card's grouped experts compute it:
    float up_softcap = 0.f;     ///< h = cg tanh(g / cg) * sigmoid(g) * cu tanh(u / cu), the sigmoid on the raw gate
    bool gelu = false;          ///< the GELU gate (tanh form) instead: h = gelu_tanh(g) * u, as the card's grouped experts
};

// static: every kernel's object keeps its own copy, built for that object's instruction set
static inline float swiglu(const SwiGlu& s, float g, float u) {
    if (s.gelu) return 0.5f * g * (1.f + std::tanh(0.7978845608028654f * (g + 0.044715f * g * g * g))) * u;
    if (s.gate_softcap > 0.f) {
        const float uc = s.up_softcap > 0.f ? s.up_softcap * std::tanh(u / s.up_softcap) : u;
        return (s.gate_softcap * std::tanh(g / s.gate_softcap)) * (1.f / (1.f + std::exp(-g))) * uc;
    }
    if (s.limit > 0.f) {
        g = std::fmin(g, s.limit);
        u = std::fmax(std::fmin(u, s.limit), -s.limit);
    }
    return (g / (1.f + std::exp(-g))) * u;
}

}  // namespace basalt::kernels::cpu
