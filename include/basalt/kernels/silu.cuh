// include/basalt/kernels/silu.cuh - SiLU's quotient g / (1 + exp(-g)) for a kernel epilogue that computes many of them.
// Device code only (no kernel): include from a .cu.
#pragma once

#include <cuda_runtime.h>

#include <cmath>

namespace basalt::kernels {

/// g / (1 + exp(-g)), bitwise the plain expression `g / (1.0f + __expf(-g))` for every float.
///
/// The plain expression compiles to the exponential as 2^(-g log2 e) (squared from half the exponent below 2^-126), then
/// an IEEE divide: a reciprocal refined once, the quotient corrected once by its remainder, a range check, and a called
/// slow path when the check fails.  The check and the call put a branch after every divide, so a row of them runs one
/// after another.  This is the same arithmetic with the three cases the check exists for selected instead (an infinite
/// divisor, a zero or an infinite dividend): no branch, and the compiler runs the rows' chains side by side.
/// tests/kernels/silu_parity.cu compares the two on all 2^32 inputs.
__device__ __forceinline__ float silu_quotient(float g) {
    float t = __fmul_rn(g, -1.4426950216293334961f);
    const bool low = t < -126.0f;
    t = low ? __fmul_rn(t, 0.5f) : t;
    float e;
    asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(t));
    e = low ? __fmul_rn(e, e) : e;
    const float d = __fadd_rn(e, 1.0f);
    float y;
    asm("rcp.approx.f32 %0, %1;" : "=f"(y) : "f"(d));
    y = __fmaf_rn(y, __fmaf_rn(-d, y, 1.0f), y);
    const float q0 = __fmaf_rn(g, y, 0.0f);
    float q = __fmaf_rn(y, __fmaf_rn(-d, q0, g), q0);
    q = d == INFINITY ? __fmul_rn(g, 0.0f) : q;
    return (g == 0.0f || g == INFINITY) ? g : q;
}

}  // namespace basalt::kernels
