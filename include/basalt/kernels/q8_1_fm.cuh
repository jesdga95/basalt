// include/basalt/kernels/q8_1_fm.cuh - native_quantize_q8_1's 32-value block (built with fast math) written with the
// instructions it compiles to, so a kernel built without fast math quantizes its own output to the same bits.
#pragma once

#include "basalt/kernels/q8_1_finite.hpp"

#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

namespace basalt::kernels {

/// The q8_1 block: fp16 scale and sum, then 32 codes.
struct Q81Fm {
    half2 ds;
    int8_t qs[32];
};

__device__ __forceinline__ float fm_max(float a, float b) { float y; asm("max.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y; }
__device__ __forceinline__ float fm_abs(float a) { float y; asm("abs.ftz.f32 %0, %1;" : "=f"(y) : "f"(a)); return y; }
__device__ __forceinline__ float fm_add(float a, float b) { float y; asm("add.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y; }
__device__ __forceinline__ float fm_div(float a, float b) { float y; asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y; }

/// The code of `xi` in a block of scale `d` (finite) and largest magnitude `amax`.
__device__ __forceinline__ int8_t q8_1_code_fm(float xi, float d, float amax) {
    if (amax == 0.0f) return 0;
    const float r = roundf(fm_div(xi, d));
    return (int8_t) (r > 127.0f ? 127.0f : r < -127.0f ? -127.0f : r);
}

/// Block `ib` of `y` from a full warp: lane l holds value l of the block.
__device__ __forceinline__ void q8_1_store_fm(float xi, Q81Fm* y, size_t ib, int lane) {
    float amax = fm_abs(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fm_max(amax, __shfl_xor_sync(0xffffffffu, amax, o));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum = fm_add(sum, __shfl_xor_sync(0xffffffffu, sum, o));
    const float d = q8_1_finite(fm_div(amax, 127.0f));
    y[ib].qs[lane] = q8_1_code_fm(xi, d, amax);
    if (lane == 0) y[ib].ds = q8_1_ds(d, sum);
}

}  // namespace basalt::kernels
