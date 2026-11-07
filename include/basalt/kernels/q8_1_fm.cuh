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

/// Block `ib` of `y` from an aligned group of 8 lanes: lane l holds values 4 (l & 7) .. 4 (l & 7) + 3 in v. The sum
/// adds in the order of the warp-wide butterfly above (value bits 4, 3, 2 across the lanes, then bits 1 and 0).
__device__ __forceinline__ void q8_1_store4_fm(float4 v, Q81Fm* y, size_t ib, int lane) {
    const unsigned mask = 0xffu << (lane & 24);
    float amax = fm_max(fm_max(fm_abs(v.x), fm_abs(v.y)), fm_max(fm_abs(v.z), fm_abs(v.w)));
    float s0 = v.x, s1 = v.y, s2 = v.z, s3 = v.w;
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) {
        amax = fm_max(amax, __shfl_xor_sync(mask, amax, o));
        s0 = fm_add(s0, __shfl_xor_sync(mask, s0, o));
        s1 = fm_add(s1, __shfl_xor_sync(mask, s1, o));
        s2 = fm_add(s2, __shfl_xor_sync(mask, s2, o));
        s3 = fm_add(s3, __shfl_xor_sync(mask, s3, o));
    }
    const float sum = fm_add(fm_add(s0, s2), fm_add(s1, s3));
    const float d = q8_1_finite(fm_div(amax, 127.0f));
    const char4 q = make_char4(q8_1_code_fm(v.x, d, amax), q8_1_code_fm(v.y, d, amax), q8_1_code_fm(v.z, d, amax),
                               q8_1_code_fm(v.w, d, amax));
    *reinterpret_cast<char4*>(&y[ib].qs[4 * (lane & 7)]) = q;
    if ((lane & 7) == 0) y[ib].ds = q8_1_ds(d, sum);
}

}  // namespace basalt::kernels
