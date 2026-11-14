// include/basalt/models/qwen4exp/kernels/fp4_bits.cuh - the NVFP4 rounding of kv_q4.hpp's `--kv nvfp4`, as device
// inline: an E4M3 scale per group of 16 and E2M1 codes, the same rounding the DeepSeek pack's compressed cache rows
// use (src/models/deepseek41/kernels/cache_rows.cuh, the model's own `fp4_act_quant`), so a value stored here lands on
// the same 4-bit grid the pack was quantized on. Duplicated from that file's static helpers because model layering
// keeps one model's kernels out of another's (tools/check_model_layering.sh); the thresholds are copied verbatim.
#pragma once

#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace basalt::kernels {

// the nearest E4M3 code, ties to even, saturated at +-448
__device__ __forceinline__ uint32_t fp4_e4m3_code(float v) {
    return (uint32_t) __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ float fp4_e4m3_value(uint32_t q) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t) q, __NV_E4M3)));
}

// the nearest E2M1 code (magnitudes 0 .5 1 1.5 2 3 4 6 as 0..7, bit 3 the sign, -0 kept), ties to even
__device__ __forceinline__ uint32_t fp4_e2m1_code(float v) {
    const float a = fabsf(v);
    uint32_t m;
    if (a <= 0.25f) m = 0;
    else if (a < 0.75f) m = 1;
    else if (a <= 1.25f) m = 2;
    else if (a < 1.75f) m = 3;
    else if (a <= 2.5f) m = 4;
    else if (a < 3.5f) m = 5;
    else if (a <= 5.0f) m = 6;
    else m = 7;
    return m | ((__float_as_uint(v) >> 31) << 3);
}

// an E2M1 code's value times 2^-14: the code's sign, exponent and mantissa bit are an fp16's bits 15 and 11..9, so the
// conversion is exact, subnormals too; times a scale times 2^14 it is the value times the scale, exactly
__device__ __forceinline__ float fp4_e2m1_value14(uint32_t c) {
    return __half2float(__ushort_as_half((unsigned short) (((c & 8u) << 12) | ((c & 7u) << 9))));
}

// The largest magnitude over the 16 consecutive lanes of the warp that hold v (lane groups of 16).
__device__ __forceinline__ float fp4_amax16(float v) {
    float a = fabsf(v);
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o, 16));
    return a;
}

// The E4M3 scale byte of a group whose largest |value| is amax, and the value it decodes to (the 6 is E2M1's largest
// magnitude; the floor keeps an all-zero group's scale finite, as the DeepSeek rows' does).
__device__ __forceinline__ uint32_t fp4_group_scale(float amax) {
    return fp4_e4m3_code(fmaxf(amax, 6.0f * 0.001953125f) / 6.0f);
}

// The code v quantizes to against a decoded group scale s.
__device__ __forceinline__ uint32_t fp4_group_code(float v, float s) {
    return fp4_e2m1_code(fminf(fmaxf(v / s, -6.0f), 6.0f));
}

}  // namespace basalt::kernels
