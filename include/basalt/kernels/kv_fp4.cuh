// include/basalt/kernels/kv_fp4.cuh - NVFP4 K/V rows (KvInt8::fp4): an E4M3 scale per 16 values and E2M1 codes, and the
// normalized 256-point Hadamard transform the keys are turned by.  Device inlines only.
#pragma once

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace basalt::kernels {

// the nearest E4M3 code, ties to even, saturated at +-448
__device__ __forceinline__ uint32_t kv4_e4m3_code(float v) {
    return (uint32_t) __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ float kv4_e4m3_value(uint32_t q) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t) q, __NV_E4M3)));
}

// the nearest E2M1 code (magnitudes 0 .5 1 1.5 2 3 4 6 as 0..7, bit 3 the sign), ties to even
__device__ __forceinline__ uint32_t kv4_e2m1_code(float v) {
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

// an E2M1 code's value times 2^-14: sign, exponent and mantissa bit land on an fp16's bits 15 and 11..9 (exact)
__device__ __forceinline__ float kv4_e2m1_value14(uint32_t c) {
    return __half2float(__ushort_as_half((unsigned short) (((c & 8u) << 12) | ((c & 7u) << 9))));
}

// The largest magnitude over the 16 consecutive lanes that hold v (lane groups of 16).
__device__ __forceinline__ float kv4_amax16(float v) {
    float a = fabsf(v);
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o, 16));
    return a;
}

// The E4M3 scale byte of a group whose largest |value| is amax (the floor keeps an all-zero group's scale finite).
__device__ __forceinline__ uint32_t kv4_group_scale(float amax) {
    return kv4_e4m3_code(fmaxf(amax, 6.0f * 0.001953125f) / 6.0f);
}

// The code v rounds to against a decoded group scale s.
__device__ __forceinline__ uint32_t kv4_group_code(float v, float s) {
    return kv4_e2m1_code(fminf(fmaxf(v / s, -6.0f), 6.0f));
}

// Eight values from four code bytes (value 2i in byte i's low nibble) and their group's scale byte, as fp16: each value is
// code x scale, exact in fp32 before the one rounding to fp16.
__device__ __forceinline__ uint4 kv4_decode8(uint32_t codes, uint32_t scale_byte) {
    const float s = kv4_e4m3_value(scale_byte) * 16384.0f;
    __half2 h[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const uint32_t b = (codes >> (8 * e)) & 0xffu;
        h[e] = __floats2half2_rn(kv4_e2m1_value14(b & 15u) * s, kv4_e2m1_value14(b >> 4) * s);
    }
    return *reinterpret_cast<uint4*>(h);
}

// The normalized Hadamard transform of a block's 256 values, one per thread, through shared memory `xs` (256 floats).
// Every thread of the block calls it; it returns the thread's transformed value.  Orthonormal and its own inverse.
__device__ __forceinline__ float kv4_hadamard256(float y, float* xs, int d) {
    __syncthreads();   // xs may still be read by the caller's previous step
    xs[d] = y;
    __syncthreads();
#pragma unroll
    for (int h = 1; h < 256; h <<= 1) {
        const float a = xs[d], b = xs[d ^ h];
        __syncthreads();
        xs[d] = (d & h) ? b - a : a + b;
        __syncthreads();
    }
    return xs[d] * 0.0625f;
}

}  // namespace basalt::kernels
