// DeepSeek-V4.1's packed cache rows (layouts in attn_index.hpp): the writers round a row the way the model's reference
// does (act_quant, fp4_act_quant) and store its codes and group scales; the readers turn a code and its scale back into
// the rounded value, exactly. The roundings, the code helpers and the row layout follow Project Maya
// (https://github.com/mw00/project-maya, v1.1.17) as imported into Strata (src/dsv41/ds4_kernels.cu).
// MIT License
// Copyright (c) 2026 Peasant Smith and the Project Maya contributors
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace basalt::models::deepseek41::rows {

// ---- codes ----------------------------------------------------------------------------------------------------

// 2^ceil(log2(x)) for a positive normal x
__device__ __forceinline__ float pow2_ceil(float x) {
    const uint32_t b = __float_as_uint(x);
    const int e = (int) ((b >> 23) & 0xffu) - 127 + ((b & 0x7fffffu) != 0u ? 1 : 0);
    return __uint_as_float((uint32_t) (e + 127) << 23);
}

// the nearest E4M3 code, ties to even, saturated at +-448
__device__ __forceinline__ uint32_t e4m3_code(float v) { return (uint32_t) __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3); }

__device__ __forceinline__ float e4m3_value(uint32_t q) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t) q, __NV_E4M3)));
}

// the nearest E2M1 code (magnitudes 0 .5 1 1.5 2 3 4 6 as 0..7, bit 3 the sign, -0 kept), ties to even
__device__ __forceinline__ uint32_t e2m1_code(float v) {
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
__device__ __forceinline__ float e2m1_value14(uint32_t c) {
    return __half2float(__ushort_as_half((unsigned short) (((c & 8u) << 12) | ((c & 7u) << 9))));
}

// a UE8M0 scale byte's value: the byte is the fp32 exponent, 2^(b - 127)
__device__ __forceinline__ float ue8m0_value(uint32_t b) { return __uint_as_float(b << 23); }

// The largest magnitude over the `group` consecutive lanes of the warp that hold v (group a power of two up to 32).
__device__ __forceinline__ float group_amax(float v, int group) {
    float a = fabsf(v);
    for (int o = group / 2; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    return a;
}

// ---- writers: value c of a d-wide row, one per lane of a full warp, consecutive values on consecutive lanes ---------

// Window row: E4M3 with a power-of-two scale per 32. Codes at row[c], scales at row[d + c / 32].
__device__ __forceinline__ void put_fp8(float v, int c, int d, uint8_t* row) {
    const float s = pow2_ceil(fmaxf(group_amax(v, 32), 1e-4f) * (1.0f / 448.0f));
    row[c] = (uint8_t) e4m3_code(fminf(fmaxf(v / s, -448.0f), 448.0f));
    if ((c & 31) == 0) row[d + c / 32] = (uint8_t) (__float_as_uint(s) >> 23);
}

// E2M1 code pairs: the even value in the low nibble of byte c / 2
__device__ __forceinline__ void put_e2m1(uint32_t code, int c, uint8_t* row) {
    const uint32_t hi = __shfl_down_sync(0xffffffffu, code, 1);
    if ((c & 1) == 0) row[c / 2] = (uint8_t) (code | (hi << 4));
}

// Compressed row: E2M1 with an E4M3 scale per 16. Codes at row[c / 2], scales at row[d / 2 + c / 16].
__device__ __forceinline__ void put_fp4(float v, int c, int d, uint8_t* row) {
    const uint32_t sc = e4m3_code(fmaxf(group_amax(v, 16), 6.0f * 0.001953125f) / 6.0f);
    const float s = e4m3_value(sc);
    put_e2m1(e2m1_code(fminf(fmaxf(v / s, -6.0f), 6.0f)), c, row);
    if ((c & 15) == 0) row[d / 2 + c / 16] = (uint8_t) sc;
}

// The power-of-two scale per 32 of an E2M1 group (index keys and queries).
__device__ __forceinline__ float fp4p_scale(float v) {
    return pow2_ceil(fmaxf(group_amax(v, 32), 6.0f * 1.1754943508222875e-38f) * (1.0f / 6.0f));
}

// Index key row: E2M1 with a power-of-two scale per 32. Codes at row[c / 2], scales at row[d / 2 + c / 32].
__device__ __forceinline__ void put_fp4p(float v, int c, int d, uint8_t* row) {
    const float s = fp4p_scale(v);
    put_e2m1(e2m1_code(fminf(fmaxf(v / s, -6.0f), 6.0f)), c, row);
    if ((c & 31) == 0) row[d / 2 + c / 32] = (uint8_t) (__float_as_uint(s) >> 23);
}

// The value an index key's rounding gives, for the queries, which stay fp32: E2M1 times the power-of-two scale.
__device__ __forceinline__ float round_fp4p(float v) {
    const float s = fp4p_scale(v);
    return e2m1_value14(e2m1_code(fminf(fmaxf(v / s, -6.0f), 6.0f))) * 16384.0f * s;
}

// ---- readers: codes as loaded (consecutive values from the low bits up) and their group's scale byte ----------------

// E4M3 codes with a UE8M0 scale (window rows): n of 2 or 4 values.
template <int n>
__device__ __forceinline__ void fp8_values(uint32_t codes, uint32_t scale, float* k) {
    const float s = ue8m0_value(scale);
#pragma unroll
    for (int i = 0; i < n; i += 2) {
        const __half2 h = __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) ((codes >> (8 * i)) & 0xffffu), __NV_E4M3));
        const float2 f = __half22float2(h);
        k[i] = f.x * s;
        k[i + 1] = f.y * s;
    }
}

// E2M1 codes with an E4M3 scale (compressed rows): n of 2, 4 or 8 values. The fp16 of a code's value times 2^-14
// (e2m1_value14) has only a high byte, ((c & 8) << 4) | ((c & 7) << 1): the bytes of all eight are made at once and
// placed in pairs. Times 2^14, then times the scale, both exact in fp16 (at most 6 significant bits, between 2^-10
// and 2688).
template <int n>
__device__ __forceinline__ void fp4_values(uint32_t codes, uint32_t scale, float* k) {
    const __half2 s = __half2half2(__half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t) scale, __NV_E4M3)));
    const uint32_t lo = codes & 0x0f0f0f0fu, hi = (codes >> 4) & 0x0f0f0f0fu;   // even and odd values
    const uint32_t l = ((lo & 0x07070707u) << 1) | ((lo & 0x08080808u) << 4);
    const uint32_t h = ((hi & 0x07070707u) << 1) | ((hi & 0x08080808u) << 4);
    const uint32_t a = __byte_perm(l, h, 0x5140), b = __byte_perm(l, h, 0x7362);   // values 0..3, 4..7
#pragma unroll
    for (int p = 0; p < n / 2; ++p) {
        const uint32_t bits = __byte_perm(p < 2 ? a : b, 0u, (p & 1) ? 0x3424 : 0x1404);
        const float2 f = __half22float2(__hmul2(__hmul2(*reinterpret_cast<const __half2*>(&bits), __float2half2_rn(16384.0f)), s));
        k[2 * p] = f.x;
        k[2 * p + 1] = f.y;
    }
}

// E2M1 codes with a UE8M0 scale (index keys): n of up to 8 values.
template <int n>
__device__ __forceinline__ void fp4p_values(uint32_t codes, uint32_t scale, float* k) {
    const float s = ue8m0_value(scale) * 16384.0f;
#pragma unroll
    for (int i = 0; i < n; ++i) k[i] = e2m1_value14(codes >> (4 * i)) * s;
}

}  // namespace basalt::models::deepseek41::rows
