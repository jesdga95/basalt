// src/kernels/cuda/nvfp4_planes.cu - see include/basalt/kernels/nvfp4_planes.hpp.
#include "basalt/kernels/nvfp4_planes.hpp"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdlib>

#include "basalt/log.hpp"

namespace basalt::kernels::nvfp4p {
namespace {

// One thread per (row, 64-value block): 32 code bytes re-paired, 4 scale bytes to their tile slots.
__global__ void from_blocks_kernel(const uint8_t* __restrict__ src, int64_t rows, int64_t K, uint8_t* __restrict__ dst) {
    const int64_t nb = K / 64;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * nb) return;
    const int64_t r = i / nb, b = i - r * nb;
    const uint8_t* blk = src + i * 36;
    const uint8_t* qs = blk + 4;
    // value v of the block: sub-block s = v / 16, j = v % 16: low nibble of qs[8s + j] (j < 8) or high of qs[8s + j - 8]
    auto value = [&](int v) -> unsigned {
        const int s = v >> 4, j = v & 15;
        const unsigned q = qs[8 * s + (j & 7)];
        return j < 8 ? (q & 15u) : (q >> 4);
    };
    uint32_t out[8];
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        uint32_t word = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int byte = 4 * w + k;   // values 2 byte, 2 byte + 1
            word |= (value(2 * byte) | (value(2 * byte + 1) << 4)) << (8 * k);
        }
        out[w] = word;
    }
    uint4* cd = reinterpret_cast<uint4*>(dst + r * (K / 2) + b * 32);
    cd[0] = make_uint4(out[0], out[1], out[2], out[3]);
    cd[1] = make_uint4(out[4], out[5], out[6], out[7]);
    uint8_t* sc = dst + scale_offset(rows, K);
#pragma unroll
    for (int s = 0; s < 4; ++s) sc[scale_index(r, 4 * b + s, K)] = blk[s];
}

// The inverse: one thread per (row, 64-value block).
__global__ void to_blocks_kernel(const uint8_t* __restrict__ src, int64_t rows, int64_t K, uint8_t* __restrict__ dst) {
    const int64_t nb = K / 64;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * nb) return;
    const int64_t r = i / nb, b = i - r * nb;
    const uint8_t* codes = src + r * (K / 2) + b * 32;   // byte c: values 2c (low), 2c + 1 (high)
    auto value = [&](int v) -> unsigned { const unsigned q = codes[v >> 1]; return (v & 1) ? (q >> 4) : (q & 15u); };
    uint8_t* blk = dst + i * 36;
    const uint8_t* sc = src + scale_offset(rows, K);
#pragma unroll
    for (int s = 0; s < 4; ++s) blk[s] = sc[scale_index(r, 4 * b + s, K)];
#pragma unroll
    for (int s = 0; s < 4; ++s)
#pragma unroll
        for (int j = 0; j < 8; ++j) blk[4 + 8 * s + j] = (uint8_t) (value(16 * s + j) | (value(16 * s + j + 8) << 4));
}


// Q8_0 to NVFP4 planes: the largest |value| (as uint bits: they order like the floats when non-negative), then one thread a
// 16-value group.
__device__ __forceinline__ float q8_value(const uint8_t* blocks, int64_t i) {   // value i (row-major) of a Q8_0 matrix
    const uint8_t* blk = blocks + (i / 32) * 34;
    const float d = __half2float(*reinterpret_cast<const __half*>(blk));
    return d * (float) (int8_t) blk[2 + i % 32];
}
__global__ void q8_amax_kernel(const uint8_t* __restrict__ blocks, int64_t n, unsigned* __restrict__ amax) {
    float m = 0.0f;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        m = fmaxf(m, fabsf(q8_value(blocks, i)));
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) atomicMax(amax, __float_as_uint(m));
}
__global__ void q8_gs_kernel(const unsigned* __restrict__ amax, float* __restrict__ scale) {
    const float a = __uint_as_float(*amax);
    *scale = a > 0.0f ? a / (6.0f * 448.0f) : 1.0f;
}
__global__ void q8_to_planes_kernel(const uint8_t* __restrict__ blocks, int64_t rows, int64_t K, const float* __restrict__ scale,
                                    uint8_t* __restrict__ dst) {
    const int64_t ng = K / 16;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * ng) return;
    const int64_t r = i / ng, g = i - r * ng;
    const float gs = *scale;
    float v[16], m = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        v[j] = q8_value(blocks, r * K + g * 16 + j);
        m = fmaxf(m, fabsf(v[j]));
    }
    const __nv_fp8_storage_t sb = __nv_cvt_float_to_fp8(m / (6.0f * gs), __NV_SATFINITE, __NV_E4M3);
    const float sd = __half2float(__half(__nv_cvt_fp8_to_halfraw(sb, __NV_E4M3)));
    const float inv = sd > 0.0f ? 1.0f / (sd * gs) : 0.0f;
    // E2M1 to nearest, ties to the even code, saturating (this file builds for sm_120, without the FP4 converts)
    auto e2m1 = [](float x) -> uint32_t {
        const float a = fabsf(x);
        const uint32_t c = a <= 0.25f ? 0 : a < 0.75f ? 1 : a <= 1.25f ? 2 : a < 1.75f ? 3 : a <= 2.5f ? 4 : a < 3.5f ? 5 : a <= 5.0f ? 6 : 7;
        return c | (x < 0.0f && c ? 8u : 0u);
    };
    uint32_t w[2] = {0, 0};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t c = e2m1(v[2 * j] * inv) | (e2m1(v[2 * j + 1] * inv) << 4);   // value 2j low nibble, 2j + 1 high
        w[j >> 2] |= c << (8 * (j & 3));
    }
    *reinterpret_cast<uint2*>(dst + r * (K / 2) + g * 8) = make_uint2(w[0], w[1]);
    dst[scale_offset(rows, K) + scale_index(r, g, K)] = (uint8_t) sb;
}
}  // namespace

void to_blocks(const void* planes, int64_t rows, int64_t K, void* blocks, void* stream) {
    if (!fits(rows, K)) {
        basalt::log::error("basalt", "nvfp4 planes: a [%lld x %lld] matrix", (long long) rows, (long long) K);
        std::exit(1);
    }
    const int64_t n = rows * (K / 64);
    to_blocks_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>((const uint8_t*) planes, rows, K,
                                                                                      (uint8_t*) blocks);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "nvfp4 planes: %s", cudaGetErrorString(e));
        std::exit(1);
    }
}

void from_blocks(const void* blocks, int64_t rows, int64_t K, void* planes, void* stream) {
    if (!fits(rows, K)) {
        basalt::log::error("basalt", "nvfp4 planes: a [%lld x %lld] matrix", (long long) rows, (long long) K);
        std::exit(1);
    }
    const int64_t n = rows * (K / 64);
    from_blocks_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>((const uint8_t*) blocks, rows, K,
                                                                                        (uint8_t*) planes);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "nvfp4 planes: %s", cudaGetErrorString(e));
        std::exit(1);
    }
}

void from_q8(const void* q8_blocks, int64_t rows, int64_t K, void* planes, float* scale, void* stream) {
    if (!fits(rows, K) || K % 32 != 0) {
        basalt::log::error("basalt", "nvfp4p::from_q8: %lld x %lld", (long long) rows, (long long) K);
        std::exit(1);
    }
    const cudaStream_t s = (cudaStream_t) stream;
    unsigned* amax = reinterpret_cast<unsigned*>(scale);   // the scale's word holds the maximum first
    cudaMemsetAsync(amax, 0, sizeof(unsigned), s);
    q8_amax_kernel<<<1024, 256, 0, s>>>((const uint8_t*) q8_blocks, rows * K, amax);
    q8_gs_kernel<<<1, 1, 0, s>>>(amax, scale);
    const int64_t n = rows * (K / 16);
    q8_to_planes_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>((const uint8_t*) q8_blocks, rows, K, scale, (uint8_t*) planes);
}

}  // namespace basalt::kernels::nvfp4p
