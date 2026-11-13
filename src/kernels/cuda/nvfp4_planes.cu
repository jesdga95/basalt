// src/kernels/cuda/nvfp4_planes.cu - see include/basalt/kernels/nvfp4_planes.hpp.
#include "basalt/kernels/nvfp4_planes.hpp"

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

}  // namespace basalt::kernels::nvfp4p
