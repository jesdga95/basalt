// src/models/hy4flash/dfly_kernels.cu - kernels of the DFly2 drafter (include/basalt/models/hy4flash/dfly_kernels.hpp).
#include "basalt/models/hy4flash/dfly_kernels.hpp"
#include "basalt/models/hy4flash/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <stdexcept>

namespace basalt::models::hy4flash {
namespace {

constexpr int kMaxRows = 8;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

// Block (h, part): 128 latent columns of one head. A thread takes a column pair (a warp reads 128 contiguous bytes of a key
// row) over one of kAbsorbSlices runs of the head's 192 key rows, so the block keeps that many loads in flight; the runs'
// partial sums are added in shared memory in run order (the same bits on every call).
constexpr int kAbsorbSlices = 8;
__global__ void __launch_bounds__(64 * kAbsorbSlices) absorb_k_kernel(const uint16_t* __restrict__ kvb, const float* __restrict__ q,
                                                                     int q_stride, int q_head, int rows, float* __restrict__ q_abs, int heads) {
    __shared__ float part[kAbsorbSlices][kMaxRows][128];
    const int h = blockIdx.x, x = threadIdx.x, y = threadIdx.y;
    const int c0 = blockIdx.y * 128;
    float a0[kMaxRows], a1[kMaxRows];
#pragma unroll
    for (int r = 0; r < kMaxRows; ++r) a0[r] = a1[r] = 0.f;
    const uint32_t* w = reinterpret_cast<const uint32_t*>(kvb + (size_t) h * kDflyKvbHead * kDflyRank + c0) + x;
    constexpr int kRun = kDflyNope / kAbsorbSlices;
    for (int d = y * kRun; d < (y + 1) * kRun; ++d) {
        const uint32_t pair = w[(size_t) d * (kDflyRank / 2)];
        const float w0 = __uint_as_float(pair << 16), w1 = __uint_as_float(pair & 0xffff0000u);
#pragma unroll
        for (int r = 0; r < kMaxRows; ++r)
            if (r < rows) {
                const float qv = q[(size_t) r * q_stride + (size_t) h * q_head + d];
                a0[r] = fmaf(qv, w0, a0[r]);
                a1[r] = fmaf(qv, w1, a1[r]);
            }
    }
#pragma unroll
    for (int r = 0; r < kMaxRows; ++r)
        if (r < rows) {
            part[y][r][2 * x] = a0[r];
            part[y][r][2 * x + 1] = a1[r];
        }
    __syncthreads();
    for (int i = y * 64 + x; i < rows * 128; i += 64 * kAbsorbSlices) {
        const int r = i / 128, c = i % 128;
        float t = 0.f;
#pragma unroll
        for (int k = 0; k < kAbsorbSlices; ++k) t += part[k][r][c];
        q_abs[((size_t) r * heads + h) * kDflyRank + c0 + c] = t;
    }
}

// Block (h, part): 8 value rows of one head, a warp each; the rows' latents sit in shared memory.
__global__ void expand_v_kernel(const uint16_t* __restrict__ kvb, const float* __restrict__ olat, int rows, int heads,
                                float* __restrict__ out) {
    __shared__ float s[kMaxRows][kDflyRank];
    const int h = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int i = threadIdx.x; i < rows * kDflyRank; i += blockDim.x) {
        const int r = i / kDflyRank, c = i % kDflyRank;
        s[r][c] = olat[((size_t) r * heads + h) * kDflyRank + c];
    }
    __syncthreads();
    const int v = blockIdx.y * 8 + warp;
    const uint32_t* w = reinterpret_cast<const uint32_t*>(kvb + ((size_t) h * kDflyKvbHead + kDflyNope + v) * kDflyRank);
    float acc[kMaxRows];
#pragma unroll
    for (int r = 0; r < kMaxRows; ++r) acc[r] = 0.f;
    for (int k = 0; k < kDflyRank / 64; ++k) {
        const uint32_t pair = w[k * 32 + lane];
        const float w0 = __uint_as_float(pair << 16), w1 = __uint_as_float(pair & 0xffff0000u);
        const int c = k * 64 + lane * 2;
#pragma unroll
        for (int r = 0; r < kMaxRows; ++r)
            if (r < rows) acc[r] = fmaf(w1, s[r][c + 1], fmaf(w0, s[r][c], acc[r]));
    }
#pragma unroll
    for (int r = 0; r < kMaxRows; ++r) {
        const float t = warp_sum(acc[r]);
        if (r < rows && lane == 0) out[((size_t) r * heads + h) * kDflyV + v] = t;
    }
}

__global__ void gather_kernel(const uint16_t* __restrict__ cache, const int32_t* __restrict__ cells, int width, uint16_t* __restrict__ out) {
    const int j = blockIdx.x;
    const uint16_t* src = cache + (size_t) (cells ? cells[j] : j) * width;
    for (int c = threadIdx.x; c < width; c += blockDim.x) out[(size_t) j * width + c] = src[c];
}

// An FP8 row: 512 E4M3 codes, four fp32 scales (one per 128 codes), 64 BF16 rope values.
__global__ void gather_fp8_kernel(const uint8_t* __restrict__ cache, const int32_t* __restrict__ cells, uint16_t* __restrict__ out) {
    const int j = blockIdx.x;
    const uint8_t* src = cache + (size_t) (cells ? cells[j] : j) * kFp8RowBytes;
    const float* scale = reinterpret_cast<const float*>(src + kDflyRank);
    const uint16_t* rope = reinterpret_cast<const uint16_t*>(src + kDflyRank + 16);
    for (int c = threadIdx.x; c < kDflyRowWidth; c += blockDim.x) {
        uint16_t v;
        if (c < kDflyRank) {
            const float f = __half2float(__half(__nv_cvt_fp8_to_halfraw(src[c], __NV_E4M3))) * scale[c >> 7];
            v = __half_as_ushort(__float2half_rn(f));   // exact: the value is an fp16 one
        } else {
            v = __half_as_ushort(__float2half_rn(__uint_as_float((uint32_t) rope[c - kDflyRank] << 16)));
        }
        out[(size_t) j * kDflyRowWidth + c] = v;
    }
}

__global__ void gate_add_kernel(const float* __restrict__ base, const float* __restrict__ fused, const float* __restrict__ gate, int n,
                                float* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = base[i] + gate[i] * fused[i];
}

}  // namespace

void dfly_absorb_k(const uint16_t* kvb, const float* q, int q_stride, int q_head, int rows, int heads, float* q_abs, void* stream) {
    if (rows < 1 || rows > kMaxRows) throw std::invalid_argument("dfly_absorb_k: rows");
    absorb_k_kernel<<<dim3(heads, kDflyRank / 128), dim3(64, kAbsorbSlices), 0, (cudaStream_t) stream>>>(kvb, q, q_stride, q_head, rows, q_abs, heads);
    if (!stream) cudaStreamSynchronize(nullptr);
}

void dfly_expand_v(const uint16_t* kvb, const float* olat, int rows, int heads, float* out, void* stream) {
    if (rows < 1 || rows > kMaxRows) throw std::invalid_argument("dfly_expand_v: rows");
    expand_v_kernel<<<dim3(heads, kDflyV / 8), 256, 0, (cudaStream_t) stream>>>(kvb, olat, rows, heads, out);
    if (!stream) cudaStreamSynchronize(nullptr);
}

void dfly_gather_rows(const uint16_t* cache, const int32_t* cells, int n, int width, uint16_t* out, void* stream) {
    if (n < 1) return;
    gather_kernel<<<n, 192, 0, (cudaStream_t) stream>>>(cache, cells, width, out);
    if (!stream) cudaStreamSynchronize(nullptr);
}

void dfly_gather_rows_fp8(const uint8_t* cache, const int32_t* cells, int n, uint16_t* out, void* stream) {
    if (n < 1) return;
    gather_fp8_kernel<<<n, 192, 0, (cudaStream_t) stream>>>(cache, cells, out);
    if (!stream) cudaStreamSynchronize(nullptr);
}

void dfly_gate_add(const float* base, const float* fused, const float* gate, int n, float* out, void* stream) {
    gate_add_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(base, fused, gate, n, out);
    if (!stream) cudaStreamSynchronize(nullptr);
}

}  // namespace basalt::models::hy4flash
