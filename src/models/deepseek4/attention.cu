#include "basalt/models/deepseek4/attention.hpp"

#include "cache_rows.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace basalt::models::deepseek4 {
namespace {

constexpr int D = 512;
constexpr int Rope = 64;
constexpr int QuantWidth = D - Rope;
constexpr int Threads = 512;

__device__ float reduce_sum(float value, float* partial) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) value += __shfl_xor_sync(0xffffffffu, value, offset);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) partial[warp] = value;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.f;
#pragma unroll
        for (int i = 0; i < Threads / 32; ++i) total += partial[i];
        partial[0] = total;
    }
    __syncthreads();
    return partial[0];
}

__device__ float reduce_max(float value, float* partial) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) partial[warp] = value;
    __syncthreads();
    if (threadIdx.x < Threads / 32) value = partial[threadIdx.x];
    else value = 0.f;
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    if (threadIdx.x == 0) partial[0] = value;
    __syncthreads();
    return partial[0];
}

__device__ __forceinline__ float bf16_round(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

// The KV boundary's values: BF16-rounded projection, RMSNorm, forward RoPE, BF16-rounded. The cache writer rounds them
// for its format (the reference's act_quant for packed rows); `out` keeps the unrounded values for staging.
__global__ void prepare_kv_kernel(const float* projected, const float* norm, float eps, const int32_t* positions,
                                  kernels::MqaRopeTable rope, float* out) {
    __shared__ float row[D];
    __shared__ float partial[Threads / 32];
    const int token = blockIdx.x;
    const int i = threadIdx.x;
    const size_t base = (size_t) token * D;

    row[i] = bf16_round(projected[base + i]);
    const float square_sum = reduce_sum(row[i] * row[i], partial);
    row[i] = bf16_round(row[i] * rsqrtf(square_sum / D + eps) * norm[i]);
    __syncthreads();

    if (i < Rope / 2) {
        const int d = QuantWidth + i * 2;
        float sine, cosine;
        sincosf((float) positions[token] * rope.step[i], &sine, &cosine);
        cosine *= rope.mscale;
        sine *= rope.mscale;
        const float a = row[d];
        const float b = row[d + 1];
        row[d] = bf16_round(a * cosine - b * sine);
        row[d + 1] = bf16_round(a * sine + b * cosine);
    }
    __syncthreads();
    out[base + i] = row[i];
}

__global__ void cache_window_kernel(uint8_t* ring, const float* rows, const int32_t* positions, int n_tokens,
                                    int row_bytes, int mode) {
    const int token = blockIdx.x;
    const int position = positions[token];
    if (position < 0) return;
    uint8_t* row = ring + (size_t) (position % 128) * row_bytes;
    const float* src = rows + (size_t) token * D;
    if (mode == 1) {
        rows::put_fp8(src[threadIdx.x], threadIdx.x, D, row);
    } else if (mode == 0) {
        static_cast<__half*>(static_cast<void*>(row))[threadIdx.x] = __float2half(src[threadIdx.x]);
    } else {
        static_cast<float*>(static_cast<void*>(row))[threadIdx.x] = src[threadIdx.x];
    }
}

void check_launch() {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(std::string("deepseek4 prepare_attention_kv: ") + cudaGetErrorString(error));
}

}  // namespace

void prepare_attention_kv(const float* projected, const float* norm, float rms_eps, const int32_t* positions,
                          const kernels::MqaRopeTable& rope, int n_tokens, float* out, void* stream) {
    if (!projected || !norm || !positions || !out || n_tokens < 1 ||
        !(rms_eps > 0.f) || !std::isfinite(rms_eps) || !std::isfinite(rope.mscale))
        throw std::invalid_argument("deepseek4 prepare_attention_kv: invalid arguments");
    prepare_kv_kernel<<<n_tokens, Threads, 0, (cudaStream_t) stream>>>(projected, norm, rms_eps, positions, rope, out);
    check_launch();
}

void cache_attention_window(void* ring, const float* rows, const int32_t* positions, int n_tokens, CacheFormat fmt,
                            void* stream) {
    if (!ring || !rows || !positions || n_tokens < 1)
        throw std::invalid_argument("deepseek4 cache_attention_window: invalid arguments");
    cache_window_kernel<<<n_tokens, D, 0, (cudaStream_t) stream>>>(static_cast<uint8_t*>(ring), rows, positions,
                                                                   n_tokens, win_row_bytes(fmt),
                                                                   row_mode(fmt));
    check_launch();
}

void cache_attention_window_f32(float* ring, const float* rows, const int32_t* positions, int n_tokens, void* stream) {
    if (!ring || !rows || !positions || n_tokens < 1)
        throw std::invalid_argument("deepseek4 cache_attention_window_f32: invalid arguments");
    cache_window_kernel<<<n_tokens, D, 0, (cudaStream_t) stream>>>(reinterpret_cast<uint8_t*>(ring), rows, positions,
                                                                   n_tokens, D * 4, 2);
    check_launch();
}

namespace {

__global__ void window_copy_kernel(const uint4* __restrict__ ring, uint4* __restrict__ stage, int stage_cap,
                                   int64_t p_lo) {
    const int words = blockDim.x;
    const int64_t p = p_lo + blockIdx.x;
    stage[(size_t) (p % stage_cap) * words + threadIdx.x] = ring[(size_t) (p % 128) * words + threadIdx.x];
}

__global__ void kv_window_stage_kernel(const float* __restrict__ projected, const float* __restrict__ norm, float eps,
                                       const int32_t* __restrict__ positions, kernels::MqaRopeTable rope,
                                       uint8_t* __restrict__ stage, int stage_cap, uint8_t* __restrict__ ring,
                                       int ring_from, int row_bytes, int mode) {
    __shared__ float row[D];
    __shared__ float partial[Threads / 32];
    const int token = blockIdx.x;
    const int i = threadIdx.x;
    const size_t base = (size_t) token * D;

    row[i] = bf16_round(projected[base + i]);
    const float square_sum = reduce_sum(row[i] * row[i], partial);
    row[i] = bf16_round(row[i] * rsqrtf(square_sum / D + eps) * norm[i]);
    __syncthreads();

    if (i < Rope / 2) {
        const int d = QuantWidth + i * 2;
        float sine, cosine;
        sincosf((float) positions[token] * rope.step[i], &sine, &cosine);
        cosine *= rope.mscale;
        sine *= rope.mscale;
        const float a = row[d];
        const float b = row[d + 1];
        row[d] = bf16_round(a * cosine - b * sine);
        row[d + 1] = bf16_round(a * sine + b * cosine);
    }
    __syncthreads();

    const int position = positions[token];
    if (position < 0) return;
    const float val = row[i];
    uint8_t* srow = stage + (size_t) (position % stage_cap) * row_bytes;
    if (mode == 1) rows::put_fp8(val, i, D, srow);
    else if (mode == 0) static_cast<__half*>(static_cast<void*>(srow))[i] = __float2half(val);
    else static_cast<float*>(static_cast<void*>(srow))[i] = val;

    if (token >= ring_from) {
        uint8_t* rrow = ring + (size_t) (position % 128) * row_bytes;
        if (mode == 1) rows::put_fp8(val, i, D, rrow);
        else if (mode == 0) static_cast<__half*>(static_cast<void*>(rrow))[i] = __float2half(val);
        else static_cast<float*>(static_cast<void*>(rrow))[i] = val;
    }
}

}  // namespace

void window_stage_copy(const void* ring, void* stage, int stage_cap, int64_t p_lo, int count, CacheFormat fmt,
                       void* stream) {
    if (count <= 0) return;
    window_copy_kernel<<<count, win_row_bytes(fmt) / 16, 0, (cudaStream_t) stream>>>(
        static_cast<const uint4*>(ring), static_cast<uint4*>(stage), stage_cap, p_lo);
    check_launch();
}

void kv_window_stage(const float* projected, const float* norm, float rms_eps, const int32_t* positions,
                     const kernels::MqaRopeTable& rope, int n_tokens, void* stage, int stage_cap, void* ring,
                     int ring_from, CacheFormat fmt, void* stream) {
    if (!projected || !norm || !positions || !stage || !ring || n_tokens < 1 ||
        !(rms_eps > 0.f) || !std::isfinite(rms_eps) || !std::isfinite(rope.mscale))
        throw std::invalid_argument("deepseek4 kv_window_stage: invalid arguments");
    kv_window_stage_kernel<<<n_tokens, Threads, 0, (cudaStream_t) stream>>>(
        projected, norm, rms_eps, positions, rope, static_cast<uint8_t*>(stage), stage_cap,
        static_cast<uint8_t*>(ring), ring_from, win_row_bytes(fmt), row_mode(fmt));
    check_launch();
}

}  // namespace basalt::models::deepseek4
