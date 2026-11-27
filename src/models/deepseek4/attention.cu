#include "basalt/models/deepseek4/attention.hpp"

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

__device__ __forceinline__ float pow2_ceil(float value) {
    const uint32_t bits = __float_as_uint(value);
    const int exponent = (int) ((bits >> 23) & 0xffu) - 127 + ((bits & 0x7fffffu) != 0u);
    return __uint_as_float((uint32_t) (exponent + 127) << 23);
}

__global__ void prepare_kv_kernel(const float* projected, const float* norm, float eps, const int32_t* positions,
                                  kernels::MqaRopeTable rope, float* out) {
    __shared__ float row[D];
    __shared__ float partial[Threads / 32];
    __shared__ float scales[QuantWidth / 64];
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

#pragma unroll
    for (int group = 0; group < QuantWidth / 64; ++group) {
        const float local = i >= group * 64 && i < (group + 1) * 64 ? fabsf(row[i]) : 0.f;
        const float amax = reduce_max(local, partial);
        if (i == 0) scales[group] = pow2_ceil(fmaxf(amax, 1e-4f) * (1.f / 448.f));
        __syncthreads();
    }

    float value = row[i];
    if (i < QuantWidth) {
        const float scale = scales[i / 64];
        const float quantized = fminf(448.f, fmaxf(-448.f, value / scale));
        value = bf16_round((float) __nv_fp8_e4m3(quantized) * scale);
    }
    out[base + i] = value;
}

__global__ void cache_window_kernel(float* ring, const float* rows, const int32_t* positions, int n_tokens) {
    const int token = blockIdx.x;
    const int position = positions[token];
    if (position < 0) return;
    const size_t dst = (size_t) (position % 128) * D;
    const size_t src = (size_t) token * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) ring[dst + i] = rows[src + i];
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

void cache_attention_window(float* ring, const float* rows, const int32_t* positions, int n_tokens, void* stream) {
    if (!ring || !rows || !positions || n_tokens < 1)
        throw std::invalid_argument("deepseek4 cache_attention_window: invalid arguments");
    cache_window_kernel<<<n_tokens, 256, 0, (cudaStream_t) stream>>>(ring, rows, positions, n_tokens);
    check_launch();
}

}  // namespace basalt::models::deepseek4
