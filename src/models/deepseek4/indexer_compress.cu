#include "basalt/models/deepseek4/compress.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace basalt::models::deepseek4 {
namespace {

constexpr int D = kIndexerHeadDim;
constexpr int R = 64;
constexpr int Ratio = kIndexerRatio;
constexpr int Width = 2 * D;
constexpr int StateRows = 2 * Ratio;

__device__ __forceinline__ float bf16_rn(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

__device__ float block_sum(float value, float* partial) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) partial[warp] = value;
    __syncthreads();
    float sum = 0.f;
    if (threadIdx.x == 0) {
        for (int i = 0; i < blockDim.x / 32; ++i) sum += partial[i];
        partial[0] = sum;
    }
    __syncthreads();
    sum = partial[0];
    __syncthreads();
    return sum;
}

__device__ float block_max(float value, float* partial) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x / 32;
    __syncthreads();
    if (lane == 0) partial[warp] = value;
    __syncthreads();
    value = threadIdx.x < warps ? partial[threadIdx.x] : 0.f;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    if (threadIdx.x == 0) partial[0] = value;
    __syncthreads();
    value = partial[0];
    __syncthreads();
    return value;
}

__device__ __forceinline__ float pow2_ceil(float positive) {
    const uint32_t bits = __float_as_uint(positive);
    const int exponent = (int) ((bits >> 23) & 0xffu) - 127 + ((bits & 0x7fffffu) != 0u);
    return __uint_as_float((uint32_t) (exponent + 127) << 23);
}

__device__ __forceinline__ float fp4_e2m1_rn(float value) {
    constexpr float levels[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float magnitude = fabsf(value);
    int selected = 7;
#pragma unroll
    for (int i = 0; i < 7; ++i) {
        const float midpoint = (levels[i] + levels[i + 1]) * .5f;
        if (magnitude < midpoint || (magnitude == midpoint && (i & 1) == 0)) {
            selected = i;
            break;
        }
    }
    return copysignf(levels[selected], value);
}

__device__ __forceinline__ void rotate_pair(float& x0, float& x1, float position, float frequency) {
    float sine, cosine;
    sincosf(position * frequency, &sine, &cosine);
    const float y0 = x0 * cosine - x1 * sine;
    const float y1 = x0 * sine + x1 * cosine;
    x0 = y0;
    x1 = y1;
}

__global__ void indexer_compress_kernel(int n_tokens, const float* kv, const float* score, const float* ape,
                                       const int32_t* pos, const float* norm_w, float eps, const float* rope_freq,
                                       float* state_kv, float* state_score, float* out, int32_t* row_of) {
    __shared__ float values[D];
    __shared__ float hadamard_a[D];
    __shared__ float hadamard_b[D];
    __shared__ float partial[32];
    __shared__ float fp4_scales[D / 32];
    __shared__ int row_index;
    const int c = threadIdx.x;

    for (int t = 0; t < n_tokens; ++t) {
        const int p = pos[t];
        if (c == 0) row_index = -1;
        if (p < 0) {
            if (c == 0) row_of[t] = -1;
            __syncthreads();
            continue;
        }
        const int slot = p % StateRows;
        state_kv[(size_t) slot * Width + c] = kv[(size_t) t * Width + c];
        state_kv[(size_t) slot * Width + D + c] = kv[(size_t) t * Width + D + c];
        state_score[(size_t) slot * Width + c] = score[(size_t) t * Width + c] + ape[(size_t) (p % Ratio) * Width + c];
        state_score[(size_t) slot * Width + D + c] = score[(size_t) t * Width + D + c] + ape[(size_t) (p % Ratio) * Width + D + c];
        if ((p + 1) % Ratio) {
            if (c == 0) row_of[t] = -1;
            __syncthreads();
            continue;
        }
        const int row = p / Ratio;
        const int start = p + 1 - Ratio;
        if (c == 0) {
            row_index = row;
            row_of[t] = row;
        }
        __syncthreads();
        float maximum = -INFINITY;
        for (int j = 0; j < 2 * Ratio; ++j) {
            const int source_pos = j < Ratio ? start - Ratio + j : start + j - Ratio;
            const int lane = j < Ratio ? c : D + c;
            const float s = source_pos < 0 ? -INFINITY : state_score[(size_t) (source_pos % StateRows) * Width + lane];
            maximum = fmaxf(maximum, s);
        }
        float denominator = 0.f, numerator = 0.f;
        for (int j = 0; j < 2 * Ratio; ++j) {
            const int source_pos = j < Ratio ? start - Ratio + j : start + j - Ratio;
            if (source_pos < 0) continue;
            const int lane = j < Ratio ? c : D + c;
            const size_t index = (size_t) (source_pos % StateRows) * Width + lane;
            const float weight = expf(state_score[index] - maximum);
            denominator += weight;
            numerator += state_kv[index] * weight;
        }
        const float pooled = bf16_rn(denominator > 0.f ? numerator / denominator : 0.f);
        const float square_sum = block_sum(pooled * pooled, partial);
        float normalized = bf16_rn(pooled * rsqrtf(square_sum / D + eps) * norm_w[c]);
        values[c] = normalized;
        __syncthreads();
        if (c >= D - R) {
            const int pair = (c - (D - R)) >> 1;
            float x0 = values[D - R + pair * 2];
            float x1 = values[D - R + pair * 2 + 1];
            rotate_pair(x0, x1, (float) start, rope_freq[pair]);
            normalized = bf16_rn((c & 1) ? x1 : x0);
        }
        values[c] = normalized;
        __syncthreads();

        // Source Hadamard transform is unnormalized until the final 1/sqrt(D) factor.
        float* input = hadamard_a;
        float* output = hadamard_b;
        input[c] = values[c];
        __syncthreads();
        for (int stride = 1; stride < D; stride <<= 1) {
            const float own = input[c];
            const float other = input[c ^ stride];
            output[c] = bf16_rn((c & stride) ? other - own : own + other);
            __syncthreads();
            float* swap = input;
            input = output;
            output = swap;
        }
        const float rotated = bf16_rn(input[c] * rsqrtf((float) D));
        values[c] = rotated;
        __syncthreads();

        constexpr float kFp4MinAmax = 6.f * 0x1p-126f;
        for (int group = 0; group < D / 32; ++group) {
            const float local = c >= group * 32 && c < (group + 1) * 32 ? fabsf(values[c]) : 0.f;
            const float amax = fmaxf(block_max(local, partial), kFp4MinAmax);
            if (c == 0) fp4_scales[group] = pow2_ceil(amax * (1.f / 6.f));
            __syncthreads();
        }
        const float scale = fp4_scales[c / 32];
        const float dequantized = fp4_e2m1_rn(fminf(6.f, fmaxf(-6.f, values[c] / scale))) * scale;
        out[(size_t) row_index * D + c] = bf16_rn(dequantized);
        __syncthreads();
    }
}

__global__ void reset_indexer_state(float* kv, float* score, size_t count) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        kv[i] = 0.f;
        score[i] = -INFINITY;
    }
}

void check_launch(const char* name) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(std::string(name) + ": " + cudaGetErrorString(error));
}

}  // namespace

void indexer_compress_projected(int n_tokens, const float* kv, const float* score, const float* ape,
                                const int32_t* pos, const float* norm_w, float rms_eps, const float* rope_freq,
                                float* state_kv, float* state_score, float* out, int32_t* row_of, void* stream) {
    if (n_tokens < 1 || !kv || !score || !ape || !pos || !norm_w || !(rms_eps > 0.f) || !rope_freq ||
        !state_kv || !state_score || !out || !row_of)
        throw std::invalid_argument("deepseek4 indexer compressor: invalid arguments");
    indexer_compress_kernel<<<1, D, 0, (cudaStream_t) stream>>>(n_tokens, kv, score, ape, pos, norm_w, rms_eps,
                                                                 rope_freq, state_kv, state_score, out, row_of);
    check_launch("deepseek4 indexer compressor");
}

void reset_indexer_compress_state(float* state_kv, float* state_score, void* stream) {
    if (!state_kv || !state_score) throw std::invalid_argument("deepseek4 reset indexer compressor: invalid state");
    constexpr size_t count = (size_t) StateRows * Width;
    reset_indexer_state<<<(unsigned) ((count + 255) / 256), 256, 0, (cudaStream_t) stream>>>(state_kv, state_score, count);
    check_launch("deepseek4 reset indexer compressor");
}

}  // namespace basalt::models::deepseek4
