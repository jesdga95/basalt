#include "basalt/models/deepseek4/compress.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace basalt::models::deepseek4 {
namespace {

constexpr int kHeadDim = 512;
constexpr int kRopeDim = 64;
constexpr int kCsaRatio = 4;
constexpr int kHcaRatio = 128;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    return value;
}

__device__ float block_sum(float value, float* warp_sums) {
    value = warp_sum(value);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    __syncthreads();
    if (lane == 0) warp_sums[warp] = value;
    __syncthreads();
    float sum = 0.0f;
    for (int i = 0; i < warps; ++i) sum += warp_sums[i];
    return sum;
}

__device__ __forceinline__ void rotate_pair(float& x0, float& x1, float position, float frequency) {
    float sine, cosine;
    sincosf(position * frequency, &sine, &cosine);
    const float y0 = x0 * cosine - x1 * sine;
    const float y1 = x0 * sine + x1 * cosine;
    x0 = y0;
    x1 = y1;
}

__global__ void compress_kernel(CompressKind kind, int ratio, int n_tokens, const float* kv, const float* score,
                                const float* ape, const int32_t* pos, const float* norm_w, float rms_eps,
                                const float* rope_freq, float* state_kv, float* state_score, float* out,
                                int32_t* row_of) {
    __shared__ float values[kHeadDim];
    __shared__ float warp_sums[32];
    __shared__ int32_t row_index;
    const int c = threadIdx.x;
    const int width = kind == CompressKind::Csa ? 2 * kHeadDim : kHeadDim;
    const int state_rows = kind == CompressKind::Csa ? 2 * ratio : ratio;

    for (int t = 0; t < n_tokens; ++t) {
        const int p = pos[t];
        if (c == 0) row_index = -1;
        if (p < 0) {
            if (c == 0) row_of[t] = -1;
            __syncthreads();
            continue;
        }

        const int slot = p % state_rows;
        state_kv[(size_t) slot * width + c] = kv[(size_t) t * width + c];
        state_score[(size_t) slot * width + c] = score[(size_t) t * width + c] +
            ape[(size_t) (p % ratio) * width + c];
        if (kind == CompressKind::Csa) {
            state_kv[(size_t) slot * width + kHeadDim + c] = kv[(size_t) t * width + kHeadDim + c];
            state_score[(size_t) slot * width + kHeadDim + c] = score[(size_t) t * width + kHeadDim + c] +
                ape[(size_t) (p % ratio) * width + kHeadDim + c];
        }
        if ((p + 1) % ratio != 0) {
            if (c == 0) row_of[t] = -1;
            __syncthreads();
            continue;
        }

        if (c == 0) {
            row_index = p / ratio;
            row_of[t] = row_index;
        }
        __syncthreads();
        const int source_start = p + 1 - ratio;
        float max_score = -INFINITY;
        const int candidates = kind == CompressKind::Csa ? 2 * ratio : ratio;
        for (int j = 0; j < candidates; ++j) {
            int source_pos;
            int lane;
            if (kind == CompressKind::Csa) {
                const bool previous = j < ratio;
                source_pos = previous ? source_start - ratio + j : source_start + (j - ratio);
                lane = previous ? c : kHeadDim + c;
            } else {
                source_pos = source_start + j;
                lane = c;
            }
            const float s = source_pos < 0 ? -INFINITY :
                state_score[(size_t) (source_pos % state_rows) * width + lane];
            max_score = fmaxf(max_score, s);
        }
        float denominator = 0.0f;
        float numerator = 0.0f;
        if (isfinite(max_score)) {
            for (int j = 0; j < candidates; ++j) {
                int source_pos;
                int lane;
                if (kind == CompressKind::Csa) {
                    const bool previous = j < ratio;
                    source_pos = previous ? source_start - ratio + j : source_start + (j - ratio);
                    lane = previous ? c : kHeadDim + c;
                } else {
                    source_pos = source_start + j;
                    lane = c;
                }
                if (source_pos < 0) continue;
                const size_t index = (size_t) (source_pos % state_rows) * width + lane;
                const float weight = expf(state_score[index] - max_score);
                denominator += weight;
                numerator += state_kv[index] * weight;
            }
        }
        const float compressed = denominator > 0.0f ? numerator / denominator : 0.0f;
        values[c] = compressed;
        const float square_sum = block_sum(compressed * compressed, warp_sums);
        float normalized = compressed * rsqrtf(square_sum / kHeadDim + rms_eps) * norm_w[c];
        values[c] = normalized;
        __syncthreads();
        if (c >= kHeadDim - kRopeDim) {
            const int pair = (c - (kHeadDim - kRopeDim)) >> 1;
            float x0 = values[kHeadDim - kRopeDim + pair * 2];
            float x1 = values[kHeadDim - kRopeDim + pair * 2 + 1];
            rotate_pair(x0, x1, (float) source_start, rope_freq[pair]);
            normalized = (c & 1) ? x1 : x0;
        }
        out[(size_t) row_index * kHeadDim + c] = normalized;
        __syncthreads();
    }
}

__global__ void clear_state(float* kv, float* score, size_t elements) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < elements) {
        kv[i] = 0.0f;
        score[i] = -INFINITY;
    }
}

int expected_ratio(CompressKind kind) {
    return kind == CompressKind::Csa ? kCsaRatio : kHcaRatio;
}

void check_launch(const char* name) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(std::string(name) + ": " + cudaGetErrorString(error));
}

}  // namespace

void compress_projected(CompressKind kind, int ratio, int n_tokens, const float* kv, const float* score,
                        const float* ape, const int32_t* pos, const float* norm_w, float rms_eps,
                        const float* rope_freq, float* state_kv, float* state_score, float* out,
                        int32_t* row_of, void* stream) {
    if (ratio != expected_ratio(kind) || n_tokens < 1 || n_tokens > 8 || !kv || !score || !ape || !pos ||
        !norm_w || !rope_freq || !state_kv || !state_score || !out || !row_of || !(rms_eps > 0.0f))
        throw std::invalid_argument("deepseek4 compress_projected: invalid arguments");
    compress_kernel<<<1, kHeadDim, 0, (cudaStream_t) stream>>>(kind, ratio, n_tokens, kv, score, ape, pos, norm_w,
                                                                rms_eps, rope_freq, state_kv, state_score, out, row_of);
    check_launch("deepseek4 compress_projected");
}

void reset_compress_state(CompressKind kind, int ratio, float* state_kv, float* state_score, void* stream) {
    if (ratio != expected_ratio(kind) || !state_kv || !state_score)
        throw std::invalid_argument("deepseek4 reset_compress_state: invalid arguments");
    const int width = kind == CompressKind::Csa ? 2 * kHeadDim : kHeadDim;
    const size_t elements = (size_t) (kind == CompressKind::Csa ? 2 * ratio : ratio) * width;
    clear_state<<<(unsigned) ((elements + 255) / 256), 256, 0, (cudaStream_t) stream>>>(state_kv, state_score, elements);
    check_launch("deepseek4 reset_compress_state");
}

}  // namespace basalt::models::deepseek4
