#include "basalt/models/deepseek4/compress.hpp"

#include "cache_rows.cuh"
#include "basalt/kernels/bf16_gemv.hpp"
#include "basalt/kernels/dense_ops.hpp"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
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

__device__ float block_max(float value, float* warp_maxima) {
    for (int offset = 16; offset > 0; offset >>= 1)
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    __syncthreads();
    if (lane == 0) warp_maxima[warp] = value;
    __syncthreads();
    value = threadIdx.x < warps ? warp_maxima[threadIdx.x] : 0.f;
    for (int offset = 16; offset > 0; offset >>= 1)
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    if (threadIdx.x == 0) warp_maxima[0] = value;
    __syncthreads();
    value = warp_maxima[0];
    __syncthreads();
    return value;
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
                                const float* rope_freq, float* state_kv, float* state_score, void* out_bytes,
                                int32_t* row_of, int mode, int row_bytes) {
    __shared__ float values[kHeadDim];
    __shared__ float warp_sums[32];
    __shared__ float fp8_scales[(kHeadDim - kRopeDim) / 64];
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
        // The reference casts pooled KV to the input dtype (BF16) before RMSNorm.
        values[c] = __bfloat162float(__float2bfloat16_rn(compressed));
        const float square_sum = block_sum(values[c] * values[c], warp_sums);
        float normalized = __bfloat162float(__float2bfloat16_rn(
            values[c] * rsqrtf(square_sum / kHeadDim + rms_eps) * norm_w[c]));
        values[c] = normalized;
        __syncthreads();
        if (c >= kHeadDim - kRopeDim) {
            const int pair = (c - (kHeadDim - kRopeDim)) >> 1;
            float x0 = values[kHeadDim - kRopeDim + pair * 2];
            float x1 = values[kHeadDim - kRopeDim + pair * 2 + 1];
            rotate_pair(x0, x1, (float) source_start, rope_freq[pair]);
            normalized = __bfloat162float(__float2bfloat16_rn((c & 1) ? x1 : x0));
        }
        values[c] = normalized;
        __syncthreads();
        // The reference rounds the cached row for its format: E2M1 codes with an E4M3 scale per 16, or plain fp16.
        uint8_t* row = static_cast<uint8_t*>(out_bytes) + (size_t) row_index * row_bytes;
        if (mode == 1) rows::put_fp4(normalized, c, kHeadDim, row);
        else if (mode == 0) static_cast<__half*>(static_cast<void*>(row))[c] = __float2half(normalized);
        else static_cast<float*>(static_cast<void*>(row))[c] = normalized;
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
                        const float* rope_freq, float* state_kv, float* state_score, void* out,
                        int32_t* row_of, CacheFormat fmt, void* stream) {
    if (ratio != expected_ratio(kind) || n_tokens < 1 || !kv || !score || !ape || !pos ||
        !norm_w || !rope_freq || !state_kv || !state_score || !out || !row_of || !(rms_eps > 0.0f))
        throw std::invalid_argument("deepseek4 compress_projected: invalid arguments");
    compress_kernel<<<1, kHeadDim, 0, (cudaStream_t) stream>>>(kind, ratio, n_tokens, kv, score, ape, pos, norm_w,
                                                                rms_eps, rope_freq, state_kv, state_score, out, row_of,
                                                                row_mode(fmt), comp_row_bytes(fmt));
    check_launch("deepseek4 compress_projected");
}

void project_and_compress(CompressKind kind, int ratio, int n_embd, int n_tokens, const float* x,
                          const uint16_t* kv_weight, const uint16_t* score_weight, const float* ape,
                          const int32_t* pos, const float* norm_w, float rms_eps, const float* rope_freq,
                          float* kv_projected, float* score_projected, float* state_kv, float* state_score,
                          void* out, int32_t* row_of, CacheFormat fmt, void* stream) {
    const int width = kind == CompressKind::Csa ? 2 * kHeadDim : kHeadDim;
    if (ratio != expected_ratio(kind) || n_embd < 8 || n_embd % 8 != 0 || n_tokens < 1 ||
        !x || !kv_weight || !score_weight || !ape || !pos || !norm_w || !rope_freq || !kv_projected ||
        !score_projected || !state_kv || !state_score || !out || !row_of || !stream || !(rms_eps > 0.0f))
        throw std::invalid_argument("deepseek4 project_and_compress: invalid arguments");
    if (n_tokens <= 8) {
        kernels::bf16_gemv_fp32_mmvf_multi(x, n_embd, kv_weight, kv_projected, width, n_embd, width, n_tokens, stream);
        kernels::bf16_gemv_fp32_mmvf_multi(x, n_embd, score_weight, score_projected, width, n_embd, width, n_tokens, stream);
    } else {
        kernels::bf16_rows_dot(x, kv_weight, kv_projected, n_tokens, width, n_embd, stream);
        kernels::bf16_rows_dot(x, score_weight, score_projected, n_tokens, width, n_embd, stream);
    }
    compress_projected(kind, ratio, n_tokens, kv_projected, score_projected, ape, pos, norm_w, rms_eps, rope_freq,
                       state_kv, state_score, out, row_of, fmt, stream);
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
