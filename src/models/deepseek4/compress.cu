#include "basalt/models/deepseek4/compress.hpp"

#include "cache_rows.cuh"
#include "basalt/kernels/bf16_gemv.hpp"
#include "basalt/kernels/dense_ops.hpp"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <mma.h>

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

namespace {

namespace wmma = nvcuda::wmma;
constexpr int kGemmM = 64, kGemmN = 64, kGemmK = 32, kGemmLd = kGemmK + 8, kGemmCLd = kGemmN + 4;

__global__ void __launch_bounds__(128) gemm_bf16_kernel(const uint16_t* __restrict__ w, const float* __restrict__ x,
                                                        int T, int in, int out, float* __restrict__ y) {
    __shared__ __align__(128) __nv_bfloat16 xh[kGemmM * kGemmLd];
    __shared__ __align__(128) __nv_bfloat16 xl[kGemmM * kGemmLd];
    __shared__ __align__(128) __nv_bfloat16 xm[kGemmM * kGemmLd];
    __shared__ __align__(128) __nv_bfloat16 ws[kGemmN * kGemmLd];
    __shared__ __align__(128) float cs[kGemmM * kGemmCLd];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int m0 = blockIdx.y * kGemmM, n0 = blockIdx.x * kGemmN;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) wmma::fill_fragment(acc[u], 0.f);
    for (int k0 = 0; k0 < in; k0 += kGemmK) {
        {
            const int row = tid >> 1, c0 = (tid & 1) * 16;
            const int m = m0 + row;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (m < T) v = *reinterpret_cast<const float4*>(x + (size_t) m * in + k0 + c0 + 4 * q);
                const float f[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const __nv_bfloat16 hi = __float2bfloat16(f[i]);
                    const float r1 = f[i] - __bfloat162float(hi);
                    const __nv_bfloat16 mid = __float2bfloat16(r1);
                    xh[row * kGemmLd + c0 + 4 * q + i] = hi;
                    xm[row * kGemmLd + c0 + 4 * q + i] = mid;
                    xl[row * kGemmLd + c0 + 4 * q + i] = __float2bfloat16(r1 - __bfloat162float(mid));
                }
            }
        }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int idx = tid + 128 * j, row = idx >> 2, c = (idx & 3) * 8;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (n0 + row < out) v = *reinterpret_cast<const uint4*>(w + (size_t) (n0 + row) * in + k0 + c);
            *reinterpret_cast<uint4*>(ws + row * kGemmLd + c) = v;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < kGemmK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fh, fm, fl;
            wmma::load_matrix_sync(fh, xh + 16 * warp * kGemmLd + kk, kGemmLd);
            wmma::load_matrix_sync(fm, xm + 16 * warp * kGemmLd + kk, kGemmLd);
            wmma::load_matrix_sync(fl, xl + 16 * warp * kGemmLd + kk, kGemmLd);
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fw;
                wmma::load_matrix_sync(fw, ws + 16 * u * kGemmLd + kk, kGemmLd);
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> part;
                wmma::fill_fragment(part, 0.f);
                wmma::mma_sync(part, fl, fw, part);
                wmma::mma_sync(part, fm, fw, part);
                wmma::mma_sync(part, fh, fw, part);
#pragma unroll
                for (int i = 0; i < part.num_elements; ++i) acc[u].x[i] += part.x[i];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int u = 0; u < 4; ++u)
        wmma::store_matrix_sync(cs + 16 * warp * kGemmCLd + 16 * u, acc[u], kGemmCLd, wmma::mem_row_major);
    __syncwarp();
    for (int r = 0; r < 16; ++r) {
        const int m = m0 + 16 * warp + r;
        if (m >= T) break;
        for (int c = lane; c < kGemmN; c += 32)
            if (n0 + c < out) y[(size_t) m * out + n0 + c] = cs[(16 * warp + r) * kGemmCLd + c];
    }
}

__global__ void compress_batch_kernel(CompressKind kind, int ratio, const float* kv, const float* score,
                                      const float* ape, const int32_t* pos, const float* norm_w, float rms_eps,
                                      const float* rope_freq, const float* state_kv, const float* state_score,
                                      void* out_bytes, int32_t* row_of, int mode, int row_bytes) {
    __shared__ float values[kHeadDim];
    __shared__ float warp_sums[32];
    const int t = blockIdx.x, c = threadIdx.x;
    const int p = pos[t];
    const bool close = p >= 0 && (p + 1) % ratio == 0;
    if (c == 0) row_of[t] = close ? p / ratio : -1;
    if (!close) return;

    const int width = kind == CompressKind::Csa ? 2 * kHeadDim : kHeadDim;
    const int state_rows = kind == CompressKind::Csa ? 2 * ratio : ratio;
    const int candidates = kind == CompressKind::Csa ? 2 * ratio : ratio;
    const int source_start = p + 1 - ratio;

    float max_score = -INFINITY;
    for (int j = 0; j < candidates; ++j) {
        int source_pos, lane;
        if (kind == CompressKind::Csa) {
            const bool previous = j < ratio;
            source_pos = previous ? source_start - ratio + j : source_start + (j - ratio);
            lane = previous ? c : kHeadDim + c;
        } else {
            source_pos = source_start + j;
            lane = c;
        }
        float s = -INFINITY;
        if (source_pos >= 0) {
            const int tj = t - (p - source_pos);
            s = tj >= 0 ? score[(size_t) tj * width + lane] + ape[(size_t) (source_pos % ratio) * width + lane]
                        : state_score[(size_t) (source_pos % state_rows) * width + lane];
        }
        max_score = fmaxf(max_score, s);
    }

    float denominator = 0.0f, numerator = 0.0f;
    if (isfinite(max_score)) {
        for (int j = 0; j < candidates; ++j) {
            int source_pos, lane;
            if (kind == CompressKind::Csa) {
                const bool previous = j < ratio;
                source_pos = previous ? source_start - ratio + j : source_start + (j - ratio);
                lane = previous ? c : kHeadDim + c;
            } else {
                source_pos = source_start + j;
                lane = c;
            }
            if (source_pos < 0) continue;
            const int tj = t - (p - source_pos);
            const float kv_v = tj >= 0 ? kv[(size_t) tj * width + lane]
                                       : state_kv[(size_t) (source_pos % state_rows) * width + lane];
            const float sc_v = tj >= 0 ? score[(size_t) tj * width + lane] + ape[(size_t) (source_pos % ratio) * width + lane]
                                       : state_score[(size_t) (source_pos % state_rows) * width + lane];
            const float weight = expf(sc_v - max_score);
            denominator += weight;
            numerator += kv_v * weight;
        }
    }

    const float compressed = denominator > 0.0f ? numerator / denominator : 0.0f;
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
    uint8_t* row = static_cast<uint8_t*>(out_bytes) + (size_t) (p / ratio) * row_bytes;
    if (mode == 1) rows::put_fp4(normalized, c, kHeadDim, row);
    else if (mode == 0) static_cast<__half*>(static_cast<void*>(row))[c] = __float2half(normalized);
    else static_cast<float*>(static_cast<void*>(row))[c] = normalized;
}

__global__ void compress_state_kernel(CompressKind kind, int ratio, int n_tokens, const float* kv, const float* score,
                                      const float* ape, const int32_t* pos, float* state_kv, float* state_score) {
    const int c = threadIdx.x;
    const int width = kind == CompressKind::Csa ? 2 * kHeadDim : kHeadDim;
    const int state_rows = kind == CompressKind::Csa ? 2 * ratio : ratio;
    for (int d = state_rows - 1; d >= 0; --d) {
        const int tj = n_tokens - 1 - d;
        if (tj < 0) continue;
        const int p = pos[tj];
        if (p < 0) continue;
        const int slot = p % state_rows;
        state_kv[(size_t) slot * width + c] = kv[(size_t) tj * width + c];
        state_score[(size_t) slot * width + c] = score[(size_t) tj * width + c] + ape[(size_t) (p % ratio) * width + c];
        if (kind == CompressKind::Csa) {
            state_kv[(size_t) slot * width + kHeadDim + c] = kv[(size_t) tj * width + kHeadDim + c];
            state_score[(size_t) slot * width + kHeadDim + c] =
                score[(size_t) tj * width + kHeadDim + c] + ape[(size_t) (p % ratio) * width + kHeadDim + c];
        }
    }
}

}  // namespace

void gemm_bf16(const uint16_t* w, const float* x, int n_tokens, int in, int out, float* y, void* stream) {
    if (n_tokens < 1 || in % kGemmK) throw std::invalid_argument("deepseek4 gemm_bf16: bad shape");
    gemm_bf16_kernel<<<dim3((out + kGemmN - 1) / kGemmN, (n_tokens + kGemmM - 1) / kGemmM), 128, 0,
                       (cudaStream_t) stream>>>(w, x, n_tokens, in, out, y);
    check_launch("deepseek4 gemm_bf16");
}

void compress_batch_projected(CompressKind kind, int ratio, int n_tokens, const float* kv, const float* score,
                              const float* ape, const int32_t* pos, const float* norm_w, float rms_eps,
                              const float* rope_freq, float* state_kv, float* state_score, void* out,
                              int32_t* row_of, CacheFormat fmt, void* stream) {
    if (ratio != expected_ratio(kind) || n_tokens < 1 || n_tokens > 65535 || !kv || !score || !ape || !pos ||
        !norm_w || !rope_freq || !state_kv || !state_score || !out || !row_of || !(rms_eps > 0.0f))
        throw std::invalid_argument("deepseek4 compress_batch_projected: invalid arguments");
    cudaStream_t s = (cudaStream_t) stream;
    compress_batch_kernel<<<n_tokens, kHeadDim, 0, s>>>(kind, ratio, kv, score, ape, pos, norm_w, rms_eps, rope_freq,
                                                        state_kv, state_score, out, row_of, row_mode(fmt),
                                                        comp_row_bytes(fmt));
    check_launch("deepseek4 compress_batch_projected");
    compress_state_kernel<<<1, kHeadDim, 0, s>>>(kind, ratio, n_tokens, kv, score, ape, pos, state_kv, state_score);
    check_launch("deepseek4 compress_state");
}

}  // namespace basalt::models::deepseek4
