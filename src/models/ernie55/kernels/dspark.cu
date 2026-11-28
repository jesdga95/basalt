// src/models/ernie55/kernels/dspark.cu - ERNIE-5.5 DSpark speculative drafter CUDA kernels:
//   - mhc_stream_mean: averages 4 mHC residual streams for target feature tap
//   - dspark_kv_ctx_store: RMSNorm + plain 8M interleaved RoPE -> FP16 context ring
//   - dspark_swa_block_attn: bidirectional block + sliding-window context attention with per-head attn_sink
//   - dspark_markov_multimax_argmax: fused (base + markov_bias) -> multimax -> GPU argmax
//   - dspark_confidence: sigmoid(dot(conf_w, [hcout; markov_embed]))
#include "basalt/models/ernie55/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>

#include <cfloat>
#include <cmath>

namespace basalt::models::ernie55 {
namespace {

constexpr int kRopeDim = 64;
constexpr int kRopePairs = 32;
constexpr int kArgmaxBlocks = 256;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

__device__ __forceinline__ void warp_argmax(float& val, int32_t& idx) {
#pragma unroll
    for (int m = 16; m; m >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, val, m);
        const int32_t oi = __shfl_xor_sync(0xffffffffu, idx, m);
        if (ov > val || (ov == val && oi < idx)) {
            val = ov;
            idx = oi;
        }
    }
}

__device__ __forceinline__ void rotate_pair(float& x0, float& x1, float pos, float step, float mscale) {
    float s, c;
    sincosf(pos * step, &s, &c);
    c *= mscale;
    s *= mscale;
    const float a = x0 * c - x1 * s;
    const float b = x0 * s + x1 * c;
    x0 = a;
    x1 = b;
}

__global__ void mhc_stream_mean_kernel(const float* __restrict__ res, int n_tokens, int n_embd,
                                       float* __restrict__ out, int out_stride) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (c >= n_embd || t >= n_tokens) return;
    const float* base = res + (size_t) t * kHcMult * n_embd + c;
    out[(size_t) t * out_stride + c] =
        0.25f * (base[0] + base[n_embd] + base[2 * n_embd] + base[3 * n_embd]);
}

__global__ void dspark_kv_ctx_store_kernel(const float* __restrict__ kv_in, const float* __restrict__ norm_w,
                                           int64_t pos0, int head_dim, int win_cap, float rms_eps, RopeTable rt,
                                           uint16_t* __restrict__ ctx_ring_f16) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    extern __shared__ float sh[];   // [head_dim + 16]
    float* red = sh + head_dim;
    const float* src = kv_in + (size_t) t * head_dim;
    float ss = 0.0f;
    for (int c = tid; c < head_dim; c += blockDim.x) {
        const float v = src[c];
        sh[c] = v;
        ss += v * v;
    }
    ss = warp_sum(ss);
    if ((tid & 31) == 0) red[tid >> 5] = ss;
    __syncthreads();
    float tot = 0.0f;
    for (int w = 0; w < blockDim.x / 32; ++w) tot += red[w];
    const float inv = rsqrtf(tot / (float) head_dim + rms_eps);
    __syncthreads();

    for (int c = tid; c < head_dim; c += blockDim.x) sh[c] = sh[c] * inv * norm_w[c];
    __syncthreads();

    const int64_t p = pos0 + t;
    if (tid < kRopePairs) {
        const int idx = head_dim - kRopeDim + 2 * tid;
        float x0 = sh[idx], x1 = sh[idx + 1];
        rotate_pair(x0, x1, (float) p, rt.step[tid], rt.mscale);
        sh[idx] = x0;
        sh[idx + 1] = x1;
    }
    __syncthreads();

    const size_t slot = (size_t) (p % (int64_t) win_cap);
    __half* dst = reinterpret_cast<__half*>(ctx_ring_f16) + slot * (size_t) head_dim;
    for (int c = tid; c < head_dim; c += blockDim.x) dst[c] = __float2half_rn(sh[c]);
}

// Block (head, row): the scores of every key at once (a warp a key: the window's ring keys, then the block's), the
// softmax with the sink over them, then each thread its columns of the weighted sum (the keys are the values).
__global__ void dspark_swa_block_attn_kernel(const float* __restrict__ q, const uint16_t* __restrict__ ctx_ring_f16,
                                             const float* __restrict__ block_kv, const float* __restrict__ attn_sink,
                                             int64_t pos0, int64_t ctx_start, int n_block, int n_heads, int head_dim,
                                             int win_size, int win_cap, float* __restrict__ out) {
    const int h = blockIdx.x;
    const int r = blockIdx.y;
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, nw = blockDim.x >> 5;
    const float scale = rsqrtf((float) head_dim);

    extern __shared__ float smem[];   // [head_dim (q) + win_size + n_block (scores, then weights)]
    float* sq = smem;
    float* sc = smem + head_dim;

    const float* qrow = q + ((size_t) r * n_heads + h) * head_dim;
    for (int c = tid; c < head_dim; c += blockDim.x) sq[c] = qrow[c];

    // Context window: [max(ctx_start, pos0 - win_size) .. pos0 - 1]
    int64_t lo = pos0 - (int64_t) win_size;
    if (lo < ctx_start) lo = ctx_start;
    if (lo < 0) lo = 0;
    const int n_ctx = (int) (pos0 - lo), n_keys = n_ctx + n_block;
    const __half2* ring = reinterpret_cast<const __half2*>(ctx_ring_f16);
    const int half_dim = head_dim / 2;
    __syncthreads();

    for (int j = wid; j < n_keys; j += nw) {
        float dot = 0.0f;
        if (j < n_ctx) {
            const __half2* krow = ring + (size_t) ((lo + j) % (int64_t) win_cap) * half_dim;
            for (int c = lane; c < half_dim; c += 32) {
                const float2 k = __half22float2(krow[c]);
                dot += sq[2 * c] * k.x + sq[2 * c + 1] * k.y;
            }
        } else {
            const float2* krow = reinterpret_cast<const float2*>(block_kv + (size_t) (j - n_ctx) * head_dim);
            for (int c = lane; c < half_dim; c += 32) {
                const float2 k = krow[c];
                dot += sq[2 * c] * k.x + sq[2 * c + 1] * k.y;
            }
        }
        dot = warp_sum(dot);
        if (lane == 0) sc[j] = dot * scale;
    }
    __syncthreads();

    // the sink's mass exp(sink - m) with V = 0, beside every key's
    float m = attn_sink[h];
    for (int j = 0; j < n_keys; ++j) m = fmaxf(m, sc[j]);
    float l = expf(attn_sink[h] - m);
    for (int j = 0; j < n_keys; ++j) l += expf(sc[j] - m);
    const float inv_l = 1.0f / l;
    __syncthreads();
    for (int j = tid; j < n_keys; j += blockDim.x) sc[j] = expf(sc[j] - m) * inv_l;
    __syncthreads();

    float* orow = out + ((size_t) r * n_heads + h) * head_dim;
    for (int c = tid; c < half_dim; c += blockDim.x) {
        float ax = 0.0f, ay = 0.0f;
        for (int j = 0; j < n_ctx; ++j) {
            const float2 v = __half22float2(ring[(size_t) ((lo + j) % (int64_t) win_cap) * half_dim + c]);
            ax += sc[j] * v.x;
            ay += sc[j] * v.y;
        }
        for (int j = 0; j < n_block; ++j) {
            const float2 v = reinterpret_cast<const float2*>(block_kv + (size_t) j * head_dim)[c];
            ax += sc[n_ctx + j] * v.x;
            ay += sc[n_ctx + j] * v.y;
        }
        orow[2 * c] = ax;
        orow[2 * c + 1] = ay;
    }
}

__global__ void dspark_markov_multimax_argmax_p1(const float* __restrict__ base_logits,
                                                 const float* __restrict__ markov_bias,
                                                 const float* __restrict__ mmr, const float* __restrict__ mmt,
                                                 int n_vocab, float* __restrict__ blk_val,
                                                 int32_t* __restrict__ blk_idx) {
    const float r0 = mmr[0], r1 = mmr[1], r2 = mmr[2], r3 = mmr[3];
    const float t0 = mmt[0], t1 = mmt[1], t2 = mmt[2], t3 = mmt[3];

    float best_v = -INFINITY;
    int32_t best_i = 0;
    const int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_vocab; i += stride) {
        float x = base_logits[i] + markov_bias[i];
        if (!isinf(x)) {
            const float d0 = fmaxf(r0 - x, 0.0f);
            const float d1 = fmaxf(x - r1, 0.0f);
            const float d2 = fmaxf(r2 - x, 0.0f);
            const float d3 = fmaxf(x - r3, 0.0f);
            x = x + t0 * d0 + t1 * d1 + t2 * d2 * d2 + t3 * d3 * d3;
        }
        if (x > best_v || (x == best_v && i < best_i)) {
            best_v = x;
            best_i = i;
        }
    }

    warp_argmax(best_v, best_i);
    __shared__ float sv[8];
    __shared__ int32_t si[8];
    const int wid = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) {
        sv[wid] = best_v;
        si[wid] = best_i;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (blockDim.x >> 5) ? sv[threadIdx.x] : -INFINITY;
        int32_t id = threadIdx.x < (blockDim.x >> 5) ? si[threadIdx.x] : n_vocab;
        warp_argmax(v, id);
        if (threadIdx.x == 0) {
            blk_val[blockIdx.x] = v;
            blk_idx[blockIdx.x] = id;
        }
    }
}

__global__ void dspark_markov_multimax_argmax_p2(const float* __restrict__ blk_val,
                                                 const int32_t* __restrict__ blk_idx, int n_blocks,
                                                 int32_t* __restrict__ out_tok) {
    float v = threadIdx.x < n_blocks ? blk_val[threadIdx.x] : -INFINITY;
    int32_t id = threadIdx.x < n_blocks ? blk_idx[threadIdx.x] : 0x7fffffff;
    warp_argmax(v, id);
    __shared__ float sv[8];
    __shared__ int32_t si[8];
    const int wid = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) {
        sv[wid] = v;
        si[wid] = id;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        float vv = threadIdx.x < (blockDim.x >> 5) ? sv[threadIdx.x] : -INFINITY;
        int32_t ii = threadIdx.x < (blockDim.x >> 5) ? si[threadIdx.x] : 0x7fffffff;
        warp_argmax(vv, ii);
        if (threadIdx.x == 0) *out_tok = ii;
    }
}

__global__ void dspark_confidence_kernel(const float* __restrict__ conf_w, const float* __restrict__ hcout_row,
                                         const float* __restrict__ markov_embed, int n_embd, int rank,
                                         float* __restrict__ out_conf) {
    float s = 0.0f;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) s += conf_w[i] * hcout_row[i];
    for (int j = threadIdx.x; j < rank; j += blockDim.x) s += conf_w[n_embd + j] * markov_embed[j];
    s = warp_sum(s);
    __shared__ float sh[8];
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        for (int w = 0; w < blockDim.x / 32; ++w) tot += sh[w];
        *out_conf = 1.0f / (1.0f + expf(-tot));
    }
}

}  // namespace

void mhc_stream_mean(const float* res, int n_tokens, int n_embd, float* out, int out_stride, void* stream) {
    if (n_tokens <= 0 || n_embd <= 0) return;
    const dim3 grid((n_embd + 255) / 256, n_tokens);
    mhc_stream_mean_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(res, n_tokens, n_embd, out, out_stride);
}

void dspark_kv_ctx_store(const float* kv_in, const float* kv_norm_w, int64_t pos0, int n_tokens, int head_dim,
                         int win_cap, float rms_eps, const RopeTable& rt, uint16_t* ctx_ring_f16, void* stream) {
    if (n_tokens <= 0) return;
    const size_t smem = (size_t) (head_dim + 16) * sizeof(float);
    dspark_kv_ctx_store_kernel<<<n_tokens, 256, smem, (cudaStream_t) stream>>>(kv_in, kv_norm_w, pos0, head_dim,
                                                                               win_cap, rms_eps, rt, ctx_ring_f16);
}

void dspark_swa_block_attn(const float* q, const uint16_t* ctx_ring_f16, const float* block_kv, const float* attn_sink,
                           int64_t pos0, int64_t ctx_start, int n_block, int n_heads, int head_dim, int win_size,
                           int win_cap, float* out, void* stream) {
    if (n_block <= 0) return;
    if (head_dim % 64) throw std::runtime_error("ernie55 dspark_swa_block_attn: head_dim");
    const size_t smem = (size_t) (head_dim + win_size + n_block) * sizeof(float);
    dspark_swa_block_attn_kernel<<<dim3(n_heads, n_block), 256, smem, (cudaStream_t) stream>>>(
        q, ctx_ring_f16, block_kv, attn_sink, pos0, ctx_start, n_block, n_heads, head_dim, win_size, win_cap, out);
}

void dspark_markov_multimax_argmax(const float* base_logits, const float* markov_bias, const float* mmr,
                                   const float* mmt, int n_vocab, float* scratch_val, int32_t* scratch_idx,
                                   int32_t* out_tok, void* stream) {
    dspark_markov_multimax_argmax_p1<<<kArgmaxBlocks, 256, 0, (cudaStream_t) stream>>>(
        base_logits, markov_bias, mmr, mmt, n_vocab, scratch_val, scratch_idx);
    dspark_markov_multimax_argmax_p2<<<1, kArgmaxBlocks, 0, (cudaStream_t) stream>>>(
        scratch_val, scratch_idx, kArgmaxBlocks, out_tok);
}

void dspark_confidence(const float* conf_w, const float* hcout_row, const float* markov_embed, int n_embd, int rank,
                       float* out_conf, void* stream) {
    dspark_confidence_kernel<<<1, 256, 0, (cudaStream_t) stream>>>(conf_w, hcout_row, markov_embed, n_embd, rank,
                                                                   out_conf);
}

}  // namespace basalt::models::ernie55
