// src/models/ernie55/kernels/hca.cu - ERNIE-5.5 Heavily Compressed Attention (HCA) CUDA kernels:
//   - Q interleaved RoPE (last 64 dims of head_dim=512)
//   - KV RMSNorm + interleaved RoPE + sliding-window ring store (FP8 or FP16)
//   - 128:1 streaming compressor with additive APE[pos % 128], softmax-weighted sum, RMSNorm, YaRN RoPE, cache store
//   - Multi-query attention over compressed blocks + sliding window with per-head attn_sink
//   - Inverse interleaved RoPE on VHA-mixed attention output
#include "basalt/models/ernie55/kernels.hpp"

#include "basalt/kernels/mma_f16.cuh"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/q8_1_finite.hpp"
#include "basalt/models/ernie55/cache_layout.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <stdexcept>

namespace basalt::models::ernie55 {
namespace {

namespace bk = basalt::kernels;

constexpr int kRopeDim = layout::kHcaRopeDim;
constexpr int kRopePairs = kRopeDim / 2;
constexpr int kHcaFp8Stride = layout::kHcaFp8RowBytes;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m));
    return v;
}

__device__ __forceinline__ void rotate_pair(float& x0, float& x1, float pos, float step, float mscale, bool inverse) {
    float s, c;
    sincosf(pos * step, &s, &c);
    c *= mscale;
    s *= mscale;
    if (inverse) s = -s;
    const float a = x0 * c - x1 * s;
    const float b = x0 * s + x1 * c;
    x0 = a;
    x1 = b;
}

__device__ __forceinline__ void store_hca_row(uint8_t* cache, size_t slot, const float* val, int head_dim, bool fp8,
                                              float* sh_red) {
    if (!fp8) {
        __half* dst = reinterpret_cast<__half*>(cache) + slot * (size_t) head_dim;
        for (int c = threadIdx.x; c < head_dim; c += blockDim.x) dst[c] = __float2half_rn(val[c]);
        return;
    }
    // FP8: nope [0..head_dim-64) quantized per 64 values with power-of-two UE8M0-compatible FP32 scale, rope [head_dim-64..head_dim) BF16
    const int nope = head_dim - kRopeDim;
    const int ngrp = nope / 64;
    uint8_t* row = cache + slot * (size_t) kHcaFp8Stride;
    float* scales = reinterpret_cast<float*>(row + nope);
    uint16_t* rope_out = reinterpret_cast<uint16_t*>(row + nope + ngrp * 4);

    // One warp per 64-value group (requires blockDim.x >= 256; handles arbitrary head_dim via loop)
    for (int g = threadIdx.x / 32; g < ngrp; g += blockDim.x / 32) {
        const int lane = threadIdx.x & 31;
        const int c0 = g * 64 + lane;
        const int c1 = c0 + 32;
        const float v0 = __bfloat162float(__float2bfloat16_rn(val[c0]));
        const float v1 = __bfloat162float(__float2bfloat16_rn(val[c1]));
        const float am = warp_max(fmaxf(fabsf(v0), fabsf(v1)));
        const float sc = exp2f(ceilf(log2f(fmaxf(am / 448.0f, 1e-4f))));
        if (lane == 0) scales[g] = sc;
        const float inv = 1.0f / sc;
        row[c0] = __nv_cvt_float_to_fp8(v0 * inv, __NV_SATFINITE, __NV_E4M3);
        row[c1] = __nv_cvt_float_to_fp8(v1 * inv, __NV_SATFINITE, __NV_E4M3);
    }
    for (int i = threadIdx.x; i < kRopeDim; i += blockDim.x) {
        rope_out[i] = __bfloat16_as_ushort(__float2bfloat16_rn(val[nope + i]));
    }
    (void) sh_red;
}

__device__ __forceinline__ float load_hca_elem(const uint8_t* cache, size_t slot, int c, int head_dim, bool fp8) {
    if (!fp8) {
        const __half* src = reinterpret_cast<const __half*>(cache) + slot * (size_t) head_dim;
        return __half2float(src[c]);
    }
    const int nope = head_dim - kRopeDim;
    const uint8_t* row = cache + slot * (size_t) kHcaFp8Stride;
    if (c < nope) {
        const float* scales = reinterpret_cast<const float*>(row + nope);
        const float sc = scales[c >> 6];
        const __nv_fp8_e4m3 q = *reinterpret_cast<const __nv_fp8_e4m3*>(row + c);
        return (float) q * sc;
    }
    const int ngrp = nope / 64;
    const uint16_t* rope_in = reinterpret_cast<const uint16_t*>(row + nope + ngrp * 4);
    return __bfloat162float(__ushort_as_bfloat16(rope_in[c - nope]));
}

// Weightless per-head RMSNorm (DeepSeek-V4 / ERNIE-5.5 qk_norm_type="per_head") + forward interleaved RoPE on last 64 dims.
// One warp (32 lanes) per (token, head) so hca_q_rope and hca_qkv_rope_commit produce bit-identical Q tensors.
__device__ __forceinline__ void hca_q_norm_rope_warp(float* head_ptr, int head_dim, float pos_t, float rms_eps,
                                                     const RopeTable& rt, int lane) {
    float ss = 0.0f;
    for (int c = lane; c < head_dim; c += 32) {
        const float v = head_ptr[c];
        ss += v * v;
    }
    ss = warp_sum(ss);
    const float inv = rsqrtf(ss / (float) head_dim + rms_eps);
    for (int c = lane; c < head_dim; c += 32) head_ptr[c] *= inv;
    __syncwarp();
    float* base = head_ptr + (head_dim - kRopeDim) + 2 * lane;
    float x0 = base[0], x1 = base[1];
    rotate_pair(x0, x1, pos_t, rt.step[lane], rt.mscale, false);
    base[0] = x0;
    base[1] = x1;
}

__global__ void hca_q_norm_rope_kernel(float* __restrict__ q, const int32_t* __restrict__ pos, int n_heads,
                                       int head_dim, float rms_eps, RopeTable rt) {
    const int lane = threadIdx.x;   // 0..31
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    if (lane >= kRopePairs) return;
    float* head_ptr = q + ((size_t) t * n_heads + h) * head_dim;
    hca_q_norm_rope_warp(head_ptr, head_dim, (float) pos[t], rms_eps, rt, lane);
}

__global__ void hca_inv_rope_kernel(float* __restrict__ x, const int32_t* __restrict__ pos, int n_heads, int head_dim,
                                    RopeTable rt) {
    const int pr = threadIdx.x;   // 0..31
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    if (pr >= kRopePairs) return;
    float* base = x + ((size_t) t * n_heads + h) * head_dim + (head_dim - kRopeDim) + 2 * pr;
    float x0 = base[0], x1 = base[1];
    rotate_pair(x0, x1, (float) pos[t], rt.step[pr], rt.mscale, true);
    base[0] = x0;
    base[1] = x1;
}

__global__ void hca_kv_norm_rope_kernel(const float* __restrict__ kv_in, const float* __restrict__ norm_w,
                                        const int32_t* __restrict__ pos, int head_dim, float rms_eps, RopeTable rt,
                                        float* __restrict__ kv_rot) {
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

    if (tid < kRopePairs) {
        const int idx = head_dim - kRopeDim + 2 * tid;
        float x0 = sh[idx], x1 = sh[idx + 1];
        rotate_pair(x0, x1, (float) pos[t], rt.step[tid], rt.mscale, false);
        sh[idx] = x0;
        sh[idx + 1] = x1;
    }
    __syncthreads();

    float* dst = kv_rot + (size_t) t * head_dim;
    for (int c = tid; c < head_dim; c += blockDim.x) dst[c] = sh[c];
}

__global__ void hca_win_commit_kernel(const float* __restrict__ kv_rot, const int32_t* __restrict__ pos, int start_t,
                                      int head_dim, int win_cap, uint8_t* __restrict__ win_cache, bool fp8) {
    const int t = start_t + blockIdx.x;
    const int tid = threadIdx.x;
    extern __shared__ float sh[];   // [head_dim + 16]
    float* red = sh + head_dim;
    const float* src = kv_rot + (size_t) t * head_dim;
    for (int c = tid; c < head_dim; c += blockDim.x) sh[c] = src[c];
    __syncthreads();
    const int slot = pos[t] % win_cap;
    store_hca_row(win_cache, (size_t) slot, sh, head_dim, fp8, red);
}

__global__ void hca_kv_store_kernel(const float* __restrict__ kv_in, const float* __restrict__ norm_w,
                                    const int32_t* __restrict__ pos, int start_t, int head_dim, int win_size,
                                    float rms_eps, RopeTable rt, uint8_t* __restrict__ win_cache, bool fp8) {
    const int t = start_t + blockIdx.x;
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

    if (tid < kRopePairs) {
        const int idx = head_dim - kRopeDim + 2 * tid;
        float x0 = sh[idx], x1 = sh[idx + 1];
        rotate_pair(x0, x1, (float) pos[t], rt.step[tid], rt.mscale, false);
        sh[idx] = x0;
        sh[idx + 1] = x1;
    }
    __syncthreads();

    const int slot = pos[t] % win_size;
    store_hca_row(win_cache, (size_t) slot, sh, head_dim, fp8, red);
}

// Parallel multi-block compressor for full ratio-aligned blocks: one block (head_dim threads) per closed block.
__global__ void hca_compress_blocks_kernel(const float* __restrict__ ckv, const float* __restrict__ cscore,
                                           const float* __restrict__ ape, const float* __restrict__ norm_w,
                                           int pos0, int ratio, int head_dim, float rms_eps, RopeTable comp_rt,
                                           uint8_t* __restrict__ comp_cache, bool fp8) {
    extern __shared__ float sh[];   // [head_dim + 16]
    float* red = sh + head_dim;
    const int b = blockIdx.x;
    const int c = threadIdx.x;
    if (c >= head_dim) return;

    const int t0 = b * ratio;
    const int p_start = pos0 + t0;
    float m = -FLT_MAX;
    for (int j = 0; j < ratio; ++j) {
        const float sc = cscore[(size_t) (t0 + j) * head_dim + c] + ape[(size_t) j * head_dim + c];
        m = fmaxf(m, sc);
    }
    float den = 0.0f, num = 0.0f;
    for (int j = 0; j < ratio; ++j) {
        const float sc = cscore[(size_t) (t0 + j) * head_dim + c] + ape[(size_t) j * head_dim + c];
        const float e = expf(sc - m);
        den += e;
        num += ckv[(size_t) (t0 + j) * head_dim + c] * e;
    }
    const float comp = num / den;
    float ss = warp_sum(comp * comp);
    if ((c & 31) == 0) red[c >> 5] = ss;
    __syncthreads();
    float tot = 0.0f;
    for (int w = 0; w < head_dim / 32; ++w) tot += red[w];
    const float y = comp * rsqrtf(tot / (float) head_dim + rms_eps) * norm_w[c];
    sh[c] = y;
    __syncthreads();

    if (c < kRopePairs) {
        const int idx = head_dim - kRopeDim + 2 * c;
        float x0 = sh[idx], x1 = sh[idx + 1];
        rotate_pair(x0, x1, (float) p_start, comp_rt.step[c], comp_rt.mscale, false);
        sh[idx] = x0;
        sh[idx + 1] = x1;
    }
    __syncthreads();

    store_hca_row(comp_cache, (size_t) (p_start / ratio), sh, head_dim, fp8, red);
}

// Single-block sequential streaming compressor over n_tokens: guarantees exact ordering when a chunk crosses
// multiple 128-token block boundaries. One thread per channel (blockDim.x = head_dim, e.g. 512).
__global__ void hca_compress_kernel(const float* __restrict__ ckv, const float* __restrict__ cscore,
                                    const float* __restrict__ ape, const float* __restrict__ norm_w,
                                    const int32_t* __restrict__ pos, int n_tokens, int ratio, int head_dim,
                                    float rms_eps, RopeTable comp_rt, float* __restrict__ state_kv,
                                    float* __restrict__ state_score, uint8_t* __restrict__ comp_cache, bool fp8) {
    bk::pdl_wait();
    extern __shared__ float sh[];   // [head_dim + 16]
    float* red = sh + head_dim;
    const int c = threadIdx.x;
    if (c >= head_dim) return;

    for (int t = 0; t < n_tokens; ++t) {
        const int p = pos[t];
        const int slot = p % ratio;
        state_kv[(size_t) slot * head_dim + c] = ckv[(size_t) t * head_dim + c];
        state_score[(size_t) slot * head_dim + c] =
            cscore[(size_t) t * head_dim + c] + ape[(size_t) slot * head_dim + c];
        __syncthreads();

        if ((p + 1) % ratio != 0) continue;

        // Softmax-weighted sum over the closed group [0 .. ratio)
        float m = -FLT_MAX;
        for (int j = 0; j < ratio; ++j) m = fmaxf(m, state_score[(size_t) j * head_dim + c]);
        float den = 0.0f, num = 0.0f;
        for (int j = 0; j < ratio; ++j) {
            const float e = expf(state_score[(size_t) j * head_dim + c] - m);
            den += e;
            num += state_kv[(size_t) j * head_dim + c] * e;
        }
        const float comp = num / den;
        float ss = warp_sum(comp * comp);
        if ((c & 31) == 0) red[c >> 5] = ss;
        __syncthreads();
        float tot = 0.0f;
        for (int w = 0; w < head_dim / 32; ++w) tot += red[w];
        const float y = comp * rsqrtf(tot / (float) head_dim + rms_eps) * norm_w[c];
        sh[c] = y;
        __syncthreads();

        if (c < kRopePairs) {
            const int idx = head_dim - kRopeDim + 2 * c;
            float x0 = sh[idx], x1 = sh[idx + 1];
            rotate_pair(x0, x1, (float) (p + 1 - ratio), comp_rt.step[c], comp_rt.mscale, false);
            sh[idx] = x0;
            sh[idx + 1] = x1;
        }
        __syncthreads();

        const size_t blk_id = (size_t) (p / ratio);
        store_hca_row(comp_cache, blk_id, sh, head_dim, fp8, red);
        __syncthreads();
    }
}

// One block (256 threads) per (head, token): online softmax attention over compressed blocks [0..n_comp)
// + sliding window [win_start..pos] + per-head attn_sink.
__global__ void hca_attention_kernel(const float* __restrict__ q, const uint8_t* __restrict__ comp_cache,
                                     const uint8_t* __restrict__ win_cache, const float* __restrict__ kv_rot,
                                     const float* __restrict__ attn_sink, const int32_t* __restrict__ pos,
                                     int n_heads, int head_dim, int win_size, int win_cap, int ratio, bool fp8,
                                     float* __restrict__ out) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int tid = threadIdx.x;
    const int p = pos[t];
    const int pos0 = pos[0];
    const int win_start = p - (win_size - 1) > 0 ? p - (win_size - 1) : 0;
    const int win_n = p - win_start + 1;
    const int n_comp = (p + 1) / ratio;
    const float scale = rsqrtf((float) head_dim);

    extern __shared__ float smem[];   // [head_dim (q) + head_dim (acc) + 16 (red)]
    float* sq = smem;
    float* acc = smem + head_dim;
    float* red = acc + head_dim;
    const float* qrow = q + ((size_t) t * n_heads + h) * head_dim;
    for (int c = tid; c < head_dim; c += blockDim.x) {
        sq[c] = qrow[c];
        acc[c] = 0.0f;
    }
    __syncthreads();

    // Initialize online softmax with attention sink (no V contribution, only logit mass exp(sink))
    float m = attn_sink[h];
    float l = 1.0f;   // exp(sink - sink) = 1

    auto step_cache_row = [&](const uint8_t* cache, size_t slot) {
        float dot = 0.0f;
        for (int c = tid; c < head_dim; c += blockDim.x) dot += sq[c] * load_hca_elem(cache, slot, c, head_dim, fp8);
        dot = warp_sum(dot);
        if ((tid & 31) == 0) red[tid >> 5] = dot;
        __syncthreads();
        float s = 0.0f;
        for (int w = 0; w < blockDim.x / 32; ++w) s += red[w];
        s *= scale;
        const float m_new = fmaxf(m, s);
        const float alpha = expf(m - m_new);
        const float beta = expf(s - m_new);
        l = l * alpha + beta;
        m = m_new;
        for (int c = tid; c < head_dim; c += blockDim.x) {
            const float v = load_hca_elem(cache, slot, c, head_dim, fp8);
            acc[c] = acc[c] * alpha + beta * v;
        }
        __syncthreads();
    };

    auto step_f32_row = [&](const float* row) {
        float dot = 0.0f;
        for (int c = tid; c < head_dim; c += blockDim.x) dot += sq[c] * row[c];
        dot = warp_sum(dot);
        if ((tid & 31) == 0) red[tid >> 5] = dot;
        __syncthreads();
        float s = 0.0f;
        for (int w = 0; w < blockDim.x / 32; ++w) s += red[w];
        s *= scale;
        const float m_new = fmaxf(m, s);
        const float alpha = expf(m - m_new);
        const float beta = expf(s - m_new);
        l = l * alpha + beta;
        m = m_new;
        for (int c = tid; c < head_dim; c += blockDim.x) acc[c] = acc[c] * alpha + beta * row[c];
        __syncthreads();
    };

    for (int b = 0; b < n_comp; ++b) step_cache_row(comp_cache, (size_t) b);
    for (int j = 0; j < win_n; ++j) {
        const int jp = win_start + j;
        if (kv_rot && jp >= pos0) step_f32_row(kv_rot + (size_t) (jp - pos0) * head_dim);
        else step_cache_row(win_cache, (size_t) (jp % win_cap));
    }

    const float inv_l = 1.0f / l;
    float* orow = out + ((size_t) t * n_heads + h) * head_dim;
    for (int c = tid; c < head_dim; c += blockDim.x) orow[c] = acc[c] * inv_l;
}

// Fast GQA-tiled HCA attention for head_dim == 512: 8 warps (256 threads) = 8 query heads per block sharing
// 1 KV head in 4-row tiles of shared memory (skv[4][512] = 8 KB). Cuts global KV loads by 16x and barriers by 4x.
__global__ void hca_attention_gqa512_kernel(const float* __restrict__ q, const uint8_t* __restrict__ comp_cache,
                                            const uint8_t* __restrict__ win_cache, const float* __restrict__ kv_rot,
                                            const float* __restrict__ attn_sink, const int32_t* __restrict__ pos,
                                            int n_heads, int win_size, int win_cap, int ratio, bool fp8,
                                            float* __restrict__ out) {
    constexpr int kHeadDim = 512;
    constexpr int kTile = 4;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int h = blockIdx.x * 8 + wid;
    const int t = blockIdx.y;
    const bool active = h < n_heads;

    const int p = pos[t];
    const int pos0 = pos[0];
    const int win_start = p - (win_size - 1) > 0 ? p - (win_size - 1) : 0;
    const int win_n = p - win_start + 1;
    const int n_comp = (p + 1) / ratio;
    constexpr float scale = 0.04419417382415922f;   // 1 / sqrt(512)

    __shared__ float skv[kTile][kHeadDim];

    float q_reg[16];
    float acc_reg[16];
    if (active) {
        const float* qrow = q + ((size_t) t * n_heads + h) * kHeadDim;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            q_reg[i] = qrow[lane + 32 * i];
            acc_reg[i] = 0.0f;
        }
    } else {
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            q_reg[i] = 0.0f;
            acc_reg[i] = 0.0f;
        }
    }

    float m = active ? attn_sink[h] : 0.0f;
    float l = 1.0f;

    auto process_tile = [&](int cnt) {
        if (!active) return;
        for (int r = 0; r < cnt; ++r) {
            const float* kv = skv[r];
            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < 16; ++i) dot = fmaf(q_reg[i], kv[lane + 32 * i], dot);
            const float s = warp_sum(dot) * scale;
            const float m_new = fmaxf(m, s);
            const float alpha = expf(m - m_new);
            const float beta = expf(s - m_new);
            l = l * alpha + beta;
            m = m_new;
#pragma unroll
            for (int i = 0; i < 16; ++i) acc_reg[i] = fmaf(acc_reg[i], alpha, beta * kv[lane + 32 * i]);
        }
    };

    // 1. Compressed blocks [0 .. n_comp) in tiles of 4
    for (int b0 = 0; b0 < n_comp; b0 += kTile) {
        const int cnt = (n_comp - b0 < kTile) ? (n_comp - b0) : kTile;
        for (int r = 0; r < cnt; ++r) {
            const size_t slot = (size_t) (b0 + r);
            skv[r][tid] = load_hca_elem(comp_cache, slot, tid, kHeadDim, fp8);
            skv[r][tid + 256] = load_hca_elem(comp_cache, slot, tid + 256, kHeadDim, fp8);
        }
        __syncthreads();
        process_tile(cnt);
        __syncthreads();
    }

    // 2. Sliding window [win_start .. p] in tiles of 4
    for (int j0 = 0; j0 < win_n; j0 += kTile) {
        const int cnt = (win_n - j0 < kTile) ? (win_n - j0) : kTile;
        for (int r = 0; r < cnt; ++r) {
            const int jp = win_start + j0 + r;
            if (kv_rot && jp >= pos0) {
                const float* row = kv_rot + (size_t) (jp - pos0) * kHeadDim;
                skv[r][tid] = row[tid];
                skv[r][tid + 256] = row[tid + 256];
            } else {
                const size_t slot = (size_t) (jp % win_cap);
                skv[r][tid] = load_hca_elem(win_cache, slot, tid, kHeadDim, fp8);
                skv[r][tid + 256] = load_hca_elem(win_cache, slot, tid + 256, kHeadDim, fp8);
            }
        }
        __syncthreads();
        process_tile(cnt);
        __syncthreads();
    }

    if (active) {
        const float inv_l = 1.0f / l;
        float* orow = out + ((size_t) t * n_heads + h) * kHeadDim;
#pragma unroll
        for (int i = 0; i < 16; ++i) orow[lane + 32 * i] = acc_reg[i] * inv_l;
    }
}

// ---- Blackwell tensor-core prefill attention (the prefill pass: n_tokens > 8) --------------------------------------------------
// Modeled directly on hy4flash/kernels/mla.cu's attend_mma_kernel: one 256-thread block (8 warps) per token, all
// n_heads <= 32 query heads in a single M = 32 MMA tile. The token's keys (compressed blocks + sliding window +
// kv_rot rows of the current chunk) go through shared memory 32 at a time, converted to packed FP16 as they are
// fetched (FP8 and FP16 cache rows, F32 kv_rot rows); S = Q K^T and O += P V run on mma.m16n8k16 with the scaled
// query and the probabilities in FP16, accumulated in FP32. The softmax is a running one, the sink its first element.
// Needs sm_80+ (ldmatrix / mma.sync); the kernel and its launch are compiled for __CUDA_ARCH__ >= 800 (this build
// targets sm_120+), elsewhere hca_attention() falls back to hca_attention_gqa512_kernel.

// FP16 shared-memory row strides (avoid bank conflicts; kMmaLd * 2 % 16 == 0 keeps the uint4 staging aligned)
constexpr int kMmaCells = 32;   // KV tile height: 32 keys per iteration
constexpr int kMmaG = 2;        // M = 16 * kMmaG = 32 padded query heads per block
constexpr int kMmaLd = 520;     // qh/ks stride in FP16 elements: 512 + 8
constexpr int kMmaPLd = 40;     // ph (softmax probabilities) stride in FP16 elements

constexpr float kHcaScale512 = 0.04419417382415922f;   // 1 / sqrt(512)

constexpr size_t mma512_smem() {
    constexpr int M = 16 * kMmaG;
    return (size_t) (M * kMmaLd + kMmaCells * kMmaLd + M * kMmaPLd) * sizeof(uint16_t)
         + (size_t) (2 * M * kMmaCells + 2 * M) * sizeof(float);
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800

// the FP16 MMA inlines and the wider -> FP16 conversions Basalt's attention kernels share
using basalt::kernels::mma_f16::bf2_to_h2;
using basalt::kernels::mma_f16::e4m3x2_to_h2_f32scale;
using basalt::kernels::mma_f16::f32_to_h_clamped;
using basalt::kernels::mma_f16::ldsm4;
using basalt::kernels::mma_f16::ldsm4t;
using basalt::kernels::mma_f16::mma16816h;

// Eight values of a cache row's chunk ch (of 64) as eight packed FP16 values, ready for the shared-memory staging.
// FP16 rows: the eight halves. FP8 rows: chunks below 56 are eight E4M3 codes times their 64-value group's scale
// (8-value chunks, 64-value groups: scales[ch >> 3], NOT hy4flash's ch >> 4), the rest eight BF16 rope values. The
// rope tail starts at row + 448 + 7 * 4 = 476, which is only 4-byte aligned (476 % 16 == 12): it loads as four
// uint32_t, never a uint4 (that would be a misaligned-address trap). An FP8 code times its power-of-two scale stays
// normal in FP16 down to 2^-14; the smaller scales store_hca_row may hold (down to 2^-22) round through FP16
// subnormals, and the BF16 rope values and F32 kv_rot rows saturate to FP16's finite range, never to +-Inf.
template <bool FP8>
__device__ __forceinline__ uint4 fetch_row8(const uint8_t* cache, size_t slot, int ch) {
    if (!FP8) return *reinterpret_cast<const uint4*>(cache + slot * (size_t) (2 * 512) + ch * 16);
    const uint8_t* row = cache + slot * (size_t) kHcaFp8Stride;
    if (ch < 56) {
        const uint2 codes = *reinterpret_cast<const uint2*>(row + ch * 8);
        const float s = reinterpret_cast<const float*>(row + 448)[ch >> 3];
        return make_uint4(e4m3x2_to_h2_f32scale(codes.x & 0xffffu, s), e4m3x2_to_h2_f32scale(codes.x >> 16, s),
                          e4m3x2_to_h2_f32scale(codes.y & 0xffffu, s), e4m3x2_to_h2_f32scale(codes.y >> 16, s));
    }
    const uint32_t* w = reinterpret_cast<const uint32_t*>(row + 448 + 7 * 4 + (ch - 56) * 16);
    return make_uint4(bf2_to_h2(w[0]), bf2_to_h2(w[1]), bf2_to_h2(w[2]), bf2_to_h2(w[3]));
}

// The eight F32 values of a kv_rot row's chunk ch as eight packed FP16 values (the row is 16-byte aligned: its base
// is a cudaMalloc'd float buffer, the row and chunk offsets are multiples of 16 bytes), saturated to FP16's range.
__device__ __forceinline__ uint4 fetch_f32_8(const float* row, int ch) {
    const float4* src = reinterpret_cast<const float4*>(row + ch * 8);
    const float4 a = src[0], b = src[1];
    const __half2 h0 = __halves2half2(f32_to_h_clamped(a.x), f32_to_h_clamped(a.y));
    const __half2 h1 = __halves2half2(f32_to_h_clamped(a.z), f32_to_h_clamped(a.w));
    const __half2 h2 = __halves2half2(f32_to_h_clamped(b.x), f32_to_h_clamped(b.y));
    const __half2 h3 = __halves2half2(f32_to_h_clamped(b.z), f32_to_h_clamped(b.w));
    return make_uint4(*reinterpret_cast<const uint32_t*>(&h0), *reinterpret_cast<const uint32_t*>(&h1),
                      *reinterpret_cast<const uint32_t*>(&h2), *reinterpret_cast<const uint32_t*>(&h3));
}

// One block (256 threads, 8 warps) per token of a prefill chunk: online-softmax attention over the compressed
// blocks [0 .. n_comp) + the sliding window [win_start .. p] (the current chunk's rows straight from kv_rot) with
// per-head attn_sink. Padded heads (n_heads .. 31) compute but never write.
template <bool FP8>
__global__ void __launch_bounds__(256) hca_attention_mma512_kernel(const float* __restrict__ q,
                                                                   const uint8_t* __restrict__ comp_cache,
                                                                   const uint8_t* __restrict__ win_cache,
                                                                   const float* __restrict__ kv_rot,
                                                                   const float* __restrict__ attn_sink,
                                                                   const int32_t* __restrict__ pos, int n_heads,
                                                                   int win_size, int win_cap, int ratio,
                                                                   float* __restrict__ out) {
    extern __shared__ __align__(16) unsigned char smraw[];
    constexpr int W = 512, M = 16 * kMmaG;
    uint16_t* qh = (uint16_t*) smraw;
    uint16_t* ks = qh + M * kMmaLd;
    uint16_t* ph = ks + kMmaCells * kMmaLd;
    float* sp = (float*) (ph + M * kMmaPLd);   // [2][M][32]
    float* alpha_s = sp + 2 * M * 32;
    float* l_s = alpha_s + M;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    const int h0 = blockIdx.x * M;
    const int t = blockIdx.y;
    const int p = pos[t], pos0 = pos[0];
    const int n_comp = (p + 1) / ratio;
    const int win_start = p - (win_size - 1) > 0 ? p - (win_size - 1) : 0;
    const int total_keys = n_comp + (p - win_start + 1);

    // the query, scaled by 1 / sqrt(512) and in FP16; heads >= n_heads pad with zeros
    for (int i = tid; i < M * W; i += 256) {
        const int h = i / W, e = i - h * W;
        const int gh = h0 + h;
        const float v = gh < n_heads ? q[((size_t) t * n_heads + gh) * W + e] : 0.0f;
        qh[h * kMmaLd + e] = __half_as_ushort(f32_to_h_clamped(v * kHcaScale512));
    }
    // the softmax: a warp keeps 2 * kMmaG = 4 heads
    float m[2 * kMmaG], l[2 * kMmaG];
#pragma unroll
    for (int hh = 0; hh < 2 * kMmaG; ++hh) {
        const int h = w * 2 * kMmaG + hh;
        const int gh = h0 + h;
        m[hh] = gh < n_heads ? attn_sink[gh] : 0.0f;
        l[hh] = 1.0f;
    }
    if (tid < M) l_s[tid] = 1.0f;
    // O: a warp keeps the 16 heads of group rg and the 128 output dimensions dq * 128 .. dq * 128 + 127
    float acc[8 * kMmaG][4];
#pragma unroll
    for (int j = 0; j < 8 * kMmaG; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;
    const int id = lane >> 3, r8 = lane & 7, rg = w % kMmaG, dq = w / kMmaG;
    // S: a warp takes the heads of group rg, half kh of the 512 dimensions and 16 keys from key 16 * cp
    const int kh = (w / kMmaG) & 1, cp = w / (2 * kMmaG);
    // a thread's share of a tile (8 chunks of 8 values), loaded a tile ahead: the next tile's reads are in flight
    // while this one computes
    constexpr int kPer = kMmaCells * (W / 8) / 256;
    uint4 pre[kPer];
    auto fetch = [&](int j0) {
        const int tn = min(kMmaCells, total_keys - j0);
#pragma unroll
        for (int k = 0; k < kPer; ++k) {
            const int i = tid + k * 256, c = i / (W / 8), ch = i - c * (W / 8);
            pre[k] = make_uint4(0, 0, 0, 0);
            if (c < tn) {
                const int j = j0 + c;
                if (j < n_comp) {
                    pre[k] = fetch_row8<FP8>(comp_cache, (size_t) j, ch);
                } else {
                    const int jp = win_start + (j - n_comp);
                    if (kv_rot && jp >= pos0) pre[k] = fetch_f32_8(kv_rot + (size_t) (jp - pos0) * W, ch);
                    else pre[k] = fetch_row8<FP8>(win_cache, (size_t) (jp % win_cap), ch);
                }
            }
        }
    };
    fetch(0);
    for (int j0 = 0; j0 < total_keys; j0 += kMmaCells) {
        const int tn = min(kMmaCells, total_keys - j0);
        __syncthreads();
#pragma unroll
        for (int k = 0; k < kPer; ++k) {
            const int i = tid + k * 256, c = i / (W / 8), ch = i - c * (W / 8);
            *reinterpret_cast<uint4*>(ks + c * kMmaLd + ch * 8) = pre[k];
        }
        __syncthreads();
        if (j0 + kMmaCells < total_keys) fetch(j0 + kMmaCells);
        {   // S = Q K^T: the half kh of the dimensions' partial dots
            float c[kMmaG][4];
#pragma unroll
            for (int nt = 0; nt < kMmaG; ++nt) c[nt][0] = c[nt][1] = c[nt][2] = c[nt][3] = 0.0f;
#pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                const int kb = (kh * 16 + kk * 2) * 16;
                uint32_t b[kMmaG][4];
#pragma unroll
                for (int nt = 0; nt < kMmaG; ++nt)
                    ldsm4(b[nt], ks + ((cp * kMmaG + nt) * 8 + r8) * kMmaLd + kb + id * 8);
#pragma unroll
                for (int s2 = 0; s2 < 2; ++s2) {
                    const int kd = kb + s2 * 16;
                    uint32_t ah[4];
                    ldsm4(ah, qh + (rg * 16 + (id & 1) * 8 + r8) * kMmaLd + kd + (id >> 1) * 8);
#pragma unroll
                    for (int nt = 0; nt < kMmaG; ++nt) mma16816h(c[nt], ah, b[nt][2 * s2], b[nt][2 * s2 + 1]);
                }
            }
            const int row = rg * 16 + (lane >> 2);
#pragma unroll
            for (int nt = 0; nt < kMmaG; ++nt) {
                const int col = (cp * kMmaG + nt) * 8 + (lane & 3) * 2;
                sp[(kh * M + row) * 32 + col] = c[nt][0];
                sp[(kh * M + row) * 32 + col + 1] = c[nt][1];
                sp[(kh * M + row + 8) * 32 + col] = c[nt][2];
                sp[(kh * M + row + 8) * 32 + col + 1] = c[nt][3];
            }
        }
        __syncthreads();
        // the running softmax: a lane a key
#pragma unroll
        for (int hh = 0; hh < 2 * kMmaG; ++hh) {
            const int h = w * 2 * kMmaG + hh;
            const bool valid = lane < tn;
            const float sc = valid ? sp[h * 32 + lane] + sp[(M + h) * 32 + lane] : -INFINITY;
            const float mx = warp_max(sc);
            const float nm = fmaxf(m[hh], mx);
            const float alpha = expf(m[hh] - nm);
            const float pr = valid ? expf(sc - nm) : 0.0f;
            const float sum = warp_sum(pr);
            l[hh] = l[hh] * alpha + sum;
            m[hh] = nm;
            ph[h * kMmaPLd + lane] = __half_as_ushort(__float2half_rn(pr));
            if (lane == 0) {
                alpha_s[h] = alpha;
                l_s[h] = l[hh];
            }
        }
        __syncthreads();
        // O = O * alpha + P V
        {
            const float a0 = alpha_s[rg * 16 + (lane >> 2)], a1 = alpha_s[rg * 16 + (lane >> 2) + 8];
#pragma unroll
            for (int j = 0; j < 8 * kMmaG; ++j) {
                acc[j][0] *= a0; acc[j][1] *= a0;
                acc[j][2] *= a1; acc[j][3] *= a1;
            }
#pragma unroll
            for (int s2 = 0; s2 < 2; ++s2) {
                uint32_t ah[4];
                ldsm4(ah, ph + (rg * 16 + (id & 1) * 8 + r8) * kMmaPLd + s2 * 16 + (id >> 1) * 8);
#pragma unroll
                for (int jj = 0; jj < 4 * kMmaG; ++jj) {
                    uint32_t b[4];
                    ldsm4t(b, ks + (s2 * 16 + (id & 1) * 8 + r8) * kMmaLd + dq * (W / (2 * kMmaG)) + jj * 16 + (id >> 1) * 8);
                    mma16816h(acc[2 * jj], ah, b[0], b[1]);
                    mma16816h(acc[2 * jj + 1], ah, b[2], b[3]);
                }
            }
        }
    }
    __syncthreads();
    const int row = rg * 16 + (lane >> 2);
    const int g0 = h0 + row, g1 = h0 + row + 8;
    const float i0 = 1.0f / l_s[row], i1 = 1.0f / l_s[row + 8];
#pragma unroll
    for (int j = 0; j < 8 * kMmaG; ++j) {
        const int d = dq * (W / (2 * kMmaG)) + j * 8 + (lane & 3) * 2;
        if (g0 < n_heads) {
            float* o0 = out + ((size_t) t * n_heads + g0) * W + d;
            o0[0] = acc[j][0] * i0; o0[1] = acc[j][1] * i0;
        }
        if (g1 < n_heads) {
            float* o1 = out + ((size_t) t * n_heads + g1) * W + d;
            o1[0] = acc[j][2] * i1; o1[1] = acc[j][3] * i1;
        }
    }
}

#endif  // __CUDA_ARCH__ >= 800

}  // namespace

void hca_q_rope(float* q, const int32_t* pos, int n_tokens, int n_heads, int head_dim, float rms_eps,
                const RopeTable& rt, void* stream) {
    if (n_tokens <= 0) return;
    hca_q_norm_rope_kernel<<<dim3(n_heads, n_tokens), 32, 0, (cudaStream_t) stream>>>(q, pos, n_heads, head_dim,
                                                                                      rms_eps, rt);
}

void hca_inv_rope(float* o, const int32_t* pos, int n_tokens, int n_heads, int head_dim, const RopeTable& rt,
                  void* stream) {
    if (n_tokens <= 0) return;
    hca_inv_rope_kernel<<<dim3(n_heads, n_tokens), 32, 0, (cudaStream_t) stream>>>(o, pos, n_heads, head_dim, rt);
}

void hca_kv_norm_rope(const float* kv_in, const float* kv_norm_w, const int32_t* pos, int n_tokens, int head_dim,
                      float rms_eps, const RopeTable& rt, float* kv_rot, void* stream) {
    if (n_tokens <= 0) return;
    const size_t smem = (size_t) (head_dim + 16) * sizeof(float);
    hca_kv_norm_rope_kernel<<<n_tokens, 256, smem, (cudaStream_t) stream>>>(kv_in, kv_norm_w, pos, head_dim, rms_eps,
                                                                            rt, kv_rot);
}

void hca_win_commit(const float* kv_rot, const int32_t* pos, int n_tokens, int head_dim, int win_size,
                    uint8_t* win_cache, bool fp8, void* stream, int win_cap) {
    if (n_tokens <= 0) return;
    const int cap = win_cap > 0 ? win_cap : win_size;
    const int n_write = n_tokens < cap ? n_tokens : cap;
    const int start_t = n_tokens - n_write;
    const size_t smem = (size_t) (head_dim + 16) * sizeof(float);
    hca_win_commit_kernel<<<n_write, 256, smem, (cudaStream_t) stream>>>(kv_rot, pos, start_t, head_dim, cap,
                                                                         win_cache, fp8);
}

void hca_kv_store_window(const float* kv_in, const float* kv_norm_w, const int32_t* pos, int n_tokens, int head_dim,
                         int win_size, float rms_eps, const RopeTable& rt, uint8_t* win_cache, bool fp8, void* stream) {
    if (n_tokens <= 0) return;
    const int n_write = n_tokens < win_size ? n_tokens : win_size;
    const int start_t = n_tokens - n_write;
    const size_t smem = (size_t) (head_dim + 16) * sizeof(float);
    hca_kv_store_kernel<<<n_write, 256, smem, (cudaStream_t) stream>>>(kv_in, kv_norm_w, pos, start_t, head_dim,
                                                                       win_size, rms_eps, rt, win_cache, fp8);
}

void hca_compress(const float* ckv, const float* cscore, const float* ape, const float* norm_w, const int32_t* pos,
                  int n_tokens, int ratio, int head_dim, float rms_eps, const RopeTable& comp_rt,
                  float* state_kv, float* state_score, uint8_t* comp_cache, bool fp8, void* stream,
                  int64_t pos0_hint) {
    if (n_tokens <= 0) return;
    cudaStream_t st = (cudaStream_t) stream;
    const size_t smem = (size_t) (head_dim + 16) * sizeof(float);
    if (pos0_hint >= 0 && n_tokens >= ratio) {
        const int rem0 = (int) (pos0_hint % ratio);
        const int lead = rem0 == 0 ? 0 : std::min(n_tokens, ratio - rem0);
        if (lead > 0) {
            hca_compress_kernel<<<1, head_dim, smem, st>>>(ckv, cscore, ape, norm_w, pos, lead, ratio, head_dim,
                                                           rms_eps, comp_rt, state_kv, state_score, comp_cache, fp8);
        }
        const int left = n_tokens - lead;
        const int n_full = left / ratio;
        if (n_full > 0) {
            hca_compress_blocks_kernel<<<n_full, head_dim, smem, st>>>(
                ckv + (size_t) lead * head_dim, cscore + (size_t) lead * head_dim, ape, norm_w,
                (int) (pos0_hint + lead), ratio, head_dim, rms_eps, comp_rt, comp_cache, fp8);
        }
        const int tail = left - n_full * ratio;
        if (tail > 0) {
            const int off = lead + n_full * ratio;
            hca_compress_kernel<<<1, head_dim, smem, st>>>(
                ckv + (size_t) off * head_dim, cscore + (size_t) off * head_dim, ape, norm_w, pos + off, tail, ratio,
                head_dim, rms_eps, comp_rt, state_kv, state_score, comp_cache, fp8);
        }
        return;
    }
    bk::launch_pdl(hca_compress_kernel, dim3(1), dim3(head_dim), smem, st, ckv, cscore, ape, norm_w, pos,
                   n_tokens, ratio, head_dim, rms_eps, comp_rt, state_kv, state_score, comp_cache, fp8);
}

void hca_attention(const float* q, const uint8_t* comp_cache, const uint8_t* win_cache, const float* kv_rot,
                   const float* attn_sink, const int32_t* pos, int n_tokens, int n_heads, int head_dim, int win_size,
                   int ratio, bool fp8, float* out, void* stream, int win_cap) {
    if (n_tokens <= 0) return;
    const int cap = win_cap > 0 ? win_cap : win_size;
    if (head_dim == 512) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
        // the prefill pass: one tensor-core block per (32-head tile, token)
        if (n_tokens > 8) {
            // the dynamic shared-memory opt-in and the sm_80+ check, once per device and instantiation; a card that
            // rejects either keeps the scalar kernel
            static bk::mma_f16::MmaSmemCache smem_cache[2];
            const bool mma_usable =
                fp8 ? bk::mma_f16::mma_smem_ready(smem_cache[1], hca_attention_mma512_kernel<true>, mma512_smem())
                    : bk::mma_f16::mma_smem_ready(smem_cache[0], hca_attention_mma512_kernel<false>, mma512_smem());
            if (mma_usable) {
                const dim3 grid((n_heads + 16 * kMmaG - 1) / (16 * kMmaG), n_tokens);
                if (fp8)
                    hca_attention_mma512_kernel<true><<<grid, 256, mma512_smem(), (cudaStream_t) stream>>>(
                        q, comp_cache, win_cache, kv_rot, attn_sink, pos, n_heads, win_size, cap, ratio, out);
                else
                    hca_attention_mma512_kernel<false><<<grid, 256, mma512_smem(), (cudaStream_t) stream>>>(
                        q, comp_cache, win_cache, kv_rot, attn_sink, pos, n_heads, win_size, cap, ratio, out);
                return;
            }
        }
#endif
        hca_attention_gqa512_kernel<<<dim3((n_heads + 7) / 8, n_tokens), 256, 0, (cudaStream_t) stream>>>(
            q, comp_cache, win_cache, kv_rot, attn_sink, pos, n_heads, win_size, cap, ratio, fp8, out);
        return;
    }
    const size_t smem = (size_t) (2 * head_dim + 16) * sizeof(float);
    hca_attention_kernel<<<dim3(n_heads, n_tokens), 256, smem, (cudaStream_t) stream>>>(
        q, comp_cache, win_cache, kv_rot, attn_sink, pos, n_heads, head_dim, win_size, cap, ratio, fp8, out);
}


// ---- decode windows: the positions split across blocks ---------------------------------------------------------
//
// A window of a few rows has 8 blocks a row (8 heads each) for every position: the card idles. Split s of
// kAttnSplits(rows) takes a contiguous run of the row's positions and writes each head's running max, sum and
// unnormalized output; the combine kernel merges the splits. The sink counts once (in split 0).

namespace {

__device__ __forceinline__ void attn_partial_store(float* part, int t, int h, int s, int n_heads, int n_split,
                                                   int lane, float m, float l, const float* acc) {
    float* dst = part + (((size_t) t * n_heads + h) * n_split + s) * kAttnPartFloats;
#pragma unroll
    for (int i = 0; i < 16; ++i) dst[2 + lane + 32 * i] = acc[i];
    if (lane == 0) {
        dst[0] = m;
        dst[1] = l;
    }
}

__global__ void hca_attention_split_kernel(const float* __restrict__ q, const uint8_t* __restrict__ comp_cache,
                                           const uint8_t* __restrict__ win_cache, const float* __restrict__ kv_rot,
                                           const float* __restrict__ attn_sink, const int32_t* __restrict__ pos,
                                           int n_heads, int win_size, int win_cap, int ratio, bool fp8, int n_split,
                                           float* __restrict__ part) {
    bk::pdl_wait();
    constexpr int kHeadDim = 512;
    constexpr int kTile = 4;
    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
    const int h = blockIdx.x * 8 + wid;
    const int s = blockIdx.y, t = blockIdx.z;
    const bool active = h < n_heads;
    const int p = pos[t], pos0 = pos[0];
    const int win_start = p - (win_size - 1) > 0 ? p - (win_size - 1) : 0;
    const int win_n = p - win_start + 1;
    const int n_comp = (p + 1) / ratio;
    const int n_pos = n_comp + win_n;
    const int chunk = (n_pos + n_split - 1) / n_split;
    const int i0 = s * chunk, i1 = min(n_pos, i0 + chunk);
    constexpr float scale = 0.04419417382415922f;   // 1 / sqrt(512)

    __shared__ float skv[kTile][kHeadDim];
    float q_reg[16], acc[16];
    const float* qrow = q + ((size_t) t * n_heads + (active ? h : 0)) * kHeadDim;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        q_reg[i] = active ? qrow[lane + 32 * i] : 0.0f;
        acc[i] = 0.0f;
    }
    float m = s == 0 && active ? attn_sink[h] : -INFINITY;
    float l = s == 0 ? 1.0f : 0.0f;
    for (int b0 = i0; b0 < i1; b0 += kTile) {
        const int cnt = min(kTile, i1 - b0);
        for (int r = 0; r < cnt; ++r) {
            const int i = b0 + r;
            if (i < n_comp) {
                skv[r][tid] = load_hca_elem(comp_cache, (size_t) i, tid, kHeadDim, fp8);
                skv[r][tid + 256] = load_hca_elem(comp_cache, (size_t) i, tid + 256, kHeadDim, fp8);
            } else {
                const int jp = win_start + (i - n_comp);
                if (kv_rot && jp >= pos0) {
                    const float* row = kv_rot + (size_t) (jp - pos0) * kHeadDim;
                    skv[r][tid] = row[tid];
                    skv[r][tid + 256] = row[tid + 256];
                } else {
                    const size_t slot = (size_t) (jp % win_cap);
                    skv[r][tid] = load_hca_elem(win_cache, slot, tid, kHeadDim, fp8);
                    skv[r][tid + 256] = load_hca_elem(win_cache, slot, tid + 256, kHeadDim, fp8);
                }
            }
        }
        __syncthreads();
        if (active) {
            for (int r = 0; r < cnt; ++r) {
                const float* kv = skv[r];
                float dot = 0.0f;
#pragma unroll
                for (int i = 0; i < 16; ++i) dot = fmaf(q_reg[i], kv[lane + 32 * i], dot);
                const float sc = warp_sum(dot) * scale;
                const float m_new = fmaxf(m, sc);
                const float alpha = expf(m - m_new);
                const float beta = expf(sc - m_new);
                l = l * alpha + beta;
                m = m_new;
#pragma unroll
                for (int i = 0; i < 16; ++i) acc[i] = fmaf(acc[i], alpha, beta * kv[lane + 32 * i]);
            }
        }
        __syncthreads();
    }
    if (active) attn_partial_store(part, t, h, s, n_heads, n_split, lane, m, l, acc);
}

// The FP8 cache's split kernel: kTile positions a step, their rows loaded by the whole block in 16-byte pieces (a row a
// warp) into shared memory, their 8 heads' scores side by side (one max and one rescale a step), a lane's columns
// 4 lane + 128 j (j < 4) read four at a time; the positions spread evenly over the splits.
template <int kTile>
__global__ void __launch_bounds__(256) hca_attention_split_fp8_kernel(const float* __restrict__ q,
                                                                      const uint8_t* __restrict__ comp_cache,
                                                                      const uint8_t* __restrict__ win_cache,
                                                                      const float* __restrict__ kv_rot,
                                                                      const float* __restrict__ attn_sink,
                                                                      const int32_t* __restrict__ pos, int n_heads,
                                                                      int win_size, int win_cap, int ratio, int n_split,
                                                                      float* __restrict__ part) {
    bk::pdl_wait();
    constexpr int kHeadDim = 512, kNope = kHeadDim - kRopeDim;
    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
    const int h = blockIdx.x * 8 + wid;
    const int s = blockIdx.y, t = blockIdx.z;
    const bool active = h < n_heads;
    const int p = pos[t], pos0 = pos[0];
    const int win_start = p - (win_size - 1) > 0 ? p - (win_size - 1) : 0;
    const int win_n = p - win_start + 1;
    const int n_comp = (p + 1) / ratio;
    const int n_pos = n_comp + win_n;
    const int chunk = (n_pos + n_split - 1) / n_split;
    const int i0 = s * chunk, i1 = min(n_pos, i0 + chunk);
    constexpr float scale = 0.04419417382415922f;   // 1 / sqrt(512)

    __shared__ __align__(16) float skv[kTile][kHeadDim];
    float4 qv[4], acc[4];
    const float4* qrow = reinterpret_cast<const float4*>(q + ((size_t) t * n_heads + (active ? h : 0)) * kHeadDim);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        qv[j] = active ? qrow[lane + 32 * j] : make_float4(0.f, 0.f, 0.f, 0.f);
        acc[j] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    float m = s == 0 && active ? attn_sink[h] : -INFINITY;
    float l = s == 0 ? 1.0f : 0.0f;
    for (int b0 = i0; b0 < i1; b0 += kTile) {
        const int cnt = min(kTile, i1 - b0);
        for (int r = wid; r < cnt; r += 8) {   // warp wid loads row r
            const int i = b0 + r;
            const uint8_t* row = nullptr;
            const float* frow = nullptr;
            if (i < n_comp) {
                row = comp_cache + (size_t) i * kHcaFp8Stride;
            } else {
                const int jp = win_start + (i - n_comp);
                if (kv_rot && jp >= pos0) frow = kv_rot + (size_t) (jp - pos0) * kHeadDim;
                else row = win_cache + (size_t) (jp % win_cap) * kHcaFp8Stride;
            }
            float4* dst = reinterpret_cast<float4*>(skv[r]);
            if (frow) {
#pragma unroll
                for (int k = 0; k < 4; ++k) dst[lane + 32 * k] = reinterpret_cast<const float4*>(frow)[lane + 32 * k];
            } else {
                if (lane < kNope / 16) {
                    const uint4 v = *reinterpret_cast<const uint4*>(row + 16 * lane);
                    const float sc = reinterpret_cast<const float*>(row + kNope)[lane >> 2];
                    const uint32_t w4[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const __nv_fp8x4_e4m3 f = *reinterpret_cast<const __nv_fp8x4_e4m3*>(&w4[k]);
                        const float4 x = static_cast<float4>(f);
                        dst[4 * lane + k] = make_float4(x.x * sc, x.y * sc, x.z * sc, x.w * sc);
                    }
                }
                const uint32_t pr = reinterpret_cast<const uint32_t*>(row + kNope + (kNope / 64) * 4)[lane];
                reinterpret_cast<float2*>(skv[r] + kNope)[lane] =
                    make_float2(__bfloat162float(__ushort_as_bfloat16((uint16_t) (pr & 0xffffu))),
                                __bfloat162float(__ushort_as_bfloat16((uint16_t) (pr >> 16))));
            }
        }
        __syncthreads();
        if (active) {
            float sc[kTile];
#pragma unroll
            for (int r = 0; r < kTile; ++r) {
                float dot = 0.0f;
                if (r < cnt) {
                    const float4* kr = reinterpret_cast<const float4*>(skv[r]);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float4 k = kr[lane + 32 * j];
                        dot = fmaf(qv[j].x, k.x, dot);
                        dot = fmaf(qv[j].y, k.y, dot);
                        dot = fmaf(qv[j].z, k.z, dot);
                        dot = fmaf(qv[j].w, k.w, dot);
                    }
                }
                sc[r] = dot;
            }
#pragma unroll
            for (int r = 0; r < kTile; ++r) sc[r] = r < cnt ? warp_sum(sc[r]) * scale : -INFINITY;
            float mx = m;
#pragma unroll
            for (int r = 0; r < kTile; ++r) mx = fmaxf(mx, sc[r]);
            const float alpha = expf(m - mx);
            float pr[kTile], ps = 0.0f;
#pragma unroll
            for (int r = 0; r < kTile; ++r) {
                pr[r] = r < cnt ? expf(sc[r] - mx) : 0.0f;
                ps += pr[r];
            }
            l = l * alpha + ps;
            m = mx;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float4 o = make_float4(acc[j].x * alpha, acc[j].y * alpha, acc[j].z * alpha, acc[j].w * alpha);
#pragma unroll
                for (int r = 0; r < kTile; ++r)
                    if (r < cnt) {
                        const float4 k = reinterpret_cast<const float4*>(skv[r])[lane + 32 * j];
                        o.x = fmaf(pr[r], k.x, o.x);
                        o.y = fmaf(pr[r], k.y, o.y);
                        o.z = fmaf(pr[r], k.z, o.z);
                        o.w = fmaf(pr[r], k.w, o.w);
                    }
                acc[j] = o;
            }
        }
        __syncthreads();
    }
    if (active) {
        float* dst = part + (((size_t) t * n_heads + h) * n_split + s) * kAttnPartFloats;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float* d = dst + 2 + 4 * lane + 128 * j;
            d[0] = acc[j].x;
            d[1] = acc[j].y;
            d[2] = acc[j].z;
            d[3] = acc[j].w;
        }
        if (lane == 0) {
            dst[0] = m;
            dst[1] = l;
        }
    }
}

// Block (h, t): head h of row t from its splits.
__global__ void attn_split_combine_kernel(const float* __restrict__ part, int n_heads, int n_split,
                                          float* __restrict__ out) {
    bk::pdl_wait();
    const int h = blockIdx.x, t = blockIdx.y;
    const float* src = part + ((size_t) t * n_heads + h) * n_split * kAttnPartFloats;
    float M = -INFINITY;
    for (int s = 0; s < n_split; ++s) M = fmaxf(M, src[(size_t) s * kAttnPartFloats]);
    float L = 0.0f;
    for (int s = 0; s < n_split; ++s) {
        const float* ps = src + (size_t) s * kAttnPartFloats;
        if (ps[1] > 0.0f) L += ps[1] * expf(ps[0] - M);
    }
    const float inv = 1.0f / L;
    float* orow = out + ((size_t) t * n_heads + h) * 512;
    for (int c = threadIdx.x; c < 512; c += blockDim.x) {
        float o = 0.0f;
        for (int s = 0; s < n_split; ++s) {
            const float* ps = src + (size_t) s * kAttnPartFloats;
            if (ps[1] > 0.0f) o = fmaf(ps[2 + c], expf(ps[0] - M), o);
        }
        orow[c] = o * inv;
    }
}

}  // namespace

void hca_attention_split(const float* q, const uint8_t* comp_cache, const uint8_t* win_cache, const float* kv_rot,
                         const float* attn_sink, const int32_t* pos, int n_tokens, int n_heads, int win_size,
                         int ratio, bool fp8, float* out, float* part, void* stream, int win_cap) {
    if (n_tokens <= 0) return;
    const int cap = win_cap > 0 ? win_cap : win_size;
    const int ns = attn_splits(n_tokens);
    if (fp8)
        bk::launch_pdl(hca_attention_split_fp8_kernel<8>, dim3((n_heads + 7) / 8, ns, n_tokens), dim3(256), 0,
                       (cudaStream_t) stream, q, comp_cache, win_cache, kv_rot, attn_sink, pos, n_heads, win_size, cap,
                       ratio, ns, part);
    else
        bk::launch_pdl(hca_attention_split_kernel, dim3((n_heads + 7) / 8, ns, n_tokens), dim3(256), 0, (cudaStream_t) stream,
                       q, comp_cache, win_cache, kv_rot, attn_sink, pos, n_heads, win_size, cap, ratio, fp8, ns, part);
    bk::launch_pdl(attn_split_combine_kernel, dim3(n_heads, n_tokens), dim3(128), 0, (cudaStream_t) stream, (const float*) part,
                   n_heads, ns, out);
}

namespace {

// hca_q_norm_rope_kernel on the q rows, a warp a head and 8 heads a block (blocks x < n_qb), and block (n_qb, t)
// hca_kv_norm_rope_kernel's row t, then hca_win_commit_kernel's store of it.
__global__ void __launch_bounds__(256) hca_qkv_rope_commit_kernel(float* q, const float* kv_in,
                                                                  const float* __restrict__ norm_w, const int32_t* pos,
                                                                  int n_heads, int head_dim, float rms_eps, RopeTable rt,
                                                                  float* kv_rot, int win_cap, uint8_t* win_cache,
                                                                  bool fp8, int n_qb) {
    bk::pdl_wait();
    const int t = blockIdx.y, tid = threadIdx.x;
    if ((int) blockIdx.x < n_qb) {
        const int h = blockIdx.x * 8 + (tid >> 5), pr = tid & 31;
        if (h >= n_heads) return;
        float* head_ptr = q + ((size_t) t * n_heads + h) * head_dim;
        hca_q_norm_rope_warp(head_ptr, head_dim, (float) pos[t], rms_eps, rt, pr);
        return;
    }
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
    if (tid < kRopePairs) {
        const int idx = head_dim - kRopeDim + 2 * tid;
        float x0 = sh[idx], x1 = sh[idx + 1];
        rotate_pair(x0, x1, (float) pos[t], rt.step[tid], rt.mscale, false);
        sh[idx] = x0;
        sh[idx + 1] = x1;
    }
    __syncthreads();
    float* dst = kv_rot + (size_t) t * head_dim;
    for (int c = tid; c < head_dim; c += blockDim.x) dst[c] = sh[c];
    store_hca_row(win_cache, (size_t) (pos[t] % win_cap), sh, head_dim, fp8, red);
}

// native_mmvq.cu is built with --use_fast_math: its divisions are div.approx.ftz (gdn_window.cu's emulation)
__device__ __forceinline__ float div_approx_ftz(float a, float b) {
    float r;
    asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ int8_t q8_quant_fm(const float xi, const float d, const float amax) {
    if (amax == 0.0f) return 0;
    const float q = roundf(div_approx_ftz(xi, d));
    return (int8_t) (q > 127.0f ? 127.0f : q < -127.0f ? -127.0f : q);
}

struct Q81 {   // native_mmvq's q8_1 block
    half2 ds;
    int8_t qs[32];
};

// vha_postmix_kernel's block (128 columns of head h, row t), the inverse rotation of hca_rope_kernel on the last 64
// columns (a pair's two lanes each compute it), then native_quantize_q8_1 of the value at its place in the groups'
// layout ([G][T][heads_per_group * head_dim]).
__global__ void __launch_bounds__(128) hca_post_rope_q8_kernel(const float* in, const float* __restrict__ mix,
                                                               const int32_t* pos, int n_heads, int head_dim, int hpg,
                                                               int n_tokens, RopeTable rt, Q81* y) {
    bk::pdl_wait();
    const int d = blockIdx.x * blockDim.x + threadIdx.x, h = blockIdx.y, t = blockIdx.z, lane = threadIdx.x & 31;
    const float* base = in + (size_t) t * n_heads * head_dim + d;
    const float* mrow = mix + (size_t) h * n_heads;
    float acc = 0.0f;
    for (int g = 0; g < n_heads; ++g) acc = fmaf(mrow[g], base[(size_t) g * head_dim], acc);
    const int r0 = head_dim - kRopeDim;
    if ((int) (blockIdx.x * blockDim.x + blockDim.x) > r0) {   // the block holds rotated columns (whole warps)
        const float other = __shfl_xor_sync(0xffffffffu, acc, 1);
        if (d >= r0) {
            float x0 = (lane & 1) ? other : acc, x1 = (lane & 1) ? acc : other;
            rotate_pair(x0, x1, (float) pos[t], rt.step[(d - r0) >> 1], rt.mscale, true);
            acc = (lane & 1) ? x1 : x0;
        }
    }
    const int g = h / hpg, gin = hpg * head_dim;
    const size_t e = ((size_t) g * n_tokens + t) * gin + (size_t) (h % hpg) * head_dim + d;
    const float amax = warp_max(fabsf(acc));
    const float sum = warp_sum(acc);
    const float dq = bk::q8_1_finite(div_approx_ftz(amax, 127.0f));
    y[e / 32].qs[e % 32] = q8_quant_fm(acc, dq, amax);
    if ((e & 31) == 0) y[e / 32].ds = bk::q8_1_ds(dq, sum);
}

}  // namespace

void hca_qkv_rope_commit(float* q, const float* kv_in, const float* kv_norm_w, const int32_t* pos, int n_tokens,
                         int n_heads, int head_dim, float rms_eps, const RopeTable& rt, float* kv_rot, int win_cap,
                         uint8_t* win_cache, bool fp8, void* stream) {
    if (n_tokens <= 0) return;
    if (n_tokens > win_cap) throw std::invalid_argument("ernie55 hca_qkv_rope_commit: rows");
    const int n_qb = (n_heads + 7) / 8;
    const size_t smem = (size_t) (head_dim + 16) * sizeof(float);
    bk::launch_pdl(hca_qkv_rope_commit_kernel, dim3(n_qb + 1, n_tokens), dim3(256), smem, (cudaStream_t) stream, q, kv_in,
                   kv_norm_w, pos, n_heads, head_dim, rms_eps, rt, kv_rot, win_cap, win_cache, fp8, n_qb);
}

void hca_post_rope_q8(const float* in, const float* mix, const int32_t* pos, int n_tokens, int n_heads, int head_dim,
                      int heads_per_group, const RopeTable& rt, void* xq, void* stream) {
    if (n_tokens <= 0) return;
    if (head_dim % 128 || n_heads % heads_per_group) throw std::invalid_argument("ernie55 hca_post_rope_q8: geometry");
    bk::launch_pdl(hca_post_rope_q8_kernel, dim3(head_dim / 128, n_heads, n_tokens), dim3(128), 0, (cudaStream_t) stream, in,
                   mix, pos, n_heads, head_dim, heads_per_group, n_tokens, rt, (Q81*) xq);
}

}  // namespace basalt::models::ernie55
