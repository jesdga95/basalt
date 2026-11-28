// src/models/ernie55/kernels/mla_dsa.cu - ERNIE-5.5 MLA-DSA + Lightning Indexer CUDA kernels:
//   - Q/K interleaved RoPE + latent KV cache store (FP8 fp8_ds_mla or FP16)
//   - Lightning Indexer prepare: LayerNorm(k), partial NeoX RoPE (leading 64 dims), orthonormal Hadamard-128,
//     MXFP4 (FP8 mode) or FP16 indexer K cache store
//   - Lightning Indexer scoring (ReLU(iq . ik) * weights * (1 / sqrt(idx_dim * idx_heads))), top-K over [0..pos-win),
//     merged with trailing sliding window [max(0, pos-win+1)..pos]
//   - Absorbed sparse MLA attention over selected positions with per-head softmax_offset sink and Q8_0 kv_b_proj
#include "basalt/models/ernie55/kernels.hpp"
#include "basalt/kernels/mma_f16.cuh"
#include "basalt/kernels/mxfp4_scale.hpp"
#include "basalt/models/ernie55/cache_layout.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>


#include <cfloat>
#include <cmath>

namespace basalt::models::ernie55 {
namespace {

constexpr int kDsaFp8RowBytes = layout::kDsaFp8RowBytes;
constexpr int kIdxFp4RowBytes = layout::kIdxFp4RowBytes;

__device__ __constant__ float kE2M1[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

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

__device__ __forceinline__ void rotate_pair(float& x0, float& x1, float pos, float step) {
    float s, c;
    sincosf(pos * step, &s, &c);
    const float a = x0 * c - x1 * s;
    const float b = x0 * s + x1 * c;
    x0 = a;
    x1 = b;
}

__device__ __forceinline__ uint8_t quant_e2m1(float x) {
    const uint8_t sign = x < 0.0f ? 8u : 0u;
    const float a = fabsf(x);
    uint8_t mag = 0;
    if (a >= 5.0f) mag = 7;
    else if (a >= 3.5f) mag = 6;
    else if (a >= 2.5f) mag = 5;
    else if (a >= 1.75f) mag = 4;
    else if (a >= 1.25f) mag = 3;
    else if (a >= 0.75f) mag = 2;
    else if (a >= 0.25f) mag = 1;
    return sign | mag;
}

__global__ void dsa_q_rope_kernel(float* __restrict__ q, const int32_t* __restrict__ pos, int n_heads,
                                  int qk_nope, int qk_rope, RopeTable rt) {
    const int pr = threadIdx.x;
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    if (pr >= qk_rope / 2) return;
    const int qd = qk_nope + qk_rope;
    float* base = q + ((size_t) t * n_heads + h) * qd + qk_nope + 2 * pr;
    float x0 = base[0], x1 = base[1];
    rotate_pair(x0, x1, (float) pos[t], rt.step[pr]);
    base[0] = x0;
    base[1] = x1;
}

__global__ void dsa_kv_store_kernel(const float* __restrict__ kv_a, const float* __restrict__ kv_norm_w,
                                    const int32_t* __restrict__ pos, int kv_lora, int qk_rope, float rms_eps,
                                    RopeTable rt, uint8_t* __restrict__ kv_cache, bool fp8) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    extern __shared__ float sh[];   // [kv_lora + qk_rope + 16]
    float* rp = sh + kv_lora;
    float* red = rp + qk_rope;
    const int kv_dim = kv_lora + qk_rope;
    const float* src = kv_a + (size_t) t * kv_dim;

    float ss = 0.0f;
    for (int c = tid; c < kv_lora; c += blockDim.x) {
        const float v = src[c];
        sh[c] = v;
        ss += v * v;
    }
    for (int c = tid; c < qk_rope; c += blockDim.x) rp[c] = src[kv_lora + c];
    ss = warp_sum(ss);
    if ((tid & 31) == 0) red[tid >> 5] = ss;
    __syncthreads();
    float tot = 0.0f;
    for (int w = 0; w < blockDim.x / 32; ++w) tot += red[w];
    const float inv = rsqrtf(tot / (float) kv_lora + rms_eps);
    __syncthreads();

    for (int c = tid; c < kv_lora; c += blockDim.x) sh[c] = sh[c] * inv * kv_norm_w[c];
    if (tid < qk_rope / 2) {
        float x0 = rp[2 * tid], x1 = rp[2 * tid + 1];
        rotate_pair(x0, x1, (float) pos[t], rt.step[tid]);
        rp[2 * tid] = x0;
        rp[2 * tid + 1] = x1;
    }
    __syncthreads();

    const size_t cell = (size_t) pos[t];
    if (!fp8) {
        __half* dst = reinterpret_cast<__half*>(kv_cache) + cell * (size_t) kv_dim;
        for (int c = tid; c < kv_lora; c += blockDim.x) dst[c] = __float2half_rn(sh[c]);
        for (int c = tid; c < qk_rope; c += blockDim.x) dst[kv_lora + c] = __float2half_rn(rp[c]);
    } else {
        // fp8_ds_mla: 512 E4M3 codes (4 groups of 128), 4 fp32 scales, 64 BF16 rope
        uint8_t* row = kv_cache + cell * (size_t) kDsaFp8RowBytes;
        float* scales = reinterpret_cast<float*>(row + kv_lora);
        uint16_t* rope_out = reinterpret_cast<uint16_t*>(row + kv_lora + (kv_lora / 128) * 4);
        const int ngrp = kv_lora / 128;
        for (int g = tid / 32; g < ngrp; g += blockDim.x / 32) {
            const int lane = tid & 31;
            float vals[4];
            float am = 0.0f;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                vals[k] = __bfloat162float(__float2bfloat16_rn(sh[g * 128 + k * 32 + lane]));
                am = fmaxf(am, fabsf(vals[k]));
            }
            am = warp_max(am);
            const float sc = exp2f(ceilf(log2f(fmaxf(am / 448.0f, 1e-4f))));
            if (lane == 0) scales[g] = sc;
            const float inv_s = 1.0f / sc;
#pragma unroll
            for (int k = 0; k < 4; ++k)
                row[g * 128 + k * 32 + lane] = __nv_cvt_float_to_fp8(vals[k] * inv_s, __NV_SATFINITE, __NV_E4M3);
        }
        for (int c = tid; c < qk_rope; c += blockDim.x)
            rope_out[c] = __bfloat16_as_ushort(__float2bfloat16_rn(rp[c]));
    }
}

// Partial NeoX RoPE on leading `rope_dim` (64) elements of each row [n_rows, idx_dim]:
// pairs (i, i + rope_dim / 2) for i in 0..31.
__global__ void neox_partial_rope_kernel(float* __restrict__ x, const int32_t* __restrict__ pos,
                                         int rows_per_tok, int idx_dim, int rope_dim, RopeTable rt) {
    const int half = rope_dim / 2;
    const int i = threadIdx.x;
    const int r = blockIdx.x;
    const int t = blockIdx.y;
    if (i >= half) return;
    float* row = x + ((size_t) t * rows_per_tok + r) * idx_dim;
    float x0 = row[i], x1 = row[i + half];
    rotate_pair(x0, x1, (float) pos[t], rt.step[i]);
    row[i] = x0;
    row[i + half] = x1;
}

__global__ void store_idx_k_kernel(const float* __restrict__ ik, const int32_t* __restrict__ pos, int idx_dim,
                                   uint8_t* __restrict__ idx_k_cache, bool fp8) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;   // 0..127
    const float* src = ik + (size_t) t * idx_dim;
    const size_t cell = (size_t) pos[t];
    if (!fp8) {
        __half* dst = reinterpret_cast<__half*>(idx_k_cache) + cell * (size_t) idx_dim;
        if (tid < idx_dim) dst[tid] = __float2half_rn(src[tid]);
        return;
    }
    // MXFP4 (32-element blocks, 4 blocks for idx_dim=128): 64 packed bytes (even/odd nibbles) + 4 UE8M0 bytes
    __shared__ float vals[128];
    __shared__ uint8_t nibs[128];
    vals[tid] = src[tid];
    const int blk = tid >> 5;
    const int lane = tid & 31;
    const float am = warp_max(fabsf(vals[tid]));
    // E8M0 exponent: scale = 2^(e - 127) >= am / 6.0
    int exp_i = 127;
    if (am > 0.0f) {
        exp_i = (int) ceilf(log2f(am / 6.0f)) + 127;
        exp_i = max(0, min(254, exp_i));
    } else {
        exp_i = 0;
    }
    const float sc = ldexpf(1.0f, exp_i - 127);
    const float inv = sc > 0.0f ? (1.0f / sc) : 0.0f;
    nibs[tid] = quant_e2m1(vals[tid] * inv);
    __syncthreads();

    uint8_t* row = idx_k_cache + cell * (size_t) kIdxFp4RowBytes;
    if (tid < 64) row[tid] = nibs[2 * tid] | (uint8_t) (nibs[2 * tid + 1] << 4);
    if (lane == 0 && blk < 4) row[64 + blk] = (uint8_t) exp_i;
}

__device__ __forceinline__ float load_idx_k_elem(const uint8_t* cache, size_t cell, int d, int idx_dim, bool fp8) {
    if (!fp8) {
        const __half* src = reinterpret_cast<const __half*>(cache) + cell * (size_t) idx_dim;
        return __half2float(src[d]);
    }
    const uint8_t* row = cache + cell * (size_t) kIdxFp4RowBytes;
    const uint8_t byte = row[d >> 1];
    const uint8_t nib = (d & 1) ? (byte >> 4) : (byte & 0x0F);
    const uint8_t e = row[64 + (d >> 5)];
    const float sc = ldexpf(1.0f, (int) e - 127);
    return kE2M1[nib] * sc;
}

// Fast shared-Q, warp-per-key Lightning Indexer scorer for idx_dim == 128 && idx_heads <= 64.
// Loads all iq[t, 0..idx_heads, 0..128) ONCE into shared memory (32 KB for 64 heads) and scores keys with
// 4 registers/lane (zero local-memory spill, zero global Q reads in the key loop, bank-conflict-free layout).
__global__ void dsa_indexer_score_fast128_kernel(const float* __restrict__ iq, const float* __restrict__ iweights,
                                                 const uint8_t* __restrict__ idx_k_cache, const int32_t* __restrict__ pos,
                                                 int idx_heads, int top_k, int win_size, bool fp8,
                                                 float* __restrict__ scores_buf, int scores_stride) {
    constexpr int kDim = 128;
    const int t = blockIdx.y;
    const int p = pos[t];
    const int seq_len = p + 1;
    const int older_n = seq_len > win_size ? (seq_len - win_size) : 0;
    if (older_n <= top_k) return;

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;

    __shared__ float sq[64][kDim];
    __shared__ float sw[64];

    const float wscale = rsqrtf((float) (kDim * idx_heads));
    if (tid < idx_heads) sw[tid] = iweights[(size_t) t * idx_heads + tid] * wscale;

    const float* iq_t = iq + (size_t) t * idx_heads * kDim;
    const int total_q = idx_heads * kDim;
    for (int i = tid; i < total_q; i += blockDim.x) {
        sq[i >> 7][i & 127] = iq_t[i];
    }
    __syncthreads();

    float* out_s = scores_buf + (size_t) t * scores_stride;
    const int gw = blockIdx.x * 8 + wid;
    const int gstride = gridDim.x * 8;

    for (int j = gw; j < older_n; j += gstride) {
        float k0, k1, k2, k3;
        if (!fp8) {
            const __half* row = reinterpret_cast<const __half*>(idx_k_cache) + (size_t) j * kDim;
            k0 = __half2float(row[lane]);
            k1 = __half2float(row[lane + 32]);
            k2 = __half2float(row[lane + 64]);
            k3 = __half2float(row[lane + 96]);
        } else {
            const uint8_t* row = idx_k_cache + (size_t) j * (size_t) kIdxFp4RowBytes;
            const float sc0 = ldexpf(1.0f, (int) row[64] - 127);
            const float sc1 = ldexpf(1.0f, (int) row[65] - 127);
            const float sc2 = ldexpf(1.0f, (int) row[66] - 127);
            const float sc3 = ldexpf(1.0f, (int) row[67] - 127);
            const uint8_t b0 = row[lane >> 1];
            const uint8_t b1 = row[16 + (lane >> 1)];
            const uint8_t b2 = row[32 + (lane >> 1)];
            const uint8_t b3 = row[48 + (lane >> 1)];
            const bool hi = (lane & 1) != 0;
            k0 = kE2M1[hi ? (b0 >> 4) : (b0 & 0x0F)] * sc0;
            k1 = kE2M1[hi ? (b1 >> 4) : (b1 & 0x0F)] * sc1;
            k2 = kE2M1[hi ? (b2 >> 4) : (b2 & 0x0F)] * sc2;
            k3 = kE2M1[hi ? (b3 >> 4) : (b3 & 0x0F)] * sc3;
        }

        float total = 0.0f;
        for (int h = 0; h < idx_heads; ++h) {
            const float* qh = sq[h];
            float dot = fmaf(qh[lane], k0,
                        fmaf(qh[lane + 32], k1,
                        fmaf(qh[lane + 64], k2, qh[lane + 96] * k3)));
            dot = warp_sum(dot);
            if (dot > 0.0f) total = fmaf(dot, sw[h], total);
        }
        if (lane == 0) out_s[j] = total;
    }
}

// Multi-block single-pass scorer across all SMs: computes score_pos(j) once into scores_buf[t * stride + j]
// whenever older_n > top_k (zero work on fast path older_n <= top_k).
__global__ void dsa_indexer_score_kernel(const float* __restrict__ iq, const float* __restrict__ iweights,
                                         const uint8_t* __restrict__ idx_k_cache, const int32_t* __restrict__ pos,
                                         int idx_heads, int idx_dim, int top_k, int win_size, bool fp8,
                                         float* __restrict__ scores_buf, int scores_stride) {
    const int t = blockIdx.y;
    const int p = pos[t];
    const int seq_len = p + 1;
    const int older_n = seq_len > win_size ? (seq_len - win_size) : 0;
    if (older_n <= top_k) return;

    __shared__ float sw[64];
    const float wscale = rsqrtf((float) (idx_dim * idx_heads));
    if (threadIdx.x < idx_heads) sw[threadIdx.x] = iweights[(size_t) t * idx_heads + threadIdx.x] * wscale;
    __syncthreads();

    const float* iq_t = iq + (size_t) t * idx_heads * idx_dim;
    float* out_s = scores_buf + (size_t) t * scores_stride;
    const int step = gridDim.x * blockDim.x;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < older_n; j += step) {
        float k_reg[128];
        for (int d = 0; d < idx_dim; ++d) k_reg[d] = load_idx_k_elem(idx_k_cache, (size_t) j, d, idx_dim, fp8);
        float total = 0.0f;
        for (int h = 0; h < idx_heads; ++h) {
            const float* qh = iq_t + (size_t) h * idx_dim;
            float dot = 0.0f;
            for (int d = 0; d < idx_dim; ++d) dot += qh[d] * k_reg[d];
            if (dot > 0.0f) total += dot * sw[h];
        }
        out_s[j] = total;
    }
}

__device__ __forceinline__ uint32_t fp32_to_ordered(float f) {
    uint32_t u = __float_as_uint(f);
    if (u == 0x80000000u) u = 0u;   // normalize -0.0f -> +0.0f
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Fast exact top-K selection reading precomputed FP32 scores from scores_buf via 256-bin order-preserving radix
// histogram (<= 4 histogram scans + 1 emit scan vs 23 scans in 20-step binary search, bit-exact on all finite FP32).
__global__ void dsa_indexer_select_cached_kernel(const float* __restrict__ scores_buf, int scores_stride,
                                                 const int32_t* __restrict__ pos, int top_k, int win_size,
                                                 int32_t* __restrict__ out_idx) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;   // blockDim.x == 256
    const int width = top_k + win_size;
    int32_t* dst = out_idx + (size_t) t * width;
    const int p = pos[t];
    const int seq_len = p + 1;
    const int older_n = seq_len > win_size ? (seq_len - win_size) : 0;
    const int swa_n = seq_len < win_size ? seq_len : win_size;
    const int keep_k = older_n < top_k ? older_n : top_k;

    if (older_n <= top_k) {
        for (int i = tid; i < width; i += blockDim.x) {
            if (i < seq_len) dst[i] = i;
            else dst[i] = -1;
        }
        return;
    }

    const float* sc_row = scores_buf + (size_t) t * scores_stride;
    __shared__ uint32_t s_hist[256];
    __shared__ uint32_t s_prefix;
    __shared__ uint32_t s_mask;
    __shared__ int s_gt_count;
    __shared__ int s_done;
    __shared__ int s_gt_pos;
    __shared__ int s_eq_pos;

    if (tid == 0) {
        s_prefix = 0u;
        s_mask = 0u;
        s_gt_count = 0;
        s_done = 0;
    }
    __syncthreads();

    for (int shift = 24; shift >= 0; shift -= 8) {
        if (s_done) break;
        const uint32_t cur_prefix = s_prefix;
        const uint32_t cur_mask = s_mask;
        s_hist[tid] = 0u;
        __syncthreads();

        for (int j = tid; j < older_n; j += blockDim.x) {
            const uint32_t key = fp32_to_ordered(sc_row[j]);
            if ((key & cur_mask) == cur_prefix) {
                atomicAdd(&s_hist[(key >> shift) & 0xffu], 1u);
            }
        }
        __syncthreads();

        if (tid == 0) {
            int sum = s_gt_count;
            uint32_t sel_b = 0u;
            int bin_cnt = 0;
            for (int b = 255; b >= 0; --b) {
                const int c = (int) s_hist[b];
                if (sum + c >= keep_k) {
                    sel_b = (uint32_t) b;
                    bin_cnt = c;
                    break;
                }
                sum += c;
            }
            s_gt_count = sum;
            s_prefix = cur_prefix | (sel_b << shift);
            s_mask = cur_mask | (0xffu << shift);
            if (sum + bin_cnt == keep_k) s_done = 1;
        }
        __syncthreads();
    }

    const uint32_t prefix = s_prefix;
    const uint32_t prefix_hi = prefix | ~s_mask;
    __shared__ int s_warp_gt[8];
    __shared__ int s_warp_eq[8];
    if (tid == 0) {
        s_gt_pos = 0;
        s_eq_pos = s_gt_count;
    }
    __syncthreads();

    // Deterministic compaction in ascending index order (ties at the k-th score break by lowest index).
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const uint32_t lane_lt = (1u << lane) - 1u;
    for (int base = 0; base < older_n; base += blockDim.x) {
        const int j = base + tid;
        bool is_gt = false;
        bool is_eq = false;
        if (j < older_n) {
            const uint32_t key = fp32_to_ordered(sc_row[j]);
            is_gt = key > prefix_hi;
            is_eq = !is_gt && (key >= prefix);
        }
        const uint32_t gt_mask = __ballot_sync(0xffffffffu, is_gt);
        const uint32_t eq_mask = __ballot_sync(0xffffffffu, is_eq);
        if (lane == 0) {
            s_warp_gt[warp] = __popc(gt_mask);
            s_warp_eq[warp] = __popc(eq_mask);
        }
        __syncthreads();
        if (tid == 0) {
            int gp = s_gt_pos;
            int ep = s_eq_pos;
            for (int w = 0; w < 8; ++w) {
                const int gc = s_warp_gt[w];
                const int ec = s_warp_eq[w];
                s_warp_gt[w] = gp;
                s_warp_eq[w] = ep;
                gp += gc;
                ep += ec;
            }
            s_gt_pos = gp;
            s_eq_pos = ep;
        }
        __syncthreads();
        if (is_gt) {
            const int slot = s_warp_gt[warp] + __popc(gt_mask & lane_lt);
            if (slot < keep_k) dst[slot] = j;
        } else if (is_eq) {
            const int slot = s_warp_eq[warp] + __popc(eq_mask & lane_lt);
            if (slot < keep_k) dst[slot] = j;
        }
        __syncthreads();
    }

    for (int i = tid; i < width - keep_k; i += blockDim.x) {
        if (i < swa_n) dst[keep_k + i] = seq_len - swa_n + i;
        else dst[keep_k + i] = -1;
    }
}

// Dequantize one Q8_0 row (block_size = 32, 34 bytes/block: fp16 scale + 32 int8) dot with float vector `x`.
__device__ __forceinline__ float q8_0_row_dot(const uint8_t* row, const float* x, int cols) {
    const int nb = cols >> 5;
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = row + (size_t) b * 34;
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
        float s = 0.0f;
#pragma unroll
        for (int k = 0; k < 32; ++k) s = fmaf((float) qs[k], x[b * 32 + k], s);
        acc = fmaf(d, s, acc);
    }
    return acc;
}

__device__ __forceinline__ void load_dsa_kv_row(const uint8_t* kv_cache, size_t cell, int kv_lora, int qk_rope,
                                                bool fp8, float* c_kv, float* k_pe) {
    const int tid = threadIdx.x;
    if (!fp8) {
        const __half* row = reinterpret_cast<const __half*>(kv_cache) + cell * (size_t) (kv_lora + qk_rope);
        for (int c = tid; c < kv_lora; c += blockDim.x) c_kv[c] = __half2float(row[c]);
        for (int c = tid; c < qk_rope; c += blockDim.x) k_pe[c] = __half2float(row[kv_lora + c]);
    } else {
        const uint8_t* row = kv_cache + cell * (size_t) kDsaFp8RowBytes;
        const float* scales = reinterpret_cast<const float*>(row + kv_lora);
        const uint16_t* rope_in = reinterpret_cast<const uint16_t*>(row + kv_lora + (kv_lora / 128) * 4);
        for (int c = tid; c < kv_lora; c += blockDim.x) {
            const __nv_fp8_e4m3 q = *reinterpret_cast<const __nv_fp8_e4m3*>(row + c);
            c_kv[c] = (float) q * scales[c >> 7];
        }
        for (int c = tid; c < qk_rope; c += blockDim.x)
            k_pe[c] = __bfloat162float(__ushort_as_bfloat16(rope_in[c]));
    }
}

// Absorbed sparse MLA attention: one block (256 threads) per (head, token).
//   1. Absorb W_UK[h]^T into q_nope[h] -> q_lat[512]
//   2. Online softmax over selected positions sel_idx[t, 0..sel_width) with attn_sink[h] -> o_lat[512]
//   3. Project o[h] = W_UV[h] @ o_lat[512] -> out[t, h, v_dim]
__global__ void dsa_mla_attention_kernel(const float* __restrict__ q, const uint8_t* __restrict__ kv_cache,
                                         const uint8_t* __restrict__ kv_b_q8, const float* __restrict__ attn_sink,
                                         const int32_t* __restrict__ sel_idx, int n_heads, int qk_nope, int qk_rope,
                                         int kv_lora, int v_dim, int sel_width, bool fp8, float* __restrict__ out) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int tid = threadIdx.x;
    const int qd = qk_nope + qk_rope;
    const size_t q8_row_bytes = (size_t) (kv_lora / 32) * 34;
    const uint8_t* head_kv_b = kv_b_q8 + (size_t) h * (size_t) (qk_nope + v_dim) * q8_row_bytes;
    const uint8_t* w_uk = head_kv_b;
    const uint8_t* w_uv = head_kv_b + (size_t) qk_nope * q8_row_bytes;

    extern __shared__ float smem[];
    float* q_nope = smem;                       // [qk_nope]
    float* q_pe = q_nope + qk_nope;             // [qk_rope]
    float* q_lat = q_pe + qk_rope;              // [kv_lora]
    float* o_lat = q_lat + kv_lora;             // [kv_lora]
    float* c_kv = o_lat + kv_lora;              // [kv_lora]
    float* k_pe = c_kv + kv_lora;               // [qk_rope]
    float* red = k_pe + qk_rope;                // [16]

    const float* qrow = q + ((size_t) t * n_heads + h) * qd;
    for (int i = tid; i < qk_nope; i += blockDim.x) q_nope[i] = qrow[i];
    for (int i = tid; i < qk_rope; i += blockDim.x) q_pe[i] = qrow[qk_nope + i];
    for (int c = tid; c < kv_lora; c += blockDim.x) { q_lat[c] = 0.0f; o_lat[c] = 0.0f; }
    __syncthreads();

    // 1. Absorb W_UK^T: q_lat[c] = sum_{r=0}^{qk_nope-1} q_nope[r] * W_UK[r, c]
    for (int c = tid; c < kv_lora; c += blockDim.x) {
        const int blk = c >> 5;
        const int off = c & 31;
        float acc = 0.0f;
        for (int r = 0; r < qk_nope; ++r) {
            const uint8_t* b = w_uk + (size_t) r * q8_row_bytes + (size_t) blk * 34;
            const float d = __half2float(*reinterpret_cast<const __half*>(b));
            const int8_t qv = reinterpret_cast<const int8_t*>(b + 2)[off];
            acc = fmaf(q_nope[r], d * (float) qv, acc);
        }
        q_lat[c] = acc;
    }
    __syncthreads();

    // 2. Online softmax over selected positions with softmax_offset sink
    const float scale = rsqrtf((float) qd);
    float m = attn_sink[h];
    float l = 1.0f;
    const int32_t* sel = sel_idx + (size_t) t * sel_width;

    for (int s_i = 0; s_i < sel_width; ++s_i) {
        const int32_t cell = sel[s_i];
        if (cell < 0) break;
        load_dsa_kv_row(kv_cache, (size_t) cell, kv_lora, qk_rope, fp8, c_kv, k_pe);
        __syncthreads();

        float dot = 0.0f;
        for (int c = tid; c < kv_lora; c += blockDim.x) dot = fmaf(q_lat[c], c_kv[c], dot);
        for (int c = tid; c < qk_rope; c += blockDim.x) dot = fmaf(q_pe[c], k_pe[c], dot);
        dot = warp_sum(dot);
        if ((tid & 31) == 0) red[tid >> 5] = dot;
        __syncthreads();
        float sc = 0.0f;
        for (int w = 0; w < blockDim.x / 32; ++w) sc += red[w];
        sc *= scale;

        const float m_new = fmaxf(m, sc);
        const float alpha = expf(m - m_new);
        const float beta = expf(sc - m_new);
        l = l * alpha + beta;
        m = m_new;
        for (int c = tid; c < kv_lora; c += blockDim.x) o_lat[c] = o_lat[c] * alpha + beta * c_kv[c];
        __syncthreads();
    }

    const float inv_l = 1.0f / l;
    for (int c = tid; c < kv_lora; c += blockDim.x) o_lat[c] *= inv_l;
    __syncthreads();

    // 3. Project o[h, r] = W_UV[r, :] . o_lat[:]
    float* orow = out + ((size_t) t * n_heads + h) * v_dim;
    for (int r = tid; r < v_dim; r += blockDim.x) {
        orow[r] = q8_0_row_dot(w_uv + (size_t) r * q8_row_bytes, o_lat, kv_lora);
    }
}

// Fast GQA-tiled absorbed sparse MLA attention for ERNIE-5.5 shapes (kv_lora=512, qk_rope=64, qk_nope=192/128, v_dim=256/128):
// 8 warps (256 threads) = 8 query heads per block sharing 1 latent KV cache row across 4-row tiles in shared memory.
// Cuts scattered global KV cache reads by 8x and synchronization barriers by 4x.
template <int kNope, int kVdim>
__global__ void dsa_mla_attention_gqa512_kernel(const float* __restrict__ q, const uint8_t* __restrict__ kv_cache,
                                                const uint8_t* __restrict__ kv_b_q8, const float* __restrict__ attn_sink,
                                                const int32_t* __restrict__ sel_idx, int n_heads, int sel_width,
                                                bool fp8, float* __restrict__ out) {
    constexpr int kRope = 64;
    constexpr int kLora = 512;
    constexpr int kQd = kNope + kRope;
    constexpr int kTile = 4;
    constexpr size_t kQ8RowBytes = (kLora / 32) * 34;

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int h = blockIdx.x * 8 + wid;
    const int t = blockIdx.y;
    const bool active = h < n_heads;

    __shared__ union {
        float q_nope[8][kNope];
        struct {
            float c_kv[kTile][kLora];
            float k_pe[kTile][kRope];
        } tile;
        float o_lat[8][kLora];
    } sm;

    const uint8_t* head_kv_b = active ? (kv_b_q8 + (size_t) h * (size_t) (kNope + kVdim) * kQ8RowBytes) : nullptr;
    const uint8_t* w_uk = head_kv_b;
    const uint8_t* w_uv = head_kv_b ? (head_kv_b + (size_t) kNope * kQ8RowBytes) : nullptr;

    float q_pe0 = 0.0f, q_pe1 = 0.0f;
    float q_lat[16];
    float o_lat[16];
    if (active) {
        const float* qrow = q + ((size_t) t * n_heads + h) * kQd;
        for (int i = lane; i < kNope; i += 32) sm.q_nope[wid][i] = qrow[i];
        q_pe0 = qrow[kNope + lane];
        q_pe1 = qrow[kNope + lane + 32];
    }
    __syncwarp();

    // 1. Absorb W_UK^T: lane owns c = lane + 32 * blk (blk = 0..15, off = lane) -> uniform block scale per warp!
    if (active) {
#pragma unroll
        for (int blk = 0; blk < 16; ++blk) {
            float acc = 0.0f;
            const uint8_t* col_base = w_uk + (size_t) blk * 34;
            for (int r = 0; r < kNope; ++r) {
                const uint8_t* b = col_base + (size_t) r * kQ8RowBytes;
                const float d = __half2float(*reinterpret_cast<const __half*>(b));
                const int8_t qv = reinterpret_cast<const int8_t*>(b + 2)[lane];
                acc = fmaf(sm.q_nope[wid][r], d * (float) qv, acc);
            }
            q_lat[blk] = acc;
            o_lat[blk] = 0.0f;
        }
    } else {
#pragma unroll
        for (int blk = 0; blk < 16; ++blk) {
            q_lat[blk] = 0.0f;
            o_lat[blk] = 0.0f;
        }
    }
    __syncthreads();

    // 2. Online softmax over selected positions in 4-row shared-memory tiles
    const float scale = rsqrtf((float) kQd);
    float m = active ? attn_sink[h] : 0.0f;
    float l = 1.0f;
    const int32_t* sel = sel_idx + (size_t) t * sel_width;

    for (int s0 = 0; s0 < sel_width; s0 += kTile) {
        int cnt = 0;
        int32_t cells[kTile];
#pragma unroll
        for (int r = 0; r < kTile; ++r) {
            if (s0 + r < sel_width) {
                const int32_t c = sel[s0 + r];
                if (c >= 0) {
                    cells[cnt++] = c;
                    continue;
                }
            }
            break;
        }
        if (cnt == 0) break;

        for (int r = 0; r < cnt; ++r) {
            const size_t cell = (size_t) cells[r];
            if (!fp8) {
                const __half* row = reinterpret_cast<const __half*>(kv_cache) + cell * (size_t) (kLora + kRope);
                sm.tile.c_kv[r][tid] = __half2float(row[tid]);
                sm.tile.c_kv[r][tid + 256] = __half2float(row[tid + 256]);
                if (tid < kRope) sm.tile.k_pe[r][tid] = __half2float(row[kLora + tid]);
            } else {
                const uint8_t* row = kv_cache + cell * (size_t) kDsaFp8RowBytes;
                const float* scales = reinterpret_cast<const float*>(row + kLora);
                const uint16_t* rope_in = reinterpret_cast<const uint16_t*>(row + kLora + (kLora / 128) * 4);
                const __nv_fp8_e4m3 q0 = *reinterpret_cast<const __nv_fp8_e4m3*>(row + tid);
                const __nv_fp8_e4m3 q1 = *reinterpret_cast<const __nv_fp8_e4m3*>(row + tid + 256);
                sm.tile.c_kv[r][tid] = (float) q0 * scales[tid >> 7];
                sm.tile.c_kv[r][tid + 256] = (float) q1 * scales[(tid + 256) >> 7];
                if (tid < kRope) sm.tile.k_pe[r][tid] = __bfloat162float(__ushort_as_bfloat16(rope_in[tid]));
            }
        }
        __syncthreads();

        if (active) {
            for (int r = 0; r < cnt; ++r) {
                const float* ck = sm.tile.c_kv[r];
                const float* kp = sm.tile.k_pe[r];
                float dot = fmaf(q_pe0, kp[lane], q_pe1 * kp[lane + 32]);
#pragma unroll
                for (int blk = 0; blk < 16; ++blk) dot = fmaf(q_lat[blk], ck[lane + 32 * blk], dot);
                const float sc = warp_sum(dot) * scale;
                const float m_new = fmaxf(m, sc);
                const float alpha = expf(m - m_new);
                const float beta = expf(sc - m_new);
                l = l * alpha + beta;
                m = m_new;
#pragma unroll
                for (int blk = 0; blk < 16; ++blk) o_lat[blk] = fmaf(o_lat[blk], alpha, beta * ck[lane + 32 * blk]);
            }
        }
        __syncthreads();
        if (cnt < kTile) break;
    }

    // 3. Normalize o_lat into shared memory and project o[h, r] = W_UV[h, r, :] . o_lat[:]
    if (active) {
        const float inv_l = 1.0f / l;
#pragma unroll
        for (int blk = 0; blk < 16; ++blk) sm.o_lat[wid][lane + 32 * blk] = o_lat[blk] * inv_l;
    }
    __syncwarp();
    if (active) {
        float* orow = out + ((size_t) t * n_heads + h) * kVdim;
        for (int r = lane; r < kVdim; r += 32) {
            orow[r] = q8_0_row_dot(w_uv + (size_t) r * kQ8RowBytes, sm.o_lat[wid], kLora);
        }
    }
}

// Coalesced Q8_0 -> FP16 dequantizer: 8 warps/block, 1 warp = 1 Q8_0 block (32 elements).
__global__ void dequant_q8_0_to_f16_kernel(const uint8_t* __restrict__ src, uint16_t* __restrict__ dst,
                                           int64_t n_blocks) {
    const int64_t b = (int64_t) blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (b >= n_blocks) return;
    const uint8_t* blk = src + (size_t) b * 34;
    const float d = __half2float(*reinterpret_cast<const __half*>(blk));
    const int8_t qv = reinterpret_cast<const int8_t*>(blk + 2)[lane];
    dst[b * 32 + lane] = __half_as_ushort(__float2half_rn(d * (float) qv));
}

// Pack q_nope [n_tokens, n_heads, qk_nope + qk_rope] FP32 -> head-major [n_heads, n_tokens, qk_nope] FP16.
__global__ void dsa_pack_q_nope_f16_kernel(const float* __restrict__ q, uint16_t* __restrict__ dst, int n_tokens,
                                           int n_heads, int qk_nope, int qk_rope) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y;
    const int t = blockIdx.z;
    if (r >= qk_nope) return;
    const int qd = qk_nope + qk_rope;
    dst[((size_t) h * n_tokens + t) * qk_nope + r] =
        __half_as_ushort(__float2half_rn(q[((size_t) t * n_heads + h) * qd + r]));
}

// Pure pre-absorbed sparse MLA attention (kv_lora == 512, qk_rope == 64): W_UK^T and W_UV run outside via
// cuBLAS Tensor-Core batched GEMM, so this kernel does ZERO kv_b_q8 reads and uses 8-row shared-memory KV tiles.
__global__ void dsa_mla_attention_preabsorbed_gqa512_kernel(
    const float* __restrict__ q_lat_in, const float* __restrict__ q_full, const uint8_t* __restrict__ kv_cache,
    const float* __restrict__ attn_sink, const int32_t* __restrict__ sel_idx, int n_tokens, int n_heads,
    int qk_nope, int sel_width, bool fp8, uint16_t* __restrict__ o_lat_f16) {
    constexpr int kRope = 64;
    constexpr int kLora = 512;
    constexpr int kTile = 8;

    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int h = blockIdx.x * 8 + wid;
    const int t = blockIdx.y;
    const bool active = h < n_heads;

    __shared__ float s_ckv[kTile][kLora];
    __shared__ float s_kpe[kTile][kRope];

    float q_pe0 = 0.0f, q_pe1 = 0.0f;
    float q_lat[16];
    float o_lat[16];
    if (active) {
        const float* ql = q_lat_in + ((size_t) h * n_tokens + t) * kLora;
        const float* qp = q_full + ((size_t) t * n_heads + h) * (qk_nope + kRope) + qk_nope;
        q_pe0 = qp[lane];
        q_pe1 = qp[lane + 32];
#pragma unroll
        for (int blk = 0; blk < 16; ++blk) {
            q_lat[blk] = ql[lane + 32 * blk];
            o_lat[blk] = 0.0f;
        }
    } else {
#pragma unroll
        for (int blk = 0; blk < 16; ++blk) {
            q_lat[blk] = 0.0f;
            o_lat[blk] = 0.0f;
        }
    }

    const float scale = rsqrtf((float) (qk_nope + kRope));
    float m = active ? attn_sink[h] : 0.0f;
    float l = 1.0f;
    const int32_t* sel = sel_idx + (size_t) t * sel_width;

    for (int s0 = 0; s0 < sel_width; s0 += kTile) {
        int cnt = 0;
        int32_t cells[kTile];
#pragma unroll
        for (int r = 0; r < kTile; ++r) {
            if (s0 + r < sel_width) {
                const int32_t c = sel[s0 + r];
                if (c >= 0) {
                    cells[cnt++] = c;
                    continue;
                }
            }
            break;
        }
        if (cnt == 0) break;

        for (int r = 0; r < cnt; ++r) {
            const size_t cell = (size_t) cells[r];
            if (!fp8) {
                const __half* row = reinterpret_cast<const __half*>(kv_cache) + cell * (size_t) (kLora + kRope);
                s_ckv[r][tid] = __half2float(row[tid]);
                s_ckv[r][tid + 256] = __half2float(row[tid + 256]);
                if (tid < kRope) s_kpe[r][tid] = __half2float(row[kLora + tid]);
            } else {
                const uint8_t* row = kv_cache + cell * (size_t) kDsaFp8RowBytes;
                const float* scales = reinterpret_cast<const float*>(row + kLora);
                const uint16_t* rope_in = reinterpret_cast<const uint16_t*>(row + kLora + (kLora / 128) * 4);
                const __nv_fp8_e4m3 q0 = *reinterpret_cast<const __nv_fp8_e4m3*>(row + tid);
                const __nv_fp8_e4m3 q1 = *reinterpret_cast<const __nv_fp8_e4m3*>(row + tid + 256);
                s_ckv[r][tid] = (float) q0 * scales[tid >> 7];
                s_ckv[r][tid + 256] = (float) q1 * scales[(tid + 256) >> 7];
                if (tid < kRope) s_kpe[r][tid] = __bfloat162float(__ushort_as_bfloat16(rope_in[tid]));
            }
        }
        __syncthreads();

        if (active) {
            for (int r = 0; r < cnt; ++r) {
                const float* ck = s_ckv[r];
                const float* kp = s_kpe[r];
                float dot = fmaf(q_pe0, kp[lane], q_pe1 * kp[lane + 32]);
#pragma unroll
                for (int blk = 0; blk < 16; ++blk) dot = fmaf(q_lat[blk], ck[lane + 32 * blk], dot);
                const float sc = warp_sum(dot) * scale;
                const float m_new = fmaxf(m, sc);
                const float alpha = expf(m - m_new);
                const float beta = expf(sc - m_new);
                l = l * alpha + beta;
                m = m_new;
#pragma unroll
                for (int blk = 0; blk < 16; ++blk) o_lat[blk] = fmaf(o_lat[blk], alpha, beta * ck[lane + 32 * blk]);
            }
        }
        __syncthreads();
        if (cnt < kTile) break;
    }

    if (active) {
        const float inv_l = 1.0f / l;
        uint16_t* dst = o_lat_f16 + ((size_t) h * n_tokens + t) * kLora;
#pragma unroll
        for (int blk = 0; blk < 16; ++blk) {
            dst[lane + 32 * blk] = __half_as_ushort(__float2half_rn(o_lat[blk] * inv_l));
        }
    }
}

// ---- Tensor-core FP16 MMA pre-absorbed sparse MLA attention (kv_lora = 512, qk_rope = 64 -> D = 576) -----------

constexpr int kDsaMmaG = 2;       // M = 16 * kDsaMmaG = 32 heads per block
constexpr int kDsaMmaCells = 32;  // keys per shared-memory tile
constexpr int kDsaMmaLd = 584;    // 576 + 8 halfs (16-byte aligned, conflict-free)
constexpr int kDsaMmaPLd = 40;    // 32 + 8 halfs

constexpr size_t dsa_mma512_smem() {
    constexpr int M = 16 * kDsaMmaG;
    return (size_t) (M * kDsaMmaLd + kDsaMmaCells * kDsaMmaLd + M * kDsaMmaPLd) * sizeof(uint16_t)
         + (size_t) (2 * M * kDsaMmaCells + 2 * M) * sizeof(float);
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800

// the FP16 MMA inlines and the wider -> FP16 conversions Basalt's attention kernels share
using basalt::kernels::mma_f16::bf2_to_h2;
using basalt::kernels::mma_f16::e4m3x2_to_h2_f32scale;
using basalt::kernels::mma_f16::f32_to_h_clamped;
using basalt::kernels::mma_f16::ldsm4;
using basalt::kernels::mma_f16::ldsm4t;
using basalt::kernels::mma_f16::mma16816h;

// Eight values of a DSA latent KV row's chunk ch (of 72: 64 c_kv + 8 k_pe) as eight packed FP16 values.
// FP8 layout (kDsaFp8RowBytes = 656, 16-byte aligned): 512 E4M3 c_kv + 4 float scales (128-value groups -> ch >> 4)
// + 64 BF16 k_pe starting at byte 528 (528 % 16 == 0).  A code times its power-of-two scale stays normal in FP16
// down to 2^-14 (smaller scales round through FP16 subnormals); the BF16 k_pe saturates to FP16's finite range.
template <bool FP8>
__device__ __forceinline__ uint4 dsa_fetch_row8(const uint8_t* kv_cache, size_t cell, int ch) {
    if (!FP8) return *reinterpret_cast<const uint4*>(kv_cache + cell * (size_t) (2 * 576) + ch * 16);
    const uint8_t* row = kv_cache + cell * (size_t) kDsaFp8RowBytes;
    if (ch < 64) {
        const uint2 codes = *reinterpret_cast<const uint2*>(row + ch * 8);
        const float s = reinterpret_cast<const float*>(row + 512)[ch >> 4];
        return make_uint4(e4m3x2_to_h2_f32scale(codes.x & 0xffffu, s), e4m3x2_to_h2_f32scale(codes.x >> 16, s),
                          e4m3x2_to_h2_f32scale(codes.y & 0xffffu, s), e4m3x2_to_h2_f32scale(codes.y >> 16, s));
    }
    const uint4 w = *reinterpret_cast<const uint4*>(row + 528 + (ch - 64) * 16);
    return make_uint4(bf2_to_h2(w.x), bf2_to_h2(w.y), bf2_to_h2(w.z), bf2_to_h2(w.w));
}

// One block (256 threads, 8 warps) per (32-head tile, token): FP16 Tensor-Core sparse MLA attention over sel_idx.
// Q = [q_lat (512), q_pe (64)] * scale (D = 576 = 36 * 16); K = [c_kv (512), k_pe (64)]; V = c_kv (512).
template <bool FP8>
__global__ void __launch_bounds__(256) dsa_mla_attention_preabsorbed_mma512_kernel(
    const float* __restrict__ q_lat_in, const float* __restrict__ q_full, const uint8_t* __restrict__ kv_cache,
    const float* __restrict__ attn_sink, const int32_t* __restrict__ sel_idx, int n_tokens, int n_heads,
    int qk_nope, int sel_width, uint16_t* __restrict__ o_lat_f16) {
    extern __shared__ __align__(16) unsigned char smraw[];
    constexpr int W = 512, R = 64, D = W + R, M = 16 * kDsaMmaG;
    uint16_t* qh = (uint16_t*) smraw;
    uint16_t* ks = qh + M * kDsaMmaLd;
    uint16_t* ph = ks + kDsaMmaCells * kDsaMmaLd;
    float* sp = (float*) (ph + M * kDsaMmaPLd);   // [2][M][32]
    float* alpha_s = sp + 2 * M * 32;
    float* l_s = alpha_s + M;

    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    const int h0 = blockIdx.x * M;
    const int t = blockIdx.y;
    const int qd = qk_nope + R;
    const float scale = rsqrtf((float) qd);

    for (int i = tid; i < M * D; i += 256) {
        const int h = i / D, e = i - h * D;
        const int gh = h0 + h;
        float v = 0.0f;
        if (gh < n_heads) {
            if (e < W) v = q_lat_in[((size_t) gh * n_tokens + t) * W + e];
            else v = q_full[((size_t) t * n_heads + gh) * qd + qk_nope + (e - W)];
        }
        qh[h * kDsaMmaLd + e] = __half_as_ushort(f32_to_h_clamped(v * scale));
    }

    float m[2 * kDsaMmaG], l[2 * kDsaMmaG];
#pragma unroll
    for (int hh = 0; hh < 2 * kDsaMmaG; ++hh) {
        const int gh = h0 + w * 2 * kDsaMmaG + hh;
        m[hh] = gh < n_heads ? attn_sink[gh] : 0.0f;
        l[hh] = 1.0f;
    }
    if (tid < M) l_s[tid] = 1.0f;

    float acc[8 * kDsaMmaG][4];
#pragma unroll
    for (int j = 0; j < 8 * kDsaMmaG; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;

    const int id = lane >> 3, r8 = lane & 7, rg = w % kDsaMmaG, dq = w / kDsaMmaG;
    const int kh = (w / kDsaMmaG) & 1, cp = w / (2 * kDsaMmaG);
    const int32_t* sel = sel_idx + (size_t) t * sel_width;

    // 32 cells * (576 / 8) chunks / 256 threads = 9 chunks per thread
    constexpr int kPer = kDsaMmaCells * (D / 8) / 256;
    uint4 pre[kPer];
    auto fetch = [&](int j0) {
#pragma unroll
        for (int k = 0; k < kPer; ++k) {
            const int i = tid + k * 256, c = i / (D / 8), ch = i - c * (D / 8);
            const int32_t cell = (j0 + c < sel_width) ? sel[j0 + c] : -1;
            pre[k] = cell >= 0 ? dsa_fetch_row8<FP8>(kv_cache, (size_t) cell, ch) : make_uint4(0, 0, 0, 0);
        }
    };

    if (sel_width > 0 && sel[0] >= 0) fetch(0);
    for (int j0 = 0; j0 < sel_width; j0 += kDsaMmaCells) {
        const uint32_t bad = __ballot_sync(0xffffffffu, (j0 + lane >= sel_width) || (sel[j0 + lane] < 0));
        const int tn = bad ? (__ffs(bad) - 1) : kDsaMmaCells;
        if (tn == 0) break;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < kPer; ++k) {
            const int i = tid + k * 256, c = i / (D / 8), ch = i - c * (D / 8);
            *reinterpret_cast<uint4*>(ks + c * kDsaMmaLd + ch * 8) = pre[k];
        }
        __syncthreads();
        const int jnext = j0 + kDsaMmaCells;
        if (tn == kDsaMmaCells && jnext < sel_width && sel[jnext] >= 0) fetch(jnext);

        {   // S = Q K^T over D = 576 (18 k-steps of 16 per kh half = 9 pairs of 32)
            float c[kDsaMmaG][4];
#pragma unroll
            for (int nt = 0; nt < kDsaMmaG; ++nt) c[nt][0] = c[nt][1] = c[nt][2] = c[nt][3] = 0.0f;
#pragma unroll
            for (int kk = 0; kk < 9; ++kk) {
                const int kb = (kh * 18 + kk * 2) * 16;
                uint32_t b[kDsaMmaG][4];
#pragma unroll
                for (int nt = 0; nt < kDsaMmaG; ++nt)
                    ldsm4(b[nt], ks + ((cp * kDsaMmaG + nt) * 8 + r8) * kDsaMmaLd + kb + id * 8);
#pragma unroll
                for (int s2 = 0; s2 < 2; ++s2) {
                    const int kd = kb + s2 * 16;
                    uint32_t ah[4];
                    ldsm4(ah, qh + (rg * 16 + (id & 1) * 8 + r8) * kDsaMmaLd + kd + (id >> 1) * 8);
#pragma unroll
                    for (int nt = 0; nt < kDsaMmaG; ++nt) mma16816h(c[nt], ah, b[nt][2 * s2], b[nt][2 * s2 + 1]);
                }
            }
            const int row = rg * 16 + (lane >> 2);
#pragma unroll
            for (int nt = 0; nt < kDsaMmaG; ++nt) {
                const int col = (cp * kDsaMmaG + nt) * 8 + (lane & 3) * 2;
                sp[(kh * M + row) * 32 + col] = c[nt][0];
                sp[(kh * M + row) * 32 + col + 1] = c[nt][1];
                sp[(kh * M + row + 8) * 32 + col] = c[nt][2];
                sp[(kh * M + row + 8) * 32 + col + 1] = c[nt][3];
            }
        }
        __syncthreads();

        // Online softmax: 1 lane per key in the 32-key tile
#pragma unroll
        for (int hh = 0; hh < 2 * kDsaMmaG; ++hh) {
            const int h = w * 2 * kDsaMmaG + hh;
            const bool valid = lane < tn;
            const float sc = valid ? sp[h * 32 + lane] + sp[(M + h) * 32 + lane] : -INFINITY;
            const float mx = warp_max(sc);
            const float nm = fmaxf(m[hh], mx);
            const float alpha = expf(m[hh] - nm);
            const float pr = valid ? expf(sc - nm) : 0.0f;
            const float sum = warp_sum(pr);
            l[hh] = l[hh] * alpha + sum;
            m[hh] = nm;
            ph[h * kDsaMmaPLd + lane] = __half_as_ushort(__float2half_rn(pr));
            if (lane == 0) {
                alpha_s[h] = alpha;
                l_s[h] = l[hh];
            }
        }
        __syncthreads();

        // O = O * alpha + P V (V is the leading W = 512 c_kv columns of ks)
        {
            const float a0 = alpha_s[rg * 16 + (lane >> 2)], a1 = alpha_s[rg * 16 + (lane >> 2) + 8];
#pragma unroll
            for (int j = 0; j < 8 * kDsaMmaG; ++j) {
                acc[j][0] *= a0; acc[j][1] *= a0;
                acc[j][2] *= a1; acc[j][3] *= a1;
            }
#pragma unroll
            for (int s2 = 0; s2 < 2; ++s2) {
                uint32_t ah[4];
                ldsm4(ah, ph + (rg * 16 + (id & 1) * 8 + r8) * kDsaMmaPLd + s2 * 16 + (id >> 1) * 8);
#pragma unroll
                for (int jj = 0; jj < 4 * kDsaMmaG; ++jj) {
                    uint32_t b[4];
                    ldsm4t(b, ks + (s2 * 16 + (id & 1) * 8 + r8) * kDsaMmaLd + dq * (W / (2 * kDsaMmaG)) + jj * 16 +
                                      (id >> 1) * 8);
                    mma16816h(acc[2 * jj], ah, b[0], b[1]);
                    mma16816h(acc[2 * jj + 1], ah, b[2], b[3]);
                }
            }
        }
        if (tn < kDsaMmaCells) break;
    }

    __syncthreads();
    const int row = rg * 16 + (lane >> 2);
    const int g0 = h0 + row, g1 = h0 + row + 8;
    const float i0 = 1.0f / l_s[row], i1 = 1.0f / l_s[row + 8];
#pragma unroll
    for (int j = 0; j < 8 * kDsaMmaG; ++j) {
        const int d = dq * (W / (2 * kDsaMmaG)) + j * 8 + (lane & 3) * 2;
        if (g0 < n_heads) {
            uint16_t* o0 = o_lat_f16 + ((size_t) g0 * n_tokens + t) * W + d;
            o0[0] = __half_as_ushort(f32_to_h_clamped(acc[j][0] * i0));
            o0[1] = __half_as_ushort(f32_to_h_clamped(acc[j][1] * i0));
        }
        if (g1 < n_heads) {
            uint16_t* o1 = o_lat_f16 + ((size_t) g1 * n_tokens + t) * W + d;
            o1[0] = __half_as_ushort(f32_to_h_clamped(acc[j][2] * i1));
            o1[1] = __half_as_ushort(f32_to_h_clamped(acc[j][3] * i1));
        }
    }
}

#endif  // __CUDA_ARCH__ >= 800

// Dequantizes `n_keys` rows of `idx_k_cache[k0 .. k0+n_keys)` into BF16 `[n_keys, idx_dim]` for Tensor-Core GEMM.
// Every MXFP4 E2M1 value (3-bit mantissa, 8-bit UE8M0 exponent) is bit-exact in BF16 with zero overflow/underflow.
__global__ void dsa_dequant_idx_k_bf16_kernel(const uint8_t* __restrict__ idx_k_cache, int k0, int idx_dim, bool fp8,
                                              uint16_t* __restrict__ k_bf16) {
    const int r = blockIdx.x;
    const int d = threadIdx.x;
    if (d >= idx_dim) return;
    const size_t cell = (size_t) (k0 + r);
    float v;
    if (!fp8) {
        const __half* row = reinterpret_cast<const __half*>(idx_k_cache) + cell * (size_t) idx_dim;
        v = __half2float(row[d]);
    } else {
        const uint8_t* row = idx_k_cache + cell * (size_t) kIdxFp4RowBytes;
        const uint8_t byte = row[d >> 1];
        const uint8_t nib = (d & 1) ? (byte >> 4) : (byte & 0x0F);
        const float sc = ldexpf(1.0f, (int) row[64 + (d >> 5)] - 127);
        v = kE2M1[nib] * sc;
    }
    k_bf16[(size_t) r * idx_dim + d] = __bfloat16_as_ushort(__float2bfloat16(v));
}

// Reduces cuBLAS dot products `dots[(t * idx_heads + h) * tile_k + j]` across heads into `scores_buf[t * stride + k0 + j]`.
__global__ void dsa_indexer_reduce_tile_kernel(const float* __restrict__ dots, const float* __restrict__ iweights,
                                               const int32_t* __restrict__ pos, int idx_heads, int idx_dim, int top_k,
                                               int win_size, int k0, int tile_k, float* __restrict__ scores_buf,
                                               int scores_stride) {
    const int t = blockIdx.y;
    const int p = pos[t];
    const int seq_len = p + 1;
    const int older_n = seq_len > win_size ? (seq_len - win_size) : 0;
    if (older_n <= top_k) return;

    __shared__ float sw[64];
    const float wscale = rsqrtf((float) (idx_dim * idx_heads));
    if (threadIdx.x < idx_heads) sw[threadIdx.x] = iweights[(size_t) t * idx_heads + threadIdx.x] * wscale;
    __syncthreads();

    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= tile_k || k0 + j >= older_n) return;

    const float* base = dots + (size_t) t * idx_heads * tile_k + j;
    float total = 0.0f;
    for (int h = 0; h < idx_heads; ++h) {
        const float d = base[(size_t) h * tile_k];
        if (d > 0.0f) total = fmaf(d, sw[h], total);
    }
    scores_buf[(size_t) t * scores_stride + k0 + j] = total;
}

}  // namespace

void dsa_qkv_rope_and_store(float* q, const float* kv_a, const float* kv_norm_w, const int32_t* pos, int n_tokens,
                            int n_heads, int qk_nope, int qk_rope, int kv_lora, float rms_eps, const RopeTable& rt,
                            uint8_t* kv_cache, bool fp8, void* stream) {
    if (n_tokens <= 0) return;
    cudaStream_t st = (cudaStream_t) stream;
    dsa_q_rope_kernel<<<dim3(n_heads, n_tokens), 32, 0, st>>>(q, pos, n_heads, qk_nope, qk_rope, rt);
    const size_t smem = (size_t) (kv_lora + qk_rope + 16) * sizeof(float);
    dsa_kv_store_kernel<<<n_tokens, 256, smem, st>>>(kv_a, kv_norm_w, pos, kv_lora, qk_rope, rms_eps, rt, kv_cache,
                                                     fp8);
}

void dsa_indexer_prepare(float* iq, float* ik, const float* k_norm_w, const float* k_norm_b, const int32_t* pos,
                         int n_tokens, int idx_heads, int idx_dim, int rope_dim, float eps, const RopeTable& rt,
                         uint8_t* idx_k_cache, bool fp8, void* stream) {
    if (n_tokens <= 0) return;
    cudaStream_t st = (cudaStream_t) stream;
    // LayerNorm on ik in-place
    layernorm_f32(ik, k_norm_w, k_norm_b, n_tokens, idx_dim, eps, ik, stream);
    // Partial NeoX RoPE on leading rope_dim (64) dims of iq and ik
    neox_partial_rope_kernel<<<dim3(idx_heads, n_tokens), 32, 0, st>>>(iq, pos, idx_heads, idx_dim, rope_dim, rt);
    neox_partial_rope_kernel<<<dim3(1, n_tokens), 32, 0, st>>>(ik, pos, 1, idx_dim, rope_dim, rt);
    // Orthonormal 128-dim Walsh-Hadamard transform on iq and ik
    hadamard_128_f32(iq, n_tokens * idx_heads, stream);
    hadamard_128_f32(ik, n_tokens, stream);
    // Store ik into indexer K cache
    store_idx_k_kernel<<<n_tokens, 128, 0, st>>>(ik, pos, idx_dim, idx_k_cache, fp8);
}

void dsa_indexer_topk_merge(const float* iq, const float* iweights, const uint8_t* idx_k_cache, const int32_t* pos,
                            int n_tokens, int idx_heads, int idx_dim, int top_k, int win_size, bool fp8,
                            int32_t* out_idx, void* stream, float* scores_buf, int scores_stride) {
    if (n_tokens <= 0 || !scores_buf || scores_stride <= 0) return;
    cudaStream_t st = (cudaStream_t) stream;
    const int xblocks = n_tokens <= 2 ? 128 : 64;
    if (idx_dim == 128 && idx_heads <= 64) {
        dsa_indexer_score_fast128_kernel<<<dim3(xblocks, n_tokens), 256, 0, st>>>(
            iq, iweights, idx_k_cache, pos, idx_heads, top_k, win_size, fp8, scores_buf, scores_stride);
    } else {
        dsa_indexer_score_kernel<<<dim3(xblocks, n_tokens), 256, 0, st>>>(
            iq, iweights, idx_k_cache, pos, idx_heads, idx_dim, top_k, win_size, fp8, scores_buf, scores_stride);
    }
    dsa_indexer_select_cached_kernel<<<n_tokens, 256, 0, st>>>(
        scores_buf, scores_stride, pos, top_k, win_size, out_idx);
}

void dsa_dequant_idx_k_bf16(const uint8_t* idx_k_cache, int k0, int n_keys, int idx_dim, bool fp8, uint16_t* k_bf16,
                            void* stream) {
    if (n_keys <= 0) return;
    dsa_dequant_idx_k_bf16_kernel<<<n_keys, idx_dim, 0, (cudaStream_t) stream>>>(idx_k_cache, k0, idx_dim, fp8, k_bf16);
}

void dsa_indexer_reduce_tile(const float* dots, const float* iweights, const int32_t* pos, int n_tokens, int idx_heads,
                             int idx_dim, int top_k, int win_size, int k0, int tile_k, float* scores_buf,
                             int scores_stride, void* stream) {
    if (n_tokens <= 0 || tile_k <= 0) return;
    dsa_indexer_reduce_tile_kernel<<<dim3((tile_k + 255) / 256, n_tokens), 256, 0, (cudaStream_t) stream>>>(
        dots, iweights, pos, idx_heads, idx_dim, top_k, win_size, k0, tile_k, scores_buf, scores_stride);
}

void dsa_indexer_select_cached(const float* scores_buf, int scores_stride, const int32_t* pos, int n_tokens, int top_k,
                               int win_size, int32_t* out_idx, void* stream) {
    if (n_tokens <= 0) return;
    dsa_indexer_select_cached_kernel<<<n_tokens, 256, 0, (cudaStream_t) stream>>>(
        scores_buf, scores_stride, pos, top_k, win_size, out_idx);
}

void dsa_mla_attention(const float* q, const uint8_t* kv_cache, const void* kv_b_q8, const float* attn_sink,
                       const int32_t* sel_idx, int n_tokens, int n_heads, int qk_nope, int qk_rope, int kv_lora,
                       int v_dim, int sel_width, bool fp8, float* out, void* stream) {
    if (n_tokens <= 0) return;
    if (kv_lora == 512 && qk_rope == 64) {
        if (qk_nope == 192 && v_dim == 256) {
            dsa_mla_attention_gqa512_kernel<192, 256><<<dim3((n_heads + 7) / 8, n_tokens), 256, 0, (cudaStream_t) stream>>>(
                q, kv_cache, (const uint8_t*) kv_b_q8, attn_sink, sel_idx, n_heads, sel_width, fp8, out);
            return;
        }
        if (qk_nope == 128 && v_dim == 128) {
            dsa_mla_attention_gqa512_kernel<128, 128><<<dim3((n_heads + 7) / 8, n_tokens), 256, 0, (cudaStream_t) stream>>>(
                q, kv_cache, (const uint8_t*) kv_b_q8, attn_sink, sel_idx, n_heads, sel_width, fp8, out);
            return;
        }
    }
    const size_t smem = (size_t) (qk_nope + 2 * qk_rope + 3 * kv_lora + 16) * sizeof(float);
    dsa_mla_attention_kernel<<<dim3(n_heads, n_tokens), 256, smem, (cudaStream_t) stream>>>(
        q, kv_cache, (const uint8_t*) kv_b_q8, attn_sink, sel_idx, n_heads, qk_nope, qk_rope, kv_lora, v_dim,
        sel_width, fp8, out);
}

void dequant_q8_0_to_f16(const void* q8_src, uint16_t* f16_dst, int64_t n_blocks, void* stream) {
    if (n_blocks <= 0) return;
    const unsigned grid = (unsigned) ((n_blocks + 7) / 8);
    dequant_q8_0_to_f16_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>((const uint8_t*) q8_src, f16_dst, n_blocks);
}

void dsa_pack_q_nope_f16(const float* q, uint16_t* q_nope_f16, int n_tokens, int n_heads, int qk_nope, int qk_rope,
                         void* stream) {
    if (n_tokens <= 0) return;
    dsa_pack_q_nope_f16_kernel<<<dim3((qk_nope + 127) / 128, n_heads, n_tokens), 128, 0, (cudaStream_t) stream>>>(
        q, q_nope_f16, n_tokens, n_heads, qk_nope, qk_rope);
}

void dsa_mla_attention_preabsorbed(const float* q_lat, const float* q_full, const uint8_t* kv_cache,
                                   const float* attn_sink, const int32_t* sel_idx, int n_tokens, int n_heads,
                                   int qk_nope, int qk_rope, int kv_lora, int sel_width, bool fp8,
                                   uint16_t* o_lat_f16, void* stream) {
    if (n_tokens <= 0) return;
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    if (n_tokens > 8 && kv_lora == 512 && qk_rope == 64) {
        // the dynamic shared-memory opt-in and the sm_80+ check, once per device and instantiation; a card that
        // rejects either keeps the one-block-a-head-group kernel
        static basalt::kernels::mma_f16::MmaSmemCache smem_cache[2];
        const bool mma_usable =
            fp8 ? basalt::kernels::mma_f16::mma_smem_ready(smem_cache[1],
                                                           dsa_mla_attention_preabsorbed_mma512_kernel<true>,
                                                           dsa_mma512_smem())
                : basalt::kernels::mma_f16::mma_smem_ready(smem_cache[0],
                                                           dsa_mla_attention_preabsorbed_mma512_kernel<false>,
                                                           dsa_mma512_smem());
        if (mma_usable) {
            const dim3 grid((n_heads + 16 * kDsaMmaG - 1) / (16 * kDsaMmaG), n_tokens);
            if (fp8)
                dsa_mla_attention_preabsorbed_mma512_kernel<true><<<grid, 256, dsa_mma512_smem(), (cudaStream_t) stream>>>(
                    q_lat, q_full, kv_cache, attn_sink, sel_idx, n_tokens, n_heads, qk_nope, sel_width, o_lat_f16);
            else
                dsa_mla_attention_preabsorbed_mma512_kernel<false><<<grid, 256, dsa_mma512_smem(), (cudaStream_t) stream>>>(
                    q_lat, q_full, kv_cache, attn_sink, sel_idx, n_tokens, n_heads, qk_nope, sel_width, o_lat_f16);
            return;
        }
    }
#endif
    dsa_mla_attention_preabsorbed_gqa512_kernel<<<dim3((n_heads + 7) / 8, n_tokens), 256, 0, (cudaStream_t) stream>>>(
        q_lat, q_full, kv_cache, attn_sink, sel_idx, n_tokens, n_heads, qk_nope, sel_width, fp8, o_lat_f16);
}


// ---- decode windows: the selected positions split across blocks (see hca_attention_split) ---------------------

namespace {

// Block (h, t): q_lat = W_UK^T q_nope of head h, two columns a thread, the rows in order (the sums of
// dsa_mla_attention_gqa512_kernel's step 1).
template <int kNope, int kVdim>
__global__ void __launch_bounds__(256) dsa_absorb_kernel(const float* __restrict__ q, const uint8_t* __restrict__ kv_b_q8,
                                                         int n_heads, float* __restrict__ q_lat) {
    constexpr int kRope = 64, kLora = 512, kQd = kNope + kRope;
    constexpr size_t kQ8RowBytes = (kLora / 32) * 34;
    const int h = blockIdx.x, t = blockIdx.y;
    __shared__ float q_nope[kNope];
    const float* qrow = q + ((size_t) t * n_heads + h) * kQd;
    for (int i = threadIdx.x; i < kNope; i += blockDim.x) q_nope[i] = qrow[i];
    __syncthreads();
    const uint8_t* w_uk = kv_b_q8 + (size_t) h * (size_t) (kNope + kVdim) * kQ8RowBytes;
    float* dst = q_lat + ((size_t) t * n_heads + h) * kLora;
    for (int c = threadIdx.x; c < kLora; c += blockDim.x) {
        const uint8_t* col = w_uk + (size_t) (c >> 5) * 34;
        const int lane = c & 31;
        float acc = 0.0f;
        for (int r = 0; r < kNope; ++r) {
            const uint8_t* b = col + (size_t) r * kQ8RowBytes;
            const float d = __half2float(*reinterpret_cast<const __half*>(b));
            const int8_t qv = reinterpret_cast<const int8_t*>(b + 2)[lane];
            acc = fmaf(q_nope[r], d * (float) qv, acc);
        }
        dst[c] = acc;
    }
}

template <int kNope>
__global__ void dsa_attention_split_kernel(const float* __restrict__ q, const float* __restrict__ q_lat,
                                           const uint8_t* __restrict__ kv_cache, const float* __restrict__ attn_sink,
                                           const int32_t* __restrict__ sel_idx, int n_heads, int sel_width, bool fp8,
                                           int n_split, float* __restrict__ part) {
    constexpr int kRope = 64, kLora = 512, kQd = kNope + kRope, kTile = 4;
    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
    const int h = blockIdx.x * 8 + wid;
    const int s = blockIdx.y, t = blockIdx.z;
    const bool active = h < n_heads;
    __shared__ float c_kv[kTile][kLora];
    __shared__ float k_pe[kTile][kRope];
    float ql[16], o[16];
    const float* qlr = q_lat + ((size_t) t * n_heads + (active ? h : 0)) * kLora;
#pragma unroll
    for (int blk = 0; blk < 16; ++blk) {
        ql[blk] = active ? qlr[lane + 32 * blk] : 0.0f;
        o[blk] = 0.0f;
    }
    const float* qrow = q + ((size_t) t * n_heads + (active ? h : 0)) * kQd;
    const float q_pe0 = active ? qrow[kNope + lane] : 0.0f, q_pe1 = active ? qrow[kNope + lane + 32] : 0.0f;
    const float scale = rsqrtf((float) kQd);
    float m = s == 0 && active ? attn_sink[h] : -INFINITY;
    float l = s == 0 ? 1.0f : 0.0f;
    const int32_t* sel = sel_idx + (size_t) t * sel_width;
    const int chunk = (sel_width + n_split - 1) / n_split;
    const int i0 = s * chunk, i1 = min(sel_width, i0 + chunk);
    for (int s0 = i0; s0 < i1; s0 += kTile) {
        int cnt = 0;
        int32_t cells[kTile];
#pragma unroll
        for (int r = 0; r < kTile; ++r) {
            if (s0 + r < i1) {
                const int32_t c = sel[s0 + r];
                if (c >= 0) {
                    cells[cnt++] = c;
                    continue;
                }
            }
            break;
        }
        if (cnt == 0) break;
        for (int r = 0; r < cnt; ++r) {
            const size_t cell = (size_t) cells[r];
            if (!fp8) {
                const __half* row = reinterpret_cast<const __half*>(kv_cache) + cell * (size_t) (kLora + kRope);
                c_kv[r][tid] = __half2float(row[tid]);
                c_kv[r][tid + 256] = __half2float(row[tid + 256]);
                if (tid < kRope) k_pe[r][tid] = __half2float(row[kLora + tid]);
            } else {
                const uint8_t* row = kv_cache + cell * (size_t) kDsaFp8RowBytes;
                const float* scales = reinterpret_cast<const float*>(row + kLora);
                const uint16_t* rope_in = reinterpret_cast<const uint16_t*>(row + kLora + (kLora / 128) * 4);
                const __nv_fp8_e4m3 q0 = *reinterpret_cast<const __nv_fp8_e4m3*>(row + tid);
                const __nv_fp8_e4m3 q1 = *reinterpret_cast<const __nv_fp8_e4m3*>(row + tid + 256);
                c_kv[r][tid] = (float) q0 * scales[tid >> 7];
                c_kv[r][tid + 256] = (float) q1 * scales[(tid + 256) >> 7];
                if (tid < kRope) k_pe[r][tid] = __bfloat162float(__ushort_as_bfloat16(rope_in[tid]));
            }
        }
        __syncthreads();
        if (active) {
            for (int r = 0; r < cnt; ++r) {
                const float* ck = c_kv[r];
                const float* kp = k_pe[r];
                float dot = fmaf(q_pe0, kp[lane], q_pe1 * kp[lane + 32]);
#pragma unroll
                for (int blk = 0; blk < 16; ++blk) dot = fmaf(ql[blk], ck[lane + 32 * blk], dot);
                const float sc = warp_sum(dot) * scale;
                const float m_new = fmaxf(m, sc);
                const float alpha = expf(m - m_new);
                const float beta = expf(sc - m_new);
                l = l * alpha + beta;
                m = m_new;
#pragma unroll
                for (int blk = 0; blk < 16; ++blk) o[blk] = fmaf(o[blk], alpha, beta * ck[lane + 32 * blk]);
            }
        }
        __syncthreads();
        if (cnt < kTile) break;
    }
    if (active) {
        float* dst = part + (((size_t) t * n_heads + h) * n_split + s) * kAttnPartFloats;
#pragma unroll
        for (int blk = 0; blk < 16; ++blk) dst[2 + lane + 32 * blk] = o[blk];
        if (lane == 0) {
            dst[0] = m;
            dst[1] = l;
        }
    }
}

// The FP8 cache's split kernel: kTile selected positions a step, their rows loaded by the whole block in 16-byte
// pieces (a row a warp) into shared memory, their 8 heads' scores side by side (one max and one rescale a step), a
// lane's latent columns 4 lane + 128 j (j < 4) read four at a time; the valid positions (the first
// min(pos + 1, sel_width): the selection is packed) spread evenly over the splits.
template <int kNope, int kTile>
__global__ void __launch_bounds__(256) dsa_attention_split_fp8_kernel(const float* __restrict__ q,
                                                                      const float* __restrict__ q_lat,
                                                                      const uint8_t* __restrict__ kv_cache,
                                                                      const float* __restrict__ attn_sink,
                                                                      const int32_t* __restrict__ sel_idx,
                                                                      const int32_t* __restrict__ pos, int n_heads,
                                                                      int sel_width, int n_split,
                                                                      float* __restrict__ part) {
    constexpr int kRope = 64, kLora = 512, kQd = kNope + kRope;
    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
    const int h = blockIdx.x * 8 + wid;
    const int s = blockIdx.y, t = blockIdx.z;
    const bool active = h < n_heads;
    __shared__ __align__(16) float c_kv[kTile][kLora];
    __shared__ __align__(16) float k_pe[kTile][kRope];
    __shared__ int ok[kTile];
    float4 qv[4], o[4];
    const float4* qlr = reinterpret_cast<const float4*>(q_lat + ((size_t) t * n_heads + (active ? h : 0)) * kLora);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        qv[j] = active ? qlr[lane + 32 * j] : make_float4(0.f, 0.f, 0.f, 0.f);
        o[j] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    const float* qrow = q + ((size_t) t * n_heads + (active ? h : 0)) * kQd;
    const float q_pe0 = active ? qrow[kNope + 2 * lane] : 0.0f, q_pe1 = active ? qrow[kNope + 2 * lane + 1] : 0.0f;
    const float scale = rsqrtf((float) kQd);
    float m = s == 0 && active ? attn_sink[h] : -INFINITY;
    float l = s == 0 ? 1.0f : 0.0f;
    const int32_t* sel = sel_idx + (size_t) t * sel_width;
    const int n_valid = min(pos[t] + 1, sel_width);
    const int chunk = (n_valid + n_split - 1) / n_split;
    const int i0 = s * chunk, i1 = min(n_valid, i0 + chunk);
    for (int s0 = i0; s0 < i1; s0 += kTile) {
        const int cnt = min(kTile, i1 - s0);
        for (int r = wid; r < kTile; r += 8) {   // warp wid loads row r
            const int32_t c = r < cnt ? sel[s0 + r] : -1;
            if (lane == 0) ok[r] = c >= 0;
            if (c < 0) continue;
            const uint8_t* row = kv_cache + (size_t) c * kDsaFp8RowBytes;
            const uint4 v = *reinterpret_cast<const uint4*>(row + 16 * lane);
            const float sc = reinterpret_cast<const float*>(row + kLora)[lane >> 3];
            const uint32_t w4[4] = {v.x, v.y, v.z, v.w};
            float4* dst = reinterpret_cast<float4*>(c_kv[r]);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const __nv_fp8x4_e4m3 f = *reinterpret_cast<const __nv_fp8x4_e4m3*>(&w4[k]);
                const float4 x = static_cast<float4>(f);
                dst[4 * lane + k] = make_float4(x.x * sc, x.y * sc, x.z * sc, x.w * sc);
            }
            const uint32_t pr = reinterpret_cast<const uint32_t*>(row + kLora + (kLora / 128) * 4)[lane];
            reinterpret_cast<float2*>(k_pe[r])[lane] = make_float2(__bfloat162float(__ushort_as_bfloat16((uint16_t) (pr & 0xffffu))),
                                                                   __bfloat162float(__ushort_as_bfloat16((uint16_t) (pr >> 16))));
        }
        __syncthreads();
        if (active) {
            float sc[kTile];
#pragma unroll
            for (int r = 0; r < kTile; ++r) {
                const float2 kp = reinterpret_cast<const float2*>(k_pe[r])[lane];
                float dot = fmaf(q_pe0, kp.x, q_pe1 * kp.y);
                const float4* kr = reinterpret_cast<const float4*>(c_kv[r]);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float4 k = kr[lane + 32 * j];
                    dot = fmaf(qv[j].x, k.x, dot);
                    dot = fmaf(qv[j].y, k.y, dot);
                    dot = fmaf(qv[j].z, k.z, dot);
                    dot = fmaf(qv[j].w, k.w, dot);
                }
                sc[r] = dot;
            }
#pragma unroll
            for (int r = 0; r < kTile; ++r) sc[r] = ok[r] ? warp_sum(sc[r]) * scale : -INFINITY;
            float mx = m;
#pragma unroll
            for (int r = 0; r < kTile; ++r) mx = fmaxf(mx, sc[r]);
            if (mx > -INFINITY) {
                const float alpha = expf(m - mx);
                float pr[kTile], ps = 0.0f;
#pragma unroll
                for (int r = 0; r < kTile; ++r) {
                    pr[r] = ok[r] ? expf(sc[r] - mx) : 0.0f;
                    ps += pr[r];
                }
                l = l * alpha + ps;
                m = mx;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float4 a = make_float4(o[j].x * alpha, o[j].y * alpha, o[j].z * alpha, o[j].w * alpha);
#pragma unroll
                    for (int r = 0; r < kTile; ++r)
                        if (ok[r]) {
                            const float4 k = reinterpret_cast<const float4*>(c_kv[r])[lane + 32 * j];
                            a.x = fmaf(pr[r], k.x, a.x);
                            a.y = fmaf(pr[r], k.y, a.y);
                            a.z = fmaf(pr[r], k.z, a.z);
                            a.w = fmaf(pr[r], k.w, a.w);
                        }
                    o[j] = a;
                }
            }
        }
        __syncthreads();
    }
    if (active) {
        float* dst = part + (((size_t) t * n_heads + h) * n_split + s) * kAttnPartFloats;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float* d = dst + 2 + 4 * lane + 128 * j;
            d[0] = o[j].x;
            d[1] = o[j].y;
            d[2] = o[j].z;
            d[3] = o[j].w;
        }
        if (lane == 0) {
            dst[0] = m;
            dst[1] = l;
        }
    }
}

// Block (h, t), 128 threads: q_lat = W_UK^T q_nope of head h, four adjacent columns a thread (one Q8_0 block's bytes),
// each column's sum over the rows in order (dsa_absorb_kernel's bits).
template <int kNope, int kVdim>
__global__ void __launch_bounds__(128) dsa_absorb4_kernel(const float* __restrict__ q, const uint8_t* __restrict__ kv_b_q8,
                                                          int n_heads, float* __restrict__ q_lat) {
    constexpr int kRope = 64, kLora = 512, kQd = kNope + kRope;
    constexpr size_t kQ8RowBytes = (kLora / 32) * 34;
    const int h = blockIdx.x, t = blockIdx.y;
    __shared__ float q_nope[kNope];
    const float* qrow = q + ((size_t) t * n_heads + h) * kQd;
    for (int i = threadIdx.x; i < kNope; i += blockDim.x) q_nope[i] = qrow[i];
    __syncthreads();
    const int c0 = 4 * threadIdx.x;
    const uint8_t* col = kv_b_q8 + (size_t) h * (size_t) (kNope + kVdim) * kQ8RowBytes + (size_t) (c0 >> 5) * 34;
    const int off = c0 & 31;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll 8
    for (int r = 0; r < kNope; ++r) {
        const uint8_t* b = col + (size_t) r * kQ8RowBytes;
        const float d = __half2float(*reinterpret_cast<const __half*>(b));
        const uint16_t* w = reinterpret_cast<const uint16_t*>(b + 2 + off);
        const uint16_t lo = w[0], hi = w[1];
        const float qn = q_nope[r];
        a0 = fmaf(qn, d * (float) (int8_t) (lo & 0xff), a0);
        a1 = fmaf(qn, d * (float) (int8_t) (lo >> 8), a1);
        a2 = fmaf(qn, d * (float) (int8_t) (hi & 0xff), a2);
        a3 = fmaf(qn, d * (float) (int8_t) (hi >> 8), a3);
    }
    float* dst = q_lat + ((size_t) t * n_heads + h) * kLora + c0;
    dst[0] = a0;
    dst[1] = a1;
    dst[2] = a2;
    dst[3] = a3;
}

// Block (h, t): o_lat of head h from its splits, then o = W_UV o_lat (a row per thread).
template <int kNope, int kVdim>
__global__ void dsa_split_combine_kernel(const float* __restrict__ part, const uint8_t* __restrict__ kv_b_q8,
                                         int n_heads, int n_split, float* __restrict__ out) {
    constexpr int kLora = 512;
    constexpr size_t kQ8RowBytes = (kLora / 32) * 34;
    const int h = blockIdx.x, t = blockIdx.y;
    __shared__ float o_lat[kLora];
    const float* src = part + ((size_t) t * n_heads + h) * n_split * kAttnPartFloats;
    float M = -INFINITY;
    for (int s = 0; s < n_split; ++s) M = fmaxf(M, src[(size_t) s * kAttnPartFloats]);
    float L = 0.0f;
    for (int s = 0; s < n_split; ++s) {
        const float* ps = src + (size_t) s * kAttnPartFloats;
        if (ps[1] > 0.0f) L += ps[1] * expf(ps[0] - M);
    }
    const float inv = 1.0f / L;
    for (int c = threadIdx.x; c < kLora; c += blockDim.x) {
        float o = 0.0f;
        for (int s = 0; s < n_split; ++s) {
            const float* ps = src + (size_t) s * kAttnPartFloats;
            if (ps[1] > 0.0f) o = fmaf(ps[2 + c], expf(ps[0] - M), o);
        }
        o_lat[c] = o * inv;
    }
    __syncthreads();
    const uint8_t* w_uv = kv_b_q8 + (size_t) h * (size_t) (kNope + kVdim) * kQ8RowBytes + (size_t) kNope * kQ8RowBytes;
    float* orow = out + ((size_t) t * n_heads + h) * kVdim;
    for (int r = threadIdx.x; r < kVdim; r += blockDim.x) orow[r] = q8_0_row_dot(w_uv + (size_t) r * kQ8RowBytes, o_lat, kLora);
}

}  // namespace

bool dsa_mla_attention_split(const float* q, const uint8_t* kv_cache, const void* kv_b_q8, const float* attn_sink,
                             const int32_t* sel_idx, const int32_t* pos, int n_tokens, int n_heads, int qk_nope,
                             int qk_rope, int kv_lora, int v_dim, int sel_width, bool fp8, float* out, float* part,
                             float* q_lat, void* stream) {
    if (n_tokens <= 0) return true;
    if (kv_lora != 512 || qk_rope != 64 || qk_nope != 192 || v_dim != 256) return false;
    cudaStream_t st = (cudaStream_t) stream;
    const uint8_t* w = (const uint8_t*) kv_b_q8;
    if (!fp8 || !pos) {
        const int ns = attn_splits(n_tokens);
        dsa_absorb_kernel<192, 256><<<dim3(n_heads, n_tokens), 256, 0, st>>>(q, w, n_heads, q_lat);
        dsa_attention_split_kernel<192><<<dim3((n_heads + 7) / 8, ns, n_tokens), 256, 0, st>>>(
            q, q_lat, kv_cache, attn_sink, sel_idx, n_heads, sel_width, fp8, ns, part);
        dsa_split_combine_kernel<192, 256><<<dim3(n_heads, n_tokens), 256, 0, st>>>(part, w, n_heads, ns, out);
        return true;
    }
    const int ns = kAttnSplitSlots / n_tokens;   // the part buffer holds kAttnSplitSlots rows x splits
    dsa_absorb4_kernel<192, 256><<<dim3(n_heads, n_tokens), 128, 0, st>>>(q, w, n_heads, q_lat);
    dsa_attention_split_fp8_kernel<192, 4><<<dim3((n_heads + 7) / 8, ns, n_tokens), 256, 0, st>>>(
        q, q_lat, kv_cache, attn_sink, sel_idx, pos, n_heads, sel_width, ns, part);
    dsa_split_combine_kernel<192, 256><<<dim3(n_heads, n_tokens), 256, 0, st>>>(part, w, n_heads, ns, out);
    return true;
}

}  // namespace basalt::models::ernie55
