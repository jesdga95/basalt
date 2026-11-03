// src/kernels/cuda/gdn_unfused.cuh - the bodies of verify_kernels.cu's unfused GDN kernels (conv + L2 norm,
// alpha/beta, recurrence + gated norm) as device functions of one block, shared with the sm_89 mid chain
// (mid_chain_ada.cu), which runs them inside its persistent kernel.  A translation unit including this is built without
// fast math, like verify_kernels.cu.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "basalt/kernels/pdl.hpp"

namespace basalt::kernels::gdn_unfused {

constexpr int S = 128;   // GDN state size
constexpr int RG = 4;
constexpr int RPG = S / RG;

struct SyncBlock {
    __device__ __forceinline__ void operator()() const { __syncthreads(); }
};

// conv1d + SiLU of channel cb * S + tid for token t (S threads), L2-normalised over the channel block when it is a
// q or k head (cb < qk_heads); part: S / 32 floats of shared memory, sync: the S threads' barrier
template <class Sync>
__device__ __forceinline__ void conv_l2_unit(int cb, int t, int tid, float* part, Sync sync,
                                             const float* __restrict__ hist, const float* __restrict__ qkv,
                                             const float* __restrict__ w, float* __restrict__ h, int C, int qk_heads,
                                             float eps) {
    const int c = (unsigned) cb * S + (unsigned) tid;   // unsigned, as the kernel's blockIdx.x * S + threadIdx.x
    // the window of token t: [hist0, hist1, hist2, x_0, ..., x_t], its last four entries
    float win[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = t + j;          // index into [hist(3) | x...]
        win[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
    }
    const float v0 = win[0], v1 = win[1], v2 = win[2], x = qkv[(size_t) t * C + c];
    float sum = v0 * w[c * 4] + v1 * w[c * 4 + 1] + v2 * w[c * 4 + 2] + x * w[c * 4 + 3];
    float y = sum / (1.0f + __expf(-sum));
    if (cb < qk_heads) {
        float sq = y * y;
        for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
        if (((unsigned) tid & 31) == 0) part[(unsigned) tid >> 5] = sq;
        sync();
        const float ss = part[0] + part[1] + part[2] + part[3];
        y *= rsqrtf(ss + eps);
    }
    h[(size_t) t * C + c] = y;
}

// alpha/beta rows 2 * blk and 2 * blk + 1 (64 threads, one warp per row) for rows of exactly 32 * NJ uint4: every
// weight load issued before the sums and the token loop unrolled, so a warp waits for memory once instead of once per
// step.  The sums of gdn_ab_multi_kernel in the same order: the same bits.
template <int MAX_T, int NJ>
__device__ __forceinline__ void ab_rows_unrolled(int blk, int tid, const float* __restrict__ x,
                                                 const uint16_t* __restrict__ wa, const uint16_t* __restrict__ wb,
                                                 const float* __restrict__ dt, const float* __restrict__ ssm_a,
                                                 float* __restrict__ gate, float* __restrict__ beta, int n, int h_v,
                                                 int T) {
    const int row = (unsigned) blk * 2 + ((unsigned) tid >> 5), lane = (unsigned) tid & 31;
    const bool valid = row < 2 * h_v;
    const bool is_beta = row >= h_v;
    const int r = is_beta ? row - h_v : row;
    const uint4* w4 = reinterpret_cast<const uint4*>((is_beta ? wb : wa) + (size_t) (valid ? r : 0) * n);
    uint4 wv[NJ];
#pragma unroll
    for (int q = 0; q < NJ; ++q) wv[q] = __ldg(w4 + lane + 32 * q);
    pdl_wait();   // the weights above are constant; x is the previous kernel's
    if (!valid) return;
    float acc[MAX_T];
#pragma unroll
    for (int t = 0; t < MAX_T; ++t) acc[t] = 0.0f;
#pragma unroll
    for (int q = 0; q < NJ; ++q) {
        const int j = lane + 32 * q;
#pragma unroll
        for (int t = 0; t < MAX_T; ++t) {
            if (t < T) {
                const float* xt = x + (size_t) t * n;
                const float4 xa = *reinterpret_cast<const float4*>(xt + j * 8);
                const float4 xb = *reinterpret_cast<const float4*>(xt + j * 8 + 4);
                float a = acc[t];
                a = fmaf(__uint_as_float(wv[q].x << 16), xa.x, a); a = fmaf(__uint_as_float(wv[q].x & 0xffff0000u), xa.y, a);
                a = fmaf(__uint_as_float(wv[q].y << 16), xa.z, a); a = fmaf(__uint_as_float(wv[q].y & 0xffff0000u), xa.w, a);
                a = fmaf(__uint_as_float(wv[q].z << 16), xb.x, a); a = fmaf(__uint_as_float(wv[q].z & 0xffff0000u), xb.y, a);
                a = fmaf(__uint_as_float(wv[q].w << 16), xb.z, a); a = fmaf(__uint_as_float(wv[q].w & 0xffff0000u), xb.w, a);
                acc[t] = a;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < MAX_T; ++t) {
        if (t >= T) break;
        float a = acc[t];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane != 0) continue;
        if (is_beta) {
            beta[(size_t) t * h_v + r] = 1.0f / (1.0f + __expf(-a));
        } else {
            const float v = a + dt[r];
            const float sp = v > 20.0f ? v : log1pf(__expf(v));
            gate[(size_t) t * h_v + r] = sp * ssm_a[r];
        }
    }
}

// value head `head`'s recurrence over n tokens from `state` (written back when `write`), the gated RMS norm into y;
// S * RG threads: column col of row group rg
__device__ __forceinline__ void step_norm_head(int head, int col, int rg, float* __restrict__ state,
                                               const float* __restrict__ hbuf, int C, const float* __restrict__ gate,
                                               const float* __restrict__ beta, const float* __restrict__ z,
                                               const float* __restrict__ gamma, float eps, float* __restrict__ y,
                                               int h_k, int h_v, int n, bool write, int t_out_begin) {
    __shared__ float sk[S], sq[S];
    __shared__ float red[RG][S];
    __shared__ float wsum[S * RG / 32];
    const int tid = rg * S + col;
    const int qh = head % h_k;
    const int qk = S * h_k;             // q at [0, qk), k at [qk, 2qk), v at [2qk, ...)
    const int value_dim = S * h_v;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    for (int t = 0; t < n; ++t) {
        const float* ht = hbuf + (size_t) t * C;
        const bool need_out = (t >= t_out_begin);
        __syncthreads();                // the previous token is done with sk/sq/red/wsum
        if (tid < S) {
            sk[tid] = ht[qk + qh * S + tid];
            if (need_out) sq[tid] = ht[qh * S + tid];
        }
        __syncthreads();
        const float g = __expf(gate[(size_t) t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red[rg][col] = kv;
        __syncthreads();
        const float kv_col = red[0][col] + red[1][col] + red[2][col] + red[3][col];
        const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
        if (!need_out) {
#pragma unroll
            for (int r = 0; r < RPG; ++r) s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            continue;
        }
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        __syncthreads();
        red[rg][col] = o;
        __syncthreads();
        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) {
            oc = (red[0][col] + red[1][col] + red[2][col] + red[3][col]) * rsqrtf((float) S);
            sq_part = oc * oc;
        }
        for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);
        if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
        __syncthreads();
        if (rg == 0) {
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float scale = rsqrtf(ss / (float) S + eps);
            const float zz = z[(size_t) t * value_dim + head * S + col];
            y[(size_t) t * value_dim + head * S + col] = oc * scale * gamma[col] * (1.0f / (1.0f + __expf(-zz)));
        }
    }
    if (write && n > 0) {
#pragma unroll
        for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
    }
}

}  // namespace basalt::kernels::gdn_unfused
