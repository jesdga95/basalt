// include/basalt/kernels/gdn_commit.cuh - the GDN commit's device code (no kernels): the conv history and the
// recurrent state replayed over a window's accepted rows, from the rows dec_gdn_rec stored (gdn_window.hpp).
// Shared by the plain commit (src/kernels/cuda/gdn_commit.cu) and the commit variants a model schedules itself.
#pragma once

#include "basalt/kernels/gdn_window.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace basalt::kernels::gdn_dev {

constexpr int S = 128;          // GDN state size
constexpr int RG = 4;
constexpr int RPG = S / RG;

__device__ __forceinline__ void gdn_conv_commit_n(float* __restrict__ hist, const float* __restrict__ qkv, int C, int n,
                                                  int c) {
    if (c >= C) return;
    if (n <= 0) return;
    float seq[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = n + j;          // the last three of [hist(3) | x_0..x_{n-1}]
        seq[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
    }
    hist[c * 3] = seq[0];
    hist[c * 3 + 1] = seq[1];
    hist[c * 3 + 2] = seq[2];
}


// One state item's inputs, held in registers so the next item's loads are in flight while this one computes.
struct CommitPre {
    float s[RPG];
    float k[kDecMaxT], v[kDecMaxT], g[kDecMaxT], b[kDecMaxT];
};

__device__ __forceinline__ void commit_pre_load(const GdnCommitLayers& a, int per_layer, int n, int i, CommitPre& p) {
    const int l = i / per_layer, j = i - l * per_layer;
    const int head = j >> 2, cblk = j & 3;
    const float* hbuf = a.h + (size_t) l * a.h_stride;
    const float* gate = a.gate + (size_t) l * a.gb_stride;
    const float* beta = a.beta + (size_t) l * a.gb_stride;
    const int col = cblk * 32 + threadIdx.x;
    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int qk = S * a.h_k;
    const int qh = head % a.h_k;
#pragma unroll
    for (int t = 0; t < kDecMaxT; ++t) {
        if (t >= n) break;
        const float* ht = hbuf + (size_t) t * a.C;
        p.k[t] = ht[qk + qh * S + tid];
        p.v[t] = ht[2 * qk + head * S + col];
        p.g[t] = gate[(size_t) t * a.h_v + head];
        p.b[t] = beta[(size_t) t * a.h_v + head];
    }
    const float* base = a.state + (size_t) l * a.state_stride + ((size_t) (threadIdx.y * RPG) * a.h_v + head) * S + col;
    const size_t row_stride = (size_t) a.h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) p.s[r] = base[r * row_stride];
}

// The step-commit math (gdn_step_commit_block) on registers loaded earlier, every token's inputs up front.
__device__ __forceinline__ void commit_pre_run(const GdnCommitLayers& a, int per_layer, int n, int i, CommitPre& p) {
    __shared__ float sk[kDecMaxT][S];
    __shared__ float red[2][RG][32];
    const int l = i / per_layer, j = i - l * per_layer;
    const int head = j >> 2, cblk = j & 3;
    const int col_local = threadIdx.x, rg = threadIdx.y;
    const int col = cblk * 32 + col_local;
    const int tid = rg * 32 + col_local;
    __syncthreads();                    // the previous item is done with sk/red
#pragma unroll
    for (int t = 0; t < kDecMaxT; ++t) {
        if (t >= n) break;
        sk[t][tid] = p.k[t];
    }
    __syncthreads();
#pragma unroll
    for (int t = 0; t < kDecMaxT; ++t) {
        if (t >= n) break;
        const float gt = __expf(p.g[t]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(p.s[r], sk[t][rg * RPG + r], kv);
        red[t & 1][rg][col_local] = kv;
        __syncthreads();
        const float kv_col = red[t & 1][0][col_local] + red[t & 1][1][col_local] + red[t & 1][2][col_local] +
                             red[t & 1][3][col_local];
        const float delta = (p.v[t] - gt * kv_col) * p.b[t];
#pragma unroll
        for (int r = 0; r < RPG; ++r) p.s[r] = fmaf(gt, p.s[r], sk[t][rg * RPG + r] * delta);
    }
    float* base = a.state + (size_t) l * a.state_stride + ((size_t) (rg * RPG) * a.h_v + head) * S + col;
    const size_t row_stride = (size_t) a.h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * row_stride] = p.s[r];
}

// The grid-stride items with the next state item's loads issued before the current item runs.
__device__ __forceinline__ void commit_layers_body(const GdnCommitLayers& a, int n, int conv_blocks) {
    const int per_state = a.h_v * (S / 32), per_layer = per_state + conv_blocks, total = per_layer * a.n_layers;
    const int step = gridDim.x;
    CommitPre cur, nxt;
    bool cur_state = (int) blockIdx.x < total && (int) blockIdx.x % per_layer < per_state;
    if (cur_state) commit_pre_load(a, per_layer, n, blockIdx.x, cur);
    for (int i = blockIdx.x; i < total; i += step) {
        const int in = i + step;
        const bool next_state = in < total && in % per_layer < per_state;
        if (next_state) commit_pre_load(a, per_layer, n, in, nxt);
        if (cur_state) {
            commit_pre_run(a, per_layer, n, i, cur);
        } else {
            const int l = i / per_layer, j = i - l * per_layer;
            gdn_conv_commit_n(a.state + (size_t) l * a.state_stride + (size_t) S * a.h_v * S,
                              a.qkv + (size_t) l * a.h_stride, a.C, n,
                              (j - per_state) * 32 * RG + threadIdx.y * 32 + threadIdx.x);
        }
        cur_state = next_state;
        if (next_state) cur = nxt;
    }
}

}  // namespace basalt::kernels::gdn_dev
