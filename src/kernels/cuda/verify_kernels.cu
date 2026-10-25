// src/kernels/cuda/verify_kernels.cu - see include/basalt/kernels/verify_kernels.hpp.
//
// The per-token arithmetic of every kernel here is transcribed from its single-token original (fused_gdn.cu,
// elementwise.cu) with the same operation order, so a verify window reproduces plain decode bit for bit.
#include "basalt/kernels/verify_kernels.hpp"
#include "basalt/kernels/dp4a.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/core/draft_cum.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "basalt/log.hpp"

namespace basalt::kernels {
namespace {

constexpr int S = 128;          // GDN state size
constexpr int RG = 4;
constexpr int RPG = S / RG;

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { basalt::log::error("basalt", "%s: %s", what, cudaGetErrorString(e)); std::exit(1); }
}

__global__ void __launch_bounds__(S) gdn_conv_l2_multi_kernel(const float* __restrict__ hist,
                                                              const float* __restrict__ qkv,
                                                              const float* __restrict__ w, float* __restrict__ h,
                                                              int C, int qk_heads, float eps, int t_begin) {
    pdl_wait();
    __shared__ float part[S / 32];
    const int t = t_begin + blockIdx.y;
    const int c = blockIdx.x * S + threadIdx.x;
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
    if ((int) blockIdx.x < qk_heads) {
        float sq = y * y;
        for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
        if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
        __syncthreads();
        const float ss = part[0] + part[1] + part[2] + part[3];
        y *= rsqrtf(ss + eps);
    }
    h[(size_t) t * C + c] = y;
}

// gdn_conv_l2_multi_kernel over a segmented window: row t = blockIdx.y reads its own segment's history and rows
__global__ void __launch_bounds__(S) gdn_conv_l2_seg_kernel(const SegRow* __restrict__ rows, size_t hist_off,
                                                            const float* __restrict__ qkv, const float* __restrict__ w,
                                                            float* __restrict__ h, int C, int qk_heads, float eps) {
    pdl_wait();
    __shared__ float part[S / 32];
    const int t = blockIdx.y;
    const SegRow r = rows[t];
    const float* hist = r.gdn + hist_off;
    const int c = blockIdx.x * S + threadIdx.x;
    float win[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = r.j + j;
        win[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t) (r.row0 + src - 3) * C + c];
    }
    const float v0 = win[0], v1 = win[1], v2 = win[2], x = qkv[(size_t) t * C + c];
    float sum = v0 * w[c * 4] + v1 * w[c * 4 + 1] + v2 * w[c * 4 + 2] + x * w[c * 4 + 3];
    float y = sum / (1.0f + __expf(-sum));
    if ((int) blockIdx.x < qk_heads) {
        float sq = y * y;
        for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
        if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
        __syncthreads();
        const float ss = part[0] + part[1] + part[2] + part[3];
        y *= rsqrtf(ss + eps);
    }
    h[(size_t) t * C + c] = y;
}

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

__device__ __forceinline__ void gdn_conv_commit_one(float* __restrict__ hist, const float* __restrict__ qkv, int C,
                                                    const int32_t* __restrict__ n_keep, int c) {
    gdn_conv_commit_n(hist, qkv, C, *n_keep, c);
}

__global__ void gdn_conv_commit_kernel(float* __restrict__ hist, const float* __restrict__ qkv, int C,
                                       const int32_t* __restrict__ n_keep) {
    gdn_conv_commit_one(hist, qkv, C, n_keep, blockIdx.x * blockDim.x + threadIdx.x);
}

template <int MAX_T = kVerifyMaxT>
__global__ void __launch_bounds__(64) gdn_ab_multi_kernel(const float* __restrict__ x, const uint16_t* __restrict__ wa,
                                                          const uint16_t* __restrict__ wb,
                                                          const float* __restrict__ dt,
                                                          const float* __restrict__ ssm_a, float* __restrict__ gate,
                                                          float* __restrict__ beta, int n, int h_v, int T) {
    pdl_wait();
    const int row = blockIdx.x * 2 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= 2 * h_v) return;
    const bool is_beta = row >= h_v;
    const int r = is_beta ? row - h_v : row;
    const uint4* w4 = reinterpret_cast<const uint4*>((is_beta ? wb : wa) + (size_t) r * n);
    float acc[MAX_T];
#pragma unroll
    for (int t = 0; t < MAX_T; ++t) acc[t] = 0.0f;
    for (int j = lane; j < n / 8; j += 32) {
        const uint4 wv = __ldg(w4 + j);
#pragma unroll
        for (int t = 0; t < MAX_T; ++t) {
            if (t >= T) break;
            const float* xt = x + (size_t) t * n;
            const float4 xa = *reinterpret_cast<const float4*>(xt + j * 8);
            const float4 xb = *reinterpret_cast<const float4*>(xt + j * 8 + 4);
            float a = acc[t];
            a = fmaf(__uint_as_float(wv.x << 16), xa.x, a); a = fmaf(__uint_as_float(wv.x & 0xffff0000u), xa.y, a);
            a = fmaf(__uint_as_float(wv.y << 16), xa.z, a); a = fmaf(__uint_as_float(wv.y & 0xffff0000u), xa.w, a);
            a = fmaf(__uint_as_float(wv.z << 16), xb.x, a); a = fmaf(__uint_as_float(wv.z & 0xffff0000u), xb.y, a);
            a = fmaf(__uint_as_float(wv.w << 16), xb.z, a); a = fmaf(__uint_as_float(wv.w & 0xffff0000u), xb.w, a);
            acc[t] = a;
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

// gdn_ab_multi_kernel for rows of exactly 32 * NJ uint4: every weight load issued before the sums and the token loop
// unrolled, so a warp waits for memory once instead of once per step.  The same sums in the same order: the same bits.
template <int MAX_T, int NJ>
__global__ void __launch_bounds__(64) gdn_ab_multi_unrolled_kernel(const float* __restrict__ x, const uint16_t* __restrict__ wa,
                                                                   const uint16_t* __restrict__ wb,
                                                                   const float* __restrict__ dt,
                                                                   const float* __restrict__ ssm_a, float* __restrict__ gate,
                                                                   float* __restrict__ beta, int n, int h_v, int T) {
    const int row = blockIdx.x * 2 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
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

// State-only commit kernel: each head's 128 independent state columns are split across 4 blocks of 32 columns
// (48 * 4 = 192 blocks of 128 threads across all SMs, vs 48 blocks of 512 threads), with no sq/o/norm/z/y work.
__device__ __forceinline__ void gdn_step_commit_block(float* __restrict__ state, const float* __restrict__ hbuf,
                                                      int C, const float* __restrict__ gate,
                                                      const float* __restrict__ beta, int h_k, int h_v,
                                                      const int32_t* __restrict__ n_keep, int head, int cblk) {
    const int n = *n_keep;
    if (n <= 0) return;
    __shared__ float sk[S];
    __shared__ float red[RG][32];
    const int col_local = threadIdx.x;
    const int col = cblk * 32 + col_local;
    const int rg = threadIdx.y;
    const int tid = rg * 32 + col_local;    // 0 .. 127
    const int qh = head % h_k;
    const int qk = S * h_k;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    for (int t = 0; t < n; ++t) {
        const float* ht = hbuf + (size_t) t * C;
        __syncthreads();
        sk[tid] = ht[qk + qh * S + tid];
        __syncthreads();
        const float g = __expf(gate[(size_t) t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red[rg][col_local] = kv;
        __syncthreads();
        const float kv_col = red[0][col_local] + red[1][col_local] + red[2][col_local] + red[3][col_local];
        const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
#pragma unroll
        for (int r = 0; r < RPG; ++r) s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
}

__global__ void __launch_bounds__(32 * RG) gdn_step_commit_kernel(float* __restrict__ state,
                                                                  const float* __restrict__ hbuf, int C,
                                                                  const float* __restrict__ gate,
                                                                  const float* __restrict__ beta,
                                                                  int h_k, int h_v,
                                                                  const int32_t* __restrict__ n_keep) {
    gdn_step_commit_block(state, hbuf, C, gate, beta, h_k, h_v, n_keep, blockIdx.x, blockIdx.y);
}

// One state item's inputs, held in registers so the next item's loads are in flight while this one computes.
struct CommitPre {
    float s[RPG];
    float k[kVerifyMaxT], v[kVerifyMaxT], g[kVerifyMaxT], b[kVerifyMaxT];
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
    for (int t = 0; t < kVerifyMaxT; ++t) {
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
    __shared__ float sk[kVerifyMaxT][S];
    __shared__ float red[2][RG][32];
    const int l = i / per_layer, j = i - l * per_layer;
    const int head = j >> 2, cblk = j & 3;
    const int col_local = threadIdx.x, rg = threadIdx.y;
    const int col = cblk * 32 + col_local;
    const int tid = rg * 32 + col_local;
    __syncthreads();                    // the previous item is done with sk/red
#pragma unroll
    for (int t = 0; t < kVerifyMaxT; ++t) {
        if (t >= n) break;
        sk[t][tid] = p.k[t];
    }
    __syncthreads();
#pragma unroll
    for (int t = 0; t < kVerifyMaxT; ++t) {
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

// ---- the commit's items claimed from a device counter: the commit kernel and the finisher share them.
// A counter set is four words on four lines: claim, done, started (the epoch whose begin kernel ran), pre (the epoch
// whose indexer and PLE commits have completed).
constexpr int kCtlWords = 32;
__device__ __forceinline__ uint32_t* ctl_word(CommitCtl* c, int set, int which) {
    return c->w + (size_t) (set * 4 + which) * kCtlWords;
}
__device__ __forceinline__ uint32_t ld_acquire(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_release(uint32_t* p, uint32_t v) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned long long globaltimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
// Block-wide wait until *p == want (bounded: a commit that never ran traps instead of hanging).
__device__ __forceinline__ void spin_until_eq(const uint32_t* p, uint32_t want) {
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        const unsigned long long t0 = globaltimer();
        while (ld_acquire(p) != want) {
            __nanosleep(64);
            if (globaltimer() - t0 > 2000000000ull) {
                printf("basalt: error: commit finisher: block %d waited 2 s for %u (word %u)\n", (int) blockIdx.x, want, ld_acquire(p));
                __trap();
            }
        }
    }
    __syncthreads();
}

// Items claimed one counter step ahead of the loads, which are one item ahead of the run: the next item's loads and the
// claim after it are in flight while this item computes.  Finished items are added to `done` four at a time (one fence
// per batch) and at the end.
__device__ __forceinline__ void commit_drain(const GdnCommitLayers& a, int n, int conv_blocks, uint32_t* claim,
                                             uint32_t* done) {
    const int per_state = a.h_v * (S / 32), per_layer = per_state + conv_blocks, total = per_layer * a.n_layers;
    __shared__ uint32_t s_c[2];
    const bool lead = threadIdx.x == 0 && threadIdx.y == 0;
    if (lead) {
        s_c[0] = atomicAdd(claim, 1u);
        s_c[1] = atomicAdd(claim, 1u);
    }
    __syncthreads();
    int i = (int) s_c[0], in = (int) s_c[1];
    __syncthreads();
    CommitPre cur, nxt;
    bool cur_state = i < total && i % per_layer < per_state;
    if (cur_state) commit_pre_load(a, per_layer, n, i, cur);
    int unpub = 0;
    while (i < total) {
        uint32_t nn = 0;
        if (lead) nn = atomicAdd(claim, 1u);
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
        const bool flush = ++unpub == 4;
        if (flush) __syncthreads();
        if (lead) {
            if (flush) {
                __threadfence();
                atomicAdd(done, 4u);
            }
            s_c[0] = nn;
        }
        if (flush) unpub = 0;
        __syncthreads();
        i = in;
        in = (int) s_c[0];
        cur_state = next_state;
        if (next_state) cur = nxt;
    }
    if (unpub > 0) {
        __syncthreads();
        if (lead) {
            __threadfence();
            atomicAdd(done, (uint32_t) unpub);
        }
    }
}

// The epoch's counters are reset here, before anything of the epoch is claimed; `started` publishes the epoch, and the
// host word `begun` tells the host that the device side knows it.
__global__ void gdn_commit_begin_kernel(CommitCtl* ctl, const int32_t* __restrict__ epoch_src,
                                        volatile uint32_t* begun) {
    const uint32_t e = (uint32_t) *(const volatile int32_t*) epoch_src;
    *ctl_word(ctl, (int) (e & 1u), 0) = 0;
    *ctl_word(ctl, (int) (e & 1u), 1) = 0;
    ctl->w[kCtlWords * 8] = e;
    ctl->w[kCtlWords * 10] = 0;   // the lazy layers' claims (gdn_commit_lazy)
    __threadfence();
    st_release(ctl_word(ctl, (int) (e & 1u), 2), e);
    *begun = e;
}

__global__ void __launch_bounds__(32 * RG) gdn_commit_layers_dyn_kernel(GdnCommitLayers a, int conv_blocks,
                                                                       CommitCtl* ctl) {
    const uint32_t e = ctl->w[kCtlWords * 8];
    const int set = (int) (e & 1u);
    if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) st_release(ctl_word(ctl, set, 3), e);
    commit_drain(a, min(*a.n_keep, kVerifyMaxT), conv_blocks, ctl_word(ctl, set, 0), ctl_word(ctl, set, 1));
}

// The first node of a window: when the newest epoch has not been finished by a finisher yet, claims what the commit
// kernel has not and returns once every item of the epoch is done and its indexer and PLE commits have completed; the
// epoch is then marked finished.  The host has seen the epoch's begin kernel run before it launches the window.
__global__ void __launch_bounds__(32 * RG) gdn_commit_finish_kernel(GdnCommitLayers a, int conv_blocks, CommitCtl* ctl) {
    __shared__ uint32_t s_e, s_f;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_e = ld_acquire(ctl->w + (size_t) kCtlWords * 8);
        s_f = ld_acquire(ctl->w + (size_t) kCtlWords * 9);
    }
    __syncthreads();
    const uint32_t e = s_e;
    if (e == s_f) return;
    const int set = (int) (e & 1u);
    spin_until_eq(ctl_word(ctl, set, 2), e);
    const int n = min(__ldcg(a.n_keep), kVerifyMaxT);
    commit_drain(a, n, conv_blocks, ctl_word(ctl, set, 0), ctl_word(ctl, set, 1));
    const int total = (a.h_v * (S / 32) + conv_blocks) * a.n_layers;
    spin_until_eq(ctl_word(ctl, set, 1), (uint32_t) total);
    spin_until_eq(ctl_word(ctl, set, 3), e);
    if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        __threadfence();
        st_release(ctl->w + (size_t) kCtlWords * 9, e);
    }
}

__global__ void __launch_bounds__(32 * RG) gdn_commit_layers_pipe_kernel(GdnCommitLayers a, int conv_blocks) {
    const int n = min(*a.n_keep, kVerifyMaxT);
    if (n <= 0) return;
    commit_layers_body(a, n, conv_blocks);
}

// The lazy layers' items (layer-major), claimed from the epoch's lazy counter (line 10, reset by the begin kernel): every
// item of the first `must_layers`, then more while *stop != stop_value.  A block claims only after that check, and
// commits every item it claims.
constexpr int kLazyPfBlocks = 32;          // the blocks that read ahead into L2
constexpr uint64_t kLazyPfStep = 128 << 10;  // bytes a block requests before it looks at the stop word again
__global__ void __launch_bounds__(32 * RG) gdn_commit_lazy_kernel(GdnCommitLayers a, int conv_blocks, CommitCtl* ctl,
                                                                  int must_layers, const uint32_t* stop,
                                                                  uint32_t stop_value, GdnLazyPf pf) {
    const int per_state = a.h_v * (S / 32), per_layer = per_state + conv_blocks, total = per_layer * a.n_layers;
    const int must = per_layer * must_layers;
    uint32_t* const claim = ctl->w + (size_t) kCtlWords * 10;
    __shared__ int s_i;
    const bool lead = threadIdx.x == 0 && threadIdx.y == 0;
    const int n = min(__ldcg(a.n_keep), kVerifyMaxT);
    for (;;) {
        if (lead) {
            const int c = (int) min(__ldcg(claim), (uint32_t) total);
            const bool more = c < must || (stop != nullptr && ld_acquire(stop) != stop_value);
            s_i = c < total && more ? (int) atomicAdd(claim, 1u) : total;
        }
        __syncthreads();
        const int i = s_i;
        __syncthreads();
        if (i >= total) break;
        const int l = i / per_layer, j = i - l * per_layer;
        if (j < per_state) {
            CommitPre p;
            commit_pre_load(a, per_layer, n, i, p);
            commit_pre_run(a, per_layer, n, i, p);
        } else {
            gdn_conv_commit_n(a.state + (size_t) l * a.state_stride + (size_t) S * a.h_v * S,
                              a.qkv + (size_t) l * a.h_stride, a.C, n,
                              (j - per_state) * 32 * RG + threadIdx.y * 32 + threadIdx.x);
        }
    }
    // the next layer's weights into L2 while the combine waits for the pool (only when the layer has entries off CUDA0)
    if (stop == nullptr || pf.n[0] == 0 || (int) blockIdx.x >= kLazyPfBlocks) return;
    if (threadIdx.y == 0) {
        bool off = false;
        for (int i = threadIdx.x; i < pf.n_ids; i += 32) {
            const int id = __ldcg(pf.ids + i);
            off |= id >= 0 && id < pf.n_expert && __ldcg(pf.ptab + id) == 0ull;
        }
        off = __any_sync(0xffffffffu, off);
        if (threadIdx.x == 0) s_i = off ? 1 : 0;
    }
    __syncthreads();
    if (s_i == 0 || !lead) return;
    const uint32_t waiting = stop_value | 0x80000000u;
    uint32_t v;
    while ((v = ld_acquire(stop)) != waiting && v != stop_value) __nanosleep(256);
    if (v != waiting) return;
    // step k of every block covers [k * blocks + b, +1) * step of the ranges laid end to end; after its requests a block
    // reads the step's last word (it lands once that part is in L2), then looks at the stop word again
    uint64_t total_b = 0;
    for (int r = 0; r < kLazyPfRanges; ++r) total_b += pf.n[r];
    uint32_t acc = 0;
    for (uint64_t o0 = (uint64_t) blockIdx.x * kLazyPfStep; o0 < total_b; o0 += (uint64_t) kLazyPfBlocks * kLazyPfStep) {
        const uint64_t o1 = min(o0 + kLazyPfStep, total_b);
        const char* last = nullptr;
        for (uint64_t o = o0; o < o1;) {
            int r = 0;
            uint64_t ro = o;
            while (r < kLazyPfRanges - 1 && ro >= pf.n[r]) { ro -= pf.n[r]; ++r; }
            const uint64_t take = min(min(o1 - o, pf.n[r] - ro), (uint64_t) 32768);
            const char* src = (const char*) pf.p[r] + ro;
            if (take >= 16) {
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"((unsigned) (take & ~15ull)) : "memory");
                last = src + (take & ~15ull) - 16;
            }
            o += take;
        }
        if (last != nullptr) {
            const uint4 w = __ldcg((const uint4*) last);
            acc ^= w.x ^ w.y ^ w.z ^ w.w;
        }
        if (ld_acquire(stop) != waiting) break;
    }
    if (acc == 0x9e3779b9u) ctl->w[(size_t) kCtlWords * 10 + 1] = acc;   // keeps the reads (never in practice)
}

// the same over a segmented window: blockIdx.y = the window row whose segment it commits (only a segment's first row)
__global__ void __launch_bounds__(32 * RG) gdn_commit_layers_seg_kernel(GdnCommitSeg s, int conv_blocks) {
    const SegRow r = s.rows[blockIdx.y];
    const int n = min(r.n_keep, kVerifyMaxT);
    if (r.j != 0 || n <= 0) return;
    GdnCommitLayers a;
    a.state = r.gdn + s.state_off;
    a.h = s.h + (size_t) r.row0 * s.C; a.qkv = s.qkv + (size_t) r.row0 * s.C;
    a.gate = s.gate + (size_t) r.row0 * s.h_v; a.beta = s.beta + (size_t) r.row0 * s.h_v;
    a.state_stride = s.state_stride; a.h_stride = s.h_stride; a.gb_stride = s.gb_stride;
    a.C = s.C; a.h_k = s.h_k; a.h_v = s.h_v; a.n_layers = s.n_layers;
    commit_layers_body(a, n, conv_blocks);
}

// n tokens from `state` (written back when `write`)
__device__ __forceinline__ void gdn_step_norm_body(float* __restrict__ state, const float* __restrict__ hbuf, int C,
                                                   const float* __restrict__ gate, const float* __restrict__ beta,
                                                   const float* __restrict__ z, const float* __restrict__ gamma,
                                                   float eps, float* __restrict__ y, int h_k, int h_v, int n,
                                                   bool write, int t_out_begin) {
    __shared__ float sk[S], sq[S];
    __shared__ float red[RG][S];
    __shared__ float wsum[S * RG / 32];
    const int head = blockIdx.x;
    const int col = threadIdx.x;
    const int rg = threadIdx.y;
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

__global__ void __launch_bounds__(S * RG) gdn_step_norm_multi_kernel(float* __restrict__ state,
                                                                     const float* __restrict__ hbuf, int C,
                                                                     const float* __restrict__ gate,
                                                                     const float* __restrict__ beta,
                                                                     const float* __restrict__ z,
                                                                     const float* __restrict__ gamma, float eps,
                                                                     float* __restrict__ y, int h_k, int h_v, int T,
                                                                     const int32_t* __restrict__ n_keep, int t_out_begin) {
    pdl_wait();
    gdn_step_norm_body(state, hbuf, C, gate, beta, z, gamma, eps, y, h_k, h_v, n_keep ? *n_keep : T, n_keep != nullptr,
                       t_out_begin);
}

// the verify form over a segmented window: block row y is a segment's first row; the state is only read
__global__ void __launch_bounds__(S * RG) gdn_step_norm_seg_kernel(const SegRow* __restrict__ rows, size_t state_off,
                                                                   const float* __restrict__ hbuf, int C,
                                                                   const float* __restrict__ gate,
                                                                   const float* __restrict__ beta,
                                                                   const float* __restrict__ z,
                                                                   const float* __restrict__ gamma, float eps,
                                                                   float* __restrict__ y, int h_k, int h_v) {
    pdl_wait();
    const SegRow r = rows[blockIdx.y];
    if (r.j != 0) return;
    const size_t vd = (size_t) S * h_v;
    gdn_step_norm_body(r.gdn + state_off, hbuf + (size_t) r.row0 * C, C, gate + (size_t) r.row0 * h_v,
                       beta + (size_t) r.row0 * h_v, z + r.row0 * vd, gamma, eps, y + r.row0 * vd, h_k, h_v, r.n, false, 0);
}

__global__ void embedding_gather_dev_kernel(const uint8_t* __restrict__ codes, const float* __restrict__ scales,
                                            const float* __restrict__ offsets, const int32_t* __restrict__ tokens,
                                            int64_t n, int code_bits, int code_bias, int group_elems,
                                            unsigned long long row_codes, unsigned long long row_groups,
                                            float* __restrict__ out) {
    const int t = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned long long token = (unsigned long long) tokens[t];
    const uint8_t* c = codes + token * row_codes;
    const float* sc = scales + token * row_groups;
    const float* of = offsets ? offsets + token * row_groups : nullptr;
    const int per_byte = 8 / code_bits;
    const unsigned mask = (1u << code_bits) - 1u;
    const int code = (c[i / per_byte] >> ((i % per_byte) * code_bits)) & mask;
    const int64_t group = i / group_elems;
    const float product = __fmul_rn((float) (code + code_bias), sc[group]);
    out[(size_t) t * n + i] = __fadd_rn(product, of ? of[group] : 0.0f);
}

__global__ void broadcast_streams_kernel(const float* __restrict__ x, float* __restrict__ R, int64_t n, int hc) {
    const int t = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * hc) return;
    R[(size_t) t * n * hc + i] = x[(size_t) t * n + i % n];
}

__global__ void copy_indexed_kernel(float* __restrict__ dst, const float* __restrict__ src, int64_t stride,
                                    const int32_t* __restrict__ index, int64_t n) {
    const int idx = *index;
    if (idx < 0) return;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        dst[i] = src[(size_t) idx * stride + i];
}

// dir 0: work row t <- slot's array.  dir 1: slot's array <- work row (row0 + n_keep - 1), segment first rows only.
__global__ void ple_hist_seg_kernel(float* __restrict__ work, float* const* __restrict__ slots,
                                    const SegRow* __restrict__ rows, int64_t n, int dir) {
    const int t = blockIdx.y;
    const SegRow r = rows[t];
    float* slot = slots[r.slot];
    const float* src = nullptr;
    float* dst = nullptr;
    if (dir == 0) {
        src = slot;
        dst = work + (size_t) t * n;
    } else {
        if (r.j != 0 || r.n_keep <= 0) return;
        src = work + (size_t) (r.row0 + r.n_keep - 1) * n;
        dst = slot;
    }
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        dst[i] = src[i];
}

__global__ void fetch_blobs_kernel(const unsigned long long* __restrict__ src, const int32_t* __restrict__ n,
                                   uint4* __restrict__ dst, long long per) {
    const long long total = (long long) *n * per;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long long) gridDim.x * blockDim.x) {
        const long long k = i / per, off = i - k * per;
        dst[i] = ((const uint4*) src[k])[off];
    }
}

__global__ void rebase_ptrs_kernel(unsigned long long* ptr, const int32_t* n, unsigned long long base, long long bytes) {
    const int k = threadIdx.x;
    if (k < *n) ptr[k] = base + (unsigned long long) k * (unsigned long long) bytes;
}

__global__ void add_streams_broadcast_kernel(const float* __restrict__ h, const float* __restrict__ e,
                                             float* __restrict__ R, int64_t n, int hc) {
    const int t = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * hc) return;
    R[(size_t) t * n * hc + i] = h[(size_t) t * n * hc + i] + e[(size_t) t * n + i % n];
}

__global__ void ident_hits_kernel(const int32_t* __restrict__ ids, int n, int32_t* __restrict__ slot,
                                  int32_t* __restrict__ dst, int32_t* __restrict__ count) {
    const int i = threadIdx.x;
    if (i < n) { slot[i] = ids[i]; dst[i] = i; }
    if (i == 0) *count = n;
}

// E = the widest element the row size divides into (16, 4 or 1 bytes): a Q6_K head row of 2560 values is 2100 bytes
template<typename E>
__global__ void gather_rows_kernel(const E* __restrict__ src, long long row_e, const int32_t* __restrict__ ids,
                                   long long n, E* __restrict__ dst) {
    const long long total = n * row_e;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long long) gridDim.x * blockDim.x) {
        const long long r = i / row_e, o = i - r * row_e;
        dst[i] = src[(long long) ids[r] * row_e + o];
    }
}

__global__ void map_ids_kernel(int32_t* ids, const int32_t* __restrict__ table, int n) {
    const int i = threadIdx.x;
    if (i < n) ids[i] = table[ids[i]];
}

__global__ void row_top_prob_kernel(const float* __restrict__ logits, int n_vocab, const int32_t* __restrict__ ids,
                                    float* __restrict__ probs) {
    __shared__ float part[32];
    const int t = blockIdx.x;
    const float* l = logits + (size_t) t * n_vocab;
    const float m = l[ids[t]];
    float s = 0.0f;
    for (int i = threadIdx.x; i < n_vocab; i += blockDim.x) s += __expf(l[i] - m);
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        for (int w = 0; w < (int) (blockDim.x >> 5); ++w) tot += part[w];
        probs[t] = 1.0f / tot;
    }
}

__global__ void dense_steps_kernel(const int32_t* __restrict__ cells, int n, int32_t* __restrict__ steps) {
    const int i = threadIdx.x;
    if (i >= n) return;
    const int c = cells[i];
    steps[i * 4 + 0] = c;
    steps[i * 4 + 1] = c + 1;
    steps[i * 4 + 2] = (c + 1) / 4;
    steps[i * 4 + 3] = c + 1;
}

}  // namespace

void fetch_blobs(const unsigned long long* src, const int32_t* n, uint8_t* dst, int64_t blob_bytes, int cap, void* stream) {
    if (cap <= 0) return;
    if (blob_bytes % 16 != 0) { basalt::log::error("basalt", "fetch_blobs: blob size must be a multiple of 16"); std::exit(1); }
    fetch_blobs_kernel<<<48 * 8, 256, 0, (cudaStream_t) stream>>>(src, n, (uint4*) dst, (long long) (blob_bytes / 16));
    check("fetch_blobs");
}

void rebase_ptrs(unsigned long long* ptr, const int32_t* n, uint8_t* base, int64_t blob_bytes, void* stream) {
    rebase_ptrs_kernel<<<1, 128, 0, (cudaStream_t) stream>>>(ptr, n, (unsigned long long) base, (long long) blob_bytes);
    check("rebase_ptrs");
}

void add_streams_broadcast(const float* h, const float* e, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    add_streams_broadcast_kernel<<<dim3((unsigned) ((n_embd * hc + 255) / 256), (unsigned) n_tok), 256, 0,
                                   (cudaStream_t) stream>>>(h, e, R, n_embd, hc);
    check("add_streams_broadcast");
}

void ident_hits(const int32_t* ids, int n, int32_t* slot, int32_t* dst, int32_t* count, void* stream) {
    if (n < 1 || n > 1024) { basalt::log::error("basalt", "ident_hits: n out of range"); std::exit(1); }
    ident_hits_kernel<<<1, 1024, 0, (cudaStream_t) stream>>>(ids, n, slot, dst, count);
    check("ident_hits");
}

void gather_rows(const uint8_t* src, int64_t row_bytes, const int32_t* ids, int64_t n, uint8_t* dst, void* stream) {
    cudaStream_t s = (cudaStream_t) stream;
    if (row_bytes % 16 == 0)
        gather_rows_kernel<<<48 * 8, 256, 0, s>>>((const uint4*) src, row_bytes / 16, ids, n, (uint4*) dst);
    else if (row_bytes % 4 == 0)
        gather_rows_kernel<<<48 * 8, 256, 0, s>>>((const uint32_t*) src, row_bytes / 4, ids, n, (uint32_t*) dst);
    else
        gather_rows_kernel<<<48 * 8, 256, 0, s>>>(src, row_bytes, ids, n, dst);
    check("gather_rows");
}

void map_ids(int32_t* ids, const int32_t* table, int n, void* stream) {
    map_ids_kernel<<<1, 64, 0, (cudaStream_t) stream>>>(ids, table, n);
    check("map_ids");
}

void row_top_prob(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void* stream) {
    row_top_prob_kernel<<<n_rows, 1024, 0, (cudaStream_t) stream>>>(logits, n_vocab, ids, probs);
    check("row_top_prob");
}

namespace {
__global__ void mtp_take_row_kernel(float* R, long long rs, float* inj, long long is, float* mixed, long long ms,
                                    const int32_t* __restrict__ row_dev) {
    const int row = *row_dev;
    if (row <= 0) return;
    const long long total = rs + is + ms;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long long) gridDim.x * blockDim.x) {
        if (i < rs) R[i] = R[(long long) row * rs + i];
        else if (i < rs + is) inj[i - rs] = inj[(long long) row * is + (i - rs)];
        else mixed[i - rs - is] = mixed[(long long) row * ms + (i - rs - is)];
    }
}
}  // namespace

void mtp_take_row(float* R, int64_t R_stride, float* inj, int64_t inj_stride, float* mixed, int64_t mixed_stride,
                  const int32_t* row_dev, void* stream) {
    const long long total = (long long) (R_stride + inj_stride + mixed_stride);
    const unsigned blocks = (unsigned) ((total + 255) / 256 < 64 ? (total + 255) / 256 : 64);
    mtp_take_row_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>(R, (long long) R_stride, inj, (long long) inj_stride,
                                                                   mixed, (long long) mixed_stride, row_dev);
    check("mtp_take_row");
}

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12040
namespace {
struct CondHandles { unsigned long long h[8]; };
__global__ void mtp_draft_cond_kernel(CondHandles hs, int n, const float* __restrict__ probs,
                                      const int32_t* __restrict__ row, const int32_t* __restrict__ th_bits,
                                      const float* hist, int j) {
    const float p = probs[*row];
    // the same products, in the same order, as draft_keep on the host
    float c = 1.0f;
    for (int i = 0; i < j; ++i) c = c * basalt::core::draft_cum_acc(((const volatile float*) hist)[i]);
    c = c * basalt::core::draft_cum_acc(p);
    const bool keep = c >= __int_as_float(*th_bits);
    cudaGraphSetConditional((cudaGraphConditionalHandle) hs.h[0], keep ? 1u : 0u);
    for (int k = 1; k < n; ++k) cudaGraphSetConditional((cudaGraphConditionalHandle) hs.h[k], 0u);
}
}  // namespace

bool mtp_draft_cond(const unsigned long long* h, int n, const float* probs, const int32_t* row,
                    const int32_t* th_bits, void* stream, const float* hist, int j) {
    if (n < 1 || n > 8) { basalt::log::error("basalt", "mtp_draft_cond: n out of range"); std::exit(1); }
    CondHandles hs{};
    for (int k = 0; k < n; ++k) hs.h[k] = h[k];
    mtp_draft_cond_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(hs, n, probs, row, th_bits, hist, j);
    check("mtp_draft_cond");
    return true;
}
#else
bool mtp_draft_cond(const unsigned long long*, int, const float*, const int32_t*, const int32_t*, void*, const float*,
                    int) { return false; }
#endif

namespace {
struct MtpStageArgs {
    MtpStageSeg s[6];
    int ns;
    float4* fdst;
    const volatile float4* fsrc;
    long long n4;
};
__global__ void mtp_stage_kernel(MtpStageArgs a) {
    if ((int) blockIdx.x < a.ns) {
        const MtpStageSeg g = a.s[blockIdx.x];
        for (int i = threadIdx.x; i < g.n; i += blockDim.x) g.dst[i] = ((const volatile int32_t*) g.src)[i];
        return;
    }
    const int nb = (int) gridDim.x - a.ns;
    for (long long i = (long long) (blockIdx.x - a.ns) * blockDim.x + threadIdx.x; i < a.n4;
         i += (long long) nb * blockDim.x) {
        const float4 v = const_cast<const float4*>(a.fsrc)[i];
        a.fdst[i] = v;
    }
}
}  // namespace

void mtp_stage(const MtpStageSeg* segs, int n_segs, float* f_dst, const float* f_src, int64_t n_f, void* stream) {
    if (n_segs < 0 || n_segs > 6 || n_f % 4 != 0 || ((uintptr_t) f_dst | (uintptr_t) f_src) % 16 != 0) {
        basalt::log::error("basalt", "mtp_stage: bad arguments");
        std::exit(1);
    }
    MtpStageArgs a{};
    for (int i = 0; i < n_segs; ++i) a.s[i] = segs[i];
    a.ns = n_segs;
    a.fdst = (float4*) f_dst;
    a.fsrc = (const volatile float4*) f_src;
    a.n4 = n_f / 4;
    const int fb = n_f > 0 ? (int) std::min<long long>((a.n4 + 255) / 256, 30) : 0;
    mtp_stage_kernel<<<n_segs + fb, 256, 0, (cudaStream_t) stream>>>(a);
    check("mtp_stage");
}

namespace {
struct StepEndCond { unsigned long long h[8]; };
__global__ void mtp_step_end_kernel(const int32_t* __restrict__ ids, const int32_t* __restrict__ row_dev,
                                    int32_t* __restrict__ tok_dst, int32_t* out, int j, const float* probs, float* out_p,
                                    StepEndCond hs, int n, const int32_t* __restrict__ th_bits,
                                    const float* hist) {
    const int row = *row_dev;
    const int32_t tok = ids[row];
    const float p = probs[row];
    *tok_dst = tok;
    ((volatile float*) out_p)[j] = p;
    __threadfence_system();
    ((volatile int32_t*) out)[j] = tok;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12040
    if (n <= 0) return;
    // the same products, in the same order, as draft_keep on the host
    float c = 1.0f;
    for (int i = 0; i < j; ++i) c = c * basalt::core::draft_cum_acc(((const volatile float*) hist)[i]);
    c = c * basalt::core::draft_cum_acc(p);
    const bool keep = c >= __int_as_float(*th_bits);
    cudaGraphSetConditional((cudaGraphConditionalHandle) hs.h[0], keep ? 1u : 0u);
    for (int k = 1; k < n; ++k) cudaGraphSetConditional((cudaGraphConditionalHandle) hs.h[k], 0u);
#endif
}
}  // namespace

bool mtp_step_end(const int32_t* ids, const int32_t* row_dev, int32_t* tok_dst, int32_t* out, int j, const float* probs,
                  float* out_p, const unsigned long long* h, int n, const int32_t* th_bits,
                  const float* hist, void* stream) {
#if !(defined(CUDART_VERSION) && CUDART_VERSION >= 12040)
    if (n > 0) return false;
#endif
    if (n > 8) { basalt::log::error("basalt", "mtp_step_end: n out of range"); std::exit(1); }
    StepEndCond hs{};
    for (int k = 0; k < n; ++k) hs.h[k] = h[k];
    mtp_step_end_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(ids, row_dev, tok_dst, out, j, probs, out_p, hs, n, th_bits,
                                                            hist);
    check("mtp_step_end");
    return true;
}

namespace {
__global__ void window_ids_kernel(int32_t* steps, int window, int32_t* ids, long long stride) {
    const int q = blockIdx.y;
    int32_t* st = steps + q * 4;
    const int n_kv = st[1];
    const int start = n_kv > window ? n_kv - window : 0;
    const int width = n_kv - start;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < width; j += gridDim.x * blockDim.x)
        ids[q * stride + j] = start + j;
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0) st[3] = width;
}
}  // namespace

void window_ids(int32_t* steps, int n, int window, int32_t* ids, int64_t ids_stride, void* stream) {
    window_ids_kernel<<<dim3(8, (unsigned) n), 256, 0, (cudaStream_t) stream>>>(steps, window, ids, (long long) ids_stride);
    check("window_ids");
}

void dense_steps(const int32_t* cells, int n, int32_t* steps, void* stream) {
    dense_steps_kernel<<<1, 64, 0, (cudaStream_t) stream>>>(cells, n, steps);
    check("dense_steps");
}

void gdn_conv_l2_multi(const float* history, const float* qkv, const float* conv_w, float* h, int channels,
                       int qk_heads, float eps, int n_tok, void* stream, int t_begin) {
    if (!history || !qkv || !conv_w || !h || channels % S != 0 || n_tok < 1 || n_tok > kVerifyMaxT) {
        basalt::log::error("basalt", "gdn_conv_l2_multi: invalid arguments");
        std::exit(1);
    }
    launch_pdl(gdn_conv_l2_multi_kernel, dim3((unsigned) (channels / S), (unsigned) n_tok), dim3(S), 0,
               (cudaStream_t) stream, history, qkv, conv_w, h, channels, qk_heads, eps, t_begin);
    check("gdn_conv_l2_multi");
}

void gdn_conv_l2_seg(const SegRow* rows, size_t hist_off, const float* qkv, const float* conv_w, float* h, int channels,
                     int qk_heads, float eps, int n_rows, void* stream) {
    if (!rows || !qkv || !conv_w || !h || channels % S != 0 || n_rows < 1) {
        basalt::log::error("basalt", "gdn_conv_l2_seg: invalid arguments");
        std::exit(1);
    }
    launch_pdl(gdn_conv_l2_seg_kernel, dim3((unsigned) (channels / S), (unsigned) n_rows), dim3(S), 0,
               (cudaStream_t) stream, rows, hist_off, qkv, conv_w, h, channels, qk_heads, eps);
    check("gdn_conv_l2_seg");
}

void gdn_step_norm_seg(const SegRow* rows, size_t state_off, const float* h, int conv_channels, const float* gate,
                       const float* beta, const float* z, const float* gamma, float eps, float* y, int h_k, int h_v,
                       int n_rows, void* stream) {
    if (!rows || !h || !gate || !beta || !z || !gamma || !y || h_k <= 0 || h_v % h_k || n_rows < 1) {
        basalt::log::error("basalt", "gdn_step_norm_seg: invalid arguments");
        std::exit(1);
    }
    launch_pdl(gdn_step_norm_seg_kernel, dim3((unsigned) h_v, (unsigned) n_rows), dim3(S, RG), 0, (cudaStream_t) stream,
               rows, state_off, h, conv_channels, gate, beta, z, gamma, eps, y, h_k, h_v);
    check("gdn_step_norm_seg");
}

void gdn_commit_layers_seg(const GdnCommitSeg& a, void* stream, int blocks) {
    if (a.n_layers <= 0 || a.n_rows <= 0) return;
    const int conv_blocks = (a.C + 32 * RG - 1) / (32 * RG);
    gdn_commit_layers_seg_kernel<<<dim3((unsigned) (blocks > 0 ? blocks : 1), (unsigned) a.n_rows), dim3(32, RG), 0,
                                   (cudaStream_t) stream>>>(a, conv_blocks);
    check("gdn_commit_layers_seg");
}

void gdn_commit_begin(CommitCtl* ctl, const int32_t* epoch, uint32_t* begun, void* stream) {
    gdn_commit_begin_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(ctl, epoch, begun);
    check("gdn_commit_begin");
}

void gdn_commit_layers_dyn(const GdnCommitLayers& a, CommitCtl* ctl, void* stream, int blocks) {
    if (a.n_layers <= 0) return;
    const int conv_blocks = (a.C + 32 * RG - 1) / (32 * RG);
    gdn_commit_layers_dyn_kernel<<<(unsigned) (blocks > 0 ? blocks : 1), dim3(32, RG), 0, (cudaStream_t) stream>>>(
        a, conv_blocks, ctl);
    check("gdn_commit_layers_dyn");
}

void gdn_commit_finish(const GdnCommitLayers& a, CommitCtl* ctl, void* stream, int blocks) {
    if (a.n_layers <= 0) return;
    const int conv_blocks = (a.C + 32 * RG - 1) / (32 * RG);
    gdn_commit_finish_kernel<<<(unsigned) (blocks > 0 ? blocks : 1), dim3(32, RG), 0, (cudaStream_t) stream>>>(
        a, conv_blocks, ctl);
    check("gdn_commit_finish");
}

void gdn_commit_layers(const GdnCommitLayers& a, void* stream, int blocks) {
    if (a.n_layers <= 0) return;
    const int conv_blocks = (a.C + 32 * RG - 1) / (32 * RG);
    gdn_commit_layers_pipe_kernel<<<(unsigned) (blocks > 0 ? blocks : 1), dim3(32, RG), 0, (cudaStream_t) stream>>>(
        a, conv_blocks);
    check("gdn_commit_layers");
}

void gdn_commit_lazy(const GdnCommitLayers& a, CommitCtl* ctl, int must_layers, const uint32_t* stop,
                     uint32_t stop_value, void* stream, int blocks, const GdnLazyPf* pf) {
    if (a.n_layers <= 0) return;
    const int conv_blocks = (a.C + 32 * RG - 1) / (32 * RG);
    gdn_commit_lazy_kernel<<<(unsigned) (blocks > 0 ? blocks : 1), dim3(32, RG), 0, (cudaStream_t) stream>>>(
        a, conv_blocks, ctl, must_layers, stop, stop_value, pf != nullptr ? *pf : GdnLazyPf{});
    check("gdn_commit_lazy");
}

bool gdn_commit_lazy_none(CommitCtl* ctl) {
    const uint32_t none = 0x40000000u;
    return cudaMemcpy(ctl->w + (size_t) kCtlWords * 10, &none, 4, cudaMemcpyHostToDevice) == cudaSuccess;
}

void gdn_conv_commit(float* history, const float* qkv, int channels, const int32_t* n_keep, void* stream) {
    gdn_conv_commit_kernel<<<(unsigned) ((channels + 255) / 256), 256, 0, (cudaStream_t) stream>>>(history, qkv,
                                                                                                 channels, n_keep);
    check("gdn_conv_commit");
}

void gdn_ab_multi(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, int n_tok, void* stream) {
    if (n_embd % 8 != 0 || n_tok < 1 || n_tok > kVerifyMaxT) {
        basalt::log::error("basalt", "gdn_ab_multi: invalid arguments");
        std::exit(1);
    }
    const unsigned blocks = (unsigned) ((2 * h_v + 1) / 2);
    if (n_embd == 32 * 8 * 10) {   // rows of 2560 take the unrolled kernel (the same bits)
        if (n_tok <= 4)
            launch_pdl(gdn_ab_multi_unrolled_kernel<4, 10>, dim3(blocks), dim3(64), 0, (cudaStream_t) stream,
                       x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok);
        else if (n_tok <= 6)
            launch_pdl(gdn_ab_multi_unrolled_kernel<6, 10>, dim3(blocks), dim3(64), 0, (cudaStream_t) stream,
                       x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok);
        else
            launch_pdl(gdn_ab_multi_unrolled_kernel<kVerifyMaxT, 10>, dim3(blocks), dim3(64), 0, (cudaStream_t) stream,
                       x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok);
    } else if (n_tok <= 4) {
        launch_pdl(gdn_ab_multi_kernel<4>, dim3(blocks), dim3(64), 0, (cudaStream_t) stream,
                   x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok);
    } else {
        launch_pdl(gdn_ab_multi_kernel<kVerifyMaxT>, dim3(blocks), dim3(64), 0, (cudaStream_t) stream,
                   x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok);
    }
    check("gdn_ab_multi");
}

void gdn_step_norm_multi(float* state, const float* h, int conv_channels, const float* gate, const float* beta,
                         const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int n_tok,
                         const int32_t* n_keep, void* stream, int t_out_begin) {
    if (!state || !h || !gate || !beta || !z || !gamma || !y || h_k <= 0 || h_v % h_k || n_tok < 1 ||
        n_tok > kVerifyMaxT) {
        basalt::log::error("basalt", "gdn_step_norm_multi: invalid arguments");
        std::exit(1);
    }
    if (n_keep != nullptr && t_out_begin >= n_tok) {
        gdn_step_commit_kernel<<<dim3((unsigned) h_v, 4u), dim3(32, RG), 0, (cudaStream_t) stream>>>(
            state, h, conv_channels, gate, beta, h_k, h_v, n_keep);
    } else {
        launch_pdl(gdn_step_norm_multi_kernel, dim3((unsigned) h_v), dim3(S, RG), 0, (cudaStream_t) stream,
                   state, h, conv_channels, gate, beta, z, gamma, eps, y, h_k, h_v, n_tok, n_keep, t_out_begin);
    }
    check("gdn_step_norm_multi");
}

namespace {
__global__ void wait_flag_ge_kernel(const volatile uint32_t* flag, uint32_t value) {
    while (*flag < value) spin_pause();
    __threadfence_system();
}
}  // namespace

namespace {
// one thread per window entry: up to kVerifyMaxT tokens x 10 routed experts (80), so 128 (a smaller table drops the last entries of a long window)
constexpr int kResidentPlanMax = 128;
static_assert(kVerifyMaxT * 10 <= kResidentPlanMax, "resident_plan: one thread per entry");
__global__ void __launch_bounds__(kResidentPlanMax) resident_plan_kernel(const int32_t* __restrict__ ids, int n, int k, const int32_t* __restrict__ res,
                                     int n_expert, const uint8_t* cache_base, const unsigned long long* slot_off,
                                     long long blob, int32_t* __restrict__ pl, long long capx, uint32_t* skip,
                                     uint32_t ring, volatile uint32_t* plan_err) {
    __shared__ int32_t s_ids[kResidentPlanMax];
    __shared__ unsigned long long s_ptr[kResidentPlanMax];
    __shared__ int32_t s_first[kResidentPlanMax];
    __shared__ int32_t s_cnt[kResidentPlanMax];
    __shared__ int32_t s_gstart[kResidentPlanMax];
    __shared__ int s_bad;
    const int tid = threadIdx.x;
    if (tid == 0) s_bad = 0;
    __syncthreads();

    int32_t eid = -1;
    if (tid < n) {
        eid = ids[tid];
        s_ids[tid] = eid;
        const int32_t slot = (eid >= 0 && eid < n_expert) ? res[eid] : -1;
        if (slot < 0) {
            atomicOr(&s_bad, 1);
        } else {
            s_ptr[tid] = (unsigned long long) (cache_base + (slot_off ? (size_t) slot_off[slot] : (size_t) slot * (size_t) blob));
        }
    }
    __syncthreads();
    if (s_bad) {
        if (tid == 0 && skip != nullptr) *skip = 0;
        if (tid == 0 && skip == nullptr) {   // no host plan to fall back on: an empty plan, and the host is told
            pl[0] = 0; pl[1] = 0; pl[2] = 0;
            if (plan_err != nullptr) { *plan_err = 1; __threadfence_system(); }
        }
        return;
    }

    int first_j = tid;
    int rank_in_group = 0;
    int count_same = 0;
    if (tid < n) {
        for (int j = 0; j < n; ++j) {
            if (s_ids[j] == eid) {
                if (j < first_j) first_j = j;
                if (j < tid) ++rank_in_group;
                ++count_same;
            }
        }
        const bool is_first = (first_j == tid);
        s_first[tid] = is_first ? 1 : 0;
        s_cnt[tid] = is_first ? count_same : 0;
    }
    __syncthreads();

    int32_t* counts = pl;
    int32_t* start = pl + 4;
    int32_t* dst = start + capx + 1;
    int32_t* tok = dst + capx;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    unsigned long long* ptr = (unsigned long long*) (pl + ptr_off);
    int32_t* start2 = pl + ptr_off + 4 * capx;

    if (tid < n && first_j == tid) {
        int grp_idx = 0;
        int ent_start = 0;
        for (int j = 0; j < tid; ++j) {
            grp_idx += s_first[j];
            ent_start += s_cnt[j];
        }
        ptr[grp_idx] = s_ptr[tid];
        start[grp_idx] = ent_start;
        s_gstart[tid] = ent_start;
    }
    __syncthreads();

    if (tid < n) {
        const int out_idx = s_gstart[first_j] + rank_in_group;
        dst[out_idx] = tid;
        tok[out_idx] = tid / k;
    }
    if (tid == 0) {
        int groups = 0;
        for (int j = 0; j < n; ++j) groups += s_first[j];
        start[groups] = n;
        start2[0] = n;
        counts[0] = groups;
        counts[1] = n;
        counts[2] = 0;
        if (skip != nullptr) {
            __threadfence();
            *skip = ring;
        }
    }
}
__global__ void wait_flag_ge_or_kernel(const volatile uint32_t* flag, uint32_t value, const volatile uint32_t* skip) {
    if (*skip == value) return;
    while (*flag < value) spin_pause();
    __threadfence_system();
}
__global__ void copy_i32_unless_kernel(int32_t* __restrict__ dst, const volatile int32_t* src, int n,
                                       const uint32_t* skip, uint32_t value) {
    if (*skip == value) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
}
__global__ void copy_or_zero_kernel(float4* __restrict__ dst, const volatile float4* src, long long n4,
                                    const uint32_t* skip, uint32_t value) {
    const bool zero = *skip == value;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (long long) gridDim.x * blockDim.x)
        dst[i] = zero ? make_float4(0.f, 0.f, 0.f, 0.f) : const_cast<const float4*>(src)[i];
}
}  // namespace

void resident_plan(const int32_t* ids, int n_entries, int k, const int32_t* res_layer, int n_expert,
                   const uint8_t* cache_base, const unsigned long long* slot_off, long long blob, int32_t* plan,
                   long long capx, uint32_t* skip, uint32_t ring, void* stream, uint32_t* plan_err) {
    resident_plan_kernel<<<1, kResidentPlanMax, 0, (cudaStream_t) stream>>>(ids, n_entries, k, res_layer, n_expert, cache_base, slot_off,
                                                              blob, plan, capx, skip, ring, plan_err);
    check("resident_plan");
}
void wait_flag_ge_or(const uint32_t* flag, uint32_t value, const uint32_t* skip, void* stream) {
    wait_flag_ge_or_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(flag, value, skip);
    check("wait_flag_ge_or");
}
void copy_i32_from_mapped_unless(int32_t* dst, const int32_t* src, long long n, const uint32_t* skip, uint32_t value,
                                 void* stream) {
    if (n <= 0) return;
    copy_i32_unless_kernel<<<1, 128, 0, (cudaStream_t) stream>>>(dst, (const volatile int32_t*) src, (int) n, skip, value);
    check("copy_i32_from_mapped_unless");
}
void copy_or_zero_from_mapped(float* dst, const float* src, long long n, const uint32_t* skip, uint32_t value,
                              void* stream) {
    if (n <= 0) return;
    const long long n4 = n / 4;
    const int blocks = (int) ((n4 + 255) / 256 < 64 ? (n4 + 255) / 256 : 64);
    copy_or_zero_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>((float4*) dst, (const volatile float4*) src, n4, skip,
                                                                    value);
    check("copy_or_zero_from_mapped");
}

void wait_flag_ge(const uint32_t* flag, uint32_t value, void* stream) {
    wait_flag_ge_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(flag, value);
    check("wait_flag_ge");
}

void embedding_gather_dev(const uint8_t* codes, const float* scales, const float* offsets, const int32_t* tokens,
                          int n_tok, int64_t n, int code_bits, int code_bias, int group_elems, uint64_t row_codes,
                          uint64_t row_groups, float* out, void* stream) {
    embedding_gather_dev_kernel<<<dim3((unsigned) ((n + 255) / 256), (unsigned) n_tok), 256, 0,
                                  (cudaStream_t) stream>>>(codes, scales, offsets, tokens, n, code_bits, code_bias,
                                                           group_elems, row_codes, row_groups, out);
    check("embedding_gather_dev");
}

void broadcast_streams(const float* x, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    broadcast_streams_kernel<<<dim3((unsigned) ((n_embd * hc + 255) / 256), (unsigned) n_tok), 256, 0,
                               (cudaStream_t) stream>>>(x, R, n_embd, hc);
    check("broadcast_streams");
}

void copy_indexed(float* dst, const float* src, int64_t stride, const int32_t* index, int64_t n, void* stream) {
    const unsigned blocks = (unsigned) ((n + 255) / 256 < 64 ? (n + 255) / 256 : 64);
    copy_indexed_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>(dst, src, stride, index, n);
    check("copy_indexed");
}

void ple_hist_gather(float* work, float* const* slots, const SegRow* rows, int64_t n, int n_rows, void* stream) {
    const unsigned blocks = (unsigned) ((n + 255) / 256 < 64 ? (n + 255) / 256 : 64);
    ple_hist_seg_kernel<<<dim3(blocks, (unsigned) n_rows), 256, 0, (cudaStream_t) stream>>>(work, slots, rows, n, 0);
    check("ple_hist_gather");
}

void ple_hist_scatter(float* const* slots, const float* work, const SegRow* rows, int64_t n, int n_rows, void* stream) {
    const unsigned blocks = (unsigned) ((n + 255) / 256 < 64 ? (n + 255) / 256 : 64);
    ple_hist_seg_kernel<<<dim3(blocks, (unsigned) n_rows), 256, 0, (cudaStream_t) stream>>>((float*) work, slots, rows, n, 1);
    check("ple_hist_scatter");
}

// a GPU timestamp (ns, %globaltimer) into buf[i] - the verify window's stage profiler
namespace { __global__ void gpu_stamp_kernel(unsigned long long* buf, int i) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    buf[i] = t;
} }
void gpu_stamp(unsigned long long* buf, int i, void* stream) {
    gpu_stamp_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(buf, i);
}

}  // namespace basalt::kernels
