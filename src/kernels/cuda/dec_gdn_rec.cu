// src/kernels/cuda/dec_gdn_rec.cu - see include/basalt/kernels/dec_gdn.hpp: the recurrence stage.
//
// Built with the flags of verify_kernels.cu, whose kernels it fuses: gdn_conv_l2_multi (conv1d + SiLU + q/k L2
// norm), gdn_ab_multi (alpha/beta gates), gdn_step_norm_multi (the recurrence and the gated RMS norm).  The q8_1 of
// the output is native_quantize_q8_1's (a fast-math file), its two divisions written as the instruction it uses.
//
// A cluster of four blocks per value head; block `rank` owns state columns [32 rank, 32 rank + 32), thread
// (warp rg, lane c) the rows [32 rg, 32 rg + 32) of column 32 rank + c, as gdn_step_commit_kernel splits them.
// Each column's sums are the reference's: four 32-row fmaf chains added in row-group order.  The head's RMS norm
// adds the four warps' butterflies in rank order through distributed shared memory; rank 0 computes alpha, rank 1
// beta, and the others read them the same way.  The state and every constant are loaded before the grid
// dependency wait (the window never writes the state).
#include "basalt/kernels/dec_gdn.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/verify_kernels.hpp"
#include "basalt/kernels/q8_1_finite.hpp"

#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include "basalt/log.hpp"

namespace cg = cooperative_groups;

namespace basalt::kernels {
namespace {

constexpr int S = 128;   // state size
constexpr int RG = 4;    // row groups
constexpr int RPG = S / RG;
constexpr int NJ = 10;   // uint4 of bf16 per lane in a 2560-wide alpha/beta row

struct Q81 {
    half2 ds;
    int8_t qs[32];
};

__device__ __forceinline__ float div_approx_ftz(float a, float b) {
    float r;
    asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

// q8_1_quant with native_mmvq.cu's division
__device__ __forceinline__ int8_t q8_1_quant_fm(const float xi, const float d, const float amax) {
    if (amax == 0.0f) return 0;
    const float q = roundf(div_approx_ftz(xi, d));
    return (int8_t) (q > 127.0f ? 127.0f : q < -127.0f ? -127.0f : q);
}

// conv1d over [hist(3) | x_0 .. x_t] of one channel (xv[t] = its raw projection of token t), then SiLU:
// gdn_conv_l2_multi_kernel's expression
template<int MAXT>
__device__ __forceinline__ float conv_silu(const float (&hist)[3], const float (&w)[4], const float (&xv)[MAXT], int t) {
    float win[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = t + j;
        win[j] = src < 3 ? hist[src] : xv[src - 3];
    }
    const float v0 = win[0], v1 = win[1], v2 = win[2], x = xv[t];
    float sum = v0 * w[0] + v1 * w[1] + v2 * w[2] + x * w[3];
    return sum / (1.0f + __expf(-sum));
}

// per-phase %globaltimer of each block's thread 0 (a bench hook; null in the engine)
__device__ __forceinline__ void stamp(const DecRecArgs& a, int i) {
    if (a.stamps == nullptr || threadIdx.x != 0) return;
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    a.stamps[blockIdx.x * 10 + i] = t;
}

struct RecSmem {
    float qn[kDecMaxT][S], kn[kDecMaxT][S];
    float v[kDecMaxT][32];
    float oc[kDecMaxT][32];
    float red_kv[RG][32], red_o[RG][32];
    float part[2][kDecMaxT][RG];   // the q/k L2 partials
    float g[kDecMaxT], b[kDecMaxT];
    float wsum[kDecMaxT][4];       // every rank's sum of squares per token, pushed by the ranks
    uint4 wab[2][32 * NJ];         // the head's alpha and beta rows
};

// alpha (is_beta false) or beta of `head` for token t: gdn_ab_multi_unrolled_kernel's sum, one warp.  The row's
// weights are in shared memory; the token's 80 activations per lane are loaded at once.
__device__ __forceinline__ float ab_one(const DecRecArgs& a, const uint4* __restrict__ w_sm, bool is_beta, int t, int lane,
                                        float dt, float sa) {
    const float4* xt = reinterpret_cast<const float4*>(a.x + (size_t) t * a.n_embd);
    float4 xa[NJ], xb[NJ];
#pragma unroll
    for (int q = 0; q < NJ; ++q) {
        const int j = lane + 32 * q;
        xa[q] = xt[2 * j];
        xb[q] = xt[2 * j + 1];
    }
    float v = 0.0f;
#pragma unroll
    for (int q = 0; q < NJ; ++q) {
        const uint4 w = w_sm[lane + 32 * q];
        v = fmaf(__uint_as_float(w.x << 16), xa[q].x, v); v = fmaf(__uint_as_float(w.x & 0xffff0000u), xa[q].y, v);
        v = fmaf(__uint_as_float(w.y << 16), xa[q].z, v); v = fmaf(__uint_as_float(w.y & 0xffff0000u), xa[q].w, v);
        v = fmaf(__uint_as_float(w.z << 16), xb[q].x, v); v = fmaf(__uint_as_float(w.z & 0xffff0000u), xb[q].y, v);
        v = fmaf(__uint_as_float(w.w << 16), xb[q].z, v); v = fmaf(__uint_as_float(w.w & 0xffff0000u), xb[q].w, v);
    }
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (is_beta) return 1.0f / (1.0f + __expf(-v));
    const float u = v + dt;
    const float sp = u > 20.0f ? u : log1pf(__expf(u));
    return sp * sa;
}

// The gated RMS norm of one lane's value (ss: the head's sum of squares) and the q8_1 of its 32-value block.
__device__ __forceinline__ void norm_q8(const DecRecArgs& a, int t, int head, int rank, int lane, float oc, float ss,
                                        float gam, float z) {
    const int value_dim = S * a.h_v, col = rank * 32 + lane;
    const float scale = rsqrtf(ss / (float) S + a.eps);
    const float yv = oc * scale * gam * (1.0f / (1.0f + __expf(-z)));
    if (a.y) a.y[(size_t) t * value_dim + head * S + col] = yv;
    float amax = fabsf(yv), sum = yv;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o, 32));
        sum += __shfl_xor_sync(0xffffffffu, sum, o, 32);
    }
    const float d = q8_1_finite(div_approx_ftz(amax, 127.0f));
    Q81* blk = reinterpret_cast<Q81*>(a.yq) + (size_t) t * (value_dim / 32) + head * (S / 32) + rank;
    blk->qs[lane] = q8_1_quant_fm(yv, d, amax);
    if (lane == 0) blk->ds = q8_1_ds(d, sum);
}

// sum of squares over the warp's 32 values, the recurrence's butterfly
__device__ __forceinline__ float sq32(float v) {
    float q = v * v;
    for (int o = 16; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
    return q;
}

// One block of a head's cluster.  `head` = value head, `rank` = its column group.  SPLIT: no cluster and no z; the
// head's raw output goes to a.o for dec_gdn_rec_norm.
template<int MAXT, bool SPLIT>
__device__ __forceinline__ void rec_block(const DecRecArgs& a, int T, int head, int rank, RecSmem& sm,
                                          cg::cluster_group* cluster) {
    const int rg = threadIdx.x / 32, lane = threadIdx.x % 32, tid = threadIdx.x;
    stamp(a, 0);
    const int col = rank * 32 + lane;
    const int qh = head % a.h_k;
    const int qk = S * a.h_k;
    const int value_dim = S * a.h_v;
    const int C = a.C;
    // ---- constants (the state too: the window does not write it)
    float s[RPG];
    const float* base = a.state + ((size_t) (rg * RPG) * a.h_v + head) * S + col;
    const size_t row_stride = (size_t) a.h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    const float* hist = a.state + (size_t) S * a.h_v * S;
    const int cq = qh * S + tid, ck = qk + qh * S + tid, cv = 2 * qk + head * S + col;
    float wq[4], wk[4], wv[4], hq[3], hk[3], hv[3];
#pragma unroll
    for (int j = 0; j < 4; ++j) { wq[j] = a.conv_w[cq * 4 + j]; wk[j] = a.conv_w[ck * 4 + j]; wv[j] = a.conv_w[cv * 4 + j]; }
#pragma unroll
    for (int j = 0; j < 3; ++j) { hq[j] = hist[cq * 3 + j]; hk[j] = hist[ck * 3 + j]; hv[j] = hist[cv * 3 + j]; }
    for (int i = tid; i < 2 * 32 * NJ; i += S) {   // the alpha/beta rows into shared memory
        const int r = i / (32 * NJ), k = i % (32 * NJ);
        const uint4* src = reinterpret_cast<const uint4*>((r ? a.w_beta : a.w_alpha) + (size_t) head * a.n_embd) + k;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"((uint32_t) __cvta_generic_to_shared(&sm.wab[r][k])),
                     "l"(src) : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    const float dt = a.dt[head], sa = a.ssm_a[head], gam = a.gamma[col];
    stamp(a, 1);
    if (a.early_trigger) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");   // the out projection may stage its weights
    // this block has started: the barrier the ranks wait on before writing into each other's shared memory
    if (!SPLIT) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    // alpha and beta read only the hc output and constants, both complete before this kernel can launch (its
    // predecessor passed its own wait on the hc read), so they are computed while the projections still run
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
    for (int job = rg; job < 2 * T; job += RG) {   // (gate, token) jobs over the four warps
        const int t = job >> 1;
        const bool is_beta = job & 1;
        const float v = ab_one(a, sm.wab[is_beta], is_beta, t, lane, dt, sa);
        if (lane == 0) (is_beta ? sm.b : sm.g)[t] = v;
    }
    pdl_wait();
    stamp(a, 2);

    // ---- every load of the previous kernels' outputs, issued together
    float xq[MAXT], xk[MAXT], xv[MAXT], zz[MAXT];
#pragma unroll
    for (int t = 0; t < MAXT; ++t) {
        xq[t] = t < T ? a.qkv[(size_t) t * C + cq] : 0.0f;
        xk[t] = t < T ? a.qkv[(size_t) t * C + ck] : 0.0f;
        xv[t] = t < T && rg == 1 ? a.qkv[(size_t) t * C + cv] : 0.0f;
        zz[t] = t < T && rg == 0 && !SPLIT ? a.z[(size_t) t * value_dim + head * S + col] : 0.0f;
    }
    stamp(a, 3);

    // ---- conv + SiLU of the head's q and k channels (L2-normalized) and of the block's v columns
    float yq[MAXT], yk[MAXT];
#pragma unroll
    for (int t = 0; t < MAXT; ++t) {
        if (t >= T) break;
        yq[t] = conv_silu<MAXT>(hq, wq, xq, t);
        yk[t] = conv_silu<MAXT>(hk, wk, xk, t);
        float sq = yq[t] * yq[t], sk = yk[t] * yk[t];
        for (int o = 16; o > 0; o >>= 1) {
            sq += __shfl_xor_sync(0xffffffffu, sq, o);
            sk += __shfl_xor_sync(0xffffffffu, sk, o);
        }
        if (lane == 0) { sm.part[0][t][rg] = sq; sm.part[1][t][rg] = sk; }
        if (rg == 1) {
            const float yv = conv_silu<MAXT>(hv, wv, xv, t);
            sm.v[t][lane] = yv;
            a.h[(size_t) t * C + cv] = yv;
        }
    }
    __syncthreads();
    const bool write_qk = rank == 0 && head < a.h_k;
#pragma unroll
    for (int t = 0; t < MAXT; ++t) {
        if (t >= T) break;
        const float ssq = sm.part[0][t][0] + sm.part[0][t][1] + sm.part[0][t][2] + sm.part[0][t][3];
        const float ssk = sm.part[1][t][0] + sm.part[1][t][1] + sm.part[1][t][2] + sm.part[1][t][3];
        yq[t] *= rsqrtf(ssq + a.eps);
        yk[t] *= rsqrtf(ssk + a.eps);
        sm.qn[t][tid] = yq[t];
        sm.kn[t][tid] = yk[t];
        if (write_qk) {
            a.h[(size_t) t * C + cq] = yq[t];
            a.h[(size_t) t * C + ck] = yk[t];
        }
    }
    if (tid < T) {   // the gates, for the commit
        if (rank == 0) a.gate[(size_t) tid * a.h_v + head] = sm.g[tid];
        if (rank == 1) a.beta[(size_t) tid * a.h_v + head] = sm.b[tid];
    }
    stamp(a, 4);
    __syncthreads();
    stamp(a, 5);

    // ---- the recurrence: gdn_step_norm_multi_kernel's sums
    float wsum_t = 0.0f;   // lane t of warp 0: this rank's sum of squares of token t
    for (int t = 0; t < T; ++t) {
        const float g = __expf(sm.g[t]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sm.kn[t][rg * RPG + r], kv);
        sm.red_kv[rg][lane] = kv;
        __syncthreads();
        const float kv_col = sm.red_kv[0][lane] + sm.red_kv[1][lane] + sm.red_kv[2][lane] + sm.red_kv[3][lane];
        const float delta = (sm.v[t][lane] - g * kv_col) * sm.b[t];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sm.kn[t][rg * RPG + r] * delta);
            o = fmaf(s[r], sm.qn[t][rg * RPG + r], o);
        }
        sm.red_o[rg][lane] = o;
        __syncthreads();
        if (rg == 0) {
            const float oc = (sm.red_o[0][lane] + sm.red_o[1][lane] + sm.red_o[2][lane] + sm.red_o[3][lane]) *
                             rsqrtf((float) S);
            if (SPLIT) {
                a.o[(size_t) t * value_dim + head * S + col] = oc;
            } else {
                const float sq_part = sq32(oc);
                sm.oc[t][lane] = oc;
                if (lane == t) wsum_t = sq_part;
            }
        }
    }
    stamp(a, 6);
    if (SPLIT) return;
    // every rank's sums of squares, pushed into every rank's shared memory (once every rank has started) before one
    // cluster barrier
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
    if (rg == 0) {
        const int t = lane / 4, r = lane % 4;
        const float v = __shfl_sync(0xffffffffu, wsum_t, t);
        if (t < T) *cluster->map_shared_rank(&sm.wsum[t][rank], r) = v;
    }
    cluster->sync();
    stamp(a, 7);

    // ---- the gated RMS norm and the q8_1 of the output: one warp, one q8_1 block per token
    if (rg == 0) {
#pragma unroll
        for (int t = 0; t < MAXT; ++t) {
            if (t >= T) break;
            const float ss = sm.wsum[t][0] + sm.wsum[t][1] + sm.wsum[t][2] + sm.wsum[t][3];
            norm_q8(a, t, head, rank, lane, sm.oc[t][lane], ss, gam, zz[t]);
        }
    }
    stamp(a, 8);
    stamp(a, 9);
}

template<int MAXT>
__global__ void __cluster_dims__(4, 1, 1) __launch_bounds__(S) rec_kernel(const DecRecArgs a, int T) {
    __shared__ RecSmem sm;
    cg::cluster_group cl = cg::this_cluster();
    rec_block<MAXT, false>(a, T, blockIdx.x / 4, blockIdx.x % 4, sm, &cl);
}

template<int MAXT>
__global__ void __launch_bounds__(S) rec_split_kernel(const DecRecArgs a, int T) {
    __shared__ RecSmem sm;
    rec_block<MAXT, true>(a, T, blockIdx.x / 4, blockIdx.x % 4, sm, nullptr);
}

// rec_kernel over a segmented window: cluster row y is a window row, of which only a segment's first works, on the
// segment's rows and state
template<int MAXT>
__global__ void __cluster_dims__(4, 1, 1) __launch_bounds__(S) rec_seg_kernel(const DecRecArgs a0, const SegRow* rows,
                                                                              size_t state_off) {
    const SegRow r = rows[blockIdx.y];
    if (r.j != 0) return;
    __shared__ RecSmem sm;
    cg::cluster_group cl = cg::this_cluster();
    DecRecArgs a = a0;
    const size_t vd = (size_t) S * a.h_v, r0 = (size_t) r.row0;
    a.state = r.gdn + state_off;
    a.qkv += r0 * a.C; a.x += r0 * a.n_embd; a.z += r0 * vd; a.h += r0 * a.C;
    a.gate += r0 * a.h_v; a.beta += r0 * a.h_v;
    if (a.y) a.y += r0 * vd;
    a.yq = static_cast<char*>(a.yq) + r0 * (vd / 32) * sizeof(Q81);
    rec_block<MAXT, false>(a, r.n, blockIdx.x / 4, blockIdx.x % 4, sm, &cl);
}

// one warp per (token, value head): the head's four 32-value sums of squares in rank order, then norm_q8 per rank
__global__ void __launch_bounds__(128) rec_norm_kernel(const DecRecArgs a, int T) {
    if (a.early_trigger) pdl_launch_dependents();   // the out projection may stage its weights
    pdl_wait();
    const int w = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (w >= T * a.h_v) return;
    const int t = w / a.h_v, head = w % a.h_v;
    const size_t base = (size_t) t * S * a.h_v + head * S + lane;
    float oc[4], q[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) oc[r] = a.o[base + 32 * r];
#pragma unroll
    for (int r = 0; r < 4; ++r) q[r] = sq32(oc[r]);
    const float ss = q[0] + q[1] + q[2] + q[3];
#pragma unroll
    for (int r = 0; r < 4; ++r)
        norm_q8(a, t, head, r, lane, oc[r], ss, a.gamma[32 * r + lane], a.z[base + 32 * r]);
}

}  // namespace

#ifndef FZ_DEV_ONLY   // host side: another translation unit (the mid chain) takes only the device code
void dec_gdn_rec(const DecRecArgs& a, int n_tok, void* stream) {
    if (n_tok < 1 || n_tok > kDecMaxT || a.n_embd != 32 * 8 * NJ || a.h_k <= 0 || a.h_v % a.h_k != 0) {
        basalt::log::error("basalt", "dec_gdn_rec: invalid arguments");
        std::exit(1);
    }
    const dim3 grid((unsigned) (a.h_v * 4)), block(S);
    const cudaStream_t s = (cudaStream_t) stream;
    if (n_tok <= 2) launch_pdl(rec_kernel<2>, grid, block, 0, s, a, n_tok);
    else if (n_tok <= 4) launch_pdl(rec_kernel<4>, grid, block, 0, s, a, n_tok);
    else launch_pdl(rec_kernel<kDecMaxT>, grid, block, 0, s, a, n_tok);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { basalt::log::error("basalt", "dec_gdn_rec: %s", cudaGetErrorString(e)); std::exit(1); }
}

void dec_gdn_rec_seg(const DecRecArgs& a, const SegRow* rows, size_t state_off, int n_rows, int max_n, void* stream) {
    if (n_rows < 1 || max_n < 1 || max_n > kDecMaxT || !rows || a.n_embd != 32 * 8 * NJ || a.h_k <= 0 ||
        a.h_v % a.h_k != 0) {
        basalt::log::error("basalt", "dec_gdn_rec_seg: invalid arguments");
        std::exit(1);
    }
    const dim3 grid((unsigned) (a.h_v * 4), (unsigned) n_rows), block(S);
    const cudaStream_t s = (cudaStream_t) stream;
    if (max_n <= 2) launch_pdl(rec_seg_kernel<2>, grid, block, 0, s, a, rows, state_off);
    else if (max_n <= 4) launch_pdl(rec_seg_kernel<4>, grid, block, 0, s, a, rows, state_off);
    else launch_pdl(rec_seg_kernel<kDecMaxT>, grid, block, 0, s, a, rows, state_off);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { basalt::log::error("basalt", "dec_gdn_rec_seg: %s", cudaGetErrorString(e)); std::exit(1); }
}

void dec_gdn_rec_split(const DecRecArgs& a, int n_tok, void* stream) {
    if (n_tok < 1 || n_tok > kDecMaxT || a.n_embd != 32 * 8 * NJ || a.h_k <= 0 || a.h_v % a.h_k != 0 || !a.o) {
        basalt::log::error("basalt", "dec_gdn_rec_split: invalid arguments");
        std::exit(1);
    }
    const dim3 grid((unsigned) (a.h_v * 4)), block(S);
    const cudaStream_t s = (cudaStream_t) stream;
    if (n_tok <= 2) launch_pdl(rec_split_kernel<2>, grid, block, 0, s, a, n_tok);
    else if (n_tok <= 4) launch_pdl(rec_split_kernel<4>, grid, block, 0, s, a, n_tok);
    else launch_pdl(rec_split_kernel<kDecMaxT>, grid, block, 0, s, a, n_tok);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { basalt::log::error("basalt", "dec_gdn_rec_split: %s", cudaGetErrorString(e)); std::exit(1); }
}

void dec_gdn_rec_norm(const DecRecArgs& a, int n_tok, void* stream) {
    const unsigned blocks = (unsigned) ((n_tok * a.h_v + 3) / 4);
    launch_pdl(rec_norm_kernel, dim3(blocks), dim3(128), 0, (cudaStream_t) stream, a, n_tok);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { basalt::log::error("basalt", "dec_gdn_rec_norm: %s", cudaGetErrorString(e)); std::exit(1); }
}

#endif
}  // namespace basalt::kernels
