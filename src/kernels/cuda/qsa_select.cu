// src/kernels/cuda/qsa_select.cu - see include/basalt/kernels/qsa_select.hpp.
#include <cstdlib>
#include <cstring>
#include "basalt/kernels/qsa_select.hpp"
#include "basalt/kernels/qsa_seg.hpp"
#include "basalt/kernels/pdl.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "basalt/log.hpp"

namespace basalt::kernels {
namespace {

constexpr int IDX_DIM = 128, IDX_HEADS = 4, R = 4;
constexpr int SCORE_WARPS = 8;
constexpr int TOPK_T = 256;

__device__ __forceinline__ uint32_t order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    const uint32_t b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__global__ void __launch_bounds__(SCORE_WARPS * 32) block_scores_kernel(const float* __restrict__ pooled,
                                                                        const float* __restrict__ dead,
                                                                        const float* __restrict__ q_idx,
                                                                        const int32_t* __restrict__ steps,
                                                                        int64_t max_blocks, float* __restrict__ out) {
    const int64_t qi = blockIdx.y;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid];
    const int64_t b = (int64_t) blockIdx.x * SCORE_WARPS + (threadIdx.x >> 5);
    if (b > n_bid || b >= max_blocks) return;
    const int lane = threadIdx.x & 31;
    const float* key = (b == n_bid) ? dead : pooled + b * IDX_DIM;
    const float4 k4 = *reinterpret_cast<const float4*>(key + lane * 4);
    const float* q = q_idx + qi * IDX_HEADS * IDX_DIM + lane * 4;
    float score = 0.0f;
#pragma unroll
    for (int h = 0; h < IDX_HEADS; ++h) {
        const float4 q4 = *reinterpret_cast<const float4*>(q + h * IDX_DIM);
        float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
        score += d > 0.0f ? d : 0.0f;
    }
    if (lane == 0) {
        if (b == n_bid && n_kv % R != 0) score += 1e9f;
        out[qi * max_blocks + b] = score;
    }
}

__global__ void __launch_bounds__(TOPK_T) block_topk_kernel(const float* __restrict__ scores,
                                                            const int32_t* __restrict__ steps, int64_t max_blocks,
                                                            int64_t cap, int32_t* __restrict__ ids) {
    __shared__ int hist[256];
    __shared__ int s_a[TOPK_T], s_b[TOPK_T];
    __shared__ int s_digit, s_above;
    const int64_t qi = blockIdx.x;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = threadIdx.x;
    if (n_kv <= width) {                               // everything is selected: the identity, ascending
        for (int64_t j = t; j < n_kv; j += TOPK_T) out[j] = (int32_t) j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;                      // blocks 0..n_bid, the last possibly empty
    const int64_t per = (nb + TOPK_T - 1) / TOPK_T;
    const int64_t b0 = (int64_t) t * per, b1 = (b0 + per < nb) ? b0 + per : nb;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    // ---- radix select: the largest key thr with (cells with key >= thr) >= width, 8 bits at a time
    uint32_t prefix = 0;
    int above = 0;                                     // cells strictly above the digits fixed so far
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = t; i < 256; i += TOPK_T) hist[i] = 0;
        __syncthreads();
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int64_t b = b0; b < b1; ++b) {
            const int w = weight(b);
            if (w == 0) continue;
            const uint32_t k = order_key(sc[b]);
            if ((k & hi_mask) == (prefix & hi_mask)) atomicAdd(&hist[(k >> shift) & 255], w);
        }
        __syncthreads();
        if (t == 0) {
            int cum = above, d = 255;
            for (; d > 0; --d) {
                if (cum + hist[d] >= width) break;
                cum += hist[d];
            }
            s_digit = d;
            s_above = cum;
        }
        __syncthreads();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
        __syncthreads();
    }
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;          // cells equal to thr that fit, lowest index first
    // ---- per-thread counts of cells above and at the threshold, then their exclusive prefixes
    int gt = 0, eq = 0;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
    s_a[t] = gt;
    s_b[t] = eq;
    __syncthreads();
    if (t == 0) {
        int ag = 0, ae = 0;
        for (int i = 0; i < TOPK_T; ++i) {
            const int g = s_a[i], e = s_b[i];
            s_a[i] = ag; s_b[i] = ae;
            ag += g; ae += e;
        }
    }
    __syncthreads();
    const int64_t eq_before = s_b[t];
    int64_t my_eq = eq_budget - eq_before;
    if (my_eq < 0) my_eq = 0;
    if (my_eq > eq) my_eq = eq;
    const int sel = gt + (int) my_eq;
    __syncthreads();
    s_a[t] = sel;
    __syncthreads();
    if (t == 0) {
        int a = 0;
        for (int i = 0; i < TOPK_T; ++i) { const int c = s_a[i]; s_a[i] = a; a += c; }
    }
    __syncthreads();
    int64_t wpos = s_a[t];
    int64_t eq_left = my_eq;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) {
            for (int c = 0; c < w; ++c) out[wpos++] = (int32_t) (b * R + c);
        } else if (k == thr) {
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left) out[wpos++] = (int32_t) (b * R + c);
        }
    }
}


// ---- QSA select on tensor cores the block scores of many queries are one GEMM,
// rows (query, indexer head) x columns (blocks), K = 128, with relu per head summed. 3xTF32 (each operand split
// into a TF32 hi and lo part, hi*hi + hi*lo + lo*hi) keeps FP32-level accuracy; the summation order differs from
// the warp kernel, so a score can move in its last bits and a near-tie can select differently (not bitwise).
// Blocks < n_bid only; the tail block n_bid (the `dead` key, +1e9) is scored by the warp kernel's own code.
constexpr int TC_QT = 16;                 // queries per CTA (one m16 tile per indexer head)
constexpr int TC_NB = 32;                 // blocks per tile (4 warps x n8)
constexpr int TC_ITER = 4;                // tiles per CTA (the query tile is loaded once)
constexpr int TC_QS = IDX_HEADS * IDX_DIM + 4;   // query row stride in floats
constexpr int TC_KS = IDX_DIM + 4;               // key row stride

__device__ __forceinline__ uint32_t tf32_hi(float x) {
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
}
__device__ __forceinline__ void mma_tf32(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__global__ void __launch_bounds__(128) block_scores_tc_kernel(const float* __restrict__ pooled,
                                                              const float* __restrict__ q_idx,
                                                              const int32_t* __restrict__ steps, int64_t nq,
                                                              int64_t max_blocks, int64_t reach,
                                                              float* __restrict__ out) {
    extern __shared__ __align__(16) float sm[];
    float* sQ = sm;                                   // [TC_QT][TC_QS]
    float* sK = sm + TC_QT * TC_QS;                   // [TC_NB][TC_KS]
    __shared__ int s_nbid[TC_QT];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, gid = lane >> 2, tig = lane & 3;
    const int64_t q0 = (int64_t) blockIdx.y * TC_QT;
    for (int i = t; i < TC_QT * IDX_HEADS * IDX_DIM / 4; i += 128) {
        const int r = i / (IDX_HEADS * IDX_DIM / 4), c = i % (IDX_HEADS * IDX_DIM / 4);
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (q0 + r < nq) v = reinterpret_cast<const float4*>(q_idx + (q0 + r) * IDX_HEADS * IDX_DIM)[c];
        *reinterpret_cast<float4*>(sQ + r * TC_QS + c * 4) = v;
    }
    if (t < TC_QT) s_nbid[t] = q0 + t < nq ? steps[(q0 + t) * kStepCount + kStepNBid] : 0;
    int lo_nbid = 0x7fffffff, hi_nbid = 0;
    __syncthreads();
    for (int i = 0; i < TC_QT; ++i) {
        if (q0 + i >= nq) break;
        lo_nbid = min(lo_nbid, s_nbid[i]);
        hi_nbid = max(hi_nbid, s_nbid[i]);
    }
    for (int it = 0; it < TC_ITER; ++it) {
        const int64_t b0 = ((int64_t) blockIdx.x * TC_ITER + it) * TC_NB;
        if (b0 >= reach || b0 >= hi_nbid) break;      // blocks >= every query's n_bid: nothing to score
        __syncthreads();                              // the previous tile's reads are done
        for (int i = t; i < TC_NB * IDX_DIM / 4; i += 128) {
            const int r = i / (IDX_DIM / 4), c = i % (IDX_DIM / 4);
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (b0 + r < hi_nbid) v = reinterpret_cast<const float4*>(pooled + (b0 + r) * IDX_DIM)[c];
            *reinterpret_cast<float4*>(sK + r * TC_KS + c * 4) = v;
        }
        __syncthreads();
        float acc[IDX_HEADS][4];
#pragma unroll
        for (int h = 0; h < IDX_HEADS; ++h) acc[h][0] = acc[h][1] = acc[h][2] = acc[h][3] = 0.f;
        const float* kr = sK + (warp * 8 + gid) * TC_KS;
#pragma unroll 4
        for (int k0 = 0; k0 < IDX_DIM; k0 += 8) {
            const float kx0 = kr[k0 + tig], kx1 = kr[k0 + tig + 4];
            uint32_t bh[2], bl[2];
            bh[0] = tf32_hi(kx0);
            bh[1] = tf32_hi(kx1);
            bl[0] = tf32_hi(kx0 - __uint_as_float(bh[0]));
            bl[1] = tf32_hi(kx1 - __uint_as_float(bh[1]));
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const float* qa = sQ + h * IDX_DIM + k0 + tig;
                const float x0 = qa[gid * TC_QS], x1 = qa[(gid + 8) * TC_QS];
                const float x2 = qa[gid * TC_QS + 4], x3 = qa[(gid + 8) * TC_QS + 4];
                uint32_t ah[4], al[4];
                ah[0] = tf32_hi(x0); ah[1] = tf32_hi(x1); ah[2] = tf32_hi(x2); ah[3] = tf32_hi(x3);
                al[0] = tf32_hi(x0 - __uint_as_float(ah[0]));
                al[1] = tf32_hi(x1 - __uint_as_float(ah[1]));
                al[2] = tf32_hi(x2 - __uint_as_float(ah[2]));
                al[3] = tf32_hi(x3 - __uint_as_float(ah[3]));
                mma_tf32(acc[h], al, bh);
                mma_tf32(acc[h], ah, bl);
                mma_tf32(acc[h], ah, bh);
            }
        }
        // relu per head, heads added in order (as the warp kernel), written where the block completed for the query
        const int64_t bc = b0 + warp * 8 + 2 * tig;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int qr = gid + half * 8;
            const int64_t qi = q0 + qr;
            if (qi >= nq) continue;
            const int nb = s_nbid[qr];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int64_t b = bc + j;
                if (b >= nb || b >= max_blocks) continue;
                float score = 0.0f;
#pragma unroll
                for (int h = 0; h < IDX_HEADS; ++h) {
                    const float d = acc[h][half * 2 + j];
                    score += d > 0.0f ? d : 0.0f;
                }
                out[qi * max_blocks + b] = score;
            }
        }
    }
}


// the tail block n_bid of each query: exactly block_scores_kernel's arithmetic for that block
__global__ void __launch_bounds__(32) block_scores_tail_kernel(const float* __restrict__ dead,
                                                               const float* __restrict__ q_idx,
                                                               const int32_t* __restrict__ steps,
                                                               int64_t max_blocks, float* __restrict__ out) {
    const int64_t qi = blockIdx.x;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid];
    if (n_bid >= max_blocks) return;
    const int lane = threadIdx.x & 31;
    const float4 k4 = *reinterpret_cast<const float4*>(dead + lane * 4);
    const float* q = q_idx + qi * IDX_HEADS * IDX_DIM + lane * 4;
    float score = 0.0f;
#pragma unroll
    for (int h = 0; h < IDX_HEADS; ++h) {
        const float4 q4 = *reinterpret_cast<const float4*>(q + h * IDX_DIM);
        float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
        score += d > 0.0f ? d : 0.0f;
    }
    if (lane == 0) {
        if (n_kv % R != 0) score += 1e9f;
        out[qi * max_blocks + n_bid] = score;
    }
}

// ---- the same top-k with each query's keys read once: 1,024 threads hold up to TK_PER consecutive blocks' keys in
// registers (contexts up to 4 * 1024 * TK_PER cells), per-warp histograms, block-wide scans. The selection rule is
// block_topk_kernel's (radix threshold, ties to the lowest index, cells ascending): identical ids.
constexpr int TK_T = 1024;
constexpr int TK_PER = 33;
constexpr int TK_PER_MAX = TK_PER;

__device__ __forceinline__ int block_excl_scan(int v, int* s_warp, int& total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) s_warp[warp] = x;
    __syncthreads();
    if (warp == 0) {
        int w = s_warp[lane];
        int z = w;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, z, o);
            if (lane >= o) z += y;
        }
        s_warp[lane] = z - w;               // exclusive per warp
        if (lane == 31) s_warp[32] = z;     // total
    }
    __syncthreads();
    const int r = s_warp[warp] + x - v;
    total = s_warp[32];
    __syncthreads();
    return r;
}

template <int PER>
__global__ void __launch_bounds__(TK_T) block_topk_reg_kernel(const float* __restrict__ scores,
                                                              const int32_t* __restrict__ steps, int64_t max_blocks,
                                                              int64_t cap, int32_t* __restrict__ ids) {
    __shared__ int hist[TK_T / 32][256];
    __shared__ int s_warp[33];
    __shared__ int s_digit, s_above;
    const int64_t qi = blockIdx.x;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    if (n_kv <= width) {
        for (int64_t j = t; j < n_kv; j += TK_T) out[j] = (int32_t) j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;
    const int64_t per = (nb + TK_T - 1) / TK_T;       // <= PER (the caller checks)
    const int64_t b0 = (int64_t) t * per, b1 = (b0 + per < nb) ? b0 + per : nb;
    uint32_t key[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) key[j] = (b0 + j < b1) ? order_key(sc[b0 + j]) : 0u;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    uint32_t prefix = 0;
    int above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = lane; i < 256; i += 32) hist[warp][i] = 0;
        __syncwarp();
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int64_t b = b0 + j;
            if (b >= b1) break;
            const int w = weight(b);
            if (w == 0) continue;
            if ((key[j] & hi_mask) == (prefix & hi_mask)) atomicAdd(&hist[warp][(key[j] >> shift) & 255], w);
        }
        __syncthreads();
        if (t < 256) {                                // fold the warps' histograms into warp 0's
            int s = 0;
#pragma unroll
            for (int w2 = 0; w2 < TK_T / 32; ++w2) s += hist[w2][t];
            hist[0][t] = s;
        }
        __syncthreads();
        if (warp == 0) {
            // the largest digit d with above + (bins over d) + hist[d] >= width (0 if none), as a warp scan:
            // lane l holds bins 8l..8l+7
            int h[8], loc = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) { h[i] = hist[0][lane * 8 + i]; loc += h[i]; }
            int sfx = loc;                            // inclusive sum over lanes >= this one
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int y = __shfl_down_sync(0xffffffffu, sfx, o);
                if (lane + o < 32) sfx += y;
            }
            int cum = above + sfx - loc;              // above plus the bins over this lane's
            int d = -1, ca = 0;
#pragma unroll
            for (int i = 7; i >= 0; --i) {
                if (d < 0 && cum + h[i] >= width) { d = lane * 8 + i; ca = cum; }
                cum += h[i];
            }
            const unsigned hit = __ballot_sync(0xffffffffu, d >= 0);
            if (hit) {
                const int src = 31 - __clz(hit);      // the highest lane with a hit holds the largest digit
                d = __shfl_sync(0xffffffffu, d, src);
                ca = __shfl_sync(0xffffffffu, ca, src);
            } else {
                d = 0;
                ca = above + sfx - h[0];                  // read on lane 0: every bin over 0
            }
            if (lane == 0) { s_digit = d; s_above = ca; }
        }
        __syncthreads();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
        __syncthreads();
    }
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;
    int gt = 0, eq = 0;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t b = b0 + j;
        if (b >= b1) break;
        const int w = weight(b);
        if (w == 0) continue;
        if (key[j] > thr) gt += w;
        else if (key[j] == thr) eq += w;
    }
    int tot;
    const int eq_before = block_excl_scan(eq, s_warp, tot);
    int64_t my_eq = eq_budget - eq_before;
    if (my_eq < 0) my_eq = 0;
    if (my_eq > eq) my_eq = eq;
    const int sel = gt + (int) my_eq;
    int64_t wpos = block_excl_scan(sel, s_warp, tot);
    int64_t eq_left = my_eq;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t b = b0 + j;
        if (b >= b1) break;
        const int w = weight(b);
        if (w == 0) continue;
        if (key[j] > thr) {
            for (int c = 0; c < w; ++c) out[wpos++] = (int32_t) (b * R + c);
        } else if (key[j] == thr) {
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left) out[wpos++] = (int32_t) (b * R + c);
        }
    }
}

// ---- a query with more blocks than the register kernel holds (contexts past 4 * 1024 * TK_PER cells): the register
// kernel's threads, per-warp histograms and warp scans, with each key read from memory again on every pass.  A
// histogram has no order, so on the four radix passes thread t reads blocks t, t + 1024, ...: a warp reads 32
// neighbours at a time.  The cells are emitted ascending in the order (warp, row, lane): warp w holds the `per` rows
// of 32 consecutive blocks from block w * 32 * per.  block_topk_kernel's selection rule (radix threshold, ties to the
// lowest index): identical ids.
__global__ void __launch_bounds__(TK_T) block_topk_wide_kernel(const float* __restrict__ scores,
                                                               const int32_t* __restrict__ steps, int64_t max_blocks,
                                                               int64_t cap, int32_t* __restrict__ ids) {
    __shared__ int hist[TK_T / 32][256];
    __shared__ int s_gt[TK_T / 32], s_eq[TK_T / 32];
    __shared__ int s_digit, s_above;
    const int64_t qi = blockIdx.x;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    if (n_kv <= width) {
        for (int64_t j = t; j < n_kv; j += TK_T) out[j] = (int32_t) j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;
    const int64_t per = (nb + TK_T - 1) / TK_T;       // rows of 32 blocks per warp
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    uint32_t prefix = 0;
    int above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = lane; i < 256; i += 32) hist[warp][i] = 0;
        __syncwarp();
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int64_t b = t; b < nb; b += TK_T) {
            const int w = weight(b);
            if (w == 0) continue;
            const uint32_t k = order_key(sc[b]);
            if ((k & hi_mask) == (prefix & hi_mask)) atomicAdd(&hist[warp][(k >> shift) & 255], w);
        }
        __syncthreads();
        if (t < 256) {                                // fold the warps' histograms into warp 0's
            int sum = 0;
            for (int w2 = 0; w2 < TK_T / 32; ++w2) sum += hist[w2][t];
            hist[0][t] = sum;
        }
        __syncthreads();
        if (t == 0) {
            int cum = above, d = 255;
            for (; d > 0; --d) {
                if (cum + hist[0][d] >= width) break;
                cum += hist[0][d];
            }
            s_digit = d;
            s_above = cum;
        }
        __syncthreads();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
        __syncthreads();
    }
    const uint32_t thr = prefix;
    const int eq_budget = (int) (width - above);      // cells equal to thr that fit, lowest index first
    const int64_t w0 = (int64_t) warp * 32 * per;
    int gt = 0, eq = 0;
    for (int64_t b = w0 + lane; b < nb && b < w0 + 32 * per; b += 32) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        gt += __shfl_xor_sync(0xffffffffu, gt, o);
        eq += __shfl_xor_sync(0xffffffffu, eq, o);
    }
    if (lane == 0) { s_gt[warp] = gt; s_eq[warp] = eq; }
    __syncthreads();
    if (t == 0) {                                     // per warp: the selected cells and the tied cells before it
        int eb = 0, sb = 0;
        for (int w2 = 0; w2 < TK_T / 32; ++w2) {
            const int g2 = s_gt[w2], e2 = s_eq[w2];
            const int take = eq_budget - eb < 0 ? 0 : (eq_budget - eb > e2 ? e2 : eq_budget - eb);
            s_gt[w2] = sb;
            s_eq[w2] = eb;
            sb += g2 + take;
            eb += e2;
        }
    }
    __syncthreads();
    int run_sel = s_gt[warp], run_eq = s_eq[warp];
    for (int64_t r0 = w0; r0 < nb && r0 < w0 + 32 * per; r0 += 32) {
        const int64_t b = r0 + lane;
        const int w = b < nb ? weight(b) : 0;
        const uint32_t k = w ? order_key(sc[b]) : 0u;
        const int my_gt = (w && k > thr) ? w : 0, my_eq = (w && k == thr) ? w : 0;
        if (__ballot_sync(0xffffffffu, (my_gt | my_eq) != 0) == 0u) continue;   // a row without a selected block
        int pe = my_eq;                               // tied cells up to and with this lane
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, pe, o);
            if (lane >= o) pe += y;
        }
        const int left = eq_budget - (run_eq + pe - my_eq);
        const int take = left < 0 ? 0 : (left > my_eq ? my_eq : left);
        const int my_sel = my_gt + take;
        int ps = my_sel;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, ps, o);
            if (lane >= o) ps += y;
        }
        int32_t* dst = out + run_sel + ps - my_sel;
        for (int c = 0; c < my_sel; ++c) dst[c] = (int32_t) (b * R + c);
        run_eq += __shfl_sync(0xffffffffu, pe, 31);
        run_sel += __shfl_sync(0xffffffffu, ps, 31);
    }
}


// Most queries a decode window scores in one call.
constexpr int MQ = 8;

// Block scores over rows of different sequences: row qi reads its slot's pooled / dead. The queries and the per-row
// counts are staged once; per (block, row) the arithmetic and its order are block_scores_kernel's.
constexpr int MQS = kQsaSegMaxRows;
__global__ void __launch_bounds__(SCORE_WARPS * 32) block_scores_seg_kernel(QsaSegTable tab,
                                                                            const float* __restrict__ q_idx,
                                                                            const int32_t* __restrict__ steps, int nq,
                                                                            int64_t max_blocks, float* __restrict__ out,
                                                                            int skip_identity) {
    __shared__ __align__(16) float qs[MQS * IDX_HEADS * IDX_DIM];
    __shared__ int64_t s_nkv[MQS], s_nbid[MQS];
    __shared__ const float* s_pooled[MQS];
    __shared__ const float* s_dead[MQS];
    pdl_launch_dependents();
    pdl_wait();
    if (skip_identity) {
        bool all = true;
        for (int q = 0; q < nq; ++q) all = all && steps[q * kStepCount + kStepNKv] <= steps[q * kStepCount + kStepWidth];
        if (all) return;
    }
    if (threadIdx.x < nq) {
        const QsaSegSlot& sl = tab.slots[tab.rows[threadIdx.x].slot];
        s_nkv[threadIdx.x] = steps[threadIdx.x * kStepCount + kStepNKv];
        s_nbid[threadIdx.x] = steps[threadIdx.x * kStepCount + kStepNBid];
        s_pooled[threadIdx.x] = sl.pooled;
        s_dead[threadIdx.x] = sl.dead;
    }
    __syncthreads();
    int64_t top = 0;
    for (int q = 0; q < nq; ++q) top = s_nbid[q] > top ? s_nbid[q] : top;
    if ((int64_t) blockIdx.x * SCORE_WARPS > top) return;
    for (int i = threadIdx.x; i < nq * IDX_HEADS * IDX_DIM; i += blockDim.x) qs[i] = q_idx[i];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int64_t wstride = (int64_t) gridDim.x * SCORE_WARPS;
    for (int64_t b = (int64_t) blockIdx.x * SCORE_WARPS + (threadIdx.x >> 5); b <= top && b < max_blocks; b += wstride) {
        for (int qi = 0; qi < nq; ++qi) {
            const int64_t n_bid = s_nbid[qi];
            if (b > n_bid) continue;
            const float4 k4 = *reinterpret_cast<const float4*>((b == n_bid ? s_dead[qi] : s_pooled[qi] + b * IDX_DIM) + lane * 4);
            const float* q = qs + qi * IDX_HEADS * IDX_DIM + lane * 4;
            float score = 0.0f;
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const float4 q4 = *reinterpret_cast<const float4*>(q + h * IDX_DIM);
                float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
                score += d > 0.0f ? d : 0.0f;
            }
            if (lane == 0) {
                if (b == n_bid && s_nkv[qi] % R != 0) score += 1e9f;
                out[qi * max_blocks + b] = score;
            }
        }
    }
}

// The four heads' dot products summed across the warp with the xor butterfly's pairing (16, 8, 4, 2, 1), but transposed so
// that each exchange moves only the values the partner keeps: the sums are the butterfly's bit for bit (an add commutes).
// Lane 0 returns the relu-sum over the heads in head order.
__device__ __forceinline__ float heads_score(float a0, float a1, float a2, float a3, int lane) {
    const bool hi = (lane & 16) != 0, b = (lane & 8) != 0;
    const float r0 = __shfl_xor_sync(0xffffffffu, hi ? a0 : a2, 16);
    const float r1 = __shfl_xor_sync(0xffffffffu, hi ? a1 : a3, 16);
    const float e0 = (hi ? a2 : a0) + r0, e1 = (hi ? a3 : a1) + r1;
    const float r2 = __shfl_xor_sync(0xffffffffu, b ? e0 : e1, 8);
    float f = (b ? e1 : e0) + r2;
    f += __shfl_xor_sync(0xffffffffu, f, 4);
    f += __shfl_xor_sync(0xffffffffu, f, 2);
    f += __shfl_xor_sync(0xffffffffu, f, 1);
    const float r = f > 0.0f ? f : 0.0f;
    const float g1 = __shfl_sync(0xffffffffu, r, 8), g2 = __shfl_sync(0xffffffffu, r, 16),
                g3 = __shfl_sync(0xffffffffu, r, 24);
    return ((r + g1) + g2) + g3;
}

// Block scores with the queries in registers (a lane holds its four dims of every head and query), so a CTA stages
// nothing and needs no barrier, and the next block's key row is loaded before the current one is scored. Per (block,
// query) the sums are block_scores_kernel's bit for bit. NQ is the register capacity: CTA row y takes the call's queries
// [y NQ, y NQ + NQ), so a launch of gridDim.y rows scores up to gridDim.y NQ queries.
template <int NQ>
__global__ void __launch_bounds__(SCORE_WARPS * 32) block_scores_rg_kernel(const float* __restrict__ pooled,
                                                                           const float* __restrict__ dead,
                                                                           const float* __restrict__ q_idx,
                                                                           const int32_t* __restrict__ steps, int nq,
                                                                           int64_t max_blocks, float* __restrict__ out,
                                                                           int skip_identity) {
    pdl_launch_dependents();
    pdl_wait();
    if (skip_identity) {
        bool all = true;
        for (int q = 0; q < nq; ++q) all = all && steps[q * kStepCount + kStepNKv] <= steps[q * kStepCount + kStepWidth];
        if (all) return;
    }
    {   // this CTA row's queries
        const int q0 = (int) blockIdx.y * NQ;
        if (q0 >= nq) return;
        q_idx += (size_t) q0 * IDX_HEADS * IDX_DIM;
        steps += (size_t) q0 * kStepCount;
        out += (size_t) q0 * max_blocks;
        nq = nq - q0 < NQ ? nq - q0 : NQ;
    }
    int64_t top = 0;
    for (int q = 0; q < nq; ++q) {
        const int64_t nb = steps[q * kStepCount + kStepNBid];
        top = nb > top ? nb : top;
    }
    const int lane = threadIdx.x & 31;
    const int64_t wstride = (int64_t) gridDim.x * SCORE_WARPS;
    int64_t b = (int64_t) blockIdx.x * SCORE_WARPS + (threadIdx.x >> 5);
    const int64_t lim = top < max_blocks - 1 ? top : max_blocks - 1;
    if (b > lim) return;
    float4 kp = *reinterpret_cast<const float4*>(pooled + b * IDX_DIM + lane * 4);
    int64_t nbid[NQ], nkv[NQ];
    float4 qr[NQ][IDX_HEADS];
#pragma unroll
    for (int qi = 0; qi < NQ; ++qi) {
        const int qc = qi < nq ? qi : nq - 1;
        nbid[qi] = qi < nq ? steps[qc * kStepCount + kStepNBid] : -1;
        nkv[qi] = steps[qc * kStepCount + kStepNKv];
#pragma unroll
        for (int h = 0; h < IDX_HEADS; ++h)
            qr[qi][h] = *reinterpret_cast<const float4*>(q_idx + (qc * IDX_HEADS + h) * IDX_DIM + lane * 4);
    }
    const float4 kd = *reinterpret_cast<const float4*>(dead + lane * 4);
    for (; b <= lim; b += wstride) {
        const float4 cur = kp;
        if (b + wstride <= lim) kp = *reinterpret_cast<const float4*>(pooled + (b + wstride) * IDX_DIM + lane * 4);
#pragma unroll
        for (int qi = 0; qi < NQ; ++qi) {
            if (b > nbid[qi]) continue;
            const float4 k4 = (b == nbid[qi]) ? kd : cur;
            float dd[IDX_HEADS];
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const float4 q4 = qr[qi][h];
                dd[h] = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
            }
            float score = heads_score(dd[0], dd[1], dd[2], dd[3], lane);
            if (lane == 0) {
                if (b == nbid[qi] && nkv[qi] % R != 0) score += 1e9f;
                out[qi * max_blocks + b] = score;
            }
        }
    }
}

// ---- the decode top-k on a thread-block cluster (needs thread-block clusters).  One CTA per query (block_topk_reg_kernel, or
// block_topk_kernel above 33,792 blocks: a --max-context over ~135K) makes four radix passes and two scans over up to
// 65,538 blocks on ONE SM while the rest of the GPU idles - a decode window has 1-5 queries.  Here a cluster of CL_N
// CTAs shares a query: CTA r holds the keys of blocks [r * per, (r + 1) * per) in shared memory, builds the digit
// histogram of its keys, and the CTAs total the histograms by a reduce-scatter in distributed shared memory (below), so all
// of them take the same digit. The emit pass needs each CTA's cells above / at the threshold: sent the same way, then
// each CTA offsets its own by the ranks before it.
// The one-CTA kernels' digit search is serial too: one thread over 256 bins per pass.
//
// IDENTICAL IDS, by construction rather than by luck: the threshold thr (the width-th largest key, cells counted with
// their weights) and `above` (cells with a larger key) are pure functions of the multiset of (key, weight) - integer
// histograms, summed in any order - and the digit rule is block_topk_kernel's, written as a scan (the largest digit d
// whose cells at or above it reach `need`; 0 when none). The cells emitted are then fixed: every cell of a block with
// key > thr, and the first eq_budget = width - above cells at thr in ascending order (a block at thr may be cut),
// written in ascending cell order. Position of a thread's first cell = (cells above thr before it) + min(cells at thr
// before it, eq_budget): the telescoped sum of the reference's per-thread clamp. NaN keys are 0 (order_key) as there.
//
// Synchronisation: one barrier.cluster arrive at entry (after the CTA's mbarriers are initialised), waited before the first
// remote store. After that CTAs talk only through st.async stores that add their bytes to the receiver's mbarrier, one
// mbarrier per exchange and each used once, so no cluster barrier or fence is needed. A CTA may exit once it has waited
// for every exchange it receives; the stores it sent are complete by then at their receivers.
constexpr int CL_N = 8;        // CTAs per query (the portable cluster size)
constexpr int CL_T = 1024;     // threads per CTA
constexpr int CL_MAXQ = 16;    // larger calls (prefill sub-batches) fill the GPU with one CTA per query already
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define SEL_CLUSTER 1
#else
#define SEL_CLUSTER 0   // an older target's code is a trap; qsa_block_topk_cluster never launches it there
#endif

#if SEL_CLUSTER
__device__ __forceinline__ void cl_arrive() { asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void cl_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }
__device__ __forceinline__ unsigned cl_rank() {
    unsigned r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r));
    return r;
}
// exclusive block scan of a 64-bit value over CL_T threads; `total` = the CTA's sum
__device__ __forceinline__ unsigned long long cl_excl_scan64(unsigned long long v, unsigned long long* s_warp,
                                                             unsigned long long& total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    unsigned long long x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const unsigned long long y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) s_warp[warp] = x;
    __syncthreads();
    if (warp == 0) {
        const unsigned long long w = s_warp[lane];
        unsigned long long z = w;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const unsigned long long y = __shfl_up_sync(0xffffffffu, z, o);
            if (lane >= o) z += y;
        }
        s_warp[lane] = z - w;
        if (lane == 31) s_warp[32] = z;
    }
    __syncthreads();
    total = s_warp[32];
    return s_warp[warp] + x - v;
}
#endif

#if SEL_CLUSTER
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint32_t a, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(a), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_expect(uint32_t a, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(a), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t a) {   // phase 0 of a barrier used once
    asm volatile(
        "{\n .reg .pred P1;\n LAB_WAIT:\n mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], 0;\n @P1 bra DONE;\n bra LAB_WAIT;\n DONE:\n}\n" ::"r"(a)
        : "memory");
}
__device__ __forceinline__ uint32_t cl_map32(uint32_t a, unsigned rank) {   // a in CTA `rank`'s shared memory
    uint32_t o;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(o) : "r"(a), "r"(rank));
    return o;
}
// stores into another CTA's shared memory that add their bytes to that CTA's mbarrier
__device__ __forceinline__ void st_async4(uint32_t dst, int4 v, uint32_t dst_bar) {
    asm volatile("st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];\n" ::"r"(dst),
                 "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "r"(dst_bar)
                 : "memory");
}
__device__ __forceinline__ void st_async1(uint32_t dst, int v, uint32_t dst_bar) {
    asm volatile("st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];\n" ::"r"(dst), "r"(v), "r"(dst_bar)
                 : "memory");
}
__device__ __forceinline__ void st_async2(uint32_t dst, unsigned long long v, uint32_t dst_bar) {
    asm volatile("st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];\n" ::"r"(dst),
                 "r"((uint32_t) v), "r"((uint32_t) (v >> 32)), "r"(dst_bar)
                 : "memory");
}
#endif

// The top-k of block_topk_kernel on a thread-block cluster, the radix passes exchanged as a reduce-scatter. Each pass: a CTA sends bins
// [32 o, 32 o + 32) of its histogram to CTA o as 16-bit counts; CTA o totals them and sends its slice total to every CTA;
// all see which slice holds the threshold digit, whose owner finds the digit and sends it (with the cells above it) to
// every CTA. The stores land on the receiver's mbarrier, so there is no cluster barrier or release fence per pass. Needs
// a CTA's cell count under 65,536 (the host checks).
// grid (CL_N, nq), cluster (CL_N, 1, 1), CL_T threads, dynamic shared memory: ceil(max_blocks / CL_N) keys
__global__ void __launch_bounds__(CL_T) block_topk_cluster_kernel(const float* __restrict__ scores,
                                                                  const int32_t* __restrict__ steps, int64_t max_blocks,
                                                                  int64_t cap, int32_t* __restrict__ ids) {
#if SEL_CLUSTER
    extern __shared__ uint32_t keys[];                  // this CTA's blocks' keys
    __shared__ __align__(16) uint32_t hin[4][CL_N][16];   // per pass and sender: its counts for this CTA's 32 bins, 16 bits each
    __shared__ int stot[4][CL_N];                        // per pass: each CTA's slice total
    __shared__ __align__(8) int sdig[4][2];              // per pass: the threshold digit, the cells above it in this level
    __shared__ __align__(8) unsigned long long mbar[13];  // per pass: slices, totals, digit; then the cell counts
    __shared__ __align__(16) int hloc[2][256];          // this CTA's histogram
    __shared__ unsigned long long cnt[CL_N];            // per rank: its cells above thr (low 32 bits) and at it (high)
    __shared__ unsigned long long s_warp[33];
    __shared__ uint32_t s_prefix;
    __shared__ int s_above;
    const unsigned rank = cl_rank();
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    if (t == 0) {
        for (int i = 0; i < 13; ++i) mbar_init(smem_u32(&mbar[i]), 1);
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
        for (int i = 0; i < 4; ++i) {
            mbar_expect(smem_u32(&mbar[3 * i]), CL_N * 64);
            mbar_expect(smem_u32(&mbar[3 * i + 1]), CL_N * 4);
            mbar_expect(smem_u32(&mbar[3 * i + 2]), 8);
        }
        mbar_expect(smem_u32(&mbar[12]), CL_N * 8);
    }
    cl_arrive();                                        // phase 0: this CTA runs and its barriers exist
    pdl_launch_dependents();
    pdl_wait();
    const int64_t qi = blockIdx.y;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    if (n_kv <= width) {                                // the identity, split over the cluster; no remote access
        for (int64_t j = (int64_t) rank * CL_T + t; j < n_kv; j += (int64_t) CL_N * CL_T) out[j] = (int32_t) j;
        return;
    }
    // the tail block n_bid weighs its cells (0..3); with none it is not a candidate (the reference skips w == 0)
    const int wtail = (int) (n_kv - n_bid * R);
    const int64_t nbe = n_bid + (wtail > 0 ? 1 : 0);
    const int64_t per = (nbe + CL_N - 1) / CL_N;       // <= ceil(max_blocks / CL_N), the keys the host sized
    const int64_t lo = (int64_t) rank * per < nbe ? (int64_t) rank * per : nbe;
    const int64_t hi = lo + per < nbe ? lo + per : nbe;
    const int n = (int) (hi - lo);
    const int tail_i = (wtail > 0 && n_bid >= lo && n_bid < hi) ? (int) (n_bid - lo) : -1;
    const float* sc = scores + qi * max_blocks + lo;
    for (int i = t; i < n; i += CL_T) keys[i] = order_key(sc[i]);
    for (int i = t; i < 2 * 256; i += CL_T) (&hloc[0][0])[i] = 0;
    __syncthreads();
    uint32_t prefix = 0;
    int above = 0;                                      // cells strictly above the digits fixed so far
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        const uint32_t hmask = pass == 0 ? 0u : (0xffffffffu << (shift + 8));
        int* hl = hloc[pass & 1];
        // warp-aggregated: the scores share their top bits, so plain atomics would queue on a few bins
        for (int i0 = 0; i0 < n; i0 += CL_T) {
            const int i = i0 + t;
            int bin = -1;
            if (i < n) {
                const uint32_t k = keys[i];
                if ((k & hmask) == prefix) bin = (int) ((k >> shift) & 255);
            }
            const unsigned same = __match_any_sync(0xffffffffu, bin);
            if (bin >= 0 && lane == __ffs(same) - 1) atomicAdd(&hl[bin], __popc(same) * R);
            if (bin >= 0 && i == tail_i) atomicAdd(&hl[bin], wtail - R);
        }
        __syncthreads();
        if (pass == 0) cl_wait();                       // phase 0 done: every CTA's shared memory is there
        for (int i = t; i < 256; i += CL_T) hloc[(pass + 1) & 1][i] = 0;
        if (warp == 0) {
            // reduce-scatter: this CTA's 32 bins [32 o, 32 o + 32) go to CTA o as 16-bit counts, 4 int4 stores each
            {
                const int dst = lane >> 2, c = lane & 3;
                const int* h = hl + dst * 32 + c * 8;
                int4 v;
                v.x = (int) ((uint32_t) h[0] | ((uint32_t) h[1] << 16));
                v.y = (int) ((uint32_t) h[2] | ((uint32_t) h[3] << 16));
                v.z = (int) ((uint32_t) h[4] | ((uint32_t) h[5] << 16));
                v.w = (int) ((uint32_t) h[6] | ((uint32_t) h[7] << 16));
                st_async4(cl_map32(smem_u32(reinterpret_cast<int4*>(&hin[pass][rank][0]) + c), (unsigned) dst), v,
                          cl_map32(smem_u32(&mbar[3 * pass]), (unsigned) dst));
            }
            mbar_wait(smem_u32(&mbar[3 * pass]));
            // this CTA owns bins 32 rank .. 32 rank + 31: lane j totals bin 32 rank + j over the senders
            int tj = 0;
#pragma unroll
            for (int r = 0; r < CL_N; ++r) tj += (int) ((hin[pass][r][lane >> 1] >> ((lane & 1) * 16)) & 0xffffu);
            int stotal = tj;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) stotal += __shfl_xor_sync(0xffffffffu, stotal, o);
            if (lane < CL_N)
                st_async1(cl_map32(smem_u32(&stot[pass][rank]), (unsigned) lane), stotal,
                          cl_map32(smem_u32(&mbar[3 * pass + 1]), (unsigned) lane));
            mbar_wait(smem_u32(&mbar[3 * pass + 1]));
            // the slice holding the need-th cell from the top (slice 7 is the top bins)
            const int need = (int) width - above;
            const int sv = lane < CL_N ? stot[pass][CL_N - 1 - lane] : 0;
            int incl = sv;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int y = __shfl_up_sync(0xffffffffu, incl, o);
                if (lane >= o) incl += y;
            }
            const int excl = incl - sv;
            const unsigned hit = __ballot_sync(0xffffffffu, lane < CL_N && excl < need && incl >= need);
            // no hit cannot happen (the cells total n_kv > width); the reference would then take digit 0
            const int who = hit ? __ffs(hit) - 1 : CL_N - 1;
            const int owner = CL_N - 1 - who;
            const int excl_s = __shfl_sync(0xffffffffu, excl, who);
            if ((int) rank == owner) {
                // within the slice: bin j of the slice, from the top
                const int needc = need - excl_s;
                int x = tj;
#pragma unroll
                for (int o = 1; o < 32; o <<= 1) {
                    const int y = __shfl_down_sync(0xffffffffu, x, o);
                    if (lane + o < 32) x += y;
                }
                const int ex = x - tj;
                const unsigned h2 = __ballot_sync(0xffffffffu, ex < needc && x >= needc);
                const int w2 = h2 ? __ffs(h2) - 1 : 0;
                const int dig = 32 * owner + w2;
                const int add = excl_s + __shfl_sync(0xffffffffu, ex, w2);
                if (lane < CL_N)
                    st_async2(cl_map32(smem_u32(&sdig[pass][0]), (unsigned) lane),
                              (unsigned long long) (uint32_t) dig | ((unsigned long long) (uint32_t) add << 32),
                              cl_map32(smem_u32(&mbar[3 * pass + 2]), (unsigned) lane));
            }
            mbar_wait(smem_u32(&mbar[3 * pass + 2]));
            if (lane == 0) {
                s_prefix = prefix | ((uint32_t) sdig[pass][0] << shift);
                s_above = above + sdig[pass][1];
            }
        }
        __syncthreads();
        prefix = s_prefix;
        above = s_above;
    }
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;           // cells at thr that fit, lowest index first
    // each thread a contiguous run of the CTA's keys; an odd run length keeps the shared reads conflict-free
    int seg = (n + CL_T - 1) / CL_T;
    if ((seg & 1) == 0 && seg > 0) ++seg;
    const int s0 = t * seg < n ? t * seg : n, s1 = s0 + seg < n ? s0 + seg : n;
    uint32_t gt = 0, eq = 0;
    for (int i = s0; i < s1; ++i) {
        const uint32_t k = keys[i], w = i == tail_i ? (uint32_t) wtail : (uint32_t) R;
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
    unsigned long long cta;
    const unsigned long long before =
        cl_excl_scan64((unsigned long long) gt | ((unsigned long long) eq << 32), s_warp, cta);
    if (t < CL_N) st_async2(cl_map32(smem_u32(&cnt[rank]), (unsigned) t), cta, cl_map32(smem_u32(&mbar[12]), (unsigned) t));
    mbar_wait(smem_u32(&mbar[12]));
    int64_t gt_off = 0, eq_off = 0;
    for (unsigned r = 0; r < rank; ++r) {
        gt_off += (int64_t) (cnt[r] & 0xffffffffu);
        eq_off += (int64_t) (cnt[r] >> 32);
    }
    const int64_t eq_before = eq_off + (int64_t) (before >> 32);
    int64_t wpos = gt_off + (int64_t) (before & 0xffffffffu) + (eq_before < eq_budget ? eq_before : eq_budget);
    int64_t eq_left = eq_budget - eq_before;
    if (eq_left < 0) eq_left = 0;
    for (int i = s0; i < s1; ++i) {
        const uint32_t k = keys[i];
        const int w = i == tail_i ? wtail : R;
        const int64_t b = lo + i;
        if (k > thr) {
            for (int c = 0; c < w; ++c) out[wpos++] = (int32_t) (b * R + c);
        } else if (k == thr) {
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left) out[wpos++] = (int32_t) (b * R + c);
        }
    }
#else
    (void) scores; (void) steps; (void) max_blocks; (void) cap; (void) ids;
    __trap();
#endif
}
}  // namespace

void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                      int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t active_blocks,
                      bool skip_identity) {
    if (nq <= 0) return;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R || nq > 65535) {
        basalt::log::error("basalt", "qsa_block_scores: unsupported indexer geometry");
        std::exit(1);
    }
    // a block past a query's n_bid returns at once: the grid need only reach the batch's largest n_bid (C-1)
    if (nq <= MQ && active_blocks <= 0) {   // no active count: decode (captured or not) and prefill's pooled16
        static int grid_of[64] = {};   // two CTAs per multiprocessor
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); dev = 0; }
        if (grid_of[dev] == 0) {
            int sms = 0;
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
            grid_of[dev] = 2 * (sms > 0 ? sms : 64);
        }
        const dim3 g((unsigned) grid_of[dev]);
        if (nq <= 2)
            launch_pdl(block_scores_rg_kernel<2>, g, dim3(SCORE_WARPS * 32), 0, (cudaStream_t) stream, pooled, dead, q_idx,
                       steps, (int) nq, max_blocks, scores, skip_identity ? 1 : 0);
        else if (nq <= 4)
            launch_pdl(block_scores_rg_kernel<4>, g, dim3(SCORE_WARPS * 32), 0, (cudaStream_t) stream, pooled, dead, q_idx,
                       steps, (int) nq, max_blocks, scores, skip_identity ? 1 : 0);
        else {   // rows of four queries side by side, the grid's CTAs split between them; a later row's key reads
                 // mostly hit L2
            const unsigned rows = (unsigned) ((nq + 3) / 4);
            launch_pdl(block_scores_rg_kernel<4>, dim3(g.x / rows > 0 ? g.x / rows : 1u, rows), dim3(SCORE_WARPS * 32), 0,
                       (cudaStream_t) stream, pooled, dead, q_idx, steps, (int) nq, max_blocks, scores, skip_identity ? 1 : 0);
        }
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { basalt::log::error("basalt", "qsa_block_scores rg: %s", cudaGetErrorString(e)); std::exit(1); }
        return;
    }
    const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
    const dim3 grid((unsigned) ((reach + SCORE_WARPS - 1) / SCORE_WARPS), (unsigned) nq);
    block_scores_kernel<<<grid, SCORE_WARPS * 32, 0, (cudaStream_t) stream>>>(pooled, dead, q_idx, steps, max_blocks,
                                                                              scores);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { basalt::log::error("basalt", "qsa_block_scores: %s", cudaGetErrorString(e)); std::exit(1); }
}

void qsa_block_scores_seg(const QsaSegTable& t, const float* q_idx, const int32_t* steps, int64_t nq, int64_t max_blocks,
                          const QsaShapes& s, float* scores, void* stream, bool skip_identity) {
    if (nq <= 0) return;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R || nq > MQS || t.n_rows < nq) {
        basalt::log::error("basalt", "qsa_block_scores_seg: unsupported indexer geometry or row count");
        std::exit(1);
    }
    launch_pdl(block_scores_seg_kernel, dim3(256), dim3(SCORE_WARPS * 32), 0, (cudaStream_t) stream, t, q_idx, steps,
               (int) nq, max_blocks, scores, skip_identity ? 1 : 0);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { basalt::log::error("basalt", "qsa_block_scores_seg: %s", cudaGetErrorString(e)); std::exit(1); }
}

bool qsa_block_scores_tc(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                         int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t active_blocks) {
    if (nq <= 0) return true;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R || nq > 65535 * TC_QT) return false;
    {   // TF32 MMA needs a recent card; an older one keeps the warp kernel
        static int cc_major[64] = {};
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
        if (cc_major[dev] == 0) {
            int major = 0;
            if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) {
                cudaGetLastError();
                return false;
            }
            cc_major[dev] = major;
        }
        if (cc_major[dev] < 8) return false;
    }
    static std::atomic<bool> attr[64];   // the shared-memory opt-in is per device (a layer split runs it on several)
    int adev = 0;
    cudaGetDevice(&adev);
    const int bytes = (TC_QT * TC_QS + TC_NB * TC_KS) * (int) sizeof(float);
    if (!attr[adev].load(std::memory_order_acquire)) {
        if (cudaFuncSetAttribute(block_scores_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) !=
            cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        attr[adev].store(true, std::memory_order_release);
    }
    const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
    const int64_t per = (int64_t) TC_NB * TC_ITER;
    const dim3 grid((unsigned) ((reach + per - 1) / per), (unsigned) ((nq + TC_QT - 1) / TC_QT));
    block_scores_tc_kernel<<<grid, 128, bytes, (cudaStream_t) stream>>>(pooled, q_idx, steps, nq, max_blocks, reach,
                                                                         scores);
    block_scores_tail_kernel<<<(unsigned) nq, 32, 0, (cudaStream_t) stream>>>(dead, q_idx, steps, max_blocks, scores);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { basalt::log::error("basalt", "qsa_block_scores_tc: %s", cudaGetErrorString(e)); std::exit(1); }
    return true;
}

void qsa_block_topk_ref(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                        const QsaShapes& s, int32_t* ids, void* stream) {
    if (nq <= 0) return;
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s)) {
        basalt::log::error("basalt", "qsa_block_topk: unsupported geometry or cap");
        std::exit(1);
    }
    block_topk_kernel<<<(unsigned) nq, TOPK_T, 0, (cudaStream_t) stream>>>(scores, steps, max_blocks, cap, ids);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { basalt::log::error("basalt", "qsa_block_topk: %s", cudaGetErrorString(e)); std::exit(1); }
}

namespace {
bool topk_cluster_launch(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                         const QsaShapes& s, int32_t* ids, void* stream) {
    if (nq <= 0) return true;
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s) || nq > 65535 || max_blocks <= 0) return false;
    const int64_t per_cta = (max_blocks + CL_N - 1) / CL_N;
    if (per_cta * R + R >= 65536) return false;   // the kernel's 16-bit counts: the one-CTA kernels take it
    const size_t smem = (size_t) per_cta * sizeof(uint32_t);
    const auto kern = block_topk_cluster_kernel;
    // Per device (a layer split runs on several): 1 the cluster kernel runs here, 2 it does not. It needs thread-block clusters (the
    // card's) AND code built for it: a build with only older code JIT-compiles their PTX, whose
    // copy of this kernel is a trap - the function's PTX version says which. `opt`: the dynamic shared memory opted in.
    static int ok[64] = {};
    static size_t opt[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (ok[dev] == 0) {
        int major = 0;
        cudaFuncAttributes fa{};
        const bool code = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                          major >= 9 &&
                          cudaFuncGetAttributes(&fa, kern) == cudaSuccess && fa.ptxVersion >= 90 &&
                          fa.binaryVersion >= 90;
        cudaGetLastError();
        ok[dev] = code ? 1 : 2;
    }
    if (ok[dev] != 1) return false;
    if (smem > opt[dev]) {   // a larger capacity: opt in, and check that a cluster of CL_N such CTAs can be resident
        int clusters = 0;
        cudaLaunchConfig_t q{};
        cudaLaunchAttribute qa[1];
        qa[0].id = cudaLaunchAttributeClusterDimension;
        qa[0].val.clusterDim.x = CL_N;
        qa[0].val.clusterDim.y = 1;
        qa[0].val.clusterDim.z = 1;
        q.gridDim = dim3(CL_N, 1, 1);
        q.blockDim = dim3(CL_T, 1, 1);
        q.dynamicSmemBytes = smem;
        q.attrs = qa;
        q.numAttrs = 1;
        if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem) !=
                cudaSuccess ||
            cudaOccupancyMaxActiveClusters(&clusters, kern, &q) != cudaSuccess || clusters < 1) {
            cudaGetLastError();
            return false;    // this capacity takes the one-CTA kernels; a smaller one may still fit
        }
        opt[dev] = smem;
    }
    cudaLaunchConfig_t cfg{};
    cudaLaunchAttribute at[2];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = CL_N;
    at[0].val.clusterDim.y = 1;
    at[0].val.clusterDim.z = 1;
    cfg.gridDim = dim3(CL_N, (unsigned) nq, 1);
    cfg.blockDim = dim3(CL_T, 1, 1);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = (cudaStream_t) stream;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    if (pdl_launch_wanted()) {   // a dependent of the scores kernel
        at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        at[1].val.programmaticStreamSerializationAllowed = 1;
        cfg.numAttrs = 2;
    }
    const cudaError_t e = cudaLaunchKernelEx(&cfg, kern, scores, steps, max_blocks, cap, ids);
    if (e != cudaSuccess) { basalt::log::error("basalt", "qsa_block_topk cluster: %s", cudaGetErrorString(e)); std::exit(1); }
    return true;
}
}  // namespace

bool qsa_block_topk_cluster(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                            const QsaShapes& s, int32_t* ids, void* stream) {
    return topk_cluster_launch(scores, steps, nq, max_blocks, cap, s, ids, stream);
}

void qsa_block_topk(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                    const QsaShapes& s, int32_t* ids, void* stream, int64_t active_blocks) {
    // keys in registers when every query's blocks fit (contexts up to ~135K cells), else the same threads reading
    // them from memory; the same ids
    if (nq <= 0) return;
    // a cluster of CL_N CTAs per query for the calls of a few queries (decode windows); the same ids
    if (nq <= CL_MAXQ && qsa_block_topk_cluster(scores, steps, nq, max_blocks, cap, s, ids, stream)) return;
    const int64_t reach = max_blocks;
    const int64_t fit = (int64_t) TK_T * TK_PER;
    constexpr int64_t kWideMinBlocks = 4608;   // a prompt batch below ~18K cells is faster on the reference kernel
    if (reach > fit && active_blocks > 0 && active_blocks < kWideMinBlocks) {
        qsa_block_topk_ref(scores, steps, nq, max_blocks, cap, s, ids, stream);
        return;
    }
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s)) {
        basalt::log::error("basalt", "qsa_block_topk: unsupported geometry or cap");
        std::exit(1);
    }
    if (reach > fit)   // past the register kernel's reach
        block_topk_wide_kernel<<<(unsigned) nq, TK_T, 0, (cudaStream_t) stream>>>(scores, steps, max_blocks, cap, ids);
    else
    if (reach <= (int64_t) TK_T * TK_PER)
        block_topk_reg_kernel<TK_PER><<<(unsigned) nq, TK_T, 0, (cudaStream_t) stream>>>(scores, steps, max_blocks, cap, ids);
    else
        block_topk_reg_kernel<TK_PER_MAX><<<(unsigned) nq, TK_T, 0, (cudaStream_t) stream>>>(scores, steps, max_blocks, cap, ids);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { basalt::log::error("basalt", "qsa_block_topk: %s", cudaGetErrorString(e)); std::exit(1); }
}

}  // namespace basalt::kernels
