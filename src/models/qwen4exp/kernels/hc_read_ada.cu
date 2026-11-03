// src/kernels/cuda/hc_read_ada.cu - see include/basalt/kernels/hc_read_ada.hpp.
//
// One persistent kernel, one CTA of 512 threads per SM, with hc8_gr.cu's arithmetic and reduction tree.  hc8_down's block
// is five warps, a warp's lanes take the scale groups g and g + 160, and its butterfly merges lanes by their high bits
// first, so the lanes with the same low bits form a subtree.  A subtree's columns are a class: a CTA serves one class,
// staging its part of w_norm and of every token's R and bo into shared memory once, and each lane of its warps takes a
// pair of rows over them.  So an SM reads a small slice of those arrays, however many rows it serves.  The subtrees' sums
// meet in global memory, where lanes of the other 256 threads add them in hc8_down's order.
//
// The norm runs on those 256 threads of the first CTAs, beside the first group's work.  R keeps its old value until
// every CTA is done reading it: each reader forms R' itself, and a CTA writes its columns' R' after the barrier.  Hand-offs
// are words tagged with the read's number, usable the moment they land, and one grid barrier before the up phase.  Up
// takes a contiguous run of columns per CTA, with its weights and the epilogues' inputs staged from the start: hc8_up's
// warp per output column, then the epilogues, a lane per (column, token), from shared memory.
//
// The kernel needs every CTA resident.  Built without fast math, like hc8_gr.cu.
#include "basalt/models/qwen4exp/kernels/hc_fused.hpp"
#include "basalt/models/qwen4exp/kernels/hc_read_ada.hpp"
#include "basalt/kernels/wait_abort.hpp"
#include "basalt/log.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>

namespace basalt::kernels {
namespace {

constexpr int N = 2560, HC = 4, D = N * HC, LR = 320, G8 = 32;
constexpr int MT = kFusedGrMaxT;
constexpr int kThreads = 512, kGroup = kThreads / 2, kGroupWarps = kGroup / 32;
constexpr int kDownWarps = 5;             // hc8_down's warps
constexpr int kSplit = 4;                 // subtrees of a warp's butterfly
constexpr int kLanes = 32 / kSplit;       // a subtree's lanes
constexpr int kClasses = kDownWarps * kSplit;
constexpr int kPlaneF4 = 2 * (G8 / 4) * kLanes;   // float4 of a class per array in shared memory: [2 runs][8 chunks][lanes]
constexpr int kUpChunks = LR / 4;         // a row of up in chunks of four values
constexpr int kUpPasses = (kUpChunks + 31) / 32;
constexpr int kUpQF4 = LR / 16;           // a row of up's codes in float4
constexpr int kUpSF4 = 2;                 // a row of up's scales in float4
constexpr int kUpColF4 = HC * (kUpQF4 + kUpSF4);
constexpr int kSmemMin = 50 * 1024;       // two CTAs do not fit one SM
constexpr int kSmemMax = 98 * 1024;
constexpr unsigned long long kWaitNs = 200000000ull;   // a wait's bound
// lo goes over the tile: (2 T + 1) planes hold T rows of it
static_assert(D / G8 == 2 * kDownWarps * 32 && (kSplit & (kSplit - 1)) == 0 && 3 * kPlaneF4 * 4 >= LR, "class geometry");

struct HcAdaArgs {
    const float* R[MT];
    float* R_out[MT];
    const float* bo[MT];
    const float* inj_prev[MT];
    float* lo[MT];
    float* rs[MT];
    float* inj_out[MT];
    float* mixed[MT];
    const float* w_norm;
    const int8_t* qd;
    const uint16_t* sd;
    const int8_t* qu;
    const uint16_t* su;
    const int8_t* qi;
    const uint16_t* si;
    int sd_row, su_row, si_row;
    int nrows;                 // down rows + inject rows
    bool apply;
    bool stage_up;             // up's weights in shared memory
    float eps;
    unsigned long long* red;   // [T * nrows][kClasses] the subtrees' sums, tagged
    unsigned long long* rs64;  // [T][HC] rs, tagged
    unsigned long long* ctr;   // grid barrier arrivals; a read starts at a multiple of the grid
    unsigned long long* word;
    unsigned long long* stamps;
};

// float4 chunks that hold a CTA's run of output columns, at most
__host__ __device__ __forceinline__ int up_chunks_max(int G) { return ((N + G - 1) / G + 3 + 3) >> 2; }

struct BarGroup {   // the barrier of one 256-thread group (ids 1, 2; __syncthreads is 0)
    int id;
    __device__ __forceinline__ void operator()() const { asm volatile("bar.sync %0, 256;" ::"r"(id) : "memory"); }
};

__device__ __forceinline__ float s8(uint32_t w, int j) { return (float) (int) (int8_t) (w >> (8 * j)); }
__device__ __forceinline__ float f16f(uint16_t h) { return __half2float(__ushort_as_half(h)); }
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const unsigned sa = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sa), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }
__device__ __forceinline__ uint32_t u4_at(const uint4& a, int i) { return i == 0 ? a.x : i == 1 ? a.y : i == 2 ? a.z : a.w; }
__device__ __forceinline__ float f4_at(const float4& a, int i) { return i == 0 ? a.x : i == 1 ? a.y : i == 2 ? a.z : a.w; }
__device__ __forceinline__ unsigned long long now() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ unsigned long long ld64_acquire(const unsigned long long* p) {
    unsigned long long v;
    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
// a load past the L1 with no fence: a tagged word needs no ordering
__device__ __forceinline__ unsigned long long ld64_fresh(const unsigned long long* p) {
    unsigned long long v;
    asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ unsigned long long tagged(unsigned tag, float v) {
    return ((unsigned long long) tag << 32) | __float_as_uint(v);
}
static __device__ __noinline__ unsigned long long tag_wait(const unsigned long long* a, unsigned tag, unsigned long long* word, int line) {
    const unsigned long long t0 = now();
    for (;;) {
        const unsigned long long x = ld64_fresh(a);
        if ((unsigned) (x >> 32) == tag) return x;
        const unsigned long long e = now() - t0;
        if (e > kWaitPollNs &&
            wait_give_up(word, e, kWaitNs, wait_abort_code_of(kWaitHcRead, 3, (unsigned) line, blockIdx.x)))
            return x;
    }
}
__device__ __forceinline__ float tag_load(const unsigned long long* a, unsigned tag, unsigned long long* word, int line) {
    unsigned long long x = ld64_fresh(a);
    if ((unsigned) (x >> 32) != tag) x = tag_wait(a, tag, word, line);
    return __uint_as_float((unsigned) x);
}

// hc8_down's warp `w`, subtree `sb`, of kSplit pairs of rows from base * kSplit, a lane per (pair, the subtree's lane).
// tile: the class, [plane][2 runs][8 chunks][kLanes] float4 (plane 0 w_norm, plane 1 + t token t's R').  The first pass
// of a warp meets the CTA's barrier before it applies rs, which the norm may still be forming.
template <int T>
__device__ __forceinline__ void down_unit(const HcAdaArgs& p, int base, int w, int sb, int lane, const float4* tile,
                                          const float* srs, unsigned tag, bool meet) {
    const int np = p.nrows / 2, li = lane % kLanes, u0 = base * kSplit + lane / kLanes;
    const bool live = u0 < np;
    const int u = live ? u0 : 0;   // a lane past the rows repeats the first pair and writes nothing
    const int8_t* q[2];
    const uint16_t* s[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int k = u * 2 + r;
        if (k < LR) { q[r] = p.qd + (size_t) k * D; s[r] = p.sd + (size_t) k * p.sd_row; }
        else { q[r] = p.qi + (size_t) (k - LR) * D; s[r] = p.si + (size_t) (k - LR) * p.si_row; }
    }
    // the weights of both groups first: their loads are in flight together
    uint4 qa[2][2], qb[2][2];
    uint16_t sh[2][2];
#pragma unroll
    for (int gi = 0; gi < 2; ++gi) {
        const int g = 32 * (w + kDownWarps * gi) + sb + kSplit * li;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const uint4* qp = reinterpret_cast<const uint4*>(q[r] + (size_t) g * G8);
            qa[gi][r] = qp[0];
            qb[gi][r] = qp[1];
            sh[gi][r] = s[r][g];
        }
    }
    float dot[2][2][T];
#pragma unroll
    for (int gi = 0; gi < 2; ++gi) {
        const int run = gi * (kPlaneF4 / 2) + li;
        float wq[2][G8];
#pragma unroll
        for (int j = 0; j < G8 / 4; ++j) {
            const float4 wn = tile[run + j * kLanes];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const uint32_t x = j < 4 ? u4_at(qa[gi][r], j) : u4_at(qb[gi][r], j - 4);
#pragma unroll
                for (int i = 0; i < 4; ++i) wq[r][4 * j + i] = s8(x, i) * f4_at(wn, i);
            }
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float4* x4 = tile + (t + 1) * kPlaneF4 + run;
            float d0 = 0.0f, d1 = 0.0f;
#pragma unroll
            for (int j = 0; j < G8 / 4; ++j) {
                const float4 v = x4[j * kLanes];
                d0 = fmaf(wq[0][4 * j], v.x, d0);
                d1 = fmaf(wq[1][4 * j], v.x, d1);
                d0 = fmaf(wq[0][4 * j + 1], v.y, d0);
                d1 = fmaf(wq[1][4 * j + 1], v.y, d1);
                d0 = fmaf(wq[0][4 * j + 2], v.z, d0);
                d1 = fmaf(wq[1][4 * j + 2], v.z, d1);
                d0 = fmaf(wq[0][4 * j + 3], v.w, d0);
                d1 = fmaf(wq[1][4 * j + 3], v.w, d1);
            }
            dot[gi][0][t] = d0;
            dot[gi][1][t] = d1;
        }
    }
    if (meet) {
        __syncthreads();   // this CTA's dots are done
        __syncthreads();   // srs is valid
    }
    float acc[2][T];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int t = 0; t < T; ++t) acc[r][t] = 0.0f;
#pragma unroll
    for (int gi = 0; gi < 2; ++gi) {
        const int g = 32 * (w + kDownWarps * gi) + sb + kSplit * li, c = g * G8 / N;
        const float sc0 = f16f(sh[gi][0]), sc1 = f16f(sh[gi][1]);
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float rs = srs[t * HC + c];
            acc[0][t] = fmaf(sc0 * rs, dot[gi][0][t], acc[0][t]);
            acc[1][t] = fmaf(sc1 * rs, dot[gi][1][t], acc[1][t]);
        }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int t = 0; t < T; ++t) {
            float v = acc[r][t];
#pragma unroll
            for (int o = kLanes / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (li == 0 && live)
                p.red[((size_t) t * p.nrows + u * 2 + r) * kClasses + w * kSplit + sb] = tagged(tag, v);
        }
}

// hc8_up's warp for output column d (the CTA's k-th), up to the butterfly's sums, which go to spart [k][T] float4 by
// stream.  slo: [T][LR] in shared memory; sq / ss: the staged codes [k][HC][LR] and scales [k][HC][2 float4]
template <int T>
__device__ __forceinline__ void up_column(const HcAdaArgs& p, int k, int d, int lane, const float* slo, float* spart,
                                          const int8_t* sq, const uint16_t* ss) {
    // the weights of the column's rows first: their loads are in flight together
    uint32_t qw[HC][kUpPasses];
    uint16_t sh[HC][kUpPasses];
#pragma unroll
    for (int c = 0; c < HC; ++c) {
#pragma unroll
        for (int j = 0; j < kUpPasses; ++j) {
            const int ch = lane + 32 * j;
            const int cl = ch < kUpChunks ? ch : lane;   // a lane past the row's end repeats a load it needs anyway
            if (p.stage_up) {
                qw[c][j] = reinterpret_cast<const uint32_t*>(sq + (k * HC + c) * LR)[cl];
                sh[c][j] = ss[(k * HC + c) * (kUpSF4 * 8) + cl / (G8 / 4)];
            } else {
                const size_t row = (size_t) c * N + d;
                qw[c][j] = reinterpret_cast<const uint32_t*>(p.qu + row * LR)[cl];
                sh[c][j] = p.su[row * p.su_row + cl / (G8 / 4)];
            }
        }
    }
    float part[HC][T];
#pragma unroll
    for (int c = 0; c < HC; ++c) {
#pragma unroll
        for (int t = 0; t < T; ++t) part[c][t] = 0.0f;
#pragma unroll
        for (int j = 0; j < kUpPasses; ++j) {
            const int ch = lane + 32 * j;
            if (ch < kUpChunks) {
                const uint32_t w = qw[c][j];
                const float sc = f16f(sh[c][j]);
                const float q0 = s8(w, 0), q1 = s8(w, 1), q2 = s8(w, 2), q3 = s8(w, 3);
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    const float4 l = *reinterpret_cast<const float4*>(slo + t * LR + 4 * ch);
                    float v = q0 * l.x;
                    v = fmaf(q1, l.y, v);
                    v = fmaf(q2, l.z, v);
                    v = fmaf(q3, l.w, v);
                    part[c][t] = fmaf(sc, v, part[c][t]);
                }
            }
        }
    }
#pragma unroll
    for (int c = 0; c < HC; ++c)
#pragma unroll
        for (int t = 0; t < T; ++t) {
            float v = part[c][t];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0) spart[(k * T + t) * HC + c] = v;
        }
}

template <int T>
__global__ void __launch_bounds__(kThreads, 1) k_hc_ada(const __grid_constant__ HcAdaArgs p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    extern __shared__ __align__(16) float4 sm[];
    __shared__ float nred[8];
    __shared__ float srs[T * HC];
    __shared__ float sw[T * HC];
    __shared__ unsigned stag[2];
    const int tid = threadIdx.x, b = blockIdx.x, G = gridDim.x, lane = tid & 31;
    // the CTA's run of output columns, its float4 chunks of the arrays' columns
    const int d0 = b * N / G, d1 = (b + 1) * N / G, ncol = d1 - d0, a0 = d0 & ~3, nf4 = (d1 - a0 + 3) >> 2;
    const int ncol_max = (N + G - 1) / G, nf4_max = up_chunks_max(G);
    const int planes = p.apply ? 2 * T + 1 : T + 1;
    float4* s_up = sm + (2 * T + 1) * kPlaneF4;
    int8_t* sq = reinterpret_cast<int8_t*>(s_up);
    uint16_t* ss = reinterpret_cast<uint16_t*>(sq + (size_t) ncol_max * HC * LR);
    float* spart = reinterpret_cast<float*>(s_up + (p.stage_up ? ncol_max * kUpColF4 : 0));
    float4* sx = reinterpret_cast<float4*>(spart) + ncol_max * T;   // [T][HC][nf4] R', then [HC][nf4] w_norm, [T][nf4] bo
    float4* sn = sx + T * HC * nf4_max;
    float4* sb = sn + HC * nf4_max;
    const BarGroup gsync{1 + (tid >> 8)};
    auto mark = [&](int i) {
        if (p.stamps != nullptr && tid == 0) {
            __threadfence();
            p.stamps[(size_t) b * kHcAdaStamps + i] = now();
        }
    };
    mark(0);
    if (tid < kGroup) {
        // ======== the class's columns into shared memory: w_norm, R, and bo when R' is to be formed
        const int cls = b % kClasses, w = cls / kSplit, sbt = cls % kSplit;
        if (tid == 0) stag[0] = (unsigned) (ld64_acquire(p.ctr) / G + 1);
        if (p.apply && tid < T * HC) sw[tid] = 2.0f / (1.0f + expf(-p.inj_prev[tid / HC][tid % HC] / (float) HC));
        for (int i = tid; i < planes * kPlaneF4; i += kGroup) {
            const int pl = i / kPlaneF4, off = i - pl * kPlaneF4;
            const int gi = off / (kPlaneF4 / 2), rest = off - gi * (kPlaneF4 / 2), li = rest / 8, j = rest - li * 8;
            const int g = 32 * (w + kDownWarps * gi) + sbt + kSplit * li, col = g * G8 + 4 * j;
            const float* src = pl == 0 ? p.w_norm + col : pl <= T ? p.R[pl - 1] + col : p.bo[pl - T - 1] + col % N;
            cp_async16(sm + pl * kPlaneF4 + gi * (kPlaneF4 / 2) + j * kLanes + li, src);
        }
        cp_async_commit();
        cp_async_wait();
        gsync();
        mark(1);
        // ======== R' in shared memory
        if (p.apply) {
            for (int i = tid; i < T * kPlaneF4; i += kGroup) {
                const int t = i / kPlaneF4, off = i - t * kPlaneF4;
                const int gi = off / (kPlaneF4 / 2), rest = off - gi * (kPlaneF4 / 2);
                const int g = 32 * (w + kDownWarps * gi) + sbt + kSplit * (rest % kLanes);
                const float wc = sw[t * HC + g * G8 / N];
                float4 x = sm[(t + 1) * kPlaneF4 + off];
                const float4 bv = sm[(T + 1 + t) * kPlaneF4 + off];
                x.x = fmaf(bv.x, wc, x.x); x.y = fmaf(bv.y, wc, x.y); x.z = fmaf(bv.z, wc, x.z); x.w = fmaf(bv.w, wc, x.w);
                sm[(t + 1) * kPlaneF4 + off] = x;
            }
            gsync();
        }
        mark(2);
        // ======== down: a warp takes kSplit pairs of rows
        {
            const int ci = b / kClasses, nc = (G - 1 - cls) / kClasses + 1, np = p.nrows / 2;
            const unsigned tag = stag[0];
            bool first = true;
            for (int base = (tid >> 5) * nc + ci; base * kSplit < np; base += kGroupWarps * nc) {
                down_unit<T>(p, base, w, sbt, lane, sm, srs, tag, first);
                first = false;
            }
            if (first) {
                __syncthreads();
                __syncthreads();
            }
        }
        mark(3);
    } else {
        const int ti = tid - kGroup;
        if (ti == 0) stag[1] = (unsigned) (ld64_acquire(p.ctr) / G + 1);
        // ======== the columns of up's epilogues into shared memory
        {
            const int nx = T * HC * nf4, nn = HC * nf4, nb = p.apply ? T * nf4 : 0;
            for (int i = ti; i < nx + nn + nb; i += kGroup) {
                if (i < nx) {
                    const int t = i / (HC * nf4), rest = i - t * (HC * nf4), c = rest / nf4, f = rest - c * nf4;
                    cp_async16(sx + (size_t) (t * HC + c) * nf4_max + f, p.R[t] + (size_t) c * N + a0 + 4 * f);
                } else if (i < nx + nn) {
                    const int c = (i - nx) / nf4, f = i - nx - c * nf4;
                    cp_async16(sn + c * nf4_max + f, p.w_norm + (size_t) c * N + a0 + 4 * f);
                } else {
                    const int t = (i - nx - nn) / nf4, f = i - nx - nn - t * nf4;
                    cp_async16(sb + t * nf4_max + f, p.bo[t] + a0 + 4 * f);
                }
            }
            cp_async_commit();
        }
        // ======== up's weights on their way
        if (p.stage_up) {
            for (int i = ti; i < ncol * kUpColF4; i += kGroup) {
                const int k = i / kUpColF4, rest = i - k * kUpColF4;
                const int c = rest / (kUpQF4 + kUpSF4), x = rest - c * (kUpQF4 + kUpSF4);
                const size_t row = (size_t) c * N + d0 + k;
                if (x < kUpQF4) cp_async16(sq + (size_t) (k * HC + c) * LR + x * 16, p.qu + row * LR + x * 16);
                else cp_async16(ss + (size_t) (k * HC + c) * (kUpSF4 * 8) + (x - kUpQF4) * 8, p.su + row * p.su_row + (x - kUpQF4) * 8);
            }
            cp_async_commit();
        }
        gsync();
        const unsigned tag = stag[1];
        // ======== norm: (stream, token) units of 256 threads, as hc8_norm's blocks
        for (int k = b; k < T * HC; k += G) {
            const int t = k / HC, c = k - t * HC;
            const float* r = p.R[t] + (size_t) c * N;
            float w = 0.0f;
            if (p.apply) w = 2.0f / (1.0f + expf(-p.inj_prev[t][c] / (float) HC));
            float s2 = 0.0f;
            for (int i = ti; i < N; i += kGroup) {
                float v = r[i];
                if (p.apply) v = fmaf(p.bo[t][i], w, v);
                s2 = fmaf(v, v, s2);
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) s2 += __shfl_xor_sync(0xffffffffu, s2, o);
            if (lane == 0) nred[ti >> 5] = s2;
            gsync();
            if (ti == 0) {
                float s = 0.0f;
#pragma unroll
                for (int x = 0; x < 8; ++x) s += nred[x];
                const float rs = rsqrtf(s / (float) N + p.eps);
                p.rs[t][c] = rs;
                p.rs64[(size_t) t * HC + c] = tagged(tag, rs);
            }
            gsync();
        }
        // ======== rs for this CTA, once its dots are done
        __syncthreads();
        cp_async_wait();
        if (ti < T * HC) srs[ti] = tag_load(p.rs64 + ti, tag, p.word, __LINE__);
        __syncthreads();
        // ======== a row's sums in hc8_down's order, a lane per (token, row)
        for (int k = ti * G + b; k < T * p.nrows; k += kGroup * G) {
            const int t = k / p.nrows, row = k - t * p.nrows;
            const unsigned long long* rd = p.red + (size_t) k * kClasses;
            unsigned long long x[kClasses];
            // all of a row's words in flight together; a word still missing is waited for alone
            for (;;) {
                int miss = -1;
#pragma unroll
                for (int i = 0; i < kClasses; ++i) x[i] = ld64_fresh(rd + i);
#pragma unroll
                for (int i = 0; i < kClasses; ++i)
                    if ((unsigned) (x[i] >> 32) != tag) miss = i;
                if (miss < 0) break;
                if ((unsigned) (tag_wait(rd + miss, tag, p.word, __LINE__) >> 32) != tag) break;
            }
            float v = 0.0f;
#pragma unroll
            for (int w = 0; w < kDownWarps; ++w) {
                float a[kSplit];
#pragma unroll
                for (int i = 0; i < kSplit; ++i) a[i] = __uint_as_float((unsigned) x[w * kSplit + i]);
#pragma unroll
                for (int o = kSplit / 2; o > 0; o >>= 1)
#pragma unroll
                    for (int i = 0; i < o; ++i) a[i] += a[i + o];
                v += a[0];
            }
            if (row < LR) {
                const float xv = v / (float) HC;
                p.lo[t][row] = xv / (1.0f + expf(-xv));
            } else {
                p.inj_out[t][row - LR] = v;
            }
        }
        if (ti * G + b < T * p.nrows) __threadfence();
    }
    __syncthreads();
    mark(4);
    // ======== every CTA's lo is written, and every CTA is done reading R
    if (tid == 0) {
        const unsigned long long old = atomicAdd(p.ctr, 1ull);
        const unsigned long long target = (old / G + 1) * G;
        const unsigned long long t0 = now();
        while (ld64_acquire(p.ctr) < target) {
            __nanosleep(32);
            const unsigned long long e = now() - t0;
            if (e > kWaitPollNs &&
                wait_give_up(p.word, e, kWaitNs, wait_abort_code_of(kWaitHcRead, 2, (unsigned) __LINE__, blockIdx.x)))
                break;
        }
    }
    __syncthreads();
    mark(5);
    // ======== lo into shared memory; R' of the CTA's columns, in shared memory and out to R_out
    float* slo = reinterpret_cast<float*>(sm);
    for (int i = tid; i < T * LR; i += kThreads) slo[i] = p.lo[i / LR][i % LR];
    if (p.apply) {
        for (int i = tid; i < T * HC * nf4; i += kThreads) {
            const int t = i / (HC * nf4), rest = i - t * (HC * nf4), c = rest / nf4, f = rest - c * nf4;
            const int ic = (t * HC + c) * nf4_max + f, col = a0 + 4 * f;
            const float wc = sw[t * HC + c];
            float4 x = sx[ic];
            const float4 bv = sb[t * nf4_max + f];
            x.x = fmaf(bv.x, wc, x.x); x.y = fmaf(bv.y, wc, x.y); x.z = fmaf(bv.z, wc, x.z); x.w = fmaf(bv.w, wc, x.w);
            sx[ic] = x;
            float* out = p.R_out[t] + (size_t) c * N + col;
            if (col >= d0 && col + 4 <= d1) {
                *reinterpret_cast<float4*>(out) = x;
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    if (col + j >= d0 && col + j < d1) out[j] = f4_at(x, j);
            }
        }
    }
    __syncthreads();
    mark(6);
    // ======== up: a warp per output column, then the epilogues, a lane per (column, token)
    for (int k = tid >> 5; k < ncol; k += kThreads / 32) up_column<T>(p, k, d0 + k, lane, slo, spart, sq, ss);
    __syncthreads();
    mark(7);
    const float* sxf = reinterpret_cast<const float*>(sx);
    const float* snf = reinterpret_cast<const float*>(sn);
    for (int i = tid; i < ncol * T; i += kThreads) {
        const int t = i % T, k = i / T, e = d0 + k - a0;
        const float4 pc = reinterpret_cast<const float4*>(spart)[k * T + t];
        float m = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            const float xn = sxf[(size_t) (t * HC + c) * nf4_max * 4 + e] * snf[(size_t) c * nf4_max * 4 + e] * srs[t * HC + c];
            m = fmaf(xn, 1.0f / (1.0f + expf(-f4_at(pc, c))), m);
        }
        p.mixed[t][d0 + k] = m / (float) HC;
    }
    mark(8);
#elif defined(__CUDA_ARCH__)
    __trap();
#endif
}

using HFn = void (*)(HcAdaArgs);
HFn ada_kernel(int T) {
    switch (T) {
        case 1: return k_hc_ada<1>;
        case 2: return k_hc_ada<2>;
        case 3: return k_hc_ada<3>;
        case 4: return k_hc_ada<4>;
        case 5: return k_hc_ada<5>;
        case 6: return k_hc_ada<6>;
        case 7: return k_hc_ada<7>;
        default: return k_hc_ada<8>;
    }
}
// the shared memory a launch takes: the tile, up's staged weights when they fit, up's dots, the epilogues' columns
int ada_smem_base(int T, int G) {
    const int nf = up_chunks_max(G);
    return (2 * T + 1) * kPlaneF4 * 16 + ((N + G - 1) / G * T + T * HC * nf + HC * nf + T * nf) * 16;
}
bool ada_stage_up(int T, int G) { return ada_smem_base(T, G) + (N + G - 1) / G * kUpColF4 * 16 <= kSmemMax; }
int ada_smem(int T, int G) {
    const int need = ada_smem_base(T, G) + (ada_stage_up(T, G) ? (N + G - 1) / G * kUpColF4 * 16 : 0);
    return need > kSmemMin ? need : kSmemMin;
}

struct AdaDev {
    bool checked = false, ok = false, ada = false;
    int ctas = 0;
    unsigned long long *ctr = nullptr, *red = nullptr, *rs64 = nullptr;
};
AdaDev g_dev[16];
bool g_on = false;
int g_test_grid = 0;
unsigned long long* g_test_stamps = nullptr;

AdaDev& ada_dev() {
    int dev = 0;
    cudaGetDevice(&dev);
    static AdaDev none;
    if (dev < 0 || dev >= 16) return none;
    AdaDev& d = g_dev[dev];
    if (!d.checked) {
        d.checked = true;
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        d.ada = major == 8 && minor == 9;
    }
    return d;
}

bool al(const void* p, uintptr_t m) { return p != nullptr && ((uintptr_t) p & (m - 1)) == 0; }
int grid_of(const AdaDev& d) { return g_test_grid > 0 ? g_test_grid : d.ctas; }

}  // namespace

void hc_read_ada_enable(bool on) { g_on = on; }

void hc_read_ada_test_grid(int ctas) {
    g_test_grid = ctas;
    AdaDev& d = ada_dev();   // the barrier's count is a multiple of the grid, and a new count starts the tags again
    cudaMemset(d.ctr, 0, 8);
    cudaMemset(d.rs64, 0, (size_t) MT * HC * 8);
    cudaMemset(d.red, 0, (size_t) MT * (LR + HC) * kClasses * 8);
}

void hc_read_ada_test_stamps(unsigned long long* stamps, int ord) {
    g_test_stamps = stamps == nullptr ? nullptr : stamps + (size_t) ord * grid_of(ada_dev()) * kHcAdaStamps;
}

void hc_read_ada_init() {
    AdaDev& d = ada_dev();
    if (!g_on || !d.ada || d.ok) return;
    int dev = 0, sms = 0, optin = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    wait_abort_init();
    bool ok = sms >= kClasses && optin >= kSmemMax + 256 &&
              cudaMalloc((void**) &d.ctr, 8) == cudaSuccess && cudaMemset(d.ctr, 0, 8) == cudaSuccess &&
              cudaMalloc((void**) &d.rs64, (size_t) MT * HC * 8) == cudaSuccess && cudaMemset(d.rs64, 0, (size_t) MT * HC * 8) == cudaSuccess &&
              cudaMalloc((void**) &d.red, (size_t) MT * (LR + HC) * kClasses * 8) == cudaSuccess &&
              cudaMemset(d.red, 0, (size_t) MT * (LR + HC) * kClasses * 8) == cudaSuccess;
    for (int T = 1; ok && T <= MT; ++T) {
        const HFn f = ada_kernel(T);
        int occ = 0;
        ok = cudaFuncSetAttribute((const void*) f, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemMax) == cudaSuccess &&
             cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, f, kThreads, (size_t) ada_smem(T, sms)) == cudaSuccess && occ >= 1;
    }
    d.ok = ok && cudaGetLastError() == cudaSuccess;
    d.ctas = sms;
    if (!d.ok) {
        basalt::log::error("basalt", "hc_read_ada: FATAL: kernel initialisation failed (%s)", cudaGetErrorString(cudaGetLastError()));
        std::exit(1);
    }
    basalt::log::info("basalt", "hc read: sm_89 kernel, %d CTAs of %d threads", d.ctas, kThreads);
}

int hc_read_ada_occupancy(int n_tok, int* smem_bytes) {
    int occ = 0;
    const int g = grid_of(ada_dev());
    if (smem_bytes != nullptr) *smem_bytes = ada_smem(n_tok, g);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, ada_kernel(n_tok), kThreads, (size_t) ada_smem(n_tok, g));
    return occ;
}

bool hc_read_ada_supported(const FusedGrArgs* a, int n_tok) {
    const AdaDev& d = ada_dev();
    if (!g_on || !d.ok || a == nullptr || n_tok < 1 || n_tok > MT || grid_of(d) < kClasses) return false;
    Hc8View vd, vu, vi;
    const bool inj = a[0].w_inject != nullptr;
    if (!hc8_view(a[0].w_down, vd) || !hc8_view(a[0].w_up, vu) || (inj && !hc8_view(a[0].w_inject, vi))) return false;
    if (vd.cols != D || vd.rows != LR || vu.rows != D || vu.cols != LR || (inj && (vi.rows != HC || vi.cols != D))) return false;
    if (!al(a[0].w_norm, 16) || !al(vu.q, 16) || !al(vu.s, 16) || vu.srow != kUpSF4 * 8) return false;
    for (int t = 0; t < n_tok; ++t) {
        const FusedGrArgs& x = a[t];
        if (!al(x.R, 16) || !x.w_norm || !x.lo || !x.rs || !x.mixed || (inj && !x.inject_out) || x.apply != a[0].apply ||
            (x.apply && (!al(x.bo_prev, 16) || !x.inj_prev || !x.R_out)) || x.w_down != a[0].w_down || x.w_up != a[0].w_up ||
            x.w_inject != a[0].w_inject || x.w_norm != a[0].w_norm)
            return false;
    }
    return true;
}

void hc_read_ada(const FusedGrArgs* a, int n_tok, void* stream) {
    AdaDev& d = ada_dev();
    if (!hc_read_ada_supported(a, n_tok)) {
        basalt::log::error("basalt", "hc_read_ada: unsupported arguments or device");
        std::exit(1);
    }
    Hc8View vd, vu, vi;
    const bool inj = a[0].w_inject != nullptr;
    hc8_view(a[0].w_down, vd);
    hc8_view(a[0].w_up, vu);
    if (inj) hc8_view(a[0].w_inject, vi);
    HcAdaArgs p{};
    for (int t = 0; t < n_tok; ++t) {
        const FusedGrArgs& x = a[t];
        p.R[t] = x.R; p.R_out[t] = x.R_out; p.bo[t] = x.bo_prev; p.inj_prev[t] = x.inj_prev;
        p.lo[t] = x.lo; p.rs[t] = x.rs; p.inj_out[t] = x.inject_out; p.mixed[t] = x.mixed;
    }
    const int g = grid_of(d);
    p.w_norm = a[0].w_norm;
    p.qd = vd.q; p.sd = vd.s; p.sd_row = vd.srow;
    p.qu = vu.q; p.su = vu.s; p.su_row = vu.srow;
    if (inj) { p.qi = vi.q; p.si = vi.s; p.si_row = vi.srow; }
    p.nrows = LR + (inj ? HC : 0);
    p.apply = a[0].apply;
    p.stage_up = ada_stage_up(n_tok, g);
    p.eps = a[0].eps;
    p.red = d.red;
    p.rs64 = d.rs64;
    p.ctr = d.ctr;
    p.word = wait_abort_device_word();
    p.stamps = g_test_stamps;
    ada_kernel(n_tok)<<<g, kThreads, (size_t) ada_smem(n_tok, g), (cudaStream_t) stream>>>(p);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "hc_read_ada: %s", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace basalt::kernels
