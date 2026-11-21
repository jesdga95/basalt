// src/models/inkling2/kernels/glue.cu - Inkling 2's fused glue for a decode window: the block-residual mixes with the norm
// and the q8_1 rounding of the products' inputs, the loop controllers (down, up, blend, norm and rounding), the attention
// preparation a key/value head a block, the attention output's convolution with the residual and the MLP's mix, and the
// experts' sum with its convolution and the residual.
//
// A block takes one row (or one row and head, or one stream and row), and no row's arithmetic reads another row's partial
// sums, so a row's bits do not depend on the rows beside it. A window's convolutions read the rows before them in the
// window and the ring for the positions before the window, and each row writes its own ring slot: a window of at most
// kDecodeRows rows touches distinct slots (kDecodeRows + kConv - 1 <= kConvRing).
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/q8_1_finite.hpp"
#include "basalt/models/inkling2/kernels.hpp"
#include "prep.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace basalt::models::inkling2 {
namespace {

namespace bk = basalt::kernels;

static_assert(kDecodeRows + kConv - 1 <= kConvRing, "a window's rows and the positions they read fit the ring");

constexpr int kThreads = 768;
constexpr int kWarpsB = kThreads / 32;
constexpr int kPer = kWidth / kThreads;   // 3 values a thread: thread t holds t, t + 768, t + 1536
static_assert(kWidth % kThreads == 0, "the width splits evenly over a block");

struct Q81 {
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q81) == 36, "a q8_1 block is 36 bytes");

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("inkling2 ") + what + ": " + cudaGetErrorString(e));
}

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

// Sums of N values per thread over the block, in a fixed order (each warp's, then the warps' in order); every thread gets
// the N results. `red` holds N * (kWarpsB + 1) floats.
template <int N>
__device__ __forceinline__ void block_sums(float (&v)[N], float* red) {
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] = warp_sum(v[i]);
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0)
#pragma unroll
        for (int i = 0; i < N; ++i) red[i * kWarpsB + w] = v[i];
    __syncthreads();
    if (threadIdx.x < N) {
        float s = 0.f;
#pragma unroll
        for (int j = 0; j < kWarpsB; ++j) s += red[threadIdx.x * kWarpsB + j];
        red[N * kWarpsB + threadIdx.x] = s;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] = red[N * kWarpsB + i];
    __syncthreads();
}

// Value c of a q8_1 row, from the warp that holds the block's 32 values (one a lane, c % 32 the lane): d = amax / 127.
__device__ __forceinline__ void q8_store(float x, Q81* row, int c) {
    const float amax = warp_max(fabsf(x));
    const float sum = warp_sum(x);
    const float d = bk::q8_1_finite(amax / 127.0f);
    Q81& b = row[c / 32];
    b.qs[c % 32] = bk::q8_1_quant(x, d, amax);
    if ((c & 31) == 0) b.ds = bk::q8_1_ds(d, sum);
}

// The block-residual mix of the n states v[i] (thread-held values) scored by NQ queries (thread-held values qv, loaded
// before the kernel's pdl_wait): scores q . s / sqrt(kWidth) over the RMS-normed states (eps 1e-6), softmax, the weighted
// sum of the normed states.
template <int NQ>
__device__ __forceinline__ void mix_rows(const float (&v)[kMaxMix][kPer], int n, const float (&qv)[NQ][kPer],
                                         float (&mixed)[NQ][kPer], float* red) {
    float part[kMaxMix * (NQ + 1)];
#pragma unroll
    for (int i = 0; i < kMaxMix * (NQ + 1); ++i) part[i] = 0.f;
#pragma unroll
    for (int i = 0; i < kMaxMix; ++i) {
        if (i >= n) break;
#pragma unroll
        for (int j = 0; j < kPer; ++j) {
            const float x = v[i][j];
            part[i * (NQ + 1)] = fmaf(x, x, part[i * (NQ + 1)]);
#pragma unroll
            for (int q = 0; q < NQ; ++q) part[i * (NQ + 1) + 1 + q] = fmaf(qv[q][j], x, part[i * (NQ + 1) + 1 + q]);
        }
    }
    block_sums<kMaxMix * (NQ + 1)>(part, red);
    float inv[kMaxMix];
#pragma unroll
    for (int i = 0; i < kMaxMix; ++i) inv[i] = i < n ? rsqrtf(part[i * (NQ + 1)] / (float) kWidth + 1e-6f) : 0.f;
    const float scale = rsqrtf((float) kWidth);
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
        float sc[kMaxMix], m = -INFINITY;
#pragma unroll
        for (int i = 0; i < kMaxMix; ++i) {
            sc[i] = i < n ? part[i * (NQ + 1) + 1 + q] * inv[i] * scale : -INFINITY;
            m = fmaxf(m, sc[i]);
        }
        float den = 0.f;
#pragma unroll
        for (int i = 0; i < kMaxMix; ++i) {
            sc[i] = i < n ? expf(sc[i] - m) : 0.f;
            den += sc[i];
        }
#pragma unroll
        for (int j = 0; j < kPer; ++j) {
            float acc = 0.f;
#pragma unroll
            for (int i = 0; i < kMaxMix; ++i)
                if (i < n) acc = fmaf(sc[i] / den, v[i][j] * inv[i], acc);
            mixed[q][j] = acc;
        }
    }
}

// Thread t's kPer values of a kWidth vector (t, t + 768, t + 1536): a kernel loads its weights so before its pdl_wait.
__device__ __forceinline__ void load_row(const float* __restrict__ p, float (&x)[kPer]) {
#pragma unroll
    for (int j = 0; j < kPer; ++j) x[j] = p[threadIdx.x + j * kThreads];
}

// RMS norm of the thread-held row x with the thread-held weights w, into q8_1 (and fp32 when `f32` is set).
__device__ __forceinline__ void norm_q8(const float (&x)[kPer], const float (&w)[kPer], float eps, void* q8, float* f32,
                                        int r, float* red) {
    float ss[1] = {0.f};
#pragma unroll
    for (int j = 0; j < kPer; ++j) ss[0] = fmaf(x[j], x[j], ss[0]);
    block_sums<1>(ss, red);
    const float inv = rsqrtf(ss[0] / (float) kWidth + eps);
    Q81* row = reinterpret_cast<Q81*>(q8) + (size_t) r * (kWidth / 32);
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        const int c = threadIdx.x + j * kThreads;
        const float y = x[j] * inv * w[j];
        if (f32) f32[(size_t) r * kWidth + c] = y;
        q8_store(y, row, c);
    }
}

using prep::conv_in;

// ---- mixes -------------------------------------------------------------------------------------------------------------

// Block (row, output q): the mix scored by query q (or the last state), then its norm into q8_1, or the mix as it is.
// The older key/value cells of a visit's attention into L2 (MixNorm's kv_*): the block's threads request 32 KiB pieces of
// the one or two runs (the ring may wrap) of the keys and of the values. Up to kKvAheadCells positions: a longer run does
// not stay in L2 through the projections and is read twice.
constexpr int kKvAheadCells = 256;
__device__ __forceinline__ void kv_read_ahead(const MixNorm& a) {
    const int p0 = a.kv_pos[0];
    if (a.ring_k && threadIdx.x < 3 * 3) {   // ring entry j = 1..3 positions back of the key, value and output rings
        const int j = threadIdx.x % 3 + 1, which = threadIdx.x / 3;
        const int pj = p0 - j;
        if (pj >= 0) {
            const int w = which == 2 ? kWidth : kKvHeads * kHeadDim;
            const float* ring = which == 0 ? a.ring_k : which == 1 ? a.ring_v : a.ring_a;
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(ring + (size_t) (pj % kConvRing) * w), "r"((unsigned) (w * 4))
                         : "memory");
        }
    }
    const int lo = a.kv_window > 0 ? max(0, p0 - a.kv_window + 1) : 0;
    const int n = p0 - lo;
    if (n <= 0 || n > kKvAheadCells) return;
    const int64_t c0 = lo % a.kv_cap, run1 = min((int64_t) n, a.kv_cap - c0), run2 = n - run1;
    const uint64_t b1 = (uint64_t) run1 * a.kv_cell, b2 = (uint64_t) run2 * a.kv_cell;
    constexpr uint64_t kPiece = 32768;
    const uint64_t n1 = (b1 + kPiece - 1) / kPiece, n2 = (b2 + kPiece - 1) / kPiece;
    for (uint64_t j = threadIdx.x; j < 2 * (n1 + n2); j += blockDim.x) {
        const char* base = (const char*) (j < n1 + n2 ? a.kv_k : a.kv_v);
        const uint64_t i = j % (n1 + n2);
        const uint64_t start = i < n1 ? (uint64_t) c0 * a.kv_cell + i * kPiece : (i - n1) * kPiece;
        const uint64_t end = i < n1 ? (uint64_t) c0 * a.kv_cell + b1 : b2;
        const uint64_t take = min(kPiece, end - start);
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(base + start), "r"((unsigned) take) : "memory");
    }
}

__global__ void __launch_bounds__(kThreads) mix_norm_kernel(MixNorm a) {
    const int r = blockIdx.x, q = blockIdx.y, n = a.st.n;
    float qv[1][kPer], nw[kPer];   // the weights, read while the kernel before finishes
    if (a.queries) load_row(a.queries + (size_t) q * kWidth, qv[0]);
    if (!a.mix_out) load_row(a.norm[q], nw);
    if (a.kv_k && r == 0 && q == 0) kv_read_ahead(a);   // constant through the window: before the wait
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    __shared__ float red[kMaxMix * 2 * (kWarpsB + 1)];
    float v[kMaxMix][kPer];
#pragma unroll
    for (int i = 0; i < kMaxMix; ++i) {
        if (i >= n) break;
#pragma unroll
        for (int j = 0; j < kPer; ++j) v[i][j] = a.st.s[i][(size_t) r * kWidth + threadIdx.x + j * kThreads];
    }
    float mixed[1][kPer];
    if (a.queries == nullptr) {   // the last state as it is (one output)
#pragma unroll
        for (int i = 0; i < kMaxMix; ++i)
            if (i == n - 1)
#pragma unroll
                for (int j = 0; j < kPer; ++j) mixed[0][j] = v[i][j];
    } else {
        mix_rows<1>(v, n, qv, mixed, red);
    }
    if (a.mix_out) {
#pragma unroll
        for (int j = 0; j < kPer; ++j)
            a.mix_out[(size_t) q * a.out_stride + (size_t) r * kWidth + threadIdx.x + j * kThreads] = mixed[0][j];
        return;
    }
    norm_q8(mixed[0], nw, a.eps, a.q8[q], q == 0 ? a.norm_out : nullptr, r, red);
}

// ---- loop controllers --------------------------------------------------------------------------------------------------

// hid[s][r][o] = down_s[o] . [rms(dest_s[r]) | rms(msg_s[r])]: block (o0 / O, s) takes the O outputs o0.., warp w a
// slice of the 2 kWidth inputs (the first half of the warps dest's, the second msg's); each warp's dots and sums of squares
// over its slice, then the halves' sums in warp order, each half's dot scaled by its row's norm. A window's inputs are read
// once for O outputs; each output's sums are added in the same order whatever O is, so every O gives the same bits.
constexpr int kDownWarps = 8;
constexpr int kSlice = 2 * kWidth / kDownWarps;   // 576 inputs a warp
static_assert(kWidth % (kSlice / 2) == 0 && kSlice % 64 == 0, "a slice is whole pairs of every lane, inside one half");
// RMAX: the rows a launch takes at most (2, 4 or kDecodeRows). A lane loads all of its input pairs of up to 4 rows before
// their first sum; 8 rows take two such passes (registers).
template<int O, int RMAX>
__global__ void __launch_bounds__(kDownWarps * 32) ctrl_down_kernel(CtrlArgs a, int rows) {
    __shared__ float part[kDownWarps][O][RMAX][2];
    const int o0 = blockIdx.x * O, s = blockIdx.y, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const bool second = w >= kDownWarps / 2;
    const int c0 = (w % (kDownWarps / 2)) * kSlice;   // the slice's first channel in its half
    const float* in = second ? a.msg[s] : a.dest[s];
    const uint32_t* wr = reinterpret_cast<const uint32_t*>(a.down[s] + (size_t) o0 * 2 * kWidth + (second ? kWidth : 0) + c0);
    // kIn of a lane's input pairs of every row are loaded before their sums, so the loads overlap instead of each waiting
    // for the one before it (a window's 8 rows were 72 loads in a row); the sums are added in the same order as one at a
    // time. The lane's weights are all read before pdl_wait, while the kernel before finishes.
    constexpr int kIn = 9;
    constexpr int kPass = RMAX < 4 ? RMAX : 4;   // rows whose inputs are loaded together
    constexpr int kIters = kSlice / 2 / (32 * kIn);
    static_assert((kSlice / 2) % (32 * kIn) == 0, "a lane's pairs split into whole batches");
    uint32_t wu[kIters][kIn][O];
#pragma unroll
    for (int it = 0; it < kIters; ++it)
#pragma unroll
        for (int b = 0; b < kIn; ++b)
#pragma unroll
            for (int j = 0; j < O; ++j) wu[it][b][j] = wr[(size_t) j * kWidth + lane + 32 * (it * kIn + b)];   // output o0 + j: 2 kWidth bf16, kWidth words
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    float dot[O][RMAX], sq[RMAX];
#pragma unroll
    for (int t = 0; t < RMAX; ++t) {
        sq[t] = 0.f;
#pragma unroll
        for (int j = 0; j < O; ++j) dot[j][t] = 0.f;
    }
#pragma unroll
    for (int it = 0; it < kIters; ++it) {
        const int i0 = lane + it * 32 * kIn;
#pragma unroll
        for (int h = 0; h < RMAX; h += kPass) {   // rows h.. h + kPass: their batch of inputs loaded, then summed
            float2 x[kIn][kPass];
#pragma unroll
            for (int b = 0; b < kIn; ++b)
#pragma unroll
                for (int t = 0; t < kPass; ++t)
                    if (h + t < rows) x[b][t] = reinterpret_cast<const float2*>(in + (size_t) (h + t) * kWidth + c0)[i0 + 32 * b];
#pragma unroll
            for (int b = 0; b < kIn; ++b) {
                float w0[O], w1[O];
#pragma unroll
                for (int j = 0; j < O; ++j) {
                    const uint32_t u = wu[it][b][j];
                    w0[j] = __uint_as_float(u << 16);
                    w1[j] = __uint_as_float(u & 0xffff0000u);
                }
#pragma unroll
                for (int t = 0; t < kPass; ++t) {
                    if (h + t < rows) {
#pragma unroll
                        for (int j = 0; j < O; ++j) dot[j][h + t] = fmaf(w1[j], x[b][t].y, fmaf(w0[j], x[b][t].x, dot[j][h + t]));
                        sq[h + t] = fmaf(x[b][t].y, x[b][t].y, fmaf(x[b][t].x, x[b][t].x, sq[h + t]));
                    }
                }
            }
        }
    }
#pragma unroll
    for (int t = 0; t < RMAX; ++t) {
        if (t >= rows) break;
        const float q = warp_sum(sq[t]);
#pragma unroll
        for (int j = 0; j < O; ++j) {
            const float d = warp_sum(dot[j][t]);
            if (lane == 0) {
                part[w][j][t][0] = d;
                part[w][j][t][1] = q;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < rows * O) {
        const int t = threadIdx.x % rows, j = threadIdx.x / rows;
        float dd = 0.f, sd = 0.f, dm = 0.f, sm = 0.f;
#pragma unroll
        for (int k = 0; k < kDownWarps / 2; ++k) {
            dd += part[k][j][t][0];
            sd += part[k][j][t][1];
            dm += part[kDownWarps / 2 + k][j][t][0];
            sm += part[kDownWarps / 2 + k][j][t][1];
        }
        a.scratch[((size_t) s * kDecodeRows + t) * a.rank + o0 + j] =
            dd * rsqrtf(sd / (float) kWidth + 1e-6f) + dm * rsqrtf(sm / (float) kWidth + 1e-6f);
    }
}

// gf[s][r][i] = up_s[i] . hid[s][r] for the 2 kWidth rows of up_s: eight lanes a row (rank / 8 values each), 32 rows a
// block.
constexpr int kUpWords = 8;   // a lane's first weight words, read before pdl_wait (all of them up to rank 128)
__global__ void __launch_bounds__(256) ctrl_up_kernel(CtrlArgs a, int rows) {
    extern __shared__ float hs[];   // [rows][rank]
    const int s = blockIdx.y;
    const int row = blockIdx.x * 32 + (threadIdx.x >> 3), part = threadIdx.x & 7;
    const int per = a.rank / 8;
    const uint32_t* ur = reinterpret_cast<const uint32_t*>(a.up[s] + (size_t) row * a.rank + part * per);
    uint32_t u0w[kUpWords];
#pragma unroll
    for (int k = 0; k < kUpWords; ++k) u0w[k] = row < 2 * kWidth && k < per / 2 ? ur[k] : 0u;
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    const float* hid = a.scratch + (size_t) s * kDecodeRows * a.rank;
    for (int i = threadIdx.x; i < rows * a.rank; i += blockDim.x) hs[i] = hid[i];
    __syncthreads();
    float acc[kDecodeRows];
#pragma unroll
    for (int t = 0; t < kDecodeRows; ++t) acc[t] = 0.f;
    if (row < 2 * kWidth) {
#pragma unroll
        for (int k = 0; k < kUpWords; ++k) {
            if (k >= per / 2) break;
            const uint32_t u = u0w[k];
            const float u0 = __uint_as_float(u << 16), u1 = __uint_as_float(u & 0xffff0000u);
            const int kk = part * per + 2 * k;
#pragma unroll
            for (int t = 0; t < kDecodeRows; ++t) {
                if (t >= rows) break;
                acc[t] = fmaf(u1, hs[t * a.rank + kk + 1], fmaf(u0, hs[t * a.rank + kk], acc[t]));
            }
        }
        for (int k = kUpWords; k < per / 2; ++k) {
            const uint32_t u = ur[k];
            const float u0 = __uint_as_float(u << 16), u1 = __uint_as_float(u & 0xffff0000u);
            const int kk = part * per + 2 * k;
#pragma unroll
            for (int t = 0; t < kDecodeRows; ++t) {
                if (t >= rows) break;
                acc[t] = fmaf(u1, hs[t * a.rank + kk + 1], fmaf(u0, hs[t * a.rank + kk], acc[t]));
            }
        }
    }
    float* gf = a.scratch + (size_t) kMaxStreams * kDecodeRows * a.rank;
#pragma unroll
    for (int t = 0; t < kDecodeRows; ++t) {
        if (t >= rows) break;
        float v = acc[t];
        v += __shfl_xor_sync(0xffffffffu, v, 1);
        v += __shfl_xor_sync(0xffffffffu, v, 2);
        v += __shfl_xor_sync(0xffffffffu, v, 4);
        if (part == 0 && row < 2 * kWidth) gf[((size_t) s * kDecodeRows + t) * 2 * kWidth + row] = v;
    }
}

// The blend of stream s and row r, then its norm into q8_1.
__global__ void __launch_bounds__(kThreads) ctrl_blend_kernel(CtrlArgs a) {
    const int r = blockIdx.x, s = blockIdx.y;
    float nw[kPer];
    load_row(a.norm[s], nw);
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    __shared__ float red[kWarpsB + 1];
    const float* gf = a.scratch + (size_t) kMaxStreams * kDecodeRows * a.rank + ((size_t) s * kDecodeRows + r) * 2 * kWidth;
    float y[kPer];
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        const int c = threadIdx.x + j * kThreads;
        const float g = tanhf(gf[c]), f = tanhf(gf[kWidth + c]);
        y[j] = (1.f - f) * a.dest[s][(size_t) r * kWidth + c] + g * a.msg[s][(size_t) r * kWidth + c];
    }
    norm_q8(y, nw, a.eps, a.q8[s], s == 0 ? a.norm_out : nullptr, r, red);
}

// ---- attention preparation ---------------------------------------------------------------------------------------------

// Block (row, key/value head g), four warps: warp 0 the head's keys (convolved, clipped, cached; the raw inputs into the
// ring), warp 1 its values (normed), warps 2 and 3 its three query heads, their relative states and sinks (prep.cuh).
__global__ void __launch_bounds__(128) prepare_kernel(AttnPrep a) {
    const int r = blockIdx.x, g = blockIdx.y, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float wt[4][kConv];   // the weights, read while the kernel before finishes
    prep::load_weights(a, g, w, lane, wt);
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    const int p = a.pos[r];
    if (w < 2) {
        float val[4], raw[4], f;
        prep::kv_row(a, r, p, g, w, lane, wt, val, raw, f);
        void* cache = w ? a.v_cache : a.k_cache;
        const size_t cell = ((size_t) (p % a.cap) * kKvHeads + g) * kHeadDim + lane * 4;
#pragma unroll
        for (int u = 0; u < 4; ++u) prep::store_kv(cache, a.fp8, cell + u, val[u] * f);
        float* ring = w ? a.v_ring : a.k_ring;
        *reinterpret_cast<float4*>(ring + (size_t) (p % kConvRing) * (kKvHeads * kHeadDim) + g * kHeadDim + lane * 4) =
            make_float4(raw[0], raw[1], raw[2], raw[3]);
        return;
    }
    prep::q_heads(a, r, p, g, w, lane, wt, a.q + ((size_t) r * kHeads + g * 3) * kHeadDim, a.r + ((size_t) r * kHeads + g * 3) * kRel,
                  a.sink + (size_t) r * kHeads + g * 3);
}

// ---- after the attention -----------------------------------------------------------------------------------------------

__global__ void __launch_bounds__(kThreads) post_attn_kernel(PostAttn a) {
    float taps[kPer][kConv], qv[1][kPer], nw[kPer];   // the weights, read while the kernel before finishes
#pragma unroll
    for (int j = 0; j < kPer; ++j)
#pragma unroll
        for (int k = 0; k < kConv; ++k) taps[j][k] = a.taps[(threadIdx.x + j * kThreads) * kConv + (kConv - 1 - k)];
    load_row(a.query, qv[0]);
    if (!a.mix_out) load_row(a.norm, nw);
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    __shared__ float red[kMaxMix * 2 * (kWarpsB + 1)];
    const int r = blockIdx.x;
    float v[kMaxMix][kPer];
    const int nb = a.n_blocks + (a.boundary ? 1 : 0), n = nb + 1;
#pragma unroll
    for (int i = 0; i < kMaxMix - 1; ++i) {
        if (i >= nb) break;
        const float* s = i < a.n_blocks ? a.blocks[i] : a.hid;
#pragma unroll
        for (int j = 0; j < kPer; ++j) v[i][j] = s[(size_t) r * kWidth + threadIdx.x + j * kThreads];
    }
    // the convolution of the output projection's rows, the residual, the ring slot of this row: every input is read
    // before the first store (the ring is one of them)
    float xin[kPer][kConv], hres[kPer];
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        const int c = threadIdx.x + j * kThreads;
#pragma unroll
        for (int k = 0; k < kConv; ++k) xin[j][k] = conv_in(a.o, kWidth, a.ring, kWidth, a.pos, r, k, c);
        hres[j] = a.boundary ? 0.f : a.hid[(size_t) r * kWidth + c];
    }
    const int slot = a.pos[r] % kConvRing;
    float res[kPer];
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        const int c = threadIdx.x + j * kThreads;
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < kConv; ++k) acc = fmaf(taps[j][k], xin[j][k], acc);
        const float conv = __fmul_rn(acc + xin[j][0], rsqrtf((float) kConv + 1.f));
        res[j] = a.boundary ? conv : conv + hres[j];
        a.resid[(size_t) r * kWidth + c] = res[j];
        a.ring[(size_t) slot * kWidth + c] = xin[j][0];
    }
#pragma unroll
    for (int i = 0; i < kMaxMix; ++i)
        if (i == nb)
#pragma unroll
            for (int j = 0; j < kPer; ++j) v[i][j] = res[j];
    float mixed[1][kPer];
    mix_rows<1>(v, n, qv, mixed, red);
    if (a.mix_out) {
#pragma unroll
        for (int j = 0; j < kPer; ++j) a.mix_out[(size_t) r * kWidth + threadIdx.x + j * kThreads] = mixed[0][j];
        return;
    }
    norm_q8(mixed[0], nw, a.eps, a.q8, a.norm_out, r, red);
}

// ---- experts -----------------------------------------------------------------------------------------------------------

// Block (row, 256 channels), a thread a channel: the experts' weighted sums of this row and the rows before it in the
// window that its convolution reads, the ring for the positions before the window, plus the residual; the ring takes this
// row's sum.
__global__ void __launch_bounds__(256) moe_finish_kernel(MoeFinish a) {
    const int r = blockIdx.x;
    const int c = blockIdx.y * 256 + threadIdx.x;
    float taps[kConv];   // read while the kernel before finishes
#pragma unroll
    for (int j = 0; j < kConv; ++j) taps[j] = a.taps[c * kConv + (kConv - 1 - j)];
    if (threadIdx.x < kConv - 1) {   // the block's ring entries of the three positions before this row into L2
        const int pj = a.pos[r] - 1 - (int) threadIdx.x;
        if (pj >= 0)
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(a.ring + (size_t) (pj % kConvRing) * kWidth + blockIdx.y * 256),
                         "r"(256u * 4u) : "memory");
    }
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    {
        float acc = 0.f, x0 = 0.f;
#pragma unroll
        for (int j = 0; j < kConv; ++j) {
            float x;
            if (r - j >= 0) {
                x = 0.f;
                for (int s = 0; s < a.k; ++s) x = fmaf(a.w[(r - j) * a.k + s], a.rows[((size_t) (r - j) * a.k + s) * kWidth + c], x);
            } else {
                const int p = a.pos[r] - j;
                x = p < 0 ? 0.f : a.ring[(size_t) (p % kConvRing) * kWidth + c];
            }
            if (j == 0) x0 = x;
            acc = fmaf(taps[j], x, acc);
        }
        const size_t i = (size_t) r * kWidth + c;
        const float y = __fmul_rn(acc + x0, rsqrtf((float) kConv + 1.f)) + a.resid[i];
        a.out[i] = y;
        if (a.tap) a.tap[i] = y;
        a.ring[(size_t) (a.pos[r] % kConvRing) * kWidth + c] = x0;
    }
}

void check_rows(int rows, const char* what) {
    if (rows > kDecodeRows) throw std::runtime_error(std::string("inkling2 ") + what + ": more rows than a decode window");
}

}  // namespace

void mix_norm(const MixNorm& a, int n_out, int rows, void* stream) {
    if (rows <= 0) return;
    check_rows(rows, "mix_norm");
    if (a.st.n < 1 || a.st.n > kMaxMix || n_out < 1 || (!a.mix_out && n_out > 3))
        throw std::runtime_error("inkling2 mix_norm: states or outputs");
    bk::launch_pdl(mix_norm_kernel, dim3((unsigned) rows, (unsigned) n_out), dim3(kThreads), 0, (cudaStream_t) stream, a);
    check("mix_norm");
}

size_t ctrl_scratch_floats(int rank) {
    return (size_t) kMaxStreams * kDecodeRows * ((size_t) rank + 2 * kWidth);
}

void controllers(const CtrlArgs& a, int streams, int rows, void* stream) {
    if (rows <= 0) return;
    check_rows(rows, "controllers");
    if (streams < 1 || streams > kMaxStreams || a.rank <= 0 || a.rank % 16) throw std::runtime_error("inkling2 controllers: streams or rank");
    cudaStream_t s = (cudaStream_t) stream;
    // a window of 3+ rows reads its inputs once for 4 outputs (they are 8 rows x 18 KB), a lone row keeps rank blocks
    if (rows >= 5) bk::launch_pdl(ctrl_down_kernel<4, kDecodeRows>, dim3((unsigned) a.rank / 4, (unsigned) streams), dim3(kDownWarps * 32), 0, s, a, rows);
    else if (rows >= 3) bk::launch_pdl(ctrl_down_kernel<4, 4>, dim3((unsigned) a.rank / 4, (unsigned) streams), dim3(kDownWarps * 32), 0, s, a, rows);
    else bk::launch_pdl(ctrl_down_kernel<1, 2>, dim3((unsigned) a.rank, (unsigned) streams), dim3(kDownWarps * 32), 0, s, a, rows);
    check("controllers/down");
    bk::launch_pdl(ctrl_up_kernel, dim3((unsigned) (2 * kWidth / 32), (unsigned) streams), dim3(256), (size_t) rows * a.rank * 4, s, a, rows);
    check("controllers/up");
    bk::launch_pdl(ctrl_blend_kernel, dim3((unsigned) rows, (unsigned) streams), dim3(kThreads), 0, s, a);
    check("controllers/blend");
}

void decode_prepare(const AttnPrep& a, int rows, void* stream) {
    if (rows <= 0) return;
    check_rows(rows, "decode_prepare");
    bk::launch_pdl(prepare_kernel, dim3((unsigned) rows, kKvHeads), dim3(128), 0, (cudaStream_t) stream, a);
    check("decode_prepare");
}

void post_attn(const PostAttn& a, int rows, void* stream) {
    if (rows <= 0) return;
    check_rows(rows, "post_attn");
    if (a.n_blocks + (a.boundary ? 1 : 0) + 1 > kMaxMix) throw std::runtime_error("inkling2 post_attn: blocks");
    bk::launch_pdl(post_attn_kernel, dim3((unsigned) rows), dim3(kThreads), 0, (cudaStream_t) stream, a);
    check("post_attn");
}

void moe_finish(const MoeFinish& a, int rows, void* stream) {
    if (rows <= 0) return;
    check_rows(rows, "moe_finish");
    bk::launch_pdl(moe_finish_kernel, dim3((unsigned) rows, kWidth / 256), dim3(256), 0, (cudaStream_t) stream, a);
    check("moe_finish");
}

}  // namespace basalt::models::inkling2
