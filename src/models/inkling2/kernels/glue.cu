// src/models/inkling2/kernels/glue.cu - Inkling 2's fused glue for a decode window: the block-residual mixes with the norm
// and the q8_1 rounding of the products' inputs, the loop controllers (down, up, blend, norm and rounding), the attention
// preparation a key/value head a block, and the attention output's convolution with the residual and the MLP's mix
// (into the experts' activation rows).
//
// A block takes one row (or one row and head, or one stream and row), and no row's arithmetic reads another row's partial
// sums, so a row's bits do not depend on the rows beside it. A window's convolutions read the rows before them in the
// window and the ring for the positions before the window, and each row writes its own ring slot: a window of at most
// kDecodeRows rows touches distinct slots, and none of the kConv - 1 its rows read before it (kDecodeRows + kConv - 1 <=
// kConvRing).
#include "basalt/kernels/expert_rows.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/q8_1_finite.hpp"
#include "basalt/models/inkling2/kernels.hpp"
#include "prep.cuh"

#include <cuda_bf16.h>
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

// The warp's totals of the 12 values each lane holds, each summed by warp_sum's tree (lanes 16, 8, 4, 2, then 1 apart
// added pairwise): at each step a lane adds, from its partner, only the values its side of the pair goes on with, so the
// warp shuffles 13 times for the 12 totals. Every lane ends with the total of value 6 b4 + 3 b3 + (b2 ? 2 : b1), b the bits of
// its lane index; every lane holding a value holds warp_sum's bits of it.
__device__ __forceinline__ float warp_totals12(const float (&p)[12], int lane) {
    const bool b4 = lane & 16, b3 = lane & 8, b2 = lane & 4, b1 = lane & 2;
    float q[6], r[3];
#pragma unroll
    for (int k = 0; k < 6; ++k) q[k] = (b4 ? p[6 + k] : p[k]) + __shfl_xor_sync(0xffffffffu, b4 ? p[k] : p[6 + k], 16);
#pragma unroll
    for (int k = 0; k < 3; ++k) r[k] = (b3 ? q[3 + k] : q[k]) + __shfl_xor_sync(0xffffffffu, b3 ? q[k] : q[3 + k], 8);
    // lanes 4 apart: the side with b2 clear goes on with values 0 and 1 of its three, the other side with value 2
    const float g1 = __shfl_xor_sync(0xffffffffu, b2 ? r[0] : r[2], 4), g2 = __shfl_xor_sync(0xffffffffu, r[1], 4);
    const float t0 = (b2 ? r[2] : r[0]) + g1, t1 = r[1] + g2;
    // lanes 2 apart: with b2 clear each side goes on with value b1 of its two
    const float u = (b2 || !b1 ? t0 : t1) + __shfl_xor_sync(0xffffffffu, b2 || b1 ? t0 : t1, 2);
    return u + __shfl_xor_sync(0xffffffffu, u, 1);
}

// The q8_1 blocks w, w + 24 and w + 48 of a row (warp w: lane l holds value l of each, y[k] for block w + 24 k): each
// block's amax and sum reduced together across the warp as warp_totals12 does (the lanes with b4 set take the maxima,
// exact in any order; the others the sums, by warp_sum's tree), 8 shuffles, then the maxima handed to every lane (3); d =
// amax / 127, the sum stored by the lane that holds it.
__device__ __forceinline__ void q8_store3(const float (&y)[kPer], Q81* row, int w, int lane) {
    static_assert(kPer == 3, "a lane holds one value of each of three blocks");
    const bool b4 = lane & 16, b3 = lane & 8, b2 = lane & 4;
    float a[3];
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const float o = __shfl_xor_sync(0xffffffffu, b4 ? y[k] : fabsf(y[k]), 16);
        a[k] = b4 ? fmaxf(fabsf(y[k]), o) : y[k] + o;
    }
    // lanes 8 apart: the side with b3 clear goes on with blocks 0 and 1, the other side with block 2
    const float g1 = __shfl_xor_sync(0xffffffffu, b3 ? a[0] : a[2], 8), g2 = __shfl_xor_sync(0xffffffffu, a[1], 8);
    const float s0 = b3 ? a[2] : a[0];
    const float t0 = b4 ? fmaxf(s0, g1) : s0 + g1, t1 = b4 ? fmaxf(a[1], g2) : a[1] + g2;
    // lanes 4 apart: with b3 clear each side goes on with block b2 of its two
    const float k3 = b3 || !b2 ? t0 : t1, o3 = __shfl_xor_sync(0xffffffffu, b3 || b2 ? t0 : t1, 4);
    float u = b4 ? fmaxf(k3, o3) : k3 + o3;
    const float o4 = __shfl_xor_sync(0xffffffffu, u, 2);
    u = b4 ? fmaxf(u, o4) : u + o4;
    const float o5 = __shfl_xor_sync(0xffffffffu, u, 1);
    u = b4 ? fmaxf(u, o5) : u + o5;
    // block k's amax at lane 16 + 4 k (k < 2) or 24, its sum at lane 4 k or 8
    const float amax[3] = {__shfl_sync(0xffffffffu, u, 16), __shfl_sync(0xffffffffu, u, 20), __shfl_sync(0xffffffffu, u, 24)};
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const float d = bk::q8_1_finite(amax[k] / 127.0f);
        Q81& b = row[w + kWarpsB * k];
        b.qs[lane] = bk::q8_1_quant(y[k], d, amax[k]);
        if (lane == (k < 2 ? 4 * k : 8)) b.ds = bk::q8_1_ds(d, u);
    }
}

// The block-residual mix of the n states v[i] (thread-held values) scored by the query qv (thread-held values, loaded
// before the kernel's pdl_wait): scores q . s / sqrt(kWidth) over the RMS-normed states (eps 1e-6), softmax, the weighted
// sum of the normed states. A state's sum of squares and score are each warp's total, then the warps' added in warp order
// by warp 0, which takes the norms and the softmax once for the block and hands them over in shared memory (`red`:
// 2 kMaxMix (kWarpsB + 1) floats).
__device__ __forceinline__ void mix_rows(const float (&v)[kMaxMix][kPer], int n, const float (&qv)[kPer], float (&mixed)[kPer],
                                         float* red) {
    static_assert(2 * kMaxMix == 12, "a state's two sums, every state's in one warp_totals12");
    float part[2 * kMaxMix];
#pragma unroll
    for (int i = 0; i < 2 * kMaxMix; ++i) part[i] = 0.f;
#pragma unroll
    for (int i = 0; i < kMaxMix; ++i) {
        if (i >= n) break;
#pragma unroll
        for (int j = 0; j < kPer; ++j) {
            const float x = v[i][j];
            part[2 * i] = fmaf(x, x, part[2 * i]);
            part[2 * i + 1] = fmaf(qv[j], x, part[2 * i + 1]);
        }
    }
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float tot = warp_totals12(part, lane);
    if (!(lane & 1) && (!(lane & 4) || !(lane & 2)))   // one lane of each value
        red[(6 * ((lane >> 4) & 1) + 3 * ((lane >> 3) & 1) + ((lane & 4) ? 2 : (lane >> 1) & 1)) * kWarpsB + w] = tot;
    __syncthreads();
    float* hand = red + 2 * kMaxMix * kWarpsB;   // [kMaxMix] the softmax weights, then [kMaxMix] the norms
    if (w == 0) {
        float s = 0.f;   // lane 2 i: state i's sum of squares, lane 2 i + 1: its score
        if (lane < 2 * kMaxMix)
#pragma unroll
            for (int j = 0; j < kWarpsB; ++j) s += red[lane * kWarpsB + j];
        const float inv_l = rsqrtf(s / (float) kWidth + 1e-6f);
        // lane i (< n) takes state i: its norm and score, then the softmax over the lanes (the maximum is exact in any
        // order; the sum in state order)
        const int i2 = 2 * (lane % kMaxMix);
        const float ii = __shfl_sync(0xffffffffu, inv_l, i2), dot = __shfl_sync(0xffffffffu, s, i2 + 1);
        const bool mine = lane < n;
        const float inv = mine ? ii : 0.f;
        const float sc = mine ? dot * inv * rsqrtf((float) kWidth) : -INFINITY;
        float m = sc;
#pragma unroll
        for (int o = 4; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));   // lanes 0..7: states, then -inf
        m = __shfl_sync(0xffffffffu, m, 0);
        const float e = mine ? expf(sc - m) : 0.f;
        float den = 0.f;
#pragma unroll
        for (int i = 0; i < kMaxMix; ++i) den += __shfl_sync(0xffffffffu, e, i);
        if (lane < kMaxMix) {
            hand[lane] = e / den;
            hand[kMaxMix + lane] = inv;
        }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        float acc = 0.f;
#pragma unroll
        for (int i = 0; i < kMaxMix; ++i)
            if (i < n) acc = fmaf(hand[i], v[i][j] * hand[kMaxMix + i], acc);
        mixed[j] = acc;
    }
}

// Thread t's kPer values of a kWidth vector (t, t + 768, t + 1536): a kernel loads its weights so before its pdl_wait.
__device__ __forceinline__ void load_row(const float* __restrict__ p, float (&x)[kPer]) {
#pragma unroll
    for (int j = 0; j < kPer; ++j) x[j] = p[threadIdx.x + j * kThreads];
}

// RMS norm of the thread-held row x with the thread-held weights w, into q8_1, or the experts' activation rows when `xh`
// is set (f16, or bf16 with `bf16`, at xh_pos), and fp32 when `f32` is set: the sum of squares each warp's, then the
// warps' in warp order. The kernel's last use of `red` (kWarpsB + 1 floats).
__device__ __forceinline__ void norm_out(const float (&x)[kPer], const float (&w)[kPer], float eps, void* q8, void* xh, bool bf16,
                                         float* f32, int r, float* red) {
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < kPer; ++j) ss = fmaf(x[j], x[j], ss);
    ss = warp_sum(ss);
    const int wp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) red[wp] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.f;
#pragma unroll
        for (int j = 0; j < kWarpsB; ++j) s += red[j];
        red[kWarpsB] = s;
    }
    __syncthreads();
    const float inv = rsqrtf(red[kWarpsB] / (float) kWidth + eps);
    float y[kPer];
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        y[j] = x[j] * inv * w[j];
        if (f32) f32[(size_t) r * kWidth + threadIdx.x + j * kThreads] = y[j];
    }
    if (xh) {
#pragma unroll
        for (int j = 0; j < kPer; ++j) {
            const size_t at = (size_t) r * kWidth + bk::xt::xh_pos(threadIdx.x + j * kThreads);
            if (bf16) static_cast<uint16_t*>(xh)[at] = __bfloat16_as_ushort(__float2bfloat16_rn(y[j]));
            else static_cast<uint16_t*>(xh)[at] = __half_as_ushort(__float2half_rn(y[j]));
        }
        return;
    }
    q8_store3(y, reinterpret_cast<Q81*>(q8) + (size_t) r * (kWidth / 32), wp, lane);
}

using prep::conv_in;

// ---- mixes -------------------------------------------------------------------------------------------------------------

// Block (row, output q): the mix scored by query q (or the last state), then its norm into q8_1, or the mix as it is.
// The older key/value cells of a visit's attention into L2 (MixNorm's kv_*), by a block of its own after the rows' that
// exits before the wait (so the rows' blocks reach their wait at once): its threads request 32 KiB pieces of the one or
// two runs (the ring may wrap) of the keys and of the values. Up to kKvAheadCells positions: a longer run does
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
    if (a.kv_k && r == (int) gridDim.x - 1) {   // the read-ahead's block: constant through the window, no wait
        if (q == 0) kv_read_ahead(a);
        return;
    }
    float qv[kPer], nw[kPer];   // the weights, read while the kernel before finishes
    if (a.queries) load_row(a.queries + (size_t) q * kWidth, qv);
    if (!a.mix_out) load_row(a.norm[q], nw);
    // and every state but the last: only the last one may be the kernel before's
    float v[kMaxMix][kPer];
#pragma unroll
    for (int i = 0; i < kMaxMix; ++i) {
        if (i >= n - 1) break;
#pragma unroll
        for (int j = 0; j < kPer; ++j) v[i][j] = a.st.s[i][(size_t) r * kWidth + threadIdx.x + j * kThreads];
    }
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    __shared__ float red[kMaxMix * 2 * (kWarpsB + 1)];
#pragma unroll
    for (int i = 0; i < kMaxMix; ++i)
        if (i == n - 1)
#pragma unroll
            for (int j = 0; j < kPer; ++j) v[i][j] = a.st.s[i][(size_t) r * kWidth + threadIdx.x + j * kThreads];
    float mixed[kPer];
    if (a.queries == nullptr) {   // the last state as it is (one output)
#pragma unroll
        for (int i = 0; i < kMaxMix; ++i)
            if (i == n - 1)
#pragma unroll
                for (int j = 0; j < kPer; ++j) mixed[j] = v[i][j];
    } else {
        mix_rows(v, n, qv, mixed, red);
    }
    if (a.mix_out) {
#pragma unroll
        for (int j = 0; j < kPer; ++j)
            a.mix_out[(size_t) q * a.out_stride + (size_t) r * kWidth + threadIdx.x + j * kThreads] = mixed[j];
        return;
    }
    norm_out(mixed, nw, a.eps, a.q8[q], nullptr, false, q == 0 ? a.norm_out : nullptr, r, red);
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
__global__ void __launch_bounds__(kDownWarps * 32) ctrl_down_kernel(CtrlArgs a, int rows_all) {
    __shared__ float part[kDownWarps][O][RMAX][2];
    const int r0 = (int) blockIdx.z * RMAX;
    const int rows = min(RMAX, rows_all - r0);
    const int o0 = blockIdx.x * O, s = blockIdx.y, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const bool second = w >= kDownWarps / 2;
    const int c0 = (w % (kDownWarps / 2)) * kSlice;   // the slice's first channel in its half
    const float* in = (second ? a.msg[s] : a.dest[s]) + (size_t) r0 * kWidth;
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
        a.scratch[((size_t) s * kBatchRows + r0 + t) * a.rank + o0 + j] =
            dd * rsqrtf(sd / (float) kWidth + 1e-6f) + dm * rsqrtf(sm / (float) kWidth + 1e-6f);
    }
}

// One level of warp_sum's butterfly (mask M) over a lane's L values, keeping half: the lane with bit M set keeps the upper
// half, its partner the lower, each adding the other's copy (a + b is b + a: warp_sum's bits for every value it keeps).
template <int L, int M, int N>
__device__ __forceinline__ void butterfly_halve(float (&v)[N], int lane) {
    static_assert(L % 2 == 0 && L <= N);
    const bool up = lane & M;
#pragma unroll
    for (int i = 0; i < L / 2; ++i) {
        const float lo = v[i], hi = v[i + L / 2];
        const float send = up ? lo : hi, keep = up ? hi : lo;
        v[i] = keep + __shfl_xor_sync(0xffffffffu, send, M);
    }
}

// ctrl_down_kernel's sums for a wide window: block (o0 / O, s, r0 / RMAX) holds its O outputs' weights in shared memory, so
// a window's inputs are read once for O outputs instead of 2 (the inputs, not the dots, bound the wide windows). Each lane,
// warp and output adds the same products in the same order as ctrl_down_kernel: the same bits.
template<int O, int RMAX>
__global__ void __launch_bounds__(kDownWarps * 32) ctrl_down_wide_kernel(CtrlArgs a, int rows_all) {
    extern __shared__ uint32_t sw[];   // [O][kWidth]: output o0 + j's 2 kWidth bf16
    __shared__ float part[kDownWarps][O][RMAX][2];
    const int r0 = (int) blockIdx.z * RMAX;
    const int rows = min(RMAX, rows_all - r0);
    const int o0 = blockIdx.x * O, s = blockIdx.y, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const bool second = w >= kDownWarps / 2;
    const int c0 = (w % (kDownWarps / 2)) * kSlice;
    {   // the weights, before pdl_wait (constant)
        const uint4* src = reinterpret_cast<const uint4*>(a.down[s] + (size_t) o0 * 2 * kWidth);
        uint4* dst = reinterpret_cast<uint4*>(sw);
        for (int i = threadIdx.x; i < O * kWidth / 4; i += kDownWarps * 32) dst[i] = src[i];
    }
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    __syncthreads();
    const float* in = (second ? a.msg[s] : a.dest[s]) + (size_t) r0 * kWidth;
    const uint32_t* wr = sw + ((second ? kWidth : 0) + c0) / 2;
    constexpr int kIn = 9;
    constexpr int kPass = RMAX < 4 ? RMAX : 4;
    static_assert(kSlice / 2 == 32 * kIn, "a lane's pairs are one batch");
    if constexpr (RMAX == 16) {
        if (rows == RMAX) {
            // every row present: the same chains unpredicated, row t's O dots and its square sum at v[t (O + 1)..], then
            // warp_sum's levels 16, 8, 4, 2 each keeping half the values (one row's in the end, row bits 3..0 = lane bits
            // 4..1) and level 1 whole
            constexpr int V = O + 1;
            float v[RMAX * V];
#pragma unroll
            for (int i = 0; i < RMAX * V; ++i) v[i] = 0.f;
#pragma unroll
            for (int h = 0; h < RMAX; h += kPass) {
                float2 x[kIn][kPass];
#pragma unroll
                for (int b = 0; b < kIn; ++b)
#pragma unroll
                    for (int t = 0; t < kPass; ++t) x[b][t] = reinterpret_cast<const float2*>(in + (size_t) (h + t) * kWidth + c0)[lane + 32 * b];
#pragma unroll
                for (int b = 0; b < kIn; ++b) {
#pragma unroll
                    for (int t = 0; t < kPass; ++t)
                        v[(h + t) * V + O] = fmaf(x[b][t].y, x[b][t].y, fmaf(x[b][t].x, x[b][t].x, v[(h + t) * V + O]));
#pragma unroll
                    for (int j = 0; j < O; ++j) {
                        const uint32_t u = wr[(size_t) j * kWidth + lane + 32 * b];
                        const float w0 = __uint_as_float(u << 16), w1 = __uint_as_float(u & 0xffff0000u);
#pragma unroll
                        for (int t = 0; t < kPass; ++t)
                            v[(h + t) * V + j] = fmaf(w1, x[b][t].y, fmaf(w0, x[b][t].x, v[(h + t) * V + j]));
                    }
                }
            }
            butterfly_halve<RMAX * V, 16>(v, lane);
            butterfly_halve<RMAX * V / 2, 8>(v, lane);
            butterfly_halve<RMAX * V / 4, 4>(v, lane);
            butterfly_halve<RMAX * V / 8, 2>(v, lane);
#pragma unroll
            for (int i = 0; i < V; ++i) v[i] += __shfl_xor_sync(0xffffffffu, v[i], 1);
            if (!(lane & 1)) {
                const int t = (lane & 16 ? 8 : 0) + (lane & 8 ? 4 : 0) + (lane & 4 ? 2 : 0) + (lane & 2 ? 1 : 0);
#pragma unroll
                for (int j = 0; j < O; ++j) {
                    part[w][j][t][0] = v[j];
                    part[w][j][t][1] = v[O];
                }
            }
            __syncthreads();
            for (int i = threadIdx.x; i < rows * O; i += kDownWarps * 32) {
                const int t = i % rows, j = i / rows;
                float dd = 0.f, sd = 0.f, dm = 0.f, sm = 0.f;
#pragma unroll
                for (int k = 0; k < kDownWarps / 2; ++k) {
                    dd += part[k][j][t][0];
                    sd += part[k][j][t][1];
                    dm += part[kDownWarps / 2 + k][j][t][0];
                    sm += part[kDownWarps / 2 + k][j][t][1];
                }
                a.scratch[((size_t) s * kBatchRows + r0 + t) * a.rank + o0 + j] =
                    dd * rsqrtf(sd / (float) kWidth + 1e-6f) + dm * rsqrtf(sm / (float) kWidth + 1e-6f);
            }
            return;
        }
    }
    float dot[O][RMAX], sq[RMAX];
#pragma unroll
    for (int t = 0; t < RMAX; ++t) {
        sq[t] = 0.f;
#pragma unroll
        for (int j = 0; j < O; ++j) dot[j][t] = 0.f;
    }
#pragma unroll
    for (int h = 0; h < RMAX; h += kPass) {
        float2 x[kIn][kPass];
#pragma unroll
        for (int b = 0; b < kIn; ++b)
#pragma unroll
            for (int t = 0; t < kPass; ++t)
                if (h + t < rows) x[b][t] = reinterpret_cast<const float2*>(in + (size_t) (h + t) * kWidth + c0)[lane + 32 * b];
#pragma unroll
        for (int b = 0; b < kIn; ++b) {
#pragma unroll
            for (int t = 0; t < kPass; ++t)
                if (h + t < rows) sq[h + t] = fmaf(x[b][t].y, x[b][t].y, fmaf(x[b][t].x, x[b][t].x, sq[h + t]));
#pragma unroll
            for (int j = 0; j < O; ++j) {
                const uint32_t u = wr[(size_t) j * kWidth + lane + 32 * b];
                const float w0 = __uint_as_float(u << 16), w1 = __uint_as_float(u & 0xffff0000u);
#pragma unroll
                for (int t = 0; t < kPass; ++t)
                    if (h + t < rows) dot[j][h + t] = fmaf(w1, x[b][t].y, fmaf(w0, x[b][t].x, dot[j][h + t]));
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
    for (int i = threadIdx.x; i < rows * O; i += kDownWarps * 32) {
        const int t = i % rows, j = i / rows;
        float dd = 0.f, sd = 0.f, dm = 0.f, sm = 0.f;
#pragma unroll
        for (int k = 0; k < kDownWarps / 2; ++k) {
            dd += part[k][j][t][0];
            sd += part[k][j][t][1];
            dm += part[kDownWarps / 2 + k][j][t][0];
            sm += part[kDownWarps / 2 + k][j][t][1];
        }
        a.scratch[((size_t) s * kBatchRows + r0 + t) * a.rank + o0 + j] =
            dd * rsqrtf(sd / (float) kWidth + 1e-6f) + dm * rsqrtf(sm / (float) kWidth + 1e-6f);
    }
}

// gf[s][r][i] = up_s[i] . hid[s][r] for the 2 kWidth rows of up_s: eight lanes a row (rank / 8 values each), 32 rows a
// block.
constexpr int kUpWords = 8;   // a lane's first weight words, read before pdl_wait (all of them up to rank 128)
__global__ void __launch_bounds__(256) ctrl_up_kernel(CtrlArgs a, int rows_all) {
    extern __shared__ float hs[];   // [rows][rank]
    const int r0 = (int) blockIdx.z * kDecodeRows;
    const int rows = min(kDecodeRows, rows_all - r0);
    const int s = blockIdx.y;
    const int row = blockIdx.x * 32 + (threadIdx.x >> 3), part = threadIdx.x & 7;
    const int per = a.rank / 8;
    const uint32_t* urg = reinterpret_cast<const uint32_t*>(a.up[s] + (size_t) row * a.rank + part * per);
    const uint32_t* urf = reinterpret_cast<const uint32_t*>(a.up[s] + (size_t) (row + kWidth) * a.rank + part * per);
    uint32_t ugw[kUpWords], ufw[kUpWords];
#pragma unroll
    for (int k = 0; k < kUpWords; ++k) {
        const bool ok = row < kWidth && k < per / 2;
        ugw[k] = ok ? urg[k] : 0u;
        ufw[k] = ok ? urf[k] : 0u;
    }
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    const float* hid = a.scratch + ((size_t) s * kBatchRows + r0) * a.rank;
    for (int i = threadIdx.x; i < rows * a.rank; i += blockDim.x) hs[i] = hid[i];
    __syncthreads();
    float acc_g[kDecodeRows], acc_f[kDecodeRows];
#pragma unroll
    for (int t = 0; t < kDecodeRows; ++t) {
        acc_g[t] = 0.f;
        acc_f[t] = 0.f;
    }
    if (row < kWidth) {
#pragma unroll
        for (int k = 0; k < kUpWords; ++k) {
            if (k >= per / 2) break;
            const uint32_t ug = ugw[k], uf = ufw[k];
            const float g0 = __uint_as_float(ug << 16), g1 = __uint_as_float(ug & 0xffff0000u);
            const float f0 = __uint_as_float(uf << 16), f1 = __uint_as_float(uf & 0xffff0000u);
            const int kk = part * per + 2 * k;
#pragma unroll
            for (int t = 0; t < kDecodeRows; ++t) {
                if (t >= rows) break;
                const float h0 = hs[t * a.rank + kk], h1 = hs[t * a.rank + kk + 1];
                acc_g[t] = fmaf(g1, h1, fmaf(g0, h0, acc_g[t]));
                acc_f[t] = fmaf(f1, h1, fmaf(f0, h0, acc_f[t]));
            }
        }
        for (int k = kUpWords; k < per / 2; ++k) {
            const uint32_t ug = urg[k], uf = urf[k];
            const float g0 = __uint_as_float(ug << 16), g1 = __uint_as_float(ug & 0xffff0000u);
            const float f0 = __uint_as_float(uf << 16), f1 = __uint_as_float(uf & 0xffff0000u);
            const int kk = part * per + 2 * k;
#pragma unroll
            for (int t = 0; t < kDecodeRows; ++t) {
                if (t >= rows) break;
                const float h0 = hs[t * a.rank + kk], h1 = hs[t * a.rank + kk + 1];
                acc_g[t] = fmaf(g1, h1, fmaf(g0, h0, acc_g[t]));
                acc_f[t] = fmaf(f1, h1, fmaf(f0, h0, acc_f[t]));
            }
        }
    }
    float* yb = a.scratch + (size_t) kMaxStreams * kBatchRows * a.rank;
#pragma unroll
    for (int t = 0; t < kDecodeRows; ++t) {
        if (t >= rows) break;
        float vg = acc_g[t], vf = acc_f[t];
        vg += __shfl_xor_sync(0xffffffffu, vg, 1);
        vf += __shfl_xor_sync(0xffffffffu, vf, 1);
        vg += __shfl_xor_sync(0xffffffffu, vg, 2);
        vf += __shfl_xor_sync(0xffffffffu, vf, 2);
        vg += __shfl_xor_sync(0xffffffffu, vg, 4);
        vf += __shfl_xor_sync(0xffffffffu, vf, 4);
        if (part == 0 && row < kWidth) {
            float g, f;
            asm("tanh.approx.f32 %0, %1;" : "=f"(g) : "f"(vg));
            asm("tanh.approx.f32 %0, %1;" : "=f"(f) : "f"(vf));
            const size_t idx = (size_t) (r0 + t) * kWidth + row;
            yb[((size_t) s * kBatchRows + r0 + t) * kWidth + row] = (1.f - f) * a.dest[s][idx] + g * a.msg[s][idx];
        }
    }
}

// ctrl_up_kernel for a rank known at compile time: a thread holds one of the 2 kWidth rows' weights in registers and
// forms that row's gate and filter for 2 of the block's 16 rows whole: the 8 parts' chains ctrl_up_kernel's eight lanes
// form (part p: weights p rank / 8 .. in order), added in its butterfly's tree (a + b is b + a), so the same bits
// without shuffles; warp w takes rows 2 w and 2 w + 1, lane l output row 32 blockIdx.x + l.
template <int RANK>
__global__ void __launch_bounds__(256) ctrl_up_rows_kernel(CtrlArgs a, int rows_all) {
    constexpr int kParts = 8, kPer = RANK / kParts, kT = 2, kWords = RANK / 2;
    static_assert(RANK % 16 == 0 && kT * 8 == kDecodeRows, "parts of whole words; 8 warps of kT rows");
    __shared__ __align__(16) float hs[kDecodeRows][RANK];
    const int r0 = (int) blockIdx.z * kDecodeRows;
    const int rows = min(kDecodeRows, rows_all - r0);
    const int s = blockIdx.y;
    const int row = blockIdx.x * 32 + (threadIdx.x & 31), t0 = (threadIdx.x >> 5) * kT;
    uint32_t ug[kWords], uf[kWords];
    {
        const uint4* pg = reinterpret_cast<const uint4*>(a.up[s] + (size_t) row * RANK);
        const uint4* pf = reinterpret_cast<const uint4*>(a.up[s] + (size_t) (row + kWidth) * RANK);
#pragma unroll
        for (int i = 0; i < kWords / 4; ++i) {
            const uint4 g = pg[i], f = pf[i];
            ug[4 * i] = g.x, ug[4 * i + 1] = g.y, ug[4 * i + 2] = g.z, ug[4 * i + 3] = g.w;
            uf[4 * i] = f.x, uf[4 * i + 1] = f.y, uf[4 * i + 2] = f.z, uf[4 * i + 3] = f.w;
        }
    }
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    const float* hid = a.scratch + ((size_t) s * kBatchRows + r0) * RANK;
    for (int i = threadIdx.x; i < rows * RANK; i += blockDim.x) hs[i / RANK][i % RANK] = hid[i];
    __syncthreads();
    float pg[kT][kParts], pf[kT][kParts];
#pragma unroll
    for (int t = 0; t < kT; ++t)
#pragma unroll
        for (int p = 0; p < kParts; ++p) pg[t][p] = pf[t][p] = 0.f;
#pragma unroll
    for (int p = 0; p < kParts; ++p)
#pragma unroll
        for (int k = 0; k < kPer / 2; ++k) {
            const int w = p * kPer / 2 + k, kk = 2 * w;
            const float g0 = __uint_as_float(ug[w] << 16), g1 = __uint_as_float(ug[w] & 0xffff0000u);
            const float f0 = __uint_as_float(uf[w] << 16), f1 = __uint_as_float(uf[w] & 0xffff0000u);
#pragma unroll
            for (int t = 0; t < kT; ++t) {
                const float2 h = *reinterpret_cast<const float2*>(&hs[t0 + t][kk]);
                pg[t][p] = fmaf(g1, h.y, fmaf(g0, h.x, pg[t][p]));
                pf[t][p] = fmaf(f1, h.y, fmaf(f0, h.x, pf[t][p]));
            }
        }
    float* yb = a.scratch + (size_t) kMaxStreams * kBatchRows * a.rank;
#pragma unroll
    for (int tt = 0; tt < kT; ++tt) {
        const int t = t0 + tt;
        if (t >= rows) break;
        const float vg = ((pg[tt][0] + pg[tt][1]) + (pg[tt][2] + pg[tt][3])) + ((pg[tt][4] + pg[tt][5]) + (pg[tt][6] + pg[tt][7]));
        const float vf = ((pf[tt][0] + pf[tt][1]) + (pf[tt][2] + pf[tt][3])) + ((pf[tt][4] + pf[tt][5]) + (pf[tt][6] + pf[tt][7]));
        float g, f;
        asm("tanh.approx.f32 %0, %1;" : "=f"(g) : "f"(vg));
        asm("tanh.approx.f32 %0, %1;" : "=f"(f) : "f"(vf));
        const size_t idx = (size_t) (r0 + t) * kWidth + row;
        yb[((size_t) s * kBatchRows + r0 + t) * kWidth + row] = (1.f - f) * a.dest[s][idx] + g * a.msg[s][idx];
    }
}

// The norm into q8_1 (stream 0: or the experts' rows) of stream s and row r's blended controller state.
__global__ void __launch_bounds__(kThreads) ctrl_blend_kernel(CtrlArgs a) {
    const int r = blockIdx.x, s = blockIdx.y;
    float nw[kPer];
    load_row(a.norm[s], nw);
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    __shared__ float red[kWarpsB + 1];
    const float* yb = a.scratch + (size_t) kMaxStreams * kBatchRows * a.rank + ((size_t) s * kBatchRows + r) * kWidth;
    float y[kPer];
    load_row(yb, y);
    norm_out(y, nw, a.eps, a.q8[s], s == 0 ? a.xh : nullptr, a.xh_bf16, s == 0 ? a.norm_out : nullptr, r, red);
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
        // a batch window's row: its conversation's caches, and its rows of the window from its segment's first
        const bk::WindowSegs& at = seg_table(a.seg);
        const int k = bk::seg_of(at, r);
        const bool b = at.n > 0;
        const SegCaches& sc = a.seg.c[k];
        float* ring = w ? (b ? sc.v_ring : a.v_ring) : (b ? sc.k_ring : a.k_ring);
        void* cache = w ? (b ? sc.v : a.v_cache) : (b ? sc.k : a.k_cache);
        float val[4], raw[4], f, scale[2] = {};
        prep::kv_row(a, r, p, g, w, lane, wt, val, raw, f, ring, b ? at.row0[k] : 0, scale);
        const size_t row_bytes = a.fp8 ? kHeadDim : a.int8 ? kKvInt8RowBytes : 2 * kHeadDim;
        void* row = static_cast<uint8_t*>(cache) + ((size_t) (p % a.cap) * kKvHeads + g) * row_bytes;
#pragma unroll
        for (int u = 0; u < 4; ++u) prep::store_kv(row, a.fp8, a.int8, lane * 4 + u, val[u] * f, scale, lane);
        *reinterpret_cast<float4*>(ring + (size_t) (p % kConvRing) * (kKvHeads * kHeadDim) + g * kHeadDim + lane * 4) =
            make_float4(raw[0], raw[1], raw[2], raw[3]);
        return;
    }
    prep::q_heads(a, r, p, g, w, lane, wt, a.q + ((size_t) r * kHeads + g * 3) * kHeadDim, a.r + ((size_t) r * kHeads + g * 3) * kRel,
                  a.sink + (size_t) r * kHeads + g * 3);
}

// ---- after the attention -----------------------------------------------------------------------------------------------

__global__ void __launch_bounds__(kThreads) post_attn_kernel(PostAttn a) {
    float taps[kPer][kConv], qv[kPer], nw[kPer];   // the weights, read while the kernel before finishes
#pragma unroll
    for (int j = 0; j < kPer; ++j)
#pragma unroll
        for (int k = 0; k < kConv; ++k) taps[j][k] = a.taps[(threadIdx.x + j * kThreads) * kConv + (kConv - 1 - k)];
    load_row(a.query, qv);
    if (!a.mix_out) load_row(a.norm, nw);
    // also before the wait, what no kernel since the attention's projections writes (the kernels since have passed their
    // waits before this one starts): the finished blocks, the layer's input, the ring's entries of the positions before
    // the window; every input is read before the first store (the ring is one of them)
    const int r = blockIdx.x, pr = a.pos[r];
    // a batch window's row: its conversation's ring, and its rows of the window from its segment's first
    const bk::WindowSegs& at = seg_table(a.seg);
    const int sk = bk::seg_of(at, r), first = at.n > 0 ? at.row0[sk] : 0;
    float* const ring = at.n > 0 ? a.seg.c[sk].a_ring : a.ring;
    float v[kMaxMix][kPer];
    const int nb = a.n_blocks + (a.boundary ? 1 : 0), n = nb + 1;
#pragma unroll
    for (int i = 0; i < kMaxMix - 1; ++i) {
        if (i >= nb) break;
        const float* s = i < a.n_blocks ? a.blocks[i] : a.hid;
#pragma unroll
        for (int j = 0; j < kPer; ++j) v[i][j] = s[(size_t) r * kWidth + threadIdx.x + j * kThreads];
    }
    float xin[kPer][kConv], hres[kPer];
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        const int c = threadIdx.x + j * kThreads;
#pragma unroll
        for (int k = 1; k < kConv; ++k)
            if (r - k < first) xin[j][k] = prep::conv_ring(ring, kWidth, pr - k, c);
        hres[j] = a.boundary ? 0.f : a.hid[(size_t) r * kWidth + c];
    }
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    __shared__ float red[kMaxMix * 2 * (kWarpsB + 1)];
    // the convolution's rows of the output projection: this row's and the window's rows before it
#pragma unroll
    for (int j = 0; j < kPer; ++j)
#pragma unroll
        for (int k = 0; k < kConv; ++k)
            if (r - k >= first) xin[j][k] = a.o[(size_t) (r - k) * kWidth + threadIdx.x + j * kThreads];
    const int slot = pr % kConvRing;
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
        ring[(size_t) slot * kWidth + c] = xin[j][0];
    }
#pragma unroll
    for (int i = 0; i < kMaxMix; ++i)
        if (i == nb)
#pragma unroll
            for (int j = 0; j < kPer; ++j) v[i][j] = res[j];
    float mixed[kPer];
    mix_rows(v, n, qv, mixed, red);
    if (a.mix_out) {
#pragma unroll
        for (int j = 0; j < kPer; ++j) a.mix_out[(size_t) r * kWidth + threadIdx.x + j * kThreads] = mixed[j];
        return;
    }
    norm_out(mixed, nw, a.eps, nullptr, a.xh, a.xh_bf16, a.norm_out, r, red);
}

void check_rows(int rows, const char* what) {
    if (rows > kBatchRows) throw std::runtime_error(std::string("inkling2 ") + what + ": more rows than a batch window");
}

}  // namespace

void mix_norm(const MixNorm& a, int n_out, int rows, void* stream) {
    if (rows <= 0) return;
    check_rows(rows, "mix_norm");
    if (a.st.n < 1 || a.st.n > kMaxMix || n_out < 1 || (!a.mix_out && n_out > 3))
        throw std::runtime_error("inkling2 mix_norm: states or outputs");
    bk::launch_pdl(mix_norm_kernel, dim3((unsigned) (rows + (a.kv_k ? 1 : 0)), (unsigned) n_out), dim3(kThreads), 0, (cudaStream_t) stream,
                   a);
    check("mix_norm");
}

size_t ctrl_scratch_floats(int rank) {
    return (size_t) kMaxStreams * kBatchRows * ((size_t) rank + 2 * kWidth);
}

void controllers(const CtrlArgs& a, int streams, int rows, void* stream) {
    if (rows <= 0) return;
    check_rows(rows, "controllers");
    if (streams < 1 || streams > kMaxStreams || a.rank <= 0 || a.rank % 16) throw std::runtime_error("inkling2 controllers: streams or rank");
    cudaStream_t s = (cudaStream_t) stream;
    const unsigned z = (unsigned) ((rows + kDecodeRows - 1) / kDecodeRows);
    const int up_rows = std::min(rows, kDecodeRows);
    // 3-4 rows read their inputs once for 4 outputs (they are 8 rows x 18 KB), 5+ rows once for 2 (twice the blocks, half
    // the dots and sums on each one's chain), a lone row keeps rank blocks
    if (rows > kDecodeRows) {   // a batch window
        constexpr int O = 8, RM = kDecodeRows;   // 8 outputs' weights are 72 KB of shared memory
        auto* k = ctrl_down_wide_kernel<O, RM>;
        const size_t sm = (size_t) O * kWidth * 4;
        static thread_local int set_dev = -1;
        int dev = 0;
        cudaGetDevice(&dev);
        if (set_dev != dev) {
            cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm);
            set_dev = dev;
        }
        if (a.rank % O) throw std::runtime_error("inkling2 controllers: rank");
        bk::launch_pdl(k, dim3((unsigned) (a.rank / O), (unsigned) streams, (unsigned) ((rows + RM - 1) / RM)), dim3(kDownWarps * 32), sm, s, a, rows);
    } else if (rows >= 9) bk::launch_pdl(ctrl_down_kernel<2, kDecodeRows>, dim3((unsigned) a.rank / 2, (unsigned) streams, z), dim3(kDownWarps * 32), 0, s, a, rows);
    else if (rows >= 5) bk::launch_pdl(ctrl_down_kernel<2, 8>, dim3((unsigned) a.rank / 2, (unsigned) streams, 1u), dim3(kDownWarps * 32), 0, s, a, rows);
    else if (rows >= 3) bk::launch_pdl(ctrl_down_kernel<4, 4>, dim3((unsigned) a.rank / 4, (unsigned) streams, 1u), dim3(kDownWarps * 32), 0, s, a, rows);
    else bk::launch_pdl(ctrl_down_kernel<1, 2>, dim3((unsigned) a.rank, (unsigned) streams, 1u), dim3(kDownWarps * 32), 0, s, a, rows);
    check("controllers/down");
    if (a.rank == 64) bk::launch_pdl(ctrl_up_rows_kernel<64>, dim3((unsigned) (kWidth / 32), (unsigned) streams, z), dim3(256), 0, s, a, rows);
    else bk::launch_pdl(ctrl_up_kernel, dim3((unsigned) (kWidth / 32), (unsigned) streams, z), dim3(256), (size_t) up_rows * a.rank * 4, s, a, rows);
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

}  // namespace basalt::models::inkling2
