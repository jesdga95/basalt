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

// The block-residual mix of the n states v[i] (thread-held values) scored by NQ queries: scores q . s / sqrt(kWidth) over
// the RMS-normed states (eps 1e-6), softmax, the weighted sum of the normed states.
template <int NQ>
__device__ __forceinline__ void mix_rows(const float (&v)[kMaxMix][kPer], int n, const float* __restrict__ queries,
                                         float (&mixed)[NQ][kPer], float* red) {
    float part[kMaxMix * (NQ + 1)];
#pragma unroll
    for (int i = 0; i < kMaxMix * (NQ + 1); ++i) part[i] = 0.f;
#pragma unroll
    for (int i = 0; i < kMaxMix; ++i) {
        if (i >= n) break;
#pragma unroll
        for (int j = 0; j < kPer; ++j) {
            const int c = threadIdx.x + j * kThreads;
            const float x = v[i][j];
            part[i * (NQ + 1)] = fmaf(x, x, part[i * (NQ + 1)]);
#pragma unroll
            for (int q = 0; q < NQ; ++q) part[i * (NQ + 1) + 1 + q] = fmaf(queries[q * kWidth + c], x, part[i * (NQ + 1) + 1 + q]);
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

// RMS norm of the thread-held row x with weights w, into q8_1 (and fp32 when `f32` is set).
__device__ __forceinline__ void norm_q8(const float (&x)[kPer], const float* __restrict__ w, float eps, void* q8, float* f32,
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
        const float y = x[j] * inv * w[c];
        if (f32) f32[(size_t) r * kWidth + c] = y;
        q8_store(y, row, c);
    }
}

// The convolution input of channel c at position pos[r] - j: a row of the window (stride ld), the ring, or zero before
// position 0.
__device__ __forceinline__ float conv_in(const float* __restrict__ y, int64_t ld, const float* __restrict__ ring, int ring_w,
                                         const int32_t* __restrict__ pos, int r, int j, int c) {
    if (r - j >= 0) return y[(size_t) (r - j) * ld + c];
    const int p = pos[r] - j;
    if (p < 0) return 0.f;
    return ring[(size_t) (p % kConvRing) * ring_w + c];
}

// ---- mixes -------------------------------------------------------------------------------------------------------------

// Block (row, output q): the mix scored by query q (or the last state), then its norm into q8_1, or the mix as it is.
__global__ void __launch_bounds__(kThreads) mix_norm_kernel(MixNorm a) {
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    __shared__ float red[kMaxMix * 2 * (kWarpsB + 1)];
    const int r = blockIdx.x, q = blockIdx.y, n = a.st.n;
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
        mix_rows<1>(v, n, a.queries + (size_t) q * kWidth, mixed, red);
    }
    if (a.mix_out) {
#pragma unroll
        for (int j = 0; j < kPer; ++j)
            a.mix_out[(size_t) q * a.out_stride + (size_t) r * kWidth + threadIdx.x + j * kThreads] = mixed[0][j];
        return;
    }
    norm_q8(mixed[0], a.norm[q], a.eps, a.q8[q], q == 0 ? a.norm_out : nullptr, r, red);
}

// ---- loop controllers --------------------------------------------------------------------------------------------------

// hid[s][r][o] = down_s[o] . [rms(dest_s[r]) | rms(msg_s[r])]: block (o0 / O, s) takes the O outputs o0.., warp w a
// slice of the 2 kWidth inputs (the first half of the warps dest's, the second msg's); each warp's dots and sums of squares
// over its slice, then the halves' sums in warp order, each half's dot scaled by its row's norm. A window's inputs are read
// once for O outputs; each output's sums are added in the same order whatever O is, so every O gives the same bits.
constexpr int kDownWarps = 8;
constexpr int kSlice = 2 * kWidth / kDownWarps;   // 576 inputs a warp
static_assert(kWidth % (kSlice / 2) == 0 && kSlice % 64 == 0, "a slice is whole pairs of every lane, inside one half");
template<int O>
__global__ void __launch_bounds__(kDownWarps * 32) ctrl_down_kernel(CtrlArgs a, int rows) {
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    __shared__ float part[kDownWarps][O][kDecodeRows][2];
    const int o0 = blockIdx.x * O, s = blockIdx.y, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const bool second = w >= kDownWarps / 2;
    const int c0 = (w % (kDownWarps / 2)) * kSlice;   // the slice's first channel in its half
    const float* in = second ? a.msg[s] : a.dest[s];
    const uint32_t* wr = reinterpret_cast<const uint32_t*>(a.down[s] + (size_t) o0 * 2 * kWidth + (second ? kWidth : 0) + c0);
    float dot[O][kDecodeRows], sq[kDecodeRows];
#pragma unroll
    for (int t = 0; t < kDecodeRows; ++t) {
        sq[t] = 0.f;
#pragma unroll
        for (int j = 0; j < O; ++j) dot[j][t] = 0.f;
    }
    // kIn of a lane's input pairs of every row are loaded before their sums, so the loads overlap instead of each waiting
    // for the one before it (a window's 8 rows were 72 loads in a row); the sums are added in the same order as one at a
    // time
    constexpr int kIn = 3;
    static_assert((kSlice / 2) % (32 * kIn) == 0, "a lane's pairs split into whole batches");
    for (int i0 = lane; i0 < kSlice / 2; i0 += 32 * kIn) {
        float2 x[kIn][kDecodeRows];
        float w0[kIn][O], w1[kIn][O];
#pragma unroll
        for (int b = 0; b < kIn; ++b) {
            const int i = i0 + 32 * b;
#pragma unroll
            for (int j = 0; j < O; ++j) {
                const uint32_t u = wr[(size_t) j * kWidth + i];   // output o0 + j: 2 kWidth bf16, kWidth words
                w0[b][j] = __uint_as_float(u << 16);
                w1[b][j] = __uint_as_float(u & 0xffff0000u);
            }
#pragma unroll
            for (int t = 0; t < kDecodeRows; ++t)
                if (t < rows) x[b][t] = reinterpret_cast<const float2*>(in + (size_t) t * kWidth + c0)[i];
        }
#pragma unroll
        for (int b = 0; b < kIn; ++b) {
#pragma unroll
            for (int t = 0; t < kDecodeRows; ++t) {
                if (t < rows) {
#pragma unroll
                    for (int j = 0; j < O; ++j) dot[j][t] = fmaf(w1[b][j], x[b][t].y, fmaf(w0[b][j], x[b][t].x, dot[j][t]));
                    sq[t] = fmaf(x[b][t].y, x[b][t].y, fmaf(x[b][t].x, x[b][t].x, sq[t]));
                }
            }
        }
    }
#pragma unroll
    for (int t = 0; t < kDecodeRows; ++t) {
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
__global__ void __launch_bounds__(256) ctrl_up_kernel(CtrlArgs a, int rows) {
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    extern __shared__ float hs[];   // [rows][rank]
    const int s = blockIdx.y;
    const float* hid = a.scratch + (size_t) s * kDecodeRows * a.rank;
    for (int i = threadIdx.x; i < rows * a.rank; i += blockDim.x) hs[i] = hid[i];
    __syncthreads();
    const int row = blockIdx.x * 32 + (threadIdx.x >> 3), part = threadIdx.x & 7;
    const int per = a.rank / 8;
    float acc[kDecodeRows];
#pragma unroll
    for (int t = 0; t < kDecodeRows; ++t) acc[t] = 0.f;
    if (row < 2 * kWidth) {
        const uint32_t* ur = reinterpret_cast<const uint32_t*>(a.up[s] + (size_t) row * a.rank + part * per);
        for (int k = 0; k < per / 2; ++k) {
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
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    __shared__ float red[kWarpsB + 1];
    const int r = blockIdx.x, s = blockIdx.y;
    const float* gf = a.scratch + (size_t) kMaxStreams * kDecodeRows * a.rank + ((size_t) s * kDecodeRows + r) * 2 * kWidth;
    float y[kPer];
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        const int c = threadIdx.x + j * kThreads;
        const float g = tanhf(gf[c]), f = tanhf(gf[kWidth + c]);
        y[j] = (1.f - f) * a.dest[s][(size_t) r * kWidth + c] + g * a.msg[s][(size_t) r * kWidth + c];
    }
    norm_q8(y, a.norm[s], a.eps, a.q8[s], s == 0 ? a.norm_out : nullptr, r, red);
}

// ---- attention preparation ---------------------------------------------------------------------------------------------

__device__ __forceinline__ float tau_of(int p, float floor_, float alpha) {
    return 1.f + alpha * logf(fmaxf((float) (p + 1) / floor_, 1.f));
}

__device__ __forceinline__ void store_kv(void* cache, bool fp8, size_t idx, float v) {
    if (fp8) {
        __nv_fp8_e4m3 q(v);
        static_cast<uint8_t*>(cache)[idx] = *reinterpret_cast<uint8_t*>(&q);
    } else {
        static_cast<__half*>(cache)[idx] = __float2half_rn(v);
    }
}

// Block (row, key/value head g), four warps: warp 0 the head's keys (convolved, clipped, cached; the raw inputs into the
// ring), warp 1 its values (normed), warps 2 and 3 its three query heads, their relative states and sinks.
__global__ void __launch_bounds__(128) prepare_kernel(AttnPrep a) {
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    const int r = blockIdx.x, g = blockIdx.y, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int p = a.pos[r];
    if (w < 2) {
        const float* yall = w ? a.yv : a.yk;
        const int64_t ld = w ? a.ldv : a.ldk;
        const float* y = yall + (size_t) r * ld;
        const float* taps = w ? a.v_conv : a.k_conv;
        float* ring = w ? a.v_ring : a.k_ring;
        const float inv_k = a.full ? (1.f / 3.f) : rsqrtf((float) kConv + 1.f);
        float val[4], raw[4], ss = 0.f;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int c = g * kHeadDim + lane * 4 + u;
            float acc = 0.f;
            raw[u] = 0.f;
#pragma unroll
            for (int j = 0; j < kConv; ++j) {
                const float x = conv_in(yall, ld, ring, kKvHeads * kHeadDim, a.pos, r, j, c);
                if (j == 0) raw[u] = x;
                float t = taps[c * kConv + (kConv - 1 - j)];
                if (a.full) t += y[kOffTaps + g * kConv + j];
                acc = fmaf(t, x, acc);
            }
            val[u] = (acc + raw[u]) * inv_k;
            ss = fmaf(val[u], val[u], ss);
        }
        ss = warp_sum(ss);
        const float f = w == 0 ? fminf(a.qk_clip / (sqrtf(ss) + 1e-6f), 1.f) : rsqrtf(ss / (float) kHeadDim + a.eps);
        void* cache = w ? a.v_cache : a.k_cache;
        const size_t cell = ((size_t) (p % a.cap) * kKvHeads + g) * kHeadDim + lane * 4;
#pragma unroll
        for (int u = 0; u < 4; ++u) store_kv(cache, a.fp8, cell + u, val[u] * f);
        *reinterpret_cast<float4*>(ring + (size_t) (p % kConvRing) * (kKvHeads * kHeadDim) + g * kHeadDim + lane * 4) =
            make_float4(raw[0], raw[1], raw[2], raw[3]);
        return;
    }
    const float* yq = a.yq + (size_t) r * a.ldq;
    const float tau = a.full ? tau_of(p, a.log_floor, a.log_alpha) : 1.f;
    // warp 2: heads 3g and 3g + 1; warp 3: head 3g + 2, then the three relative states and sinks
    const int h0 = g * 3 + (w == 2 ? 0 : 2), h1 = w == 2 ? h0 + 2 : h0 + 1;
    for (int h = h0; h < h1; ++h) {
        float q[4], ss = 0.f;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int i = lane * 4 + u;
            q[u] = yq[kOffQ + h * kHeadDim + i] * a.q_scale[h * kHeadDim + i];
            ss = fmaf(q[u], q[u], ss);
        }
        ss = warp_sum(ss);
        const float f = fminf(a.qk_clip / (sqrtf(ss) + 1e-6f), 1.f) * tau;
        *reinterpret_cast<float4*>(a.q + ((size_t) r * kHeads + h) * kHeadDim + lane * 4) =
            make_float4(q[0] * f, q[1] * f, q[2] * f, q[3] * f);
    }
    if (w == 3) {
        for (int h = g * 3; h < g * 3 + 3; ++h) {
            const float rv = lane < kRel ? yq[kOffR + h * kRel + lane] * a.r_scale[h * kRel + lane] : 0.f;
            const float rs = warp_sum(rv * rv);
            const float fr = fminf(a.rel_clip / (sqrtf(rs) + 1e-6f), 1.f) * tau;
            if (lane < kRel) a.r[((size_t) r * kHeads + h) * kRel + lane] = rv * fr;
            if (lane == 0) a.sink[(size_t) r * kHeads + h] = yq[kOffSink + h];
        }
    }
}

// ---- after the attention -----------------------------------------------------------------------------------------------

__global__ void __launch_bounds__(kThreads) post_attn_kernel(PostAttn a) {
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
    // the convolution of the output projection's rows, the residual, the ring slot of this row
    float res[kPer];
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        const int c = threadIdx.x + j * kThreads;
        float acc = 0.f, x0 = 0.f;
#pragma unroll
        for (int k = 0; k < kConv; ++k) {
            const float x = conv_in(a.o, kWidth, a.ring, kWidth, a.pos, r, k, c);
            if (k == 0) x0 = x;
            acc = fmaf(a.taps[c * kConv + (kConv - 1 - k)], x, acc);
        }
        const float conv = __fmul_rn(acc + x0, rsqrtf((float) kConv + 1.f));
        res[j] = a.boundary ? conv : conv + a.hid[(size_t) r * kWidth + c];
        a.resid[(size_t) r * kWidth + c] = res[j];
        a.ring[(size_t) (a.pos[r] % kConvRing) * kWidth + c] = x0;
    }
#pragma unroll
    for (int i = 0; i < kMaxMix; ++i)
        if (i == nb)
#pragma unroll
            for (int j = 0; j < kPer; ++j) v[i][j] = res[j];
    float mixed[1][kPer];
    mix_rows<1>(v, n, a.query, mixed, red);
    if (a.mix_out) {
#pragma unroll
        for (int j = 0; j < kPer; ++j) a.mix_out[(size_t) r * kWidth + threadIdx.x + j * kThreads] = mixed[0][j];
        return;
    }
    norm_q8(mixed[0], a.norm, a.eps, a.q8, a.norm_out, r, red);
}

// ---- experts -----------------------------------------------------------------------------------------------------------

// Block (row, 256 channels), a thread a channel: the experts' weighted sums of this row and the rows before it in the
// window that its convolution reads, the ring for the positions before the window, plus the residual; the ring takes this
// row's sum.
__global__ void __launch_bounds__(256) moe_finish_kernel(MoeFinish a) {
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    const int r = blockIdx.x;
    {
        const int c = blockIdx.y * 256 + threadIdx.x;
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
            acc = fmaf(a.taps[c * kConv + (kConv - 1 - j)], x, acc);
        }
        const size_t i = (size_t) r * kWidth + c;
        a.out[i] = __fmul_rn(acc + x0, rsqrtf((float) kConv + 1.f)) + a.resid[i];
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
    if (rows >= 3) bk::launch_pdl(ctrl_down_kernel<4>, dim3((unsigned) a.rank / 4, (unsigned) streams), dim3(kDownWarps * 32), 0, s, a, rows);
    else bk::launch_pdl(ctrl_down_kernel<1>, dim3((unsigned) a.rank, (unsigned) streams), dim3(kDownWarps * 32), 0, s, a, rows);
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
