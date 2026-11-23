// src/models/inkling2/kernels/ops.cu - Inkling 2's block-residual mixes, loop-controller glue, attention preparation
// (short convolutions, clips, cache writes), output gate, post-sublayer convolutions, expert plan and sum, and softcap.
#include "basalt/kernels/doorbell.cuh"
#include "basalt/kernels/pdl.hpp"
#include "basalt/models/inkling2/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>

namespace basalt::models::inkling2 {
namespace {

namespace bk = basalt::kernels;

constexpr int kThreads = 256;
constexpr int kPerThread = kWidth / kThreads;   // 9
static_assert(kWidth % kThreads == 0, "the width splits evenly over a block");

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("inkling2 ") + what + ": " + cudaGetErrorString(e));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}
__device__ __forceinline__ void int8_scales(const float (&v)[4], float f, int lane, float (&scale)[2]) {
    float a = 0.f;
#pragma unroll
    for (int u = 0; u < 4; ++u) a = fmaxf(a, fabsf(v[u] * f));
#pragma unroll
    for (int m = 8; m; m >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, m));
    scale[lane >> 4] = __half2float(__float2half_rn(a / 127.f));
}

// Sums of N values per thread over a block of kThreads, in a fixed order; every thread gets the N results.
template <int N>
__device__ __forceinline__ void block_sums(float (&v)[N], float* red) {
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] = warp_sum(v[i]);
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0)
#pragma unroll
        for (int i = 0; i < N; ++i) red[i * (kThreads / 32) + w] = v[i];
    __syncthreads();
#pragma unroll
    for (int i = 0; i < N; ++i) {
        float s = 0.f;
#pragma unroll
        for (int j = 0; j < kThreads / 32; ++j) s += red[i * (kThreads / 32) + j];
        v[i] = s;
    }
    __syncthreads();
}

__device__ __forceinline__ float sigmoidf_(float v) { return 1.f / (1.f + expf(-v)); }

// ---- block residuals ---------------------------------------------------------------------------------------------------

constexpr int kMixThreads = 192;
constexpr int kMixVecs = kWidth / (4 * kMixThreads);   // 3 float4s (12 floats) per thread
static_assert(kWidth == 4 * kMixThreads * kMixVecs, "kWidth splits into float4s over kMixThreads");

template <int N>
__device__ __forceinline__ void block_sums_mix(float (&v)[N], float* red) {
    constexpr int kW = kMixThreads / 32;
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] = warp_sum(v[i]);
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0)
#pragma unroll
        for (int i = 0; i < N; ++i) red[i * kW + w] = v[i];
    __syncthreads();
#pragma unroll
    for (int i = 0; i < N; ++i) {
        float s = 0.f;
#pragma unroll
        for (int j = 0; j < kW; ++j) s += red[i * kW + j];
        v[i] = s;
    }
    __syncthreads();
}

template <int NQ>
__global__ void __launch_bounds__(kMixThreads) attnres_mix_kernel(MixStates st, const float* __restrict__ queries, float eps,
                                                                  float* __restrict__ out, int64_t out_stride) {
    __shared__ float red[kMaxMix * (NQ + 1) * (kMixThreads / 32)];
    const int r = blockIdx.x, n = st.n;
    float4 qv[NQ][kMixVecs];
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
        const float4* qp = reinterpret_cast<const float4*>(queries + (size_t) q * kWidth);
#pragma unroll
        for (int j = 0; j < kMixVecs; ++j) qv[q][j] = qp[threadIdx.x + j * kMixThreads];
    }
    float4 v[kMaxMix][kMixVecs];
    float part[kMaxMix * (NQ + 1)];
#pragma unroll
    for (int i = 0; i < kMaxMix * (NQ + 1); ++i) part[i] = 0.f;
#pragma unroll
    for (int i = 0; i < kMaxMix; ++i) {
        if (i >= n) break;
        const float4* sp = reinterpret_cast<const float4*>(st.s[i] + (size_t) r * kWidth);
#pragma unroll
        for (int j = 0; j < kMixVecs; ++j) {
            const float4 x = sp[threadIdx.x + j * kMixThreads];
            v[i][j] = x;
            float s = part[i * (NQ + 1)];
            s = fmaf(x.x, x.x, s);
            s = fmaf(x.y, x.y, s);
            s = fmaf(x.z, x.z, s);
            s = fmaf(x.w, x.w, s);
            part[i * (NQ + 1)] = s;
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                float d = part[i * (NQ + 1) + 1 + q];
                d = fmaf(qv[q][j].x, x.x, d);
                d = fmaf(qv[q][j].y, x.y, d);
                d = fmaf(qv[q][j].z, x.z, d);
                d = fmaf(qv[q][j].w, x.w, d);
                part[i * (NQ + 1) + 1 + q] = d;
            }
        }
    }
    block_sums_mix<kMaxMix * (NQ + 1)>(part, red);
    float inv[kMaxMix];
#pragma unroll
    for (int i = 0; i < kMaxMix; ++i) inv[i] = i < n ? rsqrtf(part[i * (NQ + 1)] / (float) kWidth + eps) : 0.f;
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
        const float inv_den = 1.f / den;
        float w[kMaxMix];
#pragma unroll
        for (int i = 0; i < kMaxMix; ++i) w[i] = i < n ? (sc[i] * inv_den) * inv[i] : 0.f;
        float4* o = reinterpret_cast<float4*>(out + (size_t) q * out_stride + (size_t) r * kWidth);
#pragma unroll
        for (int j = 0; j < kMixVecs; ++j) {
            float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
            for (int i = 0; i < kMaxMix; ++i)
                if (i < n) {
                    acc.x = fmaf(w[i], v[i][j].x, acc.x);
                    acc.y = fmaf(w[i], v[i][j].y, acc.y);
                    acc.z = fmaf(w[i], v[i][j].z, acc.z);
                    acc.w = fmaf(w[i], v[i][j].w, acc.w);
                }
            o[threadIdx.x + j * kMixThreads] = acc;
        }
    }
}

// ---- loop controllers --------------------------------------------------------------------------------------------------

__global__ void __launch_bounds__(kThreads) ctrl_pair_norm_kernel(const float* __restrict__ dest, const float* __restrict__ msg,
                                                                  float* __restrict__ pair) {
    __shared__ float red[2 * (kThreads / 32)];
    const int r = blockIdx.x;
    float a[kPerThread], b[kPerThread], ss[2] = {0.f, 0.f};
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
        const int c = threadIdx.x + j * kThreads;
        a[j] = dest[(size_t) r * kWidth + c];
        b[j] = msg[(size_t) r * kWidth + c];
        ss[0] = fmaf(a[j], a[j], ss[0]);
        ss[1] = fmaf(b[j], b[j], ss[1]);
    }
    block_sums<2>(ss, red);
    const float ia = rsqrtf(ss[0] / (float) kWidth + 1e-6f), ib = rsqrtf(ss[1] / (float) kWidth + 1e-6f);
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
        const int c = threadIdx.x + j * kThreads;
        pair[(size_t) r * 2 * kWidth + c] = a[j] * ia;
        pair[(size_t) r * 2 * kWidth + kWidth + c] = b[j] * ib;
    }
}

__global__ void ctrl_blend_kernel(const float* __restrict__ dest, const float* __restrict__ msg, const float* __restrict__ gf,
                                  int n, float* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int r = i / kWidth, c = i - r * kWidth;
    const float g = tanhf(gf[(size_t) r * 2 * kWidth + c]), f = tanhf(gf[(size_t) r * 2 * kWidth + kWidth + c]);
    out[i] = (1.f - f) * dest[i] + g * msg[i];
}

// ---- attention preparation ---------------------------------------------------------------------------------------------

__device__ __forceinline__ float tau_of(int p, float floor_, float alpha) {
    return 1.f + alpha * logf(fmaxf((float) (p + 1) / floor_, 1.f));
}

// Four contiguous convolution input channels c..c+3 at position pos[r] - j: window row, ring, or zero before 0.
__device__ __forceinline__ float4 conv_in4(const float* __restrict__ y, int ld, const float* __restrict__ ring,
                                           int ring_w, const int32_t* __restrict__ pos, int r, int j, int c) {
    if (r - j >= 0) return *reinterpret_cast<const float4*>(y + (size_t) (r - j) * ld + c);
    const int p = pos[r] - j;
    if (p < 0) return make_float4(0.f, 0.f, 0.f, 0.f);
    return *reinterpret_cast<const float4*>(ring + (size_t) (p % kConvRing) * ring_w + c);
}

__device__ __forceinline__ void store_kv4(void* cache, bool fp8, size_t idx, float v0, float v1, float v2, float v3) {
    if (fp8) {
        __nv_fp8_e4m3 q0(v0), q1(v1), q2(v2), q3(v3);
        const uint32_t packed = (uint32_t) *reinterpret_cast<uint8_t*>(&q0) |
                                ((uint32_t) *reinterpret_cast<uint8_t*>(&q1) << 8) |
                                ((uint32_t) *reinterpret_cast<uint8_t*>(&q2) << 16) |
                                ((uint32_t) *reinterpret_cast<uint8_t*>(&q3) << 24);
        *reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(cache) + idx) = packed;
    } else {
        const __half2 h0 = __floats2half2_rn(v0, v1), h1 = __floats2half2_rn(v2, v3);
        uint2 u;
        u.x = *reinterpret_cast<const uint32_t*>(&h0);
        u.y = *reinterpret_cast<const uint32_t*>(&h1);
        *reinterpret_cast<uint2*>(static_cast<__half*>(cache) + idx) = u;
    }
}

__device__ __forceinline__ void store_kv(void* row, bool fp8, bool int8, size_t idx, float v, const float (&scale)[2], int lane) {
    if (int8) {
        const int group = (int) idx / 64;
        const float s = scale[group];
        const int q = s > 0.f ? max(-127, min(127, __float2int_rn(v / s))) : 0;
        static_cast<int8_t*>(row)[idx] = (int8_t) q;
        if (lane == group * 16)
            reinterpret_cast<__half*>(static_cast<uint8_t*>(row) + kHeadDim)[group] = __float2half_rn(s);
        if (idx == kHeadDim - 1)   // the last channel writes the row padding as zeros
            for (int b = kHeadDim + 4; b < kKvInt8RowBytes; b += 4) *reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(row) + b) = 0u;
    } else if (fp8) {
        __nv_fp8_e4m3 q(v);
        static_cast<uint8_t*>(row)[idx] = *reinterpret_cast<uint8_t*>(&q);
    } else {
        static_cast<__half*>(row)[idx] = __float2half_rn(v);
    }
}

// A block per row: warp w holds kv head w's 128 key and value channels (4 a lane), then three query heads and their
// relative states.
__global__ void __launch_bounds__(kThreads) attn_prepare_kernel(AttnPrep a) {
    const int r = blockIdx.x, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float* yq = a.yq + (size_t) r * a.ldq;
    const int p = a.pos[r];
    const float inv_k = a.full ? (1.f / 3.f) : rsqrtf((float) kConv + 1.f);
    const float tau = a.full ? tau_of(p, a.log_floor, a.log_alpha) : 1.f;
    // keys and values
#pragma unroll
    for (int which = 0; which < 2; ++which) {
        const float* yall = which ? a.yv : a.yk;
        const int64_t ld = which ? a.ldv : a.ldk;
        const float* y = yall + (size_t) r * ld;
        const float* taps = which ? a.v_conv : a.k_conv;
        const float* ring = which ? a.v_ring : a.k_ring;
        const int c0 = w * kHeadDim + lane * 4;
        const float4 tp0 = *reinterpret_cast<const float4*>(taps + (c0 + 0) * kConv);
        const float4 tp1 = *reinterpret_cast<const float4*>(taps + (c0 + 1) * kConv);
        const float4 tp2 = *reinterpret_cast<const float4*>(taps + (c0 + 2) * kConv);
        const float4 tp3 = *reinterpret_cast<const float4*>(taps + (c0 + 3) * kConv);
        const float4 dt = a.full ? *reinterpret_cast<const float4*>(y + kOffTaps + w * kConv)
                                 : make_float4(0.f, 0.f, 0.f, 0.f);
        const float4 x0 = conv_in4(yall, (int) ld, ring, kKvHeads * kHeadDim, a.pos, r, 0, c0);
        const float4 x1 = conv_in4(yall, (int) ld, ring, kKvHeads * kHeadDim, a.pos, r, 1, c0);
        const float4 x2 = conv_in4(yall, (int) ld, ring, kKvHeads * kHeadDim, a.pos, r, 2, c0);
        const float4 x3 = conv_in4(yall, (int) ld, ring, kKvHeads * kHeadDim, a.pos, r, 3, c0);
        auto eval = [&](const float4& tp, float a0, float a1, float a2, float a3) {
            float acc = 0.f;
            acc = fmaf(a.full ? tp.w + dt.x : tp.w, a0, acc);
            acc = fmaf(a.full ? tp.z + dt.y : tp.z, a1, acc);
            acc = fmaf(a.full ? tp.y + dt.z : tp.y, a2, acc);
            acc = fmaf(a.full ? tp.x + dt.w : tp.x, a3, acc);
            return (acc + a0) * inv_k;
        };
        const float val0 = eval(tp0, x0.x, x1.x, x2.x, x3.x);
        const float val1 = eval(tp1, x0.y, x1.y, x2.y, x3.y);
        const float val2 = eval(tp2, x0.z, x1.z, x2.z, x3.z);
        const float val3 = eval(tp3, x0.w, x1.w, x2.w, x3.w);
        float ss = fmaf(val0, val0, 0.f);
        ss = fmaf(val1, val1, ss);
        ss = fmaf(val2, val2, ss);
        ss = fmaf(val3, val3, ss);
        ss = warp_sum(ss);
        const float f = which == 0 ? fminf(a.qk_clip / (sqrtf(ss) + 1e-6f), 1.f) : rsqrtf(ss / (float) kHeadDim + a.eps);
        float scale[2] = {};
        const int64_t cell = p % a.cap;
        const size_t row_bytes = a.fp8 ? kHeadDim : a.int8 ? kKvInt8RowBytes : 2 * kHeadDim;
        void* cache = which ? a.v_cache : a.k_cache;
        if (a.int8) {
            float val[4] = {val0, val1, val2, val3};
            int8_scales(val, f, lane, scale);
            void* row = static_cast<uint8_t*>(cache) + ((size_t) cell * kKvHeads + w) * row_bytes;
#pragma unroll
            for (int u = 0; u < 4; ++u) store_kv(row, false, true, lane * 4 + u, val[u] * f, scale, lane);
        } else {
            store_kv4(cache, a.fp8, ((size_t) cell * kKvHeads + w) * kHeadDim + lane * 4,
                      val0 * f, val1 * f, val2 * f, val3 * f);
        }
    }
    // queries: warp w takes heads 3w .. 3w + 2
    for (int h = w * 3; h < w * 3 + 3; ++h) {
        const int i = lane * 4;
        const float4 yv = *reinterpret_cast<const float4*>(yq + kOffQ + h * kHeadDim + i);
        const float4 sv = *reinterpret_cast<const float4*>(a.q_scale + h * kHeadDim + i);
        const float q0 = yv.x * sv.x, q1 = yv.y * sv.y, q2 = yv.z * sv.z, q3 = yv.w * sv.w;
        float ss = fmaf(q0, q0, 0.f);
        ss = fmaf(q1, q1, ss);
        ss = fmaf(q2, q2, ss);
        ss = fmaf(q3, q3, ss);
        ss = warp_sum(ss);
        const float f = fminf(a.qk_clip / (sqrtf(ss) + 1e-6f), 1.f) * tau;
        *reinterpret_cast<float4*>(a.q + ((size_t) r * kHeads + h) * kHeadDim + i) =
            make_float4(q0 * f, q1 * f, q2 * f, q3 * f);
        // the relative state: 16 values, lanes 0..15
        const float rv = lane < kRel ? yq[kOffR + h * kRel + lane] * a.r_scale[h * kRel + lane] : 0.f;
        const float rs = warp_sum(rv * rv);
        const float fr = fminf(a.rel_clip / (sqrtf(rs) + 1e-6f), 1.f) * tau;
        if (lane < kRel) a.r[((size_t) r * kHeads + h) * kRel + lane] = rv * fr;
        if (lane == 0) a.sink[(size_t) r * kHeads + h] = yq[kOffSink + h];
    }
}

// The rings take the inputs of the window's last positions (after every row read the positions before the window).
__global__ void ring_store_kernel(const float* __restrict__ x, int ld, int off, int width, const int32_t* __restrict__ pos,
                                  int rows, float* __restrict__ ring) {
    const int first = rows > kConvRing ? rows - kConvRing : 0;
    const int r = first + blockIdx.y;
    if (r >= rows) return;
    float* dst = ring + (size_t) (pos[r] % kConvRing) * width;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < width; c += gridDim.x * blockDim.x)
        dst[c] = x[(size_t) r * ld + off + c];
}

void ring_store(const float* x, int ld, int off, int width, const int32_t* pos, int rows, float* ring, cudaStream_t s) {
    const int n = rows < kConvRing ? rows : kConvRing;
    ring_store_kernel<<<dim3((unsigned) ((width + 255) / 256), (unsigned) n), 256, 0, s>>>(x, ld, off, width, pos, rows, ring);
}

__global__ void attn_gate_kernel(const float* __restrict__ attn, const float* __restrict__ yq, int64_t yq_ld, int n4,
                                 float* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    constexpr int kOut4 = (kHeads * kHeadDim) / 4;
    const int r = i / kOut4, c4 = i - r * kOut4;
    const float4 a = reinterpret_cast<const float4*>(attn)[i];
    const float4 g = *reinterpret_cast<const float4*>(yq + (size_t) r * yq_ld + kOffGate + c4 * 4);
    reinterpret_cast<float4*>(out)[i] =
        make_float4(a.x * sigmoidf_(g.x), a.y * sigmoidf_(g.y), a.z * sigmoidf_(g.z), a.w * sigmoidf_(g.w));
}

constexpr int kPostConvThreads = 192;
constexpr int kPostConvTile = 32;
static_assert(kWidth % (4 * kPostConvThreads) == 0, "kWidth splits into float4 blocks for post_conv");

__global__ void __launch_bounds__(kPostConvThreads)
    post_conv_kernel(const float* __restrict__ x, const int32_t* __restrict__ pos, const float* __restrict__ taps,
                     const float* __restrict__ ring, const float* __restrict__ add, int rows, float* __restrict__ out) {
    const int c = (blockIdx.x * kPostConvThreads + threadIdx.x) * 4;
    const int r0 = blockIdx.y * kPostConvTile;
    const int r1 = min(r0 + kPostConvTile, rows);
    const float inv = rsqrtf((float) kConv + 1.f);
    const float4 tp0 = *reinterpret_cast<const float4*>(taps + (c + 0) * kConv);
    const float4 tp1 = *reinterpret_cast<const float4*>(taps + (c + 1) * kConv);
    const float4 tp2 = *reinterpret_cast<const float4*>(taps + (c + 2) * kConv);
    const float4 tp3 = *reinterpret_cast<const float4*>(taps + (c + 3) * kConv);

    auto emit = [&](int r, const float4& v0, const float4& v1, const float4& v2, const float4& v3) {
        auto one = [&](const float4& tp, float a0, float a1, float a2, float a3) {
            float acc = 0.f;
            acc = fmaf(tp.w, a0, acc);
            acc = fmaf(tp.z, a1, acc);
            acc = fmaf(tp.y, a2, acc);
            acc = fmaf(tp.x, a3, acc);
            return __fmul_rn(acc + a0, inv);
        };
        float4 res = make_float4(one(tp0, v0.x, v1.x, v2.x, v3.x), one(tp1, v0.y, v1.y, v2.y, v3.y),
                                 one(tp2, v0.z, v1.z, v2.z, v3.z), one(tp3, v0.w, v1.w, v2.w, v3.w));
        const size_t off = (size_t) r * kWidth + c;
        if (add) {
            const float4 ad = *reinterpret_cast<const float4*>(add + off);
            res.x += ad.x;
            res.y += ad.y;
            res.z += ad.z;
            res.w += ad.w;
        }
        *reinterpret_cast<float4*>(out + off) = res;
    };

    int r = r0;
    for (; r < min(r1, kConv - 1); ++r) {
        const float4 v0 = conv_in4(x, kWidth, ring, kWidth, pos, r, 0, c);
        const float4 v1 = conv_in4(x, kWidth, ring, kWidth, pos, r, 1, c);
        const float4 v2 = conv_in4(x, kWidth, ring, kWidth, pos, r, 2, c);
        const float4 v3 = conv_in4(x, kWidth, ring, kWidth, pos, r, 3, c);
        emit(r, v0, v1, v2, v3);
    }
    if (r >= r1) return;
    float4 v1 = *reinterpret_cast<const float4*>(x + (size_t) (r - 1) * kWidth + c);
    float4 v2 = *reinterpret_cast<const float4*>(x + (size_t) (r - 2) * kWidth + c);
    float4 v3 = *reinterpret_cast<const float4*>(x + (size_t) (r - 3) * kWidth + c);
    for (; r < r1; ++r) {
        const float4 v0 = *reinterpret_cast<const float4*>(x + (size_t) r * kWidth + c);
        emit(r, v0, v1, v2, v3);
        v3 = v2;
        v2 = v1;
        v1 = v0;
    }
}

// ---- experts -----------------------------------------------------------------------------------------------------------

// One block of kPlanThreads, a thread an expert: the (row, slot) pairs counted by expert (the experts on the card); every
// warp scans its experts' counts at once and adds the warps' before it, giving each used expert its group and first entry;
// each pair then takes its place in its group from the pairs before it that chose the same expert (the rows' order). The
// blob table (constant) is staged in shared memory before pdl_wait, the pairs' choices after it.
constexpr int kPlanThreads = 256;
__global__ void __launch_bounds__(kPlanThreads) moe_plan_kernel(const int32_t* __restrict__ ids, int n, int n_expert,
                                                                 const unsigned long long* __restrict__ blobs, unsigned long long* __restrict__ ptr,
                                                                 int32_t* __restrict__ start, int32_t* __restrict__ n_groups,
                                                                 int32_t* __restrict__ dst, int32_t* __restrict__ tok, int k,
                                                                 uint32_t* __restrict__ tally, unsigned long long* __restrict__ rec_blobs,
                                                                 int32_t* __restrict__ rec_n) {
    extern __shared__ unsigned long long smb[];
    unsigned long long* blob = smb;                                // [n_expert]
    unsigned long long* gblob = smb + n_expert;                    // [n_expert]: group g's blob
    int32_t* count = reinterpret_cast<int32_t*>(smb + 2 * n_expert);   // [n_expert]
    int32_t* first = count + n_expert;                             // [n_expert]: the expert's first entry
    int32_t* sid = first + n_expert;                               // [n]: the pairs' experts
    __shared__ int wsum[kPlanThreads / 32][2];                     // each warp's entries and groups
    __shared__ int groups;
    const int e = threadIdx.x, w = e >> 5, lane = e & 31;
    if (e < n_expert) {
        blob[e] = blobs[e];
        count[e] = 0;
    }
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int x = ids[i];
        sid[i] = x;
        if (blob[x]) atomicAdd(&count[x], 1);
    }
    __syncthreads();
    const int c = e < n_expert ? count[e] : 0;
    int s = c, used = c > 0;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int t = __shfl_up_sync(0xffffffffu, s, o), u = __shfl_up_sync(0xffffffffu, used, o);
        if (lane >= o) {
            s += t;
            used += u;
        }
    }
    if (lane == 31) {
        wsum[w][0] = s;
        wsum[w][1] = used;
    }
    __syncthreads();
    int at = 0, g = 0;
    for (int j = 0; j < w; ++j) {
        at += wsum[j][0];
        g += wsum[j][1];
    }
    if (e < n_expert) first[e] = at + s - c;
    if (c > 0) {
        gblob[g + used - 1] = blob[e];
        ptr[g + used - 1] = blob[e];
        start[g + used - 1] = at + s - c;
    }
    if (e == kPlanThreads - 1) {   // the totals
        start[g + used] = at + s;
        start[g + used + 1] = at + s;
        *n_groups = g + used;
        groups = g + used;
    }
    __syncthreads();
    if (rec_blobs && e < kAheadExperts) {   // the window's first groups, for the next window's read-ahead
        const int ng = groups;
        rec_blobs[e] = e < ng ? gblob[e] : 0ull;
        if (e == 0) *rec_n = ng < kAheadExperts ? ng : kAheadExperts;
    }
    if (tally && c)   // a reduction (its result unused), so no thread waits on the load a plain add would make
        atomicAdd(&tally[e], (uint32_t) c);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int x = sid[i];
        if (!blob[x]) continue;
        int before = 0;
        for (int j = 0; j < i; ++j) before += sid[j] == x;
        const int p = first[x] + before;
        dst[p] = i;
        tok[p] = i / k;
    }
}

// One block classifies rows before touching the sequence or mapped memory. A RAM visit reserves a ticket even if
// aborted, so its merge cannot match an earlier visit's completion.
constexpr int kRingThreads = 256;
static_assert(kDecodeRows <= 32, "a visit's rows fit one mask word");
__global__ void __launch_bounds__(kRingThreads) moe_cpu_ring_kernel(const int32_t* __restrict__ ids, int rows, int k,
                                                                    const unsigned long long* __restrict__ blobs,
                                                                    const float* __restrict__ x, int layer, float* x_m,
                                                                    int32_t* ids_m, CpuBell* bell, uint32_t* seq) {
    __shared__ uint32_t ram_rows, stopped;
    const int n = rows * k;
    if (threadIdx.x == 0) ram_rows = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += kRingThreads)
        if (!blobs[ids[i]]) atomicOr(&ram_rows, 1u << (i / k));
    __syncthreads();
    const uint32_t m = ram_rows;
    if (!m) return;
    uint32_t ticket = 0;
    if (threadIdx.x == 0) {
        ticket = *seq + 1;
        if (!ticket) ticket = 1;
        *seq = ticket;
        stopped = bk::doorbell_load_acquire(&bell->abort);
    }
    __syncthreads();
    if (stopped) return;
    for (int i = threadIdx.x; i < n; i += kRingThreads) ids_m[i] = ids[i];
    constexpr int kVecs = kWidth / 4;
    for (int r = 0; r < rows; ++r)
        if (m >> r & 1)
            for (int c = threadIdx.x; c < kVecs; c += kRingThreads)
                reinterpret_cast<float4*>(x_m)[(size_t) r * kVecs + c] = reinterpret_cast<const float4*>(x)[(size_t) r * kVecs + c];
    if (threadIdx.x == 0) bk::doorbell_store_relaxed(&bell->ticket, ticket);
    __threadfence_system();   // every writer's payload before the barrier, so the release below covers it
    __syncthreads();
    if (threadIdx.x == 0) bk::doorbell_store_release(&bell->ring, 1u | (uint32_t) layer << 8 | (uint32_t) rows << 16);
}

constexpr int kMergeThreads = 256;
__global__ void __launch_bounds__(kMergeThreads) moe_cpu_merge_kernel(const int32_t* __restrict__ ids,
                                                                      const unsigned long long* __restrict__ blobs,
                                                                      const float* cpu, float* __restrict__ output,
                                                                      uint32_t* __restrict__ tally, CpuBell* bell,
                                                                      const uint32_t* seq) {
    const int p = blockIdx.x, e = ids[p];
    if (blobs[e]) return;
    if (bell) {   // the pool's rows land in mapped memory while this waits
        __shared__ uint32_t taken;
        if (threadIdx.x == 0) {
            const uint32_t ticket = *seq;
            taken = bk::doorbell_spin(&bell->done, ticket, &bell->abort, 8);
        }
        __syncthreads();
        if (!taken) return;
    }
    for (int c = threadIdx.x; c < kWidth; c += kMergeThreads) output[(size_t) p * kWidth + c] = __ldcv(cpu + (size_t) p * kWidth + c);
    if (threadIdx.x == 0 && tally) atomicAdd(tally + e, 1u);
}

constexpr int kAheadBlocks = 32;
constexpr uint64_t kAheadStep = 128 << 10;
// Block b requests steps b, b + blocks, ... of the blobs' first `bytes` laid end to end, 32 KiB at a time, then reads the
// step's last 16 bytes, so each block keeps one step in flight.
__global__ void __launch_bounds__(32) expert_read_ahead_kernel(const unsigned long long* __restrict__ blobs, const int32_t* __restrict__ n,
                                                               uint64_t bytes) {
    if (threadIdx.x != 0) return;
    const uint64_t total = (uint64_t) *n * bytes;
    uint32_t acc = 0;
    for (uint64_t o0 = (uint64_t) blockIdx.x * kAheadStep; o0 < total; o0 += (uint64_t) gridDim.x * kAheadStep) {
        const uint64_t o1 = min(o0 + kAheadStep, total);
        const char* last = nullptr;
        for (uint64_t o = o0; o < o1;) {
            const uint64_t i = o / bytes, ro = o - i * bytes;
            const uint64_t take = min(min(o1 - o, bytes - ro), (uint64_t) 32768);
            const char* src = (const char*) blobs[i] + ro;
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"((unsigned) take) : "memory");
            last = src + take - 16;
            o += take;
        }
        const uint4 w = __ldcg((const uint4*) last);
        acc ^= w.x ^ w.y ^ w.z ^ w.w;
    }
    if (acc == 0x9e3779b9u && total == 1) asm volatile("" ::"r"(acc));   // keeps the reads
}

// Every block loads its share of the experts' first `bytes`, 16 bytes a thread at a time, four loads in flight.
constexpr int kPrefetchBlocks = 170, kPrefetchThreads = 256, kPrefetchMax = 16;
__global__ void __launch_bounds__(kPrefetchThreads) expert_prefetch_kernel(const int32_t* __restrict__ ids, int n,
                                                                            const unsigned long long* __restrict__ blobs, uint64_t bytes,
                                                                            uint32_t* __restrict__ sink) {
    __shared__ unsigned long long addr[kPrefetchMax];
    __shared__ int nu;
    if (threadIdx.x == 0) {   // the distinct experts
        int m = 0;
        for (int i = 0; i < n; ++i) {
            bool seen = false;
            for (int j = 0; j < i; ++j) seen |= ids[j] == ids[i];
            if (!seen && blobs[ids[i]]) addr[m++] = blobs[ids[i]];   // 0: in RAM, the CPU's
        }
        nu = m;
    }
    __syncthreads();
    const uint64_t n16 = bytes / 16, total = (uint64_t) nu * n16, T = (uint64_t) gridDim.x * blockDim.x;
    uint32_t acc = 0;
    for (uint64_t c = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x; c < total; c += 4 * T) {
        uint4 v[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const uint64_t i = c + u * T;
            v[u] = i < total ? __ldcg(reinterpret_cast<const uint4*>((const char*) addr[i / n16] + (i % n16) * 16)) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
    }
    if (acc == 0x9e3779b9u) *sink = acc;   // so the loads are not dropped as unused
}

__global__ void moe_combine_kernel(const float* __restrict__ out, const float* __restrict__ w, int k, float* __restrict__ y) {
    const int r = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= kWidth) return;
    float acc = 0.f;
    for (int s = 0; s < k; ++s) acc = fmaf(w[r * k + s], out[((size_t) r * k + s) * kWidth + c], acc);
    y[(size_t) r * kWidth + c] = acc;
}

__global__ void softcap_kernel(float* __restrict__ x, int n, int64_t ld, float cap) {
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    const int r = blockIdx.y;
    const float inv = 1.f / cap;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float& v = x[(size_t) r * ld + i];
        v = cap * tanhf(v * inv);
    }
}

}  // namespace

void attnres_mix(const MixStates& st, const float* queries, int n_q, int rows, float eps, float* out, int64_t out_stride,
                 void* stream) {
    if (rows <= 0) return;
    if (st.n < 1 || st.n > kMaxMix) throw std::runtime_error("inkling2 attnres_mix: state count");
    cudaStream_t s = (cudaStream_t) stream;
    switch (n_q) {
        case 1: attnres_mix_kernel<1><<<rows, kMixThreads, 0, s>>>(st, queries, eps, out, out_stride); break;
        case 3: attnres_mix_kernel<3><<<rows, kMixThreads, 0, s>>>(st, queries, eps, out, out_stride); break;
        case 4: attnres_mix_kernel<4><<<rows, kMixThreads, 0, s>>>(st, queries, eps, out, out_stride); break;
        default: throw std::runtime_error("inkling2 attnres_mix: query count");
    }
    check("attnres_mix");
}

void ctrl_pair_norm(const float* dest, const float* msg, int rows, float* pair, void* stream) {
    if (rows <= 0) return;
    ctrl_pair_norm_kernel<<<rows, kThreads, 0, (cudaStream_t) stream>>>(dest, msg, pair);
    check("ctrl_pair_norm");
}

void ctrl_blend(const float* dest, const float* msg, const float* gf, int rows, float* out, void* stream) {
    const int n = rows * kWidth;
    if (n <= 0) return;
    ctrl_blend_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(dest, msg, gf, n, out);
    check("ctrl_blend");
}

void attn_prepare(const AttnPrep& a, int rows, void* stream) {
    if (rows <= 0) return;
    cudaStream_t s = (cudaStream_t) stream;
    attn_prepare_kernel<<<rows, kThreads, 0, s>>>(a);
    check("attn_prepare");
    ring_store(a.yk, (int) a.ldk, 0, kKvHeads * kHeadDim, a.pos, rows, a.k_ring, s);
    ring_store(a.yv, (int) a.ldv, 0, kKvHeads * kHeadDim, a.pos, rows, a.v_ring, s);
    check("attn_prepare/rings");
}

void attn_gate(const float* attn, const float* yq, int rows, float* out, void* stream, int64_t yq_ld) {
    const int n4 = rows * (kHeads * kHeadDim / 4);
    if (n4 <= 0) return;
    attn_gate_kernel<<<(n4 + 255) / 256, 256, 0, (cudaStream_t) stream>>>(attn, yq, yq_ld, n4, out);
    check("attn_gate");
}

void post_conv(const float* x, const int32_t* pos, const float* taps, float* ring, const float* add, int rows, float* out,
               void* stream) {
    if (rows <= 0) return;
    cudaStream_t s = (cudaStream_t) stream;
    post_conv_kernel<<<dim3(kWidth / (4 * kPostConvThreads), (unsigned) ((rows + kPostConvTile - 1) / kPostConvTile)),
                       kPostConvThreads, 0, s>>>(x, pos, taps, ring, add, rows, out);
    check("post_conv");
    ring_store(x, kWidth, 0, kWidth, pos, rows, ring, s);
    check("post_conv/ring");
}

void moe_cpu_ring(const int32_t* ids, int rows, int k, const unsigned long long* blobs, const float* x, int layer, float* x_m,
                  int32_t* ids_m, CpuBell* bell, uint32_t* seq, void* stream) {
    if (rows <= 0) return;
    if (rows > kDecodeRows || rows * k > 1024 || layer < 0 || layer > 0xff)
        throw std::runtime_error("inkling2 moe_cpu_ring: rows, pairs or layer");
    moe_cpu_ring_kernel<<<1, kRingThreads, 0, (cudaStream_t) stream>>>(ids, rows, k, blobs, x, layer, x_m, ids_m, bell, seq);
    check("moe_cpu_ring");
}

void moe_cpu_merge(const int32_t* ids, int rows, int k, const unsigned long long* blobs, const float* cpu, float* output,
                   uint32_t* tally, void* stream, CpuBell* bell, const uint32_t* seq) {
    if (rows <= 0) return;
    moe_cpu_merge_kernel<<<rows * k, kMergeThreads, 0, (cudaStream_t) stream>>>(ids, blobs, cpu, output, tally, bell, seq);
    check("moe_cpu_merge");
}

void expert_read_ahead(const unsigned long long* blobs, const int32_t* n, uint64_t bytes, void* stream) {
    if (bytes == 0 || bytes % 16) throw std::runtime_error("inkling2 expert_read_ahead: bytes");
    expert_read_ahead_kernel<<<kAheadBlocks, 32, 0, (cudaStream_t) stream>>>(blobs, n, bytes);
    check("expert_read_ahead");
}

void expert_prefetch(const int32_t* ids, int n, const unsigned long long* blobs, uint64_t bytes, uint32_t* sink, void* stream) {
    if (n <= 0) return;
    if (n > kPrefetchMax || bytes == 0 || bytes % 16) throw std::runtime_error("inkling2 expert_prefetch: experts or bytes");
    expert_prefetch_kernel<<<kPrefetchBlocks, kPrefetchThreads, 0, (cudaStream_t) stream>>>(ids, n, blobs, bytes, sink);
    check("expert_prefetch");
}

void moe_plan(const int32_t* ids, int rows, int k, int n_expert, const unsigned long long* blobs, unsigned long long* ptr,
              int32_t* start, int32_t* n_groups, int32_t* dst, int32_t* tok, void* stream, uint32_t* tally,
              unsigned long long* rec_blobs, int32_t* rec_n) {
    const int n = rows * k;
    if (n <= 0) return;
    if (n > 1024) throw std::runtime_error("inkling2 moe_plan: more than 1024 pairs");
    if (n_expert > kPlanThreads) throw std::runtime_error("inkling2 moe_plan: more experts than a block's threads");
    bk::launch_pdl(moe_plan_kernel, dim3(1), dim3(kPlanThreads),
                   (size_t) n_expert * (2 * sizeof(unsigned long long) + 2 * sizeof(int32_t)) + (size_t) n * sizeof(int32_t), (cudaStream_t) stream,
                   ids, n, n_expert, blobs, ptr, start, n_groups, dst, tok, k, tally, rec_blobs, rec_n);
    check("moe_plan");
}

void moe_combine(const float* out, const float* w, int rows, int k, float* y, void* stream) {
    if (rows <= 0) return;
    moe_combine_kernel<<<dim3((kWidth + 255) / 256, (unsigned) rows), 256, 0, (cudaStream_t) stream>>>(out, w, k, y);
    check("moe_combine");
}

void softcap(float* x, int rows, int n, int64_t ld, float cap, void* stream) {
    if (rows <= 0 || n <= 0) return;
    bk::launch_pdl(softcap_kernel, dim3(256, (unsigned) rows), dim3(256), 0, (cudaStream_t) stream, x, n, ld, cap);
    check("softcap");
}

}  // namespace basalt::models::inkling2
