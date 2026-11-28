// src/models/ernie55/kernels/ops.cu - ERNIE-5.5-Flash CUDA kernels:
//   - mHC with ERNIE's exact RMS factor 1 / (sqrt(mean) + rms_eps) and row-major destination-first comb[d, s]
//   - mHC final contraction head (hc_head_fn, hc_head_scale, hc_head_base)
//   - FP32 LM-head multimax
//   - SiTU-GLU (beta * tanh(g / beta) * sigmoid(g) * linear_beta * tanh(u / linear_beta))
//   - Split-feature MoE router (sigmoid(l0) + sigmoid(l1) + bias + top-K + renorm + learnable per-expert scale)
//   - VHA PostMix, sigmoid output gate, LayerNorm, orthonormal Hadamard-128, weighted sum, RoPE tables
#include "basalt/models/ernie55/kernels.hpp"

#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/q8_1_finite.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <stdexcept>

namespace basalt::models::ernie55 {

// ---- RoPE tables ----------------------------------------------------------------------------------------------

namespace {

float yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * std::log((float) n_ctx_orig / (n_rot * 2.0f * 3.14159265358979323846f)) / (2.0f * std::log(base));
}

}  // namespace

RopeTable make_rope_table(float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast,
                          float beta_slow, int n_ctx_orig) {
    constexpr int n_dims = 64;
    const float lo = ext_factor != 0.0f ? std::max(0.0f, std::floor(yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base))) : 0.0f;
    const float hi = ext_factor != 0.0f ? std::min((float) (n_dims - 1), std::ceil(yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base))) : 0.0f;
    const float theta_scale = std::pow(freq_base, -2.0f / (float) n_dims);
    RopeTable t{};
    float extrap = 1.0f;
    for (int i = 0; i < n_dims / 2; ++i) {
        float step = freq_scale * extrap;
        if (ext_factor != 0.0f) {
            const float y = ((float) i - lo) / std::max(0.001f, hi - lo);
            const float ramp = (1.0f - std::min(1.0f, std::max(0.0f, y))) * ext_factor;
            step = step * (1.0f - ramp) + extrap * ramp;
        }
        t.step[i] = step;
        extrap *= theta_scale;
    }
    t.mscale = attn_factor;
    if (ext_factor != 0.0f) t.mscale *= 1.0f + 0.1f * std::log(1.0f / freq_scale);
    return t;
}

RopeTable rope_table_plain(float freq_base) { return make_rope_table(freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0); }

RopeTable rope_table_yarn(float freq_base, float factor, float beta_fast, float beta_slow, int n_ctx_orig) {
    const float freq_scale = 1.0f / factor;
    const float attn_factor = 1.0f / (1.0f + 0.1f * std::log(1.0f / freq_scale));
    return make_rope_table(freq_base, freq_scale, 1.0f, attn_factor, beta_fast, beta_slow, n_ctx_orig);
}

namespace {

namespace bk = basalt::kernels;

constexpr int kThreads = 256;
constexpr int kSlots = kHcCoef + 1;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

__device__ __forceinline__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = 0.f;
#pragma unroll
    for (int i = 0; i < kThreads / 32; ++i) r += red[i];
    __syncthreads();
    return r;
}

__device__ __forceinline__ float sigmoidf_ieee(float v) { return 1.0f / (1.0f + expf(-v)); }

// ---- mHC kernels ----------------------------------------------------------------------------------------------

__global__ void mhc_partial_kernel(const float* __restrict__ x, const float* __restrict__ fn, int flat, int n_mix,
                                   float* __restrict__ ws) {
    bk::pdl_wait();
    __shared__ float red[kThreads / 32];
    const int split = blockIdx.x, mix = blockIdx.y, t = blockIdx.z;
    const int len = flat / kHcSplits, base = split * len;
    const float4* xv = reinterpret_cast<const float4*>(x + (size_t) t * flat + base);
    const float4* wv = reinterpret_cast<const float4*>(fn + (size_t) mix * flat + base);
    float dot = 0.f, ss = 0.f;
    for (int i = threadIdx.x; i < len / 4; i += kThreads) {
        const float4 a = xv[i], w = wv[i];
        dot = fmaf(a.x, w.x, dot); dot = fmaf(a.y, w.y, dot);
        dot = fmaf(a.z, w.z, dot); dot = fmaf(a.w, w.w, dot);
        if (mix == 0) {
            ss = fmaf(a.x, a.x, ss); ss = fmaf(a.y, a.y, ss);
            ss = fmaf(a.z, a.z, ss); ss = fmaf(a.w, a.w, ss);
        }
    }
    dot = block_sum(dot, red);
    if (mix == 0) ss = block_sum(ss, red);
    if (threadIdx.x == 0) {
        float* part = ws + ((size_t) t * kHcSplits + split) * (n_mix + 1);
        part[mix] = dot;
        if (mix == 0) part[n_mix] = ss;
    }
}

// One warp per token: computes ERNIE 5.5 mHC coefficients with RMS = 1 / (sqrt(mean) + rms_eps).
// Output layout per token (24 floats): [pre(0..3), post(4..7), comb(8..23)] where comb[d*4 + s] is row d, col s.
__global__ void mhc_coef_ernie_kernel(const float* __restrict__ ws, const float* __restrict__ scale,
                                      const float* __restrict__ base, int flat, float rms_eps, float hc_eps,
                                      int sinkhorn_iters, float* __restrict__ coef) {
    bk::pdl_wait();
    const int t = blockIdx.x, lane = threadIdx.x & 31;
    const float* w = ws + (size_t) t * kHcSplits * kSlots;
    float ss = 0.f;
    for (int s = 0; s < kHcSplits; ++s) ss += w[(size_t) s * kSlots + kHcCoef];
    // ERNIE 5.5 exact RMS factor: eps outside sqrt
    const float inv_rms = 1.0f / (sqrtf(ss / (float) flat) + rms_eps);

    __shared__ float mix[kHcCoef];
    if (lane < kHcCoef) {
        float acc = 0.f;
        for (int s = 0; s < kHcSplits; ++s) acc += w[(size_t) s * kSlots + lane];
        mix[lane] = acc * inv_rms;
    }
    __syncwarp();

    float* out = coef + (size_t) t * kHcCoef;
    if (lane < 4) {
        out[lane] = sigmoidf_ieee(mix[lane] * scale[0] + base[lane]) + hc_eps;
    } else if (lane < 8) {
        out[lane] = sigmoidf_ieee(mix[lane] * scale[1] + base[lane]) * 2.0f;
    }

    // Sinkhorn-Knopp on 4x4 matrix comb[d, s] = mix[8 + d*4 + s] * scale[2] + base[8 + d*4 + s]: lane d*4 + s holds
    // comb[d, s] and every sum adds its four terms in the order a serial pass would, so the bits are a serial pass's
    const int r = (lane >> 2) & 3, c = lane & 3;
    const unsigned all = 0xffffffffu;
    float v = mix[8 + (lane & 15)] * scale[2] + base[8 + (lane & 15)];
    const auto row = [&](float x, int j) { return __shfl_sync(all, x, (lane & 16) | (r << 2) | j); };
    const auto col = [&](float x, int i) { return __shfl_sync(all, x, (lane & 16) | (i << 2) | c); };
    {   // row softmax + hc_eps
        const float m = fmaxf(fmaxf(row(v, 0), row(v, 1)), fmaxf(row(v, 2), row(v, 3)));
        v = expf(v - m);
        float s = 0.f;
        s += row(v, 0);
        s += row(v, 1);
        s += row(v, 2);
        s += row(v, 3);
        const float inv = 1.0f / s;
        v = v * inv + hc_eps;
    }
    {   // first column normalization
        const float cs = col(v, 0) + col(v, 1) + col(v, 2) + col(v, 3) + hc_eps;
        v *= 1.0f / cs;
    }
    for (int it = 1; it < sinkhorn_iters; ++it) {   // the remaining iterations: row norm then column norm
        const float rs = row(v, 0) + row(v, 1) + row(v, 2) + row(v, 3) + hc_eps;
        v *= 1.0f / rs;
        const float cs = col(v, 0) + col(v, 1) + col(v, 2) + col(v, 3) + hc_eps;
        v *= 1.0f / cs;
    }
    if (lane < 16) out[8 + lane] = v;
}

// Single-pass prefill mHC mixes: 1 CTA (8 warps = 256 threads) processes up to 8 tokens simultaneously, reading x
// ONCE from VRAM (instead of 24x in mhc_partial_kernel) and hc_fn ONCE per 8 tokens from L2, then computing the exact
// ERNIE RMS + Sinkhorn-Knopp coefficients in-register without touching workspace memory.
__global__ void __launch_bounds__(256) mhc_mixes_prefill_kernel(const float* __restrict__ x,
                                                                const float* __restrict__ fn,
                                                                const float* __restrict__ scale,
                                                                const float* __restrict__ base, int n_tokens, int flat,
                                                                float rms_eps, float hc_eps, int sinkhorn_iters,
                                                                float* __restrict__ coef) {
    constexpr int kBT = 8;
    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
    const int t0 = blockIdx.x * kBT;
    const int bt_n = min(kBT, n_tokens - t0);
    const int n4 = flat >> 2;

    __shared__ float4 sm_x[kBT][64];
    __shared__ float sm_dot[kBT][kHcCoef];
    __shared__ float sm_ss[kBT];

    float dot[3][kBT];
#pragma unroll
    for (int j = 0; j < 3; ++j)
#pragma unroll
        for (int b = 0; b < kBT; ++b) dot[j][b] = 0.0f;
    float ss_acc = 0.0f;

    const float4* x_w = (wid < bt_n) ? reinterpret_cast<const float4*>(x + (size_t) (t0 + wid) * flat) : nullptr;
    const float4* fn0 = reinterpret_cast<const float4*>(fn + (size_t) (wid * 3 + 0) * flat);
    const float4* fn1 = reinterpret_cast<const float4*>(fn + (size_t) (wid * 3 + 1) * flat);
    const float4* fn2 = reinterpret_cast<const float4*>(fn + (size_t) (wid * 3 + 2) * flat);

    for (int k4 = 0; k4 < n4; k4 += 64) {
        if (x_w) {
            const float4 a0 = x_w[k4 + lane];
            const float4 a1 = x_w[k4 + lane + 32];
            sm_x[wid][lane] = a0;
            sm_x[wid][lane + 32] = a1;
            ss_acc = fmaf(a0.x, a0.x, ss_acc); ss_acc = fmaf(a0.y, a0.y, ss_acc);
            ss_acc = fmaf(a0.z, a0.z, ss_acc); ss_acc = fmaf(a0.w, a0.w, ss_acc);
            ss_acc = fmaf(a1.x, a1.x, ss_acc); ss_acc = fmaf(a1.y, a1.y, ss_acc);
            ss_acc = fmaf(a1.z, a1.z, ss_acc); ss_acc = fmaf(a1.w, a1.w, ss_acc);
        } else {
            sm_x[wid][lane] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            sm_x[wid][lane + 32] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        __syncthreads();

        const float4 w0a = fn0[k4 + lane], w0b = fn0[k4 + lane + 32];
        const float4 w1a = fn1[k4 + lane], w1b = fn1[k4 + lane + 32];
        const float4 w2a = fn2[k4 + lane], w2b = fn2[k4 + lane + 32];
#pragma unroll
        for (int b = 0; b < kBT; ++b) {
            const float4 xa = sm_x[b][lane];
            const float4 xb = sm_x[b][lane + 32];
            dot[0][b] = fmaf(xa.x, w0a.x, dot[0][b]); dot[0][b] = fmaf(xa.y, w0a.y, dot[0][b]);
            dot[0][b] = fmaf(xa.z, w0a.z, dot[0][b]); dot[0][b] = fmaf(xa.w, w0a.w, dot[0][b]);
            dot[0][b] = fmaf(xb.x, w0b.x, dot[0][b]); dot[0][b] = fmaf(xb.y, w0b.y, dot[0][b]);
            dot[0][b] = fmaf(xb.z, w0b.z, dot[0][b]); dot[0][b] = fmaf(xb.w, w0b.w, dot[0][b]);

            dot[1][b] = fmaf(xa.x, w1a.x, dot[1][b]); dot[1][b] = fmaf(xa.y, w1a.y, dot[1][b]);
            dot[1][b] = fmaf(xa.z, w1a.z, dot[1][b]); dot[1][b] = fmaf(xa.w, w1a.w, dot[1][b]);
            dot[1][b] = fmaf(xb.x, w1b.x, dot[1][b]); dot[1][b] = fmaf(xb.y, w1b.y, dot[1][b]);
            dot[1][b] = fmaf(xb.z, w1b.z, dot[1][b]); dot[1][b] = fmaf(xb.w, w1b.w, dot[1][b]);

            dot[2][b] = fmaf(xa.x, w2a.x, dot[2][b]); dot[2][b] = fmaf(xa.y, w2a.y, dot[2][b]);
            dot[2][b] = fmaf(xa.z, w2a.z, dot[2][b]); dot[2][b] = fmaf(xa.w, w2a.w, dot[2][b]);
            dot[2][b] = fmaf(xb.x, w2b.x, dot[2][b]); dot[2][b] = fmaf(xb.y, w2b.y, dot[2][b]);
            dot[2][b] = fmaf(xb.z, w2b.z, dot[2][b]); dot[2][b] = fmaf(xb.w, w2b.w, dot[2][b]);
        }
        __syncthreads();
    }

    ss_acc = warp_sum(ss_acc);
    if (lane == 0) sm_ss[wid] = ss_acc;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
#pragma unroll
        for (int b = 0; b < kBT; ++b) {
            const float s = warp_sum(dot[j][b]);
            if (lane == 0) sm_dot[b][wid * 3 + j] = s;
        }
    }
    __syncthreads();

    if (wid >= bt_n) return;
    const float inv_rms = 1.0f / (sqrtf(sm_ss[wid] / (float) flat) + rms_eps);
    const float mix_val = (lane < kHcCoef) ? (sm_dot[wid][lane] * inv_rms) : 0.0f;
    float* out = coef + (size_t) (t0 + wid) * kHcCoef;
    if (lane < 4) {
        out[lane] = sigmoidf_ieee(mix_val * scale[0] + base[lane]) + hc_eps;
    } else if (lane < 8) {
        out[lane] = sigmoidf_ieee(mix_val * scale[1] + base[lane]) * 2.0f;
    }
    const int r = (lane >> 2) & 3, c = lane & 3;
    const unsigned all = 0xffffffffu;
    const float m_comb = sm_dot[wid][8 + (lane & 15)] * inv_rms;
    float v = m_comb * scale[2] + base[8 + (lane & 15)];
    const auto row = [&](float x_, int j) { return __shfl_sync(all, x_, (lane & 16) | (r << 2) | j); };
    const auto col = [&](float x_, int i) { return __shfl_sync(all, x_, (lane & 16) | (i << 2) | c); };
    {
        const float m = fmaxf(fmaxf(row(v, 0), row(v, 1)), fmaxf(row(v, 2), row(v, 3)));
        v = expf(v - m);
        float s = 0.f;
        s += row(v, 0);
        s += row(v, 1);
        s += row(v, 2);
        s += row(v, 3);
        const float inv = 1.0f / s;
        v = v * inv + hc_eps;
    }
    {
        const float cs = col(v, 0) + col(v, 1) + col(v, 2) + col(v, 3) + hc_eps;
        v *= 1.0f / cs;
    }
    for (int it = 1; it < sinkhorn_iters; ++it) {
        const float rs = row(v, 0) + row(v, 1) + row(v, 2) + row(v, 3) + hc_eps;
        v *= 1.0f / rs;
        const float cs = col(v, 0) + col(v, 1) + col(v, 2) + col(v, 3) + hc_eps;
        v *= 1.0f / cs;
    }
    if (lane < 16) out[8 + lane] = v;
}

__global__ void mhc_contract_coef_kernel(const float* __restrict__ ws, const float* __restrict__ scale,
                                         const float* __restrict__ base, int flat, float rms_eps, float hc_eps,
                                         float* __restrict__ coef4) {
    bk::pdl_wait();
    const int t = blockIdx.x, lane = threadIdx.x & 31;
    const float* w = ws + (size_t) t * kHcSplits * 5;
    float ss = 0.f;
    for (int s = 0; s < kHcSplits; ++s) ss += w[(size_t) s * 5 + 4];
    const float inv_rms = 1.0f / (sqrtf(ss / (float) flat) + rms_eps);
    if (lane < 4) {
        float acc = 0.f;
        for (int s = 0; s < kHcSplits; ++s) acc += w[(size_t) s * 5 + lane];
        coef4[(size_t) t * 4 + lane] = sigmoidf_ieee((acc * inv_rms) * scale[0] + base[lane]) + hc_eps;
    }
}

__global__ void mhc_pre_kernel(const float* __restrict__ x, const float* __restrict__ coef, int coef_stride,
                               int n_embd, float* __restrict__ out) {
    bk::pdl_wait();
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int n4 = n_embd >> 2;
    if (i >= n4) return;
    const float* p = coef + (size_t) t * coef_stride;
    const float4* xt = reinterpret_cast<const float4*>(x + (size_t) t * kHcMult * n_embd);
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int s = 0; s < kHcMult; ++s) {
        const float w = p[s];
        const float4 v = xt[(size_t) s * n4 + i];
        acc.x = fmaf(w, v.x, acc.x); acc.y = fmaf(w, v.y, acc.y);
        acc.z = fmaf(w, v.z, acc.z); acc.w = fmaf(w, v.w, acc.w);
    }
    reinterpret_cast<float4*>(out + (size_t) t * n_embd)[i] = acc;
}

// f2 (optional): f + f2 is the sublayer's output (add_f32's sum, then the update)
__global__ void mhc_post_kernel(const float* f, const float* f2, const float* x, const float* coef, int coef_stride,
                                int n_embd, float* out) {
    bk::pdl_wait();
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int n4 = n_embd >> 2;
    if (i >= n4) return;
    const float* cf = coef + (size_t) t * coef_stride;
    float4 fv = reinterpret_cast<const float4*>(f + (size_t) t * n_embd)[i];
    if (f2) {
        const float4 g = reinterpret_cast<const float4*>(f2 + (size_t) t * n_embd)[i];
        fv.x += g.x; fv.y += g.y; fv.z += g.z; fv.w += g.w;
    }
    const float4* xt = reinterpret_cast<const float4*>(x + (size_t) t * kHcMult * n_embd);
    float4 xs[kHcMult];
#pragma unroll
    for (int s = 0; s < kHcMult; ++s) xs[s] = xt[(size_t) s * n4 + i];
    float4* ot = reinterpret_cast<float4*>(out + (size_t) t * kHcMult * n_embd);
#pragma unroll
    for (int d = 0; d < kHcMult; ++d) {
        const float post = cf[4 + d];
        float4 o = make_float4(fv.x * post, fv.y * post, fv.z * post, fv.w * post);
#pragma unroll
        for (int s = 0; s < kHcMult; ++s) {
            // vLLM mhc_post_torch: out[d] = post[d] * f + sum_s comb[s, d] * x[s]
            const float c = cf[8 + s * 4 + d];
            o.x = fmaf(xs[s].x, c, o.x); o.y = fmaf(xs[s].y, c, o.y);
            o.z = fmaf(xs[s].z, c, o.z); o.w = fmaf(xs[s].w, c, o.w);
        }
        ot[(size_t) d * n4 + i] = o;
    }
}

__global__ void broadcast_streams_kernel(const float* __restrict__ x, int n_embd, float* __restrict__ out) {
    bk::pdl_wait();
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    const float v = x[(size_t) t * n_embd + i];
#pragma unroll
    for (int s = 0; s < kHcMult; ++s) out[((size_t) t * kHcMult + s) * n_embd + i] = v;
}

// ---- Elementwise kernels --------------------------------------------------------------------------------------

__device__ __forceinline__ float relu_nan(float x) { return x <= 0.0f ? 0.0f : x; }

__global__ void multimax_kernel(float* __restrict__ logits, const float* __restrict__ r,
                                const float* __restrict__ ts, int n_vocab) {
    bk::pdl_wait();
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_vocab) return;
    float* p = logits + (size_t) t * n_vocab + i;
    const float x = *p;
    const float a = relu_nan(r[0] - x);
    const float b = relu_nan(x - r[1]);
    const float c = relu_nan(r[2] - x);
    const float d = relu_nan(x - r[3]);
    float y = __fadd_rn(x, __fmul_rn(ts[0], a));
    y = __fadd_rn(y, __fmul_rn(ts[1], b));
    y = __fadd_rn(y, __fmul_rn(ts[2], __fmul_rn(c, c)));
    y = __fadd_rn(y, __fmul_rn(ts[3], __fmul_rn(d, d)));
    *p = y;
}

__global__ void situ_glu_kernel(const float* gate, const float* up, int64_t n,
                                float beta, float linear_beta, float* out) {
    bk::pdl_wait();
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i], u = up[i];
    const float sg = __fmul_rn(beta * tanhf(g / beta), sigmoidf_ieee(g));
    const float su = linear_beta > 0.0f ? (linear_beta * tanhf(u / linear_beta)) : u;
    out[i] = __fmul_rn(sg, su);
}

// One block (256 threads) per token: split-feature sigmoid sum + bias-corrected top-K + renorm + per-expert scale.
// Thread i asks the copy engine of the SM (cp.async.bulk.prefetch) for pieces i, i + 256, ... of each span, into L2.
__global__ void l2_prefetch_kernel(L2Spans s) {
    constexpr uint32_t kPiece = 32u << 10;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    for (int i = 0; i < s.n; ++i) {
        const char* base = static_cast<const char*>(s.ptr[i]);
        for (uint32_t off = threadIdx.x * kPiece; off < s.bytes[i]; off += blockDim.x * kPiece) {
            const uint32_t n = min(kPiece, s.bytes[i] - off) & ~15u;
            if (n) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(base + off), "r"(n) : "memory");
        }
    }
#endif
}

__global__ void bf16_widen_kernel(const uint16_t* __restrict__ in, float* __restrict__ out, int64_t n) {
    for (int64_t i = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) * 4; i < n; i += (int64_t) gridDim.x * blockDim.x * 4)
        for (int k = 0; k < 4 && i + k < n; ++k) out[i + k] = __uint_as_float((uint32_t) in[i + k] << 16);
}

__global__ void split_sigmoid_route_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                                           const float* __restrict__ scale, int n_expert, int top_k, bool renorm,
                                           int32_t* __restrict__ ids, float* __restrict__ weights,
                                           kernels::ResidentTabs held, float route_prior) {
    bk::pdl_wait();
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    extern __shared__ float smem[];
    float* scores = smem;               // [n_expert]
    float* sel = smem + n_expert;       // [n_expert]
    const float* row = logits + (size_t) t * (2 * n_expert);
    for (int e = tid; e < n_expert; e += blockDim.x) {
        const float s = sigmoidf_ieee(row[e]) + sigmoidf_ieee(row[n_expert + e]);
        scores[e] = s;
        float sv = s + bias[e];
        if (route_prior > 0.0f && held.held(e)) sv += route_prior;
        sel[e] = sv;
    }
    __syncthreads();

    // top-K: the largest value, the lowest id among equals (a serial scan's picks, in order). Each warp picks the top-K
    // of its own span by warp argmax (shuffles only), then warp 0 picks the top-K of those: the same ids.
    constexpr int kSpanLanes = 4, kMaxCand = 128;
    __shared__ float cand_v[kMaxCand];
    __shared__ int cand_i[kMaxCand];
    __shared__ int picked[64];
    const int lane = tid & 31, wid = tid >> 5, nw = (blockDim.x + 31) >> 5;
    const auto better = [](float a, int ia, float b, int ib) { return a > b || (a == b && ia < ib); };
    const auto warp_best = [&](float& v, int& i) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, o);
            const int oi = __shfl_xor_sync(0xffffffffu, i, o);
            if (better(ov, oi, v, i)) {
                v = ov;
                i = oi;
            }
        }
    };
    const int span = (n_expert + nw - 1) / nw;
    if (span <= 32 * kSpanLanes && top_k * nw <= kMaxCand) {
        float v[kSpanLanes];
        int id[kSpanLanes];
#pragma unroll
        for (int j = 0; j < kSpanLanes; ++j) {
            const int c = lane + 32 * j, e = wid * span + c;
            const bool in = c < span && e < n_expert;
            v[j] = in ? sel[e] : -FLT_MAX;
            id[j] = in ? e : 0x7fffffff;
        }
        for (int k = 0; k < top_k; ++k) {
            float bv = -FLT_MAX;
            int bi = 0x7fffffff;
#pragma unroll
            for (int j = 0; j < kSpanLanes; ++j)
                if (v[j] > bv) {   // a lane's own values in ascending id: the first of equals stays
                    bv = v[j];
                    bi = id[j];
                }
            warp_best(bv, bi);
#pragma unroll
            for (int j = 0; j < kSpanLanes; ++j)
                if (id[j] == bi) {
                    v[j] = -FLT_MAX;
                    id[j] = 0x7fffffff;
                }
            if (lane == 0) {
                cand_v[wid * top_k + k] = bv;
                cand_i[wid * top_k + k] = bi;
            }
        }
        __syncthreads();
        if (wid == 0) {
            const int nc = nw * top_k;
            for (int k = 0; k < top_k; ++k) {
                float bv = -FLT_MAX;
                int bi = 0x7fffffff, slot = -1;
                for (int c = lane; c < nc; c += 32)
                    if (better(cand_v[c], cand_i[c], bv, bi)) {
                        bv = cand_v[c];
                        bi = cand_i[c];
                        slot = c;
                    }
                float wv = bv;
                int wi = bi;
                warp_best(wv, wi);
                if (slot >= 0 && bi == wi && bv == wv) {
                    cand_v[slot] = -FLT_MAX;
                    cand_i[slot] = 0x7fffffff;
                }
                if (lane == 0) picked[k] = wi == 0x7fffffff ? 0 : wi;
                __syncwarp();
            }
        }
        __syncthreads();
    } else {
        __shared__ float red_v[32];
        __shared__ int red_i[32];
        for (int k = 0; k < top_k; ++k) {
            float bv = -FLT_MAX;
            int bi = 0x7fffffff;
            for (int e = tid; e < n_expert; e += blockDim.x)
                if (sel[e] > bv) {
                    bv = sel[e];
                    bi = e;
                }
            warp_best(bv, bi);
            if (lane == 0) {
                red_v[wid] = bv;
                red_i[wid] = bi;
            }
            __syncthreads();
            if (tid == 0) {
                float v = -FLT_MAX;
                int id = 0x7fffffff;
                for (int w = 0; w < nw; ++w)
                    if (better(red_v[w], red_i[w], v, id)) {
                        v = red_v[w];
                        id = red_i[w];
                    }
                if (id == 0x7fffffff) id = 0;
                picked[k] = id;
                sel[id] = -FLT_MAX;
            }
            __syncthreads();
        }
    }

    // the weights: the picks' scores over their sum (in pick order), times their scale; one thread a pick, each written once
    __shared__ float denom;
    if (tid == 0) {
        float total = 0.0f;
        for (int k = 0; k < top_k; ++k) total += scores[picked[k]];
        denom = fmaxf(total, 1e-20f);
    }
    __syncthreads();
    for (int k = tid; k < top_k; k += blockDim.x) {
        const int id = picked[k];
        float w = scores[id];
        if (renorm) w /= denom;
        w *= scale[id];
        ids[(size_t) t * top_k + k] = id;
        weights[(size_t) t * top_k + k] = w;
    }
}

// split_sigmoid_route_kernel's picks by one bitonic sort of the n_expert <= 512 keys (value, then the lower id first)
// in shared memory: the same ids in the same order, the same weights.
__global__ void __launch_bounds__(256) split_sigmoid_route_sort_kernel(const float* logits, const float* __restrict__ bias,
                                                                       const float* __restrict__ scale, int n_expert,
                                                                       int top_k, bool renorm, int32_t* ids,
                                                                       float* weights, kernels::ResidentTabs held,
                                                                       float route_prior) {
    bk::pdl_wait();
    const int t = blockIdx.x, tid = threadIdx.x;
    __shared__ float scores[512];
    __shared__ unsigned long long key[512];
    const float* row = logits + (size_t) t * (2 * n_expert);
    for (int e = tid; e < 512; e += 256) {
        unsigned long long k = 0ull;   // past n_expert: below every expert
        if (e < n_expert) {
            const float sc = sigmoidf_ieee(row[e]) + sigmoidf_ieee(row[n_expert + e]);
            scores[e] = sc;
            float sv = sc + bias[e];
            if (route_prior > 0.0f && held.held(e)) sv += route_prior;
            uint32_t u = __float_as_uint(sv);
            u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);   // ordered as the floats
            k = ((unsigned long long) u << 32) | (unsigned long long) (0xffffffffu - (uint32_t) e);
        }
        key[e] = k;
    }
    __syncthreads();
    for (int size = 2; size <= 512; size <<= 1)   // descending bitonic sort, a compare-exchange a thread a step
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            const int i = 2 * tid - (tid & (stride - 1)), j = i + stride;
            const bool desc = (i & size) == 0;
            const unsigned long long a = key[i], b = key[j];
            if ((a < b) == desc) {
                key[i] = b;
                key[j] = a;
            }
            __syncthreads();
        }
    __shared__ float denom;
    if (tid == 0) {
        float total = 0.0f;
        for (int k = 0; k < top_k; ++k) total += scores[0xffffffffu - (uint32_t) key[k]];
        denom = fmaxf(total, 1e-20f);
    }
    __syncthreads();
    for (int k = tid; k < top_k; k += blockDim.x) {
        const int id = (int) (0xffffffffu - (uint32_t) key[k]);
        float w = scores[id];
        if (renorm) w /= denom;
        w *= scale[id];
        ids[(size_t) t * top_k + k] = id;
        weights[(size_t) t * top_k + k] = w;
    }
}

__global__ void vha_postmix_kernel(const float* __restrict__ in, const float* __restrict__ mix,
                                   int n_tokens, int n_heads, int head_dim, bool head_major, float* __restrict__ out) {
    bk::pdl_wait();
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y;
    const int t = blockIdx.z;
    if (d >= head_dim) return;
    const float* mrow = mix + (size_t) h * n_heads;
    float acc = 0.0f;
    if (!head_major) {
        const float* base = in + (size_t) t * n_heads * head_dim + d;
        for (int g = 0; g < n_heads; ++g) acc = fmaf(mrow[g], base[(size_t) g * head_dim], acc);
    } else {
        const float* base = in + (size_t) t * head_dim + d;
        const size_t stride = (size_t) n_tokens * head_dim;
        for (int g = 0; g < n_heads; ++g) acc = fmaf(mrow[g], base[(size_t) g * stride], acc);
    }
    out[((size_t) t * n_heads + h) * head_dim + d] = acc;
}

__global__ void sigmoid_gate_mul_kernel(float* __restrict__ z, const float* __restrict__ gate, int64_t n) {
    bk::pdl_wait();
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] *= sigmoidf_ieee(gate[i]);
}

__global__ void layernorm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                 const float* __restrict__ b, int dim, float eps, float* __restrict__ out) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float red[8];
    const float* xr = x + (size_t) t * dim;
    float sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) sum += xr[i];
    sum = warp_sum(sum);
    if ((tid & 31) == 0) red[tid >> 5] = sum;
    __syncthreads();
    float mean = 0.0f;
    for (int w_i = 0; w_i < blockDim.x / 32; ++w_i) mean += red[w_i];
    mean /= (float) dim;
    __syncthreads();

    float var = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        const float d = xr[i] - mean;
        var += d * d;
    }
    var = warp_sum(var);
    if ((tid & 31) == 0) red[tid >> 5] = var;
    __syncthreads();
    float vsum = 0.0f;
    for (int w_i = 0; w_i < blockDim.x / 32; ++w_i) vsum += red[w_i];
    const float inv = rsqrtf(vsum / (float) dim + eps);
    float* yr = out + (size_t) t * dim;
    for (int i = tid; i < dim; i += blockDim.x) yr[i] = (xr[i] - mean) * inv * w[i] + b[i];
}

// Orthonormal 128-point Walsh-Hadamard transform: one block of 128 threads per row (7 butterfly stages).
__global__ void hadamard_128_kernel(float* __restrict__ x) {
    __shared__ float s[128];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    s[tid] = x[(size_t) row * 128 + tid];
    __syncthreads();
#pragma unroll
    for (int step = 1; step < 128; step <<= 1) {
        const int partner = tid ^ step;
        const float a = s[tid], b = s[partner];
        __syncthreads();
        s[tid] = (tid < partner) ? (a + b) : (b - a);
        __syncthreads();
    }
    // 1 / sqrt(128) = 0.08838834764831845
    x[(size_t) row * 128 + tid] = s[tid] * 0.08838834764831845f;
}

__global__ void weighted_sum_kernel(const float* __restrict__ exp_out, const float* __restrict__ weights,
                                    int top_k, int dim, float* __restrict__ out) {
    bk::pdl_wait();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (i >= dim) return;
    const float* wt = weights + (size_t) t * top_k;
    const float* eb = exp_out + (size_t) t * top_k * dim + i;
    float acc = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        const float w = wt[k];
        if (w != 0.0f) acc = fmaf(w, eb[(size_t) k * dim], acc);
    }
    out[(size_t) t * dim + i] = acc;
}

__global__ void add_f32_kernel(float* __restrict__ dst, const float* __restrict__ src, int64_t n) {
    bk::pdl_wait();
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

__global__ void weighted_sum_mapped_kernel(const float* __restrict__ pair_rows, const int32_t* __restrict__ pair_map,
                                           const float* __restrict__ weights, int top_k, int dim,
                                           float* __restrict__ out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (c >= dim) return;
    const float* w = weights + (size_t) t * top_k;
    const int32_t* pm = pair_map + (size_t) t * top_k;
    float acc = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        const int32_t row = pm[k];
        if (row >= 0) acc = fmaf(w[k], pair_rows[(size_t) row * dim + c], acc);
    }
    out[(size_t) t * dim + c] = acc;
}

__global__ void hca_group_split_kernel(const float* __restrict__ src, int n_groups, int grp, int gin,
                                       float* __restrict__ dst) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (c < gin) dst[(size_t) t * gin + c] = src[((size_t) t * n_groups + grp) * gin + c];
}

__global__ void hca_group_merge_kernel(const float* __restrict__ src, int n_groups, int grp, int gout,
                                       float* __restrict__ dst) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (c < gout) dst[((size_t) t * n_groups + grp) * gout + c] = src[(size_t) t * gout + c];
}

}  // namespace

size_t mhc_workspace_bytes(int n_tokens) {
    return (size_t) n_tokens * kHcSplits * kSlots * sizeof(float);
}

void mhc_mixes(const float* x, const float* hc_fn, const float* hc_scale, const float* hc_base, int n_tokens,
               int n_embd, float rms_eps, float hc_eps, int sinkhorn_iters, float* workspace, float* coef,
               void* stream) {
    const int flat = kHcMult * n_embd;
    if (flat % (kHcSplits * 4) || sinkhorn_iters < 1) throw std::invalid_argument("ernie55 mhc_mixes: geometry");
    cudaStream_t st = (cudaStream_t) stream;
    if (n_tokens > 8 && (flat % 256) == 0) {
        mhc_mixes_prefill_kernel<<<(n_tokens + 7) / 8, 256, 0, st>>>(
            x, hc_fn, hc_scale, hc_base, n_tokens, flat, rms_eps, hc_eps, sinkhorn_iters, coef);
        return;
    }
    bk::launch_pdl(mhc_partial_kernel, dim3(kHcSplits, kHcCoef, n_tokens), dim3(kThreads), 0, st, x, hc_fn, flat, kHcCoef, workspace);
    bk::launch_pdl(mhc_coef_ernie_kernel, dim3(n_tokens), dim3(32), 0, st, (const float*) workspace, hc_scale, hc_base, flat,
                   rms_eps, hc_eps, sinkhorn_iters, coef);
}

void mhc_pre(const float* x, const float* coef, int coef_stride, int n_tokens, int n_embd, float* out, void* stream) {
    const int n4 = n_embd / 4;
    bk::launch_pdl(mhc_pre_kernel, dim3((n4 + 127) / 128, n_tokens), dim3(128), 0, (cudaStream_t) stream, x, coef, coef_stride,
                   n_embd, out);
}

void mhc_post(const float* f, const float* x, const float* coef, int coef_stride, int n_tokens, int n_embd,
              float* out, void* stream, const float* f2) {
    const int n4 = n_embd / 4;
    bk::launch_pdl(mhc_post_kernel, dim3((n4 + 127) / 128, n_tokens), dim3(128), 0, (cudaStream_t) stream, f, f2, x, coef,
                   coef_stride, n_embd, out);
}

void mhc_contract(const float* x, const float* hc_head_fn, const float* hc_head_scale, const float* hc_head_base,
                  int n_tokens, int n_embd, float rms_eps, float hc_eps, float* workspace, float* coef4, float* out,
                  void* stream) {
    const int flat = kHcMult * n_embd;
    cudaStream_t st = (cudaStream_t) stream;
    bk::launch_pdl(mhc_partial_kernel, dim3(kHcSplits, 4, n_tokens), dim3(kThreads), 0, st, x, hc_head_fn, flat, 4, workspace);
    bk::launch_pdl(mhc_contract_coef_kernel, dim3(n_tokens), dim3(32), 0, st, (const float*) workspace, hc_head_scale, hc_head_base,
                   flat, rms_eps, hc_eps, coef4);
    mhc_pre(x, coef4, 4, n_tokens, n_embd, out, stream);
}

void broadcast_streams(const float* x, int n_tokens, int n_embd, float* out, void* stream) {
    bk::launch_pdl(broadcast_streams_kernel, dim3((n_embd + 255) / 256, n_tokens), dim3(256), 0, (cudaStream_t) stream, x, n_embd, out);
}

void multimax_f32(float* logits, const float* ranges, const float* ts, int n_tokens, int n_vocab, void* stream) {
    if (n_tokens <= 0 || n_vocab <= 0) return;
    bk::launch_pdl(multimax_kernel, dim3((n_vocab + 255) / 256, n_tokens), dim3(256), 0, (cudaStream_t) stream, logits, ranges, ts,
                   n_vocab);
}

void situ_glu_f32(const float* gate, const float* up, int n_tokens, int n_ff, float beta, float linear_beta,
                  float* out, void* stream) {
    const int64_t n = (int64_t) n_tokens * n_ff;
    if (n <= 0) return;
    bk::launch_pdl(situ_glu_kernel, dim3((unsigned) ((n + 255) / 256)), dim3(256), 0, (cudaStream_t) stream, gate, up, n, beta,
                   linear_beta, out);
}

void l2_prefetch(const L2Spans& spans, void* stream) {
    if (spans.n <= 0) return;
    l2_prefetch_kernel<<<1, 256, 0, (cudaStream_t) stream>>>(spans);
}

void bf16_widen_f32(const uint16_t* in, float* out, int64_t n, void* stream) {
    if (n <= 0) return;
    bf16_widen_kernel<<<(unsigned) ((n + 1023) / 1024), 256, 0, (cudaStream_t) stream>>>(in, out, n);
}

void split_sigmoid_route(const float* logits, const float* bias, const float* scale, int n_tokens, int n_expert,
                         int top_k, bool renorm, int32_t* ids, float* weights, void* stream,
                         kernels::ResidentTabs held, float route_prior) {
    if (n_tokens <= 0) return;
    if (n_expert <= 512 && top_k <= n_expert) {
        bk::launch_pdl(split_sigmoid_route_sort_kernel, dim3(n_tokens), dim3(256), 0, (cudaStream_t) stream, logits, bias, scale,
                       n_expert, top_k, renorm, ids, weights, held, route_prior);
        return;
    }
    const size_t smem = (size_t) (2 * n_expert) * sizeof(float);
    bk::launch_pdl(split_sigmoid_route_kernel, dim3(n_tokens), dim3(256), smem, (cudaStream_t) stream, logits, bias, scale, n_expert,
                   top_k, renorm, ids, weights, held, route_prior);
}

void vha_postmix_f32(const float* in, const float* mix, int n_tokens, int n_heads, int head_dim, float* out,
                     void* stream, bool head_major) {
    if (n_tokens <= 0) return;
    bk::launch_pdl(vha_postmix_kernel, dim3((head_dim + 127) / 128, n_heads, n_tokens), dim3(128), 0, (cudaStream_t) stream, in, mix,
                   n_tokens, n_heads, head_dim, head_major, out);
}

void sigmoid_gate_mul_f32(float* z, const float* gate, int64_t n, void* stream) {
    if (n <= 0) return;
    bk::launch_pdl(sigmoid_gate_mul_kernel, dim3((unsigned) ((n + 255) / 256)), dim3(256), 0, (cudaStream_t) stream, z, gate, n);
}

void layernorm_f32(const float* x, const float* weight, const float* bias, int n_tokens, int dim, float eps,
                   float* out, void* stream) {
    if (n_tokens <= 0) return;
    const int threads = dim <= 128 ? 128 : 256;
    layernorm_kernel<<<n_tokens, threads, 0, (cudaStream_t) stream>>>(x, weight, bias, dim, eps, out);
}

void hadamard_128_f32(float* x, int n_rows, void* stream) {
    if (n_rows <= 0) return;
    hadamard_128_kernel<<<n_rows, 128, 0, (cudaStream_t) stream>>>(x);
}

void weighted_sum_f32(const float* exp_out, const float* weights, int n_tokens, int top_k, int dim, float* out,
                      void* stream) {
    if (n_tokens <= 0) return;
    bk::launch_pdl(weighted_sum_kernel, dim3((dim + 255) / 256, n_tokens), dim3(256), 0, (cudaStream_t) stream, exp_out, weights,
                   top_k, dim, out);
}

void weighted_sum_mapped_f32(const float* pair_rows, const int32_t* pair_map, const float* weights, int n_tokens,
                             int top_k, int dim, float* out, void* stream) {
    if (n_tokens <= 0) return;
    weighted_sum_mapped_kernel<<<dim3((dim + 255) / 256, n_tokens), 256, 0, (cudaStream_t) stream>>>(
        pair_rows, pair_map, weights, top_k, dim, out);
}

namespace {
// [T][G][w] <-> [G][T][w]: block (row, group) moves one row of w floats
__global__ void group_transpose_kernel(const float* __restrict__ src, int n_tokens, int n_groups, int w, bool to_groups,
                                       float* __restrict__ dst) {
    bk::pdl_wait();
    const int t = blockIdx.x, g = blockIdx.y;
    const float* s = src + (to_groups ? ((size_t) t * n_groups + g) : ((size_t) g * n_tokens + t)) * w;
    float* d = dst + (to_groups ? ((size_t) g * n_tokens + t) : ((size_t) t * n_groups + g)) * w;
    for (int c = threadIdx.x * 4; c < w; c += blockDim.x * 4)
        *reinterpret_cast<float4*>(d + c) = *reinterpret_cast<const float4*>(s + c);
}
}  // namespace

void hca_groups_split_f32(const float* src, int n_tokens, int n_groups, int gin, float* dst, void* stream) {
    if (n_tokens <= 0) return;
    bk::launch_pdl(group_transpose_kernel, dim3(n_tokens, n_groups), dim3(256), 0, (cudaStream_t) stream, src, n_tokens, n_groups, gin,
                   true, dst);
}

void hca_groups_merge_f32(const float* src, int n_tokens, int n_groups, int gout, float* dst, void* stream) {
    if (n_tokens <= 0) return;
    bk::launch_pdl(group_transpose_kernel, dim3(n_tokens, n_groups), dim3(256), 0, (cudaStream_t) stream, src, n_tokens, n_groups, gout,
                   false, dst);
}

void hca_group_split_f32(const float* src, int n_tokens, int n_groups, int grp, int gin, float* dst, void* stream) {
    if (n_tokens <= 0) return;
    hca_group_split_kernel<<<dim3((gin + 255) / 256, n_tokens), 256, 0, (cudaStream_t) stream>>>(src, n_groups, grp,
                                                                                                 gin, dst);
}

void hca_group_merge_f32(const float* src, int n_tokens, int n_groups, int grp, int gout, float* dst, void* stream) {
    if (n_tokens <= 0) return;
    hca_group_merge_kernel<<<dim3((gout + 255) / 256, n_tokens), 256, 0, (cudaStream_t) stream>>>(src, n_groups, grp,
                                                                                                  gout, dst);
}

void add_f32(float* dst, const float* src, int64_t n, void* stream) {
    if (n <= 0) return;
    bk::launch_pdl(add_f32_kernel, dim3((unsigned) ((n + 255) / 256)), dim3(256), 0, (cudaStream_t) stream, dst, src, n);
}

// ---- decode windows' fused launches (the bits of the sequences they replace) --------------------------------------

namespace {

struct Q81 {   // native_mmvq's q8_1 block: scale and sum as half2, 32 values
    half2 ds;
    int8_t qs[32];
};

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m));
    return v;
}

// native_mmvq.cu is built with --use_fast_math: its divisions are div.approx.ftz (gdn_window.cu's emulation)
__device__ __forceinline__ float div_approx_ftz(float a, float b) {
    float r;
    asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ int8_t q8_quant_fm(const float xi, const float d, const float amax) {
    if (amax == 0.0f) return 0;
    const float q = roundf(div_approx_ftz(xi, d));
    return (int8_t) (q > 127.0f ? 127.0f : q < -127.0f ? -127.0f : q);
}

// Element e of a q8_1 vector, its warp holding the 32 elements of e's block in lane order: native_quantize_q8_1's bits.
__device__ __forceinline__ void q8_store(float v, Q81* y, size_t e) {
    const float amax = warp_max(fabsf(v));
    const float sum = warp_sum(v);
    const float d = bk::q8_1_finite(div_approx_ftz(amax, 127.0f));
    y[e / 32].qs[e % 32] = q8_quant_fm(v, d, amax);
    if ((e & 31) == 0) y[e / 32].ds = bk::q8_1_ds(d, sum);
}

constexpr int kNormThreads = 256;   // row_norm.cu's rms_norm_rows: a 256-thread block, float4 columns i + 256 c
constexpr int kNormVec = 8;         // float4 a thread at most (cols <= 8192)

// Four consecutive elements e0.. of a q8_1 vector, the 8 lanes of an aligned group holding a block's 32 in lane order:
// native_quantize_q8_1's bits (its sum's butterfly: lanes 4 apart are its offset 16, 2 apart 8, 1 apart 4, then the
// components 2 and 1 apart).
__device__ __forceinline__ void q8_store4(float4 y, Q81* q, size_t e0) {
    float amax = fmaxf(fmaxf(fabsf(y.x), fabsf(y.y)), fmaxf(fabsf(y.z), fabsf(y.w)));
    float s0 = y.x, s1 = y.y, s2 = y.z, s3 = y.w;
#pragma unroll
    for (int m = 4; m; m >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, m));
        s0 += __shfl_xor_sync(0xffffffffu, s0, m);
        s1 += __shfl_xor_sync(0xffffffffu, s1, m);
        s2 += __shfl_xor_sync(0xffffffffu, s2, m);
        s3 += __shfl_xor_sync(0xffffffffu, s3, m);
    }
    const float sum = (s0 + s2) + (s1 + s3);
    const float d = bk::q8_1_finite(div_approx_ftz(amax, 127.0f));
    Q81& b = q[e0 / 32];
    const size_t k = e0 % 32;
    b.qs[k] = q8_quant_fm(y.x, d, amax);
    b.qs[k + 1] = q8_quant_fm(y.y, d, amax);
    b.qs[k + 2] = q8_quant_fm(y.z, d, amax);
    b.qs[k + 3] = q8_quant_fm(y.w, d, amax);
    if (k == 0) b.ds = bk::q8_1_ds(d, sum);
}

// row_norm.cu's rms_norm_rows of the row this block holds (thread i: float4 columns i + 256 c, c < cols / 1024), into
// out (if set) and the q8_1 row q (if set). cols a multiple of 1024.
__device__ __forceinline__ void norm_finish(const float4 (&v)[kNormVec], int cols, const float* w, float eps, float* out,
                                            Q81* q) {
    __shared__ float red[kNormThreads / 32];
    const int nv = cols / (4 * kNormThreads);
    float ss = 0.f;
    for (int c = 0; c < nv; ++c) {
        ss = fmaf(v[c].x, v[c].x, ss);
        ss = fmaf(v[c].y, v[c].y, ss);
        ss = fmaf(v[c].z, v[c].z, ss);
        ss = fmaf(v[c].w, v[c].w, ss);
    }
    ss = warp_sum(ss);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    float t = 0.f;
#pragma unroll
    for (int i = 0; i < kNormThreads / 32; ++i) t += red[i];
    const float scale = rsqrtf(t / (float) cols + eps);
    for (int c = 0; c < nv; ++c) {
        const int i = threadIdx.x + c * kNormThreads;
        const float4 wv = make_float4(w[4 * i], w[4 * i + 1], w[4 * i + 2], w[4 * i + 3]);
        const float4 y = make_float4(v[c].x * scale * wv.x, v[c].y * scale * wv.y, v[c].z * scale * wv.z, v[c].w * scale * wv.w);
        if (out) reinterpret_cast<float4*>(out)[i] = y;
        if (q) q8_store4(y, q, (size_t) 4 * i);
    }
}

// mhc_mixes + mhc_pre + rms_norm_rows (+ native_quantize_q8_1) of a decode window: the blocks of mhc_partial_kernel; the
// last of a row's blocks to finish (cnt[t], which it resets) runs mhc_coef_ernie_kernel's warp, mhc_pre_kernel's sums
// and norm_finish for the row.
__global__ void __launch_bounds__(kThreads) mhc_pre_norm_kernel(const float* x, const float* __restrict__ fn,
                                                                const float* __restrict__ scale,
                                                                const float* __restrict__ base, int n_embd, float rms_eps,
                                                                float hc_eps, int sinkhorn_iters,
                                                                const float* __restrict__ norm_w, float norm_eps,
                                                                float* ws, unsigned* cnt, float* coef, float* x_norm,
                                                                Q81* xq) {
    bk::pdl_wait();
    __shared__ float red[kThreads / 32];
    __shared__ float cf[kHcCoef], mix[kHcCoef];
    __shared__ bool last;
    const int flat = kHcMult * n_embd;
    const int split = blockIdx.x, m = blockIdx.y, t = blockIdx.z;
    {   // mhc_partial_kernel
        const int len = flat / kHcSplits, b0 = split * len;
        const float4* xv = reinterpret_cast<const float4*>(x + (size_t) t * flat + b0);
        const float4* wv = reinterpret_cast<const float4*>(fn + (size_t) m * flat + b0);
        float dot = 0.f, ss = 0.f;
        for (int i = threadIdx.x; i < len / 4; i += kThreads) {
            const float4 a = xv[i], w = wv[i];
            dot = fmaf(a.x, w.x, dot); dot = fmaf(a.y, w.y, dot);
            dot = fmaf(a.z, w.z, dot); dot = fmaf(a.w, w.w, dot);
            if (m == 0) {
                ss = fmaf(a.x, a.x, ss); ss = fmaf(a.y, a.y, ss);
                ss = fmaf(a.z, a.z, ss); ss = fmaf(a.w, a.w, ss);
            }
        }
        dot = block_sum(dot, red);
        if (m == 0) ss = block_sum(ss, red);
        if (threadIdx.x == 0) {
            float* part = ws + ((size_t) t * kHcSplits + split) * kSlots;
            part[m] = dot;
            if (m == 0) part[kHcCoef] = ss;
            __threadfence();
            last = atomicAdd(cnt + t, 1u) == (unsigned) (kHcSplits * kHcCoef - 1);
        }
        __syncthreads();
        if (!last) return;
        __threadfence();
    }
    if (threadIdx.x < 32) {   // mhc_coef_ernie_kernel
        const int lane = threadIdx.x;
        const float* w = ws + (size_t) t * kHcSplits * kSlots;
        float ss = 0.f;
        for (int s = 0; s < kHcSplits; ++s) ss += __ldcg(w + (size_t) s * kSlots + kHcCoef);
        const float inv_rms = 1.0f / (sqrtf(ss / (float) flat) + rms_eps);
        if (lane < kHcCoef) {
            float acc = 0.f;
            for (int s = 0; s < kHcSplits; ++s) acc += __ldcg(w + (size_t) s * kSlots + lane);
            mix[lane] = acc * inv_rms;
        }
        __syncwarp();
        float* out = coef + (size_t) t * kHcCoef;
        if (lane < 4) {
            cf[lane] = out[lane] = sigmoidf_ieee(mix[lane] * scale[0] + base[lane]) + hc_eps;
        } else if (lane < 8) {
            cf[lane] = out[lane] = sigmoidf_ieee(mix[lane] * scale[1] + base[lane]) * 2.0f;
        }
        const int r = (lane >> 2) & 3, c = lane & 3;
        const unsigned all = 0xffffffffu;
        float v = mix[8 + (lane & 15)] * scale[2] + base[8 + (lane & 15)];
        const auto row = [&](float y, int j) { return __shfl_sync(all, y, (lane & 16) | (r << 2) | j); };
        const auto col = [&](float y, int i) { return __shfl_sync(all, y, (lane & 16) | (i << 2) | c); };
        {
            const float mx = fmaxf(fmaxf(row(v, 0), row(v, 1)), fmaxf(row(v, 2), row(v, 3)));
            v = expf(v - mx);
            float s = 0.f;
            s += row(v, 0);
            s += row(v, 1);
            s += row(v, 2);
            s += row(v, 3);
            const float inv = 1.0f / s;
            v = v * inv + hc_eps;
        }
        {
            const float cs = col(v, 0) + col(v, 1) + col(v, 2) + col(v, 3) + hc_eps;
            v *= 1.0f / cs;
        }
        for (int it = 1; it < sinkhorn_iters; ++it) {
            const float rs = row(v, 0) + row(v, 1) + row(v, 2) + row(v, 3) + hc_eps;
            v *= 1.0f / rs;
            const float cs = col(v, 0) + col(v, 1) + col(v, 2) + col(v, 3) + hc_eps;
            v *= 1.0f / cs;
        }
        if (lane < 16) cf[8 + lane] = out[8 + lane] = v;
        if (lane == 0) cnt[t] = 0;
    }
    __syncthreads();
    // mhc_pre_kernel's sums for the columns this thread normalizes
    const float* xt = x + (size_t) t * flat;
    const int nv = n_embd / (4 * kNormThreads);
    float4 v[kNormVec];
    for (int c = 0; c < nv; ++c) {
        const int i = threadIdx.x + c * kNormThreads;
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int s = 0; s < kHcMult; ++s) {
            const float4 xv = reinterpret_cast<const float4*>(xt + (size_t) s * n_embd)[i];
            acc.x = fmaf(cf[s], xv.x, acc.x); acc.y = fmaf(cf[s], xv.y, acc.y);
            acc.z = fmaf(cf[s], xv.z, acc.z); acc.w = fmaf(cf[s], xv.w, acc.w);
        }
        v[c] = acc;
    }
    norm_finish(v, n_embd, norm_w, norm_eps, x_norm + (size_t) t * n_embd,
                xq ? xq + (size_t) t * (n_embd / 32) : nullptr);
}

// rms_norm_rows (+ native_quantize_q8_1) of row blockIdx.x of x, or with `wts`, of weighted_sum_kernel's row (x then
// holds the row's top_k expert outputs).
__global__ void __launch_bounds__(kNormThreads) norm_q8_kernel(const float* x, const float* wts, int top_k,
                                                               const float* __restrict__ w, int cols, float eps,
                                                               float* out, Q81* xq) {
    bk::pdl_wait();
    const int t = blockIdx.x, nv = cols / (4 * kNormThreads);
    float4 v[kNormVec];
    for (int c = 0; c < nv; ++c) {
        const int i = threadIdx.x + c * kNormThreads;
        if (wts) {
            const float* wt = wts + (size_t) t * top_k;
            const float* eb = x + (size_t) t * top_k * cols + 4 * i;
            float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int j = 0; j < top_k; ++j) {
                const float wj = wt[j];
                if (wj != 0.0f)
#pragma unroll
                    for (int k = 0; k < 4; ++k) a[k] = fmaf(wj, eb[(size_t) j * cols + k], a[k]);
            }
            v[c] = make_float4(a[0], a[1], a[2], a[3]);
        } else {
            v[c] = reinterpret_cast<const float4*>(x + (size_t) t * cols)[i];
        }
    }
    norm_finish(v, cols, w, eps, out ? out + (size_t) t * cols : nullptr, xq ? xq + (size_t) t * (cols / 32) : nullptr);
}

// situ_glu_kernel + native_quantize_q8_1 of its n values (n a multiple of 32).
__global__ void situ_glu_q8_kernel(const float* gate, const float* up, int64_t n, float beta, float linear_beta, Q81* y) {
    bk::pdl_wait();
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i], u = up[i];
    const float sg = __fmul_rn(beta * tanhf(g / beta), sigmoidf_ieee(g));
    const float su = linear_beta > 0.0f ? (linear_beta * tanhf(u / linear_beta)) : u;
    q8_store(__fmul_rn(sg, su), y, (size_t) i);
}

// hca_groups_merge_f32 ([G][T][gout] -> [T][G][gout]), sigmoid_gate_mul_kernel and native_quantize_q8_1 of the T rows
// of G * gout values (G = 1: the gate and the quantization alone).
__global__ void merge_gate_q8_kernel(const float* src, const float* gate, int n_tokens, int n_groups, int gout, Q81* y) {
    bk::pdl_wait();
    const int64_t e = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t row = (int64_t) n_groups * gout;
    if (e >= row * n_tokens) return;
    const int t = (int) (e / row), r = (int) (e % row), g = r / gout, c = r % gout;
    float z = src[((size_t) g * n_tokens + t) * gout + c];
    z *= sigmoidf_ieee(gate[e]);
    q8_store(z, y, (size_t) e);
}

}  // namespace

void mhc_pre_norm(const float* x, const float* hc_fn, const float* hc_scale, const float* hc_base, int n_tokens,
                  int n_embd, float rms_eps, float hc_eps, int sinkhorn_iters, const float* norm_w, float norm_eps,
                  float* workspace, unsigned* counters, float* coef, float* x_norm, void* xq, void* stream) {
    if (n_tokens <= 0) return;
    if ((kHcMult * n_embd) % (kHcSplits * 4) || n_embd % (4 * kNormThreads) || n_embd > 4 * kNormThreads * kNormVec ||
        sinkhorn_iters < 1)
        throw std::invalid_argument("ernie55 mhc_pre_norm: geometry");
    bk::launch_pdl(mhc_pre_norm_kernel, dim3(kHcSplits, kHcCoef, n_tokens), dim3(kThreads), 0, (cudaStream_t) stream, x,
                   hc_fn, hc_scale, hc_base, n_embd, rms_eps, hc_eps, sinkhorn_iters, norm_w, norm_eps, workspace,
                   counters, coef, x_norm, (Q81*) xq);
}

void rms_norm_q8(const float* x, const float* w, int n_tokens, int cols, float eps, float* out, void* xq, void* stream,
                 const float* wts, int top_k) {
    if (n_tokens <= 0) return;
    if (cols % (4 * kNormThreads) || cols > 4 * kNormThreads * kNormVec) throw std::invalid_argument("ernie55 rms_norm_q8: cols");
    bk::launch_pdl(norm_q8_kernel, dim3(n_tokens), dim3(kNormThreads), 0, (cudaStream_t) stream, x, wts, top_k, w, cols, eps, out,
                   (Q81*) xq);
}

void situ_glu_q8(const float* gate, const float* up, int n_tokens, int n_ff, float beta, float linear_beta, void* xq,
                 void* stream) {
    const int64_t n = (int64_t) n_tokens * n_ff;
    if (n <= 0) return;
    if (n_ff % 32) throw std::invalid_argument("ernie55 situ_glu_q8: n_ff");
    bk::launch_pdl(situ_glu_q8_kernel, dim3((unsigned) ((n + 255) / 256)), dim3(256), 0, (cudaStream_t) stream, gate, up, n,
                   beta, linear_beta, (Q81*) xq);
}

void merge_gate_q8(const float* src, const float* gate, int n_tokens, int n_groups, int gout, void* xq, void* stream) {
    const int64_t n = (int64_t) n_tokens * n_groups * gout;
    if (n <= 0) return;
    if (gout % 32) throw std::invalid_argument("ernie55 merge_gate_q8: gout");
    bk::launch_pdl(merge_gate_q8_kernel, dim3((unsigned) ((n + 255) / 256)), dim3(256), 0, (cudaStream_t) stream, src, gate,
                   n_tokens, n_groups, gout, (Q81*) xq);
}

}  // namespace basalt::models::ernie55
