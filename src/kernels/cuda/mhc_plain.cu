// Hyper-connections without a cross-stream mix (identity connections): the mixing coefficients of a sublayer
// (projection and sigmoids, no softmax and no Sinkhorn) and of the final head.
#include "basalt/kernels/mhc.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

namespace basalt::kernels {
namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

// Block sum in a fixed order; every thread gets the result.
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

// Stage: partial dot of one hc_fn row slice with one token's flattened streams (and, for mix 0, the sum of squares).
// A token's partials are kHcSplits rows of n_mix dot products and the sum of squares.
__global__ void plain_partial_kernel(const float* __restrict__ x, const float* __restrict__ fn, int flat, int n_mix,
                                     float* __restrict__ ws) {
    __shared__ float red[kThreads / 32];
    const int split = blockIdx.x, mix = blockIdx.y, t = blockIdx.z;
    const int len = flat / kHcSplits, base = split * len;
    const float4* xv = reinterpret_cast<const float4*>(x + (size_t) t * flat + base);
    const float4* wv = reinterpret_cast<const float4*>(fn + (size_t) mix * flat + base);
    float dot = 0.f, ss = 0.f;
    for (int i = threadIdx.x; i < len / 4; i += kThreads) {
        const float4 a = xv[i], w = wv[i];
        dot = fmaf(a.x, w.x, dot); dot = fmaf(a.y, w.y, dot); dot = fmaf(a.z, w.z, dot); dot = fmaf(a.w, w.w, dot);
        if (mix == 0) { ss = fmaf(a.x, a.x, ss); ss = fmaf(a.y, a.y, ss); ss = fmaf(a.z, a.z, ss); ss = fmaf(a.w, a.w, ss); }
    }
    dot = block_sum(dot, red);
    if (mix == 0) ss = block_sum(ss, red);
    if (threadIdx.x == 0) {
        float* part = ws + ((size_t) t * kHcSplits + split) * (n_mix + 1);
        part[mix] = dot;
        if (mix == 0) part[n_mix] = ss;
    }
}

__device__ __forceinline__ float sigmoidf(float v) { return 1.f / (1.f + expf(-v)); }

// One token, one warp: lane m owns mix m.
__global__ void plain_coef_kernel(const float* __restrict__ ws, const float* __restrict__ scale,
                                  const float* __restrict__ base, int flat, int n_mix, float rms_eps, float hc_eps,
                                  float magnitude, float* __restrict__ coef, int coef_stride) {
    const int t = blockIdx.x, lane = threadIdx.x & 31;
    const float* w = ws + (size_t) t * kHcSplits * (n_mix + 1);
    float ss = 0.f;
    for (int s = 0; s < kHcSplits; ++s) ss += w[(size_t) s * (n_mix + 1) + n_mix];
    const float rs = rsqrtf(ss / (float) flat + rms_eps);
    float m = 0.f;
    if (lane < n_mix) {
        for (int s = 0; s < kHcSplits; ++s) m += w[(size_t) s * (n_mix + 1) + lane];
        m *= rs;
    }
    float* out = coef + (size_t) t * coef_stride;
    if (n_mix == 8) {
        if (lane < 4) out[lane] = sigmoidf(m * scale[0] + base[lane]) + hc_eps;
        else if (lane < 8) out[lane] = magnitude * sigmoidf(m * scale[1] + base[lane]) + hc_eps;
        if (lane >= 8 && lane < kHcCoef) out[lane] = ((lane - 8) >> 2) == ((lane - 8) & 3) ? 1.f : 0.f;
    } else if (lane < n_mix) {
        out[lane] = sigmoidf(m * scale[0] + base[lane]) + hc_eps;
    }
}

}  // namespace

void hc_mixes_plain(const float* x, const float* hc_fn, const float* hc_scale, const float* hc_base, int n_tokens,
                    int n_mix, const HcPlainArgs& a, float* workspace, float* coef, void* stream) {
    const int flat = kHcMult * a.n_embd;
    if (flat % (kHcSplits * 4) || (n_mix != 8 && n_mix != 4)) throw std::invalid_argument("hc_mixes_plain: geometry");
    cudaStream_t st = (cudaStream_t) stream;
    plain_partial_kernel<<<dim3(kHcSplits, n_mix, n_tokens), kThreads, 0, st>>>(x, hc_fn, flat, n_mix, workspace);
    plain_coef_kernel<<<n_tokens, 32, 0, st>>>(workspace, hc_scale, hc_base, flat, n_mix, a.rms_eps, a.hc_eps,
                                               a.magnitude, coef, hc_plain_coef_stride(n_mix));
    if (!stream) cudaStreamSynchronize(nullptr);
}

}  // namespace basalt::kernels
