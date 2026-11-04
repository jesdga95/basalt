// DeepSeek-V4.1 row norms: a weighted RMSNorm over one fp32 row per block, and the same norm fused behind the
// hyper-connection collapse (hc_pre), so the collapsed row never makes a round trip through memory before it is
// normalised.
#include "basalt/models/deepseek41/kernels.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

namespace basalt::models::deepseek41 {
namespace {

constexpr int kThreads = 256;
constexpr int kMaxVec = 8;    // float4 per thread: n <= 4 * kMaxVec * kThreads

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

// Stage: normalise the row held in v (this thread's float4 columns threadIdx.x + c*kThreads) and write it. The sum
// of squares runs warp then block in a fixed order.
__device__ __forceinline__ void rms_scale_store(float4 (&v)[kMaxVec], int n, const float* __restrict__ w, float eps,
                                                float* __restrict__ out, float* red) {
    const int n4 = n >> 2;
    float ss = 0.f;
#pragma unroll
    for (int c = 0; c < kMaxVec; ++c) {
        if (threadIdx.x + c * kThreads < n4) {
            ss = fmaf(v[c].x, v[c].x, ss);
            ss = fmaf(v[c].y, v[c].y, ss);
            ss = fmaf(v[c].z, v[c].z, ss);
            ss = fmaf(v[c].w, v[c].w, ss);
        }
    }
    ss = warp_sum(ss);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    float t = 0.f;
#pragma unroll
    for (int i = 0; i < kThreads / 32; ++i) t += red[i];
    const float scale = rsqrtf(t / (float) n + eps);
#pragma unroll
    for (int c = 0; c < kMaxVec; ++c) {
        const int i = threadIdx.x + c * kThreads;
        if (i < n4) {
            // the weight row has no 16-byte alignment guarantee
            const float4 wv = make_float4(w[4 * i], w[4 * i + 1], w[4 * i + 2], w[4 * i + 3]);
            reinterpret_cast<float4*>(out)[i] =
                make_float4(v[c].x * scale * wv.x, v[c].y * scale * wv.y, v[c].z * scale * wv.z, v[c].w * scale * wv.w);
        }
    }
}

__global__ void __launch_bounds__(kThreads)
rms_norm_rows_kernel(const float* __restrict__ x, const float* __restrict__ w, int n, float eps,
                     float* __restrict__ out) {
    __shared__ float red[kThreads / 32];
    const int n4 = n >> 2;
    const float4* xr = reinterpret_cast<const float4*>(x + (size_t) blockIdx.x * n);
    float4 v[kMaxVec];
#pragma unroll
    for (int c = 0; c < kMaxVec; ++c) {
        const int i = threadIdx.x + c * kThreads;
        if (i < n4) v[c] = xr[i];
    }
    rms_scale_store(v, n, w, eps, out + (size_t) blockIdx.x * n, red);
}

// Stage: collapse the four hyper-connection streams of one token with pre[0..3] (the arithmetic of hc_pre), store
// the collapsed row, and normalise it.
__global__ void __launch_bounds__(kThreads)
hc_pre_rms_norm_kernel(const float* __restrict__ x, const float* __restrict__ pre, int pre_stride,
                       const float* __restrict__ w, int n, float eps, float* __restrict__ cur,
                       float* __restrict__ out) {
    __shared__ float red[kThreads / 32];
    const int t = blockIdx.x, n4 = n >> 2;
    float p[kHcMult];
#pragma unroll
    for (int s = 0; s < kHcMult; ++s) p[s] = pre[(size_t) t * pre_stride + s];
    const float4* xv = reinterpret_cast<const float4*>(x + (size_t) t * kHcMult * n);
    float4 v[kMaxVec];
#pragma unroll
    for (int c = 0; c < kMaxVec; ++c) {
        const int i = threadIdx.x + c * kThreads;
        if (i < n4) {
            float4 acc = xv[i];
            acc.x *= p[0]; acc.y *= p[0]; acc.z *= p[0]; acc.w *= p[0];
#pragma unroll
            for (int s = 1; s < kHcMult; ++s) {
                const float4 q = xv[(size_t) s * n4 + i];
                acc.x = fmaf(q.x, p[s], acc.x); acc.y = fmaf(q.y, p[s], acc.y);
                acc.z = fmaf(q.z, p[s], acc.z); acc.w = fmaf(q.w, p[s], acc.w);
            }
            v[c] = acc;
            reinterpret_cast<float4*>(cur + (size_t) t * n)[i] = acc;
        }
    }
    rms_scale_store(v, n, w, eps, out + (size_t) t * n, red);
}

void check_geometry(int n) {
    if (n % 4 || n > 4 * kMaxVec * kThreads) throw std::invalid_argument("deepseek41 norm: unsupported width");
}

}  // namespace

void rms_norm_rows(const float* x, const float* w, int n_tokens, int n, float eps, float* out, void* stream) {
    check_geometry(n);
    rms_norm_rows_kernel<<<n_tokens, kThreads, 0, (cudaStream_t) stream>>>(x, w, n, eps, out);
    if (!stream) cudaStreamSynchronize(nullptr);
}

void hc_pre_rms_norm(const float* x, const float* pre, int pre_stride, const float* w, int n_tokens, int n, float eps,
                     float* cur, float* out, void* stream) {
    check_geometry(n);
    hc_pre_rms_norm_kernel<<<n_tokens, kThreads, 0, (cudaStream_t) stream>>>(x, pre, pre_stride, w, n, eps, cur, out);
    if (!stream) cudaStreamSynchronize(nullptr);
}

}  // namespace basalt::models::deepseek41
