// src/models/glm5next/kernels/kda_front.cu - see kda_front.hpp: the KDA front for one token (conv+SiLU, q/k L2 norm,
// the per-channel decay and beta).
//
// One block per head, one thread per channel. The thread convolves its channel in all three streams (the window is
// the block's shared history of 3 positions plus this token), SiLU's it, and takes the head's L2 sums by warp
// butterfly. Then the gates: the decay is exp(kGateLowerBound * sigmoid(-a[head] * (a + dt))) - the bounded smooth
// form of the reference with a gate lower bound (a = ssm_a = -exp(A_log), per head), beta is sigmoid.
// hc <= 1024 (the model's is 128).
#include "models/glm5next/kernels/kda_front.hpp"

#include <cuda_runtime.h>

namespace basalt::models::glm5next {
namespace {

__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

__global__ void kda_front_kernel(const float* __restrict__ x, size_t x_stride, const float* __restrict__ hist,
                                 const float* __restrict__ conv, const float* __restrict__ a, const float* __restrict__ dt,
                                 const float* __restrict__ a_neg, const float* __restrict__ b_raw, float* __restrict__ qn,
                                 float* __restrict__ kn, float* __restrict__ vout, float* __restrict__ g,
                                 float* __restrict__ beta, int hc) {
    const int head = blockIdx.x, ch = threadIdx.x;
    const int c = head * hc + ch;
    const int n = gridDim.x * hc;

    float y[3];
#pragma unroll
    for (int s = 0; s < 3; ++s) {  // q, k, v: causal width-4 conv over [hist(3) | x], then SiLU. Each stream is
        const float* xs = x + (size_t) s * x_stride;   // n channels; conv taps stride by n (the GGUF's [4, 1,
        const float* hs = hist + (size_t) s * n;       // channels] layout); hist is 3 rows of the whole 3-stream
        const float* wv = conv + (size_t) s * 4 * n;   // token (row stride 3n), oldest first.
        float acc = wv[3 * n + c] * xs[c];  // tap 3 is the current token
        acc = fmaf(wv[2 * n + c], hs[2 * 3 * n + c], acc);
        acc = fmaf(wv[1 * n + c], hs[1 * 3 * n + c], acc);
        acc = fmaf(wv[0 * n + c], hs[0 * 3 * n + c], acc);
        y[s] = acc / (1.0f + expf(-acc));  // SiLU
    }

    // the head's L2 sums, folded in warp butterfly order
    float sq = y[0] * y[0], sk = y[1] * y[1];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        sq += __shfl_down_sync(0xffffffffu, sq, o);
        sk += __shfl_down_sync(0xffffffffu, sk, o);
    }
    __shared__ float ssum_q[32], ssum_k[32], snorm_q, snorm_k;
    const int lane = ch & 31, warp = ch >> 5;
    const int nwarps = (hc + 31) >> 5;
    if (lane == 0) { ssum_q[warp] = sq; ssum_k[warp] = sk; }
    __syncthreads();
    if (ch == 0) {
        float aq = 0.0f, ak = 0.0f;
        for (int w = 0; w < nwarps; ++w) { aq += ssum_q[w]; ak += ssum_k[w]; }
        // the FLA l2norm: x / sqrt(sum + kL2Eps) - the epsilon is inside the root (modeling_kimi_linear.l2norm)
        snorm_q = rsqrtf(aq + kL2Eps);
        snorm_k = rsqrtf(ak + kL2Eps);
    }
    __syncthreads();
    const float qscale = rsqrtf((float) hc);
    qn[c] = y[0] * snorm_q * qscale;
    kn[c] = y[1] * snorm_k;
    vout[c] = y[2];

    // the reference's bounded decay with a gate lower bound (llama.cpp glm5-next): g = lb * sigmoid(-(ssm_a * S))
    // with S = a + dt and ssm_a = -exp(A_log), so the exponent argument is exp(A_log) * S
    const float gr = kGateLowerBound * sigmoid_f(-(a_neg[head] * (a[c] + dt[c])));
    g[c] = expf(gr);
    if (ch == 0) beta[head] = sigmoid_f(b_raw[head]);
}

}  // namespace

void kda_front(const float* x, size_t x_stride, const float* hist, const float* conv, const float* a, const float* dt,
               const float* a_neg, const float* b_raw, float* qn, float* kn, float* vout, float* g, float* beta,
               int n_heads, int hc, cudaStream_t stream) {
    if (n_heads <= 0 || hc <= 0 || hc > 1024) return;
    kda_front_kernel<<<n_heads, hc, 0, stream>>>(x, x_stride, hist, conv, a, dt, a_neg, b_raw, qn, kn, vout, g, beta,
                                                 hc);
}

}  // namespace basalt::models::glm5next
