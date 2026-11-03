// src/models/deepseek41/prefill_ffn.cu - the clamped SwiGLU between the routed experts' two MMQ products.
#include "models/deepseek41/prefill_ffn.hpp"

#include <cuda_runtime.h>

namespace basalt::models::deepseek41 {
namespace {

// h[r][k] = silu(min(gate, limit)) * clamp(up, -limit, limit); gate = gu[r][k], up = gu[r][n_ff + k]. TAILS: gate and up
// times row r's expert's s_gate and s_up first, h times its s_down.
template <bool TAILS>
__global__ void k_swiglu_split(const float* gu, float* h, int64_t n, int n_ff, float limit, ExpertTails t) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t r = i / n_ff;
    const int k = (int) (i - r * n_ff);
    float g = gu[r * 2 * n_ff + k], u = gu[r * 2 * n_ff + n_ff + k];
    float sd = 1.f;
    if constexpr (TAILS) {
        int q = 0;   // the row's expert: the last q with bounds[q] <= r
        while (q + 1 < t.n && t.bounds[q + 1] <= r) ++q;
        const float* s = (const float*) (t.blobs + (size_t) q * t.stride + t.tail_off);
        g *= s[0];
        u *= s[1];
        sd = s[2];
    }
    g = fminf(g, limit);
    u = fminf(fmaxf(u, -limit), limit);
    if constexpr (TAILS) h[i] = g / (1.f + __expf(-g)) * u * sd;
    else h[i] = g / (1.f + __expf(-g)) * u;
}

__global__ void k_wsum_tokens(const float* rows, const int32_t* map, const float* w, int n_rows, int n, float* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const size_t t = blockIdx.y;
    const int32_t* mt = map + t * n_rows;
    const float* wt = w + t * n_rows;
    float s = 0.f;
    for (int k = 0; k < n_rows; ++k) s += wt[k] * rows[(size_t) mt[k] * n + i];
    out[t * n + i] = s;
}

}  // namespace

void weighted_sum_tokens(const float* rows, const int32_t* map, const float* w, int n_tokens, int n_rows, int n, float* out,
                         void* stream) {
    if (n_tokens <= 0) return;
    k_wsum_tokens<<<dim3((n + 255) / 256, n_tokens), 256, 0, (cudaStream_t) stream>>>(rows, map, w, n_rows, n, out);
}

void swiglu_split_clamped(const float* gu, float* h, int64_t rows, int n_ff, float limit, void* stream,
                          const ExpertTails* tails) {
    const int64_t n = rows * n_ff;
    if (n <= 0) return;
    const unsigned blocks = (unsigned) ((n + 255) / 256);
    if (tails && tails->n > 0)
        k_swiglu_split<true><<<blocks, 256, 0, (cudaStream_t) stream>>>(gu, h, n, n_ff, limit, *tails);
    else
        k_swiglu_split<false><<<blocks, 256, 0, (cudaStream_t) stream>>>(gu, h, n, n_ff, limit, ExpertTails{});
}

}  // namespace basalt::models::deepseek41
