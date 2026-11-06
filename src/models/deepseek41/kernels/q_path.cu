// DeepSeek-V4.1 query path: wq_a, q_norm, wq_b, RoPE on the last 64 dims of each head (build_attention41 in the
// llama.cpp DeepSeek-V4.1 support).
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/models/deepseek41/attn.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

namespace basalt::models::deepseek41 {
namespace {

// y = x * rsqrt(mean(x^2) + eps) * w, in place, one block per row.
__global__ void __launch_bounds__(256) rms_norm_rows_kernel(float* x, const float* w, int n, float eps) {
    float* row = x + (size_t) blockIdx.x * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += 256) ss = fmaf(row[i], row[i], ss);
#pragma unroll
    for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    __shared__ float part[8];
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) tot += part[i];
    const float scale = rsqrtf(tot / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += 256) row[i] = row[i] * scale * w[i];
}

}  // namespace

size_t q_path_scratch_bytes(int n_tokens) { return kernels::native_q8_1_bytes(kAttnQRank, n_tokens); }

void q_path(const void* x_q8_1, const QPathWeights& w, int n_tokens, const int32_t* pos, const RopeTable& rope,
            void* xq_scratch, float* qr, float* q, void* stream) {
    if (n_tokens < 1 || n_tokens > kAttnMaxTokens) throw std::invalid_argument("q_path: n_tokens");
    kernels::native_mmvq(w.wq_a_type, w.wq_a, x_q8_1, qr, kAttnEmbd, kAttnQRank, n_tokens, stream);
    rms_norm_rows_kernel<<<n_tokens, 256, 0, (cudaStream_t) stream>>>(qr, w.q_norm, kAttnQRank, w.rms_eps);
    kernels::native_quantize_q8_1(qr, xq_scratch, kAttnQRank, n_tokens, stream);
    kernels::native_mmvq(w.wq_b_type, w.wq_b, xq_scratch, q, kAttnQRank, kAttnHeads * kAttnHeadDim, n_tokens, stream);
    rope_rows(q, n_tokens * kAttnHeads, kAttnHeadDim, kAttnHeads, pos, rope, false, stream);
}

}  // namespace basalt::models::deepseek41
