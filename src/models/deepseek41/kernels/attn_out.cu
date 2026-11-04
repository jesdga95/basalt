// DeepSeek-V4.1 grouped low-rank attention output: wo_a block-diagonal over 8 groups, then wo_b (build_attention41
// in the llama.cpp DeepSeek-V4.1 support).
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
#include "attn_out_grouped.hpp"

#include <algorithm>
#include <stdexcept>

namespace basalt::models::deepseek41 {

constexpr int kAttnOutWidth = kAttnGroups * kAttnOutRank;   // 8192

size_t attn_out_scratch_bytes(int n_tokens) {
    return std::max(kernels::native_q8_1_bytes(kAttnHeads * kAttnHeadDim, n_tokens),
                    kernels::native_q8_1_bytes(kAttnOutWidth, n_tokens));
}

size_t attn_out_scratch_floats(int n_tokens) { return (size_t) n_tokens * kAttnOutWidth; }

void attn_out(const float* attn, const AttnOutWeights& w, int n_tokens, void* xq_scratch, float* oa_scratch,
              float* out, void* stream) {
    if (n_tokens < 1 || n_tokens > kAttnMaxTokens) throw std::invalid_argument("attn_out: n_tokens");
    // the T x 64 x 512 result is T * 8 contiguous columns of 4096
    kernels::native_quantize_q8_1(attn, xq_scratch, kAttnHeads * kAttnHeadDim, n_tokens, stream);
    if (!wo_a_grouped(w.wo_a_type, w.wo_a, xq_scratch, oa_scratch, n_tokens, stream)) {
        const size_t col_q8 = kernels::native_q8_1_bytes(kAttnGroupIn, 1);
        const size_t group_w = kernels::native_mmvq_weight_bytes(w.wo_a_type, kAttnGroupIn, kAttnOutRank);
        for (int t = 0; t < n_tokens; ++t) {
            for (int g = 0; g < kAttnGroups; ++g) {
                kernels::native_mmvq(w.wo_a_type, (const char*) w.wo_a + (size_t) g * group_w,
                                     (const char*) xq_scratch + (size_t) (t * kAttnGroups + g) * col_q8,
                                     oa_scratch + (size_t) t * kAttnOutWidth + (size_t) g * kAttnOutRank, kAttnGroupIn,
                                     kAttnOutRank, 1, stream);
            }
        }
    }
    kernels::native_quantize_q8_1(oa_scratch, xq_scratch, kAttnOutWidth, n_tokens, stream);
    kernels::native_mmvq(w.wo_b_type, w.wo_b, xq_scratch, out, kAttnOutWidth, kAttnEmbd, n_tokens, stream);
}

}  // namespace basalt::models::deepseek41
