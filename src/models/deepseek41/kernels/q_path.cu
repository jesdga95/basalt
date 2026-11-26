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
#include "basalt/kernels/sparse_mqa.hpp"
#include "basalt/models/deepseek41/attn.hpp"

namespace basalt::models::deepseek41 {
namespace {

kernels::SparseMqaQWeights shared_q_weights(const QPathWeights& w) {
    return {w.wq_a, w.wq_a_type, w.q_norm, w.wq_b, w.wq_b_type, w.rms_eps, w.n_embd, w.q_rank};
}

kernels::MqaRopeTable shared_rope(const RopeTable& r) {
    kernels::MqaRopeTable out{};
    for (int i = 0; i < kAttnRopeDim / 2; ++i) out.step[i] = r.step[i];
    out.mscale = r.mscale;
    return out;
}

}  // namespace

size_t q_path_scratch_bytes(int n_tokens) { return q_path_scratch_bytes(n_tokens, kAttnQRank); }
size_t q_path_scratch_bytes(int n_tokens, int q_rank) { return kernels::sparse_mqa_q_scratch_bytes(n_tokens, q_rank); }

void q_latent(const void* x_q8_1, const QPathWeights& w, int n_tokens, float* qr, void* qr_q8_1, void* stream) {
    kernels::sparse_mqa_q_latent(x_q8_1, shared_q_weights(w), n_tokens, qr, qr_q8_1, stream);
}

void q_heads(const QPathWeights& w, int n_tokens, const int32_t* pos, const RopeTable& rope, const void* qr_q8_1,
             float* q, void* stream) {
    kernels::sparse_mqa_q_heads(shared_q_weights(w), n_tokens, pos, shared_rope(rope), qr_q8_1, q, stream);
}

void q_path(const void* x_q8_1, const QPathWeights& w, int n_tokens, const int32_t* pos, const RopeTable& rope,
            void* qr_q8_1, float* qr, float* q, void* stream) {
    q_latent(x_q8_1, w, n_tokens, qr, qr_q8_1, stream);
    q_heads(w, n_tokens, pos, rope, qr_q8_1, q, stream);
}

}  // namespace basalt::models::deepseek41
