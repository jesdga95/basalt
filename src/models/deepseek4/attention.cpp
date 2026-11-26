#include "basalt/models/deepseek4/attention.hpp"

#include "basalt/kernels/native_mmvq.hpp"

#include <stdexcept>

namespace basalt::models::deepseek4 {

size_t attention_kv_scratch_bytes(int n_tokens) {
    if (n_tokens < 1 || n_tokens > 8) throw std::invalid_argument("deepseek4 attention_kv_scratch_bytes: token count");
    return kernels::native_q8_1_bytes(4096, n_tokens);
}

void project_attention_kv(const float* x, int wkv_type, const void* wkv, const float* norm, float rms_eps,
                          const int32_t* positions, const kernels::MqaRopeTable& rope, int n_tokens,
                          void* q8_scratch, float* projected, float* out, void* stream) {
    attention_kv_scratch_bytes(n_tokens);
    if (!x || !wkv || !norm || !positions || !q8_scratch || !projected || !out || !stream ||
        !kernels::native_mmvq_supported(wkv_type))
        throw std::invalid_argument("deepseek4 project_attention_kv: invalid arguments or weight type");
    (void) kernels::native_mmvq_weight_bytes(wkv_type, 4096, 512);
    kernels::native_quantize_q8_1(x, q8_scratch, 4096, n_tokens, stream);
    kernels::native_mmvq(wkv_type, wkv, q8_scratch, projected, 4096, 512, n_tokens, stream);
    prepare_attention_kv(projected, norm, rms_eps, positions, rope, n_tokens, out, stream);
}

void attention_step(const float* x, const AttentionStepWeights& w, const kernels::MqaRopeTable& rope,
                   float* window_ring, kernels::SparseMqaArgs selection, const AttentionStepScratch& s,
                   void* stream) {
    if (!x || !window_ring || !stream || selection.n_tokens <= 0 || selection.n_tokens > 8 || !selection.pos || !selection.win_start ||
        !selection.win_n || !w.sinks || !s.input_q8 || !s.q_latent || !s.q_latent_q8 || !s.q_heads ||
        !s.kv_projected || !s.kv_row || !s.mqa_output || !s.mqa_workspace || !s.output_q8 ||
        !s.output_latent || !s.output || w.q.n_embd != 4096 || w.q.q_rank != 1024 || w.q.head_rms_eps <= 0.f ||
        !w.kv || !w.kv_norm || !w.output_a || !w.output_b)
        throw std::invalid_argument("deepseek4 attention_step: invalid window, weights, state or workspace");

    const int n_tokens = selection.n_tokens;
    kernels::native_quantize_q8_1(x, s.input_q8, 4096, n_tokens, stream);
    kernels::sparse_mqa_q_latent(s.input_q8, w.q, n_tokens, s.q_latent, s.q_latent_q8, stream);
    kernels::sparse_mqa_q_heads(w.q, n_tokens, selection.pos, rope, s.q_latent_q8, s.q_heads, stream);
    project_attention_kv(x, w.kv_type, w.kv, w.kv_norm, w.q.rms_eps, selection.pos, rope, n_tokens, s.input_q8,
                         s.kv_projected, s.kv_row, stream);
    // Stage all projections together, then make each row's cache visible only after its causal attention has read the
    // previous ring. A bulk cache write would let a later row overwrite a wrapped slot still needed by an earlier row.
    for (int t = 0; t < n_tokens; ++t) {
        float* mqa_out = s.mqa_output + (size_t) t * 64 * 512;
        cache_attention_window(window_ring, s.kv_row + (size_t) t * 512, selection.pos + t, 1, stream);
        kernels::SparseMqaArgs row = selection;
        row.n_tokens = 1;
        row.q = s.q_heads + (size_t) t * 64 * 512;
        row.win = window_ring;
        row.win_type = kernels::MqaRowType::F32;
        row.win_cap = kernels::kSparseMqaWindow;
        row.win_start = selection.win_start + t;
        row.win_n = selection.win_n + t;
        row.pos = selection.pos + t;
        if (row.sel) row.sel += (size_t) t * selection.n_sel;
        if (row.comp_limit) row.comp_limit += t;
        row.sinks = w.sinks;
        row.rope = rope;
        row.round_output_bf16 = true;
        row.out = mqa_out;
        kernels::sparse_mqa(row, s.mqa_workspace, stream);
    }
    kernels::sparse_mqa_output(s.mqa_output, w.output_a_type, w.output_a, w.output_b_type, w.output_b,
                               w.q.n_embd, n_tokens, s.output_q8, s.output_latent, s.output, stream);
}

}  // namespace basalt::models::deepseek4
