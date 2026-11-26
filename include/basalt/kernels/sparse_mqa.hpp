// include/basalt/kernels/sparse_mqa.hpp - shared sparse multi-query attention over 512-wide latent rows.
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::kernels {

constexpr int kSparseMqaHeads = 64;
constexpr int kSparseMqaDim = 512;
constexpr int kSparseMqaRopeDim = 64;
constexpr int kSparseMqaWindow = 128;
constexpr int kSparseMqaMaxTokens = 8;
constexpr int kSparseMqaTopK = 512;
constexpr int kSparseMqaOutputGroups = 8;
constexpr int kSparseMqaOutputRank = 1024;
constexpr int kSparseMqaGroupInput = kSparseMqaHeads * kSparseMqaDim / kSparseMqaOutputGroups;

enum class MqaRowType { F32, F16, BF16 };

/// Inverse RoPE parameters applied after sparse attention; the last 64 output dimensions are rotated.
struct MqaRopeTable {
    float step[kSparseMqaRopeDim / 2]{};
    float mscale = 1.0f;
};

/// Shared low-rank query projections used by sparse MQA models. `wq_a` is [q_rank][n_embd], `q_norm` is [q_rank],
/// and `wq_b` is [64 * 512][q_rank], in native_mmvq's GGUF block format.
struct SparseMqaQWeights {
    const void* wq_a = nullptr;
    int wq_a_type = 0;
    const float* q_norm = nullptr;
    const void* wq_b = nullptr;
    int wq_b_type = 0;
    float rms_eps = 1e-20f;
    int n_embd = 0;
    int q_rank = 0;
    /// Optional per-head RMS normalization before RoPE; zero disables it.
    float head_rms_eps = 0.f;
};

size_t sparse_mqa_q_scratch_bytes(int n_tokens, int q_rank);
void sparse_mqa_q_latent(const void* x_q8_1, const SparseMqaQWeights& weights, int n_tokens, float* qr,
                         void* qr_q8_1, void* stream);
void sparse_mqa_q_heads(const SparseMqaQWeights& weights, int n_tokens, const int32_t* pos,
                        const MqaRopeTable& rope, const void* qr_q8_1, float* q, void* stream);

/// In-place RoPE of the last 64 dimensions of float rows at positions pos[row / rows_per_pos].
void sparse_mqa_rope_rows(float* rows, int n_rows, int row_stride, int rows_per_pos, const int32_t* pos,
                          const MqaRopeTable& rope, bool inverse, void* stream);

/// Shared grouped low-rank attention output projection. `wo_a` stores eight [1024][4096] matrices and `wo_b`
/// stores [n_embd][8192], all in native_mmvq's GGUF block format. Scratch is Q8_1 bytes for 64*512 values per token.
size_t sparse_mqa_output_scratch_bytes(int n_tokens);
size_t sparse_mqa_output_workspace_floats(int n_tokens);
void sparse_mqa_output(const float* attn, int wo_a_type, const void* wo_a, int wo_b_type, const void* wo_b,
                       int n_embd, int n_tokens, void* q8_scratch, float* latent, float* out, void* stream);
/// Same projection when `q8_scratch` already contains Q8_1 rows for the attention input.
void sparse_mqa_output_q8(int wo_a_type, const void* wo_a, int wo_b_type, const void* wo_b, int n_embd,
                          int n_tokens, void* q8_scratch, float* latent, float* out, void* stream);

/// Decode sparse MQA. Query rows are [T][64][512] float; window and selected rows are 512-wide F32/F16/BF16.
/// Supported row pairs are F32/F32, F16/F16, F16/F32, and BF16/BF16 (the selected-row type is ignored when n_sel is 0).
/// Each score is dot(q, key) * scale. `sinks[head]` joins the softmax denominator without a value. The window is a
/// ring read in logical order from `win_start[t]` for at most 128 rows; selected IDs outside [0, comp_limit[t]) are masked.
/// `round_output_bf16` applies the source-style BF16 output boundary after inverse RoPE while retaining float storage.
struct SparseMqaArgs {
    int n_tokens = 1;
    const float* q = nullptr;
    const void* win = nullptr;
    MqaRowType win_type = MqaRowType::F32;
    int win_cap = kSparseMqaWindow;
    const int32_t* win_start = nullptr;
    const int32_t* win_n = nullptr;
    const void* comp = nullptr;
    MqaRowType comp_type = MqaRowType::F32;
    const int32_t* sel = nullptr;
    int n_sel = 0;
    const int32_t* comp_limit = nullptr;
    const float* sinks = nullptr;
    float scale = 0.044194174f;
    const int32_t* pos = nullptr;
    MqaRopeTable rope{};
    bool round_output_bf16 = false;
    float* out = nullptr;
};

/// Floats of workspace needed by sparse_mqa for these bounds.
size_t sparse_mqa_workspace_floats(int n_tokens, int n_sel);

/// Enqueues sparse MQA work on `stream`; inputs, output, and workspace are device pointers.
void sparse_mqa(const SparseMqaArgs& args, float* workspace, void* stream);

}  // namespace basalt::kernels
