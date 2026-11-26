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

enum class MqaRowType { F32, F16, BF16 };

/// Inverse RoPE parameters applied after sparse attention; the last 64 output dimensions are rotated.
struct MqaRopeTable {
    float step[kSparseMqaRopeDim / 2]{};
    float mscale = 1.0f;
};

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
