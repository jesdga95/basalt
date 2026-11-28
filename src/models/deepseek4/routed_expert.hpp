#pragma once

namespace basalt::models::deepseek4 {

struct WindowRowPtrs {
    const float* p[48]{};
};

void weighted_route_sum(const float* rows, const float* weights, int n_tokens, int n_embd, float* output, void* stream);
void weighted_route_sum_ptrs(const WindowRowPtrs& ptrs, const float* weights, const float* shared,
                             int n_tokens, int n_embd, float* output, void* stream);
/// Apply DeepSeek4's clamped SwiGLU to split MMQ rows [gate | up].
void swiglu_split_clamped(const float* gate_up, int n_rows, int n_ff, float limit, float* output, void* stream);

}  // namespace basalt::models::deepseek4
