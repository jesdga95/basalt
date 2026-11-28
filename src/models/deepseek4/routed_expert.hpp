#pragma once

#include <cstdint>

namespace basalt::models::deepseek4 {

struct WindowRowPtrs {
    const float* p[48]{};
};

void weighted_route_sum(const float* rows, const float* weights, int n_tokens, int n_embd, float* output, void* stream);
void weighted_route_sum_ptrs(const WindowRowPtrs& ptrs, const float* weights, const float* shared,
                             int n_tokens, int n_embd, float* output, void* stream);
/// Apply DeepSeek4's clamped SwiGLU to split MMQ rows [gate | up].
void swiglu_split_clamped(const float* gate_up, int n_rows, int n_ff, float limit, float* output, void* stream);

void stage_copy_floats(const float* src, float* dst, int64_t n, void* stream);
void stage_copy_words(const uint32_t* src, uint32_t* dst, int n, void* stream);
void stage_copy_strided(const float* src, float* dst, int n_rows, int row_floats, int dst_stride_floats, void* stream);

}  // namespace basalt::models::deepseek4
