#pragma once

namespace basalt::models::deepseek4 {

void swiglu_clamped(const float* gate, const float* up, int width, float limit, float* output, int n_tokens,
                    void* stream);

}  // namespace basalt::models::deepseek4
