#pragma once

namespace basalt::models::deepseek4 {

void weighted_route_sum(const float* rows, const float* weights, int n_tokens, int n_embd, float* output, void* stream);

}  // namespace basalt::models::deepseek4
