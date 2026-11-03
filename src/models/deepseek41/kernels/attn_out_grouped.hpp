// DeepSeek-V4.1 grouped wo_a decode kernel: one launch for all groups and tokens.
#pragma once

namespace basalt::models::deepseek41 {

/// oa[t][g*1024 + r] = wo_a[g] row r . q8_1 column (t*8+g) of `xq`, bitwise equal to one native single-column
/// MMVQ launch per (token, group). Supports GGUF types 8 (Q8_0), 13 (Q5_K) and 14 (Q6_K); returns false for others.
bool wo_a_grouped(int type, const void* wo_a, const void* xq, float* oa, int n_tokens, void* stream);

}  // namespace basalt::models::deepseek41
