#pragma once

namespace basalt::kernels {

bool sparse_mqa_wo_a_grouped(int type, const void* wo_a, const void* xq, float* oa, int n_tokens, void* stream);

}  // namespace basalt::kernels
