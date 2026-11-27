// src/models/glm5next/kernels/head_transpose.hpp - see head_transpose.cu: [token][head][width] to
// [head][token][width], the layout GLM-5.3-Flash's per-head MLA products want their activations in.
#pragma once

#include <cuda_runtime.h>

namespace basalt::models::glm5next {

/// `xt[h][t][d] = x[t][h][d]` over `n_tok` tokens of `n_head` heads, `width` floats a head. Both buffers are
/// contiguous and the width stays innermost: only the token and head axes trade places. `width` must be a multiple
/// of 4 (the MLA's widths are), and neither `x` nor `xt` may overlap.
void head_major(const float* x, float* xt, int n_tok, int n_head, int width, cudaStream_t stream);

/// `y[t][head][d] = src[t][d]`: one head's contiguous `[token][width]` block back into its slice of a token-major
/// row, the single-head inverse of `head_major`. `width` must be a multiple of 4; `src` may not overlap `y`.
void head_store(const float* src, float* y, int n_tok, int n_head, int head, int width, cudaStream_t stream);

}  // namespace basalt::models::glm5next
