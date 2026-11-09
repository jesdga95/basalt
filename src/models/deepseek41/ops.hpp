// src/models/deepseek41/ops.hpp - small device helpers the layer loop needs besides the kernel agents' pieces.
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::models::deepseek41 {

/// out[t][i] = x[t][i] / sqrt(mean(x[t]^2) + eps) * w[i] for n_tokens rows of n floats (out may alias x).
void rms_norm(const float* x, const float* w, int n_tokens, int n, float eps, float* out, void* stream);

/// out[i] += a[i].
void add_inplace(float* out, const float* a, int n, void* stream);

/// out[i] = a[i] + b[i].
void add(const float* a, const float* b, float* out, int n, void* stream);

/// Holds the stream until *word (mapped host memory the host writes) equals `value`.
void wait_word(const uint32_t* word, uint32_t value, void* stream);

/// out[i] = silu(min(gate[i], limit)) * clamp(up[i], -limit, limit).
void swiglu_clamped(const float* gate, const float* up, int n, float limit, float* out, void* stream);

/// dst[0..n_words) = src[0..n_words) as a kernel, so mapped host memory reaches the device without a copy-engine job.
void copy_words(const void* src, void* dst, int n_words, void* stream);

/// Most words put_words carries.
constexpr int kPutWords = 256;

/// dst[0..n_words) = src[0..n_words) (n_words <= kPutWords) for `src` in host memory: the words travel in the kernel's
/// parameters, so the device never reads host memory, which a busy bus makes slow.
void put_words(const void* src, void* dst, int n_words, void* stream);

/// dst[0..n) = src[0..n) (n a multiple of 4, device rows to mapped host memory or back), as a kernel of one vector move
/// per thread.
void copy_rows(const float* src, float* dst, int n, void* stream);

/// One row pointer per routed expert; a pointer may be device memory or mapped host memory.
struct RowPtrs {
    const float* p[8];
};

/// out[i] = sum_k w[k] * r.p[k][i] for k < n_rows (the same order of summation as weighted_sum_rows).
/// With `add` set, out[i] = (sum) + add[i].
void weighted_sum_row_ptrs(const RowPtrs& rows, const float* w, int n_rows, int n, float* out, void* stream,
                           const float* add = nullptr);

/// Row pointers of a window: token t's k-th routed expert is p[t * n_rows + k].
struct WindowRowPtrs {
    const float* p[48];
};

/// out[t][i] = sum_k w[t * n_rows + k] * r.p[t * n_rows + k][i] (+ add[t][i]), for t < n_tokens; each token sums in the
/// same order as weighted_sum_row_ptrs.
void weighted_sum_window(const WindowRowPtrs& rows, const float* w, int n_tokens, int n_rows, int n, float* out,
                         void* stream, const float* add = nullptr);

/// out[r] = index of the first largest of x[r][0..n) for r < n_rows (one launch per row, sharing `ws`).
void argmax_rows(const float* x, int n_rows, int n, int32_t* out, void* ws, void* stream);

/// *out = index of the first largest of x[0..n); `ws` is device scratch of argmax_workspace_bytes().
void argmax_index(const float* x, int n, int32_t* out, void* ws, void* stream);
size_t argmax_workspace_bytes();

/// out[i] = sum_k w[k] * rows[k][i] for k < n_rows.
void weighted_sum_rows(const float* rows, const float* w, int n_rows, int n, float* out, void* stream);

}  // namespace basalt::models::deepseek41
