// include/basalt/kernels/dense_ops.hpp - the element-wise steps of a dense layer between its products.
//
// Rows are tokens, row-major, fp32 unless said otherwise. `scale` arguments are device pointers to one fp32 value
// (an NVFP4 matrix's tensor scale) or null for 1.
#pragma once

#include <cstdint>

namespace basalt::kernels {

/// out[r][c] = x[r][c] * rsqrt(mean_c x[r]^2 + eps) * w[c]  (w holds an offset norm's 1 + w already); out rows
/// `ld_out` floats apart (0: cols).
void rms_norm_rows(const float* x, const float* w, float* out, int64_t rows, int64_t cols, float eps, void* stream,
                   int64_t ld_out = 0);
/// x[i] += y[i] * scale  (n values).
void add_scaled(float* x, const float* y, const float* scale, int64_t n, void* stream);
/// y[i] *= scale  (n values).
void scale_values(float* y, const float* scale, int64_t n, void* stream);
/// h[i] = silu(g[i] * sg) * (u[i] * su)  (n values; g, u and h may each have their own row stride: rows x width).
void swiglu_scaled(const float* g, const float* u, const float* sg, const float* su, float* h, int64_t rows,
                   int64_t width, int64_t ld_gu, void* stream);
/// y[t][n] = x[t] . w[n]  for BF16 rows w [N][K]: the small BF16 projections of a prompt chunk.
void bf16_rows_dot(const float* x, const uint16_t* w, float* y, int64_t T, int64_t N, int64_t K, void* stream);

/// The rotary part of a gated GQA attention layer: `rot` leading dims of each head rotate NeoX-style (pairs i and
/// i + rot/2) with inverse frequency theta^(-2i/rot); pair i takes its position from axis `axis[i]` (0 t, 1 h, 2 w:
/// interleaved M-RoPE; all 0 for plain rope).
struct RopeSpec {
    int rot = 0;
    float theta = 0.0f;
    int8_t axis[64] = {};   // per rotated pair
};

/// One window's or chunk's query and key heads: q_full [T][n_head][2 hd] (query | gate per head), k and v
/// [T][n_kv][hd]. Each q and k head is RMS-normed (q_norm / k_norm, with the offset folded) and rotated at its
/// row's position (pos [T][3]); q goes to q_out [T][n_head][hd], k and v to the fp16 cache at cells
/// cell0 .. cell0 + T - 1 (cache [cell][n_kv][hd]). hd = 256.
void attn_qk_prepare(const float* q_full, const float* k, const float* v, const float* q_norm, const float* k_norm,
                     float eps, const RopeSpec& rope, const int32_t* pos, int64_t T, int n_head, int n_kv,
                     float* q_out, uint16_t* k_cache, uint16_t* v_cache, int64_t cell0, void* stream);
/// out[t][h][d] = attn[t][h][d] * sigmoid(q_full[t][h][hd + d])  (hd = 256).
void attn_gate(const float* attn, const float* q_full, float* out, int64_t T, int n_head, void* stream);

}  // namespace basalt::kernels
