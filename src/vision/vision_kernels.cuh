// src/vision/vision_kernels.cuh - the encoder's device kernels (launchers on a stream).
//
// Activations: x is the fp32 residual stream [n][1152]; `act` is bf16, the input of the next product; `g` is the
// fp32 output of a product.  q, k, v are fp16 [16 heads][n][80] with columns 72..79 zero.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace basalt::vision {

/// x[t] += patch bias + bilinear position embedding (48x48 table) for the patch at merge-order row t.
void launch_embed_finish(float* x, const float* bias, const float* pos_table, int n, int gx, int gy, cudaStream_t s);
/// tab[t][j] = (cos, sin) of the rotation of pair j (0..35) for merge-order row t.
void launch_rope_table(float* tab, int n, int gx, cudaStream_t s);
/// y[t] = (x[t] - mean) / sqrt(var + eps) * w + b, as bf16; rows of 1152.
void launch_layernorm(const float* x, const float* w, const float* b, __nv_bfloat16* y, int n, float eps, cudaStream_t s);
/// g [n][3456] + bias, rotary on q and k, split into the fp16 head-major q, k, v.
void launch_qkv_epilogue(const float* g, const float* bias, const float* rope, __half* q, __half* k, __half* v, int n,
                         cudaStream_t s);
/// Full attention over n tokens, 16 heads of 72: out [n][1152] bf16 (head h at columns h * 72).
void launch_attention(const __half* q, const __half* k, const __half* v, __nv_bfloat16* out, int n, float scale_log2,
                      cudaStream_t s);
/// x[t][c] += g[t][c] + bias[c]
void launch_bias_residual(float* x, const float* g, const float* bias, int n, int cols, cudaStream_t s);
/// y[t][c] = bf16(gelu(g[t][c] + bias[c]))
void launch_bias_gelu(const float* g, const float* bias, __nv_bfloat16* y, int n, int cols, cudaStream_t s);
/// y[t][c] = g[t][c] + bias[c]
void launch_bias_copy(const float* g, const float* bias, float* y, int n, int cols, cudaStream_t s);

// The SigLIP2-derived ViT (SigLIP2-derived: a learned position table, no rotation, an RMS norm and an erf-GELU projector).
/// x[t] += patch bias + the position table (table_edge x table_edge entries of 1152) bilinearly resampled to the gx x gy
/// grid with the reference's scale factor (size + 0.1) / table_edge, for the patch at merge-order row t.
void launch_embed_finish_interp(float* x, const float* bias, const float* pos_table, int n, int gx, int gy, int table_edge,
                                cudaStream_t s);
/// g [n][3456] + bias split into the fp16 head-major q, k, v (no rotation).
void launch_qkv_plain(const float* g, const float* bias, __half* q, __half* k, __half* v, int n, cudaStream_t s);
/// y[t] = x[t] / sqrt(mean(x[t]^2) + eps) * w, as bf16; rows of 1152.
void launch_rmsnorm(const float* x, const float* w, __nv_bfloat16* y, int n, float eps, cudaStream_t s);
/// y[t][c] = bf16(gelu_erf(g[t][c] + bias[c]))
void launch_bias_gelu_erf(const float* g, const float* bias, __nv_bfloat16* y, int n, int cols, cudaStream_t s);
/// The m merged rows g[m][cols] + bias as lines of `ow` rows, each followed by `newline`: y[m + m / ow][cols].
void launch_bias_copy_lines(const float* g, const float* bias, const float* newline, float* y, int m, int cols, int ow,
                            cudaStream_t s);

}  // namespace basalt::vision
