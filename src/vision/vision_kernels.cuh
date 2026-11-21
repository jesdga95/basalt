// src/vision/vision_kernels.cuh - the encoder's device kernels (launchers on a stream).
//
// Activations: x is the fp32 residual stream [n][width]; `act` is bf16, the input of the next product; `g` is the fp32
// output of a product.  q, k, v are fp16 [16 heads][n][padded head width]: [n][80] with columns 72..79 zero for the
// 72-wide heads, [n][64] for the 64-wide ones.
#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace basalt::vision {

/// x[t] += patch bias + bilinear position embedding (48x48 table) for the patch at merge-order row t.
void launch_embed_finish(float* x, const float* bias, const float* pos_table, int n, int gx, int gy, cudaStream_t s);
/// tab[t][j] = (cos, sin) of the rotation of pair j (0..35) for merge-order row t.
void launch_rope_table(float* tab, int n, int gx, cudaStream_t s);
/// y[t] = (x[t] - mean) / sqrt(var + eps) * w + b, as bf16; rows of `cols` (1152 or 1024).
void launch_layernorm(const float* x, const float* w, const float* b, __nv_bfloat16* y, int n, int cols, float eps,
                      cudaStream_t s);
/// g [n][3456] + bias, rotary on q and k, split into the fp16 head-major q, k, v.
void launch_qkv_epilogue(const float* g, const float* bias, const float* rope, __half* q, __half* k, __half* v, int n,
                         cudaStream_t s);
/// Full attention of the queries [q_begin, q_end) over all n tokens, 16 heads of `head_dim` (72 or 64): out
/// [q_end - q_begin][16 * head_dim] bf16 (head h at columns h * head_dim).
void launch_attention(const __half* q, const __half* k, const __half* v, __nv_bfloat16* out, int n, int q_begin, int q_end,
                      int head_dim, float scale_log2, cudaStream_t s);
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

// The ViT with a class token: 20-pixel patches of a picture on whole 40-pixel cells, a learned
// position table resampled per grid, a class token in front, 2D rotary positions on interleaved pairs, layer scale.
/// The picture's per-channel normalisation: v' = (v - mean) / std on the 0-255 scale.
struct PixelNorm {
    float mean[3], std[3];
};
/// rows[r] (bf16 [count][3 * patch * patch]: channel, row, column) for the patches r0 .. r0 + count of a gx-wide grid in merge
/// order, from the w x h picture `rgb` (interleaved u8) at the canvas' top left; a pixel past it is -1 before normalising.
void launch_patch_rows(const uint8_t* rgb, int w, int h, int gx, int patch, PixelNorm norm, __nv_bfloat16* rows, int r0,
                       int count, cudaStream_t s);
/// x[0] = cls; x[1 + t] += the position table (edge x edge rows of `cols`) resampled bilinearly to the gx x gy grid
/// (align_corners false, taps clamped) for the patch at merge-order row t (0 <= t < gx * gy).
void launch_embed_cls(float* x, const float* cls, const float* pos_table, int gx, int gy, int edge, int cols, cudaStream_t s);
/// x[t] = (x[t] - mean) / sqrt(var + eps) * w + b in place; rows of 1024.
void launch_layernorm_inplace(float* x, const float* w, const float* b, int n, float eps, cudaStream_t s);
/// tab[t][j] = (cos, sin) for row t of the sequence (the class token, then the patches in merge order) and pair j of 32:
/// pairs 0..15 turn by the patch's column + 1, pairs 16..31 by its row + 1, each pair j by 10000^(-(j % 16) / 16); the class
/// token sits at (0, 0).
void launch_rope_axial(float* tab, int n, int gx, cudaStream_t s);
/// g [count][3072] (rows r0 .. r0 + count of the sequence) + bias, the interleaved pairs of q and k turned by `rope`, into the
/// fp16 head-major q, k, v of 16 heads x 64 for a sequence of n rows.
void launch_qkv_axial(const float* g, const float* bias, const float* rope, __half* q, __half* k, __half* v, int n, int r0,
                      int count, cudaStream_t s);
/// x[t][c] += scale[c] * (g[t][c] + bias[c])
void launch_bias_scale_residual(float* x, const float* g, const float* bias, const float* scale, int n, int cols, cudaStream_t s);
/// y[i] = bf16(x[i]) for i < count
void launch_to_bf16(const float* x, __nv_bfloat16* y, size_t count, cudaStream_t s);

}  // namespace basalt::vision
