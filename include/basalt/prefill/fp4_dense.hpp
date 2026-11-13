// include/basalt/prefill/fp4_dense.hpp - a dense matrix's prompt-chunk products on the FP4 block-scaled tensor cores,
// weights in NVFP4_PLANES (include/basalt/kernels/nvfp4_planes.hpp), activations rounded to FP4 per token (W4A4).
//
// The product is a TMA-fed, warp-specialized GEMM tile of 128 weight rows x 256 tokens x K 128: one producer warpgroup
// issues the tensor loads into a three-stage mbarrier pipeline, eight consumer warps run the block-scaled MMA.
// Adapted from Infernix's nvfp4_a4_tma_kernel (see src/prefill/fp4_dense.cu).
//
// Activations: the codes [tokens][K / 2] (byte b of a row holds value 2b in its low nibble and 2b + 1 in its high one,
// as the weight planes do), then the UE4M3 scales, one per 16 values, in tiles of 256 tokens x 256 values (4096 bytes:
// token-major rows of 16 groups), tile (token / 256) * (K / 256) + k / 256, the last token tile padded to 256 tokens.
// Each token has its own FP32 scale gs[t] = amax / (6 * 448), applied in the product's epilogue.
//
// Every output keeps one K order whatever the number of tokens, so a token's row is the same bits at any chunk length.
// Shapes: weight rows a multiple of 128, K a multiple of 256, tokens >= 1.  Output is token-major FP32, dst[t * ld + r];
// dst and ld must be even (float2 stores).
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::prefill::fp4_dense {

/// Bytes of the activation planes (codes, then the scales of the tokens padded to whole 256-token tiles) of `rows`
/// tokens of `cols` values (a multiple of 256).
size_t act_bytes(int64_t rows, int64_t cols);

/// Rounds `rows` rows of x (`ld` floats apart) into the activation planes `act` (act_bytes) and the token scales
/// gs[rows].
void quantize(const float* x, int64_t rows, int64_t cols, int64_t ld, void* act, float* gs, void* stream);
/// As quantize, of the RMSNorm'd rows x * rsqrt(mean(x^2) + eps) * w (rows `cols` apart); xn (optional) gets the normed rows.
/// As quantize, with each row's largest |value| given (rowmax[rows], e.g. from swiglu's amax), so the rows are read once:
/// the same codes and scales.
void quantize_amax(const float* x, const float* rowmax, int64_t rows, int64_t cols, int64_t ld, void* act, float* gs,
                   void* stream);
void rms_norm_quantize(const float* x, const float* w, float eps, int64_t rows, int64_t cols, void* act, float* gs, float* xn,
                       void* stream);

/// dst[t * ld + r] (+)= gs[t] * (*w_scale) * sum_k w[r][k] x[t][k], r < rows, t < tokens.  w: planes of rows x K
/// (nvfp4p::bytes); w_scale: the matrix's device FP32 scale.  add: dst += (the residual epilogue), else dst = .
void product(const void* w, const float* w_scale, int64_t rows, int64_t K, const void* act, const float* gs,
             int64_t tokens, float* dst, int64_t ld, bool add, void* stream);

/// dst[t * ld + r] = silu(s_gate * g) * (s_up * u), g and u the products of gate / up (two plane matrices of rows x K,
/// the gate and up scales on the device) with the token's scale applied to both; r < rows.
/// amax (optional, `tokens` floats): each token's largest |output|, for quantize_amax.
void swiglu(const void* w_gate, const void* w_up, const float* s_gate, const float* s_up, int64_t rows, int64_t K,
            const void* act, const float* gs, int64_t tokens, float* dst, int64_t ld, void* stream,
            float* amax = nullptr);

}  // namespace basalt::prefill::fp4_dense
