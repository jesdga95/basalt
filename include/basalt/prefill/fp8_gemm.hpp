// include/basalt/prefill/fp8_gemm.hpp - a prompt chunk's products against FP8 E4M3 weights on the FP8 tensor cores
// (W8A8): the activations are rounded to E4M3 once per row with an FP32 scale per row (amax / 448), the weights keep
// their stored codes with one FP32 scale per matrix (NVIDIA's FP8 checkpoints: weight_scale), and every product
// accumulates in fp32 (mma m16n8k32 e4m3) with both scales applied once at the end.
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::prefill::fp8 {

/// Bytes of `rows` activation rows of `cols` values (cols a multiple of 64): the codes [rows][cols], then the row
/// scales [rows] as fp32 (padded to 16 bytes).
size_t act_bytes(int64_t rows, int64_t cols);
/// The row scales inside `act` (rows of `cols` values), for a producer that writes the activations itself.
float* act_scales(void* act, int64_t rows, int64_t cols);

/// Rounds rows of x (`ld` floats apart) to E4M3 with scale amax / 448 per row (round to nearest even, saturating).
void quantize(const float* x, int64_t rows, int64_t cols, int64_t ld, void* act, void* stream);
/// As quantize, of the RMSNorm'd rows x * rsqrt(mean(x^2) + eps) * w (rows `cols` apart); xn (optional) gets the normed rows.
void rms_norm_quantize(const float* x, const float* w, float eps, int64_t rows, int64_t cols, void* act, float* xn, void* stream);
/// As rms_norm_quantize, of BF16 rows (the residual stream); xn (optional) gets the normed rows in BF16.
void rms_norm_quantize(const uint16_t* x, const float* w, float eps, int64_t rows, int64_t cols, void* act, uint16_t* xn,
                       void* stream);

/// Store / Add: FP32 dst; StoreBf16 / AddBf16: BF16 dst (Add widens it, adds in fp32 and rounds once).
enum class Epilogue { Store, Add, StoreBf16, AddBf16 };
/// dst[t * ld_dst + r] (=, or += with Add) = w_scale * a_scale[t] * sum_k W[r][k] A[t][k] for t < rows, r < w_rows:
/// W [w_rows][w_cols] E4M3 codes (row-major, 16-byte aligned rows: w_cols a multiple of 64), `act` from quantize with
/// the same cols, `w_scale` a device fp32 scalar (or null for 1).  Row tiles past w_rows and token tiles past rows are
/// masked; w_rows a multiple of 128 and ld_dst >= w_rows.  An FP32 dst is 8-byte aligned with ld_dst even; a BF16 dst is
/// 16-byte aligned with ld_dst a multiple of 8 (its rows go out by tensor stores).  A row's bits do not depend on `rows`.
void gemm(const uint8_t* w, int64_t w_rows, int64_t w_cols, const float* w_scale, const void* act, int64_t rows, void* dst,
          int64_t ld_dst, Epilogue epi, void* stream);

/// One weight matrix of a segmented product: the same contract as gemm's w, w_rows, w_scale, dst, ld_dst.
struct Segment {
    const uint8_t* w;
    int64_t rows;
    const float* w_scale;
    void* dst;   // FP32 or BF16, as the epilogue says
    int64_t ld_dst;
};
/// One product over up to three weight matrices that share the activations (GDN qkv + z, attention q + k + v): every
/// segment gets what a gemm of it alone would give (bitwise), in one launch.
void gemm_segmented(const Segment* seg, int nseg, int64_t w_cols, const void* act, int64_t rows, Epilogue epi, void* stream);

}  // namespace basalt::prefill::fp8
