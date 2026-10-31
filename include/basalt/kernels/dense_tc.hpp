// include/basalt/kernels/dense_tc.hpp - dense projections of a verify window on int8 tensor cores.
//
// y[c][r] = sum over the row's 32-value blocks of the format's float term on the block's exact integer dot with
// column c's q8_1 codes (mma.sync, 16 rows x 8 columns per tile).  Terms per block:
//   Q4_K / Q5_K   d8 * (d * sc * dot - dmin * m * sum)      (sum: the block's q8_1 code sum)
//   Q6_K          d * d8 * (sc0 * dot0 + sc1 * dot1)        (the two 16-value halves)
//   IQ4_XS        d * d8 * (ls - 32) * dot
//   IQ3_S         d * d8 * (1 + 2 * sc) * dot
//   Q3_K          d * d8 * ((sc0 - 32) * dot0 + (sc1 - 32) * dot1)   (the two 16-value halves)
//   Q2_0          d * d8 * dot                              (d per 64 values)
//   Q8_0          d * d8 * dot                              (the block's int32 dot is exact)
// The terms are the native kernels' math with fewer roundings, not their bits.  Every column is summed in one order
// fixed by the shape (n_in, n_out) alone, so a column's value does not depend on how many columns ride along or where
// it sits.  The pair stage rounds a partial last tile's bulk copy up to 16 bytes: up to 15 bytes past the matrix's
// last row are read, and weight allocations carry that slack.
//
// Weights are the GGUF row-major blocks; x_q8_1 holds ncols (1..8) columns of n_in / 32 q8_1 blocks.  n_in must be a
// multiple of 512, x_q8_1 16-byte aligned.  Needs thread-block clusters.
#pragma once

#include <cstddef>

namespace basalt::kernels {

bool dense_tc_supported(int ggml_type, int n_in) noexcept;
/// The same for this call's shape and weights: IQ3_S only where each 16-row tile takes a block
/// of its own (dense_tc.cu's one-tile path).
bool dense_tc_supported(int ggml_type, int n_in, int n_out, const void* weights) noexcept;

void dense_tc(int ggml_type, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
              void* stream);

/// Two projections of one input in one launch (whole 16-row tiles, the first matrix's then the second's; types as
/// above plus IQ3_S): each output is bitwise the same for every ncols and column.
bool dense_tc_pair_supported(int type0, int type1, int n_in) noexcept;
void dense_tc_pair(int type0, const void* w0, float* y0, int n_out0, int type1, const void* w1, float* y1, int n_out1,
                   const void* x_q8_1, int n_in, int ncols, void* stream);

/// The pair's options (bits): 2 each warp's x fragments in registers across its tiles (n_in <= 3072; the same sums),
/// 4 two warp-sum buffers where one block per SM, 16 the hc read before the GDN qkv+z pair lets the pair in at its
/// start, so the pair's blocks stage their first tiles on the SMs the read leaves.
int dense_tc_pair_mask();

/// The 16-byte aligned `bytes` at w into L2 (bulk prefetches), as a kernel on the stream that lets the next kernel in
/// at once and ends after the kernel before it.
void dense_tc_l2_prefetch(const void* w, std::size_t bytes, void* stream);

/// Which verify projections take dense_tc for every window width (bits): 1 GDN qkv, 2 GDN z, 4 GDN out, 8 QSA q,
/// 16 QSA out, 32 QSA k/v.
int dense_tc_mask();
/// The weight types dense_tc takes: Q4_K, Q5_K, Q6_K, IQ4_XS, IQ3_S, Q3_K, Q2_0, Q8_0.
bool dense_tc_type_on(int ggml_type) noexcept;

}  // namespace basalt::kernels
