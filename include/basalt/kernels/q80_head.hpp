// include/basalt/kernels/q80_head.hpp - the Q8_0 x q8_1 GEMV of a wide head (the output projection) for 1..8 columns,
// bitwise equal to native_mmvq (type 8, the exact layout): one warp per row holds what four warps did, a block reads
// the columns' activations once into shared memory for all its rows, and one resident wave walks the rows.
#pragma once

namespace basalt::kernels {

bool q80_head_supported(int n_in, int ncols);
/// y[j * n_out + r] = row r of `weights` (Q8_0, n_in / 32 blocks per row) . column j of `x_q8_1`.  Each warp loads its
/// first row before the kernel waits on its predecessor (a PdlScope applies).
void q80_head_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream);

}  // namespace basalt::kernels
