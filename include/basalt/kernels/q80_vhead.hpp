// include/basalt/kernels/q80_vhead.hpp - a Q8_0 output head (n_in 2560) held as bit planes, its exact kernels and its
// two-stage verify head.
//
// The planes: a row keeps its 2720 bytes, laid out as the 80 block scales (fp16), then for every code u = q + 128 its
// bits 2..5 (a nibble plane), its bits 6..7 and its bits 0..1 (two 2-bit planes).  The exact kernels read the planes
// back into the Q8_0 codes and run native_mmvq's arithmetic: a logit is the same bits as the Q8_0 head's.
//
// The two-stage head reads the scales and the top six bits of every code.  With the low two bits l in [0, 3] unknown,
// sum q y = 4 sum (u >> 2) y + sum l y - 128 sum y lies within 1.5 sum |y| of its value at l = 1.5, so every row's
// logit is inside [c - w, c + w] (w widened for the float rounding of both sides).  A lower bound of the K-th largest
// c - w is the column's threshold; the rows whose c + w reaches it are scored exactly.  Every row of the exact top K
// is among them, so the K largest entries of the column, their order and their values are the exact head's; every
// other entry holds its upper bound, below all K of them.
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::kernels {

bool q80p_supported(int n_in, int ncols);
/// Rows [0, n) of a Q8_0 head (2720 bytes each) rewritten as planes, in place.
void q80p_pack(void* rows, int64_t n, void* stream);
/// Planes back to Q8_0 rows, in place.
void q80p_unpack(void* rows, int64_t n, void* stream);
/// Every row exact: y[j * n_out + r] = row r . column j, the bits of native_mmvq on the Q8_0 rows.
void q80p_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream);
/// One column: y[r] = row rows[r] . x, the same bits; a negative entry reads row 0.
void q80p_mmvq_rows(const void* weights, const int32_t* rows, const void* x_q8_1, float* y, int n_in, int n_out,
                    void* stream);

/// The two-stage head over 1..8 columns.  `k` (device): the exact list's length, 1..64; 0 runs every row exactly.
/// `scratch`: q80p_vhead_scratch_bytes(), zeroed once before first use (each call leaves it zeroed).
size_t q80p_vhead_scratch_bytes();
void q80p_vhead(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, const int* k,
                void* scratch, void* stream);
/// The widest exact list the two-stage head keeps.
constexpr int kQ80pKeepMax = 64;

}  // namespace basalt::kernels
