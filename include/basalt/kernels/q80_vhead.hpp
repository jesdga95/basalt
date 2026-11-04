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
/// `n_screen` (two-stage only): the rows [0, n_screen) are screened and scored as above; the entries of rows
/// [n_screen, n_out) are set to -inf, for another card's share of the head.  `link_pub` / `link_sub`: the split's link
/// (q80p_vhead_link_bytes, below) on this card (its thresholds are published there for the window) or on the share's
/// card (each threshold is raised to the one published for this window, when it comes in time).
size_t q80p_vhead_scratch_bytes();
void q80p_vhead(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, const int* k,
                void* scratch, void* stream, int n_screen = -1, unsigned* link_pub = nullptr,
                const unsigned* link_sub = nullptr);
/// The widest exact list the two-stage head keeps.
constexpr int kQ80pKeepMax = 64;

/// A share of the head on another card, rows [first, n_out) of the whole head.  The two cards meet in host memory both
/// map, without a stream between them: the link (q80p_vhead_link_bytes, zeroed once; its first word, the window's
/// number, is written by the host before each window and is never 0), the input block (q80p_vhead_input_bytes) and the
/// list block (q80p_vhead_list_bytes).  This card posts the window's q8_1 columns and list length (q80p_vhead_post),
/// runs q80p_vhead with n_screen = first and link_pub, then merges (q80p_vhead_merge: once the share's list of this
/// window is in, its logits go into this card's columns).  The share's card fetches the input (q80p_vhead_fetch), runs
/// q80p_vhead on its copy of the rows with link_sub, then lists every entry at or above its column's threshold
/// (q80p_vhead_list: the scored rows whose logit reaches it, per column a count and the threshold, then (row, logit
/// bits) pairs) and marks the list with the window.  Either threshold is at or below the K-th largest logit of its own
/// rows, so every row of a column's K largest is scored on its card and listed or kept, and the K largest entries of
/// the merged column are the exact head's; the share's other entries are -inf.  With k 0 this card's exact pass takes
/// every row and the merge does nothing.  A side that waits past its limit for the other leaves the window marked
/// (q80p_vhead_link_failed): its head is not to be used.
size_t q80p_vhead_link_bytes();
size_t q80p_vhead_input_bytes(int n_in, int ncols);
size_t q80p_vhead_list_bytes(int ncols, int n_out);
void q80p_vhead_post(const void* x_q8_1, int n_in, int ncols, const int* k, void* input, void* link, void* stream);
void q80p_vhead_fetch(const void* input, const void* link, void* x_q8_1, int* k, int n_in, int ncols, void* scratch,
                      void* stream);
void q80p_vhead_list(const float* y, int n_out, int ncols, const int* k, void* scratch, void* list, void* link,
                     void* stream);
void q80p_vhead_merge(float* y, int n_out, int first, int ncols, int list_rows, const int* k, const void* list,
                      void* link, void* stream);
bool q80p_vhead_link_failed(const void* link);
/// Loads the kernels of the two-stage head and its split on the current device: their first launch must not wait for a
/// module load behind a kernel that waits for another card.
void q80p_vhead_prepare();

}  // namespace basalt::kernels
