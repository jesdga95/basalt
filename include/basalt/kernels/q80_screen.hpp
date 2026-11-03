// include/basalt/kernels/q80_screen.hpp - the argmax of a plain Q8_0 head (ggml's block_q8_0 rows, row r = token r) over one
// q8_1 activation, screened by a 2-bit copy of every row (head_proxy.cuh): the approximate pass over the copy, the rows
// near its best scored exactly with native_mmvq's one-column arithmetic (the same bits as its logits), the exact best
// is the pick.  A draft head: the pick is the full head's whenever its best row is among the rescored ones.
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::kernels {

/// Whether rows of `n_in` values have a screen (the widths the kernels are built for).
bool q80_screen_supported(int n_in) noexcept;
/// The 2-bit copy of `n_rows` rows of `n_in` values, and the scratch a call needs.
size_t q80_screen_bytes(int n_in, int64_t n_rows);
size_t q80_screen_scratch_bytes(int64_t n_rows);

/// Packs the copy (q80_screen_bytes) from the head's rows.
void q80_screen_pack(const void* head_q8_0, int n_in, int64_t n_rows, void* screen, void* stream);

/// out_id = the head's argmax over x_q8_1 (n_in values as q8_1 blocks): the lowest row on a tie.  out_logit, when given:
/// the pick's logit.
void q80_screen_top1(const void* screen, const void* head_q8_0, const void* x_q8_1, int n_in, int64_t n_rows,
                     void* scratch, int32_t* out_id, void* stream, float* out_logit = nullptr);

}  // namespace basalt::kernels
