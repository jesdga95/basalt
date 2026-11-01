// include/basalt/kernels/q80_proxy.hpp - the two-stage draft head for a Q8_0 head (n_in 2560): the 2-bit copy of
// q5k_head.hpp packed from Q8_0 rows, the rows near the best approximate score rescored exactly, and the Q8_0 rows of
// a subset read from the whole head through a row map.  The rows are the head's bit planes (q80_vhead.hpp), 2720
// bytes each; every value is the Q8_0 row's.
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::kernels {

/// The 2-bit copy (q5k_proxy_bytes(n_cap) bytes) packed from the head's rows (row r at r * 2720 bytes): copy rows
/// [row0, row0 + n).  With `rows`, copy row r is packed from head row rows[r]; a negative entry leaves the row as it is.
void q80_proxy_pack(const void* q80_rows, int64_t row0, int64_t n, void* proxy, int64_t n_cap, void* stream,
                    const int32_t* rows = nullptr);

/// q5k_proxy_top1 for Q8_0 rows: the approximate pass over the 2-bit copy, the near-best rows rescored exactly (the
/// bits of native_q8_0_mmvq's one-column logits), the exact best is the pick.  `scratch`: q5k_proxy_scratch_bytes(n_cap).
/// `rows_by_map`: q80_rows is the whole head and the copy's row r is its row map[r].
bool q80_proxy_top1(const void* proxy, const void* q80_rows, const void* x_q8_1, int n_rows, int64_t n_cap,
                    const int32_t* map, int* out_id, float* out_prob, void* scratch, void* stream,
                    bool rows_by_map = false);

/// One column: y[r] = row rows[r] of `weights` (Q8_0, n_in 2560) . x_q8_1, the bits of native_q8_0_mmvq on the
/// gathered rows, the weights read cache-streaming.  A negative entry reads row 0 (its logit is masked afterwards).
void q80_mmvq_rows(const void* weights, const int32_t* rows, const void* x_q8_1, float* y, int n_in, int n_out,
                   void* stream);

}  // namespace basalt::kernels
