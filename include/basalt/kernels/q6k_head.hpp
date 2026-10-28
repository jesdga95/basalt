// include/basalt/kernels/q6k_head.hpp - the Q6_K x q8_1 GEMV of a wide head (the output projection) for 1..8
// columns with n_in 2560, bitwise equal to native_mmvq (type 14, the exact layout): one warp per row holds what four
// warps did, a block reads the columns' activations once into shared memory for all its rows, and a warp loads its
// next row's weights while it computes this one.
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::kernels {

bool q6k_head_supported(int n_in, int ncols);
/// y[j * n_out + r] = row r of `weights` (Q6_K, n_in / 256 blocks per row) . column j of `x_q8_1`
void q6k_head_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream);

/// One column: the logits as above, and in the same launch the first row of the largest logit (through `map` when
/// non-null; rows with a negative id are skipped, their logits set to -inf) and its softmax probability over the
/// n_out rows.  The probability's sum runs in a different order than row_top_prob's (last bits differ).  `scratch`:
/// q6k_head_top1_scratch_bytes(), zeroed once before first use.  False: the shape is not supported.
size_t q6k_head_top1_scratch_bytes();
bool q6k_head_top1(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, const int32_t* map,
                   int* out_id, float* out_prob, void* scratch, void* stream);

/// The 2-bit copy of q5k_head.hpp packed from Q6_K head rows (n_in 2560, row r at r * 2100 bytes); its size is
/// q5k_proxy_bytes(n_cap).  With `rows`, copy row r is packed from head row rows[r].
void q6k_proxy_pack(const void* q6k_rows, int64_t row0, int64_t n, void* proxy, int64_t n_cap, void* stream,
                    const int32_t* rows = nullptr);

/// q5k_proxy_top1 for Q6_K rows: the approximate pass over the 2-bit copy, the near-best rows rescored exactly (the
/// same bits as q6k_head_top1's logits), the exact best is the pick.  `scratch`: q5k_proxy_scratch_bytes(n_cap).
/// `rows_by_map`: q6k_rows is the whole head and the copy's row r is its row map[r] (map entries >= 0).
bool q6k_proxy_top1(const void* proxy, const void* q6k_rows, const void* x_q8_1, int n_rows, int64_t n_cap,
                    const int32_t* map, int* out_id, float* out_prob, void* scratch, void* stream,
                    bool rows_by_map = false);

/// The verify head's 2-bit copy of all n rows of a Q6_K head (n_in 2560, n a multiple of 16), fitted as the draft
/// copy (head_proxy.cuh) and laid out for int8 tensor-core scoring; q6k_vproxy_bytes(n) bytes.
size_t q6k_vproxy_bytes(int64_t n);
void q6k_vproxy_pack(const void* q6k_rows, int64_t n, void* proxy, void* stream);

/// The two-stage verify head for 1..8 columns: y[j * n_out + r] is row r's approximate logit from the copy, except
/// for the rows within a margin of column j's 64th largest approximate logit, which are rescored exactly (the bits of
/// q6k_head_mmvq).  The sampler's top-k list and the argmax come from the exact rows; the other rows keep their
/// approximate logits, so log-probabilities stay finite.  `scratch`: q6k_vhead_scratch_bytes(), zeroed once before
/// first use (each call leaves it zeroed).
size_t q6k_vhead_scratch_bytes();
bool q6k_vhead_supported(int n_in, int n_out, int ncols);
void q6k_vhead(const void* proxy, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
               void* scratch, void* stream);

}  // namespace basalt::kernels
