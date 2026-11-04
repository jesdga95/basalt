// include/basalt/models/qwen4exp/kernels/qsa_select.hpp - the QSA indexer's block scores and top-k selection,
// for many queries at once and at long context.
//
// The per-token path scores every pooled block with one FP64 block per row (qsa_index_kernel) and selects with a
// single-block kernel that makes 32 bit-serial passes over every CELL (topk_kernel), once per QSA layer and token - a
// large share of a decode token and of a long prompt's time.  Here:
//
//   qsa_block_scores : one warp per (query, block), FP32; relu per indexer head, summed; block n_bid (the
//                      incomplete tail) scores the `dead` key and gets +1e9 when it has cells - which is exactly
//                      what the per-token path reads there, because pooled[n_bid] holds `dead` at that time (in
//                      a chunk it may already hold a block completed later, so it is not read).
//   qsa_block_topk   : one block per query; a 4-pass radix select over the query's n_bid + 1 blocks, each
//                      weighted by its cell count, then the cells emitted in ascending order with ties to the
//                      lowest index - the same selection as topk_kernel, over a quarter of the elements.  On devices with
//                      thread-block clusters a call of up to 16 queries runs it on a cluster of 8 CTAs per query: the same ids.
//
// Queries carry their own step record (pos, n_kv, n_bid, width) as everywhere else in QSA.
#pragma once

#include "basalt/models/qwen4exp/kernels/qsa.hpp"

#include <cstdint>

namespace basalt::kernels {

/// scores [nq, max_blocks]; q_idx [nq, idx_n_head, idx_dim] (normed and rotated); steps [nq, kStepCount].
void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                      int64_t max_blocks, const QsaShapes& s, float* scores, void* stream,
                      int64_t active_blocks = -1,    ///< > 0 launches only this many blocks (the
                                                     ///< batch's largest n_bid + 1; not for a captured graph)
                      bool skip_identity = false);   ///< nothing written when every query's selection is the identity
                                                     ///< (n_kv <= width: the top-k reads no score)

/// The same scores on tensor cores (3xTF32, FP32-level accuracy but another summation order: not bitwise; the tail
/// block n_bid is the warp kernel's arithmetic). For the prompt path; false (nothing launched) on another geometry.
bool qsa_block_scores_tc(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                         int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t active_blocks);

/// ids [nq, cap] (cells, ascending); `cap` >= the largest selection width.
void qsa_block_topk(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                    const QsaShapes& s, int32_t* ids, void* stream,
                    int64_t active_blocks = -1);   ///< > 0: no query of the call has more than this many blocks (n_bid + 1;
                                                   ///< the same contract as qsa_block_scores's). The register kernel is
                                                   ///< chosen by this, not by the capacity max_blocks (a long
                                                   ///< --max-context otherwise sends every short prompt to the slow one).
                                                                                                      ///< Omit it for captured graphs whose context can grow after capture.
/// The same ids on a thread-block cluster of 8 CTAs per query (needs thread-block clusters). qsa_block_topk takes it for calls
/// of up to 16 queries. False (nothing launched) where it cannot run: a device without
/// thread-block clusters, or a capacity whose keys do not fit one cluster's shared memory. Capturable.
bool qsa_block_topk_cluster(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                            const QsaShapes& s, int32_t* ids, void* stream);
/// The original kernel (256 threads, one shared histogram, keys read from memory on every radix pass), for tests and
/// The same ids.
void qsa_block_topk_ref(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                        const QsaShapes& s, int32_t* ids, void* stream);

}  // namespace basalt::kernels
