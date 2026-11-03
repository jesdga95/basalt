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

/// Two-pass scores for the prompt path (the bound and the rule: qsa_two_pass.hpp): a scores row that qsa_block_topk
/// turns into the ids of qsa_block_scores_tc's row.  Every block is scored once in single-pass TF32, the blocks whose
/// error bound reaches the query's top-k cut-off are scored again with qsa_block_scores_tc's 3xTF32 arithmetic (bit
/// for bit), and a query whose bound cannot be settled has its 16-query tile scored again in full.  The kept blocks
/// hold their single-pass score, so the row is for qsa_block_topk only.  `scratch` (qsa_two_pass_scratch_bytes(nq_cap,
/// max_blocks), 16-byte aligned) holds the key norms qsa_two_pass_keys wrote for this layer and chunk.  False (nothing
/// launched): a card where qsa_block_scores_tc does not run, another geometry, nq > nq_cap, too little scratch.
uint64_t qsa_two_pass_scratch_bytes(int64_t nq_cap, int64_t max_blocks);
/// Once per layer and chunk, after the indexer appends and before the calls: the norms of the pooled keys [0, n_blocks).
bool qsa_two_pass_keys(const float* pooled, int64_t n_blocks, int64_t nq_cap, int64_t max_blocks, void* scratch,
                       uint64_t scratch_bytes, void* stream);
bool qsa_block_scores_two_pass(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps,
                               int64_t nq, int64_t nq_cap, int64_t max_blocks, const QsaShapes& s, float* scores,
                               void* scratch, uint64_t scratch_bytes, void* stream, int64_t active_blocks);
/// A check right after a two-pass call and its qsa_block_topk, against the 3xTF32 row and ids of the same queries
/// (ref_scores from qsa_block_scores_tc, ref_ids from qsa_block_topk on them): counts the queries whose ids differ,
/// the scored-again blocks whose score is not the 3xTF32 one bit for bit, and the kept blocks whose single-pass score
/// lies outside the bound (two_pass::Stat).  False (nothing launched): too many blocks for its shared-memory marks.
bool qsa_two_pass_check(const float* scores, const int32_t* ids, const float* ref_scores, const int32_t* ref_ids,
                        const int32_t* steps, int64_t nq, int64_t nq_cap, int64_t max_blocks, int64_t cap,
                        void* scratch, void* stream);
/// The counters (two_pass::kStatCount of them, this device), read on `stream` (synchronized); reset: zeroed after.
bool qsa_two_pass_stats(unsigned long long* out, bool reset, void* stream);

}  // namespace basalt::kernels
