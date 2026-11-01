// include/basalt/kernels/qsa_seg.hpp - the QSA decode chain over SEGMENTS: the rows of one window belong to several
// sequences (slots), a segment being the consecutive rows of one slot, and the per-sequence state each kernel touches
// comes from a device table instead of kernel arguments. Every row's result is bit for bit what the single-sequence
// kernel gives that slot's rows alone.
//
//   QsaSegSlot   one slot's cache and indexer pointers for one layer; static, uploaded once per layer
//   QsaSeg       a row's slot and its segment (first row, row count); uploaded per window, one entry per row
//                (`rows`) and one per segment (`segs`)
//
// The chain, with the kernel each stage uses:
//   K/V post + indexer appends   qsa_dec_kv_post_seg    (the indexer block of a segment appends its rows in order)
//   query post                   qsa_dec_q_post         (no per-sequence state: one call over all rows)
//   block scores                 qsa_block_scores_seg
//   top-k                        qsa_block_topk         (reads scores and steps only: one call over all rows)
//   residency (streamed slots)   kv_stream_resolve_seg  (each segment resolves into its own slot's map)
//   chunks and merge             qsa_dec_attn_seg
//   commit (tail restore + appends)  native_qsa_indexer_commit_seg
// A streamed slot (kv_mode 1, INT8) carries its residency map; a resident one a null map, which the resolve skips.
// Rows are `steps`-indexed as in the single-sequence kernels (the row's own position and counts), INT8 or FP16 pools.
#pragma once

#include "basalt/kernels/kv_stream.hpp"
#include "basalt/kernels/native_qsa_indexer.hpp"
#include "basalt/kernels/qsa.hpp"
#include "basalt/kernels/qsa_dec_fused.hpp"
#include "basalt/kernels/qsa_decode_attn.hpp"
#include "basalt/kernels/rope_scaling.hpp"

#include <cstdint>

#ifdef __CUDACC__
#define QSA_SEG_HD __host__ __device__
#else
#define QSA_SEG_HD
#endif

namespace basalt::kernels {

struct QsaSegSlot {
    uint16_t *k_pool = nullptr, *v_pool = nullptr;   ///< FP16 pools, or null
    int8_t *k_q = nullptr, *v_q = nullptr;            ///< INT8 codes
    uint16_t *k_scale = nullptr, *v_scale = nullptr;
    int32_t* page_table = nullptr;
    KvHostPools host;                                 ///< the host copy; all null when resident
    KvStreamMap map;                                  ///< a streamed slot's residency map; null `ctl` when resident
    float *tail = nullptr, *dead = nullptr, *pooled = nullptr;
    float* tail_snap = nullptr;                       ///< ((idx_block - 1) * idx_dim) floats: the tail before the window
    int32_t* block_pos = nullptr;
    int32_t max_cells = 0, pad = 0;

    QSA_SEG_HD QsaAttnPools pools() const {
        QsaAttnPools p;
        p.k_pool = k_pool; p.v_pool = v_pool; p.k_q = k_q; p.v_q = v_q; p.k_scale = k_scale; p.v_scale = v_scale;
        p.page_table = page_table;
        return p;
    }
};

struct QsaSeg {
    int32_t slot = 0;    ///< index into the layer's QsaSegSlot array
    int32_t first = 0;   ///< the segment's first row
    int32_t len = 0;     ///< the segment's row count
    int32_t pad = 0;
};

/// Device pointers. `rows[r]` for every row r, `segs[j]` for every segment j (distinct slots).  `n_seg` is the number
/// of entries the kernels visit; an entry with a negative slot is skipped, so one launch covers any split of the rows.
struct QsaSegTable {
    const QsaSegSlot* slots = nullptr;
    const QsaSeg* rows = nullptr;
    const QsaSeg* segs = nullptr;
    int n_rows = 0, n_seg = 0;
};

constexpr int kQsaSegMaxRows = 16;

/// qsa_dec_kv_post's contract with the per-sequence fields (k_q .. block_pos, tail_snap, max_cells) taken from the
/// row's slot; `a.tail_snap` non-null: each segment's indexer block first copies its slot's tail to the slot's
/// tail_snap. `n` is the window's row count.
void qsa_dec_kv_post_seg(const QsaDecKvPost& a, const QsaSegTable& t, int n, const QsaShapes& s,
                         const RopeScaling& scaling, void* stream);

/// qsa_block_scores over rows of different sequences: row r reads its slot's pooled / dead. nq <= kQsaSegMaxRows.
void qsa_block_scores_seg(const QsaSegTable& t, const float* q_idx, const int32_t* steps, int64_t nq, int64_t max_blocks,
                          const QsaShapes& s, float* scores, void* stream, bool skip_identity = false);

/// kv_stream_resolve over segments, INT8 pools: segment j's rows (their ids and steps from its first row on) resolve
/// into its slot's map and the misses are copied from that slot's host copy into its pool, for each slot what
/// kv_stream_resolve does over that slot's rows alone.  A segment with a negative slot or no rows, and a resident slot
/// (null map), are skipped.  Capturable.  Every streamed slot holds at least kKvStreamMinSlots pages.
void kv_stream_resolve_seg(const QsaSegTable& t, const int32_t* ids, const int32_t* steps, int64_t cap,
                           const QsaShapes& s, void* stream);

/// qsa_dec_attn over rows of different sequences (INT8 pools when `int8`, else FP16). Same outputs and scratch.
void qsa_dec_attn_seg(const float* q, const QsaSegTable& t, bool int8, const int32_t* ids, const int32_t* steps,
                      int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream,
                      const float* q_full = nullptr, float* attn_gated = nullptr, void* xq = nullptr);

struct QsaSegCommitLayer {
    const QsaSegSlot* slots = nullptr;   ///< this layer's slots
    const float* raw = nullptr;          ///< [row][idx_dim]: the window's raw indexer keys of this layer
    const float* gamma = nullptr;
};
struct QsaSegCommit {
    QsaSegCommitLayer layer[kQsaCommitMaxLayers];
    const QsaSeg* segs = nullptr;
    const int32_t* pos = nullptr;   ///< segment j's rows t: pos[slot * pos_stride + t] (< 0 skipped)
    int pos_stride = 0;
    float epsilon = 0.0f;
    int n_layers = 0, n_seg = 0;
};
/// Per layer and segment: the slot's tail restored from its tail_snap, then the segment's rows appended in order -
/// native_qsa_indexer_commit_layers on that slot alone, bitwise.
void native_qsa_indexer_commit_seg(const QsaSegCommit& a, const QsaShapes& s, const RopeScaling& scaling, void* stream);

}  // namespace basalt::kernels
