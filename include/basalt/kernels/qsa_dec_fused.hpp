// include/basalt/kernels/qsa_dec_fused.hpp - the per-token work of a QSA decode layer between its projections, for
// the T rows of a window in one launch each, with the arithmetic of the separate kernels bit for bit:
//
//   qsa_dec_kv_post   K: RMS norm + RoPE (native_qsa_rms_norm_weighted, native_rope_apply); K and V of every row into
//                     the INT8/FP16 cache and its host copy; the indexer tail snapshot, then the rows'
//                     indexer keys in order (native_qsa_indexer_append)
//   qsa_dec_q_post    the query heads out of q_full with norm + RoPE, the indexer query heads with norm + RoPE
//
// INT8 or FP16 K/V without rotation (qsa_dec_fused_supported).
#pragma once

#include "basalt/kernels/kv_stream.hpp"
#include "basalt/kernels/qsa.hpp"
#include "basalt/kernels/rope_scaling.hpp"

#include <cstdint>

namespace basalt::kernels {

struct QsaDecKvPost {
    float* kcur = nullptr;              ///< [n][n_head_kv][head_dim], normed and rotated in place
    const float* vcur = nullptr;        ///< [n][n_head_kv][head_dim]
    const int32_t* pos_k = nullptr;     ///< [n][pos_stride] rotary positions of the K rows
    int pos_stride = 0;                 ///< 0: n_head_kv
    const float* k_norm = nullptr;      ///< [head_dim]
    const int32_t* steps = nullptr;     ///< [n][kStepCount]: each row's cell at kStepPos
    int8_t *k_q = nullptr, *v_q = nullptr;
    uint16_t *k_scale = nullptr, *v_scale = nullptr;
    const int32_t* page_table = nullptr;
    KvHostPools host;                   ///< the host copy, when the cache streams
    const float* idx_raw = nullptr;     ///< [n][idx_dim] raw indexer keys; null: no indexer (a dense layer)
    const float* idx_k_norm = nullptr;  ///< [idx_dim]
    float *tail = nullptr, *dead = nullptr, *pooled = nullptr;
    int32_t* block_pos = nullptr;
    float* tail_snap = nullptr;         ///< non-null: the tail ((idx_block - 1) * idx_dim) copied here first
    int max_cells = 0;
    float eps = 1e-6f;
    int part = 0;                       ///< 0: K/V rows and indexer; 1: K/V rows only; 2: indexer only
    // ---- Jev: resident FP16 segment appends use the existing conversion ----
    uint16_t *k_pool = nullptr, *v_pool = nullptr;
    // ---- end Jev ----
};

struct QsaDecQPost {
    const float* q_full = nullptr;      ///< [n][n_head][2 * head_dim]: query, gate
    float* qcur = nullptr;              ///< [n][n_head][head_dim]
    const int32_t* pos_q = nullptr;     ///< [n][n_head]
    const float* q_norm = nullptr;      ///< [head_dim]
    float* q_idx = nullptr;             ///< [n][idx_n_head][idx_dim], normed and rotated in place; null: none
    const int32_t* pos_i = nullptr;     ///< [n][idx_n_head]
    const float* idx_q_norm = nullptr;  ///< [idx_dim]
    float eps = 1e-6f;
};

bool qsa_dec_fused_supported(const QsaShapes& s);
void qsa_dec_kv_post(const QsaDecKvPost& a, int n, const QsaShapes& s, const RopeScaling& scaling, void* stream);
void qsa_dec_q_post(const QsaDecQPost& a, int n, const QsaShapes& s, const RopeScaling& scaling, void* stream);

}  // namespace basalt::kernels
