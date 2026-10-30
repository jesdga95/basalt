#include "common.hpp"
#include "record.hpp"
#include "expert_plan.hpp"
#include "basalt/core/verify.hpp"
#include "basalt/kernels/bf16_gemv.hpp"
#include "basalt/kernels/elementwise.hpp"
#include "basalt/kernels/kv_q4.hpp"
#include "basalt/kernels/kv_q8.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/native_qsa.hpp"
#include "basalt/kernels/native_qsa_indexer.hpp"
#include "basalt/kernels/native_rope.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/qsa.hpp"
#include "basalt/kernels/qsa_decode_attn.hpp"
#include "basalt/kernels/qsa_dec_attn.hpp"
#include "basalt/kernels/qsa_dec_fused.hpp"
#include "basalt/kernels/qsa_select.hpp"

namespace basalt::core {
using namespace verify_detail;

bool Verifier::WindowRecorder::qsa(int64_t l, const LayerView& v, bool read_q8) {
    using namespace basalt::kernels;
    const int n = T;
    float* xm = v_.mixed_;
    // ======================= QSA =======================
    const int64_t qi = qsa_idx[(size_t) l];
    const QsaState& st = ss.qsa_states[qi];
    const WeightRef *wik = need(v, "indexer.k_proj.weight", err), *wq = need(v, "attn_q.weight", err),
                    *wk = need(v, "attn_k.weight", err), *wv = need(v, "attn_v.weight", err),
                    *wo = need(v, "attn_output.weight", err), *wiq = need(v, "indexer.q_proj.weight", err),
                    *wqn = need(v, "attn_q_norm.weight", err), *wkn = need(v, "attn_k_norm.weight", err),
                    *wiqn = need(v, "indexer.q_norm.weight", err), *wikn = need(v, "indexer.k_norm.weight", err);
    if (!wik || !wq || !wk || !wv || !wo || !wiq || !wqn || !wkn || !wiqn || !wikn) return false;
    if (!native_of(wq, v.name("attn_q.weight"), err) || !native_of(wk, v.name("attn_k.weight"), err) ||
        !native_of(wv, v.name("attn_v.weight"), err) || !native_of(wo, v.name("attn_output.weight"), err))
        return false;
    auto norm_rope = [&](float* data, const WeightRef* norm, int rows, int cols, const int32_t* pos) {
        native_qsa_rms_norm_weighted(data, (const float*) norm->data, data, cols, rows, EPS, cs);
        native_rope_apply(data, data, rows, cols, (int) s.n_rot, rope_scaling(), pos, cs);
    };
    float* idx_raw = v_.idx_raw_L_ + (size_t) qi * MT * ID;
    // the per-token GEMVs / norms / RoPEs / copies of this layer as one launch over the window's rows each
    // - row-wise identical arithmetic
    const bool qb = n > 1 && !st.kv_q4;
    // the K norm + RoPE, the INT8 appends, the tail snapshot and the indexer appends of the group's rows in
    // one launch, the same bits
    const bool fused_pre = st.kv_int8 && !st.kv_rot && !st.kv_hybrid && !st.kv_q4 &&
                           qsa_dec_fused_supported(s);
    auto kv_post_args = [&] {
        QsaDecKvPost kp;
        kp.kcur = v_.kcur_; kp.vcur = v_.vcur_; kp.pos_k = pos_k;
        kp.k_norm = (const float*) wkn->data; kp.steps = v_.step_;
        kp.k_q = st.k_q; kp.v_q = st.v_q; kp.k_scale = st.k_scale; kp.v_scale = st.v_scale;
        kp.page_table = st.page_table; kp.host = st.host;
        kp.idx_raw = idx_raw; kp.idx_k_norm = (const float*) wikn->data;
        kp.tail = st.idx_tail; kp.dead = st.idx_dead; kp.pooled = st.idx_pooled; kp.block_pos = st.idx_block_pos;
        kp.tail_snap = v_.batch_rec_ ? v_.tail_snap_b_ : v_.tail_snap_ + (size_t) qi * TS;
        kp.max_cells = (int) st.max_cells; kp.eps = EPS;
        return kp;
    };
    // a batch window's rows are different sequences: the per-slot pointers come from the layer's table
    const QsaSegTable seg_tab{v_.batch_rec_ ? v_.d_qslots_[(size_t) qi] : nullptr, v_.d_qrows_, v_.d_qsegs_, (int) n, (int) n};
    auto kv_post = [&](const QsaDecKvPost& a, cudaStream_t q) {
        if (v_.batch_rec_) qsa_dec_kv_post_seg(a, seg_tab, (int) n, s, rope_scaling(), q);
        else qsa_dec_kv_post(a, n, s, rope_scaling(), q);
    };
    auto q_post_args = [&] {
        QsaDecQPost qp;
        qp.q_full = v_.qfull_; qp.qcur = v_.qcur_; qp.pos_q = v_.pos_;
        qp.q_norm = (const float*) wqn->data; qp.q_idx = v_.qidx_; qp.pos_i = pos_i;
        qp.idx_q_norm = (const float*) wiqn->data; qp.eps = EPS;
        return qp;
    };
    // K, V and the indexer key/query each on a branch of their own, beside the query (one row: the
    // multi-row GEMV is the one-row one)
    const bool qsaf_wide = fused_pre && (qb || n == 1) && v_.qs_[0] != nullptr;
    // the indexer appends gate the scores, the K/V appends the attention chunks: each on its own branch,
    // joined after the query post, which runs beside them
    if (qsaf_wide) {
        {
            PdlScope ps(pdl);   // after the hc read
            if (!read_q8) native_quantize_q8_1(xm, v_.xq_, (int) N, n, cs);
        }
        cudaEventRecord(v_.qev_[0], cs);
        for (cudaStream_t q : v_.qs_) cudaStreamWaitEvent(q, v_.qev_[0], 0);
        {
            PdlScope pz(pdl, true);
            dense_proj(32, wv->native_type, wv->native_data, v_.xq_, v_.vcur_, (int) N, (int) (NKV * HD),
                       n, v_.qs_[1]);
        }
        cudaEventRecord(v_.qev_[1], v_.qs_[1]);
        {
            PdlScope pz(pdl, true);
            bf16_gemv_fp32_mmvf_multi(v_.mixed_, N, (const uint16_t*) wik->data, idx_raw, ID, N, ID,
                                      n, v_.qs_[2]);
        }
        {
            QsaDecKvPost ia = kv_post_args();
            ia.part = 2;
            kv_post(ia, v_.qs_[2]);
        }
        cudaEventRecord(v_.qev_[2], v_.qs_[2]);
        {
            PdlScope pz(pdl, true);
            bf16_gemv_fp32_mmvf_multi(v_.mixed_, N, (const uint16_t*) wiq->data, v_.qidx_,
                                      IQ * ID, N, IQ * ID, n, v_.qs_[1]);
        }
        {
            PdlScope pz(pdl, true);
            dense_proj(32, wk->native_type, wk->native_data, v_.xq_, v_.kcur_, (int) N, (int) (NKV * HD),
                       n, v_.qs_[0]);
        }
        cudaStreamWaitEvent(v_.qs_[0], v_.qev_[1], 0);
        {
            QsaDecKvPost ka = kv_post_args();
            ka.part = 1;
            kv_post(ka, v_.qs_[0]);
        }
        cudaEventRecord(v_.qev_[3], v_.qs_[0]);
        cudaEventRecord(v_.qev_[4], v_.qs_[1]);
        {
            PdlScope pq(pdl);   // after the q8_1 kernel
            dense_proj(8, wq->native_type, wq->native_data, v_.xq_, v_.qfull_, (int) N,
                       (int) (NH * 2 * HD), n, cs);
        }
        cudaStreamWaitEvent(cs, v_.qev_[4], 0);
        for (int i = 7; i <= 9; ++i) stamp(l, i);
        qsa_dec_q_post(q_post_args(), n, s, rope_scaling(), cs);
        cudaStreamWaitEvent(cs, v_.qev_[3], 0);
        cudaStreamWaitEvent(cs, v_.qev_[2], 0);
    } else {
    {
        PdlScope ps(pdl);   // after the hc read, up to the K/V norms
        if (!read_q8) native_quantize_q8_1(xm, v_.xq_, (int) N, n, cs);
        if (qb) bf16_gemv_fp32_mmvf_multi(v_.mixed_, N, (const uint16_t*) wik->data, idx_raw, ID, N, ID, n, cs);
        else for (int t = 0; t < T; ++t)
            bf16_gemv_fp32_mmvf(v_.mixed_ + t * N, (const uint16_t*) wik->data, idx_raw + t * ID, (int) N, (int) ID, cs);
        stamp(l, 7);
        dense_proj(32, wk->native_type, wk->native_data, v_.xq_, v_.kcur_, (int) N, (int) (NKV * HD), n, cs);
        dense_proj(32, wv->native_type, wv->native_data, v_.xq_, v_.vcur_, (int) N, (int) (NKV * HD), n, cs);
    }
    if (fused_pre) {
        kv_post(kv_post_args(), cs);
        stamp(l, 8);
    } else {
        if (qb) norm_rope(v_.kcur_, wkn, (int) (n * NKV), (int) HD, pos_k);
        else for (int t = 0; t < T; ++t) norm_rope(v_.kcur_ + t * NKV * HD, wkn, (int) NKV, (int) HD, v_.pos_ + t * NH);
        if (st.kv_rot) {   // K and V rotated before they are stored (kv_q4.hpp)
            fwht256_inplace_cuda(v_.kcur_, (int64_t) n * NKV, cs);
            fwht256_inplace_cuda(v_.vcur_, (int64_t) n * NKV, cs);
        } else if (st.kv_hybrid) {   // K8V4: only V is rotated
            fwht256_inplace_cuda(v_.vcur_, (int64_t) n * NKV, cs);
        }
        stamp(l, 8);
        copy_from_mapped(v_.tail_snap_ + (size_t) qi * TS, st.idx_tail, TS, cs);
        for (int t = 0; t < T; ++t) {
                                    const int32_t* step_t = v_.step_ + t * kStepCount;
            if (st.kv_hybrid) {   // K8V4: the unused half's lanes folded onto the used pool (layer.cpp)
                kv_append_q8_step(st.k_q, st.k_q, st.k_scale, st.k_scale, st.page_table, step_t,
                                  v_.kcur_ + t * NKV * HD, v_.kcur_ + t * NKV * HD, s, cs, nullptr);
                kv_append_q4_step(st.v_q4, st.v_q4, st.page_table, step_t, v_.vcur_ + t * NKV * HD,
                                  v_.vcur_ + t * NKV * HD, s, cs, nullptr);
            } else if (st.kv_q4)
                kv_append_q4_step(st.k_q4, st.v_q4, st.page_table, step_t, v_.kcur_ + t * NKV * HD,
                                  v_.vcur_ + t * NKV * HD, s, cs, &st.host);
            else if (st.kv_int8)
                kv_append_q8_step(st.k_q, st.v_q, st.k_scale, st.v_scale, st.page_table, step_t,
                                  v_.kcur_ + t * NKV * HD, v_.vcur_ + t * NKV * HD, s, cs, &st.host);
            else
                kv_append_step(st.k_pool, st.v_pool, st.page_table, step_t, v_.kcur_ + t * NKV * HD,
                               v_.vcur_ + t * NKV * HD, s, cs, &st.host);
        }
        for (int t = 0; t < T; ++t) {
            const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
            native_qsa_indexer_append(idx_raw + t * ID, v_.step_ + t * kStepCount + kStepPos, 0,
                                      (const float*) wikn->data, EPS, ib, s, st.max_cells,
                                      rope_scaling(), cs);
        }
    }
    stamp(l, 9);
    {
        PdlScope pq(pdl2 && fused_pre && qb);   // PDL2: after the K/V post kernel
        dense_proj(8, wq->native_type, wq->native_data, v_.xq_, v_.qfull_, (int) N,
                   (int) (NH * 2 * HD), n, cs);
        if (fused_pre && qb)
            bf16_gemv_fp32_mmvf_multi(v_.mixed_, N, (const uint16_t*) wiq->data, v_.qidx_,
                                      IQ * ID, N, IQ * ID, n, cs);
    }
    if (fused_pre) {   // the query split, norms and RoPEs in one launch
        if (qb) {}   // the indexer query projection ran above
        else for (int t = 0; t < T; ++t)
            bf16_gemv_fp32_mmvf(v_.mixed_ + t * N, (const uint16_t*) wiq->data, v_.qidx_ + t * IQ * ID, (int) N,
                                (int) (IQ * ID), cs);
        qsa_dec_q_post(q_post_args(), n, s, rope_scaling(), cs);
    } else if (qb) {
        if (cudaMemcpy2DAsync(v_.qcur_, (size_t) HD * 4, v_.qfull_, (size_t) HD * 2 * 4,
                              (size_t) HD * 4, (size_t) (n * NH), cudaMemcpyDeviceToDevice, cs) != cudaSuccess) {
            err = "verify: the q/gate split failed";
            return false;
        }
        norm_rope(v_.qcur_, wqn, (int) (n * NH), (int) HD, v_.pos_);
        if (st.kv_rot) fwht256_inplace_cuda(v_.qcur_, (int64_t) n * NH, cs);   // <Hq, Hk> = <q, k>
        bf16_gemv_fp32_mmvf_multi(v_.mixed_, N, (const uint16_t*) wiq->data, v_.qidx_, IQ * ID,
                                  N, IQ * ID, n, cs);
        norm_rope(v_.qidx_, wiqn, (int) (n * IQ), (int) ID, pos_i);
    } else {
    for (int t = 0; t < T; ++t) {
        float* qc = v_.qcur_ + t * NH * HD;
        if (cudaMemcpy2DAsync(qc, (size_t) HD * 4, v_.qfull_ + t * NH * 2 * HD, (size_t) HD * 2 * 4,
                              (size_t) HD * 4, (size_t) NH, cudaMemcpyDeviceToDevice, cs) != cudaSuccess) {
            err = "verify: the q/gate split failed";
            return false;
        }
        norm_rope(qc, wqn, (int) NH, (int) HD, v_.pos_ + t * NH);
        if (st.kv_rot) fwht256_inplace_cuda(qc, NH, cs);   // <Hq, Hk> = <q, k>
    }
    for (int t = 0; t < T; ++t) {
        float* qx = v_.qidx_ + t * IQ * ID;
        bf16_gemv_fp32_mmvf(v_.mixed_ + t * N, (const uint16_t*) wiq->data, qx, (int) N, (int) (IQ * ID), cs);
        norm_rope(qx, wiqn, (int) IQ, (int) ID, v_.pos_ + t * NH);
    }
    }
    }
    stamp(l, 10);
    bool fused_gate = false;
    bool attn_q8 = false;   // the merge wrote the q8_1 rows
    if (v_.batch_rec_) {   // each row selects and attends over its own slot's K/V, from the layer's table
        qsa_block_scores_seg(seg_tab, v_.qidx_, v_.step_, n, v_.max_blocks_, s, v_.scores_, cs, true);
        qsa_block_topk(v_.scores_, v_.step_, n, v_.max_blocks_, v_.cap_, s, v_.sel_, cs);
        stamp(l, 11);
        stamp(l, 12);
        fused_gate = true;
        attn_q8 = true;
        qsa_dec_attn_seg(v_.qcur_, seg_tab, true, v_.sel_, v_.step_, v_.cap_, s, v_.attn_scratch_, v_.attn_, n, cs, v_.qfull_,
                         v_.attn32_, v_.xq_);
    } else {
    {
        {
            PdlScope pb(pdl);   // after the query post kernel
            qsa_block_scores(st.idx_pooled, st.idx_dead, v_.qidx_, v_.step_, n,
                             v_.max_blocks_, s, v_.scores_, cs, -1, true);
        }
        PdlScope pt(pdl);   // after the scores kernel
        qsa_block_topk(v_.scores_, v_.step_, n, v_.max_blocks_, v_.cap_, s,
                       v_.sel_, cs);
    }
    stamp(l, 11);
    // KV streaming: the n selections' blocks resident (device-side, inside the graph)
    qsa_kv_resolve(st, *v_.g_, v_.sel_, v_.step_, n, v_.cap_, cs);
    stamp(l, 12);
    const QsaAttnPools pools = qsa_attn_pools(st);
    // staged chunks and the gate in the merge, the same bits
    fused_gate = !st.kv_rot && !st.kv_hybrid &&
                 qsa_dec_attn_supported(pools, s);
    attn_q8 = fused_gate && !st.kv_rot && !st.kv_hybrid;
    PdlScope pa(pdl);   // the chunks after the top-k, the merge after them
    if (fused_gate)
        qsa_dec_attn(v_.qcur_, pools, v_.sel_, v_.step_, v_.cap_, s,
                     v_.attn_scratch_, v_.attn_, n, cs,
                     v_.qfull_, v_.attn32_, attn_q8 ? v_.xq_ : nullptr);
    else
    qsa_decode_attn_batch(v_.qcur_, pools, v_.sel_, v_.step_, v_.cap_,
                          s, v_.attn_scratch_, v_.attn_, n, cs);
    }
    stamp(l, 13);
    if (st.kv_rot || st.kv_hybrid) fwht256_inplace_cuda(v_.attn_, (int64_t) n * NH, cs);   // back: H^-1 = H
    if (fused_gate) {}   // the merge wrote attn32_
    else if (qb) native_qsa_gate_apply(v_.attn_, v_.qfull_, v_.attn32_,
                                  (int) (n * NH), (int) HD, cs);
    else
    for (int t = 0; t < T; ++t) {
        native_qsa_gate_apply(v_.attn_ + t * NH * HD, v_.qfull_ + t * NH * 2 * HD, v_.attn32_ + t * NH * HD,
                              (int) NH, (int) HD, cs);
    }
    stamp(l, 14);
    PdlScope ps(pdl);   // after the gate kernel
    if (!attn_q8) native_quantize_q8_1(v_.attn32_, v_.xq_, (int) (NH * HD), n, cs);
    dense_proj(16, wo->native_type, wo->native_data, v_.xq_, v_.bo_, (int) (NH * HD), (int) N, n, cs);
    return true;
}

}  // namespace basalt::core
