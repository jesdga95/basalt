#include "run.hpp"
#include "basalt/prefill/prefill.hpp"
#include "basalt/core/mtp.hpp"
#include "basalt/core/layout.hpp"
#include "basalt/kernels/native_qsa_indexer.hpp"
#include "basalt/kernels/qsa.hpp"
#include "basalt/kernels/kv_stream.hpp"
#include "basalt/kernels/kv_q4.hpp"
#include "basalt/core/layer.hpp"
#include "basalt/kernels/qsa_decode_attn.hpp"
#include "basalt/kernels/qsa_prompt_attn.hpp"
#include "basalt/kernels/qsa_select.hpp"
#include "basalt/prefill/kernels.hpp"
#include <algorithm>
#include <exception>
#include <string>

namespace basalt::prefill {
using namespace detail;

bool Prefill::Run::Chunk::qsa(const core::LayerView& v) {
    // ======================= QSA =======================
    const core::QsaState& st = ss.qsa_states[qsa_index];
    const core::WeightRef *wq = need(v, "attn_q.weight", err), *wk = need(v, "attn_k.weight", err),
                          *wv = need(v, "attn_v.weight", err), *wo = need(v, "attn_output.weight", err),
                          *wik = need(v, "indexer.k_proj.weight", err),
                          *wiq = need(v, "indexer.q_proj.weight", err),
                          *wqn = need(v, "attn_q_norm.weight", err), *wkn = need(v, "attn_k_norm.weight", err),
                          *wiqn = need(v, "indexer.q_norm.weight", err),
                          *wikn = need(v, "indexer.k_norm.weight", err);
    if (!wq || !wk || !wv || !wo || !wik || !wiq || !wqn || !wkn || !wiqn || !wikn) return false;
    if (!native_proj(m.gemm, wk, m.mixed_h, m.Kc, T, v.name("attn_k.weight"), err)) return false;
    if (!native_proj(m.gemm, wv, m.mixed_h, m.Vc, T, v.name("attn_v.weight"), err)) return false;
    if (!native_proj(m.gemm, wq, m.mixed_h, m.Qf, T, v.name("attn_q.weight"), err)) return false;
    if (!bf16_proj(m.gemm, wik, m.mixed_bf, m.idx_raw, T, v.name("indexer.k_proj.weight"), err)) return false;
    if (!bf16_proj(m.gemm, wiq, m.mixed_bf, m.q_idx, T, v.name("indexer.q_proj.weight"), err)) return false;
    rms_rows(m.Kc, (const float*) wkn->data, T * 2, 256, 256, EPS, m.cs);
    rope(m.Kc, T, 2, 256, 512, p0, basalt::kernels::rope_scaling(), m.cs);
    // KV streaming: this layer's cells [0, p0) come in from the host copy to the staging pool, and the
    // chunk's cells go to the host copy, the staging pool, and the VRAM slots of resident blocks
    const bool staged = st.kv_mode == 1;
    if (staged) {
        basalt::kernels::kv_stage_from_host(pools_of(m.stage, m.ident_table), st.host,
                                            core::qsa_kv_format(st),
                                            (p0 + s.page_size - 1) / s.page_size, s, m.cs);
    }
    const basalt::kernels::KvHostPools* host_w = &st.host;
    if (st.kv_hybrid) {   // K8V4: K INT8 unrotated, V rotated Q4_0 (only V and the output rotate)
        basalt::kernels::fwht256_inplace_cuda(m.Vc, T * 2, m.cs);
        kv_append(m.Kc, m.Kc, T, p0, st.page_table, s.page_size, nullptr, nullptr,
                  st.k_q, st.k_q, st.k_scale, st.k_scale, m.cs, nullptr,   // mode 0: no host mirror
                  staged ? &m.stage : nullptr);
        basalt::kernels::kv_append_q4(st.v_q4, st.v_q4, st.page_table, p0, T, m.Vc, m.Vc, s, m.cs,
                                      nullptr, staged ? &m.stage : nullptr);
    } else {
        if (st.kv_rot) {   // rotated K and V (kv_q4.hpp), the queries below too, the output back
            basalt::kernels::fwht256_inplace_cuda(m.Kc, T * 2, m.cs);
            basalt::kernels::fwht256_inplace_cuda(m.Vc, T * 2, m.cs);
        }
        if (st.kv_q4)
            basalt::kernels::kv_append_q4(st.k_q4, st.v_q4, st.page_table, p0, T, m.Kc, m.Vc, s, m.cs,
                                          host_w, staged ? &m.stage : nullptr);
        else
            kv_append(m.Kc, m.Vc, T, p0, st.page_table, s.page_size, st.kv_int8 ? nullptr : st.k_pool,
                      st.kv_int8 ? nullptr : st.v_pool, st.k_q, st.v_q, st.k_scale, st.v_scale, m.cs,
                      host_w, staged ? &m.stage : nullptr);
    }
    split_q(m.Qf, m.q, T, m.cs);
    rms_rows(m.q, (const float*) wqn->data, T * 24, 256, 256, EPS, m.cs);
    rope(m.q, T, 24, 256, 6144, p0, basalt::kernels::rope_scaling(), m.cs);
    if (st.kv_rot) basalt::kernels::fwht256_inplace_cuda(m.q, T * 24, m.cs);
    rms_rows(m.q_idx, (const float*) wiqn->data, T * 4, 128, 128, EPS, m.cs);
    rope(m.q_idx, T, 4, 128, 512, p0, basalt::kernels::rope_scaling(), m.cs);
    // the indexer appends, token by token; then scores + selection for many queries at once:
    // a query reads completed blocks (final once completed) and `dead` for its own tail block
    const basalt::kernels::QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
    // the chunk's appends in three launches instead of one per token (the same end state:
    // the queries below read it only after the whole chunk is appended).
    try {
        basalt::kernels::native_qsa_indexer_append_batch(m.idx_raw, T, p0, 0, (const float*) wikn->data,
                                                         EPS, ib, s, st.max_cells,
                                                         basalt::kernels::rope_scaling(), m.cs);
    } catch (const std::exception& e) { err = std::string("prefill indexer: ") + e.what(); return false; }
    for (int64_t t0 = 0; t0 < T; t0 += m.sel_batch) {
        const int64_t nb = std::min(m.sel_batch, T - t0);
        const int32_t* steps0 = m.steps_dev + t0 * basalt::kernels::kStepCount;
        // the grid reaches the batch's last query's n_bid (they rise with the position)
        const int64_t active = (int64_t) m.steps_host[(size_t) ((t0 + nb - 1) * basalt::kernels::kStepCount +
                                                                basalt::kernels::kStepNBid)] + 1;
        // the scores on tensor cores (3xTF32: FP32-level, not bitwise)
        if (!basalt::kernels::qsa_block_scores_tc(st.idx_pooled, st.idx_dead, m.q_idx + t0 * 512,
                                                             steps0, nb, m.max_blocks, s, m.sel_scores,
                                                             m.cs, active))
            basalt::kernels::qsa_block_scores(st.idx_pooled, st.idx_dead, m.q_idx + t0 * 512, steps0, nb,
                                              m.max_blocks, s, m.sel_scores, m.cs, active);
        basalt::kernels::qsa_block_topk(m.sel_scores, steps0, nb, m.max_blocks, m.cap, s,
                                        m.sel_ids + t0 * m.cap, m.cs, active);
    }
    const basalt::kernels::QsaAttnPools pools = staged ? pools_of(m.stage, m.ident_table)
                                                       : core::qsa_attn_pools(st);
    // the whole chunk on tensor cores, one block per (query, KV head), FP32-level
    // accuracy but not bitwise (qsa_prompt_attn.hpp). Q4_0 KV (or a shape it does not take): the
    // decode kernel, 32 queries at a time (K8V4 runs the tensor kernel's mode 3: INT8 K,
    // V dequantized from its q4_0 blocks to fp16 at gather)
    if (!basalt::kernels::qsa_prompt_attn_batch(m.q, pools, m.sel_ids, m.steps_dev, m.cap, s,
                                                            m.attn, T, m.cs))
        for (int64_t t0 = 0; t0 < T; t0 += m.attn_batch) {
            const int64_t nb = std::min(m.attn_batch, T - t0);
            basalt::kernels::qsa_decode_attn_batch(m.q + t0 * ZV, pools, m.sel_ids + t0 * m.cap,
                                                   m.steps_dev + t0 * basalt::kernels::kStepCount, m.cap,
                                                   s, m.attn_scratch, m.attn + t0 * ZV, nb, m.cs);
        }
    if (st.kv_rot || st.kv_hybrid) basalt::kernels::fwht256_inplace_cuda(m.attn, T * 24, m.cs);
    gate_attn(m.attn, m.Qf, m.attn_h, T, m.cs);
    if (!native_proj(m.gemm, wo, m.attn_h, m.bo, T, v.name("attn_output.weight"), err)) return false;
    ++qsa_index;
    return true;
}

}  // namespace basalt::prefill
