#include "common.hpp"

namespace basalt::core {

using namespace mtp_detail;

namespace {

// The round and chain graphs run independent branches on a second stream (the embedding beside the hidden input,
// K/V beside Q, the shared expert beside the routed ones, the next step's hidden input beside the head); the same
// kernels on the same inputs, so the same bits.
constexpr bool dec_draft_par() { return true; }
// Every hyper-connection read of those graphs (and the MLP read's router) as the one-kernel read, whose bits are
// fused_gr_read_multi's.
bool dec_draft_hcf() {
    static const bool on = basalt::kernels::hc_fused_supported();
    return on;
}

// The one-kernel hyper-connection read of the rows fa[0..T) (rows at the layer's fixed strides)
basalt::kernels::HcFusedArgs hcf_args(const basalt::kernels::FusedGrArgs* fa, int T, float* xn) {
    basalt::kernels::HcFusedArgs h;
    h.T = T; h.R = fa[0].R; h.R_out = fa[0].R_out; h.apply = fa[0].apply;
    h.bo_prev = fa[0].bo_prev; h.inj_prev = fa[0].inj_prev;
    h.w_norm = fa[0].w_norm; h.w_down = fa[0].w_down; h.w_up = fa[0].w_up; h.w_inject = fa[0].w_inject;
    h.eps = fa[0].eps; h.lo = fa[0].lo; h.rs = fa[0].rs; h.inject_out = fa[0].w_inject ? fa[0].inject_out : nullptr;
    h.mixed = fa[0].mixed; h.xn = xn;
    return h;
}

}

// The hidden input branch: h2_ <- fc_hidden(rms(R) * w) per stream, T rows.
void MtpDrafter::record_hidden(const float* R, int T, uint8_t* xq, cudaStream_t cs) {
    using namespace basalt::kernels;
    const int64_t N = g_->n_embd, HC = g_->hc;
    native_qsa_rms_norm_weighted(R, f32("pre_fc_norm_hidden.weight"), hn_, (int) (HC * N), T, EPS, cs);
    for (int c0 = 0; c0 < T * HC; c0 += 8) {
        const int nc = (int) std::min<int64_t>(8, T * HC - c0);
        native_quantize_q8_1(hn_ + (size_t) c0 * N, xq, (int) N, nc, cs);
        native_mmvq(GGML_Q8_0, q8("fc_hidden.weight"), xq, h2_ + (size_t) c0 * N, (int) N, (int) N, nc, cs);
    }
}

// The layer for T rows.  full = false stops after the K/V append (the prompt only needs the cache).
bool MtpDrafter::record_forward(int T, int step_row0, cudaStream_t cs, std::string& err, bool reuse_row) {
    using namespace basalt::kernels;
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    // step_row0 >= 0: the full layer on step rows [step_row0, +T); step_row0 < 0: K/V only on rows [-1 - step_row0, +T)
    const bool full = step_row0 >= 0;
    const int row0 = full ? step_row0 : -1 - step_row0;
    if (full && T != 1) { err = "mtp: the full layer runs one row at a time (its attention scratch is sized for one)"; return false; }
    if (reuse_row && !full) { err = "mtp: only the full layer reuses a catch-up row"; return false; }
    const int64_t N = g.n_embd, HC = g.hc, K = ss.k, NH = g.n_head, HD = g.head_dim, NKV = g.n_head_kv;
    const QsaShapes s = shapes_of(g);
    const GrShapes gs{g.n_embd, g.hc, g.hc_lr};
    const int32_t* step = step_ + row0 * 4;
    const int32_t* pos = pos_ + row0 * NH;
    const bool kv_fused = st_.kv_int8 && !st_.kv_q4 && !st_.kv_rot &&
                          qsa_dec_fused_supported(s);
    const bool kv_par = par_ && full && !reuse_row && kv_fused;
    // chain-step kernels that start while the one before them still runs (the paths below all use the fused kernels)
    const bool pdl_ok = kv_par && !st_.kv_rot && qsa_dec_fused_supported(s) &&
                        qsa_dec_attn_supported(qsa_attn_pools(st_), s) && par_ && dec_draft_hcf() && have_pre_;
    std::unique_ptr<PdlScope> psc;
    auto pscope = [&](bool first_plain) {
        psc.reset();
        if (pdl_ok) psc.reset(new PdlScope(true, first_plain));
    };
    auto pscope_off = [&] { psc.reset(); };
    try {
        auto norm_rope = [&](float* data, const float* gamma, int rows, int cols, const int32_t* p) {
            native_qsa_rms_norm_weighted(data, gamma, data, cols, rows, EPS, cs);
            native_rope_apply(data, data, rows, cols, (int) s.n_rot, rope_scaling(), p, cs);
        };
        if (reuse_row) {
            // the catch-up already ran this row's input branches, attention hyper-connection and K/V append, with
            // the same per-row arithmetic: its R, injection and mixed rows become row 0
            mtp_take_row(R_, HC * N, inj_, HC, mixed_, N, row_, cs);
            native_quantize_q8_1(mixed_, xq_, (int) N, T, cs);
        } else {
            // ---- the two input branches
            const WeightRef* we = wt_->find("token_embd.weight");
            if (!we) { err = "mtp: token_embd.weight is missing"; return false; }
            // par_: the embedding branch on the second stream, beside the hidden one (or the hidden input the
            // previous step already computed beside its head)
            if (par_) fork(cs);
            const cudaStream_t ce = par_ ? cs2_ : cs;
            uint8_t* const xqe = par_ ? xq2_ : xq_;
            pscope(true);   // the embedding branch follows the fork
            if (const NativeEmbed* ne = native_embed()) {   // the GGUF-form table
                ne->gather_dev(tok_, T, emb_, ce);
            } else {
                const auto* codes = (const uint8_t*) we->data;
                const auto* scales = (const float*) (codes + we->codes_bytes);
                const auto* offsets = we->has_offset ? (const float*) (codes + we->codes_bytes + we->scales_bytes) : nullptr;
                embedding_gather_dev(codes, scales, offsets, tok_, T, we->ne0, we->code_bits, we->code_bias, we->group_elems,
                                     (uint64_t) (we->ne0 / (8 / we->code_bits)), (uint64_t) (we->ne0 / we->group_elems), emb_, ce);
            }
            native_qsa_rms_norm_weighted(emb_, f32("pre_fc_norm_embedding.weight"), en_, (int) N, T, EPS, ce);
            native_quantize_q8_1(en_, xqe, (int) N, T, ce);
            native_mmvq(GGML_Q8_0, q8("fc_embedding.weight"), xqe, e2_, (int) N, (int) N, T, ce);
            pscope_off();
            if (!(have_pre_ && full)) record_hidden(Rin_, T, xq_, cs);
            if (par_) join(cs);
            // the round's catch-up brings the attention weights into L2; a chain step prefetches none, so with its
            // query and head weights streamed the layer's other weights stay in L2 from step to step
            if (!full) pf_fork(cs);
            pscope(false);
            add_streams_broadcast(h2_, e2_, R_, N, (int) HC, T, cs);
            // ---- the attention hyper-connection
            bool q8_done = false;
            {
                FusedGrArgs fa[kFusedGrMaxT];
                for (int t = 0; t < T; ++t) {
                    fa[t].R = R_ + (size_t) t * HC * N; fa[t].R_out = R_ + (size_t) t * HC * N; fa[t].apply = false;
                    fa[t].w_norm = f32("attn_hyper_connection.hc_norm.weight");
                    fa[t].w_down = bf16("attn_hyper_connection.input_mix_weight_down.weight");
                    fa[t].w_up = bf16("attn_hyper_connection.input_mix_weight_up.weight");
                    fa[t].w_inject = bf16("attn_hyper_connection.block_inject_weight.weight");
                    fa[t].eps = EPS; fa[t].lo = lo_ + t * g.hc_lr; fa[t].rs = rs_ + t * HC;
                    fa[t].inject_out = inj_ + t * HC; fa[t].mixed = mixed_ + t * N;
                }
                if (par_ && dec_draft_hcf()) {
                    HcFusedArgs h = hcf_args(fa, T, xn_);
                    if (hc_fused_q8_rows_ok()) { h.xq = xq_; q8_done = true; }   // mixed's q8_1 rows from the read
                    h.next_early = pdl_ok && q8_done;   // the query projection's blocks start with the read
                    hc_fused_read(h, cs);
                } else if (T == 1)   // one row: the per-row read (its own rounding)
                    gr_read(R_, fa[0].w_norm, fa[0].w_down, fa[0].w_up, fa[0].w_inject, EPS, gs, ss.block.gr, mixed_, inj_,
                            cs);
                else
                    fused_gr_read_multi(fa, T, xn_, cs);
            }
            // ---- attention: K/V into the layer's own cache, then (full) dense attention over every cell
            if (!q8_done) native_quantize_q8_1(mixed_, xq_, (int) N, T, cs);
            if (kv_par) fork(cs);   // K/V on the second stream, beside Q
            const cudaStream_t ckv = kv_par ? cs2_ : cs;
            pscope(true);   // the K/V branch follows the fork
            native_mmvq(GGML_Q8_0, q8("self_attn.k_proj.weight"), xq_, kcur_, (int) N, (int) (NKV * HD), T, ckv);
            native_mmvq(GGML_Q8_0, q8("self_attn.v_proj.weight"), xq_, vcur_, (int) N, (int) (NKV * HD), T, ckv);
            if (kv_fused) {   // the rows' K norm + RoPE and appends in one launch
                QsaDecKvPost kp;
                kp.kcur = kcur_; kp.vcur = vcur_; kp.pos_k = pos; kp.pos_stride = (int) NH;
                kp.k_norm = f32("self_attn.k_norm.weight"); kp.steps = step;
                kp.k_q = st_.k_q; kp.v_q = st_.v_q; kp.k_scale = st_.k_scale; kp.v_scale = st_.v_scale;
                kp.page_table = st_.page_table; kp.host = st_.host; kp.eps = EPS;
                qsa_dec_kv_post(kp, T, s, rope_scaling(), ckv);
            } else
            for (int t = 0; t < T; ++t) {
                norm_rope(kcur_ + t * NKV * HD, f32("self_attn.k_norm.weight"), (int) NKV, (int) HD, pos + t * NH);
                if (st_.kv_rot) {   // rotated K and V (kv_q4.hpp): Q4_0, and INT8
                    fwht256_inplace_cuda(kcur_ + t * NKV * HD, NKV, cs);
                    fwht256_inplace_cuda(vcur_ + t * NKV * HD, NKV, cs);
                }
                // stored in the state's own format (rotated INT8 K/V are not appended as Q4_0)
                if (st_.kv_q4)
                    kv_append_q4_step(st_.k_q4, st_.v_q4, st_.page_table, step + t * 4, kcur_ + t * NKV * HD,
                                      vcur_ + t * NKV * HD, s, cs, &st_.host);
                else if (st_.kv_int8)
                    kv_append_q8_step(st_.k_q, st_.v_q, st_.k_scale, st_.v_scale, st_.page_table, step + t * 4,
                                      kcur_ + t * NKV * HD, vcur_ + t * NKV * HD, s, cs, &st_.host);
                else
                    kv_append_step(st_.k_pool, st_.v_pool, st_.page_table, step + t * 4, kcur_ + t * NKV * HD,
                                   vcur_ + t * NKV * HD, s, cs, &st_.host);
            }
        }
        if (!full) return true;
        pscope(false);   // the query follows the hc read (or its quantize)
        if (T == 1)   // read once a step: streamed past the layer's other weights in L2
            native_q8_0_mmvq_stream(q8("self_attn.q_proj.weight"), xq_, qfull_, (int) N, (int) (NH * 2 * HD), cs);
        else
            native_mmvq(GGML_Q8_0, q8("self_attn.q_proj.weight"), xq_, qfull_, (int) N, (int) (NH * 2 * HD), T, cs);
        if (!st_.kv_rot && qsa_dec_fused_supported(s)) {
            QsaDecQPost qp;   // the query split, norm and RoPE in one launch
            qp.q_full = qfull_; qp.qcur = qcur_; qp.pos_q = pos; qp.q_norm = f32("self_attn.q_norm.weight"); qp.eps = EPS;
            qsa_dec_q_post(qp, T, s, rope_scaling(), cs);
        } else
        for (int t = 0; t < T; ++t) {
            float* qc = qcur_ + t * NH * HD;
            if (cudaMemcpy2DAsync(qc, (size_t) HD * 4, qfull_ + t * NH * 2 * HD, (size_t) HD * 2 * 4, (size_t) HD * 4,
                                  (size_t) NH, cudaMemcpyDeviceToDevice, cs) != cudaSuccess) {
                err = "mtp: q split failed";
                return false;
            }
            norm_rope(qc, f32("self_attn.q_norm.weight"), (int) NH, (int) HD, pos + t * NH);
            if (st_.kv_rot) fwht256_inplace_cuda(qc, NH, cs);
        }
        const QsaAttnPools pools = qsa_attn_pools(st_);
        if (window_ > 0) window_ids(const_cast<int32_t*>(step), T, (int) window_, ident_, cap_, cs);
        if (kv_par) join(cs);
        pscope(true);   // the attention follows the join
        bool aq_done = false;
        if (!st_.kv_rot && qsa_dec_attn_supported(pools, s)) {   // the gate in the merge writes the q8_1 rows
            aq_done = true;
            qsa_dec_attn(qcur_, pools, ident_, step, cap_, s, attn_scratch_, attn_, T, cs, qfull_, attn32_, (void*) xq_);
        } else {
            qsa_decode_attn_batch(qcur_, pools, ident_, step, cap_, s, attn_scratch_, attn_, T, cs);
            if (st_.kv_rot) fwht256_inplace_cuda(attn_, (int64_t) T * NH, cs);
            for (int t = 0; t < T; ++t)
                native_qsa_gate_apply(attn_ + t * NH * HD, qfull_ + t * NH * 2 * HD, attn32_ + t * NH * HD, (int) NH,
                                      (int) HD, cs);
        }
        if (!aq_done) native_quantize_q8_1(attn32_, xq_, (int) (NH * HD), T, cs);
        native_mmvq(GGML_Q8_0, q8("self_attn.o_proj.weight"), xq_, bo_, (int) (NH * HD), (int) N, T, cs);
        // ---- the MLP hyper-connection (the attention write folded in)
        bool routed = false;   // the read ran the router too
        {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                fa[t].R = R_ + (size_t) t * HC * N; fa[t].R_out = R_ + (size_t) t * HC * N; fa[t].apply = true;
                fa[t].bo_prev = bo_ + t * N; fa[t].inj_prev = inj_ + t * HC;
                fa[t].w_norm = f32("mlp_hyper_connection.hc_norm.weight");
                fa[t].w_down = bf16("mlp_hyper_connection.input_mix_weight_down.weight");
                fa[t].w_up = bf16("mlp_hyper_connection.input_mix_weight_up.weight");
                fa[t].w_inject = bf16("mlp_hyper_connection.block_inject_weight.weight");
                fa[t].eps = EPS; fa[t].lo = lo_ + t * g.hc_lr; fa[t].rs = rs_ + t * HC;
                fa[t].inject_out = inj2_ + t * HC; fa[t].mixed = mixed_ + t * N;
            }
            if (par_ && dec_draft_hcf()) {   // the read with its write folded in, and the router when it is native
                HcFusedArgs h = hcf_args(fa, T, xn_);
                if (g.n_expert == 512 && K == 10) {
                    h.w_router = bf16("mlp.gate.weight"); h.logits = logits_; h.ids = ids_; h.weights = w_;
                    routed = true;
                }
                hc_fused_read(h, cs);
            } else if (T == 1) {   // one row: the write, then the per-row read (its own rounding)
                gr_write(R_, bo_, inj_, gs, R_, cs);
                gr_read(R_, fa[0].w_norm, fa[0].w_down, fa[0].w_up, fa[0].w_inject, EPS, gs, ss.block.gr, mixed_, inj2_, cs);
            } else {
                fused_gr_read_multi(fa, T, xn_, cs);
            }
        }
        pscope_off();
        // ---- MoE: router, the 512 resident experts, the shared expert, the combine, the write
        // par_: the shared expert on the second stream, beside the router and the routed experts
        const bool sh_par = par_;
        if (sh_par) fork(cs);
        for (int t = 0; t < T && !routed; ++t) {
            bf16_gemv_fp32_mmvf(mixed_ + t * N, bf16("mlp.gate.weight"), logits_ + t * g.n_expert, (int) N, (int) g.n_expert, cs);
            native_router_top10(logits_ + t * g.n_expert, ids_ + t * K, w_ + t * K, cs);
        }
        moe_group_resident(ids_, (int) (T * K), (int) K, experts_, (int64_t) basalt::kernels::cpu::BLOB, grp_ptr_,
                           grp_start_, grp_counts_, hit_dst_, hit_slot_, cs);
        quantize_q8_0_scaled(mixed_, hit_xq_, hit_xs_, (int64_t) T * N, cs);
        moe_grouped_s2(grp_ptr_, grp_start_, grp_counts_, hit_dst_, hit_slot_, (int64_t) T * K, (int64_t) T * K, hit_xq_,
                       hit_xs_, hit_scratch_, parts_, cs);
        NativeSharedWeights nsw;
        nsw.gate_type = GGML_Q8_0; nsw.gate_data = q8("mlp.shared_expert.gate_proj.weight");
        nsw.up_type = GGML_Q8_0; nsw.up_data = q8("mlp.shared_expert.up_proj.weight");
        nsw.down_type = GGML_Q8_0; nsw.down_data = q8("mlp.shared_expert.down_proj.weight");
        nsw.q8_1 = xq_;
        const SForm none{};
        const bool fold_write = par_ && dec_draft_hcf();   // the mixer read applies the write
        const cudaStream_t csh = sh_par ? cs2_ : cs;
        for (int t = 0; t < T; ++t) {
            shared_expert(nullptr, nullptr, none, nullptr, nullptr, nullptr, none, nullptr, nullptr, nullptr, none,
                          nullptr, nullptr, nullptr, bf16("mlp.shared_expert_gate.weight"), sh_scratch_, shared_ + t * N,
                          N, g.n_ff, 32, csh, mixed_ + t * N, &nsw);
            if (sh_par) continue;
            native_moe_combine(parts_ + (size_t) t * K * N, w_ + t * K, shared_ + t * N, y_ + t * N, N, K, cs);
            if (!fold_write)
                gr_write(R_ + (size_t) t * HC * N, y_ + t * N, inj2_ + t * HC, gs, R_ + (size_t) t * HC * N, cs);
        }
        if (sh_par) {
            join(cs);
            for (int t = 0; t < T; ++t) {
                native_moe_combine(parts_ + (size_t) t * K * N, w_ + t * K, shared_ + t * N, y_ + t * N, N, K, cs);
                if (!fold_write)
                    gr_write(R_ + (size_t) t * HC * N, y_ + t * N, inj2_ + t * HC, gs, R_ + (size_t) t * HC * N, cs);
            }
        }
        // pre_next_: the next chain step's hidden input from this row's final residual (what the step end hands on),
        // on the second stream beside the mixer read and the head; joined by the caller
        auto fork_next_hidden = [&] {
            if (!(pre_next_ && par_ && T == 1)) return;
            fork(cs);
            record_hidden(R_, 1, xq2_, cs2_);
            fork_open_ = true;
        };
        if (!fold_write) fork_next_hidden();
        // ---- the final mixer and the main model's head
        bool mixer_q8 = false;
        if (par_ && dec_draft_hcf()) {
            pscope(false);   // after the combine
            FusedGrArgs fa[1];
            fa[0].R = R_; fa[0].R_out = R_; fa[0].apply = true; fa[0].bo_prev = y_; fa[0].inj_prev = inj2_;
            fa[0].w_norm = f32("hyper_connection_mixer.hc_norm.weight");
            fa[0].w_down = bf16("hyper_connection_mixer.input_mix_weight_down.weight");
            fa[0].w_up = bf16("hyper_connection_mixer.input_mix_weight_up.weight");
            fa[0].w_inject = nullptr; fa[0].eps = EPS; fa[0].lo = lo_; fa[0].rs = rs_; fa[0].mixed = sample_;
            if (T != 1) { err = "mtp: the one-kernel mixer read runs one row"; return false; }
            HcFusedArgs h = hcf_args(fa, 1, xn_);
            if (hc_fused_q8_rows_ok()) { h.xq = xq_; mixer_q8 = true; }
            h.next_early = pdl_ok;   // the head's blocks start with the read
            hc_fused_read(h, cs);
            pscope_off();
        } else {
            for (int t = 0; t < T; ++t)
                gr_read(R_ + (size_t) t * HC * N, f32("hyper_connection_mixer.hc_norm.weight"),
                        bf16("hyper_connection_mixer.input_mix_weight_down.weight"),
                        bf16("hyper_connection_mixer.input_mix_weight_up.weight"), nullptr, EPS, gs, ss.block.gr,
                        sample_ + t * N, dummy_inj_, cs);
        }
        if (fold_write) fork_next_hidden();
        if (mixer_q8) pscope(false);   // the head follows the read
        if (!mixer_q8) native_quantize_q8_1(sample_, xq_, (int) N, T, cs);
        const bool sub = dvocab_ != nullptr;
        // the argmax drafts over the base rows only on tier 0
        const int64_t nv = sub ? (tier_ == 0 ? n_tier1_ : n_dvocab_) : n_vocab_;
        // The head, its argmax and the argmax's probability in one launch (the probability's sum in
        // another order: last bits)
        if (T == 1 && sub && proxy_ != nullptr && nv == n_tier1_ && diag_ == nullptr &&
            (head_->type() == 13
                 ? basalt::kernels::q5k_proxy_top1(proxy_, dhead_, xq_, (int) nv, n_tier1_, dvocab_, out_ids_, probs_,
                                                   proxy_scratch_, cs)
                 : basalt::kernels::q6k_proxy_top1(proxy_, rows_mapped_ ? head_->weights() : (const void*) dhead_, xq_,
                                                   (int) nv, n_tier1_, dvocab_, out_ids_, probs_, proxy_scratch_, cs,
                                                   rows_mapped_)))
            return true;
        if (T == 1 && head_->type() == 13 && top_scratch_ != nullptr && diag_ == nullptr &&
            q5k_head_top1(sub ? dhead_ : head_->weights(), xq_, head_logits_, (int) N, (int) nv, sub ? dvocab_ : nullptr,
                          out_ids_, probs_, top_scratch_, cs))
            return true;
        if (rows_mapped_)   // the subset's rows of the main head, one row a step: the bits of the copy's rows
            native_q6_k_mmvq_rows(head_->weights(), dvocab_, xq_, head_logits_, (int) N, (int) nv, cs);
        else if (head_->type() == 13 && q5k_head_supported((int) N, T))   // the same bits
            q5k_head_mmvq(sub ? dhead_ : head_->weights(), xq_, head_logits_, (int) N, (int) nv, T, cs);
        else if (head_->type() == GGML_Q8_0 && T == 1)   // read once a step: streamed past the layer's other weights
            native_q8_0_mmvq_stream(sub ? dhead_ : head_->weights(), xq_, head_logits_, (int) N, (int) nv, cs);
        else
            native_mmvq(head_->type(), sub ? dhead_ : head_->weights(), xq_, head_logits_, (int) N, (int) nv, T, cs);
        if (sub && n_xcap_ > 0)   // a context row without a token never wins (the other rows always have one)
            for (int t = 0; t < T; ++t)
                q5k_head_mask_unmapped(head_logits_ + (size_t) t * (size_t) nv + (size_t) xbase_, dvocab_ + xbase_,
                                       (int) n_xcap_, cs);
        if (diag_ != nullptr && T == 1)   // the step's logits to the host, for the draft-tree study
            cudaMemcpyAsync(diag_ + (size_t) chain_j_ * (size_t) diag_stride_, head_logits_, (size_t) nv * sizeof(float),
                            cudaMemcpyDeviceToHost, cs);
        if (sample_greedy_top_prob(head_logits_, T, (int) nv, sub ? dvocab_ : nullptr, out_ids_, probs_, cs))
            return true;
        SamplerParams sp;
        sp.greedy = true;
        sp.temperature = 0.0f;
        sample_tokens(head_logits_, T, (int) nv, nullptr, 0, sp, out_ids_, cs);
        row_top_prob(head_logits_, T, (int) nv, out_ids_, probs_, cs);
        if (sub) map_ids(out_ids_, dvocab_, T, cs);
    } catch (const std::exception& e) {
        err = std::string("mtp: ") + e.what();
        return false;
    }
    return true;
}


}
