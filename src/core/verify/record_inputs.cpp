#include "common.hpp"
#include "record.hpp"
#include "expert_plan.hpp"
#include "basalt/core/verify.hpp"
#include "basalt/core/native_head.hpp"
#include "basalt/kernels/bf16_gemv.hpp"
#include "basalt/kernels/elementwise.hpp"
#include "basalt/kernels/verify_kernels.hpp"
#include "basalt/log.hpp"
#include <algorithm>
#include <exception>
#include <string>

namespace basalt::core {
using namespace verify_detail;

bool Verifier::WindowRecorder::inputs() {
    using namespace basalt::kernels;
    if (v_.prof_on_ && !v_.batch_rec_) gpu_stamp(v_.prof_, (int) (g.n_layers * kProfPer + 2), cs);
    // layer 1's PLE projections read only the host-gathered rows, so they run beside layer 0 from the window's start;
    // layer 1 joins them before the PLE window
    ple_side = ple_on && !v_.batch_rec_ && dm && v_.lb_ <= 1 && v_.le_ > 1 && !v_.prof_on_ &&
                          v_.ple_s_ != nullptr && v_.ev_ple_fork_ != nullptr && v_.ev_ple_join_ != nullptr &&
                          ss.ple.w.key_bf16 != nullptr && ss.ple.w.value_bf16 != nullptr;
    if (!v_.batch_rec_) v_.ple_side_ = ple_side;
    if (ple_side) {
        try {
            cudaEventRecord(v_.ev_ple_fork_, cs);
            cudaStreamWaitEvent(v_.ple_s_, v_.ev_ple_fork_, 0);
            wait_flag_ge(v_.m_flag_ + 8, 1, v_.ple_s_);
            copy_from_mapped(v_.ple_, v_.m_ple_, (int64_t) T * N, v_.ple_s_);
            bf16_gemv_fp32_mmvf_multi(v_.ple_, N, ss.ple.w.key_bf16, v_.ple_key_, HC * N, N, HC * N, T, v_.ple_s_);
            bf16_gemv_fp32_mmvf_multi(v_.ple_, N, ss.ple.w.value_bf16, v_.ple_val_, N, N, N, T, v_.ple_s_);
            cudaEventRecord(v_.ev_ple_join_, v_.ple_s_);
        } catch (const std::exception& e) {
            err = std::string("verify PLE side: ") + e.what();
            return false;
        }
    }
    // ---- the async commit's remainder (a no-op when none is pending): nothing below reads or writes what it works on until
    // this node returns - the GDN state, conv history, indexer buffers and PLE history, nor the rows it reads (h_L_,
    // qkv_L_, gate_L_, beta_L_, idx_raw_L_, tail_snap_, hist_snap_), which the first kernels' launches order after it
    if (v_.finisher_on() && !v_.batch_rec_ && cs == v_.cs_) {
        const GdnCommitLayers fa = v_.commit_gdn_args();
        if (fa.n_layers > 0) gdn_commit_finish(fa, v_.commit_ctl_, cs, v_.fin_blocks_);
    }
    // ---- the window's inputs, from mapped staging
    copy_i32x3_from_mapped(v_.tok_, v_.m_tok_, T, v_.step_, v_.m_step_, (int64_t) T * kStepCount, v_.pos_, v_.m_pos_,
                           (int64_t) MT * (NH + NKV + IQ), cs);
    if (v_.batch_rec_) {   // the segment tables: which slot each row is
        copy_i32_from_mapped((int32_t*) v_.segb_, (const int32_t*) v_.m_segb_, (int64_t) T * 8, cs);
        copy_i32_from_mapped((int32_t*) v_.d_qrows_, (const int32_t*) v_.m_qseg_, (int64_t) (kQsaSegMaxRows + 8) * 4, cs);
        if (ple_on) ple_hist_gather(v_.ple_work_, v_.d_ple_slots_, v_.segb_, HS, T, cs);
    }
    // per-ROW positions of the K rows [t][NKV] and the indexer query rows [t][IQ] (for batched RoPE)
    pos_k = v_.pos_ + MT * NH;
    pos_i = v_.pos_ + MT * (NH + NKV);

    // ---- the embeddings, broadcast to the hc streams - or, in a later stage of a layer split, the previous stage's
    // residual, pending write and inject (see set_stage)
    if (v_.lb_ > 0) {
        if (!v_.batch_rec_) wait_flag_ge(v_.m_flag_ + 12, 1, cs);   // the previous stage's graph raised it after its hand-off
        const float* hin = v_.hand_in_ + (size_t) hrow0 * HB;   // the group's rows: [R][bo][inj] contiguous
        copy_from_mapped(v_.R_, hin, (int64_t) T * HC * N, cs);
        copy_from_mapped(v_.bo_, hin + (size_t) T * HC * N, (int64_t) T * N, cs);
        copy_from_mapped(v_.inj2_, hin + (size_t) T * (HC + 1) * N, (int64_t) T * HC, cs);
    } else if (const NativeEmbed* ne = native_embed()) {   // the GGUF-form table
        ne->gather_dev(v_.tok_, T, v_.emb_, cs);
        broadcast_streams(v_.emb_, v_.R_, N, (int) HC, T, cs);
    } else {
        const WeightRef* w = wt.find("token_embd.weight");
        if (w == nullptr || w->codebook_iq4nl || (w->code_bits != 2 && w->code_bits != 4 && w->code_bits != 8)) {
            err = "verify: token_embd.weight is missing or not an S2/S4/S8 tensor";
            return false;
        }
        const auto* codes = (const uint8_t*) w->data;
        const auto* scales = (const float*) (codes + w->codes_bytes);
        const auto* offsets = w->has_offset ? (const float*) (codes + w->codes_bytes + w->scales_bytes) : nullptr;
        const uint64_t row_codes = (uint64_t) (w->ne0 / (8 / w->code_bits));
        const uint64_t row_groups = (uint64_t) (w->ne0 / w->group_elems);
        embedding_gather_dev(codes, scales, offsets, v_.tok_, T, w->ne0, w->code_bits, w->code_bias, w->group_elems,
                             row_codes, row_groups, v_.emb_, cs);
        broadcast_streams(v_.emb_, v_.R_, N, (int) HC, T, cs);
    }

    // per-layer state indices (GDN and QSA layers are numbered separately)
    gdn_idx.assign((size_t) g.n_layers, -1);
    qsa_idx.assign((size_t) g.n_layers, -1);
    {
        int64_t qi = 0, gi = 0;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            if (is_qsa_layer(g, l)) qsa_idx[(size_t) l] = qi++;
            else gdn_idx[(size_t) l] = gi++;
        }
    }
    // a lazy GDN layer: one after the stage's first, in a solo window (lazy_on)
    gdn_first = -1;
    for (int64_t l = v_.lb_; l < v_.le_ && gdn_first < 0; ++l) gdn_first = gdn_idx[(size_t) l];
    lazy = v_.lazy_on() && !v_.batch_rec_ && cs == v_.cs_ && dm;
    lazy_args = lazy ? v_.commit_gdn_lazy_args() : GdnCommitLayers{};
    lz_open = false;   // a lazy launch not joined yet
    return true;
}

}  // namespace basalt::core
