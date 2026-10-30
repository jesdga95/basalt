#include "common.hpp"
#include "record.hpp"
#include "expert_plan.hpp"
#include "basalt/core/verify.hpp"
#include "basalt/core/remote_expert_opt.hpp"
#include "basalt/core/native_head.hpp"
#include "basalt/platform/compiler.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/native_moe.hpp"
#include "basalt/kernels/elementwise.hpp"
#include "basalt/kernels/cvec.hpp"
#include "basalt/kernels/gr.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/s2_expert_grouped.hpp"
#include "basalt/kernels/verify_kernels.hpp"
#include "basalt/kernels/dec_moe.hpp"
#include <exception>
#include <string>

namespace basalt::core {
using namespace verify_detail;

bool Verifier::WindowRecorder::post(int64_t l) {
    using namespace basalt::kernels;
    const int n = T;
    const uint32_t ring = (uint32_t) (l - v_.lb_ + 1);
    const int64_t cap = (int64_t) n * K, capx = (int64_t) v_.max_t_ * K;
    int32_t* pl = v_.plan_;
    const int32_t* p_counts = pl;
    const int32_t* p_start = pl + 4;
    const int32_t* p_dst = p_start + capx + 1;
    const int32_t* p_tok = p_dst + capx;
    const int64_t ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    const unsigned long long* p_ptr = (const unsigned long long*) (pl + ptr_off);
    const unsigned long long* p_ptr2 = p_ptr + capx;
    const int32_t* p_start2 = pl + ptr_off + 4 * capx;
    float* hit_out = v_.hit_out_;
    float* parts_out = v_.parts_;
    const auto& lay = basalt::kernels::cpu::expert_layout();
    // the VRAM groups now; the PCIe groups once the copy engine has landed them in staging.
    // `gy`: the native launch's groups side by side (0: cap, one block row per possible group).
    auto grouped = [&](const unsigned long long* gp, const int32_t* gs, const int32_t* gn, int64_t gy, float* dst_buf) {
        if (lay.native) {
            // the layer's GGUF formats (i-quant gate/up, Q2_0 / IQ4_NL down)
            const auto& f = lay.fmt[(size_t) l];
            const NativeExpertLayout L = native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
            native_expert_grouped(L, gp, gs, gn, p_dst, p_tok, cap, cap,
                                  v_.nat_xq_, v_.hit_scratch_, dst_buf, cs, gy);
        } else {
            moe_grouped_s2(gp, gs, gn, p_dst, p_tok, cap, cap, v_.hit_xq_,
                           v_.hit_xs_, v_.hit_scratch_, dst_buf, cs);
        }
    };
    if (dm) {   // CUDA0's entries straight into their parts rows; the other tiers are read by the combine
        stamp(l, 19);
        const auto& f = lay.fmt[(size_t) l];
        const NativeExpertLayout L = native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
        DmPlan* P = v_.dm_->plan;
        // the next GDN layer's commit (and later ones while this layer waits for the CPU), and the next layer's
        // projection weights into L2 while this layer's combine waits
        const bool lz_next = lazy && l + 1 < v_.le_ && (lazy_layer(l + 1) || is_qsa_layer(g, l + 1));
        if (lz_next) {
            GdnLazyPf pf;
            {
                const LayerView nv(wt, l + 1);
                const char* names_gdn[] = {"attn_qkv.weight", "attn_gate.weight", "ssm_out.weight"};
                const char* names_qsa[] = {"attn_q.weight", "attn_k.weight", "attn_v.weight", "indexer.q_proj.weight",
                                           "indexer.k_proj.weight", "attn_output.weight"};
                const bool qsa = is_qsa_layer(g, l + 1);
                const int nr = qsa ? 6 : 3;
                int r = 0;
                auto add = [&](const void* p, uint64_t b) {
                    if (p != nullptr && (b & ~15ull) != 0 && r < kLazyPfRanges) { pf.p[r] = p; pf.n[r++] = b & ~15ull; }
                };
                for (int i = 0; i < nr; ++i) {
                    const WeightRef* w = nv.get(qsa ? names_qsa[i] : names_gdn[i]);
                    if (w == nullptr) continue;
                    if (w->native_data != nullptr)
                        add(w->native_data, (uint64_t) iq_row_bytes(w->native_type, w->ne0) * (uint64_t) w->ne1);
                    else
                        add(w->data, w->bytes);
                }
                pf.ids = v_.ids_; pf.n_ids = n * (int) K; pf.n_expert = (int) g.n_expert; pf.ptab = v_.dm_->ptab0 + l * g.n_expert;
            }
            cudaEventRecord(v_.lz_fork_, cs);
            cudaStreamWaitEvent(v_.lz_s_, v_.lz_fork_, 0);
            gdn_commit_lazy(lazy_args, v_.commit_ctl_,
                            lazy_layer(l + 1) ? (int) (gdn_idx[(size_t) (l + 1)] - gdn_first) : 0,
                            v_.dm_->tags ? v_.dm_->cseen : nullptr, ring, v_.lz_s_, v_.fin_blocks_, &pf);
            cudaEventRecord(v_.lz_join_, v_.lz_s_);
            lz_open = true;
        }
        if (glue_on) {
            DmGlueArgs ga;
            ga.ids = v_.ids_; ga.ptab = v_.dm_->ptab0 + l * g.n_expert;
            ga.n_tok = n; ga.k = (int) K; ga.n_expert = (int) g.n_expert;
            ga.plan = P; ga.xq = v_.nat_xq_;
            { ga.ring_on = true; ga.ring = glue_ring; }
            if (v_.dm_->landed_pin != nullptr) {
                ga.landed = v_.dm_->landed + (l - v_.lb_);
                ga.landed_need = v_.dm_->landed + v_.dm_->rings;
                ga.landed_stop = v_.dm_->m_w + 60;
            }
            dec_moe_glue(L, ga, v_.hit_scratch_, parts_out, (int) cap, cs);
        } else if (v_.dm_->vram >= 0) {
            v_.landed_glue_ = false;
            dec_moe_vram(L, P, v_.nat_xq_, v_.dm_->vscr, parts_out, v_.dm_->vsync, cs, v_.dm_->vram,
                         (int) cap);
        } else {
            v_.landed_glue_ = false;
            native_expert_grouped(L, P->ptr, P->start, P->counts, P->dst, P->tok, cap, cap,
                                  v_.nat_xq_, v_.hit_scratch_, parts_out, cs);
        }
        if (v_.dm_->stage_src != nullptr && v_.batch_rec_) {   // the staged share, once its copy is in staging
            cudaStreamWaitEvent(cs, v_.dm_->st_join, 0);
            PdlScope pst(false);
            native_expert_grouped(L, P->sptr, P->sstart, P->scounts, P->sdst, P->stok, cap, cap, v_.nat_xq_,
                                  v_.hit_scratch_, parts_out, cs, kPcieGroupRows);
        }
        stamp(l, 20);
        stamp(l, 21);
        stamp(l, 22);
    } else if (v_.ar_on()) {
        stamp(l, 19);
        grouped(p_ptr, p_start, p_counts, 0, parts_out);
        stamp(l, 20);
    } else {
        if (v_.device_plan_) {   // skipped when the device planned this group (all its experts resident)
            wait_flag_ge_or(v_.m_flagA_, ring, v_.skip_, cs);
            copy_i32_from_mapped_unless(pl, v_.m_plan_, v_.plan_i32_, v_.skip_, ring, cs);
        } else {
            wait_flag_ge(v_.m_flagA_, ring, cs);                  // the pool published this group's GPU plan
            copy_i32_from_mapped(pl, v_.m_plan_, v_.plan_i32_, cs);
        }
        stamp(l, 19);
        grouped(p_ptr, p_start, p_counts, 0, hit_out);
        stamp(l, 20);
        if (v_.device_plan_) wait_flag_ge_or(v_.m_flagB_, ring, v_.skip_, cs);
        else wait_flag_ge(v_.m_flagB_, ring, cs);                 // the PCIe groups are planned
        {   // stage them with a copy kernel, then point at staging
            const int64_t per = kStagingBlobs;
            uint8_t* stage = v_.staging_;
            fetch_blobs(p_ptr2, p_counts + 2, stage, (int64_t) lay.blob_bytes(l), (int) per, cs);
            rebase_ptrs((unsigned long long*) p_ptr2, p_counts + 2, stage, (int64_t) lay.blob_bytes(l), cs);
        }
        stamp(l, 21);
        // the PCIe share is pcie_frac of the misses: a few groups when the cache is cold, usually none (none when the
        // share is 0), so its launch is kPcieGroupRows block rows striding over the groups, not cap of them
        grouped(p_ptr2, p_start2, p_counts + 2, kPcieGroupRows, hit_out);
        stamp(l, 22);
        if (v_.device_plan_) {   // no CPU share when the device planned the group: its rows are zeros
            wait_flag_ge_or(v_.m_flag_, ring, v_.skip_, cs);
            copy_or_zero_from_mapped(parts_out, v_.m_ymiss_, (long long) n * K * N,
                                     v_.skip_, ring, cs);
        } else {
            wait_flag_ge(v_.m_flag_, ring, cs);               // the CPU's share is in the mapped rows
            stamp(l, 23);
            if (v_.remote_opt_)   // the helper's rows come back reduced; skip them as well
                v_.remote_opt_->copy_rows(parts_out, v_.m_ymiss_, 0, n, p_dst, p_counts + 1, cs);
            else   // only the CPU rows cross PCIe (p_dst[0, counts[1]) = the GPU's own rows)
                copy_rows_from_mapped(parts_out, v_.m_ymiss_, (int64_t) n * K, N,
                                      p_dst, p_counts + 1, cs);
        }
        moe_hit_add(parts_out, hit_out, p_dst, p_counts + 1, cap, N, cs);
    }
    if (!v_.prof_on_ && v_.sh_cs_ != nullptr && v_.ev_fork_ != nullptr && v_.ev_join_ != nullptr) {
        cudaStreamWaitEvent(cs, v_.ev_join_, 0);
        if (ring_fork) cudaStreamWaitEvent(cs, v_.ev_ring_join_, 0);   // the ring's launch
    }
    if (dm) {
        // the combine polls the peers' rows themselves (with further peers always: the wait counts one done word);
        // one peer's done word also ends the poll, but with further peers only a release does (word 60).  Without
        // --wait-stats it also waits for the CPU pool's word itself, with no wait launch before it.
        const bool dm_tag = v_.dm_->tags || (v_.dm_->tag && (v_.dm_->wstat == nullptr || !v_.dm_->xp.empty()));
        const volatile uint32_t* tag_done = !dm_tag ? nullptr : v_.dm_->xp.empty() ? v_.dm_->m_w + 16 : v_.dm_->m_w + 60;
        if (!v_.dm_->tags) {
            PdlScope pw(false);
            dec_moe_wait(v_.dm_->plan, v_.dm_->m_w + 16, v_.m_flag_, ring, cs, v_.dm_->wstat, (int) l, true, dm_tag);
        }   // true: the combine is the wait's programmatic dependent
        stamp(l, 23);
        try {
            dec_moe_combine(v_.parts_, v_.dm_->m_rows, v_.m_ymiss_, v_.dm_->plan, v_.w_, v_.shared_, v_.bo_, N, K, n, nullptr, nullptr,
                            ring, cs, !v_.dm_->tags, tag_done, v_.dm_->tags ? v_.m_flag_ : nullptr, v_.dm_->cseen,
                            sh_read ? v_.sh_g_ : nullptr);
        } catch (const std::exception& e) { err = "verify combine: " + std::string(e.what()); return false; }
    } else if (n > 1) {   // one launch for the window's rows
        try {
            native_moe_combine_multi(v_.parts_, v_.w_, v_.shared_, v_.bo_, N, K, n, cs);
        } catch (const std::exception& e) { err = "verify combine: " + std::string(e.what()); return false; }
    } else
    for (int t = 0; t < T; ++t) {
        MoEBuffers mb = ss.moe;
        mb.weights = v_.w_ + t * K; mb.shared = v_.shared_ + t * N;
        if (!moe_combine_parts(g, l, K, mb, v_.parts_ + (size_t) t * K * N, v_.bo_ + t * N, cs, err)) return false;
    }
    if (v_.remote_opt_) v_.remote_opt_->combine(v_.bo_, v_.y_dummy_, 0, n,
                          v_.device_plan_ ? v_.skip_ : nullptr, ring, cs);
    stamp(l, 24);
    if (l == g.n_layers - 1) {
        if (!fuse_head_gr) {
            for (int t = 0; t < T; ++t) gr_write(Rt(t), v_.bo_ + t * N, v_.inj2_ + t * HC, gs, Rt(t), cs);
            if (cvec().covers(l)) cvec_apply(Rt(0), l, n, HC * N, nullptr, 0, nullptr, 0, false, cs);
        }
    } else if (cvec().covers(l)) {
        cvec_apply(Rt(0), l, n, HC * N, v_.bo_, N, v_.inj2_, HC, true, cs);
    }
    return true;
}

}  // namespace basalt::core
