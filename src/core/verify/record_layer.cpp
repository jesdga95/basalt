#include "common.hpp"
#include "record.hpp"
#include "expert_plan.hpp"
#include "basalt/core/load_progress.hpp"
#include "basalt/core/verify.hpp"
#include "basalt/core/native_head.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/bf16_gemv.hpp"
#include "basalt/kernels/native_router.hpp"
#include "basalt/kernels/elementwise.hpp"
#include "basalt/kernels/fused_gr.hpp"
#include "basalt/kernels/mid_chain.hpp"
#include "basalt/kernels/hc_fused.hpp"
#include "basalt/kernels/cvec.hpp"
#include "basalt/kernels/dec_gdn.hpp"
#include "basalt/kernels/dense_tc.hpp"
#include "basalt/kernels/gr.hpp"
#include "basalt/kernels/native_ple_postops.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/ple.hpp"
#include "basalt/kernels/qsa_dec_fused.hpp"
#include "basalt/kernels/quantize_act.hpp"
#include "basalt/kernels/shared_expert.hpp"
#include "basalt/kernels/verify_kernels.hpp"
#include "basalt/kernels/dec_moe.hpp"
#include <algorithm>
#include <exception>
#include <string>

namespace basalt::core {
using namespace verify_detail;

// pre(l, group): up to the ring
bool Verifier::WindowRecorder::pre(int64_t l) {
    using namespace basalt::kernels;
    const int n = T;
    stamp(l, 0);
    const LayerView v(wt, l);
    const char* pfx[2] = {"hc_attn_", "hc_ffn_"};
    const WeightRef *wn[2], *wd[2], *wu[2], *wi[2];
    for (int h = 0; h < 2; ++h) {
        wn[h] = need(v, (std::string(pfx[h]) + "norm.weight").c_str(), err);
        wd[h] = need(v, (std::string(pfx[h]) + "down.weight").c_str(), err);
        wu[h] = need(v, (std::string(pfx[h]) + "up.weight").c_str(), err);
        wi[h] = need(v, (std::string(pfx[h]) + "inject.weight").c_str(), err);
        if (!wn[h] || !wd[h] || !wu[h] || !wi[h]) return false;
    }
    // the previous layer's FFN write, folded into this layer's first read (a control vector after it has
    // already applied it)
    bool pending = l > 0 && !cvec().covers(l - 1);
    if (l == 1 && ple_on) {
        if (!ple_side) {
            if (v_.ar_on() || dm) wait_flag_ge(v_.m_flag_, 1, cs);
            copy_from_mapped(v_.ple_, v_.m_ple_, (int64_t) T * N, cs);
        }
        float* normalized = (float*) ((uint8_t*) ss.ple.scratch + ple_block_scratch_bytes());
        const bool ple_batch_kv = n > 1 && ss.ple.w.key_bf16 != nullptr &&
                                  ss.ple.w.value_bf16 != nullptr;
        // the rows' write, post-ops, history advance and snapshots in two launches, the value
        // projection on the side stream beside the key's
        // (one row too: there the multi-row GEMVs are ple_block's own)
        const bool ple_win = ple_batch_kv || (n == 1 && ss.ple.w.key_bf16 != nullptr && ss.ple.w.value_bf16 != nullptr);
        if (v_.batch_rec_) v_.batch_seg_ple_ = ple_win;
        cudaStream_t pside = v_.sh_hi_ != nullptr ? v_.sh_hi_ : v_.sh_cs_;
        const bool ple_fork = ple_win && pside != nullptr && v_.ev_fork_ != nullptr && v_.ev_join_ != nullptr;
        if (ple_side) {
            cudaStreamWaitEvent(cs, v_.ev_ple_join_, 0);
        } else if (ple_batch_kv || ple_win) {
            try {
                if (ple_fork) {
                    cudaEventRecord(v_.ev_fork_, cs);
                    cudaStreamWaitEvent(pside, v_.ev_fork_, 0);
                    bf16_gemv_fp32_mmvf_multi(v_.ple_, N, ss.ple.w.value_bf16,
                                              v_.z_, N, N, N, n, pside);
                    cudaEventRecord(v_.ev_join_, pside);
                }
                bf16_gemv_fp32_mmvf_multi(v_.ple_, N, ss.ple.w.key_bf16,
                                          v_.xn_, HC * N, N, HC * N, n, cs);
                if (ple_fork) cudaStreamWaitEvent(cs, v_.ev_join_, 0);
                else bf16_gemv_fp32_mmvf_multi(v_.ple_, N, ss.ple.w.value_bf16,
                                               v_.z_, N, N, N, n, cs);
            } catch (const std::exception& e) {
                err = std::string("verify PLE multi: ") + e.what();
                return false;
            }
        }
        if (ple_win) {
            PleWindowArgs pa;
            pa.R = Rt(0); pa.block_out = v_.bo_; pa.inject = v_.inj2_;
            pa.key = ple_side ? v_.ple_key_ : v_.xn_;
            pa.value = ple_side ? v_.ple_val_ : v_.z_;
            pa.history = v_.batch_rec_ ? v_.ple_work_ : ss.ple.hist; pa.segs = v_.batch_rec_ ? v_.segb_ : nullptr;
            pa.hist_snap = v_.hist_snap_; pa.scratch = v_.ple_win_; pa.T = n;
            try {
                native_ple_window(pa, ss.ple.w, cs);
            } catch (const std::exception& e) {
                err = std::string("verify PLE window: ") + e.what();
                return false;
            }
        }
        for (int t = 0; t < T && !ple_win; ++t) {
            gr_write(Rt(t), v_.bo_ + t * N, v_.inj2_ + t * HC, gs, Rt(t), cs);
            PleOut po;
            po.normalized = normalized;
            po.result = Rt(t);
            float* hist = v_.batch_rec_ ? v_.ple_work_ + (size_t) t * HS : ss.ple.hist;   // the row's own history
            try {
                if (ple_batch_kv) {
                    ple_block_projected(v_.xn_ + (size_t) t * HC * N, v_.z_ + (size_t) t * N, Rt(t), hist,
                                        ss.ple.w, po, ss.ple.scratch, cs);
                } else {
                    ple_block(v_.ple_ + t * N, Rt(t), hist, ss.ple.w, po, ss.ple.scratch, cs);
                }
                ple_history_advance(hist, normalized, cs);
            } catch (const std::exception& e) {
                err = std::string("verify PLE: ") + e.what();
                return false;
            }
            copy_from_mapped(v_.hist_snap_ + (size_t) t * HS, hist, HS, cs);
        }
        pending = false;
    }
    auto gr_read_group = [&](int half, bool apply, float* inj_prev, float* inj_out) {
        FusedGrArgs fa[kFusedGrMaxT];
        for (int t = 0; t < T; ++t) {
            FusedGrArgs& a = fa[t];
            a.R = Rt(t); a.R_out = Rt(t); a.apply = apply;
            a.bo_prev = v_.bo_ + t * N; a.inj_prev = inj_prev + t * HC;
            a.w_norm = (const float*) wn[half]->data; a.w_down = (const uint16_t*) wd[half]->data;
            a.w_up = (const uint16_t*) wu[half]->data; a.w_inject = (const uint16_t*) wi[half]->data;
            a.eps = EPS; a.lo = v_.lo_ + t * g.hc_lr; a.rs = v_.rs_ + t * HC;
            a.inject_out = inj_out + t * HC; a.mixed = v_.mixed_ + t * N;
        }
        fused_gr_read_multi(fa, n, v_.xn_, cs, v_.prof_on_ ? v_.prof_ : nullptr,
                            (int) (l * kProfPer + (half == 0 ? 27 : 30)));
    };
    auto hc_args = [&](int half, bool apply, float* inj_prev, float* inj_out) {
        HcFusedArgs h;
        h.T = n; h.R = Rt(0); h.R_out = Rt(0); h.apply = apply;
        h.bo_prev = v_.bo_; h.inj_prev = inj_prev;
        h.w_norm = (const float*) wn[half]->data; h.w_down = (const uint16_t*) wd[half]->data;
        h.w_up = (const uint16_t*) wu[half]->data; h.w_inject = (const uint16_t*) wi[half]->data;
        h.eps = EPS; h.lo = v_.lo_; h.rs = v_.rs_; h.inject_out = inj_out;
        h.mixed = v_.mixed_; h.xn = v_.xn_;
        return h;
    };
    const bool read_q8 = dec_hc && hc_fused_q8_rows_ok();
    const WeightRef* w_router = v.get("ffn_gate_inp.weight");
    const bool router_multi = w_router != nullptr && NE == 512 && K == 10;
    // the GDN mixer split by key head: the layer's q8_1 rows and a doorbell for the peer's key heads; in a batch
    // window CUDA0 runs the whole layer and the doorbell only passes
    const bool gs_ring = dm && v_.gs_nk0_ > 0 && !is_qsa_layer(g, l);
    const bool gs_split = gs_ring && !v_.batch_rec_;
    // a GDN layer's middle as one persistent kernel (mid_chain.hpp) between the layer's two hc reads: the qkv + z pair,
    // the recurrence and the output projection, bitwise the kernels it replaces
    bool mid_ok = false;
    MidChainArgs ma;
    if (dec_hc && read_q8 && router_multi && !is_qsa_layer(g, l) && !v_.prof_on_ && !v_.batch_rec_ && !gs_split &&
        n <= kMidChainMaxT && N == 2560 && dec_gdnf_mode() != 0 && (dense_tc_mask() & 4)) {
        const WeightRef *wqkv = v.get("attn_qkv.weight"), *wg = v.get("attn_gate.weight"),
                        *wout = v.get("ssm_out.weight"), *wa = v.get("ssm_alpha.weight"),
                        *wb = v.get("ssm_beta.weight"), *wc = v.get("ssm_conv1d.weight"),
                        *wnm = v.get("ssm_norm.weight"), *wdt = v.get("ssm_dt.bias"), *wsa = v.get("ssm_a");
        if (wqkv && wg && wout && wa && wb && wc && wnm && wdt && wsa && wqkv->native_data && wg->native_data &&
            wout->native_data && dense_tc_pair_supported(wqkv->native_type, wg->native_type, (int) N) &&
            dense_tc_supported(wout->native_type, (int) ZV) &&
            dense_tc_type_on(wout->native_type)) {
            const int64_t gi = gdn_idx[(size_t) l];
            ma.T = n;
            ma.xq = v_.xq_;
            ma.wq = wqkv->native_data; ma.wz = wg->native_data; ma.wo = wout->native_data;
            ma.type_q = wqkv->native_type; ma.type_z = wg->native_type; ma.type_o = wout->native_type;
            DecRecArgs& ra = ma.rec;
            ra.state = ss.gdn_state + (size_t) (gi - ss.gdn_ord0) * gdn_floats;
            ra.qkv = v_.qkv_L_ + (size_t) gi * MT * C; ra.conv_w = (const float*) wc->data; ra.x = v_.mixed_;
            ra.w_alpha = (const uint16_t*) wa->data; ra.w_beta = (const uint16_t*) wb->data;
            ra.dt = (const float*) wdt->data; ra.ssm_a = (const float*) wsa->data; ra.z = v_.z_;
            ra.gamma = (const float*) wnm->data; ra.h = v_.h_L_ + (size_t) gi * MT * C;
            ra.gate = v_.gate_L_ + (size_t) gi * MT * HV; ra.beta = v_.beta_L_ + (size_t) gi * MT * HV; ra.yq = v_.xq_;
            ra.C = (int) C; ra.n_embd = (int) N; ra.h_k = (int) HK; ra.h_v = (int) HV; ra.eps = EPS;
            ma.bo = v_.bo_;
            mid_ok = mid_chain_supported(ma);
        }
    }
    {
        // the read's first kernel follows a combine, copy or wait: a plain launch, except after the device-planned
        // tail's combine kernel (PDL2)
        const bool after_kernel = pdl && dm && dec_hc && l > v_.lb_ && !(l == 1 && ple_on);
        PdlScope ps(pdl, !after_kernel);
        if (dec_hc) {
            HcFusedArgs h = hc_args(0, pending, v_.inj2_, v_.inj_);
            if (read_q8) h.xq = v_.xq_;
            // the GDN qkv+z pair kernel follows (dense_tc_pair_mask & 16): its blocks start on the SMs the read leaves
            h.next_early = mid_ok || (read_q8 && (kernels::dense_tc_pair_mask() & 16) &&
                           !is_qsa_layer(g, l) && N == 2560 && n <= kDecMaxT &&
                           dec_gdnf_mode() != 0);
            // a QSA layer's query projection follows (qsaf_wide path): its blocks start with the read
            if (read_q8 && is_qsa_layer(g, l) && N == 2560 && n <= kDecMaxT) {
                const QsaState& st0 = ss.qsa_states[qsa_idx[(size_t) l]];
                h.next_early = st0.kv_int8 && !st0.kv_rot && !st0.kv_hybrid && !st0.kv_q4 &&
                               qsa_dec_fused_supported(s) &&
                               v_.qs_[0] != nullptr;
            }
            h.pf[0] = wn[1]->data; h.pf_bytes[0] = (size_t) HC * N * sizeof(float);
            hc_fused_read(h, cs);
        } else {
            gr_read_group(0, pending, v_.inj2_, v_.inj_);
        }
    }
    stamp(l, 1);
    if (gs_ring) {
        if (!read_q8) native_quantize_q8_1(v_.mixed_, v_.xq_, (int) N, n, cs);
        const int64_t r = l - v_.lb_;
        dec_moe_mix_publish(v_.xq_, v_.mixed_, n, (int) N, v_.gs_m_xq_ + (size_t) r * v_.dm_->xq_stride,
                            v_.gs_m_x_ + (size_t) r * v_.dm_->x_stride, v_.dm_->m_w + 48, (uint32_t) (r + 1), gs_split, cs);
    }
    if (lazy_layer(l)) {   // the layer's commit, forked in the previous post
        cudaStreamWaitEvent(cs, v_.lz_join_, 0);
        lz_open = false;
    }
    if (mid_ok) {
        // the first read's dependent: its first pair tiles are already issued by the time the read ends (PDL)
        PdlScope ps(pdl, mid_ord == 0);
        ma.ord = mid_ord++;
        mid_chain(ma, cs);
    }
    float* xm = v_.mixed_;
    try {
        if (mid_ok) {
            // the middle ran in the mid chain
        } else if (!is_qsa_layer(g, l)) {
            if (!gdn(l, v, read_q8, gs_split)) return false;
        } else {
            if (!qsa(l, v, read_q8)) return false;
        }
    } catch (const std::exception& e) {
        err = "verify layer " + std::to_string(l) + ": " + e.what();
        return false;
    }
    if (gs_split)   // the peer's key heads added after CUDA0's
        dec_moe_mix_combine(v_.dm_->m_w + 52, (uint32_t) (l - v_.lb_ + 1), v_.gs_m_part_, v_.bo_, n * (int) N, cs);
    else if (gs_ring)
        dec_moe_mix_wait(v_.dm_->m_w + 52, (uint32_t) (l - v_.lb_ + 1), cs);
    stamp(l, 16);
    bool routed = false;   // the fused read ran the router too
    const bool sh_fork = !v_.prof_on_ && v_.sh_cs_ != nullptr && v_.ev_fork_ != nullptr &&
                         v_.ev_join_ != nullptr;
    cudaStream_t sh_stream = sh_fork ? (v_.sh_hi_ != nullptr ? v_.sh_hi_ : v_.sh_cs_) : cs;
    // the ring on its own stream, forked after the router, joined where the side branch is
    const bool pk_ring = dm && sh_fork && sh_stream == v_.sh_hi_ && v_.ring_hi_ != nullptr;
    ring_fork = pk_ring;
    // the read writes the q8_1 rows, post's gate/up launch plans CUDA0's share
    glue_on = false;
    if (pk_ring && v_.dm_->ptab0 != nullptr && dec_hc && router_multi && hc_fused_q8_rows_ok()) {
        const auto& f = basalt::kernels::cpu::expert_layout().fmt[(size_t) l];
        glue_on = v_.dm_->vram < 0 && dec_moe_glue_ok(native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff));
        if (!glue_on) note_fallback("glue off for the layer's experts", f.gu_type, f.d_type);
    }
    // the read's router items also dot the shared expert's gate input and write its q8_1 rows: its side branch is
    // its three projections (gate and up side by side), and post's combine applies the gate
    const WeightRef* w_gi = v.get("ffn_gate_inp_shexp.weight");
    sh_read = glue_on && w_gi != nullptr && w_gi->bytes == (uint64_t) N * 2 && ((uintptr_t) w_gi->data & 15u) == 0;
    {
        PdlScope ps(pdl);   // after the out projection's kernel
        if (dec_hc) {
            HcFusedArgs h = hc_args(1, true, v_.inj_, v_.inj2_);
            if (router_multi) {
                h.w_router = (const uint16_t*) w_router->data;
                h.logits = v_.logits_; h.ids = v_.ids_; h.weights = v_.w_;
                if (glue_on) {   // the ring's block starts with the read and follows its counts
                    h.xq = v_.nat_xq_;
                    h.pub_ctr = v_.dm_->pub + 4 * (l - v_.lb_);
                    h.next_early = true;
                }
                if (sh_read) {
                    h.w_gate_sh = (const uint16_t*) w_gi->data;
                    h.gate_sh = v_.sh_g_;
                    h.xq_sh = v_.sh_xq_;
                }
                routed = true;
            }
            const WeightRef* next_norm = nullptr;
            if (l + 1 < g.n_layers) next_norm = LayerView(wt, l + 1).get("hc_attn_norm.weight");
            else next_norm = wt.find("output_hc_norm.weight");
            if (next_norm != nullptr) { h.pf[0] = next_norm->data; h.pf_bytes[0] = (size_t) HC * N * sizeof(float); }
            hc_fused_read(h, cs);
        } else {
            gr_read_group(1, true, v_.inj_, v_.inj2_);
        }
    }
    if (sh_fork) {
        cudaEventRecord(v_.ev_fork_, cs);
        cudaStreamWaitEvent(sh_stream, v_.ev_fork_, 0);
        if (sh_read) cudaStreamWaitEvent(v_.sh_hi2_, v_.ev_fork_, 0);
    }
    // the window's rows routed in 2 launches (one router GEMV reading the weight once, one top-10) instead of 2
    // per token; every row's arithmetic is the single-token call's
    if (routed) {
    } else if (router_multi) {
        try {
            PdlScope ps(pdl);   // after the hc read's last kernel (the fork adds no node on this stream)
            bf16_gemv_fp32_mmvf_multi(v_.mixed_, N, (const uint16_t*) w_router->data, v_.logits_, NE, N,
                                      NE, n, cs);
            native_router_top10_multi(v_.logits_, v_.ids_, v_.w_, n, cs);
        } catch (const std::exception& e) { err = "verify router: " + std::string(e.what()); return false; }
    } else
    for (int t = 0; t < T; ++t) {
        MoEBuffers mb = ss.moe;
        mb.logits = v_.logits_ + t * NE; mb.ids = v_.ids_ + t * K; mb.weights = v_.w_ + t * K;
        if (!moe_route(wt, g, l, K, mb, v_.mixed_ + t * N, cs, err, nullptr)) return false;
    }
    if (dm) {   // the tiers, CUDA0's plan, the q8_1 rows and the rings in one launch
        DmRouteArgs ra;
        ra.ids = v_.ids_; ra.x = xm; ra.n_tok = n; ra.k = (int) K; ra.n_embd = (int) N;
        ra.n_expert = (int) g.n_expert; ra.layer = (int) l;
        ra.res0 = v_.dm_->res0 + l * g.n_expert;
        ra.res1 = v_.dm_->pdev >= 0 ? v_.dm_->res1 + l * g.n_expert : nullptr;
        ra.cache_base = v_.hits_.cache_base; ra.slot_off = v_.dm_->slot_off; ra.blob = (long long) v_.hits_.blob;
        ra.plan = v_.dm_->plan; ra.xq = v_.nat_xq_;
        const int64_t r = l - v_.lb_;
        ra.ring_m = v_.dm_->m_ring + r; ra.x_m = v_.dm_->m_x + (size_t) r * v_.dm_->x_stride;
        ra.xq_m = v_.dm_->m_xq + (size_t) r * v_.dm_->xq_stride;
        ra.host_seq_m = v_.m_seq_; ra.peer_seq_m = v_.dm_->m_w;
        ra.n_x = (int) v_.dm_->xp.size();
        for (int q = 0; q < ra.n_x; ++q) {
            const DecMoe::XPeer& x = v_.dm_->xp[(size_t) q];
            ra.xres[q] = x.res + l * g.n_expert;
            ra.xring_m[q] = x.m_ring + r;
            ra.xseq_m[q] = x.m_w;
        }
        ra.ring = (uint32_t) (r + 1); ra.ctr = v_.dm_->ctr;
        ra.stats = v_.dm_->wstat;
        ra.peer_cap = 0;
        ra.rows = v_.dm_->rq != nullptr || v_.dm_->warm != nullptr;
        if (v_.dm_->stage_src != nullptr && v_.batch_rec_) {   // batch windows: many rows, many misses
            ra.stage_src = v_.dm_->stage_src + l * g.n_expert;
            ra.stage_k = v_.dm_->stage_k;
        }
        PdlScope pr(false);   // after the router's kernel: a plain launch
        if (glue_on) {   // the ring's work inside post's gate/up launch, released by the read's counts
            glue_ring = ra;
            glue_ring.mode = 2; glue_ring.kinds = true;
            glue_ring.pub_ctr = v_.dm_->pub + 4 * r; glue_ring.pub_target = hc_fused_pub_arrivals();
            ring_fork = false;
        } else if (pk_ring) {   // the ring beside, CUDA0's plan and rows here
            cudaEventRecord(v_.ev_ring_, cs);
            cudaStreamWaitEvent(v_.ring_hi_, v_.ev_ring_, 0);
            DmRouteArgs rp = ra;
            rp.mode = 2; rp.xq_pub = v_.dm_->xq_pub; rp.kinds = glue_on;
            {
                PdlScope pp(false);
                dec_moe_route(rp, v_.ring_hi_);
            }
            cudaEventRecord(v_.ev_ring_join_, v_.ring_hi_);
            ra.mode = 1;
        }
        if (!glue_on) dec_moe_route(ra, cs);
        if (ra.stage_src != nullptr) {   // the staged share's copy, beside the ring and CUDA0's own experts
            cudaEventRecord(v_.dm_->st_fork, cs);
            cudaStreamWaitEvent(v_.dm_->st_s, v_.dm_->st_fork, 0);
            PdlScope pst(false);
            dec_moe_stage(ra, v_.staging_, v_.dm_->stage_stride,
                          (int64_t) basalt::kernels::cpu::expert_layout().blob_bytes(l), v_.dm_->stage_blocks, v_.dm_->st_s);
            cudaEventRecord(v_.dm_->st_join, v_.dm_->st_s);
        }
    } else if (v_.ar_on()) {
        resident_plan(v_.ids_, n * (int) K, (int) K, v_.hits_.d_res + l * g.n_expert, (int) g.n_expert,
                      v_.hits_.cache_base, v_.slot_off_d_, (long long) v_.hits_.blob,
                      v_.plan_, (long long) v_.max_t_ * K, nullptr, 0, cs,
                      v_.m_plan_err_);
    } else {
        if (v_.device_plan_)   // every routed expert resident: this group's plan without the host
            resident_plan(v_.ids_, n * (int) K, (int) K, v_.hits_.d_res + l * g.n_expert, (int) g.n_expert,
                          v_.hits_.cache_base, v_.slot_off_d_, (long long) v_.hits_.blob,
                          v_.plan_, (long long) v_.max_t_ * K, v_.skip_,
                          (uint32_t) (l - v_.lb_ + 1), cs);
        if (v_.remote_opt_)   // the helper GPUs reduce with the routing weights - publish them
            doorbell_publish(xm, v_.ids_, v_.w_, (int64_t) n * N, (int64_t) n * K, v_.m_x_,
                             v_.m_ids_, v_.m_w_, v_.m_seq_, cs);
        else {
            const int32_t* layer_res = v_.hits_.d_res != nullptr ? (v_.hits_.d_res + l * g.n_expert) : nullptr;
            doorbell_publish_res(xm, v_.ids_, layer_res, (int) g.n_expert, (int64_t) n * N, (int64_t) n * K,
                                 v_.m_x_, v_.m_ids_, v_.m_seq_, cs);
        }
    }
    stamp(l, 17);
    {
        const WeightRef *wgi = need(v, "ffn_gate_inp_shexp.weight", err), *wsg = need(v, "ffn_gate_shexp.weight", err),
                        *wsu = need(v, "ffn_up_shexp.weight", err), *wsd = need(v, "ffn_down_shexp.weight", err);
        if (!wgi || !wsg || !wsu || !wsd) return false;
        if (!native_of(wsg, v.name("ffn_gate_shexp.weight"), err) || !native_of(wsu, v.name("ffn_up_shexp.weight"), err) ||
            !native_of(wsd, v.name("ffn_down_shexp.weight"), err))
            return false;
        NativeSharedWeights nsw;
        nsw.gate_type = wsg->native_type; nsw.gate_data = wsg->native_data;
        nsw.up_type = wsu->native_type; nsw.up_data = wsu->native_data;
        nsw.down_type = wsd->native_type; nsw.down_data = wsd->native_data;
        nsw.q8_1 = sh_fork ? v_.sh_xq_ : v_.xq_;
        {
        try {
            PdlScope psh(false);   // the side chain's kernels follow each other
            if (sh_read)
                shared_expert_multi_rows(n, nsw, v_.sh_gate_, v_.sh_up_, v_.shared_, N, g.n_ff, sh_stream, v_.sh_hi2_,
                                         v_.ev_sh2_);
            else
                shared_expert_multi(n, xm, nsw, (const uint16_t*) wgi->data, v_.sh_gate_,
                                    v_.sh_up_, v_.sh_g_, v_.shared_, N, g.n_ff, sh_stream);
        } catch (const std::exception& e) {
            err = std::string("verify shared expert: ") + e.what();
            return false;
        }
        if (sh_fork) cudaEventRecord(v_.ev_join_, sh_stream);
        }
    }
    if (dm) {}   // the route kernel wrote nat_xq_
    else if (basalt::kernels::cpu::expert_layout().native)
        quantize_q8_1_rows(xm, n, N, v_.nat_xq_, cs);
    else
        quantize_q8_0_scaled(xm, v_.hit_xq_, v_.hit_xs_, (int64_t) n * N, cs);
    stamp(l, 18);
    return true;
}

}  // namespace basalt::core
