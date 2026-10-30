#include "common.hpp"
#include "expert_plan.hpp"
#include "diagnostics.hpp"
#include "basalt/core/load_progress.hpp"
#include "basalt/core/verify.hpp"
#include "basalt/core/diag.hpp"
#include "basalt/core/native_head.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/platform/compiler.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/kernels/fused_gr.hpp"
#include "basalt/kernels/mid_chain.hpp"
#include "basalt/kernels/hc_fused.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/ngram.hpp"
#include "basalt/kernels/qsa.hpp"
#include "basalt/kernels/qsa_decode_attn.hpp"
#include "basalt/kernels/q6k_head.hpp"
#include "basalt/kernels/s2_expert_grouped.hpp"
#include "basalt/core/progress.hpp"
#include "basalt/kernels/verify_kernels.hpp"
#include "basalt/log.hpp"
#include <algorithm>
#include <cmath>
#include <string>

namespace basalt::core {
using namespace verify_detail;

// src/core/verify.cpp - see include/basalt/core/verify.hpp.
Verifier::~Verifier() {
    const Verifier* self = this;
    g_diag_verifier.compare_exchange_strong(self, nullptr);
    for (auto& slot : g_live) {
        Verifier* me = this;
        slot.compare_exchange_strong(me, nullptr);
    }
    if (cs_) cudaStreamSynchronize(cs_);
    if (cm_) cudaStreamSynchronize(cm_);
    if (sh_cs_) cudaStreamSynchronize(sh_cs_);
    if (sh_hi_) cudaStreamSynchronize(sh_hi_);
    if (sh_hi2_) cudaStreamSynchronize(sh_hi2_);
    if (ring_hi_) cudaStreamSynchronize(ring_hi_);
    for (cudaStream_t q : qs_)
        if (q) cudaStreamSynchronize(q);
    dm_free();
    for (auto& e : exec_)
        if (e) cudaGraphExecDestroy(e);
    for (auto& e : exec_nr_)
        if (e) cudaGraphExecDestroy(e);
    if (commit_exec_) cudaGraphExecDestroy(commit_exec_);
    for (auto& kv : solo_sets_) {
        for (auto& e : kv.second.exec)
            if (e) cudaGraphExecDestroy(e);
        for (auto& e : kv.second.exec_nr)
            if (e) cudaGraphExecDestroy(e);
        if (kv.second.commit) cudaGraphExecDestroy(kv.second.commit);
    }
    for (auto& kv : exec_bm_)
        if (kv.second) cudaGraphExecDestroy(kv.second);
    for (auto& kv : commit_bm_)
        if (kv.second) cudaGraphExecDestroy(kv.second);
    if (arena_b_) cudaFree(arena_b_);
    if (logit_bias_device_) cudaFree(logit_bias_device_);
    if (row_bias_device_) cudaFree(row_bias_device_);
    if (h_commitb_) cudaFreeHost(h_commitb_);
    if (h_segb_) cudaFreeHost(h_segb_);
    if (h_qseg_) cudaFreeHost(h_qseg_);
    if (cs_) cudaStreamDestroy(cs_);
    if (cm_) cudaStreamDestroy(cm_);
    if (ev_commit_go_) cudaEventDestroy(ev_commit_go_);
    if (commit_ctl_) cudaFree(commit_ctl_);
    if (lz_s_) cudaStreamDestroy(lz_s_);
    if (lz_fork_) cudaEventDestroy(lz_fork_);
    if (lz_join_) cudaEventDestroy(lz_join_);
    if (sh_cs_) cudaStreamDestroy(sh_cs_);
    if (sh_hi_) cudaStreamDestroy(sh_hi_);
    if (sh_hi2_) cudaStreamDestroy(sh_hi2_);
    if (ev_sh2_) cudaEventDestroy(ev_sh2_);
    if (ring_hi_) cudaStreamDestroy(ring_hi_);
    if (ple_s_) cudaStreamDestroy(ple_s_);
    if (ev_ple_fork_) cudaEventDestroy(ev_ple_fork_);
    if (ev_ple_join_) cudaEventDestroy(ev_ple_join_);
    for (cudaStream_t q : qs_)
        if (q) cudaStreamDestroy(q);
    for (cudaEvent_t e : qev_)
        if (e) cudaEventDestroy(e);
    if (ev_ring_) cudaEventDestroy(ev_ring_);
    if (ev_ring_join_) cudaEventDestroy(ev_ring_join_);
    if (commit_done_) cudaEventDestroy(commit_done_);
    if (ev_fork_) cudaEventDestroy(ev_fork_);
    if (ev_join_) cudaEventDestroy(ev_join_);
    if (arena_) cudaFree(arena_);
    void* hosts[] = {h_tok_, h_step_, h_pos_, h_commit_, h_ple_, h_out_, h_x_, h_ids_, h_w_, h_seq_, h_flag_, h_ymiss_, h_pend_,
                     h_flagA_, h_plan_, h_flagB_, h_q_, h_qp_};
    for (void* h : hosts)
        if (h) cudaFreeHost(h);
}

bool Verifier::init(const WeightTable& wt, const ModelGeometry& g, SessionState& ss, const VerifyHits& hits,
                    const NativeHead* head, int max_t, std::string& err) {
    g_diag_verifier.store(this);
    diag_verify_fn().store(&diag_active_verifier);
    for (auto& slot : g_live) {
        Verifier* none = nullptr;
        if (slot.load() == this || slot.compare_exchange_strong(none, this)) break;
    }
    release_gpu_fn().store(&release_live_verifiers);
    cudaGetDevice(&device_);   // a layer split's stage on another GPU: its streams, graphs and buffers live there
    basalt::kernels::fused_gr_check();   // once per card: which bitwise-equal hyper-connection read runs there
    basalt::kernels::hc_fused_init();
    basalt::kernels::mid_chain_init();
    wt_ = &wt;
    g_ = &g;
    ss_ = &ss;
    hits_ = hits;
    head_ = head;
    max_t_ = max_t;
    sampling_.greedy = true;      // a fresh verifier samples greedily until set_sampling says otherwise
    sampling_.temperature = 0.0f;
    if (max_t < 2 || max_t > basalt::kernels::kVerifyMaxT || max_t > basalt::kernels::cpu::MAXT) {
        err = "verify: the window must hold 2.." + std::to_string(basalt::kernels::kVerifyMaxT) + " tokens";
        return false;
    }
    if (hits.d_res == nullptr || hits.cache_base == nullptr || hits.blob <= 0) {
        err = "verify: needs the profile-filled VRAM expert tier (the model file's profile and --expert-cache); with "
              "--expert-cache auto, no VRAM was left for it - the 'no VRAM is left for the expert cache' line above "
              "says how much is short and what makes room (a smaller --max-context, --kv q4_0, a smaller draft vocabulary, "
              "images on the CPU)";
        return false;
    }
    std::string why;
    if (!layer_verify_compatible(why)) {
        err = "verify: " + why + " (the verify window reproduces the default native decode path)";
        return false;
    }
    if (!basalt::kernels::fused_gr_supported(g.n_embd, g.hc, g.hc_lr) || ss.k != 10 || g.ssm_state_size != 128 ||
        g.ssm_d_conv != 4) {
        err = "verify: geometry differs from the artifact's";
        return false;
    }
    if (le_ < 0) le_ = g.n_layers;
    if (lb_ < 0 || lb_ >= le_ || le_ > g.n_layers || (lb_ > 0 && hand_in_ == nullptr) ||
        (le_ < g.n_layers && hand_out_ == nullptr)) {
        err = "verify: the stage's layer range or its hand-off buffers are wrong";
        return false;
    }
    const WeightRef* wo = wt.find("output.weight");
    if (wo == nullptr) { err = "verify: output.weight is missing"; return false; }
    n_vocab_ = wo->ne1;

    const basalt::kernels::QsaShapes s = shapes_of(g);
    cap_ = basalt::kernels::qsa_selection_width(basalt::kernels::kTopkMaxCells, s);
    max_blocks_ = ss.qsa_states[ss.qsa_primary()].max_cells / s.idx_block + 2;
    attn_scratch_floats_ = (int64_t) basalt::kernels::qsa_decode_attn_scratch_floats(cap_, s);

    const uint64_t T = (uint64_t) max_t, N = (uint64_t) g.n_embd, HC = (uint64_t) g.hc, K = (uint64_t) ss.k;
    const uint64_t C = (uint64_t) g.ssm_conv_channels, ZV = (uint64_t) g.ssm_value_dim, HV = (uint64_t) g.ssm_v_heads;
    const uint64_t NH = (uint64_t) g.n_head, HD = (uint64_t) g.head_dim, NKV = (uint64_t) g.n_head_kv;
    const uint64_t IQ = (uint64_t) g.idx_q_heads, ID = (uint64_t) g.idx_key_dim;
    const uint64_t nG = (uint64_t) g.n_gdn_layers(), nQ = (uint64_t) g.n_qsa_layers();
    const uint64_t HS = (uint64_t) basalt::kernels::NG_HIST * basalt::kernels::NG_HC_DIM;
    const uint64_t TS = (uint64_t) (s.idx_block - 1) * ID;
    const int max_in = (int) std::max<uint64_t>(std::max<uint64_t>(N, ZV), NH * HD);

    // ---- mapped staging
    bool ok = mapped(T * 4, (void**) &h_tok_, (void**) &m_tok_) &&
              mapped(T * basalt::kernels::kStepCount * 4, (void**) &h_step_, (void**) &m_step_) &&
              mapped(T * (NH + NKV + IQ) * 4, (void**) &h_pos_, (void**) &m_pos_) &&
              mapped((2 + T) * 4 + 16, (void**) &h_commit_, (void**) &m_commit_) &&
              mapped(64, (void**) &h_pend_, (void**) &m_pend_) &&
              mapped(T * N * 4, (void**) &h_ple_, (void**) &m_ple_) &&
              mapped(T * 4 + 16, (void**) &h_out_, (void**) &m_out_) &&
              mapped(T * 4 + 16, (void**) &h_q_, (void**) &m_q_) &&
              mapped(T * 4 + 16, (void**) &h_qp_, (void**) &m_qp_) &&
              mapped(T * N * 4, (void**) &h_x_, (void**) &m_x_) &&
              mapped(T * K * 4, (void**) &h_ids_, (void**) &m_ids_) &&
              mapped(T * K * 4, (void**) &h_w_, (void**) &m_w_) &&
              mapped(64, (void**) &h_seq_, (void**) &m_seq_) &&
              mapped(64, (void**) &h_flag_, (void**) &m_flag_) &&
              mapped(64, (void**) &h_flagA_, (void**) &m_flagA_) &&
              mapped(64, (void**) &h_flagB_, (void**) &m_flagB_) &&
              mapped(64, (void**) &h_plan_err_, (void**) &m_plan_err_) &&
              mapped(T * K * N * 4, (void**) &h_ymiss_, (void**) &m_ymiss_);
    if (!ok) { err = "verify: mapped staging allocation failed"; return false; }
    // the GPU plan: counts(4) | start(cap+1) | dst(cap) | tok(cap) | pad | ptr(cap u64) | ptr2(cap u64) | start2(cap+1)
    {
        const int64_t cap = (int64_t) (T * K);
        const int64_t i32 = 4 + (cap + 1) + cap + cap;
        const int64_t ptr_off = (i32 + 1) & ~1ll;
        plan_i32_ = ptr_off + 4 * cap + (cap + 1) + 1;
        if (!mapped((size_t) plan_i32_ * 4 + 64, (void**) &h_plan_, (void**) &m_plan_)) {
            err = "verify: mapped plan allocation failed";
            return false;
        }
        sink_.counts = h_plan_;
        sink_.start = h_plan_ + 4;
        sink_.dst = sink_.start + cap + 1;
        sink_.tok = sink_.dst + cap;
        sink_.ptr = (unsigned long long*) (h_plan_ + ptr_off);
        sink_.ptr2 = sink_.ptr + cap;
        sink_.start2 = h_plan_ + ptr_off + 4 * cap;
        sink_.cap = cap;
        sink_.publish = &Verifier::publish_plan;
        sink_.staged = &Verifier::raise_flag_b;
        sink_.ctx = this;
    }

    // ---- the device arena: the same sequence counted, then carved
    auto carve = [&](Bump& b) {
        tok_ = b.take<int32_t>(T); step_ = b.take<int32_t>(T * basalt::kernels::kStepCount);
        pos_ = b.take<int32_t>(T * (NH + NKV + IQ)); commit_ = b.take<int32_t>(2 + T);
        ple_ = b.take<float>(T * N); emb_ = b.take<float>(T * N); R_ = b.take<float>(T * HC * N);
        mixed_ = b.take<float>(T * N); bo_ = b.take<float>(T * N);
        inj_ = b.take<float>(T * HC); inj2_ = b.take<float>(T * HC);
        lo_ = b.take<float>(T * (uint64_t) g.hc_lr); rs_ = b.take<float>(T * HC); xn_ = b.take<float>(T * HC * N);
        xq_ = b.take<uint8_t>(basalt::kernels::native_q8_1_bytes(max_in, (int) T));
        sh_xq_ = b.take<uint8_t>(basalt::kernels::native_q8_1_bytes(max_in, (int) T));
        qkv_L_ = b.take<float>(nG * T * C); h_L_ = b.take<float>(nG * T * C);
        gate_L_ = b.take<float>(nG * T * HV); beta_L_ = b.take<float>(nG * T * HV);
        z_ = b.take<float>(T * ZV); y_ = b.take<float>(T * ZV); y_dummy_ = b.take<float>(T * ZV);
        qfull_ = b.take<float>(T * NH * 2 * HD); qcur_ = b.take<float>(T * NH * HD);
        kcur_ = b.take<float>(T * NKV * HD); vcur_ = b.take<float>(T * NKV * HD);
        idx_raw_L_ = b.take<float>(nQ * T * ID); qidx_ = b.take<float>(T * IQ * ID);
        scores_ = b.take<float>(T * (uint64_t) max_blocks_); sel_ = b.take<int32_t>(T * (uint64_t) cap_);
        attn_ = b.take<float>(T * NH * HD); attn32_ = b.take<float>(T * NH * HD);
        attn_scratch_ = b.take<float>(T * (uint64_t) attn_scratch_floats_);
        tail_snap_ = b.take<float>(nQ * TS);
        logits_ = b.take<float>(T * (uint64_t) g.n_expert); w_ = b.take<float>(T * K); ids_ = b.take<int32_t>(T * K);
        shared_ = b.take<float>(T * N); parts_ = b.take<float>(T * K * N); hit_out_ = b.take<float>(T * K * N);
        hit_slot_ = b.take<int32_t>(T * K); hit_dst_ = b.take<int32_t>(T * K); hit_count_ = b.take<int32_t>(4);
        plan_ = b.take<int32_t>((uint64_t) plan_i32_ + 16);
        staging_ = b.take<uint8_t>((uint64_t) kStagingBlobs * basalt::kernels::cpu::expert_layout().max_blob);
        hit_xq_ = b.take<uint8_t>(T * (N / 32) * 34); hit_xs_ = b.take<float>(T * (N / 32));
        nat_xq_ = b.take<uint8_t>(T * (N / 32) * 36);
        hit_scratch_ = b.take<uint8_t>(std::max<uint64_t>(
            basalt::kernels::moe_hit_grouped_scratch_bytes((int64_t) (T * K), g.n_embd, g.n_ff),
            basalt::kernels::native_expert_scratch_bytes((int64_t) (T * K), g.n_ff)));
        head_mixed_ = b.take<float>(T * N); head_inj_ = b.take<float>(HC);
        sh_gate_ = b.take<float>(T * (uint64_t) g.n_ff);
        sh_up_ = b.take<float>(T * (uint64_t) g.n_ff); sh_g_ = b.take<float>(T + 4);
        head_logits_ = b.take<float>(T * (uint64_t) n_vocab_);
        if (head_ != nullptr && head_->vproxy() != nullptr)
            vhead_scratch_ = b.take<uint8_t>(basalt::kernels::q6k_vhead_scratch_bytes());
        hist_snap_ = b.take<float>(T * HS);
        ple_win_ = b.take<float>(2 * T * HC * N);
        ple_key_ = b.take<float>(T * HC * N); ple_val_ = b.take<float>(T * N);
    };
    Bump count;
    carve(count);
    if (cudaMalloc(&arena_, count.used) != cudaSuccess) {
        err = "verify: the device arena (" + std::to_string(count.used >> 20) + " MiB) does not fit";
        return false;
    }
    cudaMemset(arena_, 0, count.used);
    prof_on_ = diagnostics().profile;
    if (prof_on_) decode_gaps().host_on = true;
    if (next_ == nullptr && exp_flag("host_ev", 0) != 0) decode_events().enable(4000);
    if (prof_on_) {
        const size_t np = (size_t) g.n_layers * kProfPer + 8;
        if (cudaMalloc((void**) &prof_, np * 8) != cudaSuccess) { prof_on_ = false; prof_ = nullptr; cudaGetLastError(); }
        else { cudaMemset(prof_, 0, np * 8); prof_h_.assign(np, 0); }
    }
    Bump real;
    real.base = (uint8_t*) arena_;
    carve(real);
    sink_.staging_cap = kStagingBlobs;
    (void) TS;
    int cs_least = 0, cs_greatest = 0;
    cudaDeviceGetStreamPriorityRange(&cs_least, &cs_greatest);
    const int cs_prio = main_high_ ? cs_greatest : 0;
    if (cudaStreamCreateWithPriority(&cs_, cudaStreamNonBlocking, cs_prio) != cudaSuccess ||
        cudaStreamCreateWithFlags(&sh_cs_, cudaStreamNonBlocking) != cudaSuccess) {
        err = "verify: stream create failed";
        return false;
    }
    {
        int least = 0, greatest = 0;
        if (cudaDeviceGetStreamPriorityRange(&least, &greatest) != cudaSuccess ||
            cudaStreamCreateWithPriority(&sh_hi_, cudaStreamNonBlocking, greatest) != cudaSuccess ||
            cudaStreamCreateWithPriority(&sh_hi2_, cudaStreamNonBlocking, greatest) != cudaSuccess ||
            cudaEventCreateWithFlags(&ev_sh2_, cudaEventDisableTiming) != cudaSuccess ||
            cudaStreamCreateWithPriority(&ring_hi_, cudaStreamNonBlocking, greatest) != cudaSuccess ||
            cudaEventCreateWithFlags(&ev_ring_, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&ev_ring_join_, cudaEventDisableTiming) != cudaSuccess ||
            cudaStreamCreateWithFlags(&ple_s_, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags(&ev_ple_fork_, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&ev_ple_join_, cudaEventDisableTiming) != cudaSuccess) {
            err = "verify: side stream create failed";
            return false;
        }
        basalt::log::info("basalt verify", "side branches at stream priority %d", greatest);
    }
    if (cudaStreamCreateWithPriority(&cm_, cudaStreamNonBlocking, cs_prio) != cudaSuccess ||
        cudaEventCreateWithFlags(&ev_commit_go_, cudaEventDisableTiming) != cudaSuccess ||
        cudaMalloc((void**) &commit_ctl_, sizeof(basalt::kernels::CommitCtl)) != cudaSuccess ||
        cudaMemset(commit_ctl_, 0, sizeof(basalt::kernels::CommitCtl)) != cudaSuccess) {
        err = "verify: commit stream create failed";
        return false;
    }
    {
        int sms = 0;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device_ >= 0 ? device_ : 0);
        fin_blocks_ = sms;
    }
    if (!basalt::kernels::gdn_commit_lazy_none(commit_ctl_) ||
        cudaStreamCreateWithPriority(&lz_s_, cudaStreamNonBlocking, cs_prio) != cudaSuccess ||
        cudaEventCreateWithFlags(&lz_fork_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&lz_join_, cudaEventDisableTiming) != cudaSuccess) {
        err = "verify: lazy commit stream create failed";
        return false;
    }
    *(volatile uint32_t*) h_pend_ = 0;
    if (cudaEventCreateWithFlags(&commit_done_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&ev_fork_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&ev_join_, cudaEventDisableTiming) != cudaSuccess) {
        err = "verify: event create failed";
        return false;
    }
    {
        int least = 0, greatest = 0;
        bool qok = cudaDeviceGetStreamPriorityRange(&least, &greatest) == cudaSuccess;
        for (auto& q : qs_) qok = qok && cudaStreamCreateWithPriority(&q, cudaStreamNonBlocking, greatest) == cudaSuccess;
        for (auto& e : qev_) qok = qok && cudaEventCreateWithFlags(&e, cudaEventDisableTiming) == cudaSuccess;
        if (!qok) { err = "verify: QSA branch stream create failed"; return false; }
    }
    // Check whether 100% of experts across [lb_, le_) are resident in this stage's VRAM cache.
    // When true, every layer plans on device and writes directly into parts_ without any CPU doorbells,
    // wait_flag_ge spins, PCIe empty launches, or moe_hit_add copies.
    all_resident_ = false;
    ar_off_ = false;
    if (h_plan_err_ != nullptr) *(volatile uint32_t*) h_plan_err_ = 0;
    if (hits.h_res != nullptr && hits.d_res != nullptr && hits.cache_base != nullptr) {
        {
            bool all_ok = true;
            for (int64_t l = lb_; l < le_ && all_ok; ++l) {
                for (int64_t e = 0; e < g.n_expert; ++e) {
                    if (hits.h_res[l * g.n_expert + e] < 0) {
                        all_ok = false;
                        break;
                    }
                }
            }
            all_resident_ = all_ok;
        }
    }
    device_plan_ = false;
    if (all_resident_ || device_plan_) {
        bool ok2 = true;
        if (device_plan_)
            ok2 = cudaMalloc((void**) &skip_, 64) == cudaSuccess && cudaMemset(skip_, 0, 64) == cudaSuccess;
        if (ok2 && hits.slot_off != nullptr && hits.n_slots > 0) {
            ok2 = cudaMalloc((void**) &slot_off_d_, (size_t) hits.n_slots * sizeof(unsigned long long)) == cudaSuccess &&
                  cudaMemcpy(slot_off_d_, hits.slot_off, (size_t) hits.n_slots * sizeof(unsigned long long),
                             cudaMemcpyHostToDevice) == cudaSuccess;
        }
        if (!ok2) { cudaGetLastError(); all_resident_ = false; device_plan_ = false; }
    }
    if (!dm_init(err)) return false;
    basalt::log::info("basalt verify", "window up to %d tokens, %.1f MiB of device buffers%s", max_t,
                 (double) count.used / 1048576.0,
                 all_resident_ ? " (100% VRAM resident: zero-doorbell graph)" : "");
    return true;
}

const float* Verifier::final_R(int t) const { return R_ + (size_t) t * (size_t) (g_->hc * g_->n_embd); }

}  // namespace basalt::core
