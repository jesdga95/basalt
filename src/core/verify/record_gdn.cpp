#include "common.hpp"
#include "record.hpp"
#include "expert_plan.hpp"
#include "basalt/core/verify.hpp"
#include "basalt/kernels/dec_gdn.hpp"
#include "basalt/kernels/dense_tc.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/verify_kernels.hpp"

namespace basalt::core {
using namespace verify_detail;

bool Verifier::WindowRecorder::gdn(int64_t l, const LayerView& v, bool read_q8, bool gs_split) {
    using namespace basalt::kernels;
    const int n = T;
    float* xm = v_.mixed_;
        // ======================= GDN =======================
        const WeightRef *wqkv = need(v, "attn_qkv.weight", err), *wg = need(v, "attn_gate.weight", err),
                        *wout = need(v, "ssm_out.weight", err), *wa = need(v, "ssm_alpha.weight", err),
                        *wb = need(v, "ssm_beta.weight", err), *wc = need(v, "ssm_conv1d.weight", err),
                        *wnm = need(v, "ssm_norm.weight", err), *wdt = need(v, "ssm_dt.bias", err),
                        *wsa = need(v, "ssm_a", err);
        if (!wqkv || !wg || !wout || !wa || !wb || !wc || !wnm || !wdt || !wsa) return false;
        if (!native_of(wqkv, v.name("attn_qkv.weight"), err) || !native_of(wg, v.name("attn_gate.weight"), err) ||
            !native_of(wout, v.name("ssm_out.weight"), err))
            return false;
        const int64_t gi = gdn_idx[(size_t) l];
        float* state = ss.gdn_state + (size_t) (gi - ss.gdn_ord0) * gdn_floats;
        float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
        float* qkv = v_.qkv_L_ + (size_t) gi * MT * C;
        float* hb = v_.h_L_ + (size_t) gi * MT * C;
        float* gate = v_.gate_L_ + (size_t) gi * MT * HV;
        float* beta = v_.beta_L_ + (size_t) gi * MT * HV;
        PdlScope ps(pdl);   // the layer's GDN kernels, after the hc read
        if (gs_split) {   // CUDA0's key heads in place (dec_gdn_rec's head map); the peer's arrive in the combine
            const GsLayer& GL = v_.gs_layers_[(size_t) gi];
            const int C0 = (int) (C / HK * v_.gs_nk0_), Z0 = (int) (ZV / HK * v_.gs_nk0_);
            kernels::dense_tc_pair(GL.tq, GL.wqkv0, v_.gs_qkv0_, C0, GL.tz, GL.wz0, v_.gs_z0_, Z0, v_.xq_, (int) N, n, cs);
            DecRecArgs ra;
            ra.state = state; ra.qkv = v_.gs_qkv0_; ra.conv_w = (const float*) wc->data; ra.x = xm;
            ra.w_alpha = (const uint16_t*) wa->data; ra.w_beta = (const uint16_t*) wb->data;
            ra.dt = (const float*) wdt->data; ra.ssm_a = (const float*) wsa->data; ra.z = v_.gs_z0_;
            ra.gamma = (const float*) wnm->data; ra.h = hb; ra.gate = gate; ra.beta = beta; ra.yq = v_.xq_;
            ra.C = (int) C; ra.n_embd = (int) N; ra.h_k = (int) HK; ra.h_v = (int) HV; ra.eps = EPS;
            ra.k0 = 0; ra.nk = v_.gs_nk0_; ra.qkv_out = qkv;
            dec_gdn_rec(ra, n, cs);
            dense_proj(4, GL.to, GL.wo0, v_.xq_, v_.bo_, Z0, (int) N, n, cs);
            goto gdn_done;
        }
        // conv, gates, recurrence, gated norm and the output's q8_1 in dec_gdn_rec's one or two launches; at
        // 2/3 the z projection runs on the side stream beside the recurrence (dec_gdn.hpp)
        {
        // a batch window may take the fused recurrence where the qkv and z projections are one pair launch
        const bool pair_ok = kernels::dense_tc_pair_supported(wqkv->native_type, wg->native_type, (int) N) &&
                             ((uintptr_t) wqkv->native_data & 15) == 0 && ((uintptr_t) wg->native_data & 15) == 0;
        const int gdnf = ((!v_.batch_rec_ || pair_ok) && N == 2560 && n <= kDecMaxT) ? dec_gdnf_mode() : 0;
        auto rec = [&](const DecRecArgs& ra) {
            if (v_.batch_rec_) dec_gdn_rec_seg(ra, v_.segb_, (size_t) gi * gdn_floats, n, n, cs);
            else dec_gdn_rec(ra, n, cs);
        };
        const bool gdnf_fork = gdnf >= 2 && v_.sh_cs_ != nullptr && v_.ev_fork_ != nullptr && v_.ev_join_ != nullptr;
        auto gdnf_z_side = [&] {
            cudaEventRecord(v_.ev_fork_, cs);
            cudaStreamWaitEvent(v_.sh_cs_, v_.ev_fork_, 0);
            PdlScope pz(pdl, true);
            dense_proj(2, wg->native_type, wg->native_data, v_.xq_, v_.z_, (int) N, (int) ZV, n, v_.sh_cs_);
            cudaEventRecord(v_.ev_join_, v_.sh_cs_);
        };
        if (!read_q8) native_quantize_q8_1(xm, v_.xq_, (int) N, n, cs);
        // qkv and z in one dense_tc_pair launch for every window width, then the
        // one-launch recurrence with the gated norm
        if (gdnf && !pair_ok)
            note_fallback("GDN qkv + z without dense_tc_pair", wqkv->native_type,
                          wg->native_type);
        if (gdnf && pair_ok) {
            {
                PdlScope pp(pdl);   // after the hc read or the q8_1 kernel
                kernels::dense_tc_pair(wqkv->native_type, wqkv->native_data, qkv, (int) C,
                                       wg->native_type, wg->native_data, v_.z_, (int) ZV, v_.xq_, (int) N, n, cs);
            }
            for (int i = 2; i <= 5; ++i) stamp(l, i);
            DecRecArgs ra;
            ra.state = state; ra.qkv = qkv; ra.conv_w = (const float*) wc->data; ra.x = xm;
            ra.w_alpha = (const uint16_t*) wa->data; ra.w_beta = (const uint16_t*) wb->data;
            ra.dt = (const float*) wdt->data; ra.ssm_a = (const float*) wsa->data; ra.z = v_.z_;
            ra.gamma = (const float*) wnm->data; ra.h = hb; ra.gate = gate; ra.beta = beta; ra.yq = v_.xq_;
            ra.C = (int) C; ra.n_embd = (int) N; ra.h_k = (int) HK; ra.h_v = (int) HV; ra.eps = EPS;
            ra.early_trigger = pdl;   // PDL2: the out projection stages its weights
            {
                PdlScope pr(pdl);
                rec(ra);
            }
            stamp(l, 6);
            dense_proj(4, wout->native_type, wout->native_data, v_.xq_, v_.bo_, (int) ZV, (int) N, n, cs);
            goto gdn_done;
        }
        const bool pk_z = gdnf_fork && v_.sh_hi_ != nullptr && n <= 2;
        if (pk_z) {   // z at the greatest priority beside qkv, then the one-launch recurrence
            cudaEventRecord(v_.ev_fork_, cs);
            cudaStreamWaitEvent(v_.sh_hi_, v_.ev_fork_, 0);
            {
                PdlScope pz(pdl, true);
                dense_proj(2, wg->native_type, wg->native_data, v_.xq_, v_.z_, (int) N, (int) ZV, n, v_.sh_hi_);
            }
            cudaEventRecord(v_.ev_join_, v_.sh_hi_);
            dense_proj(1, wqkv->native_type, wqkv->native_data, v_.xq_, qkv, (int) N, (int) C, n, cs);
            stamp(l, 2);
            DecRecArgs ra;
            ra.state = state; ra.qkv = qkv; ra.conv_w = (const float*) wc->data; ra.x = xm;
            ra.w_alpha = (const uint16_t*) wa->data; ra.w_beta = (const uint16_t*) wb->data;
            ra.dt = (const float*) wdt->data; ra.ssm_a = (const float*) wsa->data; ra.z = v_.z_;
            ra.gamma = (const float*) wnm->data; ra.h = hb; ra.gate = gate; ra.beta = beta; ra.yq = v_.xq_;
            ra.C = (int) C; ra.n_embd = (int) N; ra.h_k = (int) HK; ra.h_v = (int) HV; ra.eps = EPS;
            cudaStreamWaitEvent(cs, v_.ev_join_, 0);
            for (int i = 3; i <= 5; ++i) stamp(l, i);
            {
                PdlScope pr(pdl, true);   // after the join
                dec_gdn_rec(ra, n, cs);
            }
            stamp(l, 6);
            dense_proj(4, wout->native_type, wout->native_data, v_.xq_, v_.bo_, (int) ZV, (int) N, n, cs);
            goto gdn_done;
        }
        if (gdnf_fork && gdnf == 3) gdnf_z_side();
        dense_proj(1, wqkv->native_type, wqkv->native_data, v_.xq_, qkv, (int) N, (int) C, n, cs);
        stamp(l, 2);
        if (gdnf) {
            DecRecArgs ra;
            ra.state = state; ra.qkv = qkv; ra.conv_w = (const float*) wc->data; ra.x = xm;
            ra.w_alpha = (const uint16_t*) wa->data; ra.w_beta = (const uint16_t*) wb->data;
            ra.dt = (const float*) wdt->data; ra.ssm_a = (const float*) wsa->data; ra.z = v_.z_;
            ra.gamma = (const float*) wnm->data; ra.h = hb; ra.gate = gate; ra.beta = beta; ra.yq = v_.xq_;
            ra.C = (int) C; ra.n_embd = (int) N; ra.h_k = (int) HK; ra.h_v = (int) HV; ra.eps = EPS;
            ra.early_trigger = pdl;   // PDL2: the out projection stages its weights
            if (gdnf_fork) {
                if (gdnf == 2) gdnf_z_side();
                ra.o = v_.y_;
                dec_gdn_rec_split(ra, n, cs);
                for (int i = 3; i <= 5; ++i) stamp(l, i);
                cudaStreamWaitEvent(cs, v_.ev_join_, 0);
                PdlScope pn(pdl, true);   // the norm follows the join
                dec_gdn_rec_norm(ra, n, cs);
                stamp(l, 6);
                dense_proj(4, wout->native_type, wout->native_data, v_.xq_, v_.bo_, (int) ZV, (int) N, n, cs);
            } else {
                dense_proj(2, wg->native_type, wg->native_data, v_.xq_, v_.z_, (int) N, (int) ZV, n, cs);
                for (int i = 3; i <= 5; ++i) stamp(l, i);
                dec_gdn_rec(ra, n, cs);
                stamp(l, 6);
                dense_proj(4, wout->native_type, wout->native_data, v_.xq_, v_.bo_, (int) ZV, (int) N, n, cs);
            }
            goto gdn_done;
        }
        }
        if (v_.batch_rec_) {   // each row from its segment's conv history, the slots read from the table
            gdn_conv_l2_seg(v_.segb_, (size_t) gi * gdn_floats + (size_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size,
                            qkv, (const float*) wc->data, hb, (int) C, (int) (2 * HK), EPS, n, cs);
        } else
        gdn_conv_l2_multi(conv, qkv, (const float*) wc->data, hb, (int) C, (int) (2 * HK), EPS, n, cs, 0);
        stamp(l, 3);
        gdn_ab_multi(xm, (const uint16_t*) wa->data, (const uint16_t*) wb->data, (const float*) wdt->data,
                     (const float*) wsa->data, gate, beta, (int) N, (int) HV,
                     n, cs);
        stamp(l, 4);
        dense_proj(2, wg->native_type, wg->native_data, v_.xq_, v_.z_, (int) N, (int) ZV, n, cs);
        stamp(l, 5);
        // the recurrence from the untouched state over tokens [0, T); outputs only for this group's
        if (v_.batch_rec_) {   // each segment's recurrence from its own slot's state
            gdn_step_norm_seg(v_.segb_, (size_t) gi * gdn_floats, hb, (int) C, gate, beta, v_.z_, (const float*) wnm->data,
                              EPS, v_.y_, (int) HK, (int) HV, n, cs);
        } else
        gdn_step_norm_multi(state, hb, (int) C, gate, beta, v_.z_, (const float*) wnm->data, EPS, v_.y_, (int) HK,
                            (int) HV, T, nullptr, cs, 0);
        stamp(l, 6);
        native_quantize_q8_1(v_.y_, v_.xq_, (int) ZV, n, cs);
        dense_proj(4, wout->native_type, wout->native_data, v_.xq_, v_.bo_, (int) ZV, (int) N, n, cs);
    gdn_done:;
    return true;
}

}  // namespace basalt::core
