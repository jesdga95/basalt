#include "common.hpp"
#include "record.hpp"
#include "expert_plan.hpp"
#include "basalt/core/verify.hpp"
#include "basalt/core/native_head.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/kernels/fused_gr.hpp"
#include "basalt/kernels/hc_fused.hpp"
#include "basalt/kernels/cvec.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/q5k_head.hpp"
#include "basalt/kernels/q6k_head.hpp"
#include "basalt/kernels/q80_head.hpp"
#include "basalt/kernels/sampler.hpp"
#include <algorithm>
#include <exception>
#include <string>

namespace basalt::core {
using namespace verify_detail;

bool Verifier::WindowRecorder::head() {
    using namespace basalt::kernels;
    // ---- the head, T columns, and the argmax of each
    stamp(g.n_layers, 0);
    {
        const WeightRef *hn = wt.find("output_hc_norm.weight"), *hd = wt.find("output_hc_down.weight"),
                        *hu = wt.find("output_hc_up.weight");
        if (!hn || !hd || !hu) { err = "verify: an output_hc_* weight is missing"; return false; }
        if (fuse_head_gr) {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                FusedGrArgs& a = fa[t];
                a.R = Rt(t); a.R_out = Rt(t); a.apply = true;
                a.bo_prev = v_.bo_ + t * N; a.inj_prev = v_.inj2_ + t * HC;
                a.w_norm = (const float*) hn->data;
                a.w_down = (const uint16_t*) hd->data;
                a.w_up = (const uint16_t*) hu->data;
                a.w_inject = nullptr;
                a.eps = EPS;
                a.lo = v_.lo_ + t * g.hc_lr;
                a.rs = v_.rs_ + t * HC;
                a.inject_out = v_.head_inj_;
                a.mixed = v_.head_mixed_ + t * N;
            }
            if (dec_hc) {
                HcFusedArgs h;
                h.T = T; h.R = Rt(0); h.R_out = Rt(0); h.apply = true; h.bo_prev = v_.bo_; h.inj_prev = v_.inj2_;
                h.w_norm = (const float*) hn->data; h.w_down = (const uint16_t*) hd->data; h.w_up = (const uint16_t*) hu->data;
                h.eps = EPS; h.lo = v_.lo_; h.rs = v_.rs_; h.mixed = v_.head_mixed_; h.xn = v_.xn_;
                hc_fused_read(h, cs);
            } else {
                fused_gr_read_multi(fa, T, v_.xn_, cs);
            }
        } else {
            for (int t = 0; t < T; ++t) {
                BlockBuffers bb = ss.block;
                bb.R = Rt(t);
                bb.mixed = v_.head_mixed_ + t * N;
                if (v_.head_ != nullptr && v_.head_->loaded()) {
                    if (!lm_head_mix(wt, g, bb, cs, err)) return false;
                } else if (!lm_head(wt, g, bb, v_.head_logits_ + (size_t) t * v_.n_vocab_, cs, err)) {
                    return false;
                }
            }
        }
        if (v_.head_ != nullptr && v_.head_->loaded()) {
            try {
                native_quantize_q8_1(v_.head_mixed_, v_.xq_, (int) N, T, cs);
                // rows [0, n) of `w` into y (T rows of n); the same kernel and bits as the whole head
                auto head_rows = [&](const void* w, float* y, int n, cudaStream_t s) {
                    const void* x = s == cs ? (const void*) v_.xq_ : v_.hs_xq_;
                    if (v_.head_->type() == 13 && q5k_head_supported((int) N, T))
                        q5k_head_mmvq(w, x, y, (int) N, n, T, s);
                    else if (v_.head_->type() == 14 && q6k_head_supported((int) N, T))
                        q6k_head_mmvq(w, x, y, (int) N, n, T, s);
                    else if (v_.head_->type() == 8 && T >= 4 && q80_head_supported((int) N, T))
                        q80_head_mmvq(w, x, y, (int) N, n, T, s);
                    else
                        native_mmvq(v_.head_->type(), w, x, y, (int) N, n, T, s);
                };
                if (v_.hs_rows_ > 0) {   // the peer's rows while CUDA0 does its own; its logits land beside CUDA0's
                    const int r0 = v_.hs_rows_, r1 = (int) v_.n_vocab_ - r0;
                    const size_t xb = native_q8_1_bytes((int) N, T);
                    bool ok = cudaMemcpyAsync(v_.hs_hxq_, v_.xq_, xb, cudaMemcpyDeviceToHost, cs) == cudaSuccess &&
                              cudaEventRecord(v_.hs_in_, cs) == cudaSuccess;
                    {
                        const OnDevice on(v_.hs_dev_);
                        ok = ok && cudaStreamWaitEvent(v_.hs_s_, v_.hs_in_, 0) == cudaSuccess &&
                             cudaMemcpyAsync(v_.hs_xq_, v_.hs_hxq_, xb, cudaMemcpyHostToDevice, v_.hs_s_) == cudaSuccess;
                        if (ok) head_rows(v_.hs_w_, v_.hs_y_, r1, v_.hs_s_);
                        ok = ok && cudaMemcpyAsync(v_.hs_hy_, v_.hs_y_, (size_t) T * r1 * 4, cudaMemcpyDeviceToHost, v_.hs_s_) == cudaSuccess &&
                             cudaEventRecord(v_.hs_out_, v_.hs_s_) == cudaSuccess;
                    }
                    head_rows(v_.head_->weights(), v_.hs_y0_, r0, cs);
                    ok = ok && cudaMemcpy2DAsync(v_.head_logits_, (size_t) v_.n_vocab_ * 4, v_.hs_y0_, (size_t) r0 * 4, (size_t) r0 * 4, T,
                                                 cudaMemcpyDeviceToDevice, cs) == cudaSuccess &&
                         cudaStreamWaitEvent(cs, v_.hs_out_, 0) == cudaSuccess &&
                         cudaMemcpy2DAsync(v_.head_logits_ + r0, (size_t) v_.n_vocab_ * 4, v_.hs_hy_, (size_t) r1 * 4, (size_t) r1 * 4, T,
                                           cudaMemcpyHostToDevice, cs) == cudaSuccess;
                    if (!ok) { err = "verify: the split head's hand-off failed"; return false; }
                } else if (v_.head_->type() == 13 && q5k_head_supported((int) N, T))   // the same bits, one wave
                    q5k_head_mmvq(v_.head_->weights(), v_.xq_, v_.head_logits_, (int) N, (int) v_.n_vocab_, T, cs);
                else if (v_.head_->type() == 14 && v_.vhead_scratch_ != nullptr && q6k_vhead_supported((int) N, (int) v_.n_vocab_, T))
                    q6k_vhead(v_.head_->vproxy(), v_.head_->weights(), v_.xq_, v_.head_logits_, (int) N, (int) v_.n_vocab_, T,
                              v_.vhead_scratch_, cs);
                else if (v_.head_->type() == 14 && q6k_head_supported((int) N, T))
                    q6k_head_mmvq(v_.head_->weights(), v_.xq_, v_.head_logits_, (int) N, (int) v_.n_vocab_, T, cs);
                else if (v_.head_->type() == 8 && T >= 4 && q80_head_supported((int) N, T))   // four columns or more: the same bits as one wave
                    q80_head_mmvq(v_.head_->weights(), v_.xq_, v_.head_logits_, (int) N, (int) v_.n_vocab_, T, cs);
                else
                    native_mmvq(v_.head_->type(), v_.head_->weights(), v_.xq_, v_.head_logits_, (int) N, (int) v_.n_vocab_, T, cs);
            } catch (const std::exception& e) {
                err = std::string("verify head: ") + e.what();
                return false;
            }
        }
        // Greedy, the default, is recorded here as before (no extra launch or sync per window). A request that
        // samples or penalizes is sampled again host-side after the replay (run()) with its own parameters and a
        // fresh draw counter: a captured sampler would bake them in and replay the same draws forever.
        SamplerParams sp;
        sp.greedy = true;
        sp.temperature = 0.0f;
        sample_tokens(v_.head_logits_, T, (int) v_.n_vocab_, nullptr, 0, sp, v_.m_out_, cs);
    }
    stamp(g.n_layers, 1);
    return true;
}

}  // namespace basalt::core
