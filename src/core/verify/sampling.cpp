#include "common.hpp"
#include "expert_plan.hpp"
#include "basalt/core/verify.hpp"
#include "basalt/core/native_head.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/platform/compiler.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/q6k_head.hpp"
#include "basalt/kernels/q80_vhead.hpp"
#include "basalt/kernels/sampler.hpp"
#include <exception>
#include <string>
#include <vector>

namespace basalt::core {
using namespace verify_detail;

bool Verifier::set_logit_bias(const std::vector<float>& bias, std::string& err) {
    if (next_ && !next_->set_logit_bias(bias, err)) return false;
    if (next_) return true;   // an earlier stage: the head, and so the sampling, is on the last one
    const OnDevice on_device(device_);
    if (!bias.empty() && bias.size() != (size_t) n_vocab_) {
        err = "logit_bias: the vocabulary size does not match";
        return false;
    }
    if (bias != logit_bias_host_ && !bias.empty()) {
        const size_t bytes = bias.size() * sizeof(float);
        if ((!logit_bias_device_ && cudaMalloc((void**) &logit_bias_device_, bytes) != cudaSuccess) ||
            cudaMemcpy(logit_bias_device_, bias.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "logit_bias: the upload failed";
            return false;
        }
    }
    logit_bias_host_ = bias;
    sampling_.logit_bias = bias.empty() ? nullptr : logit_bias_device_;
    return true;
}

int Verifier::head_keep() const {
    using namespace basalt::kernels;
    if (!head_sampling_) return 1;   // the recorded greedy pick only
    if (sampling_.logit_bias != nullptr) return 0;
    if (hist_d_ != nullptr && sampling_.penalty_last_n > 0 &&
        (sampling_.penalty_repeat != 1.0f || sampling_.penalty_freq != 0.0f || sampling_.penalty_present != 0.0f))
        return 0;
    if (sampling_.greedy || sampling_.temperature <= 0.0f) return 1;
    const int n = sampler_list_len(sampling_.top_k, (int) n_vocab_);
    return n <= kQ80pKeepMax ? n : 0;
}

void Verifier::set_head_keep(int keep) {
    // the window records the two-stage head only where it ends the model, without the head split
    const bool on = head_keep_d_ != nullptr && le_ >= g_->n_layers && (hs_rows_ == 0 || hs_two_);
    head_screened_ = on && keep > 0;
    if (!on || keep == head_keep_set_) return;
    cudaMemsetAsync(head_keep_d_, keep, 1, cs_);   // the low byte: the value is 0..64, the rest of the word is 0
    head_keep_set_ = keep;
}

// on cs_, behind the window: whatever follows on the stream reads exact logits
bool Verifier::head_exact_again() const {
    if (!head_screened_) return true;
    const OnDevice on_device(device_);
    basalt::kernels::q80p_mmvq(head_->weights(), xq_, head_logits_, (int) g_->n_embd, (int) n_vocab_, last_t_, cs_);
    head_screened_ = false;
    return cudaPeekAtLastError() == cudaSuccess;
}

bool Verifier::exact_head_rows(int T, std::string& err) {
    using namespace basalt::kernels;
    if (head_screened_) {
        if (head_exact_again()) return true;
        err = "verify head: the exact pass failed";
        return false;
    }
    if (head_ == nullptr || !head_->loaded() || head_->type() != 14 || vhead_scratch_ == nullptr || hs_rows_ > 0 ||
        g_ == nullptr || !q6k_vhead_supported((int) g_->n_embd, (int) n_vocab_, T))
        return true;                         // this window's head was exact
    const OnDevice on_device(device_);
    const int N = (int) g_->n_embd;
    try {
        if (q6k_head_supported(N, T))
            q6k_head_mmvq(head_->weights(), xq_, head_logits_, N, (int) n_vocab_, T, cs_);
        else
            native_mmvq(head_->type(), head_->weights(), xq_, head_logits_, N, (int) n_vocab_, T, cs_);
    } catch (const std::exception& e) {
        err = std::string("verify head: ") + e.what();
        return false;
    }
    return true;
}

bool Verifier::reserve_wide_sampler(std::string& err) {
    if (next_ != nullptr) return next_->reserve_wide_sampler(err);   // a layer split: the head is on the last stage
    const OnDevice on_device(device_);
    if (basalt::kernels::sampler_wide_reserve((int) n_vocab_, cs_)) return true;
    err = "the sampler has no VRAM for a list longer than 64 entries (top_k 0 or above 64): about " +
          std::to_string((16 * (int64_t) n_vocab_ >> 20) + 1) + " MiB plus the sort's own";
    return false;
}

bool Verifier::resample_row(int t, int64_t pos0, const std::vector<float>& bias, int32_t& token, std::string& err) {
    if (next_ != nullptr) return next_->resample_row(t, pos0, bias, token, err);   // a layer split: the head is on the last stage
    const OnDevice on_device(device_);
    if (head_logits_ == nullptr || t < 0 || bias.size() != (size_t) n_vocab_) { err = "resample_row: no such row or a bad bias"; return false; }
    if (!head_exact_again()) { err = "resample_row: the exact head failed"; return false; }   // the mask reads every row
    const size_t bytes = bias.size() * sizeof(float);
    // on cs_ itself: the stream does not order against a plain cudaMemcpy, so the draw could read the previous bias
    if ((!row_bias_device_ && cudaMalloc((void**) &row_bias_device_, bytes) != cudaSuccess) ||
        cudaMemcpyAsync(row_bias_device_, bias.data(), bytes, cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
        err = "resample_row: the bias upload failed";
        return false;
    }
    // a batch window's row draws with its own slot's sampling and position, as the window drew it
    basalt::kernels::SamplerParams sp;
    const int32_t* hist = nullptr;
    int hist_n = 0;
    if (last_batch_ && t < last_t_) {
        sp = slot_sp_[(size_t) last_rows_[t]];
        sp.counter = (uint64_t) last_pos_b_[t];
        if (bh_d_ != nullptr && sp.penalty_last_n > 0) { hist = bh_d_ + (size_t) t * (size_t) bh_len_; hist_n = bh_len_; }
        else sp.penalty_last_n = 0;
    } else {
        sp = sampling_;
        sp.counter = (uint64_t) pos0 + (uint64_t) t;
        if (hist_d_ != nullptr) { hist = hist_d_ + (size_t) t * (size_t) hist_len_; hist_n = (int) hist_len_; }
    }
    sp.logit_bias = row_bias_device_;
    basalt::kernels::sample_tokens(head_logits_ + (size_t) t * (size_t) n_vocab_, 1, (int) n_vocab_, hist, hist_n, sp,
                  m_out_, cs_);
    if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = "resample_row: the sampling failed"; return false; }
    token = ((volatile int32_t*) h_out_)[0];
    return true;
}

bool Verifier::copy_logits(int t, float* host) const {
    if (next_ != nullptr) return next_->copy_logits(t, host);   // a layer split: the head is on the last stage
    if (head_logits_ == nullptr || host == nullptr || t < 0 || n_vocab_ <= 0) return false;
    if (head_screened_ && (!head_exact_again() || cudaStreamSynchronize(cs_) != cudaSuccess)) return false;
    return cudaMemcpy(host, head_logits_ + (size_t) t * (size_t) n_vocab_, (size_t) n_vocab_ * sizeof(float),
                      cudaMemcpyDeviceToHost) == cudaSuccess;
}

}  // namespace basalt::core
