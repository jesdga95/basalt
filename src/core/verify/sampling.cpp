#include "common.hpp"
#include "expert_plan.hpp"
#include "basalt/core/verify.hpp"
#include "basalt/core/native_head.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/platform/compiler.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/sampler.hpp"
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

bool Verifier::resample_row(int t, int64_t pos0, const std::vector<float>& bias, int32_t& token, std::string& err) {
    if (next_ != nullptr) return next_->resample_row(t, pos0, bias, token, err);   // a layer split: the head is on the last stage
    const OnDevice on_device(device_);
    if (head_logits_ == nullptr || t < 0 || bias.size() != (size_t) n_vocab_) { err = "resample_row: no such row or a bad bias"; return false; }
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
        sp.penalty_last_n = 0;
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
    return cudaMemcpy(host, head_logits_ + (size_t) t * (size_t) n_vocab_, (size_t) n_vocab_ * sizeof(float),
                      cudaMemcpyDeviceToHost) == cudaSuccess;
}

}  // namespace basalt::core
