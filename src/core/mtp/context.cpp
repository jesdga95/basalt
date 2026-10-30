#include "common.hpp"

namespace basalt::core {

using namespace mtp_detail;

void MtpDrafter::ctx_note(int64_t t) {
    if (n_xcap_ == 0 || t < 0 || t >= n_vocab_ || in_sub_[(size_t) t]) return;
    if ((int64_t) xlist_.size() + (int64_t) xpend_.size() >= n_xcap_) return;
    in_sub_[(size_t) t] = 1;
    xpend_.push_back((int32_t) t);
}

void MtpDrafter::begin_request(const int64_t* prompt, int64_t n) {
    if (n_xcap_ == 0) return;
    const OnDevice on_device(device_);
    for (int32_t t : xlist_) in_sub_[(size_t) t] = 0;
    for (int32_t t : xpend_) in_sub_[(size_t) t] = 0;
    if (n_xused_ > 0) cudaMemsetAsync(dvocab_ + xbase_, 0xff, (size_t) n_xused_ * 4, cs_);   // ids -1
    if (!diag_map_.empty()) std::fill(diag_map_.begin() + xbase_, diag_map_.begin() + xbase_ + n_xcap_, -1);
    xlist_.clear();
    xpend_.clear();
    n_xused_ = 0;
    // a prompt with more such tokens than its share of the rows keeps its latest ones, which the reply is likelier
    // to repeat: counted from the end (marked 2), then noted in prompt order from the first of them
    const int64_t cap = std::min(n_xcap_, kDraftCtxPromptRows);
    int64_t from = 0, distinct = 0;
    for (int64_t i = n - 1; i >= 0; --i) {
        const int64_t t = prompt[i];
        if (t < 0 || t >= n_vocab_ || in_sub_[(size_t) t] != 0) continue;
        if (distinct == cap) { from = i + 1; break; }
        in_sub_[(size_t) t] = 2;
        ++distinct;
    }
    for (int64_t i = from; i < n; ++i)
        if (prompt[i] >= 0 && prompt[i] < n_vocab_ && in_sub_[(size_t) prompt[i]] == 2) in_sub_[(size_t) prompt[i]] = 0;
    for (int64_t i = from; i < n; ++i) ctx_note(prompt[i]);
}

// The queued tokens' head rows into the next free context rows, ahead of the draft graphs on the same stream.
void MtpDrafter::ctx_flush(cudaStream_t cs) {
    if (xpend_.empty()) return;
    const int64_t n = (int64_t) xpend_.size();
    const int64_t row_bytes = (int64_t) head_->row_bytes();
    const int64_t at = xbase_ + n_xused_;
    cudaMemcpyAsync(dxids_, xpend_.data(), (size_t) n * 4, cudaMemcpyHostToDevice, cs);
    basalt::kernels::gather_rows((const uint8_t*) head_->weights(), row_bytes, dxids_, n, dhead_ + at * row_bytes, cs);
    if (proxy_ != nullptr) basalt::kernels::q5k_proxy_pack(dhead_, at, n, proxy_, n_tier1_, cs);
    cudaMemcpyAsync(dvocab_ + at, dxids_, (size_t) n * 4, cudaMemcpyDeviceToDevice, cs);
    if (!diag_map_.empty()) std::copy(xpend_.begin(), xpend_.end(), diag_map_.begin() + at);
    xlist_.insert(xlist_.end(), xpend_.begin(), xpend_.end());
    n_xused_ += n;
    xpend_.clear();
}


}
