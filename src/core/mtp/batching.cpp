#include "common.hpp"

namespace basalt::core {

using namespace mtp_detail;

const float* MtpDrafter::keep_rows(const float* R_rows, int rows, std::string& err) {
    if (rkeep_ == nullptr || rows > kKeepRows) { err = "mtp: no room to keep the window rows"; return nullptr; }
    const OnDevice on_device(device_);
    if (cudaMemcpyAsync(rkeep_, R_rows, (size_t) rows * (size_t) (g_->hc * g_->n_embd) * sizeof(float),
                        cudaMemcpyDeviceToDevice, cs_) != cudaSuccess) {
        err = "mtp: keeping the window rows failed";
        return nullptr;
    }
    return rkeep_;
}

bool MtpDrafter::draft_seq(int id, const float* R_rows, int T, const int32_t* tokens, int64_t p, int a, int32_t* drafts,
                           std::string& err, float* probs, int* n_drafts) {
    if (!flush(id, err) || !use_store(id, err)) return false;
    const OnDevice on_device(device_);
    const size_t bytes = (size_t) T * (size_t) (g_->hc * g_->n_embd) * sizeof(float);
    if (R_rows != window_R_) {   // through a copy: the rows may overlap the ones the graphs read
        if (rstage_ == nullptr) { err = "mtp: no row staging"; return false; }
        if (cudaMemcpyAsync(rstage_, R_rows, bytes, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess ||
            cudaMemcpyAsync((void*) window_R_, rstage_, bytes, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess) {
            err = "mtp: staging the window rows failed";
            return false;
        }
    }
    return draft_rows(T, tokens, p, a, drafts, err, probs, n_drafts);
}

bool MtpDrafter::flush(int id, std::string& err) {
    if (id < 0 || id >= (int) pend_tok_.size() || pend_tok_[(size_t) id].empty()) return true;
    std::vector<int32_t> tk;
    tk.swap(pend_tok_[(size_t) id]);
    if (!use_store(id, err)) return false;
    return prefill_cur(pend_R_[(size_t) id], tk.data(), (int64_t) tk.size(), pend_cell0_[(size_t) id], err);
}

bool MtpDrafter::defer_rows(int id, const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0, std::string& err) {
    if (id < 0 || id >= (int) pend_tok_.size()) { err = "mtp: no such K/V store"; return false; }
    std::vector<int32_t>& tk = pend_tok_[(size_t) id];
    if (!tk.empty() && (pend_cell0_[(size_t) id] + (int64_t) tk.size() != cell0 || (int64_t) tk.size() + n > kPendRows)) {
        if (!flush(id, err)) return false;
    }
    if (n > kPendRows) return catch_up(id, R_rows, next_tokens, n, cell0, err);
    const OnDevice on_device(device_);
    const size_t HCN = (size_t) (g_->hc * g_->n_embd);
    if (tk.empty()) pend_cell0_[(size_t) id] = cell0;
    if (cudaMemcpyAsync(pend_R_[(size_t) id] + tk.size() * HCN, R_rows, (size_t) n * HCN * sizeof(float),
                        cudaMemcpyDeviceToDevice, cs_) != cudaSuccess) {
        err = "mtp: keeping the window rows failed";
        return false;
    }
    tk.insert(tk.end(), next_tokens, next_tokens + n);
    return true;
}

bool MtpDrafter::catch_up(int id, const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0, std::string& err) {
    if (!flush(id, err) || !use_store(id, err)) return false;
    return prefill_cur(R_rows, next_tokens, n, cell0, err);
}


}
