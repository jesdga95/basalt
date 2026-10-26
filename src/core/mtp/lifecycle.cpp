// src/core/mtp/lifecycle.cpp - see include/basalt/core/mtp.hpp.
#include "common.hpp"

namespace basalt::core {

using namespace mtp_detail;

bool MtpDrafter::reset_model_state(std::string& err) {
    const OnDevice on_device(device_);
    auto checked = [&](cudaError_t status, const char* what) {
        if (status == cudaSuccess) return true;
        err = std::string("mtp reset: ") + what + ": " + cudaGetErrorString(status);
        return false;
    };
    int current = -1;
    if (!checked(cudaGetDevice(&current), "querying device")) return false;
    if (device_ < 0 || current != device_ || g_ == nullptr || cs_ == nullptr) {
        err = "mtp reset: the drafter is not loaded on its owning device";
        return false;
    }
    for (cudaStream_t stream : {cs_, cs2_, cs3_})
        if (stream && !checked(cudaStreamSynchronize(stream), "draining stream")) return false;
    auto clear = [&](void* p, size_t bytes, int value = 0) {
        return p == nullptr || checked(cudaMemsetAsync(p, value, bytes, cs_), "clearing sequence state");
    };
    const size_t T = (size_t) max_t_, NH = (size_t) g_->n_head;
    if (!clear(tok_, T * sizeof(int32_t)) ||
        !clear(step_, 2 * T * 4 * sizeof(int32_t)) ||
        !clear(pos_, 2 * T * NH * sizeof(int32_t)) ||
        !clear(row_, 4 * sizeof(int32_t)) ||
        !clear(out_ids_, (T + 4) * sizeof(int32_t), 0xff) ||
        !clear(probs_, (T + 4) * sizeof(float)) ||
        !clear(st_.step, 4 * sizeof(int32_t)) ||
        !clear(st_.pos_dev, NH * sizeof(int32_t)) ||
        !clear(gstamp_, 2 * sizeof(unsigned long long)) ||
        (n_xcap_ > 0 && !clear(dvocab_ + xbase_, (size_t) n_xcap_ * sizeof(int32_t), 0xff)))
        return false;
    if (!checked(cudaStreamSynchronize(cs_), "finishing sequence reset")) return false;

    std::fill_n(h_tok_, T, 0);
    std::fill_n(h_step_, 2 * T * 4, 0);
    std::fill_n(h_pos_, 2 * T * NH, 0);
    std::fill_n(h_row_, 4, 0);
    std::fill_n(h_out_, T, -1);
    std::fill_n(h_prob_, T, 0.0f);
    if (st_.host_step) std::fill_n(st_.host_step, 4, 0);
    if (st_.host_pos) std::fill_n(st_.host_pos, NH, 0);
    for (int32_t t : xlist_) in_sub_[(size_t) t] = 0;
    for (int32_t t : xpend_) in_sub_[(size_t) t] = 0;
    if (!diag_map_.empty()) std::fill(diag_map_.begin() + xbase_, diag_map_.begin() + xbase_ + n_xcap_, -1);
    xlist_.clear();
    xpend_.clear();
    n_xused_ = 0;
    tier2_recent_ = 0;
    tier_ = tier2_.empty() ? 1 : 0;
    for (auto& tokens : pend_tok_) tokens.clear();
    std::fill(pend_cell0_.begin(), pend_cell0_.end(), 0);
    std::fill(store_hi_.begin(), store_hi_.end(), 0);
    // Dense attention reads only newly written cells after a position-zero restart, including ring and slot stores.
    dirty_lo_ = dirty_hi_ = 0;
    cur_ = n_stores_;
    prompt_len_ = 0;
    launch_hook_ = nullptr;
    par_ = pre_next_ = have_pre_ = fork_open_ = pf_open_ = false;
    ctail_ = nullptr;
    ctail_done_ = false;
    chain_j_ = chain_steps_ = n_drafted_ = 0;
    rounds = 0;
    ms_draft = ms_prefill = 0;
    return true;
}

MtpDrafter::~MtpDrafter() {
    if (cs_) cudaStreamSynchronize(cs_);
    if (ev_done_) cudaEventDestroy(ev_done_);
    for (auto& e : prefill_exec_) if (e) cudaGraphExecDestroy(e);
    for (auto& e : prefill_dev_exec_) if (e) cudaGraphExecDestroy(e);
    if (pf_dev_) cudaFree(pf_dev_);
    for (auto& st : stores_)
        for (void* a : st) if (a) cudaFree(a);
    if (rstage_) cudaFree(rstage_);
    if (rkeep_) cudaFree(rkeep_);
    if (own_R_) cudaFree(own_R_);
    if (relay_) cudaFree(relay_);
    for (float* r : pend_R_) if (r) cudaFree(r);
    for (auto& row : round_exec_)
        for (auto& e : row) if (e) cudaGraphExecDestroy(e);
    for (auto& row : step_exec_)
        for (auto& e : row) if (e) cudaGraphExecDestroy(e);
    for (auto& row : cond_exec_)
        for (auto& e : row) if (e) cudaGraphExecDestroy(e);
    if (cs_) cudaStreamDestroy(cs_);
    if (cs2_) cudaStreamDestroy(cs2_);
    if (cs3_) cudaStreamDestroy(cs3_);
    if (ev_pf_) cudaEventDestroy(ev_pf_);
    if (ev_pfj_) cudaEventDestroy(ev_pfj_);
    if (ev_fork_) cudaEventDestroy(ev_fork_);
    if (ev_join_) cudaEventDestroy(ev_join_);
    if (dense_) cudaFree(dense_);
    if (experts_) cudaFree(experts_);
    if (state_arena_) cudaFree(state_arena_);
    if (arena_) cudaFree(arena_);
    if (head_logits_) cudaFree(head_logits_);
    if (dhead_) cudaFree(dhead_);
    if (dvocab_) cudaFree(dvocab_);
    if (top_scratch_) cudaFree(top_scratch_);
    if (proxy_) cudaFree(proxy_);
    if (proxy_scratch_) cudaFree(proxy_scratch_);
    if (gstamp_) cudaFree(gstamp_);
    void* hosts[] = {h_tok_, h_step_, h_pos_, h_row_, h_out_, h_prob_};
    for (void* h : hosts) if (h) cudaFreeHost(h);
}


}
