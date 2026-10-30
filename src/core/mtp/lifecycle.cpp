// src/core/mtp.cpp - see include/basalt/core/mtp.hpp.
#include "common.hpp"

namespace basalt::core {

using namespace mtp_detail;

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
