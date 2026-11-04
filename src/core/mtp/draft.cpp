#include "common.hpp"

namespace basalt::core {

using namespace mtp_detail;

bool MtpDrafter::draft(int T, const int32_t* tokens, int64_t p, int a, int32_t* drafts, std::string& err,
                       float* probs, int* n_drafts) {
    if (relay_src_ != nullptr && T >= 1 && T <= max_t_) {
        const OnDevice on_device(device_);
        if (cudaMemcpyAsync(own_R_, relay_src_, (size_t) T * (size_t) (g_->hc * g_->n_embd) * sizeof(float),
                            cudaMemcpyDeviceToDevice, cs_) != cudaSuccess) {
            err = "mtp: relaying the window rows failed";
            return false;
        }
    }
    return draft_rows(T, tokens, p, a, drafts, err, probs, n_drafts);
}

bool MtpDrafter::draft_rows(int T, const int32_t* tokens, int64_t p, int a, int32_t* drafts, std::string& err,
                            float* probs, int* n_drafts) {
    const OnDevice on_device(device_);
    if (T < 1 || T > std::min(max_t_, basalt::kernels::kVerifyMaxT) || a < 0 || a >= T) {
        err = "mtp: draft arguments out of range";
        return false;
    }
    note_cells(p, p + T + max_t_ + 1);
    if (n_xcap_ > 0) {   // the window's committed picks (rows 0..a) may hold names the subset lacks
        for (int t = 0; t <= a; ++t) ctx_note(tokens[t]);
        ctx_flush(cs_);
    }
    const int max_steps = std::min(max_t_ - 1, max_drafts_);
    if (!tier2_.empty()) {   // every row for a while after a token only the added rows hold
        for (int t = 0; t < T; ++t)
            if (tokens[t] >= 0 && tokens[t] < (int32_t) tier2_.size() && tier2_[(size_t) tokens[t]]) tier2_recent_ = 32;
        tier_ = tier2_recent_ > 0 ? 1 : 0;
        if (tier2_recent_ > 0) --tier2_recent_;
    }
    // the chain in one graph (falls back to the graphs below for good if it cannot be built)
    const bool cond = !cond_off_ && max_steps > 1 && capture_cond(T, max_steps);
    if (!cond) {
        if (!capture_round(T, err)) return false;
        for (int j = 1; j < max_steps; ++j)
            if (!capture_step(j, err)) return false;
    }
    const Clock::time_point t0 = Clock::now();
    const int64_t NH = g_->n_head;
    auto put = [&](int row, int64_t cell) {
        h_step_[row * 4 + 0] = (int32_t) cell;
        h_step_[row * 4 + 1] = (int32_t) (cell + 1);
        h_step_[row * 4 + 2] = (int32_t) ((cell + 1) / 4);
        h_step_[row * 4 + 3] = (int32_t) (cell + 1);
        for (int64_t h = 0; h < NH; ++h) h_pos_[row * NH + h] = (int32_t) cell;
    };
    for (int t = 0; t < T; ++t) {
        h_tok_[t] = tokens[t];
        put(t, p + t);
    }
    put(2 * max_t_ - 1, p + a);   // p + a: draft 0's cell
    for (int j = 1; j < max_steps; ++j)
        put(max_t_ + j - 1, p + a + j);
    for (int j = 0; j < max_steps; ++j) ((volatile int32_t*) h_out_)[j] = -1;
    h_row_[0] = a;
    h_row_[1] = 0;
    // slot 2 is reserved: the row layout is fixed by the graphs that copy it
    const float cum_th = cum_th_ >= 0.0f ? cum_th_ : draft_cum_base();
    std::memcpy(&h_row_[3], &cum_th, sizeof(float));   // the window's threshold, read by the conditional graph
    std::atomic_thread_fence(std::memory_order_seq_cst);

    SessionState* const pss = ple_ss_ ? ple_ss_ : ss_;
    const bool do_ple = pss != nullptr && pss->ple.ready() && pss->ple.table != nullptr;
    int32_t ple_prev[2] = {do_ple ? pss->ple_prev[0] : 0, do_ple ? pss->ple_prev[1] : 0};
    auto prefetch_ple = [&](int32_t tok) {
        if (!do_ple || tok < 0) return;
        uint32_t rows16[basalt::kernels::PLE_N_HEADS];
        basalt::kernels::ngram_rows(&tok, ple_prev, 1, pss->ple.consts, rows16);
        ple_prev[0] = ple_prev[1];
        ple_prev[1] = tok;
        pss->ple.table->prefetch_rows(rows16);
    };

    int n = 0;
    if (max_steps > 0) {
        decode_gaps().mark(6);
        const cudaGraphExec_t first = cond ? cond_exec_[tier_][T] : round_exec_[tier_][T];
        decode_events().record(DecodeEvents::kDraftPre, cs_);
        if (cudaGraphLaunch(first, cs_) != cudaSuccess) {
            err = std::string("mtp draft: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        decode_events().record(DecodeEvents::kDraftPost, cs_);
        (void) cudaStreamQuery(cs_);
        decode_gaps().mark(7);
        run_launch_hook();
        prefetch_ple(tokens[a]);
        auto wait_step = [&](int j) {
            uint32_t spins = 0;
            while (((volatile int32_t*) h_out_)[j] < 0) {
                basalt::platform::cpu_relax();
                if ((++spins & 1023u) == 0 && cudaStreamQuery(cs_) != cudaErrorNotReady) break;
            }
        };
        wait_step(0);
        // every draft arrived: the graph's kernels are done, only its skipped steps' conditional nodes remain
        bool arrived = cond && ((volatile int32_t*) h_out_)[0] >= 0;
        drafts[0] = ((volatile int32_t*) h_out_)[0];
        float pr[kMtpChainMax];
        float pj = ((volatile float*) h_prob_)[0];
        pr[0] = pj;
        if (probs) probs[0] = pj;
        n = 1;
        const float host_th = host_th_ >= 0.0f ? host_th_ : cum_th;
        chain_steps_ = max_steps;
        for (int j = 1; j < max_steps && draft_keep(pr, j - 1, host_th); ++j) {
            if (!cond) {   // the conditional graph runs step j on the device by the same comparison
                if (cudaGraphLaunch(step_exec_[tier_][j], cs_) != cudaSuccess) {
                    err = std::string("mtp draft step: ") + cudaGetErrorString(cudaGetLastError());
                    return false;
                }
                (void) cudaStreamQuery(cs_);
            }
            prefetch_ple(drafts[j - 1]);
            wait_step(j);
            if (cond && ((volatile int32_t*) h_out_)[j] < 0) {   // the graph ended without it: the sync says why
                arrived = false;
                break;
            }
            drafts[j] = ((volatile int32_t*) h_out_)[j];
            pj = ((volatile float*) h_prob_)[j];
            pr[j] = pj;
            if (probs) probs[j] = pj;
            ++n;
        }
        // the last draft is a window row only when it passed the cutoff; a row nobody needs would make the window's
        // PLE collect wait for its SSD read
        if (draft_keep(pr, n - 1, host_th)) prefetch_ple(drafts[n - 1]);
        decode_gaps().mark(8);
        // the skipped steps' nodes finish beside the next window; the profile stamps and the logits copy need the end
        const bool sync = !arrived || gstamp_ != nullptr || diag_ != nullptr;
        if (sync && cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = std::string("mtp draft: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
    }
    run_launch_hook();   // no graph ran: the hook still runs, once
    for (int j = n; j < max_t_ - 1; ++j) { drafts[j] = 0; if (probs) probs[j] = 0.0f; }
    if (n_drafts) *n_drafts = n;
    decode_gaps().mark(9);
    if (gstamp_ != nullptr) {   // --profile: the GPU's idle time before the round and the draft's span
        unsigned long long st[2] = {0, 0};
        cudaMemcpy(st, gstamp_, 16, cudaMemcpyDeviceToHost);
        DecodeGaps& dg = decode_gaps();
        if (dg.verify_end != 0 && st[0] > dg.verify_end && st[1] > st[0]) {
            dg.v2d += (double) (st[0] - dg.verify_end);
            dg.draft_gpu += (double) (st[1] - st[0]);
            ++dg.n_v2d;
        }
        dg.draft_end = st[1];
    }
    decode_gaps().mark(10);
    ms_draft += ms_since(t0);
    ++rounds;
    return true;
}

// The drafts the device made past the ones draft() waited for: its conditional graph runs the chain on while the host
// does other work.  Those that have arrived are taken; the host never waits here, because a window on this card may be
// holding its SMs until the host serves its rings, and the chain's kernels would wait behind it.
cudaEvent_t MtpDrafter::done_event() {
    const OnDevice on_device(device_);
    if (ev_done_ == nullptr && cudaEventCreateWithFlags(&ev_done_, cudaEventDisableTiming) != cudaSuccess) return nullptr;
    cudaEventRecord(ev_done_, cs_);
    return ev_done_;
}

int MtpDrafter::draft_more(int have, int32_t* drafts, float* probs) {
    int n = have;
    for (; n < chain_steps_ && n < kMtpChainMax; ++n) {
        if (((volatile int32_t*) h_out_)[n] < 0) break;
        drafts[n] = ((volatile int32_t*) h_out_)[n];
        if (probs) probs[n] = ((volatile float*) h_prob_)[n];
    }
    return n;
}

bool MtpDrafter::stage_rows(const float* R_rows, int n, const float** staged, std::string& err) {
    *staged = R_rows;
    if (relay_ == nullptr) return true;
    if (n < 1 || n > kRelayRows) { err = "mtp: staging rows out of range"; return false; }
    const OnDevice on_device(device_);
    if (cudaMemcpyAsync(relay_, R_rows, (size_t) n * (size_t) (g_->hc * g_->n_embd) * sizeof(float),
                        cudaMemcpyDeviceToDevice, cs_) != cudaSuccess ||
        cudaStreamSynchronize(cs_) != cudaSuccess) {
        err = "mtp: staging the window rows failed";
        return false;
    }
    *staged = relay_;
    return true;
}

bool MtpDrafter::draft_from(const float* R_rows, int T, const int32_t* tokens, int64_t p, int a, int32_t* drafts,
                            std::string& err, float* probs, int* n_drafts) {
    if (R_rows != window_R_) {
        const OnDevice on_device(device_);
        if (cudaMemcpyAsync((void*) window_R_, R_rows, (size_t) T * (size_t) (g_->hc * g_->n_embd) * sizeof(float),
                            cudaMemcpyDeviceToDevice, cs_) != cudaSuccess) {
            err = "mtp: staging the window rows failed";
            return false;
        }
    }
    return draft_rows(T, tokens, p, a, drafts, err, probs, n_drafts);
}

bool MtpDrafter::draft_first(int T, const float* R_row, int32_t token, int64_t cell, int32_t* drafts, std::string& err,
                             float* probs, int* n_drafts) {
    const OnDevice on_device(device_);
    // row 0 is the real pair; rows 1.. repeat it and only write cells the next round overwrites
    const int64_t HCN = g_->hc * g_->n_embd;
    for (int t = 0; t < T; ++t)
        if (cudaMemcpy((void*) (window_R_ + (size_t) t * HCN), R_row, (size_t) HCN * sizeof(float),
                       cudaMemcpyDeviceToDevice) != cudaSuccess) {
            err = "mtp: staging the first residual failed";
            return false;
        }
    std::vector<int32_t> toks((size_t) T, token);
    return draft_rows(T, toks.data(), cell, 0, drafts, err, probs, n_drafts);
}


}  // namespace basalt::core
