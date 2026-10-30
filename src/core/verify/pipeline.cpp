#include "common.hpp"
#include "basalt/core/verify.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/platform/compiler.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/core/progress.hpp"
#include "basalt/kernels/verify_kernels.hpp"
#include <algorithm>
#include <atomic>
#include <string>
#include <immintrin.h>

namespace basalt::core {
using namespace verify_detail;

// ================================ PIPELINED WINDOWS (see run_pair) ================================

uint64_t Verifier::pair_snapshot_bytes(const ModelGeometry& g, const SessionState& ss, int64_t lb, int64_t le) {
    (void) ss;
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    uint64_t n_gdn = 0, n_qsa = 0;
    for (int64_t l = lb; l < le; ++l) (is_qsa_layer(g, l) ? n_qsa : n_gdn) += 1;
    return n_gdn * gdn_floats * 4 + n_qsa * (uint64_t) (kPairRows * g.idx_key_dim + 4) * 4 + 256;
}

// The state an optimistic commit changes, taken before it: the stage's recurrent state and conv history, and per QSA layer
// the indexer's pooled rows it can complete and its block position.
bool Verifier::pl_save(std::string& err) {
    if (pair_snap_ == nullptr) { err = "verify: the pair has no snapshot"; return false; }
    using namespace basalt::kernels;
    const OnDevice on_device(device_);
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    int64_t gdn0 = 0, n_gdn = 0, qsa0 = 0;
    for (int64_t l = 0; l < lb_; ++l) (is_qsa_layer(g, l) ? qsa0 : gdn0) += 1;
    for (int64_t l = lb_; l < le_; ++l) n_gdn += is_qsa_layer(g, l) ? 0 : 1;
    float* const snap = (float*) pair_snap_;
    if (n_gdn > 0 &&
        cudaMemcpyAsync(snap, ss.gdn_state + (size_t) (gdn0 - ss.gdn_ord0) * gdn_floats, (size_t) n_gdn * gdn_floats * 4,
                        cudaMemcpyDeviceToDevice, cs_) != cudaSuccess) {
        err = "verify: the pair's state snapshot";
        return false;
    }
    snap_pos0_ = last_pos0_;
    snap_ple_[0] = ss.ple_prev[0];
    snap_ple_[1] = ss.ple_prev[1];
    return pl_qsa_copy(true, err);
}

bool Verifier::pl_restore(std::string& err) {
    using namespace basalt::kernels;
    const OnDevice on_device(device_);
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    int64_t gdn0 = 0, n_gdn = 0;
    for (int64_t l = 0; l < lb_; ++l) gdn0 += is_qsa_layer(g, l) ? 0 : 1;
    for (int64_t l = lb_; l < le_; ++l) n_gdn += is_qsa_layer(g, l) ? 0 : 1;
    if (n_gdn > 0 &&
        cudaMemcpyAsync(ss.gdn_state + (size_t) (gdn0 - ss.gdn_ord0) * gdn_floats, pair_snap_, (size_t) n_gdn * gdn_floats * 4,
                        cudaMemcpyDeviceToDevice, cs_) != cudaSuccess) {
        err = "verify: the pair's state restore";
        return false;
    }
    if (!pl_qsa_copy(false, err)) return false;
    const cudaError_t se = cudaStreamSynchronize(cs_);
    if (se != cudaSuccess) { err = std::string("verify: the pair's state restore: ") + cudaGetErrorString(se); return false; }
    ss.ple_prev[0] = snap_ple_[0];
    ss.ple_prev[1] = snap_ple_[1];
    return true;
}

// The indexer part of the snapshot, to (save) or from (restore) the buffer after the recurrent state.
bool Verifier::pl_qsa_copy(bool save, std::string& err) {
    using namespace basalt::kernels;
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    int64_t n_gdn = 0, qsa0 = 0;
    for (int64_t l = 0; l < lb_; ++l) qsa0 += is_qsa_layer(g, l) ? 1 : 0;
    for (int64_t l = lb_; l < le_; ++l) n_gdn += is_qsa_layer(g, l) ? 0 : 1;
    const int64_t ID = g.idx_key_dim;
    float* buf = (float*) pair_snap_ + (size_t) n_gdn * gdn_floats;
    const int64_t block = (int64_t) shapes_of(g).idx_block;
    WordCopyList list;
    int64_t q = qsa0;
    for (int64_t l = lb_; l < le_; ++l) {
        if (!is_qsa_layer(g, l)) continue;
        const QsaState& st = ss.qsa_states[q++ - 0];
        const int64_t b0 = snap_pos0_ / block;
        const int64_t rows = std::max<int64_t>(0, std::min<int64_t>(kPairRows, st.idx_pooled_rows - b0));
        float* const mine = buf;
        buf += kPairRows * ID + 4;
        auto add = [&](const void* state, void* snap, int words) {
            WordCopy& e = list.e[list.n++];
            e.src = (const int32_t*) (save ? state : snap);
            e.dst = (int32_t*) (save ? snap : (void*) state);
            e.n = words;
            if (list.n == kWordCopyMax) {
                word_copy_list(list, cs_);
                list.n = 0;
            }
        };
        if (rows > 0) add(st.idx_pooled + (size_t) b0 * ID, mine, (int) (rows * ID));
        add(st.idx_block_pos, mine + kPairRows * ID, 1);
    }
    if (list.n > 0) word_copy_list(list, cs_);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { err = std::string("verify: the pair's indexer snapshot: ") + cudaGetErrorString(e); return false; }
    return true;
}

// Commit `n_keep` rows of the last window behind it on this verifier's stream, without waiting: the next window of
// another verifier follows `ev_opt_`.  The host's side (the PLE history) advances now.
bool Verifier::pl_commit_opt(int n_keep, std::string& err) {
    const OnDevice on_device(device_);
    if (n_keep < 1 || n_keep > last_t_) { err = "verify: commit count out of range"; return false; }
    h_commit_[0] = n_keep;
    h_commit_[1] = n_keep - 1;
    for (int t = 0; t < max_t_; ++t) h_commit_[2 + t] = t < n_keep ? (int32_t) (last_pos0_ + t) : -1;
    if (++commit_epoch_ == 0) ++commit_epoch_;
    h_commit_[2 + max_t_] = (int32_t) commit_epoch_;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const cudaError_t le = cudaGraphLaunch(commit_exec_, cs_);
    if (le != cudaSuccess) { err = std::string("verify: commit launch: ") + cudaGetErrorString(le); return false; }
    if (ev_opt_ == nullptr && cudaEventCreateWithFlags(&ev_opt_, cudaEventDisableTiming) != cudaSuccess) {
        err = "verify: event create failed";
        return false;
    }
    cudaEventRecord(ev_opt_, cs_);
    (void) cudaStreamQuery(cs_);
    if (ple_stage())
        for (int t = 0; t < n_keep; ++t) {
            ss_->ple_prev[0] = ss_->ple_prev[1];
            ss_->ple_prev[1] = last_tokens_[t];
        }
    return true;
}

bool Verifier::run_pair(Verifier& b, int ta, const int32_t* toka, int64_t pa, const std::function<int(int32_t*)>& guess,
                        int32_t* tokb, int& tb, PoolMultiFn pool, void* usera, void* userb, int32_t* outa,
                        std::string& err) {
    Verifier* const a0 = this;
    Verifier* const a1 = next_;
    Verifier* const b0 = &b;
    Verifier* const b1 = b.next_;
    tb = 0;
    if (a1 == nullptr || b1 == nullptr || a1->next_ != nullptr || b1->next_ != nullptr || pair_snap_ == nullptr ||
        b.pair_snap_ == nullptr) {
        err = "verify: a pipelined pair needs two stages and a snapshot on each chain's first stage";
        return false;
    }
    for (Verifier* v : {a1, b1}) *(volatile uint32_t*) (v->h_flag_ + 12) = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    progress_at("pipelined windows: launching the window", ta);
    if (!a0->run_begin(ta, toka, pa, err) || !a1->run_begin(ta, toka, pa, err)) return false;
    if (!a0->run_serve(pool, usera, err)) return false;
    // stage 0's last ring is served; the guess comes from drafts the drafter makes beside the window
    progress_at("pipelined windows: waiting for the drafts of the next window", ta);
    tb = guess(tokb);
    if (tb > 0) {
        // what the window changes in stage 0's state is saved, A is committed as if every row is kept, and B follows on
        // the same card
        progress_at("pipelined windows: launching the next window", tb);
        if (!a0->pl_save(err) || !a0->pl_commit_opt(ta, err)) return false;
        if (!b0->run_begin(tb, tokb, pa + ta, err, a0->ev_opt_)) return false;
    }
    progress_at("pipelined windows: serving the window and the next one", ta);
    void* const user1 = a0->next_user_;
    bool b_done = tb == 0;
    for (;;) {
        const int r = a1->serve_poll(pool, user1, err);
        if (r < 0) return false;
        if (r > 0) break;
        if (!b_done) {
            const int rb = b0->serve_poll(pool, userb, err);
            if (rb < 0) return false;
            b_done = rb > 0;
        }
        _mm_pause();
    }
    progress_at("pipelined windows: finishing the window", ta);
    return a1->run_end(outa, err) && a0->run_end(nullptr, err);
}

// Stage 1 keeps A whole.
bool Verifier::pair_hold_commit(int ta, std::string& err) { return next_->pl_commit_opt(ta, err); }

// Stage 1 runs B behind that commit.  Nothing may block on the card's other streams from here until B's rings are
// served: its kernels hold the card's SMs until the host answers them.
bool Verifier::pair_hold_begin(Verifier& b, int ta, int64_t pa, int tb, const int32_t* tokb, std::string& err) {
    return b.next_->run_begin(tb, tokb, pa + ta, err, next_->ev_opt_);
}

bool Verifier::pair_hold_finish_first(Verifier& b, PoolMultiFn pool, void* userb, std::string& err) {
    Verifier* const b0 = &b;
    progress_at("pipelined windows: finishing the next window's first stage", b0->fl_.T);
    return b0->run_serve(pool, userb, err) && b0->run_end(nullptr, err);
}

bool Verifier::pair_hold_finish(Verifier& b, PoolMultiFn pool, void* userb, int32_t* outb, std::string& err) {
    Verifier* const b0 = &b;
    Verifier* const b1 = b.next_;
    progress_at("pipelined windows: finishing the next window", b0->fl_.T);
    return b1->run_serve(pool, b0->next_user_, err) && b1->run_end(outb, err);
}

bool Verifier::pair_miss(Verifier& b, PoolMultiFn pool, void* userb, std::string& err) {
    Verifier* const b0 = &b;
    progress_at("pipelined windows: the guess missed, draining", b0->fl_.T);
    if (!b0->run_serve(pool, userb, err) || !b0->run_end(nullptr, err)) return false;   // stage 0 finishes the guess
    return pl_restore(err);
}

}  // namespace basalt::core
