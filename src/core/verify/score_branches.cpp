// ---- Jev: noncommitting suffix waves over one resident recurrent prefix ----
#include "basalt/core/verify.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/kernels/ngram.hpp"
#include <algorithm>
#include <numeric>

namespace basalt::core {

bool Verifier::score_branches_supported() const {
    if (!ss_ || next_ || lb_ != 0 || le_ != g_->n_layers || b_running_) return false;
    for (int64_t q = 0; q < g_->n_qsa_layers(); ++q) {
        const auto& st = ss_->qsa_states[q];
        if (st.kv_mode || st.kv_rot || st.kv_q4 || st.kv_hybrid) return false;
    }
    return true;
}

void Verifier::swap_score_workspace(Verifier& o) {
    using std::swap;
    swap(slots_, o.slots_); swap(slot_sp_, o.slot_sp_);
    swap(exec_bm_, o.exec_bm_); swap(commit_bm_, o.commit_bm_);
    swap(arena_b_, o.arena_b_); swap(batch_bytes_, o.batch_bytes_); swap(batch_dm_, o.batch_dm_);
    swap(h_commitb_, o.h_commitb_); swap(m_commitb_, o.m_commitb_); swap(commitb_, o.commitb_);
    swap(tail_snap_b_, o.tail_snap_b_); swap(h_segb_, o.h_segb_); swap(m_segb_, o.m_segb_); swap(segb_, o.segb_);
    swap(h_qseg_, o.h_qseg_); swap(m_qseg_, o.m_qseg_); swap(d_qrows_, o.d_qrows_); swap(d_qsegs_, o.d_qsegs_);
    swap(d_qslots_, o.d_qslots_); swap(d_ple_slots_, o.d_ple_slots_); swap(ple_work_, o.ple_work_);
    swap(bh_d_, o.bh_d_); swap(bh_len_, o.bh_len_);
    swap(last_batch_, o.last_batch_); swap(last_t_, o.last_t_); swap(last_rows_, o.last_rows_);
    swap(last_pos_b_, o.last_pos_b_); swap(last_nseg_, o.last_nseg_);
    swap(last_seg_slot_, o.last_seg_slot_); swap(last_seg_len_, o.last_seg_len_);
}

bool Verifier::score_branches(std::vector<ScoreBranch>& branches, int64_t prefix, PoolMultiFn pool, void* user,
                              uint64_t& workspace_bytes, uint64_t free_floor, std::string& err) {
    const OnDevice on_device(device_);
    workspace_bytes = 0;
    if (!score_branches_supported() || !wait_commit(err)) return false;
    const int S = (int) branches.size();
    int R = 0, longest = 0;
    for (const auto& b : branches) {
        if (b.tokens.empty() || b.candidates.empty() || b.candidates.size() > 255 ||
            std::any_of(b.tokens.begin(), b.tokens.end(), [&](int t) { return t < 0 || t >= n_vocab_; }) ||
            std::any_of(b.candidates.begin(), b.candidates.end(), [&](int t) { return t < 0 || t >= n_vocab_; })) return false;
        R += (int) b.tokens.size();
        longest = std::max(longest, (int) b.tokens.size());
    }
    if (S < 2 || S > max_t_ || R > max_t_ || prefix < 0 || prefix + longest > ss_->max_cells) return false;
    const int64_t cells = std::min(ss_->max_cells, (prefix + longest + 255) & ~int64_t{255});
    const int64_t nQ = g_->n_qsa_layers();
    const uint64_t per = (qsa_prompt_state_bytes(*g_, cells) + 255) & ~255ull;
    const uint64_t hist = ((uint64_t) basalt::kernels::NG_HIST * basalt::kernels::NG_HC_DIM * 4 + 255) & ~255ull;
    const uint64_t bytes = S * (nQ * per + hist);
    constexpr uint64_t cap = 128ull << 20, margin = 8ull << 20;
    if (bytes + margin > cap) return false;
    if (!score_workspace_ || score_base_ != ss_ || score_cells_ < cells || (int) score_views_.size() < S) {
        score_workspace_.reset();
        if (score_arena_) cudaFree(score_arena_);
        score_arena_ = nullptr; score_bytes_ = 0; score_cells_ = 0;
        size_t free = 0, total = 0;
        if (cudaMemGetInfo(&free, &total) != cudaSuccess || free < bytes + margin + free_floor) return false;
        if (cudaMalloc(&score_arena_, bytes) != cudaSuccess) { cudaGetLastError(); return false; }
        score_bytes_ = bytes; score_cells_ = cells; score_base_ = ss_;
        score_views_.resize(S); score_qsa_.assign((size_t) S * nQ, QsaState{});
        auto* base = (uint8_t*) score_arena_;
        for (int s = 0; s < S; ++s) {
            auto& view = score_views_[s];
            view = *ss_; view.max_cells = cells; view.qsa_states = score_qsa_.data() + s * nQ;
            for (int64_t q = 0; q < nQ; ++q) {
                qsa_prompt_state_carve(*g_, cells, base, view.qsa_states[q]); base += per;
                qsa_prompt_state_clear(view.qsa_states[q], *g_, cs_);
            }
            view.ple_hist = (float*) base; base += hist;
        }
        score_workspace_ = std::make_unique<Verifier>();
        swap_score_workspace(*score_workspace_);
        std::vector<SessionState*> ptrs;
        for (auto& view : score_views_) ptrs.push_back(&view);
        const bool ok = init_slots(ptrs, err);
        swap_score_workspace(*score_workspace_);
        if (!ok) { score_workspace_.reset(); return false; }
    }
    const auto copy = [&](void* dst, const void* src, uint64_t n) {
        return cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice, cs_) == cudaSuccess;
    };
    for (int s = 0; s < S; ++s) {
        auto& view = score_views_[s];
        view.gdn_state = ss_->gdn_state;
        std::copy_n(ss_->ple_prev, 2, view.ple_prev);
        view.ple = ss_->ple; view.ple.hist = view.ple_hist; view.ple.prev = view.ple_prev; view.ple.token = &view.ple_token;
        for (int64_t q = 0; q < nQ; ++q) {
            if (prefix == 0) qsa_prompt_state_clear(view.qsa_states[q], *g_, cs_);
            if (!qsa_state_copy_cells(view.qsa_states[q], ss_->qsa_states[q], *g_, 0, prefix, copy, err)) return false;
        }
        if (ss_->ple_hist && !copy(view.ple_hist, ss_->ple_hist,
                                  (uint64_t) basalt::kernels::NG_HIST * basalt::kernels::NG_HC_DIM * 4)) {
            err = "Jev: PLE history copy failed"; return false;
        }
    }
    swap_score_workspace(*score_workspace_);
    struct Restore {
        Verifier& v;
        ~Restore() {
            v.swap_score_workspace(*v.score_workspace_);
            // Shared scratch no longer contains a committable solo window.
            v.last_t_ = 0; v.last_batch_ = false;
        }
    } restore{*this};
    int rows[8], lens[8], out[8];
    int64_t positions[8];
    std::vector<int32_t> tokens;
    for (int i = 0; i < S; ++i) {
        rows[i] = i; lens[i] = (int) branches[i].tokens.size(); positions[i] = prefix;
        tokens.insert(tokens.end(), branches[i].tokens.begin(), branches[i].tokens.end());
        basalt::kernels::SamplerParams sp{}; sp.greedy = true; sp.temperature = 0;
        set_slot_sampling(i, sp);
    }
    if (!run_slot_segs(rows, lens, S, tokens.data(), positions, pool, user, out, err)) return false;
    std::vector<float> logits((size_t) n_vocab_);
    if (!exact_head_rows(R, err)) return false;
    if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = "Jev: exact head failed"; return false; }
    for (int i = 0, end = 0; i < S; ++i) {
        end += lens[i];
        if (!copy_logits(end - 1, logits.data())) { err = "Jev: target logit copy failed"; return false; }
        auto& b = branches[i]; b.logits.clear();
        for (int id : b.candidates) b.logits.push_back(logits[(size_t) id]);
    }
    // No commit: each segment read the same recurrence and wrote only its private QSA/PLE window.
    workspace_bytes = score_bytes_ + batch_bytes_;
    return true;
}

} // namespace basalt::core
// ---- end Jev ----
