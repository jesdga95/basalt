// ---- Jev: independent sequences over one resident recurrent prefix ----
#include "basalt/core/verify.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/core/decision_schedule.hpp"
#include "basalt/kernels/ngram.hpp"
#include <algorithm>
#include <numeric>

namespace basalt::core {

bool Verifier::run_score_prefix(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool, void* user,
                                int32_t* out, std::string& err) {
    if (next_) return run(T, tokens, pos0, pool, user, out, err);
    struct Restore {
        Verifier& v;
        bool sampling;
        ~Restore() { v.score_prefill_ = false; v.head_sampling_ = sampling; }
    } restore{*this, head_sampling_};
    score_prefill_ = true;
    head_sampling_ = false;
    return run(T, tokens, pos0, pool, user, out, err);
}

void Verifier::release_score_prefix_graphs() {
    const OnDevice on_device(device_);
    for (auto& variant : score_exec_)
        for (auto& graph : variant)
            if (graph) { cudaGraphExecDestroy(graph); graph = nullptr; }
}

bool Verifier::score_branches_supported() const {
    if (!ss_ || next_ || lb_ != 0 || le_ != g_->n_layers || b_running_) return false;
    for (int64_t q = 0; q < g_->n_qsa_layers(); ++q) {
        const auto& st = ss_->qsa_states[q];
        if (st.kv_mode || st.kv_rot || st.kv_q4 || st.kv_hybrid) return false;
    }
    return true;
}

void Verifier::release_score_workspace() {
    release_score_prefix_graphs();
    release_score_branches();
}

void Verifier::release_score_branches() {
    if (!score_workspace_ && !score_arena_) return;
    const OnDevice on_device(device_);
    score_workspace_.reset();
    if (score_arena_) cudaFree(score_arena_);
    score_arena_ = nullptr; score_bytes_ = 0; score_cells_ = 0;
    score_views_.clear(); score_qsa_.clear(); score_base_ = nullptr;
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
                              uint64_t& workspace_bytes, uint64_t free_floor, std::string& err,
                              const std::function<bool()>& cancelled,
                              const std::function<bool(std::string&)>& before_window) {
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
    if (S < 2 || S > max_t_ || prefix < 0 || prefix + longest > ss_->max_cells) return false;
    const bool recurrent = R > max_t_;
    const int64_t cells = std::min(ss_->max_cells, (prefix + longest + 255) & ~int64_t{255});
    const int64_t nQ = g_->n_qsa_layers();
    const uint64_t per = (qsa_prompt_state_bytes(*g_, cells) + 255) & ~255ull;
    const uint64_t hist = ((uint64_t) basalt::kernels::NG_HIST * basalt::kernels::NG_HC_DIM * 4 + 255) & ~255ull;
    const uint64_t gdn_bytes = (uint64_t) g_->n_gdn_layers() *
        ((uint64_t) g_->ssm_state_size * g_->ssm_v_heads * g_->ssm_state_size +
         (uint64_t) g_->ssm_conv_channels * (g_->ssm_d_conv - 1)) * sizeof(float);
    const uint64_t gdn = (gdn_bytes + 255) & ~255ull;
    const uint64_t bytes = S * (nQ * per + hist + (recurrent ? gdn : 0));
    constexpr uint64_t margin = 8ull << 20;
    if (!score_workspace_ || score_base_ != ss_ || score_cells_ < cells ||
        (int) score_views_.size() < S || recurrent != score_recurrent_) {
        release_score_branches();
        size_t free = 0, total = 0;
        if (cudaMemGetInfo(&free, &total) != cudaSuccess || free < bytes + margin + free_floor) return false;
        if (cudaMalloc(&score_arena_, bytes) != cudaSuccess) { cudaGetLastError(); return false; }
        score_bytes_ = bytes; score_cells_ = cells; score_base_ = ss_; score_recurrent_ = recurrent;
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
            if (recurrent) { view.gdn_state = (float*) base; base += gdn; }
        }
        score_workspace_ = std::make_unique<Verifier>();
        swap_score_workspace(*score_workspace_);
        std::vector<SessionState*> ptrs;
        for (auto& view : score_views_) ptrs.push_back(&view);
        const bool ok = init_slots(ptrs, err);
        swap_score_workspace(*score_workspace_);
        if (!ok) {
            const auto failure = cudaGetLastError();
            release_score_branches();
            if (failure == cudaErrorMemoryAllocation) err.clear();
            return false;
        }
        if (cudaMemGetInfo(&free, &total) != cudaSuccess || free < free_floor) {
            release_score_branches(); return false;
        }
    }
    const auto copy = [&](void* dst, const void* src, uint64_t n) {
        return cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice, cs_) == cudaSuccess;
    };
    for (int s = 0; s < S; ++s) {
        auto& view = score_views_[s];
        if (recurrent) {
            if (!copy(view.gdn_state, ss_->gdn_state, gdn_bytes)) { err = "Jev: recurrence copy failed"; return false; }
        } else view.gdn_state = ss_->gdn_state;
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
            v.score_prefill_ = false;
        }
    } restore{*this};
    int out[8], offsets[8]{}, remaining[8]{};
    int64_t positions[8];
    for (int i = 0; i < S; ++i) {
        branches[i].logits.clear(); branches[i].windows = 0;
        basalt::kernels::SamplerParams sp{}; sp.greedy = true; sp.temperature = 0;
        set_slot_sampling(i, sp);
    }
    std::vector<float> logits((size_t) n_vocab_);
    for (;;) {
        if (cancelled && cancelled()) return false;
        for (int i = 0; i < S; ++i) remaining[i] = (int) branches[i].tokens.size() - offsets[i];
        const auto plan = decision_window(std::span<const int>(remaining, S), max_t_);
        const int active = plan.size, used = plan.rows;
        if (!active) break;
        if (before_window && !before_window(err)) return false;
        const auto& rows = plan.slots;
        const auto& lens = plan.lengths;
        std::vector<int32_t> tokens;
        bool terminal = false;
        for (int j = 0; j < active; ++j) {
            const int i = rows[j];
            // Every live sequence advances; unused rows go to longer suffixes without padding.
            positions[j] = prefix + offsets[i];
            tokens.insert(tokens.end(), branches[i].tokens.begin() + offsets[i],
                          branches[i].tokens.begin() + offsets[i] + lens[j]);
            terminal = terminal || lens[j] == remaining[i];
        }
        score_prefill_ = !terminal;
        if (!run_slot_segs(rows.data(), lens.data(), active, tokens.data(), positions, pool, user, out, err)) return false;
        if (terminal) {
            if (!exact_head_rows(used, err)) return false;
            if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = "Jev: exact head failed"; return false; }
        }
        for (int j = 0, end = 0; j < active; ++j) {
            const int i = rows[j]; end += lens[j]; offsets[i] += lens[j];
            if (offsets[i] != (int) branches[i].tokens.size()) continue;
            if (!copy_logits(end - 1, logits.data())) { err = "Jev: target logit copy failed"; return false; }
            for (int id : branches[i].candidates) branches[i].logits.push_back(logits[(size_t) id]);
        }
        ++branches.front().windows;
        // Only private recurrence may commit; the common prefix remains a reusable source.
        if (recurrent && !commit_slots(err)) return false;
    }
    workspace_bytes = score_bytes_ + batch_bytes_;
    return true;
}

} // namespace basalt::core
// ---- end Jev ----
