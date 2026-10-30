#include "common.hpp"
#include "expert_plan.hpp"
#include "diagnostics.hpp"
#include "basalt/core/load_progress.hpp"
#include "basalt/core/verify.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/platform/compiler.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/kernels/dec_gdn.hpp"
#include "basalt/kernels/native_ple_postops.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/ngram.hpp"
#include "basalt/kernels/qsa.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/core/progress.hpp"
#include "basalt/kernels/verify_kernels.hpp"
#include "basalt/kernels/dec_moe.hpp"
#include "basalt/log.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <string>
#include <immintrin.h>

namespace basalt::core {
using namespace verify_detail;

bool Verifier::stage_batch(const int* slots, const int* lens, int S, int hbase, const int32_t* tokens,
                           const int64_t* pos0, std::string& err) {
    using namespace basalt::kernels;
    int R = 0;
    for (int s = 0; s < S && s < 9; ++s) R += lens != nullptr ? lens[s] : 1;
    if (S < 1 || S > max_t_ || R < S || R > max_t_ || hbase < 0 || (hbase > 0 && hbase + R > (int) slots_.size())) {
        err = "verify: batch rows out of range (init_slots)";
        return false;
    }
    for (int s = 0; s < S; ++s) {
        bool dup = false;
        for (int u = 0; u < s; ++u) dup = dup || slots[u] == slots[s];
        if (slots[s] < 0 || slots[s] >= (int) slots_.size() || dup || (lens != nullptr && lens[s] < 1)) {
            err = "verify: a batch segment's slot is out of range or twice, or its length is empty";
            return false;
        }
    }
    if (released_.load()) { err = "verify: an earlier window never finished on the GPU; restart the engine"; return false; }
    const ModelGeometry& g = *g_;
    for (int s = 0; s < S; ++s)
        if (pos0[s] < 0 || pos0[s] + (lens != nullptr ? lens[s] : 1) > slots_[(size_t) slots[s]]->max_cells) {
            err = "verify: slot " + std::to_string(slots[s]) + " runs past its context";
            return false;
        }
    refresh_ar();
    if (!capture_batch(R, hbase, err) || !capture_commit_batch(R, err)) return false;
    if (R > S && !batch_seg_ple_) {
        err = "verify: batch segments of several rows need the PLE window kernels";
        return false;
    }
    const Clock::time_point t0 = Clock::now();
    const QsaShapes s = shapes_of(g);
    const int64_t CB = 2 + max_t_;
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const bool do_ple = ss_->ple.ready() && ple_stage();
    uint32_t ple_rows[kVerifyMaxT * PLE_N_HEADS];   // (not `rows`: that is the slots of the window's rows)
    int32_t* const pk = h_pos_ + (size_t) max_t_ * g.n_head;
    int32_t* const pi = pk + (size_t) max_t_ * g.n_head_kv;
    for (int t = 0; t < kQsaSegMaxRows + 8; ++t) h_qseg_[t] = QsaSeg{-1, 0, 0, 0};   // unused until set
    for (int sg = 0, r0 = 0; sg < S; ++sg) {
        const int n = lens != nullptr ? lens[sg] : 1;
        const SessionState& sx = *slots_[(size_t) slots[sg]];
        int32_t prev[2] = {sx.ple_prev[0], sx.ple_prev[1]};
        h_qseg_[kQsaSegMaxRows + sg] = QsaSeg{slots[sg], r0, n, 0};
        int32_t* c = h_commitb_ + (size_t) slots[sg] * CB;   // the commit's rows, all kept until commit_slots says less
        c[0] = n;
        c[1] = n - 1;
        for (int64_t j = 0; j < CB - 2; ++j) c[2 + j] = j < n ? (int32_t) (pos0[sg] + j) : -1;
        for (int j = 0; j < n; ++j) {
            const int t = r0 + j;
            const int64_t pos = pos0[sg] + j;
            h_tok_[t] = tokens[t];
            qsa_step_fill(h_step_ + t * kStepCount, pos, s);
            for (int64_t h = 0; h < g.n_head; ++h) h_pos_[t * g.n_head + h] = (int32_t) pos;
            for (int64_t h = 0; h < g.n_head_kv; ++h) pk[t * g.n_head_kv + h] = (int32_t) pos;
            for (int64_t h = 0; h < g.idx_q_heads; ++h) pi[t * g.idx_q_heads + h] = (int32_t) pos;
            if (do_ple) {
                ngram_rows(&tokens[t], prev, 1, ss_->ple.consts, ple_rows + t * PLE_N_HEADS);
                prev[0] = prev[1];
                prev[1] = tokens[t];
            }
            h_qseg_[t] = QsaSeg{slots[sg], r0, n, 0};
            SegRow& e = h_segb_[t];   // the state of layer 0 of the slot's session
            e.gdn = (float*) ((uintptr_t) sx.gdn_state - (uintptr_t) sx.gdn_ord0 * gdn_floats * sizeof(float));
            e.row0 = r0; e.j = j; e.n = n; e.n_keep = n; e.slot = slots[sg]; e.pos = (int32_t) pos;
            last_rows_[t] = slots[sg];
            last_tokens_[t] = tokens[t];
            last_pos_b_[t] = pos;
        }
        last_seg_slot_[sg] = slots[sg];
        last_seg_len_[sg] = n;
        r0 += n;
    }
    if (do_ple && !ss_->ple.table->gather_batch(ple_rows, (size_t) R, h_ple_, err)) return false;
    *(volatile uint32_t*) h_seq_ = 0;
    *(volatile uint32_t*) h_flag_ = 0;
    *(volatile uint32_t*) h_flagA_ = 0;
    *(volatile uint32_t*) h_flagB_ = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    last_t_ = R;
    last_nseg_ = S;
    row_base_ = hbase;   // the graphs' key (and nothing else) reads it until the next stage_batch
    last_batch_ = true;
    ms_host += ms_since(t0);
    return true;
}

bool Verifier::run_slots(int S, const int32_t* tokens, const int64_t* pos, PoolMultiFn pool, void* user, int32_t* out,
                         std::string& err) {
    int rows[8] = {};
    for (int t = 0; t < S && t < 8; ++t) rows[t] = t;
    return run_slot_rows(rows, S, tokens, pos, pool, user, out, err);
}

bool Verifier::run_slot_rows(const int* rows, int S, const int32_t* tokens, const int64_t* pos, PoolMultiFn pool,
                             void* user, int32_t* out, std::string& err) {
    return run_slot_segs(rows, nullptr, S, tokens, pos, pool, user, out, err);
}

// The staged share's RAM addresses (a mapped, 16-byte aligned copy or 0), read again when the source's places moved
// since: between windows, on the window's stream ahead of it.  The last window synchronized its stream, so the host
// copy is free to rewrite.
bool Verifier::dm_stage_places(std::string& err) {
    DecMoe* d = dm_;
    if (d == nullptr || d->stage_src == nullptr) return true;
    const uint64_t gen = warm_src_->places_generation();
    if (gen == d->stage_gen) return true;
    const ModelGeometry& g = *g_;
    const int64_t LE = g.n_layers * g.n_expert;
    bool changed = false;
    for (int64_t i = 0; i < LE; ++i) {
        const unsigned long long p = (unsigned long long) warm_src_->device_alias(i / g.n_expert, i % g.n_expert);
        const unsigned long long v = (p & 15ull) == 0 ? p : 0ull;
        changed = changed || d->stage_host[i] != v;
        d->stage_host[i] = v;
    }
    d->stage_gen = gen;
    if (changed && cudaMemcpyAsync(d->stage_src, d->stage_host, (size_t) LE * 8, cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
        err = std::string("verify: the staged share's addresses: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    return true;
}

bool Verifier::run_slot_segs(const int* rows, const int* lens, int S, const int32_t* tokens, const int64_t* pos,
                             PoolMultiFn pool, void* user, int32_t* out, std::string& err) {
    using namespace basalt::kernels;
    const OnDevice on_device(device_);
    const ModelGeometry& g = *g_;
    if (gs_live_ && !gs_move(false, err)) return false;
    gs_split_ = false;
    if (!stage_batch(rows, lens, S, 0, tokens, pos, err)) return false;
    const int R = last_t_;   // the window's rows
    const bool dm = dm_ != nullptr && batch_dm_;
    if (dm && !dm_begin(err, nullptr)) return false;
    if (dm && !dm_stage_places(err)) return false;
    order_after_commit();
    const cudaError_t le = cudaGraphLaunch(exec_bm_[bkey(R, 0)], cs_);
    if (le != cudaSuccess) { err = std::string("verify: batch launch: ") + cudaGetErrorString(le); return false; }
    (void) cudaStreamQuery(cs_);
    volatile uint32_t* const seq = h_seq_;
    volatile uint32_t* const flag = h_flag_;
    const int64_t steps = le_ - lb_;
    for (int64_t k = 0; k < steps; ++k) {
        const int64_t l = lb_ + k;
        const uint32_t want = (uint32_t) (k + 1);
        const Clock::time_point a = Clock::now();
        auto last_flush = a;
        uint32_t spins = 0;
        progress_at("verify batch: waiting for the GPU to reach layer", l);
        while (*seq < want) {
            _mm_pause();
            if ((++spins & 1023u) != 0) continue;
            const auto now = Clock::now();
            if (now - last_flush > std::chrono::microseconds(2000)) {
                last_flush = now;
                const cudaError_t q = cudaStreamQuery(cs_);
                if (q != cudaErrorNotReady && *seq < want) {
                    err = "verify batch: layer " + std::to_string(l) + " never rang (" +
                          (q == cudaSuccess ? std::string("graph finished") : std::string(cudaGetErrorString(q))) + ")";
                    return false;
                }
            }
            if (now - a > std::chrono::seconds(20)) {
                err = "verify batch: timed out at layer " + std::to_string(l) + released_note(release_gpu_waits(5000));
                return false;
            }
        }
        const Clock::time_point b = Clock::now();
        cur_layer_ = want - 1;
        set_plan_slot();
        progress_at("verify batch: the CPU experts of layer", l);
        if (dm) {   // the ring's own rows and tiers: the pool computes the CPU's entries only
            const basalt::kernels::DmRing& Rg = dm_->h_ring[k];
            sink_.ext_kind = Rg.kind;
            if (pool != nullptr) pool(user, dm_->h_x + (size_t) k * dm_->x_stride, Rg.ids, R, ss_->k, h_ymiss_, l);
            sink_.ext_kind = nullptr;
        } else if (pool != nullptr) pool(user, h_x_, h_ids_, R, ss_->k, h_ymiss_, l);
        progress_tick();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        if (*(volatile uint32_t*) h_flagA_ != want) {        // the pool did not publish a plan: an empty one
            sink_.counts[0] = 0;
            sink_.counts[1] = 0;
            sink_.counts[2] = 0;
            sink_.start[0] = 0;
            sink_.start2[0] = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) h_flagA_ = want;
            raise_flag(h_flagB_, want);
        }
        *flag = want;
        ms_wait += std::chrono::duration<double, std::milli>(b - a).count();
        ms_pool += ms_since(b);
    }
    const cudaError_t se = cudaStreamSynchronize(cs_);
    if (dm && !dm_end(err)) return false;
    if (se != cudaSuccess) { err = std::string("verify batch: ") + cudaGetErrorString(se); return false; }
    if (prof_on_) collect_profile();
    ++windows;
    if (le_ < g.n_layers) return next_ == nullptr || next_->run_slot_segs(rows, lens, S, tokens, pos, pool, next_user_, out, err);
    if (!sample_rows(R, err)) return false;
    for (int t = 0; t < R; ++t) out[t] = ((volatile int32_t*) h_out_)[t];
    progress_at("decode");
    progress_beat();
    return true;
}

bool Verifier::commit_slots(std::string& err) { return commit_slots(nullptr, err); }

bool Verifier::commit_slots(const int* n_keep, std::string& err) {
    return commit_slots_begin(n_keep, err) && commit_slots_end(err);
}

bool Verifier::commit_slots_begin(const int* n_keep, std::string& err) {
    const OnDevice on_device(device_);
    if (!last_batch_ || last_t_ < 1) { err = "verify: commit_slots without a batch window"; return false; }
    commit_t0_ = Clock::now();
    const int64_t CB = 2 + max_t_;
    if (n_keep != nullptr) {
        for (int sg = 0; sg < last_nseg_; ++sg)
            if (n_keep[sg] < 1 || n_keep[sg] > last_seg_len_[sg]) { err = "verify: commit count out of range"; return false; }
        for (int sg = 0, r0 = 0; sg < last_nseg_; r0 += last_seg_len_[sg], ++sg) {
            const int k = n_keep[sg], n = last_seg_len_[sg];
            for (int j = 0; j < n; ++j) h_segb_[r0 + j].n_keep = k;
            int32_t* c = h_commitb_ + (size_t) last_seg_slot_[sg] * CB;
            c[0] = k;
            c[1] = k - 1;
            for (int j = k; j < n; ++j) c[2 + j] = -1;
        }
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    order_after_commit();
    const cudaError_t le = cudaGraphLaunch(commit_bm_[last_t_], cs_);
    if (le != cudaSuccess) { err = std::string("verify: batch commit launch: ") + cudaGetErrorString(le); return false; }
    // (the PLE history the commit leaves is known from the tokens: it advances now, the graph runs beside the caller)
    if (ple_stage())
        for (int sg = 0, r0 = 0; sg < last_nseg_; r0 += last_seg_len_[sg], ++sg) {
            SessionState& sx = *slots_[(size_t) last_seg_slot_[sg]];
            const int k = n_keep != nullptr ? n_keep[sg] : last_seg_len_[sg];
            for (int j = 0; j < k; ++j) {
                sx.ple_prev[0] = sx.ple_prev[1];
                sx.ple_prev[1] = last_tokens_[r0 + j];
            }
        }
    commit_next_keep_.assign(n_keep != nullptr ? n_keep : nullptr, n_keep != nullptr ? n_keep + last_nseg_ : nullptr);
    return true;
}

bool Verifier::commit_slots_end(std::string& err) {
    const OnDevice on_device(device_);
    const cudaError_t se = cudaStreamSynchronize(cs_);
    if (se != cudaSuccess) { err = std::string("verify: batch commit: ") + cudaGetErrorString(se); return false; }
    ms_commit += ms_since(commit_t0_);
    return next_ == nullptr || next_->commit_slots(commit_next_keep_.empty() ? nullptr : commit_next_keep_.data(), err);
}

bool Verifier::sample_rows(int S, std::string& err) {
    bool any = false;
    for (int t = 0; t < S; ++t) {
        basalt::kernels::SamplerParams sp = slot_sp_[(size_t) last_rows_[t]];
        if (sp.greedy || sp.temperature <= 0.0f) continue;
        sp.counter = (uint64_t) last_pos_b_[t];   // Philox(seed, position): the solo window's draw for this position
        sp.penalty_last_n = 0;
        basalt::kernels::sample_tokens(head_logits_ + (size_t) t * (size_t) n_vocab_, 1, (int) n_vocab_, nullptr, 0, sp,
                                       m_out_ + t, cs_);
        any = true;
    }
    if (any && cudaStreamSynchronize(cs_) != cudaSuccess) { err = "verify batch: the row sampling failed"; return false; }
    return true;
}

bool Verifier::batch_launch(int base, int S, const int32_t* tokens, const int64_t* pos, std::string& err) {
    const OnDevice on_device(device_);
    if (b_running_) { err = "verify: batch_launch while this stage is busy"; return false; }
    int rows[8] = {};
    for (int t = 0; t < S && t < 8; ++t) rows[t] = base + t;
    if (gs_live_ && !gs_move(false, err)) return false;
    gs_split_ = false;
    if (!stage_batch(rows, nullptr, S, base, tokens, pos, err)) return false;
    order_after_commit();
    cudaError_t le = cudaGraphLaunch(exec_bm_[bkey(S, base)], cs_);
    if (le == cudaSuccess) le = cudaGraphLaunch(commit_bm_[S], cs_);   // right behind it: every row is kept
    if (le != cudaSuccess) { err = std::string("verify: batch launch: ") + cudaGetErrorString(le); return false; }
    (void) cudaStreamQuery(cs_);
    if (ple_stage())   // the host's side of the commit (the hash's last two tokens)
        for (int t = 0; t < S; ++t) {
            SessionState& sx = *slots_[(size_t) rows[t]];
            sx.ple_prev[0] = sx.ple_prev[1];
            sx.ple_prev[1] = tokens[t];
        }
    b_running_ = true;
    b_k_ = 0;
    b_steps_ = le_ - lb_;
    b_last_ = Clock::now();
    return true;
}

int Verifier::batch_poll(PoolMultiFn pool, void* user, std::string& err) {
    if (!b_running_) return 1;
    const OnDevice on_device(device_);
    volatile uint32_t* const seq = h_seq_;
    const int S = last_t_;
    while (b_k_ < b_steps_) {
        const uint32_t want = (uint32_t) (b_k_ + 1);
        if (*seq < want) {
            const auto now = Clock::now();
            if (now - b_last_ > std::chrono::milliseconds(2)) {
                const cudaError_t q = cudaStreamQuery(cs_);
                if (q != cudaErrorNotReady && *seq < want) {
                    err = "verify batch: layer " + std::to_string(lb_ + b_k_) + " never rang (" +
                          (q == cudaSuccess ? std::string("graph finished") : std::string(cudaGetErrorString(q))) + ")";
                    b_running_ = false;
                    return -1;
                }
                if (now - b_last_ > std::chrono::seconds(20)) {
                    err = "verify batch: timed out at layer " + std::to_string(lb_ + b_k_) + released_note(release_gpu_waits(5000));
                    b_running_ = false;
                    return -1;
                }
            }
            return 0;
        }
        const Clock::time_point b = Clock::now();
        cur_layer_ = want - 1;
        set_plan_slot();
        if (pool != nullptr) pool(user, h_x_, h_ids_, S, ss_->k, h_ymiss_, lb_ + b_k_);
        progress_tick();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        if (*(volatile uint32_t*) h_flagA_ != want) {        // the pool did not publish a plan: an empty one
            sink_.counts[0] = 0;
            sink_.counts[1] = 0;
            sink_.counts[2] = 0;
            sink_.start[0] = 0;
            sink_.start2[0] = 0;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile uint32_t*) h_flagA_ = want;
            raise_flag(h_flagB_, want);
        }
        *(volatile uint32_t*) h_flag_ = want;
        ms_pool += ms_since(b);
        b_last_ = Clock::now();
        ++b_k_;
    }
    const cudaError_t q = cudaStreamQuery(cs_);
    if (q == cudaErrorNotReady) return 0;
    if (q != cudaSuccess) { err = std::string("verify batch: ") + cudaGetErrorString(q); b_running_ = false; return -1; }
    if (prof_on_) collect_profile();
    if (last_stage()) {
        if (!sample_rows(S, err)) { b_running_ = false; return -1; }
        for (int t = 0; t < S; ++t) b_out_[t] = ((volatile int32_t*) h_out_)[t];
    }
    ++windows;
    b_running_ = false;
    progress_beat();
    return 1;
}

}  // namespace basalt::core
