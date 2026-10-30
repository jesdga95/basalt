#include "common.hpp"
#include "expert_plan.hpp"
#include "basalt/core/verify.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/platform/compiler.hpp"
#include "basalt/kernels/elementwise.hpp"
#include "basalt/kernels/native_qsa_indexer.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/ngram.hpp"
#include "basalt/kernels/qsa.hpp"
#include "basalt/kernels/verify_kernels.hpp"
#include <algorithm>
#include <atomic>
#include <exception>
#include <string>

namespace basalt::core {
using namespace verify_detail;

namespace verify_detail {
// The commit graph is two launches, its GDN part from 32 persistent blocks so it trickles beside the draft instead of
// filling the card
bool g_commit_async = false;   // see Verifier::set_commit_async
}

basalt::kernels::GdnCommitLayers Verifier::commit_gdn_layer(int64_t gdn_index) const {
    using namespace basalt::kernels;
    const ModelGeometry& g = *g_;
    const SessionState& ss = *ss_;
    const int64_t C = g.ssm_conv_channels, HV = g.ssm_v_heads, MT = max_t_;
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    GdnCommitLayers ga;
    ga.state = ss.gdn_state + (size_t) (gdn_index - ss.gdn_ord0) * gdn_floats;
    ga.h = h_L_ + (size_t) gdn_index * MT * C; ga.qkv = qkv_L_ + (size_t) gdn_index * MT * C;
    ga.gate = gate_L_ + (size_t) gdn_index * MT * HV; ga.beta = beta_L_ + (size_t) gdn_index * MT * HV;
    ga.state_stride = gdn_floats; ga.h_stride = (size_t) MT * C; ga.gb_stride = (size_t) MT * HV;
    ga.C = (int) C; ga.h_k = (int) g.ssm_k_heads; ga.h_v = (int) HV; ga.n_keep = commit_;
    ga.n_layers = 1;
    return ga;
}

// the stage's GDN layers from its first (only that one when the later ones are lazy)
basalt::kernels::GdnCommitLayers Verifier::commit_gdn_args() const {
    const ModelGeometry& g = *g_;
    int64_t gdn_index = 0;
    for (int64_t l = 0; l < lb_; ++l) gdn_index += is_qsa_layer(g, l) ? 0 : 1;
    basalt::kernels::GdnCommitLayers ga = commit_gdn_layer(gdn_index);
    ga.n_layers = 0;
    for (int64_t l = lb_; l < le_; ++l) ga.n_layers += is_qsa_layer(g, l) ? 0 : 1;
    if (lazy_on()) ga.n_layers = std::min(ga.n_layers, 1);
    return ga;
}

bool Verifier::finisher_on() const {
    return g_commit_async && next_ == nullptr && lb_ == 0 && le_ >= g_->n_layers && commit_ctl_ != nullptr &&
           fin_blocks_ > 0;
}

bool Verifier::lazy_on() const { return finisher_on() && dm_ != nullptr && gs_nk0_ == 0 && lz_s_ != nullptr; }

// the stage's GDN layers after its first, the lazy ones
basalt::kernels::GdnCommitLayers Verifier::commit_gdn_lazy_args() const {
    const ModelGeometry& g = *g_;
    int64_t gdn_index = 0, n = 0;
    for (int64_t l = 0; l < lb_; ++l) gdn_index += is_qsa_layer(g, l) ? 0 : 1;
    for (int64_t l = lb_; l < le_; ++l) n += is_qsa_layer(g, l) ? 0 : 1;
    basalt::kernels::GdnCommitLayers ga = commit_gdn_layer(gdn_index + 1);
    ga.n_layers = (int) std::max<int64_t>(n - 1, 0);
    return ga;
}

// every lazy layer the last commit still owes, on the commit's stream (what a window already took is done)
void Verifier::apply_lazy() {
    if (!lz_pending_) return;
    lz_pending_ = false;
    const basalt::kernels::GdnCommitLayers la = commit_gdn_lazy_args();
    basalt::kernels::gdn_commit_lazy(la, commit_ctl_, la.n_layers, nullptr, 0, cm_, fin_blocks_);
    cudaEventRecord(commit_done_, cm_);
    commit_pending_ = true;
}

bool Verifier::commit_begun() const {
    const Clock::time_point t0 = Clock::now();
    while (*(volatile uint32_t*) h_pend_ != commit_epoch_) {
        if (ms_since(t0) > 0.5) return false;
    }
    return true;
}

void Verifier::order_after_commit(bool solo) {
    if (!solo) apply_lazy();
    if (commit_pending_) cudaStreamWaitEvent(cs_, commit_done_, 0);
}

// The commit on its own stream: the epoch's counters, the indexer and PLE commits, then the GDN items from `kCommitBlocks`
// blocks that claim them one at a time (the next window's finisher claims what is left).
bool Verifier::capture_commit(std::string& err) {
    if (commit_exec_ != nullptr) return true;
    using namespace basalt::kernels;
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    const QsaShapes s = shapes_of(g);
    const int64_t ID = g.idx_key_dim, MT = max_t_;
    const int64_t TS = (s.idx_block - 1) * ID;
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    if (cudaStreamBeginCapture(cm_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "verify: begin commit capture failed";
        return false;
    }
    bool ok = true;
    try {
        if (prof_on_) gpu_stamp(prof_, (int) (g.n_layers * kProfPer + 3), cm_);
        copy_i32_from_mapped(commit_, m_commit_, 2 + MT, cm_);
        gdn_commit_begin(commit_ctl_, m_commit_ + 2 + MT, m_pend_, cm_);
        int64_t qsa_index = 0;
        for (int64_t l = 0; l < lb_; ++l) qsa_index += is_qsa_layer(g, l) ? 1 : 0;
        {   // every QSA layer's indexer appends in one launch, every GDN layer's replay in another
            const GdnCommitLayers ga = commit_gdn_args();
            QsaCommitLayers qa;
            qa.pos = commit_ + 2; qa.epsilon = EPS; qa.n_tok = (int) MT;
            for (int64_t l = lb_; l < le_ && ok; ++l) {
                if (!is_qsa_layer(g, l)) continue;
                const LayerView v(*wt_, l);
                const WeightRef* wikn = need(v, "indexer.k_norm.weight", err);
                if (!wikn || qa.n_layers >= kQsaCommitMaxLayers) { ok = false; break; }
                const QsaState& st = ss.qsa_states[qsa_index + qa.n_layers];
                QsaCommitLayer& q = qa.layer[qa.n_layers++];
                q.raw = idx_raw_L_ + (size_t) ((qsa_index + qa.n_layers - 1) * MT) * ID;
                q.tail_snap = tail_snap_ + (size_t) (qsa_index + qa.n_layers - 1) * TS;
                q.gamma = (const float*) wikn->data;
                q.tail = st.idx_tail; q.dead = st.idx_dead; q.pooled = st.idx_pooled; q.block_pos = st.idx_block_pos;
                q.max_cells = (int) st.max_cells;
            }
            if (ok) native_qsa_indexer_commit_layers(qa, s, rope_scaling(), cm_);
            if (ok && ss.ple.ready() && ple_stage()) copy_indexed(ss.ple.hist, hist_snap_, HS, commit_ + 1, HS, cm_);
            if (ok) gdn_commit_layers_dyn(ga, commit_ctl_, cm_, kCommitBlocks);
        }
        if (prof_on_) gpu_stamp(prof_, (int) (g.n_layers * kProfPer + 4), cm_);
    } catch (const std::exception& e) {
        err = std::string("verify commit: ") + e.what();
        ok = false;
    }
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cm_, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        return false;
    }
    if (ce != cudaSuccess || cudaGraphInstantiate(&commit_exec_, graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("verify: commit capture: ") + cudaGetErrorString(ce);
        return false;
    }
    cudaGraphDestroy(graph);
    return true;
}

void Verifier::set_commit_async(bool on) { g_commit_async = on; }

bool Verifier::chain_on() { return true; }

bool Verifier::commit(int n_keep, std::string& err, int parts) {
    const OnDevice on_device(device_);
    if (n_keep < 1 || n_keep > last_t_) { err = "verify: commit count out of range"; return false; }
    const Clock::time_point t0 = Clock::now();
    if (parts & 1) {
        h_commit_[0] = n_keep;
        h_commit_[1] = n_keep - 1;
        for (int t = 0; t < max_t_; ++t) h_commit_[2 + t] = t < n_keep ? (int32_t) (last_pos0_ + t) : -1;
        if (++commit_epoch_ == 0) ++commit_epoch_;
        h_commit_[2 + max_t_] = (int32_t) commit_epoch_;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        // after the window that produced the rows, on the commit's own stream
        cudaEventRecord(ev_commit_go_, cs_);
        cudaStreamWaitEvent(cm_, ev_commit_go_, 0);
        decode_events().record(DecodeEvents::kCommitPre, cm_);
        const cudaError_t le = cudaGraphLaunch(commit_exec_, cm_);
        decode_events().record(DecodeEvents::kCommitPost, cm_);
        if (gs_split_ && gs_live_) {   // the peer's heads, on its stream behind its window
            const OnDevice on(gs_dev_);
            gs_ca_.n_keep = h_commit_;
            basalt::kernels::gdn_commit_layers(gs_ca_, dm_->ps, kCommitBlocks);
        }
        if (le != cudaSuccess) { err = std::string("verify: commit launch: ") + cudaGetErrorString(le); return false; }
        // set_commit_async: no wait here - the next window's first node waits for the epoch (the host names it in
        // h_pend_), and the drafter (its own stream) reads only this window's final rows and its own K/V.  h_commit_ is
        // next written after the next window's results are read, i.e. after this graph has run.  Everything else waits
        // on commit_done_ (wait_commit).
        if (!g_commit_async || next_ != nullptr) {
            const cudaError_t se = cudaStreamSynchronize(cm_);
            if (se != cudaSuccess) { err = std::string("verify: commit: ") + cudaGetErrorString(se); return false; }
        } else {
            const cudaError_t re = cudaEventRecord(commit_done_, cm_);
            if (re != cudaSuccess) { err = std::string("verify: commit event: ") + cudaGetErrorString(re); return false; }
            (void) cudaStreamQuery(cm_);
            commit_pending_ = true;
            lz_pending_ = lazy_on();   // its later GDN layers: the next window, or apply_lazy
        }
        decode_gaps().mark(4);
    }
    if ((parts & 2) && ple_stage())   // stages that share one session must advance it once
        for (int t = 0; t < n_keep; ++t) {
            ss_->ple_prev[0] = ss_->ple_prev[1];
            ss_->ple_prev[1] = last_tokens_[t];
        }
    ms_commit += ms_since(t0);
    return next_ == nullptr || next_->commit(n_keep, err, parts);
}

bool Verifier::wait_commit(std::string& err) {
    if (commit_pending_ || lz_pending_) {
        const OnDevice on_device(device_);
        apply_lazy();
        commit_pending_ = false;
        const cudaError_t se = cudaEventSynchronize(commit_done_);
        if (se != cudaSuccess) { err = std::string("verify: commit: ") + cudaGetErrorString(se); return false; }
    }
    if (gs_live_ && !gs_move(false, err)) return false;
    return next_ == nullptr || next_->wait_commit(err);
}

}  // namespace basalt::core
