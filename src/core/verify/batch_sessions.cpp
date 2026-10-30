#include "common.hpp"
#include "expert_plan.hpp"
#include "basalt/core/load_progress.hpp"
#include "basalt/core/verify.hpp"
#include "basalt/core/remote_expert_opt.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/kernels/dec_gdn.hpp"
#include "basalt/kernels/native_ple_postops.hpp"
#include "basalt/kernels/ngram.hpp"
#include "basalt/kernels/qsa.hpp"
#include "basalt/kernels/qsa_dec_fused.hpp"
#include "basalt/kernels/verify_kernels.hpp"
#include "basalt/log.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace basalt::core {
using namespace verify_detail;

// ================================ BATCH WINDOWS (see init_slots) ================================

bool Verifier::init_slots(const std::vector<SessionState*>& slots, std::string& err) {
    const OnDevice on_device(device_);
    if (g_ == nullptr || ss_ == nullptr) { err = "verify: init_slots before init"; return false; }
    if (slots.empty() || (int) slots.size() > max_t_) {
        err = "verify: init_slots needs 1.." + std::to_string(max_t_) + " sessions";
        return false;
    }
    for (SessionState* x : slots) {
        if (x == nullptr || x->layer_lo != ss_->layer_lo || x->layer_hi != ss_->layer_hi ||
            x->max_cells > ss_->max_cells || x->gdn_ord0 != ss_->gdn_ord0 || x->qsa_ord0 != ss_->qsa_ord0) {
            err = "verify: a slot's session is not carved like the stage's own (layer range, a context at most the stage's)";
            return false;
        }
    }
    const basalt::kernels::QsaShapes s = shapes_of(*g_);
    const int64_t S = (int64_t) slots.size(), CB = 2 + max_t_;
    const int64_t TS = (s.idx_block - 1) * g_->idx_key_dim, nQ = g_->n_qsa_layers();
    void* d = nullptr;
    if (!mapped((size_t) (S * CB * 4 + 16), (void**) &h_commitb_, (void**) &m_commitb_)) {
        err = "verify: the batch commit staging failed";
        return false;
    }
    if (!mapped((size_t) max_t_ * sizeof(basalt::kernels::SegRow), (void**) &h_segb_, (void**) &m_segb_)) {
        err = "verify: the batch segment staging failed";
        return false;
    }
    using namespace basalt::kernels;
    if (!mapped((size_t) (basalt::kernels::kQsaSegMaxRows + 8) * sizeof(QsaSeg), (void**) &h_qseg_, (void**) &m_qseg_)) {
        err = "verify: the batch QSA table staging failed";
        return false;
    }
    std::memset(h_qseg_, 0, (size_t) (basalt::kernels::kQsaSegMaxRows + 8) * sizeof(QsaSeg));
    // the QSA chain's segment kernels need resident, unrotated INT8 K/V in every slot
    if (!(qsa_dec_fused_supported(s))) {
        err = "verify: batch windows need the native QSA kernels";
        return false;
    }
    for (int64_t l = lb_; l < le_; ++l) {
        if (!is_qsa_layer(*g_, l)) continue;
        int64_t qi = 0;
        for (int64_t m = 0; m < l; ++m) qi += is_qsa_layer(*g_, m) ? 1 : 0;
        for (SessionState* x : slots) {
            const QsaState& st = x->qsa_states[qi];
            if (st.kv_mode != 0 || !st.kv_int8 || st.kv_rot || st.kv_hybrid || st.kv_q4) {
                err = "verify: batch windows need resident INT8 K/V without rotation (--kv int8)";
                return false;
            }
        }
    }
    const uint64_t a = ((uint64_t) S * CB * 4 + 255) & ~255ull;
    const uint64_t b = (a + (uint64_t) S * std::max<int64_t>(nQ, 1) * TS * 4 + 255) & ~255ull;
    const uint64_t b2 = (b + (uint64_t) max_t_ * sizeof(basalt::kernels::SegRow) + 255) & ~255ull;
    const uint64_t b3 = (b2 + (uint64_t) (basalt::kernels::kQsaSegMaxRows + 8) * sizeof(QsaSeg) + 255) & ~255ull;
    const uint64_t HSb = (uint64_t) NG_HIST * NG_HC_DIM * sizeof(float);
    const uint64_t b4 = (b3 + (uint64_t) S * sizeof(float*) + 255) & ~255ull;
    const uint64_t b5 = (b4 + (uint64_t) max_t_ * HSb + 255) & ~255ull;
    if (cudaMalloc(&d, b5 + (uint64_t) std::max<int64_t>(nQ, 1) * S * sizeof(QsaSegSlot)) != cudaSuccess) {
        err = "verify: the batch buffers do not fit";
        return false;
    }
    arena_b_ = d;
    commitb_ = (int32_t*) d;
    tail_snap_b_ = (float*) ((uint8_t*) d + a);
    segb_ = (basalt::kernels::SegRow*) ((uint8_t*) d + b);
    d_qrows_ = (QsaSeg*) ((uint8_t*) d + b2);
    d_qsegs_ = d_qrows_ + basalt::kernels::kQsaSegMaxRows;
    d_ple_slots_ = (float**) ((uint8_t*) d + b3);
    ple_work_ = (float*) ((uint8_t*) d + b4);
    {   // static per-slot pointers: the PLE histories and, per QSA layer, the slots' caches and indexer state
        std::vector<float*> ph((size_t) S);
        for (int64_t i = 0; i < S; ++i) ph[(size_t) i] = slots[(size_t) i]->ple_hist;
        cudaMemcpy(d_ple_slots_, ph.data(), (size_t) S * sizeof(float*), cudaMemcpyHostToDevice);
        d_qslots_.assign((size_t) nQ, nullptr);
        QsaSegSlot* dev = (QsaSegSlot*) ((uint8_t*) d + b5);
        for (int64_t l = lb_; l < le_; ++l) {
            if (!is_qsa_layer(*g_, l)) continue;
            int64_t qi = 0;
            for (int64_t m = 0; m < l; ++m) qi += is_qsa_layer(*g_, m) ? 1 : 0;
            std::vector<QsaSegSlot> hs((size_t) S);
            for (int64_t i = 0; i < S; ++i) {
                const QsaState& st = slots[(size_t) i]->qsa_states[qi];
                QsaSegSlot& e = hs[(size_t) i];
                e.k_pool = st.k_pool; e.v_pool = st.v_pool; e.k_q = st.k_q; e.v_q = st.v_q;
                e.k_scale = st.k_scale; e.v_scale = st.v_scale; e.page_table = st.page_table; e.host = st.host;
                e.tail = st.idx_tail; e.dead = st.idx_dead; e.pooled = st.idx_pooled;
                e.tail_snap = tail_snap_b_ + ((size_t) i * nQ + qi) * TS;
                e.block_pos = st.idx_block_pos; e.max_cells = (int32_t) st.max_cells;
            }
            cudaMemcpy(dev, hs.data(), (size_t) S * sizeof(QsaSegSlot), cudaMemcpyHostToDevice);
            d_qslots_[(size_t) qi] = dev;
            dev += S;
        }
    }
    slots_ = slots;
    batch_dm_ = dm_ != nullptr;
    slot_sp_.assign(slots.size(), sampling_);   // greedy until set_slot_sampling
    basalt::log::info("basalt verify", "batch windows of up to %lld sequences (layers [%lld, %lld))", (long long) S,
                 (long long) lb_, (long long) le_);
    if (dm_ != nullptr && dm_->stage_src != nullptr)
        basalt::log::info("basalt verify", "batch windows: a layer with %d+ CPU misses sends a share of them over PCIe to "
                          "CUDA0", dm_->stage_min);
    return true;
}

bool Verifier::bind_session(SessionState* s, std::string& err) {
    if (s == nullptr || ss_ == nullptr) { err = "verify: bind_session without a session"; return false; }
    if (s == ss_) return true;
    if (!wait_commit(err)) return false;   // the last commit still works on the old session's state
    SoloGraphs cur;
    std::copy(std::begin(exec_), std::end(exec_), std::begin(cur.exec));
    std::copy(std::begin(exec_nr_), std::end(exec_nr_), std::begin(cur.exec_nr));
    cur.commit = commit_exec_;
    solo_sets_[ss_] = cur;
    SoloGraphs nx;
    const auto it = solo_sets_.find(s);
    if (it != solo_sets_.end()) {
        nx = it->second;
        solo_sets_.erase(it);
    }
    std::copy(std::begin(nx.exec), std::end(nx.exec), std::begin(exec_));
    std::copy(std::begin(nx.exec_nr), std::end(nx.exec_nr), std::begin(exec_nr_));
    commit_exec_ = nx.commit;
    ss_ = s;
    return true;
}

bool Verifier::warm_session(SessionState* s, std::string& err) {
    SessionState* home = ss_;
    if (!bind_session(s, err)) return false;
    refresh_ar();
    bool ok = capture_commit(err);
    for (int T = 1; ok && T <= max_t_; ++T) ok = capture(T, err);
    std::string e2;
    if (!bind_session(home, e2) && ok) { err = e2; return false; }
    return ok;
}

bool Verifier::warm_graphs(std::string& err) {
    refresh_ar();
    for (int T = 1; T <= max_t_; ++T)
        if (!capture(T, err)) return false;
    if (slots_.empty()) return true;
    for (int R = 1; R <= max_t_; ++R)
        if (!capture_batch(R, 0, err) || !capture_commit_batch(R, err)) return false;
    return true;
}

}  // namespace basalt::core
