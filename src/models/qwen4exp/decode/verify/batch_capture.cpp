#include "common.hpp"
#include "basalt/core/load_progress.hpp"
#include "basalt/core/verify.hpp"
#include "basalt/kernels/elementwise.hpp"
#include "basalt/kernels/native_qsa_indexer.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/ngram.hpp"
#include "basalt/kernels/qsa.hpp"
#include "basalt/kernels/verify_kernels.hpp"
#include <cmath>
#include <exception>
#include <string>

namespace basalt::core {
using namespace verify_detail;

bool Verifier::capture_batch(int R, int hbase, std::string& err) {
    const uint64_t key = bkey(R, hbase);
    cudaGraphExec_t& ex = exec_bm_[key];
    if (ex != nullptr) return true;
    const bool pdl_try = basalt::kernels::pdl_requested() && !pdl_off_;
    // a window that does not capture or instantiate with programmatic dependent launch is captured again without it
    auto retry_plain = [&](const std::string& why) {
        basalt::log::warn("basalt verify", "the %d-row batch window failed with programmatic dependent launch (%s): "
                             "capturing it without", R, why.c_str());
        (void) cudaGetLastError();
        pdl_off_ = true;
        ex = nullptr;
        return capture_batch(R, hbase, err);
    };
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "verify: begin batch capture failed";
        return false;
    }
    batch_rec_ = true;
    pdl_rec_ = pdl_try;
    row_base_ = hbase;
    std::string rerr;
    const bool ok = record_window(R, cs_, rerr);
    batch_rec_ = false;
    pdl_rec_ = false;
    row_base_ = 0;
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok || ce != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        if (pdl_try) return retry_plain(!ok ? rerr : cudaGetErrorString(ce));
        err = !ok ? rerr : std::string("verify: end batch capture: ") + cudaGetErrorString(ce);
        return false;
    }
    if (!instantiate_batch(ex, graph, err, true)) {
        if (pdl_try) return retry_plain(err);
        return false;
    }
    basalt::log::info("basalt verify", "captured the batch window for %d rows", R);
    return true;
}

bool Verifier::instantiate_batch(cudaGraphExec_t& ex, cudaGraph_t graph, std::string& err, bool priority) {
    const cudaError_t ie = priority ? cudaGraphInstantiateWithFlags(&ex, graph, cudaGraphInstantiateFlagUseNodePriority)
                                    : cudaGraphInstantiate(&ex, graph, 0);
    cudaGraphDestroy(graph);
    if (ie != cudaSuccess) { ex = nullptr; err = std::string("verify: batch instantiate: ") + cudaGetErrorString(ie); return false; }
    cudaGraphUpload(ex, cs_);
    cudaStreamSynchronize(cs_);
    return true;
}

bool Verifier::capture_commit_batch(int R, std::string& err) {
    cudaGraphExec_t& cex = commit_bm_[R];
    if (cex != nullptr) return true;
    using namespace basalt::kernels;
    const ModelGeometry& g = *g_;
    const QsaShapes s = shapes_of(g);
    const int64_t C = g.ssm_conv_channels, HV = g.ssm_v_heads, ID = g.idx_key_dim, MT = max_t_;
    const int64_t CB = 2 + MT;
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    const int64_t HS = (int64_t) NG_HIST * NG_HC_DIM;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "verify: begin batch commit capture failed";
        return false;
    }
    bool ok = true;
    try {
        copy_i32_from_mapped(commitb_, m_commitb_, (int64_t) slots_.size() * CB, cs_);
        copy_i32_from_mapped((int32_t*) segb_, (const int32_t*) m_segb_, (int64_t) R * 8, cs_);
        copy_i32_from_mapped((int32_t*) d_qrows_, (const int32_t*) m_qseg_, (int64_t) (kQsaSegMaxRows + 8) * 4, cs_);
        int64_t qsa_index = 0, gdn_index = 0;
        for (int64_t l = 0; l < lb_; ++l) (is_qsa_layer(g, l) ? qsa_index : gdn_index) += 1;
        {   // every segment's GDN layers in one launch (the keep counts come from the table)
            GdnCommitSeg ga;
            ga.rows = segb_; ga.state_off = (size_t) gdn_index * gdn_floats; ga.state_stride = gdn_floats;
            ga.h = h_L_ + (size_t) gdn_index * MT * C; ga.qkv = qkv_L_ + (size_t) gdn_index * MT * C;
            ga.gate = gate_L_ + (size_t) gdn_index * MT * HV; ga.beta = beta_L_ + (size_t) gdn_index * MT * HV;
            ga.h_stride = (size_t) MT * C; ga.gb_stride = (size_t) MT * HV;
            ga.C = (int) C; ga.h_k = (int) g.ssm_k_heads; ga.h_v = (int) HV; ga.n_rows = R;
            for (int64_t l = lb_; l < le_; ++l) ga.n_layers += is_qsa_layer(g, l) ? 0 : 1;
            gdn_commit_layers_seg(ga, cs_, kCommitBlocks);
        }
        {   // every QSA layer's indexer tail restore and appends, per segment, in one launch
            QsaSegCommit qc;
            qc.segs = d_qsegs_; qc.pos = commitb_ + 2; qc.pos_stride = (int) CB; qc.epsilon = EPS; qc.n_seg = R;
            for (int64_t l = lb_; l < le_ && ok; ++l) {
                if (!is_qsa_layer(g, l)) continue;
                const LayerView v(*wt_, l);
                const WeightRef* wikn = need(v, "indexer.k_norm.weight", err);
                if (!wikn || qc.n_layers >= kQsaCommitMaxLayers) { ok = false; break; }
                QsaSegCommitLayer& q = qc.layer[qc.n_layers++];
                q.slots = d_qslots_[(size_t) qsa_index];
                q.raw = idx_raw_L_ + (size_t) (qsa_index * MT) * ID;
                q.gamma = (const float*) wikn->data;
                ++qsa_index;
            }
            if (ok) native_qsa_indexer_commit_seg(qc, s, rope_scaling(), cs_);
        }
        if (ok && ss_->ple.ready() && ple_stage()) ple_hist_scatter(d_ple_slots_, hist_snap_, segb_, HS, R, cs_);
    } catch (const std::exception& e) {
        err = std::string("verify batch commit: ") + e.what();
        ok = false;
    }
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        return false;
    }
    if (ce != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("verify: batch commit capture: ") + cudaGetErrorString(ce);
        return false;
    }
    return instantiate_batch(cex, graph, err);
}

}  // namespace basalt::core
