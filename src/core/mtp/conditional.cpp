#include "common.hpp"

namespace basalt::core {

using namespace mtp_detail;

// [round, h1 <- rule(p0)] -> IF h1 [step 1, h2 <- rule(p0, p1)] -> IF h2 [...] ...  Every record
// is copied once, by the round.  The host applies the same rule to the same probabilities, so it waits only for
// the drafts the graph makes; the device side only removes the host round trip between steps.
bool MtpDrafter::prewarm(std::string& err) {
    const OnDevice on_device(device_);
    const int max_steps = std::min(max_t_ - 1, max_drafts_);
    if (cond_off_ || max_steps < 2) return true;
    const int saved = tier_;
    bool ok = true;
    for (int tier = 0; ok && tier < (tier2_.empty() ? 1 : 2); ++tier) {
        tier_ = tier2_.empty() ? saved : tier;
        for (int T = 1; ok && T <= std::min(max_t_, basalt::kernels::kVerifyMaxT) && !cond_off_; ++T)
            ok = capture_cond(T, max_steps);
    }
    tier_ = saved;
    if (!ok && cond_off_) return true;   // the per-step graphs take over, as at a lazy failure
    if (!ok) err = "mtp: draft graph capture";
    return ok;
}

bool MtpDrafter::capture_cond(int T, int steps) {
    cudaGraphExec_t& exec = cond_exec_[tier_][T];
    if (exec && cond_steps_[tier_][T] == steps) return true;
    if (exec) {
        cudaGraphExecDestroy(exec);
        exec = nullptr;
    }
    using namespace basalt::kernels;
    std::string err;
    cudaGraph_t top = nullptr;
    cudaGraphConditionalHandle h[kMtpChainMax] = {};
    unsigned long long hv[kMtpChainMax] = {};
    bool ok = steps >= 2 && steps <= kMtpChainMax - 1 && cudaGraphCreate(&top, 0) == cudaSuccess;
    for (int j = 1; ok && j < steps; ++j) {
        // reset to 0 at every launch: a skipped step leaves the next one's handle at 0
        ok = cudaGraphConditionalHandleCreate(&h[j], top, 0, cudaGraphCondAssignDefault) == cudaSuccess;
        hv[j] = (unsigned long long) h[j];
    }
    cudaGraphNode_t prev = nullptr;
    if (ok) ok = cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
    if (ok) {
        CondTail ct{};
        for (int k = 0; k + 1 < steps; ++k) ct.h[k] = hv[k + 1];
        ct.n = steps - 1;
        ctail_ = &ct;
        ctail_done_ = false;
        bool rec = record_round(T, true, err);
        ctail_ = nullptr;
        if (rec && !ctail_done_)
            rec = mtp_draft_cond(hv + 1, steps - 1, probs_, row_ + 1, row_ + 3, cs_, m_prob_, 0);
        cudaGraph_t g0 = nullptr;
        const bool ended = cudaStreamEndCapture(cs_, &g0) == cudaSuccess;
        ok = rec && ended && g0 != nullptr && cudaGraphAddChildGraphNode(&prev, top, nullptr, 0, g0) == cudaSuccess;
        if (g0) cudaGraphDestroy(g0);
    }
    for (int j = 1; ok && j < steps; ++j) {
        cudaGraphNodeParams np{};
        np.type = cudaGraphNodeTypeConditional;
        np.conditional.handle = h[j];
        np.conditional.type = cudaGraphCondTypeIf;
        np.conditional.size = 1;
        cudaGraphNode_t node = nullptr;
        ok = cudaGraphAddNode(&node, top, &prev, nullptr, 1, &np) == cudaSuccess;
        if (!ok) break;
        cudaGraph_t body = np.conditional.phGraph_out[0];
        ok = cudaStreamBeginCaptureToGraph(cs_, body, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal) ==
             cudaSuccess;
        if (!ok) break;
        CondTail ct{};
        ct.h[0] = hv[j + 1];
        ct.n = 1;
        ctail_ = j + 1 < steps ? &ct : nullptr;
        ctail_done_ = false;
        bool rec = record_step(j, false, err);
        ctail_ = nullptr;
        if (rec && j + 1 < steps && !ctail_done_)
            rec = mtp_draft_cond(hv + j + 1, 1, probs_, row_ + 1, row_ + 3, cs_, m_prob_, j);
        cudaGraph_t captured = nullptr;   // the body itself, owned by the node
        ok = cudaStreamEndCapture(cs_, &captured) == cudaSuccess && rec;
        prev = node;
    }
    if (ok) ok = cudaGraphInstantiate(&exec, top, 0) == cudaSuccess;
    if (top) cudaGraphDestroy(top);
    if (ok) {
        cudaGraphUpload(exec, cs_);
        ok = cudaStreamSynchronize(cs_) == cudaSuccess;
    }
    if (!ok) {
        const cudaError_t e = cudaGetLastError();
        basalt::log::warn("basalt mtp", "the conditional draft graph failed (%s%s%s); one graph per "
                             "step from now on",
                     cudaGetErrorString(e), err.empty() ? "" : ", ", err.c_str());
        if (exec) cudaGraphExecDestroy(exec);
        exec = nullptr;
        cond_off_ = true;
        return false;
    }
    cond_steps_[tier_][T] = steps;
    return true;
}


}
