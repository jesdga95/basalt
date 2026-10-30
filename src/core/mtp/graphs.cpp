#include "common.hpp"

namespace basalt::core {

using namespace mtp_detail;

namespace {
bool finish_capture(cudaStream_t cs, bool ok, cudaGraphExec_t& exec, const char* what, std::string& err) {
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        return false;
    }
    if (ce != cudaSuccess || cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("mtp: ") + what + " capture: " + cudaGetErrorString(ce);
        return false;
    }
    cudaGraphDestroy(graph);
    // an explicit upload: the first launch's implicit one blocked behind a device-side spin (src/core/verify/)
    cudaGraphUpload(exec, cs);
    cudaStreamSynchronize(cs);
    return true;
}
}  // namespace

bool MtpDrafter::capture_prefill(int T, std::string& err) {
    if (prefill_exec_[T]) return true;
    using namespace basalt::kernels;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "mtp: begin capture"; return false; }
    copy_i32_from_mapped(tok_, m_tok_, T, cs_);
    copy_i32_from_mapped(step_, m_step_, (int64_t) T * 4, cs_);
    copy_i32_from_mapped(pos_, m_pos_, (int64_t) T * g_->n_head, cs_);
    const bool ok = record_forward(T, -1, cs_, err);   // K/V only, rows [0, T)
    return finish_capture(cs_, ok, prefill_exec_[T], "prefill", err);
}

bool MtpDrafter::capture_prefill_dev(int T, std::string& err) {
    if (prefill_dev_exec_[T]) return true;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "mtp: begin capture"; return false; }
    const bool ok = record_forward(T, -1, cs_, err);   // K/V only, rows [0, T); tok_/step_/pos_ filled before launch
    return finish_capture(cs_, ok, prefill_dev_exec_[T], "prefill (device inputs)", err);
}

// The round: the catch-up's K/V for the window's T cells, then the full layer for row a only (its cell's K/V is written
// again, identically, unless the row is taken from the catch-up), staged by the host in step row
// 2*max_t - 1; draft 0 and its probability to the mapped outputs.  The draft chain is one graph per step
// (`capture_step`) so the host can stop it when a draft is unlikely, or conditional nodes (`capture_cond`).
bool MtpDrafter::record_round(int T, bool all_rows, std::string& err) {
    using namespace basalt::kernels;
    const int64_t HCN = g_->hc * g_->n_embd, NH = g_->n_head;
    const int64_t rows = all_rows ? 2 * (int64_t) max_t_ : 2 * (int64_t) T;
    const int ra = 2 * max_t_ - 1;
    {   // the catch-up reads rows below 2T only: the draft row's records are staged with the rest
        MtpStageSeg sg[6] = {{tok_, m_tok_, T}, {step_, m_step_, (int) (rows * 4)}, {pos_, m_pos_, (int) (rows * NH)},
                             {row_, m_row_, all_rows ? 4 : 2}, {}, {}};
        int ns = 4;
        if (!all_rows) {
            sg[ns++] = {step_ + ra * 4, m_step_ + ra * 4, 4};
            sg[ns++] = {pos_ + ra * NH, m_pos_ + ra * NH, (int) NH};
        }
        mtp_stage(sg, ns, Rin_, window_R_, (int64_t) T * HCN, cs_);
    }
    if (gstamp_ != nullptr) gpu_stamp(gstamp_, 0, cs_);
    par_ = true;
    bool ok = record_forward(T, -1, cs_, err);
    if (ok) {
        chain_j_ = 0;
        pre_next_ = par_;
        ok = record_forward(1, ra, cs_, err, true);
    }
    end_branches();
    if (ok) ok = step_end(0);
    if (ok && gstamp_ != nullptr) gpu_stamp(gstamp_, 1, cs_);
    return ok;
}

// Chain step j (1..max_t-2): one row at the cell staged in step row `max_t + j - 1`, from the previous step's
// residual and token (left in tok_[0] by the previous step); draft j and its probability to the mapped outputs.
bool MtpDrafter::record_step(int j, bool copy_records, std::string& err) {
    using namespace basalt::kernels;
    const int64_t NH = g_->n_head;
    const int row = max_t_ + j - 1;
    if (copy_records) {
        copy_i32_from_mapped(step_ + row * 4, m_step_ + row * 4, 4, cs_);
        copy_i32_from_mapped(pos_ + row * NH, m_pos_ + row * NH, NH, cs_);
    }
    chain_j_ = j;
    par_ = true;
    have_pre_ = pre_next_ = par_;   // every full row hands the next its hidden input
    bool ok = record_forward(1, row, cs_, err);
    end_branches();
    if (ok) ok = step_end(j);
    if (ok && gstamp_ != nullptr) gpu_stamp(gstamp_, 1, cs_);
    return ok;
}

bool MtpDrafter::capture_round(int T, std::string& err) {
    cudaGraphExec_t& exec = round_exec_[tier_][T];
    if (exec) return true;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "mtp: begin capture"; return false; }
    const bool ok = record_round(T, false, err);
    return finish_capture(cs_, ok, exec, "round", err);
}

bool MtpDrafter::capture_step(int j, std::string& err) {
    cudaGraphExec_t& exec = step_exec_[tier_][j];
    if (exec) return true;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "mtp: begin capture"; return false; }
    const bool ok = record_step(j, true, err);
    return finish_capture(cs_, ok, exec, "step", err);
}


}
