#include "common.hpp"
#include "basalt/core/load_progress.hpp"
#include "basalt/core/verify.hpp"
#include "basalt/kernels/pdl.hpp"
#include <algorithm>
#include <cmath>
#include <string>

namespace basalt::core {
using namespace verify_detail;

// Whether the zero-doorbell graph is right for the next window: only while every expert of [lb_, le_) is in VRAM
// now (the host table is the one the device copy was uploaded from).
void Verifier::refresh_ar() {
    if (!all_resident_ || hits_.h_res == nullptr || g_ == nullptr) return;
    const int64_t ne = g_->n_expert;
    const int32_t* r = hits_.h_res + lb_ * ne;
    const int64_t n = (le_ < 0 ? g_->n_layers : le_) * ne - lb_ * ne;
    int32_t lo = 0;
    for (int64_t i = 0; i < n; ++i) lo = std::min(lo, r[i]);
    const bool off = lo < 0;
    if (off == ar_off_) return;
    ar_off_ = off;
    basalt::log::info("basalt verify", "layers [%lld, %lld) %s", (long long) lb_,
                 (long long) (le_ < 0 ? g_->n_layers : le_),
                 off ? "have experts out of VRAM: the doorbell graph runs the windows"
                     : "are 100% VRAM resident again: the zero-doorbell graph");
}

bool Verifier::capture(int T, std::string& err) {
    cudaGraphExec_t& exec_t = ar_off_ ? exec_nr_[T] : exec_[T];
    if (exec_t != nullptr) return true;
    const bool pdl_try = basalt::kernels::pdl_requested() && !pdl_off_;
    // a window that does not capture or instantiate with programmatic dependent launch is captured again without it
    auto retry_plain = [&](const std::string& why) {
        basalt::log::warn("basalt verify", "the %d-token window failed with programmatic dependent launch (%s): capturing it without", T,
                     why.c_str());
        (void) cudaGetLastError();
        pdl_off_ = true;
        exec_t = nullptr;
        return capture(T, err);
    };
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "verify: begin capture failed";
        return false;
    }
    std::string rerr;
    pdl_rec_ = pdl_try;
    basalt::kernels::pdl_count() = 0;
    const bool ok = record_window(T, cs_, rerr);
    pdl_rec_ = false;
    const long pdl_n = basalt::kernels::pdl_count();
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs_, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        if (pdl_try) return retry_plain(rerr);
        err = rerr;
        return false;
    }
    if (ce != cudaSuccess) {
        if (pdl_try) return retry_plain(cudaGetErrorString(ce));
        err = std::string("verify: end capture: ") + cudaGetErrorString(ce);
        return false;
    }
    const cudaError_t ie = cudaGraphInstantiateWithFlags(&exec_t, graph, cudaGraphInstantiateFlagUseNodePriority);
    cudaGraphDestroy(graph);
    if (ie != cudaSuccess) {
        if (pdl_try) return retry_plain(cudaGetErrorString(ie));
        err = std::string("verify: instantiate: ") + cudaGetErrorString(ie);
        return false;
    }
    const cudaError_t ue = cudaGraphUpload(exec_t, cs_);
    const cudaError_t us = cudaStreamSynchronize(cs_);
    if (pdl_try)
        basalt::log::info("basalt verify", "captured the %d-token window (upload %s, sync %s), %ld programmatic "
                             "dependent launches", T, cudaGetErrorString(ue), cudaGetErrorString(us), pdl_n);
    else
        basalt::log::info("basalt verify", "captured the %d-token window (upload %s, sync %s)", T,
                     cudaGetErrorString(ue), cudaGetErrorString(us));
    return true;
}

}  // namespace basalt::core
