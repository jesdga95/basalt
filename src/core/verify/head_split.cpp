#include "common.hpp"
#include "expert_plan.hpp"
#include "basalt/core/verify.hpp"
#include "basalt/core/native_head.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/kernels/cvec.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/q5k_head.hpp"
#include <cmath>

namespace basalt::core {
using namespace verify_detail;

// The head's rows are split by the time each card takes for its share: CUDA0 streams its rows, the peer streams
// its rows and its logits cross PCIe twice (to the host, to CUDA0).  A split that leaves CUDA0 most of the rows
// does not pay for its two hand-offs, so it is taken only when the peer's share is large.
// CUDA0's rows of the head split by vocabulary (0: no split), from the cards' bandwidths.
static int hs_plan(size_t rb, int type, int V, int N, int T, int dev0, int dev1) {
    if (type == 13 ? !basalt::kernels::q5k_head_supported(N, T) : type == 14) return 0;
    const double bw0 = card_bandwidth(dev0), bw1 = card_bandwidth(dev1);
    if (bw0 <= 0.0 || bw1 <= 0.0 || rb == 0) return 0;
    const double pcie = 25e9, cols = 3.0;   // the hand-off rate and a typical window
    const double c0 = (double) rb / bw0, c1 = (double) rb / bw1 + 2.0 * cols * 4.0 / pcie;
    const double share0 = c1 / (c0 + c1);
    if (share0 > 0.75) return 0;
    const int rows = (int) (share0 * V) / 256 * 256;
    return rows <= 0 || rows >= V ? 0 : rows;
}

int64_t Verifier::hs_peer_bytes(size_t row_bytes, int type, int64_t n_vocab, int64_t n_embd, int max_t, int dev0, int dev1) {
    if (dev1 < 0 || dev1 == dev0) return 0;
    const int rows = hs_plan(row_bytes, type, (int) n_vocab, (int) n_embd, max_t, dev0, dev1);
    if (rows == 0) return 0;
    const int64_t r1 = n_vocab - rows;
    return r1 * (int64_t) row_bytes + (int64_t) basalt::kernels::native_q8_1_bytes((int) n_embd, max_t) +
           (int64_t) max_t * r1 * 4;
}

void Verifier::hs_init() {
    if (head_ == nullptr || !head_->loaded() || head_->vproxy() != nullptr || dm_ == nullptr || dm_->pdev < 0) return;
    const int N = (int) g_->n_embd, V = (int) n_vocab_, T = max_t_;
    const size_t rb = head_->row_bytes();
    const int rows = hs_plan(rb, head_->type(), V, N, T, device_, dm_->pdev);
    if (rows == 0) return;
    const int r1 = V - rows;
    const size_t xb = basalt::kernels::native_q8_1_bytes(N, T);
    bool ok;
    {
        const OnDevice on(dm_->pdev);
        ok = cudaMalloc(&hs_w_, (size_t) r1 * rb) == cudaSuccess && cudaMalloc(&hs_xq_, xb) == cudaSuccess &&
             cudaMalloc((void**) &hs_y_, (size_t) T * r1 * 4) == cudaSuccess &&
             cudaStreamCreateWithFlags(&hs_s_, cudaStreamNonBlocking) == cudaSuccess &&
             cudaEventCreateWithFlags(&hs_out_, cudaEventDisableTiming) == cudaSuccess;
        ok = ok && cudaMemcpyPeer(hs_w_, dm_->pdev, (const uint8_t*) head_->weights() + (size_t) rows * rb, device_,
                                  (size_t) r1 * rb) == cudaSuccess;
    }
    {
        const OnDevice on(device_);
        ok = ok && cudaMalloc((void**) &hs_y0_, (size_t) T * rows * 4) == cudaSuccess &&
             cudaEventCreateWithFlags(&hs_in_, cudaEventDisableTiming) == cudaSuccess &&
             cudaHostAlloc(&hs_hxq_, xb, cudaHostAllocPortable) == cudaSuccess &&
             cudaHostAlloc((void**) &hs_hy_, (size_t) T * r1 * 4, cudaHostAllocPortable) == cudaSuccess;
    }
    if (!ok) {
        cudaGetLastError();
        hs_free();
        basalt::log::warn("basalt verify", "the head stays whole on CUDA0: no room for the peer's share");
        return;
    }
    hs_rows_ = rows;
    hs_dev_ = dm_->pdev;
    basalt::log::info("basalt verify", "the head is split by vocabulary: CUDA0 %d rows, the peer %d rows", rows, r1);
}

void Verifier::hs_free() {
    if (hs_dev_ >= 0 || hs_w_ != nullptr || hs_s_ != nullptr) {
        const OnDevice on(dm_ != nullptr && dm_->pdev >= 0 ? dm_->pdev : hs_dev_);
        if (hs_s_) { cudaStreamSynchronize(hs_s_); cudaStreamDestroy(hs_s_); }
        if (hs_out_) cudaEventDestroy(hs_out_);
        for (void* p : {hs_w_, hs_xq_, (void*) hs_y_}) if (p) cudaFree(p);
    }
    {
        const OnDevice on(device_);
        if (hs_y0_) cudaFree(hs_y0_);
        if (hs_in_) cudaEventDestroy(hs_in_);
        if (hs_hxq_) cudaFreeHost(hs_hxq_);
        if (hs_hy_) cudaFreeHost(hs_hy_);
    }
    hs_rows_ = 0; hs_dev_ = -1;
    hs_w_ = hs_xq_ = hs_hxq_ = nullptr;
    hs_y_ = hs_y0_ = hs_hy_ = nullptr;
    hs_s_ = nullptr;
    hs_in_ = hs_out_ = nullptr;
}

}  // namespace basalt::core
