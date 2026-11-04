#include "common.hpp"
#include "expert_plan.hpp"
#include "basalt/core/verify.hpp"
#include "basalt/core/native_head.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/kernels/cvec.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/q5k_head.hpp"
#include "basalt/kernels/q80_vhead.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <exception>
#include <string>

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

// The two-stage head's split on a slower peer: each card screens its rows' top six bits at its own bandwidth, and the
// peer's share also pays its input's and its list's trips over the host, so it takes a little less than its bandwidth's
// share.  CUDA0's rows (0: no split).
static int hs_plan_two(int V, int dev0, int dev1) {
    const double bw0 = card_bandwidth(dev0), bw1 = card_bandwidth(dev1);
    if (bw0 <= 0.0 || bw1 <= 0.0 || bw1 >= bw0) return 0;
    constexpr double screen_row = 2080.0, hand_off = 20e-6;   // the bytes a row's screen reads; the trips and the list
    const double c0 = screen_row / bw0, c1 = screen_row / bw1;
    const double r1 = ((double) V * c0 - hand_off) / (c0 + c1);   // r1 c1 + hand_off = (V - r1) c0
    const int rows = (int) (((double) V - r1) / 256.0) * 256;
    return rows > 0 && V - rows >= 4096 ? rows : 0;
}

bool Verifier::hs_two_stage(int type, int64_t n_vocab, int64_t n_embd, int max_t, int dev0, int dev1) {
    return dev1 >= 0 && dev1 != dev0 && type == 8 && basalt::kernels::q80p_supported((int) n_embd, std::min(max_t, 8)) &&
           hs_plan_two((int) n_vocab, dev0, dev1) > 0;
}

// the peer's copy of the input: the q8_1 columns, then the list length
static size_t hs_two_input_bytes(int N, int T) { return basalt::kernels::q80p_vhead_input_bytes(N, T); }

int64_t Verifier::hs_peer_bytes(size_t row_bytes, int type, int64_t n_vocab, int64_t n_embd, int max_t, int dev0, int dev1) {
    if (dev1 < 0 || dev1 == dev0) return 0;
    if (hs_two_stage(type, n_vocab, n_embd, max_t, dev0, dev1)) {
        const int64_t r1 = n_vocab - hs_plan_two((int) n_vocab, dev0, dev1);
        return r1 * (int64_t) row_bytes + (int64_t) hs_two_input_bytes((int) n_embd, max_t) + (int64_t) max_t * r1 * 4 +
               (int64_t) basalt::kernels::q80p_vhead_scratch_bytes();
    }
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
    if (head_->planes() && head_keep_d_ != nullptr && le_ >= g_->n_layers &&
        hs_two_stage(head_->type(), V, N, T, device_, dm_->pdev)) {   // the two-stage head's split
        const int rows = hs_plan_two(V, device_, dm_->pdev), r1 = V - rows;
        const size_t xk = hs_two_input_bytes(N, T), lb = basalt::kernels::q80p_vhead_list_bytes(8, r1);
        bool ok;
        {
            const OnDevice on(dm_->pdev);
            ok = cudaMalloc(&hs_w_, (size_t) r1 * rb) == cudaSuccess && cudaMalloc(&hs_xq_, xk) == cudaSuccess &&
                 cudaMalloc((void**) &hs_y_, (size_t) T * r1 * 4) == cudaSuccess &&
                 cudaMalloc(&hs_scr_, basalt::kernels::q80p_vhead_scratch_bytes()) == cudaSuccess &&
                 cudaMemset(hs_scr_, 0, basalt::kernels::q80p_vhead_scratch_bytes()) == cudaSuccess;
            ok = ok && cudaMemcpyPeer(hs_w_, dm_->pdev, (const uint8_t*) head_->weights() + (size_t) rows * rb, device_,
                                      (size_t) r1 * rb) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
            if (ok) basalt::kernels::q80p_vhead_prepare();   // launched behind the peer's persistent window later
        }
        {
            const OnDevice on(device_);
            void* d = nullptr;
            ok = ok && cudaEventCreateWithFlags(&hs_fork_, cudaEventDisableTiming) == cudaSuccess &&
                 cudaEventCreateWithFlags(&hs_post_, cudaEventDisableTiming) == cudaSuccess &&
                 cudaStreamCreateWithFlags(&hs_cs_, cudaStreamNonBlocking) == cudaSuccess &&
                 mapped(basalt::kernels::q80p_vhead_link_bytes(), &hs_link_, &d) &&
                 mapped(basalt::kernels::q80p_vhead_input_bytes(N, T), &hs_input_, &d) && mapped(lb, &hs_list_, &d);
        }
        if (!ok) {
            cudaGetLastError();
            hs_free();
            basalt::log::warn("basalt verify", "the head stays whole on CUDA0: no room for the peer's share");
            return;
        }
        hs_rows_ = rows;
        hs_dev_ = dm_->pdev;
        hs_two_ = true;
        hs_seq_ = 0;
        hs_k_ = (int*) ((uint8_t*) hs_xq_ + (xk - 16));
        {   // one pass of the peer's part at load (window 0, list length 0, met at once by the zeroed link): its first
            // launches on the persistent window's stream are not paid by a request
            const OnDevice on(dm_->pdev);
            basalt::kernels::q80p_vhead_fetch(hs_input_, hs_link_, hs_xq_, hs_k_, N, T, hs_scr_, dm_->ps);
            basalt::kernels::q80p_vhead(hs_w_, hs_xq_, hs_y_, N, r1, T, hs_k_, hs_scr_, dm_->ps, -1, nullptr,
                                        (const unsigned*) hs_link_);
            basalt::kernels::q80p_vhead_list(hs_y_, r1, T, hs_k_, hs_scr_, hs_list_, hs_link_, dm_->ps);
            if (cudaStreamSynchronize(dm_->ps) != cudaSuccess) {
                cudaGetLastError();
                hs_free();
                basalt::log::warn("basalt verify", "the head stays whole on CUDA0: the peer's share did not run");
                return;
            }
        }
        basalt::log::info("basalt verify", "the head is split by vocabulary: CUDA0 screens %d rows, the peer %d", rows, r1);
        return;
    }
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
        if (dm_ != nullptr && dm_->ps != nullptr) cudaStreamSynchronize(dm_->ps);   // the peer's share of a window
        if (hs_out_) cudaEventDestroy(hs_out_);
        for (void* p : {hs_w_, hs_xq_, (void*) hs_y_, hs_scr_}) if (p) cudaFree(p);
    }
    {
        const OnDevice on(device_);
        if (hs_cs_) { cudaStreamSynchronize(hs_cs_); cudaStreamDestroy(hs_cs_); }
        if (hs_y0_) cudaFree(hs_y0_);
        if (hs_in_) cudaEventDestroy(hs_in_);
        if (hs_fork_) cudaEventDestroy(hs_fork_);
        if (hs_post_) cudaEventDestroy(hs_post_);
        for (void* p : {hs_hxq_, (void*) hs_hy_, hs_link_, hs_input_, hs_list_}) if (p) cudaFreeHost(p);
    }
    hs_rows_ = 0; hs_dev_ = -1;
    hs_two_ = false;
    hs_w_ = hs_xq_ = hs_hxq_ = hs_scr_ = hs_link_ = hs_input_ = hs_list_ = nullptr;
    hs_y_ = hs_y0_ = hs_hy_ = nullptr;
    hs_k_ = nullptr;
    hs_s_ = hs_cs_ = nullptr;
    hs_in_ = hs_out_ = hs_fork_ = hs_post_ = nullptr;
}

bool Verifier::hs_launch(int T, std::string& err) {
    if (!hs_two_ || !head_screened_ || dm_ == nullptr || dm_->ps == nullptr) return true;
    if (++hs_seq_ == 0) hs_seq_ = 1;   // never 0: the link starts zeroed
    *(volatile uint32_t*) hs_link_ = hs_seq_;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const OnDevice on(dm_->pdev);
    const int N = (int) g_->n_embd, r1 = (int) n_vocab_ - hs_rows_;
    try {
        basalt::kernels::q80p_vhead_fetch(hs_input_, hs_link_, hs_xq_, hs_k_, N, T, hs_scr_, dm_->ps);
        basalt::kernels::q80p_vhead(hs_w_, hs_xq_, hs_y_, N, r1, T, hs_k_, hs_scr_, dm_->ps, -1, nullptr,
                                    (const unsigned*) hs_link_);
        basalt::kernels::q80p_vhead_list(hs_y_, r1, T, hs_k_, hs_scr_, hs_list_, hs_link_, dm_->ps);
    } catch (const std::exception& e) {
        err = std::string("verify: the peer's share of the head: ") + e.what();
        return false;
    }
    if (cudaPeekAtLastError() != cudaSuccess) {
        err = std::string("verify: the peer's share of the head: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    return true;
}

}  // namespace basalt::core
