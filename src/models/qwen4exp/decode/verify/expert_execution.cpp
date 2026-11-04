#include "common.hpp"
#include "expert_plan.hpp"
#include "basalt/core/verify.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/platform/compiler.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/dec_moe.hpp"
#include <atomic>
#include <string>

namespace basalt::core {
using namespace verify_detail;

// The device-planned tail's window start: residency tables, the flag words, and the peer card's persistent window.
bool Verifier::dm_begin(std::string& err, DecodeGaps* gm) {
    if (!dm_sync(err)) return false;
    decode_events().stamp(4);
    if (gm) gm->mark(13);
    *(volatile uint32_t*) (dm_->h_w + 0) = 0;
    *(volatile uint32_t*) (dm_->h_w + 16) = 0;
    *(volatile uint32_t*) (dm_->h_w + 32) = 0;
    *(volatile uint32_t*) (dm_->h_w + 48) = 0;
    *(volatile uint32_t*) (dm_->h_w + 52) = 0;
    *(volatile uint32_t*) (dm_->h_w + 60) = 0;
    if (dm_->tag && dm_->tag_dirty) {   // a released window left rows written: back to unwritten
        for (size_t i = 0; i < dm_->rows_words; ++i)
            ((volatile uint32_t*) dm_->h_rows)[i] = basalt::kernels::dec_moe_tag_word();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        dm_->tag_dirty = false;
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (dm_->pdev >= 0) {
        const OnDevice on(dm_->pdev);
        basalt::kernels::DmPeerArgs pa;
        pa.rings_m = dm_->h_ring; pa.xq_m = dm_->h_xq; pa.xq_stride = dm_->xq_stride;
        pa.seq_m = dm_->h_w + 0; pa.done_m = dm_->h_w + 16; pa.rows_m = dm_->h_rows;
        pa.ptr = dm_->ptab; pa.lay = dm_->lay; pa.n_expert = (int) g_->n_expert; pa.k = (int) ss_->k;
        pa.n_embd = (int) g_->n_embd; pa.n_ff = (int) basalt::kernels::cpu::expert_layout().fmt[0].n_ff;
        pa.steps = (int) dm_->rings; pa.scratch = dm_->pscr; pa.arrive = dm_->parr; pa.diag_m = dm_->h_w + 32;
        pa.rq = dm_->rq; pa.rs = dm_->rs;
        if (gs_nk0_ > 0) {
            pa.mix = gs_mix_; pa.mseq_m = dm_->h_w + 48; pa.mdone_m = dm_->h_w + 52;
            pa.mxq_m = gs_h_xq_; pa.mxq = gs_pxq_;
            pa.mx_m = gs_h_x_; pa.mixl = gs_mixl_; pa.part_m = gs_h_part_;
        }
        if (!basalt::kernels::dec_moe_zero_words(dm_->parr, 16, dm_->ps) ||
            !basalt::kernels::dec_moe_peer_launch(pa, dm_->pblocks, dm_->ps, dm_->pq2)) {
            err = "verify: the peer window did not launch";
            return false;
        }
        dm_->peer_live = true;
    }
    for (DecMoe::XPeer& x : dm_->xp) {   // further peers: the same window on their own ring
        for (int w : {0, 16, 32}) *(volatile uint32_t*) (x.h_w + w) = 0;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        const OnDevice on(x.dev);
        basalt::kernels::DmPeerArgs pa;
        pa.rings_m = x.h_ring; pa.xq_m = dm_->h_xq; pa.xq_stride = dm_->xq_stride;
        pa.seq_m = x.h_w + 0; pa.done_m = x.h_w + 16; pa.rows_m = dm_->h_rows;
        pa.ptr = x.ptab; pa.lay = x.lay; pa.n_expert = (int) g_->n_expert; pa.k = (int) ss_->k;
        pa.n_embd = (int) g_->n_embd; pa.n_ff = (int) basalt::kernels::cpu::expert_layout().fmt[0].n_ff;
        pa.steps = (int) dm_->rings; pa.scratch = x.pscr; pa.arrive = x.parr; pa.diag_m = x.h_w + 32;
        pa.rq = x.rq; pa.rs = x.rs;
        if (!basalt::kernels::dec_moe_zero_words(x.parr, 16, x.ps) ||
            !basalt::kernels::dec_moe_peer_launch(pa, x.pblocks, x.ps, dm_->pq2)) {
            err = "verify: a further peer's window did not launch";
            return false;
        }
        x.live = true;
    }
    if (gm) gm->mark(14);
    return true;
}

// The peer window ends with CUDA0's last ring.
bool Verifier::dm_end(std::string& err) {
    if (dm_ == nullptr) return true;
    for (DecMoe::XPeer& x : dm_->xp) {
        if (!x.live) continue;
        const OnDevice on(x.dev);
        x.live = false;
        const cudaError_t pe = cudaStreamSynchronize(x.ps);
        if (pe != cudaSuccess) { err = std::string("verify: a further peer's window: ") + cudaGetErrorString(pe); return false; }
        const std::string ab = basalt::kernels::dec_moe_peer_diag(x.h_w + 32);
        if (!ab.empty()) { err = "verify: a further peer's window " + ab; return false; }
    }
    if (!dm_->peer_live) return true;
    const OnDevice on(dm_->pdev);
    dm_->peer_live = false;
    const cudaError_t pe = cudaStreamSynchronize(dm_->ps);
    if (pe != cudaSuccess) { err = std::string("verify: the peer window: ") + cudaGetErrorString(pe); return false; }
    const std::string ab = basalt::kernels::dec_moe_peer_diag(dm_->h_w + 32);
    if (!ab.empty()) { err = "verify: the peer window " + ab; return false; }
    return true;
}

}  // namespace basalt::core
