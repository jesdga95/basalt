#include "common.hpp"
#include "expert_plan.hpp"
#include "basalt/core/load_progress.hpp"
#include "basalt/core/verify.hpp"
#include "basalt/core/diag.hpp"
#include "basalt/core/remote_expert_opt.hpp"
#include "basalt/core/native_head.hpp"
#include "basalt/core/gdn_head_split.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/platform/compiler.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/core/progress.hpp"
#include "basalt/kernels/verify_kernels.hpp"
#include "basalt/kernels/dec_moe.hpp"
#include "basalt/log.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace basalt::core {
using namespace verify_detail;

bool Verifier::dm_init(std::string& err) {
    using namespace basalt::kernels;
    const ModelGeometry& g = *g_;
    const auto& lay = basalt::kernels::cpu::expert_layout();
    const char* why = nullptr;
    if (!lay.native) why = "the experts are not a native pack";
    else if (remote_opt_ != nullptr) why = "--remote-expert-opt";
    else if (all_resident_) why = "every expert is resident";
    else if (hits_.h_res == nullptr) why = "no host residency table";
    else if ((int64_t) max_t_ * ss_->k > kDmCap) why = "the window is too large";
    for (int64_t l = 0; why == nullptr && l < g.n_layers; ++l) {
        const auto& f = lay.fmt[(size_t) l];
        if (!dec_moe_formats_ok(f.gu_type, f.d_type, f.n_embd, f.n_ff)) why = "an expert format the kernels do not cover";
    }
    if (why != nullptr) {
        basalt::log::warn("basalt verify", "the device-planned MoE tail is off: %s", why);
        return true;
    }
    DecMoe* d = new DecMoe;
    dm_ = d;
    const int64_t LE = g.n_layers * g.n_expert, N = g.n_embd, T = max_t_, K = ss_->k;
    d->rings = le_ - lb_;
    d->x_stride = (size_t) (T * N);
    d->xq_stride = (size_t) (T * (N / 32) * 36);
    bool ok = cudaMalloc((void**) &d->res0, (size_t) LE * 4) == cudaSuccess &&
              cudaMalloc((void**) &d->res1, (size_t) LE * 4) == cudaSuccess &&
              cudaMalloc((void**) &d->plan, sizeof(DmPlan)) == cudaSuccess &&
              cudaMalloc((void**) &d->ctr, 64) == cudaSuccess && cudaMemset(d->ctr, 0, 64) == cudaSuccess &&
              cudaMemset(d->res1, 0xff, (size_t) LE * 4) == cudaSuccess &&
              mapped((size_t) d->rings * sizeof(DmRing), (void**) &d->h_ring, (void**) &d->m_ring) &&
              mapped((size_t) d->rings * d->x_stride * 4, (void**) &d->h_x, (void**) &d->m_x) &&
              mapped((size_t) d->rings * d->xq_stride, (void**) &d->h_xq, (void**) &d->m_xq) &&
              mapped(256, (void**) &d->h_w, (void**) &d->m_w) &&
              mapped((size_t) (T * K * N) * 4, (void**) &d->h_rows, (void**) &d->m_rows) &&
              cudaMalloc(&d->vscr, dec_moe_vram_scratch_bytes((int) lay.fmt[0].n_ff)) == cudaSuccess &&
              cudaMalloc((void**) &d->vsync, (kDmCap + 1) * 4) == cudaSuccess &&
              cudaMemset(d->vsync, 0, (kDmCap + 1) * 4) == cudaSuccess &&
              cudaMalloc(&d->xq_pub, d->xq_stride) == cudaSuccess &&
              cudaMalloc((void**) &d->pub, (size_t) d->rings * 16) == cudaSuccess &&
              cudaMemset(d->pub, 0, (size_t) d->rings * 16) == cudaSuccess &&
              cudaMalloc((void**) &d->cseen, 4) == cudaSuccess && cudaMemset(d->cseen, 0, 4) == cudaSuccess &&
              cudaMalloc((void**) &d->landed, (size_t) (d->rings + 1) * 4) == cudaSuccess &&
              cudaMemset(d->landed, 0, (size_t) (d->rings + 1) * 4) == cudaSuccess;
    if (ok && cudaHostAlloc((void**) &d->landed_pin, DecMoe::kPinSlots * 4, cudaHostAllocPortable) != cudaSuccess) {
        cudaGetLastError();
        d->landed_pin = nullptr;
    }
    d->vram = -1;
    if (ok && hits_.slot_off != nullptr && hits_.n_slots > 0)
        ok = cudaMalloc((void**) &d->slot_off, (size_t) hits_.n_slots * 8) == cudaSuccess &&
             cudaMemcpy(d->slot_off, hits_.slot_off, (size_t) hits_.n_slots * 8, cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok)
        ok = cudaMalloc((void**) &d->ptab0, (size_t) LE * 8) == cudaSuccess &&
             cudaMemset(d->ptab0, 0, (size_t) LE * 8) == cudaSuccess;
    if (ok && diagnostics().wait_stats)
        ok = cudaMalloc((void**) &d->wstat, kDmStatWords * 8) == cudaSuccess &&
             cudaMemset(d->wstat, 0, kDmStatWords * 8) == cudaSuccess;
    if (!ok) { err = "verify: device-planned MoE buffers"; return false; }
    d->rows_words = (size_t) (T * K * N);
    d->tags = d->wstat == nullptr;
    d->shadow0.assign((size_t) LE, INT32_MIN);
    d->shadow1.assign((size_t) LE, -1);
    d->pin_stride = (size_t) LE * 24;   // [res0 | ptab0 | res1 | ptab1] per slot
    if (cudaHostAlloc((void**) &d->pin, d->pin_stride * DecMoe::kPinSlots, cudaHostAllocPortable) != cudaSuccess) {
        cudaGetLastError();
        d->pin = nullptr;
    }
    // the warm-ups' predictors, quantized once: every layer's router in int8 for the peer's, in 4-bit for the CPU pool's
    const bool cpu_warm = warm_pool_ != nullptr && warm_src_ != nullptr && K <= 16 &&
                          std::all_of(lay.fmt.begin(), lay.fmt.end(), [&](const basalt::kernels::cpu::NativeFmt& f) {
                              return f.bytes == lay.fmt[0].bytes && f.down_off == lay.fmt[0].down_off;
                          });
    std::vector<int8_t> rq_h;
    std::vector<float> rs_h;
    std::vector<uint8_t> rq4_h;
    std::vector<float> rs4_h;
    const bool peer_on = peer_ != nullptr && peer_->valid();
    const bool codes_ok = (cpu_warm || peer_on) &&
                          router_codes(*wt_, g, peer_on ? &rq_h : nullptr, peer_on ? &rs_h : nullptr,
                                       cpu_warm ? &rq4_h : nullptr, cpu_warm ? &rs4_h : nullptr);
    const bool router_ok = peer_on && codes_ok;
    if (peer_ != nullptr && peer_->valid()) {
        const OnDevice on(peer_->device());
        d->pdev = peer_->device();
        std::vector<NativeExpertLayout> ls((size_t) g.n_layers);
        for (int64_t l = 0; l < g.n_layers; ++l) {
            const auto& f = lay.fmt[(size_t) l];
            ls[(size_t) l] = native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
            d->pq2 |= f.gu_type == 42;
        }
        d->pblocks = dec_moe_peer_blocks(d->pq2);
        ok = d->pblocks > 0 && cudaStreamCreateWithFlags(&d->ps, cudaStreamNonBlocking) == cudaSuccess &&
             basalt::kernels::dec_moe_table_load(d->ps) &&
             cudaMalloc((void**) &d->ptab, (size_t) LE * 8) == cudaSuccess &&
             cudaMemset(d->ptab, 0, (size_t) LE * 8) == cudaSuccess &&
             cudaMalloc((void**) &d->lay, ls.size() * sizeof(NativeExpertLayout)) == cudaSuccess &&
             cudaMemcpy(d->lay, ls.data(), ls.size() * sizeof(NativeExpertLayout), cudaMemcpyHostToDevice) == cudaSuccess &&
             cudaMalloc(&d->pscr, dec_moe_peer_scratch_bytes((int) N, (int) lay.fmt[0].n_ff)) == cudaSuccess &&
             cudaMemset(d->pscr, 0, dec_moe_peer_scratch_bytes((int) N, (int) lay.fmt[0].n_ff)) == cudaSuccess &&
             cudaMalloc((void**) &d->parr, 64) == cudaSuccess;
        d->tag = true;
        for (size_t i = 0; i < d->rows_words; ++i) ((volatile uint32_t*) d->h_rows)[i] = basalt::kernels::dec_moe_tag_word();
        if (!ok) { err = "verify: device-planned MoE peer buffers"; return false; }
        d->shadow1.assign((size_t) LE, INT32_MIN);
        if (!router_ok || !warm_router(rq_h, rs_h, &d->rq, &d->rs))
            basalt::log::warn("basalt verify", "the peer's L2 warm-up is off (a router the predictor does not cover)");
        // further peers: a window, ring and doorbell each; cards that hold several peers split their blocks
        std::vector<int> devs = {d->pdev};
        for (PeerExperts* xpe : xpeers_)
            if (xpe != nullptr && xpe->valid()) devs.push_back(xpe->device());
        auto sharing = [&](int dev) { return (int) std::count(devs.begin(), devs.end(), dev); };
        d->pblocks = std::max(1, d->pblocks / sharing(d->pdev));
        for (PeerExperts* xpe : xpeers_) {
            if (xpe == nullptr || !xpe->valid()) continue;
            if ((int) d->xp.size() == kDmMaxX) {
                basalt::log::warn("basalt verify", "the device plan serves %d further peers; the others' experts run on the CPU",
                                  kDmMaxX);
                break;
            }
            d->xp.emplace_back();
            DecMoe::XPeer& x = d->xp.back();
            x.peer = xpe;
            x.dev = xpe->device();
            {   // CUDA0 reads its residency
                const OnDevice on0(device_);
                ok = cudaMalloc((void**) &x.res, (size_t) LE * 4) == cudaSuccess &&
                     cudaMemset(x.res, 0xff, (size_t) LE * 4) == cudaSuccess &&
                     mapped((size_t) d->rings * sizeof(DmRing), (void**) &x.h_ring, (void**) &x.m_ring) &&
                     mapped(256, (void**) &x.h_w, (void**) &x.m_w);
            }
            {
                const OnDevice onx(x.dev);
                x.pblocks = std::max(1, dec_moe_peer_blocks(d->pq2) / sharing(x.dev));
                ok = ok && cudaStreamCreateWithFlags(&x.ps, cudaStreamNonBlocking) == cudaSuccess &&
                     basalt::kernels::dec_moe_table_load(x.ps) &&
                     cudaMalloc((void**) &x.ptab, (size_t) LE * 8) == cudaSuccess &&
                     cudaMemset(x.ptab, 0, (size_t) LE * 8) == cudaSuccess &&
                     cudaMalloc((void**) &x.lay, ls.size() * sizeof(NativeExpertLayout)) == cudaSuccess &&
                     cudaMemcpy(x.lay, ls.data(), ls.size() * sizeof(NativeExpertLayout), cudaMemcpyHostToDevice) == cudaSuccess &&
                     cudaMalloc(&x.pscr, dec_moe_peer_scratch_bytes((int) N, (int) lay.fmt[0].n_ff)) == cudaSuccess &&
                     cudaMemset(x.pscr, 0, dec_moe_peer_scratch_bytes((int) N, (int) lay.fmt[0].n_ff)) == cudaSuccess &&
                     cudaMalloc((void**) &x.parr, 64) == cudaSuccess;
                if (ok && router_ok && d->rq != nullptr) (void) warm_router(rq_h, rs_h, &x.rq, &x.rs);
            }
            x.shadow.assign((size_t) LE, INT32_MIN);
            if (!ok) { err = "verify: device-planned MoE buffers of a further peer"; return false; }
        }
    }
    if (d->pdev >= 0) {
        hs_init();
        gs_init();
    }
    if (cpu_warm && codes_ok) {
        CpuWarm* w = new CpuWarm;
        w->rq = std::move(rq4_h);
        w->rs = std::move(rs4_h);
        w->n_expert = (int) g.n_expert; w->n_embd = (int) N; w->k = (int) K; w->src = warm_src_;
        w->bytes = lay.fmt[0].bytes;
        w->down_off = lay.fmt[0].down_off;
        w->logits.assign((size_t) kVerifyMaxT * g.n_expert, 0.0f);
        d->warm = w;
    }
    // The staged share: on one card, a batch window's layer with enough CPU misses sends some of them to CUDA0, copied
    // over PCIe from their RAM copies into staging while the pool reads the rest.  An expert without a mapped RAM copy
    // is never staged; the addresses are read again before a batch window when the source's places moved (a swap).
    // The count staged for m misses balances the pool's time for the rest against the copy's, both reading host
    // memory at once.
    bool stage = d->pdev < 0 && d->xp.empty() && lb_ == 0 && le_ == g.n_layers && hand_in_ == nullptr &&
                 warm_src_ != nullptr && staging_ != nullptr && warm_src_->pcie_layer(0) && lay.fmt[0].bytes % 16 == 0;
    for (int64_t l = 1; stage && l < g.n_layers; ++l) stage = lay.fmt[(size_t) l].bytes == lay.fmt[0].bytes;
    const int64_t blob = (int64_t) lay.fmt[0].bytes, stride = dec_moe_stage_stride(blob);
    const int stage_slots =
        (int) std::min<int64_t>(kDmStageMax, (int64_t) (kStagingBlobs * (int64_t) lay.max_blob) / stride);
    if (stage && stage_slots >= 1) {
        // per layer: the pool's experts after its doorbell, the staged ones after the copy, their own launches and
        // the combine's wait for them
        constexpr double kPoolBytesPerUs = 45e3, kLinkBytesPerUs = 18e3, kPoolFixUs = 15.0, kStageFixUs = 35.0;
        const double cpu_us = (double) blob / kPoolBytesPerUs, link_us = (double) blob / kLinkBytesPerUs;
        std::vector<int32_t> kt((size_t) kDmCap + 1, 0);
        for (int m = 1; m <= kDmCap; ++m) {
            double best = kPoolFixUs + m * cpu_us;
            for (int k = 1; k <= std::min(m, stage_slots); ++k) {
                const double t = std::max(kPoolFixUs + (m - k) * cpu_us, kStageFixUs + k * link_us);
                if (t < best) { best = t; kt[(size_t) m] = k; }
            }
        }
        int st_least = 0, st_greatest = 0;   // the copy's blocks go first as the FFN read's leave the SMs
        bool sok = cudaMalloc((void**) &d->stage_src, (size_t) LE * 8) == cudaSuccess &&
                   cudaMemset(d->stage_src, 0, (size_t) LE * 8) == cudaSuccess &&
                   cudaMallocHost((void**) &d->stage_host, (size_t) LE * 8) == cudaSuccess &&
                   cudaMalloc((void**) &d->stage_k, kt.size() * 4) == cudaSuccess &&
                   cudaMemcpy(d->stage_k, kt.data(), kt.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess &&
                   cudaDeviceGetStreamPriorityRange(&st_least, &st_greatest) == cudaSuccess &&
                   cudaStreamCreateWithPriority(&d->st_s, cudaStreamNonBlocking, st_greatest) == cudaSuccess &&
                   cudaEventCreateWithFlags(&d->st_fork, cudaEventDisableTiming) == cudaSuccess &&
                   cudaEventCreateWithFlags(&d->st_join, cudaEventDisableTiming) == cudaSuccess;
        if (!sok) { err = "verify: the staged share's buffers"; return false; }
        std::memset(d->stage_host, 0, (size_t) LE * 8);
        d->stage_gen = warm_src_->places_generation() + 1;   // read at the first batch window
        d->stage_stride = stride;
        d->stage_blocks = 8;
        d->stage_min = 1;
        while (d->stage_min < kDmCap && kt[(size_t) d->stage_min] == 0) ++d->stage_min;
    }
    std::string served;
    if (d->pdev >= 0) {
        served = ", the peer card served by a persistent kernel (" + std::to_string(d->pblocks) + " blocks";
        for (const DecMoe::XPeer& x : d->xp) served += ", then " + std::to_string(x.pblocks) + " on CUDA" + std::to_string(x.dev);
        served += ")";
    }
    basalt::log::info("basalt verify", "the MoE tail planned on the device%s", served.c_str());
    if (d->warm != nullptr)
        basalt::log::info("basalt verify", "the CPU pool warms its likeliest next-layer expert between doorbells");
    return true;
}

// A host-to-device upload in pieces of at most kSmallCopy bytes: a copy that small passes the expert copies queued on
// the card's copy engine (an adaptive round's), a larger one waits behind all of them.

void Verifier::dm_free() {
    DecMoe* d = dm_;
    if (d == nullptr) return;
    hs_free();
    gs_free();
    if (d->warm != nullptr) {
        warm_pool_->idle_stop();
        delete d->warm;
    }
    if (d->pdev >= 0) {
        const OnDevice on(d->pdev);
        if (d->ps) { cudaStreamSynchronize(d->ps); cudaStreamDestroy(d->ps); }
        for (void* p : {(void*) d->ptab, (void*) d->lay, d->pscr, (void*) d->parr, (void*) d->rq, (void*) d->rs})
            if (p) cudaFree(p);
    }
    for (DecMoe::XPeer& x : d->xp) {
        {
            const OnDevice on(x.dev);
            if (x.ps) { cudaStreamSynchronize(x.ps); cudaStreamDestroy(x.ps); }
            for (void* p : {(void*) x.ptab, (void*) x.lay, x.pscr, (void*) x.parr, (void*) x.rq, (void*) x.rs})
                if (p) cudaFree(p);
        }
        if (x.res) cudaFree(x.res);
        for (void* h : {(void*) x.h_ring, (void*) x.h_w})
            if (h) cudaFreeHost(h);
    }
    if (d->st_s) { cudaStreamSynchronize(d->st_s); cudaStreamDestroy(d->st_s); }
    for (cudaEvent_t e : {d->st_fork, d->st_join})
        if (e) cudaEventDestroy(e);
    for (void* p : {(void*) d->res0, (void*) d->res1, (void*) d->plan, (void*) d->ctr, (void*) d->slot_off, d->vscr,
                    (void*) d->vsync, d->xq_pub, (void*) d->wstat,
                    (void*) d->ptab0, (void*) d->pub, (void*) d->cseen, (void*) d->landed,
                    (void*) d->stage_src, (void*) d->stage_k})
        if (p) cudaFree(p);
    if (d->stage_host) cudaFreeHost(d->stage_host);
    if (d->pin) cudaFreeHost(d->pin);
    if (d->landed_pin) cudaFreeHost(d->landed_pin);
    for (void* h : {(void*) d->h_ring, (void*) d->h_x, (void*) d->h_xq, (void*) d->h_w, (void*) d->h_rows})
        if (h) cudaFreeHost(h);
    delete d;
    dm_ = nullptr;
}

}  // namespace basalt::core
