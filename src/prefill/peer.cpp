#include "state.hpp"
#include "layout.hpp"
#include "basalt/prefill/prefill.hpp"
#include "basalt/core/mtp.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/kernels/ngram.hpp"
#include "basalt/core/native_head.hpp"
#include "basalt/prefill/moe_mmq.hpp"
#include "basalt/prefill/moe_fp4.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/platform/memory.hpp"
#include "basalt/log.hpp"
#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <memory>

namespace basalt::prefill {
using namespace detail;

void Prefill::set_ring_peer(core::PeerExperts* peer) {
    impl_->ring_peer = peer != nullptr && peer->valid() ? peer : nullptr;
}

void Prefill::set_ring_remote(const int32_t* res, const core::ExpertCache* cache, int device) {
    const bool on = res != nullptr && cache != nullptr && device >= 0;
    impl_->remote_res = on ? res : nullptr;
    impl_->remote_cache = on ? cache : nullptr;
    impl_->remote_dev = on ? device : -1;
}

void Prefill::set_peer_share(bool on) { impl_->peer_share = on; }
int64_t Prefill::peer_routed_rows_max() const { return impl_->pp ? impl_->pp->T_max : 0; }

void Prefill::set_stage(int64_t layer_begin, int64_t layer_end, Prefill* next) {
    if (layer_end < 0 && impl_->g != nullptr) layer_end = impl_->g->n_layers;   // after init: resolved here
    stage_lb_ = layer_begin;
    stage_le_ = layer_end;
    next_ = next;
    if (layer_begin == 0) {   // the first stage reads its tokens, not rows a stage before it left
        hand_in_ = nullptr;
        rows_in_place_ = false;
    }
}

// The peer's prompt share for chunks of up to `T_max` tokens (set_peer); `tf`: PeerPrefill::T_fp4, 0 for none.
// The peer card's prompt buffers for chunks of up to `Tp` tokens (`tf` > 0: with the FP4 share's), each from `take(bytes)`,
// in one place so that the tier's reserve and the allocation count the same bytes.
template <class Take>
static void peer_layout(PeerPrefill& pp, int64_t Tp, int64_t tf, const MmqPlan& mp, Take&& take) {
    pp.mixed = (float*) take((size_t) Tp * N * 4);
    pp.cap_rows = Tp * K;         // no row cap: only the row tables grow with it
    pp.G = pp.T_max;              // one expert never has more rows than the chunk has tokens
    const int64_t Gr = pp.G;
    pp.Xq_g = take(mmq::q8_bytes(Gr, N));
    pp.Hq_g = take(mmq::q8_bytes(Gr, 640));
    pp.GU_g = (float*) take((size_t) Gr * 1280 * 4);
    pp.H_g = (float*) take((size_t) Gr * 640 * 4);
    pp.Dm_b[0] = (float*) take((size_t) Gr * N * 4);
    pp.Dm_b[1] = (float*) take((size_t) Gr * N * 4);
    const int64_t Rt = pp.cap_rows;
    pp.src = (int32_t*) take((size_t) Rt * 4);
    pp.ident = (int32_t*) take((size_t) Rt * 4);
    pp.bounds = (int32_t*) take((size_t) (2 * (NE + NE / MMQ_GROUP + 2)) * 4);
    pp.grp_gu = (uint8_t*) take(MMQ_GROUP * mp.gu_max + MMQ_TAIL);
    pp.grp_d = (uint8_t*) take(MMQ_GROUP * mp.d_max + MMQ_TAIL + MMQ_SCALES);
    if (tf > 0) {   // the FP4 share's group buffers (GU, H and the result rows are the MMQ group's)
        pp.f_a4x = take(fp4::token_bytes(Tp, N));
        pp.f_gsx = (float*) take((size_t) Tp * 4);
        pp.f_a4h = take(fp4::a4_bytes(Gr, 640, MMQ_GROUP));
        pp.f_gsh = (float*) take(fp4::scale_count(Gr, MMQ_GROUP) * 4);
        pp.f_toh = (int32_t*) take((size_t) 2 * (MMQ_GROUP + 1) * 4);
        // a piece per expert and one per whole tile run cut off a long one, and per group one bound more
        pp.f_rel_cap = (size_t) 2 * (NE + Tp * K / fp4::kTileTokens + 1);
        pp.f_rel = (int32_t*) take(pp.f_rel_cap * 4);
    }
}

static bool make_peer(core::PeerExperts* peer, int64_t T_max, int64_t cap_rows, int64_t tf,
                      std::unique_ptr<PeerPrefill>& out, std::string& err) {
    auto pp = std::make_unique<PeerPrefill>();
    pp->peer = peer;
    pp->T_max = std::min<int64_t>(T_max, stream_all_min() - 1);
    pp->T_fp4 = tf;
    const int64_t Tp = std::max(pp->T_max, tf);   // the longest chunk the peer takes
    pp->cap_rows = std::max<int64_t>(1, std::min<int64_t>(cap_rows, Tp * K));
    const MmqPlan& mp = mmq_plan();
    cudaGetDevice(&pp->prim);
    if (cudaEventCreateWithFlags(&pp->ev_in, cudaEventDisableTiming) != cudaSuccess) { err = "prefill peer: event"; return false; }
    bool ok = true;
    if (tf > 0) {   // the primary's side of the FP4 share
        int lo = 0, hi = 0;   // the rows' upload at the greatest priority: the layer's end waits for it
        cudaDeviceGetStreamPriorityRange(&lo, &hi);
        ok = cudaStreamCreateWithFlags(&pp->x_out, cudaStreamNonBlocking) == cudaSuccess &&
             cudaStreamCreateWithPriority(&pp->res_in, cudaStreamNonBlocking, hi) == cudaSuccess &&
             cudaEventCreateWithFlags(&pp->ev_hx, cudaEventDisableTiming) == cudaSuccess &&
             cudaEventCreateWithFlags(&pp->ev_res, cudaEventDisableTiming) == cudaSuccess;
        for (int b = 0; ok && b < PeerPrefill::kRes; ++b)
            ok = cudaEventCreateWithFlags(&pp->ev_h2d[b], cudaEventDisableTiming) == cudaSuccess;
    }
    pp->dev = peer->device();
    cudaSetDevice(pp->dev);
    ok = ok && cudaStreamCreateWithFlags(&pp->s, cudaStreamNonBlocking) == cudaSuccess &&
         cudaEventCreateWithFlags(&pp->ev_done, cudaEventDisableTiming) == cudaSuccess;
    if (ok) {
        ok = cudaStreamCreateWithFlags(&pp->s_out, cudaStreamNonBlocking) == cudaSuccess;
        for (int i = 0; ok && i < PeerPrefill::kGrpEv; ++i)
            ok = cudaEventCreateWithFlags(&pp->ev_grp[i], cudaEventDisableTiming) == cudaSuccess;
    }
    auto take = [&](size_t bytes) -> void* {
        void* p = nullptr;
        if (!ok || cudaMalloc(&p, bytes) != cudaSuccess) { ok = false; return nullptr; }
        pp->owned.push_back(p);
        return p;
    };
    peer_layout(*pp, Tp, tf, mp, take);
    for (int b = 0; b < 2 && ok; ++b) ok = cudaEventCreateWithFlags(&pp->ev_dm[b], cudaEventDisableTiming) == cudaSuccess;
    if (tf > 0) {
        ok = ok && cudaEventCreateWithFlags(&pp->ev_hx_free, cudaEventDisableTiming) == cudaSuccess;
        for (int b = 0; ok && b < PeerPrefill::kRes; ++b)
            ok = cudaEventCreateWithFlags(&pp->ev_d2h[b], cudaEventDisableTiming) == cudaSuccess;
    }
    if (ok) {
        pp->ctx = std::make_unique<mmq::Context>();
        pp->run_ctx = pp->ctx.get();
        mmq::iota(pp->ident, pp->cap_rows, pp->s);
        ok = cudaStreamSynchronize(pp->s) == cudaSuccess;
    }
    size_t fb = 0, tb = 0;
    cudaMemGetInfo(&fb, &tb);
    cudaSetDevice(pp->prim);
    if (ok && tf > 0) {   // pinned for both cards
        ok = cudaHostAlloc((void**) &pp->hx, (size_t) Tp * N * 4, cudaHostAllocPortable) == cudaSuccess;
        for (int b = 0; ok && b < PeerPrefill::kRes; ++b)
            ok = cudaHostAlloc((void**) &pp->hres[b], (size_t) pp->G * N * 4, cudaHostAllocPortable) == cudaSuccess;
    }
    if (!ok) {
        cudaGetLastError();
        err = "prefill peer: the peer's buffers do not fit (raise --peer-reserve-mib or lower --peer-prefill-rows)";
        return false;
    }
    basalt::log::info("basalt prefill", "peer GPU %d computes its experts' rows of each prompt chunk%s, with compact group "
                         "buffers%s; %zu MiB left free on it", pp->dev,
                 (" up to " + std::to_string(Tp) + " tokens" +
                  (tf > 0 ? " (a streamed one with the primary's FP4 kernels)" : "")).c_str(),
                 "", fb >> 20);
    out = std::move(pp);
    return true;
}

uint64_t Prefill::peer_bytes(int64_t chunk, bool streamed_share) {
    if (!mmq_plan().any) return 0;
    PeerPrefill pp;
    pp.T_max = std::min<int64_t>(chunk, streamed_share ? stream_all_min() - 1 : kPeerSlowRows);
    const int64_t t_fp4 = !streamed_share || !fp4_every_layer() || fused_ring() || stream_all_min() < FP4_MIN_TOKENS
                              ? 0 : std::min<int64_t>(chunk, kPeerFp4Tokens);
    const int64_t tf = t_fp4 >= stream_all_min() ? t_fp4 : 0;
    const int64_t Tp = std::max(pp.T_max, tf);
    uint64_t sum = 0;
    peer_layout(pp, Tp, tf, mmq_plan(), [&](size_t bytes) -> void* { sum += (bytes + 255) & ~size_t(255); return nullptr; });
    return sum;
}

bool Prefill::set_peer(core::PeerExperts* peer, int64_t cap_rows, bool streamed_share, std::string& err) {
    Impl& m = *impl_;
    if (peer == nullptr || !peer->valid()) { m.pp.reset(); return true; }
    // The peer takes the rows of the routed-only chunks (below stream_all_min()): the activations and
    // result rows cross through host staging (cudaMemcpyPeerAsync) and the streamed ring stays off.  It also takes
    // those of a streamed chunk of up to kPeerFp4Tokens (PeerPrefill::T_fp4) where those buffers fit too.
    if (!mmq_plan().any) { err = "prefill peer: needs the MMQ prompt path"; return false; }
    const int64_t t_fp4 = !streamed_share || !fp4_every_layer() || fused_ring() || stream_all_min() < FP4_MIN_TOKENS
                              ? 0 : std::min<int64_t>(m.T_max, kPeerFp4Tokens);
    std::unique_ptr<PeerPrefill> pp;
    const int64_t t_routed = streamed_share ? m.T_max : std::min<int64_t>(m.T_max, kPeerSlowRows);
    bool ok = make_peer(peer, t_routed, cap_rows, t_fp4 >= stream_all_min() ? t_fp4 : 0, pp, err);
    if (!ok && t_fp4 >= stream_all_min()) {   // the routed-only chunks alone
        pp.reset();
        err.clear();
        ok = make_peer(peer, t_routed, cap_rows, 0, pp, err);
    }
    if (!ok) return false;
    m.pp = std::move(pp);
    return true;
}

}  // namespace basalt::prefill
