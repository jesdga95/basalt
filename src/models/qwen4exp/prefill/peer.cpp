#include "state.hpp"
#include "layout.hpp"
#include "basalt/models/qwen4exp/prefill/prefill.hpp"
#include "basalt/models/qwen4exp/decode/mtp.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/models/qwen4exp/kernels/ngram.hpp"
#include "basalt/models/qwen4exp/decode/native_head.hpp"
#include "basalt/prefill/moe_mmq.hpp"
#include "basalt/models/qwen4exp/prefill/moe_fp4.hpp"
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

// The peer card's prompt buffers for chunks of up to `Tp` tokens in groups of `G` rows: the shared transport's
// dimensions, and (`tf` > 0) the FP4 share's own buffers, each from `take(bytes)`, in one place so that the tier's reserve
// and the allocation count the same bytes.
static PeerShare::Dims share_dims(int64_t Tp, int64_t G, int64_t tf, const MmqPlan& mp) {
    PeerShare::Dims d;
    d.n_embd = N;
    d.n_ff = 640;
    d.group_bytes = peer_down_at(mp) + MMQ_GROUP * mp.d_max + MMQ_TAIL + MMQ_SCALES;
    d.act_bytes = mmq::q8_bytes(Tp, N);
    if (tf > 0) d.act_bytes = std::max(d.act_bytes, peer_scales_at(Tp) + (size_t) Tp * 4);
    d.max_tokens = (int) Tp;
    d.max_rows = Tp * K;   // no row cap: only the row tables grow with it
    d.n_expert = (int) NE;
    d.launch_experts = MMQ_GROUP;
    d.launch_rows = G;
    // one group buffer, as many as its own gathers had: the group's gather follows the last group's products
    d.stage_bufs = 1;
    d.bufs = 2;
    return d;
}
template <class Take>
static void fp4_layout(PeerPrefill& pp, int64_t Tp, Take&& take) {
    const int64_t Gr = pp.G;
    pp.f_a4x = take(fp4::token_bytes(Tp, N));
    pp.f_gsx = (float*) take((size_t) Tp * 4);
    pp.f_h = (float*) take((size_t) Gr * 640 * 4);
    pp.f_a4h = take(fp4::a4_bytes(Gr, 640, MMQ_GROUP));
    pp.f_gsh = (float*) take(fp4::scale_count(Gr, MMQ_GROUP) * 4);
    pp.f_toh = (int32_t*) take((size_t) 2 * (MMQ_GROUP + 1) * 4);
}

static bool make_peer(core::PeerExperts* peer, int64_t T_max, int64_t tf, std::unique_ptr<PeerPrefill>& out,
                      std::string& err) {
    auto pp = std::make_unique<PeerPrefill>();
    pp->peer = peer;
    pp->T_max = std::min<int64_t>(T_max, stream_all_min() - 1);
    pp->T_fp4 = tf;
    pp->G = pp->T_max;   // one expert never has more rows than the chunk has tokens
    const int64_t Tp = std::max(pp->T_max, tf);   // the longest chunk the peer takes
    pp->cap_rows = Tp * K;
    int prim = 0;
    cudaGetDevice(&prim);
    pp->dev = peer->device();
    bool ok = true;
    size_t fb = 0, tb = 0;
    {
        const core::OnDevice od(pp->dev);
        if (tf > 0)
            fp4_layout(*pp, Tp, [&](size_t bytes) -> void* {
                void* p = nullptr;
                if (!ok || cudaMalloc(&p, bytes) != cudaSuccess) { ok = false; return nullptr; }
                pp->owned.push_back(p);
                return p;
            });
    }
    if (ok) {
        pp->share = std::make_unique<PeerShare>();
        std::string e2;
        ok = pp->share->open(prim, pp->dev, share_dims(Tp, pp->G, tf, mmq_plan()), e2);
    }
    {
        const core::OnDevice od(pp->dev);
        cudaMemGetInfo(&fb, &tb);
    }
    if (!ok) {
        cudaGetLastError();
        err = "prefill peer: the peer's buffers do not fit (raise --peer-reserve-mib or use --no-peer-prefill)";
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
    pp.G = pp.T_max;
    const int64_t t_fp4 = !streamed_share || !fp4_every_layer() || fused_ring() || stream_all_min() < FP4_MIN_TOKENS
                              ? 0 : std::min<int64_t>(chunk, kPeerFp4Tokens);
    const int64_t tf = t_fp4 >= stream_all_min() ? t_fp4 : 0;
    const int64_t Tp = std::max(pp.T_max, tf);
    uint64_t sum = PeerShare::peer_bytes(share_dims(Tp, pp.G, tf, mmq_plan()));
    if (tf > 0) fp4_layout(pp, Tp, [&](size_t bytes) -> void* { sum += (bytes + 255) & ~size_t(255); return nullptr; });
    return sum;
}

bool Prefill::set_peer(core::PeerExperts* peer, bool streamed_share, std::string& err) {
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
    bool ok = make_peer(peer, t_routed, t_fp4 >= stream_all_min() ? t_fp4 : 0, pp, err);
    if (!ok && t_fp4 >= stream_all_min()) {   // the routed-only chunks alone
        pp.reset();
        err.clear();
        ok = make_peer(peer, t_routed, 0, pp, err);
    }
    if (!ok) return false;
    m.pp = std::move(pp);
    return true;
}

}  // namespace basalt::prefill
