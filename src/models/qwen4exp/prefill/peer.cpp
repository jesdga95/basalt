#include "basalt/core/gpu_mem_info.hpp"
#include "state.hpp"
#include "layout.hpp"
#include "basalt/models/qwen4exp/prefill/prefill.hpp"
#include "basalt/models/qwen4exp/decode/mtp.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/models/qwen4exp/kernels/ngram.hpp"
#include "basalt/models/qwen4exp/decode/native_head.hpp"
#include "basalt/prefill/moe_mmq.hpp"
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

// The peer card's prompt buffers for routed-only chunks of up to `Tp` tokens in groups of `G` rows: the shared
// transport's dimensions, in one place so that the tier's reserve and the allocation count the same bytes.
static PeerShare::Dims share_dims(int64_t Tp, int64_t G, const MmqPlan& mp) {
    PeerShare::Dims d;
    d.n_embd = N;
    d.n_ff = 640;
    d.group_bytes = peer_down_at(mp) + MMQ_GROUP * mp.d_max + MMQ_TAIL + MMQ_SCALES;
    d.act_bytes = mmq::q8_bytes(Tp, N);
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

// The longest routed-only chunk the peer takes for chunks of up to `chunk` tokens.
static int64_t peer_tokens(int64_t chunk) { return std::min<int64_t>({chunk, kPeerSlowRows, stream_all_min() - 1}); }

uint64_t Prefill::peer_bytes(int64_t chunk) {
    if (!mmq_plan().any) return 0;
    const int64_t Tp = peer_tokens(chunk);
    return PeerShare::peer_bytes(share_dims(Tp, Tp, mmq_plan()));
}

bool Prefill::set_peer(core::PeerExperts* peer, std::string& err) {
    Impl& m = *impl_;
    if (peer == nullptr || !peer->valid()) { m.pp.reset(); return true; }
    // The peer takes the rows of the routed-only chunks (below stream_all_min()) up to kPeerSlowRows tokens: the
    // activations and result rows cross through host staging (cudaMemcpyPeerAsync) and the streamed ring stays off.
    if (!mmq_plan().any) { err = "prefill peer: needs the MMQ prompt path"; return false; }
    auto pp = std::make_unique<PeerPrefill>();
    pp->peer = peer;
    pp->T_max = peer_tokens(m.T_max);
    pp->G = pp->T_max;   // one expert never has more rows than the chunk has tokens
    pp->cap_rows = pp->T_max * K;
    int prim = 0;
    cudaGetDevice(&prim);
    pp->dev = peer->device();
    pp->share = std::make_unique<PeerShare>();
    std::string e2;
    const bool ok = pp->share->open(prim, pp->dev, share_dims(pp->T_max, pp->G, mmq_plan()), e2);
    size_t fb = 0, tb = 0;
    {
        const core::OnDevice od(pp->dev);
        basalt::core::gpu_mem_info(&fb, &tb);
    }
    if (!ok) {
        cudaGetLastError();
        err = "prefill peer: the peer's buffers do not fit (raise --peer-reserve-mib)";
        return false;
    }
    basalt::log::info("basalt prefill", "peer GPU %d computes its experts' rows of each prompt chunk up to %lld tokens, with "
                      "compact group buffers; %zu MiB left free on it", pp->dev, (long long) pp->T_max, fb >> 20);
    m.pp = std::move(pp);
    return true;
}

}  // namespace basalt::prefill
