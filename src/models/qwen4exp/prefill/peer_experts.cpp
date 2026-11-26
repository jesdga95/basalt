#include "experts.hpp"
#include "basalt/models/qwen4exp/prefill/prefill.hpp"
#include "basalt/models/qwen4exp/decode/mtp.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/models/qwen4exp/decode/native_head.hpp"
#include "basalt/models/qwen4exp/prefill/gemm.hpp"
#include "basalt/models/qwen4exp/prefill/moe_fused.hpp"
#include "basalt/prefill/moe_mmq.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/log.hpp"
#include <algorithm>

namespace basalt::prefill {
using namespace detail;

bool Prefill::Run::Chunk::Experts::Group::peer_experts() {
    peer_now = use_mmq && !order_peer.empty();
    if (!peer_now) return true;
    PeerPrefill& P = *ppc;
    const auto& f = lay.fmt[(size_t) l];
    std::vector<PeerShare::Expert> ex;
    PeerShare::Layer L;
    L.gu_type = mmq_gt;
    L.d_type = mmq_dt;
    L.up_type = mmq_ut;
    L.up_at = mmq_gb;
    const size_t n = order_peer.size();
    // groups of up to MMQ_GROUP experts and at most G rows, each one launch
    size_t g0 = 0;
    int64_t gr = 0;
    for (size_t j = 0; j < n; ++j) {
        const int32_t e = order_peer[j];
        const int64_t c = m.cnt[(size_t) e];
        PeerShare::Expert x;
        x.blob = P.peer->slot_ptr(l, e);
        x.on_peer = true;
        x.tok = src_h + m.off[(size_t) e];
        x.n = (int) c;
        x.group = j == 0 || j - g0 == (size_t) MMQ_GROUP || gr + c > P.G;
        if (x.group) {
            g0 = j;
            gr = 0;
        }
        gr += c;
        ex.push_back(x);
    }
    // a group's blobs out of the peer's slots into the MMQ layout of a group buffer (peer_down_at)
    const size_t gub = mmq_gub, gb = mmq_gb, ub = mmq_ub, db = mmq_db, down_at = peer_down_at(mmq_plan());
    const bool native = lay.native;
    const size_t up_off = f.up_off, down_off = f.down_off, tail = f.tail, bytes = f.bytes;
    L.gu_stride = gub;
    L.down_off = down_at;
    L.d_stride = db;
    L.stage = [=](const std::vector<std::pair<const uint8_t*, bool>>& blobs, uint8_t* grp, void* stream) {
        uint8_t* grp_d = grp + down_at;
        for (size_t q = 0; q < blobs.size(); ++q) {
            const uint8_t* bd = blobs[q].first;
            if (native)
                mmq::gather_native(bd, bd + up_off, gb, bd + down_off, db, grp + q * gub, grp_d + q * db, stream,
                                   tail ? bd + bytes - tail : nullptr, grp_scales(grp_d) + 4 * q, ub);
            else
                mmq::gather_q2(bd, grp + q * gub, grp_d + q * db, stream);
        }
        cudaMemsetAsync(grp + blobs.size() * gub, 0, MMQ_TAIL, (cudaStream_t) stream);
        cudaMemsetAsync(grp_d + blobs.size() * db, 0, MMQ_TAIL, (cudaStream_t) stream);
    };
    L.act = [native, tail, down_at](const float* gu, float* h, const PeerShare::Launch& lv, void* stream) {
        const float* sc = native && tail ? grp_scales((uint8_t*) lv.stage + down_at) + 4 * lv.q0 : nullptr;
        mmq::swiglu(gu, h, lv.rows, 640, !native, stream, sc, lv.bounds, lv.n);
    };
    return P.share->run(L, m.mixed, (int) T, ex, m.Dm + (size_t) rows_local * N, m.cs, err);
}

}  // namespace basalt::prefill
