#include "experts.hpp"
#include "basalt/models/qwen4exp/prefill/prefill.hpp"
#include "basalt/models/qwen4exp/decode/mtp.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/models/qwen4exp/decode/native_head.hpp"
#include "basalt/models/qwen4exp/prefill/gemm.hpp"
#include "basalt/models/qwen4exp/prefill/moe_fused.hpp"
#include "basalt/prefill/moe_mmq.hpp"
#include "basalt/models/qwen4exp/prefill/moe_fp4.hpp"
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
    if (peer_fp4) {
        // a streamed chunk: the FP4 kernels in pieces of at most G rows (PeerPrefill::T_fp4), the blobs read in place
        if (!fp4_l) {
            err = "prefill: the peer's share of a streamed chunk needs the FP4 experts";
            return false;
        }
        std::vector<int32_t> bounds(n + 1);
        for (size_t j = 0; j < n; ++j) bounds[j] = m.off[(size_t) order_peer[j]] - (int32_t) rows_local;
        bounds[n] = (int32_t) rows_peer;
        fp4::cut_pieces(bounds.data(), (int) n, P.G, P.pieces);
        const fp4::Pieces& pc = P.pieces;
        for (size_t g2 = 0; g2 + 1 < pc.group.size(); ++g2)
            for (int i = pc.group[g2]; i < pc.group[g2 + 1]; ++i) {
                PeerShare::Expert x;
                x.blob = P.peer->slot_ptr(l, order_peer[(size_t) pc.expert[(size_t) i]]);
                x.on_peer = true;
                x.tok = src_h + rows_local + pc.bounds[(size_t) i];
                x.n = pc.bounds[(size_t) i + 1] - pc.bounds[(size_t) i];
                x.group = i == pc.group[g2];
                ex.push_back(x);
            }
        const size_t scales_at = peer_scales_at(std::max(P.T_max, P.T_fp4));
        // the layer's activations as FP4 once per token (the primary's bytes), on the peer before its first launch
        L.quantize = [scales_at](const float* x, int tokens, void* dst, void* stream) {
            fp4::quantize_tokens(x, tokens, N, N, dst, (float*) ((uint8_t*) dst + scales_at), stream);
        };
        L.stage = [](const std::vector<std::pair<const uint8_t*, bool>>&, uint8_t*, void*) {};
        PeerPrefill* pp = &P;
        const size_t down_off = f.down_off, tail_off = f.bytes - f.tail;
        L.products = [pp, scales_at, down_off, tail_off](const PeerShare::Launch& lv, void* stream) {
            const cudaStream_t s = (cudaStream_t) stream;
            if (lv.first) {
                cudaMemcpyAsync(pp->f_a4x, lv.act, fp4::token_bytes(lv.n_tokens, N), cudaMemcpyHostToDevice, s);
                cudaMemcpyAsync(pp->f_gsx, (const uint8_t*) lv.act + scales_at, (size_t) lv.n_tokens * 4,
                                cudaMemcpyHostToDevice, s);
            }
            fp4::Group fg;
            fg.n = lv.n;
            for (int i = 0; i < lv.n; ++i) fg.blob[i] = lv.blob[i];
            fg.bounds_host = lv.bounds_host;
            fg.bounds_dev = lv.bounds;
            fg.ids = lv.tok; fg.a4x = pp->f_a4x; fg.gsx = pp->f_gsx; fg.tokens = lv.n_tokens;
            fg.down_off = down_off;
            fg.tail_off = tail_off;
            fg.h = pp->f_h; fg.dm = lv.out;
            fg.a4h = pp->f_a4h; fg.gsh = pp->f_gsh; fg.toh = pp->f_toh; fg.rel = pp->f_toh + (MMQ_GROUP + 1);
            fp4::run_group(fg, pp->G, stream);
        };
    } else {
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
    }
    return P.share->run(L, m.mixed, (int) T, ex, m.Dm + (size_t) rows_local * N, m.cs, err);
}

}  // namespace basalt::prefill
