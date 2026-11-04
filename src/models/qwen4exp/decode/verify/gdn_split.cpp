#include "common.hpp"
#include "expert_plan.hpp"
#include "basalt/core/load_progress.hpp"
#include "basalt/core/verify.hpp"
#include "basalt/core/diag.hpp"
#include "basalt/core/native_head.hpp"
#include "basalt/core/gdn_head_split.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/kernels/dec_gdn.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/core/progress.hpp"
#include "basalt/kernels/verify_kernels.hpp"
#include "basalt/kernels/dec_moe.hpp"
#include "basalt/log.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

namespace basalt::core {
using namespace verify_detail;

// The GDN mixer splits by key head when the peer card streams the mixer's bytes faster than CUDA0.  The unit is the
// fewest key heads whose value-head runs fill whole out-projection blocks; the peer's share of the units follows both
// cards' memory bandwidth, rounded down, and is taken only from a quarter up, CUDA0 keeping at least one unit.  The
// peer holds its heads' weights and state in gdn_head_split.hpp's layout.
// The key heads CUDA0 keeps when the GDN mixer splits by key head between CUDA0 (dev0) and the peer (dev1); 0 when the
// split is not taken.  Units of key heads whose out-projection columns fill whole 256-value blocks; the peer's units
// follow the two cards' memory bandwidth, taken only from a quarter of them.
static bool g_gs_off = false;   // set by Verifier::gs_decide when the split does not pay

static int gs_plan_bw(const ModelGeometry& g, int dev0, int dev1) {
    const int HK = (int) g.ssm_k_heads, HV = (int) g.ssm_v_heads, S = (int) g.ssm_state_size, HIST = (int) g.ssm_d_conv - 1;
    const int64_t N = g.n_embd, C = g.ssm_conv_channels, ZV = g.ssm_value_dim;
    constexpr int kBlockCols = 256;
    int unit = 1;
    while (unit <= HK && (unit * S) % kBlockCols != 0) ++unit;
    if (HK <= 0 || HV % HK != 0 || unit > HK || HK % unit != 0 || N % kBlockCols != 0 || HIST < 1 ||
        C != (int64_t) (2 * HK + HV) * S || ZV != (int64_t) HV * S)
        return 0;
    const double bw0 = card_bandwidth(dev0), bw1 = card_bandwidth(dev1);
    if (bw0 <= 0.0 || bw1 <= 0.0) return 0;
    const int units = HK / unit;
    const int u1 = std::min(units - 1, (int) std::floor(units * bw1 / (bw0 + bw1)));
    if (u1 * 4 < units) return 0;
    return HK - u1 * unit;
}

static int gs_plan(const ModelGeometry& g, int dev0, int dev1) { return g_gs_off ? 0 : gs_plan_bw(g, dev0, dev1); }

// The split's start-time price, in ms per verify window.  It saves the main card's stream of the peer's heads' mixer
// weights at its achieved bandwidth, less the peer item's growth per window token beyond the first.  Its reserve
// displaces the same bytes of experts from the peer to RAM, which decode pays as CPU-tier misses.  The prompt path is
// not priced: it streams the displaced experts with the rest of a chunk's, at no measurable cost.
bool Verifier::gs_decide(const WeightTable& wt, const ModelGeometry& g, int dev0, int dev1, int max_t,
                         int64_t displaced) {
    g_gs_off = false;
    if (dev1 < 0 || dev1 == dev0) return false;
    const int nk0 = gs_plan_bw(g, dev0, dev1);
    if (nk0 == 0) return false;
    constexpr double kMainEff = 0.71;        // the main card's achieved share of its bandwidth on the mixer
    constexpr double kTokGrowth = 0.71;      // the peer item's growth per token, in units of its weight stream
    constexpr double kMissMsPerGB = 0.263;   // decode CPU-tier misses per window per GB of displaced experts
    const int64_t HK = g.ssm_k_heads, N = g.n_embd, nk1 = HK - nk0;
    const int64_t C1 = g.ssm_conv_channels / HK * nk1, Z1 = g.ssm_value_dim / HK * nk1;
    double bm = 0.0;
    int layers = 0;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        if (is_qsa_layer(g, l)) continue;
        const LayerView v(wt, l);
        const WeightRef *q = v.get("attn_qkv.weight"), *z = v.get("attn_gate.weight"), *o = v.get("ssm_out.weight");
        if (!q || !z || !o) return false;
        bm += (double) basalt::kernels::native_mmvq_weight_bytes(q->native_type, (int) N, (int) C1) +
              (double) basalt::kernels::native_mmvq_weight_bytes(z->native_type, (int) N, (int) Z1) +
              (double) basalt::kernels::native_mmvq_weight_bytes(o->native_type, (int) Z1, (int) N);
        ++layers;
    }
    if (layers == 0) return false;
    bm /= layers;
    const double bw0 = card_bandwidth(dev0), bw1 = card_bandwidth(dev1);
    const double t_exp = std::max(1.0, (max_t + 2) / 3.0);   // the expected window: a third of the drafted run
    const double save = layers * (bm / (kMainEff * bw0) - kTokGrowth * (t_exp - 1.0) * bm / bw1) * 1e3;
    const double dgb = (double) std::max<int64_t>(displaced, 0) / 1e9;
    const double miss = kMissMsPerGB * dgb;
    const bool take = save - miss > 0.0;
    g_gs_off = !take;
    basalt::log::info("basalt verify", "GDN mixer split by key head: saves %.2f ms per %.1f-token window, costs %.2f "
                         "(%.2f GB of experts off the peer, read as decode misses) -> %s",
                 save, t_exp, miss, dgb, take ? "split" : "whole on CUDA0");
    return take;
}

int64_t Verifier::gs_cuda0_bytes(const WeightTable& wt, const ModelGeometry& g, int dev0, int dev1) {
    if (dev1 < 0 || dev1 == dev0) return 0;
    const int nk0 = gs_plan(g, dev0, dev1);
    if (nk0 == 0) return 0;
    const int64_t HK = g.ssm_k_heads, N = g.n_embd, C0 = g.ssm_conv_channels / HK * nk0, Z0 = g.ssm_value_dim / HK * nk0;
    int64_t b = 0;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        if (is_qsa_layer(g, l)) continue;
        const LayerView v(wt, l);
        const WeightRef *q = v.get("attn_qkv.weight"), *z = v.get("attn_gate.weight"), *o = v.get("ssm_out.weight");
        if (!q || !z || !o) return 0;
        b += (int64_t) basalt::kernels::native_mmvq_weight_bytes(q->native_type, (int) N, (int) C0) +
             (int64_t) basalt::kernels::native_mmvq_weight_bytes(z->native_type, (int) N, (int) Z0) +
             (int64_t) basalt::kernels::native_mmvq_weight_bytes(o->native_type, (int) Z0, (int) N);
    }
    return b;
}

int64_t Verifier::gs_peer_bytes(const WeightTable& wt, const ModelGeometry& g, int dev0, int dev1) {
    if (dev1 < 0 || dev1 == dev0) return 0;
    const int nk0 = gs_plan(g, dev0, dev1);
    if (nk0 == 0) return 0;
    const int HK = (int) g.ssm_k_heads, HV = (int) g.ssm_v_heads, S = (int) g.ssm_state_size, HIST = (int) g.ssm_d_conv - 1;
    const basalt::core::GdnHeadSlice sl{HK, HV, S, nk0, HK - nk0};
    const int64_t N = g.n_embd, C1 = sl.c(), Z1 = sl.zv(), V1 = sl.nv();
    int64_t b = 0, layers = 0;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        if (is_qsa_layer(g, l)) continue;
        const LayerView v(wt, l);
        const WeightRef *q = v.get("attn_qkv.weight"), *z = v.get("attn_gate.weight"), *o = v.get("ssm_out.weight");
        if (!q || !z || !o) return 0;
        b += (int64_t) basalt::kernels::native_mmvq_weight_bytes(q->native_type, (int) N, (int) C1) +
             (int64_t) basalt::kernels::native_mmvq_weight_bytes(z->native_type, (int) N, (int) Z1) +
             (int64_t) basalt::kernels::native_mmvq_weight_bytes(o->native_type, (int) Z1, (int) N) +
             C1 * (HIST + 1) * 4 + V1 * N * 2 * 2 + V1 * 4 * 2 + (int64_t) S * 4 +
             (int64_t) basalt::core::gdn_slice_state_floats(sl, HIST) * 4;
        ++layers;
    }
    // the mixer's rows and per-allocation rounding
    return b + layers * (2 << 20);
}

void Verifier::gs_init() {
    using basalt::core::GdnHeadSlice;
    if (dm_ == nullptr || dm_->pdev < 0 || wt_ == nullptr) return;
    const ModelGeometry& g = *g_;
    const int HK = (int) g.ssm_k_heads, HV = (int) g.ssm_v_heads, S = (int) g.ssm_state_size, HIST = (int) g.ssm_d_conv - 1;
    const int64_t N = g.n_embd, C = g.ssm_conv_channels, ZV = g.ssm_value_dim;
    constexpr int kBlockCols = 256;
    const int nk0 = gs_plan(g, device_, dm_->pdev);
    if (nk0 == 0) return;
    const GdnHeadSlice sl{HK, HV, S, nk0, HK - nk0};
    auto blocked = [](int t) { return t == 11 || t == 12 || t == 13 || t == 14 || t == 23; };   // 256-value blocks
    // every GDN layer's tensors, as the verify window reads them
    struct Src {
        const WeightRef *qkv, *z, *out, *a, *b, *conv, *norm, *dt, *sa;
    };
    std::vector<Src> src;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        if (is_qsa_layer(g, l)) continue;
        const LayerView v(*wt_, l);
        Src s{v.get("attn_qkv.weight"), v.get("attn_gate.weight"), v.get("ssm_out.weight"), v.get("ssm_alpha.weight"),
              v.get("ssm_beta.weight"), v.get("ssm_conv1d.weight"), v.get("ssm_norm.weight"), v.get("ssm_dt.bias"),
              v.get("ssm_a")};
        if (!s.qkv || !s.z || !s.out || !s.a || !s.b || !s.conv || !s.norm || !s.dt || !s.sa || !s.qkv->native_data ||
            !s.z->native_data || !s.out->native_data || !blocked(s.qkv->native_type) || !blocked(s.z->native_type) ||
            !blocked(s.out->native_type) || s.conv->bytes != (uint64_t) C * (HIST + 1) * 4 ||
            s.a->bytes != (uint64_t) HV * N * 2 || s.b->bytes != (uint64_t) HV * N * 2 ||
            s.dt->bytes != (uint64_t) HV * 4 || s.sa->bytes != (uint64_t) HV * 4 || s.norm->bytes != (uint64_t) S * 4) {
            basalt::log::warn("basalt verify", "the GDN mixer stays whole on CUDA0: a layer the split does not cover");
            return;
        }
        src.push_back(s);
    }
    if (src.empty()) return;
    // each tensor comes down from CUDA0, is sliced on the host and goes up to the peer
    std::vector<uint8_t> full, part;
    size_t peer_bytes = 0;
    bool ok = true;
    auto down = [&](const void* d, size_t n) {
        full.resize(n);
        const OnDevice on(device_);
        ok = ok && cudaMemcpy(full.data(), d, n, cudaMemcpyDeviceToHost) == cudaSuccess;
    };
    size_t own_bytes = 0;
    auto up = [&](size_t n, void** dst, int dev) {
        const OnDevice on(dev);
        ok = ok && cudaMalloc(dst, n) == cudaSuccess && cudaMemcpy(*dst, part.data(), n, cudaMemcpyHostToDevice) == cudaSuccess;
        (dev == device_ ? own_bytes : peer_bytes) += n;
    };
    auto rows = [&](const void* d, size_t row_bytes, int64_t n_rows, const std::vector<basalt::core::GdnRun>& runs,
                    void** dst, int dev) {
        if (!ok) return;
        down(d, row_bytes * (size_t) n_rows);
        part.resize(row_bytes * (size_t) basalt::core::gdn_runs_len(runs));
        basalt::core::gdn_gather_rows(full.data(), row_bytes, runs, part.data());
        up(part.size(), dst, dev);
    };
    // the out projection's columns of a slice's value heads
    auto cols = [&](const void* d, int type, const GdnHeadSlice& h, void** dst, int dev) {
        if (!ok) return;
        const size_t rb = basalt::kernels::native_mmvq_weight_bytes(type, (int) ZV, 1);
        down(d, rb * (size_t) N);
        part.resize(basalt::kernels::native_mmvq_weight_bytes(type, (int) h.zv(), (int) N));
        basalt::core::gdn_gather_cols(full.data(), N, ZV, kBlockCols, rb / (size_t) (ZV / kBlockCols),
                                      basalt::core::gdn_vhead_runs(h, S), part.data());
        up(part.size(), dst, dev);
    };
    const int P = dm_->pdev, D0 = device_;
    // CUDA0 runs its own heads in place in the full state (dec_gdn_rec's head map) from packed qkv/z rows and out columns
    const GdnHeadSlice sl0{HK, HV, S, 0, nk0};
    const std::vector<basalt::core::GdnRun> qr = basalt::core::gdn_qkv_runs(sl), v1 = basalt::core::gdn_vhead_runs(sl, 1),
                                           vs = basalt::core::gdn_vhead_runs(sl, S),
                                           qr0 = basalt::core::gdn_qkv_runs(sl0), vs0 = basalt::core::gdn_vhead_runs(sl0, S);
    gs_layers_.assign(src.size(), GsLayer{});
    gs_dev_ = dm_->pdev;
    for (size_t i = 0; i < src.size() && ok; ++i) {
        const Src& s = src[i];
        GsLayer& L = gs_layers_[i];
        L.tq = s.qkv->native_type; L.tz = s.z->native_type; L.to = s.out->native_type;
        const size_t rq = basalt::kernels::native_mmvq_weight_bytes(L.tq, (int) N, 1),
                     rz = basalt::kernels::native_mmvq_weight_bytes(L.tz, (int) N, 1);
        rows(s.qkv->native_data, rq, C, qr, &L.wqkv, P);
        rows(s.z->native_data, rz, ZV, vs, &L.wz, P);
        rows(s.conv->data, (size_t) (HIST + 1) * 4, C, qr, (void**) &L.conv_w, P);
        rows(s.a->data, (size_t) N * 2, HV, v1, (void**) &L.wa, P);
        rows(s.b->data, (size_t) N * 2, HV, v1, (void**) &L.wb, P);
        rows(s.dt->data, 4, HV, v1, (void**) &L.dt, P);
        rows(s.sa->data, 4, HV, v1, (void**) &L.ssm_a, P);
        rows(s.norm->data, (size_t) S * 4, 1, {{0, 1}}, (void**) &L.gamma, P);
        cols(s.out->native_data, L.to, sl, &L.wo, P);
        rows(s.qkv->native_data, rq, C, qr0, &L.wqkv0, D0);
        rows(s.z->native_data, rz, ZV, vs0, &L.wz0, D0);
        cols(s.out->native_data, L.to, sl0, &L.wo0, D0);
    }
    const size_t state_bytes = basalt::core::gdn_slice_state_floats(sl, HIST) * 4 * src.size();
    if (ok) {
        const OnDevice on(dm_->pdev);
        ok = cudaMalloc((void**) &gs_state_, state_bytes) == cudaSuccess && cudaMemset(gs_state_, 0, state_bytes) == cudaSuccess;
        peer_bytes += state_bytes;
        // the mixer doorbell: one per GDN layer's ring
        std::vector<uint8_t> mix((size_t) dm_->rings, 0);
        for (int64_t l = lb_; l < le_; ++l) mix[(size_t) (l - lb_)] = is_qsa_layer(g, l) ? 0 : 1;
        ok = ok && cudaMalloc((void**) &gs_mix_, mix.size()) == cudaSuccess &&
             cudaMemcpy(gs_mix_, mix.data(), mix.size(), cudaMemcpyHostToDevice) == cudaSuccess &&
             cudaMalloc(&gs_pxq_, dm_->xq_stride) == cudaSuccess;
        // the peer's scratch: per layer qkv, h, gate, beta (kept for its commit); shared x, z, o, yq
        const int64_t MT = max_t_, C1 = sl.c(), Z1 = sl.zv(), V1 = sl.nv();
        const size_t per = (size_t) (MT * (2 * C1 + 2 * V1)), shared = (size_t) (MT * (N + 2 * Z1) + MT * Z1 / 32 * 9);
        const size_t nl = src.size();
        ok = ok && cudaMalloc((void**) &gs_pscr_, (per * nl + shared) * 4) == cudaSuccess;
        peer_bytes += (per * nl + shared) * 4;
        std::vector<basalt::kernels::DmMixLayer> tab((size_t) dm_->rings);
        if (ok) {
            float* sh = gs_pscr_ + per * nl;
            float *px = sh, *pz = px + MT * N, *po = pz + MT * Z1;
            void* pyq = po + MT * Z1;
            size_t i = 0;
            for (int64_t l = lb_; l < le_; ++l) {
                if (is_qsa_layer(g, l)) continue;
                const GsLayer& L = gs_layers_[i];
                basalt::kernels::DmMixLayer& m = tab[(size_t) (l - lb_)];
                m.wqkv = L.wqkv; m.wz = L.wz; m.wo = L.wo; m.tq = L.tq; m.tz = L.tz; m.to = L.to;
                basalt::kernels::DecRecArgs& r = m.rec;
                float* pl = gs_pscr_ + per * i;
                r.state = gs_state_ + basalt::core::gdn_slice_state_floats(sl, HIST) * i;
                r.qkv = pl; r.h = pl + MT * C1; r.gate = pl + 2 * MT * C1; r.beta = r.gate + MT * V1;
                r.conv_w = L.conv_w; r.x = px; r.w_alpha = L.wa; r.w_beta = L.wb; r.dt = L.dt; r.ssm_a = L.ssm_a;
                r.z = pz; r.gamma = L.gamma; r.o = po; r.yq = pyq;
                r.C = (int) C1; r.n_embd = (int) N; r.h_k = HK - nk0; r.h_v = (int) V1; r.eps = EPS;
                r.early_trigger = false;
                ++i;
            }
        }
        gs_ca_ = basalt::kernels::GdnCommitLayers{};
        gs_ca_.state = gs_state_; gs_ca_.qkv = gs_pscr_; gs_ca_.h = gs_pscr_ + MT * C1;
        gs_ca_.gate = gs_pscr_ + 2 * MT * C1; gs_ca_.beta = gs_ca_.gate + MT * V1;
        gs_ca_.state_stride = basalt::core::gdn_slice_state_floats(sl, HIST); gs_ca_.h_stride = per; gs_ca_.gb_stride = per;
        gs_ca_.C = (int) C1; gs_ca_.h_k = HK - nk0; gs_ca_.h_v = (int) V1; gs_ca_.n_layers = (int) nl;
        ok = ok && cudaMalloc((void**) &gs_mixl_, tab.size() * sizeof(tab[0])) == cudaSuccess &&
             cudaMemcpy(gs_mixl_, tab.data(), tab.size() * sizeof(tab[0]), cudaMemcpyHostToDevice) == cudaSuccess;
    }
    ok = ok && mapped((size_t) dm_->rings * dm_->xq_stride, (void**) &gs_h_xq_, (void**) &gs_m_xq_) &&
         mapped((size_t) dm_->rings * dm_->x_stride * 4, (void**) &gs_h_x_, (void**) &gs_m_x_) &&
         mapped((size_t) max_t_ * N * 4, (void**) &gs_h_part_, (void**) &gs_m_part_);
    if (ok) {   // gs_move's staging, pinned so the moves run at the bus rate
        const size_t full_f = (size_t) S * HV * S + (size_t) C * HIST, nl = (size_t) gs_ca_.n_layers;
        ok = cudaHostAlloc((void**) &gs_h_full_, full_f * nl * 4, cudaHostAllocPortable) == cudaSuccess &&
             cudaHostAlloc((void**) &gs_h_slice_, basalt::core::gdn_slice_state_floats(sl, HIST) * nl * 4,
                           cudaHostAllocPortable) == cudaSuccess;
    }
    for (const Src& s : src)
        ok = ok && basalt::kernels::dec_moe_mix_type_ok(s.qkv->native_type) &&
             basalt::kernels::dec_moe_mix_type_ok(s.z->native_type) && basalt::kernels::dec_moe_mix_type_ok(s.out->native_type);
    if (ok) {
        const OnDevice on(device_);
        const GdnHeadSlice s0{HK, HV, S, 0, nk0};
        ok = cudaMalloc((void**) &gs_qkv0_, (size_t) max_t_ * s0.c() * 4) == cudaSuccess &&
             cudaMalloc((void**) &gs_z0_, (size_t) max_t_ * s0.zv() * 4) == cudaSuccess;
        own_bytes += (size_t) max_t_ * (s0.c() + s0.zv()) * 4;
    }
    if (!ok) {
        cudaGetLastError();
        gs_free();
        basalt::log::warn("basalt verify", "the GDN mixer stays whole on CUDA0: no room for the peer's share");
        return;
    }
    gs_nk0_ = nk0;
    basalt::log::info("basalt verify",
                      "the GDN mixer is split by key head: CUDA0 %d of %d key heads (%.0f MiB), the peer %d (%.0f MiB)",
                      nk0, HK, own_bytes / 1048576.0, HK - nk0, peer_bytes / 1048576.0);
}

void Verifier::gs_free() {
    if (gs_dev_ >= 0) {
        const OnDevice on(gs_dev_);
        for (GsLayer& L : gs_layers_)
            for (void* p : {L.wqkv, L.wz, L.wo, (void*) L.conv_w, (void*) L.dt, (void*) L.ssm_a, (void*) L.gamma,
                            (void*) L.wa, (void*) L.wb})
                if (p) cudaFree(p);
        if (gs_state_) cudaFree(gs_state_);
        if (gs_mix_) cudaFree(gs_mix_);
        if (gs_pxq_) cudaFree(gs_pxq_);
        if (gs_mixl_) cudaFree(gs_mixl_);
        if (gs_pscr_) cudaFree(gs_pscr_);
        const OnDevice on0(device_);
        for (GsLayer& L : gs_layers_)
            for (void* p : {L.wqkv0, L.wz0, L.wo0})
                if (p) cudaFree(p);
        if (gs_qkv0_) cudaFree(gs_qkv0_);
        if (gs_z0_) cudaFree(gs_z0_);
    }
    for (void* h : {(void*) gs_h_xq_, (void*) gs_h_x_, (void*) gs_h_part_, (void*) gs_h_full_, (void*) gs_h_slice_})
        if (h) cudaFreeHost(h);
    gs_h_full_ = gs_h_slice_ = nullptr;
    gs_h_x_ = gs_m_x_ = gs_h_part_ = gs_m_part_ = nullptr;
    gs_mixl_ = nullptr; gs_pscr_ = nullptr; gs_qkv0_ = gs_z0_ = nullptr;
    gs_live_ = gs_split_ = false;
    gs_ss_ = nullptr;
    gs_layers_.clear();
    gs_state_ = nullptr;
    gs_mix_ = nullptr; gs_pxq_ = nullptr; gs_h_xq_ = nullptr; gs_m_xq_ = nullptr;
    gs_nk0_ = 0;
    gs_dev_ = -1;
}

bool Verifier::gs_move(bool to_peer, std::string& err) {
    using basalt::core::GdnHeadSlice;
    const ModelGeometry& g = *g_;
    const int HK = (int) g.ssm_k_heads, HV = (int) g.ssm_v_heads, S = (int) g.ssm_state_size, HIST = (int) g.ssm_d_conv - 1;
    const GdnHeadSlice sl{HK, HV, S, gs_nk0_, HK - gs_nk0_};
    const size_t full_f = (size_t) S * HV * S + (size_t) g.ssm_conv_channels * HIST;
    const size_t part_f = basalt::core::gdn_slice_state_floats(sl, HIST), nl = (size_t) gs_ca_.n_layers;
    SessionState& ss = to_peer ? *ss_ : *gs_ss_;
    const auto t0 = std::chrono::steady_clock::now();
    int64_t gdn_index = 0;
    for (int64_t l = 0; l < lb_; ++l) gdn_index += is_qsa_layer(g, l) ? 0 : 1;
    float* const full_d = ss.gdn_state + (size_t) (gdn_index - ss.gdn_ord0) * full_f;
    float* const full = gs_h_full_;
    float* const part = gs_h_slice_;
    const size_t full_n = full_f * nl, part_n = part_f * nl;
    bool ok;
    {
        const OnDevice on(device_);
        ok = cudaStreamSynchronize(cs_) == cudaSuccess && cudaStreamSynchronize(cm_) == cudaSuccess &&
             cudaMemcpy(full, full_d, full_n * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    if (to_peer) {
        for (size_t i = 0; i < nl; ++i) basalt::core::gdn_slice_state(sl, HIST, full + full_f * i, part + part_f * i);
        const OnDevice on(gs_dev_);
        ok = ok && cudaStreamSynchronize(dm_->ps) == cudaSuccess &&
             cudaMemcpy(gs_state_, part, part_n * 4, cudaMemcpyHostToDevice) == cudaSuccess;
        gs_ss_ = ss_;
    } else {
        {
            const OnDevice on(gs_dev_);
            ok = ok && cudaStreamSynchronize(dm_->ps) == cudaSuccess &&
                 cudaMemcpy(part, gs_state_, part_n * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
        }
        for (size_t i = 0; i < nl; ++i) basalt::core::gdn_unslice_state(sl, HIST, part + part_f * i, full + full_f * i);
        const OnDevice on(device_);
        ok = ok && cudaMemcpy(full_d, full, full_n * 4, cudaMemcpyHostToDevice) == cudaSuccess;
    }
    if (!ok) {
        err = std::string("verify: the GDN state slice did not move: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    gs_live_ = to_peer;
    if (diagnostics().decode_timing)
        basalt::log::info("basalt verify", "the GDN state slice moved %s in %.1f ms (%.0f MiB)", to_peer ? "to the peer" : "back",
                          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
                          (double) (full_n + part_n) * 4 / (1 << 20));
    return true;
}

}  // namespace basalt::core
