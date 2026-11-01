#include "state.hpp"
#include "layout.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/prefill/prefill.hpp"
#include "basalt/core/mtp.hpp"
#include "basalt/core/layout.hpp"
#include "basalt/kernels/ngram.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/kernels/qsa.hpp"
#include "basalt/kernels/kv_stream.hpp"
#include "basalt/kernels/kv_q4.hpp"
#include "basalt/core/layer.hpp"
#include "basalt/kernels/qsa_decode_attn.hpp"
#include "basalt/prefill/gemm.hpp"
#include "basalt/prefill/moe_fused.hpp"
#include "basalt/prefill/moe_fused_iq.hpp"
#include "basalt/prefill/moe_mmq.hpp"
#include "basalt/prefill/moe_fp4.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/prefill/kernels.hpp"
#include "basalt/log.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <memory>

namespace basalt::prefill {
using namespace detail;

namespace detail {
int64_t g_stream_all_min = 6144;
int g_ring_depth = kPrefillRingDeep;
double g_pinned_share = 1.0;
// Whether the fused experts' buffers apply: a streamed chunk of a pack the fused kernels cover.
bool fused_ring() {
    if (!fused::enabled()) return false;
    const basalt::kernels::cpu::ExpertLayout& lay = basalt::kernels::cpu::expert_layout();
    if (!lay.native) return true;
    // EVERY layer: fused_layout() shrinks the MoE buffers to the fused path's needs, so a layer the native kernels do
    // not cover would run MMQ in them at the full chunk and overflow them.  A pack with such a layer keeps MMQ's
    // buffers; its covered layers still take the fused kernels.
    static const bool all = [&lay] {
        for (const auto& f : lay.fmt)
            if (!fused::native_supported(f.gu_type, f.d_type)) return false;
        return !lay.fmt.empty();
    }();
    return all;
}

// the staging pool of a streamed session: every page of one layer (same sequence in init and bytes_needed)
void take_stage(Alloc& o, const core::SessionState& ss, const basalt::kernels::QsaShapes& s,
                basalt::kernels::KvHostPools& st, bool& ok) {
    const core::QsaState& q0 = ss.qsa_states[ss.qsa_primary()];
    if (q0.kv_mode != 1) return;
    const size_t rows = (size_t) q0.n_pages * s.n_head_kv * s.page_size;
    if (q0.kv_q4) {
        st.k_q4 = o.take<uint8_t>(rows * basalt::kernels::kv_q4_bytes_per_head((int) s.head_dim), ok);
        st.v_q4 = o.take<uint8_t>(rows * basalt::kernels::kv_q4_bytes_per_head((int) s.head_dim), ok);
    } else if (q0.kv_int8) {
        st.k_q = o.take<int8_t>(rows * s.head_dim, ok);
        st.v_q = o.take<int8_t>(rows * s.head_dim, ok);
        st.k_scale = o.take<uint16_t>(rows * (s.head_dim / 64), ok);
        st.v_scale = o.take<uint16_t>(rows * (s.head_dim / 64), ok);
    } else {
        st.k_pool = o.take<uint16_t>(rows * s.head_dim, ok);
        st.v_pool = o.take<uint16_t>(rows * s.head_dim, ok);
    }
}
basalt::kernels::QsaAttnPools pools_of(const basalt::kernels::KvHostPools& h, const int32_t* table) {
    basalt::kernels::QsaAttnPools p;
    p.k_pool = h.k_pool; p.v_pool = h.v_pool; p.k_q = h.k_q; p.v_q = h.v_q; p.k_scale = h.k_scale; p.v_scale = h.v_scale;
    p.k_q4 = h.k_q4; p.v_q4 = h.v_q4;
    p.page_table = table;
    return p;
}

// THE ATTENTION HALF AND THE MoE HALF SHARE THEIR BUFFERS.  A layer runs its attention (GDN or QSA), writes it back
// into the residual, and only then its MoE, so the three sets of scratch are never live at once: one region the size
// of the largest holds them all.  That is a large part of what a prompt token costs, which is what lets a chunk
// grow (every expert is streamed once per chunk, so a bigger chunk streams fewer bytes per token).  The sizes are
// counted with the same `take` sequence `init` uses; a mismatch makes `init` fail with "do not fit", never overlap.
// the conv input, which the recurrence reuses as its scratch
uint64_t gdn_qkv_bytes(size_t T) { return std::max<uint64_t>((uint64_t) T * C * 4, gdn_work_bytes((int64_t) T)); }
uint64_t gdn_set_bytes(size_t T) {
    Alloc a; a.count_only = true; bool ok = true;
    a.take<uint8_t>(gdn_qkv_bytes(T), ok); a.take<float>(T * ZV, ok); a.take<float>(T * 2 * HV, ok); a.take<float>(T * HV, ok);
    a.take<float>(T * HV, ok); a.take<float>(T * C, ok); a.take<float>(T * ZV, ok); a.take<uint16_t>(T * ZV, ok);
    return a.used;
}
// the hyper-connection's rows kept between halves: the mix weights, the bottleneck, the normed residual and its scales
uint64_t hc_set_bytes(size_t T) {
    Alloc a; a.count_only = true; bool ok = true;
    a.take<float>(T * D, ok); a.take<float>(T * LR, ok); a.take<uint16_t>(T * LR, ok);
    a.take<uint16_t>(T * D, ok); a.take<float>(T * HC, ok);
    return a.used;
}
uint64_t qsa_set_bytes(size_t T, int64_t cap, int64_t max_blocks, int64_t sel_batch, int64_t attn_batch,
                       const basalt::kernels::QsaShapes& s) {
    Alloc a; a.count_only = true; bool ok = true;
    a.take<float>(T * 512, ok); a.take<float>(T * 512, ok); a.take<float>(T * 12288, ok); a.take<float>(T * ZV, ok);
    a.take<float>(T * 128, ok); a.take<float>(T * 512, ok); a.take<float>(T * ZV, ok); a.take<uint16_t>(T * ZV, ok);
    a.take<int32_t>(T * (size_t) cap, ok);
    a.take<float>((size_t) sel_batch * (size_t) max_blocks, ok);
    a.take<float>((size_t) attn_batch * basalt::kernels::qsa_decode_attn_scratch_floats(cap, s), ok);
    return a.used;
}
const MmqPlan& mmq_plan() {
    static const MmqPlan plan = [] {
        MmqPlan p;
        const auto& lay = basalt::kernels::cpu::expert_layout();
        const bool on = mmq::built();
        const int64_t layers = lay.native ? (int64_t) lay.fmt.size() : lay.n_layers;
        p.layer.assign((size_t) std::max<int64_t>(layers, 0), 0);
        p.fallback = !on || layers <= 0;
        for (int64_t l = 0; on && l < layers; ++l) {
            const int gt = lay.native ? lay.fmt[(size_t) l].gu_type : 42, dt = lay.native ? lay.fmt[(size_t) l].d_type : 42;
            // a tile on every GPU for these shapes (gate+up: 1280 rows, down: N rows), else the FP16 path
            if (!mmq::fits(gt, 1280) || !mmq::fits(dt, N)) {
                // Q6_K experts never take the FP16 path: a build without their MMQ instance refuses the pack
                if (gt == 14 || dt == 14) {
                    basalt::log::error("basalt", "prefill: layer %lld has Q6_K experts and no MMQ kernel takes them", (long long) l);
                    std::exit(1);
                }
                p.fallback = true;
                continue;
            }
            p.layer[(size_t) l] = 1;
            p.any = true;
            p.gu_max = std::max(p.gu_max, mmq::matrix_bytes(gt, 1280, N));
            p.d_max = std::max(p.d_max, mmq::matrix_bytes(dt, N, 640));
        }
        return p;
    }();
    return plan;
}
// The scale slots of a group buffer (see MMQ_SCALES)
float* grp_scales(uint8_t* grp_d) { return (float*) (grp_d + MMQ_GROUP * mmq_plan().d_max + MMQ_TAIL); }
// A layout whose chunks run the fused experts (a streamed chunk of stream_all_min() tokens or more).  Its GU, H and Xq hold only the fused path's grouping tables, int8 H and per-token
// int8 activations, and Hq nothing: far less than MMQ's FP32 GU / H and per-slot q8_1 rows, which is what
// lets a bigger chunk or ring fit in the slots the prompt path borrows.  A last chunk below stream_all_min() still runs
// MMQ in the same buffers, so each keeps MMQ's size for stream_all_min() - 1 tokens.  `src`: the layout streams experts
// (Prefill::init got an ExpertSource; without one no chunk takes the streamed walk, so no chunk is fused).
bool fused_layout(size_t T, bool src) {
    return src && fused_ring() && mmq_plan().any && ring_slots(T) > STAGE && (int64_t) T >= stream_all_min();
}
// The MoE buffers MMQ and the fused path share: GU and H in floats, Xq and Hq in bytes.  Without `fused` (the
// default): MMQ's, for T tokens.
struct MoeBufs { size_t gu, h, xq, hq; };
// An NVFP4 pack's prompt experts run on the FP4 tensor cores: the activation planes, token scales and tile tables sit
// in Xq (the layer's rows) and Hq (one group's H, a group of up to MMQ_GROUP experts).
bool fp4_pack() {
    static const bool any = [] {
        const auto& lay = basalt::kernels::cpu::expert_layout();
        if (!lay.native) return false;
        for (const auto& f : lay.fmt)
            if (f.gu_type == GGML_TYPE_NVFP4 && f.d_type == GGML_TYPE_NVFP4 && f.tail != 0) return true;
        return false;
    }();
    return any;
}
size_t fp4_x_bytes(size_t rows, int64_t ne) {
    return al256(fp4::a4_bytes((int64_t) rows, N, (int) ne)) + al256(fp4::scale_count((int64_t) rows, (int) ne) * 4) +
           al256((size_t) (ne + 1) * 4);
}
size_t fp4_h_bytes(size_t rows) {
    return al256(fp4::a4_bytes((int64_t) rows, 640, MMQ_GROUP)) + al256(fp4::scale_count((int64_t) rows, MMQ_GROUP) * 4) +
           al256((size_t) 2 * (MMQ_GROUP + 1) * 4);
}
// Every layer's experts are NVFP4 on the FP4 path, so a chunk of FP4_MIN_TOKENS or more never runs MMQ.
bool fp4_every_layer() {
    static const bool all = [] {
        const auto& lay = basalt::kernels::cpu::expert_layout();
        const MmqPlan& mp = mmq_plan();
        if (!lay.native || lay.fmt.empty() || !mp.any || mp.layer.size() < lay.fmt.size()) return false;
        for (size_t l = 0; l < lay.fmt.size(); ++l)
            if (!mp.layer[l] || lay.fmt[l].gu_type != GGML_TYPE_NVFP4 || lay.fmt[l].d_type != GGML_TYPE_NVFP4 ||
                lay.fmt[l].tail == 0)
                return false;
        return true;
    }();
    return all;
}
// Rows of GU and H a group's products hold at once.  The FP4 path indexes them by the group's own rows and splits a
// group above this cap (an expert has at most T rows); a chunk below FP4_MIN_TOKENS runs MMQ, which indexes them by
// the layer's absolute row, so the cap also covers that chunk's rows.
size_t group_rows_cap(size_t T) {
    if (!(fp4_every_layer() && (int64_t) T >= FP4_MIN_TOKENS)) return T * K;
    return std::min(T * K, std::max(T, (size_t) (FP4_MIN_TOKENS - 1) * K));
}
MoeBufs moe_bufs(size_t T, int64_t n_expert, bool fused) {
    if (!fused) {
        // MMQ's int8 planes are only needed for rows a chunk below FP4_MIN_TOKENS runs; the FP4 planes cover the rest
        const size_t qt = fp4_every_layer() && (int64_t) T >= FP4_MIN_TOKENS ? (size_t) (FP4_MIN_TOKENS - 1) : T;
        const size_t gr = group_rows_cap(T);
        size_t xq = mmq::q8_bytes((int64_t) (qt * K), N), hq = mmq::q8_bytes((int64_t) (qt * K), 640);
        if (fp4_pack()) { xq = std::max(xq, fp4_x_bytes(T * K, n_expert)); hq = std::max(hq, fp4_h_bytes(gr)); }
        return {gr * 1280, gr * 640, xq, hq};
    }
    const size_t ts = (size_t) std::min<int64_t>((int64_t) T, stream_all_min() - 1);   // MMQ's last small chunk
    return {std::max(ts * K * 1280, (fused::group_bytes((int64_t) (T * K), (int) n_expert) + 3) / 4),
            std::max(ts * K * 640, (fused::act_bytes((int64_t) (T * K), 640) + 3) / 4),
            std::max(mmq::q8_bytes((int64_t) (ts * K), N), fused::act_bytes((int64_t) T, N)),
            mmq::q8_bytes((int64_t) (ts * K), 640)};
}
Fp4Ptrs fp4_ptrs(void* xq, void* hq, int64_t T, int64_t n_expert) {
    const size_t rows = (size_t) (T * K), ne = (size_t) n_expert;
    Fp4Ptrs p;
    uint8_t* x = (uint8_t*) xq;
    p.a4x = x;
    p.gsx = (float*) (x + al256(fp4::a4_bytes((int64_t) rows, N, (int) ne)));
    uint8_t* h = (uint8_t*) hq;
    p.a4h = h;
    const size_t hrows = group_rows_cap((size_t) T);
    p.gsh = (float*) (h + al256(fp4::a4_bytes((int64_t) hrows, 640, MMQ_GROUP)));
    p.toh = (int32_t*) ((uint8_t*) p.gsh + al256(fp4::scale_count((int64_t) hrows, MMQ_GROUP) * 4));
    p.rel = p.toh + (MMQ_GROUP + 1);
    return p;
}
uint64_t moe_set_bytes(size_t T, int64_t n_expert, bool fused) {
    const MmqPlan& mp = mmq_plan();
    const MoeBufs mb = moe_bufs(T, n_expert, fused);
    Alloc a; a.count_only = true; bool ok = true;
    a.take<float>(T * n_expert, ok); a.take<float>(T * K, ok); a.take<int32_t>(T * K, ok); a.take<int32_t>(T * K, ok);
    a.take<int32_t>(T * K, ok);
    if (mp.fallback) a.take<uint16_t>(T * K * N, ok);
    a.take<float>(mb.gu, ok);
    if (mp.fallback) a.take<uint16_t>(T * K * 640, ok);
    a.take<float>(T * K * N, ok); a.take<float>(T * 640, ok);
    a.take<float>(T * 640, ok); a.take<uint16_t>(T * 640, ok); a.take<float>(T * N, ok); a.take<float>(T, ok);
    if (mp.any) {
        a.take<uint8_t>(mb.xq, ok);
        a.take<float>(mb.h, ok);
        a.take<uint8_t>(mb.hq, ok);
    }
    return a.used;
}

}

bool Prefill::carve(size_t T, void* alloc) {
    Impl& m = *impl_;
    Alloc& o = *static_cast<Alloc*>(alloc);
    const core::ModelGeometry& g = *m.g;
    core::SessionState& ss = *m.ss;
    bool ok = true;
    m.R = o.take<float>(T * D, ok);
    m.inj = o.take<float>(T * HC, ok);
    m.mixed = o.take<float>(T * N, ok); m.mixed_bf = o.take<uint16_t>(T * N, ok);
    m.mixed_h = o.take<uint16_t>(T * N, ok); m.bo = o.take<float>(T * N, ok);
    m.steps_dev = o.take<int32_t>(T * basalt::kernels::kStepCount, ok);
    basalt::kernels::QsaShapes s = basalt::kernels::qsa_real_shapes();
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    m.cap = basalt::kernels::qsa_selection_width(basalt::kernels::kTopkMaxCells, s);
    m.max_blocks = ss.qsa_states[ss.qsa_primary()].max_cells / s.idx_block + 2;
    {
        // one region for the attention half's and the MoE half's scratch (see gdn_set_bytes)
        const bool fz = fused_layout(T, m.src != nullptr);
        const MoeBufs mb = moe_bufs(T, m.g->n_expert, fz);
        const uint64_t region = std::max({gdn_set_bytes(T), qsa_set_bytes(T, m.cap, m.max_blocks, m.sel_batch,
                                                                           m.attn_batch, s), moe_set_bytes(T, m.g->n_expert, fz),
                                          hc_set_bytes(T)});
        uint8_t* base = o.take<uint8_t>((size_t) region, ok);
        m.region = base;
        m.region_bytes = region;
        {
            // the hyper-connection's rows are only live between one half's end and the next half's mix, when the
            // region holds no scratch, and the embedding rows before the first layer's: they sit at its start
            Alloc h;
            h.base = base; h.cap = region; h.owned = &m.owned;
            m.gated = h.take<float>(T * D, ok); m.lo = h.take<float>(T * LR, ok); m.lo16 = h.take<uint16_t>(T * LR, ok);
            m.xn16 = h.take<uint16_t>(T * D, ok); m.grs = h.take<float>(T * HC, ok);
            m.emb = m.gated;
        }
        Alloc a;
        a.base = base; a.cap = region; a.owned = &m.owned;
        m.qkv = (float*) a.take<uint8_t>(gdn_qkv_bytes(T), ok); m.z = a.take<float>(T * ZV, ok); m.ab = a.take<float>(T * 2 * HV, ok);
        m.gate = a.take<float>(T * HV, ok); m.beta = a.take<float>(T * HV, ok); m.hbuf = a.take<float>(T * C, ok);
        m.y = a.take<float>(T * ZV, ok); m.y_h = a.take<uint16_t>(T * ZV, ok);
        Alloc b;
        b.base = base; b.cap = region; b.owned = &m.owned;
        m.Kc = b.take<float>(T * 512, ok); m.Vc = b.take<float>(T * 512, ok); m.Qf = b.take<float>(T * 12288, ok);
        m.q = b.take<float>(T * ZV, ok); m.idx_raw = b.take<float>(T * 128, ok); m.q_idx = b.take<float>(T * 512, ok);
        m.attn = b.take<float>(T * ZV, ok); m.attn_h = b.take<uint16_t>(T * ZV, ok);
        m.sel_ids = b.take<int32_t>(T * (size_t) m.cap, ok);
        m.sel_scores = b.take<float>((size_t) m.sel_batch * (size_t) m.max_blocks, ok);
        m.attn_scratch = b.take<float>((size_t) m.attn_batch * basalt::kernels::qsa_decode_attn_scratch_floats(m.cap, s), ok);
        Alloc c;
        c.base = base; c.cap = region; c.owned = &m.owned;
        m.logits = c.take<float>(T * m.g->n_expert, ok); m.w = c.take<float>(T * K, ok); m.ids = c.take<int32_t>(T * K, ok);
        m.slot_dev = c.take<int32_t>(T * K, ok); m.src_dev = c.take<int32_t>(T * K, ok);
        const MmqPlan& mp = mmq_plan();
        m.Xs = mp.fallback ? c.take<uint16_t>(T * K * N, ok) : nullptr;
        m.GU = c.take<float>(mb.gu, ok);
        m.Hh = mp.fallback ? c.take<uint16_t>(T * K * 640, ok) : nullptr;
        m.Dm = c.take<float>(T * K * N, ok);
        m.sgate = c.take<float>(T * 640, ok); m.sup = c.take<float>(T * 640, ok); m.sh_h = c.take<uint16_t>(T * 640, ok);
        m.shared = c.take<float>(T * N, ok); m.sg = c.take<float>(T, ok);
        if (mp.any) {
            m.Xq = c.take<uint8_t>(mb.xq, ok);
            m.H = c.take<float>(mb.h, ok);
            m.Hq = c.take<uint8_t>(mb.hq, ok);
        }
        if (base == nullptr) ok = false;
    }
    for (int i = 0; i < DQ; ++i) { m.dq_gu[i] = o.take<uint16_t>(1280 * 2560, ok); m.dq_d[i] = o.take<uint16_t>(2560 * 640, ok); }
    if (mmq_plan().any) {
        const MmqPlan& mp = mmq_plan();
        m.ids_identity = o.take<int32_t>(T * K, ok);
        m.bounds_dev = o.take<int32_t>((size_t) (2 * (m.g->n_expert + m.g->n_expert / MMQ_GROUP + 2)), ok);
        m.grp_gu = o.take<uint8_t>(MMQ_GROUP * mp.gu_max + MMQ_TAIL, ok);
        m.grp_d = o.take<uint8_t>(MMQ_GROUP * mp.d_max + MMQ_TAIL + MMQ_SCALES, ok);
        // (written at every run's start, not here: when serving, these are live expert-cache slots until a request
        // lends them - a write now would corrupt a resident expert)
        if (!m.mmq_ctx) m.mmq_ctx = std::make_unique<mmq::Context>();
    }
    m.ring = ring_slots(T);
    for (int i = 0; i < m.ring; ++i) {
        m.stage_dev[i] = o.take<uint8_t>((size_t) MAXBLOB(), ok);
        m.stage_live[i] = false;                        // a new buffer: nothing of an earlier layout to wait for
        m.used_of[i] = i;
    }
    m.ple_emb = o.take<float>(T * N, ok);
    m.ple_norm = o.take<float>((size_t) basalt::kernels::NG_HC_DIM, ok);
    take_stage(o, ss, s, m.stage, ok);
    m.T = (int64_t) T;
    return ok;
}

bool Prefill::relayout(int64_t chunk, void* borrow, uint64_t borrow_bytes, std::string& err) {
    Impl& m = *impl_;
    if (!m.borrowed || borrow == nullptr || chunk <= 0 || chunk > m.T_max) {
        err = "prefill: relayout needs borrowed buffers and a chunk of at most " + std::to_string(m.T_max);
        return false;
    }
    if (cudaStreamSynchronize(m.cs) != cudaSuccess || cudaStreamSynchronize(m.copy) != cudaSuccess) {
        err = "prefill: relayout: the stream failed";
        return false;
    }
    bool ok = true;
    Alloc o;
    o.base = (uint8_t*) borrow;
    o.cap = borrow_bytes;
    o.owned = &m.owned;
    uint16_t* gs = o.take<uint16_t>((size_t) GEMM_SCRATCH, ok);
    void* ws = o.take<uint8_t>(GEMM_WS, ok);
    if (ok) m.gemm.rebind(gs, GEMM_SCRATCH, ws, GEMM_WS);
    if (!ok || !carve((size_t) chunk, &o)) {
        err = "prefill: device buffers for a chunk of " + std::to_string(chunk) + " tokens do not fit";
        return false;
    }
    return true;
}

int64_t Prefill::chunk() const { return impl_->T; }

void Prefill::set_pinned_share(double share) { g_pinned_share = share; }
void Prefill::set_stream_all_min(int64_t tokens) { g_stream_all_min = tokens; }
void Prefill::set_cpu_pool(kernels::cpu::ExpertPool* pool) { cpu_pool_ = pool; }
void Prefill::set_cpu_share(double share) { impl_->cpu_share_now = std::max(0.0, std::min(share, 1.0)); }
void Prefill::set_ring_depth(int slots) { g_ring_depth = std::max(STAGE + 1, std::min(slots, kPrefillRingDeep)); }
int Prefill::ring_depth() const { return impl_->ring; }
double Prefill::pinned_share() { return g_pinned_share; }
bool Prefill::fp4_experts() { return fp4_pack(); }

namespace { constexpr uint64_t kSizingSlack = 8u << 20; }   // alignment slack of a borrowed region

uint64_t Prefill::bytes_needed(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk) {
    // the same allocation sequence as `init`, counted
    const size_t T = (size_t) chunk;
    bool ok = true;
    Alloc o;
    o.count_only = true;
    o.take<uint16_t>((size_t) GEMM_SCRATCH, ok);
    o.take<uint8_t>(GEMM_WS, ok);
    auto f = [&](size_t n) { o.take<float>(n, ok); };
    // `carve`'s order, buffer for buffer: R, inj, mixed, mixed_bf, mixed_h, bo (the hyper-connection rows live in the region).
    f(T * D);
    f(T * HC); f(T * N); o.take<uint16_t>(T * N, ok); o.take<uint16_t>(T * N, ok); f(T * N);
    o.take<int32_t>(T * basalt::kernels::kStepCount, ok);
    basalt::kernels::QsaShapes s = basalt::kernels::qsa_real_shapes();
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    const int64_t cap = basalt::kernels::qsa_selection_width(basalt::kernels::kTopkMaxCells, s);
    const int64_t max_blocks = ss.qsa_states[ss.qsa_primary()].max_cells / s.idx_block + 2;
    o.take<uint8_t>((size_t) std::max({gdn_set_bytes(T), qsa_set_bytes(T, cap, max_blocks, 256, 32, s),
                                       moe_set_bytes(T, g.n_expert, fused_layout(T, true)),
                                       hc_set_bytes(T)}), ok);
    for (int i = 0; i < DQ; ++i) { o.take<uint16_t>(1280 * 2560, ok); o.take<uint16_t>(2560 * 640, ok); }
    if (mmq_plan().any) {
        const MmqPlan& mp = mmq_plan();
        o.take<int32_t>(T * K, ok);
        o.take<int32_t>((size_t) (2 * (g.n_expert + g.n_expert / MMQ_GROUP + 2)), ok);
        o.take<uint8_t>(MMQ_GROUP * mp.gu_max + MMQ_TAIL, ok);
        o.take<uint8_t>(MMQ_GROUP * mp.d_max + MMQ_TAIL + MMQ_SCALES, ok);
    }
    for (int i = 0; i < ring_slots(T); ++i) o.take<uint8_t>((size_t) MAXBLOB(), ok);
    f(T * N);
    f((size_t) basalt::kernels::NG_HC_DIM);
    basalt::kernels::KvHostPools stage;
    take_stage(o, ss, s, stage, ok);
    return o.used + kSizingSlack;
}

bool Prefill::layout_fits(const core::ModelGeometry& g, core::SessionState& ss, int64_t chunk, bool with_source,
                          uint64_t* used, uint64_t* capacity) {
    Prefill p;
    Impl& m = *p.impl_;
    m.g = &g;
    m.ss = &ss;
    m.src = with_source ? reinterpret_cast<core::ExpertSource*>(alignof(std::max_align_t)) : nullptr;   // only tested for null
    const uint64_t cap = bytes_needed(g, ss, chunk) - kSizingSlack;   // the layout must fit the count itself
    bool ok = true;
    Alloc o;
    o.base = reinterpret_cast<uint8_t*>(uint64_t{1} << 20);   // never dereferenced
    o.cap = cap;
    o.owned = &m.owned;
    o.take<uint16_t>((size_t) GEMM_SCRATCH, ok);
    o.take<uint8_t>(GEMM_WS, ok);
    ok = p.carve((size_t) chunk, &o) && ok;
    if (used) *used = o.used;
    if (capacity) *capacity = cap;
    return ok && o.used <= cap;
}

int64_t Prefill::ring_slots_for(int64_t chunk) { return ring_slots((size_t) chunk); }

}  // namespace basalt::prefill
