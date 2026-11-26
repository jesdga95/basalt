// src/models/inkling2/dflash2.cpp - Inkling 2's embedded DFlash 2 block-diffusion drafter (see dflash2.hpp).
#include "models/inkling2/dflash2.hpp"

#include "basalt/core/device_copy.hpp"
#include "basalt/kernels/dense_ops.hpp"
#include "basalt/kernels/gqa_attn.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/nvfp4_planes.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/row_norm.hpp"
#include "basalt/kernels/window_gemm.hpp"
#include "basalt/log.hpp"
#include "basalt/models/inkling2/dflash2_kernels.hpp"
#include "basalt/models/inkling2/dspark_kernels.hpp"
#include "basalt/models/inkling2/kernels.hpp"
#include "basalt/prefill/moe_mmq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace basalt::models::inkling2 {
namespace {

namespace bk = basalt::kernels;

constexpr int kTypeF32 = 0;
constexpr int kTypeQ8_0 = 8;
constexpr int kTypeBF16 = 30;
constexpr int kE = kWidth;
constexpr int kSlice = 8;
constexpr int kSoloCommitMax = 16;
constexpr int kRingExtra = 64;
constexpr int kFp4Mats = 7;   // q, k, v, o, gate, up, down

uint64_t q8_bytes(uint64_t rows, uint64_t n_in) { return rows * (n_in / 32) * 34; }

int64_t ring_cells(const Dflash2Geometry& d) { return (int64_t) d.window + kRingExtra; }

void ws_need(const Dflash2Geometry& d, int n_vocab, bool head_fp4, size_t& partial, size_t& tickets) {
    namespace wg = bk::wgemm;
    partial = tickets = 0;
    const auto need = [&](wg::Format fmt, int64_t K, std::initializer_list<int64_t> rows, wg::Epilogue epi) {
        wg::Call c;
        c.fmt = fmt;
        c.K = K;
        c.epi = epi;
        c.n = 0;
        for (int64_t r : rows) c.m[c.n++].rows = r;
        size_t p = 0, t = 0;
        wg::workspace_need(c, p, t);
        partial = std::max(partial, p);
        tickets = std::max(tickets, t);
    };
    need(wg::Format::Q8, kDf2Targets * kE, {kE}, wg::Epilogue::Store);
    need(wg::Format::Nvfp4, kE, {kDf2KV, kDf2KV}, wg::Epilogue::Store);
    need(wg::Format::Q8, kE, {kDf2Dyn}, wg::Epilogue::Store);
    need(wg::Format::Nvfp4, kE, {kDf2Q, kDf2KV, kDf2KV}, wg::Epilogue::Store);
    need(wg::Format::Nvfp4, kDf2Q, {kE}, wg::Epilogue::Store);
    need(wg::Format::Nvfp4, kE, {d.n_ff, d.n_ff}, wg::Epilogue::SwiGLU);
    need(wg::Format::Nvfp4, d.n_ff, {kE}, wg::Epilogue::Store);
    need(head_fp4 ? wg::Format::Nvfp4 : wg::Format::Q8, kE, {n_vocab}, wg::Epilogue::Store);
    need(wg::Format::Bf16, kE, {kDf2Rank}, wg::Epilogue::Store);
}

void window_range(const Dflash2Geometry& d, int64_t pos, int64_t lo, int64_t& win, bool& block_win) {
    if (lo <= 0 || pos - lo >= (int64_t) d.window - 1) {
        win = d.window;
        block_win = false;
    } else {
        win = std::max<int64_t>(1, pos - lo + 1);
        block_win = true;
    }
}

}  // namespace

// ---- weights ----------------------------------------------------------------------------------------------------------

Dflash2Weights::~Dflash2Weights() {
    for (void* p : allocs_) cudaFree(p);
}

void* Dflash2Weights::dev(size_t bytes, bool& ok) {
    void* p = nullptr;
    if (!ok || cudaMalloc(&p, std::max<size_t>(bytes, 16)) != cudaSuccess) {
        ok = false;
        return nullptr;
    }
    allocs_.push_back(p);
    bytes_ += bytes;
    return p;
}

model::Result dflash2_plan(const Pack& pack, core::LoadPlan& reads, int file, int device, std::unique_ptr<Dflash2Weights>& out) {
    out.reset();
    const Dflash2Geometry& d = pack.dflash2;
    if (!d.present) return {};
    const Geometry& g = pack.geo;
    auto fail = [](const std::string& e) { return model::Result{model::Status::Error, "inkling2: dflash2 drafter: " + e}; };

    if (d.n_layers < 1 || d.n_layers > kDf2MaxLayers || d.block != kDf2Block || d.n_head != kDf2Heads ||
        d.n_head_kv != kDf2KvHeads || d.head_dim != kDf2HeadDim || d.conv_kernel != 2 || d.conv_group != kDf2ConvGroup ||
        d.selector_rank != kDf2Rank || d.selector_top_k != kDf2TopK || d.n_ff <= 0 || d.n_ff % 256 || d.window <= 0 ||
        d.mask_id < 0 || d.mask_id >= g.n_vocab || (int) d.targets.size() != kDf2Targets || g.n_embd != kE) {
        return fail("a DFlash 2 layout this engine does not run (width 2304, 1..8 layers of 32 query and 8 KV heads of 128, "
                    "block 8, 2-tap group-16 convs, rank-256 top-16 selector, 4 target features)");
    }
    for (int k = 0; k + 1 < kDf2Targets; ++k)
        if (d.targets[(size_t) k] < 0 || d.targets[(size_t) k] >= (int) g.visits.size() ||
            (k > 0 && d.targets[(size_t) k] <= d.targets[(size_t) k - 1]))
            return fail("its target visits are not in order within the target's visits");
    if (d.targets.back() != (int) g.visits.size())
        return fail("its last target is not the target's final normed state");

    auto w = std::make_unique<Dflash2Weights>();
    w->geo = d;
    w->n_vocab = g.n_vocab;
    w->target_eps = g.rms_eps;
    bool ok = true;
    std::string err;

    auto load = [&](const std::string& name, int type, uint64_t want_bytes) -> void* {
        if (!ok) return nullptr;
        const TensorRef* t = pack.find(name);
        if (!t || t->type != type || t->bytes != want_bytes) {
            err = name + (t ? " has another type or size" : " is missing");
            ok = false;
            return nullptr;
        }
        void* p = w->dev(want_bytes, ok);
        if (!ok) {
            err = "out of device memory for " + name;
            return nullptr;
        }
        core::LoadItem it;
        it.file = file;
        it.off = t->offset;
        it.bytes = t->bytes;
        it.dev = p;
        it.device = device;
        it.conv = core::Conv::Copy;
        reads.add(it);
        return p;
    };
    auto f32 = [&](const std::string& name, uint64_t n) { return (float*) load(name, kTypeF32, n * 4); };
    auto bf16 = [&](const std::string& name, uint64_t n) { return (uint16_t*) load(name, kTypeBF16, n * 2); };
    auto q8 = [&](const std::string& name, uint64_t rows, uint64_t n_in) {
        return load(name, kTypeQ8_0, q8_bytes(rows, n_in));
    };

    w->fc_q8 = q8("dflash2.fc.weight", kE, (uint64_t) kDf2Targets * kE);
    w->fc_p = w->dev(q8_bytes(kE, (uint64_t) kDf2Targets * kE), ok);
    w->hidden_norm = f32("dflash2.hidden_norm.weight", kE);
    w->norm = f32("dflash2.norm.weight", kE);
    w->sel_h = bf16("dflash2.selector.hidden.weight", (uint64_t) kDf2Rank * kE);
    w->pred = bf16("dflash2.selector.pred", (uint64_t) g.n_vocab * kDf2Rank);
    w->succ = bf16("dflash2.selector.succ", (uint64_t) g.n_vocab * kDf2Rank);
    w->mask = (float*) w->dev((size_t) kE * 4, ok);

    w->layer.assign((size_t) d.n_layers, Dflash2Layer{});
    for (int l = 0; l < d.n_layers; ++l) {
        Dflash2Layer& L = w->layer[(size_t) l];
        const std::string b = "dflash2.blk." + std::to_string(l) + ".";
        L.wq = q8(b + "attn_q.weight", kDf2Q, kE);
        L.wk_q8 = q8(b + "attn_k.weight", kDf2KV, kE);
        L.wv_q8 = q8(b + "attn_v.weight", kDf2KV, kE);
        L.wk = w->dev(bk::nvfp4p::bytes(kDf2KV, kE), ok);
        L.wv = w->dev(bk::nvfp4p::bytes(kDf2KV, kE), ok);
        L.wo = q8(b + "attn_output.weight", kE, kDf2Q);
        L.gate = q8(b + "ffn_gate.weight", (uint64_t) d.n_ff, kE);
        L.up = q8(b + "ffn_up.weight", (uint64_t) d.n_ff, kE);
        L.down = q8(b + "ffn_down.weight", kE, (uint64_t) d.n_ff);
        L.conv_a = q8(b + "attn_conv.weight", kDf2Dyn, kE);
        L.conv_f = q8(b + "ffn_conv.weight", kDf2Dyn, kE);
        L.base_a = f32(b + "attn_conv.base", 4 * kE);
        L.base_f = f32(b + "ffn_conv.base", 4 * kE);
        L.attn_norm = f32(b + "attn_norm.weight", kE);
        L.post_norm = f32(b + "post_attention_norm.weight", kE);
        L.q_norm = f32(b + "attn_q_norm.weight", kDf2HeadDim);
        L.k_norm = f32(b + "attn_k_norm.weight", kDf2HeadDim);
        L.fp4_scale = (float*) w->dev(kFp4Mats * 4, ok);
    }

    if (bk::nvfp4p::fits(g.n_vocab, kE)) {
        w->head_fp4 = w->dev(bk::nvfp4p::bytes(g.n_vocab, kE), ok);
        w->head_fp4_scale = (float*) w->dev(4, ok);
    }
    if (!ok) return fail(err.empty() ? "out of device memory" : err);

    basalt::log::info("basalt inkling2",
                      "drafter: DFlash 2, %d non-causal sliding-window layers (window %d) reading target visits %d, %d, %d, "
                      "%.2f GiB on the GPU (NVFP4 block/head/KV planes, Q8 convs & fc, rank-%d selector)",
                      d.n_layers, d.window, d.targets[0], d.targets[1], d.targets[2],
                      (double) w->bytes() / (double) (1ull << 30), d.selector_rank);
    out = std::move(w);
    return {};
}

model::Result Dflash2Weights::finish(const void* head_q8, const uint16_t* emb, const float* en, float teps, void* stream) {
    head = head_q8;
    embedding = emb;
    embed_norm = en;
    target_eps = teps;
    if (finished_) return {};
    const cudaStream_t s = (cudaStream_t) stream;
    try {
        dflash2_make_mask(emb, geo.mask_id, en, target_eps, mask, s);
        const uint64_t ff = (uint64_t) geo.n_ff;
        const size_t tmp_bytes = std::max<size_t>({q8_bytes(kE, (uint64_t) kDf2Targets * kE),
                                                   q8_bytes(ff, kE),
                                                   q8_bytes(kDf2Q, kE)});
        void* tmp = nullptr;
        if (cudaMalloc(&tmp, tmp_bytes) != cudaSuccess)
            throw std::runtime_error("out of device memory converting DFlash 2 planes");

        // fc_p gets Q8 planes [2304][9216] for Tensor Core wgemm; fc_q8 becomes 4 [2304][2304] Q8_0 slices for prompt MMQ.
        bk::wgemm::q8_planes(fc_q8, kE, (int64_t) kDf2Targets * kE, fc_p, s);
        cudaMemcpyAsync(tmp, fc_q8, q8_bytes(kE, (uint64_t) kDf2Targets * kE), cudaMemcpyDeviceToDevice, s);
        dflash2_split_fc_q8(tmp, fc_q8, s);

        if (head_fp4) bk::nvfp4p::from_q8(head_q8, n_vocab, kE, head_fp4, head_fp4_scale, s);

        for (Dflash2Layer& L : layer) {
            bk::nvfp4p::from_q8(L.wk_q8, kDf2KV, kE, L.wk, L.fp4_scale + 1, s);
            bk::nvfp4p::from_q8(L.wv_q8, kDf2KV, kE, L.wv, L.fp4_scale + 2, s);

            cudaMemcpyAsync(tmp, L.conv_a, q8_bytes(kDf2Dyn, kE), cudaMemcpyDeviceToDevice, s);
            bk::wgemm::q8_planes(tmp, kDf2Dyn, kE, L.conv_a, s);
            cudaMemcpyAsync(tmp, L.conv_f, q8_bytes(kDf2Dyn, kE), cudaMemcpyDeviceToDevice, s);
            bk::wgemm::q8_planes(tmp, kDf2Dyn, kE, L.conv_f, s);

            const struct {
                void* w;
                uint64_t rows, K;
                int sc_idx;
            } fp4[5] = {{L.wq, kDf2Q, kE, 0}, {L.wo, kE, kDf2Q, 3}, {L.gate, ff, kE, 4}, {L.up, ff, kE, 5}, {L.down, kE, ff, 6}};
            for (const auto& m : fp4) {
                cudaMemcpyAsync(tmp, m.w, q8_bytes(m.rows, m.K), cudaMemcpyDeviceToDevice, s);
                bk::nvfp4p::from_q8(tmp, (int64_t) m.rows, (int64_t) m.K, m.w, L.fp4_scale + m.sc_idx, s);
            }
        }
        const cudaError_t e = cudaStreamSynchronize(s);
        cudaFree(tmp);
        if (e != cudaSuccess) throw std::runtime_error("converting DFlash 2 planes failed");
    } catch (const std::exception& e) {
        return {model::Status::Error, std::string("inkling2: dflash2 drafter: ") + e.what()};
    }
    finished_ = true;
    return {};
}

uint64_t dflash2_session_bytes(const Pack& pack) {
    const Dflash2Geometry& d = pack.dflash2;
    if (!d.present) return 0;
    return (uint64_t) d.n_layers * 2ull * (uint64_t) ring_cells(d) * kDf2KV * 2 + 4096;
}

const Dflash2Weights::SharedScratch* Dflash2Weights::ensure_shared_scratch(std::string& err) const {
    auto* self = const_cast<Dflash2Weights*>(this);
    SharedScratch& s = self->shared_scr_;
    if (s.x != nullptr) return &s;
    const int64_t cells = ring_cells(geo);
    const size_t E = kE, ff = (size_t) geo.n_ff, R = kDf2Block, D = kDf2Drafts;
    const size_t CR_solo = std::max<size_t>(R, kSoloCommitMax);
    bool ok = true;
    auto alloc_z = [&](size_t bytes) -> void* {
        void* p = self->dev(bytes, ok);
        if (p) cudaMemset(p, 0, std::max<size_t>(bytes, 16));
        return p;
    };
    auto f = [&](size_t n) { return (float*) alloc_z(n * 4); };

    s.att_cells = cells;
    s.tap_bf16 = alloc_z(bk::wgemm::act_bytes((int64_t) kDf2Targets * kE));
    s.c_raw = f(CR_solo * E);
    s.kb = f(CR_solo * kDf2KV);
    s.vb = f(CR_solo * kDf2KV);
    s.ring_bounds = (int32_t*) alloc_z(2 * 4);
    if (ok) {
        const int32_t rb[2] = {0, (int32_t) cells};
        if (basalt::core::device_copy(s.ring_bounds, rb, sizeof(rb), cudaMemcpyHostToDevice) != cudaSuccess) ok = false;
    }

    s.x = f(R * E);
    s.xn = f(R * E);
    s.z = f(R * E);
    s.dyn = f(R * kDf2Dyn);
    s.q = f(R * kDf2Q);
    s.qr = f(R * kDf2Q);
    s.attn = f(R * kDf2Q);
    s.act_a = alloc_z(bk::wgemm::act_bytes(kE));
    s.act_q = alloc_z(bk::wgemm::act_bytes(kDf2Q));
    s.act_h = alloc_z(bk::wgemm::act_bytes((int64_t) ff));
    {
        size_t pf = 0, tk = 0;
        ws_need(geo, n_vocab, head_fp4 != nullptr, pf, tk);
        s.ws.partial = f(std::max<size_t>(pf, 1));
        s.ws.tickets = (int32_t*) alloc_z(std::max<size_t>(tk, 1) * 4);
        s.ws.partial_floats = pf;
        s.ws.tickets_n = tk;
    }
    s.att_scr = f(bk::gqa_decode_scratch_floats(cells, kDf2Block, kDf2Heads, kDf2HeadDim));
    s.logits = f(D * (size_t) n_vocab);
    s.g = f(D * kDf2Rank);
    s.tail_lse = f(D * 2);
    s.hq = alloc_z(bk::native_q8_1_bytes(kE, (int) D));
    s.top_scr = alloc_z(dflash2_top16_scratch_bytes((int) D, n_vocab));
    s.cand = (int32_t*) alloc_z(D * 2 * kDf2TopK * 4);

    if (!ok) {
        err = "out of device memory for the shared DFlash 2 scratch";
        return nullptr;
    }
    return &s;
}

// ---- the session's hook -----------------------------------------------------------------------------------------------

Dflash2Hook::~Dflash2Hook() {
    for (cudaGraphExec_t g : g_stage_)
        if (g) cudaGraphExecDestroy(g);
    for (cudaGraphExec_t g : g_head_)
        if (g) cudaGraphExecDestroy(g);
    for (void* p : allocs_) cudaFree(p);
    for (void* p : pinned_) cudaFreeHost(p);
}

void* Dflash2Hook::dev(size_t bytes, bool& ok) {
    void* p = nullptr;
    if (!ok || cudaMalloc(&p, std::max<size_t>(bytes, 16)) != cudaSuccess) {
        ok = false;
        return nullptr;
    }
    cudaMemset(p, 0, std::max<size_t>(bytes, 16));
    allocs_.push_back(p);
    return p;
}

bool Dflash2Hook::init(const Dflash2Weights* w, const Geometry& g, int64_t context, int tap_rows, float* tap, std::string& err) {
    if (!w) return true;
    const Dflash2Geometry& d = w->geo;
    if (!w->ensure_shared_scratch(err)) return false;
    bool ok = true;
    tap_rows_ = std::max(tap_rows, kSlice);
    context_ = context;
    tap_ = tap;
    slot_.assign(g.visits.size(), -1);
    for (int k = 0; k + 1 < kDf2Targets; ++k) slot_[(size_t) d.targets[(size_t) k]] = k;

    const int64_t cap = ring_cells(d);
    const size_t ring_b = (size_t) cap * kDf2KV * 2;
    void* slab = dev((size_t) d.n_layers * 2 * ring_b, ok);
    cache_.assign((size_t) d.n_layers, Cache{});
    for (int l = 0; ok && l < d.n_layers; ++l) {
        Cache& c = cache_[(size_t) l];
        c.cap = cap;
        auto* base = static_cast<uint8_t*>(slab) + (size_t) l * 2 * ring_b;
        c.k = reinterpret_cast<uint16_t*>(base);
        c.v = reinterpret_cast<uint16_t*>(base + ring_b);
    }

    meta_d_ = (int32_t*) dev(2 * 4, ok);
    ids_d_ = (int32_t*) dev(kDf2Block * 4, ok);
    conf_d_ = (float*) dev(kDf2Block * 4, ok);
    if (ok && cudaMallocHost((void**) &meta_h_, 2 * 4) != cudaSuccess) ok = false;
    if (meta_h_) pinned_.push_back(meta_h_);
    if (ok && cudaMallocHost((void**) &ids_h_, kDf2Block * 4) != cudaSuccess) ok = false;
    if (ids_h_) pinned_.push_back(ids_h_);
    if (ok && cudaMallocHost((void**) &conf_h_, kDf2Block * 4) != cudaSuccess) ok = false;
    if (conf_h_) pinned_.push_back(conf_h_);

    if (!ok) {
        err = "out of device memory for the DFlash 2 session";
        return false;
    }
    w_ = w;
    return true;
}

int Dflash2Hook::slot_of_visit(int visit) const {
    return w_ && visit >= 0 && visit < (int) slot_.size() ? slot_[(size_t) visit] : -1;
}

void Dflash2Hook::commit(const int32_t* pos_d, int rows, void* stream, int row0) {
    if (!w_ || rows <= 0) return;
    namespace wg = bk::wgemm;
    bk::PdlScope pdl(true);
    const Dflash2Geometry& d = w_->geo;
    const auto& s = w_->shared_scr_;
    const auto mat = [](const void* p, int64_t r, const float* scale, float* y, int64_t ld_y) {
        return wg::Matrix{(const uint8_t*) p, r, scale, y, ld_y};
    };

    for (int r0 = 0; r0 < rows; r0 += kSoloCommitMax) {
        const int n = std::min(kSoloCommitMax, rows - r0);
        dflash2_pack_tap_bf16(tap_ + (size_t) (row0 + r0) * kE, (int64_t) tap_rows_ * kE, n, s.tap_bf16, stream);
        wg::Call fc;
        fc.fmt = wg::Format::Q8;
        fc.K = kDf2Targets * kE;
        fc.m[0] = mat(w_->fc_p, kE, nullptr, s.c_raw, kE);
        wg::gemm(fc, s.tap_bf16, n, s.ws, stream);
        wg::norm_pack(s.c_raw, w_->hidden_norm, d.rms_eps, kE, n, s.act_a, nullptr, stream);

        for (int l = 0; l < d.n_layers; ++l) {
            const Dflash2Layer& L = w_->layer[(size_t) l];
            const Cache& c = cache_[(size_t) l];
            wg::Call kv;
            kv.fmt = wg::Format::Nvfp4;
            kv.K = kE;
            kv.n = 2;
            kv.m[0] = mat(L.wk, kDf2KV, L.fp4_scale + 1, s.kb, kDf2KV);
            kv.m[1] = mat(L.wv, kDf2KV, L.fp4_scale + 2, s.vb, kDf2KV);
            wg::gemm(kv, s.act_a, n, s.ws, stream);

            DsCommitSegs segs;
            segs.k[0] = c.k;
            segs.v[0] = c.v;
            dspark_kv_commit_batch(s.kb, s.vb, L.k_norm, d.rms_eps, d.rope_base, pos_d + r0, n, segs, c.cap, stream);
        }
    }
}

void Dflash2Hook::commit_prompt(const int32_t* pos_d, int rows, basalt::prefill::mmq::Context& mmq,
                                const DsparkHook::PromptScratch& ps, void* stream) {
    if (!w_ || rows <= 0) return;
    if (rows <= kSoloCommitMax) {
        commit(pos_d, rows, stream);
        return;
    }
    namespace pm = basalt::prefill::mmq;
    const Dflash2Geometry& d = w_->geo;
    const auto& scr = w_->shared_scr_;
    const int64_t cap = cache_[0].cap;
    const int keep = (int) std::min<int64_t>(rows, cap);
    const int r0 = rows - keep;
    const int32_t* bounds = keep == rows ? ps.bounds : scr.ring_bounds;

    float* acc = ps.mix;
    float* tmp = ps.mix + (size_t) keep * kE;
    const size_t fc_slice_b = q8_bytes(kE, kE);
    for (int k = 0; k < kDf2Targets; ++k) {
        pm::quantize(tap(k) + (size_t) r0 * kE, nullptr, ps.xq, 8, kE, kE, keep, stream);
        pm::Product p;
        p.type = 8;
        p.w_rows = kE;
        p.w_cols = kE;
        p.expert_bytes = pm::matrix_bytes(8, kE, kE);
        p.n = 1;
        p.xq = ps.xq;
        p.bounds = bounds;
        p.ids = ps.ident;
        p.total_rows = keep;
        p.max_rows = keep;
        p.ld_dst = kE;
        p.w = static_cast<const uint8_t*>(w_->fc_q8) + (size_t) k * fc_slice_b;
        p.dst = k == 0 ? acc : tmp;
        mmq.run(p, stream);
        if (k > 0) bk::add_scaled(acc, tmp, nullptr, (int64_t) keep * kE, stream);
    }
    bk::rms_norm_rows(acc, w_->hidden_norm, keep, kE, d.rms_eps, tmp, stream);
    pm::quantize(tmp, nullptr, ps.xq, 8, kE, kE, keep, stream);

    for (int l = 0; l < d.n_layers; ++l) {
        const Dflash2Layer& L = w_->layer[(size_t) l];
        const Cache& c = cache_[(size_t) l];
        pm::Product p;
        p.type = 8;
        p.w_rows = kDf2KV;
        p.w_cols = kE;
        p.expert_bytes = pm::matrix_bytes(8, kDf2KV, kE);
        p.n = 1;
        p.xq = ps.xq;
        p.bounds = bounds;
        p.ids = ps.ident;
        p.total_rows = keep;
        p.max_rows = keep;
        p.ld_dst = kDf2KV;
        p.w = static_cast<const uint8_t*>(L.wk_q8);
        p.dst = ps.kb;
        mmq.run(p, stream);
        p.w = static_cast<const uint8_t*>(L.wv_q8);
        p.dst = ps.vb;
        mmq.run(p, stream);
        DsCommitSegs segs;
        segs.k[0] = c.k;
        segs.v[0] = c.v;
        dspark_kv_commit_batch(ps.kb, ps.vb, L.k_norm, d.rms_eps, d.rope_base, pos_d + r0, keep, segs, c.cap, stream);
    }
}

void Dflash2Hook::note(int64_t pos0, int rows) {
    if (!w_ || rows <= 0) return;
    const int64_t cap = cache_[0].cap;
    const int64_t end = pos0 + rows;
    const int64_t start = rows > cap ? end - cap : pos0;
    if (pos0 == 0 || start > hi_ || end < lo_) lo_ = start;
    else lo_ = std::max(std::min(lo_, start), end - cap);
    hi_ = end;
}

void Dflash2Hook::stage(int k, int n, void* stream) {
    bk::PdlScope pdl(true);
    const Dflash2Weights& w = *w_;
    const Dflash2Geometry& d = w.geo;
    const auto& s = w.shared_scr_;
    namespace wg = bk::wgemm;
    const auto mat = [](const void* p, int64_t rows, const float* scale, float* y, int64_t ld_y) {
        return wg::Matrix{(const uint8_t*) p, rows, scale, y, ld_y};
    };

    if (k == 0) {
        dflash2_block_inputs_batch(w.embedding, meta_d_, w.embed_norm, w.target_eps, w.mask, kDf2Block, 1, s.x, stream);
        wg::norm_pack(s.x, w.layer[0].attn_norm, d.rms_eps, kE, kDf2Block, s.act_a, s.xn, stream);
    }
    if (k > 0) {
        const Dflash2Layer& L = w.layer[(size_t) k - 1];
        const float* sc = L.fp4_scale;
        wg::pack(s.attn, kDf2Q, kDf2Q, kDf2Block, s.act_q, stream);
        wg::Call o;
        o.fmt = wg::Format::Nvfp4;
        o.K = kDf2Q;
        o.m[0] = mat(L.wo, kE, sc + 3, s.z, kE);
        wg::gemm(o, s.act_q, kDf2Block, s.ws, stream);
        dflash2_conv2_add_norm_pack(s.z, s.dyn, kDf2Dyn, L.base_a, kDf2Block, 1, s.x, L.post_norm, d.rms_eps, s.act_a,
                                    s.xn, stream);

        wg::Call cf;
        cf.fmt = wg::Format::Q8;
        cf.K = kE;
        cf.m[0] = mat(L.conv_f, kDf2Dyn, nullptr, s.dyn, kDf2Dyn);
        wg::gemm(cf, s.act_a, kDf2Block, s.ws, stream);
        dflash2_conv2_pack(s.xn, s.dyn, kDf2Dyn, L.base_f, kDf2Block, 1, s.act_a, nullptr, stream);

        wg::Call gu;
        gu.fmt = wg::Format::Nvfp4;
        gu.K = kE;
        gu.n = 2;
        gu.epi = wg::Epilogue::SwiGLU;
        gu.m[0] = mat(L.gate, d.n_ff, sc + 4, nullptr, d.n_ff);
        gu.m[1] = mat(L.up, d.n_ff, sc + 5, nullptr, d.n_ff);
        gu.act_out = s.act_h;
        wg::gemm(gu, s.act_a, kDf2Block, s.ws, stream);

        wg::Call dn;
        dn.fmt = wg::Format::Nvfp4;
        dn.K = d.n_ff;
        dn.m[0] = mat(L.down, kE, sc + 6, s.z, kE);
        wg::gemm(dn, s.act_h, kDf2Block, s.ws, stream);
        if (k < d.n_layers) {
            dflash2_conv2_add_norm_pack(s.z, s.dyn, kDf2Dyn, L.base_f, kDf2Block, 1, s.x,
                                        w.layer[(size_t) k].attn_norm, d.rms_eps, s.act_a, s.xn, stream);
        } else {
            dflash2_conv2_add(s.z, s.dyn, kDf2Dyn, L.base_f, kDf2Block, 1, s.x, stream);
        }
    }
    if (k < d.n_layers) {
        const Dflash2Layer& L = w.layer[(size_t) k];
        const Cache& c = cache_[(size_t) k];
        const float* sc = L.fp4_scale;
        wg::Call ca;
        ca.fmt = wg::Format::Q8;
        ca.K = kE;
        ca.m[0] = mat(L.conv_a, kDf2Dyn, nullptr, s.dyn, kDf2Dyn);
        wg::gemm(ca, s.act_a, kDf2Block, s.ws, stream);
        dflash2_conv2_pack(s.xn, s.dyn, kDf2Dyn, L.base_a, kDf2Block, 1, s.act_a, nullptr, stream);

        wg::Call qkv;
        qkv.fmt = wg::Format::Nvfp4;
        qkv.K = kE;
        qkv.n = 3;
        qkv.m[0] = mat(L.wq, kDf2Q, sc + 0, s.q, kDf2Q);
        qkv.m[1] = mat(L.wk, kDf2KV, sc + 1, s.kb, kDf2KV);
        qkv.m[2] = mat(L.wv, kDf2KV, sc + 2, s.vb, kDf2KV);
        wg::gemm(qkv, s.act_a, kDf2Block, s.ws, stream);

        DsBatchKv kv;
        kv.k[0] = c.k;
        kv.v[0] = c.v;
        dflash2_block_qk_prep_batch(s.q, s.kb, s.vb, L.q_norm, L.k_norm, d.rms_eps, d.rope_base, meta_d_, kDf2Block, 1,
                                    s.qr, kv, c.cap, stream);
        return;
    }

    dflash2_norm_draft_pack(s.x, w.norm, d.rms_eps, kDf2Block, 1, n, s.act_a, s.hq, nullptr, stream);
    wg::Call hc;
    hc.fmt = w.head_fp4 ? wg::Format::Nvfp4 : wg::Format::Q8;
    hc.K = kE;
    hc.m[0] = mat(w.head_fp4 ? w.head_fp4 : w.head, w.n_vocab, w.head_fp4_scale, s.logits, w.n_vocab);
    wg::gemm(hc, s.act_a, n, s.ws, stream);

    wg::Call sc_call;
    sc_call.fmt = wg::Format::Bf16;
    sc_call.K = kE;
    sc_call.m[0] = mat(w.sel_h, kDf2Rank, nullptr, s.g, kDf2Rank);
    wg::gemm(sc_call, s.act_a, n, s.ws, stream);

    dflash2_top16(s.logits, w.head, true, s.hq, n, w.n_vocab, d.logit_softcap, s.top_scr, s.cand, s.tail_lse, stream);
    dflash2_select(meta_d_, 2, s.cand, s.tail_lse, s.g, w.pred, w.succ, 1, n, d.logit_softcap, ids_d_, kDf2Block,
                   conf_d_, kDf2Block, stream);
    cudaMemcpyAsync(ids_h_, ids_d_, (size_t) n * 4, cudaMemcpyDeviceToHost, (cudaStream_t) stream);
    cudaMemcpyAsync(conf_h_, conf_d_, (size_t) n * 4, cudaMemcpyDeviceToHost, (cudaStream_t) stream);
}

bool Dflash2Hook::stage_graph(int k, int n, void* stream) {
    cudaGraphExec_t& g = k < w_->geo.n_layers ? g_stage_[k] : g_head_[n];
    const cudaStream_t st = (cudaStream_t) stream;
    if (!g) {
        cudaGraph_t graph = nullptr;
        if (cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed) != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
        stage(k, n, stream);
        const cudaError_t ce = cudaStreamEndCapture(st, &graph);
        const cudaError_t ci = ce == cudaSuccess ? cudaGraphInstantiateWithFlags(&g, graph, 0) : ce;
        if (graph) cudaGraphDestroy(graph);
        if (ci != cudaSuccess) {
            (void) cudaGetLastError();
            g = nullptr;
            basalt::log::warn("basalt inkling2", "dflash2: block graph failed (%s); running without graphs",
                              cudaGetErrorString(ci));
            return false;
        }
    }
    return cudaGraphLaunch(g, st) == cudaSuccess;
}

model::Result Dflash2Hook::draft_begin(int32_t anchor, int64_t pos, int n, void* stream) {
    if (!w_) return {model::Status::NotImplemented, "inkling2: the pack has no drafter"};
    pending_ = 0;
    batch_owner_ = nullptr;
    batch_slot_ = -1;
    n = std::min(n, kDf2Drafts);
    if (n < 1 || pos < 1 || pos - 1 >= hi_ || pos - 1 < lo_ || pos + kDf2Block > backed_cells()) return {};

    const Dflash2Geometry& d = w_->geo;
    const auto& s = w_->shared_scr_;
    meta_h_[0] = anchor;
    meta_h_[1] = (int32_t) pos;
    cudaMemcpyAsync(meta_d_, meta_h_, 2 * sizeof(int32_t), cudaMemcpyHostToDevice, (cudaStream_t) stream);
    if (!stream) graphs_ = false;
    try {
        for (int k = 0; k <= d.n_layers; ++k) {
            if (k > 0) {
                const Cache& c = cache_[(size_t) k - 1];
                int64_t win = d.window;
                bool block_win = false;
                window_range(d, pos, lo_, win, block_win);
                const int64_t cells = pos + kDf2Block - std::max<int64_t>(0, pos - win + 1) + 64;
                bk::gqa_window_attn(s.qr, c.k, c.v, kDf2Heads, kDf2KvHeads, kDf2HeadDim, pos, kDf2Block,
                                    pos + kDf2Block, win, c.cap, s.att_scr, std::min(s.att_cells, cells), s.attn,
                                    stream, block_win);
            }
            if (!graphs_ || !stage_graph(k, n, stream)) {
                graphs_ = false;
                stage(k, n, stream);
            }
        }
    } catch (const std::exception& e) {
        return {model::Status::Error, std::string("inkling2: dflash2 drafter: ") + e.what()};
    }
    pending_ = n;
    return {};
}

model::Result Dflash2Hook::draft_end(int32_t* ids, float* conf, int max_n, int& n, void* stream) {
    n = 0;
    if (!w_) return {model::Status::NotImplemented, "inkling2: the pack has no drafter"};
    if (pending_ == 0) return {};
    if (batch_owner_ != nullptr) {
        Dflash2Batch* owner = batch_owner_;
        const int slot = batch_slot_, pend = pending_;
        batch_owner_ = nullptr;
        batch_slot_ = -1;
        pending_ = 0;
        return owner->take(slot, pend, ids, conf, max_n, n);
    }
    if (cudaStreamSynchronize((cudaStream_t) stream) != cudaSuccess)
        return {model::Status::Error, std::string("inkling2: dflash2 drafter: ") + cudaGetErrorString(cudaGetLastError())};
    const int lim = std::min(pending_, max_n);
    for (int i = 0; i < lim; ++i) {
        ids[i] = ids_h_[i];
        conf[i] = conf_h_[i];
        ++n;
    }
    pending_ = 0;
    return {};
}

size_t Dflash2Hook::running_bytes() const {
    if (!w_) return 0;
    size_t n = kRunHeadBytes;
    for (const Cache& c : cache_) n += 2 * (size_t) c.cap * kDf2KV * 2;
    return n;
}

// All 5 DFlash 2 layers are sliding-window rings (cell p at p % cap) with intra-block 2-tap convolutions, so
// RunningState carries both the GPU KV rings and the valid context span below `pos` (as dense::Session does).
void Dflash2Hook::running_state(int64_t pos, bool save, core::RunningState& st) const {
    if (!w_) return;
    if (save) {
        const int64_t W = std::min<int64_t>(w_->geo.window, cache_[0].cap);
        const int64_t valid = (pos >= lo_ && pos <= hi_)
                                  ? std::clamp<int64_t>(pos - lo_, 0, std::min<int64_t>(W, pos))
                                  : 0;
        const int64_t h[2] = {pos, valid};
        st.head.insert(st.head.end(), reinterpret_cast<const uint8_t*>(h),
                       reinterpret_cast<const uint8_t*>(h) + sizeof(h));
    } else {
        st.head.insert(st.head.end(), kRunHeadBytes, 0);
    }
    for (const Cache& c : cache_) {
        st.regions.push_back({c.k, (size_t) c.cap * kDf2KV * 2});
        st.regions.push_back({c.v, (size_t) c.cap * kDf2KV * 2});
    }
}

bool Dflash2Hook::running_restored(int64_t pos, const uint8_t* head) {
    if (!w_) return true;
    pending_ = 0;
    batch_owner_ = nullptr;
    batch_slot_ = -1;
    if (!head) {
        lo_ = hi_ = pos;
        return false;
    }
    int64_t h[2] = {};
    std::memcpy(h, head, sizeof(h));
    const int64_t W = std::min<int64_t>(w_->geo.window, cache_[0].cap);
    if (h[0] != pos || h[1] < 0 || h[1] > W || h[1] > pos) {
        lo_ = hi_ = pos;
        return false;
    }
    lo_ = pos - h[1];
    hi_ = pos;
    return true;
}

// DFlash 2 has no full-attention layers: its 5 sliding-window rings travel in RunningState (`state_a`).
bool Dflash2Hook::save_cells(int64_t positions, void* stream) const {
    if (!w_) return true;
    if (positions < 0 || positions > backed_cells()) return false;
    return !stream || cudaStreamSynchronize((cudaStream_t) stream) == cudaSuccess;
}

bool Dflash2Hook::restore_cells(size_t kv_size, size_t at, int64_t positions, void* stream) {
    if (!w_) return true;
    if (positions < 0 || positions > backed_cells() || at != kv_size) return false;
    if (stream && cudaStreamSynchronize((cudaStream_t) stream) != cudaSuccess) return false;
    pending_ = 0;
    batch_owner_ = nullptr;
    batch_slot_ = -1;
    lo_ = hi_ = positions;   // empty until running_restored mounts the matching ring window
    return true;
}

// ---- multi-session batched DFlash 2 commit & draft -------------------------------------------------------------------

Dflash2Batch::~Dflash2Batch() {
    for (void* p : allocs_) cudaFree(p);
    for (void* p : pinned_) cudaFreeHost(p);
}

void* Dflash2Batch::dev(size_t bytes, bool& ok) {
    void* p = nullptr;
    if (!ok || cudaMalloc(&p, std::max<size_t>(bytes, 16)) != cudaSuccess) {
        ok = false;
        return nullptr;
    }
    cudaMemset(p, 0, std::max<size_t>(bytes, 16));
    allocs_.push_back(p);
    return p;
}

bool Dflash2Batch::init(const Dflash2Weights* w, int max_sessions, int max_rows, int tap_rows, float* tap, std::string& err) {
    if (!w || max_sessions <= 1) return true;
    namespace wg = bk::wgemm;
    const Dflash2Geometry& d = w->geo;
    const size_t E = kE, ff = (size_t) d.n_ff;
    const int S_all = std::min(max_sessions, (int) bk::kWindowSegsMax);
    const int CR = kChunkSlots * kDf2Block;       // 64
    const int CD = kChunkSlots * kDf2Drafts;      // 56
    bool ok = true;
    auto f = [&](size_t n) { return (float*) dev(n * 4, ok); };

    max_sessions_ = S_all;
    max_rows_ = max_rows;
    tap_rows_ = std::max(tap_rows, kSlice);
    tap_ = tap;

    c_tap_bf16_ = dev(wg::act_bytes((int64_t) kDf2Targets * kE), ok);
    c_raw_ = f((size_t) wg::kRows * E);
    c_act_ = dev(std::max(wg::act_bytes(kE), (size_t) (max_rows + wg::kRows) * E * sizeof(uint16_t)), ok);
    c_kb_ = f((size_t) max_rows * kDf2KV);
    c_vb_ = f((size_t) max_rows * kDf2KV);

    x_ = f((size_t) CR * E);
    xn_ = f((size_t) CR * E);
    z_ = f((size_t) CR * E);
    dyn_ = f((size_t) CR * kDf2Dyn);
    q_ = f((size_t) CR * kDf2Q);
    kb_ = f((size_t) CR * kDf2KV);
    vb_ = f((size_t) CR * kDf2KV);
    qr_ = f((size_t) CR * kDf2Q);
    attn_ = f((size_t) CR * kDf2Q);
    act_a_ = dev(wg::act_bytes(kE), ok);
    act_q_ = dev(wg::act_bytes(kDf2Q), ok);
    act_h_ = dev(wg::act_bytes((int64_t) ff), ok);
    {
        size_t pf = 0, tk = 0;
        ws_need(d, w->n_vocab, w->head_fp4 != nullptr, pf, tk);
        ws_.partial = f(std::max<size_t>(pf, 1));
        ws_.tickets = (int32_t*) dev(std::max<size_t>(tk, 1) * 4, ok);
        ws_.partial_floats = pf;
        ws_.tickets_n = tk;
    }
    att_cells_ = ring_cells(d);
    att_scr_ = f(bk::gqa_decode_scratch_floats(att_cells_, CR, kDf2Heads, kDf2HeadDim));

    logits_ = f((size_t) CD * (size_t) w->n_vocab);
    g_ = f((size_t) CD * kDf2Rank);
    tail_lse_ = f((size_t) CD * 2);
    hq_ = dev(bk::native_q8_1_bytes(kE, CD), ok);
    top_scr_ = dev(dflash2_top16_scratch_bytes(CD, w->n_vocab), ok);
    cand_ = (int32_t*) dev((size_t) CD * 2 * kDf2TopK * 4, ok);

    meta_d_ = (int32_t*) dev((size_t) S_all * 2 * 4, ok);
    ids_d_ = (int32_t*) dev((size_t) S_all * kDf2Block * 4, ok);
    conf_d_ = f((size_t) S_all * kDf2Block);
    if (ok && cudaMallocHost((void**) &meta_h_, (size_t) S_all * 2 * 4) != cudaSuccess) ok = false;
    if (meta_h_) pinned_.push_back(meta_h_);
    if (ok && cudaMallocHost((void**) &ids_h_, (size_t) S_all * kDf2Block * 4) != cudaSuccess) ok = false;
    if (ids_h_) pinned_.push_back(ids_h_);
    if (ok && cudaMallocHost((void**) &conf_h_, (size_t) S_all * kDf2Block * 4) != cudaSuccess) ok = false;
    if (conf_h_) pinned_.push_back(conf_h_);

    if (!ok) {
        cudaGetLastError();
        err = "out of device memory for the batched DFlash 2 drafter";
        return false;
    }
    w_ = w;
    return true;
}

void Dflash2Batch::commit(Dflash2Hook* const* hooks, int n_seg, const bk::WindowSegs* at_d, const int32_t* pos_d,
                          int rows, void* stream) {
    if (!w_ || rows <= 0 || n_seg <= 0) return;
    namespace wg = bk::wgemm;
    bk::PdlScope pdl(true);
    const Dflash2Geometry& d = w_->geo;
    const auto mat = [](const void* p, int64_t r, const float* scale, float* y, int64_t ld_y) {
        return wg::Matrix{(const uint8_t*) p, r, scale, y, ld_y};
    };

    for (int r0 = 0; r0 < rows; r0 += wg::kRows) {
        const int n = std::min(wg::kRows, rows - r0);
        dflash2_pack_tap_bf16(tap_ + (size_t) r0 * kE, (int64_t) tap_rows_ * kE, n, c_tap_bf16_, stream);
        wg::Call fc;
        fc.fmt = wg::Format::Q8;
        fc.K = kDf2Targets * kE;
        fc.m[0] = mat(w_->fc_p, kE, nullptr, c_raw_, kE);
        wg::gemm(fc, c_tap_bf16_, n, ws_, stream);
        wg::norm_pack(c_raw_, w_->hidden_norm, d.rms_eps, kE, n, static_cast<uint16_t*>(c_act_) + (size_t) r0 * kE,
                      nullptr, stream);
    }

    DsCommitSegs segs;
    segs.at_d = at_d;
    const int64_t cap = hooks[0]->cache_[0].cap;
    for (int l = 0; l < d.n_layers; ++l) {
        const Dflash2Layer& L = w_->layer[(size_t) l];
        for (int i = 0; i < n_seg; ++i) {
            segs.k[i] = hooks[i]->cache_[(size_t) l].k;
            segs.v[i] = hooks[i]->cache_[(size_t) l].v;
        }
        for (int r0 = 0; r0 < rows; r0 += wg::kRows) {
            const int n = std::min(wg::kRows, rows - r0);
            wg::Call kv;
            kv.fmt = wg::Format::Nvfp4;
            kv.K = kE;
            kv.n = 2;
            kv.m[0] = mat(L.wk, kDf2KV, L.fp4_scale + 1, c_kb_ + (size_t) r0 * kDf2KV, kDf2KV);
            kv.m[1] = mat(L.wv, kDf2KV, L.fp4_scale + 2, c_vb_ + (size_t) r0 * kDf2KV, kDf2KV);
            wg::gemm(kv, static_cast<const uint16_t*>(c_act_) + (size_t) r0 * kE, n, ws_, stream);
        }
        dspark_kv_commit_batch(c_kb_, c_vb_, L.k_norm, d.rms_eps, d.rope_base, pos_d, rows, segs, cap, stream);
    }
}

model::Result Dflash2Batch::draft(Dflash2Hook* const* hooks, const int32_t* anchors, const int64_t* pos, const int* n,
                                  int count, void* stream) {
    if (!w_) return {model::Status::NotImplemented, "inkling2: the pack has no drafter"};
    if (count <= 0) return {};
    if (count > max_sessions_) {
        for (int i = 0; i < count; ++i)
            if (model::Result r = hooks[i]->draft_begin(anchors[i], pos[i], n[i], stream); !r.ok()) return r;
        return {};
    }

    Dflash2Hook* act[bk::kWindowSegsMax] = {};
    int act_n[bk::kWindowSegsMax] = {};
    int64_t act_pos[bk::kWindowSegsMax] = {};
    int B = 0;
    for (int i = 0; i < count; ++i) {
        Dflash2Hook& hk = *hooks[i];
        hk.pending_ = 0;
        hk.batch_owner_ = nullptr;
        hk.batch_slot_ = -1;
        const int ni = std::min(n[i], kDf2Drafts);
        const int64_t pi = pos[i];
        if (ni < 1 || pi < 1 || pi - 1 >= hk.hi_ || pi - 1 < hk.lo_ || pi + kDf2Block > hk.backed_cells()) continue;
        meta_h_[2 * B] = anchors[i];
        meta_h_[2 * B + 1] = (int32_t) pi;
        act[B] = &hk;
        act_n[B] = ni;
        act_pos[B] = pi;
        ++B;
    }
    if (B == 0) return {};
    if (B == 1) return act[0]->draft_begin(meta_h_[0], act_pos[0], act_n[0], stream);

    const Dflash2Weights& w = *w_;
    const Dflash2Geometry& d = w.geo;
    const cudaStream_t st = (cudaStream_t) stream;
    cudaMemcpyAsync(meta_d_, meta_h_, (size_t) B * 2 * sizeof(int32_t), cudaMemcpyHostToDevice, st);

    namespace wg = bk::wgemm;
    const auto mat = [](const void* p, int64_t rows, const float* scale, float* y, int64_t ld_y) {
        return wg::Matrix{(const uint8_t*) p, rows, scale, y, ld_y};
    };

    try {
        bk::PdlScope pdl(true);
        for (int s0 = 0; s0 < B; s0 += kChunkSlots) {
            const int cn = std::min(kChunkSlots, B - s0);
            const int R = cn * kDf2Block;
            int sub_nmax = 1;
            for (int c = 0; c < cn; ++c) sub_nmax = std::max(sub_nmax, act_n[s0 + c]);
            const int D_rows = cn * sub_nmax;

            dflash2_block_inputs_batch(w.embedding, meta_d_ + 2 * s0, w.embed_norm, w.target_eps, w.mask, kDf2Block, cn,
                                       x_, st);
            wg::norm_pack(x_, w.layer[0].attn_norm, d.rms_eps, kE, R, act_a_, xn_, st);
            for (int l = 0; l < d.n_layers; ++l) {
                const Dflash2Layer& L = w.layer[(size_t) l];
                const float* sc = L.fp4_scale;
                wg::Call ca;
                ca.fmt = wg::Format::Q8;
                ca.K = kE;
                ca.m[0] = mat(L.conv_a, kDf2Dyn, nullptr, dyn_, kDf2Dyn);
                wg::gemm(ca, act_a_, R, ws_, st);
                dflash2_conv2_pack(xn_, dyn_, kDf2Dyn, L.base_a, kDf2Block, cn, act_a_, nullptr, st);

                wg::Call qkv;
                qkv.fmt = wg::Format::Nvfp4;
                qkv.K = kE;
                qkv.n = 3;
                qkv.m[0] = mat(L.wq, kDf2Q, sc + 0, q_, kDf2Q);
                qkv.m[1] = mat(L.wk, kDf2KV, sc + 1, kb_, kDf2KV);
                qkv.m[2] = mat(L.wv, kDf2KV, sc + 2, vb_, kDf2KV);
                wg::gemm(qkv, act_a_, R, ws_, st);

                DsBatchKv kv;
                bk::GqaWindowSeg gsegs[bk::kWindowSegsMax] = {};
                int64_t max_cells = 0;
                const int64_t cap = act[s0]->cache_[(size_t) l].cap;
                for (int c = 0; c < cn; ++c) {
                    const Dflash2Hook::Cache& cc = act[s0 + c]->cache_[(size_t) l];
                    kv.k[c] = cc.k;
                    kv.v[c] = cc.v;
                    const int64_t p0 = act_pos[s0 + c];
                    int64_t win = d.window;
                    bool block_win = false;
                    window_range(d, p0, act[s0 + c]->lo_, win, block_win);
                    const int64_t cells = p0 + kDf2Block - std::max<int64_t>(0, p0 - win + 1) + 64;
                    max_cells = std::max(max_cells, std::min(att_cells_, cells));
                    gsegs[c] = {cc.k, cc.v, p0, kDf2Block, p0 + kDf2Block, win, cc.cap, block_win};
                }
                dflash2_block_qk_prep_batch(q_, kb_, vb_, L.q_norm, L.k_norm, d.rms_eps, d.rope_base, meta_d_ + 2 * s0,
                                            kDf2Block, cn, qr_, kv, cap, st);
                bk::gqa_window_attn_segs(qr_, kDf2Heads, kDf2KvHeads, kDf2HeadDim, gsegs, cn, att_scr_, max_cells, attn_, st);

                wg::pack(attn_, kDf2Q, kDf2Q, R, act_q_, st);
                wg::Call o;
                o.fmt = wg::Format::Nvfp4;
                o.K = kDf2Q;
                o.m[0] = mat(L.wo, kE, sc + 3, z_, kE);
                wg::gemm(o, act_q_, R, ws_, st);
                dflash2_conv2_add_norm_pack(z_, dyn_, kDf2Dyn, L.base_a, kDf2Block, cn, x_, L.post_norm, d.rms_eps,
                                            act_a_, xn_, st);

                wg::Call cf;
                cf.fmt = wg::Format::Q8;
                cf.K = kE;
                cf.m[0] = mat(L.conv_f, kDf2Dyn, nullptr, dyn_, kDf2Dyn);
                wg::gemm(cf, act_a_, R, ws_, st);
                dflash2_conv2_pack(xn_, dyn_, kDf2Dyn, L.base_f, kDf2Block, cn, act_a_, nullptr, st);

                wg::Call gu;
                gu.fmt = wg::Format::Nvfp4;
                gu.K = kE;
                gu.n = 2;
                gu.epi = wg::Epilogue::SwiGLU;
                gu.m[0] = mat(L.gate, d.n_ff, sc + 4, nullptr, d.n_ff);
                gu.m[1] = mat(L.up, d.n_ff, sc + 5, nullptr, d.n_ff);
                gu.act_out = act_h_;
                wg::gemm(gu, act_a_, R, ws_, st);

                wg::Call dn;
                dn.fmt = wg::Format::Nvfp4;
                dn.K = d.n_ff;
                dn.m[0] = mat(L.down, kE, sc + 6, z_, kE);
                wg::gemm(dn, act_h_, R, ws_, st);
                if (l + 1 < d.n_layers) {
                    dflash2_conv2_add_norm_pack(z_, dyn_, kDf2Dyn, L.base_f, kDf2Block, cn, x_,
                                                w.layer[(size_t) l + 1].attn_norm, d.rms_eps, act_a_, xn_, st);
                } else {
                    dflash2_conv2_add(z_, dyn_, kDf2Dyn, L.base_f, kDf2Block, cn, x_, st);
                }
            }

            dflash2_norm_draft_pack(x_, w.norm, d.rms_eps, kDf2Block, cn, sub_nmax, act_a_, hq_, nullptr, st);
            wg::Call hc;
            hc.fmt = w.head_fp4 ? wg::Format::Nvfp4 : wg::Format::Q8;
            hc.K = kE;
            hc.m[0] = mat(w.head_fp4 ? w.head_fp4 : w.head, w.n_vocab, w.head_fp4_scale, logits_, w.n_vocab);
            wg::gemm(hc, act_a_, D_rows, ws_, st);

            wg::Call sc_call;
            sc_call.fmt = wg::Format::Bf16;
            sc_call.K = kE;
            sc_call.m[0] = mat(w.sel_h, kDf2Rank, nullptr, g_, kDf2Rank);
            wg::gemm(sc_call, act_a_, D_rows, ws_, st);

            dflash2_top16(logits_, w.head, true, hq_, D_rows, w.n_vocab, d.logit_softcap, top_scr_, cand_, tail_lse_, st);
            dflash2_select(meta_d_ + 2 * s0, 2, cand_, tail_lse_, g_, w.pred, w.succ, cn, sub_nmax, d.logit_softcap,
                           ids_d_ + (size_t) s0 * kDf2Block, kDf2Block, conf_d_ + (size_t) s0 * kDf2Block, kDf2Block, st);
        }
        cudaMemcpyAsync(ids_h_, ids_d_, (size_t) B * kDf2Block * 4, cudaMemcpyDeviceToHost, st);
        cudaMemcpyAsync(conf_h_, conf_d_, (size_t) B * kDf2Block * 4, cudaMemcpyDeviceToHost, st);
    } catch (const std::exception& e) {
        return {model::Status::Error, std::string("inkling2: dflash2 drafter: ") + e.what()};
    }

    sync_stream_ = stream;
    synced_ = false;
    for (int b = 0; b < B; ++b) {
        act[b]->pending_ = act_n[b];
        act[b]->batch_owner_ = this;
        act[b]->batch_slot_ = b;
    }
    return {};
}

model::Result Dflash2Batch::take(int slot, int pending, int32_t* ids, float* conf, int max_n, int& n) {
    n = 0;
    if (!synced_) {
        if (cudaStreamSynchronize((cudaStream_t) sync_stream_) != cudaSuccess)
            return {model::Status::Error, std::string("inkling2: dflash2 drafter: ") + cudaGetErrorString(cudaGetLastError())};
        synced_ = true;
    }
    const int lim = std::min(pending, max_n);
    const int32_t* si = ids_h_ + (size_t) slot * kDf2Block;
    const float* sc = conf_h_ + (size_t) slot * kDf2Block;
    for (int i = 0; i < lim; ++i) {
        ids[i] = si[i];
        conf[i] = sc[i];
        ++n;
    }
    return {};
}

}  // namespace basalt::models::inkling2
