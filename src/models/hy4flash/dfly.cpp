// src/models/hy4flash/dfly.cpp - the DFly2 drafter on the device: weights, the tap, and the block pass (dfly.hpp).
#include "basalt/core/device_copy.hpp"
#include "models/hy4flash/dfly.hpp"

#include "basalt/kernels/bf16_gemv.hpp"
#include "basalt/kernels/dense_ops.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/q80_screen.hpp"
#include "basalt/log.hpp"
#include "basalt/models/hy4flash/dfly_kernels.hpp"
#include "basalt/models/hy4flash/kernels.hpp"
#include "models/hy4flash/pack.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace basalt::models::hy4flash {
namespace {
namespace bk = basalt::kernels;
}  // namespace

// ---- weights ------------------------------------------------------------------------------------------------------------

DflyWeights::~DflyWeights() {
    for (auto& [name, p] : slots_)
        if (p) cudaFree(p);
    if (head_screen) cudaFree(head_screen);
}

bool DflyWeights::allocate(const DflyDims& d, std::string& err) {
    dims = d;
    if (!bk::q80_screen_supported(d.E)) {
        err = "no draft head screen for a width of " + std::to_string(d.E);
        return false;
    }
    const size_t screen = bk::q80_screen_bytes(d.E, d.vocab);
    if (cudaMalloc(&head_screen, screen) != cudaSuccess) {
        err = "out of device memory for the drafter's head screen";
        return false;
    }
    bytes_ += screen;
    for (const DflySpec& s : dfly_specs(d)) {
        if ((s.q8 || s.q4) && (s.shape.size() != 2 || s.shape[1] % 32 != 0)) {
            err = "the drafter's " + s.name + " rows are not whole blocks of 32";
            return false;
        }
        const uint64_t b = s.bytes();
        void* p = nullptr;
        if (cudaMalloc(&p, b) != cudaSuccess) {
            err = "out of device memory for the drafter's " + s.name;
            return false;
        }
        slots_[s.name] = p;
        bytes_ += b;
    }
    auto bf = [&](const std::string& n) { return (const uint16_t*) slot(n); };
    auto q = [&](const std::string& n) { return (const void*) slot(n); };
    auto f = [&](const std::string& n) { return (const float*) slot(n); };
    layer.clear();
    for (int i = 0; i < d.layers; ++i) {
        const std::string p = std::to_string(i) + ".";
        DflyLayer l;
        l.q_a = q(p + "self_attn.q_a_proj.weight");
        l.q_b = q(p + "self_attn.q_b_proj.weight");
        l.kv_a = q(p + "self_attn.kv_a_proj_with_mqa.weight");
        l.kv_b = bf(p + "self_attn.kv_b_proj.weight");
        l.gate = q(p + "self_attn.linear_gate.weight");
        l.o = q(p + "self_attn.o_proj.weight");
        l.mlp_gate = q(p + "mlp.gate_proj.weight");
        l.mlp_up = q(p + "mlp.up_proj.weight");
        l.mlp_down = q(p + "mlp.down_proj.weight");
        l.in_ln = f(p + "input_layernorm.weight");
        l.q_a_ln = f(p + "self_attn.q_a_layernorm.weight");
        l.kv_a_ln = f(p + "self_attn.kv_a_layernorm.weight");
        l.post_proj_ln = f(p + "self_attn.post_proj_norm.weight");
        l.post_attn_ln = f(p + "post_attention_layernorm.weight");
        l.sink = f(p + "self_attn.learnable_sink_param");
        layer.push_back(l);
    }
    eh_proj = q("mtp_fusion.eh_proj.weight");
    hc_gate = q("hidden_correction.gate_proj.weight");
    hc_up = q("hidden_correction.up_proj.weight");
    hc_down = q("hidden_correction.down_proj.weight");
    enorm = f("mtp_fusion.enorm.weight");
    hnorm = f("mtp_fusion.hnorm.weight");
    fuse_gate = f("mtp_fusion.gate");
    hc_hidden_norm = f("hidden_correction.hidden_norm.weight");
    hc_embed_norm = f("hidden_correction.embed_norm.weight");
    final_norm = f("final_norm.weight");
    conf_w = f("confidence_head.proj.weight");
    conf_b = f("confidence_head.proj.bias");
    return true;
}

void* DflyWeights::slot(const std::string& name) const {
    auto it = slots_.find(name);
    return it == slots_.end() ? nullptr : it->second;
}

bool DflyWeights::pack_screen(void* stream, std::string& err) {
    if (!head_q8 || !head_screen) {
        err = "the drafter has no head to screen";
        return false;
    }
    bk::q80_screen_pack(head_q8, dims.E, dims.vocab, head_screen, stream);
    if (cudaStreamSynchronize((cudaStream_t) stream) != cudaSuccess) {
        err = std::string("packing the drafter's head screen: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    return true;
}

model::Result dfly_plan(const Pack& pack, core::LoadPlan& reads, int file, int device, std::unique_ptr<DflyWeights>& out) {
    out.reset();
    const core::Json* meta = pack.file.meta().get("dfly2");
    if (!meta || !meta->is_object()) return {};
    const core::ModelFile::Section* sec = pack.file.find("dfly");
    if (!sec) return {model::Status::Error, "hy4flash: meta.dfly2 names no dfly section"};
    const Geometry& g = pack.geo;
    DflyPack dp;
    std::string err;
    if (!parse_dfly_meta(*meta, sec->offset, dp, err) || !dfly_check_shapes(dp.tensors, dp.dims, err))
        return {model::Status::Error, "hy4flash: " + err};
    DflyDims& d = dp.dims;
    d.vocab = g.n_vocab;
    d.rms_eps = g.rms_eps;
    d.rope_base = g.rope_base;
    if (d.E != g.n_embd || d.mask_id >= g.n_vocab || d.topk != g.idx_top_k || d.heads != g.heads[(size_t) d.kv_layers[0]])
        return {model::Status::Error, "hy4flash: the drafter's width, mask token, selection width or head count are not the target's"};
    for (int i = 0; i < d.layers; ++i) {
        const int kv = d.kv_layers[(size_t) i], ix = d.idx_layers[(size_t) i];
        if (kv < 0 || kv >= g.n_layers || ix < 0 || ix >= g.n_layers || g.swa[(size_t) kv] || !g.indexed[(size_t) ix] ||
            g.heads[(size_t) kv] != d.heads)
            return {model::Status::Error, "hy4flash: draft layer " + std::to_string(i) + " names a target layer that is not a sparse one"};
    }
    auto w = std::make_unique<DflyWeights>();
    if (!w->allocate(d, err)) return {model::Status::Error, "hy4flash: " + err};
    for (const DflySpec& s : dfly_specs(d)) {
        const DflyTensor& t = dp.tensors.at(s.name);
        core::LoadItem it;
        it.file = file;
        it.off = t.offset;
        it.bytes = t.bytes;
        it.dev = w->slot(s.name);
        it.device = device;
        it.conv = s.f32 ? core::Conv::Bf16ToF32 : s.q4 ? core::Conv::Bf16ToQ4_0 : s.q8 ? core::Conv::Bf16ToQ8_0 : core::Conv::Copy;
        reads.add(it);
    }
    basalt::log::info("basalt hy4flash", "drafter: %d layers reading target layers %d..%d, %.2f GiB on the GPU (Q8_0 and Q4_0 projections)", d.layers,
                      d.kv_layers.front(), d.kv_layers.back(), (double) w->bytes() / (double) (1ull << 30));
    out = std::move(w);
    return {};
}

// ---- the tap ------------------------------------------------------------------------------------------------------------

std::unique_ptr<DflyTap> DflyTap::create(const DflyDims& d, int sel_cap, std::string& err) {
    std::unique_ptr<DflyTap> t(new DflyTap());
    t->E_ = d.E;
    t->cap_ = sel_cap;
    t->layers_ = d.layers;
    t->idx_layers_ = d.idx_layers;
    if (cudaMalloc(&t->hid_, (size_t) kDflyTapRows * d.E * 4) != cudaSuccess ||
        cudaMalloc(&t->sel_, (size_t) d.layers * kDflyTapRows * sel_cap * 4) != cudaSuccess) {
        err = "out of device memory for the drafter's tap";
        return nullptr;
    }
    return t;
}

DflyTap::~DflyTap() {
    if (hid_) cudaFree(hid_);
    if (sel_) cudaFree(sel_);
}

void DflyTap::begin_window(int64_t pos0) {
    pos0_ = pos0;
    hid_ok_ = 0;
    std::fill(std::begin(sel_ok_), std::end(sel_ok_), 0u);
}

void DflyTap::invalidate() { pos0_ = -1; hid_ok_ = 0; std::fill(std::begin(sel_ok_), std::end(sel_ok_), 0u); }

void DflyTap::hidden(int row, const float* fin, void* stream) {
    if (row < 0 || row >= kDflyTapRows) return;
    cudaMemcpyAsync(hid_ + (size_t) row * E_, fin, (size_t) E_ * 4, cudaMemcpyDeviceToDevice, (cudaStream_t) stream);
    hid_ok_ |= 1u << row;
}

void DflyTap::selection(int layer, int row, const int32_t* cells, int n, void* stream) {
    if (row < 0 || row >= kDflyTapRows || n > cap_) return;
    for (int i = 0; i < layers_; ++i) {
        if (idx_layers_[(size_t) i] != layer) continue;
        n_[row][i] = n;
        if (n > 0)
            cudaMemcpyAsync(sel_ + ((size_t) i * kDflyTapRows + row) * cap_, cells, (size_t) n * 4, cudaMemcpyDeviceToDevice, (cudaStream_t) stream);
        sel_ok_[row] |= 1u << i;
    }
}

int DflyTap::source_row(int64_t pos) const { return dfly_source_row(pos0_, hid_ok_, sel_ok_, layers_, pos); }

// ---- the block pass -----------------------------------------------------------------------------------------------------

struct DflyRunner::Impl {
    Env env;
    const DflyWeights* w = nullptr;
    DflyDims d;
    std::vector<void*> bufs;
    std::vector<void*> pinned;
    float *X = nullptr, *X1 = nullptr, *xn = nullptr, *o = nullptr, *o2 = nullptr, *mlp = nullptr, *z = nullptr;
    float *qa = nullptr, *qan = nullptr, *q = nullptr, *kvraw = nullptr, *qabs = nullptr, *qcur = nullptr, *olat = nullptr,
          *vout = nullptr, *gate = nullptr, *g = nullptr, *u = nullptr, *h = nullptr;
    uint16_t* ctx = nullptr;   ///< the rows a layer attends, fp16: the selected ones, then the block's own
    uint8_t* blk8 = nullptr;   ///< with FP8 caches, the block's own rows as FP8 rows before they are decoded into ctx
    int32_t *meta_d = nullptr, *meta_h = nullptr;   ///< per layer, the rows' first cells (0) and counts
    void* attw = nullptr;                           ///< the attention's workspace (zeroed once)
    float *cat = nullptr, *fused = nullptr, *emb0 = nullptr, *pe = nullptr, *hg = nullptr, *hu = nullptr, *hh = nullptr,
          *hd = nullptr, *hcout = nullptr, *craw = nullptr;
    void* screen = nullptr;   ///< the head screen's scratch
    int32_t* pick = nullptr;
    float *h_emb = nullptr, *h_pe = nullptr, *h_conf = nullptr;
    int32_t* h_tok = nullptr;
    std::vector<float> mask;
    float bias = 0.f;          ///< the confidence head's bias
    bool ready = false;        ///< the mask row and the bias are read (at the first block: a session is made before the weights are placed)
    int n = 0;                 ///< drafts of the block begin() enqueued
    int32_t anchor = 0;
    bool pending = false;

    ~Impl() {
        for (void* p : bufs) cudaFree(p);
        for (void* p : pinned) cudaFreeHost(p);
    }
    float* dev(size_t floats, bool& ok) {
        void* p = nullptr;
        if (!ok || cudaMalloc(&p, std::max<size_t>(floats * 4, 16)) != cudaSuccess) {
            ok = false;
            return nullptr;
        }
        bufs.push_back(p);
        return (float*) p;
    }
    float* host(size_t floats, bool& ok) {
        void* p = nullptr;
        if (!ok || cudaMallocHost(&p, std::max<size_t>(floats * 4, 16)) != cudaSuccess) {
            ok = false;
            return nullptr;
        }
        pinned.push_back(p);
        return (float*) p;
    }

    // y[r] = W x[r] for `rows` (at most a block's) packed rows, W [n_out][n_in] in Q8_0 (or Q4_0 with q4): the rows in q8_1,
    // then MMVQ
    void mm(const void* wt, const float* x, int ldx, float* y, int ldy, int n_in, int n_out, int rows, void* st, bool q4 = false) {
        if (ldx != n_in || ldy != n_out || rows > kDflyBlock) throw std::logic_error("dfly: a projection's rows are not packed");
        bk::native_quantize_q8_1(x, act_q8, n_in, rows, st);
        mmq(wt, act_q8, y, n_in, n_out, rows, st, q4);
    }
    // the MMVQ alone, over rows already in q8_1
    void mmq(const void* wt, const void* xq, float* y, int n_in, int n_out, int rows, void* st, bool q4 = false) {
        if (rows > kDflyBlock) throw std::logic_error("dfly: a projection's rows are not packed");
        bk::native_mmvq(q4 ? 2 : 8, wt, xq, y, n_in, n_out, rows, st);   // GGUF type ids: Q4_0, Q8_0
    }
    void* act_q8 = nullptr;   // a projection's input rows in q8_1
    void* in_q8 = nullptr;    // rows that several projections read, in q8_1 once
};

std::unique_ptr<DflyRunner> DflyRunner::create(const Env& env, std::string& err) {
    std::unique_ptr<DflyRunner> r(new DflyRunner());
    r->p_ = std::make_unique<Impl>();
    Impl& p = *r->p_;
    p.env = env;
    p.w = env.w;
    p.d = env.w->dims;
    const DflyDims& d = p.d;
    const size_t R = kDflyBlock, E = (size_t) d.E, H = (size_t) d.heads;
    bool ok = true;
    p.X = p.dev(R * E, ok); p.X1 = p.dev(R * E, ok); p.xn = p.dev(R * E, ok); p.o = p.dev(R * E, ok); p.o2 = p.dev(R * E, ok);
    p.mlp = p.dev(R * E, ok); p.z = p.dev(R * E, ok);
    p.qa = p.dev(R * d.q_lora, ok); p.qan = p.dev(R * d.q_lora, ok); p.q = p.dev(R * H * 256, ok); p.kvraw = p.dev(R * kDflyRowWidth, ok);
    p.qabs = p.dev(R * H * kDflyRank, ok); p.qcur = p.dev(R * H * kDflyRowWidth, ok); p.olat = p.dev(R * H * kDflyRank, ok);
    p.vout = p.dev(R * H * kDflyV, ok); p.gate = p.dev(R * H * kDflyV, ok);
    p.g = p.dev(R * d.ffn, ok); p.u = p.dev(R * d.ffn, ok); p.h = p.dev(R * d.ffn, ok);
    p.ctx = (uint16_t*) p.dev((size_t) (env.sel_cap + kDflyBlock) * kDflyRowWidth / 2 + 1, ok);
    if (env.fp8) p.blk8 = (uint8_t*) p.dev((size_t) kDflyBlock * kFp8RowBytes / 4, ok);
    p.meta_d = (int32_t*) p.dev(kDflyMaxLayers * 16, ok);
    const size_t attw = mla_attend_work_bytes((int) H, env.sel_cap + kDflyBlock, kDflyBlock);
    p.attw = p.dev((attw + 3) / 4, ok);
    if (ok) cudaMemset(p.attw, 0, attw);
    p.cat = p.dev(2 * E, ok); p.fused = p.dev(E, ok); p.emb0 = p.dev(E, ok); p.pe = p.dev(E, ok);
    p.hg = p.dev(d.hc_ffn, ok); p.hu = p.dev(d.hc_ffn, ok); p.hh = p.dev(d.hc_ffn, ok); p.hd = p.dev(E, ok); p.hcout = p.dev(E, ok);
    p.craw = p.dev(1, ok);
    p.screen = p.dev(bk::q80_screen_scratch_bytes(d.vocab) / 4 + 1, ok);
    const int widest_in = std::max({2 * d.E, d.q_lora, d.heads * (int) kDflyV, d.ffn, d.hc_ffn});
    p.act_q8 = p.dev(bk::native_q8_1_bytes(widest_in, kDflyBlock) / 4 + 1, ok);
    p.in_q8 = p.dev(bk::native_q8_1_bytes(widest_in, kDflyBlock) / 4 + 1, ok);
    p.pick = (int32_t*) p.dev(1, ok);
    p.h_emb = p.host(R * E, ok); p.h_pe = p.host(E, ok); p.h_conf = p.host(1, ok);
    p.h_tok = (int32_t*) p.host(1, ok);
    p.meta_h = (int32_t*) p.host(kDflyMaxLayers * 16, ok);
    if (!ok) {
        err = "out of memory for the drafter's scratch";
        return nullptr;
    }
    p.mask.resize(E);
    return r;
}

DflyRunner::~DflyRunner() = default;

bool DflyRunner::begin(int32_t anchor, int64_t pos, int n, const DflyTap& tap, int row, const void* const* cache, void* stream,
                       std::string& err, int64_t rope_pos) {
    Impl& p = *p_;
    const DflyDims& d = p.d;
    const DflyWeights& w = *p.w;
    if (n < 1 || n > kDflyDrafts || row < 0) {
        err = "dfly: a block of " + std::to_string(n) + " drafts";
        return false;
    }
    cudaStream_t st = (cudaStream_t) stream;
    if (p.pending) cudaStreamSynchronize(st);   // the host rows the last block copied from are free again
    const int R = n + 1, E = d.E, H = d.heads;
    if (!p.ready) {   // a session is made before the load places the embedding and the weights: these are read now
        if (!p.env.embed || !p.env.embed(d.mask_id, p.mask.data())) {
            err = "no embedding row for the drafter's mask token";
            return false;
        }
        basalt::core::device_copy(&p.bias, w.conf_b, 4, cudaMemcpyDeviceToHost);
        p.ready = true;
    }
    try {
        if (!p.env.embed(anchor, p.h_emb)) {
            err = "dfly: the anchor token has no embedding row";
            return false;
        }
        for (int r = 1; r < R; ++r) std::memcpy(p.h_emb + (size_t) r * E, p.mask.data(), (size_t) E * 4);
        cudaMemcpyAsync(p.emb0, p.h_emb, (size_t) E * 4, cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync(p.X + E, p.h_emb + E, (size_t) (R - 1) * E * 4, cudaMemcpyHostToDevice, st);

        // Gated Fusion on the anchor's row: x0 = e + gate * eh_proj(enorm(e) :: hnorm(h))
        bk::rms_norm_rows(p.emb0, w.enorm, p.cat, 1, E, d.rms_eps, st, 2 * E);
        bk::rms_norm_rows(tap.hidden_row(row), w.hnorm, p.cat + E, 1, E, d.rms_eps, st, 2 * E);
        p.mm(w.eh_proj, p.cat, 2 * E, p.fused, E, 2 * E, E, 1, st);
        dfly_gate_add(p.emb0, p.fused, w.fuse_gate, E, p.X, st);

        MlaArgs a;
        a.heads = H;
        a.nope = kDflyNope;
        a.rope = kDflyRopeDim;
        a.rank = kDflyRank;
        a.rope_base = d.rope_base;
        a.scale = 1.0f / std::sqrt((float) (kDflyNope + kDflyRopeDim));
        a.rms_eps = d.rms_eps;
        const int QW = 256;
        // the rows each layer attends: nctx of the target's (the cells its row selected, or every position before the anchor),
        // then the block's rows up to the query's own
        int nctxs[kDflyMaxLayers];
        for (int i = 0; i < d.layers; ++i) {
            const int cnt = tap.count(i, row);
            nctxs[i] = cnt < 0 ? (int) pos : cnt;
            if (nctxs[i] + R > 4096 || nctxs[i] > p.env.sel_cap) {
                err = "dfly: " + std::to_string(nctxs[i]) + " selected rows";
                return false;
            }
            for (int r = 0; r < kDflyBlock; ++r) {
                p.meta_h[i * 16 + r] = 0;
                p.meta_h[i * 16 + 8 + r] = r < R ? nctxs[i] + r + 1 : 1;
            }
        }
        cudaMemcpyAsync(p.meta_d, p.meta_h, (size_t) d.layers * 16 * 4, cudaMemcpyHostToDevice, st);
        for (int i = 0; i < d.layers; ++i) {
            const DflyLayer& L = w.layer[(size_t) i];
            // the target's rows this block reads: the cells its row selected (or every position before the anchor)
            const int cnt = tap.count(i, row), nctx = nctxs[i];
            const int32_t* cells = cnt < 0 ? nullptr : tap.cells(i, row);
            if (p.env.fp8) dfly_gather_rows_fp8((const uint8_t*) cache[i], cells, nctx, p.ctx, st);
            else dfly_gather_rows((const uint16_t*) cache[i], cells, nctx, kDflyRowWidth, p.ctx, st);

            bk::rms_norm_rows(p.X, L.in_ln, p.xn, R, E, d.rms_eps, st);
            bk::native_quantize_q8_1(p.xn, p.in_q8, E, R, st);   // q_a, kv_a and the gate read it
            p.mmq(L.q_a, p.in_q8, p.qa, E, d.q_lora, R, st);
            bk::rms_norm_rows(p.qa, L.q_a_ln, p.qan, R, d.q_lora, d.rms_eps, st);
            p.mm(L.q_b, p.qan, d.q_lora, p.q, H * QW, d.q_lora, H * QW, R, st);
            p.mmq(L.kv_a, p.in_q8, p.kvraw, E, kDflyRowWidth, R, st);
            dfly_absorb_k(L.kv_b, p.q, H * QW, QW, R, H, p.qabs, st);
            // the block's rows land after the selected ones, turned by their rotary positions: position rp + r of this base is
            // row nctx + r of ctx
            const int64_t rp = rope_pos < 0 ? pos : rope_pos;
            if (p.env.fp8) {   // the block's rows as the target stores them: FP8 rows at rp + r of blk8's base, decoded after the selected ones
                mla_prepare(p.q, p.qabs, p.kvraw, L.kv_a_ln, a, (int) rp, R, 0, p.qcur, p.blk8 - rp * kFp8RowBytes, st);
                dfly_gather_rows_fp8(p.blk8, nullptr, R, p.ctx + (int64_t) nctx * kDflyRowWidth, st);
            } else {
                mla_prepare(p.q, p.qabs, p.kvraw, L.kv_a_ln, a, (int) rp, R, 0, p.qcur, p.ctx + ((int64_t) nctx - rp) * kDflyRowWidth, st);
            }
            RowSpans rs;
            rs.firsts = p.meta_d + i * 16;
            rs.counts = p.meta_d + i * 16 + 8;
            mla_attend(p.qcur, p.ctx, rs, R, L.sink, a, 0, p.env.sel_cap + kDflyBlock, p.attw, p.olat, st);
            dfly_expand_v(L.kv_b, p.olat, R, H, p.vout, st);
            p.mmq(L.gate, p.in_q8, p.gate, E, H * kDflyV, R, st, true);
            mul_sigmoid(p.gate, p.vout, R * H * kDflyV, st);
            p.mm(L.o, p.vout, H * kDflyV, p.o, E, H * kDflyV, E, R, st, true);
            bk::rms_norm_rows(p.o, L.post_proj_ln, p.o2, R, E, d.rms_eps, st);
            add_vectors(p.X, p.o2, p.X1, R * E, st);
            bk::rms_norm_rows(p.X1, L.post_attn_ln, p.xn, R, E, d.rms_eps, st);
            bk::native_quantize_q8_1(p.xn, p.in_q8, E, R, st);   // the MLP's gate and up read it
            p.mmq(L.mlp_gate, p.in_q8, p.g, E, d.ffn, R, st, true);
            p.mmq(L.mlp_up, p.in_q8, p.u, E, d.ffn, R, st, true);
            bk::swiglu_scaled(p.g, p.u, nullptr, nullptr, p.h, R, d.ffn, d.ffn, st);
            p.mm(L.mlp_down, p.h, d.ffn, p.mlp, E, d.ffn, E, R, st, true);
            add_vectors(p.X1, p.mlp, p.X, R * E, st);
        }
        bk::rms_norm_rows(p.X, w.final_norm, p.z, R, E, d.rms_eps, st);
    } catch (const std::exception& e) {
        err = std::string("dfly: ") + e.what();
        return false;
    }
    const cudaError_t ce = cudaGetLastError();
    if (ce != cudaSuccess) {
        err = std::string("dfly: CUDA: ") + cudaGetErrorString(ce);
        return false;
    }
    p.n = n;
    p.anchor = anchor;
    p.pending = true;
    return true;
}

void DflyRunner::copy_block(float* host, int rows, void* stream) const {
    cudaMemcpyAsync(host, p_->z, (size_t) rows * p_->d.E * 4, cudaMemcpyDeviceToHost, (cudaStream_t) stream);
    cudaStreamSynchronize((cudaStream_t) stream);
}

bool DflyRunner::finish(int32_t* ids, float* conf, int& made, void* stream, std::string& err,
                        const std::function<void(int, int32_t)>& on_row, const std::function<bool(const float*, int)>& need) {
    Impl& p = *p_;
    made = 0;
    if (!p.pending) {
        err = "dfly: no block was begun";
        return false;
    }
    p.pending = false;
    const DflyDims& d = p.d;
    const DflyWeights& w = *p.w;
    cudaStream_t st = (cudaStream_t) stream;
    const int E = d.E;
    std::string why;
    const auto synced = [&] {
        if (cudaStreamSynchronize(st) == cudaSuccess) return true;
        why = std::string("dfly: CUDA: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    };
    // a draft's chance comes before its head: with `need`, the head (the whole vocabulary's rows) runs only for a draft
    // a window can hold
    const auto step = [&](int j, int32_t prev, int32_t& tok, float& chance, bool& go) {
        try {
            if (!p.env.embed(prev, p.h_pe)) {
                why = "dfly: a draft token has no embedding row";
                return false;
            }
            cudaMemcpyAsync(p.pe, p.h_pe, (size_t) E * 4, cudaMemcpyHostToDevice, st);
            // Hidden-Correction: h' = h + down(silu(gate(c)) * up(c)), c = hidden_norm(h) :: embed_norm(embed(previous))
            const float* zrow = p.z + (size_t) (j + 1) * E;
            bk::rms_norm_rows(zrow, w.hc_hidden_norm, p.cat, 1, E, d.rms_eps, st, 2 * E);
            bk::rms_norm_rows(p.pe, w.hc_embed_norm, p.cat + E, 1, E, d.rms_eps, st, 2 * E);
            bk::native_quantize_q8_1(p.cat, p.in_q8, 2 * E, 1, st);   // the gate and up read it
            p.mmq(w.hc_gate, p.in_q8, p.hg, 2 * E, d.hc_ffn, 1, st);
            p.mmq(w.hc_up, p.in_q8, p.hu, 2 * E, d.hc_ffn, 1, st);
            bk::swiglu_scaled(p.hg, p.hu, nullptr, nullptr, p.hh, 1, d.hc_ffn, d.hc_ffn, st);
            p.mm(w.hc_down, p.hh, d.hc_ffn, p.hd, E, d.hc_ffn, E, 1, st);
            add_vectors(zrow, p.hd, p.hcout, E, st);
            f32_gemm_rows(p.hcout, w.conf_w, 1, E, 1.0f, 1, p.craw, st);
            cudaMemcpyAsync(p.h_conf, p.craw, 4, cudaMemcpyDeviceToHost, st);
            if (need) {
                if (!synced()) return false;
                chance = 1.0f / (1.0f + std::exp(-(*p.h_conf + p.bias)));   // conf[j]: the chances so far
                if (!need(conf, j)) {
                    if (on_row && j > 0) on_row(j, prev);   // the token before it may still be in the window
                    go = false;
                    return true;
                }
            }
            bk::native_quantize_q8_1(p.hcout, p.act_q8, E, 1, st);
            bk::q80_screen_top1(w.head_screen, w.head_q8, p.act_q8, E, d.vocab, p.screen, p.pick, st);
            cudaMemcpyAsync(p.h_tok, p.pick, 4, cudaMemcpyDeviceToHost, st);
            if (on_row && j > 0) on_row(j, prev);
            if (!synced()) return false;
        } catch (const std::exception& e) {
            why = std::string("dfly: ") + e.what();
            return false;
        }
        tok = *p.h_tok;
        chance = 1.0f / (1.0f + std::exp(-(*p.h_conf + p.bias)));
        return true;
    };
    if (!dfly_draft_loop(p.n, p.anchor, step, ids, conf, made)) {
        err = why.empty() ? "dfly: the heads failed" : why;
        return false;
    }
    return true;
}

// ---- what a session holds -----------------------------------------------------------------------------------------------

DflyHook::~DflyHook() = default;

bool DflyHook::init(const DflyWeights* w, int sel_cap, int64_t ctx, std::function<bool(int32_t, float*)> embed,
                    const std::string& dump, bool fp8, std::string& err) {
    if (!w) return true;
    fp8_ = fp8;
    dump_ = dump;
    embed_ = embed;
    w_ = w;
    ctx_ = ctx;
    tap_ = DflyTap::create(w->dims, sel_cap, err);
    if (!tap_) return false;
    DflyRunner::Env env;
    env.w = w;
    env.embed = std::move(embed);
    env.sel_cap = sel_cap;
    env.fp8 = fp8;
    runner_ = DflyRunner::create(env, err);
    return runner_ != nullptr;
}

model::Result DflyHook::draft_begin(int32_t anchor, int64_t pos, int n, const std::vector<uint8_t*>& cache, void* stream,
                                    int64_t rope_pos) {
    if (!runner_) return {model::Status::NotImplemented, "hy4flash: the pack has no drafter, or --spec is 0"};
    pending_ = 0;
    const int row = tap_->source_row(pos);
    if (row < 0 || pos + kDflyBlock > ctx_) return {};   // no hidden state to draft from (a prompt's end): the window goes plain
    const void* caches[kDflyMaxLayers];
    for (int i = 0; i < w_->dims.layers; ++i) caches[i] = cache[(size_t) w_->dims.kv_layers[(size_t) i]];
    n = std::clamp(n, 1, kDflyDrafts);
    std::string err;
    if (!runner_->begin(anchor, pos, n, *tap_, row, caches, stream, err, rope_pos)) return {model::Status::Error, "hy4flash: " + err};
    pending_ = n;
    if (!dump_.empty() && dumped_ < 200) {   // --diag-dump: this block's inputs, for a comparison with a reference
        ++dumped_;
        cudaStreamSynchronize((cudaStream_t) stream);
        const std::string base = dump_ + "/dfly_" + std::to_string(pos) + "_";
        auto put = [&](const std::string& name, const void* dev, size_t bytes) {
            std::vector<uint8_t> h(bytes);
            basalt::core::device_copy(h.data(), dev, bytes, cudaMemcpyDeviceToHost);
            if (std::FILE* f = std::fopen((base + name).c_str(), "wb")) {
                std::fwrite(h.data(), 1, bytes, f);
                std::fclose(f);
            }
        };
        dump_pos_ = pos;
        dump_anchor_ = anchor;
        dump_n_ = n;
        put("hid.f32", tap_->hidden_row(row), (size_t) w_->dims.E * 4);
        for (int i = 0; i < w_->dims.layers; ++i)
            if (tap_->count(i, row) < 0 && !fp8_) put("kv" + std::to_string(i) + ".f16", caches[i], (size_t) pos * kDflyRowWidth * 2);
    }
    return {};
}

model::Result DflyHook::draft_end(int32_t* ids, float* conf, int max_n, int& n, void* stream,
                                  const model::ModelSession::DraftNeed& need) {
    if (!runner_) return {model::Status::NotImplemented, "hy4flash: the pack has no drafter, or --spec is 0"};
    n = 0;
    if (pending_ == 0) return {};
    std::string err;
    int32_t tok[kDflyDrafts];
    float ch[kDflyDrafts];
    const int block = pending_, most = std::min(max_n, block);
    pending_ = 0;
    int made = 0;
    std::function<bool(const float*, int)> wanted;
    if (need) wanted = [&](const float* c, int j) { return j < most && need(c, j, most); };
    if (!runner_->finish(tok, ch, made, stream, err, on_row_, wanted)) return {model::Status::Error, "hy4flash: " + err};
    if (dump_pos_ >= 0 && !dump_.empty()) {   // --diag-dump: what the block computed
        std::vector<float> z((size_t) (block + 1) * w_->dims.E), a(w_->dims.E), m(w_->dims.E);
        runner_->copy_block(z.data(), block + 1, stream);
        embed_(dump_anchor_, a.data());
        embed_(w_->dims.mask_id, m.data());
        const std::string base = dump_ + "/dfly_" + std::to_string(dump_pos_) + "_out_";
        auto wr = [&](const char* name, const void* p, size_t bytes) {
            if (std::FILE* f = std::fopen((base + name).c_str(), "wb")) { std::fwrite(p, 1, bytes, f); std::fclose(f); }
        };
        wr("z.f32", z.data(), z.size() * 4);
        wr("anchor_emb.f32", a.data(), a.size() * 4);
        wr("mask_emb.f32", m.data(), m.size() * 4);
        wr("tok.i32", tok, (size_t) made * 4);
        dump_pos_ = -1;
    }
    n = std::min(max_n, made);
    std::copy(tok, tok + n, ids);
    std::copy(ch, ch + n, conf);
    return {};
}

}  // namespace basalt::models::hy4flash
