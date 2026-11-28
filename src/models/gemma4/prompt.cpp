// src/models/gemma4/prompt.cpp - the prompt path of a Gemma 4 session (prompt.hpp).
#include "models/gemma4/prompt.hpp"

#include "basalt/kernels/dense_ops.hpp"
#include "basalt/kernels/expert_plan.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/router.hpp"
#include "basalt/kernels/row_norm.hpp"
#include "basalt/prefill/fp4_dense.hpp"
#include "basalt/prefill/fp8_gemm.hpp"
#include "basalt/prefill/moe_fp4.hpp"
#include "basalt/prefill/moe_mxfp8.hpp"
#include "models/gemma4/experts.hpp"
#include "models/gemma4/model.hpp"

#include <algorithm>

namespace basalt::models::gemma4 {

namespace bk = basalt::kernels;
namespace fp8 = basalt::prefill::fp8;
namespace fp4 = basalt::prefill::fp4;
namespace mx8 = basalt::prefill::mxfp8;
namespace fp4d = basalt::prefill::fp4_dense;

constexpr int kTypeMxfp8 = 46;

// One allocation carved into the chunk\x27s buffers; with no base it only counts, so the sizing and the allocation agree.
struct PromptPath::Carve {
    uint8_t* base = nullptr;
    size_t at = 0;
    template <class T>
    T* take(size_t n) {
        at = (at + 255) & ~(size_t) 255;
        T* p = base ? reinterpret_cast<T*>(base + at) : nullptr;
        at += n * sizeof(T);
        return p;
    }
};

void PromptPath::layout(Carve& c, const Geometry& g, int64_t F, bool mlp_nvfp4, int64_t context, KvType kv, bool mxfp8) {
    const size_t C = kChunk, E = (size_t) g.n_embd;
    size_t qmax = 0, kmax = 0;
    for (int l = 0; l < g.n_layers; ++l) {
        const LayerShape s = g.shape(l);
        qmax = std::max(qmax, (size_t) s.n_head * s.hd);
        kmax = std::max(kmax, (size_t) s.n_kv * s.hd);
    }
    io_ = c.take<int32_t>(2 * C);
    x_ = c.take<float>(C * E), tmp_ = c.take<float>(C * E), mlp_ = c.take<float>(C * E), xn_ = c.take<float>(C * E);
    q_ = c.take<float>(C * qmax), qp_ = c.take<float>(C * qmax), k_ = c.take<float>(C * kmax), v_ = c.take<float>(C * kmax);
    g_ = c.take<float>(C * (size_t) F), u_ = c.take<float>(C * (size_t) F);
    act_ = c.take<uint8_t>(fp8::act_bytes((int64_t) C, (int64_t) std::max({E, qmax, (size_t) F})));
    if (mlp_nvfp4) a4m_ = c.take<uint8_t>(fp4d::act_bytes((int64_t) C, (int64_t) std::max(E, (size_t) F))), gsm_ = c.take<float>(C);
    size_t one = 0;   // a full layer's cache cell whose keys are its values
    for (int l = 0; l < g.n_layers; ++l)
        if (g.shape(l).one) one = std::max(one, (size_t) g.shape(l).n_kv * g.shape(l).hd);
    k16_ = one ? c.take<uint16_t>((size_t) context * one) : nullptr;
    v16_ = one && kv == KvType::F8 ? c.take<uint16_t>((size_t) context * one) : nullptr;
    if (!g.moe()) return;
    const size_t NE = (size_t) g.n_expert, P = C * (size_t) g.n_used, Fe = (size_t) g.n_ff_exp;
    tabs_ = c.take<unsigned long long>((size_t) g.n_layers * NE);
    rlog_ = c.take<float>(C * NE), rw_ = c.take<float>(P), ids_ = c.take<int32_t>(P);
    bounds_ = c.take<int32_t>(NE + 1), tile_off_ = c.take<int32_t>(NE + 1), blk_off_ = c.take<int32_t>(NE + 1);
    tok_ = c.take<int32_t>(P), dst_ = c.take<int32_t>(P), exp_ = c.take<int32_t>(P);
    sort_scr_ = c.take<uint8_t>(bk::moe_sort_scratch((int) P, (int) NE));
    a4x_ = c.take<uint8_t>(fp4::token_bytes((int64_t) C, 2 * (int64_t) E)), gsx_ = c.take<float>(C);
    a4h_ = c.take<uint8_t>(fp4::a4_bytes((int64_t) P, 2 * (int64_t) ((Fe + 127) / 128 * 128), (int) NE));
    gsh_ = c.take<float>(fp4::scale_count((int64_t) P, (int) NE));
    h_ = c.take<float>(P * Fe), ex_ = c.take<float>(P * E);
    if (!mxfp8) return;
    const size_t Fk = (Fe + 127) / 128 * 128;   // an MXFP8 down reads whole 128-value blocks: H's columns past Fe stay zero
    mxx_ = c.take<uint8_t>(mx8::act_bytes((int64_t) C, (int64_t) E));
    mxh_ = c.take<uint8_t>(mx8::act_bytes((int64_t) P, (int64_t) Fk));
    h16_ = c.take<uint16_t>(P * Fk), mx_blk_ = c.take<int32_t>(NE + 1);
}

uint64_t PromptPath::bytes(const Geometry& g, int64_t n_ff_pad, bool mlp_nvfp4, int64_t context, KvType kv, bool mxfp8) {
    PromptPath p;
    Carve c;
    p.layout(c, g, n_ff_pad, mlp_nvfp4, context, kv, mxfp8);
    return c.at;
}

PromptPath::~PromptPath() {
    end();
    for (cudaEvent_t e : {copied_[0], copied_[1], done_[0], done_[1]})
        if (e) cudaEventDestroy(e);
    if (copy_) cudaStreamDestroy(copy_);
    if (staged_) cudaEventDestroy(staged_);
    if (io_h_) cudaFreeHost(io_h_);
    if (base_) cudaFree(base_);
}

bool PromptPath::init(const Gemma4Model& m, std::string& err) {
    m_ = &m;
    if (cudaMallocHost((void**) &io_h_, 2 * (size_t) kChunk * 4) != cudaSuccess ||
        cudaEventCreateWithFlags(&staged_, cudaEventDisableTiming) != cudaSuccess ||
        cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&copied_[0], cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&copied_[1], cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&done_[0], cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&done_[1], cudaEventDisableTiming) != cudaSuccess) {
        cudaGetLastError();
        err = "out of memory for the prompt path\x27s staging";
        return false;
    }
    return unpark(err);
}

void PromptPath::park() {
    if (!base_) return;
    cudaFree(base_);
    base_ = nullptr;
}

bool PromptPath::unpark(std::string& err) {
    if (base_) return true;
    const Gemma4Model& m = *m_;
    Carve c;
    layout(c, m.geo(), m.n_ff_pad(), m.mlp_nvfp4(), m.context(), m.kv_type(), m.mxfp8_experts());
    if (cudaMalloc(&base_, c.at) != cudaSuccess) {
        cudaGetLastError();
        base_ = nullptr;
        err = "out of memory for the prompt path\x27s buffers";
        return false;
    }
    c = Carve{static_cast<uint8_t*>(base_)};
    layout(c, m.geo(), m.n_ff_pad(), m.mlp_nvfp4(), m.context(), m.kv_type(), m.mxfp8_experts());
    if (h16_) cudaMemset(h16_, 0, (size_t) kChunk * m.geo().n_used * ((m.geo().n_ff_exp + 127) / 128 * 128) * 2);   // the pad columns stay zero
    return true;
}

bool PromptPath::begin(cudaStream_t st, std::string& err) {
    const Gemma4Model& m = *m_;
    const Geometry& g = m.geo();
    core::ExpertTier* tier = m.tier();
    if (!tier) return true;
    if (!m.experts()->prompt_begin(st, err)) return false;
    // every copy lands (the promotions the last window queued too); with experts off the card, the copies beside the
    // homes give the staging its room
    if (m.stage_experts() > 0) tier->shrink();
    tier->settle();
    const int NE = g.n_expert;
    ram_.assign((size_t) g.n_layers, {});
    int most = 0;
    for (int l = 0; l < g.n_layers; ++l) {
        int n = 0;
        for (int e = 0; e < NE; ++e) n += tier->locate(l, e).dev == nullptr;
        most = std::max(most, n);
    }
    if (most > m.stage_experts()) {
        if (m.stage_experts() > 0) tier->grow();
        err = "the prompt's staging holds " + std::to_string(m.stage_experts()) + " experts a layer, a layer has " +
              std::to_string(most) + " off the card";
        return false;
    }
    if (most > 0) {
        stage_n_ = most;
        // zeroed bytes after the last blob: the products' last loads read past it
        const size_t bytes = 2 * (size_t) most * m.stage_slot() + 4096;
        if (cudaMalloc((void**) &staging_, bytes) != cudaSuccess || cudaMemset(staging_, 0, bytes) != cudaSuccess) {
            cudaGetLastError();
            staging_ = nullptr;
            if (m.stage_experts() > 0) tier->grow();
            err = "no device memory for the prompt's staging of the experts off the card (" + std::to_string(bytes >> 20) + " MiB)";
            return false;
        }
    }
    began_ = true;
    std::vector<unsigned long long> tab((size_t) g.n_layers * NE);
    for (int l = 0; l < g.n_layers; ++l) {
        int k = 0;
        for (int e = 0; e < NE; ++e) {
            const core::ExpertTier::Source s = tier->locate(l, e);
            if (s.dev) {
                tab[(size_t) l * NE + e] = (unsigned long long) s.dev;
                continue;
            }
            tab[(size_t) l * NE + e] = (unsigned long long) (staging_ + ((size_t) (l & 1) * stage_n_ + k) * m.stage_slot());
            ram_[(size_t) l].push_back({k++, tier->host_blob(l, e), m.layer(l).blob_bytes});
        }
    }
    if (cudaMemcpyAsync(tabs_, tab.data(), tab.size() * sizeof(unsigned long long), cudaMemcpyHostToDevice, st) != cudaSuccess ||
        cudaStreamSynchronize(st) != cudaSuccess) {
        cudaGetLastError();
        err = "cannot write the prompt's expert tables";
        return false;
    }
    return true;
}

void PromptPath::end() {
    if (!began_) return;
    began_ = false;
    if (staging_) {
        cudaStreamSynchronize(copy_);
        cudaFree(staging_);
        staging_ = nullptr;
        stage_n_ = 0;
    }
    if (m_->stage_experts() > 0) m_->tier()->grow();
}

void PromptPath::issue(int l) {
    const int b = l & 1;
    cudaStreamWaitEvent(copy_, done_[b], 0);
    for (const Staged& x : ram_[(size_t) l])
        cudaMemcpyAsync(staging_ + ((size_t) b * stage_n_ + x.k) * m_->stage_slot(), x.host, x.bytes, cudaMemcpyHostToDevice, copy_);
    cudaEventRecord(copied_[b], copy_);
}

void PromptPath::stage(const int32_t* tokens, int rows, int64_t pos0, cudaStream_t st) {
    cudaEventSynchronize(staged_);
    for (int i = 0; i < rows; ++i) {
        if (tokens) io_h_[i] = tokens[i];
        io_h_[kChunk + i] = (int32_t) (pos0 + i);
    }
    if (tokens) cudaMemcpyAsync(io_, io_h_, (size_t) rows * 4, cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(io_ + kChunk, io_h_ + kChunk, (size_t) rows * 4, cudaMemcpyHostToDevice, st);
    cudaEventRecord(staged_, st);
}

void PromptPath::chunk(const int32_t* tokens, int rows, int64_t pos0, const std::vector<SegKv>& caches, cudaStream_t st) {
    const Geometry& g = m_->geo();
    stage(tokens, rows, pos0, st);
    embed_rows(m_->embedding(), io_, rows, g.n_embd, g.embed_scale, x_, st);
    layers(rows, pos0, false, caches, st);
}

void PromptPath::picture(const float* emb, int rows, int64_t pos0, const std::vector<SegKv>& caches, cudaStream_t st) {
    stage(nullptr, rows, pos0, st);
    cudaMemcpyAsync(x_, emb, (size_t) rows * m_->geo().n_embd * sizeof(float), cudaMemcpyHostToDevice, st);
    layers(rows, pos0, true, caches, st);
}

void PromptPath::layers(int rows, int64_t pos0, bool span, const std::vector<SegKv>& caches, cudaStream_t st) {
    const Gemma4Model& m = *m_;
    const Geometry& g = m.geo();
    const int E = g.n_embd;
    const int64_t F = m.n_ff_pad();
    const float eps = g.rms_eps;
    const int32_t* pos = io_ + kChunk;
    if (staging_)
        for (int l = 0; l < std::min(2, g.n_layers); ++l) issue(l);
    for (int l = 0; l < g.n_layers; ++l) {
        const LayerWeights& w = m.layer(l);
        const LayerShape& s = w.sh;
        const int64_t qd = (int64_t) s.n_head * s.hd, kd = (int64_t) s.n_kv * s.hd;
        // attention
        fp8::rms_norm_quantize(x_, w.attn_norm, eps, rows, E, act_, nullptr, st);
        const fp8::Segment qkv[3] = {{w.q.w, w.q.rows, w.q.scale, q_, qd}, {w.k.w, w.k.rows, w.k.scale, k_, kd},
                                     {w.v.w, w.v.rows, w.v.scale, v_, kd}};
        fp8::gemm_segmented(qkv, s.one ? 2 : 3, E, act_, rows, fp8::Epilogue::Store, st);
        Prep p;
        p.q = q_, p.k = k_, p.v = v_, p.ldq = qd, p.ldk = kd, p.ldv = kd, p.pos = pos;
        p.q_norm = w.q_norm, p.k_norm = w.k_norm, p.rope = w.rope, p.eps = eps, p.sh = s, p.kv_type = m.kv_type();
        p.q_out = qp_;
        p.kv.c[0] = caches[(size_t) l];
        attn_prep(p, rows, st);
        AttnPrompt a;
        a.q = qp_, a.pos0 = pos0, a.span = span, a.k_norm = w.k_norm, a.rope = w.rope, a.sh = s, a.kv_type = m.kv_type();
        a.kv = caches[(size_t) l];
        if (s.one) {   // its keys rotated once into fp16, read in place with the values
            const bool f8 = m.kv_type() == KvType::F8;
            derive_keys(a.kv, w.k_norm, w.rope, s, m.kv_type(), pos0 + rows, k16_, f8 ? v16_ : nullptr, st);
            a.kv = {k16_, f8 ? v16_ : a.kv.k, a.kv.cap};
            a.sh.one = false;
            a.kv_type = KvType::F16;
        }
        a.out = q_;   // the projections are spent
        attn_prompt(a, rows, st);
        fp8::quantize(q_, rows, qd, qd, act_, st);
        fp8::gemm(w.o.w, E, qd, w.o.scale, act_, rows, tmp_, E, fp8::Epilogue::Store, st);
        add_norm(x_, tmp_, w.post_attn, eps, rows, E, st);
        // the MLP
        if (m.mlp_nvfp4()) {
            fp4d::rms_norm_quantize(x_, w.ffn_norm, eps, rows, E, a4m_, gsm_, nullptr, st);
            fp4d::product(w.gate.w, w.gate.scale, F, E, a4m_, gsm_, rows, g_, F, false, st);
            fp4d::product(w.up.w, w.up.scale, F, E, a4m_, gsm_, rows, u_, F, false, st);
            gelu_mul(g_, u_, rows, g.n_ff, F, F, g_, st);
            fp4d::quantize(g_, rows, F, F, a4m_, gsm_, st);
            fp4d::product(w.down.w, w.down.scale, E, F, a4m_, gsm_, rows, mlp_, E, false, st);
        } else {
            fp8::rms_norm_quantize(x_, w.ffn_norm, eps, rows, E, act_, nullptr, st);
            const fp8::Segment gu[2] = {{w.gate.w, w.gate.rows, w.gate.scale, g_, F}, {w.up.w, w.up.rows, w.up.scale, u_, F}};
            fp8::gemm_segmented(gu, 2, E, act_, rows, fp8::Epilogue::Store, st);
            gelu_mul(g_, u_, rows, g.n_ff, F, F, g_, st);
            fp8::quantize(g_, rows, F, F, act_, st);
            fp8::gemm(w.down.w, E, F, w.down.scale, act_, rows, mlp_, E, fp8::Epilogue::Store, st);
        }
        FfnEnd fe;
        fe.x = x_, fe.mlp = mlp_, fe.n_post = w.post_ffw, fe.scalar = w.scalar, fe.eps = eps;
        if (g.moe()) {   // the router reads the residual unweighted-normed, the experts its pre_ffw2 norm
            const int NE = g.n_expert, k = g.n_used, Fe = g.n_ff_exp, P = rows * k;
            bk::rms_norm_rows(x_, m.ones(), rows, E, eps, xn_, st);
            bk::f32_rows_dot(xn_, w.router, rlog_, rows, NE, E, st);
            bk::RouterArgs ra;
            ra.n_expert = NE, ra.n_embd = E, ra.k = k, ra.weight_scale = 1.f, ra.gate = bk::RouterGate::Softmax;
            bk::router_select(rlog_, nullptr, rows, ra, ids_, rw_, st);
            bk::rms_norm_rows(x_, w.pre_ffw2, rows, E, eps, xn_, st);
            bk::moe_sort(ids_, rows, k, NE, sort_scr_, bounds_, tok_, dst_, exp_, st);
            const unsigned long long* blobs = tabs_ + (size_t) l * NE;
            if (staging_) cudaStreamWaitEvent(st, copied_[l & 1], 0);
            const bk::NativeExpertLayout L = bk::native_expert_layout(w.gu_type, w.d_type, E, Fe);
            if (w.gu_type == kTypeMxfp8) {   // E4M3 activations per 32 values; H rounded to fp16 between the products
                const int64_t Fk = (Fe + 127) / 128 * 128;   // the down rows' padded width: H's columns past Fe stay zero
                mx8::plan_blocks(bounds_, NE, mx_blk_, st);
                mx8::quantize_tokens(xn_, rows, E, E, mxx_, st);
                mx8::Product gu;
                gu.blobs = blobs, gu.w_off = 0, gu.w_rows = 2 * Fe, gu.w_cols = E, gu.n = NE;
                gu.act = mxx_, gu.act_rows = rows, gu.bounds = bounds_, gu.blk_off = mx_blk_, gu.blocks = mx8::max_blocks(P, NE);
                gu.ids = tok_, gu.dst_f16 = h16_, gu.ld_dst = Fk, gu.swiglu = true, gu.gelu = true;
                mx8::run(gu, st);
                mx8::quantize_f16(h16_, P, Fk, Fk, mxh_, st);
                mx8::Product dn = gu;
                dn.swiglu = false, dn.gelu = false, dn.ids = nullptr, dn.w_off = L.down_off, dn.w_rows = E, dn.w_cols = Fk;
                dn.act = mxh_, dn.act_rows = P, dn.dst_f16 = nullptr, dn.dst = ex_, dn.dst_ids = dst_, dn.ld_dst = E;
                mx8::run(dn, st);
            } else {   // NVFP4
                fp4::plan_blocks(bounds_, NE, tile_off_, blk_off_, st);
                fp4::quantize_tokens(xn_, rows, E, E, a4x_, gsx_, st, true);
                fp4::Product up;
                up.blobs = blobs, up.blk_off = blk_off_, up.blocks = fp4::max_blocks(P, NE), up.n = NE;
                up.w_rows = 2 * Fe, up.w_cols = E, up.a4 = a4x_, up.gs = gsx_, up.bounds = bounds_, up.ids = tok_, up.tokens = rows;
                up.swiglu = true, up.gelu = true, up.tail_off = L.bytes - 16, up.dst = h_, up.ld_dst = Fe, up.two = true;
                fp4::run(up, st);
                fp4::quantize(h_, nullptr, bounds_, tile_off_, NE, a4h_, gsh_, Fe, Fe, P, st, true);
                fp4::Product dn = up;
                dn.swiglu = false, dn.gelu = false, dn.ids = nullptr, dn.w_off = L.down_off, dn.w_rows = E, dn.w_cols = Fe;
                dn.a4 = a4h_, dn.gs = gsh_, dn.tile_off = tile_off_, dn.dst = ex_, dn.dst_ids = dst_, dn.ld_dst = E;
                fp4::run(dn, st);
            }
            if (staging_) {   // the buffer is free for the layer after the next
                cudaEventRecord(done_[l & 1], st);
                if (l + 2 < g.n_layers) issue(l + 2);
            }
            fe.experts = ex_, fe.weights = rw_, fe.k = k, fe.n1 = w.post_ffw1, fe.n2 = w.post_ffw2;
        }
        ffn_end(fe, rows, E, st);
    }
}

}  // namespace basalt::models::gemma4
