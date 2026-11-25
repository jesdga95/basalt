// src/models/gemma4/prompt.cpp - the prompt path of a Gemma 4 session (prompt.hpp).
#include "models/gemma4/prompt.hpp"

#include "basalt/kernels/dense_ops.hpp"
#include "basalt/kernels/expert_plan.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/router.hpp"
#include "basalt/kernels/row_norm.hpp"
#include "basalt/prefill/fp8_gemm.hpp"
#include "basalt/prefill/moe_fp4.hpp"
#include "models/gemma4/model.hpp"

#include <algorithm>

namespace basalt::models::gemma4 {

namespace bk = basalt::kernels;
namespace fp8 = basalt::prefill::fp8;
namespace fp4 = basalt::prefill::fp4;

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

void PromptPath::layout(Carve& c, const Geometry& g, int64_t F, int64_t context, KvType kv) {
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
    size_t one = 0;   // a full layer's cache cell whose keys are its values
    for (int l = 0; l < g.n_layers; ++l)
        if (g.shape(l).one) one = std::max(one, (size_t) g.shape(l).n_kv * g.shape(l).hd);
    k16_ = one ? c.take<uint16_t>((size_t) context * one) : nullptr;
    v16_ = one && kv == KvType::F8 ? c.take<uint16_t>((size_t) context * one) : nullptr;
    if (!g.moe()) return;
    const size_t NE = (size_t) g.n_expert, P = C * (size_t) g.n_used, Fe = (size_t) g.n_ff_exp;
    rlog_ = c.take<float>(C * NE), rw_ = c.take<float>(P), ids_ = c.take<int32_t>(P);
    bounds_ = c.take<int32_t>(NE + 1), tile_off_ = c.take<int32_t>(NE + 1), blk_off_ = c.take<int32_t>(NE + 1);
    tok_ = c.take<int32_t>(P), dst_ = c.take<int32_t>(P), exp_ = c.take<int32_t>(P);
    sort_scr_ = c.take<uint8_t>(bk::moe_sort_scratch((int) P, (int) NE));
    a4x_ = c.take<uint8_t>(fp4::token_bytes((int64_t) C, 2 * (int64_t) E)), gsx_ = c.take<float>(C);
    a4h_ = c.take<uint8_t>(fp4::a4_bytes((int64_t) P, 2 * (int64_t) ((Fe + 127) / 128 * 128), (int) NE));
    gsh_ = c.take<float>(fp4::scale_count((int64_t) P, (int) NE));
    h_ = c.take<float>(P * Fe), ex_ = c.take<float>(P * E);
}

uint64_t PromptPath::bytes(const Geometry& g, int64_t n_ff_pad, int64_t context, KvType kv) {
    PromptPath p;
    Carve c;
    p.layout(c, g, n_ff_pad, context, kv);
    return c.at;
}

PromptPath::~PromptPath() {
    if (staged_) cudaEventDestroy(staged_);
    if (io_h_) cudaFreeHost(io_h_);
    if (base_) cudaFree(base_);
}

bool PromptPath::init(const Gemma4Model& m, std::string& err) {
    m_ = &m;
    Carve c;
    layout(c, m.geo(), m.n_ff_pad(), m.context(), m.kv_type());
    if (cudaMalloc(&base_, c.at) != cudaSuccess || cudaMallocHost((void**) &io_h_, 2 * (size_t) kChunk * 4) != cudaSuccess ||
        cudaEventCreateWithFlags(&staged_, cudaEventDisableTiming) != cudaSuccess) {
        cudaGetLastError();
        err = "out of memory for the prompt path\x27s buffers";
        return false;
    }
    c = Carve{static_cast<uint8_t*>(base_)};
    layout(c, m.geo(), m.n_ff_pad(), m.context(), m.kv_type());
    return true;
}

void PromptPath::chunk(const int32_t* tokens, int rows, int64_t pos0, const std::vector<SegKv>& caches, cudaStream_t st) {
    const Gemma4Model& m = *m_;
    const Geometry& g = m.geo();
    const int E = g.n_embd;
    const int64_t F = m.n_ff_pad();
    const float eps = g.rms_eps;
    cudaEventSynchronize(staged_);
    for (int i = 0; i < rows; ++i) {
        io_h_[i] = tokens[i];
        io_h_[kChunk + i] = (int32_t) (pos0 + i);
    }
    cudaMemcpyAsync(io_, io_h_, (size_t) rows * 4, cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(io_ + kChunk, io_h_ + kChunk, (size_t) rows * 4, cudaMemcpyHostToDevice, st);
    cudaEventRecord(staged_, st);
    const int32_t* pos = io_ + kChunk;
    embed_rows(m.embedding(), io_, rows, E, g.embed_scale, x_, st);
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
        a.q = qp_, a.pos0 = pos0, a.k_norm = w.k_norm, a.rope = w.rope, a.sh = s, a.kv_type = m.kv_type();
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
        fp8::rms_norm_quantize(x_, w.ffn_norm, eps, rows, E, act_, nullptr, st);
        const fp8::Segment gu[2] = {{w.gate.w, w.gate.rows, w.gate.scale, g_, F}, {w.up.w, w.up.rows, w.up.scale, u_, F}};
        fp8::gemm_segmented(gu, 2, E, act_, rows, fp8::Epilogue::Store, st);
        gelu_mul(g_, u_, rows, g.n_ff, F, F, g_, st);
        fp8::quantize(g_, rows, F, F, act_, st);
        fp8::gemm(w.down.w, E, F, w.down.scale, act_, rows, mlp_, E, fp8::Epilogue::Store, st);
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
            fp4::plan_blocks(bounds_, NE, tile_off_, blk_off_, st);
            fp4::quantize_tokens(xn_, rows, E, E, a4x_, gsx_, st, true);
            const bk::NativeExpertLayout L = bk::native_expert_layout(w.gu_type, w.d_type, E, Fe);
            fp4::Product up;
            up.base = w.experts, up.stride = w.expert_stride, up.blk_off = blk_off_, up.blocks = fp4::max_blocks(P, NE), up.n = NE;
            up.w_rows = 2 * Fe, up.w_cols = E, up.a4 = a4x_, up.gs = gsx_, up.bounds = bounds_, up.ids = tok_, up.tokens = rows;
            up.swiglu = true, up.gelu = true, up.tail_off = L.bytes - 16, up.dst = h_, up.ld_dst = Fe, up.two = true;
            fp4::run(up, st);
            fp4::quantize(h_, nullptr, bounds_, tile_off_, NE, a4h_, gsh_, Fe, Fe, P, st, true);
            fp4::Product dn = up;
            dn.swiglu = false, dn.gelu = false, dn.ids = nullptr, dn.w_off = L.down_off, dn.w_rows = E, dn.w_cols = Fe;
            dn.a4 = a4h_, dn.gs = gsh_, dn.tile_off = tile_off_, dn.dst = ex_, dn.dst_ids = dst_, dn.ld_dst = E;
            fp4::run(dn, st);
            fe.experts = ex_, fe.weights = rw_, fe.k = k, fe.n1 = w.post_ffw1, fe.n2 = w.post_ffw2;
        }
        ffn_end(fe, rows, E, st);
    }
}

}  // namespace basalt::models::gemma4
