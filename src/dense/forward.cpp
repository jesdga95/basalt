// src/dense/forward.cpp - the layer loop of a decode window and of a prompt chunk, and the drafter's passes.
#include "internal.hpp"

#include "basalt/kernels/gdn_prompt.hpp"
#include "basalt/kernels/gqa_attn.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/logprobs.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/nvfp4_planes.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/window_gemm.hpp"
#include "basalt/prefill/fp4_dense.hpp"
#include "basalt/prefill/fp8_gemm.hpp"

#include <algorithm>
#include <initializer_list>
#include <cstddef>
#include <vector>

namespace basalt::dense {
namespace {


bool cuda_ok(const char* what, std::string& err) {
    const cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) return true;
    err = std::string(what) + ": " + cudaGetErrorString(e);
    return false;
}

namespace wg = kernels::wgemm;

// A decode window's products: F8 and NVFP4-plane matrices on window_gemm, whose rows' arithmetic never depends on the
// window's rows (up to kBatch) and which applies the tensor scale; others (the drafters' Q8_0) on MMQ with the tile of
// kNarrow or kBatch rows whatever the window's rows (activations quantized for the whole tile, the bounds {0, T} saying
// which rows are written), the scale applied after.  The window gemm's activations go to m.ws.xq, MMQ's to m.ws.x8.
bool on_wg(int type) { return type == kTypeF8 || type == kTypeNvfp4Planes || type == kTypeQ8P; }
// the scale the step after a prompt chunk product applies: an NVFP4 matrix's (its product leaves it out), none after F8
const float* post_scale(const Linear& l) { return l.type == kTypeF8 ? nullptr : l.scale; }
int mmq_tile(int T) { return T <= kNarrow ? kNarrow : kBatch; }
// the vocabulary rows the block drafter's head scores: the first kDraftVocab when the head is in planes, else all
int64_t draft_vocab(const Model& m) {
    const Linear& hd = m.w.head;
    return hd.type == kTypeNvfp4Planes && hd.rows > kDraftVocab ? kDraftVocab : m.g.n_vocab;
}

// rows 0..T-1 of x readied for products of `type`
const void* wact(Model& m, const float* x, int type, int64_t cols, int T, cudaStream_t s) {
    if (on_wg(type)) {
        wg::pack(x, cols, cols, T, m.ws.xq, s);
        return m.ws.xq;
    }
    prefill::mmq::quantize(x, nullptr, m.ws.x8, type, cols, cols, mmq_tile(T), s);
    return m.ws.x8;
}
// the RMS norm of x's rows into xn, readied for products of `type` (one kernel on the window gemm)
const void* wnorm(Model& m, const float* x, const float* w, float* xn, int type, int64_t cols, int T, float eps,
                  cudaStream_t s) {
    if (on_wg(type)) {
        wg::norm_pack(x, w, eps, cols, T, m.ws.xq, xn, s);
        return m.ws.xq;
    }
    kernels::rms_norm_rows(x, w, xn, T, cols, eps, s);
    return wact(m, xn, type, cols, T, s);
}
// the same for BF16 rows (the residual)
const void* wnorm(Model& m, const uint16_t* x, const float* w, float* xn, int type, int64_t cols, int T, float eps,
                  cudaStream_t s) {
    if (on_wg(type)) {
        wg::norm_pack(x, w, eps, cols, T, m.ws.xq, xn, s);
        return m.ws.xq;
    }
    kernels::rms_norm_rows(x, w, xn, T, cols, eps, s);
    return wact(m, xn, type, cols, T, s);
}
// the BF16 residual rows readied for a product: as they are on the window gemm, else widened and quantized
const void* resid_act(Model& m, const Linear& l, int T, cudaStream_t s) {
    if (on_wg(l.type)) return m.ws.x;
    kernels::to_f32(m.ws.x, m.ws.tmp, (int64_t) T * l.cols, s);
    return wact(m, m.ws.tmp, l.type, l.cols, T, s);
}
// the window's first rows: the embedding (BF16 in the pack) into the BF16 residual, exactly
void embed_rows(Model& m, const int32_t* tok, int T, cudaStream_t s) {
    const int H = m.g.n_embd;
    kernels::iq_embed_rows(30, m.w.embd, (size_t) H * 2, tok, T, H, m.ws.tmp, s);
    kernels::to_bf16(m.ws.tmp, m.ws.x, (int64_t) T * H, s);
}
void mmq_run(Model& m, const Linear& l, const void* xq, float* y, int T, cudaStream_t s) {
    prefill::mmq::Product p;
    p.w = l.w;
    p.type = l.type;
    p.w_rows = l.rows;
    p.w_cols = l.cols;
    p.expert_bytes = prefill::mmq::matrix_bytes(l.type, l.rows, l.cols);
    p.n = 1;
    p.xq = xq;
    p.bounds = m.ws.bounds;
    p.ids = m.ws.iota;
    p.total_rows = mmq_tile(T);
    p.max_rows = mmq_tile(T);
    p.opt_rows = mmq_tile(T);
    p.dst = y;
    p.ld_dst = l.rows;
    m.mmq->run(p, s);
}
wg::Call wcall(const Linear& l, wg::Epilogue epi = wg::Epilogue::Store) {
    wg::Call c;
    c.fmt = l.type == kTypeF8 ? wg::Format::F8 : l.type == kTypeQ8P ? wg::Format::Q8 : wg::Format::Nvfp4;
    c.K = l.cols;
    c.epi = epi;
    return c;
}
wg::Matrix wmat(const Linear& l, float* y) { return {(const uint8_t*) l.w, l.rows, l.scale, y, l.rows}; }

// y [T][rows] = scale W x
void prod(Model& m, const Linear& l, const void* xq, float* y, int T, cudaStream_t s) {
    if (on_wg(l.type)) {
        wg::Call c = wcall(l);
        c.m[0] = wmat(l, y);
        wg::gemm(c, xq, T, m.ws.w8, s);
        return;
    }
    mmq_run(m, l, xq, y, T, s);
    if (l.scale) kernels::scale_values(y, l.scale, (int64_t) T * l.rows, s);
}
// Several matrices of one input (x, readied in xq for `type`): one window gemm call when they all run there alike.
struct Out {
    const Linear* l;
    float* y;
};
void prod_group(Model& m, const float* x, const void* xq, int type, std::initializer_list<Out> outs, int T, cudaStream_t s) {
    bool one = on_wg(type) && outs.size() <= (size_t) wg::kMaxMatrices;
    for (const Out& o : outs) one = one && o.l->type == type && o.l->cols == outs.begin()->l->cols;
    if (one) {
        wg::Call c = wcall(*outs.begin()->l);
        c.n = 0;
        for (const Out& o : outs) c.m[c.n++] = wmat(*o.l, o.y);
        wg::gemm(c, xq, T, m.ws.w8, s);
        return;
    }
    for (const Out& o : outs) {
        if (o.l->type != type) xq = wact(m, x, type = o.l->type, o.l->cols, T, s);
        prod(m, *o.l, xq, o.y, T, s);
    }
}
// x [T][rows] += scale W y into the BF16 residual (the add in fp32, rounded once; in the window gemm's epilogue)
void prod_add(Model& m, const Linear& l, const void* yq, uint16_t* x, int T, cudaStream_t s) {
    if (on_wg(l.type)) {
        wg::Call c = wcall(l, wg::Epilogue::AddBf16);
        c.m[0] = wmat(l, nullptr);
        c.m[0].y16 = x;
        wg::gemm(c, yq, T, m.ws.w8, s);
        return;
    }
    mmq_run(m, l, yq, m.ws.tmp, T, s);
    kernels::add_scaled(x, m.ws.tmp, l.scale, (int64_t) T * l.rows, s);
}
// h [T][rows] = silu(s_g gate x) (s_u up x): one window gemm call with the SwiGLU epilogue, which can also leave h as the
// next product's bf16 rows (`h16`; h may then be null)
void prod_swiglu(Model& m, const Linear& gate, const Linear& up, const float* x, const void* xq, float* h, int T,
                 cudaStream_t s, void* h16 = nullptr) {
    if (gate.type == kTypeNvfp4Planes && up.type == gate.type) {
        wg::Call c = wcall(gate, wg::Epilogue::SwiGLU);
        c.n = 2;
        c.m[0] = wmat(gate, h);
        c.m[1] = wmat(up, nullptr);
        c.act_out = h16;
        wg::gemm(c, xq, T, m.ws.w8, s);
        return;
    }
    WindowScratch& b = m.ws;
    if (on_wg(gate.type)) {   // two products (their scales applied), then SwiGLU
        prod_group(m, x, xq, gate.type, {{&gate, b.g}, {&up, b.u}}, T, s);
        kernels::swiglu_scaled(b.g, b.u, nullptr, nullptr, h, T, gate.rows, gate.rows, s);
    } else {
        mmq_run(m, gate, xq, b.g, T, s);
        if (up.type != gate.type) xq = wact(m, x, up.type, up.cols, T, s);
        mmq_run(m, up, xq, b.u, T, s);
        kernels::swiglu_scaled(b.g, b.u, gate.scale, up.scale, h, T, gate.rows, gate.rows, s);
    }
    if (h16) wg::pack(h, gate.rows, gate.rows, T, h16, s);
}

// y [T][rows] = W . x on MMQ (xq quantized for l.type), or for F8 on the FP8 tensor cores with the tensor scale applied
void gemm(Model& m, const Linear& l, const void* xq, float* y, int T, cudaStream_t s) {
    if (l.type == kTypeF8) {
        prefill::fp8::gemm((const uint8_t*) l.w, l.rows, l.cols, l.scale, xq, T, y, l.rows, prefill::fp8::Epilogue::Store, s);
        return;
    }
    if (l.type == kTypeNvfp4Planes) {
        prefill::fp4_dense::product(l.w, l.scale, l.rows, l.cols, m.cs.a4, m.cs.gs, T, y, l.rows, false, s);
        return;
    }
    prefill::mmq::Product p;
    p.w = l.w;
    p.type = l.type;
    p.w_rows = l.rows;
    p.w_cols = l.cols;
    p.expert_bytes = prefill::mmq::matrix_bytes(l.type, l.rows, l.cols);
    p.n = 1;
    p.xq = xq;
    p.bounds = m.cs.bounds;
    p.ids = m.cs.iota;
    p.total_rows = T;
    p.max_rows = T;
    p.dst = y;
    p.ld_dst = l.rows;
    m.mmq->run(p, s);
}

void quantize_mmq(Model& m, const float* x, void* xq, int type, int64_t cols, int T, cudaStream_t s) {
    if (type == kTypeF8) prefill::fp8::quantize(x, T, cols, cols, xq, s);
    else if (type == kTypeNvfp4Planes) prefill::fp4_dense::quantize(x, T, cols, cols, m.cs.a4, m.cs.gs, s);
    else prefill::mmq::quantize(x, nullptr, xq, type, cols, cols, T, s);
}

// rope positions [T][3] of cells pos.. for text rows
void text_positions(std::vector<int32_t>& p3, int T, int64_t pos, int64_t delta) {
    p3.resize((size_t) T * 3);
    for (int t = 0; t < T; ++t) p3[(size_t) t * 3] = p3[(size_t) t * 3 + 1] = p3[(size_t) t * 3 + 2] = (int32_t) (pos + t + delta);
}

// ---- decode window layers: the residual rows are m.ws.x

// the gated attention output's product: gate, ready and add into the residual rows (on the window gemm the attention
// merge left the gated bf16 rows in m.ws.xq)
void win_attn_out(Model& m, const Layer& L, int T, cudaStream_t s) {
    const Geometry& g = m.g;
    WindowScratch& b = m.ws;
    const void* yq;
    if (on_wg(L.o.type)) {
        yq = b.xq;
    } else {
        kernels::attn_gate(b.ao, b.qfull, b.ag, T, g.n_head, s);
        yq = wact(m, b.ag, L.o.type, (int64_t) g.n_head * g.head_dim, T, s);
    }
    prod_add(m, L.o, yq, b.x, T, s);
}

// a GDN layer's recurrence arguments but its rows and state (the window's: the qkv rows, b.xn, b.z in, b.y out)
kernels::DecRecArgs gdn_args(Model& m, const Layer& L) {
    const Geometry& g = m.g;
    WindowScratch& b = m.ws;
    const int H = g.n_embd;
    kernels::DecRecArgs a;
    a.conv_w = L.conv;
    a.x = b.xn;
    a.w_alpha = L.w_ab;
    a.w_beta = L.w_ab + (size_t) g.gdn_v_heads * H;
    a.dt = L.dt;
    a.ssm_a = L.ssm_a;
    a.z = b.z;
    a.gamma = L.gamma;
    a.yq = b.x8;   // its q8_1 rows go unread
    a.y = b.y;
    a.C = g.conv_channels();
    a.n_embd = H;
    a.h_k = g.gdn_k_heads;
    a.h_v = g.gdn_v_heads;
    a.eps = g.eps;
    a.early_trigger = true;   // the out projection stages its weights while the recurrence runs
    a.gate_act = g.gdn_gate;
    return a;
}

void win_attn(Model& m, const Layer& L, uint16_t* kc, uint16_t* vc, int T, int64_t pos, cudaStream_t s,
              kernels::KvInt8 q8 = {}) {
    const Geometry& g = m.g;
    WindowScratch& b = m.ws;
    const int H = g.n_embd;
    const void* xq = wnorm(m, b.x, L.attn_norm, b.xn, L.q.type, H, T, g.eps, s);
    prod_group(m, b.xn, xq, L.q.type, {{&L.q, b.qfull}, {&L.k, b.k}, {&L.v, b.v}}, T, s);
    kernels::attn_qk_prepare(b.qfull, b.k, b.v, L.q_norm, L.k_norm, g.eps, g.rope, b.pos, T, g.n_head, g.n_kv, b.qo, kc, vc,
                             pos, s, 256, true, 0, b.pos0, q8);
    const bool gated = on_wg(L.o.type);   // the merge gates and readies the rows for o
    kernels::gqa_decode_attn(b.qo, kc, vc, g.n_head, g.n_kv, pos, T, b.attn, m.o.max_context, gated ? nullptr : b.ao, s, b.pos0, q8,
                             gated ? b.qfull : nullptr, gated ? (uint16_t*) b.xq : nullptr);
    win_attn_out(m, L, T, s);
}

void win_gdn(Model& m, SessionState& st, const Layer& L, int gi, int T, cudaStream_t s) {
    const Geometry& g = m.g;
    WindowScratch& b = m.ws;
    const int H = g.n_embd, C = g.conv_channels();
    const void* xq = wnorm(m, b.x, L.attn_norm, b.xn, L.qkv.type, H, T, g.eps, s);
    float* qkv = st.rec_qkv + (size_t) gi * kWindow * C;
    prod_group(m, b.xn, xq, L.qkv.type, {{&L.qkv, qkv}, {&L.z, b.z}}, T, s);
    kernels::DecRecArgs a = gdn_args(m, L);
    a.state = st.gdn + (size_t) gi * st.gdn_floats;
    a.qkv = qkv;
    a.h = st.rec_h + (size_t) gi * kWindow * C;
    a.gate = st.rec_gate + (size_t) gi * kWindow * g.gdn_v_heads;
    a.beta = st.rec_beta + (size_t) gi * kWindow * g.gdn_v_heads;
    if (on_wg(L.out.type)) a.y16 = (uint16_t*) b.xq;   // the out product's rows
    if (T == 1 && st.gdn_alt) a.state_out = st.gdn_alt + (size_t) gi * st.gdn_floats;   // the state after the row
    kernels::dec_gdn_rec(a, T, s);
    prod_add(m, L.out, on_wg(L.out.type) ? b.xq : wact(m, b.y, L.out.type, g.value_width(), T, s), b.x, T, s);
}

void win_ffn(Model& m, const Layer& L, int T, cudaStream_t s) {
    const Geometry& g = m.g;
    WindowScratch& b = m.ws;
    const int H = g.n_embd;
    const void* xq = wnorm(m, b.x, L.ffn_norm, b.xn, L.gate.type, H, T, g.eps, s);
    if (on_wg(L.down.type)) {   // h straight into the down product's bf16 rows
        prod_swiglu(m, L.gate, L.up, b.xn, xq, nullptr, T, s, b.hq);
        prod_add(m, L.down, b.hq, b.x, T, s);
        return;
    }
    prod_swiglu(m, L.gate, L.up, b.xn, xq, b.h, T, s);
    prod_add(m, L.down, wact(m, b.h, L.down.type, g.n_ff, T, s), b.x, T, s);
}

// ---- prompt chunk layers: the residual rows are m.cs.x

bool all_f8(std::initializer_list<const Linear*> ls) {
    for (const Linear* l : ls)
        if (l->type != kTypeF8) return false;
    return true;
}
prefill::fp8::Segment f8_seg(const Linear& l, void* dst) { return {(const uint8_t*) l.w, l.rows, l.scale, dst, l.rows}; }

// The prompt chunk's residual stream: BF16 rows in ChunkScratch::x's buffer; its activations between the F8 / NVFP4
// products and the GDN and attention kernels are BF16 too (a model on other matrix types keeps FP32 activations).
uint16_t* residual(ChunkScratch& b) { return reinterpret_cast<uint16_t*>(b.x); }
uint16_t* bf(float* p) { return reinterpret_cast<uint16_t*>(p); }

// x += W . a for a chunk's output projection on other matrix types (FP32 a): through tmp into the BF16 residual
void chunk_residual(Model& m, const Linear& l, const float* a, int64_t cols, int T, cudaStream_t s) {
    ChunkScratch& b = m.cs;
    if (l.type == kTypeNvfp4Planes) {
        prefill::fp4_dense::quantize(a, T, cols, cols, b.a4, b.gs, s);
        prefill::fp4_dense::product_add_bf16(l.w, l.scale, l.rows, cols, b.a4, b.gs, T, residual(b), l.rows, s);
        return;
    }
    quantize_mmq(m, a, b.xq, l.type, cols, T, s);
    gemm(m, l, b.xq, b.tmp, T, s);
    kernels::add_scaled(residual(b), b.tmp, post_scale(l), (int64_t) T * l.rows, s);
}
// x += W . a for an F8 output projection whose input rows its producer already rounded into b.xq
void chunk_residual_f8(Model& m, const Linear& l, int T, cudaStream_t s) {
    prefill::fp8::gemm((const uint8_t*) l.w, l.rows, l.cols, l.scale, m.cs.xq, T, residual(m.cs), l.rows,
                       prefill::fp8::Epilogue::AddBf16, s);
}

// `cells_only`: nothing reads the layer's rows (a chunk's last layer), so only its K and V cells are made: the same k
// and v products and the same cache write, without the queries, the attention and the output projection.
void chunk_attn(Model& m, const Layer& L, uint16_t* kc, uint16_t* vc, int T, int64_t pos, cudaStream_t s,
                kernels::KvInt8 q8 = {}, bool cells_only = false) {
    const Geometry& g = m.g;
    ChunkScratch& b = m.cs;
    const int H = g.n_embd, AQ = g.n_head * g.head_dim;
    const bool f8 = all_f8({&L.q, &L.k, &L.v, &L.o});
    if (f8 && cells_only) {
        prefill::fp8::rms_norm_quantize(residual(b), L.attn_norm, g.eps, T, H, b.xq, nullptr, s);
        const prefill::fp8::Segment seg[2] = {f8_seg(L.k, b.k), f8_seg(L.v, b.v)};
        prefill::fp8::gemm_segmented(seg, 2, H, b.xq, T, prefill::fp8::Epilogue::StoreBf16, s);
        kernels::attn_qk_prepare_bf16(nullptr, bf(b.k), bf(b.v), nullptr, L.k_norm, g.eps, g.rope, b.pos, T, 0, g.n_kv, nullptr,
                                      pos, s, q8, kc, vc);
        return;
    }
    if (f8) {   // the norm in the quantizer, one FP8 product for q, k and v (BF16 out)
        prefill::fp8::rms_norm_quantize(residual(b), L.attn_norm, g.eps, T, H, b.xq, nullptr, s);
        const prefill::fp8::Segment seg[3] = {f8_seg(L.q, b.qfull), f8_seg(L.k, b.k), f8_seg(L.v, b.v)};
        prefill::fp8::gemm_segmented(seg, 3, H, b.xq, T, prefill::fp8::Epilogue::StoreBf16, s);
        kernels::attn_qk_prepare_bf16(bf(b.qfull), bf(b.k), bf(b.v), L.q_norm, L.k_norm, g.eps, g.rope, b.pos, T, g.n_head, g.n_kv,
                                      bf(b.qo), pos, s, q8, kc, vc);
        const size_t split = kernels::gqa_prompt_split_floats(kChunk, g.n_head, kAttnSplits);
        // the gate rounds its rows to E4M3 for o itself
        if (q8.k) {   // BF16 queries and output rows
            kernels::gqa_prompt_attn_bf16(bf(b.qo), g.n_head, g.n_kv, pos, T, bf(b.ao), s, q8, b.attn_split, split);
            kernels::attn_gate_e4m3(bf(b.ao), bf(b.qfull), T, g.n_head, (uint8_t*) b.xq, prefill::fp8::act_scales(b.xq, T, AQ), s);
        } else {      // an fp16 cache's prompt kernel reads FP32 queries
            kernels::to_f32(bf(b.qo), b.ag, (int64_t) T * AQ, s);
            kernels::gqa_prompt_attn(b.ag, kc, vc, g.n_head, g.n_kv, pos, T, b.ao, s, q8, b.attn_split, split);
            kernels::attn_gate_e4m3(b.ao, bf(b.qfull), T, g.n_head, (uint8_t*) b.xq, prefill::fp8::act_scales(b.xq, T, AQ), s);
        }
        chunk_residual_f8(m, L.o, T, s);
        return;
    }
    kernels::rms_norm_rows(residual(b), L.attn_norm, b.xn, T, H, g.eps, s);
    if (cells_only) {
        quantize_mmq(m, b.xn, b.xq, L.k.type, H, T, s);
        gemm(m, L.k, b.xq, b.k, T, s);
        gemm(m, L.v, b.xq, b.v, T, s);
        kernels::attn_qk_prepare(nullptr, b.k, b.v, nullptr, L.k_norm, g.eps, g.rope, b.pos, T, 0, g.n_kv, nullptr, kc, vc, pos, s,
                                 256, true, 0, nullptr, q8);
        return;
    }
    quantize_mmq(m, b.xn, b.xq, L.q.type, H, T, s);
    gemm(m, L.q, b.xq, b.qfull, T, s);
    if (L.k.type != L.q.type) quantize_mmq(m, b.xn, b.xq, L.k.type, H, T, s);
    gemm(m, L.k, b.xq, b.k, T, s);
    gemm(m, L.v, b.xq, b.v, T, s);
    kernels::attn_qk_prepare(b.qfull, b.k, b.v, L.q_norm, L.k_norm, g.eps, g.rope, b.pos, T, g.n_head, g.n_kv, b.qo, kc, vc,
                             pos, s, 256, true, 0, nullptr, q8);
    kernels::gqa_prompt_attn(b.qo, kc, vc, g.n_head, g.n_kv, pos, T, b.ao, s, q8, b.attn_split,
                             kernels::gqa_prompt_split_floats(kChunk, g.n_head, kAttnSplits));
    kernels::attn_gate(b.ao, b.qfull, b.ag, T, g.n_head, s);
    chunk_residual(m, L.o, b.ag, AQ, T, s);
}

void chunk_gdn(Model& m, SessionState& st, const Layer& L, int gi, int T, cudaStream_t s) {
    const Geometry& g = m.g;
    ChunkScratch& b = m.cs;
    const int H = g.n_embd, VD = g.value_width(), HV = g.gdn_v_heads;
    float* state = st.gdn + (size_t) gi * st.gdn_floats;
    float* hist = state + (size_t) g.gdn_state * HV * g.gdn_state;
    if (all_f8({&L.qkv, &L.z, &L.out})) {
        // the norm in the quantizer (the normed rows kept for a/b, BF16), one product for qkv and z (BF16 out), the conv and
        // the recurrence on BF16 rows, the closing norm straight to E4M3 for out
        prefill::fp8::rms_norm_quantize(residual(b), L.attn_norm, g.eps, T, H, b.xq, bf(b.xn), s);
        const prefill::fp8::Segment seg[2] = {f8_seg(L.qkv, b.qkv), f8_seg(L.z, b.z)};
        prefill::fp8::gemm_segmented(seg, 2, H, b.xq, T, prefill::fp8::Epilogue::StoreBf16, s);
        kernels::gdn_gating_proj(bf(b.xn), L.w_ab, L.dt, L.ssm_a, b.gate, b.beta, T, HV, H, s);
        kernels::gdn_conv(hist, bf(b.qkv), L.conv, bf(b.hc), T, s);
        kernels::gdn_recurrence_bf16(state, bf(b.hc), b.gate, b.beta, bf(b.z), L.gamma, g.eps, b.work, b.y, T, s, g.gdn_gate,
                                     (uint8_t*) b.xq, prefill::fp8::act_scales(b.xq, T, VD));
        chunk_residual_f8(m, L.out, T, s);
        return;
    }
    {
        kernels::rms_norm_rows(residual(b), L.attn_norm, b.xn, T, H, g.eps, s);
        quantize_mmq(m, b.xn, b.xq, L.qkv.type, H, T, s);
        gemm(m, L.qkv, b.xq, b.qkv, T, s);
        if (L.z.type != L.qkv.type) quantize_mmq(m, b.xn, b.xq, L.z.type, H, T, s);
        gemm(m, L.z, b.xq, b.z, T, s);
    }
    kernels::gdn_gating_proj(b.xn, L.w_ab, L.dt, L.ssm_a, b.gate, b.beta, T, HV, H, s);
    kernels::gdn_conv(hist, b.qkv, L.conv, b.hc, T, s);
    kernels::gdn_recurrence(state, b.hc, b.gate, b.beta, b.z, L.gamma, g.eps, b.work, b.y, nullptr, T, s, g.gdn_gate, b.yn);
    chunk_residual(m, L.out, b.yn, VD, T, s);
}


void chunk_ffn(Model& m, const Layer& L, int T, cudaStream_t s) {
    const Geometry& g = m.g;
    ChunkScratch& b = m.cs;
    const int H = g.n_embd;
    if (L.gate.type == kTypeNvfp4Planes && L.up.type == kTypeNvfp4Planes && L.down.type == kTypeNvfp4Planes) {
        // NVFP4 planes: the norm in the quantizer, W4A4 on the TMA products, gate and up as one SwiGLU product, the residual
        // added by down's; H is BF16 (in b.h's buffer), read once by its quantizer
        uint16_t* h16 = reinterpret_cast<uint16_t*>(b.h);
        prefill::fp4_dense::rms_norm_quantize(residual(b), L.ffn_norm, g.eps, T, H, b.a4, b.gs, s);
        prefill::fp4_dense::swiglu(L.gate.w, L.up.w, L.gate.scale, L.up.scale, L.gate.rows, H, b.a4, b.gs, T, h16, g.n_ff, s,
                                   b.hmax);
        prefill::fp4_dense::quantize_amax(h16, b.hmax, T, g.n_ff, g.n_ff, b.a4, b.gs, s);
        prefill::fp4_dense::product_add_bf16(L.down.w, L.down.scale, L.down.rows, g.n_ff, b.a4, b.gs, T, residual(b), H, s);
        return;
    }
    kernels::rms_norm_rows(residual(b), L.ffn_norm, b.xn, T, H, g.eps, s);
    quantize_mmq(m, b.xn, b.xq, L.gate.type, H, T, s);
    gemm(m, L.gate, b.xq, b.g, T, s);
    if (L.up.type != L.gate.type) quantize_mmq(m, b.xn, b.xq, L.up.type, H, T, s);
    gemm(m, L.up, b.xq, b.u, T, s);
    kernels::swiglu_scaled(b.g, b.u, L.gate.scale, L.up.scale, b.h, T, g.n_ff, g.n_ff, s);
    quantize_mmq(m, b.h, b.xq, L.down.type, g.n_ff, T, s);
    gemm(m, L.down, b.xq, b.tmp, T, s);
    kernels::add_scaled(residual(b), b.tmp, L.down.scale, (int64_t) T * H, s);
}

// The block drafter's rope: plain positions over the whole head.
kernels::RopeSpec block_rope(const BlockDrafter& bd) {
    kernels::RopeSpec r;
    r.rot = bd.head_dim;
    r.theta = bd.theta;
    return r;
}

// The drafter's input rows: cat[t] = [norm_e(embedding) | norm_h(hidden)] into x via fc (window or chunk scratch).
void mtp_input(Model& m, const float* emb, const float* hid, float* cat, int n, cudaStream_t s) {
    const Geometry& g = m.g;
    const Mtp& d = m.w.mtp;
    const int H = g.n_embd;
    kernels::rms_norm_rows(emb, d.norm_e, cat, n, H, g.eps, s, 2 * H);
    kernels::rms_norm_rows(hid, d.norm_h, cat + H, n, H, g.eps, s, 2 * H);
}

}  // namespace

void commit_window(Model& m, SessionState& st, int n, cudaStream_t s) {
    const Geometry& g = m.g;
    const int C = g.conv_channels();
    if (g.gdn_layers() == 0 || n <= 0) return;
    cudaMemcpyAsync(st.n_keep, &n, sizeof(int32_t), cudaMemcpyHostToDevice, s);
    kernels::GdnCommitLayers a;
    a.state = st.gdn;
    a.h = st.rec_h;
    a.qkv = st.rec_qkv;
    a.gate = st.rec_gate;
    a.beta = st.rec_beta;
    a.state_stride = st.gdn_floats;
    a.h_stride = (size_t) kWindow * C;
    a.gb_stride = (size_t) kWindow * g.gdn_v_heads;
    a.C = C;
    a.h_k = g.gdn_k_heads;
    a.h_v = g.gdn_v_heads;
    a.n_layers = g.gdn_layers();
    a.n_keep = st.n_keep;
    kernels::gdn_commit_layers(a, s, 64, n);
}

namespace {

// The window's work on the device, all of it from device inputs (tokens, positions, rows, first cell), so it can be
// captured once per row count and replayed at any position.
void window_body(Model& m, SessionState& st, int T, int64_t pos) {
    const Geometry& g = m.g;
    const Weights& w = m.w;
    WindowScratch& b = m.ws;
    const cudaStream_t s = st.stream;
    const int H = g.n_embd;
    kernels::PdlScope pdl(true);   // the window gemm and its packs start early: their weights are read meanwhile
    embed_rows(m, b.tok, T, s);
    const BlockDrafter& bd = m.bd_dec;
    const bool block = m.o.drafter == Drafter::Block;
    size_t next_target = 0;
    for (int il = 0; il < g.n_layers; ++il) {
        const Layer& L = w.layers[(size_t) il];
        if (L.kind == Mixer::Gdn) {
            win_gdn(m, st, L, m.gdn_index[(size_t) il], T, s);
        } else {
            const int ai = m.attn_index[(size_t) il];
            win_attn(m, L, st.kc[(size_t) ai], st.vc[(size_t) ai], T, pos, s, st.kq[(size_t) ai]);
        }
        win_ffn(m, L, T, s);
        if (block && next_target < bd.target_layers.size() && bd.target_layers[next_target] == il) {
            // the block drafter's features: this layer's rows through its block of the projection, summed
            const Linear& fc = bd.fc[next_target];
            prod(m, fc, resid_act(m, fc, T, s), next_target == 0 ? st.dacc : b.tmp, T, s);
            if (next_target > 0) kernels::add_scaled(st.dacc, b.tmp, nullptr, (int64_t) T * H, s);
            ++next_target;
        }
    }
    prod(m, w.head, wnorm(m, b.x, w.out_norm, b.xn, w.head.type, H, T, g.eps, s), st.logits, T, s);
    if (st.hid) cudaMemcpyAsync(st.hid, b.xn, (size_t) T * H * sizeof(float), cudaMemcpyDeviceToDevice, s);   // the drafter's
}

}  // namespace

bool forward_window(Model& m, SessionState& st, const int32_t* tokens, int T, int64_t pos, const int32_t* pos3,
                    std::string& err) {
    WindowScratch& b = m.ws;
    const cudaStream_t s = st.stream;
    const int32_t bounds[2] = {0, T}, first[2] = {(int32_t) pos, 0};
    cudaMemcpyAsync(b.bounds, bounds, sizeof bounds, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(b.pos0, first, sizeof first, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(b.tok, tokens, (size_t) T * sizeof(int32_t), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(b.pos, pos3, (size_t) T * 3 * sizeof(int32_t), cudaMemcpyHostToDevice, s);
    // a one-row window writes the other state buffer (st.alt_ready); the graph is the one of the buffer in use
    st.alt_ready = T == 1 && st.gdn_alt != nullptr;
    cudaGraphExec_t& wg = st.wg[st.gdn != st.gdn_first ? 1 : 0][T];
    if (wg) {
        if (cudaGraphLaunch(wg, s) != cudaSuccess) return cuda_ok("dense window graph", err) && false;
        return cuda_ok("dense window", err);
    }
    window_body(m, st, T, pos);   // the first window of T rows runs as launched, then is captured for the next ones
    if (!cuda_ok("dense window", err)) return false;
    cudaGraph_t graph = nullptr;
    if (cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed) != cudaSuccess) return cuda_ok("dense window capture", err);
    window_body(m, st, T, pos);
    if (cudaStreamEndCapture(s, &graph) != cudaSuccess || cudaGraphInstantiate(&wg, graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        wg = nullptr;
        return cuda_ok("dense window capture", err);
    }
    cudaGraphDestroy(graph);
    return true;
}


namespace {

// ---- a batch window: the rows of several sessions packed (m.ws.x), the sequence-bound steps per segment

int32_t* batch_pos0(Model& m) { return reinterpret_cast<int32_t*>(reinterpret_cast<char*>(m.ws.btab) + offsetof(BatchTable, pos0)); }

void batch_attn(Model& m, const Layer& L, int ai, const BatchSeg* segs, int S, int rows, cudaStream_t s) {
    const Geometry& g = m.g;
    WindowScratch& b = m.ws;
    const int H = g.n_embd, AQ = g.n_head * g.head_dim, KV = g.n_kv * g.head_dim;
    const void* xq = wnorm(m, b.x, L.attn_norm, b.xn, L.q.type, H, rows, g.eps, s);
    prod_group(m, b.xn, xq, L.q.type, {{&L.q, b.qfull}, {&L.k, b.k}, {&L.v, b.v}}, rows, s);
    // each segment's queries and cache cells, then every segment's attention in one launch
    int row = 0;
    kernels::GqaSeg as[kBatch];
    for (int k = 0; k < S; ++k) {
        SessionState& st = *segs[k].st;
        const int T = segs[k].T;
        const int32_t* p0 = batch_pos0(m) + k;
        kernels::attn_qk_prepare(b.qfull + (size_t) row * 2 * AQ, b.k + (size_t) row * KV, b.v + (size_t) row * KV, L.q_norm,
                                 L.k_norm, g.eps, g.rope, b.pos + (size_t) row * 3, T, g.n_head, g.n_kv,
                                 b.qo + (size_t) row * AQ, st.kc[(size_t) ai], st.vc[(size_t) ai], segs[k].pos, s, 256, true, 0, p0,
                                 st.kq[(size_t) ai]);
        as[k].T = T;
        as[k].k_cache = st.kc[(size_t) ai];
        as[k].v_cache = st.vc[(size_t) ai];
        as[k].pos0 = segs[k].pos;
        as[k].pos_dev = p0;
        as[k].q8 = st.kq[(size_t) ai];
        row += T;
    }
    const bool gated = on_wg(L.o.type);
    kernels::gqa_decode_attn_segs(b.qo, g.n_head, g.n_kv, as, S, b.attn, m.o.max_context, gated ? nullptr : b.ao, s,
                                  gated ? b.qfull : nullptr, gated ? (uint16_t*) b.xq : nullptr);
    win_attn_out(m, L, rows, s);
}

void batch_gdn(Model& m, const Layer& L, int gi, const BatchSeg* segs, int S, int rows, int max_n, cudaStream_t s) {
    const Geometry& g = m.g;
    WindowScratch& b = m.ws;
    const int H = g.n_embd, C = g.conv_channels();
    const void* xq = wnorm(m, b.x, L.attn_norm, b.xn, L.qkv.type, H, rows, g.eps, s);
    kernels::DecRecArgs a = gdn_args(m, L);
    a.qkv = b.rq;
    // the gates once per row and head (they read b.xn only), beside the qkv/z product
    cudaEventRecord(m.gate_fork, s);
    cudaStreamWaitEvent(m.gate_side, m.gate_fork, 0);
    {
        kernels::PdlScope p(true, true);
        kernels::dec_gdn_gates(a, rows, b.gab, m.gate_side);
    }
    cudaEventRecord(m.gate_join, m.gate_side);
    prod_group(m, b.xn, xq, L.qkv.type, {{&L.qkv, b.rq}, {&L.z, b.z}}, rows, s);
    cudaStreamWaitEvent(s, m.gate_join, 0);
    // the raw output in b.y, normed there in place when the out product reads fp32 rows
    a.o = b.y;
    if (on_wg(L.out.type)) { a.y16 = (uint16_t*) b.xq; a.y = nullptr; }
    // a block row a segment, its GDN rows straight into its session's store (the commit's)
    // in a graph the recurrence still starts early behind the product (its edge from the gates is programmatic too,
    // and it reads them after its wait); launched directly it follows the join the plain way
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(s, &cap);
    {
        kernels::PdlScope p(true, cap != cudaStreamCaptureStatusActive);
        kernels::dec_gdn_rec_seg_split(a, b.btab->first, (size_t) gi * segs[0].st->gdn_floats, S, max_n, rows, s,
                                       b.btab->sdst, (size_t) gi * kWindow * C, (size_t) gi * kWindow * g.gdn_v_heads, b.gab);
    }
    prod_add(m, L.out, on_wg(L.out.type) ? b.xq : wact(m, b.y, L.out.type, g.value_width(), rows, s), b.x, rows, s);
}

}  // namespace

namespace {

// The batch window's work on the device, all of it from device inputs (tokens, positions, the segment table), so it can be
// captured once per composition (the sessions and their row counts) and replayed at any positions.
void batch_body(Model& m, const BatchSeg* segs, int S, int rows, int max_n) {
    const Geometry& g = m.g;
    const Weights& w = m.w;
    WindowScratch& b = m.ws;
    const cudaStream_t s = segs[0].st->stream;
    const int H = g.n_embd;
    {
        kernels::PdlScope pdl(true);
        embed_rows(m, b.tok, rows, s);
        const BlockDrafter& bd = m.bd_dec;
        const bool block = m.o.drafter == Drafter::Block;
        size_t next_target = 0;
        for (int il = 0; il < g.n_layers; ++il) {
            const Layer& L = w.layers[(size_t) il];
            if (L.kind == Mixer::Gdn) batch_gdn(m, L, m.gdn_index[(size_t) il], segs, S, rows, max_n, s);
            else batch_attn(m, L, m.attn_index[(size_t) il], segs, S, rows, s);
            win_ffn(m, L, rows, s);
            if (block && next_target < bd.target_layers.size() && bd.target_layers[next_target] == il) {
                // the block drafter's features, packed in b.dacc until every session has its own
                const Linear& fc = bd.fc[next_target];
                prod(m, fc, resid_act(m, fc, rows, s), next_target == 0 ? b.dacc : b.tmp, rows, s);
                if (next_target > 0) kernels::add_scaled(b.dacc, b.tmp, nullptr, (int64_t) rows * H, s);
                ++next_target;
            }
        }
        prod(m, w.head, wnorm(m, b.x, w.out_norm, b.xn, w.head.type, H, rows, g.eps, s), b.hl, rows, s);
    }
    int row = 0;
    for (int k = 0; k < S; row += segs[k++].T) {
        SessionState& st = *segs[k].st;
        const size_t T = (size_t) segs[k].T;
        if (m.o.drafter == Drafter::Block)
            cudaMemcpyAsync(st.dacc, b.dacc + (size_t) row * H, T * H * sizeof(float), cudaMemcpyDeviceToDevice, s);
        if (st.hid)   // the drafter's
            cudaMemcpyAsync(st.hid, b.xn + (size_t) row * H, T * H * sizeof(float), cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(st.logits, b.hl + (size_t) row * g.n_vocab, T * g.n_vocab * sizeof(float), cudaMemcpyDeviceToDevice, s);
    }
}

}  // namespace

bool forward_batch(Model& m, const BatchSeg* segs, int S, const int32_t* tokens, const int32_t* pos3, std::string& err) {
    WindowScratch& b = m.ws;
    const cudaStream_t s = segs[0].st->stream;
    int rows = 0, max_n = 1;
    BatchTable tab{};
    Model::BatchKey key;
    for (int k = 0; k < S; ++k) {
        SessionState& st = *segs[k].st;
        const int T = segs[k].T;
        key.emplace_back(&st, T);
        st.alt_ready = false;   // the batch recurrence leaves its state to the commit
        tab.pos0[k] = (int32_t) segs[k].pos;
        tab.first[k] = {st.gdn, rows, 0, T, T, k, (int32_t) segs[k].pos};
        tab.sdst[k] = {st.rec_qkv, st.rec_h, st.rec_gate, st.rec_beta};
        rows += T;
        max_n = std::max(max_n, T);
    }
    const int32_t bounds[2] = {0, rows};
    cudaMemcpyAsync(b.bounds, bounds, sizeof bounds, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(b.btab, &tab, sizeof tab, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(b.tok, tokens, (size_t) rows * sizeof(int32_t), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(b.pos, pos3, (size_t) rows * 3 * sizeof(int32_t), cudaMemcpyHostToDevice, s);
    for (Model::BatchGraph& bg : m.batch_graphs)
        if (bg.key == key) {
            bg.used = ++m.batch_clock;
            if (cudaGraphLaunch(bg.exec, s) != cudaSuccess) return cuda_ok("dense batch graph", err) && false;
            return cuda_ok("dense batch window", err);
        }
    batch_body(m, segs, S, rows, max_n);   // a composition's first window runs as launched, then is captured for the next ones
    if (!cuda_ok("dense batch window", err)) return false;
    cudaGraph_t graph = nullptr;
    if (cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed) != cudaSuccess) return cuda_ok("dense batch capture", err);
    batch_body(m, segs, S, rows, max_n);
    cudaGraphExec_t exec = nullptr;
    if (cudaStreamEndCapture(s, &graph) != cudaSuccess || cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        return cuda_ok("dense batch capture", err);
    }
    cudaGraphDestroy(graph);
    if (m.batch_graphs.size() >= Model::kBatchGraphs) {   // the least recently used composition goes
        auto old = std::min_element(m.batch_graphs.begin(), m.batch_graphs.end(),
                                    [](const Model::BatchGraph& x, const Model::BatchGraph& y) { return x.used < y.used; });
        cudaGraphExecDestroy(old->exec);
        m.batch_graphs.erase(old);
    }
    m.batch_graphs.push_back({std::move(key), exec, ++m.batch_clock});
    return true;
}

bool forward_chunk(Model& m, SessionState& st, const int32_t* tokens, const float* rows, int T, int64_t pos,
                   const int32_t* pos3, std::string& err) {
    if (!m.ensure_chunk(err)) return false;
    const Geometry& g = m.g;
    const Weights& w = m.w;
    ChunkScratch& b = m.cs;
    const cudaStream_t s = st.stream;
    const int H = g.n_embd;
    const int32_t bounds[2] = {0, T};
    cudaMemcpyAsync(b.bounds, bounds, sizeof bounds, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(b.pos, pos3, (size_t) T * 3 * sizeof(int32_t), cudaMemcpyHostToDevice, s);
    // the chunk's input rows in FP32 (tmp), then the BF16 residual (the BF16 embedding rows exactly)
    if (rows != nullptr) {
        cudaMemcpyAsync(b.tmp, rows, (size_t) T * H * sizeof(float), cudaMemcpyHostToDevice, s);
    } else {
        cudaMemcpyAsync(b.tok, tokens, (size_t) T * sizeof(int32_t), cudaMemcpyHostToDevice, s);
        kernels::iq_embed_rows(30, w.embd, (size_t) H * 2, b.tok, T, H, b.tmp, s);
    }
    kernels::to_bf16(b.tmp, residual(b), (int64_t) T * H, s);
    // the drafter's history needs the chunk's inputs: its next-token embeddings are the chunk's own rows 1..T-1
    const bool drafter = st.mtp_kc != nullptr;
    if (drafter) cudaMemcpyAsync(b.cat, b.tmp, (size_t) T * H * sizeof(float), cudaMemcpyDeviceToDevice, s);
    const BlockDrafter& bd = w.dflash;
    const bool block = m.o.drafter == Drafter::Block;
    // A chunk gives no logits (a prompt's last token runs as a decode window), so the last layer's rows are read only by
    // a drafter: the MTP history takes every row's final hidden state, a block drafter the rows of its target layers.
    // When neither reads them, the last layer leaves only what the session keeps: an attention layer its cells; its FFN
    // is not run.
    const bool last_read = drafter || (block && std::find(bd.target_layers.begin(), bd.target_layers.end(), g.n_layers - 1) !=
                                                    bd.target_layers.end());
    size_t next_target = 0;
    for (int il = 0; il < g.n_layers; ++il) {
        const Layer& L = w.layers[(size_t) il];
        const bool unread = il == g.n_layers - 1 && !last_read;
        if (L.kind == Mixer::Gdn) {
            chunk_gdn(m, st, L, m.gdn_index[(size_t) il], T, s);
        } else {
            const int ai = m.attn_index[(size_t) il];
            chunk_attn(m, L, st.kc[(size_t) ai], st.vc[(size_t) ai], T, pos, s, st.kq[(size_t) ai], unread);
        }
        if (unread) break;
        chunk_ffn(m, L, T, s);
        if (block && next_target < bd.target_layers.size() && bd.target_layers[next_target] == il) {
            const Linear& fc = bd.fc[next_target];
            kernels::to_f32(residual(b), b.xn, (int64_t) T * H, s);   // the target layer's rows for the drafter's fc
            quantize_mmq(m, b.xn, b.xq, fc.type, H, T, s);
            gemm(m, fc, b.xq, next_target == 0 ? b.dacc : b.tmp, T, s);
            if (next_target > 0) kernels::add_scaled(b.dacc, b.tmp, nullptr, (int64_t) T * H, s);
            ++next_target;
        }
    }
    if (block) {   // the chunk's rows become the block drafter's context
        kernels::rms_norm_rows(b.dacc, bd.hidden_norm, b.xn, T, H, bd.eps, s);
        quantize_mmq(m, b.xn, b.xq, bd.layers[0].k.type, H, T, s);
        std::vector<int32_t> p3;
        text_positions(p3, T, pos, 0);
        cudaMemcpyAsync(b.pos, p3.data(), p3.size() * sizeof(int32_t), cudaMemcpyHostToDevice, s);
        kernels::RopeSpec rope = block_rope(bd);
        for (size_t l = 0; l < bd.layers.size(); ++l) {
            const BlockDrafter::Layer& D = bd.layers[l];
            gemm(m, D.k, b.xq, b.k, T, s);
            gemm(m, D.v, b.xq, b.v, T, s);
            kernels::attn_qk_prepare(nullptr, b.k, b.v, nullptr, D.k_norm, bd.eps, rope, b.pos, T, 0, bd.n_kv, nullptr,
                                     st.dk[l], st.dv[l], pos, s, bd.head_dim, false, st.dctx_cap);
        }
    }
    if (!cuda_ok("dense chunk", err)) return false;
    if (!drafter) return true;
    // the drafter's pairs (main's normed row at cell c, the input row at c + 1): the waiting tail first
    kernels::rms_norm_rows(residual(b), w.out_norm, b.hn + H, T, H, g.eps, s);
    const bool tail = st.tail_cell == pos - 1 && pos > 0;
    if (tail) cudaMemcpyAsync(b.hn, st.tail, (size_t) H * sizeof(float), cudaMemcpyDeviceToDevice, s);
    cudaMemcpyAsync(st.tail, b.hn + (size_t) T * H, (size_t) H * sizeof(float), cudaMemcpyDeviceToDevice, s);
    st.tail_cell = pos + T - 1;
    // the embeddings moved past the normed rows: emb rows [T][H] at b.cat, normed rows at b.hn (+H when no tail)
    const int n = tail ? T : T - 1;
    if (n <= 0) return true;
    return mtp_chunk(m, st, tail ? b.hn : b.hn + H, nullptr, tail ? b.cat : b.cat + H, n, tail ? pos - 1 : pos, err);
}

bool block_ctx_window(Model& m, SessionState& st, int n, int64_t pos, std::string& err) {
    const BlockDrafter& bd = m.bd_dec;
    WindowScratch& b = m.ws;
    const cudaStream_t s = st.stream;
    const int H = m.g.n_embd;
    const int32_t bounds[2] = {0, n};
    cudaMemcpyAsync(b.bounds, bounds, sizeof bounds, cudaMemcpyHostToDevice, s);
    std::vector<int32_t> p3;
    text_positions(p3, n, pos, 0);
    cudaMemcpyAsync(b.bpos, p3.data(), p3.size() * sizeof(int32_t), cudaMemcpyHostToDevice, s);
    const void* xq = wnorm(m, st.dacc, bd.hidden_norm, b.xn, bd.layers[0].k.type, H, n, bd.eps, s);
    const kernels::RopeSpec rope = block_rope(bd);
    for (size_t l = 0; l < bd.layers.size(); ++l) {
        const BlockDrafter::Layer& D = bd.layers[l];
        prod_group(m, b.xn, xq, D.k.type, {{&D.k, b.bk}, {&D.v, b.bv}}, n, s);
        kernels::attn_qk_prepare(nullptr, b.bk, b.bv, nullptr, D.k_norm, bd.eps, rope, b.bpos, n, 0, bd.n_kv, nullptr,
                                 st.dk[l], st.dv[l], pos, s, bd.head_dim, false, st.dctx_cap);
    }
    return cuda_ok("block drafter context", err);
}

bool block_draft(Model& m, SessionState& st, int32_t anchor, int64_t pos, int width, std::string& err) {
    SessionState* sts[1] = {&st};
    return block_draft_batch(m, sts, &anchor, &pos, &width, 1, err);
}

bool block_draft_batch(Model& m, SessionState* const* sts, const int32_t* anchors, const int64_t* pos, const int* widths, int S,
                       std::string& err, cudaStream_t stream) {
    const BlockDrafter& bd = m.bd_dec;
    const Geometry& g = m.g;
    WindowScratch& b = m.ws;
    const cudaStream_t s = stream ? stream : sts[0]->stream;
    const int H = g.n_embd, AQ = bd.n_head * bd.head_dim, KV = bd.n_kv * bd.head_dim;
    const int groups = H / bd.conv_group;
    // session k's rows: r0[k] .. r0[k] + widths[k] - 1
    int r0[kBatch + 1] = {};
    for (int k = 0; k < S; ++k) {
        if (widths[k] < bd.block || widths[k] > kWindow) {
            err = "block drafter: a block of " + std::to_string(widths[k]) + " rows";
            return false;
        }
        r0[k + 1] = r0[k] + widths[k];
    }
    const int R = r0[S];
    if (S < 1 || R > kBatch) {
        err = "block drafter: a batch of " + std::to_string(R) + " rows";
        return false;
    }
    // each block: its anchor, then mask tokens, at positions pos ..
    int32_t tok[kBatch];
    std::vector<int32_t> p3((size_t) R * 3);
    for (int k = 0; k < S; ++k) {
        tok[r0[k]] = anchors[k];
        for (int i = 1; i < widths[k]; ++i) tok[r0[k] + i] = bd.mask_id;
        for (int i = 0; i < widths[k]; ++i) {
            int32_t* q = &p3[(size_t) (r0[k] + i) * 3];
            q[0] = q[1] = q[2] = (int32_t) (pos[k] + i);
        }
        cudaMemcpyAsync(sts[k]->win_tok, &tok[r0[k]], sizeof(int32_t), cudaMemcpyHostToDevice, s);
    }
    const int32_t bounds[2] = {0, R};
    cudaMemcpyAsync(b.bounds, bounds, sizeof bounds, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(b.tok, tok, (size_t) R * sizeof(int32_t), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(b.bpos, p3.data(), p3.size() * sizeof(int32_t), cudaMemcpyHostToDevice, s);
    kernels::PdlScope pdl(true, true);   // after the uploads: each product stages its weights during the kernel before
    embed_rows(m, b.tok, R, s);
    const kernels::RopeSpec rope = block_rope(bd);
    // the dynamic convolution runs along each block's own rows into b.bh
    const auto conv = [&](const float* x, const float* dyn, int64_t ld_dyn, const float* base) {
        for (int k = 0; k < S; ++k)
            kernels::dyn_conv2(x + (size_t) r0[k] * H, dyn + (size_t) r0[k] * ld_dyn, ld_dyn, base, b.bh + (size_t) r0[k] * H,
                               widths[k], H, bd.conv_group, s);
    };
    for (size_t l = 0; l < bd.layers.size(); ++l) {
        const BlockDrafter::Layer& D = bd.layers[l];
        // attention, its input and output through the dynamic convolution
        prod(m, D.conv_a, wnorm(m, b.x, D.attn_norm, b.xn, D.conv_a.type, H, R, bd.eps, s), b.dyn, R, s);
        conv(b.xn, b.dyn, D.conv_a.rows, D.base_a);
        prod_group(m, b.bh, wact(m, b.bh, D.q.type, H, R, s), D.q.type, {{&D.q, b.bq}, {&D.k, b.bk}, {&D.v, b.bv}}, R, s);
        for (int k = 0; k < S; ++k) {
            SessionState& st = *sts[k];
            const size_t q0 = (size_t) r0[k];
            const int B = widths[k];
            kernels::attn_qk_prepare(b.bq + q0 * AQ, b.bk + q0 * KV, b.bv + q0 * KV, D.q_norm, D.k_norm, bd.eps, rope,
                                     b.bpos + q0 * 3, B, bd.n_head, bd.n_kv, b.bqo + q0 * AQ, st.dk[l], st.dv[l], pos[k], s,
                                     bd.head_dim, false, st.dctx_cap);
            const int64_t window = std::min<int64_t>(bd.window, pos[k] - st.dctx_from + 1);   // context below dctx_from is gone
            kernels::gqa_window_attn(b.bqo + q0 * AQ, st.dk[l], st.dv[l], bd.n_head, bd.n_kv, bd.head_dim, pos[k], B, pos[k] + B,
                                     window, st.dctx_cap, b.battn, st.dctx_cap, b.bao + q0 * AQ, s);
        }
        prod(m, D.o, wact(m, b.bao, D.o.type, AQ, R, s), b.tmp, R, s);
        conv(b.tmp, b.dyn + 2 * groups, D.conv_a.rows, D.base_a + 2 * H);
        kernels::add_scaled(b.x, b.bh, nullptr, (int64_t) R * H, s);
        // the FFN, likewise
        prod(m, D.conv_f, wnorm(m, b.x, D.ffn_norm, b.xn, D.conv_f.type, H, R, bd.eps, s), b.dyn, R, s);
        conv(b.xn, b.dyn, D.conv_f.rows, D.base_f);
        prod_swiglu(m, D.gate, D.up, b.bh, wact(m, b.bh, D.gate.type, H, R, s), b.h, R, s);
        prod(m, D.down, wact(m, b.h, D.down.type, D.gate.rows, R, s), b.tmp, R, s);
        conv(b.tmp, b.dyn + 2 * groups, D.conv_f.rows, D.base_f + 2 * H);
        kernels::add_scaled(b.x, b.bh, nullptr, (int64_t) R * H, s);
    }
    // slots 1 .. B-1 of each block: the head's candidates, then one path through them
    // the head over the vocabulary's first rows only (a BPE vocabulary's ids run roughly by frequency): its picks there
    const int64_t nv = draft_vocab(m);
    const void* hq = wnorm(m, b.x, bd.norm, b.xn, m.w.head.type, H, R, bd.eps, s);
    if (nv < g.n_vocab) {
        const Linear& hd = m.w.head;
        wg::Call c = wcall(hd);
        c.m[0] = {(const uint8_t*) hd.w, nv, hd.scale, b.hl, nv, (const uint8_t*) hd.w + kernels::nvfp4p::scale_offset(hd.rows, hd.cols)};
        wg::gemm(c, hq, R, m.ws.w8, s);
    } else {
        prod(m, m.w.head, hq, b.hl, R, s);
    }
    {   // the selector's projection of every row (BF16 weights) on the window gemm
        wg::Call c;
        c.fmt = wg::Format::Bf16;
        c.K = H;
        c.m[0] = {(const uint8_t*) bd.sel_h, bd.rank, nullptr, b.bsel, bd.rank};
        wg::pack(b.xn, H, H, R, b.xq, s);
        wg::gemm(c, b.xq, R, m.ws.w8, s);
    }
    for (int k = 0; k < S; ++k) {
        const size_t q0 = (size_t) r0[k];
        const int B = widths[k];
        kernels::row_logprobs(b.hl + (q0 + 1) * nv, B - 1, nv, b.tok + q0 + 1, bd.top_k, b.blp, b.lp_scratch, s);
        kernels::candidate_path(b.blp, kernels::kLogprobStride, bd.top_k, b.bsel + (q0 + 1) * bd.rank, bd.pred, bd.succ, bd.rank,
                                sts[k]->win_tok, B - 1, sts[k]->dids, sts[k]->dconf, s);
    }
    return cuda_ok("block drafter", err);
}

bool mtp_chunk(Model& m, SessionState& st, const float* hid, const int32_t* tok, const float* emb, int n, int64_t c0,
               std::string& err) {
    const Geometry& g = m.g;
    const Mtp& d = m.w.mtp;
    ChunkScratch& b = m.cs;
    const cudaStream_t s = st.stream;
    const int H = g.n_embd;
    (void) tok;
    // the input [n][2H] goes to b.qkv-sized phase memory: b.g holds n_ff >= 2H floats per row
    float* cat = b.g;
    mtp_input(m, emb, hid, cat, n, s);
    const int32_t bounds[2] = {0, n};
    cudaMemcpyAsync(b.bounds, bounds, sizeof bounds, cudaMemcpyHostToDevice, s);
    std::vector<int32_t> p3;
    text_positions(p3, n, c0, st.rope_delta);
    cudaMemcpyAsync(b.pos, p3.data(), p3.size() * sizeof(int32_t), cudaMemcpyHostToDevice, s);
    quantize_mmq(m, cat, b.xq, d.fc.type, 2 * H, n, s);
    gemm(m, d.fc, b.xq, b.tmp, n, s);
    kernels::to_bf16(b.tmp, residual(b), (int64_t) n * H, s);   // the chunk path's residual is BF16
    chunk_attn(m, d.layer, st.mtp_kc, st.mtp_vc, n, c0, s);
    return cuda_ok("drafter history", err);
}

bool mtp_window(Model& m, SessionState& st, const float* hid, const int32_t* tok, int n, int64_t c0, float* out,
                float* lp_row, std::string& err) {
    const Geometry& g = m.g;
    const Mtp& d = m.w.mtp;
    WindowScratch& b = m.ws;
    const cudaStream_t s = st.stream;
    const int H = g.n_embd;
    std::vector<int32_t> p3;
    text_positions(p3, n, c0, st.rope_delta);
    const int32_t bounds[2] = {0, n}, first[2] = {(int32_t) c0, 0};
    cudaMemcpyAsync(b.bounds, bounds, sizeof bounds, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(b.pos0, first, sizeof first, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(b.pos, p3.data(), p3.size() * sizeof(int32_t), cudaMemcpyHostToDevice, s);
    kernels::iq_embed_rows(30, m.w.embd, (size_t) H * 2, tok, n, H, b.tmp, s);
    mtp_input(m, b.tmp, hid, b.cat, n, s);
    prod(m, d.fc, wact(m, b.cat, d.fc.type, 2 * H, n, s), b.tmp, n, s);
    kernels::to_bf16(b.tmp, b.x, (int64_t) n * H, s);
    win_attn(m, d.layer, st.mtp_kc, st.mtp_vc, n, c0, s);
    if (out) {
        win_ffn(m, d.layer, n, s);
        // the head's rows (scaled), of which the last is read
        prod(m, m.w.head, wnorm(m, b.x, d.norm, b.xn, m.w.head.type, H, n, g.eps, s), b.hl, n, s);
        cudaMemcpyAsync(out, b.xn, (size_t) n * H * sizeof(float), cudaMemcpyDeviceToDevice, s);
        cudaMemcpyAsync(st.dlogits, b.hl + (size_t) (n - 1) * g.n_vocab, (size_t) g.n_vocab * sizeof(float),
                        cudaMemcpyDeviceToDevice, s);
        kernels::row_logprobs(st.dlogits, 1, g.n_vocab, tok + n - 1, 1, lp_row, b.lp_scratch, s);
    }
    return cuda_ok("drafter", err);
}

}  // namespace basalt::dense
