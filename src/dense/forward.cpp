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

#include "basalt/core/on_device.hpp"
#include "basalt/kernels/elementwise.hpp"

#include <algorithm>
#include <atomic>
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
// the scale the step after a prompt chunk product applies: an NVFP4 matrix's on MMQ (its product leaves it out), none after
// F8 and NVFP4 planes (their products apply it)
const float* post_scale(const Linear& l) { return l.type == kTypeF8 || l.type == kTypeNvfp4Planes ? nullptr : l.scale; }
int mmq_tile(int T) { return T <= kNarrow ? kNarrow : kBatch; }
// the vocabulary rows the block drafter's head scores: the first kDraftVocab when the head is in planes or F8, else all
int64_t draft_vocab(const Model& m) {
    const Linear& hd = m.w.head;
    return (hd.type == kTypeNvfp4Planes || hd.type == kTypeF8) && hd.rows > kDraftVocab ? kDraftVocab : m.g.n_vocab;
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
wg::Matrix wmat(const Linear& l, float* y) {
    wg::Matrix w{(const uint8_t*) l.w, l.rows, l.scale, y, l.rows};
    w.row_scales = l.row_scales;
    return w;
}

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

// y [T][rows] = W . x from chunk_act's activations: on MMQ, or for F8 and NVFP4 planes on the FP8 / FP4 tensor cores with
// the tensor scale applied
void gemm(Model& m, const Linear& l, float* y, int T, cudaStream_t s) {
    const void* xq = m.cs.xq;
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

// a prompt chunk's FP32 rows x readied for products of `type`: E4M3 (a scale per row) or MMQ's blocks in cs.xq; for NVFP4
// planes W4A4, E2M1 with a UE4M3 scale per 16 values against a per-row FP32 scale, in cs.a4 and cs.gs
void chunk_act(Model& m, const float* x, int type, int64_t cols, int T, cudaStream_t s) {
    ChunkScratch& b = m.cs;
    if (type == kTypeF8) prefill::fp8::quantize(x, T, cols, cols, b.xq, s);
    else if (type == kTypeNvfp4Planes) prefill::fp4_dense::quantize(x, T, cols, cols, b.a4, b.gs, s);
    else prefill::mmq::quantize(x, nullptr, b.xq, type, cols, cols, T, s);
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
              kernels::KvInt8 q8 = {}, kernels::KvFar far_kv = {}) {
    const Geometry& g = m.g;
    WindowScratch& b = m.ws;
    const int H = g.n_embd;
    const void* xq = wnorm(m, b.x, L.attn_norm, b.xn, L.q.type, H, T, g.eps, s);
    prod_group(m, b.xn, xq, L.q.type, {{&L.q, b.qfull}, {&L.k, b.k}, {&L.v, b.v}}, T, s);
    kernels::attn_qk_prepare(b.qfull, b.k, b.v, L.q_norm, L.k_norm, g.eps, g.rope, b.pos, T, g.n_head, g.n_kv, b.qo, kc, vc,
                             pos, s, 256, true, 0, b.pos0, q8, far_kv);
    const bool gated = on_wg(L.o.type);   // the merge gates and readies the rows for o
    kernels::gqa_decode_attn(b.qo, kc, vc, g.n_head, g.n_kv, pos, T, b.attn, m.o.max_context, gated ? nullptr : b.ao, s, b.pos0, q8,
                             gated ? b.qfull : nullptr, gated ? (uint16_t*) b.xq : nullptr, far_kv);
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

bool all_of(int type, std::initializer_list<const Linear*> ls) {
    for (const Linear* l : ls)
        if (l->type != type) return false;
    return true;
}
bool all_f8(std::initializer_list<const Linear*> ls) { return all_of(kTypeF8, ls); }
bool all_nvfp4(std::initializer_list<const Linear*> ls) { return all_of(kTypeNvfp4Planes, ls); }
prefill::fp8::Segment f8_seg(const Linear& l, void* dst) { return {(const uint8_t*) l.w, l.rows, l.scale, dst, l.rows}; }

// The prompt chunk's residual stream: BF16 rows in ChunkScratch::x's buffer; its activations between the F8 / NVFP4
// products and the GDN and attention kernels are BF16 too (a model on other matrix types keeps FP32 activations).
uint16_t* residual(ChunkScratch& b) { return reinterpret_cast<uint16_t*>(b.x); }
uint16_t* bf(float* p) { return reinterpret_cast<uint16_t*>(p); }

// x += W . a for a chunk's output projection on other matrix types (FP32 a): NVFP4 planes add into the BF16 residual in
// their epilogue (in fp32, rounded once), others through tmp
void chunk_residual(Model& m, const Linear& l, const float* a, int64_t cols, int T, cudaStream_t s) {
    ChunkScratch& b = m.cs;
    chunk_act(m, a, l.type, cols, T, s);
    if (l.type == kTypeNvfp4Planes) {
        prefill::fp4_dense::product_add_bf16(l.w, l.scale, l.rows, cols, b.a4, b.gs, T, residual(b), l.rows, s);
        return;
    }
    gemm(m, l, b.tmp, T, s);
    kernels::add_scaled(residual(b), b.tmp, post_scale(l), (int64_t) T * l.rows, s);
}
// x += W . a for an F8 output projection whose input rows its producer already rounded into b.xq
void chunk_residual_f8(Model& m, const Linear& l, int T, cudaStream_t s) {
    prefill::fp8::gemm((const uint8_t*) l.w, l.rows, l.cols, l.scale, m.cs.xq, T, residual(m.cs), l.rows,
                       prefill::fp8::Epilogue::AddBf16, s);
}
// x += W . a for an NVFP4 planes output projection of BF16 rows a whose producer gave each row's largest |value| (rowmax)
void chunk_residual_fp4(Model& m, const Linear& l, const uint16_t* a, const float* rowmax, int64_t cols, int T, cudaStream_t s) {
    ChunkScratch& b = m.cs;
    prefill::fp4_dense::quantize_amax(a, rowmax, T, cols, cols, b.a4, b.gs, s);
    prefill::fp4_dense::product_add_bf16(l.w, l.scale, l.rows, cols, b.a4, b.gs, T, residual(b), l.rows, s);
}

// K/V streaming, a prompt chunk: an attention layer reads its far cells from the model's VRAM stage instead of host
// memory, one bus copy per layer and chunk rather than one per query tile.  The cells earlier chunks wrote go there on
// the stage stream while the layers before it run (stage_prefetch, once the previous attention layer is done with the
// stage); the chunk's own far cells follow its cache write (stage_far).

// rows [r0, r1) of a far region, host to stage, on s
void stage_rows(Model& m, const kernels::KvFar& far_kv, int64_t r0, int64_t r1, cudaStream_t s) {
    if (r1 <= r0) return;
    const kernels::KvFar& st = m.stage;
    const size_t row = (size_t) m.g.n_kv * m.g.head_dim, o = (size_t) r0 * row, n = (size_t) (r1 - r0) * row;
    if (far_kv.q8.k) {
        cudaMemcpyAsync(st.q8.k + o, far_kv.q8.k + o, n, cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(st.q8.v + o, far_kv.q8.v + o, n, cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(st.q8.ks + o / 64, far_kv.q8.ks + o / 64, n / 64 * 2, cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(st.q8.vs + o / 64, far_kv.q8.vs + o / 64, n / 64 * 2, cudaMemcpyHostToDevice, s);
    } else {
        cudaMemcpyAsync(st.k + o, far_kv.k + o, n * 2, cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(st.v + o, far_kv.v + o, n * 2, cudaMemcpyHostToDevice, s);
    }
}

// the far cells below `pos` (earlier chunks') to the stage on the stage stream, after what s has queued
void stage_prefetch(Model& m, const kernels::KvFar& far_kv, int64_t pos, cudaStream_t s) {
    if (pos <= far_kv.at) return;
    cudaEventRecord(m.stage_free, s);
    cudaStreamWaitEvent(m.stage_side, m.stage_free, 0);
    stage_rows(m, far_kv, 0, pos - far_kv.at, m.stage_side);
    cudaEventRecord(m.stage_ready, m.stage_side);
}

// the cells a chunk at pos.. of T rows reads: `far` itself when they are all on the card, else the stage, completed with
// the chunk's own far cells (after its cache write) once the prefetch is in
kernels::KvFar stage_far(Model& m, const kernels::KvFar& far_kv, int64_t pos, int T, cudaStream_t s) {
    if (pos + T <= far_kv.at) return far_kv;
    if (pos > far_kv.at) cudaStreamWaitEvent(s, m.stage_ready, 0);
    stage_rows(m, far_kv, std::max<int64_t>(0, pos - far_kv.at), pos + T - far_kv.at, s);
    return m.stage;
}

// `cells_only`: nothing reads the layer's rows (a chunk's last layer), so only its K and V cells are made: the same k
// and v products and the same cache write, without the queries, the attention and the output projection.  `far`: a
// streamed cache's far cells (the chunk's rows are written there and read from the stage; stage_prefetch has run).
void chunk_attn(Model& m, const Layer& L, uint16_t* kc, uint16_t* vc, int T, int64_t pos, cudaStream_t s,
                kernels::KvInt8 q8 = {}, bool cells_only = false, kernels::KvFar far_kv = {}) {
    const Geometry& g = m.g;
    ChunkScratch& b = m.cs;
    const int H = g.n_embd, AQ = g.n_head * g.head_dim;
    const bool f8 = all_f8({&L.q, &L.k, &L.v, &L.o}), nvfp4 = all_nvfp4({&L.q, &L.k, &L.v, &L.o});
    if (f8 || nvfp4) {
        // the norm in the quantizer, the q, k and v products BF16 out (F8: one product), the gate's rows straight to E4M3 for
        // an F8 o, else to BF16 rows with each row's amax, read once by o's FP4 quantizer
        const auto product = [&](const Linear& l, float* dst) {
            prefill::fp4_dense::product_bf16(l.w, l.scale, l.rows, H, b.a4, b.gs, T, bf(dst), l.rows, s);
        };
        if (f8) prefill::fp8::rms_norm_quantize(residual(b), L.attn_norm, g.eps, T, H, b.xq, nullptr, s);
        else prefill::fp4_dense::rms_norm_quantize(residual(b), L.attn_norm, g.eps, T, H, b.a4, b.gs, nullptr, s);
        if (cells_only) {
            if (f8) {
                const prefill::fp8::Segment seg[2] = {f8_seg(L.k, b.k), f8_seg(L.v, b.v)};
                prefill::fp8::gemm_segmented(seg, 2, H, b.xq, T, prefill::fp8::Epilogue::StoreBf16, s);
            } else {
                product(L.k, b.k);
                product(L.v, b.v);
            }
            kernels::attn_qk_prepare_bf16(nullptr, bf(b.k), bf(b.v), nullptr, L.k_norm, g.eps, g.rope, b.pos, T, 0, g.n_kv, nullptr,
                                          pos, s, q8, kc, vc, far_kv);
            return;
        }
        if (f8) {
            const prefill::fp8::Segment seg[3] = {f8_seg(L.q, b.qfull), f8_seg(L.k, b.k), f8_seg(L.v, b.v)};
            prefill::fp8::gemm_segmented(seg, 3, H, b.xq, T, prefill::fp8::Epilogue::StoreBf16, s);
        } else {
            product(L.q, b.qfull);
            product(L.k, b.k);
            product(L.v, b.v);
        }
        kernels::attn_qk_prepare_bf16(bf(b.qfull), bf(b.k), bf(b.v), L.q_norm, L.k_norm, g.eps, g.rope, b.pos, T, g.n_head, g.n_kv,
                                      bf(b.qo), pos, s, q8, kc, vc, far_kv);
        const kernels::KvFar read = stage_far(m, far_kv, pos, T, s);
        const size_t split = kernels::gqa_prompt_split_floats(kChunk, g.n_head, kAttnSplits);
        uint8_t* e4 = (uint8_t*) b.xq;
        float* e4s = prefill::fp8::act_scales(b.xq, T, AQ);
        if (q8.k) {   // BF16 queries and output rows
            kernels::gqa_prompt_attn_bf16(bf(b.qo), g.n_head, g.n_kv, pos, T, bf(b.ao), s, q8, b.attn_split, split, read);
            if (f8) kernels::attn_gate_e4m3(bf(b.ao), bf(b.qfull), T, g.n_head, e4, e4s, s);
            else kernels::attn_gate_bf16(bf(b.ao), bf(b.qfull), T, g.n_head, bf(b.ag), b.hmax, s);
        } else {      // an fp16 cache's prompt kernel reads FP32 queries (the gate's BF16 rows reuse their buffer)
            kernels::to_f32(bf(b.qo), b.ag, (int64_t) T * AQ, s);
            kernels::gqa_prompt_attn(b.ag, kc, vc, g.n_head, g.n_kv, pos, T, b.ao, s, q8, b.attn_split, split, read);
            if (f8) kernels::attn_gate_e4m3(b.ao, bf(b.qfull), T, g.n_head, e4, e4s, s);
            else kernels::attn_gate_bf16(b.ao, bf(b.qfull), T, g.n_head, bf(b.ag), b.hmax, s);
        }
        if (f8) chunk_residual_f8(m, L.o, T, s);
        else chunk_residual_fp4(m, L.o, bf(b.ag), b.hmax, AQ, T, s);
        return;
    }
    kernels::rms_norm_rows(residual(b), L.attn_norm, b.xn, T, H, g.eps, s);
    if (cells_only) {
        chunk_act(m, b.xn, L.k.type, H, T, s);
        gemm(m, L.k, b.k, T, s);
        gemm(m, L.v, b.v, T, s);
        kernels::attn_qk_prepare(nullptr, b.k, b.v, nullptr, L.k_norm, g.eps, g.rope, b.pos, T, 0, g.n_kv, nullptr, kc, vc, pos, s,
                                 256, true, 0, nullptr, q8, far_kv);
        return;
    }
    chunk_act(m, b.xn, L.q.type, H, T, s);
    gemm(m, L.q, b.qfull, T, s);
    if (L.k.type != L.q.type) chunk_act(m, b.xn, L.k.type, H, T, s);
    gemm(m, L.k, b.k, T, s);
    gemm(m, L.v, b.v, T, s);
    kernels::attn_qk_prepare(b.qfull, b.k, b.v, L.q_norm, L.k_norm, g.eps, g.rope, b.pos, T, g.n_head, g.n_kv, b.qo, kc, vc,
                             pos, s, 256, true, 0, nullptr, q8, far_kv);
    kernels::gqa_prompt_attn(b.qo, kc, vc, g.n_head, g.n_kv, pos, T, b.ao, s, q8, b.attn_split,
                             kernels::gqa_prompt_split_floats(kChunk, g.n_head, kAttnSplits), stage_far(m, far_kv, pos, T, s));
    kernels::attn_gate(b.ao, b.qfull, b.ag, T, g.n_head, s);
    chunk_residual(m, L.o, b.ag, AQ, T, s);
}

void chunk_gdn(Model& m, SessionState& st, const Layer& L, int gi, int T, cudaStream_t s) {
    const Geometry& g = m.g;
    ChunkScratch& b = m.cs;
    const int H = g.n_embd, VD = g.value_width(), HV = g.gdn_v_heads;
    float* state = st.gdn + (size_t) gi * st.gdn_floats;
    float* hist = state + (size_t) g.gdn_state * HV * g.gdn_state;
    const bool f8 = all_f8({&L.qkv, &L.z, &L.out}), nvfp4 = all_nvfp4({&L.qkv, &L.z, &L.out});
    if (f8 || nvfp4) {
        // the norm in the quantizer (the normed rows kept for a/b, BF16), the qkv and z products BF16 out (F8: one product),
        // the conv and the recurrence on BF16 rows, the closing norm straight to E4M3 for an F8 out, else to BF16 rows with
        // each row's amax, read once by out's FP4 quantizer
        if (f8) {
            prefill::fp8::rms_norm_quantize(residual(b), L.attn_norm, g.eps, T, H, b.xq, bf(b.xn), s);
            const prefill::fp8::Segment seg[2] = {f8_seg(L.qkv, b.qkv), f8_seg(L.z, b.z)};
            prefill::fp8::gemm_segmented(seg, 2, H, b.xq, T, prefill::fp8::Epilogue::StoreBf16, s);
        } else {
            prefill::fp4_dense::rms_norm_quantize(residual(b), L.attn_norm, g.eps, T, H, b.a4, b.gs, bf(b.xn), s);
            prefill::fp4_dense::product_bf16(L.qkv.w, L.qkv.scale, L.qkv.rows, H, b.a4, b.gs, T, bf(b.qkv), L.qkv.rows, s);
            prefill::fp4_dense::product_bf16(L.z.w, L.z.scale, L.z.rows, H, b.a4, b.gs, T, bf(b.z), L.z.rows, s);
        }
        kernels::gdn_gating_proj(bf(b.xn), L.w_ab, L.dt, L.ssm_a, b.gate, b.beta, T, HV, H, s);
        kernels::gdn_conv(hist, bf(b.qkv), L.conv, bf(b.hc), T, s);
        if (f8) {
            kernels::gdn_recurrence_bf16(state, bf(b.hc), b.gate, b.beta, bf(b.z), L.gamma, g.eps, b.work, b.y, T, s, g.gdn_gate,
                                         (uint8_t*) b.xq, prefill::fp8::act_scales(b.xq, T, VD));
            chunk_residual_f8(m, L.out, T, s);
            return;
        }
        kernels::gdn_recurrence_bf16(state, bf(b.hc), b.gate, b.beta, bf(b.z), L.gamma, g.eps, b.work, b.y, T, s, g.gdn_gate,
                                     nullptr, nullptr, bf(b.yn), b.hmax);
        chunk_residual_fp4(m, L.out, bf(b.yn), b.hmax, VD, T, s);
        return;
    }
    {
        kernels::rms_norm_rows(residual(b), L.attn_norm, b.xn, T, H, g.eps, s);
        chunk_act(m, b.xn, L.qkv.type, H, T, s);
        gemm(m, L.qkv, b.qkv, T, s);
        if (L.z.type != L.qkv.type) chunk_act(m, b.xn, L.z.type, H, T, s);
        gemm(m, L.z, b.z, T, s);
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
        prefill::fp4_dense::rms_norm_quantize(residual(b), L.ffn_norm, g.eps, T, H, b.a4, b.gs, nullptr, s);
        prefill::fp4_dense::swiglu(L.gate.w, L.up.w, L.gate.scale, L.up.scale, L.gate.rows, H, b.a4, b.gs, T, h16, g.n_ff, s,
                                   b.hmax);
        prefill::fp4_dense::quantize_amax(h16, b.hmax, T, g.n_ff, g.n_ff, b.a4, b.gs, s);
        prefill::fp4_dense::product_add_bf16(L.down.w, L.down.scale, L.down.rows, g.n_ff, b.a4, b.gs, T, residual(b), H, s);
        return;
    }
    kernels::rms_norm_rows(residual(b), L.ffn_norm, b.xn, T, H, g.eps, s);
    chunk_act(m, b.xn, L.gate.type, H, T, s);
    gemm(m, L.gate, b.g, T, s);
    if (L.up.type != L.gate.type) chunk_act(m, b.xn, L.up.type, H, T, s);
    gemm(m, L.up, b.u, T, s);
    kernels::swiglu_scaled(b.g, b.u, L.gate.scale, L.up.scale, b.h, T, g.n_ff, g.n_ff, s);
    chunk_act(m, b.h, L.down.type, g.n_ff, T, s);
    gemm(m, L.down, b.tmp, T, s);
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
    if (m.n_gdn == 0 || n <= 0) return;
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
    a.n_layers = m.n_gdn;
    a.n_keep = st.n_keep;
    kernels::gdn_commit_layers(a, s, 64, n);
}

namespace {

// The block drafter's features of target layer j: its rows (`rows16`, the BF16 residual after it) through its block of the
// projection into `dacc` (j == 0 writes it, later ones add).
void win_tap(Model& m, size_t j, const uint16_t* rows16, float* dacc, int T, cudaStream_t s) {
    const Linear& fc = m.bd_dec.fc[j];
    const void* xq = rows16;
    if (!on_wg(fc.type)) {
        kernels::to_f32(rows16, m.ws.tmp, (int64_t) T * fc.cols, s);
        xq = wact(m, m.ws.tmp, fc.type, fc.cols, T, s);
    }
    prod(m, fc, xq, j == 0 ? dacc : m.ws.tmp, T, s);
    if (j > 0) kernels::add_scaled(dacc, m.ws.tmp, nullptr, (int64_t) T * m.g.n_embd, s);
}

// A window's layers [m.lb, m.le) of `rows` rows in m.ws.x, the drafter's features of its target layers into `dacc`; on a layer
// split, the window's first rows from the main card (the second card's model) and its later layers' rows from the second
// card (the main card's), each card raising the other's flag (SplitHand).  `layer` runs one layer.
template <class LayerFn>
void window_layers(Model& m, int rows, float* dacc, LayerFn layer, cudaStream_t s) {
    const Geometry& g = m.g;
    WindowScratch& b = m.ws;
    const SplitHand* h = m.hand;
    const int64_t half = (int64_t) rows * g.n_embd / 2;   // the rows' BF16 values, as floats for the mapped copies
    const size_t slot = (size_t) kBatch * g.n_embd;
    const BlockDrafter& bd = m.bd_dec;
    const bool block = m.o.drafter == Drafter::Block;
    if (m.lb > 0) {
        kernels::wait_flag_ge(h->flag, 1, s);
        kernels::copy_from_mapped((float*) b.x, (const float*) h->fwd, half, s);
    }
    size_t next_target = 0, next_tap = 0;
    for (int il = m.lb; il < m.le; ++il) {
        layer(il);
        if (block && next_target < bd.target_layers.size() && bd.target_layers[next_target] == il) win_tap(m, next_target++, b.x, dacc, rows, s);
        if (m.lb > 0 && next_tap < h->taps.size() && h->taps[next_tap] == il)
            kernels::copy_from_mapped((float*) (h->ret + (1 + next_tap++) * slot), (const float*) b.x, half, s);
    }
    if (m.lb > 0) {   // the second card: its last rows back
        kernels::copy_from_mapped((float*) h->ret, (const float*) b.x, half, s);
        kernels::signal_flag(h->flag + 1, 1, s);
    } else if (h) {   // the main card: on to the second card, then its rows back and the features of its target layers
        kernels::copy_from_mapped((float*) h->fwd, (const float*) b.x, half, s);
        kernels::signal_flag(h->flag, 1, s);
        kernels::wait_flag_ge(h->flag + 1, 1, s);
        kernels::copy_from_mapped((float*) b.x, (const float*) h->ret, half, s);
        for (size_t q = 0; block && q < h->taps.size(); ++q) {
            kernels::copy_from_mapped((float*) b.hq, (const float*) (h->ret + (1 + q) * slot), half, s);
            win_tap(m, h->first_tap + q, (const uint16_t*) b.hq, dacc, rows, s);
        }
    }
}

// The window's work on the device, all of it from device inputs (tokens, positions, rows, first cell), so it can be
// captured once per row count and replayed at any position.
void window_body(Model& m, SessionState& st, int T, int64_t pos) {
    const Geometry& g = m.g;
    const Weights& w = m.w;
    WindowScratch& b = m.ws;
    const cudaStream_t s = st.stream;
    const int H = g.n_embd;
    kernels::PdlScope pdl(true);   // the window gemm and its packs start early: their weights are read meanwhile
    if (m.lb == 0) embed_rows(m, b.tok, T, s);
    window_layers(m, T, st.dacc, [&](int il) {
        const Layer& L = w.layers[(size_t) il];
        if (L.kind == Mixer::Gdn) {
            win_gdn(m, st, L, m.gdn_index[(size_t) il], T, s);
        } else {
            const int ai = m.attn_index[(size_t) il];
            win_attn(m, L, st.kc[(size_t) ai], st.vc[(size_t) ai], T, pos, s, st.kq[(size_t) ai], st.far_kv[(size_t) ai]);
        }
        win_ffn(m, L, T, s);
    }, s);
    if (m.lb > 0) return;   // the second card of a split has no head
    prod(m, w.head, wnorm(m, b.x, w.out_norm, b.xn, w.head.type, H, T, g.eps, s), st.logits, T, s);
    if (st.hid) cudaMemcpyAsync(st.hid, b.xn, (size_t) T * H * sizeof(float), cudaMemcpyDeviceToDevice, s);   // the drafter's
}

}  // namespace

namespace {

// One card's window: its inputs uploaded, then its graph for T rows (the first window of T rows runs as launched, then is
// captured for the next ones).  `tokens` null: the card takes its rows from the other (a split's second card).
bool launch_window(Model& m, SessionState& st, const int32_t* tokens, int T, int64_t pos, const int32_t* pos3, std::string& err) {
    WindowScratch& b = m.ws;
    const cudaStream_t s = st.stream;
    const int32_t bounds[2] = {0, T}, first[2] = {(int32_t) pos, 0};
    cudaMemcpyAsync(b.bounds, bounds, sizeof bounds, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(b.pos0, first, sizeof first, cudaMemcpyHostToDevice, s);
    if (tokens) cudaMemcpyAsync(b.tok, tokens, (size_t) T * sizeof(int32_t), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(b.pos, pos3, (size_t) T * 3 * sizeof(int32_t), cudaMemcpyHostToDevice, s);
    // a one-row window writes the other state buffer (st.alt_ready); the graph is the one of the buffer in use
    st.alt_ready = T == 1 && st.gdn_alt != nullptr;
    cudaGraphExec_t& wg = st.wg[st.gdn != st.gdn_first ? 1 : 0][T];
    if (wg) {
        if (cudaGraphLaunch(wg, s) != cudaSuccess) return cuda_ok("dense window graph", err) && false;
        return cuda_ok("dense window", err);
    }
    // The first window of T rows runs as launched, then is captured for the next ones.  A layer split's window waits on the
    // other card's inside its graph: it is captured first and only its graph runs, so no kernel loads lazily (which waits for
    // the card to idle) while a wait holds the card.
    const bool split = m.hand != nullptr;
    if (!split) {
        window_body(m, st, T, pos);
        if (!cuda_ok("dense window", err)) return false;
    }
    cudaGraph_t graph = nullptr;
    if (cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed) != cudaSuccess) return cuda_ok("dense window capture", err);
    window_body(m, st, T, pos);
    if (cudaStreamEndCapture(s, &graph) != cudaSuccess || cudaGraphInstantiate(&wg, graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        wg = nullptr;
        return cuda_ok("dense window capture", err);
    }
    cudaGraphDestroy(graph);
    if (split && cudaGraphLaunch(wg, s) != cudaSuccess) return cuda_ok("dense window graph", err) && false;
    return cuda_ok("dense window", err);
}

// The split's flags lowered for the next window: both cards are past the last one's (the main card's stream has met the
// second card's flag, and the second card met the main card's before raising it).
void lower_flags(SplitHand& h) {
    volatile uint32_t* f = h.flag;
    f[0] = 0;
    f[1] = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

}  // namespace

bool forward_window(Model& m, SessionState& st, const int32_t* tokens, int T, int64_t pos, const int32_t* pos3,
                    std::string& err) {
    if (m.split) {   // the second card's graph first: it waits for the main card's rows
        lower_flags(*m.hand);
        const core::OnDevice on(m.split_dev);
        if (!launch_window(*m.split_model, *st.peer, nullptr, T, pos, pos3, err)) return false;
    }
    return launch_window(m, st, tokens, T, pos, pos3, err);
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
                                 st.kq[(size_t) ai], st.far_kv[(size_t) ai]);
        as[k].T = T;
        as[k].k_cache = st.kc[(size_t) ai];
        as[k].v_cache = st.vc[(size_t) ai];
        as[k].pos0 = segs[k].pos;
        as[k].pos_dev = p0;
        as[k].q8 = st.kq[(size_t) ai];
        as[k].far_kv = st.far_kv[(size_t) ai];
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
        if (m.lb == 0) embed_rows(m, b.tok, rows, s);
        // the block drafter's features packed in b.dacc until every session has its own
        window_layers(m, rows, b.dacc, [&](int il) {
            const Layer& L = w.layers[(size_t) il];
            if (L.kind == Mixer::Gdn) batch_gdn(m, L, m.gdn_index[(size_t) il], segs, S, rows, max_n, s);
            else batch_attn(m, L, m.attn_index[(size_t) il], segs, S, rows, s);
            win_ffn(m, L, rows, s);
        }, s);
        if (m.lb > 0) return;   // the second card of a split has no head
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

namespace {

// One card's batch window (forward_batch); `tokens` null on a split's second card.
bool launch_batch(Model& m, const BatchSeg* segs, int S, const int32_t* tokens, const int32_t* pos3, std::string& err) {
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
    if (tokens) cudaMemcpyAsync(b.tok, tokens, (size_t) rows * sizeof(int32_t), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(b.pos, pos3, (size_t) rows * 3 * sizeof(int32_t), cudaMemcpyHostToDevice, s);
    for (Model::BatchGraph& bg : m.batch_graphs)
        if (bg.key == key) {
            bg.used = ++m.batch_clock;
            if (cudaGraphLaunch(bg.exec, s) != cudaSuccess) return cuda_ok("dense batch graph", err) && false;
            return cuda_ok("dense batch window", err);
        }
    // a composition's first window runs as launched, then is captured for the next ones (a layer split's: only captured, as
    // launch_window's)
    const bool split = m.hand != nullptr;
    if (!split) {
        batch_body(m, segs, S, rows, max_n);
        if (!cuda_ok("dense batch window", err)) return false;
    }
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
    if (split && cudaGraphLaunch(exec, s) != cudaSuccess) return cuda_ok("dense batch graph", err) && false;
    return cuda_ok("dense batch window", err);
}

}  // namespace

bool forward_batch(Model& m, const BatchSeg* segs, int S, const int32_t* tokens, const int32_t* pos3, std::string& err) {
    if (m.split) {   // the second card's sessions, their graph first: it waits for the main card's rows
        lower_flags(*m.hand);
        std::vector<BatchSeg> ps((size_t) S);
        for (int k = 0; k < S; ++k) ps[(size_t) k] = {segs[k].st->peer.get(), segs[k].T, segs[k].pos};
        const core::OnDevice on(m.split_dev);
        if (!launch_batch(*m.split_model, ps.data(), S, nullptr, pos3, err)) return false;
    }
    return launch_batch(m, segs, S, tokens, pos3, err);
}

void chunk_start(Model& m, SessionState& st, const int32_t* tokens, const float* rows, int T, const int32_t* pos3,
                 const int32_t* bounds) {
    const Geometry& g = m.g;
    ChunkScratch& b = m.cs;
    const cudaStream_t s = st.stream;
    const int H = g.n_embd;
    const int32_t own[2] = {0, T};
    cudaMemcpyAsync(b.bounds, bounds ? bounds : own, sizeof own, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(b.pos, pos3, (size_t) T * 3 * sizeof(int32_t), cudaMemcpyHostToDevice, s);
    if (tokens == nullptr && rows == nullptr) return;   // the residual is placed by the caller
    // the chunk's input rows in FP32 (tmp), then the BF16 residual (the BF16 embedding rows exactly)
    if (rows != nullptr) {
        cudaMemcpyAsync(b.tmp, rows, (size_t) T * H * sizeof(float), cudaMemcpyHostToDevice, s);
    } else {
        cudaMemcpyAsync(b.tok, tokens, (size_t) T * sizeof(int32_t), cudaMemcpyHostToDevice, s);
        kernels::iq_embed_rows(30, m.w.embd, (size_t) H * 2, b.tok, T, H, b.tmp, s);
    }
    kernels::to_bf16(b.tmp, residual(b), (int64_t) T * H, s);
    // the drafter's history needs the chunk's inputs: its next-token embeddings are the chunk's own rows 1..T-1
    if (st.mtp_kc != nullptr) cudaMemcpyAsync(b.cat, b.tmp, (size_t) T * H * sizeof(float), cudaMemcpyDeviceToDevice, s);
}

void chunk_layers(Model& m, SessionState& st, int T, int64_t pos, int l0, int l1, const BlockDrafter* taps, bool read_last) {
    const Geometry& g = m.g;
    const Weights& w = m.w;
    const cudaStream_t s = st.stream;
    // A chunk gives no logits (a prompt's last token runs as a decode window), so the last layer's rows are read only by
    // a drafter: the MTP history takes every row's final hidden state, a block drafter the rows of its target layers.
    // When neither reads them, the last layer leaves only what the session keeps: an attention layer its cells; its FFN
    // is not run.
    const bool last_read = read_last || st.mtp_kc != nullptr ||
                           (taps && std::find(taps->target_layers.begin(), taps->target_layers.end(), g.n_layers - 1) !=
                                        taps->target_layers.end());
    // K/V streaming: the next attention layer that attends gets its earlier far cells staged meanwhile
    const auto prefetch_from = [&](int il0) {
        if (m.stage.at >= pos) return;
        for (int j = il0; j < l1; ++j)
            if (w.layers[(size_t) j].kind == Mixer::Attention) {
                if (j < g.n_layers - 1 || last_read) stage_prefetch(m, st.far_kv[(size_t) m.attn_index[(size_t) j]], pos, s);
                return;
            }
    };
    prefetch_from(l0);
    size_t next_target = 0;
    if (taps)
        while (next_target < taps->target_layers.size() && taps->target_layers[next_target] < l0) ++next_target;
    for (int il = l0; il < l1; ++il) {
        const Layer& L = w.layers[(size_t) il];
        const bool unread = il == g.n_layers - 1 && !last_read;
        if (L.kind == Mixer::Gdn) {
            chunk_gdn(m, st, L, m.gdn_index[(size_t) il], T, s);
        } else {
            const int ai = m.attn_index[(size_t) il];
            chunk_attn(m, L, st.kc[(size_t) ai], st.vc[(size_t) ai], T, pos, s, st.kq[(size_t) ai], unread, st.far_kv[(size_t) ai]);
            if (!unread) prefetch_from(il + 1);
        }
        if (unread) break;
        chunk_ffn(m, L, T, s);
        if (taps && next_target < taps->target_layers.size() && taps->target_layers[next_target] == il)
            chunk_tap(m, st, T, *taps, next_target++);
    }
}

void chunk_tap(Model& m, SessionState& st, int T, const BlockDrafter& taps, size_t j) {
    ChunkScratch& b = m.cs;
    const cudaStream_t s = st.stream;
    const int H = m.g.n_embd;
    const Linear& fc = taps.fc[j];
    kernels::to_f32(residual(b), b.xn, (int64_t) T * H, s);   // the target layer's rows for the drafter's fc
    chunk_act(m, b.xn, fc.type, H, T, s);
    gemm(m, fc, j == 0 ? b.dacc : b.tmp, T, s);
    if (j > 0) kernels::add_scaled(b.dacc, b.tmp, nullptr, (int64_t) T * H, s);
}

void block_ctx_chunk(Model& m, SessionState& st, const float* dacc, int n, int64_t pos) {
    const BlockDrafter& bd = m.w.dflash;
    ChunkScratch& b = m.cs;
    const cudaStream_t s = st.stream;
    const int H = m.g.n_embd;
    kernels::rms_norm_rows(dacc, bd.hidden_norm, b.xn, n, H, bd.eps, s);
    chunk_act(m, b.xn, bd.layers[0].k.type, H, n, s);
    std::vector<int32_t> p3;
    text_positions(p3, n, pos, 0);
    cudaMemcpyAsync(b.pos, p3.data(), p3.size() * sizeof(int32_t), cudaMemcpyHostToDevice, s);
    kernels::RopeSpec rope = block_rope(bd);
    for (size_t l = 0; l < bd.layers.size(); ++l) {
        const BlockDrafter::Layer& D = bd.layers[l];
        gemm(m, D.k, b.k, n, s);
        gemm(m, D.v, b.v, n, s);
        kernels::attn_qk_prepare(nullptr, b.k, b.v, nullptr, D.k_norm, bd.eps, rope, b.pos, n, 0, bd.n_kv, nullptr,
                                 st.dk[l], st.dv[l], pos, s, bd.head_dim, false, st.dctx_cap);
    }
}

bool chunk_finish(Model& m, SessionState& st, int T, int64_t pos, std::string& err) {
    const Geometry& g = m.g;
    const Weights& w = m.w;
    ChunkScratch& b = m.cs;
    const cudaStream_t s = st.stream;
    const int H = g.n_embd;
    if (m.o.drafter == Drafter::Block) block_ctx_chunk(m, st, b.dacc, T, pos);   // the chunk's rows become its context
    if (!cuda_ok("dense chunk", err)) return false;
    if (st.mtp_kc == nullptr) return true;
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

bool forward_chunk(Model& m, SessionState& st, const int32_t* tokens, const float* rows, int T, int64_t pos,
                   const int32_t* pos3, std::string& err) {
    if (m.split) return split_chunk(m, st, tokens, rows, T, pos, pos3, err);
    if (!m.ensure_chunk(err)) return false;
    chunk_start(m, st, tokens, rows, T, pos3);
    chunk_layers(m, st, T, pos, 0, m.g.n_layers, m.o.drafter == Drafter::Block ? &m.w.dflash : nullptr);
    return chunk_finish(m, st, T, pos, err);
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
        const Linear& hd = m.w.head;   // its first nv rows: F8 rows (and their scales) lead; planes keep the whole scale plane
        wg::Call c = wcall(hd);
        c.m[0] = wmat(hd, b.hl);
        c.m[0].rows = c.m[0].ld_y = nv;
        if (hd.type == kTypeNvfp4Planes) c.m[0].scales = (const uint8_t*) hd.w + kernels::nvfp4p::scale_offset(hd.rows, hd.cols);
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
        kernels::CandidateDraw draw;
        draw.temperature = sts[k]->dtemp;
        draw.seed = sts[k]->dseed;
        draw.pos0 = pos[k] + 1;
        draw.q_ids = sts[k]->dq_ids;
        draw.q_p = sts[k]->dq_p;
        kernels::candidate_path(b.blp, kernels::kLogprobStride, bd.top_k, b.bsel + (q0 + 1) * bd.rank, bd.pred, bd.succ, bd.rank,
                                sts[k]->win_tok, B - 1, sts[k]->dids, sts[k]->dconf, draw, s);
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
    chunk_act(m, cat, d.fc.type, 2 * H, n, s);
    gemm(m, d.fc, b.tmp, n, s);
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
