// src/dense/forward.cpp - the layer loop of a decode window and of a prompt chunk, and the drafter's passes.
#include "internal.hpp"

#include "basalt/kernels/gdn_prompt.hpp"
#include "basalt/kernels/gqa_attn.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/logprobs.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/prefill/moe_fp4.hpp"

#include <algorithm>
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

// A decode window's products run on MMQ with the tile of a full window whatever the window's rows, so a row's output
// does not depend on how many rows the window has: the activations are always quantized for kWindow rows (the rows past
// T hold finite stale values) and the bounds {0, T} say which rows are written.
void wquant(const float* x, void* xq, int type, int64_t cols, cudaStream_t s) {
    prefill::mmq::quantize(x, nullptr, xq, type, cols, cols, kWindow, s);
}
void gemv(Model& m, const Linear& l, const void* xq, float* y, cudaStream_t s) {
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
    p.total_rows = kWindow;
    p.max_rows = kWindow;
    p.opt_rows = kWindow;
    p.dst = y;
    p.ld_dst = l.rows;
    m.mmq->run(p, s);
}

// y [T][rows] = W . x on MMQ (xq quantized for l.type)
void gemm(Model& m, const Linear& l, const void* xq, float* y, int T, cudaStream_t s) {
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

void quantize_mmq(const float* x, void* xq, int type, int64_t cols, int T, cudaStream_t s) {
    prefill::mmq::quantize(x, nullptr, xq, type, cols, cols, T, s);
}

// rope positions [T][3] of cells pos.. for text rows
void text_positions(std::vector<int32_t>& p3, int T, int64_t pos, int64_t delta) {
    p3.resize((size_t) T * 3);
    for (int t = 0; t < T; ++t) p3[(size_t) t * 3] = p3[(size_t) t * 3 + 1] = p3[(size_t) t * 3 + 2] = (int32_t) (pos + t + delta);
}

// ---- decode window layers: the residual rows are m.ws.x

void win_attn(Model& m, const Layer& L, uint16_t* kc, uint16_t* vc, int T, int64_t pos, cudaStream_t s,
              kernels::KvInt8 q8 = {}) {
    const Geometry& g = m.g;
    WindowScratch& b = m.ws;
    const int H = g.n_embd, AQ = g.n_head * g.head_dim;
    kernels::rms_norm_rows(b.x, L.attn_norm, b.xn, T, H, g.eps, s);
    wquant(b.xn, b.xq, L.q.type, H, s);
    gemv(m, L.q, b.xq, b.qfull, s);
    if (L.k.type != L.q.type) wquant(b.xn, b.xq, L.k.type, H, s);
    gemv(m, L.k, b.xq, b.k, s);
    gemv(m, L.v, b.xq, b.v, s);
    kernels::attn_qk_prepare(b.qfull, b.k, b.v, L.q_norm, L.k_norm, g.eps, g.rope, b.pos, T, g.n_head, g.n_kv, b.qo, kc, vc,
                             pos, s, 256, true, 0, b.pos0, q8);
    kernels::gqa_decode_attn(b.qo, kc, vc, g.n_head, g.n_kv, pos, T, b.attn, m.o.max_context, b.ao, s, b.pos0, q8);
    kernels::attn_gate(b.ao, b.qfull, b.ag, T, g.n_head, s);
    wquant(b.ag, b.yq, L.o.type, AQ, s);
    gemv(m, L.o, b.yq, b.tmp, s);
    kernels::add_scaled(b.x, b.tmp, L.o.scale, (int64_t) T * H, s);
}

void win_gdn(Model& m, SessionState& st, const Layer& L, int gi, int T, cudaStream_t s) {
    const Geometry& g = m.g;
    WindowScratch& b = m.ws;
    const int H = g.n_embd, C = g.conv_channels();
    kernels::rms_norm_rows(b.x, L.attn_norm, b.xn, T, H, g.eps, s);
    wquant(b.xn, b.xq, L.qkv.type, H, s);
    float* qkv = st.rec_qkv + (size_t) gi * kWindow * C;
    gemv(m, L.qkv, b.xq, qkv, s);
    if (L.z.type != L.qkv.type) wquant(b.xn, b.xq, L.z.type, H, s);
    gemv(m, L.z, b.xq, b.z, s);
    kernels::DecRecArgs a;
    a.state = st.gdn + (size_t) gi * st.gdn_floats;
    a.qkv = qkv;
    a.conv_w = L.conv;
    a.x = b.xn;
    a.w_alpha = L.w_ab;
    a.w_beta = L.w_ab + (size_t) g.gdn_v_heads * H;
    a.dt = L.dt;
    a.ssm_a = L.ssm_a;
    a.z = b.z;
    a.gamma = L.gamma;
    a.h = st.rec_h + (size_t) gi * kWindow * C;
    a.gate = st.rec_gate + (size_t) gi * kWindow * g.gdn_v_heads;
    a.beta = st.rec_beta + (size_t) gi * kWindow * g.gdn_v_heads;
    a.yq = b.yq;
    a.y = b.y;
    a.C = C;
    a.n_embd = H;
    a.h_k = g.gdn_k_heads;
    a.h_v = g.gdn_v_heads;
    a.eps = g.eps;
    a.early_trigger = false;
    a.gate_act = g.gdn_gate;
    kernels::dec_gdn_rec(a, T, s);
    wquant(b.y, b.yq, L.out.type, g.value_width(), s);
    gemv(m, L.out, b.yq, b.tmp, s);
    kernels::add_scaled(b.x, b.tmp, L.out.scale, (int64_t) T * H, s);
}

void win_ffn(Model& m, const Layer& L, int T, cudaStream_t s) {
    const Geometry& g = m.g;
    WindowScratch& b = m.ws;
    const int H = g.n_embd;
    kernels::rms_norm_rows(b.x, L.ffn_norm, b.xn, T, H, g.eps, s);
    wquant(b.xn, b.xq, L.gate.type, H, s);
    gemv(m, L.gate, b.xq, b.g, s);
    if (L.up.type != L.gate.type) wquant(b.xn, b.xq, L.up.type, H, s);
    gemv(m, L.up, b.xq, b.u, s);
    kernels::swiglu_scaled(b.g, b.u, L.gate.scale, L.up.scale, b.h, T, g.n_ff, g.n_ff, s);
    wquant(b.h, b.yq, L.down.type, g.n_ff, s);
    gemv(m, L.down, b.yq, b.tmp, s);
    kernels::add_scaled(b.x, b.tmp, L.down.scale, (int64_t) T * H, s);
}

// ---- prompt chunk layers: the residual rows are m.cs.x

void chunk_attn(Model& m, const Layer& L, uint16_t* kc, uint16_t* vc, int T, int64_t pos, cudaStream_t s,
                kernels::KvInt8 q8 = {}) {
    const Geometry& g = m.g;
    ChunkScratch& b = m.cs;
    const int H = g.n_embd, AQ = g.n_head * g.head_dim;
    kernels::rms_norm_rows(b.x, L.attn_norm, b.xn, T, H, g.eps, s);
    quantize_mmq(b.xn, b.xq, L.q.type, H, T, s);
    gemm(m, L.q, b.xq, b.qfull, T, s);
    if (L.k.type != L.q.type) quantize_mmq(b.xn, b.xq, L.k.type, H, T, s);
    gemm(m, L.k, b.xq, b.k, T, s);
    gemm(m, L.v, b.xq, b.v, T, s);
    kernels::attn_qk_prepare(b.qfull, b.k, b.v, L.q_norm, L.k_norm, g.eps, g.rope, b.pos, T, g.n_head, g.n_kv, b.qo, kc, vc,
                             pos, s, 256, true, 0, nullptr, q8);
    kernels::gqa_prompt_attn(b.qo, kc, vc, g.n_head, g.n_kv, pos, T, b.ao, s, q8);
    kernels::attn_gate(b.ao, b.qfull, b.ag, T, g.n_head, s);
    quantize_mmq(b.ag, b.xq, L.o.type, AQ, T, s);
    gemm(m, L.o, b.xq, b.tmp, T, s);
    kernels::add_scaled(b.x, b.tmp, L.o.scale, (int64_t) T * H, s);
}

void chunk_gdn(Model& m, SessionState& st, const Layer& L, int gi, int T, cudaStream_t s) {
    const Geometry& g = m.g;
    ChunkScratch& b = m.cs;
    const int H = g.n_embd, VD = g.value_width(), HV = g.gdn_v_heads;
    float* state = st.gdn + (size_t) gi * st.gdn_floats;
    float* hist = state + (size_t) g.gdn_state * HV * g.gdn_state;
    kernels::rms_norm_rows(b.x, L.attn_norm, b.xn, T, H, g.eps, s);
    quantize_mmq(b.xn, b.xq, L.qkv.type, H, T, s);
    gemm(m, L.qkv, b.xq, b.qkv, T, s);
    if (L.z.type != L.qkv.type) quantize_mmq(b.xn, b.xq, L.z.type, H, T, s);
    gemm(m, L.z, b.xq, b.z, T, s);
    kernels::bf16_rows_dot(b.xn, L.w_ab, b.ab, T, 2 * HV, H, s);
    kernels::gdn_gates(b.ab, L.dt, L.ssm_a, b.gate, b.beta, T, s);
    kernels::gdn_conv(hist, b.qkv, L.conv, b.hc, T, s);
    kernels::gdn_recurrence(state, b.hc, b.gate, b.beta, b.z, L.gamma, g.eps, b.work, b.y, nullptr, T, s, g.gdn_gate, b.yn);
    quantize_mmq(b.yn, b.xq, L.out.type, VD, T, s);
    gemm(m, L.out, b.xq, b.tmp, T, s);
    kernels::add_scaled(b.x, b.tmp, L.out.scale, (int64_t) T * H, s);
}

// y [T][rows] = W . x for an NVFP4 matrix on the FP4 tensor cores, x already rounded to FP4 (fp4_quant)
void gemm_fp4(Model& m, const Linear& l, float* y, int T, cudaStream_t s) {
    prefill::fp4::Product p;
    p.blob[0] = (const uint8_t*) l.w;
    p.w_rows = l.rows;
    p.w_cols = l.cols;
    p.n = 1;
    p.a4 = m.cs.a4;
    p.gs = m.cs.gs;
    p.bounds = m.cs.bounds;
    p.tile_off = m.cs.tile_off;
    p.dst = y;
    p.ld_dst = l.rows;
    p.max_rows = T;
    prefill::fp4::run(p, s);
}
void fp4_quant(Model& m, const float* x, int64_t cols, int T, cudaStream_t s) {
    prefill::fp4::plan(m.cs.bounds, 1, m.cs.tile_off, s);
    prefill::fp4::quantize(x, nullptr, m.cs.bounds, m.cs.tile_off, 1, m.cs.a4, m.cs.gs, cols, cols, T, s);
}

void chunk_ffn(Model& m, const Layer& L, int T, cudaStream_t s) {
    const Geometry& g = m.g;
    ChunkScratch& b = m.cs;
    const int H = g.n_embd;
    kernels::rms_norm_rows(b.x, L.ffn_norm, b.xn, T, H, g.eps, s);
    if (T >= m.o.fp4_rows && L.gate.type == 40 && L.up.type == 40 && L.down.type == 40) {   // a long chunk: W4A4
        fp4_quant(m, b.xn, H, T, s);
        gemm_fp4(m, L.gate, b.g, T, s);
        gemm_fp4(m, L.up, b.u, T, s);
        prefill::fp4::quantize_swiglu(b.g, b.u, L.gate.scale, L.up.scale, m.cs.bounds, m.cs.tile_off, 1, m.cs.a4, m.cs.gs,
                                      g.n_ff, g.n_ff, T, s);   // the tile plan is the gate/up input's
        gemm_fp4(m, L.down, b.tmp, T, s);
        kernels::add_scaled(b.x, b.tmp, L.down.scale, (int64_t) T * H, s);
        return;
    }
    quantize_mmq(b.xn, b.xq, L.gate.type, H, T, s);
    gemm(m, L.gate, b.xq, b.g, T, s);
    if (L.up.type != L.gate.type) quantize_mmq(b.xn, b.xq, L.up.type, H, T, s);
    gemm(m, L.up, b.xq, b.u, T, s);
    kernels::swiglu_scaled(b.g, b.u, L.gate.scale, L.up.scale, b.h, T, g.n_ff, g.n_ff, s);
    quantize_mmq(b.h, b.xq, L.down.type, g.n_ff, T, s);
    gemm(m, L.down, b.xq, b.tmp, T, s);
    kernels::add_scaled(b.x, b.tmp, L.down.scale, (int64_t) T * H, s);
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

void commit_window(Model& m, SessionState& st, int n) {
    const Geometry& g = m.g;
    const int C = g.conv_channels();
    if (g.gdn_layers() == 0 || n <= 0) return;
    cudaMemcpyAsync(st.n_keep, &n, sizeof(int32_t), cudaMemcpyHostToDevice, st.stream);
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
    kernels::gdn_commit_layers(a, st.stream, 64);
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
    kernels::iq_embed_rows(30, w.embd, (size_t) H * 2, b.tok, T, H, b.x, s);
    const BlockDrafter& bd = w.dflash;
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
            wquant(b.x, b.xq, fc.type, H, s);
            gemv(m, fc, b.xq, next_target == 0 ? st.dacc : b.tmp, s);
            if (next_target > 0) kernels::add_scaled(st.dacc, b.tmp, nullptr, (int64_t) T * H, s);
            ++next_target;
        }
    }
    kernels::rms_norm_rows(b.x, w.out_norm, b.xn, T, H, g.eps, s);
    if (st.hid) cudaMemcpyAsync(st.hid, b.xn, (size_t) T * H * sizeof(float), cudaMemcpyDeviceToDevice, s);   // the drafter's
    wquant(b.xn, b.xq, w.head.type, H, s);
    gemv(m, w.head, b.xq, st.logits, s);
    kernels::scale_values(st.logits, w.head.scale, (int64_t) T * g.n_vocab, s);
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
    if (st.wg[T]) {
        if (cudaGraphLaunch(st.wg[T], s) != cudaSuccess) return cuda_ok("dense window graph", err) && false;
        return cuda_ok("dense window", err);
    }
    window_body(m, st, T, pos);   // the first window of T rows runs as launched, then is captured for the next ones
    if (!cuda_ok("dense window", err)) return false;
    cudaGraph_t graph = nullptr;
    if (cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed) != cudaSuccess) return cuda_ok("dense window capture", err);
    window_body(m, st, T, pos);
    if (cudaStreamEndCapture(s, &graph) != cudaSuccess || cudaGraphInstantiate(&st.wg[T], graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        st.wg[T] = nullptr;
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
    kernels::rms_norm_rows(b.x, L.attn_norm, b.xn, rows, H, g.eps, s);
    wquant(b.xn, b.xq, L.q.type, H, s);
    gemv(m, L.q, b.xq, b.qfull, s);
    if (L.k.type != L.q.type) wquant(b.xn, b.xq, L.k.type, H, s);
    gemv(m, L.k, b.xq, b.k, s);
    gemv(m, L.v, b.xq, b.v, s);
    int row = 0;
    for (int k = 0; k < S; ++k) {
        SessionState& st = *segs[k].st;
        const int T = segs[k].T;
        const int32_t* p0 = batch_pos0(m) + k;
        kernels::attn_qk_prepare(b.qfull + (size_t) row * 2 * AQ, b.k + (size_t) row * KV, b.v + (size_t) row * KV, L.q_norm,
                                 L.k_norm, g.eps, g.rope, b.pos + (size_t) row * 3, T, g.n_head, g.n_kv,
                                 b.qo + (size_t) row * AQ, st.kc[(size_t) ai], st.vc[(size_t) ai], segs[k].pos, s, 256, true, 0, p0,
                                 st.kq[(size_t) ai]);
        kernels::gqa_decode_attn(b.qo + (size_t) row * AQ, st.kc[(size_t) ai], st.vc[(size_t) ai], g.n_head, g.n_kv, segs[k].pos, T,
                                 b.attn, m.o.max_context, b.ao + (size_t) row * AQ, s, p0, st.kq[(size_t) ai]);
        row += T;
    }
    kernels::attn_gate(b.ao, b.qfull, b.ag, rows, g.n_head, s);
    wquant(b.ag, b.yq, L.o.type, AQ, s);
    gemv(m, L.o, b.yq, b.tmp, s);
    kernels::add_scaled(b.x, b.tmp, L.o.scale, (int64_t) rows * H, s);
}

void batch_gdn(Model& m, const Layer& L, int gi, const BatchSeg* segs, int rows, int max_n, cudaStream_t s) {
    const Geometry& g = m.g;
    WindowScratch& b = m.ws;
    const int H = g.n_embd, C = g.conv_channels();
    kernels::rms_norm_rows(b.x, L.attn_norm, b.xn, rows, H, g.eps, s);
    wquant(b.xn, b.xq, L.qkv.type, H, s);
    gemv(m, L.qkv, b.xq, b.rq, s);
    if (L.z.type != L.qkv.type) wquant(b.xn, b.xq, L.z.type, H, s);
    gemv(m, L.z, b.xq, b.z, s);
    kernels::DecRecArgs a;
    a.qkv = b.rq;
    a.conv_w = L.conv;
    a.x = b.xn;
    a.w_alpha = L.w_ab;
    a.w_beta = L.w_ab + (size_t) g.gdn_v_heads * H;
    a.dt = L.dt;
    a.ssm_a = L.ssm_a;
    a.z = b.z;
    a.gamma = L.gamma;
    a.h = b.rh;
    a.gate = b.rgate;
    a.beta = b.rbeta;
    a.yq = b.yq;
    a.y = b.y;
    a.C = C;
    a.n_embd = H;
    a.h_k = g.gdn_k_heads;
    a.h_v = g.gdn_v_heads;
    a.eps = g.eps;
    a.early_trigger = false;
    a.gate_act = g.gdn_gate;
    kernels::dec_gdn_rec_seg(a, b.btab->seg, (size_t) gi * segs[0].st->gdn_floats, rows, max_n, s);
    kernels::gdn_rows_scatter(b.btab->dst, rows, b.rq, b.rh, b.rgate, b.rbeta, C, g.gdn_v_heads, gi, kWindow, s);
    wquant(b.y, b.yq, L.out.type, g.value_width(), s);
    gemv(m, L.out, b.yq, b.tmp, s);
    kernels::add_scaled(b.x, b.tmp, L.out.scale, (int64_t) rows * H, s);
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
    kernels::iq_embed_rows(30, w.embd, (size_t) H * 2, b.tok, rows, H, b.x, s);
    const BlockDrafter& bd = w.dflash;
    const bool block = m.o.drafter == Drafter::Block;
    size_t next_target = 0;
    for (int il = 0; il < g.n_layers; ++il) {
        const Layer& L = w.layers[(size_t) il];
        if (L.kind == Mixer::Gdn) batch_gdn(m, L, m.gdn_index[(size_t) il], segs, rows, max_n, s);
        else batch_attn(m, L, m.attn_index[(size_t) il], segs, S, rows, s);
        win_ffn(m, L, rows, s);
        if (block && next_target < bd.target_layers.size() && bd.target_layers[next_target] == il) {
            // the block drafter's features, packed in the head's rows until every session has its own
            const Linear& fc = bd.fc[next_target];
            wquant(b.x, b.xq, fc.type, H, s);
            gemv(m, fc, b.xq, next_target == 0 ? b.hl : b.tmp, s);
            if (next_target > 0) kernels::add_scaled(b.hl, b.tmp, nullptr, (int64_t) rows * H, s);
            ++next_target;
        }
    }
    int row = 0;
    if (block)
        for (int k = 0; k < S; row += segs[k++].T)
            cudaMemcpyAsync(segs[k].st->dacc, b.hl + (size_t) row * H, (size_t) segs[k].T * H * sizeof(float), cudaMemcpyDeviceToDevice, s);
    kernels::rms_norm_rows(b.x, w.out_norm, b.xn, rows, H, g.eps, s);
    row = 0;
    for (int k = 0; k < S; row += segs[k++].T)
        if (segs[k].st->hid)   // the drafter's
            cudaMemcpyAsync(segs[k].st->hid, b.xn + (size_t) row * H, (size_t) segs[k].T * H * sizeof(float), cudaMemcpyDeviceToDevice, s);
    wquant(b.xn, b.xq, w.head.type, H, s);
    gemv(m, w.head, b.xq, b.hl, s);
    kernels::scale_values(b.hl, w.head.scale, (int64_t) rows * g.n_vocab, s);
    row = 0;
    for (int k = 0; k < S; row += segs[k++].T)
        cudaMemcpyAsync(segs[k].st->logits, b.hl + (size_t) row * g.n_vocab, (size_t) segs[k].T * g.n_vocab * sizeof(float),
                        cudaMemcpyDeviceToDevice, s);
}

}  // namespace

bool forward_batch(Model& m, const BatchSeg* segs, int S, const int32_t* tokens, const int32_t* pos3, std::string& err) {
    const Geometry& g = m.g;
    WindowScratch& b = m.ws;
    const cudaStream_t s = segs[0].st->stream;
    const int C = g.conv_channels(), HV = g.gdn_v_heads;
    int rows = 0, max_n = 1;
    BatchTable tab{};
    Model::BatchKey key;
    for (int k = 0; k < S; ++k) {
        SessionState& st = *segs[k].st;
        const int T = segs[k].T;
        key.emplace_back(&st, T);
        tab.pos0[k] = (int32_t) segs[k].pos;
        for (int j = 0; j < T; ++j) {
            tab.seg[rows + j] = {st.gdn, rows, j, T, T, k, (int32_t) (segs[k].pos + j)};
            if (st.rec_qkv)
                tab.dst[rows + j] = {st.rec_qkv + (size_t) j * C, st.rec_h + (size_t) j * C, st.rec_gate + (size_t) j * HV,
                                     st.rec_beta + (size_t) j * HV};
        }
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
    if (rows != nullptr) {
        cudaMemcpyAsync(b.x, rows, (size_t) T * H * sizeof(float), cudaMemcpyHostToDevice, s);
    } else {
        cudaMemcpyAsync(b.tok, tokens, (size_t) T * sizeof(int32_t), cudaMemcpyHostToDevice, s);
        kernels::iq_embed_rows(30, w.embd, (size_t) H * 2, b.tok, T, H, b.x, s);
    }
    // the drafter's history needs the chunk's inputs: its next-token embeddings are the chunk's own rows 1..T-1
    const bool drafter = st.mtp_kc != nullptr;
    if (drafter) cudaMemcpyAsync(b.cat, b.x, (size_t) T * H * sizeof(float), cudaMemcpyDeviceToDevice, s);
    const BlockDrafter& bd = w.dflash;
    const bool block = m.o.drafter == Drafter::Block;
    size_t next_target = 0;
    for (int il = 0; il < g.n_layers; ++il) {
        const Layer& L = w.layers[(size_t) il];
        if (L.kind == Mixer::Gdn) {
            chunk_gdn(m, st, L, m.gdn_index[(size_t) il], T, s);
        } else {
            const int ai = m.attn_index[(size_t) il];
            chunk_attn(m, L, st.kc[(size_t) ai], st.vc[(size_t) ai], T, pos, s, st.kq[(size_t) ai]);
        }
        chunk_ffn(m, L, T, s);
        if (block && next_target < bd.target_layers.size() && bd.target_layers[next_target] == il) {
            const Linear& fc = bd.fc[next_target];
            quantize_mmq(b.x, b.xq, fc.type, H, T, s);
            gemm(m, fc, b.xq, next_target == 0 ? b.dacc : b.tmp, T, s);
            if (next_target > 0) kernels::add_scaled(b.dacc, b.tmp, nullptr, (int64_t) T * H, s);
            ++next_target;
        }
    }
    if (block) {   // the chunk's rows become the block drafter's context
        kernels::rms_norm_rows(b.dacc, bd.hidden_norm, b.xn, T, H, bd.eps, s);
        quantize_mmq(b.xn, b.xq, bd.layers[0].k.type, H, T, s);
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
    kernels::rms_norm_rows(b.x, w.out_norm, b.hn + H, T, H, g.eps, s);
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
    const BlockDrafter& bd = m.w.dflash;
    WindowScratch& b = m.ws;
    const cudaStream_t s = st.stream;
    const int H = m.g.n_embd;
    const int32_t bounds[2] = {0, n};
    cudaMemcpyAsync(b.bounds, bounds, sizeof bounds, cudaMemcpyHostToDevice, s);
    std::vector<int32_t> p3;
    text_positions(p3, n, pos, 0);
    cudaMemcpyAsync(b.bpos, p3.data(), p3.size() * sizeof(int32_t), cudaMemcpyHostToDevice, s);
    kernels::rms_norm_rows(st.dacc, bd.hidden_norm, b.xn, n, H, bd.eps, s);
    wquant(b.xn, b.xq, bd.layers[0].k.type, H, s);
    const kernels::RopeSpec rope = block_rope(bd);
    for (size_t l = 0; l < bd.layers.size(); ++l) {
        const BlockDrafter::Layer& D = bd.layers[l];
        gemv(m, D.k, b.xq, b.bk, s);
        gemv(m, D.v, b.xq, b.bv, s);
        kernels::attn_qk_prepare(nullptr, b.bk, b.bv, nullptr, D.k_norm, bd.eps, rope, b.bpos, n, 0, bd.n_kv, nullptr,
                                 st.dk[l], st.dv[l], pos, s, bd.head_dim, false, st.dctx_cap);
    }
    return cuda_ok("block drafter context", err);
}

bool block_draft(Model& m, SessionState& st, int32_t anchor, int64_t pos, std::string& err) {
    const BlockDrafter& bd = m.w.dflash;
    const Geometry& g = m.g;
    WindowScratch& b = m.ws;
    const cudaStream_t s = st.stream;
    const int H = g.n_embd, B = bd.block, AQ = bd.n_head * bd.head_dim;
    const int groups = H / bd.conv_group;
    // the block: the anchor, then mask tokens, at positions pos ..
    int32_t tok[kWindow];
    tok[0] = anchor;
    for (int i = 1; i < B; ++i) tok[i] = bd.mask_id;
    const int32_t bounds[2] = {0, B};
    cudaMemcpyAsync(b.bounds, bounds, sizeof bounds, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(b.tok, tok, (size_t) B * sizeof(int32_t), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(st.win_tok, tok, sizeof(int32_t), cudaMemcpyHostToDevice, s);
    std::vector<int32_t> p3;
    text_positions(p3, B, pos, 0);
    cudaMemcpyAsync(b.bpos, p3.data(), p3.size() * sizeof(int32_t), cudaMemcpyHostToDevice, s);
    kernels::iq_embed_rows(30, m.w.embd, (size_t) H * 2, b.tok, B, H, b.x, s);
    const kernels::RopeSpec rope = block_rope(bd);
    for (size_t l = 0; l < bd.layers.size(); ++l) {
        const BlockDrafter::Layer& D = bd.layers[l];
        // attention, its input and output through the dynamic convolution
        kernels::rms_norm_rows(b.x, D.attn_norm, b.xn, B, H, bd.eps, s);
        wquant(b.xn, b.xq, D.conv_a.type, H, s);
        gemv(m, D.conv_a, b.xq, b.dyn, s);
        kernels::dyn_conv2(b.xn, b.dyn, D.conv_a.rows, D.base_a, b.bh, B, H, bd.conv_group, s);
        wquant(b.bh, b.xq, D.q.type, H, s);
        gemv(m, D.q, b.xq, b.bq, s);
        gemv(m, D.k, b.xq, b.bk, s);
        gemv(m, D.v, b.xq, b.bv, s);
        kernels::attn_qk_prepare(b.bq, b.bk, b.bv, D.q_norm, D.k_norm, bd.eps, rope, b.bpos, B, bd.n_head, bd.n_kv, b.bqo,
                                 st.dk[l], st.dv[l], pos, s, bd.head_dim, false, st.dctx_cap);
        const int64_t window = std::min<int64_t>(bd.window, pos - st.dctx_from + 1);   // context below dctx_from is gone
        kernels::gqa_window_attn(b.bqo, st.dk[l], st.dv[l], bd.n_head, bd.n_kv, bd.head_dim, pos, B, pos + B, window,
                                 st.dctx_cap, b.battn, st.dctx_cap, b.bao, s);
        wquant(b.bao, b.yq, D.o.type, AQ, s);
        gemv(m, D.o, b.yq, b.tmp, s);
        kernels::dyn_conv2(b.tmp, b.dyn + 2 * groups, D.conv_a.rows, D.base_a + 2 * H, b.bh, B, H, bd.conv_group, s);
        kernels::add_scaled(b.x, b.bh, nullptr, (int64_t) B * H, s);
        // the FFN, likewise
        kernels::rms_norm_rows(b.x, D.ffn_norm, b.xn, B, H, bd.eps, s);
        wquant(b.xn, b.xq, D.conv_f.type, H, s);
        gemv(m, D.conv_f, b.xq, b.dyn, s);
        kernels::dyn_conv2(b.xn, b.dyn, D.conv_f.rows, D.base_f, b.bh, B, H, bd.conv_group, s);
        wquant(b.bh, b.xq, D.gate.type, H, s);
        gemv(m, D.gate, b.xq, b.g, s);
        gemv(m, D.up, b.xq, b.u, s);
        kernels::swiglu_scaled(b.g, b.u, nullptr, nullptr, b.h, B, D.gate.rows, D.gate.rows, s);
        wquant(b.h, b.yq, D.down.type, D.gate.rows, s);
        gemv(m, D.down, b.yq, b.tmp, s);
        kernels::dyn_conv2(b.tmp, b.dyn + 2 * groups, D.conv_f.rows, D.base_f + 2 * H, b.bh, B, H, bd.conv_group, s);
        kernels::add_scaled(b.x, b.bh, nullptr, (int64_t) B * H, s);
    }
    // slots 1 .. B-1: the head's candidates, then one path through them
    kernels::rms_norm_rows(b.x, bd.norm, b.xn, B, H, bd.eps, s);
    wquant(b.xn, b.xq, m.w.head.type, H, s);
    gemv(m, m.w.head, b.xq, b.hl, s);
    kernels::scale_values(b.hl, m.w.head.scale, (int64_t) B * g.n_vocab, s);
    kernels::row_logprobs(b.hl + g.n_vocab, B - 1, g.n_vocab, b.tok + 1, bd.top_k, b.blp, b.lp_scratch, s);
    kernels::bf16_rows_dot(b.xn + H, bd.sel_h, b.bsel, B - 1, bd.rank, H, s);
    kernels::candidate_path(b.blp, kernels::kLogprobStride, bd.top_k, b.bsel, bd.pred, bd.succ, bd.rank, st.win_tok, B - 1,
                            st.dids, st.dconf, s);
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
    quantize_mmq(cat, b.xq, d.fc.type, 2 * H, n, s);
    gemm(m, d.fc, b.xq, b.x, n, s);
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
    wquant(b.cat, b.xq, d.fc.type, 2 * H, s);
    gemv(m, d.fc, b.xq, b.x, s);
    win_attn(m, d.layer, st.mtp_kc, st.mtp_vc, n, c0, s);
    if (out) {
        win_ffn(m, d.layer, n, s);
        kernels::rms_norm_rows(b.x, d.norm, b.xn, n, H, g.eps, s);
        cudaMemcpyAsync(out, b.xn, (size_t) n * H * sizeof(float), cudaMemcpyDeviceToDevice, s);
        // the head reads the last row: bounds {n - 1, n}
        const int32_t last[2] = {n - 1, n};
        cudaMemcpyAsync(b.bounds, last, sizeof last, cudaMemcpyHostToDevice, s);
        wquant(b.xn, b.xq, m.w.head.type, H, s);
        gemv(m, m.w.head, b.xq, b.hl, s);
        cudaMemcpyAsync(st.dlogits, b.hl + (size_t) (n - 1) * g.n_vocab, (size_t) g.n_vocab * sizeof(float),
                        cudaMemcpyDeviceToDevice, s);
        kernels::scale_values(st.dlogits, m.w.head.scale, g.n_vocab, s);
        kernels::row_logprobs(st.dlogits, 1, g.n_vocab, tok + n - 1, 1, lp_row, b.lp_scratch, s);
    }
    return cuda_ok("drafter", err);
}

}  // namespace basalt::dense
