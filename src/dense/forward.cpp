// src/dense/forward.cpp - the layer loop of a decode window and of a prompt chunk, and the drafter's passes.
#include "internal.hpp"

#include "basalt/kernels/gdn_prompt.hpp"
#include "basalt/kernels/gqa_attn.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/logprobs.hpp"
#include "basalt/kernels/native_mmvq.hpp"

#include <algorithm>
#include <vector>

namespace basalt::dense {
namespace {

using kernels::iq_mmvq;
using kernels::native_quantize_q8_1;

bool cuda_ok(const char* what, std::string& err) {
    const cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) return true;
    err = std::string(what) + ": " + cudaGetErrorString(e);
    return false;
}

// y [T][rows] = W . x for q8_1 activations xq (T columns of W.cols values)
void gemv(const Linear& l, const void* xq, float* y, int T, cudaStream_t s) {
    iq_mmvq(l.type, l.w, xq, y, (int) l.cols, (int) l.rows, T, s);
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

void win_attn(Model& m, const Layer& L, uint16_t* kc, uint16_t* vc, int T, int64_t pos, cudaStream_t s) {
    const Geometry& g = m.g;
    WindowScratch& b = m.ws;
    const int H = g.n_embd, AQ = g.n_head * g.head_dim;
    kernels::rms_norm_rows(b.x, L.attn_norm, b.xn, T, H, g.eps, s);
    native_quantize_q8_1(b.xn, b.xq, H, T, s);
    gemv(L.q, b.xq, b.qfull, T, s);
    gemv(L.k, b.xq, b.k, T, s);
    gemv(L.v, b.xq, b.v, T, s);
    kernels::attn_qk_prepare(b.qfull, b.k, b.v, L.q_norm, L.k_norm, g.eps, g.rope, b.pos, T, g.n_head, g.n_kv, b.qo, kc, vc,
                             pos, s);
    kernels::gqa_decode_attn(b.qo, kc, vc, g.n_head, g.n_kv, pos, T, b.attn, m.o.max_context, b.ao, s);
    kernels::attn_gate(b.ao, b.qfull, b.ag, T, g.n_head, s);
    native_quantize_q8_1(b.ag, b.yq, AQ, T, s);
    gemv(L.o, b.yq, b.tmp, T, s);
    kernels::add_scaled(b.x, b.tmp, L.o.scale, (int64_t) T * H, s);
}

void win_gdn(Model& m, SessionState& st, const Layer& L, int gi, int T, cudaStream_t s) {
    const Geometry& g = m.g;
    WindowScratch& b = m.ws;
    const int H = g.n_embd, C = g.conv_channels();
    kernels::rms_norm_rows(b.x, L.attn_norm, b.xn, T, H, g.eps, s);
    native_quantize_q8_1(b.xn, b.xq, H, T, s);
    float* qkv = st.rec_qkv + (size_t) gi * kWindow * C;
    gemv(L.qkv, b.xq, qkv, T, s);
    gemv(L.z, b.xq, b.z, T, s);
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
    a.C = C;
    a.n_embd = H;
    a.h_k = g.gdn_k_heads;
    a.h_v = g.gdn_v_heads;
    a.eps = g.eps;
    a.early_trigger = false;
    a.gate_act = g.gdn_gate;
    kernels::dec_gdn_rec(a, T, s);
    gemv(L.out, b.yq, b.tmp, T, s);
    kernels::add_scaled(b.x, b.tmp, L.out.scale, (int64_t) T * H, s);
}

void win_ffn(Model& m, const Layer& L, int T, cudaStream_t s) {
    const Geometry& g = m.g;
    WindowScratch& b = m.ws;
    const int H = g.n_embd;
    kernels::rms_norm_rows(b.x, L.ffn_norm, b.xn, T, H, g.eps, s);
    native_quantize_q8_1(b.xn, b.xq, H, T, s);
    gemv(L.gate, b.xq, b.g, T, s);
    gemv(L.up, b.xq, b.u, T, s);
    kernels::swiglu_scaled(b.g, b.u, L.gate.scale, L.up.scale, b.h, T, g.n_ff, g.n_ff, s);
    native_quantize_q8_1(b.h, b.yq, g.n_ff, T, s);
    gemv(L.down, b.yq, b.tmp, T, s);
    kernels::add_scaled(b.x, b.tmp, L.down.scale, (int64_t) T * H, s);
}

// ---- prompt chunk layers: the residual rows are m.cs.x

void chunk_attn(Model& m, const Layer& L, uint16_t* kc, uint16_t* vc, int T, int64_t pos, cudaStream_t s) {
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
                             pos, s);
    kernels::gqa_prompt_attn(b.qo, kc, vc, g.n_head, g.n_kv, pos, T, b.ao, s);
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

void chunk_ffn(Model& m, const Layer& L, int T, cudaStream_t s) {
    const Geometry& g = m.g;
    ChunkScratch& b = m.cs;
    const int H = g.n_embd;
    kernels::rms_norm_rows(b.x, L.ffn_norm, b.xn, T, H, g.eps, s);
    quantize_mmq(b.xn, b.xq, L.gate.type, H, T, s);
    gemm(m, L.gate, b.xq, b.g, T, s);
    if (L.up.type != L.gate.type) quantize_mmq(b.xn, b.xq, L.up.type, H, T, s);
    gemm(m, L.up, b.xq, b.u, T, s);
    kernels::swiglu_scaled(b.g, b.u, L.gate.scale, L.up.scale, b.h, T, g.n_ff, g.n_ff, s);
    quantize_mmq(b.h, b.xq, L.down.type, g.n_ff, T, s);
    gemm(m, L.down, b.xq, b.tmp, T, s);
    kernels::add_scaled(b.x, b.tmp, L.down.scale, (int64_t) T * H, s);
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

bool forward_window(Model& m, SessionState& st, const int32_t* tokens, int T, int64_t pos, const int32_t* pos3,
                    std::string& err) {
    const Geometry& g = m.g;
    const Weights& w = m.w;
    WindowScratch& b = m.ws;
    const cudaStream_t s = st.stream;
    const int H = g.n_embd;
    cudaMemcpyAsync(b.tok, tokens, (size_t) T * sizeof(int32_t), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(b.pos, pos3, (size_t) T * 3 * sizeof(int32_t), cudaMemcpyHostToDevice, s);
    kernels::iq_embed_rows(30, w.embd, (size_t) H * 2, b.tok, T, H, b.x, s);
    for (int il = 0; il < g.n_layers; ++il) {
        const Layer& L = w.layers[(size_t) il];
        if (L.kind == Mixer::Gdn) {
            win_gdn(m, st, L, m.gdn_index[(size_t) il], T, s);
        } else {
            const int ai = m.attn_index[(size_t) il];
            win_attn(m, L, st.kc[(size_t) ai], st.vc[(size_t) ai], T, pos, s);
        }
        win_ffn(m, L, T, s);
    }
    float* normed = st.hid ? st.hid : b.xn;   // the drafter reads the normed rows
    kernels::rms_norm_rows(b.x, w.out_norm, normed, T, H, g.eps, s);
    native_quantize_q8_1(normed, b.xq, H, T, s);
    gemv(w.head, b.xq, st.logits, T, s);
    kernels::scale_values(st.logits, w.head.scale, (int64_t) T * g.n_vocab, s);
    return cuda_ok("dense window", err);
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
    for (int il = 0; il < g.n_layers; ++il) {
        const Layer& L = w.layers[(size_t) il];
        if (L.kind == Mixer::Gdn) {
            chunk_gdn(m, st, L, m.gdn_index[(size_t) il], T, s);
        } else {
            const int ai = m.attn_index[(size_t) il];
            chunk_attn(m, L, st.kc[(size_t) ai], st.vc[(size_t) ai], T, pos, s);
        }
        chunk_ffn(m, L, T, s);
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
    cudaMemcpyAsync(b.pos, p3.data(), p3.size() * sizeof(int32_t), cudaMemcpyHostToDevice, s);
    kernels::iq_embed_rows(30, m.w.embd, (size_t) H * 2, tok, n, H, b.tmp, s);
    mtp_input(m, b.tmp, hid, b.cat, n, s);
    native_quantize_q8_1(b.cat, b.xq, 2 * H, n, s);
    gemv(d.fc, b.xq, b.x, n, s);
    win_attn(m, d.layer, st.mtp_kc, st.mtp_vc, n, c0, s);
    if (out) {
        win_ffn(m, d.layer, n, s);
        kernels::rms_norm_rows(b.x, d.norm, out, n, H, g.eps, s);
        const float* last = out + (size_t) (n - 1) * H;
        native_quantize_q8_1(last, b.xq, H, 1, s);
        gemv(m.w.head, b.xq, st.dlogits, 1, s);
        kernels::scale_values(st.dlogits, m.w.head.scale, g.n_vocab, s);
        kernels::row_logprobs(st.dlogits, 1, g.n_vocab, tok + n - 1, 1, lp_row, b.lp_scratch, s);
    }
    return cuda_ok("drafter", err);
}

}  // namespace basalt::dense
