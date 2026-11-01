#include "state.hpp"
#include "layout.hpp"
#include "basalt/prefill/prefill.hpp"
#include "basalt/core/mtp.hpp"
#include "basalt/core/layout.hpp"
#include "basalt/kernels/qsa.hpp"
#include "basalt/kernels/kv_q4.hpp"
#include "basalt/core/layer.hpp"
#include "basalt/core/native_head.hpp"
#include "basalt/kernels/verify_kernels.hpp"
#include "basalt/prefill/gemm.hpp"
#include "basalt/prefill/moe_fused.hpp"
#include "basalt/prefill/moe_mmq.hpp"
#include "basalt/prefill/moe_fp4.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/prefill/kernels.hpp"
#include <algorithm>
#include <string>
#include <vector>
#include <memory>

namespace basalt::prefill {
using namespace detail;

bool Prefill::draft_kv(core::MtpDrafter& mtp, const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0,
                       std::string& err) try {
    Impl& m = *impl_;
    // A ring (KV streaming: the drafter's window, page p in slot p % n_slots over a host copy) takes the same appends
    // with its own page table and host copy, as a streamed main layer does; the cells written are those the window can
    // still reach (r0 below), which the ring holds, so no two of them share a slot..
    core::QsaState& st = mtp.kv_state_rw();
    if (n <= 0 || m.g == nullptr || m.region == nullptr || (st.kv_mode != 0 && st.kv_mode != 2) ||
        st.kv_hybrid || mtp.device() != m.device)
        return false;
    const auto t0 = Clock::now();
    const core::ModelGeometry& g = *m.g;
    const int64_t Nn = g.n_embd, HCN = g.hc * g.n_embd, KV = g.n_head_kv * g.head_dim;
    constexpr int kQ8_0 = 8;   // GGML_TYPE_Q8_0
    const float* w_ne = mtp.tensor_f32("pre_fc_norm_embedding.weight");
    const void* w_fe = mtp.tensor_q8("fc_embedding.weight");
    const float* w_nh = mtp.tensor_f32("pre_fc_norm_hidden.weight");
    const void* w_fh = mtp.tensor_q8("fc_hidden.weight");
    const float* w_hn = mtp.tensor_f32("attn_hyper_connection.hc_norm.weight");
    const uint16_t* w_dn = mtp.tensor_bf16("attn_hyper_connection.input_mix_weight_down.weight");
    const uint16_t* w_up = mtp.tensor_bf16("attn_hyper_connection.input_mix_weight_up.weight");
    const void* w_k = mtp.tensor_q8("self_attn.k_proj.weight");
    const void* w_v = mtp.tensor_q8("self_attn.v_proj.weight");
    const float* w_kn = mtp.tensor_f32("self_attn.k_norm.weight");
    if (!w_ne || !w_fe || !w_nh || !w_fh || !w_hn || !w_dn || !w_up || !w_k || !w_v || !w_kn) return false;
    const core::NativeEmbed* nemb = core::native_embed();
    const core::WeightRef* wemb = nemb ? nullptr : m.wt->find("token_embd.weight");
    if (!nemb && (wemb == nullptr || wemb->codebook_iq4nl || wemb->ne0 != g.n_embd || wemb->group_elems <= 0 ||
                  (wemb->code_bits != 2 && wemb->code_bits != 4 && wemb->code_bits != 8)))
        return false;
    // the cells the drafter's window can still reach
    const int64_t r0 = std::max<int64_t>(0, mtp.first_needed() - cell0);
    if (r0 >= n) return true;
    // per row: emb/e2 (N), en16 (N half), hn/h2/Rm/gated (HCN), hn16/xn16 (HCN half), lo (LR) + lo16, grs, mixed (N) +
    // mixed_h, K and V (KV each), the token id
    // the drafter's Q8_0 matrices through Q8_1 x Q8_0 MMQ - its own pass's integer dot products (mmvq), so its K/V
    // stay close to what the drafter computes itself
    const bool q8 = mmq::built() && mmq::fits(kQ8_0, Nn) && mmq::fits(kQ8_0, KV);
    const uint64_t per_row = 4 * (2 * Nn + 4 * HCN + LR + HC + Nn + 2 * KV + 1) + 2 * (Nn + 2 * HCN + LR + Nn) + 64 +
                             (q8 ? (uint64_t) mmq::q8_bytes(g.hc, Nn) + 4 * g.hc : 0);
    int64_t B = std::min<int64_t>(n - r0, (int64_t) (m.region_bytes / per_row) & ~(int64_t) 63);
    if (st.kv_mode == 2)   // a ring: one batch's cells must not share a slot (a batch can straddle one page more)
        B = std::min<int64_t>(B, ((st.n_slots - 1) * basalt::kernels::qsa_real_shapes().page_size) & ~(int64_t) 63);
    if (B < 64) return false;
    uint8_t* q = m.region;
    auto carve = [&](size_t bytes) { void* p = q; q += (bytes + 255) & ~(size_t) 255; return p; };
    float* emb = (float*) carve((size_t) B * Nn * 4);
    float* e2 = (float*) carve((size_t) B * Nn * 4);
    uint16_t* en16 = (uint16_t*) carve((size_t) B * Nn * 2);
    float* hn = (float*) carve((size_t) B * HCN * 4);
    float* h2 = (float*) carve((size_t) B * HCN * 4);
    float* Rm = (float*) carve((size_t) B * HCN * 4);
    float* gated = (float*) carve((size_t) B * HCN * 4);
    uint16_t* hn16 = (uint16_t*) carve((size_t) B * HCN * 2);
    uint16_t* xn16 = (uint16_t*) carve((size_t) B * HCN * 2);
    float* lo = (float*) carve((size_t) B * LR * 4);
    uint16_t* lo16 = (uint16_t*) carve((size_t) B * LR * 2);
    float* grs = (float*) carve((size_t) B * HC * 4);
    float* mixed = (float*) carve((size_t) B * Nn * 4);
    uint16_t* mixed_h = (uint16_t*) carve((size_t) B * Nn * 2);
    float* Kc = (float*) carve((size_t) B * KV * 4);
    float* Vc = (float*) carve((size_t) B * KV * 4);
    int32_t* tok = (int32_t*) carve((size_t) B * 4);
    void* xq = q8 ? carve(mmq::q8_bytes(B * g.hc, Nn)) : nullptr;
    int32_t* ident = q8 ? (int32_t*) carve((size_t) B * g.hc * 4) : nullptr;
    int32_t* bnd = q8 ? (int32_t*) carve(16) : nullptr;
    if ((uint64_t) (q - m.region) > m.region_bytes) return false;
    if (!mtp.idle(err)) return false;   // the drafter's own stream (its graph uploads) before this writes its K/V
    basalt::kernels::QsaShapes s = basalt::kernels::qsa_real_shapes();
    s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    std::vector<int32_t> tk((size_t) B);
    if (q8) {
        if (!m.mmq_ctx) m.mmq_ctx = std::make_unique<mmq::Context>();
        mmq::iota(ident, B * g.hc, m.cs);
    }
    // y[rows, n_out] = x[rows, k] . w^T for a Q8_0 matrix: MMQ from the FP32 rows (bounds slot 0: rows, 1: rows*hc)
    auto proj = [&](const float* x, const uint16_t* x16, const void* w, float* y, int64_t rows, int64_t n_out,
                    int64_t k, int slot) {
        if (!q8) { m.gemm.native(x16, kQ8_0, w, y, rows, n_out, k); return; }
        mmq::quantize(x, nullptr, xq, kQ8_0, k, k, rows, m.cs);
        mmq::Product p;
        p.w = w; p.type = kQ8_0; p.w_rows = n_out; p.w_cols = k; p.expert_bytes = mmq::matrix_bytes(kQ8_0, n_out, k);
        p.n = 1; p.xq = xq; p.bounds = bnd + 2 * slot; p.ids = ident; p.total_rows = rows; p.max_rows = rows;
        p.dst = y; p.ld_dst = n_out;
        m.mmq_ctx->run(p, m.cs);
    };
    for (int64_t b0 = r0; b0 < n; b0 += B) {
        const int64_t nb = std::min(B, n - b0), c0 = cell0 + b0;
        for (int64_t i = 0; i < nb; ++i) tk[(size_t) i] = next_tokens[b0 + i];
        const int32_t bh[4] = {0, (int32_t) nb, 0, (int32_t) (nb * g.hc)};
        if (q8 && cudaMemcpyAsync(bnd, bh, sizeof bh, cudaMemcpyHostToDevice, m.cs) != cudaSuccess) {
            err = "prefill: the draft bounds' upload failed";
            return false;
        }
        if (cudaMemcpyAsync(tok, tk.data(), (size_t) nb * 4, cudaMemcpyHostToDevice, m.cs) != cudaSuccess) {
            err = "prefill: the draft tokens' upload failed";
            return false;
        }
        // the input branches: the next token's embedding, and this cell's final residual rows
        if (nemb) {
            nemb->gather_dev(tok, nb, emb, m.cs);
        } else {
            const auto* codes = (const uint8_t*) wemb->data;
            const auto* scales = (const float*) (codes + wemb->codes_bytes);
            const auto* offsets = wemb->has_offset ? (const float*) (codes + wemb->codes_bytes + wemb->scales_bytes)
                                                   : nullptr;
            basalt::kernels::embedding_gather_dev(codes, scales, offsets, tok, (int) nb, wemb->ne0, wemb->code_bits,
                                                  wemb->code_bias, wemb->group_elems,
                                                  (uint64_t) (wemb->ne0 / (8 / wemb->code_bits)),
                                                  (uint64_t) (wemb->ne0 / wemb->group_elems), emb, m.cs);
        }
        rms_rows(emb, w_ne, nb, Nn, Nn, EPS, m.cs);
        if (!q8) to_f16(emb, en16, nb * Nn, m.cs);
        proj(emb, en16, w_fe, e2, nb, Nn, Nn, 0);
        cudaMemcpyAsync(hn, R_rows + (size_t) b0 * HCN, (size_t) nb * HCN * 4, cudaMemcpyDefault, m.cs);   // (or pinned host rows)
        rms_rows(hn, w_nh, nb, HCN, HCN, EPS, m.cs);
        if (!q8) to_f16(hn, hn16, nb * HCN, m.cs);
        proj(hn, hn16, w_fh, h2, nb * g.hc, Nn, Nn, 1);   // every stream through fc_hidden
        basalt::kernels::add_streams_broadcast(h2, e2, Rm, Nn, (int) g.hc, (int) nb, m.cs);
        // the attention hyper-connection's read (its mixed input only: this pass writes nothing back)
        gr_norm_rs(Rm, w_hn, EPS, grs, xn16, nb, m.cs);
        m.gemm.bf16_w(xn16, w_dn, lo, nb, LR, HCN);
        gr_silu(lo, lo16, nb, m.cs);
        up_mix(m.gemm, w_up, lo16, gated, Rm, grs, w_hn, mixed, nullptr, mixed_h, nb, m.cs);
        // K and V into the drafter's cache, as the prompt path's QSA layers append theirs
        proj(mixed, mixed_h, w_k, Kc, nb, KV, Nn, 0);
        proj(mixed, mixed_h, w_v, Vc, nb, KV, Nn, 0);
        rms_rows(Kc, w_kn, nb * g.n_head_kv, g.head_dim, g.head_dim, EPS, m.cs);
        rope(Kc, nb, g.n_head_kv, g.head_dim, KV, c0, basalt::kernels::rope_scaling(), m.cs);
        if (st.kv_rot) {   // rotated as the drafter's own decode stores them (src/core/mtp/forward.cpp)
            basalt::kernels::fwht256_inplace_cuda(Kc, nb * g.n_head_kv, m.cs);
            basalt::kernels::fwht256_inplace_cuda(Vc, nb * g.n_head_kv, m.cs);
        }
        if (st.kv_q4) {
            basalt::kernels::kv_append_q4(st.k_q4, st.v_q4, st.page_table, c0, nb, Kc, Vc, s, m.cs, &st.host);
        } else {
            kv_append(Kc, Vc, nb, c0, st.page_table, s.page_size, st.kv_int8 ? nullptr : st.k_pool,
                      st.kv_int8 ? nullptr : st.v_pool, st.k_q, st.v_q, st.k_scale, st.v_scale, m.cs, &st.host);
        }
    }
    if (cudaStreamSynchronize(m.cs) != cudaSuccess) {
        err = std::string("prefill: the draft layer's K/V: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    mtp.wrote(cell0 + r0, cell0 + n);
    mtp.ms_prefill += ms_since(t0);
    return true;
} catch (const GemmPlanError& e) {
    if (impl_ && impl_->cs) (void) cudaStreamSynchronize(impl_->cs);
    err = e.what();
    return false;
}

}  // namespace basalt::prefill
