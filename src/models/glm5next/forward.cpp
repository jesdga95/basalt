// src/models/glm5next/forward.cpp - see forward.hpp. The layer composition follows the sibling DeepSeek 4.1 session,
// which runs the same shared mHC kernels: a sublayer's mixes run beside its collapse, the attention sublayer's `pre`
// collapses the streams for the FFN, and the previous layer's FFN `pre` feeds this layer's attention.
//
// One product helper (`proj`) sits under everything: y[t][n] = x[t] . w[n], over fp32 rows or - when the pack stores
// that weight packed - through the engine's mmvq kernels with the activations quantized to q8_1. Two weights the
// kernels want contiguous are packed once at init - the KDA front's q|k|v, and the three conv tap streams - and those
// two only come out of an fp32 pack.
//
// The DSA indexer scores POOLS, so the pool still being filled lives in the session's pending rows: kpool-1 of them,
// because the kpool-th token of a pool is the one being computed. See dsa_cells for the tail cells, the one place this
// pass reads more into the reference than the kernels prove.
#include "models/glm5next/forward.hpp"

#include "basalt/artifact/dequant.hpp"        // the host dequantizers for an embedding row the pack stores packed
#include "basalt/artifact/gguf_reader.hpp"  // ggml_type_name, for the dense types a pack carries
#include "basalt/core/expert_tier.hpp"
#include "basalt/kernels/dense_ops.hpp"
#include "basalt/kernels/elementwise.hpp"   // add_inplace, the drafter block's residual adds
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/mhc.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/row_norm.hpp"
#include "basalt/kernels/router.hpp"
#include "basalt/platform/file.hpp"  // pread_all, for one embedding row
#include "basalt/prefill/moe_mmq.hpp"  // the wide-window product against a packed weight

#include "models/glm5next/kernels/cell_expand.hpp"
#include "models/glm5next/kernels/expert_sum.hpp"
#include "models/glm5next/kernels/head_transpose.hpp"
#include "models/glm5next/kernels/idx_score.hpp"
#include "models/glm5next/kernels/idx_select.hpp"
#include "models/glm5next/kernels/kda.hpp"
#include "models/glm5next/kernels/kda_front.hpp"
#include "models/glm5next/kernels/kda_norm.hpp"
#include "models/glm5next/kernels/kpool.hpp"
#include "models/glm5next/kernels/mla_sparse.hpp"
#include "models/glm5next/model.hpp"
#include "models/glm5next/session.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace basalt::models::glm5next {

namespace {

constexpr int kF32 = 0, kQ8_0 = 8, kQ4_K = 12, kQ5_K = 13, kQ6_K = 14;  // ggml's type ids
constexpr int kHcCoef = kernels::kHcCoef;   // 24: pre(4) + post(4) + comb(16)
constexpr int kHcMult = kernels::kHcMult;   // the stream count
constexpr int32_t kNoCell = INT32_MIN;      // cell_expand's sentinel for a missing cell

bool ck(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string("glm5next forward: ") + what + ": " + cudaGetErrorString(e);
    return false;
}

}  // namespace

/// The per-window scratch, sized for a window of at most `cap` tokens. Every pointer is a device block.
struct GlmForward::Scratch {
    float *streams[2] = {};      // the mHC streams [T][4][E], alternating between the two hc_posts' target
    float *cur = nullptr;        // the sublayer's collapsed streams [T][E]
    float *normed = nullptr;     // ... and its RMS norm
    float *coef_a = nullptr;     // the attention sublayer's mixes [T][24]
    float *coef_f = nullptr;     // the FFN sublayer's mixes [T][24]
    float *hc_ws = nullptr;      // hc_mixes' workspace
    float *hc_head = nullptr;    // [T][4] = 1/4 each: the head's collapse is the unweighted mean of the streams
    float *hsum = nullptr;       // the head's collapsed streams [T][E]

    float *qkv = nullptr;        // [3][T][D] the KDA front's x: one [T][D] slab per stream (q, k, v)
    float *a_mid = nullptr;      // [T][head_dim] and [T][head_dim]: the f and g paths' first projection
    float *z_mid = nullptr;
    float *a = nullptr;          // [T][D] the decay projection
    float *z = nullptr;          // [T][D] the out gate, raw (kda_norm's z)
    float *b_raw = nullptr;      // [T][n_head]
    float *front = nullptr;      // [5][D]: the recurrence's inputs for one token - qn, kn, vout, g, beta
    float *kda_o = nullptr;      // [T][D] the recurrence's read
    float *kda_n = nullptr;      // [T][D] after the gated norm
    float *conv_win = nullptr;   // [conv-1][3D] the conv window while the window runs

    float *q_lat = nullptr;      // [T][q_lora]
    float *q_norm = nullptr;     // ... after its norm: the query's and the indexer's source
    float *lat_norm = nullptr;   // [T][kv_lora] the compressed latent, normed, as the cache takes it
    float *q = nullptr;          // [T][n_head*qk]
    float *latent_row = nullptr;  // [T][kv_lora] this window's compressed latents
    float *ik = nullptr;         // [T][idx_dim] the indexer's keys
    float *ln_key = nullptr;     // [T][idx_dim] ... after their LayerNorm
    float *ig = nullptr;         // [T][idx_dim] the compressor gates
    float *iq = nullptr;         // [T][idx_heads*idx_dim]
    float *iw = nullptr;         // [T][idx_heads]
    float *q_abs = nullptr;      // [T][n_head][kv_lora] the query with the key projection absorbed
    float *attn_h = nullptr;     // [T][n_head][kv_lora] the sparse attention's per-head output
    float *head_t = nullptr;     // [n_head][T][max(qk,kv_lora)] a window's activations, heads to the front
    float *head_out = nullptr;   // [T][max(kv_lora, v_dim)] one head's product rows, before they join the token's row
    float *tmp = nullptr;        // [T][max(kv_lora, n_head*v_dim)] a second home for a norm or a product
    float *stage_k = nullptr;    // [pools this window completes][kpool][idx_dim], for kpool_compress
    float *stage_g = nullptr;
    float *mla_ws = nullptr;

    float *attn_out = nullptr;   // [T][E]
    float *ffn_out = nullptr;    // [T][E]
    float *mtp_e = nullptr;      // [T][E] the drafter's block: the embedding of the token after each row
    float *mtp_cat = nullptr;    // [T][2E] ... and the row eh_proj reads: that embedding beside the model's hidden
    float *g_d = nullptr;        // the dense MLP's gate, and the shared experts' down projection
    float *u_d = nullptr;
    float *h_d = nullptr;
    float *sh_g = nullptr;       // the shared experts' gate/up/activation [T][shared_ffn()*n_shared]
    float *sh_u = nullptr;
    float *sh_h = nullptr;
    float *rlogits = nullptr;    // [T][n_expert] the router's logits (its workspace)
    float *rweights = nullptr;   // [T][n_used]
    int32_t *rids = nullptr;     // [T][n_used]

    float *scores = nullptr;     // [T][n_pool]
    int32_t *sel_pools = nullptr;  // [T][n_top_pool] the pools the indexer chose
    int32_t *ids = nullptr;      // [T][n_sel] the cells mla_sparse attends
    int32_t *out_n = nullptr;    // [T]
    int32_t *pos_i32 = nullptr;  // [T]
    int32_t *cache_len = nullptr;  // [T]
    int32_t *pool_idxs = nullptr;  // [kpool][n_pool] each pool's member cells
    int32_t *tail_idxs = nullptr;  // [kpool-1][T]
};

/// The mmvq product takes one to eight columns; the q8_1 block holds that many rows plus slack the kernels may read
/// past the last block, so it is a constant of the pass rather than of the window.
constexpr int kMmvqCols = 8;
constexpr size_t kQ8Slack = 1 << 16;

GlmForward::GlmForward(GlmSession& session, std::string& err)
    : session_(session), model_(session.model()), geo_(session.model().geo()) {
    const int E = geo_.n_embd, D = geo_.n_head * geo_.kda_head_dim;
    if (E <= 0 || D <= 0 || geo_.n_vocab <= 0 || geo_.kpool <= 0 || geo_.idx_top_k <= 0 || geo_.conv < 2 ||
        geo_.kda_head_dim > 1024) {
        err = "glm5next forward: the pack's geometry is not usable";
        return;
    }
    // The sparse-MLA stage is written against a fixed contract (mla_sparse.hpp): a latent row is
    // mqa::kLatent = 512 floats wide, and the heads are processed in groups of eight. A launcher outside that
    // contract returns without launching, so the branch contributes exactly zero and reads as a small attention
    // rather than as a refusal: geometry the kernels cannot serve is refused here, at the load.
    if (geo_.kv_lora != 512 || geo_.n_head % 8) {
        err = "glm5next forward: the sparse-MLA kernels want kv_lora_rank == 512 and head_count % 8 == 0";
        return;
    }
    // Each of these is another launcher that returns without launching, so the branch would contribute exactly zero
    // and only show up as an attention that looks small: kda.cu caps the KDA key head width at its kMaxDk
    // shared-memory row, idx_select.cu caps the selected pools at its kIdxTopkMax, and head_transpose.cu moves
    // float4, so a per-head width it is given must be a multiple of 4.
    if (geo_.kda_head_dim > 256 || geo_.kda_head_dim % 4 || geo_.idx_dim % 4 || geo_.v_dim % 4 || geo_.kv_lora % 4 ||
        geo_.idx_top_k / geo_.kpool > 2048) {
        err = "glm5next forward: a geometry the kernels refuse - kda head_dim at most 256 and a multiple of 4, "
              "index_head_dim/value_length_mla/index_kpool-selected-pools all within the kernels' rows (top_k / "
              "kpool at most 2048)";
        return;
    }
    // The KDA front bakes the reference's bounded-decay constant into the kernel. A pack with another bound would
    // otherwise be decayed by a number its checkpoint never had, and every magnitude downstream stays plausible.
    if (geo_.gate_lower_bound != kGateLowerBound) {
        err = "glm5next forward: the pack's kda.gate_lower_bound is " + std::to_string(geo_.gate_lower_bound) +
              ", but the KDA kernel is built for " + std::to_string(kGateLowerBound);
        return;
    }
    if (E % 128 || geo_.n_expert % 32 || D % 16 || E % 16) {
        err = "glm5next forward: a geometry the kernels refuse - the router wants n_embd % 128 and n_expert % 32, the "
              "fp32 product wants every input width a multiple of 16";
        return;
    }
    // A dense tensor is either fp32, which the fp32 product reads, or a packed type this engine has an mmvq kernel
    // for, which `proj` reads against activations quantized to q8_1. Anything else is refused here: a weight read in
    // the wrong format is a wrong answer, not a failure. The byte count is checked against the same kernel's idea of
    // the shape, so a pack whose packed rows are laid out differently is refused too - and the widest input of the
    // lot sizes the q8_1 activation block grow() allocates.
    for (const auto& [name, t] : model_.dev_tensors()) {
        if (t.type == 0) continue;
        if (!kernels::native_mmvq_supported((int) t.type)) {
            err = "glm5next forward: dense tensor '" + name + "' is " + ggml_type_name(t.type) +
                  ", which this engine has no product kernel for";
            return;
        }
        if (t.shape.empty()) {
            err = "glm5next forward: dense tensor '" + name + "' carries no shape";
            return;
        }
        uint64_t out_rows = 1;
        for (size_t a = 1; a < t.shape.size(); ++a) out_rows *= t.shape[a];
        const size_t want = kernels::native_mmvq_weight_bytes((int) t.type, (int) t.shape[0], (int) out_rows);
        if (want != t.bytes) {
            err = "glm5next forward: dense tensor '" + name + "' is " + ggml_type_name(t.type) + " " +
                  std::to_string(t.shape[0]) + "x" + std::to_string(out_rows) + " in " + std::to_string(t.bytes) +
                  " bytes, the product kernel wants " + std::to_string(want);
            return;
        }
        if ((int) t.shape[0] > kqmax_) kqmax_ = (int) t.shape[0];
    }
    // A packed dense projection's activations: one q8_1 row per column of a decode window, wide enough for the
    // widest packed weight in the pack, plus slack the mmvq kernels may read past the last block - zeroed, so that
    // read multiplies nothing (deepseek41/dspark.cpp allows the same for its q8_1 rows). Like the expert buffers,
    // this is not the window's worth, so it is not what grow() allocates and prefillp::window_units bounds.
    if (err.empty() && kqmax_ > 0) {
        const size_t qbytes = kernels::native_q8_1_bytes(kqmax_, kMmvqCols) + kQ8Slack;
        if (!ck(cudaMalloc(&dq8_, qbytes), "quantized projection activations", err) ||
            !ck(cudaMemset(dq8_, 0, qbytes), "quantized projection activations", err)) {
            if (dq8_) cudaFree(dq8_);
            dq8_ = nullptr;
        }
    }
    // A window wider than those eight columns multiplies a packed weight through the engine's prefill MMQ path: the
    // weight stays in the pack's own bytes, the activations round to q8_1, and int8 tensor cores do the sum. Whether
    // this build has it is a fact about the build, so the context is made here and `proj` says which way a wide
    // product goes rather than discovering it in the middle of a prompt.
    if (err.empty() && kqmax_ > 0 && prefill::mmq::built()) {
        mmq_ = new prefill::mmq::Context;
        if (!mmq_) {
            err = "glm5next forward: out of memory for the wide-window product";
            return;
        }
    }
    // The routed experts' device buffers, one block of tokens' worth: the rows the card's own experts write, the q8_1
    // activations they read, the grouped kernels' scratch, and the two plans (the experts the tier homes, then the ones
    // its link is still copying in). A block is the tier's window, not the prompt chunk's, so these do not grow with
    // the window: they are not part of what grow() allocates and prefillp::window_units bounds.
    if (model_.tier()) {
        if (!ck(cudaMalloc(&x_q8_, kernels::native_q8_1_bytes(E, kExpertRows)), "expert activations", err) ||
            !ck(cudaMalloc(&ex_scr_, kernels::native_expert_scratch_bytes(kMaxSumRows, geo_.n_ff_exp)),
                "expert scratch", err) ||
            !ck(cudaMalloc(reinterpret_cast<void**>(&ex_rows_), (size_t) kMaxSumRows * E * sizeof(float)),
                "expert rows", err) ||
            !ck(cudaMalloc(reinterpret_cast<void**>(&plan_dev_[0]), sizeof(ExpertPlan)), "expert plan", err) ||
            !ck(cudaMalloc(reinterpret_cast<void**>(&plan_dev_[1]), sizeof(ExpertPlan)), "expert plan 2", err))
            return;
    }
    // Pack the KDA conv taps once. They arrive as three [conv,1,D] tensors (D contiguous) and kda_front wants
    // [3][conv][D] (D contiguous), so each tap stream is transposed. The projections themselves are not packed:
    // the release pack keeps them Q5_K, three 23 MB matrices a layer, which is a product per token, not a copy.
    int n_kda = 0;
    for (int l = 0; l < geo_.n_model_layers(); ++l) if (!geo_.is_mla(l)) ++n_kda;
    if (n_kda == 0) { ok_ = true; return; }
    if (!ck(cudaMalloc(reinterpret_cast<void**>(&conv_pack_),
                       (size_t) n_kda * 3 * geo_.conv * D * sizeof(float)), "packed KDA conv", err)) return;
    std::vector<float> host, trans;
    int kl = 0;
    // err doubles as the stop flag: the first failure ends the pass, and the pack is rejected as a whole rather than
    // left with some KDA layers packed and others not.
    for (int l = 0; l < geo_.n_model_layers() && err.empty(); ++l) {
        if (geo_.is_mla(l)) continue;
        const char* pn[3] = {"q", "k", "v"};
        for (int s = 0; s < 3 && err.empty(); ++s) {
            const DevTensor* c = model_.tensor(l, (std::string("ssm_conv1d_") + pn[s] + ".weight").c_str());
            if (!c || c->shape.size() < 3 || (int) c->shape[0] != geo_.conv || (int) c->shape[2] != D) {
                err = "glm5next forward: layer " + std::to_string(l) +
                      " does not carry the KDA conv taps the geometry describes";
                break;
            }
            if (c->type != 0) {
                err = "glm5next forward: layer " + std::to_string(l) + " carries a KDA conv tap of type " +
                      ggml_type_name(c->type) + ", and this front packs the taps out of fp32 rows";
                break;
            }
            host.resize((size_t) geo_.conv * D);
            if (!ck(cudaMemcpy(host.data(), c->ptr, host.size() * sizeof(float), cudaMemcpyDeviceToHost),
                    "KDA conv tap", err)) break;
            // The pack's dim 0 (the taps) is contiguous: element (tap, chan) sits at chan*conv + tap. kda_front wants
            // conv[s][tap][chan] with chan contiguous, so the two axes swap here, on the host, once per layer.
            // kda_front pairs the pack's last tap with the current token, the reference causal conv's convention. The
            // four weights say nothing about which end is which, so a run is the only arbitration there: pairing them
            // the other way round changes a real prompt's logits measurably (mean |diff| 1.23 against a logit rms of
            // 1.73, every argmax moving) and leaves them just as void of vocabulary structure either way, so the
            // direction is live in the pass and is not what is wrong.
            trans.resize((size_t) geo_.conv * D);
            for (int d = 0; d < D; ++d)
                for (int tap = 0; tap < geo_.conv; ++tap)
                    trans[(size_t) tap * D + d] = host[(size_t) d * geo_.conv + tap];
            if (!ck(cudaMemcpy(conv_pack_ + ((size_t) kl * 3 + s) * geo_.conv * D, trans.data(),
                              trans.size() * sizeof(float), cudaMemcpyHostToDevice), "packed KDA conv", err)) break;
        }
        if (!err.empty()) break;
        ++kl;
    }
    if (err.empty() && kl != n_kda) err = "glm5next forward: the pack has " + std::to_string(kl) +
                                          " KDA layers with the geometry's projections, its geometry describes " +
                                          std::to_string(n_kda);
    ok_ = err.empty();
    // The drafter runs when the pack carries the block, shares its indexer with the model's, and its experts have both
    // a grouped kernel and a tier of their own. Short of any of that every token stays a full forward, which is what
    // the line below says in so many words.
    if (ok_ && geo_.nextn > 0 && geo_.index_share_mtp && model_.has_draft_experts() && model_.draft_tier()) {
        if (ck(cudaMalloc(reinterpret_cast<void**>(&mtp_prev_h_), (size_t) kMtpKeep * E * sizeof(float)),
               "drafter hidden", err) &&
            ck(cudaMalloc(reinterpret_cast<void**>(&mtp_logits_), (size_t) geo_.n_vocab * sizeof(float)),
               "drafter logits", err))
            drafts_ = true;
    } else if (ok_ && geo_.nextn > 0 && model_.drafting_asked()) {
        // Only a start that asked for drafts has anything to be told about: without --spec the block's experts were never
        // asked for, and a full forward per token is the answer the start chose.
        basalt::log::warn("basalt glm5next",
                          "blk.%d is a next-n drafter the pass cannot run (its indexer is %s, its experts %s): every "
                          "token is a full forward",
                          mtp_layer(), geo_.index_share_mtp ? "shared with the model" : "its own, which this pass does "
                          "not run",
                          model_.has_draft_experts() ? "have no tier" : "have no grouped kernel for the card");
    }
    ok_ = ok_ && err.empty();
    if (!ok_) {
        cudaFree(conv_pack_);
        conv_pack_ = nullptr;
        cudaFree(mtp_prev_h_);
        cudaFree(mtp_logits_);
        mtp_prev_h_ = mtp_logits_ = nullptr;
        drafts_ = false;
        if (err.empty()) err = "glm5next forward: the pack has fewer KDA layers than its geometry describes";
    } else if (drafts_) {
        basalt::log::info("basalt glm5next",
                          "blk.%d runs as a drafter: the block reads the model's indexer cells and its own latents, "
                          "and drafts one token per step",
                          mtp_layer());
    }
}

GlmForward::~GlmForward() {
    free_scratch();
    cudaFree(big_val_);
    for (float* p : fp32_val_)
        if (p) cudaFree(p);
    if (conv_pack_) cudaFree(conv_pack_);
    if (x_q8_) cudaFree(x_q8_);
    if (dq8_) cudaFree(dq8_);
    if (mmq_xq_) cudaFree(mmq_xq_);
    if (mmq_ids_) cudaFree(mmq_ids_);
    if (mmq_bounds_) cudaFree(mmq_bounds_);
    if (row_rows_) cudaFree(row_rows_);
    if (mtp_prev_h_) cudaFree(mtp_prev_h_);
    if (mtp_prev_ids_) cudaFree(mtp_prev_ids_);
    if (mtp_logits_) cudaFree(mtp_logits_);
    for (void* p : mmq_pad_)
        if (p) cudaFree(p);
    delete mmq_;
    if (ex_scr_) cudaFree(ex_scr_);
    if (ex_rows_) cudaFree(ex_rows_);
    for (void* p : plan_dev_)
        if (p) cudaFree(p);
    delete s_;
}

void GlmForward::free_scratch() {
    for (void* b : scratch_blocks_) cudaFree(b);
    scratch_blocks_.clear();
    scratch_units_ = 0;
    cap_ = 0;
    delete s_;
    s_ = nullptr;
}

/// (Re)allocate the window scratch. Every block is one cudaMalloc in scratch_blocks_, so a resize frees the old set
/// exactly; the packed KDA weights are not in that list and outlive every resize.
bool GlmForward::grow(int max_tokens, std::string& err) {
    const int E = geo_.n_embd, nH = geo_.n_head, HD = geo_.kda_head_dim, D = nH * HD, KV = geo_.kv_lora;
    const int T = max_tokens, KP = geo_.kpool, IDD = geo_.idx_dim;
    const uint64_t n_pool_max = session_.caches().context() / (uint64_t) KP;
    const int n_top_max = std::max(1, std::min((int) n_pool_max, geo_.idx_top_k / KP));
    const int n_sel_max = cell_expand_n_sel(KP, n_top_max, true);
    const int sh = geo_.shared_ffn() * geo_.n_shared;
    const int wide = std::max(geo_.n_ff, geo_.n_ff_exp);
    const int wider = std::max(KV, nH * geo_.v_dim);

    free_scratch();
    s_ = new Scratch;
    Scratch* s = s_;
    auto alloc = [&](float** p, size_t n, const char* what) {
        if (!n) return true;
        if (!ck(cudaMalloc(reinterpret_cast<void**>(p), n * sizeof(float)), what, err)) return false;
        scratch_blocks_.push_back(*p);
        scratch_units_ += n;
        return true;
    };
    auto alloc_i = [&](int32_t** p, size_t n, const char* what) {
        if (!n) return true;
        if (!ck(cudaMalloc(reinterpret_cast<void**>(p), n * sizeof(int32_t)), what, err)) return false;
        scratch_blocks_.push_back(*p);
        scratch_units_ += n;
        return true;
    };
    const size_t TE = (size_t) T * E, TD = (size_t) T * D;
    bool good = true;
    good &= alloc(&s->streams[0], TE * kHcMult, "streams");
    good &= alloc(&s->streams[1], TE * kHcMult, "streams");
    good &= alloc(&s->cur, TE, "collapsed streams");
    good &= alloc(&s->normed, TE, "normed streams");
    good &= alloc(&s->coef_a, (size_t) T * kHcCoef, "mHC coefficients");
    good &= alloc(&s->coef_f, (size_t) T * kHcCoef, "mHC coefficients");
    good &= alloc(&s->hc_ws, kernels::hc_mixes_workspace_floats(T), "mHC workspace");
    good &= alloc(&s->hc_head, (size_t) T * kHcMult, "the head collapse");
    good &= alloc(&s->hsum, TE, "head collapse");
    good &= alloc(&s->qkv, (size_t) 3 * T * D, "KDA q|k|v");   // three D-wide slabs: the front strides by T*D, not T*E
    good &= alloc(&s->a_mid, (size_t) T * HD * 2, "KDA gate projections");
    s->z_mid = s->a_mid + (size_t) T * HD;
    good &= alloc(&s->a, TD, "KDA decay");
    good &= alloc(&s->z, TD, "KDA out gate");
    good &= alloc(&s->b_raw, (size_t) T * nH, "KDA beta");
    good &= alloc(&s->front, (size_t) D * 4 + nH, "KDA recurrence inputs");
    good &= alloc(&s->kda_o, TD, "KDA read");
    good &= alloc(&s->kda_n, TD, "KDA normed read");
    good &= alloc(&s->conv_win, (size_t) (geo_.conv - 1) * 3 * D, "KDA conv window");
    good &= alloc(&s->q_lat, (size_t) T * geo_.q_lora, "MLA query latent");
    good &= alloc(&s->q_norm, (size_t) T * geo_.q_lora, "MLA normed query latent");
    good &= alloc(&s->lat_norm, (size_t) T * KV, "MLA normed latent");
    good &= alloc(&s->q, (size_t) T * nH * geo_.qk_dim, "MLA query");
    good &= alloc(&s->latent_row, (size_t) T * KV, "MLA latent");
    good &= alloc(&s->ik, (size_t) T * IDD * 3, "indexer keys and gates");
    s->ln_key = s->ik + (size_t) T * IDD; s->ig = s->ik + (size_t) T * IDD * 2;
    good &= alloc(&s->iq, (size_t) T * geo_.idx_heads * IDD, "indexer query");
    good &= alloc(&s->iw, (size_t) T * geo_.idx_heads, "indexer query weights");
    good &= alloc(&s->q_abs, (size_t) T * nH * KV, "MLA absorbed query");
    good &= alloc(&s->attn_h, (size_t) T * nH * KV, "MLA attention");
    good &= alloc(&s->head_t, (size_t) T * nH * std::max(geo_.qk_dim, KV), "MLA head-major rows");
    head_wide_ = std::max(KV, geo_.v_dim);
    good &= alloc(&s->head_out, (size_t) T * head_wide_, "MLA per-head product rows");
    good &= alloc(&s->tmp, (size_t) T * wider, "scratch rows");
    good &= alloc(&s->stage_k, ((size_t) T / KP + 2) * KP * IDD, "pool staging");
    good &= alloc(&s->stage_g, ((size_t) T / KP + 2) * KP * IDD, "pool staging");
    good &= alloc(&s->mla_ws, mla_sparse_workspace_floats(T, nH, n_sel_max), "MLA workspace");
    good &= alloc(&s->attn_out, TE, "attention output");
    good &= alloc(&s->ffn_out, TE, "FFN output");
    if (drafts_) {
        good &= alloc(&s->mtp_e, TE, "drafter embeddings");
        good &= alloc(&s->mtp_cat, TE * 2, "drafter merge rows");
        // The cells of the window's last row outlive the window in `mtp_prev_ids_`, at the same row stride.
        if (mtp_sel_stride_ != n_sel_max) {
            cudaFree(mtp_prev_ids_);
            mtp_prev_ids_ = nullptr;
            mtp_sel_stride_ = 0;
            mtp_prev_pos_ = -1;
            mtp_keep_ = 0;
            if (ck(cudaMalloc(reinterpret_cast<void**>(&mtp_prev_ids_),
                              (size_t) kMtpKeep * n_sel_max * sizeof(int32_t)), "drafter cells", err))
                mtp_sel_stride_ = n_sel_max;
            else
                good = false;
        }
    }
    good &= alloc(&s->g_d, (size_t) T * wide, "MLP gate");
    good &= alloc(&s->u_d, (size_t) T * wide, "MLP up");
    good &= alloc(&s->h_d, (size_t) T * wide, "MLP activation");
    good &= alloc(&s->sh_g, (size_t) T * sh, "shared expert gate");
    good &= alloc(&s->sh_u, (size_t) T * sh, "shared expert up");
    good &= alloc(&s->sh_h, (size_t) T * sh, "shared expert activation");
    good &= alloc(&s->rlogits, (size_t) T * geo_.n_expert, "router logits");
    good &= alloc(&s->rweights, (size_t) T * geo_.n_used, "router weights");
    good &= alloc_i(&s->rids, (size_t) T * geo_.n_used, "expert ids");
    good &= alloc(&s->scores, (size_t) T * n_pool_max, "indexer scores");
    good &= alloc_i(&s->sel_pools, (size_t) T * n_top_max, "chosen pools");
    good &= alloc_i(&s->ids, (size_t) T * n_sel_max, "attended cells");
    good &= alloc_i(&s->out_n, (size_t) T, "chosen pool counts");
    good &= alloc_i(&s->pos_i32, (size_t) T, "positions");
    good &= alloc_i(&s->cache_len, (size_t) T, "cache lengths");
    good &= alloc_i(&s->pool_idxs, (size_t) KP * n_pool_max, "pool members");
    good &= alloc_i(&s->tail_idxs, (size_t) std::max(1, KP - 1) * T, "tail cells");
    if (!good) { free_scratch(); return false; }

    // The head takes the unweighted mean of the four streams: the pack carries no head mix of its own because there is
    // nothing to learn - the reference model's HyperHead is a mean over the streams, unlike deepseek41, whose head
    // collapses with learned pre weights (feeding it the last block's FFN pre instead summed the streams at ~2, not 1,
    // and mixed them by the FFN's idea of which stream matters).
    std::vector<float> h((size_t) T * kHcMult, 1.0f / kHcMult);
    if (!ck(cudaMemcpy(s->hc_head, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice), "the head collapse",
            err)) { free_scratch(); return false; }
    if (mmq_ && !grow_mmq(T, err)) { free_scratch(); return false; }
    cap_ = T;
    return true;
}

bool GlmForward::proj(const DevTensor* w, const float* x, float* y, int T, int N, int K, std::string& err) {
    if (!w || !w->ptr) {
        err = "glm5next forward: the pack is missing a projection the geometry describes";
        return false;
    }
    // GGUF shapes are innermost first and the pack's bytes match, so dim 0 (the product's input) is the contiguous
    // one: the tensor is [K] followed by the N output rows. A 3-D weight (MLA's per-head projections) is the same
    // flat [N][K] with its outer axes split, so only the first axis and the total say anything here.
    uint64_t out_rows = 1;
    for (size_t a = 1; a < w->shape.size(); ++a) out_rows *= w->shape[a];
    if (w->shape.empty() || w->shape[0] != (uint64_t) K || out_rows != (uint64_t) N) {
        err = "glm5next forward: '" + w->name + "' is " + std::to_string(w->shape.empty() ? 0 : w->shape[0]) + "x" +
              std::to_string(out_rows) + " (" + std::to_string(w->shape.size()) + "-D), the geometry wants " +
              std::to_string(N) + " rows over " + std::to_string(K) + " inputs";
        return false;
    }
    return proj_at(w, x, y, T, N, K, err);
}

/// The product itself, past any shape check, from a tensor the pass assembled rather than read from the pack.
bool GlmForward::proj_at(const DevTensor* w, const float* x, float* y, int T, int N, int K, std::string& err) {
    if (w->type == 0) return proj_rows((const float*) w->ptr, w->name.c_str(), x, y, T, N, K, err);
    // A packed weight: the mmvq kernels want this window's activations as one q8_1 row per token. The constructor
    // checked the type and that the pack's bytes are exactly what the kernel wants for [N][K], so the only thing left
    // to say here is that the block was never allocated - which only a caller that skipped grow() could arrange.
    if (!dq8_) {
        err = "glm5next forward: '" + w->name + "' is " + ggml_type_name(w->type) +
              " but no q8_1 block was allocated to quantize its activations into";
        return false;
    }
    // The mmvq kernels are the decode-shaped product: one to eight columns. A wider window goes to the prefill MMQ
    // path, which reads the same packed bytes at tensor-core speed; a build without that path says so by name rather
    // than multiplying the whole weight once per eight tokens.
    if (T > kMmvqCols) {
        if (!mmq_) {
            err = "glm5next forward: '" + w->name + "' is " + ggml_type_name(w->type) + " and the window carries " +
                  std::to_string(T) + " tokens; this build has no wide-window product for a packed weight, and the "
                  "narrow one runs one to " + std::to_string(kMmvqCols) + " at a time";
            return false;
        }
        return proj_mmq(*w, x, y, T, N, K, err);
    }
    kernels::native_quantize_q8_1(x, dq8_, K, T, session_.stream());
    kernels::native_mmvq((int) w->type, w->ptr, dq8_, y, K, N, T, session_.stream());
    return true;
}

/// A per-head projection: `w` is [K][N][H], dim 0 contiguous, so head h's matrix is the h-th block of N rows of K
/// inputs, and every head multiplies its own slice of `x`'s [T][H][K] row. The shared products read a token's row as
/// `K` values after the last, so a window gets one head-major copy of `x` first; a lone token is already in that
/// shape, and a window that fits the mmvq columns or not, packed or not, is `proj_at`'s business, not this one's.
bool GlmForward::proj_heads(const DevTensor* w, const float* x, float* y, int T, int H, int N, int K,
                            std::string& err) {
    if (!w || !w->ptr) {
        err = "glm5next forward: the pack is missing a projection the geometry describes";
        return false;
    }
    // A per-head weight comes as its three dimensions, or flattened to two: the pack lists the drafter's pool tensors
    // that way, and GGUF's order makes the two the same bytes - element (k, n, h) sits at k + K*(n + N*h) either way.
    const bool shaped = w->shape.size() == 3 && w->shape[0] == (uint64_t) K && w->shape[1] == (uint64_t) N &&
                        w->shape[2] == (uint64_t) H;
    const bool flat = w->shape.size() == 2 && w->shape[0] == (uint64_t) K && w->shape[1] == (uint64_t) N * (uint64_t) H;
    if (!shaped && !flat) {
        std::string got;
        for (uint64_t d : w->shape) got += (got.empty() ? "" : "x") + std::to_string(d);
        err = "glm5next forward: '" + w->name + "' is " + (got.empty() ? std::string("scalar") : got) +
              ", the geometry wants " + std::to_string(K) + "x" + std::to_string(N) + "x" + std::to_string(H) +
              " for its per-head product";
        return false;
    }
    if (K % 4 || N % 4) {
        err = "glm5next forward: '" + w->name + "' has a " + std::to_string(K) + "-by-" + std::to_string(N) +
              " per-head matrix, whose widths the per-head product does not take";
        return false;
    }
    if (T > 1 && N > head_wide_) {
        err = "glm5next forward: a per-head product of '" + w->name + "' wants " + std::to_string(N) +
              " outputs a head, wider than the " + std::to_string(head_wide_) + " the scratch stages for one";
        return false;
    }
    const size_t head_bytes = w->bytes / (size_t) H;
    if (!head_bytes || head_bytes * (size_t) H != w->bytes) {
        err = "glm5next forward: '" + w->name + "' is " + std::to_string(w->bytes) +
              " bytes, which does not divide into per-head matrices";
        return false;
    }
    DevTensor head;
    head.name = w->name;
    head.type = w->type;
    head.bytes = head_bytes;
    head.shape = {(uint64_t) K, (uint64_t) N};
    const uint8_t* wsrc = (const uint8_t*) w->ptr;
    if (T == 1) {
        for (int h = 0; h < H; ++h) {
            head.ptr = (void*) (wsrc + (size_t) h * head_bytes);
            if (!proj_at(&head, x + (size_t) h * K, y + (size_t) h * N, 1, N, K, err)) return false;
        }
        return true;
    }
    head_major(x, s_->head_t, T, H, K, session_.stream());
    if (!ck(cudaGetLastError(), "head-major activations", err)) return false;
    // A product writes its rows one after another, so one head's [T][N] block needs a home of its own; `y`'s rows are
    // token-major, with the heads inside them, so each block goes back into its head's slice of every token's row.
    for (int h = 0; h < H; ++h) {
        head.ptr = (void*) (wsrc + (size_t) h * head_bytes);
        if (!proj_at(&head, s_->head_t + (size_t) h * T * K, s_->head_out, T, N, K, err)) return false;
        head_store(s_->head_out, y, T, H, h, N, session_.stream());
    }
    return true;
}

bool GlmForward::proj_rows(const float* w, const char* what, const float* x, float* y, int T, int N, int K,
                           std::string& err) {
    if (K % 16) {
        err = "glm5next forward: '" + std::string(what) + "' has an input width of " + std::to_string(K) +
              ", which the fp32 product does not take";
        return false;
    }
    kernels::f32_rows_dot(x, w, y, T, N, K, session_.stream());
    return true;
}

bool GlmForward::grow_mmq(int rows, std::string& err) {
    if (rows <= mmq_cap_) return true;
    void* xq = nullptr;
    int32_t* ids = nullptr;
    if (!ck(cudaMalloc(&xq, prefill::mmq::q8_bytes(rows, kqmax_)), "wide-window activations", err) ||
        !ck(cudaMalloc(reinterpret_cast<void**>(&ids), (size_t) rows * sizeof(int32_t)), "wide-window row map", err)) {
        if (xq) cudaFree(xq);
        if (ids) cudaFree(ids);
        return false;
    }
    if (!mmq_bounds_ &&
        !ck(cudaMalloc(reinterpret_cast<void**>(&mmq_bounds_), 2 * sizeof(int32_t)), "wide-window bounds", err)) {
        cudaFree(xq);
        cudaFree(ids);
        return false;
    }
    if (mmq_xq_) cudaFree(mmq_xq_);
    if (mmq_ids_) cudaFree(mmq_ids_);
    mmq_xq_ = xq;
    mmq_ids_ = ids;
    mmq_cap_ = rows;
    mmq_ids_n_ = 0;
    return true;
}

const void* GlmForward::mmq_weight(const DevTensor& w, int K, std::string& err) {
    const size_t past = prefill::mmq::read_past_bytes(w.type, K);
    if (!past) return w.ptr;  // whole tiles: rows of 256 values or a multiple, nothing read past the last one
    // MMQ's tiles step along a weight's rows 256 values at a time, so the last row of a matrix whose rows are shorter
    // is read on to the next multiple. The activations out there are zeros, so a zeroed tail multiplies nothing - but
    // the pack's allocation is exactly the tensor, so such a weight gets a padded copy of its own bytes.
    for (size_t i = 0; i < mmq_src_.size(); ++i)
        if (mmq_src_[i] == w.ptr) return mmq_pad_[i];
    void* pad = nullptr;
    if (!ck(cudaMalloc(&pad, w.bytes + past), w.name.c_str(), err)) return nullptr;
    if (!ck(cudaMemcpy(pad, w.ptr, w.bytes, cudaMemcpyDeviceToDevice), w.name.c_str(), err) ||
        !ck(cudaMemset((uint8_t*) pad + w.bytes, 0, past), w.name.c_str(), err)) {
        cudaFree(pad);
        return nullptr;
    }
    mmq_src_.push_back(w.ptr);
    mmq_pad_.push_back(pad);
    memo_bytes_ += w.bytes + past;
    return pad;
}

bool GlmForward::proj_mmq(const DevTensor& w, const float* x, float* y, int T, int N, int K, std::string& err) {
    namespace mmq = prefill::mmq;
    if (!mmq::supported(w.type) || !mmq::fits(w.type, N)) {
        err = "glm5next forward: '" + w.name + "' is " + ggml_type_name(w.type) + " and the window carries " +
              std::to_string(T) + " tokens; MMQ has no product for that type at " + std::to_string(N) + " rows";
        return false;
    }
    if (!mmq_xq_ || T > mmq_cap_) {
        err = "glm5next forward: '" + w.name + "' wants the wide-window product and its buffers were never sized";
        return false;
    }
    const void* wdev = mmq_weight(w, K, err);
    if (!wdev) return false;
    const int64_t tile = ((int64_t) w.type << 32) | (uint32_t) N;  // a type's tiles are readied once per row count
    if (std::find(mmq_tiles_.begin(), mmq_tiles_.end(), tile) == mmq_tiles_.end()) {
        mmq::load(w.type, N);
        mmq_tiles_.push_back(tile);
    }
    if (mmq_ids_n_ != T) {
        mmq::iota(mmq_ids_, T, session_.stream());
        mmq_ids_n_ = T;
    }
    mmq::quantize(x, nullptr, mmq_xq_, w.type, K, K, T, session_.stream());
    host_bounds_ = {0, T};  // one matrix, so one expert's rows are all of them
    if (!ck(cudaMemcpyAsync(mmq_bounds_, host_bounds_.data(), host_bounds_.size() * sizeof(int32_t),
                            cudaMemcpyHostToDevice, session_.stream()), "wide-window bounds", err))
        return false;
    mmq::Product p;
    p.w = wdev;
    p.type = w.type;
    p.w_rows = N;
    p.w_cols = K;
    p.expert_bytes = mmq::matrix_bytes(w.type, N, K);
    p.n = 1;
    p.xq = mmq_xq_;
    p.bounds = mmq_bounds_;
    p.ids = mmq_ids_;
    p.total_rows = T;
    p.max_rows = T;
    p.dst = y;
    p.ld_dst = N;
    mmq_->run(p, session_.stream());
    return true;
}

/// fp32 rows this many values or more are what the memo below cannot afford: 16 MiB of floats, where the weights the
/// pass multiplies as rows that it is willing to keep - norms, biases, the router's gate, the mHC mixes - are all
/// under it, and one block's MLA key is two orders over.
constexpr uint64_t kFp32SlotValues = 4ull << 20;

const float* GlmForward::weight(int layer, const char* name, std::string& err) {
    const DevTensor* t = layer < 0 ? model_.tensor(name) : model_.tensor(layer, name);
    if (!t || !t->ptr) {
        err = "glm5next forward: layer " + std::to_string(layer) + " is missing '" + name +
              "' that the geometry describes";
        if (layer < 0) err = "glm5next forward: the pack is missing '" + std::string(name) + "'";
        return nullptr;
    }
    if (t->type == 0) return (const float*) t->ptr;
    // Not fp32. These are the weights the code multiplies as rows rather than through the packed product - the mHC
    // mixes, MLA's per-head key, the router's gate, the norms and biases - and a pack is free to store any of them
    // packed, as the release pack does. The bytes come up fp32 once, on the first window that needs them, and stay.
    for (size_t i = 0; i < fp32_src_.size(); ++i)
        if (fp32_src_[i] == t->ptr) return fp32_val_[i];
    uint64_t values = 1;
    for (uint64_t d : t->shape) values *= d;
    if (values >= kFp32SlotValues) {  // too big to keep: the next one of these uploads over this one
        if (big_src_ == t->ptr) return big_val_;
        float* up = dequantized(*t, err, big_values_ >= values ? big_val_ : nullptr, big_values_);
        if (!up) return nullptr;
        if (up != big_val_) {
            cudaFree(big_val_);
            big_val_ = up;
            big_values_ = values;
        }
        big_src_ = t->ptr;
        return up;
    }
    float* up = dequantized(*t, err);
    if (!up) return nullptr;
    fp32_src_.push_back(t->ptr);
    fp32_val_.push_back(up);
    memo_bytes_ += values * sizeof(float);
    return up;
}

/// A pack tensor's bytes, run through the same host dequantizers the embedding row uses - over the whole tensor this
/// time, in the order the tensor holds its values, which is the order fp32 rows want.
float* GlmForward::dequantized(const DevTensor& t, std::string& err, float* reuse, uint64_t reuse_values) {
    uint64_t blck = 0, bytes_per_block = 0;
    switch (t.type) {
        case kQ8_0: blck = 32; bytes_per_block = 34; break;
        case kQ4_K: blck = 256; bytes_per_block = 144; break;
        case kQ5_K: blck = 256; bytes_per_block = 176; break;
        case kQ6_K: blck = 256; bytes_per_block = 210; break;
        default: break;
    }
    uint64_t values = 1;
    for (uint64_t d : t.shape) values *= d;
    if (!blck || values % blck || t.bytes != values / blck * bytes_per_block) {
        err = "glm5next forward: '" + t.name + "' is " + ggml_type_name(t.type) +
              ", which this pass cannot bring up as fp32 rows";
        return nullptr;
    }
    std::vector<uint8_t> raw(t.bytes);
    if (!ck(cudaMemcpy(raw.data(), t.ptr, raw.size(), cudaMemcpyDeviceToHost), t.name.c_str(), err)) return nullptr;
    std::vector<float> vals(values);
    for (uint64_t b = 0, nb = values / blck; b < nb; ++b) {
        const uint8_t* p = raw.data() + b * bytes_per_block;
        float* o = vals.data() + b * blck;
        switch (t.type) {
            case kQ8_0: dequantize_q8_0(p, o); break;
            case kQ4_K: dequantize_q4_K(p, o); break;
            case kQ5_K: dequantize_q5_K(p, o); break;
            default: dequantize_q6_K(p, o); break;
        }
    }
    if (reuse && reuse_values >= values) {
        if (!ck(cudaMemcpy(reuse, vals.data(), vals.size() * sizeof(float), cudaMemcpyHostToDevice), t.name.c_str(),
                  err))
            return nullptr;
        return reuse;
    }
    float* out = nullptr;
    if (!ck(cudaMalloc(reinterpret_cast<void**>(&out), values * sizeof(float)), t.name.c_str(), err)) return nullptr;
    if (!ck(cudaMemcpy(out, vals.data(), vals.size() * sizeof(float), cudaMemcpyHostToDevice), t.name.c_str(), err)) {
        cudaFree(out);
        return nullptr;
    }
    return out;
}

/// The embedding: token_embd rows read straight from the pack, which is where that tensor deliberately stays. GGUF
/// order puts dim 0 - the embedding width - first, so a token's row is one contiguous run of the type's blocks, and
/// one read of that row is all it costs. The release pack does not store it in fp32, so the row comes out quantized
/// and is dequantized here, into the [T][E] block the streams want: the same trade deepseek41's embedding makes.
bool GlmForward::embed(const int32_t* tokens, int T, float* out, std::string& err) {
    const int E = geo_.n_embd;
    const TensorRef* te = model_.pack().find("token_embd.weight");
    if (!te || (int) te->shape[0] != E) {
        err = "glm5next forward: token_embd.weight is not the geometry's width";
        return false;
    }
    // One row's bytes, and the block the type quantizes over: the width has to fill whole blocks of it.
    int blck = 0;
    uint64_t row = 0;
    switch (te->type) {
        case kF32:  blck = 1;   row = 4ull * E; break;
        case kQ8_0: blck = 32;  row = (uint64_t) E / 32 * 34; break;
        case kQ4_K: blck = 256; row = (uint64_t) E / 256 * 144; break;
        case kQ5_K: blck = 256; row = (uint64_t) E / 256 * 176; break;
        case kQ6_K: blck = 256; row = (uint64_t) E / 256 * 210; break;
        default: break;
    }
    if (!row || E % blck) {
        err = "glm5next forward: token_embd.weight is " + std::string(ggml_type_name(te->type)) +
              ", which this pass cannot turn into a row of " + std::to_string(E) +
              " floats; it reads fp32, q8_0, q4_k, q5_k and q6_k on the host";
        return false;
    }
    if (te->bytes < row * te->shape[1]) {
        err = "glm5next forward: token_embd.weight holds " + std::to_string(te->bytes / row) +
              " rows of " + std::to_string(row) + " bytes, fewer than the " + std::to_string(te->shape[1]) +
              " its shape claims";
        return false;
    }
    std::vector<float> host((size_t) T * E);
    std::vector<uint8_t> buf(te->type == kF32 ? 0 : row);
    for (int t = 0; t < T; ++t) {
        const int32_t v = tokens[t];
        if (v < 0 || v >= (int32_t) geo_.n_vocab) {
            err = "glm5next forward: token id " + std::to_string(v) + " is outside the pack's vocab";
            return false;
        }
        float* dst = host.data() + (size_t) t * E;
        void* raw = te->type == kF32 ? (void*) dst : (void*) buf.data();
        if (!platform::pread_all(model_.read_fd(), raw, row, te->offset + (uint64_t) v * row)) {
            err = "glm5next forward: cannot read the embedding of token " + std::to_string(v);
            return false;
        }
        const uint8_t* p = buf.data();
        if (te->type == kQ8_0)
            for (int b = 0; b < E / 32; ++b) dequantize_q8_0(p + b * 34, dst + b * 32);
        else if (te->type == kQ4_K)
            for (int b = 0; b < E / 256; ++b) dequantize_q4_K(p + b * 144, dst + b * 256);
        else if (te->type == kQ5_K)
            for (int b = 0; b < E / 256; ++b) dequantize_q5_K(p + b * 176, dst + b * 256);
        else if (te->type == kQ6_K)
            for (int b = 0; b < E / 256; ++b) dequantize_q6_K(p + b * 210, dst + b * 256);
    }
    return ck(cudaMemcpyAsync(out, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice,
                              session_.stream()), "embedding", err);
}

bool GlmForward::layernorm_rows(const float* x, const float* w, const float* b, float* y, int T, int n, float eps,
                                std::string& err) {
    // Everything here is on the host, so the two norm vectors come over first: they live on the device like every
    // other weight, and reading them where the arithmetic runs is what this function exists to avoid.
    std::vector<float> h((size_t) T * n), o((size_t) T * n), ww((size_t) n), bb((size_t) n);
    if (!ck(cudaMemcpy(ww.data(), w, ww.size() * sizeof(float), cudaMemcpyDeviceToHost), "indexer key norm", err) ||
        !ck(cudaMemcpy(bb.data(), b, bb.size() * sizeof(float), cudaMemcpyDeviceToHost), "indexer key norm", err) ||
        !ck(cudaMemcpy(h.data(), x, h.size() * sizeof(float), cudaMemcpyDeviceToHost), "indexer key", err))
        return false;
    for (int t = 0; t < T; ++t) {
        const float* r = h.data() + (size_t) t * n;
        float* d = o.data() + (size_t) t * n;
        double mean = 0, var = 0;
        for (int i = 0; i < n; ++i) mean += r[i];
        mean /= n;
        for (int i = 0; i < n; ++i) { const double dv = r[i] - mean; var += dv * dv; }
        var /= n;
        const double inv = 1.0 / std::sqrt(var + eps);
        for (int i = 0; i < n; ++i) d[i] = (float) ((r[i] - mean) * inv * ww[i] + bb[i]);
    }
    return ck(cudaMemcpy(y, o.data(), o.size() * sizeof(float), cudaMemcpyHostToDevice), "indexer key", err);
}

/// The KDA (linear-attention) attention of one layer over a window. The recurrence is one token at a time, so the conv
/// window walks with it: kda_front reads the (conv-1) positions before a token from `hist` and never writes it, so the
/// caller shifts the window on after each token and writes the cache back at the end.
bool GlmForward::kda_attention(int layer, int kda_layer, float* normed, int T, std::string& err) {
    const int E = geo_.n_embd, nH = geo_.n_head, HD = geo_.kda_head_dim, D = nH * HD, CONV = geo_.conv;
    Scratch* s = s_;
    float* qn = s->front;
    float* kn = qn + D;
    float* vout = qn + 2 * D;
    float* gdec = qn + 3 * D;
    float* beta = qn + 4 * D;
    const float* ssm_a = weight(layer, "ssm_a", err);
    const float* dt = weight(layer, "ssm_dt.bias", err);
    const float* wnorm = weight(layer, "ssm_norm.weight", err);
    if (!ssm_a || !dt || !wnorm) return false;
    // One [T][D] slab per stream: the release pack keeps these three projections Q5_K, so each is its own product,
    // and kda_front walks the slabs by the window's token count rather than a contiguous q|k|v row.
    const size_t TD = (size_t) T * D;
    if (!proj(model_.tensor(layer, "attn_q.weight"), normed, s->qkv, T, D, E, err) ||
        !proj(model_.tensor(layer, "attn_k.weight"), normed, s->qkv + TD, T, D, E, err) ||
        !proj(model_.tensor(layer, "attn_v.weight"), normed, s->qkv + 2 * TD, T, D, E, err)) return false;
    if (!proj(model_.tensor(layer, "ssm_f_a.weight"), normed, s->a_mid, T, HD, E, err)) return false;
    if (!proj(model_.tensor(layer, "ssm_f_b.weight"), s->a_mid, s->a, T, D, HD, err)) return false;
    if (!proj(model_.tensor(layer, "ssm_g_a.weight"), normed, s->z_mid, T, HD, E, err)) return false;
    if (!proj(model_.tensor(layer, "ssm_g_b.weight"), s->z_mid, s->z, T, D, HD, err)) return false;
    if (!proj(model_.tensor(layer, "ssm_beta.weight"), normed, s->b_raw, T, nH, E, err)) return false;

    float* state = session_.caches().kda_state(0, kda_layer);
    float* hist = session_.caches().kda_conv(0, kda_layer);
    const float* convp = conv_pack_ + (size_t) kda_layer * 3 * CONV * D;
    const size_t win = (size_t) (CONV - 1) * 3 * D;
    if (!ck(cudaMemcpyAsync(s->conv_win, hist, win * sizeof(float), cudaMemcpyDeviceToDevice, session_.stream()),
            "conv window", err)) return false;

    for (int t = 0; t < T; ++t) {
        kda_front(s->qkv + (size_t) t * D, TD, s->conv_win, convp, s->a + (size_t) t * D, dt, ssm_a,
                  s->b_raw + (size_t) t * nH, qn, kn, vout, gdec, beta, nH, HD, session_.stream());
        kda_step(state, qn, kn, vout, gdec, beta, s->kda_o + (size_t) t * D, nH, HD, HD, session_.stream());
        if (CONV > 1 &&
            !ck(cudaMemcpyAsync(s->conv_win, s->conv_win + 3 * D, (size_t) (CONV - 2) * 3 * D * sizeof(float),
                                cudaMemcpyDeviceToDevice, session_.stream()), "conv shift", err)) return false;
        // The arriving position enters the window as one contiguous q|k|v row, gathered from the three slabs.
        float* row = s->conv_win + win - (size_t) 3 * D;
        for (int strm = 0; strm < 3; ++strm)
            if (!ck(cudaMemcpyAsync(row + (size_t) strm * D, s->qkv + (size_t) strm * TD + (size_t) t * D,
                                    (size_t) D * sizeof(float), cudaMemcpyDeviceToDevice, session_.stream()),
                    "conv append", err)) return false;
    }
    if (!ck(cudaMemcpyAsync(hist, s->conv_win, win * sizeof(float), cudaMemcpyDeviceToDevice, session_.stream()),
            "conv window", err)) return false;
    // the gated norm: z is the raw out gate (ssm_g_b's output), the weights are per head channel
    kda_norm(s->kda_o, s->z, wnorm, s->kda_n, T * nH, HD, geo_.rms_eps, session_.stream());
    return proj(model_.tensor(layer, "attn_output.weight"), s->kda_n, s->attn_out, T, E, D, err);
}

/// The indexer's pool bookkeeping and the cells this window attends.
///
/// A pool is complete when its kpool-th token arrives: its first kpool-1 rows are then the whole pending window, in
/// order, and the kpool-th is the token being computed - which is why the cache holds kpool-1 rows and why the pending
/// slot of position g is g % kpool.
///
/// The tail cells are the one place this pass reads more into the reference than the kernels prove. mla_sparse attends
/// exactly the ids cell_expand emits, and a token that attends nothing is undefined, so tail slot 0 always holds a
/// token's own cell and the rest hold the cells of its own pool before it (padded with the sentinel). That guarantees
/// the causal diagonal. cell_expand masks a tail cell whose pool this token already attends, because the pool's own
/// expansion emitted that cell: attending it twice doubles the weight of the freshest kpool-1 rows, which is most of a
/// short context and nothing in a long one. Checked against the reference in §5.6.
bool GlmForward::dsa_cells(int layer, int idx_layer, int T, int64_t pos, int& n_sel, std::string& err) {
    const int IDD = geo_.idx_dim, KP = geo_.kpool;
    Scratch* s = s_;
    float* pend = session_.caches().idx_pending(0, idx_layer);
    float* pooled = session_.caches().idx_pooled(0, idx_layer);
    const int pend_slots = KP - 1;
    const size_t pend_rows = (size_t) kIndexerPendingRows / 2;   // the keys, then the gates
    int n_new = 0;
    for (int t = 0; t < T; ++t) {
        const int64_t gi = pos + t;
        // The kpool-th token of a pool is the one completing it, and it is not part of the pending window: its row is
        // read from this window's own output when the pool is staged. Storing it would need a kpool-th slot, which the
        // cache does not have, and would land in the gates' rows and past the block.
        if ((gi + 1) % KP != 0) {
            const int slot = (int) (gi % KP);   // 0..kpool-2, in order, because only the completing token is skipped
            if (!ck(cudaMemcpyAsync(pend + (size_t) slot * IDD, s->ik + (size_t) t * IDD, (size_t) IDD * sizeof(float),
                                    cudaMemcpyDeviceToDevice, session_.stream()), "indexer pending key", err))
                return false;
            if (!ck(cudaMemcpyAsync(pend + (pend_rows + slot) * IDD, s->ig + (size_t) t * IDD,
                                    (size_t) IDD * sizeof(float), cudaMemcpyDeviceToDevice, session_.stream()),
                    "indexer pending gate", err)) return false;
            continue;
        }
        // the pool is complete: gather it (the pending window in order, then this token) and compress it into the cache
        float* dk = s->stage_k + (size_t) n_new * KP * IDD;
        float* dg = s->stage_g + (size_t) n_new * KP * IDD;
        const size_t run = (size_t) pend_slots * IDD * sizeof(float);
        const size_t one = (size_t) IDD * sizeof(float);
        if (pend_slots > 0 &&
            (!ck(cudaMemcpyAsync(dk, pend, run, cudaMemcpyDeviceToDevice, session_.stream()), "pool staging key", err) ||
             !ck(cudaMemcpyAsync(dg, pend + pend_rows * IDD, run, cudaMemcpyDeviceToDevice, session_.stream()),
                 "pool staging gate", err))) return false;
        if (!ck(cudaMemcpyAsync(dk + (size_t) pend_slots * IDD, s->ik + (size_t) t * IDD, one,
                                cudaMemcpyDeviceToDevice, session_.stream()), "pool staging key", err) ||
            !ck(cudaMemcpyAsync(dg + (size_t) pend_slots * IDD, s->ig + (size_t) t * IDD, one,
                                cudaMemcpyDeviceToDevice, session_.stream()), "pool staging gate", err)) return false;
        ++n_new;
    }
    const int n_pool = (int) ((pos + T) / KP);
    if (n_new > 0) {
        const float* ape = weight(layer, "indexer_compressor_ape.weight", err);
        if (!ape) return false;
        kpool_compress(s->stage_k, s->stage_g, ape, pooled + (size_t) (n_pool - n_new) * IDD, n_new, KP, IDD,
                       session_.stream());
    }

    // the tail: each token's own cell, then the cells of its own pool before it. Held on the host because the cells
    // are pure position arithmetic.
    std::vector<int32_t> tail((size_t) std::max(1, pend_slots) * T, kNoCell);
    for (int t = 0; t < T; ++t) {
        const int64_t gi = pos + t, start = (gi / KP) * KP;
        for (int sl = 0; sl < pend_slots; ++sl) {
            const int64_t cell = gi - sl;
            tail[(size_t) sl * T + t] = cell >= start ? (int32_t) cell : kNoCell;
        }
    }
    if (!ck(cudaMemcpyAsync(s->tail_idxs, tail.data(), tail.size() * sizeof(int32_t), cudaMemcpyHostToDevice,
                            session_.stream()), "tail cells", err)) return false;

    int n_top = 0;
    if (n_pool > 0) {
        n_top = std::max(1, std::min(n_pool, geo_.idx_top_k / KP));
        const float w_scale = 1.0f / std::sqrt((float) (IDD * geo_.idx_heads));
        idx_score(s->iq, s->iw, pooled, s->pos_i32, s->scores, T, n_pool, geo_.idx_heads, IDD, KP, w_scale,
                  session_.stream());
        idx_select(s->scores, s->sel_pools, s->out_n, T, n_pool, n_top, session_.stream());
        // A pool's member cells are `pool * kpool + slot`, which is how the latent cache is laid out, so the member
        // table is the identity over the pools that exist.
        std::vector<int32_t> host((size_t) KP * n_pool);
        for (int p = 0; p < n_pool; ++p)
            for (int sl = 0; sl < KP; ++sl) host[(size_t) sl * n_pool + p] = (int32_t) ((int64_t) p * KP + sl);
        if (!ck(cudaMemcpyAsync(s->pool_idxs, host.data(), host.size() * sizeof(int32_t), cudaMemcpyHostToDevice,
                                session_.stream()), "pool members", err)) return false;
    }
    // With no pool complete yet (the first tokens of a sequence) n_top is 0 and only the tail is attended, which is
    // what keeps the very first token's attention defined.
    cell_expand(s->sel_pools, s->pool_idxs, s->tail_idxs, kNoCell, s->ids, T, n_pool, KP, n_top, true,
                session_.stream());
    n_sel = cell_expand_n_sel(KP, n_top, true);
    return true;
}

/// The sparse MLA attention of one layer: the low-rank projections, this window's latents appended to the cache, the
/// indexer's cell list, then attention over those cells with the key projection absorbed into the query and the value
/// projection into the output.
bool GlmForward::mla_latents(int layer, float* normed, int T, int64_t first_pos, float* latent, std::string& err) {
    const int E = geo_.n_embd, nH = geo_.n_head, QL = geo_.q_lora, KV = geo_.kv_lora, QK = geo_.qk_dim;
    Scratch* s = s_;
    if (!proj(model_.tensor(layer, "attn_q_a.weight"), normed, s->q_lat, T, QL, E, err)) return false;
    kernels::rms_norm_rows(s->q_lat, weight(layer, "attn_q_a_norm.weight", err), T, QL, geo_.rms_eps, s->q_norm,
                           session_.stream());
    if (err.size()) return false;
    if (!proj(model_.tensor(layer, "attn_q_b.weight"), s->q_norm, s->q, T, nH * QK, QL, err)) return false;
    if (!proj(model_.tensor(layer, "attn_kv_a_mqa.weight"), normed, s->latent_row, T, KV, E, err)) return false;
    kernels::rms_norm_rows(s->latent_row, weight(layer, "attn_kv_a_norm.weight", err), T, KV, geo_.rms_eps,
                           s->lat_norm, session_.stream());
    if (err.size()) return false;
    return ck(cudaMemcpyAsync(latent + (size_t) first_pos * KV, s->lat_norm, (size_t) T * KV * sizeof(float),
                              cudaMemcpyDeviceToDevice, session_.stream()), "latent cache", err);
}

bool GlmForward::mla_read(int layer, int T, float* latent, const int32_t* ids, int n_sel, const int32_t* host_len,
                          std::string& err) {
    const int E = geo_.n_embd, nH = geo_.n_head, KV = geo_.kv_lora, QK = geo_.qk_dim, V = geo_.v_dim;
    Scratch* s = s_;
    // Absorb the key projection into the query: q_abs[t][h] = W_kb[h] . q[t][h] over qk_dim, into kv_lora. W_kb is
    // [qk_dim][kv_lora][n_head], one matrix a head, so this is a per-head product, not one wide one.
    if (!proj_heads(model_.tensor(layer, "attn_k_b.weight"), s->q, s->q_abs, T, nH, KV, QK, err)) return false;
    if (!ck(cudaMemcpyAsync(s->cache_len, host_len, (size_t) T * sizeof(int32_t), cudaMemcpyHostToDevice,
                            session_.stream()), "cache lengths", err))
        return false;
    mla_sparse(s->q_abs, latent, ids, s->cache_len, s->attn_h, 1.0f / std::sqrt((float) QK), T, nH, n_sel, s->mla_ws,
               session_.stream());
    if (!ck(cudaGetLastError(), "mla_sparse", err)) return false;
    // The attention's per-head latents, each through its own [kv_lora][v_dim] matrix, into the [n_head*v_dim] row
    // the output projection reads. `attn_h` is [T][n_head][kv_lora], so this is the same per-head product as the key
    // absorb above; multiplying it as one [n_head*kv_lora] row would give every head but the first the wrong slice.
    if (!proj_heads(model_.tensor(layer, "attn_v_b.weight"), s->attn_h, s->tmp, T, nH, V, KV, err)) return false;
    return proj(model_.tensor(layer, "attn_output.weight"), s->tmp, s->attn_out, T, E, nH * V, err);
}

bool GlmForward::mla_attention(int layer, int mla_layer, float* normed, int T, int64_t pos, std::string& err) {
    const int E = geo_.n_embd, QL = geo_.q_lora;
    Scratch* s = s_;
    float* latent = session_.caches().mla_latent(0, mla_layer);
    if (!mla_latents(layer, normed, T, pos, latent, err)) return false;

    // the indexer: its query comes from the same normed low-rank query the attention's heads come from, its key and
    // its compressor gate from the hidden state
    if (!proj(model_.tensor(layer, "indexer.attn_q_b.weight"), s->q_norm, s->iq, T, geo_.idx_heads * geo_.idx_dim, QL,
              err)) return false;
    if (!proj(model_.tensor(layer, "indexer.attn_k.weight"), normed, s->ik, T, geo_.idx_dim, E, err)) return false;
    if (!proj(model_.tensor(layer, "indexer_compressor_gate.weight"), normed, s->ig, T, geo_.idx_dim, E, err))
        return false;
    {
        const float* kw = weight(layer, "indexer.k_norm.weight", err);
        const float* kb = weight(layer, "indexer.k_norm.bias", err);
        if (!kw || !kb) return false;
        if (!layernorm_rows(s->ik, kw, kb, s->ln_key, T, geo_.idx_dim, geo_.ln_eps, err)) return false;
    }
    if (!ck(cudaMemcpyAsync(s->ik, s->ln_key, (size_t) T * geo_.idx_dim * sizeof(float), cudaMemcpyDeviceToDevice,
                            session_.stream()), "indexer key", err)) return false;
    if (!proj(model_.tensor(layer, "indexer.proj.weight"), normed, s->iw, T, geo_.idx_heads, E, err)) return false;
    if (!ck(cudaMemcpyAsync(s->pos_i32, host_pos_.data(), (size_t) T * sizeof(int32_t), cudaMemcpyHostToDevice,
                            session_.stream()), "positions", err)) return false;

    int n_sel = 0;
    if (!dsa_cells(layer, mla_layer, T, pos, n_sel, err)) return false;   // the indexer lives on the sparse layers
    // The drafter's block has no indexer of its own (the pack's `index_share_mtp`); it reads these cells after the
    // pass leaves them, so the last indexer's choice is the pass's to remember.
    last_n_sel_ = n_sel;
    return mla_read(layer, T, latent, s->ids, n_sel, host_len_.data(), err);
}

/// The drafter's block, in the shape the pack describes it: `eh_proj` over the normed embedding of the token that
/// follows a position beside the normed hidden *at* it, then one ordinary transformer block - attention over latents
/// of its own and the routed experts - and `shared_head_norm` before the shared output head. It is not part of the
/// mHC streams: the merge leaves a single stream, and the block carries it with the plain residual adds its
/// `attn_norm` and `ffn_norm` imply. Its indexer is the model's own (`index_share_mtp`), so its attention reads the
/// cells the last indexer chose rather than choosing its own.
bool GlmForward::mtp_block(const float* h, const float* emb_next, int T, int64_t first_pos, const int32_t* ids,
                           int n_sel, std::string& err) {
    const int E = geo_.n_embd;
    const int layer = mtp_layer();
    Scratch* s = s_;
    host_mtp_len_.resize((size_t) T);
    for (int t = 0; t < T; ++t) host_mtp_len_[t] = (int32_t) (first_pos + t + 1);

    const float* ew = weight(layer, "nextn.enorm.weight", err);
    const float* hw = weight(layer, "nextn.hnorm.weight", err);
    if (!ew || !hw) return false;
    // The two norms stage in the attention's and the FFN's output rows: both are dead by the time the block runs, and
    // the copy below reads them before either projection writes them again.
    kernels::rms_norm_rows(emb_next, ew, T, E, geo_.rms_eps, s->attn_out, session_.stream());
    kernels::rms_norm_rows(h, hw, T, E, geo_.rms_eps, s->ffn_out, session_.stream());
    if (err.size()) return false;
    auto cat = [&](const float* src, size_t off, const char* what) {
        return ck(cudaMemcpy2DAsync(s->mtp_cat + off, 2 * E * sizeof(float), src, E * sizeof(float), E * sizeof(float),
                                    T, cudaMemcpyDeviceToDevice, session_.stream()),
                  what, err);
    };
    if (!cat(s->attn_out, 0, "drafter merge") || !cat(s->ffn_out, (size_t) E, "drafter merge")) return false;
    if (!proj(model_.tensor(layer, "nextn.eh_proj.weight"), s->mtp_cat, s->cur, T, E, 2 * E, err)) return false;

    const float* aw = weight(layer, "attn_norm.weight", err);
    if (!aw) return false;
    kernels::rms_norm_rows(s->cur, aw, T, E, geo_.rms_eps, s->normed, session_.stream());
    if (err.size()) return false;
    float* latent = session_.caches().draft_latent(0, 0);
    if (!mla_latents(layer, s->normed, T, first_pos, latent, err)) return false;
    if (!mla_read(layer, T, latent, ids, n_sel, host_mtp_len_.data(), err)) return false;
    kernels::add_inplace(s->cur, s->attn_out, (int64_t) T * E, session_.stream());

    const float* fw = weight(layer, "ffn_norm.weight", err);
    if (!fw) return false;
    kernels::rms_norm_rows(s->cur, fw, T, E, geo_.rms_eps, s->normed, session_.stream());
    if (err.size()) return false;
    if (core::ExpertTier* dt = model_.draft_tier()) dt->begin_token(T);
    if (!mlp(layer, s->normed, T, err)) return false;
    kernels::add_inplace(s->cur, s->ffn_out, (int64_t) T * E, session_.stream());
    return ck(cudaGetLastError(), "the drafter's block", err);
}

/// The cache rows a window lets the block complete. Its attention reads its own latents at every position the model's
/// indexer names, so those rows have to have been written: for each position the window carries the token after, the
/// block can run - and for the one before the window, the hidden and the cells the pass kept last time round are
/// enough. A position the block never ran leaves a stale row, never a wrong one that survives: the same position is
/// computed again the moment the window that carries its next token arrives.
bool GlmForward::mtp_rows(const int32_t* tokens, int T, int64_t pos, std::string& err) {
    const int E = geo_.n_embd;
    Scratch* s = s_;
    if (!embed(tokens, T, s->mtp_e, err)) return false;   // row t wants the token after position pos+t-1: tokens[t]
    // The row before the window, as long as a kept hidden still covers that position.
    if (const int row = mtp_keep_row(pos - 1); row >= 0) {
        if (!mtp_block(mtp_prev_h_ + (size_t) row * E, s->mtp_e, 1, pos,
                       mtp_prev_ids_ + (size_t) row * mtp_sel_stride_, mtp_prev_n_sel_[row], err))
            return false;
    }
    // The window's own rows: hsum row t is position pos+t, the token after it is tokens[t+1], and the indexer's row t
    // is the cells that position attended - all three in the same row order.
    if (T > 1 && !mtp_block(s->hsum, s->mtp_e + (size_t) E, T - 1, pos + 1, s->ids, last_n_sel_, err)) return false;

    // Keep this window's last rows for the one after it - and for a draft that follows a rejected draft, whose anchor
    // is the row before the last one the pass ran.
    const int n = T < kMtpKeep ? T : kMtpKeep;
    if (!ck(cudaMemcpyAsync(mtp_prev_h_, s->hsum + (size_t) (T - n) * E, (size_t) n * E * sizeof(float),
                            cudaMemcpyDeviceToDevice, session_.stream()), "drafter hidden", err))
        return false;
    if (!ck(cudaMemcpyAsync(mtp_prev_ids_, s->ids + (size_t) (T - n) * mtp_sel_stride_,
                            (size_t) n * mtp_sel_stride_ * sizeof(int32_t), cudaMemcpyDeviceToDevice,
                            session_.stream()),
                    "drafter cells", err))
        return false;
    for (int i = 0; i < kMtpKeep; ++i) mtp_prev_n_sel_[i] = i < n ? last_n_sel_ : 0;
    mtp_keep_ = n;
    mtp_prev_pos_ = pos + T - 1;
    return true;
}

/// One draft: the block over the anchor's row, then `shared_head_norm` and the shared output head it shares with the
/// model, so the draft token is drawn the way the model draws its own.
bool GlmForward::draft(int32_t anchor, int64_t pos, int32_t& id, float& conf, std::string& err) {
    if (!drafts_) {
        err = "glm5next forward: this pack carries no drafter this build can run";
        return false;
    }
    if (!s_) { err = "glm5next forward: a draft needs a window before it"; return false; }
    if (pos < 1 || pos >= (int64_t) session_.caches().context()) {
        err = "glm5next forward: the draft anchor sits at position " + std::to_string(pos) + ", outside the cache";
        return false;
    }
    const int row = mtp_keep_row(pos - 1);
    if (row < 0) {
        err = "glm5next forward: the draft needs the hidden of position " + std::to_string(pos - 1) +
              ", and the last pass left its tail at position " + std::to_string(mtp_prev_pos_);
        return false;
    }
    Scratch* s = s_;
    const int E = geo_.n_embd;
    if (!embed(&anchor, 1, s->mtp_e, err)) return false;
    if (!mtp_block(mtp_prev_h_ + (size_t) row * E, s->mtp_e, 1, pos,
                   mtp_prev_ids_ + (size_t) row * mtp_sel_stride_, mtp_prev_n_sel_[row], err))
        return false;
    const float* nw = weight(mtp_layer(), "nextn.shared_head_norm.weight", err);
    if (!nw) return false;
    kernels::rms_norm_rows(s->cur, nw, 1, E, geo_.rms_eps, s->normed, session_.stream());
    if (err.size()) return false;
    if (!proj(model_.tensor("output.weight"), s->normed, mtp_logits_, 1, geo_.n_vocab, E, err)) return false;
    mtp_host_.resize((size_t) geo_.n_vocab);
    if (!ck(cudaMemcpy(mtp_host_.data(), mtp_logits_, (size_t) geo_.n_vocab * sizeof(float), cudaMemcpyDeviceToHost),
            "drafter logits", err))
        return false;
    const int V = geo_.n_vocab;
    int best = 0;
    float best_v = mtp_host_[0];
    for (int v = 1; v < V; ++v)
        if (mtp_host_[v] > best_v) {
            best_v = mtp_host_[v];
            best = v;
        }
    double sum = 0.0;
    for (int v = 0; v < V; ++v) sum += std::exp((double) mtp_host_[v] - (double) best_v);
    id = best;
    conf = sum > 0.0 ? (float) (1.0 / sum) : 0.0f;
    return ck(cudaGetLastError(), "the drafter", err);
}

// A second-card share armed for a block that never reached its plan still has to be released: its kernels wait on the
// card for the plan word, and the next `arm` waits for that share to leave the stream. Releasing a share that was
// raised costs nothing (core::PeerExperts::raise returns at once when nothing is armed).
struct PeerArm {
    core::PeerExperts* peer = nullptr;
    ~PeerArm() {
        std::string e;
        if (peer) peer->raise(0, nullptr, nullptr, e);
    }
};

/// The MLP of one layer: the dense lead layers' SwiGLU, the rest the router over the resident experts plus the shared
/// ones added on top. The result lands in `ffn_out`.
bool GlmForward::mlp(int layer, const float* normed, int T, std::string& err) {
    Scratch* s = s_;
    const int E = geo_.n_embd;
    // A pack states the SwiGLU clamp per block (swiglu_clamp_exp / swiglu_clamp_shexp); one that states none, or 0 for a
    // layer, wants the unclamped gate.
    auto clamp_of = [](const std::vector<float>& v, int l) { return l < (int) v.size() ? v[(size_t) l] : 0.f; };
    const float clamp_exp = clamp_of(geo_.clamp_exp, layer), clamp_shexp = clamp_of(geo_.clamp_shexp, layer);
    if (layer < geo_.leading_dense) {
        if (!proj(model_.tensor(layer, "ffn_gate.weight"), normed, s->g_d, T, geo_.n_ff, E, err)) return false;
        if (!proj(model_.tensor(layer, "ffn_up.weight"), normed, s->u_d, T, geo_.n_ff, E, err)) return false;
        if (clamp_exp > 0.f)
            kernels::swiglu_clamped(s->g_d, s->u_d, clamp_exp, s->h_d, T, geo_.n_ff, geo_.n_ff, session_.stream());
        else
            kernels::swiglu_scaled(s->g_d, s->u_d, nullptr, nullptr, s->h_d, T, geo_.n_ff, geo_.n_ff, session_.stream());
        return proj(model_.tensor(layer, "ffn_down.weight"), s->h_d, s->ffn_out, T, E, geo_.n_ff, err);
    }
    core::ExpertTier* tier = layer == mtp_layer() ? model_.draft_tier() : model_.tier();
    // The drafter's block has a tier of its own because its experts are a section of their own, in formats of their
    // own: that tier covers one layer, so it knows its block as layer 0.
    const int tlayer = layer == mtp_layer() ? 0 : layer;
    if (!tier || (layer != mtp_layer() && !model_.has_experts(layer))) {
        err = "glm5next forward: layer " + std::to_string(layer) +
              " routes to experts and the model opened no expert tier for them";
        return false;
    }
    const int K = geo_.n_used;
    const kernels::NativeExpertLayout& L =
        layer == mtp_layer() ? model_.draft_expert_layout() : model_.expert_layout(layer);
    const float* gate = weight(layer, "ffn_gate_inp.weight", err);
    const float* bias = weight(layer, "exp_probs_b.bias", err);
    if (!gate || !bias) return false;
    kernels::RouterArgs ra;
    ra.n_expert = geo_.n_expert;
    ra.n_embd = E;
    ra.k = K;
    ra.weight_scale = geo_.expert_scale;
    // The pack's own gate (expert_gating_func); open_pack refuses a pack that states any other, so this is the only
    // one a glm5next pack can reach. DeepSeek 4.1, which shares this router, leaves the default sqrt-softplus because
    // that is its model's gate.
    ra.gate = kernels::RouterGate::Sigmoid;
    kernels::router(normed, gate, bias, T, ra, s->rlogits, s->rids, s->rweights, session_.stream());
    // The tier takes its ids on the host and its CPU pool reads the activation there, so the router's ids and weights
    // and this window's rows come across once per layer. What each side writes stays where it was written: the pool's
    // rows in the pinned block, the card's in `ex_rows_`, and one kernel per block sums the two.
    host_rids_.resize((size_t) T * K);
    host_rweights_.resize((size_t) T * K);
    host_act_.resize((size_t) T * E);
    if (!ck(cudaMemcpy(host_rids_.data(), s->rids, host_rids_.size() * sizeof(int32_t), cudaMemcpyDeviceToHost),
            "expert ids", err) ||
        !ck(cudaMemcpy(host_rweights_.data(), s->rweights, host_rweights_.size() * sizeof(float),
                       cudaMemcpyDeviceToHost),
            "expert weights", err) ||
        !ck(cudaMemcpy(host_act_.data(), normed, host_act_.size() * sizeof(float), cudaMemcpyDeviceToHost),
            "expert activation", err))
        return false;
    if (!ck(cudaMemsetAsync(s->ffn_out, 0, (size_t) T * E * sizeof(float), session_.stream()), "expert sum", err))
        return false;
    // The second card that holds this layer's experts, if the tier was dealt one. It computes its experts' rows with the
    // same grouped kernels the card's own experts use, reading this block's activations from the model's pinned block,
    // and scatters the rows into the same pinned rows the CPU pool writes into: neither card here can read the other's
    // memory (no peer access between consumer cards), so every row a peer computes comes home over the host - and the
    // sum already reads those rows for every pick the card did not write itself.
    core::PeerExperts* peer = tier->peer_for(tlayer);
    kernels::NativeExpertLayout playout = L;   // the peer's grouped kernels take the SwiGLU clamp inside the layout
    playout.swiglu_limit = clamp_exp;
    PeerArm armed;
    // The window in blocks of the tier's size. Everything a block needs is decided by one lookup of its distinct
    // picks, so the pool, the card's link and the card itself all start on this block's experts at once.
    for (int t0 = 0; t0 < T; t0 += kExpertRows) {
        const int nt = std::min(kExpertRows, T - t0);
        // The peer's share goes out before anything else: its kernels wait on the card for the plan, so it computes
        // while this block's lookups and the card's launches go out. Its activations are the block's rows, which the
        // pool reads from the pageable copy, so they are copied to the pinned block the peer's card maps.
        if (peer) {
            std::memcpy(model_.host_act(), host_act_.data() + (size_t) t0 * E, (size_t) nt * E * sizeof(float));
            if (!peer->arm(playout, model_.host_act(), nt, K, core::policy::peer_groups(nt), err)) return false;
            armed.peer = peer;
        }
        // what this layer's head left the bus of the promotions queued before it
        tier->pump(tlayer, 2 + nt);
        // one q8_1 activation per token of the block, for every expert the card computes
        kernels::native_quantize_q8_1(normed + (size_t) t0 * E, x_q8_, E, nt, session_.stream());
        // the block's distinct picks, and which of its rows each one covers
        int32_t uniq[kMaxSumRows];
        int uidx[kMaxSumRows];   // [t * K + k] -> index into uniq
        core::ExpertTier::Use use[kMaxSumRows];
        int nu = 0;
        for (int t = 0; t < nt; ++t)
            for (int k = 0; k < K; ++k) {
                const int32_t e = host_rids_[(size_t) (t0 + t) * K + k];
                int u = 0;
                while (u < nu && uniq[u] != e) ++u;
                if (u == nu) {
                    uniq[nu] = e;
                    use[nu] = core::ExpertTier::Use{};
                    ++nu;
                }
                uidx[t * K + k] = u;
                core::ExpertTier::Use& us = use[u];
                us.tok[us.nt] = t;
                us.row[us.nt] = t * K + k;   // the same row index in the pool's block, the card's, and the sum's
                ++us.nt;
            }
        core::ExpertTier::Lookup lk;
        if (!tier->lookup(tlayer, uniq, nu, lk, err)) return false;
        const int n_fetched = tier->fetch(tlayer, lk);   // the pinned misses the card's link copies in
        // The peer's plan, raised before its rows are needed and before the pool starts: entry t * K + k of the block,
        // 0 not its expert, 2 one of the experts it holds, 3 one the tier is copying into its scratch slots. A pick
        // the router dropped has no source at all, so the peer is told nothing about it.
        if (peer) {
            int32_t kind[kMaxSumRows];
            for (int r = 0; r < nt * K; ++r) {
                const int i = uidx[r];
                kind[r] = !lk.peer[i] ? 0 : lk.fetched[i] ? 3 : 2;
            }
            if (!peer->raise(tlayer, host_rids_.data() + (size_t) t0 * K, kind, err)) return false;
        }
        bool cpu = false, reading = false;
        for (int i = 0; i < nu; ++i) {
            cpu = cpu || (!lk.dev[i] && !lk.peer[i] && !lk.reading[i] && !lk.dropped[i]);
            reading = reading || lk.reading[i];
        }
        // The pool's share goes first: it takes the longest, and its workers run while the card's share launches.
        if (cpu) tier->start_cpu_window(tlayer, host_act_.data() + (size_t) t0 * E, lk, use, model_.host_rows());
        // Where every row of the block will come from: the card's own experts, the pool's pinned rows (mapped, so one
        // launch reads both), or nothing at all for a pick the router dropped.
        RowPtrs rp{};
        for (int r = 0; r < nt * K; ++r) {
            const int i = uidx[r];
            rp.p[r] = lk.dev[i] ? ex_rows_ + (size_t) r * E
                                : lk.dropped[i] ? tier->zero_row()
                                                : model_.host_rows_dev() + (size_t) r * E;
        }
        // The card's share: the experts the tier homes, in a plan of its own, and behind it the ones its link is still
        // copying in. The plan travels to the device in a kernel's parameters, so it goes out in stream order and this
        // thread may reuse it on the next line.
        const auto gpu_share = [&](ExpertPlan& pl, int ngroups, int nentries, void* dev_plan) {
            put_words(&pl, dev_plan, kPlanWords, session_.stream());
            const ExpertPlan* dp = (const ExpertPlan*) dev_plan;
            kernels::native_expert_grouped_clamped(L, clamp_exp, dp->ptr, dp->start, dp->ng, dp->dst, dp->tok, ngroups,
                                                   nentries, x_q8_, ex_scr_, ex_rows_, session_.stream());
        };
        ExpertPlan held{}, fetched{};
        int ngh = 0, neh = 0, ngf = 0, nef = 0;
        for (int i = 0; i < nu; ++i) {
            if (!lk.dev[i]) continue;
            ExpertPlan& pl = lk.fetched[i] ? fetched : held;
            int& ng = lk.fetched[i] ? ngf : ngh;
            int& ne = lk.fetched[i] ? nef : neh;
            pl.ptr[ng] = (unsigned long long) lk.dev[i];
            pl.start[ng] = ne;
            for (int j = 0; j < use[i].nt; ++j) {
                pl.dst[ne] = use[i].row[j];
                pl.tok[ne] = use[i].tok[j];
                ++ne;
            }
            ++ng;
        }
        held.start[ngh] = neh;
        held.ng[0] = ngh;
        if (ngh) gpu_share(held, ngh, neh, plan_dev_[0]);
        if (n_fetched) {
            fetched.start[ngf] = nef;
            fetched.ng[0] = ngf;
            tier->await_fetch(session_.stream());   // the copies the fetch started, now landed
            if (ngf) gpu_share(fetched, ngf, nef, plan_dev_[1]);
        }
        // The experts the peer is fetching into its own scratch slots go behind their copies, on its card.
        if (peer && !peer->run_fetched(err)) return false;
        // The pool's rows have landed by the time the sum runs, and a place still being read from the pack joins the
        // pool's share once its read does.
        if (cpu) tier->finish_cpu();
        if (reading) {
            if (!tier->await_reads(err)) return false;
            tier->start_cpu_window(tlayer, host_act_.data() + (size_t) t0 * E, lk, use, model_.host_rows(),
                                   /*reading=*/true);
            tier->finish_cpu();
        }
        // The peer's rows are in the same pinned rows by now: the host waits for the peer the way it waits for the
        // pool, with both cards still working while it does.
        if (peer && !peer->finish(nullptr, err)) return false;
        weighted_sum_block(rp, s->rweights + (size_t) t0 * K, nt, K, E, s->ffn_out + (size_t) t0 * E,
                           session_.stream());
        if (cpu || reading) tier->promote(tlayer, lk);
        tier->mark_gap(session_.stream());   // the promotions' copies start behind the sum
    }
    const int sh = geo_.shared_ffn() * geo_.n_shared;
    if (sh > 0) {
        if (!proj(model_.tensor(layer, "ffn_gate_shexp.weight"), normed, s->sh_g, T, sh, E, err)) return false;
        if (!proj(model_.tensor(layer, "ffn_up_shexp.weight"), normed, s->sh_u, T, sh, E, err)) return false;
        if (clamp_shexp > 0.f)
            kernels::swiglu_clamped(s->sh_g, s->sh_u, clamp_shexp, s->sh_h, T, sh, sh, session_.stream());
        else
            kernels::swiglu_scaled(s->sh_g, s->sh_u, nullptr, nullptr, s->sh_h, T, sh, sh, session_.stream());
        if (!proj(model_.tensor(layer, "ffn_down_shexp.weight"), s->sh_h, s->g_d, T, E, sh, err)) return false;
        kernels::add_scaled(s->ffn_out, s->g_d, nullptr, (int64_t) T * E, session_.stream());
    }
    return true;
}

bool GlmForward::run(int seq, const int32_t* tokens, int n_tokens, int64_t pos, std::string& err, int logit_rows) {
    logit_rows_ = 1;
    if (!ok_) { err = "glm5next forward: this pack is not usable"; return false; }
    if (n_tokens <= 0) { err = "glm5next forward: an empty window"; return false; }
    if (seq != 0) {
        err = "glm5next forward: the indexer's pending-window bookkeeping is kept for one sequence at a time";
        return false;
    }
    const uint64_t end = (uint64_t) pos + (uint64_t) n_tokens;
    if (pos < 0 || end > session_.caches().context()) {
        err = "glm5next forward: the window reaches position " + std::to_string(end) + ", past the " +
              std::to_string(session_.caches().context()) + "-token cache";
        return false;
    }
    if (n_tokens > cap_ && !grow(n_tokens, err)) return false;
    // The tier counts tokens for its placement bookkeeping; a window is one row per token.
    if (core::ExpertTier* tier = model_.tier()) tier->begin_token(n_tokens);
    Scratch* s = s_;
    const int E = geo_.n_embd;
    host_pos_.resize(n_tokens);
    host_len_.resize(n_tokens);
    for (int t = 0; t < n_tokens; ++t) {
        host_pos_[t] = (int32_t) (pos + t);
        host_len_[t] = (int32_t) (pos + t + 1);
    }

    if (!embed(tokens, n_tokens, s->normed, err)) return false;
    // The embedding seeds every stream, so the head of layer 0 (stream 0 of the row) sees the embedding itself. The
    // streams are [T][4][E], so each stream i is one strided copy of the [T][E] embedding.
    for (int i = 0; i < kHcMult; ++i) {
        if (!ck(cudaMemcpy2DAsync(s->streams[0] + (size_t) i * E, kHcMult * E * sizeof(float), s->normed,
                                  E * sizeof(float), E * sizeof(float), n_tokens, cudaMemcpyDeviceToDevice,
                                  session_.stream()), "streams", err)) return false;
    }

    kernels::HcArgs hca;
    hca.n_embd = E;
    hca.rms_eps = geo_.rms_eps;   // the mHC input norm is the model's own unweighted rms norm
    hca.hc_eps = geo_.hc_eps;
    hca.sinkhorn_iters = geo_.sinkhorn_iters;

    int kda_layer = 0, mla_layer = 0;
    float* x = s->streams[0];
    float* other = s->streams[1];
    for (int il = 0, nl = geo_.n_model_layers(); il < nl; ++il) {
        // --- the attention sublayer: its own mixes first, then the collapse those mixes feed, then the expansion
        // back. Each sublayer collapses with its own hyper-connection's `pre` (the reference's attn_hc / ffn_hc), and
        // the same mix set also carries the post/comb that expand that sublayer's own branch output.
        const float* hfn = weight(il, "hc_attn_fn.weight", err);
        const float* hsc = weight(il, "hc_attn_scale.weight", err);
        const float* hbs = weight(il, "hc_attn_base.weight", err);
        const float* aw = weight(il, "attn_norm.weight", err);
        if (!hfn || !hsc || !hbs || !aw) return false;
        kernels::hc_mixes(x, hfn, hsc, hbs, n_tokens, hca, s->hc_ws, s->coef_a, session_.stream());
        kernels::hc_pre_rms_norm(x, s->coef_a, kHcCoef, aw, n_tokens, E, geo_.rms_eps, s->cur, s->normed,
                                 session_.stream());
        if (geo_.is_mla(il)) {
            if (!mla_attention(il, mla_layer, s->normed, n_tokens, pos, err)) return false;
            ++mla_layer;
        } else {
            if (!kda_attention(il, kda_layer, s->normed, n_tokens, err)) return false;
            ++kda_layer;
        }
        kernels::hc_post(s->attn_out, x, s->coef_a, kHcCoef, n_tokens, E, other, session_.stream());
        // --- the FFN sublayer: its own mixes, then the collapse those same mixes feed
        const float* ffn = weight(il, "hc_ffn_fn.weight", err);
        const float* fsc = weight(il, "hc_ffn_scale.weight", err);
        const float* fbs = weight(il, "hc_ffn_base.weight", err);
        const float* fw = weight(il, "ffn_norm.weight", err);
        if (!ffn || !fsc || !fbs || !fw) return false;
        kernels::hc_mixes(other, ffn, fsc, fbs, n_tokens, hca, s->hc_ws, s->coef_f, session_.stream());
        kernels::hc_pre_rms_norm(other, s->coef_f, kHcCoef, fw, n_tokens, E, geo_.rms_eps, s->cur, s->normed,
                                 session_.stream());
        if (!mlp(il, s->normed, n_tokens, err)) return false;
        kernels::hc_post(s->ffn_out, other, s->coef_f, kHcCoef, n_tokens, E, x, session_.stream());
        // The block ends where it began: `x` holds the streams its FFN left, and the next block reads `x`. The two
        // hc_posts alternate inside the block - attention reads `x` and writes `other`, the FFN reads `other` and
        // writes `x` - so swapping buffers here would drop this block's FFN out of the residual stream.
    }

    // the head: the unweighted mean of the four streams, its norm, then the projection of the window's last row
    kernels::hc_pre(x, s->hc_head, kHcMult, n_tokens, E, s->hsum, session_.stream());
    const float* ow = weight(-1, "output_norm.weight", err);
    if (!ow) return false;
    kernels::rms_norm_rows(s->hsum, ow, n_tokens, E, geo_.rms_eps, s->normed, session_.stream());
    // A drafted window needs a pick per row, so its rows all go through the output projection. Their buffer is the one
    // allocation here that the window plan does not count: at most a few rows of vocab floats, inside the reserve the
    // tier is told to leave for the pass.
    const int rows = std::min(std::max(logit_rows, 1), n_tokens);
    if (rows == 1) {
        if (!proj(model_.tensor("output.weight"), s->normed + (size_t) (n_tokens - 1) * E, session_.logits(), 1,
                  geo_.n_vocab, E, err)) return false;
    } else {
        const uint64_t bytes = (uint64_t) rows * geo_.n_vocab * sizeof(float);
        if (bytes > row_bytes_) {
            cudaFree(row_rows_);
            row_rows_ = nullptr;
            row_bytes_ = 0;
            if (cudaMalloc(reinterpret_cast<void**>(&row_rows_), bytes) != cudaSuccess) {
                err = "glm5next forward: the row logits of a " + std::to_string(rows) + "-row window: out of memory";
                return false;
            }
            row_bytes_ = bytes;
        }
        if (!proj(model_.tensor("output.weight"), s->normed + (size_t) (n_tokens - rows) * E, row_rows_, rows,
                  geo_.n_vocab, E, err)) {
            cudaFree(row_rows_);
            row_rows_ = nullptr;
            row_bytes_ = 0;
            return false;
        }
        logit_rows_ = rows;
    }
    // The drafter's cache rows this window completes. It runs inside the window's own pass because everything it
    // reads - the collapsed hiddens, the indexer's cells, the scratch - belongs to this window.
    if (drafts_ && !mtp_rows(tokens, n_tokens, pos, err)) return false;
    return ck(cudaStreamSynchronize(session_.stream()), "the forward pass", err);
}

bool GlmForward::copy_logits(float* out, std::string& err) {
    return ck(cudaMemcpy(out, session_.logits(), (size_t) geo_.n_vocab * sizeof(float), cudaMemcpyDeviceToHost),
              "logits", err);
}

bool GlmForward::copy_row_logits(float* out, std::string& err) {
    if (!logit_rows_) {
        err = "glm5next forward: this window kept no row logits";
        return false;
    }
    return ck(cudaMemcpy(out, row_rows_, (size_t) logit_rows_ * geo_.n_vocab * sizeof(float), cudaMemcpyDeviceToHost),
              "row logits", err);
}

uint64_t forward_footprint_bytes(const Geometry& g, const Pack& p) {
    // The widest packed input sets the q8_1 activation block; the constructor measures the same thing over the tensors
    // it accepts, and the pack's table is that set plus anything it refuses, so the answer here is never smaller.
    int kqmax = 0;
    uint64_t memo = 0;
    for (const auto& [name, t] : p.tensors) {
        if (t.type == 0) continue;
        if (!t.shape.empty() && (int) t.shape[0] > kqmax) kqmax = (int) t.shape[0];
        uint64_t values = 1;
        for (uint64_t d : t.shape) values *= d;
        if (values < kFp32SlotValues) memo += values * sizeof(float);
    }
    const int D = g.n_head * g.kda_head_dim;
    uint64_t bytes = kQ8Slack;
    if (kqmax > 0) bytes += kernels::native_q8_1_bytes(kqmax, kMmvqCols);
    bytes += kernels::native_q8_1_bytes(g.n_embd, kExpertRows);
    bytes += kernels::native_expert_scratch_bytes(kMaxSumRows, g.n_ff_exp);
    bytes += (uint64_t) kMaxSumRows * g.n_embd * sizeof(float);
    bytes += 2 * sizeof(ExpertPlan);
    int n_kda = 0;
    for (int l = 0; l < g.n_model_layers(); ++l) if (!g.is_mla(l)) ++n_kda;
    if (D > 0) bytes += (uint64_t) n_kda * 3 * g.conv * D * sizeof(float);
    // A window wider than the narrow product's columns multiplies a packed weight through MMQ, which steps along its
    // rows 256 values at a time: a weight whose rows are shorter than a whole tile gets a padded copy of its own bytes,
    // kept for the rest of the session. Which of them that is comes from the same kernel's rule.
    for (const auto& [name, t] : p.tensors)
        if (t.type != 0 && !t.shape.empty() && prefill::mmq::read_past_bytes(t.type, (int64_t) t.shape[0]))
            bytes += t.bytes;
    return bytes + memo;
}

}  // namespace basalt::models::glm5next
