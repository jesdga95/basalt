// src/models/hy4flash/session.cpp - one conversation of Hy4-flash: its latent caches, indexer rows and pooled keys, and the
// forward of one token that every window and prompt row runs.
//
// Per layer: the Engram step (on its layers), the attention's hyper-connection pre-mix and norm, the absorbed MLA over
// the shared latent (a window ring on the window layers, the indexer's selection on the others), the gate, the output
// projection and its post-norm, the stream expansion, and the same for the FFN: the dense MLP, or the latent MoE (router,
// the routed experts on the CPU beside the shared expert on the card, the up projection).
#include "basalt/core/expert_tier.hpp"
#include "basalt/kernels/bf16_gemv.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/dense_ops.hpp"
#include "basalt/kernels/logprobs.hpp"
#include "basalt/kernels/mhc.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/log.hpp"
#include "basalt/models/hy4flash/kernels.hpp"
#include "models/hy4flash/engram.hpp"
#include "models/hy4flash/model.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace basalt::models::hy4flash {
namespace {

namespace bk = basalt::kernels;

constexpr int kRowWidth = 576;   // a cache row: 512 latent values and 64 rotated
constexpr uint64_t kModuleRoom = 7ull << 27;   // kernel modules still to load when the tier opens
constexpr uint64_t kPromptRoom = 5ull << 29;   // the prompt path's workspaces beside the homes

// One layer's GPU plan for the grouped expert kernel, uploaded with a single copy.
struct ExpertPlan {
    unsigned long long ptr[16];
    int32_t start[17];
    int32_t ng[3];
    int32_t dst[16];
    int32_t tok[16];
};

class Session final : public model::ModelSession {
public:
    explicit Session(Hy4Model& m) : m_(m), g_(m.geo()) {}
    ~Session() override {
        for (void* p : bufs_) cudaFree(p);
        for (void* p : pinned_) cudaFreeHost(p);
        if (ev_) cudaEventDestroy(ev_);
        if (stream_) cudaStreamDestroy(stream_);
    }

    bool init(std::string& err) {
        if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags(&ev_, cudaEventDisableTiming) != cudaSuccess) {
            err = "cannot create a stream";
            return false;
        }
        const int64_t E = g_.n_embd, ctx = m_.context();
        int max_heads = 0;
        for (int h : g_.heads) max_heads = std::max(max_heads, h);
        bool ok = true;
        auto f = [&](int64_t n) { return (float*) alloc((size_t) n * 4, ok); };
        auto pin = [&](size_t n) {
            void* p = nullptr;
            if (cudaMallocHost(&p, n * 4) != cudaSuccess) ok = false;
            else pinned_.push_back(p);
            return (float*) p;
        };
        X_ = f(4 * E); Xn_ = f(4 * E); coef_ = f(bk::kHcCoef); hcws_ = f((int64_t) bk::hc_plain_workspace_floats(1));
        e_ = f(E); cur_ = f(E); xn_ = f(E); attn_ = f(E); attn_raw_ = f(E); mlp_ = f(E); mlp2_ = f(E); fin_ = f(E);
        qa_ = f(g_.q_lora); qan_ = f(g_.q_lora); q_ = f((int64_t) max_heads * g_.qk_dim); kvraw_ = f(kRowWidth);
        qabs_ = f((int64_t) max_heads * g_.kv_lora); qcur_ = f((int64_t) max_heads * kRowWidth);
        olat_ = f((int64_t) max_heads * g_.kv_lora); vout_ = f((int64_t) max_heads * g_.v_dim); gate_ = f((int64_t) max_heads * g_.v_dim);
        ik_ = f(g_.idx_dim); ikn_ = f(g_.idx_dim); iq_ = f((int64_t) g_.idx_heads * g_.idx_dim); iw_ = f(g_.idx_heads);
        const int64_t pools = ctx / g_.idx_pool + 1;
        iscore_ = f(pools);
        icells_ = (int32_t*) alloc((size_t) (g_.idx_top_k + g_.idx_pool) * 4, ok);
        gu_ = f(2 * (int64_t) g_.n_ff); h_ = f(g_.n_ff);
        rlog_ = f(g_.n_expert); rw_ = f(16); ids_ = (int32_t*) alloc(16 * 4, ok);
        xlat_ = f(g_.n_latent); lat_out_ = f(g_.n_latent); moe_out_ = f(E); sh_ = f(E);
        logits_ = f(g_.n_vocab); pick_ = (int32_t*) alloc(4 * 16, ok);
        erows_ = f(g_.n_embd);   // the Engram's normed rows: 16 x 96 values
        ekv_ = f(5 * E);
        lp_ = f((int64_t) bk::row_logprobs_scratch_floats(1, (int) g_.n_vocab) + bk::kLogprobStride + 16);
        const size_t q8 = std::max<size_t>(bk::native_q8_1_bytes(16384, 1), 1 << 16);
        q8a_ = alloc(q8, ok); q8b_ = alloc(q8, ok); q8c_ = alloc(q8, ok);
        h_rw_ = pin(16); h_ids_ = (int32_t*) pin(16);
        h_erows_ = pin(g_.n_embd); h_e_ = pin(E);
        ex_xq_ = alloc(bk::native_q8_1_bytes(g_.n_latent, 1), ok);
        ex_scr_ = alloc(bk::native_expert_scratch_bytes(16, g_.n_ff_exp), ok);
        ex_rows_ = f((int64_t) 16 * g_.n_latent);
        ex_plan_ = alloc(sizeof(ExpertPlan), ok);
        if (cudaHostAlloc((void**) &h_plan_, sizeof(ExpertPlan), cudaHostAllocPortable) != cudaSuccess) ok = false;
        else pinned_.push_back(h_plan_);
        cache_.assign((size_t) g_.n_layers, nullptr);
        ival_.assign((size_t) g_.n_layers, nullptr);
        igate_.assign((size_t) g_.n_layers, nullptr);
        ipool_.assign((size_t) g_.n_layers, nullptr);
        for (int l = 0; ok && l < g_.n_layers; ++l) {
            if (g_.swa[(size_t) l]) {
                cache_[(size_t) l] = f((int64_t) g_.window * kRowWidth);
            } else {
                cache_[(size_t) l] = f(ctx * kRowWidth);
                ival_[(size_t) l] = f(ctx * g_.idx_dim);
                igate_[(size_t) l] = f(ctx * g_.idx_dim);
                ipool_[(size_t) l] = f(pools * g_.idx_dim);
            }
        }
        if (!ok) {
            err = "out of device memory for the session";
            return false;
        }
        return true;
    }

    model::Result decode_window(const model::DecodeRequest& r) override {
        if (r.n_tokens < 1) return {model::Status::Error, "hy4flash: an empty window"};
        std::string err;
        for (int t = 0; t < r.n_tokens; ++t) {
            const bool last = t == r.n_tokens - 1;
            const bool want = r.row_tokens || (last && (r.next_token || r.logits));
            if (!forward(r.tokens[t], r.pos + t, want, err)) return {model::Status::Error, "hy4flash: " + err};
            if (want) {
                int32_t tok = 0;
                if (!pick(r, t, tok, err)) return {model::Status::Error, "hy4flash: " + err};
                if (r.row_tokens) r.row_tokens[t] = tok;
                if (last && r.next_token) *r.next_token = tok;
                if (last && r.logits) cudaMemcpyAsync(r.logits, logits_, (size_t) g_.n_vocab * 4, cudaMemcpyDeviceToHost, stream_);
            }
        }
        if (cudaStreamSynchronize(stream_) != cudaSuccess) return {model::Status::Error, "hy4flash: the window failed"};
        return {};
    }

    model::Result prefill_chunk(const model::PrefillRequest& r) override {
        std::string err;
        for (int t = 0; t < r.n_tokens; ++t) {
            if (r.stop && r.stop()) break;
            if (!forward(r.tokens[t], r.pos + t, false, err)) return {model::Status::Error, "hy4flash: " + err};
            if (r.progress && ((t + 1) % 16 == 0 || t + 1 == r.n_tokens)) {
                cudaStreamSynchronize(stream_);
                r.progress(t + 1);
            }
        }
        if (cudaStreamSynchronize(stream_) != cudaSuccess) return {model::Status::Error, "hy4flash: the prompt failed"};
        return {};
    }

    model::Result prepare() override {
        std::string e;
        if (!ensure_tier(e)) return {model::Status::Error, "hy4flash: " + e};
        return {};
    }

    ExpertCounts expert_counts() const override {
        const core::ExpertTier* t = tier_.get();
        return t == nullptr ? ExpertCounts{} : ExpertCounts{t->hits(), t->lookups(), t->offloaded(), t->file_lookups()};
    }

    std::string info_words() const override { return tier_ ? "expert_slots=" + std::to_string(tier_->slots()) : std::string(); }

    model::Result copy_logits(int row, float* out) override {
        if (row != last_row_) return {model::Status::Error, "hy4flash: the session keeps the last row's logits only"};
        cudaMemcpy(out, logits_, (size_t) g_.n_vocab * 4, cudaMemcpyDeviceToHost);
        return {};
    }

    model::Result row_logprobs(int row0, int n, const int32_t* picks, int top, float* out) override {
        if (n != 1 || row0 != last_row_) return {model::Status::Error, "hy4flash: the session keeps the last row's logits only"};
        float* d_out = lp_ + bk::row_logprobs_scratch_floats(1, (int) g_.n_vocab);
        int32_t* d_pick = pick_ + 8;
        cudaMemcpyAsync(d_pick, picks, 4, cudaMemcpyHostToDevice, stream_);
        bk::row_logprobs(logits_, 1, (int) g_.n_vocab, d_pick, top, d_out, lp_, stream_);
        cudaMemcpyAsync(out, d_out, bk::kLogprobStride * 4, cudaMemcpyDeviceToHost, stream_);
        return cudaStreamSynchronize(stream_) == cudaSuccess ? model::Result{} : model::Result{model::Status::Error, "hy4flash: logprobs failed"};
    }

private:
    void* alloc(size_t bytes, bool& ok) {
        void* p = nullptr;
        if (!ok || cudaMalloc(&p, std::max<size_t>(bytes, 16)) != cudaSuccess) {
            ok = false;
            return nullptr;
        }
        bufs_.push_back(p);
        return p;
    }

    // y = W x for the Q8_1-quantised activation in `q8`
    void mm(const DevTensor* w, const void* q8, float* y, int n_in, int n_out) {
        bk::native_mmvq(w->type, w->ptr, q8, y, n_in, n_out, 1, stream_);
    }
    void quant(const float* x, void* q8, int n) { bk::native_quantize_q8_1(x, q8, n, 1, stream_); }
    const DevTensor* T(int l, const char* s) const { return m_.tensor(l, s); }
    const float* F(int l, const char* s) const { return (const float*) m_.tensor(l, s)->ptr; }

    MlaArgs mla_args(int heads, float base) const {
        MlaArgs a;
        a.heads = heads;
        a.nope = g_.qk_dim - g_.rope_dim;
        a.rope = g_.rope_dim;
        a.rank = g_.kv_lora;
        a.rope_base = base;
        a.scale = 1.0f / std::sqrt((float) g_.qk_dim);
        a.rms_eps = g_.rms_eps;
        return a;
    }

    bool attention(int l, int64_t pos, std::string& err) {
        const int H = g_.heads[(size_t) l];
        const bool swa = g_.swa[(size_t) l];
        const int E = g_.n_embd, rank = g_.kv_lora;
        const MlaArgs a = mla_args(H, swa ? g_.rope_base_swa : g_.rope_base);
        quant(xn_, q8a_, E);
        mm(T(l, "attn_q_a.weight"), q8a_, qa_, E, g_.q_lora);
        bk::rms_norm_rows(qa_, F(l, "attn_q_a_norm.weight"), qan_, 1, g_.q_lora, g_.rms_eps, stream_);
        quant(qan_, q8b_, g_.q_lora);
        mm(T(l, "attn_q_b.weight"), q8b_, q_, g_.q_lora, H * g_.qk_dim);
        mm(T(l, "attn_kv_a_mqa.weight"), q8a_, kvraw_, E, rank + g_.rope_dim);
        q8_0_heads_gemv(T(l, "attn_k_b.weight")->ptr, q_, g_.qk_dim, H, rank, a.nope, qabs_, stream_);
        const int64_t slot = swa ? pos % g_.window : pos;
        float* cache = cache_[(size_t) l];
        mla_prepare(q_, qabs_, kvraw_, F(l, "attn_kv_a_norm.weight"), a, (int) pos, qcur_, cache + (size_t) slot * kRowWidth, stream_);
        if (swa) {
            const int first = (int) std::max<int64_t>(0, pos + 1 - g_.window);
            mla_attend(qcur_, cache, first, (int) (pos + 1 - first), F(l, "attn_sinks.weight"), a, olat_, stream_, nullptr, g_.window);
        } else {
            const int n = indexer(l, pos);
            if (n < 0) mla_attend(qcur_, cache, 0, (int) pos + 1, F(l, "attn_sinks.weight"), a, olat_, stream_);
            else mla_attend(qcur_, cache, 0, n, F(l, "attn_sinks.weight"), a, olat_, stream_, icells_);
        }
        q8_0_heads_gemv(T(l, "attn_v_b.weight")->ptr, olat_, rank, H, g_.v_dim, rank, vout_, stream_);
        mm(T(l, "attn_gate.weight"), q8a_, gate_, E, H * g_.v_dim);
        mul_sigmoid(gate_, vout_, H * g_.v_dim, stream_);
        quant(vout_, q8c_, H * g_.v_dim);
        mm(T(l, "attn_output.weight"), q8c_, attn_raw_, H * g_.v_dim, E);
        bk::rms_norm_rows(attn_raw_, F(l, "post_attention_norm.weight"), attn_, 1, E, g_.rms_eps, stream_);
        (void) err;
        return true;
    }

    // The indexer's per-position rows and pooled keys of layer `l` for the token at `pos`; with more pools than it keeps, its
    // selection: the cells in icells_ and their count, else -1 (every position is attended).
    int indexer(int l, int64_t pos) {
        const int D = g_.idx_dim, P = g_.idx_pool, E = g_.n_embd;
        IndexerArgs a;
        a.heads = g_.idx_heads;
        a.dim = D;
        a.pool = P;
        a.rope_base = g_.rope_base;
        a.eps = g_.rms_eps;
        mm(T(l, "indexer.attn_k.weight"), q8a_, ik_, E, D);
        bk::rms_norm_rows(ik_, F(l, "indexer.k_norm.weight"), ikn_, 1, D, g_.rms_eps, stream_);
        quant(ikn_, q8d(), D);
        float* vals = ival_[(size_t) l];
        float* gates = igate_[(size_t) l];
        mm(T(l, "indexer_compressor_value.weight"), q8d(), vals + (size_t) pos * D, D, D);
        mm(T(l, "indexer_compressor_gate.weight"), q8d(), gates + (size_t) pos * D, D, D);
        if ((pos + 1) % P == 0) {
            const int64_t p0 = pos + 1 - P;
            idx_pool(vals + (size_t) p0 * D, gates + (size_t) p0 * D, F(l, "indexer_compressor_ape.weight"),
                     F(l, "indexer_compressor_norm.weight"), a, (int) p0, ipool_[(size_t) l] + (size_t) (pos / P) * D, stream_);
        }
        const int keep = g_.idx_top_k / P - (g_.idx_select_tail ? 1 : 0);
        const int pools = (int) (pos / P);   // complete groups before the token's own
        if (pools <= keep) return -1;
        mm(T(l, "indexer.attn_q_b.weight"), q8b_, iq_, g_.q_lora, a.heads * D);
        idx_rope_heads(iq_, a, (int) pos, stream_);
        f32_gemv(xn_, F(l, "indexer.proj.weight"), a.heads, E, 1.0f / std::sqrt((float) (D * a.heads)), iw_, stream_);
        idx_scores(iq_, iw_, ipool_[(size_t) l], pools, a, iscore_, stream_);
        const int tail0 = pools * P, tail_n = (int) (pos - tail0 + 1);
        idx_select(iscore_, pools, keep, P, tail0, tail_n, icells_, stream_);
        return keep * P + tail_n;
    }

    void* q8d() { return q8c_; }   // the indexer's rows are small: the output projection's scratch is free until it runs

    // The Engram step of layer `l` on the streams.
    bool engram(int l, int64_t pos, int32_t token, std::string& err) {
        const EngramInfo& en = m_.pack().engram;
        int e = -1;
        for (size_t k = 0; k < en.layers.size(); ++k)
            if (en.layers[k] == l) e = (int) k;
        if (e < 0) return true;
        const int n_cols = en.n_cols(), W = en.row_values;
        uint32_t ids[16];
        float valid = 1.f;
        const int32_t p1 = pos >= 1 ? hist_[(size_t) pos - 1] : -1, p2 = pos >= 2 ? hist_[(size_t) pos - 2] : -1;
        engram_ids(en, e, token, p1, p2, pos, false, ids, valid);
        uint8_t raw[16 * 96];
        ngram::PleReader& rd = m_.engram_reader();
        const ngram::PleReader::Ticket t = rd.issue(ids, (size_t) n_cols, raw);
        if (!rd.collect(t, err)) return false;
        const float* lookup = (const float*) m_.tensor(l, "engram_lookup_norm.weight")->ptr;
        if (lookup_host_.empty()) lookup_host_.resize((size_t) n_cols * W * (en.layers.size()));
        float* nrm = lookup_host_.data() + (size_t) e * n_cols * W;
        if (!lookup_loaded_[(size_t) e]) {
            cudaMemcpy(nrm, lookup, (size_t) n_cols * W * 4, cudaMemcpyDeviceToHost);
            lookup_loaded_[(size_t) e] = true;
        }
        for (int c = 0; c < n_cols; ++c) {
            float ss = 0.f;
            float* row = h_erows_ + (size_t) c * W;
            for (int d = 0; d < W; ++d) {
                row[d] = e4m3_to_f32(raw[(size_t) c * W + d]);
                ss += row[d] * row[d];
            }
            const float r = 1.0f / std::sqrt(ss / (float) W + 1.1920929e-07f);
            for (int d = 0; d < W; ++d) row[d] = row[d] * r * nrm[(size_t) c * W + d] * valid;
        }
        cudaMemcpyAsync(erows_, h_erows_, (size_t) n_cols * W * 4, cudaMemcpyHostToDevice, stream_);
        quant(erows_, q8a_, n_cols * W);
        mm(T(l, "engram_kv.weight"), q8a_, ekv_, n_cols * W, 5 * g_.n_embd);
        EngramArgs a;
        a.n_embd = g_.n_embd;
        engram_apply(ekv_, F(l, "engram_kv_b.weight"), F(l, "engram_gate.weight"), F(l, "engram_pre_gamma.weight"),
                     F(l, "engram_hc_scale.weight"), valid, a, X_, stream_);
        return true;
    }

    bool ffn(int l, std::string& err) {
        const int E = g_.n_embd;
        quant(xn_, q8a_, E);
        if (l < g_.n_dense) {
            mm(T(l, "ffn_gate.weight"), q8a_, gu_, E, g_.n_ff);
            mm(T(l, "ffn_up.weight"), q8a_, gu_ + g_.n_ff, E, g_.n_ff);
            bk::swiglu_scaled(gu_, gu_ + g_.n_ff, nullptr, nullptr, h_, 1, g_.n_ff, 0, stream_);
            quant(h_, q8b_, g_.n_ff);
            mm(T(l, "ffn_down.weight"), q8b_, mlp_, g_.n_ff, E);
            return true;
        }
        return moe(l, err);
    }

    // The expert tier, opened by the first session while the model starts and shared by the rest: the sessions run one at a
    // time over the same pinned rows.
    bool ensure_tier(std::string& err) {
        if (tier_) return true;
        Hy4Model::SharedTier& shared = m_.shared_tier();
        constexpr unsigned kMapped = cudaHostAllocPortable | cudaHostAllocMapped;
        if (!shared.tier) {
            if (cudaHostAlloc((void**) &shared.h_x, (size_t) g_.n_latent * 4, kMapped) != cudaSuccess ||
                cudaHostAlloc((void**) &shared.h_rows, (size_t) 16 * g_.n_latent * 4, kMapped) != cudaSuccess) {
                err = "cannot allocate the pinned expert staging";
                return false;
            }
            auto t = std::make_shared<core::ExpertTier>(m_.tier_spec());
            core::ExpertTier::Reserve r;
            r.copies = kModuleRoom;
            r.homes = kPromptRoom + kModuleRoom;
            r.beside_homes = 1ull << 30;
            r.modules = kModuleRoom;
            core::LoadPlan* reads = m_.planning();
            if (reads == nullptr) {
                err = "the expert tier opens only while the model starts";
                return false;
            }
            if (!t->open(r, shared.h_rows, *reads, err)) return false;
            shared.tier = std::move(t);
        }
        tier_ = shared.tier;
        h_x_ = shared.h_x;
        h_rows_ = shared.h_rows;
        if (cudaHostGetDevicePointer((void**) &d_rows_, h_rows_, 0) != cudaSuccess) {
            err = "cannot map the expert rows";
            return false;
        }
        return true;
    }

    // The latent MoE of a layer over its normed input: the router, routed_down, the experts the tier places (the
    // card's slots and the second cards' on the GPUs, the rest on the CPU) beside the shared expert, routed_up.
    bool moe(int l, std::string& err) {
        const int E = g_.n_embd, K = g_.n_used, L = g_.n_latent;
        const int tl = l - g_.n_dense;   // the tier's layer
        const ExpertLayer& el = m_.pack().experts[(size_t) l];
        kernels::NativeExpertLayout lay = kernels::native_expert_layout(el.gu_type, el.d_type, L, g_.n_ff_exp);
        if (lay.bytes != el.blob_bytes) {
            err = "layer " + std::to_string(l) + "'s expert blob size does not match the layout";
            return false;
        }
        cudaStream_t st = stream_;
        router_sigmoid(xn_, F(l, "ffn_gate_inp.weight"), F(l, "exp_probs_b.bias"), g_.n_expert, E, K, g_.expert_scale, rlog_, ids_, rw_, st);
        mm(T(l, "ffn_routed_down.weight"), q8a_, xlat_, E, L);
        bk::native_quantize_q8_1(xlat_, ex_xq_, L, 1, st);
        cudaMemcpyAsync(h_x_, xlat_, (size_t) L * 4, cudaMemcpyDeviceToHost, st);
        cudaMemcpyAsync(h_ids_, ids_, (size_t) K * 4, cudaMemcpyDeviceToHost, st);
        cudaMemcpyAsync(h_rw_, rw_, (size_t) K * 4, cudaMemcpyDeviceToHost, st);
        cudaEventRecord(ev_, st);
        // the shared expert on the card while the experts run elsewhere
        mm(T(l, "ffn_gate_shexp.weight"), q8a_, gu_, E, g_.n_ff_exp);
        mm(T(l, "ffn_up_shexp.weight"), q8a_, gu_ + g_.n_ff_exp, E, g_.n_ff_exp);
        bk::swiglu_scaled(gu_, gu_ + g_.n_ff_exp, nullptr, nullptr, h_, 1, g_.n_ff_exp, 0, st);
        quant(h_, q8b_, g_.n_ff_exp);
        mm(T(l, "ffn_down_shexp.weight"), q8b_, sh_, g_.n_ff_exp, E);
        core::PeerExperts* peer = tier_->peer_for(tl);
        if (peer) {
            kernels::NativeExpertLayout limited = lay;
            limited.swiglu_limit = g_.clamp_exp[(size_t) l];
            if (!peer->arm(limited, h_x_, 1, K, core::policy::peer_groups(1), err)) return false;
        }
        tier_->pump(tl, 3);
        if (cudaEventSynchronize(ev_) != cudaSuccess) {
            err = "the router's ids did not arrive";
            return false;
        }
        for (int i = 0; i < K; ++i)
            if (h_ids_[i] < 0 || h_ids_[i] >= g_.n_expert) {
                err = "the router picked expert " + std::to_string(h_ids_[i]);
                return false;
            }
        core::ExpertTier::Lookup lk;
        if (!tier_->lookup(tl, h_ids_, K, lk, err)) return false;
        const int n_fetched = tier_->fetch(tl, lk);
        if (peer) {
            int32_t kind[core::ExpertTier::kMaxK] = {};
            for (int k = 0; k < K; ++k) kind[k] = !lk.peer[k] ? 0 : lk.fetched[k] ? 3 : 2;
            if (!peer->raise(tl, lk.experts, kind, err)) return false;
        }
        bool cpu = false, reading = false;
        for (int k = 0; k < K; ++k) {
            cpu = cpu || (!lk.dev[k] && !lk.peer[k] && !lk.reading[k]);
            reading = reading || lk.reading[k];
        }
        if (cpu) tier_->start_cpu(tl, h_x_, lk, h_rows_);
        // the card's share: the experts its slots hold, one group each, writing the row of their router index; the fetched
        // ones in a second plan, once their copies landed
        const auto launch = [&](bool fetched, int count) {
            ExpertPlan* p = h_plan_;
            int m = 0;
            for (int k = 0; k < K; ++k) {
                if (!lk.dev[k] || lk.fetched[k] != fetched) continue;
                p->ptr[m] = (unsigned long long) lk.dev[k];
                p->start[m] = m;
                p->dst[m] = k;
                p->tok[m] = 0;
                ++m;
            }
            p->start[m] = m;
            p->ng[0] = m;
            cudaMemcpyAsync(ex_plan_, p, sizeof(ExpertPlan), cudaMemcpyHostToDevice, st);
            ExpertPlan* dp = (ExpertPlan*) ex_plan_;
            kernels::native_expert_grouped_clamped(lay, g_.clamp_exp[(size_t) l], dp->ptr, dp->start, dp->ng, dp->dst, dp->tok, m, m,
                                                   ex_xq_, ex_scr_, ex_rows_, st);
            cudaStreamSynchronize(st);   // the plan buffer is reused by the next launch
            (void) count;
        };
        int nhits = 0;
        for (int k = 0; k < K; ++k) nhits += lk.dev[k] && !lk.fetched[k];
        if (nhits) launch(false, nhits);
        if (n_fetched) {
            tier_->await_fetch(st);
            launch(true, n_fetched);
        }
        if (peer && !peer->run_fetched(err)) return false;
        if (cpu) tier_->finish_cpu();
        if (reading) {
            if (!tier_->await_reads(err)) return false;
            tier_->start_cpu(tl, h_x_, lk, h_rows_, /*reading=*/true);
            tier_->finish_cpu();
        }
        if (peer && !peer->finish(nullptr, err)) return false;
        RowPtrs rp{};
        for (int k = 0; k < K; ++k) rp.p[k] = lk.dev[k] ? ex_rows_ + (size_t) k * L : d_rows_ + (size_t) k * L;
        weighted_sum_rows(rp, rw_, K, L, lat_out_, st);
        if (cpu || reading) tier_->promote(tl, lk);
        tier_->mark_gap(st);
        quant(lat_out_, q8c_, L);
        mm(T(l, "ffn_routed_up.weight"), q8c_, moe_out_, L, E);
        add_vectors(moe_out_, sh_, mlp_, E, st);
        return true;
    }

    bool forward(int32_t token, int64_t pos, bool logits, std::string& err) {
        if (pos >= m_.context()) {
            err = "position " + std::to_string(pos) + " is past the context of " + std::to_string(m_.context());
            return false;
        }
        if (token < 0 || token >= g_.n_vocab) {
            err = "token " + std::to_string(token) + " is out of the vocabulary";
            return false;
        }
        if (pos > (int64_t) hist_.size()) {
            err = "position " + std::to_string(pos) + " does not follow the state's " + std::to_string(hist_.size());
            return false;
        }
        hist_.resize((size_t) pos);   // a position before the end drops the rows after it
        if (tier_) tier_->begin_token(1);
        hist_.push_back(token);
        if (lookup_loaded_.empty()) lookup_loaded_.assign(m_.pack().engram.layers.size(), false);
        const int E = g_.n_embd;
        bk::HcPlainArgs hc;
        hc.n_embd = E;
        hc.rms_eps = g_.rms_eps;
        hc.hc_eps = g_.hc_eps;
        hc.magnitude = g_.hc_magnitude;
        if (!m_.embedding_row(token, h_e_)) {
            err = "no embedding row";
            return false;
        }
        cudaMemcpyAsync(e_, h_e_, (size_t) E * 4, cudaMemcpyHostToDevice, stream_);
        fill_streams(e_, E, X_, stream_);
        for (int l = 0; l < g_.n_layers; ++l) {
            if (!engram(l, pos, token, err)) return false;
            bk::hc_mixes_plain(X_, F(l, "hc_attn_fn.weight"), F(l, "hc_attn_scale.weight"), F(l, "hc_attn_base.weight"), 1, 8, hc, hcws_, coef_, stream_);
            bk::hc_pre(X_, coef_, bk::kHcCoef, 1, E, cur_, stream_);
            bk::rms_norm_rows(cur_, F(l, "attn_norm.weight"), xn_, 1, E, g_.rms_eps, stream_);
            if (!attention(l, pos, err)) return false;
            bk::hc_post(attn_, X_, coef_, bk::kHcCoef, 1, E, Xn_, stream_);
            std::swap(X_, Xn_);
            bk::hc_mixes_plain(X_, F(l, "hc_ffn_fn.weight"), F(l, "hc_ffn_scale.weight"), F(l, "hc_ffn_base.weight"), 1, 8, hc, hcws_, coef_, stream_);
            bk::hc_pre(X_, coef_, bk::kHcCoef, 1, E, cur_, stream_);
            bk::rms_norm_rows(cur_, F(l, "ffn_norm.weight"), xn_, 1, E, g_.rms_eps, stream_);
            if (!ffn(l, err)) return false;
            bk::rms_norm_rows(mlp_, F(l, "post_ffw_norm.weight"), mlp2_, 1, E, g_.rms_eps, stream_);
            bk::hc_post(mlp2_, X_, coef_, bk::kHcCoef, 1, E, Xn_, stream_);
            std::swap(X_, Xn_);
            if (!m_.diag_dump().empty()) dump("l_out-" + std::to_string(l), X_, 4 * E);
        }
        if (logits) {
            bk::hc_mixes_plain(X_, (const float*) m_.tensor("output_hc_fn.weight")->ptr, (const float*) m_.tensor("output_hc_scale.weight")->ptr,
                               (const float*) m_.tensor("output_hc_base.weight")->ptr, 1, 4, hc, hcws_, coef_, stream_);
            bk::hc_pre(X_, coef_, bk::hc_plain_coef_stride(4), 1, E, cur_, stream_);
            bk::rms_norm_rows(cur_, (const float*) m_.tensor("output_norm.weight")->ptr, fin_, 1, E, g_.rms_eps, stream_);
            const DevTensor* head = m_.tensor("output.weight");
            bk::bf16_gemv_fp32_mmvf(fin_, (const uint16_t*) head->ptr, logits_, E, g_.n_vocab, stream_);
            if (!m_.diag_dump().empty()) {
                dump("result_norm", fin_, E);
                dump("result_output", logits_, g_.n_vocab);
            }
            last_row_ = 0;
        }
        if (tier_) tier_->pump(-1, core::ExpertTier::kAllPieces);
        const cudaError_t ce = cudaGetLastError();
        if (ce != cudaSuccess) {
            err = std::string("CUDA: ") + cudaGetErrorString(ce);
            return false;
        }
        return true;
    }

    // diagnostic: a tensor of this position appended to <dir>/<name>.f32
    void dump(const std::string& name, const float* d, int64_t n) {
        std::vector<float> h((size_t) n);
        cudaMemcpyAsync(h.data(), d, (size_t) n * 4, cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
        const std::string path = m_.diag_dump() + "/" + name + ".f32";
        std::FILE* f = std::fopen(path.c_str(), "ab");
        if (!f) {
            basalt::log::warn("basalt hy4flash", "--diag-dump: cannot write %s", path.c_str());
            return;
        }
        std::fwrite(h.data(), 4, h.size(), f);
        std::fclose(f);
    }

    bool pick(const model::DecodeRequest& r, int row, int32_t& tok, std::string& err) {
        bk::SamplerParams sp;
        if (r.sampling) sp = *r.sampling;
        else sp.greedy = true;
        if (!r.sampling) sp.temperature = 0.f;
        sp.counter += (uint64_t) row;
        bk::sample_tokens(logits_, 1, (int) g_.n_vocab, nullptr, 0, sp, pick_, stream_);
        if (cudaMemcpyAsync(&tok, pick_, 4, cudaMemcpyDeviceToHost, stream_) != cudaSuccess || cudaStreamSynchronize(stream_) != cudaSuccess) {
            err = "the pick failed";
            return false;
        }
        return true;
    }

    Hy4Model& m_;
    const Geometry& g_;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t ev_ = nullptr;
    std::vector<void*> bufs_, pinned_;
    float *X_ = nullptr, *Xn_ = nullptr, *coef_ = nullptr, *hcws_ = nullptr;
    float *e_ = nullptr, *cur_ = nullptr, *xn_ = nullptr, *attn_ = nullptr, *attn_raw_ = nullptr, *mlp_ = nullptr, *mlp2_ = nullptr, *fin_ = nullptr;
    float *qa_ = nullptr, *qan_ = nullptr, *q_ = nullptr, *kvraw_ = nullptr, *qabs_ = nullptr, *qcur_ = nullptr, *olat_ = nullptr, *vout_ = nullptr, *gate_ = nullptr;
    float *ik_ = nullptr, *ikn_ = nullptr, *iq_ = nullptr, *iw_ = nullptr, *iscore_ = nullptr;
    int32_t* icells_ = nullptr;
    float *gu_ = nullptr, *h_ = nullptr, *rlog_ = nullptr, *rw_ = nullptr;
    int32_t* ids_ = nullptr;
    float *xlat_ = nullptr, *lat_out_ = nullptr, *moe_out_ = nullptr, *sh_ = nullptr, *logits_ = nullptr, *lp_ = nullptr, *erows_ = nullptr, *ekv_ = nullptr;
    int32_t* pick_ = nullptr;
    void *q8a_ = nullptr, *q8b_ = nullptr, *q8c_ = nullptr;
    float *h_x_ = nullptr, *h_rows_ = nullptr, *d_rows_ = nullptr, *h_rw_ = nullptr, *h_erows_ = nullptr, *h_e_ = nullptr;
    int32_t* h_ids_ = nullptr;
    void *ex_xq_ = nullptr, *ex_scr_ = nullptr, *ex_plan_ = nullptr;
    float* ex_rows_ = nullptr;
    ExpertPlan* h_plan_ = nullptr;
    std::shared_ptr<core::ExpertTier> tier_;
    std::vector<float> lookup_host_;
    std::vector<bool> lookup_loaded_;
    std::vector<float*> cache_, ival_, igate_, ipool_;
    std::vector<int32_t> hist_;
    int last_row_ = -1;
};

}  // namespace

std::unique_ptr<model::ModelSession> make_session(Hy4Model& model, std::string& err) {
    auto s = std::make_unique<Session>(model);
    if (!s->init(err)) return nullptr;
    return s;
}

}  // namespace basalt::models::hy4flash
