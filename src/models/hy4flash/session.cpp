// src/models/hy4flash/session.cpp - one conversation of Hy4-flash: its latent caches, indexer rows and pooled keys, and the
// forward of a run of rows that every decode window (up to 8 rows) and every prompt chunk (up to kMaxChunk) goes through.
//
// Per layer: the Engram step (on its layers), the attention's hyper-connection pre-mix and norm, the absorbed MLA over the
// shared latent (a ring on the window layers, the indexer's selection on the others), the gate, the output projection and its
// post-norm, the stream expansion, and the same for the FFN: the dense MLP, or the latent MoE (router, routed_down, the experts
// the tier places in groups of up to 8 rows beside the shared expert, routed_up). A row's result does not depend on the other
// rows of its call: the dense products take 8 rows at most per launch with the single-row kernels' reduction order, the
// attention kernels take a block per row, and the experts are run per row.
#include "basalt/core/conversation_cache.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/core/elastic_array.hpp"
#include "basalt/core/expert_tier.hpp"
#include "basalt/kernels/bf16_gemv.hpp"
#include "basalt/kernels/dense_ops.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/logprobs.hpp"
#include "basalt/kernels/mhc.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/router.hpp"
#include "basalt/kernels/row_norm.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/prefill/moe_mmq.hpp"
#include "basalt/log.hpp"
#include "basalt/models/hy4flash/kernels.hpp"
#include "models/hy4flash/context_policy.hpp"
#include "models/hy4flash/dfly.hpp"   // DFLY HOOK
#include "models/hy4flash/engram.hpp"
#include "models/hy4flash/model.hpp"
#include "models/hy4flash/vision_encoder.hpp"   // [vision]

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <cstring>
#include <memory>
#include <set>
#include <utility>
#include <string>
#include <vector>

namespace basalt::models::hy4flash {
namespace {

namespace bk = basalt::kernels;

constexpr int kRowWidth = 576;                 // a cache row: 512 latent values and 64 rotated
constexpr int kMaxChunk = 16384;               // rows a prompt chunk may carry (the residual streams and the experts' pairs)
constexpr int kAttRows = 2048;                 // rows of one attention step inside a chunk (the rings are sized for it)
constexpr int kStageExperts = 16;              // experts of one prompt launch (a blob each in the staging)
constexpr int kLaunchRows = 16384;              // most (row, expert) pairs of one prompt launch
constexpr int kMaxWindow = 8;                  // rows of a decode window, and of one expert group
constexpr int kRawRing = kAttRows + 16;        // the indexer's rows of the last positions (a multiple of the pool size)
constexpr int64_t kGrowPositions = 16384;      // positions the caches are backed in
constexpr uint64_t kModuleRoom = 7ull << 27;   // kernel modules still to load when the tier opens
constexpr uint64_t kPromptRoom = 5ull << 29;   // the prompt path's workspaces beside the homes
constexpr int kLookupK = core::ExpertTier::kMaxK;

// One group's GPU plan for the grouped expert kernel, uploaded with a single copy.
struct WindowPlan {
    unsigned long long ptr[kLookupK];
    int32_t start[kLookupK + 2];
    int32_t ng[4];
    int32_t dst[kLookupK];
    int32_t tok[kLookupK];
};

// The buffers that scale with the rows of a call: a decode window's, or a prompt chunk's (opened while a prompt runs, with the tier's
// slots lent to them).
struct Bufs {
    int rows = 0, arows = 0;   // the chunk's rows, and the rows of an attention step
    std::vector<void*> dev, host;
    float *X = nullptr, *Xn = nullptr, *coef = nullptr, *hcws = nullptr, *e = nullptr, *cur = nullptr, *xn = nullptr, *attn = nullptr,
          *attn_raw = nullptr, *mlp = nullptr, *mlp2 = nullptr, *qa = nullptr, *qan = nullptr, *q = nullptr, *kvraw = nullptr,
          *qabs = nullptr, *qcur = nullptr, *olat = nullptr, *vout = nullptr, *gate = nullptr, *ik = nullptr, *ikn = nullptr,
          *ivt = nullptr, *igt = nullptr, *iq = nullptr, *iw = nullptr, *gu = nullptr, *h = nullptr, *rlog = nullptr, *rw = nullptr,
          *xlat = nullptr, *lat_out = nullptr, *moe_out = nullptr, *sh = nullptr, *erows = nullptr, *ekv = nullptr, *evalid = nullptr,
          *pair_rows = nullptr, *h_erows = nullptr, *h_e = nullptr, *h_evalid = nullptr;
    int32_t *icells = nullptr, *meta_d = nullptr, *meta_h = nullptr, *ids = nullptr, *pair_map = nullptr, *pair_ident = nullptr,
            *mmq_ident = nullptr;
    void* mmq_xq = nullptr;
    void release() {
        for (void* p : dev) cudaFree(p);
        for (void* p : host) cudaFreeHost(p);
        *this = Bufs();
    }
};

class Session final : public model::ModelSession {
public:
    explicit Session(Hy4Model& m) : m_(m), g_(m.geo()) {}
    ~Session() override {
        dec_.release();
        big_.release();
        for (void* p : bufs_) cudaFree(p);
        if (d_hist_) cudaFree(d_hist_);
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
        (void) E;
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
        fin_ = f(kMaxWindow * E);
        const int64_t pools = ctx / g_.idx_pool + 1;
        iscore_ = f(pools);
        cells_stride_ = g_.idx_top_k + 2 * g_.idx_pool;
        max_heads_ = max_heads;
        logits_ = f((int64_t) kMaxWindow * g_.n_vocab); pick_ = (int32_t*) alloc(4 * 32, ok);
        lp_ = f((int64_t) bk::row_logprobs_scratch_floats(kMaxWindow, (int) g_.n_vocab) + (int64_t) kMaxWindow * (bk::kLogprobStride + 1));
        q8_ = alloc(std::max<size_t>(bk::native_q8_1_bytes(16384, kMaxWindow), 1 << 16), ok);
        mmq_bounds_ = (int32_t*) alloc(2 * 4, ok);
        ex_xq_ = alloc(bk::native_q8_1_bytes(g_.n_latent, kMaxWindow), ok);
        ex_scr_ = alloc(bk::native_expert_scratch_bytes(kLookupK, g_.n_ff_exp), ok);
        ex_rows_ = f((int64_t) kLookupK * g_.n_latent);
        ex_plan_ = alloc(sizeof(WindowPlan), ok);
        // the prompt path's experts: two staged launches (blobs, rows and bounds), the pairs' outputs, and the quantised rows
        {
            namespace mmq = basalt::prefill::mmq;
            size_t blob = 0;
            for (const ExpertLayer& el : m_.pack().experts) blob = std::max<size_t>(blob, el.blob_bytes);
            // MMQ steps from expert to expert in whole blocks of each matrix type: the stride is a multiple of every type size
            size_t unit = 16;
            for (const ExpertLayer& el : m_.pack().experts) {
                if (!el.present) continue;
                for (int t : {el.gu_type, el.d_type}) unit = std::lcm(unit, ggml_type_size((ggml_type) t));
            }
            stride_ = (blob + unit - 1) / unit * unit;
            for (Stage& b : stage_) {
                b.blobs = (uint8_t*) alloc(stride_ * kStageExperts, ok);
                b.tok = (int32_t*) alloc((size_t) kLaunchRows * 4, ok);
                b.bounds = (int32_t*) alloc((kStageExperts + 1) * 4, ok);
                if (cudaEventCreateWithFlags(&b.copied, cudaEventDisableTiming) != cudaSuccess || cudaEventCreateWithFlags(&b.done, cudaEventDisableTiming) != cudaSuccess) ok = false;
            }
            if (cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking) != cudaSuccess) ok = false;
            xq1_ = alloc(mmq::q8_bytes(kLaunchRows, g_.n_latent), ok);
            xq2_ = alloc(mmq::q8_bytes(kLaunchRows, g_.n_ff_exp), ok);
        }
        rowptr_d_ = (const float**) alloc(kLookupK * sizeof(void*), ok);
        h_rowptr_ = (const float**) pin(kLookupK * 2);
        h_rw_ = pin(kMaxWindow * 16); h_ids_ = (int32_t*) pin(kMaxWindow * 16);
        rope_dev_ = (int32_t*) alloc((size_t) ctx * 12, ok);   // [vision] the 3-D rotary position of every cell
        if (cudaHostAlloc((void**) &h_plan_, sizeof(WindowPlan), cudaHostAllocPortable) != cudaSuccess) ok = false;
        else pinned_.push_back(h_plan_);
        cache_.assign((size_t) g_.n_layers, nullptr);
        ival_.assign((size_t) g_.n_layers, nullptr);
        igate_.assign((size_t) g_.n_layers, nullptr);
        ipool_.assign((size_t) g_.n_layers, nullptr);
        ea_.resize((size_t) g_.n_layers);
        pea_.resize((size_t) g_.n_layers);
        for (int l = 0; ok && l < g_.n_layers; ++l) {
            if (g_.swa[(size_t) l]) {
                cache_[(size_t) l] = (uint8_t*) alloc((size_t) swa_ring() * kRowWidth * elem(), ok);
            } else {
                ea_[(size_t) l] = std::make_unique<core::ElasticArray>();
                pea_[(size_t) l] = std::make_unique<core::ElasticArray>();
                if (!ea_[(size_t) l]->open((uint64_t) ctx * sparse_row_bytes(), err) || !pea_[(size_t) l]->open((uint64_t) pools * g_.idx_dim * 2, err)) return false;
                cache_[(size_t) l] = ea_[(size_t) l]->data();
                ipool_[(size_t) l] = (uint16_t*) pea_[(size_t) l]->data();
                ival_[(size_t) l] = f((int64_t) kRawRing * g_.idx_dim);
                igate_[(size_t) l] = f((int64_t) kRawRing * g_.idx_dim);
            }
        }
        if (!ok) {
            err = "out of device memory for the session";
            return false;
        }
        if (!make_bufs(dec_, kMaxWindow, err)) return false;
        use(dec_);
        if (m_.dfly() && fp8()) {
            err = "--spec reads the target's BF16 latent rows: it cannot run with --kv fp8 (use --kv bf16 or --spec 0)";
            return false;
        }
        // DFLY HOOK: the drafter's tap and scratch (nothing without a drafter in the pack)
        if (!dfly_.init(m_.dfly(), (int) (g_.idx_top_k + g_.idx_pool), ctx, m_.dfly() ? (const uint16_t*) m_.tensor("output.weight")->ptr : nullptr,
                        [this](int32_t t, float* o) { return m_.embedding_row(t, o); }, err))
            return false;
        return true;
    }

    // The buffers for `R` rows; false (with nothing left allocated) when the card has no room.
    bool make_bufs(Bufs& b, int R, std::string& err) {
        const int64_t E = g_.n_embd;
        bool ok = true;
        auto f = [&](int64_t n) {
            void* p = nullptr;
            if (!ok || cudaMalloc(&p, std::max<size_t>((size_t) n * 4, 16)) != cudaSuccess) {
                ok = false;
                return (float*) nullptr;
            }
            b.dev.push_back(p);
            return (float*) p;
        };
        auto pin = [&](int64_t n) {
            void* p = nullptr;
            if (!ok || cudaMallocHost(&p, (size_t) n * 4) != cudaSuccess) {
                ok = false;
                return (float*) nullptr;
            }
            b.host.push_back(p);
            return (float*) p;
        };
        const int A = std::min(R, kAttRows);
        b.rows = R;
        b.arows = A;
        b.X = f((int64_t) R * 4 * E); b.Xn = f((int64_t) R * 4 * E); b.coef = f((int64_t) R * bk::kHcCoef);
        b.hcws = f((int64_t) bk::hc_plain_workspace_floats(R));
        b.e = f((int64_t) R * E); b.cur = f((int64_t) R * E); b.xn = f((int64_t) R * E); b.attn = f((int64_t) R * E);
        b.attn_raw = f((int64_t) A * E); b.mlp = f((int64_t) R * E); b.mlp2 = f((int64_t) R * E);
        b.qa = f((int64_t) A * g_.q_lora); b.qan = f((int64_t) A * g_.q_lora); b.q = f((int64_t) A * max_heads_ * g_.qk_dim);
        b.kvraw = f((int64_t) A * kRowWidth); b.qabs = f((int64_t) A * max_heads_ * g_.kv_lora);
        b.qcur = f((int64_t) A * max_heads_ * kRowWidth); b.olat = f((int64_t) A * max_heads_ * g_.kv_lora);
        b.vout = f((int64_t) A * max_heads_ * g_.v_dim); b.gate = f((int64_t) A * max_heads_ * g_.v_dim);
        b.ik = f((int64_t) A * g_.idx_dim); b.ikn = f((int64_t) A * g_.idx_dim); b.ivt = f((int64_t) A * g_.idx_dim);
        b.igt = f((int64_t) A * g_.idx_dim); b.iq = f((int64_t) A * g_.idx_heads * g_.idx_dim); b.iw = f((int64_t) A * g_.idx_heads);
        b.icells = (int32_t*) f((int64_t) A * cells_stride_);
        b.meta_d = (int32_t*) f(3 * (int64_t) A);
        b.meta_h = (int32_t*) pin((int64_t) g_.n_layers * 3 * A);
        // gate/up products of the dense layer (A rows of n_ff), of the shared expert (R rows of the expert width) and of a prompt
        // launch (kLaunchRows rows of twice the expert width); the SwiGLU outputs likewise
        const int64_t gu_n = std::max<int64_t>({2 * (int64_t) A * g_.n_ff, 2 * (int64_t) R * g_.n_ff_exp, 2 * (int64_t) kLaunchRows * g_.n_ff_exp});
        const int64_t h_n = std::max<int64_t>({(int64_t) A * g_.n_ff, (int64_t) R * g_.n_ff_exp, (int64_t) kLaunchRows * g_.n_ff_exp});
        b.gu = f(gu_n); b.h = f(h_n);
        b.rlog = f((int64_t) R * g_.n_expert); b.rw = f((int64_t) R * 16); b.ids = (int32_t*) f((int64_t) R * 16);
        b.xlat = f((int64_t) R * g_.n_latent); b.lat_out = f((int64_t) R * g_.n_latent); b.moe_out = f((int64_t) R * E); b.sh = f((int64_t) R * E);
        b.erows = f((int64_t) A * g_.n_embd); b.ekv = f((int64_t) A * 5 * E); b.evalid = f(A);
        b.pair_rows = f((int64_t) R * g_.n_used * g_.n_latent);
        b.pair_map = (int32_t*) f((int64_t) R * g_.n_used); b.pair_ident = (int32_t*) f((int64_t) R * g_.n_used);
        b.mmq_ident = (int32_t*) f(A);
        b.h_erows = pin((int64_t) A * g_.n_embd); b.h_e = pin((int64_t) R * E); b.h_evalid = pin(A);
        if (ok) {
            void* q = nullptr;
            if (cudaMalloc(&q, basalt::prefill::mmq::q8_bytes(A, 16384)) != cudaSuccess) ok = false;
            else {
                b.dev.push_back(q);
                b.mmq_xq = q;
            }
        }
        if (!ok) {
            b.release();
            err = "out of device memory for " + std::to_string(R) + " rows";
            return false;
        }
        if (basalt::prefill::mmq::built()) {
            basalt::prefill::mmq::iota(b.mmq_ident, A, nullptr);
            basalt::prefill::mmq::iota(b.pair_ident, (int64_t) R * g_.n_used, nullptr);
        }
        cudaDeviceSynchronize();
        return true;
    }

    // The session's row buffers are `b`'s from here on.
    void use(const Bufs& b) {
        rows_ = b.rows;
        arows_ = b.arows;
        X_ = b.X; Xn_ = b.Xn; coef_ = b.coef; hcws_ = b.hcws; e_ = b.e; cur_ = b.cur; xn_ = b.xn; attn_ = b.attn; attn_raw_ = b.attn_raw;
        mlp_ = b.mlp; mlp2_ = b.mlp2; qa_ = b.qa; qan_ = b.qan; q_ = b.q; kvraw_ = b.kvraw; qabs_ = b.qabs; qcur_ = b.qcur;
        olat_ = b.olat; vout_ = b.vout; gate_ = b.gate; ik_ = b.ik; ikn_ = b.ikn; ivt_ = b.ivt; igt_ = b.igt; iq_ = b.iq; iw_ = b.iw;
        icells_ = b.icells; meta_d_ = b.meta_d; meta_h_ = b.meta_h; gu_ = b.gu; h_ = b.h; rlog_ = b.rlog; rw_ = b.rw; ids_ = b.ids;
        xlat_ = b.xlat; lat_out_ = b.lat_out; moe_out_ = b.moe_out; sh_ = b.sh; erows_ = b.erows; ekv_ = b.ekv; evalid_ = b.evalid;
        pair_rows_ = b.pair_rows; pair_map_ = b.pair_map; pair_ident_ = b.pair_ident; mmq_ident_ = b.mmq_ident; mmq_xq_ = b.mmq_xq;
        h_erows_ = b.h_erows; h_e_ = b.h_e; h_evalid_ = b.h_evalid;
    }

    // A prompt of `total` rows: the tier lends its slots and the row buffers of a chunk (up to kMaxChunk rows) are made.
    bool open_prompt(int total, std::string& err) {
        if (big_.rows) {
            use(big_);
            return true;
        }
        cudaStreamSynchronize(stream_);
        if (tier_) tier_->lend_all();
        int want = kMaxWindow;
        while (want < total && want < kMaxChunk) want *= 2;
        for (int R = std::min(want, kMaxChunk); R >= kMaxWindow; R /= 2)
            if (make_bufs(big_, R, err)) {
                use(big_);
                return true;
            }
        if (tier_) {
            std::string e;
            tier_->reclaim(e);
            tier_->grow();
        }
        return false;
    }

    // The prompt's buffers go back; the tier takes its slots again.
    void release_prompt() {
        if (!big_.rows) return;
        cudaStreamSynchronize(stream_);
        basalt::log::info("basalt hy4flash", "prompt time: %.0f ms queued ahead of the MoE (attention, dense), %.0f ms in the MoE, %.0f ms in the Engram's host step", t_pre_ms_, t_moe_ms_, t_eng_ms_);
        t_pre_ms_ = t_moe_ms_ = t_eng_ms_ = 0;
        use(dec_);
        big_.release();
        if (tier_) {
            std::string e;
            if (!tier_->reclaim(e)) basalt::log::warn("basalt hy4flash", "the expert tier stays small: %s", e.c_str());
            tier_->grow();
        }
    }

    void end_prompt() override { release_prompt(); }

    // DFLY HOOK: --spec sets the window (DeepSeek's rule: 0 or 1 is plain decode), and the drafter's calls (dfly.hpp)
    int max_window() const override { return std::clamp(m_.spec(), 1, kMaxWindow); }
    model::Result draft_begin(int32_t anchor, int64_t pos, int n) override {
        std::vector<uint16_t*> rows;   // the drafter reads BF16 rows (--spec with --kv fp8 is refused at the start)
        for (uint8_t* p : cache_) rows.push_back((uint16_t*) p);
        return dfly_.draft_begin(anchor, pos, n, rows, stream_);
    }
    model::Result draft_end(int32_t* ids, float* conf, int max_n, int& n) override { return dfly_.draft_end(ids, conf, max_n, n, stream_); }

    model::Result decode_window(const model::DecodeRequest& r) override {
        if (r.n_tokens < 1) return {model::Status::Error, "hy4flash: an empty window"};
        std::string err;
        release_prompt();
        dfly_.begin_window(r.pos);   // DFLY HOOK
        for (int t0 = 0; t0 < r.n_tokens; t0 += kMaxWindow) {
            const int n = std::min(kMaxWindow, r.n_tokens - t0);
            dfly_.run(t0);   // DFLY HOOK
            const bool last = t0 + n == r.n_tokens;
            const bool want = r.row_tokens || (last && (r.next_token || r.logits));
            const int head = !want ? 0 : r.row_tokens ? n : 1;
            if (!run_rows(r.tokens + t0, n, r.pos + t0, head, err)) return {model::Status::Error, "hy4flash: " + err};
            dfly_.run(-1);   // DFLY HOOK
            if (!want) continue;
            if (!pick(r, t0, head, err)) return {model::Status::Error, "hy4flash: " + err};
            if (r.row_tokens)
                for (int i = 0; i < n; ++i) r.row_tokens[t0 + i] = picks_[(size_t) i];
            if (last && r.next_token) *r.next_token = picks_[(size_t) head - 1];
            if (last && r.logits) cudaMemcpyAsync(r.logits, logits_ + (size_t) (head - 1) * g_.n_vocab, (size_t) g_.n_vocab * 4, cudaMemcpyDeviceToHost, stream_);
        }
        if (cudaStreamSynchronize(stream_) != cudaSuccess) return {model::Status::Error, "hy4flash: the window failed"};
        return {};
    }

    model::Result prefill_chunk(const model::PrefillRequest& r) override {
        std::string err;
        dfly_.invalidate();   // DFLY HOOK
        if (!open_prompt(std::max(r.total_tokens, r.n_tokens), err)) return {model::Status::Error, "hy4flash: " + err};
        const int chunk = m_.diag_rows() > 0 ? std::min(rows_, m_.diag_rows()) : rows_;
        for (int t0 = 0; t0 < r.n_tokens; t0 += chunk) {
            if (r.stop && r.stop()) break;
            const int n = std::min(chunk, r.n_tokens - t0);
            if (!run_rows(r.tokens + t0, n, r.pos + t0, 0, err)) return {model::Status::Error, "hy4flash: " + err};
            if (r.progress) {
                cudaStreamSynchronize(stream_);
                r.progress(t0 + n);
            }
        }
        if (cudaStreamSynchronize(stream_) != cudaSuccess) return {model::Status::Error, "hy4flash: the prompt failed"};
        if (!r.more) release_prompt();
        return {};
    }

    // ---- [vision] images (vision.hpp, vision_encoder.hpp) ------------------------------------------------------------------
    // The Engram of an image row: llama.cpp's hy-v4-flash treats an embedding position as a token that cannot be hashed, so its own
    // contribution is masked (valid 0) and, to the rows after it, it is a padded history entry. The history therefore holds -1 for
    // an image row, which engram_ids reads as "no n-gram" on both counts.
    model::Result encode_image(const uint8_t* data, size_t bytes, std::vector<float>& rows, int& n_rows, model::ImageGrid& grid) override {
        vision::Image img;
        std::string err;
        if (!vision::decode_image(data, bytes, img, err) || !vis_.encode(m_.pack().vision, img, rows, n_rows, grid, err))
            return {model::Status::Error, "hy4flash: " + err};
        if (n_rows <= 0 || rows.size() != (size_t) n_rows * (size_t) g_.n_embd)
            return {model::Status::Error, "hy4flash: the projector's rows are not the model's width"};
        return {};
    }
    void images_done() override { vis_.release(); }
    // The rows of a picture block (newline rows included) at positions pos..: chunks of the row pass with the rows as the embeddings.
    model::Result prefill_embeddings(const float* rows, int n_rows, int64_t pos, model::ImageGrid grid) override {
        if (grid.nx <= 0 || (int64_t) grid.nx * grid.ny != n_rows) return {model::Status::Error, "hy4flash: an image block needs its grid"};
        sync_rope(pos);
        img_start_ = rope_.next();
        img_nx_ = grid.nx;
        std::string err;
        if (!open_prompt(n_rows, err)) return {model::Status::Error, "hy4flash: " + err};
        const std::vector<int32_t> dead((size_t) rows_, -1);
        for (int at = 0; at < n_rows; at += rows_) {
            const int n = std::min(rows_, n_rows - at);
            emb_rows_ = rows + (size_t) at * (size_t) g_.n_embd;
            img_j_ = at;
            const bool ok = run_rows(dead.data(), n, pos + at, 0, err);
            emb_rows_ = nullptr;
            if (!ok) return {model::Status::Error, "hy4flash: " + err};
        }
        if (cudaStreamSynchronize(stream_) != cudaSuccess) return {model::Status::Error, "hy4flash: the image rows failed"};
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

    // SLEEP / WAKE: the tier's card slots go back to the card (every expert keeps its RAM copy and its slot is remembered), and
    // return at the wake. The latent caches, the rings and the drafter's rows are conversation state and stay.
    model::Result set_low_memory(bool low, int64_t& expert_slots) override {
        std::string e;
        if (!tier_) return {model::Status::Error, "hy4flash: the expert tier is not open"};
        cudaStreamSynchronize(stream_);
        if (low) {
            tier_->lend_all();
        } else {
            if (!tier_->reclaim(e)) return {model::Status::Error, "hy4flash: " + e};
            tier_->grow();
        }
        expert_slots = tier_->slots();
        return {};
    }

    model::Result set_vram_reserve(int64_t mib, int64_t& expert_slots) override {
        std::string e;
        if (!tier_) return {model::Status::Error, "hy4flash: the expert tier is not open"};
        cudaStreamSynchronize(stream_);
        if (!tier_->replan(mib, e)) return {model::Status::Error, "hy4flash: " + e};
        expert_slots = tier_->slots();
        return {};
    }

    // ---- conversations: the latent rows and pooled keys by position (save_cells / restore_cells), the rings by checkpoint
    // (running_state). A ring holds the last swa_ring() (window layers) or kRawRing (indexer rows) positions, so a checkpoint
    // at `pos` is intact while the session has not run more than kMaxRows rows past it.

    int cell_format() const { return fp8() ? 1 : f32() ? 2 : 0; }
    size_t ring_bytes() const { return (size_t) swa_ring() * kRowWidth * elem(); }
    size_t raw_bytes() const { return (size_t) kRawRing * g_.idx_dim * 4; }

    size_t running_bytes() const override {
        size_t n = 0;
        for (int l = 0; l < g_.n_layers; ++l) n += g_.swa[(size_t) l] ? ring_bytes() : 2 * raw_bytes();
        return n;
    }

    model::Result running_state(int64_t pos, bool save, core::RunningState& st) override {
        if (save && (int64_t) hist_.size() - pos > kAttRows)
            return {model::Status::Error, "hy4flash: the rings have moved on more than " + std::to_string(kAttRows) + " positions past the checkpoint"};
        if (save && pos > (int64_t) hist_.size()) return {model::Status::Error, "hy4flash: a checkpoint past the session's position"};
        for (int l = 0; l < g_.n_layers; ++l) {
            if (g_.swa[(size_t) l]) {
                st.regions.push_back({cache_[(size_t) l], ring_bytes()});
            } else {
                st.regions.push_back({ival_[(size_t) l], raw_bytes()});
                st.regions.push_back({igate_[(size_t) l], raw_bytes()});
            }
        }
        st.stream = stream_;
        return {};
    }

    model::Result running_restored(int64_t pos, const uint8_t*) override {
        if ((int64_t) hist_.size() > pos) hist_.resize((size_t) pos);
        dfly_.invalidate();   // DFLY HOOK
        return {};
    }

    size_t cells_bytes(int64_t positions) const override {
        int sparse = 0;
        for (int l = 0; l < g_.n_layers; ++l) sparse += !g_.swa[(size_t) l];
        return ctxp::cache_bytes(positions, sparse, fp8()) * (f32() ? 2 : 1);
    }

    model::Result save_cells(std::vector<core::ConversationKv>& kv, int64_t positions) override {
        if (positions > backed_) return {model::Status::Error, "hy4flash: positions that were never run"};
        cudaStreamSynchronize(stream_);
        kv.clear();
        const size_t rb = (size_t) sparse_row_bytes(), pools = (size_t) (positions / g_.idx_pool + 1), pb = (size_t) g_.idx_dim * 2;
        for (int l = 0; l < g_.n_layers; ++l) {
            if (g_.swa[(size_t) l]) continue;
            core::ConversationKv k;
            k.format = cell_format();
            k.page_size = g_.idx_pool;
            k.cells = positions;
            k.pooled_rows = (int64_t) pools;
            k.idx_dim = g_.idx_dim;
            k.k.resize_for_overwrite((size_t) positions * rb);
            k.pooled.resize_for_overwrite(pools * pb);
            const auto down = [&](const uint8_t* dev, core::ConversationBuffer& to) {
                return to.visit(0, to.size(), [&](uint8_t* p, size_t n, size_t at) { return cudaMemcpy(p, dev + at, n, cudaMemcpyDeviceToHost) == cudaSuccess; });
            };
            if (!down(cache_[(size_t) l], k.k) || !down((const uint8_t*) ipool_[(size_t) l], k.pooled))
                return {model::Status::Error, "hy4flash: reading a cache back failed"};
            kv.push_back(std::move(k));
        }
        return {};
    }

    model::Result restore_cells(const std::vector<core::ConversationKv>& kv, int64_t positions) override {
        std::string e;
        if (!ensure_positions(positions, e)) return {model::Status::Error, "hy4flash: " + e};
        cudaStreamSynchronize(stream_);
        size_t i = 0;
        for (int l = 0; l < g_.n_layers; ++l) {
            if (g_.swa[(size_t) l]) continue;
            if (i >= kv.size() || kv[i].format != cell_format() || kv[i].page_size != g_.idx_pool || kv[i].cells != positions)
                return {model::Status::Error, "hy4flash: a parked conversation has other caches"};
            const auto up = [&](uint8_t* dev, const core::ConversationBuffer& from) {
                return from.visit(0, from.size(), [&](const uint8_t* p, size_t n, size_t at) { return cudaMemcpy(dev + at, p, n, cudaMemcpyHostToDevice) == cudaSuccess; });
            };
            if (!up(cache_[(size_t) l], kv[i].k) || !up((uint8_t*) ipool_[(size_t) l], kv[i].pooled))
                return {model::Status::Error, "hy4flash: writing a cache back failed"};
            ++i;
        }
        dfly_.invalidate();   // DFLY HOOK
        return {};
    }

    // the tokens below `n` come back with a restored conversation: the Engram's history
    model::Result set_history(const int32_t* tokens, int64_t n) override {
        hist_.assign(tokens, tokens + n);
        return {};
    }

    model::Result set_logit_bias(const float* bias) override {
        bias_on_ = bias != nullptr;
        if (!bias_on_) return {};
        std::string e;
        if (!device_bias(d_bias_, bias, e)) return {model::Status::Error, "hy4flash: " + e};
        return {};
    }

    // a draw of one row again, with the tokens the mask bans
    model::Result resample_row(int row, const float* bias, int32_t& token) override {
        std::string e;
        if (row < 0 || row >= last_rows_) return {model::Status::Error, "hy4flash: the row is not in the last window"};
        if (!device_bias(d_mask_, bias, e)) return {model::Status::Error, "hy4flash: " + e};
        bk::SamplerParams sp = sampling_;
        sp.logit_bias = d_mask_;
        sp.counter += (uint64_t) row;
        bk::sample_tokens(logits_ + (size_t) row * g_.n_vocab, 1, (int) g_.n_vocab, history_len_ > 0 ? d_hist_ + (size_t) row * history_len_ : nullptr,
                          history_len_, sp, pick_ + 24, stream_);
        int32_t t = 0;
        if (cudaMemcpyAsync(&t, pick_ + 24, 4, cudaMemcpyDeviceToHost, stream_) != cudaSuccess || cudaStreamSynchronize(stream_) != cudaSuccess)
            return {model::Status::Error, "hy4flash: the masked draw failed"};
        token = t;
        return {};
    }

    model::Result copy_logits(int row, float* out) override {
        if (row < 0 || row >= last_rows_) return {model::Status::Error, "hy4flash: the session keeps the last window's logits only"};
        cudaMemcpy(out, logits_ + (size_t) row * g_.n_vocab, (size_t) g_.n_vocab * 4, cudaMemcpyDeviceToHost);
        return {};
    }

    model::Result row_logprobs(int row0, int n, const int32_t* picks, int top, float* out) override {
        if (row0 < 0 || n < 1 || row0 + n > last_rows_) return {model::Status::Error, "hy4flash: the session keeps the last window's logits only"};
        float* d_out = lp_ + bk::row_logprobs_scratch_floats(n, (int) g_.n_vocab);
        int32_t* d_pick = pick_ + 16;
        cudaMemcpyAsync(d_pick, picks, (size_t) n * 4, cudaMemcpyHostToDevice, stream_);
        bk::row_logprobs(logits_ + (size_t) row0 * g_.n_vocab, n, (int) g_.n_vocab, d_pick, top, d_out, lp_, stream_);
        cudaMemcpyAsync(out, d_out, (size_t) n * bk::kLogprobStride * 4, cudaMemcpyDeviceToHost, stream_);
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

    int swa_ring() const { return g_.window + kAttRows; }
    bool fp8() const { return m_.cache_fp8(); }
    bool f32() const { return m_.cache_f32(); }
    int elem() const { return f32() ? 4 : 2; }
    int sparse_row_bytes() const { return fp8() ? kFp8RowBytes : kRowWidth * elem(); }

    // y[r] = W x[r] for R rows, 8 at a time: a row's product does not depend on the rows beside it
    void gemm(const DevTensor* w, const float* x, float* y, int n_in, int n_out, int R) {
        namespace mmq = basalt::prefill::mmq;
        if (R > kMaxWindow && mmq::built() && mmq::fits(w->type, n_out)) {   // a chunk: the tensor-core MMQ product, arows_ rows a time
            if (loaded_.insert({w->type, n_out}).second) mmq::load(w->type, n_out);
            for (int r0 = 0; r0 < R; r0 += arows_) {
                const int nb = std::min(arows_, R - r0);
                const int32_t b[2] = {0, nb};
                cudaMemcpyAsync(mmq_bounds_, b, sizeof b, cudaMemcpyHostToDevice, stream_);
                mmq::quantize(x + (size_t) r0 * n_in, nullptr, mmq_xq_, w->type, n_in, n_in, nb, stream_);
                mmq::Product p;
                p.w = w->ptr;
                p.type = w->type;
                p.w_rows = n_out;
                p.w_cols = n_in;
                p.expert_bytes = mmq::matrix_bytes(w->type, n_out, n_in);
                p.n = 1;
                p.xq = mmq_xq_;
                p.bounds = mmq_bounds_;
                p.ids = mmq_ident_;
                p.total_rows = nb;
                p.max_rows = nb;
                p.dst = y + (size_t) r0 * n_out;
                p.ld_dst = n_out;
                mmq_.run(p, stream_);
            }
            return;
        }
        for (int c0 = 0; c0 < R; c0 += kMaxWindow) {
            const int nc = std::min(kMaxWindow, R - c0);
            bk::native_quantize_q8_1(x + (size_t) c0 * n_in, q8_, n_in, nc, stream_);
            bk::native_mmvq(w->type, w->ptr, q8_, y + (size_t) c0 * n_out, n_in, n_out, nc, stream_);
        }
    }
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
        a.cell_rope = rope_on_ ? rope_dev_ : nullptr;   // [vision]
        a.sec_h = kMropeH;
        a.sec_w = kMropeW;
        return a;
    }

    // Attention of layer `l` over R rows at pos0.. of the normed input xn_; the post-normed result goes to attn_. Steps of arows_ rows run
    // one after the other (each writes its rows' cache cells before it attends, so a row sees the rows before it), over the streams
    // offset to the step.
    bool attention(int l, int64_t pos0, int R, std::string& err) {
        float *xn0 = xn_, *at0 = attn_;
        const int E = g_.n_embd;
        bool ok = true;
        for (int off = 0; ok && off < R; off += arows_) {
            xn_ = xn0 + (size_t) off * E;
            attn_ = at0 + (size_t) off * E;
            ok = attention_step(l, pos0 + off, std::min(arows_, R - off), err);
        }
        xn_ = xn0;
        attn_ = at0;
        return ok;
    }

    bool attention_step(int l, int64_t pos0, int R, std::string& err) {
        const int H = g_.heads[(size_t) l];
        const bool swa = g_.swa[(size_t) l];
        const int E = g_.n_embd, rank = g_.kv_lora;
        const MlaArgs a = mla_args(H, swa ? g_.rope_base_swa : g_.rope_base);
        gemm(T(l, "attn_q_a.weight"), xn_, qa_, E, g_.q_lora, R);
        bk::rms_norm_rows(qa_, F(l, "attn_q_a_norm.weight"), qan_, R, g_.q_lora, g_.rms_eps, stream_);
        gemm(T(l, "attn_q_b.weight"), qan_, q_, g_.q_lora, H * g_.qk_dim, R);
        gemm(T(l, "attn_kv_a_mqa.weight"), xn_, kvraw_, E, rank + g_.rope_dim, R);
        q8_0_heads_gemv(T(l, "attn_k_b.weight")->ptr, q_, g_.qk_dim, H * g_.qk_dim, H, rank, a.nope, R, qabs_, stream_);
        uint8_t* cache = cache_[(size_t) l];
        const int ring = swa ? swa_ring() : 0;
        const bool f8 = !swa && fp8();
        const float* kvn = F(l, "attn_kv_a_norm.weight");
        if (f8) mla_prepare(q_, qabs_, kvraw_, kvn, a, (int) pos0, R, ring, qcur_, cache, stream_);
        else if (f32()) mla_prepare(q_, qabs_, kvraw_, kvn, a, (int) pos0, R, ring, qcur_, (float*) cache, stream_);
        else mla_prepare(q_, qabs_, kvraw_, kvn, a, (int) pos0, R, ring, qcur_, (uint16_t*) cache, stream_);
        if (diag(l) && f32() && !swa) {   // the inputs of the attention, for the reference's tensors (fp32 caches only)
            cudaStreamSynchronize(stream_);
            dump("q_resid-" + std::to_string(l), qan_, (int64_t) R * g_.q_lora);
            std::vector<float> rows((size_t) R * kRowWidth), lat((size_t) R * rank), pe((size_t) R * g_.rope_dim), qp((size_t) R * H * g_.rope_dim),
                               qc((size_t) R * H * kRowWidth);
            cudaMemcpy(rows.data(), (const float*) cache + (size_t) pos0 * kRowWidth, rows.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(qc.data(), qcur_, qc.size() * 4, cudaMemcpyDeviceToHost);
            for (int t = 0; t < R; ++t) {
                for (int c = 0; c < rank; ++c) lat[(size_t) t * rank + c] = rows[(size_t) t * kRowWidth + c];
                for (int c = 0; c < g_.rope_dim; ++c) pe[(size_t) t * g_.rope_dim + c] = rows[(size_t) t * kRowWidth + rank + c];
                for (int h = 0; h < H; ++h)
                    for (int c = 0; c < g_.rope_dim; ++c) qp[((size_t) t * H + h) * g_.rope_dim + c] = qc[((size_t) t * H + h) * kRowWidth + rank + c];
            }
            auto put = [&](const std::string& n, const std::vector<float>& v) {
                std::FILE* f = std::fopen((m_.diag_dump() + "/" + n + ".f32").c_str(), "ab");
                if (f) { std::fwrite(v.data(), 4, v.size(), f); std::fclose(f); }
            };
            put("kv_cmpr-" + std::to_string(l), lat);
            put("k_pe-" + std::to_string(l), pe);
            put("q_pe-" + std::to_string(l), qp);
        }
        // the cells each row attends
        int32_t* meta = meta_h_ + (size_t) l * 3 * arows_;   // firsts, counts, fill
        int32_t *firsts = meta, *counts = meta + arows_, *fill = meta + 2 * arows_;
        bool selecting = false;
        std::vector<int> pools_of(R, 0);
        if (swa) {
            for (int t = 0; t < R; ++t) {
                const int64_t pos = pos0 + t, first = std::max<int64_t>(0, pos + 1 - g_.window);
                firsts[t] = (int32_t) first;
                counts[t] = (int32_t) (pos + 1 - first);
                fill[t] = 0;
            }
        } else {
            const int keep = g_.idx_top_k / g_.idx_pool - (g_.idx_select_tail ? 1 : 0);
            for (int t = 0; t < R; ++t) {
                const int64_t pos = pos0 + t;
                pools_of[(size_t) t] = (int) (pos / g_.idx_pool);   // complete groups before the row's own
                firsts[t] = 0;
                if (pools_of[(size_t) t] > keep) {
                    selecting = true;
                    counts[t] = keep * g_.idx_pool + (int32_t) (pos - (int64_t) pools_of[(size_t) t] * g_.idx_pool + 1);
                    fill[t] = 0;
                } else {
                    counts[t] = (int32_t) (pos + 1);
                    fill[t] = counts[t];
                }
            }
            indexer(l, pos0, R, selecting, pools_of, err);
            for (int t = 0; t < R; ++t) {   // DFLY HOOK: each row's selection, for the drafter
                const bool sel = pools_of[(size_t) t] > keep;
                dfly_.selection(l, t, sel ? icells_ + (size_t) t * cells_stride_ : nullptr, sel ? counts[t] : -1, stream_);
            }
        }
        cudaMemcpyAsync(meta_d_, meta, (size_t) 3 * arows_ * 4, cudaMemcpyHostToDevice, stream_);
        RowSpans rs;
        rs.firsts = meta_d_;
        rs.counts = meta_d_ + arows_;
        if (!swa) {
            iota_cells(icells_, cells_stride_, meta_d_ + 2 * arows_, R, stream_);
            rs.cells = icells_;
            rs.cells_stride = cells_stride_;
        }
        if (f8) mla_attend(qcur_, (const uint8_t*) cache, rs, R, F(l, "attn_sinks.weight"), a, ring, olat_, stream_);
        else if (f32()) mla_attend(qcur_, (const float*) cache, rs, R, F(l, "attn_sinks.weight"), a, ring, olat_, stream_);
        else mla_attend(qcur_, (const uint16_t*) cache, rs, R, F(l, "attn_sinks.weight"), a, ring, olat_, stream_);
        q8_0_heads_gemv(T(l, "attn_v_b.weight")->ptr, olat_, rank, H * rank, H, g_.v_dim, rank, R, vout_, stream_);
        if (diag(l)) dump("attn_kqv-" + std::to_string(l), vout_, (int64_t) R * H * g_.v_dim);
        gemm(T(l, "attn_gate.weight"), xn_, gate_, E, H * g_.v_dim, R);
        mul_sigmoid(gate_, vout_, R * H * g_.v_dim, stream_);
        if (diag(l)) dump("attn_gated-" + std::to_string(l), vout_, (int64_t) R * H * g_.v_dim);
        gemm(T(l, "attn_output.weight"), vout_, attn_raw_, H * g_.v_dim, E, R);
        bk::rms_norm_rows(attn_raw_, F(l, "post_attention_norm.weight"), attn_, R, E, g_.rms_eps, stream_);
        if (diag(l)) dump("attn_out-" + std::to_string(l), attn_, (int64_t) R * E);
        return true;
    }

    // The indexer of sparse layer `l` for R rows at pos0..: the rows of the keys and the pooled keys of the groups they
    // complete; for the rows with more pools than it keeps, its selection into icells_.
    void indexer(int l, int64_t pos0, int R, bool selecting, const std::vector<int>& pools_of, std::string&) {
        const int D = g_.idx_dim, P = g_.idx_pool, E = g_.n_embd;
        IndexerArgs a;
        a.heads = g_.idx_heads;
        a.dim = D;
        a.pool = P;
        a.rope_base = g_.rope_base;
        a.eps = g_.rms_eps;
        a.cell_rope = rope_on_ ? rope_dev_ : nullptr;   // [vision]
        a.sec_h = kMropeH;
        a.sec_w = kMropeW;
        gemm(T(l, "indexer.attn_k.weight"), xn_, ik_, E, D, R);
        bk::rms_norm_rows(ik_, F(l, "indexer.k_norm.weight"), ikn_, R, D, g_.rms_eps, stream_);
        gemm(T(l, "indexer_compressor_value.weight"), ikn_, ivt_, D, D, R);
        gemm(T(l, "indexer_compressor_gate.weight"), ikn_, igt_, D, D, R);
        // into the ring of the last positions, at cell pos % kRawRing (one wrap at most)
        const int64_t s0 = pos0 % kRawRing;
        const int n1 = (int) std::min<int64_t>(R, kRawRing - s0);
        for (auto [src, dst] : {std::pair<const float*, float*>{ivt_, ival_[(size_t) l]}, {igt_, igate_[(size_t) l]}}) {
            cudaMemcpyAsync(dst + (size_t) s0 * D, src, (size_t) n1 * D * 4, cudaMemcpyDeviceToDevice, stream_);
            if (n1 < R) cudaMemcpyAsync(dst, src + (size_t) n1 * D, (size_t) (R - n1) * D * 4, cudaMemcpyDeviceToDevice, stream_);
        }
        // the groups whose last position is among the rows
        const int64_t last = pos0 + R - 1;
        const int64_t g0 = pos0 >= 3 ? (pos0 - 3 + P - 1) / P : 0, g1 = last >= 3 ? (last - 3) / P : -1;
        if (g1 >= g0) idx_pool(ival_[(size_t) l], igate_[(size_t) l], F(l, "indexer_compressor_ape.weight"), F(l, "indexer_compressor_norm.weight"),
                               a, kRawRing, (int) (g0 * P), (int) (g1 - g0 + 1), ipool_[(size_t) l], stream_);
        if (!selecting) return;
        const int keep = g_.idx_top_k / P - (g_.idx_select_tail ? 1 : 0);
        gemm(T(l, "indexer.attn_q_b.weight"), qan_, iq_, g_.q_lora, a.heads * D, R);
        idx_rope_heads(iq_, a, (int) pos0, R, stream_);
        f32_gemm_rows(xn_, F(l, "indexer.proj.weight"), a.heads, E, 1.0f / std::sqrt((float) (D * a.heads)), R, iw_, stream_);
        for (int t = 0; t < R; ++t) {
            const int pools = pools_of[(size_t) t];
            if (pools <= keep) continue;
            const int64_t pos = pos0 + t;
            const int tail0 = pools * P, tail_n = (int) (pos - tail0 + 1);
            idx_scores(iq_ + (size_t) t * a.heads * D, iw_ + (size_t) t * a.heads, ipool_[(size_t) l], pools, a, iscore_, stream_);
            idx_select(iscore_, pools, keep, P, tail0, tail_n, icells_ + (size_t) t * cells_stride_, stream_);
        }
    }

    // The Engram step of layer `l` on the streams of R rows, arows_ rows a step.
    bool engram(int l, int64_t pos0, int R, const int32_t* tokens, std::string& err) {
        float* x0 = X_;
        bool ok = true;
        for (int off = 0; ok && off < R; off += arows_) {
            X_ = x0 + (size_t) off * 4 * g_.n_embd;
            ok = engram_step(l, pos0 + off, std::min(arows_, R - off), tokens + off, err);
        }
        X_ = x0;
        return ok;
    }

    bool engram_step(int l, int64_t pos0, int R, const int32_t* tokens, std::string& err) {
        const double te = now_ms();
        struct Lap {
            double& acc;
            double t;
            ~Lap() { acc += now_ms() - t; }
        } lap{t_eng_ms_, te};
        const EngramInfo& en = m_.pack().engram;
        int e = -1;
        for (size_t k = 0; k < en.layers.size(); ++k)
            if (en.layers[k] == l) e = (int) k;
        if (e < 0) return true;
        const int n_cols = en.n_cols(), W = en.row_values;
        std::vector<uint32_t> ids((size_t) R * n_cols);
        for (int t = 0; t < R; ++t) {
            const int64_t pos = pos0 + t;
            const int32_t p1 = pos >= 1 ? hist_[(size_t) pos - 1] : -1, p2 = pos >= 2 ? hist_[(size_t) pos - 2] : -1;
            float valid = 1.f;
            engram_ids(en, e, tokens[t], p1, p2, pos, false, ids.data() + (size_t) t * n_cols, valid);
            h_evalid_[t] = valid;
        }
        std::vector<uint8_t> raw((size_t) R * n_cols * W);
        ngram::PleReader& rd = m_.engram_reader();
        const ngram::PleReader::Ticket tk = rd.issue(ids.data(), ids.size(), raw.data());
        if (!rd.collect(tk, err)) return false;
        if (lookup_host_.empty()) lookup_host_.resize((size_t) n_cols * W * en.layers.size());
        float* nrm = lookup_host_.data() + (size_t) e * n_cols * W;
        if (lookup_loaded_.empty()) lookup_loaded_.assign(en.layers.size(), false);
        if (!lookup_loaded_[(size_t) e]) {
            cudaMemcpy(nrm, T(l, "engram_lookup_norm.weight")->ptr, (size_t) n_cols * W * 4, cudaMemcpyDeviceToHost);
            lookup_loaded_[(size_t) e] = true;
        }
        for (int t = 0; t < R; ++t) {
            const float valid = h_evalid_[t];
            for (int c = 0; c < n_cols; ++c) {
                float* row = h_erows_ + ((size_t) t * n_cols + c) * W;
                const uint8_t* src = raw.data() + ((size_t) t * n_cols + c) * W;
                float ss = 0.f;
                for (int d = 0; d < W; ++d) {
                    row[d] = e4m3_to_f32(src[d]);
                    ss += row[d] * row[d];
                }
                const float r = 1.0f / std::sqrt(ss / (float) W + 1.1920929e-07f);
                for (int d = 0; d < W; ++d) row[d] = row[d] * r * nrm[(size_t) c * W + d] * valid;
            }
        }
        cudaMemcpyAsync(erows_, h_erows_, (size_t) R * n_cols * W * 4, cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(evalid_, h_evalid_, (size_t) R * 4, cudaMemcpyHostToDevice, stream_);
        gemm(T(l, "engram_kv.weight"), erows_, ekv_, n_cols * W, 5 * g_.n_embd, R);
        EngramArgs a;
        a.n_embd = g_.n_embd;
        engram_apply(ekv_, F(l, "engram_kv_b.weight"), F(l, "engram_gate.weight"), F(l, "engram_pre_gamma.weight"),
                     F(l, "engram_hc_scale.weight"), evalid_, R, a, X_, stream_);
        return true;
    }

    bool ffn(int l, int R, std::string& err) {
        const int E = g_.n_embd;
        if (l < g_.n_dense) {
            float *xn0 = xn_, *ml0 = mlp_;
            for (int off = 0; off < R; off += arows_) {
                const int n = std::min(arows_, R - off);
                xn_ = xn0 + (size_t) off * E;
                mlp_ = ml0 + (size_t) off * E;
                float *gate = gu_, *up = gu_ + (size_t) n * g_.n_ff;
                gemm(T(l, "ffn_gate.weight"), xn_, gate, E, g_.n_ff, n);
                gemm(T(l, "ffn_up.weight"), xn_, up, E, g_.n_ff, n);
                bk::swiglu_scaled(gate, up, nullptr, nullptr, h_, n, g_.n_ff, g_.n_ff, stream_);
                gemm(T(l, "ffn_down.weight"), h_, mlp_, g_.n_ff, E, n);
            }
            xn_ = xn0;
            mlp_ = ml0;
            return true;
        }
        return moe(l, R, err);
    }

    // The expert tier, opened by the first session while the model starts and shared by the rest: the sessions run one at a
    // time over the same pinned rows.
    bool ensure_tier(std::string& err) {
        if (tier_) return true;
        Hy4Model::SharedTier& shared = m_.shared_tier();
        constexpr unsigned kMapped = cudaHostAllocPortable | cudaHostAllocMapped;
        if (!shared.tier) {
            if (cudaHostAlloc((void**) &shared.h_x, (size_t) kMaxWindow * g_.n_latent * 4, kMapped) != cudaSuccess ||
                cudaHostAlloc((void**) &shared.h_rows, (size_t) kLookupK * g_.n_latent * 4, kMapped) != cudaSuccess) {
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

    // The latent MoE of a layer over its R normed rows: the router and routed_down and the shared expert for all rows, the
    // routed experts in groups of up to 8 rows (the tier places them: the card's slots and the second cards' on the GPUs, the rest
    // on the CPU), then routed_up.
    bool moe(int l, int R, std::string& err) {
        const int E = g_.n_embd, K = g_.n_used, L = g_.n_latent;
        const int tl = l - g_.n_dense;   // the tier's layer
        const ExpertLayer& el = m_.pack().experts[(size_t) l];
        kernels::NativeExpertLayout lay = kernels::native_expert_layout(el.gu_type, el.d_type, L, g_.n_ff_exp);
        if (lay.bytes != el.blob_bytes) {
            err = "layer " + std::to_string(l) + "'s expert blob size does not match the layout";
            return false;
        }
        cudaStream_t st = stream_;
        bk::RouterArgs ra;
        ra.n_expert = g_.n_expert;
        ra.n_embd = E;
        ra.k = K;
        ra.weight_scale = g_.expert_scale;
        ra.gate = bk::RouterGate::Sigmoid;
        bk::router(xn_, F(l, "ffn_gate_inp.weight"), F(l, "exp_probs_b.bias"), R, ra, rlog_, ids_, rw_, st);
        gemm(T(l, "ffn_routed_down.weight"), xn_, xlat_, E, L, R);
        // the shared expert on the card while the experts run elsewhere
        float *sg = gu_, *su = gu_ + (size_t) R * g_.n_ff_exp;
        gemm(T(l, "ffn_gate_shexp.weight"), xn_, sg, E, g_.n_ff_exp, R);
        gemm(T(l, "ffn_up_shexp.weight"), xn_, su, E, g_.n_ff_exp, R);
        bk::swiglu_scaled(sg, su, nullptr, nullptr, h_, R, g_.n_ff_exp, g_.n_ff_exp, st);
        gemm(T(l, "ffn_down_shexp.weight"), h_, sh_, g_.n_ff_exp, E, R);
        if (R > kMaxWindow && basalt::prefill::mmq::built()) {
            if (!moe_prompt(l, tl, R, lay, err)) return false;
        } else {
            for (int g0 = 0; g0 < R; g0 += kMaxWindow)
                if (!moe_group(l, tl, lay, g0, std::min(kMaxWindow, R - g0), err)) return false;
        }
        if (diag(l)) dump("moe_lat-" + std::to_string(l), lat_out_, (int64_t) R * L);
        if (diag(l)) dump("xlat-" + std::to_string(l), xlat_, (int64_t) R * L);
        gemm(T(l, "ffn_routed_up.weight"), lat_out_, moe_out_, L, E, R);
        add_vectors(moe_out_, sh_, mlp_, R * E, st);
        return true;
    }

    // The routed experts of a prompt chunk, by expert: each used expert's blob is staged on the card (a copy from its slot, its
    // pinned RAM or the second card), its rows (the pairs routed to it, with their activations quantised for MMQ) go through the
    // gate/up product, the clamped SwiGLU and the down product, and a pair's output row is kept for the weighted sum.
    bool moe_prompt(int l, int tl, int R, const kernels::NativeExpertLayout& lay, std::string& err) {
        namespace mmq = basalt::prefill::mmq;
        const int K = g_.n_used, L = g_.n_latent, F = g_.n_ff_exp;
        const ExpertLayer& el = m_.pack().experts[(size_t) l];
        if (!mmq::fits(el.gu_type, 2 * F) || !mmq::fits(el.d_type, L)) {
            err = "the prompt MMQ path does not cover this layer's expert formats";
            return false;
        }
        if (loaded_.insert({el.gu_type, 2 * F}).second) mmq::load(el.gu_type, 2 * F);
        if (loaded_.insert({el.d_type, L}).second) mmq::load(el.d_type, L);
        cudaStream_t st = stream_;
        std::vector<int32_t> hid((size_t) R * K);
        const double t0 = now_ms();
        cudaStreamSynchronize(st);
        cudaMemcpy(hid.data(), ids_, hid.size() * 4, cudaMemcpyDeviceToHost);
        const double t1 = now_ms();
        t_pre_ms_ += t1 - t0;
        std::vector<std::vector<int32_t>> by((size_t) g_.n_expert);   // pair = t * K + slot
        for (int p = 0; p < R * K; ++p) {
            if (hid[(size_t) p] < 0 || hid[(size_t) p] >= g_.n_expert) {
                err = "the router returned expert " + std::to_string(hid[(size_t) p]);
                return false;
            }
            by[(size_t) hid[(size_t) p]].push_back(p);
        }
        std::vector<int> used;
        for (int e = 0; e < g_.n_expert; ++e)
            if (!by[(size_t) e].empty()) used.push_back(e);
        std::vector<int32_t> map_h((size_t) R * K);
        int64_t pos = 0;
        for (size_t b0 = 0; b0 < used.size();) {
            int nb = 0;
            int64_t rows = 0, max_rows = 0;
            while (b0 + (size_t) nb < used.size() && nb < kStageExperts) {
                const int64_t c = (int64_t) by[(size_t) used[b0 + (size_t) nb]].size();
                if (nb > 0 && rows + c > kLaunchRows) break;
                rows += c;
                max_rows = std::max(max_rows, c);
                ++nb;
            }
            Stage& B = stage_[launch_++ & 1];
            cudaStreamWaitEvent(copy_, B.done, 0);   // the device buffers of the launch before last are free
            for (int q = 0; q < nb; ++q) {
                const int e = used[b0 + (size_t) q];
                uint8_t* dst = B.blobs + stride_ * (size_t) q;
                const core::ExpertTier::Source src = tier_->locate(tl, e);
                if (src.dev) {
                    cudaMemcpyAsync(dst, src.dev, el.blob_bytes, cudaMemcpyDeviceToDevice, copy_);
                } else if (src.pinned) {
                    cudaMemcpyAsync(dst, src.pinned, el.blob_bytes, cudaMemcpyHostToDevice, copy_);
                } else if (src.peer) {
                    core::PeerExperts* pe = tier_->peer_for(tl);
                    cudaMemcpyPeerAsync(dst, m_.device(), src.peer, pe->device(), el.blob_bytes, copy_);
                } else {
                    err = "expert " + std::to_string(e) + " of layer " + std::to_string(l) + " is not in RAM (a pack larger than RAM is not run yet)";
                    return false;
                }
            }
            cudaEventRecord(B.copied, copy_);
            std::vector<int32_t> tok, bounds{0};
            for (int q = 0; q < nb; ++q) {
                for (int32_t p : by[(size_t) used[b0 + (size_t) q]]) {
                    map_h[(size_t) p] = (int32_t) (pos + (int64_t) tok.size());
                    tok.push_back(p / K);
                }
                bounds.push_back((int32_t) tok.size());
            }
            cudaMemcpyAsync(B.tok, tok.data(), tok.size() * 4, cudaMemcpyHostToDevice, st);
            cudaMemcpyAsync(B.bounds, bounds.data(), bounds.size() * 4, cudaMemcpyHostToDevice, st);
            cudaStreamWaitEvent(st, B.copied, 0);
            mmq::quantize(xlat_, B.tok, xq1_, el.gu_type, L, L, rows, st);
            mmq::Product p;
            p.w = B.blobs;
            p.type = el.gu_type;
            p.w_rows = 2 * (int64_t) F;
            p.w_cols = L;
            p.expert_bytes = stride_;
            p.n = nb;
            p.xq = xq1_;
            p.bounds = B.bounds;
            p.ids = pair_ident_;
            p.total_rows = rows;
            p.max_rows = max_rows;
            p.dst = gu_;
            p.ld_dst = p.w_rows;
            mmq_.run(p, st);
            swiglu_split_clamped(gu_, h_, rows, F, g_.clamp_exp[(size_t) l], st);
            mmq::quantize(h_, nullptr, xq2_, el.d_type, F, F, rows, st);
            mmq::Product d;
            d.w = B.blobs + lay.down_off;
            d.type = el.d_type;
            d.w_rows = L;
            d.w_cols = F;
            d.expert_bytes = stride_;
            d.n = nb;
            d.xq = xq2_;
            d.bounds = B.bounds;
            d.ids = pair_ident_ + pos;
            d.total_rows = rows;
            d.max_rows = max_rows;
            d.dst = pair_rows_;
            d.ld_dst = L;
            mmq_.run(d, st);
            cudaEventRecord(B.done, st);
            b0 += (size_t) nb;
            pos += rows;
        }
        cudaMemcpyAsync(pair_map_, map_h.data(), (size_t) R * K * 4, cudaMemcpyHostToDevice, st);
        weighted_sum_tokens(pair_rows_, pair_map_, rw_, R, K, L, lat_out_, st);
        cudaStreamSynchronize(st);   // the host's map is read until the copy lands
        t_moe_ms_ += now_ms() - t1;
        return true;
    }

    // The routed experts of rows g0..g0+G of a layer: the tier is asked once per distinct expert of the group; each card runs
    // what it holds over every row that routed to it, the CPU does the same for the rest.
    bool moe_group(int l, int tl, const kernels::NativeExpertLayout& lay, int g0, int G, std::string& err) {
        const int K = g_.n_used, L = g_.n_latent;
        cudaStream_t st = stream_;
        cudaMemcpyAsync(h_x_, xlat_ + (size_t) g0 * L, (size_t) G * L * 4, cudaMemcpyDeviceToHost, st);
        cudaMemcpyAsync(h_ids_, ids_ + (size_t) g0 * K, (size_t) G * K * 4, cudaMemcpyDeviceToHost, st);
        cudaMemcpyAsync(h_rw_, rw_ + (size_t) g0 * K, (size_t) G * K * 4, cudaMemcpyDeviceToHost, st);
        bk::native_quantize_q8_1(xlat_ + (size_t) g0 * L, ex_xq_, L, G, st);
        cudaEventRecord(ev_, st);
        core::PeerExperts* peer = tier_->peer_for(tl);
        tier_->pump(tl, 2 + G);
        if (cudaEventSynchronize(ev_) != cudaSuccess) {
            err = "the router's ids did not arrive";
            return false;
        }
        if (peer) {
            kernels::NativeExpertLayout limited = lay;
            limited.swiglu_limit = g_.clamp_exp[(size_t) l];
            if (!peer->arm(limited, h_x_, G, K, core::policy::peer_groups(G), err)) return false;
        }
        int32_t uniq[kLookupK];
        int uidx[kLookupK], nu = 0;
        core::ExpertTier::Use use[kLookupK];
        for (int t = 0; t < G; ++t)
            for (int k = 0; k < K; ++k) {
                const int32_t e = h_ids_[t * K + k];
                if (e < 0 || e >= g_.n_expert) {
                    err = "the router picked expert " + std::to_string(e);
                    return false;
                }
                int u = 0;
                while (u < nu && uniq[u] != e) ++u;
                if (u == nu) {
                    uniq[nu++] = e;
                    use[u] = core::ExpertTier::Use{};
                }
                core::ExpertTier::Use& us = use[u];
                us.tok[us.nt] = t;
                us.row[us.nt] = t * K + k;
                ++us.nt;
                uidx[t * K + k] = u;
            }
        core::ExpertTier::Lookup lk;
        if (!tier_->lookup(tl, uniq, nu, lk, err)) return false;
        const int n_fetched = tier_->fetch(tl, lk);
        if (peer) {
            int32_t kind[kLookupK] = {};
            for (int r = 0; r < G * K; ++r) kind[r] = !lk.peer[uidx[r]] ? 0 : lk.fetched[uidx[r]] ? 3 : 2;
            if (!peer->raise(tl, h_ids_, kind, err)) return false;
        }
        bool cpu = false, reading = false;
        for (int i = 0; i < nu; ++i) {
            cpu = cpu || (!lk.dev[i] && !lk.peer[i] && !lk.reading[i]);
            reading = reading || lk.reading[i];
        }
        if (cpu) tier_->start_cpu_window(tl, h_x_, lk, use, h_rows_);
        const auto launch = [&](bool fetched) {
            WindowPlan* p = h_plan_;
            int ng = 0, ne = 0;
            for (int i = 0; i < nu; ++i) {
                if (!lk.dev[i] || lk.fetched[i] != fetched) continue;
                p->ptr[ng] = (unsigned long long) lk.dev[i];
                p->start[ng] = ne;
                for (int j = 0; j < use[i].nt; ++j) {
                    p->dst[ne] = use[i].row[j];
                    p->tok[ne] = use[i].tok[j];
                    ++ne;
                }
                ++ng;
            }
            p->start[ng] = ne;
            p->ng[0] = ng;
            if (!ng) return;
            cudaMemcpyAsync(ex_plan_, p, sizeof(WindowPlan), cudaMemcpyHostToDevice, st);
            WindowPlan* dp = (WindowPlan*) ex_plan_;
            kernels::native_expert_grouped_clamped(lay, g_.clamp_exp[(size_t) l], dp->ptr, dp->start, dp->ng, dp->dst, dp->tok, ng, ne,
                                                   ex_xq_, ex_scr_, ex_rows_, st);
            cudaStreamSynchronize(st);   // the plan buffer is reused by the next launch
        };
        launch(false);
        if (n_fetched) {
            tier_->await_fetch(st);
            launch(true);
        }
        if (peer && !peer->run_fetched(err)) return false;
        if (cpu) tier_->finish_cpu();
        if (reading) {
            if (!tier_->await_reads(err)) return false;
            tier_->start_cpu_window(tl, h_x_, lk, use, h_rows_, /*reading=*/true);
            tier_->finish_cpu();
        }
        if (peer && !peer->finish(nullptr, err)) return false;
        for (int r = 0; r < G * K; ++r) h_rowptr_[r] = lk.dev[uidx[r]] ? ex_rows_ + (size_t) r * L : d_rows_ + (size_t) r * L;
        cudaMemcpyAsync(rowptr_d_, h_rowptr_, (size_t) G * K * sizeof(void*), cudaMemcpyHostToDevice, st);
        weighted_sum_window(rowptr_d_, rw_ + (size_t) g0 * K, G, K, L, lat_out_ + (size_t) g0 * L, st);
        if (cpu || reading) tier_->promote(tl, lk);
        tier_->mark_gap(st);
        cudaStreamSynchronize(st);   // the row pointer table and the staged rows are reused by the next group
        return true;
    }

    // R rows at pos0..: the whole forward; the logits of the last `head_rows` rows go to logits_.
    bool run_rows(const int32_t* tokens, int R, int64_t pos0, int head_rows, std::string& err) {
        if (pos0 + R > m_.context()) {
            err = "position " + std::to_string(pos0 + R - 1) + " is past the context of " + std::to_string(m_.context());
            return false;
        }
        for (int t = 0; t < R; ++t)
            if ((tokens[t] < 0 && !emb_rows_) || tokens[t] >= g_.n_vocab) {   // [vision] -1: an embedding row
                err = "token " + std::to_string(tokens[t]) + " is out of the vocabulary";
                return false;
            }
        if (pos0 > (int64_t) hist_.size()) {
            err = "position " + std::to_string(pos0) + " does not follow the state's " + std::to_string(hist_.size());
            return false;
        }
        hist_.resize((size_t) pos0);   // a position before the end drops the rows after it
        hist_.insert(hist_.end(), tokens, tokens + R);
        if (!ensure_positions(pos0 + R, err)) return false;
        place_rope(pos0, R);   // [vision]
        if (tier_) tier_->begin_token(std::min(R, kMaxWindow));
        const int E = g_.n_embd;
        bk::HcPlainArgs hc;
        hc.n_embd = E;
        hc.rms_eps = g_.rms_eps;
        hc.hc_eps = g_.hc_eps;
        hc.magnitude = g_.hc_magnitude;
        for (int t = 0; t < R; ++t)
            if (emb_rows_) {   // [vision] an image row stands in for the token's embedding
                std::memcpy(h_e_ + (size_t) t * E, emb_rows_ + (size_t) t * E, (size_t) E * 4);
            } else if (!m_.embedding_row(tokens[t], h_e_ + (size_t) t * E)) {
                err = "no embedding row";
                return false;
            }
        cudaMemcpyAsync(e_, h_e_, (size_t) R * E * 4, cudaMemcpyHostToDevice, stream_);
        fill_streams(e_, E, R, X_, stream_);
        if (!m_.diag_dump().empty()) dump("hc_init", X_, (int64_t) R * 4 * E);
        for (int l = 0; l < g_.n_layers; ++l) {
            if (!engram(l, pos0, R, tokens, err)) return false;
            if (diag(l) && !g_.swa.empty() && l >= 1) dump("engram_out-" + std::to_string(l), X_, (int64_t) R * 4 * E);
            bk::hc_mixes_plain(X_, F(l, "hc_attn_fn.weight"), F(l, "hc_attn_scale.weight"), F(l, "hc_attn_base.weight"), R, 8, hc, hcws_, coef_, stream_);
            bk::hc_pre_rms_norm(X_, coef_, bk::kHcCoef, F(l, "attn_norm.weight"), R, E, g_.rms_eps, cur_, xn_, stream_);
            if (diag(l)) dump("attn_norm-" + std::to_string(l), xn_, (int64_t) R * E);
            if (!attention(l, pos0, R, err)) return false;
            bk::hc_post(attn_, X_, coef_, bk::kHcCoef, R, E, Xn_, stream_);
            std::swap(X_, Xn_);
            if (diag(l)) dump("hc_attn_out-" + std::to_string(l), X_, (int64_t) R * 4 * E);
            bk::hc_mixes_plain(X_, F(l, "hc_ffn_fn.weight"), F(l, "hc_ffn_scale.weight"), F(l, "hc_ffn_base.weight"), R, 8, hc, hcws_, coef_, stream_);
            bk::hc_pre_rms_norm(X_, coef_, bk::kHcCoef, F(l, "ffn_norm.weight"), R, E, g_.rms_eps, cur_, xn_, stream_);
            if (diag(l)) dump("ffn_norm-" + std::to_string(l), xn_, (int64_t) R * E);
            if (!ffn(l, R, err)) return false;
            bk::rms_norm_rows(mlp_, F(l, "post_ffw_norm.weight"), mlp2_, R, E, g_.rms_eps, stream_);
            if (diag(l)) dump("ffn_out-" + std::to_string(l), mlp2_, (int64_t) R * E);
            bk::hc_post(mlp2_, X_, coef_, bk::kHcCoef, R, E, Xn_, stream_);
            std::swap(X_, Xn_);
            if (!m_.diag_dump().empty()) dump("l_out-" + std::to_string(l), X_, (int64_t) R * 4 * E);
        }
        if (tier_) tier_->pump(-1, core::ExpertTier::kAllPieces);
        if (head_rows > 0) {
            const float* x = X_ + (size_t) (R - head_rows) * 4 * E;
            bk::hc_mixes_plain(x, (const float*) m_.tensor("output_hc_fn.weight")->ptr, (const float*) m_.tensor("output_hc_scale.weight")->ptr,
                               (const float*) m_.tensor("output_hc_base.weight")->ptr, head_rows, 4, hc, hcws_, coef_, stream_);
            bk::hc_pre(x, coef_, bk::hc_plain_coef_stride(4), head_rows, E, cur_, stream_);
            bk::rms_norm_rows(cur_, (const float*) m_.tensor("output_norm.weight")->ptr, fin_, head_rows, E, g_.rms_eps, stream_);
            dfly_.hidden(R - head_rows, head_rows, fin_, stream_);   // DFLY HOOK
            const DevTensor* head = m_.tensor("output.weight");
            bk::bf16_gemv_fp32_mmvf_multi(fin_, E, (const uint16_t*) head->ptr, logits_, g_.n_vocab, E, g_.n_vocab, head_rows, stream_);
            if (!m_.diag_dump().empty()) {
                dump("result_norm", fin_, (int64_t) head_rows * E);
                dump("result_output", logits_, (int64_t) head_rows * g_.n_vocab);
            }
            last_rows_ = head_rows;
        }
        const cudaError_t ce = cudaGetLastError();
        if (ce != cudaSuccess) {
            err = std::string("CUDA: ") + cudaGetErrorString(ce);
            return false;
        }
        return true;
    }

    // Backs the sparse layers' caches for `positions` positions, in steps of kGrowPositions.
    bool ensure_positions(int64_t positions, std::string& err) {
        if (positions <= backed_) return true;
        const int64_t target = std::min<int64_t>(m_.context(), (positions + kGrowPositions - 1) / kGrowPositions * kGrowPositions);
        for (int l = 0; l < g_.n_layers; ++l) {
            if (g_.swa[(size_t) l]) continue;
            if (!ea_[(size_t) l]->ensure((uint64_t) target * sparse_row_bytes(), 256ull << 20, err) ||
                !pea_[(size_t) l]->ensure((uint64_t) (target / g_.idx_pool + 1) * g_.idx_dim * 2, 256ull << 20, err)) {
                err = "the latent caches cannot grow to " + std::to_string(target) + " positions: " + err;
                return false;
            }
        }
        backed_ = target;
        return true;
    }

    // [vision] The rotary positions of the R cells about to be filled (text, or the rows of the picture block being fed), kept per
    // cell on the host and, once a picture is in the session, on the device where the kernels read them (MlaArgs::cell_rope).
    void sync_rope(int64_t pos) {
        rope_.rewind(pos);
        while (rope_.size() < pos) rope_.push_text();   // cells the session never placed (a restored history) count as text
    }
    void place_rope(int64_t pos0, int R) {
        sync_rope(pos0);
        const bool was_on = rope_on_;
        for (int t = 0; t < R; ++t) {
            if (emb_rows_) { rope_.push_row(img_start_, img_nx_, img_j_ + t); rope_on_ = true; }
            else rope_.push_text();
        }
        if (!rope_on_) return;
        std::vector<int32_t> tab;
        const int64_t from = was_on ? pos0 : 0;
        for (int64_t c = from; c < pos0 + R; ++c) tab.insert(tab.end(), {rope_.at(c).t, rope_.at(c).h, rope_.at(c).w});
        cudaMemcpyAsync(rope_dev_ + from * 3, tab.data(), tab.size() * 4, cudaMemcpyHostToDevice, stream_);
        cudaStreamSynchronize(stream_);   // the host table is a temporary
    }

    // diagnostic: a layer whose inner tensors are dumped (the first three)
    bool diag(int l) const { return !m_.diag_dump().empty() && l < 3; }

    // diagnostic: a tensor of this call appended to <dir>/<name>.f32
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

    bool device_bias(float*& dev, const float* host, std::string& err) {
        const size_t bytes = (size_t) g_.n_vocab * sizeof(float);
        if (dev == nullptr) {
            void* v = nullptr;
            if (cudaMalloc(&v, bytes) != cudaSuccess) { err = "out of device memory for a logit bias"; return false; }
            bufs_.push_back(v);
            dev = (float*) v;
        }
        if (cudaMemcpyAsync(dev, host, bytes, cudaMemcpyHostToDevice, stream_) != cudaSuccess) { err = "the logit bias upload failed"; return false; }
        return true;
    }

    // The picks of the last `rows` rows of logits_ (row t of the window is draw counter + t0 + t).
    bool pick(const model::DecodeRequest& r, int t0, int rows, std::string& err) {
        bk::SamplerParams sp;
        if (r.sampling) sp = *r.sampling;
        else sp.greedy = true;
        if (!r.sampling) sp.temperature = 0.f;
        sp.counter += (uint64_t) t0;
        sp.logit_bias = bias_on_ ? d_bias_ : nullptr;
        history_len_ = r.history != nullptr ? r.history_len : 0;
        if (history_len_ > 0) {   // the penalties' history of each row of the window
            const size_t n = (size_t) kMaxWindow * history_len_;
            if (d_hist_len_ < n) {
                if (d_hist_) cudaFree(d_hist_);
                if (cudaMalloc((void**) &d_hist_, n * 4) != cudaSuccess) { err = "out of device memory for the sampling history"; return false; }
                d_hist_len_ = n;
            }
            cudaMemcpyAsync(d_hist_, r.history, (size_t) rows * history_len_ * 4, cudaMemcpyHostToDevice, stream_);
        }
        sampling_ = sp;
        bk::sample_tokens(logits_, rows, (int) g_.n_vocab, history_len_ > 0 ? d_hist_ : nullptr, history_len_, sp, pick_, stream_);
        picks_.assign((size_t) rows, 0);
        if (cudaMemcpyAsync(picks_.data(), pick_, (size_t) rows * 4, cudaMemcpyDeviceToHost, stream_) != cudaSuccess || cudaStreamSynchronize(stream_) != cudaSuccess) {
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
    float *ik_ = nullptr, *ikn_ = nullptr, *ivt_ = nullptr, *igt_ = nullptr, *iq_ = nullptr, *iw_ = nullptr, *iscore_ = nullptr;
    int32_t *icells_ = nullptr, *meta_d_ = nullptr, *meta_h_ = nullptr;
    int cells_stride_ = 0, rows_ = 0, arows_ = 0, max_heads_ = 0;
    Bufs dec_, big_;   // a window's row buffers, and a prompt chunk's while one runs
    float *gu_ = nullptr, *h_ = nullptr, *rlog_ = nullptr, *rw_ = nullptr;
    int32_t* ids_ = nullptr;
    float *xlat_ = nullptr, *lat_out_ = nullptr, *moe_out_ = nullptr, *sh_ = nullptr, *logits_ = nullptr, *lp_ = nullptr, *erows_ = nullptr, *ekv_ = nullptr, *evalid_ = nullptr;
    int32_t* pick_ = nullptr;
    struct Stage {
        uint8_t* blobs = nullptr;
        int32_t *tok = nullptr, *bounds = nullptr;
        cudaEvent_t copied = nullptr, done = nullptr;
    } stage_[2];
    cudaStream_t copy_ = nullptr;
    size_t stride_ = 0;
    uint64_t launch_ = 0;
    float* pair_rows_ = nullptr;
    int32_t *pair_map_ = nullptr, *pair_ident_ = nullptr;
    void *xq1_ = nullptr, *xq2_ = nullptr;
    basalt::prefill::mmq::Context mmq_;
    std::set<std::pair<int, int>> loaded_;   // (type, rows) the MMQ tiles are loaded for
    void *mmq_xq_ = nullptr;
    int32_t *mmq_bounds_ = nullptr, *mmq_ident_ = nullptr;
    void *q8_ = nullptr, *ex_xq_ = nullptr, *ex_scr_ = nullptr, *ex_plan_ = nullptr;
    float* ex_rows_ = nullptr;
    const float** rowptr_d_ = nullptr;
    const float** h_rowptr_ = nullptr;
    float *h_x_ = nullptr, *h_rows_ = nullptr, *d_rows_ = nullptr, *h_rw_ = nullptr, *h_erows_ = nullptr, *h_e_ = nullptr, *h_evalid_ = nullptr;
    int32_t* h_ids_ = nullptr;
    WindowPlan* h_plan_ = nullptr;
    std::shared_ptr<core::ExpertTier> tier_;
    std::vector<float> lookup_host_;
    std::vector<bool> lookup_loaded_;
    std::vector<uint8_t*> cache_;   // a sparse layer's rows in the --kv format, a window layer's BF16
    std::vector<float*> ival_, igate_;
    std::vector<uint16_t*> ipool_;   // fp16 pooled keys
    DflyHook dfly_;   // DFLY HOOK
    std::vector<std::unique_ptr<core::ElasticArray>> ea_, pea_;
    int64_t backed_ = 0;   // positions the sparse layers' caches are backed for
    std::vector<int32_t> hist_, picks_;
    bool bias_on_ = false;
    float *d_bias_ = nullptr, *d_mask_ = nullptr;
    int* d_hist_ = nullptr;
    size_t d_hist_len_ = 0;
    int history_len_ = 0;
    bk::SamplerParams sampling_;
    // [vision]
    ImageEncoder vis_;
    RopeTrack rope_;
    int32_t* rope_dev_ = nullptr;
    bool rope_on_ = false;               // a picture has been fed: the kernels read the rotary table
    const float* emb_rows_ = nullptr;    // the embeddings of the rows being run, in place of the tokens'
    int32_t img_start_ = 0;              // the first position of the block being fed
    int img_nx_ = 1, img_j_ = 0;         // its columns (newline included) and the row of its first run row
    int last_rows_ = 0;
    // prompt timing (host wall time at the sync points): the work queued before a layer's MoE, the MoE itself, the Engram's reads
    double t_pre_ms_ = 0, t_moe_ms_ = 0, t_eng_ms_ = 0;
    static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
};

}  // namespace

std::unique_ptr<model::ModelSession> make_session(Hy4Model& model, std::string& err) {
    auto s = std::make_unique<Session>(model);
    if (!s->init(err)) return nullptr;
    return s;
}

}  // namespace basalt::models::hy4flash
