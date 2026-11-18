// src/models/hy4flash/session.cpp - one conversation of Hy4-flash: its latent caches, indexer rows and pooled keys, and the
// forward of a run of rows that every decode window (up to 8 rows) and every prompt chunk (up to kMaxRows) goes through.
//
// Per layer: the Engram step (on its layers), the attention's hyper-connection pre-mix and norm, the absorbed MLA over the
// shared latent (a ring on the window layers, the indexer's selection on the others), the gate, the output projection and its
// post-norm, the stream expansion, and the same for the FFN: the dense MLP, or the latent MoE (router, routed_down, the experts
// the tier places in groups of up to 8 rows beside the shared expert, routed_up). A row's result does not depend on the other
// rows of its call: the dense products take 8 rows at most per launch with the single-row kernels' reduction order, the
// attention kernels take a block per row, and the experts are run per row.
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
#include "models/hy4flash/engram.hpp"
#include "models/hy4flash/model.hpp"
#include "models/hy4flash/vision_encoder.hpp"   // [vision]

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
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
constexpr int kMaxRows = 256;                  // rows one call carries: a prompt chunk
constexpr int kMaxWindow = 8;                  // rows of a decode window, and of one expert group
constexpr int kRawRing = kMaxRows + 16;        // the indexer's rows of the last positions (a multiple of the pool size)
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
        const int64_t E = g_.n_embd, ctx = m_.context(), R = kMaxRows;
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
        X_ = f(R * 4 * E); Xn_ = f(R * 4 * E); coef_ = f(R * bk::kHcCoef); hcws_ = f((int64_t) bk::hc_plain_workspace_floats(kMaxRows));
        e_ = f(R * E); cur_ = f(R * E); xn_ = f(R * E); attn_ = f(R * E); attn_raw_ = f(R * E); mlp_ = f(R * E); mlp2_ = f(R * E);
        fin_ = f(kMaxWindow * E);
        qa_ = f(R * g_.q_lora); qan_ = f(R * g_.q_lora); q_ = f(R * max_heads * g_.qk_dim); kvraw_ = f(R * kRowWidth);
        qabs_ = f(R * max_heads * g_.kv_lora); qcur_ = f(R * max_heads * kRowWidth);
        olat_ = f(R * max_heads * g_.kv_lora); vout_ = f(R * max_heads * g_.v_dim); gate_ = f(R * max_heads * g_.v_dim);
        ik_ = f(R * g_.idx_dim); ikn_ = f(R * g_.idx_dim); ivt_ = f(R * g_.idx_dim); igt_ = f(R * g_.idx_dim);
        iq_ = f(R * g_.idx_heads * g_.idx_dim); iw_ = f(R * g_.idx_heads);
        const int64_t pools = ctx / g_.idx_pool + 1;
        iscore_ = f(pools);
        cells_stride_ = g_.idx_top_k + 2 * g_.idx_pool;
        icells_ = (int32_t*) alloc((size_t) R * cells_stride_ * 4, ok);
        meta_d_ = (int32_t*) alloc((size_t) 3 * R * 4, ok);
        meta_h_ = (int32_t*) pin((size_t) g_.n_layers * 3 * R);
        gu_ = f(2 * R * (int64_t) g_.n_ff); h_ = f(R * (int64_t) g_.n_ff);
        rlog_ = f(R * g_.n_expert); rw_ = f(R * 16); ids_ = (int32_t*) alloc((size_t) R * 16 * 4, ok);
        xlat_ = f(R * g_.n_latent); lat_out_ = f(R * g_.n_latent); moe_out_ = f(R * E); sh_ = f(R * E);
        logits_ = f((int64_t) kMaxWindow * g_.n_vocab); pick_ = (int32_t*) alloc(4 * 32, ok);
        erows_ = f(R * g_.n_embd);   // the Engram's normed rows: 16 x 96 values a row
        ekv_ = f(R * 5 * E);
        evalid_ = f(R);
        lp_ = f((int64_t) bk::row_logprobs_scratch_floats(kMaxWindow, (int) g_.n_vocab) + (int64_t) kMaxWindow * (bk::kLogprobStride + 1));
        q8_ = alloc(std::max<size_t>(bk::native_q8_1_bytes(16384, kMaxWindow), 1 << 16), ok);
        mmq_xq_ = alloc(basalt::prefill::mmq::q8_bytes(kMaxRows, 16384), ok);
        mmq_bounds_ = (int32_t*) alloc(2 * 4, ok);
        mmq_ident_ = (int32_t*) alloc((size_t) kMaxRows * 4, ok);
        if (ok && basalt::prefill::mmq::built()) basalt::prefill::mmq::iota(mmq_ident_, kMaxRows, nullptr);
        ex_xq_ = alloc(bk::native_q8_1_bytes(g_.n_latent, kMaxWindow), ok);
        ex_scr_ = alloc(bk::native_expert_scratch_bytes(kLookupK, g_.n_ff_exp), ok);
        ex_rows_ = f((int64_t) kLookupK * g_.n_latent);
        ex_plan_ = alloc(sizeof(WindowPlan), ok);
        rowptr_d_ = (const float**) alloc(kLookupK * sizeof(void*), ok);
        h_rowptr_ = (const float**) pin(kLookupK * 2);
        h_rw_ = pin(kMaxWindow * 16); h_ids_ = (int32_t*) pin(kMaxWindow * 16);
        rope_dev_ = (int32_t*) alloc((size_t) ctx * 12, ok);   // [vision] the 3-D rotary position of every cell
        h_erows_ = pin(R * g_.n_embd); h_e_ = pin(R * E); h_evalid_ = pin(R);
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
                cache_[(size_t) l] = (uint8_t*) alloc((size_t) swa_ring() * kRowWidth * 2, ok);
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
        return true;
    }

    int max_window() const override { return kMaxWindow; }

    model::Result decode_window(const model::DecodeRequest& r) override {
        if (r.n_tokens < 1) return {model::Status::Error, "hy4flash: an empty window"};
        std::string err;
        for (int t0 = 0; t0 < r.n_tokens; t0 += kMaxWindow) {
            const int n = std::min(kMaxWindow, r.n_tokens - t0);
            const bool last = t0 + n == r.n_tokens;
            const bool want = r.row_tokens || (last && (r.next_token || r.logits));
            const int head = !want ? 0 : r.row_tokens ? n : 1;
            if (!run_rows(r.tokens + t0, n, r.pos + t0, head, err)) return {model::Status::Error, "hy4flash: " + err};
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
        for (int t0 = 0; t0 < r.n_tokens; t0 += kMaxRows) {
            if (r.stop && r.stop()) break;
            const int n = std::min(kMaxRows, r.n_tokens - t0);
            if (!run_rows(r.tokens + t0, n, r.pos + t0, 0, err)) return {model::Status::Error, "hy4flash: " + err};
            if (r.progress) {
                cudaStreamSynchronize(stream_);
                r.progress(t0 + n);
            }
        }
        if (cudaStreamSynchronize(stream_) != cudaSuccess) return {model::Status::Error, "hy4flash: the prompt failed"};
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
        const std::vector<int32_t> dead((size_t) kMaxRows, -1);
        for (int at = 0; at < n_rows; at += kMaxRows) {
            const int n = std::min(kMaxRows, n_rows - at);
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

    int swa_ring() const { return g_.window + kMaxRows; }
    bool fp8() const { return m_.cache_fp8(); }
    int sparse_row_bytes() const { return fp8() ? kFp8RowBytes : kRowWidth * 2; }

    // y[r] = W x[r] for R rows, 8 at a time: a row's product does not depend on the rows beside it
    void gemm(const DevTensor* w, const float* x, float* y, int n_in, int n_out, int R) {
        namespace mmq = basalt::prefill::mmq;
        if (R > kMaxWindow && mmq::built() && mmq::fits(w->type, n_out)) {   // a chunk: the tensor-core MMQ product
            if (loaded_.insert({w->type, n_out}).second) mmq::load(w->type, n_out);
            const int32_t b[2] = {0, R};
            cudaMemcpyAsync(mmq_bounds_, b, sizeof b, cudaMemcpyHostToDevice, stream_);
            mmq::quantize(x, nullptr, mmq_xq_, w->type, n_in, n_in, R, stream_);
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
            p.total_rows = R;
            p.max_rows = R;
            p.dst = y;
            p.ld_dst = n_out;
            mmq_.run(p, stream_);
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

    // Attention of layer `l` over R rows at pos0.. of the normed input xn_; the post-normed result goes to attn_.
    bool attention(int l, int64_t pos0, int R, std::string& err) {
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
        else mla_prepare(q_, qabs_, kvraw_, kvn, a, (int) pos0, R, ring, qcur_, (uint16_t*) cache, stream_);
        // the cells each row attends
        int32_t* meta = meta_h_ + (size_t) l * 3 * kMaxRows;   // firsts, counts, fill
        int32_t *firsts = meta, *counts = meta + kMaxRows, *fill = meta + 2 * kMaxRows;
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
        }
        cudaMemcpyAsync(meta_d_, meta, (size_t) 3 * kMaxRows * 4, cudaMemcpyHostToDevice, stream_);
        RowSpans rs;
        rs.firsts = meta_d_;
        rs.counts = meta_d_ + kMaxRows;
        if (!swa) {
            iota_cells(icells_, cells_stride_, meta_d_ + 2 * kMaxRows, R, stream_);
            rs.cells = icells_;
            rs.cells_stride = cells_stride_;
        }
        if (f8) mla_attend(qcur_, (const uint8_t*) cache, rs, R, F(l, "attn_sinks.weight"), a, ring, olat_, stream_);
        else mla_attend(qcur_, (const uint16_t*) cache, rs, R, F(l, "attn_sinks.weight"), a, ring, olat_, stream_);
        q8_0_heads_gemv(T(l, "attn_v_b.weight")->ptr, olat_, rank, H * rank, H, g_.v_dim, rank, R, vout_, stream_);
        gemm(T(l, "attn_gate.weight"), xn_, gate_, E, H * g_.v_dim, R);
        mul_sigmoid(gate_, vout_, R * H * g_.v_dim, stream_);
        gemm(T(l, "attn_output.weight"), vout_, attn_raw_, H * g_.v_dim, E, R);
        bk::rms_norm_rows(attn_raw_, F(l, "post_attention_norm.weight"), attn_, R, E, g_.rms_eps, stream_);
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

    // The Engram step of layer `l` on the streams of R rows.
    bool engram(int l, int64_t pos0, int R, const int32_t* tokens, std::string& err) {
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
            float *gate = gu_, *up = gu_ + (size_t) R * g_.n_ff;
            gemm(T(l, "ffn_gate.weight"), xn_, gate, E, g_.n_ff, R);
            gemm(T(l, "ffn_up.weight"), xn_, up, E, g_.n_ff, R);
            bk::swiglu_scaled(gate, up, nullptr, nullptr, h_, R, g_.n_ff, g_.n_ff, stream_);
            gemm(T(l, "ffn_down.weight"), h_, mlp_, g_.n_ff, E, R);
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
        for (int g0 = 0; g0 < R; g0 += kMaxWindow)
            if (!moe_group(l, tl, lay, g0, std::min(kMaxWindow, R - g0), err)) return false;
        gemm(T(l, "ffn_routed_up.weight"), lat_out_, moe_out_, L, E, R);
        add_vectors(moe_out_, sh_, mlp_, R * E, st);
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
        for (int l = 0; l < g_.n_layers; ++l) {
            if (!engram(l, pos0, R, tokens, err)) return false;
            bk::hc_mixes_plain(X_, F(l, "hc_attn_fn.weight"), F(l, "hc_attn_scale.weight"), F(l, "hc_attn_base.weight"), R, 8, hc, hcws_, coef_, stream_);
            bk::hc_pre_rms_norm(X_, coef_, bk::kHcCoef, F(l, "attn_norm.weight"), R, E, g_.rms_eps, cur_, xn_, stream_);
            if (!attention(l, pos0, R, err)) return false;
            bk::hc_post(attn_, X_, coef_, bk::kHcCoef, R, E, Xn_, stream_);
            std::swap(X_, Xn_);
            bk::hc_mixes_plain(X_, F(l, "hc_ffn_fn.weight"), F(l, "hc_ffn_scale.weight"), F(l, "hc_ffn_base.weight"), R, 8, hc, hcws_, coef_, stream_);
            bk::hc_pre_rms_norm(X_, coef_, bk::kHcCoef, F(l, "ffn_norm.weight"), R, E, g_.rms_eps, cur_, xn_, stream_);
            if (!ffn(l, R, err)) return false;
            bk::rms_norm_rows(mlp_, F(l, "post_ffw_norm.weight"), mlp2_, R, E, g_.rms_eps, stream_);
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

    // The picks of the last `rows` rows of logits_ (row t of the window is draw counter + t0 + t).
    bool pick(const model::DecodeRequest& r, int t0, int rows, std::string& err) {
        bk::SamplerParams sp;
        if (r.sampling) sp = *r.sampling;
        else sp.greedy = true;
        if (!r.sampling) sp.temperature = 0.f;
        sp.counter += (uint64_t) (t0 + (r.row_tokens ? 0 : 0));
        bk::sample_tokens(logits_, rows, (int) g_.n_vocab, nullptr, 0, sp, pick_, stream_);
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
    int cells_stride_ = 0;
    float *gu_ = nullptr, *h_ = nullptr, *rlog_ = nullptr, *rw_ = nullptr;
    int32_t* ids_ = nullptr;
    float *xlat_ = nullptr, *lat_out_ = nullptr, *moe_out_ = nullptr, *sh_ = nullptr, *logits_ = nullptr, *lp_ = nullptr, *erows_ = nullptr, *ekv_ = nullptr, *evalid_ = nullptr;
    int32_t* pick_ = nullptr;
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
    std::vector<std::unique_ptr<core::ElasticArray>> ea_, pea_;
    int64_t backed_ = 0;   // positions the sparse layers' caches are backed for
    std::vector<int32_t> hist_, picks_;
    // [vision]
    ImageEncoder vis_;
    RopeTrack rope_;
    int32_t* rope_dev_ = nullptr;
    bool rope_on_ = false;               // a picture has been fed: the kernels read the rotary table
    const float* emb_rows_ = nullptr;    // the embeddings of the rows being run, in place of the tokens'
    int32_t img_start_ = 0;              // the first position of the block being fed
    int img_nx_ = 1, img_j_ = 0;         // its columns (newline included) and the row of its first run row
    int last_rows_ = 0;
};

}  // namespace

std::unique_ptr<model::ModelSession> make_session(Hy4Model& model, std::string& err) {
    auto s = std::make_unique<Session>(model);
    if (!s->init(err)) return nullptr;
    return s;
}

}  // namespace basalt::models::hy4flash
