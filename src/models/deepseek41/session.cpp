// src/models/deepseek41/session.cpp - the DeepSeek-V4.1 layer loop for one token at a time.
//
// Per layer: Engram (on its layers), hyper-connection mixes and collapse, attention, expansion, the FFN's mixes and
// collapse, router, routed and shared experts, expansion. Prefill runs the same path token by token.
#include "basalt/log.hpp"
#include "models/deepseek41/session.hpp"
#include "basalt/ggml/ggml.h"

#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/models/deepseek41/attn_index.hpp"
#include "basalt/models/deepseek41/kernels.hpp"
#include "models/deepseek41/attn_batch.hpp"
#include "models/deepseek41/dspark.hpp"
#include "models/deepseek41/dspark_kernels.hpp"
#include "models/deepseek41/layer_ops.hpp"
#include "models/deepseek41/model.hpp"
#include "models/deepseek41/ops.hpp"
#include "models/deepseek41/prefill.hpp"
#include "models/deepseek41/prefill_plan.hpp"
#include "models/deepseek41/tier.hpp"
#include "basalt/models/deepseek41/vision.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "basalt/artifact/dequant.hpp"

namespace basalt::models::deepseek41 {
namespace {
constexpr int kLendTierTokens = 4096;   // a prompt this long reads without the VRAM tier, in larger chunks
constexpr int kStepTailTokens = 256;     // a tail this short may read one decode step at a time when no workspaces fit

#define DS_CUDA(call)                                                                                   \
    do {                                                                                                \
        cudaError_t e_ = (call);                                                                        \
        if (e_ != cudaSuccess) return fail(std::string(#call) + ": " + cudaGetErrorString(e_));         \
    } while (0)

constexpr int kMaxWindow = kMaxT;   // rows one decode window verifies

class Session final : public model::ModelSession {
public:
    explicit Session(DsModel& m) : m_(m), g_(m.geo()) {}
    ~Session() override {
        if (drafter_) drafter_->report();
        for (void* p : allocs_) cudaFree(p);
        if (h_next_) cudaFreeHost(h_next_);
        if (h_dfeat_) cudaFreeHost(h_dfeat_);
        for (cudaGraphExec_t e : head_g_)
            if (e) cudaGraphExecDestroy(e);
        for (cudaGraphExec_t e : tail_g_)
            if (e) cudaGraphExecDestroy(e);
        for (cudaEvent_t e : mix_ev_)
            if (e) cudaEventDestroy(e);
        if (mix_st_) cudaStreamDestroy(mix_st_);
        for (cudaEvent_t e : eng_ev_)
            if (e) cudaEventDestroy(e);
        if (eng_go_) cudaEventDestroy(eng_go_);
        if (eng_st_) cudaStreamDestroy(eng_st_);
        for (cudaEvent_t e : row_ev_)
            if (e) cudaEventDestroy(e);
        if (row_st_) cudaStreamDestroy(row_st_);
        for (float* h : erow_h_) cudaFreeHost(h);
        if (stream_) cudaStreamDestroy(stream_);
    }

    bool init(std::string& err) {
        // the layer's chain runs ahead of the mixes branch, one step below the attention's branches, which hold the chain
        // up on the compressor's layers (graph nodes keep their stream's priority)
        int least = 0, greatest = 0;
        if (cudaDeviceGetStreamPriorityRange(&least, &greatest) != cudaSuccess ||
            cudaStreamCreateWithPriority(&stream_, cudaStreamDefault, greatest < least ? greatest + 1 : greatest) != cudaSuccess ||
            cudaStreamCreateWithPriority(&mix_st_, cudaStreamNonBlocking, least) != cudaSuccess) {
            err = "cannot create a CUDA stream";
            return false;
        }
        for (cudaEvent_t& e : mix_ev_)
            if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) {
                err = "cannot create a CUDA event";
                return false;
            }
        const size_t n = (size_t) g_.n_embd;
        head_g_.assign((size_t) g_.n_layers * 2 * kMaxWindow, nullptr);   // [T - 1][layer][with previous tail]
        tail_g_.assign((size_t) g_.n_layers * kMaxWindow, nullptr);
        constexpr int W = kMaxWindow;
        auto buf = [&](float*& p, size_t floats) {
            void* v = nullptr;
            if (cudaMalloc(&v, floats * sizeof(float)) != cudaSuccess) return false;
            allocs_.push_back(v);
            p = (float*) v;
            return true;
        };
        const size_t cols = (size_t) m_.pack().engram.hash.n_cols() * 256;
        bool ok = buf(x_, 4 * n * W) && buf(x2_, 4 * n * W) && buf(cur_, n * W) && buf(normed_, n * W) &&
                  buf(attn_out_, n * W) && buf(moe_out_, n * W) && buf(sh_out_, n * W) && buf(ffn_sum_, n * W) &&
                  buf(coef_a_, kHcCoef * W) && buf(coef_f_[0], kHcCoef * W) && buf(coef_f_[1], kHcCoef * W) &&
                  buf(hc_ws_, hc_mixes_workspace_floats(W)) && buf(rlogits_, (size_t) g_.n_expert * W) &&
                  buf(rw_, 64) && buf(logits_, (size_t) g_.n_vocab * W);
        for (size_t k = 0; ok && k < m_.pack().engram.layers.size(); ++k) {
            float *h = nullptr, *d = nullptr, *kv = nullptr;
            void* q = nullptr;
            ok = cudaHostAlloc((void**) &h, cols * W * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped) == cudaSuccess;
            if (h) erow_h_.push_back(h);
            ok = ok && cudaHostGetDevicePointer((void**) &d, h, 0) == cudaSuccess && buf(kv, 5 * n * W) &&
                 cudaMalloc(&q, kernels::native_q8_1_bytes((int) cols, W)) == cudaSuccess;
            if (q) allocs_.push_back(q);
            erow_d_.push_back(d);
            ekv_.push_back(kv);
            eq8_.push_back(q);
            eng_ev_.push_back(nullptr);
            ok = ok && cudaEventCreateWithFlags(&eng_ev_.back(), cudaEventDisableTiming) == cudaSuccess;
        }
        ok = ok && cudaStreamCreateWithFlags(&eng_st_, cudaStreamNonBlocking) == cudaSuccess &&
             cudaEventCreateWithFlags(&eng_go_, cudaEventDisableTiming) == cudaSuccess &&
             cudaStreamCreateWithFlags(&row_st_, cudaStreamNonBlocking) == cudaSuccess &&
             cudaEventCreateWithFlags(&row_ev_[0], cudaEventDisableTiming) == cudaSuccess &&
             cudaEventCreateWithFlags(&row_ev_[1], cudaEventDisableTiming) == cudaSuccess;
        void* v = nullptr;
        ok = ok && cudaMalloc(&v, 64 * sizeof(int32_t)) == cudaSuccess;
        ids_ = (int32_t*) v;
        if (v) allocs_.push_back(v);
        v = nullptr;
        ok = ok && cudaMalloc(&v, W * sizeof(int32_t)) == cudaSuccess;
        am_dev_ = (int32_t*) v;
        if (v) allocs_.push_back(v);
        v = nullptr;
        ok = ok && cudaMalloc(&v, argmax_workspace_bytes()) == cudaSuccess && cudaMemset(v, 0, argmax_workspace_bytes()) == cudaSuccess;
        am_ws_ = v;
        if (v) allocs_.push_back(v);
        ok = ok && cudaHostAlloc((void**) &h_next_, W * sizeof(int32_t), cudaHostAllocPortable) == cudaSuccess;
        v = nullptr;
        const size_t q8 = std::max<size_t>(kernels::native_q8_1_bytes((int) std::max<size_t>(cols, n), W), 1 << 16);
        ok = ok && cudaMalloc(&v, q8) == cudaSuccess;
        q8_ = v;
        if (v) allocs_.push_back(v);
        if (!ok) {
            err = "out of device memory for the session";
            return false;
        }
        ops_ = make_kernel_ops(m_, err);
        if (!ops_) return false;
        dump_dir_ = m_.diag_dump();
        if (m_.tiers().draft && !m_.pack().dspark.empty() && m_.sessions() == 1) {
            // the drafter takes its memory before the tiers size themselves to what is left
            const int device = m_.tiers().drafter_device >= 0 ? m_.tiers().drafter_device : m_.device();
            // the ranking that says which experts each stage keeps ships beside the expert profiles
            const std::string ranking = m_.tiers().profile_dir.empty() ? std::string()
                : m_.tiers().profile_dir + "/deepseek41-" + std::to_string(g_.n_layers) + "x" + std::to_string(g_.n_expert) + "-dspark.prof";
            drafter_ = DsparkDrafter::create(m_, device, ranking, m_.tiers().draft_experts, err);
            if (!drafter_) return false;
            basalt::log::info("basalt deepseek41", "drafter on GPU %d: %.2f GiB, %s, windows of at most %d tokens (--spec 0: off)", device,
                              (double) drafter_->device_bytes() / (1ull << 30), drafter_->kept().c_str(),
                              std::min(max_window(), kDsparkBlock + 1));
            const size_t ff = (size_t) drafter_->feature_floats();
            if (!buf(dfeat_, ff * kDsparkWindow) ||
                cudaHostAlloc((void**) &h_dfeat_, ff * kDsparkWindow * sizeof(float), cudaHostAllocPortable) != cudaSuccess) {
                err = "out of memory for the drafter's feature rows";
                return false;
            }
        }
        return true;
    }

    int max_window() const override { return m_.sessions() > 1 ? 1 : std::clamp(m_.tiers().spec, 1, kMaxWindow); }

    model::Result draft_begin(int32_t anchor, int64_t pos) override {
        if (!drafter_) return {model::Status::NotImplemented, "deepseek41: no drafter is loaded"};
        std::string err;
        ops_->commit_window(pos, stream_);
        if (!flush_features(pos, err) || !drafter_->draft_begin(anchor, pos, std::min(max_window() - 1, kDsparkBlock), err))
            return {model::Status::Error, err};
        ops_->end_window(true);   // the copies queued behind the drafter's inputs run while it drafts
        return {};
    }

    model::Result draft_end(int32_t* ids, float* conf, int max_n, int& n) override {
        if (!drafter_) return {model::Status::NotImplemented, "deepseek41: no drafter is loaded"};
        std::string err;
        DsparkDraft d;
        if (!drafter_->draft_end(d, err)) return {model::Status::Error, err};
        n = std::min(max_n, d.n);
        std::copy(d.token, d.token + n, ids);
        std::copy(d.conf, d.conf + n, conf);
        return {};
    }

    model::Result decode_window(const model::DecodeRequest& r) override {
        if (r.row_tokens) {
            if (r.n_tokens > kMaxWindow) return {model::Status::Error, "deepseek41: the decode window is too long"};
            std::string err;
            if (!prepare_draw(r, err)) return {model::Status::Error, err};
            if (!step(r.tokens, r.n_tokens, r.pos, true, r.row_tokens, r.logits, err)) return {model::Status::Error, err};
            return {};
        }
        for (int i = 0; i < r.n_tokens; ++i) {
            const bool last = i + 1 == r.n_tokens;
            std::string err;
            if (last && !prepare_draw(r, err)) return {model::Status::Error, err};
            if (!step(r.tokens + i, 1, r.pos + i, last, last ? r.next_token : nullptr, last ? r.logits : nullptr, err))
                return {model::Status::Error, err};
        }
        return {};
    }

    model::Result prefill_chunk(const model::PrefillRequest& r) override {
        if (!prefillp::may_resume(r.pos, (int64_t) hist_.size(), whole_from_)) return {model::Status::Error, kSkippedRows};
        sync_rewind(r.pos);
        settle_tier();
        {
            std::string fe;
            if (!flush_features(r.pos, fe)) return {model::Status::Error, fe};
        }
        if (r.n_tokens > 1 && std::max(r.total_tokens, r.n_tokens) > prefillp::kStepPrompt) {
            std::string err;
            const int batched = r.n_tokens;
            const int plan_tokens = std::max(r.total_tokens, r.n_tokens);
            {
                if (plan_tokens > prefillp::kShortPrompt) release_short_prompt();
                open_prompt_path(plan_tokens);
                if (plan_tokens >= kLendTierTokens) {
                    std::string te;
                    if (ExpertTier* t = ops_->expert_tier(te)) t->lend_all();
                }
                // the caches take their memory before the prompt workspaces do
                if (!ops_->ensure_positions(r.pos + r.n_tokens, err)) return prompt_failed(err);
                if (!prefill_ && !make_prefill_fit(plan_tokens, err)) {
                    // a long context's caches left no room: a short tail takes the decode path, a longer read the tier's memory
                    if (r.n_tokens <= kStepTailTokens) {
                        close_prompt_path();
                        for (int i = 0; i < r.n_tokens; ++i)
                            if (!step(r.tokens + i, 1, r.pos + i, false, nullptr, nullptr, err)) return {model::Status::Error, err};
                        return {};
                    }
                    std::string te;
                    err.clear();
                    short_prompt_ = false;
                    if (ExpertTier* t = ops_->expert_tier(te)) t->lend_all();
                    if (!make_prefill_fit(plan_tokens, err)) return prompt_failed(err);
                }
                prefill_->progress = r.progress;
                prefill_->live = r.live;
                prefill_->stop = r.stop;
                tap_features();
                if (!prefill_->run(r.tokens, batched, r.pos, r.rows_after, hist_, err)) return prompt_failed(err);
                if (prefill_->left_rows()) whole_from_ = prefillp::whole_from(r.pos + r.n_tokens + r.rows_after, kWindow);
                if (!r.more) close_prompt_path();
            }
            return {};
        }
        for (int i = 0; i < r.n_tokens; ++i) {
            std::string err;
            if (!step(r.tokens + i, 1, r.pos + i, false, nullptr, nullptr, err)) return {model::Status::Error, err};
        }
        return {};
    }

    model::Result prepare() override {
        std::string e;
        if (ops_ && !ops_->expert_tier(e) && !e.empty()) return {model::Status::Error, e};
        return {};
    }

    size_t running_bytes() const override { return ops_ ? ops_->turn_bytes() : 0; }

    model::Result save_running(uint8_t* host, int64_t pos) override {
        if (!prefillp::may_save(pos, whole_from_)) return {model::Status::Error, kSkippedRows};
        sync_rewind(pos);
        if (!ops_) return {model::Status::NotImplemented, "no state yet"};
        ops_->save_turn(stream_, host);
        return {};
    }

    model::Result restore_running(const uint8_t* host, int64_t pos) override {
        if (!ops_) return {model::Status::Error, "no state yet"};
        ops_->restore_turn(stream_, host);
        whole_from_ = 0;
        if ((int64_t) hist_.size() > pos) hist_.resize((size_t) pos);
        feat_n_ = 0;
        if (drafter_) drafter_->forget_from(pos);
        return {};
    }

    model::Result set_low_memory(bool low, int64_t& expert_slots) override {
        std::string e;
        ExpertTier* t = ops_ ? ops_->expert_tier(e) : nullptr;
        if (t == nullptr) return {model::Status::Error, e.empty() ? std::string("this session has no expert tier") : e};
        cudaStreamSynchronize(stream_);
        release_short_prompt();
        if (low) {
            t->lend_all();
        } else {
            if (!t->reclaim(e)) return {model::Status::Error, e};
            t->grow();
        }
        expert_slots = t->slots();
        return {};
    }

    model::Result set_vram_reserve(int64_t mib, int64_t& expert_slots) override {
        std::string e;
        ExpertTier* t = ops_ ? ops_->expert_tier(e) : nullptr;
        if (t == nullptr) return {model::Status::Error, e.empty() ? std::string("this session has no expert tier") : e};
        cudaStreamSynchronize(stream_);
        release_short_prompt();
        t->set_free_floor(mib);
        t->lend_all();
        if (!t->reclaim(e)) return {model::Status::Error, e};
        t->grow();
        expert_slots = t->slots();
        return {};
    }

    model::Result probe_env(program::ProbeEnv& env) override {
        std::string e;
        ExpertTier* t = ops_ ? ops_->expert_tier(e) : nullptr;
        if (t == nullptr) return {model::Status::Error, e.empty() ? std::string("this session has no expert tier") : e};
        cudaStreamSynchronize(stream_);
        t->probe_env(env);
        return {};
    }

    // a read of a few thousand tokens streams every expert over the bus
    int64_t disk_min_tokens() const override { return 2048; }

    ExpertCounts expert_counts() const override {
        std::string e;
        const ExpertTier* t = ops_ ? ops_->expert_tier(e) : nullptr;
        return t == nullptr ? ExpertCounts{} : ExpertCounts{t->hits(), t->lookups(), t->offloaded(), t->file_lookups()};
    }

    std::string info_words() const override {
        std::string e;
        ExpertTier* t = ops_ ? ops_->expert_tier(e) : nullptr;
        return t == nullptr ? std::string("kv=fp16")
                            : "expert_slots=" + std::to_string(t->slots()) + " kv=fp16 " + program::spill_words(t->spill());
    }

    size_t cells_bytes(int64_t positions) const override { return ops_ ? ops_->cache_bytes(positions) : 0; }

    model::Result save_cells(std::vector<core::ConversationKv>& kv, int64_t positions) override {
        if (!ops_) return {model::Status::Error, "no state yet"};
        std::string e;
        if (!ops_->save_cache(stream_, positions, kv, e)) return {model::Status::Error, e};
        return {};
    }

    // The tokens below `positions` come back with them, so the n-gram history is the caller's `tokens`.
    model::Result restore_cells(const std::vector<core::ConversationKv>& kv, int64_t positions) override {
        if (!ops_) return {model::Status::Error, "no state yet"};
        std::string e;
        if (!ops_->ensure_positions(positions, e) || !ops_->restore_cache(stream_, kv, e)) return {model::Status::Error, e};
        feat_n_ = 0;
        if (drafter_) drafter_->reset();
        return {};
    }

    model::Result set_history(const int32_t* tokens, int64_t n) override {
        hist_.assign(tokens, tokens + n);
        return {};
    }

    void end_prompt() override {
        if (prefill_) close_prompt_path();
    }

    model::Result set_logit_bias(const float* bias) override {
        bias_on_ = bias != nullptr;
        if (!bias_on_) return {};
        std::string e;
        if (!device_bias(d_bias_, bias, e)) return {model::Status::Error, e};
        return {};
    }

    model::Result resample_row(int row, const float* bias, int32_t& token) override {
        std::string e;
        if (!device_bias(d_mask_, bias, e)) return {model::Status::Error, e};
        kernels::SamplerParams sp = sampling_;
        sp.logit_bias = d_mask_;
        sp.counter += (uint64_t) row;
        kernels::sample_tokens(logits_ + (size_t) row * g_.n_vocab, 1, g_.n_vocab, history_len_ > 0 ? d_hist_ + (size_t) row * history_len_ : nullptr,
                               history_len_, sp, am_dev_, stream_);
        if (cudaMemcpyAsync(h_next_, am_dev_, sizeof(int32_t), cudaMemcpyDeviceToHost, stream_) != cudaSuccess ||
            cudaStreamSynchronize(stream_) != cudaSuccess)
            return {model::Status::Error, "the masked draw failed"};
        token = *h_next_;
        return {};
    }

    model::Result copy_logits(float* out) override {
        cudaStreamSynchronize(stream_);
        if (cudaMemcpy(out, logits_, (size_t) g_.n_vocab * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
            return {model::Status::Error, "the head logits copy failed"};
        return {};
    }

    model::Result encode_image(const uint8_t* data, size_t bytes, std::vector<float>& rows, int& n_rows) override {
        std::string err;
        if (!image_) {   // this slot's image buffers, allocated with the tier
            ExpertTier* tier = ops_ ? ops_->expert_tier(err) : nullptr;
            image_ = tier ? tier->host().take_image() : nullptr;
            if (!image_) return {model::Status::Error, err.empty() ? "deepseek41: no image buffers for this slot" : err};
        }
        vision::Image& img = image_->picture;
        if (!vision::decode_image(data, bytes, img, err)) return {model::Status::Error, err};
        if (!vis_) {
            release_short_prompt();
            // the encoder takes the memory of the tier's extra copies, and of all its copies when that is short;
            // images_done gives it back
            std::string te;
            ExpertTier* t = ops_->expert_tier(te);
            cudaStreamSynchronize(stream_);
            vis_borrowed_ = true;
            if (t) t->shrink();
            vision::Mmproj mp;
            if (!m_.open_mmproj(mp, err)) { images_done(); return {model::Status::Error, err}; }
            const VisionPlan pl = plan_image(img.w, img.h, mp.hp().merge, mp.hp().min_pixels);
            vis_ = std::make_unique<VisionEncoder>();
            const auto above_floor = [&] {
                size_t free_b = 0, total_b = 0;
                return !t || cudaMemGetInfo(&free_b, &total_b) != cudaSuccess || free_b >= t->free_floor();
            };
            bool opened = vis_->init(mp, std::max(pl.n_patches(), 4096), err) && above_floor();
            if (!opened && t) {
                vis_ = std::make_unique<VisionEncoder>();
                t->lend_all();
                err.clear();
                opened = vis_->init(mp, std::max(pl.n_patches(), 4096), err);
                if (opened && !above_floor()) {
                    opened = false;
                    err = "not enough free device memory for the image encoder above the VRAM reserve";
                }
            }
            if (!opened) { images_done(); return {model::Status::Error, err}; }
            vis_merge_ = mp.hp().merge;
            vis_min_pixels_ = mp.hp().min_pixels;
        }
        const VisionPlan pl = plan_image(img.w, img.h, vis_merge_, vis_min_pixels_);
        std::vector<float>& patches = image_->patches;
        image_patches(img, pl, patches);
        if (!vis_->encode(patches, pl, rows, err)) { images_done(); return {model::Status::Error, err}; }
        n_rows = pl.n_tokens();
        return {};
    }

    void images_done() override {
        if (!vis_borrowed_) return;
        vis_borrowed_ = false;
        cudaStreamSynchronize(stream_);
        vis_.reset();
        std::string e;
        if (ExpertTier* t = ops_->expert_tier(e)) {
            if (!t->reclaim(e)) basalt::log::warn("basalt deepseek41", "VRAM expert tier stays closed: %s", e.c_str());
            t->grow();
        }
    }

    model::Result prefill_embeddings(const float* rows, int n_rows, int64_t pos) override {
        if (!prefillp::may_resume(pos, (int64_t) hist_.size(), whole_from_)) return {model::Status::Error, kSkippedRows};
        sync_rewind(pos);
        settle_tier();
        std::string err;
        if (n_rows > prefillp::kShortPrompt) release_short_prompt();
        open_prompt_path(n_rows);
        if (!ops_->ensure_positions(pos + n_rows, err)) return prompt_failed(err);
        if (!prefill_ && !make_prefill_fit(n_rows, err)) {
            std::string te;
            err.clear();
            short_prompt_ = false;
            if (ExpertTier* t = ops_->expert_tier(te)) t->lend_all();
            if (!make_prefill_fit(n_rows, err)) return prompt_failed(err);
        }
        tap_features();
        if (!prefill_->run_embeddings(rows, n_rows, pos, hist_, err)) return prompt_failed(err);
        close_prompt_path();
        return {};
    }

private:
    // The prompt path's workspaces and the expert tier share the card. The tier opens first; the workspaces size to
    // the free memory beside it. Once a prompt has run they are released and the tier takes the memory as extra slots,
    // which it gives back before the next prompt allocates them again.
    // The prompt workspaces for the largest chunk the free memory carries (a long context's caches have taken part of
    // the card). The estimate picks the first size to try, and the free memory left after the allocation confirms it.
    bool make_prefill_fit(int tokens, std::string& err) {
        const prefillp::Dims d = m_.prompt_dims();
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
            err = "deepseek41: cannot read the free device memory";
            return false;
        }
        // the reserve stays free through the read: the chunk halves to fit above it
        std::string te;
        const ExpertTier* tier = ops_->expert_tier(te);
        const bool beside = short_prompt_;
        const uint64_t margin = std::max(beside ? prefillp::kShortMargin : prefillp::kMargin, tier ? tier->free_floor() : 0);
        if (beside) tokens = prefillp::kShortChunk;   // these workspaces stay for the next short read, whatever its size
        const prefillp::Fit fit = prefillp::plan_chunk(d, free_b, margin, tokens);
        if (!fit.chunk) {
            err = "deepseek41: " + fit.error;
            return false;
        }
        for (int cap = fit.chunk; cap >= prefillp::kMinChunk; cap /= 2) {
            std::string e;
            prefill_ = make_prefill(m_, *ops_, stream_, cap, e);
            const uint64_t after = prefillp::chunk_bytes(d, cap, tokens) - prefillp::workspace_bytes(d, cap) + margin;
            if (prefill_ && cudaMemGetInfo(&free_b, &total_b) == cudaSuccess && free_b >= after) {
                cudaGetLastError();   // a larger size that did not fit left its error behind
                return true;
            }
            prefill_.reset();
            if (!e.empty()) err = e;
        }
        if (err.empty()) err = "deepseek41: not enough free device memory for the prompt workspaces";
        return false;
    }

    // A prompt that cannot run gives its workspaces back and lets the tier take the memory again, so the engine holds
    // nothing for it; the request fails with `err` and the next one starts clean.
    model::Result prompt_failed(const std::string& err) {
        cudaStreamSynchronize(stream_);
        prefill_.reset();
        short_prompt_ = false;
        std::string e;
        if (ExpertTier* t = ops_->expert_tier(e)) {
            if (!t->reclaim(e)) basalt::log::warn("basalt deepseek41", "VRAM expert tier stays closed: %s", e.c_str());
            t->grow();
        }
        return {model::Status::Error, err};
    }

    void open_prompt_path(int n_tokens) {
        std::string e;
        // the tier opens at its normal size before anything else takes the card; the workspaces size to what is left
        ExpertTier* t = ops_->expert_tier(e);
        if (!t || prefill_) return;
        short_prompt_ = n_tokens <= prefillp::kShortPrompt;   // its workspaces fit beside the copies
        if (!short_prompt_) t->shrink();
    }
    void settle_tier() {
        std::string e;
        if (ExpertTier* t = ops_->expert_tier(e)) t->settle();
    }
    void release_short_prompt() {
        if (!short_prompt_) return;
        cudaStreamSynchronize(stream_);
        prefill_.reset();
        short_prompt_ = false;
    }
    // A short read's workspaces stay for the next one: the tier planned for them.
    void close_prompt_path() {
        std::string e;
        ExpertTier* t = ops_->expert_tier(e);
        if (!t) return;
        if (short_prompt_) return;
        cudaStreamSynchronize(stream_);
        prefill_.reset();
        if (!t->reclaim(e)) basalt::log::warn("basalt deepseek41", "VRAM expert tier stays closed: %s", e.c_str());
        t->grow();
    }

    // A chunk hands the drafter the target features of its last rows: the window a draft after it attends.
    void tap_features() {
        if (!drafter_) return;
        prefill_->layer_tap = [this](int il, const float* x, int T, void* st) {
            const int slot = drafter_->feature_slot(il);
            if (slot < 0) return;
            tap_rows_ = std::min(T, kDsparkWindow);
            const size_t n = (size_t) g_.n_embd;
            dspark_hc_mean(x + (size_t) (T - tap_rows_) * kHcMult * n, tap_rows_, (int) n, dfeat_ + (size_t) slot * n, drafter_->feature_floats(), st);
        };
        prefill_->chunk_done = [this](int T, int64_t pos0, std::string& err) {
            feat_pos0_ = pos0 + T - tap_rows_;
            feat_n_ = tap_rows_;
            tap_rows_ = 0;
            return flush_features(pos0 + T, err);
        };
    }

    // The features of the last rows before `upto` (the accepted ones) go to the drafter.
    bool flush_features(int64_t upto, std::string& err) {
        if (!drafter_ || !feat_n_) return true;
        const int q = (int) std::clamp<int64_t>(upto - feat_pos0_, 0, feat_n_);
        feat_n_ = 0;
        if (!q) return true;
        const size_t ff = (size_t) drafter_->feature_floats();
        if (cudaMemcpyAsync(h_dfeat_, dfeat_, ff * q * sizeof(float), cudaMemcpyDeviceToHost, stream_) != cudaSuccess ||
            cudaStreamSynchronize(stream_) != cudaSuccess) {
            err = "deepseek41: cannot read the drafter's features";
            return false;
        }
        return drafter_->inject(feat_pos0_, q, h_dfeat_, err);
    }

    bool device_bias(float*& dev, const float* host, std::string& err) {
        const size_t bytes = (size_t) g_.n_vocab * sizeof(float);
        if (dev == nullptr) {
            void* v = nullptr;
            if (cudaMalloc(&v, bytes) != cudaSuccess) { err = "out of device memory for a logit bias"; return false; }
            allocs_.push_back(v);
            dev = (float*) v;
        }
        if (cudaMemcpy(dev, host, bytes, cudaMemcpyHostToDevice) != cudaSuccess) { err = "the logit bias upload failed"; return false; }
        return true;
    }

    // How the next pick is drawn: the shared sampler when the request samples, biases or penalizes, else the argmax.
    bool prepare_draw(const model::DecodeRequest& r, std::string& err) {
        history_len_ = 0;
        draw_kernel_ = false;
        if (r.sampling == nullptr) return true;
        sampling_ = *r.sampling;
        sampling_.logit_bias = bias_on_ ? d_bias_ : nullptr;
        history_len_ = r.history != nullptr ? r.history_len : 0;
        draw_kernel_ = !sampling_.greedy || bias_on_ || history_len_ > 0;
        if (!sampling_.greedy && kernels::sampler_list_len(sampling_.top_k, g_.n_vocab) > kernels::kSamplerNarrowMax &&
            !kernels::sampler_wide_reserve(g_.n_vocab, stream_)) {
            err = "out of device memory for the sampler";
            return false;
        }
        if (history_len_ > 0) {
            if (d_hist_ == nullptr) {
                void* v = nullptr;
                if (cudaMalloc(&v, (size_t) kHistoryCap * kMaxWindow * sizeof(int32_t)) != cudaSuccess) { err = "out of device memory for the penalties"; return false; }
                allocs_.push_back(v);
                d_hist_ = (int32_t*) v;
            }
            if (cudaMemcpy(d_hist_, r.history, (size_t) r.n_tokens * history_len_ * sizeof(int32_t), cudaMemcpyHostToDevice) != cudaSuccess) {
                err = "the penalty history upload failed";
                return false;
            }
        }
        return true;
    }

    bool fail(const std::string& s) {
        err_ = s;
        return false;
    }

    const float* wf(int layer, const char* suffix) {
        const DevTensor* t = m_.tensor(layer, suffix);
        return t ? (const float*) t->ptr : nullptr;
    }

    // dev -> file `<dir>/<name>.<layer>.<pos>.bin` for the parity comparison
    void dump(const char* name, int layer, int64_t pos, const void* dev, size_t bytes) {
        if (dump_dir_.empty()) return;
        std::vector<char> h(bytes);
        cudaStreamSynchronize(stream_);
        cudaMemcpy(h.data(), dev, bytes, cudaMemcpyDefault);   // device or mapped host memory
        const std::string path = dump_dir_ + "/" + name + "." + std::to_string(layer) + "." + std::to_string(pos) + ".bin";
        if (FILE* f = std::fopen(path.c_str(), "wb")) {
            std::fwrite(h.data(), 1, bytes, f);
            std::fclose(f);
        }
    }




    bool engram_at(int il) const {
        for (int l : m_.pack().engram.layers)
            if (l == il) return true;
        return false;
    }
    // The Engram projection of layer `il` runs behind the previous layer's head (graphs only); not behind layer 0's,
    // which starts right after the rows were asked for.
    bool engram_ahead(int il) const { return fused_ && il > 1 && engram_at(il); }

    // The layer's work up to the router and the expert inputs' copy to the host, optionally behind the previous
    // layer's tail. Everything here is capturable: positions are read from device memory.
    bool enqueue_head(int il, int64_t pos, int32_t token, void* st, bool with_prev_tail, int T = 1) {
        const size_t n = (size_t) g_.n_embd;
        if (with_prev_tail && !enqueue_tail(il - 1, pos, st, T)) return false;
        if (with_prev_tail && engram_ahead(il)) {
            const std::vector<int>& el = m_.pack().engram.layers;
            for (size_t k = 0; k < el.size(); ++k) {
                if (el[k] != il) continue;
                const DevTensor* gw = m_.tensor(il, "engram_gate.weight");
                if (!gw) return fail("layer " + std::to_string(il) + " has no Engram tensors");
                EngramArgs ea;
                ea.n_embd = (int) n;
                engram_apply(ekv_[k], (const float*) gw->ptr, nullptr, T, ea, x_, st);
            }
        }
        if (drafter_) {
            const int slot = drafter_->feature_slot(il);
            if (slot >= 0) dspark_hc_mean(x_, T, (int) n, dfeat_ + (size_t) slot * n, drafter_->feature_floats(), st);
        }
        const float* pre_prev = il > 0 ? coef_f_[(il - 1) & 1] : nullptr;
        float* coef_f = coef_f_[il & 1];
        HcArgs hca;
        hca.n_embd = (int) n;
        hca.rms_eps = 1e-20f;
        hca.hc_eps = g_.hc_eps;
        hca.sinkhorn_iters = g_.sinkhorn_iters;
        RouterArgs ra;
        ra.n_expert = g_.n_expert;
        ra.n_embd = (int) n;
        ra.k = g_.n_used;
        ra.weight_scale = g_.weight_scale;

        // each sublayer's mixes run beside its collapse: only its expansion reads them
        DS_CUDA(cudaEventRecord(mix_ev_[0], stream_));
        DS_CUDA(cudaStreamWaitEvent(mix_st_, mix_ev_[0], 0));
        hc_mixes(x_, wf(il, "hc_attn_fn.weight"), wf(il, "hc_attn_scale.weight"), wf(il, "hc_attn_base.weight"), T, hca,
                 hc_ws_, coef_a_, mix_st_);
        DS_CUDA(cudaEventRecord(mix_ev_[1], mix_st_));
        if (pre_prev) {
            hc_pre_rms_norm(x_, pre_prev, kHcCoef, wf(il, "attn_norm.weight"), T, (int) n, g_.rms_eps, cur_, normed_, st);
        } else {
            // the collapse of layer 0 is stream 0 of each row
            for (int t = 0; t < T; ++t)
                DS_CUDA(cudaMemcpyAsync(cur_ + (size_t) t * n, x_ + (size_t) t * kHcMult * n, n * sizeof(float),
                                        cudaMemcpyDeviceToDevice, stream_));
            rms_norm(cur_, wf(il, "attn_norm.weight"), T, (int) n, g_.rms_eps, normed_, st);
        }
        dump("hc_attn_pre", il, pos, cur_, n * sizeof(float));
        if (T == 1 ? !ops_->attention(il, pos, token, normed_, attn_out_, st, err_)
                   : !ops_->attention_window(il, T, normed_, attn_out_, st, err_))
            return false;
        dump("attn_out", il, pos, attn_out_, n * sizeof(float));
        DS_CUDA(cudaStreamWaitEvent(stream_, mix_ev_[1], 0));
        hc_post(attn_out_, x_, coef_a_, kHcCoef, T, (int) n, x2_, st);
        dump("hc_attn_post", il, pos, x2_, 4 * n * sizeof(float));

        // FFN sublayer (the collapse uses the attention sublayer's pre-mix: single-pass mHC); its mixes are read by the
        // next layer, so they join at the end of the head
        DS_CUDA(cudaEventRecord(mix_ev_[2], stream_));
        DS_CUDA(cudaStreamWaitEvent(mix_st_, mix_ev_[2], 0));
        hc_mixes(x2_, wf(il, "hc_ffn_fn.weight"), wf(il, "hc_ffn_scale.weight"), wf(il, "hc_ffn_base.weight"), T, hca,
                 hc_ws_, coef_f, mix_st_);
        DS_CUDA(cudaEventRecord(mix_ev_[3], mix_st_));
        hc_pre_rms_norm(x2_, coef_a_, kHcCoef, wf(il, "ffn_norm.weight"), T, (int) n, g_.rms_eps, cur_, normed_, st);
        dump("hc_ffn_pre", il, pos, cur_, n * sizeof(float));
        // the routed experts' rows go out beside the router; the ids follow both
        DS_CUDA(cudaEventRecord(row_ev_[0], stream_));
        DS_CUDA(cudaStreamWaitEvent(row_st_, row_ev_[0], 0));
        ops_->stage_rows(T, normed_, row_st_);
        DS_CUDA(cudaEventRecord(row_ev_[1], row_st_));
        router(normed_, wf(il, "ffn_gate_inp.weight"), wf(il, "exp_probs_b.bias"), T, ra, rlogits_, ids_, rw_, st);
        dump("ffn_moe_logits", il, pos, rlogits_, (size_t) g_.n_expert * sizeof(float));
        dump("ffn_moe_topk", il, pos, ids_, (size_t) g_.n_used * sizeof(int32_t));
        dump("ffn_moe_weights_scaled", il, pos, rw_, (size_t) g_.n_used * sizeof(float));
        DS_CUDA(cudaStreamWaitEvent(stream_, row_ev_[1], 0));
        ops_->stage_ids(T, ids_, st);
        DS_CUDA(cudaStreamWaitEvent(stream_, mix_ev_[3], 0));
        // it needs no routing: it runs while the host finds the routed experts' places
        if (!fused_) return true;
        return T == 1 ? ops_->shared_expert(il, normed_, sh_out_, st, err_) : ops_->shared_expert_window(il, T, normed_, sh_out_, st, err_);
    }

    // Shared expert and its sum with the routed result (unless the head and the experts' call already did both), expansion into
    // the next layer's input.
    bool enqueue_tail(int il, int64_t pos, void* st, int T = 1) {
        const size_t n = (size_t) g_.n_embd;
        dump("ffn_moe_out", il, pos, moe_out_, n * sizeof(float));
        if (!fused_) {
            if (!ops_->shared_expert(il, normed_, sh_out_, st, err_)) return false;
            dump("ffn_shexp", il, pos, sh_out_, n * sizeof(float));
            add(moe_out_, sh_out_, ffn_sum_, (int) n, st);
        }
        hc_post(ffn_sum_, x2_, coef_f_[il & 1], kHcCoef, T, (int) n, x_, st);
        dump("l_out", il, pos, x_, 4 * n * sizeof(float));
        return true;
    }

    // Captures `body` once into a graph and launches it; later calls launch the instance.
    template <class F>
    bool run_graph(cudaGraphExec_t& exec, F&& body) {
        if (!exec) {
            if (cudaStreamBeginCapture(stream_, cudaStreamCaptureModeRelaxed) != cudaSuccess)
                return fail("cannot begin a graph capture");
            const bool ok = body();
            cudaGraph_t g = nullptr;
            const cudaError_t ce = cudaStreamEndCapture(stream_, &g);
            if (!ok || ce != cudaSuccess) {
                if (g) cudaGraphDestroy(g);
                return ok ? fail(std::string("graph capture: ") + cudaGetErrorString(ce)) : false;
            }
            const cudaError_t ci = cudaGraphInstantiateWithFlags(&exec, g, cudaGraphInstantiateFlagUseNodePriority);
            cudaGraphDestroy(g);
            if (ci != cudaSuccess) return fail(std::string("graph instantiate: ") + cudaGetErrorString(ci));
        }
        DS_CUDA(cudaGraphLaunch(exec, stream_));
        return true;
    }

    int graph_row(int T) const { return T - 1; }

    bool run_head(int il, int64_t pos, int32_t token, bool with_prev_tail, int T = 1) {
        return run_graph(head_g_[(((size_t) graph_row(T) * g_.n_layers) + il) * 2 + (with_prev_tail ? 1 : 0)],
                         [&]() { return enqueue_head(il, pos, token, stream_, with_prev_tail, T); });
    }
    bool run_tail(int il, int64_t pos, int T = 1) {
        return run_graph(tail_g_[(size_t) graph_row(T) * g_.n_layers + il], [&]() { return enqueue_tail(il, pos, stream_, T); });
    }

    bool step(const int32_t* tokens, int T, int64_t pos, bool want_out, int32_t* next, float* logits_out, std::string& err) {
        err_.clear();
        bool ok = flush_features(pos, err_) && step_impl(tokens, T, pos, want_out, next, logits_out);
        if (ok && drafter_) {
            feat_pos0_ = pos;
            feat_n_ = T;
        }
        if (!ok) err = err_.empty() ? "deepseek41: step failed" : err_;
        return ok;
    }

    // Rows of a window past `pos` were not accepted: the compressor state keeps only the ones before it. A later position
    // than `pos` drops out of the n-gram history.
    void sync_rewind(int64_t pos) {
        if (pos == 0) whole_from_ = 0;
        if (ops_) ops_->commit_window(pos, stream_);
        if (ops_ && pos == 0) ops_->trim_positions();
        if (pos == 0 && drafter_) drafter_->reset();
        if (pos > 0 && (int64_t) hist_.size() > pos) hist_.resize((size_t) pos);
    }



    // `T` rows at pos..; with T > 1 every row's pick lands in next[T] and the rows stay until a later call drops those past it.
    bool step_impl(const int32_t* tokens, int T, int64_t pos, bool want_out, int32_t* next, float* logits_out) {
        const size_t n = (size_t) g_.n_embd;
        void* st = stream_;
        if (!prefillp::may_resume(pos, (int64_t) hist_.size(), whole_from_)) return fail(kSkippedRows);
        if (T > 1 && !dump_dir_.empty()) return fail("deepseek41: a decode window cannot run with tensor dumps");
        if (pos == 0) {
            hist_.clear();
            whole_from_ = 0;
            ops_->reset();
            if (drafter_) drafter_->reset();
        }
        if ((int64_t) hist_.size() < pos) return fail("positions must be consecutive from 0 (the session keeps one sequence)");
        hist_.resize((size_t) pos);   // a position before the end drops the rejected rows of a window
        hist_.insert(hist_.end(), tokens, tokens + T);
        ops_->begin_token(pos, T);
        {
            std::string se;
            if (!ops_->stage_token(pos, T, st, se)) return fail(se);
        }
        const bool graphs = dump_dir_.empty() && ops_->graph_capable();
        fused_ = graphs;

        // embeddings, each repeated into the four streams
        emb4_.resize(n * kHcMult * T);
        for (int t = 0; t < T; ++t) {
            float* row = emb4_.data() + (size_t) t * kHcMult * n;
            std::string e;
            if (!m_.embedding_row(tokens[t], row, e)) return fail(e);
            for (int s = 1; s < kHcMult; ++s) std::copy(row, row + n, row + (size_t) s * n);
        }
        DS_CUDA(cudaMemcpyAsync(x_, emb4_.data(), emb4_.size() * sizeof(float), cudaMemcpyHostToDevice, stream_));
        dump("hc_init", -1, pos, x_, 4 * n * sizeof(float));

        // Engram rows for every Engram layer, requested now so the SSD reads overlap layer 0
        const EngramInfo& en = m_.pack().engram;
        const int n_cols = en.hash.n_cols();
        std::vector<std::vector<uint8_t>> raw(en.layers.size());
        std::vector<ngram::PleReader::Ticket> tickets(en.layers.size());
        if (!en.layers.empty()) {
            int32_t prev[8] = {0};
            const int np = en.hash.n_gram - 1;
            for (int j = 0; j < np; ++j) {
                const int64_t q = pos - np + j;   // oldest first
                prev[j] = q >= 0 ? hist_[(size_t) q] : 0;
            }
            for (size_t k = 0; k < en.layers.size(); ++k) {
                std::vector<int32_t> ids((size_t) n_cols * T);
                engram_hash(en.hash, (int) k, tokens, T, pos, pos == 0 ? nullptr : prev, ids.data());
                std::vector<uint32_t> rows(ids.begin(), ids.end());
                raw[k].assign((size_t) n_cols * T * en.row_bytes, 0);
                tickets[k] = m_.engram_reader().issue(rows.data(), rows.size(), raw[k].data());
            }
        }

        const int32_t token = tokens[0];
        bool tail_pending = false;
        std::string err;

        // A layer's rows are dequantized into mapped memory that its projection reads, so no copy waits behind the
        // expert copies. The projection of an Engram layer past layer 1 runs one layer ahead, behind that layer's head,
        // while its experts run; the Engram layer's head then only adds the result.
        const auto engram_rows = [&](size_t k) {
            if (!m_.engram_reader().collect(tickets[k], err)) return false;
            const int blocks = 256 / 32;
            for (int r = 0; r < n_cols * T; ++r)
                for (int b = 0; b < blocks; ++b)
                    dequantize_iq4_nl(raw[k].data() + (size_t) r * en.row_bytes + (size_t) b * 18,
                                      erow_h_[k] + (size_t) r * 256 + (size_t) b * 32);
            return true;
        };
        const auto engram_project = [&](size_t k, int il, void* s) {
            const DevTensor* kvw = m_.tensor(il, "engram_kv.weight");
            if (!kvw) return false;
            dense_f32(*kvw, erow_d_[k], eq8_[k], ekv_[k], n_cols * 256, (int) (5 * n), T, s);
            return true;
        };

        for (int il = 0; il < g_.n_layers; ++il) {
            // Engram
            for (size_t k = 0; k < en.layers.size(); ++k) {
                if (en.layers[k] != il) continue;
                if (engram_ahead(il)) {
                    DS_CUDA(cudaStreamWaitEvent(stream_, eng_ev_[k], 0));
                    continue;
                }
                if (tail_pending) {
                    if (!run_tail(il - 1, pos, T)) return false;
                    tail_pending = false;
                }
                const DevTensor* gw = m_.tensor(il, "engram_gate.weight");
                if (!engram_rows(k)) return fail(err);
                dump("engram_rows", il, pos, erow_d_[k], (size_t) n_cols * T * 256 * sizeof(float));
                if (!gw || !engram_project(k, il, st)) return fail("layer " + std::to_string(il) + " has no Engram tensors");
                dump("engram_kv", il, pos, ekv_[k], 5 * n * sizeof(float));
                EngramArgs ea;
                ea.n_embd = (int) n;
                engram_apply(ekv_[k], (const float*) gw->ptr, nullptr, T, ea, x_, st);
                dump("engram_out", il, pos, x_, 4 * n * sizeof(float));
            }

            if (!graphs) {
                if (!enqueue_head(il, pos, token, st, false)) return false;
                if (!ops_->routed_experts_staged(il, normed_, ids_, rw_, moe_out_, st, err_)) return false;
                if (!enqueue_tail(il, pos, st)) return false;
            } else {
                if (!run_head(il, pos, token, tail_pending, T)) return false;
                tail_pending = true;
                for (size_t k = 0; k < en.layers.size(); ++k) {
                    if (en.layers[k] != il + 1 || !engram_ahead(il + 1)) continue;
                    if (!engram_rows(k)) return fail(err);
                    DS_CUDA(cudaEventRecord(eng_go_, stream_));
                    DS_CUDA(cudaStreamWaitEvent(eng_st_, eng_go_, 0));
                    if (!engram_project(k, il + 1, eng_st_)) return fail("layer " + std::to_string(il + 1) + " has no Engram tensors");
                    DS_CUDA(cudaEventRecord(eng_ev_[k], eng_st_));
                }
                if (T == 1 ? !ops_->routed_and_shared_staged(il, normed_, ids_, rw_, ffn_sum_, sh_out_, st, err_)
                           : !ops_->routed_shared_window(il, T, rw_, ffn_sum_, sh_out_, st, err_))
                    return false;
            }
        }
        if (tail_pending && !run_tail(g_.n_layers - 1, pos, T)) return false;
        if (!want_out && !next && !logits_out) {
            ops_->end_window(false);
            DS_CUDA(cudaStreamSynchronize(stream_));
            return true;
        }

        // head: collapse with the last FFN's pre-mix, norm, Q6_K output matrix
        hc_pre(x_, coef_f_[(g_.n_layers - 1) & 1], kHcCoef, T, (int) n, cur_, st);
        dump("hc_out", -1, pos, cur_, n * sizeof(float));
        rms_norm(cur_, (const float*) m_.tensor("output_norm.weight")->ptr, T, (int) n, g_.rms_eps, normed_, st);
        dump("result_norm", -1, pos, normed_, n * sizeof(float));
        const DevTensor* head = m_.tensor("output.weight");
        dense_f32(*head, normed_, q8_, logits_, (int) n, g_.n_vocab, T, st);
        ops_->end_window(false);
        if (T > 1) {
            if (draw_kernel_) kernels::sample_tokens(logits_, T, g_.n_vocab, history_len_ > 0 ? d_hist_ : nullptr, history_len_, sampling_, am_dev_, st);
            else argmax_rows(logits_, T, g_.n_vocab, am_dev_, am_ws_, st);
            DS_CUDA(cudaMemcpyAsync(h_next_, am_dev_, (size_t) T * sizeof(int32_t), cudaMemcpyDeviceToHost, stream_));
            if (logits_out) DS_CUDA(cudaMemcpyAsync(logits_out, logits_, (size_t) T * g_.n_vocab * sizeof(float), cudaMemcpyDeviceToHost, stream_));
            DS_CUDA(cudaStreamSynchronize(stream_));
            std::copy(h_next_, h_next_ + T, next);
            return true;
        }
        if (next && !logits_out && draw_kernel_) {
            kernels::sample_tokens(logits_, 1, g_.n_vocab, history_len_ > 0 ? d_hist_ : nullptr, history_len_, sampling_, am_dev_, st);
            DS_CUDA(cudaMemcpyAsync(h_next_, am_dev_, sizeof(int32_t), cudaMemcpyDeviceToHost, stream_));
            DS_CUDA(cudaStreamSynchronize(stream_));
            *next = *h_next_;
            return true;
        }
        if (next && !logits_out) {
            // greedy: the id comes back, not the logits
            argmax_index(logits_, g_.n_vocab, am_dev_, am_ws_, st);
            DS_CUDA(cudaMemcpyAsync(h_next_, am_dev_, sizeof(int32_t), cudaMemcpyDeviceToHost, stream_));
            DS_CUDA(cudaStreamSynchronize(stream_));
            *next = *h_next_;
            return true;
        }
        std::vector<float> host((size_t) g_.n_vocab);
        DS_CUDA(cudaMemcpyAsync(host.data(), logits_, host.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_));
        DS_CUDA(cudaStreamSynchronize(stream_));
        if (logits_out) std::copy(host.begin(), host.end(), logits_out);
        if (next) {
            size_t best = 0;
            for (size_t i = 1; i < host.size(); ++i)
                if (host[i] > host[best]) best = i;
            *next = (int32_t) best;
        }
        return true;
    }

    DsModel& m_;
    const Geometry& g_;
    std::vector<cudaGraphExec_t> head_g_, tail_g_;
    static constexpr int kHistoryCap = 4096;
    kernels::SamplerParams sampling_{};
    bool draw_kernel_ = false, bias_on_ = false;
    int history_len_ = 0;
    float* d_bias_ = nullptr;
    float* d_mask_ = nullptr;
    int32_t* d_hist_ = nullptr;
    cudaStream_t stream_ = nullptr;
    cudaStream_t mix_st_ = nullptr;   // the hyper-connection mixes, a branch of the head
    cudaEvent_t mix_ev_[4] = {};      // its fork and join points: attention sublayer, FFN sublayer
    cudaStream_t row_st_ = nullptr;   // the routed experts' rows, a branch beside the router
    cudaEvent_t row_ev_[2] = {};      // its fork and join
    std::unique_ptr<LayerOps> ops_;
    std::unique_ptr<Prefill> prefill_;
    bool short_prompt_ = false;   // prefill_ is a short read's and stays between reads
    std::unique_ptr<VisionEncoder> vis_;
    int vis_merge_ = 3, vis_min_pixels_ = 0;
    bool vis_borrowed_ = false;
    std::vector<void*> allocs_;
    std::vector<int32_t> hist_;
    std::string dump_dir_, err_;
    int64_t whole_from_ = 0;   ///< the first position to save at or go back to, after a read that left rows short (prefillp::whole_from)
    static constexpr const char* kSkippedRows = "deepseek41: that position's rows were left short by a prompt read: state is saved or resumed only at a read's stop";
    float *x_ = nullptr, *x2_ = nullptr, *cur_ = nullptr, *normed_ = nullptr, *attn_out_ = nullptr, *moe_out_ = nullptr,
          *sh_out_ = nullptr, *ffn_sum_ = nullptr, *coef_a_ = nullptr, *coef_f_[2] = {nullptr, nullptr},
          *hc_ws_ = nullptr, *rlogits_ = nullptr, *rw_ = nullptr, *logits_ = nullptr;
    // per Engram layer: its dequantized rows (mapped host memory) as the host and the kernels see them, the projection's
    // q8_1 scratch and result; the stream a projection one layer ahead runs on, its start and its end per layer
    std::vector<float*> erow_h_, erow_d_, ekv_;
    std::vector<void*> eq8_;
    cudaStream_t eng_st_ = nullptr;
    cudaEvent_t eng_go_ = nullptr;
    std::vector<cudaEvent_t> eng_ev_;
    int32_t* ids_ = nullptr;
    int32_t *am_dev_ = nullptr, *h_next_ = nullptr;
    void* am_ws_ = nullptr;
    std::vector<float> emb4_;
    bool fused_ = false;
    void* q8_ = nullptr;
    // the drafter and the main model's side of its feature hand-off
    std::unique_ptr<DsparkDrafter> drafter_;
    PromptHost::Image* image_ = nullptr;   ///< this slot's image buffers
    float* dfeat_ = nullptr;     // [rows][target layers][n_embd]
    float* h_dfeat_ = nullptr;
    int64_t feat_pos0_ = 0;      // the first position of the rows in dfeat_ that await the drafter, and how many
    int feat_n_ = 0;
    int tap_rows_ = 0;           // the rows the running prompt chunk leaves in dfeat_
};

}  // namespace

std::unique_ptr<model::ModelSession> make_session_impl(DsModel& model, std::string& err) {
    auto s = std::make_unique<Session>(model);
    if (!s->init(err)) return nullptr;
    return s;
}

}  // namespace basalt::models::deepseek41
