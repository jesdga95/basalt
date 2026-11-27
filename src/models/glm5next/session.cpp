// src/models/glm5next/session.cpp - one glm5-next conversation's state: the three caches, a CUDA stream and the
// logits buffer, and the forward pass the session runs a request through.
//
// The session owns the caches and the logits; the per-window scratch belongs to the forward pass it builds on first
// use, so a pack the pass cannot run still loads and says so when a request arrives.
#include "models/glm5next/kernels/cell_expand.hpp"
#include "models/glm5next/kernels/mla_sparse.hpp"
#include "models/glm5next/session.hpp"

#include "basalt/log.hpp"
#include "basalt/core/expert_tier.hpp"          // lend(), the card room a window borrows back from the tier's copies
#include "basalt/kernels/sampler.hpp"
#include "models/glm5next/forward.hpp"
#include "models/glm5next/model.hpp"
#include "models/glm5next/prefill_plan.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace basalt::models::glm5next {

GlmSession::GlmSession(GlmModel& model, const CacheDims& dims, uint64_t context, int n_seq, std::string& err)
    : model_(model), device_(model.device()), dims_(dims), context_(context), n_seq_(n_seq), n_vocab_(model.desc().n_vocab) {
    if (cudaStreamCreate(&stream_) != cudaSuccess) {
        err = "glm5next session: cannot create CUDA stream";
        return;
    }
    if (!caches_.alloc(dims, context, n_seq, err)) return;
    const size_t n_logits = (size_t) n_vocab_;
    const cudaError_t e = cudaMalloc(reinterpret_cast<void**>(&logits_), n_logits * sizeof(float));
    if (e != cudaSuccess) {
        err = "glm5next session: out of device memory for logits";
        return;
    }
    if (cudaMalloc(reinterpret_cast<void**>(&pick_), (size_t) kExpertRows * sizeof(int32_t)) != cudaSuccess) {
        err = "glm5next session: out of device memory for the sampled pick";
        return;
    }
    basalt::log::info("basalt glm5next session", "context=%llu n_seq=%d caches=%llu bytes", (unsigned long long) context,
                      n_seq, (unsigned long long) caches_.bytes());
}

GlmSession::~GlmSession() {
    if (logits_) cudaFree(logits_);
    if (pick_) cudaFree(pick_);
    if (d_bias_) cudaFree(d_bias_);
    if (stream_) cudaStreamDestroy(stream_);
    caches_.free();
}

GlmForward* GlmSession::forward(std::string& err) {
    if (fwd_err_.size()) { err = fwd_err_; return nullptr; }
    if (!forward_) {
        std::string e;
        auto f = std::make_unique<GlmForward>(*this, e);
        if (!e.empty()) { fwd_err_ = e; err = e; return nullptr; }
        forward_ = std::move(f);
    }
    return forward_.get();
}

namespace {
/// The scratch the sparse attention adds per token: the one part of a window's cost the plan cannot size, because the
/// number belongs to that kernel (the contract prefill_plan.hpp spells out). Leaving it out under-books the window.
uint64_t mla_scratch_units(const Geometry& g, uint64_t context) {
    const int kp = g.kpool > 0 ? g.kpool : 1;
    const int n_top = std::max(1, std::min((int) (context / (uint64_t) kp), g.idx_top_k / kp));
    return mla_sparse_workspace_floats(1, g.n_head, cell_expand_n_sel(g.kpool, n_top, true));
}
}  // namespace

/// A window the free device memory cannot carry is refused with the number that would fit, rather than failing inside
/// cudaMalloc halfway through a request. The caches are already allocated, so what is free now is what a window has,
/// less what the pass holds outside one - the memory plan() made the expert tier promise to leave (model.cpp). When
/// something else spent it anyway, the tier's card copies are the room to ask for: `lend` gives up the fewest slots,
/// and a prompt this wide routes to most of a layer's experts and reaches them in RAM either way.
int GlmSession::window_cap() {
    const Geometry& g = model_.geo();
    const uint64_t extra = mla_scratch_units(g, context_), own = model_.footprint_bytes();
    // One token's worth of window, the plan's own cost plus the attention's workspace, and the margin the plan leaves.
    const uint64_t one = prefillp::window_bytes(g, context_, 1) + 4ull * extra;
    for (int pass = 0;; ++pass) {
        size_t free_bytes = 0, total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) return std::numeric_limits<int>::max();
        const int cap = prefillp::max_chunk(g, context_, free_bytes > own ? free_bytes - own : 0, extra);
        if (cap > 0) return cap;
        if (pass) {
            basalt::log::warn("basalt glm5next session",
                              "no window fits: %.0f MiB free, the pass holds %.0f MiB outside a window and a token's "
                              "window costs %.0f MiB",
                              free_bytes / 1048576.0, own / 1048576.0, one / 1048576.0);
            return 0;
        }
        core::ExpertTier* t = model_.tier();
        if (!t) return 0;
        const uint64_t need = own + one + prefillp::kMargin > free_bytes ? own + one + prefillp::kMargin - free_bytes : 1;
        if (!t->partial_lend()) {
            basalt::log::warn("basalt glm5next session",
                              "%.0f MiB short of a window, and the expert tier's card arena is one allocation it cannot "
                              "lend part of",
                              need / 1048576.0);
            return 0;
        }
        if (!t->lend(need)) {
            basalt::log::warn("basalt glm5next session",
                              "%.0f MiB short of a window, and the expert tier cannot give up that much of its card copies",
                              need / 1048576.0);
            return 0;
        }
        basalt::log::warn("basalt glm5next session",
                          "the expert tier gives up %.0f MiB of card copies so a window fits; fewer experts stay on the card",
                          need / 1048576.0);
    }
}

model::Result GlmSession::check_fits(int n_tokens) {
    const int cap = window_cap();
    if (n_tokens <= cap) return {model::Status::Ok, ""};
    return {model::Status::Refused,
            "glm5next: a window of " + std::to_string(n_tokens) +
                " tokens does not fit in the free device memory; about " + std::to_string(cap) + " fits right now"};
}

/// The request's per-token biases. Null turns them off, which is what a request without `logit_bias` asks for - and
/// the engine asks on every request, so this has to answer rather than refuse. A bias of -inf bans its token.
model::Result GlmSession::set_logit_bias(const float* bias) {
    bias_on_ = bias != nullptr;
    if (!bias_on_) return {};
    const size_t bytes = (size_t) n_vocab_ * sizeof(float);
    if (!d_bias_) {
        void* v = nullptr;
        if (cudaMalloc(&v, bytes) != cudaSuccess)
            return {model::Status::Error, "glm5next: out of device memory for a logit bias"};
        d_bias_ = (float*) v;
    }
    // on this session's stream: a plain cudaMemcpy does not order against it, so a draw could read the previous bias
    if (cudaMemcpyAsync(d_bias_, bias, bytes, cudaMemcpyHostToDevice, stream_) != cudaSuccess)
        return {model::Status::Error, "glm5next: the logit bias upload failed"};
    host_bias_.assign(bias, bias + n_vocab_);
    return {};
}

/// Whether this session drafts at all. `--spec` is asked first; then the block, the shared indexer and the tier are the
/// model's answer; whether the pass could build itself around them is the forward's, and `draft_begin` asks it there.
int GlmSession::max_window() const {
    if (!model_.wide_windows_asked()) return 1;   // --spec governs the window, as it does for every drafter
    const Geometry& g = model_.geo();
    // The tier of the block's own experts is not asked for here: without it `draft_begin` says the block cannot draft,
    // which the generate loop takes as a window of one row, and an oracle's tokens need no tier at all.
    return g.nextn > 0 && g.index_share_mtp ? 2 : 1;
}

model::ModelSession::ExpertCounts GlmSession::expert_counts() const {
    // The tier counts from the first lookup it served, so these are totals: the serve layer deltas them against what
    // it read when the request started. The drafter's tier is not folded in - the same choice the deepseek41 and
    // hy4flash sessions make - because a draft is one row, and its experts would blur the per-request split.
    core::ExpertTier* t = model_.tier();
    if (t == nullptr) return {};
    return {t->hits(), t->lookups(), t->offloaded(), t->file_lookups()};
}

std::vector<model::ModelSession::CardExperts> GlmSession::experts_by_card() const {
    // One line per card the tier keeps a card arena on: the main card and, with `--gpu-mode peer`, each second card.
    // The drafter's tier keeps arenas of its own and is not folded in, as in the deepseek41 and hy4flash sessions,
    // which report the one tier their sessions have.
    std::vector<CardExperts> out;
    if (core::ExpertTier* t = model_.tier())
        for (const core::ExpertTier::CardSlots& c : t->slots_by_card()) out.push_back({c.device, c.slots, c.bytes});
    return out;
}

bool GlmSession::expert_map(int64_t& total, std::vector<uint8_t>& homes, std::vector<uint32_t>& routed) {
    core::ExpertTier* t = model_.tier();
    if (t == nullptr) return false;
    t->expert_map(total, homes, routed);
    return true;
}

model::Result GlmSession::draft_begin(int32_t anchor, int64_t pos, int n) {
    draft_ready_ = false;
    if (max_window() < 2 || n < 1) return {};
    std::string err;
    GlmForward* f = forward(err);
    if (!f) return {model::Status::Error, err};
    if (!f->drafts())
        return {model::Status::NotImplemented, "glm5next: this pack's next-n block cannot run as a drafter"};
    int32_t id = -1;
    float conf = 0.f;
    if (!f->draft(anchor, pos, id, conf, err)) return {model::Status::Error, err};
    draft_id_ = id;
    draft_conf_ = conf;
    draft_ready_ = true;
    return {};
}

model::Result GlmSession::draft_end(int32_t* ids, float* conf, int max_n, int& n, const DraftNeed&) {
    n = 0;
    if (!draft_ready_) return {};   // draft_begin refused, or never came: no drafts, nothing to verify
    n = std::min(max_n, 1);         // one next-n block, one draft
    if (n > 0) {
        ids[0] = draft_id_;
        if (conf) conf[0] = draft_conf_;
    }
    draft_ready_ = false;
    return {};
}

/// One row of the last window's head logits, the way the deepseek41 session does it: whatever the sampler
/// read, never a recomputed head. The engine reads rows through this for `logprobs=K` and for every row of
/// a ppl-gate window, so a session without it fails those requests with the adapter's "this model keeps no
/// logits" - which is what the GLM gate hit at its first chunk.
model::Result GlmSession::copy_logits(int row, float* out) {
    // The pass has to exist already: the first window creates it, and a window is the only thing that leaves
    // logits worth reading. Without this the call below would build a pass and hand back its empty buffers.
    if (!forward_) return {model::Status::Error, "glm5next: no window has run, so there are no logits to read"};
    std::string err;
    GlmForward* f = forward(err);
    if (!f) return {model::Status::Error, err};
    const int rows = f->logit_rows();   // 1 when the window's last row went to the session's own buffer
    if (row < 0 || row >= rows)
        return {model::Status::Error, "glm5next: row " + std::to_string(row) +
                                          " is outside the last window's " + std::to_string(rows) + " rows of logits"};
    if (cudaStreamSynchronize(stream_) != cudaSuccess)
        return {model::Status::Error, "glm5next: the logits read could not wait for the window"};
    if (rows == 1) {
        if (cudaMemcpy(out, logits_, (size_t) n_vocab_ * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
            return {model::Status::Error, "glm5next: the head logits copy failed"};
        return {};
    }
    std::vector<float> host((size_t) rows * n_vocab_);
    if (!f->copy_row_logits(host.data(), err)) return {model::Status::Error, err};
    std::memcpy(out, host.data() + (size_t) row * n_vocab_, (size_t) n_vocab_ * sizeof(float));
    return {};
}

/// The pick after the window's last row, or after each of its rows when the engine asks for them one by one - which is
/// how it checks a drafted window's drafts.
model::Result GlmSession::decode_window(const model::DecodeRequest& r) {
    if (r.n_tokens <= 0) return {model::Status::Refused, "glm5next: an empty decode window"};
    if (r.sampling && r.sampling->penalty_last_n > 0) {
        return {model::Status::NotImplemented,
                "glm5next: repetition penalties need the window's token history, which this session does not keep"};
    }
    // One pick per row needs the rows' logits, so their rows all go through the output projection. kExpertRows is the
    // row count the session's own buffers are sized for; no drafter this model runs asks for more.
    const int rows = r.row_tokens ? r.n_tokens : 1;
    if (rows > kExpertRows)
        return {model::Status::Refused,
                "glm5next: a drafted window of " + std::to_string(r.n_tokens) + " rows; at most " +
                    std::to_string(kExpertRows) + " are checked row by row"};
    if (model::Result fit = check_fits(r.n_tokens); !fit.ok()) return fit;
    // Same rule as prefill_chunk: position 0 of a sequence is a new conversation, and the engine keeps this session
    // across them. The tiny pack's pass test runs position 0 twice in one session for exactly this reason. The
    // drafter's kept tail is not part of it: `mtp_keep_row` selects a row by absolute position, and the window this
    // call is about rewrites `mtp_prev_pos_` and the rows with its own tail, so no window can read the tail of the
    // conversation before it - which is why this model needs no `drafter_->reset()` at the place deepseek41 needs one.
    if (r.pos == 0) caches_.reset(0, stream_);
    std::string err;
    GlmForward* f = forward(err);
    if (!f) return {model::Status::Error, err};
    if (!f->run(0, r.tokens, r.n_tokens, r.pos, err, rows)) return {model::Status::Error, err};
    std::vector<int32_t> picks((size_t) rows, -1);
    const bool sampled = r.sampling && !r.sampling->greedy;
    if (sampled) {
        // The chain the engine asked for, on each row. The logits and the picks are both device buffers, which is what
        // sample_tokens requires of them.
        kernels::SamplerParams sp = *r.sampling;
        sp.counter = draws_++;
        sp.logit_bias = bias_on_ ? d_bias_ : nullptr;
        kernels::sample_tokens(rows > 1 ? f->row_logits() : logits_, rows, n_vocab_, nullptr, 0, sp, pick_, stream_);
        if (cudaGetLastError() != cudaSuccess ||
            cudaMemcpyAsync(picks.data(), pick_, (size_t) rows * sizeof(int32_t), cudaMemcpyDeviceToHost, stream_) !=
                cudaSuccess ||
            cudaStreamSynchronize(stream_) != cudaSuccess) {
            return {model::Status::Error, "glm5next: the sampled pick failed"};
        }
        // The engine may still want the row's logits; they are only worth copying when it does. A drafted window kept
        // its rows in the pass's own buffer, not the session's single-row one.
        if (r.logits) {
            const float* src = rows == 1 ? logits_ : f->row_logits() + (size_t) (rows - 1) * n_vocab_;
            if (cudaMemcpy(r.logits, src, (size_t) n_vocab_ * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
                return {model::Status::Error, "glm5next: the logits copy failed"};
        }
    } else {
        // Greedy, with the grammar's bias added before the argmax. One host copy holds the rows, the last of them
        // last; `logits` asks for one row only, so the rows land in a buffer of our own and its last one goes over.
        std::vector<float> host;
        float* out = rows == 1 ? r.logits : nullptr;
        if (!out) {
            host.resize((size_t) rows * n_vocab_);
            out = host.data();
        }
        const bool got = rows == 1 ? f->copy_logits(out, err) : f->copy_row_logits(out, err);
        if (!got) return {model::Status::Error, err};
        if (rows > 1 && r.logits)
            std::memcpy(r.logits, out + (size_t) (rows - 1) * n_vocab_, (size_t) n_vocab_ * sizeof(float));
        for (int row_i = 0; row_i < rows; ++row_i) {
            const float* lg = out + (size_t) row_i * n_vocab_;
            int best = 0;
            float best_v = lg[0] + (bias_on_ ? host_bias_[0] : 0.0f);
            for (int v = 1; v < n_vocab_; ++v) {
                const float biased = lg[v] + (bias_on_ ? host_bias_[v] : 0.0f);
                if (biased > best_v) {
                    best_v = biased;
                    best = v;
                }
            }
            picks[(size_t) row_i] = best;
        }
    }
    if (r.row_tokens)
        for (int i = 0; i < rows; ++i) r.row_tokens[i] = picks[(size_t) i];
    if (r.next_token) *r.next_token = picks[(size_t) rows - 1];
    return {model::Status::Ok, ""};
}

/// A prompt as one or more windows. The caches keep every layer's state for the whole context, so a window only needs
/// the tokens and the position it starts at: a read longer than the free device memory carries is run in windows of
/// what does fit rather than refused, which is what `prefillp::max_chunk` exists to say. `more` and `rows_after` ask
/// for nothing here for the same reason the caches keep: the next window starts where this one stopped.
model::Result GlmSession::prefill_chunk(const model::PrefillRequest& r) {
    if (r.n_tokens <= 0) return {model::Status::Refused, "glm5next: an empty prompt chunk"};
    std::string err;
    GlmForward* f = forward(err);
    if (!f) return {model::Status::Error, err};
    const int cap = window_cap();
    if (cap <= 0)
        return {model::Status::Refused,
                "glm5next: the free device memory does not carry even one prompt token's window; the glm5next session "
                "log says what the card has left and what the pass needs of it"};
    // A prompt starting at position 0 is a new conversation, and the engine keeps this session across them with no
    // checkpoint to restore, so the KDA state and conv window are the one thing that cannot start clean by itself.
    // Carried over, a conversation's first tokens run on the previous one's state; a later part of the same read has
    // pos past 0 and is left alone.
    if (r.pos == 0) caches_.reset(0, stream_);
    for (int done = 0; done < r.n_tokens; done += cap) {
        const int piece = std::min(cap, r.n_tokens - done);
        if (!f->run(0, r.tokens + done, piece, r.pos + done, err)) return {model::Status::Error, err};
        if (r.progress) r.progress(done + piece);
        // The read ends here and cannot be continued, which is all `stop` asks of a model that finishes a window.
        if (r.stop && r.stop()) return {model::Status::Ok, ""};
    }
    return {model::Status::Ok, ""};
}

std::unique_ptr<model::ModelSession> make_session_impl(GlmModel& model, const CacheDims& dims,
                                                       uint64_t context, int n_seq, std::string& err) {
    auto s = std::make_unique<GlmSession>(model, dims, context, n_seq, err);
    // Any allocation failure above leaves err set and a partially built session; the caller treats non-ok err as null.
    if (!err.empty()) return nullptr;
    return s;
}

}  // namespace basalt::models::glm5next
