// src/models/glm5next/session.hpp - one glm5-next conversation's state: the three caches, a CUDA stream and the
// logits buffer, plus the per-batch forward scratch owned and sized here.
//
// The session does not yet run the forward pass (§5.2 fills decode_window/prefill_chunk in); this layer sizes and
// owns everything the forward needs so plan() can hand a ready session to the engine.
#pragma once

#include "basalt/model/adapter.hpp"
#include "models/glm5next/cache_buf.hpp"

#include <cuda_runtime.h>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace basalt::models::glm5next {

class GlmModel;
class GlmForward;

/// One conversation's state and the forward-pass scratch: the three caches plus a CUDA stream and the logits buffer.
/// The per-batch DSA/attention scratch is sized and owned by the forward pass this session builds on first use.
class GlmSession final : public model::ModelSession {
public:
    GlmSession(GlmModel& model, const CacheDims& dims, uint64_t context, int n_seq, std::string& err);
    ~GlmSession() override;

    GlmSession(const GlmSession&) = delete;
    GlmSession& operator=(const GlmSession&) = delete;

    const GlmModel& model() const { return model_; }
    GlmCaches& caches() { return caches_; }
    int device() const { return device_; }
    cudaStream_t stream() const { return stream_; }
    /// Device logits buffer (n_vocab floats), filled by the forward pass.
    float* logits() { return logits_; }
    const float* logits() const { return logits_; }
    int n_vocab() const { return n_vocab_; }
    uint64_t context() const { return context_; }
    int n_seq() const { return n_seq_; }
    /// The forward pass, built on first use. A pack it cannot run (quantized dense weights, a geometry the kernels
    /// refuse) says so here, once, so loading such a pack still works: the refusal is at the request, not the load.
    GlmForward* forward(std::string& err);
    /// A window of new tokens, greedy: the last row's logits and its argmax, or one pick per row when the engine asks
    /// for them - which is how it checks the rows of a drafted window.
    model::Result decode_window(const model::DecodeRequest& r) override;
    /// One row of the last window's head logits, for `logprobs=K` and the ppl gate (decode_loop.hpp).
    model::Result copy_logits(int row, float* out) override;
    /// A chunk of a prompt. A read longer than the free device memory carries is run as several windows, the caches
    /// holding everything the next one needs; `rows_after` and `more` therefore ask for nothing here.
    model::Result prefill_chunk(const model::PrefillRequest& r) override;
    /// The request's per-token biases, added to the logits before the pick. The engine calls this on every request,
    /// with null when the request carries none, so the inherited refusal would stop every chat at its first token.
    model::Result set_logit_bias(const float* bias) override;
    /// Two rows when the pack's next-n block runs as a drafter: the token the model already picked beside the one the
    /// drafter guessed after it.
    int max_window() const override;
    /// The guess. The block runs here, on the hidden and the indexer's cells the last pass left behind; `draft_end`
    /// only hands the token and its chance back.
    model::Result draft_begin(int32_t anchor, int64_t pos, int n) override;
    model::Result draft_end(int32_t* ids, float* conf, int max_n, int& n, const DraftNeed& need) override;
    /// What the expert tier has served since the session opened: the serve reporting layer's source for the stats
    /// line's expert split and for the `DONE` line a client counts decode experts from (`tools/ppl_gate.py`). The
    /// inherited answer is zeros, which reads as a request that touched no experts.
    ExpertCounts expert_counts() const override;

private:
    /// How many tokens a window can carry in the device memory free right now: the plan's chunk for that much memory,
    /// which is itself clamped to the context and to the plan's ceiling. `INT_MAX` when the free-memory query itself
    /// fails, which refuses nothing.
    int window_cap();
    /// Refused when a window of `n_tokens` cannot fit in the device memory free right now, with what would fit in the
    /// message. Ok otherwise, including when the free-memory query itself fails.
    model::Result check_fits(int n_tokens);

    GlmModel& model_;
    int device_;
    cudaStream_t stream_;
    CacheDims dims_;
    GlmCaches caches_;
    uint64_t context_ = 0;
    int n_seq_ = 0;
    int n_vocab_ = 0;
    float* logits_ = nullptr;
    int32_t* pick_ = nullptr;     ///< device: the token the sampler chain chose for the window's last row
    float* d_bias_ = nullptr;     ///< device: the request's [n_vocab] bias, added by the sampler
    std::vector<float> host_bias_;  ///< the same bias for the greedy argmax, which never enters a kernel
    bool bias_on_ = false;
    uint64_t draws_ = 0;          ///< absolute draw index, which Philox counts from
    std::unique_ptr<GlmForward> forward_;
    std::string fwd_err_;   ///< why the forward pass cannot run here, once it has said so
    int32_t draft_id_ = -1;   ///< the block's guess, between draft_begin and draft_end
    float draft_conf_ = 0.f;  ///< ... and its probability under the drafter's head
    bool draft_ready_ = false;
};

/// Factory mirroring ds41's `make_session_impl`: builds one ready session, or null with `err` set.
std::unique_ptr<model::ModelSession> make_session_impl(GlmModel& model, const CacheDims& dims,
                                                       uint64_t context, int n_seq, std::string& err);

}  // namespace basalt::models::glm5next
