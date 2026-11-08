// src/models/deepseek41/layer_ops.hpp - the parts of a layer that other kernel groups provide: attention, the routed
// experts and the shared expert. The layer loop calls these; the default implementation reports what is missing.
#pragma once

#include "basalt/core/conversation_cache.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace basalt::models::deepseek41 {

class DsModel;
class ExpertTier;

class LayerOps {
public:
    virtual ~LayerOps() = default;
    /// Attention of `layer` for the token at `pos`. `x` is the normed sublayer input [n_embd] (device); the result
    /// goes to `out` [n_embd] (device), after the output projection.
    virtual bool attention(int layer, int64_t pos, int32_t token, const float* x, float* out, void* stream,
                           std::string& err) = 0;
    /// Routed experts: `ids` [k] and `weights` [k] (device, weights already scaled) over the normed input `x`;
    /// the weighted sum goes to `out` [n_embd] (device). Stages its row and ids itself.
    virtual bool routed_experts(int layer, const float* x, const int32_t* ids, const float* weights, float* out,
                                void* stream, std::string& err) = 0;
    /// The shared expert over `x`; the result goes to `out` [n_embd] (device).
    virtual bool shared_expert(int layer, const float* x, float* out, void* stream, std::string& err) = 0;
    /// routed_experts after stage_rows and stage_ids of the same layer ran on the stream (the default runs routed_experts).
    virtual bool routed_experts_staged(int layer, const float* x, const int32_t* ids, const float* weights, float* out,
                                       void* stream, std::string& err) {
        return routed_experts(layer, x, ids, weights, out, stream, err);
    }
    /// routed_experts_staged plus the shared expert: `out` = routed sum + shared result, with `shared_out` as the
    /// shared expert's scratch. An implementation may run the shared expert while the experts' host phase is in flight.
    virtual bool routed_and_shared_staged(int layer, const float* x, const int32_t* ids, const float* weights,
                                          float* out, float* shared_out, void* stream, std::string& err);
    /// The same three ops over T consecutive prompt tokens at pos0.. (x, out: [T][n_embd]; ids, weights: [T][k]).
    /// The defaults run the single-token ops one after another; an implementation may batch them.
    virtual bool attention_batch(int layer, int64_t pos0, int n_tokens, const int32_t* tokens, const float* x,
                                 float* out, int n_embd, void* stream, std::string& err);
    virtual bool routed_experts_batch(int layer, int64_t pos0, int n_tokens, const float* x, const int32_t* ids,
                                      const float* weights, int k, float* out, int n_embd, void* stream,
                                      std::string& err);
    virtual bool shared_expert_batch(int layer, int64_t pos0, int n_tokens, const float* x, float* out, int n_embd,
                                     void* stream, std::string& err);
    /// Only what later layers read of T tokens' attention at `layer`: the compressed latents and index keys, in place of
    /// attention_batch for rows no later token's attention can reach.
    virtual bool compress_batch(int layer, int64_t pos0, int n_tokens, const float* x, void* stream, std::string& err) {
        (void) layer; (void) pos0; (void) n_tokens; (void) x; (void) stream;
        err = "deepseek41: this layer ops implementation has no batched compressor";
        return false;
    }
    /// The expert tier (VRAM slots and the pinned RAM copy), opened on first use; null when this implementation has none.
    virtual ExpertTier* expert_tier(std::string& err) { (void) err; return nullptr; }
    /// Backs the per-position state (the compressed caches) for positions below `positions`; the layer calls need it
    /// before they reach those positions.
    virtual bool ensure_positions(int64_t positions, std::string& err) {
        (void) positions; (void) err;
        return true;
    }
    /// A new sequence starts: gives back the cache memory a long earlier one left mapped (`ensure_positions` maps it
    /// again as positions arrive).
    virtual void trim_positions() {}
    /// Copies the rolling attention state (window rings, compressor partial groups) to the host, and back.
    /// The compressed caches below `positions`, to and from the host; ensure_positions backs them first.
    virtual size_t cache_bytes(int64_t positions) const { (void) positions; return 0; }
    virtual bool save_cache(void* stream, int64_t positions, std::vector<core::ConversationKv>& kv, std::string& err) { (void) stream; (void) positions; (void) kv; (void) err; return true; }
    virtual bool restore_cache(void* stream, const std::vector<core::ConversationKv>& kv, std::string& err) { (void) stream; (void) kv; (void) err; return true; }
    virtual size_t turn_bytes() const { return 0; }
    virtual void save_turn(void* stream, uint8_t* host) { (void) stream; (void) host; }
    virtual void restore_turn(void* stream, const uint8_t* host) { (void) stream; (void) host; }
    /// Forgets the conversation (a new sequence starts at position 0).
    virtual void reset() {}
    /// Called once per token, or per window of `n_tokens`, before its layers run.
    virtual void begin_token(int64_t pos, int n_tokens = 1) { (void) pos; (void) n_tokens; }
    /// Uploads what the layers read from device memory for the `n_tokens` tokens at `pos` (positions and window bounds),
    /// so the per-layer calls need no host input and can be captured into graphs. Called after begin_token. The rows of
    /// an earlier window before `pos` are kept; the rest are dropped.
    virtual bool stage_token(int64_t pos, int n_tokens, void* stream, std::string& err) {
        (void) pos; (void) n_tokens; (void) stream; (void) err;
        return true;
    }
    /// The rows of the last window before `next_pos` join the carried state; the rest leave no trace.
    virtual void commit_window(int64_t next_pos, void* stream) { (void) next_pos; (void) stream; }
    /// What waits for an idle bus goes now: a few pieces once a token's or window's last launches went out (`all` false),
    /// the rest once the drafter's inputs went out (`all` true); the next token issues what is left.
    virtual void end_window(bool all) { (void) all; }
    /// A verify window: attention of the `n_tokens` staged rows; `x`, `out`: [n_tokens][n_embd].
    virtual bool attention_window(int layer, int n_tokens, const float* x, float* out, void* stream, std::string& err) = 0;
    /// Enqueues what the routed experts read of `n_tokens` normed rows x [n_tokens][n_embd]: their copy in pinned host
    /// memory and the cards' input. stage_ids of the same rows runs after it on the stream or behind it through an event.
    virtual void stage_rows(int n_tokens, const float* x, void* stream) = 0;
    /// Enqueues the copy of the router's expert ids [n_tokens][k] to pinned host memory: once they land, the host reads
    /// the rows too.
    virtual void stage_ids(int n_tokens, const int32_t* ids, void* stream) = 0;
    /// The shared expert over the rows of a window.
    virtual bool shared_expert_window(int layer, int n_tokens, const float* x, float* out, void* stream, std::string& err) = 0;
    /// routed_and_shared_staged for the rows of a window: each distinct expert runs once over the rows that routed to it.
    virtual bool routed_shared_window(int layer, int n_tokens, const float* weights, float* out, float* shared_out, void* stream,
                                      std::string& err) = 0;
    /// True when attention, the staging and shared_expert only enqueue work on the stream (graph capturable).
    virtual bool graph_capable() const { return false; }
};

/// Ops built from the kernel groups' entry points (ops_kernels.cpp).
std::unique_ptr<LayerOps> make_kernel_ops(const DsModel& model, std::string& err);

}  // namespace basalt::models::deepseek41
