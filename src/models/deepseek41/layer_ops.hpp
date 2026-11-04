// src/models/deepseek41/layer_ops.hpp - the parts of a layer that other kernel groups provide: attention, the routed
// experts and the shared expert. The layer loop calls these; the default implementation reports what is missing.
#pragma once

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
    /// the weighted sum goes to `out` [n_embd] (device). Follows routed_stage of the same layer on the stream.
    virtual bool routed_experts(int layer, const float* x, const int32_t* ids, const float* weights, float* out,
                                void* stream, std::string& err) = 0;
    /// The shared expert over `x`; the result goes to `out` [n_embd] (device).
    virtual bool shared_expert(int layer, const float* x, float* out, void* stream, std::string& err) = 0;
    /// routed_experts after routed_stage of the same layer ran on the stream (the default stages nothing).
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
    /// Forgets the conversation (a new sequence starts at position 0).
    virtual void reset() {}
    /// Called once per token before its layers run.
    virtual void begin_token(int64_t pos) { (void) pos; }
    /// Uploads what the layers read from device memory for the token at `pos` (positions and window bounds), so the
    /// per-layer calls need no host input and can be captured into graphs. Called after begin_token.
    virtual bool stage_token(int64_t pos, void* stream, std::string& err) {
        (void) pos; (void) stream; (void) err;
        return true;
    }
    /// True when attention, routed_stage and shared_expert only enqueue work on the stream (graph capturable).
    virtual bool graph_capable() const { return false; }
    /// A decode window of T tokens at pos0..: the verify pass of a speculative step. Rows past the next call's position
    /// are rolled back by that call (see commit_window). The default implementation has no window path.
    virtual bool window_supported() const { return false; }
    /// Folds the state of the last window into the conversation: tokens before `next_pos` stay, the rest are dropped.
    virtual void commit_window(int64_t next_pos, void* stream) { (void) next_pos; (void) stream; }
    /// Uploads positions and window bounds of the T tokens at pos0.. (after begin_token for each).
    virtual bool stage_window(int64_t pos0, int T, void* stream, std::string& err) {
        (void) pos0; (void) T; (void) stream; (void) err;
        return false;
    }
    /// attention() for the staged window: x and out are [T][n_embd].
    virtual bool attention_window(int layer, int T, const float* x, float* out, void* stream, std::string& err) {
        (void) layer; (void) T; (void) x; (void) out; (void) stream;
        err = "deepseek41: this layer ops implementation has no window attention";
        return false;
    }
    /// routed_stage() for T rows: ids [T][k], x [T][n_embd] (device) go to pinned host memory.
    virtual void routed_stage_window(int T, const float* x, const int32_t* ids, void* stream) {
        (void) T; (void) x; (void) ids; (void) stream;
    }
    /// routed_and_shared_staged() for T rows: weights [T][k]; out and shared_out are [T][n_embd].
    virtual bool routed_shared_window(int layer, int T, const float* x, const float* weights, float* out,
                                      float* shared_out, void* stream, std::string& err) {
        (void) layer; (void) T; (void) x; (void) weights; (void) out; (void) shared_out; (void) stream;
        err = "deepseek41: this layer ops implementation has no window experts";
        return false;
    }
    /// Enqueues the copy of the router's expert ids and the activation to pinned host memory, for routed_experts.
    virtual void routed_stage(int layer, const float* x, const int32_t* ids, void* stream) {
        (void) layer; (void) x; (void) ids; (void) stream;
    }
};

/// The implementation used until the kernels are wired: every call fails with a message naming the missing piece.
std::unique_ptr<LayerOps> make_unavailable_ops();
/// Ops built from the kernel groups' entry points (src/models/deepseek41/ops_kernels.cpp). `fixture` is a diagnostic:
/// when non-empty, the routed and shared expert outputs are replayed from that recorded fixture (fixture_ops.cpp).
std::unique_ptr<LayerOps> make_kernel_ops(const DsModel& model, const std::string& fixture, std::string& err);
/// Dev only: replays recorded outputs; `which` selects what it serves (attention too when `with_attention`).
std::unique_ptr<LayerOps> make_fixture_ops(const DsModel& model, const std::string& fixture, std::string& err);

}  // namespace basalt::models::deepseek41
