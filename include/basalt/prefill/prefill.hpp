// include/basalt/prefill/prefill.hpp - batched prompt processing.
//
// The prompt's positions [pos0, pos0 + n) are processed in chunks of `chunk` tokens through all 48 layers, leaving
// the session state (GDN recurrence and conv state, QSA KV pools and indexer, PLE history) where the token path
// would have left it; the decode loop then continues with the next token.  Per layer: the projections are
// tensor-core GEMMs (quantized weights dequantized to FP16 on the fly, BF16 weights as they are), the recurrences
// walk the chunk inside one kernel, and the routed experts are grouped by expert: resident ones are read from the
// VRAM tier, the others streamed from the host arena through a pinned ring on a copy stream.
//
// Requires the native weights: every quantized projection must carry its GGUF blocks.
#pragma once

#include "basalt/core/expert_cache.hpp"
#include "basalt/core/expert_source.hpp"
#include "basalt/core/layer.hpp"
#include "basalt/core/session.hpp"
#include "basalt/core/weights.hpp"

#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace basalt::core { class PeerExperts; }
namespace basalt::kernels::cpu { class ExpertPool; }

namespace basalt::prefill {

struct PrefillStats {
    int64_t tokens = 0;
    int64_t chunks = 0;
    double ms_total = 0;
    double ms_experts_host = 0;     ///< host time staging non-resident experts
    int64_t experts_streamed = 0;   ///< expert blobs copied host -> device
    int64_t experts_dma = 0;        ///< ...of which straight from the pinned arena (no CPU copy)
    int64_t experts_resident = 0;   ///< expert-layer groups served from the VRAM tier
    int64_t experts_cpu = 0;        ///< experts computed on the CPU pool instead of streamed
    double cpu_share = 0;           ///< the share of the streamed experts the CPU took last
    double ms_ple = 0;
};

}  // namespace basalt::prefill
namespace basalt::core { class MtpDrafter; }
namespace basalt::prefill {

class Prefill {
public:
    /// The draft layer's K/V for prompt cells [cell0, cell0 + n) from their final residual rows `R_rows`
    /// (device) and `next_tokens` (host: the token at cell+1), in batches through this path's GEMMs and its idle
    /// scratch - call it from on_chunk.  false with `err` empty: not applicable here (a hybrid K/V, another
    /// device, too little scratch), the caller runs the drafter's own pass.  Not bit-identical to that pass: the
    /// drafts may differ, never the target's tokens' logits.
    bool draft_kv(core::MtpDrafter& mtp, const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0,
                  std::string& err);
    Prefill();
    ~Prefill();
    Prefill(const Prefill&) = delete;
    Prefill& operator=(const Prefill&) = delete;

    /// Frees every buffer, stream and event `init` made (as the destructor does) and starts over empty, so `init` can
    /// run again - with a smaller chunk when the first one did not fit.  The stage range (`set_stage`) and the
    /// callbacks stay.  The device `init` ran on must be current.
    void reset();

    /// `host_res`: the static residency table (n_layers x n_expert, slot or -1) or null; `cache` its slots.
    /// `borrow`/`borrow_bytes`: device memory to carve every buffer from (the top slots of the expert cache,
    /// lent for the prompt and refilled after it); null = allocate normally.
    bool init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
              core::ExpertSource* src, const core::ExpertCache* cache, const int32_t* host_res, int64_t chunk,
              void* stream, std::string& err, void* borrow = nullptr, uint64_t borrow_bytes = 0);

    /// With borrowed buffers: lay them out again for chunks of `chunk` tokens (at most `init`'s) in `borrow` - a
    /// request lends only the slots its prompt needs.  The stream must be idle (between prompts).
    bool relayout(int64_t chunk, void* borrow, uint64_t borrow_bytes, std::string& err);
    int64_t chunk() const;
    /// Points the prompt path at another session's state (a view of `ss` with a batch slot's state fields) for the
    /// next runs; the buffers and the layer range stay.
    void set_session(core::SessionState& ss);

    /// The share of the streamed experts' bytes DMA-able straight from pinned RAM (1 = all); sizes the prompt path's
    /// lend cap.
    static void set_pinned_share(double share);

    /// The chunk size from which every expert streams (below it only the routed experts are staged and the peer card
    /// computes the rows of its own); before init, which sizes the buffers by it.  Default 1024.
    static void set_stream_all_min(int64_t tokens);

    /// The streamed ring's depth for the layouts made from now on (`relayout`, `bytes_needed`, `ring_slots_for`); at
    /// most the one `init` laid out.  A shallower ring takes fewer of the lent cache slots.
    static void set_ring_depth(int slots);
    /// The ring depth of the current layout.
    int ring_depth() const;
    static double pinned_share();

    /// Device bytes `init` needs for a chunk of `chunk` tokens (what a borrowed region must hold).
    static uint64_t bytes_needed(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk);

    /// Lays every buffer of a chunk of `chunk` tokens out over a region of exactly `bytes_needed` bytes without
    /// touching device memory (the pointers are never dereferenced) and reports the bytes the layout took; false when
    /// a buffer, or one of the sets that share the scratch region, does not fit.  `with_source`: the layout streams
    /// experts.  For tests of the sizing.
    static bool layout_fits(const core::ModelGeometry& g, core::SessionState& ss, int64_t chunk, bool with_source,
                            uint64_t* used, uint64_t* capacity);

    /// The streamed ring's slot count for a chunk of `chunk` tokens.
    static int64_t ring_slots_for(int64_t chunk);

    /// Positions [pos0, pos0 + n) holding `tokens`; `ss.ple_prev` must be the two tokens before pos0 (oldest
    /// first, -1 for none) and is advanced to the last two of these.
    bool run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err);

    /// For the next `run` (which must start at `run_pos0`): once its first chunk's n-gram rows have landed, the rows of
    /// `tokens[from, to)` (the prompt's later parts) are read into the row cache on a thread, so the runs that read
    /// them find the rows in memory.  Needs `from >= 2`; `to` is cut to what the cache holds.  Cleared by that run.
    void set_ple_warm(const int64_t* tokens, int64_t run_pos0, int64_t from, int64_t to);

    const PrefillStats& stats() const { return stats_; }

    /// Starts counting, per (layer, expert), the routed entries of the prompt rows this stage and the stages after it
    /// read, from zero.  Call with the stream idle (between requests).
    void route_hist_reset();
    /// Adds the counts since `route_hist_reset` (n_layers x n_expert, each stage fills its own layers) into `out`
    /// and the prompt tokens they cover into `tokens`, and stops counting.  False when counting was not started.
    bool route_hist_take(std::vector<uint32_t>& out, int64_t& tokens);

    /// multi-GPU: the experts the peer GPU holds are computed THERE for every prompt chunk (up to `cap_rows` routed
    /// rows per layer; the rest of the peer's experts are read by this GPU over P2P).  Allocates the peer's buffers for
    /// chunks of up to init's `chunk` tokens.  Without P2P: the routed-only chunks, and the streamed ones up to a fixed
    /// size on the FP4 experts (the same bits as this GPU's), with the rows crossing through host memory.
    bool set_peer(core::PeerExperts* peer, int64_t cap_rows, std::string& err);
    /// --peer-exclusive: experts only `peer` holds have no host copy; the prompt path reads them from its memory
    /// over P2P (into the ring, or in place).  Independent of set_peer.
    void set_ring_peer(core::PeerExperts* peer);

    /// The CPU expert pool (idle while a prompt is read). A chunk below stream_all_min() tokens hands it the
    /// non-resident experts with a host copy that few of its tokens route to, instead of streaming them; their rows
    /// go to the tail of Dm unweighted, like the peer's, and the combine weights them. The share is set by
    /// set_cpu_share(). Only for a pool no other thread runs meanwhile. Null: every expert on the GPU.
    void set_cpu_pool(kernels::cpu::ExpertPool* pool);
    /// The share, 0 to 1 (default 0.6), of the streamed experts the CPU computes; 0 leaves them all to the GPU. Fixed
    /// for the run, so the output does not depend on timing.
    void set_cpu_share(double share);

    /// Called after every chunk with the chunk's final multi-stream residual rows (device,
    /// T x hc*n_embd, valid until the next chunk) and the chunk's first position; the MTP draft layer builds its
    /// K/V from them.  The prefill stream is synchronized before the call.
    std::function<bool(const float* R_rows, int64_t T, int64_t pos0, std::string& err)> on_chunk;

    /// Called on the host before each layer of a chunk with the layers done so far, the layer count, the chunk's
    /// length and first position.  No device work is added; the MoE layers' routing readback keeps the host in step
    /// with the GPU.  Not called under a layer split.
    std::function<void(int64_t layers_done, int64_t n_layers, int64_t T, int64_t pos0)> on_layer;

    /// Layer split: called by every stage when it has read a chunk, with the position reached, while its own state
    /// is still at that chunk's end (its stream synchronized; the last stage calls it just before `on_chunk`).  An
    /// earlier stage is a chunk or more ahead of the last one by the time `on_chunk` runs, so this is where a
    /// mid-prompt checkpoint takes each stage's part.  Runs on that stage's thread, with its device current.
    std::function<bool(int64_t done, std::string& err)> on_stage_chunk;

    /// Checked before every chunk: true stops the prompt early (`run` returns false with err "cancelled").
    std::function<bool()> should_stop;

    /// The vision path: HOST rows (n_embd floats) indexed by absolute position, read in place of the token
    /// embedding where non-null (an image's <|image_pad|> cells).  Null (default): every position embeds its token.
    const float* const* embd_rows = nullptr;

    /// LAYER SPLIT (multi-GPU): this prompt path runs layers [layer_begin, layer_end) (-1: to the last) on the
    /// device `init` runs on.  A stage that does not start at layer 0 reads each chunk's residual rows from the
    /// previous stage instead of embedding the tokens; a stage that does not end at the last layer copies its rows
    /// to pinned host buffers (two, allocated by `init`) and runs `next` on them - on a thread, so the next stage
    /// reads chunk c while this one reads chunk c + 1.  `on_chunk` belongs on the last stage.  Set before `init`.
    void set_stage(int64_t layer_begin, int64_t layer_end, Prefill* next) {
        stage_lb_ = layer_begin; stage_le_ = layer_end; next_ = next;
    }


private:
    // Stage-1 pipeline: intermediate stages return after handing their chunk to
    // the direct successor. The public run() drains the chain once at prompt end.
    bool run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err);
    bool drain_pipeline(std::string& err);

    kernels::cpu::ExpertPool* cpu_pool_ = nullptr;
    int64_t stage_lb_ = 0, stage_le_ = -1;
    Prefill* next_ = nullptr;
    bool single_chunk_ = false;         ///< a later stage: the prompt is one chunk (set by the stage before)
    const float* hand_in_ = nullptr;    ///< the previous stage's rows of the chunk being read (host, pinned)

    std::string next_err_;
    std::future<bool> next_run_;
    int hand_buf_ = 0;

    bool carve(std::size_t T, void* alloc);   // the device buffers of a chunk (prefill.cpp's Alloc)
    void release();                          // the destructor's cleanup (also `reset`'s)
    struct Impl;
    std::unique_ptr<Impl> impl_;
    PrefillStats stats_;
};

}  // namespace basalt::prefill
