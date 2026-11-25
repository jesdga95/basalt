// src/models/inkling2/dspark.hpp - Inkling 2's DSpark drafter (the checkpoint's mtp.safetensors, the pack's dspark.*
// tensors): three dense layers over a block of the anchor and seven mask rows, reading context keys and values made from
// the target's features, then a recurrent Markov head over the target's head. The inference reference is vLLM's
// inkling2 DSpark draft (vllm/models/inkling2/nvidia/dspark.py); llama.cpp's DSpark graph for Inkling 2 agrees.
//
// What the target leaves (the tap): for every position it runs, the inputs of its visits 5, 23 and 41 (in walk order: layer
// 5, the loop's layer 11 in pass 1, its layer 17 in pass 2) and its final normed state. Every row of every forward goes in
// (commit), accepted or not: a block overwrites the cells of the positions it covers, so a rejected row's are never read.
//
//   context   mix_l = rms(sum_k softmax(fuse[l])_k f_k) * dflare_norm[l] for draft layer l, and the Markov head's
//             f = rms(sum_k softmax(fuse[3])_k f_k) * markov.dflare_norm. Position p's keys and values in layer l:
//             k = rope(rms_head(k_ctx mix_l) * k_ctx_norm, p), v = v_ctx mix_l, into the layer's cache at cell p.
//   block     anchor `a` (the target's pick, not yet run) at position p, rows p .. p + 7: x0 = rms(embed(a)) * embed_norm
//             (the target's normed embedding), x1..7 = mask_embed. Each layer: y = attn_norm(x); q, k, v of y, q and k
//             normed per head (q_norm, k_norm) and rotated at their rows' positions (NeoX, theta 1e6); the block's keys and
//             values go to the cache at cells p .. p + 7; every row attends the layer's context ([p - window, p) in a
//             sliding-window layer, [0, p) in the full one) and all eight block rows, scale 1/sqrt(128); x += o_proj;
//             x += down(silu(gate(z)) * up(z)), z = post_attention_norm(x). h = norm(x).
//   head      draft i (from block row i + 1) in turn: c_i = rnn_b + W_h h_{i+1} + W_f f (f of the row at p - 1);
//             raw = c_i + W_sp [state ; w1[prev]] (state 0 and prev = a before draft 0); gate = sigmoid(raw[0:256]),
//             state = gate * state + (1 - gate) * tanh(raw[256:512]), latent = tanh(raw[512:768]);
//             logits = softcap30(head(h_{i+1}) + fold latent + fold_b), fold = head x hidden_out and fold_b = head x
//             hidden_out's bias made once at load; the draft is the first argmax and its chance the softmax probability.
//             The argmax is screened by 2-bit copies of the head and the fold (dspark_head_pick): the head term of all
//             the drafts in one pass over its copy, the fold term per draft, the rows near the best rescored exactly.
//
// Numerics: the projections are the pack's BF16 rounded to Q8_0 as they load; the block's (q, k, v, o, gate, up, down) are
// then rounded on to NVFP4 planes and run on the window gemm over bf16 rows (fused: q, k and v one call, gate and up one
// with SwiGLU, o and down adding into the residual), the context's k and v stay Q8_0 over q8_1 rows; the head is the
// target's own Q8_0 head, the recurrence's small products stay BF16 with fp32 activations, and the rest is fp32. The
// reference rounds to bf16 at every step; a draft only changes the speed (the target verifies every row).
#pragma once

#include "basalt/core/checkpoint_regions.hpp"
#include "basalt/core/conversation_cache.hpp"
#include "basalt/core/elastic_array.hpp"
#include "basalt/core/loader.hpp"
#include "basalt/model/adapter.hpp"
#include "basalt/models/inkling2/dspark_kernels.hpp"
#include "basalt/models/inkling2/kernels.hpp"
#include "basalt/kernels/window_gemm.hpp"
#include "basalt/program/options.hpp"
#include "models/inkling2/pack.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace basalt::prefill::mmq {
class Context;
}

namespace basalt::models::inkling2 {

constexpr int kDsMaxLayers = 4;

struct DsparkLayer {
    void *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr;            ///< NVFP4 planes (Q8_0 until finish())
    void *gate = nullptr, *up = nullptr, *down = nullptr;                        ///< NVFP4 planes (Q8_0 until finish())
    float* fp4_scale = nullptr;                                                  ///< the planes' FP32 scales [7]: q k v o gate up down
    void *k_ctx = nullptr, *v_ctx = nullptr;                                     ///< Q8_0 (prompt MMQ)
    void *k_ctx_p = nullptr, *v_ctx_p = nullptr;                                 ///< Q8 planes (1..32-row decode commit)
    float *attn_norm = nullptr, *post_norm = nullptr, *dflare_norm = nullptr;    ///< [width]
    float *q_norm = nullptr, *k_norm = nullptr, *k_ctx_norm = nullptr;           ///< [head width]
    bool local = false;
};

/// The drafter's weights on the main card, and what it shares with the target.
class DsparkWeights {
public:
    ~DsparkWeights();
    /// After the start's reads have landed: the recurrence's projection split, the head folds made from the target's
    /// Q8_0 head (`head`, n_vocab rows), and the target's embedding (BF16 rows, device-readable) and embedding norm noted.
    model::Result finish(const void* head, const uint16_t* embedding, const float* embed_norm, void* stream);
    uint64_t bytes() const { return bytes_; }
    DsHead head_args() const;

    DsparkGeometry geo;
    int n_vocab = 0;
    float logit_softcap = 30.f;
    std::vector<DsparkLayer> layer;
    float *norm = nullptr, *mask = nullptr, *markov_norm = nullptr, *rnn_b = nullptr, *out_b = nullptr;
    float* alpha = nullptr;              ///< softmax(fuse_weight) [kDsMixes][kDsTargets]
    uint16_t* w1 = nullptr;              ///< BF16 [n_vocab][rank], mapped host memory (a step reads one row)
    uint16_t* rnn_w = nullptr;           ///< BF16 [3 rank][2 rank + 2 width], as loaded; freed by finish()
    uint16_t *w_sp = nullptr, *w_h = nullptr, *w_f = nullptr;   ///< its columns, split by finish()
    uint16_t* out_w = nullptr;           ///< BF16 [width][rank]
    void* fold = nullptr;                ///< Q8_0 [n_vocab][rank]
    float* fold_b = nullptr;             ///< [n_vocab]
    float* fold_bp = nullptr;            ///< fold_b in the screen's row order
    void *head_sc = nullptr, *fold_sc = nullptr;   ///< the head's and the fold's 2-bit copies (the draft head's screen)
    const void* head = nullptr;          ///< the target's Q8_0 head [n_vocab][width]
    const uint16_t* embedding = nullptr; ///< the target's BF16 embedding rows
    const float* embed_norm = nullptr;

private:
    friend model::Result dspark_plan(const Pack&, core::LoadPlan&, int, int, std::unique_ptr<DsparkWeights>&);
    void* dev(size_t bytes, bool& ok);
    void* mapped(size_t bytes, bool& ok);   ///< mapped host memory; the device's pointer to it
    std::vector<void*> allocs_, pinned_;
    uint64_t bytes_ = 0;
    bool finished_ = false;
};

/// The pack's drafter queued onto `reads` (file `file`, device `device`): `out` stays null when the pack has none. An error
/// when it has one this engine does not run.
model::Result dspark_plan(const Pack& pack, core::LoadPlan& reads, int file, int device, std::unique_ptr<DsparkWeights>& out);

/// The device bytes a session's drafter takes for `context` positions with a tap of `tap_rows` rows (0 without a drafter).
/// A session's drafter bytes (its caches and its steps' scratch), the shared tap aside.
uint64_t dspark_session_bytes(const Pack& pack, int64_t context);
/// The tap of `tap_rows` rows: the target's features of a call's rows, which every session's commit reads (one shared).
uint64_t dspark_tap_bytes(int tap_rows);

class DsparkBatch;

/// What a session holds of the drafter: the tap the target fills, the drafter's caches and scratch, and the calls the session
/// forwards. Without a drafter every call does nothing and draft_begin says the model has none.
class DsparkHook {
public:
    ~DsparkHook();
    /// `w` null: no drafter. `context` the session's positions; `tap_rows` the most rows one forward fills the tap with.
    /// `dump` (--diag-dump's directory, may be empty): every committed run's features and the first blocks' outputs (their
    /// final rows, every draft's logits, the drafts) go to dspark.bin there in order, to replay against a reference.
    /// `tap`: the shared tap of `tap_rows` rows (dspark_tap_bytes). `backed` > 0: the full layers' caches reserve the context and
    /// back only the cells of the first `backed` positions (and a block past them), growing with the target's (ensure).
    bool init(const DsparkWeights* w, const Geometry& g, int64_t context, int tap_rows, float* tap, const std::string& dump,
              int64_t backed, std::string& err);
    bool on() const { return w_ != nullptr; }
    bool dumping() const { return dump_ != nullptr; }
    /// The tap slot visit `visit` fills with its input rows, or -1.
    int slot_of_visit(int visit) const;
    /// Slot k's rows (fp32, kWidth each, contiguous): slots 0..2 the visits' inputs, slot 3 the final normed state.
    float* tap(int slot) const { return tap_ + (size_t) slot * tap_rows_ * kWidth; }
    /// Caller-owned chunk buffers reused by commit_prompt() (no extra device allocation).
    struct PromptScratch {
        float* mix = nullptr;         ///< [3][rows][kWidth]
        float* kb = nullptr;          ///< [rows][kKvHeads * kHeadDim]
        float* vb = nullptr;          ///< [rows][kKvHeads * kHeadDim]
        void* xq = nullptr;           ///< MMQ q8_1 rows for [rows][kWidth]
        int32_t* pos3 = nullptr;      ///< [rows][3]
        const int32_t* bounds = nullptr;  ///< {0, rows}
        const int32_t* ident = nullptr;   ///< 0..rows-1
    };
    /// Enqueues the context keys and values of the forward's `rows` rows (positions pos_d[0..rows), contiguous, on the
    /// device) from the tap. Only fixed addresses: a captured forward may hold it.
    /// The context of `rows` rows from the tap's rows row0.. (a batch window's segment starts past row 0).
    void commit(const int32_t* pos_d, int rows, void* stream, int row0 = 0);
    /// Batched commit for prompt chunks: runs dspark_mix once and projects k_ctx / v_ctx through MMQ.
    void commit_prompt(const int32_t* pos_d, int rows, basalt::prefill::mmq::Context& mmq, const PromptScratch& s, void* stream);
    /// The host's record of the same rows (outside a captured forward).
    void note(int64_t pos0, int rows);
    model::Result draft_begin(int32_t anchor, int64_t pos, int n, void* stream);
    model::Result draft_end(int32_t* ids, float* conf, int max_n, int& n, const model::ModelSession::DraftNeed& need, void* stream);
    /// Conversations: the sliding-window layers' rings are running state, the full layers' cells go with the target's.
    size_t running_bytes() const;
    void running_regions(core::RunningState& st) const;
    size_t cells_bytes(int64_t positions) const;
    /// The elastic caches (init's `backed`): the device bytes ensure(positions) maps, backing the cells of `positions`
    /// positions and a block past them; trim gives back what lies past them.
    uint64_t grow_bytes(int64_t positions) const;
    bool ensure(int64_t positions, std::string& err);
    void trim(int64_t positions);
    int64_t backed_cells() const { return backed_cells_; }
    bool usable() const {
        for (const auto& a : full_)
            if (!a->usable()) return false;
        return true;
    }
    /// The last ensure failed only because the device lacked the memory.
    bool out_of_memory() const {
        for (const auto& a : full_)
            if (a->out_of_memory()) return true;
        return false;
    }
    uint64_t backing_bytes() const {
        uint64_t n = 0;
        for (const auto& a : full_) n += a->backed();
        return n;
    }
    bool save_cells(std::vector<core::ConversationKv>& kv, int64_t positions, void* stream) const;
    /// Restores from kv[at..]; `at` moves past what it read. False when kv holds no drafter cells of that shape.
    bool restore_cells(const std::vector<core::ConversationKv>& kv, size_t& at, int64_t positions, void* stream);

private:
    friend class DsparkBatch;
    struct Cache {
        uint16_t *k = nullptr, *v = nullptr;   ///< fp16 [cap][kv heads][head width], position p at cell p % cap
        int64_t cap = 0;
        bool ring = false;
    };
    void* dev(size_t bytes, bool& ok);
    void count_backed();                    ///< backed_cells_ from the elastic caches
    void stage(int k, int n, void* stream);
    bool stage_graph(int k, int n, void* stream);
    const DsparkWeights* w_ = nullptr;
    std::vector<void*> allocs_, pinned_;
    std::vector<Cache> cache_;
    std::vector<std::unique_ptr<core::ElasticArray>> full_;   ///< the full layers' K and V when elastic
    int64_t backed_cells_ = 0;              ///< cells every full layer's cache backs
    std::vector<int> slot_;                 ///< by visit
    int tap_rows_ = 0;
    int64_t context_ = 0;
    float* tap_ = nullptr;
    // commit
    float* mix_ = nullptr;                  ///< [3][8][width]
    float* fr_ = nullptr;                   ///< the Markov head's context by position: [kFrRing][width]
    int32_t* pos3_ = nullptr;               ///< [8][3]
    int32_t* ring_bounds_ = nullptr;        ///< {0, ring_cells} for batched sliding-window commits
    float *kb_ = nullptr, *vb_ = nullptr;   ///< [8][1024]
    void* act_ = nullptr;                   ///< q8_1 rows
    // block
    float *x_ = nullptr, *q_ = nullptr, *qr_ = nullptr, *attn_ = nullptr, *h_ = nullptr, *att_scr_ = nullptr;
    void *act_a_ = nullptr, *act_h_ = nullptr;   ///< the window gemm's bf16 rows: a product's input, SwiGLU's output
    basalt::kernels::wgemm::Workspace ws_;
    int32_t* pos3b_ = nullptr;
    int64_t att_cells_ = 0;
    // head
    float *approx_ = nullptr, *ctx_h_ = nullptr, *ctx_f_ = nullptr, *state_ = nullptr, *latent_ = nullptr;
    void *hq_ = nullptr, *lat_q_ = nullptr, *head_scr_ = nullptr;   ///< q8_1 draft rows and latents [kDsDrafts], the screen's scratch
    int32_t *prev_ = nullptr, *ids_d_ = nullptr;
    uint16_t* w1_row_ = nullptr;            ///< the previous draft's row of the Markov head's w1 (host memory), on the card
    float* conf_d_ = nullptr;
    int32_t* ids_h_ = nullptr;
    // the block's stages as graphs: the layers' (by layer), the last with the head (by drafts)
    int32_t *meta_h_ = nullptr, *meta_d_ = nullptr;   ///< {anchor, position} of the block, pinned and on the device
    float* frc_ = nullptr;                  ///< the Markov head's context of the block (block inputs copy it)
    cudaGraphExec_t g_stage_[kDsMaxLayers] = {};
    cudaGraphExec_t g_head_[kDsBlock] = {};
    bool graphs_ = true;
    float* conf_h_ = nullptr;
    // the host's record: positions [lo_, hi_) have their context in the caches; a block is in flight
    int64_t lo_ = 0, hi_ = 0;
    int pending_ = 0;
    DsparkBatch* batch_owner_ = nullptr;
    int batch_slot_ = -1;
    // --diag-dump
    std::FILE* dump_ = nullptr;
    int dumped_ = 0;
    float* dump_logits_ = nullptr;          ///< [kDsDrafts][n_vocab]: each draft's exact logits (dspark_head_exact)
    int64_t dump_drafts_ = 0, dump_differ_ = 0;   ///< drafts dumped, and those whose pick is not the exact head's
    int64_t block_pos_ = 0;
    int32_t block_anchor_ = 0;
};

/// Shared multi-session DSpark scratch for batched commit_batch (inside target decode_batch) and draft_batch.
class DsparkBatch {
public:
    ~DsparkBatch();
    bool init(const DsparkWeights* w, int max_sessions, int max_rows, int tap_rows, float* tap, int64_t context, std::string& err);
    bool on() const { return w_ != nullptr; }
    void commit(DsparkHook* const* hooks, int n_seg, const basalt::kernels::WindowSegs* at_d, const int32_t* pos_d, int rows,
                void* stream);
    model::Result draft(DsparkHook* const* hooks, const int32_t* anchors, const int64_t* pos, const int* n, int count,
                        void* stream);
    model::Result take(int slot, int pending, int32_t* ids, float* conf, int max_n, int& n);

private:
    void* dev(size_t bytes, bool& ok);
    const DsparkWeights* w_ = nullptr;
    std::vector<void*> allocs_, pinned_;
    int max_sessions_ = 0, max_rows_ = 0, tap_rows_ = 0;
    int64_t att_cells_ = 0;
    float* tap_ = nullptr;
    // commit_batch scratch [max_rows]
    float *c_mix_ = nullptr, *c_kb_ = nullptr, *c_vb_ = nullptr;
    void* c_act_ = nullptr;
    // block chunk scratch [kChunkSlots * kDsBlock <= 32]
    static constexpr int kChunkSlots = 8;
    static constexpr int kMaxDraftSessions = 16;
    float *x_ = nullptr, *h_ = nullptr, *q_ = nullptr, *kb_ = nullptr, *vb_ = nullptr, *qr_ = nullptr, *attn_ = nullptr;
    float* att_scr_ = nullptr;
    void *act_a_ = nullptr, *act_h_ = nullptr;
    basalt::kernels::wgemm::Workspace ws_;
    // head batch scratch [max_sessions]
    float *hd_ = nullptr, *approx_ = nullptr, *ctx_h_ = nullptr, *ctx_f_ = nullptr, *state_ = nullptr, *latent_ = nullptr;
    float *frc_ = nullptr, *conf_d_ = nullptr, *conf_h_ = nullptr;
    void *hq_ = nullptr, *lat_q_ = nullptr, *head_scr_ = nullptr;
    size_t head_scr_stride_ = 0;
    int32_t *prev_ = nullptr, *ids_d_ = nullptr, *ids_h_ = nullptr, *meta_d_ = nullptr, *meta_h_ = nullptr;
    uint16_t* w1_row_ = nullptr;
    void* sync_stream_ = nullptr;
    bool synced_ = true;
};

/// The flags an Inkling 2 start takes for speculative decoding: --spec N, --draft-log FILE. False when `a` is neither.
bool dspark_parse_flag(const std::string& a, model::FlagArgs& args, program::CommonOptions& o);
/// With a drafter in the pack (o.model's) --spec defaults to the whole block (8), else 0; a --spec of 2 or more without one
/// is dropped with a warning.
void dspark_finish_flags(program::CommonOptions& o);

}  // namespace basalt::models::inkling2
