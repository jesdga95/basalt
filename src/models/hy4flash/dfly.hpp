// src/models/hy4flash/dfly.hpp - the DFly2 drafter of Hy4-flash: its pack section and weights, what the target leaves for it
// (a tap of each window's last-layer hidden states and index selections), the block pass over the target's own KV, and the
// flags. The forward and its assumptions are below; the inference reference is vLLM's hy_v4 DFly drafter.
//
// A block is the anchor token (the last pick, whose position the target has not run) and `n` mask tokens. The anchor's
// embedding is fused with the target's last-layer hidden state at the row before it (Gated Fusion); five dense MLA layers
// run over the block's rows, each attending the rows its target layer selected for that row (Index Share) in that layer's
// cache (KV Share) and the block's own rows before it; each mask row's output then passes the sequential Hidden-Correction
// with the embedding of the token drafted before it, and the target's output head gives the draft. The drafter keeps no
// cache and no indexer: a rejected block leaves nothing behind. Drafts are greedy; the target's own pick decides every row,
// so a draft only changes the speed.
//
// The forward. A block is the anchor `a` at position `pos` and n (1..5) mask tokens, rows at pos .. pos + n; `s` = pos - 1 is
// the source row, the target row whose output chose `a`; e() is the target's token embedding.
//   1. Gated Fusion, row 0 only: x0 = e(a) + gate * eh_proj(enorm(e(a)) :: hnorm(h_s)), h_s the target's last-layer hidden
//      state at row s after the output norm. Rows 1..n: e(mask).
//   2. Five dense layers, one residual stream (no hyper-connections), pre-norm: y = input_ln(x);
//      x += post_proj_norm(o_proj(attention(y))); x += mlp(post_attention_ln(x)), a plain SwiGLU (no clamp). z = final_norm(x).
//   3. Attention of layer i: the target's absorbed MLA with the drafter's own weights. q = q_b(q_a_ln(q_a(y))), heads of 192
//      nope + 64 rope; latent kv_a_ln(kv_a(y)[:512]), rope part kv_a(y)[512:]; interleaved-pair rope at the target sparse layers'
//      base, position pos + r; kv_b rows per head are 192 key rows then 256 value rows, used absorbed; scale 1/sqrt(256);
//      the sink adds exp(sink) to the softmax denominator; the output is gated by sigmoid(linear_gate(y)), then o_proj.
//      KV Share: the keys and values are target layer kv_layers[i]'s cache rows (FP8 rows decoded exactly under --kv fp8).
//      Index Share: the rows are the cells target layer idx_layers[i] selected for row s (every position before pos while its
//      indexer keeps them all), the same for every block row; block row r also attends the block's rows 0..r, whose K/V come
//      from the drafter's own kv_a and live in scratch (rounded to FP8 rows first under --kv fp8). Nothing is written into
//      the target's cache.
//   4. Hidden-Correction, in order, draft j = 0..n-1 from row j + 1 and the token before it (a, then draft j - 1):
//      c = hidden_norm(z[j+1]) :: embed_norm(e(prev)), h' = z[j+1] + down(silu(gate(c)) * up(c)).
//   5. Head: the target's output head over h', argmax, screened by a 2-bit copy of the head's rows: the rows near the copy's
//      best are scored exactly as the target scores them (kernels/q80_screen.hpp). Row 0's output is not used; its K/V feed
//      the later rows.
// Assumptions the reference leaves open: the confidence is sigmoid(confidence_head(h')) (vLLM does not load the head; it only
// steers the draft policy's window choice); rope positions are text positions, so after a picture the block turns by the
// text's rotary position; the draft layers read sparse target layers (checked at load); a pack with a context local window
// (dfly_context_local_window) is refused.
#pragma once

#include "basalt/core/loader.hpp"
#include "basalt/core/model_file.hpp"
#include "basalt/model/adapter.hpp"
#include "basalt/program/options.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace basalt::models::hy4flash {

struct Pack;

constexpr int kDflyBlock = 6;       ///< rows of a full block: the anchor and five masks
constexpr int kDflyDrafts = kDflyBlock - 1;
constexpr int kDflyTapRows = 16;    ///< rows of a window the tap keeps (the draft policy's widest window)
constexpr int kDflyMaxLayers = 8;

/// The drafter's shape: from the pack's meta.dfly2 (the checkpoint's `dfly_*` settings) and the tensors' shapes.
struct DflyDims {
    int layers = 0, E = 0, q_lora = 0, heads = 0, ffn = 0, hc_ffn = 0;
    int vocab = 0, mask_id = 0;
    int topk = 2048;                  ///< the target's selection width (dfly_target_index_topk)
    float rms_eps = 1e-5f, rope_base = 1e7f;
    std::vector<int> kv_layers;       ///< the target layer whose cache each draft layer reads
    std::vector<int> idx_layers;      ///< the target layer whose selection each draft layer reads
};

struct DflyTensor {
    uint64_t offset = 0, bytes = 0;   ///< absolute in the file
    std::vector<uint64_t> shape;      ///< safetensors order, outermost first
};

struct DflyPack {
    bool present = false;
    DflyDims dims;
    std::map<std::string, DflyTensor> tensors;   ///< by name without `model.dfly_layers.`
};

/// Reads meta.dfly2 (null or absent: present stays false) of a pack whose `dfly` section starts at `section_offset`;
/// offsets in the record are relative to it. False with `err` when the record is malformed.
bool parse_dfly_meta(const core::Json& dfly2, uint64_t section_offset, DflyPack& out, std::string& err);

/// One tensor the engine reads: its name, shape and whether it is held as FP32 (the norms, the sink, the gate, the
/// confidence head) or as the checkpoint's BF16.
struct DflySpec {
    std::string name;
    std::vector<uint64_t> shape;
    bool f32 = false;   ///< a vector widened to FP32
    bool q8 = false;    ///< a projection, rounded to Q8_0 as it loads
    bool q4 = false;    ///< a projection rounded to Q4_0 as it loads: the attention's gate and output and the MLP
    uint64_t elems() const;
    uint64_t bytes() const { return q4 ? elems() / 32 * 18 : q8 ? elems() / 32 * 34 : elems() * (f32 ? 4 : 2); }   ///< on the card
};
/// Every tensor of a drafter of these dims.
std::vector<DflySpec> dfly_specs(const DflyDims& d);
/// Fills the dims the shapes carry (E, q_lora, heads, ffn, hc_ffn) from `tensors`, and checks every spec against them.
bool dfly_check_shapes(const std::map<std::string, DflyTensor>& tensors, DflyDims& d, std::string& err);

struct DflyLayer {
    const void *q_a, *q_b, *kv_a;                        // Q8_0
    const void *gate, *o, *mlp_gate, *mlp_up, *mlp_down;  // Q4_0
    const uint16_t* kv_b;                                                    // BF16: the per-head kernels read it
    const float *in_ln, *q_a_ln, *kv_a_ln, *post_proj_ln, *post_attn_ln, *sink;
};

/// The drafter's weights on the main card: its projections rounded as they load (so the expert tier is sized after the
/// smaller drafter), the attention's gate and output and the MLP to Q4_0 and the rest to Q8_0, kv_b in BF16 as stored, FP32
/// vectors, and the 2-bit copy of the target's head that screens the drafts.
class DflyWeights {
public:
    ~DflyWeights();
    /// Device memory for every tensor of `dims` and the head's copy on the current device; `slot` then says where each goes.
    bool allocate(const DflyDims& dims, std::string& err);
    void* slot(const std::string& name) const;   ///< null for an unknown name
    uint64_t bytes() const { return bytes_; }
    /// Packs the head's copy from head_q8 (once its bytes are on the card).
    bool pack_screen(void* stream, std::string& err);

    DflyDims dims;
    std::vector<DflyLayer> layer;
    const void *eh_proj = nullptr, *hc_gate = nullptr, *hc_up = nullptr, *hc_down = nullptr;   // Q8_0
    const float *enorm = nullptr, *hnorm = nullptr, *fuse_gate = nullptr, *hc_hidden_norm = nullptr, *hc_embed_norm = nullptr,
                *final_norm = nullptr, *conf_w = nullptr, *conf_b = nullptr;
    const void* head_q8 = nullptr;   ///< the target's Q8_0 head (the model's tensor): the drafts read it as the target does
    void* head_screen = nullptr;     ///< its 2-bit copy (kernels/q80_screen.hpp)

private:
    std::map<std::string, void*> slots_;
    uint64_t bytes_ = 0;
};

/// The pack's drafter queued onto `reads` (file `file`, device `device`) when `pack` has one: `out` is null otherwise.
/// An error when it has one that does not fit the target (`geometry`).
model::Result dfly_plan(const Pack& pack, core::LoadPlan& reads, int file, int device, std::unique_ptr<DflyWeights>& out);

/// The window row whose output chose the token at `pos`: row pos - 1 - pos0 of a window that began at `pos0`, when its hidden
/// state (bit r of `hid_ok`) and every one of the `layers` selections (bits of sel_ok[r]) were kept; else -1.
inline int dfly_source_row(int64_t pos0, unsigned hid_ok, const unsigned* sel_ok, int layers, int64_t pos) {
    const int64_t r = pos - 1 - pos0;
    if (pos0 < 0 || r < 0 || r >= kDflyTapRows || !(hid_ok & (1u << r)) || sel_ok[r] != (1u << layers) - 1u) return -1;
    return (int) r;
}

/// What the target leaves of the rows of its last decode window: each row's last-layer hidden state (after the output
/// norm) and, per draft layer, the cells that row's attention selected in the layer that layer reads. The drafter after a
/// window reads the row of the last accepted token. Rows are valid only for the positions of that one window.
class DflyTap {
public:
    static std::unique_ptr<DflyTap> create(const DflyDims& d, int sel_cap, std::string& err);
    ~DflyTap();
    void begin_window(int64_t pos0);   ///< a window starts at `pos0`: the earlier rows are dropped
    void invalidate();                 ///< the state moved on by another path (a prompt chunk): no row is valid
    void hidden(int row, const float* fin, void* stream);
    /// Target layer `layer` selected `n` cells (device) for row `row`; n < 0: every position up to the row.
    void selection(int layer, int row, const int32_t* cells, int n, void* stream);
    /// The row whose output chose the token at position `pos` (the row at pos - 1), or -1 when it was not kept whole.
    int source_row(int64_t pos) const;
    const float* hidden_row(int row) const { return hid_ + (size_t) row * E_; }
    const int32_t* cells(int slot, int row) const { return sel_ + ((size_t) slot * kDflyTapRows + row) * cap_; }
    int count(int slot, int row) const { return n_[row][slot]; }

private:
    DflyTap() = default;
    int E_ = 0, cap_ = 0, layers_ = 0;
    std::vector<int> idx_layers_;
    float* hid_ = nullptr;
    int32_t* sel_ = nullptr;
    int64_t pos0_ = -1;
    unsigned hid_ok_ = 0;
    unsigned sel_ok_[kDflyTapRows] = {};
    int n_[kDflyTapRows][kDflyMaxLayers] = {};
};

/// The block pass and the sequential heads, with their own scratch (one per session).
class DflyRunner {
public:
    struct Env {
        const DflyWeights* w = nullptr;
        int device = 0;
        std::function<bool(int32_t, float*)> embed;              ///< a token's embedding row (E floats, host)
        int sel_cap = 2056;                                      ///< cells of a selection
        bool fp8 = false;                                        ///< the target's caches hold --kv fp8 rows (kFp8RowBytes)
    };
    static std::unique_ptr<DflyRunner> create(const Env& env, std::string& err);
    ~DflyRunner();

    /// Enqueues the block of `n` drafts (1..5) after `anchor` at `pos` on `stream`: the target's caches `cache[i]` (draft
    /// layer i's: fp16 rows of 576, or FP8 rows with Env::fp8) and the tap's row `row`. Nothing is read from the host but the
    /// embeddings. With FP8 caches the block's own rows are rounded to FP8 rows too, as the target's would be. `rope_pos`: the
    /// rotary position of cell `pos` (< 0: `pos`); after a picture the text runs behind the cells (vision.hpp).
    bool begin(int32_t anchor, int64_t pos, int n, const DflyTap& tap, int row, const void* const* cache, void* stream,
               std::string& err, int64_t rope_pos = -1);
    /// Runs the heads over the block begin() made, in order, each with the token before it: the `made` drafts (the
    /// block's n, or fewer with `need`) and each one's estimated chance of acceptance. `on_row(j, token)`, when set, is
    /// called in head step j >= 1 once the step is queued on the card, with window row j's token (draft j - 1): host work
    /// there runs while the card computes the step. `need(conf, j)`, when set, is asked with draft j's chance before its
    /// head runs: false ends the block before draft j.
    bool finish(int32_t* ids, float* conf, int& made, void* stream, std::string& err,
                const std::function<void(int, int32_t)>& on_row = nullptr,
                const std::function<bool(const float*, int)>& need = nullptr);
    /// The block's outputs after the final norm, `rows` rows of E floats, to the host (for the tests against a reference).
    void copy_block(float* host, int rows, void* stream) const;

private:
    DflyRunner() = default;
    struct Impl;
    std::unique_ptr<Impl> p_;
};


/// What a session holds of the drafter: its tap, its runner, and the calls the session forwards. Without a drafter in the
/// pack every call does nothing and `draft_begin` says the model has none.
class DflyHook {
public:
    ~DflyHook();
    /// `w` null: no drafter. `ctx` the context in positions.
    /// `dump` (--diag-dump's directory, may be empty): each block's inputs (the source row's hidden state, the cache rows it attends)
    /// are written there for the first few blocks, to compare with a reference.
    /// `fp8`: the target's sparse caches hold --kv fp8 rows.
    bool init(const DflyWeights* w, int sel_cap, int64_t ctx, std::function<bool(int32_t, float*)> embed,
              const std::string& dump, bool fp8, std::string& err);
    void release() { tap_.reset(); runner_.reset(); row0_ = -1; pending_ = 0; }
    bool on() const { return runner_ != nullptr; }
    void begin_window(int64_t pos0) { if (tap_) tap_->begin_window(pos0); }
    void invalidate() { if (tap_) tap_->invalidate(); }
    void run(int row0) { row0_ = row0; }   ///< the next run of rows starts at this row of the window (-1: none is kept)
    /// The run's rows `first` .. `first` + n - 1 have the hidden states fin[0..n) (rows of E floats).
    void hidden(int first, int n, const float* fin, void* stream) {
        if (tap_ && row0_ >= 0)
            for (int i = 0; i < n; ++i) tap_->hidden(row0_ + first + i, fin + (size_t) i * w_->dims.E, stream);
    }
    /// Run row `t`'s selection in target layer `layer`: n cells (device), or n < 0 for every position up to the row.
    void selection(int layer, int t, const int32_t* cells, int n, void* stream) { if (tap_ && row0_ >= 0) tap_->selection(layer, row0_ + t, cells, n, stream); }
    /// `cache`: the session's per-layer cache rows. A block is made only where the tap holds the row before `pos`. `rope_pos`
    /// as DflyRunner::begin.
    model::Result draft_begin(int32_t anchor, int64_t pos, int n, const std::vector<uint8_t*>& cache, void* stream,
                              int64_t rope_pos = -1);
    model::Result draft_end(int32_t* ids, float* conf, int max_n, int& n, void* stream, const model::ModelSession::DraftNeed& need);
    /// Called in the heads with each draft row's token as it is known (DflyRunner::finish's on_row).
    void on_row(std::function<void(int, int32_t)> f) { on_row_ = std::move(f); }

private:
    std::unique_ptr<DflyTap> tap_;
    std::unique_ptr<DflyRunner> runner_;
    const DflyWeights* w_ = nullptr;
    int row0_ = -1, pending_ = 0, dumped_ = 0, dump_n_ = 0;
    int64_t dump_pos_ = -1;
    int32_t dump_anchor_ = 0;
    std::function<bool(int32_t, float*)> embed_;
    std::function<void(int, int32_t)> on_row_;
    std::string dump_;
    int64_t ctx_ = 0;
    bool fp8_ = false;
};

/// The drafts of a block in order, `step(j, previous)` giving draft j after the token before it (the anchor first), or
/// ending the block before it (`go` false): `made` drafts. The loop DflyRunner::finish runs, apart from the device, so a
/// test can run it over fake logits. False when a step fails.
bool dfly_draft_loop(int n, int32_t anchor,
                     const std::function<bool(int j, int32_t previous, int32_t& token, float& conf, bool& go)>& step,
                     int32_t* ids, float* conf, int& made);

/// The flags a Hy4-flash start takes for speculative decoding, as DeepSeek's: --spec N, --spec-oracle FILE, --diag-teacher,
/// --draft-log FILE. False when `a` is none of them.
bool dfly_parse_flag(const std::string& a, model::FlagArgs& args, program::CommonOptions& o);
/// With the pack's `dfly` section present --spec defaults to the whole block (6), else 0; a --spec of 2 or more without a
/// drafter or an oracle file is dropped with a warning.
void dfly_finish_flags(program::CommonOptions& o);

}  // namespace basalt::models::hy4flash
