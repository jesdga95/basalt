// src/models/glm5next/forward.hpp - the glm5-next forward pass: the embedding, the model layers and the output head.
//
// One GlmForward belongs to one session and owns the per-window scratch, grown to the widest window it is asked for.
// Every width it multiplies comes from the pack's geometry, never from a tensor's shape (the same rule the cache sizing
// follows); a tensor whose shape disagrees with the geometry is an error, not a silent reshape.
//
// The dense layers run in fp32 over fp32 weights. A pack with quantized dense tensors is refused by name rather than run
// wrong: kernels::f32_rows_dot is not the kernel a quantized weight needs, and that path belongs with a pack that
// carries quantized weights, where it can be checked. The routed experts are the opposite case: they come from the
// shared expert tier (core/expert_tier.hpp) in their packed formats, the card computing the ones it homes with the
// grouped kernels and the CPU pool the ones in pinned RAM, and one kernel per block sums the two.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "models/glm5next/pack.hpp"

namespace basalt::prefill::mmq {
class Context;
}  // namespace basalt::prefill::mmq

namespace basalt::models::glm5next {

class GlmModel;
class GlmSession;
struct Geometry;
struct DevTensor;

/// What the pass holds on the card outside a window: the buffers its constructor allocates once (the q8_1 rows the
/// packed products read, the routed experts' block, the packed KDA conv taps) and the fp32 copies it keeps of the
/// packed weights it multiplies as rows. `prefillp::window_units` prices only what scales with the token count, so a
/// window that ignores this runs the card out of memory part way through the first block; the session's window cap and
/// the expert tier's reserve both ask here instead.
uint64_t forward_footprint_bytes(const Geometry& g, const Pack& p);

/// One session's forward pass. `run` takes a window of tokens, advances the caches over it and leaves the last row's
/// logits in the session's logits buffer, which `copy_logits` brings back.
class GlmForward final {
public:
    /// Checks that this session's pack is a shape the pass can run, and packs the two KDA weights that several tensors
    /// must become for the kernels to read them. False with `err` otherwise; nothing is allocated for the window until
    /// the first `run`.
    GlmForward(GlmSession& session, std::string& err);
    ~GlmForward();

    GlmForward(const GlmForward&) = delete;
    GlmForward& operator=(const GlmForward&) = delete;

    bool ok() const { return ok_; }

    /// The model over `n_tokens` tokens at positions `pos .. pos + n_tokens - 1` of sequence `seq`. `logit_rows` is how
    /// many of the window's last rows get logits: 1, the pick after the window, is what a plain decode asks for; more is
    /// what a drafted window needs, one pick per row.
    bool run(int seq, const int32_t* tokens, int n_tokens, int64_t pos, std::string& err, int logit_rows = 1);

    /// The session's device logits into host `out` (n_vocab floats).
    bool copy_logits(float* out, std::string& err);
    /// The window's last `logit_rows()` rows' logits, oldest first, `logit_rows() * n_vocab` floats into host `out`.
    /// A run that asked for one row leaves these empty; a drafted window asks for its rows.
    bool copy_row_logits(float* out, std::string& err);
    /// The same rows on the device, for a sampler that reads them there.
    float* row_logits() const { return row_rows_; }
    int logit_rows() const { return logit_rows_; }

    /// The pack's next-n drafter block, and whether this build can run it: the block is there, its indexer is the
    /// model's own, its experts have a card path, and its tier opened. One draft per call, after the pass has run the
    /// token before `anchor` (see draft_begin in the session adapter).
    bool drafts() const { return drafts_; }
    /// The draft token after `anchor`, the token at position `pos`, and its probability under the drafter's head.
    bool draft(int32_t anchor, int64_t pos, int32_t& id, float& conf, std::string& err);

    /// What the pass has taken on the card for the weight copies it keeps: the fp32 rows of the packed weights it
    /// multiplies as rows, the padded copies the wide-window product makes, and the one big weight it reuses a slot
    /// for. `forward_footprint_bytes` bounds these for the window cap and the tier reserve; the forward test holds the
    /// two against each other, so a bound that stops covering them cannot pass unnoticed.
    uint64_t memo_bytes() const { return memo_bytes_ + (big_val_ ? big_values_ * sizeof(float) : 0); }

    /// The 4-byte units of device scratch the current window actually allocated, and the window it is sized for. The
    /// prefill plan's bound is checked against these by the forward test, which is what keeps the two honest.
    uint64_t scratch_units() const { return scratch_units_; }
    int window_cap() const { return cap_; }

private:
    struct Scratch;

    bool grow(int max_tokens, std::string& err);
    void free_scratch();

    /// y[T][N] = x[T][K] . w[N][K]. `w` is a pack tensor, checked against the geometry's widths before it is read.
    bool proj(const DevTensor* w, const float* x, float* y, int T, int N, int K, std::string& err);
    /// The same product against a tensor this pass assembled rather than read from the pack, shape unchecked: MLA's
    /// per-head weights are slices of one bigger tensor, and a slice carries the same [N][K] layout.
    bool proj_at(const DevTensor* w, const float* x, float* y, int T, int N, int K, std::string& err);
    /// A per-head projection against a [K][N][H] weight: `x` is [T][H][K] and `y` comes back [T][H][N], each head's
    /// product taken over its own [N][K] matrix. A single token needs no reshaping; a window gets one head-major
    /// copy of `x` and then one ordinary `proj_at` per head.
    bool proj_heads(const DevTensor* w, const float* x, float* y, int T, int H, int N, int K, std::string& err);
    /// The same product from a raw device pointer, for the packed weights this class built itself.
    bool proj_rows(const float* w, const char* what, const float* x, float* y, int T, int N, int K, std::string& err);
    /// A weight this pass requires, or null with `err` (layer < 0 reads a global tensor name). A weight the code
    /// multiplies as rows but the pack stores packed comes up fp32 on first use and is kept.
    const float* weight(int layer, const char* name, std::string& err);
    /// One packed tensor of `t`'s, dequantized into a device block of fp32 rows, or null with `err`. `reuse` is an
    /// existing block that already holds `reuse_values` values: when that covers `t`, the rows are uploaded into it
    /// and it comes back, so a weight the pass wants in every block does not cost an allocation every time.
    float* dequantized(const DevTensor& t, std::string& err, float* reuse = nullptr, uint64_t reuse_values = 0);

    /// A window wider than the mmvq kernels' columns, against a packed weight: the engine's prefill MMQ path, the
    /// weight staying packed and the activations rounded to q8_1.
    bool proj_mmq(const DevTensor& w, const float* x, float* y, int T, int N, int K, std::string& err);
    /// `w`'s device bytes with the zeroed tail MMQ's tiles read past its last row, when its rows need one.
    const void* mmq_weight(const DevTensor& w, int K, std::string& err);
    /// Sizes the wide window's activation and row-map block for `rows` tokens, outside the window's scratch budget.
    bool grow_mmq(int rows, std::string& err);

    bool embed(const int32_t* tokens, int T, float* out, std::string& err);
    bool kda_attention(int layer, int kda_layer, float* normed, int T, std::string& err);
    bool mla_attention(int layer, int mla_layer, float* normed, int T, int64_t pos, std::string& err);
    /// The indexer's pools and the cell list a window attends; 0 when the caches hold nothing scoreable yet.
    bool dsa_cells(int layer, int idx_layer, int T, int64_t pos, int& n_sel, std::string& err);
    /// The indexer's key LayerNorm (weight and bias). There is no shared LayerNorm kernel in the engine, and the row
    /// is one token's worth of the indexer's key width, so this one goes through the host rather than shipping a new
    /// kernel without a parity oracle for it.
    bool layernorm_rows(const float* x, const float* w, const float* b, float* y, int T, int n, float eps,
                        std::string& err);
    bool mlp(int layer, const float* normed, int T, std::string& err);

    /// The MLA split where the indexer sits: the projections up to the cache write, then the read of the cache back.
    /// `latent` is the cache's rows and `first_pos` the position of the window's first row.
    bool mla_latents(int layer, float* normed, int T, int64_t first_pos, float* latent, std::string& err);
    /// ... and the read: the key projection absorbed into the query, the sparse attention over the `n_sel` cells of
    /// each row of `ids`, and the output projection. `host_len[t]` is row t's cache length.
    bool mla_read(int layer, int T, float* latent, const int32_t* ids, int n_sel, const int32_t* host_len,
                  std::string& err);
    /// The drafter's block over T rows: `h[t]` is the model's collapsed hidden at position `first_pos - 1 + t` and
    /// `emb_next[t]` the embedding of the token that follows it. Its latents land at positions `first_pos + t` in its
    /// own cache, read back through the model's indexer's cells; the block's hidden comes back in `s->cur`.
    bool mtp_block(const float* h, const float* emb_next, int T, int64_t first_pos, const int32_t* ids, int n_sel,
                   std::string& err);
    /// The drafter's cache rows a window completes: every position whose next token the window carries, plus the one
    /// the pass kept from the window before. Without these the block's attention would read cache it never wrote.
    bool mtp_rows(const int32_t* tokens, int T, int64_t pos, std::string& err);
    /// The block's index, and whether the pack, the card and the tier all let this pass run it.
    int mtp_layer() const { return geo_.n_layers - geo_.nextn; }
    bool drafts_ = false;
    /// The last windows' collapsed hiddens and the cells their rows attended, kept outside the scratch. Two of them:
    /// when the engine rejects a draft, the anchor is the row *before* the last one the pass ran, and the drafter
    /// needs that row's hidden, not the one it computed for the token it threw away.
    static constexpr int kMtpKeep = 2;
    float* mtp_prev_h_ = nullptr;      ///< [kMtpKeep][E], oldest first
    int32_t* mtp_prev_ids_ = nullptr;  ///< [kMtpKeep][mtp_sel_stride_], the same rows
    int mtp_prev_n_sel_[kMtpKeep] = {};
    int mtp_keep_ = 0;                 ///< how many of the kept rows hold a window's tail
    int64_t mtp_prev_pos_ = -1;        ///< the position of the newest kept row
    int mtp_sel_stride_ = 0;           ///< the row stride of the pass's cell list, which `mtp_prev_ids_` matches
    /// The kept row for position `p`, or -1. Its cells and their count come back with it.
    int mtp_keep_row(int64_t p) const {
        if (p < 0 || mtp_keep_ == 0 || p > mtp_prev_pos_) return -1;
        const int64_t back = mtp_prev_pos_ - p;
        return back < mtp_keep_ ? mtp_keep_ - 1 - (int) back : -1;
    }
    int last_n_sel_ = 0;               ///< the cells the last indexer chose for this window's rows
    float* mtp_logits_ = nullptr;      ///< the drafter head's logits for one row, on the device
    std::vector<float> mtp_host_;      ///< ... and the host copy the pick is taken from
    std::vector<int32_t> host_mtp_len_;

    GlmSession& session_;
    const GlmModel& model_;
    const Geometry& geo_;
    bool ok_ = false;
    int cap_ = 0;                      ///< the window the scratch is sized for
    int head_wide_ = 0;                ///< the widest per-head MLA product the scratch stages a head of
    uint64_t scratch_units_ = 0;       ///< the 4-byte units grow() allocated for it
    Scratch* s_ = nullptr;             ///< owned; null until the first window sizes it
    std::vector<void*> scratch_blocks_;  ///< the window scratch, freed together whenever it is resized
    float* conv_pack_ = nullptr;       ///< per KDA layer: the three conv tap streams as kda_front wants [3][conv][D]
    std::vector<const void*> fp32_src_;  ///< packed weights brought up as fp32 rows, keyed by the tensor's bytes
    std::vector<float*> fp32_val_;       ///< their fp32 copies, owned here
    /// The weights too big to keep. `attn_k_b.weight` is a quarter of a gigabyte packed and a gigabyte as fp32 rows,
    /// and every block's is the same size, so they take turns in one block instead of growing the list above past the
    /// card - which is sound only because the pass consumes each of them on this stream the moment it asks, so the
    /// next block's upload is ordered after the kernel that was reading the previous one.
    const void* big_src_ = nullptr;      ///< the packed tensor whose rows are currently in big_val_
    float* big_val_ = nullptr;           ///< its fp32 rows; the next big weight uploads over them
    uint64_t big_values_ = 0;            ///< values big_val_ can hold

    std::vector<int32_t> host_pos_;    ///< this window's positions and cache lengths, in the int32 the kernels read
    std::vector<int32_t> host_len_;
    std::vector<int32_t> host_rids_;   ///< the router's per-token expert choice, read back to drive the tier's lookup
    std::vector<float> host_rweights_; ///< the router's per-token weights, which the sum reads off the card
    std::vector<float> host_act_;      ///< the window's activations, which the CPU pool reads from the host

    /// The routed experts of one block, allocated once with the pass (a block is the tier's window, not the chunk's).
    void* x_q8_ = nullptr;             ///< the block's activations, one q8_1 row per token
    void* dq8_ = nullptr;              ///< a packed dense projection's activations: one q8_1 row per window token
    int kqmax_ = 0;                    ///< the widest input the pack's packed dense weights multiply over
    prefill::mmq::Context* mmq_ = nullptr;  ///< the wide-window product, null when this build has no MMQ path
    void* mmq_xq_ = nullptr;           ///< that product's activations, quantized the way MMQ reads them
    int32_t* mmq_ids_ = nullptr;       ///< the identity row map the wide product reads through
    int32_t* mmq_bounds_ = nullptr;    ///< {0, rows}: the one matrix a dense product runs over
    int mmq_cap_ = 0;                  ///< the rows mmq_xq_ and mmq_ids_ are sized for
    int mmq_ids_n_ = 0;                ///< the rows mmq_ids_ currently maps
    std::vector<int64_t> mmq_tiles_;   ///< the (type, rows) pairs whose MMQ tiles have been readied
    std::vector<const void*> mmq_src_; ///< packed weights copied out with a zeroed read-past tail
    std::vector<void*> mmq_pad_;       ///< those copies, owned here
    float* row_rows_ = nullptr;        ///< the drafted window's rows of logits, oldest first (`row_bytes_` of them)
    uint64_t row_bytes_ = 0;
    int logit_rows_ = 1;               ///< the rows the last run left there
    /// The bytes this pass has taken for the copies it keeps - the fp32 rows of packed weights and the padded MMQ
    /// copies, not the one big slot it reuses - so `memo_bytes` can say what it ended up holding.
    uint64_t memo_bytes_ = 0;
    std::vector<int32_t> host_bounds_; ///< {0, rows}, staged for the copy to mmq_bounds_
    void* ex_scr_ = nullptr;           ///< kernels::native_expert_grouped's scratch
    float* ex_rows_ = nullptr;         ///< the experts the card computed: kMaxSumRows rows of n_embd
    void* plan_dev_[2] = {};           ///< the grouped plan: the homed experts, then the ones still being copied in
};

}  // namespace basalt::models::glm5next
