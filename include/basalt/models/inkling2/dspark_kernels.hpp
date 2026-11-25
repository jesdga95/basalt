// include/basalt/models/inkling2/dspark_kernels.hpp - the kernels of Inkling 2's DSpark drafter that the shared ones do not
// cover: the mixes of the target's features, the block's input rows, the Markov head's recurrence, its screened head, and
// the head folds made once at load.
//
// Rows are kWidth (2304) fp32 values, row-major; every pointer is device memory. A row's result never depends on how many
// rows share its call. `stream` may be null (the call then synchronises).
#pragma once

#include "basalt/kernels/window_segs.hpp"

#include <cstddef>
#include <cstdint>

namespace basalt::models::inkling2 {

constexpr int kDsBlock = 8;          ///< rows of a block: the anchor and seven masks
constexpr int kDsDrafts = kDsBlock - 1;
constexpr int kDsTargets = 4;         ///< target features a position carries: visits 5, 23, 41 and the final normed state
constexpr int kDsMixes = 4;           ///< mixes of them: one per draft layer, then the Markov head's context
constexpr int kDsRank = 256;          ///< the Markov head's rank
constexpr int kDsRnn = 3 * kDsRank;   ///< its recurrence's rows: the gate, the candidate and the latent

/// The mixes' weights: alpha [kDsMixes][kDsTargets] (each row softmaxed over the targets), norm[j] mix j's RMS norm weights.
struct DsMixArgs {
    const float* alpha = nullptr;
    const float* norm[kDsMixes] = {};
    float eps = 1e-6f;
};

/// For each of `rows` rows: mix j = rms(sum_k alpha[j][k] f_k) * norm[j], the targets summed in order in fp32, f_k = the
/// row of target slot k at tap + k * tap_stride + r * kWidth. Mixes 0..2 go to out + j * out_stride + r * kWidth; mix 3
/// (the Markov head's context) to the ring row of the row's position, fr + (pos[r] % fr_cap) * kWidth. pos3[r] gets
/// {pos[r], pos[r], pos[r]}, the rotary positions attn_qk_prepare reads.
void dspark_mix(const float* tap, int64_t tap_stride, int rows, const DsMixArgs& a, float* out, int64_t out_stride,
                const int32_t* pos, float* fr, int fr_cap, int32_t* pos3, void* stream);

/// A block's input rows, its anchor and position read from the device (meta = {anchor, pos0}, so a captured block takes any):
/// row 0 = rms(embedding row anchor) * embed_norm (BF16 rows, device-readable), rows 1..block-1 = `mask`; pos3[t] =
/// {pos0 + t} x 3. The Markov head starts from them: *prev = anchor, state[0..kDsRank) = 0, fr_out = its context of position
/// pos0 - 1, ring row (pos0 - 1) % fr_cap of fr, w1_row = the anchor's row of w1 (BF16 [vocab][kDsRank], device-readable).
void dspark_block_inputs(const uint16_t* embedding, const int32_t* meta, const float* embed_norm, float eps, const float* mask,
                         int block, float* x, int32_t* pos3, int32_t* prev, float* state, const float* fr, int fr_cap, float* fr_out,
                         const uint16_t* w1, uint16_t* w1_row, void* stream);

/// One step of the Markov head's recurrence. raw = ctx_h + ctx_f + bias + w_sp [state_in ; w1_row] (w_sp BF16
/// [kDsRnn][2 kDsRank], w1_row the previous token's row of w1, BF16 [kDsRank] on the device); gate = sigmoid(raw[0:r]); state_out = gate * state_in + (1 - gate) *
/// tanh(raw[r:2r]); latent = tanh(raw[2r:3r]) into `latent` (fp32 [kDsRank]).
void dspark_markov_step(const float* ctx_h, const float* ctx_f, const float* bias, const uint16_t* w_sp, const uint16_t* w1_row,
                        const float* state_in, float* state_out, float* latent, void* stream);

/// The draft head: draft i's logits l[v] = cap tanh((head_v . h + fold_v . lat + fold_b[v]) / cap) over v < n_vocab, h = block
/// row i + 1 (q8_1, kWidth) and lat its latent (q8_1, kDsRank). Its argmax is screened (basalt/kernels/head_proxy.cuh) by
/// 2-bit copies of the head and the fold: the copies score every row, the rows near the best are scored exactly, the exact
/// best is the pick. The pick is dspark_head_exact's first argmax whenever that row is among the rescored ones.
struct DsHead {
    const void* head = nullptr;      ///< Q8_0 [n_vocab][kWidth]
    const void* fold = nullptr;      ///< Q8_0 [n_vocab][kDsRank]
    const float* fold_b = nullptr;   ///< [n_vocab]
    const float* fold_bp = nullptr;  ///< fold_b in the copies' row order (dspark_screen_order)
    const void* head_sc = nullptr;   ///< the copies (dspark_screen_pack)
    const void* fold_sc = nullptr;
    int64_t n_vocab = 0;
    float cap = 30.f;
    bool head_planes = false;
};

/// The 2-bit copy of `n_rows` Q8_0 rows of n_in (kWidth or kDsRank) values, packed from the rows. The copy holds the rows
/// transposed over its tiles of 32 (copy row r is row (r % 32) T + r / 32, T the tiles), so that rows that are often best
/// together (a tokenizer's punctuation) fall in different tiles; it has dspark_screen_rows rows.
size_t dspark_screen_bytes(int n_in, int64_t n_rows);
void dspark_screen_pack(const void* q8_0, int n_in, int64_t n_rows, void* screen, void* stream);
int64_t dspark_screen_rows(int64_t n_rows);
/// by_row[r] = by_token[the copy row r's row], 0 past n_rows (by_row has dspark_screen_rows values).
void dspark_screen_order(const float* by_token, int64_t n_rows, float* by_row, void* stream);
/// Scratch bytes dspark_head_pick needs.
size_t dspark_head_scratch_bytes(int64_t n_vocab);
/// The head term of the copy for the block's draft rows, once: approx[r][copy row] for the `rows` (1..kDsDrafts) q8_1 rows of
/// hq (approx: kDsDrafts x dspark_screen_rows floats).
void dspark_head_base(const DsHead& h, const void* hq, int rows, float* approx, void* stream);
/// Draft i (from 0): over hq's row i, the latent `latq` and approx (dspark_head_base's): ids[i] = *prev = the pick (the lowest
/// row on a tie), conf[i] = its softmax probability (the rows not rescored at their approximate logits), w1_row = the pick's
/// row of w1 (the rescored rows' are read while they are scored, so the next step reads the card, not w1's host memory).
void dspark_head_pick(const DsHead& h, const void* hq, const void* latq, const float* approx, int i, void* scratch, int32_t* prev,
                      int32_t* ids, float* conf, const uint16_t* w1, uint16_t* w1_row, void* stream);
/// The reference: every row's exact logit (the arithmetic the pick rescores with) into logits[n_vocab].
void dspark_head_exact(const DsHead& h, const void* hq, const void* latq, int i, float* logits, void* stream);

/// The folds of the hidden_out projection into the head, made once: fold (Q8_0 [n_rows][kDsRank], ggml's reference
/// rounding) = head (Q8_0 [n_rows][kWidth]) times w (BF16 [kWidth][kDsRank]) in fp32, and fold_b[v] = head[v] . b.
void dspark_fold(const void* head_q8_0, int64_t n_rows, const uint16_t* w, const float* b, void* fold_q8_0, float* fold_b,
                 void* stream);

/// The recurrence's projection (BF16 [kDsRnn][2 kDsRank + 2 kWidth]: state, previous token, hidden, context columns) split
/// into w_sp [kDsRnn][2 kDsRank], w_h and w_f [kDsRnn][kWidth].
void dspark_split_rnn(const uint16_t* w, uint16_t* w_sp, uint16_t* w_h, uint16_t* w_f, void* stream);

// ---- batched multi-session commit & draft kernels -----------------------------------------------------------------------

struct DsCommitSegs {
    const basalt::kernels::WindowSegs* at_d = nullptr;
    float* fr[basalt::kernels::kWindowSegsMax] = {};
    uint16_t* k[basalt::kernels::kWindowSegsMax] = {};
    uint16_t* v[basalt::kernels::kWindowSegsMax] = {};
};

struct DsBatchFr {
    const float* fr[basalt::kernels::kWindowSegsMax] = {};
};

struct DsBatchKv {
    uint16_t* k[basalt::kernels::kWindowSegsMax] = {};
    uint16_t* v[basalt::kernels::kWindowSegsMax] = {};
};

void dspark_mix_batch(const float* tap, int64_t tap_stride, int rows, const DsMixArgs& a, float* out, int64_t out_stride,
                      const int32_t* pos, const DsCommitSegs& segs, int fr_cap, void* stream);

void dspark_kv_commit_batch(const float* kb, const float* vb, const float* k_norm, float eps, float rope_base,
                            const int32_t* pos, int rows, const DsCommitSegs& segs, int64_t cap, void* stream);

void dspark_block_inputs_batch(const uint16_t* embedding, const int32_t* meta, const float* embed_norm, float eps,
                               const float* mask, int block, int count, float* x, int32_t* prev, float* state,
                               const DsBatchFr& frs, int fr_cap, float* fr_out, const uint16_t* w1, uint16_t* w1_row,
                               void* stream);

void dspark_block_qk_prep_batch(const float* q, const float* kb, const float* vb, const float* q_norm, const float* k_norm,
                                float eps, float rope_base, const int32_t* meta, int block, int count, float* q_out,
                                const DsBatchKv& kv, int64_t cap, void* stream);

void dspark_pack_draft_rows(const float* h_chunk, int block, int count_chunk, int slot0, int n_max, float* hd_all,
                            void* stream);

void dspark_head_base_batch(const DsHead& h, const void* hq, int count, int n_max, float* approx, void* stream);

void dspark_markov_step_batch(const float* ctx_h, const float* ctx_f, const float* bias, const uint16_t* w_sp,
                              const uint16_t* w1_row, float* state, int step_i, int n_max, int count, float* latent,
                              void* stream);

void dspark_head_pick_batch(const DsHead& h, const void* hq, const void* latq_step, const float* approx, int step_i,
                            int n_max, int count, void* scratch, size_t scratch_stride, int32_t* prev, int32_t* ids,
                            float* conf, const uint16_t* w1, uint16_t* w1_row, void* stream);

}  // namespace basalt::models::inkling2
