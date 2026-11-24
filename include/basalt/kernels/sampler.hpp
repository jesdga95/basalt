// include/basalt/kernels/sampler.hpp - the sampler chain, host-callable.
//
//     penalties -> top_k -> tfs -> typical -> top_a -> top_p -> min_p -> xtc -> temperature -> pick
//
// llama.cpp's default chain: ONE penalties stage, first, and temperature AFTER the truncation filters;
// top_p cuts before min_p.  tfs, typical, top_a and xtc run where llama.cpp's do (after top_k, before the
// temperature stage); xtc keeps the TAIL of the list (it drops top tokens, llama_sample_xtc_apply).  The
// temperature stage is temperature, or koboldcpp's dynatemp + smoothing when their keys are set (the
// temperature-0 greedy path never reaches the tail).  tfs/typical/top_a/xtc are list-scoped like
// koboldcpp's: they run on the top_k survivors, so they need top_k <= 64 (the narrow path) and are
// refused wider.  `src/kernels/cuda/sampler.cu` has the details and `sampler_parity` pins them.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace basalt::kernels {

// The sampled list: top_k as given, 0 (off) or more than the vocabulary the whole vocabulary.  The top_k kernels keep
// lists up to kSamplerNarrowMax; a longer one takes the wide path (sorted rows), which needs sampler_wide_reserve.
constexpr int kSamplerNarrowMax = 64;
inline int sampler_list_len(int top_k, int n_vocab) { return (top_k > 0 && top_k < n_vocab) ? top_k : n_vocab; }

struct SamplerParams {
    int top_k = 20;              // sampled path: as given; 0 (off), or more than the vocabulary, keeps every token
    float top_p = 0.95f;         // 1.0 disables the filter
    float min_p = 0.0f;          // 0 disables; keeps tokens with p >= min_p * p_max (a prefix of the top_k list)
    float temperature = 1.0f;    // <= 0 means greedy
    int min_keep = 1;            // top_p keeps at least this many
    // The horde/koboldcpp truncation filters, list-scoped (the top_k survivors, in selection order).  All off
    // at 0 (or typical at >= 1, tfs at >= 1).  xtc keeps the tail below xtc_thresh with chance xtc_prob
    // (llama_sample_xtc_apply), its draw from a Philox stream tagged apart from the pick's.  typical needs
    // the narrow path (k <= 64): it re-orders by |entropy-shifted score|, which the wide path does not do.
    float tfs_z = 0.0f;          // tail-free sampling's z: cut where the normalized second-derivative mass passes 1-z
    float typical_p = 0.0f;      // locally typical: keep the smallest shifted-score set holding this much mass
    float top_a = 0.0f;          // keep p >= top_a * p_max^2
    float xtc_prob = 0.0f;       // chance of dropping the top tokens at/above xtc_thresh (0..0.5)
    float xtc_thresh = 0.1f;
    // The temperature stage, koboldcpp's dynamic temperature and smoothing (sample_entropy/sample_temperature):
    // dynatemp off when dyna_low == dyna_high; else temp' = low + (high-low) * (normalized list entropy)^exp.
    // smoothing then reshapes the divided logits: l' = -(k*sf*(l-h)^2) + (s*sf*(l-h)^3) + h, h their max,
    // k = (3-curve)/2, s = (curve-1)/2, and the pick softmaxes those.  temperature 0 (greedy) never reaches here.
    float dyna_low = 0.0f;
    float dyna_high = 0.0f;
    float dyna_exp = 1.0f;
    float smooth_factor = 0.0f;  // 0 disables the smoothing transform
    float smooth_curve = 1.0f;
    int penalty_last_n = 0;      // 0 disables; the window over `history` to count occurrences in
    float penalty_repeat = 1.0f;
    float penalty_freq = 0.0f;
    float penalty_present = 0.0f;
    uint64_t seed = 0;           // drives Philox, which is counter-based on (seed, token index)
    uint64_t counter = 0;        // absolute draw index of row 0; advance across decode calls
    bool greedy = false;
    const float* logit_bias = nullptr;   // optional device array [n_vocab] added to the logits first; -inf bans
    int* top_ids = nullptr;      // optional device array [n_tokens][n_top]: the sampled path's kept list, best first,
    int n_top = 0;               // -1 after it (0..32 ids per row; the greedy path leaves it unwritten)
    // Drawn drafts (speculative sampling), device arrays: rows t < n_drawn carry the draft drafts[t] (the token after
    // row t), drawn from the proposal q_p[t][j] over the ids q_ids[t][j], j < q_n.  Such a row keeps its draft with
    // probability min(1, p(d) / q(d)) and otherwise draws from max(p - q, 0) renormalized: its token has the row's
    // distribution p exactly and is the draft whenever the draft is kept.  The sampled path's narrow lists only (64 or
    // fewer); the greedy path ignores them.
    // The request lets its drafter draw drafts for the rows above.  The reply then has the model's distribution, but which
    // tokens come out depends on the windows the draft policy picked, so a request that fixes its seed (a reproducible
    // reply) leaves this off and its drafts are kept only where the row's own draw is the draft.
    bool draw_drafts = false;
    const int* drafts = nullptr;
    const int* q_ids = nullptr;
    const float* q_p = nullptr;
    int q_n = 0;
    int n_drawn = 0;
};

// logits (n_tokens, n_vocab) -> one sampled token id per row in `out`.
//
// **`logits` AND `out` ARE DEVICE POINTERS.**  This is a CUDA kernel launch, not a host function, and nothing
// in the parameter names or the types says so - `const float* logits` reads exactly like a host buffer.
// Passing host memory for `logits` faults inside the kernel with an ILLEGAL MEMORY ACCESS that is
// reported by whatever synchronising call happens NEXT, which will be somewhere else entirely and will name a
// buffer that has nothing to do with it.  A sticky async fault does not know where it came from.
//
// `history` is (n_tokens, history_len) int32 - ONE ROW PER TOKEN ROW, at a stride of `history_len` - the most
// recent tokens that row's pick follows, with any unused slots set to -1; only the last `p.penalty_last_n` of
// each row are counted, and ids outside [0, n_vocab) are ignored.  Pass nullptr and 0 when no penalties apply.
// A verify window's rows need DIFFERENT histories: row t follows the window's drafts 1..t (`penalty_rows`).
void sample_tokens(const float* logits, int n_tokens, int n_vocab, const int* history, int history_len,
                   const SamplerParams& p, int* out, void* stream);

// sample_tokens, and on the sampled path also qprob[t] (t < n_query) = the probability of token query[t] under row
// t's final distribution (the chain's survivors after temperature; 0 outside them).  The picks are sample_tokens'
// own.  On the greedy path qprob is not written.  `query` and `qprob` are device (or mapped) pointers.
void sample_tokens_probe(const float* logits, int n_tokens, int n_vocab, const int* history, int history_len,
                         const SamplerParams& p, int* out, const int* query, int n_query, float* qprob, void* stream);

// sample_tokens_probe on a chosen implementation of the sampled path's split, for sampler_parity: 0 the two-launch
// split, 1 the one-launch split with argmax rounds, 2 the one-launch split with bitonic networks (the engine's), 3 the
// wide path (sorted rows; the engine's for a list longer than 64).  All pick the same token and give the same
// probabilities, bit for bit.
void sample_tokens_split(int split, const float* logits, int n_tokens, int n_vocab, const int* history,
                         int history_len, const SamplerParams& p, int* out, const int* query, int n_query, float* qprob,
                         void* stream);

// The wide path's buffers on `stream` (about 16 bytes per vocabulary entry plus the sort's), made now: false when the
// memory cannot be had.  A request that samples a list longer than 64 entries reserves them before its first window.
bool sampler_wide_reserve(int n_vocab, void* stream);

// The greedy pick (no penalties) on a thread-block cluster of 8 CTAs per row (needs thread-block clusters): the same token as
// sample_tokens' one-block argmax.  False (nothing launched) where it
// cannot run: a device without thread-block clusters.  Device pointers; capturable.
bool sample_greedy_cluster(const float* logits, int n_tokens, int n_vocab, int* out, void* stream);

// The draft head's tail in one launch (MTP_FUSE_TOP): sample_greedy_cluster's pick, its probability summed as
// row_top_prob sums it (verify_kernels.hpp), then map_ids through `map` (nullptr: none).  Bitwise those three calls;
// false (nothing launched) where sample_greedy_cluster cannot run.  Device pointers; capturable.
bool sample_greedy_top_prob(const float* logits, int n_tokens, int n_vocab, const int32_t* map, int* out, float* probs,
                            void* stream);

// The penalty-history rows of a verify window, on the host: row t of `out` (T rows of `h` slots) is the last `h`
// tokens of `tail[0..n_tail)` followed by `window[0..t]`, most recent LAST, -1 in the unused front slots.
// `tail` is what the state consumed before the window, `window[0]` the fed-back token and `window[1..]` the
// drafts: row t is exactly the history plain decode counts when it picks the token after `window[t]`, so drafting
// cannot change which tokens are penalised. 
inline void penalty_rows(const int32_t* tail, int64_t n_tail, const int32_t* window, int T, int h, int32_t* out) {
    for (int t = 0; t < T; ++t) {
        int32_t* row = out + (size_t) t * (size_t) h;
        const int64_t avail = n_tail + t + 1;                      // tail + window[0..t]
        const int take = (int) std::min<int64_t>(h, avail);
        std::fill(row, row + (h - take), -1);
        for (int j = 0; j < take; ++j) {
            const int64_t i = avail - take + j;                     // index into (tail, window)
            row[h - take + j] = i < n_tail ? tail[i] : window[i - n_tail];
        }
    }
}

}  // namespace basalt::kernels
