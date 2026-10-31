// include/basalt/kernels/sampler.hpp - the sampler chain, host-callable.
//
//     penalties -> top_k -> top_p -> min_p -> temperature -> pick
//
// llama.cpp's default chain: ONE penalties stage, first, and temperature AFTER the truncation filters;
// top_p cuts before min_p.  `src/kernels/cuda/sampler.cu` has the details and `sampler_parity` pins them.
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
