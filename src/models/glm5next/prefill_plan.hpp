// src/models/glm5next/prefill_plan.hpp - what a prompt chunk costs the forward pass, and how long a chunk the free
// device memory can carry.
//
// Free of CUDA, so a test can drive it without a device (the convention the other models' policy headers follow: the
// arithmetic next to the model, its test beside it).
//
// One piece of the window's scratch grows with the *context* rather than the chunk: the DSA indexer scores every pool
// the sequence has filled, so its score row is context/kpool wide. That is the term a dense model does not have, and
// the one that ends a long prompt's chunk on a long context.
//
// The byte count is an upper bound, and the forward test checks it against what GlmForward::grow actually allocates,
// so the two cannot drift apart silently: a scratch that outgrew its plan shows up as a failing test rather than as an
// out-of-memory in the middle of somebody's request.
#pragma once

#include "models/glm5next/pack.hpp"

#include <cstdint>

namespace basalt::models::glm5next::prefillp {

constexpr int kMaxChunk = 8192;            // past this, a chunk is limited by the memory bound anyway
constexpr uint64_t kMargin = 1ull << 26;   // slack beside a chunk for what the plan still does not count (a few MB of
                                           // mmq activation rows and cudaMalloc rounding; the sparse attention's
                                           // workspace is no longer among them - check_fits carries it per token)

/// The 4-byte units of device scratch the forward pass allocates for a window of `tokens` over a `context`-token
/// sequence: the per-token rows, the per-token hyper-connection coefficients and mix workspace, and the per-window
/// pieces - of which the score row and the pool member table grow with the context.
///
/// Every term here has a cudaMalloc of its own in GlmForward::grow(). What is NOT here is the sparse attention's
/// workspace, whose size belongs to that kernel (`mla_sparse_workspace_floats`); a caller adding this to a free-memory
/// budget adds that alongside, and the forward test checks the sum against what the pass really allocated.
/// cudaMalloc's own rounding to 256 bytes is counted as slack below.
inline uint64_t window_units(const Geometry& g, uint64_t context, int tokens) {
    const uint64_t T = (uint64_t) (tokens > 0 ? tokens : 0);
    const uint64_t E = (uint64_t) g.n_embd, D = (uint64_t) g.n_head * g.kda_head_dim;
    const uint64_t idw = (uint64_t) g.idx_dim;
    const uint64_t n_pool = g.kpool > 0 ? context / (uint64_t) g.kpool : 0;
    const uint64_t n_top = g.kpool > 0 ? (uint64_t) g.idx_top_k / (uint64_t) g.kpool : 0;
    const uint64_t sh = (uint64_t) g.n_ff_exp * g.n_shared;
    const uint64_t wide = (uint64_t) (g.n_ff > g.n_ff_exp ? g.n_ff : g.n_ff_exp);
    const uint64_t wider = (uint64_t) (g.kv_lora > g.n_head * g.v_dim ? g.kv_lora : g.n_head * g.v_dim);
    const uint64_t pend = g.kpool > 1 ? (uint64_t) (g.kpool - 1) : 1;

    const uint64_t per_token = 2ull * 4 * E                            // the streams, both buffers
                             + 3ull * E                                // collapsed, normed, head-collapsed
                             + 3ull * D                                // the KDA front's q|k|v (three D-wide slabs)
                             + 4ull * D                                // decay, out gate, read, normed read
                             + 2ull * g.kda_head_dim                   // the f and g paths' first projection
                             + (uint64_t) g.n_head                     // the beta projection
                             + (uint64_t) g.q_lora * 2                 // the low-rank query and its norm
                             + (uint64_t) g.n_head * g.qk_dim          // the query, per head
                             + 2ull * g.n_head * g.kv_lora             // the absorbed query, the attention's output
                             + (uint64_t) g.n_head *               // the per-head products' head-major copy, as wide as
                                   (g.qk_dim > g.kv_lora ? g.qk_dim : g.kv_lora)  // the wider of the two products' inputs
                             + (g.kv_lora > g.v_dim ? g.kv_lora : g.v_dim)  // one head's product rows, staged before they
                                                                            // join the token-major row the next stage reads
                             + (uint64_t) g.kv_lora * 2                // the latent rows and their norm
                             + 3ull * idw                              // indexer keys, their norm, compressor gates
                             + (uint64_t) g.idx_heads * idw            // the indexer's query
                             + (uint64_t) g.idx_heads                  // the indexer's query weights
                             + wider                                   // a product's second home
                             + 2ull * idw                              // the pools this window stages to compress
                             + 2ull * E                                // the attention's and the FFN's output
                             + (g.nextn > 0 ? 3ull * E : 0ull)         // the drafter's block, when the pack carries
                                                                        // one: the next token's embedding and the
                                                                        // [2E] row eh_proj reads
                             + 3ull * wide                             // the dense MLP's gate, up, activation
                             + 3ull * sh                               // the shared experts' gate, up, activation
                             + (uint64_t) g.n_expert                   // the router's logits
                             + 2ull * g.n_used                         // the router's ids and weights
                             + n_top + (2 * n_top + 1) + 3             // chosen pools, cells, counts, positions, lengths
                             + pend                                    // the tail cells
                             + 24 * 2                                  // the two sublayers' mixes
                             + 8 * 25                                  // the mixes' workspace (8 splits of coef+1)
                             + 8;                                      // the two constant collapse rows
    // ... and the rows a window holds once, per window rather than per token
    const uint64_t per_window = (uint64_t) 3 * (g.conv - 1) * D        // the KDA conv window
                              + 4 * D + g.n_head                        // the recurrence's inputs for one token
                              + 4 * g.kpool * idw;                     // the pool staging's slack
    // The per-token score row and the pool member table: what the context, not the chunk, decides.
    const uint64_t context_rows = n_pool * (T + (uint64_t) g.kpool);
    // cudaMalloc's per-block rounding, counted for every block grow() asks for.
    const uint64_t rounding = (T + 1 + (g.nextn > 0 ? 2 : 0)) * 64;
    return T * (per_token + n_pool) + context_rows + per_window + rounding;
}

/// The device bytes a window of `tokens` costs.
inline uint64_t window_bytes(const Geometry& g, uint64_t context, int tokens) {
    return window_units(g, context, tokens) * 4ull;
}

/// The longest window the free device memory can carry, `kMargin` of it left behind. The three caches are allocated
/// with the session, so `free_bytes` is what is left after them; the answer is 0 when nothing fits.
///
/// `extra_units_per_token` prices scratch the plan itself cannot see because it belongs to a kernel - the sparse
/// attention's workspace, sized by mla_sparse_workspace_floats. It is linear in the token count like the rest, so it
/// joins the slope; a caller that leaves it out under-books the window by however much that kernel costs per token.
///
/// window_units is linear in the token count, so a window's worth divides by one token's slope instead of searching.
inline int max_chunk(const Geometry& g, uint64_t context, uint64_t free_bytes, uint64_t extra_units_per_token = 0) {
    if (free_bytes <= kMargin || g.n_embd <= 0 || g.n_head <= 0) return 0;
    const uint64_t room = (free_bytes - kMargin) / 4ull;   // in the units window_units counts
    const uint64_t one = window_units(g, context, 1) + extra_units_per_token;
    const uint64_t zero = window_units(g, context, 0);
    const uint64_t slope = one > zero ? one - zero : one;
    if (slope == 0 || room < one) return 0;
    uint64_t fit = (room - zero) / slope;
    if (fit > (uint64_t) kMaxChunk) fit = kMaxChunk;
    if (fit > context) fit = context;
    return (int) fit;
}

}  // namespace basalt::models::glm5next::prefillp
