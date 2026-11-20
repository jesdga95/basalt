// include/basalt/models/inkling2/kernels.hpp - the Inkling 2 kernels: the block-residual mixes, the loop controllers'
// glue, the attention's short convolutions, clips and caches, attention with a per-query relative bias and sink, the
// post-sublayer convolutions, the experts' plan and sum, and the logit softcap.
//
// All activations are fp32 rows ([rows][width], row-major) and every pointer is device memory. A row's result never
// depends on how many rows share its call or where it sits among them. `stream` may be null (the call then synchronises).
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::models::inkling2 {

constexpr int kWidth = 2304;        ///< the hidden width the kernels are built for
constexpr int kHeads = 24;          ///< query heads
constexpr int kKvHeads = 8;         ///< key/value heads
constexpr int kHeadDim = 128;
constexpr int kRel = 16;            ///< relative state per head
constexpr int kExtent = 1024;       ///< distances the relative bias covers (also the window)
constexpr int kConv = 4;            ///< short convolution taps
constexpr int kConvRing = 16;       ///< positions a convolution ring keeps: a window of up to 8 rows and the 3 before it
constexpr int kMaxMix = 6;          ///< states one mix reads: the finished blocks and the partial

/// A layer's fused projection rows, in this order: q, r, gate, sink (read from the query input), k and the key
/// convolution's per-token taps (from the key input), v and its taps (from the value input). A sliding-window layer has no
/// taps (its rows are zero). The outputs land in three row-major arrays: yq [rows][kProjQ], yk and yv [rows][kProjKV].
constexpr int kOffQ = 0, kOffR = 3072, kOffGate = 3456, kOffSink = 6528, kProjQ = 6552;
constexpr int kOffTaps = 1024, kProjKV = 1056;
constexpr int kRowK = kProjQ, kRowV = kProjQ + kProjKV, kProj = kProjQ + 2 * kProjKV;   ///< first rows of the ranges

// ---- block residuals -------------------------------------------------------------------------------------------------

/// The states one mix reads: n of them, state i's row r at s[i] + r * kWidth.
struct MixStates {
    const float* s[kMaxMix] = {};
    int n = 0;
};

/// The block-attention residual: for each row, every state is normed (unweighted RMS, eps), each of the n_q queries
/// (query j at queries + j * kWidth) scores it as q . s / sqrt(kWidth), and out_j = sum_i softmax_i * normed s_i, all in
/// fp32. Output j of row r at out + j * out_stride + r * kWidth.
void attnres_mix(const MixStates& st, const float* queries, int n_q, int rows, float eps, float* out, int64_t out_stride,
                 void* stream);

// ---- loop controllers -------------------------------------------------------------------------------------------------

/// pair[r] = [rms(dest[r]) | rms(msg[r])] (unweighted, eps 1e-6): the controller's [2 kWidth] input.
void ctrl_pair_norm(const float* dest, const float* msg, int rows, float* pair, void* stream);
/// out[r] = (1 - tanh(f[r])) dest[r] + tanh(g[r]) msg[r], with gf[r] = [g | f] (2 kWidth). `out` may be `dest`.
void ctrl_blend(const float* dest, const float* msg, const float* gf, int rows, float* out, void* stream);

// ---- attention ---------------------------------------------------------------------------------------------------------

/// What a visit's attention preparation reads and writes.
struct AttnPrep {
    const float* yq = nullptr;         ///< the projection outputs [rows][kProjQ], [rows][kProjKV], [rows][kProjKV]
    const float* yk = nullptr;
    const float* yv = nullptr;
    const int32_t* pos = nullptr;      ///< position of each row [rows]
    bool full = false;                 ///< a full-attention layer: per-token taps, log-scaled q and r
    const float* k_conv = nullptr;     ///< the key convolution's static taps [1024][kConv], tap kConv-1 the current token
    const float* v_conv = nullptr;
    const float* q_scale = nullptr;    ///< [kHeads][kHeadDim]
    const float* r_scale = nullptr;    ///< [kHeads][kRel]
    float qk_clip = 22.627417f, rel_clip = 8.f, eps = 1e-6f;
    float log_floor = 8192.f, log_alpha = 0.1f;
    float* k_ring = nullptr;           ///< the convolution inputs of the last kConvRing positions [kConvRing][1024], slot pos % kConvRing
    float* v_ring = nullptr;
    void* k_cache = nullptr;           ///< [kKvHeads][cap][kHeadDim], position p at p % cap; fp16 or E4M3 (fp8 set)
    void* v_cache = nullptr;
    int64_t cap = 0;
    bool fp8 = false;                  ///< unit-scale E4M3 codes (keys are clipped and values normed, so they fit)
    float* q = nullptr;                ///< out: the clipped (and scaled) queries [rows][kHeads][kHeadDim]
    float* r = nullptr;                ///< out: the relative states [rows][kHeads][kRel]
    float* sink = nullptr;             ///< out: [rows][kHeads]
};
/// Convolves the window's keys and values (positions before the window read from the rings, below position 0 zero),
/// clips the keys, norms the values, writes both to the cache, and prepares q, r and the sinks. Then the rings take the
/// window's convolution inputs. `rows` <= 8 in a decode window; a prompt chunk passes up to its whole length.
void attn_prepare(const AttnPrep& a, int rows, void* stream);

/// What one attention call reads.
struct AttnArgs {
    const float* q = nullptr;          ///< [rows][kHeads][kHeadDim]
    const float* r = nullptr;          ///< [rows][kHeads][kRel]
    const float* sink = nullptr;       ///< [rows][kHeads]
    const float* rel_table = nullptr;  ///< [kRel][kExtent]: bias(d) = r . table[:, d] for 0 <= d < kExtent
    const int32_t* pos = nullptr;      ///< [rows]
    const void* k_cache = nullptr;
    const void* v_cache = nullptr;
    int64_t cap = 0;                   ///< the cache's cells (a ring when window is set)
    bool fp8 = false;
    int window = 0;                    ///< > 0: keys at distance < window only (the cache is then a ring of cap >= window + rows)
    float scale = 0.08838834764831845f;
};
/// Bytes of scratch attention_decode needs for `rows` rows over at most `max_keys` keys.
size_t attention_decode_scratch(int rows, int64_t max_keys);
/// out[r][h] = sum_j softmax_j(scale q.k_j + bias(pos_r - pos_j)) v_j over the keys at positions <= pos_r (and within the
/// window), with exp(sink[r][h]) beside them in the softmax's denominator. `max_keys` bounds the keys any row reads.
void attention_decode(const AttnArgs& a, int rows, int64_t max_keys, void* scratch, float* out, void* stream);

/// out[r] = attn[r] * sigmoid(yq[r][kOffGate..]), the attention output gated by its own row's gate projection.
void attn_gate(const float* attn, const float* yq, int rows, float* out, void* stream);

// ---- the sublayers' output convolutions --------------------------------------------------------------------------------

/// conv[r] = (x[r] + sum_j taps[c][j] x[p - 3 + j][c]) / sqrt(kConv + 1) over kWidth channels, positions before the window
/// from `ring` (zero below 0); the ring then takes the window's inputs. Then out[r] = conv[r] + (add ? add[r] : 0).
/// `out` may be `add`.
void post_conv(const float* x, const int32_t* pos, const float* taps, float* ring, const float* add, int rows, float* out,
               void* stream);

// ---- experts -----------------------------------------------------------------------------------------------------------

/// The grouped expert kernel's plan from the router's choices: group g is an expert (blob at base + e * blob_bytes) and its
/// entries are the (row, slot) pairs that chose it, in row order; entry e of the pairs writes row r * k + slot of the
/// output. ptr [k rows], start [k rows + 2], n_groups [1], dst and tok [k rows]. rows * k <= 1024.
void moe_plan(const int32_t* ids, int rows, int k, int n_expert, const uint8_t* base, size_t blob_bytes,
              unsigned long long* ptr, int32_t* start, int32_t* n_groups, int32_t* dst, int32_t* tok, void* stream);
/// y[r] = sum_slot w[r][slot] * out[r * k + slot] in slot order (fp32).
void moe_combine(const float* out, const float* w, int rows, int k, float* y, void* stream);

// ---- head --------------------------------------------------------------------------------------------------------------

/// x = cap * tanh(x / cap) over rows of n values (row stride ld).
void softcap(float* x, int rows, int n, int64_t ld, float cap, void* stream);

}  // namespace basalt::models::inkling2
