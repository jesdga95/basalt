// include/basalt/models/inkling2/kernels.hpp - the Inkling 2 kernels: the block-residual mixes, the loop controllers'
// glue, the attention's short convolutions, clips and caches, attention with a per-query relative bias and sink, the
// post-sublayer convolutions, the experts' plan and sum, and the logit softcap; and the decode window's fused forms of
// them (kernels/glue.cu), which hand the products their inputs as q8_1 blocks.
//
// All activations are fp32 rows ([rows][width], row-major) and every pointer is device memory. A row's result never
// depends on how many rows share its call or where it sits among them. `stream` may be null (the call then synchronises).
#pragma once

#include "basalt/kernels/window_segs.hpp"

#include <cstddef>
#include <cstdint>

namespace basalt::models::inkling2 {

constexpr int kWidth = 2304;        ///< the hidden width the kernels are built for
constexpr int kHeads = 24;          ///< query heads
constexpr int kKvHeads = 8;         ///< key/value heads
constexpr int kHeadDim = 128;
constexpr int kKvInt8RowBytes = (kHeadDim + 4 + 15) & ~15;   ///< INT8 values and two fp16 scales, zero-padded to 16 bytes
constexpr int kv_cache_row_bytes(bool fp8, bool int8) { return fp8 ? kHeadDim : int8 ? kKvInt8RowBytes : 2 * kHeadDim; }
constexpr int kRel = 16;            ///< relative state per head
constexpr int kExtent = 1024;       ///< distances the relative bias covers (also the window)
constexpr int kConv = 4;            ///< short convolution taps
constexpr int kConvRing = 16;       ///< positions a convolution ring keeps: a window of up to 8 rows and the 3 before it
constexpr int kMaxMix = 6;          ///< states one mix reads: the finished blocks and the partial
constexpr int kDecodeRows = 8;      ///< rows of a decode window's fused kernels (the products take 8 columns at most)
constexpr int kMaxStreams = 4;      ///< loop controllers one call runs side by side

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

// ---- batch windows ----------------------------------------------------------------------------------------------------

/// One conversation's caches of a visit in a batch window: what its rows read and write in place of the arguments' own.
struct SegCaches {
    void* k = nullptr;
    void* v = nullptr;
    float* k_ring = nullptr;
    float* v_ring = nullptr;
    float* a_ring = nullptr;
    float* m_ring = nullptr;
};
/// A decode window of several conversations (docs/design/batching.md): segment i's rows (at.row0[i]..) take c[i]'s caches,
/// and a row's convolutions read only the rows of its own segment before it. at.n = 0 (the default): one conversation, the
/// arguments' own caches. Every row gives the bits it gives in its conversation's own call.
struct Segments {
    basalt::kernels::WindowSegs at;
    SegCaches c[kDecodeRows];
};

// ---- attention ---------------------------------------------------------------------------------------------------------

/// What a visit's attention preparation reads and writes.
struct AttnPrep {
    const float* yq = nullptr;         ///< the projection outputs: row r's at yq + r * ldq, yk + r * ldk, yv + r * ldv
    const float* yk = nullptr;
    const float* yv = nullptr;
    int64_t ldq = kProjQ, ldk = kProjKV, ldv = kProjKV;
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
    void* k_cache = nullptr;           ///< [cap][kKvHeads][kHeadDim], position p in cell p % cap; fp16 or E4M3 (fp8 set)
    void* v_cache = nullptr;
    int64_t cap = 0;
    bool fp8 = false;                  ///< unit-scale E4M3 codes (keys are clipped and values normed, so they fit)
    bool int8 = false;                 ///< symmetric INT8 codes with two per-row FP16 scales
    float* q = nullptr;                ///< out: the clipped (and scaled) queries [rows][kHeads][kHeadDim]
    float* r = nullptr;                ///< out: the relative states [rows][kHeads][kRel]
    float* sink = nullptr;             ///< out: [rows][kHeads]
    Segments seg;                      ///< a batch window's (decode_prepare only): each segment's caches and rings
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
    bool int8 = false;
    int window = 0;                    ///< > 0: keys at distance < window only (the cache is then a ring of cap >= window + rows)
    float scale = 0.08838834764831845f;
    const float* gate = nullptr;       ///< set: the output is gated by sigmoid(gate[r * gate_ld + h * kHeadDim + d])
    int64_t gate_ld = 0;
    void* q8 = nullptr;                ///< set: the (gated) output also as q8_1 blocks, [rows][kHeads * kHeadDim / 32]
    Segments seg;                      ///< a batch window's: each segment's caches
};
/// The kernels' shared-memory opt-ins are per device (a layer split runs them on two cards): the current device's index
/// into their flags, below kOptInDevices.
constexpr int kOptInDevices = 64;
int opt_in_device();

/// Bytes of scratch attention_decode needs for `rows` rows over at most `max_keys` keys.
size_t attention_decode_scratch(int rows, int64_t max_keys);
/// out[r][h] = sum_j softmax_j(scale q.k_j + bias(pos_r - pos_j)) v_j over the keys at positions <= pos_r (and within the
/// window), with exp(sink[r][h]) beside them in the softmax's denominator. `max_keys` bounds the keys any row reads (it
/// sizes the launch only: a row's keys split in slices fixed by its own key count). `out` may be null when q8 is set.
void attention_decode(const AttnArgs& a, int rows, int64_t max_keys, void* scratch, float* out, void* stream);

/// out[r] = attn[r] * sigmoid(yq[r][kOffGate..]), the attention output gated by its own row's gate projection.
void attn_gate(const float* attn, const float* yq, int rows, float* out, void* stream, int64_t yq_ld = kProjQ);

// ---- the sublayers' output convolutions --------------------------------------------------------------------------------

/// conv[r] = (x[r] + sum_j taps[c][j] x[p - 3 + j][c]) / sqrt(kConv + 1) over kWidth channels, positions before the window
/// from `ring` (zero below 0); the ring then takes the window's inputs. Then out[r] = conv[r] + (add ? add[r] : 0).
/// `out` may be `add`.
void post_conv(const float* x, const int32_t* pos, const float* taps, float* ring, const float* add, int rows, float* out,
               void* stream);

// ---- experts -----------------------------------------------------------------------------------------------------------

/// Doorbell words in mapped host memory, accessed atomically at system scope. The ring's release publishes the choices,
/// RAM rows and nonzero visit ticket. The host acquires the ring, takes the ticket, clears the ring and computes its rows.
/// Its release store to `done` publishes those rows with that ticket. The merge acquires only a completion for its visit.
/// `abort` is sticky: a RAM visit that observes it does not publish, and waiting merges check it periodically.
struct CpuBell {
    alignas(64) uint32_t ring;
    uint32_t ticket;
    alignas(64) uint32_t done;
    uint32_t abort;
};

/// Each visit with a RAM pair advances the device word `seq` (initialized to zero), even after an abort. A live RAM visit
/// publishes its choices to ids_m, its RAM rows to x_m, and its ticket before raising `ring` with `layer` and `rows`.
/// A visit with no RAM pair touches neither `seq` nor mapped host memory. rows <= kDecodeRows, rows * k <= 1024.
void moe_cpu_ring(const int32_t* ids, int rows, int k, const unsigned long long* blobs, const float* x, int layer, float* x_m,
                  int32_t* ids_m, CpuBell* bell, uint32_t* seq, void* stream);

/// The CPU's rows of a window's experts (cpu_experts.hpp): pair p = r * k + slot whose expert is not on the card (blobs[e]
/// 0) takes row p of `cpu` (mapped host memory, the pool's output) into row p of `output`, and adds its lookup to tally[e]
/// when `tally` is set; the card's pairs are left as they are. With `bell` set, wait for its completion ticket to match the
/// visit's `seq`, or take nothing if abort is seen first. Visits sharing a sequence run ring and merge in order.
void moe_cpu_merge(const int32_t* ids, int rows, int k, const unsigned long long* blobs, const float* cpu, float* output,
                   uint32_t* tally, void* stream, CpuBell* bell = nullptr, const uint32_t* seq = nullptr);

/// The grouped expert kernel's plan from the router's choices: group g is an expert on the card (its blob at blobs[e], a
/// device address; 0 when it is not on the card) and its entries are the (row, slot) pairs that chose it, in row order;
/// an entry writes row r * k + slot of the output. A pair whose expert is not on the card is left out (the caller
/// supplies its row). ptr [k rows], start [k rows + 2], n_groups [1], dst and tok [k rows]. rows * k <= 1024. With
/// `tally` [n_expert], each expert on the card adds its pairs to its entry (the console's expert map).
void moe_plan(const int32_t* ids, int rows, int k, int n_expert, const unsigned long long* blobs, unsigned long long* ptr,
              int32_t* start, int32_t* n_groups, int32_t* dst, int32_t* tok, void* stream, uint32_t* tally = nullptr,
              unsigned long long* rec_blobs = nullptr, int32_t* rec_n = nullptr);
constexpr int kAheadExperts = 8;   ///< the groups of a visit moe_plan records (rec_blobs, rec_n) for the next window
/// Requests into L2 the first `bytes` of each of the *n blobs (moe_plan's record of a visit), 32 blocks each keeping one
/// 128 KiB step in flight: run on a side stream while the glue before the experts runs; reads only.
void expert_read_ahead(const unsigned long long* blobs, const int32_t* n, uint64_t bytes, void* stream);
/// Requests into L2 the first `bytes` of the blobs of the distinct experts among ids[0..n) (blobs[e]: expert e's device
/// address, 0 when it is not on the card: skipped; a guess at a visit's experts), by 16-byte loads over every block: run on a side stream while the attention and the glue
/// before the experts run. Reads only; `sink` (4 bytes) is written on a 2^-32 chance so the loads are kept.
void expert_prefetch(const int32_t* ids, int n, const unsigned long long* blobs, uint64_t bytes, uint32_t* sink, void* stream);
/// y[r] = sum_slot w[r][slot] * out[r * k + slot] in slot order (fp32).
void moe_combine(const float* out, const float* w, int rows, int k, float* y, void* stream, const int32_t* ids = nullptr);

// ---- a decode window's fused glue (kernels/glue.cu): at most kDecodeRows rows --------------------------------------------
//
// A window's rows sit at consecutive positions. Each row's convolutions read the rows before it in the window and the rings
// for the positions before the window, and write the row's own ring slot. The products' inputs come out as q8_1 blocks
// (row r at r * width / 32 blocks of 36 bytes: an fp16 scale and sum, 32 int8).

/// Mixes, then norms: output j is the mix of `st` scored by query j (queries + j * kWidth), or the last state as it is
/// when `queries` is null; then its RMS norm with norm[j] (eps) rounded to q8_1 at q8[j], and output 0's
/// normed rows in fp32 at norm_out when set. With `mix_out` set the mixes are written there instead (output j's row r at
/// mix_out + j * out_stride + r * kWidth) and nothing is normed.
struct MixNorm {
    MixStates st;
    const float* queries = nullptr;
    const float* norm[3] = {};
    void* q8[3] = {};
    float* norm_out = nullptr;
    float eps = 1e-6f;
    float* mix_out = nullptr;
    int64_t out_stride = 0;
    /// set: the attention's older cells (positions lo..kv_pos[0]-1, lo = kv_pos[0] - kv_window + 1 or 0, when there are at
    /// most 256 of them) are requested into L2 before the wait, so the attention after the projections reads them from L2
    const int32_t* kv_pos = nullptr;
    const void* kv_k = nullptr;
    const void* kv_v = nullptr;
    int64_t kv_cap = 0;
    int kv_window = 0;
    int kv_cell = 0;                   ///< bytes of a cell (every key/value head)
    /// set with kv_pos: the convolution rings' entries of the three positions before the window (the attention's key and
    /// value rings, [kConvRing][1024], and the attention output's, [kConvRing][kWidth]) into L2 the same way, whatever
    /// the context
    const float* ring_k = nullptr;
    const float* ring_v = nullptr;
    const float* ring_a = nullptr;
};
/// n_out: 1 to 3, or any number with mix_out (the messages of a whole pass).
void mix_norm(const MixNorm& a, int n_out, int rows, void* stream);

/// Up to kMaxStreams loop controllers over the same rows: stream s blends dest[s] with msg[s] (rows at r * kWidth),
/// out = (1 - tanh f) dest + tanh g msg with [g | f] = up_s . (down_s . [rms(dest) | rms(msg)]), then norms the blend with
/// norm[s] (eps) into q8_1 at q8[s]; stream 0's normed rows also in fp32 at norm_out when set. down_s is BF16
/// [rank][2 kWidth], up_s BF16 [2 kWidth][rank] (the gain rows, then the forget rows). `scratch`: ctrl_scratch_floats.
struct CtrlArgs {
    const float* dest[kMaxStreams] = {};
    const float* msg[kMaxStreams] = {};
    const uint16_t* down[kMaxStreams] = {};
    const uint16_t* up[kMaxStreams] = {};
    const float* norm[kMaxStreams] = {};
    void* q8[kMaxStreams] = {};
    float* norm_out = nullptr;
    float* scratch = nullptr;
    int rank = 0;
    float eps = 1e-6f;
};
size_t ctrl_scratch_floats(int rank);
void controllers(const CtrlArgs& a, int streams, int rows, void* stream);

/// attn_prepare for a decode window: a block per row and key/value head.
void decode_prepare(const AttnPrep& a, int rows, void* stream);

/// A decode window's attention from its projections: decode_prepare(pa) then attention_decode(a) into a.q8 (which must be
/// set; a's q, r and sink are pa's outputs), the same bits. A window of up to 3 rows whose keys split in at most 8 slices
/// (max_keys up to 1024) runs as one kernel: a cluster of blocks a row and key/value head prepares, attends and merges. A
/// batch window (pa.seg and a.seg set alike) keeps the two calls.
void attention_window(const AttnPrep& pa, const AttnArgs& a, int rows, int64_t max_keys, void* scratch, void* stream);

/// After the attention's output projection: its convolution over `o` (the ring for the positions before the window) plus
/// the residual (the layer's input `hid`, unless the layer starts a block) into `resid`; then the MLP's input, the mix of
/// the finished blocks (and `hid` when the layer starts a block) and `resid` scored by `query`, normed with `norm` (eps)
/// into q8_1 at `q8` and fp32 at `norm_out`, or the mix as it is at `mix_out` when set (for a loop controller).
struct PostAttn {
    const float* o = nullptr;
    const int32_t* pos = nullptr;
    const float* taps = nullptr;       ///< [kWidth][kConv], tap kConv-1 the current token
    float* ring = nullptr;             ///< the convolution's inputs [kConvRing][kWidth]
    const float* hid = nullptr;
    bool boundary = false;
    float* resid = nullptr;
    const float* blocks[kMaxMix] = {};
    int n_blocks = 0;
    const float* query = nullptr;
    const float* norm = nullptr;
    float eps = 1e-6f;
    void* q8 = nullptr;
    float* norm_out = nullptr;
    float* mix_out = nullptr;
    Segments seg;                      ///< a batch window's: each segment's ring (a_ring)
};
void post_attn(const PostAttn& a, int rows, void* stream);

/// The experts' sum of each row (w[r][slot] * rows[r * k + slot], slot order), its convolution (the ring for the positions
/// before the window) plus `resid` into `out`.
struct MoeFinish {
    const float* rows = nullptr;       ///< [rows][k][kWidth]
    const float* w = nullptr;          ///< [rows][k]
    int k = 0;
    const int32_t* pos = nullptr;
    const float* taps = nullptr;
    float* ring = nullptr;
    const float* resid = nullptr;
    float* out = nullptr;
    float* tap = nullptr;              ///< set: the output rows also here (the drafter's copy of the next visit's input)
    Segments seg;                      ///< a batch window's: each segment's ring (m_ring)
};
void moe_finish(const MoeFinish& a, int rows, void* stream);

// ---- head --------------------------------------------------------------------------------------------------------------

/// x = cap * tanh(x / cap) over rows of n values (row stride ld).
void softcap(float* x, int rows, int n, int64_t ld, float cap, void* stream);

}  // namespace basalt::models::inkling2
