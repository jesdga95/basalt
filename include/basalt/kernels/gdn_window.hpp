// include/basalt/kernels/gdn_window.hpp - the gated delta-net (GDN) block of a decode window, and its commit.
//
//   dec_gdn_rec      conv1d + SiLU + q/k L2 norm, the alpha/beta gates, the delta-rule recurrence over the
//                    window from the untouched state, the gated RMS norm, and the q8_1 of its output
//   gdn_commit_*     once acceptance is known: the first n_keep rows replayed into the state and conv history
//
// A window never writes the recurrent state, so a rejected draft never touches it and nothing is snapshotted.  Every
// value is the one the per-stage kernels produce (the same operations on the same values in the same order), so a
// token's outputs do not depend on T or its column.  Needs thread-block clusters.
//
// Geometry: state size 128; key and value heads, the block width (a multiple of 256 up to 5120) and the output
// gate's activation (GdnGate) are the model's.  State layout [S rows][h_v][S cols] fp32, then the conv history
// [C][3]; value head h meets key head h % h_k.
#pragma once
#include <cstddef>
#include <cstdint>

namespace basalt::kernels {

constexpr int kDecMaxT = 16;   // rows a GDN window kernel takes (a dense decode window's)

/// The activation on z in the closing gated RMS norm: y = rms_norm(o) * gamma * act(z).
enum class GdnGate : int32_t { Sigmoid = 0, Silu = 1 };

struct DecRecArgs {
    const float* state = nullptr;    // [S rows][h_v][S cols] fp32, then the conv history [C][3]
    const float* qkv = nullptr;      // raw projections, token t at qkv + t * C
    const float* conv_w = nullptr;   // [C][4]
    const float* x = nullptr;        // the block input (fp32), token t at x + t * n_embd
    const uint16_t* w_alpha = nullptr;   // bf16 [h_v][n_embd]
    const uint16_t* w_beta = nullptr;
    const float* dt = nullptr;
    const float* ssm_a = nullptr;
    const float* z = nullptr;        // token t at z + t * h_v * S
    const float* gamma = nullptr;    // [S]
    float* h = nullptr;              // conv output (q/k normalized), token t at h + t * C
    float* gate = nullptr;           // token t at gate + t * h_v
    float* beta = nullptr;
    float* y = nullptr;              // optional fp32 output, token t at y + t * h_v * S
    uint16_t* y16 = nullptr;         // optional bf16 output, laid out as y
    float* o = nullptr;              // dec_gdn_rec_split: the raw (pre-norm) output, laid out as y
    void* yq = nullptr;              // q8_1 of the output, token t at block t * h_v * S / 32
    int C = 0, n_embd = 0, h_k = 0, h_v = 0;
    float eps = 0.0f;
    // Head map (gdn_head_split.hpp): with nk > 0 the launch runs only key heads [k0, k0 + nk) and their value heads,
    // in place in the layer's full state, conv weights, gates and h (strides C, h_v).  qkv and z are then the card's
    // packed rows (qkv stride (2 nk + nv) S, z stride nv S), and y, o and yq are written packed (stride nv S).
    // qkv_out, when set, gets the raw qkv channels of these heads at the full layout (the commit's conv history).
    // nk = 0: every head, the layer's own layout.
    int k0 = 0, nk = 0;
    float* qkv_out = nullptr;
    // state_out (dec_gdn_rec, nk = 0): the state and conv history after all T rows, laid out as `state` (which is not
    // written) - the commit of all T rows, bitwise, without its pass
    float* state_out = nullptr;
    unsigned long long* stamps = nullptr;   // bench: 10 per-phase timestamps per block
    bool early_trigger = true;   // let the next kernel launch once the constants are loaded
    GdnGate gate_act = GdnGate::Sigmoid;
};
/// The block widths dec_gdn_rec is built for.
bool dec_gdn_rec_supported(int n_embd);
/// The recurrence of tokens [0, n_tok) from the untouched state; one cluster of four blocks per value head (of the
/// head map's value heads when a.nk > 0).
void dec_gdn_rec(const DecRecArgs& a, int n_tok, void* stream);

/// A window of several sequences ("segments"): each segment is n consecutive rows with its own GDN state.  The table
/// has one entry per window row (a segment's rows all carry the segment's fields), uploaded to the device per window.
/// A kernel launched over the entries lets only the segment's first row (j == 0) do the segment's work.
struct SegRow {
    float* gdn;        // the segment's GDN state of layer 0 (layer l's state at gdn + l * state floats)
    int32_t row0;      // the segment's first window row
    int32_t j;         // this row's index in its segment
    int32_t n;         // the segment's rows
    int32_t n_keep;    // rows of the segment to commit
    int32_t slot;      // the sequence's slot
    int32_t pos;       // this row's position
};
static_assert(sizeof(SegRow) == 32, "SegRow is copied as eight words");

/// dec_gdn_rec over a segmented window: each segment's tokens from its own state (the layer's at rows[].gdn +
/// state_off), one launch for all n_rows window rows; a.state is unused.  max_n: the most rows a segment holds (at
/// most kDecMaxT).  Every output is the segment's own dec_gdn_rec's, bitwise.
void dec_gdn_rec_seg(const DecRecArgs& a, const SegRow* rows, size_t state_off, int n_rows, int max_n, void* stream);
/// dec_gdn_rec in two launches: the split kernel (conv, gates, recurrence; never reads z) writes the raw output to
/// a.o, dec_gdn_rec_norm applies the gated RMS norm and writes a.yq (and a.y); the same bits as dec_gdn_rec.  The z
/// projection can run beside the split kernel.
void dec_gdn_rec_split(const DecRecArgs& a, int n_tok, void* stream);
void dec_gdn_rec_norm(const DecRecArgs& a, int n_tok, void* stream);
/// The verify window's GDN block around native q8_1 + qkv/z/out projections, as run: dec_gdn_rec_split beside the z
/// projection (z forked after the q8_1, beside qkv too), then dec_gdn_rec_norm.  Mode 1 is dec_gdn_rec after z, 2 forks
/// z after qkv.
inline constexpr int dec_gdnf_mode() { return 3; }

/// history <- the last 3 entries of [history | qkv_0 .. qkv_{n-1}], n = *n_keep (0 leaves it as it was).
void gdn_conv_commit(float* history, const float* qkv, int channels, const int32_t* n_keep, void* stream);
/// n_layers consecutive GDN layers' commits in one launch (layer l's state at state + l * state_stride, its conv
/// history right after the S x h_v x S state; h / qkv at l * h_stride, gate / beta at l * gb_stride): the conv
/// history and the state replayed over the first *n_keep rows (at most max_n: 8, or else kDecMaxT), from the rows
/// dec_gdn_rec stored.
struct GdnCommitLayers {
    float* state = nullptr;
    const float *h = nullptr, *qkv = nullptr, *gate = nullptr, *beta = nullptr;
    size_t state_stride = 0, h_stride = 0, gb_stride = 0;
    int C = 0, h_k = 0, h_v = 0, n_layers = 0;
    const int32_t* n_keep = nullptr;
};
/// From `blocks` persistent blocks (grid-stride); each loads its next item while the current one runs.
void gdn_commit_layers(const GdnCommitLayers& a, void* stream, int blocks = 32, int max_n = 8);

}  // namespace basalt::kernels
