// include/basalt/kernels/dec_gdn.hpp - the recurrence of the GDN block of a verify window.
//
//   dec_gdn_rec      conv1d + SiLU + q/k L2 norm, the alpha/beta gates, the delta-rule recurrence over the
//                    window from the untouched state, the gated RMS norm, and the q8_1 of its output
//
// Every value is the one the per-stage kernels produce (gdn_conv_l2_multi, gdn_ab_multi, gdn_step_norm_multi,
// native_quantize_q8_1): the same operations on the same values in the same order, so a token's outputs do not
// depend on T or its column.  Needs thread-block clusters.
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::kernels {

constexpr int kDecMaxT = 8;

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
    unsigned long long* stamps = nullptr;   // bench: 10 per-phase timestamps per block
    bool early_trigger = true;   // let the next kernel launch once the constants are loaded
};

/// The recurrence of tokens [0, n_tok) from the untouched state; one cluster of four blocks per value head (of the
/// head map's value heads when a.nk > 0).
void dec_gdn_rec(const DecRecArgs& a, int n_tok, void* stream);

/// dec_gdn_rec over a segmented window (verify_kernels.hpp SegRow): each segment's tokens from its own state (the
/// layer's at rows[].gdn + state_off), one launch for all n_rows window rows; a.state is unused.  max_n: the most rows
/// a segment holds (at most kDecMaxT).  Every output is the segment's own dec_gdn_rec's, bitwise.
struct SegRow;
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

}  // namespace basalt::kernels
