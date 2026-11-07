// include/basalt/models/deepseek41/attn.hpp - DeepSeek-V4.1 attention, compute side: the query path, sparse MQA over
// the window and the selected compressed latents with an attention sink, the inverse RoPE, and the grouped low-rank
// output projection.
//
// All activations are fp32 device pointers; a call takes T tokens (1..8) and the result for a token does not depend
// on T or on the token's place in the call. Dense projections run basalt::kernels::native_mmvq on the GGUF blocks.
// Every launch enqueues on `stream`, which must be non-null (native_mmvq's rule).
//
// Layout summary (details in the function comments):
//   x_q8_1  the layer's attention input (after attn_norm) quantised with native_quantize_q8_1(x, xq, 5120, T)
//   qr      [T][1280]      normalised low-rank query (also the indexer's input)
//   q       [T][64][512]   query heads, RoPE applied to dims 448..511
//   attn    [T][64][512]   attention result, inverse RoPE applied to dims 448..511
//   out     [T][5120]      attention output (before the mHC post-mix)
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::models::deepseek41 {

constexpr int kAttnEmbd = 5120;
constexpr int kAttnHeads = 64;
constexpr int kAttnHeadDim = 512;
constexpr int kAttnRopeDim = 64;                       ///< the last dims of a head are rotated (interleaved pairs)
constexpr int kAttnNopeDim = kAttnHeadDim - kAttnRopeDim;
constexpr int kAttnQRank = 1280;
constexpr int kAttnGroups = 8;
constexpr int kAttnGroupIn = kAttnHeads * kAttnHeadDim / kAttnGroups;   ///< 4096
constexpr int kAttnOutRank = 1024;                     ///< per group
constexpr int kAttnWindow = 128;
constexpr int kAttnMaxTokens = 8;
constexpr int kAttnChunk = 64;                         ///< keys per partial-attention block

// ---- RoPE -------------------------------------------------------------------------------------------------------

/// Per-pair angular steps of one RoPE flavour (YaRN folded in): the angle of pair i at position p is p * step[i].
struct RopeTable {
    float step[kAttnRopeDim / 2];
    float mscale;                                      ///< multiplies cos and sin
};

/// ggml's rope_ext parameters -> the table. ext_factor = 0 disables YaRN.
RopeTable make_rope_table(float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast,
                          float beta_slow, int n_ctx_orig);

/// Layers with compress ratio 0 (window only): base 10000, no scaling.
RopeTable rope_table_window();
/// Layers with a compressed cache (also the compressed rows themselves): the compressed base and the YaRN extension
/// (`factor` over `n_ctx_orig` trained positions); the defaults are the published ones.
RopeTable rope_table_compressed(float freq_base = 160000.0f, float factor = 16.0f, float beta_fast = 32.0f,
                                float beta_slow = 1.0f, int n_ctx_orig = 65536);

/// In-place RoPE of the last 64 dims of rows[n_rows][row_stride] (fp32) at positions pos[row / rows_per_pos].
/// `inverse` rotates the other way (the attention output's inverse RoPE).
void rope_rows(float* rows, int n_rows, int row_stride, int rows_per_pos, const int32_t* pos, const RopeTable& rope,
               bool inverse, void* stream);

// ---- query path -------------------------------------------------------------------------------------------------

struct QPathWeights {
    const void* wq_a;                                  ///< GGUF blocks, [1280 rows][5120]
    int wq_a_type;                                     ///< ggml type id (Q6_K = 14, Q8_0 = 8, ...)
    const float* q_norm;                               ///< [1280]
    const void* wq_b;                                  ///< [32768 rows][1280]
    int wq_b_type;
    float rms_eps = 1e-20f;
};

/// Bytes of q8_1 scratch the query path needs for n_tokens.
size_t q_path_scratch_bytes(int n_tokens);

/// qr = rms_norm(wq_a . x) * q_norm; q = rope(wq_b . qr). `xq_scratch` holds q_path_scratch_bytes; `pos` is the
/// device array of the T absolute token positions.
void q_path(const void* x_q8_1, const QPathWeights& w, int n_tokens, const int32_t* pos, const RopeTable& rope,
            void* xq_scratch, float* qr, float* q, void* stream);
/// The two halves of q_path: the normed latent qr, then the heads q from it.
void q_latent(const void* x_q8_1, const QPathWeights& w, int n_tokens, float* qr, void* stream);
void q_heads(const QPathWeights& w, int n_tokens, const int32_t* pos, const RopeTable& rope, void* xq_scratch,
             const float* qr, float* q, void* stream);

// ---- sparse MQA -------------------------------------------------------------------------------------------------

enum class LatentType { F32, F16 };

/// Keys and values are the same 512-wide latent. Per token the keys are
///   window: win_n[t] rows, row j at slot (win_start[t] + j) % win_cap of `win` (ring order is irrelevant to the sum)
///   selected: sel[t][0..n_sel) row ids of `comp`; an id < 0 or >= comp_limit[t] is masked out.
/// score = q . k * scale; the per-head sink joins the softmax denominator and has no value.
struct MqaArgs {
    int n_tokens = 1;
    const float* q = nullptr;                          ///< [T][64][512]
    const void* win = nullptr;                         ///< ring [win_cap][512]
    LatentType win_type = LatentType::F32;
    int win_cap = kAttnWindow;
    const int32_t* win_start = nullptr;                ///< [T]
    const int32_t* win_n = nullptr;                    ///< [T], at most 128
    const void* comp = nullptr;                        ///< compressed rows, row r at comp + r * comp_stride (elements)
    LatentType comp_type = LatentType::F32;
    int64_t comp_stride = kAttnHeadDim;
    const int32_t* sel = nullptr;                      ///< [T][n_sel]; null or n_sel == 0: window only
    int n_sel = 0;                                     ///< at most 512
    const int32_t* comp_limit = nullptr;               ///< [T]: visible rows (ids at or above are masked)
    const float* sinks = nullptr;                      ///< [64]
    float scale = 0.044194174f;                        ///< 512^-0.5
    const int32_t* pos = nullptr;                      ///< [T] absolute positions (inverse RoPE)
    RopeTable rope;                                    ///< the layer's flavour (the window rows' and queries')
    float* out = nullptr;                              ///< [T][64][512], inverse RoPE applied
};

/// Floats of workspace sparse_mqa needs.
size_t sparse_mqa_workspace_floats(int n_tokens, int n_sel);

void sparse_mqa(const MqaArgs& a, float* workspace, void* stream);

// ---- grouped output ---------------------------------------------------------------------------------------------

struct AttnOutWeights {
    const void* wo_a;                                  ///< [8 groups * 1024 rows][4096], group g = rows g*1024..
    int wo_a_type;
    const void* wo_b;                                  ///< [5120 rows][8192]
    int wo_b_type;
};

size_t attn_out_scratch_bytes(int n_tokens);
size_t attn_out_scratch_floats(int n_tokens);

/// oa[t][g*1024 + r] = wo_a[g] . attn[t][g*4096 ..]; out = wo_b . oa. `xq_scratch`: attn_out_scratch_bytes,
/// `oa_scratch`: attn_out_scratch_floats.
void attn_out(const float* attn, const AttnOutWeights& w, int n_tokens, void* xq_scratch, float* oa_scratch,
              float* out, void* stream);

}  // namespace basalt::models::deepseek41
