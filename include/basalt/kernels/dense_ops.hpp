// include/basalt/kernels/dense_ops.hpp - the element-wise steps of a dense layer between its products.
//
// Rows are tokens, row-major, fp32 unless said otherwise. `scale` arguments are device pointers to one fp32 value
// (an NVFP4 matrix's tensor scale) or null for 1.
#pragma once

#include "basalt/kernels/gqa_attn.hpp"

#include <cstdint>

namespace basalt::kernels {

/// out[r][c] = x[r][c] * rsqrt(mean_c x[r]^2 + eps) * w[c]  (w holds an offset norm's 1 + w already); out rows
/// `ld_out` floats apart (0: cols).
void rms_norm_rows(const float* x, const float* w, float* out, int64_t rows, int64_t cols, float eps, void* stream,
                   int64_t ld_out = 0);
/// x[i] += y[i] * scale  (n values).
void add_scaled(float* x, const float* y, const float* scale, int64_t n, void* stream);
/// The BF16 residual stream: x += y * (*scale) (scale null: 1) in fp32, rounded once; and conversions to and from FP32.
void add_scaled(uint16_t* x, const float* y, const float* scale, int64_t n, void* stream);
void to_bf16(const float* x, uint16_t* y, int64_t n, void* stream);
void to_f32(const uint16_t* x, float* y, int64_t n, void* stream);
/// rms_norm_rows of BF16 rows (FP32 math and output).
void rms_norm_rows(const uint16_t* x, const float* w, float* out, int64_t rows, int64_t cols, float eps, void* stream);
/// y[i] *= scale  (n values).
void scale_values(float* y, const float* scale, int64_t n, void* stream);
/// h[i] = silu(g[i] * sg) * (u[i] * su)  (n values; g, u and h may each have their own row stride: rows x width).
void swiglu_scaled(const float* g, const float* u, const float* sg, const float* su, float* h, int64_t rows,
                   int64_t width, int64_t ld_gu, void* stream);
/// y[t][n] = x[t] . w[n]  for BF16 rows w [N][K]: the small BF16 projections of a prompt chunk.
void bf16_rows_dot(const float* x, const uint16_t* w, float* y, int64_t T, int64_t N, int64_t K, void* stream);
/// The GDN a/b projections of a prompt chunk with the gates fused: w_ab BF16 [2 HV][K] (the a rows, then the b rows),
/// gate[t][h] = softplus(a + dt[h]) * ssm_a[h], beta[t][h] = sigmoid(b).  Replaces bf16_rows_dot + gdn_gates (HV = 48, K % 512 == 0).
void gdn_gating_proj(const float* x, const uint16_t* w_ab, const float* dt, const float* ssm_a, float* gate, float* beta,
                     int64_t T, int64_t HV, int64_t K, void* stream);
/// As gdn_gating_proj, of BF16 rows (exact: one BF16 term).
void gdn_gating_proj(const uint16_t* x, const uint16_t* w_ab, const float* dt, const float* ssm_a, float* gate, float* beta,
                     int64_t T, int64_t HV, int64_t K, void* stream);

/// The rotary part of a gated GQA attention layer: `rot` leading dims of each head rotate NeoX-style (pairs i and
/// i + rot/2) with inverse frequency theta^(-2i/rot); pair i takes its position from axis `axis[i]` (0 t, 1 h, 2 w:
/// interleaved M-RoPE; all 0 for plain rope).
struct RopeSpec {
    int rot = 0;
    float theta = 0.0f;
    int8_t axis[64] = {};   // per rotated pair
    float inv_freq[64] = {};   // filled by attn_qk_prepare from theta and rot (the kernels read the table)
};

/// One window's or chunk's query and key heads: q_full [T][n_head][2 hd] (query | gate per head; `gated` false:
/// [T][n_head][hd], queries only), k and v [T][n_kv][hd]. Each q and k head is RMS-normed (q_norm / k_norm, with any
/// offset folded) and rotated at its row's position (pos [T][3]); q goes to q_out [T][n_head][hd], k and v to the fp16
/// cache at cells (cell0 + t) % cap (cap 0: cell0 + t; cache [cell][n_kv][hd]; cell0_dev: cell0 read from the device).
/// hd 128 or 256. `q8`: K and V go to that int8 cache instead (gqa_attn.hpp).
void attn_qk_prepare(const float* q_full, const float* k, const float* v, const float* q_norm, const float* k_norm,
                     float eps, const RopeSpec& rope, const int32_t* pos, int64_t T, int n_head, int n_kv,
                     float* q_out, uint16_t* k_cache, uint16_t* v_cache, int64_t cell0, void* stream, int hd = 256,
                     bool gated = true, int64_t cap = 0, const int32_t* cell0_dev = nullptr, KvInt8 q8 = {});
/// The prompt chunk's form: BF16 q_full (gated, hd 256), k and v; BF16 q_out; the cache at cells cell0 + t.
void attn_qk_prepare_bf16(const uint16_t* q_full, const uint16_t* k, const uint16_t* v, const float* q_norm, const float* k_norm,
                          float eps, const RopeSpec& rope, const int32_t* pos, int64_t T, int n_head, int n_kv, uint16_t* q_out,
                          int64_t cell0, void* stream, KvInt8 q8, uint16_t* k_cache, uint16_t* v_cache);
/// out[t][h][d] = attn[t][h][d] * sigmoid(q_full[t][h][hd + d])  (hd = 256).
void attn_gate(const float* attn, const float* q_full, float* out, int64_t T, int n_head, void* stream);
/// attn_gate's rows rounded to E4M3 for the out projection: codes [T][n_head * 256], scales[T] = amax / 448 per token
/// (prefill::fp8::quantize of attn_gate's output, bitwise).
void attn_gate_e4m3(const float* attn, const float* q_full, int64_t T, int n_head, uint8_t* codes, float* scales, void* stream);
void attn_gate_e4m3(const float* attn, const uint16_t* q_full, int64_t T, int n_head, uint8_t* codes, float* scales, void* stream);
void attn_gate_e4m3(const uint16_t* attn, const uint16_t* q_full, int64_t T, int n_head, uint8_t* codes, float* scales,
                    void* stream);
/// attn_gate's rows rounded to BF16 with per-token max |output| (rowmax) for FP4 quantization.
void attn_gate_bf16(const uint16_t* attn, const uint16_t* q_full, int64_t T, int n_head, uint16_t* out, float* rowmax,
                    void* stream);
void attn_gate_bf16(const float* attn, const uint16_t* q_full, int64_t T, int n_head, uint16_t* out, float* rowmax,
                    void* stream);

/// A block drafter's two-tap grouped dynamic convolution along a block's rows (DFlash 2):
/// out[t][c] = sum over o in {0, 1} of (base[o][c] + dyn[t][o * groups + c / group]) * x[t - o][c], x[-1] = 0.
/// x, out [T][H]; base [2][H]; dyn rows `ld_dyn` floats apart.
void dyn_conv2(const float* x, const float* dyn, int64_t ld_dyn, const float* base, float* out, int64_t T, int64_t H,
               int group, void* stream);
/// A block drafter's candidate path (DFlash 2's selector): for positions 0 .. n-1 in turn, among the row's candidates
/// (rows of row_logprobs output, `top` each: ids and log probabilities) the one maximizing
/// logprob + sum_r pred[prev][r] * h[pos][r] * succ[cand][r], prev the pick before it (`anchor` first, device).
/// pred and succ: bf16 [vocab][rank]; h: fp32 [n][rank]. Writes ids[n] and conf[n] (the pick's softmax over the scores).
void candidate_path(const float* lp_rows, int stride, int top, const float* h, const uint16_t* pred, const uint16_t* succ,
                    int rank, const int32_t* anchor, int n, int32_t* ids, float* conf, void* stream);

}  // namespace basalt::kernels
