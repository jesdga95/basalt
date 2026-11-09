// src/models/deepseek41/dspark_kernels.hpp - device kernels of the DSpark drafter that no shared kernel covers:
// MXFP4 routed experts, the window-latent rotation without a ring write, the Markov head and the confidence dot.
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::models::deepseek41 {

constexpr int kMxfp4BlockElems = 32;
constexpr int kMxfp4BlockBytes = 17;   ///< one E8M0 scale byte, then 16 bytes of packed 4-bit values
constexpr int kMxfp4Pad = 16;          ///< bytes a matrix allocation carries past its end

/// Bytes of one expert matrix row of `n_in` elements (n_in a multiple of 128).
inline size_t mxfp4_row_bytes(int n_in) { return (size_t) (n_in / kMxfp4BlockElems) * kMxfp4BlockBytes; }

/// h[p][r] = swiglu(gate_e[r] . x[t], up_e[r] . x[t]) for pair p = t * K + k, e = ids[p]; the weights are MXFP4
/// matrices [n_expert][n_ff][n_in] (expert e at e * n_ff * row bytes, 4-byte aligned, followed by kMxfp4Pad readable
/// bytes); x is q8_1 [T][n_in] (kernels::native_quantize_q8_1), the dots int8 (dp4a) per 32 values as llama.cpp's
/// MXFP4 kernels take them; the clamp is the SwiGLU limit. n_in is a multiple of 128 and at most 6144.
void dspark_moe_gate_up(const void* gate_w, const void* up_w, int n_in, int n_ff, const void* x_q8_1, const int32_t* ids,
                        int n_tokens, int k, float clamp, float* h, void* stream);

/// d[p][r] = down_e[r] . h[p] with down_e [n_out][n_ff] MXFP4 (expert e at e * n_out * row bytes) and h q8_1
/// [pairs][n_ff].
void dspark_moe_down(const void* down_w, int n_ff, int n_out, const void* h_q8_1, const int32_t* ids, int n_tokens, int k,
                     float* d, void* stream);

/// out[t][i] = sum_k w[t*K + k] * d[(t*K + k)][i] + shared[t][i].
void dspark_moe_combine(const float* d, const float* w, const float* shared, int n_tokens, int k, int n, float* out,
                        void* stream);

/// kv = rope(rms_norm(raw) * norm_w) for rows of 512 with the last 64 dims rotated at pos[t]; written to kv_out
/// [T][512] (fp32, may be null) and, when `ring` is not null, as fp16 to slot pos % ring_slots of ring.
void dspark_kv_rope(const float* raw, const float* norm_w, const int32_t* pos, int n_tokens, float eps,
                    const float* rope_freq, float* kv_out, uint16_t* ring, int ring_slots, void* stream);

/// out[r * out_stride + i] = mean over the four hyper-connection streams of x[r][s][i], for `n_rows` rows.
void dspark_hc_mean(const float* x, int n_rows, int n, float* out, int out_stride, void* stream);

/// out[r] = bf16 row `tok[0]` of w1 [rows][256], as fp32.
void dspark_markov_gather(const uint16_t* w1, const int32_t* tok, float* out, void* stream);

/// col[v] = logits[v] + w2[v] . w1p for v < n_vocab (w2 bf16 [n_vocab][256], w1p fp32 [256]).
void dspark_markov_bias(const uint16_t* w2, const float* w1p, const float* logits, int n_vocab, float* col,
                        void* stream);

/// *out = sigmoid(w[0..n_hidden) . h + w[n_hidden..n_hidden+256) . w1p).
void dspark_conf(const float* w, const float* h, const float* w1p, int n_hidden, float* out, void* stream);

}  // namespace basalt::models::deepseek41
