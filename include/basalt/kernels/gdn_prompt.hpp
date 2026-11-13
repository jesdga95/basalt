// include/basalt/kernels/gdn_prompt.hpp - the gated delta-net (GDN) layer of a prompt chunk, between its projections.
//
// T tokens at a time, row-major with the token as the slow index.  The geometry is the GDN of the Qwen hybrid models:
// state size 128, 16 key heads, 48 value heads, 4 conv taps (gdn_prompt_supported checks a model's).  The recurrence
// runs in 16-token chunks on the tensor cores; see src/kernels/cuda/gdn_prompt.cu.
#pragma once

#include "basalt/kernels/gdn_window.hpp"

#include <cstdint>

namespace basalt::kernels {

/// Whether these kernels take a model's GDN geometry.
bool gdn_prompt_supported(int state, int k_heads, int v_heads, int conv_taps);
/// gate[t,h] = softplus(ab[t,h] + dt[h]) * ssm_a[h];  beta[t,h] = sigmoid(ab[t, 48 + h])  (ab: [T, 96])
void gdn_gates(const float* ab, const float* dt, const float* ssm_a, float* gate, float* beta, int64_t T, void* stream);
/// The 4-tap causal conv + SiLU over the chunk (history [C][3] in, updated to the chunk's last three inputs).
/// h: [T, C].
void gdn_conv(float* history, const float* qkv, const float* conv_w, float* h, int64_t T, void* stream);
/// The bytes of scratch gdn_recurrence needs for T tokens.
uint64_t gdn_work_bytes(int64_t T);
/// The recurrence over the chunk: q/k L2-normalized (eps) and rounded to BF16, the state [key][value head][value]
/// kept in FP32; then the closing norm rmsnorm(o) * gamma * act(z) into y16 (FP16, may be null) and yn (FP32, may be
/// null).  y is FP32 scratch, work gdn_work_bytes(T) bytes.
void gdn_recurrence(float* state, const float* h, const float* gate, const float* beta, const float* z,
                    const float* gamma, float eps, void* work, float* y, uint16_t* y16, int64_t T, void* stream,
                    GdnGate act = GdnGate::Sigmoid, float* yn = nullptr);

}  // namespace basalt::kernels
