// include/basalt/models/hy4flash/dfly_kernels.hpp - the device kernels of the DFly2 drafter that no other kernel covers:
// the absorbed MLA's two products over the drafter's own BF16 `kv_b_proj` (rows of a block at once), the gather of the
// target's selected cache rows (fp16, or FP8 decoded to fp16), and the fusion's gated add.
//
// The drafter's matrices are BF16 and read by the shared BF16 products (basalt/kernels/bf16_gemv.hpp); its attention reuses
// mla_prepare and mla_attend (kernels.hpp). Activations are fp32, pointers device memory; `stream` may be null.
#pragma once

#include <cstdint>

namespace basalt::models::hy4flash {

constexpr int kDflyNope = 192, kDflyRank = 512, kDflyV = 256, kDflyRopeDim = 64;
constexpr int kDflyKvbHead = kDflyNope + kDflyV;   ///< rows of one head in kv_b_proj: 192 key (nope) rows, then 256 value rows
constexpr int kDflyRowWidth = kDflyRank + kDflyRopeDim;

/// q_abs[r][h][c] = sum_d q[r * q_stride + h * q_head + d] * kvb[(h * 448 + d) * 512 + c] for d < 192, c < 512: the query's
/// nope part carried to the latent through the key half of `kvb` (BF16 [heads * 448][512]). rows <= 8; q_head is the
/// query's per-head width (nope + rope).
void dfly_absorb_k(const uint16_t* kvb, const float* q, int q_stride, int q_head, int rows, int heads, float* q_abs, void* stream);

/// out[r][h][v] = sum_c olat[r][h][c] * kvb[(h * 448 + 192 + v) * 512 + c] for v < 256: the value half of `kvb`. rows <= 8.
void dfly_expand_v(const uint16_t* kvb, const float* olat, int rows, int heads, float* out, void* stream);

/// out[j] = cache[cells[j]] for j < n (fp16 rows of `width` values, the target's cache form); cells null: out[j] = cache[j].
void dfly_gather_rows(const uint16_t* cache, const int32_t* cells, int n, int width, uint16_t* out, void* stream);
/// The same from FP8 rows (--kv fp8: kFp8RowBytes each, kernels.hpp) to fp16 rows of 576: a code times its power-of-two scale
/// is an fp16 value, so the decoded latent is exact (the BF16 rope values round to fp16).
void dfly_gather_rows_fp8(const uint8_t* cache, const int32_t* cells, int n, uint16_t* out, void* stream);

/// out[i] = base[i] + gate[i] * fused[i] for i < n (the Gated Fusion's residual).
void dfly_gate_add(const float* base, const float* fused, const float* gate, int n, float* out, void* stream);

}  // namespace basalt::models::hy4flash
