// include/basalt/prefill/moe_mxfp8.hpp - the prompt path's native MXFP8 experts on the Blackwell FP8 block-scaled
// tensor cores (mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32).
// Weights stay in the pack's block_mxfp8 blocks (128 values = 4 UE8M0 scales + 128 E4M3 codes = 132 bytes, read in
// place from each expert's blob); activations are rounded per 32-value block to E4M3 codes + UE8M0 scales.
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::prefill::mxfp8 {

constexpr int kBlockTokens = 128;

/// Bytes of activation planes for `rows` rows of `cols` values (cols a multiple of 128): E4M3 codes [rows][cols],
/// then UE8M0 block scales [rows][cols / 32] (padded to 16 bytes).
size_t act_bytes(int64_t rows, int64_t cols);

/// Plans 128-row GEMM blocks per expert into blk_off[0..n] (n <= 1024).
void plan_blocks(const int32_t* bounds, int n, int32_t* blk_off, void* stream);
inline int64_t max_blocks(int64_t rows, int n) { return (rows + kBlockTokens - 1) / kBlockTokens + n; }

/// Quantizes `rows` FP32 rows (`ld` floats apart) into MXFP8 activation planes (E4M3 codes + UE8M0 per-32 scales).
void quantize_tokens(const float* x, int64_t rows, int64_t cols, int64_t ld, void* act, void* stream);
/// Quantizes `rows` FP16 rows (`ld` halfs apart) into MXFP8 activation planes.
void quantize_f16(const uint16_t* x, int64_t rows, int64_t cols, int64_t ld, void* act, void* stream);

struct Product {
    const uint8_t* base = nullptr;      ///< expert 0's blob; expert j is at base + j * stride
    size_t stride = 0;
    const unsigned long long* blobs = nullptr;   ///< in place of base + j * stride: expert j's blob at blobs[j] (device [n])
    size_t w_off = 0;
    int64_t w_rows = 0, w_cols = 0;     ///< w_rows a multiple of 128, w_cols a multiple of 128
    int n = 0;                          ///< number of experts (<= 1024)
    const void* act = nullptr;          ///< MXFP8 activation planes (act_bytes(act_rows, w_cols))
    int64_t act_rows = 0;               ///< rows in `act` (tokens for GATHER, pairs otherwise)
    const int32_t* bounds = nullptr;    ///< [n + 1] expert row bounds
    const int32_t* blk_off = nullptr;   ///< [n + 1] from plan_blocks (128-row GEMM blocks per expert)
    int64_t blocks = 0;                 ///< max GEMM blocks (max_blocks(pairs, n))
    const int32_t* ids = nullptr;       ///< GATHER: row i reads act row ids[i]; null: row i reads act row i
    const int32_t* dst_ids = nullptr;   ///< optional output scatter: row i writes dst[dst_ids[i] * ld_dst]
    float* dst = nullptr;               ///< FP32 output (when dst_f16 is null)
    uint16_t* dst_f16 = nullptr;        ///< optional FP16 output (replaces dst)
    int64_t ld_dst = 0;
    bool swiglu = false;                ///< fused gate/up + SwiGLU (w_rows = 2 * F, outputs F values per row)
    bool gelu = false;                  ///< with swiglu: the GELU (tanh form) gate, h = gelu(g) u (not with the softcaps)
    float gate_softcap = 0.0f;          ///< > 0: cg * tanh(g / cg) * sigmoid(g) * cu * tanh(u / cu)
    float up_softcap = 0.0f;
};

void run(const Product& p, void* stream);

}  // namespace basalt::prefill::mxfp8
