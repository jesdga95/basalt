// src/models/deepseek41/engram_rows.hpp - an Engram table row as floats. IQ4_NL (18 bytes per 32 values), MXFP4 (17 per
// 32) and MXFP8 (33 per 32: one E8M0 scale, then 32 E4M3 codes) block 32 values; Q5_K (176 bytes) and Q6_K (210 bytes)
// are one super-block of 256 values, so a row of 256 values is one block.
#pragma once

#include "basalt/artifact/dequant.hpp"

#include <cstddef>
#include <cstdint>

namespace basalt::models::deepseek41 {

// ggml type ids. 43 is llama.cpp's F8_E4M3, one code per value with no scale (Hy4-flash's table); Unsloth's standalone
// MXFP8 table stores it as blocks of a scale and 32 codes.
inline constexpr int kEngramQ5K = 13, kEngramQ6K = 14, kEngramIq4Nl = 20, kEngramMxfp4 = 39, kEngramMxfp8 = 43;

/// Values and bytes of one block of an Engram row type; {0, 0} for a type the rows cannot have.
struct EngramBlock {
    int values = 0;
    size_t bytes = 0;
};

/// Keep in sync with ENGRAM_BLOCKS in tools/convert/arch/deepseek41.py.
inline EngramBlock engram_block(int type) {
    switch (type) {
        case kEngramQ5K: return {256, 176};
        case kEngramQ6K: return {256, 210};
        case kEngramIq4Nl: return {32, 18};
        case kEngramMxfp4: return {32, 17};
        case kEngramMxfp8: return {32, 33};
        default: return {0, 0};
    }
}

/// Bytes of an Engram row of `type` holding `row_values` values; 0 for a type the rows cannot have, and for a row that
/// is not whole blocks of it.
inline size_t engram_row_bytes(int type, int row_values) {
    const EngramBlock b = engram_block(type);
    if (b.values == 0 || row_values <= 0 || row_values % b.values) return 0;
    return (size_t) row_values / (size_t) b.values * b.bytes;
}

/// 32 values of an MXFP8 block: its E8M0 scale, then the E4M3 codes.
inline void dequantize_mxfp8_scaled(const uint8_t* block, float* out) {
    const float sc = ue8m0_to_fp32(block[0]);
    for (int j = 0; j < 32; ++j) out[j] = e4m3_to_fp32(block[1 + j]) * sc;
}

/// `n` values (whole blocks of `type`) of a row of `type` as floats.
inline void engram_row_f32(int type, int n, const uint8_t* row, float* out) {
    const EngramBlock b = engram_block(type);
    if (b.values == 0) return;
    for (int i = 0; i < n / b.values; ++i) {
        const uint8_t* blk = row + (size_t) i * b.bytes;
        float* dst = out + (size_t) i * b.values;
        switch (type) {
            case kEngramQ5K: dequantize_q5_K(blk, dst); break;
            case kEngramQ6K: dequantize_q6_K(blk, dst); break;
            case kEngramIq4Nl: dequantize_iq4_nl(blk, dst); break;
            case kEngramMxfp4: dequantize_mxfp4(blk, dst); break;
            case kEngramMxfp8: dequantize_mxfp8_scaled(blk, dst); break;
        }
    }
}

}  // namespace basalt::models::deepseek41
