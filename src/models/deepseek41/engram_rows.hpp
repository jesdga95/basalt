// src/models/deepseek41/engram_rows.hpp - an Engram table row as floats: IQ4_NL (18 bytes per 32 values), MXFP4 (17 per
// 32) or Q5_K (one 176-byte super-block of 256 values, so a row of 256 values is one block).
#pragma once

#include "basalt/artifact/dequant.hpp"

#include <cstddef>
#include <cstdint>

namespace basalt::models::deepseek41 {

inline constexpr int kEngramIq4Nl = 20, kEngramQ5K = 13, kEngramMxfp4 = 39;   // ggml type ids

/// Bytes of an Engram row of `type` holding `row_values` values; 0 for a type the rows cannot have, and for a row that
/// is not whole blocks of it. IQ4_NL and MXFP4 block 32 values (18 and 17 bytes), Q5_K one block of 256 (176 bytes).
inline size_t engram_row_bytes(int type, int row_values) {
    const bool q5k = type == kEngramQ5K;
    const size_t block = type == kEngramIq4Nl ? 18 : type == kEngramMxfp4 ? 17 : q5k ? 176 : 0;
    const int vals = q5k ? 256 : 32;
    if (block == 0 || row_values <= 0 || row_values % vals) return 0;
    return (size_t) row_values / (size_t) vals * block;
}

/// `n` values (whole blocks of `type`) of a row of `type` as floats.
inline void engram_row_f32(int type, int n, const uint8_t* row, float* out) {
    if (type == kEngramMxfp4)
        for (int b = 0; b < n / 32; ++b) dequantize_mxfp4(row + (size_t) b * 17, out + (size_t) b * 32);
    else if (type == kEngramQ5K)
        for (int b = 0; b < n / 256; ++b) dequantize_q5_K(row + (size_t) b * 176, out + (size_t) b * 256);
    else
        for (int b = 0; b < n / 32; ++b) dequantize_iq4_nl(row + (size_t) b * 18, out + (size_t) b * 32);
}

}  // namespace basalt::models::deepseek41
