// src/models/deepseek41/engram_rows.hpp - an Engram table row as floats: IQ4_NL (18 bytes per 32 values) or MXFP4 (17).
#pragma once

#include "basalt/artifact/dequant.hpp"

#include <cstddef>
#include <cstdint>

namespace basalt::models::deepseek41 {

inline constexpr int kEngramIq4Nl = 20, kEngramMxfp4 = 39;   // ggml type ids

/// Bytes of 32 values of an Engram row type; 0 for a type the rows cannot have.
inline size_t engram_block_bytes(int type) { return type == kEngramIq4Nl ? 18 : type == kEngramMxfp4 ? 17 : 0; }

/// `n` values (a multiple of 32) of a row of `type` as floats.
inline void engram_row_f32(int type, int n, const uint8_t* row, float* out) {
    if (type == kEngramMxfp4)
        for (int b = 0; b < n / 32; ++b) dequantize_mxfp4(row + (size_t) b * 17, out + (size_t) b * 32);
    else
        for (int b = 0; b < n / 32; ++b) dequantize_iq4_nl(row + (size_t) b * 18, out + (size_t) b * 32);
}

}  // namespace basalt::models::deepseek41
