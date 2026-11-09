// src/models/deepseek41/engram_rows_test.cpp - Engram table rows as floats against the formats' definitions: MXFP4 (OCP
// microscaling: E2M1 codes times 2^(e - 127) per 32 values; low nibble value j, high nibble j + 16) for every scale
// exponent and code, and IQ4_NL rows equal to the block dequantizer they always took. No GPU.
#include "models/deepseek41/engram_rows.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace basalt::models::deepseek41;

int main() {
    int bad = 0;
    const float e2m1[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    std::mt19937 rng(3);
    // MXFP4: a row of 8 blocks per scale exponent 0..254 (255 is NaN), random codes covering all 16
    for (int e = 0; e < 255; ++e) {
        std::vector<uint8_t> row(8 * 17);
        std::vector<float> want(256), got(256);
        for (int b = 0; b < 8; ++b) {
            uint8_t* blk = row.data() + b * 17;
            blk[0] = (uint8_t) e;
            for (int j = 0; j < 16; ++j) {
                const int lo = (b * 16 + j) & 15, hi = (int) (rng() & 15);
                blk[1 + j] = (uint8_t) (lo | hi << 4);
                want[(size_t) b * 32 + j] = e2m1[lo] * std::ldexp(1.f, e - 127);
                want[(size_t) b * 32 + j + 16] = e2m1[hi] * std::ldexp(1.f, e - 127);
            }
        }
        engram_row_f32(kEngramMxfp4, 256, row.data(), got.data());
        for (int i = 0; i < 256; ++i) bad += got[(size_t) i] != want[(size_t) i];
    }
    std::printf("MXFP4 rows, scale exponents 0..254, all codes: %s\n", bad ? "FAIL" : "exact");
    // IQ4_NL: the row is its eight blocks through dequantize_iq4_nl
    int bad2 = 0;
    std::vector<uint8_t> row(8 * 18);
    for (auto& c : row) c = (uint8_t) rng();
    for (int b = 0; b < 8; ++b) {
        const uint16_t h = (uint16_t) (0x2000 + (rng() & 0x0fff));   // a finite fp16 scale
        std::memcpy(row.data() + b * 18, &h, 2);
    }
    std::vector<float> got(256), want(256);
    engram_row_f32(kEngramIq4Nl, 256, row.data(), got.data());
    for (int b = 0; b < 8; ++b) basalt::dequantize_iq4_nl(row.data() + b * 18, want.data() + b * 32);
    bad2 = std::memcmp(got.data(), want.data(), 256 * sizeof(float)) != 0;
    std::printf("IQ4_NL rows: %s\n", bad2 ? "FAIL" : "same as the block dequantizer");
    const bool sizes = engram_block_bytes(kEngramIq4Nl) == 18 && engram_block_bytes(kEngramMxfp4) == 17 && engram_block_bytes(8) == 0;
    std::printf("block sizes: %s\n", sizes ? "ok" : "FAIL");
    return bad || bad2 || !sizes ? 1 : 0;
}
