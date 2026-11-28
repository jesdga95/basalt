// src/models/deepseek41/engram_rows_test.cpp - Engram table rows as floats against the formats' definitions: MXFP4 (OCP
// microscaling: E2M1 codes times 2^(e - 127) per 32 values; low nibble value j, high nibble j + 16) for every scale
// exponent and code, Q5_K and Q6_K super-blocks bit-exact against ports of ggml's dequantize_row_q5_K and
// dequantize_row_q6_K, MXFP8 blocks against their definition, and IQ4_NL rows equal to the block dequantizer they
// always took. No GPU.
#include "models/deepseek41/engram_rows.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace basalt::models::deepseek41;

namespace {

uint16_t read_u16(const uint8_t* p) { return (uint16_t) (p[0] | (uint16_t) (p[1] << 8)); }

/// ggml's get_scale_min_k4 (ggml-quants.c, 3cf03257): the 12 scale bytes hold 16 6-bit scale/min pairs, the second
/// eight of them split between the low nibble of q[j+4] and the top two bits of q[j-4] / q[j].
void ref_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (uint8_t) ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        m = (uint8_t) ((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
    }
}

/// One ggml Q5_K super-block (176 bytes: `fp16 d`, `fp16 dmin`, `scales[12]`, `qh[32]`, `qs[128]`) as 256 floats.
/// Ported from dequantize_row_q5_K (ggml-quants.c, 3cf03257), block for block, as the reference for the rows.
void ref_dequantize_row_q5_K(const uint8_t* x, float* y) {
    const uint8_t* ql = x + 48;        // x->qs
    const uint8_t* qh = x + 16;        // x->qh
    const float d = basalt::fp16_to_fp32(read_u16(x));
    const float min = basalt::fp16_to_fp32(read_u16(x + 2));
    int is = 0;
    uint8_t sc, m;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        ref_scale_min_k4(is + 0, x + 4, sc, m);
        const float d1 = d * (float) sc, m1 = min * (float) m;
        ref_scale_min_k4(is + 1, x + 4, sc, m);
        const float d2 = d * (float) sc, m2 = min * (float) m;
        for (int l = 0; l < 32; ++l) *y++ = d1 * (float) ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
        for (int l = 0; l < 32; ++l) *y++ = d2 * (float) ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
        ql += 32;
        is += 2;
        u1 = (uint8_t) (u1 << 2);
        u2 = (uint8_t) (u2 << 2);
    }
}

/// One ggml Q6_K super-block (210 bytes: `ql[128]`, `qh[64]`, `int8 scales[16]`, `fp16 d`) as 256 floats.
/// Ported from dequantize_row_q6_K (ggml-quants.c, 3cf03257), as the reference for the rows.
void ref_dequantize_row_q6_K(const uint8_t* x, float* y) {
    const uint8_t* ql = x;
    const uint8_t* qh = x + 128;
    const int8_t* sc = (const int8_t*) (x + 192);
    const float d = basalt::fp16_to_fp32(read_u16(x + 208));
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[n + l] = d * sc[is + 0] * q1;
            y[n + l + 32] = d * sc[is + 2] * q2;
            y[n + l + 64] = d * sc[is + 4] * q3;
            y[n + l + 96] = d * sc[is + 6] * q4;
        }
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

}  // namespace

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
    // Q5_K: rows of 176-byte super-blocks (one per model row, two to cover the stride) bit-exact against the reference.
    // Both fp16 scales stay finite (exponent < 31) so the compare is on values, not on NaN payloads.
    int bad3 = 0;
    for (int trial = 0; trial < 512; ++trial) {
        const int values = trial % 2 ? 512 : 256;
        const int blocks = values / 256;
        std::vector<uint8_t> row((size_t) blocks * 176);
        for (auto& c : row) c = (uint8_t) rng();
        for (int b = 0; b < blocks; ++b)
            for (int w = 0; w < 2; ++w) {
                const uint16_t h = (uint16_t) (rng() & 0x7bff);
                std::memcpy(row.data() + (size_t) b * 176 + w * 2, &h, 2);
            }
        std::vector<float> want(values), got(values);
        for (int b = 0; b < blocks; ++b)
            ref_dequantize_row_q5_K(row.data() + (size_t) b * 176, want.data() + (size_t) b * 256);
        engram_row_f32(kEngramQ5K, values, row.data(), got.data());
        bad3 += std::memcmp(got.data(), want.data(), (size_t) values * sizeof(float)) != 0;
    }
    std::printf("Q5_K rows, 512 random super-blocks: %s\n", bad3 ? "FAIL" : "bit-exact with ggml");
    // Q6_K: the same against dequantize_row_q6_K; the fp16 scale stays finite
    int bad4 = 0;
    for (int trial = 0; trial < 512; ++trial) {
        const int values = trial % 2 ? 512 : 256;
        const int blocks = values / 256;
        std::vector<uint8_t> row((size_t) blocks * 210);
        for (auto& c : row) c = (uint8_t) rng();
        for (int b = 0; b < blocks; ++b) {
            const uint16_t h = (uint16_t) (rng() & 0x7bff);
            std::memcpy(row.data() + (size_t) b * 210 + 208, &h, 2);
        }
        std::vector<float> want(values), got(values);
        for (int b = 0; b < blocks; ++b)
            ref_dequantize_row_q6_K(row.data() + (size_t) b * 210, want.data() + (size_t) b * 256);
        engram_row_f32(kEngramQ6K, values, row.data(), got.data());
        bad4 += std::memcmp(got.data(), want.data(), (size_t) values * sizeof(float)) != 0;
    }
    std::printf("Q6_K rows, 512 random super-blocks: %s\n", bad4 ? "FAIL" : "bit-exact with ggml");
    // MXFP8: 512 random rows against the definition (an E4M3 code times 2^(scale - 127) per 32 values). The scales keep
    // every product finite; the reserved NaN code reads as 0.
    int bad5 = 0;
    for (int trial = 0; trial < 512; ++trial) {
        const int values = trial % 2 ? 512 : 256;
        const int blocks = values / 32;
        std::vector<uint8_t> row((size_t) blocks * 33), scales((size_t) blocks);
        for (auto& c : row) c = (uint8_t) rng();
        for (int b = 0; b < blocks; ++b) row[(size_t) b * 33] = scales[(size_t) b] = (uint8_t) (100 + rng() % 21);
        std::vector<float> got(values), want(values);
        engram_row_f32(kEngramMxfp8, values, row.data(), got.data());
        for (int b = 0; b < blocks; ++b)
            for (int j = 0; j < 32; ++j) {
                const uint8_t c = row[(size_t) b * 33 + 1 + j];
                const int e = c >> 3 & 15, m = c & 7;
                float v;
                if (e == 15 && m == 7) v = 0.f;
                else {
                    v = e == 0 ? std::ldexp((float) m, -9) : std::ldexp(1.f + m * 0.125f, e - 7);
                    if (c & 0x80) v = -v;
                }
                want[(size_t) b * 32 + j] = v * std::ldexp(1.f, scales[(size_t) b] - 127);
            }
        bad5 += std::memcmp(got.data(), want.data(), (size_t) values * sizeof(float)) != 0;
    }
    std::printf("MXFP8 rows, 512 random rows: %s\n", bad5 ? "FAIL" : "exact");
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
    // the row geometry the pack's meta must agree with: 256 values a row, whole blocks only, 0 otherwise
    const bool sizes = engram_row_bytes(kEngramIq4Nl, 256) == 144 && engram_row_bytes(kEngramMxfp4, 256) == 136 &&
                       engram_row_bytes(kEngramQ5K, 256) == 176 && engram_row_bytes(kEngramQ5K, 512) == 352 &&
                       engram_row_bytes(kEngramQ6K, 256) == 210 && engram_row_bytes(kEngramQ6K, 128) == 0 &&
                       engram_row_bytes(kEngramMxfp8, 256) == 264 && engram_row_bytes(kEngramMxfp8, 512) == 528 &&
                       engram_row_bytes(kEngramMxfp8, 33) == 0 && engram_row_bytes(kEngramQ5K, 128) == 0 &&
                       engram_row_bytes(kEngramIq4Nl, 33) == 0 && engram_row_bytes(kEngramQ5K, 0) == 0 &&
                       engram_row_bytes(8, 256) == 0;
    std::printf("row sizes: %s\n", sizes ? "ok" : "FAIL");
    return bad || bad2 || bad3 || bad4 || bad5 || !sizes ? 1 : 0;
}
