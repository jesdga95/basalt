// src/kernels/cuda/q80p.cuh - the bit-plane layout of a Q8_0 head row (include/basalt/kernels/q80_vhead.hpp).
//
// A row of 80 blocks (n_in 2560) keeps its 2720 bytes: the 80 fp16 scales, then per block 16 bytes of bits 2..5 of
// u = q + 128 (word p, quarter p of the block: byte j holds the nibbles of values 8p + j and 8p + 4 + j), 8 bytes of
// bits 6..7 and 8 bytes of bits 0..1 (word h, quarters 2h and 2h + 1: byte j holds the 2-bit fields of values
// 16h + j, 16h + 4 + j, 16h + 8 + j, 16h + 12 + j from its low bits up).
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

namespace basalt::kernels::q80p {

constexpr int kBlocks = 80;                 // blocks of 32 per row
constexpr int kRowBytes = 2720;             // as the Q8_0 row
constexpr int kScales = 0, kNib = 160, kTop = 1440, kLow = 2080;   // byte offsets in the row
constexpr int kScreenBytes = kLow;          // what the screen reads: scales, bits 2..7

// The screen's codes of quarter p: u >> 2 for values 8p..8p+3 (h0) and 8p+4..8p+7 (h1), one per byte (0..63).
__device__ __forceinline__ void hi6(uint32_t nib, uint32_t top, int p, uint32_t& h0, uint32_t& h1) {
    const int s = 4 * (p & 1);
    h0 = (nib & 0x0F0F0F0Fu) | (((top >> s) & 0x03030303u) << 4);
    h1 = ((nib >> 4) & 0x0F0F0F0Fu) | (((top >> (s + 2)) & 0x03030303u) << 4);
}

// The Q8_0 words of quarter p (values 8p..8p+3 and 8p+4..8p+7), as the Q8_0 row holds them.
__device__ __forceinline__ void codes(uint32_t nib, uint32_t top, uint32_t low, int p, int& v0, int& v1) {
    uint32_t h0, h1;
    hi6(nib, top, p, h0, h1);
    const int s = 4 * (p & 1);
    v0 = (int) (((h0 << 2) | ((low >> s) & 0x03030303u)) ^ 0x80808080u);
    v1 = (int) (((h1 << 2) | ((low >> (s + 2)) & 0x03030303u)) ^ 0x80808080u);
}

// The raw words of one (row, block, quarter) and the block's scale.
struct Quarter {
    uint32_t nib, top, low;
    float d;
};
template <bool Stream>
__device__ __forceinline__ Quarter load_quarter(const uint8_t* __restrict__ row, int kb, int p) {
    const uint32_t* n = reinterpret_cast<const uint32_t*>(row + kNib + kb * 16) + p;
    const uint32_t* t = reinterpret_cast<const uint32_t*>(row + kTop + kb * 8) + (p >> 1);
    const uint32_t* l = reinterpret_cast<const uint32_t*>(row + kLow + kb * 8) + (p >> 1);
    const half* d = reinterpret_cast<const half*>(row + kScales) + kb;
    Quarter q;
    if constexpr (Stream) {
        q.nib = __ldcs(n); q.top = __ldcs(t); q.low = __ldcs(l); q.d = __half2float(__ldcs(d));
    } else {
        q.nib = __ldg(n); q.top = __ldg(t); q.low = __ldg(l); q.d = __half2float(__ldg(d));
    }
    return q;
}

// The 32 codes of block b of a plane row, as int8.
__device__ __forceinline__ void block_codes(const uint8_t* __restrict__ row, int b, int8_t (&q)[32], float& d) {
    d = __half2float(reinterpret_cast<const half*>(row + kScales)[b]);
    for (int p = 0; p < 4; ++p) {
        const uint32_t nib = reinterpret_cast<const uint32_t*>(row + kNib + b * 16)[p];
        const uint32_t top = reinterpret_cast<const uint32_t*>(row + kTop + b * 8)[p >> 1];
        const uint32_t low = reinterpret_cast<const uint32_t*>(row + kLow + b * 8)[p >> 1];
        int v0, v1;
        codes(nib, top, low, p, v0, v1);
        for (int j = 0; j < 4; ++j) {
            q[8 * p + j] = (int8_t) (v0 >> (8 * j));
            q[8 * p + 4 + j] = (int8_t) (v1 >> (8 * j));
        }
    }
}

}  // namespace basalt::kernels::q80p
