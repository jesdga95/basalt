// src/kernels/cpu/tq_states.hpp - the trellis states of one TQ2_T / TQK6 / TQK7 block (32) or TQP5 / TQP6 / TQP7
// block (64) in plain C++ (all three kernels: scalar code beat byte-shuffle versions on AVX2 and AVX-512).  Internal
// to src/kernels/cpu/tq_*.cpp.
//
// TQ2_T (K = 8): state t = qs[(t + 31) % 32] << 8 | qs[t], masked to 15 bits.
// TQK (K = 6, 7): the 16 stream bits from bit (31 - t)*K, LSB first, circular modulo 32*K bits, masked to 15 bits.
// Every window is one unaligned 32-bit little-endian read inside the block's 4*K bytes, except the windows that run
// past the end (t = 0 and 1), which read a 64-bit view of the last four bytes followed by the first four.
// TQP (K = 5, 6, 7): the 16 stream bits from bit (63 - t)*K, LSB first, circular modulo 64*K bits, masked to 15
// bits.  The 8 states of lane l (steps 8l..8l+7, weights 16l..16l+15) lie in the 64-bit window from byte (7 - l)*K,
// step 8l+j at bit (7 - j)*K of it (7*K + 15 <= 64); only lane 0's window runs past the end of the 8*K bytes.
#pragma once

#include <cstdint>
#include <cstring>
#include "basalt/platform/compiler.hpp"

namespace basalt::kernels::cpu::tq_detail {

inline constexpr uint32_t kTqStateMask = 0x7fffu;

inline uint32_t tq_ld32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;   // x86 is little-endian
}

/// State t alone (t a constant after unrolling); `wrap` = tq_wrap<KB>(qs) for the TQK windows past the end.
template <int KB>
inline uint64_t tq_wrap(const uint8_t* qs) {
    if constexpr (KB == 8) return 0;
    else return (uint64_t) tq_ld32(qs + 4 * KB - 4) | ((uint64_t) tq_ld32(qs) << 32);
}
template <int KB>
inline uint32_t tq_state(const uint8_t* qs, uint64_t wrap, int t) {
    if constexpr (KB == 8) {
        return (((uint32_t) qs[(t + 31) & 31] << 8) | qs[t]) & kTqStateMask;
    } else {
        constexpr int NB = 4 * KB;
        const int off = (31 - t) * KB, b = off >> 3;
        const uint32_t v = b + 4 <= NB ? tq_ld32(qs + b) >> (off & 7) : (uint32_t) (wrap >> (off - 8 * (NB - 4)));
        return v & kTqStateMask;
    }
}

template <int KB>
inline void tq_states(const uint8_t* qs, uint32_t* st) {
    if constexpr (KB == 8) {
        st[0] = (((uint32_t) qs[31] << 8) | qs[0]) & kTqStateMask;
        BASALT_UNROLL(31)
        for (int t = 1; t < 32; ++t) st[t] = (((uint32_t) qs[t - 1] << 8) | qs[t]) & kTqStateMask;
    } else {
        constexpr int NB = 4 * KB;
        const uint64_t wrap = (uint64_t) tq_ld32(qs + NB - 4) | ((uint64_t) tq_ld32(qs) << 32);
        BASALT_UNROLL(32)
        for (int t = 0; t < 32; ++t) {
            const int off = (31 - t) * KB, b = off >> 3;
            const uint32_t v = b + 4 <= NB ? tq_ld32(qs + b) >> (off & 7) : (uint32_t) (wrap >> (off - 8 * (NB - 4)));
            st[t] = v & kTqStateMask;
        }
    }
}

inline uint64_t tq_ld64(const uint8_t* p) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    return v;
}

/// TQP: lane l's 64-bit stream window (see the top of this file).
template <int K>
inline uint64_t tqp_window(const uint8_t* qs, int l) {
    static_assert(K >= 5 && K <= 7, "TQP has K = 5, 6, 7");
    if (l == 0) return (tq_ld64(qs + 8 * K - 8) >> (8 * (8 - K))) | (tq_ld64(qs) << (8 * K));
    return tq_ld64(qs + (7 - l) * K);
}
/// TQP: state of step 8l+j from lane l's window.
template <int K>
inline uint32_t tqp_state(uint64_t win, int j) {
    return (uint32_t) (win >> ((7 - j) * K)) & kTqStateMask;
}
/// TQP: the codebook point of a state (its hash's first point; the TQ types' first pair).
inline uint32_t tqp_point(uint32_t s) { return (s * 0x9e3779b1u) >> 21; }

/// fp16 bits -> float, exact (the block scales and the Q8_0 activation scales).
inline float tq_half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu, man = h & 0x3ffu, bits;
    if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (man << 13);
    } else if (exp != 0) {
        bits = sign | ((exp + 112u) << 23) | (man << 13);
    } else if (man == 0) {
        bits = sign;
    } else {   // subnormal: man * 2^-24, exact in float
        const float m = (float) man * (1.0f / 16777216.0f);
        return sign ? -m : m;
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

inline uint16_t tq_ld16(const uint8_t* p) {
    uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}

}  // namespace basalt::kernels::cpu::tq_detail
