// src/kernels/cpu/router_q4_neon.cpp - see router_q4.hpp. NEON with the dot product instructions.
#include "basalt/kernels/cpu/router_q4.hpp"

#include <arm_neon.h>
#include <cstddef>
#include <cstring>

namespace basalt::kernels::cpu {

void router_q4_logits(const uint8_t* rows, const float* scales, int n_embd, int n_expert, int e0, int e1,
                      const uint8_t* xq, int nt, float* logits, const std::atomic<bool>& stop) {
    const int NB = n_embd / 32;
    const uint8x16_t lo4 = vdupq_n_u8(0x0f);
    const int8x16_t off = vdupq_n_s8(8);
    for (int e = e0; e < e1; ++e) {
        if (stop.load(std::memory_order_relaxed)) return;
        float32x4_t acc[kRouterQ4MaxT];
        for (int t = 0; t < nt; ++t) acc[t] = vdupq_n_f32(0.0f);
        const uint8_t* row = rows + (size_t) e * (n_embd / 2);
        for (int b = 0; b < NB; ++b) {
            const uint8x16_t q = vld1q_u8(row + b * 16);
            const int8x16_t wl = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q, lo4)), off);
            const int8x16_t wh = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q, 4)), off);
            const float sw = scales[(size_t) e * NB + b];
            for (int t = 0; t < nt; ++t) {
                const uint8_t* blk = xq + ((size_t) t * NB + b) * 36;   // block_q8_1: half d, half s, 32 x int8
                uint16_t dh;
                std::memcpy(&dh, blk, 2);
                const float dx = vgetq_lane_f32(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(dh))), 0);
                const int8x16_t x0 = vld1q_s8((const int8_t*) (blk + 4));
                const int8x16_t x1 = vld1q_s8((const int8_t*) (blk + 20));
                const int32x4_t p = vdotq_s32(vdotq_s32(vdupq_n_s32(0), wl, x0), wh, x1);
                acc[t] = vfmaq_f32(acc[t], vcvtq_f32_s32(p), vdupq_n_f32(sw * dx));
            }
        }
        for (int t = 0; t < nt; ++t) logits[(size_t) t * n_expert + e] = vaddvq_f32(acc[t]);
    }
}

}  // namespace basalt::kernels::cpu
