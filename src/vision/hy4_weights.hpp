// src/vision/hy4_weights.hpp - the projector file's weights of the SigLIP2-derived tower (llama.cpp's hy_v4_flash GGUF) in the form
// both encoders compute with: bf16 matrices, the merge convolution as one product.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>

#include "basalt/vision/mmproj.hpp"

namespace basalt::vision::hy4 {

inline uint16_t bf16_of(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x40);   // a NaN stays one
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1)) >> 16);                  // round to nearest even
}

/// The merge convolution `mm.0.weight` (F32 [4E out][E in][2][2]) as the bf16 product [4E][4E] over a group's four patch
/// rows in merge order: column p * E + c takes the weight of channel c at kernel position p = ky * 2 + kx.
inline bool merge_matrix(const Mmproj& m, int E, uint16_t* dst, std::string& err) {
    const TensorRef* w = m.need("mm.0.weight", 0, (int64_t) 4 * E * 4 * E, err);
    if (!w) return false;
    const float* s = (const float*) w->data;
    for (int64_t o = 0; o < 4 * E; ++o)
        for (int c = 0; c < E; ++c)
            for (int p = 0; p < 4; ++p) dst[o * 4 * E + (int64_t) p * E + c] = bf16_of(s[(o * E + c) * 4 + p]);
    return true;
}

/// The 1 x 1 convolution `mm.2.weight` (F32 [proj out][4E in]) as the bf16 product [proj][4E].
inline bool out_matrix(const Mmproj& m, int E, int proj, uint16_t* dst, std::string& err) {
    const TensorRef* w = m.need("mm.2.weight", 0, (int64_t) 4 * E * proj, err);
    if (!w) return false;
    const float* s = (const float*) w->data;
    for (int64_t i = 0; i < (int64_t) 4 * E * proj; ++i) dst[i] = bf16_of(s[i]);
    return true;
}

}  // namespace basalt::vision::hy4
