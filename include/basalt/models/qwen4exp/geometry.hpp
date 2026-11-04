// basalt/models/qwen4exp/geometry.hpp - Flash-Next's expert widths and its blob's layout.
#pragma once

#include "basalt/kernels/cpu/expert.hpp"

namespace basalt::kernels::cpu {

inline constexpr int K = 10;        // experts the router picks per token
inline constexpr int H = 2560;      // n_embd
inline constexpr int FF = 640;      // expert intermediate width
inline constexpr int NE = 512;      // routed experts per layer
inline constexpr int ROW_GU = H * 2 / 8;         // 640 B of codes per gate/up row
inline constexpr int ROW_D = FF * 2 / 8;         // 160 B of codes per down row
inline constexpr int SC_GU = H / QK;             // 40 weight blocks per gate/up row
inline constexpr int SC_D = FF / QK;             // 10 weight blocks per down row
inline constexpr size_t BLOB = 3ull * (H * FF * BB / QK);   // 1,382,400
inline constexpr size_t O_GU_CODES = 0;
inline constexpr size_t O_D_CODES = 2ull * FF * ROW_GU;
inline constexpr size_t O_GU_SCALES = O_D_CODES + 1ull * H * ROW_D;
inline constexpr size_t O_D_SCALES = O_GU_SCALES + 2ull * FF * SC_GU * 2;

}  // namespace basalt::kernels::cpu
