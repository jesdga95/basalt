// include/basalt/kernels/nvfp4_planes.hpp - NVFP4 matrices in planes, the pack's tensor type 45 (NVFP4_PLANES): the layout
// the dense stack's prompt and decode products read (TMA tiles in prompt chunks, 16-byte loads in decode windows). It is
// NInfer / Infernix's block_scale_k16_m128x4 layout minus its FP32 word: the matrix's FP32 scale stays a separate
// multiplier (`<name>.scale`, Linear::scale).
//
// For a matrix of `rows` x `K` (rows a multiple of 128, K a multiple of 64), one buffer:
//   codes   [rows][K / 2] bytes at offset 0: byte b of row r holds E2M1 value 2b in its low nibble and 2b + 1 in its high
//           nibble;
//   scales  rows * K / 16 UE4M3 bytes at offset rows * K / 2 (one per 16 values), byte of (row r, group g = k / 16) at
//           scale_index(r, g, K): 512-byte tiles of 128 rows x 4 groups, tiles row-block major, within a tile
//           (r % 32) * 16 + ((r % 128) / 32) * 4 + g % 4.
// The buffer is as long as ggml's NVFP4 blocks (36 bytes per 64 values), so a matrix converts in place through a copy.
// ggml's block: four UE4M3 scales (sub-blocks of 16), then qs[32] where qs[8s + j] holds value 16s + j in its low nibble
// and 16s + j + 8 in its high nibble; the scale bytes move unchanged.
#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define BASALT_NVFP4P_HD __host__ __device__
#else
#define BASALT_NVFP4P_HD
#endif

namespace basalt::kernels::nvfp4p {

BASALT_NVFP4P_HD inline constexpr bool fits(int64_t rows, int64_t K) {
    return rows > 0 && K > 0 && rows % 128 == 0 && K % 64 == 0;
}
BASALT_NVFP4P_HD inline constexpr std::size_t scale_offset(int64_t rows, int64_t K) {
    return (std::size_t) rows * (std::size_t) K / 2;
}
BASALT_NVFP4P_HD inline constexpr std::size_t bytes(int64_t rows, int64_t K) {
    return (std::size_t) rows * (std::size_t) K / 64 * 36;
}

BASALT_NVFP4P_HD inline int64_t scale_index(int64_t row, int64_t group, int64_t K) {
    return ((row / 128) * (K / 64) + group / 4) * 512 + (row & 31) * 16 + ((row & 127) >> 5) * 4 + (group & 3);
}

/// The planes of a matrix of ggml NVFP4 blocks (`blocks`, rows x K) into `planes` (distinct buffers, `bytes` each).
void from_blocks(const void* blocks, int64_t rows, int64_t K, void* planes, void* stream);
/// The ggml NVFP4 blocks of a plane matrix, for products that read blocks (distinct buffers, `bytes` each).
void to_blocks(const void* planes, int64_t rows, int64_t K, void* blocks, void* stream);

}  // namespace basalt::kernels::nvfp4p
