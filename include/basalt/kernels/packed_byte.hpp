// include/basalt/kernels/packed_byte.hpp - byte reads out of register values by shift.
#pragma once

#include <cstdint>

/// Byte `index` of a packed word by shift: bytes of a register value are taken this way, not through a byte pointer.
__device__ __forceinline__ uint32_t packed_byte(uint32_t word, int index) { return (word >> (8 * index)) & 0xffu; }
__device__ __forceinline__ uint32_t packed_byte(int2 words, int index) {
    return packed_byte(index < 4 ? (uint32_t) words.x : (uint32_t) words.y, index & 3);
}
__device__ __forceinline__ uint32_t packed_byte(uint2 words, int index) {
    return packed_byte(index < 4 ? words.x : words.y, index & 3);
}
/// Byte `index` of a packed word as a signed 8-bit value.
__device__ __forceinline__ int packed_s8(uint32_t word, int index) { return (int) (int8_t) packed_byte(word, index); }
/// The Q4_K / Q5_K scales (bytes 0..1) and mins (bytes 2..3) the two aux halves hold, as one word.
__device__ __forceinline__ uint32_t pack_scale_min(const uint16_t aux[2]) { return (uint32_t) aux[0] | ((uint32_t) aux[1] << 16); }
