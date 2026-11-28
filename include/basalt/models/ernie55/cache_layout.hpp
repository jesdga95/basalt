// include/basalt/models/ernie55/cache_layout.hpp - single source of truth for ERNIE-5.5 kernel dimensions and KV cache row layouts.
#pragma once

#include <cstdint>

namespace basalt::models::ernie55::layout {

// Fixed dimensions assumed by the ERNIE-5.5 CUDA kernels (validated at pack load in pack.cpp).
constexpr int kHcaHeadDim = 512;
constexpr int kHcaVDim = 512;
constexpr int kHcaKvHeads = 1;
constexpr int kHcaKvLora = 512;
constexpr int kHcaRopeDim = 64;
constexpr int kHcaNopeDim = kHcaHeadDim - kHcaRopeDim;   // 448
constexpr int kHcaFp8Group = 64;

constexpr int kMlaKvLora = 512;
constexpr int kMlaQkNope = 192;
constexpr int kMlaQkRope = 64;
constexpr int kMlaVDim = 256;
constexpr int kDsaFp8Group = 128;

constexpr int kIdxDim = 128;
constexpr int kIdxMaxHeads = 64;
constexpr int kIdxFp4Group = 32;

constexpr int kNumExperts = 512;
constexpr int kHcMult = 4;

// FP8 HCA row: 448 E4M3 nope codes (448 B) + 7 F32 scales (28 B, per 64 values) + 64 BF16 rope values (128 B) = 604 B -> 608 B aligned.
constexpr int kHcaFp8RawBytes = kHcaNopeDim + (kHcaNopeDim / kHcaFp8Group) * 4 + kHcaRopeDim * 2;
constexpr int kHcaFp8RowBytes = (kHcaFp8RawBytes + 7) & ~7;
static_assert(kHcaFp8RowBytes == 608, "unexpected FP8 HCA row stride");

// FP8 MLA-DSA row: 512 E4M3 latent codes (512 B) + 4 F32 scales (16 B, per 128 values) + 64 BF16 rope values (128 B) = 656 B.
constexpr int kDsaFp8RowBytes = kMlaKvLora + (kMlaKvLora / kDsaFp8Group) * 4 + kMlaQkRope * 2;
static_assert(kDsaFp8RowBytes == 656, "unexpected FP8 DSA row stride");

// MXFP4 indexer K row: 128 E2M1 nibbles (64 B) + 4 UE8M0 scales (4 B, per 32 values) = 68 B.
constexpr int kIdxFp4RowBytes = kIdxDim / 2 + kIdxDim / kIdxFp4Group;
static_assert(kIdxFp4RowBytes == 68, "unexpected MXFP4 indexer K row stride");

constexpr uint64_t hca_row_bytes(bool fp8, int head_dim = kHcaHeadDim) {
    return fp8 ? (uint64_t) kHcaFp8RowBytes : (uint64_t) head_dim * 2;
}

constexpr uint64_t dsa_row_bytes(bool fp8, int kv_lora = kMlaKvLora, int rope_dim = kMlaQkRope) {
    return fp8 ? (uint64_t) kDsaFp8RowBytes : (uint64_t) (kv_lora + rope_dim) * 2;
}

constexpr uint64_t idx_row_bytes(bool fp8, int idx_dim = kIdxDim) {
    return fp8 ? (uint64_t) kIdxFp4RowBytes : (uint64_t) idx_dim * 2;
}

}  // namespace basalt::models::ernie55::layout
