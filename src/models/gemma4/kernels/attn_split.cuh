// src/models/gemma4/kernels/attn_split.cuh - how the decode attention splits a row's keys (attn.cu, attn_mma.cu): in
// chunks by the row's own key count, so a row's bits do not depend on the other rows of its call; each chunk's
// (max, sum, unnormalized output) per query head goes to the scratch [row][head][chunk][hd + 2], merged by attn.cu.
#pragma once

#include "basalt/models/gemma4/kernels.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace basalt::models::gemma4 {

constexpr int kMaxChunks = 64;     // chunks a row's keys split into at most
constexpr int kMinChunk = 16;      // keys a chunk holds at least

// A row's split: its first key and key count, its chunk length and count, from its own position only.
struct Split {
    int64_t lo, n;
    int64_t len;
    int chunks;
};
__host__ __device__ __forceinline__ int64_t chunk_len(int64_t n) {
    const int64_t len = (n + kMaxChunks - 1) / kMaxChunks;
    return len < kMinChunk ? kMinChunk : (len + 31) / 32 * 32;
}
// The most chunks any row of at most n keys splits into (a launch's grid; chunk_len is not monotonic in n).
inline int64_t chunk_bound(int64_t n) {
    const int64_t b = (n + kMinChunk - 1) / kMinChunk;
    return b < kMaxChunks ? b : kMaxChunks;
}
__device__ __forceinline__ Split split_of(int pos, int window) {
    Split s;
    s.lo = window > 0 && pos - window + 1 > 0 ? pos - window + 1 : 0;
    s.n = pos - s.lo + 1;
    s.len = chunk_len(s.n);
    s.chunks = (int) ((s.n + s.len - 1) / s.len);
    return s;
}

// The chunks of every row (`a.max_keys` bounds the grid) into a.scratch, on the tensor cores (attn_mma.cu).
void attn_chunks(const Attn& a, int rows, cudaStream_t st);

}  // namespace basalt::models::gemma4
