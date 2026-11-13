// include/basalt/kernels/logprobs.hpp - a pick's log probability and its row's likeliest tokens, on the GPU.
//
// For row t of `logits` (n_rows x n_vocab) and its pick picks[t], writes kLogprobStride floats at out + t * stride:
//   [0] the pick's log probability, logits[pick] - logsumexp(row)
//   [1] logsumexp(row)
//   [2 + 2j] the j-th likeliest id (int bits; -1 past the row's end), [3 + 2j] its log probability, j < top
// The likeliest come first, ties to the lower id. The row's raw logits: no temperature, penalty or bias. A row is cut
// into kLogprobChunk-wide chunks (one block each: max, exp sum, top candidates), then one block per row merges them in
// chunk order with the sum in double, so a row's values do not depend on the batch width or its place in it.
//
// `logits` and `scratch` (row_logprobs_scratch_floats) are DEVICE pointers; `picks` and `out` device or mapped host
// pointers. Two launches on `stream`; capturable.
#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define LOGPROBS_HD __host__ __device__
#else
#define LOGPROBS_HD
#endif

namespace basalt::kernels {

constexpr int kLogprobMaxTop = 20;
constexpr int kLogprobStride = 2 + 2 * kLogprobMaxTop;
constexpr int kLogprobChunk = 2048;

LOGPROBS_HD inline int row_logprobs_chunks(int n_vocab) { return (n_vocab + kLogprobChunk - 1) / kLogprobChunk; }
inline size_t row_logprobs_scratch_floats(int n_rows, int n_vocab) {
    return (size_t) n_rows * (size_t) row_logprobs_chunks(n_vocab) * (4 + 2 * kLogprobMaxTop);
}

void row_logprobs(const float* logits, int n_rows, int n_vocab, const int32_t* picks, int top, float* out, float* scratch,
                  void* stream);

}  // namespace basalt::kernels
