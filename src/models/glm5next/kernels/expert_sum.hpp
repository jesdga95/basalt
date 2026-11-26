// src/models/glm5next/kernels/expert_sum.hpp - the last launch of one glm5next expert block: the weighted sum of a
// token's expert rows, each row read from wherever the tier keeps it.
//
// The tier hands an expert back in one of three places: the main card's VRAM homes (the grouped kernels wrote its row),
// the pinned RAM places the CPU pool wrote its row into (mapped, so the card can read them), and - for a pick the
// router's tail skip dropped - the tier's zero row. One kernel sums the three without the host copying anything.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace basalt::models::glm5next {

/// Most experts a token routes to, and most experts one block sums: its tokens (GlmModel::kExpertRows, the tier's
/// window) times that. GlmModel refuses a pack whose n_used exceeds the first at the open.
constexpr int kMaxExpertsPerToken = 8;
constexpr int kMaxSumRows = 8 * kMaxExpertsPerToken;

/// A block's expert rows in the order the sum reads them: token t's k-th pick is p[t * K + k]. A pointer is device
/// memory, mapped host memory, or the tier's zero row; never null.
struct RowPtrs {
    const float* p[kMaxSumRows];
};

/// out[t][i] = sum over k < K of w[t * K + k] * p[t * K + k][i], for t < n_tokens (K ascending, the order the host
/// reference sums it in). `n` is the model's width; `w` and `out` are device memory.
void weighted_sum_block(const RowPtrs& rows, const float* w, int n_tokens, int K, int n, float* out,
                        cudaStream_t stream);

/// The grouped experts of one block, in the shape kernels::native_expert_grouped reads: group g's blob sits at the
/// device address ptr[g] and covers the entries [start[g], start[g + 1]); entry e reads token tok[e]'s q8_1 activation
/// and writes row dst[e] of the kernel's output. The counts are read on the device.
struct ExpertPlan {
    unsigned long long ptr[kMaxSumRows];
    int32_t start[kMaxSumRows + 1];
    int32_t ng[1];
    int32_t dst[kMaxSumRows];
    int32_t tok[kMaxSumRows];
};

constexpr int kPlanWords = (int) (sizeof(ExpertPlan) / 4);

/// dst[0..n_words) = src[0..n_words) (n_words <= kPlanWords) for `src` in host memory: the words ride in the kernel's
/// parameters, so the plan reaches the device in stream order instead of queueing a copy-engine job behind the tier's
/// copies, and the host may overwrite `src` as soon as the call returns.
void put_words(const void* src, void* dst, int n_words, cudaStream_t stream);

}  // namespace basalt::models::glm5next
