#include "basalt/models/deepseek4/indexer.hpp"

#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace basalt::models::deepseek4 {
namespace {

constexpr int kRatio = 4;
constexpr int kThreads = 1024;
constexpr int kWarps = kThreads / 32;
constexpr int kMaxTokens = 8;
constexpr int kMaxRows = 1 << 20;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    return value;
}

__device__ __forceinline__ uint32_t ordered_float(float value) {
    const uint32_t bits = __float_as_uint(value);
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

__global__ void score_rows_kernel(const float* __restrict__ q, const float* __restrict__ head_weights,
                                  const float* __restrict__ keys, int n_rows, unsigned long long* __restrict__ sort_keys) {
    __shared__ float head_scores[kIndexerHeads];
    const int row = (int) blockIdx.x;
    const int warp = (int) (threadIdx.x >> 5);
    const int lane = (int) (threadIdx.x & 31);
    const float* row_q = q;
    const float* row_w = head_weights;
    const float* row_key = keys + (size_t) row * kIndexerDim;
    if (row >= n_rows) return;

    for (int head_base = 0; head_base < kIndexerHeads; head_base += kWarps) {
        const int head = head_base + warp;
        float dot = 0.f;
#pragma unroll
        for (int part = 0; part < 4; ++part) {
            const int d = lane + 32 * part;
            dot = fmaf(row_q[(size_t) head * kIndexerDim + d], row_key[d], dot);
        }
        dot = warp_sum(dot);
        if (lane == 0) head_scores[head] = fmaxf(dot, 0.f) * row_w[head];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float score = 0.f;
#pragma unroll
        for (int head = 0; head < kIndexerHeads; ++head) score += head_scores[head];
        const uint64_t composite = ((uint64_t) ordered_float(score) << 32) | (uint32_t) (0xffffffffu - (uint32_t) row);
        sort_keys[row] = (unsigned long long) composite;
    }
}

__global__ void write_ids_kernel(const unsigned long long* __restrict__ sorted, int visible, int count,
                                 int32_t* __restrict__ out, int32_t* __restrict__ n_selected) {
    const int i = (int) threadIdx.x;
    if (i < kIndexerTopK) {
        if (i < count && i < visible) {
            const uint32_t encoded = (uint32_t) sorted[i];
            out[i] = (int32_t) (0xffffffffu - encoded);
        } else {
            out[i] = -1;
        }
    }
    if (i == 0) n_selected[0] = count;
}

__global__ void empty_ids_kernel(int32_t* __restrict__ out, int32_t* __restrict__ n_selected) {
    const int i = (int) threadIdx.x;
    if (i < kIndexerTopK) out[i] = -1;
    if (i == 0) n_selected[0] = 0;
}

void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) throw std::runtime_error(std::string("deepseek4 indexer ") + operation + ": " +
                                                       cudaGetErrorString(error));
}

size_t align256(size_t bytes) { return (bytes + 255u) & ~size_t(255u); }

}  // namespace

size_t indexer_select_workspace_bytes(int max_rows) {
    if (max_rows < 1 || max_rows > kMaxRows) throw std::invalid_argument("deepseek4 indexer: invalid row capacity");
    size_t temp_bytes = 0;
    check(cub::DeviceRadixSort::SortKeysDescending(nullptr, temp_bytes, (const unsigned long long*) nullptr,
                                                   (unsigned long long*) nullptr, max_rows, 0, 64),
          "workspace query");
    const size_t arrays = (size_t) max_rows * 2 * sizeof(unsigned long long);
    return align256(arrays) + align256(temp_bytes) + 256;
}

void indexer_select(const float* q, const float* head_weights, const float* keys, int n_comp, int start_pos,
                    int n_tokens, int32_t* out_ids, int32_t* n_selected, void* workspace, size_t workspace_bytes,
                    void* stream) {
    if (!q || !head_weights || !keys || !out_ids || !n_selected || !workspace || n_comp < 1 || n_comp > kMaxRows ||
        start_pos < 0 || n_tokens < 1 || n_tokens > kMaxTokens || start_pos > 1048576 - n_tokens ||
        n_comp < (start_pos + n_tokens + kRatio - 1) / kRatio ||
        workspace_bytes < align256((size_t) n_comp * 2 * sizeof(unsigned long long)) + 256)
        throw std::invalid_argument("deepseek4 indexer_select: invalid arguments");

    auto* base = static_cast<unsigned char*>(workspace);
    auto* keys_in = reinterpret_cast<unsigned long long*>(base);
    auto* keys_out = reinterpret_cast<unsigned long long*>(base + (size_t) n_comp * sizeof(unsigned long long));
    const size_t arrays = align256((size_t) n_comp * 2 * sizeof(unsigned long long));
    void* temp = base + arrays;
    size_t temp_bytes = workspace_bytes - arrays - 256;
    const cudaStream_t cuda_stream = (cudaStream_t) stream;

    for (int t = 0; t < n_tokens; ++t) {
        const int visible = (start_pos + t + 1) / kRatio;
        const int count = std::min(kIndexerTopK, visible);
        int32_t* out = out_ids + (size_t) t * kIndexerTopK;
        int32_t* out_n = n_selected + t;
        if (visible == 0) {
            empty_ids_kernel<<<1, kIndexerTopK, 0, cuda_stream>>>(out, out_n);
            check(cudaGetLastError(), "empty selection launch");
            continue;
        }
        score_rows_kernel<<<visible, kThreads, 0, cuda_stream>>>(q + (size_t) t * kIndexerHeads * kIndexerDim,
                                                                head_weights + (size_t) t * kIndexerHeads,
                                                                keys, visible, keys_in);
        check(cudaGetLastError(), "score launch");
        check(cub::DeviceRadixSort::SortKeysDescending(temp, temp_bytes, keys_in, keys_out, visible, 0, 64,
                                                       cuda_stream),
              "radix sort");
        write_ids_kernel<<<1, kIndexerTopK, 0, cuda_stream>>>(keys_out, visible, count, out, out_n);
        check(cudaGetLastError(), "selection output launch");
    }
}

}  // namespace basalt::models::deepseek4
