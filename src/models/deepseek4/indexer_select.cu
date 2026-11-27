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
constexpr int kSingleBlockRows = 12288;
constexpr int kSelectThreads = 1024;
constexpr int kSelectWarps = kSelectThreads / 32;

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

__device__ __forceinline__ int select_scan(int v, int& total, int* ws) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    __syncthreads();
    if (lane == 31) ws[warp] = x;
    __syncthreads();
    if (warp == 0) {
        int w = lane < kSelectWarps ? ws[lane] : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, w, o);
            if (lane >= o) w += y;
        }
        if (lane < kSelectWarps) ws[lane] = w;
    }
    __syncthreads();
    total = ws[kSelectWarps - 1];
    return (warp ? ws[warp - 1] : 0) + x - v;
}

__device__ __forceinline__ int select_pick_bin(int c, int need, uint32_t prefix, int shift,
                                                uint32_t& out_prefix, int& out_need, int* ws) {
    const int b = threadIdx.x, lane = b & 31, warp = b >> 5;
    int above = c;
    if (b < 256) {
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_down_sync(0xffffffffu, above, o);
            if (lane + o < 32) above += y;
        }
        if (lane == 0) ws[warp] = above;
    }
    __syncthreads();
    int total = 0;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        if (b < 256 && w > warp) above += ws[w];
        total += ws[w];
    }
    const int n_hit = __syncthreads_count(b < 256 && above >= need);
    if (b < 256 && (n_hit == 0 ? b == 0 : b == n_hit - 1)) {
        out_prefix = prefix | ((uint32_t) (n_hit == 0 ? 0 : b) << shift);
        out_need = need - (n_hit == 0 ? above : above - c);
    }
    __syncthreads();
    return total;
}

// Selects the top K finite scores in one block, then orders only those K rows by score and row id.
__global__ void __launch_bounds__(kSelectThreads)
select_top_one_kernel(const float* __restrict__ scores, int n, int k, int32_t* __restrict__ out,
                      int32_t* __restrict__ out_n) {
    __shared__ uint32_t hist[kSelectWarps][256];
    __shared__ int ws[kSelectWarps];
    __shared__ uint32_t prefix;
    __shared__ int need, all, count;
    __shared__ unsigned long long sorted[ kIndexerTopK ];
    const int warp = threadIdx.x >> 5;
    if (threadIdx.x == 0) { prefix = 0; need = k; all = 0; }
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        for (int i = threadIdx.x; i < kSelectWarps * 256; i += kSelectThreads)
            (&hist[0][0])[i] = 0;
        __syncthreads();
        if (all) break;
        const uint32_t current = prefix;
#pragma unroll 4
        for (int i = threadIdx.x; i < n; i += kSelectThreads) {
            const float v = scores[i];
            if (!(v > -INFINITY)) continue;
            const uint32_t key = ordered_float(v);
            if (pass == 0 || (key >> (shift + 8)) == (current >> (shift + 8)))
                atomicAdd(&hist[warp][(key >> shift) & 255], 1u);
        }
        __syncthreads();
        int c = 0;
        if (threadIdx.x < 256)
            for (int w = 0; w < kSelectWarps; ++w) c += (int) hist[w][threadIdx.x];
        const int rows = select_pick_bin(c, need, current, shift, prefix, need, ws);
        if (pass == 0 && threadIdx.x == 0 && rows <= k) all = 1;
    }
    const bool take_all = all != 0;
    const uint32_t threshold = prefix;
    const int ties = take_all ? 0 : need;
    const int run = (n + kSelectThreads - 1) / kSelectThreads;
    const int first = min(n, (int) threadIdx.x * run);
    const int last = min(n, first + run);
    int gt = 0, eq = 0;
#pragma unroll 4
    for (int i = first; i < last; ++i) {
        const float v = scores[i];
        if (!(v > -INFINITY)) continue;
        const uint32_t key = ordered_float(v);
        if (take_all || key > threshold) ++gt;
        else if (key == threshold) ++eq;
    }
    int gt_total, eq_total;
    const int gt_before = select_scan(gt, gt_total, ws);
    const int eq_before = select_scan(eq, eq_total, ws);
    int write = gt_before + min(eq_before, ties), equal_rank = eq_before;
#pragma unroll 4
    for (int i = first; i < last; ++i) {
        const float v = scores[i];
        if (!(v > -INFINITY)) continue;
        const uint32_t key = ordered_float(v);
        if (take_all || key > threshold) out[write++] = i;
        else if (key == threshold) {
            if (equal_rank < ties) out[write++] = i;
            ++equal_rank;
        }
    }
    if (threadIdx.x == 0) count = gt_total + min(eq_total, ties);
    __syncthreads();

    for (int i = threadIdx.x; i < kIndexerTopK; i += kSelectThreads) {
        if (i < count) {
            const uint32_t row = (uint32_t) out[i];
            const uint32_t key = ordered_float(scores[row]);
            sorted[i] = ((unsigned long long) key << 32) | (0xffffffffu - row);
        } else sorted[i] = 0;
    }
    __syncthreads();
    for (int size = 2; size <= kIndexerTopK; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = threadIdx.x; i < kIndexerTopK; i += kSelectThreads) {
                const int peer = i ^ stride;
                if (peer > i) {
                    const bool ascending = (i & size) == 0;
                    const unsigned long long a = sorted[i], b = sorted[peer];
                    if ((a > b) != ascending) { sorted[i] = b; sorted[peer] = a; }
                }
            }
            __syncthreads();
        }
    }
    for (int i = threadIdx.x; i < kIndexerTopK; i += kSelectThreads) {
        const uint32_t encoded = (uint32_t) sorted[i];
        out[i] = i < count ? (int32_t) (0xffffffffu - encoded) : -1;
    }
    if (threadIdx.x == 0) *out_n = count;
}

__global__ void score_rows_kernel(const float* __restrict__ q, const float* __restrict__ head_weights,
                                  const float* __restrict__ keys, int n_rows,
                                  unsigned long long* __restrict__ sort_keys, float* __restrict__ scores) {
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
        if (scores) scores[row] = score;
        else {
            const uint64_t composite = ((uint64_t) ordered_float(score) << 32) |
                                       (uint32_t) (0xffffffffu - (uint32_t) row);
            sort_keys[row] = (unsigned long long) composite;
        }
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
    if (max_rows <= kSingleBlockRows) return align256((size_t) max_rows * sizeof(float)) + 256;
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
    const int required_rows = start_pos >= 0 && n_tokens > 0
        ? (int) (((int64_t) start_pos + n_tokens) / kRatio) : 0;
    const size_t minimum_bytes = n_comp > 0
        ? (n_comp <= kSingleBlockRows ? align256((size_t) n_comp * sizeof(float)) + 256
                                      : align256((size_t) n_comp * 2 * sizeof(unsigned long long)) + 256)
        : 256;
    if (!q || !head_weights || !keys || !out_ids || !n_selected || !workspace || n_comp < 1 || n_comp > kMaxRows ||
        start_pos < 0 || n_tokens < 1 || n_tokens > kMaxTokens || start_pos > 1048576 - n_tokens ||
        n_comp < required_rows || workspace_bytes < minimum_bytes)
        throw std::invalid_argument("deepseek4 indexer_select: invalid arguments (rows=" + std::to_string(n_comp) +
            ", required=" + std::to_string(required_rows) + ", pos=" + std::to_string(start_pos) +
            ", tokens=" + std::to_string(n_tokens) + ", workspace=" + std::to_string(workspace_bytes) +
            ", minimum=" + std::to_string(minimum_bytes) + ")");

    auto* base = static_cast<unsigned char*>(workspace);
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
        if (visible <= kSingleBlockRows) {
            auto* scores = reinterpret_cast<float*>(base);
            score_rows_kernel<<<visible, kThreads, 0, cuda_stream>>>(q + (size_t) t * kIndexerHeads * kIndexerDim,
                head_weights + (size_t) t * kIndexerHeads, keys, visible, nullptr, scores);
            check(cudaGetLastError(), "score launch");
            select_top_one_kernel<<<1, kSelectThreads, 0, cuda_stream>>>(scores, visible, count, out, out_n);
            check(cudaGetLastError(), "single-block selection launch");
            continue;
        }
        auto* keys_in = reinterpret_cast<unsigned long long*>(base);
        auto* keys_out = reinterpret_cast<unsigned long long*>(base + (size_t) n_comp * sizeof(unsigned long long));
        const size_t arrays = align256((size_t) n_comp * 2 * sizeof(unsigned long long));
        void* temp = base + arrays;
        size_t temp_bytes = workspace_bytes - arrays - 256;
        score_rows_kernel<<<visible, kThreads, 0, cuda_stream>>>(q + (size_t) t * kIndexerHeads * kIndexerDim,
            head_weights + (size_t) t * kIndexerHeads, keys, visible, keys_in, nullptr);
        check(cudaGetLastError(), "score launch");
        check(cub::DeviceRadixSort::SortKeysDescending(temp, temp_bytes, keys_in, keys_out, visible, 0, 64,
                                                       cuda_stream),
              "radix sort");
        write_ids_kernel<<<1, kIndexerTopK, 0, cuda_stream>>>(keys_out, visible, count, out, out_n);
        check(cudaGetLastError(), "selection output launch");
    }
}

}  // namespace basalt::models::deepseek4
