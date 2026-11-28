#include "basalt/models/deepseek4/indexer.hpp"
#include "cache_rows.cuh"

#include <cuda_fp8.h>

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
constexpr int kMaxTokens = 128;
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

// ---- the index key's rounding and reading (cache_rows.hpp): E2M1 codes with a power-of-two scale per 32 ----------
__device__ __forceinline__ float pow2_ceil_local(float x) {
    const uint32_t b = __float_as_uint(x);
    const int e = (int) ((b >> 23) & 0xffu) - 127 + ((b & 0x7fffffu) != 0u);
    return __uint_as_float((uint32_t) (e + 127) << 23);
}
__device__ __forceinline__ float e2m1_round14(float v) {
    const float a = fabsf(v);
    const float m = a <= 0.25f ? 0.f : a < 0.75f ? .5f : a <= 1.25f ? 1.f : a < 1.75f ? 1.5f : a <= 2.5f ? 2.f
                    : a < 3.5f ? 3.f : a <= 5.f ? 4.f : 6.f;
    return copysignf(m, v) * 16384.0f;   // times 2^-14, the fp16-bit decoding's factor
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
                                  const uint8_t* __restrict__ keys, int n_rows, int mode, int row_bytes,
                                  unsigned long long* __restrict__ sort_keys, float* __restrict__ scores) {
    __shared__ float head_scores[kIndexerHeads];
    const int row = (int) blockIdx.x;
    const int warp = (int) (threadIdx.x >> 5);
    const int lane = (int) (threadIdx.x & 31);
    const float* row_q = q;
    const float* row_w = head_weights;
    const uint8_t* row_key = keys + (size_t) row * row_bytes;
    if (row >= n_rows) return;

    for (int head_base = 0; head_base < kIndexerHeads; head_base += kWarps) {
        const int head = head_base + warp;
        float dot = 0.f;
#pragma unroll
        for (int part = 0; part < 4; ++part) {
            const int d = lane + 32 * part;
            const float qv = row_q[(size_t) head * kIndexerDim + d];
            float amax = fabsf(qv);
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
            const float qscale = pow2_ceil_local(fmaxf(amax, 6.f * 1.1754943508222875e-38f) * (1.f / 6.f));
            const float qr = e2m1_round14(fminf(6.f, fmaxf(-6.f, qv / qscale))) * qscale * (1.0f / 16384.0f);
            float kv;
            if (mode == 1) {
                const uint8_t scale = row_key[kIndexerDim / 2 + part];
                const uint8_t code = row_key[d / 2];
                const uint8_t nib = (d & 1) ? (code >> 4) : (code & 15);
                const float m14 = (float) (nib & 7u);   // magnitude code
                const float mag = m14 == 0.f ? 0.f : m14 == 1.f ? .5f : m14 == 2.f ? 1.f : m14 == 3.f ? 1.5f
                                : m14 == 4.f ? 2.f : m14 == 5.f ? 3.f : m14 == 6.f ? 4.f : 6.f;
                kv = copysignf(mag, (nib & 8u) ? -1.f : 1.f) * __uint_as_float((uint32_t) scale << 23);
            } else if (mode == 0) {
                kv = __half2float(*reinterpret_cast<const __half*>(row_key + 2 * d));
            } else {
                kv = *reinterpret_cast<const float*>(row_key + 4 * d);
            }
            dot = fmaf(qr, kv, dot);
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

void indexer_select(const float* q, const float* head_weights, const void* keys, int n_comp, int start_pos,
                    int n_tokens, int32_t* out_ids, int32_t* n_selected, void* workspace, size_t workspace_bytes,
                    CacheFormat fmt, void* stream) {
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
                head_weights + (size_t) t * kIndexerHeads, static_cast<const uint8_t*>(keys), visible,
                row_mode(fmt), idx_row_bytes(fmt), nullptr, scores);
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
            head_weights + (size_t) t * kIndexerHeads, static_cast<const uint8_t*>(keys), visible,
            row_mode(fmt), idx_row_bytes(fmt), keys_in, nullptr);
        check(cudaGetLastError(), "score launch");
        check(cub::DeviceRadixSort::SortKeysDescending(temp, temp_bytes, keys_in, keys_out, visible, 0, 64,
                                                       cuda_stream),
              "radix sort");
        write_ids_kernel<<<1, kIndexerTopK, 0, cuda_stream>>>(keys_out, visible, count, out, out_n);
        check(cudaGetLastError(), "selection output launch");
    }
}

namespace {

constexpr int kWinScoreBlocks = 1360;
constexpr int kWinSelThreads = 256;
constexpr int kWinSelOneBlock = 12288;
constexpr int kWinSelBlocks = 128;
constexpr int kWinSelGroups = 2052;
constexpr int kWinTopkThreads = 1024;
constexpr int kWinTopkWarps = kWinTopkThreads / 32;

struct WinSelState {
    uint32_t hist[4][256];
    uint32_t prefix[4];
    int32_t need[4];
    uint32_t thr;
    int32_t need_eq;
    int32_t all;
};

__device__ __forceinline__ int win_eff_rows(const int32_t* pos, int n_tokens, int max_comp) {
    return min(max_comp, (pos[n_tokens - 1] + 1) / kRatio);
}

__device__ __forceinline__ int win_sel_rows(int max_comp, const int32_t* pos) {
    return min(max_comp, (pos[0] + 1) / kRatio);
}

__device__ __forceinline__ WinSelState* win_sel_state(WinSelState* st, size_t stride) {
    return reinterpret_cast<WinSelState*>(reinterpret_cast<char*>(st) + stride * blockIdx.y);
}

template <bool kPacked>
__global__ void __launch_bounds__(128) win_idx_score_kernel(const float* __restrict__ iq,
                                                             const float* __restrict__ iw,
                                                             const int32_t* __restrict__ pos,
                                                             const uint8_t* __restrict__ ik,
                                                             int n_comp, int row_bytes,
                                                             float* __restrict__ scores) {
    __shared__ float q[kIndexerHeads * kIndexerDim];
    __shared__ float w[kIndexerHeads];
    const int t = blockIdx.y;
    const int n_eff = win_eff_rows(pos, gridDim.y, n_comp);
    if (blockIdx.x * blockDim.x >= n_eff) return;
    const float* qi = iq + (size_t) t * kIndexerHeads * kIndexerDim;
    for (int i = threadIdx.x; i < kIndexerHeads * kIndexerDim / 4; i += blockDim.x)
        reinterpret_cast<float4*>(q)[i] = reinterpret_cast<const float4*>(qi)[i];
    if (threadIdx.x < kIndexerHeads) w[threadIdx.x] = iw[(size_t) t * kIndexerHeads + threadIdx.x];
    __syncthreads();
    const int nv = (pos[t] + 1) / kRatio;
    for (int base = blockIdx.x * blockDim.x; base < n_eff; base += gridDim.x * blockDim.x) {
        const int j = base + threadIdx.x;
        if (j >= n_eff) continue;
        float* out = scores + (size_t) t * n_comp + j;
        if (j >= nv) {
            *out = -INFINITY;
            continue;
        }
        float acc[kIndexerHeads];
#pragma unroll
        for (int h = 0; h < kIndexerHeads; ++h) acc[h] = 0.f;
        const uint8_t* row = ik + (size_t) j * row_bytes;
        const uint32_t sw = kPacked ? *reinterpret_cast<const uint32_t*>(row + kIndexerDim / 2) : 0u;
        for (int c = 0; c < kIndexerDim / 8; ++c) {
            float k[8];
            if (kPacked) {
                rows::fp4p_values<8>(reinterpret_cast<const uint32_t*>(row)[c], (sw >> (8 * (c / 4))) & 0xffu, k);
            } else {
                const uint4 u = reinterpret_cast<const uint4*>(row)[c];
                const __half2* hp = reinterpret_cast<const __half2*>(&u);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float2 f = __half22float2(hp[i]);
                    k[2 * i] = f.x;
                    k[2 * i + 1] = f.y;
                }
            }
#pragma unroll
            for (int h = 0; h < kIndexerHeads; ++h) {
                const float4 a = *reinterpret_cast<const float4*>(&q[h * kIndexerDim + c * 8]);
                const float4 b = *reinterpret_cast<const float4*>(&q[h * kIndexerDim + c * 8 + 4]);
                acc[h] += k[0] * a.x + k[1] * a.y + k[2] * a.z + k[3] * a.w + k[4] * b.x + k[5] * b.y + k[6] * b.z +
                          k[7] * b.w;
            }
        }
        float s = 0.f;
#pragma unroll
        for (int h = 0; h < kIndexerHeads; ++h) s += fmaxf(acc[h], 0.f) * w[h];
        *out = s;
    }
}

__device__ __forceinline__ int win_block_scan(int flag, int& total, int* wsum) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const uint32_t b = __ballot_sync(0xffffffffu, flag);
    const int rank = __popc(b & ((1u << lane) - 1u));
    __syncthreads();
    if (lane == 0) wsum[warp] = __popc(b);
    __syncthreads();
    int base = 0, tot = 0;
    for (int i = 0; i < nw; ++i) {
        if (i < warp) base += wsum[i];
        tot += wsum[i];
    }
    total = tot;
    return base + rank;
}

__device__ void win_topk_one(const float* __restrict__ s, int n, int K, int32_t* __restrict__ out,
                             int32_t* __restrict__ out_n) {
    __shared__ uint32_t wh[kWinTopkWarps][256];
    __shared__ int ws[kWinTopkWarps];
    __shared__ uint32_t s_prefix;
    __shared__ int s_need, s_all, s_cnt;
    const int warp = threadIdx.x >> 5;
    if (threadIdx.x == 0) {
        s_prefix = 0;
        s_need = K;
        s_all = 0;
    }
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        for (int i = threadIdx.x; i < kWinTopkWarps * 256; i += kWinTopkThreads) (&wh[0][0])[i] = 0;
        __syncthreads();
        if (s_all) break;
        const uint32_t prefix = s_prefix;
#pragma unroll 4
        for (int i = threadIdx.x; i < n; i += kWinTopkThreads) {
            const float v = s[i];
            if (!(v > -INFINITY)) continue;
            const uint32_t key = ordered_float(v);
            if (pass == 0 || (key >> (shift + 8)) == (prefix >> (shift + 8))) atomicAdd(&wh[warp][(key >> shift) & 255], 1u);
        }
        __syncthreads();
        int c = 0;
        if (threadIdx.x < 256)
            for (int w = 0; w < kWinTopkWarps; ++w) c += (int) wh[w][threadIdx.x];
        const int rows = select_pick_bin(c, s_need, prefix, shift, s_prefix, s_need, ws);
        if (pass == 0 && threadIdx.x == 0 && rows <= K) s_all = 1;
    }
    const bool all = s_all != 0;
    const uint32_t thr = s_prefix;
    const int need_eq = all ? 0 : s_need;
    const int run = (n + kWinTopkThreads - 1) / kWinTopkThreads;
    const int r0 = min(n, (int) threadIdx.x * run), r1 = min(n, r0 + run);
    int gt = 0, eq = 0;
#pragma unroll 4
    for (int i = r0; i < r1; ++i) {
        const float v = s[i];
        if (!(v > -INFINITY)) continue;
        const uint32_t key = ordered_float(v);
        if (all || key > thr) ++gt;
        else if (key == thr) ++eq;
    }
    int gt_total, eq_total;
    const int gt_before = select_scan(gt, gt_total, ws);
    const int eq_before = select_scan(eq, eq_total, ws);
    int o = gt_before + min(eq_before, need_eq), r = eq_before;
#pragma unroll 4
    for (int i = r0; i < r1; ++i) {
        const float v = s[i];
        if (!(v > -INFINITY)) continue;
        const uint32_t key = ordered_float(v);
        if (all || key > thr) out[o++] = i;
        else if (key == thr) {
            if (r < need_eq) out[o++] = i;
            ++r;
        }
    }
    if (threadIdx.x == 0) s_cnt = gt_total + min(eq_total, need_eq);
    __syncthreads();
    const int cnt = s_cnt;
    for (int i = cnt + threadIdx.x; i < kIndexerTopK; i += kWinTopkThreads) out[i] = -1;
    if (threadIdx.x == 0) *out_n = cnt;
}

__global__ void __launch_bounds__(kWinTopkThreads) win_select_begin_kernel(const float* __restrict__ s, int n, int K,
                                                                            int32_t* __restrict__ out,
                                                                            int32_t* __restrict__ out_n,
                                                                            const int32_t* __restrict__ pos,
                                                                            WinSelState* st, size_t stride) {
    const int rows = win_sel_rows(n, pos + blockIdx.y);
    if (rows <= kWinSelOneBlock) {
        win_topk_one(s + (size_t) n * blockIdx.y, rows, K, out + (size_t) kIndexerTopK * blockIdx.y, out_n + blockIdx.y);
        return;
    }
    st = win_sel_state(st, stride);
    for (int i = threadIdx.x; i < 4 * 256; i += blockDim.x) st->hist[i / 256][i % 256] = 0;
}

__device__ __forceinline__ int win_sel_pick(WinSelState* st, int done, int K, uint32_t& s_prefix, int& s_need) {
    __shared__ int ws[8];
    return select_pick_bin((int) st->hist[done][threadIdx.x], done ? st->need[done] : K, done ? st->prefix[done] : 0u,
                           24 - 8 * done, s_prefix, s_need, ws);
}

__global__ void __launch_bounds__(kWinSelThreads) win_select_pass_kernel(const float* __restrict__ s, int n, int K,
                                                                          const int32_t* __restrict__ pos, int pass,
                                                                          WinSelState* st, size_t stride) {
    __shared__ uint32_t wh[kWinSelThreads / 32][256];
    __shared__ uint32_t s_prefix;
    __shared__ int s_need;
    s += (size_t) n * blockIdx.y;
    pos += blockIdx.y;
    st = win_sel_state(st, stride);
    n = win_sel_rows(n, pos);
    if (n <= kWinSelOneBlock || blockIdx.x * kWinSelThreads >= n) return;
    if (threadIdx.x == 0) { s_prefix = 0; s_need = K; }
    if (pass > 0) {
        const int rows = win_sel_pick(st, pass - 1, K, s_prefix, s_need);
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            st->prefix[pass] = s_prefix;
            st->need[pass] = s_need;
            if (pass == 1) st->all = rows <= K;
        }
    }
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int w = 0; w < kWinSelThreads / 32; ++w) wh[w][threadIdx.x] = 0;
    __syncthreads();
    const uint32_t prefix = s_prefix;
    const int shift = 24 - 8 * pass;
    for (int base = blockIdx.x * kWinSelThreads; base < n; base += gridDim.x * kWinSelThreads) {
        const int j = base + threadIdx.x;
        if (j >= n || !(s[j] > -INFINITY)) continue;
        const uint32_t key = ordered_float(s[j]);
        if (pass == 0 || (key >> (shift + 8)) == (prefix >> (shift + 8))) atomicAdd(&wh[warp][(key >> shift) & 255], 1u);
    }
    __syncthreads();
    uint32_t h = 0;
#pragma unroll
    for (int w = 0; w < kWinSelThreads / 32; ++w) h += wh[w][threadIdx.x];
    if (h) atomicAdd(&st->hist[pass][threadIdx.x], h);
}

__global__ void __launch_bounds__(kWinSelThreads) win_select_mask_kernel(const float* __restrict__ s, int n, int K,
                                                                          const int32_t* __restrict__ pos,
                                                                          WinSelState* st, size_t stride) {
    __shared__ uint32_t w_gt[8], w_eq[8];
    __shared__ uint32_t s_thr;
    __shared__ int s_need;
    const size_t groups = (size_t) (n + kWinSelThreads - 1) / kWinSelThreads + 1;
    s += (size_t) n * blockIdx.y;
    pos += blockIdx.y;
    st = win_sel_state(st, stride);
    uint32_t* gt = reinterpret_cast<uint32_t*>(st + 1);
    uint32_t* eq = gt + groups * 8;
    uint32_t* gc = eq + groups * 8;
    n = win_sel_rows(n, pos);
    if (n <= kWinSelOneBlock || blockIdx.x * kWinSelThreads >= n) return;
    win_sel_pick(st, 3, K, s_thr, s_need);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        st->thr = s_thr;
        st->need_eq = s_need;
    }
    const uint32_t thr = s_thr;
    const bool all = st->all != 0;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int base = blockIdx.x * kWinSelThreads; base < n; base += gridDim.x * kWinSelThreads) {
        const int j = base + threadIdx.x;
        const bool valid = j < n && s[j] > -INFINITY;
        const uint32_t key = valid ? ordered_float(s[j]) : 0u;
        const uint32_t bg = __ballot_sync(0xffffffffu, valid && (all || key > thr));
        const uint32_t be = __ballot_sync(0xffffffffu, valid && !all && key == thr);
        const int g = base / kWinSelThreads;
        if (lane == 0) {
            gt[(size_t) g * 8 + warp] = bg;
            eq[(size_t) g * 8 + warp] = be;
            w_gt[warp] = __popc(bg);
            w_eq[warp] = __popc(be);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            uint32_t a = 0, b = 0;
            for (int w = 0; w < 8; ++w) { a += w_gt[w]; b += w_eq[w]; }
            gc[(size_t) g * 2] = a;
            gc[(size_t) g * 2 + 1] = b;
        }
        __syncthreads();
    }
}

__global__ void __launch_bounds__(kWinTopkThreads) win_select_final_kernel(int n, int32_t* __restrict__ out,
                                                                            int32_t* __restrict__ out_n,
                                                                            const int32_t* __restrict__ pos,
                                                                            WinSelState* st, size_t stride) {
    __shared__ int list[kWinSelGroups], cgt[kWinSelGroups], ceq[kWinSelGroups], at[kWinSelGroups];
    __shared__ int wsum[32];
    __shared__ int s_cnt, s_total, s_left;
    const size_t groups = (size_t) (n + kWinSelThreads - 1) / kWinSelThreads + 1;
    out += (size_t) kIndexerTopK * blockIdx.y;
    out_n += blockIdx.y;
    pos += blockIdx.y;
    st = win_sel_state(st, stride);
    const uint32_t* gt = reinterpret_cast<const uint32_t*>(st + 1);
    const uint32_t* eq = gt + groups * 8;
    const uint32_t* gc = eq + groups * 8;
    n = win_sel_rows(n, pos);
    if (n <= kWinSelOneBlock) return;
    const int ng = (n + kWinSelThreads - 1) / kWinSelThreads;
    if (threadIdx.x == 0) { s_cnt = 0; s_total = 0; s_left = st->all ? 0 : st->need_eq; }
    __syncthreads();
    for (int i0 = 0; i0 < ng; i0 += blockDim.x) {
        const int g = i0 + threadIdx.x;
        const int a = g < ng ? (int) gc[(size_t) g * 2] : 0, b = g < ng ? (int) gc[(size_t) g * 2 + 1] : 0;
        int tot_eq;
        const int eq_before = win_block_scan(b != 0 ? 1 : 0, tot_eq, wsum);
        const bool ok = a != 0 || (b != 0 && eq_before < s_left);
        int tot;
        const int r = win_block_scan(ok ? 1 : 0, tot, wsum);
        if (ok) {
            list[s_cnt + r] = g;
            cgt[s_cnt + r] = a;
            ceq[s_cnt + r] = b;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            s_cnt += tot;
            s_left = max(0, s_left - tot_eq);
        }
        __syncthreads();
    }
    {
        const int need0 = st->all ? 0 : st->need_eq, cnt = s_cnt;
        const int run = (cnt + kWinTopkThreads - 1) / kWinTopkThreads;
        const int i0 = min(cnt, (int) threadIdx.x * run), i1 = min(cnt, i0 + run);
        int g = 0, e = 0;
        for (int i = i0; i < i1; ++i) {
            g += cgt[i];
            e += ceq[i];
        }
        int g_tot, e_tot;
        int G = select_scan(g, g_tot, wsum), E = select_scan(e, e_tot, wsum);
        for (int i = i0; i < i1; ++i) {
            const int take = min(ceq[i], max(0, need0 - E));
            at[i] = G + min(E, need0);
            G += cgt[i];
            E += ceq[i];
            ceq[i] = take;
        }
        if (threadIdx.x == 0) s_total = g_tot + min(e_tot, need0);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < s_cnt; i += blockDim.x) {
        const int g = list[i];
        int k = at[i], left = ceq[i];
        for (int w = 0; w < 8; ++w) {
            uint32_t m = gt[(size_t) g * 8 + w];
            uint32_t be = eq[(size_t) g * 8 + w];
            while (be != 0 && left > 0) {
                m |= be & (~be + 1u);
                be &= be - 1u;
                --left;
            }
            while (m != 0) {
                out[k++] = g * kWinSelThreads + w * 32 + __ffs(m) - 1;
                m &= m - 1u;
            }
        }
    }
    for (int i = threadIdx.x; i < kIndexerTopK; i += blockDim.x)
        if (i >= s_total) out[i] = -1;
    if (threadIdx.x == 0) out_n[0] = s_total;
}

size_t win_sel_stride(int max_comp) {
    const size_t groups = (size_t) (max_comp + kWinSelThreads - 1) / kWinSelThreads + 1;
    const size_t bytes = sizeof(WinSelState) + groups * (16 + 2) * sizeof(uint32_t);
    return (bytes + 255) / 256 * 256;
}

}  // namespace

size_t indexer_select_window_workspace_bytes(int max_comp) {
    if (max_comp < 1 || max_comp > kMaxRows) throw std::invalid_argument("deepseek4 indexer window: invalid capacity");
    return align256((size_t) kIndexerWindow * max_comp * sizeof(float)) +
           (size_t) kIndexerWindow * win_sel_stride(max_comp) + 256;
}

void indexer_select_window(const float* q, const float* head_weights, const void* keys, int max_comp,
                           const int32_t* pos, int n_tokens, int32_t* out_ids, int32_t* n_selected,
                           void* workspace, size_t workspace_bytes, CacheFormat fmt, void* stream) {
    if (!q || !head_weights || !keys || !pos || !out_ids || !n_selected || !workspace ||
        max_comp < 1 || max_comp > kMaxRows || n_tokens < 1 || n_tokens > kIndexerWindow ||
        workspace_bytes < indexer_select_window_workspace_bytes(max_comp) || fmt == CacheFormat::F32)
        throw std::invalid_argument("deepseek4 indexer_select_window: invalid arguments");
    auto* base = static_cast<unsigned char*>(workspace);
    auto* scores = reinterpret_cast<float*>(base);
    auto* st = reinterpret_cast<WinSelState*>(base + align256((size_t) kIndexerWindow * max_comp * sizeof(float)));
    const size_t stride = win_sel_stride(max_comp);
    const cudaStream_t s = (cudaStream_t) stream;
    dim3 sgrid(std::min((max_comp + 127) / 128, kWinScoreBlocks), n_tokens);
    (fmt == CacheFormat::Fp8 ? win_idx_score_kernel<true> : win_idx_score_kernel<false>)<<<sgrid, 128, 0, s>>>(
        q, head_weights, pos, static_cast<const uint8_t*>(keys), max_comp, idx_row_bytes(fmt), scores);
    check(cudaGetLastError(), "window score launch");
    const int grid = std::min((max_comp + kWinSelThreads - 1) / kWinSelThreads, kWinSelBlocks);
    win_select_begin_kernel<<<dim3(1, n_tokens), kWinTopkThreads, 0, s>>>(
        scores, max_comp, kIndexerTopK, out_ids, n_selected, pos, st, stride);
    for (int pass = 0; pass < 4; ++pass)
        win_select_pass_kernel<<<dim3(grid, n_tokens), kWinSelThreads, 0, s>>>(
            scores, max_comp, kIndexerTopK, pos, pass, st, stride);
    win_select_mask_kernel<<<dim3(grid, n_tokens), kWinSelThreads, 0, s>>>(
        scores, max_comp, kIndexerTopK, pos, st, stride);
    win_select_final_kernel<<<dim3(1, n_tokens), kWinTopkThreads, 0, s>>>(
        max_comp, out_ids, n_selected, pos, st, stride);
    check(cudaGetLastError(), "window selection launch");
}

}  // namespace basalt::models::deepseek4
