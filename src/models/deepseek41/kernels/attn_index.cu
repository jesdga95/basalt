// DeepSeek-V4.1 attention, index side: window ring, KV compressor, two-level sparse indexer.
// Structure and numerics follow build_attention41 of the llama.cpp DeepSeek-V4.1 support (src/models/deepseek41.cpp).
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "basalt/models/deepseek41/attn_index.hpp"
#include "basalt/models/deepseek41/prefill_index.hpp"
#include "cache_rows.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <stdexcept>

namespace basalt::models::deepseek41 {

// ---- host: rotary table ---------------------------------------------------------------------------------------

void rope_freq_table(const RopeParams& p, float* out) {
    const int n_dims = kRopeDim;
    const float theta_scale = powf(p.freq_base, -2.0f / n_dims);
    float low = 0.f, high = (float) (n_dims - 1);
    if (p.ext_factor != 0.f) {
        const auto corr = [&](float n_rot) {
            return n_dims * logf(p.n_ctx_orig / (n_rot * 2 * 3.14159265358979323846f)) / (2 * logf(p.freq_base));
        };
        low = std::max(0.f, floorf(corr(p.beta_fast)));
        high = std::min((float) (n_dims - 1), ceilf(corr(p.beta_slow)));
    }
    for (int i = 0; i < kRopePairs; ++i) {
        const float extrap = powf(theta_scale, (float) i);
        float mix = 0.f;
        if (p.ext_factor != 0.f) {
            const float y = ((float) i - low) / std::max(0.001f, high - low);
            mix = (1.0f - std::min(1.0f, std::max(0.0f, y))) * p.ext_factor;
        }
        out[i] = extrap * (p.freq_scale * (1 - mix) + mix);
    }
}

namespace {

constexpr int kNoRow = -1;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

__device__ __forceinline__ float bf16_to_f32(uint16_t b) { return __uint_as_float((uint32_t) b << 16); }

// Sum over a block whose size is a multiple of 32 (at most 1024); every thread gets the total. `scratch` >= 32 floats.
__device__ float block_sum(float v, float* scratch) {
    v = warp_sum(v);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float t = 0.f;
    for (int i = 0; i < nw; ++i) t += scratch[i];
    return t;
}

// Interleaved rotation of the pair (x0, x1) at pair index `pair`.
__device__ __forceinline__ void rotate_pair(float& x0, float& x1, float pos, float freq) {
    float s, c;
    sincosf(pos * freq, &s, &c);
    const float a = x0 * c - x1 * s;
    const float b = x0 * s + x1 * c;
    x0 = a;
    x1 = b;
}

// ---- gemv -----------------------------------------------------------------------------------------------------

// One warp per output row; the weights are read once for all tokens.
__global__ void gemv_bf16_kernel(const uint16_t* __restrict__ w, const float* __restrict__ x, int n_tokens, int in,
                                 int out, float* __restrict__ y) {
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= out) return;
    float acc[kMaxT] = {};
    const uint4* wr = reinterpret_cast<const uint4*>(w + (size_t) row * in);
    for (int c = lane; c < in / 8; c += 32) {
        const uint4 q = wr[c];
        const uint32_t u[4] = {q.x, q.y, q.z, q.w};
        float wf[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            wf[2 * i] = __uint_as_float(u[i] << 16);
            wf[2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u);
        }
#pragma unroll
        for (int t = 0; t < kMaxT; ++t) {
            if (t < n_tokens) {
                const float4 a = *reinterpret_cast<const float4*>(x + (size_t) t * in + c * 8);
                const float4 b = *reinterpret_cast<const float4*>(x + (size_t) t * in + c * 8 + 4);
                acc[t] += wf[0] * a.x + wf[1] * a.y + wf[2] * a.z + wf[3] * a.w + wf[4] * b.x + wf[5] * b.y +
                          wf[6] * b.z + wf[7] * b.w;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < kMaxT; ++t) {
        const float s = warp_sum(acc[t]);
        if (lane == 0 && t < n_tokens) y[(size_t) t * out + row] = s;
    }
}

inline cudaStream_t as_stream(void* s) { return static_cast<cudaStream_t>(s); }

void check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// ---- cache rows -----------------------------------------------------------------------------------------------

// Value c of row r in the cache format: packed (a full warp per group, cache_rows.cuh) or fp16 as it is.
__device__ __forceinline__ void put_win(float o, int c, uint8_t* rows0, size_t r, bool packed) {
    if (packed) rows::put_fp8(o, c, kLatent, rows0 + r * kWinRowBytes);
    else reinterpret_cast<__half*>(rows0)[r * kLatent + c] = __float2half(o);
}
__device__ __forceinline__ void put_comp(float o, int c, uint8_t* rows0, size_t r, bool packed) {
    if (packed) rows::put_fp4(o, c, kLatent, rows0 + r * kCompRowBytes);
    else reinterpret_cast<__half*>(rows0)[r * kLatent + c] = __float2half(o);
}
__device__ __forceinline__ void put_idx(float o, int c, uint8_t* rows0, size_t r, bool packed) {
    if (!packed) {
        reinterpret_cast<__half*>(rows0)[r * kIdxDim + c] = __float2half(o);
        return;
    }
    uint8_t* dst = rows0 + r * kIdxRowBytes;
    rows::put_fp4p(o, c, kIdxDim, dst);
    if (c < kIdxRowBytes - kIdxDim / 2 - kIdxDim / 32) dst[kIdxDim / 2 + kIdxDim / 32 + c] = 0;   // the padding
}

// ---- window ring ----------------------------------------------------------------------------------------------

__global__ void kv_window_kernel(const float* __restrict__ raw, const float* __restrict__ norm_w,
                                 const int32_t* __restrict__ pos, float eps, const float* __restrict__ freq,
                                 float* __restrict__ kv_out, uint8_t* __restrict__ ring, bool packed) {
    __shared__ float sh[kLatent];
    __shared__ float red[32];
    const int t = blockIdx.x, c = threadIdx.x;
    const float v = raw[(size_t) t * kLatent + c];
    const float ss = block_sum(v * v, red);
    const float r = 1.0f / sqrtf(ss / kLatent + eps);
    const float y = v * r * norm_w[c];
    sh[c] = y;
    __syncthreads();
    float o = y;
    if (c >= kLatent - kRopeDim) {
        const int p = (c - (kLatent - kRopeDim)) >> 1;
        float x0 = sh[kLatent - kRopeDim + 2 * p], x1 = sh[kLatent - kRopeDim + 2 * p + 1];
        rotate_pair(x0, x1, (float) pos[t], freq[p]);
        o = (c & 1) ? x1 : x0;
    }
    if (kv_out) kv_out[(size_t) t * kLatent + c] = o;
    put_win(o, c, ring, (size_t) (pos[t] % kWindowSlots), packed);
}

// ---- compressor -----------------------------------------------------------------------------------------------

// One block, 512 threads (a channel each); tokens run in order so a group closed inside the call sees the rows of
// the call and of earlier calls alike.
__global__ void compress_kernel(const float* __restrict__ ckv, const float* __restrict__ csc,
                                const int32_t* __restrict__ pos, int n_tokens, int ratio, float eps,
                                const float* __restrict__ norm_w, const float* __restrict__ freq,
                                float* __restrict__ state_kv, float* __restrict__ state_score,
                                uint8_t* __restrict__ comp_kv, float* __restrict__ comp_pre,
                                int32_t* __restrict__ row_of, bool packed) {
    __shared__ float sh[kLatent];
    __shared__ float red[32];
    const int c = threadIdx.x;
    float kv[2], sc[2];
    kv[0] = state_kv[c];
    sc[0] = state_score[c];
    kv[1] = ratio > 1 ? state_kv[kLatent + c] : 0.f;
    sc[1] = ratio > 1 ? state_score[kLatent + c] : 0.f;
    for (int t = 0; t < n_tokens; ++t) {
        const int p = pos[t];
        const int slot = p % ratio;
        const float nkv = ckv[(size_t) t * kLatent + c];
        const float nsc = csc ? csc[(size_t) t * kLatent + c] : 0.f;
        if (slot == 0) { kv[0] = nkv; sc[0] = nsc; } else { kv[1] = nkv; sc[1] = nsc; }
        const bool close = (p + 1) % ratio == 0;
        if (c == 0) row_of[t] = close ? p / ratio : kNoRow;
        if (!close) continue;
        float m = sc[0];
        if (ratio > 1) m = fmaxf(m, sc[1]);
        float den = 0.f, num = 0.f;
        for (int j = 0; j < ratio; ++j) {
            const float e = expf(sc[j] - m);
            den += e;
            num += kv[j] * e;
        }
        const float comp = num / den;
        const float ss = block_sum(comp * comp, red);
        const float y = comp * (1.0f / sqrtf(ss / kLatent + eps)) * norm_w[c];
        comp_pre[(size_t) t * kLatent + c] = y;
        sh[c] = y;
        __syncthreads();
        float o = y;
        if (c >= kLatent - kRopeDim) {
            const int pr = (c - (kLatent - kRopeDim)) >> 1;
            float x0 = sh[kLatent - kRopeDim + 2 * pr], x1 = sh[kLatent - kRopeDim + 2 * pr + 1];
            rotate_pair(x0, x1, (float) (p + 1 - ratio), freq[pr]);
            o = (c & 1) ? x1 : x0;
        }
        put_comp(o, c, comp_kv, (size_t) (p / ratio), packed);
        __syncthreads();
    }
    state_kv[c] = kv[0];
    state_score[c] = sc[0];
    if (ratio > 1) {
        state_kv[kLatent + c] = kv[1];
        state_score[kLatent + c] = sc[1];
    }
}

// Index key of a closed group: rms_norm(w_idx_k . comp) * norm_w, rotated at the group's first position.
__global__ void index_key_kernel(const float* __restrict__ comp_pre, const int32_t* __restrict__ row_of,
                                 const int32_t* __restrict__ pos, int ratio, float eps,
                                 const uint16_t* __restrict__ w, const float* __restrict__ norm_w,
                                 const float* __restrict__ freq, uint8_t* __restrict__ comp_ik, bool packed) {
    const int t = blockIdx.x;
    const int row = row_of[t];
    if (row < 0) return;
    __shared__ float x[kLatent];
    __shared__ float y[kIdxDim];
    __shared__ float red[32];
    for (int i = threadIdx.x; i < kLatent; i += blockDim.x) x[i] = comp_pre[(size_t) t * kLatent + i];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    for (int o = warp; o < kIdxDim; o += nw) {
        float acc = 0.f;
        for (int i = lane; i < kLatent; i += 32) acc += bf16_to_f32(w[(size_t) o * kLatent + i]) * x[i];
        acc = warp_sum(acc);
        if (lane == 0) y[o] = acc;
    }
    __syncthreads();
    const float v = threadIdx.x < kIdxDim ? y[threadIdx.x] : 0.f;
    const float ss = block_sum(v * v, red);
    const float r = 1.0f / sqrtf(ss / kIdxDim + eps);
    if (threadIdx.x < kIdxDim) y[threadIdx.x] = v * r * norm_w[threadIdx.x];
    __syncthreads();
    if (threadIdx.x < kIdxDim) {
        const int c = threadIdx.x;
        float o = y[c];
        if (c >= kIdxDim - kRopeDim) {
            const int p = (c - (kIdxDim - kRopeDim)) >> 1;
            float x0 = y[kIdxDim - kRopeDim + 2 * p], x1 = y[kIdxDim - kRopeDim + 2 * p + 1];
            rotate_pair(x0, x1, (float) (pos[t] + 1 - ratio), freq[p]);
            o = (c & 1) ? x1 : x0;
        }
        put_idx(o, c, comp_ik, (size_t) row, packed);
    }
}

// ---- indexer --------------------------------------------------------------------------------------------------

// Rotates the last kRopeDim dims of every query head, then (`round`) rounds the head as an index key, in place.
// Block (token, head), a thread per dim.
__global__ void idx_rope_round_kernel(float* __restrict__ iq, const int32_t* __restrict__ pos, const float* __restrict__ freq,
                                      bool round) {
    const int t = blockIdx.x, c = threadIdx.x;
    float* v = iq + ((size_t) t * kIdxHeads + blockIdx.y) * kIdxDim;
    float o = v[c];
    const float other = __shfl_xor_sync(0xffffffffu, o, 1);
    if (c >= kIdxDim - kRopeDim) {
        float x0 = (c & 1) ? other : o, x1 = (c & 1) ? o : other;
        rotate_pair(x0, x1, (float) pos[t], freq[(c - (kIdxDim - kRopeDim)) >> 1]);
        o = (c & 1) ? x1 : x0;
    }
    v[c] = round ? rows::round_fp4p(o) : o;
}

// Rows any token of the call can see: positions ascend, so the last token sees the most. `cap` bounds the buffers.
__device__ __forceinline__ int eff_rows(const int32_t* pos, int n_tokens, int ratio, int cap) {
    return min(cap, (pos[n_tokens - 1] + 1) / ratio);
}

// score[t][j] over j < n_comp (the buffer capacity; only the rows a token of the call can see are written);
// -inf where the row is not visible to the token or its block is not a candidate.
template <bool kPacked>
__global__ void idx_score_kernel(const float* __restrict__ iq, const float* __restrict__ iw,
                                 const int32_t* __restrict__ pos, const uint8_t* __restrict__ ik, int n_comp,
                                 int ratio, int block_size, const uint8_t* __restrict__ blk_ok, int blk_ld,
                                 float w_scale, float* __restrict__ scores) {
    __shared__ __align__(16) float q[kIdxHeads * kIdxDim];
    __shared__ float w[kIdxHeads];
    const int t = blockIdx.y;
    const int n_eff = eff_rows(pos, gridDim.y, ratio, n_comp);
    if (blockIdx.x * blockDim.x >= n_eff) return;
    for (int i = threadIdx.x; i < kIdxHeads * kIdxDim; i += blockDim.x) q[i] = iq[(size_t) t * kIdxHeads * kIdxDim + i];
    if (threadIdx.x < kIdxHeads) w[threadIdx.x] = iw[t * kIdxHeads + threadIdx.x];
    __syncthreads();
    // the grid is capped, so a block walks the row groups gridDim.x apart
    for (int base = blockIdx.x * blockDim.x; base < n_eff; base += gridDim.x * blockDim.x) {
        const int j = base + threadIdx.x;
        if (j >= n_eff) continue;
        const int nv = (pos[t] + 1) / ratio;
        float* out = scores + (size_t) t * n_comp + j;
        if (j >= nv || (blk_ok && !blk_ok[(size_t) t * blk_ld + j / block_size])) {
            *out = -INFINITY;
            continue;
        }
        float acc[kIdxHeads];
    #pragma unroll
        for (int h = 0; h < kIdxHeads; ++h) acc[h] = 0.f;
        const uint8_t* row = ik + (size_t) j * (kPacked ? kIdxRowBytes : kIdxDim * 2);
        const uint32_t sw = kPacked ? *reinterpret_cast<const uint32_t*>(row + kIdxDim / 2) : 0u;   // the groups' scales
        for (int c = 0; c < kIdxDim / 8; ++c) {
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
            for (int h = 0; h < kIdxHeads; ++h) {
                const float4 a = *reinterpret_cast<const float4*>(&q[h * kIdxDim + c * 8]);
                const float4 b = *reinterpret_cast<const float4*>(&q[h * kIdxDim + c * 8 + 4]);
                acc[h] += k[0] * a.x + k[1] * a.y + k[2] * a.z + k[3] * a.w + k[4] * b.x + k[5] * b.y + k[6] * b.z +
                          k[7] * b.w;
            }
        }
        float s = 0.f;
    #pragma unroll
        for (int h = 0; h < kIdxHeads; ++h) s += fmaxf(acc[h], 0.f) * (w[h] * w_scale);
        *out = s;
    }
}

// Best score per block, plus 1e30 on the block holding the newest visible row.
__global__ void block_max_kernel(const float* __restrict__ scores, const int32_t* __restrict__ pos, int n_comp,
                                 int ratio, int block_size, float* __restrict__ bscore) {
    const int t = blockIdx.y;
    const int n_blk = (n_comp + block_size - 1) / block_size;
    const int n_eff = eff_rows(pos, gridDim.y, ratio, n_comp);
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= (n_eff + block_size - 1) / block_size) return;
    float m = -INFINITY;
    for (int j = b * block_size; j < min(n_eff, (b + 1) * block_size); ++j) m = fmaxf(m, scores[(size_t) t * n_comp + j]);
    const int nv = (pos[t] + 1) / ratio;
    if (nv > 0 && (nv - 1) / block_size == b) m += 1e30f;
    bscore[(size_t) t * n_blk + b] = m;
}

__global__ void mark_blocks_kernel(const int32_t* __restrict__ cand, const int32_t* __restrict__ cand_n, int n_blk,
                                   const int32_t* __restrict__ pos, int n_comp, int ratio, int block_size,
                                   uint8_t* __restrict__ ok) {
    const int t = blockIdx.x;
    const int n_eff = (eff_rows(pos, gridDim.x, ratio, n_comp) + block_size - 1) / block_size;
    for (int i = threadIdx.x; i < n_eff; i += blockDim.x) ok[(size_t) t * n_blk + i] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < cand_n[t]; i += blockDim.x) {
        const int b = cand[(size_t) t * kMaxCandBlocks + i];
        if (b >= 0 && b < n_eff) ok[(size_t) t * n_blk + b] = 1;
    }
}

// ---- top-k ----------------------------------------------------------------------------------------------------

constexpr int kScoreBlocks = 1360;   // score blocks per token; each walks the row groups beyond the grid
constexpr int kTopkThreads = 1024;
constexpr int kTopkMax = 2048;

__device__ __forceinline__ uint32_t order_key(float f) {
    if (f == 0.f) f = 0.f;  // -0 and +0 tie
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Exclusive scan of a 0/1 flag over the block; `total` is the block sum. wsum >= 32 ints.
__device__ int block_scan(int flag, int& total, int* wsum) {
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

// Stage: one token's top-k from one block over its first n scores s[0..n). Keeps the K highest finite scores (ties to
// the lower index), sorted by index ascending, and pads out[0..out_ld) with -1; *out_n gets the count. A score of
// -inf is never kept.
__device__ void topk_block(const float* __restrict__ s, int n, int K, int32_t* __restrict__ out, int out_ld,
                           int32_t* __restrict__ out_n) {
    __shared__ int ids[kTopkMax];
    __shared__ uint32_t hist[256];
    __shared__ int wsum[32];
    __shared__ uint32_t s_prefix;
    __shared__ int s_need, s_cnt, s_base;

    int nvalid_local = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) nvalid_local += s[i] > -INFINITY;
    int nvalid;
    {
        int dummy;
        // total of a per-thread count: reuse the scan on a flag-free path
        const uint32_t unused = 0; (void) unused; (void) dummy;
        __shared__ int red[32];
        float v = warp_sum((float) nvalid_local);
        __syncthreads();
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = (int) v;
        __syncthreads();
        nvalid = 0;
        for (int i = 0; i < (blockDim.x >> 5); ++i) nvalid += red[i];
    }

    uint32_t thr = 0;
    int need_eq = 0;
    const bool all = nvalid <= K;
    if (!all) {
        if (threadIdx.x == 0) { s_prefix = 0; s_need = K; }
        for (int pass = 0; pass < 4; ++pass) {
            const int shift = 24 - 8 * pass;
            for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
            __syncthreads();
            const uint32_t prefix = s_prefix;
            for (int i = threadIdx.x; i < n; i += blockDim.x) {
                if (!(s[i] > -INFINITY)) continue;
                const uint32_t key = order_key(s[i]);
                if (pass == 0 || (key >> (shift + 8)) == (prefix >> (shift + 8))) atomicAdd(&hist[(key >> shift) & 255], 1u);
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                int need = s_need, cum = 0, sel = 0;
                for (int b = 255; b >= 0; --b) {
                    if (cum + (int) hist[b] >= need) { sel = b; break; }
                    cum += hist[b];
                }
                s_need = need - cum;
                s_prefix = prefix | ((uint32_t) sel << shift);
            }
            __syncthreads();
        }
        thr = s_prefix;
        need_eq = s_need;
    }
    if (threadIdx.x == 0) { s_cnt = 0; s_base = 0; }
    __syncthreads();

    // keepers: everything above the threshold (or all valid), then the lowest-index entries equal to it
    for (int i0 = 0; i0 < n; i0 += blockDim.x) {
        const int i = i0 + threadIdx.x;
        const bool ok = i < n && s[i] > -INFINITY && (all || order_key(s[i]) > thr);
        int tot;
        const int r = block_scan(ok ? 1 : 0, tot, wsum);
        if (ok) ids[s_cnt + r] = i;
        __syncthreads();
        if (threadIdx.x == 0) s_cnt += tot;
        __syncthreads();
    }
    if (!all) {
        for (int i0 = 0; i0 < n && s_base < need_eq; i0 += blockDim.x) {
            const int i = i0 + threadIdx.x;
            const bool eq = i < n && s[i] > -INFINITY && order_key(s[i]) == thr;
            int tot;
            const int r = block_scan(eq ? 1 : 0, tot, wsum);
            if (eq && s_base + r < need_eq) ids[s_cnt + s_base + r] = i;
            __syncthreads();
            if (threadIdx.x == 0) s_base += min(tot, need_eq - s_base);
            __syncthreads();
        }
        if (threadIdx.x == 0) s_cnt += s_base;
        __syncthreads();
    }
    const int cnt = s_cnt;
    int N = 1;
    while (N < cnt) N <<= 1;
    for (int i = cnt + threadIdx.x; i < N; i += blockDim.x) ids[i] = INT32_MAX;
    __syncthreads();
    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < N; i += blockDim.x) {
                const int ix = i ^ j;
                if (ix > i) {
                    const bool up = (i & k) == 0;
                    const int a = ids[i], b = ids[ix];
                    if ((a > b) == up) { ids[i] = b; ids[ix] = a; }
                }
            }
            __syncthreads();
        }
    }
    for (int i = threadIdx.x; i < out_ld; i += blockDim.x) out[i] = i < cnt ? ids[i] : -1;
    if (threadIdx.x == 0) *out_n = cnt;
}

// One block per token: topk_block over the rows the call's last position sees (all of n without `pos`).
__global__ void __launch_bounds__(kTopkThreads) topk_kernel(const float* __restrict__ scores, int n, int K,
                                                            int32_t* __restrict__ out, int out_ld,
                                                            int32_t* __restrict__ out_n, const int32_t* __restrict__ pos,
                                                            int ratio, int block_size) {
    const int t = blockIdx.x;
    const float* s = scores + (size_t) t * n;   // rows are n apart; the call may see fewer
    if (pos) {
        n = eff_rows(pos, gridDim.x, ratio, n * (block_size ? block_size : 1));
        if (block_size) n = (n + block_size - 1) / block_size;
    }
    topk_block(s, n, K, out + (size_t) t * out_ld, out_ld, out_n + t);
}


// ---- top-k of each token over many blocks ----------------------------------------------------------------------
//
// The same result as topk_kernel, which walks every row from one block. Each pass narrows the threshold by one byte of the
// order key over all the blocks: a pass reads the previous pass's histogram, so every block picks the same prefix. A mask
// then marks the rows above the threshold and those equal to it, and one block lists them in row order. The grid's y is
// the token: each token has its scores, its position, its state and its row bits, and its result does not depend on the
// others. A token that sees few rows takes its list from one block in the first launch, as topk_kernel would, and the
// launches after it return at once for that token: the passes cost more than the walk there.

constexpr int kSelThreads = 256;     // rows per group
constexpr int kSelOneBlock = 1024;   // a token with at most this many rows gets its list from one block (topk_block)
constexpr int kSelBlocks = 128;      // the passes' grid; each block walks the groups beyond the grid
constexpr int kSelGroups = 2052;     // groups the final block can list: each of them keeps a row

struct SelState {
    uint32_t hist[4][256];
    uint32_t prefix[4];
    int32_t need[4];
    uint32_t thr;
    int32_t need_eq;
    int32_t all;
};

__device__ __forceinline__ int sel_rows(int n, const int32_t* pos, int ratio, int block_size) {
    const int rows = min(n * (block_size ? block_size : 1), (pos[0] + 1) / ratio);
    return block_size ? (rows + block_size - 1) / block_size : rows;
}

// token blockIdx.y's state: the regions follow each other `stride` bytes apart
__device__ __forceinline__ SelState* sel_state(SelState* st, size_t stride) {
    return reinterpret_cast<SelState*>(reinterpret_cast<char*>(st) + stride * blockIdx.y);
}

// A token with few rows gets its list here and the passes leave it alone; otherwise its histograms start at zero.
__global__ void __launch_bounds__(kTopkThreads) select_begin_kernel(const float* __restrict__ s, int n, int K,
                                                                    int32_t* __restrict__ out, int out_ld,
                                                                    int32_t* __restrict__ out_n,
                                                                    const int32_t* __restrict__ pos, int ratio,
                                                                    int block_size, SelState* st, size_t stride) {
    const int rows = sel_rows(n, pos + blockIdx.y, ratio, block_size);
    if (rows <= kSelOneBlock) {
        topk_block(s + (size_t) n * blockIdx.y, rows, K, out + (size_t) out_ld * blockIdx.y, out_ld, out_n + blockIdx.y);
        return;
    }
    st = sel_state(st, stride);
    for (int i = threadIdx.x; i < 4 * 256; i += blockDim.x) st->hist[i / 256][i % 256] = 0;
}

// The prefix and the rows still needed after pass `done`, from its histogram; the same in every block.
__device__ __forceinline__ void sel_pick(SelState* st, int done, int K, uint32_t* lh, uint32_t& s_prefix, int& s_need) {
    lh[threadIdx.x] = st->hist[done][threadIdx.x];
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t prefix = done ? st->prefix[done] : 0u;
        const int need = done ? st->need[done] : K;
        int cum = 0, sel = 0;
        for (int b = 255; b >= 0; --b) {
            if (cum + (int) lh[b] >= need) { sel = b; break; }
            cum += (int) lh[b];
        }
        s_prefix = prefix | ((uint32_t) sel << (24 - 8 * done));
        s_need = need - cum;
    }
    __syncthreads();
}

__global__ void __launch_bounds__(kSelThreads) select_pass_kernel(const float* __restrict__ s, int n, int K,
                                                                  const int32_t* __restrict__ pos, int ratio, int block_size,
                                                                  int pass, SelState* st, size_t stride) {
    __shared__ uint32_t h[256], lh[256];
    __shared__ uint32_t s_prefix;
    __shared__ int s_need;
    s += (size_t) n * blockIdx.y;
    pos += blockIdx.y;
    st = sel_state(st, stride);
    n = sel_rows(n, pos, ratio, block_size);
    if (n <= kSelOneBlock || blockIdx.x * kSelThreads >= n) return;
    if (threadIdx.x == 0) { s_prefix = 0; s_need = K; }
    if (pass > 0) {
        sel_pick(st, pass - 1, K, lh, s_prefix, s_need);
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            st->prefix[pass] = s_prefix;
            st->need[pass] = s_need;
        }
    }
    h[threadIdx.x] = 0;
    __syncthreads();
    const uint32_t prefix = s_prefix;
    const int shift = 24 - 8 * pass;
    for (int base = blockIdx.x * kSelThreads; base < n; base += gridDim.x * kSelThreads) {
        const int j = base + threadIdx.x;
        if (j >= n || !(s[j] > -INFINITY)) continue;
        const uint32_t key = order_key(s[j]);
        if (pass == 0 || (key >> (shift + 8)) == (prefix >> (shift + 8))) atomicAdd(&h[(key >> shift) & 255], 1u);
    }
    __syncthreads();
    if (h[threadIdx.x]) atomicAdd(&st->hist[pass][threadIdx.x], h[threadIdx.x]);
}

// Row bits per group: above the threshold in `gt`, equal to it in `eq`; `gc` counts both.
__global__ void __launch_bounds__(kSelThreads) select_mask_kernel(const float* __restrict__ s, int n, int K,
                                                                  const int32_t* __restrict__ pos, int ratio, int block_size,
                                                                  SelState* st, size_t stride) {
    __shared__ uint32_t lh[256], w_gt[8], w_eq[8];
    __shared__ uint32_t s_thr;
    __shared__ int s_need;
    const size_t groups = (size_t) (n + kSelThreads - 1) / kSelThreads + 1;
    s += (size_t) n * blockIdx.y;
    pos += blockIdx.y;
    st = sel_state(st, stride);
    uint32_t* gt = reinterpret_cast<uint32_t*>(st + 1);
    uint32_t* eq = gt + groups * 8;
    uint32_t* gc = eq + groups * 8;
    n = sel_rows(n, pos, ratio, block_size);
    if (n <= kSelOneBlock || blockIdx.x * kSelThreads >= n) return;
    sel_pick(st, 3, K, lh, s_thr, s_need);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        st->thr = s_thr;
        st->need_eq = s_need;
    }
    const uint32_t thr = s_thr;
    // every valid row is kept when they are at most K
    __shared__ int s_all;
    lh[threadIdx.x] = st->hist[0][threadIdx.x];
    __syncthreads();
    if (threadIdx.x == 0) {
        int total = 0;
        for (int b = 0; b < 256; ++b) total += (int) lh[b];
        s_all = total <= K;
        if (blockIdx.x == 0) st->all = s_all;
    }
    __syncthreads();
    const bool all = s_all != 0;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int base = blockIdx.x * kSelThreads; base < n; base += gridDim.x * kSelThreads) {
        const int j = base + threadIdx.x;
        const bool valid = j < n && s[j] > -INFINITY;
        const uint32_t key = valid ? order_key(s[j]) : 0u;
        const uint32_t bg = __ballot_sync(0xffffffffu, valid && (all || key > thr));
        const uint32_t be = __ballot_sync(0xffffffffu, valid && !all && key == thr);
        const int g = base / kSelThreads;
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

__global__ void __launch_bounds__(kTopkThreads) select_final_kernel(int n, int32_t* __restrict__ out, int out_ld,
                                                                    int32_t* __restrict__ out_n, const int32_t* __restrict__ pos,
                                                                    int ratio, int block_size, SelState* st, size_t stride) {
    __shared__ int list[kSelGroups], cgt[kSelGroups], ceq[kSelGroups], at[kSelGroups];
    __shared__ int wsum[32];
    __shared__ int s_cnt, s_total, s_left;
    const size_t groups = (size_t) (n + kSelThreads - 1) / kSelThreads + 1;
    out += (size_t) out_ld * blockIdx.y;
    out_n += blockIdx.y;
    pos += blockIdx.y;
    st = sel_state(st, stride);
    const uint32_t* gt = reinterpret_cast<const uint32_t*>(st + 1);
    const uint32_t* eq = gt + groups * 8;
    const uint32_t* gc = eq + groups * 8;
    n = sel_rows(n, pos, ratio, block_size);
    if (n <= kSelOneBlock) return;
    const int ng = (n + kSelThreads - 1) / kSelThreads;
    if (threadIdx.x == 0) { s_cnt = 0; s_total = 0; s_left = st->all ? 0 : st->need_eq; }
    __syncthreads();
    for (int i0 = 0; i0 < ng; i0 += blockDim.x) {
        const int g = i0 + threadIdx.x;
        const int a = g < ng ? (int) gc[(size_t) g * 2] : 0, b = g < ng ? (int) gc[(size_t) g * 2 + 1] : 0;
        // a group with equal rows counts only while lower groups cannot have supplied all of them
        int tot_eq;
        const int eq_before = block_scan(b != 0 ? 1 : 0, tot_eq, wsum);
        const bool ok = a != 0 || (b != 0 && eq_before < s_left);
        int tot;
        const int r = block_scan(ok ? 1 : 0, tot, wsum);
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
    if (threadIdx.x == 0) {
        // the equals kept are the lowest rows
        int need = st->all ? 0 : st->need_eq, total = 0;
        for (int i = 0; i < s_cnt; ++i) {
            const int take = min(ceq[i], need);
            need -= take;
            ceq[i] = take;
            at[i] = total;
            total += cgt[i] + take;
        }
        s_total = total;
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
                out[k++] = g * kSelThreads + w * 32 + __ffs(m) - 1;
                m &= m - 1u;
            }
        }
    }
    for (int i = threadIdx.x; i < out_ld; i += blockDim.x)
        if (i >= s_total) out[i] = -1;
    if (threadIdx.x == 0) out_n[0] = s_total;
}

// ---- batched prompt entry points: kernels ---------------------------------------------------------------------

namespace wmma = nvcuda::wmma;

constexpr int kGemmM = 64, kGemmN = 64, kGemmK = 32, kGemmLd = kGemmK + 8, kGemmCLd = kGemmN + 4;

// y[T][out] = x . w^T. Block tile 64 tokens x 64 outputs; each warp owns 16 tokens. x is split into three bf16 parts
// (24 bits) so the product keeps fp32 accuracy; the weights are exact bf16.
__global__ void __launch_bounds__(128) gemm_bf16_kernel(const uint16_t* __restrict__ w, const float* __restrict__ x,
                                                        int T, int in, int out, float* __restrict__ y) {
    __shared__ __align__(128) __nv_bfloat16 xh[kGemmM * kGemmLd];
    __shared__ __align__(128) __nv_bfloat16 xl[kGemmM * kGemmLd];
    __shared__ __align__(128) __nv_bfloat16 xm[kGemmM * kGemmLd];
    __shared__ __align__(128) __nv_bfloat16 ws[kGemmN * kGemmLd];
    __shared__ __align__(128) float cs[kGemmM * kGemmCLd];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int m0 = blockIdx.y * kGemmM, n0 = blockIdx.x * kGemmN;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) wmma::fill_fragment(acc[u], 0.f);
    for (int k0 = 0; k0 < in; k0 += kGemmK) {
        {
            const int row = tid >> 1, c0 = (tid & 1) * 16;
            const int m = m0 + row;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (m < T) v = *reinterpret_cast<const float4*>(x + (size_t) m * in + k0 + c0 + 4 * q);
                const float f[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const __nv_bfloat16 hi = __float2bfloat16(f[i]);
                    const float r1 = f[i] - __bfloat162float(hi);
                    const __nv_bfloat16 mid = __float2bfloat16(r1);
                    xh[row * kGemmLd + c0 + 4 * q + i] = hi;
                    xm[row * kGemmLd + c0 + 4 * q + i] = mid;
                    xl[row * kGemmLd + c0 + 4 * q + i] = __float2bfloat16(r1 - __bfloat162float(mid));
                }
            }
        }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int idx = tid + 128 * j, row = idx >> 2, c = (idx & 3) * 8;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (n0 + row < out) v = *reinterpret_cast<const uint4*>(w + (size_t) (n0 + row) * in + k0 + c);
            *reinterpret_cast<uint4*>(ws + row * kGemmLd + c) = v;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < kGemmK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fh, fm, fl;
            wmma::load_matrix_sync(fh, xh + 16 * warp * kGemmLd + kk, kGemmLd);
            wmma::load_matrix_sync(fm, xm + 16 * warp * kGemmLd + kk, kGemmLd);
            wmma::load_matrix_sync(fl, xl + 16 * warp * kGemmLd + kk, kGemmLd);
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fw;
                wmma::load_matrix_sync(fw, ws + 16 * u * kGemmLd + kk, kGemmLd);
                // a fresh accumulator per k16 step, added in fp32, keeps the long sums at fp32 accuracy
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> part;
                wmma::fill_fragment(part, 0.f);
                wmma::mma_sync(part, fl, fw, part);
                wmma::mma_sync(part, fm, fw, part);
                wmma::mma_sync(part, fh, fw, part);
#pragma unroll
                for (int i = 0; i < part.num_elements; ++i) acc[u].x[i] += part.x[i];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int u = 0; u < 4; ++u)
        wmma::store_matrix_sync(cs + 16 * warp * kGemmCLd + 16 * u, acc[u], kGemmCLd, wmma::mem_row_major);
    __syncwarp();
    for (int r = 0; r < 16; ++r) {
        const int m = m0 + 16 * warp + r;
        if (m >= T) break;
        for (int c = lane; c < kGemmN; c += 32)
            if (n0 + c < out) y[(size_t) m * out + n0 + c] = cs[(16 * warp + r) * kGemmCLd + c];
    }
}

// A row of blockDim.x 16-byte words per block.
__global__ void window_copy_kernel(const uint4* __restrict__ ring, uint4* __restrict__ stage, int stage_cap, int64_t p_lo) {
    const int words = blockDim.x;
    const int64_t p = p_lo + blockIdx.x;
    stage[(size_t) (p % stage_cap) * words + threadIdx.x] = ring[(size_t) (p % kWindowSlots) * words + threadIdx.x];
}

__global__ void kv_window_stage_kernel(const float* __restrict__ raw, const float* __restrict__ norm_w,
                                       const int32_t* __restrict__ pos, float eps, const float* __restrict__ freq,
                                       uint8_t* __restrict__ stage, int stage_cap, uint8_t* __restrict__ ring,
                                       int ring_from, bool packed) {
    __shared__ float sh[kLatent];
    __shared__ float red[32];
    const int t = blockIdx.x, c = threadIdx.x;
    const float v = raw[(size_t) t * kLatent + c];
    const float ss = block_sum(v * v, red);
    const float r = 1.0f / sqrtf(ss / kLatent + eps);
    const float y = v * r * norm_w[c];
    sh[c] = y;
    __syncthreads();
    float o = y;
    if (c >= kLatent - kRopeDim) {
        const int p = (c - (kLatent - kRopeDim)) >> 1;
        float x0 = sh[kLatent - kRopeDim + 2 * p], x1 = sh[kLatent - kRopeDim + 2 * p + 1];
        rotate_pair(x0, x1, (float) pos[t], freq[p]);
        o = (c & 1) ? x1 : x0;
    }
    put_win(o, c, stage, (size_t) (pos[t] % stage_cap), packed);
    if (t >= ring_from) put_win(o, c, ring, (size_t) (pos[t] % kWindowSlots), packed);
}

// One block per token; a token closing a group reads its group's rows from the call, or from the carried state for
// the part of the group before the call. Same arithmetic as compress_kernel.
__global__ void compress_batch_kernel(const float* __restrict__ ckv, const float* __restrict__ csc,
                                      const int32_t* __restrict__ pos, int ratio, float eps,
                                      const float* __restrict__ norm_w, const float* __restrict__ freq,
                                      const float* __restrict__ state_kv, const float* __restrict__ state_score,
                                      uint8_t* __restrict__ comp_kv, float* __restrict__ comp_pre,
                                      int32_t* __restrict__ row_of, bool packed) {
    __shared__ float sh[kLatent];
    __shared__ float red[32];
    const int t = blockIdx.x, c = threadIdx.x;
    const int p = pos[t];
    const bool close = (p + 1) % ratio == 0;
    if (c == 0) row_of[t] = close ? p / ratio : kNoRow;
    if (!close) return;
    float kv[2] = {0.f, 0.f}, sc[2] = {0.f, 0.f};
    for (int j = 0; j < ratio; ++j) {
        const int tj = t - (ratio - 1 - j);
        if (tj >= 0) {
            kv[j] = ckv[(size_t) tj * kLatent + c];
            sc[j] = csc ? csc[(size_t) tj * kLatent + c] : 0.f;
        } else {
            kv[j] = state_kv[j * kLatent + c];
            sc[j] = state_score[j * kLatent + c];
        }
    }
    float m = sc[0];
    if (ratio > 1) m = fmaxf(m, sc[1]);
    float den = 0.f, num = 0.f;
    for (int j = 0; j < ratio; ++j) {
        const float e = expf(sc[j] - m);
        den += e;
        num += kv[j] * e;
    }
    const float comp = num / den;
    const float ss = block_sum(comp * comp, red);
    const float y = comp * (1.0f / sqrtf(ss / kLatent + eps)) * norm_w[c];
    comp_pre[(size_t) t * kLatent + c] = y;
    sh[c] = y;
    __syncthreads();
    float o = y;
    if (c >= kLatent - kRopeDim) {
        const int pr = (c - (kLatent - kRopeDim)) >> 1;
        float x0 = sh[kLatent - kRopeDim + 2 * pr], x1 = sh[kLatent - kRopeDim + 2 * pr + 1];
        rotate_pair(x0, x1, (float) (p + 1 - ratio), freq[pr]);
        o = (c & 1) ? x1 : x0;
    }
    put_comp(o, c, comp_kv, (size_t) (p / ratio), packed);
}

// The carried state after the call: the last token of each slot.
__global__ void compress_state_kernel(const float* __restrict__ ckv, const float* __restrict__ csc,
                                      const int32_t* __restrict__ pos, int n_tokens, int ratio,
                                      float* __restrict__ state_kv, float* __restrict__ state_score) {
    const int c = threadIdx.x;
    for (int d = 0; d < ratio; ++d) {
        const int tj = n_tokens - 1 - d;
        if (tj < 0) break;
        const int slot = pos[tj] % ratio;
        state_kv[slot * kLatent + c] = ckv[(size_t) tj * kLatent + c];
        state_score[slot * kLatent + c] = csc ? csc[(size_t) tj * kLatent + c] : 0.f;
    }
}

}  // namespace

// ---- launchers ------------------------------------------------------------------------------------------------

void gemv_bf16(const uint16_t* w, const float* x, int n_tokens, int in, int out, float* y, void* stream) {
    if (n_tokens < 1 || n_tokens > kMaxT || in % 8) throw std::invalid_argument("gemv_bf16: bad shape");
    gemv_bf16_kernel<<<(out + 7) / 8, 256, 0, as_stream(stream)>>>(w, x, n_tokens, in, out, y);
    check_launch("gemv_bf16");
}

void kv_window(const float* kv_raw, const float* norm_w, const int32_t* pos, int n_tokens, float rms_eps,
               const float* rope_freq, float* kv_out, uint8_t* ring, CacheFormat fmt, void* stream) {
    if (n_tokens < 1 || n_tokens > kMaxT) throw std::invalid_argument("kv_window: bad token count");
    kv_window_kernel<<<n_tokens, kLatent, 0, as_stream(stream)>>>(kv_raw, norm_w, pos, rms_eps, rope_freq, kv_out, ring,
                                                                  fmt == CacheFormat::Fp8);
    check_launch("kv_window");
}

size_t compress_workspace_bytes() { return (size_t) kMaxT * (3 * kLatent * sizeof(float) + sizeof(int32_t)) + 256; }

void compress(const CompressArgs& a, const float* x, const int32_t* pos, int n_tokens, void* workspace, void* stream) {
    if (n_tokens < 1 || n_tokens > kMaxT || (a.ratio != 1 && a.ratio != 2)) throw std::invalid_argument("compress: bad args");
    cudaStream_t s = as_stream(stream);
    float* ckv = static_cast<float*>(workspace);
    float* csc = ckv + kMaxT * kLatent;
    float* pre = csc + kMaxT * kLatent;
    int32_t* row_of = reinterpret_cast<int32_t*>(pre + kMaxT * kLatent);
    gemv_bf16(a.w_kv, x, n_tokens, a.n_embd, kLatent, ckv, stream);
    if (a.ratio > 1) gemv_bf16(a.w_gate, x, n_tokens, a.n_embd, kLatent, csc, stream);
    compress_kernel<<<1, kLatent, 0, s>>>(ckv, a.ratio > 1 ? csc : nullptr, pos, n_tokens, a.ratio, a.rms_eps, a.norm_w,
                                          a.rope_freq, a.state_kv, a.state_score,
                                          a.comp_kv, pre, row_of, a.format == CacheFormat::Fp8);
    check_launch("compress");
    if (a.w_idx_k && a.comp_ik) {
        index_key_kernel<<<n_tokens, 256, 0, s>>>(pre, row_of, pos, a.ratio, a.rms_eps, a.w_idx_k, a.idx_k_norm_w,
                                                  a.rope_freq, a.comp_ik, a.format == CacheFormat::Fp8);
        check_launch("index_key");
    }
}


void compress_window(const CompressArgs& a, const float* x, const int32_t* pos, int n_tokens, void* workspace, void* stream) {
    if (n_tokens < 1 || n_tokens > kMaxT || (a.ratio != 1 && a.ratio != 2)) throw std::invalid_argument("compress_window: bad args");
    cudaStream_t s = as_stream(stream);
    float* ckv = static_cast<float*>(workspace);
    float* csc = ckv + kMaxT * kLatent;
    float* pre = csc + kMaxT * kLatent;
    int32_t* row_of = reinterpret_cast<int32_t*>(pre + kMaxT * kLatent);
    gemv_bf16(a.w_kv, x, n_tokens, a.n_embd, kLatent, ckv, stream);
    if (a.ratio > 1) gemv_bf16(a.w_gate, x, n_tokens, a.n_embd, kLatent, csc, stream);
    compress_batch_kernel<<<n_tokens, kLatent, 0, s>>>(ckv, a.ratio > 1 ? csc : nullptr, pos, a.ratio, a.rms_eps, a.norm_w,
                                                       a.rope_freq, a.state_kv, a.state_score,
                                                       a.comp_kv, pre, row_of, a.format == CacheFormat::Fp8);
    check_launch("compress_window");
    if (a.w_idx_k && a.comp_ik) {
        index_key_kernel<<<n_tokens, 256, 0, s>>>(pre, row_of, pos, a.ratio, a.rms_eps, a.w_idx_k, a.idx_k_norm_w,
                                                  a.rope_freq, a.comp_ik, a.format == CacheFormat::Fp8);
        check_launch("index_key");
    }
}

void compress_window_commit(const CompressArgs& a, const int32_t* pos, int n_accepted, void* workspace, void* stream) {
    if (n_accepted < 1) return;
    float* ckv = static_cast<float*>(workspace);
    float* csc = ckv + kMaxT * kLatent;
    compress_state_kernel<<<1, kLatent, 0, as_stream(stream)>>>(ckv, a.ratio > 1 ? csc : nullptr, pos, n_accepted, a.ratio,
                                                                a.state_kv, a.state_score);
    check_launch("compress_window_commit");
}


namespace {
struct IdxWs {
    float* iq;
    float* iw;
    float* scores;
    float* bscore;
    uint8_t* blk_ok;
    SelState* sel;
};
size_t sel_bytes(int max_rows) {
    const size_t groups = (size_t) (max_rows + kSelThreads - 1) / kSelThreads + 1;
    return sizeof(SelState) + groups * (16 + 2) * sizeof(uint32_t);
}
// one token's state region in a window's run of them
size_t sel_stride(int max_rows) { return (sel_bytes(max_rows) + 255) / 256 * 256; }
IdxWs carve(void* base, int max_comp) {
    IdxWs w;
    float* p = static_cast<float*>(base);
    w.iq = p;      p += (size_t) kMaxT * kIdxHeads * kIdxDim;
    w.iw = p;      p += (size_t) kMaxT * kIdxHeads;
    w.scores = p;  p += (size_t) kMaxT * max_comp;
    w.bscore = p;  p += (size_t) kMaxT * max_comp;
    w.sel = reinterpret_cast<SelState*>(p);
    w.blk_ok = reinterpret_cast<uint8_t*>(p) + (size_t) kMaxT * sel_stride(max_comp);
    return w;
}

// The top-k of each of `n_tok` tokens over many blocks: token t's scores at t * n, its position pos[t], its list at
// out + t * out_ld, its state t * sel_stride(n) bytes after `st`. `n` rows are the buffers' capacity, the positions say
// how many each token sees.
void select_top(const float* scores, int n, int K, int32_t* out, int out_ld, int32_t* out_n, const int32_t* pos, int ratio,
                int block_size, SelState* st, int n_tok, cudaStream_t s) {
    const size_t stride = sel_stride(n);
    const int grid = std::min((n + kSelThreads - 1) / kSelThreads, kSelBlocks);
    select_begin_kernel<<<dim3(1, n_tok), kTopkThreads, 0, s>>>(scores, n, K, out, out_ld, out_n, pos, ratio, block_size,
                                                                 st, stride);
    for (int pass = 0; pass < 4; ++pass)
        select_pass_kernel<<<dim3(grid, n_tok), kSelThreads, 0, s>>>(scores, n, K, pos, ratio, block_size, pass, st, stride);
    select_mask_kernel<<<dim3(grid, n_tok), kSelThreads, 0, s>>>(scores, n, K, pos, ratio, block_size, st, stride);
    select_final_kernel<<<dim3(1, n_tok), kTopkThreads, 0, s>>>(n, out, out_ld, out_n, pos, ratio, block_size, st, stride);
}
}  // namespace

size_t indexer_workspace_bytes(int max_comp) {
    return (size_t) kMaxT * (kIdxHeads * kIdxDim + kIdxHeads) * sizeof(float) +
           (size_t) kMaxT * max_comp * (2 * sizeof(float) + 1) + (size_t) kMaxT * sel_stride(max_comp) + 256;
}

void top_rows(const float* scores, int n, int K, int32_t* out, int out_ld, int32_t* out_n, const int32_t* pos, int ratio,
              int block_size, void* workspace, void* stream) {
    cudaStream_t s = as_stream(stream);
    if (workspace) select_top(scores, n, K, out, out_ld, out_n, pos, ratio, block_size, static_cast<SelState*>(workspace), 1, s);
    else topk_kernel<<<1, kTopkThreads, 0, s>>>(scores, n, K, out, out_ld, out_n, pos, ratio, block_size);
}

size_t top_rows_workspace_bytes(int max_rows) { return sel_bytes(max_rows); }

void indexer_weights(const IndexerArgs& a, const float* x, int n_tokens, void* workspace, void* stream) {
    if (n_tokens < 1 || n_tokens > kMaxT || a.n_comp < 1) throw std::invalid_argument("indexer_weights: bad args");
    gemv_bf16(a.w_proj, x, n_tokens, a.n_embd, kIdxHeads, carve(workspace, a.n_comp).iw, stream);
}

void indexer_select(const IndexerArgs& a, const float* qr, const float* x, const int32_t* pos, int n_tokens,
                    void* workspace, void* stream) {
    indexer_weights(a, x, n_tokens, workspace, stream);
    indexer_rank(a, qr, pos, n_tokens, workspace, stream);
}

void indexer_rank(const IndexerArgs& a, const float* qr, const int32_t* pos, int n_tokens, void* workspace, void* stream) {
    if (n_tokens < 1 || n_tokens > kMaxT || a.n_comp < 1) throw std::invalid_argument("indexer_rank: bad args");
    cudaStream_t s = as_stream(stream);
    const IdxWs w = carve(workspace, a.n_comp);
    gemv_bf16(a.w_q, qr, n_tokens, 1280, kIdxHeads * kIdxDim, w.iq, stream);
    idx_rope_round_kernel<<<dim3(n_tokens, kIdxHeads), kIdxDim, 0, s>>>(w.iq, pos, a.rope_freq, a.format == CacheFormat::Fp8);
    const int n_blk = (a.n_comp + a.block_size - 1) / a.block_size;
    const float w_scale = 1.0f / sqrtf((float) (kIdxHeads * kIdxDim));
    if (a.mode == IndexMode::Level2) {
        mark_blocks_kernel<<<n_tokens, 256, 0, s>>>(a.cand_ids, a.cand_n, n_blk, pos, a.n_comp, a.ratio,
                                                          a.block_size, w.blk_ok);
    }
    dim3 grid(std::min((a.n_comp + 127) / 128, kScoreBlocks), n_tokens);
    (a.format == CacheFormat::Fp8 ? idx_score_kernel<true> : idx_score_kernel<false>)<<<grid, 128, 0, s>>>(w.iq, w.iw, pos, a.comp_ik, a.n_comp, a.ratio,
                                          a.block_size, a.mode == IndexMode::Level2 ? w.blk_ok : nullptr, n_blk,
                                          w_scale, w.scores);
    check_launch("idx_score");
    if (a.mode == IndexMode::Level1) {
        dim3 g((n_blk + 127) / 128, n_tokens);
        block_max_kernel<<<g, 128, 0, s>>>(w.scores, pos, a.n_comp, a.ratio, a.block_size, w.bscore);
        select_top(w.bscore, n_blk, std::min(a.cand_blocks, kMaxCandBlocks), a.cand_ids, kMaxCandBlocks, a.cand_n, pos, a.ratio,
                   a.block_size, w.sel, n_tokens, s);
        check_launch("cand_blocks");
    }
    select_top(w.scores, a.n_comp, kTopK, a.out_ids, kTopK, a.out_n, pos, a.ratio, 0, w.sel, n_tokens, s);
    check_launch("idx_topk");
}

// ---- batched prompt entry points ------------------------------------------------------------------------------

void gemm_bf16(const uint16_t* w, const float* x, int n_tokens, int in, int out, float* y, void* stream) {
    if (n_tokens < 1 || in % kGemmK) throw std::invalid_argument("gemm_bf16: bad shape");
    gemm_bf16_kernel<<<dim3((out + kGemmN - 1) / kGemmN, (n_tokens + kGemmM - 1) / kGemmM), 128, 0, as_stream(stream)>>>(
        w, x, n_tokens, in, out, y);
    check_launch("gemm_bf16");
}

void window_stage_copy(const uint8_t* ring, uint8_t* stage, int stage_cap, int64_t p_lo, int count, CacheFormat fmt,
                       void* stream) {
    if (count <= 0) return;
    window_copy_kernel<<<count, win_row_bytes(fmt) / 16, 0, as_stream(stream)>>>(reinterpret_cast<const uint4*>(ring),
                                                                    reinterpret_cast<uint4*>(stage), stage_cap, p_lo);
    check_launch("window_stage_copy");
}

void kv_window_stage(const float* kv_raw, const float* norm_w, const int32_t* pos, int n_tokens, float rms_eps,
                     const float* rope_freq, uint8_t* stage, int stage_cap, uint8_t* ring, int ring_from, CacheFormat fmt,
                     void* stream) {
    if (n_tokens < 1) throw std::invalid_argument("kv_window_stage: bad token count");
    kv_window_stage_kernel<<<n_tokens, kLatent, 0, as_stream(stream)>>>(kv_raw, norm_w, pos, rms_eps, rope_freq, stage,
                                                                        stage_cap, ring, ring_from, fmt == CacheFormat::Fp8);
    check_launch("kv_window_stage");
}

size_t compress_batch_workspace_bytes(int max_tokens) {
    return (size_t) max_tokens * (3 * kLatent * sizeof(float) + sizeof(int32_t)) + 256;
}

void compress_batch(const CompressArgs& a, const float* x, const int32_t* pos, int n_tokens, void* workspace, void* stream) {
    if (n_tokens < 1 || (a.ratio != 1 && a.ratio != 2)) throw std::invalid_argument("compress_batch: bad args");
    cudaStream_t s = as_stream(stream);
    float* ckv = static_cast<float*>(workspace);
    float* csc = ckv + (size_t) n_tokens * kLatent;
    float* pre = csc + (size_t) n_tokens * kLatent;
    int32_t* row_of = reinterpret_cast<int32_t*>(pre + (size_t) n_tokens * kLatent);
    gemm_bf16(a.w_kv, x, n_tokens, a.n_embd, kLatent, ckv, stream);
    if (a.ratio > 1) gemm_bf16(a.w_gate, x, n_tokens, a.n_embd, kLatent, csc, stream);
    const float* sc = a.ratio > 1 ? csc : nullptr;
    compress_batch_kernel<<<n_tokens, kLatent, 0, s>>>(ckv, sc, pos, a.ratio, a.rms_eps, a.norm_w, a.rope_freq, a.state_kv,
                                                       a.state_score, a.comp_kv, pre, row_of, a.format == CacheFormat::Fp8);
    check_launch("compress_batch");
    compress_state_kernel<<<1, kLatent, 0, s>>>(ckv, sc, pos, n_tokens, a.ratio, a.state_kv, a.state_score);
    check_launch("compress_state");
    if (a.w_idx_k && a.comp_ik) {
        index_key_kernel<<<n_tokens, 256, 0, s>>>(pre, row_of, pos, a.ratio, a.rms_eps, a.w_idx_k, a.idx_k_norm_w,
                                                  a.rope_freq, a.comp_ik, a.format == CacheFormat::Fp8);
        check_launch("index_key");
    }
}

namespace {
IdxWs carve_batch(void* base, int T, int max_comp) {
    IdxWs w;
    float* p = static_cast<float*>(base);
    w.iq = p;      p += (size_t) T * kIdxHeads * kIdxDim;
    w.iw = p;      p += (size_t) T * kIdxHeads;
    w.scores = p;  p += (size_t) T * max_comp;
    w.bscore = p;  p += (size_t) T * max_comp;
    w.blk_ok = reinterpret_cast<uint8_t*>(p);
    return w;
}
}  // namespace

size_t indexer_batch_workspace_bytes(int max_tokens, int max_comp) {
    return (size_t) max_tokens * (kIdxHeads * kIdxDim + kIdxHeads) * sizeof(float) +
           (size_t) max_tokens * max_comp * (2 * sizeof(float) + 1) + 256;
}

void indexer_select_batch(const IndexerArgs& a, const float* qr, const float* x, const int32_t* pos, int n_tokens,
                          void* workspace, void* stream) {
    if (n_tokens < 1 || n_tokens > 65535 || a.n_comp < 1) throw std::invalid_argument("indexer_select_batch: bad args");
    cudaStream_t s = as_stream(stream);
    const IdxWs w = carve_batch(workspace, n_tokens, a.n_comp);
    gemm_bf16(a.w_q, qr, n_tokens, 1280, kIdxHeads * kIdxDim, w.iq, stream);
    idx_rope_round_kernel<<<dim3(n_tokens, kIdxHeads), kIdxDim, 0, s>>>(w.iq, pos, a.rope_freq, a.format == CacheFormat::Fp8);
    gemm_bf16(a.w_proj, x, n_tokens, a.n_embd, kIdxHeads, w.iw, stream);
    const int n_blk = (a.n_comp + a.block_size - 1) / a.block_size;
    const float w_scale = 1.0f / sqrtf((float) (kIdxHeads * kIdxDim));
    if (a.mode == IndexMode::Level2) mark_blocks_kernel<<<n_tokens, 256, 0, s>>>(a.cand_ids, a.cand_n, n_blk, pos, a.n_comp, a.ratio,
                                                          a.block_size, w.blk_ok);
    dim3 grid(std::min((a.n_comp + 127) / 128, kScoreBlocks), n_tokens);
    (a.format == CacheFormat::Fp8 ? idx_score_kernel<true> : idx_score_kernel<false>)<<<grid, 128, 0, s>>>(w.iq, w.iw, pos, a.comp_ik, a.n_comp, a.ratio,
                                          a.block_size, a.mode == IndexMode::Level2 ? w.blk_ok : nullptr, n_blk,
                                          w_scale, w.scores);
    check_launch("idx_score");
    if (a.mode == IndexMode::Level1) {
        dim3 g((n_blk + 127) / 128, n_tokens);
        block_max_kernel<<<g, 128, 0, s>>>(w.scores, pos, a.n_comp, a.ratio, a.block_size, w.bscore);
        topk_kernel<<<n_tokens, kTopkThreads, 0, s>>>(w.bscore, n_blk, std::min(a.cand_blocks, kMaxCandBlocks), a.cand_ids,
                                                      kMaxCandBlocks, a.cand_n, pos, a.ratio, a.block_size);
        check_launch("cand_blocks");
    }
    topk_kernel<<<n_tokens, kTopkThreads, 0, s>>>(w.scores, a.n_comp, kTopK, a.out_ids, kTopK, a.out_n, pos, a.ratio, 0);
    check_launch("idx_topk");
}

}  // namespace basalt::models::deepseek41
