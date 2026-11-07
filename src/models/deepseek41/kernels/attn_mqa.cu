// Sparse MQA with an attention sink, the inverse RoPE of its output, and the RoPE tables: adapted from the
// DeepSeek-V4.1 attention graph (build_attention41) and ggml's rope_ext / rope_ext_back in the llama.cpp
// DeepSeek-V4.1 support.
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
#include "basalt/models/deepseek41/attn.hpp"
#include "basalt/models/deepseek41/attn_index.hpp"
#include "cache_rows.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace basalt::models::deepseek41 {

// ---- RoPE tables ------------------------------------------------------------------------------------------------

namespace {

float yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * std::log((float) n_ctx_orig / (n_rot * 2.0f * 3.14159265358979323846f)) / (2.0f * std::log(base));
}

}  // namespace

RopeTable make_rope_table(float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast,
                          float beta_slow, int n_ctx_orig) {
    constexpr int n_dims = kAttnRopeDim;
    const float lo = std::max(0.0f, std::floor(yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base)));
    const float hi = std::min((float) (n_dims - 1), std::ceil(yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base)));
    const float theta_scale = std::pow(freq_base, -2.0f / n_dims);
    RopeTable t{};
    float extrap = 1.0f;
    for (int i = 0; i < n_dims / 2; ++i) {
        float step = freq_scale * extrap;
        if (ext_factor != 0.0f) {
            const float y = ((float) i - lo) / std::max(0.001f, hi - lo);
            const float ramp = (1.0f - std::min(1.0f, std::max(0.0f, y))) * ext_factor;
            step = step * (1.0f - ramp) + extrap * ramp;
        }
        t.step[i] = step;
        extrap *= theta_scale;
    }
    t.mscale = attn_factor;
    if (ext_factor != 0.0f) t.mscale *= 1.0f + 0.1f * std::log(1.0f / freq_scale);
    return t;
}

RopeTable rope_table_window() { return make_rope_table(10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0); }

RopeTable rope_table_compressed(float freq_base, float factor, float beta_fast, float beta_slow, int n_ctx_orig) {
    const float freq_scale = 1.0f / factor;
    // the YaRN magnitude correction is cancelled, as the model's reference applies none
    const float attn_factor = 1.0f / (1.0f + 0.1f * std::log(1.0f / freq_scale));
    return make_rope_table(freq_base, freq_scale, 1.0f, attn_factor, beta_fast, beta_slow, n_ctx_orig);
}

namespace {

constexpr int kHeadsPerBlock = 16;
constexpr int kBlockThreads = 256;

__device__ __forceinline__ void rotate_pair(float& x0, float& x1, float pos, float step, float mscale, bool inverse) {
    float s, c;
    sincosf(pos * step, &s, &c);
    c *= mscale;
    s *= mscale;
    if (inverse) s = -s;
    const float a = x0 * c - x1 * s;
    const float b = x0 * s + x1 * c;
    x0 = a;
    x1 = b;
}

// One warp per row: the rope pairs of the last 64 dims.
__global__ void rope_rows_kernel(float* rows, int n_rows, int row_stride, int rows_per_pos, const int32_t* pos,
                                 RopeTable rope, bool inverse) {
    const int row = blockIdx.x;
    const int i = threadIdx.x;  // pair
    if (row >= n_rows || i >= kAttnRopeDim / 2) return;
    float* p = rows + (size_t) row * row_stride + (row_stride - kAttnRopeDim) + 2 * i;
    float x0 = p[0], x1 = p[1];
    rotate_pair(x0, x1, (float) pos[row / rows_per_pos], rope.step[i], rope.mscale, inverse);
    p[0] = x0;
    p[1] = x1;
}

// ---- partial attention: one block per (chunk of 64 keys, group of 16 heads, token) ---------------------------

// How a row type is read: Raw4 values 4i..4i+3 (i < 128) and Raw2 values 2i, 2i+1 (i < 256) as loaded, with their scale
// where the row has one; converted to float at use.
struct NoScales {};
struct RowF32 {
    using Raw4 = float4;
    using Raw2 = float2;
    using Scales = NoScales;
    static __device__ __forceinline__ Scales scales(const uint8_t*) { return {}; }
    static __device__ __forceinline__ void cvt4(const Raw4& v, const Scales&, int, float k[4]) { cvt4(v, k); }
    static constexpr int kRowBytes = kAttnHeadDim * 4;
    static __device__ __forceinline__ Raw4 load4(const uint8_t* r, int i) { return reinterpret_cast<const float4*>(r)[i]; }
    static __device__ __forceinline__ Raw2 load2(const uint8_t* r, int i) { return reinterpret_cast<const float2*>(r)[i]; }
    static __device__ __forceinline__ void cvt4(const Raw4& v, float k[4]) { k[0] = v.x; k[1] = v.y; k[2] = v.z; k[3] = v.w; }
    static __device__ __forceinline__ void cvt2(const Raw2& v, float k[2]) { k[0] = v.x; k[1] = v.y; }
};
struct RowF16 {
    using Raw4 = uint2;
    using Raw2 = uint32_t;
    using Scales = NoScales;
    static __device__ __forceinline__ Scales scales(const uint8_t*) { return {}; }
    static __device__ __forceinline__ void cvt4(const Raw4& v, const Scales&, int, float k[4]) { cvt4(v, k); }
    static constexpr int kRowBytes = kAttnHeadDim * 2;
    static __device__ __forceinline__ Raw4 load4(const uint8_t* r, int i) { return reinterpret_cast<const uint2*>(r)[i]; }
    static __device__ __forceinline__ Raw2 load2(const uint8_t* r, int i) { return reinterpret_cast<const uint32_t*>(r)[i]; }
    static __device__ __forceinline__ void cvt4(const Raw4& v, float k[4]) {
        const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&v.x));
        const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&v.y));
        k[0] = a.x; k[1] = a.y; k[2] = b.x; k[3] = b.y;
    }
    static __device__ __forceinline__ void cvt2(const Raw2& v, float k[2]) {
        const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&v));
        k[0] = f.x; k[1] = f.y;
    }
};
// window rows: the codes, then the group's scale (Raw4's y; above bit 16 of Raw2)
struct RowFp8 {
    using Raw4 = uint2;
    using Raw2 = uint32_t;
    using Scales = NoScales;
    static constexpr int kRowBytes = kWinRowBytes;
    static __device__ __forceinline__ Scales scales(const uint8_t*) { return {}; }
    static __device__ __forceinline__ Raw4 load4(const uint8_t* r, int i) {
        return make_uint2(reinterpret_cast<const uint32_t*>(r)[i], r[kLatent + i / 8]);
    }
    static __device__ __forceinline__ Raw2 load2(const uint8_t* r, int i) {
        return (uint32_t) reinterpret_cast<const uint16_t*>(r)[i] | ((uint32_t) r[kLatent + i / 16] << 16);
    }
    static __device__ __forceinline__ void cvt4(const Raw4& v, const Scales&, int, float k[4]) { rows::fp8_values<4>(v.x, v.y, k); }
    static __device__ __forceinline__ void cvt2(const Raw2& v, float k[2]) { rows::fp8_values<2>(v & 0xffffu, v >> 16, k); }
};
// compressed rows: the codes in the low bits, the group's scale above them
struct RowFp4 {
    using Raw4 = uint32_t;
    using Raw2 = uint32_t;
    using Scales = NoScales;
    static constexpr int kRowBytes = kCompRowBytes;
    static __device__ __forceinline__ Scales scales(const uint8_t*) { return {}; }
    static __device__ __forceinline__ Raw4 load4(const uint8_t* r, int i) {
        return (uint32_t) reinterpret_cast<const uint16_t*>(r)[i] | ((uint32_t) r[kLatent / 2 + i / 4] << 16);
    }
    static __device__ __forceinline__ Raw2 load2(const uint8_t* r, int i) {
        return (uint32_t) r[i] | ((uint32_t) r[kLatent / 2 + i / 8] << 8);
    }
    static __device__ __forceinline__ void cvt4(const Raw4& v, const Scales&, int, float k[4]) { rows::fp4_values<4>(v & 0xffffu, v >> 16, k); }
    static __device__ __forceinline__ void cvt2(const Raw2& v, float k[2]) { rows::fp4_values<2>(v & 0xffu, v >> 8, k); }
};

constexpr int kScoreRows = kAttnChunk / (kBlockThreads / 32);  // rows per warp
constexpr int kScoreKeys = 4;                                   // keys scored per read of the query
constexpr int kPvBatch = 32;                                   // rows whose loads are in flight together

// The chunk's scores, softmax statistics and weighted sum over rows of type R. Each loop issues all its row loads
// before using them.
template <class R>
__device__ __forceinline__ void mqa_chunk(const uint8_t* base, float scale, const float4* qs, float (*sc)[kAttnChunk],
                                          const int* rowid, float* mlp0, size_t ml_head, float* o_part0, size_t o_head) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

    // scores
    // kScoreKeys keys at a time share each read of the query from shared memory; every key's dot products and
    // reductions keep their own order
#pragma unroll 1
    for (int i0 = 0; i0 < kScoreRows; i0 += kScoreKeys) {
        typename R::Raw4 raw[kScoreKeys][4];
        typename R::Scales scl[kScoreKeys];
#pragma unroll
        for (int u = 0; u < kScoreKeys; ++u) {
            const int row = rowid[warp + (i0 + u) * (kBlockThreads / 32)];
            scl[u] = typename R::Scales{};
            if (row >= 0) scl[u] = R::scales(base + (size_t) row * R::kRowBytes);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                raw[u][j] = typename R::Raw4{};
                if (row >= 0) raw[u][j] = R::load4(base + (size_t) row * R::kRowBytes, lane + 32 * j);
            }
        }
        float v[kScoreKeys][kHeadsPerBlock];
#pragma unroll
        for (int u = 0; u < kScoreKeys; ++u)
#pragma unroll
            for (int h = 0; h < kHeadsPerBlock; ++h) v[u][h] = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float kv[kScoreKeys][4];
#pragma unroll
            for (int u = 0; u < kScoreKeys; ++u) R::cvt4(raw[u][j], scl[u], j, kv[u]);
#pragma unroll
            for (int h = 0; h < kHeadsPerBlock; ++h) {
                const float4 q4 = qs[h * (kAttnHeadDim / 4) + lane + 32 * j];
#pragma unroll
                for (int u = 0; u < kScoreKeys; ++u) {
                    v[u][h] = fmaf(q4.x, kv[u][0], v[u][h]);
                    v[u][h] = fmaf(q4.y, kv[u][1], v[u][h]);
                    v[u][h] = fmaf(q4.z, kv[u][2], v[u][h]);
                    v[u][h] = fmaf(q4.w, kv[u][3], v[u][h]);
                }
            }
        }
#pragma unroll
        for (int u = 0; u < kScoreKeys; ++u) {
            const int kk = warp + (i0 + u) * (kBlockThreads / 32);
            // transposed reduction: every lane ends with one head's sum
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float send = (lane & 16) ? v[u][i] : v[u][i + 8];
                const float keep = (lane & 16) ? v[u][i + 8] : v[u][i];
                v[u][i] = keep + __shfl_xor_sync(0xffffffffu, send, 16);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float send = (lane & 8) ? v[u][i] : v[u][i + 4];
                const float keep = (lane & 8) ? v[u][i + 4] : v[u][i];
                v[u][i] = keep + __shfl_xor_sync(0xffffffffu, send, 8);
            }
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const float send = (lane & 4) ? v[u][i] : v[u][i + 2];
                const float keep = (lane & 4) ? v[u][i + 2] : v[u][i];
                v[u][i] = keep + __shfl_xor_sync(0xffffffffu, send, 4);
            }
            {
                const float send = (lane & 2) ? v[u][0] : v[u][1];
                const float keep = (lane & 2) ? v[u][1] : v[u][0];
                v[u][0] = keep + __shfl_xor_sync(0xffffffffu, send, 2);
            }
            v[u][0] += __shfl_xor_sync(0xffffffffu, v[u][0], 1);
            const int head =
                ((lane >> 4) & 1) * 8 + ((lane >> 3) & 1) * 4 + ((lane >> 2) & 1) * 2 + ((lane >> 1) & 1);
            if ((lane & 1) == 0) sc[head][kk] = rowid[kk] >= 0 ? v[u][0] * scale : -INFINITY;
        }
    }
    __syncthreads();

    // chunk softmax statistics; scores become unnormalised weights
    for (int h = warp; h < kHeadsPerBlock; h += kBlockThreads / 32) {
        const float s0 = sc[h][lane], s1 = sc[h][lane + 32];
        float m = fmaxf(s0, s1);
#pragma unroll
        for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        const float p0 = s0 == -INFINITY ? 0.f : expf(s0 - m);
        const float p1 = s1 == -INFINITY ? 0.f : expf(s1 - m);
        sc[h][lane] = p0;
        sc[h][lane + 32] = p1;
        float l = p0 + p1;
#pragma unroll
        for (int o = 16; o; o >>= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
        if (lane == 0) {
            float* mlp = mlp0 + (size_t) h * ml_head;
            mlp[0] = m;
            mlp[1] = l;
        }
    }
    __syncthreads();

    // weighted sum of the chunk's rows
    float acc[kHeadsPerBlock][2];
#pragma unroll
    for (int h = 0; h < kHeadsPerBlock; ++h) acc[h][0] = acc[h][1] = 0.f;
#pragma unroll 1
    for (int k0 = 0; k0 < kAttnChunk; k0 += kPvBatch) {
        typename R::Raw2 r2[kPvBatch];
#pragma unroll
        for (int i = 0; i < kPvBatch; ++i) {
            const int row = rowid[k0 + i];
            r2[i] = typename R::Raw2{};
            if (row >= 0) r2[i] = R::load2(base + (size_t) row * R::kRowBytes, tid);
        }
        // masked rows hold zeros and zero weights, so their terms add exact zeros
#pragma unroll
        for (int i4 = 0; i4 < kPvBatch; i4 += 4) {
            float k2[4][2];
#pragma unroll
            for (int u = 0; u < 4; ++u) R::cvt2(r2[i4 + u], k2[u]);
#pragma unroll
            for (int h = 0; h < kHeadsPerBlock; ++h) {
                const float4 p4 = *reinterpret_cast<const float4*>(&sc[h][k0 + i4]);  // four rows' weights
                const float pu[4] = {p4.x, p4.y, p4.z, p4.w};
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    acc[h][0] = fmaf(pu[u], k2[u][0], acc[h][0]);
                    acc[h][1] = fmaf(pu[u], k2[u][1], acc[h][1]);
                }
            }
        }
    }
#pragma unroll
    for (int h = 0; h < kHeadsPerBlock; ++h) {
        float2* o = reinterpret_cast<float2*>(o_part0 + (size_t) h * o_head);
        o[tid] = make_float2(acc[h][0], acc[h][1]);
    }
}

// Blocks [0, n_win) are window chunks (rows of type W, indexing the ring), the rest compressed chunks (rows of type C,
// indexing the selected ids).
template <class W, class C>
__global__ void __launch_bounds__(kBlockThreads) mqa_partial_kernel(MqaArgs a, int n_win, int n_chunks_total, float* o_part,
                                                                    float* ml) {
    const int slot = blockIdx.x;
    const bool kWindow = slot < n_win;
    const int c = kWindow ? slot : slot - n_win;
    const int g = blockIdx.y;
    const int t = blockIdx.z;
    const int tid = threadIdx.x;

    __shared__ float4 qs[kHeadsPerBlock * kAttnHeadDim / 4];
    __shared__ float sc[kHeadsPerBlock][kAttnChunk];
    __shared__ int rowid[kAttnChunk];

    int r = -1;
    if (tid < kAttnChunk) {
        const int j = c * kAttnChunk + tid;
        if (kWindow) {
            if (j < a.win_n[t]) {
                const int cap = a.win_cap;
                r = (int) (((int64_t) a.win_start[t] + j) % cap);
                if (r < 0) r += cap;
            }
        } else if (j < a.n_sel) {
            const int id = a.sel[(size_t) t * a.n_sel + j];
            if (id >= 0 && id < a.comp_limit[t]) r = id;
        }
        rowid[tid] = r;
    }
    const bool any = __syncthreads_or(tid < kAttnChunk && r >= 0);
    const int head0 = g * kHeadsPerBlock;
    float* mlp0 = ml + (((size_t) t * kAttnHeads + head0) * n_chunks_total + slot) * 2;
    if (!any) {
        if (tid < kHeadsPerBlock) {
            float* m = mlp0 + (size_t) tid * n_chunks_total * 2;
            m[0] = -INFINITY;
            m[1] = 0.f;
        }
        return;
    }

    const float4* qg = reinterpret_cast<const float4*>(a.q + ((size_t) t * kAttnHeads + head0) * kAttnHeadDim);
    for (int i = tid; i < kHeadsPerBlock * kAttnHeadDim / 4; i += kBlockThreads) qs[i] = qg[i];
    __syncthreads();

    float* o0 = o_part + (((size_t) t * kAttnHeads + head0) * n_chunks_total + slot) * kAttnHeadDim;
    if (kWindow)
        mqa_chunk<W>(static_cast<const uint8_t*>(a.win), a.scale, qs, sc, rowid, mlp0, (size_t) n_chunks_total * 2, o0,
                     (size_t) n_chunks_total * kAttnHeadDim);
    else
        mqa_chunk<C>(static_cast<const uint8_t*>(a.comp), a.scale, qs, sc, rowid, mlp0, (size_t) n_chunks_total * 2, o0,
                     (size_t) n_chunks_total * kAttnHeadDim);
}

// ---- merge of the chunks with the sink, then the inverse RoPE -------------------------------------------------

__global__ void __launch_bounds__(256) mqa_merge_kernel(const float* o_part, const float* ml, int n_chunks,
                                                        const float* sinks, const int32_t* pos, RopeTable rope,
                                                        float* out) {
    const int h = blockIdx.x, t = blockIdx.y;
    const int tid = threadIdx.x;
    const size_t head = (size_t) t * kAttnHeads + h;
    const float* mlp = ml + head * n_chunks * 2;
    const float sink = sinks[h];
    float M = sink;
    for (int c = 0; c < n_chunks; ++c) M = fmaxf(M, mlp[2 * c]);
    float denom = expf(sink - M);
    float acc0 = 0.f, acc1 = 0.f;
    for (int c = 0; c < n_chunks; ++c) {
        const float m = mlp[2 * c];
        if (m == -INFINITY) continue;
        const float w = expf(m - M);
        denom = fmaf(mlp[2 * c + 1], w, denom);
        const float2 o = reinterpret_cast<const float2*>(o_part + (head * n_chunks + c) * kAttnHeadDim)[tid];
        acc0 = fmaf(o.x, w, acc0);
        acc1 = fmaf(o.y, w, acc1);
    }
    const float inv = 1.0f / denom;
    acc0 *= inv;
    acc1 *= inv;
    const int d = 2 * tid;
    if (d >= kAttnNopeDim) rotate_pair(acc0, acc1, (float) pos[t], rope.step[(d - kAttnNopeDim) / 2], rope.mscale, true);
    reinterpret_cast<float2*>(out + head * kAttnHeadDim)[tid] = make_float2(acc0, acc1);
}

void check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

template <class W, class C>
void launch_partial(const MqaArgs& a, int n_win, int n_total, float* o_part, float* ml, cudaStream_t s) {
    mqa_partial_kernel<W, C><<<dim3(n_total, kAttnHeads / kHeadsPerBlock, a.n_tokens), kBlockThreads, 0, s>>>(
        a, n_win, n_total, o_part, ml);
    check_launch("mqa_partial");
}

}  // namespace

void rope_rows(float* rows, int n_rows, int row_stride, int rows_per_pos, const int32_t* pos, const RopeTable& rope,
               bool inverse, void* stream) {
    rope_rows_kernel<<<n_rows, 32, 0, (cudaStream_t) stream>>>(rows, n_rows, row_stride, rows_per_pos, pos, rope,
                                                                inverse);
    check_launch("rope_rows");
}

namespace {
int win_chunks() { return (kAttnWindow + kAttnChunk - 1) / kAttnChunk; }
int sel_chunks(int n_sel) { return (n_sel + kAttnChunk - 1) / kAttnChunk; }
}  // namespace

size_t sparse_mqa_workspace_floats(int n_tokens, int n_sel) {
    const size_t n_chunks = (size_t) win_chunks() + sel_chunks(n_sel);
    return (size_t) n_tokens * kAttnHeads * n_chunks * (kAttnHeadDim + 2);
}

void sparse_mqa(const MqaArgs& a, float* workspace, void* stream) {
    if (a.n_tokens < 1 || a.n_tokens > kAttnMaxTokens) throw std::invalid_argument("sparse_mqa: n_tokens");
    if (a.n_sel > 512 || a.n_sel < 0) throw std::invalid_argument("sparse_mqa: n_sel");
    cudaStream_t s = (cudaStream_t) stream;
    const bool sparse = a.sel != nullptr && a.n_sel > 0;
    const int nw = win_chunks();
    const int n_chunks = nw + sel_chunks(sparse ? a.n_sel : 0);
    float* o_part = workspace;
    float* ml = workspace + (size_t) a.n_tokens * kAttnHeads * n_chunks * kAttnHeadDim;

    // window and compressed chunks share one launch
    const LatentType w = a.win_type,
                     c = sparse ? a.comp_type : w == LatentType::Fp8 ? LatentType::Fp4 : w == LatentType::F16 ? LatentType::F16 : LatentType::F32;
    if (w == LatentType::Fp8 && c == LatentType::Fp4) launch_partial<RowFp8, RowFp4>(a, nw, n_chunks, o_part, ml, s);
    else if (w == LatentType::F16 && c == LatentType::F16) launch_partial<RowF16, RowF16>(a, nw, n_chunks, o_part, ml, s);
    else if (w == LatentType::F16 && c == LatentType::F32) launch_partial<RowF16, RowF32>(a, nw, n_chunks, o_part, ml, s);
    else if (w == LatentType::F32 && c == LatentType::F32) launch_partial<RowF32, RowF32>(a, nw, n_chunks, o_part, ml, s);
    else throw std::invalid_argument("sparse_mqa: latent types");
    mqa_merge_kernel<<<dim3(kAttnHeads, a.n_tokens), 256, 0, s>>>(o_part, ml, n_chunks, a.sinks, a.pos, a.rope, a.out);
    check_launch("mqa_merge");
}

}  // namespace basalt::models::deepseek41
