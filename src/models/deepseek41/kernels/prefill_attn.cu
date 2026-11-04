// DeepSeek-V4.1 prompt-chunk sparse MQA on tensor cores: one block per (chunk of 64 keys, group of 16 heads, token).
// The key rows are gathered once into shared memory and serve as both K and V. The chunk statistics and the merge with
// the sink follow the layout of the scalar kernel in attn_mqa.cu.
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
#include "basalt/models/deepseek41/prefill_attn.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <stdexcept>
#include <string>

namespace basalt::models::deepseek41 {
namespace {

namespace wmma = nvcuda::wmma;

constexpr int kHeads = 16;      // heads per block (one wmma row tile)
constexpr int kKeys = kAttnChunk;
constexpr int kLd = kAttnHeadDim + 8;   // padded half row
constexpr int kScLd = kKeys + 4;
constexpr int kPLd = kKeys + 8;
constexpr int kThreads = 256;

constexpr size_t kSmemBytes = (size_t) kHeads * kLd * 2 + (size_t) kKeys * kLd * 2 + 2 * (size_t) kHeads * kScLd * 4 +
                              (size_t) kHeads * kPLd * 2 + kKeys * 4;

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

template <bool kWindow>
__global__ void __launch_bounds__(kThreads) mqa_tc_partial(MqaArgs a, int chunk0, int n_total, float* o_part, float* ml) {
    extern __shared__ __align__(128) unsigned char smem[];
    half* qs = reinterpret_cast<half*>(smem);
    half* ks = qs + kHeads * kLd;
    float* sc0 = reinterpret_cast<float*>(ks + kKeys * kLd);
    float* sc1 = sc0 + kHeads * kScLd;
    half* ps = reinterpret_cast<half*>(sc1 + kHeads * kScLd);
    int* rowid = reinterpret_cast<int*>(ps + kHeads * kPLd);

    const int c = blockIdx.x, g = blockIdx.y, t = blockIdx.z;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int slot = chunk0 + c;
    const int head0 = g * kHeads;

    int r = -1;
    if (tid < kKeys) {
        const int j = c * kKeys + tid;
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
    const bool any = __syncthreads_or(tid < kKeys && r >= 0);
    if (!any) {
        if (tid < kHeads) {
            float* m = ml + (((size_t) t * kAttnHeads + head0 + tid) * n_total + slot) * 2;
            m[0] = -INFINITY;
            m[1] = 0.f;
        }
        return;
    }

    // queries to fp16
    const float4* qg = reinterpret_cast<const float4*>(a.q + ((size_t) t * kAttnHeads + head0) * kAttnHeadDim);
    for (int i = tid; i < kHeads * kAttnHeadDim / 4; i += kThreads) {
        const float4 v = qg[i];
        const int h = i / (kAttnHeadDim / 4), d = (i % (kAttnHeadDim / 4)) * 4;
        __half2* dst = reinterpret_cast<__half2*>(qs + h * kLd + d);
        dst[0] = __floats2half2_rn(v.x, v.y);
        dst[1] = __floats2half2_rn(v.z, v.w);
    }
    // key rows (fp16); masked rows are zero
    const half* base = reinterpret_cast<const half*>(kWindow ? a.win : a.comp);
    const int64_t stride = kWindow ? kAttnHeadDim : a.comp_stride;
    for (int i = tid; i < kKeys * (kAttnHeadDim / 8); i += kThreads) {
        const int kk = i / (kAttnHeadDim / 8), d = (i % (kAttnHeadDim / 8)) * 8;
        const int row = rowid[kk];
        uint4 v = make_uint4(0, 0, 0, 0);
        if (row >= 0) v = *reinterpret_cast<const uint4*>(base + (size_t) row * stride + d);
        *reinterpret_cast<uint4*>(ks + kk * kLd + d) = v;
    }
    __syncthreads();

    // scores: warp w -> key tile w & 3, half of the 512 dims w >> 2
    {
        const int nt = warp & 3, kh = warp >> 2;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.f);
        for (int k0 = kh * 256; k0 < kh * 256 + 256; k0 += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fq;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fk;
            wmma::load_matrix_sync(fq, qs + k0, kLd);
            wmma::load_matrix_sync(fk, ks + nt * 16 * kLd + k0, kLd);
            wmma::mma_sync(acc, fq, fk, acc);
        }
        wmma::store_matrix_sync((kh ? sc1 : sc0) + nt * 16, acc, kScLd, wmma::mem_row_major);
    }
    __syncthreads();

    // chunk statistics; the weights go to fp16 for the value product
    for (int h = warp; h < kHeads; h += kThreads / 32) {
        float s[2];
        float m = -INFINITY;
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            const int kk = lane + 32 * u;
            s[u] = rowid[kk] >= 0 ? (sc0[h * kScLd + kk] + sc1[h * kScLd + kk]) * a.scale : -INFINITY;
            m = fmaxf(m, s[u]);
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        float l = 0.f;
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            const float p = s[u] == -INFINITY ? 0.f : expf(s[u] - m);
            ps[h * kPLd + lane + 32 * u] = __float2half(p);
            l += p;
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
        if (lane == 0) {
            float* mlp = ml + (((size_t) t * kAttnHeads + head0 + h) * n_total + slot) * 2;
            mlp[0] = m;
            mlp[1] = l;
        }
    }
    __syncthreads();

    // weighted sum: warp w -> 4 tiles of 16 dims
    {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) wmma::fill_fragment(acc[u], 0.f);
#pragma unroll
        for (int k0 = 0; k0 < kKeys; k0 += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fp;
            wmma::load_matrix_sync(fp, ps + k0, kPLd);
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fv;
                wmma::load_matrix_sync(fv, ks + k0 * kLd + (warp * 4 + u) * 16, kLd);
                wmma::mma_sync(acc[u], fp, fv, acc[u]);
            }
        }
        float* o = o_part + (((size_t) t * kAttnHeads + head0) * n_total + slot) * kAttnHeadDim;
#pragma unroll
        for (int u = 0; u < 4; ++u)
            wmma::store_matrix_sync(o + (warp * 4 + u) * 16, acc[u], (size_t) n_total * kAttnHeadDim, wmma::mem_row_major);
    }
}

__global__ void __launch_bounds__(256) mqa_tc_merge(const float* o_part, const float* ml, int n_chunks, const float* sinks,
                                                    const int32_t* pos, RopeTable rope, float* out) {
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

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

template <bool kWindow>
void launch(const MqaArgs& a, int n_chunks, int chunk0, int n_total, float* o_part, float* ml, cudaStream_t s) {
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(mqa_tc_partial<kWindow>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kSmemBytes);
        attr = true;
    }
    mqa_tc_partial<kWindow><<<dim3(n_chunks, kAttnHeads / kHeads, a.n_tokens), kThreads, kSmemBytes, s>>>(a, chunk0, n_total,
                                                                                                          o_part, ml);
    check("mqa_tc_partial");
}

}  // namespace

void sparse_mqa_tc(const MqaArgs& a, float* workspace, void* stream) {
    if (a.win_type != LatentType::F16 || (a.sel && a.n_sel > 0 && a.comp_type != LatentType::F16)) {
        sparse_mqa(a, workspace, stream);
        return;
    }
    if (a.n_tokens < 1 || a.n_sel > 512 || a.n_sel < 0) throw std::invalid_argument("sparse_mqa_tc: arguments");
    cudaStream_t s = (cudaStream_t) stream;
    const bool sparse = a.sel != nullptr && a.n_sel > 0;
    const int nw = (kAttnWindow + kKeys - 1) / kKeys, ns = sparse ? (a.n_sel + kKeys - 1) / kKeys : 0;
    const int n_chunks = nw + ns;
    float* o_part = workspace;
    float* ml = workspace + (size_t) a.n_tokens * kAttnHeads * n_chunks * kAttnHeadDim;
    launch<true>(a, nw, 0, n_chunks, o_part, ml, s);
    if (sparse) launch<false>(a, ns, nw, n_chunks, o_part, ml, s);
    mqa_tc_merge<<<dim3(kAttnHeads, a.n_tokens), 256, 0, s>>>(o_part, ml, n_chunks, a.sinks, a.pos, a.rope, a.out);
    check("mqa_tc_merge");
}

}  // namespace basalt::models::deepseek41
