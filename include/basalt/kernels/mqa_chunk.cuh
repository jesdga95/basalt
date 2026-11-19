// include/basalt/kernels/mqa_chunk.cuh - one chunk of a decode MQA over a shared 512-wide latent: the scores of 64 rows
// for 8 or 16 heads, their softmax statistics and the unnormalised weighted sum of the rows' latents, and the merge of
// a head's chunks with an attention sink. Device code only (no kernel): include from a .cu. Adapted from the
// llama.cpp latent MQA attention graph (build_attention41).
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
//
// A key is the 512-wide latent, followed by `Rope` more dims (0 or a multiple of 64) that only the scores read; the
// value is the latent. A row type R tells how a row is read:
//   R::kRowBytes, R::Raw4 / load4(row, i) / cvt4(raw, scales, j, k[4]): latent values 4i..4i+3 (i < 128)
//   R::Raw2 / load2(row, i) / cvt2(raw, k[2]): latent values 2i, 2i+1 (i < 256)
//   R::Scales / scales(row): whatever a row carries for cvt4 (NoScales when nothing)
//   with Rope > 0, R::RawR / loadr(row, i) / cvtr(raw, k[2]): values 512 + 2i, 512 + 2i + 1 (i < Rope / 2)
// Loads come first and conversions at use, so a lane's reads of several rows are in flight together.
#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace basalt::kernels::mqa {

constexpr int kThreads = 256;   ///< a chunk block
constexpr int kChunk = 64;      ///< rows of a chunk
constexpr int kLatent = 512;    ///< the value width and the keys' first dims

struct NoScales {};

constexpr int kScoreRows = kChunk / (kThreads / 32);   // rows per warp
constexpr int kScoreKeys = 4;                          // keys scored per read of the query
// Rows whose loads are in flight together. The batch also sets the size of the unrolled weighted sum, which keeps the
// kernel's code within what the instruction cache holds at launch.
constexpr int kPvBatch = 16;

// R::RawR where the keys have rope dims (a placeholder where they have none)
template <class R, bool HasRope> struct RawROf { using T = uint32_t; };
template <class R> struct RawROf<R, true> { using T = typename R::RawR; };

// A lane's values of kScoreKeys keys (4 at lane + 32 j of each key's latent, j < 4, and Rope / 64 pairs of the rest),
// rows of type R; masked keys give zeros. All row loads are issued before the first use.
template <class R, int Rope>
__device__ __forceinline__ void score_keys(const uint8_t* base, const int* rowid, int i0, float (*kv)[4][4], float (*kr)[2 * (Rope > 0 ? Rope / 64 : 1)]) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    constexpr int NR = Rope / 64;   // rope pairs per lane
    constexpr int NRA = NR > 0 ? NR : 1;
    typename R::Raw4 raw[kScoreKeys][4];
    typename R::Scales scl[kScoreKeys];
    using RawR = typename RawROf<R, (NR > 0)>::T;
    [[maybe_unused]] RawR rr[kScoreKeys][NRA];
#pragma unroll
    for (int u = 0; u < kScoreKeys; ++u) {
        const int row = rowid[warp + (i0 + u) * (kThreads / 32)];
        scl[u] = typename R::Scales{};
        if (row >= 0) scl[u] = R::scales(base + (size_t) row * R::kRowBytes);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            raw[u][j] = typename R::Raw4{};
            if (row >= 0) raw[u][j] = R::load4(base + (size_t) row * R::kRowBytes, lane + 32 * j);
        }
        if constexpr (NR > 0) {
#pragma unroll
            for (int p = 0; p < NR; ++p) {
                rr[u][p] = RawR{};
                if (row >= 0) rr[u][p] = R::loadr(base + (size_t) row * R::kRowBytes, lane + 32 * p);
            }
        }
    }
#pragma unroll
    for (int u = 0; u < kScoreKeys; ++u) {
#pragma unroll
        for (int j = 0; j < 4; ++j) R::cvt4(raw[u][j], scl[u], j, kv[u][j]);
        if constexpr (NR > 0) {
#pragma unroll
            for (int p = 0; p < NR; ++p) R::cvtr(rr[u][p], &kr[u][2 * p]);
        }
    }
}

// A thread's two values of kPvBatch rows from k0, rows of type R; masked rows give zeros. All loads are issued first.
template <class R>
__device__ __forceinline__ void pv_rows(const uint8_t* base, const int* rowid, int k0, float (*k2)[2]) {
    typename R::Raw2 r2[kPvBatch];
#pragma unroll
    for (int i = 0; i < kPvBatch; ++i) {
        const int row = rowid[k0 + i];
        r2[i] = typename R::Raw2{};
        if (row >= 0) r2[i] = R::load2(base + (size_t) row * R::kRowBytes, threadIdx.x);
    }
#pragma unroll
    for (int i = 0; i < kPvBatch; ++i) R::cvt2(r2[i], k2[i]);
}

// One step of the transposed reduction: a lane gives away the half of its N head sums its partner keeps (the upper
// half where `off` is set in its lane id) and adds the partner's half to its own.
template <int N>
__device__ __forceinline__ void halve_heads(float* v, int lane, int off) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
        const float send = (lane & off) ? v[i] : v[i + N];
        const float keep = (lane & off) ? v[i + N] : v[i];
        v[i] = keep + __shfl_xor_sync(0xffffffffu, send, off);
    }
}

// The chunk's scores, softmax statistics and weighted sum; its rows are of type W (`window`) or C. Only the row loads
// and their decoding depend on the type: the arithmetic is one body for both, so the kernel's code is not doubled.
// qs: the HB query rows in shared memory, (kLatent + Rope) / 4 float4 each, already in place; sc: [HB][kChunk] shared;
// rowid: the chunk's kChunk rows (< 0: masked). Writes each head's (max, sum of weights) at mlp0 + h * ml_head and its
// unnormalised weighted sum (kLatent floats) at o_part0 + h * o_head.
template <class W, class C, int HB, int Rope>
__device__ __forceinline__ void chunk(bool window, const uint8_t* base, float scale, const float4* qs, float (*sc)[kChunk],
                                      const int* rowid, float* mlp0, size_t ml_head, float* o_part0, size_t o_head) {
    static_assert(Rope % 64 == 0, "the keys' extra dims come in pairs per lane");
    constexpr int NR = Rope / 64;
    constexpr int QS = (kLatent + Rope) / 4;   // float4 per query row
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

    // scores
    // kScoreKeys keys at a time share each read of the query from shared memory; every key's dot products and
    // reductions keep their own order
#pragma unroll 1
    for (int i0 = 0; i0 < kScoreRows; i0 += kScoreKeys) {
        float kv[kScoreKeys][4][4];
        float kr[kScoreKeys][2 * (NR > 0 ? NR : 1)];
        if (window) score_keys<W, Rope>(base, rowid, i0, kv, kr);
        else score_keys<C, Rope>(base, rowid, i0, kv, kr);
        float v[kScoreKeys][HB];
#pragma unroll
        for (int u = 0; u < kScoreKeys; ++u)
#pragma unroll
            for (int h = 0; h < HB; ++h) v[u][h] = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int h = 0; h < HB; ++h) {
                const float4 q4 = qs[h * QS + lane + 32 * j];
#pragma unroll
                for (int u = 0; u < kScoreKeys; ++u) {
                    v[u][h] = fmaf(q4.x, kv[u][j][0], v[u][h]);
                    v[u][h] = fmaf(q4.y, kv[u][j][1], v[u][h]);
                    v[u][h] = fmaf(q4.z, kv[u][j][2], v[u][h]);
                    v[u][h] = fmaf(q4.w, kv[u][j][3], v[u][h]);
                }
            }
        }
        if constexpr (NR > 0) {
            const float2* qs2 = reinterpret_cast<const float2*>(qs);
#pragma unroll
            for (int p = 0; p < NR; ++p)
#pragma unroll
                for (int h = 0; h < HB; ++h) {
                    const float2 q2 = qs2[h * 2 * QS + kLatent / 2 + lane + 32 * p];
#pragma unroll
                    for (int u = 0; u < kScoreKeys; ++u) {
                        v[u][h] = fmaf(q2.x, kr[u][2 * p], v[u][h]);
                        v[u][h] = fmaf(q2.y, kr[u][2 * p + 1], v[u][h]);
                    }
                }
        }
#pragma unroll
        for (int u = 0; u < kScoreKeys; ++u) {
            const int kk = warp + (i0 + u) * (kThreads / 32);
            // transposed reduction: the xor steps from 16 halve the heads a lane holds until one is left, the rest
            // add whole sums; every head's 32 lane sums meet in the same pairs and order whatever HB is
            static_assert(HB == 8 || HB == 16, "8 or 16 heads a block");
            halve_heads<HB / 2>(v[u], lane, 16);
            halve_heads<HB / 4>(v[u], lane, 8);
            halve_heads<HB / 8>(v[u], lane, 4);
            if constexpr (HB == 16) halve_heads<1>(v[u], lane, 2);
            else v[u][0] += __shfl_xor_sync(0xffffffffu, v[u][0], 2);
            v[u][0] += __shfl_xor_sync(0xffffffffu, v[u][0], 1);
            // lane bits 4, 3, 2 (and 1 at 16 heads) name the head, high to low
            const int head = HB == 16 ? ((lane >> 4) & 1) * 8 + ((lane >> 3) & 1) * 4 + ((lane >> 2) & 1) * 2 + ((lane >> 1) & 1)
                                      : ((lane >> 4) & 1) * 4 + ((lane >> 3) & 1) * 2 + ((lane >> 2) & 1);
            if ((lane & (32 / HB - 1)) == 0) sc[head][kk] = rowid[kk] >= 0 ? v[u][0] * scale : -INFINITY;
        }
    }
    __syncthreads();

    // chunk softmax statistics; scores become unnormalised weights
    for (int h = warp; h < HB; h += kThreads / 32) {
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
    float acc[HB][2];
#pragma unroll
    for (int h = 0; h < HB; ++h) acc[h][0] = acc[h][1] = 0.f;
#pragma unroll 1
    for (int k0 = 0; k0 < kChunk; k0 += kPvBatch) {
        float k2[kPvBatch][2];
        if (window) pv_rows<W>(base, rowid, k0, k2);
        else pv_rows<C>(base, rowid, k0, k2);
        // masked rows hold zeros and zero weights, so their terms add exact zeros
#pragma unroll
        for (int i4 = 0; i4 < kPvBatch; i4 += 4) {
#pragma unroll
            for (int h = 0; h < HB; ++h) {
                const float4 p4 = *reinterpret_cast<const float4*>(&sc[h][k0 + i4]);  // four rows' weights
                const float pu[4] = {p4.x, p4.y, p4.z, p4.w};
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    acc[h][0] = fmaf(pu[u], k2[i4 + u][0], acc[h][0]);
                    acc[h][1] = fmaf(pu[u], k2[i4 + u][1], acc[h][1]);
                }
            }
        }
    }
#pragma unroll
    for (int h = 0; h < HB; ++h) {
        float2* o = reinterpret_cast<float2*>(o_part0 + (size_t) h * o_head);
        o[tid] = make_float2(acc[h][0], acc[h][1]);
    }
}

// The merge of HB heads' chunks with their sinks by one block: head h's chunk c has its (max, sum of weights) at
// mlp + h * ml_head + 2 c and its unnormalised weighted sum at o + h * o_head + c * kLatent; the sink joins the denominator
// and has no value. Thread tid gets latent values 2 tid, 2 tid + 1 of every head (out[h]); blockDim.x is kLatent / 2. Each
// head's sums run over its chunks in order, whatever HB is. scratch: merge_scratch_floats(HB, n_chunks) floats of shared
// memory (none for one head). L2: read through L2 only (the chunks were written by other blocks of the same launch).
__host__ __device__ constexpr int merge_scratch_floats(int hb, int n_chunks) { return 2 * hb * n_chunks + hb; }

template <int HB, bool L2>
__device__ __forceinline__ void merge(const float* o, size_t o_head, const float* mlp, size_t ml_head, int n_chunks, const float* sinks,
                                      float* scratch, float2 (&out)[HB]) {
    auto ld = [](const float* p) { return L2 ? __ldcg(p) : *p; };
    auto ld2 = [](const float* p) { return L2 ? __ldcg(reinterpret_cast<const float2*>(p)) : *reinterpret_cast<const float2*>(p); };
    const int tid = threadIdx.x;
    if constexpr (HB == 1) {   // one head: the chunks stream through once, no scratch
        const float sink = sinks[0];
        float M = sink;
        for (int c = 0; c < n_chunks; ++c) M = fmaxf(M, ld(mlp + 2 * c));
        float denom = expf(sink - M);
        float acc0 = 0.f, acc1 = 0.f;
        for (int c = 0; c < n_chunks; ++c) {
            const float m = ld(mlp + 2 * c);
            if (m == -INFINITY) continue;
            const float e = expf(m - M);
            denom = fmaf(ld(mlp + 2 * c + 1), e, denom);
            const float2 v = ld2(o + (size_t) c * kLatent + 2 * tid);
            acc0 = fmaf(v.x, e, acc0);
            acc1 = fmaf(v.y, e, acc1);
        }
        const float inv = 1.0f / denom;
        out[0] = make_float2(acc0 * inv, acc1 * inv);
        return;
    }
    float* w = scratch;                       // [HB][n_chunks]: the chunk's weight, -1 for a chunk without rows
    float* l = scratch + HB * n_chunks;       // [HB][n_chunks]
    float* inv = scratch + 2 * HB * n_chunks; // [HB]
    for (int i = tid; i < HB * n_chunks; i += blockDim.x) {
        const int h = i / n_chunks, c = i - h * n_chunks;
        w[i] = ld(mlp + h * ml_head + 2 * c);
        l[i] = ld(mlp + h * ml_head + 2 * c + 1);
    }
    __syncthreads();
    if (tid < HB) {
        float* wh = w + tid * n_chunks;
        const float* lh = l + tid * n_chunks;
        const float sink = sinks[tid];
        float M = sink;
        for (int c = 0; c < n_chunks; ++c) M = fmaxf(M, wh[c]);
        float denom = expf(sink - M);
        for (int c = 0; c < n_chunks; ++c) {
            const float m = wh[c];
            if (m == -INFINITY) {
                wh[c] = -1.f;
                continue;
            }
            const float e = expf(m - M);
            denom = fmaf(lh[c], e, denom);
            wh[c] = e;
        }
        inv[tid] = 1.0f / denom;
    }
    __syncthreads();
    float acc[HB][2];
#pragma unroll
    for (int h = 0; h < HB; ++h) acc[h][0] = acc[h][1] = 0.f;
    // 32 chunk loads of a thread in flight
#pragma unroll (32 / HB)
    for (int c = 0; c < n_chunks; ++c) {
        float2 v[HB];
#pragma unroll
        for (int h = 0; h < HB; ++h) v[h] = w[h * n_chunks + c] >= 0.f ? ld2(o + h * o_head + (size_t) c * kLatent + 2 * tid) : make_float2(0.f, 0.f);
#pragma unroll
        for (int h = 0; h < HB; ++h) {
            const float e = w[h * n_chunks + c];
            if (e < 0.f) continue;
            acc[h][0] = fmaf(v[h].x, e, acc[h][0]);
            acc[h][1] = fmaf(v[h].y, e, acc[h][1]);
        }
    }
#pragma unroll
    for (int h = 0; h < HB; ++h) out[h] = make_float2(acc[h][0] * inv[h], acc[h][1] * inv[h]);
}

}  // namespace basalt::kernels::mqa
