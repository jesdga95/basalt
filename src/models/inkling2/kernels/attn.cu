// src/models/inkling2/kernels/attn.cu - Inkling 2's decode attention: a query head's scores take a per-query relative bias
// over the last kExtent positions, and a per-query sink logit sits in the softmax's denominator.
//
// A block takes one row, one key/value head (its three query heads) and one slice of the keys; its four warps walk the
// slice in tiles of 32 keys through shared memory (a lane scores one key, then the lanes sum the values over the tile).
// The slices of a row and head merge in a second kernel with the sink. Every slice boundary comes from the row's own key
// count, so a row's result does not depend on the other rows of the call.
#include "basalt/models/inkling2/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace basalt::models::inkling2 {
namespace {

constexpr int kWarps = 4;
constexpr int kTile = 32;
constexpr int kGroup = kHeads / kKvHeads;   // 3 query heads a key/value head
constexpr int kKStride = kHeadDim / 2 + 1;  // words a key row takes in shared memory: the lanes read their rows without conflicts
constexpr int kMaxSplits = 64;

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("inkling2 ") + what + ": " + cudaGetErrorString(e));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m));
    return v;
}

// Four cache values of a row as fp32 (fp16 or unit-scale E4M3).
__device__ __forceinline__ float4 load4(const void* cache, bool fp8, size_t idx) {
    if (fp8) {
        const uint32_t u = *reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(cache) + idx);
        __nv_fp8x2_e4m3 lo, hi;
        lo.__x = (__nv_fp8x2_storage_t) (u & 0xFFFF);
        hi.__x = (__nv_fp8x2_storage_t) (u >> 16);
        const float2 a = (float2) lo, b = (float2) hi;
        return make_float4(a.x, a.y, b.x, b.y);
    }
    const uint2 u = *reinterpret_cast<const uint2*>(static_cast<const __half*>(cache) + idx);
    const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
    const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
    return make_float4(a.x, a.y, b.x, b.y);
}

struct Smem {
    float q[kGroup][kHeadDim];
    float rs[kGroup][kRel];
    uint32_t k[kWarps][kTile][kKStride];      // a tile's keys as fp16 pairs
    __half2 v[kWarps][kTile][kHeadDim / 2];  // its values (fp16 and E4M3 caches are exact in fp16)
    float m[kWarps][kGroup], l[kWarps][kGroup];
    float acc[kWarps][kGroup][kHeadDim];
};

__global__ void __launch_bounds__(kWarps * 32) attn_decode_kernel(AttnArgs a, int splits, float* __restrict__ part_ml,
                                                                  float* __restrict__ part_acc) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
    const int r = blockIdx.x, g = blockIdx.y, s = blockIdx.z;
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int p = a.pos[r];
    const int lo = a.window > 0 ? max(0, p - a.window + 1) : 0, hi = p + 1;
    const int n = hi - lo;
    const int chunk = ((n + splits - 1) / splits + kTile - 1) / kTile * kTile;
    const int s_lo = lo + s * chunk, s_hi = min(hi, s_lo + chunk);
    for (int i = threadIdx.x; i < kGroup * kHeadDim; i += blockDim.x)
        sm.q[i / kHeadDim][i % kHeadDim] = a.q[((size_t) r * kHeads + g * kGroup) * kHeadDim + i];
    for (int i = threadIdx.x; i < kGroup * kRel; i += blockDim.x)
        sm.rs[i / kRel][i % kRel] = a.r[((size_t) r * kHeads + g * kGroup) * kRel + i];
    __syncthreads();
    float m[kGroup], l[kGroup], acc[kGroup][4];
#pragma unroll
    for (int h = 0; h < kGroup; ++h) {
        m[h] = -INFINITY;
        l[h] = 0.f;
#pragma unroll
        for (int u = 0; u < 4; ++u) acc[h][u] = 0.f;
    }
    const size_t head_base = (size_t) g * a.cap;
    for (int t0 = s_lo + w * kTile; t0 < s_hi; t0 += kWarps * kTile) {
        // the tile into this warp's shared rows
        for (int j = 0; j < kTile; ++j) {
            const int key = t0 + j;
            float4 kv = make_float4(0.f, 0.f, 0.f, 0.f), vv = kv;
            if (key < s_hi) {
                const size_t idx = (head_base + (size_t) (key % a.cap)) * kHeadDim + lane * 4;
                kv = load4(a.k_cache, a.fp8, idx);
                vv = load4(a.v_cache, a.fp8, idx);
            }
            const __half2 k0 = __floats2half2_rn(kv.x, kv.y), k1 = __floats2half2_rn(kv.z, kv.w);
            sm.k[w][j][lane * 2] = *reinterpret_cast<const uint32_t*>(&k0);
            sm.k[w][j][lane * 2 + 1] = *reinterpret_cast<const uint32_t*>(&k1);
            sm.v[w][j][lane * 2] = __floats2half2_rn(vv.x, vv.y);
            sm.v[w][j][lane * 2 + 1] = __floats2half2_rn(vv.z, vv.w);
        }
        __syncwarp();
        // lane `lane` scores key t0 + lane for the three heads
        const int key = t0 + lane;
        float sc[kGroup] = {0.f, 0.f, 0.f};
#pragma unroll 8
        for (int d = 0; d < kHeadDim / 2; ++d) {
            const uint32_t u = sm.k[w][lane][d];
            const float2 kf = __half22float2(*reinterpret_cast<const __half2*>(&u));
#pragma unroll
            for (int h = 0; h < kGroup; ++h) sc[h] = fmaf(sm.q[h][2 * d], kf.x, fmaf(sm.q[h][2 * d + 1], kf.y, sc[h]));
        }
        const int dist = p - key;
#pragma unroll
        for (int h = 0; h < kGroup; ++h) {
            if (key >= s_hi) {
                sc[h] = -INFINITY;
                continue;
            }
            float b = 0.f;
            if (dist < kExtent)
#pragma unroll
                for (int i = 0; i < kRel; ++i) b = fmaf(sm.rs[h][i], a.rel_table[i * kExtent + dist], b);
            sc[h] = sc[h] * a.scale + b;
        }
        float pr[kGroup];
#pragma unroll
        for (int h = 0; h < kGroup; ++h) {
            const float mt = warp_max(sc[h]);
            const float mn = fmaxf(m[h], mt);
            const float corr = m[h] == -INFINITY ? 0.f : expf(m[h] - mn);
            pr[h] = sc[h] == -INFINITY ? 0.f : expf(sc[h] - mn);
            l[h] = l[h] * corr + warp_sum(pr[h]);
#pragma unroll
            for (int u = 0; u < 4; ++u) acc[h][u] *= corr;
            m[h] = mn;
        }
        for (int j = 0; j < kTile; ++j) {
            const float2 v01 = __half22float2(sm.v[w][j][lane * 2]), v23 = __half22float2(sm.v[w][j][lane * 2 + 1]);
            const float4 vv = make_float4(v01.x, v01.y, v23.x, v23.y);
#pragma unroll
            for (int h = 0; h < kGroup; ++h) {
                const float pj = __shfl_sync(0xffffffffu, pr[h], j);
                acc[h][0] = fmaf(pj, vv.x, acc[h][0]);
                acc[h][1] = fmaf(pj, vv.y, acc[h][1]);
                acc[h][2] = fmaf(pj, vv.z, acc[h][2]);
                acc[h][3] = fmaf(pj, vv.w, acc[h][3]);
            }
        }
        __syncwarp();
    }
    // the four warps' parts of this slice, merged in warp order
#pragma unroll
    for (int h = 0; h < kGroup; ++h) {
        if (lane == 0) {
            sm.m[w][h] = m[h];
            sm.l[w][h] = l[h];
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) sm.acc[w][h][lane * 4 + u] = acc[h][u];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < kGroup * kHeadDim; i += blockDim.x) {
        const int h = i / kHeadDim, d = i % kHeadDim;
        float mm = -INFINITY;
        for (int ww = 0; ww < kWarps; ++ww) mm = fmaxf(mm, sm.m[ww][h]);
        float ll = 0.f, aa = 0.f;
        for (int ww = 0; ww < kWarps; ++ww) {
            if (sm.m[ww][h] == -INFINITY) continue;
            const float f = expf(sm.m[ww][h] - mm);
            ll = fmaf(sm.l[ww][h], f, ll);
            aa = fmaf(sm.acc[ww][h][d], f, aa);
        }
        const size_t slot = ((size_t) r * kHeads + g * kGroup + h) * splits + s;
        part_acc[slot * kHeadDim + d] = aa;
        if (d == 0) {
            part_ml[slot * 2] = mm;
            part_ml[slot * 2 + 1] = ll;
        }
    }
}

__global__ void attn_combine_kernel(const float* __restrict__ part_ml, const float* __restrict__ part_acc,
                                    const float* __restrict__ sink, int splits, float* __restrict__ out) {
    const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const size_t base = ((size_t) r * kHeads + h) * splits;
    const float t = sink[(size_t) r * kHeads + h];
    float mm = t;
    for (int s = 0; s < splits; ++s) mm = fmaxf(mm, part_ml[(base + s) * 2]);
    float ll = expf(t - mm), aa = 0.f;
    for (int s = 0; s < splits; ++s) {
        const float ms = part_ml[(base + s) * 2];
        if (ms == -INFINITY) continue;
        const float f = expf(ms - mm);
        ll = fmaf(part_ml[(base + s) * 2 + 1], f, ll);
        aa = fmaf(part_acc[(base + s) * kHeadDim + d], f, aa);
    }
    out[((size_t) r * kHeads + h) * kHeadDim + d] = aa / ll;
}

int splits_for(int64_t max_keys) {
    const int64_t s = (max_keys + 511) / 512;
    return (int) std::min<int64_t>(kMaxSplits, std::max<int64_t>(1, s));
}

}  // namespace

size_t attention_decode_scratch(int rows, int64_t max_keys) {
    const size_t slots = (size_t) rows * kHeads * (size_t) splits_for(max_keys);
    return slots * (2 + kHeadDim) * sizeof(float);
}

void attention_decode(const AttnArgs& a, int rows, int64_t max_keys, void* scratch, float* out, void* stream) {
    if (rows <= 0) return;
    cudaStream_t s = (cudaStream_t) stream;
    const int splits = splits_for(max_keys);
    float* part_ml = static_cast<float*>(scratch);
    float* part_acc = part_ml + (size_t) rows * kHeads * splits * 2;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(attn_decode_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(Smem));
        attr = true;
    }
    attn_decode_kernel<<<dim3((unsigned) rows, kKvHeads, (unsigned) splits), kWarps * 32, sizeof(Smem), s>>>(a, splits, part_ml,
                                                                                                          part_acc);
    check("attention_decode");
    attn_combine_kernel<<<dim3((unsigned) rows, kHeads), kHeadDim, 0, s>>>(part_ml, part_acc, a.sink, splits, out);
    check("attention_decode/combine");
}

}  // namespace basalt::models::inkling2
