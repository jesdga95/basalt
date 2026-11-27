// src/models/inkling2/kernels/attn.cu - Inkling 2's decode attention: a query head's scores take a per-query relative bias
// over the last kExtent positions, and a per-query sink logit sits in the softmax's denominator.
//
// A row's keys split in slices whose size comes from the row's own key count alone (whole tiles of the block's four warps, at
// most the card's slice count for an INT8 cache, slices.hpp, at most kMaxSplits for the others), so a row's bits do not depend
// on the other rows of its call or on the launch's bound. A block takes one row, one key/value head (its three query heads)
// and one slice; each warp walks its tiles of 32 keys with a lane a key: the lane loads and scores its key in registers,
// stages V in shared memory, then sums four output channels. For deep INT8 reads, a block instead shares its slice's K/V tiles
// across consecutive rows of the same conversation. Both paths merge a row's slices with its sink, gate the heads and round to
// the output projection's q8_1 blocks.
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/q8_1_finite.hpp"
#include "basalt/log.hpp"
#include "basalt/models/inkling2/kernels.hpp"
#include "attn_mma.cuh"
#include "prep.cuh"
#include "slices.hpp"

#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace basalt::models::inkling2 {
namespace {

namespace bk = basalt::kernels;

constexpr int kWarps = 4;
constexpr int kTile = 32;
constexpr int kUnit = kWarps * kTile;        // keys of one pass of the block's warps
constexpr int kGroup = kHeads / kKvHeads;    // 3 query heads a key/value head
// the most slices a row takes (scratch, tables); a card's own count (slices.hpp, MmaDevice::slices) is at most this
constexpr int kMaxSplits = 64;
constexpr int kWindowMmaSplits = 32;
constexpr int kMmaMinKeys = kMmaDeepMinKeys;
constexpr bool kMmaSingleton = true;  // a deep row uses the same arithmetic alone and in a verify window
constexpr int kVStride = kHeadDim / 2 + 4;   // words a value row takes in shared memory (the lanes' stores do not collide)

struct Q81 {
    half2 ds;
    int8_t qs[32];
};

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

// The card's slice count for INT8 caches (slices.hpp), set once when its attention plans are probed (mma_device). FP16 and
// E4M3 caches keep kMaxSplits.
__constant__ int c_slices = kMaxSplits;

// The keys a slice of a row with n keys covers: whole passes of the block, at most `slices` slices.
__host__ __device__ __forceinline__ int span_of(int n, int slices) {
    const int units = (n + kUnit * slices - 1) / (kUnit * slices);
    return (units > 1 ? units : 1) * kUnit;
}

__device__ __forceinline__ void row_keys(const AttnArgs& a, int r, int& p, int& lo, int& n) {
    p = a.pos[r];
    lo = a.window > 0 ? max(0, p - a.window + 1) : 0;
    n = p + 1 - lo;
}

__device__ __forceinline__ int wrap_cap(int key, int cap, bool pow2) {
    if (key < cap) return key;
    return pow2 ? (key & (cap - 1)) : (key % cap);
}

__device__ __forceinline__ void attn_cp16(void* dst, const void* src, bool ok) {
    const unsigned sa = (unsigned) __cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::"r"(sa), "l"(src), "r"(ok ? 16 : 0) : "memory");
}

// 8 cache values as fp32 from 16 bytes of fp16 (or the first 8 of 16 bytes of E4M3, `hi` picking the other 8)
__device__ __forceinline__ void unpack8_f16(const uint4 u, float (&f)[8]) {
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 t = __half22float2(*reinterpret_cast<const __half2*>(&w[i]));
        f[2 * i] = t.x;
        f[2 * i + 1] = t.y;
    }
}
__device__ __forceinline__ void unpack16_fp8(const uint4 u, float (&f)[16]) {
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        __nv_fp8x2_e4m3 lo, hi;
        lo.__x = (__nv_fp8x2_storage_t) (w[i] & 0xFFFF);
        hi.__x = (__nv_fp8x2_storage_t) (w[i] >> 16);
        const float2 a = (float2) lo, b = (float2) hi;
        f[4 * i] = a.x;
        f[4 * i + 1] = a.y;
        f[4 * i + 2] = b.x;
        f[4 * i + 3] = b.y;
    }
}
// 16 INT8 codes times their fp16 scale as fp32 (exact: a code times an fp16 scale fits fp32's mantissa). A code's float is
// its byte, biased by 128, placed under 2^23's exponent, less 2^23 + 128: no integer conversion.
__device__ __forceinline__ void unpack16_int8(const uint4 u, float scale, float (&f)[16]) {
    const uint32_t w[4] = {u.x ^ 0x80808080u, u.y ^ 0x80808080u, u.z ^ 0x80808080u, u.w ^ 0x80808080u};
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            f[4 * i + j] = (__uint_as_float(__byte_perm(w[i], 0x4b000000u, 0x7650u | j)) - 8388736.f) * scale;
}
// A code's exact half is its biased byte under 1024's exponent, less 1152. When scaled, the fp16 product has the same
// bits as the exact fp32 product rounded to fp16.
template <bool Scaled = true>
__device__ __forceinline__ void unpack16_int8_h2(const uint4 u, __half2 scale, uint32_t (&h)[8]) {
    const uint32_t w[4] = {u.x ^ 0x80808080u, u.y ^ 0x80808080u, u.z ^ 0x80808080u, u.w ^ 0x80808080u};
    const __half2 bias = __half2half2(__ushort_as_half(0x6480u));   // 1152
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const uint32_t b = __byte_perm(w[i], 0x64646464u, j ? 0x7362u : 0x5140u);
            __half2 v = __hsub2(*reinterpret_cast<const __half2*>(&b), bias);
            if constexpr (Scaled) v = __hmul2(v, scale);
            h[2 * i + j] = *reinterpret_cast<const uint32_t*>(&v);
        }
}
// A row's two fp16 scales (after its kHeadDim codes) as fp32.
__device__ __forceinline__ void row_scales(const uint8_t* row, float (&s)[2]) {
    const __half2 h = *reinterpret_cast<const __half2*>(row + kHeadDim);
    s[0] = __low2float(h);
    s[1] = __high2float(h);
}

struct Smem {
    float q[kGroup][kHeadDim];
    float rs[kGroup][kRel];
    uint32_t v[kWarps][kTile][kVStride];   // a tile's value rows as fp16 pairs
    float m[kWarps][kGroup], l[kWarps][kGroup];
    float acc[kWarps][kGroup][kHeadDim];
};

// Tensor-core MMA helpers (same m16n8k16 / ldmatrix PTX as src/kernels/cuda/gqa_attn.cu); attn_mma.cuh holds the shared ones
using attn_mma::ldsm_x2;
using attn_mma::ldsm_x4;
using attn_mma::mma16816;
__device__ __forceinline__ void ldsm_x2_t(unsigned& r0, unsigned& r1, const void* p) {
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];" : "=r"(r0), "=r"(r1) : "r"(a));
}

// Convert 8 E4M3 bytes (packed in uint2) to 8 fp16 values (uint4) via hardware cvt.rn.f16x2.e4m3x2.
__device__ __forceinline__ uint4 fp8x8_to_f16x8(uint2 u) {
    __half2_raw h[4];
    h[0] = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) (u.x & 0xFFFFu), __NV_E4M3);
    h[1] = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) (u.x >> 16), __NV_E4M3);
    h[2] = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) (u.y & 0xFFFFu), __NV_E4M3);
    h[3] = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) (u.y >> 16), __NV_E4M3);
    return *reinterpret_cast<const uint4*>(h);
}

template <bool FP8, bool INT8, bool SkipDeep = false>
__global__ void __launch_bounds__(kWarps * 32, 2) attn_split_kernel(AttnArgs a, int z, float* __restrict__ part_ml,
                                                                 float* __restrict__ part_acc) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
    const int r = blockIdx.x, g = blockIdx.y, s = blockIdx.z;
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    int p, lo, n;
    row_keys(a, r, p, lo, n);
    if constexpr (SkipDeep) {
        if (n >= kMmaMinKeys) return;
    }
    const int span = span_of(n, INT8 ? c_slices : kMaxSplits);
    if (s * span >= n) return;   // past this row's slices
    const int s_lo = lo + s * span, s_hi = min(p + 1, s_lo + span);
    constexpr int kRowBytes = FP8 ? kHeadDim : INT8 ? kKvInt8RowBytes : kHeadDim * 2;
    constexpr int kLoads = (FP8 || INT8) ? kHeadDim / 16 : kHeadDim / 8;
    const bk::WindowSegs& at = seg_table(a.seg);
    const SegCaches& sc = a.seg.c[bk::seg_of(at, r)];   // a batch window's row: its conversation's caches
    const uint8_t* kbase = static_cast<const uint8_t*>(at.n > 0 ? sc.k : a.k_cache) + (size_t) g * kRowBytes;
    const uint8_t* vbase = static_cast<const uint8_t*>(at.n > 0 ? sc.v : a.v_cache) + (size_t) g * kRowBytes;
    const bool cap_pow2 = a.cap > 0 && (a.cap & (a.cap - 1)) == 0;
    uint32_t* vrow = sm.v[w][lane];
    constexpr int kVRegs = (FP8 || INT8) ? kLoads : 1;
    const auto load_k_and_v = [&](int t0, uint4 (&kr)[kLoads], uint4 (&vr)[kVRegs], float (&ks)[2], float (&vs)[2]) {
        const int key = t0 + lane;
        const bool valid = key < s_hi;
        ks[0] = ks[1] = vs[0] = vs[1] = 1.f;
        const size_t cell = valid ? (size_t) wrap_cap(key, a.cap, cap_pow2) * kKvHeads * kRowBytes : 0;
        const uint4* kp = reinterpret_cast<const uint4*>(kbase + cell);
        const uint4* vp = reinterpret_cast<const uint4*>(vbase + cell);
        if (valid) {
#pragma unroll
            for (int i = 0; i < kLoads; ++i) kr[i] = __ldg(kp + i);
            if constexpr (!FP8 && !INT8) {
#pragma unroll
                for (int i = 0; i < kLoads; ++i) attn_cp16(vrow + 4 * i, vp + i, true);
                asm volatile("cp.async.commit_group;" ::: "memory");
            } else {
#pragma unroll
                for (int i = 0; i < kLoads; ++i) vr[i] = __ldg(vp + i);
                if constexpr (INT8) {
                    row_scales(kbase + cell, ks);
                    row_scales(vbase + cell, vs);
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < kLoads; ++i) kr[i] = make_uint4(0, 0, 0, 0);
            if constexpr (!FP8 && !INT8) {
#pragma unroll
                for (int i = 0; i < kLoads; ++i) attn_cp16(vrow + 4 * i, vbase, false);
                asm volatile("cp.async.commit_group;" ::: "memory");
            } else {
#pragma unroll
                for (int i = 0; i < kLoads; ++i) vr[i] = make_uint4(0, 0, 0, 0);
            }
        }
    };
    const int t_first = s_lo + w * kTile;
    uint4 kr[kLoads];
    uint4 vr[kVRegs];
    float ks[2], vs[2];
    if (t_first < s_hi) load_k_and_v(t_first, kr, vr, ks, vs);
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
    for (int t0 = t_first; t0 < s_hi; t0 += kUnit) {
        const int key = t0 + lane;
        const bool valid = key < s_hi;
        if (t0 != t_first) load_k_and_v(t0, kr, vr, ks, vs);
        float sc[kGroup] = {0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < kLoads; ++i) {
            if constexpr (FP8 || INT8) {
                float f[16];
                if constexpr (FP8) unpack16_fp8(kr[i], f);
                else unpack16_int8(kr[i], ks[i / 4], f);
#pragma unroll
                for (int d = 0; d < 16; ++d)
#pragma unroll
                    for (int h = 0; h < kGroup; ++h) sc[h] = fmaf(sm.q[h][16 * i + d], f[d], sc[h]);
            } else {
                float f[8];
                unpack8_f16(kr[i], f);
#pragma unroll
                for (int d = 0; d < 8; ++d)
#pragma unroll
                    for (int h = 0; h < kGroup; ++h) sc[h] = fmaf(sm.q[h][8 * i + d], f[d], sc[h]);
            }
        }
        const int dist = p - key;
        const bool has_rel = valid && dist < kExtent;
        float b_acc[kGroup] = {0.f, 0.f, 0.f};
        if (has_rel) {
#pragma unroll
            for (int i = 0; i < kRel; ++i) {
                const float tv = __ldg(a.rel_table + i * kExtent + dist);
#pragma unroll
                for (int h = 0; h < kGroup; ++h) b_acc[h] = fmaf(sm.rs[h][i], tv, b_acc[h]);
            }
        }
#pragma unroll
        for (int h = 0; h < kGroup; ++h) sc[h] = valid ? sc[h] * a.scale + b_acc[h] : -INFINITY;
        // INT8 and FP8 convert V into sm.v[w][lane]; FP16 V is already streaming via cp.async
        if constexpr (INT8) {
#pragma unroll
            for (int i = 0; i < kLoads; ++i) {
                uint32_t h[8];
                unpack16_int8_h2(vr[i], __float2half2_rn(vs[i / 4]), h);
#pragma unroll
                for (int d = 0; d < 8; ++d) vrow[8 * i + d] = h[d];
            }
        } else if constexpr (FP8) {
#pragma unroll
            for (int i = 0; i < kLoads; ++i) {
                const uint4 u = vr[i];
                const uint32_t ww[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const __half2_raw h0 = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) (ww[j] & 0xFFFFu), __NV_E4M3);
                    const __half2_raw h1 = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) (ww[j] >> 16), __NV_E4M3);
                    vrow[8 * i + 2 * j + 0] = (uint32_t) h0.x | ((uint32_t) h0.y << 16);
                    vrow[8 * i + 2 * j + 1] = (uint32_t) h1.x | ((uint32_t) h1.y << 16);
                }
            }
        }
        // the online softmax over the tile (overlaps with in-flight FP16 V cp.async)
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
        if constexpr (!FP8 && !INT8) asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncwarp();
        for (int j = 0; j < kTile; ++j) {
            const uint2 u = *reinterpret_cast<const uint2*>(&sm.v[w][j][lane * 2]);
            const float2 v01 = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
            const float2 v23 = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
#pragma unroll
            for (int h = 0; h < kGroup; ++h) {
                const float pj = __shfl_sync(0xffffffffu, pr[h], j);
                acc[h][0] = fmaf(pj, v01.x, acc[h][0]);
                acc[h][1] = fmaf(pj, v01.y, acc[h][1]);
                acc[h][2] = fmaf(pj, v23.x, acc[h][2]);
                acc[h][3] = fmaf(pj, v23.y, acc[h][3]);
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
        const size_t slot = ((size_t) r * kHeads + g * kGroup + h) * z + s;
        part_acc[slot * kHeadDim + d] = aa;
        if (d == 0) {
            part_ml[slot * 2] = mm;
            part_ml[slot * 2 + 1] = ll;
        }
    }
}

// Tensor-core shared-KV split kernel for full-attention decode windows (a.window == 0).
// One block handles up to BM query rows (BM=16: up to 5 verify rows; BM=32: up to 10 verify rows) per KV head and split,
// reading K and V once per chunk of 64 cells with coalesced 128-bit loads and pipelined prefetch.
template <bool FP8, int BM, int KH = 64>
__global__ void __launch_bounds__(128, KH == 32 ? (BM == 16 ? 5 : 3) : (BM == 16 ? 3 : 2))
attn_mma_kernel(AttnArgs a, int total_rows, int tiles_per_seg, int z, int z_stride,
                float* __restrict__ part_ml, float* __restrict__ part_acc) {
    constexpr int HD = kHeadDim;
    constexpr int CHUNK = 64;
    constexpr int LD = HD + 8;       // padded fp16 stride: bank-conflict-free ldmatrix
    constexpr int NT = HD / 32;      // 4 tiles of 8 dims per warp
    constexpr int MT = BM / 16;      // 1 or 2 M-tiles of 16 query rows
    constexpr float kLog2e = 1.4426950408889634f;
    constexpr float kLn2 = 0.6931471805599453f;

    union alignas(16) ScoreRow {
        float s[CHUNK + 1];
        __half p[CHUNK + 8];
    };
    __shared__ __align__(16) __half qs[BM][LD];
    __shared__ __align__(16) __half kvs[KH][LD];
    __shared__ ScoreRow sp[BM];
    __shared__ float rs[KH == 64 ? BM : 1][KH == 64 ? kRel : 1];
    __shared__ float rm[BM], rl[BM], ra[BM];
    __shared__ int rpos[BM], rlo[BM];

    bk::pdl_wait();
    bk::pdl_launch_dependents();

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    const int kvh = blockIdx.y, s = blockIdx.z;
    const bk::WindowSegs& at = seg_table(a.seg);
    const bool segmented = at.n > 0;
    const int seg = segmented ? (int) blockIdx.x / tiles_per_seg : 0;
    const int tile = segmented ? (int) blockIdx.x % tiles_per_seg : (int) blockIdx.x;
    if (segmented && seg >= at.n) return;
    const int row0 = segmented ? at.row0[seg] : 0;
    const int rows = (segmented ? at.row0[seg + 1] : total_rows) - row0;
    const int M = rows * kGroup, m0 = tile * BM;
    if (m0 >= M) return;
    const void* k_ptr = segmented ? a.seg.c[seg].k : a.k_cache;
    const void* v_ptr = segmented ? a.seg.c[seg].v : a.v_cache;

    if (tid < BM) {
        const int m = m0 + tid;
        const int p = m < M ? a.pos[row0 + m / kGroup] : -1;
        rpos[tid] = p;
        rlo[tid] = p >= 0 && a.window > 0 ? max(0, p - a.window + 1) : 0;
        rm[tid] = -INFINITY;
        rl[tid] = 0.0f;
    }
    __syncthreads();

    int p_max = -1, lo_min = INT32_MAX;
#pragma unroll
    for (int i = 0; i < BM; ++i) {
        if (rpos[i] >= 0) {
            p_max = max(p_max, rpos[i]);
            lo_min = min(lo_min, rlo[i]);
        }
    }
    if (p_max < 0) return;

    const int n_chunks = (p_max + 1 + CHUNK - 1) / CHUNK;
    const int c_min = a.window > 0 ? (lo_min >> 6) : 0;
    const int rem = a.window > 0 ? (c_min & (kWindowMmaSplits - 1)) : 0;
    const int first_chunk = c_min + (s >= rem ? s - rem : z - rem + s);
    if (first_chunk >= n_chunks) return;

    const float qscale = a.scale * kLog2e;
    for (int idx = tid; idx < BM * HD; idx += 128) {
        const int i = idx / HD, d = idx - i * HD, m = m0 + i;
        const float v = m < M ? a.q[((size_t) (row0 + m / kGroup) * kHeads + (size_t) kvh * kGroup + (m % kGroup)) * HD + d] * qscale : 0.0f;
        qs[i][d] = __float2half_rn(v);
    }
    if constexpr (KH == 64) {
        for (int idx = tid; idx < BM * kRel; idx += 128) {
            const int i = idx / kRel, k = idx - i * kRel, m = m0 + i;
            rs[i][k] = m < M ? a.r[((size_t) (row0 + m / kGroup) * kHeads + (size_t) kvh * kGroup + (m % kGroup)) * kRel + k] * kLog2e : 0.0f;
        }
    }

    constexpr int kRowBytes = FP8 ? HD : HD * 2;
    const uint8_t* kbase = static_cast<const uint8_t*>(k_ptr) + (size_t) kvh * kRowBytes;
    const uint8_t* vbase = static_cast<const uint8_t*>(v_ptr) + (size_t) kvh * kRowBytes;

    constexpr int kElems = FP8 ? 16 : 8;
    constexpr int NI = KH * HD / kElems / 128;
    const uint32_t cap32 = (uint32_t) a.cap;

    float acc[MT][NT][4];
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int n = 0; n < NT; ++n) acc[t][n][0] = acc[t][n][1] = acc[t][n][2] = acc[t][n][3] = 0.0f;

    if constexpr (KH == 64) {
        uint4 pk[NI], pv[NI];
        unsigned pmask = 0;
        const auto fetch = [&](int c0) {
            pmask = 0;
            const uint32_t base_cell = a.window > 0 ? ((uint32_t) c0 % cap32) : (uint32_t) c0;
#pragma unroll
            for (int j = 0; j < NI; ++j) {
                const int idx = tid + j * 128;
                const int cell = idx / (HD / kElems), d0 = (idx - cell * (HD / kElems)) * kElems;
                const int key = c0 + cell;
                if (key >= lo_min && key <= p_max) {
                    uint32_t rkey = base_cell + (uint32_t) cell;
                    if (a.window > 0 && rkey >= cap32) rkey -= cap32;
                    const size_t off = (size_t) rkey * kKvHeads * kRowBytes + (size_t) d0 * (FP8 ? 1 : 2);
                    pk[j] = *reinterpret_cast<const uint4*>(kbase + off);
                    pv[j] = *reinterpret_cast<const uint4*>(vbase + off);
                    pmask |= 1u << j;
                }
            }
        };
        const auto commit_buf = [&](const uint4 (&p)[NI]) {
#pragma unroll
            for (int j = 0; j < NI; ++j) {
                const int idx = tid + j * 128;
                const int cell = idx / (HD / kElems), d0 = (idx - cell * (HD / kElems)) * kElems;
                if constexpr (FP8) {
                    uint4 w0 = make_uint4(0, 0, 0, 0), w1 = make_uint4(0, 0, 0, 0);
                    if ((pmask >> j) & 1u) {
                        w0 = fp8x8_to_f16x8(make_uint2(p[j].x, p[j].y));
                        w1 = fp8x8_to_f16x8(make_uint2(p[j].z, p[j].w));
                    }
                    *reinterpret_cast<uint4*>(&kvs[cell][d0]) = w0;
                    *reinterpret_cast<uint4*>(&kvs[cell][d0 + 8]) = w1;
                } else {
                    const uint4 w = ((pmask >> j) & 1u) ? p[j] : make_uint4(0, 0, 0, 0);
                    *reinterpret_cast<uint4*>(&kvs[cell][d0]) = w;
                }
            }
        };

        fetch(first_chunk * CHUNK);
        __syncthreads();

        for (int chunk = first_chunk; chunk < n_chunks; chunk += z) {
            const int c0 = chunk * CHUNK;
            commit_buf(pk);
            __syncthreads();

            for (int nc = warp; nc < CHUNK / 8; nc += 4) {
#pragma unroll
                for (int t = 0; t < MT; ++t) {
                    float s4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                    for (int k = 0; k < HD; k += 16) {
                        unsigned a0, a1, a2, a3, b0, b1;
                        ldsm_x4(a0, a1, a2, a3, &qs[t * 16 + (lane & 15)][k + (lane >> 4) * 8]);
                        ldsm_x2(b0, b1, &kvs[nc * 8 + (lane & 7)][k + ((lane >> 3) & 1) * 8]);
                        mma16816(s4, a0, a1, a2, a3, b0, b1);
                    }
                    sp[t * 16 + g].s[nc * 8 + 2 * c] = s4[0];
                    sp[t * 16 + g].s[nc * 8 + 2 * c + 1] = s4[1];
                    sp[t * 16 + g + 8].s[nc * 8 + 2 * c] = s4[2];
                    sp[t * 16 + g + 8].s[nc * 8 + 2 * c + 1] = s4[3];
                }
            }
            __syncthreads();

            commit_buf(pv);
            if (chunk + z < n_chunks) fetch((chunk + z) * CHUNK);

            for (int i = warp; i < BM; i += 4) {
                const int p = rpos[i], lo = rlo[i];
                const bool maybe_rel = p >= 0 && (p - c0) < (kExtent + CHUNK);
                float pv_row[CHUNK / 32], mx = -INFINITY;
#pragma unroll
                for (int u = 0; u < CHUNK / 32; ++u) {
                    const int cc = lane + 32 * u;
                    const int key = c0 + cc;
                    float val = -INFINITY;
                    if (p >= 0 && key >= lo && key <= p) {
                        val = sp[i].s[cc];
                        if (maybe_rel) {
                            const int dist = p - key;
                            if (dist < kExtent) {
                                float b = 0.0f;
#pragma unroll
                                for (int ri = 0; ri < kRel; ++ri) b = fmaf(rs[i][ri], a.rel_table[ri * kExtent + dist], b);
                                val += b;
                            }
                        }
                    }
                    pv_row[u] = val;
                    mx = fmaxf(mx, val);
                }
                mx = warp_max(mx);
                const float mold = rm[i], mnew = fmaxf(mold, mx);
                float l = 0.0f;
#pragma unroll
                for (int u = 0; u < CHUNK / 32; ++u) {
                    const float pr = pv_row[u] == -INFINITY ? 0.0f : exp2f(pv_row[u] - mnew);
                    sp[i].p[lane + 32 * u] = __float2half_rn(pr);
                    l += pr;
                }
                l = warp_sum(l);
                if (lane == 0) {
                    const float alpha = mx == -INFINITY ? 1.0f : (mold == -INFINITY ? 0.0f : exp2f(mold - mnew));
                    ra[i] = alpha;
                    if (mx > -INFINITY) {
                        rl[i] = fmaf(rl[i], alpha, l);
                        rm[i] = mnew;
                    }
                }
            }
            __syncthreads();

#pragma unroll
            for (int t = 0; t < MT; ++t) {
                const float ag = ra[t * 16 + g], ag8 = ra[t * 16 + g + 8];
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    acc[t][n][0] *= ag;
                    acc[t][n][1] *= ag;
                    acc[t][n][2] *= ag8;
                    acc[t][n][3] *= ag8;
                }
            }
#pragma unroll
            for (int kk = 0; kk < CHUNK; kk += 16) {
                unsigned b0[NT], b1[NT];
#pragma unroll
                for (int n = 0; n < NT; ++n) ldsm_x2_t(b0[n], b1[n], &kvs[kk + (lane & 15)][warp * (HD / 4) + n * 8]);
#pragma unroll
                for (int t = 0; t < MT; ++t) {
                    unsigned a0, a1, a2, a3;
                    ldsm_x4(a0, a1, a2, a3, &sp[t * 16 + (lane & 15)].p[kk + (lane >> 4) * 8]);
#pragma unroll
                    for (int n = 0; n < NT; ++n) mma16816(acc[t][n], a0, a1, a2, a3, b0[n], b1[n]);
                }
            }
            __syncthreads();
        }
    } else {
        uint4 pbuf[NI];
        unsigned pmask = 0;
        const auto fetch_half = [&](const uint8_t* base, int c0) {
            pmask = 0;
            const uint32_t base_cell = a.window > 0 ? ((uint32_t) c0 % cap32) : (uint32_t) c0;
#pragma unroll
            for (int j = 0; j < NI; ++j) {
                const int idx = tid + j * 128;
                const int cell = idx / (HD / kElems), d0 = (idx - cell * (HD / kElems)) * kElems;
                const int key = c0 + cell;
                if (key >= lo_min && key <= p_max) {
                    uint32_t rkey = base_cell + (uint32_t) cell;
                    if (a.window > 0 && rkey >= cap32) rkey -= cap32;
                    const size_t off = (size_t) rkey * kKvHeads * kRowBytes + (size_t) d0 * (FP8 ? 1 : 2);
                    pbuf[j] = __ldcg(reinterpret_cast<const uint4*>(base + off));
                    pmask |= 1u << j;
                }
            }
        };
        const auto commit_buf = [&]() {
#pragma unroll
            for (int j = 0; j < NI; ++j) {
                const int idx = tid + j * 128;
                const int cell = idx / (HD / kElems), d0 = (idx - cell * (HD / kElems)) * kElems;
                if constexpr (FP8) {
                    uint4 w0 = make_uint4(0, 0, 0, 0), w1 = make_uint4(0, 0, 0, 0);
                    if ((pmask >> j) & 1u) {
                        w0 = fp8x8_to_f16x8(make_uint2(pbuf[j].x, pbuf[j].y));
                        w1 = fp8x8_to_f16x8(make_uint2(pbuf[j].z, pbuf[j].w));
                    }
                    *reinterpret_cast<uint4*>(&kvs[cell][d0]) = w0;
                    *reinterpret_cast<uint4*>(&kvs[cell][d0 + 8]) = w1;
                } else {
                    const uint4 w = ((pmask >> j) & 1u) ? pbuf[j] : make_uint4(0, 0, 0, 0);
                    *reinterpret_cast<uint4*>(&kvs[cell][d0]) = w;
                }
            }
        };
        const auto qk_half = [&](int col0) {
            const int nc = warp;
#pragma unroll
            for (int t = 0; t < MT; ++t) {
                float s4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                for (int k = 0; k < HD; k += 16) {
                    unsigned a0, a1, a2, a3, b0, b1;
                    ldsm_x4(a0, a1, a2, a3, &qs[t * 16 + (lane & 15)][k + (lane >> 4) * 8]);
                    ldsm_x2(b0, b1, &kvs[nc * 8 + (lane & 7)][k + ((lane >> 3) & 1) * 8]);
                    mma16816(s4, a0, a1, a2, a3, b0, b1);
                }
                sp[t * 16 + g].s[col0 + nc * 8 + 2 * c] = s4[0];
                sp[t * 16 + g].s[col0 + nc * 8 + 2 * c + 1] = s4[1];
                sp[t * 16 + g + 8].s[col0 + nc * 8 + 2 * c] = s4[2];
                sp[t * 16 + g + 8].s[col0 + nc * 8 + 2 * c + 1] = s4[3];
            }
        };
        const auto pv_half = [&](int kk0) {
#pragma unroll
            for (int kk = 0; kk < KH; kk += 16) {
                unsigned b0[NT], b1[NT];
#pragma unroll
                for (int n = 0; n < NT; ++n) ldsm_x2_t(b0[n], b1[n], &kvs[kk + (lane & 15)][warp * (HD / 4) + n * 8]);
#pragma unroll
                for (int t = 0; t < MT; ++t) {
                    unsigned a0, a1, a2, a3;
                    ldsm_x4(a0, a1, a2, a3, &sp[t * 16 + (lane & 15)].p[kk0 + kk + (lane >> 4) * 8]);
#pragma unroll
                    for (int n = 0; n < NT; ++n) mma16816(acc[t][n], a0, a1, a2, a3, b0[n], b1[n]);
                }
            }
        };

        fetch_half(kbase, first_chunk * CHUNK);
        __syncthreads();

        for (int chunk = first_chunk; chunk < n_chunks; chunk += z) {
            const int c0 = chunk * CHUNK;
            commit_buf();
            __syncthreads();
            fetch_half(kbase, c0 + KH);
            qk_half(0);
            __syncthreads();

            commit_buf();
            __syncthreads();
            fetch_half(vbase, c0);
            qk_half(KH);
            __syncthreads();

            commit_buf();
            for (int i = warp; i < BM; i += 4) {
                const int p = rpos[i], lo = rlo[i];
                const bool maybe_rel = p >= 0 && (p - c0) < (kExtent + CHUNK);
                const int m = m0 + i;
                const float* rptr = maybe_rel
                                        ? (a.r + ((size_t) (row0 + m / kGroup) * kHeads + (size_t) kvh * kGroup + (m % kGroup)) * kRel)
                                        : nullptr;
                float pv_row[CHUNK / 32], mx = -INFINITY;
#pragma unroll
                for (int u = 0; u < CHUNK / 32; ++u) {
                    const int cc = lane + 32 * u;
                    const int key = c0 + cc;
                    float val = -INFINITY;
                    if (p >= 0 && key >= lo && key <= p) {
                        val = sp[i].s[cc];
                        if (maybe_rel) {
                            const int dist = p - key;
                            if (dist < kExtent) {
                                float b = 0.0f;
#pragma unroll
                                for (int ri = 0; ri < kRel; ++ri) {
                                    const float rv = __fmul_rn(__ldg(rptr + ri), kLog2e);
                                    b = fmaf(rv, __ldg(a.rel_table + ri * kExtent + dist), b);
                                }
                                val += b;
                            }
                        }
                    }
                    pv_row[u] = val;
                    mx = fmaxf(mx, val);
                }
                mx = warp_max(mx);
                const float mold = rm[i], mnew = fmaxf(mold, mx);
                float l = 0.0f;
#pragma unroll
                for (int u = 0; u < CHUNK / 32; ++u) {
                    const float pr = pv_row[u] == -INFINITY ? 0.0f : exp2f(pv_row[u] - mnew);
                    sp[i].p[lane + 32 * u] = __float2half_rn(pr);
                    l += pr;
                }
                l = warp_sum(l);
                if (lane == 0) {
                    const float alpha = mx == -INFINITY ? 1.0f : (mold == -INFINITY ? 0.0f : exp2f(mold - mnew));
                    ra[i] = alpha;
                    if (mx > -INFINITY) {
                        rl[i] = fmaf(rl[i], alpha, l);
                        rm[i] = mnew;
                    }
                }
            }
            __syncthreads();

            fetch_half(vbase, c0 + KH);
#pragma unroll
            for (int t = 0; t < MT; ++t) {
                const float ag = ra[t * 16 + g], ag8 = ra[t * 16 + g + 8];
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    acc[t][n][0] *= ag;
                    acc[t][n][1] *= ag;
                    acc[t][n][2] *= ag8;
                    acc[t][n][3] *= ag8;
                }
            }
            pv_half(0);
            __syncthreads();

            commit_buf();
            __syncthreads();
            if (chunk + z < n_chunks) fetch_half(kbase, (chunk + z) * CHUNK);
            pv_half(KH);
            __syncthreads();
        }
    }

#pragma unroll
    for (int t = 0; t < MT; ++t) {
#pragma unroll
        for (int hr = 0; hr < 2; ++hr) {
            const int i = t * 16 + g + 8 * hr, m = m0 + i;
            if (m >= M) continue;
            const int r = row0 + m / kGroup, h = m % kGroup;
            const size_t slot = ((size_t) r * kHeads + (size_t) kvh * kGroup + h) * z_stride + s;
            if (warp == 0 && c == 0) {
                part_ml[slot * 2] = rm[i] == -INFINITY ? -INFINITY : rm[i] * kLn2;
                part_ml[slot * 2 + 1] = rl[i];
            }
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                const int d = warp * (HD / 4) + n * 8 + 2 * c;
                const float v0 = rm[i] == -INFINITY ? 0.0f : acc[t][n][2 * hr] + 0.0f;
                const float v1 = rm[i] == -INFINITY ? 0.0f : acc[t][n][2 * hr + 1] + 0.0f;
                part_acc[slot * HD + d] = v0;
                part_acc[slot * HD + d + 1] = v1;
            }
        }
    }
}

// A split streams one tile of keys at a time. Q and the running numerator stay live across tiles.
constexpr int kMmaRows = 16;
constexpr int kMmaLd = kHeadDim + 8;
constexpr int kMmaPanel = kHeadDim / kWarps;
constexpr float kMmaProbabilityScale = 0x1p15f;

struct MmaVTile {
    __half row[kTile][kMmaLd];
};

struct MmaRawTile {
    uint4 code[2][kTile][kHeadDim / 16];
    uint32_t scale[2][kTile];
};

__device__ __forceinline__ void mma_copy16(void* dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::
                 "r"(attn_mma::smem_u32(dst)), "l"(src), "r"(valid ? 16 : 0) : "memory");
}

__device__ __forceinline__ void mma_prefetch(MmaRawTile& raw, const uint8_t* kc, const uint8_t* vc,
                                            int tile, int hi, int g, int64_t cap, int tid) {
    constexpr int pieces = kHeadDim / 16;
    constexpr int keys_per_pass = kWarps * 32 / pieces;
    const int piece = tid % pieces;
    if (tile < hi) {
        const bool whole = tile + kTile <= hi;
        const bool unwrapped = hi <= cap;
#pragma unroll
        for (int j = tid / pieces; j < kTile; j += keys_per_pass) {
            const int key = tile + j;
            const bool valid = whole || key < hi;
            const int64_t cell = !valid ? 0 : unwrapped ? key : key % cap;
            const size_t offset = ((size_t) cell * kKvHeads + g) * kKvInt8RowBytes;
            mma_copy16(&raw.code[0][j][piece], kc + offset + 16 * piece, valid);
            mma_copy16(&raw.code[1][j][piece], vc + offset + 16 * piece, valid);
            if (piece == 0) {
                const auto copy_scale = [valid](uint32_t* dst, const uint8_t* src) {
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::
                                 "r"(attn_mma::smem_u32(dst)), "l"(src), "r"(valid ? 4 : 0) : "memory");
                };
                copy_scale(&raw.scale[0][j], kc + offset + kHeadDim);
                copy_scale(&raw.scale[1][j], vc + offset + kHeadDim);
            }
        }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

__device__ __forceinline__ void mma_wait(int outstanding) {
    if (outstanding == 2) asm volatile("cp.async.wait_group 2;" ::: "memory");
    else if (outstanding == 1) asm volatile("cp.async.wait_group 1;" ::: "memory");
    else asm volatile("cp.async.wait_group 0;" ::: "memory");
}

constexpr int kMmaPieces = kHeadDim / 16;
constexpr int kMmaKeysPerPass = kWarps * 32 / kMmaPieces;
constexpr int kMmaPasses = kTile / kMmaKeysPerPass;
static_assert(kTile % kMmaKeysPerPass == 0);

__device__ __forceinline__ void mma_store_halves(__half* dst, const uint32_t (&halves)[8]) {
    *reinterpret_cast<uint4*>(dst) = make_uint4(halves[0], halves[1], halves[2], halves[3]);
    *reinterpret_cast<uint4*>(dst + 8) = make_uint4(halves[4], halves[5], halves[6], halves[7]);
}

__device__ __forceinline__ void mma_store_key_piece(uint4 code, uint32_t scales, __half* dst, __half2& ksc, int piece) {
    uint32_t halves[8];
    unpack16_int8_h2<false>(code, __half2{}, halves);
    mma_store_halves(dst, halves);
    if (piece == 0) ksc = *reinterpret_cast<const __half2*>(&scales);
}

__device__ __forceinline__ void mma_store_value_piece(uint4 code, unsigned scales, __half* dst, int piece) {
    scales = __shfl_sync(0xffffffffu, scales, 0, kMmaPieces);
    const __half scale = __ushort_as_half((uint16_t) (scales >> (16 * (piece / 4))));
    uint32_t halves[8];
    unpack16_int8_h2(code, __half2half2(scale), halves);
    mma_store_halves(dst, halves);
}

__device__ __forceinline__ void mma_expand(const MmaRawTile& raw, __half (&k)[kTile][kMmaLd], __half2 (&ksc)[kTile],
                                           MmaVTile& v, int tid) {
    const int piece = tid % kMmaPieces;
#pragma unroll
    for (int j = tid / kMmaPieces; j < kTile; j += kMmaKeysPerPass) {
        const uint32_t key_scales = piece == 0 ? raw.scale[0][j] : 0u;
        const uint32_t value_scales = piece == 0 ? raw.scale[1][j] : 0u;
        mma_store_key_piece(raw.code[0][j][piece], key_scales, &k[j][piece * 16], ksc[j], piece);
        mma_store_value_piece(raw.code[1][j][piece], value_scales, &v.row[j][piece * 16], piece);
    }
}

// Each lane holds its own part of the next tile, without a shared raw-copy ring.
struct MmaRegisterTile {
    uint4 code[2][kMmaPasses];
    uint32_t scale[2][kMmaPasses];
};

__device__ __forceinline__ MmaRegisterTile mma_load_register_tile(const uint8_t* kc, const uint8_t* vc,
                                                                 int tile, int hi, int g, int64_t cap, int tid) {
    MmaRegisterTile result{};
    const int piece = tid % kMmaPieces;
    const bool unwrapped = hi <= cap;
#pragma unroll
    for (int pass = 0; pass < kMmaPasses; ++pass) {
        const int key = tile + tid / kMmaPieces + pass * kMmaKeysPerPass;
        if (key < hi) {
            const int64_t cell = unwrapped ? key : key % cap;
            const size_t offset = ((size_t) cell * kKvHeads + g) * kKvInt8RowBytes;
            result.code[0][pass] = *reinterpret_cast<const uint4*>(kc + offset + 16 * piece);
            result.code[1][pass] = *reinterpret_cast<const uint4*>(vc + offset + 16 * piece);
            if (piece == 0) {
                result.scale[0][pass] = *reinterpret_cast<const uint32_t*>(kc + offset + kHeadDim);
                result.scale[1][pass] = *reinterpret_cast<const uint32_t*>(vc + offset + kHeadDim);
            }
        }
    }
    return result;
}

__device__ __forceinline__ void mma_expand(const MmaRegisterTile& raw, __half (&k)[kTile][kMmaLd], __half2 (&ksc)[kTile],
                                           MmaVTile& v, int tid) {
    const int piece = tid % kMmaPieces;
#pragma unroll
    for (int pass = 0; pass < kMmaPasses; ++pass) {
        const int j = tid / kMmaPieces + pass * kMmaKeysPerPass;
        mma_store_key_piece(raw.code[0][pass], raw.scale[0][pass], &k[j][piece * 16], ksc[j], piece);
        mma_store_value_piece(raw.code[1][pass], raw.scale[1][pass], &v.row[j][piece * 16], piece);
    }
}

template <int Slabs>
struct alignas(16) MmaSmem {
    __half k[kTile][kMmaLd];
    MmaVTile v;
    __half2 ksc[kTile];
    float score[Slabs * kMmaRows][kTile + 2];
    __half prob[Slabs * kMmaRows][kTile + 8];
    float m[Slabs * kMmaRows], l[Slabs * kMmaRows], corr[Slabs * kMmaRows];
};

template <int Slabs, bool Direct = false>
__device__ __forceinline__ void attn_mma_slice(const AttnArgs& a, int first, int end, int g, int lo, int hi, int s, int z,
                                               const uint8_t* kc, const uint8_t* vc, MmaSmem<Slabs>& sm,
                                               MmaRawTile* raw, MmaVTile* alternate_v, int stages, bool primed,
                                               float* part_ml, float* part_acc, int tid) {
    using namespace attn_mma;
    constexpr int m_rows = Slabs * kMmaRows;
    constexpr int warps_per_slab = kWarps / Slabs;
    const int active = (end - first) * kGroup;
    const int warp = tid / 32, lane = tid % 32;
    const int lane_row = lane / 4, lane_pair = 2 * (lane % 4);
    if (tid < m_rows) {
        sm.m[tid] = -INFINITY;
        sm.l[tid] = 0.f;
    }
    const int score_m = (warp / warps_per_slab) * kMmaRows;
    constexpr int n_fragments = kTile / (8 * warps_per_slab);
    const int score_n = (warp % warps_per_slab) * n_fragments * 8;
    const int quad = tid / 4, member = tid % 4;
    const int pos = quad < active ? a.pos[first + quad / kGroup] : -1;
    const size_t query_head = (size_t) (first + quad / kGroup) * kHeads + g * kGroup + quad % kGroup;
    const auto query = [&](int m) -> const float* {
        const size_t qh = (size_t) (first + m / kGroup) * kHeads + g * kGroup + m % kGroup;
        return m < active ? a.q + qh * kHeadDim + lane_pair : nullptr;
    };
    const auto pack = [](const float* q, int d) -> unsigned {
        const float2 f = q ? *reinterpret_cast<const float2*>(q + d) : make_float2(0.f, 0.f);
        const __half2 pair = __floats2half2_rn(f.x, f.y);
        return *reinterpret_cast<const unsigned*>(&pair);
    };
    const float* qa = query(score_m + lane_row);
    const float* qb = query(score_m + lane_row + 8);
    MmaRegisterTile next{};
    if constexpr (Direct) {
        next = mma_load_register_tile(kc, vc, lo, hi, g, a.cap, tid);
    } else {
        for (int ahead = primed ? 1 : 0; ahead < stages - 1; ++ahead)
            mma_prefetch(raw[ahead], kc, vc, lo + ahead * kTile, hi, g, a.cap, tid);
    }
    unsigned q_frag[kHeadDim / 16][4];
#pragma unroll
    for (int d = 0; d < kHeadDim / 16; ++d) {
        q_frag[d][0] = pack(qa, d * 16);
        q_frag[d][1] = pack(qb, d * 16);
        q_frag[d][2] = pack(qa, d * 16 + 8);
        q_frag[d][3] = pack(qb, d * 16 + 8);
    }
    float acc[Slabs][kMmaPanel / 8][4] = {};
    int current = 0, future = stages - 1;
    bool odd = false;
    for (int tile = lo; tile < hi; tile += kTile) {
        MmaVTile& v = alternate_v && odd ? *alternate_v : sm.v;
        if constexpr (Direct) {
            const MmaRegisterTile current_tile = next;
            if (tile + kTile < hi) next = mma_load_register_tile(kc, vc, tile + kTile, hi, g, a.cap, tid);
            mma_expand(current_tile, sm.k, sm.ksc, v, tid);
        } else {
            if (!primed || stages > 1 || tile != lo)
                mma_prefetch(raw[future], kc, vc, tile + (stages - 1) * kTile, hi, g, a.cap, tid);
            mma_wait(stages - 1);
            // Each thread consumes its own copies; the value scale's copying lane broadcasts inside its warp.
            mma_expand(raw[current], sm.k, sm.ksc, v, tid);
        }
        __syncthreads();

        // Keep QK partials separate for each key's scale groups; combine them in fp32.
        constexpr int scale_group = kHeadDim / 2;
        float scores[2][n_fragments][4] = {};
#pragma unroll
        for (int d = 0; d < kHeadDim; d += 16) {
            const auto& q = q_frag[d / 16];
#pragma unroll
            for (int n = 0; n < n_fragments; ++n) {
                unsigned b0, b1;
                ldsm_x2(b0, b1, &sm.k[score_n + 8 * n + (lane & 7)][d + ((lane >> 3) & 1) * 8]);
                mma16816(scores[d / scale_group][n], q[0], q[1], q[2], q[3], b0, b1);
            }
        }
#pragma unroll
        for (int n = 0; n < n_fragments; ++n) {
            const int col = score_n + 8 * n + lane_pair;
            const float2 key0 = __half22float2(sm.ksc[col]), key1 = __half22float2(sm.ksc[col + 1]);
            const auto& s0 = scores[0][n];
            const auto& s1 = scores[1][n];
            *reinterpret_cast<float2*>(&sm.score[score_m + lane_row][col]) =
                make_float2(fmaf(s1[0], key0.y, s0[0] * key0.x), fmaf(s1[1], key1.y, s0[1] * key1.x));
            *reinterpret_cast<float2*>(&sm.score[score_m + lane_row + 8][col]) =
                make_float2(fmaf(s1[2], key0.y, s0[2] * key0.x), fmaf(s1[3], key1.y, s0[3] * key1.x));
        }
        __syncthreads();

        // A quad reduces one row. The denominator keeps the full FP32 weights, not the rounded PV operands.
        if (quad < active) {
            const unsigned quad_mask = 0xfu << (lane & ~3);
            float weights[kTile / 4];
            float m = sm.m[quad];
#pragma unroll
            for (int j = 0; j < kTile / 4; ++j) {
                const int key_col = 4 * j + member;
                const int dist = pos - (tile + key_col);
                float score = -INFINITY;
                if (dist >= 0) {
                    float bias = 0.f;
                    if (dist < kExtent) {
#pragma unroll
                        for (int i = 0; i < kRel; ++i)
                            bias = fmaf(a.r[query_head * kRel + i], a.rel_table[i * kExtent + dist], bias);
                    }
                    score = sm.score[quad][key_col] * a.scale + bias;
                }
                weights[j] = score;
                m = fmaxf(m, score);
            }
            m = fmaxf(m, __shfl_xor_sync(quad_mask, m, 1));
            m = fmaxf(m, __shfl_xor_sync(quad_mask, m, 2));
            const float corr = m == -INFINITY ? 1.f : expf(sm.m[quad] - m);
            float sum = 0.f;
#pragma unroll
            for (int j = 0; j < kTile / 4; ++j) {
                const float p = weights[j] == -INFINITY ? 0.f : expf(weights[j] - m);
                sm.prob[quad][4 * j + member] = __float2half_rn(p * kMmaProbabilityScale);
                sum += p;
            }
            sum += __shfl_xor_sync(quad_mask, sum, 1);
            sum += __shfl_xor_sync(quad_mask, sum, 2);
            // All lanes finish reading the old maximum before the first lane changes it.
            __syncwarp(quad_mask);
            if (member == 0) {
                sm.m[quad] = m;
                sm.l[quad] = sm.l[quad] * corr + sum;
                sm.corr[quad] = corr;
            }
        } else if (quad < m_rows) {
#pragma unroll
            for (int j = 0; j < kTile / 4; ++j) sm.prob[quad][4 * j + member] = __float2half_rn(0.f);
            if (member == 0) sm.corr[quad] = 1.f;
        }
        __syncthreads();

        // PV: each warp keeps its channel panel in registers through the whole slice.
#pragma unroll
        for (int slab = 0; slab < Slabs; ++slab) {
            const int m = slab * kMmaRows;
            const float c0 = sm.corr[m + lane_row], c1 = sm.corr[m + lane_row + 8];
#pragma unroll
            for (int n = 0; n < kMmaPanel / 8; ++n) {
                acc[slab][n][0] *= c0;
                acc[slab][n][1] *= c0;
                acc[slab][n][2] *= c1;
                acc[slab][n][3] *= c1;
            }
#pragma unroll
            for (int j = 0; j < kTile; j += 16) {
                unsigned a0, a1, a2, a3;
                ldsm_x4(a0, a1, a2, a3, &sm.prob[m + (lane & 15)][j + (lane >> 4) * 8]);
#pragma unroll
                for (int n = 0; n < kMmaPanel / 8; n += 2) {
                    const int col = warp * kMmaPanel + n * 8;
                    unsigned b0, b1, b2, b3;
                    ldsm_x4_t(b0, b1, b2, b3, &v.row[j + (lane & 15)][col + (lane >> 4) * 8]);
                    mma16816(acc[slab][n], a0, a1, a2, a3, b0, b1);
                    mma16816(acc[slab][n + 1], a0, a1, a2, a3, b2, b3);
                }
            }
        }
        // K is dead after the score barrier. With alternating V buffers the next iteration can
        // stage K/V before the other warps finish PV; the next score barrier protects P and corr.
        if (!alternate_v) __syncthreads();
        odd = !odd;
        if constexpr (!Direct) {
            if (++current == stages) current = 0;
            if (++future == stages) future = 0;
        }
    }
    if constexpr (!Direct) mma_wait(0);

    // Undo the PV operand's scale before the shared split merge sees the numerator.
    constexpr float inv_prob_scale = 1.f / kMmaProbabilityScale;
#pragma unroll
    for (int slab = 0; slab < Slabs; ++slab)
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
            const int m = slab * kMmaRows + lane_row + pair * 8;
            if (m >= active) continue;
            const size_t qh = (size_t) (first + m / kGroup) * kHeads + g * kGroup + m % kGroup;
            const size_t slot = qh * z + s;
#pragma unroll
            for (int n = 0; n < kMmaPanel / 8; ++n) {
                const int col = warp * kMmaPanel + n * 8 + lane_pair;
                *reinterpret_cast<float2*>(part_acc + slot * kHeadDim + col) =
                    make_float2(acc[slab][n][2 * pair] * inv_prob_scale,
                                acc[slab][n][2 * pair + 1] * inv_prob_scale);
            }
            if (warp == 0 && lane_pair == 0) {
                part_ml[slot * 2] = sm.m[m];
                part_ml[slot * 2 + 1] = sm.l[m];
            }
        }
    __syncthreads();
}

template <int MaxSlabs, bool Narrow, bool Direct = false>
__global__ void __launch_bounds__(kWarps * 32, 2) attn_mma_split_kernel(AttnArgs a, int rows, int min_keys, int z,
                                                                       int stages, bool ping_pong, int prime_span,
                                                                       float* part_ml, float* part_acc) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto* extra = smem_raw + sizeof(MmaSmem<MaxSlabs>);
    auto* alternate_v = ping_pong ? reinterpret_cast<MmaVTile*>(extra) : nullptr;
    auto* raw = reinterpret_cast<MmaRawTile*>(extra + (ping_pong ? sizeof(MmaVTile) : 0));
    const int seg = blockIdx.x, g = blockIdx.y;
    const bk::WindowSegs& at = seg_table(a.seg);
    const bool batched = at.n > 0;
    if (batched && seg >= at.n) return;
    const int begin = batched ? at.row0[seg] : 0;
    const int end = batched ? at.row0[seg + 1] : rows;
    const auto* kc = static_cast<const uint8_t*>(batched ? a.seg.c[seg].k : a.k_cache);
    const auto* vc = static_cast<const uint8_t*>(batched ? a.seg.c[seg].v : a.v_cache);
    int primed_tile = -1;
    if constexpr (Narrow && !Direct) {
        // Only the already committed, unwrapped prefix is safe to touch before the prepare wait.
        const int tile = blockIdx.z * prime_span;
        if (prime_span > 0 && tile + kTile < a.min_full_keys && tile + kTile <= a.cap) {
            mma_prefetch(raw[0], kc, vc, tile, tile + kTile, g, a.cap, threadIdx.x);
            primed_tile = tile;
        }
    }
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    if constexpr (Direct) {
        const int n = a.pos[begin] + 1;
        if (n < min_keys) return;
        const int span = span_of(n, c_slices), splits = (n + span - 1) / span;
        auto& sm = *reinterpret_cast<MmaSmem<1>*>(smem_raw);
        for (int s = blockIdx.z; s < splits; s += gridDim.z) {
            const int lo = s * span, hi = min(n, lo + span);
            attn_mma_slice<1, true>(a, begin, begin + 1, g, lo, hi, s, z, kc, vc, sm, nullptr,
                                    alternate_v, 0, false, part_ml, part_acc, threadIdx.x);
        }
    } else {
    for (int s = blockIdx.z; s < z; s += gridDim.z) {
        // Partition at a span or a tail tile boundary so a row takes the same steps in every window.
        for (int first = begin; first < end;) {
            const int n = a.pos[first] + 1, span = span_of(n, c_slices), lo = s * span;
            if (n < min_keys || lo >= n) {
                ++first;
                continue;
            }
            int hi = min(n, lo + span);
            const int tiles = (hi + kTile - 1) / kTile;
            int last = first + 1;
            for (; last < end && (last - first + 1) * kGroup <= MaxSlabs * kMmaRows; ++last) {
                const int next = a.pos[last] + 1, next_hi = min(next, lo + span);
                if (next < min_keys || next <= lo || span_of(next, c_slices) != span || (next_hi + kTile - 1) / kTile != tiles ||
                    (last - first + 1) * kGroup > MaxSlabs * kMmaRows)
                    break;
                hi = max(hi, next_hi);
            }
            const bool primed = primed_tile == lo && hi >= lo + kTile;
            if (primed_tile >= 0 && !primed) mma_wait(0);
            primed_tile = -1;
            if constexpr (MaxSlabs == 2) {
                if ((last - first) * kGroup > kMmaRows) {
                    attn_mma_slice(a, first, last, g, lo, hi, s, z, kc, vc, *reinterpret_cast<MmaSmem<2>*>(smem_raw),
                                   raw, alternate_v, stages, primed, part_ml, part_acc, threadIdx.x);
                    first = last;
                    continue;
                }
            }
            attn_mma_slice(a, first, last, g, lo, hi, s, z, kc, vc, *reinterpret_cast<MmaSmem<1>*>(smem_raw),
                           raw, alternate_v, stages, primed, part_ml, part_acc, threadIdx.x);
            first = last;
        }
    }
    if (primed_tile >= 0) mma_wait(0);
    }
}

// A 32-thread warp block per (row, head, 32-channel Q8_1 block): merges the row's slices with the sink, gates, and
// writes fp32 and/or q8_1 while keeping 96*rows blocks active across SMs with coalesced part_ml reads and 8-way MLP.
__global__ void __launch_bounds__(32) attn_combine_kernel(AttnArgs a, int z, int mma_z, const float* __restrict__ part_ml,
                                                          const float* __restrict__ part_acc, float* __restrict__ out) {
    __shared__ float sm_f[kMaxSplits], sm_l[kMaxSplits];
    const int r = blockIdx.x, h = blockIdx.y >> 2, wb = blockIdx.y & 3, lane = threadIdx.x;
    const int d = wb * 32 + lane;
    const int c = h * kHeadDim + d;
    int p, lo, n;
    row_keys(a, r, p, lo, n);
    const float gmul = a.gate ? (1.f / (1.f + expf(-a.gate[(size_t) r * a.gate_ld + c]))) : 1.f;
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    const size_t base = ((size_t) r * kHeads + h) * z;
    const float t = a.sink[(size_t) r * kHeads + h];
    float mm = t, ll = 0.f, aa = 0.f;
    if (mma_z > 0 && a.window > 0) {
        const int k0 = lo >> 6, k1 = p >> 6, cnt = k1 - k0 + 1;
        float mlane = t;
        for (int i = lane; i < cnt; i += 32) {
            const int s = (k0 + i) & (kWindowMmaSplits - 1);
            const float2 ml = *reinterpret_cast<const float2*>(part_ml + (base + s) * 2);
            sm_f[i] = ml.x;
            sm_l[i] = ml.y;
            mlane = fmaxf(mlane, ml.x);
        }
        mm = warp_max(mlane);
        __syncwarp();
        for (int i = lane; i < cnt; i += 32) sm_f[i] = expf(sm_f[i] - mm);
        __syncwarp();
        ll = expf(t - mm);
        const float* pa = part_acc + base * kHeadDim + d;
        int i = 0;
        for (; i + 4 <= cnt; i += 4) {
            const float a0 = pa[((k0 + i + 0) & (kWindowMmaSplits - 1)) * kHeadDim];
            const float a1 = pa[((k0 + i + 1) & (kWindowMmaSplits - 1)) * kHeadDim];
            const float a2 = pa[((k0 + i + 2) & (kWindowMmaSplits - 1)) * kHeadDim];
            const float a3 = pa[((k0 + i + 3) & (kWindowMmaSplits - 1)) * kHeadDim];
            ll = fmaf(sm_l[i + 0], sm_f[i + 0], ll); aa = fmaf(a0, sm_f[i + 0], aa);
            ll = fmaf(sm_l[i + 1], sm_f[i + 1], ll); aa = fmaf(a1, sm_f[i + 1], aa);
            ll = fmaf(sm_l[i + 2], sm_f[i + 2], ll); aa = fmaf(a2, sm_f[i + 2], aa);
            ll = fmaf(sm_l[i + 3], sm_f[i + 3], ll); aa = fmaf(a3, sm_f[i + 3], aa);
        }
        for (; i < cnt; ++i) {
            const float a0 = pa[((k0 + i) & (kWindowMmaSplits - 1)) * kHeadDim];
            ll = fmaf(sm_l[i], sm_f[i], ll);
            aa = fmaf(a0, sm_f[i], aa);
        }
    } else {
        const int span = span_of(n, a.int8 ? c_slices : kMaxSplits);
        const int splits = mma_z > 0 ? min(mma_z, (n + 63) / 64) : (n + span - 1) / span;
        float mlane = t;
        for (int s = lane; s < splits; s += 32) {
            const float2 ml = *reinterpret_cast<const float2*>(part_ml + (base + s) * 2);
            sm_f[s] = ml.x;
            sm_l[s] = ml.y;
            mlane = fmaxf(mlane, ml.x);
        }
        mm = warp_max(mlane);
        __syncwarp();
        for (int s = lane; s < splits; s += 32) sm_f[s] = expf(sm_f[s] - mm);
        __syncwarp();
        ll = expf(t - mm);
        const float* pa = part_acc + base * kHeadDim + d;
        int s = 0;
        for (; s + 8 <= splits; s += 8) {
            const float a0 = pa[(s + 0) * kHeadDim];
            const float a1 = pa[(s + 1) * kHeadDim];
            const float a2 = pa[(s + 2) * kHeadDim];
            const float a3 = pa[(s + 3) * kHeadDim];
            const float a4 = pa[(s + 4) * kHeadDim];
            const float a5 = pa[(s + 5) * kHeadDim];
            const float a6 = pa[(s + 6) * kHeadDim];
            const float a7 = pa[(s + 7) * kHeadDim];
            ll = fmaf(sm_l[s + 0], sm_f[s + 0], ll); aa = fmaf(a0, sm_f[s + 0], aa);
            ll = fmaf(sm_l[s + 1], sm_f[s + 1], ll); aa = fmaf(a1, sm_f[s + 1], aa);
            ll = fmaf(sm_l[s + 2], sm_f[s + 2], ll); aa = fmaf(a2, sm_f[s + 2], aa);
            ll = fmaf(sm_l[s + 3], sm_f[s + 3], ll); aa = fmaf(a3, sm_f[s + 3], aa);
            ll = fmaf(sm_l[s + 4], sm_f[s + 4], ll); aa = fmaf(a4, sm_f[s + 4], aa);
            ll = fmaf(sm_l[s + 5], sm_f[s + 5], ll); aa = fmaf(a5, sm_f[s + 5], aa);
            ll = fmaf(sm_l[s + 6], sm_f[s + 6], ll); aa = fmaf(a6, sm_f[s + 6], aa);
            ll = fmaf(sm_l[s + 7], sm_f[s + 7], ll); aa = fmaf(a7, sm_f[s + 7], aa);
        }
        for (; s < splits; ++s) {
            const float a0 = pa[s * kHeadDim];
            ll = fmaf(sm_l[s], sm_f[s], ll);
            aa = fmaf(a0, sm_f[s], aa);
        }
    }
    float o = aa / ll;
    if (a.gate) o *= gmul;
    if (out) out[(size_t) r * kHeads * kHeadDim + c] = o;
    if (a.q8) {
        const float amax = warp_max(fabsf(o));
        const float sum = warp_sum(o);
        const float dq = bk::q8_1_finite(amax / 127.0f);
        Q81& b = reinterpret_cast<Q81*>(a.q8)[(size_t) r * (kHeads * kHeadDim / 32) + c / 32];
        b.qs[lane] = bk::q8_1_quant(o, dq, amax);
        if (lane == 0) b.ds = bk::q8_1_ds(dq, sum);
    }
}

// ---- the fused window attention ----------------------------------------------------------------------------------------
//
// A window of up to kFusedRows rows whose keys split in at most kClusterSplits slices of kUnit keys (every sliding-window
// visit; a full visit up to 1024 keys) runs its preparation, its slices and their merge in one kernel: a cluster of
// kClusterSplits blocks per (row, key/value head), block s taking slice s with one tile of kTile keys a warp. Before its
// wait a block reads what no kernel of the window writes: its tile's keys and values of the positions before the window
// (the cache holds them from earlier windows) and their relative-bias rows. It prepares what it reads (prep.cuh) in place
// of prepare_kernel: its three query heads' queries, relative states and sinks, and the keys and values of the window's
// rows whose positions fall in its slice, rounded as the cache stores them and read from its own shared memory; the block
// whose slice holds its own row's position writes that row's into the cache and the rings. The slice then runs as
// attn_split_kernel, and blocks 0..2 of the cluster merge one head each from the slices' shared memory as
// attn_combine_kernel does: the same arithmetic in the same order as the three kernels. A longer window keeps the three:
// each block would prepare up to all of its rows, and a cluster holds its idle slices' blocks.
constexpr int kClusterSplits = 8;
constexpr int kFusedRows = 3;

// ROWS: the window rows a block may prepare (2 for windows of 1-2 rows, 3 for 3)
template <int ROWS>
struct FusedSmem {
    Smem s;
    alignas(16) uint8_t kw[ROWS][kHeadDim * 2];   // the window's rows' keys in the cache's format (fp16 or E4M3)
    alignas(16) uint8_t vw[ROWS][kHeadDim * 2];
    float pm[kGroup], pl[kGroup];                 // the slice's merged maximum and sum of each head
    float pacc[kGroup][kHeadDim];
};

template <bool FP8, bool INT8, int ROWS>
__global__ void __cluster_dims__(1, 1, kClusterSplits) __launch_bounds__(kWarps * 32) attn_fused_kernel(AttnPrep pa, AttnArgs a) {
    namespace cg = cooperative_groups;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    FusedSmem<ROWS>& fs = *reinterpret_cast<FusedSmem<ROWS>*>(smem_raw);
    Smem& sm = fs.s;
    const int r = blockIdx.x, g = blockIdx.y, s = blockIdx.z;
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    constexpr int kRowBytes = FP8 ? kHeadDim : INT8 ? kKvInt8RowBytes : kHeadDim * 2;
    constexpr int kLoads = (FP8 || INT8) ? kHeadDim / 16 : kHeadDim / 8;
    // before the wait: the positions (staged before the window), the slice, the warp's tile and what of it is older than
    // the window, and the weights
    int p, lo, n;
    row_keys(a, r, p, lo, n);
    const int span = span_of(n, INT8 ? c_slices : kMaxSplits), splits = (n + span - 1) / span;
    const bool active = s < splits;
    const int s_lo = lo + s * span, s_hi = min(p + 1, s_lo + span);
    const int p0 = a.pos[0];   // the window's first position: keys from it on are the window's rows (consecutive positions)
    const int key = s_lo + w * kTile + lane;
    const bool valid = active && key < s_hi;
    const uint8_t* kbase = static_cast<const uint8_t*>(a.k_cache) + (size_t) g * kRowBytes;
    const uint8_t* vbase = static_cast<const uint8_t*>(a.v_cache) + (size_t) g * kRowBytes;
    const bool cap_pow2 = a.cap > 0 && (a.cap & (a.cap - 1)) == 0;
    constexpr int kVRegs = (FP8 || INT8) ? kLoads : 1;
    uint4 kr[kLoads];
    uint4 vr[kVRegs];
    float ks[2] = {1.f, 1.f}, vs[2] = {1.f, 1.f};
    const bool old_kv = valid && key < p0;
    if (old_kv) {
        const size_t cell = (size_t) wrap_cap(key, a.cap, cap_pow2) * kKvHeads * kRowBytes;
        const uint4* kp = reinterpret_cast<const uint4*>(kbase + cell);
        const uint4* vp = reinterpret_cast<const uint4*>(vbase + cell);
#pragma unroll
        for (int i = 0; i < kLoads; ++i) kr[i] = __ldg(kp + i);
        if constexpr (!FP8 && !INT8) {
#pragma unroll
            for (int i = 0; i < kLoads; ++i) attn_cp16(sm.v[w][lane] + 4 * i, vp + i, true);
            asm volatile("cp.async.commit_group;" ::: "memory");
        } else {
#pragma unroll
            for (int i = 0; i < kLoads; ++i) vr[i] = __ldg(vp + i);
            if constexpr (INT8) {
                row_scales(kbase + cell, ks);
                row_scales(vbase + cell, vs);
            }
        }
    } else {
#pragma unroll
        for (int i = 0; i < kLoads; ++i) kr[i] = make_uint4(0, 0, 0, 0);
        if constexpr (!FP8 && !INT8) {
#pragma unroll
            for (int i = 0; i < kLoads; ++i) attn_cp16(sm.v[w][lane] + 4 * i, vbase, false);
            asm volatile("cp.async.commit_group;" ::: "memory");
        } else {
#pragma unroll
            for (int i = 0; i < kLoads; ++i) vr[i] = make_uint4(0, 0, 0, 0);
        }
    }
    const int dist = p - key;
    float tv[kRel];
#pragma unroll
    for (int i = 0; i < kRel; ++i) tv[i] = valid && dist < kExtent ? a.rel_table[i * kExtent + dist] : 0.f;
    float wt[4][kConv];
    prep::load_weights(pa, g, w, lane, wt);
    bk::pdl_wait();
    // the merging blocks' sink and gate, read early
    float sink = 0.f, gate = 0.f;
    if (s < kGroup) {
        const int h = g * kGroup + s;
        sink = pa.yq[(size_t) r * pa.ldq + kOffSink + h];   // prepare_kernel's sink: the projection as it is
        if (a.gate) gate = a.gate[(size_t) r * a.gate_ld + h * kHeadDim + threadIdx.x];
    }
    if (active) {
        if (w >= 2) {
            float s3[kGroup];
            prep::q_heads(pa, r, p, g, w, lane, wt, &sm.q[0][0], &sm.rs[0][0], s3);
        } else {
            // the window's rows of this slice up to the block's own: both prepared from one round of loads
            const int rr0 = max(0, s_lo - p0), nr = min(r, s_hi - 1 - p0) - rr0 + 1;
            if (nr > 0) {
                float val[ROWS][4], raw[ROWS][4], f[ROWS], scale[ROWS][2] = {};
                prep::kv_rows<ROWS>(pa, rr0, nr, p0 + rr0, g, w, lane, wt, val, raw, f, scale);
#pragma unroll
                for (int k = 0; k < ROWS; ++k) {
                    if (k >= nr) break;
                    const int rr = rr0 + k;
                    uint8_t* row = w ? fs.vw[rr] : fs.kw[rr];
#pragma unroll
                    for (int u = 0; u < 4; ++u) prep::store_kv(row, FP8, INT8, (size_t) lane * 4 + u, val[k][u] * f[k], scale[k], lane);
                    if (rr == r) {   // the block's own row: into the cache and the ring
                        void* cache = w ? pa.v_cache : pa.k_cache;
                        const size_t row_bytes = (size_t) kRowBytes;
                        void* cache_row = static_cast<uint8_t*>(cache) + ((size_t) (p % pa.cap) * kKvHeads + g) * row_bytes;
#pragma unroll
                        for (int u = 0; u < 4; ++u) prep::store_kv(cache_row, FP8, INT8, (size_t) lane * 4 + u, val[k][u] * f[k], scale[k], lane);
                        float* ring = w ? pa.v_ring : pa.k_ring;
                        *reinterpret_cast<float4*>(ring + (size_t) (p % kConvRing) * (kKvHeads * kHeadDim) + g * kHeadDim + lane * 4) =
                            make_float4(raw[k][0], raw[k][1], raw[k][2], raw[k][3]);
                    }
                }
            }
        }
    }
    __syncthreads();
    // the dependents in only now: post_attn and the output projection read their weights before their waits, and
    // started at the wait they put those reads in front of the preparation's (2-3x its time in the window)
    bk::pdl_launch_dependents();
    if (active) {
        if constexpr (!FP8 && !INT8) asm volatile("cp.async.wait_group 0;" ::: "memory");
        if (valid && key >= p0) {   // a row of this window, as prepared above
            const uint4* kp = reinterpret_cast<const uint4*>(fs.kw[key - p0]);
            const uint4* vp = reinterpret_cast<const uint4*>(fs.vw[key - p0]);
#pragma unroll
            for (int i = 0; i < kLoads; ++i) kr[i] = kp[i];
            if constexpr (!FP8 && !INT8) {
#pragma unroll
                for (int i = 0; i < kLoads; ++i) *reinterpret_cast<uint4*>(sm.v[w][lane] + 4 * i) = vp[i];
            } else {
#pragma unroll
                for (int i = 0; i < kLoads; ++i) vr[i] = vp[i];
                if constexpr (INT8) {
                    row_scales(fs.kw[key - p0], ks);
                    row_scales(fs.vw[key - p0], vs);
                }
            }
        }
        float m[kGroup], l[kGroup], acc[kGroup][4];
#pragma unroll
        for (int h = 0; h < kGroup; ++h) {
            m[h] = -INFINITY;
            l[h] = 0.f;
#pragma unroll
            for (int u = 0; u < 4; ++u) acc[h][u] = 0.f;
        }
        if (s_lo + w * kTile < s_hi) {   // the warp's tile (attn_split_kernel's loop body, once: a slice is kUnit keys)
            float sc[kGroup] = {0.f, 0.f, 0.f};
#pragma unroll
            for (int i = 0; i < kLoads; ++i) {
                if constexpr (FP8 || INT8) {
                    float f[16];
                    if constexpr (FP8) unpack16_fp8(kr[i], f);
                    else unpack16_int8(kr[i], ks[i / 4], f);
#pragma unroll
                    for (int d = 0; d < 16; ++d)
#pragma unroll
                        for (int h = 0; h < kGroup; ++h) sc[h] = fmaf(sm.q[h][16 * i + d], f[d], sc[h]);
                } else {
                    float f[8];
                    unpack8_f16(kr[i], f);
#pragma unroll
                    for (int d = 0; d < 8; ++d)
#pragma unroll
                        for (int h = 0; h < kGroup; ++h) sc[h] = fmaf(sm.q[h][8 * i + d], f[d], sc[h]);
                }
            }
#pragma unroll
            for (int h = 0; h < kGroup; ++h) {
                float b = 0.f;
#pragma unroll
                for (int i = 0; i < kRel; ++i) b = fmaf(sm.rs[h][i], tv[i], b);
                sc[h] = valid ? sc[h] * a.scale + b : -INFINITY;
            }
            uint32_t* vrow = sm.v[w][lane];
            if constexpr (INT8) {
#pragma unroll
                for (int i = 0; i < kLoads; ++i) {
                    uint32_t h[8];
                    unpack16_int8_h2(vr[i], __float2half2_rn(vs[i / 4]), h);
#pragma unroll
                    for (int d = 0; d < 8; ++d) vrow[8 * i + d] = h[d];
                }
            } else if constexpr (FP8) {
#pragma unroll
                for (int i = 0; i < kLoads; ++i) {
                    float f[16];
                    unpack16_fp8(vr[i], f);
#pragma unroll
                    for (int d = 0; d < 8; ++d) {
                        const __half2 hv = __floats2half2_rn(f[2 * d], f[2 * d + 1]);
                        vrow[8 * i + d] = *reinterpret_cast<const uint32_t*>(&hv);
                    }
                }
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
            __syncwarp();
            for (int j = 0; j < kTile; ++j) {
                const uint2 u = *reinterpret_cast<const uint2*>(&sm.v[w][j][lane * 2]);
                const float2 v01 = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
                const float2 v23 = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
#pragma unroll
                for (int h = 0; h < kGroup; ++h) {
                    const float pj = __shfl_sync(0xffffffffu, pr[h], j);
                    acc[h][0] = fmaf(pj, v01.x, acc[h][0]);
                    acc[h][1] = fmaf(pj, v01.y, acc[h][1]);
                    acc[h][2] = fmaf(pj, v23.x, acc[h][2]);
                    acc[h][3] = fmaf(pj, v23.y, acc[h][3]);
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
            fs.pacc[h][d] = aa;
            if (d == 0) {
                fs.pm[h] = mm;
                fs.pl[h] = ll;
            }
        }
    }
    // the slices' shared memory to the merging blocks: a release and an acquire of shared memory only (a cluster-wide
    // release of every state space is a GPU-scope fence that waits for the cache and ring stores; no block of the cluster
    // reads those)
    asm volatile("fence.release.sync_restrict::shared::cta.cluster;\n\tbarrier.cluster.arrive.relaxed.aligned;\n\t"
                 "barrier.cluster.wait.aligned;\n\tfence.acquire.sync_restrict::shared::cluster.cluster;" ::: "memory");
    if (s < kGroup) {   // head g * 3 + s: the row's slices merged with the sink, gated, rounded to q8_1 (a warp a block of 32)
        const int h = s, d = threadIdx.x;
        cg::cluster_group cluster = cg::this_cluster();
        // every slice's maximum, sum and value read at once, then merged in slice order
        float pm[kClusterSplits], pl[kClusterSplits], pv[kClusterSplits];
#pragma unroll
        for (int q = 0; q < kClusterSplits; ++q)
            if (q < splits) {
                const FusedSmem<ROWS>* part = cluster.map_shared_rank(&fs, q);
                pm[q] = part->pm[h];
                pl[q] = part->pl[h];
                pv[q] = part->pacc[h][d];
            }
        float mm = sink;
#pragma unroll
        for (int q = 0; q < kClusterSplits; ++q)
            if (q < splits) mm = fmaxf(mm, pm[q]);
        float ll = expf(sink - mm), aa = 0.f;
#pragma unroll
        for (int q = 0; q < kClusterSplits; ++q)
            if (q < splits) {
                const float f = expf(pm[q] - mm);
                ll = fmaf(pl[q], f, ll);
                aa = fmaf(pv[q], f, aa);
            }
        float o = aa / ll;
        const int c = (g * kGroup + h) * kHeadDim + d;
        if (a.gate) o *= 1.f / (1.f + expf(-gate));
        const float amax = warp_max(fabsf(o));
        const float sum = warp_sum(o);
        const float dq = bk::q8_1_finite(amax / 127.0f);
        Q81& b = reinterpret_cast<Q81*>(a.q8)[(size_t) r * (kHeads * kHeadDim / 32) + c / 32];
        b.qs[c % 32] = bk::q8_1_quant(o, dq, amax);
        if ((c & 31) == 0) b.ds = bk::q8_1_ds(dq, sum);
    }
    // no block leaves while the merging blocks still read its slice: an arrival without release semantics (a release
    // here is a GPU-scope fence that waits for the q8_1 stores to be acknowledged); the merging blocks' remote loads
    // are complete, their values are in the stores above
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n\tbarrier.cluster.wait.aligned;" ::: "memory");
}

int launch_splits(int64_t max_keys, int slices) {
    return (int) std::min<int64_t>(slices, std::max<int64_t>(kWindowMmaSplits, (max_keys + 63) / 64));
}

}  // namespace

int opt_in_device() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= kOptInDevices) throw std::runtime_error("inkling2: device out of range");
    return dev;
}

size_t attention_decode_scratch(int rows, int64_t max_keys) {
    const size_t slots = (size_t) rows * kHeads * (size_t) launch_splits(max_keys, kMaxSplits);   // any card's count
    return slots * (2 + kHeadDim) * sizeof(float);
}

namespace {

struct MmaLaunch {
    size_t shared = 0;
    int stages = 0, resident = 0;
    bool ping_pong = false;
};

struct MmaDevice {
    int sms = 0, l2 = 0;
    int slices = kMaxSplits;   // an INT8 row's most slices on this card (slices.hpp), the same for every plan and reader
    MmaLaunch plans[3][2];  // narrow, one slab, two slabs; L2-resident/streaming
    MmaLaunch two_stage_ring, three_stage_ring, one_stage_ping_pong;
    MmaLaunch singleton;
};

void check_mma_probe(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(std::string("inkling2 attention device probe: ") + cudaGetErrorString(error));
}

int mma_attribute(int dev, cudaDeviceAttr attr) {
    int value = 0;
    check_mma_probe(cudaDeviceGetAttribute(&value, attr, dev));
    return value;
}

template <int Slabs, bool Narrow>
void mma_probe_plans(MmaDevice& device, int per_block, int per_sm, int reserved) {
    const auto kernel = attn_mma_split_kernel<Slabs, Narrow>;
    check_mma_probe(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, per_block));
    if constexpr (!Narrow)
        check_mma_probe(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
    const size_t compact_budget = (size_t) std::max(0, std::min(per_block / 2, per_sm / 2 - reserved));
    auto& plans = device.plans[Narrow ? 0 : Slabs];
    const auto rank = [compact_budget](const MmaLaunch& plan) {
        // Wide windows take two copy stages at equal residency (a third is no faster), then deeper pipelines.
        if constexpr (Slabs == 2) return std::make_tuple(plan.resident, plan.stages == 2, plan.stages, plan.ping_pong);
        else return std::make_tuple(!Narrow || plan.shared <= compact_budget, plan.resident, plan.ping_pong, plan.stages);
    };
    for (const bool ping_pong : {true, false}) {
        for (int stages = 1; stages <= 3; ++stages) {
            const size_t shared = sizeof(MmaSmem<Slabs>) + (ping_pong ? sizeof(MmaVTile) : 0) + stages * sizeof(MmaRawTile);
            if (shared > (size_t) per_block) continue;
            int resident = 0;
            check_mma_probe(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, kWarps * 32, shared));
            resident = std::min(resident, per_sm / ((int) shared + reserved));
            if constexpr (Slabs != 2) resident = std::min(resident, 2);
            if (resident == 0) continue;
            const MmaLaunch candidate{shared, stages, resident, ping_pong};
            if constexpr (Slabs == 2) {
                if (!ping_pong && stages == 2) device.two_stage_ring = candidate;
                if (!ping_pong && stages == 3) device.three_stage_ring = candidate;
                if (ping_pong && stages == 1) device.one_stage_ping_pong = candidate;
            }
            for (int streaming = stages == 3; streaming < 2; ++streaming) {
                auto& best = plans[streaming];
                if (best.resident == 0 || rank(candidate) > rank(best)) best = candidate;
            }
        }
    }
    if (plans[0].resident == 0 || plans[1].resident == 0)
        throw std::runtime_error("inkling2: no resident attention MMA block");
}

MmaDevice& probed_device(int dev) {
    static MmaDevice devices[kOptInDevices];
    static std::once_flag probed[kOptInDevices];
    std::call_once(probed[dev], [dev] {
        MmaDevice& d = devices[dev];
        d.sms = mma_attribute(dev, cudaDevAttrMultiProcessorCount);
        d.l2 = mma_attribute(dev, cudaDevAttrL2CacheSize);
        const int per_block = mma_attribute(dev, cudaDevAttrMaxSharedMemoryPerBlockOptin);
        const int per_sm = mma_attribute(dev, cudaDevAttrMaxSharedMemoryPerMultiprocessor);
        const int reserved = mma_attribute(dev, cudaDevAttrReservedSharedMemoryPerBlock);
        mma_probe_plans<1, true>(d, per_block, per_sm, reserved);
        mma_probe_plans<1, false>(d, per_block, per_sm, reserved);
        mma_probe_plans<2, false>(d, per_block, per_sm, reserved);
        constexpr int shared = sizeof(MmaSmem<1>) + sizeof(MmaVTile);
        if (shared <= per_block) {
            int resident = 0;
            check_mma_probe(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &resident, attn_mma_split_kernel<1, true, true>, kWarps * 32, shared));
            resident = std::min({resident, per_sm / (shared + reserved), 3});
            d.singleton = {shared, 0, resident, true};
        }
        const auto log_plan = [&](const char* geometry, const char* regime, const MmaLaunch& plan) {
            const char* name = plan.stages == 0 ? "direct" : plan.ping_pong ? "ping-pong-ring" : "ring";
            basalt::log::info("basalt inkling2", "INT8 MMA CUDA%d %s/%s plan=%s stages=%d shared=%zuB resident=%d CTAs/SM slots=%d",
                              dev, geometry, regime, name, plan.stages, plan.shared, plan.resident, d.sms * plan.resident);
        };
        // one slice count for every plan, so a row's slices do not depend on its window's width
        std::vector<int> resident;
        for (const MmaLaunch& plan : {d.plans[0][1], d.plans[1][1], d.plans[2][1], d.singleton})
            if (plan.resident > 0 && std::find(resident.begin(), resident.end(), plan.resident) == resident.end())
                resident.push_back(plan.resident);
        std::sort(resident.begin(), resident.end());
        d.slices = slice_count(d.sms, resident, kKvHeads, kWindowMmaSplits, kMaxSplits);
        check_mma_probe(cudaMemcpyToSymbol(c_slices, &d.slices, sizeof(int)));
        std::string per;
        for (const int r : resident) per += (per.empty() ? "" : "/") + std::to_string(r);
        basalt::log::info("basalt inkling2", "attention slices %d (SMs %d, resident %s)", d.slices, d.sms, per.c_str());
        if (d.singleton.resident > 0) log_plan("singleton", "all", d.singleton);
        const char* geometries[] = {"narrow", "one-slab", "two-slab"};
        for (int g = 0; g < 3; ++g) {
            log_plan(geometries[g], "L2", d.plans[g][0]);
            log_plan(geometries[g], "streaming", d.plans[g][1]);
        }
    });
    return devices[dev];
}

const MmaDevice& mma_device(int dev) { return probed_device(dev); }

// A row's slices as the shard model sees them: their count and the last one's share of a full slice's keys.
struct SliceShape {
    int splits = 0;
    double last = 0.;
};

SliceShape slice_shape(int64_t n, int slices) {
    const int span = span_of((int) n, slices), splits = (int) ((n + span - 1) / span);
    return {splits, (double) (n - (int64_t) (splits - 1) * span) / span};
}

// Slice times until the launch's last block ends in a heuristic model of it: equal slots, no per-block setup, a full
// slice taking one unit. Shard z's blocks (a head and segment each) take the row's slices z, z + shards, ...; a shard past
// the row's slices ends at once. Blocks start in launch order on the slots that free first.
double mma_makespan(int shards, SliceShape row, int blocks_per_shard, int slots) {
    struct Run {
        double end;
        int slots;
    };
    const auto by_end = [](const Run& x, const Run& y) { return x.end < y.end; };
    // The slots by the time they free, earliest first. A shard leaves at most one run partly taken, so the runs stay
    // within one more than the shards.
    Run free[kMaxSplits + 2];
    int runs = 1;
    free[0] = {0., slots};
    double end = 0.;
    for (int z = 0; z < std::min({shards, row.splits, kMaxSplits}); ++z) {
        const int behind = row.splits - 1 - z;   // the row's slices from this shard's first to its last
        const double work = behind / shards + (behind % shards == 0 ? row.last : 1.);
        for (int left = blocks_per_shard; left > 0;) {
            const Run start = free[0];
            const int k = std::min(left, start.slots);
            left -= k;
            if (k == start.slots) {
                std::copy(free + 1, free + runs, free);
                --runs;
            } else {
                free[0].slots -= k;
            }
            const Run done{start.end + work, k};
            end = std::max(end, done.end);
            Run* at = std::upper_bound(free, free + runs, done, by_end);
            std::copy_backward(at, free + runs, free + runs + 1);
            *at = done;
            ++runs;
        }
    }
    return end;
}

// The fewest slices a row of `fewest` keys or more can have, below a larger slice size: within one slice size the count
// grows with the keys, and the first count of each larger size grows with it.
int least_splits(int64_t fewest, int slices) {
    const int64_t size = (int64_t) kUnit * slices;
    return std::min(slice_shape(fewest, slices).splits, slice_shape((fewest + size - 1) / size * size + 1, slices).splits);
}

// The shard count for rows of `fewest` to `widest` keys. Another count replaces `previous` only where the model rates it
// no slower on every shape the rows can take and faster on one; where the model cannot tell, `previous` stays. The model
// is a heuristic: the device can rank two counts it ties, or even orders, the other way.
int mma_shards(int z, int64_t fewest, int64_t widest, int blocks_per_shard, int slots, int previous, int slices) {
    // Rows of one slice size take the widest row's shape. A range across slice sizes (a padded launch bound, the graph
    // bucket's, where only the bound is known) holds rows of any slice count from its fewest up to z, each taken as full.
    SliceShape shapes[kMaxSplits];
    int count = 0;
    if (span_of((int) fewest, slices) == span_of((int) widest, slices)) {
        shapes[count++] = slice_shape(widest, slices);
    } else {
        for (int splits = least_splits(fewest, slices); splits <= z; ++splits) shapes[count++] = {splits, 1.};
    }
    constexpr double kEps = 1e-6;
    double base[kMaxSplits];
    double best_total = 0.;
    for (int i = 0; i < count; ++i) best_total += base[i] = mma_makespan(previous, shapes[i], blocks_per_shard, slots);
    int best = previous;
    for (int shards = 1; shards <= z; ++shards) {
        if (z % shards != 0 || shards == previous) continue;
        bool no_slower = true, faster = false;
        double total = 0.;
        for (int i = 0; i < count && no_slower; ++i) {
            const double t = mma_makespan(shards, shapes[i], blocks_per_shard, slots);
            no_slower = t <= base[i] + kEps;
            faster |= t < base[i] - kEps;
            total += t;
        }
        // Of the counts that beat `previous`, the shortest total time, then the fewer blocks.
        if (no_slower && faster && (total < best_total - kEps || (total <= best_total + kEps && shards < best))) {
            best = shards;
            best_total = total;
        }
    }
    return best;
}

// mma_shards for a launch: computed once per distinct input and then only looked up. Rows of one slice size are keyed by
// the widest row, a range across slice sizes by its fewest slice count (with z, the shapes it spans).
int mma_shards_cached(int z, int64_t fewest, int64_t widest, int blocks_per_shard, int slots, int previous, int slices) {
    struct Entry {
        int z, blocks, slots, previous, slices;
        int64_t shape;
        int shards;
    };
    static std::mutex lock;
    static Entry entries[64];
    static int used = 0, next = 0;
    const int64_t shape =
        span_of((int) fewest, slices) == span_of((int) widest, slices) ? widest : -(int64_t) least_splits(fewest, slices);
    std::lock_guard<std::mutex> hold(lock);
    for (int i = 0; i < used; ++i) {
        const Entry& e = entries[i];
        if (e.z == z && e.blocks == blocks_per_shard && e.slots == slots && e.previous == previous && e.slices == slices &&
            e.shape == shape)
            return e.shards;
    }
    const int shards = mma_shards(z, fewest, widest, blocks_per_shard, slots, previous, slices);
    entries[next] = {z, blocks_per_shard, slots, previous, slices, shape, shards};
    next = (next + 1) % 64;
    used = std::max(used, next == 0 ? 64 : next);
    return shards;
}

template <int Slabs, bool Narrow = false, bool Direct = false>
void launch_mma(const AttnArgs& a, int rows, int min_keys, int64_t max_keys, int z, float* part_ml,
                float* part_acc, cudaStream_t stream, const MmaDevice& device, const MmaLaunch* forced,
                Int8MmaPlanInfo* info, Int8MmaShards shard_rule, int shard_count) {
    const int segments = a.seg.at.n > 0 ? a.seg.at.n : (a.seg.at_d ? (a.seg.n_segs > 0 ? a.seg.n_segs : std::min(rows, (int) bk::kWindowSegsMax)) : 1);
    const int64_t bytes_per_segment = max_keys * (2 * kKvHeads * kKvInt8RowBytes);
    const auto& plan = forced ? *forced : Direct ? device.singleton :
                       device.plans[Narrow ? 0 : Slabs][bytes_per_segment > device.l2 / segments];
    const int slots = device.sms * plan.resident;
    const int blocks_per_shard = segments * kKvHeads;
    // The halving rule: fewer, longer shards while the launch still fills its floor of slots.
    int previous = z;
    const int wave_floor = slots * (Narrow && !Direct ? 2 : 1);
    while (previous % 2 == 0 && previous / 2 * blocks_per_shard >= wave_floor) previous /= 2;
    const int64_t fewest = std::min(max_keys, std::max({a.min_full_keys, (int64_t) min_keys, (int64_t) 1}));
    const int shards = shard_count > 0 ? std::min(shard_count, z) : shard_rule == Int8MmaShards::Previous ? previous :
                       mma_shards_cached(z, fewest, max_keys, blocks_per_shard, slots, previous, device.slices);
    if (info) {
        const SliceShape widest = slice_shape(max_keys, device.slices);
        *info = {plan.shared, plan.stages, plan.resident, plan.ping_pong, shards, shards * blocks_per_shard, slots,
                 std::min(shards, widest.splits) * blocks_per_shard, mma_makespan(shards, widest, blocks_per_shard, slots)};
    }
    const auto& pdl = bk::pdl_state();
    const int prime_span = Narrow && !Direct && pdl.on && pdl.plain == 0 && a.seg.at.n == 0 && a.seg.at_d == nullptr &&
                           a.min_full_keys >= min_keys && max_keys <= a.cap ? span_of((int) a.min_full_keys, device.slices) : 0;
    const dim3 grid(segments, kKvHeads, shards);
    bk::launch_pdl(attn_mma_split_kernel<Slabs, Narrow, Direct>, grid, dim3(kWarps * 32), plan.shared, stream,
                   a, rows, min_keys, z, plan.stages, plan.ping_pong, prime_span, part_ml, part_acc);
}

}  // namespace

void attention_decode_int8_init() { (void) mma_device(opt_in_device()); }

// Only full-cache INT8 has an MMA reader. Its cutoff is a property of the row, not the padded launch bound.
static bool mma_decode(const AttnArgs& a, int rows, int64_t max_keys) {
    const int limit = (a.seg.at.n > 0 || a.seg.at_d != nullptr) ? kBatchRows : kDecodeRows;
    return a.int8 && a.window == 0 && rows <= limit && (kMmaSingleton || rows >= 2) && max_keys >= kMmaMinKeys;
}

static bool launch_attention_decode(const AttnArgs& a, int rows, int64_t max_keys, void* scratch, float* out, void* stream,
                                    bool mma, int mma_min_keys, bool scalar_prefix, Int8MmaPlan plan = Int8MmaPlan::Auto,
                                    Int8MmaPlanInfo* info = nullptr, Int8MmaShards shards = Int8MmaShards::Auto,
                                    int shard_count = 0) {
    if (rows <= 0) return true;
    cudaStream_t s = (cudaStream_t) stream;
    const int z = launch_splits(max_keys, a.int8 ? mma_device(opt_in_device()).slices : kMaxSplits);
    float* part_ml = static_cast<float*>(scratch);
    float* part_acc = part_ml + (size_t) rows * kHeads * z * 2;
    static bool attr[kOptInDevices] = {};   // the opt-in is per device: a layer split launches these on two cards
    const int dev = opt_in_device();
    if (bool& set = attr[dev]; !set) {
        cudaFuncSetAttribute(attn_split_kernel<false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(Smem));
        cudaFuncSetAttribute(attn_split_kernel<true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(Smem));
        cudaFuncSetAttribute(attn_split_kernel<false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(Smem));
        cudaFuncSetAttribute(attn_split_kernel<false, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(Smem));
        set = true;

    }
    const bool seg_mode = a.seg.at.n > 0 || a.seg.at_d != nullptr;
    const bool use_mma = !a.int8;
    const int mma_z = a.window > 0 ? kWindowMmaSplits : z;
    if (use_mma) {
        int nseg = 1, max_seg_rows = rows;
        if (a.seg.at.n > 0) {
            nseg = a.seg.at.n;
            max_seg_rows = 1;
            for (int k = 0; k < nseg; ++k) max_seg_rows = std::max(max_seg_rows, a.seg.at.row0[k + 1] - a.seg.at.row0[k]);
        } else if (a.seg.at_d != nullptr) {
            nseg = a.seg.n_segs > 0 ? a.seg.n_segs : std::min(rows, (int) bk::kWindowSegsMax);
            max_seg_rows = a.seg.max_seg_rows > 0
                               ? a.seg.max_seg_rows
                               : (a.seg.n_segs > 0 ? std::min(kDecodeRows, std::max(1, rows - (a.seg.n_segs - 1)))
                                                   : std::min(rows, kDecodeRows));
        }
        static int sms_of[kOptInDevices] = {};
        if (!sms_of[dev]) cudaDeviceGetAttribute(&sms_of[dev], cudaDevAttrMultiProcessorCount, dev);
        const int bm = max_seg_rows * kGroup > 16 ? 32 : 16;
        const int tiles_per_seg = (max_seg_rows * kGroup + bm - 1) / bm;
        const int slots64 = sms_of[dev] * (bm == 16 ? 3 : 2);
        const int64_t act_splits = std::min<int64_t>(mma_z, std::max<int64_t>(1, (max_keys + 63) / 64));
        const bool half_kv = (a.window == 0) && ((int64_t) (nseg * tiles_per_seg) * kKvHeads * act_splits > slots64) &&
                             max_keys > 4096;
        const dim3 grid((unsigned) (nseg * tiles_per_seg), kKvHeads, (unsigned) mma_z);
        if (bm == 32) {
            if (half_kv) {
                if (a.fp8) bk::launch_pdl(attn_mma_kernel<true, 32, 32>, grid, dim3(128), 0, s, a, rows, tiles_per_seg, mma_z, z, part_ml, part_acc);
                else bk::launch_pdl(attn_mma_kernel<false, 32, 32>, grid, dim3(128), 0, s, a, rows, tiles_per_seg, mma_z, z, part_ml, part_acc);
            } else {
                if (a.fp8) bk::launch_pdl(attn_mma_kernel<true, 32, 64>, grid, dim3(128), 0, s, a, rows, tiles_per_seg, mma_z, z, part_ml, part_acc);
                else bk::launch_pdl(attn_mma_kernel<false, 32, 64>, grid, dim3(128), 0, s, a, rows, tiles_per_seg, mma_z, z, part_ml, part_acc);
            }
        } else {
            if (half_kv) {
                if (a.fp8) bk::launch_pdl(attn_mma_kernel<true, 16, 32>, grid, dim3(128), 0, s, a, rows, tiles_per_seg, mma_z, z, part_ml, part_acc);
                else bk::launch_pdl(attn_mma_kernel<false, 16, 32>, grid, dim3(128), 0, s, a, rows, tiles_per_seg, mma_z, z, part_ml, part_acc);
            } else {
                if (a.fp8) bk::launch_pdl(attn_mma_kernel<true, 16, 64>, grid, dim3(128), 0, s, a, rows, tiles_per_seg, mma_z, z, part_ml, part_acc);
                else bk::launch_pdl(attn_mma_kernel<false, 16, 64>, grid, dim3(128), 0, s, a, rows, tiles_per_seg, mma_z, z, part_ml, part_acc);
            }
        }
    } else {
        const dim3 grid((unsigned) rows, kKvHeads, (unsigned) z);
        if (mma) {
            if (scalar_prefix) {
                bk::launch_pdl(attn_split_kernel<false, true, true>, grid, dim3(kWarps * 32), sizeof(Smem), s, a, z, part_ml, part_acc);
                check("attention_decode/shallow");
            }
            const MmaDevice& device = mma_device(dev);
            if (plan == Int8MmaPlan::Direct && device.singleton.resident == 0) return false;
            if (!seg_mode && rows == 1 && plan != Int8MmaPlan::NarrowRing && device.singleton.resident > 0)
                launch_mma<1, true, true>(a, rows, mma_min_keys, max_keys, z, part_ml, part_acc, s, device, nullptr, info, shards,
                                          shard_count);
            else if (rows <= 2)
                launch_mma<1, true>(a, rows, mma_min_keys, max_keys, z, part_ml, part_acc, s, device, nullptr, info, shards,
                                    shard_count);
            else if (rows * kGroup <= kMmaRows)
                launch_mma<1>(a, rows, mma_min_keys, max_keys, z, part_ml, part_acc, s, device, nullptr, info, shards,
                              shard_count);
            else {
                const MmaLaunch* forced = nullptr;
                if (plan == Int8MmaPlan::TwoStageRing) forced = &device.two_stage_ring;
                if (plan == Int8MmaPlan::ThreeStageRing) forced = &device.three_stage_ring;
                if (plan == Int8MmaPlan::OneStagePingPong) forced = &device.one_stage_ping_pong;
                if (forced && forced->resident == 0) return false;
                launch_mma<2>(a, rows, mma_min_keys, max_keys, z, part_ml, part_acc, s, device, forced, info, shards, shard_count);
            }
        } else if (a.int8) bk::launch_pdl(attn_split_kernel<false, true>, grid, dim3(kWarps * 32), sizeof(Smem), s, a, z, part_ml, part_acc);
        else if (a.fp8) bk::launch_pdl(attn_split_kernel<true, false>, grid, dim3(kWarps * 32), sizeof(Smem), s, a, z, part_ml, part_acc);
        else bk::launch_pdl(attn_split_kernel<false, false>, grid, dim3(kWarps * 32), sizeof(Smem), s, a, z, part_ml, part_acc);
    }
    check("attention_decode");
    bk::launch_pdl(attn_combine_kernel, dim3((unsigned) rows, kHeads * (kHeadDim / 32)), dim3(32), 0, s, a, z, use_mma ? mma_z : 0, (const float*) part_ml, (const float*) part_acc, out);
    check("attention_decode/combine");
    return true;
}

void attention_decode(const AttnArgs& a, int rows, int64_t max_keys, void* scratch, float* out, void* stream) {
    launch_attention_decode(a, rows, max_keys, scratch, out, stream, mma_decode(a, rows, max_keys), kMmaMinKeys,
                            a.min_full_keys < kMmaMinKeys);
}

bool attention_decode_int8_test(const AttnArgs& a, int rows, int64_t max_keys, void* scratch, float* out, void* stream,
                                bool mma, Int8MmaPlan plan, Int8MmaPlanInfo* info, Int8MmaShards shards, int shard_count) {
    const int limit = (a.seg.at.n > 0 || a.seg.at_d != nullptr) ? kBatchRows : kDecodeRows;
    if (!a.int8 || a.window != 0 || rows < 1 || rows > limit)
        throw std::invalid_argument("inkling2: test needs a full-cache INT8 decode window");
    if (plan != Int8MmaPlan::Auto) {
        const bool singleton_plan = plan == Int8MmaPlan::NarrowRing || plan == Int8MmaPlan::Direct;
        if (!mma || (singleton_plan ? rows != 1 : rows * kGroup <= kMmaRows))
            throw std::invalid_argument("inkling2: forced MMA plan does not match the test window");
    }
    if (!mma && (shards != Int8MmaShards::Auto || shard_count != 0))
        throw std::invalid_argument("inkling2: shard override needs the MMA reader");
    if (shard_count < 0) throw std::invalid_argument("inkling2: negative shard count");
    return launch_attention_decode(a, rows, max_keys, scratch, out, stream, mma, 1, false, plan, info, shards, shard_count);
}

int attention_decode_int8_test_min_keys() { return kMmaMinKeys; }

int attention_decode_int8_slices() { return mma_device(opt_in_device()).slices; }

void attention_decode_int8_test_set_slices(int slices) {
    if (slices < kWindowMmaSplits || slices > kMaxSplits) throw std::invalid_argument("inkling2: test slice count out of range");
    MmaDevice& d = probed_device(opt_in_device());
    d.slices = slices;
    check_mma_probe(cudaMemcpyToSymbol(c_slices, &d.slices, sizeof(int)));
}

void attention_window(const AttnPrep& pa, const AttnArgs& a, int rows, int64_t max_keys, void* scratch, void* stream) {
    if (rows <= 0) return;
    const int fused_splits = (int) std::max<int64_t>(1, (max_keys + kUnit - 1) / kUnit);
    if (!a.int8 || rows > kFusedRows || fused_splits > kClusterSplits || a.q8 == nullptr || a.seg.at.n > 0 ||
        a.seg.at_d != nullptr || mma_decode(a, rows, max_keys)) {
        decode_prepare(pa, rows, stream);
        attention_decode(a, rows, max_keys, scratch, nullptr, stream);
        return;
    }
    // 1-2 rows, a block an SM: each block asks for more than half of an SM's shared memory (it uses sizeof(FusedSmem)), so
    // a cluster's blocks spread over as many SMs instead of sharing them two by two through the preparation and the tile,
    // while the products' small blocks still fit beside one; 3 rows, two blocks an SM, so the 3 x 8 x 8 blocks take one wave
    static size_t smem_of[kOptInDevices] = {};   // per device, as above
    const int dev = opt_in_device();
    (void) mma_device(dev);   // the card's slice count
    size_t& smem = smem_of[dev];
    if (!smem) {
        int per_sm = 0, reserved = 0;
        cudaDeviceGetAttribute(&per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
        cudaDeviceGetAttribute(&reserved, cudaDevAttrReservedSharedMemoryPerBlock, dev);
        smem = std::max(sizeof(FusedSmem<2>), (size_t) std::max(0, per_sm / 2 - reserved + 128));
        cudaFuncSetAttribute(attn_fused_kernel<false, false, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
        cudaFuncSetAttribute(attn_fused_kernel<true, false, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
        cudaFuncSetAttribute(attn_fused_kernel<false, true, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
        cudaFuncSetAttribute(attn_fused_kernel<false, false, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(FusedSmem<3>));
        cudaFuncSetAttribute(attn_fused_kernel<true, false, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(FusedSmem<3>));
        cudaFuncSetAttribute(attn_fused_kernel<false, true, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(FusedSmem<3>));
    }
    const dim3 grid((unsigned) rows, kKvHeads, kClusterSplits);
    cudaStream_t s = (cudaStream_t) stream;
    if (rows <= 2) {
        if (a.int8) bk::launch_pdl(attn_fused_kernel<false, true, 2>, grid, dim3(kWarps * 32), smem, s, pa, a);
        else if (a.fp8) bk::launch_pdl(attn_fused_kernel<true, false, 2>, grid, dim3(kWarps * 32), smem, s, pa, a);
        else bk::launch_pdl(attn_fused_kernel<false, false, 2>, grid, dim3(kWarps * 32), smem, s, pa, a);
    } else {
        if (a.int8) bk::launch_pdl(attn_fused_kernel<false, true, 3>, grid, dim3(kWarps * 32), sizeof(FusedSmem<3>), s, pa, a);
        else if (a.fp8) bk::launch_pdl(attn_fused_kernel<true, false, 3>, grid, dim3(kWarps * 32), sizeof(FusedSmem<3>), s, pa, a);
        else bk::launch_pdl(attn_fused_kernel<false, false, 3>, grid, dim3(kWarps * 32), sizeof(FusedSmem<3>), s, pa, a);
    }
    check("attention_window");
}

}  // namespace basalt::models::inkling2
