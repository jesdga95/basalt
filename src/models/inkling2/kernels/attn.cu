// src/models/inkling2/kernels/attn.cu - Inkling 2's decode attention: a query head's scores take a per-query relative bias
// over the last kExtent positions, and a per-query sink logit sits in the softmax's denominator.
//
// A row's keys split in slices whose size comes from the row's own key count alone (whole tiles of the block's four warps,
// at most kMaxSplits slices), so a row's bits do not depend on the other rows of its call or on the launch's bound. A block
// takes one row, one key/value head (its three query heads) and one slice; each warp walks its tiles of 32 keys with a lane
// a key: the lane loads and scores its key in registers, stages V in shared memory, then sums four output channels.
// For deep INT8 reads, a block instead shares its slice's K/V tiles across consecutive rows of the same conversation.
// Both paths merge a row's slices with its sink, gate the heads and round to the output projection's q8_1 blocks.
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/q8_1_finite.hpp"
#include "basalt/models/inkling2/kernels.hpp"
#include "attn_mma.cuh"
#include "prep.cuh"

#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>

namespace basalt::models::inkling2 {
namespace {

namespace bk = basalt::kernels;

constexpr int kWarps = 4;
constexpr int kTile = 32;
constexpr int kUnit = kWarps * kTile;        // keys of one pass of the block's warps
constexpr int kGroup = kHeads / kKvHeads;    // 3 query heads a key/value head
constexpr int kMaxSplits = 64;
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

// The keys a slice of a row with n keys covers: whole passes of the block, at most kMaxSplits slices.
__host__ __device__ __forceinline__ int span_of(int n) {
    const int units = (n + kUnit * kMaxSplits - 1) / (kUnit * kMaxSplits);
    return (units > 1 ? units : 1) * kUnit;
}

__device__ __forceinline__ void row_keys(const AttnArgs& a, int r, int& p, int& lo, int& n) {
    p = a.pos[r];
    lo = a.window > 0 ? max(0, p - a.window + 1) : 0;
    n = p + 1 - lo;
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
// 16 INT8 codes times their fp16 scale, rounded to fp16 pairs: the same bits as the fp32 product rounded (the product of a
// code and an fp16 scale is exact before the one rounding). A code's half is its biased byte under 1024's exponent, less 1152.
__device__ __forceinline__ void unpack16_int8_h2(const uint4 u, __half2 scale, uint32_t (&h)[8]) {
    const uint32_t w[4] = {u.x ^ 0x80808080u, u.y ^ 0x80808080u, u.z ^ 0x80808080u, u.w ^ 0x80808080u};
    const __half2 bias = __half2half2(__ushort_as_half(0x6480u));   // 1152
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const uint32_t b = __byte_perm(w[i], 0x64646464u, j ? 0x7362u : 0x5140u);
            const __half2 v = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&b), bias), scale);
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

template <bool FP8, bool INT8, bool SkipDeep = false>
__global__ void __launch_bounds__(kWarps * 32) attn_split_kernel(AttnArgs a, int z, float* __restrict__ part_ml,
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
    const int span = span_of(n);
    if (s * span >= n) return;   // past this row's slices
    const int s_lo = lo + s * span, s_hi = min(p + 1, s_lo + span);
    constexpr int kRowBytes = FP8 ? kHeadDim : INT8 ? kKvInt8RowBytes : kHeadDim * 2;
    constexpr int kLoads = (FP8 || INT8) ? kHeadDim / 16 : kHeadDim / 8;
    const SegCaches& sc = a.seg.c[bk::seg_of(a.seg.at, r)];   // a batch window's row: its conversation's caches
    const uint8_t* kbase = static_cast<const uint8_t*>(a.seg.at.n > 0 ? sc.k : a.k_cache) + (size_t) g * kRowBytes;
    const uint8_t* vbase = static_cast<const uint8_t*>(a.seg.at.n > 0 ? sc.v : a.v_cache) + (size_t) g * kRowBytes;
    // a tile's key and value rows, the lane's key's (zero past the slice); the warp's first tile is loaded with the queries
    const auto load_tile = [&](int t0, uint4 (&kr)[kLoads], uint4 (&vr)[kLoads], float (&ks)[2], float (&vs)[2]) {
        const int key = t0 + lane;
        ks[0] = ks[1] = vs[0] = vs[1] = 1.f;
        if (key < s_hi) {
            const size_t cell = (size_t) (key % a.cap) * kKvHeads * kRowBytes;
            const uint4* kp = reinterpret_cast<const uint4*>(kbase + cell);
            const uint4* vp = reinterpret_cast<const uint4*>(vbase + cell);
#pragma unroll
            for (int i = 0; i < kLoads; ++i) kr[i] = kp[i];
#pragma unroll
            for (int i = 0; i < kLoads; ++i) vr[i] = vp[i];
            if constexpr (INT8) {
                row_scales(kbase + cell, ks);
                row_scales(vbase + cell, vs);
            }
        } else {
#pragma unroll
            for (int i = 0; i < kLoads; ++i) kr[i] = vr[i] = make_uint4(0, 0, 0, 0);
        }
    };
    const int t_first = s_lo + w * kTile;
    uint4 kr[kLoads], vr[kLoads];
    float ks[2], vs[2];
    if (t_first < s_hi) load_tile(t_first, kr, vr, ks, vs);
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
        if (t0 != t_first) load_tile(t0, kr, vr, ks, vs);
        // the lane's key scored for the three heads
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
        float tv[kRel];
#pragma unroll
        for (int i = 0; i < kRel; ++i) tv[i] = valid && dist < kExtent ? a.rel_table[i * kExtent + dist] : 0.f;
#pragma unroll
        for (int h = 0; h < kGroup; ++h) {
            float b = 0.f;
#pragma unroll
            for (int i = 0; i < kRel; ++i) b = fmaf(sm.rs[h][i], tv[i], b);
            sc[h] = valid ? sc[h] * a.scale + b : -INFINITY;
        }
        // the lane's value row into shared memory as fp16 pairs (E4M3 is exact in fp16)
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
        } else {
#pragma unroll
            for (int i = 0; i < kLoads; ++i) *reinterpret_cast<uint4*>(vrow + 4 * i) = vr[i];
        }
        // the online softmax over the tile
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
        const size_t slot = ((size_t) r * kHeads + g * kGroup + h) * z + s;
        part_acc[slot * kHeadDim + d] = aa;
        if (d == 0) {
            part_ml[slot * 2] = mm;
            part_ml[slot * 2 + 1] = ll;
        }
    }
}

// A split streams one tile of keys at a time. Q and the running numerator stay live across tiles.
constexpr int kMmaRows = 16;
constexpr int kMmaLd = kHeadDim + 8;
constexpr int kMmaPanel = kHeadDim / kWarps;

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

__device__ __forceinline__ void mma_store_piece(uint4 code, unsigned scales, __half* dst, int piece) {
    scales = __shfl_sync(0xffffffffu, scales, 0, kMmaPieces);
    const __half scale = __ushort_as_half((uint16_t) (scales >> (16 * (piece / 4))));
    uint32_t halves[8];
    unpack16_int8_h2(code, __half2half2(scale), halves);
    *reinterpret_cast<uint4*>(dst) = make_uint4(halves[0], halves[1], halves[2], halves[3]);
    *reinterpret_cast<uint4*>(dst + 8) = make_uint4(halves[4], halves[5], halves[6], halves[7]);
}

__device__ __forceinline__ void mma_expand(const MmaRawTile& raw, __half (&k)[kTile][kMmaLd], MmaVTile& v, int tid) {
    const int piece = tid % kMmaPieces;
#pragma unroll
    for (int value = 0; value < 2; ++value) {
#pragma unroll
        for (int j = tid / kMmaPieces; j < kTile; j += kMmaKeysPerPass) {
            const unsigned scales = piece == 0 ? raw.scale[value][j] : 0u;
            __half* dst = value ? &v.row[j][piece * 16] : &k[j][piece * 16];
            mma_store_piece(raw.code[value][j][piece], scales, dst, piece);
        }
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

__device__ __forceinline__ void mma_expand(const MmaRegisterTile& raw, __half (&k)[kTile][kMmaLd], MmaVTile& v, int tid) {
    const int piece = tid % kMmaPieces;
#pragma unroll
    for (int value = 0; value < 2; ++value) {
#pragma unroll
        for (int pass = 0; pass < kMmaPasses; ++pass) {
            const int j = tid / kMmaPieces + pass * kMmaKeysPerPass;
            __half* dst = value ? &v.row[j][piece * 16] : &k[j][piece * 16];
            mma_store_piece(raw.code[value][pass], raw.scale[value][pass], dst, piece);
        }
    }
}

template <int Slabs>
struct alignas(16) MmaSmem {
    __half k[kTile][kMmaLd];
    MmaVTile v;
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
            mma_expand(current_tile, sm.k, v, tid);
        } else {
            if (!primed || stages > 1 || tile != lo)
                mma_prefetch(raw[future], kc, vc, tile + (stages - 1) * kTile, hi, g, a.cap, tid);
            mma_wait(stages - 1);
            // Each thread consumes its own copies; the scale's copying lane broadcasts inside its warp.
            mma_expand(raw[current], sm.k, v, tid);
        }
        __syncthreads();

        // QK: one warp owns one or two eight-key columns of its M slab.
        float scores[n_fragments][4] = {};
#pragma unroll
        for (int d = 0; d < kHeadDim; d += 16) {
            const auto& q = q_frag[d / 16];
#pragma unroll
            for (int n = 0; n < n_fragments; ++n) {
                unsigned b0, b1;
                ldsm_x2(b0, b1, &sm.k[score_n + 8 * n + (lane & 7)][d + ((lane >> 3) & 1) * 8]);
                mma16816(scores[n], q[0], q[1], q[2], q[3], b0, b1);
            }
        }
#pragma unroll
        for (int n = 0; n < n_fragments; ++n) {
            const int col = score_n + 8 * n + lane_pair;
            *reinterpret_cast<float2*>(&sm.score[score_m + lane_row][col]) = make_float2(scores[n][0], scores[n][1]);
            *reinterpret_cast<float2*>(&sm.score[score_m + lane_row + 8][col]) = make_float2(scores[n][2], scores[n][3]);
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
                sm.prob[quad][4 * j + member] = __float2half_rn(p);
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
                    make_float2(acc[slab][n][2 * pair], acc[slab][n][2 * pair + 1]);
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
    const bool batched = a.seg.at.n > 0;
    const int begin = batched ? a.seg.at.row0[seg] : 0;
    const int end = batched ? a.seg.at.row0[seg + 1] : rows;
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
        const int span = span_of(n), splits = (n + span - 1) / span;
        auto& sm = *reinterpret_cast<MmaSmem<1>*>(smem_raw);
        for (int s = blockIdx.z; s < splits; s += gridDim.z) {
            const int lo = s * span, hi = min(n, lo + span);
            attn_mma_slice<1, true>(a, begin, begin + 1, g, lo, hi, s, z, kc, vc, sm, nullptr,
                                    alternate_v, 0, false, part_ml, part_acc, threadIdx.x);
        }
        return;
    }
    for (int s = blockIdx.z; s < z; s += gridDim.z) {
        // Partition at a span or a tail tile boundary so a row takes the same steps in every window.
        for (int first = begin; first < end;) {
            const int n = a.pos[first] + 1, span = span_of(n), lo = s * span;
            if (n < min_keys || lo >= n) {
                ++first;
                continue;
            }
            int hi = min(n, lo + span);
            const int tiles = (hi + kTile - 1) / kTile;
            int last = first + 1;
            for (; last < end; ++last) {
                const int next = a.pos[last] + 1, next_hi = min(next, lo + span);
                if (next < min_keys || next <= lo || span_of(next) != span || (next_hi + kTile - 1) / kTile != tiles) break;
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

// A block per (row, head), a thread a channel: the row's slices merged with the sink, gated, written in fp32 and/or as
// q8_1 (a warp a block of 32 channels).
__global__ void __launch_bounds__(kHeadDim) attn_combine_kernel(AttnArgs a, int z, const float* __restrict__ part_ml,
                                                                const float* __restrict__ part_acc, float* __restrict__ out) {
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    int p, lo, n;
    row_keys(a, r, p, lo, n);
    const int span = span_of(n), splits = (n + span - 1) / span;
    const size_t base = ((size_t) r * kHeads + h) * z;
    const float t = a.sink[(size_t) r * kHeads + h];
    float mm = t;
    for (int s = 0; s < splits; ++s) mm = fmaxf(mm, part_ml[(base + s) * 2]);
    float ll = expf(t - mm), aa = 0.f;
    for (int s = 0; s < splits; ++s) {
        const float f = expf(part_ml[(base + s) * 2] - mm);
        ll = fmaf(part_ml[(base + s) * 2 + 1], f, ll);
        aa = fmaf(part_acc[(base + s) * kHeadDim + d], f, aa);
    }
    float o = aa / ll;
    const int c = h * kHeadDim + d;
    if (a.gate) o *= 1.f / (1.f + expf(-a.gate[(size_t) r * a.gate_ld + c]));
    if (out) out[(size_t) r * kHeads * kHeadDim + c] = o;
    if (a.q8) {
        const float amax = warp_max(fabsf(o));
        const float sum = warp_sum(o);
        const float dq = bk::q8_1_finite(amax / 127.0f);
        Q81& b = reinterpret_cast<Q81*>(a.q8)[(size_t) r * (kHeads * kHeadDim / 32) + c / 32];
        b.qs[c % 32] = bk::q8_1_quant(o, dq, amax);
        if ((c & 31) == 0) b.ds = bk::q8_1_ds(dq, sum);
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
    const int span = span_of(n), splits = (n + span - 1) / span;
    const bool active = s < splits;
    const int s_lo = lo + s * span, s_hi = min(p + 1, s_lo + span);
    const int p0 = a.pos[0];   // the window's first position: keys from it on are the window's rows (consecutive positions)
    const int key = s_lo + w * kTile + lane;
    const bool valid = active && key < s_hi;
    const uint8_t* kbase = static_cast<const uint8_t*>(a.k_cache) + (size_t) g * kRowBytes;
    const uint8_t* vbase = static_cast<const uint8_t*>(a.v_cache) + (size_t) g * kRowBytes;
    uint4 kr[kLoads], vr[kLoads];
    float ks[2] = {1.f, 1.f}, vs[2] = {1.f, 1.f};
    if (valid && key < p0) {
        const size_t cell = (size_t) (key % a.cap) * kKvHeads * kRowBytes;
#pragma unroll
        for (int i = 0; i < kLoads; ++i) kr[i] = reinterpret_cast<const uint4*>(kbase + cell)[i];
#pragma unroll
        for (int i = 0; i < kLoads; ++i) vr[i] = reinterpret_cast<const uint4*>(vbase + cell)[i];
        if constexpr (INT8) {
            row_scales(kbase + cell, ks);
            row_scales(vbase + cell, vs);
        }
    } else {
#pragma unroll
        for (int i = 0; i < kLoads; ++i) kr[i] = vr[i] = make_uint4(0, 0, 0, 0);
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
        if (valid && key >= p0) {   // a row of this window, as prepared above
            const uint4* kp = reinterpret_cast<const uint4*>(fs.kw[key - p0]);
            const uint4* vp = reinterpret_cast<const uint4*>(fs.vw[key - p0]);
#pragma unroll
            for (int i = 0; i < kLoads; ++i) kr[i] = kp[i];
#pragma unroll
            for (int i = 0; i < kLoads; ++i) vr[i] = vp[i];
            if constexpr (INT8) {
                row_scales(fs.kw[key - p0], ks);
                row_scales(fs.vw[key - p0], vs);
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
            } else {
#pragma unroll
                for (int i = 0; i < kLoads; ++i) *reinterpret_cast<uint4*>(vrow + 4 * i) = vr[i];
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

int launch_splits(int64_t max_keys) {
    return (int) std::min<int64_t>(kMaxSplits, std::max<int64_t>(1, (max_keys + kUnit - 1) / kUnit));
}

}  // namespace

int opt_in_device() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= kOptInDevices) throw std::runtime_error("inkling2: device out of range");
    return dev;
}

size_t attention_decode_scratch(int rows, int64_t max_keys) {
    const size_t slots = (size_t) rows * kHeads * (size_t) launch_splits(max_keys);
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
    MmaLaunch plans[3][2];  // narrow, one slab, two slabs; L2-resident/streaming
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
void mma_probe_plans(MmaLaunch (&plans)[2], int per_block, int per_sm, int reserved) {
    const auto kernel = attn_mma_split_kernel<Slabs, Narrow>;
    check_mma_probe(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, per_block));
    if constexpr (!Narrow)
        check_mma_probe(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
    const size_t compact_budget = (size_t) std::max(0, std::min(per_block / 2, per_sm / 2 - reserved));
    for (int streaming = 0; streaming < 2; ++streaming) {
        MmaLaunch best;
        bool best_compact = false;
        // Prefer two resident CTAs, then an alternating V tile, then more copies in flight.
        for (const bool ping_pong : {true, false}) {
            for (int stages = streaming ? 3 : 2; stages >= 1; --stages) {
                const size_t shared = sizeof(MmaSmem<Slabs>) + (ping_pong ? sizeof(MmaVTile) : 0) + stages * sizeof(MmaRawTile);
                if (shared > (size_t) per_block) continue;
                int resident = 0;
                check_mma_probe(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, kernel, kWarps * 32, shared));
                resident = std::min({resident, 2, per_sm / ((int) shared + reserved)});
                const bool compact = !Narrow || shared <= compact_budget;
                if (resident > 0 && (best.resident == 0 || compact > best_compact ||
                                     (compact == best_compact && resident > best.resident))) {
                    best = {shared, stages, resident, ping_pong};
                    best_compact = compact;
                }
                if (best_compact && best.resident == 2) break;
            }
            if (best_compact && best.resident == 2) break;
        }
        if (best.resident == 0) throw std::runtime_error("inkling2: no resident attention MMA block");
        plans[streaming] = best;
    }
}

const MmaDevice& mma_device(int dev) {
    static MmaDevice devices[kOptInDevices];
    static std::once_flag probed[kOptInDevices];
    std::call_once(probed[dev], [dev] {
        MmaDevice& d = devices[dev];
        d.sms = mma_attribute(dev, cudaDevAttrMultiProcessorCount);
        d.l2 = mma_attribute(dev, cudaDevAttrL2CacheSize);
        const int per_block = mma_attribute(dev, cudaDevAttrMaxSharedMemoryPerBlockOptin);
        const int per_sm = mma_attribute(dev, cudaDevAttrMaxSharedMemoryPerMultiprocessor);
        const int reserved = mma_attribute(dev, cudaDevAttrReservedSharedMemoryPerBlock);
        mma_probe_plans<1, true>(d.plans[0], per_block, per_sm, reserved);
        mma_probe_plans<1, false>(d.plans[1], per_block, per_sm, reserved);
        mma_probe_plans<2, false>(d.plans[2], per_block, per_sm, reserved);
        constexpr int shared = sizeof(MmaSmem<1>) + sizeof(MmaVTile);
        if (shared <= per_block) {
            int resident = 0;
            check_mma_probe(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &resident, attn_mma_split_kernel<1, true, true>, kWarps * 32, shared));
            resident = std::min({resident, per_sm / (shared + reserved), 3});
            d.singleton = {shared, 0, resident, true};
        }
    });
    return devices[dev];
}

template <int Slabs, bool Narrow = false, bool Direct = false>
void launch_mma(const AttnArgs& a, int rows, int min_keys, int64_t max_keys, int z, float* part_ml,
                float* part_acc, cudaStream_t stream, const MmaDevice& device) {
    const int segments = std::max(a.seg.at.n, 1);
    const int64_t bytes_per_segment = max_keys * (2 * kKvHeads * kKvInt8RowBytes);
    const auto& plan = Direct ? device.singleton : device.plans[Narrow ? 0 : Slabs][bytes_per_segment > device.l2 / segments];
    // Longer slices need more waves around concurrent side streams. Short singleton slices can
    // share a block while keeping the logical split order and filling the device.
    const int wave_floor = device.sms * plan.resident * (Narrow && !Direct ? 2 : 1);
    int shards = z;
    while (shards % 2 == 0 && shards / 2 * segments * kKvHeads >= wave_floor) shards /= 2;
    const auto& pdl = bk::pdl_state();
    const int prime_span = Narrow && !Direct && pdl.on && pdl.plain == 0 && a.seg.at.n == 0 &&
                           a.min_full_keys >= min_keys && max_keys <= a.cap ? span_of((int) a.min_full_keys) : 0;
    const dim3 grid(segments, kKvHeads, shards);
    bk::launch_pdl(attn_mma_split_kernel<Slabs, Narrow, Direct>, grid, dim3(kWarps * 32), plan.shared, stream,
                   a, rows, min_keys, z, plan.stages, plan.ping_pong, prime_span, part_ml, part_acc);
}

}  // namespace

// Only full-cache INT8 has an MMA reader. Its cutoff is a property of the row, not the padded launch bound.
static bool mma_decode(const AttnArgs& a, int rows, int64_t max_keys) {
    return a.int8 && a.window == 0 && rows <= kDecodeRows && (kMmaSingleton || rows >= 2) && max_keys >= kMmaMinKeys;
}

static void launch_attention_decode(const AttnArgs& a, int rows, int64_t max_keys, void* scratch, float* out, void* stream,
                                    bool mma, int mma_min_keys, bool scalar_prefix) {
    if (rows <= 0) return;
    cudaStream_t s = (cudaStream_t) stream;
    const int z = launch_splits(max_keys);
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
    const dim3 grid((unsigned) rows, kKvHeads, (unsigned) z);
    if (mma) {
        if (scalar_prefix) {
            bk::launch_pdl(attn_split_kernel<false, true, true>, grid, dim3(kWarps * 32), sizeof(Smem), s, a, z, part_ml, part_acc);
            check("attention_decode/shallow");
        }
        const MmaDevice& device = mma_device(dev);
        if (rows == 1 && max_keys <= kMmaSingletonShallowKeys && device.singleton.resident > 0)
            launch_mma<1, true, true>(a, rows, mma_min_keys, max_keys, z, part_ml, part_acc, s, device);
        else if (rows <= 2) launch_mma<1, true>(a, rows, mma_min_keys, max_keys, z, part_ml, part_acc, s, device);
        else if (rows * kGroup <= kMmaRows) launch_mma<1>(a, rows, mma_min_keys, max_keys, z, part_ml, part_acc, s, device);
        else launch_mma<2>(a, rows, mma_min_keys, max_keys, z, part_ml, part_acc, s, device);
    } else if (a.int8) bk::launch_pdl(attn_split_kernel<false, true>, grid, dim3(kWarps * 32), sizeof(Smem), s, a, z, part_ml, part_acc);
    else if (a.fp8) bk::launch_pdl(attn_split_kernel<true, false>, grid, dim3(kWarps * 32), sizeof(Smem), s, a, z, part_ml, part_acc);
    else bk::launch_pdl(attn_split_kernel<false, false>, grid, dim3(kWarps * 32), sizeof(Smem), s, a, z, part_ml, part_acc);
    check("attention_decode");
    bk::launch_pdl(attn_combine_kernel, dim3((unsigned) rows, kHeads), dim3(kHeadDim), 0, s, a, z, (const float*) part_ml, (const float*) part_acc, out);
    check("attention_decode/combine");
}

void attention_decode(const AttnArgs& a, int rows, int64_t max_keys, void* scratch, float* out, void* stream) {
    launch_attention_decode(a, rows, max_keys, scratch, out, stream, mma_decode(a, rows, max_keys), kMmaMinKeys,
                            a.min_full_keys < kMmaMinKeys);
}

void attention_decode_int8_test(const AttnArgs& a, int rows, int64_t max_keys, void* scratch, float* out, void* stream, bool mma) {
    if (!a.int8 || a.window != 0 || rows < 1 || rows > kDecodeRows)
        throw std::invalid_argument("inkling2: test needs a full-cache INT8 decode window");
    launch_attention_decode(a, rows, max_keys, scratch, out, stream, mma, 1, false);
}

void attention_window(const AttnPrep& pa, const AttnArgs& a, int rows, int64_t max_keys, void* scratch, void* stream) {
    if (rows <= 0) return;
    if (rows > kFusedRows || launch_splits(max_keys) > kClusterSplits || a.q8 == nullptr || a.seg.at.n > 0 ||
        mma_decode(a, rows, max_keys)) {
        decode_prepare(pa, rows, stream);
        attention_decode(a, rows, max_keys, scratch, nullptr, stream);
        return;
    }
    // 1-2 rows, a block an SM: each block asks for more than half of an SM's shared memory (it uses sizeof(FusedSmem)), so
    // a cluster's blocks spread over as many SMs instead of sharing them two by two through the preparation and the tile,
    // while the products' small blocks still fit beside one; 3 rows, two blocks an SM, so the 3 x 8 x 8 blocks take one wave
    static size_t smem_of[kOptInDevices] = {};   // per device, as above
    const int dev = opt_in_device();
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
