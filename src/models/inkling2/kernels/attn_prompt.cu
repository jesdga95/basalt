// src/models/inkling2/kernels/attn_prompt.cu - Inkling 2's prompt attention: a chunk's rows against every key before them,
// flash style on the tensor cores, with each query's relative bias over the last kExtent positions and its sink.
//
// A block takes kPos positions of the chunk and one key/value head. Its rows are the group's three query heads over those
// positions, head-major: warp w owns query head w / kWarpsPerHead of the group at 16 consecutive positions. Keys stream through shared
// memory in tiles of kKeys. S = Q K^T runs on fp16 tensor cores with fp32 accumulation. The relative bias of a warp's rows
// against a tile is one more product, R (the rows' relative states) times the band of the bias table the tile's distances
// span, moved through shared memory into the scores' layout. The softmax is online in fp32 (base 2), O += P V runs on fp16
// tensor cores with fp32 accumulation, and the sink joins the denominator at the end. A key at or after the chunk's first
// position comes from the chunk's own buffer (a full layer's cache, a sliding-window layer's stage), an earlier one from
// the cache (a window ring or the full cache).
#include "basalt/models/inkling2/prompt_kernels.hpp"
#include "attn_mma.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace basalt::models::inkling2 {
namespace {

constexpr int kPos = 64;                  // positions a block
constexpr int kWarpsPerHead = kPos / 16;  // 16 positions a warp
constexpr int kWarps = 3 * kWarpsPerHead;
constexpr int kThreads = kWarps * 32;
constexpr int kKeys = 32;                 // keys a tile
constexpr int kLd = kHeadDim + 8;         // padded fp16 row of a key tile (halves): ldmatrix without bank conflicts
constexpr int kBand = 96;                 // band rows: the distances a tile spans for the block (kPos + kKeys - 1)
constexpr int kBandLd = kRel + 8;         // padded band row (halves)
constexpr int kGCols = 48;                // distances a warp's rows span against a tile (16 + kKeys - 1), whole n-tiles
constexpr int kGLd = 55;                  // padded row of the exchanged bias (floats): two-way bank conflicts at most
constexpr float kLog2e = 1.4426950408889634f;
static_assert(kBand >= kPos + kKeys - 1 && kGCols >= 16 + kKeys - 1 && kGCols % 8 == 0 && kGLd >= kGCols, "band sizes");

struct KvTile {
    __half k[kKeys][kLd];
    __half v[kKeys][kLd];
};

template <bool FP8>
struct RawKvTile {
    static constexpr int kRowB = FP8 ? kHeadDim : kKvInt8RowBytes;
    alignas(16) uint8_t k[kKeys][kRowB];
    alignas(16) uint8_t v[kKeys][kRowB];
};

// Keep K/V ping-pong tiles separate from the relative-bias exchange so every tile (including sliding-window and near
// tiles within kExtent) overlaps its cp.async transfer with tensor-core compute.
constexpr int kFarStages = 4;
template <bool FP8, bool INT8>
union Smem {
    struct {
        RawKvTile<FP8> raw[kFarStages];
        KvTile kt[2];
    } far_q;
    struct {
        RawKvTile<FP8> raw[2];
        KvTile kt;
        __half band[kBand][kBandLd];
        float gx[kWarps][16][kGLd];
    } near_q;
};

template <>
union Smem<false, false> {
    KvTile tf[kFarStages];
    struct {
        KvTile t0;
        KvTile t1;
        __half band[kBand][kBandLd];
        float gx[kWarps][16][kGLd];
    } near;
};

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("inkling2 ") + what + ": " + cudaGetErrorString(e));
}

using attn_mma::ldsm_x2;
using attn_mma::ldsm_x4;
using attn_mma::ldsm_x4_t;
using attn_mma::mma16816;
using attn_mma::smem_u32;

__device__ __forceinline__ void cp16(void* dst, const void* src, bool ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ unsigned pack_h2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&h);
}
__device__ __forceinline__ float quad_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}
__device__ __forceinline__ float quad_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v + __shfl_xor_sync(0xffffffffu, v, 2);
}

__device__ __forceinline__ uint32_t e4m3x2_to_h2(uint16_t u) {
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) u, __NV_E4M3);
    return (uint32_t) h.x | ((uint32_t) h.y << 16);
}

__device__ __forceinline__ int64_t wrap_key(int64_t key, int64_t cap, bool pow2) {
    if (key < cap) return key;
    return pow2 ? (key & (cap - 1)) : (key % cap);
}

__device__ __forceinline__ uint32_t tile_slot0(int64_t k0, int64_t cap, bool pow2) {
    if (k0 < cap) return (uint32_t) k0;
    if (pow2) return (uint32_t) (k0 & (cap - 1));
    return (uint32_t) ((uint64_t) k0 % (uint32_t) cap);
}

__device__ __forceinline__ void load_tile_f16(const AttnPromptArgs& a, KvTile& sm, int kvh, int64_t k0, int64_t lo, int64_t hi,
                                              int tid, bool cap_pow2, bool chunk_pow2) {
    constexpr int kPieces = kKeys * kHeadDim / 8;
    if (k0 >= lo && k0 + kKeys - 1 <= hi && (k0 + kKeys <= a.pos0 || k0 >= a.pos0)) {
        const bool fresh = k0 >= a.pos0;
        const int64_t cap = fresh ? a.chunk_cap : a.cap;
        const uint32_t cap32 = (uint32_t) cap;
        const uint32_t s0 = tile_slot0(k0, cap, fresh ? chunk_pow2 : cap_pow2);
        const __half* kc = static_cast<const __half*>(fresh ? a.k_chunk : a.k_cache) + (size_t) kvh * kHeadDim;
        const __half* vc = static_cast<const __half*>(fresh ? a.v_chunk : a.v_cache) + (size_t) kvh * kHeadDim;
        if (s0 + kKeys <= cap32) {
#pragma unroll
            for (int c = tid; c < 2 * kPieces; c += kThreads) {
                const bool isv = c >= kPieces;
                const int rem = isv ? c - kPieces : c;
                const int j = rem >> 4, col = (rem & 15) * 8;
                const size_t off = (size_t) (s0 + (uint32_t) j) * (kKvHeads * kHeadDim) + col;
                cp16(isv ? &sm.v[j][col] : &sm.k[j][col], (isv ? vc : kc) + off, true);
            }
        } else {
#pragma unroll
            for (int c = tid; c < 2 * kPieces; c += kThreads) {
                const bool isv = c >= kPieces;
                const int rem = isv ? c - kPieces : c;
                const int j = rem >> 4, col = (rem & 15) * 8;
                uint32_t sl = s0 + (uint32_t) j;
                if (sl >= cap32) sl -= cap32;
                const size_t off = (size_t) sl * (kKvHeads * kHeadDim) + col;
                cp16(isv ? &sm.v[j][col] : &sm.k[j][col], (isv ? vc : kc) + off, true);
            }
        }
        return;
    }
    for (int c = tid; c < 2 * kPieces; c += kThreads) {
        const bool isv = c >= kPieces;
        const int rem = isv ? c - kPieces : c;
        const int j = rem >> 4, col = (rem & 15) * 8;
        const int64_t key = k0 + j;
        const bool ok = key >= lo && key <= hi;
        const bool fresh = key >= a.pos0;
        const int64_t cap = fresh ? a.chunk_cap : a.cap;
        const int64_t slot = ok ? wrap_key(key, cap, fresh ? chunk_pow2 : cap_pow2) : 0;
        const __half* base = static_cast<const __half*>(fresh ? (isv ? a.v_chunk : a.k_chunk) : (isv ? a.v_cache : a.k_cache));
        const __half* src = base + ((size_t) slot * kKvHeads + kvh) * kHeadDim + col;
        cp16(isv ? &sm.v[j][col] : &sm.k[j][col], src, ok);
    }
}

template <bool FP8>
__device__ __forceinline__ void load_raw_tile(const AttnPromptArgs& a, RawKvTile<FP8>& rt, int kvh, int64_t k0, int64_t lo,
                                              int64_t hi, int tid, bool cap_pow2, bool chunk_pow2) {
    constexpr int kRowB = RawKvTile<FP8>::kRowB;
    constexpr int kVecs = kRowB / 16;
    constexpr int kPieces = kKeys * kVecs;
    if (k0 >= lo && k0 + kKeys - 1 <= hi && (k0 + kKeys <= a.pos0 || k0 >= a.pos0)) {
        const bool fresh = k0 >= a.pos0;
        const int64_t cap = fresh ? a.chunk_cap : a.cap;
        const uint32_t cap32 = (uint32_t) cap;
        const uint32_t s0 = tile_slot0(k0, cap, fresh ? chunk_pow2 : cap_pow2);
        const uint8_t* kc = static_cast<const uint8_t*>(fresh ? a.k_chunk : a.k_cache) + (size_t) kvh * kRowB;
        const uint8_t* vc = static_cast<const uint8_t*>(fresh ? a.v_chunk : a.v_cache) + (size_t) kvh * kRowB;
        if (s0 + kKeys <= cap32) {
#pragma unroll
            for (int c = tid; c < 2 * kPieces; c += kThreads) {
                const bool isv = c >= kPieces;
                const int rem = isv ? c - kPieces : c;
                const int j = rem / kVecs, col = (rem - j * kVecs) * 16;
                const size_t off = (size_t) (s0 + (uint32_t) j) * (kKvHeads * kRowB) + col;
                cp16(isv ? &rt.v[j][col] : &rt.k[j][col], (isv ? vc : kc) + off, true);
            }
        } else {
#pragma unroll
            for (int c = tid; c < 2 * kPieces; c += kThreads) {
                const bool isv = c >= kPieces;
                const int rem = isv ? c - kPieces : c;
                const int j = rem / kVecs, col = (rem - j * kVecs) * 16;
                uint32_t sl = s0 + (uint32_t) j;
                if (sl >= cap32) sl -= cap32;
                const size_t off = (size_t) sl * (kKvHeads * kRowB) + col;
                cp16(isv ? &rt.v[j][col] : &rt.k[j][col], (isv ? vc : kc) + off, true);
            }
        }
        return;
    }
    for (int c = tid; c < 2 * kPieces; c += kThreads) {
        const bool isv = c >= kPieces;
        const int rem = isv ? c - kPieces : c;
        const int j = rem / kVecs, col = (rem - j * kVecs) * 16;
        const int64_t key = k0 + j;
        const bool ok = key >= lo && key <= hi;
        const bool fresh = key >= a.pos0;
        const int64_t cap = fresh ? a.chunk_cap : a.cap;
        const int64_t slot = ok ? wrap_key(key, cap, fresh ? chunk_pow2 : cap_pow2) : 0;
        const uint8_t* base = static_cast<const uint8_t*>(fresh ? (isv ? a.v_chunk : a.k_chunk) : (isv ? a.v_cache : a.k_cache));
        const uint8_t* src = base + ((size_t) slot * kKvHeads + kvh) * kRowB + col;
        cp16(isv ? &rt.v[j][col] : &rt.k[j][col], src, ok);
    }
}

template <bool FP8>
__device__ __forceinline__ void unpack_raw_tile(const RawKvTile<FP8>& rt, KvTile& sm, int tid) {
    constexpr int kPieces = kKeys * kHeadDim / 16;
#pragma unroll
    for (int c = tid; c < 2 * kPieces; c += kThreads) {
        const bool isv = c >= kPieces;
        const int rem = isv ? c - kPieces : c;
        const int j = rem >> 3, col = (rem & 7) * 16;
        const uint8_t* rptr = isv ? rt.v[j] : rt.k[j];
        const uint4 u = *reinterpret_cast<const uint4*>(rptr + col);
        uint32_t h[8];
        if constexpr (FP8) {
            const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                h[2 * i] = e4m3x2_to_h2((uint16_t) (w[i] & 0xFFFFu));
                h[2 * i + 1] = e4m3x2_to_h2((uint16_t) (w[i] >> 16));
            }
        } else {
            const __half sc = reinterpret_cast<const __half*>(rptr + kHeadDim)[col / 64];
            const uint32_t w[4] = {u.x ^ 0x80808080u, u.y ^ 0x80808080u, u.z ^ 0x80808080u, u.w ^ 0x80808080u};
            const __half2 bias = __half2half2(__ushort_as_half(0x6480u)), s2 = __half2half2(sc);
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j2 = 0; j2 < 2; ++j2) {
                    const uint32_t b = __byte_perm(w[i], 0x64646464u, j2 ? 0x7362u : 0x5140u);
                    const __half2 v = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&b), bias), s2);
                    h[2 * i + j2] = *reinterpret_cast<const uint32_t*>(&v);
                }
        }
        __half* dst = isv ? &sm.v[j][col] : &sm.k[j][col];
        *reinterpret_cast<uint4*>(dst) = make_uint4(h[0], h[1], h[2], h[3]);
        *reinterpret_cast<uint4*>(dst + 8) = make_uint4(h[4], h[5], h[6], h[7]);
    }
}

template <bool FP8, bool INT8>
__global__ void __launch_bounds__(kThreads, 1) attn_prompt_kernel(const AttnPromptArgs a, const int rows, float* __restrict__ out) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem<FP8, INT8>& sm = *reinterpret_cast<Smem<FP8, INT8>*>(smem_raw);
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int g = lane >> 2, tq = lane & 3;
    const int n_pos_blk = (int) gridDim.x;
    const int cta = (int) blockIdx.y * n_pos_blk + (int) blockIdx.x;
    const int64_t kv_span = a.window > 0 ? min(a.pos0 + rows, (int64_t) a.window + rows) : (a.pos0 + rows);
    const int kv_grp = (kv_span <= 16384) ? kKvHeads : (kv_span <= 49152 ? 2 : 1);
    const int super = cta / (n_pos_blk * kv_grp);
    const int rem_cta = cta - super * (n_pos_blk * kv_grp);
    const int pos_blk = rem_cta / kv_grp;
    const int kvh = super * kv_grp + (rem_cta % kv_grp);
    const int i0 = (n_pos_blk - 1 - pos_blk) * kPos;
    const int64_t q0 = a.pos0 + i0;
    const int head = kvh * 3 + w / kWarpsPerHead;
    const int jw = (w % kWarpsPerHead) * 16;
    const int64_t pw = q0 + jw;
    const int ia = i0 + jw + g, ib = ia + 8;
    const bool va = ia < rows, vb = ib < rows;
    const int64_t hi = a.pos0 + min(rows, i0 + kPos) - 1;
    const int64_t lo = a.window > 0 ? max((int64_t) 0, q0 - a.window + 1) : 0;

    uint32_t qf[kHeadDim / 16][4];
    {
        const float* qa = a.q + ((size_t) (va ? ia : 0) * kHeads + head) * kHeadDim + 2 * tq;
        const float* qb = a.q + ((size_t) (vb ? ib : 0) * kHeads + head) * kHeadDim + 2 * tq;
#pragma unroll
        for (int kk = 0; kk < kHeadDim / 16; ++kk) {
            const float2 x0 = *reinterpret_cast<const float2*>(qa + 16 * kk);
            const float2 x1 = *reinterpret_cast<const float2*>(qb + 16 * kk);
            const float2 x2 = *reinterpret_cast<const float2*>(qa + 16 * kk + 8);
            const float2 x3 = *reinterpret_cast<const float2*>(qb + 16 * kk + 8);
            qf[kk][0] = pack_h2(x0.x, x0.y);
            qf[kk][1] = pack_h2(x1.x, x1.y);
            qf[kk][2] = pack_h2(x2.x, x2.y);
            qf[kk][3] = pack_h2(x3.x, x3.y);
        }
    }

    float o[kHeadDim / 8][4];
#pragma unroll
    for (int n = 0; n < kHeadDim / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    float m_a = -INFINITY, m_b = -INFINITY, l_a = 0.f, l_b = 0.f;
    const float scale2 = a.scale * kLog2e;
    const bool cap_pow2 = a.cap > 0 && (a.cap & (a.cap - 1)) == 0;
    const bool chunk_pow2 = a.chunk_cap > 0 && (a.chunk_cap & (a.chunk_cap - 1)) == 0;

    int64_t k_first = (lo / kKeys) * kKeys;
    if (a.window == 0) {
        const int64_t k_fast_end = min(a.pos0 - kKeys, q0 - kExtent - (kKeys - 1));
        const int n_fast = (k_fast_end >= k_first) ? (int) ((k_fast_end - k_first) / kKeys + 1) : 0;
        if (n_fast >= kFarStages) {
            if constexpr (!FP8 && !INT8) {
#pragma unroll
                for (int s = 0; s < kFarStages - 1; ++s) {
                    load_tile_f16(a, sm.tf[s], kvh, k_first + (int64_t) s * kKeys, lo, hi, tid, cap_pow2, chunk_pow2);
                    asm volatile("cp.async.commit_group;\n" ::: "memory");
                }
            } else {
#pragma unroll
                for (int s = 0; s < kFarStages - 1; ++s) {
                    load_raw_tile<FP8>(a, sm.far_q.raw[s], kvh, k_first + (int64_t) s * kKeys, lo, hi, tid, cap_pow2, chunk_pow2);
                    asm volatile("cp.async.commit_group;\n" ::: "memory");
                }
                asm volatile("cp.async.wait_group 2;\n" ::: "memory");
                __syncthreads();
                unpack_raw_tile<FP8>(sm.far_q.raw[0], sm.far_q.kt[0], tid);
                load_raw_tile<FP8>(a, sm.far_q.raw[3], kvh, k_first + 3 * kKeys, lo, hi, tid, cap_pow2, chunk_pow2);
                asm volatile("cp.async.commit_group;\n" ::: "memory");
            }
            for (int ti = 0; ti < n_fast; ++ti) {
                if constexpr (!FP8 && !INT8) {
                    const int rem_after = min(kFarStages - 2, (n_fast - 1) - ti);
                    if (rem_after == 2) asm volatile("cp.async.wait_group 2;\n" ::: "memory");
                    else if (rem_after == 1) asm volatile("cp.async.wait_group 1;\n" ::: "memory");
                    else asm volatile("cp.async.wait_group 0;\n" ::: "memory");
                    __syncthreads();
                    if (ti + 3 < n_fast) {
                        load_tile_f16(a, sm.tf[(ti + 3) & 3], kvh, k_first + (int64_t) (ti + 3) * kKeys, lo, hi, tid, cap_pow2, chunk_pow2);
                        asm volatile("cp.async.commit_group;\n" ::: "memory");
                    }
                } else {
                    if (ti + 1 < n_fast) {
                        const int last_issued = min(n_fast - 1, ti + 3);
                        const int rem_after = last_issued - (ti + 1);
                        if (rem_after == 2) asm volatile("cp.async.wait_group 2;\n" ::: "memory");
                        else if (rem_after == 1) asm volatile("cp.async.wait_group 1;\n" ::: "memory");
                        else asm volatile("cp.async.wait_group 0;\n" ::: "memory");
                    }
                    __syncthreads();
                    if (ti + 4 < n_fast) {
                        load_raw_tile<FP8>(a, sm.far_q.raw[ti & 3], kvh, k_first + (int64_t) (ti + 4) * kKeys, lo, hi, tid, cap_pow2, chunk_pow2);
                        asm volatile("cp.async.commit_group;\n" ::: "memory");
                    }
                    if (ti + 1 < n_fast) {
                        unpack_raw_tile<FP8>(sm.far_q.raw[(ti + 1) & 3], sm.far_q.kt[(ti + 1) & 1], tid);
                    }
                }
                const KvTile& kt = [&]() -> const KvTile& {
                    if constexpr (!FP8 && !INT8) return sm.tf[ti & 3];
                    else return sm.far_q.kt[ti & 1];
                }();
                float s[kKeys / 8][4];
#pragma unroll
                for (int n = 0; n < kKeys / 8; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
#pragma unroll
                for (int kk = 0; kk < kHeadDim / 16; ++kk) {
#pragma unroll
                    for (int n = 0; n < kKeys / 8; n += 2) {
                        unsigned b0, b1, b2, b3;
                        ldsm_x4(b0, b1, b2, b3, &kt.k[n * 8 + (lane & 7) + ((lane >> 4) << 3)][kk * 16 + ((lane >> 3) & 1) * 8]);
                        mma16816(s[n], qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3], b0, b1);
                        mma16816(s[n + 1], qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3], b2, b3);
                    }
                }
                float mx_a = m_a, mx_b = m_b;
#pragma unroll
                for (int n = 0; n < kKeys / 8; ++n) {
                    s[n][0] *= scale2; s[n][1] *= scale2; s[n][2] *= scale2; s[n][3] *= scale2;
                    mx_a = fmaxf(mx_a, fmaxf(s[n][0], s[n][1]));
                    mx_b = fmaxf(mx_b, fmaxf(s[n][2], s[n][3]));
                }
                mx_a = quad_max(mx_a);
                mx_b = quad_max(mx_b);
                const float ca = exp2f(m_a - mx_a), cb = exp2f(m_b - mx_b);
                m_a = mx_a;
                m_b = mx_b;
                float ps_a = 0.f, ps_b = 0.f;
#pragma unroll
                for (int n = 0; n < kKeys / 8; ++n) {
                    s[n][0] = exp2f(s[n][0] - m_a);
                    s[n][1] = exp2f(s[n][1] - m_a);
                    s[n][2] = exp2f(s[n][2] - m_b);
                    s[n][3] = exp2f(s[n][3] - m_b);
                    ps_a += s[n][0] + s[n][1];
                    ps_b += s[n][2] + s[n][3];
                }
                l_a = l_a * ca + ps_a;
                l_b = l_b * cb + ps_b;
#pragma unroll
                for (int n = 0; n < kHeadDim / 8; ++n) {
                    o[n][0] *= ca; o[n][1] *= ca; o[n][2] *= cb; o[n][3] *= cb;
                }
                unsigned pf[kKeys / 16][4];
#pragma unroll
                for (int j = 0; j < kKeys / 16; ++j) {
                    pf[j][0] = pack_h2(s[2 * j][0], s[2 * j][1]);
                    pf[j][1] = pack_h2(s[2 * j][2], s[2 * j][3]);
                    pf[j][2] = pack_h2(s[2 * j + 1][0], s[2 * j + 1][1]);
                    pf[j][3] = pack_h2(s[2 * j + 1][2], s[2 * j + 1][3]);
                }
#pragma unroll
                for (int n = 0; n < kHeadDim / 8; n += 2)
#pragma unroll
                    for (int j = 0; j < kKeys / 16; ++j) {
                        unsigned b0, b1, b2, b3;
                        ldsm_x4_t(b0, b1, b2, b3, &kt.v[j * 16 + (lane & 15)][n * 8 + (lane >> 4) * 8]);
                        mma16816(o[n], pf[j][0], pf[j][1], pf[j][2], pf[j][3], b0, b1);
                        mma16816(o[n + 1], pf[j][0], pf[j][1], pf[j][2], pf[j][3], b2, b3);
                    }
            }
            __syncthreads();
            k_first += (int64_t) n_fast * kKeys;
        }
    }

    uint32_t rf[4];
    {
        const float* ra = a.r + ((size_t) (va ? ia : 0) * kHeads + head) * kRel + 2 * tq;
        const float* rb = a.r + ((size_t) (vb ? ib : 0) * kHeads + head) * kRel + 2 * tq;
        rf[0] = pack_h2(ra[0], ra[1]);
        rf[1] = pack_h2(rb[0], rb[1]);
        rf[2] = pack_h2(ra[8], ra[9]);
        rf[3] = pack_h2(rb[8], rb[9]);
    }
    const int bt = tid / 24, bb = (tid % 24) * 4;
    static_assert(kThreads == kRel * 24 && kBand == 24 * 4, "band tile thread map");
    const auto far = [&](int64_t k0) { return q0 - k0 - (kKeys - 1) >= kExtent; };

    __half (*band)[kBandLd] = [&]() -> __half (*)[kBandLd] {
        if constexpr (!FP8 && !INT8) return sm.near.band;
        else return sm.near_q.band;
    }();
    float (*gx)[kGLd] = [&]() -> float (*)[kGLd] {
        if constexpr (!FP8 && !INT8) return sm.near.gx[w];
        else return sm.near_q.gx[w];
    }();

    int cur = 0;
    if (k_first <= hi) {
        if constexpr (!FP8 && !INT8)
            load_tile_f16(a, sm.near.t0, kvh, k_first, lo, hi, tid, cap_pow2, chunk_pow2);
        else
            load_raw_tile<FP8>(a, sm.near_q.raw[0], kvh, k_first, lo, hi, tid, cap_pow2, chunk_pow2);
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    }
    for (int64_t k0 = k_first; k0 <= hi; k0 += kKeys) {
        const bool has_next = k0 + kKeys <= hi;
        if (has_next) {
            if constexpr (!FP8 && !INT8)
                load_tile_f16(a, cur ? sm.near.t0 : sm.near.t1, kvh, k0 + kKeys, lo, hi, tid, cap_pow2, chunk_pow2);
            else
                load_raw_tile<FP8>(a, sm.near_q.raw[cur ^ 1], kvh, k0 + kKeys, lo, hi, tid, cap_pow2, chunk_pow2);
            asm volatile("cp.async.commit_group;\n" ::: "memory");
        }
        if (!far(k0)) {
            const int64_t d0 = q0 - k0 - (kKeys - 1) + bb;
            const float* row = a.rel_table + (size_t) bt * kExtent;
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int64_t d = d0 + u;
                band[bb + u][bt] = __float2half_rn(d >= 0 && d < kExtent ? row[d] : 0.f);
            }
        }
        if (has_next) asm volatile("cp.async.wait_group 1;\n" ::: "memory");
        else asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();
        if constexpr (FP8 || INT8) {
            unpack_raw_tile<FP8>(sm.near_q.raw[cur], sm.near_q.kt, tid);
            __syncthreads();
        }
        const KvTile& kt = [&]() -> const KvTile& {
            if constexpr (!FP8 && !INT8) return cur ? sm.near.t1 : sm.near.t0;
            else return sm.near_q.kt;
        }();

        const int64_t p_first = pw, p_last = pw + 15;
        const bool active = !(k0 > p_last || (a.window > 0 && k0 + kKeys - 1 <= p_first - a.window));
        if (active) {
            const bool whole = k0 + kKeys - 1 <= p_first && (a.window <= 0 || k0 > p_last - a.window);
            float s[kKeys / 8][4];
#pragma unroll
            for (int n = 0; n < kKeys / 8; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
#pragma unroll
            for (int kk = 0; kk < kHeadDim / 16; ++kk) {
#pragma unroll
                for (int n = 0; n < kKeys / 8; n += 2) {
                    unsigned b0, b1, b2, b3;
                    ldsm_x4(b0, b1, b2, b3, &kt.k[n * 8 + (lane & 7) + ((lane >> 4) << 3)][kk * 16 + ((lane >> 3) & 1) * 8]);
                    mma16816(s[n], qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3], b0, b1);
                    mma16816(s[n + 1], qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3], b2, b3);
                }
            }
            if (p_first - (k0 + kKeys - 1) < kExtent) {
                float ga[kGCols / 8][4];
#pragma unroll
                for (int nt = 0; nt < kGCols / 8; ++nt) {
                    ga[nt][0] = ga[nt][1] = ga[nt][2] = ga[nt][3] = 0.f;
                    unsigned b0, b1;
                    ldsm_x2(b0, b1, &band[jw + nt * 8 + (lane & 7)][((lane >> 3) & 1) * 8]);
                    mma16816(ga[nt], rf[0], rf[1], rf[2], rf[3], b0, b1);
                }
#pragma unroll
                for (int nt = 0; nt < kGCols / 8; ++nt) {
                    gx[g][nt * 8 + 2 * tq] = ga[nt][0];
                    gx[g][nt * 8 + 2 * tq + 1] = ga[nt][1];
                    gx[g + 8][nt * 8 + 2 * tq] = ga[nt][2];
                    gx[g + 8][nt * 8 + 2 * tq + 1] = ga[nt][3];
                }
                __syncwarp();
#pragma unroll
                for (int n = 0; n < kKeys / 8; ++n)
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const int j = g + (e >> 1) * 8, kk = n * 8 + 2 * tq + (e & 1);
                        s[n][e] = fmaf(s[n][e], scale2, gx[j][j - kk + kKeys - 1] * kLog2e);
                    }
                __syncwarp();
            } else {
#pragma unroll
                for (int n = 0; n < kKeys / 8; ++n)
#pragma unroll
                    for (int e = 0; e < 4; ++e) s[n][e] *= scale2;
            }
            if (!whole) {
#pragma unroll
                for (int n = 0; n < kKeys / 8; ++n)
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const int64_t key = k0 + n * 8 + 2 * tq + (e & 1), p = pw + g + (e >> 1) * 8;
                        if (key > p || (a.window > 0 && key <= p - a.window)) s[n][e] = -INFINITY;
                    }
            }
            float mx_a = m_a, mx_b = m_b;
#pragma unroll
            for (int n = 0; n < kKeys / 8; ++n) {
                mx_a = fmaxf(mx_a, fmaxf(s[n][0], s[n][1]));
                mx_b = fmaxf(mx_b, fmaxf(s[n][2], s[n][3]));
            }
            mx_a = quad_max(mx_a);
            mx_b = quad_max(mx_b);
            float ca, cb, ps_a = 0.f, ps_b = 0.f;
            if (whole) {
                ca = exp2f(m_a - mx_a);
                cb = exp2f(m_b - mx_b);
                m_a = mx_a;
                m_b = mx_b;
#pragma unroll
                for (int n = 0; n < kKeys / 8; ++n) {
                    s[n][0] = exp2f(s[n][0] - m_a);
                    s[n][1] = exp2f(s[n][1] - m_a);
                    s[n][2] = exp2f(s[n][2] - m_b);
                    s[n][3] = exp2f(s[n][3] - m_b);
                    ps_a += s[n][0] + s[n][1];
                    ps_b += s[n][2] + s[n][3];
                }
            } else {
                ca = mx_a == -INFINITY ? 1.f : exp2f(m_a - mx_a);
                cb = mx_b == -INFINITY ? 1.f : exp2f(m_b - mx_b);
                m_a = mx_a;
                m_b = mx_b;
#pragma unroll
                for (int n = 0; n < kKeys / 8; ++n) {
                    s[n][0] = s[n][0] == -INFINITY ? 0.f : exp2f(s[n][0] - m_a);
                    s[n][1] = s[n][1] == -INFINITY ? 0.f : exp2f(s[n][1] - m_a);
                    s[n][2] = s[n][2] == -INFINITY ? 0.f : exp2f(s[n][2] - m_b);
                    s[n][3] = s[n][3] == -INFINITY ? 0.f : exp2f(s[n][3] - m_b);
                    ps_a += s[n][0] + s[n][1];
                    ps_b += s[n][2] + s[n][3];
                }
            }
            l_a = l_a * ca + ps_a;
            l_b = l_b * cb + ps_b;
#pragma unroll
            for (int n = 0; n < kHeadDim / 8; ++n) {
                o[n][0] *= ca;
                o[n][1] *= ca;
                o[n][2] *= cb;
                o[n][3] *= cb;
            }
            unsigned pf[kKeys / 16][4];
#pragma unroll
            for (int j = 0; j < kKeys / 16; ++j) {
                pf[j][0] = pack_h2(s[2 * j][0], s[2 * j][1]);
                pf[j][1] = pack_h2(s[2 * j][2], s[2 * j][3]);
                pf[j][2] = pack_h2(s[2 * j + 1][0], s[2 * j + 1][1]);
                pf[j][3] = pack_h2(s[2 * j + 1][2], s[2 * j + 1][3]);
            }
#pragma unroll
            for (int n = 0; n < kHeadDim / 8; n += 2)
#pragma unroll
                for (int j = 0; j < kKeys / 16; ++j) {
                    unsigned b0, b1, b2, b3;
                    ldsm_x4_t(b0, b1, b2, b3, &kt.v[j * 16 + (lane & 15)][n * 8 + (lane >> 4) * 8]);
                    mma16816(o[n], pf[j][0], pf[j][1], pf[j][2], pf[j][3], b0, b1);
                    mma16816(o[n + 1], pf[j][0], pf[j][1], pf[j][2], pf[j][3], b2, b3);
                }
        }
        __syncthreads();
        cur ^= 1;
    }

    l_a = quad_sum(l_a);
    l_b = quad_sum(l_b);
    const float sa = va ? a.sink[(size_t) ia * kHeads + head] * kLog2e : 0.f;
    const float sb = vb ? a.sink[(size_t) ib * kHeads + head] * kLog2e : 0.f;
    const float ma = fmaxf(m_a, sa), mb = fmaxf(m_b, sb);
    const float fa = exp2f(m_a - ma), fb = exp2f(m_b - mb);
    const float inv_a = fa / (l_a * fa + exp2f(sa - ma)), inv_b = fb / (l_b * fb + exp2f(sb - mb));
#pragma unroll
    for (int n = 0; n < kHeadDim / 8; ++n) {
        const int col = n * 8 + 2 * tq;
        if (va) *reinterpret_cast<float2*>(out + ((size_t) ia * kHeads + head) * kHeadDim + col) = make_float2(o[n][0] * inv_a, o[n][1] * inv_a);
        if (vb) *reinterpret_cast<float2*>(out + ((size_t) ib * kHeads + head) * kHeadDim + col) = make_float2(o[n][2] * inv_b, o[n][3] * inv_b);
    }
}

}  // namespace

void attention_prompt(const AttnPromptArgs& a, int rows, float* out, void* stream) {
    if (rows <= 0) return;
    const dim3 grid((unsigned) ((rows + kPos - 1) / kPos), kKvHeads);
    cudaStream_t s = (cudaStream_t) stream;
    static bool sized[kOptInDevices] = {};
    if (bool& z = sized[opt_in_device()]; !z) {
        cudaFuncSetAttribute(attn_prompt_kernel<true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(Smem<true, false>));
        cudaFuncSetAttribute(attn_prompt_kernel<false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(Smem<false, false>));
        cudaFuncSetAttribute(attn_prompt_kernel<false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(Smem<false, true>));
        z = true;
    }
    if (a.int8) attn_prompt_kernel<false, true><<<grid, kThreads, sizeof(Smem<false, true>), s>>>(a, rows, out);
    else if (a.fp8) attn_prompt_kernel<true, false><<<grid, kThreads, sizeof(Smem<true, false>), s>>>(a, rows, out);
    else attn_prompt_kernel<false, false><<<grid, kThreads, sizeof(Smem<false, false>), s>>>(a, rows, out);
    check("attention_prompt");
}

}  // namespace basalt::models::inkling2
