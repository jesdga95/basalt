// src/prefill/moe_fp4.cu - see include/basalt/prefill/moe_fp4.hpp.
// The block-scaled MMA wrapper and the tile structure (64-token tiles x 128 weight rows, eight warps, activation
// fragments through ldmatrix, scale registers per 64-wide K step) follow Infernix's grouped NVFP4 GEMM
// (src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh, src/ops/common/mma.cuh).
// Adapted from Infernix (Ian Ranson and contributors), Apache-2.0: see THIRD_PARTY_LICENSES.
#include "basalt/prefill/moe_fp4.hpp"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "basalt/log.hpp"

namespace basalt::prefill::fp4 {
namespace {

constexpr int kRowsPerBlock = 128;
constexpr int kKTile = 128;                         // values per pipeline stage
constexpr int kStages = 4;
constexpr int kACodeBytes = kTileTokens * kKTile / 2;   // 4096
constexpr int kAScaleBytes = kTileTokens * kKTile / 16; // 512
constexpr int kAStage = kACodeBytes + kAScaleBytes;     // 4608
constexpr int kBRowBytes = kKTile / 64 * 36;            // 72
constexpr int kBRowStride = 80;                         // 16B-aligned row stride in smem (allows 8B alignment shift, conflict-free)
constexpr int kBStage = kRowsPerBlock * kBRowStride;    // 10240
constexpr int kBlockTiles = 2;                          // 64-token tiles per GEMM block
// TWO: a stage holds both activation terms of its 128 columns and the weights once, three stages (99 KB a block at most)
template <int TM> __host__ __device__ constexpr int stages() { return TM == 1 ? kStages : 3; }
template <int TM> __host__ __device__ constexpr int smem_bytes() { return stages<TM>() * (TM * kBlockTiles * kAStage + kBStage); }
constexpr int kThreads = 256;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "prefill fp4: %s: %s", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

__device__ __forceinline__ uint32_t smem_addr(const void* p) { return (uint32_t) __cvta_generic_to_shared(p); }

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

// c += A (16 tokens x 64 values, E2M1 with a UE4M3 scale per 16) * B (64 x 8 weight rows, same format)
__device__ __forceinline__ void mma_fp4(float* c, const unsigned* a, unsigned b0, unsigned b1, unsigned sfa,
                                        unsigned sfb) {
    constexpr unsigned short kId = 0;
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "h"(kId), "h"(kId),
                   "r"(sfb), "h"(kId), "h"(kId));
}

__device__ __forceinline__ void cp_async8(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(smem_addr(dst)), "l"(src));
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr(dst)), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// the byte of a 64-byte tile row holding logical byte lb of row rr: the four 16-byte chunks rotate with the row so
// that eight rows of an ldmatrix read different banks
__device__ __forceinline__ int swz(int rr, int lb) { return (((lb >> 4) ^ ((rr >> 1) & 3)) << 4) + (lb & 15); }

// ------------------------------------------------------------------------------------------ planning

__global__ void plan_kernel(const int32_t* __restrict__ bounds, int n, int32_t* __restrict__ tile_off) {
    if (threadIdx.x != 0) return;
    int t = 0;
    for (int j = 0; j < n; ++j) {
        tile_off[j] = t;
        t += (bounds[j + 1] - bounds[j] + kTileTokens - 1) / kTileTokens;
    }
    tile_off[n] = t;
}

// tile_off and blk_off of n experts (n <= 1024): one block, a scan of each expert's tiles and blocks
__global__ void plan_blocks_kernel(const int32_t* __restrict__ bounds, int n, int32_t* __restrict__ tile_off,
                                   int32_t* __restrict__ blk_off) {
    __shared__ int32_t ts[1024], bs[1024];
    const int j = threadIdx.x;
    const int cnt = j < n ? bounds[j + 1] - bounds[j] : 0;
    ts[j] = (cnt + kTileTokens - 1) / kTileTokens;
    bs[j] = (cnt + kBlockTiles * kTileTokens - 1) / (kBlockTiles * kTileTokens);
    __syncthreads();
    for (int o = 1; o < (int) blockDim.x; o <<= 1) {   // inclusive scans
        const int32_t a = j >= o ? ts[j - o] : 0, b = j >= o ? bs[j - o] : 0;
        __syncthreads();
        ts[j] += a;
        bs[j] += b;
        __syncthreads();
    }
    if (j < n) {
        tile_off[j + 1] = ts[j];
        blk_off[j + 1] = bs[j];
    }
    if (j == 0) {
        tile_off[0] = 0;
        blk_off[0] = 0;
    }
}

__global__ void rebase_kernel(const int32_t* __restrict__ bounds, int n, int32_t* __restrict__ out) {
    const int i = threadIdx.x;
    if (i <= n) out[i] = bounds[i] - bounds[0];
}

// ---------------------------------------------------------------------------------------- quantizer

// The widths built: rows of 2560 / 640 values (an MoE's expert input and output), 5120 / 17408 (a dense MLP's); NBL = the
// 16-value blocks per lane.
// One warp per row: token scale g = amax / (6 * 448); per 16 values a UE4M3 scale of block amax / (6 g); codes
// round to nearest even on x / (scale * g).  A sub-block's 8 bytes hold value j in the low nibble of byte j and value
// j + 8 in the high one (the layout of an NVFP4 weight sub-block).  TOKENS: row r goes to the token planes (codes
// [rows][cols / 2], then scales [rows][cols / 16], gs[r]), else to its expert's tile.  The arithmetic is the same.
// An E2M1 code's magnitude (the low three bits)
__device__ __forceinline__ float e2m1_mag(unsigned c) {
    const unsigned m = c & 7u;
    if (m < 2u) return 0.5f * (float) m;
    return ((m & 1u) ? 1.5f : 1.0f) * (float) (1u << ((m >> 1) - 1u));
}

// 16 values times inv into E2M1 codes (the sub-block layout below)
__device__ __forceinline__ void e2m1_block(const float* x, float inv, uint32_t* w) {
    w[0] = w[1] = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        unsigned short c;
        asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %2, %1; cvt.u16.u8 %0, t; }"
            : "=h"(c) : "f"(x[j] * inv), "f"(x[j + 8] * inv));
        w[j >> 2] |= (uint32_t) (c & 0xff) << (8 * (j & 3));
    }
}

// A sub-block's 16 dequantized values (codes w, scale sd_g) subtracted from x: the residual the second term rounds
__device__ __forceinline__ float e2m1_residual(float* x, const uint32_t* w, float sd_g) {
    float m = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const unsigned c = (w[(j & 7) >> 2] >> (8 * (j & 3) + (j >= 8 ? 4 : 0))) & 15u;
        const float d = e2m1_mag(c) * sd_g;
        x[j] -= (c & 8u) ? -d : d;
        m = fmaxf(m, fabsf(x[j]));
    }
    return m;
}

// TWO: each row as two terms, its FP4 rounding and the FP4 rounding of what that leaves (one token scale): the planes
// hold 2 * cols values a row, the k-tiles alternating first term / second term of the same 128 columns, so a product
// over them with each weight k-tile used twice is the product with x to about 8 bits.
template <int NBL, bool TOKENS, bool TWO = false>
__global__ void quantize_kernel(const float* __restrict__ x, const int32_t* __restrict__ ids,
                                const int32_t* __restrict__ bounds, const int32_t* __restrict__ tile_off, int n,
                                uint8_t* __restrict__ a4, float* __restrict__ gs, int cols, int64_t ld, int64_t rows) {
    const int lane = threadIdx.x & 31;
    const int64_t r = (int64_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r >= rows) return;
    int64_t pos = r, tile = 0;
    int rr = 0;
    if (!TOKENS) {
        int lo = 0, hi = n;   // the segment: last j with bounds[j] <= r
        while (hi - lo > 1) {
            const int mid = (lo + hi) >> 1;
            if (bounds[mid] <= r) lo = mid; else hi = mid;
        }
        pos = (int64_t) tile_off[lo] * kTileTokens + (r - bounds[lo]);
        tile = pos / kTileTokens;
        rr = (int) (pos % kTileTokens);
    }
    const int nkt = (cols + kKTile - 1) / kKTile * (TWO ? 2 : 1), nb = cols / 16;
    const float* xr = x + (ids ? ids[r] : r) * ld;
    float bm[NBL];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < NBL; ++i) {
        const int b = lane + 32 * i;
        float m = 0.0f;
        if (b < nb) {
            const float4* p = (const float4*) (xr + 16 * b);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float4 v = p[q];
                m = fmaxf(m, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
            }
        }
        bm[i] = m;
        amax = fmaxf(amax, m);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float g = amax * (1.0f / (6.0f * 448.0f));
    if (lane == 0) gs[pos] = g;
    const float raw_inv_g = g > 0.0f ? 1.0f / g : 0.0f;
    const float inv_g = isfinite(raw_inv_g) ? raw_inv_g : 0.0f;
    uint8_t* tbase = a4 + tile * ((int64_t) nkt * kAStage);
#pragma unroll
    for (int i = 0; i < NBL; ++i) {
        const int b = lane + 32 * i;
        if (b >= nb) continue;
        const __nv_fp8_storage_t sb = __nv_cvt_float_to_fp8(bm[i] * inv_g * (1.0f / 6.0f), __NV_SATFINITE, __NV_E4M3);
        const float sd = __half2float(__half(__nv_cvt_fp8_to_halfraw(sb, __NV_E4M3)));
        const float sd_g = sd * g;
        const float raw_inv = sd_g > 0.0f ? 1.0f / sd_g : 0.0f;
        const float inv = isfinite(raw_inv) ? raw_inv : 0.0f;
        const float4* p = (const float4*) (xr + 16 * b);
        uint32_t w[2] = {0, 0};
        if constexpr (TWO) {
            float x[16];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float4 t = p[q];
                x[4 * q] = t.x; x[4 * q + 1] = t.y; x[4 * q + 2] = t.z; x[4 * q + 3] = t.w;
            }
            e2m1_block(x, inv, w);
            {   // the second term: the residual, its own E4M3 scale per 16 (largest |value| to 6)
                const float m2 = e2m1_residual(x, w, sd_g);
                const __nv_fp8_storage_t sb2 = __nv_cvt_float_to_fp8(m2 * inv_g * (1.0f / 6.0f), __NV_SATFINITE, __NV_E4M3);
                const float sd2 = __half2float(__half(__nv_cvt_fp8_to_halfraw(sb2, __NV_E4M3)));
                const float sd2_g = sd2 * g;
                const float raw_inv2 = sd2_g > 0.0f ? 1.0f / sd2_g : 0.0f;
                const float inv2 = isfinite(raw_inv2) ? raw_inv2 : 0.0f;
                uint32_t w2[2];
                e2m1_block(x, inv2, w2);
                // logical block (2 kt + t) * 8 + b % 8 of 2 nb
                const int b1 = (b / 8) * 16 + (b % 8), b2 = b1 + 8;
                if (TOKENS) {
                    *(uint2*) (a4 + r * cols + 8 * b1) = make_uint2(w[0], w[1]);
                    *(uint2*) (a4 + r * cols + 8 * b2) = make_uint2(w2[0], w2[1]);
                    a4[rows * cols + r * 2 * nb + b1] = (uint8_t) sb;
                    a4[rows * cols + r * 2 * nb + b2] = (uint8_t) sb2;
                } else {
                    const int lb = (b % 8) * 8;
                    uint8_t* k1 = tbase + (int64_t) (2 * (b / 8)) * kAStage;
                    uint8_t* k2 = k1 + kAStage;
                    *(uint2*) (k1 + rr * 64 + swz(rr, lb)) = make_uint2(w[0], w[1]);
                    k1[kACodeBytes + rr * 8 + (b % 8)] = (uint8_t) sb;
                    *(uint2*) (k2 + rr * 64 + swz(rr, lb)) = make_uint2(w2[0], w2[1]);
                    k2[kACodeBytes + rr * 8 + (b % 8)] = (uint8_t) sb2;
                }
                continue;
            }
        } else {
            float v[16];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float4 t = p[q];
                v[4 * q] = t.x * inv; v[4 * q + 1] = t.y * inv; v[4 * q + 2] = t.z * inv; v[4 * q + 3] = t.w * inv;
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                unsigned short c;
                asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %2, %1; cvt.u16.u8 %0, t; }"
                    : "=h"(c) : "f"(v[j]), "f"(v[j + 8]));
                w[j >> 2] |= (uint32_t) (c & 0xff) << (8 * (j & 3));
            }
        }
        if (TOKENS) {
            *(uint2*) (a4 + r * (cols / 2) + 8 * b) = make_uint2(w[0], w[1]);
            a4[rows * (cols / 2) + r * nb + b] = (uint8_t) sb;
            continue;
        }
        const int kt = b / 8, lb = (b % 8) * 8;
        uint8_t* ktp = tbase + (int64_t) kt * kAStage;
        *(uint2*) (ktp + rr * 64 + swz(rr, lb)) = make_uint2(w[0], w[1]);
        ktp[kACodeBytes + rr * 8 + (b % 8)] = (uint8_t) sb;
    }
}

// ----------------------------------------------------------------------------------------------- GEMM

// Single-pass fp16 -> two-term FP4 quantizer for small NBL (e.g. cols = 1152 -> NBL = 3): loads each 16-value sub-block
// once as two uint4s (32 B) into registers, finds row amax, then quantizes both FP4 terms without re-reading global memory.
template <int NBL>
__global__ void quantize_f16_kernel(const uint16_t* __restrict__ x, const int32_t* __restrict__ bounds,
                                    const int32_t* __restrict__ tile_off, int n, uint8_t* __restrict__ a4,
                                    float* __restrict__ gs, int cols, int64_t ld, int64_t rows) {
    const int lane = threadIdx.x & 31;
    const int64_t r = (int64_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r >= rows || r >= bounds[n]) return;   // rows past the sorted ones belong to no expert (pairs left out)
    int lo = 0, hi = n;
    while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (bounds[mid] <= r) lo = mid; else hi = mid;
    }
    const int64_t pos = (int64_t) tile_off[lo] * kTileTokens + (r - bounds[lo]);
    const int64_t tile = pos / kTileTokens;
    const int rr = (int) (pos % kTileTokens);
    const int nkt = cols / kKTile * 2, nb = cols / 16;
    const uint16_t* xr = x + r * ld;

    float xv[NBL][16];
    float bm[NBL];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < NBL; ++i) {
        const int b = lane + 32 * i;
        float m = 0.0f;
        if (b < nb) {
            const uint4* p = (const uint4*) (xr + 16 * b);
            const uint4 u0 = p[0], u1 = p[1];
            const uint32_t words[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                const float2 fv = __half22float2(*reinterpret_cast<const __half2*>(&words[q]));
                xv[i][2 * q] = fv.x;
                xv[i][2 * q + 1] = fv.y;
                m = fmaxf(m, fmaxf(fabsf(fv.x), fabsf(fv.y)));
            }
        }
        bm[i] = m;
        amax = fmaxf(amax, m);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float g = amax * (1.0f / (6.0f * 448.0f));
    if (lane == 0) gs[pos] = g;
    const float raw_inv_g = g > 0.0f ? 1.0f / g : 0.0f;
    const float inv_g = isfinite(raw_inv_g) ? raw_inv_g : 0.0f;
    uint8_t* tbase = a4 + tile * ((int64_t) nkt * kAStage);
#pragma unroll
    for (int i = 0; i < NBL; ++i) {
        const int b = lane + 32 * i;
        if (b >= nb) continue;
        const __nv_fp8_storage_t sb = __nv_cvt_float_to_fp8(bm[i] * inv_g * (1.0f / 6.0f), __NV_SATFINITE, __NV_E4M3);
        const float sd = __half2float(__half(__nv_cvt_fp8_to_halfraw(sb, __NV_E4M3)));
        const float sd_g = sd * g;
        const float raw_inv = sd_g > 0.0f ? 1.0f / sd_g : 0.0f;
        const float inv = isfinite(raw_inv) ? raw_inv : 0.0f;
        uint32_t w[2] = {0, 0};
        e2m1_block(xv[i], inv, w);
        const float m2 = e2m1_residual(xv[i], w, sd_g);
        const __nv_fp8_storage_t sb2 = __nv_cvt_float_to_fp8(m2 * inv_g * (1.0f / 6.0f), __NV_SATFINITE, __NV_E4M3);
        const float sd2 = __half2float(__half(__nv_cvt_fp8_to_halfraw(sb2, __NV_E4M3)));
        const float sd2_g = sd2 * g;
        const float raw_inv2 = sd2_g > 0.0f ? 1.0f / sd2_g : 0.0f;
        const float inv2 = isfinite(raw_inv2) ? raw_inv2 : 0.0f;
        uint32_t w2[2];
        e2m1_block(xv[i], inv2, w2);
        const int lb = (b % 8) * 8;
        uint8_t* k1 = tbase + (int64_t) (2 * (b / 8)) * kAStage;
        uint8_t* k2 = k1 + kAStage;
        *(uint2*) (k1 + rr * 64 + swz(rr, lb)) = make_uint2(w[0], w[1]);
        k1[kACodeBytes + rr * 8 + (b % 8)] = (uint8_t) sb;
        *(uint2*) (k2 + rr * 64 + swz(rr, lb)) = make_uint2(w2[0], w2[1]);
        k2[kACodeBytes + rr * 8 + (b % 8)] = (uint8_t) sb2;
    }
}

struct GemmArgs {
    const uint8_t* blob[kMaxExperts];
    size_t w_off;
    int w_cols, half;
    const uint8_t* a4;        // tiles, or (GATHER) the token codes
    const uint8_t* a4s;       // GATHER: the token sub-block scales
    const float* gs;
    const int32_t* bounds;
    const int32_t* tile_off;
    const int32_t* ids;       // GATHER: row -> token
    float* dst;
    uint16_t* dst_f16;
    int64_t ld_dst;
    size_t tail_off;          // SWIGLU: the blob's {s_gate, s_up, s_down, -}
    const uint8_t* base;      // FLAT: expert j's blob at base + j * stride (or blobs[j]), its blocks from blk_off
    size_t stride;
    const unsigned long long* blobs;
    const int32_t* blk_off;
    int n;
    const int32_t* dst_ids;   // FLAT: output row r to dst row dst_ids[r] (null: r)
    float cg, cu;             // FLAT SWIGLU: the softcaps (each applied when > 0)
    int gelu;                 // FLAT SWIGLU: the tail-scaled GELU gate in place of the SwiGLU
};

// The weight row behind B tile row sr of block rb: 128 consecutive rows, or (SWIGLU) the block's 64 gate rows and the
// 64 up rows of the same outputs
template <bool SWIGLU> __device__ __forceinline__ int64_t b_row(int rb, int sr, int half) {
    if (!SWIGLU) return (int64_t) rb * kRowsPerBlock + sr;
    return sr < 64 ? (int64_t) rb * 64 + sr : (int64_t) half + rb * 64 + (sr - 64);
}
// The first B tile row of a warp's n-tile r: with SWIGLU, tiles 0-1 are gate rows and 2-3 the up rows of the same outputs
template <bool SWIGLU> __device__ __forceinline__ int b_tile_row(int warp_n, int r) {
    if (!SWIGLU) return warp_n * 32 + r * 8;
    return r < 2 ? warp_n * 16 + r * 8 : 64 + warp_n * 16 + (r - 2) * 8;
}

// mmq::swiglu's tail-scaled SwiGLU in the instructions of its fast-math build (FTZ products, ex2 and rcp
// approximations, the division as a reciprocal product), so h is the same bits
// mmq::swiglu's softcapped form: (cg tanh(g / cg)) sigmoid(g) (cu tanh(u / cu)) s_down, g and u tail-scaled
// (either softcap <= 0 leaves that branch uncapped)
__device__ __forceinline__ float swiglu_cap(float g, float u, float4 s, float cg, float cu, float inv_cg, float inv_cu) {
    g *= s.x;
    u *= s.y;
    float gc = g;
    if (cg > 0.0f) {
        float tg;
        asm("tanh.approx.f32 %0, %1;" : "=f"(tg) : "f"(g * inv_cg));
        gc = cg * tg;
    }
    float uc = u;
    if (cu > 0.0f) {
        float tu;
        asm("tanh.approx.f32 %0, %1;" : "=f"(tu) : "f"(u * inv_cu));
        uc = cu * tu;
    }
    float t, e, d, sig;
    asm("mul.rn.ftz.f32 %0, %1, 0fBFB8AA3B;" : "=f"(t) : "f"(g));
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(t));
    asm("add.rn.ftz.f32 %0, %1, 0f3F800000;" : "=f"(d) : "f"(e));
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(sig) : "f"(d));
    return __fmul_rn(__fmul_rn(gc, sig), uc) * s.z;
}

// The tail-scaled GELU gate (tanh form): gelu_tanh(g s_gate) (u s_up) s_down
__device__ __forceinline__ float gelu_gate(float g, float u, float4 s) {
    g *= s.x;
    return 0.5f * g * (1.0f + tanhf(0.7978845608028654f * (g + 0.044715f * g * g * g))) * (u * s.y) * s.z;
}
// The FLAT gate/up product's activation: the GELU gate or the softcapped SwiGLU
__device__ __forceinline__ float flat_gate(const GemmArgs& a, float g, float u, float4 s, float inv_cg, float inv_cu) {
    return a.gelu ? gelu_gate(g, u, s) : swiglu_cap(g, u, s, a.cg, a.cu, inv_cg, inv_cu);
}

__device__ __forceinline__ float swiglu_tail(float g, float u, float4 s) {
    float g1, u1, t, e, d, rc, q, p, h;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(g1) : "f"(g), "f"(s.x));
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(u1) : "f"(u), "f"(s.y));
    asm("mul.rn.ftz.f32 %0, %1, 0fBFB8AA3B;" : "=f"(t) : "f"(g1));
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(t));
    asm("add.rn.ftz.f32 %0, %1, 0f3F800000;" : "=f"(d) : "f"(e));
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(rc) : "f"(d));
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(q) : "f"(g1), "f"(rc));
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(p) : "f"(u1), "f"(q));
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(h) : "f"(s.z), "f"(p));
    return h;
}

// One pipeline stage: the block's activation tiles (runs of the tiled planes, or GATHER: each row's token codes, their
// 16-byte chunks swizzled as quantize() places them, and its scales) and the block's weight rows
template <bool GATHER, bool SWIGLU, int TM = 1>
__device__ __forceinline__ void load_stage(int stage, int kt, uint8_t* smem, const uint8_t* a_src, size_t tile_stride,
                                           int ntiles, const uint8_t* const* a_code, const int* a_dst,
                                           const uint8_t* a_scale, int s_dst, const uint8_t* w_src, size_t row_bytes,
                                           int rb, int half) {
    // kt is a weight k-tile; term u of it is activation k-tile TM kt + u, at as + u * kBlockTiles * kAStage
    uint8_t* as0 = smem + stage * TM * kBlockTiles * kAStage;
    uint8_t* bs = smem + stages<TM>() * TM * kBlockTiles * kAStage + stage * kBStage;
#pragma unroll
    for (int u = 0; u < TM; ++u) {
        uint8_t* as = as0 + u * kBlockTiles * kAStage;
        const int ka = TM * kt + u;
        if (GATHER) {   // chunk i of a thread lies in tile i, its scale row in tile threadIdx.x / 64
#pragma unroll
            for (int i = 0; i < kBlockTiles * kTileTokens * 4 / kThreads; ++i)
                if (i < ntiles) cp_async16(as + a_dst[i], a_code[i] + (size_t) ka * 64);
            if ((int) threadIdx.x < ntiles * kTileTokens) cp_async8(as + s_dst, a_scale + (size_t) ka * 8);
        } else {
            for (int i = threadIdx.x; i < ntiles * (kAStage / 16); i += kThreads) {
                const int t = i / (kAStage / 16), c = i % (kAStage / 16);
                cp_async16(as + t * kAStage + 16 * c, a_src + t * tile_stride + (size_t) ka * kAStage + 16 * c);
            }
        }
    }
    // 256 threads cover 128 B rows (2 threads per row): 4x cp_async16 (.cg) + 1x cp_async8, shifted by 8B in smem when src is 8 mod 16
    {
        const int row = (int) threadIdx.x >> 1, part = (int) threadIdx.x & 1;
        const uint8_t* src = w_src + (size_t) b_row<SWIGLU>(rb, row, half) * row_bytes + (size_t) kt * kBRowBytes;
        const int odd = (int) (((uintptr_t) src >> 3) & 1u);
        uint8_t* dst = bs + row * kBRowStride;
        if (!odd) {
            if (part == 0) {
                cp_async16(dst + 0, src + 0);
                cp_async16(dst + 16, src + 16);
                cp_async16(dst + 32, src + 32);
            } else {
                cp_async16(dst + 48, src + 48);
                cp_async8(dst + 64, src + 64);
            }
        } else {
            if (part == 0) {
                cp_async8(dst + 8, src + 0);
                cp_async16(dst + 16, src + 8);
                cp_async16(dst + 32, src + 24);
            } else {
                cp_async16(dst + 48, src + 40);
                cp_async16(dst + 64, src + 56);
            }
        }
    }
}

// One block: up to two consecutive 64-token tiles of an expert x 128 weight rows; each warp owns one tile's 64 tokens
// and 32 rows, so a weight fragment feeds four token groups.  GATHER: a row's activations are its token's (token planes,
// row -> token from ids); SWIGLU: the gate/up product's outputs leave as the tail-scaled SwiGLU of each gate/up pair.
// Every output keeps the same MMA chain over K as the plain product of one tile, so the products are the same bits.
// TWO (FLAT only): the activations are two terms a row (quantize's TWO planes: 2 * w_cols values), each weight k-tile
// used for both.
template <bool GATHER, bool SWIGLU, bool FLAT = false, bool TWO = false>
__global__ __launch_bounds__(kThreads, 1) void gemm_kernel(const GemmArgs a) {
    constexpr int TM = TWO ? 2 : 1, NS = stages<TM>();
    extern __shared__ __align__(1024) uint8_t smem[];
    __shared__ int32_t tok_s[kBlockTiles * kTileTokens];
    int j = blockIdx.z, tp = blockIdx.y;
    const int rb = blockIdx.x;
    if (FLAT) {   // block y of the layer's blocks: its expert (the last j with blk_off[j] <= y) and its place there
        const int y = blockIdx.y;
        if (y >= a.blk_off[a.n]) return;
        int lo = 0, hi = a.n;
        while (hi - lo > 1) {
            const int mid = (lo + hi) >> 1;
            if (a.blk_off[mid] <= y) lo = mid; else hi = mid;
        }
        j = lo;
        tp = y - a.blk_off[lo];
    }
    const int cnt = a.bounds[j + 1] - a.bounds[j];
    const int first = tp * kBlockTiles * kTileTokens;   // the block's first row inside the expert
    if (first >= cnt) return;
    const int ntiles = first + kTileTokens < cnt ? 2 : 1;
    const int nkt = (a.w_cols + kKTile - 1) / kKTile;   // weight k-tiles (TM activation k-tiles each); the last may be half
    const size_t row_bytes = (size_t) (a.w_cols / 64) * 36;
    const size_t tile_stride = (size_t) TM * nkt * kAStage;
    const uint8_t* blob = !FLAT ? a.blob[j] : a.blobs ? (const uint8_t*) a.blobs[j] : a.base + (size_t) j * a.stride;
    const uint8_t* w_src = blob + a.w_off;
    const int64_t tile0 = GATHER ? 0 : a.tile_off[j] + kBlockTiles * tp;
    const uint8_t* a_src = GATHER ? nullptr : a.a4 + tile0 * (int64_t) tile_stride;
    const int64_t brow0 = a.bounds[j] + first;          // the block's first absolute row
    const uint8_t* a_code[kBlockTiles * kTileTokens * 4 / kThreads] = {};
    int a_dst[kBlockTiles * kTileTokens * 4 / kThreads] = {};
    const uint8_t* a_scale = nullptr;
    int s_dst = 0;
    if (GATHER) {
        // a padded row reads the block's first token: its outputs are never stored
        if (threadIdx.x < kBlockTiles * kTileTokens)
            tok_s[threadIdx.x] = a.ids[brow0 + (first + (int) threadIdx.x < cnt ? (int) threadIdx.x : 0)];
        __syncthreads();
#pragma unroll
        for (int i = 0; i < kBlockTiles * kTileTokens * 4 / kThreads; ++i) {
            const int c = (int) threadIdx.x + i * kThreads, r = c >> 2, q = c & 3, rr = r % kTileTokens;
            a_code[i] = a.a4 + (size_t) tok_s[r] * (size_t) (TM * a.w_cols / 2) + 16 * q;
            a_dst[i] = (r / kTileTokens) * kAStage + rr * 64 + ((q ^ ((rr >> 1) & 3)) << 4);
        }
        if (threadIdx.x < kBlockTiles * kTileTokens) {
            const int r = threadIdx.x;
            a_scale = a.a4s + (size_t) tok_s[r] * (size_t) (TM * a.w_cols / 16);
            s_dst = (r / kTileTokens) * kAStage + kACodeBytes + (r % kTileTokens) * 8;
        }
    }

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int warp_m = warp >> 2, warp_n = warp & 3;   // tile x 4 row quarters
    const bool live = warp_m < ntiles;
    const int group = lane >> 2, tig = lane & 3;
    const int a_row = (lane & 7) + (((lane >> 3) & 1) << 3);
    const int a_col = (lane >> 4) * 16;
    const int sfa_row = ((lane & 1) << 3) | (lane >> 2);

    float acc[4][4][4] = {};
#pragma unroll
    for (int s = 0; s < NS - 1; ++s) {
        if (s < nkt)
            load_stage<GATHER, SWIGLU, TM>(s, s, smem, a_src, tile_stride, ntiles, a_code, a_dst, a_scale, s_dst, w_src,
                                       row_bytes, rb, a.half);
        cp_commit();
    }
    for (int kt = 0; kt < nkt; ++kt) {
        cp_wait<NS - 2>();
        __syncthreads();
        if (kt + NS - 1 < nkt)
            load_stage<GATHER, SWIGLU, TM>((kt + NS - 1) % NS, kt + NS - 1, smem, a_src, tile_stride, ntiles,
                                       a_code, a_dst, a_scale, s_dst, w_src, row_bytes, rb, a.half);
        cp_commit();
        if (!live) continue;
        const int stage = kt % NS;
        const uint8_t* as0 = smem + stage * TM * kBlockTiles * kAStage + warp_m * kAStage;
        const uint8_t* bs = smem + NS * TM * kBlockTiles * kAStage + stage * kBStage;
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            if (s == 1 && kt == nkt - 1 && (a.w_cols & (kKTile - 1))) break;   // a half last k-tile
            unsigned b0[4], b1[4], sb[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int sr = b_tile_row<SWIGLU>(warp_n, r) + group;
                const int odd = (int) (((uintptr_t) (w_src + (size_t) b_row<SWIGLU>(rb, sr, a.half) * row_bytes + (size_t) kt * kBRowBytes) >> 3) & 1u);
                const uint8_t* p = bs + sr * kBRowStride + (odd << 3) + s * 36;
                b0[r] = *(const uint32_t*) (p + 4 + 4 * tig);
                b1[r] = *(const uint32_t*) (p + 20 + 4 * tig);
                // scale bytes: bit 7 ignored, 0x7F (NaN in hardware) as 0
                unsigned v = *(const uint32_t*) p & 0x7f7f7f7fu;
                sb[r] = v & ~__vcmpeq4(v, 0x7f7f7f7fu);
            }
#pragma unroll
            for (int u = 0; u < TM; ++u) {   // the weight fragments serve both terms
                const uint8_t* as = as0 + u * kBlockTiles * kAStage;
                unsigned af[4][4], asc[4];
#pragma unroll
                for (int mt = 0; mt < 4; ++mt) {
                    const int row = mt * 16 + a_row;
                    ldmatrix_x4(af[mt][0], af[mt][1], af[mt][2], af[mt][3],
                                smem_addr(as + row * 64 + swz(row, s * 32 + a_col)));
                    asc[mt] = *(const uint32_t*) (as + kACodeBytes + (mt * 16 + sfa_row) * 8 + 4 * s);
                }
#pragma unroll
                for (int mt = 0; mt < 4; ++mt)
#pragma unroll
                    for (int r = 0; r < 4; ++r) mma_fp4(acc[mt][r], af[mt], b0[r], b1[r], asc[mt], sb[r]);
            }
        }
    }
    cp_wait<0>();
    if (!live) return;

    const int tl = kBlockTiles * tp + warp_m;   // this warp's tile inside the expert
    const int64_t tile = tile0 + warp_m;
    const int64_t row0 = a.bounds[j] + (int64_t) tl * kTileTokens;
    float4 tail = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (SWIGLU) tail = *(const float4*) (blob + a.tail_off);
    const float inv_cg = (SWIGLU && a.cg > 0.0f) ? 1.0f / a.cg : 0.0f;
    const float inv_cu = (SWIGLU && a.cu > 0.0f) ? 1.0f / a.cu : 0.0f;
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int t = mt * 16 + group + 8 * h;
            if (tl * kTileTokens + t >= cnt) continue;
            const float g = GATHER ? a.gs[tok_s[warp_m * kTileTokens + t]] : a.gs[tile * kTileTokens + t];
            if (FLAT && SWIGLU) {
                float* out = a.dst + (row0 + t) * a.ld_dst + rb * 64 + warp_n * 16 + 2 * tig;
#pragma unroll
                for (int r = 0; r < 2; ++r)
                    *(float2*) (out + 8 * r) =
                        make_float2(flat_gate(a, acc[mt][r][2 * h] * g, acc[mt][r + 2][2 * h] * g, tail, inv_cg, inv_cu),
                                    flat_gate(a, acc[mt][r][2 * h + 1] * g, acc[mt][r + 2][2 * h + 1] * g, tail, inv_cg, inv_cu));
            } else if (FLAT) {
                const int64_t orow = a.dst_ids ? a.dst_ids[row0 + t] : row0 + t;
                float* out = a.dst + orow * a.ld_dst + rb * kRowsPerBlock + warp_n * 32 + 2 * tig;
#pragma unroll
                for (int r = 0; r < 4; ++r) *(float2*) (out + 8 * r) = make_float2(acc[mt][r][2 * h] * g, acc[mt][r][2 * h + 1] * g);
            } else if (SWIGLU) {
                // the gate/up product's values as the plain product stores them, then mmq::swiglu's arithmetic
                float* out = a.dst + (row0 + t) * a.ld_dst + rb * 64 + warp_n * 16 + 2 * tig;
#pragma unroll
                for (int r = 0; r < 2; ++r)
                    *(float2*) (out + 8 * r) =
                        make_float2(swiglu_tail(__fmul_rn(acc[mt][r][2 * h], g), __fmul_rn(acc[mt][r + 2][2 * h], g), tail),
                                    swiglu_tail(__fmul_rn(acc[mt][r][2 * h + 1], g), __fmul_rn(acc[mt][r + 2][2 * h + 1], g), tail));
            } else {
                float* out = a.dst + (row0 + t) * a.ld_dst + rb * kRowsPerBlock + warp_n * 32 + 2 * tig;
#pragma unroll
                for (int r = 0; r < 4; ++r) *(float2*) (out + 8 * r) = make_float2(acc[mt][r][2 * h] * g, acc[mt][r][2 * h + 1] * g);
            }
        }
}

// 256 weight rows x 128 tokens (2 tiles), 2 activation terms, 2 pipeline stages (76 KiB smem): halves activation
// loads across N-blocks and doubles MMA density per stage (8 warps, each 64 tokens x 64 rows = 4x8 MMA tiles).
constexpr int kRows256 = 256;
constexpr int kBRowStride256 = 112;   // a k-tile's 72 bytes from any 4-byte offset in whole 16-byte chunks; 28 words: no bank conflicts
constexpr int kBStage256 = kRows256 * kBRowStride256;                // 28672
constexpr int kSmem256 = 2 * (2 * kBlockTiles * kAStage + kBStage256); // 94208

template <bool SWIGLU> __device__ __forceinline__ int64_t b_row256(int rb, int sr, int half) {
    if (!SWIGLU) return (int64_t) rb * kRows256 + sr;
    return sr < 128 ? (int64_t) rb * 128 + sr : (int64_t) half + rb * 128 + (sr - 128);
}
template <bool SWIGLU> __device__ __forceinline__ int b_tile_row256(int warp_n, int r) {
    if (!SWIGLU) return warp_n * 64 + r * 8;
    return r < 4 ? warp_n * 32 + r * 8 : 128 + warp_n * 32 + (r - 4) * 8;
}

template <bool GATHER, bool SWIGLU>
__device__ __forceinline__ void load_stage256(int stage, int kt, uint8_t* smem, const uint8_t* a_src, size_t tile_stride,
                                              int ntiles, const uint8_t* const* a_code, const int* a_dst,
                                              const uint8_t* a_scale, int s_dst, const uint8_t* w_src, size_t row_bytes,
                                              int rb, int half) {
    uint8_t* as0 = smem + stage * 2 * kBlockTiles * kAStage;
    uint8_t* bs = smem + 2 * 2 * kBlockTiles * kAStage + stage * kBStage256;
#pragma unroll
    for (int u = 0; u < 2; ++u) {
        uint8_t* as = as0 + u * kBlockTiles * kAStage;
        const int ka = 2 * kt + u;
        if (GATHER) {
#pragma unroll
            for (int i = 0; i < kBlockTiles * kTileTokens * 4 / kThreads; ++i)
                if (i < ntiles) cp_async16(as + a_dst[i], a_code[i] + (size_t) ka * 64);
            if ((int) threadIdx.x < ntiles * kTileTokens) cp_async8(as + s_dst, a_scale + (size_t) ka * 8);
        } else {
            for (int i = threadIdx.x; i < ntiles * (kAStage / 16); i += kThreads) {
                const int t = i / (kAStage / 16), c = i % (kAStage / 16);
                cp_async16(as + t * kAStage + 16 * c, a_src + t * tile_stride + (size_t) ka * kAStage + 16 * c);
            }
        }
    }
    // A row's k-tile as the 16-byte chunks that hold it (rows 4-byte aligned: five or six), eight lanes a row, coalesced
#pragma unroll
    for (int p = 0; p < kRows256 / (kThreads / 8); ++p) {
        const int row = p * (kThreads / 8) + (int) (threadIdx.x >> 3), c = (int) (threadIdx.x & 7);
        const uintptr_t src = (uintptr_t) (w_src + (size_t) b_row256<SWIGLU>(rb, row, half) * row_bytes + (size_t) kt * kBRowBytes);
        const int shift = (int) (src & 15u);
        if (c < (shift + kBRowBytes + 15) / 16) cp_async16(bs + row * kBRowStride256 + 16 * c, (const void*) (src - shift + 16 * c));
    }
}

template <bool GATHER, bool SWIGLU>
__global__ __launch_bounds__(kThreads, 1) void gemm_kernel_256(const GemmArgs a) {
    extern __shared__ __align__(1024) uint8_t smem[];
    __shared__ int32_t tok_s[kBlockTiles * kTileTokens];
    const int rb = blockIdx.x, y = blockIdx.y;
    if (y >= a.blk_off[a.n]) return;
    int lo = 0, hi = a.n;
    while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (a.blk_off[mid] <= y) lo = mid; else hi = mid;
    }
    const int j = lo, tp = y - a.blk_off[lo];
    const int cnt = a.bounds[j + 1] - a.bounds[j];
    const int first = tp * kBlockTiles * kTileTokens;
    if (first >= cnt) return;
    const int ntiles = first + kTileTokens < cnt ? 2 : 1;
    const int nkt = (a.w_cols + kKTile - 1) / kKTile;   // the last may be half
    const size_t row_bytes = (size_t) (a.w_cols / 64) * 36;
    const size_t tile_stride = (size_t) 2 * nkt * kAStage;
    const uint8_t* blob = a.blobs ? (const uint8_t*) a.blobs[j] : a.base + (size_t) j * a.stride;
    const uint8_t* w_src = blob + a.w_off;
    const int64_t tile0 = GATHER ? 0 : a.tile_off[j] + kBlockTiles * tp;
    const uint8_t* a_src = GATHER ? nullptr : a.a4 + tile0 * (int64_t) tile_stride;
    const int64_t brow0 = a.bounds[j] + first;
    const uint8_t* a_code[kBlockTiles * kTileTokens * 4 / kThreads] = {};
    int a_dst[kBlockTiles * kTileTokens * 4 / kThreads] = {};
    const uint8_t* a_scale = nullptr;
    int s_dst = 0;
    if (GATHER) {
        if (threadIdx.x < kBlockTiles * kTileTokens)
            tok_s[threadIdx.x] = a.ids[brow0 + (first + (int) threadIdx.x < cnt ? (int) threadIdx.x : 0)];
        __syncthreads();
#pragma unroll
        for (int i = 0; i < kBlockTiles * kTileTokens * 4 / kThreads; ++i) {
            const int c = (int) threadIdx.x + i * kThreads, r = c >> 2, q = c & 3, rr = r % kTileTokens;
            a_code[i] = a.a4 + (size_t) tok_s[r] * (size_t) (2 * a.w_cols / 2) + 16 * q;
            a_dst[i] = (r / kTileTokens) * kAStage + rr * 64 + ((q ^ ((rr >> 1) & 3)) << 4);
        }
        if (threadIdx.x < kBlockTiles * kTileTokens) {
            const int r = threadIdx.x;
            a_scale = a.a4s + (size_t) tok_s[r] * (size_t) (2 * a.w_cols / 16);
            s_dst = (r / kTileTokens) * kAStage + kACodeBytes + (r % kTileTokens) * 8;
        }
    }

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int warp_m = warp >> 2, warp_n = warp & 3;
    const bool live = warp_m < ntiles;
    const int group = lane >> 2, tig = lane & 3;
    const int a_row = (lane & 7) + (((lane >> 3) & 1) << 3);
    const int a_col = (lane >> 4) * 16;
    const int sfa_row = ((lane & 1) << 3) | (lane >> 2);

    int row_shift[8];   // each fragment row's weight row start, mod 16
#pragma unroll
    for (int r = 0; r < 8; ++r)
        row_shift[r] = (int) (((uintptr_t) w_src + (uintptr_t) b_row256<SWIGLU>(rb, b_tile_row256<SWIGLU>(warp_n, r) + group, a.half) *
                               row_bytes) & 15u);

    float acc[4][8][4] = {};
    load_stage256<GATHER, SWIGLU>(0, 0, smem, a_src, tile_stride, ntiles, a_code, a_dst, a_scale, s_dst, w_src, row_bytes, rb,
                                  a.half);
    cp_commit();
    for (int kt = 0; kt < nkt; ++kt) {
        const bool has_next = kt + 1 < nkt;
        if (has_next) {
            load_stage256<GATHER, SWIGLU>((kt + 1) & 1, kt + 1, smem, a_src, tile_stride, ntiles, a_code, a_dst, a_scale, s_dst,
                                          w_src, row_bytes, rb, a.half);
            cp_commit();
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();
        if (live) {
            const int stage = kt & 1;
            const uint8_t* as0 = smem + stage * 2 * kBlockTiles * kAStage + warp_m * kAStage;
            const uint8_t* bs = smem + 2 * 2 * kBlockTiles * kAStage + stage * kBStage256;
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                if (s == 1 && kt == nkt - 1 && (a.w_cols & (kKTile - 1))) break;   // a half last k-tile
                unsigned b0[8], b1[8], sb[8];
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const int sr = b_tile_row256<SWIGLU>(warp_n, r) + group;
                    // the row's k-tile sits at its source address's offset in its 16-byte chunk (a k-tile is 72 bytes)
                    const uint8_t* p = bs + sr * kBRowStride256 + ((row_shift[r] + 8 * kt) & 15) + s * 36;
                    b0[r] = *(const uint32_t*) (p + 4 + 4 * tig);
                    b1[r] = *(const uint32_t*) (p + 20 + 4 * tig);
                    unsigned v = *(const uint32_t*) p & 0x7f7f7f7fu;
                    sb[r] = v & ~__vcmpeq4(v, 0x7f7f7f7fu);
                }
#pragma unroll
                for (int u = 0; u < 2; ++u) {
                    const uint8_t* as = as0 + u * kBlockTiles * kAStage;
                    unsigned af[4][4], asc[4];
#pragma unroll
                    for (int mt = 0; mt < 4; ++mt) {
                        const int row = mt * 16 + a_row;
                        ldmatrix_x4(af[mt][0], af[mt][1], af[mt][2], af[mt][3],
                                    smem_addr(as + row * 64 + swz(row, s * 32 + a_col)));
                        asc[mt] = *(const uint32_t*) (as + kACodeBytes + (mt * 16 + sfa_row) * 8 + 4 * s);
                    }
#pragma unroll
                    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
                        for (int r = 0; r < 8; ++r) mma_fp4(acc[mt][r], af[mt], b0[r], b1[r], asc[mt], sb[r]);
                }
            }
        }
        if (has_next) __syncthreads();
    }
    if (!live) return;

    const int tl = kBlockTiles * tp + warp_m;
    const int64_t tile = tile0 + warp_m;
    const int64_t row0 = a.bounds[j] + (int64_t) tl * kTileTokens;
    float4 tail = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (SWIGLU) tail = *(const float4*) (blob + a.tail_off);
    const float inv_cg = (SWIGLU && a.cg > 0.0f) ? 1.0f / a.cg : 0.0f;
    const float inv_cu = (SWIGLU && a.cu > 0.0f) ? 1.0f / a.cu : 0.0f;
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int t = mt * 16 + group + 8 * h;
            if (tl * kTileTokens + t >= cnt) continue;
            const float g = GATHER ? a.gs[tok_s[warp_m * kTileTokens + t]] : a.gs[tile * kTileTokens + t];
            if (SWIGLU && a.dst_f16) {
                uint16_t* out = a.dst_f16 + (row0 + t) * a.ld_dst + rb * 128 + warp_n * 32 + 2 * tig;
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const __half2 h2 = __floats2half2_rn(
                        flat_gate(a, acc[mt][r][2 * h] * g, acc[mt][r + 4][2 * h] * g, tail, inv_cg, inv_cu),
                        flat_gate(a, acc[mt][r][2 * h + 1] * g, acc[mt][r + 4][2 * h + 1] * g, tail, inv_cg, inv_cu));
                    *reinterpret_cast<uint32_t*>(out + 8 * r) = *reinterpret_cast<const uint32_t*>(&h2);
                }
            } else if (SWIGLU) {
                float* out = a.dst + (row0 + t) * a.ld_dst + rb * 128 + warp_n * 32 + 2 * tig;
#pragma unroll
                for (int r = 0; r < 4; ++r)
                    *(float2*) (out + 8 * r) =
                        make_float2(flat_gate(a, acc[mt][r][2 * h] * g, acc[mt][r + 4][2 * h] * g, tail, inv_cg, inv_cu),
                                    flat_gate(a, acc[mt][r][2 * h + 1] * g, acc[mt][r + 4][2 * h + 1] * g, tail, inv_cg, inv_cu));
            } else if (a.dst_f16) {
                const int64_t orow = a.dst_ids ? a.dst_ids[row0 + t] : row0 + t;
                uint16_t* out = a.dst_f16 + orow * a.ld_dst + rb * kRows256 + warp_n * 64 + 2 * tig;
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const __half2 h2 = __floats2half2_rn(acc[mt][r][2 * h] * g, acc[mt][r][2 * h + 1] * g);
                    *reinterpret_cast<uint32_t*>(out + 8 * r) = *reinterpret_cast<const uint32_t*>(&h2);
                }
            } else {
                const int64_t orow = a.dst_ids ? a.dst_ids[row0 + t] : row0 + t;
                float* out = a.dst + orow * a.ld_dst + rb * kRows256 + warp_n * 64 + 2 * tig;
#pragma unroll
                for (int r = 0; r < 8; ++r) *(float2*) (out + 8 * r) = make_float2(acc[mt][r][2 * h] * g, acc[mt][r][2 * h + 1] * g);
            }
        }
}

__global__ void tails_kernel(GemmArgs a, size_t off, int n, float* dst) {
    const int j = threadIdx.x;
    if (j < n) ((float4*) dst)[j] = *(const float4*) (a.blob[j] + off);
}

}  // namespace

size_t a4_bytes(int64_t rows, int64_t cols, int experts) {
    return (size_t) ((rows + kTileTokens - 1) / kTileTokens + experts) * (size_t) ((cols + kKTile - 1) / kKTile) * kAStage;
}
size_t scale_count(int64_t rows, int experts) {
    return (size_t) ((rows + kTileTokens - 1) / kTileTokens + experts) * kTileTokens;
}

void plan(const int32_t* bounds, int n, int32_t* tile_off, void* stream) {
    plan_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(bounds, n, tile_off);
    ck(cudaGetLastError(), "plan");
}

void plan_blocks(const int32_t* bounds, int n, int32_t* tile_off, int32_t* blk_off, void* stream) {
    if (n <= 0 || n > 1024) { basalt::log::error("basalt", "prefill fp4: %d experts to plan", n); std::exit(1); }
    plan_blocks_kernel<<<1, (n + 31) / 32 * 32, 0, (cudaStream_t) stream>>>(bounds, n, tile_off, blk_off);
    ck(cudaGetLastError(), "plan blocks");
}

void rebase(const int32_t* bounds, int n, int32_t* out, void* stream) {
    rebase_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(bounds, n, out);
    ck(cudaGetLastError(), "rebase");
}

void quantize(const float* x, const int32_t* ids, const int32_t* bounds, const int32_t* tile_off, int n, void* a4,
              float* gs, int64_t cols, int64_t ld, int64_t rows, void* stream, bool two) {
    if (rows <= 0) return;
    const unsigned grid = (unsigned) ((rows + 7) / 8);
    const cudaStream_t s = (cudaStream_t) stream;
    if (two) {
        if (cols == 1152) quantize_kernel<3, false, true><<<grid, 256, 0, s>>>(x, ids, bounds, tile_off, n, (uint8_t*) a4, gs, (int) cols, ld, rows);
        else if (cols == 704) quantize_kernel<2, false, true><<<grid, 256, 0, s>>>(x, ids, bounds, tile_off, n, (uint8_t*) a4, gs, (int) cols, ld, rows);
        else { basalt::log::error("basalt", "prefill fp4: %lld columns (two terms)", (long long) cols); std::exit(1); }
    } else if (cols == 2560) quantize_kernel<5, false><<<grid, 256, 0, s>>>(x, ids, bounds, tile_off, n, (uint8_t*) a4, gs, (int) cols, ld, rows);
    else if (cols == 640) quantize_kernel<2, false><<<grid, 256, 0, s>>>(x, ids, bounds, tile_off, n, (uint8_t*) a4, gs, (int) cols, ld, rows);
    else if (cols == 5120) quantize_kernel<10, false><<<grid, 256, 0, s>>>(x, ids, bounds, tile_off, n, (uint8_t*) a4, gs, (int) cols, ld, rows);
    else if (cols == 17408) quantize_kernel<34, false><<<grid, 256, 0, s>>>(x, ids, bounds, tile_off, n, (uint8_t*) a4, gs, (int) cols, ld, rows);
    else { basalt::log::error("basalt", "prefill fp4: %lld columns", (long long) cols); std::exit(1); }
    ck(cudaGetLastError(), "quantize");
}

void quantize_f16(const uint16_t* x, const int32_t* bounds, const int32_t* tile_off, int n, void* a4,
                  float* gs, int64_t cols, int64_t ld, int64_t rows, void* stream) {
    if (rows <= 0) return;
    const unsigned grid = (unsigned) ((rows + 7) / 8);
    const cudaStream_t s = (cudaStream_t) stream;
    if (cols == 1152) quantize_f16_kernel<3><<<grid, 256, 0, s>>>(x, bounds, tile_off, n, (uint8_t*) a4, gs, (int) cols, ld, rows);
    else { basalt::log::error("basalt", "prefill fp4: %lld fp16 columns", (long long) cols); std::exit(1); }
    ck(cudaGetLastError(), "quantize_f16");
}

size_t token_bytes(int64_t tokens, int64_t cols) { return (size_t) tokens * (size_t) (cols / 2 + cols / 16); }

void quantize_tokens(const float* x, int64_t tokens, int64_t cols, int64_t ld, void* a4, float* gs, void* stream,
                     bool two) {
    if (tokens <= 0) return;
    const unsigned grid = (unsigned) ((tokens + 7) / 8);
    const cudaStream_t s = (cudaStream_t) stream;
    if (two) {
        if (cols == 2304) quantize_kernel<5, true, true><<<grid, 256, 0, s>>>(x, nullptr, nullptr, nullptr, 0, (uint8_t*) a4, gs, (int) cols, ld, tokens);
        else if (cols == 2816) quantize_kernel<6, true, true><<<grid, 256, 0, s>>>(x, nullptr, nullptr, nullptr, 0, (uint8_t*) a4, gs, (int) cols, ld, tokens);
        else { basalt::log::error("basalt", "prefill fp4: %lld token columns (two terms)", (long long) cols); std::exit(1); }
    } else if (cols == 2560) quantize_kernel<5, true><<<grid, 256, 0, s>>>(x, nullptr, nullptr, nullptr, 0, (uint8_t*) a4, gs, (int) cols, ld, tokens);
    else if (cols == 640) quantize_kernel<2, true><<<grid, 256, 0, s>>>(x, nullptr, nullptr, nullptr, 0, (uint8_t*) a4, gs, (int) cols, ld, tokens);
    else { basalt::log::error("basalt", "prefill fp4: %lld columns", (long long) cols); std::exit(1); }
    ck(cudaGetLastError(), "quantize tokens");
}

void gather_tails(const Product& p, size_t tail_off, float* dst, void* stream) {
    GemmArgs a{};
    for (int j = 0; j < p.n; ++j) a.blob[j] = p.blob[j];
    tails_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(a, tail_off, p.n, dst);
    ck(cudaGetLastError(), "tails");
}

// the shared-memory opt-in is per device and per kernel (a layer split runs these on several, from different threads)
template <bool GATHER, bool SWIGLU, bool FLAT = false, bool TWO = false> void launch(const GemmArgs& a, dim3 grid, cudaStream_t s) {
    static std::atomic<bool> configured[64];
    int dev = 0;
    ck(cudaGetDevice(&dev), "device");
    if (dev < 0 || dev >= 64) ck(cudaErrorInvalidDevice, "device range");
    if (!configured[dev].load(std::memory_order_acquire)) {
        ck(cudaFuncSetAttribute(gemm_kernel<GATHER, SWIGLU, FLAT, TWO>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                smem_bytes<TWO ? 2 : 1>()), "smem");
        configured[dev].store(true, std::memory_order_release);
    }
    gemm_kernel<GATHER, SWIGLU, FLAT, TWO><<<grid, kThreads, smem_bytes<TWO ? 2 : 1>(), s>>>(a);
    ck(cudaGetLastError(), "gemm");
}
template <bool GATHER, bool SWIGLU> void launch256(const GemmArgs& a, dim3 grid, cudaStream_t s) {
    static std::atomic<bool> configured[64];
    int dev = 0;
    ck(cudaGetDevice(&dev), "device");
    if (dev < 0 || dev >= 64) ck(cudaErrorInvalidDevice, "device range");
    if (!configured[dev].load(std::memory_order_acquire)) {
        ck(cudaFuncSetAttribute(gemm_kernel_256<GATHER, SWIGLU>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem256), "smem");
        configured[dev].store(true, std::memory_order_release);
    }
    gemm_kernel_256<GATHER, SWIGLU><<<grid, kThreads, kSmem256, s>>>(a);
    ck(cudaGetLastError(), "gemm256");
}

void run(const Product& p, void* stream) {
    if (p.base || p.blobs) {   // a layer's experts in one launch
        if (p.n <= 0 || p.blocks <= 0) return;
        if ((p.ids != nullptr) != p.swiglu || !p.blk_off) {
            basalt::log::error("basalt", "prefill fp4: a layer's experts need blk_off and token rows with the gate");
            std::exit(1);
        }
        const bool wide = p.two && p.w_rows % kRows256 == 0;
        if (p.w_cols % 64 || (!wide && (p.w_cols / 64 * 36) % 8)) {
            basalt::log::error("basalt", "prefill fp4: %lld weight columns (whole 64-value blocks; the 128-row tiles take 8-byte rows)",
                               (long long) p.w_cols);
            std::exit(1);
        }
        GemmArgs a{};
        a.w_off = p.w_off; a.w_cols = (int) p.w_cols; a.half = (int) (p.w_rows / 2); a.a4 = (const uint8_t*) p.a4;
        a.gs = p.gs; a.bounds = p.bounds; a.tile_off = p.tile_off; a.ids = p.ids; a.dst = p.dst; a.dst_f16 = p.dst_f16; a.ld_dst = p.ld_dst;
        a.tail_off = p.tail_off;
        a.a4s = p.ids ? (const uint8_t*) p.a4 + (size_t) p.tokens * (size_t) ((p.two ? 2 : 1) * p.w_cols / 2) : nullptr;
        a.base = p.base; a.stride = p.stride; a.blobs = p.blobs; a.blk_off = p.blk_off; a.n = p.n; a.dst_ids = p.dst_ids;
        a.cg = p.gate_softcap; a.cu = p.up_softcap; a.gelu = p.gelu;
        const dim3 grid((unsigned) (p.w_rows / kRowsPerBlock), (unsigned) p.blocks, 1);
        if (p.two) {
            if (wide) {
                const dim3 grid256((unsigned) (p.w_rows / kRows256), (unsigned) p.blocks, 1);
                if (p.swiglu) launch256<true, true>(a, grid256, (cudaStream_t) stream);
                else launch256<false, false>(a, grid256, (cudaStream_t) stream);
            } else {
                if (p.swiglu) launch<true, true, true, true>(a, grid, (cudaStream_t) stream);
                else launch<false, false, true, true>(a, grid, (cudaStream_t) stream);
            }
        } else if (p.swiglu) launch<true, true, true>(a, grid, (cudaStream_t) stream);
        else launch<false, false, true>(a, grid, (cudaStream_t) stream);
        return;
    }
    if (p.n <= 0 || p.max_rows <= 0) return;
    if ((p.ids != nullptr) != p.swiglu || p.gelu || p.w_cols % kKTile) {
        basalt::log::error("basalt", "prefill fp4: token rows go with the SwiGLU output only (whole k-tiles, no GELU)");
        std::exit(1);
    }
    GemmArgs a{};
    for (int j = 0; j < p.n; ++j) a.blob[j] = p.blob[j];
    a.w_off = p.w_off; a.w_cols = (int) p.w_cols; a.half = (int) (p.w_rows / 2); a.a4 = (const uint8_t*) p.a4;
    a.gs = p.gs; a.bounds = p.bounds; a.tile_off = p.tile_off; a.ids = p.ids; a.dst = p.dst; a.ld_dst = p.ld_dst;
    a.tail_off = p.tail_off;
    a.a4s = p.ids ? (const uint8_t*) p.a4 + (size_t) p.tokens * (size_t) (p.w_cols / 2) : nullptr;
    const int block_rows = kBlockTiles * kTileTokens;
    const dim3 grid((unsigned) (p.w_rows / kRowsPerBlock), (unsigned) ((p.max_rows + block_rows - 1) / block_rows),
                    (unsigned) p.n);
    if (p.swiglu) launch<true, true>(a, grid, (cudaStream_t) stream);
    else launch<false, false>(a, grid, (cudaStream_t) stream);
}

int run_length(const Group& g, int i0, int64_t row_cap) {
    int n = 1;
    int64_t rows = g.bounds_host[i0 + 1] - g.bounds_host[i0];
    while (i0 + n < g.n && rows + (g.bounds_host[i0 + n + 1] - g.bounds_host[i0 + n]) <= row_cap) {
        rows += g.bounds_host[i0 + n + 1] - g.bounds_host[i0 + n];
        ++n;
    }
    return n;
}

void run_experts(const Group& g, int i0, int n, void* stream) {
    constexpr int kN = 2560, kFF = 640;
    const int32_t* bh = g.bounds_host + i0;
    const int64_t r0 = bh[0], nr = bh[n] - r0;
    int64_t maxr = 0;
    for (int i = 0; i < n; ++i) maxr = std::max<int64_t>(maxr, bh[i + 1] - bh[i]);
    Product gu;   // gate/up from the token planes, straight to this run's H rows
    for (int i = 0; i < n; ++i) gu.blob[i] = g.blob[i0 + i];
    gu.w_off = 0; gu.w_rows = 2 * kFF; gu.w_cols = kN; gu.n = n;
    gu.a4 = g.a4x; gu.gs = g.gsx; gu.ids = g.ids; gu.tokens = g.tokens; gu.bounds = g.bounds_dev + i0;
    gu.swiglu = true; gu.tail_off = g.tail_off;
    gu.dst = g.h - r0 * kFF; gu.ld_dst = kFF; gu.max_rows = maxr;
    run(gu, stream);
    rebase(g.bounds_dev + i0, n, g.rel, stream);   // this run's own row bounds
    plan(g.rel, n, g.toh, stream);
    quantize(g.h, nullptr, g.rel, g.toh, n, g.a4h, g.gsh, kFF, kFF, nr, stream);
    Product dn;
    for (int i = 0; i < n; ++i) dn.blob[i] = gu.blob[i];
    dn.w_off = g.down_off; dn.w_rows = kN; dn.w_cols = kFF; dn.n = n;
    dn.a4 = g.a4h; dn.gs = g.gsh; dn.bounds = g.rel; dn.tile_off = g.toh;
    dn.dst = g.dm + r0 * kN; dn.ld_dst = kN; dn.max_rows = maxr;
    run(dn, stream);
}

int run_group(const Group& g, int64_t row_cap, void* stream) {
    int runs = 0;
    for (int i0 = 0; i0 < g.n;) {
        const int n = run_length(g, i0, row_cap);
        run_experts(g, i0, n, stream);
        i0 += n;
        ++runs;
    }
    return runs;
}

}  // namespace basalt::prefill::fp4
