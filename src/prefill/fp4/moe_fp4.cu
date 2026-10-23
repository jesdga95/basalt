// src/prefill/fp4/moe_fp4.cu - see include/basalt/prefill/moe_fp4.hpp.
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
constexpr int kBStage = kRowsPerBlock * kBRowBytes;     // 9216
constexpr int kBlockTiles = 2;                          // 64-token tiles per GEMM block
constexpr int kSmem = kStages * (kBlockTiles * kAStage + kBStage);
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

__global__ void rebase_kernel(const int32_t* __restrict__ bounds, int n, int32_t* __restrict__ out) {
    const int i = threadIdx.x;
    if (i <= n) out[i] = bounds[i] - bounds[0];
}

// ---------------------------------------------------------------------------------------- quantizer

// One warp per row: token scale g = amax / (6 * 448); per 16 values a UE4M3 scale of block amax / (6 g); codes
// round to nearest even on x / (scale * g).  A sub-block's 8 bytes hold value j in the low nibble of byte j and value
// j + 8 in the high one (the layout of an NVFP4 weight sub-block).  TOKENS: row r goes to the token planes (codes
// [rows][cols / 2], then scales [rows][cols / 16], gs[r]), else to its expert's tile.  The arithmetic is the same.
template <int NBL, bool TOKENS>
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
    const int nkt = cols / kKTile, nb = cols / 16;
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
    const float inv_g = amax > 0.0f ? 1.0f / g : 0.0f;
    uint8_t* tbase = a4 + tile * ((int64_t) nkt * kAStage);
#pragma unroll
    for (int i = 0; i < NBL; ++i) {
        const int b = lane + 32 * i;
        if (b >= nb) continue;
        const __nv_fp8_storage_t sb = __nv_cvt_float_to_fp8(bm[i] * inv_g * (1.0f / 6.0f), __NV_SATFINITE, __NV_E4M3);
        const float sd = __half2float(__half(__nv_cvt_fp8_to_halfraw(sb, __NV_E4M3)));
        const float inv = sd > 0.0f ? 1.0f / (sd * g) : 0.0f;
        const float4* p = (const float4*) (xr + 16 * b);
        float v[16];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float4 t = p[q];
            v[4 * q] = t.x * inv; v[4 * q + 1] = t.y * inv; v[4 * q + 2] = t.z * inv; v[4 * q + 3] = t.w * inv;
        }
        uint32_t w[2] = {0, 0};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            unsigned short c;
            asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %2, %1; cvt.u16.u8 %0, t; }"
                : "=h"(c) : "f"(v[j]), "f"(v[j + 8]));
            w[j >> 2] |= (uint32_t) (c & 0xff) << (8 * (j & 3));
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
    int64_t ld_dst;
    size_t tail_off;          // SWIGLU: the blob's {s_gate, s_up, s_down, -}
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
template <bool GATHER, bool SWIGLU>
__device__ __forceinline__ void load_stage(int stage, int kt, uint8_t* smem, const uint8_t* a_src, size_t tile_stride,
                                           int ntiles, const uint8_t* const* a_code, const int* a_dst,
                                           const uint8_t* a_scale, int s_dst, const uint8_t* w_src, size_t row_bytes,
                                           int rb, int half) {
    uint8_t* as = smem + stage * kBlockTiles * kAStage;
    uint8_t* bs = smem + kStages * kBlockTiles * kAStage + stage * kBStage;
    if (GATHER) {   // chunk i of a thread lies in tile i, its scale row in tile threadIdx.x / 64
#pragma unroll
        for (int i = 0; i < kBlockTiles * kTileTokens * 4 / kThreads; ++i)
            if (i < ntiles) cp_async16(as + a_dst[i], a_code[i] + (size_t) kt * 64);
        if ((int) threadIdx.x < ntiles * kTileTokens) cp_async8(as + s_dst, a_scale + (size_t) kt * 8);
    } else {
        for (int i = threadIdx.x; i < ntiles * (kAStage / 16); i += kThreads) {
            const int t = i / (kAStage / 16), c = i % (kAStage / 16);
            cp_async16(as + t * kAStage + 16 * c, a_src + t * tile_stride + (size_t) kt * kAStage + 16 * c);
        }
    }
    for (int i = threadIdx.x; i < kRowsPerBlock * (kBRowBytes / 8); i += kThreads) {
        const int row = i / (kBRowBytes / 8), c = i % (kBRowBytes / 8);
        cp_async8(bs + row * kBRowBytes + 8 * c,
                  w_src + (size_t) b_row<SWIGLU>(rb, row, half) * row_bytes + (size_t) kt * kBRowBytes + 8 * c);
    }
}

// One block: up to two consecutive 64-token tiles of an expert x 128 weight rows; each warp owns one tile's 64 tokens
// and 32 rows, so a weight fragment feeds four token groups.  GATHER: a row's activations are its token's (token planes,
// row -> token from ids); SWIGLU: the gate/up product's outputs leave as the tail-scaled SwiGLU of each gate/up pair.
// Every output keeps the same MMA chain over K as the plain product of one tile, so the products are the same bits.
template <bool GATHER, bool SWIGLU>
__global__ __launch_bounds__(kThreads, 1) void gemm_kernel(const GemmArgs a) {
    extern __shared__ __align__(1024) uint8_t smem[];
    __shared__ int32_t tok_s[kBlockTiles * kTileTokens];
    const int j = blockIdx.z, tp = blockIdx.y, rb = blockIdx.x;
    const int cnt = a.bounds[j + 1] - a.bounds[j];
    const int first = tp * kBlockTiles * kTileTokens;   // the block's first row inside the expert
    if (first >= cnt) return;
    const int ntiles = first + kTileTokens < cnt ? 2 : 1;
    const int nkt = a.w_cols / kKTile;
    const size_t row_bytes = (size_t) (a.w_cols / 64) * 36;
    const size_t tile_stride = (size_t) nkt * kAStage;
    const uint8_t* w_src = a.blob[j] + a.w_off;
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
            a_code[i] = a.a4 + (size_t) tok_s[r] * (size_t) (a.w_cols / 2) + 16 * q;
            a_dst[i] = (r / kTileTokens) * kAStage + rr * 64 + ((q ^ ((rr >> 1) & 3)) << 4);
        }
        if (threadIdx.x < kBlockTiles * kTileTokens) {
            const int r = threadIdx.x;
            a_scale = a.a4s + (size_t) tok_s[r] * (size_t) (a.w_cols / 16);
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
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < nkt)
            load_stage<GATHER, SWIGLU>(s, s, smem, a_src, tile_stride, ntiles, a_code, a_dst, a_scale, s_dst, w_src,
                                       row_bytes, rb, a.half);
        cp_commit();
    }
    for (int kt = 0; kt < nkt; ++kt) {
        cp_wait<kStages - 2>();
        __syncthreads();
        if (kt + kStages - 1 < nkt)
            load_stage<GATHER, SWIGLU>((kt + kStages - 1) % kStages, kt + kStages - 1, smem, a_src, tile_stride, ntiles,
                                       a_code, a_dst, a_scale, s_dst, w_src, row_bytes, rb, a.half);
        cp_commit();
        if (!live) continue;
        const int stage = kt % kStages;
        const uint8_t* as = smem + stage * kBlockTiles * kAStage + warp_m * kAStage;
        const uint8_t* bs = smem + kStages * kBlockTiles * kAStage + stage * kBStage;
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            unsigned af[4][4], asc[4];
#pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                const int row = mt * 16 + a_row;
                ldmatrix_x4(af[mt][0], af[mt][1], af[mt][2], af[mt][3],
                            smem_addr(as + row * 64 + swz(row, s * 32 + a_col)));
                asc[mt] = *(const uint32_t*) (as + kACodeBytes + (mt * 16 + sfa_row) * 8 + 4 * s);
            }
            unsigned b0[4], b1[4], sb[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const uint8_t* p = bs + (b_tile_row<SWIGLU>(warp_n, r) + group) * kBRowBytes + s * 36;
                b0[r] = *(const uint32_t*) (p + 4 + 4 * tig);
                b1[r] = *(const uint32_t*) (p + 20 + 4 * tig);
                // scale bytes: bit 7 ignored, 0x7F (NaN in hardware) as 0
                unsigned v = *(const uint32_t*) p & 0x7f7f7f7fu;
                sb[r] = v & ~__vcmpeq4(v, 0x7f7f7f7fu);
            }
#pragma unroll
            for (int mt = 0; mt < 4; ++mt)
#pragma unroll
                for (int r = 0; r < 4; ++r) mma_fp4(acc[mt][r], af[mt], b0[r], b1[r], asc[mt], sb[r]);
        }
    }
    cp_wait<0>();
    if (!live) return;

    const int tl = kBlockTiles * tp + warp_m;   // this warp's tile inside the expert
    const int64_t tile = tile0 + warp_m;
    const int64_t row0 = a.bounds[j] + (int64_t) tl * kTileTokens;
    float4 tail = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (SWIGLU) tail = *(const float4*) (a.blob[j] + a.tail_off);
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int t = mt * 16 + group + 8 * h;
            if (tl * kTileTokens + t >= cnt) continue;
            const float g = GATHER ? a.gs[tok_s[warp_m * kTileTokens + t]] : a.gs[tile * kTileTokens + t];
            if (SWIGLU) {
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

__global__ void tails_kernel(GemmArgs a, size_t off, int n, float* dst) {
    const int j = threadIdx.x;
    if (j < n) ((float4*) dst)[j] = *(const float4*) (a.blob[j] + off);
}

}  // namespace

size_t a4_bytes(int64_t rows, int64_t cols, int experts) {
    return (size_t) ((rows + kTileTokens - 1) / kTileTokens + experts) * (size_t) (cols / kKTile) * kAStage;
}
size_t scale_count(int64_t rows, int experts) {
    return (size_t) ((rows + kTileTokens - 1) / kTileTokens + experts) * kTileTokens;
}

void plan(const int32_t* bounds, int n, int32_t* tile_off, void* stream) {
    plan_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(bounds, n, tile_off);
    ck(cudaGetLastError(), "plan");
}

void rebase(const int32_t* bounds, int n, int32_t* out, void* stream) {
    rebase_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(bounds, n, out);
    ck(cudaGetLastError(), "rebase");
}

void quantize(const float* x, const int32_t* ids, const int32_t* bounds, const int32_t* tile_off, int n, void* a4,
              float* gs, int64_t cols, int64_t ld, int64_t rows, void* stream) {
    if (rows <= 0) return;
    const unsigned grid = (unsigned) ((rows + 7) / 8);
    const cudaStream_t s = (cudaStream_t) stream;
    if (cols == 2560) quantize_kernel<5, false><<<grid, 256, 0, s>>>(x, ids, bounds, tile_off, n, (uint8_t*) a4, gs, (int) cols, ld, rows);
    else if (cols == 640) quantize_kernel<2, false><<<grid, 256, 0, s>>>(x, ids, bounds, tile_off, n, (uint8_t*) a4, gs, (int) cols, ld, rows);
    else { basalt::log::error("basalt", "prefill fp4: %lld columns", (long long) cols); std::exit(1); }
    ck(cudaGetLastError(), "quantize");
}

size_t token_bytes(int64_t tokens, int64_t cols) { return (size_t) tokens * (size_t) (cols / 2 + cols / 16); }

void quantize_tokens(const float* x, int64_t tokens, int64_t cols, int64_t ld, void* a4, float* gs, void* stream) {
    if (tokens <= 0) return;
    const unsigned grid = (unsigned) ((tokens + 7) / 8);
    const cudaStream_t s = (cudaStream_t) stream;
    if (cols == 2560) quantize_kernel<5, true><<<grid, 256, 0, s>>>(x, nullptr, nullptr, nullptr, 0, (uint8_t*) a4, gs, (int) cols, ld, tokens);
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
template <bool GATHER, bool SWIGLU> void launch(const GemmArgs& a, dim3 grid, cudaStream_t s) {
    static std::atomic<bool> configured[64];
    int dev = 0;
    ck(cudaGetDevice(&dev), "device");
    if (dev < 0 || dev >= 64) ck(cudaErrorInvalidDevice, "device range");
    if (!configured[dev].load(std::memory_order_acquire)) {
        ck(cudaFuncSetAttribute(gemm_kernel<GATHER, SWIGLU>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem), "smem");
        configured[dev].store(true, std::memory_order_release);
    }
    gemm_kernel<GATHER, SWIGLU><<<grid, kThreads, kSmem, s>>>(a);
    ck(cudaGetLastError(), "gemm");
}

void run(const Product& p, void* stream) {
    if (p.n <= 0 || p.max_rows <= 0) return;
    if ((p.ids != nullptr) != p.swiglu) {
        basalt::log::error("basalt", "prefill fp4: token rows go with the SwiGLU output only");
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

void cut_pieces(const int32_t* bounds, int n, int64_t cap, Pieces& out) {
    out.expert.clear();
    out.bounds.clear();
    out.group.clear();
    const int64_t step = std::max<int64_t>(kTileTokens, cap / kTileTokens * kTileTokens);
    for (int j = 0; j < n; ++j)
        for (int64_t r = bounds[j]; r < bounds[j + 1]; r += step) {   // an expert without rows has no piece
            out.expert.push_back(j);
            out.bounds.push_back((int32_t) r);
        }
    out.bounds.push_back(n > 0 ? bounds[n] : 0);
    const int np = (int) out.expert.size();
    int64_t rows = 0;
    for (int i = 0; i < np; ++i) {
        const int64_t c = out.bounds[(size_t) i + 1] - out.bounds[(size_t) i];
        if (out.group.empty() || i - out.group.back() == kMaxExperts || rows + c > cap) {
            out.group.push_back(i);
            rows = 0;
        }
        rows += c;
    }
    out.group.push_back(np);
}

}  // namespace basalt::prefill::fp4
