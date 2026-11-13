// src/prefill/fp8_gemm.cu - see include/basalt/prefill/fp8_gemm.hpp.
// Adapted from Infernix (Ian Ranson and contributors), Apache-2.0: see THIRD_PARTY_LICENSES.
// The product follows Infernix's fp8_a8_tma_mma_kernel (src/ops/linear/fp8/fp8_a8_tma_mma.cuh, fp8_a8_mma_common.cuh,
// src/ops/common/mbarrier.cuh, mma.cuh): one producer warp streams 128-byte-swizzled TMA tiles (128 tokens x 128 K of
// activations, 256 weight rows x 128 K) through a two-stage mbarrier ring, eight consumer warps as 2 (tokens) x 4
// (rows), each 64 tokens x 64 rows, ldmatrix fragments ping-ponged over the four k32 steps, one CTA per SM, the token
// tiles fastest in the grid.  The activations are the mma's A (m16) and the weight rows its B (n8), so a thread's
// accumulator pair is two adjacent weight rows of one token: the epilogue stores float2 straight to the token-major
// output.  The mma is the block-scaled form with both scales 2^0 (the plain e4m3 form accumulates at half rate on the
// RTX 5090); fp32 accumulation over K in order, the two scales applied with rounded multiplies once at the end, so a
// row's bits do not depend on the chunk length or the tile it falls in.  One product may span up to three weight
// matrices that share the activations (each with its own scale, output and stride): their row tiles run in one grid.
#include "basalt/prefill/fp8_gemm.hpp"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>

#include "basalt/log.hpp"

namespace basalt::prefill::fp8 {
namespace {

constexpr int BT = 128, BR = 256, BK = 128, ST = 2;     // tokens, weight rows, K per stage, stages
constexpr int WT = 2, WR = 4;                           // consumer warps over tokens, rows
constexpr int NCONS = WT * WR * 32, NT = NCONS + 32;    // consumer threads; plus the producer warp
constexpr int WTOK = BT / WT, WROW = BR / WR;           // a warp's 64 tokens x 64 rows
constexpr int MT = WTOK / 16, MR = WROW / 8;            // its mma tiles: 4 x 8
constexpr int ACT_BYTES = ST * BT * BK, W_BYTES = ST * BR * BK;
constexpr int SMEM = ACT_BYTES + W_BYTES + ST * 2 * 8;
constexpr int MAX_SEG = 3;

void ck(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "prefill fp8: %s: %s", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// A tensor map as a kernel argument: the descriptor type is aligned more strictly than the hardware needs on newer
// toolkits, and a by-value argument with that alignment is not representable on every host compiler (MSVC), so the
// argument carries the bytes at the 64-byte boundary the hardware requires and the kernel reads them as a descriptor.
struct alignas(64) TmaDesc {
    unsigned long long w[16];
};
static_assert(sizeof(TmaDesc) == sizeof(CUtensorMap), "a descriptor argument is the descriptor's size");
static_assert(alignof(CUtensorMap) >= 64, "CUtensorMap lost its alignas: the build needs /Zc:__cplusplus");

struct Seg {
    const float* scale;
    void* dst;
    int64_t ld;
    int rows;
    int tile0;   // first row tile of this segment in the grid
};

struct Args {
    TmaDesc tm[1 + MAX_SEG];   // activations, then each segment's weights
    Seg seg[MAX_SEG];
    const float* a_scale;
    int nseg, ntok, tok_tiles, k_tiles, epi;
};

__device__ __forceinline__ unsigned su(const void* p) { return (unsigned) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldsm4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldsm2(unsigned& r0, unsigned& r1, unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(a));
}
// m16n8k32 e4m3 x e4m3 -> f32, block-scaled with both scales 0x7f (2^0): the raw dot product at full accumulate rate
__device__ __forceinline__ void mma_e4m3(float& c0, float& c1, float& c2, float& c3, unsigned a0, unsigned a1, unsigned a2,
                                         unsigned a3, unsigned b0, unsigned b1) {
    constexpr unsigned short z = 0;
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0x7Fu), "h"(z), "h"(z), "r"(0x7Fu), "h"(z), "h"(z));
}
__device__ __forceinline__ void mb_init(unsigned long long* b, unsigned n) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(su(b)), "r"(n) : "memory");
}
__device__ __forceinline__ void mb_wait(unsigned long long* b, unsigned phase) {
    asm volatile("{\n.reg .pred d;\nw_%=:\nmbarrier.try_wait.parity.shared::cta.b64 d, [%0], %1, %2;\n@d bra e_%=;\nbra w_%=;\ne_%=:\n}\n"
                 ::"r"(su(b)), "r"(phase), "r"(0x989680u) : "memory");
}
__device__ __forceinline__ void mb_arrive(unsigned long long* b) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(su(b)) : "memory");
}
__device__ __forceinline__ void mb_expect(unsigned long long* b, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(su(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void tma2d(void* dst, const TmaDesc* map, int k, int row, unsigned long long* bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 ::"r"(su(dst)), "l"(map), "r"(k), "r"(row), "r"(su(bar)) : "memory");
}
// byte offset of (row, logical byte) in a 128-byte-row tile the TMA wrote with the 128-byte swizzle
__device__ __forceinline__ int sw(int row, int byte) { return row * BK + (byte ^ ((row & 7) << 4)); }

// BF16: the outputs are BF16 (Store rounds the product, Add widens the BF16 residual, adds in fp32 and rounds once)
template <bool BF16>
__global__ void __launch_bounds__(NT, 1) gemm_kernel(const __grid_constant__ Args p) {
    extern __shared__ __align__(1024) unsigned char raw[];
    unsigned char* act = raw;
    unsigned char* wt = raw + ACT_BYTES;
    unsigned long long* full = reinterpret_cast<unsigned long long*>(raw + ACT_BYTES + W_BYTES);
    unsigned long long* empty = full + ST;
    // the grid: weight row tiles of every segment, the token tiles fastest (a weight tile serves them back to back)
    const int row_tile = blockIdx.x / p.tok_tiles, tok_tile = blockIdx.x - row_tile * p.tok_tiles;
    int si = 0;
    for (int i = 1; i < p.nseg; ++i)
        if (row_tile >= p.seg[i].tile0) si = i;
    const Seg sg = p.seg[si];
    const int row0 = (row_tile - sg.tile0) * BR, tok0 = tok_tile * BT;
    if (threadIdx.x == 0) {
#pragma unroll
        for (int s = 0; s < ST; ++s) {
            mb_init(full + s, 1);
            mb_init(empty + s, WT * WR);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        if (threadIdx.x == 0) {
            const TmaDesc* tw = &p.tm[1 + si];
            for (int kt = 0; kt < p.k_tiles; ++kt) {
                const int st = kt % ST;
                mb_wait(empty + st, 1u ^ ((kt / ST) & 1u));
                mb_expect(full + st, (BT + BR) * BK);   // a box past the tensor still counts whole (zero filled)
                tma2d(act + st * BT * BK, &p.tm[0], kt * BK, tok0, full + st);
                tma2d(wt + st * BR * BK, tw, kt * BK, row0, full + st);
            }
        }
        return;
    }
    const int tid = threadIdx.x - 32, lane = tid & 31, warp = tid >> 5, wtk = warp / WR, wrw = warp % WR;
    float acc[MT][MR][4] = {};
    const int am = lane >> 3, a_row = (lane & 7) + ((am & 1) << 3), a_col = (am >> 1) * 16;
    const int b_row = lane & 7, b_col = ((lane >> 3) & 1) * 16;
    for (int kt = 0; kt < p.k_tiles; ++kt) {
        const int st = kt % ST;
        mb_wait(full + st, (kt / ST) & 1u);
        const unsigned char* as = act + st * BT * BK;
        const unsigned char* ws = wt + st * BR * BK;
        unsigned af[2][MT][4], bf[2][MR][2];
        auto load = [&](int ks, unsigned (&a)[MT][4], unsigned (&b)[MR][2]) {
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
                ldsm4(a[mt][0], a[mt][1], a[mt][2], a[mt][3], su(as + sw(wtk * WTOK + mt * 16 + a_row, ks * 32 + a_col)));
#pragma unroll
            for (int mr = 0; mr < MR; ++mr)
                ldsm2(b[mr][0], b[mr][1], su(ws + sw(wrw * WROW + mr * 8 + b_row, ks * 32 + b_col)));
        };
        load(0, af[0], bf[0]);
#pragma unroll
        for (int ks = 0; ks < BK / 32; ++ks) {
            const int slot = ks & 1;
            if (ks + 1 < BK / 32) load(ks + 1, af[slot ^ 1], bf[slot ^ 1]);
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                for (int mr = 0; mr < MR; ++mr)
                    mma_e4m3(acc[mt][mr][0], acc[mt][mr][1], acc[mt][mr][2], acc[mt][mr][3], af[slot][mt][0], af[slot][mt][1],
                             af[slot][mt][2], af[slot][mt][3], bf[slot][mr][0], bf[slot][mr][1]);
        }
        // the stage's reads (generic proxy) are ordered before the producer's async-proxy refill
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncwarp();
        if (lane == 0) mb_arrive(empty + st);
    }
    const float ws = sg.scale ? *sg.scale : 1.0f;
    const bool add = p.epi == (int) Epilogue::Add || p.epi == (int) Epilogue::AddBf16;
    // a token row at a time: with Add, its residual pairs are all loaded before any is written back (the loads overlap),
    // and only one row's worth is live beside the accumulators (no spills at 168 registers)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int tk = tok0 + wtk * WTOK + mt * 16 + (lane >> 2) + h * 8;
            if (tk >= p.ntok) continue;
            const float as = __ldg(p.a_scale + tk);
            float* orow = static_cast<float*>(sg.dst) + (int64_t) tk * sg.ld;
            __nv_bfloat16* hrow = static_cast<__nv_bfloat16*>(sg.dst) + (int64_t) tk * sg.ld;
            float2 res[MR];
#pragma unroll
            for (int mr = 0; mr < MR; ++mr) {
                const int r = row0 + wrw * WROW + mr * 8 + 2 * (lane & 3);
                res[mr] = make_float2(0.0f, 0.0f);
                if (add && r < sg.rows) {
                    if constexpr (BF16) {
                        const unsigned u = __ldcg(reinterpret_cast<const unsigned*>(hrow + r));
                        res[mr] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u));
                    } else {
                        res[mr] = __ldcg(reinterpret_cast<const float2*>(orow + r));
                    }
                }
            }
#pragma unroll
            for (int mr = 0; mr < MR; ++mr) {
                const int r = row0 + wrw * WROW + mr * 8 + 2 * (lane & 3);
                if (r >= sg.rows) continue;
                // rounded products, then the add: no contraction into an fma, so the bits do not depend on the instance
                float2 v = make_float2(__fmul_rn(__fmul_rn(acc[mt][mr][2 * h], as), ws),
                                       __fmul_rn(__fmul_rn(acc[mt][mr][2 * h + 1], as), ws));
                if (add) v = make_float2(__fadd_rn(res[mr].x, v.x), __fadd_rn(res[mr].y, v.y));
                if constexpr (BF16) *reinterpret_cast<__nv_bfloat162*>(hrow + r) = __floats2bfloat162_rn(v.x, v.y);
                else *reinterpret_cast<float2*>(orow + r) = v;
            }
        }
}

size_t scale_off(int64_t rows, int64_t cols) { return ((size_t) rows * (size_t) cols + 15) & ~(size_t) 15; }

// One block per row: amax over the row, then E4M3 codes of x * 448 / amax (saturating, round to nearest even).
// NORM: the row is RMSNorm'd first (x * rsqrt(mean(x^2) + eps) * nw), the normed row optionally written to xn.
// In: float or __nv_bfloat16 rows (xn, when written, in the same type).
__device__ __forceinline__ float4 load4(const float* p) { return *reinterpret_cast<const float4*>(p); }
__device__ __forceinline__ float4 load4(const __nv_bfloat16* p) {
    const uint2 u = *reinterpret_cast<const uint2*>(p);
    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
    const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
    return make_float4(a.x, a.y, b.x, b.y);
}
__device__ __forceinline__ void store4(float* p, float4 v) { *reinterpret_cast<float4*>(p) = v; }
__device__ __forceinline__ void store4(__nv_bfloat16* p, float4 v) {
    const __nv_bfloat162 a = __floats2bfloat162_rn(v.x, v.y), b = __floats2bfloat162_rn(v.z, v.w);
    *reinterpret_cast<uint2*>(p) = make_uint2(*reinterpret_cast<const unsigned*>(&a), *reinterpret_cast<const unsigned*>(&b));
}
template <bool NORM, class In = float>
__global__ void __launch_bounds__(256) quantize_kernel(const In* __restrict__ x, int64_t cols, int64_t ld,
                                                      uint8_t* __restrict__ codes, float* __restrict__ scale,
                                                      const float* __restrict__ nw = nullptr, float eps = 0.0f,
                                                      In* __restrict__ xn = nullptr) {
    __shared__ float red[8];
    __shared__ float inv_s, inv_rms_s;
    const In* xr = x + (int64_t) blockIdx.x * ld;
    float inv_rms = 1.0f;
    if (NORM) {
        float ss = 0.0f;
        for (int64_t i = threadIdx.x * 4; i < cols; i += 1024) {
            const float4 v = load4(xr + i);
            ss = fmaf(v.x, v.x, fmaf(v.y, v.y, fmaf(v.z, v.z, fmaf(v.w, v.w, ss))));
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
        __syncthreads();
        if (threadIdx.x == 0) {
            float t = 0.0f;
            for (int i = 0; i < 8; ++i) t += red[i];
            inv_rms_s = rsqrtf(t / (float) cols + eps);
        }
        __syncthreads();
        inv_rms = inv_rms_s;
    }
    auto val4 = [&](int64_t i) -> float4 {
        float4 v = load4(xr + i);
        if (NORM) {
            const float4 w = *reinterpret_cast<const float4*>(nw + i);
            v = make_float4(v.x * inv_rms * w.x, v.y * inv_rms * w.y, v.z * inv_rms * w.z, v.w * inv_rms * w.w);
        }
        return v;
    };
    float m = 0.0f;
    for (int64_t i = threadIdx.x * 4; i < cols; i += 1024) {
        const float4 v = val4(i);
        if (NORM && xn) store4(xn + (int64_t) blockIdx.x * cols + i, v);
        m = fmaxf(m, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int i = 0; i < 8; ++i) t = fmaxf(t, red[i]);
        scale[blockIdx.x] = t / 448.0f;
        inv_s = t > 0.0f ? 448.0f / t : 0.0f;
    }
    __syncthreads();
    const float inv = inv_s;
    uint8_t* cr = codes + (int64_t) blockIdx.x * cols;
    for (int64_t i = threadIdx.x * 4; i < cols; i += 1024) {
        const float4 v = val4(i);
        const __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(make_float2(v.x * inv, v.y * inv), __NV_SATFINITE, __NV_E4M3);
        const __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(make_float2(v.z * inv, v.w * inv), __NV_SATFINITE, __NV_E4M3);
        *reinterpret_cast<uint32_t*>(cr + i) = (uint32_t) lo | ((uint32_t) hi << 16);
    }
}

}  // namespace

size_t act_bytes(int64_t rows, int64_t cols) { return scale_off(rows, cols) + (size_t) rows * sizeof(float) + 16; }
float* act_scales(void* act, int64_t rows, int64_t cols) { return (float*) ((uint8_t*) act + scale_off(rows, cols)); }

void quantize(const float* x, int64_t rows, int64_t cols, int64_t ld, void* act, void* stream) {
    if (rows <= 0) return;
    if (cols % 64 != 0 || ld % 4 != 0) {
        basalt::log::error("basalt", "prefill fp8: rows of %lld values (ld %lld)", (long long) cols, (long long) ld);
        std::exit(1);
    }
    uint8_t* codes = (uint8_t*) act;
    float* scale = (float*) (codes + scale_off(rows, cols));
    quantize_kernel<false><<<(unsigned) rows, 256, 0, (cudaStream_t) stream>>>(x, cols, ld, codes, scale);
    ck("quantize");
}

void rms_norm_quantize(const float* x, const float* w, float eps, int64_t rows, int64_t cols, void* act, float* xn, void* stream) {
    if (rows <= 0) return;
    if (cols % 64 != 0 || cols % 4 != 0) {
        basalt::log::error("basalt", "prefill fp8: rows of %lld values (norm)", (long long) cols);
        std::exit(1);
    }
    uint8_t* codes = (uint8_t*) act;
    float* scale = (float*) (codes + scale_off(rows, cols));
    quantize_kernel<true><<<(unsigned) rows, 256, 0, (cudaStream_t) stream>>>(x, cols, cols, codes, scale, w, eps, xn);
    ck("quantize");
}

void rms_norm_quantize(const uint16_t* x, const float* w, float eps, int64_t rows, int64_t cols, void* act, uint16_t* xn,
                       void* stream) {
    if (rows <= 0) return;
    if (cols % 64 != 0 || cols % 4 != 0) {
        basalt::log::error("basalt", "prefill fp8: rows of %lld values (norm)", (long long) cols);
        std::exit(1);
    }
    uint8_t* codes = (uint8_t*) act;
    float* scale = (float*) (codes + scale_off(rows, cols));
    quantize_kernel<true, __nv_bfloat16><<<(unsigned) rows, 256, 0, (cudaStream_t) stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x), cols, cols, codes, scale, w, eps, reinterpret_cast<__nv_bfloat16*>(xn));
    ck("quantize");
}

namespace {
using EncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                              const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                              CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
EncodeFn encode_fn() {
    static const EncodeFn f = [] {
        void* fp = nullptr;
        cudaDriverEntryPointQueryResult q{};
        if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fp, 12000, cudaEnableDefault, &q) != cudaSuccess ||
            q != cudaDriverEntryPointSuccess) {
            cudaGetLastError();
            return (EncodeFn) nullptr;
        }
        return reinterpret_cast<EncodeFn>(fp);
    }();
    return f;
}
// A row-major [rows][cols] byte matrix as a 128-byte-swizzled map with boxes of 128 bytes x box_rows.
void tmap(TmaDesc* out, const void* base, int64_t rows, int64_t cols, int box_rows) {
    const EncodeFn f = encode_fn();
    CUtensorMap m{};
    const cuuint64_t dim[2] = {(cuuint64_t) cols, (cuuint64_t) rows}, str[1] = {(cuuint64_t) cols};
    const cuuint32_t box[2] = {(cuuint32_t) BK, (cuuint32_t) box_rows}, es[2] = {1, 1};
    if (f == nullptr || f(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base), dim, str, box, es,
                          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) {
        basalt::log::error("basalt", "prefill fp8: tensor map for a [%lld x %lld] matrix", (long long) rows, (long long) cols);
        std::exit(1);
    }
    std::memcpy(out, &m, sizeof m);
}
}  // namespace

void gemm_segmented(const Segment* seg, int nseg, int64_t w_cols, const void* act, int64_t rows, Epilogue epi, void* stream) {
    if (rows <= 0 || nseg < 1) return;
    if (nseg > MAX_SEG || w_cols % 64 != 0) {
        basalt::log::error("basalt", "prefill fp8: %d segments of %lld columns", nseg, (long long) w_cols);
        std::exit(1);
    }
    static bool raised = false;
    if (!raised) {
        if (cudaFuncSetAttribute(gemm_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM) != cudaSuccess ||
            cudaFuncSetAttribute(gemm_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM) != cudaSuccess) {
            basalt::log::error("basalt", "prefill fp8: shared-memory opt-in failed");
            std::exit(1);
        }
        raised = true;
    }
    const bool bf16 = epi == Epilogue::StoreBf16 || epi == Epilogue::AddBf16;
    Args p{};
    tmap(&p.tm[0], act, rows, w_cols, BT);
    int tiles = 0;
    for (int i = 0; i < nseg; ++i) {
        const Segment& s = seg[i];
        if (s.rows <= 0 || s.rows % 128 != 0 || ((uintptr_t) s.w & 15) != 0 || ((uintptr_t) s.dst & (bf16 ? 3 : 7)) != 0 || s.ld_dst % 2 != 0) {
            basalt::log::error("basalt", "prefill fp8: segment %d: [%lld x %lld] matrix, ld %lld", i, (long long) s.rows,
                               (long long) w_cols, (long long) s.ld_dst);
            std::exit(1);
        }
        tmap(&p.tm[1 + i], s.w, s.rows, w_cols, BR);
        p.seg[i] = {s.w_scale, s.dst, s.ld_dst, (int) s.rows, tiles};
        tiles += (int) ((s.rows + BR - 1) / BR);
    }
    p.a_scale = (const float*) ((const uint8_t*) act + scale_off(rows, w_cols));
    p.nseg = nseg;
    p.ntok = (int) rows;
    p.tok_tiles = (int) ((rows + BT - 1) / BT);
    p.k_tiles = (int) ((w_cols + BK - 1) / BK);   // a K tail past the tensor reads as zeros
    p.epi = (int) epi;
    if (bf16)
        gemm_kernel<true><<<(unsigned) (tiles * p.tok_tiles), NT, SMEM, (cudaStream_t) stream>>>(p);
    else
        gemm_kernel<false><<<(unsigned) (tiles * p.tok_tiles), NT, SMEM, (cudaStream_t) stream>>>(p);
    ck("gemm");
}

void gemm(const uint8_t* w, int64_t w_rows, int64_t w_cols, const float* w_scale, const void* act, int64_t rows, void* dst,
          int64_t ld_dst, Epilogue epi, void* stream) {
    const Segment s{w, w_rows, w_scale, dst, ld_dst};
    gemm_segmented(&s, 1, w_cols, act, rows, epi, stream);
}

}  // namespace basalt::prefill::fp8
