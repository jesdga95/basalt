// src/prefill/moe_fused_iq.cu - see include/basalt/prefill/moe_fused_iq.hpp (the native prompt experts on
// the fused int8 kernels of moe_fused.cu).
//
// The arithmetic.  An activation block of 32 values is x = d_x * a, a = round(x / d_x) in -127..127, stored in natural
// order (per 64 values: 64 codes, then {d_0, d_1} and 8 unused bytes - 80 bytes, the Q2_0 path's size).  A weight
// block of 32 (or 16) values is w = d_w * q with q an int8: the i-quants' codebook entries with their signs applied
// (IQ2_XXS / IQ2_XS / IQ2_S: 0..43, IQ3_XXS: 0..62, IQ3_S: 0..15, IQ4_XS / IQ4_NL: -127..113), Q2_0's code - 1.  So
//     sum w x = d_w * d_x * sum q a,     |sum q a| <= 32 * 127 * 127 < 2^22,
// and the int32 dot becomes a float exactly with one float add: the mma starts from C = 0x4B400000 (the bits of
// 1.5 * 2^23), so as_float(D) - 1.5 * 2^23 is the dot.  Per output and 32 values: that add, d_w * d_x, one fma (the
// formats with a scale per 16 values: two dots, each times its d_w, then d_x).  d_w is llama.cpp's own: IQ2_XXS
// d (2s + 1) / 8, IQ2_XS / IQ2_S d (2s + 1) / 8 per 16, IQ3_XXS d (2s + 1) / 4, IQ3_S d (2s + 1), IQ4_XS d (s - 32),
// IQ4_NL and Q2_0 d (dequantize_row_* in ggml-quants.c, mmq-load-tiles.cuh).
//
// The load stage.  Per 64 values of K (a stage), a thread decodes one 32-value sub-block of one of the work item's
// weight rows: its block bytes are read from the blob with 16-bit loads (the GGUF blocks are 2-byte aligned: 66 / 74 /
// 82 / 98 / 110 / 136 / 18 bytes) one stage ahead into registers (and two 256-value blocks ahead into L2), then turned
// into 32 int8 and the sub-block's scales in shared memory, double-buffered.  The codebooks live in shared memory (up
// to 8 KB, IQ2_S).  The activations arrive by cp.async, four stages deep, as in moe_fused.cu; the fragments come by
// ldmatrix.  The products are mma.sync m16n8k32 (m16n8k16 for the formats with a scale per 16 values); the epilogues
// (SwiGLU, H to int8 per 32 features; down into the per-slot rows) are moe_fused.cu's.
//
// v2 (IQ3_XXS gate/up, IQ4_NL down).  The kernels above decode a stage's weights into shared
// memory for 64 routed rows, read each weight from global memory with 16-bit loads that touch one line per row, and
// pay three float operations per output and 32 values.  v2 instead:
//   - copies each weight row's raw GGUF bytes with 16-byte cp.async (from the 16-byte aligned chunks around them;
//     the bytes keep their offset within 16) - coalesced, no decode stage in between;
//   - decodes the weights straight into the mma A fragments (lane (g, t) needs entries t and 4 + t of rows g and
//     g + 8 of a 32-value sub-block: two codebook words with their signs, or one IQ4_NL word through the table);
//   - gives every warp its own 32 weight rows against all the item's routed rows (up to 128, 16 n8 tiles), so each
//     weight is decoded once per item and no warp waits for another's decode;
//   - takes the dot's scale in one fma: d_w is a half times a small integer, so 1.5 * 2^23 * d_w is exact and
//     fma(as_float(D), d_w, -1.5 * 2^23 * d_w) is round(dot * d_w); then acc = fma(that, d_x, acc);
//   - hands out items through an atomic counter (items differ in size: their routed rows), and prefetches the next
//     item's first weights into L2 while the current one runs.
// Same math as above, rounded in another order: the products agree to the last bits, H's int8 codes almost always.
#include "basalt/prefill/moe_fused_iq.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "basalt/ggml/ggml.h"

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "basalt/ggml/ggml_common.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include "basalt/log.hpp"

namespace basalt::prefill::fused {
namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "prefill fused experts (native): %s: %s", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

constexpr int AB = 80;                  // activation bytes per 64 values: 64 codes, {d0, d1}, 8 unused
constexpr int MAGIC = 0x4B400000;       // the bits of 1.5 * 2^23
constexpr float MAGICF = 12582912.0f;
// 16 warps, each 64 weight rows x 16 routed rows; WW of them along the weight rows and 16 / WW along the routed rows: a
// work item is (WR = 64 WW weight rows) x (TR = 256 / WW routed rows of one expert) - 256 x 64 or 128 x 128, chosen
// per launch by the rows per expert (pick_ww)
constexpr int THREADS = 512;
constexpr int WLD = 80;                 // bytes per decoded weight row of a stage: 64 int8 + 16 (no bank conflicts)
__host__ __device__ constexpr int tile_rows(int ww) { return 256 / ww; }
__host__ __device__ constexpr int weight_rows(int ww) { return 64 * ww; }
constexpr int ASTAGES = 4;              // cp.async depth of the activations (64 values of K a stage)
constexpr int GU_ROWS_K = 2560, D_ROWS_K = 640;   // K of gate/up (n_embd) and of down (n_ff)

// the formats (ggml type ids)
constexpr int T_IQ2_XXS = GGML_TYPE_IQ2_XXS, T_IQ2_XS = GGML_TYPE_IQ2_XS, T_IQ2_S = GGML_TYPE_IQ2_S,
              T_IQ3_XXS = GGML_TYPE_IQ3_XXS, T_IQ3_S = GGML_TYPE_IQ3_S, T_IQ4_XS = GGML_TYPE_IQ4_XS,
              T_IQ4_NL = GGML_TYPE_IQ4_NL, T_Q2_0 = GGML_TYPE_Q2_0;
static_assert(sizeof(block_iq2_xxs) == 66 && sizeof(block_iq2_xs) == 74 && sizeof(block_iq2_s) == 82 &&
              sizeof(block_iq3_xxs) == 98 && sizeof(block_iq3_s) == 110 && sizeof(block_iq4_xs) == 136 &&
              sizeof(block_iq4_nl) == 18 && sizeof(block_q2_0) == 18, "the block layouts this file decodes");

// block bytes, scale per 16 values, codebook bytes in shared memory
__host__ __device__ constexpr int block_bytes(int t) {
    return t == T_IQ2_XXS ? 66 : t == T_IQ2_XS ? 74 : t == T_IQ2_S ? 82 : t == T_IQ3_XXS ? 98 : t == T_IQ3_S ? 110
         : t == T_IQ4_XS ? 136 : 18;
}
__host__ __device__ constexpr bool per16(int t) { return t == T_IQ2_XS || t == T_IQ2_S; }
__host__ __device__ constexpr int grid_bytes(int t) {
    return t == T_IQ2_XXS ? 256 * 8 : t == T_IQ2_XS ? 512 * 8 : t == T_IQ2_S ? 1024 * 8 : t == T_IQ3_XXS ? 256 * 4
         : t == T_IQ3_S ? 512 * 4 : 0;
}
__host__ __device__ constexpr size_t smem_bytes(int t, int ww) {
    return (size_t) 2 * weight_rows(ww) * (WLD + 16) + (size_t) ASTAGES * tile_rows(ww) * AB +
           (size_t) tile_rows(ww) * 4 + grid_bytes(t);
}

// moe_fused.cu's grouping tables (the same layout: fused::group writes them)
struct Tables {
    int32_t *cnt, *off, *fill, *ts;
    int2* tiles;
};
size_t tiles_at(int n_expert) { return ((size_t) (4 * n_expert + 2) * 4 + 15) / 16 * 16; }
Tables tables(void* scratch, int n_expert) {
    int32_t* p = (int32_t*) scratch;
    return {p, p + n_expert, p + 2 * n_expert + 1, p + 3 * n_expert + 1,
            (int2*) ((uint8_t*) scratch + tiles_at(n_expert))};
}

// ---- activations in natural order: one warp per 64 values
__global__ void quant_act_nat_kernel(const float* __restrict__ x, int64_t nblk, uint8_t* __restrict__ xa) {
    const int64_t w = (int64_t) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (w >= nblk) return;
    const float v0 = x[w * 64 + lane], v1 = x[w * 64 + 32 + lane];
    float a0 = fabsf(v0), a1 = fabsf(v1);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        a0 = fmaxf(a0, __shfl_xor_sync(0xffffffffu, a0, o));
        a1 = fmaxf(a1, __shfl_xor_sync(0xffffffffu, a1, o));
    }
    uint8_t* out = xa + w * AB;
    out[lane] = (uint8_t) (int8_t) (a0 > 0.0f ? __float2int_rn(v0 * (127.0f / a0)) : 0);
    out[32 + lane] = (uint8_t) (int8_t) (a1 > 0.0f ? __float2int_rn(v1 * (127.0f / a1)) : 0);
    if (lane == 0) *(float4*) (out + 64) = make_float4(a0 / 127.0f, a1 / 127.0f, 0.0f, 0.0f);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(dst)),
                 "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void pf_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];\n" ::"l"(p)); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
// d = MAGIC + A (16 x 32 s8, row) * B (32 x 8 s8, col): the int32 dot with the magic bias already added (the C
// operand), so as_float(d) - 1.5 * 2^23 is the dot as a float
__device__ __forceinline__ void mma32(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(MAGIC));
}
// the same at m16n8k16 (A 16 x 16, B 16 x 8)
__device__ __forceinline__ void mma16(int (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(b0), "r"(MAGIC));
}
__device__ __forceinline__ float dotf(int d) { return __int_as_float(d) - MAGICF; }
// four 8 x 16-byte matrices, row addresses from lanes 0-7, 8-15, 16-23, 24-31: lane (g, t) gets bytes 4t..4t+3 of row g
// of each - an mma fragment
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((unsigned) __cvta_generic_to_shared(p))
                 : "memory");
}

// ---- the load stage: a 32-value sub-block's bytes into registers, then int8 and scales
__device__ __forceinline__ uint32_t ld16(const uint8_t* p) { return *(const uint16_t*) p; }
__device__ __forceinline__ uint32_t ld32(const uint8_t* p) { return ld16(p) | (ld16(p + 2) << 16); }
__device__ __forceinline__ float half_at(uint32_t w) { return __half2float(__ushort_as_half((unsigned short) w)); }
// llama.cpp's sign unpacking: 7 bits of signs, the 8th their parity (bit 7 of v may be anything)
__device__ __forceinline__ uint32_t unpack_ksigns(uint32_t v) {
    v &= 0xFF;
    const uint32_t p = __popc(v) & 1;
    return (v ^ p << 7) * 0x01010101u;
}
// 8 bytes of a codebook entry with the sign byte `s` (broadcast) applied: bits 0-3 to .x, 4-7 to .y
__device__ __forceinline__ void signed8(uint32_t gx, uint32_t gy, uint32_t s, uint32_t& qx, uint32_t& qy) {
    const uint32_t m0 = __vcmpne4(s & 0x08040201u, 0), m1 = __vcmpne4(s & 0x80402010u, 0);
    qx = __vsub4(gx ^ m0, m0);
    qy = __vsub4(gy ^ m1, m1);
}
// 8 nibbles of q4 through a 16-entry int8 table (4 words): the low nibbles' values in .x, the high ones' in .y
__device__ __forceinline__ void table16(uint32_t q4, const uint32_t (&t)[4], uint32_t& lo, uint32_t& hi) {
    uint32_t tmp[2];
    const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t sh = 16 * i;
        const uint32_t l = __byte_perm(t[0], t[1], q4 >> sh), h = __byte_perm(t[2], t[3], q4 >> sh);
        tmp[i] = __byte_perm(l, h, sel >> sh);
    }
    lo = __byte_perm(tmp[0], tmp[1], 0x6420);
    hi = __byte_perm(tmp[0], tmp[1], 0x7531);
}

// Raw bytes of sub-block `ib` of the block at `bp` (IQ: the 256-value super-block, ib 0..7; Q2_0: the 64-value block,
// ib 0..1; IQ4_NL: the 32-value block).
template <int T> __device__ __forceinline__ void load_unit(const uint8_t* bp, int ib, uint32_t (&w)[5]) {
    if constexpr (T == T_IQ2_XXS) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld16(bp);
    } else if constexpr (T == T_IQ2_XS) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16);
    } else if constexpr (T == T_IQ2_S) {
        w[0] = ld32(bp + 2 + 4 * ib); w[1] = ld32(bp + 34 + 4 * ib);
        w[2] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16) | ((uint32_t) bp[74 + ib] << 24);
    } else if constexpr (T == T_IQ3_XXS) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld32(bp + 66 + 4 * ib); w[3] = ld16(bp);
    } else if constexpr (T == T_IQ3_S) {
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld32(bp + 74 + 4 * ib);
        w[3] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16) | ((uint32_t) ((bp[106 + ib / 2] >> (4 * (ib & 1))) & 15) << 24);
    } else if constexpr (T == T_IQ4_XS) {
#pragma unroll
        for (int k = 0; k < 4; ++k) w[k] = ld32(bp + 8 + 16 * ib + 4 * k);
        const uint32_t ls = ((bp[4 + ib / 2] >> (4 * (ib & 1))) & 15) | (((ld16(bp + 2) >> (2 * ib)) & 3) << 4);
        w[4] = ld16(bp) | (ls << 16);
    } else if constexpr (T == T_IQ4_NL) {
#pragma unroll
        for (int k = 0; k < 4; ++k) w[k] = ld32(bp + 2 + 4 * k);
        w[4] = ld16(bp);
    } else {   // Q2_0
        w[0] = ld32(bp + 2 + 8 * ib); w[1] = ld32(bp + 6 + 8 * ib); w[2] = ld16(bp);
    }
}

// The sub-block as 32 int8 (q[0..7], natural order) and its scales (s0: values 0-15, s1: 16-31).
template <int T>
__device__ __forceinline__ void convert(const uint32_t (&w)[5], const uint8_t* grid, const uint32_t (&kv)[4],
                                        uint32_t (&q)[8], float& s0, float& s1) {
    if constexpr (T == T_IQ2_XXS) {
        const uint2* g = (const uint2*) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 e = g[(w[0] >> (8 * l)) & 255];
            signed8(e.x, e.y, unpack_ksigns(w[1] >> (7 * l)), q[2 * l], q[2 * l + 1]);
        }
        s0 = s1 = half_at(w[2]) * (float) ((w[1] >> 27) | 1) * 0.125f;
    } else if constexpr (T == T_IQ2_XS) {
        const uint2* g = (const uint2*) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t c = (w[l >> 1] >> (16 * (l & 1))) & 0xFFFF;
            const uint2 e = g[c & 511];
            signed8(e.x, e.y, unpack_ksigns(c >> 9), q[2 * l], q[2 * l + 1]);
        }
        const float d = half_at(w[2]);
        const uint32_t sc = w[2] >> 16;
        s0 = d * (float) (2 * (sc & 15) + 1) * 0.125f;
        s1 = d * (float) (2 * ((sc >> 4) & 15) + 1) * 0.125f;
    } else if constexpr (T == T_IQ2_S) {
        const uint2* g = (const uint2*) grid;
        const uint32_t qh = (w[2] >> 16) & 255;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 e = g[((w[0] >> (8 * l)) & 255) | ((qh << (8 - 2 * l)) & 0x300)];
            signed8(e.x, e.y, ((w[1] >> (8 * l)) & 255) * 0x01010101u, q[2 * l], q[2 * l + 1]);
        }
        const float d = half_at(w[2]);
        const uint32_t sc = w[2] >> 24;
        s0 = d * (float) (2 * (sc & 15) + 1) * 0.125f;
        s1 = d * (float) (2 * (sc >> 4) + 1) * 0.125f;
    } else if constexpr (T == T_IQ3_XXS) {
        const uint32_t* g = (const uint32_t*) grid;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t i0 = (w[l >> 1] >> (16 * (l & 1))) & 255, i1 = (w[l >> 1] >> (16 * (l & 1) + 8)) & 255;
            signed8(g[i0], g[i1], unpack_ksigns(w[2] >> (7 * l)), q[2 * l], q[2 * l + 1]);
        }
        s0 = s1 = half_at(w[3]) * (float) (2 * (w[2] >> 28) + 1) * 0.25f;
    } else if constexpr (T == T_IQ3_S) {
        const uint32_t* g = (const uint32_t*) grid;
        const uint32_t qh = (w[3] >> 16) & 255;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t i0 = (w[l >> 1] >> (16 * (l & 1))) & 255, i1 = (w[l >> 1] >> (16 * (l & 1) + 8)) & 255;
            signed8(g[i0 | ((qh << (8 - 2 * l)) & 256)], g[i1 | ((qh << (7 - 2 * l)) & 256)],
                    ((w[2] >> (8 * l)) & 255) * 0x01010101u, q[2 * l], q[2 * l + 1]);
        }
        s0 = s1 = half_at(w[3]) * (float) (1 + 2 * (w[3] >> 24));
    } else if constexpr (T == T_IQ4_XS || T == T_IQ4_NL) {
#pragma unroll
        for (int k = 0; k < 4; ++k) table16(w[k], kv, q[k], q[4 + k]);
        s0 = s1 = T == T_IQ4_XS ? half_at(w[4]) * (float) ((int) (w[4] >> 16) - 32) : half_at(w[4]);
    } else {   // Q2_0: code - 1 via a byte table {-1, 0, 1, 2}
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            const uint32_t c = (w[h >> 1] >> (16 * (h & 1))) & 0xFFFF;
            const uint32_t qe = __byte_perm(0x020100FFu, 0x020100FFu, c & 0x7777);
            const uint32_t qo = __byte_perm(0x020100FFu, 0x020100FFu, (c >> 2) & 0x7777);
            q[2 * h] = __byte_perm(qe, qo, 0x5140);
            q[2 * h + 1] = __byte_perm(qe, qo, 0x7362);
        }
        s0 = s1 = half_at(w[2]);
    }
}

template <int T> __device__ __forceinline__ const void* grid_src() {
    if constexpr (T == T_IQ2_XXS) return iq2xxs_grid;
    else if constexpr (T == T_IQ2_XS) return iq2xs_grid;
    else if constexpr (T == T_IQ2_S) return iq2s_grid;
    else if constexpr (T == T_IQ3_XXS) return iq3xxs_grid;
    else if constexpr (T == T_IQ3_S) return iq3s_grid;
    else return nullptr;
}

// ---- v2's helpers
__device__ __forceinline__ void cp4(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"((unsigned) __cvta_generic_to_shared(dst)),
                 "l"(src));
}
// 32 bits at an even offset of shared memory
__device__ __forceinline__ uint32_t lds32_2a(const uint8_t* sm, uint32_t off) {
    const uint32_t* q = (const uint32_t*) (sm + (off & ~3u));
    return __byte_perm(q[0], q[1], (off & 2u) ? 0x5432u : 0x3210u);
}
__device__ __forceinline__ uint32_t lds16(const uint8_t* sm, uint32_t off) { return *(const uint16_t*) (sm + off); }
// four codebook bytes with sign bits `nib` (bit k: byte k negative); the IQ3_XXS bytes are 4..62, so the
// per-byte negation (b ^ 0xFF) + 1 never carries into the next byte
__device__ __forceinline__ uint32_t signed4(uint32_t gw, uint32_t nib) {
    const uint32_t s01 = (nib * 0x00204081u) & 0x01010101u;
    return (gw ^ (s01 * 0xFFu)) + s01;
}
// IQ3_XXS sub-block `ib` of the super-block at shared offset `ro`, as lane t's part of an A fragment row: entries t
// (values 4t..4t+3) and 4 + t (16 + 4t..); `d4` is the super-block's d / 4
__device__ __forceinline__ void dec_iq3xxs(const uint8_t* sm, uint32_t ro, int ib, uint32_t t, const uint32_t* grid,
                                           float d4, uint32_t& lo, uint32_t& hi, float& dw) {
    const uint32_t qa = sm[ro + 2 + 8 * ib + t], qb = sm[ro + 6 + 8 * ib + t];
    const uint32_t w2 = lds32_2a(sm, ro + 66 + 4 * ib);
    // entry e's signs: bits 4 (e & 1).. of group e / 2 (7 stored bits, the 8th their parity)
    const uint32_t hn = t & 1, sh = 7 * (t >> 1);
    const uint32_t sa = (w2 >> sh) & 127u, sb = (w2 >> (sh + 14)) & 127u;
    lo = signed4(grid[qa], ((sa >> (4 * hn)) & 15u) | (((uint32_t) __popc(sa) & hn) << 3));
    hi = signed4(grid[qb], ((sb >> (4 * hn)) & 15u) | (((uint32_t) __popc(sb) & hn) << 3));
    dw = d4 * (float) (2 * (w2 >> 28) + 1);
}
// IQ4_NL block at shared offset `bo`: lane t's word of nibbles gives values 4t..4t+3 (low) and 16 + 4t.. (high)
__device__ __forceinline__ void dec_iq4nl(const uint8_t* sm, uint32_t bo, uint32_t t, const uint32_t (&kv)[4],
                                          uint32_t& lo, uint32_t& hi, float& dw) {
    table16(lds32_2a(sm, bo + 2 + 4 * t), kv, lo, hi);
    dw = half_at(lds16(sm, bo));
}
#endif

// One work item = (TR routed rows of expert e - one or two of group()'s 64-row tiles -, a block of WR weight rows);
// a persistent grid walks the batch's items.  GU: gate/up rows of WR / 2 features (format WT) against the rows'
// tokens' activations, SwiGLU, H to int8 per 32 features into `out`.  !GU: down rows (format WT) against H, FP32 into
// `dm`.  Warp (wf, wt): weight rows 64 wf.. as four m16 tiles (GU: a tile = 8 features, gate rows as mma rows 0-7 and
// up rows as 8-15, so a lane holds gate and up of one feature), routed rows 16 wt.. as two n8.
template <int WT, bool GU, int WW>
__global__ void __launch_bounds__(THREADS, 1)
native_kernel(const Batch b, const NativeGeom geo, const Tables tb, const uint8_t* __restrict__ act,
              const int32_t* __restrict__ src, uint8_t* __restrict__ out, float* __restrict__ dm) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int TR = tile_rows(WW), WR = weight_rows(WW);
    constexpr int WT_BYTES = WR * WLD, WS_FLOATS = WR * 4, ACT_STAGE = TR * AB;
    constexpr int NS = (GU ? GU_ROWS_K : D_ROWS_K) / 64;   // 64-value stages along K
    constexpr int NFB = (GU ? 1280 : 2560) / WR;          // weight-row blocks per tile
    constexpr int ACT_LD = NS * AB;                       // bytes per activation row: 3200 (a token), 800 (a row's H)
    constexpr int BS = block_bytes(WT);
    constexpr bool K16 = per16(WT);
    extern __shared__ __align__(16) uint8_t smem[];
    uint8_t* wt = smem;                                               // [2][WR][WLD] decoded int8 weights
    float* ws = (float*) (smem + 2 * WT_BYTES);                       // [2][WR][4] their scales per 16 values
    uint8_t* stages = (uint8_t*) (ws + 2 * WS_FLOATS);                // [ASTAGES][TR][AB] activations
    int* srow = (int*) (stages + ASTAGES * ACT_STAGE);                // [TR] the activation row of each tile row
    uint8_t* sgrid = (uint8_t*) (srow + TR);                          // the codebook

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, tig = lane & 3;
    const int wf = warp % WW, nb0 = 16 * (warp / WW);
    if constexpr (grid_bytes(WT) > 0) {
        const uint32_t* gs = (const uint32_t*) grid_src<WT>();
        for (int i = tid; i < grid_bytes(WT) / 4; i += THREADS) ((uint32_t*) sgrid)[i] = gs[i];
    }
    uint32_t kv[4] = {0, 0, 0, 0};
    if constexpr (WT == T_IQ4_XS || WT == T_IQ4_NL) {
#pragma unroll
        for (int k = 0; k < 16; ++k) kv[k >> 2] |= (uint32_t) (uint8_t) kvalues_iq4nl[k] << (8 * (k & 3));
    }
    // Local weight rows: m tile i of warp wf is rows 64 wf + 16 i.. (mma rows g and g + 8: rows +g and +8+g).  For
    // gate/up a tile is 8 features, its gate rows then its up rows, so a lane holds gate and up of one feature.  The
    // fragments come by ldmatrix: lane l gives the address of row (l & 7) of matrix l >> 3 - A: {rows 0-7, rows 8-15}
    // x {bytes 0-15, 16-31} of a 32-value chunk; B: {tokens nb0 + 0-7, + 8-15} x {bytes 0-15, 16-31}.
    const int lm = lane >> 3, l8 = lane & 7;
    const int a_off = (64 * wf + 8 * (lm & 1) + l8) * WLD + 16 * (lm >> 1);
    const int b_off = (nb0 + 8 * (lm >> 1) + l8) * AB + 16 * (lm & 1);
    // this thread's decode unit (the first 2 WR threads): local weight row ur, sub-block uj of each stage
    const bool dec = tid < 2 * WR;
    const int ur = dec ? tid >> 1 : 0, uj = tid & 1;
    const int t0 = tb.ts[b.e0], nwork = (tb.ts[b.e1] - t0) * NFB;
    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int2 tl = tb.tiles[t0 + w / NFB];
        const int fb = w % NFB, e = tl.x, row0 = tl.y;
        if ((row0 - tb.off[e]) % TR != 0) continue;              // a 64-row tile inside an earlier item's TR rows
        const int nrows = min(TR, tb.off[e + 1] - row0);
        const uint8_t* blob = b.blob[e - b.e0];
        const int rbase = fb * WR;
        // local row ur: gate/up - feature (WR / 2) fb + 8 (ur >> 4) + (ur & 7), its up row when bit 3 is set
        const uint8_t* wrow = GU ? blob + ((ur & 8) ? geo.up_off : 0) +
                                       (size_t) (fb * (WR / 2) + 8 * (ur >> 4) + (ur & 7)) * geo.gu_row
                                 : blob + geo.down_off + (size_t) (rbase + ur) * geo.d_row;
        auto unit = [&](int s) -> const uint8_t* {                  // the block of stage s's sub-block
            if (GU) return wrow + (s >> 2) * BS;
            return WT == T_IQ4_NL ? wrow + (2 * s + uj) * BS : wrow + s * BS;
        };
        auto sub = [&](int s) { return GU ? 2 * (s & 3) + uj : uj; };
        // the weight bytes into L2 ahead of the register loads (one stage ahead only - less than a DRAM round trip):
        // gate/up PF super-blocks ahead (the row's two threads take a block's first and last line), down the whole row
        // slice at once (180 or 360 bytes a row)
        constexpr int PF = 2;
        auto prefetch_sb = [&](int sb) {
            if (dec && sb < GU_ROWS_K / 256) pf_l2(wrow + sb * BS + (uj ? BS - 1 : 0));
        };
        if (GU) {
#pragma unroll
            for (int sb = 0; sb < PF; ++sb) prefetch_sb(sb);
        } else if (dec) {
            for (size_t o = 128 * (size_t) uj; o < geo.d_row + 127; o += 256) pf_l2(wrow + min(o, geo.d_row - 1));
        }
        __syncthreads();                                          // the previous item is done with the buffers
        if (tid < TR) srow[tid] = tid < nrows ? (GU ? src[row0 + tid] : row0 + tid) : -1;
        __syncthreads();                                          // srow
        // the rows past the item's end are not loaded: a warp entirely past it skips the products, and the columns
        // of a partial one are never written (no reduction mixes columns)
        auto load_act = [&](int s) {
            uint8_t* st = stages + (s % ASTAGES) * ACT_STAGE;
            for (int c = tid; c < TR * 5; c += THREADS) {
                const int r = c / 5, q = c % 5;
                if (r < nrows) cp16(st + r * AB + q * 16, act + (size_t) srow[r] * ACT_LD + s * AB + q * 16);
            }
        };
        auto put = [&](const uint32_t (&raw)[5], int buf) {
            if (!dec) return;
            uint32_t q[8];
            float s0, s1;
            convert<WT>(raw, sgrid, kv, q, s0, s1);
            uint4* d = (uint4*) (wt + buf * WT_BYTES + ur * WLD + 32 * uj);
            d[0] = make_uint4(q[0], q[1], q[2], q[3]);
            d[1] = make_uint4(q[4], q[5], q[6], q[7]);
            *(float2*) (ws + buf * WS_FLOATS + ur * 4 + 2 * uj) = make_float2(s0, s1);
        };
#pragma unroll
        for (int s = 0; s < ASTAGES - 1; ++s) {
            if (s < NS) load_act(s);
            cp_commit();
        }
        uint32_t raw[5] = {0, 0, 0, 0, 0};
        if (dec) load_unit<WT>(unit(0), sub(0), raw);
        put(raw, 0);
        if (dec) load_unit<WT>(unit(1), sub(1), raw);
        // a warp whose rows are all past the tile's end only takes part in the loads
        const bool on0 = nb0 < nrows, on1 = nb0 + 8 < nrows;
        float acc[4][2][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int n = 0; n < 2; ++n)
#pragma unroll
                for (int q = 0; q < 4; ++q) acc[i][n][q] = 0.0f;
        for (int s = 0; s < NS; ++s) {
            cp_wait<ASTAGES - 2>();
            __syncthreads();
            if (s + ASTAGES - 1 < NS) load_act(s + ASTAGES - 1);
            cp_commit();
            if (GU && (s & 3) == 0) prefetch_sb((s >> 2) + PF);
            if (on0) {
                const uint8_t* W = wt + (s & 1) * WT_BYTES;
                const float* S = ws + (s & 1) * WS_FLOATS;
                const uint8_t* sa = stages + (s % ASTAGES) * ACT_STAGE;
                // the stage's scales: per weight row 4 (per 16 values), per routed row 2 (per 32)
                float2 dx[2][2];                                  // [n8 tile][column 2 tig + cc]
#pragma unroll
                for (int n = 0; n < 2; ++n)
#pragma unroll
                    for (int cc = 0; cc < 2; ++cc) dx[n][cc] = *(const float2*) (sa + (nb0 + 8 * n + 2 * tig + cc) * AB + 64);
#pragma unroll
                for (int h = 0; h < 2; ++h) {                     // the stage's two 32-value halves
                    uint32_t bq[4];                               // {n0 bytes 0-15, n0 16-31, n1 0-15, n1 16-31}
                    ldsm4(bq, sa + b_off + 32 * h);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        uint32_t a[4];                            // {rows g, g + 8} x {bytes 0-15, 16-31}
                        ldsm4(a, W + a_off + 16 * i * WLD + 32 * h);
                        const float2 swa = *(const float2*) (S + (64 * wf + 16 * i + g) * 4 + 2 * h);
                        const float2 swb = *(const float2*) (S + (64 * wf + 16 * i + 8 + g) * 4 + 2 * h);
                        const float wa0 = swa.x, wa1 = swa.y, wb0 = swb.x, wb1 = swb.y;
#pragma unroll
                        for (int n = 0; n < 2; ++n) {
                            if (n == 1 && !on1) break;
                            if constexpr (K16) {
                                // a scale per 16 weights: the two halves' dots, each times its scale, then d_x
                                int d0[4], d1[4];
                                mma16(d0, a[0], a[1], bq[2 * n]);
                                mma16(d1, a[2], a[3], bq[2 * n + 1]);
#pragma unroll
                                for (int q = 0; q < 4; ++q) {
                                    const float v = fmaf(q < 2 ? wa1 : wb1, dotf(d1[q]), (q < 2 ? wa0 : wb0) * dotf(d0[q]));
                                    acc[i][n][q] = fmaf(h ? dx[n][q & 1].y : dx[n][q & 1].x, v, acc[i][n][q]);
                                }
                            } else {
                                int d[4];
                                mma32(d, a, bq[2 * n], bq[2 * n + 1]);
#pragma unroll
                                for (int q = 0; q < 4; ++q) {
                                    const float p = (q < 2 ? wa0 : wb0) * (h ? dx[n][q & 1].y : dx[n][q & 1].x);
                                    acc[i][n][q] = fmaf(p, dotf(d[q]), acc[i][n][q]);
                                }
                            }
                        }
                    }
                }
            }
            // the next stage's weights (into the other buffer, read after the next barrier).  (Decoding them before
            // this stage's products instead was slower.)
            if (s + 1 < NS) {
                put(raw, (s + 1) & 1);
                if (dec && s + 2 < NS) load_unit<WT>(unit(s + 2), sub(s + 2), raw);
            }
        }
        if (!on0) continue;
        if (GU) {
            // SwiGLU of feature (WR / 2) fb + 32 wf + 8 i + g (lane g of m tile i: rows 64 wf + 16 i + g and + 8), tile
            // rows nb0 + 8 n + 2 tig + c; H to int8 per row over the warp's 32 features (one 32-value half-block of the
            // down product's K)
            const int blk = WW * fb + wf;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    float h[4], am = 0.0f;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float gt = acc[i][n][c], up = acc[i][n][2 + c];
                        h[i] = gt / (1.0f + __expf(-gt)) * up;
                        am = fmaxf(am, fabsf(h[i]));
                    }
#pragma unroll
                    for (int o = 4; o < 32; o <<= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
                    const float inv = am > 0.0f ? 127.0f / am : 0.0f;
                    const int r = nb0 + 8 * n + 2 * tig + c;
                    if (r < nrows) {
                        uint8_t* o = out + (size_t) (row0 + r) * (10 * AB) + (blk >> 1) * AB;
                        const int hh = blk & 1;
#pragma unroll
                        for (int i = 0; i < 4; ++i) o[32 * hh + 8 * i + g] = (uint8_t) (int8_t) __float2int_rn(h[i] * inv);
                        if (g == 0) *(float*) (o + 64 + 4 * hh) = am / 127.0f;
                    }
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (n == 1 && !on1) break;
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int r = nb0 + 8 * n + 2 * tig + (q & 1);
                        if (r < nrows) dm[(size_t) (row0 + r) * 2560 + rbase + 64 * wf + 16 * i + 8 * (q >> 1) + g] = acc[i][n][q];
                    }
                }
        }
    }
#endif
}

// ---- v2 (see the top of the file).  A work item: 256 weight rows (gate/up: 128 features, the gate and up rows of 8
// features per m16 tile as above; down: 256 outputs) x up to 128 routed rows of one expert (group()'s 64-row tiles
// in pairs).  8 warps, warp w: weight rows 32 w.. (two m16 tiles) x all the item's routed rows (16 n8 tiles).  Along
// K: chunks of 128 values (the activations: codes and d_x per 32, two stages) and units of raw weight bytes (an
// IQ3_XXS super-block = two chunks, or four IQ4_NL blocks = one chunk, two buffers), one barrier per chunk.
constexpr int V2_THREADS = 256;
constexpr int V2_WR = 256;                                  // weight rows per item
constexpr int V2_NR = 128;                                  // routed rows per item
constexpr int V2_CSTR = 144;                                // bytes per routed row of a chunk: 128 codes + 16
constexpr int V2_ACT = V2_NR * V2_CSTR + 4 * V2_NR * 4;     // a chunk: codes [NR][CSTR], then d_x [4][NR]
constexpr int V2_PF = 2;                                    // raw units into L2 ahead
static_assert(V2_THREADS == V2_WR, "a thread per weight row for the L2 prefetches");
template <bool GU> struct V2 {
    static constexpr int K = GU ? GU_ROWS_K : D_ROWS_K;
    static constexpr int NC = K / 128;                      // chunks
    static constexpr int RB = GU ? 98 : 72;                 // raw bytes of a row per unit
    static constexpr int CPR = GU ? 2 : 1;                  // chunks per unit
    static constexpr int NU = NC / CPR;
    // bytes per row of a raw buffer: >= 14 + RB (the offset within 16), a multiple of 16, 28 words (rows g = 0..7
    // of a decode load fall in distinct banks)
    static constexpr int SLOT = 112;
    static constexpr int NP = SLOT / 16;
    static constexpr int NFB = (GU ? 1280 : 2560) / V2_WR;  // items per routed-row pair of tiles
    static constexpr int ACT_LD = K / 64 * AB;              // bytes per activation row (a token, or a row of H)
    static constexpr int OFF_RAW = 2 * V2_ACT;
    static constexpr int RAWBUF = V2_WR * SLOT;
    static constexpr int OFF_SROW = OFF_RAW + 2 * RAWBUF;
    static constexpr int OFF_GRID = OFF_SROW + V2_NR * 4;
    static constexpr int OFF_MISC = OFF_GRID + (GU ? 1024 : 0);
    static constexpr int SMEM = OFF_MISC + 16;
    static_assert(SLOT >= 14 + RB && SLOT % 16 == 0 && NC % CPR == 0, "v2 layout");
    static_assert(V2_WR * NP % V2_THREADS == 0 && V2_NR * 12 % V2_THREADS == 0, "v2 copy loops");
    static_assert(8 * V2_NR * 4 <= V2_ACT, "the H scale exchange fits in a chunk stage");
};

// GU: IQ3_XXS gate/up against the tokens' activations, SwiGLU, H to int8 per 32 features into `out`.  !GU: IQ4_NL
// down against H, FP32 into `dm`.  `ctr`: the launch's item counter, zero at the start.
template <bool GU>
__global__ void __launch_bounds__(V2_THREADS, 1)
v2_kernel(const Batch b, const NativeGeom geo, const Tables tb, const uint8_t* __restrict__ act,
          const int32_t* __restrict__ src, uint8_t* __restrict__ out, float* __restrict__ dm, int* __restrict__ ctr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using C = V2<GU>;
    extern __shared__ __align__(16) uint8_t smem[];
    int* srow = (int*) (smem + C::OFF_SROW);                   // the activation row of each routed row
    int* misc = (int*) (smem + C::OFF_MISC);                   // [0] an item, [1] the next item
    [[maybe_unused]] const uint32_t* sgrid = (const uint32_t*) (smem + C::OFF_GRID);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2;
    const uint32_t t = lane & 3;
    if constexpr (GU) {
        const uint32_t* gs = (const uint32_t*) iq3xxs_grid;
        for (int i = tid; i < 256; i += V2_THREADS) ((uint32_t*) (smem + C::OFF_GRID))[i] = gs[i];
    }
    [[maybe_unused]] uint32_t kv[4] = {0, 0, 0, 0};
    if constexpr (!GU) {
#pragma unroll
        for (int k = 0; k < 16; ++k) kv[k >> 2] |= (uint32_t) (uint8_t) kvalues_iq4nl[k] << (8 * (k & 3));
    }
    const int t0 = tb.ts[b.e0], nwork = (tb.ts[b.e1] - t0) * C::NFB;
    // ldsm4 of a pair of n8 tiles: lanes 0-7 tile 2p bytes 0-15, 8-15 tile 2p bytes 16-31, 16-31 the same of 2p + 1
    const int b_lane = ((lane & 7) + 8 * (lane >> 4)) * V2_CSTR + 16 * ((lane >> 3) & 1);
    // local weight row r of item block f: gate/up - feature 128 f + 8 (r >> 4) + (r & 7), its up row when bit 3 is set
    auto row_ptr = [&](const uint8_t* bl, int f, int r) -> const uint8_t* {
        if constexpr (GU)
            return bl + ((r & 8) ? geo.up_off : 0) + (size_t) (128 * f + 8 * (r >> 4) + (r & 7)) * geo.gu_row;
        else
            return bl + geo.down_off + (size_t) (V2_WR * f + r) * geo.d_row;
    };
    auto prefetch_unit = [&](const uint8_t* p, int u) {   // a unit spans at most two lines
        pf_l2(p + u * C::RB);
        pf_l2(p + u * C::RB + C::RB - 1);
    };

    if (tid == 0) misc[0] = atomicAdd(ctr, 1);
    __syncthreads();                                            // misc[0], the codebook
    int w = misc[0];
    while (w < nwork) {
        const int2 tl = tb.tiles[t0 + w / C::NFB];
        const int fb = w % C::NFB, e = tl.x, row0 = tl.y;
        if (((row0 - tb.off[e]) & (V2_NR - 1)) != 0) {         // the second tile of an item: the next one
            __syncthreads();                                    // every thread has read misc[0]
            if (tid == 0) misc[0] = atomicAdd(ctr, 1);
            __syncthreads();
            w = misc[0];
            continue;
        }
        const int nrows = min(V2_NR, tb.off[e + 1] - row0), npairs = (nrows + 15) >> 4;
        const uint8_t* blob = b.blob[e - b.e0];
        int nxt = 0;
        if (tid == 0) nxt = atomicAdd(ctr, 1);                  // the next item, read after the first chunk
        if (tid < V2_NR) srow[tid] = tid < nrows ? (GU ? src[row0 + tid] : row0 + tid) : 0;
        // this lane's weight rows (m tile i, half hh: local row 32 warp + 16 i + 8 hh + g): their offset within 16
        uint32_t lo4[2][2];
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int hh = 0; hh < 2; ++hh)
                lo4[i][hh] = (uint32_t) (uintptr_t) row_ptr(blob, fb, 32 * warp + 16 * i + 8 * hh + g) & 15u;
        const uint8_t* myrow = row_ptr(blob, fb, tid);
        __syncthreads();                                        // srow

        auto load_raw = [&](int u) {                            // unit u of every row into buffer u & 1
            const int dst = C::OFF_RAW + (u & 1) * C::RAWBUF;
#pragma unroll
            for (int k = 0; k < V2_WR * C::NP / V2_THREADS; ++k) {
                const int idx = tid + k * V2_THREADS, r = idx / C::NP, p = idx - r * C::NP;
                const uint8_t* a = row_ptr(blob, fb, r) + u * C::RB;
                const uint32_t mis = (uint32_t) (uintptr_t) a & 15u;
                if (p < (int) ((mis + C::RB + 15) >> 4)) cp16(smem + dst + r * C::SLOT + 16 * p, a - mis + 16 * p);
            }
        };
        auto load_act = [&](int c) {                            // chunk c of every routed row into stage c & 1
            const int so = (c & 1) * V2_ACT;
#pragma unroll
            for (int k = 0; k < V2_NR * 12 / V2_THREADS; ++k) {
                const int idx = tid + k * V2_THREADS, r = idx / 12, p = idx - r * 12;
                if (r >= nrows) break;
                const uint8_t* s = act + (size_t) srow[r] * C::ACT_LD + (size_t) (2 * c) * AB;
                if (p < 8) cp16(smem + so + r * V2_CSTR + 16 * p, s + (p >> 2) * AB + 16 * (p & 3));
                else cp4(smem + so + V2_NR * V2_CSTR + 4 * ((p - 8) * V2_NR + r),
                         s + ((p - 8) >> 1) * AB + 64 + 4 * (p & 1));
            }
        };
        load_raw(0);
        load_act(0);
        cp_commit();
#pragma unroll
        for (int u = 1; u <= V2_PF; ++u)
            if (u < C::NU) prefetch_unit(myrow, u);

        float acc[2][16][4];
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int n = 0; n < 16; ++n)
#pragma unroll
                for (int q = 0; q < 4; ++q) acc[i][n][q] = 0.0f;
        uint32_t pr[2][2];                                      // the unit's rows in shared memory
        [[maybe_unused]] float dd[2][2];                        // IQ3_XXS: the super-block's d / 4
#pragma unroll 1
        for (int c = 0; c < C::NC; ++c) {
            cp_wait<0>();
            __syncthreads();                                    // chunk c landed; chunk c - 1 is done everywhere
            if (c + 1 < C::NC) {
                load_act(c + 1);
                if ((c + 1) % C::CPR == 0) {
                    const int u = (c + 1) / C::CPR;
                    load_raw(u);
                    if (u + V2_PF < C::NU) prefetch_unit(myrow, u + V2_PF);
                }
            }
            cp_commit();
            if (c == 1) {                                       // the next item's first units into L2
                const int wn = misc[1];
                if (wn < nwork) {
                    const int2 tn = tb.tiles[t0 + wn / C::NFB];
                    if (((tn.y - tb.off[tn.x]) & (V2_NR - 1)) == 0) {
                        const uint8_t* nrow = row_ptr(b.blob[tn.x - b.e0], wn % C::NFB, tid);
#pragma unroll
                        for (int u = 0; u < V2_PF; ++u)
                            if (u < C::NU) prefetch_unit(nrow, u);
                    }
                }
            }
            if (c % C::CPR == 0) {
                const int u = c / C::CPR;
#pragma unroll
                for (int i = 0; i < 2; ++i)
#pragma unroll
                    for (int hh = 0; hh < 2; ++hh) {
                        pr[i][hh] = C::OFF_RAW + (u & 1) * C::RAWBUF + (32 * warp + 16 * i + 8 * hh + g) * C::SLOT +
                                    ((lo4[i][hh] + u * C::RB) & 15u);
                        if constexpr (GU) dd[i][hh] = half_at(lds16(smem, pr[i][hh])) * 0.25f;
                    }
            }
            const int so = (c & 1) * V2_ACT;
#pragma unroll
            for (int j = 0; j < 4; ++j) {                       // the chunk's 32-value sub-blocks
                // A fragments: {row g lo, row g + 8 lo, row g hi, row g + 8 hi}; per row d_w and -1.5 * 2^23 * d_w
                uint32_t a[2][4];
                float dw[2][2], nm[2][2];
#pragma unroll
                for (int i = 0; i < 2; ++i)
#pragma unroll
                    for (int hh = 0; hh < 2; ++hh) {
                        uint32_t lo, hi;
                        float d;
                        if constexpr (GU) dec_iq3xxs(smem, pr[i][hh], 4 * (c & 1) + j, t, sgrid, dd[i][hh], lo, hi, d);
                        else dec_iq4nl(smem, pr[i][hh] + 18 * j, t, kv, lo, hi, d);
                        a[i][hh] = lo;
                        a[i][2 + hh] = hi;
                        dw[i][hh] = d;
                        nm[i][hh] = -MAGICF * d;
                    }
                const float* sdx = (const float*) (smem + so + V2_NR * V2_CSTR) + j * V2_NR;
#pragma unroll
                for (int np = 0; np < 8; ++np) {
                    if (np >= npairs) break;
                    uint32_t bq[4];                             // {tile 2np b0, b1, tile 2np + 1 b0, b1}
                    ldsm4(bq, smem + so + b_lane + np * 16 * V2_CSTR + 32 * j);
#pragma unroll
                    for (int nn = 0; nn < 2; ++nn) {
                        const float2 dx = *(const float2*) (sdx + 16 * np + 8 * nn + 2 * t);
#pragma unroll
                        for (int i = 0; i < 2; ++i) {
                            int d[4];
                            mma32(d, a[i], bq[2 * nn], bq[2 * nn + 1]);
                            float* ac = acc[i][2 * np + nn];
                            // D = 1.5 * 2^23 + dot (rows g, g + 8 x columns 2t, 2t + 1)
                            ac[0] = __fmaf_rn(__fmaf_rn(__int_as_float(d[0]), dw[i][0], nm[i][0]), dx.x, ac[0]);
                            ac[1] = __fmaf_rn(__fmaf_rn(__int_as_float(d[1]), dw[i][0], nm[i][0]), dx.y, ac[1]);
                            ac[2] = __fmaf_rn(__fmaf_rn(__int_as_float(d[2]), dw[i][1], nm[i][1]), dx.x, ac[2]);
                            ac[3] = __fmaf_rn(__fmaf_rn(__int_as_float(d[3]), dw[i][1], nm[i][1]), dx.y, ac[3]);
                        }
                    }
                }
            }
            if (c == 0 && tid == 0) misc[1] = nxt;
        }

        // columns past nrows hold garbage: they are never written, and no reduction below mixes columns
        if constexpr (GU) {
            // SwiGLU of feature 128 fb + 16 warp + 8 i + g; H per routed row and 32 features (warps 2k and 2k + 1):
            // each warp's max over its 16, exchanged through the chunk stage the last chunk did not use
            float* sam = (float*) (smem + (C::NC & 1) * V2_ACT);   // [warp][routed row]
#pragma unroll
            for (int n = 0; n < 16; ++n)
#pragma unroll
                for (int cc = 0; cc < 2; ++cc) {
                    float am = 0.0f;
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        const float gt = acc[i][n][cc], up = acc[i][n][2 + cc];
                        const float h = gt / (1.0f + __expf(-gt)) * up;
                        acc[i][n][cc] = h;
                        am = fmaxf(am, fabsf(h));
                    }
#pragma unroll
                    for (int o = 4; o < 32; o <<= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
                    acc[0][n][2 + cc] = am;
                    if (g == 0) sam[warp * V2_NR + 8 * n + 2 * t + cc] = am;
                }
            __syncthreads();
            const int blk = 4 * fb + (warp >> 1), hh = blk & 1;
#pragma unroll
            for (int n = 0; n < 16; ++n)
#pragma unroll
                for (int cc = 0; cc < 2; ++cc) {
                    const int r = 8 * n + 2 * (int) t + cc;
                    const float am = fmaxf(acc[0][n][2 + cc], sam[(warp ^ 1) * V2_NR + r]);
                    if (r < nrows) {
                        const float inv = am > 0.0f ? 127.0f / am : 0.0f;
                        uint8_t* o = out + (size_t) (row0 + r) * (10 * AB) + (blk >> 1) * AB;
#pragma unroll
                        for (int i = 0; i < 2; ++i)
                            o[32 * hh + 16 * (warp & 1) + 8 * i + g] =
                                (uint8_t) (int8_t) __float2int_rn(acc[i][n][cc] * inv);
                        if ((warp & 1) == 0 && g == 0) *(float*) (o + 64 + 4 * hh) = am / 127.0f;
                    }
                }
        } else {
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int n = 0; n < 16; ++n)
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int r = 8 * n + 2 * (int) t + (q & 1);
                        if (r < nrows)
                            dm[(size_t) (row0 + r) * 2560 + V2_WR * fb + 32 * warp + 16 * i + 8 * (q >> 1) + g] =
                                acc[i][n][q];
                    }
        }
        __syncthreads();                                        // the item is done with every buffer; misc[1]
        w = misc[1];
    }
#endif
}

unsigned blocks(int64_t n, int per) { return (unsigned) ((n + per - 1) / per); }

// per device: whether every kernel here runs (int8 tensor cores, device code in this build, fits), and their occupancy
struct DevInfo {
    bool done = false, ok = false;
    int sms = 0, occ = 1;
    int v2 = -1, v2_occ = 1;   // the v2 kernels: -1 not set up yet, 0 unavailable, 1 ready
};
std::mutex g_mu;
DevInfo g_dev[32];

template <int T, bool GU, int WW> bool setup_ww(int& occ) {
    cudaFuncAttributes fa{};
    if (cudaFuncGetAttributes(&fa, native_kernel<T, GU, WW>) != cudaSuccess || fa.ptxVersion < 80) return false;
    if (cudaFuncSetAttribute(native_kernel<T, GU, WW>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int) smem_bytes(T, WW)) != cudaSuccess)
        return false;
    int o = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o, native_kernel<T, GU, WW>, THREADS, smem_bytes(T, WW)) !=
            cudaSuccess || o < 1)
        return false;
    occ = std::min(occ, o);
    return true;
}
template <int T, bool GU> bool setup_one(int& occ) { return setup_ww<T, GU, 4>(occ) && setup_ww<T, GU, 2>(occ); }
template <bool GU> bool setup_v2(int& occ) {
    cudaFuncAttributes fa{};
    if (cudaFuncGetAttributes(&fa, v2_kernel<GU>) != cudaSuccess || fa.ptxVersion < 80) return false;
    if (cudaFuncSetAttribute(v2_kernel<GU>, cudaFuncAttributeMaxDynamicSharedMemorySize, V2<GU>::SMEM) != cudaSuccess)
        return false;
    int o = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o, v2_kernel<GU>, V2_THREADS, V2<GU>::SMEM) != cudaSuccess ||
        o < 1)
        return false;
    occ = std::min(occ, o);
    basalt::log::info("basalt", "prompt experts, v2 %s kernel: %d registers, %zu bytes local, %d bytes shared",
                 GU ? "gate/up" : "down", fa.numRegs, (size_t) fa.localSizeBytes, V2<GU>::SMEM);
    return true;
}

const DevInfo& dev_info() {
    int dev = 0;
    cudaGetDevice(&dev);
    std::lock_guard<std::mutex> lk(g_mu);
    DevInfo& d = g_dev[dev & 31];
    if (d.done) return d;
    d.done = true;
    int major = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, dev);
    if (major < 8) return d;
    int occ = 1 << 20;
    d.ok = setup_one<T_IQ2_XXS, true>(occ) && setup_one<T_IQ2_XS, true>(occ) && setup_one<T_IQ2_S, true>(occ) &&
           setup_one<T_IQ3_XXS, true>(occ) && setup_one<T_IQ3_S, true>(occ) && setup_one<T_IQ4_XS, true>(occ) &&
           setup_one<T_Q2_0, false>(occ) && setup_one<T_IQ4_NL, false>(occ);
    d.occ = d.ok ? occ : 1;
    cudaGetLastError();
    return d;
}

// the v2 kernels are used unless a test overrides it
std::atomic<int> g_v2_override{-1};
bool v2_wanted() {
    const int o = g_v2_override.load();
    return o >= 0 ? o == 1 : true;
}
// the v2 kernels' blocks per SM when they are wanted and run on the current device, else 0
int v2_occ() {
    if (!v2_wanted() || !dev_info().ok) return 0;
    int dev = 0;
    cudaGetDevice(&dev);
    std::lock_guard<std::mutex> lk(g_mu);
    DevInfo& d = g_dev[dev & 31];
    if (d.v2 < 0) {
        d.v2 = 0;
        int occ = 1 << 20;
        if (setup_v2<true>(occ) && setup_v2<false>(occ)) {
            d.v2 = 1;
            d.v2_occ = occ;
        } else {
            basalt::log::warn("basalt", "the v2 fused prompt kernels do not run on this device; the native kernels stay");
        }
        cudaGetLastError();
    }
    return d.v2 == 1 ? d.v2_occ : 0;
}
// the v2 kernels' item counters, past the grouping tables (group_bytes() leaves 64 bytes for them)
int* v2_counters(const void* scratch, int64_t n, int n_expert) {
    const size_t tiles = (size_t) (n + kTileRows - 1) / kTileRows + (size_t) n_expert;
    const size_t at = (tiles_at(n_expert) + tiles * sizeof(int2) + 15) / 16 * 16;
    return (int*) ((uint8_t*) const_cast<void*>(scratch) + at);
}

bool gu_covered(int t) {
    return t == T_IQ2_XXS || t == T_IQ2_XS || t == T_IQ2_S || t == T_IQ3_XXS || t == T_IQ3_S || t == T_IQ4_XS;
}
bool d_covered(int t) { return t == T_Q2_0 || t == T_IQ4_NL; }

template <int T, bool GU>
void launch(int ww, unsigned grid, const Batch& b, const NativeGeom& g, const Tables& tb, const void* act,
            const int32_t* src, void* out, float* dm, cudaStream_t s) {
    const uint8_t* a = (const uint8_t*) act;
    uint8_t* o = (uint8_t*) out;
    if (ww == 4) native_kernel<T, GU, 4><<<grid, THREADS, smem_bytes(T, 4), s>>>(b, g, tb, a, src, o, dm);
    else native_kernel<T, GU, 2><<<grid, THREADS, smem_bytes(T, 2), s>>>(b, g, tb, a, src, o, dm);
}

template <bool GU>
void launch_v2(unsigned grid, const Batch& b, const NativeGeom& g, const Tables& tb, const void* act,
               const int32_t* src, void* out, float* dm, int* ctr, cudaStream_t s) {
    v2_kernel<GU><<<grid, V2_THREADS, V2<GU>::SMEM, s>>>(b, g, tb, (const uint8_t*) act, src, (uint8_t*) out, dm, ctr);
}

// The work item's shape for a layer of `n` routed rows over `n_expert` experts: 128 routed rows when an expert has
// about one such tile (56 to 112 rows on average), else 64.
int pick_ww(int64_t n, int n_expert) {
    const double avg = (double) n / std::max(n_expert, 1);
    return avg > 56.0 && avg <= 112.0 ? 2 : 4;
}

}  // namespace

bool native_supported(int gu_type, int d_type) {
    return requested() && gu_covered(gu_type) && d_covered(d_type) && dev_info().ok;
}

void native_v2_override(int mode) { g_v2_override.store(mode); }
bool native_v2_ready() { return v2_occ() > 0; }

void quantize_act_native(const float* x, int64_t rows, int64_t cols, void* xa, void* stream) {
    if (rows <= 0) return;
    const int64_t nblk = rows * (cols / 64);
    quant_act_nat_kernel<<<blocks(nblk, 8), 256, 0, (cudaStream_t) stream>>>(x, nblk, (uint8_t*) xa);
    ck(cudaGetLastError(), "quantize_act_native");
}

void experts_native(const Batch& b, const NativeGeom& g, int n_expert, int64_t n, const void* scratch, const void* xa,
                    const int32_t* src, void* ha, float* dm, void* stream) {
    if (b.e1 <= b.e0 || n <= 0) return;
    const DevInfo& d = dev_info();
    if (!d.ok || !gu_covered(g.gu_type) || !d_covered(g.d_type)) {
        basalt::log::error("basalt", "prefill fused experts (native): types %d / %d are not covered here", g.gu_type, g.d_type);
        std::exit(1);
    }
    const cudaStream_t s = (cudaStream_t) stream;
    const Tables tb = tables(const_cast<void*>(scratch), n_expert);
    const int ww = pick_ww(n, n_expert);
    // the most 64-row tiles the batch can have (every row in it, plus a partial tile per expert) - the items of the
    // ones inside a larger item end at once
    const int64_t tiles = (n + kTileRows - 1) / kTileRows + (b.e1 - b.e0);
    const unsigned g_gu = (unsigned) std::min<int64_t>(tiles * (1280 / weight_rows(ww)), (int64_t) d.sms * d.occ);
    const unsigned g_d = (unsigned) std::min<int64_t>(tiles * (2560 / weight_rows(ww)), (int64_t) d.sms * d.occ);
    // v2: IQ3_XXS gate/up and IQ4_NL down, each on its own (H is the same either way)
    const int occ2 = (g.gu_type == T_IQ3_XXS || g.d_type == T_IQ4_NL) ? v2_occ() : 0;
    const bool v2_gu = occ2 > 0 && g.gu_type == T_IQ3_XXS, v2_d = occ2 > 0 && g.d_type == T_IQ4_NL;
    int* ctr = nullptr;
    if (occ2 > 0) {
        ctr = v2_counters(scratch, n, n_expert);
        ck(cudaMemsetAsync(ctr, 0, 2 * sizeof(int), s), "v2 counters");
        static std::atomic<bool> said{false};
        if (!said.exchange(true))
            basalt::log::info("basalt", "prompt experts on the v2 native kernels");
    }
    if (v2_gu) {
        const unsigned gr = (unsigned) std::min<int64_t>(tiles * V2<true>::NFB, (int64_t) d.sms * occ2);
        launch_v2<true>(gr, b, g, tb, xa, src, ha, nullptr, ctr, s);
    } else switch (g.gu_type) {
        case T_IQ2_XXS: launch<T_IQ2_XXS, true>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
        case T_IQ2_XS: launch<T_IQ2_XS, true>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
        case T_IQ2_S: launch<T_IQ2_S, true>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
        case T_IQ3_XXS: launch<T_IQ3_XXS, true>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
        case T_IQ3_S: launch<T_IQ3_S, true>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
        default: launch<T_IQ4_XS, true>(ww, g_gu, b, g, tb, xa, src, ha, nullptr, s); break;
    }
    if (v2_d) {
        const unsigned gr = (unsigned) std::min<int64_t>(tiles * V2<false>::NFB, (int64_t) d.sms * occ2);
        launch_v2<false>(gr, b, g, tb, ha, src, nullptr, dm, ctr + 1, s);
    } else if (g.d_type == T_Q2_0) launch<T_Q2_0, false>(ww, g_d, b, g, tb, ha, src, nullptr, dm, s);
    else launch<T_IQ4_NL, false>(ww, g_d, b, g, tb, ha, src, nullptr, dm, s);
    ck(cudaGetLastError(), "experts_native");
}

}  // namespace basalt::prefill::fused
