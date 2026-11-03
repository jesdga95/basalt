// src/prefill/fp4_dense.cu - see include/basalt/prefill/fp4_dense.hpp.
// The tile structure follows Infernix's nvfp4_a4_tma_kernel (src/ops/linear/nvfp4/nvfp4_a4_tma.cuh, nvfp4_schedule.cuh,
// src/ops/common/mbarrier.cuh, mma.cuh): TMA loads into a mbarrier full/empty pipeline, a producer warpgroup beside
// eight consumer warps (setmaxnreg), the block-scaled m16n8k64 MMA, 128 weight rows x 256 tokens x K 128 per stage,
// token tiles walking fastest so the CTAs that run together share a weight tile.
// Adapted from Infernix (Ian Ranson and contributors), Apache-2.0: see THIRD_PARTY_LICENSES.
//
// BF16 outputs (SwiGLU's H, the down product's residual add) leave through shared memory: once the K loop is over, the
// code stages' memory is the block's output tile, slabs of 64 rows x 256 tokens in the 128-byte swizzle (the eight
// tokens a warp instruction touches fall in different banks).  An Add's residual slabs come in by TMA (their lines
// asked for a few stages before the K loop ends), the consumers write their rounded values into the tile and TMA
// stores write it out, so no thread waits on a load or a store of its own; the tile's token scales are copied into
// shared memory at the block's start for the same reason.  FP32 outputs go straight from the accumulators.
//
// Compute capability 8.9 has no FP4 tensor cores, TMA or mbarrier transaction counts: there kernel_ada runs the product on
// the FP8 tensor cores, one scale group per instruction (see its comment), and the quantizer rounds to E2M1 without the
// FP4 convert.
#include "basalt/prefill/fp4_dense.hpp"

#include "basalt/kernels/silu.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "basalt/log.hpp"

// the device pass is below compute capability 9.0 (8.9): the TMA kernel traps and kernel_ada has its body
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
#define FP4_ADA_PASS 1
#endif

namespace basalt::prefill::fp4_dense {
namespace {

constexpr int kBT = 256;                  // tokens per block
constexpr int kBR = 128;                  // weight rows per block (SwiGLU: 64 gate rows and the same 64 up rows)
constexpr int kStages = 3;
constexpr int kConsumerWarps = 8;         // 4 token quarters x 2 row halves
constexpr int kProducerThreads = 128;     // one warpgroup, so setmaxnreg can move registers to the consumers
constexpr int kThreads = kConsumerWarps * 32 + kProducerThreads;
constexpr int kWarpTokens = kBT / 4, kWarpRows = kBR / 2;
[[maybe_unused]] constexpr int kMmaTokens = kWarpTokens / 16, kMmaRows = kWarpRows / 8;
constexpr int kK64 = 2;                   // 64-value MMA steps per stage (stage K = 128)
constexpr int kScaleSlots = (kStages + 2) / 2;   // an activation scale tile (256 values) serves two stages
constexpr int kScaleTileBytes = kBT * 16;        // 256 tokens x 16 groups

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "fp4 dense: %s: %s", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// ------------------------------------------------------------------------------------------------- device

__device__ __forceinline__ uint32_t smem_addr(const void* p) { return (uint32_t) __cvta_generic_to_shared(p); }

#ifndef FP4_ADA_PASS   // the TMA kernel's helpers (its body is a trap in an 8.9 pass)

__device__ __forceinline__ void mbar_init(uint64_t* b, uint32_t n) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_addr(b)), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t phase) {
    asm volatile("{\n.reg .pred done;\nwait_loop_%=:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1, %2;\n"
                 "@done bra wait_done_%=;\nbra wait_loop_%=;\nwait_done_%=:\n}\n"
                 :
                 : "r"(smem_addr(b)), "r"(phase), "r"(0x989680u)
                 : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_addr(b)) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_addr(b)), "r"(bytes) : "memory");
}
// Each thread that read a stage with generic-proxy loads orders them before the async-proxy refill (the empty arrive
// alone orders them only against generic writes).
__device__ __forceinline__ void fence_reads_before_refill() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }

__device__ __forceinline__ void tma_load_2d(void* dst, const CUtensorMap* map, int c0, int c1, uint64_t* bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 :
                 : "r"(smem_addr(dst)), "l"(map), "r"(c0), "r"(c1), "r"(smem_addr(bar))
                 : "memory");
}

// The tile at `src` out to the map at (c0, c1); stores_read waits until every store issued has read its tile (the block
// may then end).
__device__ __forceinline__ void tma_store_2d(const CUtensorMap* map, int c0, int c1, const void* src) {
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                 :
                 : "l"(map), "r"(c0), "r"(c1), "r"(smem_addr(src))
                 : "memory");
}
__device__ __forceinline__ void stores_read() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}

// 16 bytes to shared memory, the first `n` of them from src and zeros after
__device__ __forceinline__ void cp16z(void* dst, const void* src, int n) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(smem_addr(dst)), "l"(src), "r"(n) : "memory");
}

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x2(unsigned& r0, unsigned& r1, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(addr));
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

#endif

// A tensor map as a kernel argument: the descriptor type is aligned more strictly than the hardware needs on newer
// toolkits and a by-value argument with that alignment is not representable on every host compiler, so the argument
// carries the bytes at the boundary the hardware requires and the kernel reads them as descriptors.
struct alignas(64) TmaDesc {
    unsigned long long w[16];
};
static_assert(sizeof(TmaDesc) == sizeof(CUtensorMap), "a descriptor argument is the descriptor's size");
static_assert(alignof(CUtensorMap) >= 64, "CUtensorMap lost its alignas: the build needs /Zc:__cplusplus");

struct Params {
    TmaDesc a_codes, a_scales, b_codes[2], b_scales[2];
    TmaDesc out;          // the BF16 output rows (SwiGLU's H, a BF16 product's rows or the BF16 add's residual)
    const float* gs;
    const float* ws[2];   // the matrix scales (SwiGLU: gate, up)
    float* dst;           // the plain product's output (FP32, or BF16 under BF)
    __nv_bfloat16* hdst;  // SwiGLU's output (BF16)
    int64_t ld;
    int tokens, K, tok_tiles;
    float* amax;          // SwiGLU (optional): each token's largest |output|, raised with atomicMax (zeroed by the caller)
};
static_assert(offsetof(Params, a_codes) % 64 == 0 && sizeof(TmaDesc) % 64 == 0, "descriptors sit on the map boundary");

struct alignas(128) Shared {
    alignas(128) uint8_t a_codes[kStages][kBT * 64];
    alignas(128) uint8_t b_codes[kStages][kBR * 64];
    alignas(16) uint32_t a_scale4[kScaleSlots][kBT * 4];
    alignas(16) uint8_t b_scales[kStages][2][kBR * kK64 * 4];
    alignas(8) uint64_t full[kStages];
    alignas(8) uint64_t empty[kStages];
    alignas(8) uint64_t tile_in;    // a BF16 output: the tile is the consumers' (an Add's residual is in it)
    alignas(8) uint64_t tile_out;   // a BF16 output: every consumer warp has written its part of the tile
    alignas(16) float gs[kBT];      // a BF16 output: the tile's token scales (zero past the call's tokens)
};
constexpr int kSmem = (int) sizeof(Shared);
static_assert(kSmem <= 99 * 1024, "one block's shared memory");
// A block's BF16 output tile takes the code stages' memory once the K loop is over: slabs of kSlabRows rows (128 bytes
// a token).
[[maybe_unused]] constexpr int kSlabRows = 64, kSlabBytes = kBT * kSlabRows * 2;
[[maybe_unused]] constexpr int kFetchStages = 12;   // stages before the K loop's end an Add asks for its residual lines
static_assert(offsetof(Shared, a_codes) == 0 && offsetof(Shared, b_codes) == sizeof(Shared::a_codes), "the code stages are one run");
static_assert(kBT * kBR * 2 <= (int) (sizeof(Shared::a_codes) + sizeof(Shared::b_codes)), "the output tile fits the code stages");
static_assert(kWarpRows == kSlabRows && kMmaRows == 8, "a warp's rows are one slab, its fragments the slab's 16-byte pieces");

template <int N, class F> __device__ __forceinline__ void with_live(int live, F&& f) {
    if (live >= N) f(std::integral_constant<int, N>{});
    else if constexpr (N > 1) with_live<N - 1>(live, f);
}

// SWIGLU: a block's 128 rows are 64 gate rows and the same 64 up rows (two descriptor sets, each warp holds the gate and
// the up product of the same outputs), and the output is silu(s_gate g) (s_up u).  ADD: dst += the product.
// BF: the plain product's dst is BF16, rounded once (Add widens it, adds in fp32 and rounds once).
template <bool SWIGLU, bool ADD, bool BF = false>
__global__ __launch_bounds__(kThreads, 1) void kernel(const __grid_constant__ Params p) {
#ifdef FP4_ADA_PASS
    __trap();   // never launched below compute capability 9.0 (launch takes kernel_ada there)
#else
    constexpr int branches = SWIGLU ? 2 : 1;
    constexpr int loaded_rows = kBR / branches;
    extern __shared__ __align__(128) unsigned char shared_bytes[];
    Shared& sh = *reinterpret_cast<Shared*>(shared_bytes);
    const CUtensorMap* a_codes_map = reinterpret_cast<const CUtensorMap*>(&p.a_codes);
    const CUtensorMap* a_scales_map = reinterpret_cast<const CUtensorMap*>(&p.a_scales);
    const CUtensorMap* b_codes_map[2] = {reinterpret_cast<const CUtensorMap*>(&p.b_codes[0]),
                                         reinterpret_cast<const CUtensorMap*>(&p.b_codes[1])};
    const CUtensorMap* b_scales_map[2] = {reinterpret_cast<const CUtensorMap*>(&p.b_scales[0]),
                                          reinterpret_cast<const CUtensorMap*>(&p.b_scales[1])};
    // BF16 outputs go through the block's tile: `rows_out` rows of 256 tokens, in slabs
    constexpr bool staged = SWIGLU || BF;
    constexpr int rows_out = SWIGLU ? kBR / 2 : kBR;
    constexpr int slabs = rows_out / kSlabRows;
    const CUtensorMap* out_map = reinterpret_cast<const CUtensorMap*>(&p.out);
    unsigned char* tile = shared_bytes;

    // token tiles fastest: the CTAs that run together share one weight tile, so the matrix is read about once
    const int tok_tile = (int) blockIdx.x % p.tok_tiles;
    const int row_begin = ((int) blockIdx.x / p.tok_tiles) * loaded_rows;
    const int token_begin = tok_tile * kBT;
    const int K = p.K;
    const int ktiles = K / 128;

    if (threadIdx.x == 0) {
#pragma unroll
        for (int s = 0; s < kStages; ++s) {
            mbar_init(&sh.full[s], 1);
            mbar_init(&sh.empty[s], kConsumerWarps);
        }
        mbar_init(&sh.tile_in, 1);
        mbar_init(&sh.tile_out, kConsumerWarps);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if (threadIdx.x < kProducerThreads) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" ::: "memory");
        if (threadIdx.x >= 32) return;
        // the producer warp: every lane keeps the ring's pace (so the lanes can each do a share of the copies and
        // prefetches below), the first lane issues the tensor copies
        const bool lead = threadIdx.x == 0;
        const auto issue = [&](int kt) {
            const int stage = kt % kStages;
            mbar_wait(&sh.empty[stage], 1u ^ ((kt / kStages) & 1u));
            if (lead) {
                constexpr uint32_t kScaleBytes = kScaleTileBytes;
                constexpr uint32_t kTx = kBT * 64 + kBR * 64 + kScaleBytes + branches * kBR * kK64 * 4;
                // a scale tile covers two K stages: fetched on the even one, the odd one expects that many bytes fewer
                const bool load_scales = (kt & 1) == 0;
                mbar_expect_tx(&sh.full[stage], load_scales ? kTx : kTx - kScaleBytes);
                tma_load_2d(sh.a_codes[stage], a_codes_map, kt * 64, token_begin, &sh.full[stage]);
#pragma unroll
                for (int b = 0; b < branches; ++b)
                    tma_load_2d(sh.b_codes[stage] + b * loaded_rows * 64, b_codes_map[b], kt * 64, row_begin,
                                &sh.full[stage]);
                if (load_scales) {
                    const int scale_tile = tok_tile * (K / 256) + kt / 2;
                    tma_load_2d(sh.a_scale4[(kt / 2) % kScaleSlots], a_scales_map, 0, scale_tile * 16, &sh.full[stage]);
                }
#pragma unroll
                for (int b = 0; b < branches; ++b) {
                    const int scale_row = ((row_begin / 128) * (K / 64) + kt * kK64) * 32;
                    tma_load_2d(sh.b_scales[stage][b], b_scales_map[b], 0, scale_row, &sh.full[stage]);
                }
            }
        };
        const int first = min(kStages, ktiles);   // the ring's first fill waits for no stage
#pragma unroll 1
        for (int kt = 0; kt < first; ++kt) issue(kt);
        if constexpr (staged) {   // the tile's token scales, two 16-byte pieces a lane, while the first stages come in
#pragma unroll
            for (int i = 0; i < kBT / 4 / 32; ++i) {
                const int c = (int) threadIdx.x + i * 32;
                const int t = token_begin + c * 4;
                cp16z(sh.gs + c * 4, p.gs + t, min(max(p.tokens - t, 0), 4) * 4);
            }
            asm volatile("cp.async.commit_group;" ::: "memory");
            asm volatile("cp.async.wait_group 0;" ::: "memory");
        }
        const int fetch_at = max(first, ktiles - kFetchStages);
#pragma unroll 1
        for (int kt = first; kt < ktiles; ++kt) {
            issue(kt);
            if constexpr (ADD && BF) {
                if (kt == fetch_at) {   // the residual tile's lines (8 tokens x 2 lines a lane) toward the last level cache
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const int t = token_begin + (int) threadIdx.x * 8 + (i >> 1);
                        if (t < p.tokens)
                            asm volatile("prefetch.global.L2 [%0];" ::"l"(reinterpret_cast<const __nv_bfloat16*>(p.dst) + (int64_t) t * p.ld + row_begin + (i & 1) * 64) : "memory");
                    }
                }
            }
        }
        if constexpr (staged) {
            // the stages are the consumers' until every warp has left the last one; then the code stages' memory is the
            // output tile: an Add's residual comes in first, the consumers write it, and it goes out
            for (int kt = ktiles; kt < ktiles + kStages; ++kt) mbar_wait(&sh.empty[kt % kStages], 1u ^ ((kt / kStages) & 1u));
            if (lead) {
                if constexpr (ADD) {
                    mbar_expect_tx(&sh.tile_in, slabs * kSlabBytes);
#pragma unroll
                    for (int s = 0; s < slabs; ++s)
                        tma_load_2d(tile + s * kSlabBytes, out_map, row_begin + s * kSlabRows, token_begin, &sh.tile_in);
                } else {
                    mbar_arrive(&sh.tile_in);
                }
            }
            mbar_wait(&sh.tile_out, 0);
            if (lead) {
#pragma unroll
                for (int s = 0; s < slabs; ++s) tma_store_2d(out_map, row_begin + s * kSlabRows, token_begin, tile + s * kSlabBytes);
                stores_read();
            }
        }
        return;
    }

    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;" ::: "memory");
    const int ct = (int) threadIdx.x - kProducerThreads;
    const int lane = ct & 31, warp = ct >> 5;
    const int warp_m = warp >> 1, warp_n = warp & 1;   // token quarter, row half

    const int a_matrix = lane >> 3;
    const int a_row_offset = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_column_byte = (a_matrix >> 1) * 16;
    const int b_row_offset = lane & 7;
    const int b_column_byte = ((lane >> 3) & 1) * 16;
    const int sfa_row = ((lane & 1) << 3) | (lane >> 2);
    const int sfb_row = lane >> 2;

    float acc[kMmaTokens][kMmaRows][4] = {};
    // `live` of this warp's 16-token fragments hold tokens of the call: the others run no MMA, their sums are never
    // stored, and every stored output keeps its own K order
    const auto consume = [&](const auto live) {
#pragma unroll 1
        for (int kt = 0; kt < ktiles; ++kt) {
            const int stage = kt % kStages;
            mbar_wait(&sh.full[stage], (kt / kStages) & 1u);
#pragma unroll
            for (int k64 = 0; k64 < kK64; ++k64) {
                unsigned af[kMmaTokens][4], bf[kMmaRows][2], asc[kMmaTokens], bsc[kMmaRows];
#pragma unroll
                for (int mm = 0; mm < kMmaTokens; ++mm) {
                    if (mm >= decltype(live)::value) continue;
                    const int plane_row = warp_m * kWarpTokens + mm * 16;
                    const int row = plane_row + a_row_offset;
                    const int lb = k64 * 32 + a_column_byte;
                    const int pb = ((lb >> 4) ^ ((row >> 1) & 3)) * 16 + (lb & 15);
                    ldmatrix_x4(af[mm][0], af[mm][1], af[mm][2], af[mm][3], smem_addr(sh.a_codes[stage] + row * 64 + pb));
                    asc[mm] = sh.a_scale4[(kt / 2) % kScaleSlots][(plane_row + sfa_row) * 4 + (kt & 1) * kK64 + k64];
                }
#pragma unroll
                for (int mn = 0; mn < kMmaRows; ++mn) {
                    constexpr int per_branch = kMmaRows / branches;
                    const int branch = mn / per_branch, fragment = mn % per_branch;
                    const int pair_row = warp_n * (kWarpRows / branches) + fragment * 8;
                    const int row = branch * loaded_rows + pair_row + b_row_offset;
                    const int lb = k64 * 32 + b_column_byte;
                    const int pb = ((lb >> 4) ^ ((row >> 1) & 3)) * 16 + (lb & 15);
                    ldmatrix_x2(bf[mn][0], bf[mn][1], smem_addr(sh.b_codes[stage] + row * 64 + pb));
                    const int srow = (row_begin & 127) + pair_row + sfb_row;
                    bsc[mn] = *reinterpret_cast<const unsigned*>(sh.b_scales[stage][branch] +
                                                                 (k64 * 32 + (srow & 31)) * 16 + (srow >> 5) * 4);
                }
#pragma unroll
                for (int mm = 0; mm < kMmaTokens; ++mm) {
                    if (mm >= decltype(live)::value) continue;
#pragma unroll
                    for (int mn = 0; mn < kMmaRows; ++mn) mma_fp4(acc[mm][mn], af[mm], bf[mn][0], bf[mn][1], asc[mm], bsc[mn]);
                }
            }
            fence_reads_before_refill();
            __syncwarp();
            if (lane == 0) mbar_arrive(&sh.empty[stage]);
        }
    };
    const int live_fragments = (p.tokens - token_begin - warp_m * kWarpTokens + 15) / 16;
    if (live_fragments > 0) {
        with_live<kMmaTokens>(live_fragments, consume);
    } else {   // no live token: the barrier protocol alone keeps the producer's refills going
#pragma unroll 1
        for (int kt = 0; kt < ktiles; ++kt) {
            const int stage = kt % kStages;
            mbar_wait(&sh.full[stage], (kt / kStages) & 1u);
            fence_reads_before_refill();
            __syncwarp();
            if (lane == 0) mbar_arrive(&sh.empty[stage]);
        }
        if constexpr (staged)
            if (lane == 0) mbar_arrive(&sh.tile_out);   // it writes nothing into the tile
        return;
    }

    // The epilogue straight from the accumulators: thread (lane) holds tokens lane / 4 and lane / 4 + 8 of each 16-token
    // fragment, weight rows 2 * (lane & 3) and +1 of each 8-row fragment.  The scale is a rounded multiply, and the
    // add after it a separate rounded add, so a column's bits do not depend on whether its tile is full.
    const float s0 = *p.ws[0];
    const float s1 = SWIGLU ? *p.ws[1] : 0.0f;
    if constexpr (staged) {
        // Into the block's tile, a token row at a time.  A token past the call has a zero scale: its rows are written
        // too (the store leaves them out), so the loops have no branch and the rows' arithmetic runs side by side.
        mbar_wait(&sh.tile_in, 0);
        [[maybe_unused]] float amax[kMmaTokens][2];   // SwiGLU: each of the thread's tokens' largest |value|
#pragma unroll
        for (int mm = 0; mm < kMmaTokens; ++mm)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int tl = warp_m * kWarpTokens + mm * 16 + (lane >> 2) + 8 * h;
                const float g = sh.gs[tl];
                if constexpr (SWIGLU) {
                    // the thread's pair of output fragment f: token row tl, 16-byte piece 4 warp_n + f (swizzled by the token)
                    unsigned char* pr = tile + tl * 128 + (lane & 3) * 4;
                    const float ag = __fmul_rn(g, s0), au = __fmul_rn(g, s1);
                    float m = 0.0f;
#pragma unroll
                    for (int f = 0; f < 4; ++f) {
                        float v[2];
#pragma unroll
                        for (int e = 0; e < 2; ++e) {
                            const float gg = __fmul_rn(acc[mm][f][2 * h + e], ag);
                            const float uu = __fmul_rn(acc[mm][f + 4][2 * h + e], au);
                            v[e] = __fmul_rn(kernels::silu_quotient(gg), uu);
                        }
                        const __nv_bfloat162 hv = __floats2bfloat162_rn(v[0], v[1]);   // the amax is of the stored values
                        m = fmaxf(m, fmaxf(fabsf(__low2float(hv)), fabsf(__high2float(hv))));
                        *reinterpret_cast<__nv_bfloat162*>(pr + (((warp_n * 4 + f) ^ (tl & 7)) << 4)) = hv;
                    }
                    amax[mm][h] = m;
                } else {
                    // the thread's pair of fragment f: the warp's slab, token row tl, 16-byte piece f (swizzled by the token)
                    unsigned char* pr = tile + warp_n * kSlabBytes + tl * 128 + (lane & 3) * 4;
                    const float alpha = __fmul_rn(g, s0);
                    float2 res[kMmaRows];   // Add: every residual pair read before any is written back
                    if constexpr (ADD) {
#pragma unroll
                        for (int f = 0; f < kMmaRows; ++f)
                            res[f] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(pr + ((f ^ (tl & 7)) << 4)));
                    }
#pragma unroll
                    for (int f = 0; f < kMmaRows; ++f) {
                        float2 v = make_float2(__fmul_rn(acc[mm][f][2 * h], alpha), __fmul_rn(acc[mm][f][2 * h + 1], alpha));
                        if constexpr (ADD) v = make_float2(__fadd_rn(res[f].x, v.x), __fadd_rn(res[f].y, v.y));
                        *reinterpret_cast<__nv_bfloat162*>(pr + ((f ^ (tl & 7)) << 4)) = __floats2bfloat162_rn(v.x, v.y);
                    }
                }
            }
        // the tile's writes (generic proxy) are ordered before the store's async-proxy read
        fence_reads_before_refill();
        __syncwarp();
        if (lane == 0) mbar_arrive(&sh.tile_out);
        if constexpr (SWIGLU) {
            if (p.amax) {   // each token's largest |value|: the four lanes of the token, then one atomic (max is exact)
#pragma unroll
                for (int mm = 0; mm < kMmaTokens; ++mm)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int t = token_begin + warp_m * kWarpTokens + mm * 16 + (lane >> 2) + 8 * h;
                        float m = amax[mm][h];
                        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
                        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
                        if ((lane & 3) == 0 && t < p.tokens) atomicMax(reinterpret_cast<int*>(p.amax + t), __float_as_int(m));
                    }
            }
        }
    } else {
#pragma unroll
        for (int mm = 0; mm < kMmaTokens; ++mm)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int t = token_begin + warp_m * kWarpTokens + mm * 16 + (lane >> 2) + 8 * h;
                if (t >= p.tokens) continue;
                const float g = p.gs[t];
                float* out = p.dst + (int64_t) t * p.ld;
                const float alpha = __fmul_rn(g, s0);
                const int r0 = row_begin + warp_n * kWarpRows + 2 * (lane & 3);
                float2 res[kMmaRows];   // ADD: every residual pair loaded before any is written back
                if constexpr (ADD) {
#pragma unroll
                    for (int f = 0; f < kMmaRows; ++f) res[f] = __ldcg(reinterpret_cast<const float2*>(out + r0 + 8 * f));
                }
#pragma unroll
                for (int f = 0; f < kMmaRows; ++f) {
                    float2 v = make_float2(__fmul_rn(acc[mm][f][2 * h], alpha), __fmul_rn(acc[mm][f][2 * h + 1], alpha));
                    if constexpr (ADD) v = make_float2(__fadd_rn(res[f].x, v.x), __fadd_rn(res[f].y, v.y));
                    *reinterpret_cast<float2*>(out + r0 + 8 * f) = v;
                }
            }
    }
#endif
}

// ---------------------------------------------------------------------------------------- compute capability 8.9
// kernel_ada: kernel's product on the FP8 tensor cores, one 16-value scale group per instruction.  Every E2M1 code is an
// E4M3 value, so each warp turns both operands' codes into E4M3 bytes and runs mma m16n8k16 e4m3 from a zero
// accumulator for each group: a group's code products are multiples of 1/4 of at most 36, so its sum is exact in the
// MMA's accumulator.  That sum times the token's and the row's group scales is added into fp32 with one fma, group after
// group in K order, so a token's row keeps one order at any chunk length.  The block tile is 128 tokens x 256 weight
// rows (SwiGLU: 128 gate rows and the same 128 up rows), eight warps as 2 (tokens) x 4 (rows) of 64 x 64, K in stages of
// 64 values (four groups; four stages, filled by cp.async from every thread).  In step s of a stage thread (g, c) gives
// the MMA values 16s + 4c .. 16s + 4c + 3 of its rows, a 16-bit load of four codes; a row's two 16-byte halves are
// swapped in every other group of four rows, so the eight rows of a load fall in different banks.  At each stage's start
// a warp turns the group scales of its 64 tokens and 64 rows into fp32 in its own shared memory.  The epilogues are
// kernel's arithmetic, from the accumulators to global memory.
constexpr int kAdaBT = 128, kAdaBR = 256, kAdaStages = 4, kAdaThreads = 256;
constexpr int kAdaACodes = kAdaBT * 32, kAdaBCodes = kAdaBR * 32;               // a stage's 64 values: 32 code bytes a row
constexpr int kAdaAScales = kAdaBT * 4, kAdaBScales = kAdaBR * 4;               // and four scale bytes
constexpr int kAdaStage = kAdaACodes + kAdaBCodes + kAdaAScales + kAdaBScales;  // 13,824 bytes
constexpr int kAdaWarpScales = 2 * 4 * 64;   // a warp's group scales of a stage in fp32: its 64 tokens and 64 rows, 4 groups
constexpr int kAdaSmem = kAdaStages * kAdaStage + kAdaThreads / 32 * kAdaWarpScales * 4;

struct AdaParams {
    const uint8_t* a_codes;    // [tokens][K / 2]
    const uint8_t* a_scales;   // the scale tiles (act_bytes)
    const uint8_t* w[2];       // the plane matrices (SwiGLU: gate, up)
    const float* gs;
    const float* ws[2];        // the matrix scales
    float* dst;                // the plain product's output (FP32, or BF16 under BF)
    __nv_bfloat16* hdst;       // SwiGLU's output (BF16)
    float* amax;               // SwiGLU (optional)
    int64_t ld, rows, K;
    int tokens, tok_tiles;
};

#ifdef FP4_ADA_PASS
__device__ __forceinline__ void cp_async16(void* dst, const void* src, int n) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(smem_addr(dst)), "l"(src), "r"(n) : "memory");
}
__device__ __forceinline__ void cp_async4(void* dst, const void* src, int n) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(smem_addr(dst)), "l"(src), "r"(n) : "memory");
}
__device__ __forceinline__ unsigned prmt(unsigned a, unsigned b, unsigned sel) {
    unsigned d;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
    return d;
}
// the four E2M1 codes of the low 16 bits of w (the first value in the lowest nibble) as four E4M3 bytes (exact): the
// magnitude from a table of the eight values' E4M3 codes, the sign moved from bit 3 of the nibble to bit 7 of the byte
__device__ __forceinline__ unsigned e2m1x4_e4m3(unsigned w) {
    const unsigned mag = prmt(0x3c383000u, 0x4c484440u, w & 0x7777u);
    const unsigned sign = prmt(0u, 0x80808080u, (w >> 1) & 0x4444u);
    return mag | sign;
}
// two UE4M3 scale bytes (the low 16 bits of v) as floats (exact), the low byte in x
__device__ __forceinline__ float2 ue4m3x2_float2(unsigned v) {
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) v, __NV_E4M3);
    return __half22float2(*reinterpret_cast<const __half2*>(&h));
}
// d = A (16 tokens x 16 values, E4M3) * B (16 values x 8 weight rows, E4M3), from a zero accumulator
__device__ __forceinline__ void mma_e4m3(float* d, unsigned a0, unsigned a1, unsigned b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a0), "r"(a1), "r"(b), "f"(0.0f));
}
#endif

// the block's row of the warp's 8-row fragment f (SwiGLU: gate fragments 0..3, the same rows' up fragments 4..7)
template <bool SWIGLU> __device__ __forceinline__ int ada_row(int wrw, int f) {
    if constexpr (SWIGLU) return f < 4 ? 32 * wrw + 8 * f : kAdaBR / 2 + 32 * wrw + 8 * (f - 4);
    else return 64 * wrw + 8 * f;
}

template <bool SWIGLU, bool ADD, bool BF = false>
__global__ __launch_bounds__(kAdaThreads, 1) void kernel_ada(const __grid_constant__ AdaParams p) {
#ifndef FP4_ADA_PASS
    __trap();   // 9.0 and up run kernel
#else
    extern __shared__ __align__(128) unsigned char smem[];
    constexpr int loaded = SWIGLU ? kAdaBR / 2 : kAdaBR;   // the rows a block reads of each matrix
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, c = lane & 3;
    const int wtk = warp >> 2, wrw = warp & 3;
    const int tok_tile = (int) blockIdx.x % p.tok_tiles;
    const int64_t row_begin = (int64_t) ((int) blockIdx.x / p.tok_tiles) * loaded;
    const int token_begin = tok_tile * kAdaBT;
    const int64_t K = p.K, half = K / 2;
    const int ktiles = (int) (K / 64);
    const size_t scale_off = (size_t) p.rows * (size_t) half;   // nvfp4p::scale_offset
    // stage st <- K tile kt (values 64 kt ..): codes and scales, zeros past the tokens and the rows; a row's 16-byte
    // halves (groups 0, 1 and 2, 3) trade places when bit 2 of the row is set
    const auto load = [&](int kt, int st) {
        unsigned char* ac = smem + st * kAdaStage;
        unsigned char* bc = ac + kAdaACodes;
        unsigned char* asc = bc + kAdaBCodes;
        unsigned char* bsc = asc + kAdaAScales;
        {
            const int r = tid >> 1, pc = tid & 1, t = token_begin + r;
            const bool ok = t < p.tokens;
            const uint8_t* src = p.a_codes + (int64_t) (ok ? t : 0) * half + kt * 32 + pc * 16;
            cp_async16(ac + r * 32 + (pc ^ ((r >> 2) & 1)) * 16, src, ok ? 16 : 0);
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int idx = tid + i * kAdaThreads, r = idx >> 1, pc = idx & 1;
            const int m = SWIGLU ? r / loaded : 0;
            const int64_t wr = row_begin + (SWIGLU ? r % loaded : r);
            const bool ok = wr < p.rows;
            const uint8_t* src = p.w[m] + (ok ? wr : 0) * half + kt * 32 + pc * 16;
            cp_async16(bc + r * 32 + (pc ^ ((r >> 2) & 1)) * 16, src, ok ? 16 : 0);
        }
        if (tid < kAdaBT) {   // a token's four groups: scale tile (t / 256) * (K / 256) + kt / 4, row t % 256
            const int t = token_begin + tid;
            const bool ok = t < p.tokens;
            const int64_t tt = ok ? t : 0;
            const uint8_t* src = p.a_scales + ((tt / kBT) * (K / 256) + kt / 4) * kScaleTileBytes + (tt % kBT) * 16 + (kt % 4) * 4;
            cp_async4(asc + tid * 4, src, ok ? 4 : 0);
        }
        {   // a row's four groups: nvfp4p::scale_index(row, 4 kt .. 4 kt + 3, K)
            const int r = tid;
            const int m = SWIGLU ? r / loaded : 0;
            const int64_t wr = row_begin + (SWIGLU ? r % loaded : r);
            const bool ok = wr < p.rows;
            const int64_t rr = ok ? wr : 0;
            const uint8_t* src = p.w[m] + scale_off + ((rr / 128) * (K / 64) + kt) * 512 + (rr & 31) * 16 + ((rr & 127) >> 5) * 4;
            cp_async4(bsc + r * 4, src, ok ? 4 : 0);
        }
    };
    float acc[4][8][4] = {};
    // the warp's group scales of the stage in fp32: tokens as [group][g][mt][h] (token 16 mt + 8 h + g of its 64), rows
    // as [group][c][f][e] (row 2c + e of fragment f), so a step's scales of a thread are 8 and 16 consecutive floats
    float* const tsc = reinterpret_cast<float*>(smem + kAdaStages * kAdaStage) + warp * kAdaWarpScales;
    float* const rsc = tsc + kAdaWarpScales / 2;
    const int half_swap = (g >> 2) & 1;   // the thread's rows (g mod 8) have their 16-byte halves swapped
#pragma unroll
    for (int st = 0; st < kAdaStages - 1; ++st) {
        if (st < ktiles) load(st, st);
        asm volatile("cp.async.commit_group;" ::: "memory");
    }
#pragma unroll 1
    for (int kt = 0; kt < ktiles; ++kt) {
        asm volatile("cp.async.wait_group %0;" ::"n"(kAdaStages - 2) : "memory");
        __syncthreads();   // stage kt is in, and every warp has left stage kt - 1, which the next load refills
        if (kt + kAdaStages - 1 < ktiles) load(kt + kAdaStages - 1, (kt + kAdaStages - 1) % kAdaStages);
        asm volatile("cp.async.commit_group;" ::: "memory");
        const unsigned char* ac = smem + (kt % kAdaStages) * kAdaStage;
        const unsigned char* bc = ac + kAdaACodes;
        const unsigned char* asc = bc + kAdaBCodes;
        const unsigned char* bsc = asc + kAdaAScales;
        {   // lane l converts the four group scales of the warp's tokens 2l, 2l + 1 and of its rows 2l, 2l + 1
            const int r = 2 * lane;
            const uint2 tb = *reinterpret_cast<const uint2*>(asc + (wtk * 64 + r) * 4);
            const uint2 rb = *reinterpret_cast<const uint2*>(bsc + (ada_row<SWIGLU>(wrw, r >> 3) + (r & 7)) * 4);
            const int tg = r & 7, tmt = r >> 4, th = (r >> 3) & 1, rf = r >> 3, rc = (r & 7) >> 1;
            __syncwarp();   // every lane is done with the last stage's scales
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const unsigned sel = (unsigned) q | (unsigned) (4 + q) << 4;   // byte q of each of the two rows
                const float2 t = ue4m3x2_float2(prmt(tb.x, tb.y, sel));
                tsc[((q * 8 + tg) * 4 + tmt) * 2 + th] = t.x;
                tsc[((q * 8 + tg + 1) * 4 + tmt) * 2 + th] = t.y;
                *reinterpret_cast<float2*>(rsc + ((q * 4 + rc) * 8 + rf) * 2) = ue4m3x2_float2(prmt(rb.x, rb.y, sel));
            }
            __syncwarp();
        }
#pragma unroll
        for (int s = 0; s < 4; ++s) {   // group s: the thread's codes 4c .. 4c + 3 of it, from each of its rows
            const int off = (((s >> 1) ^ half_swap) << 4) + ((s & 1) << 3) + 2 * c;
            const auto codes = [&](const unsigned char* tile, int row) {
                return e2m1x4_e4m3(*reinterpret_cast<const unsigned short*>(tile + row * 32 + off));
            };
            unsigned a[4][2], b[8];
#pragma unroll
            for (int mt = 0; mt < 4; ++mt)
#pragma unroll
                for (int h = 0; h < 2; ++h) a[mt][h] = codes(ac, wtk * 64 + mt * 16 + g + 8 * h);
#pragma unroll
            for (int f = 0; f < 8; ++f) b[f] = codes(bc, ada_row<SWIGLU>(wrw, f) + g);
            float ts[8], rs[16];   // the step's scales of tokens (mt, h) and of rows (f, e)
#pragma unroll
            for (int i = 0; i < 2; ++i)
                *reinterpret_cast<float4*>(ts + 4 * i) = *reinterpret_cast<const float4*>(tsc + (s * 8 + g) * 8 + 4 * i);
#pragma unroll
            for (int i = 0; i < 4; ++i)
                *reinterpret_cast<float4*>(rs + 4 * i) = *reinterpret_cast<const float4*>(rsc + (s * 4 + c) * 16 + 4 * i);
#pragma unroll
            for (int f = 0; f < 8; ++f)
#pragma unroll
                for (int mt = 0; mt < 4; ++mt) {
                    float d[4];
                    mma_e4m3(d, a[mt][0], a[mt][1], b[f]);
                    acc[mt][f][0] = fmaf(d[0], ts[2 * mt] * rs[2 * f], acc[mt][f][0]);
                    acc[mt][f][1] = fmaf(d[1], ts[2 * mt] * rs[2 * f + 1], acc[mt][f][1]);
                    acc[mt][f][2] = fmaf(d[2], ts[2 * mt + 1] * rs[2 * f], acc[mt][f][2]);
                    acc[mt][f][3] = fmaf(d[3], ts[2 * mt + 1] * rs[2 * f + 1], acc[mt][f][3]);
                }
        }
    }
    // kernel's epilogue arithmetic: thread (g, c) holds tokens g and g + 8 of each 16-token fragment, rows 2c and 2c + 1
    // of each 8-row fragment; the scale a rounded multiply, the add after it a separate rounded add
    const float s0 = *p.ws[0];
    const float s1 = SWIGLU ? *p.ws[1] : 0.0f;
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int t = token_begin + wtk * 64 + mt * 16 + g + 8 * h;
            const bool live = t < p.tokens;
            const float gt = live ? p.gs[t] : 0.0f;
            if constexpr (SWIGLU) {
                const float ag = __fmul_rn(gt, s0), au = __fmul_rn(gt, s1);
                float m = 0.0f;
#pragma unroll
                for (int f = 0; f < 4; ++f) {
                    float v[2];
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const float gg = __fmul_rn(acc[mt][f][2 * h + e], ag);
                        const float uu = __fmul_rn(acc[mt][f + 4][2 * h + e], au);
                        v[e] = __fmul_rn(kernels::silu_quotient(gg), uu);
                    }
                    const __nv_bfloat162 hv = __floats2bfloat162_rn(v[0], v[1]);   // the amax is of the stored values
                    m = fmaxf(m, fmaxf(fabsf(__low2float(hv)), fabsf(__high2float(hv))));
                    if (live) *reinterpret_cast<__nv_bfloat162*>(p.hdst + (int64_t) t * p.ld + row_begin + 32 * wrw + 8 * f + 2 * c) = hv;
                }
                if (p.amax) {   // the token's largest |value|: its four lanes, then one atomic (max is exact)
                    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
                    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
                    if (c == 0 && live) atomicMax(reinterpret_cast<int*>(p.amax + t), __float_as_int(m));
                }
            } else {
                if (!live) continue;
                const float alpha = __fmul_rn(gt, s0);
                float2 res[8];   // ADD: every residual pair loaded before any is written back
#pragma unroll
                for (int f = 0; f < 8; ++f) {
                    const int64_t r = row_begin + 64 * wrw + 8 * f + 2 * c;
                    res[f] = make_float2(0.0f, 0.0f);
                    if (ADD && r < p.rows) {
                        if constexpr (BF)
                            res[f] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(
                                reinterpret_cast<const __nv_bfloat16*>(p.dst) + (int64_t) t * p.ld + r));
                        else
                            res[f] = __ldcg(reinterpret_cast<const float2*>(p.dst + (int64_t) t * p.ld + r));
                    }
                }
#pragma unroll
                for (int f = 0; f < 8; ++f) {
                    const int64_t r = row_begin + 64 * wrw + 8 * f + 2 * c;
                    if (r >= p.rows) continue;
                    float2 v = make_float2(__fmul_rn(acc[mt][f][2 * h], alpha), __fmul_rn(acc[mt][f][2 * h + 1], alpha));
                    if constexpr (ADD) v = make_float2(__fadd_rn(res[f].x, v.x), __fadd_rn(res[f].y, v.y));
                    if constexpr (BF)
                        *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(p.dst) + (int64_t) t * p.ld + r) =
                            __floats2bfloat162_rn(v.x, v.y);
                    else
                        *reinterpret_cast<float2*>(p.dst + (int64_t) t * p.ld + r) = v;
                }
            }
        }
#endif
}

// E2M1 to nearest, ties to the even code, saturating: the rounding of cvt.rn.satfinite.e2m1x2.f32, which 8.9 lacks
// (nvfp4_planes.cu rounds the same way)
__device__ __forceinline__ unsigned e2m1_rn(float x) {
    const float a = fabsf(x);
    const unsigned c = a <= 0.25f ? 0 : a < 0.75f ? 1 : a <= 1.25f ? 2 : a < 1.75f ? 3 : a <= 2.5f ? 4 : a < 3.5f ? 5 : a <= 5.0f ? 6 : 7;
    return c | (x < 0.0f && c ? 8u : 0u);
}

// One warp per token row (and per padding row of the last scale tile).  Token scale g = amax / (6 * 448); per 16 values
// a UE4M3 scale of block amax / (6 g); codes round to nearest even on x / (scale * g), value 2b in byte b's low nibble.
// NORM: the row is RMSNorm'd first (x * rsqrt(mean(x^2) + eps) * nw), the normed row optionally written to xn (Xn: FP32,
// or rounded to BF16).
// AMAX: each row's largest |value| is given (rowmax), so the row is read once.  In: float or __nv_bfloat16 rows.
template <bool NORM, bool AMAX = false, class In = float, class Xn = float>
__global__ void quantize_kernel(const In* __restrict__ x, uint8_t* __restrict__ codes, uint8_t* __restrict__ scales,
                                float* __restrict__ gs, int cols, int64_t ld, int64_t rows, const float* __restrict__ nw = nullptr,
                                float eps = 0.0f, Xn* __restrict__ xn = nullptr, const float* __restrict__ rowmax = nullptr) {
    const int lane = threadIdx.x & 31;
    const int64_t r = (int64_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int nb = cols / 16, ktiles = cols / 256;
    const int64_t tile_base = (r / kBT) * ktiles;
    uint8_t* srow = scales + (r % kBT) * 16;
    if (r >= rows) {   // padding of the last token tile: its scales are defined (zero), no codes are read
        for (int kt = lane; kt < ktiles; kt += 32) *reinterpret_cast<uint4*>(srow + (tile_base + kt) * kScaleTileBytes) = make_uint4(0, 0, 0, 0);
        return;
    }
    const In* xr = x + r * ld;
    float inv_rms = 1.0f;
    if (NORM) {
        float ss = 0.0f;
        for (int b = lane; b < nb; b += 32) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float4 v;
                if constexpr (std::is_same_v<In, float>) {
                    v = *(const float4*) (xr + 16 * b + 4 * i);
                } else {
                    const uint2 u = *(const uint2*) (xr + 16 * b + 4 * i);
                    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
                    const float2 c = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
                    v = make_float4(a.x, a.y, c.x, c.y);
                }
                ss = fmaf(v.x, v.x, fmaf(v.y, v.y, fmaf(v.z, v.z, fmaf(v.w, v.w, ss))));
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        inv_rms = rsqrtf(ss / (float) cols + eps);
    }
    // value c of the row as the quantizer sees it
    auto val4 = [&](int c) -> float4 {
        float4 v;
        if constexpr (std::is_same_v<In, float>) {
            v = *(const float4*) (xr + c);
        } else {
            const uint2 u = *(const uint2*) (xr + c);
            const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
            const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
            v = make_float4(a.x, a.y, b.x, b.y);
        }
        if (NORM) {
            const float4 w = *(const float4*) (nw + c);
            v = make_float4(v.x * inv_rms * w.x, v.y * inv_rms * w.y, v.z * inv_rms * w.z, v.w * inv_rms * w.w);
        }
        return v;
    };
    float amax = 0.0f;
    if (AMAX) {
        amax = rowmax[r];
    } else {
        for (int b = lane; b < nb; b += 32) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 v = val4(16 * b + 4 * i);
                if (NORM && xn) {
                    if constexpr (std::is_same_v<Xn, float>) {
                        *(float4*) (xn + r * cols + 16 * b + 4 * i) = v;
                    } else {
                        const __nv_bfloat162 a = __floats2bfloat162_rn(v.x, v.y);
                        const __nv_bfloat162 c = __floats2bfloat162_rn(v.z, v.w);
                        *reinterpret_cast<uint2*>(xn + r * cols + 16 * b + 4 * i) =
                            make_uint2(*reinterpret_cast<const unsigned*>(&a), *reinterpret_cast<const unsigned*>(&c));
                    }
                }
                amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    }
    const float g = amax * (1.0f / (6.0f * 448.0f));
    if (lane == 0) gs[r] = g;
    const float inv_g = amax > 0.0f ? 1.0f / g : 0.0f;
    for (int b = lane; b < nb; b += 32) {
        float v[16];
        float m = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float4 t = val4(16 * b + 4 * i);
            v[4 * i] = t.x; v[4 * i + 1] = t.y; v[4 * i + 2] = t.z; v[4 * i + 3] = t.w;
            m = fmaxf(m, fmaxf(fmaxf(fabsf(t.x), fabsf(t.y)), fmaxf(fabsf(t.z), fabsf(t.w))));
        }
        const __nv_fp8_storage_t sb = __nv_cvt_float_to_fp8(m * inv_g * (1.0f / 6.0f), __NV_SATFINITE, __NV_E4M3);
        const float sd = __half2float(__half(__nv_cvt_fp8_to_halfraw(sb, __NV_E4M3)));
        const float inv = sd > 0.0f ? 1.0f / (sd * g) : 0.0f;
        uint32_t w[2] = {0, 0};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            unsigned short c;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
            c = (unsigned short) (e2m1_rn(v[2 * j] * inv) | (e2m1_rn(v[2 * j + 1] * inv) << 4));
#else
            asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %2, %1; cvt.u16.u8 %0, t; }"
                : "=h"(c) : "f"(v[2 * j] * inv), "f"(v[2 * j + 1] * inv));   // the first source is the high nibble
#endif
            w[j >> 2] |= (uint32_t) (c & 0xff) << (8 * (j & 3));
        }
        *reinterpret_cast<uint2*>(codes + r * (cols / 2) + 8 * b) = make_uint2(w[0], w[1]);
        srow[(tile_base + b / 16) * kScaleTileBytes + (b & 15)] = (uint8_t) sb;
    }
}

// ----------------------------------------------------------------------------------------------------- host

using EncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                              const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                              CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
EncodeFn encode_fn() {
    static const EncodeFn f = [] {
        void* q = nullptr;
        cudaDriverEntryPointQueryResult r{};
        if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &q, 12000, cudaEnableDefault, &r) != cudaSuccess ||
            r != cudaDriverEntryPointSuccess) {
            cudaGetLastError();
            return (EncodeFn) nullptr;
        }
        return reinterpret_cast<EncodeFn>(q);
    }();
    return f;
}

// A byte matrix [rows][cols] (row stride `stride` bytes), box [box_r][box_c], into a kernel argument.
TmaDesc map_2d(const void* base, uint64_t cols, uint64_t rows, uint64_t stride, uint32_t box_c, uint32_t box_r,
               CUtensorMapSwizzle swizzle, CUtensorMapL2promotion promo, const char* what,
               CUtensorMapDataType type = CU_TENSOR_MAP_DATA_TYPE_UINT8) {
    const EncodeFn f = encode_fn();
    if (f == nullptr) {
        basalt::log::error("basalt", "fp4 dense: cuTensorMapEncodeTiled is unavailable");
        std::exit(1);
    }
    CUtensorMap m;
    const cuuint64_t dim[2] = {cols, rows}, str[1] = {stride};
    const cuuint32_t box[2] = {box_c, box_r}, es[2] = {1, 1};
    if (f(&m, type, 2, const_cast<void*>(base), dim, str, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
          swizzle, promo, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) {
        basalt::log::error("basalt", "fp4 dense: tensor map of %s failed", what);
        std::exit(1);
    }
    TmaDesc d;
    std::memcpy(&d, &m, sizeof d);
    return d;
}

int64_t padded(int64_t tokens) { return (tokens + kBT - 1) / kBT * kBT; }

// the current device has TMA and the FP4 tensor cores (compute capability 9.0 and up: the build's 12.0), per device
bool tma_device() {
    static std::atomic<signed char> known[64];   // per device: 0 unknown, 1 no, 2 yes
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { (void) cudaGetLastError(); return true; }
    signed char k = known[dev].load(std::memory_order_acquire);
    if (k == 0) {
        int major = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) {
            (void) cudaGetLastError();
            return true;
        }
        k = major >= 9 ? 2 : 1;
        known[dev].store(k, std::memory_order_release);
    }
    return k == 2;
}

template <bool SWIGLU, bool ADD, bool BF>
void launch_ada(const Params& q, const void* w0, const void* w1, int64_t rows, int64_t K, const void* act, int64_t tokens,
                cudaStream_t s) {
    constexpr int loaded = SWIGLU ? kAdaBR / 2 : kAdaBR;
    static std::atomic<bool> configured[64];   // the shared-memory opt-in is per device and per kernel
    int dev = 0;
    ck(cudaGetDevice(&dev), "device");
    if (dev < 0 || dev >= 64) ck(cudaErrorInvalidDevice, "device range");
    if (!configured[dev].load(std::memory_order_acquire)) {
        ck(cudaFuncSetAttribute(kernel_ada<SWIGLU, ADD, BF>, cudaFuncAttributeMaxDynamicSharedMemorySize, kAdaSmem), "smem");
        configured[dev].store(true, std::memory_order_release);
    }
    AdaParams p{};
    p.a_codes = (const uint8_t*) act;
    p.a_scales = (const uint8_t*) act + tokens * (K / 2);
    p.w[0] = (const uint8_t*) w0;
    p.w[1] = (const uint8_t*) (SWIGLU ? w1 : w0);
    p.gs = q.gs;
    p.ws[0] = q.ws[0];
    p.ws[1] = q.ws[1];
    p.dst = q.dst;
    p.hdst = q.hdst;
    p.amax = q.amax;
    p.ld = q.ld;
    p.rows = rows;
    p.K = K;
    p.tokens = (int) tokens;
    p.tok_tiles = (int) ((tokens + kAdaBT - 1) / kAdaBT);
    const int64_t row_tiles = (rows + loaded - 1) / loaded;
    kernel_ada<SWIGLU, ADD, BF><<<(unsigned) (row_tiles * p.tok_tiles), kAdaThreads, kAdaSmem, s>>>(p);
    ck(cudaGetLastError(), "product (8.9)");
}

template <bool SWIGLU, bool ADD, bool BF = false> void launch(Params p, const void* w0, const void* w1, int64_t rows, int64_t K,
                                             const void* act, int64_t tokens, cudaStream_t s) {
    if (!tma_device()) {
        launch_ada<SWIGLU, ADD, BF>(p, w0, w1, rows, K, act, tokens, s);
        return;
    }
    constexpr int loaded_rows = kBR / (SWIGLU ? 2 : 1);
    static std::atomic<bool> configured[64];   // the shared-memory opt-in is per device and per kernel
    int dev = 0;
    ck(cudaGetDevice(&dev), "device");
    if (dev < 0 || dev >= 64) ck(cudaErrorInvalidDevice, "device range");
    if (!configured[dev].load(std::memory_order_acquire)) {
        ck(cudaFuncSetAttribute(kernel<SWIGLU, ADD, BF>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem), "smem");
        configured[dev].store(true, std::memory_order_release);
    }
    const uint8_t* a = (const uint8_t*) act;
    const int64_t code_bytes = tokens * (K / 2);
    p.a_codes = map_2d(a, K / 2, tokens, K / 2, 64, kBT, CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_NONE, "activation codes");
    p.a_scales = map_2d(a + code_bytes, 256, padded(tokens) / kBT * (K / 256) * 16, 256, 256, 16,
                        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, "activation scales");
    const void* w[2] = {w0, w1};
    for (int b = 0; b < (SWIGLU ? 2 : 1); ++b) {
        const uint8_t* wb = (const uint8_t*) w[b];
        p.b_codes[b] = map_2d(wb, K / 2, rows, K / 2, 64, loaded_rows, CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, "weight codes");
        p.b_scales[b] = map_2d(wb + rows * K / 2, 16, (uint64_t) rows * K / 256, 16, 16, 64, CU_TENSOR_MAP_SWIZZLE_NONE,
                               CU_TENSOR_MAP_L2_PROMOTION_NONE, "weight scales");
    }
    if constexpr (SWIGLU || BF) {   // the BF16 rows a block's tile is stored to (and, for the add, loaded from)
        const void* out = SWIGLU ? (const void*) p.hdst : (const void*) p.dst;
        p.out = map_2d(out, (uint64_t) rows, (uint64_t) tokens, (uint64_t) p.ld * 2, kSlabRows, kBT, CU_TENSOR_MAP_SWIZZLE_128B,
                       CU_TENSOR_MAP_L2_PROMOTION_NONE, "output rows", CU_TENSOR_MAP_DATA_TYPE_UINT16);
    }
    p.K = (int) K;
    p.tokens = (int) tokens;
    p.tok_tiles = (int) ((tokens + kBT - 1) / kBT);
    const int64_t row_tiles = rows / loaded_rows;
    kernel<SWIGLU, ADD, BF><<<(unsigned) (row_tiles * p.tok_tiles), kThreads, kSmem, s>>>(p);
    ck(cudaGetLastError(), "product");
}

// `bf16`: the output rows are BF16, written by tensor stores (16-byte aligned, a multiple of 16 bytes apart).
void check(const void* a, const void* b, int64_t rows, int64_t K, int64_t tokens, const void* act, const void* dst, int64_t ld,
           const char* what, bool bf16 = false, const float* gs = nullptr) {
    const auto al = [](const void* v, size_t n) { return (reinterpret_cast<uintptr_t>(v) & (n - 1)) == 0; };
    if (rows <= 0 || rows % 128 || K <= 0 || K % 256 || tokens <= 0 || ld < rows || (ld & (bf16 ? 7 : 1)) || !al(a, 16) || !al(b, 16) ||
        !al(act, 16) || !al(dst, bf16 ? 16 : 8) || (bf16 && !al(gs, 16))) {
        basalt::log::error("basalt", "fp4 dense: %s: rows %lld K %lld tokens %lld ld %lld", what, (long long) rows, (long long) K,
                           (long long) tokens, (long long) ld);
        std::exit(1);
    }
}

}  // namespace

size_t act_bytes(int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) (cols / 2) + (size_t) padded(rows) * (size_t) (cols / 16);
}

void quantize(const float* x, int64_t rows, int64_t cols, int64_t ld, void* act, float* gs, void* stream) {
    if (rows <= 0) return;
    if (cols % 256 || (ld & 3) || (reinterpret_cast<uintptr_t>(x) & 15) || (reinterpret_cast<uintptr_t>(act) & 15)) {
        basalt::log::error("basalt", "fp4 dense: quantize %lld columns, ld %lld", (long long) cols, (long long) ld);
        std::exit(1);
    }
    uint8_t* codes = (uint8_t*) act;
    uint8_t* scales = codes + (size_t) rows * (size_t) (cols / 2);
    const unsigned grid = (unsigned) ((padded(rows) + 7) / 8);
    quantize_kernel<false><<<grid, 256, 0, (cudaStream_t) stream>>>(x, codes, scales, gs, (int) cols, ld, rows);
    ck(cudaGetLastError(), "quantize");
}

void quantize_amax(const uint16_t* x, const float* rowmax, int64_t rows, int64_t cols, int64_t ld, void* act, float* gs,
                   void* stream) {
    if (rows <= 0) return;
    if (cols % 256 || (ld & 3) || (reinterpret_cast<uintptr_t>(x) & 15) || (reinterpret_cast<uintptr_t>(act) & 15)) {
        basalt::log::error("basalt", "fp4 dense: quantize %lld columns, ld %lld", (long long) cols, (long long) ld);
        std::exit(1);
    }
    uint8_t* codes = (uint8_t*) act;
    uint8_t* scales = codes + (size_t) rows * (size_t) (cols / 2);
    const unsigned grid = (unsigned) ((padded(rows) + 7) / 8);
    quantize_kernel<false, true, __nv_bfloat16><<<grid, 256, 0, (cudaStream_t) stream>>>(reinterpret_cast<const __nv_bfloat16*>(x), codes, scales, gs, (int) cols, ld, rows, nullptr, 0.0f,
                                                                          static_cast<float*>(nullptr), rowmax);
    ck(cudaGetLastError(), "quantize_amax");
}

void rms_norm_quantize(const float* x, const float* w, float eps, int64_t rows, int64_t cols, void* act, float* gs, float* xn,
                       void* stream) {
    if (rows <= 0) return;
    if (cols % 256 || (reinterpret_cast<uintptr_t>(x) & 15) || (reinterpret_cast<uintptr_t>(act) & 15)) {
        basalt::log::error("basalt", "fp4 dense: rms_norm_quantize %lld columns", (long long) cols);
        std::exit(1);
    }
    uint8_t* codes = (uint8_t*) act;
    uint8_t* scales = codes + (size_t) rows * (size_t) (cols / 2);
    const unsigned grid = (unsigned) ((padded(rows) + 7) / 8);
    quantize_kernel<true><<<grid, 256, 0, (cudaStream_t) stream>>>(x, codes, scales, gs, (int) cols, cols, rows, w, eps, xn);
    ck(cudaGetLastError(), "rms_norm_quantize");
}

void product(const void* w, const float* w_scale, int64_t rows, int64_t K, const void* act, const float* gs, int64_t tokens,
             float* dst, int64_t ld, bool add, void* stream) {
    check(w, w, rows, K, tokens, act, dst, ld, "product");
    Params p{};
    p.gs = gs;
    p.ws[0] = w_scale;
    p.dst = dst;
    p.ld = ld;
    if (add) launch<false, true>(p, w, nullptr, rows, K, act, tokens, (cudaStream_t) stream);
    else launch<false, false>(p, w, nullptr, rows, K, act, tokens, (cudaStream_t) stream);
}

void product_bf16(const void* w, const float* w_scale, int64_t rows, int64_t K, const void* act, const float* gs,
                  int64_t tokens, uint16_t* dst, int64_t ld, void* stream) {
    check(w, w, rows, K, tokens, act, dst, ld, "product_bf16", true, gs);
    Params p{};
    p.gs = gs;
    p.ws[0] = w_scale;
    p.dst = reinterpret_cast<float*>(dst);
    p.ld = ld;
    launch<false, false, true>(p, w, nullptr, rows, K, act, tokens, (cudaStream_t) stream);
}

void product_add_bf16(const void* w, const float* w_scale, int64_t rows, int64_t K, const void* act, const float* gs,
                      int64_t tokens, uint16_t* dst, int64_t ld, void* stream) {
    check(w, w, rows, K, tokens, act, dst, ld, "product_add_bf16", true, gs);
    Params p{};
    p.gs = gs;
    p.ws[0] = w_scale;
    p.dst = reinterpret_cast<float*>(dst);
    p.ld = ld;
    launch<false, true, true>(p, w, nullptr, rows, K, act, tokens, (cudaStream_t) stream);
}

void rms_norm_quantize(const uint16_t* x, const float* w, float eps, int64_t rows, int64_t cols, void* act, float* gs,
                       uint16_t* xn, void* stream) {
    if (rows <= 0) return;
    if (cols % 256 || (reinterpret_cast<uintptr_t>(x) & 15) || (reinterpret_cast<uintptr_t>(act) & 15) ||
        (xn && (reinterpret_cast<uintptr_t>(xn) & 15))) {
        basalt::log::error("basalt", "fp4 dense: rms_norm_quantize %lld columns", (long long) cols);
        std::exit(1);
    }
    uint8_t* codes = (uint8_t*) act;
    uint8_t* scales = codes + (size_t) rows * (size_t) (cols / 2);
    const unsigned grid = (unsigned) ((padded(rows) + 7) / 8);
    quantize_kernel<true, false, __nv_bfloat16, __nv_bfloat16><<<grid, 256, 0, (cudaStream_t) stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x), codes, scales, gs, (int) cols, cols, rows, w, eps,
        reinterpret_cast<__nv_bfloat16*>(xn));
    ck(cudaGetLastError(), "rms_norm_quantize");
}

void swiglu(const void* w_gate, const void* w_up, const float* s_gate, const float* s_up, int64_t rows, int64_t K,
            const void* act, const float* gs, int64_t tokens, uint16_t* dst, int64_t ld, void* stream, float* amax) {
    check(w_gate, w_up, rows, K, tokens, act, dst, ld, "swiglu", true, gs);
    Params p{};
    p.amax = amax;
    if (amax && cudaMemsetAsync(amax, 0, (size_t) tokens * sizeof(float), (cudaStream_t) stream) != cudaSuccess)
        ck(cudaGetLastError(), "swiglu amax");
    p.gs = gs;
    p.ws[0] = s_gate;
    p.ws[1] = s_up;
    p.hdst = reinterpret_cast<__nv_bfloat16*>(dst);
    p.ld = ld;
    launch<true, false>(p, w_gate, w_up, rows, K, act, tokens, (cudaStream_t) stream);
}

}  // namespace basalt::prefill::fp4_dense
