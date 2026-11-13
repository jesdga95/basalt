// src/prefill/fp4_dense.cu - see include/basalt/prefill/fp4_dense.hpp.
// The tile structure follows Infernix's nvfp4_a4_tma_kernel (src/ops/linear/nvfp4/nvfp4_a4_tma.cuh, nvfp4_schedule.cuh,
// src/ops/common/mbarrier.cuh, mma.cuh): TMA loads into a mbarrier full/empty pipeline, a producer warpgroup beside
// eight consumer warps (setmaxnreg), the block-scaled m16n8k64 MMA, 128 weight rows x 256 tokens x K 128 per stage,
// token tiles walking fastest so the CTAs that run together share a weight tile.
// Adapted from Infernix (Ian Ranson and contributors), Apache-2.0: see THIRD_PARTY_LICENSES.
#include "basalt/prefill/fp4_dense.hpp"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "basalt/log.hpp"

namespace basalt::prefill::fp4_dense {
namespace {

constexpr int kBT = 256;                  // tokens per block
constexpr int kBR = 128;                  // weight rows per block (SwiGLU: 64 gate rows and the same 64 up rows)
constexpr int kStages = 3;
constexpr int kConsumerWarps = 8;         // 4 token quarters x 2 row halves
constexpr int kProducerThreads = 128;     // one warpgroup, so setmaxnreg can move registers to the consumers
constexpr int kThreads = kConsumerWarps * 32 + kProducerThreads;
constexpr int kWarpTokens = kBT / 4, kWarpRows = kBR / 2;
constexpr int kMmaTokens = kWarpTokens / 16, kMmaRows = kWarpRows / 8;
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
    const float* gs;
    const float* ws[2];   // the matrix scales (SwiGLU: gate, up)
    float* dst;
    int64_t ld;
    int tokens, K, tok_tiles;
};
static_assert(offsetof(Params, a_codes) % 64 == 0 && sizeof(TmaDesc) % 64 == 0, "descriptors sit on the map boundary");

struct alignas(128) Shared {
    alignas(128) uint8_t a_codes[kStages][kBT * 64];
    alignas(128) uint8_t b_codes[kStages][kBR * 64];
    alignas(16) uint32_t a_scale4[kScaleSlots][kBT * 4];
    alignas(16) uint8_t b_scales[kStages][2][kBR * kK64 * 4];
    alignas(8) uint64_t full[kStages];
    alignas(8) uint64_t empty[kStages];
};
constexpr int kSmem = (int) sizeof(Shared);
static_assert(kSmem <= 99 * 1024, "one block's shared memory");

template <int N, class F> __device__ __forceinline__ void with_live(int live, F&& f) {
    if (live >= N) f(std::integral_constant<int, N>{});
    else if constexpr (N > 1) with_live<N - 1>(live, f);
}

// SWIGLU: a block's 128 rows are 64 gate rows and the same 64 up rows (two descriptor sets, each warp holds the gate and
// the up product of the same outputs), and the output is silu(s_gate g) (s_up u).  ADD: dst += the product.
template <bool SWIGLU, bool ADD>
__global__ __launch_bounds__(kThreads, 1) void kernel(const __grid_constant__ Params p) {
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
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if (threadIdx.x < kProducerThreads) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" ::: "memory");
        if (threadIdx.x == 0) {
#pragma unroll 1
            for (int kt = 0; kt < ktiles; ++kt) {
                const int stage = kt % kStages;
                mbar_wait(&sh.empty[stage], 1u ^ ((kt / kStages) & 1u));
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
        return;
    }

    // The epilogue straight from the accumulators: thread (lane) holds tokens lane / 4 and lane / 4 + 8 of each 16-token
    // fragment, weight rows 2 * (lane & 3) and +1 of each 8-row fragment.  The scale is a rounded multiply, and the
    // add after it a separate rounded add, so a column's bits do not depend on whether its tile is full.
    const float s0 = *p.ws[0];
    const float s1 = SWIGLU ? *p.ws[1] : 0.0f;
#pragma unroll
    for (int mm = 0; mm < kMmaTokens; ++mm)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int t = token_begin + warp_m * kWarpTokens + mm * 16 + (lane >> 2) + 8 * h;
            if (t >= p.tokens) continue;
            const float g = p.gs[t];
            float* out = p.dst + (int64_t) t * p.ld;
            if constexpr (SWIGLU) {
                const float ag = __fmul_rn(g, s0), au = __fmul_rn(g, s1);
                const int r0 = row_begin + warp_n * 32 + 2 * (lane & 3);
#pragma unroll
                for (int f = 0; f < 4; ++f) {
                    float v[2];
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const float gg = __fmul_rn(acc[mm][f][2 * h + e], ag);
                        const float uu = __fmul_rn(acc[mm][f + 4][2 * h + e], au);
                        v[e] = gg / (1.0f + __expf(-gg)) * uu;
                    }
                    *reinterpret_cast<float2*>(out + r0 + 8 * f) = make_float2(v[0], v[1]);
                }
            } else {
                const float alpha = __fmul_rn(g, s0);
                const int r0 = row_begin + warp_n * kWarpRows + 2 * (lane & 3);
#pragma unroll
                for (int f = 0; f < kMmaRows; ++f) {
                    float2 v = make_float2(__fmul_rn(acc[mm][f][2 * h], alpha), __fmul_rn(acc[mm][f][2 * h + 1], alpha));
                    float2* d = reinterpret_cast<float2*>(out + r0 + 8 * f);
                    if constexpr (ADD) {
                        const float2 o = *d;
                        v = make_float2(__fadd_rn(o.x, v.x), __fadd_rn(o.y, v.y));
                    }
                    *d = v;
                }
            }
        }
}

// One warp per token row (and per padding row of the last scale tile).  Token scale g = amax / (6 * 448); per 16 values
// a UE4M3 scale of block amax / (6 g); codes round to nearest even on x / (scale * g), value 2b in byte b's low nibble.
// NORM: the row is RMSNorm'd first (x * rsqrt(mean(x^2) + eps) * nw), the normed row optionally written to xn.
template <bool NORM>
__global__ void quantize_kernel(const float* __restrict__ x, uint8_t* __restrict__ codes, uint8_t* __restrict__ scales,
                                float* __restrict__ gs, int cols, int64_t ld, int64_t rows, const float* __restrict__ nw = nullptr,
                                float eps = 0.0f, float* __restrict__ xn = nullptr) {
    const int lane = threadIdx.x & 31;
    const int64_t r = (int64_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int nb = cols / 16, ktiles = cols / 256;
    const int64_t tile_base = (r / kBT) * ktiles;
    uint8_t* srow = scales + (r % kBT) * 16;
    if (r >= rows) {   // padding of the last token tile: its scales are defined (zero), no codes are read
        for (int kt = lane; kt < ktiles; kt += 32) *reinterpret_cast<uint4*>(srow + (tile_base + kt) * kScaleTileBytes) = make_uint4(0, 0, 0, 0);
        return;
    }
    const float* xr = x + r * ld;
    float inv_rms = 1.0f;
    if (NORM) {
        float ss = 0.0f;
        for (int b = lane; b < nb; b += 32) {
            const float4* q = (const float4*) (xr + 16 * b);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 v = q[i];
                ss = fmaf(v.x, v.x, fmaf(v.y, v.y, fmaf(v.z, v.z, fmaf(v.w, v.w, ss))));
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        inv_rms = rsqrtf(ss / (float) cols + eps);
    }
    // value c of the row as the quantizer sees it
    auto val4 = [&](int c) -> float4 {
        float4 v = *(const float4*) (xr + c);
        if (NORM) {
            const float4 w = *(const float4*) (nw + c);
            v = make_float4(v.x * inv_rms * w.x, v.y * inv_rms * w.y, v.z * inv_rms * w.z, v.w * inv_rms * w.w);
        }
        return v;
    };
    float amax = 0.0f;
    for (int b = lane; b < nb; b += 32) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float4 v = val4(16 * b + 4 * i);
            if (NORM && xn) *(float4*) (xn + r * cols + 16 * b + 4 * i) = v;
            amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
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
            asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %2, %1; cvt.u16.u8 %0, t; }"
                : "=h"(c) : "f"(v[2 * j] * inv), "f"(v[2 * j + 1] * inv));   // the first source is the high nibble
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
               CUtensorMapSwizzle swizzle, CUtensorMapL2promotion promo, const char* what) {
    const EncodeFn f = encode_fn();
    if (f == nullptr) {
        basalt::log::error("basalt", "fp4 dense: cuTensorMapEncodeTiled is unavailable");
        std::exit(1);
    }
    CUtensorMap m;
    const cuuint64_t dim[2] = {cols, rows}, str[1] = {stride};
    const cuuint32_t box[2] = {box_c, box_r}, es[2] = {1, 1};
    if (f(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base), dim, str, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
          swizzle, promo, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) {
        basalt::log::error("basalt", "fp4 dense: tensor map of %s failed", what);
        std::exit(1);
    }
    TmaDesc d;
    std::memcpy(&d, &m, sizeof d);
    return d;
}

int64_t padded(int64_t tokens) { return (tokens + kBT - 1) / kBT * kBT; }

template <bool SWIGLU, bool ADD> void launch(Params p, const void* w0, const void* w1, int64_t rows, int64_t K,
                                             const void* act, int64_t tokens, cudaStream_t s) {
    constexpr int loaded_rows = kBR / (SWIGLU ? 2 : 1);
    static std::atomic<bool> configured[64];   // the shared-memory opt-in is per device and per kernel
    int dev = 0;
    ck(cudaGetDevice(&dev), "device");
    if (dev < 0 || dev >= 64) ck(cudaErrorInvalidDevice, "device range");
    if (!configured[dev].load(std::memory_order_acquire)) {
        ck(cudaFuncSetAttribute(kernel<SWIGLU, ADD>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem), "smem");
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
    p.K = (int) K;
    p.tokens = (int) tokens;
    p.tok_tiles = (int) ((tokens + kBT - 1) / kBT);
    const int64_t row_tiles = rows / loaded_rows;
    kernel<SWIGLU, ADD><<<(unsigned) (row_tiles * p.tok_tiles), kThreads, kSmem, s>>>(p);
    ck(cudaGetLastError(), "product");
}

void check(const void* a, const void* b, int64_t rows, int64_t K, int64_t tokens, const void* act, const float* dst, int64_t ld,
           const char* what) {
    const auto al = [](const void* v, size_t n) { return (reinterpret_cast<uintptr_t>(v) & (n - 1)) == 0; };
    if (rows <= 0 || rows % 128 || K <= 0 || K % 256 || tokens <= 0 || ld < rows || (ld & 1) || !al(a, 16) || !al(b, 16) ||
        !al(act, 16) || !al(dst, 8)) {
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

void swiglu(const void* w_gate, const void* w_up, const float* s_gate, const float* s_up, int64_t rows, int64_t K,
            const void* act, const float* gs, int64_t tokens, float* dst, int64_t ld, void* stream) {
    check(w_gate, w_up, rows, K, tokens, act, dst, ld, "swiglu");
    Params p{};
    p.gs = gs;
    p.ws[0] = s_gate;
    p.ws[1] = s_up;
    p.dst = dst;
    p.ld = ld;
    launch<true, false>(p, w_gate, w_up, rows, K, act, tokens, (cudaStream_t) stream);
}

}  // namespace basalt::prefill::fp4_dense
