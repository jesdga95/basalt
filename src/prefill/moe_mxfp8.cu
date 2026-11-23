// src/prefill/moe_mxfp8.cu - see include/basalt/prefill/moe_mxfp8.hpp.
#include "basalt/prefill/moe_mxfp8.hpp"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdlib>

#include "basalt/log.hpp"

namespace basalt::prefill::mxfp8 {
namespace {

constexpr int kTileTokens = 64;
constexpr int kBlockTiles = 4;
constexpr int kBT = kBlockTiles * kTileTokens;   // 256 tokens per CTA
constexpr int kBR = 64;                          // 64 weight rows per CTA (32 gate + 32 up for SWIGLU)
constexpr int kBK = 128;                         // 128 K per stage (= 1 block_mxfp8 = 4 x k32 MMA)
constexpr int kThreads = 256;                    // 8 warps: 4 (tokens) x 2 (weight rows)
constexpr int kNS = 2;                           // double-buffered cp.async stages
constexpr int kBlockBytes = 132;                 // sizeof(block_mxfp8): 4 UE8M0 + 128 E4M3
constexpr int kBRowStride = 144;                 // 16B-aligned staging per weight row (covers pad <= 12 + 132)
constexpr int kACodeBytes = kBT * kBK;           // 32768 B
constexpr int kAScaleBytes = kBT * 4;            // 1024 B (1 uint32_t = 4 UE8M0 per row)
constexpr int kAStage = kACodeBytes + kAScaleBytes;
constexpr int kBStage = kBR * kBRowStride;       // 9216 B
constexpr int kSmemBytes = kNS * (kAStage + kBStage);   // 86016 B (< 99 KiB)

void ck(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "prefill mxfp8: %s: %s", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

size_t scale_off(int64_t rows, int64_t cols) { return ((size_t) rows * (size_t) cols + 15) & ~(size_t) 15; }

__device__ __forceinline__ unsigned smem_addr(const void* p) { return (unsigned) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldsm4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void cp_async4(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(smem_addr(dst)), "l"(src));
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr(dst)), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ int swa(int rr, int lb) { return rr * kBK + (lb ^ ((rr & 7) << 4)); }

template <int KS>
__device__ __forceinline__ void mma_mxfp8(float* c, const unsigned* a, unsigned b0, unsigned b1, unsigned sfa,
                                          unsigned sfb) {
    constexpr unsigned short kBid = (unsigned short) KS;
    constexpr unsigned short kTid = 0;
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
                   "r"(sfa), "h"(kBid), "h"(kTid), "r"(sfb), "h"(kBid), "h"(kTid));
}

// ---------------------------------------------------------------------------------------- planning & quantizer

__global__ void plan_blocks_kernel(const int32_t* __restrict__ bounds, int n, int32_t* __restrict__ blk_off) {
    __shared__ int32_t bs[1024];
    const int j = threadIdx.x;
    const int cnt = j < n ? bounds[j + 1] - bounds[j] : 0;
    bs[j] = (cnt + kBT - 1) / kBT;
    __syncthreads();
    for (int o = 1; o < (int) blockDim.x; o <<= 1) {
        const int32_t v = j >= o ? bs[j - o] : 0;
        __syncthreads();
        bs[j] += v;
        __syncthreads();
    }
    if (j < n) blk_off[j + 1] = bs[j];
    if (j == 0) blk_off[0] = 0;
}

__device__ __forceinline__ float fast_tanh(float x) {
    float r;
    asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float fast_sigmoid(float x) {
    float t, e, d, r;
    asm("mul.rn.ftz.f32 %0, %1, 0fBFB8AA3B;" : "=f"(t) : "f"(x));
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(t));
    asm("add.rn.ftz.f32 %0, %1, 0f3F800000;" : "=f"(d) : "f"(e));
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(d));
    return r;
}

__device__ __forceinline__ void quant_block32(float v, int lane, uint8_t* code_out, uint8_t* scale_out) {
    float amax = fabsf(v);
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, m));
    uint8_t sb = 0u;
    float inv = 0.0f;
    if (amax > 0.0f) {
        const float ratio = amax * (1.0f / 448.0f);
        const uint32_t bits = __float_as_uint(ratio);
        int e = (int) ((bits >> 23) & 0xFFu) - 127 + ((bits & 0x7FFFFFu) != 0u ? 1 : 0);
        if (e < -127) e = -127;
        if (e > 127) e = 127;
        sb = (uint8_t) (e + 127);
        inv = sb == 0u ? __uint_as_float(0x7F000000u) : __uint_as_float((uint32_t) (254 - sb) << 23);
    }
    *code_out = (uint8_t) __nv_cvt_float_to_fp8(v * inv, __NV_SATFINITE, __NV_E4M3);
    if (lane == 0) *scale_out = sb;
}

template <typename T>
__global__ void quantize_mxfp8_kernel(const T* __restrict__ x, int64_t rows, int64_t cols, int64_t ld,
                                      uint8_t* __restrict__ codes, uint8_t* __restrict__ scales) {
    const uint32_t bpr = (uint32_t) (cols >> 5);
    const uint32_t blk = ((uint32_t) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if ((int64_t) blk >= rows * (int64_t) bpr) return;
    const int64_t r = (int64_t) (blk / bpr);
    const int64_t b = (int64_t) (blk - (uint32_t) r * bpr);
    float v;
    if constexpr (sizeof(T) == 4) {
        v = x[r * ld + (b << 5) + lane];
    } else {
        __half h;
        const uint16_t raw = x[r * ld + (b << 5) + lane];
        memcpy(&h, &raw, 2);
        v = __half2float(h);
    }
    quant_block32(v, lane, codes + r * cols + (b << 5) + lane, scales + blk);
}

// ---------------------------------------------------------------------------------------- GEMM

struct GemmArgs {
    const uint8_t* base;
    size_t stride, w_off;
    const uint8_t* a_codes;
    const uint8_t* a_scales;
    const int32_t *bounds, *blk_off, *ids, *dst_ids;
    float* dst;
    uint16_t* dst_f16;
    int64_t ld_dst;
    int n, w_cols, half;
    float gate_softcap, up_softcap;
};

template <bool SWIGLU>
__device__ __forceinline__ int b_row(int rb, int sr, int half) {
    return SWIGLU ? (sr < 32 ? rb * 32 + sr : half + rb * 32 + (sr - 32)) : rb * kBR + sr;
}
template <bool SWIGLU>
__device__ __forceinline__ int b_tile_row(int warp_n, int r) {
    return SWIGLU ? (r < 2 ? warp_n * 16 + r * 8 : 32 + warp_n * 16 + (r - 2) * 8) : warp_n * 32 + r * 8;
}

template <bool GATHER, bool SWIGLU>
__device__ __forceinline__ void load_stage(int stage, int kt, uint8_t* smem, const int32_t* tok_s, int ntiles,
                                           int brow0, int cnt, int first, const uint8_t* a_codes,
                                           const uint8_t* a_scales, int w_cols, const uint8_t* w_src,
                                           size_t row_bytes, int rb, int half) {
    uint8_t* as = smem + stage * kAStage;
    uint8_t* bs = smem + kNS * kAStage + stage * kBStage;
    const int n_rows = ntiles * kTileTokens;
    for (int i = threadIdx.x; i < n_rows * 8; i += kThreads) {
        const int r = i >> 3, q = i & 7;
        const int src_row = GATHER ? tok_s[r] : (brow0 + (first + r < cnt ? r : 0));
        cp_async16(as + swa(r, q << 4), a_codes + (size_t) src_row * (size_t) w_cols + (size_t) kt * kBK + (q << 4));
    }
    if ((int) threadIdx.x < n_rows) {
        const int r = threadIdx.x;
        const int src_row = GATHER ? tok_s[r] : (brow0 + (first + r < cnt ? r : 0));
        cp_async4(as + kACodeBytes + r * 4, a_scales + (size_t) src_row * (size_t) (w_cols >> 5) + (size_t) kt * 4);
    }
    for (int i = threadIdx.x; i < kBR * 9; i += kThreads) {
        const int row = i / 9, part = i - row * 9;
        const uint8_t* src = w_src + (size_t) b_row<SWIGLU>(rb, row, half) * row_bytes + (size_t) kt * kBlockBytes;
        const uint8_t* src_al = reinterpret_cast<const uint8_t*>(reinterpret_cast<uintptr_t>(src) & ~(uintptr_t) 15);
        cp_async16(bs + row * kBRowStride + part * 16, src_al + part * 16);
    }
}

template <bool GATHER, bool SWIGLU>
__global__ __launch_bounds__(kThreads, 1) void gemm_kernel(const GemmArgs a) {
    extern __shared__ __align__(1024) uint8_t smem[];
    __shared__ int32_t tok_s[kBT];

    const int y = blockIdx.y;
    if (y >= a.blk_off[a.n]) return;
    int lo = 0, hi = a.n;
    while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (a.blk_off[mid] <= y) lo = mid; else hi = mid;
    }
    const int j = lo;
    const int tp = y - a.blk_off[lo];
    const int rb = blockIdx.x;
    const int cnt = a.bounds[j + 1] - a.bounds[j];
    const int first = tp * kBT;
    if (first >= cnt) return;
    const int ntiles = min(kBlockTiles, (cnt - first + kTileTokens - 1) / kTileTokens);
    const int nkt = a.w_cols / kBK;
    const size_t row_bytes = (size_t) nkt * kBlockBytes;
    const uint8_t* w_src = a.base + (size_t) j * a.stride + a.w_off;
    const int brow0 = a.bounds[j] + first;

    if constexpr (GATHER) {
        if ((int) threadIdx.x < ntiles * kTileTokens)
            tok_s[threadIdx.x] = a.ids[brow0 + (first + (int) threadIdx.x < cnt ? (int) threadIdx.x : 0)];
        __syncthreads();
    }

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int warp_m = warp >> 1, warp_n = warp & 1;   // 4 token tiles x 2 row halves
    const bool live = warp_m < ntiles;
    const int group = lane >> 2, tig = lane & 3;
    const int a_row = (lane & 7) + (((lane >> 3) & 1) << 3);
    const int a_col = ((lane >> 4) & 1) * 16;
    const int sfa_row = ((lane & 1) << 3) | group;

    float acc[4][4][4] = {};

    if (0 < nkt)
        load_stage<GATHER, SWIGLU>(0, 0, smem, tok_s, ntiles, brow0, cnt, first, a.a_codes, a.a_scales, a.w_cols,
                                   w_src, row_bytes, rb, a.half);
    cp_commit();

    for (int kt = 0; kt < nkt; ++kt) {
        if (kt + 1 < nkt)
            load_stage<GATHER, SWIGLU>((kt + 1) & 1, kt + 1, smem, tok_s, ntiles, brow0, cnt, first, a.a_codes,
                                       a.a_scales, a.w_cols, w_src, row_bytes, rb, a.half);
        cp_commit();
        cp_wait<1>();
        __syncthreads();

        if (live) {
            const int st = kt & 1;
            const uint8_t* as = smem + st * kAStage;
            const uint8_t* bs = smem + kNS * kAStage + st * kBStage;
            const uint32_t* asc = reinterpret_cast<const uint32_t*>(as + kACodeBytes);

            unsigned sfa[4];
#pragma unroll
            for (int mt = 0; mt < 4; ++mt) sfa[mt] = asc[warp_m * 64 + mt * 16 + sfa_row];

            unsigned sfb[4];
            const uint8_t* bp[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int sr = b_tile_row<SWIGLU>(warp_n, r) + group;
                const uint8_t* src = w_src + (size_t) b_row<SWIGLU>(rb, sr, a.half) * row_bytes + (size_t) kt * kBlockBytes;
                const int pad = (int) (reinterpret_cast<uintptr_t>(src) & 15u);
                const uint8_t* p = bs + sr * kBRowStride + pad;
                bp[r] = p + 4;
                unsigned v = *reinterpret_cast<const uint32_t*>(p);
                sfb[r] = v & ~__vcmpeq4(v, 0xffffffffu);
            }

            auto step = [&]<int KS>() {
                unsigned af[4][4];
#pragma unroll
                for (int mt = 0; mt < 4; ++mt)
                    ldsm4(af[mt][0], af[mt][1], af[mt][2], af[mt][3],
                          smem_addr(as + swa(warp_m * 64 + mt * 16 + a_row, KS * 32 + a_col)));
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const uint8_t* q = bp[r] + KS * 32;
                    const unsigned b0 = *reinterpret_cast<const uint32_t*>(q + 4 * tig);
                    const unsigned b1 = *reinterpret_cast<const uint32_t*>(q + 16 + 4 * tig);
#pragma unroll
                    for (int mt = 0; mt < 4; ++mt) mma_mxfp8<KS>(acc[mt][r], af[mt], b0, b1, sfa[mt], sfb[r]);
                }
            };
            step.template operator()<0>();
            step.template operator()<1>();
            step.template operator()<2>();
            step.template operator()<3>();
        }
        __syncthreads();
    }
    cp_wait<0>();

    if (!live) return;

    const int base_m = warp_m * 64 + group;
    if constexpr (SWIGLU) {
        const bool sc = a.gate_softcap > 0.0f;
        const float cg = a.gate_softcap, cu = a.up_softcap;
        const float inv_cg = sc ? 1.0f / cg : 0.0f, inv_cu = sc ? 1.0f / cu : 0.0f;
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int lr = base_m + mt * 16 + h * 8;
                if (first + lr >= cnt) continue;
                const int64_t out_row = a.dst_ids ? (int64_t) a.dst_ids[brow0 + lr] : (int64_t) (brow0 + lr);
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    const int col = rb * 32 + warp_n * 16 + r * 8 + 2 * tig;
                    const float g0 = acc[mt][r][2 * h], g1 = acc[mt][r][2 * h + 1];
                    const float u0 = acc[mt][r + 2][2 * h], u1 = acc[mt][r + 2][2 * h + 1];
                    float v0, v1;
                    if (sc) {
                        v0 = __fmul_rn(__fmul_rn(cg * fast_tanh(g0 * inv_cg), fast_sigmoid(g0)), cu * fast_tanh(u0 * inv_cu));
                        v1 = __fmul_rn(__fmul_rn(cg * fast_tanh(g1 * inv_cg), fast_sigmoid(g1)), cu * fast_tanh(u1 * inv_cu));
                    } else {
                        v0 = __fmul_rn(__fmul_rn(g0, fast_sigmoid(g0)), u0);
                        v1 = __fmul_rn(__fmul_rn(g1, fast_sigmoid(g1)), u1);
                    }
                    if (a.dst_f16)
                        *reinterpret_cast<__half2*>(a.dst_f16 + out_row * a.ld_dst + col) = __floats2half2_rn(v0, v1);
                    else
                        *reinterpret_cast<float2*>(a.dst + out_row * a.ld_dst + col) = make_float2(v0, v1);
                }
            }
        }
    } else {
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int lr = base_m + mt * 16 + h * 8;
                if (first + lr >= cnt) continue;
                const int64_t out_row = a.dst_ids ? (int64_t) a.dst_ids[brow0 + lr] : (int64_t) (brow0 + lr);
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const int col = rb * kBR + warp_n * 32 + r * 8 + 2 * tig;
                    const float v0 = acc[mt][r][2 * h], v1 = acc[mt][r][2 * h + 1];
                    if (a.dst_f16)
                        *reinterpret_cast<__half2*>(a.dst_f16 + out_row * a.ld_dst + col) = __floats2half2_rn(v0, v1);
                    else
                        *reinterpret_cast<float2*>(a.dst + out_row * a.ld_dst + col) = make_float2(v0, v1);
                }
            }
        }
    }
}

}  // namespace

size_t act_bytes(int64_t rows, int64_t cols) { return scale_off(rows, cols) + (size_t) rows * (size_t) (cols >> 5); }

void plan_blocks(const int32_t* bounds, int n, int32_t* blk_off, void* stream) {
    if (n <= 0 || n > 1024) {
        basalt::log::error("basalt", "prefill mxfp8: plan_blocks n=%d", n);
        std::exit(1);
    }
    int p2 = 1;
    while (p2 < n) p2 <<= 1;
    plan_blocks_kernel<<<1, p2, 0, (cudaStream_t) stream>>>(bounds, n, blk_off);
    ck("plan_blocks");
}

void quantize_tokens(const float* x, int64_t rows, int64_t cols, int64_t ld, void* act, void* stream) {
    if (rows <= 0) return;
    if (cols % 128 != 0) {
        basalt::log::error("basalt", "prefill mxfp8: quantize_tokens cols %lld", (long long) cols);
        std::exit(1);
    }
    const int64_t nblk = rows * (cols >> 5);
    const unsigned grid = (unsigned) ((nblk + 7) / 8);
    auto* codes = static_cast<uint8_t*>(act);
    quantize_mxfp8_kernel<float><<<grid, 256, 0, (cudaStream_t) stream>>>(
        x, rows, cols, ld, codes, codes + scale_off(rows, cols));
    ck("quantize_tokens");
}

void quantize_f16(const uint16_t* x, int64_t rows, int64_t cols, int64_t ld, void* act, void* stream) {
    if (rows <= 0) return;
    if (cols % 128 != 0) {
        basalt::log::error("basalt", "prefill mxfp8: quantize_f16 cols %lld", (long long) cols);
        std::exit(1);
    }
    const int64_t nblk = rows * (cols >> 5);
    const unsigned grid = (unsigned) ((nblk + 7) / 8);
    auto* codes = static_cast<uint8_t*>(act);
    quantize_mxfp8_kernel<uint16_t><<<grid, 256, 0, (cudaStream_t) stream>>>(
        x, rows, cols, ld, codes, codes + scale_off(rows, cols));
    ck("quantize_f16");
}

void run(const Product& p, void* stream) {
    if (p.blocks <= 0 || p.n <= 0) return;
    if (p.w_rows % kBR != 0 || p.w_cols % kBK != 0 || p.n > 1024) {
        basalt::log::error("basalt", "prefill mxfp8: bad shape [%lld x %lld] n=%d",
                           (long long) p.w_rows, (long long) p.w_cols, p.n);
        std::exit(1);
    }
    static std::atomic<bool> raised[64];
    int dev = 0;
    cudaGetDevice(&dev);
    if (!raised[dev].load(std::memory_order_acquire)) {
        auto opt = [](auto fn) {
            if (cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes) != cudaSuccess) {
                basalt::log::error("basalt", "prefill mxfp8: shared-memory opt-in failed");
                std::exit(1);
            }
        };
        opt(gemm_kernel<true, true>);
        opt(gemm_kernel<false, true>);
        opt(gemm_kernel<true, false>);
        opt(gemm_kernel<false, false>);
        raised[dev].store(true, std::memory_order_release);
    }
    GemmArgs a{};
    a.base = p.base;
    a.stride = p.stride;
    a.w_off = p.w_off;
    a.a_codes = static_cast<const uint8_t*>(p.act);
    a.a_scales = a.a_codes + scale_off(p.act_rows, p.w_cols);
    a.bounds = p.bounds;
    a.blk_off = p.blk_off;
    a.ids = p.ids;
    a.dst_ids = p.dst_ids;
    a.dst = p.dst;
    a.dst_f16 = p.dst_f16;
    a.ld_dst = p.ld_dst;
    a.n = p.n;
    a.w_cols = (int) p.w_cols;
    a.half = (int) (p.w_rows / 2);
    a.gate_softcap = p.gate_softcap;
    a.up_softcap = p.up_softcap;
    const dim3 grid((unsigned) (p.w_rows / kBR), (unsigned) p.blocks);
    cudaStream_t s = (cudaStream_t) stream;
    if (p.ids && p.swiglu)       gemm_kernel<true, true><<<grid, kThreads, kSmemBytes, s>>>(a);
    else if (!p.ids && p.swiglu) gemm_kernel<false, true><<<grid, kThreads, kSmemBytes, s>>>(a);
    else if (p.ids && !p.swiglu) gemm_kernel<true, false><<<grid, kThreads, kSmemBytes, s>>>(a);
    else                         gemm_kernel<false, false><<<grid, kThreads, kSmemBytes, s>>>(a);
    ck("run");
}

}  // namespace basalt::prefill::mxfp8
