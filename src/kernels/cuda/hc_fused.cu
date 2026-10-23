// src/kernels/cuda/hc_fused.cu - see include/basalt/kernels/hc_fused.hpp.
//
// Built WITHOUT fast math, like fused_gr.cu, so the hc arithmetic below is fused_gr.cu's expressions compiled the
// same way.  The router part reproduces kernels built WITH fast math (the MMVF GEMV and the native top-10): those
// operations are spelled out as the flush-to-zero PTX instructions that build emits.
#include "basalt/core/diag.hpp"
#include "basalt/kernels/hc_fused.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/q8_1_finite.hpp"

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "basalt/log.hpp"

namespace basalt::kernels {
namespace {

constexpr int N = 2560;              // n_embd
constexpr int HC = 4;                // streams
constexpr int D = N * HC;            // 10240
constexpr int LR = 320;              // hc_lr
constexpr int NE = 512;              // experts
constexpr int KT = 10;               // routed per token
constexpr int MAXT = 8;
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int NRT = 81;              // router items: the experts split over this many CTAs
constexpr int SMEM = 98 * 1024;      // dynamic shared memory per CTA
constexpr int RT_MAX_ROWS = (NE + NRT - 1) / NRT;   // router rows per router item

// down items over half-streams (no norm tasks)
constexpr int CN_HS = 2 * HC;                  // half-streams
constexpr int CN_W = N / 2;                    // elements per half-stream
constexpr int CN_RG = 32;                      // w_down rows per item
constexpr int CN_G = LR / CN_RG;               // row groups
constexpr int CN_ND = CN_G * CN_HS + CN_HS;    // items: w_down's, then w_inject's (one per half-stream)
constexpr int CN_ROWS = LR + HC;               // dot products per (token, half-stream): w_down's rows, then w_inject's
constexpr int CN_SB = CN_W / 32 * 2;           // scale bytes per row and half-stream
constexpr int CN_WB = CN_RG * CN_W;            // weight bytes in shared memory
constexpr int CN_SWB = 4 * CN_SB;              // scale bytes per warp's four rows
constexpr int CN_SWT = 384;                    // tensor copies: a warp's scales on their own 128-byte aligned run
constexpr int CN_XOFF = CN_WB + WARPS * CN_SWT; // x after the weights and scales
constexpr int CN_MIN_NU = N / 32;              // up slices of 32 columns at most
constexpr int CN_PART = MAXT * CN_HS * CN_ROWS;   // floats of the dot products
constexpr int CN_SSN = MAXT * CN_HS;              // floats of the sums of squares
static_assert(LR % CN_RG == 0 && CN_W % 256 == 0 && CN_XOFF % 16 == 0 && CN_SWT >= CN_SWB && CN_SWT % 128 == 0 &&
              CN_WB % 128 == 0 && (4 * CN_W) % 128 == 0, "cn geometry");
static_assert(CN_XOFF + MAXT * CN_W * 4 <= SMEM, "cn item fits");
static_assert(32 * (4 * LR + 16) + 128 + 4 * 32 * 16 * 2 + (MAXT * (LR + 8) + 3 * MAXT * 4 * 32) * 4 + 10240 + 3 * (2 * 512 + 20) * 4 <= SMEM,
              "an 8-bit mma up slice of 32 columns with its B fragments and the top-10's dry run");

// One q8_1 block (iq_kernels.cu's block_q8_1) and its q8_1_store: lane i holds value i of block ib.
struct HcQ81 {
    half2 ds;
    int8_t qs[32];
};
__device__ __forceinline__ void hc_q8_1_store(float xi, HcQ81* __restrict__ y, int ib, int lane) {
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    const float d = q8_1_finite(amax / 127.0f);
    y[ib].qs[lane] = q8_1_quant(xi, d, amax);
    if (lane == 0) y[ib].ds = q8_1_ds(d, sum);
}

// native_quantize_q8_1's arithmetic (its file is built with --use_fast_math: approximate flush-to-zero divisions)
__device__ __forceinline__ float div_approx_ftz(float a, float b) {
    float r;
    asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ void hc_q8_1_store_native(float xi, HcQ81* __restrict__ y, int ib, int lane) {
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    const float d = q8_1_finite(div_approx_ftz(amax, 127.0f));
    int8_t q = 0;
    if (amax != 0.0f) {
        const float v = roundf(div_approx_ftz(xi, d));
        q = (int8_t) (v > 127.0f ? 127.0f : v < -127.0f ? -127.0f : v);
    }
    y[ib].qs[lane] = q;
    if (lane == 0) y[ib].ds = q8_1_ds(d, sum);
}

// A read without a router that writes xq, up slices of whole q8_1 blocks: the epilogue's warps write them (each warp
// stores 32 consecutive values of one token: T * nc is a multiple of 32 and THREADS of 32)
__device__ __forceinline__ bool up_q8_inline(const HcFusedArgs& a, int c0, int nc) {
    return a.xq != nullptr && a.w_router == nullptr && nc == 32 && c0 % 32 == 0;
}

// A read without a router that writes xq: per q8_1 block of a token row, the columns of the up slices done so far
// (the slice that completes a block quantizes it for every token and zeroes its count)
__device__ unsigned g_hc_q8cnt[4][N / 32];

// The up slice's mixed columns are written: count them per q8_1 block, and the slice that completes a block writes its
// q8_1 rows (on the stored values: the same bits as native_quantize_q8_1).
template <int T>
__device__ void hc_up_q8(const HcFusedArgs& a, int c0, int nc, int tid, unsigned* __restrict__ cnt) {
    __shared__ int s_done[4];
    const int j0 = c0 / 32, nj = (c0 + nc - 1) / 32 - j0 + 1;   // nc <= 32 * 3 + 1
    __threadfence();
    __syncthreads();
    if (tid < nj) {
        const int j = j0 + tid, lo = max(c0, 32 * j), hi = min(c0 + nc, 32 * j + 32);
        const unsigned n = (unsigned) (hi - lo);
        const bool done = atomicAdd(&cnt[j], n) + n == 32u;
        if (done) {
            cnt[j] = 0;
            __threadfence();
        }
        s_done[tid] = done;
    }
    __syncthreads();
    for (int w = tid >> 5; w < nj * T; w += WARPS) {
        const int jj = w / T, k = w - jj * T;
        if (!s_done[jj]) continue;
        const int q = k * (N / 32) + j0 + jj;
        hc_q8_1_store_native(__ldcg(a.mixed + (size_t) q * 32 + (tid & 31)), static_cast<HcQ81*>(a.xq), q, tid & 31);
    }
}

// each counter in its own 256-byte block: the CTAs polling one do not queue behind the atomics on another
struct alignas(256) HcCtr {
    unsigned v;
};
struct HcSync {
    HcCtr q, down, up, exit, epoch;
};
// consecutive reads take different counter sets: a read's CTAs claim work before the previous kernel is done
constexpr int NSLOT = 4;
__device__ HcSync g_hc_sync[NSLOT];
static_assert(sizeof(g_hc_q8cnt) / sizeof(g_hc_q8cnt[0]) == NSLOT, "a q8_1 count set per counter set");

// A tensor map as a kernel argument. The descriptor type is aligned more strictly than the hardware needs on newer
// toolkits, and a by-value argument with that alignment is not representable on every host compiler, so the argument
// carries the bytes at the boundary the hardware requires and the kernel reads them as descriptors.
struct alignas(64) TmaDesc {
    unsigned long long w[16];
};
static_assert(sizeof(TmaDesc) == sizeof(CUtensorMap), "a descriptor argument is the descriptor's size");

struct HcLaunch {
    HcFusedArgs a;
    int nb;          // CTAs (one per SM)
    int slot;        // counter set
    int nu;          // up slices (0: every CTA past the down items)
    unsigned long long* dbg;   // bench only: per-CTA phase times [nb][16]
    int early;       // 1: the next kernel let in at the start
    float* cnp;      // [MAXT][CN_HS][CN_ROWS] the down items' dot products
    float* cns;      // [MAXT][CN_HS] the half-streams' sums of squares
    float* t10_dummy;                      // [NE] fixed logits for the top-10's dry run
    float* t10_scratch;                    // [nb][2 * KT] its outputs
    unsigned long long* lg64;              // [MAXT][NE] logits tagged with the read's epoch (tag << 32 | f32 bits)
    TmaDesc tm[6];   // q_down, s_down, q_inject, s_inject by four rows; q_up, s_up
};

// cuda.h gives the descriptor its alignas only when the compiler reports a C++11 __cplusplus, which MSVC does not
// unless the build passes /Zc:__cplusplus. A descriptor off the boundary faults when the kernel reads it, so the build
// says so at compile time rather than at launch. The boundary is named here rather than taken from the header, whose
// spelling of it changed between toolkit versions; 64 is what the hardware requires, and the newer toolkits align
// the type more strictly, which satisfies it.
static_assert(alignof(CUtensorMap) >= 64, "CUtensorMap lost its alignas: the build needs /Zc:__cplusplus");
static_assert(offsetof(HcLaunch, tm) % 64 == 0, "the TMA descriptors must sit on the map boundary");

// ---------------------------------------------------------------- primitives
__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint64_t* bar, unsigned count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, unsigned ph) {
    unsigned done = 0;
    while (!done) {
        asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                     : "=r"(done) : "r"(smem_u32(bar)), "r"(ph) : "memory");
    }
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, unsigned bytes, uint64_t* bar) {
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
// A 2D tensor copy (box of the map) at element coordinates (x, y) into shared memory, completing on bar.
__device__ __forceinline__ void tma_2d(void* dst, const CUtensorMap* map, int x, int y, uint64_t* bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 ::"r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(map)), "r"(x), "r"(y), "r"(smem_u32(bar))
                 : "memory");
}
__device__ __forceinline__ void tma_3d(void* dst, const CUtensorMap* map, int x, int y, int z, uint64_t* bar) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
                 ::"r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(map)), "r"(x), "r"(y), "r"(z), "r"(smem_u32(bar))
                 : "memory");
}
__device__ __forceinline__ void tma_4d(void* dst, const CUtensorMap* map, int x, int y, int z, int w, uint64_t* bar) {
    asm volatile("cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5}], [%6];"
                 ::"r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(map)), "r"(x), "r"(y), "r"(z), "r"(w), "r"(smem_u32(bar))
                 : "memory");
}
__device__ __forceinline__ void prefetch_l2(const void* p, size_t bytes) {   // 16-byte aligned, multiple of 16
    const char* c = (const char*) p;
    for (size_t o = 0; o < bytes; o += 65536) {
        const unsigned sz = (unsigned) (bytes - o < 65536 ? bytes - o : 65536);
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(c + o), "r"(sz) : "memory");
    }
}
// DBG: the bench's phase times (a separate instantiation of the kernel); otherwise no code at all
template <bool DBG>
__device__ __forceinline__ void stamp(unsigned long long* dbg, int b, int i, int tid) {
    if constexpr (DBG) {
        if (dbg != nullptr && tid == 0) {
            unsigned long long t;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
            dbg[b * 16 + i] = t;
        }
    }
}
__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) {
    unsigned v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
// A value carried with the read's tag in one 64-bit access: a reader that sees the tag sees the value.
__device__ __forceinline__ unsigned long long ld_tagged_raw(const unsigned long long* p) {
    unsigned long long v;
    asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_tagged(unsigned long long* p, float v, unsigned tag) {
    const unsigned long long x = ((unsigned long long) tag << 32) | __float_as_uint(v);
    asm volatile("st.relaxed.gpu.global.u64 [%0], %1;" ::"l"(p), "l"(x) : "memory");
}
// Reads again, all together, every value of w whose tag is not `tag` (addr(i): its address), until none is missing.
template <int M, class F>
__device__ __forceinline__ void tagged_settle(unsigned long long (&w)[M], unsigned tag, const F& addr) {
    bool bad;
    do {
        bad = false;
#pragma unroll
        for (int i = 0; i < M; ++i)
            if ((unsigned) (w[i] >> 32) != tag) {
                w[i] = ld_tagged_raw(addr(i));
                bad = true;
            }
    } while (bad);
}
// every thread's writes so far, then one release increment
__device__ __forceinline__ void cta_signal(unsigned* ctr, int tid) {
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        atomicAdd(ctr, 1u);
    }
}
// fence false: the acquire load and the barrier order the CTA's later accesses (no full fence)
__device__ __forceinline__ void cta_wait(const unsigned* ctr, unsigned target, int tid, bool fence = true) {
    if (tid == 0) {
        while (ld_acquire(ctr) < target) {}
        if (fence) __threadfence();
    }
    __syncthreads();
}

// the flush-to-zero single-precision operations of a fast-math build
__device__ __forceinline__ float fma_ftz(float a, float b, float c) {
    float d;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
    return d;
}
__device__ __forceinline__ float add_ftz(float a, float b) {
    float d;
    asm("add.ftz.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}
__device__ __forceinline__ float sub_ftz(float a, float b) {
    float d;
    asm("sub.ftz.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}
__device__ __forceinline__ float mul_ftz(float a, float b) {
    float d;
    asm("mul.ftz.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}
__device__ __forceinline__ float max_ftz(float a, float b) {
    float d;
    asm("max.ftz.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}
__device__ __forceinline__ float ex2_ftz(float a) {
    float d;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a));
    return d;
}
__device__ __forceinline__ float rcp_ftz(float a) {
    float d;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a));
    return d;
}
__device__ __forceinline__ float expf_fast(float x) { return ex2_ftz(mul_ftz(x, 1.4426950216293335f)); }
__device__ __forceinline__ float warp_sum_ftz(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = add_ftz(v, __shfl_xor_sync(0xffffffffu, v, o, 32));
    return v;
}
__device__ __forceinline__ float warp_max_ftz(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = max_ftz(v, __shfl_xor_sync(0xffffffffu, v, o, 32));
    return v;
}

// fused_gr.cu's operations
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
// The 32-lane xor tree of `warp_sum`, for 16 values at once: four halving steps (offsets 16, 8, 4, 2) each send half
// of the values to the partner lane and add the other half, then one butterfly step (offset 1).  Every value is summed
// over exactly the pairs, in exactly the tree, of warp_sum (and a + b == b + a), so each total has warp_sum's bits;
// lanes 2m and 2m + 1 end with value m.
__device__ __forceinline__ float warp_sum16(float (&v)[16], int lane) {
#pragma unroll
    for (int o = 16, n = 8; o >= 2; o >>= 1, n >>= 1) {
        const bool h = (lane & o) != 0;
#pragma unroll
        for (int j = 0; j < n; ++j) {
            const float keep = h ? v[j + n] : v[j], send = h ? v[j] : v[j + n];
            v[j] = keep + __shfl_xor_sync(0xffffffffu, send, o);
        }
    }
    return v[0] + __shfl_xor_sync(0xffffffffu, v[0], 1);
}

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }
// 8-bit weights: int8 with one f16 scale per G8 consecutive values of a row
constexpr int G8 = 32;
__device__ __forceinline__ float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }
// four int8 as floats, exactly: q + 128 as the low byte of 2^23 + (q + 128), less 2^23 + 128
__device__ __forceinline__ float4 i8x4_f(uint32_t q) {
    const uint32_t u = q ^ 0x80808080u;
    return make_float4(__uint_as_float(__byte_perm(u, 0x4B000000u, 0x7650u)) - 8388736.0f,
                       __uint_as_float(__byte_perm(u, 0x4B000000u, 0x7651u)) - 8388736.0f,
                       __uint_as_float(__byte_perm(u, 0x4B000000u, 0x7652u)) - 8388736.0f,
                       __uint_as_float(__byte_perm(u, 0x4B000000u, 0x7653u)) - 8388736.0f);
}
// bytes 0, 1 (HI: 2, 3) of four int8 as an f16 pair, exactly: 1024 + (q + 128), less 1152
template <bool HI>
__device__ __forceinline__ uint32_t i8x2_h2(uint32_t q) {
    const uint32_t v = __byte_perm(q ^ 0x80808080u, 0x64646464u, HI ? 0x7362u : 0x5140u);
    const __half2 h = __hsub2(*reinterpret_cast<const __half2*>(&v), __halves2half2(__ushort_as_half(0x6480), __ushort_as_half(0x6480)));
    return *reinterpret_cast<const uint32_t*>(&h);
}
__device__ __forceinline__ float dot8ff(const float4 w0, const float4 w1, const float4 x0, const float4 x1) {
    float acc = 0.0f;
    acc = fmaf(w0.x, x0.x, acc);
    acc = fmaf(w0.y, x0.y, acc);
    acc = fmaf(w0.z, x0.z, acc);
    acc = fmaf(w0.w, x0.w, acc);
    acc = fmaf(w1.x, x1.x, acc);
    acc = fmaf(w1.y, x1.y, acc);
    acc = fmaf(w1.z, x1.z, acc);
    acc = fmaf(w1.w, x1.w, acc);
    return acc;
}

// ---------------------------------------------------------------- the stages (block index passed in)
// ---- cn down items: item d < CN_G * CN_HS takes rows [CN_RG (d / CN_HS), +CN_RG) of the 8-bit w_down over half-stream
// h = d % CN_HS (elements [CN_W h, +CN_W)), the later items w_inject's rows over half-stream d - CN_G * CN_HS.  The item
// forms R' = R + bo * gate and x = R' * w_norm of its half-stream itself, so it waits on no other item; it writes the
// dot products with x (rs not applied) and, for the first row group, the half-stream's sum of R'^2.  The up items then
// form lo = silu(sum_c rs_c (dot_2c + dot_2c+1) / hc) (cn_lo).
struct CnRows {
    const int8_t* q;
    const uint16_t* s;
    int row0, nrows, hs;
};
__device__ __forceinline__ CnRows cn_rows(const HcFusedArgs& a, int d) {
    if (d < CN_G * CN_HS) {
        const int g = d / CN_HS;
        return {a.q_down + (size_t) g * CN_RG * D, a.s_down + (size_t) g * CN_RG * (D / G8), g * CN_RG, CN_RG, d % CN_HS};
    }
    return {a.q_inject, a.s_inject, LR, HC, d - CN_G * CN_HS};
}
// Warp w's rows [4 w, 4 w + 4) (weights and scales of the half-stream) by two tensor copies into shared memory on
// bars[w], its scales at CN_SWT per warp; warps past the rows arrive alone.  False: no rows (no w_inject), the barriers
// untouched.
__device__ bool cn_issue(const HcFusedArgs& a, int d, int tid, char* smem, uint64_t* bars, const CUtensorMap* tm) {
    const CnRows w = cn_rows(a, d);
    if (w.q == nullptr) return false;
    const int warp = tid >> 5, lane = tid & 31;
    if (4 * warp >= w.nrows) {
        if (lane == 0) mbar_arrive(&bars[warp]);
        return true;
    }
    if (lane == 0) {
        const CUtensorMap* m = w.row0 < LR ? &tm[0] : &tm[2];
        const int y = (w.row0 < LR ? w.row0 : 0) + 4 * warp;
        uint64_t* bar = &bars[warp];
        mbar_expect_tx(bar, 4 * (CN_W + CN_SB));
        tma_2d(smem + 4 * warp * CN_W, m, w.hs * (CN_W / 8), y, bar);
        tma_2d(smem + CN_WB + warp * CN_SWT, m + 1, w.hs * (CN_W / G8), y, bar);
    }
    return true;
}
// x of the half-stream for every token into shared memory as two planes per token ([T][2][CN_W / 8] float4: floats 0-3,
// then 4-7 of each chunk of eight); with `ss`, the half-stream's sum of R'^2 per token (thread order, then the warps in
// order) into cns.
template <int T>
__device__ void cn_x(const HcFusedArgs& a, int hs, int tid, char* smem, float* part, float* cns, bool ss) {
    constexpr int NV = CN_W / 4 / THREADS + 1;   // float4 per thread and token: tid, then THREADS + tid when in range
    const int c = hs >> 1, lane = tid & 31, warp = tid >> 5;
    float4 r[T][NV], bb[T][NV], g[NV];
    float ip[T];   // every token's gate input first: no load waits inside the token loop
#pragma unroll
    for (int k = 0; k < T; ++k) ip[k] = a.apply ? a.inj_prev[k * HC + c] : 0.0f;
#pragma unroll
    for (int j = 0; j < NV; ++j) {
        const int e = 4 * (tid + THREADS * j);
        const bool in = e < CN_W;
        g[j] = in ? *reinterpret_cast<const float4*>(a.w_norm + hs * CN_W + e) : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int k = 0; k < T; ++k) {
            r[k][j] = in ? *reinterpret_cast<const float4*>(a.R + (size_t) k * D + hs * CN_W + e) : make_float4(0.f, 0.f, 0.f, 0.f);
            bb[k][j] = in && a.apply ? *reinterpret_cast<const float4*>(a.bo_prev + (size_t) k * N + (hs & 1) * CN_W + e)
                                     : make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }
    float4* X = reinterpret_cast<float4*>(smem + CN_XOFF);
#pragma unroll
    for (int k = 0; k < T; ++k) {
        const float gw = a.apply ? 2.0f * sigmoidf_(ip[k] / (float) HC) : 0.0f;
        float sq = 0.0f;
#pragma unroll
        for (int j = 0; j < NV; ++j) {
            const int e4 = tid + THREADS * j;
            if (4 * e4 >= CN_W) continue;
            float4 v = r[k][j];
            if (a.apply) {
                v.x = fmaf(bb[k][j].x, gw, v.x); v.y = fmaf(bb[k][j].y, gw, v.y);
                v.z = fmaf(bb[k][j].z, gw, v.z); v.w = fmaf(bb[k][j].w, gw, v.w);
            }
            sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            X[k * (CN_W / 4) + (e4 & 1) * (CN_W / 8) + (e4 >> 1)] = make_float4(v.x * g[j].x, v.y * g[j].y, v.z * g[j].z, v.w * g[j].w);
        }
        if (ss) {
            const float s = warp_sum(sq);
            if (lane == 0) part[k * WARPS + warp] = s;
        }
    }
    __syncthreads();
    if (ss && tid < T) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[tid * WARPS + w];
        cns[tid * CN_HS + hs] = s;
    }
}
// Warp w: rows 4 w .. 4 w + 3 against every token, lane L taking chunks L + 32 q, q ascending (one scale per chunk's dot
// product); the sums of all (row, token) pairs reduced 16 at a time.
template <int T>
__device__ void cn_dot(const CnRows& w, int tid, const char* smem, uint64_t* bars, unsigned ph, float* cnp) {
    const int warp = tid >> 5, lane = tid & 31;
    if (4 * warp >= w.nrows) return;
    mbar_wait(&bars[warp], ph);
    const int soff = CN_WB + warp * CN_SWT;
    const float4* X = reinterpret_cast<const float4*>(smem + CN_XOFF);
    float acc[4 * T];
#pragma unroll
    for (int i = 0; i < 4 * T; ++i) acc[i] = 0.0f;
#pragma unroll
    for (int q = 0; q < CN_W / 256; ++q) {
        const int j = lane + 32 * q;
        float4 x0[T], x1[T];
#pragma unroll
        for (int k = 0; k < T; ++k) {
            x0[k] = X[k * (CN_W / 4) + j];
            x1[k] = X[k * (CN_W / 4) + CN_W / 8 + j];
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = 4 * warp + r;
            const uint2 wq = *reinterpret_cast<const uint2*>(smem + row * CN_W + j * 8);
            const float4 w0 = i8x4_f(wq.x), w1 = i8x4_f(wq.y);
            const float sc = h2f(reinterpret_cast<const uint16_t*>(smem + soff + r * CN_SB)[j >> 2]);
#pragma unroll
            for (int k = 0; k < T; ++k) acc[r * T + k] = fmaf(sc, dot8ff(w0, w1, x0[k], x1[k]), acc[r * T + k]);
        }
    }
    constexpr int ROUNDS = (4 * T + 15) / 16;
#pragma unroll
    for (int rd = 0; rd < ROUNDS; ++rd) {
        float v[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) v[i] = rd * 16 + i < 4 * T ? acc[rd * 16 + i] : 0.0f;
        const float s = warp_sum16(v, lane);
        const int m = rd * 16 + (lane >> 1);
        if ((lane & 1) == 0 && m < 4 * T) {
            const int r = m / T, k = m - r * T;
            cnp[((size_t) k * CN_HS + w.hs) * CN_ROWS + w.row0 + 4 * warp + r] = s;
        }
    }
}
// An up item's lo (into lo_s, row stride ls) and rs (into rs_s) from the down items' sums; `out`: also the read's lo, rs
// and inject outputs.  lbar non-null: cnp and cns are a copy in shared memory that completes on lbar (phase lph).
template <int T, bool DBG>
__device__ void cn_lo(const HcFusedArgs& a, const float* cnp, const float* cns, int tid, float* lo_s, int ls,
                      float* rs_s, bool out, unsigned long long* dbg, int b, uint64_t* lbar, unsigned lph) {
    constexpr int NI = (T * LR + THREADS - 1) / THREADS;
    const bool sm = lbar != nullptr;
    const auto ld = [sm](const float* q) { return sm ? *q : __ldcg(q); };
    if (sm) mbar_wait(lbar, lph);
    float p[NI][CN_HS];
#pragma unroll
    for (int i = 0; i < NI; ++i) {
        const int idx = tid + THREADS * i, k = idx / LR, row = idx - k * LR;
#pragma unroll
        for (int h = 0; h < CN_HS; ++h)
            p[i][h] = idx < T * LR ? ld(cnp + ((size_t) k * CN_HS + h) * CN_ROWS + row) : 0.0f;
    }
    if (tid < T * HC) {
        const int k = tid / HC, c = tid - k * HC;
        const float s = ld(cns + k * CN_HS + 2 * c) + ld(cns + k * CN_HS + 2 * c + 1);
        rs_s[tid] = rsqrtf(s / (float) N + a.eps);
    }
    __syncthreads();
    stamp<DBG>(dbg, b, 12, tid);
#pragma unroll
    for (int i = 0; i < NI; ++i) {
        const int idx = tid + THREADS * i, k = idx / LR, row = idx - k * LR;
        if (idx >= T * LR) continue;
        float v = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) v = fmaf(rs_s[k * HC + c], p[i][2 * c] + p[i][2 * c + 1], v);
        const float x = v / (float) HC;
        const float l = x / (1.0f + __expf(-x));
        lo_s[k * ls + row] = l;
        if (out) a.lo[idx] = l;
    }
    stamp<DBG>(dbg, b, 14, tid);
    if (out && tid < T * HC) {
        a.rs[tid] = rs_s[tid];
        if (a.inject_out != nullptr && a.q_inject != nullptr) {
            const int k = tid / HC, r = tid - k * HC;
            float v = 0.0f;
#pragma unroll
            for (int c = 0; c < HC; ++c)
                v = fmaf(rs_s[k * HC + c], ld(cnp + ((size_t) k * CN_HS + 2 * c) * CN_ROWS + LR + r) +
                                               ld(cnp + ((size_t) k * CN_HS + 2 * c + 1) * CN_ROWS + LR + r), v);
            a.inject_out[tid] = v;
        }
    }
}


// The up CTA u of nu: columns [c0, c0 + nc) of every stream (nc <= UPCM_SL); its slice of w_up as 8-bit rows in
// shared memory.
struct UpSlice {
    int c0, nc;
};
__device__ __forceinline__ UpSlice up_slice(int u, int nu) {
    const int c0 = u * N / nu, c1 = (u + 1) * N / nu;
    return {c0, c1 - c0};
}
constexpr int SU8 = 16;   // f16 per up row's scales (LR / G8 padded: a row's scales are a 16-byte aligned run)
constexpr int UPG = 4;
constexpr int UPGB8 = UPG * LR + 16;   // 8-bit rows in groups of UPG, 16 bytes after each group
constexpr int LOS = LR + 8;   // the eight tokens' rows of one k-step on distinct banks per half-warp
constexpr int NG8 = LR / G8;  // scale groups per up row
// Shared memory of an up CTA: [the slice (UPCM bytes, or the grouped rows when larger)][lo T x LOS][rp T x 4 x nc: R']
// [px T x 4 x nc: R' * w_norm][g T x 4 x nc][the rows' scales 4 UPCM_SL x SU8 f16, 128-byte aligned]
struct UpSmem {
    float *lo, *rp, *px, *g;
    uint16_t* sc;
};
// The slice as one tensor copy (hc_up_tma): 16-byte k-chunk c of row slot r = 32 stream + col at c * UPCM_CH + 16 r
// (eight consecutive slots are one conflict-free ldmatrix matrix); scales at slot * SU8.
constexpr int UPCM_SL = 32;                    // row slots per stream
constexpr int UPCM_CH = HC * UPCM_SL * 16;     // bytes per k-chunk
constexpr int UPCM = LR / 16 * UPCM_CH;        // the slice's bytes
__device__ __forceinline__ int up_w8_bytes(int nc) { return nc * UPGB8 > UPCM ? nc * UPGB8 : UPCM; }
__device__ __forceinline__ UpSmem up_smem(char* smem, int nc, int T) {
    UpSmem u;
    u.lo = reinterpret_cast<float*>(smem + (size_t) up_w8_bytes(nc));
    u.rp = u.lo + T * LOS;
    u.px = u.rp + T * HC * nc;
    u.g = u.px + T * HC * nc;
    u.sc = reinterpret_cast<uint16_t*>((reinterpret_cast<uintptr_t>(u.g + T * HC * nc) + 127) & ~(uintptr_t) 127);
    return u;
}
// Shared memory bytes an up item of nc columns uses (the slice's area, then UpSmem without the scales).
template <int T>
__device__ __forceinline__ int up_smem_bytes(int nc) {
    return 128 + 4 * UPCM_SL * SU8 * 2 + up_w8_bytes(nc) + (T * LOS + 3 * T * HC * nc) * 4;
}

// Thread 0: the slice's 8-bit rows (32 slots per stream from column c0; slots past the slice hold other columns or
// zeros) and their scales in up_smem's chunk layout, completing on bar.
__device__ void hc_up_tma(UpSlice sl, char* smem, uint16_t* sc, const CUtensorMap* tm, uint64_t* bar) {
    mbar_expect_tx(bar, UPCM + HC * UPCM_SL * SU8 * 2);
    tma_4d(smem, &tm[0], 0, sl.c0, 0, 0, bar);
    tma_3d(sc, &tm[1], 0, sl.c0, 0, bar);
}

// Before the down items are done: R' for the slice and R' * w_norm.
template <int T>
__device__ void hc_up_pre(const HcFusedArgs& a, UpSlice sl, int tid, char* smem) {
    const int nc = sl.nc;
    const UpSmem u = up_smem(smem, nc, T);
    for (int idx = tid; idx < T * HC * nc; idx += THREADS) {
        const int k = idx / (HC * nc), rem = idx - k * HC * nc, c = rem / nc, dd = rem - c * nc;
        const int i = c * N + sl.c0 + dd;
        float rv = a.R[(size_t) k * D + i];
        if (a.apply) rv = fmaf(a.bo_prev[(size_t) k * N + sl.c0 + dd], 2.0f * sigmoidf_(a.inj_prev[k * HC + c] / (float) HC), rv);
        u.rp[idx] = rv;
        u.px[idx] = rv * a.w_norm[i];
    }
}

// R' of the slice to the read's R.
template <int T>
__device__ void hc_up_rout(const HcFusedArgs& a, UpSlice sl, int tid, char* smem) {
    const int nc = sl.nc, c0 = sl.c0;
    const UpSmem u = up_smem(smem, nc, T);
    for (int idx = tid; idx < T * HC * nc; idx += THREADS) {
        const int k = idx / (HC * nc), rem = idx - k * HC * nc, c = rem / nc, dd = rem - c * nc;
        a.R_out[(size_t) k * D + c * N + c0 + dd] = u.rp[idx];
    }
}
// The 8-bit up rows on tensor cores: one ldmatrix.x4 of int8 rows covers a whole scale group (32 values) of the tile's
// 16 rows, thread (g, t4) holding bytes 4 t4..4 t4 + 3 of each 16-value half.  Those become f16 pairs (exact), the
// mma's k order permuted alike in A and B: the pair (4 t4, 4 t4 + 1) as k (2 t4, 2 t4 + 1), (4 t4 + 2, 4 t4 + 3) as
// (2 t4 + 8, 2 t4 + 9), so a token's lo loads as one float4 per half.  lo is split into f16 hi + lo parts; each
// group's sums are scaled by their row's scale into the fp32 total.
__device__ __forceinline__ void mma_f16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void split_f16(float a, float b, uint32_t& hi, uint32_t& lo) {
    const __half2 h = __floats2half2_rn(a, b);
    const float2 hf = __half22float2(h);
    const __half2 l = __floats2half2_rn(a - hf.x, b - hf.y);
    hi = *reinterpret_cast<const uint32_t*>(&h);
    lo = *reinterpret_cast<const uint32_t*>(&l);
}
// The up rows on tensor cores: lo and rs come from the down items' sums (cn_lo; `out`: this item writes the read's lo,
// rs and inject), the slice from up_smem's chunk layout with warp w on slots 16 w .. 16 w + 15.
template <int T, bool DBG>
__device__ void hc_up_main_mma8(const HcFusedArgs& a, UpSlice sl, int tid, char* smem, unsigned long long* dbg, int b,
                                const float* cnp, const float* cns, bool out, uint64_t* lbar, unsigned lph) {
    static_assert(T <= 8, "one mma column per token");
    const int lane = tid & 31, warp = tid >> 5, nc = sl.nc, c0 = sl.c0, g = lane >> 2, t4 = lane & 3;
    const UpSmem u = up_smem(smem, nc, T);
    __shared__ float s_rsv8[MAXT * HC];
    cn_lo<T, DBG>(a, cnp, cns, tid, u.lo, LOS, s_rsv8, out, dbg, b, lbar, lph);
    if (a.apply) hc_up_rout<T>(a, sl, tid, smem);
    __syncthreads();
    float rsr[2][HC];
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int c = 0; c < HC; ++c) rsr[j][c] = 2 * t4 + j < T ? s_rsv8[(2 * t4 + j) * HC + c] : 0.0f;
    stamp<DBG>(dbg, b, 9, tid);
    constexpr int ks = 2 * UPCM_CH;   // bytes per k-step
    const unsigned addr = smem_u32(smem + (lane >> 4) * UPCM_CH + (16 * warp + (lane & 15)) * 16);
    const float* lb = u.lo + (g < T ? g : 0) * LOS + 4 * t4;
    const uint16_t* sc0 = u.sc + (16 * warp + g) * SU8;
    const uint16_t* sc1 = u.sc + (16 * warp + g + 8) * SU8;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int s = 0; s < NG8; ++s) {
        uint32_t q[4];
        asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
            : "=r"(q[0]), "=r"(q[1]), "=r"(q[2]), "=r"(q[3]) : "r"(addr + s * ks));
        float gh[4] = {0.f, 0.f, 0.f, 0.f}, gl[4] = {0.f, 0.f, 0.f, 0.f};   // the group's hi and lo chains
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint32_t af[4] = {i8x2_h2<false>(q[2 * h]), i8x2_h2<false>(q[2 * h + 1]), i8x2_h2<true>(q[2 * h]),
                                    i8x2_h2<true>(q[2 * h + 1])};
            float4 x = *reinterpret_cast<const float4*>(lb + 32 * s + 16 * h);
            if (g >= T) x = make_float4(0.f, 0.f, 0.f, 0.f);
            uint32_t h0, l0, h1, l1;
            split_f16(x.x, x.y, h0, l0);
            split_f16(x.z, x.w, h1, l1);
            mma_f16(gh, af, h0, h1);
            mma_f16(gl, af, l0, l1);
        }
        const float s0 = h2f(sc0[s]), s1 = h2f(sc1[s]);
        acc[0] = fmaf(s0, gh[0] + gl[0], acc[0]);
        acc[1] = fmaf(s0, gh[1] + gl[1], acc[1]);
        acc[2] = fmaf(s1, gh[2] + gl[2], acc[2]);
        acc[3] = fmaf(s1, gh[3] + gl[3], acc[3]);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int k = 2 * t4 + (j & 1);
        const int slot = 16 * warp + g + 8 * (j >> 1), c = slot / UPCM_SL, dd = slot - c * UPCM_SL;
        if (dd >= nc || k >= T) continue;
        const int idx = (k * HC + c) * nc + dd;
        float rsc = rsr[j & 1][0];
#pragma unroll
        for (int cc = 1; cc < HC; ++cc) rsc = c == cc ? rsr[j & 1][cc] : rsc;
        const float x = u.px[idx] * rsc;
        u.g[idx] = x * sigmoidf_(acc[j]);
    }
    __syncthreads();
    stamp<DBG>(dbg, b, 4, tid);
    const bool q8i = up_q8_inline(a, c0, nc);   // a warp's 32 values are one q8_1 block of one token
    for (int i = tid; i < T * nc; i += THREADS) {
        const int k = i / nc, col = i - k * nc;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += u.g[(k * HC + c) * nc + col];
        a.mixed[(size_t) k * N + c0 + col] = s / (float) HC;
        if (q8i) hc_q8_1_store_native(s / (float) HC, static_cast<HcQ81*>(a.xq), k * (N / 32) + c0 / 32, tid & 31);
    }
    stamp<DBG>(dbg, b, 5, tid);
}

// warp_sum16 with the flush-to-zero adds of a fast-math build (mmvf_warp_sum)
__device__ __forceinline__ float warp_sum16_ftz(float (&v)[16], int lane) {
#pragma unroll
    for (int o = 16, n = 8; o >= 2; o >>= 1, n >>= 1) {
        const bool h = (lane & o) != 0;
#pragma unroll
        for (int j = 0; j < n; ++j) {
            const float keep = h ? v[j + n] : v[j], send = h ? v[j] : v[j + n];
            v[j] = add_ftz(keep, __shfl_xor_sync(0xffffffffu, send, o));
        }
    }
    return add_ftz(v[0], __shfl_xor_sync(0xffffffffu, v[0], 1));
}

// Router rows [e0, e1) (at most RT_MAX_ROWS, their weights at wsm): bf16_f32_mmvf_multi_kernel<256, *>'s per-row
// order - thread t's pairs t + 256 j, two ordered FMAs each, the warp's xor tree, then the tree over the 8 warp sums
// and 24 zeros - for every row and token at once; the xor trees run 16 values at a time (warp_sum16_ftz).
template <int T, int RMAX = RT_MAX_ROWS>
__device__ void hc_router_rows(const HcFusedArgs& a, const uint16_t* wsm, int e0, int e1, int tid,
                               float* partials /* [RMAX * T][WARPS] */, unsigned long long* lg64, unsigned tag) {
    const int lane = tid & 31, warp = tid >> 5, ne = e1 - e0;
    constexpr int PAIRS = N / 2 / THREADS;   // 5
    constexpr int TG = T < 4 ? T : 4;        // tokens per pass
    constexpr int NV = RMAX * TG;            // values per lane per pass (28 at most: two rounds of 16)
    constexpr int ROUNDS = (NV + 15) / 16;
    static_assert(NV <= 32, "two rounds at most");
    constexpr int TP = (T + TG - 1) / TG * TG;   // the passes' tokens, padded
    float2 xa[TP][PAIRS];   // every pass's mixed loaded up front: later passes do not wait on loads
#pragma unroll
    for (int q = 0; q < TP; ++q)
#pragma unroll
        for (int j = 0; j < PAIRS; ++j)
            xa[q][j] = q < T ? __ldcg(reinterpret_cast<const float2*>(a.mixed + (size_t) q * N) + tid + THREADS * j)
                             : make_float2(0.f, 0.f);
#pragma unroll
    for (int k0 = 0; k0 < T; k0 += TG) {
        float acc[32];
#pragma unroll
        for (int e = 0; e < RMAX; ++e) {
            uint32_t wr[PAIRS];
#pragma unroll
            for (int j = 0; j < PAIRS; ++j)
                wr[j] = e < ne ? reinterpret_cast<const uint32_t*>(wsm + (size_t) e * N)[tid + THREADS * j] : 0u;
#pragma unroll
            for (int q = 0; q < TG; ++q) {
                float s = 0.0f;
#pragma unroll
                for (int j = 0; j < PAIRS; ++j) {
                    const float w0 = __uint_as_float(wr[j] << 16), w1 = __uint_as_float(wr[j] & 0xffff0000u);
                    s = fma_ftz(w0, xa[k0 + q][j].x, s);
                    s = fma_ftz(w1, xa[k0 + q][j].y, s);
                }
                acc[e * TG + q] = s;
            }
        }
#pragma unroll
        for (int i = NV; i < 32; ++i) acc[i] = 0.0f;
        // the warp trees: value m of round r lands on lanes 2m, 2m + 1
#pragma unroll
        for (int r = 0; r < ROUNDS; ++r) {
            float v[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) v[i] = acc[r * 16 + i];
            const float sum = warp_sum16_ftz(v, lane);
            const int m = r * 16 + (lane >> 1);
            if ((lane & 1) == 0 && m < NV) partials[m * WARPS + warp] = sum;
        }
        __syncthreads();
        // the tree over the warps: warp 0, lane l < 8 holds warp l's sum, the rest zeros
        if (warp == 0) {
#pragma unroll
            for (int r = 0; r < ROUNDS; ++r) {
                float v[16];
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const int m = r * 16 + i;
                    v[i] = lane < WARPS && m < NV ? partials[m * WARPS + lane] : 0.0f;
                }
                const float sum = warp_sum16_ftz(v, lane);
                const int m = r * 16 + (lane >> 1), e = m / TG, q = m - e * TG;
                if ((lane & 1) == 0 && m < NV && e < ne && k0 + q < T) {
                    a.logits[(size_t) (k0 + q) * NE + e0 + e] = sum;
                    st_tagged(lg64 + (size_t) (k0 + q) * NE + e0 + e, sum, tag);
                }
            }
        }
        __syncthreads();
    }
}

// native_router.cu `route` for token k, on one warp, selecting by candidates instead of ten argmax passes.  The ten
// passes pick, in order, the largest p with ties to the lower expert - a total order - so the same ten come out of:
// theta = the 10th largest of the 32 lane maxima (every expert of the top ten is at least theta), then every expert
// at least theta ranked by counting.  `cand` is shared memory for 2 * 512 + 2 * 10 words.
// `tg` non-null: the logits come as tagged data (settled on `tag`) instead of plain floats from `logits`.
template <bool DBG>
__device__ void hc_top10(const float* logits, int32_t* ids, float* weights, int lane, uint32_t* cand,
                         unsigned long long* dbg, int b, const unsigned long long* tg, unsigned tag) {
    float values[16];
    if (tg != nullptr) {
        unsigned long long w[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) w[i] = ld_tagged_raw(tg + lane + i * 32);
        tagged_settle(w, tag, [&](int i) { return tg + lane + i * 32; });
#pragma unroll
        for (int i = 0; i < 16; ++i) values[i] = __uint_as_float((unsigned) w[i]);
    } else {
#pragma unroll
        for (int i = 0; i < 16; ++i) values[i] = __ldcg(logits + lane + i * 32);
    }
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) maximum = max_ftz(maximum, values[i]);
    maximum = warp_max_ftz(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = expf_fast(sub_ftz(values[i], maximum));
        sum = add_ftz(sum, values[i]);
    }
    const float reciprocal = rcp_ftz(warp_sum_ftz(sum));
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = mul_ftz(values[i], reciprocal);
        if (isnan(values[i])) values[i] = -FLT_MAX;
    }
    stamp<DBG>(dbg, b, 12, lane);
    // (v, e) before (v', e') when v > v' or (v == v' and e < e')
    auto before = [](float v, int e, float v2, int e2) { return v > v2 || (v == v2 && e < e2); };
    // the lane's maximum
    float bv = values[0];
    int be = lane;
#pragma unroll
    for (int i = 1; i < 16; ++i)
        if (values[i] > bv) { bv = values[i]; be = lane + 32 * i; }
    // theta: the lane maximum with exactly nine lane maxima before it
    int rank = 0;
#pragma unroll
    for (int m = 0; m < 32; ++m) {
        const float ov = __shfl_sync(0xffffffffu, bv, m);
        const int oe = __shfl_sync(0xffffffffu, be, m);
        rank += before(ov, oe, bv, be) ? 1 : 0;
    }
    const unsigned at9 = __ballot_sync(0xffffffffu, rank == KT - 1);
    const int src = __ffs(at9) - 1;
    const float tv = __shfl_sync(0xffffffffu, bv, src);
    const int te = __shfl_sync(0xffffffffu, be, src);
    // the candidates (not before theta is false), compacted into shared memory as (value bits, expert)
    unsigned mine = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i)
        if (!before(tv, te, values[i], lane + 32 * i)) mine |= 1u << i;
    const int cnt = __popc(mine);
    int off = cnt;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, off, o);
        if (lane >= o) off += y;
    }
    const int total = __shfl_sync(0xffffffffu, off, 31);
    off -= cnt;
#pragma unroll
    for (int i = 0; i < 16; ++i)
        if (mine >> i & 1u) {
            cand[2 * off] = __float_as_uint(values[i]);
            cand[2 * off + 1] = (uint32_t) (lane + 32 * i);
            ++off;
        }
    __syncwarp();
    stamp<DBG>(dbg, b, 13, lane);
    // rank every candidate; ranks 0..9 are the selection in order
    float* top_v = reinterpret_cast<float*>(cand + 2 * NE);
    int* top_e = reinterpret_cast<int*>(cand + 2 * NE + KT);
    for (int c = lane; c < total; c += 32) {
        const float v = __uint_as_float(cand[2 * c]);
        const int e = (int) cand[2 * c + 1];
        int r = 0;
        for (int d = 0; d < total; ++d) r += before(__uint_as_float(cand[2 * d]), (int) cand[2 * d + 1], v, e) ? 1 : 0;
        if (r < KT) { top_v[r] = v; top_e[r] = e; }
    }
    __syncwarp();
    stamp<DBG>(dbg, b, 14, lane);
    // route's sums: lane L adds, in rank order, the selected values of its experts (expert & 31 == L)
    float selected_sum = 0.0f;
#pragma unroll
    for (int r = 0; r < KT; ++r)
        if ((top_e[r] & 31) == lane) selected_sum = add_ftz(selected_sum, top_v[r]);
    const float selected = lane < KT ? top_v[lane] : 0.0f;
    const int sel_e = lane < KT ? top_e[lane] : 0;
    selected_sum = max_ftz(warp_sum_ftz(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = rcp_ftz(selected_sum);
    if (lane < KT) {
        ids[lane] = sel_e;
        weights[lane] = mul_ftz(selected, inverse_selected_sum);
    }
}

// One copy of the top-10's code for the item and its dry run, so the dry run loads the code the item runs.
template <bool DBG>
__device__ __noinline__ void hc_top10_item(const float* logits, int32_t* ids, float* weights, int lane, uint32_t* cand,
                                           unsigned long long* dbg, int b, const unsigned long long* tg,
                                           unsigned tag) {
    hc_top10<DBG>(logits, ids, weights, lane, cand, dbg, b, tg, tag);
}

// ---------------------------------------------------------------- the kernel
// The read is a queue of work items in dependency order: the down items, the up slices, the router rows, the top-10.
// A CTA claims one item at a time (in claim order) and waits only on items before its own, all of them claimed by
// running CTAs, so the read finishes however few of its CTAs are resident.  The down items form R' and x of their
// half-stream themselves and wait on nothing; the up slices wait on the down items, the router rows on the up slices,
// and the top-10 item reads the tagged logits as they land.
struct Items {
    int nu, up, rt, t10, end;
    __device__ Items(int nb, int nu_, bool router) {
        nu = nu_ > 0 ? nu_ : nb - CN_ND;
        up = CN_ND;
        rt = up + nu;
        t10 = rt + (router ? NRT : 0);
        end = t10 + (router ? 1 : 0);
    }
};

constexpr int CAND_WORDS = 2 * NE + 2 * KT;   // hc_top10's shared memory per warp

template <int T, bool DBG>
__global__ void __launch_bounds__(THREADS, 1) hc_fused_kernel(const __grid_constant__ HcLaunch L) {
    extern __shared__ __align__(128) char smem[];
    __shared__ __align__(8) uint64_t bars[2];   // the up slice's copy, the router rows
    __shared__ __align__(8) uint64_t dbars[WARPS];   // the down items' rows, per warp
    __shared__ __align__(8) uint64_t lbar;   // an up item's copy of the down items' sums
    __shared__ float s_ssp[MAXT * WARPS];   // the sums of squares per (token, warp)
    __shared__ int s_item;
    __shared__ unsigned s_tag;
    const HcFusedArgs& a = L.a;
    const int b = blockIdx.x, tid = threadIdx.x, nb = L.nb;
    unsigned long long* dbg = L.dbg;
    HcSync* sy = &g_hc_sync[L.slot];
    const Items it(nb, L.nu, a.w_router != nullptr);
    stamp<DBG>(dbg, b, 0, tid);
    if (tid == 0) {
        mbar_init(&bars[0], 1);
        mbar_init(&bars[1], 1);
        for (int p = 0; p < WARPS; ++p) mbar_init(&dbars[p], 1);
        mbar_init(&lbar, 1);
        s_item = (int) atomicAdd(&sy->q.v, 1u);
        s_tag = *reinterpret_cast<const volatile unsigned*>(&sy->epoch.v) + 1u;   // the read's tag: its counter set's use count
    }
    __syncthreads();
    int item = s_item;
    const unsigned tag = s_tag;
    bool staged = false;   // the first down item's rows were requested before the wait
    bool cn_bars = false;  // the staged item uses dbars
    unsigned qpeek = 0;    // an up item: the queue's claim count seen while it runs
    const CUtensorMap* const tms = reinterpret_cast<const CUtensorMap*>(L.tm);
    if (item < it.up) {    // the item's weights do not depend on the previous kernel
        cn_bars = cn_issue(a, item, tid, smem, dbars, tms);
        staged = true;
    }
    if (L.early) pdl_launch_dependents();
    pdl_wait();
    stamp<DBG>(dbg, b, 1, tid);
    unsigned up_ph = 0, rt_ph = 0, dn_ph = 0, lo_ph = 0;
    while (item < it.end) {
        if (item < it.up) {
            const int d = item;
            if constexpr (DBG) if (tid == 0) dbg[b * 16 + 15] = 1;
            if (!staged) cn_bars = cn_issue(a, d, tid, smem, dbars, tms);
            const CnRows w = cn_rows(a, d);
            if (w.q != nullptr) {
                cn_x<T>(a, w.hs, tid, smem, s_ssp, L.cns, d < CN_HS);
                stamp<DBG>(dbg, b, 2, tid);
                cn_dot<T>(w, tid, smem, dbars, dn_ph, L.cnp);
            }
            if (cn_bars) dn_ph ^= 1u;
            cta_signal(&sy->down.v, tid);
            stamp<DBG>(dbg, b, 3, tid);
        } else if (item < it.rt) {
            const UpSlice sl = up_slice(item - it.up, it.nu);
            if constexpr (DBG) if (tid == 0) dbg[b * 16 + 15] = 2;
            uint16_t* const sc8 = up_smem(smem, sl.nc, T).sc;
            if (tid == 0) hc_up_tma(sl, smem, sc8, &tms[4], &bars[0]);   // ahead of R'
            hc_up_pre<T>(a, sl, tid, smem);
            // the down items' sums by two bulk copies once they are done (after the slice's shared memory)
            const int lst_off = (((up_smem_bytes<T>(sl.nc) + 15) & ~15) + 127) & ~127;
            const unsigned lst_p = (unsigned) (T * CN_HS * CN_ROWS * 4), lst_s = (unsigned) (T * CN_HS * 4);
            const bool lst = lst_off + (int) (lst_p + lst_s) <= SMEM;
            float* const lsm = reinterpret_cast<float*>(smem + lst_off);
            if (tid == 0) qpeek = *reinterpret_cast<const volatile unsigned*>(&sy->q.v);
            if (tid == 0) {
                while (ld_acquire(&sy->down.v) < (unsigned) CN_ND) {}
                stamp<DBG>(dbg, b, 11, 0);
                if (lst) {
                    asm volatile("fence.proxy.async;" ::: "memory");
                    mbar_expect_tx(&lbar, lst_p + lst_s);
                    bulk_g2s(lsm, L.cnp, lst_p, &lbar);
                    bulk_g2s(lsm + T * CN_HS * CN_ROWS, L.cns, lst_s, &lbar);
                }
            }
            mbar_wait(&bars[0], up_ph);
            stamp<DBG>(dbg, b, 10, tid);
            up_ph ^= 1u;
            __syncthreads();
            stamp<DBG>(dbg, b, 2, tid);
            hc_up_main_mma8<T, DBG>(a, sl, tid, smem, dbg, b, lst ? lsm : L.cnp, lst ? lsm + T * CN_HS * CN_ROWS : L.cns,
                                    item == it.up, lst ? &lbar : nullptr, lo_ph);
            if (lst) lo_ph ^= 1u;
            if (a.xq != nullptr && a.w_router == nullptr && !up_q8_inline(a, sl.c0, sl.nc))
                hc_up_q8<T>(a, sl.c0, sl.nc, tid, g_hc_q8cnt[L.slot]);
            if (a.w_router != nullptr) cta_signal(&sy->up.v, tid);   // nothing waits on it without a router
            stamp<DBG>(dbg, b, 3, tid);
        } else if (item >= it.t10) {
            // the top-10 once every router row has stored its logits: warp k takes token k
            if constexpr (DBG) if (tid == 0) dbg[b * 16 + 15] = 3;
            {   // a down CTA may claim the item: its code once on fixed logits while the router runs
                const int warp = tid >> 5;
                if (warp >= 1 && warp <= 3)
                    hc_top10_item<DBG>(L.t10_dummy, reinterpret_cast<int32_t*>(L.t10_scratch + (size_t) b * 2 * KT),
                                       L.t10_scratch + (size_t) b * 2 * KT + KT, tid & 31,
                                       reinterpret_cast<uint32_t*>(smem) + (warp - 1) * CAND_WORDS, nullptr, b, nullptr, 0);
                __syncthreads();
            }
            stamp<DBG>(dbg, b, 8, tid);
            const int k = tid >> 5;
            uint32_t* cand = reinterpret_cast<uint32_t*>(smem) + k * CAND_WORDS;
            if (k < T)
                hc_top10_item<DBG>(a.logits + (size_t) k * NE, a.ids + (size_t) k * KT, a.weights + (size_t) k * KT, tid & 31,
                                   cand, tid < 32 ? dbg : nullptr, b, L.lg64 + (size_t) k * NE, tag);
            __syncthreads();
            stamp<DBG>(dbg, b, 6, tid);
        } else {
            const int r = item - it.rt, e0 = r * NE / NRT, e1 = (r + 1) * NE / NRT;
            char* const rt_p = smem + RT_MAX_ROWS * N * 2;
            if (tid == 0) {   // the rows into shared memory while the up slices run
                const unsigned bytes = (unsigned) (e1 - e0) * N * 2;
                mbar_expect_tx(&bars[1], bytes);
                bulk_g2s(smem, a.w_router + (size_t) e0 * N, bytes, &bars[1]);
            }
            cta_wait(&sy->up.v, (unsigned) it.nu, tid, false);
            mbar_wait(&bars[1], rt_ph);
            rt_ph ^= 1u;
            stamp<DBG>(dbg, b, 4, tid);
            hc_router_rows<T>(a, reinterpret_cast<const uint16_t*>(smem), e0, e1, tid, reinterpret_cast<float*>(rt_p), L.lg64,
                              tag);
            stamp<DBG>(dbg, b, 5, tid);
            if (a.xq != nullptr) {   // the item's share of mixed's q8_1 blocks, while the top-10 runs
                const int nq = T * (N / 32), q0 = r * nq / NRT, q1 = (r + 1) * nq / NRT;
                for (int q = q0 + (tid >> 5); q < q1; q += WARPS)
                    hc_q8_1_store(__ldcg(a.mixed + (size_t) q * 32 + (tid & 31)), static_cast<HcQ81*>(a.xq), q, tid & 31);
            }
        }
        staged = false;
        __syncthreads();
        // every item already claimed (an up item saw as many claims as items): the next claim could only return past the end
        if (tid == 0) s_item = item >= it.up && item < it.rt && qpeek >= (unsigned) it.end ? it.end : (int) atomicAdd(&sy->q.v, 1u);
        __syncthreads();
        item = s_item;
    }

    // the next read's weights into L2, then the counters back to zero once every CTA is through
    if (tid == 0) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            if (a.pf[r] == nullptr || a.pf_bytes[r] < 16) continue;
            const size_t n16 = a.pf_bytes[r] / 16, lo16 = (size_t) b * n16 / nb, hi16 = (size_t) (b + 1) * n16 / nb;
            if (hi16 > lo16) prefetch_l2((const char*) a.pf[r] + lo16 * 16, (hi16 - lo16) * 16);
        }
    }
    stamp<DBG>(dbg, b, 7, tid);
    if (tid == 0) {
        __threadfence();
        if (atomicAdd(&sy->exit.v, 1u) == (unsigned) nb - 1) {
            sy->epoch.v += 1u; sy->q.v = 0; sy->down.v = 0; sy->up.v = 0; sy->exit.v = 0;
            __threadfence();
        }
    }
}

__global__ void hc_l2_prefetch_kernel(const void* p0, size_t n0, const void* p1, size_t n1, const void* p2, size_t n2,
                                      const void* p3, size_t n3) {
    if (threadIdx.x != 0) return;
    const void* p[4] = {p0, p1, p2, p3};
    const size_t n[4] = {n0, n1, n2, n3};
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        if (p[r] == nullptr || n[r] < 16) continue;
        const size_t n16 = n[r] / 16, lo16 = (size_t) blockIdx.x * n16 / gridDim.x, hi16 = (size_t) (blockIdx.x + 1) * n16 / gridDim.x;
        if (hi16 > lo16) prefetch_l2((const char*) p[r] + lo16 * 16, (hi16 - lo16) * 16);
    }
}

#ifndef FZ_DEV_ONLY   // host side: another translation unit (the mid chain) takes only the device code
using KernelFn = void (*)(HcLaunch);
constexpr KernelFn kKernels[2][MAXT] = {
    {hc_fused_kernel<1, false>, hc_fused_kernel<2, false>, hc_fused_kernel<3, false>, hc_fused_kernel<4, false>,
     hc_fused_kernel<5, false>, hc_fused_kernel<6, false>, hc_fused_kernel<7, false>, hc_fused_kernel<8, false>},
    {hc_fused_kernel<1, true>, hc_fused_kernel<2, true>, hc_fused_kernel<3, true>, hc_fused_kernel<4, true>,
     hc_fused_kernel<5, true>, hc_fused_kernel<6, true>, hc_fused_kernel<7, true>, hc_fused_kernel<8, true>}};

unsigned long long* g_dbg = nullptr;
int g_slot = 0;
#endif
}  // namespace
#ifndef FZ_DEV_ONLY   // host side: another translation unit (the mid chain) takes only the device code

bool hc_fused_q8_rows_ok() { return true; }

namespace {

struct DevInfo {
    std::atomic<bool> checked{false};
    bool ok = false;
    int sms = 0;
    float* t10 = nullptr;                 // [NE] fixed logits for the top-10's dry run, then [sms][2 * KT] float scratch
    unsigned long long* lg = nullptr;     // [NSLOT][MAXT][NE] tagged router logits (zero: no tag)
    float* cn = nullptr;                  // [NSLOT][MAXT * CN_HS * CN_ROWS + MAXT * CN_HS]
};
constexpr size_t CN_SLOT = (size_t) CN_PART + CN_SSN;
DevInfo g_dev[16];
std::mutex g_dev_mu;

const DevInfo& dev_info() {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 16) {
        static DevInfo none;
        return none;
    }
    DevInfo& d = g_dev[dev];
    if (!d.checked.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lk(g_dev_mu);
        if (!d.checked.load(std::memory_order_relaxed)) {
            int major = 0, sms = 0, optin = 0;
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
            cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
            d.sms = sms;
            d.ok = major == 12 && sms - CN_ND >= CN_MIN_NU && optin >= SMEM + 1024 &&
                   cudaMalloc((void**) &d.t10, ((size_t) NE + (size_t) sms * 2 * KT) * 4) == cudaSuccess &&
                   cudaMalloc((void**) &d.cn, (size_t) NSLOT * CN_SLOT * 4) == cudaSuccess &&
                   cudaMalloc((void**) &d.lg, (size_t) NSLOT * MAXT * NE * 8) == cudaSuccess &&
                   cudaMemset(d.lg, 0, (size_t) NSLOT * MAXT * NE * 8) == cudaSuccess;
            if (d.ok) {   // distinct logits with a few ties, so the dry run takes the usual path
                std::vector<float> f(NE);
                for (int e = 0; e < NE; ++e) f[e] = (float) (((e * 389) % NE) / 64.0 - 4.0 + (e % 97 == 0 ? 0.0 : 0.001));
                d.ok = cudaMemcpy(d.t10, f.data(), NE * 4, cudaMemcpyHostToDevice) == cudaSuccess;
            }
            for (int t = 0; t < MAXT && d.ok; ++t)
                d.ok = cudaFuncSetAttribute(kKernels[0][t], cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM) == cudaSuccess &&
                       cudaFuncSetAttribute(kKernels[1][t], cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM) == cudaSuccess;
            cudaGetLastError();
            d.checked.store(true, std::memory_order_release);
        }
    }
    return d;
}

// 8-bit copies: q = round(w / s) in [-127, 127], s = f16(absmax / 127) per G8 consecutive values of a row; a row's
// `gpr` scales start every `srow` f16 (a multiple of 8: 16-byte aligned runs).  `err` (optional): the sums of w^2
// and of the error^2.
__global__ void hc8_quant_kernel(const uint16_t* w, int64_t groups, int gpr, int srow, int8_t* q, uint16_t* sc,
                                 double* err) {
    const int64_t gi = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    double e[2] = {0.0, 0.0};
    if (gi < groups) {
        float x[G8], amax = 0.0f;
        for (int i = 0; i < G8; ++i) {
            x[i] = __uint_as_float((uint32_t) w[gi * G8 + i] << 16);
            amax = fmaxf(amax, fabsf(x[i]));
        }
        const __half h = __float2half_rn(amax / 127.0f);
        const float s = __half2float(h);
        sc[gi / gpr * srow + gi % gpr] = __half_as_ushort(h);
        for (int i = 0; i < G8; ++i) {
            const float v = s > 0.0f ? fminf(fmaxf(rintf(x[i] / s), -127.0f), 127.0f) : 0.0f;
            q[gi * G8 + i] = (int8_t) v;
            e[0] += (double) x[i] * x[i];
            e[1] += (double) (v * s - x[i]) * (v * s - x[i]);
        }
    }
    if (err != nullptr)
        for (int k = 0; k < 2; ++k) {
            double v = e[k];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if ((threadIdx.x & 31) == 0) atomicAdd(&err[k], v);
        }
}
struct Hc8Copy {
    int8_t* q = nullptr;   // then the scales, in the same allocation
    uint16_t* s = nullptr;
    size_t bytes = 0;
    int rows = 0, cols = 0, srow = 0;
    bool owned = false;    // keyed by q itself: no bf16 matrix behind it
};
std::mutex g_hc8_mu;
std::unordered_map<const void*, Hc8Copy> g_hc8;
size_t g_hc8_bytes = 0;

// The copies are carved out of large allocations: one cudaMalloc per matrix rounds each up to the allocator's
// granularity, which costs a fraction of a MiB per matrix.  Offsets stay 256-byte aligned, as separate allocations are.
// Copies are never returned, so a chunk is never freed.
cudaError_t hc8_alloc(void** out, size_t bytes) {
    constexpr size_t kChunk = (size_t) 32 << 20, kAlign = 256;
    static char* bases[16] = {};
    static size_t lefts[16] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 16) return cudaMalloc(out, bytes);
    char*& base = bases[dev];
    size_t& left = lefts[dev];
    bytes = (bytes + kAlign - 1) / kAlign * kAlign;
    if (bytes > left) {
        if (bytes > kChunk / 2) return cudaMalloc(out, bytes);
        void* p = nullptr;
        const cudaError_t e = cudaMalloc(&p, kChunk);
        if (e != cudaSuccess) return e;
        base = (char*) p;
        left = kChunk;
    }
    *out = base;
    base += bytes;
    left -= bytes;
    return cudaSuccess;
}

// the copy of the bf16 matrix w (device memory of the current card); false when there is no VRAM for it
bool hc8_make(const uint16_t* w, int rows, int cols, Hc8Copy& c) {
    const int64_t n = (int64_t) rows * cols, groups = n / G8;
    const int gpr = cols / G8, srow = (gpr + 7) & ~7;
    c.bytes = (size_t) n + (size_t) rows * srow * 2;
    c.rows = rows;
    c.cols = cols;
    c.srow = srow;
    if (hc8_alloc((void**) &c.q, c.bytes) != cudaSuccess) {
        cudaGetLastError();
        basalt::log::error("basalt hc8", "no VRAM for an 8-bit copy (%zu bytes)", c.bytes);
        return false;
    }
    c.s = reinterpret_cast<uint16_t*>(c.q + n);
    cudaMemset(c.s, 0, (size_t) rows * srow * 2);
    hc8_quant_kernel<<<(unsigned) ((groups + 255) / 256), 256>>>(w, groups, gpr, srow, c.q, c.s, nullptr);
    cudaDeviceSynchronize();
    return true;
}

__global__ void hc8_dequant_bf16_kernel(const int8_t* q, const uint16_t* s, int cols, int srow, int64_t row0, int64_t n,
                                        uint16_t* out) {
    const int64_t i = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) * 8;   // 8 values, one scale group
    if (i >= n * cols) return;
    const int64_t r = row0 + i / cols;
    const int k = (int) (i % cols);
    const float sc = __half2float(__ushort_as_half(s[r * srow + k / G8]));
    const uint2 w = *reinterpret_cast<const uint2*>(q + r * cols + k);
    uint16_t o[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int v = (int) (int8_t) ((j < 4 ? w.x : w.y) >> (8 * (j & 3)));
        o[j] = __bfloat16_as_ushort(__float2bfloat16_rn((float) v * sc));
    }
    *reinterpret_cast<uint4*>(out + i) = *reinterpret_cast<const uint4*>(o);
}

}  // namespace

bool hc8_enabled() { return true; }

bool hc8_only() { return true; }

void hc8_prepare(const uint16_t* w, int rows, int cols) {
    if (w == nullptr || rows <= 0 || cols <= 0 || cols % G8 != 0) return;
    std::lock_guard<std::mutex> lk(g_hc8_mu);
    if (g_hc8.count(w)) return;
    int dev = 0;
    cudaPointerAttributes pa{};
    cudaGetDevice(&dev);
    if (cudaPointerGetAttributes(&pa, w) != cudaSuccess || pa.type != cudaMemoryTypeDevice || pa.device != dev) {
        cudaGetLastError();
        return;   // another card's matrix: its reads stay bf16
    }
    Hc8Copy c;
    if (!hc8_make(w, rows, cols, c)) return;   // that read stays bf16
    g_hc8[w] = c;
    g_hc8_bytes += c.bytes;
}

const void* hc8_adopt(const uint16_t* w, int rows, int cols) {
    if (w == nullptr || rows <= 0 || cols <= 0 || cols % G8 != 0) return nullptr;
    std::lock_guard<std::mutex> lk(g_hc8_mu);
    Hc8Copy c;
    if (!hc8_make(w, rows, cols, c)) return nullptr;
    c.owned = true;
    g_hc8[c.q] = c;
    g_hc8_bytes += c.bytes;
    return c.q;
}

const void* hc8_upload(const uint16_t* w, const int8_t* q, const uint16_t* s, int rows, int cols) {
    if (q == nullptr || s == nullptr || rows <= 0 || cols <= 0 || cols % G8 != 0) return nullptr;
    std::lock_guard<std::mutex> lk(g_hc8_mu);
    if (w != nullptr && g_hc8.count(w)) return w;
    const int64_t n = (int64_t) rows * cols;
    Hc8Copy c;
    c.rows = rows;
    c.cols = cols;
    c.srow = ((cols / G8) + 7) & ~7;
    c.bytes = (size_t) n + (size_t) rows * c.srow * 2;
    if (hc8_alloc((void**) &c.q, c.bytes) != cudaSuccess) {
        cudaGetLastError();
        basalt::log::error("basalt hc8", "no VRAM for an 8-bit copy (%zu bytes)", c.bytes);
        return nullptr;
    }
    c.s = reinterpret_cast<uint16_t*>(c.q + n);
    if (cudaMemcpy(c.q, q, (size_t) n, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(c.s, s, (size_t) rows * c.srow * 2, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    c.owned = w == nullptr;
    const void* key = w != nullptr ? (const void*) w : (const void*) c.q;
    g_hc8[key] = c;
    g_hc8_bytes += c.bytes;
    return key;
}

bool hc8_lookup(const void* w, const int8_t** q, const uint16_t** s, size_t* bytes) {
    std::lock_guard<std::mutex> lk(g_hc8_mu);
    const auto it = g_hc8.find(w);
    if (it == g_hc8.end()) return false;
    *q = it->second.q;
    *s = it->second.s;
    if (bytes != nullptr) *bytes = it->second.bytes;
    return true;
}

bool hc8_view(const void* w, Hc8View& v) {
    std::lock_guard<std::mutex> lk(g_hc8_mu);
    const auto it = g_hc8.find(w);
    if (it == g_hc8.end()) return false;
    const Hc8Copy& c = it->second;
    v.q = c.q; v.s = c.s; v.rows = c.rows; v.cols = c.cols; v.srow = c.srow; v.bytes = c.bytes; v.owned = c.owned;
    return true;
}

void hc8_dequant_bf16(const Hc8View& v, int64_t row0, int64_t n, uint16_t* out, void* stream) {
    if (n <= 0) return;
    const int64_t units = n * v.cols / 8;
    hc8_dequant_bf16_kernel<<<(unsigned) ((units + 255) / 256), 256, 0, (cudaStream_t) stream>>>(v.q, v.s, v.cols, v.srow,
                                                                                              row0, n, out);
}

size_t hc8_bytes() {
    std::lock_guard<std::mutex> lk(g_hc8_mu);
    return g_hc8_bytes;
}

bool hc_fused_supported() { return dev_info().ok; }
void hc_fused_init() { (void) dev_info(); }

void hc_fused_set_debug(unsigned long long* dbg) { g_dbg = dbg; }

// cuTensorMapEncodeTiled from the driver; the maps of a read's 8-bit matrices, kept per q_down.
using EncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                              const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                              CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
EncodeFn encode_fn() {
    static const EncodeFn f = [] {
        void* p = nullptr;
        cudaDriverEntryPointQueryResult q{};
        if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q) != cudaSuccess ||
            q != cudaDriverEntryPointSuccess) {
            cudaGetLastError();
            return (EncodeFn) nullptr;
        }
        return reinterpret_cast<EncodeFn>(p);
    }();
    return f;
}
// dims[0] elements of the type, innermost; strides (bytes) of dims 1.. ; box per dim.
bool tmap(CUtensorMap* m, const void* base, CUtensorMapDataType ty, int rank, const cuuint64_t* dims,
          const cuuint64_t* strides, const cuuint32_t* box) {
    const EncodeFn f = encode_fn();
    if (f == nullptr || base == nullptr) return false;
    const cuuint32_t es[5] = {1, 1, 1, 1, 1};
    return f(m, ty, (cuuint32_t) rank, const_cast<void*>(base), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
             CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}
// A row-major [rows][cols] matrix (row stride `stride` bytes), box [box_r][box_c].
bool tmap_2d(CUtensorMap* m, const void* base, CUtensorMapDataType ty, uint64_t cols, uint64_t rows, uint64_t stride,
             uint32_t box_c, uint32_t box_r) {
    const cuuint64_t dim[2] = {cols, rows}, str[1] = {stride};
    const cuuint32_t box[2] = {box_c, box_r};
    return tmap(m, base, ty, 2, dim, str, box);
}
struct CnMaps {
    CUtensorMap tm[6];
    bool ok;
};
// The tensor maps of a read's 8-bit matrices; false when one failed.
bool cn_maps(const HcFusedArgs& h, CUtensorMap* tm) {
    static std::mutex mu;
    static std::unordered_map<const void*, CnMaps> cache;
    std::lock_guard<std::mutex> lk(mu);
    auto it = cache.find(h.q_down);
    if (it == cache.end()) {
        CnMaps c{};
        constexpr auto U64 = CU_TENSOR_MAP_DATA_TYPE_UINT64, U16 = CU_TENSOR_MAP_DATA_TYPE_UINT16;
        c.ok = tmap_2d(&c.tm[0], h.q_down, U64, D / 8, LR, D, CN_W / 8, 4) &&
               tmap_2d(&c.tm[1], h.s_down, U16, D / G8, LR, D / G8 * 2, CN_W / G8, 4) &&
               (h.q_inject == nullptr || (tmap_2d(&c.tm[2], h.q_inject, U64, D / 8, HC, D, CN_W / 8, 4) &&
                                          tmap_2d(&c.tm[3], h.s_inject, U16, D / G8, HC, D / G8 * 2, CN_W / G8, 4)));
        // q_up [HC][N][LR] as (16 bytes, column, stream, k-chunk) -> hc_up_tma's chunk layout; s_up [HC][N][SU8]
        const cuuint64_t qd[4] = {2, N, HC, LR / 16}, qs[3] = {LR, (cuuint64_t) N * LR, 16};
        const cuuint32_t qb[4] = {2, UPCM_SL, HC, LR / 16};
        const cuuint64_t sd[3] = {SU8, N, HC}, ss[2] = {SU8 * 2, (cuuint64_t) N * SU8 * 2};
        const cuuint32_t sb[3] = {SU8, UPCM_SL, HC};
        c.ok = c.ok && tmap(&c.tm[4], h.q_up, U64, 4, qd, qs, qb) && tmap(&c.tm[5], h.s_up, U16, 3, sd, ss, sb);
        it = cache.emplace(h.q_down, c).first;
    }
    std::copy(it->second.tm, it->second.tm + 6, tm);
    return it->second.ok;
}


void hc_fused_read(const HcFusedArgs& a, void* stream) {
    const DevInfo& d = dev_info();
    if (!d.ok || a.T < 1 || a.T > MAXT || !a.R || !a.R_out || !a.w_norm || !a.w_down || !a.w_up || !a.lo || !a.rs ||
        !a.mixed || !a.xn || (a.w_inject && !a.inject_out) || (a.apply && (!a.bo_prev || !a.inj_prev)) ||
        (a.w_router && (!a.logits || !a.ids || !a.weights)) || (a.apply && a.inj_prev == a.inject_out)) {
        basalt::log::error("basalt", "hc_fused_read: invalid arguments or unsupported device");
        std::exit(1);
    }
    HcLaunch L{};
    L.a = a;
    L.nb = d.sms;
    L.slot = g_slot;
    g_slot = (g_slot + 1) % NSLOT;
    L.dbg = g_dbg;
    L.t10_dummy = d.t10;
    L.t10_scratch = d.t10 + NE;
    L.early = a.next_early ? 1 : 0;
    // the 8-bit copies of the matrices; the prefetches then take the copies too
    {
        HcFusedArgs& h = L.a;
        bool q8 = false;
        if (h.q_down == nullptr) {
            q8 = hc8_lookup(a.w_down, &h.q_down, &h.s_down) && hc8_lookup(a.w_up, &h.q_up, &h.s_up) &&
                 (a.w_inject == nullptr || hc8_lookup(a.w_inject, &h.q_inject, &h.s_inject));
            for (int r = 0; q8 && r < 4; ++r) {
                const int8_t* q = nullptr;
                const uint16_t* sc = nullptr;
                size_t bytes = 0;
                if (h.pf[r] != nullptr && hc8_lookup(h.pf[r], &q, &sc, &bytes)) { h.pf[r] = q; h.pf_bytes[r] = bytes; }
            }
        } else {
            q8 = h.q_up != nullptr && h.s_down != nullptr && h.s_up != nullptr &&
                 (a.w_inject == nullptr || (h.q_inject != nullptr && h.s_inject != nullptr));
        }
        if (!q8) {
            basalt::log::error("basalt", "hc_fused_read: the matrices need their 8-bit copies (hc8_prepare)");
            std::exit(1);
        }
        if (a.w_inject == nullptr) h.q_inject = nullptr;
    }
    L.cnp = d.cn + (size_t) L.slot * CN_SLOT;
    L.cns = L.cnp + CN_PART;
    L.lg64 = d.lg + (size_t) L.slot * MAXT * NE;
    CUtensorMap maps[6];
    if (!cn_maps(L.a, maps)) {
        basalt::log::error("basalt", "hc_fused_read: the tensor maps of the 8-bit matrices failed");
        std::exit(1);
    }
    std::memcpy(L.tm, maps, sizeof L.tm);
    L.nu = a.xq != nullptr && a.w_router == nullptr ? CN_MIN_NU : 0;   // whole q8_1 blocks
    launch_pdl(kKernels[g_dbg != nullptr ? 1 : 0][a.T - 1], dim3((unsigned) L.nb), dim3(THREADS), (size_t) SMEM,
               (cudaStream_t) stream, L);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "hc_fused_read: %s", cudaGetErrorString(e));
        std::exit(1);
    }
}

namespace {
__global__ void hold_sms_kernel(volatile int* go, int* started) {
    extern __shared__ char hold_smem[];
    if (threadIdx.x == 0) {
        hold_smem[0] = 0;
        atomicAdd(started, 1);
        __threadfence_system();
        while (*go == 0) __nanosleep(1000);
    }
}
}  // namespace
void hc_test_hold_sms(int k, int* go, int* started, void* stream) {
    constexpr int bytes = 64 * 1024;
    cudaFuncSetAttribute(hold_sms_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    hold_sms_kernel<<<k, 32, bytes, (cudaStream_t) stream>>>(go, started);
}

void hc_l2_prefetch(const void* const* ptrs, const size_t* bytes, int n, void* stream) {
    const void* p[4] = {nullptr, nullptr, nullptr, nullptr};
    size_t b[4] = {0, 0, 0, 0};
    for (int i = 0; i < n && i < 4; ++i) { p[i] = ptrs[i]; b[i] = bytes[i]; }
    hc_l2_prefetch_kernel<<<8, 32, 0, (cudaStream_t) stream>>>(p[0], b[0], p[1], b[1], p[2], b[2], p[3], b[3]);
}

#endif
}  // namespace basalt::kernels
