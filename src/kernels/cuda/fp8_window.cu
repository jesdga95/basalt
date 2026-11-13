// src/kernels/cuda/fp8_window.cu - see include/basalt/kernels/fp8_window.hpp.
//
// A block is two warps over 32 weight rows (16 each) and one slice of K.  The slice's chunks of 256 values stream in order
// through shared memory (cp.async, four chunks in flight per block) and each warp runs its 16 rows over them on mma
// m16n8k16 with the window's rows as the 8 columns.  Thread (g, c) holds weight rows g and g + 8 and activation row g;
// in each 64-value block it takes values 16c .. 16c + 15, step s's k-slots {2c, 2c+1} values 16c + 4s, +1 and
// {2c+8, 2c+9} values 16c + 4s + 2, +3, and multiplies the same values of its activation row, so the sums are plain
// dot products.  With several slices each block writes its sums to the workspace, and the last block of a row tile to
// finish (a ticket) adds them in slice order and applies the epilogue.
#include "basalt/kernels/fp8_window.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "basalt/log.hpp"

namespace basalt::kernels::fp8w {
namespace {

constexpr int kChunk = 256;                         // values (bytes) per row per chunk
constexpr int kWarps = 2, kRowsPerBlock = 16 * kWarps, kThreads = 32 * kWarps;
constexpr int kPieces = kChunk / 16;
constexpr int kStage = kRowsPerBlock * kChunk;
constexpr int kStages = 4;
constexpr int kTargetBlocks = 170 * 4;

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { basalt::log::error("basalt", "%s: %s", what, cudaGetErrorString(e)); std::exit(1); }
}

struct Plan {
    int64_t tiles = 0;
    int ks = 1, cps = 0, nchunks = 0;
};
Plan plan_for(int64_t rows, int64_t K) {
    Plan p;
    p.nchunks = (int) (K / kChunk);
    p.tiles = (rows + kRowsPerBlock - 1) / kRowsPerBlock;
    const int ks = (int) std::min<int64_t>(p.nchunks, std::max<int64_t>(1, (kTargetBlocks + p.tiles - 1) / p.tiles));
    p.cps = (p.nchunks + ks - 1) / ks;
    p.ks = (p.nchunks + p.cps - 1) / p.cps;
    return p;
}

__device__ __forceinline__ void cp16(void* smem, const void* gmem) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void mma(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                    unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                 "{%0, %1, %2, %3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// two E4M3 bytes (the low 16 bits of x, the low byte the lower half) as fp16x2
__device__ __forceinline__ unsigned e4m3x2_f16(unsigned x) {
    unsigned h;
    const unsigned short v = (unsigned short) x;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"(v));
    return h;
}
__device__ __forceinline__ uint4 ld_nc16(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

// one 64-value block of weight rows r0 (row g) and r1 (row g + 8), activation words xa (values 16c .. 16c + 7) and xb
// (16c + 8 .. 16c + 15)
__device__ __forceinline__ void block64(float (&acc)[4], const uint8_t* r0, const uint8_t* r1, int c, uint4 xa, uint4 xb) {
    const uint4 w0 = *(const uint4*) (r0 + 16 * c), w1 = *(const uint4*) (r1 + 16 * c);
    const unsigned u0[4] = {w0.x, w0.y, w0.z, w0.w}, u1[4] = {w1.x, w1.y, w1.z, w1.w};
    const unsigned xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
    for (int s = 0; s < 4; ++s)
        mma(acc, e4m3x2_f16(u0[s]), e4m3x2_f16(u1[s]), e4m3x2_f16(u0[s] >> 16), e4m3x2_f16(u1[s] >> 16), xw[2 * s],
            xw[2 * s + 1]);
}

struct Args {
    const uint8_t* w;
    int64_t rows;
    int K, T, cps, nchunks, epi;
    const __half* act;
    const float* fac;
    const float* scale;
    float* y;
    int64_t ld_y;
    float* partial;
    int32_t* tickets;
};

__global__ void __launch_bounds__(kThreads) gemm_kernel(const Args a) {
    __shared__ __align__(16) uint8_t smem[kStages * kStage];
    __shared__ int last;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    const int64_t row0 = (int64_t) blockIdx.x * kRowsPerBlock;
    const int slice = blockIdx.y, ks = gridDim.y;
    const int cb = slice * a.cps, n = min(a.cps, a.nchunks - cb);
    const uint8_t* wbase = a.w + (int64_t) cb * kChunk;

    auto issue = [&](int i) {
        uint8_t* st = smem + (i % kStages) * kStage;
        const uint8_t* src = wbase + (int64_t) i * kChunk;
        for (int p = threadIdx.x; p < kRowsPerBlock * kPieces; p += kThreads) {
            const int r = p / kPieces, q = p - r * kPieces;
            const int64_t row = min(row0 + r, a.rows - 1);   // rows past the last read the last (not written)
            cp16(st + r * kChunk + q * 16, src + row * a.K + q * 16);
        }
    };

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const bool live = g < a.T;   // this thread's activation row
    const __half* xrow = a.act + (int64_t) g * a.K + (int64_t) cb * kChunk;
#pragma unroll
    for (int i = 0; i < kStages - 1; ++i) {
        if (i < n) issue(i);
        cp_commit();
    }
    for (int i = 0; i < n; ++i) {
        if (i + kStages - 1 < n) issue(i + kStages - 1);
        cp_commit();
        const __half* xc = xrow + i * kChunk;
        uint4 xa[4], xb[4];
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            xa[b] = live ? ld_nc16(xc + b * 64 + 16 * c) : make_uint4(0, 0, 0, 0);
            xb[b] = live ? ld_nc16(xc + b * 64 + 16 * c + 8) : make_uint4(0, 0, 0, 0);
        }
        cp_wait<kStages - 1>();
        __syncthreads();
        const uint8_t* r0 = smem + (i % kStages) * kStage + (warp * 16 + g) * kChunk;
        const uint8_t* r1 = r0 + 8 * kChunk;
#pragma unroll
        for (int b = 0; b < 4; ++b) block64(acc, r0 + b * 64, r1 + b * 64, c, xa[b], xb[b]);
        __syncthreads();
    }
    cp_wait<0>();

    const int64_t ra = row0 + warp * 16 + g, rbw = ra + 8;
    const int t0 = 2 * c, t1 = 2 * c + 1;
    if (ks > 1) {
        float* part = a.partial + (int64_t) slice * kRows * a.rows;
        if (t0 < a.T) {
            if (ra < a.rows) part[t0 * a.rows + ra] = acc[0];
            if (rbw < a.rows) part[t0 * a.rows + rbw] = acc[2];
        }
        if (t1 < a.T) {
            if (ra < a.rows) part[t1 * a.rows + ra] = acc[1];
            if (rbw < a.rows) part[t1 * a.rows + rbw] = acc[3];
        }
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) last = atomicAdd(a.tickets + blockIdx.x, 1) == ks - 1;
        __syncthreads();
        if (!last) return;
        __threadfence();
        const int64_t rr[2] = {ra, rbw};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int t = (j & 1) ? t1 : t0;
            const int64_t r = rr[j >> 1];
            if (t >= a.T || r >= a.rows) continue;
            float s = 0.0f;
            for (int k = 0; k < ks; ++k) s += __ldcg(a.partial + ((int64_t) k * kRows + t) * a.rows + r);
            acc[j] = s;
        }
        if (threadIdx.x == 0) a.tickets[blockIdx.x] = 0;
    }
    const float sc = a.scale ? *a.scale : 1.0f;
    const int64_t rr[2] = {ra, rbw};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int t = (j & 1) ? t1 : t0;
        const int64_t r = rr[j >> 1];
        if (t >= a.T || r >= a.rows) continue;
        const float v = acc[j] * a.fac[t] * sc;
        float* dst = a.y + t * a.ld_y + r;
        if (a.epi == (int) Epilogue::Add) *dst += v;
        else *dst = v;
    }
}

// ---- activations: a row's amax, then fp16 of x times the power of two that puts amax in [2^14, 2^15)

__global__ void pack_kernel(const float* __restrict__ x, int64_t ld_x, int64_t K, __half* __restrict__ out,
                            float* __restrict__ fac) {
    __shared__ float red[32];
    __shared__ float pw;
    const float* xr = x + (int64_t) blockIdx.x * ld_x;
    float m = 0.0f;
    for (int64_t c = threadIdx.x; c < K; c += blockDim.x) m = fmaxf(m, fabsf(xr[c]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int i = 0; i < (int) (blockDim.x >> 5); ++i) t = fmaxf(t, red[i]);
        int e = 0;
        if (t > 0.0f && isfinite(t)) {
            int ex;
            frexpf(t, &ex);   // t = m 2^ex, m in [0.5, 1)
            e = 15 - ex;
        }
        fac[blockIdx.x] = ldexpf(1.0f, -e);
        pw = ldexpf(1.0f, e);
    }
    __syncthreads();
    __half* o = out + (int64_t) blockIdx.x * K;
    for (int64_t c = threadIdx.x; c < K; c += blockDim.x) o[c] = __float2half_rn(xr[c] * pw);
}

}  // namespace

std::size_t act_bytes(int64_t K) { return (size_t) kRows * K * sizeof(__half) + kRows * sizeof(float); }

void pack(const float* x, int64_t ld_x, int64_t K, int T, void* act, void* stream) {
    if (T <= 0) return;
    pack_kernel<<<T, 512, 0, (cudaStream_t) stream>>>(x, ld_x, K, (__half*) act,
                                                      (float*) ((uint8_t*) act + (size_t) kRows * K * sizeof(__half)));
    check("fp8 window pack");
}

void workspace_need(int64_t rows, int64_t K, std::size_t& partial_floats, std::size_t& tickets) {
    const Plan p = plan_for(rows, K);
    partial_floats = p.ks > 1 ? (size_t) p.ks * kRows * rows : 0;
    tickets = (size_t) p.tiles;
}

void gemm(const uint8_t* w, int64_t rows, int64_t K, const void* act, const float* scale, float* y, int64_t ld_y,
          Epilogue epi, int T, const Workspace& ws, void* stream) {
    if (T <= 0 || rows <= 0) return;
    if (K % kChunk != 0 || ((uintptr_t) w & 15) != 0) {
        basalt::log::error("basalt", "fp8 window: a [%lld x %lld] matrix", (long long) rows, (long long) K);
        std::exit(1);
    }
    const Plan p = plan_for(rows, K);
    if (p.ks > 1 && ((size_t) p.ks * kRows * rows > ws.partial_floats || (size_t) p.tiles > ws.tickets_n)) {
        basalt::log::error("basalt", "fp8 window: workspace too small for [%lld x %lld]", (long long) rows, (long long) K);
        std::exit(1);
    }
    Args a;
    a.w = w;
    a.rows = rows;
    a.K = (int) K;
    a.T = T;
    a.cps = p.cps;
    a.nchunks = p.nchunks;
    a.epi = (int) epi;
    a.act = (const __half*) act;
    a.fac = (const float*) ((const uint8_t*) act + (size_t) kRows * K * sizeof(__half));
    a.scale = scale;
    a.y = y;
    a.ld_y = ld_y;
    a.partial = ws.partial;
    a.tickets = ws.tickets;
    gemm_kernel<<<dim3((unsigned) p.tiles, (unsigned) p.ks), kThreads, 0, (cudaStream_t) stream>>>(a);
    check("fp8 window");
}

}  // namespace basalt::kernels::fp8w
