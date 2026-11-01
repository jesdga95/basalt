// src/kernels/cuda/q80_vhead.cu - see include/basalt/kernels/q80_vhead.hpp.
//
// Built with --use_fast_math like native_mmvq.cu, whose Q8_0 x q8_1 dot the exact rows repeat.
#include "basalt/kernels/q80_vhead.hpp"
#include "basalt/kernels/dp4a.hpp"
#include "basalt/kernels/pdl.hpp"
#include "q80p.cuh"
#include "smem_opt_in.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include "basalt/log.hpp"

namespace basalt::kernels {
namespace {

using q80p::kBlocks;
using q80p::kRowBytes;
constexpr int WARP = 32, WPB = 8;
constexpr int VHIST = 1024;        // bins per column of the block-maximum histogram
constexpr float VBIN = 8.0f;       // bins per logit unit, centred on zero
constexpr int kListCap = 1024;     // rows per block of the exact pass

struct Q80Block {
    half d;
    int8_t qs[32];
};
struct Q81Block {
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q80Block) == 34 && sizeof(Q81Block) == 36);

struct VScratch {   // the screen's last block clears the histogram and the counter for the next call
    unsigned hist[8][VHIST];
    float thr[8];
    unsigned done;
};

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) x += __shfl_xor_sync(0xffffffff, x, offset, WARP);
    return x;
}

// The warp's sums of v[0..M) (M a power of two, 2..16), scattered: lane L ends with the sum of v[(L / (32 / M)) % M].
// Each step trades half of the values with the lane `mask` away and keeps the half its bit selects.
template <int M>
__device__ __forceinline__ float sum_scatter(float (&v)[M], int lane) {
#pragma unroll
    for (int h = M / 2; h >= 1; h /= 2) {
        const int mask = 32 * h / M;
        const bool up = (lane & mask) != 0;
#pragma unroll
        for (int i = 0; i < h; ++i) {
            const float send = up ? v[i] : v[i + h];
            v[i] = (up ? v[i + h] : v[i]) + __shfl_xor_sync(0xffffffff, send, mask);
        }
    }
    float r = v[0];
#pragma unroll
    for (int mask = 16 / M; mask >= 1; mask /= 2) r += __shfl_xor_sync(0xffffffff, r, mask);
    return r;
}

// ---- the layout

// One block of 80 threads per row: the row is read whole, then written back in the other layout.
__global__ void __launch_bounds__(kBlocks) q80p_pack_kernel(uint8_t* __restrict__ w) {
    uint8_t* row = w + (size_t) blockIdx.x * kRowBytes;
    const int b = threadIdx.x;
    const Q80Block* src = reinterpret_cast<const Q80Block*>(row) + b;
    const uint16_t d = *reinterpret_cast<const uint16_t*>(&src->d);
    uint8_t u[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) u[i] = (uint8_t) src->qs[i] ^ 0x80u;
    __syncthreads();
    uint32_t nib[4], top[2], low[2];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        uint32_t x = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            x |= ((uint32_t) ((u[8 * p + j] >> 2) & 15u) | ((uint32_t) ((u[8 * p + 4 + j] >> 2) & 15u) << 4)) << (8 * j);
        nib[p] = x;
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        uint32_t t = 0, l = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t v = u[16 * h + 4 * k + j];
                t |= (v >> 6) << (8 * j + 2 * k);
                l |= (v & 3u) << (8 * j + 2 * k);
            }
        top[h] = t;
        low[h] = l;
    }
    reinterpret_cast<uint16_t*>(row + q80p::kScales)[b] = d;
    reinterpret_cast<uint4*>(row + q80p::kNib)[b] = make_uint4(nib[0], nib[1], nib[2], nib[3]);
    reinterpret_cast<uint2*>(row + q80p::kTop)[b] = make_uint2(top[0], top[1]);
    reinterpret_cast<uint2*>(row + q80p::kLow)[b] = make_uint2(low[0], low[1]);
}

__global__ void __launch_bounds__(kBlocks) q80p_unpack_kernel(uint8_t* __restrict__ w) {
    uint8_t* row = w + (size_t) blockIdx.x * kRowBytes;
    const int b = threadIdx.x;
    int8_t q[32];
    float d;
    q80p::block_codes(row, b, q, d);
    const uint16_t dh = reinterpret_cast<const uint16_t*>(row + q80p::kScales)[b];
    __syncthreads();
    Q80Block* dst = reinterpret_cast<Q80Block*>(row) + b;
    *reinterpret_cast<uint16_t*>(&dst->d) = dh;
#pragma unroll
    for (int i = 0; i < 32; ++i) dst->qs[i] = q[i];
}

// ---- exact rows, native_mmvq's arithmetic

// native_mmvq's Q8_0 kernel runs a row on four warps: thread tid takes blocks kbx = tid / 4, += 32, words kqs = 2 (tid
// % 4), kqs + 1.  Lane L stands for its threads L, L + 32, L + 64, L + 96 (virtual threads 0..3: three blocks for 0 and
// 1, two for 2 and 3); their sums are added in the cross-warp order before the same xor tree.
constexpr int kNch = 10;
__device__ __forceinline__ int kb_of(int lane, int c) {
    constexpr int off[kNch] = {0, 32, 64, 8, 40, 72, 16, 48, 24, 56};
    return lane / 4 + off[c];
}
__device__ __forceinline__ void row_load(const uint8_t* __restrict__ row, int lane, q80p::Quarter (&r)[kNch]) {
#pragma unroll
    for (int c = 0; c < kNch; ++c) r[c] = q80p::load_quarter<false>(row, kb_of(lane, c), lane % 4);
}
// xe: per column j, q8_1 block b and word pair p, {q[2p], q[2p + 1], the block's scale as float, 0}
template <int NC>
__device__ __forceinline__ void row_dot(const q80p::Quarter (&r)[kNch], const int4* __restrict__ xe, int lane,
                                        float (&out)[NC]) {
    constexpr int vt[kNch] = {0, 0, 0, 1, 1, 1, 2, 2, 3, 3};
    const int p = lane % 4;
    float s[4][NC];
#pragma unroll
    for (int v = 0; v < 4; ++v)
#pragma unroll
        for (int j = 0; j < NC; ++j) s[v][j] = 0.0f;
#pragma unroll
    for (int c = 0; c < kNch; ++c) {
        int v0, v1;
        q80p::codes(r[c].nib, r[c].top, r[c].low, p, v0, v1);
        const int kb = kb_of(lane, c);
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            const int4 e = xe[((size_t) j * kBlocks + kb) * 4 + p];
            int sumi = 0;
            sumi = DP4A(v0, e.x, sumi);
            sumi = DP4A(v1, e.y, sumi);
            s[vt[c]][j] += r[c].d * __int_as_float(e.z) * float(sumi);
        }
    }
#pragma unroll
    for (int j = 0; j < NC; ++j) {
        float t = s[0][j];
#pragma unroll
        for (int v = 1; v < 4; ++v) t += s[v][j];
        out[j] = warp_sum(t);
    }
}

__device__ __forceinline__ void load_xe(const Q81Block* __restrict__ x, int ncols, int4* __restrict__ xe) {
    for (int i = threadIdx.x; i < ncols * kBlocks * 4; i += blockDim.x) {
        const Q81Block* blk = x + i / 4;
        const int* q8 = reinterpret_cast<const int*>(blk->qs) + 2 * (i % 4);
        const unsigned dsb = __ldg(reinterpret_cast<const unsigned*>(&blk->ds));
        xe[i] = make_int4(__ldg(q8), __ldg(q8 + 1), __float_as_int(__low2float(*reinterpret_cast<const half2*>(&dsb))), 0);
    }
}

// The exact pass.  Every row and column (k null or 0): one resident wave walks the rows, a warp per row, the next row
// loaded while this one is computed.  Otherwise a slice of rows per block: the (row, column) pairs whose upper bound in
// y reaches the column's threshold, a warp per listed row.
template <int NC>
__global__ void __launch_bounds__(WPB * WARP) q80p_rows_kernel(const uint8_t* __restrict__ w,
                                                               const Q81Block* __restrict__ x, float* __restrict__ y,
                                                               int n_out, int per_block, const int* __restrict__ k,
                                                               const VScratch* __restrict__ vs) {
    extern __shared__ __align__(16) int4 xe[];
    __shared__ uint32_t list[kListCap];
    __shared__ int cnt;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int r0 = blockIdx.x * per_block, r1 = min(n_out, r0 + per_block);
    const int keep = k != nullptr ? __ldg(k) : 0;   // set before the launch
    q80p::Quarter cur[kNch];
    if (keep == 0) {
        const int row0 = blockIdx.x * WPB + warp, stride = gridDim.x * WPB;
        if (row0 < n_out) row_load(w + (size_t) row0 * kRowBytes, lane, cur);   // weights: before the wait
        pdl_wait();
        load_xe(x, NC, xe);
        __syncthreads();
        for (int row = row0; row < n_out; row += stride) {
            q80p::Quarter nxt[kNch];
            if (row + stride < n_out) row_load(w + (size_t) (row + stride) * kRowBytes, lane, nxt);
            float v[NC];
            row_dot<NC>(cur, xe, lane, v);
#pragma unroll
            for (int j = 0; j < NC; ++j)
                if (lane == j) y[(size_t) j * n_out + row] = v[j];
#pragma unroll
            for (int c = 0; c < kNch; ++c) cur[c] = nxt[c];
        }
        return;
    }
    pdl_wait();
    load_xe(x, NC, xe);
    if (tid == 0) cnt = 0;
    __syncthreads();
    {
        float thr[NC];
#pragma unroll
        for (int j = 0; j < NC; ++j) thr[j] = __ldcg(&vs->thr[j]);
        for (int i = tid; i < r1 - r0; i += blockDim.x) {
            uint32_t m = 0u;
#pragma unroll
            for (int j = 0; j < NC; ++j)
                if (!(__ldcg(y + (size_t) j * n_out + r0 + i) < thr[j])) m |= 1u << j;   // NaN: scored
            if (m != 0u) list[atomicAdd(&cnt, 1)] = (uint32_t) i | m << 16;
        }
    }
    __syncthreads();
    const int n = cnt;
    if (warp < n) row_load(w + (size_t) (r0 + (int) (list[warp] & 0xffffu)) * kRowBytes, lane, cur);
    for (int i = warp; i < n; i += WPB) {
        q80p::Quarter nxt[kNch];
        if (i + WPB < n) row_load(w + (size_t) (r0 + (int) (list[i + WPB] & 0xffffu)) * kRowBytes, lane, nxt);
        float v[NC];
        row_dot<NC>(cur, xe, lane, v);
        const uint32_t e = list[i];
        const int r = r0 + (int) (e & 0xffffu);
#pragma unroll
        for (int j = 0; j < NC; ++j)
            if (lane == j && ((e >> (16 + j)) & 1u)) y[(size_t) j * n_out + r] = v[j];
#pragma unroll
        for (int c = 0; c < kNch; ++c) cur[c] = nxt[c];
    }
}

// ---- the screen

// Lane L takes blocks L, L + 32 and (L < 16) L + 64 of a row: their scales and bits 2..7.
struct ScreenRow {
    uint4 nib[3];
    uint2 top[3];
    float d[3];
};
__device__ __forceinline__ void screen_load(const uint8_t* __restrict__ row, int lane, ScreenRow& r) {
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int b = lane + 32 * i;
        if (b < kBlocks) {
            r.nib[i] = __ldcs(reinterpret_cast<const uint4*>(row + q80p::kNib) + b);
            r.top[i] = __ldcs(reinterpret_cast<const uint2*>(row + q80p::kTop) + b);
            r.d[i] = __half2float(__ldcs(reinterpret_cast<const half*>(row + q80p::kScales) + b));
        }
    }
}

// Every row's upper bound into y; each block adds its per-column maximum lower bound to the histogram, and the last
// block turns the histogram into the columns' thresholds: the lower edge of the bin holding the K-th largest block
// maximum, at or below the K-th largest lower bound and so at or below the K-th largest exact logit.  Blocks hold
// disjoint rows, so their maxima are K distinct rows' bounds.
template <int NC>
__global__ void __launch_bounds__(WPB * WARP) q80p_screen_kernel(const uint8_t* __restrict__ w,
                                                                 const Q81Block* __restrict__ x, float* __restrict__ y,
                                                                 int n_out, const int* __restrict__ k,
                                                                 VScratch* __restrict__ vs) {
    __shared__ __align__(16) int4 sy[NC][2][kBlocks];   // a column's codes per block, in two halves
    __shared__ __align__(16) float4 sc[NC][kBlocks];    // {4 d, -31.625 sum q, 1.5 d sum |q|, 0}
    __shared__ float wmax[WPB][NC];
    __shared__ int bad[NC];
    __shared__ bool last;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int keep = __ldg(k);   // set before the launch
    const int row0 = blockIdx.x * WPB + warp, stride = gridDim.x * WPB;
    ScreenRow cur;
    if (keep != 0 && row0 < n_out) screen_load(w + (size_t) row0 * kRowBytes, lane, cur);   // weights: before the wait
    pdl_wait();
    if (keep == 0) return;   // the exact pass takes every row
    if (tid < NC) bad[tid] = 0;
    __syncthreads();
    for (int i = tid; i < NC * kBlocks; i += blockDim.x) {
        const int j = i / kBlocks, b = i % kBlocks;
        const Q81Block* blk = x + (size_t) j * kBlocks + b;
        const int* q = reinterpret_cast<const int*>(blk->qs);
        int v[8], sum = 0;
        unsigned sabs = 0u;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            v[t] = __ldg(q + t);
            sum = DP4A(v[t], 0x01010101, sum);
            sabs = __dp4a((unsigned) __vabs4((unsigned) v[t]), 0x01010101u, sabs);   // |-128| counts 128
        }
        const unsigned dsb = __ldg(reinterpret_cast<const unsigned*>(&blk->ds));
        const float d = __low2float(*reinterpret_cast<const half2*>(&dsb));
        sy[j][0][b] = make_int4(v[0], v[1], v[2], v[3]);
        sy[j][1][b] = make_int4(v[4], v[5], v[6], v[7]);
        sc[j][b] = make_float4(4.0f * d, -31.625f * (float) sum, 1.5f * d * (float) sabs, 0.0f);
        if ((__float_as_uint(d) & 0x7f800000u) == 0x7f800000u) bad[j] = 1;   // not finite (a bit test: fast math)
    }
    __syncthreads();
    float lmax[NC];
#pragma unroll
    for (int j = 0; j < NC; ++j) lmax[j] = __int_as_float(0xff800000);
    for (int row = row0; row < n_out; row += stride) {
        const int nrow = row + stride;
        ScreenRow nxt;
        if (nrow < n_out) screen_load(w + (size_t) nrow * kRowBytes, lane, nxt);
        constexpr int P = NC <= 1 ? 1 : NC <= 2 ? 2 : NC <= 4 ? 4 : 8;   // the columns, padded to a power of two
        float v[2 * P];   // c of each column, then its half-width
#pragma unroll
        for (int j = 0; j < 2 * P; ++j) v[j] = 0.0f;
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            const int b = lane + 32 * i;
            if (b < kBlocks) {
                const uint32_t nb[4] = {cur.nib[i].x, cur.nib[i].y, cur.nib[i].z, cur.nib[i].w};
                const uint32_t tp[2] = {cur.top[i].x, cur.top[i].y};
                uint32_t h[8];
#pragma unroll
                for (int p = 0; p < 4; ++p) q80p::hi6(nb[p], tp[p >> 1], p, h[2 * p], h[2 * p + 1]);
#pragma unroll
                for (int j = 0; j < NC; ++j) {
                    const int4 a = sy[j][0][b], e = sy[j][1][b];
                    int u = 0;
                    u = DP4A((int) h[0], a.x, u);
                    u = DP4A((int) h[1], a.y, u);
                    u = DP4A((int) h[2], a.z, u);
                    u = DP4A((int) h[3], a.w, u);
                    u = DP4A((int) h[4], e.x, u);
                    u = DP4A((int) h[5], e.y, u);
                    u = DP4A((int) h[6], e.z, u);
                    u = DP4A((int) h[7], e.w, u);
                    const float4 f = sc[j][b];
                    v[j] = fmaf(cur.d[i] * f.x, (float) u + f.y, v[j]);   // (float) u + f.y is exact
                    v[P + j] = fmaf(cur.d[i], f.z, v[P + j]);
                }
            }
        }
        // lane L holds c of column (L / (16 / P)) % P when L < 16, and lane L + 16 its half-width
        const float cs = sum_scatter<2 * P>(v, lane);
        const float ws = __shfl_xor_sync(0xffffffff, cs, 16);
        const int col = (lane / (16 / P)) % P;
        // the rounding of both sides is far inside the 1/16 margin (see the header)
        const float m = fmaf(ws, 1.0625f, 1e-6f);
        if (lane < 16 && lane % (16 / P) == 0 && col < NC) y[(size_t) col * n_out + row] = cs + m;
#pragma unroll
        for (int j = 0; j < NC; ++j)
            if (col == j && lane < 16) lmax[j] = fmaxf(lmax[j], cs - m);
        cur = nxt;
    }
#pragma unroll
    for (int j = 0; j < NC; ++j) {   // the lanes holding column j share its maximum
        float m = lmax[j];
#pragma unroll
        for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
        if (lane == 0) wmax[warp][j] = m;
    }
    __syncthreads();
    if (tid < NC && !bad[tid]) {
        float m = wmax[0][tid];
#pragma unroll
        for (int v = 1; v < WPB; ++v) m = fmaxf(m, wmax[v][tid]);
        if (m > __int_as_float(0xff800000)) {
            m = fminf(fmaxf(m, -1000.0f), 1000.0f);
            const int bin = min(VHIST - 1, max(0, (int) floorf(m * VBIN) + VHIST / 2));
            atomicAdd(&vs->hist[tid][bin], 1u);
        }
    }
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        last = atomicAdd(&vs->done, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    const unsigned want = (unsigned) min(max(keep, 1), kQ80pKeepMax);
    for (int c = warp; c < NC; c += WPB) {
        unsigned v[VHIST / 32], tot[VHIST / 32];   // lane l of row i: bin 32 i + l
#pragma unroll
        for (int i = 0; i < VHIST / 32; ++i) v[i] = __ldcg(&vs->hist[c][i * 32 + lane]);
#pragma unroll
        for (int i = 0; i < VHIST / 32; ++i) tot[i] = __reduce_add_sync(0xffffffffu, v[i]);
        unsigned run = 0, row_v = 0;
        int rw = -1;
#pragma unroll
        for (int i = VHIST / 32 - 1; i >= 0; --i)
            if (rw < 0) {
                if (run + tot[i] >= want) { rw = i; row_v = v[i]; }
                else run += tot[i];
            }
        int bin = -1;
        if (rw >= 0) {
            unsigned suf = row_v;   // this bin's count and those of the bins above it in the row
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const unsigned u = __shfl_down_sync(0xffffffffu, suf, o);
                if (lane + o < 32) suf += u;
            }
            const unsigned m = __ballot_sync(0xffffffffu, run + suf >= want);
            bin = rw * 32 + 31 - __clz((int) m);
        }
        if (lane == 0)   // fewer than K maxima, the lowest bin, or a column without finite scales: every row
            vs->thr[c] = bin > 0 && !bad[c] ? (float) (bin - VHIST / 2) / VBIN : __int_as_float(0xff800000);
        for (int i = lane; i < VHIST; i += 32) vs->hist[c][i] = 0u;
    }
    if (tid == 0) vs->done = 0u;
}

// ---- launches

// One resident wave of each kernel: the screen's for eight columns at every column count (its blocks, so its thresholds,
// do not depend on the window's width), the exact pass's for its own column count.
template <int NC>
void grids(int& screen, int& rows) {
    static std::atomic<bool> set[64];
    static int g[64][2];
    const int dev = opt_in_device("q80p");
    if (!set[dev].load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lk(opt_in_mutex());
        if (!set[dev].load(std::memory_order_relaxed)) {
            int sms = 0, a = 0, b = 0;
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&a, q80p_screen_kernel<8>, WPB * WARP, 0);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, q80p_rows_kernel<NC>, WPB * WARP,
                                                          (size_t) NC * kBlocks * 4 * sizeof(int4));
            g[dev][0] = (sms > 0 ? sms : 1) * (a > 0 ? a : 1);
            g[dev][1] = (sms > 0 ? sms : 1) * (b > 0 ? b : 1);
            set[dev].store(true, std::memory_order_release);
        }
    }
    screen = g[dev][0];
    rows = g[dev][1];
}

template <int NC>
void launch(const void* w, const void* x, float* y, int n_out, const int* k, VScratch* vs, cudaStream_t s) {
    int screen = 0, grid = 0;
    grids<NC>(screen, grid);
    if (k != nullptr)
        launch_pdl(q80p_screen_kernel<NC>, dim3((unsigned) screen), dim3(WPB * WARP), 0, s, (const uint8_t*) w,
                   (const Q81Block*) x, y, n_out, k, vs);
    if ((long long) grid * kListCap < n_out) grid = (n_out + kListCap - 1) / kListCap;
    const int per_block = (n_out + grid - 1) / grid;
    grid = (n_out + per_block - 1) / per_block;
    launch_pdl(q80p_rows_kernel<NC>, dim3((unsigned) grid), dim3(WPB * WARP), (size_t) NC * kBlocks * 4 * sizeof(int4), s,
               (const uint8_t*) w, (const Q81Block*) x, y, n_out, per_block, k, (const VScratch*) vs);
}

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "%s: %s", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

void dispatch(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, const int* k, void* scratch,
              void* stream, const char* who) {
    if (!q80p_supported(n_in, ncols) || n_out <= 0 || !w || !x || !y || (k != nullptr && scratch == nullptr)) {
        basalt::log::error("basalt", "%s: unsupported shape (n_in %d, n_out %d, ncols %d)", who, n_in, n_out, ncols);
        std::exit(1);
    }
    const auto s = (cudaStream_t) stream;
    VScratch* vs = (VScratch*) scratch;
    switch (ncols) {
        case 1: launch<1>(w, x, y, n_out, k, vs, s); break;
        case 2: launch<2>(w, x, y, n_out, k, vs, s); break;
        case 3: launch<3>(w, x, y, n_out, k, vs, s); break;
        case 4: launch<4>(w, x, y, n_out, k, vs, s); break;
        case 5: launch<5>(w, x, y, n_out, k, vs, s); break;
        case 6: launch<6>(w, x, y, n_out, k, vs, s); break;
        case 7: launch<7>(w, x, y, n_out, k, vs, s); break;
        default: launch<8>(w, x, y, n_out, k, vs, s); break;
    }
    check(who);
}

}  // namespace

bool q80p_supported(int n_in, int ncols) { return n_in == kBlocks * 32 && ncols >= 1 && ncols <= 8; }

void q80p_pack(void* rows, int64_t n, void* stream) {
    if (n <= 0) return;
    q80p_pack_kernel<<<(unsigned) n, kBlocks, 0, (cudaStream_t) stream>>>((uint8_t*) rows);
    check("q80p_pack");
}

void q80p_unpack(void* rows, int64_t n, void* stream) {
    if (n <= 0) return;
    q80p_unpack_kernel<<<(unsigned) n, kBlocks, 0, (cudaStream_t) stream>>>((uint8_t*) rows);
    check("q80p_unpack");
}

void q80p_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    dispatch(weights, x_q8_1, y, n_in, n_out, ncols, nullptr, nullptr, stream, "q80p_mmvq");
}

size_t q80p_vhead_scratch_bytes() { return sizeof(VScratch); }

void q80p_vhead(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, const int* k,
                void* scratch, void* stream) {
    if (k == nullptr) {
        basalt::log::error("basalt", "q80p_vhead: no list length");
        std::exit(1);
    }
    dispatch(weights, x_q8_1, y, n_in, n_out, ncols, k, scratch, stream, "q80p_vhead");
}

}  // namespace basalt::kernels
