// src/kernels/cuda/head_proxy.cuh - the 2-bit copy of a draft head's rows (n_in 2560) that the Q5_K and Q6_K heads
// share: its layout, the fit and packing of one 32-weight group, the approximate scoring pass, and the pick among the rows the
// head's own kernel rescored exactly.  Each including file gets its own copy of these kernels.
#pragma once

#include "basalt/kernels/dp4a.hpp"
#include "basalt/kernels/pdl.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace basalt::kernels::px {
namespace {

constexpr int kNw4 = 2560 / 64;   // uint4 of weight words per row
constexpr int kPer = 4;           // candidates kept per tile of 32 rows
constexpr int kMaxCand = 64;      // rows rescored exactly
constexpr float kDelta = 5.0f;    // a row is rescored when its approximate score is within this of the best

__host__ __device__ inline int64_t tiles(int64_t n_cap) { return (n_cap + 31) / 32; }

inline size_t bytes(int64_t n_cap) { return (size_t) tiles(n_cap) * kNw4 * 32 * (16 + 4); }
inline size_t scratch_bytes(int64_t n_cap) { return (size_t) tiles(n_cap) * (kPer + 1) * sizeof(float2); }
inline uint16_t* scales(void* proxy, int64_t n_cap) {
    return (uint16_t*) ((char*) proxy + (size_t) tiles(n_cap) * kNw4 * 32 * 16);
}

struct X81 {
    half2 ds;
    int8_t qs[32];
};

// The level (0..3) of weight w at quantization step sh.
__device__ __forceinline__ int level(float w, float sh) { return min(3, max(0, (int) floorf(w / sh + 2.0f))); }

// One 32-weight group's scale: four levels with the scale (of a few candidates) that leaves the least squared error.
// `sh` gets the step the levels are taken at; the result is the stored scale.  UNSHRINK: the stored scale is then
// multiplied by sqrt(|w|^2 / <w, w_hat>); the least-squares copy scores the rows near the best low, which inflates
// the draft probability, and half the full correction leaves it close to the exact one.
template <bool UNSHRINK = false>
__device__ __forceinline__ __half fit_group(const float* w, float& sh_out) {
    float amax = 1e-9f;
    for (int l = 0; l < 32; ++l) amax = fmaxf(amax, fabsf(w[l]));
    float best_s = amax / 1.5f, best_e = INFINITY;
    for (int t = 0; t < 6; ++t) {
        const float s = amax / 1.5f * (1.0f - 0.1f * (float) t);
        const float sh = __half2float(__float2half(s));
        float e = 0.0f;
        for (int l = 0; l < 32; ++l) {
            const float r = ((float) level(w[l], sh) - 1.5f) * sh - w[l];
            e += r * r;
        }
        if (e < best_e) { best_e = e; best_s = s; }
    }
    __half hs = __float2half(best_s);
    const float sh = __half2float(hs);
    sh_out = sh;
    if constexpr (UNSHRINK) {
        float ww = 0.0f, wr = 0.0f;
        for (int l = 0; l < 32; ++l) {
            const float r = ((float) level(w[l], sh) - 1.5f) * sh;
            ww += w[l] * w[l];
            wr += w[l] * r;
        }
        if (wr > 0.0f) hs = __float2half(sh * sqrtf(ww / wr));
    }
    return hs;
}

// Group g (32 weights, dequantized) of row `row` into the copy.
template <bool UNSHRINK = false>
__device__ __forceinline__ void pack_group(const float* w, int64_t row, int g, uint4* __restrict__ wq,
                                           uint16_t* __restrict__ sc) {
    float sh;
    const __half hs = fit_group<UNSHRINK>(w, sh);
    const int tile = (int) (row >> 5), lane = (int) (row & 31), w4 = g / 2, h = g & 1;
    uint32_t words[2] = {0u, 0u};
    for (int c = 0; c < 2; ++c)
        for (int ii = 0; ii < 4; ++ii)
            for (int bb = 0; bb < 4; ++bb) words[c] |= (uint32_t) level(w[c * 16 + 4 * ii + bb], sh) << (8 * bb + 2 * ii);
    uint32_t* wd = reinterpret_cast<uint32_t*>(wq) + (((size_t) tile * kNw4 + w4) * 32 + lane) * 4 + 2 * h;
    wd[0] = words[0];
    wd[1] = words[1];
    sc[(((size_t) tile * kNw4 + w4) * 32 + lane) * 2 + h] = __half_as_ushort(hs);
}

// One tile per block, four warps splitting the row length; lane = row.  Writes the tile's best kPer approximate
// scores and the tile's softmax partial (max, sum of exp(score - max)).
__global__ void __launch_bounds__(128) score_kernel(const uint4* __restrict__ wq, const uint32_t* __restrict__ sc,
                                                   const X81* __restrict__ x, int n_rows,
                                                   const int32_t* __restrict__ map, float2* __restrict__ cand,
                                                   float2* __restrict__ part) {
    __shared__ __align__(16) int xi[640];
    __shared__ float xs[80], xdd[80];
    __shared__ float red[4][32];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, tile = blockIdx.x;
    const size_t base = ((size_t) tile * kNw4 + warp * 10) * 32 + lane;
    uint4 w[10];
    uint32_t s[10];
#pragma unroll
    for (int j = 0; j < 10; ++j) {   // constant data: read before the wait
        w[j] = __ldcs(wq + base + (size_t) j * 32);
        s[j] = __ldcs(sc + base + (size_t) j * 32);
    }
    const int row = tile * 32 + lane;
    const int32_t id = (warp == 0 && row < n_rows) ? __ldg(map + row) : 0;
    pdl_wait();
    for (int i = tid; i < 640; i += 128) xi[i] = reinterpret_cast<const int*>(x[i / 8].qs)[i % 8];
    if (tid < 80) {
        const int* q = reinterpret_cast<const int*>(x[tid].qs);
        int sum = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) sum = DP4A(0x01010101, q[i], sum);
        const float d8 = __low2float(x[tid].ds);
        xdd[tid] = d8;
        xs[tid] = d8 * (float) sum;
    }
    __syncthreads();
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < 10; ++j) {
        const int w4 = warp * 10 + j;
        const uint32_t ww[4] = {w[j].x, w[j].y, w[j].z, w[j].w};
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int g = 2 * w4 + h;
            int P = 0;
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const uint32_t v = ww[2 * h + c];
                const int4 x0 = *reinterpret_cast<const int4*>(&xi[g * 8 + c * 4]);
                P = DP4A((int) (v & 0x03030303u), x0.x, P);
                P = DP4A((int) ((v >> 2) & 0x03030303u), x0.y, P);
                P = DP4A((int) ((v >> 4) & 0x03030303u), x0.z, P);
                P = DP4A((int) ((v >> 6) & 0x03030303u), x0.w, P);
            }
            const float sg = __half2float(__ushort_as_half((unsigned short) (h ? (s[j] >> 16) : (s[j] & 0xffff))));
            acc += sg * (xdd[g] * (float) P - 1.5f * xs[g]);
        }
    }
    red[warp][lane] = acc;
    __syncthreads();
    if (warp != 0) return;
    float v = red[0][lane] + red[1][lane] + red[2][lane] + red[3][lane];
    if (row >= n_rows || id < 0) v = __int_as_float(0xff800000);
    float m = v;
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float e = v > __int_as_float(0xff800000) ? __expf(v - m) : 0.0f;
#pragma unroll
    for (int o = 16; o; o >>= 1) e += __shfl_xor_sync(0xffffffffu, e, o);
    if (lane == 0) part[tile] = make_float2(m, e);
    for (int k = 0; k < kPer; ++k) {
        float bv = v;
        int bi = lane;
#pragma unroll
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) cand[tile * kPer + k] = make_float2(bv, __int_as_float(tile * 32 + bi));
        if (lane == bi) v = __int_as_float(0xff800000);
    }
}

// The final kernel's shared state (one block of 512 threads).
struct Final {
    float wmax[16];
    int list[kMaxCand];
    float lp[kMaxCand], ex[kMaxCand];
    int cnt;
    int hist[256];
    unsigned long long want;
    int need;
};

// A candidate's selection key: its approximate score, then the lower row on a tie.  Keys of distinct rows differ.
__device__ __forceinline__ unsigned long long cand_key(float2 c) {
    const unsigned u = __float_as_uint(c.x);
    const unsigned o = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return ((unsigned long long) o << 32) | (unsigned) (0xffffffffu - (unsigned) __float_as_int(c.y));
}

// The kMaxCand-th largest key among the nc candidates within kDelta of m (more than kMaxCand of them): a radix
// select over the key's bytes, most significant first.
__device__ __forceinline__ unsigned long long kth_key(Final& f, int nc, const float2* __restrict__ cand, float m) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const float ninf = __int_as_float(0xff800000);
    if (tid == 0) { f.want = 0ull; f.need = kMaxCand; }
    unsigned long long mask = 0ull;
    for (int shift = 56; shift >= 0; shift -= 8) {
        if (tid < 256) f.hist[tid] = 0;
        __syncthreads();
        const unsigned long long want = f.want;
        for (int i = tid; i < nc; i += 512) {
            const float2 c = __ldcg(&cand[i]);
            if (!(c.x > ninf && c.x >= m - kDelta)) continue;
            const unsigned long long k = cand_key(c);
            if ((k & mask) == want) atomicAdd(&f.hist[(int) ((k >> shift) & 255u)], 1);
        }
        __syncthreads();
        if (warp == 0) {   // lane l holds bins 255 - 8l .. 248 - 8l; the digit is the bin where the count from the top reaches need
            int b[8], s = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) { b[j] = f.hist[255 - 8 * lane - j]; s += b[j]; }
            int incl = s;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int v = __shfl_up_sync(0xffffffffu, incl, o);
                if (lane >= o) incl += v;
            }
            const int need = f.need, excl = incl - s;
            if (excl < need && incl >= need) {
                int acc = excl, j = 0;
                while (acc + b[j] < need) acc += b[j++];
                f.want = want | ((unsigned long long) (255 - 8 * lane - j) << shift);
                f.need = need - acc;
            }
        }
        mask |= 255ull << shift;
        __syncthreads();
    }
    return f.want;
}

// After pdl_wait, with f.cnt zeroed by thread 0: the best approximate score, then the rows within kDelta of it into
// f.list in ascending row order; when more than kMaxCand qualify, the kMaxCand best by approximate score (the lower row
// on a tie).  The list depends on the scores alone, never on the order threads arrive in.  Returns the number listed;
// 0 when every row is masked (the pick is then written here).
__device__ __forceinline__ int collect(Final& f, int n_tiles, const int32_t* __restrict__ map,
                                       const float2* __restrict__ cand, int* __restrict__ out_id,
                                       float* __restrict__ out_prob) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int nc = n_tiles * kPer;
    float m = __int_as_float(0xff800000);
    for (int i = tid; i < nc; i += 512) m = fmaxf(m, __ldcg(&cand[i].x));
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) f.wmax[warp] = m;
    __syncthreads();
    m = f.wmax[0];
    for (int i = 1; i < 16; ++i) m = fmaxf(m, f.wmax[i]);
    const float ninf = __int_as_float(0xff800000);
    if (m == ninf) {   // every row is masked
        if (tid == 0) { *out_id = map[0] < 0 ? 0 : map[0]; *out_prob = 1.0f; }
        return 0;
    }
    for (int i = tid; i < nc; i += 512) {
        const float2 c = __ldcg(&cand[i]);
        if (c.x > ninf && c.x >= m - kDelta) {
            const int p = atomicAdd(&f.cnt, 1);
            if (p < kMaxCand) { f.list[p] = __float_as_int(c.y); f.lp[p] = c.x; }
        }
    }
    __syncthreads();
    const int total = f.cnt;
    if (total > kMaxCand) {
        const unsigned long long kth = kth_key(f, nc, cand, m);
        if (tid == 0) f.cnt = 0;
        __syncthreads();
        for (int i = tid; i < nc; i += 512) {
            const float2 c = __ldcg(&cand[i]);
            if (c.x > ninf && c.x >= m - kDelta && cand_key(c) >= kth) {
                const int p = atomicAdd(&f.cnt, 1);
                if (p < kMaxCand) { f.list[p] = __float_as_int(c.y); f.lp[p] = c.x; }
            }
        }
        __syncthreads();
    }
    const int n = total < kMaxCand ? total : kMaxCand;
    int row = 0, rank = 0;
    float lp = 0.0f;
    if (tid < n) {
        row = f.list[tid];
        lp = f.lp[tid];
        for (int j = 0; j < n; ++j) rank += f.list[j] < row;
    }
    __syncthreads();
    if (tid < n) { f.list[rank] = row; f.lp[rank] = lp; }
    __syncthreads();
    return n;
}

// Warp 0, after f.ex holds the n listed rows' exact scores: the largest exact score (the lowest row on a tie) is the
// pick; its probability is the tiles' softmax sums with the rescored rows' approximate terms swapped for exact ones.
__device__ __forceinline__ void pick(const Final& f, int n, int n_tiles, const int32_t* __restrict__ map,
                                     const float2* __restrict__ part, int* __restrict__ out_id,
                                     float* __restrict__ out_prob) {
    const int lane = threadIdx.x & 31;
    const float ninf = __int_as_float(0xff800000);
    float bv = ninf;
    int bi = 0x7fffffff;
    for (int c = lane; c < n; c += 32)
        if (f.ex[c] > bv || (f.ex[c] == bv && f.list[c] < bi)) { bv = f.ex[c]; bi = f.list[c]; }
#pragma unroll
    for (int o = 16; o; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    float sum = 0.0f;
    for (int t = lane; t < n_tiles; t += 32) {
        const float2 p = __ldcg(&part[t]);
        if (p.y > 0.0f) sum += p.y * __expf(p.x - bv);
    }
    for (int c = lane; c < n; c += 32) sum += __expf(f.ex[c] - bv) - __expf(f.lp[c] - bv);
#pragma unroll
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (lane == 0) {
        float one = 0.0f;   // the pick's own term is exactly 1
        for (int c = 0; c < n; ++c) if (f.list[c] == bi) one = 1.0f;
        sum = fmaxf(sum, one);
        *out_id = map[bi < 0x7fffffff ? bi : 0];
        *out_prob = 1.0f / sum;
    }
}

}  // namespace
}  // namespace basalt::kernels::px
