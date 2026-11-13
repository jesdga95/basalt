// src/kernels/cuda/gqa_attn.cu - see include/basalt/kernels/gqa_attn.hpp.
#include "basalt/kernels/gqa_attn.hpp"
#include "basalt/kernels/pdl.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include "basalt/log.hpp"

namespace basalt::kernels {
namespace {

constexpr int GMAX = 8;               // query heads per KV head
constexpr int CHUNK = 64;             // cells per decode chunk
constexpr int kDecThreads = 256;      // a decode block's threads
constexpr int kDecSplits = 64;        // a decode row's blocks per KV head (its chunks dealt round robin)
static_assert(CHUNK == 8 * (kDecThreads / 32), "decode attention: eight cells a warp");
constexpr int HD = 256;              // the prompt kernel's head width
constexpr float kScaleLog2 = 0.0625f * 1.4426950408889634f;   // 1/sqrt(256) * log2(e)
constexpr float kNeg = -1e30f;

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { basalt::log::error("basalt", "%s: %s", what, cudaGetErrorString(e)); std::exit(1); }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// ---------------------------------------------------------------- decode
// Row t of a window sees positions [lo_t, hi_t): hi_t = pos0 + t + 1 (causal) or `hi`, lo_t = pos0 + t - window + 1 (a
// sliding window, else 0); position p sits in cache cell p % cap.  Cells are split in CHUNK-position chunks from chunk
// `base`; block b of a row takes chunks b, b + gx, ... (gx from the capacity alone) and folds them in order, so a row's
// sums depend on its own range only, never on the window's rows or where the first cell is read from.
struct Range {
    int64_t pos0 = 0, hi = 0, window = 0, cap = 0;
    int causal = 1, base = 0;
    const int32_t* pos_dev = nullptr;   // pos0 read from the device (a captured window)
};
__device__ __forceinline__ int64_t row_pos0(const Range& r) { return r.pos_dev ? (int64_t) *r.pos_dev : r.pos0; }
__device__ __forceinline__ int64_t row_hi(const Range& r, int t) { return r.causal ? row_pos0(r) + t + 1 : r.hi; }
__device__ __forceinline__ int64_t row_lo(const Range& r, int t) {
    return r.window > 0 ? max((int64_t) 0, row_pos0(r) + t - r.window + 1) : 0;
}

// Block (split b, KV head, row t): for each of its chunks, four lanes per cell for the scores (its key row against the
// G query heads held in shared memory), one warp per head for the chunk's max and sum (a fixed butterfly), then p . V with
// thread (quad, group) over dims 4 quad .. 4 quad + 3 and cells i = group mod PG, the groups added in order; the chunk is
// folded into the block's running {max, sum, acc[HD]} per head.  Writes that partial (empty when the block had no cells).
template <int HD>
// (launched through launch_pdl: q and the cache come from the kernels before, read after pdl_wait, so not __restrict__)
__global__ void __launch_bounds__(kDecThreads) decode_chunk_kernel(const float* q, const __half* kc, const __half* vc, int n_head,
                                                                   int n_kv, Range r, int n_chunks, float scale, float* part,
                                                                   KvInt8 q8) {
    pdl_launch_dependents();
    pdl_wait();
    constexpr int NQ = HD / 4, PG = kDecThreads / NQ;   // dim quads, cell groups
    static_assert(PG >= 1 && kDecThreads % NQ == 0, "decode attention: thread split");
    __shared__ float s[CHUNK][GMAX + 1];
    __shared__ __align__(16) float qs[GMAX][HD];
    __shared__ float mx[GMAX], sm[GMAX];
    __shared__ float red[PG > 1 ? (PG - 1) * GMAX * HD : 1];   // the groups' partial sums [PG - 1][GMAX][HD]
    const int kvh = blockIdx.y, t = blockIdx.z, gx = gridDim.x;
    const int G = n_head / n_kv;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int quad = threadIdx.x % NQ, grp = threadIdx.x / NQ;
    const int64_t lo = row_lo(r, t), hi = row_hi(r, t);
    for (int h = 0; h < G; ++h)
        for (int d = threadIdx.x; d < HD; d += kDecThreads) qs[h][d] = q[((int64_t) t * n_head + kvh * G + h) * HD + d] * scale;
    __syncthreads();
    float run_m[GMAX], run_l[GMAX], run_a[GMAX][4];
#pragma unroll
    for (int h = 0; h < GMAX; ++h) {
        run_m[h] = kNeg;
        run_l[h] = 0.0f;
        run_a[h][0] = run_a[h][1] = run_a[h][2] = run_a[h][3] = 0.0f;
    }
    for (int chunk = blockIdx.x; chunk < n_chunks; chunk += gx) {
        const int64_t c0 = (int64_t) (r.base + chunk) * CHUNK;
        if (c0 >= hi) break;   // past the row's cells
        const int64_t first = max(c0, lo), end = min(c0 + CHUNK, hi);
        const int n = (int) (end - first);
        if (n <= 0) continue;   // below a sliding window
        // the chunk's cells are consecutive positions: one modulo for the chunk, a wrap at most once
        const int64_t cell0 = first % r.cap;
        auto cell_of = [&](int i) -> int64_t { const int64_t c = cell0 + i; return c >= r.cap ? c - r.cap : c; };
        // four lanes a cell, eight cells a warp: lane `sub` of cell c reads dims (4 j + sub) 8 .. + 7 of its key row
        // (16-byte pieces, the cell's row in one pass) against each head's query in shared memory, then the four
        // lanes' sums meet in a fixed butterfly
        {
            constexpr int DPL = HD / 4;   // dims per lane
            const int sub = lane & 3, c = warp * 8 + (lane >> 2);
            float d[GMAX];
#pragma unroll
            for (int h = 0; h < GMAX; ++h) d[h] = 0.0f;
            if (c < n) {
                const int64_t row = cell_of(c) * n_kv + kvh;
#pragma unroll
                for (int j = 0; j < DPL / 8; ++j) {
                    const int d0 = (4 * j + sub) * 8;
                    float kf[8];
                    if (q8.k) {
                        const float sc = __half2float(__ushort_as_half(q8.ks[row * (HD / 64) + d0 / 64]));
                        int8_t b8[8];
                        *reinterpret_cast<int2*>(b8) = *reinterpret_cast<const int2*>(q8.k + row * HD + d0);
#pragma unroll
                        for (int e = 0; e < 8; ++e) kf[e] = (float) b8[e] * sc;
                    } else {
                        const uint4 u = *reinterpret_cast<const uint4*>(kc + row * HD + d0);
                        const unsigned w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&w[e]));
                            kf[2 * e] = f.x;
                            kf[2 * e + 1] = f.y;
                        }
                    }
#pragma unroll
                    for (int h = 0; h < GMAX; ++h)
                        if (h < G) {
                            const float4 qa = *reinterpret_cast<const float4*>(&qs[h][d0]);
                            const float4 qb = *reinterpret_cast<const float4*>(&qs[h][d0 + 4]);
                            d[h] = fmaf(qa.x, kf[0], d[h]);
                            d[h] = fmaf(qa.y, kf[1], d[h]);
                            d[h] = fmaf(qa.z, kf[2], d[h]);
                            d[h] = fmaf(qa.w, kf[3], d[h]);
                            d[h] = fmaf(qb.x, kf[4], d[h]);
                            d[h] = fmaf(qb.y, kf[5], d[h]);
                            d[h] = fmaf(qb.z, kf[6], d[h]);
                            d[h] = fmaf(qb.w, kf[7], d[h]);
                        }
                }
            }
#pragma unroll
            for (int h = 0; h < GMAX; ++h) {
                d[h] += __shfl_xor_sync(0xffffffffu, d[h], 1);
                d[h] += __shfl_xor_sync(0xffffffffu, d[h], 2);
            }
            if (c < n && sub == 0)
#pragma unroll
                for (int h = 0; h < GMAX; ++h)
                    if (h < G) s[c][h] = d[h];
        }
        __syncthreads();
        if (warp < G) {
            const int h = warp;
            float m = kNeg;
            for (int i = lane; i < n; i += 32) m = fmaxf(m, s[i][h]);
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
            float l = 0.0f;
            for (int i = lane; i < n; i += 32) {
                const float p = exp2f(s[i][h] - m);
                s[i][h] = p;
                l += p;
            }
            l = warp_sum(l);
            if (lane == 0) { mx[h] = m; sm[h] = l; }
        }
        __syncthreads();
        float acc[GMAX][4] = {};
#pragma unroll 4
        for (int i = grp; i < n; i += PG) {
            const int64_t row = cell_of(i) * n_kv + kvh;
            float v[4];
            if (q8.v) {
                const char4 c = *reinterpret_cast<const char4*>(q8.v + row * HD + 4 * quad);
                const float sc = __half2float(__ushort_as_half(q8.vs[row * (HD / 64) + (4 * quad) / 64]));
                v[0] = (float) c.x * sc; v[1] = (float) c.y * sc; v[2] = (float) c.z * sc; v[3] = (float) c.w * sc;
            } else {
                const uint2 u = *reinterpret_cast<const uint2*>(vc + row * HD + 4 * quad);
                const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
                const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
                v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
            }
#pragma unroll
            for (int h = 0; h < GMAX; ++h)
                if (h < G) {
                    const float p = s[i][h];
#pragma unroll
                    for (int e = 0; e < 4; ++e) acc[h][e] = fmaf(p, v[e], acc[h][e]);
                }
        }
        if (grp > 0)
#pragma unroll
            for (int h = 0; h < GMAX; ++h)
                if (h < G)
#pragma unroll
                    for (int e = 0; e < 4; ++e) red[((grp - 1) * GMAX + h) * HD + 4 * quad + e] = acc[h][e];
        __syncthreads();
        if (grp == 0)
#pragma unroll
            for (int h = 0; h < GMAX; ++h)
                if (h < G) {
#pragma unroll
                    for (int gq = 1; gq < PG; ++gq)
#pragma unroll
                        for (int e = 0; e < 4; ++e) acc[h][e] += red[((gq - 1) * GMAX + h) * HD + 4 * quad + e];
                    // fold the chunk into the block's running partial
                    const float M = fmaxf(run_m[h], mx[h]), wa = exp2f(run_m[h] - M), wb = exp2f(mx[h] - M);
                    run_l[h] = fmaf(run_l[h], wa, sm[h] * wb);
#pragma unroll
                    for (int e = 0; e < 4; ++e) run_a[h][e] = fmaf(run_a[h][e], wa, acc[h][e] * wb);
                    run_m[h] = M;
                }
        __syncthreads();   // the shared rows are free for the block's next chunk
    }
    if (grp == 0) {
        float* out = part + (((int64_t) t * n_head + (int64_t) kvh * G) * gx + blockIdx.x) * (HD + 2);
#pragma unroll
        for (int h = 0; h < GMAX; ++h)
            if (h < G) {
                float* o = out + (int64_t) h * gx * (HD + 2);
                if (quad == 0) { o[0] = run_m[h]; o[1] = run_l[h]; }
#pragma unroll
                for (int e = 0; e < 4; ++e) o[2 + 4 * quad + e] = run_a[h][e];
            }
    }
}

// Block (query head, row t), one thread per dim: the row's splits merged in order.
template <int HD>
__global__ void __launch_bounds__(HD) decode_merge_kernel(const float* part, int n_head, int gx, float* out) {
    pdl_launch_dependents();
    pdl_wait();
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const float* p = part + ((int64_t) t * n_head + h) * gx * (HD + 2);
    float m = kNeg;
    for (int b = 0; b < gx; ++b) m = fmaxf(m, p[(int64_t) b * (HD + 2)]);
    float l = 0.0f, a = 0.0f;
    for (int b = 0; b < gx; ++b) {
        const float* pb = p + (int64_t) b * (HD + 2);
        const float w = exp2f(pb[0] - m);
        l = fmaf(pb[1], w, l);
        a = fmaf(pb[2 + d], w, a);
    }
    out[((int64_t) t * n_head + h) * HD + d] = a / l;
}

// the splits a row's chunks go to: from the capacity's chunks alone
int decode_splits(int64_t n_chunks) { return (int) std::max<int64_t>(1, std::min<int64_t>(kDecSplits, n_chunks)); }

template <int HD>
void launch_window(const float* q, const uint16_t* kc, const uint16_t* vc, int n_head, int n_kv, const Range& r, int T,
                   float* scratch, int n_chunks, float* out, cudaStream_t s, KvInt8 q8 = {}) {
    const float scale = 1.4426950408889634f / sqrtf((float) HD);
    const int gx = decode_splits(n_chunks);
    launch_pdl(decode_chunk_kernel<HD>, dim3((unsigned) gx, (unsigned) n_kv, (unsigned) T), dim3(kDecThreads), 0, s, q,
               (const __half*) kc, (const __half*) vc, n_head, n_kv, r, n_chunks, scale, scratch, q8);
    check("gqa decode attention");
    launch_pdl(decode_merge_kernel<HD>, dim3((unsigned) n_head, (unsigned) T), dim3(HD), 0, s, (const float*) scratch, n_head, gx,
               out);
    check("gqa decode attention/merge");
}

// ---------------------------------------------------------------- prompt: shared helpers
__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldsm_x2(unsigned& r0, unsigned& r1, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4_t(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma16816(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma16816_h(unsigned (&c)[2], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                           unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c[0]), "+r"(c[1])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma16832_s8(int (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                            unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned pack_h2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&h);
}
__device__ __forceinline__ void cp16(void* dst, const void* src, bool ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp8(void* dst, const void* src, bool ok) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(ok ? 8 : 0));
}
__device__ __forceinline__ float h2f(uint16_t u) { return __half2float(__ushort_as_half(u)); }
__device__ __forceinline__ float ex2(float x) {   // 2^x, -inf -> 0
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float quad_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}
__device__ __forceinline__ float quad_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v + __shfl_xor_sync(0xffffffffu, v, 2);
}

// ---------------------------------------------------------------- prompt, fp16 cache
// Block (query tile, KV head): its 96 rows are the tile's positions times the KV group's query heads (head fastest), so
// a K/V tile is read once for every head of the group.  Warp w owns rows 16 w .. 16 w + 15.  Cells stream in tiles of
// 32.  QK^T on fp16 tensor cores with fp32 accumulation; P V on fp16 tensor cores accumulating each 32-cell tile in
// fp16, added to the fp32 output.  Online softmax in fp32.
constexpr int PNW = 6, PROWS = 16 * PNW, PBN = 32;
constexpr int LDH = HD + 8;    // padded fp16 row (halves)

struct PromptSmem16 {
    __half q[PROWS][LDH];
    __half k[PBN][LDH];
    __half v[PBN][LDH];
};

__global__ void __launch_bounds__(PNW * 32) prompt_f16_kernel(const float* __restrict__ q, const __half* __restrict__ kc,
                                                              const __half* __restrict__ vc, int n_head, int n_kv,
                                                              int64_t pos0, int64_t T, float* __restrict__ out) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    PromptSmem16& sm = *reinterpret_cast<PromptSmem16*>(smem_raw);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int G = n_head / n_kv, PB = PROWS / G;   // query heads per KV head, positions per tile
    const int kvh = blockIdx.y;
    const int64_t q0 = (int64_t) (gridDim.x - 1 - blockIdx.x) * PB;   // the longest tiles first
    const int64_t last_pos = min(T, q0 + PB) - 1;
    const int64_t cells = pos0 + last_pos + 1;
    const int g = lane >> 2, tq = lane & 3;
    const int ra = warp * 16 + g, rb = ra + 8;                         // this thread's two rows
    const int64_t pos_a = q0 + ra / G, pos_b = q0 + rb / G;
    const int head_a = kvh * G + ra % G, head_b = kvh * G + rb % G;

    for (int i = tid; i < PROWS * HD / 4; i += PNW * 32) {
        const int r = i / (HD / 4), c = (i % (HD / 4)) * 4;
        const int64_t pos = q0 + r / G;
        const int head = kvh * G + r % G;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (pos < T) v = *reinterpret_cast<const float4*>(q + (pos * n_head + head) * HD + c);
        *reinterpret_cast<__half2*>(&sm.q[r][c]) = __floats2half2_rn(v.x, v.y);
        *reinterpret_cast<__half2*>(&sm.q[r][c + 2]) = __floats2half2_rn(v.z, v.w);
    }

    float o[HD / 8][4];
#pragma unroll
    for (int n = 0; n < HD / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    float m_a = kNeg, m_b = kNeg, l_a = 0.0f, l_b = 0.0f;
    const int n_tiles = (int) ((cells + PBN - 1) / PBN);

    for (int ti = 0; ti < n_tiles; ++ti) {
        const int64_t c0 = (int64_t) ti * PBN;
        __syncthreads();   // the previous tile is consumed (and the query tile is written)
        for (int i = tid; i < PBN * HD / 8; i += PNW * 32) {
            const int r = i / (HD / 8), c = (i % (HD / 8)) * 8;
            const bool ok = c0 + r < cells;
            const int64_t row = (ok ? c0 + r : 0) * n_kv + kvh;
            cp16(&sm.k[r][c], kc + row * HD + c, ok);
            cp16(&sm.v[r][c], vc + row * HD + c, ok);
        }
        asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();

        // S = Q K^T for the warp's 16 rows and the tile's 32 cells (four 8-cell tiles)
        float s[PBN / 8][4];
#pragma unroll
        for (int n = 0; n < PBN / 8; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
#pragma unroll
        for (int k = 0; k < HD; k += 16) {
            unsigned a0, a1, a2, a3;
            ldsm_x4(a0, a1, a2, a3, &sm.q[warp * 16 + (lane & 15)][k + (lane >> 4) * 8]);
#pragma unroll
            for (int n = 0; n < PBN / 8; n += 2) {
                unsigned b0, b1, b2, b3;
                ldsm_x4(b0, b1, b2, b3, &sm.k[n * 8 + (lane & 7) + ((lane >> 4) << 3)][k + ((lane >> 3) & 1) * 8]);
                mma16816(s[n], a0, a1, a2, a3, b0, b1);
                mma16816(s[n + 1], a0, a1, a2, a3, b2, b3);
            }
        }
        // mask and online softmax: thread holds rows ra (s[n][0..1]) and rb (s[n][2..3])
        float mx_a = m_a, mx_b = m_b;
#pragma unroll
        for (int n = 0; n < PBN / 8; ++n)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int64_t cell = c0 + n * 8 + tq * 2 + (i & 1);
                const int64_t pos = i < 2 ? pos_a : pos_b;
                const bool vis = cell <= pos0 + pos && cell < cells;
                s[n][i] = vis ? s[n][i] * kScaleLog2 : kNeg;
                if (i < 2) mx_a = fmaxf(mx_a, s[n][i]);
                else mx_b = fmaxf(mx_b, s[n][i]);
            }
        mx_a = quad_max(mx_a);
        mx_b = quad_max(mx_b);
        const float sc_a = exp2f(m_a - mx_a), sc_b = exp2f(m_b - mx_b);
        m_a = mx_a;
        m_b = mx_b;
        float ps_a = 0.0f, ps_b = 0.0f;
#pragma unroll
        for (int n = 0; n < PBN / 8; ++n) {
            s[n][0] = s[n][0] <= kNeg ? 0.0f : exp2f(s[n][0] - m_a);
            s[n][1] = s[n][1] <= kNeg ? 0.0f : exp2f(s[n][1] - m_a);
            s[n][2] = s[n][2] <= kNeg ? 0.0f : exp2f(s[n][2] - m_b);
            s[n][3] = s[n][3] <= kNeg ? 0.0f : exp2f(s[n][3] - m_b);
            ps_a += s[n][0] + s[n][1];
            ps_b += s[n][2] + s[n][3];
        }
        l_a = l_a * sc_a + ps_a;
        l_b = l_b * sc_b + ps_b;
        // O = O * scale + P V, the tile's P V summed in fp16 on the tensor cores
        unsigned pa[PBN / 16][4];
#pragma unroll
        for (int j = 0; j < PBN / 16; ++j) {
            pa[j][0] = pack_h2(s[2 * j][0], s[2 * j][1]);
            pa[j][1] = pack_h2(s[2 * j][2], s[2 * j][3]);
            pa[j][2] = pack_h2(s[2 * j + 1][0], s[2 * j + 1][1]);
            pa[j][3] = pack_h2(s[2 * j + 1][2], s[2 * j + 1][3]);
        }
#pragma unroll
        for (int n = 0; n < HD / 8; n += 2) {
            unsigned d0[2] = {0u, 0u}, d1[2] = {0u, 0u};
#pragma unroll
            for (int j = 0; j < PBN / 16; ++j) {
                unsigned b0, b1, b2, b3;
                // V rows are cells (k), columns dims (n): transposed loads give the col-major B of dims n*8.., n*8+8..
                ldsm_x4_t(b0, b1, b2, b3, &sm.v[j * 16 + (lane & 15)][n * 8 + (lane >> 4) * 8]);
                mma16816_h(d0, pa[j][0], pa[j][1], pa[j][2], pa[j][3], b0, b1);
                mma16816_h(d1, pa[j][0], pa[j][1], pa[j][2], pa[j][3], b2, b3);
            }
            const float2 e0 = __half22float2(*reinterpret_cast<const __half2*>(&d0[0]));
            const float2 e1 = __half22float2(*reinterpret_cast<const __half2*>(&d0[1]));
            const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&d1[0]));
            const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&d1[1]));
            o[n][0] = fmaf(o[n][0], sc_a, e0.x); o[n][1] = fmaf(o[n][1], sc_a, e0.y);
            o[n][2] = fmaf(o[n][2], sc_b, e1.x); o[n][3] = fmaf(o[n][3], sc_b, e1.y);
            o[n + 1][0] = fmaf(o[n + 1][0], sc_a, f0.x); o[n + 1][1] = fmaf(o[n + 1][1], sc_a, f0.y);
            o[n + 1][2] = fmaf(o[n + 1][2], sc_b, f1.x); o[n + 1][3] = fmaf(o[n + 1][3], sc_b, f1.y);
        }
    }
    l_a = quad_sum(l_a);
    l_b = quad_sum(l_b);
    const float ia = l_a > 0.0f ? 1.0f / l_a : 0.0f, ib = l_b > 0.0f ? 1.0f / l_b : 0.0f;
#pragma unroll
    for (int n = 0; n < HD / 8; ++n) {
        const int col = n * 8 + tq * 2;
        if (pos_a < T) *reinterpret_cast<float2*>(out + (pos_a * n_head + head_a) * HD + col) = make_float2(o[n][0] * ia, o[n][1] * ia);
        if (pos_b < T) *reinterpret_cast<float2*>(out + (pos_b * n_head + head_b) * HD + col) = make_float2(o[n][2] * ib, o[n][3] * ib);
    }
}

// ---------------------------------------------------------------- prompt, int8 cache
// Adapted from Infernix (Ian Ranson and contributors), Apache-2.0: see THIRD_PARTY_LICENSES.  The structure follows
// Infernix's fast int8 prompt kernel (src/ops/softmax_attention/dense/causal_cache/int8/fast_tiled_mma.cuh); Basalt's
// cache is unrotated and unpaged, so the queries are rounded to int8 per 64 dims directly and a key tile is gathered
// cell by cell.
//
// Block (query tile, query head), longest tiles first: warp w owns the tile's positions 16 w .. 16 w + 15 of one head
// for the whole sweep, so scores, probabilities and the 256-wide output stay in registers and the only block barrier
// publishes a key tile.  Keys stream in tiles of 64 cells, K and V double-buffered as raw int8 codes with their scales
// (16-byte pieces XOR-swizzled by the row).  QK^T: int8 tensor cores, each 64-dim group's integer sum times the query's
// and the cell's scales in fp32.  P V: V decoded in registers (ldmatrix.trans on byte pairs gives a lane two adjacent
// cells of two adjacent dims, one fp16 B fragment each for an even-dim and an odd-dim n8 tile; codes widen exactly to
// fp16 and take their fp16 group scale), fp16 tensor cores summing the 64-cell tile in fp16, added to the fp32 output
// once per tile.  A partial is at most 64 * 127 * the tile's largest V scale; a tile whose scale passes 8 decodes V
// with the scales divided by a power of two and multiplies the partial back.  Online softmax in fp32.
//
// A split launch (gridDim.z > 1) gives each block a contiguous run of its key tiles; the block writes its rows
// normalized by its own sum with their (max, sum) and prompt_i8_merge_kernel combines the splits.
constexpr int I8BC = 64;                                   // cells per key tile
constexpr int I8_TILE = I8BC * HD;                         // bytes of one tile's K (or V) codes
constexpr int I8_STAGE = 2 * I8_TILE + 2 * I8BC * 4 * 2;   // K, V codes and their fp16 scales
constexpr float kI8PartialScaleLimit = 8.0f;

template <int W> constexpr int i8_smem() { return W * 16 * HD + 2 * I8_STAGE; }

// One ldmatrix.trans lane of an int8 [cell][dim] tile holds the codes {V[k][d], V[k][d+1], V[k+1][d], V[k+1][d+1]}:
// returns the fp16 B halves {V[k][d], V[k+1][d]} and {V[k][d+1], V[k+1][d+1]}, each code widened exactly (code ^ 0x80 is
// code + 128, fp16 0x6400 | byte is 1024 + byte) and multiplied once by its cell's fp16 scale.
__device__ __forceinline__ void decode_v_pair(unsigned codes, unsigned scales, unsigned& even, unsigned& odd) {
    const unsigned biased = codes ^ 0x80808080u;
    const unsigned e = __byte_perm(biased, 0x64646464u, 0x4240), o = __byte_perm(biased, 0x64646464u, 0x4341);
    const __half2 off = __float2half2_rn(1152.0f);
    const __half2 s2 = *reinterpret_cast<const __half2*>(&scales);
    const __half2 ve = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&e), off), s2);
    const __half2 vo = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&o), off), s2);
    even = *reinterpret_cast<const unsigned*>(&ve);
    odd = *reinterpret_cast<const unsigned*>(&vo);
}

template <int W, bool Split>
__global__ void __launch_bounds__(W * 32, 1) prompt_i8_kernel(const float* __restrict__ q, int n_head, int n_kv, int pos0,
                                                              int T, float* __restrict__ out, KvInt8 q8,
                                                              float* __restrict__ part_rows, float2* __restrict__ part_stats) {
    constexpr int Threads = W * 32, Br = W * 16;
    constexpr int G4 = HD / 64;           // scale groups per row
    constexpr int NT = I8BC / 8;          // score n8 tiles per key tile
    constexpr int KS = I8BC / 16;         // P V k-steps per key tile
    constexpr int DB = HD / 16;           // 16-dim output blocks
    static_assert(I8_TILE / 16 % Threads == 0 && 2 * I8BC <= Threads, "int8 prompt attention: tile copy split");
    extern __shared__ __align__(16) unsigned char smem_raw[];
    int8_t* q_s = reinterpret_cast<int8_t*>(smem_raw);

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tq = lane & 3;
    const int linear = blockIdx.x + blockIdx.y * gridDim.x;   // the first n_head blocks take every head's last tile
    const int head = linear % n_head;
    const int q0 = ((int) gridDim.x - 1 - linear / n_head) * Br;
    const int kvh = head / (n_head / n_kv);
    const int rows = min(Br, T - q0);
    const int max_qabs = pos0 + q0 + rows - 1;
    const int key_tiles = max_qabs / I8BC + 1;
    const int split = Split ? (int) blockIdx.z : 0;
    const int per_split = Split ? (key_tiles + (int) gridDim.z - 1) / (int) gridDim.z : 0;
    const int kb_begin = Split ? min(key_tiles, split * per_split) : 0;
    const int kb_end = Split ? min(key_tiles, kb_begin + per_split) : key_tiles;

    // each warp rounds its own 16 rows to int8 per 64 dims (amax / 127); a thread keeps the scales of its two C rows
    float qsr[2][G4];
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        const int row = warp * 16 + r;
        const bool valid = row < rows;
        const float* src = q + ((int64_t) (q0 + row) * n_head + head) * HD;
        float x[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) x[k] = valid ? src[lane + 32 * k] : 0.0f;
#pragma unroll
        for (int grp = 0; grp < G4; ++grp) {
            const float amax = warp_max(fmaxf(fabsf(x[2 * grp]), fabsf(x[2 * grp + 1])));
            const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int d = grp * 64 + h * 32 + lane;
                const int c = max(-127, min(127, __float2int_rn(x[2 * grp + h] * inv)));
                q_s[row * HD + (((d >> 4) ^ (row & 7)) << 4) + (d & 15)] = (int8_t) c;
            }
            if (g == (r & 7)) qsr[r >> 3][grp] = amax / 127.0f;
        }
    }

    // a key tile: 256-byte K and V rows per cell (the cache interleaves the KV heads), cells past the block's last
    // visible one zero-filled so masked columns stay finite
    const auto issue_tile = [&](int kb) {
        unsigned char* base = smem_raw + Br * HD + (kb & 1) * I8_STAGE;
        int8_t* k_s = reinterpret_cast<int8_t*>(base);
        int8_t* v_s = k_s + I8_TILE;
        uint16_t* ks_s = reinterpret_cast<uint16_t*>(v_s + I8_TILE);
        uint16_t* vs_s = ks_s + I8BC * G4;
        const int valid = min(I8BC, max_qabs + 1 - kb * I8BC);
#pragma unroll
        for (int i = 0; i < I8_TILE / 16 / Threads; ++i) {
            const int chunk = tid + i * Threads;
            const int key = chunk >> 4, c = chunk & 15;
            const bool ok = key < valid;
            const int64_t src = ((int64_t) (kb * I8BC + (ok ? key : 0)) * n_kv + kvh) * HD + c * 16;
            const int dst = key * HD + ((c ^ (key & 7)) << 4);
            cp16(k_s + dst, q8.k + src, ok);
            cp16(v_s + dst, q8.v + src, ok);
        }
        if (tid < 2 * I8BC) {
            const int key = tid & (I8BC - 1);
            const bool ok = key < valid;
            const int64_t src = ((int64_t) (kb * I8BC + (ok ? key : 0)) * n_kv + kvh) * G4;
            if (tid < I8BC) cp8(ks_s + key * G4, q8.ks + src, ok);
            else cp8(vs_s + key * G4, q8.vs + src, ok);
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    };

    const int a_mat = lane >> 3, a_rin = lane & 7;
    const int a_row = warp * 16 + a_rin + ((a_mat & 1) << 3);
    const int a_chunk = a_mat >> 1;        // 16-byte piece within a 32-byte k step
    const int b_chunk = (lane >> 3) & 1;
    const int row0 = warp * 16 + g, row1 = row0 + 8;
    const bool warp_active = warp * 16 < rows;
    const int warp_min_qabs = pos0 + q0 + warp * 16;
    const int warp_max_qabs = pos0 + q0 + min(warp * 16 + 15, rows - 1);
    const int qabs0 = row0 < rows ? pos0 + q0 + row0 : -1;
    const int qabs1 = row1 < rows ? pos0 + q0 + row1 : -1;

    float acc[DB][2][4];
#pragma unroll
    for (int b = 0; b < DB; ++b)
#pragma unroll
        for (int p = 0; p < 2; ++p) acc[b][p][0] = acc[b][p][1] = acc[b][p][2] = acc[b][p][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;

    if (kb_begin < kb_end) issue_tile(kb_begin);
#pragma unroll 1
    for (int kb = kb_begin; kb < kb_end; ++kb) {
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();   // tile kb is in, and every warp is done with the other stage
        if (kb + 1 < kb_end) issue_tile(kb + 1);
        const int k0 = kb * I8BC;
        if (!warp_active || k0 > warp_max_qabs) continue;
        const unsigned char* base = smem_raw + Br * HD + (kb & 1) * I8_STAGE;
        const int8_t* k_s = reinterpret_cast<const int8_t*>(base);
        const int8_t* v_s = k_s + I8_TILE;
        const uint16_t* ks_s = reinterpret_cast<const uint16_t*>(v_s + I8_TILE);
        const uint16_t* vs_s = ks_s + I8BC * G4;

        // S = Q K^T: 16 rows x 64 cells
        float s[NT][4];
#pragma unroll
        for (int n = 0; n < NT; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
#pragma unroll
        for (int grp = 0; grp < G4; ++grp) {
            unsigned af[2][4];
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                const int ch = (grp * 2 + kk) * 2 + a_chunk;
                ldsm_x4(af[kk][0], af[kk][1], af[kk][2], af[kk][3], q_s + a_row * HD + ((ch ^ (a_row & 7)) << 4));
            }
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                int c[4] = {0, 0, 0, 0};
                const int key = n * 8 + a_rin;
#pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
                    const int ch = (grp * 2 + kk) * 2 + b_chunk;
                    unsigned b0, b1;
                    ldsm_x2(b0, b1, k_s + key * HD + ((ch ^ (key & 7)) << 4));
                    mma16832_s8(c, af[kk][0], af[kk][1], af[kk][2], af[kk][3], b0, b1);
                }
                const int ka = n * 8 + 2 * tq;
                const float ks0 = h2f(ks_s[ka * G4 + grp]), ks1 = h2f(ks_s[(ka + 1) * G4 + grp]);
                s[n][0] = fmaf(qsr[0][grp] * ks0, (float) c[0], s[n][0]);
                s[n][1] = fmaf(qsr[0][grp] * ks1, (float) c[1], s[n][1]);
                s[n][2] = fmaf(qsr[1][grp] * ks0, (float) c[2], s[n][2]);
                s[n][3] = fmaf(qsr[1][grp] * ks1, (float) c[3], s[n][3]);
            }
        }

        // mask (only tiles crossing the warp's diagonal) and online softmax, scores kept unscaled
        const bool full = k0 + I8BC - 1 <= warp_min_qabs;
        float bm0 = -INFINITY, bm1 = -INFINITY;
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            if (!full) {
                const int c0 = k0 + n * 8 + 2 * tq, c1 = c0 + 1;
                s[n][0] = c0 <= qabs0 ? s[n][0] : -INFINITY;
                s[n][1] = c1 <= qabs0 ? s[n][1] : -INFINITY;
                s[n][2] = c0 <= qabs1 ? s[n][2] : -INFINITY;
                s[n][3] = c1 <= qabs1 ? s[n][3] : -INFINITY;
            }
            bm0 = fmaxf(bm0, fmaxf(s[n][0], s[n][1]));
            bm1 = fmaxf(bm1, fmaxf(s[n][2], s[n][3]));
        }
        const float nm0 = fmaxf(m0, quad_max(bm0)), nm1 = fmaxf(m1, quad_max(bm1));
        const float ns0 = nm0 == -INFINITY ? 0.0f : nm0 * kScaleLog2;
        const float ns1 = nm1 == -INFINITY ? 0.0f : nm1 * kScaleLog2;
        const float al0 = m0 == -INFINITY ? 0.0f : ex2(m0 * kScaleLog2 - ns0);
        const float al1 = m1 == -INFINITY ? 0.0f : ex2(m1 * kScaleLog2 - ns1);
        m0 = nm0;
        m1 = nm1;
        // the probabilities become the P V A fragments: k step j covers cells 16 j .. 16 j + 15 (score tiles 2 j, 2 j + 1)
        unsigned pa[KS][4];
        float bl0 = 0.0f, bl1 = 0.0f;
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            const float p00 = ex2(fmaf(s[n][0], kScaleLog2, -ns0)), p01 = ex2(fmaf(s[n][1], kScaleLog2, -ns0));
            const float p10 = ex2(fmaf(s[n][2], kScaleLog2, -ns1)), p11 = ex2(fmaf(s[n][3], kScaleLog2, -ns1));
            bl0 += p00 + p01;
            bl1 += p10 + p11;
            pa[n >> 1][(n & 1) * 2] = pack_h2(p00, p01);
            pa[n >> 1][(n & 1) * 2 + 1] = pack_h2(p10, p11);
        }
        l0 = fmaf(l0, al0, bl0);   // lane-partial; the quad is summed after the sweep
        l1 = fmaf(l1, al1, bl1);

        // the tile's largest V scale bounds its fp16 partials
        int shift = 0;
        {
            const uint4 v8 = *reinterpret_cast<const uint4*>(vs_s + 8 * lane);
            const __half2 mx2 = __hmax2(__hmax2(__habs2(*reinterpret_cast<const __half2*>(&v8.x)),
                                                __habs2(*reinterpret_cast<const __half2*>(&v8.y))),
                                        __hmax2(__habs2(*reinterpret_cast<const __half2*>(&v8.z)),
                                                __habs2(*reinterpret_cast<const __half2*>(&v8.w))));
            const float vmax = warp_max(fmaxf(__low2float(mx2), __high2float(mx2)));
            if (vmax > kI8PartialScaleLimit) {
                const float bounded = fminf(vmax, 65504.0f);
                while (ldexpf(bounded, -shift) > kI8PartialScaleLimit) ++shift;
            }
        }
        const __half2 down = __float2half2_rn(ldexpf(1.0f, -shift));
        const float up = ldexpf(1.0f, shift);

        // O = O * alpha + P V
#pragma unroll
        for (int grp = 0; grp < G4; ++grp) {
            unsigned vsc[KS][2];   // the group scales of the cells this lane supplies: 16 j + 2 tq (+1), and + 8
#pragma unroll
            for (int j = 0; j < KS; ++j) {
                const int key = j * 16 + 2 * tq;
                __half2 lo = __halves2half2(__ushort_as_half(vs_s[key * G4 + grp]), __ushort_as_half(vs_s[(key + 1) * G4 + grp]));
                __half2 hi = __halves2half2(__ushort_as_half(vs_s[(key + 8) * G4 + grp]),
                                            __ushort_as_half(vs_s[(key + 9) * G4 + grp]));
                if (shift) {
                    lo = __hmul2(lo, down);
                    hi = __hmul2(hi, down);
                }
                vsc[j][0] = *reinterpret_cast<const unsigned*>(&lo);
                vsc[j][1] = *reinterpret_cast<const unsigned*>(&hi);
            }
#pragma unroll
            for (int pass = 0; pass < 2; ++pass) {
                const int db0 = grp * 4 + pass * 2;
                unsigned h[2][2][2] = {};   // [d block][even, odd dims][C regs]
#pragma unroll
                for (int j = 0; j < KS; ++j) {
                    // matrices: cells 16 j + 0..7 and 8..15 of d block db0, then of db0 + 1
                    const int key = j * 16 + ((a_mat & 1) << 3) + a_rin;
                    const int ch = db0 + (a_mat >> 1);
                    unsigned r[4];
                    ldsm_x4_t(r[0], r[1], r[2], r[3], v_s + key * HD + ((ch ^ (key & 7)) << 4));
#pragma unroll
                    for (int b = 0; b < 2; ++b) {
                        unsigned e_lo, o_lo, e_hi, o_hi;
                        decode_v_pair(r[2 * b], vsc[j][0], e_lo, o_lo);
                        decode_v_pair(r[2 * b + 1], vsc[j][1], e_hi, o_hi);
                        mma16816_h(h[b][0], pa[j][0], pa[j][1], pa[j][2], pa[j][3], e_lo, e_hi);
                        mma16816_h(h[b][1], pa[j][0], pa[j][1], pa[j][2], pa[j][3], o_lo, o_hi);
                    }
                }
#pragma unroll
                for (int b = 0; b < 2; ++b)
#pragma unroll
                    for (int p = 0; p < 2; ++p) {
                        float (&a)[4] = acc[db0 + b][p];
                        float2 r0 = __half22float2(*reinterpret_cast<const __half2*>(&h[b][p][0]));
                        float2 r1 = __half22float2(*reinterpret_cast<const __half2*>(&h[b][p][1]));
                        if (shift) {
                            r0.x *= up; r0.y *= up; r1.x *= up; r1.y *= up;
                        }
                        a[0] = fmaf(a[0], al0, r0.x);
                        a[1] = fmaf(a[1], al0, r0.y);
                        a[2] = fmaf(a[2], al1, r1.x);
                        a[3] = fmaf(a[3], al1, r1.y);
                    }
            }
        }
    }

    l0 = quad_sum(l0);
    l1 = quad_sum(l1);
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
    // even n8 tiles hold dims 16 b + 2 n, odd ones 16 b + 2 n + 1: lane (g, tq) owns dims 16 b + 4 tq .. + 3 of its rows
    const auto store = [&](int row, int e, float m, float l, float inv) {
        if (row >= rows) return;
        const int64_t idx = Split ? ((int64_t) split * T + q0 + row) * n_head + head : (int64_t) (q0 + row) * n_head + head;
        float* dst = (Split ? part_rows : out) + idx * HD;
        if (Split && tq == 0) part_stats[idx] = make_float2(m, l);
#pragma unroll
        for (int b = 0; b < DB; ++b)
            *reinterpret_cast<float4*>(dst + b * 16 + 4 * tq) =
                make_float4(acc[b][0][e] * inv, acc[b][1][e] * inv, acc[b][0][e + 1] * inv, acc[b][1][e + 1] * inv);
    };
    store(row0, 0, m0, l0, i0);
    store(row1, 2, m1, l1, i1);
}

// Block (row, query head), one thread per dim: the splits' rows weighted by sum_s 2^((m_s - M) scale).
__global__ void __launch_bounds__(HD) prompt_i8_merge_kernel(const float* __restrict__ part_rows,
                                                             const float2* __restrict__ part_stats, int n_head, int T,
                                                             int splits, float* __restrict__ out) {
    const int t = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
    float mx = -INFINITY;
    for (int s = 0; s < splits; ++s) {
        const float2 st = part_stats[((int64_t) s * T + t) * n_head + head];
        if (st.y > 0.0f) mx = fmaxf(mx, st.x);
    }
    float num = 0.0f, den = 0.0f;
    for (int s = 0; s < splits; ++s) {
        const int64_t idx = ((int64_t) s * T + t) * n_head + head;
        const float2 st = part_stats[idx];
        if (st.y > 0.0f) {
            const float w = st.y * exp2f((st.x - mx) * kScaleLog2);
            num = fmaf(w, part_rows[idx * HD + d], num);
            den += w;
        }
    }
    out[((int64_t) t * n_head + head) * HD + d] = den > 0.0f ? num / den : 0.0f;
}

template <int W, bool Split>
void launch_prompt_i8(const float* q, int n_head, int n_kv, int pos0, int T, float* out, KvInt8 q8, int splits,
                      float* scratch, cudaStream_t st) {
    static bool raised = false;
    if (!raised) {
        if (cudaFuncSetAttribute(prompt_i8_kernel<W, Split>, cudaFuncAttributeMaxDynamicSharedMemorySize, i8_smem<W>()) !=
            cudaSuccess) {
            basalt::log::error("basalt", "gqa_prompt_attn: shared-memory opt-in failed");
            std::exit(1);
        }
        raised = true;
    }
    float* rows = scratch;
    float2* stats = reinterpret_cast<float2*>(scratch + (size_t) splits * T * n_head * HD);
    const dim3 grid((unsigned) ((T + W * 16 - 1) / (W * 16)), (unsigned) n_head, (unsigned) splits);
    prompt_i8_kernel<W, Split><<<grid, W * 32, i8_smem<W>(), st>>>(q, n_head, n_kv, pos0, T, out, q8, rows, stats);
    if (Split) {
        check("gqa_prompt_attn");
        prompt_i8_merge_kernel<<<dim3((unsigned) T, (unsigned) n_head), HD, 0, st>>>(rows, stats, n_head, T, splits, out);
    }
}

// The int8 launch's block shape and key splits, as Infernix plans them (int8_fast_prompt_plan): a launch takes about
// (waves) x (visible cells / splits) sweeps of one cell by an 8-warp block; a 4-warp block covers half the rows in 0.8 of
// that time, unsplit; a split launch also writes and merges every split's rows, about half a cell's sweep per row, split
// and head over 24 heads.  Infernix measured this within 0.14 % of the fastest choice on average, 3.8 % at worst.
struct I8Plan {
    int warps = 8, splits = 1;
};
I8Plan plan_prompt_i8(int n_head, int64_t T, int64_t cells, size_t split_floats) {
    constexpr int kMaxSplits = 8;
    int dev = 0, sms = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    sms = std::max(sms, 1);
    const auto waves = [&](int64_t blocks) { return (double) ((blocks + sms - 1) / sms); };
    const double keys = (double) cells;
    I8Plan best{4, 1};
    double best_cost = waves((T + 63) / 64 * n_head) * 0.8 * keys;
    const int64_t blocks = (T + 127) / 128 * n_head;
    for (int s = 1; s <= kMaxSplits; ++s) {
        if (s > 1 && gqa_prompt_split_floats(T, n_head, s) > split_floats) break;
        double cost = waves(blocks * s) * keys / s;
        if (s > 1) cost += 0.5 / 24.0 * (double) T * s * n_head;
        if (cost < best_cost) {
            best_cost = cost;
            best = {8, s};
        }
    }
    return best;
}

}  // namespace

size_t gqa_decode_scratch_floats(int64_t max_cells, int T, int n_head, int head_dim) {
    const int64_t chunks = (max_cells + CHUNK - 1) / CHUNK + 1;
    return (size_t) T * n_head * decode_splits(chunks) * (head_dim + 2);
}

void gqa_decode_attn(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, int n_head, int n_kv,
                     int64_t pos0, int T, float* scratch, int64_t max_cells, float* out, void* stream,
                     const int32_t* pos_dev, KvInt8 q8) {
    if (T < 1 || T > kGqaMaxWindow || n_kv < 1 || n_head % n_kv || n_head / n_kv > GMAX || pos0 + T > max_cells) {
        basalt::log::error("basalt", "gqa_decode_attn: invalid arguments");
        std::exit(1);
    }
    Range r;
    r.pos0 = pos0;
    r.cap = max_cells;
    r.pos_dev = pos_dev;
    const int n_chunks = (int) ((max_cells + CHUNK - 1) / CHUNK);
    launch_window<256>(q, k_cache, v_cache, n_head, n_kv, r, T, scratch, n_chunks, out, (cudaStream_t) stream, q8);
}

void gqa_window_attn(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, int n_head, int n_kv, int head_dim,
                     int64_t pos0, int T, int64_t hi, int64_t window, int64_t cap, float* scratch, int64_t scratch_cells,
                     float* out, void* stream) {
    if (T < 1 || T > kGqaMaxWindow || n_kv < 1 || n_head % n_kv || n_head / n_kv > GMAX || window <= 0 ||
        (head_dim != 128 && head_dim != 256) || hi < pos0 + T) {
        basalt::log::error("basalt", "gqa_window_attn: invalid arguments");
        std::exit(1);
    }
    Range r;
    r.pos0 = pos0;
    r.hi = hi;
    r.window = window;
    r.cap = cap;
    r.causal = 0;
    const int64_t lo = std::max<int64_t>(0, pos0 - window + 1);
    r.base = (int) (lo / CHUNK);
    const int used = (int) ((hi + CHUNK - 1) / CHUNK) - r.base;
    const int n_chunks = (int) ((scratch_cells + CHUNK - 1) / CHUNK) + 1;
    if (used > n_chunks) {
        basalt::log::error("basalt", "gqa_window_attn: the window passes its scratch");
        std::exit(1);
    }
    if (head_dim == 128) launch_window<128>(q, k_cache, v_cache, n_head, n_kv, r, T, scratch, n_chunks, out, (cudaStream_t) stream);
    else launch_window<256>(q, k_cache, v_cache, n_head, n_kv, r, T, scratch, n_chunks, out, (cudaStream_t) stream);
}

size_t gqa_prompt_split_floats(int64_t T, int n_head, int splits) {
    return (size_t) splits * T * n_head * (HD + 2);   // the splits' rows, then their (max, sum) pairs
}

void gqa_prompt_attn(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, int n_head, int n_kv,
                     int64_t pos0, int64_t T, float* out, void* stream, KvInt8 q8, float* split_scratch, size_t split_floats) {
    if (T <= 0) return;
    if (n_kv < 1 || n_head % n_kv || (!q8.k && PROWS % (n_head / n_kv)) || pos0 + T > INT32_MAX) {
        basalt::log::error("basalt", "gqa_prompt_attn: unsupported head layout or cell count");
        std::exit(1);
    }
    const cudaStream_t st = (cudaStream_t) stream;
    if (q8.k) {
        const I8Plan p = plan_prompt_i8(n_head, T, pos0 + T, split_scratch ? split_floats : 0);
        if (p.warps == 4) launch_prompt_i8<4, false>(q, n_head, n_kv, (int) pos0, (int) T, out, q8, 1, nullptr, st);
        else if (p.splits == 1) launch_prompt_i8<8, false>(q, n_head, n_kv, (int) pos0, (int) T, out, q8, 1, nullptr, st);
        else launch_prompt_i8<8, true>(q, n_head, n_kv, (int) pos0, (int) T, out, q8, p.splits, split_scratch, st);
    } else {
        static bool raised = false;
        if (!raised) {
            if (cudaFuncSetAttribute(prompt_f16_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(PromptSmem16)) !=
                cudaSuccess) {
                basalt::log::error("basalt", "gqa_prompt_attn: shared-memory opt-in failed");
                std::exit(1);
            }
            raised = true;
        }
        const int pb = PROWS / (n_head / n_kv);
        const dim3 grid((unsigned) ((T + pb - 1) / pb), (unsigned) n_kv);
        prompt_f16_kernel<<<grid, PNW * 32, sizeof(PromptSmem16), st>>>(q, (const __half*) k_cache, (const __half*) v_cache,
                                                                         n_head, n_kv, pos0, T, out);
    }
    check("gqa_prompt_attn");
}

}  // namespace basalt::kernels
