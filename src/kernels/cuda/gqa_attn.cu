// src/kernels/cuda/gqa_attn.cu - see include/basalt/kernels/gqa_attn.hpp.
#include "basalt/kernels/gqa_attn.hpp"
#include "basalt/kernels/pdl.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include "basalt/log.hpp"

namespace basalt::kernels {
namespace {

constexpr int GMAX = 8;               // query heads per KV head
constexpr int CHUNK = 32;             // cells per decode chunk
constexpr int kDecSplits = 64;        // a decode row's blocks per KV head (its chunks dealt round robin)
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

// Block (query head, row t), one thread per dim: the row's splits merged in order.
template <int HD>
__global__ void __launch_bounds__(HD) decode_merge_kernel(const float* part, int n_head, int gx, float* out, const float* gate_q,
                                                          uint16_t* gated16) {
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
    const float o = a / l;
    const int64_t i = ((int64_t) t * n_head + h) * HD + d;
    if (out) out[i] = o;
    if (gated16) {
        const float g = gate_q[((int64_t) t * n_head + h) * 2 * HD + HD + d];
        gated16[i] = __bfloat16_as_ushort(__float2bfloat16_rn(o / (1.0f + expf(-g))));
    }
}

// the splits a row's chunks go to: from the capacity's chunks alone
int decode_splits(int64_t n_chunks) { return (int) std::max<int64_t>(1, std::min<int64_t>(kDecSplits, n_chunks)); }

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

__device__ __forceinline__ void ldsm_x2_t(unsigned& r0, unsigned& r1, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(smem_u32(p)));
}

// ---------------------------------------------------------------- decode on the tensor cores
// Block (split b, KV head, 16 query rows): the rows are the window's (row, head) pairs of the KV group, m = t G + h.  Its
// chunks b, b + gx, ... of CHUNK cells: K and V decoded to fp16 in shared memory, S = Q K^T on mma m16n8k16 (fp16 in,
// fp32 accumulate), an online softmax per row over its cells, O = O alpha + P V (P in fp16) on mma; the rows' partials
// {max, sum, O} go to the merge.  Every row's arithmetic is its own (mma columns and rows never meet), the chunks follow
// positions and the splits the capacity, so a row's bits are the same at any window width.
template <int HD>
// (launched through launch_pdl: q and the cache come from the kernels before, read after pdl_wait, so not __restrict__)
__global__ void __launch_bounds__(128) decode_mma_kernel(const float* q, const __half* kc, const __half* vc, int n_head, int n_kv,
                                                         Range r, int n_chunks, float scale, float* part, KvInt8 q8, int T) {
    constexpr int LD = HD + 8;          // padded rows (halves): ldmatrix without bank conflicts
    constexpr int NT = HD / 32;         // P V: a warp's 8-dim tiles
    __shared__ __align__(16) __half qs[16][LD];
    __shared__ __align__(16) __half ks[CHUNK][LD];
    __shared__ __align__(16) __half vs[CHUNK][LD];
    __shared__ float ss[16][CHUNK + 1];
    __shared__ __align__(16) __half ps[16][CHUNK + 8];
    __shared__ float rm[16], rl[16], ra[16];
    __shared__ int64_t rlo[16], rhi[16];
    pdl_launch_dependents();
    pdl_wait();
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    const int kvh = blockIdx.y, gx = gridDim.x, G = n_head / n_kv, M = T * G, m0 = blockIdx.z * 16;
    for (int idx = tid; idx < 16 * HD; idx += 128) {
        const int i = idx / HD, d = idx - i * HD, m = m0 + i;
        const float v = m < M ? q[((int64_t) (m / G) * n_head + kvh * G + m % G) * HD + d] * scale : 0.0f;
        qs[i][d] = __float2half_rn(v);
    }
    if (tid < 16) {
        const int m = m0 + tid;
        rlo[tid] = m < M ? row_lo(r, m / G) : 0;
        rhi[tid] = m < M ? row_hi(r, m / G) : 0;
        rm[tid] = kNeg;
        rl[tid] = 0.0f;
    }
    __syncthreads();
    int64_t hi_max = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) hi_max = max(hi_max, rhi[i]);
    float acc[NT][4];
#pragma unroll
    for (int n = 0; n < NT; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
    for (int chunk = blockIdx.x; chunk < n_chunks; chunk += gx) {
        const int64_t c0 = (int64_t) (r.base + chunk) * CHUNK;
        if (c0 >= hi_max) break;   // past every row's cells
        const int64_t cell0 = c0 % r.cap;
        // K and V of the chunk's cells as fp16 (int8: code x scale), zero past the last row's cells
        for (int idx = tid; idx < CHUNK * HD / 8; idx += 128) {
            const int cell = idx / (HD / 8), d8 = (idx - cell * (HD / 8)) * 8;
            uint4 kw = make_uint4(0, 0, 0, 0), vw = make_uint4(0, 0, 0, 0);
            if (c0 + cell < hi_max) {
                int64_t ce = cell0 + cell;
                if (ce >= r.cap) ce -= r.cap;
                const int64_t row = ce * n_kv + kvh;
                if (q8.k) {
                    const float sk = __half2float(__ushort_as_half(q8.ks[row * (HD / 64) + d8 / 64]));
                    const float sv = __half2float(__ushort_as_half(q8.vs[row * (HD / 64) + d8 / 64]));
                    int8_t kb[8], vb[8];
                    *reinterpret_cast<int2*>(kb) = *reinterpret_cast<const int2*>(q8.k + row * HD + d8);
                    *reinterpret_cast<int2*>(vb) = *reinterpret_cast<const int2*>(q8.v + row * HD + d8);
                    __half2 kh[4], vh[4];
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        kh[e] = __floats2half2_rn((float) kb[2 * e] * sk, (float) kb[2 * e + 1] * sk);
                        vh[e] = __floats2half2_rn((float) vb[2 * e] * sv, (float) vb[2 * e + 1] * sv);
                    }
                    kw = *reinterpret_cast<uint4*>(kh);
                    vw = *reinterpret_cast<uint4*>(vh);
                } else {
                    kw = *reinterpret_cast<const uint4*>(kc + row * HD + d8);
                    vw = *reinterpret_cast<const uint4*>(vc + row * HD + d8);
                }
            }
            *reinterpret_cast<uint4*>(&ks[cell][d8]) = kw;
            *reinterpret_cast<uint4*>(&vs[cell][d8]) = vw;
        }
        __syncthreads();
        // S = Q K^T: warp w the cells 8w .. 8w + 7
        for (int nc = warp; nc < CHUNK / 8; nc += 4) {
            float s4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int k = 0; k < HD; k += 16) {
                unsigned a0, a1, a2, a3, b0, b1;
                ldsm_x4(a0, a1, a2, a3, &qs[lane & 15][k + (lane >> 4) * 8]);
                ldsm_x2(b0, b1, &ks[nc * 8 + (lane & 7)][k + ((lane >> 3) & 1) * 8]);
                mma16816(s4, a0, a1, a2, a3, b0, b1);
            }
            ss[g][nc * 8 + 2 * c] = s4[0];
            ss[g][nc * 8 + 2 * c + 1] = s4[1];
            ss[g + 8][nc * 8 + 2 * c] = s4[2];
            ss[g + 8][nc * 8 + 2 * c + 1] = s4[3];
        }
        __syncthreads();
        // each row: the chunk's max over its cells, p = exp2(s - max') there (0 elsewhere), the running {max, sum}
        for (int i = warp; i < 16; i += 4) {
            float pv[CHUNK / 32], mx = kNeg;
#pragma unroll
            for (int u = 0; u < CHUNK / 32; ++u) {
                const int cc = lane + 32 * u;
                const int64_t pos = c0 + cc;
                pv[u] = pos >= rlo[i] && pos < rhi[i] ? ss[i][cc] : -INFINITY;
                mx = fmaxf(mx, pv[u]);
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float mold = rm[i], mnew = fmaxf(mold, mx);
            float l = 0.0f;
#pragma unroll
            for (int u = 0; u < CHUNK / 32; ++u) {
                const float p = pv[u] == -INFINITY ? 0.0f : exp2f(pv[u] - mnew);
                ps[i][lane + 32 * u] = __float2half_rn(p);
                l += p;
            }
            l = warp_sum(l);
            if (lane == 0) {
                const float alpha = exp2f(mold - mnew);
                ra[i] = alpha;
                rl[i] = fmaf(rl[i], alpha, l);
                rm[i] = mnew;
            }
        }
        __syncthreads();
        // O = O alpha + P V: warp w the dims w HD / 4 ..
        {
            const float ag = ra[g], ag8 = ra[g + 8];
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                acc[n][0] *= ag;
                acc[n][1] *= ag;
                acc[n][2] *= ag8;
                acc[n][3] *= ag8;
            }
#pragma unroll
            for (int kk = 0; kk < CHUNK; kk += 16) {
                unsigned a0, a1, a2, a3;
                ldsm_x4(a0, a1, a2, a3, &ps[lane & 15][kk + (lane >> 4) * 8]);
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    unsigned b0, b1;
                    ldsm_x2_t(b0, b1, &vs[kk + (lane & 15)][warp * (HD / 4) + n * 8]);
                    mma16816(acc[n], a0, a1, a2, a3, b0, b1);
                }
            }
        }
        __syncthreads();   // the shared tiles are free for the next chunk
    }
#pragma unroll
    for (int hr = 0; hr < 2; ++hr) {
        const int i = g + 8 * hr, m = m0 + i;
        if (m >= M) continue;
        float* out = part + (((int64_t) (m / G) * n_head + (int64_t) kvh * G + m % G) * gx + blockIdx.x) * (HD + 2);
        if (warp == 0 && c == 0) {
            out[0] = rm[i];
            out[1] = rl[i];
        }
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            const int d = warp * (HD / 4) + n * 8 + 2 * c;
            out[2 + d] = acc[n][2 * hr];
            out[3 + d] = acc[n][2 * hr + 1];
        }
    }
}

template <int HD>
void launch_window(const float* q, const uint16_t* kc, const uint16_t* vc, int n_head, int n_kv, const Range& r, int T,
                   float* scratch, int n_chunks, float* out, cudaStream_t s, KvInt8 q8 = {}, const float* gate_q = nullptr,
                   uint16_t* gated16 = nullptr) {
    const float scale = 1.4426950408889634f / sqrtf((float) HD);
    const int gx = decode_splits(n_chunks);
    const int tiles = (T * (n_head / n_kv) + 15) / 16;
    launch_pdl(decode_mma_kernel<HD>, dim3((unsigned) gx, (unsigned) n_kv, (unsigned) tiles), dim3(128), 0, s, q,
               (const __half*) kc, (const __half*) vc, n_head, n_kv, r, n_chunks, scale, scratch, q8, T);
    check("gqa decode attention");
    launch_pdl(decode_merge_kernel<HD>, dim3((unsigned) n_head, (unsigned) T), dim3(HD), 0, s, (const float*) scratch, n_head, gx,
               out, gate_q, gated16);
    check("gqa decode attention/merge");
}

__device__ __forceinline__ void mma16832_u8s8(int (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                              unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
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
// and the cell's scales in fp32.  P V in 8 bits (Infernix's prompt-pv8): per 64-dim V group, each probability times
// its cell's group scale, rounded to u8 against its row's largest product over the tile (255 = that maximum); u8 x s8
// tensor cores sum the stored V codes exactly in int32, and the partial times the row's step joins the fp32 output.
// Online softmax in fp32.
//
// A split launch (gridDim.z > 1) gives each block a contiguous run of its key tiles; the block writes its rows
// normalized by its own sum with their (max, sum) and prompt_i8_merge_kernel combines the splits.
constexpr int I8BC = 64;                                   // cells per key tile
constexpr int I8_TILE = I8BC * HD;                         // bytes of one tile's K (or V) codes
constexpr int I8_STAGE = 2 * I8_TILE + 2 * I8BC * 4 * 2;   // K, V codes and their fp16 scales

template <int W> constexpr int i8_smem() { return W * 16 * HD + 2 * I8_STAGE; }

// IO: the queries and the output rows in FP32 or BF16 (the prompt chunk's)
__device__ __forceinline__ float io_ld(const float* p) { return *p; }
__device__ __forceinline__ float io_ld(const __nv_bfloat16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ void io_st4(float* p, float4 v) { *reinterpret_cast<float4*>(p) = v; }
__device__ __forceinline__ void io_st4(__nv_bfloat16* p, float4 v) {
    const __nv_bfloat162 a = __floats2bfloat162_rn(v.x, v.y), b = __floats2bfloat162_rn(v.z, v.w);
    *reinterpret_cast<uint2*>(p) = make_uint2(*reinterpret_cast<const unsigned*>(&a), *reinterpret_cast<const unsigned*>(&b));
}
__device__ __forceinline__ void io_st1(float* p, float v) { *p = v; }
__device__ __forceinline__ void io_st1(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }

template <int W, bool Split, class IO = float>
__global__ void __launch_bounds__(W * 32, 1) prompt_i8_kernel(const IO* __restrict__ q, int n_head, int n_kv, int pos0,
                                                              int T, IO* __restrict__ out, KvInt8 q8,
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
        const IO* src = q + ((int64_t) (q0 + row) * n_head + head) * HD;
        float x[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) x[k] = valid ? io_ld(src + lane + 32 * k) : 0.0f;
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

        // 8-bit P V (Infernix's prompt-pv8, fast_tiled_mma.cuh): per 64-dim V group, each probability times its
        // cell's group scale, rounded to u8 against its row's largest product over the tile (255 = that maximum);
        // u8 x s8 tensor cores sum the stored V codes exactly in int32, and the partial times the row's step joins the
        // fp32 output.  A k32 step takes the lane's own cells {2t, 2t+1, 8+2t, 9+2t} (+16), as pa holds them.
        const __half2 origin = __float2half2_rn(1024.0f);
#pragma unroll
        for (int grp = 0; grp < G4; ++grp) {
            __half2 prod[KS][4];
            __half2 mx0 = __float2half2_rn(0.0f), mx1 = mx0;
#pragma unroll
            for (int j = 0; j < KS; ++j) {
                const int key = j * 16 + 2 * tq;
                const __half2 lo = __halves2half2(__ushort_as_half(vs_s[key * G4 + grp]), __ushort_as_half(vs_s[(key + 1) * G4 + grp]));
                const __half2 hi = __halves2half2(__ushort_as_half(vs_s[(key + 8) * G4 + grp]),
                                                  __ushort_as_half(vs_s[(key + 9) * G4 + grp]));
                prod[j][0] = __hmul2(*reinterpret_cast<const __half2*>(&pa[j][0]), lo);
                prod[j][1] = __hmul2(*reinterpret_cast<const __half2*>(&pa[j][1]), lo);
                prod[j][2] = __hmul2(*reinterpret_cast<const __half2*>(&pa[j][2]), hi);
                prod[j][3] = __hmul2(*reinterpret_cast<const __half2*>(&pa[j][3]), hi);
                mx0 = __hmax2(mx0, __hmax2(prod[j][0], prod[j][2]));
                mx1 = __hmax2(mx1, __hmax2(prod[j][1], prod[j][3]));
            }
            const float rm0 = quad_max(fmaxf(__low2float(mx0), __high2float(mx0)));
            const float rm1 = quad_max(fmaxf(__low2float(mx1), __high2float(mx1)));
            // codes rint(product * inverse), the fp16 inverse rounded down so none passes 255; the step is its reciprocal
            const __half inv0 = __float2half_rd(rm0 > 0.0f ? 255.0f / rm0 : 0.0f);
            const __half inv1 = __float2half_rd(rm1 > 0.0f ? 255.0f / rm1 : 0.0f);
            const float step0 = __hgt(inv0, __float2half(0.0f)) ? 1.0f / __half2float(inv0) : 0.0f;
            const float step1 = __hgt(inv1, __float2half(0.0f)) ? 1.0f / __half2float(inv1) : 0.0f;
            const __half2 i02 = __half2half2(inv0), i12 = __half2half2(inv1);
            unsigned a8[2][4];   // fp16 1024 + n holds n in its low byte
#pragma unroll
            for (int st = 0; st < 2; ++st) {
                unsigned qv[2][4];
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int j = 2 * st + h;
                    const __half2 c0 = __hfma2(prod[j][0], i02, origin), c1 = __hfma2(prod[j][1], i12, origin);
                    const __half2 c2 = __hfma2(prod[j][2], i02, origin), c3 = __hfma2(prod[j][3], i12, origin);
                    qv[h][0] = *reinterpret_cast<const unsigned*>(&c0);
                    qv[h][1] = *reinterpret_cast<const unsigned*>(&c1);
                    qv[h][2] = *reinterpret_cast<const unsigned*>(&c2);
                    qv[h][3] = *reinterpret_cast<const unsigned*>(&c3);
                }
                a8[st][0] = __byte_perm(qv[0][0], qv[0][2], 0x6420);
                a8[st][1] = __byte_perm(qv[0][1], qv[0][3], 0x6420);
                a8[st][2] = __byte_perm(qv[1][0], qv[1][2], 0x6420);
                a8[st][3] = __byte_perm(qv[1][1], qv[1][3], 0x6420);
            }
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int db = grp * 4 + b;
                int ev[4] = {0, 0, 0, 0}, od[4] = {0, 0, 0, 0};
#pragma unroll
                for (int st = 0; st < 2; ++st) {
                    // matrices: cells 32 st + 0..7, 8..15, 16..23, 24..31 of d block db; lane (g, t) gets cells
                    // {2t, 2t+1} at dims {2g, 2g+1}
                    const int key = st * 32 + (a_mat << 3) + a_rin;
                    unsigned r[4];
                    ldsm_x4_t(r[0], r[1], r[2], r[3], v_s + key * HD + ((db ^ (key & 7)) << 4));
                    mma16832_u8s8(ev, a8[st][0], a8[st][1], a8[st][2], a8[st][3], __byte_perm(r[0], r[1], 0x6420),
                                  __byte_perm(r[2], r[3], 0x6420));
                    mma16832_u8s8(od, a8[st][0], a8[st][1], a8[st][2], a8[st][3], __byte_perm(r[0], r[1], 0x7531),
                                  __byte_perm(r[2], r[3], 0x7531));
                }
                float (&e)[4] = acc[db][0];
                float (&o)[4] = acc[db][1];
                e[0] = fmaf(e[0], al0, step0 * (float) ev[0]);
                e[1] = fmaf(e[1], al0, step0 * (float) ev[1]);
                e[2] = fmaf(e[2], al1, step1 * (float) ev[2]);
                e[3] = fmaf(e[3], al1, step1 * (float) ev[3]);
                o[0] = fmaf(o[0], al0, step0 * (float) od[0]);
                o[1] = fmaf(o[1], al0, step0 * (float) od[1]);
                o[2] = fmaf(o[2], al1, step1 * (float) od[2]);
                o[3] = fmaf(o[3], al1, step1 * (float) od[3]);
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
        if (Split && tq == 0) part_stats[idx] = make_float2(m, l);
#pragma unroll
        for (int b = 0; b < DB; ++b) {
            const float4 v = make_float4(acc[b][0][e] * inv, acc[b][1][e] * inv, acc[b][0][e + 1] * inv, acc[b][1][e + 1] * inv);
            if (Split) io_st4(part_rows + idx * HD + b * 16 + 4 * tq, v);   // the splits' rows stay FP32
            else io_st4(out + idx * HD + b * 16 + 4 * tq, v);
        }
    };
    store(row0, 0, m0, l0, i0);
    store(row1, 2, m1, l1, i1);
}

// Block (row, query head), one thread per dim: the splits' rows weighted by sum_s 2^((m_s - M) scale).
template <class IO>
__global__ void __launch_bounds__(HD) prompt_i8_merge_kernel(const float* __restrict__ part_rows,
                                                             const float2* __restrict__ part_stats, int n_head, int T,
                                                             int splits, IO* __restrict__ out) {
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
    io_st1(out + ((int64_t) t * n_head + head) * HD + d, den > 0.0f ? num / den : 0.0f);
}

template <int W, bool Split, class IO = float>
void launch_prompt_i8(const IO* q, int n_head, int n_kv, int pos0, int T, IO* out, KvInt8 q8, int splits,
                      float* scratch, cudaStream_t st) {
    static bool raised = false;
    if (!raised) {
        if (cudaFuncSetAttribute(prompt_i8_kernel<W, Split, IO>, cudaFuncAttributeMaxDynamicSharedMemorySize, i8_smem<W>()) !=
            cudaSuccess) {
            basalt::log::error("basalt", "gqa_prompt_attn: shared-memory opt-in failed");
            std::exit(1);
        }
        raised = true;
    }
    float* rows = scratch;
    float2* stats = reinterpret_cast<float2*>(scratch + (size_t) splits * T * n_head * HD);
    const dim3 grid((unsigned) ((T + W * 16 - 1) / (W * 16)), (unsigned) n_head, (unsigned) splits);
    prompt_i8_kernel<W, Split, IO><<<grid, W * 32, i8_smem<W>(), st>>>(q, n_head, n_kv, pos0, T, out, q8, rows, stats);
    if (Split) {
        check("gqa_prompt_attn");
        prompt_i8_merge_kernel<IO><<<dim3((unsigned) T, (unsigned) n_head), HD, 0, st>>>(rows, stats, n_head, T, splits, out);
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
                     const int32_t* pos_dev, KvInt8 q8, const float* gate_q, uint16_t* gated16) {
    if (T < 1 || T > kGqaMaxWindow || n_kv < 1 || n_head % n_kv || n_head / n_kv > GMAX || pos0 + T > max_cells) {
        basalt::log::error("basalt", "gqa_decode_attn: invalid arguments");
        std::exit(1);
    }
    Range r;
    r.pos0 = pos0;
    r.cap = max_cells;
    r.pos_dev = pos_dev;
    const int n_chunks = (int) ((max_cells + CHUNK - 1) / CHUNK);
    launch_window<256>(q, k_cache, v_cache, n_head, n_kv, r, T, scratch, n_chunks, out, (cudaStream_t) stream, q8, gate_q, gated16);
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

void gqa_prompt_attn_bf16(const uint16_t* q, int n_head, int n_kv, int64_t pos0, int64_t T, uint16_t* out, void* stream,
                          KvInt8 q8, float* split_scratch, size_t split_floats) {
    if (T <= 0) return;
    if (!q8.k || n_kv < 1 || n_head % n_kv || pos0 + T > INT32_MAX) {
        basalt::log::error("basalt", "gqa_prompt_attn_bf16: an int8 cache and a supported head layout");
        std::exit(1);
    }
    const cudaStream_t st = (cudaStream_t) stream;
    const __nv_bfloat16* qb = reinterpret_cast<const __nv_bfloat16*>(q);
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(out);
    const I8Plan p = plan_prompt_i8(n_head, T, pos0 + T, split_scratch ? split_floats : 0);
    if (p.warps == 4) launch_prompt_i8<4, false>(qb, n_head, n_kv, (int) pos0, (int) T, ob, q8, 1, nullptr, st);
    else if (p.splits == 1) launch_prompt_i8<8, false>(qb, n_head, n_kv, (int) pos0, (int) T, ob, q8, 1, nullptr, st);
    else launch_prompt_i8<8, true>(qb, n_head, n_kv, (int) pos0, (int) T, ob, q8, p.splits, split_scratch, st);
    check("gqa_prompt_attn_bf16");
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
