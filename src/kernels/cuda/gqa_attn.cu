// src/kernels/cuda/gqa_attn.cu - see include/basalt/kernels/gqa_attn.hpp.
#include "basalt/kernels/gqa_attn.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include "basalt/log.hpp"

namespace basalt::kernels {
namespace {

constexpr int GMAX = 8;               // query heads per KV head
constexpr int CHUNK = 256;            // cells per decode block
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
// `base`, so a row's chunks and sums depend on its own range only.
struct Range {
    int64_t pos0 = 0, hi = 0, window = 0, cap = 0;
    int causal = 1, base = 0;
    const int32_t* pos_dev = nullptr;   // pos0 read from the device (a captured window): the grid covers the capacity
};
__device__ __forceinline__ int64_t row_pos0(const Range& r) { return r.pos_dev ? (int64_t) *r.pos_dev : r.pos0; }
__device__ __forceinline__ int64_t row_hi(const Range& r, int t) { return r.causal ? row_pos0(r) + t + 1 : r.hi; }
__device__ __forceinline__ int64_t row_lo(const Range& r, int t) {
    return r.window > 0 ? max((int64_t) 0, row_pos0(r) + t - r.window + 1) : 0;
}

// Block (chunk, KV head, row t), one thread per cell for the scores: its key row against the G query heads (held in
// shared memory, read by every thread at once); one warp per head for the chunk's max and sum (a fixed butterfly);
// then p . V with one thread per dim, cells in order.  Writes the chunk's partial {max, sum, acc[HD]} per query head.
template <int HD>
__global__ void __launch_bounds__(CHUNK) decode_chunk_kernel(const float* __restrict__ q, const __half* __restrict__ kc,
                                                             const __half* __restrict__ vc, int n_head, int n_kv, Range r,
                                                             int n_chunks, float scale, float* __restrict__ part, KvInt8 q8) {
    __shared__ float s[CHUNK][GMAX + 1];
    __shared__ __align__(16) float qs[GMAX][HD];
    __shared__ float mx[GMAX], sm[GMAX];
    const int chunk = blockIdx.x, kvh = blockIdx.y, t = blockIdx.z;
    const int G = n_head / n_kv;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int64_t lo = row_lo(r, t), hi = row_hi(r, t);
    const int64_t c0 = (int64_t) (r.base + chunk) * CHUNK;
    if (c0 >= hi) return;   // past the row's cells: the merge does not read it
    const int64_t first = max(c0, lo), end = min(c0 + CHUNK, hi);
    const int n = (int) (end - first);   // may be <= 0
    float* out = part + (((int64_t) t * n_head + (int64_t) kvh * G) * n_chunks + chunk) * (HD + 2);
    if (n <= 0) {
        for (int h = 0; h < G; ++h) {
            float* o = out + (int64_t) h * n_chunks * (HD + 2);
            if (threadIdx.x == 0) { o[0] = kNeg; o[1] = 0.0f; }
            for (int d = threadIdx.x; d < HD; d += CHUNK) o[2 + d] = 0.0f;
        }
        return;
    }
    for (int h = 0; h < G; ++h)
        for (int d = threadIdx.x; d < HD; d += CHUNK) qs[h][d] = q[((int64_t) t * n_head + kvh * G + h) * HD + d] * scale;
    __syncthreads();
    // the chunk's cells are consecutive positions: one modulo for the block, a wrap at most once
    const int64_t cell0 = first % r.cap;
    auto cell_of = [&](int i) -> int64_t { const int64_t c = cell0 + i; return c >= r.cap ? c - r.cap : c; };
    // a warp per cell, lanes over the dims (HD / 32 each): the key row read in one coalesced pass
    constexpr int PER = HD / 32;
    float qf[GMAX][PER];
#pragma unroll
    for (int h = 0; h < GMAX; ++h)
#pragma unroll
        for (int i = 0; i < PER; ++i) qf[h][i] = h < G ? qs[h][lane * PER + i] : 0.0f;
    for (int c = warp; c < n; c += CHUNK / 32) {
        const int64_t row = cell_of(c) * n_kv + kvh;
        float kf[PER];
        if (q8.k) {
            const int8_t* cb = q8.k + row * HD + lane * PER;
            const float sc = __half2float(__ushort_as_half(q8.ks[row * (HD / 64) + lane * PER / 64]));
#pragma unroll
            for (int i = 0; i < PER; ++i) kf[i] = (float) cb[i] * sc;
        } else {
            const __half2* k2 = reinterpret_cast<const __half2*>(kc + row * HD + lane * PER);
#pragma unroll
            for (int i = 0; i < PER / 2; ++i) {
                const float2 f = __half22float2(k2[i]);
                kf[2 * i] = f.x;
                kf[2 * i + 1] = f.y;
            }
        }
#pragma unroll
        for (int h = 0; h < GMAX; ++h)
            if (h < G) {
                float d = 0.0f;
#pragma unroll
                for (int i = 0; i < PER; ++i) d = fmaf(qf[h][i], kf[i], d);
                d = warp_sum(d);
                if (lane == 0) s[c][h] = d;
            }
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
    for (int d = threadIdx.x; d < HD; d += CHUNK) {
        float acc[GMAX] = {};
        for (int i = 0; i < n; ++i) {
            const int64_t row = cell_of(i) * n_kv + kvh;
            const float v = q8.v ? (float) q8.v[row * HD + d] * __half2float(__ushort_as_half(q8.vs[row * (HD / 64) + d / 64]))
                                 : __half2float(vc[row * HD + d]);
#pragma unroll
            for (int h = 0; h < GMAX; ++h)
                if (h < G) acc[h] = fmaf(s[i][h], v, acc[h]);
        }
#pragma unroll
        for (int h = 0; h < GMAX; ++h)
            if (h < G) {
                float* o = out + (int64_t) h * n_chunks * (HD + 2);
                if (d == 0) { o[0] = mx[h]; o[1] = sm[h]; }
                o[2 + d] = acc[h];
            }
    }
}

// Block (query head, row t), one thread per dim: the row's chunks merged in order.
template <int HD>
__global__ void __launch_bounds__(HD) decode_merge_kernel(const float* __restrict__ part, int n_head, Range r,
                                                          int n_chunks, float* __restrict__ out) {
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const int used = (int) ((row_hi(r, t) + CHUNK - 1) / CHUNK) - r.base;
    const int skip = (int) (row_lo(r, t) / CHUNK) - r.base;
    const float* p = part + ((int64_t) t * n_head + h) * n_chunks * (HD + 2);
    float m = kNeg;
    for (int c = skip; c < used; ++c) m = fmaxf(m, p[(int64_t) c * (HD + 2)]);
    float l = 0.0f, a = 0.0f;
    for (int c = skip; c < used; ++c) {
        const float* pc = p + (int64_t) c * (HD + 2);
        const float w = exp2f(pc[0] - m);
        l = fmaf(pc[1], w, l);
        a = fmaf(pc[2 + d], w, a);
    }
    out[((int64_t) t * n_head + h) * HD + d] = a / l;
}

template <int HD>
void launch_window(const float* q, const uint16_t* kc, const uint16_t* vc, int n_head, int n_kv, const Range& r, int T,
                   float* scratch, int n_chunks, int used, float* out, cudaStream_t s, KvInt8 q8 = {}) {
    const float scale = 1.4426950408889634f / sqrtf((float) HD);
    decode_chunk_kernel<HD><<<dim3((unsigned) used, (unsigned) n_kv, (unsigned) T), CHUNK, 0, s>>>(
        q, (const __half*) kc, (const __half*) vc, n_head, n_kv, r, n_chunks, scale, scratch, q8);
    check("gqa decode attention");
    decode_merge_kernel<HD><<<dim3((unsigned) n_head, (unsigned) T), HD, 0, s>>>(scratch, n_head, r, n_chunks, out);
    check("gqa decode attention/merge");
}

// ---------------------------------------------------------------- prompt
constexpr int BM = 64, BN = 32, NW = 4, LDS = HD + 8;   // query tile, cell tile, warps, padded smem row (halves)

__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned) __cvta_generic_to_shared(p); }
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
__device__ __forceinline__ unsigned pack_h2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&h);
}
__device__ __forceinline__ void cp16(void* dst, const void* src, bool ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst)), "l"(src), "r"(ok ? 16 : 0));
}

struct PromptSmem {
    __half q[BM][LDS];
    __half k[BN][LDS];
    __half v[BN][LDS];
};

// Block (query tile, query head): warp w owns query rows 16 w .. 16 w + 15 of the tile.
__global__ void __launch_bounds__(NW * 32) prompt_kernel(const float* __restrict__ q, const __half* __restrict__ kc,
                                                         const __half* __restrict__ vc, int n_head, int n_kv,
                                                         int64_t pos0, int64_t T, float* __restrict__ out, KvInt8 q8) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    PromptSmem& sm = *reinterpret_cast<PromptSmem*>(smem_raw);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int head = blockIdx.y, kvh = head / (n_head / n_kv);
    const int64_t q0 = (int64_t) blockIdx.x * BM;
    // the query tile, fp16
    for (int i = tid; i < BM * HD / 4; i += NW * 32) {
        const int r = i / (HD / 4), c = (i % (HD / 4)) * 4;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (q0 + r < T) v = *reinterpret_cast<const float4*>(q + ((q0 + r) * n_head + head) * HD + c);
        *reinterpret_cast<__half2*>(&sm.q[r][c]) = __floats2half2_rn(v.x, v.y);
        *reinterpret_cast<__half2*>(&sm.q[r][c + 2]) = __floats2half2_rn(v.z, v.w);
    }
    const int g = lane >> 2, tq = lane & 3;
    const int64_t row_a = q0 + warp * 16 + g, row_b = row_a + 8;   // this thread's two query rows (chunk index)
    float o[HD / 8][4];
#pragma unroll
    for (int n = 0; n < HD / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    float m_a = kNeg, m_b = kNeg, l_a = 0.0f, l_b = 0.0f;
    const int64_t last_row = min(T, q0 + BM) - 1;
    const int64_t cells = pos0 + last_row + 1;
    for (int64_t c0 = 0; c0 < cells; c0 += BN) {
        __syncthreads();   // the previous tile is consumed (and the query tile is written)
        for (int i = tid; i < BN * HD / 8; i += NW * 32) {
            const int r = i / (HD / 8), c = (i % (HD / 8)) * 8;
            const bool ok = c0 + r < cells;
            const int64_t row = (ok ? c0 + r : 0) * n_kv + kvh;
            if (q8.k) {   // eight codes times their group's scale, rounded to the fp16 tile
                const uint2 kw = *reinterpret_cast<const uint2*>(q8.k + row * HD + c);
                const uint2 vw = *reinterpret_cast<const uint2*>(q8.v + row * HD + c);
                const float ks = __half2float(__ushort_as_half(q8.ks[row * (HD / 64) + c / 64]));
                const float vs = __half2float(__ushort_as_half(q8.vs[row * (HD / 64) + c / 64]));
                const int8_t* kb = reinterpret_cast<const int8_t*>(&kw);
                const int8_t* vb = reinterpret_cast<const int8_t*>(&vw);
#pragma unroll
                for (int j = 0; j < 8; j += 2) {
                    *reinterpret_cast<__half2*>(&sm.k[r][c + j]) = ok ? __floats2half2_rn(kb[j] * ks, kb[j + 1] * ks) : __floats2half2_rn(0.f, 0.f);
                    *reinterpret_cast<__half2*>(&sm.v[r][c + j]) = ok ? __floats2half2_rn(vb[j] * vs, vb[j + 1] * vs) : __floats2half2_rn(0.f, 0.f);
                }
            } else {
                cp16(&sm.k[r][c], kc + row * HD + c, ok);
                cp16(&sm.v[r][c], vc + row * HD + c, ok);
            }
        }
        asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();
        // S = Q K^T for the warp's 16 rows and the tile's 32 cells (four 8-cell tiles)
        float s[BN / 8][4];
#pragma unroll
        for (int n = 0; n < BN / 8; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
#pragma unroll
        for (int k = 0; k < HD; k += 16) {
            unsigned a0, a1, a2, a3;
            ldsm_x4(a0, a1, a2, a3, &sm.q[warp * 16 + (lane & 15)][k + (lane >> 4) * 8]);
#pragma unroll
            for (int n = 0; n < BN / 8; n += 2) {
                unsigned b0, b1, b2, b3;
                // rows: cells n*8 .. n*8+15; the x4's matrices are (cells 0-7, dims k..k+7), (0-7, k+8..), (8-15, k..), (8-15, k+8..)
                ldsm_x4(b0, b1, b2, b3, &sm.k[n * 8 + (lane & 7) + ((lane >> 4) << 3)][k + ((lane >> 3) & 1) * 8]);
                mma16816(s[n], a0, a1, a2, a3, b0, b1);
                mma16816(s[n + 1], a0, a1, a2, a3, b2, b3);
            }
        }
        // mask and online softmax: thread holds rows row_a (s[n][0..1]) and row_b (s[n][2..3])
        float mx_a = m_a, mx_b = m_b;
#pragma unroll
        for (int n = 0; n < BN / 8; ++n)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int64_t cell = c0 + n * 8 + tq * 2 + (i & 1);
                const int64_t row = i < 2 ? row_a : row_b;
                const bool vis = cell <= pos0 + row && cell < cells;
                s[n][i] = vis ? s[n][i] * kScaleLog2 : kNeg;
                if (i < 2) mx_a = fmaxf(mx_a, s[n][i]);
                else mx_b = fmaxf(mx_b, s[n][i]);
            }
        mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffffu, mx_a, 1));
        mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffffu, mx_a, 2));
        mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffffu, mx_b, 1));
        mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffffu, mx_b, 2));
        const float sc_a = exp2f(m_a - mx_a), sc_b = exp2f(m_b - mx_b);
        m_a = mx_a;
        m_b = mx_b;
        float ps_a = 0.0f, ps_b = 0.0f;
#pragma unroll
        for (int n = 0; n < BN / 8; ++n) {
            s[n][0] = s[n][0] <= kNeg ? 0.0f : exp2f(s[n][0] - m_a);
            s[n][1] = s[n][1] <= kNeg ? 0.0f : exp2f(s[n][1] - m_a);
            s[n][2] = s[n][2] <= kNeg ? 0.0f : exp2f(s[n][2] - m_b);
            s[n][3] = s[n][3] <= kNeg ? 0.0f : exp2f(s[n][3] - m_b);
            ps_a += s[n][0] + s[n][1];
            ps_b += s[n][2] + s[n][3];
        }
        l_a = l_a * sc_a + ps_a;
        l_b = l_b * sc_b + ps_b;
#pragma unroll
        for (int n = 0; n < HD / 8; ++n) {
            o[n][0] *= sc_a; o[n][1] *= sc_a;
            o[n][2] *= sc_b; o[n][3] *= sc_b;
        }
        // O += P V: P's C fragments of cell tiles 2j and 2j+1 are the A fragment of k-step j
#pragma unroll
        for (int j = 0; j < BN / 16; ++j) {
            const unsigned a0 = pack_h2(s[2 * j][0], s[2 * j][1]);
            const unsigned a1 = pack_h2(s[2 * j][2], s[2 * j][3]);
            const unsigned a2 = pack_h2(s[2 * j + 1][0], s[2 * j + 1][1]);
            const unsigned a3 = pack_h2(s[2 * j + 1][2], s[2 * j + 1][3]);
#pragma unroll
            for (int n = 0; n < HD / 8; n += 2) {
                unsigned b0, b1, b2, b3;
                // V rows are cells (k), columns dims (n): transposed loads give the col-major B of dims n*8.., n*8+8..
                ldsm_x4_t(b0, b1, b2, b3, &sm.v[j * 16 + (lane & 15)][n * 8 + (lane >> 4) * 8]);
                mma16816(o[n], a0, a1, a2, a3, b0, b1);
                mma16816(o[n + 1], a0, a1, a2, a3, b2, b3);
            }
        }
    }
    l_a += __shfl_xor_sync(0xffffffffu, l_a, 1);
    l_a += __shfl_xor_sync(0xffffffffu, l_a, 2);
    l_b += __shfl_xor_sync(0xffffffffu, l_b, 1);
    l_b += __shfl_xor_sync(0xffffffffu, l_b, 2);
    const float ia = l_a > 0.0f ? 1.0f / l_a : 0.0f, ib = l_b > 0.0f ? 1.0f / l_b : 0.0f;
#pragma unroll
    for (int n = 0; n < HD / 8; ++n) {
        const int col = n * 8 + tq * 2;
        if (row_a < T) *reinterpret_cast<float2*>(out + (row_a * n_head + head) * HD + col) = make_float2(o[n][0] * ia, o[n][1] * ia);
        if (row_b < T) *reinterpret_cast<float2*>(out + (row_b * n_head + head) * HD + col) = make_float2(o[n][2] * ib, o[n][3] * ib);
    }
}

}  // namespace

size_t gqa_decode_scratch_floats(int64_t max_cells, int T, int n_head, int head_dim) {
    const int64_t chunks = (max_cells + CHUNK - 1) / CHUNK + 1;
    return (size_t) T * n_head * chunks * (head_dim + 2);
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
    const int used = pos_dev ? n_chunks : (int) ((pos0 + T + CHUNK - 1) / CHUNK);
    launch_window<256>(q, k_cache, v_cache, n_head, n_kv, r, T, scratch, n_chunks, used, out, (cudaStream_t) stream, q8);
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
    if (head_dim == 128) launch_window<128>(q, k_cache, v_cache, n_head, n_kv, r, T, scratch, n_chunks, used, out, (cudaStream_t) stream);
    else launch_window<256>(q, k_cache, v_cache, n_head, n_kv, r, T, scratch, n_chunks, used, out, (cudaStream_t) stream);
}

void gqa_prompt_attn(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, int n_head, int n_kv,
                     int64_t pos0, int64_t T, float* out, void* stream, KvInt8 q8) {
    if (T <= 0) return;
    if (n_kv < 1 || n_head % n_kv) {
        basalt::log::error("basalt", "gqa_prompt_attn: invalid arguments");
        std::exit(1);
    }
    static bool attr = false;
    if (!attr) {
        if (cudaFuncSetAttribute(prompt_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(PromptSmem)) !=
            cudaSuccess) {
            basalt::log::error("basalt", "gqa_prompt_attn: shared-memory opt-in failed");
            std::exit(1);
        }
        attr = true;
    }
    prompt_kernel<<<dim3((unsigned) ((T + BM - 1) / BM), (unsigned) n_head), NW * 32, sizeof(PromptSmem),
                    (cudaStream_t) stream>>>(q, (const __half*) k_cache, (const __half*) v_cache, n_head, n_kv, pos0, T, out, q8);
    check("gqa_prompt_attn");
}

}  // namespace basalt::kernels
