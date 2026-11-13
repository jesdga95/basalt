// src/kernels/cuda/gqa_attn.cu - see include/basalt/kernels/gqa_attn.hpp.
#include "basalt/kernels/gqa_attn.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include "basalt/log.hpp"

namespace basalt::kernels {
namespace {

constexpr int HD = 256;
constexpr int GMAX = 8;               // query heads per KV head
constexpr int CHUNK = 256;            // cells per decode block
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
// Block (chunk, KV head, row t): the scores of the chunk's visible cells for the G query heads (warp per cell, lanes
// over the dims), the chunk's max and sum per head (one thread per head, cells in order), then p . V with one
// thread per dim (cells in order).  Writes the chunk's partial {max, sum, acc[256]} per query head.
__global__ void __launch_bounds__(HD) decode_chunk_kernel(const float* __restrict__ q, const __half* __restrict__ kc,
                                                          const __half* __restrict__ vc, int n_head, int n_kv,
                                                          int64_t pos0, int n_chunks, float* __restrict__ part) {
    __shared__ float s[CHUNK][GMAX];
    __shared__ float mx[GMAX], sm[GMAX];
    const int chunk = blockIdx.x, kvh = blockIdx.y, t = blockIdx.z;
    const int G = n_head / n_kv;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int64_t visible = pos0 + t + 1;
    const int64_t c0 = (int64_t) chunk * CHUNK;
    const int n = (int) min((int64_t) CHUNK, visible - c0);   // may be <= 0
    float* out = part + (((int64_t) t * n_head + (int64_t) kvh * G) * n_chunks + chunk) * (HD + 2);
    if (n <= 0) {
        for (int h = 0; h < G; ++h) {
            float* o = out + (int64_t) h * n_chunks * (HD + 2);
            if (threadIdx.x == 0) { o[0] = kNeg; o[1] = 0.0f; }
            o[2 + threadIdx.x] = 0.0f;
        }
        return;
    }
    float qf[GMAX][8];
#pragma unroll
    for (int h = 0; h < GMAX; ++h)
        if (h < G) {
            const float4* qp = reinterpret_cast<const float4*>(q + ((int64_t) t * n_head + kvh * G + h) * HD + lane * 8);
            const float4 a = qp[0], b = qp[1];
            qf[h][0] = a.x; qf[h][1] = a.y; qf[h][2] = a.z; qf[h][3] = a.w;
            qf[h][4] = b.x; qf[h][5] = b.y; qf[h][6] = b.z; qf[h][7] = b.w;
        }
    for (int c = warp; c < n; c += HD / 32) {
        const uint4 kv = *reinterpret_cast<const uint4*>(kc + ((c0 + c) * n_kv + kvh) * HD + lane * 8);
        const __half2* k2 = reinterpret_cast<const __half2*>(&kv);
        float kf[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __half22float2(k2[i]);
            kf[2 * i] = f.x;
            kf[2 * i + 1] = f.y;
        }
#pragma unroll
        for (int h = 0; h < GMAX; ++h)
            if (h < G) {
                float d = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) d = fmaf(qf[h][i], kf[i], d);
                d = warp_sum(d);
                if (lane == 0) s[c][h] = d * kScaleLog2;
            }
    }
    __syncthreads();
    if (threadIdx.x < G) {
        const int h = threadIdx.x;
        float m = kNeg;
        for (int c = 0; c < n; ++c) m = fmaxf(m, s[c][h]);
        float l = 0.0f;
        for (int c = 0; c < n; ++c) {
            const float p = exp2f(s[c][h] - m);
            s[c][h] = p;
            l += p;
        }
        mx[h] = m;
        sm[h] = l;
    }
    __syncthreads();
    float acc[GMAX] = {};
    const int d = threadIdx.x;
    for (int c = 0; c < n; ++c) {
        const float v = __half2float(vc[((c0 + c) * n_kv + kvh) * HD + d]);
#pragma unroll
        for (int h = 0; h < GMAX; ++h)
            if (h < G) acc[h] = fmaf(s[c][h], v, acc[h]);
    }
#pragma unroll
    for (int h = 0; h < GMAX; ++h)
        if (h < G) {
            float* o = out + (int64_t) h * n_chunks * (HD + 2);
            if (d == 0) { o[0] = mx[h]; o[1] = sm[h]; }
            o[2 + d] = acc[h];
        }
}

// Block (query head, row t), one thread per dim: the chunks merged in order.
__global__ void __launch_bounds__(HD) decode_merge_kernel(const float* __restrict__ part, int n_head, int64_t pos0,
                                                          int n_chunks, float* __restrict__ out) {
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const int used = (int) ((pos0 + t + 1 + CHUNK - 1) / CHUNK);
    const float* p = part + ((int64_t) t * n_head + h) * n_chunks * (HD + 2);
    float m = kNeg;
    for (int c = 0; c < used; ++c) m = fmaxf(m, p[(int64_t) c * (HD + 2)]);
    float l = 0.0f, a = 0.0f;
    for (int c = 0; c < used; ++c) {
        const float* pc = p + (int64_t) c * (HD + 2);
        const float w = exp2f(pc[0] - m);
        l = fmaf(pc[1], w, l);
        a = fmaf(pc[2 + d], w, a);
    }
    out[((int64_t) t * n_head + h) * HD + d] = a / l;
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
                                                         int64_t pos0, int64_t T, float* __restrict__ out) {
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
            const int64_t at = ((ok ? c0 + r : 0) * n_kv + kvh) * HD + c;
            cp16(&sm.k[r][c], kc + at, ok);
            cp16(&sm.v[r][c], vc + at, ok);
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

size_t gqa_decode_scratch_floats(int64_t max_cells, int T, int n_head) {
    const int64_t chunks = (max_cells + CHUNK - 1) / CHUNK;
    return (size_t) T * n_head * chunks * (HD + 2);
}

void gqa_decode_attn(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, int n_head, int n_kv,
                     int64_t pos0, int T, float* scratch, int64_t max_cells, float* out, void* stream) {
    if (T < 1 || T > kGqaMaxWindow || n_kv < 1 || n_head % n_kv || n_head / n_kv > GMAX || pos0 + T > max_cells) {
        basalt::log::error("basalt", "gqa_decode_attn: invalid arguments");
        std::exit(1);
    }
    const int n_chunks = (int) ((max_cells + CHUNK - 1) / CHUNK);
    const int used = (int) ((pos0 + T + CHUNK - 1) / CHUNK);
    const cudaStream_t s = (cudaStream_t) stream;
    decode_chunk_kernel<<<dim3((unsigned) used, (unsigned) n_kv, (unsigned) T), HD, 0, s>>>(
        q, (const __half*) k_cache, (const __half*) v_cache, n_head, n_kv, pos0, n_chunks, scratch);
    check("gqa_decode_attn");
    decode_merge_kernel<<<dim3((unsigned) n_head, (unsigned) T), HD, 0, s>>>(scratch, n_head, pos0, n_chunks, out);
    check("gqa_decode_attn/merge");
}

void gqa_prompt_attn(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, int n_head, int n_kv,
                     int64_t pos0, int64_t T, float* out, void* stream) {
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
                    (cudaStream_t) stream>>>(q, (const __half*) k_cache, (const __half*) v_cache, n_head, n_kv, pos0, T, out);
    check("gqa_prompt_attn");
}

}  // namespace basalt::kernels
