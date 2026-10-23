// src/prefill/gdn_chunked.cu - the prompt path's gated delta rule (gdn_recurrence in
// include/basalt/prefill/kernels.hpp), in chunks of 16 tokens on the tensor cores.
// Per chunk and value head, gdn_prepare_kernel builds the L2-normalized BF16 q/k packet, the within-chunk solve
// (I + L)^-1 diag(beta), the decayed q.k products and the cumulative decays; gdn_chunk_kernel then carries the FP32
// state across the chunks: prediction K S and history Q S (TF32 / BF16 MMA), the residual through the solve, the
// output, and S = decay S + K^T diag(suffix) Delta.
// Adapted from Infernix (Ian Ranson and contributors), src/ops/linear_attention/gated_delta_net/chunked/,
// Apache-2.0: see THIRD_PARTY_LICENSES.
#include "basalt/prefill/kernels.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "basalt/log.hpp"

namespace basalt::prefill {
namespace {

constexpr int S = 128, HK = 16, HV = 48, C = 10240;
constexpr int CK = 16;     // tokens per chunk
constexpr int DV = 64;     // value columns per recurrence block
constexpr int NTH = 256;   // threads of both kernels
constexpr float kLog2E = 1.4426950408889634f;

// q/k of one key head and chunk; shared by the key head's three value heads
struct alignas(256) QkChunk {
    __nv_bfloat16 q[CK * S];
    __nv_bfloat16 k[CK * S];
};
// With G_i = sum_{t<=i} g_t and L_ij = beta_i exp(G_i - G_j) k_i.k_j (j < i):
// solve = (I + L)^-1 diag(beta), mqk_ij = exp(G_i - G_j) q_i.k_j (j <= i).
struct alignas(256) Control {
    float solve[CK * CK];
    float mqk[CK * CK];
    float prefix[CK];   // exp(G_i)
    float suffix[CK];   // exp(G_last - G_i)
    uint32_t pad[32];
};
static_assert(sizeof(QkChunk) == 8192 && sizeof(Control) == 2304, "packet sizes");

__host__ __device__ constexpr int vector_index(int row, int col) { return row * S + (col ^ (row * 8)); }
__host__ __device__ constexpr int square_index(int row, int col) { return row * CK + (col ^ (((row >> 1) & 3) * 4)); }
__device__ __forceinline__ int value_index(int row, int col) { return row * DV + (col ^ ((row & 3) * 8)); }

__device__ __forceinline__ unsigned smem_addr(const void* p) { return (unsigned) __cvta_generic_to_shared(p); }
__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
    uint32_t out;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(out) : "r"(__float_as_uint(hi)), "r"(__float_as_uint(lo)));
    return out;
}
__device__ __forceinline__ void ldsm_x2(unsigned& r0, unsigned& r1, unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void mma_bf16(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_tf32(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp16(void* dst, const void* src, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_addr(dst)), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ float sum16(float v) {
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o, 16);
    return v;
}

struct PrepareShared {
    QkChunk qk;
    Control control;
    float prefix[CK];
    float beta[CK];
    float lower[CK * CK];
};

// Block (chunk, value head): row = token in the chunk, 16 threads per row with 8 coordinates each.
__device__ __forceinline__ void gdn_prepare(int bid, const float* __restrict__ h, const float* __restrict__ gate,
                                            const float* __restrict__ beta, float eps, int T, QkChunk* __restrict__ qk_out,
                                            Control* __restrict__ ctl_out) {
    __shared__ PrepareShared sm;
    const int tid = threadIdx.x, lane = tid & 31;
    const int head = bid % HV, chunk = bid / HV, qh = head % HK;
    const int start = chunk * CK, length = min(CK, T - start);
    const int row = tid / 16, col = (tid & 15) * 8;
    float q[8] = {}, k[8] = {};
    if (row < length) {
        const float* hq = h + (int64_t) (start + row) * C + qh * S + col;
        const float4 q0 = *reinterpret_cast<const float4*>(hq), q1 = *reinterpret_cast<const float4*>(hq + 4);
        const float4 k0 = *reinterpret_cast<const float4*>(hq + HK * S), k1 = *reinterpret_cast<const float4*>(hq + HK * S + 4);
        q[0] = q0.x; q[1] = q0.y; q[2] = q0.z; q[3] = q0.w; q[4] = q1.x; q[5] = q1.y; q[6] = q1.z; q[7] = q1.w;
        k[0] = k0.x; k[1] = k0.y; k[2] = k0.z; k[3] = k0.w; k[4] = k1.x; k[5] = k1.y; k[6] = k1.z; k[7] = k1.w;
    }
    float qs = 0.0f, ks = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) { qs = fmaf(q[i], q[i], qs); ks = fmaf(k[i], k[i], ks); }
    const float qi = rsqrtf(sum16(qs) + eps), ki = rsqrtf(sum16(ks) + eps);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        sm.qk.q[vector_index(row, col + i)] = __float2bfloat16_rn(q[i] * qi);
        sm.qk.k[vector_index(row, col + i)] = __float2bfloat16_rn(k[i] * ki);
    }
    if (tid < 32) sm.control.pad[tid] = 0;
    if (tid < CK) {
        const int64_t off = (int64_t) (start + tid) * HV + head;
        float g = tid < length ? gate[off] : 0.0f;
        sm.beta[tid] = tid < length ? beta[off] : 0.0f;
#pragma unroll
        for (int o = 1; o < CK; o <<= 1) {
            const float prev = __shfl_up_sync(0xffffu, g, o, CK);
            if (tid >= o) g += prev;
        }
        sm.prefix[tid] = g;
        sm.control.prefix[tid] = ex2(g * kLog2E);
        const float last = __shfl_sync(0xffffu, g, CK - 1, CK);
        sm.control.suffix[tid] = ex2((last - g) * kLog2E);
    }
    __syncthreads();
    // K K^T (warp 0) and Q K^T (warp 1) from the BF16 q/k the recurrence reads
    if (tid < 64) {
        const __nv_bfloat16* a = tid < 32 ? sm.qk.k : sm.qk.q;
        float acc[2][4] = {};
#pragma unroll
        for (int off = 0; off < S; off += 16) {
            unsigned af[4];
            ldsm_x4(af[0], af[1], af[2], af[3],
                    smem_addr(a + vector_index((lane & 7) + ((lane >> 3) & 1) * 8, off + (lane >> 4) * 8)));
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                unsigned b[2];
                ldsm_x2(b[0], b[1], smem_addr(sm.qk.k + vector_index(n * 8 + (lane & 7), off + ((lane >> 3) & 1) * 8)));
                mma_bf16(acc[n], af[0], af[1], af[2], af[3], b[0], b[1]);
            }
        }
#pragma unroll
        for (int n = 0; n < 2; ++n)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r = lane / 4 + (i / 2) * 8, c = n * 8 + (lane & 3) * 2 + (i & 1);
                // no exponential above the diagonal: the gate sums can be large negatives
                const float v = r >= c ? acc[n][i] * ex2((sm.prefix[r] - sm.prefix[c]) * kLog2E) : 0.0f;
                if (tid < 32) sm.lower[r * CK + c] = r > c ? sm.beta[r] * v : 0.0f;
                else sm.control.mqk[square_index(r, c)] = v;
            }
    }
    __syncthreads();
    // forward substitution for (I + L)^-1, one column per lane
    if (tid < 32) {
        const int c = lane & 15;
#pragma unroll
        for (int r = 0; r < CK; ++r) {
            float v = r == c ? 1.0f : 0.0f;
#pragma unroll
            for (int j = 0; j < r; ++j) v = fmaf(-sm.lower[r * CK + j], sm.control.solve[square_index(j, c)], v);
            if (lane < 16) sm.control.solve[square_index(r, c)] = v;
            __syncwarp();
        }
    }
    __syncthreads();
    sm.control.solve[square_index(row, tid & 15)] *= sm.beta[tid & 15];
    __syncthreads();
    const int chunks = (T + CK - 1) / CK;
    uint4* dst = reinterpret_cast<uint4*>(ctl_out + (int64_t) head * chunks + chunk);
    const uint4* src = reinterpret_cast<const uint4*>(&sm.control);
    if (tid < (int) (sizeof(Control) / 16)) dst[tid] = src[tid];
    // the key head's first value head publishes q/k; the recurrence kernel launches after this one
    if (head == qh) {
        uint4* qd = reinterpret_cast<uint4*>(qk_out + (int64_t) qh * chunks + chunk);
        const uint4* qs4 = reinterpret_cast<const uint4*>(&sm.qk);
#pragma unroll
        for (int i = tid; i < (int) (sizeof(QkChunk) / 16); i += NTH) qd[i] = qs4[i];
    }
}
__global__ void __launch_bounds__(NTH, 2) gdn_prepare_kernel(const float* __restrict__ h, const float* __restrict__ gate,
                                                             const float* __restrict__ beta, float eps, int T,
                                                             QkChunk* __restrict__ qk_out, Control* __restrict__ ctl_out) {
    gdn_prepare(blockIdx.x, h, gate, beta, eps, T, qk_out, ctl_out);
}

struct alignas(256) Stage {
    QkChunk qk;
    Control control;
    float value[CK * DV];
};
struct ChunkShared {
    Stage stage[2];
    float snapshot[S * DV];   // the state at the chunk's start, [key][value]
    float delta[CK * DV];
};

__device__ __forceinline__ void load_stage(Stage& st, const float* __restrict__ h, const QkChunk* __restrict__ qk,
                                           const Control* __restrict__ ctl, int head, int vstart, int chunk, int chunks,
                                           int T, int tid) {
    const uint4* qs = reinterpret_cast<const uint4*>(qk + (int64_t) (head % HK) * chunks + chunk);
    uint4* qd = reinterpret_cast<uint4*>(&st.qk);
#pragma unroll
    for (int i = tid; i < (int) (sizeof(QkChunk) / 16); i += NTH) cp16(qd + i, qs + i, 16);
    const uint4* cs = reinterpret_cast<const uint4*>(ctl + (int64_t) head * chunks + chunk);
    uint4* cd = reinterpret_cast<uint4*>(&st.control);
    for (int i = tid; i < (int) (sizeof(Control) / 16); i += NTH) cp16(cd + i, cs + i, 16);
    // CK x DV values, four per thread
    const int row = tid / (DV / 4), col = (tid % (DV / 4)) * 4;
    const int t = chunk * CK + row;
    cp16(st.value + row * DV + col, h + (int64_t) (t < T ? t : 0) * C + 2 * HK * S + head * S + vstart + col,
         t < T ? 16 : 0);
    cp_commit();
}

// A operand of the TF32 MMA from the BF16 rows (exact widening)
__device__ __forceinline__ void load_tf32_a(const __nv_bfloat16* p, int k, int lane, unsigned (&a)[4]) {
    unsigned lo, hi;
    ldsm_x2(lo, hi, smem_addr(p + vector_index((lane & 7) + ((lane >> 3) & 1) * 8, k)));
    a[0] = lo << 16; a[2] = lo & 0xffff0000u;
    a[1] = hi << 16; a[3] = hi & 0xffff0000u;
}
__device__ __forceinline__ void load_bf16_a(const __nv_bfloat16* p, int k, int lane, unsigned (&a)[4]) {
    ldsm_x4(a[0], a[1], a[2], a[3], smem_addr(p + vector_index((lane & 7) + ((lane >> 3) & 1) * 8, k + (lane >> 4) * 8)));
}
__device__ __forceinline__ void load_bf16_at(const __nv_bfloat16* p, int key, int lane, unsigned (&a)[4]) {
    ldsm_x4_t(a[0], a[1], a[2], a[3], smem_addr(p + vector_index((lane & 7) + (lane >> 4) * 8, key + ((lane >> 3) & 1) * 8)));
}
// acc += M[16 x 16] (square layout) x delta[16 x 8 columns of tile n]
__device__ __forceinline__ void matrix_delta(float (&acc)[4], const float* m, const float* delta, int n, int lane) {
    const int g = lane / 4, t = lane & 3;
#pragma unroll
    for (int k = 0; k < CK; k += 8) {
        unsigned af[4];
        ldsm_x4(af[0], af[1], af[2], af[3], smem_addr(m + square_index((lane & 7) + ((lane >> 3) & 1) * 8, k + (lane >> 4) * 4)));
        mma_tf32(acc, af[0], af[1], af[2], af[3], __float_as_uint(delta[value_index(k + t, n * 8 + g)]),
                 __float_as_uint(delta[value_index(k + t + 4, n * 8 + g)]));
    }
}

// Block (value head, DV-column stripe); warp w owns value columns 8 w .. 8 w + 7 of the stripe, all 128 key rows.
__device__ __forceinline__ void gdn_chunk(int bid, float* __restrict__ state, const float* __restrict__ h,
                                          const QkChunk* __restrict__ qk, const Control* __restrict__ ctl,
                                          float* __restrict__ y, int T, ChunkShared& sm) {
    constexpr int MT = S / 16;   // key tiles
    const int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    const int g = lane / 4, t = lane & 3;
    const int head = bid / (S / DV), vstart = (bid % (S / DV)) * DV;
    const int chunks = (T + CK - 1) / CK;
    const size_t rs = (size_t) HV * S;   // state row stride: state[key][value head][value]
    const float scale = rsqrtf((float) S);
    float st[MT][4];   // S[key][value] tiles as MMA accumulators
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const int key = m * 16 + g, v = head * S + vstart + warp * 8 + t * 2;
        const float2 a = *reinterpret_cast<const float2*>(state + key * rs + v);
        const float2 b = *reinterpret_cast<const float2*>(state + (key + 8) * rs + v);
        st[m][0] = a.x; st[m][1] = a.y; st[m][2] = b.x; st[m][3] = b.y;
    }
    load_stage(sm.stage[0], h, qk, ctl, head, vstart, 0, chunks, T, tid);
    if (chunks > 1) load_stage(sm.stage[1], h, qk, ctl, head, vstart, 1, chunks, T, tid);
    for (int chunk = 0; chunk < chunks; ++chunk) {
        if (chunk + 1 < chunks) cp_wait<1>();
        else cp_wait<0>();
        const int col = warp * 8 + t * 2;
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const int key = m * 16 + g;
            *reinterpret_cast<float2*>(sm.snapshot + value_index(key, col)) = make_float2(st[m][0], st[m][1]);
            *reinterpret_cast<float2*>(sm.snapshot + value_index(key + 8, col)) = make_float2(st[m][2], st[m][3]);
        }
        __syncthreads();
        const Stage& sg = sm.stage[chunk & 1];
        const Control& p = sg.control;
        float hist[4] = {}, pred[4] = {};
        // pred = K S (TF32), hist = Q S (BF16)
#pragma unroll
        for (int k = 0; k < S; k += 16) {
            unsigned ka0[4], ka1[4], qa[4];
            load_tf32_a(sg.qk.k, k, lane, ka0);
            load_tf32_a(sg.qk.k, k + 8, lane, ka1);
            load_bf16_a(sg.qk.q, k, lane, qa);
            const int n = warp;
            const float b0 = sm.snapshot[value_index(k + 2 * t, n * 8 + g)];
            const float b1 = sm.snapshot[value_index(k + 2 * t + 1, n * 8 + g)];
            const float b2 = sm.snapshot[value_index(k + 2 * t + 8, n * 8 + g)];
            const float b3 = sm.snapshot[value_index(k + 2 * t + 9, n * 8 + g)];
            mma_tf32(pred, ka0[0], ka0[1], ka0[2], ka0[3], __float_as_uint(b0), __float_as_uint(b1));
            mma_tf32(pred, ka1[0], ka1[1], ka1[2], ka1[3], __float_as_uint(b2), __float_as_uint(b3));
            mma_bf16(hist, qa[0], qa[1], qa[2], qa[3], pack_bf16x2(b0, b1), pack_bf16x2(b2, b3));
        }
        {   // residual R = V - diag(prefix) K S
            const int r = lane / 4, c = warp * 8 + (lane & 3) * 2;
            const float2 v0 = *reinterpret_cast<const float2*>(sg.value + r * DV + c);
            const float2 v1 = *reinterpret_cast<const float2*>(sg.value + (r + 8) * DV + c);
            *reinterpret_cast<float2*>(sm.delta + value_index(r, c)) =
                make_float2(v0.x - p.prefix[r] * pred[0], v0.y - p.prefix[r] * pred[1]);
            *reinterpret_cast<float2*>(sm.delta + value_index(r + 8, c)) =
                make_float2(v1.x - p.prefix[r + 8] * pred[2], v1.y - p.prefix[r + 8] * pred[3]);
        }
        __syncthreads();
        {   // Delta = solve R, in place (each warp reads only its own columns)
            float d[4] = {};
            matrix_delta(d, p.solve, sm.delta, warp, lane);
            __syncwarp();
            *reinterpret_cast<float2*>(sm.delta + value_index(g, col)) = make_float2(d[0], d[1]);
            *reinterpret_cast<float2*>(sm.delta + value_index(g + 8, col)) = make_float2(d[2], d[3]);
        }
        __syncthreads();
        {   // out = diag(prefix) Q S + mqk Delta
            hist[0] *= p.prefix[g]; hist[1] *= p.prefix[g];
            hist[2] *= p.prefix[g + 8]; hist[3] *= p.prefix[g + 8];
            matrix_delta(hist, p.mqk, sm.delta, warp, lane);
            const int t0 = chunk * CK + g;
            float* o = y + ((int64_t) t0 * HV + head) * S + vstart + col;
            if (t0 < T) *reinterpret_cast<float2*>(o) = make_float2(hist[0] * scale, hist[1] * scale);
            if (t0 + 8 < T)
                *reinterpret_cast<float2*>(o + (int64_t) 8 * HV * S) = make_float2(hist[2] * scale, hist[3] * scale);
        }
        // S = decay S + K^T diag(suffix) Delta; the state stays FP32, Delta goes to BF16 for this product only
        unsigned db[2];
        {
            const int c = warp * 8 + g;
            db[0] = pack_bf16x2(p.suffix[2 * t] * sm.delta[value_index(2 * t, c)],
                                p.suffix[2 * t + 1] * sm.delta[value_index(2 * t + 1, c)]);
            db[1] = pack_bf16x2(p.suffix[2 * t + 8] * sm.delta[value_index(2 * t + 8, c)],
                                p.suffix[2 * t + 9] * sm.delta[value_index(2 * t + 9, c)]);
        }
        const float decay = p.prefix[CK - 1];
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            st[m][0] *= decay; st[m][1] *= decay; st[m][2] *= decay; st[m][3] *= decay;
            unsigned af[4];
            load_bf16_at(sg.qk.k, m * 16, lane, af);
            mma_bf16(st[m], af[0], af[1], af[2], af[3], db[0], db[1]);
        }
        __syncthreads();
        if (chunk + 2 < chunks) load_stage(sm.stage[chunk & 1], h, qk, ctl, head, vstart, chunk + 2, chunks, T, tid);
    }
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const int key = m * 16 + g, v = head * S + vstart + warp * 8 + t * 2;
        *reinterpret_cast<float2*>(state + key * rs + v) = make_float2(st[m][0], st[m][1]);
        *reinterpret_cast<float2*>(state + (key + 8) * rs + v) = make_float2(st[m][2], st[m][3]);
    }
}
__global__ void __launch_bounds__(NTH, 2) gdn_chunk_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                           const QkChunk* __restrict__ qk, const Control* __restrict__ ctl,
                                                           float* __restrict__ y, int T) {
    extern __shared__ __align__(256) unsigned char backing[];
    gdn_chunk(blockIdx.x, state, h, qk, ctl, y, T, *reinterpret_cast<ChunkShared*>(backing));
}

// the output norm over a head's 128 columns, into the FP16 copy the out projection reads
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }
__global__ void __launch_bounds__(S) gdn_out_norm_kernel(const float* __restrict__ z, const float* __restrict__ gamma,
                                                         float eps, const float* __restrict__ y,
                                                         uint16_t* __restrict__ y16) {
    __shared__ float wsum[4];
    const int64_t t = blockIdx.x;
    const int head = blockIdx.y, col = threadIdx.x;
    const size_t at = (size_t) t * HV * S + (size_t) head * S + col;
    const float oc = y[at];
    float sp = oc * oc;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sp += __shfl_xor_sync(0xffffffffu, sp, o);
    if ((col & 31) == 0) wsum[col >> 5] = sp;
    __syncthreads();
    const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
    const float v = oc * rsqrtf(ss / (float) S + eps) * gamma[col] * sigm(z[t * HV * S + head * S + col]);
    y16[at] = __half_as_ushort(__float2half_rn(v));
}

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "%s: %s", what, cudaGetErrorString(e));
        std::abort();
    }
}

}  // namespace

uint64_t gdn_work_bytes(int64_t T) {
    const int64_t chunks = (T + CK - 1) / CK;
    return (uint64_t) chunks * (HK * sizeof(QkChunk) + HV * sizeof(Control));
}

void gdn_recurrence(float* state, const float* h, const float* gate, const float* beta, const float* z,
                    const float* gamma, float eps, void* work, float* y, uint16_t* y16, int64_t T, void* stream) {
    if (T > 0) {
        const cudaStream_t s = (cudaStream_t) stream;
        const int chunks = (int) ((T + CK - 1) / CK);
        QkChunk* qk = static_cast<QkChunk*>(work);
        Control* ctl = reinterpret_cast<Control*>(qk + (int64_t) HK * chunks);
        gdn_prepare_kernel<<<(unsigned) (HV * chunks), NTH, 0, s>>>(h, gate, beta, eps, (int) T, qk, ctl);
        // the shared-memory opt-in is per device (a layer split runs this on several, from different threads)
        static std::atomic<bool> attr[64];
        int dev = 0;
        cudaGetDevice(&dev);
        if (dev < 0 || dev >= 64) { basalt::log::error("basalt", "gdn_recurrence: device %d out of range", dev); std::abort(); }
        if (!attr[dev].load(std::memory_order_acquire)) {
            if (cudaFuncSetAttribute(gdn_chunk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int) sizeof(ChunkShared)) != cudaSuccess) {
                basalt::log::error("basalt", "gdn_recurrence: shared-memory opt-in failed");
                std::abort();
            }
            attr[dev].store(true, std::memory_order_release);
        }
        gdn_chunk_kernel<<<HV * (S / DV), NTH, sizeof(ChunkShared), s>>>(state, h, qk, ctl, y, (int) T);
        gdn_out_norm_kernel<<<dim3((unsigned) T, HV), S, 0, s>>>(z, gamma, eps, y, y16);
    }
    check("gdn_recurrence");
}

}  // namespace basalt::prefill
