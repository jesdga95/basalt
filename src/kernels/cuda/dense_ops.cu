// src/kernels/cuda/dense_ops.cu - see include/basalt/kernels/dense_ops.hpp.
#include "basalt/kernels/dense_ops.hpp"
#include "basalt/kernels/gqa_attn.hpp"
#include "basalt/kernels/pdl.hpp"

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include "basalt/log.hpp"

namespace basalt::kernels {
namespace {

constexpr int HD = 256;   // attention head width

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { basalt::log::error("basalt", "%s: %s", what, cudaGetErrorString(e)); std::exit(1); }
}

unsigned blocks_for(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// sum over the block (blockDim.x a multiple of 32, at most 1024); every thread gets the total
__device__ __forceinline__ float block_sum(float v, float* red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < nw; ++i) t += red[i];
    return t;
}

__global__ void rms_norm_rows_kernel(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out,
                                     int64_t cols, int64_t ld_out, float eps) {
    __shared__ float red[32];
    const float* xr = x + (int64_t) blockIdx.x * cols;
    float* yr = out + (int64_t) blockIdx.x * ld_out;
    float ss = 0.0f;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) ss = fmaf(xr[c], xr[c], ss);
    const float inv = rsqrtf(block_sum(ss, red) / (float) cols + eps);
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) yr[c] = xr[c] * inv * w[c];
}

// (the elementwise kernels below go through launch_pdl: they read earlier kernels' outputs after pdl_wait, so those
// pointers are not __restrict__)
__global__ void add_scaled_kernel(float* x, const float* y, const float* __restrict__ scale, int64_t n) {
    pdl_launch_dependents();
    pdl_wait();
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += scale ? y[i] * *scale : y[i];
}

__global__ void rms_norm_rows_bf16_kernel(const __nv_bfloat16* __restrict__ x, const float* __restrict__ w,
                                          float* __restrict__ out, int64_t cols, float eps) {
    __shared__ float red[32];
    const __nv_bfloat16* xr = x + (int64_t) blockIdx.x * cols;
    float* yr = out + (int64_t) blockIdx.x * cols;
    float ss = 0.0f;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
        const float v = __bfloat162float(xr[c]);
        ss = fmaf(v, v, ss);
    }
    const float inv = rsqrtf(block_sum(ss, red) / (float) cols + eps);
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) yr[c] = __bfloat162float(xr[c]) * inv * w[c];
}
__global__ void add_scaled_bf16_kernel(__nv_bfloat16* x, const float* y, const float* __restrict__ scale, int64_t n) {
    pdl_launch_dependents();
    pdl_wait();
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __float2bfloat16_rn(__bfloat162float(x[i]) + (scale ? y[i] * *scale : y[i]));
}
__global__ void to_bf16_kernel(const float* __restrict__ x, __nv_bfloat16* __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2bfloat16_rn(x[i]);
}
__global__ void to_f32_kernel(const __nv_bfloat16* __restrict__ x, float* __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __bfloat162float(x[i]);
}

__global__ void scale_values_kernel(float* __restrict__ y, const float* __restrict__ scale, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] *= *scale;
}

__global__ void swiglu_scaled_kernel(const float* g, const float* u, const float* __restrict__ sg,
                                     const float* __restrict__ su, float* h, int64_t rows, int64_t width, int64_t ld) {
    pdl_launch_dependents();
    pdl_wait();
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * width) return;
    const int64_t r = i / width, c = i % width;
    const float gv = g[r * ld + c] * (sg ? *sg : 1.0f), uv = u[r * ld + c] * (su ? *su : 1.0f);
    h[i] = gv / (1.0f + expf(-gv)) * uv;
}

// y[t][n] = sum_k x[t][k] w[n][k], x fp32, w BF16, on tf32 tensor cores (mma m16n8k8): a warp per 16 rows x 48 outputs.
// x is split into two tf32 terms (hi + lo: about fp32's precision; BF16 weights are exact in tf32), fp32 accumulation.  In each 8-value step thread c's k-slots c and
// c + 4 take values 2c and 2c + 1, so its loads are contiguous pairs (the sum is the same dot product).
constexpr int RD_T = 16, RD_N = 48;
__device__ __forceinline__ unsigned tf32_of(float f) {
    unsigned r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(f));
    return r;
}
__global__ void __launch_bounds__(32) bf16_rows_dot_kernel(const float* __restrict__ x, const uint16_t* __restrict__ w,
                                                           float* __restrict__ y, int64_t T, int64_t N, int64_t K) {
    const int lane = threadIdx.x, g = lane >> 2, c = lane & 3;
    const int64_t t0 = (int64_t) blockIdx.x * RD_T, n0 = (int64_t) blockIdx.y * RD_N;
    const int64_t ta = min(t0 + g, T - 1), tb = min(t0 + g + 8, T - 1);
    const float* xa = x + ta * K + 2 * c;
    const float* xb = x + tb * K + 2 * c;
    const uint16_t* wr[RD_N / 8];
#pragma unroll
    for (int j = 0; j < RD_N / 8; ++j) wr[j] = w + min(n0 + 8 * j + g, N - 1) * K + 2 * c;
    float acc[RD_N / 8][4] = {};
    for (int64_t k = 0; k < K; k += 8) {
        const float2 fa = *reinterpret_cast<const float2*>(xa + k), fb = *reinterpret_cast<const float2*>(xb + k);
        const unsigned a0 = tf32_of(fa.x), a1 = tf32_of(fb.x), a2 = tf32_of(fa.y), a3 = tf32_of(fb.y);
        const unsigned l0 = tf32_of(fa.x - __uint_as_float(a0)), l1 = tf32_of(fb.x - __uint_as_float(a1));
        const unsigned l2 = tf32_of(fa.y - __uint_as_float(a2)), l3 = tf32_of(fb.y - __uint_as_float(a3));
#pragma unroll
        for (int j = 0; j < RD_N / 8; ++j) {
            const unsigned wv = *reinterpret_cast<const unsigned*>(wr[j] + k);
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                         : "+f"(acc[j][0]), "+f"(acc[j][1]), "+f"(acc[j][2]), "+f"(acc[j][3])
                         : "r"(l0), "r"(l1), "r"(l2), "r"(l3), "r"(wv << 16), "r"(wv & 0xFFFF0000u));
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                         : "+f"(acc[j][0]), "+f"(acc[j][1]), "+f"(acc[j][2]), "+f"(acc[j][3])
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(wv << 16), "r"(wv & 0xFFFF0000u));
        }
    }
#pragma unroll
    for (int j = 0; j < RD_N / 8; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int64_t t = t0 + g + (i >> 1) * 8, n = n0 + 8 * j + 2 * c + (i & 1);
            if (t < T && n < N) y[t * N + n] = acc[j][i];
        }
}

// The GDN a/b projections of a prompt chunk with the gates fused: a and b are the first and last HV (= 48) rows of w
// [2 HV][K] (BF16), gate[t][h] = softplus(a + dt[h]) * ssm_a[h], beta[t][h] = sigmoid(b) - the expressions of gdn_gates.
// The product runs on BF16 tensor cores (m16n8k16): x is split into two BF16 terms (hi + lo, about 17 bits of x; the BF16
// weights are exact), fp32 accumulation.  A 2,048-row chunk is only 128 token tiles of 16, so K is split over a cluster
// of GP_S blocks (each a 64-token tile over all 96 rows and a K / GP_S slice); the partial sums meet through distributed
// shared memory, summed in rank order (deterministic), and each block applies the gates to its 8 tokens.  The four warps
// of a block read the same weights, so those come from L1.
// Adapted from Infernix (Ian Ranson and contributors), Apache-2.0: see THIRD_PARTY_LICENSES - its
// bf16_gdn_gating_proj_gemm_mma fuses the same gating into the a/b projection's epilogue over a split K.
constexpr int GP_AHEAD = 1;   // K / GP_S must be a multiple of 32 GP_AHEAD
constexpr int GP_HV = 48, GP_S = 8, GP_TT = 64, GP_LD = 2 * GP_HV + 8, GP_NF = 2 * GP_HV / 8;
// In: FP32 x (two BF16 terms) or BF16 x (exact: one term)
template <class In>
__global__ void __cluster_dims__(1, 1, GP_S) __launch_bounds__(128)
    gdn_gating_proj_kernel(const In* __restrict__ x, const uint16_t* __restrict__ w, const float* __restrict__ dt,
                           const float* __restrict__ ssm_a, float* __restrict__ gate, float* __restrict__ beta, int64_t T,
                           int64_t K) {
    namespace cg = cooperative_groups;
    __shared__ __align__(16) float part[GP_TT * GP_LD];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    const int rank = blockIdx.z;
    const int64_t tile0 = (int64_t) blockIdx.x * GP_TT, t0 = tile0 + warp * 16;
    const int64_t kslice = K / GP_S, kbeg = rank * kslice;
    const int64_t ta = min(t0 + g, T - 1), tb = min(t0 + g + 8, T - 1);
    // thread c reads 8 consecutive k per 32 (an mma step takes them as the pairs 0,1 and 2,3: the k order inside a step is
    // free), so the four lanes of a row read 128 contiguous bytes of x and 64 of w, one wide load each
    const In* xa = x + ta * K + kbeg + 8 * c;
    const In* xb = x + tb * K + kbeg + 8 * c;
    const uint16_t* wp = w + (int64_t) g * K + kbeg + 8 * c;   // row 8 j + g of the 96
    float acc[GP_NF][4] = {};
    float4 qx[GP_AHEAD][4];   // the loads run GP_AHEAD steps of 32 k ahead of the products (a register queue)
    uint4 qw[GP_AHEAD][GP_NF];
    auto fetch = [&](int d, int64_t k) {
        if constexpr (sizeof(In) == 4) {
            qx[d][0] = *reinterpret_cast<const float4*>(xa + k);
            qx[d][1] = *reinterpret_cast<const float4*>(xa + k + 4);
            qx[d][2] = *reinterpret_cast<const float4*>(xb + k);
            qx[d][3] = *reinterpret_cast<const float4*>(xb + k + 4);
        } else {   // the eight BF16 values as raw bits in the first two queue slots
            const uint4 ua = *reinterpret_cast<const uint4*>(xa + k), ub = *reinterpret_cast<const uint4*>(xb + k);
            qx[d][0] = *reinterpret_cast<const float4*>(&ua);
            qx[d][2] = *reinterpret_cast<const float4*>(&ub);
        }
#pragma unroll
        for (int j = 0; j < GP_NF; ++j) qw[d][j] = *reinterpret_cast<const uint4*>(wp + (int64_t) 8 * j * K + k);
    };
    auto bf2 = [](float lo, float hi) { return __floats2bfloat162_rn(lo, hi); };
    auto bits = [](__nv_bfloat162 v) { return *reinterpret_cast<unsigned*>(&v); };
#pragma unroll
    for (int d = 0; d < GP_AHEAD; ++d) fetch(d, 32 * d);
    for (int64_t k = 0; k < kslice; k += 32 * GP_AHEAD) {
#pragma unroll
        for (int d = 0; d < GP_AHEAD; ++d) {
            float4 fx[4];
            uint4 wv[GP_NF];
#pragma unroll
            for (int i = 0; i < 4; ++i) fx[i] = qx[d][i];
#pragma unroll
            for (int j = 0; j < GP_NF; ++j) wv[j] = qw[d][j];
            if (k + 32 * (GP_AHEAD + d) < kslice) fetch(d, k + 32 * (GP_AHEAD + d));
#pragma unroll
            for (int step = 0; step < 2; ++step) {   // x = hi + lo, both BF16 (about 17 bits of x), one m16n8k16 each
                if constexpr (sizeof(In) == 2) {   // BF16 x: the codes themselves, one term
                    const uint4 ua = *reinterpret_cast<const uint4*>(&fx[0]), ub = *reinterpret_cast<const uint4*>(&fx[2]);
                    const unsigned h0 = step ? ua.z : ua.x, h1 = step ? ub.z : ub.x, h2 = step ? ua.w : ua.y, h3 = step ? ub.w : ub.y;
#pragma unroll
                    for (int j = 0; j < GP_NF; ++j) {
                        const unsigned b0 = step ? wv[j].z : wv[j].x, b1 = step ? wv[j].w : wv[j].y;
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                     : "+f"(acc[j][0]), "+f"(acc[j][1]), "+f"(acc[j][2]), "+f"(acc[j][3])
                                     : "r"(h0), "r"(h1), "r"(h2), "r"(h3), "r"(b0), "r"(b1));
                    }
                    continue;
                }
                const float4 xa4 = fx[step], xb4 = fx[2 + step];
                const __nv_bfloat162 ha0 = bf2(xa4.x, xa4.y), ha1 = bf2(xa4.z, xa4.w);
                const __nv_bfloat162 hb0 = bf2(xb4.x, xb4.y), hb1 = bf2(xb4.z, xb4.w);
                const unsigned hi0 = bits(ha0), hi1 = bits(hb0), hi2 = bits(ha1), hi3 = bits(hb1);
                const unsigned lo0 = bits(bf2(xa4.x - __low2float(ha0), xa4.y - __high2float(ha0)));
                const unsigned lo1 = bits(bf2(xb4.x - __low2float(hb0), xb4.y - __high2float(hb0)));
                const unsigned lo2 = bits(bf2(xa4.z - __low2float(ha1), xa4.w - __high2float(ha1)));
                const unsigned lo3 = bits(bf2(xb4.z - __low2float(hb1), xb4.w - __high2float(hb1)));
#pragma unroll
                for (int j = 0; j < GP_NF; ++j) {
                    const unsigned b0 = step ? wv[j].z : wv[j].x, b1 = step ? wv[j].w : wv[j].y;
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                 : "+f"(acc[j][0]), "+f"(acc[j][1]), "+f"(acc[j][2]), "+f"(acc[j][3])
                                 : "r"(lo0), "r"(lo1), "r"(lo2), "r"(lo3), "r"(b0), "r"(b1));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                 : "+f"(acc[j][0]), "+f"(acc[j][1]), "+f"(acc[j][2]), "+f"(acc[j][3])
                                 : "r"(hi0), "r"(hi1), "r"(hi2), "r"(hi3), "r"(b0), "r"(b1));
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < GP_NF; ++j) {
        *reinterpret_cast<float2*>(part + (warp * 16 + g) * GP_LD + 8 * j + 2 * c) = make_float2(acc[j][0], acc[j][1]);
        *reinterpret_cast<float2*>(part + (warp * 16 + g + 8) * GP_LD + 8 * j + 2 * c) = make_float2(acc[j][2], acc[j][3]);
    }
    cg::cluster_group cluster = cg::this_cluster();
    cluster.sync();
    const float* remote[GP_S];
#pragma unroll
    for (int q = 0; q < GP_S; ++q) remote[q] = cluster.map_shared_rank(part, q);
    for (int idx = tid; idx < (GP_TT / GP_S) * GP_HV; idx += 128) {
        const int tl = rank * (GP_TT / GP_S) + idx / GP_HV, h = idx % GP_HV;
        float a = 0.f, b = 0.f;
#pragma unroll
        for (int q = 0; q < GP_S; ++q) { a += remote[q][tl * GP_LD + h]; b += remote[q][tl * GP_LD + GP_HV + h]; }
        const int64_t t = tile0 + tl;
        if (t >= T) continue;
        const float v = a + dt[h];
        gate[t * GP_HV + h] = (v > 20.0f ? v : log1pf(__expf(v))) * ssm_a[h];
        beta[t * GP_HV + h] = 1.0f / (1.0f + __expf(-b));
    }
    cluster.sync();   // no block leaves while another still reads its partial sums
}

// Block (row t, head j): j < n_head a query head, else a key/value head; one thread per head dim.
__device__ __forceinline__ float ldf(const float* p) { return *p; }
__device__ __forceinline__ float ldf(const __nv_bfloat16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ void stf(float* p, float v) { *p = v; }
__device__ __forceinline__ void stf(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
template <int HDW, class In = float, class QO = float>
// (launched through launch_pdl: the earlier kernels' outputs are read after pdl_wait, so they are not __restrict__)
__global__ void __launch_bounds__(HDW) attn_qk_prepare_kernel(const In* q_full, const In* k, const In* v,
                                                            const float* __restrict__ q_norm,
                                                            const float* __restrict__ k_norm, float eps, RopeSpec rope,
                                                            const int32_t* pos, int n_head, int n_kv, QO* q_out,
                                                            __half* k_cache, __half* v_cache, int64_t cell0, int q_stride,
                                                            int64_t cap, const int32_t* cell0_dev, KvInt8 q8) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float red[HDW / 32];
    __shared__ float xs[HDW], ks8[HDW], vs8[HDW];
    const int64_t t = blockIdx.x;
    const int j = blockIdx.y, d = threadIdx.x;
    const bool is_q = j < n_head;
    const int head = is_q ? j : j - n_head;
    const float x = is_q ? ldf(q_full + (t * n_head + head) * q_stride + d) : ldf(k + (t * n_kv + head) * HDW + d);
    const float inv = rsqrtf(block_sum(x * x, red) / (float) HDW + eps);
    float y = x * inv * (is_q ? q_norm[d] : k_norm[d]);
    const int half = rope.rot / 2;
    xs[d] = y;
    __syncthreads();
    if (d < rope.rot) {
        const int i = d < half ? d : d - half;
        const float inv_freq = rope.inv_freq[i];
        const float ang = (float) pos[t * 3 + rope.axis[i]] * inv_freq;
        float sn, cs;
        sincosf(ang, &sn, &cs);
        y = d < half ? y * cs - xs[d + half] * sn : y * cs + xs[d - half] * sn;
    }
    if (is_q) {
        stf(q_out + (t * n_head + head) * HDW + d, y);
        return;
    }
    const int64_t c0 = cell0_dev ? (int64_t) *cell0_dev : cell0;
    const int64_t cell = cap > 0 ? (c0 + t) % cap : c0 + t;
    const int64_t at = (cell * n_kv + head) * HDW + d;
    const float vv = ldf(v + (t * n_kv + head) * HDW + d);
    if (!q8.k) {
        k_cache[at] = __float2half_rn(y);
        v_cache[at] = __float2half_rn(vv);
        return;
    }
    // int8: per 64 values, scale = amax / 127 (stored fp16), codes rounded to nearest against the stored scale; a group
    // is two warps: each warp's max by shuffles, then the pair's (a max is exact in any order)
    float ak = fabsf(y), av = fabsf(vv);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        ak = fmaxf(ak, __shfl_xor_sync(0xffffffffu, ak, o));
        av = fmaxf(av, __shfl_xor_sync(0xffffffffu, av, o));
    }
    if ((d & 31) == 0) {
        ks8[d >> 5] = ak;
        vs8[d >> 5] = av;
    }
    __syncthreads();
    const int w0 = (d >> 5) & ~1;
    ak = fmaxf(ks8[w0], ks8[w0 + 1]);
    av = fmaxf(vs8[w0], vs8[w0 + 1]);
    const __half hk = __float2half_rn(ak / 127.0f), hv = __float2half_rn(av / 127.0f);
    const float sk = __half2float(hk), sv = __half2float(hv);
    const float qk = sk > 0.0f ? rintf(y / sk) : 0.0f, qv = sv > 0.0f ? rintf(vv / sv) : 0.0f;
    q8.k[at] = (int8_t) fmaxf(-127.0f, fminf(127.0f, qk));
    q8.v[at] = (int8_t) fmaxf(-127.0f, fminf(127.0f, qv));
    if ((d & 63) == 0) {
        q8.ks[(cell * n_kv + head) * (HDW / 64) + d / 64] = __half_as_ushort(hk);
        q8.vs[(cell * n_kv + head) * (HDW / 64) + d / 64] = __half_as_ushort(hv);
    }
}

__global__ void dyn_conv2_kernel(const float* x, const float* dyn, int64_t ld_dyn, const float* __restrict__ base, float* out,
                                 int64_t T, int64_t H, int group) {
    pdl_launch_dependents();
    pdl_wait();
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * H) return;
    const int64_t t = i / H, c = i % H;
    const int64_t groups = H / group;
    const float* dr = dyn + t * ld_dyn;
    float acc = (base[c] + dr[c / group]) * x[i];
    if (t > 0) acc += (base[H + c] + dr[groups + c / group]) * x[i - H];
    out[i] = acc;
}

// One block of `rank` threads walks the positions: the predecessor's row times h into shared memory, then each warp
// scores its candidates (a warp sum over the rank), then one thread picks.  Two barriers a position.
__global__ void candidate_path_kernel(const float* __restrict__ lp_rows, int stride, int top, const float* __restrict__ h,
                                      const uint16_t* __restrict__ pred, const uint16_t* __restrict__ succ, int rank,
                                      const int32_t* __restrict__ anchor, int n, int32_t* __restrict__ ids,
                                      float* __restrict__ conf) {
    extern __shared__ float ph[];   // [rank]
    __shared__ float score[64];
    __shared__ int32_t prev_s;
    const int r = threadIdx.x, lane = r & 31, warp = r >> 5, warps = blockDim.x >> 5;
    if (r == 0) prev_s = *anchor;
    __syncthreads();
    for (int p = 0; p < n; ++p) {
        const float* row = lp_rows + (size_t) p * stride;
        ph[r] = __uint_as_float((uint32_t) pred[(size_t) prev_s * rank + r] << 16) * h[(size_t) p * rank + r];
        __syncthreads();
        for (int k = warp; k < top; k += warps) {
            const int32_t cand = __float_as_int(row[2 + 2 * k]);
            float v = 0.0f;
            if (cand >= 0)
                for (int i = lane; i < rank; i += 32) v += ph[i] * __uint_as_float((uint32_t) succ[(size_t) cand * rank + i] << 16);
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0) score[k] = cand >= 0 ? row[3 + 2 * k] + v : -1e30f;
        }
        __syncthreads();
        if (r == 0) {
            int best = 0;
            for (int k = 1; k < top; ++k)
                if (score[k] > score[best]) best = k;
            float z = 0.0f;
            for (int k = 0; k < top; ++k) z += expf(score[k] - score[best]);
            const int32_t id = __float_as_int(row[2 + 2 * best]);
            ids[p] = id;
            conf[p] = 1.0f / z;
            prev_s = id;
        }
        __syncthreads();
    }
}

__global__ void attn_gate_kernel(const float* __restrict__ attn, const float* __restrict__ q_full, float* __restrict__ out,
                                 int64_t n, int n_head) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t th = i / HD, d = i % HD;
    const float g = q_full[th * 2 * HD + HD + d];
    out[i] = attn[i] / (1.0f + expf(-g));
}

// attn_gate_kernel's values of one token's row (block per token), rounded to E4M3 with scale amax / 448 as
// prefill::fp8::quantize rounds them (bitwise)
template <class Q, class A = float>
__global__ void __launch_bounds__(256) attn_gate_e4m3_kernel(const A* __restrict__ attn, const Q* __restrict__ q_full,
                                                             int n_head, uint8_t* __restrict__ codes, float* __restrict__ scales) {
    constexpr int kPer = 8;   // values a thread keeps: n_head * HD <= 256 * 4 * kPer
    __shared__ float red[8];
    const int64_t t = blockIdx.x;
    const int cols = n_head * HD;
    float4 v[kPer];
    float m = 0.0f;
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        const int i = (j * 256 + (int) threadIdx.x) * 4;
        if (i >= cols) break;
        const int h = i / HD, d = i % HD;
        float4 a;
        if constexpr (sizeof(A) == 4) {
            a = *reinterpret_cast<const float4*>(attn + t * cols + i);
        } else {
            const uint2 u = *reinterpret_cast<const uint2*>(attn + t * cols + i);
            const float2 x0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
            const float2 x1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
            a = make_float4(x0.x, x0.y, x1.x, x1.y);
        }
        float4 g;
        if constexpr (sizeof(Q) == 4) {
            g = *reinterpret_cast<const float4*>(q_full + (t * n_head + h) * 2 * HD + HD + d);
        } else {
            const uint2 u = *reinterpret_cast<const uint2*>(q_full + (t * n_head + h) * 2 * HD + HD + d);
            const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
            const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
            g = make_float4(a.x, a.y, b.x, b.y);
        }
        v[j] = make_float4(a.x / (1.0f + expf(-g.x)), a.y / (1.0f + expf(-g.y)), a.z / (1.0f + expf(-g.z)), a.w / (1.0f + expf(-g.w)));
        m = fmaxf(m, fmaxf(fmaxf(fabsf(v[j].x), fabsf(v[j].y)), fmaxf(fabsf(v[j].z), fabsf(v[j].w))));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) amax = fmaxf(amax, red[i]);
    if (threadIdx.x == 0) scales[t] = amax / 448.0f;
    const float q = amax > 0.0f ? 448.0f / amax : 0.0f;
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        const int i = (j * 256 + (int) threadIdx.x) * 4;
        if (i >= cols) break;
        const __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(make_float2(v[j].x * q, v[j].y * q), __NV_SATFINITE, __NV_E4M3);
        const __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(make_float2(v[j].z * q, v[j].w * q), __NV_SATFINITE, __NV_E4M3);
        *reinterpret_cast<uint32_t*>(codes + t * cols + i) = (uint32_t) lo | ((uint32_t) hi << 16);
    }
}

__global__ void gdn_rows_scatter_kernel(const GdnRowDst* __restrict__ dst, const float* __restrict__ qkv, const float* __restrict__ h,
                                        const float* __restrict__ gate, const float* __restrict__ beta, int C, int h_v,
                                        int64_t layer, int window_rows) {
    const int r = blockIdx.x;
    const GdnRowDst d = dst[r];
    float* dq = d.qkv + layer * window_rows * C;
    float* dh = d.h + layer * window_rows * C;
    float* dg = d.gate + layer * window_rows * h_v;
    float* db = d.beta + layer * window_rows * h_v;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        dq[c] = qkv[(int64_t) r * C + c];
        dh[c] = h[(int64_t) r * C + c];
    }
    for (int c = threadIdx.x; c < h_v; c += blockDim.x) {
        dg[c] = gate[(int64_t) r * h_v + c];
        db[c] = beta[(int64_t) r * h_v + c];
    }
}

}  // namespace

void gdn_rows_scatter(const GdnRowDst* dst, int n_rows, const float* qkv, const float* h, const float* gate, const float* beta,
                      int C, int h_v, int64_t layer, int window_rows, void* stream) {
    if (n_rows <= 0) return;
    gdn_rows_scatter_kernel<<<(unsigned) n_rows, 256, 0, (cudaStream_t) stream>>>(dst, qkv, h, gate, beta, C, h_v, layer, window_rows);
    check("gdn_rows_scatter");
}

void rms_norm_rows(const float* x, const float* w, float* out, int64_t rows, int64_t cols, float eps, void* stream,
                   int64_t ld_out) {
    if (rows <= 0) return;
    rms_norm_rows_kernel<<<(unsigned) rows, 512, 0, (cudaStream_t) stream>>>(x, w, out, cols, ld_out > 0 ? ld_out : cols, eps);
    check("rms_norm_rows");
}

void add_scaled(float* x, const float* y, const float* scale, int64_t n, void* stream) {
    if (n <= 0) return;
    launch_pdl(add_scaled_kernel, blocks_for(n), dim3(256), 0, (cudaStream_t) stream, x, y, scale, n);
    check("add_scaled");
}

void rms_norm_rows(const uint16_t* x, const float* w, float* out, int64_t rows, int64_t cols, float eps, void* stream) {
    if (rows <= 0) return;
    rms_norm_rows_bf16_kernel<<<(unsigned) rows, 512, 0, (cudaStream_t) stream>>>(reinterpret_cast<const __nv_bfloat16*>(x), w, out,
                                                                                  cols, eps);
    check("rms_norm_rows");
}
void add_scaled(uint16_t* x, const float* y, const float* scale, int64_t n, void* stream) {
    if (n <= 0) return;
    launch_pdl(add_scaled_bf16_kernel, blocks_for(n), dim3(256), 0, (cudaStream_t) stream, reinterpret_cast<__nv_bfloat16*>(x), y,
               scale, n);
    check("add_scaled");
}
void to_bf16(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    to_bf16_kernel<<<blocks_for(n), 256, 0, (cudaStream_t) stream>>>(x, reinterpret_cast<__nv_bfloat16*>(y), n);
    check("to_bf16");
}
void to_f32(const uint16_t* x, float* y, int64_t n, void* stream) {
    if (n <= 0) return;
    to_f32_kernel<<<blocks_for(n), 256, 0, (cudaStream_t) stream>>>(reinterpret_cast<const __nv_bfloat16*>(x), y, n);
    check("to_f32");
}

void scale_values(float* y, const float* scale, int64_t n, void* stream) {
    if (n <= 0 || scale == nullptr) return;
    scale_values_kernel<<<blocks_for(n), 256, 0, (cudaStream_t) stream>>>(y, scale, n);
    check("scale_values");
}

void swiglu_scaled(const float* g, const float* u, const float* sg, const float* su, float* h, int64_t rows,
                   int64_t width, int64_t ld_gu, void* stream) {
    if (rows <= 0) return;
    launch_pdl(swiglu_scaled_kernel, blocks_for(rows * width), dim3(256), 0, (cudaStream_t) stream, g, u, sg, su, h, rows, width,
               ld_gu);
    check("swiglu_scaled");
}

void bf16_rows_dot(const float* x, const uint16_t* w, float* y, int64_t T, int64_t N, int64_t K, void* stream) {
    if (T <= 0) return;
    if (K % 8 != 0) { basalt::log::error("basalt", "bf16_rows_dot: K %lld", (long long) K); std::exit(1); }
    bf16_rows_dot_kernel<<<dim3((unsigned) ((T + RD_T - 1) / RD_T), (unsigned) ((N + RD_N - 1) / RD_N)), 32, 0,
                           (cudaStream_t) stream>>>(x, w, y, T, N, K);
    check("bf16_rows_dot");
}

void gdn_gating_proj(const float* x, const uint16_t* w_ab, const float* dt, const float* ssm_a, float* gate, float* beta,
                     int64_t T, int64_t HV, int64_t K, void* stream) {
    if (T <= 0) return;
    if (HV != GP_HV || K % (32 * GP_AHEAD * GP_S) != 0) {
        basalt::log::error("basalt", "gdn_gating_proj: HV %lld K %lld", (long long) HV, (long long) K);
        std::exit(1);
    }
    gdn_gating_proj_kernel<float><<<dim3((unsigned) ((T + GP_TT - 1) / GP_TT), 1, GP_S), 128, 0, (cudaStream_t) stream>>>(
        x, w_ab, dt, ssm_a, gate, beta, T, K);
    check("gdn_gating_proj");
}
void gdn_gating_proj(const uint16_t* x, const uint16_t* w_ab, const float* dt, const float* ssm_a, float* gate, float* beta,
                     int64_t T, int64_t HV, int64_t K, void* stream) {
    if (T <= 0) return;
    if (HV != GP_HV || K % (32 * GP_AHEAD * GP_S) != 0) {
        basalt::log::error("basalt", "gdn_gating_proj: HV %lld K %lld", (long long) HV, (long long) K);
        std::exit(1);
    }
    gdn_gating_proj_kernel<__nv_bfloat16><<<dim3((unsigned) ((T + GP_TT - 1) / GP_TT), 1, GP_S), 128, 0, (cudaStream_t) stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x), w_ab, dt, ssm_a, gate, beta, T, K);
    check("gdn_gating_proj");
}

void attn_qk_prepare(const float* q_full, const float* k, const float* v, const float* q_norm, const float* k_norm,
                     float eps, const RopeSpec& rope, const int32_t* pos, int64_t T, int n_head, int n_kv,
                     float* q_out, uint16_t* k_cache, uint16_t* v_cache, int64_t cell0, void* stream, int hd, bool gated,
                     int64_t cap, const int32_t* cell0_dev, KvInt8 q8) {
    if (T <= 0) return;
    if (rope.rot <= 0 || rope.rot > hd || rope.rot % 2 || (hd != 128 && hd != 256)) {
        basalt::log::error("basalt", "attn_qk_prepare: a rotary width of %d in %d-wide heads", rope.rot, hd);
        std::exit(1);
    }
    const dim3 grid((unsigned) T, (unsigned) (n_head + n_kv));
    const int qs = gated ? 2 * hd : hd;
    // each pair's inverse frequency, once on the host in double (the kernel used to take a double pow per thread)
    RopeSpec r = rope;
    for (int i = 0; i < rope.rot / 2; ++i) r.inv_freq[i] = (float) std::pow((double) rope.theta, -2.0 * (double) i / (double) rope.rot);
    if (hd == 256)
        launch_pdl(attn_qk_prepare_kernel<256>, grid, dim3(256), 0, (cudaStream_t) stream, q_full, k, v, q_norm, k_norm, eps, r,
                   pos, n_head, n_kv, q_out, (__half*) k_cache, (__half*) v_cache, cell0, qs, cap, cell0_dev, q8);
    else
        launch_pdl(attn_qk_prepare_kernel<128>, grid, dim3(128), 0, (cudaStream_t) stream, q_full, k, v, q_norm, k_norm, eps, r,
                   pos, n_head, n_kv, q_out, (__half*) k_cache, (__half*) v_cache, cell0, qs, cap, cell0_dev, q8);
    check("attn_qk_prepare");
}

void attn_qk_prepare_bf16(const uint16_t* q_full, const uint16_t* k, const uint16_t* v, const float* q_norm, const float* k_norm,
                          float eps, const RopeSpec& rope, const int32_t* pos, int64_t T, int n_head, int n_kv, uint16_t* q_out,
                          int64_t cell0, void* stream, KvInt8 q8, uint16_t* k_cache, uint16_t* v_cache) {
    if (T <= 0) return;
    if (rope.rot <= 0 || rope.rot > 256 || rope.rot % 2) {
        basalt::log::error("basalt", "attn_qk_prepare_bf16: a rotary width of %d", rope.rot);
        std::exit(1);
    }
    const dim3 grid((unsigned) T, (unsigned) (n_head + n_kv));
    RopeSpec r = rope;
    for (int i = 0; i < rope.rot / 2; ++i) r.inv_freq[i] = (float) std::pow((double) rope.theta, -2.0 * (double) i / (double) rope.rot);
    const __nv_bfloat16* qf = reinterpret_cast<const __nv_bfloat16*>(q_full);
    const __nv_bfloat16* kk = reinterpret_cast<const __nv_bfloat16*>(k);
    const __nv_bfloat16* vv = reinterpret_cast<const __nv_bfloat16*>(v);
    launch_pdl(attn_qk_prepare_kernel<256, __nv_bfloat16, __nv_bfloat16>, grid, dim3(256), 0, (cudaStream_t) stream, qf, kk, vv,
               q_norm, k_norm, eps, r, pos, n_head, n_kv, reinterpret_cast<__nv_bfloat16*>(q_out), (__half*) k_cache, (__half*) v_cache, cell0, 512, (int64_t) 0,
               (const int32_t*) nullptr, q8);
    check("attn_qk_prepare_bf16");
}

void dyn_conv2(const float* x, const float* dyn, int64_t ld_dyn, const float* base, float* out, int64_t T, int64_t H,
               int group, void* stream) {
    if (T <= 0) return;
    launch_pdl(dyn_conv2_kernel, blocks_for(T * H), dim3(256), 0, (cudaStream_t) stream, x, dyn, ld_dyn, base, out, T, H, group);
    check("dyn_conv2");
}

void candidate_path(const float* lp_rows, int stride, int top, const float* h, const uint16_t* pred, const uint16_t* succ,
                    int rank, const int32_t* anchor, int n, int32_t* ids, float* conf, void* stream) {
    if (n <= 0) return;
    if (top > 64 || rank % 32 || rank > 1024) {
        basalt::log::error("basalt", "candidate_path: %d candidates of rank %d", top, rank);
        std::exit(1);
    }
    candidate_path_kernel<<<1, rank, (size_t) rank * sizeof(float), (cudaStream_t) stream>>>(lp_rows, stride, top, h, pred, succ, rank, anchor, n, ids, conf);
    check("candidate_path");
}

void attn_gate(const float* attn, const float* q_full, float* out, int64_t T, int n_head, void* stream) {
    const int64_t n = T * n_head * HD;
    if (n <= 0) return;
    attn_gate_kernel<<<blocks_for(n), 256, 0, (cudaStream_t) stream>>>(attn, q_full, out, n, n_head);
    check("attn_gate");
}

void attn_gate_e4m3(const float* attn, const float* q_full, int64_t T, int n_head, uint8_t* codes, float* scales, void* stream) {
    if (T <= 0) return;
    if (n_head * HD > 256 * 4 * 8) {
        basalt::log::error("basalt", "attn_gate_e4m3: %d heads", n_head);
        std::abort();
    }
    attn_gate_e4m3_kernel<float><<<(unsigned) T, 256, 0, (cudaStream_t) stream>>>(attn, q_full, n_head, codes, scales);
    check("attn_gate_e4m3");
}
void attn_gate_e4m3(const uint16_t* attn, const uint16_t* q_full, int64_t T, int n_head, uint8_t* codes, float* scales,
                    void* stream) {
    if (T <= 0) return;
    if (n_head * HD > 256 * 4 * 8) {
        basalt::log::error("basalt", "attn_gate_e4m3: %d heads", n_head);
        std::abort();
    }
    attn_gate_e4m3_kernel<__nv_bfloat16, __nv_bfloat16><<<(unsigned) T, 256, 0, (cudaStream_t) stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(attn), reinterpret_cast<const __nv_bfloat16*>(q_full), n_head, codes, scales);
    check("attn_gate_e4m3");
}
void attn_gate_e4m3(const float* attn, const uint16_t* q_full, int64_t T, int n_head, uint8_t* codes, float* scales, void* stream) {
    if (T <= 0) return;
    if (n_head * HD > 256 * 4 * 8) {
        basalt::log::error("basalt", "attn_gate_e4m3: %d heads", n_head);
        std::abort();
    }
    attn_gate_e4m3_kernel<__nv_bfloat16><<<(unsigned) T, 256, 0, (cudaStream_t) stream>>>(
        attn, reinterpret_cast<const __nv_bfloat16*>(q_full), n_head, codes, scales);
    check("attn_gate_e4m3");
}

}  // namespace basalt::kernels
