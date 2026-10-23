// src/prefill/kernels.cu - see include/basalt/prefill/kernels.hpp.
#include "basalt/prefill/kernels.hpp"
#include "basalt/kernels/mrope.hpp"
#include "basalt/kernels/router_top10.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "basalt/log.hpp"

namespace basalt::prefill {
namespace {

constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;
constexpr int S = 128, HK = 16, HV = 48, C = 10240;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ uint16_t bf(float f) {
    uint32_t u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
// The BF16 GEMMs' second operand (PREFILL_BF16X2): what the BF16 image `hi` left out of f, itself in BF16.
// W.hi + W.lo carries ~16 mantissa bits of the activation - the decode path's FP32 x to within ~1e-5.
__device__ __forceinline__ uint16_t bf_lo(float f, uint16_t hi) { return bf(f - __uint_as_float((uint32_t) hi << 16)); }
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ uint16_t hf(float f) { return __half_as_ushort(__float2half_rn(f)); }
// A SwiGLU product for an FP16 GEMM: saturated, so a token with a massive activation cannot turn into inf and then
// NaN in the down projection (decode's q8_1 has room to ~8e6; FP16 ends at 65504).  A NaN stays NaN (fminf/fmaxf
// would make it -65504 and hide where it came from); finite values below 65504 round exactly as before.
__device__ __forceinline__ uint16_t hf_sat(float f) { return hf(isnan(f) ? f : fminf(fmaxf(f, -65504.0f), 65504.0f)); }
// block-wide sum for blockDim.x <= 1024, result broadcast
__device__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    float t = (threadIdx.x < nw) ? sh[threadIdx.x] : 0.0f;
    if (w == 0) t = warp_sum(t);
    if (threadIdx.x == 0) sh[0] = t;
    __syncthreads();
    return sh[0];
}
void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { basalt::log::error("basalt", "prefill %s: %s", what, cudaGetErrorString(e)); std::exit(1); }
}
unsigned blocks_for(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }

// ---------------------------------------------------------------- hyper-connection
__global__ void gr_norm_kernel(const float* __restrict__ R, const float* __restrict__ w, float eps,
                               float* __restrict__ xn, uint16_t* __restrict__ xn16, uint16_t* __restrict__ xn16_lo) {
    __shared__ float sh[32];
    const int64_t row = blockIdx.x;                 // t * 4 + c
    const int c = (int) (row % HC);
    const float* r = R + row * N;
    float ss = 0.0f;
    for (int d = threadIdx.x; d < N; d += blockDim.x) ss += r[d] * r[d];
    const float rs = rsqrtf(block_sum(ss, sh) / (float) N + eps);
    for (int d = threadIdx.x; d < N; d += blockDim.x) {
        const float v = r[d] * rs * w[c * N + d];
        xn[row * N + d] = v;
        const uint16_t h = bf(v);
        xn16[row * N + d] = h;
        if (xn16_lo) xn16_lo[row * N + d] = bf_lo(v, h);
    }
}
// F-1: the row scale only (and the BF16 image); gr_mix_r_kernel recomputes r * rs * w itself, in the same order,
// so the FP32 copy of the normalized rows (T x 10240 floats) is neither written nor read
__global__ void gr_norm_rs_kernel(const float* __restrict__ R, const float* __restrict__ w, float eps,
                                  float* __restrict__ rs_out, uint16_t* __restrict__ xn16,
                                  uint16_t* __restrict__ xn16_lo) {
    __shared__ float sh[32];
    const int64_t row = blockIdx.x;                 // t * 4 + c
    const int c = (int) (row % HC);
    const float* r = R + row * N;
    float ss = 0.0f;
    for (int d = threadIdx.x; d < N; d += blockDim.x) ss += r[d] * r[d];
    const float rs = rsqrtf(block_sum(ss, sh) / (float) N + eps);
    if (threadIdx.x == 0) rs_out[row] = rs;
    for (int d = threadIdx.x; d < N; d += blockDim.x) {
        const float v = r[d] * rs * w[c * N + d];
        const uint16_t h = bf(v);
        xn16[row * N + d] = h;
        if (xn16_lo) xn16_lo[row * N + d] = bf_lo(v, h);
    }
}
__global__ void gr_mix_r_kernel(const float* __restrict__ R, const float* __restrict__ rs, const float* __restrict__ w,
                                const float* __restrict__ g, float* __restrict__ mixed, uint16_t* __restrict__ mixed16,
                                int64_t T, uint16_t* __restrict__ mixed_h, uint16_t* __restrict__ mixed16_lo) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    const int64_t t = i / N, d = i % N;
    float s = 0.0f;
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const int64_t j = t * D + c * N + d;
        const float x = R[j] * rs[t * HC + c] * w[c * N + d];   // gr_norm_kernel's value, bit for bit
        s = fmaf(x, sigm(g[j]), s);
    }
    s /= (float) HC;
    if (mixed) mixed[i] = s;
    if (mixed16) {
        const uint16_t h = bf(s);
        mixed16[i] = h;
        if (mixed16_lo) mixed16_lo[i] = bf_lo(s, h);
    }
    if (mixed_h) mixed_h[i] = hf(s);
}
// The up product and gr_mix_r in one pass (gr_up_mix): a block takes UM_BM tokens and UM_BD columns d of all HC
// streams (rows c * N + d of w_up), so the FP32 gate rows stay in registers.  Every gate value is one m16n8k16 chain
// over k in order from zero, the chain cuBLAS runs for this product, and the mix is gr_mix_r_kernel's arithmetic.
constexpr int UM_BM = 128, UM_BD = 32, UM_BN = HC * UM_BD, UM_BK = 32, UM_ST = 3, UM_KT = LR / UM_BK;
constexpr int UM_SMEM = UM_ST * (UM_BM + UM_BN) * UM_BK * 2;
static_assert(LR % UM_BK == 0 && N % UM_BD == 0 && UM_KT >= UM_ST - 1, "gr_up_mix tiles");

__device__ __forceinline__ void um_cp16(void* smem, const void* gmem, bool valid) {
    const unsigned sa = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(gmem), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void um_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void um_ldsm4(uint32_t (&r)[4], const void* smem) {
    const unsigned sa = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(sa));
}
__device__ __forceinline__ void um_mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// a tile row (UM_BK bf16) is four 16-byte chunks, the chunk index XORed with the row's bits 1-2 so that the eight rows
// of an ldmatrix read sit in distinct banks
__device__ __forceinline__ int um_at(int row, int chunk) { return row * UM_BK + ((chunk ^ ((row >> 1) & 3)) << 3); }

__device__ void gr_up_mix_block(const uint16_t* __restrict__ lo16, const uint16_t* __restrict__ w_up,
                                const float* __restrict__ R, const float* __restrict__ rs, const float* __restrict__ w,
                                float* __restrict__ mixed, uint16_t* __restrict__ mixed16,
                                uint16_t* __restrict__ mixed_h, uint16_t* __restrict__ mixed16_lo, int64_t T, int bd,
                                int64_t bm, uint16_t* smem) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp & 3, wn = warp >> 2;          // a warp: 32 tokens x 16 columns d, all HC streams
    const int64_t t0 = bm * UM_BM;
    const int d0 = bd * UM_BD;
    uint16_t* As = smem;                              // [UM_ST][UM_BM][UM_BK]: lo16 rows
    uint16_t* Bs = smem + UM_ST * UM_BM * UM_BK;      // [UM_ST][UM_BN][UM_BK]: row c * UM_BD + j = w_up row c * N + d0 + j
    auto load = [&](int st, int kt) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int id = tid + i * 256, row = id >> 2, ch = id & 3;
            const bool ok = t0 + row < T;
            um_cp16(As + st * UM_BM * UM_BK + um_at(row, ch), lo16 + (ok ? (t0 + row) * LR : 0) + kt * UM_BK + ch * 8, ok);
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int id = tid + i * 256, row = id >> 2, ch = id & 3;
            const int64_t wr = (int64_t) (row / UM_BD) * N + d0 + row % UM_BD;
            um_cp16(Bs + st * UM_BN * UM_BK + um_at(row, ch), w_up + wr * LR + kt * UM_BK + ch * 8, true);
        }
        um_commit();
    };
    // the residual rows the mix reads, into L2 while the product runs
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int id = tid + i * 256;
        const int64_t t = t0 + (id >> 2);
        if (t < T) asm volatile("prefetch.global.L2 [%0];\n" ::"l"(R + t * D + (id & 3) * N + d0));
    }
    float acc[2][HC][2][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int c = 0; c < HC; ++c)
#pragma unroll
            for (int db = 0; db < 2; ++db)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[mt][c][db][e] = 0.0f;
#pragma unroll
    for (int s = 0; s < UM_ST - 1; ++s) load(s, s);
#pragma unroll 1
    for (int kt = 0; kt < UM_KT; ++kt) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(UM_ST - 2));
        __syncthreads();
        if (kt + UM_ST - 1 < UM_KT) load((kt + UM_ST - 1) % UM_ST, kt + UM_ST - 1);
        else um_commit();
        const uint16_t* a_s = As + (kt % UM_ST) * UM_BM * UM_BK;
        const uint16_t* b_s = Bs + (kt % UM_ST) * UM_BN * UM_BK;
#pragma unroll
        for (int kk = 0; kk < UM_BK / 16; ++kk) {
            uint32_t a[2][4], b[HC][4];
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) um_ldsm4(a[mt], a_s + um_at(wm * 32 + mt * 16 + (lane & 15), kk * 2 + (lane >> 4)));
#pragma unroll
            for (int c = 0; c < HC; ++c)
                um_ldsm4(b[c], b_s + um_at(c * UM_BD + wn * 16 + (lane & 7) + ((lane >> 4) << 3), kk * 2 + ((lane >> 3) & 1)));
#pragma unroll
            for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                for (int c = 0; c < HC; ++c)
#pragma unroll
                    for (int db = 0; db < 2; ++db) um_mma(acc[mt][c][db], a[mt], b[c][db * 2], b[c][db * 2 + 1]);
        }
    }
    // the mix: a thread holds tokens g and g + 8 of each 16-token tile and columns 2q, 2q + 1 of each 8-column tile,
    // for every stream
    const int g = lane >> 2, q = lane & 3;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
            const int64_t t = t0 + wm * 32 + mt * 16 + rh * 8 + g;
            if (t >= T) continue;
            const float4 r4 = *reinterpret_cast<const float4*>(rs + t * HC);
            const float rsc[HC] = {r4.x, r4.y, r4.z, r4.w};
#pragma unroll
            for (int db = 0; db < 2; ++db) {
                const int d = d0 + wn * 16 + db * 8 + 2 * q;
                float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
                for (int c = 0; c < HC; ++c) {
                    const float2 r = *reinterpret_cast<const float2*>(R + t * D + c * N + d);
                    const float2 wv = *reinterpret_cast<const float2*>(w + c * N + d);
                    const float x0 = r.x * rsc[c] * wv.x, x1 = r.y * rsc[c] * wv.y;   // gr_norm_kernel's value
                    s0 = fmaf(x0, sigm(acc[mt][c][db][rh * 2]), s0);
                    s1 = fmaf(x1, sigm(acc[mt][c][db][rh * 2 + 1]), s1);
                }
                s0 /= (float) HC;
                s1 /= (float) HC;
                const int64_t i = t * N + d;
                if (mixed) *reinterpret_cast<float2*>(mixed + i) = make_float2(s0, s1);
                if (mixed16) {
                    const uint16_t h0 = bf(s0), h1 = bf(s1);
                    *reinterpret_cast<uint32_t*>(mixed16 + i) = (uint32_t) h0 | ((uint32_t) h1 << 16);
                    if (mixed16_lo)
                        *reinterpret_cast<uint32_t*>(mixed16_lo + i) =
                            (uint32_t) bf_lo(s0, h0) | ((uint32_t) bf_lo(s1, h1) << 16);
                }
                if (mixed_h) *reinterpret_cast<uint32_t*>(mixed_h + i) = (uint32_t) hf(s0) | ((uint32_t) hf(s1) << 16);
            }
        }
}
__global__ void __launch_bounds__(256, 2) gr_up_mix_kernel(const uint16_t* __restrict__ lo16,
                                                          const uint16_t* __restrict__ w_up, const float* __restrict__ R,
                                                          const float* __restrict__ rs, const float* __restrict__ w,
                                                          float* __restrict__ mixed, uint16_t* __restrict__ mixed16,
                                                          uint16_t* __restrict__ mixed_h,
                                                          uint16_t* __restrict__ mixed16_lo, int64_t T) {
    extern __shared__ __align__(16) uint16_t um_smem[];
    gr_up_mix_block(lo16, w_up, R, rs, w, mixed, mixed16, mixed_h, mixed16_lo, T, (int) blockIdx.x,
                    (int64_t) blockIdx.y, um_smem);
}
// F-2: gr_write_kernel for one row (t, c), then gr_norm_rs_kernel's reduction over it with the next half's norm
// weights - the same thread-to-element mapping (256 threads, stride 256) and block_sum, so rs and the BF16 image are
// the same bits, and R is not read back
constexpr int GRW_PER = (N + 255) / 256;
__global__ void __launch_bounds__(256) gr_write_norm_rs_kernel(float* __restrict__ R, const float* __restrict__ bo,
                                                               const float* __restrict__ inj, int64_t inj_ld,
                                                               const float* __restrict__ w, float eps,
                                                               float* __restrict__ rs_out, uint16_t* __restrict__ xn16,
                                                               uint16_t* __restrict__ xn16_lo) {
    __shared__ float sh[32];
    const int64_t row = blockIdx.x;                 // t * 4 + c
    const int64_t t = row / HC;
    const int c = (int) (row % HC);
    float* r = R + row * N;
    const float sc = 2.0f * sigm(inj[t * inj_ld + c] / (float) HC);
    float v[GRW_PER];
    float ss = 0.0f;
    int k = 0;
#pragma unroll
    for (int d = threadIdx.x; d < N; d += 256, ++k) {
        const float x = fmaf(bo[t * N + d], sc, r[d]);
        r[d] = x;
        v[k] = x;
        ss += x * x;
    }
    const float rs = rsqrtf(block_sum(ss, sh) / (float) N + eps);
    if (threadIdx.x == 0) rs_out[row] = rs;
    k = 0;
#pragma unroll
    for (int d = threadIdx.x; d < N; d += 256, ++k) {
        const float x = v[k] * rs * w[c * N + d];
        const uint16_t h = bf(x);
        xn16[row * N + d] = h;
        if (xn16_lo) xn16_lo[row * N + d] = bf_lo(x, h);
    }
}
__global__ void gr_silu_kernel(const float* __restrict__ lo, uint16_t* __restrict__ lo16, uint16_t* __restrict__ lo16_lo,
                               int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = lo[i] / (float) HC;
    const float v = x / (1.0f + __expf(-x));
    const uint16_t h = bf(v);
    lo16[i] = h;
    if (lo16_lo) lo16_lo[i] = bf_lo(v, h);
}
__global__ void gr_mix_kernel(const float* __restrict__ xn, const float* __restrict__ g, float* __restrict__ mixed,
                              uint16_t* __restrict__ mixed16, int64_t T, uint16_t* __restrict__ mixed_h,
                              uint16_t* __restrict__ mixed16_lo) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    const int64_t t = i / N, d = i % N;
    float s = 0.0f;
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const int64_t j = t * D + c * N + d;
        s = fmaf(xn[j], sigm(g[j]), s);
    }
    s /= (float) HC;
    mixed[i] = s;
    if (mixed16) {
        const uint16_t h = bf(s);
        mixed16[i] = h;
        if (mixed16_lo) mixed16_lo[i] = bf_lo(s, h);
    }
    if (mixed_h) mixed_h[i] = hf(s);
}
__global__ void gr_write_kernel(float* __restrict__ R, const float* __restrict__ bo, const float* __restrict__ inj,
                                int64_t inj_ld, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * D) return;
    const int64_t t = i / D, c = (i % D) / N, d = i % N;
    R[i] = fmaf(bo[t * N + d], 2.0f * sigm(inj[t * inj_ld + c] / (float) HC), R[i]);
}
__global__ void gr_broadcast_kernel(const float* __restrict__ e, float* __restrict__ R, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * D) return;
    const int64_t t = i / D, d = i % N;
    R[i] = e[t * N + d];
}

// ---------------------------------------------------------------- GDN
__global__ void gdn_gates_kernel(const float* __restrict__ ab, const float* __restrict__ dt,
                                 const float* __restrict__ ssm_a, float* __restrict__ gate, float* __restrict__ beta,
                                 int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * HV) return;
    const int64_t t = i / HV, h = i % HV;
    const float v = ab[t * 2 * HV + h] + dt[h];
    gate[i] = (v > 20.0f ? v : log1pf(__expf(v))) * ssm_a[h];
    beta[i] = sigm(ab[t * 2 * HV + HV + h]);
}
// one thread per channel, walks the chunk; then a second kernel normalises
__global__ void gdn_conv_kernel(float* __restrict__ hist, const float* __restrict__ qkv, const float* __restrict__ w,
                                float* __restrict__ h, int64_t T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float v0 = hist[c * 3], v1 = hist[c * 3 + 1], v2 = hist[c * 3 + 2];
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int64_t t = 0; t < T; ++t) {
        const float x = qkv[t * C + c];
        const float s = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
        h[t * C + c] = s / (1.0f + __expf(-s));
        v0 = v1; v1 = v2; v2 = x;
    }
    hist[c * 3] = v0; hist[c * 3 + 1] = v1; hist[c * 3 + 2] = v2;
}
// C-3: the same 4-tap causal conv, tiled over tokens: thread (c, tile) reads its tile's 3 predecessors from the
// chunk (or the history before it) instead of carrying them - the conv reads inputs, not its own outputs, so the
// tiles are independent. The same expression per element (so the same bits); the history is written afterwards.
constexpr int CONV_TILE = 64;
__global__ void gdn_conv_tiled_kernel(const float* __restrict__ hist, const float* __restrict__ qkv,
                                      const float* __restrict__ w, float* __restrict__ h, int64_t T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int64_t t0 = (int64_t) blockIdx.y * CONV_TILE;
    if (t0 >= T) return;
    const int64_t t1 = t0 + CONV_TILE < T ? t0 + CONV_TILE : T;
    auto input = [&](int64_t t) -> float { return t >= 0 ? qkv[t * C + c] : hist[c * 3 + (int) (t + 3)]; };
    float v0 = input(t0 - 3), v1 = input(t0 - 2), v2 = input(t0 - 1);
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int64_t t = t0; t < t1; ++t) {
        const float x = qkv[t * C + c];
        const float s = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
        h[t * C + c] = s / (1.0f + __expf(-s));
        v0 = v1; v1 = v2; v2 = x;
    }
}
// the history after the chunk: its last three inputs (the older history where the chunk is shorter than 3)
__global__ void gdn_conv_hist_kernel(float* __restrict__ hist, const float* __restrict__ qkv, int64_t T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float v[3];
    for (int k = 0; k < 3; ++k) {
        const int64_t t = T - 3 + k;
        v[k] = t >= 0 ? qkv[t * C + c] : hist[c * 3 + (int) (t + 3)];
    }
    hist[c * 3] = v[0]; hist[c * 3 + 1] = v[1]; hist[c * 3 + 2] = v[2];
}

// ---------------------------------------------------------------- MoE
template <int REG>
__global__ void route_kernel(const float* __restrict__ logits, int32_t* __restrict__ ids, float* __restrict__ wout,
                             int64_t T) {
    const int64_t t = (int64_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (t >= T) return;
    const int lane = threadIdx.x & 31;
    const float* lg = logits + t * (REG * 32);
    float v[REG];
#pragma unroll
    for (int i = 0; i < REG; ++i) v[i] = lg[lane + i * 32];
    float mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < REG; ++i) mx = fmaxf(mx, v[i]);
    mx = warp_max(mx);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < REG; ++i) { v[i] = expf(v[i] - mx); sum += v[i]; }
    const float rcp = 1.0f / warp_sum(sum);
#pragma unroll
    for (int i = 0; i < REG; ++i) { v[i] *= rcp; if (isnan(v[i])) v[i] = -FLT_MAX; }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = v[0];
        int ex = lane;
#pragma unroll
        for (int i = 1; i < REG; ++i) if (v[i] > best) { best = v[i]; ex = lane + i * 32; }
#pragma unroll
        for (int m = 16; m; m >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, m);
            const int oi = __shfl_xor_sync(0xffffffffu, ex, m);
            if (ob > best || (ob == best && oi < ex)) { best = ob; ex = oi; }
        }
        if ((ex & 31) == lane) { v[ex / 32] = -INFINITY; selected_sum += best; }
        if (lane == 0) ids[t * 10 + rank] = ex;
        if (rank == lane) selected = best;
    }
    selected_sum = fmaxf(warp_sum(selected_sum), 6.103515625e-5f);
    if (lane < 10) wout[t * 10 + lane] = selected / selected_sum;
}
// Pack blob: gate/up codes [1280][640 B], down codes [2560][160 B], gate/up scales [1280][40] f16, down scales [2560][10] f16
template <bool HALF>
__global__ void blob_dequant_kernel(const uint8_t* __restrict__ blob, uint16_t* __restrict__ gu16,
                                    uint16_t* __restrict__ d16) {
    constexpr size_t O_D_CODES = (size_t) 1280 * 640, O_GU_SC = O_D_CODES + (size_t) 2560 * 160,
                     O_D_SC = O_GU_SC + (size_t) 1280 * 40 * 2;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // one thread per 4 weights (one code byte)
    const int64_t n_gu = 1280LL * 640, n_d = 2560LL * 160;
    if (i < n_gu) {
        const int64_t row = i / 640, byte = i % 640;
        const uint8_t c = blob[row * 640 + byte];
        const uint8_t* sp = blob + O_GU_SC + (size_t) (row * 40 + (byte * 4) / 64) * 2;
        const float d = __half2float(__ushort_as_half((uint16_t) (sp[0] | (sp[1] << 8))));
        uint16_t* o = gu16 + row * 2560 + byte * 4;
#pragma unroll
        for (int k = 0; k < 4; ++k) { const float v = (float) (((c >> (2 * k)) & 3) - 1) * d; o[k] = HALF ? hf(v) : bf(v); }
    } else if (i < n_gu + n_d) {
        const int64_t j = i - n_gu, row = j / 160, byte = j % 160;
        const uint8_t c = blob[O_D_CODES + row * 160 + byte];
        const uint8_t* sp = blob + O_D_SC + (size_t) (row * 10 + (byte * 4) / 64) * 2;
        const float d = __half2float(__ushort_as_half((uint16_t) (sp[0] | (sp[1] << 8))));
        uint16_t* o = d16 + row * 640 + byte * 4;
#pragma unroll
        for (int k = 0; k < 4; ++k) { const float v = (float) (((c >> (2 * k)) & 3) - 1) * d; o[k] = HALF ? hf(v) : bf(v); }
    }
}
__global__ void swiglu_il_kernel(const float* __restrict__ gu, uint16_t* __restrict__ h16, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 640) return;
    const int64_t r = i / 640, k = i % 640;
    const float g = gu[r * 1280 + 2 * k], u = gu[r * 1280 + 2 * k + 1];
    h16[i] = hf_sat(g / (1.0f + __expf(-g)) * u);
}
__global__ void swiglu_pair_kernel(const float* __restrict__ g, const float* __restrict__ u, uint16_t* __restrict__ h16,
                                   int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 640) return;
    const float a = g[i];
    h16[i] = hf_sat(a / (1.0f + __expf(-a)) * u[i]);
}
__global__ void gather_rows16_kernel(const uint16_t* __restrict__ x, const int32_t* __restrict__ src,
                                     uint16_t* __restrict__ dst, int64_t n, int64_t width) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // one uint4 (8 bf16)
    const int64_t per = width / 8;
    if (i >= n * per) return;
    const int64_t r = i / per, j = i % per;
    reinterpret_cast<uint4*>(dst)[r * per + j] = reinterpret_cast<const uint4*>(x)[(int64_t) src[r] * per + j];
}
__global__ void moe_combine_kernel(const float* __restrict__ Dm, const int32_t* __restrict__ slot,
                                   const float* __restrict__ w, const float* __restrict__ shared,
                                   const float* __restrict__ sg, float* __restrict__ bo, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    const int64_t t = i / N, d = i % N;
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < 10; ++k) s = fmaf(w[t * 10 + k], Dm[(int64_t) slot[t * 10 + k] * N + d], s);
    bo[i] = s + shared[i] * sigm(sg[t]);
}

// ---------------------------------------------------------------- QSA helpers
__global__ void rms_rows_kernel(float* __restrict__ x, const float* __restrict__ w, int64_t cols, int64_t ld, float eps) {
    __shared__ float sh[32];
    float* r = x + (int64_t) blockIdx.x * ld;
    float ss = 0.0f;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) ss += r[c] * r[c];
    const float s = rsqrtf(block_sum(ss, sh) / (float) cols + eps);
    __syncthreads();
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) r[c] = s * r[c] * w[c];
}
// TAB: the angles from the session's float64 table.  The host launches <false> whenever
// no table applies - the default - so the default kernel does not contain the table read (merely skipping it at run time
// would change the compiled default path's results).
template <bool TAB>
__global__ void rope_kernel(float* __restrict__ x, int64_t heads, int64_t dim, int64_t ld, int64_t pos0,
                            float theta_scale, float freq_scale, float corr_low, float corr_high,
                            float ext_factor, float mscale, const int32_t* __restrict__ mtab,
                            basalt::kernels::RopeTab rt) {
    const int64_t row = blockIdx.x;             // t * heads + h
    const int pair = threadIdx.x;               // 0..31
    const int64_t t = row / heads, h = row % heads;
    float* p = x + t * ld + h * dim;
    float c, s;
    if (!(TAB && basalt::kernels::rope_tab_cs(rt, basalt::kernels::mrope_pos(mtab, (int) (pos0 + t), pair), pair, c,
                                              s))) {
        const float theta_extrap =
            (float) basalt::kernels::mrope_pos(mtab, (int) (pos0 + t), pair) * powf(theta_scale, (float) pair);
        basalt::kernels::rope_scaled_angle(theta_extrap, freq_scale, corr_low, corr_high, ext_factor, mscale,
                                           pair, c, s);
    }
    const float a = p[pair], b = p[pair + 32];
    p[pair] = a * c - b * s;
    p[pair + 32] = a * s + b * c;
}
__global__ void split_q_kernel(const float* __restrict__ qf, float* __restrict__ q, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * 24 * 256) return;
    const int64_t t = i / (24 * 256), h = (i / 256) % 24, d = i % 256;
    q[i] = qf[t * 24 * 512 + h * 512 + d];
}
__global__ void gate_attn_kernel(const float* __restrict__ a, const float* __restrict__ qf, uint16_t* __restrict__ o16,
                                 int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * 24 * 256) return;
    const int64_t t = i / (24 * 256), h = (i / 256) % 24, d = i % 256;
    o16[i] = hf(a[i] * (1.0f / (1.0f + expf(-qf[t * 24 * 512 + h * 512 + 256 + d]))));
}

// one block per (token, kv head, 64-value group); 64 threads. KV streaming: the pool page only if the block is
// resident (table >= 0), and the host copy and the prompt path's staging pool (both identity layout) when given.
__global__ void kv_append_kernel(const float* __restrict__ K, const float* __restrict__ V, int64_t pos0,
                                 const int32_t* __restrict__ table, int64_t page_size, uint16_t* k_pool,
                                 uint16_t* v_pool, int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale,
                                 basalt::kernels::KvHostPools host, basalt::kernels::KvHostPools stage) {
    const int64_t t = blockIdx.x;
    const int kvh = blockIdx.y, g = blockIdx.z >> 1;
    const bool is_v = (blockIdx.z & 1) != 0;
    const int d = g * 64 + threadIdx.x;
    const float x = (is_v ? V : K)[t * 512 + kvh * 256 + d];
    const int64_t pos = pos0 + t;
    const int64_t page = table[pos / page_size];
    const int64_t row = (page * 2 + kvh) * page_size + pos % page_size;
    const int64_t row_id = ((pos / page_size) * 2 + kvh) * page_size + pos % page_size;
    if (k_pool != nullptr) {
        const uint16_t h = hf(x);
        if (page >= 0) (is_v ? v_pool : k_pool)[row * 256 + d] = h;
        if (host.k_pool != nullptr) (is_v ? host.v_pool : host.k_pool)[row_id * 256 + d] = h;
        if (stage.k_pool != nullptr) (is_v ? stage.v_pool : stage.k_pool)[row_id * 256 + d] = h;
        return;
    }
    float a = fabsf(x);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    __shared__ float wm[2];
    if ((threadIdx.x & 31) == 0) wm[threadIdx.x >> 5] = a;
    __syncthreads();
    const float amax = fmaxf(wm[0], wm[1]);
    const uint16_t sb = hf(amax / 127.0f);
    const float sf = __half2float(__ushort_as_half(sb));
    int q = 0;
    if (sf > 0.0f) { q = __float2int_rn(x / sf); q = q < -127 ? -127 : (q > 127 ? 127 : q); }
    if (page >= 0) {
        (is_v ? v_q : k_q)[row * 256 + d] = (int8_t) q;
        if (threadIdx.x == 0) (is_v ? v_scale : k_scale)[row * 4 + g] = sb;
    }
    if (host.k_q != nullptr) {
        (is_v ? host.v_q : host.k_q)[row_id * 256 + d] = (int8_t) q;
        if (threadIdx.x == 0) (is_v ? host.v_scale : host.k_scale)[row_id * 4 + g] = sb;
    }
    if (stage.k_q != nullptr) {
        (is_v ? stage.v_q : stage.k_q)[row_id * 256 + d] = (int8_t) q;
        if (threadIdx.x == 0) (is_v ? stage.v_scale : stage.k_scale)[row_id * 4 + g] = sb;
    }
}
__global__ void to_f16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        y[i] = hf(x[i]);
}
__global__ void round_f16_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        y[i] = __half2float(__float2half_rn(x[i]));
}
__global__ void to_bf16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, uint16_t* __restrict__ ylo,
                               int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const uint16_t h = bf(x[i]);
        y[i] = h;
        if (ylo) ylo[i] = bf_lo(x[i], h);
    }
}

}  // namespace

void kv_append(const float* K, const float* V, int64_t T, int64_t pos0, const int32_t* page_table, int64_t page_size,
               uint16_t* k_pool, uint16_t* v_pool, int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale,
               void* stream, const basalt::kernels::KvHostPools* host, const basalt::kernels::KvHostPools* stage) {
    if (T <= 0) return;
    kv_append_kernel<<<dim3((unsigned) T, 2, 8), 64, 0, (cudaStream_t) stream>>>(
        K, V, pos0, page_table, page_size, k_pool, v_pool, k_q, v_q, k_scale, v_scale,
        host ? *host : basalt::kernels::KvHostPools{}, stage ? *stage : basalt::kernels::KvHostPools{});
    check("kv_append");
}
void to_f16(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    to_f16_kernel<<<(unsigned) ((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 256, 0, (cudaStream_t) stream>>>(x, y, n);
    check("to_f16");
}
void round_f16(const float* x, float* y, int64_t n, void* stream) {
    if (n <= 0) return;
    round_f16_kernel<<<(unsigned) ((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 256, 0, (cudaStream_t) stream>>>(x, y, n);
    check("round_f16");
}
void to_bf16(const float* x, uint16_t* y, int64_t n, void* stream, uint16_t* ylo) {
    if (n <= 0) return;
    to_bf16_kernel<<<(unsigned) ((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 256, 0, (cudaStream_t) stream>>>(x, y, ylo, n);
    check("to_bf16");
}

void gr_norm(const float* R, const float* w_norm, float eps, float* xn, uint16_t* xn16, int64_t T, void* stream,
             uint16_t* xn16_lo) {
    gr_norm_kernel<<<(unsigned) (T * HC), 256, 0, (cudaStream_t) stream>>>(R, w_norm, eps, xn, xn16, xn16_lo);
    check("gr_norm");
}
void gr_norm_rs(const float* R, const float* w_norm, float eps, float* rs, uint16_t* xn16, int64_t T, void* stream,
                uint16_t* xn16_lo) {
    gr_norm_rs_kernel<<<(unsigned) (T * HC), 256, 0, (cudaStream_t) stream>>>(R, w_norm, eps, rs, xn16, xn16_lo);
    check("gr_norm_rs");
}
void gr_mix_r(const float* R, const float* rs, const float* w_norm, const float* gated, float* mixed, uint16_t* mixed16,
              int64_t T, void* stream, uint16_t* mixed_h, uint16_t* mixed16_lo) {
    gr_mix_r_kernel<<<blocks_for(T * N), 256, 0, (cudaStream_t) stream>>>(R, rs, w_norm, gated, mixed, mixed16, T,
                                                                          mixed_h, mixed16_lo);
    check("gr_mix_r");
}
bool gr_up_mix(const uint16_t* lo16, const uint16_t* w_up, const float* R, const float* rs, const float* w_norm,
               float* mixed, uint16_t* mixed16, int64_t T, void* stream, uint16_t* mixed_h, uint16_t* mixed16_lo) {
    const auto al = [](const void* p, uintptr_t a) { return ((uintptr_t) p & (a - 1)) == 0; };
    if (!al(lo16, 16) || !al(w_up, 16) || !al(R, 8) || !al(rs, 16) || !al(w_norm, 8) || !al(mixed, 8) ||
        !al(mixed16, 4) || !al(mixed_h, 4) || !al(mixed16_lo, 4))
        return false;
    if (T <= 0) return true;
    const dim3 grid(N / UM_BD, (unsigned) ((T + UM_BM - 1) / UM_BM));
    gr_up_mix_kernel<<<grid, 256, UM_SMEM, (cudaStream_t) stream>>>(lo16, w_up, R, rs, w_norm, mixed, mixed16, mixed_h,
                                                                    mixed16_lo, T);
    check("gr_up_mix");
    return true;
}
void gr_write_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w_norm_next, float eps,
                      float* rs, uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo) {
    gr_write_norm_rs_kernel<<<(unsigned) (T * HC), 256, 0, (cudaStream_t) stream>>>(R, bo, inj, inj_ld, w_norm_next, eps,
                                                                                      rs, xn16, xn16_lo);
    check("gr_write_norm_rs");
}
void gr_silu(const float* lo, uint16_t* lo16, int64_t T, void* stream, uint16_t* lo16_lo) {
    gr_silu_kernel<<<blocks_for(T * LR), 256, 0, (cudaStream_t) stream>>>(lo, lo16, lo16_lo, T * LR);
    check("gr_silu");
}
void gr_mix(const float* xn, const float* gated, float* mixed, uint16_t* mixed16, int64_t T, void* stream,
            uint16_t* mixed_h, uint16_t* mixed16_lo) {
    gr_mix_kernel<<<blocks_for(T * N), 256, 0, (cudaStream_t) stream>>>(xn, gated, mixed, mixed16, T, mixed_h, mixed16_lo);
    check("gr_mix");
}
void gr_write(float* R, const float* bo, const float* inj, int64_t inj_ld, int64_t T, void* stream) {
    gr_write_kernel<<<blocks_for(T * D), 256, 0, (cudaStream_t) stream>>>(R, bo, inj, inj_ld, T);
    check("gr_write");
}
void gr_broadcast(const float* e, float* R, int64_t T, void* stream) {
    gr_broadcast_kernel<<<blocks_for(T * D), 256, 0, (cudaStream_t) stream>>>(e, R, T);
    check("gr_broadcast");
}
void gdn_gates(const float* ab, const float* dt, const float* ssm_a, float* gate, float* beta, int64_t T, void* stream) {
    gdn_gates_kernel<<<blocks_for(T * HV), 256, 0, (cudaStream_t) stream>>>(ab, dt, ssm_a, gate, beta, T);
    check("gdn_gates");
}
void gdn_conv(float* history, const float* qkv, const float* conv_w, float* h, int64_t T, void* stream) {
    if (T <= CONV_TILE) {
        gdn_conv_kernel<<<C / 128, 128, 0, (cudaStream_t) stream>>>(history, qkv, conv_w, h, T);
    } else {
        gdn_conv_tiled_kernel<<<dim3(C / 128, (unsigned) ((T + CONV_TILE - 1) / CONV_TILE)), 128, 0,
                                (cudaStream_t) stream>>>(history, qkv, conv_w, h, T);
        gdn_conv_hist_kernel<<<C / 128, 128, 0, (cudaStream_t) stream>>>(history, qkv, T);
    }
    check("gdn_conv");
}
void route(const float* logits, int32_t* ids, float* weights, int64_t T, int64_t n_expert, void* stream) {
    if (n_expert == 512)
        route_kernel<16><<<(unsigned) ((T + 7) / 8), 256, 0, (cudaStream_t) stream>>>(logits, ids, weights, T);
    else if (n_expert == 256)
        route_kernel<8><<<(unsigned) ((T + 7) / 8), 256, 0, (cudaStream_t) stream>>>(logits, ids, weights, T);
    else
        basalt::kernels::router_top10(logits, (int) T, (int) n_expert, 10, ids, weights, stream);
    check("route");
}
__global__ void route_count_kernel(const int32_t* __restrict__ ids, int64_t n, uint32_t* __restrict__ hist, int n_expert) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const int32_t e = ids[i];
        if (e >= 0 && e < n_expert) atomicAdd(hist + e, 1u);
    }
}
void route_count(const int32_t* ids, int64_t n, uint32_t* hist, int n_expert, void* stream) {
    route_count_kernel<<<(unsigned) std::min<int64_t>(64, (n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(ids, n, hist, n_expert);
    check("route_count");
}
void blob_dequant(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream) {
    blob_dequant_kernel<false><<<blocks_for(1280LL * 640 + 2560LL * 160), 256, 0, (cudaStream_t) stream>>>(blob, gu16, down16);
    check("blob_dequant");
}
void blob_dequant_f16(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream) {
    blob_dequant_kernel<true><<<blocks_for(1280LL * 640 + 2560LL * 160), 256, 0, (cudaStream_t) stream>>>(blob, gu16, down16);
    check("blob_dequant_f16");
}
void swiglu_interleaved(const float* gu, uint16_t* h16, int64_t n, void* stream) {
    if (n <= 0) return;
    swiglu_il_kernel<<<blocks_for(n * 640), 256, 0, (cudaStream_t) stream>>>(gu, h16, n);
    check("swiglu_interleaved");
}
void swiglu_pair(const float* g, const float* u, uint16_t* h16, int64_t n, void* stream) {
    swiglu_pair_kernel<<<blocks_for(n * 640), 256, 0, (cudaStream_t) stream>>>(g, u, h16, n);
    check("swiglu_pair");
}
namespace {
__global__ void copy_i32_kernel(int32_t* __restrict__ dst, const int32_t* __restrict__ src, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        dst[i] = src[i];
}
}  // namespace
namespace {
__global__ void copy_f4_kernel(float4* __restrict__ dst, const float4* __restrict__ src, int64_t n4) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (int64_t) gridDim.x * blockDim.x)
        dst[i] = src[i];
}
}  // namespace
void copy_f32_wide(float* dst, const float* src, int64_t n, void* stream) {
    if (n <= 0) return;
    const int64_t n4 = n / 4, b = std::min<int64_t>((n4 + 255) / 256, 4096);
    if (n4 > 0) copy_f4_kernel<<<(unsigned) b, 256, 0, (cudaStream_t) stream>>>((float4*) dst, (const float4*) src, n4);
    if (n % 4) copy_i32((int32_t*) dst + n4 * 4, (const int32_t*) src + n4 * 4, n % 4, stream);
    check("copy_f32_wide");
}
void copy_i32(int32_t* dst, const int32_t* src, int64_t n, void* stream) {
    if (n <= 0) return;
    const int64_t b = (n + 255) / 256;
    copy_i32_kernel<<<(unsigned) (b < 256 ? b : 256), 256, 0, (cudaStream_t) stream>>>(dst, src, n);
    check("copy_i32");
}
void gather_rows16(const uint16_t* x16, const int32_t* src, uint16_t* dst16, int64_t n, int64_t width, void* stream) {
    if (n <= 0) return;
    gather_rows16_kernel<<<blocks_for(n * (width / 8)), 256, 0, (cudaStream_t) stream>>>(x16, src, dst16, n, width);
    check("gather_rows16");
}
void moe_combine(const float* Dm, const int32_t* slot, const float* w, const float* shared, const float* sg, float* bo,
                 int64_t T, void* stream) {
    moe_combine_kernel<<<blocks_for(T * N), 256, 0, (cudaStream_t) stream>>>(Dm, slot, w, shared, sg, bo, T);
    check("moe_combine");
}
void rms_rows(float* x, const float* w, int64_t rows, int64_t cols, int64_t ld, float eps, void* stream) {
    if (rows <= 0) return;
    rms_rows_kernel<<<(unsigned) rows, 256, 0, (cudaStream_t) stream>>>(x, w, cols, ld, eps);
    check("rms_rows");
}
void rope(float* x, int64_t T, int64_t heads, int64_t dim, int64_t ld, int64_t pos0,
          const basalt::kernels::RopeScaling& scaling, void* stream) {
    // The engine validates the resolved config at startup with the same rule (generate.cpp), so this only
    // fires for a caller that bypassed it; the prompt path has no error return here, so it stops the process.
    if (const char* why = basalt::kernels::rope_scaling_invalid(scaling)) {
        basalt::log::error("basalt", "prefill rope: invalid rope scaling: %s", why);
        std::exit(1);
    }
    const float theta_scale = powf((float) scaling.freq_base, -2.0f / 64.0f);
    const basalt::kernels::RopeKernelArgs k = scaling.kernel_args(64);   // none: the identity constants
    const basalt::kernels::RopeTab rt = basalt::kernels::rope_table_for(scaling);
    if (rt.cos != nullptr)
        rope_kernel<true><<<(unsigned) (T * heads), 32, 0, (cudaStream_t) stream>>>(
            x, heads, dim, ld, pos0, theta_scale, k.freq_scale, k.corr_low, k.corr_high, k.ext_factor, k.attn_factor,
            basalt::kernels::mrope_table(), rt);
    else
        rope_kernel<false><<<(unsigned) (T * heads), 32, 0, (cudaStream_t) stream>>>(
            x, heads, dim, ld, pos0, theta_scale, k.freq_scale, k.corr_low, k.corr_high, k.ext_factor, k.attn_factor,
            basalt::kernels::mrope_table(), rt);
    check("rope");
}
void split_q(const float* q_full, float* q, int64_t T, void* stream) {
    split_q_kernel<<<blocks_for(T * 24 * 256), 256, 0, (cudaStream_t) stream>>>(q_full, q, T);
    check("split_q");
}
void gate_attn(const float* attn, const float* q_full, uint16_t* out16, int64_t T, void* stream) {
    gate_attn_kernel<<<blocks_for(T * 24 * 256), 256, 0, (cudaStream_t) stream>>>(attn, q_full, out16, T);
    check("gate_attn");
}

}  // namespace basalt::prefill
