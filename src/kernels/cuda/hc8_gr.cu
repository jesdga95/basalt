// src/kernels/cuda/hc8_gr.cu - the hyper-connection read on the matrices' 8-bit copies (int8, one f16 scale per 32
// values of a row; hc_fused.hpp) for every caller that does not run hc_fused_read: fused_gr_read, fused_gr_read_multi
// and gr_read hand such matrices here.  Three kernels:
//
//   norm : R' = R + bo_prev * 2 sigmoid(inj_prev / hc) into R_out (when apply),  rs[c] = rsqrt(mean(R'[c]^2) + eps)
//   down : lo[k] = silu((q_down[k] . xn) / hc),  inject[c] = q_inject[c] . xn,  xn = R' * w_norm * rs (on the fly)
//   up   : mixed[d] = mean_c xn[c,d] * sigmoid(q_up[c*n_embd + d] . lo)
//
// Every token's values come from the same operations in the same order whatever the token count.
#include "basalt/kernels/fused_gr.hpp"
#include "basalt/kernels/hc_fused.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include "basalt/log.hpp"

namespace basalt::kernels {
namespace {

constexpr int MT = kFusedGrMaxT;
constexpr int MAX_HC = 4;
constexpr int MAX_LR = 512;
constexpr int G8 = 32;
constexpr int DOWN_ROWS = 2;      // rows per down CTA
constexpr int DOWN_THREADS = 160;
constexpr int UP_THREADS = 256;

struct Hc8Gr {
    const float* R[MT];
    float* R_out[MT];
    const float* bo[MT];
    const float* inj_prev[MT];
    float* lo[MT];
    float* rs[MT];
    float* inj_out[MT];
    float* mixed[MT];
    const float* w_norm;
    const int8_t* qd;
    const uint16_t* sd;
    const int8_t* qu;
    const uint16_t* su;
    const int8_t* qi;
    const uint16_t* si;
    int sd_row, su_row, si_row;
    int T, n, hc, lr, ninj;
    bool apply;
    float eps;
};

__device__ __forceinline__ const float* src_of(const Hc8Gr& p, int t) { return p.apply ? p.R_out[t] : p.R[t]; }
__device__ __forceinline__ float s8(uint32_t w, int j) { return (float) (int) (int8_t) (w >> (8 * j)); }
__device__ __forceinline__ float f16f(uint16_t h) { return __half2float(__ushort_as_half(h)); }

// one CTA per (stream, token)
__device__ void hc8_norm(const Hc8Gr& p, int c, int t) {
    __shared__ float red[32];
    const int n = p.n;
    const float* r = p.R[t] + (size_t) c * n;
    float w = 0.0f;
    if (p.apply) w = 2.0f / (1.0f + expf(-p.inj_prev[t][c] / (float) p.hc));
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = r[i];
        if (p.apply) {
            v = fmaf(p.bo[t][i], w, v);
            p.R_out[t][(size_t) c * n + i] = v;
        }
        ss = fmaf(v, v, ss);
    }
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int k = 0; k < (int) (blockDim.x >> 5); ++k) s += red[k];
        p.rs[t][c] = rsqrtf(s / (float) n + p.eps);
    }
}
__global__ void hc8_norm_kernel(Hc8Gr p) { hc8_norm(p, blockIdx.x, blockIdx.y); }

// DOWN_ROWS rows of [q_down; q_inject] per CTA; a thread takes whole scale groups
__device__ void hc8_down(const Hc8Gr& p, int blk) {
    __shared__ float red[DOWN_THREADS / 32][DOWN_ROWS][MT];
    const int D = p.n * p.hc, gpr = D / G8, nrows = p.lr + p.ninj;
    const int8_t* q[DOWN_ROWS];
    const uint16_t* s[DOWN_ROWS];
#pragma unroll
    for (int r = 0; r < DOWN_ROWS; ++r) {
        const int k = min(blk * DOWN_ROWS + r, nrows - 1);
        if (k < p.lr) { q[r] = p.qd + (size_t) k * D; s[r] = p.sd + (size_t) k * p.sd_row; }
        else { q[r] = p.qi + (size_t) (k - p.lr) * D; s[r] = p.si + (size_t) (k - p.lr) * p.si_row; }
    }
    float acc[DOWN_ROWS][MT];
#pragma unroll
    for (int r = 0; r < DOWN_ROWS; ++r)
#pragma unroll
        for (int t = 0; t < MT; ++t) acc[r][t] = 0.0f;
    for (int g = threadIdx.x; g < gpr; g += blockDim.x) {
        const int col0 = g * G8, c = col0 / p.n;
        float wq[DOWN_ROWS][G8];
        float sc[DOWN_ROWS];
        {
            float wn[G8];
            const float4* w4 = reinterpret_cast<const float4*>(p.w_norm + col0);
#pragma unroll
            for (int j = 0; j < G8 / 4; ++j) {
                const float4 v = w4[j];
                wn[4 * j] = v.x; wn[4 * j + 1] = v.y; wn[4 * j + 2] = v.z; wn[4 * j + 3] = v.w;
            }
#pragma unroll
            for (int r = 0; r < DOWN_ROWS; ++r) {
                const uint4 a = reinterpret_cast<const uint4*>(q[r] + col0)[0];
                const uint4 b = reinterpret_cast<const uint4*>(q[r] + col0)[1];
                const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
                for (int i = 0; i < G8; ++i) wq[r][i] = s8(w[i >> 2], i & 3) * wn[i];
                sc[r] = f16f(s[r][g]);
            }
        }
#pragma unroll
        for (int t = 0; t < MT; ++t) {
            if (t < p.T) {
                const float4* x4 = reinterpret_cast<const float4*>(src_of(p, t) + col0);
                float d[DOWN_ROWS];
#pragma unroll
                for (int r = 0; r < DOWN_ROWS; ++r) d[r] = 0.0f;
#pragma unroll
                for (int j = 0; j < G8 / 4; ++j) {
                    const float4 v = x4[j];
#pragma unroll
                    for (int r = 0; r < DOWN_ROWS; ++r) {
                        d[r] = fmaf(wq[r][4 * j], v.x, d[r]);
                        d[r] = fmaf(wq[r][4 * j + 1], v.y, d[r]);
                        d[r] = fmaf(wq[r][4 * j + 2], v.z, d[r]);
                        d[r] = fmaf(wq[r][4 * j + 3], v.w, d[r]);
                    }
                }
                const float rs = p.rs[t][c];
#pragma unroll
                for (int r = 0; r < DOWN_ROWS; ++r) acc[r][t] = fmaf(sc[r] * rs, d[r], acc[r][t]);
            }
        }
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll
    for (int r = 0; r < DOWN_ROWS; ++r)
#pragma unroll
        for (int t = 0; t < MT; ++t) {
            if (t < p.T) {
                float v = acc[r][t];
                for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
                if (lane == 0) red[warp][r][t] = v;
            }
        }
    __syncthreads();
    if (threadIdx.x < DOWN_ROWS * MT) {
        const int r = threadIdx.x / MT, t = threadIdx.x % MT, k = blk * DOWN_ROWS + r;
        if (t < p.T && k < nrows) {
            float v = 0.0f;
            for (int w = 0; w < (int) (blockDim.x >> 5); ++w) v += red[w][r][t];
            if (k < p.lr) {
                const float x = v / (float) p.hc;
                p.lo[t][k] = x / (1.0f + expf(-x));
            } else {
                p.inj_out[t][k - p.lr] = v;
            }
        }
    }
}
__global__ void __launch_bounds__(DOWN_THREADS) hc8_down_kernel(Hc8Gr p) { hc8_down(p, blockIdx.x); }

// one warp per output column d: the hc rows c*n + d, lo staged in shared memory; a lane takes 4-value chunks
__device__ void hc8_up(const Hc8Gr& p, int blk) {
    __shared__ __align__(16) float slo[MT * MAX_LR];
    const int lr = p.lr;
    for (int i = threadIdx.x; i < p.T * lr; i += blockDim.x) slo[i] = p.lo[i / lr][i % lr];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int d = blk * (UP_THREADS / 32) + warp;
    if (d >= p.n) return;
    const int nch = lr / 4;   // chunks of 4 per row
    float part[MAX_HC][MT];
#pragma unroll
    for (int c = 0; c < MAX_HC; ++c)
#pragma unroll
        for (int t = 0; t < MT; ++t) part[c][t] = 0.0f;
#pragma unroll
    for (int c = 0; c < MAX_HC; ++c) {
        if (c < p.hc) {
            const size_t row = (size_t) c * p.n + d;
            const uint32_t* qr = reinterpret_cast<const uint32_t*>(p.qu + row * lr);
            const uint16_t* sr = p.su + row * p.su_row;
#pragma unroll
            for (int j = 0; j < MAX_LR / 128; ++j) {
                const int ch = lane + 32 * j;
                if (ch < nch) {
                    const uint32_t w = qr[ch];
                    const float sc = f16f(sr[ch / (G8 / 4)]);
                    const float q0 = s8(w, 0), q1 = s8(w, 1), q2 = s8(w, 2), q3 = s8(w, 3);
#pragma unroll
                    for (int t = 0; t < MT; ++t) {
                        if (t < p.T) {
                            const float4 l = *reinterpret_cast<const float4*>(slo + t * lr + 4 * ch);
                            float v = q0 * l.x;
                            v = fmaf(q1, l.y, v);
                            v = fmaf(q2, l.z, v);
                            v = fmaf(q3, l.w, v);
                            part[c][t] = fmaf(sc, v, part[c][t]);
                        }
                    }
                }
            }
        }
    }
#pragma unroll
    for (int c = 0; c < MAX_HC; ++c)
#pragma unroll
        for (int t = 0; t < MT; ++t)
            if (c < p.hc && t < p.T)
                for (int o = 16; o > 0; o >>= 1) part[c][t] += __shfl_xor_sync(0xffffffffu, part[c][t], o);
#pragma unroll
    for (int t = 0; t < MT; ++t) {
        if (t < p.T && lane == t) {
            const float* src = src_of(p, t);
            float m = 0.0f;
#pragma unroll
            for (int c = 0; c < MAX_HC; ++c) {
                if (c < p.hc) {
                    const size_t i = (size_t) c * p.n + d;
                    const float xn = src[i] * p.w_norm[i] * p.rs[t][c];
                    m = fmaf(xn, 1.0f / (1.0f + expf(-part[c][t])), m);
                }
            }
            p.mixed[t][d] = m / (float) p.hc;
        }
    }
}
__global__ void __launch_bounds__(UP_THREADS) hc8_up_kernel(Hc8Gr p) { hc8_up(p, blockIdx.x); }

}  // namespace

bool hc8_gr_read_multi(const FusedGrArgs* a, int n_tok, int64_t n_embd, int64_t hc, void* stream) {
    if (!hc8_enabled() || n_tok < 1 || a == nullptr) return false;
    Hc8View vd, vu, vi;
    if (!hc8_view(a[0].w_down, vd)) return false;
    const bool inj = a[0].w_inject != nullptr;
    const int64_t D = n_embd * hc;
    const bool have = hc8_view(a[0].w_up, vu) && (!inj || hc8_view(a[0].w_inject, vi));
    if (!have) {
        if (vd.owned || vu.owned) {
            basalt::log::error("basalt", "hc8_gr_read: a read's matrices are not all 8-bit copies");
            std::exit(1);
        }
        return false;
    }
    if (n_tok > MT || hc < 1 || hc > MAX_HC || vd.cols != D || vu.rows != D || vu.cols != vd.rows || vd.rows > MAX_LR ||
        (inj && (vi.rows != hc || vi.cols != D)) || n_embd % G8 != 0) {
        basalt::log::error("basalt", "hc8_gr_read: unsupported geometry (%d tokens, n_embd %lld, hc %lld, lr %d)", n_tok,
                     (long long) n_embd, (long long) hc, vd.rows);
        std::exit(1);
    }
    Hc8Gr p{};
    for (int t = 0; t < n_tok; ++t) {
        const FusedGrArgs& x = a[t];
        if (!x.R || !x.w_norm || !x.lo || !x.rs || !x.mixed || (inj && !x.inject_out) || x.apply != a[0].apply ||
            (x.apply && (!x.bo_prev || !x.inj_prev || !x.R_out)) || x.w_down != a[0].w_down || x.w_up != a[0].w_up ||
            x.w_inject != a[0].w_inject || x.w_norm != a[0].w_norm) {
            basalt::log::error("basalt", "hc8_gr_read: invalid arguments for token %d", t);
            std::exit(1);
        }
        p.R[t] = x.R; p.R_out[t] = x.R_out; p.bo[t] = x.bo_prev; p.inj_prev[t] = x.inj_prev;
        p.lo[t] = x.lo; p.rs[t] = x.rs; p.inj_out[t] = x.inject_out; p.mixed[t] = x.mixed;
    }
    p.w_norm = a[0].w_norm;
    p.qd = vd.q; p.sd = vd.s; p.sd_row = vd.srow;
    p.qu = vu.q; p.su = vu.s; p.su_row = vu.srow;
    if (inj) { p.qi = vi.q; p.si = vi.s; p.si_row = vi.srow; }
    p.T = n_tok; p.n = (int) n_embd; p.hc = (int) hc; p.lr = vd.rows; p.ninj = inj ? (int) hc : 0;
    p.apply = a[0].apply; p.eps = a[0].eps;
    cudaStream_t st = (cudaStream_t) stream;
    hc8_norm_kernel<<<dim3((unsigned) hc, (unsigned) n_tok), 256, 0, st>>>(p);
    hc8_down_kernel<<<(unsigned) ((p.lr + p.ninj + DOWN_ROWS - 1) / DOWN_ROWS), DOWN_THREADS, 0, st>>>(p);
    hc8_up_kernel<<<(unsigned) ((n_embd + UP_THREADS / 32 - 1) / (UP_THREADS / 32)), UP_THREADS, 0, st>>>(p);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "hc8_gr_read: %s", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}

}  // namespace basalt::kernels
