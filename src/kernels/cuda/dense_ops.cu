// src/kernels/cuda/dense_ops.cu - see include/basalt/kernels/dense_ops.hpp.
#include "basalt/kernels/dense_ops.hpp"
#include "basalt/kernels/gqa_attn.hpp"

#include <cuda_fp16.h>
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

__global__ void add_scaled_kernel(float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ scale,
                                  int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += scale ? y[i] * *scale : y[i];
}

__global__ void scale_values_kernel(float* __restrict__ y, const float* __restrict__ scale, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] *= *scale;
}

__global__ void swiglu_scaled_kernel(const float* __restrict__ g, const float* __restrict__ u, const float* __restrict__ sg,
                                     const float* __restrict__ su, float* __restrict__ h, int64_t rows, int64_t width,
                                     int64_t ld) {
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

// Block (row t, head j): j < n_head a query head, else a key/value head; one thread per head dim.
template <int HDW>
__global__ void __launch_bounds__(HDW) attn_qk_prepare_kernel(const float* __restrict__ q_full, const float* __restrict__ k,
                                                            const float* __restrict__ v, const float* __restrict__ q_norm,
                                                            const float* __restrict__ k_norm, float eps, RopeSpec rope,
                                                            const int32_t* __restrict__ pos, int n_head, int n_kv,
                                                            float* __restrict__ q_out, __half* __restrict__ k_cache,
                                                            __half* __restrict__ v_cache, int64_t cell0, int q_stride,
                                                            int64_t cap, const int32_t* __restrict__ cell0_dev, KvInt8 q8) {
    __shared__ float red[HDW / 32];
    __shared__ float xs[HDW], ks8[HDW], vs8[HDW];
    const int64_t t = blockIdx.x;
    const int j = blockIdx.y, d = threadIdx.x;
    const bool is_q = j < n_head;
    const int head = is_q ? j : j - n_head;
    const float x = is_q ? q_full[(t * n_head + head) * q_stride + d] : k[(t * n_kv + head) * HDW + d];
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
        q_out[(t * n_head + head) * HDW + d] = y;
        return;
    }
    const int64_t c0 = cell0_dev ? (int64_t) *cell0_dev : cell0;
    const int64_t cell = cap > 0 ? (c0 + t) % cap : c0 + t;
    const int64_t at = (cell * n_kv + head) * HDW + d;
    const float vv = v[(t * n_kv + head) * HDW + d];
    if (!q8.k) {
        k_cache[at] = __float2half_rn(y);
        v_cache[at] = __float2half_rn(vv);
        return;
    }
    // int8: per 64 values, scale = amax / 127 (stored fp16), codes rounded to nearest against the stored scale
    ks8[d] = y;
    vs8[d] = vv;
    __syncthreads();
    const int g0 = d & ~63;
    float ak = 0.0f, av = 0.0f;
    for (int i = 0; i < 64; ++i) {
        ak = fmaxf(ak, fabsf(ks8[g0 + i]));
        av = fmaxf(av, fabsf(vs8[g0 + i]));
    }
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

__global__ void dyn_conv2_kernel(const float* __restrict__ x, const float* __restrict__ dyn, int64_t ld_dyn,
                                 const float* __restrict__ base, float* __restrict__ out, int64_t T, int64_t H, int group) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * H) return;
    const int64_t t = i / H, c = i % H;
    const int64_t groups = H / group;
    const float* dr = dyn + t * ld_dyn;
    float acc = (base[c] + dr[c / group]) * x[i];
    if (t > 0) acc += (base[H + c] + dr[groups + c / group]) * x[i - H];
    out[i] = acc;
}

// One block of `rank` threads walks the positions; per candidate a block sum over the rank.
__global__ void candidate_path_kernel(const float* __restrict__ lp_rows, int stride, int top, const float* __restrict__ h,
                                      const uint16_t* __restrict__ pred, const uint16_t* __restrict__ succ, int rank,
                                      const int32_t* __restrict__ anchor, int n, int32_t* __restrict__ ids,
                                      float* __restrict__ conf) {
    __shared__ float red[32];
    __shared__ float score[64];
    __shared__ int32_t prev_s;
    const int r = threadIdx.x;
    if (r == 0) prev_s = *anchor;
    __syncthreads();
    for (int p = 0; p < n; ++p) {
        const float* row = lp_rows + (size_t) p * stride;
        const int32_t prev = prev_s;
        const float ph = __uint_as_float((uint32_t) pred[(size_t) prev * rank + r] << 16) * h[(size_t) p * rank + r];
        for (int k = 0; k < top; ++k) {
            const int32_t cand = __float_as_int(row[2 + 2 * k]);
            const float v = cand >= 0 ? ph * __uint_as_float((uint32_t) succ[(size_t) cand * rank + r] << 16) : 0.0f;
            const float sum = block_sum(v, red);
            if (r == 0) score[k] = cand >= 0 ? row[3 + 2 * k] + sum : -1e30f;
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
    add_scaled_kernel<<<blocks_for(n), 256, 0, (cudaStream_t) stream>>>(x, y, scale, n);
    check("add_scaled");
}

void scale_values(float* y, const float* scale, int64_t n, void* stream) {
    if (n <= 0 || scale == nullptr) return;
    scale_values_kernel<<<blocks_for(n), 256, 0, (cudaStream_t) stream>>>(y, scale, n);
    check("scale_values");
}

void swiglu_scaled(const float* g, const float* u, const float* sg, const float* su, float* h, int64_t rows,
                   int64_t width, int64_t ld_gu, void* stream) {
    if (rows <= 0) return;
    swiglu_scaled_kernel<<<blocks_for(rows * width), 256, 0, (cudaStream_t) stream>>>(g, u, sg, su, h, rows, width, ld_gu);
    check("swiglu_scaled");
}

void bf16_rows_dot(const float* x, const uint16_t* w, float* y, int64_t T, int64_t N, int64_t K, void* stream) {
    if (T <= 0) return;
    if (K % 8 != 0) { basalt::log::error("basalt", "bf16_rows_dot: K %lld", (long long) K); std::exit(1); }
    bf16_rows_dot_kernel<<<dim3((unsigned) ((T + RD_T - 1) / RD_T), (unsigned) ((N + RD_N - 1) / RD_N)), 32, 0,
                           (cudaStream_t) stream>>>(x, w, y, T, N, K);
    check("bf16_rows_dot");
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
        attn_qk_prepare_kernel<256><<<grid, 256, 0, (cudaStream_t) stream>>>(q_full, k, v, q_norm, k_norm, eps, r, pos, n_head,
            n_kv, q_out, (__half*) k_cache, (__half*) v_cache, cell0, qs, cap, cell0_dev, q8);
    else
        attn_qk_prepare_kernel<128><<<grid, 128, 0, (cudaStream_t) stream>>>(q_full, k, v, q_norm, k_norm, eps, r, pos, n_head,
            n_kv, q_out, (__half*) k_cache, (__half*) v_cache, cell0, qs, cap, cell0_dev, q8);
    check("attn_qk_prepare");
}

void dyn_conv2(const float* x, const float* dyn, int64_t ld_dyn, const float* base, float* out, int64_t T, int64_t H,
               int group, void* stream) {
    if (T <= 0) return;
    dyn_conv2_kernel<<<blocks_for(T * H), 256, 0, (cudaStream_t) stream>>>(x, dyn, ld_dyn, base, out, T, H, group);
    check("dyn_conv2");
}

void candidate_path(const float* lp_rows, int stride, int top, const float* h, const uint16_t* pred, const uint16_t* succ,
                    int rank, const int32_t* anchor, int n, int32_t* ids, float* conf, void* stream) {
    if (n <= 0) return;
    if (top > 64 || rank % 32 || rank > 1024) {
        basalt::log::error("basalt", "candidate_path: %d candidates of rank %d", top, rank);
        std::exit(1);
    }
    candidate_path_kernel<<<1, rank, 0, (cudaStream_t) stream>>>(lp_rows, stride, top, h, pred, succ, rank, anchor, n, ids, conf);
    check("candidate_path");
}

void attn_gate(const float* attn, const float* q_full, float* out, int64_t T, int n_head, void* stream) {
    const int64_t n = T * n_head * HD;
    if (n <= 0) return;
    attn_gate_kernel<<<blocks_for(n), 256, 0, (cudaStream_t) stream>>>(attn, q_full, out, n, n_head);
    check("attn_gate");
}

}  // namespace basalt::kernels
