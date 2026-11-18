// src/models/hy4flash/kernels/mla.cu - the absorbed MLA of Hy4-flash over a window or chunk of rows: the per-head Q8_0
// products, the query and cache-row preparation, and the attention over the shared latent with a per-head sink. A row's
// result does not depend on the other rows or on how many there are.
#include "basalt/models/hy4flash/kernels.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

namespace basalt::models::hy4flash {
namespace {

constexpr int kThreads = 512;
constexpr int kMaxCells = 4096;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m));
    return v;
}

// Block sum in a fixed order; every thread gets the result.
template <int N>
__device__ __forceinline__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = 0.f;
#pragma unroll
    for (int i = 0; i < N / 32; ++i) r += red[i];
    __syncthreads();
    return r;
}

template <int N>
__device__ __forceinline__ float block_max(float v, float* red) {
    v = warp_max(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = red[0];
#pragma unroll
    for (int i = 1; i < N / 32; ++i) r = fmaxf(r, red[i]);
    __syncthreads();
    return r;
}

// A cache holds rows of rank + rope values as fp32, as BF16 (the reference's own rounding of the latent rows), or as the FP8 row
// the flash-MLA kernels read: 512 E4M3 codes, four fp32 scales (one per 128 codes), 64 BF16 rope values (656 bytes).
constexpr int kFp8Codes = 512, kFp8ScaleOff = 512, kFp8RopeOff = 528;
__device__ __forceinline__ float row_at(const float* cache, size_t cell, int c, int W) { return cache[cell * W + c]; }
__device__ __forceinline__ float row_at(const uint16_t* cache, size_t cell, int c, int W) {
    return __uint_as_float((uint32_t) cache[cell * W + c] << 16);
}
__device__ __forceinline__ float row_at(const uint8_t* cache, size_t cell, int c, int) {
    const uint8_t* row = cache + cell * kFp8RowBytes;
    if (c < kFp8Codes) {
        const __half_raw h = __nv_cvt_fp8_to_halfraw(row[c], __NV_E4M3);
        return __half2float(__half(h)) * reinterpret_cast<const float*>(row + kFp8ScaleOff)[c >> 7];
    }
    return __uint_as_float((uint32_t) reinterpret_cast<const uint16_t*>(row + kFp8RopeOff)[c - kFp8Codes] << 16);
}
__device__ __forceinline__ uint16_t bf16_bits(float v) { return __bfloat16_as_ushort(__float2bfloat16_rn(v)); }

// The cell's row from the normed latent lat[rank] and the rotated rope values rp[rope] (shared memory); every thread of the
// block takes part.
__device__ __forceinline__ void store_cell(float* cache, size_t cell, const float* lat, const float* rp, const MlaArgs& a, float* red) {
    float* row = cache + cell * (a.rank + a.rope);
    for (int c = threadIdx.x; c < a.rank; c += blockDim.x) row[c] = lat[c];
    for (int c = threadIdx.x; c < a.rope; c += blockDim.x) row[a.rank + c] = rp[c];
}
__device__ __forceinline__ void store_cell(uint16_t* cache, size_t cell, const float* lat, const float* rp, const MlaArgs& a, float* red) {
    uint16_t* row = cache + cell * (a.rank + a.rope);
    for (int c = threadIdx.x; c < a.rank; c += blockDim.x) row[c] = bf16_bits(lat[c]);
    for (int c = threadIdx.x; c < a.rope; c += blockDim.x) row[a.rank + c] = bf16_bits(rp[c]);
}
__device__ __forceinline__ void store_cell(uint8_t* cache, size_t cell, const float* lat, const float* rp, const MlaArgs& a, float* red) {
    uint8_t* row = cache + cell * kFp8RowBytes;
    // a scale per 128 latent values: amax / 448 (a 512-thread block: a quarter each)
    const int c = threadIdx.x, g = c >> 7;
    float amax = c < kFp8Codes ? fabsf(lat[c]) : 0.f;
    amax = warp_max(amax);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = amax;
    __syncthreads();
    const float m = fmaxf(fmaxf(red[g * 4], red[g * 4 + 1]), fmaxf(red[g * 4 + 2], red[g * 4 + 3]));
    const float scale = m > 0.f ? m / 448.f : 1.f;
    if (c < kFp8Codes) row[c] = __nv_cvt_float_to_fp8(lat[c] / scale, __NV_SATFINITE, __NV_E4M3);
    if ((c & 127) == 0 && c < kFp8Codes) reinterpret_cast<float*>(row + kFp8ScaleOff)[g] = scale;
    for (int i = threadIdx.x; i < a.rope; i += blockDim.x) reinterpret_cast<uint16_t*>(row + kFp8RopeOff)[i] = bf16_bits(rp[i]);
    __syncthreads();
}

// A warp per output row; z = the token. The activation is rounded to q8_1 per 32 values, as every other product of the model's
// Q8_0 weights does: d = amax / 127, codes round half away from zero, the block's dot is an integer sum times d_w * d.
__global__ void heads_gemv_kernel(const uint8_t* __restrict__ w, const float* __restrict__ x, int x_head_stride, int x_tok_stride,
                                  int rows, int nb, int heads, float* __restrict__ y) {
    const int h = blockIdx.y, t = blockIdx.z, lane = threadIdx.x & 31;
    const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r >= rows) return;
    const uint8_t* row = w + ((size_t) h * rows + r) * nb * 34;
    const float* xv = x + (size_t) t * x_tok_stride + (size_t) h * x_head_stride;
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = row + (size_t) b * 34;
        const float dw = __half2float(*reinterpret_cast<const __half*>(blk));
        const int qw = (int8_t) blk[2 + lane];
        const float v = xv[b * 32 + lane];
        const float amax = warp_max(fabsf(v));
        const float d = amax / 127.f;
        const float id = d != 0.f ? 1.f / d : 0.f;
        const int qx = __float2int_rn(roundf(v * id));
        int isum = qw * qx;
#pragma unroll
        for (int m = 16; m; m >>= 1) isum += __shfl_xor_sync(0xffffffffu, isum, m);
        acc = fmaf(dw * d, (float) isum, acc);
    }
    if (lane == 0) y[((size_t) t * heads + h) * rows + r] = acc;
}

// Rotation of consecutive pairs (2i, 2i + 1) by pos * base^(-2i / n_rot).
__device__ __forceinline__ void rope_pair(float x0, float x1, int i, int n_rot, float base, int pos, float& o0, float& o1) {
    const float theta = (float) pos * powf(base, -(float) (2 * i) / (float) n_rot);
    float s, c;
    sincosf(theta, &s, &c);
    o0 = x0 * c - x1 * s;
    o1 = x0 * s + x1 * c;
}

// The position pair i of the rotated part of the row at cell `cell` turns by (see MlaArgs::cell_rope).
template <typename A>
__device__ __forceinline__ int axis_pos(const A& a, int cell, int i) {
    if (a.cell_rope == nullptr) return cell;
    const int32_t* p = a.cell_rope + (size_t) cell * 3;
    if (i % 3 == 1 && i < 3 * a.sec_h) return p[1];
    if (i % 3 == 2 && i < 3 * a.sec_w) return p[2];
    return p[0];
}

// Block h < heads: the head's query row; block `heads`: the cache row. y = the token.
template <typename C>
__global__ void prepare_kernel(const float* __restrict__ q, const float* __restrict__ q_abs, const float* __restrict__ kv_raw,
                               const float* __restrict__ kv_norm, MlaArgs a, int pos0, int ring, float* __restrict__ qcur,
                               C* __restrict__ cache) {
    __shared__ float red[kThreads / 32];
    __shared__ float lat[512], rp[64];
    const int h = blockIdx.x, t = blockIdx.y, W = a.rank + a.rope, Q = a.nope + a.rope, pos = pos0 + t;
    if (h < a.heads) {
        const float* qt = q + (size_t) t * a.heads * Q;
        const float* qa = q_abs + (size_t) t * a.heads * a.rank;
        float* out = qcur + (size_t) t * a.heads * W;
        for (int c = threadIdx.x; c < a.rank; c += blockDim.x) out[(size_t) h * W + c] = qa[(size_t) h * a.rank + c];
        if (threadIdx.x < a.rope / 2) {
            const int i = threadIdx.x;
            const float* p = qt + (size_t) h * Q + a.nope + 2 * i;
            float o0, o1;
            rope_pair(p[0], p[1], i, a.rope, a.rope_base, axis_pos(a, pos, i), o0, o1);
            out[(size_t) h * W + a.rank + 2 * i] = o0;
            out[(size_t) h * W + a.rank + 2 * i + 1] = o1;
        }
        return;
    }
    const float* kr = kv_raw + (size_t) t * W;
    float ss = 0.f;
    for (int c = threadIdx.x; c < a.rank; c += blockDim.x) ss = fmaf(kr[c], kr[c], ss);
    ss = block_sum<kThreads>(ss, red);
    const float rs = rsqrtf(ss / (float) a.rank + a.rms_eps);
    for (int c = threadIdx.x; c < a.rank; c += blockDim.x) lat[c] = kr[c] * rs * kv_norm[c];
    if (threadIdx.x < a.rope / 2) {
        const int i = threadIdx.x;
        float o0, o1;
        rope_pair(kr[a.rank + 2 * i], kr[a.rank + 2 * i + 1], i, a.rope, a.rope_base, axis_pos(a, pos, i), o0, o1);
        rp[2 * i] = o0;
        rp[2 * i + 1] = o1;
    }
    __syncthreads();
    store_cell(cache, (size_t) (ring ? pos % ring : pos), lat, rp, a, red);
}

// A block per head and row: the scores (a warp per cell), the softmax with the sink, the weighted sum of the latents.
template <typename C>
__global__ void attend_kernel(const float* __restrict__ qcur, const C* __restrict__ cache, const int32_t* __restrict__ firsts,
                              const int32_t* __restrict__ counts, const int32_t* __restrict__ cells, int cells_stride,
                              const float* __restrict__ sinks, MlaArgs a, int ring, float* __restrict__ o_lat) {
    __shared__ float s[kMaxCells];
    __shared__ float red[kThreads / 32];
    const int h = blockIdx.x, t = blockIdx.y, W = a.rank + a.rope, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int n = counts[t], first = firsts[t];
    const int32_t* cl = cells ? cells + (size_t) t * cells_stride : nullptr;
    const float* qv = qcur + ((size_t) t * a.heads + h) * W;
    for (int j = warp; j < n; j += kThreads / 32) {
        const int cell = cl ? cl[j] : ring ? (first + j) % ring : first + j;
        float acc = 0.f;
        for (int c = lane; c < W; c += 32) acc = fmaf(qv[c], row_at(cache, (size_t) cell, c, W), acc);
        acc = warp_sum(acc);
        if (lane == 0) s[j] = acc * a.scale;
    }
    __syncthreads();
    const float sink = sinks[h];
    float m = sink;
    for (int j = threadIdx.x; j < n; j += kThreads) m = fmaxf(m, s[j]);
    m = block_max<kThreads>(m, red);
    float den = 0.f;
    for (int j = threadIdx.x; j < n; j += kThreads) {
        const float p = expf(s[j] - m);
        s[j] = p;
        den += p;
    }
    den = block_sum<kThreads>(den, red);
    den += expf(sink - m);
    const float inv = 1.f / den;
    for (int c = threadIdx.x; c < a.rank; c += kThreads) {
        float acc = 0.f;
        for (int j = 0; j < n; ++j) {
            const int cell = cl ? cl[j] : ring ? (first + j) % ring : first + j;
            acc = fmaf(s[j], row_at(cache, (size_t) cell, c, W), acc);
        }
        o_lat[((size_t) t * a.heads + h) * a.rank + c] = acc * inv;
    }
}

// ---- tensor-core attention over a BF16 cache -------------------------------------------------------------------------------
// A block per row and 16 heads, 8 warps. The row's cells go through shared memory 32 at a time, as fp16 (a BF16 latent row is
// exact in fp16 within its range); S = Q K^T and O += P V run on mma.m16n8k16 with the scaled query and the probabilities in
// fp16, accumulated in fp32, as llama.cpp's flash attention does. The softmax is a running one, the sink its first element.
constexpr int kMmaCells = 32, kLd = 584, kPLd = 40;   // smem row strides in fp16 elements (avoid bank conflicts)

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"((uint32_t) __cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"((uint32_t) __cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void mma16816h(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t bf2_to_h2(uint32_t v) {
    const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v));
    const __half2 h = __float22half2_rn(f);
    return *reinterpret_cast<const uint32_t*>(&h);
}

__global__ void __launch_bounds__(256) attend_mma_kernel(const float* __restrict__ qcur, const uint16_t* __restrict__ cache, const int32_t* __restrict__ firsts,
                                                         const int32_t* __restrict__ counts, const int32_t* __restrict__ cells, int cells_stride,
                                                         const float* __restrict__ sinks, MlaArgs a, int ring, float* __restrict__ o_lat) {
    extern __shared__ __align__(16) unsigned char smraw[];
    constexpr int W = 576;
    uint16_t* qh = (uint16_t*) smraw;
    uint16_t* ks = qh + 16 * kLd;
    uint16_t* ph = ks + kMmaCells * kLd;
    float* sp = (float*) (ph + 16 * kPLd);   // [2][16][32]
    float* alpha_s = sp + 2 * 16 * 32;
    float* l_s = alpha_s + 16;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    const int t = blockIdx.y, h0 = blockIdx.x * 16;
    const int n = counts[t], first = firsts[t];
    const int32_t* cl = cells ? cells + (size_t) t * cells_stride : nullptr;
    for (int i = tid; i < 16 * W; i += 256) {
        const int h = i / W, e = i - h * W;
        qh[h * kLd + e] = __half_as_ushort(__float2half_rn(qcur[((size_t) t * a.heads + h0) * W + i] * a.scale));
    }
    float m[2], l[2];
    for (int hh = 0; hh < 2; ++hh) {
        m[hh] = sinks[h0 + w * 2 + hh];
        l[hh] = 1.f;
    }
    if (tid < 16) l_s[tid] = 1.f;
    float acc[8][4];
    for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    const int nt = w & 3, kh = w >> 2, id = lane >> 3, r8 = lane & 7;
    for (int j0 = 0; j0 < n; j0 += kMmaCells) {
        const int tn = min(kMmaCells, n - j0);
        __syncthreads();
        for (int i = tid; i < kMmaCells * (W / 8); i += 256) {
            const int c = i / (W / 8), ch = i - c * (W / 8);
            uint4 v = make_uint4(0, 0, 0, 0);
            if (c < tn) {
                const int j = j0 + c;
                const int cell = cl ? cl[j] : ring ? (first + j) % ring : first + j;
                v = *reinterpret_cast<const uint4*>(cache + (size_t) cell * W + ch * 8);
                v = make_uint4(bf2_to_h2(v.x), bf2_to_h2(v.y), bf2_to_h2(v.z), bf2_to_h2(v.w));
            }
            *reinterpret_cast<uint4*>(ks + c * kLd + ch * 8) = v;
        }
        __syncthreads();
        // S: this warp's 8 cells and half of the 576 dimensions
        {
            float c[4] = {0.f, 0.f, 0.f, 0.f};
            for (int kk = 0; kk < 9; ++kk) {
                const int kb = (kh * 18 + kk * 2) * 16;
                uint32_t b[4];
                ldsm4(b, ks + (nt * 8 + r8) * kLd + kb + id * 8);
                for (int s2 = 0; s2 < 2; ++s2) {
                    const int kd = kb + s2 * 16;
                    uint32_t ah[4];
                    ldsm4(ah, qh + ((id & 1) * 8 + r8) * kLd + kd + (id >> 1) * 8);
                    mma16816h(c, ah, b[2 * s2], b[2 * s2 + 1]);
                }
            }
            const int row = lane >> 2, col = nt * 8 + (lane & 3) * 2;
            sp[(kh * 16 + row) * 32 + col] = c[0];
            sp[(kh * 16 + row) * 32 + col + 1] = c[1];
            sp[(kh * 16 + row + 8) * 32 + col] = c[2];
            sp[(kh * 16 + row + 8) * 32 + col + 1] = c[3];
        }
        __syncthreads();
        // the running softmax: a warp takes 2 heads, a lane a cell
        for (int hh = 0; hh < 2; ++hh) {
            const int h = w * 2 + hh;
            const bool valid = lane < tn;
            const float sc = valid ? sp[h * 32 + lane] + sp[(16 + h) * 32 + lane] : -INFINITY;
            float mx = sc;
            for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float nm = fmaxf(m[hh], mx);
            const float alpha = expf(m[hh] - nm);
            const float p = valid ? expf(sc - nm) : 0.f;
            float sum = p;
            for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            l[hh] = l[hh] * alpha + sum;
            m[hh] = nm;
            ph[h * kPLd + lane] = __half_as_ushort(__float2half_rn(p));
            if (lane == 0) {
                alpha_s[h] = alpha;
                l_s[h] = l[hh];
            }
        }
        __syncthreads();
        // O = O * alpha + P V: this warp's 64 latent dimensions
        {
            const float a0 = alpha_s[lane >> 2], a1 = alpha_s[(lane >> 2) + 8];
            for (int j = 0; j < 8; ++j) {
                acc[j][0] *= a0; acc[j][1] *= a0;
                acc[j][2] *= a1; acc[j][3] *= a1;
            }
            for (int s2 = 0; s2 < 2; ++s2) {
                uint32_t ah[4];
                ldsm4(ah, ph + ((id & 1) * 8 + r8) * kPLd + s2 * 16 + (id >> 1) * 8);
                for (int jj = 0; jj < 4; ++jj) {
                    uint32_t b[4];
                    ldsm4t(b, ks + (s2 * 16 + (id & 1) * 8 + r8) * kLd + w * 64 + jj * 16 + (id >> 1) * 8);
                    mma16816h(acc[2 * jj], ah, b[0], b[1]);
                    mma16816h(acc[2 * jj + 1], ah, b[2], b[3]);
                }
            }
        }
    }
    __syncthreads();
    const int row = lane >> 2;
    const float i0 = 1.f / l_s[row], i1 = 1.f / l_s[row + 8];
    for (int j = 0; j < 8; ++j) {
        const int d = w * 64 + j * 8 + (lane & 3) * 2;
        float* o0 = o_lat + ((size_t) t * a.heads + h0 + row) * a.rank + d;
        float* o1 = o_lat + ((size_t) t * a.heads + h0 + row + 8) * a.rank + d;
        o0[0] = acc[j][0] * i0; o0[1] = acc[j][1] * i0;
        o1[0] = acc[j][2] * i1; o1[1] = acc[j][3] * i1;
    }
}


__global__ void head_major_ids_kernel(int32_t* ids, int32_t* bounds, int heads, int rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < heads * rows) ids[i] = (i % rows) * heads + i / rows;
    if (i <= heads) bounds[i] = i * rows;
}

// cells[t][j] = j for j < fill[t] (rows with fill 0 are left alone).
__global__ void iota_cells_kernel(int32_t* __restrict__ cells, int stride, const int32_t* __restrict__ fill) {
    const int t = blockIdx.x, n = fill[t];
    for (int j = threadIdx.x; j < n; j += blockDim.x) cells[(size_t) t * stride + j] = j;
}

template <typename C>
void prepare_impl(const float* q, const float* q_abs, const float* kv_raw, const float* kv_norm, const MlaArgs& a, int pos0,
                  int n_tok, int ring, float* qcur, C* cache, void* stream) {
    prepare_kernel<C><<<dim3(a.heads + 1, n_tok), kThreads, 0, (cudaStream_t) stream>>>(q, q_abs, kv_raw, kv_norm, a, pos0, ring, qcur, cache);
    if (!stream) cudaStreamSynchronize(nullptr);
}

template <typename C>
void attend_impl(const float* qcur, const C* cache, const RowSpans& r, int n_tok, const float* sinks, const MlaArgs& a, int ring,
                 float* o_lat, void* stream) {
    if constexpr (std::is_same_v<C, unsigned short>) {
        if (n_tok > 8 && a.rank == 512 && a.rope == 64 && a.heads % 16 == 0) {   // a chunk of rows; a window keeps the kernel below
            const size_t smem = (size_t) (16 * kLd + kMmaCells * kLd + 16 * kPLd) * 2 + (2 * 16 * 32 + 32) * sizeof(float);
            static bool raised_mma = false;
            if (!raised_mma) {
                cudaFuncSetAttribute(attend_mma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
                raised_mma = true;
            }
            attend_mma_kernel<<<dim3(a.heads / 16, n_tok), 256, smem, (cudaStream_t) stream>>>(qcur, cache, r.firsts, r.counts, r.cells, r.cells_stride, sinks, a, ring, o_lat);
            if (!stream) cudaStreamSynchronize(nullptr);
            return;
        }
    }
    attend_kernel<C><<<dim3(a.heads, n_tok), kThreads, 0, (cudaStream_t) stream>>>(qcur, cache, r.firsts, r.counts, r.cells, r.cells_stride,
                                                                               sinks, a, ring, o_lat);
    if (!stream) cudaStreamSynchronize(nullptr);
}

}  // namespace

void head_major_ids(int32_t* ids, int32_t* bounds, int heads, int rows, void* stream) {
    const int n = std::max(heads * rows, heads + 1);
    head_major_ids_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(ids, bounds, heads, rows);
    if (!stream) cudaStreamSynchronize(nullptr);
}

void q8_0_heads_gemv(const void* w, const float* x, int x_head_stride, int x_tok_stride, int heads, int rows, int k, int n_tok,
                     float* y, void* stream) {
    if (k % 32) throw std::invalid_argument("q8_0_heads_gemv: k is not a multiple of 32");
    heads_gemv_kernel<<<dim3((rows + 7) / 8, heads, n_tok), 256, 0, (cudaStream_t) stream>>>((const uint8_t*) w, x, x_head_stride,
                                                                                         x_tok_stride, rows, k / 32, heads, y);
    if (!stream) cudaStreamSynchronize(nullptr);
}

void mla_prepare(const float* q, const float* q_abs, const float* kv_raw, const float* kv_norm, const MlaArgs& a, int pos0,
                 int n_tok, int ring, float* qcur, float* cache, void* stream) {
    prepare_impl(q, q_abs, kv_raw, kv_norm, a, pos0, n_tok, ring, qcur, cache, stream);
}
void mla_prepare(const float* q, const float* q_abs, const float* kv_raw, const float* kv_norm, const MlaArgs& a, int pos0,
                 int n_tok, int ring, float* qcur, uint16_t* cache, void* stream) {
    prepare_impl(q, q_abs, kv_raw, kv_norm, a, pos0, n_tok, ring, qcur, cache, stream);
}
void mla_attend(const float* qcur, const float* cache, const RowSpans& r, int n_tok, const float* sinks, const MlaArgs& a, int ring,
                float* o_lat, void* stream) {
    attend_impl(qcur, cache, r, n_tok, sinks, a, ring, o_lat, stream);
}
void mla_attend(const float* qcur, const uint16_t* cache, const RowSpans& r, int n_tok, const float* sinks, const MlaArgs& a, int ring,
                float* o_lat, void* stream) {
    attend_impl(qcur, cache, r, n_tok, sinks, a, ring, o_lat, stream);
}

void mla_prepare(const float* q, const float* q_abs, const float* kv_raw, const float* kv_norm, const MlaArgs& a, int pos0,
                 int n_tok, int ring, float* qcur, uint8_t* cache, void* stream) {
    prepare_impl(q, q_abs, kv_raw, kv_norm, a, pos0, n_tok, ring, qcur, cache, stream);
}
void mla_attend(const float* qcur, const uint8_t* cache, const RowSpans& r, int n_tok, const float* sinks, const MlaArgs& a, int ring,
                float* o_lat, void* stream) {
    attend_impl(qcur, cache, r, n_tok, sinks, a, ring, o_lat, stream);
}

void iota_cells(int32_t* cells, int stride, const int32_t* fill, int n_tok, void* stream) {
    iota_cells_kernel<<<n_tok, 256, 0, (cudaStream_t) stream>>>(cells, stride, fill);
    if (!stream) cudaStreamSynchronize(nullptr);
}

}  // namespace basalt::models::hy4flash
