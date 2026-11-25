// src/models/gemma4/kernels/attn.cu - the attention's preparation, the rotary tables and the decode attention's merge
// (basalt/models/gemma4/kernels.hpp; the chunks in attn_mma.cu).
//
// A warp holds one head row: lane L the dims [L * hd / 32, (L + 1) * hd / 32), so dim i and its rotary partner i + hd / 2
// sit in lanes L and L + 16. A rotation's cos and sin come from the rotary tables (RopeTab).
#include "basalt/models/gemma4/kernels.hpp"

#include "attn_split.cuh"
#include "basalt/kernels/bf16_bits.hpp"
#include "basalt/kernels/pdl.hpp"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace basalt::models::gemma4 {
namespace {

namespace bk = basalt::kernels;

void check(const char* what) {
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string("gemma4 ") + what + ": " + cudaGetErrorString(e));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ const SegKv& seg_kv(const KvSegs& kv, int row) {
    return kv.c[kv.at.n > 0 ? bk::seg_of(kv.at, row) : 0];
}

template <int D>
__device__ __forceinline__ void store_cell(void* base, KvType t, int64_t off, const float (&v)[D]) {
    if (t == KvType::F8) {
        auto* p = reinterpret_cast<__nv_fp8_e4m3*>(base) + off;
#pragma unroll
        for (int i = 0; i < D; ++i) p[i] = __nv_fp8_e4m3(v[i]);
    } else {
        auto* p = reinterpret_cast<__half*>(base) + off;
#pragma unroll
        for (int i = 0; i < D; ++i) p[i] = __float2half_rn(v[i]);
    }
}

// The rotation of a head row held across the warp: lanes 0..15 hold the first half's dims, 16..31 their partners. Dim i
// (i < rot) becomes x_i cos - x_{i+h} sin, its partner x_{i+h} cos + x_i sin, the angle's cos and sin from the tables.
template <int D>
__device__ __forceinline__ void rotate(float (&x)[D], int lane, int hd, int rot, int pos, const RopeTab& t) {
    float other[D];
#pragma unroll
    for (int i = 0; i < D; ++i) other[i] = __shfl_xor_sync(0xffffffffu, x[i], 16);
    const bool hi = lane >= 16;
    const float2* lo_row = reinterpret_cast<const float2*>(t.lo) + (size_t) (pos & 255) * (hd / 2);
    const float2* hi_row = reinterpret_cast<const float2*>(t.hi) + (size_t) (pos >> 8) * (hd / 2);
#pragma unroll
    for (int i = 0; i < D; ++i) {
        const int d = (lane & 15) * D + i;   // the pair's first dim
        if (d >= rot) continue;
        const float2 a = hi_row[d], b = lo_row[d];
        const float c = a.x * b.x - a.y * b.y, s = a.y * b.x + a.x * b.y;
        x[i] = hi ? x[i] * c + other[i] * s : x[i] * c - other[i] * s;
    }
}

// ---- preparation: one warp per (row, item); items are the q heads, then the k heads, then the v heads
template <int D>
__global__ void __launch_bounds__(128) prep_kernel(Prep p, int rows) {
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    const int lane = threadIdx.x & 31;
    const int item = blockIdx.x * 4 + (threadIdx.x >> 5), r = blockIdx.y;
    const LayerShape& sh = p.sh;
    const int nv = sh.one ? 0 : sh.n_kv;
    if (item >= (p.q_only ? sh.n_head : sh.n_head + sh.n_kv + nv) || r >= rows) return;
    const int hd = sh.hd;
    const int pos = p.pos[r];
    const SegKv& c = seg_kv(p.kv, r);
    const int64_t cell = (int64_t) pos % c.cap;
    float x[D];
    const float* src;
    const float* w = nullptr;
    int kind;   // 0 q, 1 k, 2 v
    int head;
    if (item < sh.n_head) {
        kind = 0, head = item, src = p.q + (size_t) r * p.ldq + (size_t) head * hd, w = p.q_norm;
    } else if (item < sh.n_head + sh.n_kv) {
        kind = 1, head = item - sh.n_head, src = p.k + (size_t) r * p.ldk + (size_t) head * hd, w = sh.one ? nullptr : p.k_norm;
    } else {
        kind = 2, head = item - sh.n_head - sh.n_kv, src = p.v + (size_t) r * p.ldv + (size_t) head * hd;
    }
#pragma unroll
    for (int i = 0; i < D; ++i) x[i] = src[lane * D + i];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < D; ++i) ss += x[i] * x[i];
    const float inv = 1.0f / sqrtf(warp_sum(ss) / (float) hd + p.eps);
#pragma unroll
    for (int i = 0; i < D; ++i) x[i] = w ? x[i] * inv * w[lane * D + i] : x[i] * inv;
    if (kind == 0 || (kind == 1 && !sh.one)) rotate<D>(x, lane, hd, sh.rot, pos, p.rope);
    const int64_t off = (cell * sh.n_kv + head) * hd + lane * D;
    if (kind == 0) {
        float* o = p.q_out + ((size_t) r * sh.n_head + head) * hd + lane * D;
#pragma unroll
        for (int i = 0; i < D; ++i) o[i] = x[i];
    } else if (kind == 1) {
        store_cell<D>(c.k, p.kv_type, off, x);
    } else {
        store_cell<D>(c.v, p.kv_type, off, x);
    }
}

// ---- decode attention: the chunks on the tensor cores (attn_mma.cu), then their merge

// grid (row * n_head): the row's chunks merged in chunk order, each chunk's weight exp(max - M) computed once.
__global__ void __launch_bounds__(128) attn_merge_kernel(Attn a, int rows) {
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    __shared__ float wt[kMaxChunks];
    __shared__ float inv_l;
    const LayerShape& sh = a.sh;
    const int hd = sh.hd;
    const int r = blockIdx.x / sh.n_head, head = blockIdx.x % sh.n_head;
    if (r >= rows) return;
    const Split sp = split_of(a.pos[r], sh.window);
    const float* part = a.scratch + ((size_t) r * sh.n_head + head) * kMaxChunks * (hd + 2);
    if (threadIdx.x == 0) {
        float M = -INFINITY;
        for (int c = 0; c < sp.chunks; ++c) M = fmaxf(M, part[(size_t) c * (hd + 2)]);
        float L = 0.f;
        for (int c = 0; c < sp.chunks; ++c) {
            wt[c] = expf(part[(size_t) c * (hd + 2)] - M);
            L += part[(size_t) c * (hd + 2) + 1] * wt[c];
        }
        inv_l = 1.0f / L;
    }
    __syncthreads();
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float A = 0.f;
        for (int c = 0; c < sp.chunks; ++c) A += part[(size_t) c * (hd + 2) + 2 + d] * wt[c];
        const size_t at = ((size_t) r * sh.n_head + head) * hd + d;
        if (a.act) static_cast<uint16_t*>(a.act)[at] = bk::bf16_from_f32(A * inv_l);
        else a.out[at] = A * inv_l;
    }
}

// grid (key, kv head), a thread two rotary pairs: V widened, K = rope(V * k_norm), both rounded to fp16.
template <int HD>
__global__ void __launch_bounds__(HD / 4) derive_keys_kernel(SegKv kv, const float* __restrict__ k_norm, RopeTab rope, int n_kv,
                                                             int rot, bool f8, __half* __restrict__ k16, __half* __restrict__ v16) {
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    constexpr int H = HD / 2;
    const int key = blockIdx.x, h = blockIdx.y, i = threadIdx.x * 2;
    const size_t off = ((size_t) (key % kv.cap) * n_kv + h) * HD;
    const float2* lo = reinterpret_cast<const float2*>(rope.lo) + (size_t) (key & 255) * H;
    const float2* hi = reinterpret_cast<const float2*>(rope.hi) + (size_t) (key >> 8) * H;
    float x[2], y[2];
#pragma unroll
    for (int c = 0; c < 2; ++c) {
        if (f8) {
            x[c] = float(reinterpret_cast<const __nv_fp8_e4m3*>(kv.k)[off + i + c]);
            y[c] = float(reinterpret_cast<const __nv_fp8_e4m3*>(kv.k)[off + i + H + c]);
        } else {
            x[c] = __half2float(reinterpret_cast<const __half*>(kv.k)[off + i + c]);
            y[c] = __half2float(reinterpret_cast<const __half*>(kv.k)[off + i + H + c]);
        }
    }
    if (v16) {
        *reinterpret_cast<__half2*>(v16 + off + i) = __floats2half2_rn(x[0], x[1]);
        *reinterpret_cast<__half2*>(v16 + off + i + H) = __floats2half2_rn(y[0], y[1]);
    }
#pragma unroll
    for (int c = 0; c < 2; ++c) {
        const float kx = x[c] * k_norm[i + c], ky = y[c] * k_norm[i + H + c];
        if (i + c < rot) {
            const float2 p = hi[i + c], q = lo[i + c];
            const float co = p.x * q.x - p.y * q.y, si = p.y * q.x + p.x * q.y;
            x[c] = kx * co - ky * si;
            y[c] = ky * co + kx * si;
        } else {
            x[c] = kx;
            y[c] = ky;
        }
    }
    *reinterpret_cast<__half2*>(k16 + off + i) = __floats2half2_rn(x[0], x[1]);
    *reinterpret_cast<__half2*>(k16 + off + i + H) = __floats2half2_rn(y[0], y[1]);
}

void check_shape(const LayerShape& sh) {
    if ((sh.hd != 256 && sh.hd != 512) || sh.n_kv < 1 || sh.n_head % sh.n_kv || sh.n_head > kMaxHeads ||
        sh.n_head / sh.n_kv > kMaxGroup || sh.rot < 0 || sh.rot > sh.hd / 2)
        throw std::invalid_argument("gemma4 attention: unsupported shape");
}

}  // namespace

void attn_prep(const Prep& p, int rows, void* stream) {
    if (rows <= 0) return;
    check_shape(p.sh);
    const int items = p.q_only ? p.sh.n_head : p.sh.n_head + p.sh.n_kv + (p.sh.one ? 0 : p.sh.n_kv);
    const dim3 grid((unsigned) ((items + 3) / 4), (unsigned) rows);
    if (p.sh.hd == 256) bk::launch_pdl(prep_kernel<8>, grid, dim3(128), 0, (cudaStream_t) stream, p, rows);
    else bk::launch_pdl(prep_kernel<16>, grid, dim3(128), 0, (cudaStream_t) stream, p, rows);
    check("attn_prep");
}

void rope_tables(int hd, int rot, double base, int64_t positions, float* lo, float* hi) {
    const int h = hd / 2;
    for (int i = 0; i < h; ++i) {
        const double f = i < rot ? std::pow(base, -2.0 * i / hd) : 0.0;
        for (int b = 0; b < 256; ++b) {
            lo[((size_t) b * h + i) * 2] = (float) std::cos(b * f);
            lo[((size_t) b * h + i) * 2 + 1] = (float) std::sin(b * f);
        }
        for (int64_t a = 0; a < (positions + 255) / 256; ++a) {
            hi[((size_t) a * h + i) * 2] = (float) std::cos((double) (a * 256) * f);
            hi[((size_t) a * h + i) * 2 + 1] = (float) std::sin((double) (a * 256) * f);
        }
    }
}

size_t attn_scratch_floats(int rows, int n_head, int hd) { return (size_t) rows * n_head * kMaxChunks * (hd + 2); }

void derive_keys(const SegKv& kv, const float* k_norm, RopeTab rope, const LayerShape& sh, KvType kv_type, int64_t n,
                 void* k16, void* v16, void* stream) {
    if (n <= 0) return;
    check_shape(sh);
    if (!sh.one || (kv_type == KvType::F8) != (v16 != nullptr)) throw std::invalid_argument("gemma4 derive_keys: not a layer whose keys are its values, or v16 against the cache type");
    const dim3 grid((unsigned) n, (unsigned) sh.n_kv);
    const bool f8 = kv_type == KvType::F8;
    if (sh.hd == 256)
        bk::launch_pdl(derive_keys_kernel<256>, grid, dim3(64), 0, (cudaStream_t) stream, kv, k_norm, rope, sh.n_kv, sh.rot, f8,
                       (__half*) k16, (__half*) v16);
    else
        bk::launch_pdl(derive_keys_kernel<512>, grid, dim3(128), 0, (cudaStream_t) stream, kv, k_norm, rope, sh.n_kv, sh.rot, f8,
                       (__half*) k16, (__half*) v16);
    check("derive_keys");
}

void attn_decode(const Attn& a, int rows, void* stream) {
    if (rows <= 0) return;
    check_shape(a.sh);
    cudaStream_t st = (cudaStream_t) stream;
    attn_chunks(a, rows, st);
    bk::launch_pdl(attn_merge_kernel, dim3((unsigned) (rows * a.sh.n_head)), dim3(128), 0, st, a, rows);
    check("attn_merge");
}

}  // namespace basalt::models::gemma4
