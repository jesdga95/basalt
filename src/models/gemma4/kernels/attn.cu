// src/models/gemma4/kernels/attn.cu - the attention's preparation and decode attention (basalt/models/gemma4/kernels.hpp).
//
// A warp holds one head row: lane L the dims [L * hd / 32, (L + 1) * hd / 32), so dim i and its rotary partner i + hd / 2
// sit in lanes L and L + 16. A rotation's cos and sin come from the rotary tables (RopeTab).
#include "basalt/models/gemma4/kernels.hpp"

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
__device__ __forceinline__ void load_cell(const void* base, KvType t, int64_t off, float (&v)[D]) {
    if (t == KvType::F8) {
        const auto* p = reinterpret_cast<const __nv_fp8_e4m3*>(base) + off;
#pragma unroll
        for (int i = 0; i < D; ++i) v[i] = float(p[i]);
    } else {
        const auto* p = reinterpret_cast<const __half*>(base) + off;
        if constexpr (D % 8 == 0) {
#pragma unroll
            for (int i = 0; i < D; i += 8) {
                const uint4 u = *reinterpret_cast<const uint4*>(p + i);
                const __half2* h = reinterpret_cast<const __half2*>(&u);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 f = __half22float2(h[j]);
                    v[i + 2 * j] = f.x;
                    v[i + 2 * j + 1] = f.y;
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < D; ++i) v[i] = __half2float(p[i]);
        }
    }
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

// ---- decode attention
constexpr int kAttnWarps = 4;
constexpr int kMaxChunks = 64;     // chunks a row's keys split into at most
constexpr int kMinChunk = 16;      // keys a chunk holds at least

// A row's split: its key range [lo, hi], its chunk length and count, from its own position only.
struct Split {
    int64_t lo, n;
    int64_t len;
    int chunks;
};
__device__ __forceinline__ Split split_of(int pos, int window) {
    Split s;
    s.lo = window > 0 && pos - window + 1 > 0 ? pos - window + 1 : 0;
    s.n = pos - s.lo + 1;
    int64_t len = (s.n + kMaxChunks - 1) / kMaxChunks;
    len = len < kMinChunk ? kMinChunk : (len + 31) / 32 * 32;
    s.len = len;
    s.chunks = (int) ((s.n + len - 1) / len);
    return s;
}

// grid (chunk, row * n_kv + kv head): the chunk's keys over the head group's queries; each warp takes every 4th key and
// keeps a running softmax per query head; the warps merge in order and the chunk's (max, sum, acc) per head go to the
// scratch: [row][head][chunk][hd + 2].
template <int D, int G>
__global__ void __launch_bounds__(kAttnWarps * 32) attn_chunk_kernel(Attn a, int rows) {
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    const LayerShape& sh = a.sh;
    const int hd = sh.hd;
    const int r = blockIdx.y / sh.n_kv, h = blockIdx.y % sh.n_kv;
    if (r >= rows) return;
    const int pos = a.pos[r];
    const Split sp = split_of(pos, sh.window);
    const int ch = blockIdx.x;
    if (ch >= sp.chunks) return;
    const SegKv& c = seg_kv(a.kv, r);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __shared__ float qs[G][32 * D];
    __shared__ float wm[kAttnWarps][G], wl[kAttnWarps][G];
    extern __shared__ float wacc[];   // [kAttnWarps][G][hd]
    for (int i = threadIdx.x; i < G * hd; i += blockDim.x)
        qs[i / hd][i % hd] = a.q[((size_t) r * sh.n_head + h * G + i / hd) * hd + i % hd];
    float wk[D];   // k_norm weights of this lane's dims
#pragma unroll
    for (int i = 0; i < D; ++i) wk[i] = sh.one ? a.k_norm[lane * D + i] : 1.f;
    __syncthreads();
    float m[G], l[G], acc[G][D];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        m[g] = -INFINITY;
        l[g] = 0.f;
#pragma unroll
        for (int i = 0; i < D; ++i) acc[g][i] = 0.f;
    }
    const int64_t k0 = sp.lo + (int64_t) ch * sp.len;
    const int64_t k1 = k0 + sp.len < sp.lo + sp.n ? k0 + sp.len : sp.lo + sp.n;
    for (int64_t kp = k0 + warp; kp < k1; kp += kAttnWarps) {
        const int64_t off = (((kp % c.cap) * sh.n_kv) + h) * hd + lane * D;
        float v[D], k[D];
        load_cell<D>(sh.one ? c.k : c.v, a.kv_type, off, v);
        if (sh.one) {
#pragma unroll
            for (int i = 0; i < D; ++i) k[i] = v[i] * wk[i];
            rotate<D>(k, lane, hd, sh.rot, (int) kp, a.rope);
        } else {
            load_cell<D>(c.k, a.kv_type, off, k);
        }
#pragma unroll
        for (int g = 0; g < G; ++g) {
            float s = 0.f;
#pragma unroll
            for (int i = 0; i < D; ++i) s += qs[g][lane * D + i] * k[i];
            s = warp_sum(s);
            const float mn = fmaxf(m[g], s);
            const float corr = expf(m[g] - mn), e = expf(s - mn);
            l[g] = l[g] * corr + e;
#pragma unroll
            for (int i = 0; i < D; ++i) acc[g][i] = acc[g][i] * corr + e * v[i];
            m[g] = mn;
        }
    }
    // the warps' states, merged in warp order
    if (lane == 0)
#pragma unroll
        for (int g = 0; g < G; ++g) {
            wm[warp][g] = m[g];
            wl[warp][g] = l[g];
        }
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
        for (int i = 0; i < D; ++i) wacc[((size_t) warp * G + g) * hd + lane * D + i] = acc[g][i];
    __syncthreads();
    float* out = a.scratch;
    for (int i = threadIdx.x; i < G * hd; i += blockDim.x) {
        const int g = i / hd, d = i % hd;
        float M = -INFINITY;
        for (int w = 0; w < kAttnWarps; ++w) M = fmaxf(M, wm[w][g]);
        float L = 0.f, A = 0.f;
        for (int w = 0; w < kAttnWarps; ++w) {
            const float f = wm[w][g] == -INFINITY ? 0.f : expf(wm[w][g] - M);
            L += wl[w][g] * f;
            A += wacc[((size_t) w * G + g) * hd + d] * f;
        }
        float* o = out + (((size_t) r * sh.n_head + h * G + g) * kMaxChunks + ch) * (hd + 2);
        o[2 + d] = A;
        if (d == 0) {
            o[0] = M;
            o[1] = L;
        }
    }
}

// grid (row * n_head): the row's chunks merged in chunk order.
__global__ void __launch_bounds__(128) attn_merge_kernel(Attn a, int rows) {
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    const LayerShape& sh = a.sh;
    const int hd = sh.hd;
    const int r = blockIdx.x / sh.n_head, head = blockIdx.x % sh.n_head;
    if (r >= rows) return;
    const Split sp = split_of(a.pos[r], sh.window);
    const float* part = a.scratch + ((size_t) r * sh.n_head + head) * kMaxChunks * (hd + 2);
    float M = -INFINITY;
    for (int c = 0; c < sp.chunks; ++c) M = fmaxf(M, part[(size_t) c * (hd + 2)]);
    float L = 0.f;
    for (int c = 0; c < sp.chunks; ++c) L += part[(size_t) c * (hd + 2) + 1] * expf(part[(size_t) c * (hd + 2)] - M);
    const float invL = 1.0f / L;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float A = 0.f;
        for (int c = 0; c < sp.chunks; ++c) A += part[(size_t) c * (hd + 2) + 2 + d] * expf(part[(size_t) c * (hd + 2)] - M);
        a.out[((size_t) r * sh.n_head + head) * hd + d] = A * invL;
    }
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

void attn_decode(const Attn& a, int rows, void* stream) {
    if (rows <= 0) return;
    check_shape(a.sh);
    const int64_t keys = a.sh.window > 0 && a.max_keys > a.sh.window ? a.sh.window : a.max_keys;
    int64_t len = (keys + kMaxChunks - 1) / kMaxChunks;
    len = len < kMinChunk ? kMinChunk : (len + 31) / 32 * 32;
    const unsigned chunks = (unsigned) ((keys + len - 1) / len);
    const int G = a.sh.n_head / a.sh.n_kv;
    const size_t smem = (size_t) kAttnWarps * G * a.sh.hd * sizeof(float);
    const dim3 grid(chunks, (unsigned) (rows * a.sh.n_kv));
    cudaStream_t st = (cudaStream_t) stream;
    const auto run = [&](auto kernel) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
        bk::launch_pdl(kernel, grid, dim3(kAttnWarps * 32), smem, st, a, rows);
    };
    switch (a.sh.hd * 100 + G) {
        case 25601: run(attn_chunk_kernel<8, 1>); break;
        case 25602: run(attn_chunk_kernel<8, 2>); break;
        case 25604: run(attn_chunk_kernel<8, 4>); break;
        case 25608: run(attn_chunk_kernel<8, 8>); break;
        case 51202: run(attn_chunk_kernel<16, 2>); break;
        case 51204: run(attn_chunk_kernel<16, 4>); break;
        case 51208: run(attn_chunk_kernel<16, 8>); break;
        default: throw std::invalid_argument("gemma4 attention: head width " + std::to_string(a.sh.hd) + " with " +
                                             std::to_string(G) + " query heads per key head");
    }
    check("attn_chunk");
    bk::launch_pdl(attn_merge_kernel, dim3((unsigned) (rows * a.sh.n_head)), dim3(128), 0, st, a, rows);
    check("attn_merge");
}

}  // namespace basalt::models::gemma4
