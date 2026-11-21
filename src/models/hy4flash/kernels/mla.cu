// src/models/hy4flash/kernels/mla.cu - the absorbed MLA of Hy4-flash over a window or chunk of rows: the per-head Q8_0
// products, the query and cache-row preparation, and the attention over the shared latent with a per-head sink. A row's
// result does not depend on the other rows or on how many there are.
#include "basalt/kernels/mqa_chunk.cuh"
#include "basalt/models/hy4flash/kernels.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace basalt::models::hy4flash {
namespace {

constexpr int kThreads = 512;
constexpr int kAttTok = 8;   // rows of one chunked-attention launch
constexpr int kMaxChunks = 128;   // chunks of a row (8,192 cells)
namespace mqa = basalt::kernels::mqa;

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

// A cache holds rows of rank + rope values as fp32, as fp16 (the BF16 reference's llama.cpp cache rounding; the latent and rope values
// of the model stay far inside fp16's range), or as the FP8 row the flash-MLA kernels read: 512 E4M3 codes, four fp32 scales (one per
// 128 codes), 64 BF16 rope values (656 bytes).
constexpr int kFp8Codes = 512, kFp8ScaleOff = 512, kFp8RopeOff = 528;
__device__ __forceinline__ uint16_t bf16_bits(float v) { return __bfloat16_as_ushort(__float2bfloat16_rn(v)); }
__device__ __forceinline__ uint16_t f16_bits(float v) { return __half_as_ushort(__float2half_rn(v)); }

// The cell's row from the normed latent lat[rank] and the rotated rope values rp[rope] (shared memory); every thread of the
// block takes part.
__device__ __forceinline__ void store_cell(float* cache, size_t cell, const float* lat, const float* rp, const MlaArgs& a, float* red) {
    float* row = cache + cell * (a.rank + a.rope);
    for (int c = threadIdx.x; c < a.rank; c += blockDim.x) row[c] = lat[c];
    for (int c = threadIdx.x; c < a.rope; c += blockDim.x) row[a.rank + c] = rp[c];
}
__device__ __forceinline__ void store_cell(uint16_t* cache, size_t cell, const float* lat, const float* rp, const MlaArgs& a, float* red) {
    uint16_t* row = cache + cell * (a.rank + a.rope);
    for (int c = threadIdx.x; c < a.rank; c += blockDim.x) row[c] = f16_bits(lat[c]);
    for (int c = threadIdx.x; c < a.rope; c += blockDim.x) row[a.rank + c] = f16_bits(rp[c]);
}
// vLLM's fp8_ds_mla row of the BF16 latent (concat_and_cache_ds_mla): per 128 values the scale 2^ceil(log2(max(amax / 448, 1e-4))),
// codes = E4M3(v / scale) round to nearest, saturating; the rope values BF16. A code times its power-of-two scale is a BF16 value,
// so the row decodes exactly. Needs a block of at least 512 threads (a quarter of them per scale).
__device__ __forceinline__ void store_cell(uint8_t* cache, size_t cell, const float* lat, const float* rp, const MlaArgs& a, float* red) {
    uint8_t* row = cache + cell * kFp8RowBytes;
    const int c = threadIdx.x, g = c >> 7;
    const float v = c < kFp8Codes ? __bfloat162float(__float2bfloat16_rn(lat[c])) : 0.f;
    const float amax = warp_max(fabsf(v));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = amax;
    __syncthreads();
    const float m = fmaxf(fmaxf(red[g * 4], red[g * 4 + 1]), fmaxf(red[g * 4 + 2], red[g * 4 + 3]));
    const float scale = exp2f(ceilf(log2f(fmaxf(m / 448.f, 1e-4f))));
    if (c < kFp8Codes) row[c] = __nv_cvt_float_to_fp8(v / scale, __NV_SATFINITE, __NV_E4M3);
    if ((c & 127) == 0 && c < kFp8Codes) reinterpret_cast<float*>(row + kFp8ScaleOff)[g] = scale;
    for (int i = threadIdx.x; i < a.rope; i += blockDim.x) reinterpret_cast<uint16_t*>(row + kFp8RopeOff)[i] = bf16_bits(rp[i]);
    __syncthreads();
}

// One tile of a per-head product: rows r0..r0 + tr of head h for n_tok tokens, y[t][h][r] (at y + t * y_tok + h * y_head + r)
// = w[h * rows + r] . x[t][h] (x at x + t * x_tok + h * x_head), nb Q8_0 blocks a row. The activation is rounded to q8_1 per
// 32 values, as every other product of the model's Q8_0 weights does: d = amax / 127, codes round half away from zero; a
// row's blocks add in order, each an exact integer sum times d_w * d. The tile's weights go through shared memory in
// 16-byte pieces (tr * nb * 34 a multiple of 16, at most TB bytes). sm: gemv_smem bytes.
__host__ __device__ constexpr int gemv_smem(int tr, int nb, int n_tok) {
    return ((tr * nb * 34 + 15) & ~15) + n_tok * nb * 32 + ((n_tok * nb * 4 + 15) & ~15) + tr * nb * n_tok * 8;
}

template <int NT, int NB, int TB>
__device__ __forceinline__ void gemv_tile(const uint8_t* __restrict__ w, const float* __restrict__ x, int x_head, int x_tok, int rows,
                                          int n_tok, int h, int r0, int ltr, float* __restrict__ y, int y_head, int y_tok,
                                          unsigned char* sm) {
    constexpr int nb = NB;
    const int tr = 1 << ltr;
    const int tid = threadIdx.x, lane = tid & 31, rb = nb * 34;
    uint8_t* ws = sm;
    int8_t* xq = reinterpret_cast<int8_t*>(sm + ((tr * rb + 15) & ~15));
    float* xd = reinterpret_cast<float*>(xq + n_tok * nb * 32);
    float* pd = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(xd) + ((n_tok * nb * 4 + 15) & ~15));   // [token][row][block] d_w * d
    float* ps = pd + tr * nb * n_tok;                                                                    // ... the integer sum
    const uint4* src = reinterpret_cast<const uint4*>(w + ((size_t) h * rows + r0) * rb);
    uint4* dst = reinterpret_cast<uint4*>(ws);
    const int n16 = tr * rb / 16;
    // every piece of the tile in flight at once; they land while the activation is rounded
    constexpr int kPieces = (TB / 16 + NT - 1) / NT;
    uint4 v[kPieces];
#pragma unroll
    for (int k = 0; k < kPieces; ++k) {
        const int i = k * NT + tid;
        if (i < n16) v[k] = __ldcs(src + i);
    }
    // the activation's q8_1 blocks, a warp each: four blocks' loads in flight before the first is rounded
    for (int p0 = tid >> 5; p0 < n_tok * nb; p0 += 4 * (NT / 32)) {
        float v[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int p = p0 + k * (NT / 32), t = p / nb, b = p - t * nb;
            v[k] = p < n_tok * nb ? x[(size_t) t * x_tok + (size_t) h * x_head + b * 32 + lane] : 0.f;
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int p = p0 + k * (NT / 32);
            if (p >= n_tok * nb) break;   // the same for the whole warp
            const float amax = warp_max(fabsf(v[k]));
            const float d = amax / 127.f;
            const float id = d != 0.f ? 1.f / d : 0.f;
            xq[p * 32 + lane] = (int8_t) __float2int_rn(roundf(v[k] * id));
            if (lane == 0) xd[p] = d;
        }
    }
#pragma unroll
    for (int k = 0; k < kPieces; ++k) {
        const int i = k * NT + tid;
        if (i < n16) dst[i] = v[k];
    }
    __syncthreads();
    // each (row, block) for every token: the exact integer sum and d_w * d. A block's codes are read once, as the aligned
    // words that hold them (an even block's codes start 2 bytes into a word, an odd block's at a word)
    for (int i = tid; i < tr * nb; i += NT) {
        const int b = i % nb, odd = i & 1;
        const uint32_t* wv = reinterpret_cast<const uint32_t*>(ws + ((i * 34 + 2) & ~3));   // the word with the first code
        uint32_t wd[9];
#pragma unroll
        for (int k = 0; k < 9; ++k) wd[k] = wv[k];
        const uint32_t sh = odd ? 0u : 16u;
        int c[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) c[k] = (int) __funnelshift_r(wd[k], wd[k + 1], sh);
        const float dw = __half2float(__ushort_as_half((uint16_t) (odd ? wv[-1] >> 16 : wd[0] & 0xffffu)));
        for (int t = 0; t < n_tok; ++t) {
            const int4* xi = reinterpret_cast<const int4*>(xq + (t * nb + b) * 32);
            const int4 a0 = xi[0], a1 = xi[1];
            int s = 0;
            s = __dp4a(c[0], a0.x, s);
            s = __dp4a(c[1], a0.y, s);
            s = __dp4a(c[2], a0.z, s);
            s = __dp4a(c[3], a0.w, s);
            s = __dp4a(c[4], a1.x, s);
            s = __dp4a(c[5], a1.y, s);
            s = __dp4a(c[6], a1.z, s);
            s = __dp4a(c[7], a1.w, s);
            const int o = (t * tr + i / nb) * nb + b;
            pd[o] = dw * xd[t * nb + b];
            ps[o] = (float) s;
        }
    }
    __syncthreads();
    // each (row, token): the blocks in order
    for (int i = tid; i < tr * n_tok; i += NT) {
        const int t = i >> ltr, r = i & (tr - 1);
        const float* p = pd + (t * tr + r) * nb;
        const float* q = ps + (t * tr + r) * nb;
        float acc = 0.f;
#pragma unroll
        for (int b = 0; b < nb; ++b) acc = fmaf(p[b], q[b], acc);
        y[(size_t) t * y_tok + (size_t) h * y_head + r0 + r] = acc;
    }
}

// Rows of a tile for n_tok rows, as their log2: the largest power of two up to `rows` whose weights fit `cap` bytes and whose
// shared memory fits 48 KB. The tiling does not change a row's result.
int gemv_rows_log2(int rows, int nb, int n_tok, int cap) {
    int ltr = 0;
    const auto fits = [&](int tr) { return tr <= rows && rows % tr == 0 && tr * nb * 34 <= cap && gemv_smem(tr, nb, n_tok) <= 48 * 1024; };
    while (fits(2 << ltr)) ++ltr;
    if (((1 << ltr) * nb * 34) % 16) throw std::invalid_argument("hy4flash: a heads product tile is not a multiple of 16 bytes");
    if (gemv_smem(1 << ltr, nb, n_tok) > 48 * 1024) throw std::invalid_argument("hy4flash: a heads product tile needs too much shared memory");
    return ltr;
}

// The tile bytes of the per-head products: small tiles keep more of a product's tiles in flight at once; the query
// kernel's blocks are 512 threads (its cache rows need them) and take tiles of 64 rows.
constexpr int kGemvThreads = 256, kGemvTile = 9216, kQueryTile = 13824;
// blockIdx.x = tile + head * tiles; NB: the Q8_0 blocks of a row
template <int NB>
__global__ void __launch_bounds__(kGemvThreads) heads_gemv_kernel(const uint8_t* __restrict__ w, const float* __restrict__ x, int x_head,
                                                                  int x_tok, int rows, int heads, int n_tok, int ltr, float* __restrict__ y) {
    extern __shared__ __align__(16) unsigned char sm[];
    const int tiles = rows >> ltr, h = blockIdx.x / tiles, tile = blockIdx.x - h * tiles;
    gemv_tile<kGemvThreads, NB, kGemvTile>(w, x, x_head, x_tok, rows, n_tok, h, tile << ltr, ltr, y, rows, heads * rows, sm);
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

// The rotated rope part of head h's query row of token t at `pos` (threads below rope / 2, a pair each).
__device__ __forceinline__ void query_rope(const float* __restrict__ q, const MlaArgs& a, int pos, int t, int h, float* __restrict__ qcur) {
    const int W = a.rank + a.rope, Q = a.nope + a.rope;
    if (threadIdx.x < a.rope / 2) {
        const int i = threadIdx.x;
        const float* p = q + ((size_t) t * a.heads + h) * Q + a.nope + 2 * i;
        float* out = qcur + ((size_t) t * a.heads + h) * W + a.rank;
        float o0, o1;
        rope_pair(p[0], p[1], i, a.rope, a.rope_base, axis_pos(a, pos, i), o0, o1);
        out[2 * i] = o0;
        out[2 * i + 1] = o1;
    }
}

// Token t's cache row at `pos` (modulo `ring` when set): the latent normed by kv_norm, then the rotated rope part. A block of
// kThreads threads.
template <typename C>
__device__ __forceinline__ void cache_row(const float* __restrict__ kv_raw, const float* __restrict__ kv_norm, const MlaArgs& a, int pos,
                                          int ring, int t, C* __restrict__ cache) {
    __shared__ float red[kThreads / 32];
    __shared__ float lat[512], rp[64];
    const float* kr = kv_raw + (size_t) t * (a.rank + a.rope);
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

// Block h < heads: the head's query row; block `heads`: the cache row. y = the token.
template <typename C>
__global__ void prepare_kernel(const float* __restrict__ q, const float* __restrict__ q_abs, const float* __restrict__ kv_raw,
                               const float* __restrict__ kv_norm, MlaArgs a, int pos0, int ring, float* __restrict__ qcur,
                               C* __restrict__ cache) {
    const int h = blockIdx.x, t = blockIdx.y, W = a.rank + a.rope, pos = pos0 + t;
    if (h < a.heads) {
        const float* qa = q_abs + (size_t) t * a.heads * a.rank;
        float* out = qcur + (size_t) t * a.heads * W;
        for (int c = threadIdx.x; c < a.rank; c += blockDim.x) out[(size_t) h * W + c] = qa[(size_t) h * a.rank + c];
        query_rope(q, a, pos, t, h, qcur);
        return;
    }
    cache_row(kv_raw, kv_norm, a, pos, ring, t, cache);
}

// A window's query rows and cache rows in one launch: blocks below heads * tiles take a tile of the head's k_b product into
// the query row's latent part (the first tile also its rope part), the last n_tok blocks the tokens' cache rows. Positions
// are read from pos[t].
template <typename C, int NB>
__global__ void __launch_bounds__(kThreads) query_kernel(const uint8_t* __restrict__ w_kb, const float* __restrict__ q, const float* __restrict__ kv_raw,
                                                         const float* __restrict__ kv_norm, MlaArgs a, const int32_t* __restrict__ pos, int n_tok,
                                                         int ring, int ltr, float* __restrict__ qcur, C* __restrict__ cache) {
    extern __shared__ __align__(16) unsigned char sm[];
    const int W = a.rank + a.rope, Q = a.nope + a.rope, tiles = a.rank >> ltr, b = blockIdx.x;
    if (b < a.heads * tiles) {
        const int h = b / tiles, tile = b - h * tiles;
        gemv_tile<kThreads, NB, kQueryTile>(w_kb, q, Q, a.heads * Q, a.rank, n_tok, h, tile << ltr, ltr, qcur, W, a.heads * W, sm);
        if (tile == 0)
            for (int t = 0; t < n_tok; ++t) query_rope(q, a, pos[t], t, h, qcur);
        return;
    }
    const int t = b - a.heads * tiles;
    cache_row(kv_raw, kv_norm, a, pos[t], ring, t, cache);
}

// ---- chunked attention: the shared MQA chunk stage over a window's rows ---------------------------------------------------
// How the chunk stage reads a cache row (576 values: the latent, then the rope part; mqa_chunk.cuh has the interface).
struct RowF32 {
    using Raw4 = float4;
    using Raw2 = float2;
    using RawR = float2;
    using Scales = mqa::NoScales;
    static constexpr int kRowBytes = 576 * 4;
    static __device__ __forceinline__ Scales scales(const uint8_t*) { return {}; }
    static __device__ __forceinline__ Raw4 load4(const uint8_t* r, int i) { return reinterpret_cast<const float4*>(r)[i]; }
    static __device__ __forceinline__ Raw2 load2(const uint8_t* r, int i) { return reinterpret_cast<const float2*>(r)[i]; }
    static __device__ __forceinline__ RawR loadr(const uint8_t* r, int i) { return reinterpret_cast<const float2*>(r + 2048)[i]; }
    static __device__ __forceinline__ void cvt4(const Raw4& v, const Scales&, int, float k[4]) { k[0] = v.x; k[1] = v.y; k[2] = v.z; k[3] = v.w; }
    static __device__ __forceinline__ void cvt2(const Raw2& v, float k[2]) { k[0] = v.x; k[1] = v.y; }
    static __device__ __forceinline__ void cvtr(const RawR& v, float k[2]) { k[0] = v.x; k[1] = v.y; }
};
__device__ __forceinline__ void bf16x2(uint32_t v, float* k) {
    k[0] = __uint_as_float(v << 16);
    k[1] = __uint_as_float(v & 0xffff0000u);
}
__device__ __forceinline__ void f16x2(uint32_t v, float* k) {
    const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&v));
    k[0] = f.x;
    k[1] = f.y;
}
struct RowF16 {
    using Raw4 = uint2;
    using Raw2 = uint32_t;
    using RawR = uint32_t;
    using Scales = mqa::NoScales;
    static constexpr int kRowBytes = 576 * 2;
    static __device__ __forceinline__ Scales scales(const uint8_t*) { return {}; }
    static __device__ __forceinline__ Raw4 load4(const uint8_t* r, int i) { return reinterpret_cast<const uint2*>(r)[i]; }
    static __device__ __forceinline__ Raw2 load2(const uint8_t* r, int i) { return reinterpret_cast<const uint32_t*>(r)[i]; }
    static __device__ __forceinline__ RawR loadr(const uint8_t* r, int i) { return reinterpret_cast<const uint32_t*>(r + 1024)[i]; }
    static __device__ __forceinline__ void cvt4(const Raw4& v, const Scales&, int, float k[4]) { f16x2(v.x, k); f16x2(v.y, k + 2); }
    static __device__ __forceinline__ void cvt2(const Raw2& v, float k[2]) { f16x2(v, k); }
    static __device__ __forceinline__ void cvtr(const RawR& v, float k[2]) { f16x2(v, k); }
};
// the FP8 row: a code times its group's scale (load4's j is its group: values 4 (lane + 32 j)..)
__device__ __forceinline__ float fp8_value(uint32_t code, float scale) {
    const __half_raw h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t) code, __NV_E4M3);
    return __half2float(__half(h)) * scale;
}
struct RowFp8 {
    using Raw4 = uint32_t;
    using Raw2 = uint2;   // two codes, the scale's bits
    using RawR = uint32_t;
    using Scales = float4;
    static constexpr int kRowBytes = kFp8RowBytes;
    static __device__ __forceinline__ Scales scales(const uint8_t* r) { return *reinterpret_cast<const float4*>(r + kFp8ScaleOff); }
    static __device__ __forceinline__ Raw4 load4(const uint8_t* r, int i) { return reinterpret_cast<const uint32_t*>(r)[i]; }
    static __device__ __forceinline__ Raw2 load2(const uint8_t* r, int i) {
        return make_uint2(reinterpret_cast<const uint16_t*>(r)[i], __float_as_uint(reinterpret_cast<const float*>(r + kFp8ScaleOff)[i >> 6]));
    }
    static __device__ __forceinline__ RawR loadr(const uint8_t* r, int i) { return reinterpret_cast<const uint32_t*>(r + kFp8RopeOff)[i]; }
    static __device__ __forceinline__ void cvt4(const Raw4& v, const Scales& s, int j, float k[4]) {
        const float sc = j == 0 ? s.x : j == 1 ? s.y : j == 2 ? s.z : s.w;
#pragma unroll
        for (int e = 0; e < 4; ++e) k[e] = fp8_value((v >> (8 * e)) & 0xffu, sc);
    }
    static __device__ __forceinline__ void cvt2(const Raw2& v, float k[2]) {
        const float sc = __uint_as_float(v.y);
        k[0] = fp8_value(v.x & 0xffu, sc);
        k[1] = fp8_value((v.x >> 8) & 0xffu, sc);
    }
    static __device__ __forceinline__ void cvtr(const RawR& v, float k[2]) { bf16x2(v, k); }
};

// The rows a window row attends (RowSpans): row t's j-th cell.
__device__ __forceinline__ int span_cell(const RowSpans& r, int ring, int t, int j) {
    if (r.cells && !(r.fill && r.fill[t] > 0)) return r.cells[(size_t) t * r.cells_stride + j];
    const int first = r.firsts[t];
    return ring ? (first + j) % ring : first + j;
}

// A block per (chunk of mqa::kChunk cells, HB heads, row): the chunk's scores, softmax statistics and weighted sum into the
// workspace; the last of a (row, head group)'s chunks to finish merges them with the sink into o_lat, in chunk order (the
// same bits whichever block is last). The counters return to zero.
template <class R, int HB>
__global__ void __launch_bounds__(mqa::kThreads) attend_kernel(const float* __restrict__ qcur, const uint8_t* __restrict__ cache, RowSpans r, int ring,
                                                               const float* __restrict__ sinks, int heads, float scale, int max_ch,
                                                               float* __restrict__ ml, float* __restrict__ op, int* __restrict__ counters,
                                                               float* __restrict__ o_lat) {
    constexpr int W = mqa::kLatent + 64;
    __shared__ float4 qs[HB * W / 4];
    static_assert(mqa::merge_scratch_floats(HB, kMaxChunks) <= HB * W, "the merge's scratch fits the query rows");
    __shared__ __align__(16) float sc[HB][mqa::kChunk];
    __shared__ int rowid[mqa::kChunk];
    __shared__ int last;
    const int c = blockIdx.x, g = blockIdx.y, t = blockIdx.z, tid = threadIdx.x;
    const int n = r.counts[t], n_ch = max(1, (n + mqa::kChunk - 1) / mqa::kChunk);
    if (c >= n_ch) return;
    const int head0 = g * HB;
    // the query rows' loads go out with the cell ids'
    constexpr int kQ4 = HB * W / 4, kQ = (kQ4 + mqa::kThreads - 1) / mqa::kThreads;
    float4 qv[kQ];
    const float4* qg = reinterpret_cast<const float4*>(qcur + ((size_t) t * heads + head0) * W);
#pragma unroll
    for (int k = 0; k < kQ; ++k)
        if (tid + k * mqa::kThreads < kQ4) qv[k] = qg[tid + k * mqa::kThreads];
    int row = -1;
    if (tid < mqa::kChunk) {
        const int j = c * mqa::kChunk + tid;
        if (j < n) row = span_cell(r, ring, t, j);
        rowid[tid] = row;
    }
#pragma unroll
    for (int k = 0; k < kQ; ++k)
        if (tid + k * mqa::kThreads < kQ4) qs[tid + k * mqa::kThreads] = qv[k];
    const bool any = __syncthreads_or(tid < mqa::kChunk && row >= 0);
    const size_t base = ((size_t) t * heads + head0) * max_ch;
    if (any) {
        mqa::chunk<R, R, HB, 64>(true, cache, scale, qs, sc, rowid, ml + (base + c) * 2, (size_t) max_ch * 2,
                                 op + (base + c) * mqa::kLatent, (size_t) max_ch * mqa::kLatent);
    } else if (tid < HB) {
        float* m = ml + (base + (size_t) tid * max_ch + c) * 2;
        m[0] = -INFINITY;
        m[1] = 0.f;
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        int* cnt = counters + (size_t) t * (heads / HB) + g;
        last = atomicAdd(cnt, 1) == n_ch - 1;
        if (last) *cnt = 0;
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    // the query rows are no longer needed: their room is the merge's scratch
    float2 o[HB];
    mqa::merge<HB, true>(op + base * mqa::kLatent, (size_t) max_ch * mqa::kLatent, ml + base * 2, (size_t) max_ch * 2, n_ch, sinks + head0,
                         reinterpret_cast<float*>(qs), o);
#pragma unroll
    for (int h = 0; h < HB; ++h) reinterpret_cast<float2*>(o_lat + ((size_t) t * heads + head0 + h) * mqa::kLatent)[tid] = o[h];
}

// ---- tensor-core attention over a BF16 or FP8 cache --------------------------------------------------------------------------
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

// Eight values of a cell's row (chunk ch of 72) as raw words, and the same as eight fp16 values. fp16 rows: the eight values. FP8
// rows: chunks below 64 are eight E4M3 codes then their group's scale, the rest eight BF16 rope values; a code times its
// power-of-two scale is exact in fp16 (the scales are at least 2^-13).
__device__ __forceinline__ uint4 fetch8(const uint16_t* cache, size_t cell, int ch) {
    return *reinterpret_cast<const uint4*>(cache + cell * 576 + ch * 8);
}
__device__ __forceinline__ uint4 fetch8(const uint8_t* cache, size_t cell, int ch) {
    const uint8_t* row = cache + cell * kFp8RowBytes;
    if (ch < kFp8Codes / 8) {
        const uint2 c = *reinterpret_cast<const uint2*>(row + ch * 8);
        return make_uint4(c.x, c.y, __float_as_uint(reinterpret_cast<const float*>(row + kFp8ScaleOff)[ch >> 4]), 0);
    }
    return *reinterpret_cast<const uint4*>(row + kFp8RopeOff + (ch - kFp8Codes / 8) * 16);
}
__device__ __forceinline__ uint4 half8(const uint16_t*, uint4 v, int) { return v; }
__device__ __forceinline__ uint32_t e4m3x2_to_h2(uint32_t codes, __half2 scale) {
    const __half2 h = __hmul2(__half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) codes, __NV_E4M3)), scale);
    return *reinterpret_cast<const uint32_t*>(&h);
}
__device__ __forceinline__ uint4 half8(const uint8_t*, uint4 v, int ch) {
    if (ch >= kFp8Codes / 8) return make_uint4(bf2_to_h2(v.x), bf2_to_h2(v.y), bf2_to_h2(v.z), bf2_to_h2(v.w));
    const __half2 s = __float2half2_rn(__uint_as_float(v.z));
    return make_uint4(e4m3x2_to_h2(v.x & 0xffffu, s), e4m3x2_to_h2(v.x >> 16, s), e4m3x2_to_h2(v.y & 0xffffu, s), e4m3x2_to_h2(v.y >> 16, s));
}

// G groups of 16 heads a block share each tile of cells (the cells are the row's for every head: one load, one conversion);
// every output element sums its products in the same order whatever G is, so the groups only change how the work is shared.
template <int G> constexpr size_t mma_smem() {
    return (size_t) (16 * G * kLd + kMmaCells * kLd + 16 * G * kPLd) * 2 + (2 * 16 * G * 32 + 2 * 16 * G) * sizeof(float);
}
template <typename C, int G>
__global__ void __launch_bounds__(256) attend_mma_kernel(const float* __restrict__ qcur, const C* __restrict__ cache, const int32_t* __restrict__ firsts,
                                                         const int32_t* __restrict__ counts, const int32_t* __restrict__ cells, int cells_stride,
                                                         const int32_t* __restrict__ fill, const float* __restrict__ sinks, MlaArgs a, int ring,
                                                         float* __restrict__ o_lat) {
    extern __shared__ __align__(16) unsigned char smraw[];
    constexpr int W = 576, M = 16 * G;
    uint16_t* qh = (uint16_t*) smraw;
    uint16_t* ks = qh + M * kLd;
    uint16_t* ph = ks + kMmaCells * kLd;
    float* sp = (float*) (ph + M * kPLd);   // [2][M][32]
    float* alpha_s = sp + 2 * M * 32;
    float* l_s = alpha_s + M;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    const int t = blockIdx.y, h0 = blockIdx.x * M;
    const int n = counts[t], first = firsts[t];
    const int32_t* cl = cells && !(fill && fill[t] > 0) ? cells + (size_t) t * cells_stride : nullptr;
    for (int i = tid; i < M * W; i += 256) {
        const int h = i / W, e = i - h * W;
        qh[h * kLd + e] = __half_as_ushort(__float2half_rn(qcur[((size_t) t * a.heads + h0) * W + i] * a.scale));
    }
    // the softmax: a warp keeps 2G heads
    float m[2 * G], l[2 * G];
#pragma unroll
    for (int hh = 0; hh < 2 * G; ++hh) {
        m[hh] = sinks[h0 + w * 2 * G + hh];
        l[hh] = 1.f;
    }
    if (tid < M) l_s[tid] = 1.f;
    // O: a warp keeps the 16 heads of group rg and 64G latent dimensions
    float acc[8 * G][4];
#pragma unroll
    for (int j = 0; j < 8 * G; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
    const int id = lane >> 3, r8 = lane & 7, rg = w % G, dq = w / G;
    // S: a warp takes the heads of group rg, half kh of the 576 dimensions and G tiles of 8 cells from cell 8G cp
    const int kh = (w / G) & 1, cp = w / (2 * G);
    // a thread's share of a tile, loaded a tile ahead: the next tile's reads are in flight while this one computes
    constexpr int kPer = kMmaCells * (W / 8) / 256;
    uint4 pre[kPer];
    auto fetch = [&](int j0) {
        const int tn = min(kMmaCells, n - j0);
#pragma unroll
        for (int k = 0; k < kPer; ++k) {
            const int i = tid + k * 256, c = i / (W / 8), ch = i - c * (W / 8);
            pre[k] = make_uint4(0, 0, 0, 0);
            if (c < tn) {
                const int j = j0 + c;
                const int cell = cl ? cl[j] : ring ? (first + j) % ring : first + j;
                pre[k] = fetch8(cache, (size_t) cell, ch);
            }
        }
    };
    fetch(0);
    for (int j0 = 0; j0 < n; j0 += kMmaCells) {
        const int tn = min(kMmaCells, n - j0);
        __syncthreads();
#pragma unroll
        for (int k = 0; k < kPer; ++k) {
            const int i = tid + k * 256, c = i / (W / 8), ch = i - c * (W / 8);
            *reinterpret_cast<uint4*>(ks + c * kLd + ch * 8) = half8(cache, pre[k], ch);
        }
        __syncthreads();
        if (j0 + kMmaCells < n) fetch(j0 + kMmaCells);
        {
            float c[G][4];
#pragma unroll
            for (int nt = 0; nt < G; ++nt) c[nt][0] = c[nt][1] = c[nt][2] = c[nt][3] = 0.f;
            for (int kk = 0; kk < 9; ++kk) {
                const int kb = (kh * 18 + kk * 2) * 16;
                uint32_t b[G][4];
#pragma unroll
                for (int nt = 0; nt < G; ++nt) ldsm4(b[nt], ks + ((cp * G + nt) * 8 + r8) * kLd + kb + id * 8);
#pragma unroll
                for (int s2 = 0; s2 < 2; ++s2) {
                    const int kd = kb + s2 * 16;
                    uint32_t ah[4];
                    ldsm4(ah, qh + (rg * 16 + (id & 1) * 8 + r8) * kLd + kd + (id >> 1) * 8);
#pragma unroll
                    for (int nt = 0; nt < G; ++nt) mma16816h(c[nt], ah, b[nt][2 * s2], b[nt][2 * s2 + 1]);
                }
            }
            const int row = rg * 16 + (lane >> 2);
#pragma unroll
            for (int nt = 0; nt < G; ++nt) {
                const int col = (cp * G + nt) * 8 + (lane & 3) * 2;
                sp[(kh * M + row) * 32 + col] = c[nt][0];
                sp[(kh * M + row) * 32 + col + 1] = c[nt][1];
                sp[(kh * M + row + 8) * 32 + col] = c[nt][2];
                sp[(kh * M + row + 8) * 32 + col + 1] = c[nt][3];
            }
        }
        __syncthreads();
        // the running softmax: a lane a cell
#pragma unroll
        for (int hh = 0; hh < 2 * G; ++hh) {
            const int h = w * 2 * G + hh;
            const bool valid = lane < tn;
            const float sc = valid ? sp[h * 32 + lane] + sp[(M + h) * 32 + lane] : -INFINITY;
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
        // O = O * alpha + P V
        {
            const float a0 = alpha_s[rg * 16 + (lane >> 2)], a1 = alpha_s[rg * 16 + (lane >> 2) + 8];
#pragma unroll
            for (int j = 0; j < 8 * G; ++j) {
                acc[j][0] *= a0; acc[j][1] *= a0;
                acc[j][2] *= a1; acc[j][3] *= a1;
            }
#pragma unroll
            for (int s2 = 0; s2 < 2; ++s2) {
                uint32_t ah[4];
                ldsm4(ah, ph + (rg * 16 + (id & 1) * 8 + r8) * kPLd + s2 * 16 + (id >> 1) * 8);
#pragma unroll
                for (int jj = 0; jj < 4 * G; ++jj) {
                    uint32_t b[4];
                    ldsm4t(b, ks + (s2 * 16 + (id & 1) * 8 + r8) * kLd + dq * 64 * G + jj * 16 + (id >> 1) * 8);
                    mma16816h(acc[2 * jj], ah, b[0], b[1]);
                    mma16816h(acc[2 * jj + 1], ah, b[2], b[3]);
                }
            }
        }
    }
    __syncthreads();
    const int row = rg * 16 + (lane >> 2);
    const float i0 = 1.f / l_s[row], i1 = 1.f / l_s[row + 8];
#pragma unroll
    for (int j = 0; j < 8 * G; ++j) {
        const int d = dq * 64 * G + j * 8 + (lane & 3) * 2;
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

template <typename C>
void prepare_impl(const float* q, const float* q_abs, const float* kv_raw, const float* kv_norm, const MlaArgs& a, int pos0,
                  int n_tok, int ring, float* qcur, C* cache, void* stream) {
    prepare_kernel<C><<<dim3(a.heads + 1, n_tok), kThreads, 0, (cudaStream_t) stream>>>(q, q_abs, kv_raw, kv_norm, a, pos0, ring, qcur, cache);
    if (!stream) cudaStreamSynchronize(nullptr);
}

template <typename C> struct RowOf;
template <> struct RowOf<float> { using T = RowF32; };
template <> struct RowOf<uint16_t> { using T = RowF16; };
template <> struct RowOf<uint8_t> { using T = RowFp8; };

// 16-head groups a block of the tensor-core kernel takes: 2 when the heads come in 32s
int mma_groups(int heads) {
    static const int force = [] { const char* e = std::getenv("HY4_MMA_GROUPS"); return e ? std::atoi(e) : 0; }();   // DIAG
    if (force == 1 || force == 2) return heads % (16 * force) == 0 ? force : 1;
    return heads % 32 == 0 ? 2 : 1;
}

int chunks_of(int max_cells) { return std::max(1, (max_cells + mqa::kChunk - 1) / mqa::kChunk); }
size_t counters_bytes(int heads) { return ((size_t) kAttTok * (heads / 8) * 4 + 255) & ~(size_t) 255; }

int sm_count() {
    static int sms = 0;
    if (!sms) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    }
    return sms;
}

template <typename C>
void attend_impl(const float* qcur, const C* cache, const RowSpans& r, int n_tok, const float* sinks, const MlaArgs& a, int ring,
                 int max_cells, void* work, float* o_lat, void* stream) {
    if (a.rank != mqa::kLatent || a.rope != 64 || a.heads % 8) throw std::invalid_argument("mla_attend: a 512 + 64 row and heads in eights");
    cudaStream_t st = (cudaStream_t) stream;
    if constexpr (!std::is_same_v<C, float>) {
        if (n_tok > kAttTok && a.heads % 16 == 0) {   // a chunk of rows; a window keeps the kernel below
            const int g = mma_groups(a.heads);
            static bool raised[2] = {false, false};
            if (!raised[g - 1]) {
                if (g == 2) cudaFuncSetAttribute(attend_mma_kernel<C, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) mma_smem<2>());
                else cudaFuncSetAttribute(attend_mma_kernel<C, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) mma_smem<1>());
                raised[g - 1] = true;
            }
            if (g == 2)
                attend_mma_kernel<C, 2><<<dim3(a.heads / 32, n_tok), 256, mma_smem<2>(), st>>>(qcur, cache, r.firsts, r.counts, r.cells, r.cells_stride,
                                                                                             r.fill, sinks, a, ring, o_lat);
            else
                attend_mma_kernel<C, 1><<<dim3(a.heads / 16, n_tok), 256, mma_smem<1>(), st>>>(qcur, cache, r.firsts, r.counts, r.cells, r.cells_stride,
                                                                                             r.fill, sinks, a, ring, o_lat);
            if (!stream) cudaStreamSynchronize(nullptr);
            return;
        }
    }
    using R = typename RowOf<C>::T;
    const int max_ch = chunks_of(max_cells);
    if (max_ch > kMaxChunks) throw std::invalid_argument("mla_attend: more than 8,192 cells a row");
    int* counters = (int*) work;
    float* ml = (float*) ((char*) work + counters_bytes(a.heads));
    for (int t0 = 0; t0 < n_tok; t0 += kAttTok) {
        const int T = std::min(kAttTok, n_tok - t0);
        float* op = ml + (size_t) T * a.heads * max_ch * 2;
        RowSpans rt = r;
        rt.firsts = r.firsts + t0;
        rt.counts = r.counts + t0;
        if (r.fill) rt.fill = r.fill + t0;
        if (r.cells) rt.cells = r.cells + (size_t) t0 * r.cells_stride;
        const float* q = qcur + (size_t) t0 * a.heads * (mqa::kLatent + 64);
        float* o = o_lat + (size_t) t0 * a.heads * mqa::kLatent;
        // 8 heads a block while the launch fits one block per SM, else 16; the heads of a block share no sums, so both give
        // the same bits
        if (max_ch * (a.heads / 8) * T <= sm_count() || a.heads % 16)
            attend_kernel<R, 8><<<dim3(max_ch, a.heads / 8, T), mqa::kThreads, 0, st>>>(q, (const uint8_t*) cache, rt, ring, sinks, a.heads, a.scale,
                                                                                        max_ch, ml, op, counters, o);
        else
            attend_kernel<R, 16><<<dim3(max_ch, a.heads / 16, T), mqa::kThreads, 0, st>>>(q, (const uint8_t*) cache, rt, ring, sinks, a.heads, a.scale,
                                                                                          max_ch, ml, op, counters, o);
    }
    if (!stream) cudaStreamSynchronize(nullptr);
}

template <typename C>
void query_impl(const void* w_kb, const float* q, const float* kv_raw, const float* kv_norm, const MlaArgs& a, const int32_t* pos, int n_tok,
                int ring, float* qcur, C* cache, void* stream) {
    if (n_tok < 1 || n_tok > kAttTok) throw std::invalid_argument("mla_query: 1 to 8 rows");
    if (a.nope != 192 || ((uintptr_t) w_kb & 15)) throw std::invalid_argument("mla_query: k_b rows of 192, 16-byte aligned");
    const int nb = a.nope / 32, ltr = gemv_rows_log2(a.rank, nb, n_tok, kQueryTile), tr = 1 << ltr;
    query_kernel<C, 6><<<a.heads * (a.rank / tr) + n_tok, kThreads, gemv_smem(tr, nb, n_tok), (cudaStream_t) stream>>>(
        (const uint8_t*) w_kb, q, kv_raw, kv_norm, a, pos, n_tok, ring, ltr, qcur, cache);
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
    if ((k != 512 && k != 192) || ((uintptr_t) w & 15)) throw std::invalid_argument("q8_0_heads_gemv: rows of 512 or 192, 16-byte aligned");
    const int nb = k / 32;
    const auto kern = nb == 16 ? heads_gemv_kernel<16> : heads_gemv_kernel<6>;   // v_b's rows (the latent), k_b's (the query's nope part)
    for (int t0 = 0; t0 < n_tok; t0 += kAttTok) {
        const int T = std::min(kAttTok, n_tok - t0), ltr = gemv_rows_log2(rows, nb, T, kGemvTile), tr = 1 << ltr;
        kern<<<heads * (rows / tr), kGemvThreads, gemv_smem(tr, nb, T), (cudaStream_t) stream>>>(
            (const uint8_t*) w, x + (size_t) t0 * x_tok_stride, x_head_stride, x_tok_stride, rows, heads, T, ltr, y + (size_t) t0 * heads * rows);
    }
    if (!stream) cudaStreamSynchronize(nullptr);
}

size_t mla_attend_work_bytes(int heads, int max_cells, int max_rows) {
    return counters_bytes(heads) + (size_t) std::min(max_rows, kAttTok) * heads * chunks_of(max_cells) * (mqa::kLatent + 2) * sizeof(float);
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
                int max_cells, void* work, float* o_lat, void* stream) {
    attend_impl(qcur, cache, r, n_tok, sinks, a, ring, max_cells, work, o_lat, stream);
}
void mla_query(const void* w_kb, const float* q, const float* kv_raw, const float* kv_norm, const MlaArgs& a, const int32_t* pos, int n_tok,
               int ring, float* qcur, float* cache, void* stream) {
    query_impl(w_kb, q, kv_raw, kv_norm, a, pos, n_tok, ring, qcur, cache, stream);
}
void mla_attend(const float* qcur, const uint16_t* cache, const RowSpans& r, int n_tok, const float* sinks, const MlaArgs& a, int ring,
                int max_cells, void* work, float* o_lat, void* stream) {
    attend_impl(qcur, cache, r, n_tok, sinks, a, ring, max_cells, work, o_lat, stream);
}
void mla_query(const void* w_kb, const float* q, const float* kv_raw, const float* kv_norm, const MlaArgs& a, const int32_t* pos, int n_tok,
               int ring, float* qcur, uint16_t* cache, void* stream) {
    query_impl(w_kb, q, kv_raw, kv_norm, a, pos, n_tok, ring, qcur, cache, stream);
}

void mla_prepare(const float* q, const float* q_abs, const float* kv_raw, const float* kv_norm, const MlaArgs& a, int pos0,
                 int n_tok, int ring, float* qcur, uint8_t* cache, void* stream) {
    prepare_impl(q, q_abs, kv_raw, kv_norm, a, pos0, n_tok, ring, qcur, cache, stream);
}
void mla_attend(const float* qcur, const uint8_t* cache, const RowSpans& r, int n_tok, const float* sinks, const MlaArgs& a, int ring,
                int max_cells, void* work, float* o_lat, void* stream) {
    attend_impl(qcur, cache, r, n_tok, sinks, a, ring, max_cells, work, o_lat, stream);
}
void mla_query(const void* w_kb, const float* q, const float* kv_raw, const float* kv_norm, const MlaArgs& a, const int32_t* pos, int n_tok,
               int ring, float* qcur, uint8_t* cache, void* stream) {
    query_impl(w_kb, q, kv_raw, kv_norm, a, pos, n_tok, ring, qcur, cache, stream);
}

}  // namespace basalt::models::hy4flash
