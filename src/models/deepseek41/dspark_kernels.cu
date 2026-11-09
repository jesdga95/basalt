// src/models/deepseek41/dspark_kernels.cu - device kernels of the DSpark drafter (see dspark_kernels.hpp).
#include "models/deepseek41/dspark_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace basalt::models::deepseek41 {
namespace {

constexpr int kRowsPerBlock = 8;        // one warp per output row
constexpr int kRowGroups = 4;           // rows per warp in the expert kernels, which share one staged activation row
constexpr int kLatentW = 512;
constexpr int kRopeW = 64;
constexpr int kMarkovRank = 256;

// q8_1 activations as the native quantizer writes them: per 32 values an fp16 scale (and sum), then 32 int8.
struct Q81 {
    __half2 ds;
    int8_t qs[32];
};

// One q8_1 block in shared memory with its bytes in the order an MXFP4 block's nibbles come (x[0], x[16], x[1],
// x[17], ...) and its scale; 48 bytes, so the lanes' 16-byte reads (one block each) do not share a bank.
struct XBlock {
    int4 lo, hi;
    float d;
    float pad[3];
};

// Stages `nb` q8_1 blocks into `xs` in nibble order.
__device__ __forceinline__ void stage_q8(const Q81* __restrict__ x, int nb, XBlock* xs) {
    for (int b = threadIdx.x; b < nb; b += blockDim.x) {
        const int* q = reinterpret_cast<const int*>(x[b].qs);
        int w[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int a = q[i], c = q[i + 4];   // x[4i..4i+3], x[4i+16..4i+19]
            w[2 * i] = (int) __byte_perm(a, c, 0x5140);
            w[2 * i + 1] = (int) __byte_perm(a, c, 0x7362);
        }
        xs[b].lo = make_int4(w[0], w[1], w[2], w[3]);
        xs[b].hi = make_int4(w[4], w[5], w[6], w[7]);
        xs[b].d = __low2float(x[b].ds);
    }
}

// The E2M1 values doubled as int8 (the block scale supplies the other factor of two), by magnitude code.
constexpr uint32_t kPos0 = 0x03020100u, kPos1 = 0x0c080604u;   //  0  1  2  3 |  4  6  8  12
constexpr uint32_t kNeg0 = 0xfdfeff00u, kNeg1 = 0xf4f8fafcu;   //  0 -1 -2 -3 | -4 -6 -8 -12

// The four nibbles in the low 16 bits of `q` as int8 values, in nibble order (the sign select is ggml's
// get_int_from_table_16; MIT, Copyright (c) 2023-2026 The ggml authors).
__device__ __forceinline__ int e2m1x4(uint32_t q) {
    const uint32_t pos = __byte_perm(kPos0, kPos1, q), neg = __byte_perm(kNeg0, kNeg1, q);
    return (int) __byte_perm(pos, neg, 0x3210 | ((q & 0x8888u) >> 1));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

// 2^(e-128) including the two denormal codes.
__device__ __forceinline__ float e8m0_half(uint32_t e) {
    return __uint_as_float(e < 2 ? (0x00200000u << e) : ((e - 1) << 23));
}

// The words of one MXFP4 row in global memory (4-byte aligned, `nb` blocks of 17 bytes) a lane needs: blocks lane,
// lane + 32, ... A block starts at byte 17 * b, so its misalignment is b & 3. The caller pads the matrix by 16 bytes:
// the last block reads up to three bytes past its end.
template <int ITERS>
__device__ __forceinline__ void row_load(const unsigned char* row, int nb, int lane, uint32_t (&w)[ITERS][5]) {
#pragma unroll
    for (int it = 0; it < ITERS; ++it) {
        const int b = lane + 32 * it;
        if (b < nb) {
            const uint32_t* p = reinterpret_cast<const uint32_t*>(row + ((size_t) b * kMxfp4BlockBytes & ~(size_t) 3));
#pragma unroll
            for (int k = 0; k < 5; ++k) w[it][k] = __ldg(p + k);
        }
    }
}

// Dot of a row row_load fetched with staged q8_1 activations; every lane gets the sum. A funnel shift lines each
// block's 16 nibble bytes up; their int8 values meet the activations in dp4a, the block's sum takes both scales.
template <int ITERS>
__device__ __forceinline__ float row_dot(const uint32_t (&w)[ITERS][5], const XBlock* xs, int nb, int lane) {
    float acc = 0.f;
#pragma unroll
    for (int it = 0; it < ITERS; ++it) {
        const int b = lane + 32 * it;
        if (b >= nb) continue;
        const int sh = (b & 3) * 8;
        const float scale = e8m0_half((w[it][0] >> sh) & 0xff);
        const int4 lo = xs[b].lo, hi = xs[b].hi;
        const int xw[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
        int s = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t n = __funnelshift_rc(w[it][k], w[it][k + 1], sh + 8);   // nibble bytes 4k..4k+3
            s = __dp4a(e2m1x4(n), xw[2 * k], s);
            s = __dp4a(e2m1x4(n >> 16), xw[2 * k + 1], s);
        }
        acc += scale * (xs[b].d * (float) s);
    }
    return warp_sum(acc);
}

// Grid (n_ff / (4 * kRowGroups), pairs): the block's 8 warps are 4 rows x (gate, up), kRowGroups times over the
// activation row it staged once; the next group's weights load while this one's dots run.
template <int ITERS>
__global__ void __launch_bounds__(256) k_gate_up(const unsigned char* __restrict__ gw,
                                                 const unsigned char* __restrict__ uw, int n_in, int n_ff,
                                                 const Q81* __restrict__ x, const int32_t* __restrict__ ids, int k,
                                                 float clamp, float* __restrict__ h) {
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ float res[kRowGroups][8];
    const int nb = n_in / kMxfp4BlockElems;
    const size_t row_bytes = (size_t) nb * kMxfp4BlockBytes;
    XBlock* xs = reinterpret_cast<XBlock*>(smem);
    const int pair = blockIdx.y, t = pair / k, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int e = ids[pair];
    const int r0 = blockIdx.x * kRowGroups * (kRowsPerBlock / 2);
    const unsigned char* w0 = ((warp & 1) ? uw : gw) + ((size_t) e * n_ff + r0 + (warp >> 1)) * row_bytes;
    const size_t step = (size_t) (kRowsPerBlock / 2) * row_bytes;   // the warp's row of the next group
    uint32_t w[2][ITERS][5];
    row_load<ITERS>(w0, nb, lane, w[0]);
    stage_q8(x + (size_t) t * nb, nb, xs);
    __syncthreads();
#pragma unroll
    for (int q = 0; q < kRowGroups; ++q) {
        if (q + 1 < kRowGroups) row_load<ITERS>(w0 + (q + 1) * step, nb, lane, w[(q + 1) & 1]);
        const float v = row_dot<ITERS>(w[q & 1], xs, nb, lane);
        if (lane == 0) res[q][warp] = v;
    }
    __syncthreads();
    if (threadIdx.x < kRowGroups * (kRowsPerBlock / 2)) {
        const int q = threadIdx.x / (kRowsPerBlock / 2), i = threadIdx.x % (kRowsPerBlock / 2);
        const float g = fminf(res[q][2 * i], clamp);
        const float u = fminf(fmaxf(res[q][2 * i + 1], -clamp), clamp);
        h[(size_t) pair * n_ff + r0 + threadIdx.x] = g / (1.f + __expf(-g)) * u;
    }
}

// Grid (n_out / (8 * kRowGroups), pairs): one warp per output row, kRowGroups rows per warp.
template <int ITERS>
__global__ void __launch_bounds__(256) k_down(const unsigned char* __restrict__ dw, int n_ff, int n_out,
                                              const Q81* __restrict__ h, const int32_t* __restrict__ ids,
                                              float* __restrict__ d) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int nb = n_ff / kMxfp4BlockElems;
    const size_t row_bytes = (size_t) nb * kMxfp4BlockBytes;
    XBlock* xs = reinterpret_cast<XBlock*>(smem);
    const int pair = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int e = ids[pair];
    const int r0 = blockIdx.x * kRowGroups * kRowsPerBlock + warp;
    const unsigned char* w0 = dw + ((size_t) e * n_out + r0) * row_bytes;
    const size_t step = (size_t) kRowsPerBlock * row_bytes;   // the warp's row of the next group
    uint32_t w[2][ITERS][5];
    row_load<ITERS>(w0, nb, lane, w[0]);
    stage_q8(h + (size_t) pair * nb, nb, xs);
    __syncthreads();
#pragma unroll
    for (int q = 0; q < kRowGroups; ++q) {
        if (q + 1 < kRowGroups) row_load<ITERS>(w0 + (q + 1) * step, nb, lane, w[(q + 1) & 1]);
        const float v = row_dot<ITERS>(w[q & 1], xs, nb, lane);
        if (lane == 0) d[(size_t) pair * n_out + r0 + q * kRowsPerBlock] = v;
    }
}

__global__ void k_combine(const float* __restrict__ d, const float* __restrict__ w, const float* __restrict__ shared,
                          int k, int n, float* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= n) return;
    float s = 0.f;
    for (int j = 0; j < k; ++j) s += w[t * k + j] * d[((size_t) t * k + j) * n + i];
    out[(size_t) t * n + i] = s + shared[(size_t) t * n + i];
}

__device__ float block_sum(float v, float* scratch) {
    v = warp_sum(v);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float t = 0.f;
    for (int i = 0; i < nw; ++i) t += scratch[i];
    return t;
}

__global__ void k_kv_rope(const float* __restrict__ raw, const float* __restrict__ norm_w,
                          const int32_t* __restrict__ pos, float eps, const float* __restrict__ freq,
                          float* __restrict__ kv_out, __half* __restrict__ ring, int ring_slots) {
    __shared__ float sh[kLatentW];
    __shared__ float red[32];
    const int t = blockIdx.x, c = threadIdx.x;
    const float v = raw[(size_t) t * kLatentW + c];
    const float ss = block_sum(v * v, red);
    const float y = v * (1.0f / sqrtf(ss / kLatentW + eps)) * norm_w[c];
    sh[c] = y;
    __syncthreads();
    float o = y;
    if (c >= kLatentW - kRopeW) {
        const int p = (c - (kLatentW - kRopeW)) >> 1;
        const float x0 = sh[kLatentW - kRopeW + 2 * p], x1 = sh[kLatentW - kRopeW + 2 * p + 1];
        float s, co;
        sincosf((float) pos[t] * freq[p], &s, &co);
        o = (c & 1) ? x0 * s + x1 * co : x0 * co - x1 * s;
    }
    if (kv_out) kv_out[(size_t) t * kLatentW + c] = o;
    if (ring) ring[(size_t) (pos[t] % ring_slots) * kLatentW + c] = __float2half(o);
}

__global__ void k_hc_mean(const float* __restrict__ x, int n, float* __restrict__ out, int out_stride) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t r = blockIdx.y;
    if (i >= n) return;
    const float* row = x + r * 4 * n;
    out[r * out_stride + i] = (row[i] + row[n + i] + row[2 * n + i] + row[3 * n + i]) * 0.25f;
}

__device__ __forceinline__ float bf16f(uint16_t b) { return __uint_as_float((uint32_t) b << 16); }

__global__ void k_markov_gather(const uint16_t* __restrict__ w1, const int32_t* __restrict__ tok,
                                float* __restrict__ out) {
    out[threadIdx.x] = bf16f(w1[(size_t) tok[0] * kMarkovRank + threadIdx.x]);
}

// One warp per vocab row; a lane takes 8 consecutive ranks (one 16-byte load).
__global__ void k_markov_bias(const uint16_t* __restrict__ w2, const float* __restrict__ w1p,
                              const float* __restrict__ logits, int n_vocab, float* __restrict__ col) {
    __shared__ float p[kMarkovRank];
    for (int i = threadIdx.x; i < kMarkovRank; i += blockDim.x) p[i] = w1p[i];
    __syncthreads();
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (warp >= n_vocab) return;
    const uint4 q = *reinterpret_cast<const uint4*>(w2 + (size_t) warp * kMarkovRank + lane * 8);
    const uint32_t words[4] = {q.x, q.y, q.z, q.w};
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        s += __uint_as_float(words[j] << 16) * p[lane * 8 + 2 * j];
        s += __uint_as_float(words[j] & 0xffff0000u) * p[lane * 8 + 2 * j + 1];
    }
    s = warp_sum(s);
    if (lane == 0) col[warp] = logits[warp] + s;
}

__global__ void k_conf(const float* __restrict__ w, const float* __restrict__ h, const float* __restrict__ w1p,
                       int n_hidden, float* __restrict__ out) {
    __shared__ float red[32];
    float s = 0.f;
    for (int i = threadIdx.x; i < n_hidden; i += blockDim.x) s += w[i] * h[i];
    for (int i = threadIdx.x; i < kMarkovRank; i += blockDim.x) s += w[n_hidden + i] * w1p[i];
    s = block_sum(s, red);
    if (threadIdx.x == 0) *out = 1.f / (1.f + expf(-s));
}

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

}  // namespace

void dspark_moe_gate_up(const void* gate_w, const void* up_w, int n_in, int n_ff, const void* x_q8_1, const int32_t* ids,
                        int n_tokens, int k, float clamp, float* h, void* stream) {
    const size_t smem = (size_t) (n_in / kMxfp4BlockElems) * sizeof(XBlock);
    if (n_in % kMxfp4BlockElems || n_in > 6144 || n_ff % (kRowGroups * kRowsPerBlock / 2)) throw std::invalid_argument("dspark_moe_gate_up: shape");
    const dim3 grid(n_ff / (kRowGroups * kRowsPerBlock / 2), n_tokens * k);
    const int iters = (n_in / kMxfp4BlockElems + 31) / 32;
    auto kern = iters <= 3 ? k_gate_up<3> : iters <= 5 ? k_gate_up<5> : k_gate_up<6>;
    kern<<<grid, 256, smem, (cudaStream_t) stream>>>((const unsigned char*) gate_w, (const unsigned char*) up_w, n_in,
                                                    n_ff, (const Q81*) x_q8_1, ids, k, clamp, h);
    check("dspark_moe_gate_up");
}

void dspark_moe_down(const void* down_w, int n_ff, int n_out, const void* h_q8_1, const int32_t* ids, int n_tokens, int k,
                     float* d, void* stream) {
    const size_t smem = (size_t) (n_ff / kMxfp4BlockElems) * sizeof(XBlock);
    if (n_ff % kMxfp4BlockElems || n_ff > 6144 || n_out % (kRowGroups * kRowsPerBlock)) throw std::invalid_argument("dspark_moe_down: shape");
    const dim3 grid(n_out / (kRowGroups * kRowsPerBlock), n_tokens * k);
    const int iters = (n_ff / kMxfp4BlockElems + 31) / 32;
    auto kern = iters <= 3 ? k_down<3> : iters <= 5 ? k_down<5> : k_down<6>;
    kern<<<grid, 256, smem, (cudaStream_t) stream>>>((const unsigned char*) down_w, n_ff, n_out, (const Q81*) h_q8_1, ids, d);
    check("dspark_moe_down");
}

void dspark_moe_combine(const float* d, const float* w, const float* shared, int n_tokens, int k, int n, float* out,
                        void* stream) {
    const dim3 grid((n + 255) / 256, n_tokens);
    k_combine<<<grid, 256, 0, (cudaStream_t) stream>>>(d, w, shared, k, n, out);
    check("dspark_moe_combine");
}

void dspark_kv_rope(const float* raw, const float* norm_w, const int32_t* pos, int n_tokens, float eps,
                    const float* rope_freq, float* kv_out, uint16_t* ring, int ring_slots, void* stream) {
    k_kv_rope<<<n_tokens, kLatentW, 0, (cudaStream_t) stream>>>(raw, norm_w, pos, eps, rope_freq, kv_out,
                                                                 reinterpret_cast<__half*>(ring), ring_slots);
    check("dspark_kv_rope");
}

void dspark_hc_mean(const float* x, int n_rows, int n, float* out, int out_stride, void* stream) {
    k_hc_mean<<<dim3((n + 255) / 256, n_rows), 256, 0, (cudaStream_t) stream>>>(x, n, out, out_stride);
    check("dspark_hc_mean");
}

void dspark_markov_gather(const uint16_t* w1, const int32_t* tok, float* out, void* stream) {
    k_markov_gather<<<1, kMarkovRank, 0, (cudaStream_t) stream>>>(w1, tok, out);
    check("dspark_markov_gather");
}

void dspark_markov_bias(const uint16_t* w2, const float* w1p, const float* logits, int n_vocab, float* col,
                        void* stream) {
    k_markov_bias<<<(n_vocab * 32 + 255) / 256, 256, 0, (cudaStream_t) stream>>>(w2, w1p, logits, n_vocab, col);
    check("dspark_markov_bias");
}

void dspark_conf(const float* w, const float* h, const float* w1p, int n_hidden, float* out, void* stream) {
    k_conf<<<1, 256, 0, (cudaStream_t) stream>>>(w, h, w1p, n_hidden, out);
    check("dspark_conf");
}

}  // namespace basalt::models::deepseek41
