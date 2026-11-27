#include "models/deepseek4/dspark.hpp"

#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/mxfp4_scale.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <stdexcept>
#include <string>

namespace basalt::models::deepseek4 {
namespace {

constexpr int kArgmaxThreads = 256;
constexpr int kArgmaxBlocks = 256;

__global__ void __launch_bounds__(kArgmaxThreads)
markov_argmax_stage1_kernel(const float* __restrict__ base, const float* __restrict__ bias, int n_vocab,
                            float* __restrict__ bmax, int32_t* __restrict__ bidx) {
    float best_v = -FLT_MAX;
    int32_t best_i = 0;
    for (int i = blockIdx.x * kArgmaxThreads + threadIdx.x; i < n_vocab; i += kArgmaxBlocks * kArgmaxThreads) {
        const float v = base[i] + bias[i];
        if (v > best_v) {
            best_v = v;
            best_i = i;
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, best_v, o);
        const int32_t oi = __shfl_down_sync(0xffffffffu, best_i, o);
        if (ov > best_v || (ov == best_v && oi < best_i)) {
            best_v = ov;
            best_i = oi;
        }
    }
    __shared__ float sv[8];
    __shared__ int32_t si[8];
    const int warp = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) {
        sv[warp] = best_v;
        si[warp] = best_i;
    }
    __syncthreads();
    if (warp == 0) {
        const int lane = threadIdx.x;
        float v = lane < 8 ? sv[lane] : -FLT_MAX;
        int32_t idx = lane < 8 ? si[lane] : 0;
        for (int o = 4; o > 0; o >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, v, o);
            const int32_t oi = __shfl_down_sync(0xffffffffu, idx, o);
            if (ov > v || (ov == v && oi < idx)) {
                v = ov;
                idx = oi;
            }
        }
        if (lane == 0) {
            bmax[blockIdx.x] = v;
            bidx[blockIdx.x] = idx;
        }
    }
}

__global__ void __launch_bounds__(kArgmaxBlocks)
markov_argmax_stage2_kernel(const float* __restrict__ bmax, const int32_t* __restrict__ bidx,
                            int32_t* __restrict__ out_tok) {
    float v = bmax[threadIdx.x];
    int32_t idx = bidx[threadIdx.x];
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, v, o);
        const int32_t oi = __shfl_down_sync(0xffffffffu, idx, o);
        if (ov > v || (ov == v && oi < idx)) {
            v = ov;
            idx = oi;
        }
    }
    __shared__ float sv[8];
    __shared__ int32_t si[8];
    const int warp = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) {
        sv[warp] = v;
        si[warp] = idx;
    }
    __syncthreads();
    if (warp == 0) {
        const int lane = threadIdx.x;
        float fv = lane < 8 ? sv[lane] : -FLT_MAX;
        int32_t fi = lane < 8 ? si[lane] : 0;
        for (int o = 4; o > 0; o >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, fv, o);
            const int32_t oi = __shfl_down_sync(0xffffffffu, fi, o);
            if (ov > fv || (ov == fv && oi < fi)) {
                fv = ov;
                fi = oi;
            }
        }
        if (lane == 0) *out_tok = fi;
    }
}

__global__ void __launch_bounds__(256)
confidence_kernel(const float* __restrict__ w, const float* __restrict__ h, const float* __restrict__ me,
                  int n_embd, int rank, float* __restrict__ out_conf) {
    float acc = 0.f;
    for (int i = threadIdx.x; i < n_embd; i += 256) acc = fmaf(w[i], h[i], acc);
    for (int i = threadIdx.x; i < rank; i += 256) acc = fmaf(w[n_embd + i], me[i], acc);
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    __shared__ float s[8];
    if ((threadIdx.x & 31) == 0) s[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 8) {
        float v = s[threadIdx.x];
        for (int o = 4; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (threadIdx.x == 0) *out_conf = 1.f / (1.f + expf(-v));
    }
}

__global__ void add_rows_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                float* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}


// ---- MXFP4 routed experts of the drafter's stages, kept experts packed one matrix per stage ------------------------------
// The kernels are the DeepSeek-V4.1 drafter's (models/deepseek41/dspark_kernels.cu, MIT): one warp per output row,
// staged q8_1 activations, E2M1 nibbles doubled to int8 and met in dp4a, both block scales applied once.

constexpr int kRowsPerBlock = 8;
constexpr int kMoeRowGroups = 4;
constexpr int kMxfp4BlockBytes = 17;   // one E8M0 scale byte, then 16 bytes of packed 4-bit values
constexpr int kMxfp4BlockElems = 32;

struct Q81 {
    __half2 ds;
    int8_t qs[32];
};

struct XBlock {
    int4 lo, hi;
    float d;
    float pad[3];
};

__device__ __forceinline__ void stage_q8(const Q81* __restrict__ x, int nb, XBlock* xs) {
    for (int b = threadIdx.x; b < nb; b += blockDim.x) {
        const int* q = reinterpret_cast<const int*>(x[b].qs);
        int w[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int a = q[i], c = q[i + 4];
            w[2 * i] = (int) __byte_perm(a, c, 0x5140);
            w[2 * i + 1] = (int) __byte_perm(a, c, 0x7362);
        }
        xs[b].lo = make_int4(w[0], w[1], w[2], w[3]);
        xs[b].hi = make_int4(w[4], w[5], w[6], w[7]);
        xs[b].d = __low2float(x[b].ds);
    }
}

constexpr uint32_t kPos0 = 0x03020100u, kPos1 = 0x0c080604u;
constexpr uint32_t kNeg0 = 0xfdfeff00u, kNeg1 = 0xf4f8fafcu;

__device__ __forceinline__ int e2m1x4(uint32_t q) {
    const uint32_t pos = __byte_perm(kPos0, kPos1, q), neg = __byte_perm(kNeg0, kNeg1, q);
    return (int) __byte_perm(pos, neg, 0x3210 | ((q & 0x8888u) >> 1));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

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

template <int ITERS>
__device__ __forceinline__ float row_dot(const uint32_t (&w)[ITERS][5], const XBlock* xs, int nb, int lane) {
    float acc = 0.f;
#pragma unroll
    for (int it = 0; it < ITERS; ++it) {
        const int b = lane + 32 * it;
        if (b >= nb) continue;
        const int sh = (b & 3) * 8;
        const float scale = mxfp4_scale((uint8_t) ((w[it][0] >> sh) & 0xff));
        const int4 lo = xs[b].lo, hi = xs[b].hi;
        const int xw[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
        int s = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t n = __funnelshift_rc(w[it][k], w[it][k + 1], sh + 8);
            s = __dp4a(e2m1x4(n), xw[2 * k], s);
            s = __dp4a(e2m1x4(n >> 16), xw[2 * k + 1], s);
        }
        acc += scale * (xs[b].d * (float) s);
    }
    return warp_sum(acc);
}

template <int ITERS>
__global__ void __launch_bounds__(256) k_gate_up(const unsigned char* __restrict__ gw,
                                                 const unsigned char* __restrict__ uw, int n_in, int n_ff,
                                                 const Q81* __restrict__ x, const int32_t* __restrict__ ids, int k,
                                                 float clamp, const int32_t* __restrict__ live, float* __restrict__ h) {
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ float res[kMoeRowGroups][8];
    const int nb = n_in / kMxfp4BlockElems;
    const size_t row_bytes = (size_t) nb * kMxfp4BlockBytes;
    XBlock* xs = reinterpret_cast<XBlock*>(smem);
    const int pair = blockIdx.y, t = pair / k, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (live && t >= *live) return;
    const int e = ids[pair];
    const int r0 = blockIdx.x * kMoeRowGroups * (kRowsPerBlock / 2);
    const unsigned char* w0 = ((warp & 1) ? uw : gw) + ((size_t) e * n_ff + r0 + (warp >> 1)) * row_bytes;
    const size_t step = (size_t) (kRowsPerBlock / 2) * row_bytes;
    uint32_t w[2][ITERS][5];
    row_load<ITERS>(w0, nb, lane, w[0]);
    stage_q8(x + (size_t) t * nb, nb, xs);
    __syncthreads();
#pragma unroll
    for (int q = 0; q < kMoeRowGroups; ++q) {
        if (q + 1 < kMoeRowGroups) row_load<ITERS>(w0 + (q + 1) * step, nb, lane, w[(q + 1) & 1]);
        const float v = row_dot<ITERS>(w[q & 1], xs, nb, lane);
        if (lane == 0) res[q][warp] = v;
    }
    __syncthreads();
    if (threadIdx.x < kMoeRowGroups * (kRowsPerBlock / 2)) {
        const int q = threadIdx.x / (kRowsPerBlock / 2), i = threadIdx.x % (kRowsPerBlock / 2);
        const float g = fminf(res[q][2 * i], clamp);
        const float u = fminf(fmaxf(res[q][2 * i + 1], -clamp), clamp);
        h[(size_t) pair * n_ff + r0 + threadIdx.x] = g / (1.f + __expf(-g)) * u;
    }
}

template <int ITERS>
__global__ void __launch_bounds__(256) k_down(const unsigned char* __restrict__ dw, int n_ff, int n_out,
                                              const Q81* __restrict__ h, const int32_t* __restrict__ ids, int k,
                                              const int32_t* __restrict__ live, float* __restrict__ d) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int nb = n_ff / kMxfp4BlockElems;
    const size_t row_bytes = (size_t) nb * kMxfp4BlockBytes;
    XBlock* xs = reinterpret_cast<XBlock*>(smem);
    const int pair = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = blockIdx.x * kMoeRowGroups * kRowsPerBlock + warp;
    if (live && pair / k >= *live) {
        if (lane == 0)
            for (int q = 0; q < kMoeRowGroups; ++q) d[(size_t) pair * n_out + r0 + q * kRowsPerBlock] = 0.f;
        return;
    }
    const int e = ids[pair];
    const unsigned char* w0 = dw + ((size_t) e * n_out + r0) * row_bytes;
    const size_t step = (size_t) kRowsPerBlock * row_bytes;
    uint32_t w[2][ITERS][5];
    row_load<ITERS>(w0, nb, lane, w[0]);
    stage_q8(h + (size_t) pair * nb, nb, xs);
    __syncthreads();
#pragma unroll
    for (int q = 0; q < kMoeRowGroups; ++q) {
        if (q + 1 < kMoeRowGroups) row_load<ITERS>(w0 + (q + 1) * step, nb, lane, w[(q + 1) & 1]);
        const float v = row_dot<ITERS>(w[q & 1], xs, nb, lane);
        if (lane == 0) d[(size_t) pair * n_out + r0 + q * kRowsPerBlock] = v;
    }
}

__device__ __forceinline__ float bf16f(uint16_t b) { return __uint_as_float((uint32_t) b << 16); }

__global__ void k_markov_gather(const uint16_t* __restrict__ w1, const int32_t* __restrict__ tok,
                                float* __restrict__ out) {
    out[threadIdx.x] = bf16f(w1[(size_t) tok[0] * 256 + threadIdx.x]);
}

// One warp per vocab row; a lane takes 8 consecutive ranks (one 16-byte load). The summed bias only: the argmax adds
// the row's own logits.
__global__ void k_markov_bias(const uint16_t* __restrict__ w2, const float* __restrict__ w1p, int n_vocab,
                              float* __restrict__ col) {
    __shared__ float p[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) p[i] = w1p[i];
    __syncthreads();
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (warp >= n_vocab) return;
    const uint4 q = *reinterpret_cast<const uint4*>(w2 + (size_t) warp * 256 + lane * 8);
    const uint32_t words[4] = {q.x, q.y, q.z, q.w};
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        s += __uint_as_float(words[j] << 16) * p[lane * 8 + 2 * j];
        s += __uint_as_float(words[j] & 0xffff0000u) * p[lane * 8 + 2 * j + 1];
    }
    s = warp_sum(s);
    if (lane == 0) col[warp] = s;
}

__global__ void k_commit_rows(const float* __restrict__ staged, int stage_slots, float* ring0, float* ring1,
                              float* ring2, int ring_slots, int64_t pos0) {
    const int64_t p = pos0 + blockIdx.x;
    const int l = blockIdx.y;
    float* ring = l == 0 ? ring0 : l == 1 ? ring1 : ring2;
    ring[(size_t) (p % ring_slots) * 512 + threadIdx.x] =
        staged[((size_t) l * stage_slots + (size_t) (p % stage_slots)) * 512 + threadIdx.x];
}

void check_dspark(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("deepseek4 dspark ") + what + ": " + cudaGetErrorString(e));
}

}  // namespace

void dspark_markov_argmax(const float* base_logits, const float* markov_bias, int n_vocab,
                          float* block_max, int32_t* block_idx, int32_t* out_tok, void* stream) {
    const cudaStream_t s = (cudaStream_t) stream;
    markov_argmax_stage1_kernel<<<kArgmaxBlocks, kArgmaxThreads, 0, s>>>(base_logits, markov_bias, n_vocab, block_max, block_idx);
    check_dspark("argmax stage1");
    markov_argmax_stage2_kernel<<<1, kArgmaxBlocks, 0, s>>>(block_max, block_idx, out_tok);
    check_dspark("argmax stage2");
}

void dspark_confidence(const float* conf_w, const float* h_out, const float* me,
                       int n_embd, int rank, float* out_conf, void* stream) {
    confidence_kernel<<<1, 256, 0, (cudaStream_t) stream>>>(conf_w, h_out, me, n_embd, rank, out_conf);
    check_dspark("confidence");
}

void dspark_add_rows(const float* a, const float* b, float* out, int64_t count, void* stream) {
    if (count <= 0) return;
    add_rows_kernel<<<(unsigned) ((count + 255) / 256), 256, 0, (cudaStream_t) stream>>>(a, b, out, count);
    check_dspark("add_rows");
}

void dspark_moe_gate_up(const void* gate_w, const void* up_w, int n_in, int n_ff, const void* x_q8_1,
                        const int32_t* ids, int n_tokens, int k, float clamp, const int32_t* live_rows, float* h,
                        void* stream) {
    const size_t smem = (size_t) (n_in / kMxfp4BlockElems) * sizeof(XBlock);
    if (n_in % kMxfp4BlockElems || n_in > 6144 || n_ff % (kMoeRowGroups * kRowsPerBlock / 2))
        throw std::invalid_argument("dspark_moe_gate_up: shape");
    const dim3 grid(n_ff / (kMoeRowGroups * kRowsPerBlock / 2), n_tokens * k);
    const int iters = (n_in / kMxfp4BlockElems + 31) / 32;
    auto kern = iters <= 3 ? k_gate_up<3> : iters <= 5 ? k_gate_up<5> : k_gate_up<6>;
    kern<<<grid, 256, smem, (cudaStream_t) stream>>>((const unsigned char*) gate_w, (const unsigned char*) up_w, n_in,
                                                    n_ff, (const Q81*) x_q8_1, ids, k, clamp, live_rows, h);
    check_dspark("moe gate_up");
}

void dspark_moe_down(const void* down_w, int n_ff, int n_out, const void* h_q8_1, const int32_t* ids, int n_tokens,
                     int k, const int32_t* live_rows, float* d, void* stream) {
    const size_t smem = (size_t) (n_ff / kMxfp4BlockElems) * sizeof(XBlock);
    if (n_ff % kMxfp4BlockElems || n_ff > 6144 || n_out % (kMoeRowGroups * kRowsPerBlock))
        throw std::invalid_argument("dspark_moe_down: shape");
    const dim3 grid(n_out / (kMoeRowGroups * kRowsPerBlock), n_tokens * k);
    const int iters = (n_ff / kMxfp4BlockElems + 31) / 32;
    auto kern = iters <= 3 ? k_down<3> : iters <= 5 ? k_down<5> : k_down<6>;
    kern<<<grid, 256, smem, (cudaStream_t) stream>>>((const unsigned char*) down_w, n_ff, n_out, (const Q81*) h_q8_1,
                                                    ids, k, live_rows, d);
    check_dspark("moe down");
}

void dspark_markov_gather(const uint16_t* w1, const int32_t* tok, float* out, void* stream) {
    k_markov_gather<<<1, 256, 0, (cudaStream_t) stream>>>(w1, tok, out);
    check_dspark("markov gather");
}

void dspark_markov_bias(const uint16_t* w2, const float* w1p, int n_vocab, float* col, void* stream) {
    k_markov_bias<<<(n_vocab * 32 + 255) / 256, 256, 0, (cudaStream_t) stream>>>(w2, w1p, n_vocab, col);
    check_dspark("markov bias");
}

void dspark_commit_rows(const float* staged, int stage_slots, float* ring0, float* ring1, float* ring2,
                        int ring_slots, int64_t pos0, int n, void* stream) {
    if (n <= 0) return;
    k_commit_rows<<<dim3((unsigned) n, 3), 512, 0, (cudaStream_t) stream>>>(staged, stage_slots, ring0, ring1, ring2,
                                                                           ring_slots, pos0);
    check_dspark("commit rows");
}

}  // namespace basalt::models::deepseek4
