// src/kernels/cuda/moe_ffn.cu - the routed-expert FFN steps of moe_ffn.hpp.
//
// The expert products run on INT8 dot products (dp4a): the activations are rounded to INT8 with a scale per 16 values, in the
// order the weights' nibbles come (so a 32-bit word of weights meets four activations per half), and an E2M1 code becomes
// twice its value as an INT8 through two byte permutes.  A group's integer sum scales by the weight's UE4M3 scale, half
// (for the doubled codes) and the activation scale; the tensor scale applies once at the end.
#include "basalt/kernels/moe_ffn.hpp"

#include "basalt/log.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdlib>

namespace basalt::kernels::moe {
namespace {

constexpr int kWarps = 4;              // warps of an expert-product block
constexpr int kRowsPerWarp = 4;        // output rows a warp computes
constexpr int kRowsPerBlock = kWarps * kRowsPerWarp;
constexpr int kMaxExperts = 512;       // top_k's shared probabilities
constexpr int kMaxUsed = 16;

// the value of a UE4M3 code (the codes of finite scales: below 127): 2^-9 m for the subnormals, (8 + m) 2^(e - 10) else
__device__ __forceinline__ float ue4m3(unsigned c) {
    const unsigned e = c >> 3, m = c & 7;
    return e == 0 ? (float) m * (1.0f / 512.0f) : (float) (8 + m) * __int_as_float((int) ((e + 117) << 23));
}

__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// four E2M1 codes (the nibbles of the low 16 bits of h, the lowest first) as four INT8 holding twice their values
__device__ __forceinline__ int e2m1x4_s8(unsigned h) {
    const unsigned mag = __byte_perm(0x03020100u, 0x0C080604u, h & 0x7777u);                    // 0 1 2 3 | 4 6 8 12
    const unsigned neg = __byte_perm(0x0000FF00u, 0u, (h >> 3) & 0x1111u);                      // 0xFF where the sign is set
    return (int) __vsub4(mag ^ neg, neg);
}

// one 16-value group's integer dot product: its two code words against 16 INT8 activations in the weights' nibble order
__device__ __forceinline__ int group_dot(unsigned wa, unsigned wb, const int* __restrict__ xq4) {
    int acc = __dp4a(e2m1x4_s8(wa & 0xFFFFu), xq4[0], 0);
    acc = __dp4a(e2m1x4_s8(wa >> 16), xq4[1], acc);
    acc = __dp4a(e2m1x4_s8(wb & 0xFFFFu), xq4[2], acc);
    return __dp4a(e2m1x4_s8(wb >> 16), xq4[3], acc);
}

// Rows of `cols` values (cols / 64 blocks of 36 bytes) of x against the activations in shared memory (INT8 in the nibble
// order, a scale per group), one block a lane.
__device__ __forceinline__ float row_dot(const uint8_t* __restrict__ row, int blocks, const int8_t* xq, const float* xs, int lane) {
    float sum = 0.0f;
    for (int b = lane; b < blocks; b += 32) {
        const uint32_t* w = reinterpret_cast<const uint32_t*>(row + (size_t) b * 36);
        uint32_t v[9];
#pragma unroll
        for (int i = 0; i < 9; ++i) v[i] = w[i];
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int gi = group_dot(v[1 + 2 * s], v[2 + 2 * s], reinterpret_cast<const int*>(xq + 64 * b + 16 * s));
            sum += (float) gi * (ue4m3((v[0] >> (8 * s)) & 0xFF) * 0.5f * xs[4 * b + s]);
        }
    }
    return warp_sum(sum);
}

// 16 values (16-byte aligned) rounded to INT8 with their scale amax / 127, stored in the order v0 v8 v1 v9, then the same pairs up to v7 v15
__device__ __forceinline__ void quantize_group(const float* __restrict__ x16, int8_t* __restrict__ q16, float* __restrict__ scale) {
    float v[16];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float4 f = reinterpret_cast<const float4*>(x16)[i];
        v[4 * i] = f.x; v[4 * i + 1] = f.y; v[4 * i + 2] = f.z; v[4 * i + 3] = f.w;
    }
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(v[i]));
    *scale = amax / 127.0f;
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    uint32_t out[4];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        uint32_t word = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int slot = 4 * w + b, src = (slot & 1) ? (slot >> 1) + 8 : (slot >> 1);
            word |= (uint32_t) (uint8_t) (int8_t) __float2int_rn(v[src] * inv) << (8 * b);
        }
        out[w] = word;
    }
    *reinterpret_cast<uint4*>(q16) = make_uint4(out[0], out[1], out[2], out[3]);
}

__global__ void quantize_kernel(const float* __restrict__ x, int groups, int8_t* __restrict__ q, float* __restrict__ s) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g < groups) quantize_group(x + (size_t) g * 16, q + (size_t) g * 16, s + g);
}

// gate and up of expert idx[p] over x_t: a block computes kRowsPerBlock output rows (a warp kRowsPerWarp of them, both matrices)
__global__ void __launch_bounds__(32 * kWarps) gate_up_kernel(Experts e, const int8_t* __restrict__ xq, const float* __restrict__ xs,
                                                              const int32_t* __restrict__ idx, int n_used, float* __restrict__ h) {
    extern __shared__ __align__(16) unsigned char smem[];
    int8_t* sq = reinterpret_cast<int8_t*>(smem);
    float* ss = reinterpret_cast<float*>(smem + e.hidden);
    const int p = blockIdx.y, t = p / n_used, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int i = threadIdx.x; i < e.hidden / 16; i += blockDim.x) reinterpret_cast<uint4*>(sq)[i] = reinterpret_cast<const uint4*>(xq + (size_t) t * e.hidden)[i];
    for (int i = threadIdx.x; i < e.hidden / 16; i += blockDim.x) ss[i] = xs[(size_t) t * (e.hidden / 16) + i];
    __syncthreads();
    const uint8_t* blob = e.base + (size_t) idx[p] * e.blob;
    const size_t row_bytes = (size_t) (e.hidden / 64) * 36, mat = (size_t) e.ff * row_bytes;
    const float* tail = reinterpret_cast<const float*>(blob + 3 * mat);
    const float s_gate = tail[0], s_up = tail[1];
#pragma unroll
    for (int j = 0; j < kRowsPerWarp; ++j) {
        const int r = blockIdx.x * kRowsPerBlock + warp * kRowsPerWarp + j;
        if (r >= e.ff) break;
        const float g = row_dot(blob + (size_t) r * row_bytes, e.hidden / 64, sq, ss, lane) * s_gate;
        const float u = row_dot(blob + mat + (size_t) r * row_bytes, e.hidden / 64, sq, ss, lane) * s_up;
        if (lane == 0) h[(size_t) p * e.ff + r] = silu(g) * u;
    }
}

// down of expert idx[p] over h_p (rounded to INT8 here, once per block): a lane a group of 16 of the ff inputs
__global__ void __launch_bounds__(32 * kWarps) down_kernel(Experts e, const float* __restrict__ h, const int32_t* __restrict__ idx,
                                                           float* __restrict__ y) {
    extern __shared__ __align__(16) unsigned char smem[];
    int8_t* sq = reinterpret_cast<int8_t*>(smem);
    float* ss = reinterpret_cast<float*>(smem + e.ff);
    const int p = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (threadIdx.x < e.ff / 16) quantize_group(h + (size_t) p * e.ff + 16 * threadIdx.x, sq + 16 * threadIdx.x, ss + threadIdx.x);
    __syncthreads();
    const uint8_t* blob = e.base + (size_t) idx[p] * e.blob;
    const size_t gu_row = (size_t) (e.hidden / 64) * 36, row_bytes = (size_t) (e.ff / 64) * 36;
    const uint8_t* down = blob + 2 * (size_t) e.ff * gu_row;
    const float s_down = reinterpret_cast<const float*>(down + (size_t) e.hidden * row_bytes)[2];
#pragma unroll
    for (int j = 0; j < kRowsPerWarp; ++j) {
        const int c = blockIdx.x * kRowsPerBlock + warp * kRowsPerWarp + j;
        if (c >= e.hidden) break;
        float sum = 0.0f;
        for (int g = lane; g < e.ff / 16; g += 32) {
            const uint8_t* blk = down + (size_t) c * row_bytes + (size_t) (g >> 2) * 36;
            const unsigned sc = (unsigned) blk[g & 3];
            const uint32_t* w = reinterpret_cast<const uint32_t*>(blk + 4 + 8 * (g & 3));
            const int gi = group_dot(w[0], w[1], reinterpret_cast<const int*>(sq + 16 * g));
            sum += (float) gi * (ue4m3(sc) * 0.5f * ss[g]);
        }
        sum = warp_sum(sum);
        if (lane == 0) y[(size_t) p * e.hidden + c] = sum * s_down;
    }
}

// softmax and the n_used largest of a token's logits (a block of 256 threads per token)
__global__ void top_k_kernel(const float* __restrict__ logits, int ld, int n_expert, int n_used, int32_t* __restrict__ idx,
                             float* __restrict__ wgt) {
    __shared__ float p[kMaxExperts];
    __shared__ float red[32];
    __shared__ float top_v[kMaxUsed];
    __shared__ int top_i[kMaxUsed];
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nwarps = blockDim.x >> 5;
    const float* l = logits + (size_t) t * ld;
    float m = -INFINITY;
    for (int i = tid; i < n_expert; i += blockDim.x) m = fmaxf(m, l[i]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) red[warp] = m;
    __syncthreads();
    m = red[0];
    for (int w = 1; w < nwarps; ++w) m = fmaxf(m, red[w]);
    __syncthreads();
    float z = 0.0f;
    for (int i = tid; i < n_expert; i += blockDim.x) {
        const float v = __expf(l[i] - m);
        p[i] = v;
        z += v;
    }
    z = warp_sum(z);
    if (lane == 0) red[warp] = z;
    __syncthreads();
    z = 0.0f;
    for (int w = 0; w < nwarps; ++w) z += red[w];
    __syncthreads();
    for (int i = tid; i < n_expert; i += blockDim.x) p[i] /= z;
    __syncthreads();
    if (tid < 32) {   // n_used rounds of arg-max over the warp's slice of the probabilities, ties to the lower expert
        float sum = 0.0f;
        for (int k = 0; k < n_used; ++k) {
            float bv = -1.0f;
            int bi = 0x7FFFFFFF;
            for (int i = lane; i < n_expert; i += 32)
                if (p[i] > bv) { bv = p[i]; bi = i; }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            if (lane == 0) { top_v[k] = bv; top_i[k] = bi; p[bi] = -2.0f; }
            sum += bv;
            __syncwarp();
        }
        if (lane == 0) {
            const float inv = 1.0f / fmaxf(sum, 6.103515625e-5f);
            for (int k = 0; k < n_used; ++k) {
                idx[(size_t) t * n_used + k] = top_i[k];
                wgt[(size_t) t * n_used + k] = top_v[k] * inv;
            }
        }
    }
}

// the pairs grouped by expert (one block: a prompt chunk's pairs are few)
__global__ void __launch_bounds__(1024) group_kernel(const int32_t* __restrict__ idx, int pairs, int n_used, int n_expert,
                                                     int32_t* __restrict__ bounds, int32_t* __restrict__ tok,
                                                     int32_t* __restrict__ dst, int32_t* __restrict__ exp) {
    __shared__ int32_t cnt[kMaxExperts], cur[kMaxExperts];
    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) cnt[e] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < pairs; i += blockDim.x) atomicAdd(&cnt[idx[i]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int run = 0;
        for (int e = 0; e < n_expert; ++e) {
            bounds[e] = run;
            cur[e] = run;
            run += cnt[e];
        }
        bounds[n_expert] = run;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < pairs; i += blockDim.x) {
        const int e = idx[i], at = atomicAdd(&cur[e], 1);
        tok[at] = i / n_used;
        dst[at] = i;
        exp[at] = e;
    }
}

// logits[t][r] = x_t . w_r for the router's rows: a warp a row, its weights in registers while it walks the tokens
template <int NC>
__global__ void __launch_bounds__(256) router_kernel(const float* __restrict__ x, const uint16_t* __restrict__ w, int T, int rows,
                                                     float* __restrict__ logits, int ld) {
    const int lane = threadIdx.x & 31, r = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (r >= rows) return;
    constexpr int K = NC * 256;
    float wr[NC * 8];
#pragma unroll
    for (int i = 0; i < NC; ++i) {
        const uint4 u = reinterpret_cast<const uint4*>(w + (size_t) r * K)[i * 32 + lane];
        const uint32_t v[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            wr[8 * i + 2 * j] = __uint_as_float(v[j] << 16);
            wr[8 * i + 2 * j + 1] = __uint_as_float(v[j] & 0xFFFF0000u);
        }
    }
    for (int t = 0; t < T; ++t) {
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < NC; ++i) {
            const float4* xp = reinterpret_cast<const float4*>(x + (size_t) t * K) + (size_t) (i * 32 + lane) * 2;
            const float4 a = xp[0], b = xp[1];
            acc += wr[8 * i] * a.x + wr[8 * i + 1] * a.y + wr[8 * i + 2] * a.z + wr[8 * i + 3] * a.w;
            acc += wr[8 * i + 4] * b.x + wr[8 * i + 5] * b.y + wr[8 * i + 6] * b.z + wr[8 * i + 7] * b.w;
        }
        acc = warp_sum(acc);
        if (lane == 0) logits[(size_t) t * ld + r] = acc;
    }
}

__global__ void combine_kernel(uint16_t* __restrict__ resid, const float* __restrict__ y, const float* __restrict__ wgt,
                               const float* __restrict__ sgate, int ld, const float* __restrict__ shared, int n_used, int hidden) {
    const int t = blockIdx.x;
    const float sg = 1.0f / (1.0f + __expf(-sgate[(size_t) t * ld]));
    for (int c = threadIdx.x; c < hidden; c += blockDim.x) {
        float routed = 0.0f;
        for (int k = 0; k < n_used; ++k) routed += wgt[(size_t) t * n_used + k] * y[((size_t) t * n_used + k) * hidden + c];
        const size_t at = (size_t) t * hidden + c;
        const float r = __uint_as_float((uint32_t) resid[at] << 16);
        resid[at] = __bfloat16_as_ushort(__float2bfloat16_rn((r + routed) + sg * shared[at]));
    }
}

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "%s: %s", what, cudaGetErrorString(e));
        std::abort();
    }
}

}  // namespace

size_t tail_offset(int hidden, int ff) {
    return 2 * (size_t) ff * (size_t) (hidden / 64) * 36 + (size_t) hidden * (size_t) (ff / 64) * 36;
}

size_t blob_bytes(int hidden, int ff) { return (tail_offset(hidden, ff) + 16 + 143) / 144 * 144; }

void router(const float* x, const uint16_t* w, int T, int hidden, int rows, float* logits, int ld, void* stream) {
    if (T <= 0) return;
    if (hidden != 2048) {
        basalt::log::error("basalt", "moe router: hidden %d (the kernel is built for 2048)", hidden);
        std::abort();
    }
    router_kernel<8><<<(unsigned) ((rows + 7) / 8), 256, 0, (cudaStream_t) stream>>>(x, w, T, rows, logits, ld);
    check("moe router");
}

void top_k(const float* logits, int ld, int T, int n_expert, int n_used, int32_t* idx, float* wgt, void* stream) {
    if (T <= 0) return;
    if (n_expert > kMaxExperts || n_used > kMaxUsed || n_used > n_expert) {
        basalt::log::error("basalt", "moe top_k: %d experts, %d used", n_expert, n_used);
        std::abort();
    }
    top_k_kernel<<<T, 256, 0, (cudaStream_t) stream>>>(logits, ld, n_expert, n_used, idx, wgt);
    check("moe top_k");
}

void group(const int32_t* idx, int T, int n_used, int n_expert, int32_t* bounds, int32_t* tok, int32_t* dst, int32_t* exp, void* stream) {
    if (T <= 0) return;
    if (n_expert > kMaxExperts || (size_t) T * n_used > 32768) {
        basalt::log::error("basalt", "moe group: %d experts, %d pairs", n_expert, T * n_used);
        std::abort();
    }
    group_kernel<<<1, 1024, 0, (cudaStream_t) stream>>>(idx, T * n_used, n_used, n_expert, bounds, tok, dst, exp);
    check("moe group");
}

void quantize(const float* x, int rows, int cols, int8_t* q, float* s, void* stream) {
    if (rows <= 0) return;
    const int groups = rows * (cols / 16);
    quantize_kernel<<<(unsigned) ((groups + 127) / 128), 128, 0, (cudaStream_t) stream>>>(x, groups, q, s);
    check("moe quantize");
}

void gate_up(const Experts& e, const int8_t* xq, const float* xs, const int32_t* idx, int T, int n_used, float* h, void* stream) {
    if (T <= 0) return;
    const size_t smem = (size_t) e.hidden + (size_t) e.hidden / 16 * sizeof(float);
    gate_up_kernel<<<dim3((unsigned) ((e.ff + kRowsPerBlock - 1) / kRowsPerBlock), (unsigned) (T * n_used)), 32 * kWarps, smem,
                     (cudaStream_t) stream>>>(e, xq, xs, idx, n_used, h);
    check("moe gate_up");
}

void down(const Experts& e, const float* h, const int32_t* idx, int T, int n_used, float* y, void* stream) {
    if (T <= 0) return;
    const size_t smem = (size_t) e.ff + (size_t) e.ff / 16 * sizeof(float);
    down_kernel<<<dim3((unsigned) ((e.hidden + kRowsPerBlock - 1) / kRowsPerBlock), (unsigned) (T * n_used)), 32 * kWarps, smem,
                  (cudaStream_t) stream>>>(e, h, idx, y);
    check("moe down");
}

void combine(uint16_t* resid, const float* y, const float* wgt, const float* sgate, int ld, const float* shared, int T, int n_used,
             int hidden, void* stream) {
    if (T <= 0) return;
    combine_kernel<<<T, 256, 0, (cudaStream_t) stream>>>(resid, y, wgt, sgate, ld, shared, n_used, hidden);
    check("moe combine");
}

}  // namespace basalt::kernels::moe
