// src/models/inkling2/kernels/dflash2.cu - Inkling 2's DFlash 2 block-diffusion drafter kernels
// (include/basalt/models/inkling2/dflash2_kernels.hpp).
#include "basalt/models/inkling2/dflash2_kernels.hpp"

#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/q8_1_finite.hpp"
#include "basalt/models/inkling2/kernels.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace basalt::models::inkling2 {
namespace {

using basalt::kernels::launch_pdl;
using basalt::kernels::pdl_launch_dependents;
using basalt::kernels::pdl_wait;
using basalt::kernels::q8_1_ds;
using basalt::kernels::q8_1_finite;
using basalt::kernels::q8_1_quant;

constexpr int kThreads = 256;
constexpr int kPer = kWidth / kThreads;   // 9
constexpr int kQ8Block = 34;
constexpr int kHeadNb = kWidth / 32;      // 72
constexpr int kQ8Row = kHeadNb * kQ8Block;
constexpr int kTopChunk = 4096;

struct Q81 {
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q81) == 36, "ggml's block_q8_1");

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("inkling2 dflash2 ") + what + ": " + cudaGetErrorString(e));
}

__device__ __forceinline__ float bf16f(uint16_t u) { return __uint_as_float((uint32_t) u << 16); }
__device__ __forceinline__ uint16_t f2bf16(float v) { return __bfloat16_as_ushort(__float2bfloat16_rn(v)); }

__device__ __forceinline__ float fast_tanh(float x) {
    float y;
    asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float softcap_val(float v, float cap) {
    return cap > 0.f ? cap * fast_tanh(v / cap) : v;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m));
    return v;
}

__device__ __forceinline__ float block_sum256(float v, float* red) {
    v = warp_sum(v);
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) red[w] = v;
    __syncthreads();
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) s += red[j];
    __syncthreads();
    return s;
}

// Same 16-warp reduction order as wgemm::norm_pack (512 threads = 16 warps).
__device__ __forceinline__ float block_sum512(float v, float* red) {
    v = warp_sum(v);
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j) s += red[j];
    return s;
}

__device__ __forceinline__ bool better_cand(float a, int ia, float b, int ib) {
    return a > b || (a == b && ia < ib);
}

// Strict total order over candidate slots (breaks identical invalid (-inf, INT_MAX) ties by slot index).
__device__ __forceinline__ bool strict_better(float a, int ia, int ja, float b, int ib, int jb) {
    return a > b || (a == b && (ia < ib || (ia == ib && ja < jb)));
}

// ---- context tap gathering & fc Q8_0 slice split ----------------------------------------------------------------------

// Grid (rows, 36): 256 threads/CTA (36 * 256 == 9216 == 4 * kWidth; 9 CTAs per tap slot).
__global__ void __launch_bounds__(256) pack_tap_bf16_kernel(const float* __restrict__ tap, int64_t tap_stride,
                                                            uint16_t* __restrict__ out) {
    pdl_launch_dependents();
    pdl_wait();
    const int r = (int) blockIdx.x;
    const int by = (int) blockIdx.y;
    const int slot = by / 9;
    const int col = (by - slot * 9) * 256 + (int) threadIdx.x;
    out[(int64_t) r * (kDf2Targets * kWidth) + by * 256 + (int) threadIdx.x] =
        f2bf16(tap[(int64_t) slot * tap_stride + (int64_t) r * kWidth + col]);
}

// Grid (rows, 36): 256 threads = 8 warps, each warp quantizes one 32-float Q8_1 block of [rows][4 * kWidth = 9216].
__global__ void __launch_bounds__(256) pack_tap_q8_1_kernel(const float* __restrict__ tap, int64_t tap_stride,
                                                            Q81* __restrict__ out) {
    pdl_launch_dependents();
    pdl_wait();
    constexpr int kTotalBlocks = (kDf2Targets * kWidth) / 32;   // 288
    const int r = (int) blockIdx.x;
    const int b = (int) blockIdx.y * 8 + (int) (threadIdx.x >> 5);
    if (b >= kTotalBlocks) return;
    const int lane = threadIdx.x & 31;
    const int slot = b / kHeadNb;
    const int sb = b - slot * kHeadNb;
    const float xi = tap[(int64_t) slot * tap_stride + (int64_t) r * kWidth + sb * 32 + lane];
    const float amax = warp_max(fabsf(xi));
    const float sum = warp_sum(xi);
    const float d = q8_1_finite(amax / 127.0f);
    const int8_t q = q8_1_quant(xi, d, amax);
    Q81& dst = out[(size_t) r * kTotalBlocks + b];
    dst.qs[lane] = q;
    if (lane == 0) dst.ds = q8_1_ds(d, sum);
}

// Splits [kWidth][4 * kWidth] Q8_0 (288 blocks/row) into 4 contiguous [kWidth][kWidth] Q8_0 submatrices (72 blocks/row).
__global__ void __launch_bounds__(256) split_fc_q8_kernel(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst) {
    constexpr int kFullRowBlocks = kDf2Targets * kHeadNb;   // 288
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t) kWidth * kFullRowBlocks) return;
    const int row = (int) (idx / kFullRowBlocks);
    const int blk = (int) (idx % kFullRowBlocks);
    const int part = blk / kHeadNb;
    const int pb = blk - part * kHeadNb;
    const uint8_t* s = src + idx * kQ8Block;
    uint8_t* d = dst + (((int64_t) part * kWidth + row) * kHeadNb + pb) * kQ8Block;
#pragma unroll
    for (int i = 0; i < 17; ++i) reinterpret_cast<uint16_t*>(d)[i] = reinterpret_cast<const uint16_t*>(s)[i];
}

// ---- mask row & block inputs ------------------------------------------------------------------------------------------

__global__ void __launch_bounds__(kThreads) make_mask_kernel(const uint16_t* __restrict__ emb, int32_t mask_id,
                                                             const float* __restrict__ norm, float eps,
                                                             float* __restrict__ mask_out) {
    __shared__ float red[8];
    const uint16_t* e = emb + (int64_t) mask_id * kWidth;
    float v[kPer];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        v[i] = bf16f(e[threadIdx.x + i * kThreads]);
        ss += v[i] * v[i];
    }
    const float inv = rsqrtf(block_sum256(ss, red) / (float) kWidth + eps);
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int c = threadIdx.x + i * kThreads;
        mask_out[c] = v[i] * inv * norm[c];
    }
}

__global__ void __launch_bounds__(kThreads) block_inputs_batch_kernel(const uint16_t* __restrict__ emb,
                                                                      const int32_t* __restrict__ meta,
                                                                      const float* __restrict__ norm, float eps,
                                                                      const float* __restrict__ mask, int block,
                                                                      float* __restrict__ x) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float red[8];
    const int t = (int) blockIdx.x, s = (int) blockIdx.y;
    float* row = x + ((int64_t) s * block + t) * kWidth;
    if (t > 0) {
#pragma unroll
        for (int i = 0; i < kPer; ++i) row[threadIdx.x + i * kThreads] = mask[threadIdx.x + i * kThreads];
        return;
    }
    const int32_t anchor = meta[2 * s];
    const uint16_t* e = emb + (int64_t) anchor * kWidth;
    float v[kPer];
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        v[i] = bf16f(e[threadIdx.x + i * kThreads]);
        ss += v[i] * v[i];
    }
    const float inv = rsqrtf(block_sum256(ss, red) / (float) kWidth + eps);
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int c = threadIdx.x + i * kThreads;
        row[c] = v[i] * inv * norm[c];
    }
}

// ---- dynamic 2-tap group-16 causal convolutions ----------------------------------------------------------------------

__global__ void __launch_bounds__(kThreads) conv2_pack_kernel(const float* __restrict__ z, const float* __restrict__ dyn,
                                                              int64_t ld_dyn, const float* __restrict__ base, int block,
                                                              uint16_t* __restrict__ act_bf16, float* __restrict__ y_opt) {
    pdl_launch_dependents();
    pdl_wait();
    const int r = (int) blockIdx.x;
    const int t = r % block;
    const float* dr = dyn + (int64_t) r * ld_dyn;
    const float* zr = z + (int64_t) r * kWidth;
    const float* zp = zr - kWidth;
    uint16_t* ar = act_bf16 + (int64_t) r * kWidth;
    float* yr = y_opt ? y_opt + (int64_t) r * kWidth : nullptr;
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int c = threadIdx.x + i * kThreads;
        const int g = c / kDf2ConvGroup;
        float acc = (base[c] + dr[g]) * zr[c];
        if (t > 0) acc += (base[kWidth + c] + dr[kDf2ConvGroups + g]) * zp[c];
        ar[c] = f2bf16(acc);
        if (yr) yr[c] = acc;
    }
}

__global__ void __launch_bounds__(kThreads) conv2_add_kernel(const float* __restrict__ z, const float* __restrict__ dyn,
                                                             int64_t ld_dyn, const float* __restrict__ base, int block,
                                                             float* __restrict__ x) {
    pdl_launch_dependents();
    pdl_wait();
    const int r = (int) blockIdx.x;
    const int t = r % block;
    const float* dr = dyn + (int64_t) r * ld_dyn + 2 * kDf2ConvGroups;
    const float* b2 = base + 2 * kWidth;
    const float* zr = z + (int64_t) r * kWidth;
    const float* zp = zr - kWidth;
    float* xr = x + (int64_t) r * kWidth;
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
        const int c = threadIdx.x + i * kThreads;
        const int g = c / kDf2ConvGroup;
        float acc = (b2[c] + dr[g]) * zr[c];
        if (t > 0) acc += (b2[kWidth + c] + dr[kDf2ConvGroups + g]) * zp[c];
        xr[c] += acc;
    }
}

// Fused post-conv residual add (x += conv_post(z)) + next sublayer's 16-warp RMSNorm + BF16 pack (z distinct from xn_out).
__global__ void __launch_bounds__(512) conv2_add_norm_pack_kernel(const float* __restrict__ z,
                                                                  const float* __restrict__ dyn, int64_t ld_dyn,
                                                                  const float* __restrict__ base, int block,
                                                                  float* __restrict__ x, const float* __restrict__ norm_w,
                                                                  float eps, uint16_t* __restrict__ act_bf16,
                                                                  float* __restrict__ xn_out) {
    __shared__ float red[16];
    pdl_launch_dependents();
    const int tid = threadIdx.x;
    const float* b2 = base + 2 * kWidth;
    float wv[5], b0v[5], b1v[5];
#pragma unroll
    for (int j = 0; j < 5; ++j) {
        const int c = tid + j * 512;
        wv[j] = c < kWidth ? norm_w[c] : 0.f;
        b0v[j] = c < kWidth ? b2[c] : 0.f;
        b1v[j] = c < kWidth ? b2[kWidth + c] : 0.f;
    }
    pdl_wait();
    const int r = (int) blockIdx.x;
    const int t = r % block;
    const float* dr = dyn + (int64_t) r * ld_dyn + 2 * kDf2ConvGroups;
    const float* zr = z + (int64_t) r * kWidth;
    const float* zp = zr - kWidth;
    float* xr = x + (int64_t) r * kWidth;
    float xv[5];
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < 5; ++j) {
        const int c = tid + j * 512;
        if (c < kWidth) {
            const int g = c / kDf2ConvGroup;
            float acc = (b0v[j] + dr[g]) * zr[c];
            if (t > 0) acc += (b1v[j] + dr[kDf2ConvGroups + g]) * zp[c];
            xv[j] = xr[c] + acc;
            xr[c] = xv[j];
            ss = fmaf(xv[j], xv[j], ss);
        } else {
            xv[j] = 0.f;
        }
    }
    const float inv = rsqrtf(block_sum512(ss, red) / (float) kWidth + eps);
    uint16_t* ar = act_bf16 + (int64_t) r * kWidth;
    float* yr = xn_out ? xn_out + (int64_t) r * kWidth : nullptr;
#pragma unroll
    for (int j = 0; j < 5; ++j) {
        const int c = tid + j * 512;
        if (c < kWidth) {
            const float v = xv[j] * inv * wv[j];
            if (yr) yr[c] = v;
            ar[c] = f2bf16(v);
        }
    }
}

// ---- block QK RMSNorm + NeoX RoPE + ring KV write (32 Q heads, 8 KV heads, dim 128) ---------------------------------

struct Df2RopeFreq {
    float inv[kDf2HeadDim / 2];
};

Df2RopeFreq make_df2_rope(float rope_base) {
    Df2RopeFreq r{};
    for (int i = 0; i < kDf2HeadDim / 2; ++i)
        r.inv[i] = (float) std::pow((double) rope_base, -2.0 * (double) i / (double) kDf2HeadDim);
    return r;
}

__global__ void __launch_bounds__(kDf2HeadDim) block_qk_prep_batch_kernel(
    const float* __restrict__ q, const float* __restrict__ kb, const float* __restrict__ vb,
    const float* __restrict__ q_norm, const float* __restrict__ k_norm, float eps,
    const __grid_constant__ Df2RopeFreq rope, const int32_t* __restrict__ meta, int block, float* __restrict__ q_out,
    const __grid_constant__ DsBatchKv kv, int64_t cap) {
    pdl_launch_dependents();
    pdl_wait();
    __shared__ float red[4];
    __shared__ float y_sh[kDf2HeadDim];
    const int r = (int) blockIdx.x;
    const int s = r / block;
    const int t = r - s * block;
    const int h = (int) blockIdx.y;
    const int d = threadIdx.x;
    const int lane = d & 31, warp = d >> 5;
    const bool is_q = h < kDf2Heads;
    const int head = is_q ? h : h - kDf2Heads;
    const float xv = is_q ? q[((int64_t) r * kDf2Heads + head) * kDf2HeadDim + d]
                          : kb[((int64_t) r * kDf2KvHeads + head) * kDf2HeadDim + d];
    const float ws = warp_sum(xv * xv);
    if (lane == 0) red[warp] = ws;
    __syncthreads();
    const float inv = rsqrtf((red[0] + red[1] + red[2] + red[3]) * (1.f / (float) kDf2HeadDim) + eps);
    float y = xv * inv * (is_q ? q_norm[d] : k_norm[d]);
    y_sh[d] = y;
    __syncthreads();
    const int pair = d & 63;
    const int32_t p = meta[2 * s + 1] + t;
    float sn = 0.f, cs = 1.f;
    __sincosf((float) p * rope.inv[pair], &sn, &cs);
    const float rot = (d < 64) ? (y_sh[pair] * cs - y_sh[pair + 64] * sn) : (y_sh[pair] * sn + y_sh[pair + 64] * cs);
    if (is_q) {
        q_out[((int64_t) r * kDf2Heads + head) * kDf2HeadDim + d] = rot;
        return;
    }
    const int64_t cell = (int64_t) p % cap;
    const int64_t dst_off = (cell * kDf2KvHeads + head) * kDf2HeadDim + d;
    kv.k[s][dst_off] = __half_as_ushort(__float2half_rn(rot));
    kv.v[s][dst_off] = __half_as_ushort(__float2half_rn(vb[((int64_t) r * kDf2KvHeads + head) * kDf2HeadDim + d]));
}

// ---- fused draft-row RMSNorm + BF16 & Q8_1 packing -------------------------------------------------------------------

// Grid (count * n_drafts), 512 threads = 16 warps: skips anchor row 0 of each block, norms row (1 + d) in the exact
// 16-warp reduction order of wgemm::norm_pack, writes BF16 act_bf16 (and optional fp32 hd_opt), and quantizes Q8_1 hq.
__global__ void __launch_bounds__(512) norm_draft_pack_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                                              float eps, int block, int n_drafts,
                                                              uint16_t* __restrict__ act_bf16, Q81* __restrict__ hq,
                                                              float* __restrict__ hd_opt) {
    pdl_launch_dependents();
    __shared__ float red[16];
    __shared__ float row_sh[kWidth];
    const int tid = threadIdx.x;
    float wv[5];
#pragma unroll
    for (int j = 0; j < 5; ++j) {
        const int c = tid + j * 512;
        wv[j] = c < kWidth ? w[c] : 0.f;
    }
    pdl_wait();
    const int out_r = (int) blockIdx.x;
    const int s = out_r / n_drafts;
    const int d = out_r - s * n_drafts;
    const float* xr = x + ((int64_t) s * block + 1 + d) * kWidth;
    float xv[5];
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < 5; ++j) {
        const int c = tid + j * 512;
        xv[j] = c < kWidth ? xr[c] : 0.f;
        if (c < kWidth) ss = fmaf(xv[j], xv[j], ss);
    }
    const float inv = rsqrtf(block_sum512(ss, red) / (float) kWidth + eps);
    uint16_t* ar = act_bf16 + (int64_t) out_r * kWidth;
    float* hr = hd_opt ? hd_opt + (int64_t) out_r * kWidth : nullptr;
#pragma unroll
    for (int j = 0; j < 5; ++j) {
        const int c = tid + j * 512;
        if (c < kWidth) {
            const float v = xv[j] * inv * wv[j];
            ar[c] = f2bf16(v);
            row_sh[c] = v;
            if (hr) hr[c] = v;
        }
    }
    __syncthreads();
    if (hq) {
        const int warp = tid >> 5, lane = tid & 31;
        Q81* qr = hq + (int64_t) out_r * kHeadNb;
#pragma unroll
        for (int j = 0; j < 5; ++j) {
            const int b = warp + j * 16;
            if (b < kHeadNb) {
                const float xi = row_sh[b * 32 + lane];
                const float amax = warp_max(fabsf(xi));
                const float sum = warp_sum(xi);
                const float sc = q8_1_finite(amax / 127.0f);
                qr[b].qs[lane] = q8_1_quant(xi, sc, amax);
                if (lane == 0) qr[b].ds = q8_1_ds(sc, sum);
            }
        }
    }
}

// ---- barrier-minimal exact Q8-rescored Top-16 + full-vocab tail LSE extraction ---------------------------------------

__device__ __forceinline__ int q80_word(const uint8_t* qs, int i32) {
    const auto* x = reinterpret_cast<const uint16_t*>(qs);
    return (int) x[2 * i32] | ((int) x[2 * i32 + 1] << 16);
}

__device__ __forceinline__ float q80_dot(const uint8_t* __restrict__ w, const Q81* x, int nb, int lane) {
    const int k = 2 * (lane & 3);
    float s = 0.f;
    for (int b = lane >> 2; b < nb; b += 8) {
        const uint8_t* blk = w + (size_t) b * kQ8Block;
        const int* xq = reinterpret_cast<const int*>(x[b].qs);
        int sumi = __dp4a(q80_word(blk + 2, k), xq[k], 0);
        sumi = __dp4a(q80_word(blk + 2, k + 1), xq[k + 1], sumi);
        s += __half2float(*reinterpret_cast<const __half*>(blk)) * __low2float(x[b].ds) * (float) sumi;
    }
    return warp_sum(s);
}

__device__ __forceinline__ float q8p_dot(const void* __restrict__ planes, int64_t n_vocab, int64_t v, const Q81* x, int nb,
                                         int lane) {
    const int n_in = nb * 32;
    const int8_t* codes = static_cast<const int8_t*>(planes) + v * n_in;
    const __half* scales = reinterpret_cast<const __half*>(static_cast<const int8_t*>(planes) + n_vocab * n_in) + v * nb;
    const int half_b = lane & 1;
    float s = 0.f;
    for (int b = lane >> 1; b < nb; b += 16) {
        const int4 q = __ldg(reinterpret_cast<const int4*>(codes + b * 32 + half_b * 16));
        const int* xq = reinterpret_cast<const int*>(x[b].qs) + half_b * 4;
        int sumi = __dp4a(q.x, xq[0], 0);
        sumi = __dp4a(q.y, xq[1], sumi);
        sumi = __dp4a(q.z, xq[2], sumi);
        sumi = __dp4a(q.w, xq[3], sumi);
        s += __half2float(__ldg(scales + b)) * __low2float(x[b].ds) * (float) sumi;
    }
    return warp_sum(s);
}

// Stage 1: grid (rows, chunks), 256 threads = 8 warps. Each warp extracts its exact Top-16 via pure warp shuffles (0 CTA
// barriers), then 128 threads rank the 128 warp winners in parallel (1 barrier), while computing chunk softcapped LSE.
__global__ void __launch_bounds__(256) top16_part_kernel(const float* logits, int n_vocab, int64_t ld, float cap,
                                                         float2* part, float2* part_lse) {
    __shared__ float w_val[8 * kDf2TopK];
    __shared__ int w_id[8 * kDf2TopK];
    __shared__ float lse_sh[8];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int r = (int) blockIdx.x, chunk = (int) blockIdx.y;
    const int c0 = chunk * kTopChunk;
    const float* row = logits + (int64_t) r * ld;

    pdl_wait();
    float vals[16];
    uint32_t alive = 0u;
    float tm = -INFINITY;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int col = c0 + j * 256 + tid;
        if (col < n_vocab) {
            const float v = row[col];
            vals[j] = v;
            alive |= (1u << j);
            if (part_lse) tm = fmaxf(tm, softcap_val(v, cap));
        } else {
            vals[j] = -INFINITY;
        }
    }

    for (int k = 0; k < kDf2TopK; ++k) {
        float bv = -INFINITY;
        int bi = 0x7fffffff, bj = -1;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            if (alive & (1u << j)) {
                const int col = c0 + j * 256 + tid;
                if (better_cand(vals[j], col, bv, bi)) {
                    bv = vals[j];
                    bi = col;
                    bj = j;
                }
            }
        }
        int wl = lane, wj = bj;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            const int ol = __shfl_xor_sync(0xffffffffu, wl, o);
            const int oj = __shfl_xor_sync(0xffffffffu, wj, o);
            if (better_cand(ov, oi, bv, bi)) {
                bv = ov;
                bi = oi;
                wl = ol;
                wj = oj;
            }
        }
        if (lane == wl && wj >= 0 && wj < 16) alive &= ~(1u << wj);
        if (lane == 0) {
            w_val[warp * kDf2TopK + k] = bv;
            w_id[warp * kDf2TopK + k] = bi;
        }
    }
    if (part_lse) {
        const float wm = warp_max(tm);
        if (lane == 0) lse_sh[warp] = wm;
    }
    __syncthreads();
    pdl_launch_dependents();

    float2* dst = part + ((int64_t) r * gridDim.y + chunk) * kDf2TopK;
    if (tid < 8 * kDf2TopK) {
        const float mv = w_val[tid];
        const int mi = w_id[tid];
        int rk = 0;
        for (int j = 0; j < 8 * kDf2TopK; ++j)
            if (strict_better(w_val[j], w_id[j], j, mv, mi, tid)) ++rk;
        if (rk < kDf2TopK) dst[rk] = make_float2(mv, __int_as_float(mi));
    }

    if (part_lse) {
        float cm = lse_sh[0];
#pragma unroll
        for (int w = 1; w < 8; ++w) cm = fmaxf(cm, lse_sh[w]);
        float tl = 0.f;
        if (isfinite(cm)) {
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int col = c0 + j * 256 + tid;
                if (col < n_vocab) tl += __expf(softcap_val(vals[j], cap) - cm);
            }
        }
        const float wl = warp_sum(tl);
        __syncthreads();
        if (lane == 0) lse_sh[warp] = wl;
        __syncthreads();
        if (tid == 0) {
            float sl = 0.f;
#pragma unroll
            for (int w = 0; w < 8; ++w) sl += lse_sh[w];
            part_lse[(int64_t) r * gridDim.y + chunk] = make_float2(cm, sl);
        }
    }
}

// Stage 2: grid (rows), 512 threads = 16 warps (2 candidates/thread in count-rank, 16 concurrent DRAM head rows/wave).
__global__ void __launch_bounds__(512) top16_rescore_kernel(const float2* part, const float2* part_lse, int chunks,
                                                            const void* __restrict__ head_q8, bool head_planes,
                                                            const Q81* hq, int n_vocab, float cap, int32_t* cand_out,
                                                            float* tail_lse_out) {
    __shared__ float all_v[64 * kDf2TopK];
    __shared__ int all_id[64 * kDf2TopK];
    __shared__ int cand64_id[kDf2Rescore];
    __shared__ float cand64_v[kDf2Rescore];
    __shared__ float cand64_unary[kDf2Rescore];
    __shared__ float win16_unary[kDf2TopK];
    __shared__ float m_all_sh;
    __shared__ float l_all_sh;
    __shared__ Q81 xh_sh[kHeadNb];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int r = (int) blockIdx.x;
    const int n_cand = chunks * kDf2TopK;
    const int n_rescore = head_q8 ? kDf2Rescore : kDf2TopK;
    const float2* src = part + (int64_t) r * n_cand;

    if (tid < kDf2Rescore) {
        cand64_id[tid] = 0x7fffffff;
        cand64_v[tid] = -INFINITY;
        cand64_unary[tid] = -INFINITY;
    }

    pdl_wait();
    const auto load_slot = [&](int idx, float& vo, int& io) {
        if (idx < n_cand) {
            const float2 p = src[idx];
            const int id = __float_as_int(p.y);
            if (id >= 0 && id < n_vocab) {
                vo = p.x;
                io = id;
            } else {
                vo = -INFINITY;
                io = 0x7fffffff;
            }
            all_v[idx] = vo;
            all_id[idx] = io;
        } else {
            vo = -INFINITY;
            io = 0x7fffffff;
        }
    };

    const int i0 = tid, i1 = tid + 512;
    float v0, v1;
    int id0, id1;
    load_slot(i0, v0, id0);
    load_slot(i1, v1, id1);

    if (head_q8) {
        const Q81* xh_g = hq + (int64_t) r * kHeadNb;
        for (int k = tid; k < kHeadNb * 9; k += 512)
            reinterpret_cast<int*>(xh_sh)[k] = reinterpret_cast<const int*>(xh_g)[k];
    }

    if (tail_lse_out && part_lse && warp == 15) {
        const float2* lr = part_lse + (int64_t) r * chunks;
        const int c0 = lane, c1 = lane + 32;
        const float2 p0 = c0 < chunks ? lr[c0] : make_float2(-INFINITY, 0.f);
        const float2 p1 = c1 < chunks ? lr[c1] : make_float2(-INFINITY, 0.f);
        const float m_all = warp_max(fmaxf(p0.x, p1.x));
        float l_loc = 0.f;
        if (isfinite(m_all)) {
            if (c0 < chunks && isfinite(p0.x)) l_loc += p0.y * __expf(p0.x - m_all);
            if (c1 < chunks && isfinite(p1.x)) l_loc += p1.y * __expf(p1.x - m_all);
        }
        const float l_all = warp_sum(l_loc);
        if (lane == 0) {
            m_all_sh = m_all;
            l_all_sh = l_all;
        }
    }
    __syncthreads();

    int rk0 = 0, rk1 = 0;
    for (int j = 0; j < n_cand; ++j) {
        const float vj = all_v[j];
        const int ij = all_id[j];
        if (i0 < n_cand && strict_better(vj, ij, j, v0, id0, i0)) ++rk0;
        if (i1 < n_cand && strict_better(vj, ij, j, v1, id1, i1)) ++rk1;
    }
    if (i0 < n_cand && rk0 < n_rescore) { cand64_id[rk0] = id0; cand64_v[rk0] = v0; cand64_unary[rk0] = v0; }
    if (i1 < n_cand && rk1 < n_rescore) { cand64_id[rk1] = id1; cand64_v[rk1] = v1; cand64_unary[rk1] = v1; }
    __syncthreads();

    int32_t* out = cand_out + (int64_t) r * 2 * kDf2TopK;
    if (!head_q8) {
        pdl_launch_dependents();
        if (tid < kDf2TopK) {
            out[tid] = cand64_id[tid];
            out[kDf2TopK + tid] = __float_as_int(cand64_v[tid]);
        }
        if (tail_lse_out && warp == 0) {
            const float m_all = m_all_sh;
            const float u = (lane < kDf2TopK && cand64_id[lane] < n_vocab) ? cand64_unary[lane] : -INFINITY;
            const float term = (isfinite(u) && isfinite(m_all)) ? __expf(softcap_val(u, cap) - m_all) : 0.f;
            const float s16 = warp_sum(term);
            if (lane == 0) {
                tail_lse_out[2 * r] = m_all;
                tail_lse_out[2 * r + 1] = fmaxf(0.f, l_all_sh - s16);
            }
        }
        return;
    }

    for (int c = warp; c < kDf2Rescore; c += 16) {
        const int id = cand64_id[c];
        float ex = -INFINITY;
        if (id >= 0 && id < n_vocab) {
            ex = head_planes
                     ? q8p_dot(head_q8, n_vocab, id, xh_sh, kHeadNb, lane)
                     : q80_dot(static_cast<const uint8_t*>(head_q8) + (size_t) id * kQ8Row, xh_sh, kHeadNb, lane);
        }
        if (lane == 0) cand64_v[c] = ex;
    }
    __syncthreads();
    pdl_launch_dependents();

    if (tid < kDf2Rescore) {
        const float mv = cand64_v[tid];
        const int mi = cand64_id[tid] < n_vocab ? cand64_id[tid] : 0x7fffffff;
        int rk = 0;
        for (int j = 0; j < kDf2Rescore; ++j) {
            const int ji = cand64_id[j] < n_vocab ? cand64_id[j] : 0x7fffffff;
            if (strict_better(cand64_v[j], ji, j, mv, mi, tid)) ++rk;
        }
        if (rk < kDf2TopK) {
            out[rk] = mi;
            out[kDf2TopK + rk] = __float_as_int(mv);
            win16_unary[rk] = mi < n_vocab ? cand64_unary[tid] : -INFINITY;
        }
    }
    if (tail_lse_out) {
        __syncthreads();
        if (warp == 0) {
            const float m_all = m_all_sh;
            const float u = lane < kDf2TopK ? win16_unary[lane] : -INFINITY;
            const float term = (isfinite(u) && isfinite(m_all)) ? __expf(softcap_val(u, cap) - m_all) : 0.f;
            const float s16 = warp_sum(term);
            if (lane == 0) {
                tail_lse_out[2 * r] = m_all;
                tail_lse_out[2 * r + 1] = fmaxf(0.f, l_all_sh - s16);
            }
        }
    }
}

// ---- 16-warp 128-bit vectorized bilinear candidate selector with full-vocabulary calibrated confidence ----------------

__global__ void __launch_bounds__(512) select_kernel(const int32_t* meta, int meta_stride, const int32_t* cand,
                                                     const float* tail_lse, const float* g,
                                                     const uint16_t* __restrict__ pred,
                                                     const uint16_t* __restrict__ succ, int n_drafts, float cap,
                                                     int32_t* ids_out, int ids_stride, float* conf_out, int conf_stride) {
    __shared__ __align__(16) float pg[kDf2Rank];
    __shared__ float sc_sh[kDf2TopK];
    __shared__ int32_t prev_s;

    const int s = (int) blockIdx.x;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    pdl_wait();

    for (int d = 0; d < n_drafts; ++d) {
        const int r = s * n_drafts + d;
        const int32_t prev = d == 0 ? meta[s * meta_stride] : prev_s;
        if (tid < kDf2Rank)
            pg[tid] = prev >= 0 ? bf16f(__ldg(pred + (int64_t) prev * kDf2Rank + tid)) * g[(int64_t) r * kDf2Rank + tid]
                                : 0.f;
        __syncthreads();

        const int32_t id = cand[r * 2 * kDf2TopK + warp];
        float dot = 0.f;
        if (id >= 0) {
            const uint4 sv = __ldg(reinterpret_cast<const uint4*>(succ + (int64_t) id * kDf2Rank) + lane);
            const float4 p0 = reinterpret_cast<const float4*>(pg)[2 * lane];
            const float4 p1 = reinterpret_cast<const float4*>(pg)[2 * lane + 1];
            dot += p0.x * bf16f((uint16_t) (sv.x & 0xffffu));
            dot += p0.y * bf16f((uint16_t) (sv.x >> 16));
            dot += p0.z * bf16f((uint16_t) (sv.y & 0xffffu));
            dot += p0.w * bf16f((uint16_t) (sv.y >> 16));
            dot += p1.x * bf16f((uint16_t) (sv.z & 0xffffu));
            dot += p1.y * bf16f((uint16_t) (sv.z >> 16));
            dot += p1.z * bf16f((uint16_t) (sv.w & 0xffffu));
            dot += p1.w * bf16f((uint16_t) (sv.w >> 16));
            dot = warp_sum(dot);
        }
        if (lane == 0) sc_sh[warp] = __int_as_float(cand[r * 2 * kDf2TopK + kDf2TopK + warp]) + dot;
        __syncthreads();

        if (warp == 0) {
            float bv = lane < kDf2TopK ? sc_sh[lane] : -INFINITY;
            int32_t bi = lane < kDf2TopK ? cand[r * 2 * kDf2TopK + lane] : 0x7fffffff;
            const float my_sc = bv;
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                const int32_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (better_cand(ov, oi, bv, bi)) {
                    bv = ov;
                    bi = oi;
                }
            }
            const float top = softcap_val(bv, cap);
            float m_ref = top, tail_term = 0.f;
            if (tail_lse) {
                const float m_t = tail_lse[2 * r];
                const float l_t = tail_lse[2 * r + 1];
                if (isfinite(m_t) && l_t > 0.f) {
                    m_ref = fmaxf(top, m_t);
                    tail_term = l_t * __expf(m_t - m_ref);
                }
            }
            const float my_cap = lane < kDf2TopK ? softcap_val(my_sc, cap) : -INFINITY;
            const float z16 = warp_sum(lane < kDf2TopK ? __expf(my_cap - m_ref) : 0.f);
            if (lane == 0) {
                ids_out[s * ids_stride + d] = bi;
                conf_out[s * conf_stride + d] = __expf(top - m_ref) / (z16 + tail_term);
                prev_s = bi;
            }
        }
        if (d + 1 < n_drafts) __syncthreads();
    }
    pdl_launch_dependents();
}

}  // namespace

void dflash2_pack_tap_bf16(const float* tap, int64_t tap_stride, int rows, void* act_bf16, void* stream) {
    if (rows <= 0) return;
    launch_pdl(pack_tap_bf16_kernel, dim3((unsigned) rows, 36u), dim3(256), 0, (cudaStream_t) stream, tap, tap_stride,
               static_cast<uint16_t*>(act_bf16));
    if (!stream) cudaDeviceSynchronize();
    check("pack_tap_bf16");
}

void dflash2_pack_tap_q8_1(const float* tap, int64_t tap_stride, int rows, void* act_q8_1, void* stream) {
    if (rows <= 0) return;
    launch_pdl(pack_tap_q8_1_kernel, dim3((unsigned) rows, 36u), dim3(256), 0, (cudaStream_t) stream, tap, tap_stride,
               static_cast<Q81*>(act_q8_1));
    if (!stream) cudaDeviceSynchronize();
    check("pack_tap_q8_1");
}

void dflash2_split_fc_q8(const void* fc_q8_full, void* fc_q8_parts, void* stream) {
    constexpr int64_t total = (int64_t) kWidth * kDf2Targets * kHeadNb;
    split_fc_q8_kernel<<<(unsigned) ((total + 255) / 256), 256, 0, (cudaStream_t) stream>>>(
        static_cast<const uint8_t*>(fc_q8_full), static_cast<uint8_t*>(fc_q8_parts));
    if (!stream) cudaDeviceSynchronize();
    check("split_fc_q8");
}

void dflash2_make_mask(const uint16_t* embedding, int32_t mask_id, const float* embed_norm, float eps, float* mask_out,
                       void* stream) {
    make_mask_kernel<<<1, kThreads, 0, (cudaStream_t) stream>>>(embedding, mask_id, embed_norm, eps, mask_out);
    if (!stream) cudaDeviceSynchronize();
    check("make_mask");
}

void dflash2_block_inputs_batch(const uint16_t* embedding, const int32_t* meta, const float* embed_norm, float eps,
                                const float* mask, int block, int count, float* x, void* stream) {
    if (count <= 0 || block <= 0) return;
    launch_pdl(block_inputs_batch_kernel, dim3((unsigned) block, (unsigned) count), dim3(kThreads), 0, (cudaStream_t) stream,
               embedding, meta, embed_norm, eps, mask, block, x);
    if (!stream) cudaDeviceSynchronize();
    check("block_inputs_batch");
}

void dflash2_conv2_pack(const float* z, const float* dyn, int64_t ld_dyn, const float* base, int block, int count,
                        void* act_bf16, float* y_opt, void* stream) {
    const int rows = count * block;
    if (rows <= 0) return;
    launch_pdl(conv2_pack_kernel, dim3((unsigned) rows), dim3(kThreads), 0, (cudaStream_t) stream, z, dyn, ld_dyn, base, block,
               static_cast<uint16_t*>(act_bf16), y_opt);
    if (!stream) cudaDeviceSynchronize();
    check("conv2_pack");
}

void dflash2_conv2_add(const float* z, const float* dyn, int64_t ld_dyn, const float* base, int block, int count,
                       float* x, void* stream) {
    const int rows = count * block;
    if (rows <= 0) return;
    launch_pdl(conv2_add_kernel, dim3((unsigned) rows), dim3(kThreads), 0, (cudaStream_t) stream, z, dyn, ld_dyn, base, block, x);
    if (!stream) cudaDeviceSynchronize();
    check("conv2_add");
}

void dflash2_conv2_add_norm_pack(const float* z, const float* dyn, int64_t ld_dyn, const float* base, int block,
                                 int count, float* x, const float* norm_w, float eps, void* act_bf16, float* xn_out,
                                 void* stream) {
    const int rows = count * block;
    if (rows <= 0) return;
    launch_pdl(conv2_add_norm_pack_kernel, dim3((unsigned) rows), dim3(512), 0, (cudaStream_t) stream, z, dyn, ld_dyn,
               base, block, x, norm_w, eps, static_cast<uint16_t*>(act_bf16), xn_out);
    if (!stream) cudaDeviceSynchronize();
    check("conv2_add_norm_pack");
}

void dflash2_block_qk_prep_batch(const float* q, const float* kb, const float* vb, const float* q_norm, const float* k_norm,
                                 float eps, float rope_base, const int32_t* meta, int block, int count, float* q_out,
                                 const DsBatchKv& kv, int64_t cap, void* stream) {
    const int rows = count * block;
    if (rows <= 0) return;
    launch_pdl(block_qk_prep_batch_kernel, dim3((unsigned) rows, (unsigned) (kDf2Heads + kDf2KvHeads)), dim3(kDf2HeadDim), 0,
               (cudaStream_t) stream, q, kb, vb, q_norm, k_norm, eps, make_df2_rope(rope_base), meta, block, q_out, kv, cap);
    if (!stream) cudaDeviceSynchronize();
    check("block_qk_prep_batch");
}

void dflash2_norm_draft_pack(const float* x, const float* norm_w, float eps, int block, int count, int n_drafts,
                             void* act_bf16, void* hq_q8_1, float* hd_opt, void* stream) {
    const int rows = count * n_drafts;
    if (rows <= 0) return;
    launch_pdl(norm_draft_pack_kernel, dim3((unsigned) rows), dim3(512), 0, (cudaStream_t) stream, x, norm_w, eps, block,
               n_drafts, static_cast<uint16_t*>(act_bf16), static_cast<Q81*>(hq_q8_1), hd_opt);
    if (!stream) cudaDeviceSynchronize();
    check("norm_draft_pack");
}

size_t dflash2_top16_scratch_bytes(int max_rows, int n_vocab) {
    const int chunks = (n_vocab + kTopChunk - 1) / kTopChunk;
    return (size_t) max_rows * chunks * (kDf2TopK + 1) * sizeof(float2);
}

void dflash2_top16(const float* logits, const void* head_q8, bool head_planes, const void* hq, int rows, int n_vocab,
                   int64_t ld, float cap, void* scratch, int32_t* cand_out, float* tail_lse_out, void* stream) {
    if (rows <= 0) return;
    const int chunks = (n_vocab + kTopChunk - 1) / kTopChunk;
    if (chunks > 64) throw std::runtime_error("inkling2 dflash2 top16: vocabulary exceeds 64 * 4096");
    auto* part = static_cast<float2*>(scratch);
    float2* part_lse = tail_lse_out ? part + (size_t) rows * chunks * kDf2TopK : nullptr;
    launch_pdl(top16_part_kernel, dim3((unsigned) rows, (unsigned) chunks), dim3(256), 0, (cudaStream_t) stream, logits,
               n_vocab, ld, cap, part, part_lse);
    launch_pdl(top16_rescore_kernel, dim3((unsigned) rows), dim3(512), 0, (cudaStream_t) stream, part, part_lse, chunks,
               head_q8, head_planes, static_cast<const Q81*>(hq), n_vocab, cap, cand_out, tail_lse_out);
    if (!stream) cudaDeviceSynchronize();
    check("top16");
}

void dflash2_select(const int32_t* meta, int meta_stride, const int32_t* cand, const float* tail_lse, const float* g,
                    const uint16_t* pred, const uint16_t* succ, int count, int n_drafts, float cap, int32_t* ids_out,
                    int ids_stride, float* conf_out, int conf_stride, void* stream) {
    if (count <= 0 || n_drafts <= 0) return;
    launch_pdl(select_kernel, dim3((unsigned) count), dim3(512), 0, (cudaStream_t) stream, meta, meta_stride, cand,
               tail_lse, g, pred, succ, n_drafts, cap, ids_out, ids_stride, conf_out, conf_stride);
    if (!stream) cudaDeviceSynchronize();
    check("select");
}

}  // namespace basalt::models::inkling2
