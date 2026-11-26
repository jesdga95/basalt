// src/models/hy4flash/kernels/ops.cu - the sigmoid gate, small products, the stream fill and the Engram apply of
// Hy4-flash.
#include "basalt/kernels/dp4a.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/models/hy4flash/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>
#include <string>

namespace basalt::models::hy4flash {
namespace {
namespace bk = basalt::kernels;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
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

__device__ __forceinline__ float sigmoidf(float v) { return 1.f / (1.f + expf(-v)); }

__global__ void mul_sigmoid_kernel(const float* __restrict__ g, float* __restrict__ x, int n) {
    bk::pdl_launch_dependents();
    bk::pdl_wait();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= sigmoidf(g[i]);
}

// A warp per row.
__global__ void rows_gemv_kernel(const float* __restrict__ x, const float* __restrict__ gate, int n_expert, int n_embd,
                                 float scale, float* __restrict__ logits) {
    const int e = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31, t = blockIdx.y;
    if (e >= n_expert) return;
    const float4* w = reinterpret_cast<const float4*>(gate + (size_t) e * n_embd);
    const float4* xv = reinterpret_cast<const float4*>(x + (size_t) t * n_embd);
    float acc = 0.f;
    for (int i = lane; i < n_embd / 4; i += 32) {
        const float4 a = xv[i], b = w[i];
        acc = fmaf(a.x, b.x, acc); acc = fmaf(a.y, b.y, acc); acc = fmaf(a.z, b.z, acc); acc = fmaf(a.w, b.w, acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) logits[(size_t) t * n_expert + e] = acc * scale;
}

__global__ void wsum_kernel(const float* const* __restrict__ rows, const float* __restrict__ w, int k, int n, float* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= n) return;
    float acc = 0.f;
    for (int e = 0; e < k; ++e) acc += w[t * k + e] * rows[t * k + e][i];
    out[(size_t) t * n + i] = acc;
}

struct PutWords {
    int w[kPutWords];
};

__global__ void put_words_kernel(const PutWords s, int* dst, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = s.w[i];
}

__global__ void wait_word_mark_kernel(const volatile uint32_t* word, uint32_t value, volatile uint32_t* mark) {
    while (*word != value) __nanosleep(100);
    __threadfence_system();
    *mark = value;
}

constexpr int kReadAheadBlocks = 32;              // blocks that request the ranges
constexpr uint64_t kReadAheadStep = 128 << 10;    // bytes a block requests before it looks at the mark again

// Block b requests steps b, b + blocks, ... of the ranges laid end to end, a step's pieces as bulk L2 prefetches; it then
// reads the step's last 16 bytes (they land once the step is in L2), so each block keeps one step in flight.
__device__ void l2_read_ahead_block(const L2Ranges& r, uint32_t* mark, uint32_t value, int b, int blocks) {
    uint64_t total = 0;
    for (int i = 0; i < kL2Ranges; ++i) total += r.n[i];
    uint32_t acc = 0;
    for (uint64_t o0 = (uint64_t) b * kReadAheadStep; o0 < total && *(volatile uint32_t*) mark != value;
         o0 += (uint64_t) blocks * kReadAheadStep) {
        const uint64_t o1 = min(o0 + kReadAheadStep, total);
        const char* last = nullptr;
        for (uint64_t o = o0; o < o1;) {
            int i = 0;
            uint64_t ro = o;
            while (i < kL2Ranges - 1 && ro >= r.n[i]) {
                ro -= r.n[i];
                ++i;
            }
            const uint64_t take = min(min(o1 - o, r.n[i] - ro), (uint64_t) 32768);
            const char* src = (const char*) r.p[i] + ro;
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"((unsigned) take) : "memory");
            last = src + take - 16;
            o += take;
        }
        const uint4 w = __ldcg((const uint4*) last);
        acc ^= w.x ^ w.y ^ w.z ^ w.w;
    }
    if (acc == 0x9e3779b9u && value == 0x9e3779b9u) mark[1] = acc;   // keeps the reads (never in practice)
}

__global__ void __launch_bounds__(32) l2_read_ahead_kernel(const L2Ranges r, uint32_t* mark, uint32_t value) {
    if (threadIdx.x == 0) l2_read_ahead_block(r, mark, value, (int) blockIdx.x, (int) gridDim.x);
}

template <bool TAILS>
__global__ void swiglu_split_kernel(const float* __restrict__ gu, float* __restrict__ h, int64_t rows, int n_ff, float limit,
                                    ExpertTails t) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * n_ff) return;
    const int64_t r = i / n_ff, k = i - r * n_ff;
    float g = gu[r * 2 * n_ff + k], u = gu[r * 2 * n_ff + n_ff + k], sd = 1.f;
    if constexpr (TAILS) {
        int q = 0;   // the row's expert: the last q with bounds[q] <= r
        while (q + 1 < t.n && t.bounds[q + 1] <= r) ++q;
        const float* s = (const float*) (t.blobs + (size_t) q * t.stride + t.tail_off);
        g *= s[0];
        u *= s[1];
        sd = s[2];
    }
    g = fminf(g, limit);
    u = fminf(fmaxf(u, -limit), limit);
    if constexpr (TAILS) h[i] = g / (1.f + expf(-g)) * u * sd;
    else h[i] = g / (1.f + expf(-g)) * u;
}

__global__ void wsum_tokens_kernel(const float* __restrict__ rows, const int32_t* __restrict__ map, const float* __restrict__ w, int K, int n,
                                   float* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= n) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += w[t * K + k] * rows[(size_t) map[t * K + k] * n + i];
    out[(size_t) t * n + i] = acc;
}

__global__ void fill_kernel(const float* __restrict__ e, int n, float* __restrict__ x) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= n) return;
    const float v = e[(size_t) t * n + i];
    for (int s = 0; s < 4; ++s) x[((size_t) t * 4 + s) * n + i] = v;
}

__device__ __forceinline__ float e4m3_code_to_f32(uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    if (e == 15 && m == 7) return __int_as_float(0x7fc00000);
    const float v = e ? ldexpf(1.f + (float) m * 0.125f, e - 7) : ldexpf((float) m, -9);
    return (b & 0x80) ? -v : v;
}

constexpr int kEngramNormWarps = 4;
constexpr int kEngramNormMaxW = 128;

// One warp per (token, col) row of W <= 128 F8_E4M3 codes: coalesced byte loads and float stores, exact sequential
// sum-of-squares on lane 0 (bit-for-bit identical to the host loop).
__global__ void __launch_bounds__(kEngramNormWarps * 32)
engram_norm_e4m3_kernel(const uint8_t* __restrict__ raw, const float* __restrict__ norm_w,
                        const float* __restrict__ valid_rows, int n_tok, int n_cols, int W, float* __restrict__ out) {
    __shared__ float lut[256];
    __shared__ float sbuf[kEngramNormWarps][kEngramNormMaxW];
    for (int i = threadIdx.x; i < 256; i += kEngramNormWarps * 32) lut[i] = e4m3_code_to_f32((uint8_t) i);
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int pair = (int) blockIdx.x * kEngramNormWarps + warp;
    if (pair >= n_tok * n_cols) return;
    const int t = pair / n_cols, c = pair - t * n_cols;
    const uint8_t* src = raw + (size_t) pair * W;
    for (int d = lane; d < W; d += 32) sbuf[warp][d] = lut[src[d]];
    __syncwarp();
    float r = 0.f;
    if (lane == 0) {
        float ss = 0.f;
        for (int d = 0; d < W; ++d) {
            const float v = sbuf[warp][d];
            ss = __fadd_rn(ss, __fmul_rn(v, v));
        }
        r = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, (float) W), 1.1920929e-07f)));
    }
    r = __shfl_sync(0xffffffffu, r, 0);
    const float valid = valid_rows[t];
    const float* nr = norm_w + (size_t) c * W;
    float* dst = out + (size_t) pair * W;
    for (int d = lane; d < W; d += 32) dst[d] = __fmul_rn(__fmul_rn(__fmul_rn(sbuf[warp][d], r), nr[d]), valid);
}

constexpr int kEngramThreads = 256;

// A block per stream.
__global__ void engram_kernel(const float* __restrict__ kv, const float* __restrict__ kv_b, const float* __restrict__ gate_w,
                              const float* __restrict__ pre_gamma, const float* __restrict__ hc_scale,
                              const float* __restrict__ valid_rows, EngramArgs a, float* __restrict__ x) {
    __shared__ float red[kEngramThreads / 32];
    const int i = blockIdx.x, t = blockIdx.y, n = a.n_embd;
    const float valid = valid_rows[t];
    float* xi = x + ((size_t) t * 4 + i) * n;
    const float* kvt = kv + (size_t) t * 5 * n;
    const float* key = kvt + (size_t) i * n;
    const float* kb = kv_b + (size_t) i * n;
    const float* val = kvt + (size_t) 4 * n;
    const float* vb = kv_b + (size_t) 4 * n;
    float ssx = 0.f, ssk = 0.f, ssv = 0.f;
    for (int j = threadIdx.x; j < n; j += kEngramThreads) {
        const float k = key[j] + kb[j], v = val[j] + vb[j];
        ssx = fmaf(xi[j], xi[j], ssx);
        ssk = fmaf(k, k, ssk);
        ssv = fmaf(v, v, ssv);
    }
    ssx = block_sum<kEngramThreads>(ssx, red);
    ssk = block_sum<kEngramThreads>(ssk, red);
    ssv = block_sum<kEngramThreads>(ssv, red);
    const float rx = rsqrtf(ssx / (float) n + a.eps), rk = rsqrtf(ssk / (float) n + a.eps), rv = rsqrtf(ssv / (float) n + a.eps);
    float dot = 0.f;
    for (int j = threadIdx.x; j < n; j += kEngramThreads)
        dot = fmaf((xi[j] * rx) * ((key[j] + kb[j]) * rk), gate_w[(size_t) i * n + j], dot);
    dot = block_sum<kEngramThreads>(dot, red) * rsqrtf((float) n);
    const float mag = sqrtf(fmaxf(fabsf(dot), 1e-6f));
    const float gate = sigmoidf(dot < 0.f ? -mag : mag);
    for (int j = threadIdx.x; j < n; j += kEngramThreads) {
        const float signal = ((val[j] + vb[j]) * rv) * pre_gamma[j];
        xi[j] += signal * (gate + hc_scale[(size_t) i * n + j]) * valid;
    }
}

}  // namespace

void mul_sigmoid(const float* g, float* x, int n, void* stream) {
    bk::launch_pdl(mul_sigmoid_kernel, dim3((unsigned) ((n + 255) / 256)), dim3(256), 0, (cudaStream_t) stream, g, x, n);
    if (!stream) cudaStreamSynchronize(nullptr);
}

void f32_gemm_rows(const float* x, const float* w, int rows, int cols, float scale, int n_tok, float* y, void* stream) {
    if (cols % 4) throw std::invalid_argument("f32_gemm_rows: cols");
    rows_gemv_kernel<<<dim3((rows + 7) / 8, n_tok), 256, 0, (cudaStream_t) stream>>>(x, w, rows, cols, scale, y);
    if (!stream) cudaStreamSynchronize(nullptr);
}

__global__ void add_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, int n) {
    bk::pdl_launch_dependents();
    bk::pdl_wait();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

// row blockIdx.x by blocks of 512, blockIdx.y the slice of the row's float4 groups this block writes: every block takes
// f = a + b and its RMS over the whole row exactly as rms_norm_rows takes it (same threads, same sums, so every block has the
// same bits), then hc_post's four streams of its slice from f * inv * w, each value computed as the three kernels compute it,
// so the result is theirs bit for bit
template <int E>
__global__ void __launch_bounds__(512) add_norm_hc_post_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                                               const float* __restrict__ w, float eps, const float* __restrict__ x,
                                                               const float* __restrict__ coef, int coef_stride, float* __restrict__ out) {
    bk::pdl_launch_dependents();
    bk::pdl_wait();
    static_assert(E % 512 == 0, "each of the 512 threads takes whole values");
    constexpr int kPer = E / 512, n4 = E / 4;
    __shared__ float red[16];
    const int t = blockIdx.x;
    const float* ar = a + (size_t) t * E;
    const float* br = b ? b + (size_t) t * E : nullptr;
    float f[kPer];   // all of a thread's values in flight at once, summed in rms_norm_rows' order below
#pragma unroll
    for (int j = 0; j < kPer; ++j) f[j] = br ? ar[threadIdx.x + 512 * j] + br[threadIdx.x + 512 * j] : ar[threadIdx.x + 512 * j];
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < kPer; ++j) ss = fmaf(f[j], f[j], ss);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);   // block_sum of rms_norm_rows
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) ss += red[i];
    const float inv = rsqrtf(ss / (float) E + eps);
    const float* cf = coef + (size_t) t * coef_stride;
    const float* xr = x + (size_t) t * 4 * E;
    float* orow = out + (size_t) t * 4 * E;
    const int chunk = (n4 + (int) gridDim.y - 1) / (int) gridDim.y, i1 = min(n4, ((int) blockIdx.y + 1) * chunk);
    for (int i = (int) blockIdx.y * chunk + threadIdx.x; i < i1; i += 512) {
        const int c = 4 * i;
        float4 f4 = make_float4(ar[c], ar[c + 1], ar[c + 2], ar[c + 3]);
        if (br) f4 = make_float4(f4.x + br[c], f4.y + br[c + 1], f4.z + br[c + 2], f4.w + br[c + 3]);
        const float4 fv = make_float4(f4.x * inv * w[c], f4.y * inv * w[c + 1], f4.z * inv * w[c + 2], f4.w * inv * w[c + 3]);
        float4 xs[4];
#pragma unroll
        for (int s = 0; s < 4; ++s) xs[s] = reinterpret_cast<const float4*>(xr)[(size_t) s * n4 + i];
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            const float post = cf[4 + d];
            float4 o = make_float4(fv.x * post, fv.y * post, fv.z * post, fv.w * post);
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const float cc = cf[8 + s * 4 + d];
                o.x = fmaf(xs[s].x, cc, o.x); o.y = fmaf(xs[s].y, cc, o.y);
                o.z = fmaf(xs[s].z, cc, o.z); o.w = fmaf(xs[s].w, cc, o.w);
            }
            reinterpret_cast<float4*>(orow)[(size_t) d * n4 + i] = o;
        }
    }
}

void add_norm_hc_post(const float* a, const float* b, const float* w, float eps, const float* x, const float* coef, int coef_stride,
                      int n_tokens, int E, float* out, void* stream) {
    if (E != 3072) throw std::invalid_argument("add_norm_hc_post: the model's width is 3072");
    // a decode window's few rows spread over 4 blocks each (one round of loads a thread, on 4x the SMs); a prompt chunk's rows
    // fill the card already, so a block takes a whole row and the norm is read once a row
    const unsigned slices = n_tokens <= 8 ? 4 : 1;
    bk::launch_pdl(add_norm_hc_post_kernel<3072>, dim3((unsigned) n_tokens, slices), dim3(512), 0, (cudaStream_t) stream,
                   a, b, w, eps, x, coef, coef_stride, out);
}

// a sliding-window layer's row spans from the window's first position, read on the card (so a captured graph replays them for any
// position): firsts, counts, fill and positions, `arows` apart
__global__ void swa_meta_kernel(const int32_t* __restrict__ pos0, int R, int window, int arows, int32_t* __restrict__ meta) {
    const int t = threadIdx.x;
    if (t >= R) return;
    const int pos = *pos0 + t, first = pos + 1 - window > 0 ? pos + 1 - window : 0;
    meta[t] = first;
    meta[arows + t] = pos + 1 - first;
    meta[2 * arows + t] = 0;
    meta[3 * arows + t] = pos;
}

void swa_meta(const int32_t* pos0, int R, int window, int arows, int32_t* meta, void* stream) {
    swa_meta_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(pos0, R, window, arows, meta);
}

void add_vectors(const float* a, const float* b, float* out, int n, void* stream) {
    bk::launch_pdl(add_kernel, dim3((unsigned) ((n + 255) / 256)), dim3(256), 0, (cudaStream_t) stream, a, b, out, n);
    if (!stream) cudaStreamSynchronize(nullptr);
}

void weighted_sum_window(const float* const* rows, const float* w, int n_tok, int K, int n, float* out, void* stream) {
    wsum_kernel<<<dim3((n + 255) / 256, n_tok), 256, 0, (cudaStream_t) stream>>>(rows, w, K, n, out);
    if (!stream) cudaStreamSynchronize(nullptr);
}

void put_words(const void* src, void* dst, int n_words, void* stream) {
    if (n_words < 1 || n_words > kPutWords) throw std::invalid_argument("put_words: n_words");
    PutWords s;
    std::memcpy(s.w, src, (size_t) n_words * 4);
    put_words_kernel<<<1, 256, 0, (cudaStream_t) stream>>>(s, (int*) dst, n_words);
}

void wait_word_mark(const uint32_t* word, uint32_t value, uint32_t* mark, void* stream) {
    wait_word_mark_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(word, value, mark);
}

void l2_read_ahead(const L2Ranges& r, uint32_t* mark, uint32_t value, void* stream) {
    if (r.n[0] == 0) return;
    l2_read_ahead_kernel<<<kReadAheadBlocks, 32, 0, (cudaStream_t) stream>>>(r, mark, value);
}

void swiglu_split_clamped(const float* gu, float* h, int64_t rows, int n_ff, float limit, void* stream, const ExpertTails* tails) {
    if (rows <= 0) return;
    const unsigned blocks = (unsigned) ((rows * n_ff + 255) / 256);
    if (tails && tails->n > 0) swiglu_split_kernel<true><<<blocks, 256, 0, (cudaStream_t) stream>>>(gu, h, rows, n_ff, limit, *tails);
    else swiglu_split_kernel<false><<<blocks, 256, 0, (cudaStream_t) stream>>>(gu, h, rows, n_ff, limit, ExpertTails{});
    if (!stream) cudaStreamSynchronize(nullptr);
}

void weighted_sum_tokens(const float* rows, const int32_t* map, const float* w, int n_tok, int K, int n, float* out, void* stream) {
    wsum_tokens_kernel<<<dim3((n + 255) / 256, n_tok), 256, 0, (cudaStream_t) stream>>>(rows, map, w, K, n, out);
    if (!stream) cudaStreamSynchronize(nullptr);
}

void fill_streams(const float* e, int n, int n_tok, float* x, void* stream) {
    fill_kernel<<<dim3((n + 255) / 256, n_tok), 256, 0, (cudaStream_t) stream>>>(e, n, x);
    if (!stream) cudaStreamSynchronize(nullptr);
}

void engram_norm_e4m3(const uint8_t* raw, const float* norm_w, const float* valid, int n_tok, int n_cols, int W,
                      float* out, void* stream) {
    if (W < 1 || W > kEngramNormMaxW) throw std::invalid_argument("engram_norm_e4m3: W");
    const int pairs = n_tok * n_cols;
    if (pairs <= 0) return;
    const int blocks = (pairs + kEngramNormWarps - 1) / kEngramNormWarps;
    engram_norm_e4m3_kernel<<<blocks, kEngramNormWarps * 32, 0, (cudaStream_t) stream>>>(raw, norm_w, valid, n_tok, n_cols, W, out);
    if (!stream) cudaStreamSynchronize(nullptr);
}

void engram_apply(const float* kv, const float* kv_b, const float* gate_w, const float* pre_gamma, const float* hc_scale,
                  const float* valid, int n_tok, const EngramArgs& a, float* x, void* stream) {
    engram_kernel<<<dim3(4, n_tok), kEngramThreads, 0, (cudaStream_t) stream>>>(kv, kv_b, gate_w, pre_gamma, hc_scale, valid, a, x);
    if (!stream) cudaStreamSynchronize(nullptr);
}

namespace {
struct Q80B {
    half d;
    int8_t qs[32];
};
struct Q81B {
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q80B) == 34 && sizeof(Q81B) == 36);

struct Q8xArgs {
    const Q80B* w[kQ8xMats];
    const Q81B* x[kQ8xMats];
    float* y[kQ8xMats];
    int n_out[kQ8xMats];
    int end[kQ8xMats];
    int n;
};

__device__ __forceinline__ int q8x_word_b2(const int8_t* qs, int i) {
    const auto* p = reinterpret_cast<const uint16_t*>(qs);
    int v = p[2 * i];
    v |= p[2 * i + 1] << 16;
    return v;
}

template <int NC, int R, int IT>
__global__ void __launch_bounds__(128) hy4_q8x_kernel(Q8xArgs a) {
    int m = 0;
    while (m < a.n - 1 && (int) blockIdx.x >= a.end[m]) ++m;
    const int bx = (int) blockIdx.x - (m ? a.end[m - 1] : 0);
    const Q80B* __restrict__ w = a.w[m];
    const int n_out = a.n_out[m], row0 = bx * R;
    const int tid = 32 * (int) threadIdx.y + (int) threadIdx.x;
    const int kbx0 = tid >> 2, kqs = 2 * (tid & 3);
    constexpr int bpr = IT * 32;
    int v0[R][IT], v1[R][IT];
    float d0[R][IT];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = min(row0 + r, n_out - 1);
#pragma unroll
        for (int it = 0; it < IT; ++it) {
            const Q80B* b = w + (size_t) row * bpr + kbx0 + 32 * it;
            v0[r][it] = q8x_word_b2(b->qs, kqs);
            v1[r][it] = q8x_word_b2(b->qs, kqs + 1);
            d0[r][it] = __half2float(b->d);
        }
    }
    bk::pdl_launch_dependents();
    bk::pdl_wait();
    const Q81B* __restrict__ x = a.x[m];
    float tmp[NC][R];
#pragma unroll
    for (int j = 0; j < NC; ++j)
#pragma unroll
        for (int r = 0; r < R; ++r) tmp[j][r] = 0.0f;
#pragma unroll
    for (int it = 0; it < IT; ++it) {
        const int kbx = kbx0 + 32 * it;
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            const Q81B* xj = x + (size_t) j * bpr + kbx;
            const int* u = reinterpret_cast<const int*>(xj->qs);
            const int u0 = u[kqs], u1 = u[kqs + 1];
            const float d1 = __low2float(xj->ds);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                int sumi = 0;
                sumi = DP4A(v0[r][it], u0, sumi);
                sumi = DP4A(v1[r][it], u1, sumi);
                tmp[j][r] += d0[r][it] * d1 * float(sumi);
            }
        }
    }
    __shared__ float part[3][NC * R][32];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < NC; ++j)
#pragma unroll
            for (int r = 0; r < R; ++r) part[threadIdx.y - 1][j * R + r][threadIdx.x] = tmp[j][r];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
    float* __restrict__ y = a.y[m];
#pragma unroll
    for (int j = 0; j < NC; ++j)
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float v = tmp[j][r];
#pragma unroll
            for (int l = 0; l < 3; ++l) v += part[l][j * R + r][threadIdx.x];
            v = warp_sum(v);
            if (threadIdx.x == 0 && row0 + r < n_out) y[(size_t) j * n_out + row0 + r] = v;
        }
}

constexpr int hy4_q8x_rows(int nc, int it) {
    const int r = it >= 8 ? 1 : 8 / it;
    return nc * r <= 16 ? r : (16 / nc > 0 ? 16 / nc : 1);
}

template <int NC, int IT>
void hy4_launch_q8x_it(const Q8xArgs& a, int blocks, cudaStream_t s) {
    bk::launch_pdl(hy4_q8x_kernel<NC, hy4_q8x_rows(NC, IT), IT>, dim3((unsigned) blocks), dim3(32, 4), 0, s, a);
}

template <int IT>
void hy4_launch_q8x_nc(Q8xArgs a, const Q8xMat* mats, int n, int ncols, cudaStream_t s) {
    const int R = hy4_q8x_rows(ncols, IT);
    int end = 0;
    for (int i = 0; i < n; ++i) {
        end += (mats[i].n_out + R - 1) / R;
        a.end[i] = end;
    }
    switch (ncols) {
    case 1: hy4_launch_q8x_it<1, IT>(a, end, s); break;
    case 2: hy4_launch_q8x_it<2, IT>(a, end, s); break;
    case 3: hy4_launch_q8x_it<3, IT>(a, end, s); break;
    case 4: hy4_launch_q8x_it<4, IT>(a, end, s); break;
    case 5: hy4_launch_q8x_it<5, IT>(a, end, s); break;
    case 6: hy4_launch_q8x_it<6, IT>(a, end, s); break;
    case 7: hy4_launch_q8x_it<7, IT>(a, end, s); break;
    default: hy4_launch_q8x_it<8, IT>(a, end, s); break;
    }
}
}  // namespace

void q8x_mmvq(const Q8xMat* mats, int n, int n_in, int ncols, void* stream) {
    if (n < 1 || n > kQ8xMats || ncols < 1 || ncols > 8 || (n_in != 3072 && n_in != 6144))
        throw std::runtime_error("hy4flash q8x_mmvq: matrices, columns or width");
    Q8xArgs a{};
    a.n = n;
    for (int i = 0; i < n; ++i) {
        if (mats[i].n_out <= 0) throw std::runtime_error("hy4flash q8x_mmvq: rows");
        a.w[i] = static_cast<const Q80B*>(mats[i].w);
        a.x[i] = static_cast<const Q81B*>(mats[i].x);
        a.y[i] = mats[i].y;
        a.n_out[i] = mats[i].n_out;
    }
    cudaStream_t s = (cudaStream_t) stream;
    if (n_in == 3072) hy4_launch_q8x_nc<3>(a, mats, n, ncols, s);
    else hy4_launch_q8x_nc<6>(a, mats, n, ncols, s);
    if (!stream) cudaStreamSynchronize(nullptr);
}

}  // namespace basalt::models::hy4flash
