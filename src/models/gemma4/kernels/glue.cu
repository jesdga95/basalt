// src/models/gemma4/kernels/glue.cu - the glue between the products (basalt/models/gemma4/kernels.hpp): a block per row,
// its sums in a fixed order, so a row's bits do not depend on the rows beside it.
#include "basalt/models/gemma4/kernels.hpp"

#include "basalt/kernels/bf16_bits.hpp"
#include "basalt/kernels/doorbell.cuh"
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/q8_1_fm.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace basalt::models::gemma4 {
namespace {

namespace bk = basalt::kernels;

constexpr int kThreads = 256;
constexpr int kEndThreads = 1024;   // ffn_end: one block per row reads its k experts' rows, so the block is wide

void check(const char* what) {
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess)
        throw std::runtime_error(std::string("gemma4 ") + what + ": " + cudaGetErrorString(e));
}

// The block's sum of v (every thread gets it), in a fixed order.
template <int T = kThreads>
__device__ float block_sum(float v) {
    __shared__ float part[T / 32];
    __shared__ float total;
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.f;
        for (int i = 0; i < T / 32; ++i) s += part[i];
        total = s;
    }
    __syncthreads();
    const float r = total;
    __syncthreads();
    return r;
}

// 1 / rms of a row of `width` values.
template <int T = kThreads>
__device__ float inv_rms(const float* __restrict__ x, int width, float eps) {
    float ss = 0.f;
    for (int i = threadIdx.x; i < width; i += T) ss += x[i] * x[i];
    return 1.0f / sqrtf(block_sum<T>(ss) / (float) width + eps);
}

__device__ __forceinline__ float gelu_tanh(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__global__ void __launch_bounds__(kThreads) embed_kernel(const uint16_t* table, const int32_t* __restrict__ tokens, int width,
                                                          float scale, float* __restrict__ out) {
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    const int r = blockIdx.x;
    const uint16_t* row = table + (size_t) tokens[r] * width;
    for (int i = threadIdx.x; i < width; i += kThreads)
        out[(size_t) r * width + i] = __uint_as_float((uint32_t) row[i] << 16) * scale;
}

__global__ void __launch_bounds__(kThreads) add_norm_kernel(float* __restrict__ x, const float* __restrict__ y,
                                                             const float* __restrict__ w, float eps, int width) {
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    const int r = blockIdx.x;
    const float* yr = y + (size_t) r * width;
    const float inv = inv_rms(yr, width, eps);
    for (int i = threadIdx.x; i < width; i += kThreads) x[(size_t) r * width + i] += yr[i] * inv * w[i];
}

// grid (slice, row): each block the row's 1 / rms (the whole row: x is only read), then its slice of xr and of the q8_1
// blocks, a warp a block.
__global__ void __launch_bounds__(kThreads) moe_inputs_kernel(const float* __restrict__ x, const float* __restrict__ w, float eps,
                                                              int width, float* __restrict__ xr, bk::Q81Fm* __restrict__ xq) {
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    const int r = blockIdx.y;
    const float* row = x + (size_t) r * width;
    const float inv = inv_rms(row, width, eps);
    const int lane = threadIdx.x & 31, nb = width / 32;
    const int b = blockIdx.x * (kThreads / 32) + (threadIdx.x >> 5);
    if (b >= nb) return;
    const float v = row[b * 32 + lane] * inv;
    xr[(size_t) r * width + b * 32 + lane] = v;
    bk::q8_1_store_fm(v * w[b * 32 + lane], xq, (size_t) r * nb + b, lane);
}

__global__ void __launch_bounds__(kThreads) gelu_pack_kernel(const float* __restrict__ g, const float* __restrict__ u, int n,
                                                              int64_t ld_gu, int64_t ld, uint16_t* __restrict__ act) {
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    const int r = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * kThreads + threadIdx.x;
    if (i >= ld) return;
    act[r * ld + i] = bk::bf16_from_f32(i < n ? gelu_tanh(g[r * ld_gu + i]) * u[r * ld_gu + i] : 0.f);
}

__global__ void __launch_bounds__(kThreads) gelu_mul_kernel(const float* __restrict__ g, const float* __restrict__ u, int n,
                                                             int64_t ld_gu, int64_t ld, float* __restrict__ out) {
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    const int r = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * kThreads + threadIdx.x;
    if (i >= ld) return;
    out[r * ld + i] = i < n ? gelu_tanh(g[r * ld_gu + i]) * u[r * ld_gu + i] : 0.f;
}

// One row: the experts' sum (slot order) and the MLP output normed and added, the sum normed, the residual and the scalar.
// `buf` holds the row's experts' sum while the norms read it.
__global__ void __launch_bounds__(kEndThreads) ffn_end_kernel(FfnEnd f, int width) {
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    extern __shared__ float sm[];   // [2][width]: the MLP part, then the experts' sum
    const int r = blockIdx.x;
    float* a = sm;
    float* b = sm + width;
    const float* mlp = f.mlp + (size_t) r * width;
    if (f.experts) {
        for (int i = threadIdx.x; i < width; i += kEndThreads) {
            float s = 0.f;
            for (int k = 0; k < f.k; ++k) s += f.weights[(size_t) r * f.k + k] * f.experts[((size_t) r * f.k + k) * width + i];
            b[i] = s;
        }
        __syncthreads();
        const float i1 = inv_rms<kEndThreads>(mlp, width, f.eps);
        const float i2 = inv_rms<kEndThreads>(b, width, f.eps);
        for (int i = threadIdx.x; i < width; i += kEndThreads) a[i] = mlp[i] * i1 * f.n1[i] + b[i] * i2 * f.n2[i];
    } else {
        for (int i = threadIdx.x; i < width; i += kEndThreads) a[i] = mlp[i];
    }
    __syncthreads();
    const float ip = inv_rms<kEndThreads>(a, width, f.eps);
    const float sc = f.scalar[0];
    float* x = f.x + (size_t) r * width;
    float ss = 0.f;
    for (int i = threadIdx.x; i < width; i += kEndThreads) {
        const float v = (x[i] + a[i] * ip * f.n_post[i]) * sc;
        x[i] = v;
        b[i] = v;   // each thread rereads its own values
        ss += v * v;
    }
    if (!f.next_w) return;
    const float in = 1.0f / sqrtf(block_sum<kEndThreads>(ss) / (float) width + f.eps);
    for (int i = threadIdx.x; i < width; i += kEndThreads) {
        const float v = b[i] * in * f.next_w[i];
        if (f.act) static_cast<uint16_t*>(f.act)[(size_t) r * width + i] = bk::bf16_from_f32(v);
        if (f.xn) f.xn[(size_t) r * width + i] = v;
    }
}

__global__ void softcap_kernel(float* __restrict__ x, int64_t n, float cap) {
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    const float inv = 1.0f / cap;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        x[i] = cap * tanhf(x[i] * inv);
}

constexpr int kRingThreads = 256;

// One block: the rows with a pair off the card, then (when any) the ticket, the choices, those rows' inputs and the ring.
__global__ void __launch_bounds__(kRingThreads) moe_ring_kernel(const int32_t* __restrict__ ids, int rows, int k,
                                                                const unsigned long long* __restrict__ tab,
                                                                const float* __restrict__ xr, const float* __restrict__ w,
                                                                int width, int layer, float* x_m, int32_t* ids_m, MoeBell* bell,
                                                                uint32_t* seq) {
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    __shared__ uint32_t off_rows, stopped;
    const int n = rows * k;
    if (threadIdx.x == 0) off_rows = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += kRingThreads) {
        const int e = ids[i];
        if (e >= 0 && !tab[e]) atomicOr(&off_rows, 1u << (i / k));
    }
    __syncthreads();
    const uint32_t m = off_rows;
    if (!m) return;
    uint32_t ticket = 0;
    if (threadIdx.x == 0) {
        ticket = *seq + 1;
        if (!ticket) ticket = 1;
        *seq = ticket;
        stopped = bk::doorbell_load_acquire(&bell->abort);
    }
    __syncthreads();
    if (stopped) return;
    for (int i = threadIdx.x; i < n; i += kRingThreads) ids_m[i] = ids[i];
    for (int r = 0; r < rows; ++r)
        if (m >> r & 1)
            for (int c = threadIdx.x; c < width; c += kRingThreads) x_m[(size_t) r * width + c] = xr[(size_t) r * width + c] * w[c];
    if (threadIdx.x == 0) bk::doorbell_store_relaxed(&bell->ticket, ticket);
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) bk::doorbell_store_release(&bell->ring, 1u | (uint32_t) layer << 8 | (uint32_t) rows << 16);
}

constexpr int kMergeThreads = 256;

// Block p: a pair off the card takes row p of the host's rows once its ticket is done; the card's pairs stay.
__global__ void __launch_bounds__(kMergeThreads) moe_merge_kernel(const int32_t* __restrict__ ids,
                                                                  const unsigned long long* __restrict__ tab,
                                                                  const float* host_rows, float* __restrict__ out, int width,
                                                                  MoeBell* bell, const uint32_t* seq) {
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    const int p = blockIdx.x, e = ids[p];
    if (e < 0 || tab[e]) return;
    __shared__ uint32_t taken;
    if (threadIdx.x == 0) taken = bk::doorbell_spin(&bell->done, *seq, &bell->abort, 8);
    __syncthreads();
    if (!taken) return;
    float* o = out + (size_t) p * width;
    for (int c = threadIdx.x; c < width; c += kMergeThreads) o[c] = __ldcv(host_rows + (size_t) p * width + c);
}

}  // namespace

void embed_rows(const uint16_t* table, const int32_t* tokens, int rows, int width, float scale, float* out, void* stream) {
    if (rows <= 0) return;
    bk::launch_pdl(embed_kernel, dim3((unsigned) rows), dim3(kThreads), 0, (cudaStream_t) stream, table, tokens, width, scale, out);
    check("embed_rows");
}

void add_norm(float* x, const float* y, const float* w, float eps, int rows, int width, void* stream) {
    if (rows <= 0) return;
    bk::launch_pdl(add_norm_kernel, dim3((unsigned) rows), dim3(kThreads), 0, (cudaStream_t) stream, x, y, w, eps, width);
    check("add_norm");
}

void moe_inputs(const float* x, const float* w, float eps, int rows, int width, float* xr, void* xq, void* stream) {
    if (rows <= 0) return;
    if (width % 32) throw std::invalid_argument("gemma4 moe_inputs: a width not a multiple of 32");
    const int slices = (width / 32 + kThreads / 32 - 1) / (kThreads / 32);
    bk::launch_pdl(moe_inputs_kernel, dim3((unsigned) slices, (unsigned) rows), dim3(kThreads), 0, (cudaStream_t) stream, x, w, eps,
                   width, xr, (bk::Q81Fm*) xq);
    check("moe_inputs");
}

void gelu_pack(const float* g, const float* u, int rows, int n, int64_t ld_gu, int64_t ld, void* act, void* stream) {
    if (rows <= 0) return;
    bk::launch_pdl(gelu_pack_kernel, dim3((unsigned) ((ld + kThreads - 1) / kThreads), (unsigned) rows), dim3(kThreads), 0,
                   (cudaStream_t) stream, g, u, n, ld_gu, ld, (uint16_t*) act);
    check("gelu_pack");
}

void gelu_mul(const float* g, const float* u, int rows, int n, int64_t ld_gu, int64_t ld, float* out, void* stream) {
    if (rows <= 0) return;
    bk::launch_pdl(gelu_mul_kernel, dim3((unsigned) ((ld + kThreads - 1) / kThreads), (unsigned) rows), dim3(kThreads), 0,
                   (cudaStream_t) stream, g, u, n, ld_gu, ld, out);
    check("gelu_mul");
}

void ffn_end(const FfnEnd& f, int rows, int width, void* stream) {
    if (rows <= 0) return;
    const size_t smem = (size_t) 2 * width * sizeof(float);
    cudaFuncSetAttribute(ffn_end_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
    bk::launch_pdl(ffn_end_kernel, dim3((unsigned) rows), dim3(kEndThreads), smem, (cudaStream_t) stream, f, width);
    check("ffn_end");
}

void softcap(float* x, int64_t n, float cap, void* stream) {
    if (n <= 0) return;
    bk::launch_pdl(softcap_kernel, dim3(1024), dim3(256), 0, (cudaStream_t) stream, x, n, cap);
    check("softcap");
}

void moe_ring(const int32_t* ids, int rows, int k, const unsigned long long* tab, const float* xr, const float* w, int width,
              int layer, float* x_m, int32_t* ids_m, MoeBell* bell, uint32_t* seq, void* stream) {
    if (rows <= 0) return;
    if (rows > kRows || rows * k > kMoePairs || layer < 0 || layer > 0xff)
        throw std::invalid_argument("gemma4 moe_ring: rows, pairs or layer");
    bk::launch_pdl(moe_ring_kernel, dim3(1), dim3(kRingThreads), 0, (cudaStream_t) stream, ids, rows, k, tab, xr, w, width, layer,
                   x_m, ids_m, bell, seq);
    check("moe_ring");
}

void moe_merge(const int32_t* ids, int rows, int k, const unsigned long long* tab, const float* host_rows, float* out, int width,
               MoeBell* bell, const uint32_t* seq, void* stream) {
    if (rows <= 0) return;
    if (rows * k > kMoePairs) throw std::invalid_argument("gemma4 moe_merge: pairs");
    bk::launch_pdl(moe_merge_kernel, dim3((unsigned) (rows * k)), dim3(kMergeThreads), 0, (cudaStream_t) stream, ids, tab, host_rows,
                   out, width, bell, seq);
    check("moe_merge");
}

}  // namespace basalt::models::gemma4
