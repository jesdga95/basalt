#include "models/deepseek4/dspark.hpp"

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

}  // namespace basalt::models::deepseek4
