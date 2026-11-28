#include "routed_expert.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <climits>
#include <stdexcept>
#include <string>

namespace basalt::models::deepseek4 {
namespace {

constexpr int K = 6;

__global__ void weighted_route_sum_kernel(const float* rows, const float* weights, int n_tokens, int n_embd,
                                          float* output) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_tokens * n_embd;
    if (i >= total) return;
    const int token = i / n_embd;
    const int column = i - token * n_embd;
    float value = 0.f;
#pragma unroll
    for (int route = 0; route < K; ++route) {
        const int row = token * K + route;
        value += weights[row] * rows[(size_t) row * n_embd + column];
    }
    output[i] = value;
}

__global__ void weighted_route_sum_ptrs_kernel(WindowRowPtrs ptrs, const float* weights, const float* shared,
                                                int n_tokens, int n_embd, float* output) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_tokens * n_embd;
    if (i >= total) return;
    const int token = i / n_embd;
    const int column = i - token * n_embd;
    float value = shared ? shared[i] : 0.f;
#pragma unroll
    for (int route = 0; route < K; ++route) {
        const int row = token * K + route;
        value += weights[row] * ptrs.p[row][column];
    }
    output[i] = value;
}

__global__ void swiglu_split_clamped_kernel(const float* gate_up, int64_t n, int n_ff, float limit, float* output) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t row = i / n_ff;
    const int col = (int) (i - row * n_ff);
    const float gate = fminf(gate_up[row * 2 * n_ff + col], limit);
    const float up = fminf(fmaxf(gate_up[row * 2 * n_ff + n_ff + col], -limit), limit);
    output[i] = gate / (1.f + __expf(-gate)) * up;
}

}  // namespace

void weighted_route_sum(const float* rows, const float* weights, int n_tokens, int n_embd, float* output, void* stream) {
    const int64_t total = static_cast<int64_t>(n_tokens) * n_embd;
    if (!rows || !weights || !output || n_tokens <= 0 || n_embd <= 0 || total > INT_MAX || !stream)
        throw std::invalid_argument("deepseek4 route sum: invalid arguments");
    weighted_route_sum_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        rows, weights, n_tokens, n_embd, output);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("deepseek4 route sum: ") + cudaGetErrorString(error));
}

void weighted_route_sum_ptrs(const WindowRowPtrs& ptrs, const float* weights, const float* shared,
                             int n_tokens, int n_embd, float* output, void* stream) {
    const int64_t total = static_cast<int64_t>(n_tokens) * n_embd;
    if (!weights || !output || n_tokens <= 0 || n_tokens > 8 || n_embd <= 0 || total > INT_MAX || !stream)
        throw std::invalid_argument("deepseek4 route sum ptrs: invalid arguments");
    weighted_route_sum_ptrs_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0,
                                     static_cast<cudaStream_t>(stream)>>>(ptrs, weights, shared, n_tokens, n_embd, output);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("deepseek4 route sum ptrs: ") + cudaGetErrorString(error));
}

void swiglu_split_clamped(const float* gate_up, int n_rows, int n_ff, float limit, float* output, void* stream) {
    const int64_t total = (int64_t) n_rows * n_ff;
    if (!gate_up || !output || n_rows <= 0 || n_ff <= 0 || total > INT_MAX || !(limit > 0.f) || !stream)
        throw std::invalid_argument("deepseek4 split SwiGLU: invalid arguments");
    swiglu_split_clamped_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0,
                                  static_cast<cudaStream_t>(stream)>>>(gate_up, total, n_ff, limit, output);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("deepseek4 split SwiGLU: ") + cudaGetErrorString(error));
}

namespace {

__global__ void stage_copy_floats_kernel(const float* __restrict__ src, float* __restrict__ dst, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

__global__ void stage_copy_words_kernel(const uint32_t* __restrict__ src, uint32_t* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

__global__ void stage_copy_strided_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                          int n_rows, int row_floats, int dst_stride_floats) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = (int64_t) n_rows * row_floats;
    if (i >= total) return;
    const int r = (int) (i / row_floats);
    const int c = (int) (i - (int64_t) r * row_floats);
    dst[(int64_t) r * dst_stride_floats + c] = src[i];
}

}  // namespace

void stage_copy_floats(const float* src, float* dst, int64_t n, void* stream) {
    if (n <= 0) return;
    stage_copy_floats_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        src, dst, n);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("deepseek4 stage_copy_floats: ") + cudaGetErrorString(error));
}

void stage_copy_words(const uint32_t* src, uint32_t* dst, int n, void* stream) {
    if (n <= 0) return;
    stage_copy_words_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        src, dst, n);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("deepseek4 stage_copy_words: ") + cudaGetErrorString(error));
}

void stage_copy_strided(const float* src, float* dst, int n_rows, int row_floats, int dst_stride_floats, void* stream) {
    const int64_t total = (int64_t) n_rows * row_floats;
    if (total <= 0) return;
    stage_copy_strided_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        src, dst, n_rows, row_floats, dst_stride_floats);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("deepseek4 stage_copy_strided: ") + cudaGetErrorString(error));
}

}  // namespace basalt::models::deepseek4
