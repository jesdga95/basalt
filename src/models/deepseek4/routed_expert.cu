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

}  // namespace basalt::models::deepseek4
