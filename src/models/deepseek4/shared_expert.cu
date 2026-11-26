#include "shared_expert.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <climits>
#include <stdexcept>
#include <string>

namespace basalt::models::deepseek4 {
namespace {

__global__ void swiglu_clamped_kernel(const float* gate, const float* up, int n, float limit, float* output) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = fminf(gate[i], limit);
    const float u = fminf(fmaxf(up[i], -limit), limit);
    output[i] = g / (1.f + __expf(-g)) * u;
}

}  // namespace

void swiglu_clamped(const float* gate, const float* up, int width, float limit, float* output, int n_tokens,
                    void* stream) {
    const int64_t n = static_cast<int64_t>(width) * n_tokens;
    if (!gate || !up || !output || width <= 0 || n_tokens <= 0 || n > INT_MAX || !(limit > 0.f))
        throw std::invalid_argument("deepseek4 SwiGLU: invalid arguments");
    swiglu_clamped_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        gate, up, static_cast<int>(n), limit, output);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("deepseek4 SwiGLU: ") + cudaGetErrorString(error));
}

}  // namespace basalt::models::deepseek4
