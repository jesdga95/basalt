// src/kernels/cuda/gdn_commit.cu - the plain GDN commit: see include/basalt/kernels/gdn_window.hpp.
#include "basalt/kernels/gdn_commit.cuh"

#include <cuda_runtime.h>

#include <cstdlib>
#include "basalt/log.hpp"

namespace basalt::kernels {
namespace {

using namespace gdn_dev;

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { basalt::log::error("basalt", "%s: %s", what, cudaGetErrorString(e)); std::exit(1); }
}

__global__ void gdn_conv_commit_kernel(float* __restrict__ hist, const float* __restrict__ qkv, int C,
                                       const int32_t* __restrict__ n_keep) {
    gdn_conv_commit_n(hist, qkv, C, *n_keep, blockIdx.x * blockDim.x + threadIdx.x);
}

template <int MAXT>
__global__ void __launch_bounds__(32 * RG) gdn_commit_layers_pipe_kernel(GdnCommitLayers a, int conv_blocks) {
    const int n = min(*a.n_keep, MAXT);
    if (n <= 0) return;
    commit_layers_body<MAXT>(a, n, conv_blocks);
}

}  // namespace

void gdn_commit_layers(const GdnCommitLayers& a, void* stream, int blocks, int max_n) {
    if (a.n_layers <= 0) return;
    const int conv_blocks = (a.C + 32 * RG - 1) / (32 * RG);
    const dim3 grid((unsigned) (blocks > 0 ? blocks : 1)), block(32, RG);
    if (max_n <= 8) gdn_commit_layers_pipe_kernel<8><<<grid, block, 0, (cudaStream_t) stream>>>(a, conv_blocks);
    else gdn_commit_layers_pipe_kernel<kDecMaxT><<<grid, block, 0, (cudaStream_t) stream>>>(a, conv_blocks);
    check("gdn_commit_layers");
}

void gdn_conv_commit(float* history, const float* qkv, int channels, const int32_t* n_keep, void* stream) {
    gdn_conv_commit_kernel<<<(unsigned) ((channels + 255) / 256), 256, 0, (cudaStream_t) stream>>>(history, qkv,
                                                                                                 channels, n_keep);
    check("gdn_conv_commit");
}

}  // namespace basalt::kernels
