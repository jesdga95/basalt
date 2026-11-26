// src/models/glm5next/kernels/kda_norm.cu - see kda_norm.hpp: the KDA gated output norm.
//
// One warp per row, lane owns channels q*32 + lane for q in [0, hd/32) - a fixed q-major layout so the sum of squares
// folds in a set order: the lane's q-chain first, then the warp butterfly (shfl_down, 16..1), which the test's oracle
// mirrors exactly. o and z are staged in registers (hd <= 1024, the model's is 128), everything in fp32 like the
// reference.
#include "models/glm5next/kernels/kda_norm.hpp"

#include <cuda_runtime.h>

namespace basalt::models::glm5next {
namespace {

constexpr int kMaxQ = 32;  // hd <= 1024

__global__ void kda_norm_kernel(const float* __restrict__ o, const float* __restrict__ z, const float* __restrict__ w,
                                float* __restrict__ out, int n_rows, int hd, float eps) {
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= n_rows) return;
    const int lane = threadIdx.x & 31;
    const float* orow = o + (size_t) row * hd;
    const float* zrow = z + (size_t) row * hd;
    const int nq = (hd + 31) >> 5;  // hd need not be a multiple of 32: the last quarter covers the tail

    float ov[kMaxQ], zv[kMaxQ];
    float ss = 0.0f;
    for (int q = 0; q < nq; ++q) {
        const int d = q * 32 + lane;
        ov[q] = d < hd ? orow[d] : 0.0f;
        zv[q] = d < hd ? zrow[d] : 0.0f;
        ss = fmaf(ov[q], ov[q], ss);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, off);
    ss = __shfl_sync(0xffffffffu, ss, 0);  // shfl_down leaves the full sum in lane 0 only
    const float scale = rsqrtf(ss / (float) hd + eps);

    float* orow_out = out + (size_t) row * hd;
    for (int q = 0; q < nq; ++q) {
        const int d = q * 32 + lane;
        if (d < hd) orow_out[d] = ov[q] * scale * w[d] / (1.0f + expf(-zv[q]));
    }
}

}  // namespace

void kda_norm(const float* o, const float* z, const float* w, float* out, int n_rows, int hd, float eps,
              cudaStream_t stream) {
    if (n_rows <= 0 || hd <= 0 || hd > kMaxQ * 32) return;  // the widths this kernel's quarters can hold
    const int warps = 8;
    kda_norm_kernel<<<(n_rows + warps - 1) / warps, warps * 32, 0, stream>>>(o, z, w, out, n_rows, hd, eps);
}

}  // namespace basalt::models::glm5next
