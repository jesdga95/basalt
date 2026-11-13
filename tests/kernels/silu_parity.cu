// tests/kernels/silu_parity.cu - kernels::silu_quotient (basalt/kernels/silu.cuh) against the plain expression
// g / (1 + exp(-g)) as a kernel compiles it, on every one of the 2^32 float bit patterns: the two must agree bit for
// bit, NaNs included.  Exit 77 without a GPU.
#include "basalt/kernels/silu.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

namespace {

struct Bad {
    unsigned in, want, got;
};

__global__ void compare(unsigned long long start, unsigned long long* n_bad, Bad* bad) {
    const unsigned long long i = start + (unsigned long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >> 32) return;
    const float g = __uint_as_float((unsigned) i);
    const unsigned want = __float_as_uint(g / (1.0f + __expf(-g)));
    const unsigned got = __float_as_uint(basalt::kernels::silu_quotient(g));
    if (want != got) {
        const unsigned long long k = atomicAdd(n_bad, 1ull);
        if (k < 8) bad[k] = {(unsigned) i, want, got};
    }
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) return 77;
    unsigned long long* n_bad;
    Bad* bad;
    if (cudaMalloc(&n_bad, 8) != cudaSuccess || cudaMalloc(&bad, 8 * sizeof(Bad)) != cudaSuccess) return 77;
    cudaMemset(n_bad, 0, 8);
    const unsigned long long step = 1ull << 28;
    for (unsigned long long s = 0; s < (1ull << 32); s += step) compare<<<(unsigned) (step / 256), 256>>>(s, n_bad, bad);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::printf("silu_quotient: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    unsigned long long n = 0;
    Bad h[8];
    cudaMemcpy(&n, n_bad, 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(h, bad, sizeof h, cudaMemcpyDeviceToHost);
    std::printf("silu_quotient vs g / (1 + exp(-g)): %llu of 2^32 inputs differ\n", n);
    for (unsigned long long k = 0; k < n && k < 8; ++k) {
        float in;
        std::memcpy(&in, &h[k].in, 4);
        std::printf("  in %08x (%g): plain %08x, silu_quotient %08x\n", h[k].in, in, h[k].want, h[k].got);
    }
    if (n == 0) std::printf("PASS\n");
    return n == 0 ? 0 : 1;
}
