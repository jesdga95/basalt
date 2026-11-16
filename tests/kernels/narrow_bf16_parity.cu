// tests/kernels/narrow_bf16_parity.cu - the FP8 E4M3 and FP4 E2M1 conversions to bf16x2 (basalt/kernels/narrow_bf16.cuh)
// against the f16 route the header takes before CUDA 13.2 and against the host's values: every pair of E4M3 bytes, and
// every E2M1 byte at every UE4M3 scale byte in each byte of the word, bit for bit, NaNs included (their bits against the
// other route; the host only says NaN).  Built with CUDA 13.2 or later this checks the direct instruction against the
// fallback; built with 13.0 or 13.1 both sides are the fallback and the host values check it.  Exit 77 without a GPU.
#include "basalt/kernels/narrow_bf16.cuh"

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace bk = basalt::kernels;

namespace {

constexpr int kN = 1 << 16;

// i: two E4M3 bytes
__global__ void e4m3(unsigned* got, unsigned* alt) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    got[i] = bk::e4m3x2_bf16x2(i);
    alt[i] = bk::e4m3x2_bf16x2_via_f16(i);
}

// the word for i: byte k is the low byte of i plus 37 k, so every byte position sees all 256 values; the high byte of i
// is the UE4M3 scale
__host__ __device__ unsigned e2m1_word(unsigned i) {
    unsigned w = 0;
    for (int k = 0; k < 4; ++k) w |= ((i + 37u * k) & 0xffu) << (8 * k);
    return w;
}

__global__ void e2m1(unsigned* got, unsigned* alt) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned s = i >> 8;
    const unsigned sc = bk::e4m3x2_bf16x2(s | (s << 8));
    unsigned h[4], t[4];
    bk::e2m1x8_bf16x2_scaled(e2m1_word(i), sc, h);
    bk::e2m1x8_bf16x2_via_f16(e2m1_word(i), t);
    for (int k = 0; k < 4; ++k) {
        unsigned d;
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(t[k]), "r"(bk::e4m3x2_bf16x2_via_f16(s | (s << 8))));
        got[4 * i + k] = h[k];
        alt[4 * i + k] = d;
    }
}

float e4m3f(unsigned b) {
    __nv_fp8_e4m3 v;
    const uint8_t u = (uint8_t) b;
    std::memcpy(&v, &u, 1);
    return (float) v;
}
float e2m1f(unsigned n) {
    static const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return (n & 8) ? -mag[n & 7] : mag[n & 7];
}
bool bf16_nan(unsigned h) { return (h & 0x7f80u) == 0x7f80u && (h & 0x7fu); }
// the bf16 bits of v, which bf16 holds exactly
unsigned bf16_of(float v) {
    unsigned u;
    std::memcpy(&u, &v, 4);
    return u >> 16;
}

int fails = 0;
// one half: the two routes' bits equal, and the host's value (a NaN only as a NaN)
void check(const char* what, unsigned in, int half, unsigned got, unsigned alt, float want) {
    const unsigned g = (got >> (16 * half)) & 0xffffu, a = (alt >> (16 * half)) & 0xffffu;
    const bool ok = g == a && (want != want ? bf16_nan(g) : g == bf16_of(want));
    if (!ok && fails++ < 8)
        std::printf("FAIL %s input 0x%x half %d: direct 0x%04x, via f16 0x%04x, host 0x%04x\n", what, in, half, g, a,
                    bf16_of(want));
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) return 77;
    unsigned *got, *alt;
    if (cudaMalloc(&got, 4 * kN * sizeof(unsigned)) != cudaSuccess ||
        cudaMalloc(&alt, 4 * kN * sizeof(unsigned)) != cudaSuccess)
        return 77;
    std::vector<unsigned> g(4 * kN), a(4 * kN);

    e4m3<<<kN / 256, 256>>>(got, alt);
    if (cudaMemcpy(g.data(), got, kN * sizeof(unsigned), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(a.data(), alt, kN * sizeof(unsigned), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::printf("FAIL: e4m3 kernel: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    for (unsigned i = 0; i < kN; ++i)
        for (int half = 0; half < 2; ++half) check("e4m3x2", i, half, g[i], a[i], e4m3f(i >> (8 * half)));
    std::printf("e4m3x2: NaN 0x7f -> 0x%04x, 0xff -> 0x%04x\n", g[0x7f] & 0xffffu, g[0xff] & 0xffffu);

    e2m1<<<kN / 256, 256>>>(got, alt);
    if (cudaMemcpy(g.data(), got, 4 * kN * sizeof(unsigned), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(a.data(), alt, 4 * kN * sizeof(unsigned), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::printf("FAIL: e2m1 kernel: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    for (unsigned i = 0; i < kN; ++i) {
        const float sc = e4m3f(i >> 8);
        const unsigned w = e2m1_word(i);
        for (int k = 0; k < 4; ++k)
            for (int half = 0; half < 2; ++half)
                check("e2m1x8 scaled", i, half, g[4 * i + k], a[4 * i + k], e2m1f(w >> (8 * k + 4 * half)) * sc);
    }

    std::printf("%s (%s): %d failures\n", fails ? "FAIL" : "OK",
                BASALT_CVT_BF16X2_NARROW ? "the direct cvt against the f16 route" : "the f16 route on both sides",
                fails);
    return fails ? 1 : 0;
}
