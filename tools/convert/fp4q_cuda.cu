// tools/convert/fp4q_cuda.cu - checkpoint_qwen35.fp4q_rows on the GPU: BF16 rows -> NVFP4 E2M1 codes and UE4M3 group scales.
// One thread per 16-value group, the same arithmetic in the same order as the numpy reference (float32, no fused
// multiply-add, the 16 squared errors summed as numpy's pairwise sum does), so the output is bit-identical to it.
// Build: nvcc -O3 -fmad=false -arch=sm_120 --shared -o fp4q_cuda.dll fp4q_cuda.cu
#include <cuda_runtime.h>

#include <cstdint>

namespace {

__constant__ float kUe[127];                 // the finite non-negative UE4M3 values, ascending
__constant__ float kE2[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
constexpr int kSteps = 25;                   // candidate scale codes around the group's amax / 6 (half to four times it)

// the candidate offsets in order of |d| (the lower first when tied): 0, -1, 1, then each pair up to -8, 8, then 9 to 16
__host__ __device__ inline int step(int k) {
    if (k < 17) return (k & 1) ? -((k + 1) / 2) : k / 2;
    return k - 8;
}

// UE4M3 code nearest x, ties to the even code (numpy: searchsorted, then the nearer of the two neighbours)
__device__ inline int ue4m3_round(float x) {
    x = fminf(x, 448.0f);
    int lo = 0, hi = 127;                    // first i with kUe[i] >= x
    while (lo < hi) {
        const int m = (lo + hi) >> 1;
        if (kUe[m] < x) lo = m + 1; else hi = m;
    }
    const int i = min(max(lo, 1), 126);
    const float l = kUe[i - 1], h = kUe[i];
    const bool up = (x - l > h - x) || ((x - l == h - x) && (i % 2 == 0));
    return up ? i : i - 1;
}

__device__ inline int e2m1_round(float x) {
    const float a = fabsf(x);
    const int c = (a > 0.25f) + (a >= 0.75f) + (a > 1.25f) + (a >= 1.75f) + (a > 2.5f) + (a >= 3.5f) + (a > 5.0f);
    return c | ((x < 0.0f && c > 0) ? 8 : 0);
}

__global__ void fp4q_kernel(const uint16_t* __restrict__ w, int64_t groups, float gs, uint8_t* codes, uint8_t* scales) {
    const int64_t gi = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (gi >= groups) return;
    float g[16];
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        g[j] = __uint_as_float((uint32_t) w[gi * 16 + j] << 16);
        amax = fmaxf(amax, fabsf(g[j]));
    }
    const int c0 = ue4m3_round(amax / (6.0f * gs));
    float best_err = 0.0f;
    int best = 0;
    for (int k = 0; k < kSteps; ++k) {
        const int c = min(max(c0 + step(k), 0), 126);
        const float sd = kUe[c] * gs;
        const float inv = sd > 0.0f ? 1.0f / sd : 0.0f;
        float e[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const float d = kE2[e2m1_round(g[j] * inv)] * sd - g[j];
            e[j] = d * d;
        }
        float r[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) r[j] = e[j] + e[j + 8];
        const float err = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        if (k == 0 || err < best_err) { best_err = err; best = c; }
    }
    const float sd = kUe[best] * gs;
    const float inv = sd > 0.0f ? 1.0f / sd : 0.0f;
    uint8_t* o = codes + gi * 8;
#pragma unroll
    for (int j = 0; j < 8; ++j) o[j] = (uint8_t) (e2m1_round(g[2 * j] * inv) | (e2m1_round(g[2 * j + 1] * inv) << 4));
    scales[gi] = (uint8_t) best;
}

uint16_t* d_w = nullptr;
uint8_t *d_codes = nullptr, *d_scales = nullptr;
int64_t cap = 0;

}  // namespace

extern "C" {

// ue: the 127 UE4M3 values (float32). 0 on success.
__declspec(dllexport) int fp4q_init(const float* ue) {
    return cudaMemcpyToSymbol(kUe, ue, sizeof(float) * 127) == cudaSuccess ? 0 : 1;
}

// rows x cols BF16 (host) -> codes [rows, cols / 2] and scales [rows, cols / 16] (host). 0 on success.
__declspec(dllexport) int fp4q_rows(const uint16_t* w, int64_t rows, int64_t cols, float gs, uint8_t* codes, uint8_t* scales) {
    const int64_t n = rows * cols, groups = n / 16;
    if (cols % 16 || n <= 0) return 2;
    if (n > cap) {
        cudaFree(d_w); cudaFree(d_codes); cudaFree(d_scales);
        if (cudaMalloc(&d_w, n * 2) || cudaMalloc(&d_codes, n / 2) || cudaMalloc(&d_scales, groups)) return 3;
        cap = n;
    }
    if (cudaMemcpy(d_w, w, n * 2, cudaMemcpyHostToDevice)) return 4;
    fp4q_kernel<<<(unsigned) ((groups + 127) / 128), 128>>>(d_w, groups, gs, d_codes, d_scales);
    if (cudaGetLastError()) return 5;
    if (cudaMemcpy(codes, d_codes, n / 2, cudaMemcpyDeviceToHost) || cudaMemcpy(scales, d_scales, groups, cudaMemcpyDeviceToHost)) return 6;
    return 0;
}

}
