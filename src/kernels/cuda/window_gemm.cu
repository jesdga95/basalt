// src/kernels/cuda/window_gemm.cu - see include/basalt/kernels/window_gemm.hpp.
//
// A block is four warps over 16 weight rows.  The K range is cut into chunks of 256 values; warp w takes chunks w, w + 4,
// ... and streams each (its 16 rows' blocks, the 8 activation rows' codes and scales) into shared memory with cp.async,
// three or four chunks in flight, so the weights are read at memory speed while the tensor cores do 8 columns for free.
#include "basalt/kernels/window_gemm.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>

#include "basalt/log.hpp"

namespace basalt::kernels {
namespace {

constexpr int kWarps = 4;      // warps per block, each a quarter of K
constexpr int kChunk = 256;    // values per chunk
constexpr int kXStride = kChunk + 16;   // bytes per activation row in shared memory (padded: no bank conflicts)

template <int TYPE> struct Fmt;
// stages: chunks in flight per warp, within the 99 KiB of shared memory a block may have on sm_120
template <> struct Fmt<8> {    // Q8_0: fp16 scale + 32 int8 codes
    static constexpr int kBlockVals = 32, kBlockBytes = 34, kStages = 3;
};
template <> struct Fmt<40> {   // NVFP4: four UE4M3 scales + 32 bytes of E2M1 codes
    static constexpr int kBlockVals = 64, kBlockBytes = 36, kStages = 4;
};

template <int TYPE> struct Layout {
    static constexpr int kChunkBytes = kChunk / Fmt<TYPE>::kBlockVals * Fmt<TYPE>::kBlockBytes;   // one row's chunk
    static constexpr int kW = 16 * kChunkBytes;
    static constexpr int kX = kWinRows * kXStride;
    static constexpr int kD = kWinRows * (kChunk / 32) * 4;
    static constexpr int kStage = kW + kX + kD;
    static constexpr int kStages = Fmt<TYPE>::kStages;
    static constexpr int kSmem = kWarps * kStages * kStage + kWarps * 32 * 4 * 4;
    static_assert(kSmem <= 99 * 1024, "window_gemm: shared memory");
};

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { basalt::log::error("basalt", "%s: %s", what, cudaGetErrorString(e)); std::exit(1); }
}

__device__ __forceinline__ void cp16(void* smem, const void* gmem) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void mma_k32(int (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_k16(int (&c)[4], unsigned a0, unsigned a1, unsigned b0) {
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(b0));
}

// four bytes at a 2-byte aligned shared address
__device__ __forceinline__ unsigned ld_u32_a2(const uint8_t* p) {
    const unsigned lo = *(const uint16_t*) p, hi = *(const uint16_t*) (p + 2);
    return lo | (hi << 16);
}

// UE4M3 as ggml decodes it (halved; 0 and 0x7F give 0, bit 7 ignored)
__device__ __forceinline__ float ue4m3_half(unsigned x) {
    x &= 0x7Fu;
    const unsigned e = x >> 3, m = x & 7u;
    const float sub = (float) m * 0.0009765625f;   // m * 2^-10
    const float nrm = __uint_as_float(((e + 119u) << 23) | (m << 20));
    return (x == 0u || x == 0x7Fu) ? 0.0f : (e == 0u ? sub : nrm);
}

// four E2M1 codes (one per byte, low nibble) as doubled int8 values
__device__ __forceinline__ unsigned e2m1x4(unsigned c) {
    const unsigned x = c & 0x07070707u;
    const unsigned sel = (x & 0x7u) | ((x >> 4) & 0x70u) | ((x >> 8) & 0x700u) | ((x >> 12) & 0x7000u);
    const unsigned mag = __byte_perm(0x03020100u, 0x0C080604u, sel);
    const unsigned neg = ((c >> 3) & 0x01010101u) * 0xFFu;
    return __vsub4(mag ^ neg, neg);
}

template <int TYPE>
__global__ void __launch_bounds__(kWarps * 32) window_gemm_kernel(const uint8_t* __restrict__ w, int64_t row_bytes,
                                                                 int K, const int8_t* __restrict__ xq,
                                                                 const float* __restrict__ xd, float* __restrict__ y,
                                                                 int64_t ld_y, int T) {
    using L = Layout<TYPE>;
    constexpr int kStages = L::kStages;
    extern __shared__ __align__(16) uint8_t smem[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, lid = lane & 3;
    const int64_t row0 = (int64_t) blockIdx.x * 16;
    uint8_t* my = smem + warp * kStages * L::kStage;
    const int nch = K / kChunk;
    const int mine = nch > warp ? (nch - warp + kWarps - 1) / kWarps : 0;
    const uint8_t* wrow = w + row0 * row_bytes;
    const int nd = K / 32;

    auto issue = [&](int i) {   // my i-th chunk into stage i % kStages
        const int c = warp + i * kWarps;
        uint8_t* st = my + (i % kStages) * L::kStage;
        constexpr int wp = L::kChunkBytes / 16;   // 16-byte pieces per row
        for (int p = lane; p < 16 * wp; p += 32) {
            const int r = p / wp, q = p - r * wp;
            cp16(st + r * L::kChunkBytes + q * 16, wrow + r * row_bytes + (int64_t) c * L::kChunkBytes + q * 16);
        }
        for (int p = lane; p < kWinRows * (kChunk / 16); p += 32) {
            const int t = p >> 4, q = p & 15;
            cp16(st + L::kW + t * kXStride + q * 16, xq + (int64_t) t * K + (int64_t) c * kChunk + q * 16);
        }
        if (lane < kWinRows * 2) {
            const int t = lane >> 1, q = lane & 1;
            cp16(st + L::kW + L::kX + t * 32 + q * 16, xd + (int64_t) t * nd + c * (kChunk / 32) + q * 4);
        }
    };

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int i = 0; i < kStages - 1; ++i) {
        if (i < mine) issue(i);
        cp_commit();
    }
    for (int i = 0; i < mine; ++i) {
        if (i + kStages - 1 < mine) issue(i + kStages - 1);
        cp_commit();
        cp_wait<kStages - 1>();
        __syncwarp();
        const uint8_t* ws = my + (i % kStages) * L::kStage;
        const uint8_t* xs = ws + L::kW;
        const float* ds = (const float*) (ws + L::kW + L::kX);
        const uint8_t* r0 = ws + gid * L::kChunkBytes;
        const uint8_t* r1 = ws + (gid + 8) * L::kChunkBytes;
        if constexpr (TYPE == 8) {
#pragma unroll
            for (int b = 0; b < kChunk / 32; ++b) {
                const uint8_t* p0 = r0 + b * 34;
                const uint8_t* p1 = r1 + b * 34;
                const unsigned a0 = ld_u32_a2(p0 + 2 + 4 * lid), a1 = ld_u32_a2(p1 + 2 + 4 * lid);
                const unsigned a2 = ld_u32_a2(p0 + 18 + 4 * lid), a3 = ld_u32_a2(p1 + 18 + 4 * lid);
                const uint8_t* xb = xs + gid * kXStride + b * 32 + 4 * lid;
                const unsigned b0 = *(const unsigned*) xb, b1 = *(const unsigned*) (xb + 16);
                int c[4] = {0, 0, 0, 0};
                mma_k32(c, a0, a1, a2, a3, b0, b1);
                const float wd0 = __half2float(*(const __half*) p0), wd1 = __half2float(*(const __half*) p1);
                const float ad0 = ds[(2 * lid) * 8 + b], ad1 = ds[(2 * lid + 1) * 8 + b];
                acc[0] = fmaf((float) c[0], wd0 * ad0, acc[0]);
                acc[1] = fmaf((float) c[1], wd0 * ad1, acc[1]);
                acc[2] = fmaf((float) c[2], wd1 * ad0, acc[2]);
                acc[3] = fmaf((float) c[3], wd1 * ad1, acc[3]);
            }
        } else {
            const int sh = (lid >> 1) * 4;
#pragma unroll
            for (int b = 0; b < kChunk / 64; ++b) {
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    const int off = b * 36 + 4 + 8 * s + 4 * (lid & 1);
                    const unsigned q0 = *(const unsigned*) (r0 + off), q1 = *(const unsigned*) (r1 + off);
                    const unsigned a0 = e2m1x4((q0 >> sh) & 0x0F0F0F0Fu), a1 = e2m1x4((q1 >> sh) & 0x0F0F0F0Fu);
                    const unsigned b0 = *(const unsigned*) (xs + gid * kXStride + b * 64 + s * 16 + 4 * lid);
                    int c[4] = {0, 0, 0, 0};
                    mma_k16(c, a0, a1, b0);
                    const float ws0 = ue4m3_half(r0[b * 36 + s]), ws1 = ue4m3_half(r1[b * 36 + s]);
                    const int blk = b * 2 + (s >> 1);
                    const float ad0 = ds[(2 * lid) * 8 + blk], ad1 = ds[(2 * lid + 1) * 8 + blk];
                    acc[0] = fmaf((float) c[0], ws0 * ad0, acc[0]);
                    acc[1] = fmaf((float) c[1], ws0 * ad1, acc[1]);
                    acc[2] = fmaf((float) c[2], ws1 * ad0, acc[2]);
                    acc[3] = fmaf((float) c[3], ws1 * ad1, acc[3]);
                }
            }
        }
        __syncwarp();
    }
    cp_wait<0>();

    // the four warps' partials, added in warp order
    float* red = (float*) (smem + kWarps * kStages * L::kStage);
    if (warp > 0) {
#pragma unroll
        for (int j = 0; j < 4; ++j) red[((warp * 32) + lane) * 4 + j] = acc[j];
    }
    __syncthreads();
    if (warp != 0) return;
#pragma unroll
    for (int v = 1; v < kWarps; ++v)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[j] += red[((v * 32) + lane) * 4 + j];
    const int t0 = 2 * lid, t1 = 2 * lid + 1;
    const int64_t ra = row0 + gid, rb = row0 + gid + 8;
    if (t0 < T) { y[t0 * ld_y + ra] = acc[0]; y[t0 * ld_y + rb] = acc[2]; }
    if (t1 < T) { y[t1 * ld_y + ra] = acc[1]; y[t1 * ld_y + rb] = acc[3]; }
}

// one warp per (row, 32 values): amax / 127, codes rounded to nearest
__global__ void window_quantize_kernel(const float* __restrict__ x, int64_t ld_x, int64_t K, int8_t* __restrict__ q,
                                       float* __restrict__ d) {
    const int64_t nb = K / 32;
    const int64_t item = (int64_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (item >= kWinRows * nb) return;
    const int lane = threadIdx.x & 31;
    const int64_t t = item / nb, b = item - t * nb;
    const float v = x[t * ld_x + b * 32 + lane];
    float a = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    const float s = a / 127.0f;
    q[t * K + b * 32 + lane] = (int8_t) (s > 0.0f ? __float2int_rn(v / s) : 0);
    if (lane == 0) d[t * nb + b] = s;
}

template <int TYPE> void launch(const void* w, int64_t rows, int64_t K, const void* act, float* y, int64_t ld_y, int T,
                                cudaStream_t s) {
    using L = Layout<TYPE>;
    static bool raised = false;
    if (!raised) {
        cudaFuncSetAttribute(window_gemm_kernel<TYPE>, cudaFuncAttributeMaxDynamicSharedMemorySize, L::kSmem);
        raised = true;
    }
    const int8_t* xq = (const int8_t*) act;
    const float* xd = (const float*) ((const uint8_t*) act + (size_t) kWinRows * K);
    const int64_t row_bytes = K / Fmt<TYPE>::kBlockVals * Fmt<TYPE>::kBlockBytes;
    window_gemm_kernel<TYPE><<<(unsigned) (rows / 16), kWarps * 32, L::kSmem, s>>>((const uint8_t*) w, row_bytes, (int) K,
                                                                                xq, xd, y, ld_y, T);
    check("window_gemm");
}

}  // namespace

size_t window_act_bytes(int64_t K) { return (size_t) kWinRows * (size_t) K + (size_t) kWinRows * (size_t) (K / 32) * 4; }

void window_quantize(const float* x, int64_t ld_x, int64_t K, void* act, void* stream) {
    int8_t* q = (int8_t*) act;
    float* d = (float*) ((uint8_t*) act + (size_t) kWinRows * K);
    const int64_t items = kWinRows * (K / 32);
    window_quantize_kernel<<<(unsigned) ((items + 7) / 8), 256, 0, (cudaStream_t) stream>>>(x, ld_x, K, q, d);
    check("window_quantize");
}

bool window_gemm_supported(int type, int64_t rows, int64_t K, const void* w) {
    if (type != 8 && type != 40) return false;
    if (K <= 0 || K % kChunk != 0 || rows <= 0 || rows % 16 != 0 || ((uintptr_t) w & 15u) != 0) return false;
    const int64_t row_bytes = type == 8 ? K / 32 * 34 : K / 64 * 36;
    return row_bytes % 16 == 0;
}

void window_gemm(int type, const void* w, int64_t rows, int64_t K, const void* act, float* y, int64_t ld_y, int T,
                 void* stream) {
    const cudaStream_t s = (cudaStream_t) stream;
    if (type == 8) launch<8>(w, rows, K, act, y, ld_y, T, s);
    else launch<40>(w, rows, K, act, y, ld_y, T, s);
}

}  // namespace basalt::kernels
