// src/prefill/fp8_gemm.cu - see include/basalt/prefill/fp8_gemm.hpp.
//
// A block is 128 weight rows x 128 tokens, eight warps as 2 (rows) x 4 (tokens), each warp 64 rows x 32 tokens (4 x 4
// mma tiles), two blocks per SM, the token tiles fastest in the grid (a weight tile serves them back to back); K streams
// in 64-value steps through a three-stage cp.async pipeline, both operands' 16-byte chunks XOR-swizzled per row so the
// ldmatrix reads are conflict-free.  The weight rows are the mma's A (m16), the tokens its
// B (n8); fp32 accumulation over K in order, the two scales applied once in the epilogue.
#include "basalt/prefill/fp8_gemm.hpp"

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdlib>

#include "basalt/log.hpp"

namespace basalt::prefill::fp8 {
namespace {

constexpr int BM = 128, BN = 128, BK = 64, ST = 3, NT = 256;   // three stages: two blocks per SM

void ck(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "prefill fp8: %s: %s", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

struct Smem {
    uint8_t w[ST][BM * BK];
    uint8_t a[ST][BN * BK];
};

__device__ __forceinline__ unsigned su(const void* p) { return (unsigned) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp16(void* d, const void* s, bool ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(su(d)), "l"(s), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void ldsm4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(su(p)));
}
// byte offset of (row, k) in a tile of 64-byte rows, the four 16-byte chunks XOR-swizzled by row pair
__device__ __forceinline__ int sw(int r, int k) { return r * 64 + ((((k >> 4) ^ (r >> 1)) & 3) << 4) + (k & 15); }

size_t scale_off(int64_t rows, int64_t cols) { return ((size_t) rows * (size_t) cols + 15) & ~(size_t) 15; }

struct Args {
    const uint8_t* w;
    const float* w_scale;
    const uint8_t* a;
    const float* a_scale;
    float* dst;
    int64_t ld_dst;
    int M, K, N, epi;
};

__global__ void __launch_bounds__(NT, 2) gemm_kernel(const Args p) {
    extern __shared__ __align__(16) unsigned char raw[];
    Smem& s = *reinterpret_cast<Smem*>(raw);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp & 1, wn = warp >> 1;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;   // token tiles fastest: a weight tile serves them back to back
    const int nk = p.K / BK;
    auto issue = [&](int kt) {
        const int st = kt % ST, k0 = kt * BK;
        for (int i = tid; i < BM * 4; i += NT) {   // four 16-byte chunks per row
            const int r = i >> 2, c = i & 3;
            const bool wok = m0 + r < p.M, aok = n0 + r < p.N;
            cp16(s.w[st] + sw(r, c * 16), p.w + (size_t) (wok ? m0 + r : 0) * p.K + k0 + c * 16, wok);
            cp16(s.a[st] + sw(r, c * 16), p.a + (size_t) (aok ? n0 + r : 0) * p.K + k0 + c * 16, aok);
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    };
    float acc[4][4][4] = {};
#pragma unroll
    for (int i = 0; i < ST - 1; ++i) {
        if (i < nk) issue(i);
        else asm volatile("cp.async.commit_group;\n" ::: "memory");
    }
    const int g = lane >> 2, c = lane & 3;
    for (int kt = 0; kt < nk; ++kt) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(ST - 2) : "memory");
        __syncthreads();
        if (kt + ST - 1 < nk) issue(kt + ST - 1);
        else asm volatile("cp.async.commit_group;\n" ::: "memory");
        const int st = kt % ST;
#pragma unroll
        for (int kb = 0; kb < BK / 32; ++kb) {
            unsigned af[4][4], bf[4][2];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r = wm * 64 + i * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
                ldsm4(af[i][0], af[i][1], af[i][2], af[i][3], s.w[st] + sw(r, kb * 32 + (lane >> 4) * 16));
            }
#pragma unroll
            for (int j = 0; j < 4; j += 2) {
                const int r = wn * 32 + j * 8 + (lane >> 4) * 8 + (lane & 7);
                ldsm4(bf[j][0], bf[j][1], bf[j + 1][0], bf[j + 1][1], s.a[st] + sw(r, kb * 32 + ((lane >> 3) & 1) * 16));
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
                                 "{%8,%9}, {%0,%1,%2,%3};\n"
                                 : "+f"(acc[i][j][0]), "+f"(acc[i][j][1]), "+f"(acc[i][j][2]), "+f"(acc[i][j][3])
                                 : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]), "r"(bf[j][0]), "r"(bf[j][1]));
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    const float ws = p.w_scale ? *p.w_scale : 1.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int e1 = 0; e1 < 2; ++e1) {
            const int n = n0 + wn * 32 + j * 8 + 2 * c + e1;
            if (n >= p.N) continue;
            const float sc = ws * p.a_scale[n];
            float* row = p.dst + (int64_t) n * p.ld_dst;
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int e0 = 0; e0 < 2; ++e0) {
                    const int m = m0 + wm * 64 + i * 16 + g + 8 * e0;
                    if (m >= p.M) continue;
                    const float v = acc[i][j][2 * e0 + e1] * sc;
                    if (p.epi == (int) Epilogue::Add) row[m] += v;
                    else row[m] = v;
                }
        }
}

// One block per row: amax over the row, then E4M3 codes of x * 448 / amax (saturating, round to nearest even).
__global__ void __launch_bounds__(256) quantize_kernel(const float* __restrict__ x, int64_t cols, int64_t ld,
                                                      uint8_t* __restrict__ codes, float* __restrict__ scale) {
    __shared__ float red[8];
    __shared__ float inv_s;
    const float* xr = x + (int64_t) blockIdx.x * ld;
    float m = 0.0f;
    for (int64_t i = threadIdx.x * 4; i < cols; i += 1024) {
        const float4 v = *reinterpret_cast<const float4*>(xr + i);
        m = fmaxf(m, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int i = 0; i < 8; ++i) t = fmaxf(t, red[i]);
        scale[blockIdx.x] = t / 448.0f;
        inv_s = t > 0.0f ? 448.0f / t : 0.0f;
    }
    __syncthreads();
    const float inv = inv_s;
    uint8_t* cr = codes + (int64_t) blockIdx.x * cols;
    for (int64_t i = threadIdx.x * 4; i < cols; i += 1024) {
        const float4 v = *reinterpret_cast<const float4*>(xr + i);
        const __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(make_float2(v.x * inv, v.y * inv), __NV_SATFINITE, __NV_E4M3);
        const __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(make_float2(v.z * inv, v.w * inv), __NV_SATFINITE, __NV_E4M3);
        *reinterpret_cast<uint32_t*>(cr + i) = (uint32_t) lo | ((uint32_t) hi << 16);
    }
}

}  // namespace

size_t act_bytes(int64_t rows, int64_t cols) { return scale_off(rows, cols) + (size_t) rows * sizeof(float) + 16; }

void quantize(const float* x, int64_t rows, int64_t cols, int64_t ld, void* act, void* stream) {
    if (rows <= 0) return;
    if (cols % 64 != 0 || ld % 4 != 0) {
        basalt::log::error("basalt", "prefill fp8: rows of %lld values (ld %lld)", (long long) cols, (long long) ld);
        std::exit(1);
    }
    uint8_t* codes = (uint8_t*) act;
    float* scale = (float*) (codes + scale_off(rows, cols));
    quantize_kernel<<<(unsigned) rows, 256, 0, (cudaStream_t) stream>>>(x, cols, ld, codes, scale);
    ck("quantize");
}

void gemm(const uint8_t* w, int64_t w_rows, int64_t w_cols, const float* w_scale, const void* act, int64_t rows, float* dst,
          int64_t ld_dst, Epilogue epi, void* stream) {
    if (rows <= 0 || w_rows <= 0) return;
    if (w_cols % 64 != 0 || ((uintptr_t) w & 15) != 0) {
        basalt::log::error("basalt", "prefill fp8: a [%lld x %lld] matrix", (long long) w_rows, (long long) w_cols);
        std::exit(1);
    }
    static bool raised = false;
    if (!raised) {
        if (cudaFuncSetAttribute(gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(Smem)) != cudaSuccess) {
            basalt::log::error("basalt", "prefill fp8: shared-memory opt-in failed");
            std::exit(1);
        }
        raised = true;
    }
    Args p;
    p.w = w;
    p.w_scale = w_scale;
    p.a = (const uint8_t*) act;
    p.a_scale = (const float*) ((const uint8_t*) act + scale_off(rows, w_cols));
    p.dst = dst;
    p.ld_dst = ld_dst;
    p.M = (int) w_rows;
    p.K = (int) w_cols;
    p.N = (int) rows;
    p.epi = (int) epi;
    const dim3 grid((unsigned) ((rows + BN - 1) / BN), (unsigned) ((w_rows + BM - 1) / BM));
    gemm_kernel<<<grid, NT, sizeof(Smem), (cudaStream_t) stream>>>(p);
    ck("gemm");
}

}  // namespace basalt::prefill::fp8
