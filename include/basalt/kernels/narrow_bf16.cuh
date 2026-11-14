// include/basalt/kernels/narrow_bf16.cuh - FP8 E4M3 and FP4 E2M1 codes as bf16x2 for a kernel's mma operands.  Device
// code only (no kernel): include from a .cu.
//
// Every E4M3 and E2M1 value is exact in bf16.  `cvt.rn.bf16x2.e4m3x2` and `cvt.rn.bf16x2.e2m1x2` (one instruction
// each) need PTX ISA 9.2, CUDA 13.2: the ptxas of CUDA 13.0 and 13.1 rejects them.  There the codes go to f16x2
// (`cvt.rn.f16x2.*`, also exact) and each half widens through f32 to bf16, which gives the same bits, NaNs included.
// tests/kernels/narrow_bf16_parity.cu compares the two ways on every input.
#pragma once

#include <cuda_runtime.h>

#if defined(__CUDACC_VER_MAJOR__) && \
    (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
#define BASALT_CVT_BF16X2_NARROW 1
#else
#define BASALT_CVT_BF16X2_NARROW 0
#endif

namespace basalt::kernels {

/// f16x2 as bf16x2, for values bf16 holds exactly (each half through f32)
__device__ __forceinline__ unsigned f16x2_bf16x2(unsigned f) {
    unsigned h;
    asm("{\n"
        ".reg .f16 l, u;\n"
        ".reg .f32 a, b;\n"
        "mov.b32 {l, u}, %1;\n"
        "cvt.f32.f16 a, l;\n"
        "cvt.f32.f16 b, u;\n"
        "cvt.rn.bf16x2.f32 %0, b, a;\n"
        "}\n"
        : "=r"(h)
        : "r"(f));
    return h;
}

/// two E4M3 bytes (the low 16 bits of x, the low byte the lower half) as bf16x2 through f16x2
__device__ __forceinline__ unsigned e4m3x2_bf16x2_via_f16(unsigned x) {
    unsigned f;
    const unsigned short v = (unsigned short) x;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f) : "h"(v));
    return f16x2_bf16x2(f);
}

/// the four bytes of w, each two E2M1 codes (the low nibble the lower half), as bf16x2 through f16x2
__device__ __forceinline__ void e2m1x8_bf16x2_via_f16(unsigned w, unsigned (&h)[4]) {
    unsigned f[4];
    asm("{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        "mov.b32 {b0, b1, b2, b3}, %4;\n"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
        "cvt.rn.f16x2.e2m1x2 %1, b1;\n"
        "cvt.rn.f16x2.e2m1x2 %2, b2;\n"
        "cvt.rn.f16x2.e2m1x2 %3, b3;\n"
        "}\n"
        : "=r"(f[0]), "=r"(f[1]), "=r"(f[2]), "=r"(f[3])
        : "r"(w));
#pragma unroll
    for (int i = 0; i < 4; ++i) h[i] = f16x2_bf16x2(f[i]);
}

/// two E4M3 bytes (the low 16 bits of x, the low byte the lower half) as bf16x2 (exact)
__device__ __forceinline__ unsigned e4m3x2_bf16x2(unsigned x) {
#if BASALT_CVT_BF16X2_NARROW
    unsigned h;
    const unsigned short v = (unsigned short) x;
    asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"(v));
    return h;
#else
    return e4m3x2_bf16x2_via_f16(x);
#endif
}

/// the four bytes of w, each two E2M1 codes (the low nibble the lower half), as bf16x2 times the bf16x2 scale sc (exact:
/// a code times a UE4M3 scale has at most five significant bits)
__device__ __forceinline__ void e2m1x8_bf16x2_scaled(unsigned w, unsigned sc, unsigned (&h)[4]) {
#if BASALT_CVT_BF16X2_NARROW
    asm("{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        ".reg .b32 t0, t1, t2, t3;\n"
        "mov.b32 {b0, b1, b2, b3}, %4;\n"
        "cvt.rn.bf16x2.e2m1x2 t0, b0;\n"
        "cvt.rn.bf16x2.e2m1x2 t1, b1;\n"
        "cvt.rn.bf16x2.e2m1x2 t2, b2;\n"
        "cvt.rn.bf16x2.e2m1x2 t3, b3;\n"
        "mul.rn.bf16x2 %0, t0, %5;\n"
        "mul.rn.bf16x2 %1, t1, %5;\n"
        "mul.rn.bf16x2 %2, t2, %5;\n"
        "mul.rn.bf16x2 %3, t3, %5;\n"
        "}\n"
        : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3])
        : "r"(w), "r"(sc));
#else
    unsigned t[4];
    e2m1x8_bf16x2_via_f16(w, t);
#pragma unroll
    for (int i = 0; i < 4; ++i) asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(h[i]) : "r"(t[i]), "r"(sc));
#endif
}

}  // namespace basalt::kernels
