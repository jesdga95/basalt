// include/basalt/kernels/narrow_bf16.cuh - FP8 E4M3 and FP4 E2M1 codes as bf16x2 for a kernel's mma operands.  Device
// code only (no kernel): include from a .cu.
//
// Every E4M3 and E2M1 value is exact in bf16.  `cvt.rn.bf16x2.e4m3x2` and `cvt.rn.bf16x2.e2m1x2` (one instruction
// each) need PTX ISA 9.2, CUDA 13.2: the ptxas of CUDA 13.0 and 13.1 rejects them.  There the codes go to f16x2
// (`cvt.rn.f16x2.*`, also exact) and each half widens through f32 to bf16, which gives the same bits, NaNs included.
// tests/kernels/narrow_bf16_parity.cu compares the two ways on every input.
//
// Compute capability 8.9 has the E4M3 conversion (cvt.rn.f16x2.e4m3x2) but neither E2M1 conversion nor mul.rn.bf16x2
// (9.0).  Its device pass decodes E2M1 by a byte permute from the eight magnitudes' bit patterns, the sign bit OR'd in
// (every E2M1 value is exact in f16 and bf16, so these are the instruction's bits), and multiplies by fma.rn.bf16x2 with a
// -0 addend: x * y + -0 is the exact product rounded once, as mul.rn rounds it (a zero product keeps its sign).
#pragma once

#include <cuda_runtime.h>

#if defined(__CUDACC_VER_MAJOR__) && \
    (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && \
    !(defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000)   // 8.9 has neither direct bf16x2 conversion: the f16 route
#define BASALT_CVT_BF16X2_NARROW 1
#else
#define BASALT_CVT_BF16X2_NARROW 0
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
#define BASALT_E2M1_PERMUTE 1   // no E2M1 conversion in this device pass (8.9)
#else
#define BASALT_E2M1_PERMUTE 0
#endif

namespace basalt::kernels {

/// bf16x2 a * b, both halves rounded once (mul.rn.bf16x2, 9.0+; 8.9: fma.rn.bf16x2 with a -0 addend, the same rounding of
/// the same exact product)
__device__ __forceinline__ unsigned mul_bf16x2(unsigned a, unsigned b) {
    unsigned d;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u));
#else
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
#endif
    return d;
}

#if BASALT_E2M1_PERMUTE
/// byte i of w (two E2M1 codes, the low nibble the lower half) as a pair of 16-bit patterns: hi_* and lo_* are the high
/// and low bytes of the eight magnitudes' patterns (codes 0..7) as prmt's eight-byte tables; the sign bits OR'd in
__device__ __forceinline__ unsigned e2m1x2_permute(unsigned w, int i, unsigned hi_0, unsigned hi_1, unsigned lo_0,
                                                   unsigned lo_1) {
    const unsigned b = (w >> (8 * i)) & 0xffu;
    const unsigned sel = (b & 0x7u) | ((b & 0x70u) << 4);   // the low half's magnitude in nibble 0, the high half's in 2
    unsigned h, l;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(h) : "r"(hi_0), "r"(hi_1), "r"(sel << 4));   // bytes 1, 3 (0, 2: code 0's 0x00)
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(l) : "r"(lo_0), "r"(lo_1), "r"(sel));        // bytes 0, 2 (1, 3: code 0's 0x00)
    return h | l | ((b & 0x08u) << 12) | ((b & 0x80u) << 24);
}
#endif

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
#if BASALT_E2M1_PERMUTE
    // f16 patterns of 0, 0.5, 1, 1.5, 2, 3, 4, 6: 0x0000 0x3800 0x3c00 0x3e00 0x4000 0x4200 0x4400 0x4600 (low bytes zero)
#pragma unroll
    for (int i = 0; i < 4; ++i) f[i] = e2m1x2_permute(w, i, 0x3e3c3800u, 0x46444240u, 0u, 0u);
#else
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
#endif
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
#if BASALT_E2M1_PERMUTE
    // bf16 patterns of 0, 0.5, 1, 1.5, 2, 3, 4, 6: 0x0000 0x3f00 0x3f80 0x3fc0 0x4000 0x4040 0x4080 0x40c0
#pragma unroll
    for (int i = 0; i < 4; ++i)
        h[i] = mul_bf16x2(e2m1x2_permute(w, i, 0x3f3f3f00u, 0x40404040u, 0xc0800000u, 0xc0804000u), sc);
#elif BASALT_CVT_BF16X2_NARROW
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
    for (int i = 0; i < 4; ++i) h[i] = mul_bf16x2(t[i], sc);
#endif
}

}  // namespace basalt::kernels
