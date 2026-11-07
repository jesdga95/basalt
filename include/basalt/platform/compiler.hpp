// include/basalt/platform/compiler.hpp - the compiler- and CPU-specific spellings the host code needs, behind
// one name each, so no source file asks which compiler or architecture it is built with.
//
// x86-64 and Arm64. The CPU kernels are AVX2 / AVX-512 on x86-64 and NEON on Arm64; `kX86` says which set a build
// has. Every function is inline and maps to the one instruction or builtin each compiler gives for it.
#pragma once

#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#elif defined(__aarch64__) || defined(_M_ARM64)
#if defined(_MSC_VER)
#include <arm_neon.h>
#include <intrin.h>
#endif
#else
#error "basalt builds for x86-64 and Arm64"
#endif

namespace basalt::platform {

/// Whether this build is x86-64, with the AVX2 / AVX-512 CPU kernels; false on Arm64, whose kernels are NEON.
#if defined(__x86_64__) || defined(_M_X64)
inline constexpr bool kX86 = true;
#else
inline constexpr bool kX86 = false;
#endif

// MSVC can retain temporary stack copies for vector aggregate assignments even after inlining.
// Field-wise assignment exposes the values to register allocation.
#if defined(_MSC_VER) && !defined(__clang__)
inline constexpr bool kPreferMemberwiseVectorAssignment = true;
#else
inline constexpr bool kPreferMemberwiseVectorAssignment = false;
#endif

/// CPUID leaf `leaf`, subleaf `sub`: r[0..3] = EAX, EBX, ECX, EDX. All zero off x86, which reads as no leaves and so
/// as none of the x86 features.
inline void cpuid(unsigned leaf, unsigned sub, unsigned (&r)[4]) {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER)
    int x[4];
    __cpuidex(x, (int) leaf, (int) sub);
    for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
#else
    (void) leaf;
    (void) sub;
    r[0] = r[1] = r[2] = r[3] = 0;
#endif
}

/// XCR0, which says which register state the OS saves (read only after CPUID reports OSXSAVE). 0 off x86.
inline unsigned long long xgetbv0() {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    unsigned lo = 0, hi = 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((unsigned long long) hi << 32) | lo;
#endif
#else
    return 0;
#endif
}

/// The index of the lowest set bit of a nonzero mask.
inline int ctz32(unsigned m) {
#if defined(_MSC_VER)
    unsigned long i = 0;
    _BitScanForward(&i, m);
    return (int) i;
#else
    return __builtin_ctz(m);
#endif
}

/// One iteration of a spin wait: yields the pipeline to the core's SMT sibling (x86 `pause`, Arm `yield`).
inline void cpu_relax() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#elif defined(_MSC_VER)
    __yield();
#else
    __asm__ __volatile__("yield");
#endif
}

/// Orders this thread's earlier stores before its later ones for every observer, the GPU reading mapped host memory
/// included: the fence between a payload and the flag that announces it.
inline void store_fence() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_sfence();
#elif defined(_MSC_VER)
    __dmb(_ARM64_BARRIER_ST);
#else
    __asm__ __volatile__("dmb st" ::: "memory");
#endif
}

/// The fp32 -> half -> fp32 round trip, round to nearest even. Where the compiler has a half type it does the
/// conversion; MSVC has none, and its F16C (x86) or NEON (Arm) conversion pair is the same instruction with the same
/// rounding.
inline float f16_round(float x) {
#if defined(_MSC_VER) && (defined(__x86_64__) || defined(_M_X64))
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtps_ph(_mm_set_ss(x), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)));
#elif defined(_MSC_VER)
    return vgetq_lane_f32(vcvt_f32_f16(vcvt_f16_f32(vdupq_n_f32(x))), 0);
#else
    return (float) (_Float16) x;
#endif
}

/// Whether this CPU, and the OS saving its AVX-512 state, can run AVX512_BF16 with AVX512BW. False off x86.
inline bool cpu_has_avx512_bf16() {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER)
    // The same readings GCC's run-time probe takes: the OS has to save the AVX-512 state, and leaf 7 subleaf 0
    // names AVX512BW (EBX bit 30) and AVX512_BF16 (ECX bit 5).
    unsigned r[4];
    cpuid(0, 0, r);
    if (r[0] < 7u) return false;
    cpuid(1, 0, r);
    if (!((r[2] >> 27) & 1u)) return false;                 // OSXSAVE
    if ((xgetbv0() & 0xE6) != 0xE6) return false;           // the OS saves the AVX-512 state
    cpuid(7, 0, r);
    return ((r[1] >> 30) & 1u) != 0 && ((r[2] >> 5) & 1u) != 0;   // AVX512BW, AVX512_BF16
#else
    return __builtin_cpu_supports("avx512bf16") && __builtin_cpu_supports("avx512bw");
#endif
#else
    return false;
#endif
}

}  // namespace basalt::platform