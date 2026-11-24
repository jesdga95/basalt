// include/basalt/platform/compiler.hpp - the compiler- and CPU-specific spellings the host code needs, behind
// one name each, so no source file asks which compiler it is built with.
//
// Two architectures: x86-64 (the CPU kernels are AVX2 / AVX-512) and Arm64 (NEON). Every function is inline and maps
// to the one instruction or builtin each compiler gives for it. The x86 probes (cpuid, xgetbv0) answer zero on Arm64,
// so shared code may ask them and take its "the CPU has none of this" path.
#pragma once

#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#define BASALT_ARCH_X86 1
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#elif defined(__aarch64__) || defined(_M_ARM64)
#define BASALT_ARCH_ARM64 1
#include <arm_neon.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#else
#error "basalt builds for x86-64 and Arm64"
#endif

// Marks a function that uses AVX2, FMA and F16C intrinsics in a translation unit built without those flags. MSVC
// accepts the intrinsics in any function and needs no marking.
#if defined(_MSC_VER) || !defined(BASALT_ARCH_X86)
#define BASALT_TARGET_AVX2_FMA_F16C
#else
#define BASALT_TARGET_AVX2_FMA_F16C __attribute__((target("avx2,fma,f16c")))
#endif

namespace basalt::platform {

/// Which architecture this build targets. Shared code calls an x86 kernel only inside `if constexpr (kX86)` (or from a
/// file CMake builds for x86 alone), and a NEON one only under kArm64.
#if defined(BASALT_ARCH_X86)
inline constexpr bool kX86 = true, kArm64 = false;
#else
inline constexpr bool kX86 = false, kArm64 = true;
#endif

// MSVC can retain temporary stack copies for vector aggregate assignments even after inlining.
// Field-wise assignment exposes the values to register allocation.
#if defined(_MSC_VER) && !defined(__clang__)
inline constexpr bool kPreferMemberwiseVectorAssignment = true;
#else
inline constexpr bool kPreferMemberwiseVectorAssignment = false;
#endif

/// CPUID leaf `leaf`, subleaf `sub`: r[0..3] = EAX, EBX, ECX, EDX.
inline void cpuid(unsigned leaf, unsigned sub, unsigned (&r)[4]) {
#if !defined(BASALT_ARCH_X86)
    (void) leaf;
    (void) sub;
    r[0] = r[1] = r[2] = r[3] = 0;
#elif defined(_MSC_VER)
    int x[4];
    __cpuidex(x, (int) leaf, (int) sub);
    for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
}

/// XCR0, which says which register state the OS saves (read only after CPUID reports OSXSAVE).
inline unsigned long long xgetbv0() {
#if !defined(BASALT_ARCH_X86)
    return 0;
#elif defined(_MSC_VER)
    return _xgetbv(0);
#else
    unsigned lo = 0, hi = 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((unsigned long long) hi << 32) | lo;
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

/// One iteration of a spin wait. x86: `pause`, which yields the pipeline to the core's SMT sibling and takes tens of
/// cycles. Arm64: `isb`, which drains the pipeline and costs about as much; `yield` is a no-op on the cores this runs
/// on, and a wait that counts its pauses (64 between two driver queries) would spin several times faster with it.
inline void cpu_relax() {
#if defined(BASALT_ARCH_X86)
    _mm_pause();
#elif defined(_MSC_VER)
    __isb(_ARM64_BARRIER_SY);
#else
    __asm__ volatile("isb");
#endif
}

/// After the host's stores to memory another agent reads (a GPU kernel polling page-locked RAM, a pool worker), before
/// the store that publishes them. x86: `sfence`, which also drains non-temporal stores. Arm64 orders no two stores by
/// itself: `dmb ishst` puts every earlier store before every later one.
inline void store_fence() {
#if defined(BASALT_ARCH_X86)
    _mm_sfence();
#elif defined(_MSC_VER)
    __dmb(_ARM64_BARRIER_ISHST);
#else
    __asm__ volatile("dmb ishst" ::: "memory");
#endif
}

/// After the load that saw another agent's flag, before the loads of what the flag covers. x86 keeps loads in program
/// order and needs nothing. Arm64 may satisfy the later loads first (they would read what was there before the
/// flag rose): `dmb ishld` holds them until the flag's load has completed.
inline void load_fence() {
#if defined(BASALT_ARCH_X86)
#elif defined(_MSC_VER)
    __dmb(_ARM64_BARRIER_ISHLD);
#else
    __asm__ volatile("dmb ishld" ::: "memory");
#endif
}

/// A read prefetch of the line at `p` into every cache level (`near`) or into L2 and beyond.
inline void prefetch(const void* p, bool near = true) {
#if defined(BASALT_ARCH_X86)
    if (near) _mm_prefetch((const char*) p, _MM_HINT_T0);   // the hint is a compile-time constant on MSVC
    else _mm_prefetch((const char*) p, _MM_HINT_T1);
#elif defined(_MSC_VER)
    __prefetch2(p, near ? 0 /* PLDL1KEEP */ : 2 /* PLDL2KEEP */);
#else
    if (near) __builtin_prefetch(p, 0, 3);
    else __builtin_prefetch(p, 0, 2);
#endif
}

/// The fp32 -> half -> fp32 round trip, round to nearest even. Where the compiler has a half type it does the
/// conversion; MSVC has none, and its F16C intrinsic pair is the same instruction with the same rounding.
inline float f16_round(float x) {
#if defined(BASALT_ARCH_ARM64) && defined(_MSC_VER)
    // fcvt to half and back, in the FPCR's rounding mode (nearest even, which nothing in the engine changes)
    return vgetq_lane_f32(vcvt_f32_f16(vcvt_f16_f32(vdupq_n_f32(x))), 0);
#elif defined(_MSC_VER)
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtps_ph(_mm_set_ss(x), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)));
#else
    return (float) (_Float16) x;
#endif
}

/// Whether this CPU, and the OS saving its AVX-512 state, can run AVX512_BF16 with AVX512BW.
inline bool cpu_has_avx512_bf16() {
#if !defined(BASALT_ARCH_X86)
    return false;
#elif defined(_MSC_VER)
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
}

}  // namespace basalt::platform
