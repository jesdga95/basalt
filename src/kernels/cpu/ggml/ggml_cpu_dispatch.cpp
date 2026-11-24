// Picks the build of the extracted ggml dot products for this CPU: AVX-512 or AVX2 on x86-64 (ggml_cpu_dots.c),
// NEON with the dot product on Arm64 (ggml_cpu_dots_neon.c).
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/platform/compiler.hpp"

extern "C" {
const ggml_type_traits_cpu* ggml_get_type_traits_cpu(ggml_type type) {
    namespace cpu = basalt::kernels::cpu;
    if constexpr (basalt::platform::kArm64) {
        static const bool neon = cpu::cpu_neon_ok();
        return neon ? ggml_get_type_traits_cpu_neon(type) : nullptr;
    } else {
        static const bool avx512 = cpu::cpu_avx512_ok();
        static const bool avx2 = cpu::cpu_avx2_ok();
        if (avx512) return ggml_get_type_traits_cpu_avx512(type);
        if (avx2) return ggml_get_type_traits_cpu_avx2(type);
        return nullptr;
    }
}

void ggml_mxfp8_q8_0_prep(int n, const void* y, float* out) {
    if constexpr (basalt::platform::kArm64) {
        ggml_mxfp8_q8_0_prep_neon(n, y, out);
    } else {
        static const bool avx512 = basalt::kernels::cpu::cpu_avx512_ok();
        if (avx512) ggml_mxfp8_q8_0_prep_avx512(n, y, out);
        else ggml_mxfp8_q8_0_prep_avx2(n, y, out);
    }
}

void ggml_mxfp8_q8_0_rows(int n, const void* w, size_t stride, const float* const* prep, int nt, float* const* out,
                          int r0, int r1) {
    if constexpr (basalt::platform::kArm64) {
        ggml_mxfp8_q8_0_rows_neon(n, w, stride, prep, nt, out, r0, r1);
    } else {
        static const bool avx512 = basalt::kernels::cpu::cpu_avx512_ok();
        if (avx512) ggml_mxfp8_q8_0_rows_avx512(n, w, stride, prep, nt, out, r0, r1);
        else ggml_mxfp8_q8_0_rows_avx2(n, w, stride, prep, nt, out, r0, r1);
    }
}
}
