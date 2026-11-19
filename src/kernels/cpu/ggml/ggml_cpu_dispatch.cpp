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
}
