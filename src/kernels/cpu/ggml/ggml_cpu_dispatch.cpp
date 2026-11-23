// Picks the AVX-512 or the AVX2 build of the extracted ggml dot products for this CPU (see ggml_cpu_dots.c).
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/cpu/expert_layout.hpp"

extern "C" {
const ggml_type_traits_cpu* ggml_get_type_traits_cpu_avx2(ggml_type type);
const ggml_type_traits_cpu* ggml_get_type_traits_cpu_avx512(ggml_type type);

const ggml_type_traits_cpu* ggml_get_type_traits_cpu(ggml_type type) {
    namespace cpu = basalt::kernels::cpu;
    static const bool avx512 = cpu::cpu_avx512_ok();
    static const bool avx2 = cpu::cpu_avx2_ok();
    if (avx512) return ggml_get_type_traits_cpu_avx512(type);
    if (avx2) return ggml_get_type_traits_cpu_avx2(type);
    return nullptr;
}

void ggml_mxfp8_q8_0_prep(int n, const void* y, float* out) {
    static const bool avx512 = basalt::kernels::cpu::cpu_avx512_ok();
    if (avx512) ggml_mxfp8_q8_0_prep_avx512(n, y, out);
    else ggml_mxfp8_q8_0_prep_avx2(n, y, out);
}

void ggml_mxfp8_q8_0_rows(int n, const void* w, size_t stride, const float* const* prep, int nt, float* const* out,
                          int r0, int r1) {
    static const bool avx512 = basalt::kernels::cpu::cpu_avx512_ok();
    if (avx512) ggml_mxfp8_q8_0_rows_avx512(n, w, stride, prep, nt, out, r0, r1);
    else ggml_mxfp8_q8_0_rows_avx2(n, w, stride, prep, nt, out, r0, r1);
}
}
