// src/kernels/cpu/nvfp4_avx2.cpp - NVFP4 expert rows against Q8_0 activations, several tokens at once, in 256-bit
// lanes (nvfp4_avx2_rows.inc), and the choice between this file's rows and the AVX-VNNI-INT8 build of the same code.
#include "basalt/kernels/cpu/nvfp4_avx2.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "basalt/ggml/ggml_common.h"

#include <immintrin.h>

#include <bit>
#include <cmath>
#include <cstring>

#include "basalt/kernels/cpu/expert_layout.hpp"

#define BASALT_NVFP4_VNNI_INT8 0
#define BASALT_NVFP4_EXPORT(name) name##_base
#include "nvfp4_avx2_rows.inc"

namespace basalt::kernels::cpu {

bool nvfp4_256_fits(int n_embd, int n_ff) noexcept {
    return n_embd % QK_NVFP4 == 0 && n_ff % QK_NVFP4 == 0 && n_embd <= kMaxN && n_ff <= kMaxN;
}

void nvfp4_256_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, float sg, float su,
                       const void* const* act, int nt, float* const* ff, int r0, int r1) {
    static const bool vnni8 = cpu_avxvnni_int8_ok();
    if (vnni8) nvfp4_256_gu_rows_vnni(blob, gu_row, up_off, n, sg, su, act, nt, ff, r0, r1);
    else nvfp4_256_gu_rows_base(blob, gu_row, up_off, n, sg, su, act, nt, ff, r0, r1);
}

void nvfp4_256_down_rows(const uint8_t* w, size_t row_bytes, int n, float sd, const void* const* act, int nt,
                         float* const* out, int r0, int r1) {
    static const bool vnni8 = cpu_avxvnni_int8_ok();
    if (vnni8) nvfp4_256_down_rows_vnni(w, row_bytes, n, sd, act, nt, out, r0, r1);
    else nvfp4_256_down_rows_base(w, row_bytes, n, sd, act, nt, out, r0, r1);
}

}  // namespace basalt::kernels::cpu
