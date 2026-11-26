// src/kernels/cpu/mxfp4_avx2.cpp - MXFP4 multi-token expert rows in AVX2 (mxfp4_avx2_rows.inc) + AVX-VNNI dispatch.
#include "basalt/kernels/cpu/mxfp4_avx2.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "basalt/ggml/ggml_common.h"

#include <immintrin.h>

#include <bit>
#include <cmath>
#include <cstring>

#include "basalt/kernels/cpu/expert_layout.hpp"

#define BASALT_MXFP4_VNNI 0
#define BASALT_MXFP4_EXPORT(name) name##_base
#include "mxfp4_avx2_rows.inc"

namespace basalt::kernels::cpu {

bool mxfp4_256_fits(int n_embd, int n_ff) noexcept {
    return n_embd % (2 * QK_MXFP4) == 0 && n_ff % (2 * QK_MXFP4) == 0 && n_embd <= kMaxN && n_ff <= kMaxN;
}

void mxfp4_256_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const SwiGlu& glu,
                       const void* const* act, int nt, float* const* ff, int r0, int r1) {
    static const bool vnni = cpu_avxvnni_ok();
    if (vnni) mxfp4_256_gu_rows_vnni(blob, gu_row, up_off, n, glu, act, nt, ff, r0, r1);
    else mxfp4_256_gu_rows_base(blob, gu_row, up_off, n, glu, act, nt, ff, r0, r1);
}

void mxfp4_256_down_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                         int r0, int r1) {
    static const bool vnni = cpu_avxvnni_ok();
    if (vnni) mxfp4_256_down_rows_vnni(w, row_bytes, n, act, nt, out, r0, r1);
    else mxfp4_256_down_rows_base(w, row_bytes, n, act, nt, out, r0, r1);
}

}  // namespace basalt::kernels::cpu
