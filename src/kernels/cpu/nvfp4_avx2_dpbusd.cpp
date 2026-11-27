// src/kernels/cpu/nvfp4_avx2_dpbusd.cpp - nvfp4_avx2_rows.inc built for AVX-VNNI: the same rows, with the products of
// the unsigned |weight| bytes and the signed activations summed into int32 by `vpdpbusd` in place of the `vpmaddubsw`
// and `vpmaddwd` that compute exactly those four-wise sums.  Only called behind cpu_avxvnni_ok() on a CPU without
// AVX-VNNI-INT8 (which takes the one-instruction `vpdpbssd` build instead).

#include "basalt/kernels/cpu/nvfp4_avx2.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "basalt/ggml/ggml_common.h"

#include <immintrin.h>

#include <bit>
#include <cmath>
#include <cstring>

#define BASALT_NVFP4_VNNI_INT8 0
#define BASALT_NVFP4_VNNI 1
#define BASALT_NVFP4_EXPORT(name) name##_dpbusd
#include "nvfp4_avx2_rows.inc"
