// src/kernels/cpu/nvfp4_avx2_vnni.cpp - nvfp4_avx2_rows.inc built for AVX-VNNI-INT8: the same rows, the signed int8
// products summed directly into int32 by `vpdpbssd`.  Only called behind cpu_avxvnni_int8_ok().

#include "basalt/kernels/cpu/nvfp4_avx2.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "basalt/ggml/ggml_common.h"

#include <immintrin.h>

#include <bit>
#include <cmath>
#include <cstring>

#define BASALT_NVFP4_VNNI_INT8 1
#define BASALT_NVFP4_EXPORT(name) name##_vnni
#include "nvfp4_avx2_rows.inc"

