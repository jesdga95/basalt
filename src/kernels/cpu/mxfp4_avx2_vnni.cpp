// src/kernels/cpu/mxfp4_avx2_vnni.cpp - mxfp4_avx2_rows.inc compiled with AVX-VNNI (`vpdpbusd`).
#include "basalt/kernels/cpu/mxfp4_avx2.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "basalt/ggml/ggml_common.h"

#include <immintrin.h>

#include <bit>
#include <cmath>
#include <cstring>

#include "basalt/kernels/cpu/expert_layout.hpp"

#define BASALT_MXFP4_VNNI 1
#define BASALT_MXFP4_EXPORT(name) name##_vnni
#include "mxfp4_avx2_rows.inc"
