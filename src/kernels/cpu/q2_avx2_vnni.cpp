// src/kernels/cpu/q2_avx2_vnni.cpp - q2_avx2_rows.inc built for AVX-VNNI: the same rows, the u8 x s8 products summed four
// at a time by one instruction.  Only called behind cpu_avxvnni_ok().
#include "basalt/kernels/cpu/expert.hpp"

#include <immintrin.h>

#include <cstring>

#define BASALT_Q2_VNNI 1
#define BASALT_Q2_EXPORT(name) name##_vnni
#include "q2_avx2_rows.inc"
