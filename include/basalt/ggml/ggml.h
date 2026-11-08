// SPDX-License-Identifier: MIT
// The part of ggml's type and quantization API that Basalt uses: the type ids, block size and row size of each
// format, float conversion, reference (de)quantization, and the CPU row dot products with their activation
// quantizers.  Declarations follow ggml.h / ggml-cpu.h of llama.cpp commit 3cf03257f219afbe7334045ff7c6a06ac68c627d
// (https://github.com/ggml-org/llama.cpp).  Copyright (c) 2023-2026 The ggml authors.  MIT licence: see
// THIRD_PARTY_LICENSES.
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__cplusplus)
#define GGML_RESTRICT __restrict
#else
#define GGML_RESTRICT restrict
#endif

// Ids are the GGUF tensor type numbers.
enum ggml_type {
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S = 19,
    GGML_TYPE_IQ4_NL = 20,
    GGML_TYPE_IQ3_S = 21,
    GGML_TYPE_IQ2_S = 22,
    GGML_TYPE_IQ4_XS = 23,
    GGML_TYPE_IQ1_M = 29,
    GGML_TYPE_NVFP4 = 40,
    GGML_TYPE_Q2_0 = 42,
    GGML_TYPE_COUNT = 43,
};

typedef uint16_t ggml_fp16_t;

float ggml_fp16_to_fp32(ggml_fp16_t x);
ggml_fp16_t ggml_fp32_to_fp16(float x);

// nullptr / 0 for an id this build does not carry
int64_t ggml_blck_size(enum ggml_type type);
size_t ggml_type_size(enum ggml_type type);
size_t ggml_row_size(enum ggml_type type, int64_t ne);
const char* ggml_type_name(enum ggml_type type);

typedef void (*ggml_to_float_t)(const void* GGML_RESTRICT x, float* GGML_RESTRICT y, int64_t k);
typedef void (*ggml_from_float_t)(const float* GGML_RESTRICT x, void* GGML_RESTRICT y, int64_t k);

struct ggml_type_traits {
    const char* type_name;
    int64_t blck_size;
    size_t type_size;
    ggml_to_float_t to_float;
    ggml_from_float_t from_float_ref;
};

// nullptr for an id this build does not carry
const struct ggml_type_traits* ggml_get_type_traits(enum ggml_type type);

// Quantizes `nrows` rows of `n_per_row` floats (no importance matrix); returns the bytes written.
// Only the legacy and K formats are supported; the i-quant formats are read, never written.
size_t ggml_quantize_chunk(enum ggml_type type, const float* src, void* dst, int64_t start, int64_t nrows,
                           int64_t n_per_row);

typedef void (*ggml_vec_dot_t)(int n, float* GGML_RESTRICT s, size_t bs, const void* GGML_RESTRICT x, size_t bx,
                               const void* GGML_RESTRICT y, size_t by, int nrc);

struct ggml_type_traits_cpu {
    ggml_from_float_t from_float;  // quantizes an activation row to `type`; nullptr if it is not an activation type
    ggml_vec_dot_t vec_dot;        // weight row (this type) . activation row (vec_dot_type); nullptr if none
    enum ggml_type vec_dot_type;
    int64_t nrows;
};

// The dot products run on the widest x86 vector unit the CPU has (AVX-512 build, else AVX2).  nullptr for a type
// without a dot product, and for every type on a CPU with neither.
const struct ggml_type_traits_cpu* ggml_get_type_traits_cpu(enum ggml_type type);
// Each instruction set's table, whatever this CPU has (parity tests compare them).
const struct ggml_type_traits_cpu* ggml_get_type_traits_cpu_avx2(enum ggml_type type);
const struct ggml_type_traits_cpu* ggml_get_type_traits_cpu_avx512(enum ggml_type type);

// NVFP4 row . Q8_0 row in plain C: the reference for the SIMD vec_dot (ggml_quants.c).
void ggml_vec_dot_nvfp4_q8_0_ref(int n, float* GGML_RESTRICT s, const void* GGML_RESTRICT x, const void* GGML_RESTRICT y);

#if defined(__cplusplus)
}
#endif
