// SPDX-License-Identifier: MIT
// llama.cpp commit 3cf03257f219afbe7334045ff7c6a06ac68c627d (https://github.com/ggml-org/llama.cpp),
// ggml/src/ggml-cuda/quantize.cuh.  Copyright (c) 2023-2026 The ggml authors.  MIT licence: see THIRD_PARTY_LICENSES.
#pragma once

#include "mmq.cuh"
#include "mmq_common.cuh"

#include <cstdint>

#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128

static_assert(MATRIX_ROW_PADDING % (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ) == 0, "Risk of out-of-bounds access.");

// Quantizes `ne1` rows of `ne0` floats (row i taken from x + ids[i] * s01 when `ids` is given) to block_q8_1_mmq,
// in the scale layout the weights of `type_x` multiply against.
void quantize_mmq_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, ggml_type type_x,
        int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);
// Basalt: `ne1` rows of `ne0` floats (row i at x + i * s01) quantized as above after each value becomes relu(x s_in)^2 s_down
// with its row's expert's scales (row r of expert j for bounds[j] <= r < bounds[j + 1]; s_in, s_down = tails[4 j], [4 j + 2]).
void quantize_mmq_q8_1_relu2_cuda(
        const float * x, void * vy, ggml_type type_x, int64_t ne00, int64_t s01, int64_t ne0, int64_t ne1,
        const int32_t * bounds, int n_experts, const float * tails, cudaStream_t stream);
