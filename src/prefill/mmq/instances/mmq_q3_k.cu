// SPDX-License-Identifier: MIT
// Explicit instantiation of the MMQ kernel for Q3_K; llama.cpp commit 3cf03257f219afbe7334045ff7c6a06ac68c627d,
// ggml/src/ggml-cuda/template-instances.  Copyright (c) 2023-2026 The ggml authors.  MIT licence: see THIRD_PARTY_LICENSES.
#include "../mmq.cuh"

DECL_MMQ_CASE(GGML_TYPE_Q3_K);
