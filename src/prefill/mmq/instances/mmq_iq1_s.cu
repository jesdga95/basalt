// SPDX-License-Identifier: MIT
// Explicit instantiation of the MMQ kernel for IQ1_S; llama.cpp ggml/src/ggml-cuda/template-instances (the DeepSeek-V4.1
// support tree).  Copyright (c) 2023-2026 The ggml authors.  MIT licence: see THIRD_PARTY_LICENSES.
#include "../mmq.cuh"

DECL_MMQ_CASE(GGML_TYPE_IQ1_S);
