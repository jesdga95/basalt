// SPDX-License-Identifier: MIT
// Explicit instantiation of the MMQ kernel for MXFP4 (Basalt, the IQ4_NL instantiation adapted: same traits, the
// E8M0 scale).  Copyright (c) 2023-2026 The ggml authors.  MIT licence: see THIRD_PARTY_LICENSES.
#include "../mmq.cuh"

DECL_MMQ_CASE(GGML_TYPE_MXFP4);
