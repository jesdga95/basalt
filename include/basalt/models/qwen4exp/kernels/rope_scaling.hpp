// include/basalt/models/qwen4exp/kernels/rope_scaling.hpp - Flash-Next's one rope scaling for the process.
//
// The scaling itself (types, constants, validity) is shared: basalt/kernels/rope_scaling.hpp.  Flash-Next reaches it
// two ways from one config:
//
//   - the TABLE path (the default decode path, the indexer's pooled key) never changes its kernels:
//     scaling is baked into the cos/sin contents by `build_rope_table`, which the parity test checks
//     bit-exactly against a float64 reference;
//   - the ANALYTIC paths (the native rope, the native indexer, prefill) get the resolved constants
//     as kernel arguments.
//
// **PROCESS CONSTANTS, SET ONCE BEFORE CAPTURE.**  A CUDA graph bakes kernel ARGUMENTS at capture
// time, so the scaling must be fixed before `session_init` captures the layers - which is exactly
// the constraint the per-token values already respect by living in device memory (`st.pos_dev`).
// `rope_scaling_set()` is called once from the engine's startup, after CLI and GGUF defaults meet,
// before any graph exists.
#pragma once

#include "basalt/kernels/rope_scaling.hpp"

namespace basalt::kernels {

/// The process's one rope configuration.  Set it ONCE at startup, after the CLI and the model file
/// have agreed on the values and BEFORE `session_init` (which builds the rope table and captures the
/// graphs); read it at table build time and when launching the analytic kernels.  Plain storage, no
/// atomics: one writer on the startup thread, every reader after it.
void rope_scaling_set(const RopeScaling& scaling);
const RopeScaling& rope_scaling();

}  // namespace basalt::kernels
