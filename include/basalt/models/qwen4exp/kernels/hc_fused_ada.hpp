// include/basalt/models/qwen4exp/kernels/hc_fused_ada.hpp - the one-kernel hyper-connection read on compute capability 8.9.
//
// hc_fused's read with cooperative async copies where that one uses tensor copies, for the draft graphs.  Its bits are
// fused_gr_read_multi's.  It shares hc_fused's arguments and 8-bit copies.
#pragma once

#include "basalt/models/qwen4exp/kernels/hc_fused.hpp"

namespace basalt::kernels {

/// The current device runs this read.
bool hc_fused_ada_supported();
/// Allocate the read's scratch on the current device, outside any stream capture.
void hc_fused_ada_init();
/// Enqueue the read on `stream`.  Aborts on invalid arguments or an unsupported device.
void hc_fused_ada_read(const HcFusedArgs& a, void* stream);
/// The arrivals a read adds to HcFusedArgs::pub_ctr[0] (its router items').
unsigned hc_fused_ada_pub_arrivals();
/// Tests only: per-CTA phase stamps of the next launches.
void hc_fused_ada_set_debug(unsigned long long* dbg);

}  // namespace basalt::kernels
