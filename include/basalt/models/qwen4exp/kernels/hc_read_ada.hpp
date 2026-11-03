// include/basalt/models/qwen4exp/kernels/hc_read_ada.hpp - the hyper-connection read on compute capability 8.9 as one persistent kernel.
//
// The outputs of hc8_gr_read_multi (the norm, down and up kernels of hc8_gr.cu), bit for bit, on one CTA per SM.  Each
// CTA holds a slice of the columns of w_norm and of the tokens' rows in shared memory and walks its share of the down
// rows over them, so an SM reads only that slice; the sums meet in global memory.
#pragma once

#include "basalt/models/qwen4exp/kernels/fused_gr.hpp"

namespace basalt::kernels {

/// On by default on compute capability 8.9; `--exp hc_ada=0` keeps hc8_gr's three kernels.
void hc_read_ada_enable(bool on);
/// Allocate the kernel's scratch on the current device (outside a stream capture, once); nothing on another device.
void hc_read_ada_init();
/// The current device and these arguments run the kernel (device, geometry, 8-bit copies, alignment).  The kernel needs
/// every CTA resident: a caller takes it only when no side branch runs beside the read.
bool hc_read_ada_supported(const FusedGrArgs* a, int n_tok);
/// Enqueue the read on `stream`.  Aborts on invalid arguments.
void hc_read_ada(const FusedGrArgs* a, int n_tok, void* stream);
/// Tests only: a grid smaller than the device's (0: one CTA per SM), and per-CTA phase times of the next launches into
/// stamps[launch][ctas][kHcAdaStamps] (null: off; `ord` picks the launch).
void hc_read_ada_test_grid(int ctas);
constexpr int kHcAdaStamps = 9;
void hc_read_ada_test_stamps(unsigned long long* stamps, int ord);
/// Tests only: the kernel's CTAs per SM the device can hold for n_tok tokens, and the shared memory one takes.
int hc_read_ada_occupancy(int n_tok, int* smem_bytes);

}  // namespace basalt::kernels
