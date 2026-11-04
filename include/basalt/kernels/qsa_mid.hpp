// include/basalt/kernels/qsa_mid.hpp - the QSA query projection in two halves.
//
// The query projection's rows come in heads of 512: 256 query rows, then 256 gate rows.  The attention needs only the
// query rows; the gate rows are read by the merge.  qsa_qproj_half projects one half (the 16-row tiles 32 h .. 32 h + 15
// or 32 h + 16 .. 32 h + 31 of every head h) into the full projection's output with dense_tc's whole-tile arithmetic
// for that shape: each tile's values are the bits dense_tc writes for it, so the two halves together are dense_tc's
// output.  The query half runs on every SM; the gate half on a share of them beside the attention.
#pragma once

namespace basalt::kernels {

/// dense_tc takes this projection (its bit in dense_tc_mask, the type, the shape) as whole 16-row tiles, the layout
/// the halves reproduce; n_out a multiple of 512; and the device has the SMs to run the gate half beside the attention.
bool qsa_qproj_half_supported(int ggml_type, const void* weights, int n_in, int n_out);

/// One half (0: query rows, 1: gate rows) of y = W x for ncols (1..8) q8_1 columns, written at the rows dense_tc
/// writes them (y[c * n_out + row]).  blocks: the grid (0: one block per SM for the query half, the gate share of the
/// SMs for the gate half).  The query half lets its dependents in once its last tile is read, as dense_tc; the gate
/// half at its start (whatever waits for it still waits for its end).  A programmatic dependent inside a PdlScope.
/// Aborts on arguments qsa_qproj_half_supported rejects.
void qsa_qproj_half(int ggml_type, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                    int half, int blocks, void* stream);

}  // namespace basalt::kernels
