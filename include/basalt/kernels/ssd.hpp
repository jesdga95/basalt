// include/basalt/kernels/ssd.hpp - the Mamba-2 (state-space duality) mixer of a run of tokens, and its commit.
//
//   ssd_conv         the depthwise causal convolution of the raw xBC channels over the history and the run's earlier
//                    rows, its bias, then SiLU
//   ssd_scan         the selective state recurrence: dt = softplus(dt_raw + dt_bias), dA = exp(dt A),
//                    S = S dA + (dt x) B^T, y = S C + D x, head h reading group h / (n_heads / n_groups)'s B and C
//   ssd_gated_norm   y silu(z), RMS-normed over each group of d_inner / n_groups channels, times the weight
//   ssd_commit       once acceptance is known: the first n_keep rows replayed into the state; ssd_conv_commit the same
//                    for the convolution history
//
// A decode window never writes the recurrent state or the history: a rejected draft never touches them and nothing is
// snapshotted.  A prompt chunk passes its state through (state_out may be the state).  Every row runs the same
// operations in the same order whatever the run's length, so a token's outputs do not depend on T or its row, and a
// commit of n rows is bitwise the state ssd_scan reaches after them.
//
// Layouts (fp32): the state [n_heads][head_dim][d_state]; the history [conv_dim][d_conv - 1], oldest first; xbc rows of
// conv_dim channels [x (d_inner) | B (n_groups x d_state) | C (n_groups x d_state)]; y rows of d_inner.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace basalt::kernels {

struct SsdGeometry {
    int n_heads = 0, head_dim = 0, d_state = 0, n_groups = 0, d_conv = 0;
    int d_inner() const { return n_heads * head_dim; }
    int conv_dim() const { return d_inner() + 2 * n_groups * d_state; }
    size_t state_floats() const { return (size_t) n_heads * head_dim * d_state; }
    size_t hist_floats() const { return (size_t) conv_dim() * (d_conv - 1); }
};

/// Empty when the kernels take this geometry (head_dim 64 and d_state 128, a convolution of 4 taps, whole groups of
/// 1024 or fewer channels), else why not.
std::string ssd_check(const SsdGeometry& g);

/// Rows [0, T) of the raw xBC channels (row t at raw + t * ld_raw) -> xbc [T][conv_dim]: silu(bias + the taps over the
/// history `hist` and the run's earlier rows), the taps summed in order from the oldest.
void ssd_conv(const SsdGeometry& g, const float* raw, int64_t ld_raw, const float* hist, const float* w, const float* bias,
              int T, float* xbc, void* stream);

/// The recurrence of rows [0, T) from `state` (read, not written): y [T][d_inner].  dt_raw row t at dt_raw + t * ld_dt
/// (n_heads values); A = -exp(A_log), D and dt_bias [n_heads].  Optional: dt_out and dA_out [T][n_heads], the values a
/// commit replays; state_out, the state after the T rows (it may be `state`).
void ssd_scan(const SsdGeometry& g, const float* state, const float* xbc, const float* dt_raw, int64_t ld_dt,
              const float* dt_bias, const float* A, const float* D, int T, float* y, float* dt_out, float* dA_out,
              float* state_out, void* stream);

/// state <- the recurrence over rows [0, *n_keep) of a run ssd_scan stored (xbc, dt, dA): bitwise ssd_scan's state_out
/// after that many rows.  *n_keep is read on the device (0 leaves the state as it was).
void ssd_commit(const SsdGeometry& g, float* state, const float* xbc, const float* dt, const float* dA, const int32_t* n_keep,
                void* stream);

/// hist <- the last d_conv - 1 rows of [hist | raw_0 .. raw_{n-1}], n = *n_keep (device) or `n` (host).
void ssd_conv_commit(const SsdGeometry& g, float* hist, const float* raw, int64_t ld_raw, const int32_t* n_keep, void* stream);
void ssd_conv_advance(const SsdGeometry& g, float* hist, const float* raw, int64_t ld_raw, int n, void* stream);

/// Rows [0, T): v = y silu(z) (z row t at z + t * ld_z), out = v rsqrt(mean of v^2 over its group + eps) w.  `out` fp32
/// rows of d_inner and/or `out16` bf16 rows of d_inner (a window gemm's packed activations); either may be null.
void ssd_gated_norm(const SsdGeometry& g, const float* y, const float* z, int64_t ld_z, const float* w, float eps, int T,
                    float* out, uint16_t* out16, void* stream);

}  // namespace basalt::kernels
