// include/basalt/kernels/cpu/mxfp4_avx2.hpp - AVX2 / AVX-VNNI multi-token dot products for MXFP4 expert rows against
// Q8_0 activations, bitwise equal per token to ggml's single-token AVX2 MXFP4 dot.
#pragma once

#include <cstddef>
#include <cstdint>

#include "basalt/kernels/cpu/swiglu.hpp"

namespace basalt::kernels::cpu {

#if defined(__x86_64__) || defined(_M_X64)

/// Whether a layer's geometry fits these kernels (rows a multiple of 64 values, at most 16384).
bool mxfp4_256_fits(int n_embd, int n_ff) noexcept;

/// ff[t][r] = swiglu(glu, gate_r . a[t], up_r . a[t]) for rows [r0, r1); gate rows at blob, up at blob + up_off.
void mxfp4_256_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const SwiGlu& glu,
                       const void* const* act, int nt, float* const* ff, int r0, int r1);

/// out[t][r] = w_r . a[t] for rows [r0, r1).
void mxfp4_256_down_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                         int r0, int r1);

/// Explicit ISA rungs (`_base`: AVX2 vpsignb+vpmaddubsw+vpmaddwd; `_vnni`: AVX-VNNI vpsignb+vpdpbusd behind cpu_avxvnni_ok).
void mxfp4_256_gu_rows_base(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const SwiGlu& glu,
                            const void* const* act, int nt, float* const* ff, int r0, int r1);
void mxfp4_256_gu_rows_vnni(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const SwiGlu& glu,
                            const void* const* act, int nt, float* const* ff, int r0, int r1);
void mxfp4_256_down_rows_base(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                              float* const* out, int r0, int r1);
void mxfp4_256_down_rows_vnni(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt,
                              float* const* out, int r0, int r1);

#else

inline bool mxfp4_256_fits(int, int) noexcept { return false; }
inline void mxfp4_256_gu_rows(const uint8_t*, size_t, size_t, int, const SwiGlu&, const void* const*, int, float* const*,
                              int, int) {}
inline void mxfp4_256_down_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {}
inline void mxfp4_256_gu_rows_base(const uint8_t*, size_t, size_t, int, const SwiGlu&, const void* const*, int,
                                   float* const*, int, int) {}
inline void mxfp4_256_gu_rows_vnni(const uint8_t*, size_t, size_t, int, const SwiGlu&, const void* const*, int,
                                   float* const*, int, int) {}
inline void mxfp4_256_down_rows_base(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {}
inline void mxfp4_256_down_rows_vnni(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {}

#endif

}  // namespace basalt::kernels::cpu
