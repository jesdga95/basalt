// src/models/nemotronh/ops.hpp - the Nemotron-H steps between the shared kernels' products.
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::models::nemotronh {

/// h[r][c] = relu(h[r][c] s_in)^2 s_down for rows of n values grouped by expert: row r's expert j has bounds[j] <= r <
/// bounds[j + 1] (device, n_experts + 1 entries), its scales tails[4 j] and tails[4 j + 2] (an NVFP4 blob's tail).
void relu2_scaled_rows(float* h, int64_t rows, int n, const int32_t* bounds, int n_experts, const float* tails, void* stream);

/// For each of n experts staged at blobs + e stride (blob bytes each, an NVFP4 blob's 16-byte scale tail last): the tail into
/// tails[4 e ...] and the bytes from it to the next expert zeroed (what MMQ reads past a down matrix).
void tails_aside(uint8_t* blobs, size_t stride, uint64_t blob, int n, float* tails, void* stream);

/// y[r][c] = bf16(x[r ld_x + c]) for R rows of n values (y dense, n apart), the engine's one bf16 rounding.
void bf16_rows(const float* x, int64_t ld_x, int64_t n, int R, uint16_t* y, void* stream);

/// x[i] = relu(x[i])^2 for n values.
void relu2(float* x, int64_t n, void* stream);

/// Rows [0, T) of k and v ([T][n_kv][hd] fp32) into the fp16 caches ([cell][n_kv][hd]) at cells pos0 ...
void kv_store(const float* k, const float* v, int T, int64_t pos0, int n_kv, int hd, uint16_t* kc, uint16_t* vc, void* stream);

/// x[i] += a[i] + b[i] for n values (b may be null).
void add2(float* x, const float* a, const float* b, int64_t n, void* stream);

}  // namespace basalt::models::nemotronh
