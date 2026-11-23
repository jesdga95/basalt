// src/models/nemotronh/ops.hpp - the Nemotron-H steps between the shared kernels' products.
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::models::nemotronh {

/// out[t][i] = sum over k of w[t][k] * rows[t K + k][i] for t < T, i < n: the routed experts' rows (each a device-visible
/// pointer: a card's scratch or mapped host memory) weighted by the router, summed in k order.  `rows` is a device
/// array of T K pointers.
void weighted_sum(const float* const* rows, const float* w, int T, int K, int n, float* out, void* stream);

/// x[i] = relu(x[i])^2 for n values.
void relu2(float* x, int64_t n, void* stream);

/// Rows [0, T) of k and v ([T][n_kv][hd] fp32) into the fp16 caches ([cell][n_kv][hd]) at cells pos0 ...
void kv_store(const float* k, const float* v, int T, int64_t pos0, int n_kv, int hd, uint16_t* kc, uint16_t* vc, void* stream);

/// x[i] += a[i] + b[i] for n values (b may be null).
void add2(float* x, const float* a, const float* b, int64_t n, void* stream);

}  // namespace basalt::models::nemotronh
