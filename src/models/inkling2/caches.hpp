// src/models/inkling2/caches.hpp - one visit's caches in a session: its keys and values (a ring of the window and a
// window's rows for a sliding-window visit, a cell per position for a full-attention one) and the rings of its short
// convolutions' inputs.
#pragma once

#include <cstdint>

namespace basalt::models::inkling2 {

struct VisitCache {
    void* k = nullptr;                   ///< [cap][kKvHeads][kHeadDim], position p in cell p % cap; fp16 or unit-scale E4M3
    void* v = nullptr;
    int64_t cap = 0;
    bool ring = false;                   ///< a sliding-window visit: cells are positions modulo cap
    float* k_ring = nullptr;             ///< the key convolution's inputs of the last kConvRing positions [kConvRing][1024]
    float* v_ring = nullptr;
    float* a_ring = nullptr;             ///< the attention output's, before its convolution [kConvRing][n_embd]
    float* m_ring = nullptr;             ///< the experts' sum's [kConvRing][n_embd]
};

}  // namespace basalt::models::inkling2
