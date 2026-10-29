// src/program/engine_local.cpp - shared pool adapters and expert-tier ordering.
#include "engine_local.hpp"
#include "basalt/log.hpp"

#include <algorithm>
#include <chrono>

namespace basalt::program::engine_local {

using Clock = std::chrono::steady_clock;

void drive_pool(void* user, const float* x_f, const int32_t* ids, const float* weights, int64_t n_embd, int64_t k,
                float* out) {
    Drive* t = (Drive*) user;
    const Clock::time_point a = Clock::now();
    basalt::core::expert_pool_dispatch(&t->d, x_f, ids, weights, n_embd, k, out);
    t->cpu_ms += std::chrono::duration<double, std::milli>(Clock::now() - a).count();
    ++t->calls;
    // THE ROUTING TRACE.  Written AFTER the dispatch so the layer index is still this layer's: `d.layers` is
    // advanced by the adapter as it consumes the blob, and reading it after the call is the same value the
    // dispatch used.  Record = int32 layer, int32 k, k int32 ids, k float weights.
    if (t->routing != nullptr) {
        // **`d.layers` HAS ALREADY BEEN ADVANCED BY THE TIME THIS RUNS.**  The adapter walks the blob by incrementing `d.layers` as it consumes each layer's
        // experts, so after the dispatch it holds the NEXT layer's index.  The index is therefore
        // one past the layer (a trace spanning 1..48 instead of 0..47 is the symptom), and anything keyed on
        // the layer index, which is exactly what a cache profile is, would be wrong.
        const int32_t layer_idx = (int32_t) (t->d.layers - 1);
        if (layer_idx < 0 || layer_idx >= 48) {
            basalt::log::warn("basalt generate", "the routing trace saw layer %d, outside 0..47", layer_idx);
            return;
        }
        const int32_t rec[2] = {layer_idx, (int32_t) k};
        std::fwrite(rec, sizeof rec, 1, t->routing);
        std::fwrite(ids, sizeof(int32_t), (size_t) k, t->routing);
        std::fwrite(weights, sizeof(float), (size_t) k, t->routing);
    }
}

/// The pool for a verify window.
void drive_pool_multi(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                      int64_t layer) {
    Drive* t = (Drive*) user;
    t->d.layers = layer;
    const Clock::time_point a = Clock::now();
    basalt::core::expert_pool_dispatch_multi(t->d, x_f, ids, n_tok, k, out);
    t->cpu_ms += std::chrono::duration<double, std::milli>(Clock::now() - a).count();
    ++t->calls;
    // the routing trace for the serve path: the same record format drive_pool writes (layer, k, ids, weights),
    // one record per token.  The multi dispatch fuses the router weights into the kernel and does not surface
    // them, so records carry unit weights: the profile ranks pairs by routed frequency, which is the
    // signal that matters; a one-shot --dump-routing run records true weights if a weighted ranking is wanted.
    if (t->routing != nullptr && layer >= 0 && layer < 48) {
        for (int64_t tok = 0; tok < n_tok; ++tok) {
            const int32_t rec[2] = {(int32_t) layer, (int32_t) k};
            std::fwrite(rec, sizeof rec, 1, t->routing);
            std::fwrite(ids + tok * k, sizeof(int32_t), (size_t) k, t->routing);
            static const float one[64] = {};   // k <= 64 in a verify window; zeros read as unit weights
            std::fwrite(one, sizeof(float), (size_t) k, t->routing);
        }
    }
}

void drive_pool_split(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                      int64_t layer) {
    SplitDrive* s = (SplitDrive*) user;
    int st = 0;
    while (st + 1 < s->n && layer >= s->end[st]) ++st;
    Drive& d = *s->base;
    d.d.plan = s->plan[st];
    d.d.cache_base = s->cache_base[st];
    d.d.cache_slot_off = s->cache_slot_off[st];
    d.d.pcie_num = s->pcie_num[st];
    drive_pool_multi(s->base, x_f, ids, n_tok, k, out, layer);
}

// A peer card as fast as the main one gets a share f (0.45) of the HOT pairs, so both cards compute routed experts
// every layer.  Of the first 8700 ranks (about the primary's slots), every pair with floor((r+1)f) > floor(rf) moves
// to just after that point: the primary fills past them, the peer (which takes what the primary does not hold, in
// order) gets them first.  Returns the pairs moved; `at` and `fill` get the ranks the log names.
size_t peer_hot_share_order(std::vector<std::pair<int32_t, int32_t>>& profile, size_t& at, size_t& fill) {
    const double f = 0.45;
    at = std::min(profile.size(), (size_t) 8700);
    fill = at;
    if (at == 0) return 0;
    std::vector<std::pair<int32_t, int32_t>> keep, moved;
    for (size_t r = 0; r < at; ++r) {
        const bool to_peer = (int64_t) ((double) (r + 1) * f) > (int64_t) ((double) r * f);
        (to_peer ? moved : keep).push_back(profile[r]);
    }
    const size_t n_moved = moved.size();
    // the primary's share continues with the ranks after `at` until it is full; then the moved ones
    std::vector<std::pair<int32_t, int32_t>> out;
    out.reserve(profile.size());
    out.insert(out.end(), keep.begin(), keep.end());
    fill = std::min(profile.size(), at + n_moved);   // what the primary still takes
    out.insert(out.end(), profile.begin() + (long) at, profile.begin() + (long) fill);
    out.insert(out.end(), moved.begin(), moved.end());
    out.insert(out.end(), profile.begin() + (long) fill, profile.end());
    profile.swap(out);
    return n_moved;
}

}  // namespace basalt::program::engine_local
