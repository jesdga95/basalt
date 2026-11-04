#pragma once

#include "basalt/prefill/prefill.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/prefill/gemm.hpp"
#include <algorithm>
#include <chrono>
#include <string>

namespace basalt::prefill::detail {

using Clock = std::chrono::steady_clock;
constexpr float EPS = 1e-6f;
constexpr int64_t N = 2560, HC = 4, D = N * HC, LR = 320, K = 10, NE = 512;
constexpr int64_t C = 10240, ZV = 6144, HV = 48;
// staging holds the largest blob of the pack (a native pack's blobs differ per layer)
inline int64_t MAXBLOB() { return (int64_t) basalt::kernels::cpu::expert_layout().max_blob; }
// host->device expert staging ring of a chunk below stream_all_min(): two groups of MMQ_GROUP experts, so a slot is
// copied into again only once the group that read it has run
constexpr int STAGE = 32;
// Such a chunk also holds, per layer parity, slots for the next layer's experts copied ahead of its routing, after the
// STAGE slots: about as many as the link moves while a layer's attention half runs, which grows with the chunk.
constexpr int kAheadSlots = 64;
inline int ahead_slots(size_t T) { return (int) std::min<size_t>(kAheadSlots, T / 32); }
// From the chunk size stream_all_min() on, every non-resident expert of every layer streams in a fixed order through a
// ring_slots()-slot ring (nearly every expert is routed at such a chunk), so the copy engine keeps working through the
// attention halves instead of waiting for each layer's routing.
constexpr int RING_MAX = 384;           // the arrays; the ring itself is ring_slots()
// The chunk size from which every expert streams: below it only the routed experts are staged.  Staging every
// non-resident expert costs the same whatever the chunk, so it pays only once most of them are routed anyway.
extern int64_t g_stream_all_min;
inline int64_t stream_all_min() { return g_stream_all_min; }
// The streamed ring's slots (one whole expert blob each, the same for every pack).  g_ring_depth caps the depth the
// next layout is laid out with (a prompt read beside decoding slots lends fewer cache slots with a shallower ring).
constexpr int kPrefillRing = 256;
// From kDeepRingTokens tokens on, a layer's attention half outlasts the copies a kPrefillRing-slot ring holds: the copy
// engine stops until the layer's experts start, and their walk then waits for the layer's last copies.  Such a chunk
// streams through a deeper ring (its slots come out of the same loan).
constexpr int kPrefillRingDeep = 384;
constexpr int64_t kDeepRingTokens = 16384;
extern int g_ring_depth;
bool fused_ring();
// (a prompt peer's streamed share and the fused kernels' batches follow the ring's size: those keep 256)
inline int ring_slots(size_t T) {
    if ((int64_t) T < stream_all_min()) return std::min(g_ring_depth, STAGE + 2 * ahead_slots(T));
    const bool deep = (int64_t) T >= kDeepRingTokens && !fused_ring();
    return std::min(g_ring_depth, deep ? kPrefillRingDeep : kPrefillRing);
}
constexpr int DQ = 2;              // dequantized-expert ring (FP16 gate/up + down)
inline double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

void up_mix(Gemm& gm, const uint16_t* w_up, const uint16_t* lo16, float* gated, const float* R, const float* rs,
            const float* w_norm, float* mixed, uint16_t* mixed16, uint16_t* mixed_h, int64_t T, void* stream);
const core::WeightRef* need(const core::LayerView& v, const char* suffix, std::string& err);
bool native_proj(Gemm& gm, const core::WeightRef* w, const uint16_t* X, float* Y, int64_t T, const std::string& name,
                 std::string& err, int64_t ldy = 0);
bool bf16_proj(Gemm& gm, const core::WeightRef* w, const uint16_t* X, float* Y, int64_t T, const std::string& name,
               std::string& err, int64_t ld = 0);
extern double g_pinned_share;
}  // namespace basalt::prefill::detail
