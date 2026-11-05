// basalt/models/qwen4exp/decode/expert_dispatch.hpp - the pool's side of a layer: its state and its entry points.
#pragma once

#include "basalt/core/expert_source.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/models/qwen4exp/geometry.hpp"

namespace basalt::core {

class PeerExperts;

/// The adapter's own state.  One per session, reused every layer so the token path allocates nothing.
struct ExpertDispatch {
    basalt::kernels::cpu::ExpertPool* pool = nullptr;
    ExpertSource* src = nullptr;
    int64_t n_expert = basalt::kernels::cpu::NE;

    /// Counters, for the driver to report rather than for control flow.
    int64_t layers = 0;
    int64_t experts = 0;
    int64_t missing = 0;

    // ================================ THE VRAM TIER ================================
    //
    // **THE CACHE IS CONSULTED AND FILLED HERE.**  The engine reports the hit rate `h` on its own routing,
    // on a real workload.
    //
    // The policy is compulsory-miss: the first time `(layer, expert)` is routed, if a slot is free it is
    // admitted and filled from the arena.  No eviction: an eviction policy has to be chosen from
    // measurements, and a placeholder would set the hit rate everything is sized against.
    ExpertCache* cache = nullptr;
    int64_t cache_hits = 0;      ///< lookups already resident
    int64_t cache_admitted = 0;  ///< lookups that took a slot
    int64_t cache_refused = 0;   ///< lookups with no slot free (the cache is full)
    void* cache_stream = nullptr;
    const char* cache_fail = nullptr;

    // ================================ THE HITS GO TO THE GPU ================================
    //
    // **THE SPLIT IS BY ROUTER INDEX.**  A layer routes ten experts; the resident ones are computed on the GPU and the
    // rest on the CPU, and both answers have to end up in `parts` at the index the ROUTER gave them, because that is
    // the order `moe_combine` weights against.  So the CPU's `out` rows are zeroed for the hits, and each hit carries
    // its routed index to the kernel as `dst`.
    //
    // Everything below is per-session state rather than per-call, so the token path allocates nothing.
    const uint8_t* cache_base = nullptr;   ///< the slot arena on the DEVICE
    int64_t cache_blob = 0;                ///< bytes per slot
    const uint64_t* cache_slot_off = nullptr;   // per-slot offsets when the slots differ in size
    void* hit_scratch = nullptr;           ///< `moe_hit_grouped_scratch_bytes(K, ...)`
    float* parts_out = nullptr;            ///< the graph's `parts` buffer, on the device
    /// Where the GPU's hits land, `K x n_embd`, DEVICE and separate from `parts_out` on purpose: see
    /// `HitPhase`.  Zeroed by the hit path each layer before the kernel writes it.
    float* hit_out = nullptr;
    int64_t parts_elems = 0;               ///< `K * n_embd`, the length of both buffers
    const float* mixed = nullptr;          ///< the layer's normed activation, for the hit kernel's Q8_0
    uint8_t* x_q8_0_hit = nullptr;         ///< `(n_embd/32) * 34` bytes, its own buffer
    /// **THE CPU's fp32 ACTIVATION SCALES, `(n_embd/32)` FLOATS.**  Without them the GPU's hits are
    /// computed with the `block_q8_0`'s fp16 `d` while the CPU's misses use the fp32 `ActQ::scale`
    /// (a relative difference of about 2^-11 per chunk), so a hit and a miss would give different tokens.
    /// Required whenever the hit path runs.
    float* x_q8_0_hit_scale = nullptr;
    int32_t* d_slot = nullptr;             ///< device, K entries
    int32_t* d_dst = nullptr;              ///< device, K entries
    std::vector<int32_t> h_slot, h_dst;    ///< host staging, sized at session setup
    /// **PER ROUTER INDEX, DECIDED IN `Launch` AND CONSUMED BY THE POOL.**  The two callbacks share it
    /// so the decision is made exactly once, on this layer's ids, and neither side can re-decide it.
    std::vector<uint8_t> is_hit;
    bool decided = false;
    /// Token graph: the STATIC residency table (`n_layers x n_expert`, slot or -1), the host's copy of what the device
    /// hit path reads.  When set, the pool leaves a resident expert's row at zero (the GPU computes it) without any
    /// `Launch` callback.
    const int32_t* host_res = nullptr;

    // ================================ DID THE GPU ACTUALLY START? ================================
    //
    // The hit path only overlaps with the CPU if the enqueued hit kernels begin before the host next enters the
    // driver, i.e. before `pool()` returns; that is measured rather than assumed.
    //
    // `hit_done` is recorded on the stream straight after the hit kernel.  `Combine` - which runs after the pool -
    // queries it: SUCCESS means the GPU finished while the CPU was working, NOT-READY means it had not.  Same shape as
    // the doorbell's `rings_mid_graph`, and for the same reason.
    void* hit_done = nullptr;      ///< cudaEvent_t, created at session setup
    int64_t hit_ready = 0;         ///< layers where the hit work was DONE by the time the pool returned
    int64_t hit_late = 0;          ///< layers where it was not
    int64_t n_hits = 0;                    ///< this layer's hits
    /// Set by `Launch` and consumed by `Combine`, so a `Combine` with no `Launch` in front of it cannot
    /// add a stale buffer into `parts`.
    bool hit_pending = false;
    const char* hit_fail = nullptr;

    /// Whether the hit path is wired up.  All of it or none of it: a half-configured hit path would compute
    /// some experts twice and others not at all, which is a wrong token rather than an error.
    bool hits_ready() const {
        return cache != nullptr && cache_base != nullptr && parts_out != nullptr && hit_out != nullptr &&
               mixed != nullptr &&
               hit_scratch != nullptr && x_q8_0_hit != nullptr && x_q8_0_hit_scale != nullptr && d_slot != nullptr && d_dst != nullptr;
    }

    basalt::kernels::cpu::ActQ act;
    /// Verify window: one quantized activation per token, the multi-token jobs, and the expert -> job map (reset after
    /// every layer).
    std::vector<basalt::kernels::cpu::ActQ> act_multi;
    /// a native pack's per-token activations (MAXT x kNativeActBytes).
    std::vector<uint8_t> nact_multi;
    std::vector<basalt::kernels::cpu::ExpertJobMulti> jobs_multi;
    std::vector<int16_t> job_of;
    /// The verify window's GPU plan (VRAM hits + the PCIe share of the misses); `pcie_num`/256 of each layer's distinct
    /// missed experts (the last ones in routing order) are read by the GPU over PCIe.
    GpuPlanSink* plan = nullptr;
    int pcie_num = 0;
    int64_t pcie_experts = 0;      ///< distinct experts the GPU read over PCIe in verify windows
    /// routed (token, expert) entries the GPU computed from outside its cache in verify windows: read over PCIe
    /// (the PCIe share, kind 1) or on another GPU (kind 2).  In neither cache_hits nor cache_refused.
    int64_t offload_entries = 0;
    double ms_plan = 0, ms_actq = 0, ms_jobs = 0, ms_run = 0;   ///< verify-window dispatch sections
    /// Decayed routing counts per (layer, expert) during decode (sized by the caller; empty = off), which the driver
    /// uses to swap the most-routed missing experts into the VRAM tier between rounds.
    std::vector<float> usage;
    int64_t multi_misses = 0;      ///< distinct (layer, expert) pairs the CPU computed in verify windows
    int64_t multi_entries = 0;     ///< routed (token, expert) entries the CPU served in verify windows
    /// Multi-GPU: the second GPU's tier.  Its experts are computed there instead of on the CPU (kind 2).
    PeerExperts* peer = nullptr;
    int64_t peer_entries = 0;      ///< routed entries the peer served in verify windows
    /// Set when `dispatch` could not produce an answer.  The loop itself has no error channel, so this is
    /// where a source failure surfaces: the driver checks it after `session_loop` returns rather than the
    /// engine computing from a half-filled `parts`.
    ///
    /// **`failed` LATCHES AND `fail` IS ONLY THE MESSAGE.**  Clearing the message must not re-arm the adapter:
    /// a session that failed at layer 5 has already fed `moe_combine` whatever `parts` held, so every layer
    /// after it is built on a hole - and resuming into a plausible-looking token is the exact outcome this
    /// whole mechanism exists to prevent.  `failed` is what the short-circuit reads.
    bool failed = false;
    const char* fail = nullptr;
    int64_t fail_layer = -1;
    int64_t fail_expert = -1;
};

/// `basalt::core::PoolFn`, exactly.  Silent on failure BY SIGNATURE - see `ExpertDispatch::fail`.
void expert_pool_dispatch(void* user, const float* x_f, const int32_t* ids, const float* weights, int64_t n_embd,
                          int64_t k, float* out);

/// The pool for a verify window of `n_tok` tokens.  `x_f` is (n_tok, n_embd), `ids` (n_tok, k) and `out`
/// (n_tok * k, n_embd).  Each distinct missed expert is computed once for all the tokens routed to it; resident
/// experts' rows are zeroed (the GPU adds them).  Requires `host_res` (the token-graph residency).
void expert_pool_dispatch_multi(ExpertDispatch& d, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k,
                                float* out);

}  // namespace basalt::core
