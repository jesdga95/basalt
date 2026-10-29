// include/basalt/core/peer_experts.hpp - multi-GPU: a second expert tier on another GPU.
//
// The primary GPU (device 0) keeps everything it had: the dense weights, the KV cache, the MTP layer, its own
// adaptive expert tier and the captured verify graph.  This class adds an expert tier on a SECOND card.  From
// the verify graph's point of view nothing changes: for each layer the pool thread decides which routed experts
// the primary computes (its VRAM tier, the PCIe share), and every remaining expert is a "CPU row" the graph reads
// from mapped host memory after the pool raises its flag.  The peer takes the rows whose experts it holds: the
// pool thread launches them on the peer right after publishing the primary's plan, the CPU computes the rest at
// the same time, and the peer's rows are written into the same mapped rows before the pool returns.
//
// The peer computes with the SAME kernels as the primary's hit path (q8_1 activations + native_expert_grouped),
// from the same activation values, so an expert gives bit-identical rows on either card.
//
// The tier follows the conversation like the primary's: `adapt` swaps the most-routed experts that neither card
// holds into the peer's least-used slots (per layer, so a slot keeps its layer's blob size).
#pragma once

#include "basalt/core/expert_cache.hpp"
#include "basalt/core/expert_source.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace basalt::core {

/// multi-GPU: set once at startup when a peer device is configured (--peer-device N).  The mapped host buffers
/// the peer card writes into are then allocated with cudaHostAllocPortable; without a peer the flag stays off.
void set_peer_portable(bool on);
bool peer_portable();

class PeerExperts {
public:
    PeerExperts() = default;
    ~PeerExperts();
    PeerExperts(const PeerExperts&) = delete;
    PeerExperts& operator=(const PeerExperts&) = delete;

    /// Opens the tier on `device`: fills it with the first pairs of `ranked` that `primary` does not hold, as many
    /// as fit in the card's free memory minus `reserve_mib` (`max_slots` > 0 caps the count).  `exclusive`: no
    /// other tier holds these experts, so the fill is reported as model bytes placed, not as a copy.
    bool open(int device, const std::vector<std::pair<int32_t, int32_t>>& ranked, const ExpertCache& primary,
              ExpertSource& src, int64_t n_layers, int64_t n_expert, int reserve_mib, int64_t max_slots,
              std::string& err, bool exclusive = false);
    /// `open` in two steps, for several peers sharing out one ranking: `prepare` takes the card's buffers (no slots),
    /// `fill` sizes the slots for exactly `pick` and copies those experts in.
    bool prepare(int device, ExpertSource& src, int64_t n_layers, int64_t n_expert, std::string& err);
    bool fill(const std::vector<std::pair<int32_t, int32_t>>& pick, std::string& err, bool exclusive = false);
    /// A slot's bytes for an expert of `layer`.
    static uint64_t slot_bytes(int64_t layer);
    void close();
    /// The device bytes `open` allocates besides the expert slots (buffers and kernel scratch), taken before the
    /// slots are sized from what is left.
    static uint64_t device_overhead_bytes();
    bool valid() const { return device_ >= 0; }
    int device() const { return device_; }

    /// Whether the peer holds (layer, expert) right now.
    bool has(int64_t layer, int64_t expert) const { return res_[(size_t) (layer * n_expert_ + expert)] >= 0; }

    /// The device address of (layer, expert)'s blob on the peer, or null when it is not resident.
    const uint8_t* slot_ptr(int64_t layer, int64_t expert) {
        const int32_t sl = res_[(size_t) (layer * n_expert_ + expert)];
        return sl >= 0 ? cache_.device_slot(sl) : nullptr;
    }
    /// DEC_MOE: the residency table ([n_layers * n_expert] slot or -1) and a slot's device address.
    const std::vector<int32_t>& residency() const { return res_; }
    const uint8_t* slot_address(int32_t slot) { return cache_.device_slot(slot); }
    /// The tier's slots: a prompt-time stage on this card reads its experts there and lends the last ones.
    const ExpertCache& cache() const { return cache_; }

    /// A prompt-time stage on this card takes slots `first` and up for one prompt: their experts stop being resident,
    /// and `lent` gets each one's residency index and slot.  No swap may be in flight (apply_pending first).
    bool lend(int32_t first, std::vector<std::pair<int32_t, int32_t>>& lent, std::string& err);
    /// Queues the copies that put the lent slots' experts back, from the expert source, on the refill stream.
    bool give_back(const std::vector<std::pair<int32_t, int32_t>>& lent, std::string& err);
    /// Waits for `give_back`'s copies and makes the experts resident again; `lent` is cleared.
    bool give_back_wait(std::vector<std::pair<int32_t, int32_t>>& lent, std::string& err);
    /// Between requests: every expert leaves its slot (the RAM copy keeps it) and the slots' VRAM goes back to the
    /// driver; `wake` maps it again and copies each expert back into the slot it left.  Not with an exclusive tier (no
    /// RAM copy to come back from).
    bool sleep(std::string& err);
    /// `wake` queues the copies; `wake_wait` waits for them and makes the experts resident again.  With `ready` (the
    /// RAM copy being registered again) the copies run in source address order on a thread of their own, each once
    /// `ready(src, bytes)` says its source is registered, while the caller refills the main card's cache.
    bool wake(std::string& err, std::function<bool(const void*, uint64_t)> ready = nullptr);
    bool wake_wait(std::string& err);
    bool asleep() const { return !asleep_.empty(); }

    /// Starts the peer's share of one layer: the entries i with kind[i] == 2 (x: n_tok rows of n_embd floats).
    /// multi-GPU: with `out` (pinned, portable host rows) the peer writes its rows straight into
    /// out[entry] with a scatter kernel (no D2H copy + host memcpy in finish).
    bool launch(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k, const int32_t* kind,
                std::string& err, float* out = nullptr);
    /// Waits for the launched share and writes its rows into `out` (row i = entry i, n_embd floats).
    bool finish(float* out, std::string& err);

    /// Adaptive tier: swap up to `max_swaps` experts routed at least twice that neither the primary (`res0`, slot
    /// or -1) nor the peer holds into the peer's least-used slots.  Copies run on the peer's refill stream; a
    /// swapped-in expert becomes resident at `apply_pending`.  Call only between verify windows.
    bool adapt(const float* usage, const int32_t* res0, int max_swaps, std::string& err);
    void apply_pending(bool wait);
    /// The first half of `apply_pending`: once the evictions' copies landed, queue the incoming experts' copies and
    /// return without waiting for them (`apply_pending` then lands them).  False when `wait` is off and they are not done.
    bool issue_pending(bool wait);
    bool has_pending() const { return !pending_.empty() || !evicting_.empty(); }
    /// The residency indices (layer * n_expert + expert) of the experts the peer is taking in: the last round's picks
    /// and any refill still landing.
    std::vector<size_t> incoming() const;
    /// --peer-exclusive, once the RAM copy is built: the peer's experts have no RAM place.  Swaps then keep every
    /// expert somewhere: an evicted one goes to a free RAM place before its slot is reused, and a RAM place is freed
    /// once its expert is resident on the peer.  Nothing is read from the file for them.
    void set_exclusive(FileExpertSource* fsrc) { fsrc_ = fsrc; }
    bool exclusive() const { return fsrc_ != nullptr; }
    /// Keeps (layer, expert) in its slot (adapt does not evict it) until `reassign` takes the slot: its bytes are
    /// being copied out.
    void hold(int64_t layer, int64_t expert) { hold_[(size_t) (layer * n_expert_ + expert)] = 1; }
    /// Copies (layer, expert)'s bytes from the peer's slot into `dst` on device 0 (queued on `stream`).
    bool copy_to(int64_t layer, int64_t expert, uint8_t* dst, cudaStream_t stream, std::string& err);
    /// PEER_PROMOTE: `in` moved to the primary, which gave up `out` (`blob`: its bytes in host memory).  The
    /// slot `in` held takes `out`, resident at `apply_pending`.  Call only between verify windows.
    bool reassign(int64_t layer, int32_t in, int32_t out, const uint8_t* blob, std::string& err);

    int64_t resident() const { return resident_; }
    /// the last launch writes its rows into the host rows itself (the caller must not touch them)
    bool launched_direct() const { return launched_direct_; }
    double gib() const { return cache_.gib(); }
    int64_t entries() const { return entries_; }      ///< routed entries the peer computed
    int64_t experts() const { return experts_; }      ///< distinct (layer, expert) the peer computed
    int64_t swaps() const { return swaps_; }
    double ms_wait = 0;                               ///< host time spent in finish() waiting for the peer

private:
    int device_ = -1;
    int64_t n_layers_ = 0, n_expert_ = 0;
    ExpertCache cache_;
    ExpertSource* src_ = nullptr;
    std::vector<int32_t> res_;                        ///< [n_layers * n_expert] -> peer slot or -1
    std::vector<std::pair<int32_t, int32_t>> pending_;   ///< (residency index, slot) once the copies land
    FileExpertSource* fsrc_ = nullptr;                ///< set with --peer-exclusive
    std::vector<std::pair<int32_t, int32_t>> asleep_; ///< (residency index, slot) that `sleep` took out
    std::thread wake_thread_;                          ///< `wake` with `ready`: its copies
    bool wake_failed_ = false;
    std::vector<uint8_t> hold_;                       ///< [n_layers * n_expert]: not to be evicted
    /// `off` is the RAM place the victim is copied to, or `kHasPlace` when the victim already has one
    static constexpr uint64_t kHasPlace = ~0ull;
    struct Evict { int32_t layer, in, out, slot; uint64_t off; };
    std::vector<Evict> evicting_;                     ///< exclusive swaps whose victim is still being copied to RAM
    cudaEvent_t evict_ev_ = nullptr;
    int64_t resident_ = 0, entries_ = 0, experts_ = 0, swaps_ = 0;
    cudaStream_t stream_ = nullptr, refill_ = nullptr;
    cudaEvent_t refill_ev_ = nullptr;
    float* h_x_ = nullptr;   float* d_x_ = nullptr;   // n_tok x n_embd
    float* h_out_ = nullptr; float* d_out_ = nullptr; // compact rows
    void* h_meta_ = nullptr; void* d_meta_ = nullptr; // groups: ptr | start | dst | tok | count
    uint8_t* d_q8_ = nullptr;
    void* d_scratch_ = nullptr;
    std::vector<int32_t> row_of_;                     ///< compact row -> original entry
    int64_t launched_rows_ = 0;
    bool launched_direct_ = false;
};

}  // namespace basalt::core
