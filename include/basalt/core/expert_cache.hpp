// include/basalt/core/expert_cache.hpp - the VRAM-resident expert tier.
//
// **WHY THIS EXISTS.**  The CPU expert pool is the engine's largest single cost: every token reads hundreds of MB of
// expert bytes, and the work is exposed because the residual chain is strictly serial.  The only way past it is to
// stop reading those bytes: put some of the 48 x 512 experts in VRAM and let the GPU compute them, so the CPU reads
// only the misses.
//
// **THE DESIGN IS SIZED AGAINST MEASUREMENTS.**  The expert hit rate is a held-out figure, the VRAM is what is free on a live run rather than at startup, and only a small share of
// (layer, token) pairs have all ten experts resident - so this cannot be a per-layer grouped kernel and the three-way
// split is not an optimisation.
//
// **WHAT THIS FILE IS.**  The SLOT STORAGE and the RESIDENCY TABLE: it allocates the VRAM, fills it from the host
// arena, and answers `(layer, expert) -> slot or -1`.  It computes nothing itself: the GPU computes the experts it
// holds in the verify window's grouped kernels and the prompt path (the hit/miss split), and the CPU pool the rest.
#pragma once

#include "basalt/core/model_file.hpp"
#include <cstddef>
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace basalt::core {

/// Where one expert's VRAM lives, or -1.  `int32_t` and not a bool because the slot index is what the kernel
/// needs, and a table that has to be re-derived from a bitmask is a table that will be re-derived differently
/// in the two places that use it.
inline constexpr int32_t kNotResident = -1;

/// The expert profile, the model's `profile` section: `version, n_layers, n_expert, slots, n_ranked` (uint32,
/// little-endian), then `n_ranked` `(layer, expert)` pairs (uint16 each) ranked by descending routing frequency, then
/// an `n_layers x n_expert` int32 table of each pair's rank (-1: not ranked).  `--expert-profile-save` writes the
/// same bytes.  A profile ranked by frequency over a whole trace beats filling the tier with the first distinct
/// experts a run routes.
///
/// The frequency table is not returned: the engine measures the hit rate by running rather than by re-deriving what
/// the file claims.
bool read_expert_profile(const FileRange& range, int64_t n_layers, int64_t n_expert,
                         std::vector<std::pair<int32_t, int32_t>>& ranked, int64_t& slots, std::string& err);

/// --expert-profile-save: what the adaptive tier learned, as a profile ranking EVERY (layer, expert) pair - the
/// experts resident in VRAM now first (where the swaps left the cache), then the rest; within each, by `heat` (the
/// routing the adaptive tier counted since the start, descending), then by `prior` (the profile the engine started
/// from: an expert this run never routed keeps its old place), then by index.  A start from it begins where this one
/// ended; one with more slots adds the hottest of the rest, one with fewer keeps the hottest.  `resident` and `heat`:
/// n_layers x n_expert entries (`heat` may be empty: no counts, the prior decides).
std::vector<std::pair<int32_t, int32_t>> rank_learned_profile(int64_t n_layers, int64_t n_expert,
                                                              const std::vector<uint8_t>& resident,
                                                              const std::vector<double>& heat,
                                                              const std::vector<std::pair<int32_t, int32_t>>& prior);

/// Writes `ranked` in the profile format: `1, n_layers, n_expert, n_ranked, n_ranked` (uint32), the pairs (uint16
/// layer, uint16 expert), then the n_layers x n_expert int32 table of each pair's rank (-1: not ranked).  Atomically: a `<path>.tmp` beside it, renamed over `path` once complete, so a reader (the
/// next start) never sees half a file.  False with `err` when it could not be written.
bool write_expert_profile(const std::string& path, int64_t n_layers, int64_t n_expert,
                          const std::vector<std::pair<int32_t, int32_t>>& ranked, std::string& err);

class ExpertCache {
public:
    ExpertCache() = default;
    ~ExpertCache();
    ExpertCache(const ExpertCache&) = delete;
    ExpertCache& operator=(const ExpertCache&) = delete;

    /// Allocates `n_slots * blob_bytes` of DEVICE memory and a `n_layers * n_expert` residency table, and
    /// **CHECKS THE ALLOCATION AGAINST WHAT THE CARD ACTUALLY HAS** rather than assuming the planner was right.
    /// A cache that silently allocates less than it was asked for would report a hit rate for slots it does
    /// not have, which is the shape of error this project keeps paying for.
    bool open(int64_t n_slots, int64_t n_layers, int64_t n_expert, int64_t blob_bytes, std::string& err);
    /// slots of the given sizes, back to back (a native pack's blobs differ per layer, and a profile-filled tier never
    /// moves an expert to another layer's slot, so each slot keeps its first size).
    bool open_sized(const std::vector<int64_t>& slot_bytes, int64_t n_layers, int64_t n_expert, std::string& err);
    /// Byte offset of each slot in the arena (null for uniform slots).
    const uint64_t* slot_offsets() const { return off_.empty() ? nullptr : off_.data(); }
    void close();

    bool valid() const { return base_ != nullptr; }
    /// The slots usable now: all of them, unless `shrink` gave the tail's VRAM back - then the ones wholly inside the
    /// VRAM still backed, always a prefix.  The prompt path's loan (the last slots) and every report follow this;
    /// `full_slots()` is what `open` allocated.
    int64_t slots() const { return live_slots_; }
    int64_t full_slots() const { return slots_; }
    /// Slots actually claimed.  Not the same as `slots()` - the cache does not evict, so a run that routes
    /// fewer distinct experts than there are slots leaves the rest empty.
    int64_t resident() const { return per_layer_ ? admitted_ : next_free_; }
    /// The bytes of `slots()` (all of the arena unless shrunk); `full_bytes()`: of every slot.
    int64_t bytes() const { return slot_end(live_slots_); }
    int64_t full_bytes() const { return slot_end(slots_); }
    double gib() const { return (double) bytes() / 1073741824.0; }

    /// **HOT VRAM RESIZE, OPT-IN (--vram-elastic).**  With a segment size set before `open`/`open_sized`, the arena is
    /// one reserved address range backed by physical segments of that size (CUDA virtual memory management) instead of
    /// one cudaMalloc, so `shrink` can give the tail's VRAM back to the driver - for another program - and `grow` can
    /// take it again, while every slot keeps its address (the captured graphs, the verify plan's pointers and the
    /// prompt path's loan all hold addresses into it).  0 (the default): one cudaMalloc.  The slots past `slots()`
    /// after a shrink must hold no resident expert and no loan: the CALLER evicts them first (they become CPU misses).
    void set_segment_bytes(int64_t seg_bytes) { seg_req_ = seg_bytes > 0 ? seg_bytes : 0; }
    /// **THE ELASTIC TAIL.**  With `tail_bytes` set before `open`/`open_sized` (and no segment size), the arena is
    /// segmented too: large segments up to the last `tail_bytes`, then segments of the driver's granularity, so the
    /// tail can be lent to the elastic KV pools one granule at a time (`shrink`/`grow`, same rules as above).
    void set_elastic_tail(int64_t tail_bytes) { tail_req_ = tail_bytes > 0 ? tail_bytes : 0; }
    /// The bytes of the granule-sized tail (0 when there is none).
    int64_t elastic_bytes() const;
    bool segmented() const { return !segs_.empty(); }
    int64_t segment_bytes() const { return seg_; }
    /// Bytes of the arena backed by VRAM now (a whole number of segments; the arena when not segmented).
    int64_t mapped_bytes() const;
    /// Unmaps every segment past the first `keep_bytes` (rounded UP to a segment boundary): `slots()` becomes the
    /// slots wholly inside what stays.  Waits for the device first.  False with `err` when not segmented or a driver
    /// call failed.
    bool shrink(int64_t keep_bytes, std::string& err);
    /// `slots()` after `shrink(keep_bytes)`, without shrinking.
    int64_t slots_after_shrink(int64_t keep_bytes) const;
    /// Maps segments again up to `want_bytes` (rounded DOWN to a segment, at most the arena; the last, shorter
    /// segment only when `want_bytes` covers the arena).  Stops at the first segment the driver cannot back (false,
    /// `err`: what was mapped by then stays).  The new slots are empty until the caller fills them.
    bool grow(int64_t want_bytes, std::string& err);
    /// The number of leading slots that fit wholly inside the first `bytes` bytes.
    int64_t slots_within(int64_t bytes) const;
    /// The bytes the first `n` slots span.
    int64_t bytes_of(int64_t n) const { return slot_end(n < 0 ? 0 : n > slots_ ? slots_ : n); }

    /// `(layer, expert)` -> slot index, or `kNotResident`.  Bounds-checked: a bad layer or expert returns
    /// `kNotResident` rather than reading whatever is adjacent in the table.
    int32_t slot_of(int64_t layer, int64_t expert) const;
    /// Claims the next free slot for `(layer, expert)`.  Returns the slot, or `kNotResident` when the cache is
    /// full - **it never evicts**, because an eviction policy has to be chosen from measurements, and a placeholder policy would set the hit
    /// rate that everything downstream is then sized against.
    int32_t admit(int64_t layer, int64_t expert);

    /// Publish a same-layer replacement after its slot copy has completed.
    void replace(int64_t layer, int32_t old_expert, int32_t new_expert) {
        auto& old = residency_[(size_t) layer * n_expert_ + old_expert];
        residency_[(size_t) layer * n_expert_ + new_expert] = old;
        old = kNotResident;
    }

    /// **GIVE EACH LAYER ITS OWN SLOTS.**
    ///
    /// `admit` hands out slots in ARRIVAL ORDER from one counter shared by every layer
    /// (`if (next_free_ >= slots_) return kNotResident;`).  A position looks at 10 experts in each layer, so
    /// the first `n_slots` distinct (layer, expert) pairs cover only the first few layers of POSITION 0 -
    /// the cache fills entirely inside the first position it ever sees and never changes again, so hits are
    /// confined to those few layers for the rest of the run.
    ///
    /// Spreading the slots evenly over layers is worth far more than letting a few layers take them all.
    ///
    /// With this on, layer `l` may only use slots `[l*q, (l+1)*q)` where `q = slots()/n_layers`, the last
    /// layer taking the remainder.  **EVICTION IS STILL ABSENT** - the property the comment above calls
    /// deliberate is preserved exactly; only WHICH slot a layer may take changes.  Default off, so every
    /// caller that predates this flag is bit-for-bit unaffected.
    void set_per_layer_admission(bool on) { per_layer_ = on; }
    bool per_layer_admission() const { return per_layer_; }
    /// The slot range layer `l` may admit into under per-layer admission.  Exposed so a test can check it.
    void layer_slot_range(int64_t layer, int64_t& lo, int64_t& hi) const;

    /// The device address of one slot.
    uint8_t* device_slot(int32_t slot);
    const uint8_t* device_slot(int32_t slot) const;

    /// Copies `(layer, expert)`'s blob from `host_blob` into `slot` on `stream`.  Asynchronous: the caller orders it.
    /// Returns false if the indices are out of range rather than reading past the arena.
    /// `bytes`: the blob's own size when it is smaller than the slot (a native pack); 0 = the slot.
    bool fill_slot(int32_t slot, const uint8_t* host_blob, void* stream, std::string& err, int64_t bytes = 0);

    /// **THE SAME COPY, BUT BLOCKING, AND THE PROFILE FILL NEEDS IT.**  `fill_slot` is asynchronous because on
    /// the token path the fill and the kernel that reads it are ordered by one stream and waiting would be a
    /// synchronisation per layer.  At STARTUP there is no such ordering to lean on: the source is pageable host
    /// memory, `verify_slot` reads the slot back on the LEGACY stream, and a legacy-stream copy is not ordered
    /// against a `cudaStreamNonBlocking` one.  The first version used the async form and `verify_slot` refused
    /// the whole run with "slot 0 differs from the arena at byte 0" - which is the check doing its job.
    bool fill_slot_blocking(int32_t slot, const uint8_t* host_blob, std::string& err, int64_t bytes = 0);
    /// `fill_slot_blocking`'s copy on the same (legacy) stream, but queued: many slots are refilled with one
    /// `sync_queued` at the end instead of a wait per slot. Same ordering against earlier work, same bytes.
    bool fill_slot_queued(int32_t slot, const uint8_t* host_blob, std::string& err, int64_t bytes = 0);
    bool sync_queued(std::string& err);

    /// Reads `slot` back to the host and compares it to `host_blob`, byte for byte.  **THE ONLY THING THAT SAYS
    /// THE CACHE HOLDS THE EXPERT IT CLAIMS TO.**  A slot table that is right about indices and wrong about
    /// bytes produces a plausible token, which is exactly the failure this project has paid for most often.
    bool verify_slot(int32_t slot, const uint8_t* host_blob, std::string& err, int64_t bytes = 0);

    /// Slots filled so far, for the startup report.
    int64_t fills() const { return fills_.load(); }

private:
    int64_t slot_end(int64_t n) const { return off_.empty() ? n * blob_ : (int64_t) off_[(size_t) n]; }
    bool open_segmented(uint64_t want, std::string& err);
    void release_segmented();
    uint8_t* base_ = nullptr;
    int64_t live_slots_ = 0;   // slots() - all of them unless shrunk
    int64_t seg_req_ = 0;   // the segment size asked for (0: one cudaMalloc)
    int64_t seg_ = 0;   // the segment size used (a multiple of the driver's granularity)
    uint64_t reserved_ = 0;   // the reserved address range's size
    std::vector<unsigned long long> segs_;   // each segment's physical handle (0: unmapped)
    std::vector<int64_t> seg_size_;   // each segment's size (the last one may be shorter)
    std::vector<int64_t> seg_off_;    // each segment's offset in the arena
    int64_t tail_req_ = 0;   // the elastic tail asked for (0: none)
    int64_t head_end_ = 0;   // where the granule-sized tail segments start
    int64_t mapped_segs_ = 0;   // segments [0, mapped_segs_) are backed
    std::vector<int32_t> residency_;   ///< [n_layers * n_expert] -> slot or kNotResident
    int64_t slots_ = 0;
    int64_t n_layers_ = 0;
    int64_t n_expert_ = 0;
    int64_t blob_ = 0;
    int64_t next_free_ = 0;
    std::atomic<int64_t> fills_{0};
    /// `per_layer_` off (the default) leaves `next_free_` as the only admission counter.
    bool per_layer_ = false;
    std::vector<int32_t> layer_next_;   ///< [n_layers] -> that layer's next free slot
    std::vector<uint64_t> off_;   // slot offsets (slots + 1 entries) when sized
    int64_t admitted_ = 0;
};

}  // namespace basalt::core
