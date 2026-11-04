// include/basalt/core/pinned.hpp - the host arena the expert weights live in.
//
// The expert weights are too large for a card, so they stay in host memory and are streamed.  That makes this arena
// the engine's real working set: it must be PAGE-LOCKED for the copy engine to reach full bandwidth, and it must say
// which backing it got, because the two options differ by more than twice in TLB reach:
//
//   * LARGE PAGES (2 MB) - the OS's huge-page allocation (hugetlb on Linux, large pages on Windows).  Tens of GB at 4 KB pages is millions of TLB entries,
//     which does not fit in any TLB, so every block of every expert matvec takes TLB misses.
//   * NORMAL PAGES - the fallback.  Correct, slower, and it must be REPORTED rather than silently accepted: "the
//     engine adapts to the machine it is on" is only true if the engine says what it got.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace basalt::core {

enum class PageBacking { LargePages, NormalPages, PinnedByCuda };


struct PinnedArena {
    void* base = nullptr;
    uint64_t capacity = 0;
    PageBacking backing = PageBacking::NormalPages;
    std::string note;              // why the backing is what it is, for the startup print
    uint64_t locked_bytes = 0;     // resident via the working-set lock when CUDA could not pin it
    /// when the whole arena cannot be registered, it is registered in `slice`-byte pieces from the start; this is the
    /// pinned prefix (a copy that stays inside one slice can then DMA straight from the arena).
    uint64_t registered_bytes = 0;
    uint64_t slice_bytes = 0;
    int registered_slices = 0;

    PinnedArena() = default;
    /// `slice`: the piece size for the per-slice registration fallback (0 = none).
    explicit PinnedArena(uint64_t bytes, uint64_t slice = 0);
    /// slices of different sizes (one per layer of a native pack), given as their start offsets followed by the end of
    /// the last one.  `slice_starts` holds the registered ones.
    /// `shared_file`: use a file-backed MAP_SHARED mapping instead of anonymous memory.
    /// `shared_hash` identifies the model that is allowed to populate that backing.  The file carries a small
    /// header and is refused when its stored hash does not match.  Empty `shared_file` preserves the existing
    /// allocation path.  Population/coordination and backing-file lifetime remain the caller's job.
    PinnedArena(uint64_t bytes, const std::vector<uint64_t>& bounds,
                const std::string& shared_file = {}, uint64_t shared_hash = 0);
    std::vector<uint64_t> slice_starts;
    void* mapping_base = nullptr;     ///< actual mapping start; differs from base when a shared-file header exists
    uint64_t mapping_bytes = 0;       ///< bytes to release from mapping_base
    ~PinnedArena();
    PinnedArena(const PinnedArena&) = delete;
    PinnedArena& operator=(const PinnedArena&) = delete;

    bool valid() const { return base != nullptr; }
    uint8_t* data() const { return (uint8_t*) base; }
};

// FNV-1a 64.  Per layer, so a corrupt or short read names WHICH layer rather than just failing a whole-file
// comparison.
uint64_t fnv1a64(const uint8_t* p, uint64_t n, uint64_t seed = 1469598103934665603ull);

// ---- the expert stream -----------------------------------------------------
//
// THIS IS THE NUMBER THE OBJECTIVE DEPENDS ON.  With one expert per stream, the design's own arithmetic is:
//
//     per token: E = the expert weights routed, of which only the MISSES cross the bus
//     miss fraction (1 - h) x E / measured bandwidth = milliseconds per token
//
// So `h` (the VRAM hit rate) is only worth anything if a per-expert transfer can actually
// reach the bus's bandwidth.  One expert's copy is SMALL, and if small copies run at a fraction of large
// ones then the architecture's central constant is wrong and no kernel optimisation can fix it.  That is what
// this measures: the same total bytes at several granularities, so the difference is visible.
struct StreamStats {
    double seconds = 0.0;
    uint64_t bytes = 0;
    uint64_t chunk = 0;
    double gib_per_second() const { return seconds > 0 ? (double) bytes / (1024.0 * 1024 * 1024) / seconds : 0.0; }
};

// Copy `bytes` from pinned host memory at `src` to a device buffer of `chunk` bytes, in `chunk`-sized
// transfers, `iters` times.  The destination is REUSED, so this measures the bus and not allocation.
StreamStats stream_bandwidth(const uint8_t* src, uint64_t bytes, uint64_t chunk, int iters);

}  // namespace basalt::core
