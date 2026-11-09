// include/basalt/core/ram_load.hpp - reads experts into RAM and pins them, for every model's RAM tier.
//
// A model supplies only a layout: where each blob lands, as runs grouped so one thread reads a group at a time.
// Two layouts are here (ram_arena_runs, ram_layer_runs); a new model picks one or adds its own.  Lookup of a placed
// blob stays with the model.
#pragma once

#include "basalt/core/load_progress.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace basalt::core {

/// Consecutive blobs, adjacent in both the file and RAM.
struct RamRun {
    uint8_t* dst = nullptr;   ///< where they land; null: RamLoad::sink takes them
    uint64_t file_pos = 0;
    uint64_t bytes = 0;
    uint64_t blob_bytes = 0;
    int64_t first = 0;        ///< the first blob's expert id
    int64_t count = 0;
    uint64_t progress = 0;    ///< what the run adds to the bar
};

/// Where each layer's blobs sit in the file.
struct RamGrid {
    int64_t n_layers = 0;
    int64_t n_expert = 0;
    uint64_t file_base = 0;
    std::vector<uint64_t> layer_offset;
    std::vector<uint64_t> blob_bytes;
    uint64_t file_pos(int64_t layer, int64_t expert) const {
        return file_base + layer_offset[(size_t) layer] + (uint64_t) expert * blob_bytes[(size_t) layer];
    }
};

/// Appends a blob to `runs`, extending the last run when it is adjacent in both the file and RAM.
void add_ram_blob(std::vector<RamRun>& runs, uint8_t* dst, uint64_t file_pos, uint64_t blob_bytes, int64_t expert,
                  uint64_t progress);

/// Layout: one arena.  `offsets[layer * n_expert + expert]` is a blob's place, `no_place` for one not kept here;
/// blobs flagged in `no_bar` are not counted by the bar.  One group per layer.
std::vector<std::vector<RamRun>> ram_arena_runs(const RamGrid& grid, uint8_t* base, const std::vector<uint64_t>& offsets,
                                                uint64_t no_place, const std::vector<uint8_t>& no_bar);

/// Layout: one span per layer.  `slots[layer * n_expert + expert]` is a blob's slot in its layer's span, or -1;
/// `sunk` (when not empty) marks the blobs that go to RamLoad::sink.  One group per layer.
std::vector<std::vector<RamRun>> ram_layer_runs(const RamGrid& grid, const std::vector<uint8_t*>& spans,
                                                const std::vector<int32_t>& slots, const std::vector<int32_t>& sunk);

struct RamLoad {
    int fd = -1;              ///< open on the file the runs' positions refer to
    bool unbuffered = true;   ///< read around the OS file cache where the platform can
    bool stage_pinned = false;   ///< sinks copy to a device, so their staging is pinned
    int threads = 1;
    uint64_t chunk = 16ull << 20;
    LoadProgress* progress = nullptr;   ///< the caller finishes it, once the pin is done too
    std::vector<std::vector<RamRun>> groups;

    /// Before a group that will not be read unbuffered.
    std::function<void(int64_t group, const std::vector<RamRun>& runs)> read_ahead;
    /// Places what the unbuffered reads did not (`done` bytes did land); default: reads `fd`.
    std::function<bool(int64_t group, const RamRun& run, uint64_t done, std::string& err)> fallback;
    /// Takes a run with no destination.
    std::function<bool(int64_t group, const RamRun& run, const uint8_t* bytes, std::string& err)> sink;
    /// After a group: `done` groups so far, `copied` bytes so far, whether the fallback was used.
    std::function<bool(int64_t group, int64_t done, uint64_t copied, bool fell_back, std::string& err)> after_group;
};

struct RamLoadStats {
    uint64_t copied = 0;
    uint64_t unbuffered = 0;   ///< of `copied`
    double seconds = 0.0;
};

/// Reads every run on `threads` threads; false with `err` on the first failure.
bool read_ram_runs(const RamLoad& load, RamLoadStats& stats, std::string& err);

/// What a pin got.  A range the driver refuses is pinned slice by slice when it has slice bounds, below the shared-memory
/// budget; what stays unpinned is locked in the working set.
struct RamPin {
    uint64_t registered_bytes = 0;
    bool sliced = false;
    std::vector<uint64_t> slice_starts;
    int slices = 0;
    uint64_t locked_bytes = 0;
    uint64_t lock_off = 0;
    std::string refused;     ///< the driver's answer to the whole range; empty when it took it
    bool capped = false;
    uint64_t cap = 0;
    std::string cap_why;
    std::string lock_note;
};

/// `register_bytes` 0 skips the driver.  `device`: receives the mapped alias, a range with none counting as refused.
RamPin pin_ram(void* base, uint64_t register_bytes, uint64_t lock_bytes, const std::vector<uint64_t>& bounds, bool mapped,
               const uint8_t** device = nullptr);
void unpin_ram(void* base, const RamPin& pin);

/// Kernel memory the GPU driver keeps beside `pinned` bytes of page-locked host memory (its per-page tracking): a plan
/// that pins counts it with them.
inline uint64_t pin_tracking_bytes(uint64_t pinned) { return pinned / 36; }

/// The bytes a RAM tier or copy of experts may take with `avail` bytes available: everything above `reserve`, less the
/// driver's tracking of its pages when it is `pinned`.  Every model sizes its tier with this, after it has allocated
/// (or held the RAM of) the host buffers it takes later, and with the file cache's keep (file_cache_keep) in `reserve`
/// when some experts stay in the file.
inline uint64_t ram_tier_room(uint64_t avail, uint64_t reserve, bool pinned) {
    const uint64_t above = avail > reserve ? avail - reserve : 0;
    return pinned ? above - (above + 36) / 37 : above;
}

/// What the experts a RAM tier leaves in the file read through the OS file cache, which must keep them for as long as
/// they are used: every expert of the model's largest layer (`layer_bytes`) and what one prompt chunk reads from the
/// file beside the layers it computes (`prompt_file_bytes`, 0 when it reads nothing more).
struct FileCache {
    uint64_t layer_bytes = 0;
    uint64_t prompt_file_bytes = 0;
};

/// The least RAM left to the OS file cache while any expert is read from the file.
inline constexpr uint64_t kMinFileCacheBytes = 4ull << 30;

/// The file cache's floor while `file_bytes` of `want_bytes` experts are read from the file: one prompt chunk's reads
/// and the file-held share of two layers (the one computed and the one read ahead), at least kMinFileCacheBytes; 0
/// when nothing is read from the file.  Pages evicted before their layer is over are read again at drive speed, and
/// with swap the reclaim moves the rest of the system out instead.
inline uint64_t file_cache_floor(uint64_t file_bytes, uint64_t want_bytes, const FileCache& c) {
    if (file_bytes == 0) return 0;
    const double share = want_bytes > file_bytes ? (double) file_bytes / (double) want_bytes : 1.0;
    const uint64_t pass = c.prompt_file_bytes + 2 * ((uint64_t) ((double) c.layer_bytes * share) + 1);
    return pass > kMinFileCacheBytes ? pass : kMinFileCacheBytes;
}

/// The RAM a tier sized by ram_tier_room(avail, reserve, pinned) also leaves free, toward `want_bytes` of experts: 0
/// when they all fit, else the file cache's floor for the ones left in the file.  That share depends on the floor, so
/// the iteration starts from all of them in the file and only goes down: the floor it ends on never undercounts.
inline uint64_t file_cache_keep(uint64_t avail, uint64_t reserve, bool pinned, uint64_t want_bytes, const FileCache& c) {
    if (want_bytes <= ram_tier_room(avail, reserve, pinned)) return 0;
    uint64_t file = want_bytes, keep = 0;
    for (int i = 0; i < 16; ++i) {
        keep = file_cache_floor(file, want_bytes, c);
        const uint64_t room = ram_tier_room(avail, reserve + keep, pinned);
        file = want_bytes - (room < want_bytes ? room : want_bytes);
    }
    return keep;
}

/// Memory a pin takes whole large pages of: aligned to and padded to 2 MiB.
uint64_t ram_padded(uint64_t bytes);
uint8_t* alloc_ram(uint64_t bytes);

/// Pins each span, stopping the driver at the shared-memory budget; the rest are locked.
std::vector<RamPin> pin_ram_spans(const std::vector<uint8_t*>& spans, const std::vector<uint64_t>& bytes, bool mapped);

}  // namespace basalt::core
