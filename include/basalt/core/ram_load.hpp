// include/basalt/core/ram_load.hpp - RAM for the experts of a model's RAM tier: allocation, pinning and sizing.
//
// The bytes are read into it by the loader (loader.hpp).  Lookup of a placed blob stays with the model.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace basalt::core {

/// What a pin got.  A range the driver refuses is pinned slice by slice when it has slice bounds, below the shared-memory
/// budget; what stays unpinned is locked in the working set.
struct RamPin {
    uint64_t registered_bytes = 0;
    bool allocation_backed = false;   ///< pinned by the allocation; unpin_ram must not unregister it
    bool sliced = false;
    std::vector<uint64_t> slice_starts;
    std::vector<const uint8_t*> slice_devices;   ///< independent mapped aliases, when requested
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

/// Owns an expert RAM span. Pinning may come from host VMM, otherwise pin_ram registers it after the read.
/// Ordinary registrations must be removed before reset; allocation-backed pins live until reset.
class RamAllocation {
public:
    RamAllocation() = default;
    ~RamAllocation();
    RamAllocation(RamAllocation&& other) noexcept;
    RamAllocation& operator=(RamAllocation&& other) noexcept;
    RamAllocation(const RamAllocation&) = delete;
    RamAllocation& operator=(const RamAllocation&) = delete;
    /// `pin` permits allocation-backed pinning; ordinary memory is registered separately after the read.
    bool allocate(uint64_t bytes, bool pin = true);
    void reset();
    uint8_t* data() const { return data_; }
    uint64_t bytes() const { return bytes_; }
    uint64_t capacity() const { return capacity_; }
    bool allocation_pinned() const { return allocation_pinned_; }
private:
    uint8_t* data_ = nullptr;
    uint64_t bytes_ = 0, capacity_ = 0;
    bool allocation_pinned_ = false;
};

RamPin pin_ram(const RamAllocation& allocation, const std::vector<uint64_t>& bounds, bool mapped,
               const uint8_t** device = nullptr);
std::vector<RamPin> pin_ram_spans(const std::vector<RamAllocation>& allocations, bool mapped);

/// Remaining portable-registration allowance across the visible adapters. A known exhausted budget is true/0.
bool host_pin_budget(uint64_t& limit, std::string& why);
/// Page-aligned cuts at whole-object boundaries; no object may span two registrations.
std::vector<uint64_t> ram_pin_slices(std::vector<uint64_t> starts, uint64_t end, uint64_t page, uint64_t target);
const uint8_t* ram_pin_alias(const std::vector<uint64_t>& starts, const std::vector<const uint8_t*>& devices, uint64_t off);

/// Kernel memory the GPU driver keeps beside `pinned` bytes of page-locked host memory (its per-page tracking): a plan
/// that pins counts it with them.
inline uint64_t pin_tracking_bytes(uint64_t pinned) { return pinned / 36; }

/// VRAM every card's context keeps for `pinned` bytes of registered host memory (its page tables: 8 bytes a 4 KiB page),
/// whether the context was made before the registration or after it.
inline uint64_t pin_mapping_bytes(uint64_t pinned) { return pinned / 512; }

/// Host memory an engine takes after its RAM tier has read the free RAM: the CUDA runtime's own host memory for the graphs it
/// captures, the first request's buffers, the second card's context.  The tier's plan leaves it free beside the reserve; the
/// serving RAM guard still holds the reserve alone, so the settled free RAM is above the guard and not on it.  0.2.1 left
/// 2 GiB for it (kLateHostBytes); 0.3.1 removed that and a 64 GB box started at or below its guard (#339).
inline constexpr uint64_t kPostPlanHostBytes = 2ull << 30;

/// The reserve a RAM tier is sized against: the free RAM the user asked for (`reserve`, the guard's line) and the
/// allowance for what the engine takes after the sizing.  0 (the guard off) leaves nothing, as asked.
inline uint64_t plan_reserve(uint64_t reserve) { return reserve > 0 ? reserve + kPostPlanHostBytes : 0; }

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
