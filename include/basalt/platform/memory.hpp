// include/basalt/platform/memory.hpp - keep a large host region resident.
//
// An unlocked expert arena is trimmed under memory pressure, which makes the CPU pool's speed vary between
// runs. With the n-gram table out of RAM there is headroom to lock it: mlock on Linux, the process working-set
// minimum plus VirtualLock on Windows.
#pragma once

#include <cstdint>
#include <functional>
#include <string>

namespace basalt::platform {

struct LockResult {
    bool ok = false;
    uint64_t locked_bytes = 0;   ///< may be less than requested; the rest stays pageable
    std::string note;            ///< what was done or why it failed, for the startup print
};

/// Called with the bytes locked so far, after each chunk of a lock that was given one.
using LockProgress = std::function<void(uint64_t done)>;

/// Lock [p, p + bytes) into physical memory. Partial success is reported, not hidden. With `progress` the region is
/// locked in chunks (a prefix stays locked when a later chunk is refused) and `progress` hears each one finish.
LockResult lock_resident(void* p, uint64_t bytes, const LockProgress& progress = nullptr);

/// Undo lock_resident for the same region (best effort).
void unlock_resident(void* p, uint64_t bytes);

/// Host memory whose address is a multiple of `alignment`, which has to be a power of two: a range registered with
/// the driver as a whole, or filled by reads that bypass the OS cache, needs its first byte on a boundary.
/// `free_aligned` releases what this returns, and only what this returns. nullptr when the memory is not there.
void* alloc_aligned(size_t bytes, size_t alignment);
/// The matching release for `alloc_aligned`. A null is fine.
void free_aligned(void* p);

/// Asks the OS to give back what it holds for [off, off + bytes) of a mapped file view that starts at `base` and is
/// `view_bytes` long: the range widened to whole pages, because those are what can be given back, and clipped to the
/// view. False when the OS refused or the range leaves the view. Windows takes the pages out of the process's working
/// set (they then wait on the standby list as available memory) and does not report a refusal: it is a hint there.
bool drop_mapped_view(const void* base, uint64_t view_bytes, uint64_t off, uint64_t bytes);

/// What backs a host region from `alloc_region`.
enum class RegionBacking { LargePages, NormalPages };

/// A host region of `bytes` for a large arena, with the best page size the OS gives it: 2 MiB huge pages (hugetlb on
/// Linux, large pages on Windows) first, then ordinary pages the OS is asked to back with huge ones where it can.
/// `got` says which, and `note` says what was done or why the better one was refused, for the startup print. The
/// bytes are zero-filled and committed (Windows) or demand-faulted (Linux). A non-empty `shared_file` instead maps one
/// file whose first 4 KiB identify the pack (`shared_hash`, which must not be 0) and whose remaining bytes are the
/// region - Linux only, meant for tmpfs; elsewhere it fails with a note. `mapping_base` and `mapping_bytes` are what
/// `free_region` releases: the region itself, or the file mapping that its header precedes; both are 0 when nothing
/// was allocated. nullptr on failure.
void* alloc_region(uint64_t bytes, RegionBacking& got, std::string& note, const std::string& shared_file,
                   uint64_t shared_hash, void*& mapping_base, uint64_t& mapping_bytes);
/// Releases what `alloc_region` returned through `mapping_base` and `mapping_bytes`. A null is fine.
void free_region(void* mapping_base, uint64_t mapping_bytes);

/// The most host memory the GPU may map page-locked before its shared-memory segment is full, when the OS limits it
/// beneath the driver's own refusal (`why` names where the figure comes from); false when it does not, and the driver's
/// refusal is the limit.
bool pin_budget(uint64_t& limit, std::string& why);

/// The machine's physical RAM in bytes (0 when unknown).
uint64_t total_physical_memory();

/// The RAM the OS can hand out without paging, in bytes (0 when unknown).
uint64_t available_memory();

/// What the OS could still commit, in bytes. A page-locked or fully reserved region is charged against RAM and the
/// page file together, so on Windows this is the free part of that pair - the reading a refused allocation names.
/// Linux keeps no separate charge, so there it is the available-RAM reading (0 when unknown).
uint64_t commit_headroom();

/// The RAM left alone by anything that locks or fills memory: the larger of 2 GiB and 5% of the machine's RAM.
uint64_t ram_margin();

/// Host RAM the engine takes after the resident expert copy is sized and before it serves: the GPU libraries' own
/// host memory, the prompt path's and the verifier's staging.  A resident copy leaves this much beside the margin.
inline constexpr uint64_t kLateHostBytes = 2ull << 30;

/// Host RAM one running request allocates beside its checkpoints and images: token lists, the reply, the server
/// process's own buffers.
inline constexpr uint64_t kRequestHostBytes = 256ull << 20;

/// The expert bytes one prompt chunk may read from the expert file for the slots it borrows from the GPU cache (the
/// chunk is sized to stay within it).
inline constexpr uint64_t kPromptFileReadBytes = 3584ull << 20;

/// The least RAM kept for the OS file cache while any expert is read from the file.
inline constexpr uint64_t kMinFileCacheBytes = 4ull << 30;

/// The RAM the OS file cache needs beside ram_margin() while `file_bytes` of the `complement_bytes` experts no GPU
/// tier holds are read from the file through it (0: nothing is read from the file).  A prompt chunk reads
/// `prompt_file_bytes` for the lent slots whose experts the RAM copy does not keep (at most kPromptFileReadBytes),
/// plus the file-held experts of the layer it computes and of the next one it reads ahead: two layers' file-held
/// share (`largest_layer_bytes`, every expert of the largest layer, scaled by file_bytes / complement_bytes).
/// Pages evicted before that pass is over are read again, and the reclaim then runs at drive speed.
inline uint64_t file_cache_floor(uint64_t file_bytes, uint64_t complement_bytes, uint64_t largest_layer_bytes,
                                 uint64_t prompt_file_bytes) {
    if (file_bytes == 0) return 0;
    const double share = complement_bytes > file_bytes ? (double) file_bytes / (double) complement_bytes : 1.0;
    const uint64_t layer_file = (uint64_t) ((double) largest_layer_bytes * share) + 1;
    const uint64_t pass = prompt_file_bytes + 2 * layer_file;
    return pass > kMinFileCacheBytes ? pass : kMinFileCacheBytes;
}

/// The RAM left free once the experts are placed: `margin` (ram_margin()) plus the file cache's floor.
inline uint64_t ram_floor(uint64_t margin, uint64_t file_bytes, uint64_t complement_bytes, uint64_t largest_layer_bytes,
                          uint64_t prompt_file_bytes) {
    return margin + file_cache_floor(file_bytes, complement_bytes, largest_layer_bytes, prompt_file_bytes);
}

/// The serve guard's floor (see ram_level): the margin plus one prompt chunk's file reads when experts are read from
/// the file (the margin alone when none are).  Below ram_floor() by the two layers' read-ahead, so the engine's own
/// growth after the load (conversation cache, an image's buffers) does not reach it on a machine still sized as asked.
inline uint64_t ram_guard_floor(uint64_t margin, uint64_t file_bytes, uint64_t prompt_file_bytes) {
    return margin + (file_bytes > 0 ? prompt_file_bytes : 0);
}

/// The largest RAM copy that leaves `floor` free, with `avail` bytes available and `late` bytes still to be taken by
/// the engine after the copy.
inline uint64_t max_safe_copy(uint64_t avail, uint64_t late, uint64_t floor) {
    return avail > late + floor ? avail - late - floor : 0;
}

/// The RAM a page-locked or fully resident allocation may take now: available_memory() less ram_margin().  `known`
/// is false when the OS gives no reading (nothing is refused then).
uint64_t ram_budget(bool* known = nullptr);

/// Polls available_memory() until `stop_memory_watchdog`.  Below `floor_bytes` it prints a `basalt error:` line and
/// _exit(1)s, so a load that is exhausting RAM ends before the machine starts to swap.
void start_memory_watchdog(uint64_t floor_bytes);
/// Tells the watchdog the load builds a resident RAM copy of the experts (`resident`) and how big it is so far, so
/// its error names the copy's cap instead of the resident mode.
void watch_resident_ram(bool resident, uint64_t copy_bytes);
/// The line the watchdog prints when RAM runs out: `avail` bytes free, `resident` for the resident-copy mode with a
/// copy of `copy_bytes`, `margin` the free RAM to keep. A budget flag is named only when it could shrink the copy.
std::string ram_exhausted_message(uint64_t avail, bool resident, uint64_t copy_bytes, uint64_t margin);
void stop_memory_watchdog();

/// Whether an available-RAM reading of `avail` is below `floor` (a reading of 0, unknown, is not).
inline bool ram_below_floor(uint64_t avail, uint64_t floor) { return avail > 0 && avail < floor; }

/// Whether the guard gives the mapped view back now.  `avail` is the reading, `after` the reading right after the last
/// give-back (0 = none yet) and `since_ms` the time since it.  Within `lead` of `floor`, the first give-back is due at
/// once; later ones need the cooldown over and the reading down by `refill` since `after`, i.e. the view has been
/// read in again.  A reading that stays low with nothing refilling therefore causes no repeated give-backs.
inline bool reclaim_due(uint64_t avail, uint64_t floor, uint64_t lead, uint64_t after, uint64_t since_ms,
                        uint64_t refill = 256ull << 20, uint64_t cooldown_ms = 250) {
    if (!ram_below_floor(avail, floor + lead)) return false;
    if (after == 0) return true;
    return since_ms >= cooldown_ms && avail + refill <= after;
}

/// The guard's judgement of the RAM while serving. `Low`: below its floor, where the engine sheds what it can spare and
/// a request waits a while before it starts. `Critical`: where a running request stops.
enum class RamLevel { Ok, Low, Critical };

/// Below this much available RAM a running request stops whatever the stall reading says: what a request still
/// allocates (its own buffers, a checkpoint) plus room for the OS to compress or drop other programs' idle pages.
inline constexpr uint64_t kRamHardFloor = 1ull << 30;
/// The shares of time (percent, averaged over ten seconds) in which some task, or every task, waited on memory, above
/// which RAM below the guard's floor is critical: tasks then wait on memory most of the time (thrashing). One program
/// reclaiming into compressed swap stays below them, even when it is the only task running and so stalls `full` too.
inline constexpr double kStallSomePct = 60.0;
inline constexpr double kStallFullPct = 40.0;
/// How long a request that arrives below the guard's floor waits for RAM to come back before it starts.
inline constexpr uint64_t kRamAdmitWaitMs = 30000;

/// Prompt tokens past the live conversation up to which a request only continues it (the server's wrap-up after a
/// thinking budget) rather than starting a turn of its own.
inline constexpr int64_t kRamContinueTokens = 256;
/// Whether a request that reads `fresh` tokens past the live conversation (`from_live`: it continues that
/// conversation) skips the admission wait: it adds little RAM and is part of a reply already streaming.
inline bool ram_continues_live(bool from_live, int64_t fresh) {
    return from_live && fresh >= 0 && fresh <= kRamContinueTokens;
}

/// The OS's memory-stall shares over the last ten seconds, in percent. `known` is false where the OS keeps none.
struct MemoryStalls {
    bool known = false;
    double some = 0.0;   ///< time some task waited on memory
    double full = 0.0;   ///< time every runnable task waited on memory
};
/// The stall reading now (Linux: /proc/pressure/memory; none on Windows).
MemoryStalls memory_stalls();
/// The `some` and `full` lines' avg10 of a pressure file's text; `known` false when either is missing.
MemoryStalls parse_memory_stalls(const std::string& text);

/// The guard's level for an available-RAM reading `avail` (0 = unknown) against `floor`, the hard floor `hard` and
/// the stall reading. Without a stall reading, below the floor is critical at once.
inline RamLevel ram_level(uint64_t avail, uint64_t floor, uint64_t hard, const MemoryStalls& stalls) {
    if (!ram_below_floor(avail, floor)) return RamLevel::Ok;
    if (avail < (hard < floor ? hard : floor) || !stalls.known) return RamLevel::Critical;
    return stalls.some >= kStallSomePct || stalls.full >= kStallFullPct ? RamLevel::Critical : RamLevel::Low;
}

/// Whether a request that has waited `waited_ms` at admission keeps waiting at `level`.
inline bool ram_admit_waits(RamLevel level, uint64_t waited_ms, uint64_t limit_ms = kRamAdmitWaitMs) {
    return level != RamLevel::Ok && waited_ms < limit_ms;
}

/// The error a request stops with at RamLevel::Critical: the free RAM `avail` against the floor it fell under
/// (`hard`, or the guard's `floor` with the stalls that made it critical) and who took the RAM since the engine was
/// ready: `others` bytes by other programs, `own` by the engine. The budget flag is named only when the engine's own
/// growth is the larger part.
std::string ram_stop_message(uint64_t avail, uint64_t floor, uint64_t hard, const MemoryStalls& stalls,
                             uint64_t others, uint64_t own);

/// While serving: polls available_memory() against `floor_bytes` (and, below it, memory_stalls()) and keeps the
/// level for ram_guard_level. Nothing is shrunk or exited here; the serve loop acts on the level.
void start_ram_guard(uint64_t floor_bytes);
/// The level of the guard's last reading; `avail`, `floor` and `stalls` get the reading it was judged on.
RamLevel ram_guard_level(uint64_t* avail = nullptr, uint64_t* floor = nullptr, MemoryStalls* stalls = nullptr);
/// How many times the reading has gone below the floor since the guard started: the serve loop sheds once per drop.
uint64_t ram_guard_drops();
/// Pauses the guard (`paused`): its readings are not judged, and the level stays the last one judged; for a sleeping
/// engine, whose GPU state the driver holds in RAM until the wake. Unpausing judges a fresh reading before it returns.
void ram_guard_pause(bool paused);
/// Hands the heap's free pages back to the OS (best effort).
void release_free_heap();
/// A mapped file view the guard gives back (drop_mapped_view) when a reading comes within 1 GiB of its floor, then
/// reads again; see reclaim_due for when it gives back again.
/// Windows keeps a view's pages in the process once read and does not count them as available, where Linux counts
/// them in MemAvailable; without this the guard would stop a request over pages the OS can reclaim.
/// A null `base` withdraws the view; once that returns, the guard no longer touches it, so it can be unmapped.
void ram_guard_reclaim_view(const void* base, uint64_t bytes);

/// Whether the OS file cache could keep the `read_bytes` the expert file is read for, beside
/// `arena_bytes` of RAM held by the engine's own copy of the experts and `margin` for everything else, with `avail`
/// bytes of RAM available.  The file tier passes only the expert bytes it really reads from the file (the experts
/// outside its resident RAM copy) and the RAM that copy really holds.
inline bool file_cache_keeps(uint64_t avail, uint64_t arena_bytes, uint64_t read_bytes,
                             uint64_t margin = 4ull << 30) {
    const uint64_t room = avail > arena_bytes + margin ? avail - arena_bytes - margin : 0;
    return room >= read_bytes;
}

/// Whether `advise_willneed` asks the OS for anything: it does on both platforms.
bool read_ahead_enabled();
/// Asks the OS to start bringing [p, p + bytes) of a mapped file view into memory, without waiting for the access.
/// A hint, never an error: the system skips what it will not prefetch and says nothing about it, and without the
/// hint the pages still arrive - one fault at a time, where they are touched.  The range is asked for in 128 KiB
/// windows, which keeps the reads in flight bounded.  Only a mapped view is reached this way.
void advise_willneed(const void* p, uint64_t bytes);
/// The same for [offset, offset + bytes) of an open descriptor.  Linux asks its page cache.  Windows asks nothing:
/// the cache manager takes no range hint for a handle, and a read through a descriptor is a cached read unless the
/// caller opted out of that cache - which already reads ahead around a sequential reader.  Bytes reached only
/// through a descriptor are therefore warmed as they are read, not ahead of it.
void advise_willneed(int fd, uint64_t offset, uint64_t bytes);

}  // namespace basalt::platform
