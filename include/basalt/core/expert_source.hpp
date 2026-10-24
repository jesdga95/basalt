// include/basalt/core/expert_source.hpp - the expert pool's adapter to the host loop.
//
// `session_loop` publishes, per layer, the NORMED activation `x_f`, the ten routed expert ids and their router
// weights, and expects `k x n_embd` floats back.  This is the piece that turns those four things into an
// `ExpertPool::run` call; its one job is to get the CONTRACT between them right.
//
// THE CONTRACT, and each clause is a way to be wrong:
//
//   1. **`x_f` IS THE SAME PINNED BUFFER EVERY LAYER.**  Its ADDRESS never changes, so the `ActQ` cannot be
//      cached by pointer - a cache keyed on `x_f` would quantize layer 0 and reuse it for all the remaining
//      layers, which is a finite, plausible, completely wrong token.  There is no cache here at all: the
//      conversion is one 2560-element pass, and a correct answer is worth more than the microseconds.
//   2. **THE POOL DOES NOT APPLY THE ROUTER WEIGHT.**  `moe_combine` (`src/core/layer.cpp`) sums
//      `w[i] * parts[i]` on the device.  `ExpertJob::weight` is a diagnostic field; setting it here would
//      apply the weight twice, which is invisible in a single layer and compounds over 48.
//   3. **THE KERNEL'S GEOMETRY IS FIXED BY THE ARTIFACT** (`H = 2560`, `FF = 640`, `BLOB = 1,382,400`).  A
//      different `n_embd` or a different expert width is REFUSED rather than mis-indexed, because the blob's
//      internal offsets are compile-time constants and reading a 640-wide expert as a 2560-wide one walks off
//      the end into the next expert's bytes without faulting.
#pragma once

#include "basalt/core/expert_cache.hpp"
#include "basalt/core/hit_hook.hpp"
#include "basalt/core/model_file.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/platform/mapped_file.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <cstddef>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <string>
#include <vector>

namespace basalt::kernels::cpu {
struct ExpertLayout;
}

namespace basalt::core {

class PeerExperts;   // multi-GPU: the second GPU's expert tier (peer_experts.hpp)
class RemoteExperts;
struct LoadStats;

namespace detail {

/// Sentinel used by the pure complement planner for a blob that remains in the mmap fallback.
inline constexpr uint64_t kNoCacheComplement = ~uint64_t{0};

/// Required cgroup-v2 usage counters for the conservative cache-reclaim allowance.
struct CgroupMemoryStat {
    uint64_t current = 0;
    uint64_t inactive_file = 0;
    uint64_t file_dirty = 0;
    uint64_t file_writeback = 0;
    bool valid = false;
    uint64_t file = 0;  // page cache, active + inactive; 0 when memory.stat does not report it
};

/// Parse a cgroup-v2 memory.stat text (inactive_file, file_dirty, file_writeback required; file optional).
/// False on a malformed or incomplete text.
bool parse_cgroup_memory_stat(const std::string& text, uint64_t current, CgroupMemoryStat& stat);

/// Calculate additional bytes under a finite cgroup limit after reclaiming clean file cache.
/// Returns false when the required memory.stat counters were unavailable.
bool cgroup_available_bytes(uint64_t limit, const CgroupMemoryStat& stat, uint64_t& bytes);

/// the RAM this process can get.  `available`: MemAvailable, lowered to the room under the tightest cgroup limit;
/// `cgroup_limit`: that tightest limit itself (v2 memory.max of the group and its ancestors, or v1
/// memory.limit_in_bytes), ~0 when there is none - what a container can never exceed.
struct HostMemory {
    uint64_t available = 0;
    uint64_t cgroup_limit = ~uint64_t{0};
};

/// This reads `meminfo`, `self_cgroup` and the cgroup tree under `cgroup_root` (the parameters are for tests; an
/// empty one is the system's own file, see platform::meminfo_path - and on a system with none, the OS's available-RAM
/// reading is the whole answer and no container limit is tighter than it).  A v1 memory controller line wins over the unified line (the hybrid layout): v1 reads
/// `<root>/memory/<path>` (memory.limit_in_bytes - memory.usage_in_bytes, clean inactive file cache counted as free),
/// v2 reads `<root>/<path>` (memory.max, memory.current, memory.stat).  A group not visible here is read at the mount's
/// root (a container's own namespace); groups without a readable limit add none; no cgroup file or line is MemAvailable
/// alone.  False when MemAvailable is missing or a visible v2 limit or its memory.stat is malformed.
bool host_available_memory(HostMemory& m, const std::string& meminfo = "", const std::string& self_cgroup = "",
                           const std::string& cgroup_root = "");

/// Build compact offsets for experts absent from both the primary GPU cache and an optional second GPU tier.
/// Kept CPU-only so selection and byte accounting can be tested without initializing a GPU.
bool make_cache_complement_plan(
    int64_t n_layers, int64_t n_expert, const std::vector<uint64_t>& layer_blob_bytes,
    const std::vector<std::pair<int32_t, int32_t>>& primary_gpu_pairs,
    const std::vector<std::pair<int32_t, int32_t>>& additional_gpu_pairs,
    std::vector<uint64_t>& offsets, uint64_t& bytes, std::string& err);

/// Resolve one blob through the compact copy when present, otherwise preserve its exact mapped-file fallback.
const uint8_t* cache_complement_blob_or_fallback(
    size_t index, const std::vector<uint64_t>& offsets, const uint8_t* complement_host,
    const uint8_t* mapped_fallback);

/// The resident RAM mode: which GPU-cache slots' experts are kept in RAM too.  The prompt path lends the cache's
/// LAST slots (from `lend_from` on; a short prompt lends only the last few), and a lent slot's expert is streamed
/// from RAM during the prompt and copied back into its slot after it.  `base_bytes` (every expert no slot holds)
/// must fit `budget`; slots are then added from the end down to `lend_from` while they still fit.  Returns the
/// first slot kept in RAM (`slot_bytes.size()` = none), or -1 when `base_bytes` alone exceeds `budget`.
int64_t choose_resident_keep_from(const std::vector<uint64_t>& slot_bytes, uint64_t base_bytes, uint64_t budget,
                                  int64_t lend_from);

/// The resident RAM mode's room after `base_bytes`: the adaptive swaps' free places (`spare_bytes`) are placed first,
/// since without them no swap can run, and the lent slots then fill what is left (choose_resident_keep_from).
/// `spares_fit` says whether the places fit; when they do not, the slots get the whole budget.
int64_t choose_resident_room(const std::vector<uint64_t>& slot_bytes, uint64_t base_bytes, uint64_t budget,
                             int64_t lend_from, uint64_t spare_bytes, bool& spares_fit);

/// The RAM a resident copy may take with `physical` bytes available and `headroom` left free.  The arena is ordinary
/// 4 KiB pages, so their page tables and the driver's per-page records (physical / 256) come on top of its bytes.  The
/// whole copy's safety check and the keep-what-fits budget are both sized by this one figure.
inline uint64_t resident_room(uint64_t physical, uint64_t headroom) {
    const uint64_t reserve = headroom + physical / 256;
    return physical > reserve ? physical - reserve : 0;
}

/// The adaptive tier swapped `in` into a GPU slot and `out` out of it: `out` takes `in`'s place in the compact copy
/// (the caller copies out's bytes there).  False, and nothing changed, unless `in` is in the copy and `out` is not.
bool exchange_cache_complement(std::vector<uint64_t>& offsets, size_t in, size_t out);

/// A RAM region holding experts a shrunken VRAM cache gave up.  Its places live in the offset space after the
/// complement and the top-up region: `[base, base + bytes)`.
struct HomeRegion {
    uint64_t base = 0, bytes = 0;
    void* arena = nullptr;
    const uint8_t* device = nullptr;   ///< the mapped device alias when page-locked through the driver
    uint64_t locked = 0;               ///< bytes held in the working set (a malloc'd region)
    bool pinned = false;
};
/// The region holding the `bytes` at `off`, or -1.
int home_region_find(const std::vector<HomeRegion>& regions, uint64_t off, uint64_t bytes);
/// Whether any expert's place lies in the region.
bool home_region_used(const HomeRegion& r, const std::vector<uint64_t>& offsets);

}  // namespace detail



/// The files the experts come from: the blobs, and (for the shared arena's fingerprint) the index and layout they
/// were converted with.
struct ExpertFiles {
    FileRange experts, index, layout;
    static ExpertFiles from(const ModelSources& s) {
        ExpertFiles f;
        f.experts = s.experts;
        f.index = s.dense_index;
        f.layout = s.experts_layout;
        return f;
    }
};

/// Where one routed expert's bytes come from.
///
/// The interface lets a test supply an in-memory source without a multi-GB artifact; the implementations are the
/// file-backed readers and the resident arena below.
class ExpertSource {
public:
    virtual ~ExpertSource() = default;

    /// The expert-layout blob for `(layer, expert)`, or nullptr if it cannot be produced.
    ///
    /// The pointer only has to stay valid until the next `blob()` call: with `h = 0` every expert is computed
    /// immediately and nothing is retained.  A CACHING source must return pointers into the cache, not into a
    /// reused staging buffer - otherwise the pool would read the next expert's bytes while computing this one.
    virtual const uint8_t* blob(int64_t layer, int64_t expert) = 0;

    /// Blobs touched, for the driver to report.  A source that does not count returns 0.
    virtual int64_t reads() const { return 0; }

    /// The blob of `(layer, expert)` when it sits in RAM already (no read, no count), else null.  Valid during a verify
    /// window: the places move only between windows.
    virtual const uint8_t* resident_blob(int64_t layer, int64_t expert) const { (void) layer; (void) expert; return nullptr; }

    /// Whether `blob(layer, expert)` lies in page-locked, CUDA-registered memory, so an asynchronous host-to-device
    /// copy can DMA it directly (no staging copy on the CPU).
    virtual bool pinned(int64_t layer, int64_t expert) const { (void) layer; (void) expert; return false; }

    /// The DEVICE address of a pinned, mapped blob (the GPU can read it over PCIe), or null.
    virtual const uint8_t* device_alias(int64_t layer, int64_t expert) const { (void) layer; (void) expert; return nullptr; }
    /// A file-backed source: start reading this expert's pages now (it will be needed by the CPU); no-op elsewhere.
    virtual void prefetch(int64_t layer, int64_t expert) { (void) layer; (void) expert; }
    /// A file-backed source: queue the read of these experts' file pages now (a helper thread issues it, the caller
    /// does not wait), so a layer's CPU experts come off the disk together rather than one page fault at a time;
    /// no-op for an expert held in RAM and elsewhere.
    virtual void advise_cpu(int64_t layer, const int32_t* ids, int64_t n) const { (void) layer; (void) ids; (void) n; }
    /// A file-backed source: this expert lives in VRAM, so its pages need not stay in RAM - hand them back to the
    /// kernel (a later read re-reads the file; no result depends on it).  Returns the bytes released; 0 elsewhere.
    virtual uint64_t release(int64_t layer, int64_t expert) { (void) layer; (void) expert; return 0; }
    /// Whether the verify window may give the GPU a PCIe share of this layer's misses at all (each expert is still
    /// checked with `pinned`).  The arena answers per layer through its expert 0; the resident RAM mode's compact
    /// copy has no expert 0 when the GPU cache holds it, so it answers for the whole copy.
    virtual bool pcie_layer(int64_t layer) const { return device_alias(layer, 0) != nullptr; }

    /// Whether `blob(layer, expert)` is served from the mapped model file (not from a RAM copy): its bytes are at
    /// `offset` of the open file `fd`, so they can be read with pread.  False by default.
    virtual bool file_extent(int64_t layer, int64_t expert, int& fd, uint64_t& offset) const {
        (void) layer; (void) expert; (void) fd; (void) offset; return false;
    }

    /// The blob's bytes into `dst` (blob_bytes(layer) of them).
    virtual bool copy_blob(int64_t layer, int64_t expert, uint8_t* dst);
    /// Asks the OS for these pairs' bytes ahead of the `blob` calls that read them.  False (nothing asked) by default.
    virtual bool advise_pairs(const std::pair<int32_t, int32_t>* pairs, int64_t n) const {
        (void) pairs; (void) n; return false;
    }

    /// Reader threads `read_blobs` runs; a sink is told which one it is called on.
    static constexpr int kBlobReaders = 6;
    /// Receives pair `index`'s blob from reader `worker`; the bytes are valid only during the call.
    using BlobSink = std::function<bool(int64_t index, int worker, const uint8_t* blob, std::string& err)>;
    /// Hands every pair's blob to `sink`, from up to `kBlobReaders` threads.  The default reads one blob after another
    /// through `blob()`, asking the OS for the following ones ahead.
    virtual bool read_blobs(const std::pair<int32_t, int32_t>* pairs, int64_t n, const BlobSink& sink, std::string& err);
};

/// Copies the pairs' blobs into the cache's slots (`slots[i]` for `pairs[i]`) on `device`, reading through
/// `src.read_blobs` and overlapping the reads with the host-to-device copies.  `on_bytes` is told each blob's size as
/// it lands (from several threads).  The copies are complete when this returns.
bool fill_cache_slots(ExpertSource& src, ExpertCache& cache, int device, const std::pair<int32_t, int32_t>* pairs,
                      const int32_t* slots, int64_t n, const std::function<void(uint64_t)>& on_bytes, std::string& err);

/// What the GPU computes in a verify window's layer, written by the pool (mapped host memory) right after the ring
/// and published before the CPU starts its own share.  Groups of entries that share a blob; the blob is a VRAM slot or
/// a pinned host blob read over PCIe.
struct GpuPlanSink {
    int32_t* counts = nullptr;             ///< [0] groups, [1] entries
    int32_t* start = nullptr;              ///< cap + 1
    int32_t* dst = nullptr;                ///< cap: the entry's row of parts (token * k + j)
    int32_t* tok = nullptr;                ///< cap: the entry's token
    unsigned long long* ptr = nullptr;     ///< cap: the group's blob, device address
    /// The PCIe share is a second list of groups - `ptr2[q]` is the blob's device alias in the mapped arena, `start2`
    /// indexes the same dst/tok entries - staged by a copy kernel inside the graph, then computed by the GPU.
    /// counts[0] = VRAM groups, counts[1] = all entries, counts[2] = PCIe groups.
    unsigned long long* ptr2 = nullptr;
    int32_t* start2 = nullptr;
    int64_t staging_cap = 0;
    int64_t cap = 0;
    void (*publish)(void* ctx) = nullptr;
    /// Raises flag B: the PCIe groups of this layer are planned and the graph may stage them.
    void (*staged)(void* ctx) = nullptr;
    void* ctx = nullptr;
    /// the window planned this layer on the device; ext_kind[i] is entry i's tier (0 CUDA0, 2 the peer, -1 the CPU).
    /// The pool then computes only the CPU entries, publishes no plan and launches no peer.
    const int32_t* ext_kind = nullptr;
};

/// The adapter's own state.  One per session, reused every layer so the token path allocates nothing.
struct ExpertDispatch {
    basalt::kernels::cpu::ExpertPool* pool = nullptr;
    ExpertSource* src = nullptr;
    RemoteExperts* remote[3] = {}; ///< optional CUDA1..3 tiers for otherwise CPU-served rows
    int remote_count = 0;
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

    std::vector<basalt::kernels::cpu::ExpertJob> jobs;
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

/// **THE HITS, LAUNCHED AFTER THE MISSES ARE STAGED AND BEFORE `post[l]`.**  Same shape as `PoolFn` and for the
/// same reason: `session_loop` owns the ORDER and this owns the work, so the loop needs to know nothing about
/// expert caches.  Returns immediately when there are no hits, which is every layer until the cache is warm.
///
/// It must run AFTER the host has copied the misses into `parts_dev` (it writes into the same buffer, on rows
/// the CPU zeroed) and BEFORE `post[l]` (which reads it).  Both are stream-ordered on the loop's own stream.
void expert_hit_run(void* user, void* stream, HitPhase phase, const int32_t* ids, int64_t k);
/// The pool half of the same decision; see `ExpertDispatch::is_hit`.

/// **A FILE-BACKED SOURCE: the experts section, memory-mapped, no cache.**
///
/// The experts section holds exactly `n_layers x n_expert` blobs of `BLOB` bytes, so the index is `layer * n_expert + expert`
/// with NO padding.  A mapping is therefore the whole implementation: the blob pointer is base plus a multiply, and
/// the page fault that follows is the read.
///
/// With enough RAM a warm OS page cache holds the whole file, so after the first token a read is a DRAM read, not a
/// disk read; the first token's reads come off the disk and are the cold-path cost.
///
/// IT IS NOT AN LRU, AN LFU OR A PREFETCHER: every read goes through the page cache.
class FileExpertSource : public ExpertSource {
public:
    FileExpertSource();
    ~FileExpertSource() override;
    FileExpertSource(const FileExpertSource&) = delete;
    FileExpertSource& operator=(const FileExpertSource&) = delete;

    /// Maps the model's `experts` section (a FileRange) and checks its size against the loaded expert layout.
    ///
    /// The size check is not a formality: a short file would fault at the END of a long sequence, and an over-long one
    /// means the file is not the one the geometry came from.  Refuses with the two numbers.
    bool open(const FileRange& experts, int64_t n_layers, int64_t n_expert, std::string& err);
    /// Pin a compact host mirror of experts absent from a fully filled static GPU cache. The mmap remains open as a
    /// fallback for later cache reloads. This is opt-in because the complement may still be a large allocation.
    ///
    /// The resident RAM mode (`--resident-experts`, `--resident-cpu-experts`):
    ///   - `pin`: page-locked and mapped (`cudaHostAlloc`), so the prompt path copies it by DMA and the verify window
    ///     may read a share of the misses over PCIe; when the driver refuses, ordinary memory locked in the working set
    ///     instead.  `pin = false` is ordinary pageable memory.
    ///   - `lend_from_slot` >= 0: the GPU-cache slots from there to the end are the prompt path's lend region; their
    ///     experts are kept in RAM too, from the last slot down, as far as `available RAM - headroom_bytes` allows (a
    ///     lent slot's expert is streamed during the prompt and copied back after it).
    ///   - the rest (the experts no slot holds) must fit that budget, or nothing is allocated and this returns false.
    ///
    /// `budget_bytes` > 0 (`--resident-budget-gib`): only as many of those experts as fit `budget_bytes`, taken in
    /// `rank` order (the expert profile: the hottest after the GPU cache's), are copied; the rest stay on the mapped
    /// files (the SSD tier).  No lend region then (a lent slot's expert is read from the files).
    /// `budget_bytes` = `kResidentWhatFits` is that path sized by the RAM alone (available minus the headroom and a
    /// margin) - the soft --resident-experts mode's second try when the whole complement does not fit; false when not
    /// even one expert fits.
    static constexpr uint64_t kResidentWhatFits = ~0ull;
    bool pin_cache_complement(
        const ExpertCache& cache, std::string& err, bool pin = true,
        const std::vector<std::pair<int32_t, int32_t>>& additional_gpu_pairs = {}, int64_t lend_from_slot = -1,
        uint64_t headroom_bytes = 8ull << 30, uint64_t budget_bytes = 0,
        const std::vector<std::pair<int32_t, int32_t>>* rank = nullptr);
    void close();

    bool mapped() const { return base_ != nullptr; }
    int64_t blobs() const { return blobs_; }
    uint64_t pinned_bytes() const { return complement_pinned_ ? complement_pin_limit_ : 0; }
    uint64_t resident_bytes() const { return complement_bytes_; }
    bool complement_pinned() const { return complement_pinned_; }
    bool complement_ready() const { return complement_ready_; }
    uint64_t locked_bytes() const { return complement_locked_; }
    /// Lend-region slots whose experts the compact copy holds (the last ones of the cache).
    int64_t resident_lent_slots() const { return complement_lent_slots_; }

    // ---- the resident RAM mode and the adaptive tier.  A swap puts `in` (held here) into a GPU slot and evicts
    // `out` (held only by that slot).  Before the slot is overwritten the caller copies it back into an exchange
    // buffer and calls `stage_exchange`: `out` is then read from that buffer, and `in` still from here (the CPU
    // computes both until the swap lands).  Once the slot copy has landed, `commit_exchanges` moves `out` into
    // `in`'s place, so the copy keeps holding exactly the experts the GPU does not - with no read of the file.
    /// Whether the compact copy holds `(layer, expert)`.
    bool has_resident(int64_t layer, int64_t expert) const;
    /// Whether `(layer, expert)` sits in the region added after the load (the RAM top-up) rather than the loaded copy.
    bool resident_in_top_up(int64_t layer, int64_t expert) const;
    /// Host room for `n` evicted blobs (page-locked when possible).  Idempotent for the same or a smaller `n`.
    bool reserve_exchanges(int64_t n, std::string& err);
    int64_t exchange_capacity() const { return xstage_cap_; }
    uint8_t* exchange_buffer(int64_t q) const;
    /// Requires `has_resident(layer, in)`, `!has_resident(layer, out)` and `exchange_buffer(q)` holding out's blob.
    bool stage_exchange(int64_t layer, int64_t in, int64_t out, int64_t q);
    /// After the GPU copies of every staged swap have landed.  Returns how many exchanges were applied.
    int64_t commit_exchanges();
    /// The copies of `commit_exchanges` ahead of it (it then only moves the bookkeeping), at most `max_bytes` per call
    /// (true once all are done): once every staged `in` is in its GPU slot and resident there, so nothing reads its
    /// place here any more.
    bool copy_exchanges(uint64_t max_bytes);
    bool exchanges_staged() const { return !staged_.empty(); }
    int64_t exchanges() const { return exchanges_; }
    // ---- free places in the compact copy instead of exchange buffers.  `set_spares` (before pin_cache_complement)
    // appends n blob-sized places to it.  A swap's evicted expert is copied by DMA from its GPU slot straight into a
    // free place (`spare_take`), becomes the copy's (`spare_land`) once that copy has landed, and the incoming expert's
    // place joins the free ones (`spare_release`) once it is resident on the GPU: no host copy, no exchange buffer.
    // Call between verify windows (the pool reads the offsets during one).
    void set_spares(int64_t n) { spares_want_ = n; }
    int64_t spares_free() const { return (int64_t) spare_free_.size(); }
    /// A free place for `layer`'s blob (its host address; `off` its offset), or nullptr.
    uint8_t* spare_take(int64_t layer, uint64_t& off);
    /// `out` (held by no place) is read from the place at `off` from now on.
    bool spare_land(int64_t layer, int64_t out, uint64_t off);
    /// `in`'s place is free (it is resident on the GPU now).
    bool spare_release(int64_t layer, int64_t in);
    /// A place `spare_take` gave, unused.
    void spare_return(int64_t layer, uint64_t off);
    /// After loading, before any reader runs: keeps more experts in RAM, in a second region placed after the first
    /// in the offset space, hottest first (`rank`) as far as available RAM less the RAM margin allows.  `skip(l, e)`
    /// names experts a GPU holds.  `later` is RAM still to be taken after this, kept free too.  Returns the bytes added
    /// (0: nothing done); `count` gets the experts.
    uint64_t top_up(const std::vector<std::pair<int32_t, int32_t>>& rank,
                    const std::function<bool(int32_t, int32_t)>& skip, bool pin, int64_t& count, uint64_t later = 0);
    // ---- RAM homes for the experts a shrinking VRAM cache gives up (--vram-elastic in the resident mode).  The
    // caller asks for homes (`homes_take`: one region, refused with `err` when the RAM budget cannot hold it), copies
    // each expert from its slot into `dst[k]`, waits for the copies, and calls `homes_commit`: the experts are then
    // read from RAM like the compact copy's.  `homes_abort` drops an uncommitted take.  When an expert goes back
    // into a slot, `home_release` frees its place (false when its copy is not in a home region); `homes_reclaim`
    // returns the regions no expert uses to the OS.  Only between requests, with no swap in flight.
    bool homes_take(const std::vector<std::pair<int32_t, int32_t>>& experts, std::string& err, std::vector<uint8_t*>& dst);
    void homes_commit();
    void homes_abort();
    bool home_release(int64_t layer, int64_t expert);
    uint64_t homes_reclaim();
    uint64_t homes_bytes() const;
    /// Debug: whether `bytes` equal the expert's bytes in the model files.
    bool equals_file(int64_t layer, int64_t expert, const uint8_t* bytes) const;
    /// With the compact copy ready: blobs read from the mapped file since (what the plain mmap mode may read from
    /// the SSD).  0 in a steady resident mode; lend-region experts that did not fit the RAM count here.
    int64_t file_reads() const { return file_reads_.load(std::memory_order_relaxed); }
    /// CS-T per-tier counters: blobs served from the RAM copy, and the bytes read from the mapped files (the blobs
    /// `file_reads` counts, plus the prompt path's copies of them).
    int64_t ram_reads() const { return ram_reads_.load(std::memory_order_relaxed); }
    uint64_t file_read_bytes() const { return file_read_bytes_.load(std::memory_order_relaxed); }
    /// Every expert's bytes (n_layers x n_expert blobs).
    uint64_t expert_bytes() const;
    /// Asks the OS for these pairs' blobs ahead (platform::advise_willneed).  False when read-ahead is off, or before
    /// `open`.
    bool advise_pairs(const std::pair<int32_t, int32_t>* pairs, int64_t n) const override;

    const uint8_t* blob(int64_t layer, int64_t expert) override;
    const uint8_t* resident_blob(int64_t layer, int64_t expert) const override;
    void advise_cpu(int64_t layer, const int32_t* ids, int64_t n) const override;
    bool pinned(int64_t layer, int64_t expert) const override;
    bool file_extent(int64_t layer, int64_t expert, int& fd, uint64_t& offset) const override;
    const uint8_t* device_alias(int64_t layer, int64_t expert) const override;
    bool pcie_layer(int64_t layer) const override;
    bool copy_blob(int64_t layer, int64_t expert, uint8_t* dst) override;
    /// Each blob read from the file with block-aligned unbuffered reads into small pinned staging buffers; a blob that
    /// cannot be read that way, or a file system without unbuffered reads, takes the mapped copy.
    bool read_blobs(const std::pair<int32_t, int32_t>* pairs, int64_t n, const BlobSink& sink, std::string& err) override;

    /// Blobs touched, for the driver to report.  With `h = 0` this is `48 * k` per token and the number is only
    /// interesting once Phase 3 makes it not so.
    int64_t reads() const override { return reads_; }

private:
    struct AdviseWorker;
    mutable std::unique_ptr<AdviseWorker> advise_worker_;
    mutable std::once_flag advise_once_;
    const uint8_t* mapped_blob(int64_t layer, int64_t expert) const;
    /// The blob's bytes from the mapped file into `dst`.
    bool copy_from_files(int64_t layer, int64_t expert, uint8_t* dst) const;
    static constexpr uint64_t kNoComplement = detail::kNoCacheComplement;
    std::atomic<uint64_t> file_us_{0};
    std::atomic<int64_t> ram_reads_{0};
    std::atomic<uint64_t> file_read_bytes_{0};
    platform::MappedFile view_;               ///< the mapped experts; base_ is its first byte
    const uint8_t* base_ = nullptr;
    uint64_t file_off_ = 0;                   ///< where the experts start in the mapped file
    int64_t blobs_ = 0;
    int64_t n_layers_ = 0;
    int64_t n_expert_ = 0;
    uint64_t mapped_bytes_ = 0;
    std::vector<uint64_t> layer_offsets_, layer_blob_bytes_;
    void* complement_arena_ = nullptr;
    const uint8_t* complement_host_ = nullptr;
    const uint8_t* complement_device_ = nullptr;
    uint64_t complement_bytes_ = 0;
    // the top-up region: offsets from complement_bytes_ on lie in it
    void* top_arena_ = nullptr;
    const uint8_t* top_device_ = nullptr;
    uint64_t top_bytes_ = 0, top_locked_ = 0;
    bool top_pinned_ = false;
    std::vector<detail::HomeRegion> homes_;
    uint64_t homes_next_ = 0;   // the next region's base (0: none yet)
    std::vector<std::pair<size_t, uint64_t>> homes_pending_;   // (expert index, place) of an uncommitted take
    uint64_t homes_start() const { return complement_bytes_ + top_bytes_; }
    /// The host address of a blob of `bytes` at offset `off` (either region), or nullptr.
    const uint8_t* host_at(uint64_t off, uint64_t bytes) const;
    std::vector<uint64_t> complement_offsets_;
    bool complement_pinned_ = false;
    bool complement_partial_ = false;         ///< CS-T: only the first complement_pin_limit_ bytes are registered
    uint64_t complement_pin_limit_ = 0;
    uint64_t complement_lock_off_ = 0;        ///< the working-set lock covers [lock_off, lock_off + locked)
    bool complement_ready_ = false;
    uint64_t complement_locked_ = 0;          ///< bytes held in the working set (pin refused)
    int64_t complement_lent_slots_ = 0;
    std::vector<const uint8_t*> override_;    ///< staged exchanges: an evicted expert read from its exchange buffer
    struct Exchange { size_t in, out; int64_t q; uint64_t bytes; };
    std::vector<Exchange> staged_;
    bool xcopied_ = false;   // copy_exchanges finished staged_
    uint64_t xcopy_at_ = 0;  // copy_exchanges' progress through staged_ (bytes)
    uint8_t* xstage_ = nullptr;               ///< exchange buffers, `xstage_cap_ x xstage_blob_`
    bool xstage_pinned_ = false;
    int64_t xstage_cap_ = 0;
    uint64_t xstage_blob_ = 0;
    int64_t exchanges_ = 0;
    int64_t spares_want_ = 0;
    std::vector<std::pair<uint64_t, uint64_t>> spare_free_;   // (offset, bytes) of free places
    std::atomic<int64_t> file_reads_{0};
    int64_t reads_ = 0;
    int fd_ = -1;
};

// ================================ THE RESIDENT ARENA ================================
//
// A mapped file's pages are file-backed, and file-backed pages are exactly the ones the OS drops from the standby list
// when it wants memory.  The expert stream then re-faults from disk, and the engine's rate depends on whether the
// cache happens to hold the experts section.
//
// This class reads the experts section into a `PinnedArena` once at startup, so the expert stream comes from anonymous
// memory the OS has no cheaper reason to evict.  `PinnedArena` also tries `cudaHostRegister`, which the GPU needs for
// the cache fills and the CPU/PCIe miss split - but registration is best-effort and reported, not assumed.
class ArenaExpertSource : public ExpertSource {
public:
    ArenaExpertSource() = default;
    ~ArenaExpertSource() override;
    ArenaExpertSource(const ArenaExpertSource&) = delete;
    ArenaExpertSource& operator=(const ArenaExpertSource&) = delete;

    /// Allocates and loads the model's `experts` section.  Prints nothing; the caller reports `note()` and the load
    /// rate, because those are the two numbers that say whether the arena is the one that was asked for.
    bool open(const ExpertFiles& files, int64_t n_layers, int64_t n_expert, int threads, std::string& err,
              uint64_t max_pinned_bytes = 0, const std::string& shared_arena_file = {});
    void close();

    bool mapped() const { return base_ != nullptr; }
    int64_t blobs() const { return blobs_; }
    const uint8_t* blob(int64_t layer, int64_t expert) override;
    const uint8_t* resident_blob(int64_t layer, int64_t expert) const override;
    int64_t reads() const { return reads_; }
    bool pinned(int64_t layer, int64_t expert) const override;
    const uint8_t* device_alias(int64_t layer, int64_t expert) const override;
    void prefetch(int64_t layer, int64_t expert) override;
    uint64_t release(int64_t layer, int64_t expert) override;

    /// What backing was obtained and why, for the startup print.  "The engine adapts to the machine it is on" is
    /// only true if the engine says what it got.
    const std::string& note() const { return note_; }
    /// set by `open` when the RAM available is less than the arena needs (a recommendation; it still loads)
    const std::string& ram_warning() const { return ram_warning_; }
    double load_gib_per_second() const { return gib_per_s_; }
    // Loader fix: the load, split.  `load_seconds()` is the wall clock of the load loop; the other two are
    // sums over the reader threads (see LoadStats), so on their own they say how much of that wall was spent
    // waiting for the disk and how much in memcpy + FNV-1a.
    double load_seconds() const { return load_seconds_; }
    double load_read_seconds() const { return load_read_s_; }
    double load_copy_seconds() const { return load_copy_s_; }

private:
    void* arena_ = nullptr;          ///< the PinnedArena, owned
    platform::MappedFile map_;       ///< the arena file, mapped read-only (not the PinnedArena)
    std::vector<const uint8_t*> dev_slice_;   ///< device alias of each registered slice (or of the whole range)
    uint64_t slice_bytes_ = 0;
    const uint8_t* base_ = nullptr;
    int64_t blobs_ = 0;
    int64_t n_expert_ = 0;
    int64_t reads_ = 0;
    std::string note_;
    std::string ram_warning_;
    double gib_per_s_ = 0.0;
    double load_seconds_ = 0.0;
    double load_read_s_ = 0.0;
    double load_copy_s_ = 0.0;
    uint64_t pinned_bytes_ = 0;
};

}  // namespace basalt::core
