// include/basalt/kernels/cpu/pool.hpp - the CPU expert pool.
//
// A layer runs TEN experts against ONE activation, and the expert kernel is DRAM-bound, tracking core count almost
// exactly.  So the pool's job is not to be clever - it is to keep every physical core reading expert bytes for the
// whole layer, and to be cheap enough that ten dispatches per layer cost less than one expert.
//
// WHY A FLAT BATCH AND NOT A RING.  A lock-free SPMC ring is what you need when jobs arrive while others are still
// being computed.  Here they cannot: the host must SUM all ten outputs before the next layer starts, so a layer is a
// barrier by construction and the queue never holds more than one batch.  A ring would add a wrap-around to get
// wrong and buy nothing.  What is kept is the part that matters: `head`/`done` are single fetch_add counters, one
// claim per worker, no lock.
//
// THE COMPLETION PROTOCOL, because this is where a pool usually goes wrong.  `run()` waits for `done == n` AND for
// every worker to PARK.  Waiting only for `done` is not enough: a worker can still be inside the drain loop after its
// last `done` increment, and the host resetting `head` underneath it would let that worker claim a job from the NEXT
// batch before the next batch has been published.
//
// **AND `parked` IS NOT ENOUGH EITHER.**  A worker that went to sleep (after `kSpinBeforeSleep`) is still counted as
// parked when it wakes, so for a moment after it has seen a new epoch the host believes it is idle.  When most experts
// are in VRAM the workers sleep in the middle of a request, tiny batches finish before a sleeper is awake, and that
// moment comes round constantly: a late worker could claim from the NEXT batch while the host was still writing it,
// run a job twice, or add to `done` after the host had reset it - and `done != n` then never ended, with the GPU
// waiting on the pool forever.  So every claim carries its batch: `head` is one 64-bit word `epoch | njobs | index`, a
// claim is a CAS that only succeeds for the epoch the worker woke for, and a late worker's claim simply fails.  A
// claim that succeeds belongs to the current batch, whose description the host cannot change until that job's `done`
// has landed.
#pragma once

#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/platform/cpu.hpp"

#include <atomic>
#include <cstdio>
#include <memory>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace basalt::kernels::cpu {

/// One expert evaluation.  `act` is SHARED and read-only across the whole batch - that sharing is the point
/// of `s2_expert_vnni_q` and it is what saves 480 redundant activation conversions per token.
struct ExpertJob {
    const uint8_t* blob = nullptr;   ///< one 1,382,400-byte expert
    const ActQ* act = nullptr;       ///< the layer's quantized activation, shared
    float* out = nullptr;            ///< H floats, written by exactly one worker
    float weight = 1.0f;             ///< the router weight; applied by the HOST, not here
    int slot = -1;                   ///< the job's index, for diagnostics
};

/// One expert for the `nt` tokens of a verify window that were routed to it.  Every token's output is bitwise the
/// single-token job's.
struct ExpertJobMulti {
    const uint8_t* blob = nullptr;
    int nt = 0;
    const ActQ* act[MAXT] = {};
    float* out[MAXT] = {};
    /// a native pack's activations (the layer's `vec_dot_type`), one per token.
    const void* nact[MAXT] = {};
};

/// How worker threads are allocated across physical/logical CPU cores.  `All` uses every core and is the default; the
/// hybrid-aware ones are opt-in (--pool-affinity auto|p-cores).
using PoolAffinity = platform::CpuAffinity;
using CpuTopology = platform::CpuTopology;

using platform::detect_cpu_topology;

/// One logical processor per PHYSICAL core, so a worker is never scheduled onto an SMT sibling of another worker:
/// `hardware_concurrency()/2` workers on logical processors 0..N/2 could put every worker on a sibling pair and halve
/// the useful bandwidth - which shows up as "the CPU path is slower than the model says" with no clue why.
///
/// `skip_first` drops the first core, which is reserved for the host loop.
std::vector<int> physical_cores(bool skip_first, PoolAffinity affinity = PoolAffinity::All);

class ExpertPool {
public:
    /// `n_workers` is the worker count (at least 1); the caller sizes it (see program/auto_plan).  Workers are not pinned (the OS places them) and each owns one `ExpertScratch`, so nothing in the token path allocates.
    ///
    /// The host thread joins the drain: `run()` claims jobs itself instead of spinning on `done_`, so the pool uses
    /// every core.
    explicit ExpertPool(int n_workers, PoolAffinity affinity = PoolAffinity::All);
    /// The watchdog's view of the pool: the batch, the counters, every thread's state.
    void diag(std::FILE* f) const;
    ~ExpertPool();
    ExpertPool(const ExpertPool&) = delete;
    ExpertPool& operator=(const ExpertPool&) = delete;

    int workers() const { return n_; }
    /// How many of the workers take work.  The rest stay parked and never claim.
    int active_workers() const { return active_.load(std::memory_order_relaxed); }
    /// Change the active worker count live, clamped to 1..workers(); returns the count in effect.  The threads stay as
    /// created.  Host thread only, between batches: it waits for every worker to park first.  Work is split by the
    /// active count, so a given count always cuts the same tasks, and outputs do not depend on the count.
    int set_active_workers(int k);
    bool is_hybrid() const { return topo_.is_hybrid; }
    int p_cores() const { return topo_.p_cores; }
    int p_threads() const { return topo_.p_threads; }
    int e_cores() const { return topo_.e_cores; }
    PoolAffinity affinity() const { return affinity_; }

    /// Publish `n` jobs, then block until every one has been claimed AND every worker has parked.
    /// `jobs` must outlive the call (it does, and the workers never touch it afterwards).
    void run(ExpertJob* jobs, int n);

    /// The same outputs as `run`, bitwise, with every expert split by rows across all threads (gate/up rows, then the
    /// intermediate's quantization, then down rows).  With fewer experts than threads - the case once the VRAM tier
    /// takes half of them - `run` leaves cores idle and each expert streams at one core's bandwidth; this streams every
    /// expert at all of them.  At most `kMaxSplit` experts.
    void run_split(ExpertJob* jobs, int n);
    static constexpr int kMaxSplit = 16;
    /// `run_split` for multi-token jobs (at most `kMaxSplitMulti`); the rows of each expert are read once for all of
    /// its tokens.
    void run_split_multi(ExpertJobMulti* jobs, int n);
    /// the same for a native pack's layer (ggml-cpu arithmetic, `nact` activations).
    void run_split_multi_native(const NativeFmt& f, ExpertJobMulti* jobs, int n);
    static constexpr int kMaxSplitMulti = 96;
    /// Row-range tasks per thread in each phase of the multi-token paths.  The cores differ in speed, so a thread
    /// that finishes early claims another: the more and smaller the tasks, the shorter the wait for the slowest one
    /// at the end of a phase, until the claim traffic and the short streams cost more than the tail they remove.
    static constexpr int kTasksPerThread = 12;
    /// Work for the parked workers between batches (not the host thread): `fn(ctx, worker, workers, stop)` runs once
    /// per worker index and must return soon after `stop` reads true.  Returns at once.  The next batch, `idle_stop`
    /// or the destructor raises `stop` and waits for the workers to park, so the work never overlaps a batch; it must
    /// not write anything a batch reads.
    using IdleFn = void (*)(void* ctx, int worker, int workers, const std::atomic<bool>& stop);
    void idle_start(IdleFn fn, void* ctx);
    void idle_stop();
    /// run_split_multi's phases, accumulated ms: gate/up rows, the intermediate quantization, down rows.
    double ms_multi_gu = 0, ms_multi_q = 0, ms_multi_down = 0;
    int64_t multi_bytes = 0;

    /// Total `_mm_pause` iterations spent waiting are not counted - see the note on the
    /// atomics below.  Counting needs a LOCKED read-modify-write in the park loop, which would add to the
    /// contention being measured.
    long long pauses() const { return 0; }

    /// **WHERE `run()` SPENDS ITS TIME, in milliseconds accumulated over its lifetime.**  Three phases per layer - wait
    /// for every worker to be parked, wait for the drain, wait for them to re-park - reported apart, because a pool
    /// that is slow at the WORK and one that is slow at the SYNCHRONISATION need opposite fixes: the first is a kernel
    /// problem and the second is a barrier problem.
    ///
    /// Only the host thread touches these, in `run()`, so they need no atomics.
    void phase_ms(double& wait_park, double& drain, double& repark) const {
        wait_park = ms_wait_park_;
        drain = ms_drain_;
        repark = ms_repark_;
    }

    /// **A PARKED WORKER SPINS FOR THIS LONG, THEN SLEEPS.**  The park is a `_mm_pause` spin because a layer's batches
    /// are microseconds apart and a wake-up from the OS costs more than that.  But a spin that never ends keeps every
    /// worker's core at 100% while the engine waits for a request.  Between requests the workers block on `sleep_cv_`.
    /// NOT only between requests: with most experts in VRAM many layers have no CPU work, so the workers also sleep
    /// mid-request - which the claim protocol above must survive.
    static constexpr std::chrono::milliseconds kSpinBeforeSleep{20};
    /// A pool wait that sees no completion for this long is a bug; the engine stops with a message instead of
    /// spinning forever, and the server starts it again.
    static constexpr std::chrono::seconds kStall{60};

private:
    void worker(int id);
    void drain(int id, ExpertScratch& scratch, uint32_t epoch);
    void run_phase(int mode, int n_tasks);
    /// Rows [r0, r1) of expert `e` of the native batch: gate/up with SwiGLU into the split buffer, the intermediate's
    /// quantization, and the down projection into the job's outputs.
    void native_gu_range(int e, int r0, int r1);
    void native_quantize(int e);
    void native_down_range(int e, int r0, int r1);
    /// Claim the next job of batch `epoch`, or -1 (that batch is exhausted, or it is not the current one).
    int claim(uint32_t epoch);
    /// Publish the batch whose description the caller has just written: reset `done`, then `head`, then the epoch.
    uint32_t begin_batch(int n);
    /// The host's waits, bounded by `kStall`.
    void wait_parked(const char* what);
    void wait_done(int n);
    /// The wait of a fused batch's down part for its expert's quantized intermediate, bounded the same way.
    void wait_ready(int e);
    /// After a stall was logged: release the GPU's waits, then stop the process.
    [[noreturn]] static void stalled();
    /// Bump `epoch_`, and wake the workers that went to sleep.  Every publish goes through here.
    void publish();

    int n_ = 0;
    /// Workers with id < active_ take work; written only while every worker is parked, before the next publish.
    std::atomic<int> active_{0};
    ExpertJob* jobs_ = nullptr;
    int njobs_ = 0;
    /// The host's own scratch .  A separate object rather than a share of `scratch_[i]`,
    /// because a worker may own any index and the two must not be able to collide.
    ExpertScratch host_scratch_;
    // `run()`'s three phases, accumulated.  Host-thread only; see `phase_ms`.
    double ms_wait_park_ = 0.0;
    double ms_drain_ = 0.0;
    double ms_repark_ = 0.0;
    // ---- EACH ATOMIC GETS ITS OWN CACHE LINE.
    //
    // `head_`, `done_`, `parked_` and `epoch_` are the four that workers and the host actually contend on; sharing a
    // cache line they would invalidate each other on every access.  The line the host must WRITE to publish work
    // (`epoch_`) and READ to confirm the workers are parked (`parked_`) must not be hammered by the very threads
    // waiting for it, so the parked spin carries no counter (a locked read-modify-write per iteration) and
    // `alignas(64)` stops the four sharing.
    // Diagnostics: each worker's state (kParked, kSleeping, kBetween, or the job it runs) and the host's (kIdle,
    // kWaitParked, kWaitDone, or its job), printed by the serve watchdog through `diag`
    static constexpr int32_t kParked = -1, kSleeping = -2, kBetween = -3, kIdle = -10, kWaitParked = -11,
                             kWaitDone = -12;
    std::unique_ptr<std::atomic<int32_t>[]> wstate_;
    std::atomic<int32_t> hstate_{kIdle};
    std::atomic<int64_t> hstate_ms_{0};
    alignas(64) std::atomic<uint64_t> head_{0};   // epoch << 32 | njobs << 16 | next index
    alignas(64) std::atomic<uint32_t> done_{0};
    alignas(64) std::atomic<uint32_t> parked_{0};
    alignas(64) std::atomic<uint32_t> epoch_{0};
    alignas(64) std::atomic<bool> stop_{false};
    // The sleep after `kSpinBeforeSleep`.  `sleepers_` is how `publish` knows whether anyone needs waking, so the
    // token path pays one uncontended load per publish and never takes the mutex while the workers spin.
    alignas(64) std::atomic<uint32_t> sleepers_{0};
    std::mutex sleep_mu_;
    std::condition_variable sleep_cv_;
    std::chrono::microseconds spin_before_sleep_{kSpinBeforeSleep};
    std::vector<std::thread> threads_;
    std::vector<ExpertScratch> scratch_;   // one per worker: no allocation, no false sharing of the hot data
    // idle work (mode 7): one job per worker index
    IdleFn idle_fn_ = nullptr;
    void* idle_ctx_ = nullptr;
    bool idle_live_ = false;
    alignas(64) std::atomic<bool> idle_halt_{false};
    // run_split state: mode 0 = whole experts, 1 = gate/up row parts, 2 = down row parts
    int mode_ = 0;
    int parts_a_ = 1, parts_b_ = 1;
    struct SplitBuf {
        alignas(64) float ff[FF];
        ActQ a2;
    };
    std::vector<SplitBuf> split_;
    // run_split_multi state: mode 3 = gate/up row parts, 4 = down row parts (mode 8, the native path, is below)
    ExpertJobMulti* mjobs_ = nullptr;
    int64_t mrows_ = 0;   // rows of the current multi phase across all its experts (n * FF, then n * H)
    int mtasks_ = 1;   // equal row ranges the phase is cut into
    struct SplitBufMulti {
        alignas(64) float ff[MAXT][FF];
        ActQ a2[MAXT];
        alignas(64) uint8_t hq[MAXT][kNativeHBytes];   // native down activations
    };
    const NativeFmt* nfmt_ = nullptr;
    // run_split_multi_native's one batch per layer (mode 8): per expert, the gate/up parts left to do and whether its
    // intermediate is quantized
    struct alignas(64) FusedExpert {
        std::atomic<int> gu_left{0};
        std::atomic<int> ready{0};
    };
    std::unique_ptr<FusedExpert[]> fused_;
    int fused_n_ = 0, fused_gu_parts_ = 1, fused_d_parts_ = 1;
    std::vector<SplitBufMulti> split_multi_;
    PoolAffinity affinity_ = PoolAffinity::All;
    CpuTopology topo_;
};

}  // namespace basalt::kernels::cpu
