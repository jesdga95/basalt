// basalt/program/engine_local.hpp - program-level helpers and shapes used by more than one translation unit.
// Private to src/program: nothing here is part of the engine's public surface.
#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "basalt/core/expert_cache.hpp"
#include "basalt/core/native_head.hpp"
#include "basalt/platform/compiler.hpp"
#include "basalt/core/native_dense.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/core/session.hpp"
#include "basalt/core/staged_copy.hpp"
#include "basalt/core/verify.hpp"
#include "basalt/core/weights.hpp"
#include "basalt/prefill/prefill.hpp"

#include "basalt/program/multi_gpu_plan.hpp"   // PsPrice

namespace basalt::program::engine_local {

/// One candidate split price: the plan's cost, the prompt chunk and the head chunk it was measured with, and when the
/// part is not priced (chunk 0) why not.
struct SplitPricing {
    basalt::program::PsPrice price;
    int64_t chunk = 0, head = 0;
    std::string why;
};

/// The largest multiple of 256 in [0, ceiling] that `ok` accepts.  `ok` is monotone: false once too large.
template <class Ok>
int64_t biggest_chunk(int64_t ceiling, Ok&& ok) {
    if (ceiling < 256) return 0;
    int64_t lo = 1, hi = ceiling / 256, best = 0;   // n = T / 256, and ok() is monotone in n
    while (lo <= hi) {
        const int64_t mid = lo + (hi - lo) / 2;
        if (ok(mid * 256)) { best = mid * 256; lo = mid + 1; } else hi = mid - 1;
    }
    return best;
}

// One unit of background work at a time, on a persistent worker thread (starting a thread per job costs more than
// the job's wake-up).  `join` waits for the job in flight.
class JobRunner {
public:
    JobRunner() = default;
    JobRunner(const JobRunner&) = delete;
    JobRunner& operator=(const JobRunner&) = delete;
    ~JobRunner() {
        join();
        if (th_.joinable()) {
            { std::lock_guard<std::mutex> lk(m_); stop_ = true; }
            cv_.notify_one();
            th_.join();
        }
    }
    void run(std::function<void()> fn) {
        if (!th_.joinable()) th_ = std::thread([this] { loop(); });
        {
            std::lock_guard<std::mutex> lk(m_);
            fn_ = std::move(fn);
            done_.store(false, std::memory_order_relaxed);
            has_ = true;
        }
        busy_ = true;
        cv_.notify_one();
    }
    bool joinable() const { return busy_; }
    bool finished() const { return !busy_ || done_.load(std::memory_order_acquire); }
    void join() {
        if (!busy_) return;
        // the caller has nothing else to do: poll rather than sleep, so the job's end is seen at once
        for (unsigned spins = 0; !done_.load(std::memory_order_acquire); ++spins) {
            if (spins < 20000) basalt::platform::cpu_relax();
            else std::this_thread::yield();
        }
        busy_ = false;
    }
private:
    void loop() {
        std::unique_lock<std::mutex> lk(m_);
        for (;;) {
            cv_.wait(lk, [this] { return has_ || stop_; });
            if (!has_) return;
            std::function<void()> fn = std::move(fn_);
            has_ = false;
            lk.unlock();
            fn();
            done_.store(true, std::memory_order_release);
            lk.lock();
        }
    }
    bool busy_ = false;
    bool has_ = false, stop_ = false;
    std::atomic<bool> done_{true};
    std::mutex m_;
    std::condition_variable cv_;
    std::function<void()> fn_;
    std::thread th_;
};
/// The pool's adapter plus the wall-clock it spent, so the report can say how much of the token was the CPU.
struct Drive {
    basalt::core::ExpertDispatch d;
    double cpu_ms = 0;
    int64_t calls = 0;
    /// THE ROUTING TRACE.  `drive_pool` is called once per layer from the main loop - the workers live inside
    /// `expert_pool_dispatch` - so a single FILE* here needs no locking.  `d.layers` is the CURRENT layer on entry (the
    /// adapter increments it as it walks the blob), which is why the layer index comes from there rather than from a
    /// counter of our own.
    std::FILE* routing = nullptr;
};
/// Layer split: every verify stage shares one Drive (its counters, usage and failure flags); the GPU plan, the expert
/// cache and the PCIe share the pool uses for a layer are those of the stage that runs it.
struct SplitDrive {
    static constexpr int kMax = 8;
    Drive* base = nullptr;
    int n = 0;                                    ///< stages
    int64_t end[kMax] = {};                       ///< stage i runs the layers from end[i - 1] (0) below end[i]
    basalt::core::GpuPlanSink* plan[kMax] = {};
    const uint8_t* cache_base[kMax] = {};
    const uint64_t* cache_slot_off[kMax] = {};
    int pcie_num[kMax] = {};
};
struct SlotDelete {
    bool own = true;
    void operator()(basalt::core::SessionState* p) const { if (own) delete p; }
};
using SlotSession = std::unique_ptr<basalt::core::SessionState, SlotDelete>;

struct GpuStage {
    int dev = 0;
    int64_t lb = 0, le = 0;
    double pcie_frac = 0.0;
    basalt::core::WeightTable wt;
    basalt::core::NativeDense dense;
    basalt::core::NativeHead head;
    basalt::core::SessionState ss;
    cudaStream_t stream = nullptr;
    basalt::core::ExpertCache cache;
    std::vector<std::pair<int32_t, int32_t>> profile;   ///< its layers' share of the profile, hottest first
    int32_t* d_res = nullptr;                            ///< the residency table on its device
    basalt::core::Verifier ver;
    basalt::core::Verifier ver_b;                        ///< the second chain's verifier (pipelined windows)
    basalt::prefill::Prefill sp;
    cudaStream_t adapt_stream = nullptr;
    cudaEvent_t adapt_ev = nullptr;
    bool adapt_live = false;                             ///< swaps of this request are in flight on it
    int32_t* mrope = nullptr;                            ///< --vision: the image-position table on its device
};
/// One stage of a prompt-time layer split over the expert tier cards: for a long prompt the main card reads layers
/// [0, b1) of every chunk and hands the chunk on, each tier card's stage reads the next block of layers of it while the
/// cards before read the next chunks, and the main card reads the last chunk's later layers itself once the stages have
/// given their layers' state back.  A stage's dense weights (an image in pinned RAM), session and prompt buffers live in
/// its tier's last slots, lent for the prompt and refilled after it, so outside a prompt the tier is as before.
struct PromptStage {
    int dev = -1;
    size_t tier_at = 0;                   ///< its tier among the expert tiers (0: the first)
    basalt::core::PeerExperts* tier = nullptr;
    double mem_gbps = 0, link_gbps = 0;   ///< its card's memory bandwidth and probed host->device link
    int64_t lo = 0, hi = 0;               ///< its weights cover layers [lo, hi); a read runs a range inside them
    int64_t l0 = 0, l1 = 0;               ///< the layers of the read in progress
    std::set<std::string> skip;           ///< the tensors outside its layers
    uint64_t pool = 0;                    ///< its arena's bytes
    basalt::core::WeightTable wt;         ///< its rows point into the lent slots, where the image is copied
    basalt::core::NativeDense dense;      ///< its native projections' rows (in the image) and their scratch
    uint8_t* image = nullptr;             ///< pinned: the weights' bytes, the arena then the native blocks
    uint64_t arena_bytes = 0, image_bytes = 0;
    basalt::core::SessionState ss;        ///< carved per prompt in the lent slots
    basalt::prefill::Prefill sp;
    cudaStream_t stream = nullptr;
    basalt::core::StagedCopy state_in, state_back;   ///< its layers' state from the main session and back to it
    std::vector<int32_t> res;             ///< its layers' experts in the tier's slots (n_layers x n_expert, -1 elsewhere)
    std::vector<std::pair<int32_t, int32_t>> lent;
    int64_t lent_slots = 0;
    uint8_t* buf = nullptr;               ///< the read's prompt buffers, `buf_bytes` long, below its session
    uint64_t buf_bytes = 0;
};

}  // namespace basalt::program::engine_local
