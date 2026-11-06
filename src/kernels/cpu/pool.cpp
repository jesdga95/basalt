// src/kernels/cpu/pool.cpp - the CPU expert pool.  Read pool.hpp first; it explains the protocol.
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/core/progress.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <immintrin.h>

#include <cstdio>
#include <cstdlib>
#include "basalt/log.hpp"

namespace basalt::kernels::cpu {

namespace {
constexpr uint64_t pack_head(uint32_t epoch, uint32_t n, uint32_t i) {
    return ((uint64_t) epoch << 32) | ((uint64_t) n << 16) | (uint64_t) i;
}
}  // namespace

std::vector<int> physical_cores(bool skip_first, PoolAffinity affinity) {
    return detect_cpu_topology(skip_first, affinity).worker_cores;
}

namespace {
/// The clock of one wait: the time is read every 1024 spins, and `progress` (any number that changes when the wait is
/// getting somewhere) restarts it.  `expired` is true once ExpertPool::kStall has passed without any.  It is armed by its
/// first reading, so it does not depend on how many spins that takes to come round.
class StallClock {
public:
    bool expired(uint32_t progress) {
        if ((++spins_ & 1023u) != 0) return false;
        const auto now = std::chrono::steady_clock::now();
        if (!armed_ || progress != seen_) {
            armed_ = true;
            seen_ = progress;
            t0_ = now;
            return false;
        }
        return now - t0_ > ExpertPool::kStall;
    }

private:
    uint32_t spins_ = 0, seen_ = 0;
    bool armed_ = false;
    std::chrono::steady_clock::time_point t0_{};
};

std::atomic<const ExpertPool*> g_diag_pool{nullptr};
void diag_active_pool(std::FILE* f) {
    if (const ExpertPool* p = g_diag_pool.load()) p->diag(f);
}
int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
}  // namespace

void ExpertPool::diag(std::FILE* f) const {
    const uint64_t h = head_.load();
    std::fprintf(f, "  expert pool: epoch %u, batch epoch %u: %u of %u jobs claimed, %u done; %u of %d workers parked, "
                    "%u sleeping; mode %d\n", epoch_.load(), (uint32_t) (h >> 32), (uint32_t) h & 0xffffu,
                 (uint32_t) (h >> 16) & 0xffffu, done_.load(), parked_.load(), n_, sleepers_.load(), mode_);
    std::fprintf(f, "  expert pool active workers: %d\n", active_.load());
    std::fprintf(f, "  expert pool threads:");
    for (int i = 0; i < n_; ++i) {
        const int32_t s = wstate_[(size_t) i].load();
        if (s == kParked) std::fprintf(f, " w%d=parked", i);
        else if (s == kSleeping) std::fprintf(f, " w%d=sleeping", i);
        else if (s == kBetween) std::fprintf(f, " w%d=draining", i);
        else std::fprintf(f, " w%d=job%d", i, s);
    }
    const int32_t hs = hstate_.load();
    const char* hn = hs == kIdle ? "idle" : hs == kWaitParked ? "waiting for the workers to park"
                   : hs == kWaitDone ? "waiting for the jobs to finish" : "running a job";
    std::fprintf(f, "; host %s", hn);
    if (hs >= 0) std::fprintf(f, " %d", hs);
    std::fprintf(f, " for %lld ms\n", (long long) (now_ms() - hstate_ms_.load()));
}

ExpertPool::ExpertPool(int n_workers, PoolAffinity affinity)
    : affinity_(affinity), topo_(detect_cpu_topology(true, affinity)) {
    n_ = n_workers;
    if (n_ < 1) n_ = 1;
    active_.store(n_);
    wstate_.reset(new std::atomic<int32_t>[(size_t) n_]);
    for (int i = 0; i < n_; ++i) wstate_[(size_t) i].store(kParked);
    hstate_ms_.store(now_ms());
    g_diag_pool.store(this);
    basalt::core::diag_pool_fn().store(&diag_active_pool);
    split_multi_.resize((size_t) kMaxSplitMulti);
    fused_.reset(new FusedExpert[(size_t) kMaxSplitMulti]);
    threads_.reserve((size_t) n_);
    for (int i = 0; i < n_; ++i) {
        threads_.emplace_back([this, i] { worker(i); });
    }
}

ExpertPool::~ExpertPool() {
    const ExpertPool* self = this;
    g_diag_pool.compare_exchange_strong(self, nullptr);
    idle_stop();
    stop_.store(true, std::memory_order_release);
    // Bump the epoch so a PARKED worker notices the stop flag rather than sleeping through it.
    publish();
    for (auto& t : threads_) t.join();
}

void ExpertPool::publish() {
    // Both sides are seq_cst, and that is the whole lost-wakeup argument: a worker going to sleep does
    // `sleepers_++` and then reads `epoch_`, the host does `epoch_++` and then reads `sleepers_`.  In one total
    // order at least one of them sees the other's write - the worker sees the new epoch and does not sleep, or
    // the host sees the sleeper and notifies under the mutex the worker holds until it is inside `wait`.
    // On x86 the fetch_add is a locked xadd either way, so this costs the token path nothing.
    epoch_.fetch_add(1, std::memory_order_seq_cst);
    if (sleepers_.load(std::memory_order_seq_cst) != 0) {
        std::lock_guard<std::mutex> lk(sleep_mu_);
        sleep_cv_.notify_all();
    }
}

void ExpertPool::worker(int id) {
    uint32_t seen = 0;
    // ARRIVE at the park before the first wait, so `parked_ == n_` is true from construction: a
    // `set_active_workers` before the first batch waits for `parked_ == n_` and would otherwise never return.
    parked_.fetch_add(1, std::memory_order_acq_rel);
    for (;;) {
        // Park: wait for work.  `_mm_pause` rather than a bare spin because it yields the pipeline to the sibling
        // hyperthread; `epoch_` is bumped once per LAYER, not once per expert, so most of these iterations are spent
        // here with nothing to do.
        //
        // **AND NOTHING ELSE HAPPENS IN HERE**: no counter is bumped per iteration (a locked read-modify-write, five
        // workers against one cache line, would make the workers invalidate each other's caches and the very line the
        // host writes to publish work).  See the note on the atomics in pool.hpp.
        //
        // After `kSpinBeforeYield` the worker yields its core on every check, and after `kSpinBeforeSleep` with no
        // work it sleeps.  The clock is read once every 64 checks.
        const auto parked_at = std::chrono::steady_clock::now();
        uint32_t spins = 0;
        bool yielding = false;
        while (epoch_.load(std::memory_order_acquire) == seen) {
            if (stop_.load(std::memory_order_relaxed)) return;
            if (yielding) std::this_thread::yield();
            else _mm_pause();
            if ((++spins & 63u) != 0) continue;
            const auto waited = std::chrono::steady_clock::now() - parked_at;
            yielding = waited >= kSpinBeforeYield;
            if (waited < spin_before_sleep_) continue;
            std::unique_lock<std::mutex> lk(sleep_mu_);
            wstate_[(size_t) id].store(kSleeping, std::memory_order_relaxed);
            sleepers_.fetch_add(1, std::memory_order_seq_cst);
            sleep_cv_.wait(lk, [&] {
                return epoch_.load(std::memory_order_seq_cst) != seen || stop_.load(std::memory_order_relaxed);
            });
            sleepers_.fetch_sub(1, std::memory_order_relaxed);
            wstate_[(size_t) id].store(kParked, std::memory_order_relaxed);
        }
        if (stop_.load(std::memory_order_acquire)) return;
        // acquire: the batch this epoch published (`head`, and the description before it) is visible from here
        seen = epoch_.load(std::memory_order_acquire);
        // An inactive worker stays counted as parked and claims nothing; it just waits for the next epoch.
        if (id >= active_.load(std::memory_order_acquire)) continue;
        parked_.fetch_sub(1, std::memory_order_acq_rel);   // leaving the park

        // Drain: one claim per iteration, so a slow worker takes fewer experts and a fast one takes more, with no
        // schedule made ahead.  Only this epoch's jobs: if the host has already moved on, the claims fail and the
        // worker parks again.
        wstate_[(size_t) id].store(kBetween, std::memory_order_relaxed);
        drain(id, seen);
        wstate_[(size_t) id].store(kParked, std::memory_order_relaxed);
        parked_.fetch_add(1, std::memory_order_acq_rel);   // back at the park
    }
}

int ExpertPool::claim(uint32_t epoch) {
    uint64_t h = head_.load(std::memory_order_acquire);
    for (;;) {
        if ((uint32_t) (h >> 32) != epoch) return -1;               // not the batch this thread woke for
        const uint32_t n = (uint32_t) (h >> 16) & 0xffffu, i = (uint32_t) h & 0xffffu;
        if (i >= n) return -1;                                       // exhausted
        if (head_.compare_exchange_weak(h, h + 1, std::memory_order_acq_rel, std::memory_order_acquire))
            return (int) i;
    }
}

uint32_t ExpertPool::begin_batch(int n) {
    if (n < 0 || n > 0xffff) {
        basalt::log::error("basalt", "expert pool batch of %d jobs is out of range", n);
        std::abort();
    }
    // Every job of the previous batch has completed (`wait_done`), and a claim of it can no longer succeed, so
    // nothing adds to `done` until this batch's first claim - which the release below orders after the reset.
    done_.store(0, std::memory_order_relaxed);
    const uint32_t e = epoch_.load(std::memory_order_relaxed) + 1;   // only the host bumps the epoch
    head_.store(pack_head(e, (uint32_t) n, 0), std::memory_order_release);
    publish();
    return e;
}

void ExpertPool::stalled() {
    basalt::core::release_gpu_waits(basalt::log::stream());   // the GPU may be spinning on this layer's flag
    std::abort();
}

void ExpertPool::wait_parked(const char* what) {
    hstate_.store(kWaitParked, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
    StallClock clock;
    while (parked_.load(std::memory_order_acquire) != (uint32_t) n_) {
        _mm_pause();
        if (clock.expired(parked_.load(std::memory_order_relaxed))) {
            basalt::log::error("basalt", "the CPU expert pool stalled %s (%u of %d workers parked) - stopping the engine "
                                 "so the server can start it again",
                         what, parked_.load(), n_);
            stalled();
        }
    }
}

void ExpertPool::wait_done(int n) {
    hstate_.store(kWaitDone, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
    StallClock clock;
    for (;;) {
        const uint32_t d = done_.load(std::memory_order_acquire);
        if (d >= (uint32_t) n) return;                                // `>=`: never a wait that an overshoot outlives
        _mm_pause();
        if (clock.expired(d)) {                                       // progress restarts the clock
            basalt::log::error("basalt", "the CPU expert pool stalled: %u of %d jobs done, %u of %d workers parked - "
                                 "stopping the engine so the server can start it again",
                         d, n, parked_.load(), n_);
            stalled();
        }
    }
}

void ExpertPool::wait_ready(int e) {
    StallClock clock;
    while (fused_[(size_t) e].ready.load(std::memory_order_acquire) == 0) {
        _mm_pause();
        if (clock.expired(done_.load(std::memory_order_relaxed))) {
            basalt::log::error("basalt", "the CPU expert pool stalled: the gate/up rows of expert %d of the batch did not finish "
                                 "(%u of %d tasks done, %u of %d workers parked) - stopping the engine so the server can "
                                 "start it again",
                         e, done_.load(), njobs_, parked_.load(), n_);
            stalled();
        }
    }
}

void ExpertPool::drain(int id, uint32_t epoch) {
    (void) id;
    for (;;) {
        const int ci = claim(epoch);
        if (ci < 0) break;
        const uint32_t i = (uint32_t) ci;
        if (id >= 0) wstate_[(size_t) id].store(ci, std::memory_order_relaxed);
        else { hstate_.store(ci, std::memory_order_relaxed); hstate_ms_.store(now_ms(), std::memory_order_relaxed); }
        if (mode_ == 7) {
            idle_fn_(idle_ctx_, (int) i, active_.load(std::memory_order_relaxed), idle_halt_);
        } else {
            // a native layer in one batch: gate/up parts, then down parts, each down part after its expert's quantize
            const int n = fused_n_, pg = fused_gu_parts_, pd = fused_d_parts_;
            if ((int) i < n * pg) {
                const int e = (int) i / pg, part = (int) i % pg;
                native_gu_range(e, (int) nfmt_->n_ff * part / pg, (int) nfmt_->n_ff * (part + 1) / pg);
                FusedExpert& fe = fused_[(size_t) e];
                if (fe.gu_left.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    native_quantize(e);
                    fe.ready.store(1, std::memory_order_release);
                }
            } else {
                const int k = (int) i - n * pg, e = k / pd, part = k % pd;
                // every gate/up task is claimed before any down task, so this waits only for tasks already running
                if (fused_[(size_t) e].ready.load(std::memory_order_acquire) == 0) wait_ready(e);
                native_down_range(e, (int) nfmt_->n_embd * part / pd, (int) nfmt_->n_embd * (part + 1) / pd);
            }
        }
        done_.fetch_add(1, std::memory_order_release);
    }
}

void ExpertPool::native_gu_range(int e, int r0, int r1) {
    SplitBufMulti& sb = split_multi_[(size_t) e];
    const ExpertJobMulti& j = mjobs_[e];
    if (q2_native_kernels(nfmt_->gu_type)) {
        // a native Q2_0 pack: gate and up rows on the Q2_0 kernels, then SwiGLU
        thread_local float gbuf[MAXT][kNativeMaxFF], ubuf[MAXT][kNativeMaxFF];
        float* gp[MAXT];
        float* up[MAXT];
        for (int t = 0; t < j.nt; ++t) { gp[t] = gbuf[t]; up[t] = ubuf[t]; }
        const int nbk = (int) (nfmt_->n_embd / 64);
        q2_rows_any(j.blob, nfmt_->gu_row, nbk, j.act, j.nt, gp, r0, r1);
        q2_rows_any(j.blob + nfmt_->up_off, nfmt_->gu_row, nbk, j.act, j.nt, up, r0, r1);
        for (int t = 0; t < j.nt; ++t)
            for (int r = r0; r < r1; ++r)
                sb.ff[t][r] = (gbuf[t][r] / (1.f + std::exp(-gbuf[t][r]))) * ubuf[t][r];
        return;
    }
    float* ff[MAXT];
    for (int t = 0; t < j.nt; ++t) ff[t] = sb.ff[t];
    native_gu_rows(*nfmt_, j.blob, j.nact, j.nt, ff, r0, r1);
}

void ExpertPool::native_down_range(int e, int r0, int r1) {
    SplitBufMulti& sb = split_multi_[(size_t) e];
    const ExpertJobMulti& j = mjobs_[e];
    if (q2_native_kernels(nfmt_->d_type)) {
        // Q2_0 down (most IQ layers): the AVX-512 kernel, ggml-cpu has only a scalar one on x86
        const ActQ* a2[MAXT];
        for (int t = 0; t < j.nt; ++t) a2[t] = &sb.a2[t];
        q2_rows_any(j.blob + nfmt_->down_off, nfmt_->d_row, (int) (nfmt_->n_ff / 64), a2, j.nt, j.out, r0, r1);
        return;
    }
    const void* hq[MAXT];
    for (int t = 0; t < j.nt; ++t) hq[t] = sb.hq[t];
    native_down_rows(*nfmt_, j.blob, hq, j.nt, j.out, r0, r1);
}

void ExpertPool::native_quantize(int e) {
    SplitBufMulti& sb = split_multi_[(size_t) e];
    for (int t = 0; t < mjobs_[e].nt; ++t)
        if (q2_native_kernels(nfmt_->d_type)) act_quant_any(sb.ff[t], (int) nfmt_->n_ff, sb.a2[t]);
        else native_quant_h(*nfmt_, sb.ff[t], sb.hq[t]);
}

int ExpertPool::set_active_workers(int k) {
    k = (std::min)((std::max)(k, 1), n_);
    idle_stop();
    wait_parked("before changing the active workers");
    active_.store(k, std::memory_order_release);
    return k;
}

void ExpertPool::idle_start(IdleFn fn, void* ctx) {
    if (n_ == 1 || fn == nullptr) return;
    idle_stop();
    idle_fn_ = fn;
    idle_ctx_ = ctx;
    idle_halt_.store(false, std::memory_order_relaxed);
    mode_ = 7;
    const int act = active_.load(std::memory_order_relaxed);
    njobs_ = act;
    begin_batch(act);   // the host does not drain it
    idle_live_ = true;
}

void ExpertPool::idle_stop() {
    if (!idle_live_) return;
    idle_halt_.store(true, std::memory_order_release);
    // no job of the idle batch can be claimed from here on (a late worker's claim fails), and the claimed ones end
    // before the next batch resets `done`
    uint64_t h = head_.load(std::memory_order_acquire);
    uint32_t claimed = 0;
    for (;;) {
        const uint32_t n = (uint32_t) (h >> 16) & 0xffffu, i = (uint32_t) h & 0xffffu;
        claimed = i < n ? i : n;
        if (i >= n || head_.compare_exchange_weak(h, (h & ~(uint64_t) 0xffffu) | n, std::memory_order_acq_rel,
                                                  std::memory_order_acquire))
            break;
    }
    wait_done((int) claimed);
    idle_live_ = false;
    mode_ = 0;
}

void ExpertPool::run_phase(int mode, int n_tasks) {
    idle_stop();
    mode_ = mode;
    njobs_ = n_tasks;
    const uint32_t e = begin_batch(n_tasks);
    drain(-1, e);
    wait_done(n_tasks);
    hstate_.store(kIdle, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
}

int ExpertPool::split_multi_batch(const NativeFmt& f, ExpertJobMulti* jobs, int nb, double& gu_share) {
    mjobs_ = jobs;
    nfmt_ = &f;
    const int threads = active_.load(std::memory_order_relaxed) + 1;
    // One batch for the whole layer: an expert's down rows start once its own gate/up rows are done and quantized,
    // so there is no barrier between the three steps and the cores' speed differences cost one tail, not two.
    // The share of tasks follows the share of bytes of the two projections.
    gu_share = (double) (2 * f.up_off) / (double) (2 * f.up_off + f.d_row * (size_t) f.n_embd);
    fused_n_ = nb;
    fused_gu_parts_ = (std::min)((int) f.n_ff, (std::max)(1, (int) std::lround(kTasksPerThread * threads * gu_share / nb)));
    fused_d_parts_ = (std::min)((int) f.n_embd, (std::max)(1, (int) std::lround(kTasksPerThread * threads * (1.0 - gu_share) / nb)));
    for (int e = 0; e < nb; ++e) {
        fused_[(size_t) e].gu_left.store(fused_gu_parts_, std::memory_order_relaxed);
        fused_[(size_t) e].ready.store(0, std::memory_order_relaxed);
    }
    return nb * (fused_gu_parts_ + fused_d_parts_);
}

void ExpertPool::start_split_multi_native(const NativeFmt& f, ExpertJobMulti* jobs, int n) {
    pend_tasks_ = 0;
    if (n <= 0) return;
    if (n > kMaxSplitMulti) {
        run_split_multi_native(f, jobs, n);
        return;
    }
    pend_t0_ = std::chrono::steady_clock::now();
    const int tasks = split_multi_batch(f, jobs, n, pend_gu_share_);
    idle_stop();
    mode_ = 8;
    njobs_ = tasks;
    pend_epoch_ = begin_batch(tasks);
    pend_tasks_ = tasks;
    pend_n_ = n;
}

void ExpertPool::finish_split_multi_native() {
    if (pend_tasks_ <= 0) return;
    drain(-1, pend_epoch_);
    wait_done(pend_tasks_);
    hstate_.store(kIdle, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - pend_t0_).count();
    ms_multi_gu += ms * pend_gu_share_;
    ms_multi_down += ms * (1.0 - pend_gu_share_);
    multi_bytes += (int64_t) pend_n_ * (int64_t) nfmt_->bytes;
    mode_ = 0;
    pend_tasks_ = 0;
}

void ExpertPool::run_split_multi_native(const NativeFmt& f, ExpertJobMulti* jobs, int n) {
    if (n <= 0) return;
    // more distinct experts than buffers: run them in batches
    for (int b0 = 0; b0 < n; b0 += kMaxSplitMulti) {
        const int nb = (std::min)(kMaxSplitMulti, n - b0);
        const auto a = std::chrono::steady_clock::now();
        double gu_share = 0;
        run_phase(8, split_multi_batch(f, jobs + b0, nb, gu_share));
        // the three steps are not timed apart: the batch's time is shared out by the bytes of the two projections
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
        ms_multi_gu += ms * gu_share;
        ms_multi_down += ms * (1.0 - gu_share);
    }
    multi_bytes += (int64_t) n * (int64_t) f.bytes;
    mode_ = 0;
}

}  // namespace basalt::kernels::cpu
