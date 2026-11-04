#pragma once

#include "common.hpp"
#include "basalt/prefill/prefill.hpp"
#include "basalt/core/mtp.hpp"
#include "basalt/core/progress.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/core/layer.hpp"
#include "basalt/core/native_head.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"
#include "basalt/log.hpp"
#include <algorithm>
#include <atomic>
#include <cstring>
#include <thread>
#include <utility>
#include <vector>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <cerrno>

namespace basalt::prefill {
using namespace detail;

// The experts the arena could not pin are copied into pinned buffers by these threads, ahead of the launches.  Copied
// in line by the launching thread they would leave the GPU without queued work while each memcpy runs.  Job j - a layer's
// j-th unpinned expert, in launch order - lands in host buffer j % kRing, which is free again once the DMA of job
// j - kRing (recorded by the launching thread, `issued`) is done.
struct Stager {
    // the pinned ring's depth: how far the host copies can run ahead of the DMAs of the unpinned experts' blobs
    int kRing = 32;
    // how many jobs ahead of the one being copied the OS is asked to read from the file
    static constexpr size_t kAhead = 64;
    // a job whose bytes are only in the model file (fd >= 0) is read into the buffer with pread at `off`; the others
    // are copied from `src`
    struct Job { const uint8_t* src; size_t bytes; int fd = -1; uint64_t off = 0; };
    // reader threads for the file-backed jobs: each blocks in its read, so the threads are the reads in flight
    static constexpr int kIoThreads = 16;
    std::vector<uint8_t*> buf;
    std::vector<char> pinned;
    std::vector<std::vector<uint8_t>> pageable;   // the fallback when no more RAM can be pinned
    std::vector<cudaEvent_t> dma_done;
    std::vector<Job> jobs;
    std::unique_ptr<std::atomic<int>[]> ready;
    size_t ready_cap = 0;
    // gen << 32 | n << 16 | next index: a claim is a CAS on the generation it woke for (a thread late from the
    // previous layer can never take a job of this one)
    std::atomic<uint64_t> head{0};
    std::atomic<int> issued{0}, active{0};
    uint32_t gen = 0;
    bool quit = false;
    std::mutex mu;
    std::condition_variable cv;
    std::vector<std::thread> threads;
    int device = 0;

    bool init(size_t blob_bytes, int nthreads) {
        buf.assign((size_t) kRing, nullptr);
        pinned.assign((size_t) kRing, 0);
        dma_done.assign((size_t) kRing, nullptr);
        pageable.resize(kRing);
        for (int i = 0; i < kRing; ++i) {
            pinned[i] = cudaHostAlloc((void**) &buf[i], blob_bytes, cudaHostAllocDefault) == cudaSuccess;
            if (!pinned[i]) {
                cudaGetLastError();
                pageable[(size_t) i].resize(blob_bytes);
                buf[i] = pageable[(size_t) i].data();
            }
            if (cudaEventCreateWithFlags(&dma_done[i], cudaEventDisableTiming) != cudaSuccess) return false;
        }
        cudaGetDevice(&device);
        for (int t = 0; t < nthreads; ++t) threads.emplace_back([this] { work(); });
        return true;
    }
    ~Stager() {
        finish();
        { std::lock_guard<std::mutex> lk(mu); quit = true; }
        cv.notify_all();
        for (auto& t : threads) t.join();
        for (int i = 0; i < kRing; ++i) {
            if (dma_done[i]) cudaEventDestroy(dma_done[i]);
            if (buf[i] && pinned[i]) cudaFreeHost(buf[i]);
        }
    }
    void work() {
        cudaSetDevice(device);
        uint32_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return quit || gen != seen; });
                if (quit) return;
                seen = gen;
            }
            for (;;) {
                active.fetch_add(1, std::memory_order_acq_rel);
                const int j = claim(seen);
                if (j < 0) { active.fetch_sub(1, std::memory_order_acq_rel); break; }
                const int b = j % kRing;
                if (j >= kRing)   // job j - kRing's DMA from this buffer is queued
                    while (issued.load(std::memory_order_acquire) <= j - kRing) std::this_thread::yield();
                // and done - for a generation's first kRing jobs that is the previous generation's last DMA from
                // the buffer, which nothing else waits for when a chunk ends without a sync (no MTP) or the DMA
                // was a ring entry the routing skipped (an event never recorded returns at once)
                cudaEventSynchronize(dma_done[b]);
                const Job& jb = jobs[(size_t) j];
                if (jb.fd >= 0) {
                    if (!read_file(jb, buf[b])) std::memcpy(buf[b], jb.src, jb.bytes);   // the mapping as the fallback
                } else {
                    const size_t a = (size_t) j + kAhead;
                    if (a < jobs.size() && jobs[a].fd < 0) basalt::platform::advise_willneed(jobs[a].src, jobs[a].bytes);
                    std::memcpy(buf[b], jb.src, jb.bytes);
                }
                ready[(size_t) j].store(1, std::memory_order_release);
                active.fetch_sub(1, std::memory_order_acq_rel);
            }
        }
    }
    static bool read_file(const Job& jb, uint8_t* dst) {
        size_t done = 0;
        while (done < jb.bytes) {
            // One chunk per call, so a count short of the rest is normal and this loop picks up where it stopped.
            const int64_t n = basalt::platform::pread(jb.fd, dst + done, jb.bytes - done, jb.off + done);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) return false;
            done += (size_t) n;
        }
        return true;
    }
    int claim(uint32_t g) {
        uint64_t cur = head.load(std::memory_order_acquire);
        for (;;) {
            if ((uint32_t) (cur >> 32) != g) return -1;
            const int n = (int) ((cur >> 16) & 0xffff), j = (int) (cur & 0xffff);
            if (j >= n) return -1;
            if (head.compare_exchange_weak(cur, cur + 1, std::memory_order_acq_rel, std::memory_order_acquire)) return j;
        }
    }
    /// A layer's jobs; the previous layer's are finished (finish()).
    void start(std::vector<Job>&& js) {
        if (js.empty()) return;
        std::lock_guard<std::mutex> lk(mu);
        jobs = std::move(js);
        if (ready_cap < jobs.size()) {
            ready_cap = jobs.size() * 2;
            ready.reset(new std::atomic<int>[ready_cap]);
        }
        for (size_t i = 0; i < jobs.size(); ++i) ready[i].store(0, std::memory_order_relaxed);
        bool any_file = false;
        for (size_t i = 0; i < jobs.size(); ++i) {
            if (jobs[i].fd >= 0) any_file = true;
            else if (i < kAhead) basalt::platform::advise_willneed(jobs[i].src, jobs[i].bytes);
        }
        while (any_file && (int) threads.size() < kIoThreads) threads.emplace_back([this] { work(); });
        issued.store(0);
        ++gen;
        head.store((uint64_t) gen << 32 | (uint64_t) jobs.size() << 16, std::memory_order_release);
        cv.notify_all();
    }
    /// Job j's bytes, in a pinned buffer (waits for the copy).
    const uint8_t* wait(int j) {
        while (!ready[(size_t) j].load(std::memory_order_acquire)) std::this_thread::yield();
        return buf[j % kRing];
    }
    /// The launching thread queued job j's DMA on `copy`: its buffer is free once that is done.
    void issued_one(int j, cudaStream_t copy) {
        cudaEventRecord(dma_done[j % kRing], copy);
        issued.store(j + 1, std::memory_order_release);
    }
    /// No job is running after this (the end of a layer, or an early return in the middle of one).
    void finish() {
        head.store((uint64_t) gen << 32, std::memory_order_release);   // n = 0: nothing more to claim
        issued.store(1 << 30, std::memory_order_release);
        while (active.load(std::memory_order_acquire) != 0) std::this_thread::yield();
    }
};
// A stager job for the expert's blob `b`: read from the model file when that is where it lives.
inline Stager::Job make_job(core::ExpertSource& src, int64_t l, int32_t e, const uint8_t* b, size_t bytes) {
    Stager::Job j{b, bytes};
    int fd = -1;
    uint64_t off = 0;
    if (src.file_extent(l, e, fd, off)) { j.fd = fd; j.off = off; }
    return j;
}


}  // namespace basalt::prefill
