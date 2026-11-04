// include/basalt/program/probe_rates.hpp - the engine's `PROBE rates` serve command: it measures this machine's
// transfer and compute rates inside the loaded engine and prints one result block.
//
// It runs between requests and leaves the engine as it found it: no tier, cache, KV or RNG is touched, the CPU pool's
// active worker count is restored, and every buffer it allocates is freed before it returns.  A model supplies its
// expert format, its RAM-resident blobs and its peer path.
#pragma once

#include "basalt/core/on_device.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/cpu/pool.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace basalt::program {

struct ProbeEnv;

/// A model's expert path on the second card, timed as round trips over `rings()` launches of a window.
class ProbePeerPath {
public:
    virtual ~ProbePeerPath() = default;
    /// False with the reason the block prints.
    virtual bool setup(const ProbeEnv& env, const kernels::cpu::NativeFmt& f, std::string& why) = 0;
    /// The largest round trip's entries.
    virtual int max_entries() const { return 8; }
    virtual int rings() const = 0;
    /// What the block's fit line reports as the peer's thread blocks.
    virtual int blocks() const = 0;
    /// The next windows carry `n` entries.
    virtual void fill(int n) = 0;
    /// One window; `before(r)` runs ahead of ring r, `rt` gets each ring's round trip in us.
    virtual bool window(const kernels::cpu::NativeFmt& f, const std::function<void(int)>& before, std::vector<double>& rt,
                        std::string& err) = 0;
};

struct ProbeEnv {
    kernels::cpu::ExpertPool* pool = nullptr;   ///< the engine's CPU pool (idle between requests)
    std::function<const uint8_t*(int64_t layer, int64_t expert)> ram_blob;   ///< a RAM-resident blob, or null
    std::vector<kernels::cpu::NativeFmt> fmt;   ///< [layer] the expert format; empty: none the steps can run
    int peer_device = -1;                       ///< the peer card, or -1
    std::shared_ptr<ProbePeerPath> peer_path;   ///< its expert path; null: not measured
    int64_t n_layers = 0, n_expert = 0;
    int top_k = 10;                             ///< routed experts per token
};

/// Measures hardware facts, each PCIe link both ways, GPU memory bandwidth, host memory bandwidth by CPU worker
/// count, the CPU expert rate, the peer's round trip and per-entry cost, and the copy costs into each tier.  Writes
/// the block to `out` as lines `PROBE <section> key=value ...` from `PROBE begin rates` to `PROBE end ms=<wall>`.
/// False (with `err`) only when nothing could be measured; a step that cannot run says so in the block.
bool probe_rates(const ProbeEnv& env, std::FILE* out, std::string& err);

using ProbeClock = std::chrono::steady_clock;
inline double us_since(ProbeClock::time_point t0) {
    return std::chrono::duration<double, std::micro>(ProbeClock::now() - t0).count();
}

struct DevBuf {
    void* p = nullptr;
    int dev = 0;
    DevBuf() = default;
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    bool alloc(int d, size_t n) {
        dev = d;
        core::OnDevice od(d);
        if (cudaMalloc(&p, n) == cudaSuccess) return true;
        (void) cudaGetLastError();
        p = nullptr;
        return false;
    }
    ~DevBuf() {
        if (p == nullptr) return;
        core::OnDevice od(dev);
        cudaFree(p);
    }
};

struct HostBuf {
    void* p = nullptr;
    HostBuf() = default;
    HostBuf(const HostBuf&) = delete;
    HostBuf& operator=(const HostBuf&) = delete;
    bool alloc(size_t n, unsigned flags) {
        if (cudaHostAlloc(&p, n, flags) == cudaSuccess) {
            std::memset(p, 0, n);
            return true;
        }
        (void) cudaGetLastError();
        p = nullptr;
        return false;
    }
    ~HostBuf() { if (p != nullptr) cudaFreeHost(p); }
};

struct Stream {
    cudaStream_t s = nullptr;
    int dev = 0;
    Stream() = default;
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    bool make(int d) {
        dev = d;
        core::OnDevice od(d);
        return cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess;
    }
    ~Stream() {
        if (s == nullptr) return;
        core::OnDevice od(dev);
        cudaStreamSynchronize(s);
        cudaStreamDestroy(s);
    }
};

}  // namespace basalt::program
