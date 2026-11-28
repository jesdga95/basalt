// src/models/gemma4/experts.cpp - see experts.hpp.
#include "models/gemma4/experts.hpp"

#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/log.hpp"
#include "basalt/platform/compiler.hpp"
#include "basalt/platform/cpu.hpp"
#include "basalt/platform/memory.hpp"
#include "models/gemma4/model.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace basalt::models::gemma4 {
namespace cpu = kernels::cpu;

namespace {
// The doorbell's thread spins this long after its last visit or window, then naps a step at a time until the next.
constexpr std::chrono::milliseconds kSpin{20}, kNap{1};
}  // namespace

struct TierExperts::Impl {
    core::ExpertTier* tier = nullptr;
    int n_layers = 0, n_expert = 0, k = 0, width = 0;
    bool off = false;
    // placement, written between windows: the device tables and their pinned staging; the host's view of each expert
    unsigned long long* d_tab = nullptr;
    unsigned long long* h_tab = nullptr;
    std::vector<uint8_t> on_card;        // [layer * n_expert + e]
    std::vector<const uint8_t*> blob;    // [layer * n_expert + e] the host blob of an expert off the card
    bool tables = false;
    // the windows' choices: device [layer][kMoePairs] and pinned host; the rows of the window whose log is pending
    int32_t* d_log = nullptr;
    int32_t* h_log = nullptr;
    int log_rows = 0;
    std::vector<cudaEvent_t> done_ev;    // [layer] behind the layer's merge
    // mapped: the bell, the experts' inputs [kRows][width], the choices [kMoePairs], the host's rows [kMoePairs][width]
    MoeBell* bell = nullptr;
    MoeBell* bell_dev = nullptr;
    uint32_t* seq = nullptr;
    float* x_m = nullptr;
    float* x_dev = nullptr;
    int32_t* ids_m = nullptr;
    int32_t* ids_dev = nullptr;
    float* out_m = nullptr;
    float* out_dev = nullptr;
    uint8_t* acts = nullptr;             // [kRows][kNativeActBytes]
    std::vector<cpu::ExpertJobMulti> jobs;
    std::vector<int> job_of;
    std::thread server;
    std::atomic<bool> stop{false}, napping{false};
    std::atomic<uint32_t> wakes{0};
    std::mutex nap_mu;
    std::condition_variable nap_cv;
    mutable std::mutex err_mu;
    std::string error;

    ~Impl() {
        if (server.joinable()) {
            std::atomic_ref<uint32_t>(bell->abort).store(1, std::memory_order_release);
            stop.store(true, std::memory_order_seq_cst);
            {
                std::lock_guard<std::mutex> lk(nap_mu);
                nap_cv.notify_all();
            }
            server.join();
        }
        for (cudaEvent_t e : done_ev)
            if (e) cudaEventDestroy(e);
        if (seq) cudaFree(seq);
        if (d_tab) cudaFree(d_tab);
        if (d_log) cudaFree(d_log);
        for (void* p : {(void*) bell, (void*) x_m, (void*) ids_m, (void*) h_tab, (void*) h_log})
            if (p) cudaFreeHost(p);
        if (acts) platform::free_aligned(acts);
    }

    void fail(const std::string& why) {
        {
            std::lock_guard<std::mutex> lk(err_mu);
            if (error.empty()) error = why;
        }
        basalt::log::error("basalt gemma4", "experts off the card: %s", why.c_str());
        std::atomic_ref<uint32_t>(bell->abort).store(1, std::memory_order_release);
    }

    // On the doorbell's thread: the pool computes the layer's pairs off the card, each expert's rows in jobs of up to
    // cpu::MAXT, into the mapped rows of their pairs.
    void run(int layer, int rows) {
        const int n = rows * k;
        const size_t base = (size_t) layer * n_expert;
        std::fill(job_of.begin(), job_of.end(), -1);
        bool quant[kRows] = {};
        int nj = 0;
        for (int p = 0; p < n; ++p) {
            const int e = ids_m[p], r = p / k;
            if (e < 0 || e >= n_expert || on_card[base + e]) continue;
            int& j = job_of[(size_t) e];
            if (j < 0 || jobs[(size_t) j].nt == cpu::MAXT) {
                j = nj++;
                jobs[(size_t) j] = cpu::ExpertJobMulti{};
                jobs[(size_t) j].blob = blob[base + e];
            }
            cpu::ExpertJobMulti& job = jobs[(size_t) j];
            job.nact[job.nt] = acts + (size_t) r * cpu::kNativeActBytes;
            job.out[job.nt++] = out_m + (size_t) p * width;
            quant[r] = true;
        }
        const cpu::NativeFmt& f = tier->fmt(layer);
        for (int r = 0; r < rows; ++r)
            if (quant[r]) cpu::native_quant_act(f, x_m + (size_t) r * width, acts + (size_t) r * cpu::kNativeActBytes);
        if (nj == 0) return;
        tier->pool()->start_split_multi_native(f, jobs.data(), nj);
        tier->pool()->finish_split_multi_native();
    }

    void serve() {
        std::atomic_ref<uint32_t> ring(bell->ring), ticket(bell->ticket), fin(bell->done);
        auto busy = std::chrono::steady_clock::now();
        uint32_t seen = wakes.load(std::memory_order_relaxed), spins = 0;
        for (;;) {
            if (const uint32_t w = ring.load(std::memory_order_acquire)) {
                const uint32_t visit = ticket.load(std::memory_order_relaxed);
                ring.store(0, std::memory_order_relaxed);
                run((int) (w >> 8 & 0xffu), (int) (w >> 16));
                fin.store(visit, std::memory_order_release);
                busy = std::chrono::steady_clock::now();
                continue;
            }
            if (stop.load(std::memory_order_relaxed)) return;
            basalt::platform::cpu_relax();
            if ((++spins & 63u) != 0) continue;
            const auto now = std::chrono::steady_clock::now();
            if (const uint32_t wk = wakes.load(std::memory_order_acquire); wk != seen) {
                seen = wk;
                busy = now;
            }
            if (now - busy < kSpin) continue;
            std::unique_lock<std::mutex> lk(nap_mu);
            napping.store(true, std::memory_order_seq_cst);
            nap_cv.wait_for(lk, kNap, [&] {
                return wakes.load(std::memory_order_seq_cst) != seen || stop.load(std::memory_order_relaxed) ||
                       ring.load(std::memory_order_acquire) != 0;
            });
            napping.store(false, std::memory_order_relaxed);
        }
    }

    void wake() {
        wakes.fetch_add(1, std::memory_order_seq_cst);
        if (napping.load(std::memory_order_seq_cst)) {
            std::lock_guard<std::mutex> lk(nap_mu);
            nap_cv.notify_one();
        }
    }

    // The tier's placement into the tables; false when an expert has no place at all.
    bool place(cudaStream_t stream, std::string& err) {
        bool changed = !tables;
        for (int l = 0; l < n_layers; ++l) {
            const size_t base = (size_t) l * n_expert;
            for (int e = 0; e < n_expert; ++e) {
                const core::ExpertTier::Source s = tier->locate(l, e);
                unsigned long long t = 0;
                const uint8_t* hb = nullptr;
                if (s.dev) t = (unsigned long long) s.dev;
                else if (!(hb = tier->host_blob(l, e))) {
                    err = "gemma4: expert " + std::to_string(e) + " of layer " + std::to_string(l) + " has no place";
                    return false;
                }
                changed = changed || h_tab[base + e] != t;
                h_tab[base + e] = t;
                on_card[base + e] = t != 0;
                blob[base + e] = hb;
            }
        }
        tables = true;
        if (!changed) return true;
        if (const cudaError_t ce = cudaMemcpyAsync(d_tab, h_tab, (size_t) n_layers * n_expert * sizeof(unsigned long long),
                                                   cudaMemcpyHostToDevice, stream);
            ce != cudaSuccess) {
            err = std::string("gemma4: expert tables: ") + cudaGetErrorString(ce);
            return false;
        }
        return true;
    }

    // The pending window's choices through the tier's lookups and promotions, after the copies the window before carried
    // landed and the tables took them.
    bool learn(cudaStream_t stream, std::string& err) {
        tier->settle();
        if (!place(stream, err)) return false;
        if (log_rows <= 0) return true;
        const int n = log_rows * k;
        tier->begin_token(log_rows);
        int32_t uniq[kMoePairs];
        for (int l = 0; l < n_layers; ++l) {
            const int32_t* ids = h_log + (size_t) l * kMoePairs;
            int nu = 0;
            for (int i = 0; i < n; ++i) {
                const int32_t e = ids[i];
                if (e < 0) continue;
                bool seen = false;
                for (int j = 0; j < nu && !seen; ++j) seen = uniq[j] == e;
                if (!seen) uniq[nu++] = e;
            }
            if (nu == 0) continue;
            core::ExpertTier::Lookup lk;
            if (!tier->lookup(l, uniq, nu, lk, err)) return false;
            tier->promote(l, lk);
        }
        log_rows = 0;
        return true;
    }
};

TierExperts::TierExperts() : impl_(std::make_unique<Impl>()) {}
TierExperts::~TierExperts() = default;

std::unique_ptr<TierExperts> TierExperts::make(const Gemma4Model& m, core::ExpertTier& tier, float* rows, std::string& err) {
    const Geometry& g = m.geo();
    if (g.n_used * kRows > kMoePairs || g.n_layers > 0xff + 1) {
        err = "gemma4: the experts off the card do not fit this geometry";
        return nullptr;
    }
    std::unique_ptr<TierExperts> out(new TierExperts());
    Impl& p = *out->impl_;
    p.tier = &tier;
    p.n_layers = g.n_layers;
    p.n_expert = g.n_expert;
    p.k = g.n_used;
    p.width = g.n_embd;
    const size_t tab_n = (size_t) p.n_layers * p.n_expert;
    p.on_card.assign(tab_n, 0);
    p.blob.assign(tab_n, nullptr);
    p.jobs.resize((size_t) kMoePairs);
    p.job_of.assign((size_t) g.n_expert, -1);
    p.acts = (uint8_t*) platform::alloc_aligned((size_t) kRows * cpu::kNativeActBytes, 64);
    const unsigned mapped = cudaHostAllocMapped | cudaHostAllocPortable;
    if (!p.acts || cudaMalloc(&p.d_tab, tab_n * sizeof(unsigned long long)) != cudaSuccess ||
        cudaHostAlloc((void**) &p.h_tab, tab_n * sizeof(unsigned long long), cudaHostAllocDefault) != cudaSuccess ||
        cudaMalloc(&p.d_log, (size_t) p.n_layers * kMoePairs * sizeof(int32_t)) != cudaSuccess ||
        cudaHostAlloc((void**) &p.h_log, (size_t) p.n_layers * kMoePairs * sizeof(int32_t), cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void**) &p.bell, sizeof(MoeBell), cudaHostAllocMapped) != cudaSuccess ||
        cudaHostGetDevicePointer((void**) &p.bell_dev, p.bell, 0) != cudaSuccess ||
        cudaHostAlloc((void**) &p.x_m, (size_t) kRows * p.width * sizeof(float), mapped) != cudaSuccess ||
        cudaHostGetDevicePointer((void**) &p.x_dev, p.x_m, 0) != cudaSuccess ||
        cudaHostAlloc((void**) &p.ids_m, (size_t) kMoePairs * sizeof(int32_t), mapped) != cudaSuccess ||
        cudaHostGetDevicePointer((void**) &p.ids_dev, p.ids_m, 0) != cudaSuccess || !(p.out_m = rows) ||
        cudaHostGetDevicePointer((void**) &p.out_dev, p.out_m, 0) != cudaSuccess ||
        cudaMalloc(&p.seq, sizeof(uint32_t)) != cudaSuccess || cudaMemset(p.seq, 0, sizeof(uint32_t)) != cudaSuccess) {
        cudaGetLastError();
        err = "gemma4: cannot set up the experts' tables and hand-off";
        return nullptr;
    }
    p.done_ev.assign((size_t) p.n_layers, nullptr);
    for (cudaEvent_t& e : p.done_ev)
        if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) {
            err = "gemma4: cannot create the experts' events";
            return nullptr;
        }
    std::memset(p.h_tab, 0, tab_n * sizeof(unsigned long long));
    *p.bell = MoeBell{};
    tier.settle();
    if (!p.place(nullptr, err) || cudaDeviceSynchronize() != cudaSuccess) {
        if (err.empty()) err = "gemma4: cannot write the expert tables";
        return nullptr;
    }
    // every expert a home on the card (no RAM home and no prompt staging, so nothing ever moves): no hand-off, no lookups
    p.off = m.stage_experts() > 0 || std::find(p.on_card.begin(), p.on_card.end(), (uint8_t) 0) != p.on_card.end();
    if (p.off) {
        p.server = std::thread([&p] { p.serve(); });
        basalt::log::info("basalt gemma4", "experts: the card's from its tables, the others through the doorbell (%d pool workers)",
                          tier.pool()->workers());
    }
    return out;
}

bool TierExperts::off_card() const { return impl_->off; }
const unsigned long long* TierExperts::tab(int l) const { return impl_->d_tab + (size_t) l * impl_->n_expert; }

bool TierExperts::begin_window(cudaStream_t stream, std::string& err) {
    Impl& p = *impl_;
    if (!p.off) return true;
    if (failed(err)) return false;
    p.wake();
    return p.learn(stream, err);
}

void TierExperts::ring(int l, int rows, const float* xr, const float* w, const int32_t* ids, cudaStream_t stream) {
    Impl& p = *impl_;
    moe_ring(ids, rows, p.k, tab(l), xr, w, p.width, l, p.x_dev, p.ids_dev, p.bell_dev, p.seq, stream);
    if (const cudaError_t ce = cudaMemcpyAsync(p.d_log + (size_t) l * kMoePairs, ids, (size_t) rows * p.k * sizeof(int32_t),
                                               cudaMemcpyDeviceToDevice, stream);
        ce != cudaSuccess)
        p.fail(std::string("choice log copy: ") + cudaGetErrorString(ce));
}

void TierExperts::merge(int l, int rows, const int32_t* ids, float* out, cudaStream_t stream) {
    Impl& p = *impl_;
    moe_merge(ids, rows, p.k, tab(l), p.out_dev, out, p.width, p.bell_dev, p.seq, stream);
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &cs);   // a captured window records the event as its own node
    if (const cudaError_t ce = cudaEventRecordWithFlags(p.done_ev[(size_t) l], stream,
                                                        cs == cudaStreamCaptureStatusActive ? cudaEventRecordExternal : cudaEventRecordDefault);
        ce != cudaSuccess)
        p.fail(std::string("merge event: ") + cudaGetErrorString(ce));
}

void TierExperts::issue_copies() {
    Impl& p = *impl_;
    if (!p.off) return;
    for (int l = 0; l < p.n_layers; ++l) p.tier->issue_after(l, p.done_ev[(size_t) l]);
}

void TierExperts::end_window(int rows, cudaStream_t stream) {
    Impl& p = *impl_;
    if (!p.off) return;
    p.log_rows = rows;
    if (const cudaError_t ce = cudaMemcpyAsync(p.h_log, p.d_log, (size_t) p.n_layers * kMoePairs * sizeof(int32_t),
                                               cudaMemcpyDeviceToHost, stream);
        ce != cudaSuccess)
        p.fail(std::string("choice log read-back: ") + cudaGetErrorString(ce));
}

bool TierExperts::prompt_begin(cudaStream_t stream, std::string& err) {
    Impl& p = *impl_;
    if (!p.off) return true;
    if (failed(err)) return false;
    return p.learn(stream, err);
}

bool TierExperts::failed(std::string& err) const {
    std::lock_guard<std::mutex> lk(impl_->err_mu);
    if (impl_->error.empty()) return false;
    err = "gemma4: experts off the card: " + impl_->error;
    return true;
}

}  // namespace basalt::models::gemma4
