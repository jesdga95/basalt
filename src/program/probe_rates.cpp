// src/program/probe_rates.cpp - the `PROBE rates` serve command (see the header).
//
// Each step reuses the code the microbenches use: the startup PCIe probe's burst timing and avx_bench's cold-expert
// pool loop and its idle read.  A model's own peer path is measured through its ProbePeerPath.
#include "basalt/program/probe_rates.hpp"

#include "basalt/kernels/cpu/cpu_quota.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/platform/dynlib.hpp"
#include "basalt/platform/memory.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

namespace basalt::program {
namespace {

namespace C = basalt::kernels::cpu;
double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct Event {
    cudaEvent_t e = nullptr;
    int dev = 0;
    Event() = default;
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    bool make(int d) {
        dev = d;
        core::OnDevice od(d);
        return cudaEventCreate(&e) == cudaSuccess;
    }
    ~Event() {
        if (e == nullptr) return;
        core::OnDevice od(dev);
        cudaEventDestroy(e);
    }
};

// ------------------------------------------------------------------------------------------------ NVML

class Nvml {
public:
    Nvml() {
        lib_ = platform::DynLib::open_nvml();
        if (!lib_) return;
        init_ = lib_.sym<int (*)()>("nvmlInit_v2");
        shut_ = lib_.sym<int (*)()>("nvmlShutdown");
        by_bus_ = lib_.sym<int (*)(const char*, void**)>("nvmlDeviceGetHandleByPciBusId_v2");
        gen_ = lib_.sym<int (*)(void*, unsigned*)>("nvmlDeviceGetCurrPcieLinkGeneration");
        wid_ = lib_.sym<int (*)(void*, unsigned*)>("nvmlDeviceGetCurrPcieLinkWidth");
        maxgen_ = lib_.sym<int (*)(void*, unsigned*)>("nvmlDeviceGetMaxPcieLinkGeneration");
        maxwid_ = lib_.sym<int (*)(void*, unsigned*)>("nvmlDeviceGetMaxPcieLinkWidth");
        plim_ = lib_.sym<int (*)(void*, unsigned*)>("nvmlDeviceGetPowerManagementLimit");
        clk_ = lib_.sym<int (*)(void*, int, unsigned*)>("nvmlDeviceGetClockInfo");
        ok_ = init_ != nullptr && by_bus_ != nullptr && init_() == 0;
    }
    ~Nvml() { if (ok_ && shut_ != nullptr) shut_(); }
    Nvml(const Nvml&) = delete;
    Nvml& operator=(const Nvml&) = delete;

    struct Info {
        int gen = -1, width = -1, max_gen = -1, max_width = -1, sm_mhz = -1;
        double power_w = -1;
    };
    Info info(int dev) const {
        Info r;
        char bus[64] = {};
        void* h = nullptr;
        if (!ok_ || cudaDeviceGetPCIBusId(bus, sizeof bus, dev) != cudaSuccess || by_bus_(bus, &h) != 0) return r;
        unsigned v = 0;
        if (gen_ && gen_(h, &v) == 0) r.gen = (int) v;
        if (wid_ && wid_(h, &v) == 0) r.width = (int) v;
        if (maxgen_ && maxgen_(h, &v) == 0) r.max_gen = (int) v;
        if (maxwid_ && maxwid_(h, &v) == 0) r.max_width = (int) v;
        if (plim_ && plim_(h, &v) == 0) r.power_w = v / 1000.0;
        if (clk_ && clk_(h, 1, &v) == 0) r.sm_mhz = (int) v;
        return r;
    }

private:
    platform::DynLib lib_;
    bool ok_ = false;
    int (*init_)() = nullptr;
    int (*shut_)() = nullptr;
    int (*by_bus_)(const char*, void**) = nullptr;
    int (*gen_)(void*, unsigned*) = nullptr;
    int (*wid_)(void*, unsigned*) = nullptr;
    int (*maxgen_)(void*, unsigned*) = nullptr;
    int (*maxwid_)(void*, unsigned*) = nullptr;
    int (*plim_)(void*, unsigned*) = nullptr;
    int (*clk_)(void*, int, unsigned*) = nullptr;
};

// ------------------------------------------------------------------------------------------------ steps

struct Probe {
    const ProbeEnv& env;
    std::FILE* out;
    std::vector<int> devs;   // CUDA0 and the peer card
    bool have_layout = false;
    C::NativeFmt fmt;
    std::vector<const uint8_t*> blobs;   // distinct RAM-resident blobs of the layers that share layer 0's format
    std::vector<int> host_k;             // worker counts in the host bandwidth order, best first

    template <class... A>
    void line(const char* section, const char* f, A... a) {
        std::fprintf(out, "PROBE %s ", section);
        if constexpr (sizeof...(A) == 0) std::fputs(f, out);
        else std::fprintf(out, f, a...);
        std::fprintf(out, "\n");
    }
};

// 1: facts
void step_facts(Probe& P) {
    Nvml nvml;
    for (int d : P.devs) {
        cudaDeviceProp pr;
        if (cudaGetDeviceProperties(&pr, d) != cudaSuccess) { (void) cudaGetLastError(); continue; }
        size_t fr = 0, tot = 0;
        {
            core::OnDevice od(d);
            cudaMemGetInfo(&fr, &tot);
        }
        int mclk = 0, bus = 0;
        cudaDeviceGetAttribute(&mclk, cudaDevAttrMemoryClockRate, d);
        cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, d);
        const Nvml::Info n = nvml.info(d);
        P.line("gpu", "dev=%d name=\"%s\" sm=%d cc=%d.%d vram_total_mib=%zu vram_free_mib=%zu mem_peak_gbps=%.0f "
                      "link_gen=%d link_width=%d link_max_gen=%d link_max_width=%d power_limit_w=%.0f sm_mhz=%d",
               d, pr.name, pr.multiProcessorCount, pr.major, pr.minor, tot >> 20, fr >> 20,
               2.0 * (double) mclk * 1e3 * (bus / 8.0) / 1e9, n.gen, n.width, n.max_gen, n.max_width, n.power_w, n.sm_mhz);
    }
    const C::ExpertPool& pool = *P.env.pool;
    P.line("host", "ram_total_mib=%llu ram_avail_mib=%llu threads=%u pool_workers=%d active_workers=%d "
                   "p_cores=%d e_cores=%d hybrid=%d quota_cpus=%.1f",
           (unsigned long long) (platform::total_physical_memory() >> 20),
           (unsigned long long) (platform::available_memory() >> 20), std::thread::hardware_concurrency(),
           pool.workers(), pool.active_workers(), pool.p_cores(), pool.e_cores(),
           pool.is_hybrid() ? 1 : 0, C::cgroup_cpu_quota());
    if (P.have_layout)
        P.line("pack", "layers=%lld experts=%lld gu_type=%d down_type=%d blob_bytes=%zu top_k=%d ram_blobs=%zu",
               (long long) P.env.n_layers, (long long) P.env.n_expert, P.fmt.gu_type, P.fmt.d_type, P.fmt.bytes,
               P.env.top_k, P.blobs.size());
    else
        P.line("pack", "native=0");
}

// 2: each link, both ways: pinned bursts, the fastest is the link's figure (a disturbance only slows a copy)
void step_links(Probe& P) {
    constexpr size_t kBytes = 64ull << 20;
    constexpr int kBursts = 6;
    for (int d : P.devs) {
        core::OnDevice od(d);
        HostBuf h;
        DevBuf dv;
        Stream st;
        Event ev[kBursts + 1];
        bool ok = h.alloc(kBytes, cudaHostAllocPortable) && dv.alloc(d, kBytes) && st.make(d);
        for (auto& e : ev) ok = ok && e.make(d);
        if (!ok) { P.line("link", "dev=%d h2d_gbps=n/a d2h_gbps=n/a", d); continue; }
        double best[2] = {-1.0, -1.0};
        for (int dir = 0; dir < 2; ++dir) {
            const cudaMemcpyKind kind = dir == 0 ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
            auto copy = [&] {
                if (dir == 0) cudaMemcpyAsync(dv.p, h.p, kBytes, kind, st.s);
                else cudaMemcpyAsync(h.p, dv.p, kBytes, kind, st.s);
            };
            copy();   // the context and the copy engine up
            cudaStreamSynchronize(st.s);
            cudaEventRecord(ev[0].e, st.s);
            for (int b = 0; b < kBursts; ++b) {
                copy();
                cudaEventRecord(ev[b + 1].e, st.s);
            }
            if (cudaEventSynchronize(ev[kBursts].e) != cudaSuccess) { (void) cudaGetLastError(); continue; }
            for (int b = 0; b < kBursts; ++b) {
                float ms = 0.f;
                if (cudaEventElapsedTime(&ms, ev[b].e, ev[b + 1].e) == cudaSuccess && ms > 0.01f)
                    best[dir] = std::max(best[dir], (double) kBytes / (ms * 1e-3) / 1e9);
            }
        }
        P.line("link", "dev=%d h2d_gbps=%.1f d2h_gbps=%.1f burst_mib=%zu", d, best[0], best[1], kBytes >> 20);
    }
}

// 3: GPU memory bandwidth: a device-to-device copy counts the bytes it reads and the bytes it writes
void step_gpu_mem(Probe& P) {
    constexpr size_t kBytes = 160ull << 20;
    for (int d : P.devs) {
        core::OnDevice od(d);
        DevBuf a, b;
        Stream st;
        Event e0, e1;
        if (!(a.alloc(d, kBytes) && b.alloc(d, kBytes) && st.make(d) && e0.make(d) && e1.make(d))) {
            P.line("gpu_mem", "dev=%d copy_gbps=n/a", d);
            continue;
        }
        cudaMemsetAsync(a.p, 1, kBytes, st.s);
        cudaMemcpyAsync(b.p, a.p, kBytes, cudaMemcpyDeviceToDevice, st.s);   // clocks up
        cudaStreamSynchronize(st.s);
        double best = -1.0;
        for (int rep = 0; rep < 10; ++rep) {
            cudaEventRecord(e0.e, st.s);
            cudaMemcpyAsync(b.p, a.p, kBytes, cudaMemcpyDeviceToDevice, st.s);
            cudaEventRecord(e1.e, st.s);
            cudaEventSynchronize(e1.e);
            float ms = 0.f;
            if (cudaEventElapsedTime(&ms, e0.e, e1.e) == cudaSuccess && ms > 0.01f)
                best = std::max(best, 2.0 * (double) kBytes / (ms * 1e-3) / 1e9);
        }
        P.line("gpu_mem", "dev=%d copy_gbps=%.0f buffer_mib=%zu counts=read+write", d, best, kBytes >> 20);
    }
}

struct Read {
    const uint8_t* const* blob = nullptr;
    size_t n = 0, bytes = 0;
    std::atomic<int> done{0};
    std::atomic<uint64_t> sink{0};
};

void read_fn(void* ctx, int w, int W, const std::atomic<bool>& stop) {
    Read& r = *(Read*) ctx;
    uint64_t x = 0;
    for (size_t i = (size_t) w; i < r.n && !stop.load(std::memory_order_relaxed); i += (size_t) W)
        for (size_t o = 0; o + 8 <= r.bytes; o += 64) x += *(const volatile uint64_t*) (r.blob[i] + o);
    r.sink.fetch_add(x);
    r.done.fetch_add(1);
}

std::vector<int> worker_counts(int n) {
    std::vector<int> ks;
    for (int k = 1; k < n; k *= 2) ks.push_back(k);
    ks.push_back(n);
    return ks;
}

// 4: host memory bandwidth by active worker count, reading RAM-resident expert blobs
void step_host_mem(Probe& P) {
    C::ExpertPool& pool = *P.env.pool;
    const size_t want = (size_t) (1ull << 30) / std::max<size_t>(P.fmt.bytes, 1);
    if (!P.have_layout || P.blobs.size() < std::min<size_t>(want, 64)) {
        P.line("host_mem", "gbps=n/a reason=no_ram_resident_experts");
        return;
    }
    Read rd;
    rd.blob = P.blobs.data();
    rd.n = std::min(want, P.blobs.size());
    rd.bytes = P.fmt.bytes;
    std::vector<std::pair<double, int>> rank;
    for (int k : worker_counts(pool.workers())) {
        pool.set_active_workers(k);
        double best = -1.0;
        for (int rep = 0; rep < 3; ++rep) {
            rd.done.store(0);
            const auto t0 = ProbeClock::now();
            pool.idle_start(read_fn, &rd);
            while (rd.done.load() < pool.active_workers()) {}
            const double us = us_since(t0);
            pool.idle_stop();
            best = std::max(best, (double) rd.n * (double) rd.bytes / (us * 1e3));
        }
        rank.push_back({best, k});
        P.line("host_mem", "workers=%d gbps=%.1f bytes_mib=%zu", k, best, (rd.n * rd.bytes) >> 20);
    }
    std::sort(rank.begin(), rank.end(), [](auto& a, auto& b) { return a.first > b.first; });
    for (auto& r : rank) P.host_k.push_back(r.second);
}

// 5: CPU expert rate: cold experts through the pool, n per call, at every measured worker count and the current one
void step_cpu_expert(Probe& P, int original_workers) {
    C::ExpertPool& pool = *P.env.pool;
    if (!P.have_layout || P.blobs.size() < 64) {
        P.line("cpu_expert", "us=n/a reason=no_ram_resident_experts");
        return;
    }
    const C::NativeFmt& f = P.fmt;
    constexpr int kMaxN = 8, kIters = 24;
    std::mt19937 rng(12345);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) f.n_embd);
    for (auto& v : x) v = nd(rng);
    std::vector<uint8_t> act(C::kNativeActBytes);
    C::native_quant_act(f, x.data(), act.data());
    std::vector<std::vector<float>> outs(kMaxN, std::vector<float>((size_t) f.n_embd));
    std::vector<int> ks = worker_counts(pool.workers());
    if (std::find(ks.begin(), ks.end(), original_workers) == ks.end()) ks.push_back(original_workers);
    size_t pick = 0;
    for (int k : ks) {
        pool.set_active_workers(k);
        for (int n : {1, 2, 3, 8}) {
            std::vector<double> us;
            for (int it = 0; it < kIters + 2; ++it) {
                C::ExpertJobMulti jobs[kMaxN];
                for (int j = 0; j < n; ++j) {
                    jobs[j].blob = P.blobs[pick++ % P.blobs.size()];
                    jobs[j].nt = 1;
                    jobs[j].nact[0] = act.data();
                    jobs[j].out[0] = outs[(size_t) j].data();
                }
                const auto t0 = ProbeClock::now();
                pool.run_split_multi_native(f, jobs, n);
                if (it >= 2) us.push_back(us_since(t0));
            }
            const double m = median(us);
            P.line("cpu_expert", "workers=%d entries=%d median_us=%.1f per_entry_us=%.1f gbps=%.1f", k, n, m, m / n,
                   (double) n * (double) f.bytes / (m * 1e3));
        }
    }
}

// 6 and 7: the peer's round trip by entry count, and the copy costs into each tier
void step_peer_and_copies(Probe& P) {
    const ProbeEnv& env = P.env;
    const bool peer = env.peer_device >= 0;
    ProbePeerPath* pr = env.peer_path.get();
    std::string why, err;
    bool peer_ok = false;
    if (!P.have_layout) {
        P.line("peer", "rt_us=n/a reason=not_native");
    } else if (!peer) {
        P.line("peer", "rt_us=n/a reason=no_peer");
    } else if (pr == nullptr || !pr->setup(env, P.fmt, why)) {
        P.line("peer", "rt_us=n/a reason=%s", pr == nullptr ? "no_peer_kernel" : why.c_str());
    } else {
        peer_ok = true;
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        int cnt = 0;
        std::vector<double> rt;
        for (int n = 1; n <= pr->max_entries() && peer_ok; ++n) {
            pr->fill(n);
            std::vector<double> all;
            for (int rep = 0; rep < 2 && peer_ok; ++rep) {
                peer_ok = pr->window(P.fmt, [](int) {}, rt, err);
                for (size_t i = 2; i < rt.size() && peer_ok; ++i) all.push_back(rt[i]);
            }
            if (!peer_ok) break;
            const double m = median(all);
            P.line("peer", "entries=%d rt_median_us=%.1f", n, m);
            sx += n; sy += m; sxx += (double) n * n; sxy += n * m; ++cnt;
        }
        if (peer_ok) {
            const double b = (cnt * sxy - sx * sy) / (cnt * sxx - sx * sx);
            P.line("peer", "fit base_us=%.1f per_entry_us=%.2f peer_blocks=%d", (sy - b * sx) / cnt, b, pr->blocks());
        } else {
            P.line("peer", "rt_us=n/a reason=%s", err.c_str());
        }
    }

    // copy costs: one expert-sized pinned copy into each tier's card, timed by events
    if (!P.have_layout) { P.line("copy", "us=n/a reason=not_native"); return; }
    const size_t bytes = P.fmt.bytes;
    HostBuf src;
    if (!src.alloc(bytes, cudaHostAllocPortable)) { P.line("copy", "us=n/a reason=alloc_failed"); return; }
    if (!P.blobs.empty()) std::memcpy(src.p, P.blobs[0], bytes);
    for (int d : P.devs) {
        core::OnDevice od(d);
        DevBuf dst;
        Stream st;
        Event e0, e1;
        if (!(dst.alloc(d, bytes) && st.make(d) && e0.make(d) && e1.make(d))) {
            P.line("copy", "dev=%d us=n/a", d);
            continue;
        }
        std::vector<double> us;
        for (int i = 0; i < 22; ++i) {
            cudaEventRecord(e0.e, st.s);
            cudaMemcpyAsync(dst.p, src.p, bytes, cudaMemcpyHostToDevice, st.s);
            cudaEventRecord(e1.e, st.s);
            cudaEventSynchronize(e1.e);
            float ms = 0.f;
            if (i >= 2 && cudaEventElapsedTime(&ms, e0.e, e1.e) == cudaSuccess) us.push_back(ms * 1e3);
        }
        const double m = median(us);
        P.line("copy", "dev=%d blob_bytes=%zu median_us=%.1f gbps=%.1f", d, bytes, m, (double) bytes / (m * 1e3));
    }

    // the peer's rings with one such copy running into the peer beside them
    if (!peer_ok) return;
    const int d = env.peer_device;
    DevBuf dst;
    Stream cs;
    Event ce;
    if (!(dst.alloc(d, bytes) && cs.make(d) && ce.make(d))) return;
    pr->fill(std::min(3, pr->max_entries()));
    std::vector<double> during, idle, rt;
    for (int launch = 0; launch < 30 && peer_ok; ++launch) {
        bool active = false;
        std::vector<char> busy_at((size_t) pr->rings(), 0);
        peer_ok = pr->window(P.fmt, [&](int r) {
            if (active && cudaEventQuery(ce.e) != cudaErrorNotReady) active = false;
            busy_at[(size_t) r] = active ? 1 : 0;
            if (r == 4) {
                core::OnDevice od(d);
                cudaMemcpyAsync(dst.p, src.p, bytes, cudaMemcpyHostToDevice, cs.s);
                cudaEventRecord(ce.e, cs.s);
                active = true;
            }
        }, rt, err);
        for (size_t r = 2; r < rt.size() && peer_ok; ++r) (busy_at[r] ? during : idle).push_back(rt[r]);
        { core::OnDevice od(d); cudaStreamSynchronize(cs.s); }
    }
    if (peer_ok && !during.empty())
        P.line("copy", "peer_rings_beside_copy entries=3 idle_median_us=%.1f during_median_us=%.1f during_rings=%zu",
               median(idle), median(during), during.size());
}

}  // namespace

bool probe_rates(const ProbeEnv& env, std::FILE* out, std::string& err) {
    if (env.pool == nullptr || !env.ram_blob) { err = "PROBE: no CPU pool or expert source"; return false; }
    const auto t_all = ProbeClock::now();
    cudaDeviceSynchronize();   // between requests: nothing of the engine's is in flight
    Probe P{env, out, {}, false, {}, {}, {}};
    int cur = 0;
    cudaGetDevice(&cur);
    P.devs.push_back(cur);
    if (env.peer_device >= 0) P.devs.push_back(env.peer_device);

    if (!env.fmt.empty() && env.n_layers > 0) {
        P.fmt = env.fmt[0];
        P.have_layout = true;
        std::mt19937 rng(777);
        std::vector<std::pair<int, int>> all;
        for (int l = 0; l < env.n_layers && l < (int) env.fmt.size(); ++l) {
            const C::NativeFmt& f = env.fmt[(size_t) l];
            if (f.gu_type != P.fmt.gu_type || f.d_type != P.fmt.d_type || f.bytes != P.fmt.bytes ||
                f.down_off != P.fmt.down_off)
                continue;
            for (int e = 0; e < env.n_expert; ++e) all.push_back({l, e});
        }
        std::shuffle(all.begin(), all.end(), rng);
        for (const auto& le : all) {
            const uint8_t* b = env.ram_blob(le.first, le.second);
            if (b != nullptr) P.blobs.push_back(b);
            if (P.blobs.size() >= 1024) break;
        }
    }

    const int original = env.pool->active_workers();
    std::fprintf(out, "PROBE begin rates v1\n");
    auto timed = [&](const char* name, auto&& fn) {
        const auto t0 = ProbeClock::now();
        fn();
        P.line("step", "name=%s ms=%.0f", name, us_since(t0) / 1e3);
    };
    timed("facts", [&] { step_facts(P); });
    timed("links", [&] { step_links(P); });
    timed("gpu_mem", [&] { step_gpu_mem(P); });
    timed("host_mem", [&] { step_host_mem(P); });
    timed("cpu_expert", [&] { step_cpu_expert(P, original); });
    env.pool->set_active_workers(original);
    timed("peer_copies", [&] { step_peer_and_copies(P); });
    env.pool->set_active_workers(original);
    cudaSetDevice(cur);
    (void) cudaGetLastError();
    std::fprintf(out, "PROBE end ms=%.0f workers_restored=%d\n", us_since(t_all) / 1e3, env.pool->active_workers());
    std::fflush(out);
    return true;
}

}  // namespace basalt::program
