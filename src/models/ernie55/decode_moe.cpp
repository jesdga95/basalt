// src/models/ernie55/decode_moe.cpp - see the header.
#include "models/ernie55/decode_moe.hpp"

#include "basalt/core/tier_policy.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/log.hpp"
#include "basalt/platform/cpu.hpp"
#include "basalt/platform/memory.hpp"
#include "models/ernie55/model.hpp"

#include "basalt/platform/compiler.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace basalt::models::ernie55 {
namespace cpu = kernels::cpu;

namespace {
constexpr int kRows = 8;                        // a decode window's rows at most
// Share of a visit's host experts the second card fetches over its link and computes alongside the CPU pool.
constexpr int kPeerFetchPct = 40;
// Max hinted experts warmed per layer by the CPU pool's L3 groups so cached lines survive until the visit arrives.
constexpr int kWarmExperts = 6;
// The hinted RAM experts the second card fetches over its own link ahead of the visit, the likeliest first (the pool
// warms the next ones): that link reads RAM beside the pool's warm reads.
constexpr int kPeerEarly = 1;
// The doorbell's thread spins this long after its last visit or window, then naps a step at a time until the next.
constexpr std::chrono::milliseconds kSpin{20}, kNap{1};
enum : uint8_t { kOnCard = 0, kOnPeer = 1, kOnHost = 2 };
}  // namespace

struct DecodeMoe::Impl {
    const ErnieModel* model = nullptr;
    core::ExpertTier* tier = nullptr;
    int n_layers = 0, n_expert = 0, k = 0, width = 0;
    std::vector<kernels::NativeExpertLayout> layout;   // [layer] for the second card's kernels
    // placement, written between windows: the device table and its pinned staging; the host's view of each pair
    unsigned long long* d_tab = nullptr;
    unsigned long long* h_tab = nullptr;
    std::vector<uint8_t> kind;                     // [layer * n_expert + e] kOnCard / kOnPeer / kOnHost
    std::vector<const uint8_t*> blob;              // [layer * n_expert + e] the host blob of a kOnHost expert
    bool tables = false;                           // the tables were written once
    // the choices' logs: device [layer][kMoeDecPairs] and pinned host
    int32_t* d_log = nullptr;
    int32_t* h_log = nullptr;
    int log_rows = 0;
    std::vector<cudaEvent_t> done_ev;              // [layer] behind the layer's merge
    // mapped: the bell, the rows [kRows][width], the choices [kRows * k], the host's output rows [kRows * k][width]
    MoeBell* bell = nullptr;
    MoeBell* bell_dev = nullptr;
    uint32_t* seq = nullptr;
    float* x_m = nullptr;
    float* x_dev = nullptr;
    int32_t* ids_m = nullptr;
    int32_t* ids_dev = nullptr;
    float* out_m = nullptr;                        // the tier's rows (the second card's share lands there itself)
    float* out_dev = nullptr;
    uint8_t* acts = nullptr;                       // [kRows][kNativeActBytes]
    std::vector<cpu::ExpertJobMulti> jobs;
    std::vector<int> job_of;
    std::vector<int32_t> peer_kind, peer_ids;
    std::vector<int> host_e;                       // a visit's distinct host experts
    // the hints (mapped) and their device count; the last word taken
    MoeHint* hint_m = nullptr;
    MoeHint* hint_dev = nullptr;
    uint32_t* hseq = nullptr;
    uint32_t hint_seen = 0;
    // warming, on the doorbell's thread: the layer (-1: none) and its experts
    int warm_layer = -1;
    std::vector<int> warm_e;
    // the layer the second card is armed for ahead of its visit (-1: none), and the last visit's layer this window
    int pre_layer = -1, visited = -1;
    std::vector<int> early_e;
    // a window runs (session thread: from begin_window to after_window); the windows ended (session thread) and the
    // ends whose work the doorbell's thread has stopped
    std::atomic<bool> window_open{false};
    std::atomic<uint32_t> close_req{0}, close_ack{0};
    std::thread server;
    std::atomic<bool> stop{false}, napping{false};
    std::atomic<uint32_t> wakes{0};
    std::mutex nap_mu;
    std::condition_variable nap_cv;
    mutable std::mutex err_mu;
    std::string error;

    ~Impl() {
        if (server.joinable()) {
            if (bell) std::atomic_ref<uint32_t>(bell->abort).store(1, std::memory_order_release);
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
        if (hseq) cudaFree(hseq);
        for (void* p : {(void*) bell, (void*) x_m, (void*) ids_m, (void*) h_tab, (void*) h_log, (void*) hint_m})
            if (p) cudaFreeHost(p);
        if (acts) platform::free_aligned(acts);
    }

    void fail(const std::string& why) {
        {
            std::lock_guard<std::mutex> lk(err_mu);
            if (error.empty()) error = why;
        }
        basalt::log::error("basalt ernie55", "decode experts: %s", why.c_str());
        std::atomic_ref<uint32_t>(bell->abort).store(1, std::memory_order_release);
    }

    // A new hint: the hinted layer's predicted RAM experts, the likeliest first over the rows, up to kWarmExperts; each
    // group's spans are its gate and up rows of each expert, then its down rows, the way the batch reads them.
    void take_hint(uint32_t w) {
        const int layer = (int) (w >> 8 & 0xffu), rows = (int) (w & 0xffu);
        int32_t ids[kMoeHintRows * kMoeHintK];
        if (layer >= n_layers || rows < 1 || rows > kMoeHintRows) return;
        std::memcpy(ids, hint_m->ids[(w >> 16) & 1u], (size_t) rows * kMoeHintK * sizeof(int32_t));
        const uint32_t now = std::atomic_ref<uint32_t>(hint_m->word).load(std::memory_order_acquire);
        if ((((now >> 16) - (w >> 16)) & 0xffffu) >= 2) return;   // the slot was written again while it was read
        cpu::ExpertPool* pool = tier->pool();
        const int G = pool->groups();
        if (G < 2 || !open_work()) return;
        stop_warm();
        const size_t base = (size_t) layer * n_expert;
        warm_e.clear();
        early_e.clear();
        // the second card is armed for this layer now when it can be (its visit before was the layer before: no visit
        // of an earlier layer can still come), which takes the arming off the visit's path; in a window of several rows
        // it also fetches the likeliest (in a window of one row its fetch would still be on the bus at the visit, which
        // the pool's warm copy beats)
        core::PeerExperts* peer = tier->peer_for(layer);
        const bool armable = peer && visited == layer - 1 && pre_layer < 0 && !peer->armed();
        const bool early = armable && rows >= 2;
        const int want = kWarmExperts + (early ? kPeerEarly : 0);
        for (int j = 0; j < kMoeHintK && (int) (warm_e.size() + early_e.size()) < want; ++j)
            for (int r = 0; r < rows && (int) (warm_e.size() + early_e.size()) < want; ++r) {
                const int e = ids[r * kMoeHintK + j];
                if (e < 0 || e >= n_expert || kind[base + e] != kOnHost) continue;
                if (std::find(warm_e.begin(), warm_e.end(), e) != warm_e.end() ||
                    std::find(early_e.begin(), early_e.end(), e) != early_e.end())
                    continue;
                (early && (int) early_e.size() < kPeerEarly ? early_e : warm_e).push_back(e);
            }
        if (!warm_e.empty()) start_spans(layer);
        if (armable) {   // after the warm work started: these calls are off its path
            std::string err;
            if (!peer->arm(layout[(size_t) layer], x_m, rows, k, core::policy::peer_groups(rows), err)) {
                fail("the second card: " + err);
                return;
            }
            pre_layer = layer;
            const uint64_t bytes = layout[(size_t) layer].bytes;
            for (int e : early_e) peer->fetch(e, blob[base + e], bytes);
        }
    }

    // Each group's spans of `layer`'s warm_e (its gate and up rows of each expert, then its down rows, the way the batch
    // reads them), then the warm work on them.
    void start_spans(int layer) {
        cpu::ExpertPool* pool = tier->pool();
        const size_t base = (size_t) layer * n_expert;
        const cpu::NativeFmt& f = tier->fmt(layer);
        pool->warm_clear();
        for (int e : warm_e) pool->warm_expert(f, blob[base + e]);
        warm_layer = layer;
        start_warm();
    }

    // Whether the window still runs (its work may start).
    bool open_work() const { return window_open.load(std::memory_order_seq_cst); }

    // The window has ended: the warm work stops, a share armed for a layer that did not ring is raised with an empty plan
    // (the card's stream waits for it), and the session may use the pool again.
    void close_work() {
        stop_warm();
        warm_layer = -1;
        visited = -1;
        if (pre_layer >= 0) {
            std::string err;
            if (core::PeerExperts* peer = tier->peer_for(pre_layer); peer && !peer->raise(0, nullptr, nullptr, err))
                fail("the second card: " + err);
            pre_layer = -1;
        }
    }

    void start_warm() {
        if (open_work()) tier->pool()->warm_start();
    }

    void stop_warm() { tier->pool()->warm_stop(); }

    // On the doorbell's thread: the host experts the second card fetches go out over its link first, then the pool
    // starts on the other host experts (every row of an expert in one job) and the second card's plan and kernels are
    // queued while it works (their API calls are off the pool's path), then the pool and the card are joined.
    void run(int layer, int rows) {
        const int n = rows * k;
        const size_t base = (size_t) layer * n_expert;
        core::PeerExperts* peer = tier->peer_for(layer);
        std::fill(job_of.begin(), job_of.end(), -1);
        std::string err;
        const bool pre = pre_layer == layer;   // armed at the hint, with experts already on their way
        pre_layer = -1;
        visited = layer;
        if (peer) {   // its homes (kind 2) and the host experts it fetches (kind 3)
            if (!pre && !peer->arm(layout[(size_t) layer], x_m, rows, k, core::policy::peer_groups(rows), err)) {
                fail("the second card: " + err);
                return;
            }
            host_e.clear();
            for (int p = 0; p < n; ++p) {
                const int e = ids_m[p];
                if (e < 0 || kind[base + e] != kOnHost) continue;
                if (job_of[(size_t) e] == -1) {
                    job_of[(size_t) e] = -2;   // negative token count: -(1 + nt)
                    host_e.push_back(e);
                } else {
                    --job_of[(size_t) e];
                }
            }
            // The second card takes from the back: keep L3-warmed experts at the front for the CPU pool, and place
            // multi-token experts (most negative job_of) at the back so one peer PCIe transfer serves all their rows.
            if (host_e.size() > 1 && (rows > 1 || warm_layer == layer)) {
                auto is_warm = [&](int e) {
                    return warm_layer == layer && std::find(warm_e.begin(), warm_e.end(), e) != warm_e.end();
                };
                std::stable_sort(host_e.begin(), host_e.end(), [&](int a, int b) {
                    const bool wa = is_warm(a), wb = is_warm(b);
                    if (wa != wb) return wa;
                    return job_of[(size_t) a] > job_of[(size_t) b];
                });
            }
            int want = std::min((int) host_e.size() * kPeerFetchPct / 100, peer->scratch_free());
            const uint64_t bytes = layout[(size_t) layer].bytes;
            for (int i = 0; i < (int) host_e.size(); ++i) job_of[(size_t) host_e[(size_t) i]] = -1;
            if (pre) {   // the experts fetched at the hint that the layer uses: the card computes them
                for (size_t i = 0; i < host_e.size();)
                    if (peer->fetched(host_e[i])) {
                        job_of[(size_t) host_e[i]] = -3;
                        host_e.erase(host_e.begin() + (ptrdiff_t) i);
                        want = std::max(0, want - 1);
                    } else {
                        ++i;
                    }
                want = std::min(want, std::min((int) host_e.size(), peer->scratch_free()));
            }
            for (int i = 0; i < want; ++i) {
                const int e = host_e[host_e.size() - 1 - (size_t) i];
                if (peer->fetch(e, blob[base + e], bytes)) job_of[(size_t) e] = -3;
            }
        }
        bool quant[kRows] = {};
        int nj = 0;
        for (int p = 0; p < n; ++p) {
            const int e = ids_m[p], r = p / k;
            if (e < 0 || kind[base + e] != kOnHost || job_of[(size_t) e] == -3) continue;
            int& j = job_of[(size_t) e];
            if (j < 0) {
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
        if (nj) tier->pool()->start_split_multi_native(f, jobs.data(), nj);
        bool peer_ok = true;
        if (peer) {
            for (int p = 0; p < n; ++p) {
                const int e = ids_m[p];
                const int kd = e < 0 ? 0 : kind[base + e] == kOnPeer ? 2 : job_of[(size_t) e] == -3 ? 3 : 0;
                peer_kind[(size_t) p] = kd;
                peer_ids[(size_t) p] = kd ? e : 0;
            }
            peer_ok = peer->raise(layer, peer_ids.data(), peer_kind.data(), err) && peer->run_fetched(err);
        }
        if (nj) tier->pool()->finish_split_multi_native();
        if (!peer_ok) {
            fail("the second card: " + err);
            return;
        }
        if (peer && !peer->finish(out_m, err)) fail("the second card: " + err);
    }

    void serve() {
        std::atomic_ref<uint32_t> ring(bell->ring), ticket(bell->ticket), fin(bell->done);
        auto busy = std::chrono::steady_clock::now();
        uint32_t seen = wakes.load(std::memory_order_relaxed), spins = 0;
        for (;;) {
            if (const uint32_t w = ring.load(std::memory_order_acquire)) {
                const uint32_t visit = ticket.load(std::memory_order_relaxed);
                ring.store(0, std::memory_order_relaxed);
                const int layer = (int) (w >> 8 & 0xffu);
                run(layer, (int) (w >> 16));   // a batch of the pool's stops its warm work
                fin.store(visit, std::memory_order_release);
                if (warm_layer > layer) {   // the next layer's hint came first: on with its spans
                    start_warm();
                } else if (warm_layer >= 0) {
                    stop_warm();
                    warm_layer = -1;
                }
                busy = std::chrono::steady_clock::now();
                continue;
            }
            if (const uint32_t hw = std::atomic_ref<uint32_t>(hint_m->word).load(std::memory_order_acquire);
                hw != hint_seen) {
                hint_seen = hw;
                take_hint(hw);
                continue;
            }
            if (const uint32_t req = close_req.load(std::memory_order_seq_cst); req != close_ack.load(std::memory_order_relaxed)) {
                close_work();
                close_ack.store(req, std::memory_order_seq_cst);
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
                uint8_t kd = kOnHost;
                const uint8_t* hb = nullptr;
                unsigned long long t = 0;
                if (s.dev) {
                    kd = kOnCard;
                    t = (unsigned long long) s.dev;
                } else if (s.peer) {
                    kd = kOnPeer;
                } else if (!(hb = tier->host_blob(l, e))) {
                    err = "ernie55: expert " + std::to_string(e) + " of layer " + std::to_string(l) + " has no place";
                    return false;
                }
                changed = changed || h_tab[base + e] != t;
                h_tab[base + e] = t;
                kind[base + e] = kd;
                blob[base + e] = hb;
            }
        }
        tables = true;
        if (!changed) return true;
        const cudaError_t ce = cudaMemcpyAsync(d_tab, h_tab, (size_t) n_layers * n_expert * sizeof(unsigned long long),
                                               cudaMemcpyHostToDevice, stream);
        if (ce != cudaSuccess) {
            err = std::string("ernie55: expert tables: ") + cudaGetErrorString(ce);
            return false;
        }
        return true;
    }
};

DecodeMoe::DecodeMoe() : impl_(std::make_unique<Impl>()) {}
DecodeMoe::~DecodeMoe() = default;

std::unique_ptr<DecodeMoe> DecodeMoe::make(const ErnieModel& m, core::ExpertTier& tier, float* rows, std::string& err) {
    const Geometry& g = m.geo();
    if (g.n_used * kRows > kMoeDecPairs || g.n_layers - g.n_dense > 0xff + 1) {
        err = "ernie55: the decode experts do not fit this geometry";
        return nullptr;
    }
    std::unique_ptr<DecodeMoe> out(new DecodeMoe());
    Impl& p = *out->impl_;
    p.model = &m;
    p.tier = &tier;
    p.n_layers = g.n_layers - g.n_dense;
    p.n_expert = g.n_expert;
    p.k = g.n_used;
    p.width = g.n_latent;
    for (int l = g.n_dense; l < g.n_layers; ++l) {
        const ExpertLayer& el = m.pack().experts[(size_t) l];
        kernels::NativeExpertLayout L = kernels::native_expert_layout(el.gu_type, el.d_type, g.n_latent, g.n_ff_exp);
        L.gate_softcap = g.gate_softcap;
        L.up_softcap = g.up_softcap;
        p.layout.push_back(L);
    }
    const size_t pairs = (size_t) kRows * g.n_used;
    const size_t tab_n = (size_t) p.n_layers * p.n_expert;
    p.kind.assign(tab_n, kOnHost);
    p.blob.assign(tab_n, nullptr);
    p.jobs.resize(pairs);
    p.job_of.assign((size_t) g.n_expert, -1);
    p.peer_kind.assign(pairs, 0);
    p.peer_ids.assign(pairs, 0);
    p.acts = (uint8_t*) platform::alloc_aligned((size_t) kRows * cpu::kNativeActBytes, 64);
    const unsigned mapped = cudaHostAllocMapped | cudaHostAllocPortable;
    if (!p.acts || cudaMalloc(&p.d_tab, tab_n * sizeof(unsigned long long)) != cudaSuccess ||
        cudaHostAlloc((void**) &p.h_tab, tab_n * sizeof(unsigned long long), cudaHostAllocDefault) != cudaSuccess ||
        cudaMalloc(&p.d_log, (size_t) p.n_layers * kMoeDecPairs * sizeof(int32_t)) != cudaSuccess ||
        cudaHostAlloc((void**) &p.h_log, (size_t) p.n_layers * kMoeDecPairs * sizeof(int32_t), cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void**) &p.bell, sizeof(MoeBell), cudaHostAllocMapped) != cudaSuccess ||
        cudaHostGetDevicePointer((void**) &p.bell_dev, p.bell, 0) != cudaSuccess ||
        cudaHostAlloc((void**) &p.x_m, (size_t) kRows * p.width * sizeof(float), mapped) != cudaSuccess ||
        cudaHostGetDevicePointer((void**) &p.x_dev, p.x_m, 0) != cudaSuccess ||
        cudaHostAlloc((void**) &p.ids_m, pairs * sizeof(int32_t), mapped) != cudaSuccess ||
        cudaHostGetDevicePointer((void**) &p.ids_dev, p.ids_m, 0) != cudaSuccess ||
        !(p.out_m = rows) || cudaHostGetDevicePointer((void**) &p.out_dev, p.out_m, 0) != cudaSuccess ||
        cudaMalloc(&p.seq, sizeof(uint32_t)) != cudaSuccess || cudaMemset(p.seq, 0, sizeof(uint32_t)) != cudaSuccess ||
        cudaHostAlloc((void**) &p.hint_m, sizeof(MoeHint), cudaHostAllocMapped) != cudaSuccess ||
        cudaHostGetDevicePointer((void**) &p.hint_dev, p.hint_m, 0) != cudaSuccess ||
        cudaMalloc(&p.hseq, sizeof(uint32_t)) != cudaSuccess || cudaMemset(p.hseq, 0, sizeof(uint32_t)) != cudaSuccess) {
        cudaGetLastError();
        err = "ernie55: cannot set up the decode experts' buffers";
        return nullptr;
    }
    p.done_ev.assign((size_t) p.n_layers, nullptr);
    for (cudaEvent_t& e : p.done_ev)
        if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) {
            err = "ernie55: cannot create the decode experts' events";
            return nullptr;
        }
    std::memset(p.h_tab, 0, tab_n * sizeof(unsigned long long));
    *p.bell = MoeBell{};
    std::memset((void*) p.hint_m, 0, sizeof(MoeHint));
    // the hints' warm reads need the pool's rows split by L3 group (one group: no warming)
    const bool groups = tier.pool()->split_by_groups(platform::cpu_l3_groups());
    p.server = std::thread([&p] { p.serve(); });
    basalt::log::info("basalt ernie55", "decode experts: the card's from its tables, the others through the doorbell (%d pool workers%s%s)",
                      tier.pool()->workers(), groups ? " in L3 groups, warmed from routing hints" : "",
                      tier.peer() ? ", the second card" : "");
    return out;
}

const unsigned long long* DecodeMoe::tab(int l) const { return impl_->d_tab + (size_t) l * impl_->n_expert; }
int32_t* DecodeMoe::log(int l) const { return impl_->d_log + (size_t) l * kMoeDecPairs; }

bool DecodeMoe::begin_window(cudaStream_t stream, bool dirty, std::string& err) {
    Impl& p = *impl_;
    if (failed(err)) return false;
    p.window_open.store(true, std::memory_order_seq_cst);
    p.wake();
    if (!dirty && p.tables) return true;
    p.tier->settle();
    return p.place(stream, err);
}

cudaEvent_t DecodeMoe::layer_done(int l) const { return impl_->done_ev[(size_t) l]; }

void DecodeMoe::issue_copies() {
    Impl& p = *impl_;
    for (int l = 0; l < p.n_layers; ++l) p.tier->issue_after(l, p.done_ev[(size_t) l]);
}

void DecodeMoe::ring(int l, int rows, const float* x, const int32_t* ids, cudaStream_t stream) {
    Impl& p = *impl_;
    moe_dec_ring(ids, rows, p.k, tab(l), x, p.width, l, p.x_dev, p.ids_dev, p.bell_dev, p.seq, stream);
}

void DecodeMoe::hint(int l, int rows, const int32_t* ids, cudaStream_t stream) {
    Impl& p = *impl_;
    moe_dec_hint(ids, rows, l, p.hint_dev, p.hseq, stream);
}

void DecodeMoe::merge(int l, int rows, const int32_t* ids, float* out, cudaStream_t stream) {
    Impl& p = *impl_;
    moe_dec_merge(ids, rows, p.k, tab(l), p.out_dev, out, p.width, p.bell_dev, p.seq, stream);
}

void DecodeMoe::end_window(int rows, cudaStream_t stream) {
    Impl& p = *impl_;
    p.log_rows = rows;
    cudaMemcpyAsync(p.h_log, p.d_log, (size_t) p.n_layers * kMoeDecPairs * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
}

bool DecodeMoe::after_window(cudaStream_t stream, std::string& err) {
    Impl& p = *impl_;
    // the warm work ends with the window: the pool is the session's again (the drafter, a prompt)
    p.window_open.store(false, std::memory_order_seq_cst);
    const uint32_t req = p.close_req.fetch_add(1, std::memory_order_seq_cst) + 1;
    p.wake();
    while (p.close_ack.load(std::memory_order_seq_cst) != req) {
        if (failed(err)) return false;
        basalt::platform::cpu_relax();
    }
    if (p.log_rows <= 0) return true;
    // the copies the window carried land: the tables take the places before this window's promotions claim slots
    p.tier->settle();
    if (!p.place(stream, err)) return false;
    const int n = p.log_rows * p.k;
    p.tier->begin_token(p.log_rows);
    int32_t uniq[kMoeDecPairs];
    for (int l = 0; l < p.n_layers; ++l) {
        const int32_t* ids = p.h_log + (size_t) l * kMoeDecPairs;
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
        if (!p.tier->lookup(l, uniq, nu, lk, err)) return false;
        p.tier->promote(l, lk);
    }
    p.log_rows = 0;
    return true;
}

bool DecodeMoe::failed(std::string& err) const {
    std::lock_guard<std::mutex> lk(impl_->err_mu);
    if (impl_->error.empty()) return false;
    err = "ernie55: decode experts: " + impl_->error;
    return true;
}

}  // namespace basalt::models::ernie55
