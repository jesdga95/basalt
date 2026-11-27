// src/models/inkling2/cpu_experts.cpp - see the header.
#include "models/inkling2/cpu_experts.hpp"

#include "models/inkling2/cpu_expert_drain.hpp"

#include "basalt/core/progress.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/log.hpp"
#include "basalt/platform/compiler.hpp"
#include "basalt/models/inkling2/kernels.hpp"
#include "models/inkling2/model.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace basalt::models::inkling2 {
namespace cpu = kernels::cpu;

static_assert(kBatchRows * 8 <= kMaxPairs, "a batch visit's RAM pairs fit one call");

namespace {
// The doorbell's thread spins this long after its last visit or wake, then naps a step at a time until the next.
constexpr std::chrono::milliseconds kSpin{20}, kNap{1};
// How long a path that ends the engine waits for the calls in flight to finish after it aborts the bell.
constexpr int kDrainMs = 5000;

// A CUDA call that must not fail: a visit that cannot be queued is an error, as a kernel that cannot launch is.
void must(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        cudaGetLastError();
        throw std::runtime_error(std::string("inkling2 cpu experts: ") + what + ": " + cudaGetErrorString(e));
    }
}

// The live doorbell, for an engine that ends while the card's merge may be waiting on it. The mutex ties the object's life
// to the hook that is using it: ~Impl takes it before anything is freed.
std::mutex g_live_mu;
std::function<void()> g_abort;
const void* g_owner = nullptr;
}  // namespace

struct CpuExperts::Impl {
    const InklingModel* model = nullptr;
    int n_expert = 0, k = 0;
    std::unique_ptr<cpu::ExpertPool> pool;
    std::vector<cpu::NativeFmt> fmt;               // [layer]
    core::PeerExperts* peer = nullptr;             // gpu-mode peer: the second card's share of a visit, beside the pool's
    std::vector<kernels::NativeExpertLayout> layout;   // [layer] its blobs, for the second card's kernels
    std::vector<int32_t> kind;                     // [kDecodeRows * k] a pair's computer: 2 the second card, else 0
    cudaStream_t stream = nullptr;                 // the doorbell, beside the card's experts
    cudaEvent_t ready = nullptr, done = nullptr;
    // mapped: the bell, a visit's rows [kDecodeRows][kWidth] and choices [kDecodeRows][k], a pair's row
    // [kDecodeRows * k][kWidth] from the pool; each with its device address
    CpuBell* bell = nullptr;
    CpuBell* bell_dev = nullptr;
    uint32_t* seq = nullptr;   // the device-side counters: the ring's tickets, then the rows' publications
    float* input = nullptr;
    float* input_dev = nullptr;
    int32_t* ids = nullptr;
    int32_t* ids_dev = nullptr;
    float* out = nullptr;
    float* out_dev = nullptr;
    alignas(64) std::array<std::array<uint8_t, cpu::kNativeActBytes>, kBatchRows> acts;
    std::vector<cpu::ExpertJobMulti> jobs;         // [kBatchRows * k]
    std::vector<int> job_of;                       // [n_expert]: the expert's job in this call, -1: none
    std::thread server;                            // the doorbell's thread
    std::atomic<bool> stop{false}, napping{false};
    std::atomic<uint32_t> wakes{0};
    std::mutex nap_mu;
    std::condition_variable nap_cv;
    WindowGate gate;                               // the calls whose windows may wait on `done`

    // No call is admitted after this (a call already in holds admission until its sync); abort the bell permanently,
    // then give the calls in flight a bounded time to finish. If that cannot be confirmed, skip CUDA teardown on the
    // error path. Runs under g_live_mu, which is taken before the gate's lock.
    void abort_and_drain() {
        gate.close();
        std::atomic_ref<uint32_t>(bell->abort).store(1, std::memory_order_release);
        gate.drain(kDrainMs);
    }

    ~Impl() {
        {   // no hook can reach this object past here
            std::lock_guard<std::mutex> lk(g_live_mu);
            if (g_owner == this) {
                g_abort = nullptr;
                g_owner = nullptr;
            }
        }
        if (stream) cudaStreamSynchronize(stream);
        if (server.joinable()) {
            stop.store(true, std::memory_order_seq_cst);
            {
                std::lock_guard<std::mutex> lk(nap_mu);
                nap_cv.notify_all();
            }
            server.join();
        }
        pool.reset();
        if (seq) cudaFree(seq);
        for (void* q : {(void*) bell, (void*) input, (void*) ids, (void*) out})
            if (q) cudaFreeHost(q);
        if (ready) cudaEventDestroy(ready);
        if (done) cudaEventDestroy(done);
        if (stream) cudaStreamDestroy(stream);
    }

    // The second card cannot finish a share: nothing may read the rows it owes, so the sticky abort word releases every
    // merge (the calls in flight are discarded, as when the engine ends).
    void peer_failed(const std::string& why) {
        basalt::log::error("basalt inkling2", "the second card failed: %s", why.c_str());
        std::atomic_ref<uint32_t>(bell->abort).store(1, std::memory_order_release);
    }

    // On the doorbell's thread: a visit's rows quantized for its layer as soon as they come, before its choices.
    void quantize(int layer, int rows) {
        for (int r = 0; r < rows; ++r)
            cpu::native_quant_act(fmt[(size_t) layer], input + (size_t) r * kWidth, acts[(size_t) r].data());
    }
    // On the doorbell's thread: the visit's pairs on experts the second card holds go to it first, then the RAM pairs, every
    // row of an expert in one job, on the pool; the card's rows land in `out` beside the pool's. `quantized`: its rows
    // were quantized when they came.
    void run(int layer, int rows, bool quantized) {
        std::fill(job_of.begin(), job_of.end(), -1);
        bool quant[kBatchRows] = {};
        int n = 0, on_peer = 0;
        std::string err;
        if (peer) {
            for (int p = 0; p < rows * k; ++p) {
                const int e = ids[p];
                kind[(size_t) p] = e >= 0 && e < n_expert && model->peer_serves(layer, e) ? 2 : 0;
                on_peer += kind[(size_t) p] == 2;
            }
            if (on_peer && !peer->launch_layout(layer, layout[(size_t) layer], input, ids, rows, k, kind.data(), err, out)) {
                peer_failed(err);
                on_peer = 0;
            }
        }
        for (int p = 0; p < rows * k; ++p) {
            const int e = ids[p], r = p / k;
            const uint8_t* blob = e >= 0 && e < n_expert ? model->ram_blob(layer, e) : nullptr;
            if (!blob) continue;   // on the card
            int& j = job_of[(size_t) e];
            if (j < 0 || jobs[(size_t) j].nt == cpu::MAXT) {
                j = n++;
                jobs[(size_t) j] = cpu::ExpertJobMulti{};
                jobs[(size_t) j].blob = blob;
            }
            cpu::ExpertJobMulti& job = jobs[(size_t) j];
            job.nact[job.nt] = acts[(size_t) r].data();
            job.out[job.nt++] = out + (size_t) p * kWidth;
            quant[r] = true;
        }
        for (int r = 0; r < rows && !quantized; ++r)
            if (quant[r]) cpu::native_quant_act(fmt[(size_t) layer], input + (size_t) r * kWidth, acts[(size_t) r].data());
        if (n) pool->run_split_multi_native(fmt[(size_t) layer], jobs.data(), n);
        if (on_peer && !peer->finish(out, err)) peer_failed(err);
    }
    // The doorbell's thread: a new rows publication is quantized, then acknowledged (the buffer goes back to the card); a
    // raised bell is taken, its visit computed (its rows quantized first if their publication is not the one taken), then
    // `done` set for the card's merge. It spins while visits come, naps once none has come for kSpin, and wake() ends a nap.
    void serve() {
        std::atomic_ref<uint32_t> ring(bell->ring), ticket(bell->ticket), fin(bell->done), raised(bell->rows),
            shape(bell->rows_shape), acked(bell->rows_ack), ring_rows(bell->ring_rows);
        auto busy = std::chrono::steady_clock::now();
        uint32_t seen = wakes.load(std::memory_order_relaxed), spins = 0, taken = 0;
        // the latest publication, once new: quantized, then acknowledged after the last read of the buffer. The next
        // publication waits for the acknowledgement, so none is missed or overwritten while it is read.
        auto take_rows = [&] {
            const uint32_t n = raised.load(std::memory_order_acquire);
            if (!n || n == taken) return false;
            const uint32_t w = shape.load(std::memory_order_relaxed);
            quantize((int) (w >> 8 & 0xffu), (int) (w >> 16));
            taken = n;
            acked.store(n, std::memory_order_release);
            return true;
        };
        for (;;) {
            if (take_rows()) {
                busy = std::chrono::steady_clock::now();
                continue;
            }
            if (const uint32_t w = ring.load(std::memory_order_acquire)) {
                const uint32_t visit = ticket.load(std::memory_order_relaxed);
                const uint32_t pub = ring_rows.load(std::memory_order_relaxed);
                ring.store(0, std::memory_order_relaxed);
                take_rows();   // the ring's release covers its rows' publication: one not seen before the ring is seen now
                run((int) (w >> 8 & 0xffu), (int) (w >> 16), pub != 0 && pub == taken);
                fin.store(visit, std::memory_order_release);
                busy = std::chrono::steady_clock::now();
                continue;
            }
            if (stop.load(std::memory_order_relaxed)) return;
            platform::cpu_relax();
            if ((++spins & 63u) != 0) continue;
            const auto now = std::chrono::steady_clock::now();
            if (const uint32_t wk = wakes.load(std::memory_order_acquire); wk != seen) {
                seen = wk;
                busy = now;
            }
            if (now - busy < kSpin) continue;
            // seq_cst against wake(): it sees `napping` or this sees its count, so a wake is never slept through
            std::unique_lock<std::mutex> lk(nap_mu);
            napping.store(true, std::memory_order_seq_cst);
            nap_cv.wait_for(lk, kNap, [&] {
                return wakes.load(std::memory_order_seq_cst) != seen || stop.load(std::memory_order_relaxed) ||
                       ring.load(std::memory_order_acquire) != 0 ||
                       raised.load(std::memory_order_acquire) != taken;
            });
            napping.store(false, std::memory_order_relaxed);
        }
    }
};

namespace {
void release_bell(std::FILE*) {
    std::lock_guard<std::mutex> lk(g_live_mu);
    if (g_abort) g_abort();
}
}  // namespace

CpuExperts::CpuExperts() : impl_(std::make_unique<Impl>()) {}
CpuExperts::~CpuExperts() = default;

std::unique_ptr<CpuExperts> CpuExperts::make(const InklingModel& m, std::string& err) {
    if (!m.spill().any()) return nullptr;
    const Geometry& g = m.geo();
    if (g.n_embd != kWidth || g.n_layers > 64 || g.n_used * kBatchRows > kMaxPairs) {
        err = "the CPU experts do not fit this geometry";
        return nullptr;
    }
    std::unique_ptr<CpuExperts> out(new CpuExperts());
    Impl& p = *out->impl_;
    p.model = &m;
    p.n_expert = g.n_expert;
    p.k = g.n_used;
    for (int l = 0; l < g.n_layers; ++l) {
        const LayerWeights& w = m.layer(l);
        cpu::NativeFmt f;
        if (!cpu::native_fmt(w.gu_type, w.d_type, g.n_embd, g.n_ff_exp, f, err)) return nullptr;
        if (f.bytes > w.blob_bytes) {
            err = "an expert's CPU form is larger than its blob";
            return nullptr;
        }
        f.gate_softcap = g.gate_softcap;
        f.up_softcap = g.up_softcap;
        p.fmt.push_back(f);
        kernels::NativeExpertLayout L = kernels::native_expert_layout(w.gu_type, w.d_type, g.n_embd, g.n_ff_exp);
        L.gate_softcap = g.gate_softcap;
        L.up_softcap = g.up_softcap;
        p.layout.push_back(L);
    }
    p.peer = m.peer_experts();
    p.kind.assign((size_t) kBatchRows * g.n_used, 0);
    p.jobs.resize((size_t) kBatchRows * g.n_used);
    p.job_of.assign((size_t) g.n_expert, -1);
    const size_t pairs = (size_t) kBatchRows * g.n_used;
    if (cudaStreamCreateWithFlags(&p.stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&p.ready, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&p.done, cudaEventDisableTiming) != cudaSuccess ||
        cudaHostAlloc((void**) &p.bell, sizeof(CpuBell), cudaHostAllocMapped) != cudaSuccess ||
        cudaHostGetDevicePointer((void**) &p.bell_dev, p.bell, 0) != cudaSuccess ||
        cudaHostAlloc((void**) &p.input, (size_t) kBatchRows * kWidth * 4, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
        cudaHostGetDevicePointer((void**) &p.input_dev, p.input, 0) != cudaSuccess ||
        cudaHostAlloc((void**) &p.ids, pairs * 4, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
        cudaHostGetDevicePointer((void**) &p.ids_dev, p.ids, 0) != cudaSuccess ||
        cudaHostAlloc((void**) &p.out, pairs * kWidth * 4, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
        cudaHostGetDevicePointer((void**) &p.out_dev, p.out, 0) != cudaSuccess ||
        cudaMalloc(&p.seq, 2 * sizeof(*p.seq)) != cudaSuccess ||
        cudaMemsetAsync(p.seq, 0, 2 * sizeof(*p.seq), p.stream) != cudaSuccess) {
        cudaGetLastError();
        err = "cannot set up the CPU experts' buffers";
        return nullptr;
    }
    const model::LoadRequest& fl = m.flags();
    const cpu::PoolAffinity affinity = fl.pool_affinity_given ? fl.pool_affinity : cpu::PoolAffinity::All;
    p.pool = std::make_unique<cpu::ExpertPool>(cpu::start_pool_workers(fl.pool_workers, fl.pool_affinity_given, affinity), affinity);
    *p.bell = CpuBell{};
    p.server = std::thread([&p] { p.serve(); });
    {   // published once everything it touches is live
        std::lock_guard<std::mutex> lk(g_live_mu);
        g_owner = &p;
        g_abort = [&p] { p.abort_and_drain(); };
    }
    basalt::core::release_gpu_fn().store(&release_bell);
    basalt::log::info("basalt inkling2", "experts in RAM run on the CPU: %d pool workers%s", p.pool->workers(),
                      p.peer ? ", beside the second card's" : "");
    return out;
}

void CpuExperts::rows(int l, int rows, const float* input, cudaStream_t stream) {
    Impl& p = *impl_;
    must(cudaEventRecord(p.ready, stream), "fork");
    must(cudaStreamWaitEvent(p.stream, p.ready, 0), "fork wait");
    moe_cpu_rows(input, rows, l, p.input_dev, p.bell_dev, p.seq, p.stream);
}

void CpuExperts::start(int l, int rows, const int32_t* ids, cudaStream_t stream) {
    Impl& p = *impl_;
    must(cudaEventRecord(p.ready, stream), "fork");
    must(cudaStreamWaitEvent(p.stream, p.ready, 0), "fork wait");
    moe_cpu_ring(ids, rows, p.k, p.model->layer(l).blob_table, l, p.ids_dev, p.bell_dev, p.seq, p.stream);
    must(cudaEventRecord(p.done, p.stream), "join");
}

void CpuExperts::finish(int l, int rows, const int32_t* ids, float* out, uint32_t* tally, cudaStream_t stream) {
    Impl& p = *impl_;
    must(cudaStreamWaitEvent(stream, p.done, 0), "join wait");   // the ring has advanced `seq` before the merge reads it
    moe_cpu_merge(ids, rows, p.k, p.model->layer(l).blob_table, p.out_dev, out, tally, stream, p.bell_dev, p.seq);
}

bool CpuExperts::enter() { return impl_->gate.enter(); }
bool CpuExperts::leave(cudaError_t synced, bool joined) {
    // the bell's abort word too: a merge the abort released was let go through that word
    const bool released = std::atomic_ref<uint32_t>(impl_->bell->abort).load(std::memory_order_acquire) != 0;
    return impl_->gate.leave(synced, joined) && !released;
}
bool CpuExperts::stopped() { return impl_->gate.closed(); }

void CpuExperts::wake() {
    Impl& p = *impl_;
    p.wakes.fetch_add(1, std::memory_order_seq_cst);
    if (p.napping.load(std::memory_order_seq_cst)) {
        std::lock_guard<std::mutex> lk(p.nap_mu);
        p.nap_cv.notify_one();
    }
}

}  // namespace basalt::models::inkling2
