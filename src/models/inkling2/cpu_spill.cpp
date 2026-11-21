#include "models/inkling2/cpu_spill.hpp"

#include "basalt/core/device.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/kernels/cpu/cpu_quota.hpp"
#include "basalt/log.hpp"
#include "basalt/models/inkling2/kernels.hpp"
#include "models/inkling2/model.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace basalt::models::inkling2 {
namespace cpu = kernels::cpu;
namespace {
constexpr int kUsed = 8;
constexpr int kPairs = kDecodeRows * kUsed;
void checked(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
}  // namespace

struct CpuSpill::Impl {
    const InklingModel* model = nullptr;
    std::unique_ptr<cpu::ExpertPool> pool;
    std::vector<cpu::NativeFmt> formats;
    cudaStream_t stream = nullptr;
    cudaEvent_t ready = nullptr, done = nullptr;
    float* input = nullptr;
    int32_t* ids = nullptr;
    float* output = nullptr;
    float* output_device = nullptr;
    unsigned long long* table = nullptr;
    unsigned long long* table_device = nullptr;
    bool initialized = false, active = false;
    std::atomic<bool> failed{false};
    char error[256] = {};
    alignas(64) std::array<std::array<uint8_t, cpu::kNativeActBytes>, kDecodeRows> acts;
    std::array<cpu::ExpertJobMulti, kPairs> jobs;
    struct Call { Impl* owner; int layer, rows; };
    std::vector<Call> calls;

    ~Impl() {
        if (stream) cudaStreamSynchronize(stream);
        pool.reset();
        if (table_device) cudaFree(table_device);
        for (void* p : {static_cast<void*>(input), static_cast<void*>(ids), static_cast<void*>(output), static_cast<void*>(table)})
            if (p) cudaFreeHost(p);
        if (ready) cudaEventDestroy(ready);
        if (done) cudaEventDestroy(done);
        if (stream) cudaStreamDestroy(stream);
    }

    void open(const InklingModel& m, int workers, cpu::PoolAffinity affinity) {
        model = &m;
        const auto& g = m.geo();
        if (g.n_used != kUsed) throw std::runtime_error("unsupported CPU expert count");
        pool = std::make_unique<cpu::ExpertPool>(workers, affinity);
        formats.resize(g.n_layers);
        for (int l = 0; l < g.n_layers; ++l) {
            const auto& w = m.layer(l);
            std::string reason;
            if (!cpu::native_fmt(w.gu_type, w.d_type, g.n_embd, g.n_ff_exp, formats[l], reason))
                throw std::runtime_error(reason);
            if (formats[l].bytes > w.blob_bytes) throw std::runtime_error("CPU expert exceeds its backing blob");
            formats[l].gate_softcap = g.gate_softcap;
            formats[l].up_softcap = g.up_softcap;
            for (int rows = 1; rows <= kDecodeRows; ++rows) calls.push_back({this, l, rows});
        }
        checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        checked(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        checked(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
        checked(cudaMallocHost(reinterpret_cast<void**>(&input), kDecodeRows * kWidth * sizeof(float)));
        checked(cudaMallocHost(reinterpret_cast<void**>(&ids), kPairs * sizeof(int32_t)));
        checked(cudaHostAlloc(reinterpret_cast<void**>(&output), kPairs * kWidth * sizeof(float), cudaHostAllocMapped));
        checked(cudaHostGetDevicePointer(reinterpret_cast<void**>(&output_device), output, 0));
        const size_t bytes = (size_t) g.n_layers * g.n_expert * sizeof(unsigned long long);
        checked(cudaMallocHost(reinterpret_cast<void**>(&table), bytes));
        checked(cudaMalloc(reinterpret_cast<void**>(&table_device), bytes));
        std::memset(table, 0, bytes);
    }

    void run(int layer, int rows) {
        std::array<int, 128> group;
        group.fill(-1);
        std::array<bool, kDecodeRows> quantize{};
        int n = 0;
        for (int p = 0; p < rows * kUsed; ++p) {
            const int e = ids[p], t = p / kUsed;
            if (e < 0 || e >= model->geo().n_expert) throw std::runtime_error("CPU expert id out of range");
            if (table[layer * model->geo().n_expert + e]) continue;
            int& at = group[e];
            if (at < 0) {
                at = n++;
                jobs[at] = cpu::ExpertJobMulti{};
                jobs[at].blob = model->host_blob(layer, e);
                if (!jobs[at].blob) throw std::runtime_error("CPU expert has no RAM backing");
            }
            auto& job = jobs[at];
            if (job.nt >= kDecodeRows) throw std::runtime_error("CPU expert has repeated token routing");
            job.nact[job.nt] = acts[t].data();
            job.out[job.nt++] = output + (size_t) p * kWidth;
            quantize[t] = true;
        }
        for (int t = 0; t < rows; ++t)
            if (quantize[t]) cpu::native_quant_act(formats[layer], input + (size_t) t * kWidth, acts[t].data());
        if (n) pool->run_split_multi_native(formats[layer], jobs.data(), n);
    }

    static void CUDART_CB execute(void* arg) noexcept {
        auto& c = *static_cast<Call*>(arg);
        if (c.owner->failed.load(std::memory_order_acquire)) return;
        try {
            c.owner->run(c.layer, c.rows);
        } catch (const std::exception& e) {
            std::snprintf(c.owner->error, sizeof(c.owner->error), "%s", e.what());
            c.owner->failed.store(true, std::memory_order_release);
        } catch (...) {
            std::snprintf(c.owner->error, sizeof(c.owner->error), "CPU expert execution failed");
            c.owner->failed.store(true, std::memory_order_release);
        }
    }
};

CpuSpill::CpuSpill() : impl_(std::make_unique<Impl>()) {}
CpuSpill::~CpuSpill() = default;

std::unique_ptr<CpuSpill> CpuSpill::make(const InklingModel& model, std::string& err) {
    // Unknown or wider links keep the card's mapped-RAM path.
    const double link = core::device_rates(model.device()).link_bytes;
    if (!model.spill().any() || !(link > 0 && link <= 8e9)) return nullptr;
    const auto& flags = model.flags();
    const auto affinity = flags.pool_affinity_given ? flags.pool_affinity : cpu::PoolAffinity::All;
    int workers = flags.pool_workers;
    if (workers <= 0) {
        const auto topo = cpu::detect_cpu_topology(true);
        workers = cpu::auto_pool_workers((int) topo.worker_cores.size(), topo.p_cores, topo.e_cores, topo.is_hybrid,
                                        cpu::cgroup_cpu_quota(), flags.pool_affinity_given ? affinity : cpu::PoolAffinity::Auto);
    }
    workers = std::max(1, workers / model.sessions());
    if (workers < 2) return nullptr;
    try {
        auto out = std::unique_ptr<CpuSpill>(new CpuSpill());
        out->impl_->open(model, workers, affinity);
        log::info("basalt inkling2", "CPU spill: %d workers, host link %.2f GB/s", workers, link / 1e9);
        return out;
    } catch (const std::exception& e) {
        err = e.what();
        return nullptr;
    }
}

bool CpuSpill::refresh(void* stream) {
    auto& p = *impl_;
    bool changed = !p.initialized;
    p.active = false;
    const auto& g = p.model->geo();
    for (int l = 0; l < g.n_layers; ++l)
        for (int e = 0; e < g.n_expert; ++e) {
            const auto at = reinterpret_cast<unsigned long long>(p.model->resident_blob(l, e));
            const int i = l * g.n_expert + e;
            if (p.table[i] != at) { p.table[i] = at; changed = true; }
            p.active |= at == 0;
        }
    if (changed) checked(cudaMemcpyAsync(p.table_device, p.table, (size_t) g.n_layers * g.n_expert * 8, cudaMemcpyHostToDevice, (cudaStream_t) stream));
    p.initialized = true;
    return p.active;
}

const unsigned long long* CpuSpill::blobs(int layer) const { return impl_->table_device + layer * impl_->model->geo().n_expert; }

void CpuSpill::start(int layer, int rows, const float* input, const int32_t* ids, void* stream) {
    auto& p = *impl_;
    checked(cudaEventRecord(p.ready, (cudaStream_t) stream));
    checked(cudaStreamWaitEvent(p.stream, p.ready, 0));
    checked(cudaMemcpyAsync(p.input, input, (size_t) rows * kWidth * 4, cudaMemcpyDeviceToHost, p.stream));
    checked(cudaMemcpyAsync(p.ids, ids, (size_t) rows * kUsed * 4, cudaMemcpyDeviceToHost, p.stream));
    checked(cudaLaunchHostFunc(p.stream, Impl::execute, &p.calls[layer * kDecodeRows + rows - 1]));
    checked(cudaEventRecord(p.done, p.stream));
}

void CpuSpill::finish(int layer, int rows, const int32_t* ids, float* output, uint32_t* tally, void* stream) {
    checked(cudaStreamWaitEvent((cudaStream_t) stream, impl_->done, 0));
    moe_cpu_merge(ids, rows, kUsed, blobs(layer), impl_->output_device, output, tally, stream);
}

bool CpuSpill::ok(std::string& err) const {
    if (!impl_->failed.load(std::memory_order_acquire)) return true;
    err = impl_->error;
    return false;
}

}  // namespace basalt::models::inkling2
