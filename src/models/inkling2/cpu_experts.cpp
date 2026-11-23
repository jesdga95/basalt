// src/models/inkling2/cpu_experts.cpp - see the header.
#include "models/inkling2/cpu_experts.hpp"

#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/log.hpp"
#include "basalt/models/inkling2/kernels.hpp"
#include "models/inkling2/model.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <vector>

namespace basalt::models::inkling2 {
namespace cpu = kernels::cpu;

static_assert(kDecodeRows <= cpu::MAXT, "a RAM expert takes every row of a window in one job");

struct CpuExperts::Impl {
    const InklingModel* model = nullptr;
    int n_expert = 0, k = 0;
    std::unique_ptr<cpu::ExpertPool> pool;
    std::vector<cpu::NativeFmt> fmt;               // [layer]
    cudaStream_t stream = nullptr;                 // the copies to RAM and the pool's turn, beside the card's experts
    cudaEvent_t ready = nullptr, done = nullptr;
    float* input = nullptr;                        // pinned [kDecodeRows][kWidth]
    int32_t* ids = nullptr;                        // pinned [kDecodeRows][k]
    float* out = nullptr;                          // pinned and mapped [kDecodeRows * k][kWidth]: a pair's row
    float* out_dev = nullptr;                      // its device address
    alignas(64) std::array<std::array<uint8_t, cpu::kNativeActBytes>, kDecodeRows> acts;
    std::vector<cpu::ExpertJobMulti> jobs;         // [kDecodeRows * k]
    std::vector<int> job_of;                       // [n_expert]: the expert's job in this call, -1: none
    struct Call {
        Impl* p;
        int layer, rows;
    };
    std::vector<Call> calls;                       // [layer][rows - 1]: the host function's argument, fixed for the graphs

    ~Impl() {
        if (stream) cudaStreamSynchronize(stream);
        pool.reset();
        for (void* q : {(void*) input, (void*) ids, (void*) out})
            if (q) cudaFreeHost(q);
        if (ready) cudaEventDestroy(ready);
        if (done) cudaEventDestroy(done);
        if (stream) cudaStreamDestroy(stream);
    }

    // On the host function's thread: the window's RAM pairs, every row of an expert in one job, on the pool.
    void run(int layer, int rows) {
        std::fill(job_of.begin(), job_of.end(), -1);
        bool quant[kDecodeRows] = {};
        int n = 0;
        for (int p = 0; p < rows * k; ++p) {
            const int e = ids[p], r = p / k;
            const uint8_t* blob = e >= 0 && e < n_expert ? model->ram_blob(layer, e) : nullptr;
            if (!blob) continue;   // on the card
            int& j = job_of[(size_t) e];
            if (j < 0) {
                j = n++;
                jobs[(size_t) j] = cpu::ExpertJobMulti{};
                jobs[(size_t) j].blob = blob;
            }
            cpu::ExpertJobMulti& job = jobs[(size_t) j];
            job.nact[job.nt] = acts[(size_t) r].data();
            job.out[job.nt++] = out + (size_t) p * kWidth;
            quant[r] = true;
        }
        for (int r = 0; r < rows; ++r)
            if (quant[r]) cpu::native_quant_act(fmt[(size_t) layer], input + (size_t) r * kWidth, acts[(size_t) r].data());
        if (n) pool->run_split_multi_native(fmt[(size_t) layer], jobs.data(), n);
    }
    static void CUDART_CB execute(void* arg) noexcept {
        const Call& c = *static_cast<const Call*>(arg);
        c.p->run(c.layer, c.rows);
    }
};

CpuExperts::CpuExperts() : impl_(std::make_unique<Impl>()) {}
CpuExperts::~CpuExperts() = default;

std::unique_ptr<CpuExperts> CpuExperts::make(const InklingModel& m, std::string& err) {
    if (!m.spill().any()) return nullptr;
    const Geometry& g = m.geo();
    if (g.n_embd != kWidth || g.n_layers > 64 || g.n_used * kDecodeRows > 1024) {
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
        for (int rows = 1; rows <= kDecodeRows; ++rows) p.calls.push_back({&p, l, rows});
    }
    p.jobs.resize((size_t) kDecodeRows * g.n_used);
    p.job_of.assign((size_t) g.n_expert, -1);
    const size_t pairs = (size_t) kDecodeRows * g.n_used;
    if (cudaStreamCreateWithFlags(&p.stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&p.ready, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&p.done, cudaEventDisableTiming) != cudaSuccess ||
        cudaMallocHost((void**) &p.input, (size_t) kDecodeRows * kWidth * 4) != cudaSuccess ||
        cudaMallocHost((void**) &p.ids, pairs * 4) != cudaSuccess ||
        cudaHostAlloc((void**) &p.out, pairs * kWidth * 4, cudaHostAllocMapped) != cudaSuccess ||
        cudaHostGetDevicePointer((void**) &p.out_dev, p.out, 0) != cudaSuccess) {
        cudaGetLastError();
        err = "cannot set up the CPU experts' buffers";
        return nullptr;
    }
    const model::LoadRequest& fl = m.flags();
    const cpu::PoolAffinity affinity = fl.pool_affinity_given ? fl.pool_affinity : cpu::PoolAffinity::All;
    p.pool = std::make_unique<cpu::ExpertPool>(cpu::start_pool_workers(fl.pool_workers, fl.pool_affinity_given, affinity), affinity);
    basalt::log::info("basalt inkling2", "experts in RAM run on the CPU: %d pool workers", p.pool->workers());
    return out;
}

void CpuExperts::start(int l, int rows, const float* input, const int32_t* ids, cudaStream_t stream) {
    Impl& p = *impl_;
    cudaEventRecord(p.ready, stream);
    cudaStreamWaitEvent(p.stream, p.ready, 0);
    cudaMemcpyAsync(p.input, input, (size_t) rows * kWidth * 4, cudaMemcpyDeviceToHost, p.stream);
    cudaMemcpyAsync(p.ids, ids, (size_t) rows * p.k * 4, cudaMemcpyDeviceToHost, p.stream);
    cudaLaunchHostFunc(p.stream, Impl::execute, &p.calls[(size_t) l * kDecodeRows + (size_t) rows - 1]);
    cudaEventRecord(p.done, p.stream);
}

void CpuExperts::finish(int l, int rows, const int32_t* ids, float* out, uint32_t* tally, cudaStream_t stream) {
    cudaStreamWaitEvent(stream, impl_->done, 0);
    moe_cpu_merge(ids, rows, impl_->k, impl_->model->layer(l).blob_table, impl_->out_dev, out, tally, stream);
}

}  // namespace basalt::models::inkling2
