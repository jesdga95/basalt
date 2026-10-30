#include "experts.hpp"
#include "basalt/prefill/prefill.hpp"
#include "basalt/core/mtp.hpp"
#include "basalt/core/progress.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/prefill/moe_fp4.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/log.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <utility>
#include <vector>
#include <future>

namespace basalt::prefill {
using namespace detail;

bool Prefill::Run::Chunk::Experts::Group::cpu_download() {
    if (cpu_maybe) {
        const size_t want = (size_t) T * N;
        if (m.cpu_x_n < want) {
            if (m.cpu_x) cudaFreeHost(m.cpu_x);
            m.cpu_x = nullptr;
            m.cpu_x_n = 0;
            if (cudaHostAlloc((void**) &m.cpu_x, want * sizeof(float), cudaHostAllocDefault) != cudaSuccess) {
                err = "prefill: cannot allocate the CPU experts' activations";
                return false;
            }
            m.cpu_x_n = want;
        }
        if (m.cpu_s == nullptr) {
            cudaStreamCreateWithFlags(&m.cpu_s, cudaStreamNonBlocking);
            cudaEventCreateWithFlags(&m.cpu_ev_in, cudaEventDisableTiming);
            cudaEventCreateWithFlags(&m.cpu_ev_x, cudaEventDisableTiming);
            cudaEventCreate(&m.cpu_gev[0]);
            cudaEventCreate(&m.cpu_gev[1]);
        }
        cudaEventRecord(m.cpu_ev_in, m.cs);
        cudaStreamWaitEvent(m.cpu_s, m.cpu_ev_in, 0);
        cudaMemcpyAsync(m.cpu_x, m.mixed, want * sizeof(float), cudaMemcpyDeviceToHost, m.cpu_s);
        cudaEventRecord(m.cpu_ev_x, m.cpu_s);
    }
    return true;
}

void Prefill::Run::Chunk::Experts::Group::cpu_select() {
    if (cpu_maybe) {
        // Non-resident experts with a host copy and few rows go to the CPU, fewest rows first, up to
        // the share at which the CPU and the GPU's streaming end together.
        std::vector<std::pair<int32_t, int32_t>> cand;   // (rows, expert)
        for (int32_t e = 0; e < m.g->n_expert; ++e) {
            const int32_t c = m.cnt[(size_t) e];
            if (c == 0 || (m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0) ||
                (!on_peer.empty() && on_peer[(size_t) e]) ||
                (m.ring_peer && m.ring_peer->exclusive() && m.ring_peer->has(l, e)))
                continue;
            ++n_stream;
            int fd = -1;
            uint64_t fo = 0;
            if (c <= basalt::kernels::cpu::MAXT && (m.src->pinned(l, e) || !m.src->file_extent(l, e, fd, fo)))
                cand.emplace_back(c, e);
        }
        std::sort(cand.begin(), cand.end());
        const size_t take = std::min(cand.size(), (size_t) std::llround(m.cpu_share_now * (double) n_stream));
        if (take > 0) {
            on_cpu.assign((size_t) m.g->n_expert, 0);
            for (size_t i = 0; i < take; ++i) {
                on_cpu[(size_t) cand[i].second] = 1;
                rows_cpu += cand[i].first;
            }
            n_cpu = (int64_t) take;
            rows_local -= rows_cpu;
        }
    }
}

bool Prefill::Run::Chunk::Experts::Group::cpu_start() {
    // unweighted rows land in Dm's row order, so the combine weights them like any other row
    if (rows_cpu > 0) {
        const size_t want = (size_t) rows_cpu * N;
        if (m.cpu_rows_n < want) {
            if (m.cpu_rows) cudaFreeHost(m.cpu_rows);
            m.cpu_rows = nullptr;
            m.cpu_rows_n = 0;
            const size_t cap = want * 2;   // headroom: few reallocations
            if (cudaHostAlloc((void**) &m.cpu_rows, cap * sizeof(float), cudaHostAllocDefault) != cudaSuccess) {
                err = "prefill: cannot allocate the CPU experts' rows";
                return false;
            }
            m.cpu_rows_n = cap;
        }
        const int64_t r0c = T * K - rows_cpu;
        const basalt::kernels::cpu::NativeFmt& cf = lay.fmt[(size_t) l];
        namespace kc = basalt::kernels::cpu;
        const bool q2 = kc::q2_native_kernels(cf.gu_type);
        if (q2 && m.cpu_actq.size() < (size_t) T) m.cpu_actq.resize((size_t) T);
        if (!q2 && m.cpu_nact.size() < (size_t) T * kc::kNativeActBytes) m.cpu_nact.resize((size_t) T * kc::kNativeActBytes);
        // the jobs are built here: the source's blob() is not for two threads at once
        m.cpu_need.assign((size_t) T, 0);   // the tokens the CPU's experts read
        m.cpu_jobs.clear();
        for (int32_t e = 0; e < m.g->n_expert; ++e) {
            if (!on_cpu[(size_t) e]) continue;
            kc::ExpertJobMulti j;
            j.blob = m.src->blob(l, e);
            if (j.blob == nullptr) { err = "prefill: expert source has no blob"; return false; }
            j.nt = m.cnt[(size_t) e];
            for (int r = 0; r < j.nt; ++r) {
                const int64_t pr = (int64_t) m.off[(size_t) e] + r;
                const int32_t tk = src_h[(size_t) pr];
                m.cpu_need[(size_t) tk] = 1;
                if (q2) j.act[r] = &m.cpu_actq[(size_t) tk];
                else j.nact[r] = m.cpu_nact.data() + (size_t) tk * kc::kNativeActBytes;
                j.out[r] = m.cpu_rows + (size_t) (pr - r0c) * N;
            }
            m.cpu_jobs.push_back(j);
        }
        cpu_fut = std::async(std::launch::async, [&m = m, &cf, &cpu_ms = cpu_ms, q2, T = T, &owner = owner]() -> bool {
            const auto t0 = Clock::now();
            cudaEventSynchronize(m.cpu_ev_x);   // the activations are on the host
            for (int64_t t = 0; t < T; ++t) {
                if (!m.cpu_need[(size_t) t]) continue;
                if (q2) kc::act_quant_any(m.cpu_x + (size_t) t * N, (int) N, m.cpu_actq[(size_t) t]);
                else kc::native_quant_act(cf, m.cpu_x + (size_t) t * N, m.cpu_nact.data() + (size_t) t * kc::kNativeActBytes);
            }
            owner.cpu_pool_->run_split_multi_native(cf, m.cpu_jobs.data(), (int) m.cpu_jobs.size());
            cpu_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            return true;
        });
    }
    return true;
}

}  // namespace basalt::prefill
