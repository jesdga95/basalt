#include "experts.hpp"
#include "basalt/prefill/prefill.hpp"
#include "basalt/core/mtp.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/prefill/kernels.hpp"

namespace basalt::prefill {
using namespace detail;

bool Prefill::Run::Chunk::Experts::finish() {
    if (cpu_fut.valid()) {   // CPU share: its rows into Dm's tail
        cudaEventRecord(m.cpu_gev[1], m.cs);   // after the GPU's expert work
        if (!cpu_fut.get()) { err = "prefill: a CPU expert has no blob"; return false; }
        cudaMemcpyAsync(m.Dm + (size_t) (T * K - rows_cpu) * N, m.cpu_rows, (size_t) rows_cpu * N * sizeof(float),
                        cudaMemcpyHostToDevice, m.cs);
        owner.stats_.experts_cpu += n_cpu;
        owner.stats_.cpu_share = m.cpu_share_now;
    }
    if (peer_now) {   // multi-GPU: the peer's rows are in Dm (or in host memory)
        cudaStreamWaitEvent(m.cs, peer_fp4 ? ppc->ev_res : ppc->ev_done, 0);
    }
    if (!(l == 0 && ple_on)) copy_ahead(l + 1);   // layer 1's upload of its PLE rows goes first
    moe_combine(m.Dm, m.slot_dev, m.w, m.shared, m.sg, m.bo, T, m.cs);
    return true;
}

}  // namespace basalt::prefill
