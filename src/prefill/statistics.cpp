#include "state.hpp"
#include "layout.hpp"
#include "basalt/prefill/prefill.hpp"
#include "basalt/core/mtp.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/kernels/ngram.hpp"
#include "basalt/core/layer.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/log.hpp"
#include <algorithm>
#include <vector>

namespace basalt::prefill {
using namespace detail;

void Prefill::route_hist_reset() {
    Impl& m = *impl_;
    if (m.g == nullptr) return;
    {
        const core::OnDevice od(m.device);
        if (m.hist == nullptr) {
            m.hist_n = m.g->n_layers * m.g->n_expert;
            if (cudaMalloc((void**) &m.hist, (size_t) m.hist_n * 4) != cudaSuccess) { m.hist = nullptr; m.hist_on = false; }
            else m.owned.push_back(m.hist);
        }
        if (m.hist != nullptr) {
            cudaStreamSynchronize(m.cs);
            cudaMemset(m.hist, 0, (size_t) m.hist_n * 4);
        }
        m.hist_tokens = 0;
        m.hist_on = m.hist != nullptr;
    }
    if (next_ != nullptr) next_->route_hist_reset();
}

bool Prefill::route_hist_take(std::vector<uint32_t>& out, int64_t& tokens) {
    Impl& m = *impl_;
    if (!m.hist_on) return false;
    m.hist_on = false;
    {
        const core::OnDevice od(m.device);
        cudaStreamSynchronize(m.cs);
        std::vector<uint32_t> h((size_t) m.hist_n);
        if (cudaMemcpy(h.data(), m.hist, (size_t) m.hist_n * 4, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
        if (out.size() != h.size()) out.assign(h.size(), 0u);
        for (size_t i = 0; i < h.size(); ++i) out[i] += h[i];
        tokens = std::max(tokens, m.hist_tokens);
    }
    if (next_ != nullptr) next_->route_hist_take(out, tokens);
    return true;
}

void Prefill::set_ple_warm(const int64_t* tokens, int64_t run_pos0, int64_t from, int64_t to) {
    Impl& m = *impl_;
    if (m.warm.fut.valid()) m.warm.fut.wait();
    m.warm.pending = false;
    if (m.ss == nullptr || !m.ss->ple.ready() || m.ss->ple.table == nullptr || from < 2) return;
    const int64_t cap = (int64_t) (m.ss->ple.table->cache_rows() / (2 * basalt::kernels::PLE_N_HEADS));
    to = std::min(to, from + cap);
    if (to <= from) return;
    m.warm.tok.assign(tokens + from - 2, tokens + to);
    m.warm.run_pos0 = run_pos0;
    m.warm.to = to;
    m.warm.pending = true;
}

}  // namespace basalt::prefill
