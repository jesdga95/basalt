#include "run.hpp"
#include "basalt/prefill/prefill.hpp"
#include "basalt/core/mtp.hpp"
#include "basalt/core/layout.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/core/peer_experts.hpp"
#include <algorithm>
#include <utility>
#include <vector>

namespace basalt::prefill {
using namespace detail;

// A chunk below stream_all_min(): each layer's MoE half also runs the next layer's router on its FFN rows, and the
// experts that guess gives more rows than the CPU share takes (most first; not resident, not the peer's) are
// copied ahead into the ring's slots past STAGE for that layer's parity, once this layer has every input of its
// combine (the CPU's and the peer's rows cross the same link): the link then works through this layer's end and
// the next one's attention half, when it would idle.  The next layer reads the ones its routing picked in place;
// what was copied ahead never changes what is computed or in which order.
void Prefill::Run::Chunk::copy_ahead(int64_t L) {
    if (ahead_n <= 0 || L >= LE || m.guess_layer != L) return;
    // a layer whose K/V streams from the host uploads it on the compute stream: no copies ahead of it
    if (core::is_qsa_layer(g, L)) {
        int64_t qi = 0;
        for (int64_t i = 0; i < L; ++i) qi += core::is_qsa_layer(g, i) ? 1 : 0;
        if (ss.qsa_states[(size_t) qi].kv_mode == 1) return;
    }
    if (m.ahead_ev == nullptr && cudaEventCreateWithFlags(&m.ahead_ev, cudaEventDisableTiming) != cudaSuccess) {
        m.ahead_ev = nullptr;
        return;
    }
    std::vector<std::pair<int32_t, int32_t>> want;   // (-rows, expert)
    for (int32_t e = 0; e < (int32_t) m.g->n_expert; ++e) {
        const int32_t c = m.guess_rows[(size_t) e];
        if (c <= basalt::kernels::cpu::MAXT || m.host_res[(size_t) L * m.g->n_expert + e] >= 0) continue;
        if (ppc && ppc->peer && ppc->peer->has(L, e)) continue;
        if (m.ring_peer && m.ring_peer->exclusive() && m.ring_peer->has(L, e)) continue;
        if (!m.src->pinned(L, e)) continue;
        want.emplace_back(-c, e);
    }
    std::sort(want.begin(), want.end());
    const int b = (int) (L & 1);
    std::vector<int32_t>& slot_of = m.ahead_slot[b];
    std::fill(slot_of.begin(), slot_of.end(), -1);
    m.ahead_layer[b] = L;
    cudaEventRecord(m.ahead_ev, m.cs);
    cudaStreamWaitEvent(m.copy, m.ahead_ev, 0);
    int k = 0;
    for (size_t i = 0; i < want.size() && k < ahead_n; ++i) {
        const int32_t e = want[i].second;
        const uint8_t* bl = m.src->blob(L, e);
        if (bl == nullptr) continue;
        const int sl = STAGE + b * ahead_n + k++;
        if (m.stage_live[sl]) cudaStreamWaitEvent(m.copy, m.used[m.used_of[sl]], 0);
        cudaMemcpyAsync(m.stage_dev[sl], bl, (size_t) lay0.blob_bytes(L), cudaMemcpyHostToDevice, m.copy);
        cudaEventRecord(m.copied[sl], m.copy);
        m.stage_live[sl] = true;
        slot_of[(size_t) e] = sl;
        ++owner.stats_.experts_streamed;
        ++owner.stats_.experts_dma;
    }
}

}  // namespace basalt::prefill
