#include "experts.hpp"
#include "basalt/prefill/prefill.hpp"
#include "basalt/core/mtp.hpp"
#include "basalt/core/progress.hpp"
#include "basalt/core/layout.hpp"
#include "basalt/prefill/moe_fused.hpp"
#include "basalt/prefill/moe_fp4.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/prefill/kernels.hpp"
#include "basalt/log.hpp"
#include <algorithm>
#include <string>
#include <vector>

namespace basalt::prefill {
using namespace detail;

bool Prefill::Run::Chunk::moe(int64_t l, const core::LayerView& v) {
    // ======================= MoE =======================
    const core::WeightRef *wr = need(v, "ffn_gate_inp.weight", err),
                          *wgi = need(v, "ffn_gate_inp_shexp.weight", err),
                          *wsg = need(v, "ffn_gate_shexp.weight", err),
                          *wsu = need(v, "ffn_up_shexp.weight", err),
                          *wsd = need(v, "ffn_down_shexp.weight", err);
    if (!wr || !wgi || !wsg || !wsu || !wsd) return false;
    if (!bf16_proj(m.gemm, wr, m.mixed_bf, m.logits, T, v.name("ffn_gate_inp.weight"), err)) return false;
    route(m.logits, m.ids, m.w, T, m.g->n_expert, m.cs);
    if (m.hist_on) route_count(m.ids, T * K, m.hist + l * m.g->n_expert, (int) m.g->n_expert, m.cs);
    if (ahead_n > 0 && l + 1 < LE) {   // the next layer's routing guessed (the logits are free again)
        const core::LayerView vn(*m.wt, l + 1);
        const core::WeightRef* wrn = vn.get("ffn_gate_inp.weight");
        std::string e2;
        if (wrn != nullptr && bf16_proj(m.gemm, wrn, m.mixed_bf, m.logits, T, vn.name("ffn_gate_inp.weight"), e2)) {
            route(m.logits, m.guess_ids, m.guess_w, T, m.g->n_expert, m.cs);
            copy_i32(m.guess_host_dev, m.guess_ids, T * K, m.cs);
            guess_for = l + 1;
        }
    }
    if (!shared_expert(v, wgi, wsg, wsu, wsd)) return false;
    Experts experts(*this, l);
    return experts.execute();
}

bool Prefill::Run::Chunk::Experts::Group::group() {
    // group the (token, k) pairs by expert on the host
    // (the sync below also orders this layer's writes of slot/src/bounds after the previous
    // layer's kernels that read them)
    if (grp_mapped) copy_i32(m.grp_dev, m.ids, T * K, m.cs);
    else cudaMemcpyAsync(m.ids_host.data(), m.ids, (size_t) T * K * 4, cudaMemcpyDeviceToHost, m.cs);
    // CPU share (small chunks, experts with a host copy): the layer's activations go to the host on
    // their own stream while the routing is read
    if (!cpu_download()) return false;
    // a stall here is the GPU (this layer's attention and router, or the previous layer's
    // work), not the host: the watchdog's report says so (only its text changes)
    core::progress_at("reading the prompt (batched): waiting for the GPU (attention, router) at layer",
                      l, p0);
    cudaStreamSynchronize(m.cs);
    core::progress_at("reading the prompt (batched): layer", l, p0);
    if (guess_for == l + 1) {   // the guessed rows per expert of the next layer
        m.guess_rows.assign((size_t) m.g->n_expert, 0);
        for (int64_t i = 0; i < T * K; ++i) {
            const int32_t e = m.guess_host[(size_t) i];
            if (e >= 0 && e < (int32_t) m.g->n_expert) ++m.guess_rows[(size_t) e];
        }
        m.guess_layer = l + 1;
    }
    std::fill(m.cnt.begin(), m.cnt.end(), 0);
    for (int64_t i = 0; i < T * K; ++i) {
        const int32_t e = ids_h[(size_t) i];
        if (e < 0 || e >= m.g->n_expert) { err = "prefill: routed id out of range"; return false; }
        ++m.cnt[(size_t) e];
    }
    // multi-GPU: the rows of the experts the peer computes go last, as one block [rows_local, T*K)
    if (ppc && pre_mmq) {
        on_peer.assign((size_t) m.g->n_expert, 0);
        for (int32_t e = 0; e < m.g->n_expert; ++e) {
            const int32_t c = m.cnt[(size_t) e];
            if (c == 0 || (m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0) ||
                !ppc->peer || !ppc->peer->has(l, e))
                continue;
            if (rows_peer + c > ppc->cap_rows) { continue; }
            on_peer[(size_t) e] = 1;
            rows_peer += c;
        }
        rows_local = T * K - rows_peer;
    }
    cpu_select();
    {
        int32_t r = 0;
        for (int32_t e = 0; e < m.g->n_expert; ++e)
            if ((on_peer.empty() || !on_peer[(size_t) e]) && (on_cpu.empty() || !on_cpu[(size_t) e])) {
                m.off[(size_t) e] = r;
                r += m.cnt[(size_t) e];
            }
        // the peer's rows in order_peer's order: its groups need them contiguous
        for (int32_t e = 0; e < m.g->n_expert && !on_peer.empty(); ++e)
            if (on_peer[(size_t) e]) { m.off[(size_t) e] = r; r += m.cnt[(size_t) e]; }
        for (int32_t e = 0; e < m.g->n_expert && !on_cpu.empty(); ++e)   // the CPU's block, last
            if (on_cpu[(size_t) e]) { m.off[(size_t) e] = r; r += m.cnt[(size_t) e]; }
        m.off[(size_t) m.g->n_expert] = r;
    }
    std::vector<int32_t> fill(m.off.begin(), m.off.end() - 1);
    for (int64_t i = 0; i < T * K; ++i) {
        const int32_t e = ids_h[(size_t) i];
        const int32_t p = fill[(size_t) e]++;
        slot_h[(size_t) i] = p;
        src_h[(size_t) p] = (int32_t) (i / K);
    }
    if (grp_mapped) {
        copy_i32(m.slot_dev, m.grp_dev + m.grp_tk, T * K, m.cs);
        copy_i32(m.src_dev, m.grp_dev + 2 * m.grp_tk, T * K, m.cs);
    } else {
        cudaMemcpyAsync(m.slot_dev, m.slot_host.data(), (size_t) T * K * 4, cudaMemcpyHostToDevice, m.cs);
        cudaMemcpyAsync(m.src_dev, m.src_host.data(), (size_t) T * K * 4, cudaMemcpyHostToDevice, m.cs);
    }
    // the experts, in id order: resident ones from VRAM, the others through the staging ring
    for (int32_t e = 0; e < m.g->n_expert; ++e)
        if (m.cnt[(size_t) e] > 0 && (on_cpu.empty() || !on_cpu[(size_t) e]))
            (!on_peer.empty() && on_peer[(size_t) e] ? order_peer : order).push_back(e);
    // CPU share: the pool runs the CPU's experts while this thread stages and runs the GPU's; their
    return true;
}

}  // namespace basalt::prefill
