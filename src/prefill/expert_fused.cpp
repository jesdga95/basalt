#include "experts.hpp"
#include "basalt/prefill/prefill.hpp"
#include "basalt/core/mtp.hpp"
#include "basalt/prefill/moe_fused.hpp"
#include "basalt/prefill/moe_fused_iq.hpp"
#include "basalt/core/peer_experts.hpp"
#include <algorithm>
#include <cmath>

namespace basalt::prefill {
using namespace detail;

bool Prefill::Run::Chunk::Experts::fused() {
    if (static bool said = false; !said) {
        said = true;
        basalt::log::info("basalt", "prompt experts on the fused int8 kernels");
    }
    // the layer's input to int8 once per token; the rows of each expert from the router's ids
    if (fused_nat) fused::quantize_act_native(m.mixed, T, N, m.Xq, m.cs);
    else fused::quantize_act(m.mixed, T, N, m.Xq, m.cs);
    fused::group(m.ids, T * K, (int) K, (int) m.g->n_expert, m.GU, m.slot_dev, m.src_dev, m.cs);
    // launches over the experts in id order, each at most kMaxBatch experts of which at most a
    // third of the ring streamed: the next batch's blobs arrive while one computes.  An expert
    // the routing did not pick has no tiles; its ring slot is given back with its batch.
    const size_t per = (size_t) std::max(1, m.ring / 3);
    size_t k = seq_start[(size_t) l];
    const size_t kend = seq_start[(size_t) l + 1];
    for (int32_t e = 0; e < m.g->n_expert;) {
        fused::Batch b;
        b.e0 = e;
        const size_t k0 = k;
        for (; e < m.g->n_expert && e - b.e0 < fused::kMaxBatch; ++e) {
            if (k < kend && seq[k].e == e) {
                if (k - k0 == per) break;
                b.blob[e - b.e0] = m.stage_dev[k % (size_t) m.ring];
                ++k;
            } else {                  // not streamed: resident (the walk streams all others)
                b.blob[e - b.e0] = m.cache->device_slot(m.host_res[(size_t) l * m.g->n_expert + e]);
                // counted whether routed or not (the routing stays on the GPU): at a streamed
                // chunk's size (>= 1024 tokens x 10 of 512) nearly every expert is routed
                ++owner.stats_.experts_resident;
            }
        }
        b.e1 = e;
        if (k > k0) {
            // one copy stream, in order: the batch's last blob covers the others
            wait_issued(k - 1);
            cudaStreamWaitEvent(m.cs, m.copied[(k - 1) % (size_t) m.ring], 0);
        }
        if (fused_nat) {
            const auto& f = lay.fmt[(size_t) l];
            const fused::NativeGeom ng{f.gu_type, f.d_type, f.gu_row, f.d_row, f.up_off, f.down_off};
            fused::experts_native(b, ng, (int) m.g->n_expert, T * K, m.GU, m.Xq, m.src_dev, m.H,
                                  m.Dm, m.cs);
        } else {
            fused::experts(b, (int) m.g->n_expert, T * K, m.GU, m.Xq, m.src_dev, m.H, m.Dm, m.cs);
        }
        for (size_t kk = k0; kk < k; ++kk) {
            cudaEventRecord(m.used[kk % (size_t) m.ring], m.cs);
            m.used_of[kk % (size_t) m.ring] = (int) (kk % (size_t) m.ring);
        }
        if (k > k0) {
            consumed = k;
            give_back(consumed);
        }
    }
    return true;
}

}  // namespace basalt::prefill
