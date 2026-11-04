#include "experts.hpp"
#include "basalt/prefill/prefill.hpp"
#include "basalt/core/mtp.hpp"
#include "basalt/core/native_head.hpp"
#include "basalt/prefill/gemm.hpp"
#include "basalt/prefill/moe_mmq.hpp"
#include "basalt/prefill/moe_fp4.hpp"
#include "basalt/prefill/kernels.hpp"
#include "basalt/log.hpp"
#include <algorithm>
#include <cstring>

namespace basalt::prefill {
using namespace detail;

void Prefill::Run::Chunk::Experts::Group::prepare_products() {
    if (use_mmq) {
        // the layer's activations as q8_1 rows in expert order, straight from `mixed`
        if (!fp4_l) mmq::quantize(m.mixed, m.src_dev, m.Xq, mmq_gt, N, N, T * K, m.cs);
        // each group's rows: absolute bounds (gate/up reads the layer's rows), relative ones (down
        // reads the group's own quantized H)
        const size_t n = order.size(), ng = (n + MMQ_GROUP - 1) / MMQ_GROUP;
        m.bounds_host.resize(n + 1 + ng * (MMQ_GROUP + 1));
        for (size_t j = 0; j < n; ++j) m.bounds_host[j] = m.off[(size_t) order[j]];
        m.bounds_host[n] = (int32_t) rows_local;
        for (size_t g = 0; g < ng; ++g)
            for (size_t i = 0; i <= MMQ_GROUP; ++i)
                m.bounds_host[n + 1 + g * (MMQ_GROUP + 1) + i] =
                    m.bounds_host[std::min(n, g * MMQ_GROUP + i)] - m.bounds_host[g * MMQ_GROUP];
        if (grp_mapped) {
            int32_t* bh = m.grp_host + 3 * m.grp_tk;
            std::memcpy(bh, m.bounds_host.data(), m.bounds_host.size() * 4);
            copy_i32(m.bounds_dev, m.grp_dev + 3 * m.grp_tk, (int64_t) m.bounds_host.size(), m.cs);
        } else {
            cudaMemcpyAsync(m.bounds_dev, m.bounds_host.data(), m.bounds_host.size() * 4,
                            cudaMemcpyHostToDevice, m.cs);
        }
        if (fp4_l) {   // the activations as FP4, once per token: the gate/up product reads a row's token
            const Fp4Ptrs fp = fp4_ptrs(m.Xq, m.Hq, m.T, m.g->n_expert);
            fp4::quantize_tokens(m.mixed, T, N, N, fp.a4x, fp.gsx, m.cs);
        }
    } else {
        gather_rows16(m.mixed_h, m.src_dev, m.Xs, T * K, N, m.cs);
    }
    // multi-GPU: the peer's share, enqueued before the primary's own experts so both cards work at once
}

void Prefill::Run::Chunk::Experts::Group::flush() {
    if (fp4_l) { fp4_flush(); return; }
    if (gg.n <= gg.first) return;
    const auto& f = lay.fmt[(size_t) l];
    if (gg_nslots > 0) {   // the copies land in order: the last one covers the others
        cudaStreamWaitEvent(m.cs, m.copied[gg_slots[gg_nslots - 1]], 0);
    }
    float* sc = f.tail ? grp_scales(m.grp_d) : nullptr;
    if (!mmq::gather_native_group(gg, f.up_off, mmq_gub / 2, f.down_off, mmq_db, m.grp_gu, mmq_gub,
                                  m.grp_d, mmq_db, m.cs, sc ? f.bytes - f.tail : 0, sc)) {
        for (int i = gg.first; i < gg.n; ++i) {   // not 16-byte aligned: one at a time
            const uint8_t* b = gg.blob[i];
            mmq::gather_native(b, b + f.up_off, mmq_gub / 2, b + f.down_off, mmq_db,
                               m.grp_gu + i * mmq_gub, m.grp_d + i * mmq_db, m.cs,
                               sc ? b + f.bytes - f.tail : nullptr, sc ? sc + 4 * i : nullptr);
        }
    }
    if (gg_nslots > 0) {
        const int rel = gg_slots[gg_nslots - 1];
        cudaEventRecord(m.used[rel], m.cs);
        for (int i = 0; i < gg_nslots; ++i) m.used_of[gg_slots[i]] = rel;
    }
    gg_nslots = 0;
    gg.first = gg.n;
}

bool Prefill::Run::Chunk::Experts::Group::compute_mmq(size_t j, const uint8_t* blob_dev, int slot) {
    // gather the expert into its group slot (GGUF blocks, unchanged or converted)
    const size_t q = j % MMQ_GROUP;
    if (group_gather) {
        gg.blob[q] = blob_dev;
        gg.n = (int) q + 1;
        if (slot >= 0) gg_slots[gg_nslots++] = slot;
        if (q + 1 < MMQ_GROUP && j + 1 < order.size()) return true;
        flush();
        gg = mmq::GatherGroup{};
    } else if (lay.native) {
        const auto& f = lay.fmt[(size_t) l];
        mmq::gather_native(blob_dev, blob_dev + f.up_off, mmq_gub / 2, blob_dev + f.down_off,
                           mmq_db, m.grp_gu + q * mmq_gub, m.grp_d + q * mmq_db, m.cs,
                           f.tail ? blob_dev + f.bytes - f.tail : nullptr,
                           grp_scales(m.grp_d) + 4 * q);
    } else {
        mmq::gather_q2(blob_dev, m.grp_gu + q * mmq_gub, m.grp_d + q * mmq_db, m.cs);
    }
    if (slot >= 0 && !group_gather) { cudaEventRecord(m.used[slot], m.cs); m.used_of[slot] = slot; }
    if (q + 1 < MMQ_GROUP && j + 1 < order.size()) return true;
    // the group's products: gate/up, swiglu, the group's H to q8_1, down
    const size_t j0 = j - q, g = j0 / MMQ_GROUP, n = order.size();
    const int ngx = (int) (q + 1);
    const int64_t r0 = m.bounds_host[j0], nr = m.bounds_host[j + 1] - r0;
    int64_t maxr = 0;
    for (size_t i = j0; i <= j; ++i) maxr = std::max<int64_t>(maxr, m.cnt[(size_t) order[i]]);
    // the zeroed tail after the group's last expert (see MMQ_TAIL)
    cudaMemsetAsync(m.grp_gu + (size_t) ngx * mmq_gub, 0, MMQ_TAIL, m.cs);
    cudaMemsetAsync(m.grp_d + (size_t) ngx * mmq_db, 0, MMQ_TAIL, m.cs);
    mmq::Product gu;
    gu.w = m.grp_gu; gu.type = mmq_gt; gu.w_rows = 1280; gu.w_cols = N; gu.expert_bytes = mmq_gub;
    gu.n = ngx; gu.xq = m.Xq; gu.bounds = m.bounds_dev + j0; gu.ids = m.ids_identity;
    gu.total_rows = T * K; gu.max_rows = maxr; gu.dst = m.GU; gu.ld_dst = 1280;
    m.mmq_ctx->run(gu, m.cs);
    const bool scaled = lay.native && lay.fmt[(size_t) l].tail != 0;
    mmq::swiglu(m.GU + r0 * 1280, m.H + r0 * 640, nr, 640, !lay.native, m.cs,
                scaled ? grp_scales(m.grp_d) : nullptr,
                m.bounds_dev + n + 1 + g * (MMQ_GROUP + 1), ngx);
    mmq::quantize(m.H + r0 * 640, nullptr, m.Hq, mmq_dt, 640, 640, nr, m.cs);
    mmq::Product dn;
    dn.w = m.grp_d; dn.type = mmq_dt; dn.w_rows = N; dn.w_cols = 640; dn.expert_bytes = mmq_db;
    dn.n = ngx; dn.xq = m.Hq; dn.bounds = m.bounds_dev + n + 1 + g * (MMQ_GROUP + 1);
    dn.ids = m.ids_identity; dn.total_rows = nr; dn.max_rows = maxr; dn.dst = m.Dm + r0 * N;
    dn.ld_dst = N;
    m.mmq_ctx->run(dn, m.cs);
    return true;

}

}  // namespace basalt::prefill
