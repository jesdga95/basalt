#include "experts.hpp"
#include "basalt/prefill/prefill.hpp"
#include "basalt/core/mtp.hpp"
#include "basalt/core/native_head.hpp"
#include "basalt/prefill/moe_fp4.hpp"

namespace basalt::prefill {
using namespace detail;

void Prefill::Run::Chunk::Experts::Group::fp4_flush() {
    if (fp4_n == 0) return;
    if (gg_nslots > 0) cudaStreamWaitEvent(m.cs, m.copied[gg_slots[gg_nslots - 1]], 0);
    const Fp4Ptrs fp = fp4_ptrs(m.Xq, m.Hq, m.T, m.g->n_expert);
    // runs of consecutive experts of at most the rows H holds: H indexes a run's own rows, the
    // layer's Dm and row -> token table the absolute ones
    fp4::Group fg;
    fg.n = fp4_n;
    for (int i = 0; i < fp4_n; ++i) fg.blob[i] = fp4_blob[i];
    fg.bounds_host = m.bounds_host.data() + fp4_j0;
    fg.bounds_dev = m.bounds_dev + fp4_j0;
    fg.ids = m.src_dev; fg.a4x = fp.a4x; fg.gsx = fp.gsx; fg.tokens = T;
    fg.down_off = lay.fmt[(size_t) l].down_off;
    fg.tail_off = lay.fmt[(size_t) l].bytes - lay.fmt[(size_t) l].tail;
    fg.h = m.H; fg.dm = m.Dm;
    fg.a4h = fp.a4h; fg.gsh = fp.gsh; fg.toh = fp.toh; fg.rel = fp.rel;
    fp4::run_group(fg, (int64_t) group_rows_cap((size_t) m.T), m.cs);
    if (gg_nslots > 0) {   // the blobs are read: give the ring slots back
        const int rel_slot = gg_slots[gg_nslots - 1];
        cudaEventRecord(m.used[rel_slot], m.cs);
        for (int i = 0; i < gg_nslots; ++i) m.used_of[gg_slots[i]] = rel_slot;
        gg_nslots = 0;
    }
    fp4_n = 0;
}

bool Prefill::Run::Chunk::Experts::Group::compute_fp4(size_t j, const uint8_t* blob_dev, int slot) {
    if (fp4_n == 0) fp4_j0 = j;
    fp4_blob[fp4_n++] = blob_dev;
    if (slot >= 0) gg_slots[gg_nslots++] = slot;
    // groups of MMQ_GROUP experts: the streamed walk's, and the staged walk's, whose ring
    // holds two groups (each copy is waited for before its expert joins the group)
    if ((group_gather || !stream_all) && j % MMQ_GROUP != MMQ_GROUP - 1 && j + 1 < order.size())
        return true;
    fp4_flush();
    return true;

}

}  // namespace basalt::prefill
