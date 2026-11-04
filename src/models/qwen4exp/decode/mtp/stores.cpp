#include "common.hpp"

namespace basalt::core {

using namespace mtp_detail;

int MtpDrafter::kv_layout(KvArr* out) const {
    const basalt::kernels::QsaShapes s = shapes_of(*g_);
    const uint64_t rows = (uint64_t) s.n_head_kv * (uint64_t) s.page_size;
    const uint64_t hd = (uint64_t) s.head_dim;
    if (st_.kv_hybrid) return 0;
    if (st_.kv_q4) {
        const uint64_t q4 = rows * basalt::kernels::kv_q4_bytes_per_head((int) s.head_dim);
        out[0] = {st_.k_q4, q4};
        out[1] = {st_.v_q4, q4};
        return 2;
    }
    if (st_.kv_int8) {
        const uint64_t sc = rows * (hd / basalt::kernels::KV_Q8_GROUP) * 2;
        out[0] = {st_.k_q, rows * hd};
        out[1] = {st_.v_q, rows * hd};
        out[2] = {st_.k_scale, sc};
        out[3] = {st_.v_scale, sc};
        return 4;
    }
    out[0] = {st_.k_pool, rows * hd * 2};
    out[1] = {st_.v_pool, rows * hd * 2};
    return 2;
}

void MtpDrafter::note_cells(int64_t lo, int64_t hi) {
    hi = std::min(hi, st_.max_cells);
    if (hi <= lo) return;
    if (dirty_hi_ <= dirty_lo_) { dirty_lo_ = lo; dirty_hi_ = hi; }
    else { dirty_lo_ = std::min(dirty_lo_, lo); dirty_hi_ = std::max(dirty_hi_, hi); }
    if (store_hi_.empty()) return;
    // the sequence ends where its latest write ends; a write that starts well below the end (another prompt, a
    // rolled-back turn) starts it again there, past the cells a window writes ahead of its position
    int64_t& end = store_hi_[(size_t) cur_];
    end = lo + 4 * (int64_t) max_t_ + 64 < end ? hi : std::max(end, hi);
}

void MtpDrafter::copy_pages(int to, bool to_work, int from, bool from_work, int64_t a, int64_t b, const KvArr* lay,
                            int na) {
    auto at = [&](int id, bool work, int i, int64_t p) {
        const int64_t ring = work ? 0 : store_ring_[(size_t) id];
        return (uint8_t*) (work ? lay[i].p : stores_[(size_t) id][(size_t) i]) +
               (uint64_t) (ring > 0 ? p % ring : p) * lay[i].page_bytes;
    };
    auto run = [&](int id, bool work, int64_t p) {   // pages from p before that side's ring wraps
        const int64_t ring = work ? 0 : store_ring_[(size_t) id];
        return ring > 0 ? ring - p % ring : b - p;
    };
    for (int64_t p = a; p < b;) {
        const int64_t n = std::min({b - p, run(to, to_work, p), run(from, from_work, p)});
        for (int i = 0; i < na; ++i)
            cudaMemcpyAsync(at(to, to_work, i, p), at(from, from_work, i, p), (size_t) ((uint64_t) n * lay[i].page_bytes),
                            cudaMemcpyDeviceToDevice, cs_);
        p += n;
    }
}

bool MtpDrafter::use_store(int id, std::string& err) {
    if (id == cur_) return true;
    if (id < 0 || id > n_stores_) { err = "mtp: no such K/V store"; return false; }
    const OnDevice on_device(device_);
    KvArr lay[4];
    const int na = kv_layout(lay);
    const int64_t ps = shapes_of(*g_).page_size;
    if (dirty_hi_ > dirty_lo_) {   // the written pages go back to the sequence that held them, as far as its ring reaches
        const size_t c = (size_t) cur_;
        const int64_t pg1 = std::min<int64_t>(store_pages_[c], (std::min(dirty_hi_, store_hi_[c]) + ps - 1) / ps);
        copy_pages(cur_, false, cur_, true, std::max<int64_t>(dirty_lo_ / ps, pg1 - store_ring_[c]), pg1, lay, na);
    }
    // the new sequence's last pages: all its window reads
    const int64_t n1 = std::min<int64_t>(store_pages_[(size_t) id], (store_hi_[(size_t) id] + ps - 1) / ps);
    copy_pages(id, true, id, false, std::max<int64_t>(0, n1 - store_ring_[(size_t) id]), n1, lay, na);
    cur_ = id;
    dirty_lo_ = dirty_hi_ = 0;
    if (cudaGetLastError() != cudaSuccess) { err = "mtp: swapping the K/V failed"; return false; }
    return true;
}

bool MtpDrafter::copy_store(int from, int to, int64_t cells, std::string& err) {
    if (n_stores_ == 0 || from == to) return true;
    if (from < 0 || from > n_stores_ || to < 0 || to > n_stores_) { err = "mtp: no such K/V store"; return false; }
    if (!flush(from, err)) return false;
    pend_tok_[(size_t) to].clear();
    const OnDevice on_device(device_);
    KvArr lay[4];
    const int na = kv_layout(lay);
    const int64_t ps = shapes_of(*g_).page_size;
    const int64_t n1 = std::min<int64_t>({store_pages_[(size_t) from], store_pages_[(size_t) to], (cells + ps - 1) / ps});
    // the last pages both sides keep (a store its ring, the working state all)
    const int64_t keep = std::min(from == cur_ ? n1 : store_ring_[(size_t) from], to == cur_ ? n1 : store_ring_[(size_t) to]);
    copy_pages(to, to == cur_, from, from == cur_, std::max<int64_t>(0, n1 - keep), n1, lay, na);
    if (to == cur_) note_cells(0, cells);
    store_hi_[(size_t) to] = std::min<int64_t>({cells, st_.max_cells, store_pages_[(size_t) to] * ps});
    if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = "mtp: copying the K/V failed"; return false; }
    return true;
}

bool MtpDrafter::save_store(int id, int64_t cells, ConversationKv& out, std::string& err) {
    if (n_stores_ == 0) return conversation_kv_save(out, st_, *g_, cells, false, err);
    if (id < 0 || id > n_stores_) { err = "mtp: no such K/V store"; return false; }
    const int prev = cur_;
    if (!pend_tok_[(size_t) id].empty()) {
        // its deferred rows are computed in the working state; the sequence there comes back whole within its ring
        note_cells(0, store_hi_[(size_t) prev]);
        if (!flush(id, err)) return false;
    }
    if (!conversation_kv_save(out, st_, *g_, cells, false, err)) return false;
    if (cur_ != id) {   // the pages the store keeps, over the working state's
        const OnDevice on_device(device_);
        KvArr lay[4];
        const int na = kv_layout(lay);
        ConversationBuffer* dst[4] = {&out.k, &out.v, &out.k_scale, &out.v_scale};
        const int64_t ps = shapes_of(*g_).page_size, ring = store_ring_[(size_t) id];
        const int64_t n1 = std::min<int64_t>(store_pages_[(size_t) id], (cells + ps - 1) / ps);
        for (int64_t p = std::max<int64_t>(0, n1 - ring); p < n1;) {
            const int64_t n = std::min(n1 - p, ring - p % ring);
            for (int i = 0; i < na; ++i) {
                const uint8_t* src = (const uint8_t*) stores_[(size_t) id][(size_t) i] +
                                     (uint64_t) (p % ring) * lay[i].page_bytes;
                const size_t off = (size_t) ((uint64_t) p * lay[i].page_bytes);
                if (!dst[i]->visit(off, (size_t) ((uint64_t) n * lay[i].page_bytes), [&](uint8_t* q, size_t m, size_t at) {
                        return cudaMemcpy(q, src + (at - off), m, cudaMemcpyDeviceToHost) == cudaSuccess;
                    })) {
                    err = "mtp: saving the K/V failed";
                    return false;
                }
            }
            p += n;
        }
    }
    return cur_ == prev || use_store(prev, err);
}

QsaState& MtpDrafter::kv_state_rw() {
    std::string e;
    if (!use_main(e)) basalt::log::error("basalt mtp", "%s", e.c_str());
    return st_;
}


}
