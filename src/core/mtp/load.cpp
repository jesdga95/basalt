#include "common.hpp"

namespace basalt::core {

using namespace mtp_detail;

namespace {

struct Bump {
    uint8_t* base = nullptr;
    uint64_t used = 0;
    template <typename T> T* take(uint64_t n) {
        T* p = base ? (T*) (base + used) : nullptr;
        used += (n * sizeof(T) + 255) & ~255ull;
        return p;
    }
};

bool mapped(size_t bytes, void** h, void** d) {
    if (cudaHostAlloc(h, bytes, cudaHostAllocMapped | (peer_portable() ? cudaHostAllocPortable : 0)) != cudaSuccess) return false;
    std::memset(*h, 0, bytes);
    return cudaHostGetDevicePointer(d, *h, 0) == cudaSuccess;
}

}  // namespace

bool MtpDrafter::load(const MtpFiles& files, const ModelGeometry& g, SessionState& ss, int max_t, std::string& err,
                      int64_t window) {
    cudaGetDevice(&device_);   // a layer split's last stage on another GPU: the drafter lives there
    g_ = &g;
    ss_ = &ss;
    if (ple_ss_ == nullptr) ple_ss_ = &ss;
    max_t_ = max_t;
    draft_vocab_ = files.draft_vocab;
    if (max_t < 1 || max_t > kMtpChainMax) { err = "mtp: max_t out of range"; return false; }
    // the two reads below are the whole "drafter files" cost; reported apart from the rest of the stage
    const auto t_files = std::chrono::steady_clock::now();
    // the draft layer's dense weights and experts, as one load phase
    std::optional<LoadProgress> progress;
    // ---- the index and the dense weights
    {
        if (!parse_index(files.index, tensors_, err)) return false;
        std::vector<uint8_t> blob;
        if (!read_file(files.dense, blob)) { err = "mtp: cannot read " + files.dense.path; return false; }
        if (basalt::kernels::hc8_only()) {   // the hyper-connection matrices as 8-bit copies only: out of the blob
            std::vector<uint8_t> kept;
            size_t staged = 0;
            for (Tensor& t : tensors_) {
                if (t.off + t.bytes > blob.size()) { err = "mtp: the dense index names bytes past the dense section: " + t.name; return false; }
                const bool hc = is_hc8_matrix(t);
                if (hc) {
                    // the 8-bit copy from the model file's table when it has one, else quantized from the bf16 matrix
                    t.dev = hc8_table_adopt("mtp." + t.name, nullptr, (int) t.rows, (int) t.cols);
                    if (t.dev == nullptr) {
                        void* tmp = nullptr;
                        if (cudaMalloc(&tmp, t.bytes) != cudaSuccess) { err = "mtp: no VRAM to stage " + t.name; return false; }
                        cudaMemcpy(tmp, blob.data() + t.off, t.bytes, cudaMemcpyHostToDevice);
                        t.dev = basalt::kernels::hc8_adopt((const uint16_t*) tmp, (int) t.rows, (int) t.cols);
                        cudaFree(tmp);
                    }
                    if (t.dev == nullptr) { err = "mtp: no VRAM for the 8-bit copy of " + t.name; return false; }
                    staged += t.bytes;
                    continue;
                }
                const size_t off = (kept.size() + 255) & ~(size_t) 255;
                kept.resize(off + t.bytes);
                std::memcpy(kept.data() + off, blob.data() + t.off, t.bytes);
                t.off = off;
            }
            blob.swap(kept);
            basalt::log::info("basalt hc8", "the draft layer's hyper-connection matrices as 8-bit copies only (%.1f MiB "
                                 "of bf16 left out)", (double) staged / 1048576.0);
        }
        const cudaError_t alloc = cudaMalloc((void**) &dense_, blob.size());
        if (alloc != cudaSuccess) {
            size_t free_bytes = 0, total_bytes = 0;
            const cudaError_t info = cudaMemGetInfo(&free_bytes, &total_bytes);
            err = "mtp: dense weights allocation failed (" + std::string(cudaGetErrorString(alloc)) +
                  "), requested " + std::to_string(blob.size() >> 20) + " MiB, CUDA0 free " +
                  (info == cudaSuccess ? std::to_string(free_bytes >> 20) + " MiB" : "unknown");
            return false;
        }
        progress.emplace(LoadProgress::cuda(device_), blob.size() + (uint64_t) g.n_expert * basalt::kernels::cpu::BLOB);
        cudaMemcpy(dense_, blob.data(), blob.size(), cudaMemcpyHostToDevice);
        progress->add(blob.size());
        vram_ += blob.size();
        dense_bytes_ = blob.size();
    }
    // ---- the 512 routed experts, one blob each
    {
        const uint64_t bytes = (uint64_t) g.n_expert * basalt::kernels::cpu::BLOB;
        if (files.experts.empty() || !files.experts.exists() || files.experts.size() < bytes) {
            err = "mtp: " + files.experts.path + " is missing or shorter than the draft layer's experts";
            return false;
        }
        if (cudaMalloc((void**) &experts_, bytes) != cudaSuccess) { err = "mtp: the 512 experts do not fit in VRAM"; return false; }
        std::string upload_err;
        if (!DirectUploader().upload(files.experts.path, files.experts.offset, bytes, experts_, upload_err)) {
            err = "mtp: cannot read the experts section: " + upload_err;
            return false;
        }
        progress->add(bytes);
        progress->finish();
        vram_ += bytes;
    }
    const char* required[] = {"fc_embedding.weight", "fc_hidden.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                              "self_attn.v_proj.weight", "self_attn.o_proj.weight", "mlp.shared_expert.gate_proj.weight",
                              "mlp.shared_expert.up_proj.weight", "mlp.shared_expert.down_proj.weight"};
    for (const char* n : required) if (!q8(n)) { err = std::string("mtp: ") + n + " is missing (q8_0)"; return false; }

    // ---- the layer's own K/V (dense attention: no indexer state is read)
    const basalt::kernels::QsaShapes s = shapes_of(g);
    const int64_t max_cells = ss.qsa_states[ss.qsa_primary()].max_cells;
    // KV streaming: the drafter only reads its last `window` cells, so with streaming on its K/V is a ring of the
    // window (plus the cells a round writes ahead of its queries) over a host copy, refilled on a resume. The host copy
    // is pinned after the expert arena has pinned what it could: if it does not fit, the K/V stays whole in VRAM.
    int64_t ring = (window > 0 && window < max_cells) ? window + 4 * (int64_t) max_t + 64 : 0;
    // K8V4 never applies to the drafter: its own attention paths (below, and src/core/verify/) handle whole formats
    // only, whatever ring shape it takes (0, a window, or the -1 fully-resident fallback).
    const bool kv_hybrid_was = qsa_kv_hybrid();
    const bool kv_int8_was = qsa_kv_int8();
    qsa_set_kv_hybrid(false);
    if (kv_hybrid_was) qsa_set_kv_int8(true);   // the drafter under --kv k8v4: plain INT8
    // the draft K/V is elastic like the layers' (its arena grows and shrinks with the session's), except where a
    // store per batch slot copies whole pools
    const int64_t kv_elastic_was = qsa_kv_elastic();
    if (n_stores_ > 0) qsa_set_kv_elastic(0);
    uint64_t sb = qsa_state_bytes(g, max_cells, false, ring);
    if (cudaMalloc(&state_arena_, sb) != cudaSuccess) { err = "mtp: the K/V state does not fit"; return false; }
    if (qsa_state_init(g, max_cells, state_arena_, st_, &ss.qsa_states[ss.qsa_primary()], ring) == 0) {
        if (st_.kv_mode == 0) { err = "mtp: state init failed"; return false; }
        basalt::log::warn("basalt mtp", "no pinned RAM left for the draft layer's K/V copy; keeping it in VRAM");
        cudaGetLastError();
        cudaFree(state_arena_);
        st_ = QsaState{};
        ring = -1;   // fully resident
        sb = qsa_state_bytes(g, max_cells, false, ring);
        if (cudaMalloc(&state_arena_, sb) != cudaSuccess) { err = "mtp: the K/V state does not fit"; return false; }
        if (qsa_state_init(g, max_cells, state_arena_, st_, &ss.qsa_states[ss.qsa_primary()], ring) == 0) { err = "mtp: state init failed"; return false; }
    }
    qsa_set_kv_int8(kv_int8_was);
    qsa_set_kv_hybrid(kv_hybrid_was);
    qsa_set_kv_elastic(kv_elastic_was);
    if (st_.kv_arena != nullptr) ss.kv_extra.push_back(st_.kv_arena);
    qsa_state_zero(st_, g, nullptr);
    cudaDeviceSynchronize();
    vram_ += sb;
    cur_ = n_stores_;
    uint64_t draft_kv_bytes = sb;
    if (n_stores_ > 0) {   // one K/V per slot beside the working one (batch windows)
        KvArr lay[4];
        const int na = kv_layout(lay);
        if (st_.kv_mode != 0 || na == 0) {
            basalt::log::warn("basalt mtp", "batch windows need the draft layer's K/V fully resident; no drafts there");
            n_stores_ = cur_ = 0;
        } else {
            stores_.assign((size_t) n_stores_ + 1, std::array<void*, 4>{});
            store_hi_.assign((size_t) n_stores_ + 1, 0);
            store_pages_.assign((size_t) n_stores_ + 1, st_.n_pages);
            const int64_t ps0 = shapes_of(g).page_size;
            for (size_t id = 0; id < store_cells_.size() && (int) id < n_stores_; ++id)
                store_pages_[id] = std::min<int64_t>(st_.n_pages, (store_cells_[id] + ps0 - 1) / ps0);
            // the drafter reads only its last `window` cells: a store keeps the last pages of its sequence, the window
            // and the cells a turn may roll back, as a ring (page p at p % ring)
            const int64_t ring_pages = (window > 0 && window < max_cells)
                                           ? (window + kStoreRollback + 4 * (int64_t) max_t + 64 + ps0 - 1) / ps0 + 1
                                           : st_.n_pages;
            store_ring_.assign((size_t) n_stores_ + 1, 0);
            for (size_t id = 0; id < stores_.size(); ++id) store_ring_[id] = std::min(store_pages_[id], ring_pages);
            for (size_t id = 0; id < stores_.size(); ++id)
                for (int i = 0; i < na; ++i) {
                    auto& st = stores_[id];
                    const uint64_t bytes = (uint64_t) store_ring_[id] * lay[i].page_bytes;
                    if (cudaMalloc(&st[(size_t) i], bytes) != cudaSuccess) { err = "mtp: the slots' K/V do not fit"; return false; }
                    cudaMemset(st[(size_t) i], 0, bytes);
                    vram_ += bytes;
                    draft_kv_bytes += bytes;
                }
        }
    }
    uint64_t draft_kv_max = draft_kv_bytes;
    if (st_.kv_arena != nullptr) {   // the elastic working K/V: what is backed now, and what the whole context takes
        draft_kv_bytes += st_.kv_arena->mapped_bytes();
        draft_kv_max += st_.kv_arena->bytes_for(st_.kv_arena->max_cells());
    }
    LoadProgress::kv(LoadProgress::cuda(device_), draft_kv_bytes, max_cells, "draft", draft_kv_max);

    // ---- buffers
    window_ = (window > 0 && window < max_cells) ? window : 0;
    cap_ = (((window_ > 0 ? window_ : max_cells) + 63) / 64) * 64;
    attn_scratch_floats_ = (int64_t) basalt::kernels::qsa_decode_attn_scratch_floats(cap_, s);
    const uint64_t T = (uint64_t) max_t, N = (uint64_t) g.n_embd, HC = (uint64_t) g.hc, K = (uint64_t) ss.k;
    const uint64_t NH = (uint64_t) g.n_head, HD = (uint64_t) g.head_dim, NKV = (uint64_t) g.n_head_kv;
    const uint64_t R2 = 2 * T;   // step/pos rows: T catch-up rows + up to T-2 chain steps
    bool ok = mapped(T * 4 + 64, (void**) &h_tok_, (void**) &m_tok_) &&
              mapped(R2 * 4 * 4 + 64, (void**) &h_step_, (void**) &m_step_) &&
              mapped(R2 * NH * 4 + 64, (void**) &h_pos_, (void**) &m_pos_) &&
              mapped(64, (void**) &h_row_, (void**) &m_row_) &&
              mapped(T * 4 + 64, (void**) &h_out_, (void**) &m_out_) &&
              mapped(T * 4 + 64, (void**) &h_prob_, (void**) &m_prob_);
    if (!ok) { err = "mtp: mapped staging failed"; return false; }
    auto carve = [&](Bump& b) {
        tok_ = b.take<int32_t>(T); step_ = b.take<int32_t>(R2 * 4); pos_ = b.take<int32_t>(R2 * NH); row_ = b.take<int32_t>(4);
        ident_ = b.take<int32_t>(T * (uint64_t) cap_);
        Rin_ = b.take<float>(T * HC * N); R_ = b.take<float>(T * HC * N);
        emb_ = b.take<float>(T * N); en_ = b.take<float>(T * N); e2_ = b.take<float>(T * N);
        hn_ = b.take<float>(T * HC * N); h2_ = b.take<float>(T * HC * N);
        mixed_ = b.take<float>(T * N); inj_ = b.take<float>(T * HC); inj2_ = b.take<float>(T * HC);
        lo_ = b.take<float>(T * (uint64_t) g.hc_lr); rs_ = b.take<float>(T * HC); bo_ = b.take<float>(T * N);
        xn_ = b.take<float>(T * HC * N);
        xq_ = b.take<uint8_t>(basalt::kernels::native_q8_1_bytes((int) (NH * HD), 8));
        xq2_ = b.take<uint8_t>(basalt::kernels::native_q8_1_bytes((int) (NH * HD), 8));
        qfull_ = b.take<float>(T * NH * 2 * HD); qcur_ = b.take<float>(T * NH * HD);
        kcur_ = b.take<float>(T * NKV * HD); vcur_ = b.take<float>(T * NKV * HD);
        attn_ = b.take<float>(T * NH * HD); attn32_ = b.take<float>(T * NH * HD);
        attn_scratch_ = b.take<float>((uint64_t) attn_scratch_floats_);   // the full layer runs one row at a time
        logits_ = b.take<float>(T * (uint64_t) g.n_expert); w_ = b.take<float>(T * K); ids_ = b.take<int32_t>(T * K);
        shared_ = b.take<float>(T * N); parts_ = b.take<float>(T * K * N); y_ = b.take<float>(T * N);
        sample_ = b.take<float>(T * N);
        hit_slot_ = b.take<int32_t>(T * K); hit_dst_ = b.take<int32_t>(T * K); hit_count_ = b.take<int32_t>(4);
        grp_ptr_ = b.take<unsigned long long>(T * K); grp_start_ = b.take<int32_t>(T * K + 1);
        grp_counts_ = b.take<int32_t>(4);
        hit_xq_ = b.take<uint8_t>(T * (N / 32) * 34); hit_xs_ = b.take<float>(T * (N / 32));
        hit_scratch_ = b.take<uint8_t>(basalt::kernels::moe_hit_grouped_scratch_bytes((int64_t) (T * K), g.n_embd, g.n_ff));
        sh_scratch_ = (float*) b.take<uint8_t>(basalt::kernels::shared_expert_scratch_bytes(g.n_ff));
        out_ids_ = b.take<int32_t>(T + 4);
        probs_ = b.take<float>(T + 4);
        dummy_inj_ = b.take<float>(HC);
    };
    Bump count;
    carve(count);
    if (cudaMalloc(&arena_, count.used) != cudaSuccess) { err = "mtp: buffers do not fit"; return false; }
    cudaMemset(arena_, 0, count.used);
    Bump real;
    real.base = (uint8_t*) arena_;
    carve(real);
    vram_ += count.used;
    if (n_stores_ > 0) {
        if (cudaMalloc((void**) &rstage_, T * HC * N * sizeof(float)) != cudaSuccess) { err = "mtp: row staging"; return false; }
        vram_ += T * HC * N * sizeof(float);
        if (cudaMalloc((void**) &rkeep_, kKeepRows * HC * N * sizeof(float)) != cudaSuccess) { err = "mtp: row copy"; return false; }
        vram_ += kKeepRows * HC * N * sizeof(float);
        pend_R_.assign((size_t) n_stores_ + 1, nullptr);
        pend_tok_.assign((size_t) n_stores_ + 1, std::vector<int32_t>{});
        pend_cell0_.assign((size_t) n_stores_ + 1, 0);
        for (float*& r : pend_R_) {
            if (cudaMalloc((void**) &r, (size_t) kPendRows * HC * N * sizeof(float)) != cudaSuccess) { err = "mtp: row staging"; return false; }
            vram_ += (uint64_t) kPendRows * HC * N * sizeof(float);
        }
    }
    {
        std::vector<int32_t> id((size_t) (T * (uint64_t) cap_));
        for (uint64_t t = 0; t < T; ++t)
            for (int64_t i = 0; i < cap_; ++i) id[(size_t) (t * (uint64_t) cap_ + (uint64_t) i)] = (int32_t) i;
        cudaMemcpy(ident_, id.data(), id.size() * 4, cudaMemcpyHostToDevice);
    }
    const int prio = 0;
    if (cudaStreamCreateWithPriority(&cs_, cudaStreamNonBlocking, prio) != cudaSuccess) { err = "mtp: stream"; return false; }
    if (cudaStreamCreateWithPriority(&cs2_, cudaStreamNonBlocking, prio) != cudaSuccess ||
        cudaEventCreateWithFlags(&ev_fork_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&ev_join_, cudaEventDisableTiming) != cudaSuccess) {
        err = "mtp: second stream";
        return false;
    }
    basalt::kernels::hc_fused_init();
    if (cudaStreamCreateWithPriority(&cs3_, cudaStreamNonBlocking, prio) != cudaSuccess ||
        cudaEventCreateWithFlags(&ev_pf_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&ev_pfj_, cudaEventDisableTiming) != cudaSuccess) {
        err = "mtp: prefetch stream";
        return false;
    }
    {
        if (basalt::kernels::hc8_enabled()) {   // the draft layer's reads on 8-bit copies too
            const int LR = (int) g.hc_lr, D = (int) (g.hc * g.n_embd), HCn = (int) g.hc;
            for (const char* p : {"attn_hyper_connection.", "mlp_hyper_connection.", "hyper_connection_mixer."}) {
                const std::string s(p);
                // (a copy read from the model file's table is registered the same way)
                auto prepare = [&](const std::string& n, int rows, int cols) {
                    const uint16_t* w = bf16(n.c_str());
                    if (w != nullptr && hc8_table_adopt("mtp." + n, w, rows, cols) != nullptr) return;
                    basalt::kernels::hc8_prepare(w, rows, cols);
                };
                prepare(s + "input_mix_weight_down.weight", LR, D);
                prepare(s + "input_mix_weight_up.weight", D, LR);
                prepare(s + "block_inject_weight.weight", HCn, D);
            }
        }
    }
    if (basalt::core::diagnostics().profile && cudaMalloc((void**) &gstamp_, 16) == cudaSuccess)
        cudaMemset(gstamp_, 0, 16);
    const double files_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_files).count();
    basalt::log::info("basalt mtp", "draft layer loaded, %.0f MiB of VRAM (experts %.0f, dense %.0f), files read in %.2f s (%.0f MiB/s)",
                 (double) vram_ / 1048576.0, (double) g.n_expert * basalt::kernels::cpu::BLOB / 1048576.0,
                 (double) tensors_.back().off / 1048576.0, files_s,
                 files_s > 0 ? ((double) g.n_expert * basalt::kernels::cpu::BLOB + (double) tensors_.back().off) /
                                   1048576.0 / files_s : 0.0);
    return true;
}


}
