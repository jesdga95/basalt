// src/core/mtp.cpp - see include/basalt/core/mtp.hpp.
#include "basalt/core/mtp.hpp"
#include "basalt/core/verify.hpp"
#include "basalt/core/diag.hpp"
#include "basalt/core/draft_cum.hpp"
#include "basalt/core/kv_arena.hpp"
#include "basalt/core/direct_upload.hpp"
#include "basalt/core/load_progress.hpp"
#include "basalt/core/on_device.hpp"

#include "basalt/core/native_head.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"
#include "basalt/kernels/bf16_gemv.hpp"
#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/kernels/elementwise.hpp"
#include "basalt/kernels/fused_gr.hpp"
#include "basalt/kernels/gr.hpp"
#include "basalt/kernels/hc_fused.hpp"
#include "basalt/kernels/kv_q4.hpp"
#include "basalt/kernels/kv_q8.hpp"
#include "basalt/kernels/native_moe.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/native_qsa.hpp"
#include "basalt/kernels/native_rope.hpp"
#include "basalt/kernels/native_router.hpp"
#include "basalt/kernels/qsa.hpp"
#include "basalt/kernels/qsa_decode_attn.hpp"
#include "basalt/kernels/qsa_dec_attn.hpp"
#include "basalt/kernels/qsa_dec_fused.hpp"
#include "basalt/kernels/q5k_head.hpp"
#include "basalt/kernels/q6k_head.hpp"
#include "basalt/kernels/quantize_act.hpp"
#include "basalt/kernels/rope.hpp"
#include "basalt/kernels/router_top10.hpp"
#include "basalt/kernels/s2_expert_grouped.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/kernels/shared_expert.hpp"
#include "basalt/kernels/verify_kernels.hpp"
#include "basalt/kernels/ngram.hpp"
#include "basalt/kernels/dense_tc.hpp"
#include "basalt/kernels/pdl.hpp"

#include <memory>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <optional>
#include <sstream>
#include <vector>
#include "basalt/platform/compiler.hpp"
#include "basalt/log.hpp"

namespace basalt::core {
namespace {

constexpr float EPS = 1e-6f;
constexpr int GGML_Q8_0 = 8;
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// The round and chain graphs run independent branches on a second stream (the embedding beside the hidden input,
// K/V beside Q, the shared expert beside the routed ones, the next step's hidden input beside the head); the same
// kernels on the same inputs, so the same bits.
constexpr bool dec_draft_par() { return true; }
// Every hyper-connection read of those graphs (and the MLP read's router) as the one-kernel read, whose bits are
// fused_gr_read_multi's.
bool dec_draft_hcf() {
    static const bool on = basalt::kernels::hc_fused_supported();
    return on;
}

// The one-kernel hyper-connection read of the rows fa[0..T) (rows at the layer's fixed strides)
basalt::kernels::HcFusedArgs hcf_args(const basalt::kernels::FusedGrArgs* fa, int T, float* xn) {
    basalt::kernels::HcFusedArgs h;
    h.T = T; h.R = fa[0].R; h.R_out = fa[0].R_out; h.apply = fa[0].apply;
    h.bo_prev = fa[0].bo_prev; h.inj_prev = fa[0].inj_prev;
    h.w_norm = fa[0].w_norm; h.w_down = fa[0].w_down; h.w_up = fa[0].w_up; h.w_inject = fa[0].w_inject;
    h.eps = fa[0].eps; h.lo = fa[0].lo; h.rs = fa[0].rs; h.inject_out = fa[0].w_inject ? fa[0].inject_out : nullptr;
    h.mixed = fa[0].mixed; h.xn = xn;
    return h;
}

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

basalt::kernels::QsaShapes shapes_of(const ModelGeometry& g) {
    basalt::kernels::QsaShapes s = basalt::kernels::qsa_real_shapes();
    s.n_head = g.n_head;
    s.n_head_kv = g.n_head_kv;
    s.head_dim = g.head_dim;
    s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    return s;
}

bool read_file(const FileRange& r, std::vector<uint8_t>& out) {
    if (r.empty() || !r.exists()) return false;
    // `fread` passes a request bigger than the stream buffer straight to the read call.
    FILE* f = basalt::platform::open_file_utf8(r.path, "rb");
    if (f == nullptr) return false;
    // A `FILE*` has no destructor and the short-read path below returns early: the guard closes on every path.
    struct Closer {
        FILE* f;
        ~Closer() { if (f != nullptr) std::fclose(f); }
    } closer{f};
    const uint64_t n = r.size();
    // The expert files are far larger than 4 GiB, so the seek has to be a 64-bit one.
    if (basalt::platform::fseek64(f, r.offset, SEEK_SET) != 0) return false;
    basalt::platform::advise_willneed(basalt::platform::file_fd(f), r.offset, n);
    out.resize((size_t) n);
    return n == 0 || std::fread(out.data(), 1, (size_t) n, f) == (size_t) n;
}

// spare rows the draft head carries after the base rows, for the head rows of tokens the subset lacks
constexpr int64_t kDraftCtxRows = 512;

}  // namespace

MtpDrafter::~MtpDrafter() {
    if (cs_) cudaStreamSynchronize(cs_);
    if (ev_done_) cudaEventDestroy(ev_done_);
    for (auto& e : prefill_exec_) if (e) cudaGraphExecDestroy(e);
    for (auto& e : prefill_dev_exec_) if (e) cudaGraphExecDestroy(e);
    if (pf_dev_) cudaFree(pf_dev_);
    for (auto& st : stores_)
        for (void* a : st) if (a) cudaFree(a);
    if (rstage_) cudaFree(rstage_);
    if (rkeep_) cudaFree(rkeep_);
    if (own_R_) cudaFree(own_R_);
    if (relay_) cudaFree(relay_);
    for (float* r : pend_R_) if (r) cudaFree(r);
    for (auto& row : round_exec_)
        for (auto& e : row) if (e) cudaGraphExecDestroy(e);
    for (auto& row : step_exec_)
        for (auto& e : row) if (e) cudaGraphExecDestroy(e);
    for (auto& row : cond_exec_)
        for (auto& e : row) if (e) cudaGraphExecDestroy(e);
    if (cs_) cudaStreamDestroy(cs_);
    if (cs2_) cudaStreamDestroy(cs2_);
    if (cs3_) cudaStreamDestroy(cs3_);
    if (ev_pf_) cudaEventDestroy(ev_pf_);
    if (ev_pfj_) cudaEventDestroy(ev_pfj_);
    if (ev_fork_) cudaEventDestroy(ev_fork_);
    if (ev_join_) cudaEventDestroy(ev_join_);
    if (dense_) cudaFree(dense_);
    if (experts_) cudaFree(experts_);
    if (state_arena_) cudaFree(state_arena_);
    if (arena_) cudaFree(arena_);
    if (head_logits_) cudaFree(head_logits_);
    if (dhead_) cudaFree(dhead_);
    if (dvocab_) cudaFree(dvocab_);
    if (top_scratch_) cudaFree(top_scratch_);
    if (proxy_) cudaFree(proxy_);
    if (proxy_scratch_) cudaFree(proxy_scratch_);
    if (gstamp_) cudaFree(gstamp_);
    void* hosts[] = {h_tok_, h_step_, h_pos_, h_row_, h_out_, h_prob_};
    for (void* h : hosts) if (h) cudaFreeHost(h);
}

const float* MtpDrafter::f32(const char* name) const {
    for (const auto& t : tensors_) if (t.name == name && t.kind == "f32") return (const float*) (dense_ + t.off);
    return nullptr;
}
const uint16_t* MtpDrafter::bf16(const char* name) const {
    for (const auto& t : tensors_)
        if (t.name == name && t.kind == "bf16") return t.dev ? (const uint16_t*) t.dev : (const uint16_t*) (dense_ + t.off);
    return nullptr;
}
const void* MtpDrafter::q8(const char* name) const {
    for (const auto& t : tensors_) if (t.name == name && t.kind == "q8_0") return dense_ + t.off;
    return nullptr;
}

bool MtpDrafter::parse_index(const FileRange& index, std::vector<Tensor>& out, std::string& err) {
    std::vector<uint8_t> idx_text;
    if (!read_file(index, idx_text)) { err = "mtp: cannot open " + index.path; return false; }
    std::istringstream idx(std::string(idx_text.begin(), idx_text.end()));
    std::string line;
    while (std::getline(idx, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream is(line);
        Tensor t;
        is >> t.name >> t.kind >> t.rows >> t.cols >> t.off >> t.bytes;
        if (!is) { err = "mtp: malformed dense index line: " + line; return false; }
        out.push_back(t);
    }
    return true;
}

bool MtpDrafter::is_hc8_matrix(const Tensor& t) {
    return t.kind == "bf16" && t.name.find("hyper_connection") != std::string::npos &&
           t.name.find("input_mix_weight_") != std::string::npos && (uint64_t) t.rows * (uint64_t) t.cols * 2 == t.bytes;
}

bool MtpDrafter::plan_bytes(const MtpFiles& files, const ModelGeometry& g, uint64_t& out, std::string& err) {
    uint64_t dense = files.dense.size();
    if (basalt::kernels::hc8_only()) {   // the dense blob without the matrices that are held as 8-bit copies only
        std::vector<Tensor> tensors;
        if (!parse_index(files.index, tensors, err)) return false;
        uint64_t kept = 0;
        for (const Tensor& t : tensors) {
            if (is_hc8_matrix(t)) continue;
            kept = ((kept + 255) & ~(uint64_t) 255) + t.bytes;
        }
        dense = kept;
    }
    out = dense + (uint64_t) g.n_expert * basalt::kernels::cpu::BLOB;
    return true;
}

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
            for (size_t id = 0; id < stores_.size(); ++id)
                for (int i = 0; i < na; ++i) {
                    auto& st = stores_[id];
                    const uint64_t bytes = (uint64_t) store_pages_[id] * lay[i].page_bytes;
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

std::vector<std::pair<std::string, const void*>> MtpDrafter::hc8_keys() const {
    std::vector<std::pair<std::string, const void*>> out;
    for (const Tensor& t : tensors_)
        if (t.kind == "bf16" && t.name.find("hyper_connection") != std::string::npos)
            out.emplace_back("mtp." + t.name, t.dev ? t.dev : (const void*) (dense_ + t.off));
    return out;
}

namespace {
// a Q5_K draft head keeps context rows after its base rows; a Q6_K head (no context rows) reads the main head's rows
// through its map instead of a copy
bool draft_ctx_rows(int head_type) { return head_type == 13; }
bool draft_rows_mapped(int head_type) { return head_type == 14 && !draft_ctx_rows(head_type); }
}  // namespace

uint64_t MtpDrafter::bind_bytes(uint64_t head_row_bytes, int head_type, int64_t n_vocab) const {
    uint64_t bytes = head_logits_ ? 0 : (uint64_t) max_t_ * (uint64_t) n_vocab * sizeof(float);
    if (dhead_ == nullptr && !rows_mapped_) {
        if (!draft_vocab_.empty() && draft_vocab_.exists()) {
            const long size = (long) draft_vocab_.size();
            if (size >= 4 && size % 4 == 0) {
                const uint64_t xr = (uint64_t) kDraftCtxRows;
                // a Q6_K head reads the main head's rows; others copy them
                if (!draft_rows_mapped(head_type)) bytes += (uint64_t) (size / 4 + xr) * head_row_bytes;
                bytes += (uint64_t) size + xr * 8;
                std::vector<uint8_t> raw;   // the 2-bit copy of the leading rows (the base rows and the context rows)
                if (read_file(draft_vocab_, raw) && raw.size() == (size_t) size) {
                    const int32_t* ids = (const int32_t*) raw.data();
                    int64_t n_base = size / 4;
                    for (int64_t i = 1; i < size / 4; ++i)
                        if (ids[i] < ids[i - 1]) { n_base = i; break; }
                    bytes += basalt::kernels::q5k_proxy_bytes(n_base + (int64_t) xr) +
                             basalt::kernels::q5k_proxy_scratch_bytes(n_base + (int64_t) xr);
                }
            }
        }
    }
    return bytes;
}

namespace {
// What to change when the draft head's token subset does not fit the VRAM left.  Text only, after the failure.
void draft_head_hint(int64_t n_tokens, uint64_t bytes) {
    size_t free_b = 0, total_b = 0;
    const bool have_free = cudaMemGetInfo(&free_b, &total_b) == cudaSuccess;
    std::string free_s;
    if (have_free) {
        char b[64];
        std::snprintf(b, sizeof(b), " and %.0f MiB is free", (double) free_b / 1048576.0);
        free_s = b;
    }
    basalt::log::warn("basalt mtp", "the draft head over %lld tokens needs %.0f MiB of VRAM%s.",
                 (long long) n_tokens, (double) bytes / 1048576.0, free_s.c_str());
    basalt::log::info("basalt mtp", "hint: a smaller draft vocabulary (--draft-vocab when converting the model), "
                         "a smaller --max-context, or closing what else uses the GPU leaves it room.");
}
}  // namespace

bool MtpDrafter::bind(const WeightTable& wt, const NativeHead* head, const float* window_R, std::string& err) {
    const OnDevice on_device(device_);
    wt_ = &wt;
    head_ = head;
    window_R_ = window_R;
    relay_src_ = nullptr;
    if (window_R != nullptr) {
        cudaPointerAttributes pa{};
        if (cudaPointerGetAttributes(&pa, window_R) == cudaSuccess && pa.type == cudaMemoryTypeDevice &&
            pa.device != device_) {
            const size_t row = (size_t) (g_->hc * g_->n_embd) * sizeof(float);
            if (own_R_ == nullptr &&
                (cudaMalloc((void**) &own_R_, (size_t) max_t_ * row) != cudaSuccess ||
                 cudaMalloc((void**) &relay_, (size_t) kRelayRows * row) != cudaSuccess)) {
                err = "mtp: the row relay does not fit";
                return false;
            }
            vram_ += (uint64_t) (max_t_ + kRelayRows) * row;
            relay_src_ = window_R;
            window_R_ = own_R_;
        }
        (void) cudaGetLastError();
    }
    const WeightRef* wo = wt.find("output.weight");
    if (!wo) { err = "mtp: output.weight is missing"; return false; }
    n_vocab_ = wo->ne1;
    if (head == nullptr || !head->loaded()) { err = "mtp: the draft layer needs the native head"; return false; }
    if (head_logits_ == nullptr &&
        cudaMalloc((void**) &head_logits_, (size_t) max_t_ * (size_t) n_vocab_ * sizeof(float)) != cudaSuccess) {
        err = "mtp: the draft logits do not fit";
        return false;
    }
    // the draft head's token subset, when the model file carries one
    if (dhead_ == nullptr && !rows_mapped_) {
        std::vector<uint8_t> raw;
        if (read_file(draft_vocab_, raw) && raw.size() >= 4 && raw.size() % 4 == 0) {
            const int64_t n_raw = (int64_t) (raw.size() / 4);
            const int32_t* ids = (const int32_t*) raw.data();
            for (int64_t i = 0; i < n_raw; ++i)
                if (ids[i] < 0 || ids[i] >= n_vocab_) {
                    err = "mtp: the draft vocabulary holds a token id outside the vocabulary";
                    return false;
                }
            // the base rows: the leading ids up to the first descent (added ids follow the base in id order)
            int64_t n_base = n_raw;
            for (int64_t i = 1; i < n_raw; ++i)
                if (ids[i] < ids[i - 1]) { n_base = i; break; }
            // context rows follow the base rows
            if (draft_ctx_rows(head->type()) && n_raw > 0) n_xcap_ = kDraftCtxRows;
            xbase_ = n_base;
            std::vector<int32_t> map((size_t) (n_raw + n_xcap_));
            std::copy(ids, ids + n_base, map.begin());
            std::fill(map.begin() + n_base, map.begin() + n_base + n_xcap_, -1);
            std::copy(ids + n_base, ids + n_raw, map.begin() + n_base + n_xcap_);
            n_dvocab_ = n_raw + n_xcap_;
            const int64_t row_bytes = (int64_t) head->row_bytes();   // a vocabulary row of the native head
            const bool mapped = draft_rows_mapped(head->type());
            if (cudaMalloc((void**) &dvocab_, map.size() * 4) != cudaSuccess ||
                (!mapped && cudaMalloc((void**) &dhead_, (size_t) (n_dvocab_ * row_bytes)) != cudaSuccess) ||
                (n_xcap_ > 0 && cudaMalloc((void**) &dxids_, (size_t) n_xcap_ * 4) != cudaSuccess)) {
                err = "mtp: the draft head does not fit";
                // the map, the copy unless mapped, the context rows' ids and the leading rows' 2-bit copy
                uint64_t need = map.size() * 4 + (uint64_t) n_xcap_ * 4;
                if (!mapped) need += (uint64_t) (n_dvocab_ * row_bytes);
                if (head->type() == 13 || head->type() == 14)
                    need += basalt::kernels::q5k_proxy_bytes(n_base + n_xcap_) +
                            basalt::kernels::q5k_proxy_scratch_bytes(n_base + n_xcap_);
                draft_head_hint(n_dvocab_, need);
                return false;
            }
            rows_mapped_ = mapped;
            if (mapped) {
                cudaMemcpy(dvocab_, map.data(), map.size() * 4, cudaMemcpyHostToDevice);
                vram_ += map.size() * 4;
                basalt::log::info("basalt mtp", "draft head over %lld tokens (rows read from the main head)",
                             (long long) n_dvocab_);
            } else {
                // unused context rows read the first row's weights for the gather, then take their id -1
                std::vector<int32_t> gmap = map;
                for (int32_t& v : gmap) if (v < 0) v = ids[0];
                cudaMemcpy(dvocab_, gmap.data(), gmap.size() * 4, cudaMemcpyHostToDevice);
                basalt::kernels::gather_rows((const uint8_t*) head->weights(), row_bytes, dvocab_, n_dvocab_, dhead_,
                                             nullptr);
                cudaDeviceSynchronize();
                cudaMemcpy(dvocab_, map.data(), map.size() * 4, cudaMemcpyHostToDevice);
                vram_ += (uint64_t) (n_dvocab_ * row_bytes) + map.size() * 4 + (uint64_t) n_xcap_ * 4;
                basalt::log::info("basalt mtp", "draft head over %lld tokens (%.1f MiB)", (long long) n_dvocab_,
                             (double) (n_dvocab_ * row_bytes) / 1048576.0);
            }
            if (exp_flag("spec_log", 0) != 0) diag_map_ = map;
            n_tier1_ = n_base + n_xcap_;
            if (head->type() == 13 || head->type() == 14) {   // the 2-bit copy of the leading rows: a fraction of the bytes
                const size_t pb = basalt::kernels::q5k_proxy_bytes(n_tier1_);
                const size_t sb = basalt::kernels::q5k_proxy_scratch_bytes(n_tier1_);
                if (cudaMalloc((void**) &proxy_, pb) == cudaSuccess && cudaMalloc(&proxy_scratch_, sb) == cudaSuccess) {
                    cudaMemset(proxy_, 0, pb);
                    if (head->type() == 13) basalt::kernels::q5k_proxy_pack(dhead_, 0, n_tier1_, proxy_, n_tier1_, nullptr);
                    else if (mapped)
                        basalt::kernels::q6k_proxy_pack(head->weights(), 0, n_tier1_, proxy_, n_tier1_, nullptr, dvocab_);
                    else basalt::kernels::q6k_proxy_pack(dhead_, 0, n_tier1_, proxy_, n_tier1_, nullptr);
                    cudaDeviceSynchronize();
                    vram_ += pb + sb;
                } else {
                    cudaGetLastError();
                    if (proxy_) cudaFree(proxy_);
                    proxy_ = nullptr;
                }
            }
            if (n_xcap_ > 0) {
                in_sub_.assign((size_t) n_vocab_, 0);
                for (int64_t i = 0; i < n_raw; ++i)
                    if (ids[i] >= 0 && ids[i] < n_vocab_) in_sub_[(size_t) ids[i]] = 1;
                basalt::log::info("basalt mtp", "draft head context rows: %lld", (long long) n_xcap_);
            }
            if (n_base < n_raw) {
                tier2_.assign((size_t) n_vocab_, 0);
                for (int64_t i = n_base; i < n_raw; ++i)
                    if (ids[i] >= 0 && ids[i] < n_vocab_) tier2_[(size_t) ids[i]] = 1;
                for (int64_t i = 0; i < n_base; ++i)
                    if (ids[i] >= 0 && ids[i] < n_vocab_) tier2_[(size_t) ids[i]] = 0;
                basalt::log::info("basalt mtp", "draft head tiers: %lld base rows, %lld added rows read after an "
                                     "added token", (long long) n_base, (long long) (n_raw - n_base));
            }
        }
    }
    if (diag_ == nullptr && exp_flag("spec_log", 0) != 0) {
        diag_stride_ = dvocab_ != nullptr ? n_dvocab_ : n_vocab_;
        if (cudaHostAlloc((void**) &diag_, (size_t) max_t_ * (size_t) diag_stride_ * sizeof(float), cudaHostAllocDefault) != cudaSuccess)
            diag_ = nullptr;
    }
    if (top_scratch_ == nullptr) {
        const size_t b = basalt::kernels::q5k_head_top1_scratch_bytes();
        if (cudaMalloc(&top_scratch_, b) != cudaSuccess) { err = "mtp: the draft head's argmax scratch does not fit"; return false; }
        cudaMemset(top_scratch_, 0, b);
        cudaDeviceSynchronize();
        vram_ += b;
    }
    return true;
}

// The hidden input branch: h2_ <- fc_hidden(rms(R) * w) per stream, T rows.
void MtpDrafter::record_hidden(const float* R, int T, uint8_t* xq, cudaStream_t cs) {
    using namespace basalt::kernels;
    const int64_t N = g_->n_embd, HC = g_->hc;
    native_qsa_rms_norm_weighted(R, f32("pre_fc_norm_hidden.weight"), hn_, (int) (HC * N), T, EPS, cs);
    for (int c0 = 0; c0 < T * HC; c0 += 8) {
        const int nc = (int) std::min<int64_t>(8, T * HC - c0);
        native_quantize_q8_1(hn_ + (size_t) c0 * N, xq, (int) N, nc, cs);
        native_mmvq(GGML_Q8_0, q8("fc_hidden.weight"), xq, h2_ + (size_t) c0 * N, (int) N, (int) N, nc, cs);
    }
}

// The layer for T rows.  full = false stops after the K/V append (the prompt only needs the cache).
bool MtpDrafter::record_forward(int T, int step_row0, cudaStream_t cs, std::string& err, bool reuse_row) {
    using namespace basalt::kernels;
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    // step_row0 >= 0: the full layer on step rows [step_row0, +T); step_row0 < 0: K/V only on rows [-1 - step_row0, +T)
    const bool full = step_row0 >= 0;
    const int row0 = full ? step_row0 : -1 - step_row0;
    if (full && T != 1) { err = "mtp: the full layer runs one row at a time (its attention scratch is sized for one)"; return false; }
    if (reuse_row && !full) { err = "mtp: only the full layer reuses a catch-up row"; return false; }
    const int64_t N = g.n_embd, HC = g.hc, K = ss.k, NH = g.n_head, HD = g.head_dim, NKV = g.n_head_kv;
    const QsaShapes s = shapes_of(g);
    const GrShapes gs{g.n_embd, g.hc, g.hc_lr};
    const int32_t* step = step_ + row0 * 4;
    const int32_t* pos = pos_ + row0 * NH;
    const bool kv_fused = st_.kv_int8 && !st_.kv_q4 && !st_.kv_rot &&
                          qsa_dec_fused_supported(s);
    const bool kv_par = par_ && full && !reuse_row && kv_fused;
    // chain-step kernels that start while the one before them still runs (the paths below all use the fused kernels)
    const bool pdl_ok = kv_par && !st_.kv_rot && qsa_dec_fused_supported(s) &&
                        qsa_dec_attn_supported(qsa_attn_pools(st_), s) && par_ && dec_draft_hcf() && have_pre_;
    std::unique_ptr<PdlScope> psc;
    auto pscope = [&](bool first_plain) {
        psc.reset();
        if (pdl_ok) psc.reset(new PdlScope(true, first_plain));
    };
    auto pscope_off = [&] { psc.reset(); };
    try {
        auto norm_rope = [&](float* data, const float* gamma, int rows, int cols, const int32_t* p) {
            native_qsa_rms_norm_weighted(data, gamma, data, cols, rows, EPS, cs);
            native_rope_apply(data, data, rows, cols, (int) s.n_rot, rope_scaling(), p, cs);
        };
        if (reuse_row) {
            // the catch-up already ran this row's input branches, attention hyper-connection and K/V append, with
            // the same per-row arithmetic: its R, injection and mixed rows become row 0
            mtp_take_row(R_, HC * N, inj_, HC, mixed_, N, row_, cs);
            native_quantize_q8_1(mixed_, xq_, (int) N, T, cs);
        } else {
            // ---- the two input branches
            const WeightRef* we = wt_->find("token_embd.weight");
            if (!we) { err = "mtp: token_embd.weight is missing"; return false; }
            // par_: the embedding branch on the second stream, beside the hidden one (or the hidden input the
            // previous step already computed beside its head)
            if (par_) fork(cs);
            const cudaStream_t ce = par_ ? cs2_ : cs;
            uint8_t* const xqe = par_ ? xq2_ : xq_;
            pscope(true);   // the embedding branch follows the fork
            if (const NativeEmbed* ne = native_embed()) {   // the GGUF-form table
                ne->gather_dev(tok_, T, emb_, ce);
            } else {
                const auto* codes = (const uint8_t*) we->data;
                const auto* scales = (const float*) (codes + we->codes_bytes);
                const auto* offsets = we->has_offset ? (const float*) (codes + we->codes_bytes + we->scales_bytes) : nullptr;
                embedding_gather_dev(codes, scales, offsets, tok_, T, we->ne0, we->code_bits, we->code_bias, we->group_elems,
                                     (uint64_t) (we->ne0 / (8 / we->code_bits)), (uint64_t) (we->ne0 / we->group_elems), emb_, ce);
            }
            native_qsa_rms_norm_weighted(emb_, f32("pre_fc_norm_embedding.weight"), en_, (int) N, T, EPS, ce);
            native_quantize_q8_1(en_, xqe, (int) N, T, ce);
            native_mmvq(GGML_Q8_0, q8("fc_embedding.weight"), xqe, e2_, (int) N, (int) N, T, ce);
            pscope_off();
            if (!(have_pre_ && full)) record_hidden(Rin_, T, xq_, cs);
            if (par_) join(cs);
            // the round's catch-up brings the attention weights into L2; a chain step prefetches none, so with its
            // query and head weights streamed the layer's other weights stay in L2 from step to step
            if (!full) pf_fork(cs);
            pscope(false);
            add_streams_broadcast(h2_, e2_, R_, N, (int) HC, T, cs);
            // ---- the attention hyper-connection
            bool q8_done = false;
            {
                FusedGrArgs fa[kFusedGrMaxT];
                for (int t = 0; t < T; ++t) {
                    fa[t].R = R_ + (size_t) t * HC * N; fa[t].R_out = R_ + (size_t) t * HC * N; fa[t].apply = false;
                    fa[t].w_norm = f32("attn_hyper_connection.hc_norm.weight");
                    fa[t].w_down = bf16("attn_hyper_connection.input_mix_weight_down.weight");
                    fa[t].w_up = bf16("attn_hyper_connection.input_mix_weight_up.weight");
                    fa[t].w_inject = bf16("attn_hyper_connection.block_inject_weight.weight");
                    fa[t].eps = EPS; fa[t].lo = lo_ + t * g.hc_lr; fa[t].rs = rs_ + t * HC;
                    fa[t].inject_out = inj_ + t * HC; fa[t].mixed = mixed_ + t * N;
                }
                if (par_ && dec_draft_hcf()) {
                    HcFusedArgs h = hcf_args(fa, T, xn_);
                    if (hc_fused_q8_rows_ok()) { h.xq = xq_; q8_done = true; }   // mixed's q8_1 rows from the read
                    h.next_early = pdl_ok && q8_done;   // the query projection's blocks start with the read
                    hc_fused_read(h, cs);
                } else if (T == 1)   // one row: the per-row read (its own rounding)
                    gr_read(R_, fa[0].w_norm, fa[0].w_down, fa[0].w_up, fa[0].w_inject, EPS, gs, ss.block.gr, mixed_, inj_,
                            cs);
                else
                    fused_gr_read_multi(fa, T, xn_, cs);
            }
            // ---- attention: K/V into the layer's own cache, then (full) dense attention over every cell
            if (!q8_done) native_quantize_q8_1(mixed_, xq_, (int) N, T, cs);
            if (kv_par) fork(cs);   // K/V on the second stream, beside Q
            const cudaStream_t ckv = kv_par ? cs2_ : cs;
            pscope(true);   // the K/V branch follows the fork
            native_mmvq(GGML_Q8_0, q8("self_attn.k_proj.weight"), xq_, kcur_, (int) N, (int) (NKV * HD), T, ckv);
            native_mmvq(GGML_Q8_0, q8("self_attn.v_proj.weight"), xq_, vcur_, (int) N, (int) (NKV * HD), T, ckv);
            if (kv_fused) {   // the rows' K norm + RoPE and appends in one launch
                QsaDecKvPost kp;
                kp.kcur = kcur_; kp.vcur = vcur_; kp.pos_k = pos; kp.pos_stride = (int) NH;
                kp.k_norm = f32("self_attn.k_norm.weight"); kp.steps = step;
                kp.k_q = st_.k_q; kp.v_q = st_.v_q; kp.k_scale = st_.k_scale; kp.v_scale = st_.v_scale;
                kp.page_table = st_.page_table; kp.host = st_.host; kp.eps = EPS;
                qsa_dec_kv_post(kp, T, s, rope_scaling(), ckv);
            } else
            for (int t = 0; t < T; ++t) {
                norm_rope(kcur_ + t * NKV * HD, f32("self_attn.k_norm.weight"), (int) NKV, (int) HD, pos + t * NH);
                if (st_.kv_rot) {   // rotated K and V (kv_q4.hpp): Q4_0, and INT8
                    fwht256_inplace_cuda(kcur_ + t * NKV * HD, NKV, cs);
                    fwht256_inplace_cuda(vcur_ + t * NKV * HD, NKV, cs);
                }
                // stored in the state's own format (rotated INT8 K/V are not appended as Q4_0)
                if (st_.kv_q4)
                    kv_append_q4_step(st_.k_q4, st_.v_q4, st_.page_table, step + t * 4, kcur_ + t * NKV * HD,
                                      vcur_ + t * NKV * HD, s, cs, &st_.host);
                else if (st_.kv_int8)
                    kv_append_q8_step(st_.k_q, st_.v_q, st_.k_scale, st_.v_scale, st_.page_table, step + t * 4,
                                      kcur_ + t * NKV * HD, vcur_ + t * NKV * HD, s, cs, &st_.host);
                else
                    kv_append_step(st_.k_pool, st_.v_pool, st_.page_table, step + t * 4, kcur_ + t * NKV * HD,
                                   vcur_ + t * NKV * HD, s, cs, &st_.host);
            }
        }
        if (!full) return true;
        pscope(false);   // the query follows the hc read (or its quantize)
        if (T == 1)   // read once a step: streamed past the layer's other weights in L2
            native_q8_0_mmvq_stream(q8("self_attn.q_proj.weight"), xq_, qfull_, (int) N, (int) (NH * 2 * HD), cs);
        else
            native_mmvq(GGML_Q8_0, q8("self_attn.q_proj.weight"), xq_, qfull_, (int) N, (int) (NH * 2 * HD), T, cs);
        if (!st_.kv_rot && qsa_dec_fused_supported(s)) {
            QsaDecQPost qp;   // the query split, norm and RoPE in one launch
            qp.q_full = qfull_; qp.qcur = qcur_; qp.pos_q = pos; qp.q_norm = f32("self_attn.q_norm.weight"); qp.eps = EPS;
            qsa_dec_q_post(qp, T, s, rope_scaling(), cs);
        } else
        for (int t = 0; t < T; ++t) {
            float* qc = qcur_ + t * NH * HD;
            if (cudaMemcpy2DAsync(qc, (size_t) HD * 4, qfull_ + t * NH * 2 * HD, (size_t) HD * 2 * 4, (size_t) HD * 4,
                                  (size_t) NH, cudaMemcpyDeviceToDevice, cs) != cudaSuccess) {
                err = "mtp: q split failed";
                return false;
            }
            norm_rope(qc, f32("self_attn.q_norm.weight"), (int) NH, (int) HD, pos + t * NH);
            if (st_.kv_rot) fwht256_inplace_cuda(qc, NH, cs);
        }
        const QsaAttnPools pools = qsa_attn_pools(st_);
        if (window_ > 0) window_ids(const_cast<int32_t*>(step), T, (int) window_, ident_, cap_, cs);
        if (kv_par) join(cs);
        pscope(true);   // the attention follows the join
        bool aq_done = false;
        if (!st_.kv_rot && qsa_dec_attn_supported(pools, s)) {   // the gate in the merge writes the q8_1 rows
            aq_done = true;
            qsa_dec_attn(qcur_, pools, ident_, step, cap_, s, attn_scratch_, attn_, T, cs, qfull_, attn32_, (void*) xq_);
        } else {
            qsa_decode_attn_batch(qcur_, pools, ident_, step, cap_, s, attn_scratch_, attn_, T, cs);
            if (st_.kv_rot) fwht256_inplace_cuda(attn_, (int64_t) T * NH, cs);
            for (int t = 0; t < T; ++t)
                native_qsa_gate_apply(attn_ + t * NH * HD, qfull_ + t * NH * 2 * HD, attn32_ + t * NH * HD, (int) NH,
                                      (int) HD, cs);
        }
        if (!aq_done) native_quantize_q8_1(attn32_, xq_, (int) (NH * HD), T, cs);
        native_mmvq(GGML_Q8_0, q8("self_attn.o_proj.weight"), xq_, bo_, (int) (NH * HD), (int) N, T, cs);
        // ---- the MLP hyper-connection (the attention write folded in)
        bool routed = false;   // the read ran the router too
        {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                fa[t].R = R_ + (size_t) t * HC * N; fa[t].R_out = R_ + (size_t) t * HC * N; fa[t].apply = true;
                fa[t].bo_prev = bo_ + t * N; fa[t].inj_prev = inj_ + t * HC;
                fa[t].w_norm = f32("mlp_hyper_connection.hc_norm.weight");
                fa[t].w_down = bf16("mlp_hyper_connection.input_mix_weight_down.weight");
                fa[t].w_up = bf16("mlp_hyper_connection.input_mix_weight_up.weight");
                fa[t].w_inject = bf16("mlp_hyper_connection.block_inject_weight.weight");
                fa[t].eps = EPS; fa[t].lo = lo_ + t * g.hc_lr; fa[t].rs = rs_ + t * HC;
                fa[t].inject_out = inj2_ + t * HC; fa[t].mixed = mixed_ + t * N;
            }
            if (par_ && dec_draft_hcf()) {   // the read with its write folded in, and the router when it is native
                HcFusedArgs h = hcf_args(fa, T, xn_);
                if (g.n_expert == 512 && K == 10) {
                    h.w_router = bf16("mlp.gate.weight"); h.logits = logits_; h.ids = ids_; h.weights = w_;
                    routed = true;
                }
                hc_fused_read(h, cs);
            } else if (T == 1) {   // one row: the write, then the per-row read (its own rounding)
                gr_write(R_, bo_, inj_, gs, R_, cs);
                gr_read(R_, fa[0].w_norm, fa[0].w_down, fa[0].w_up, fa[0].w_inject, EPS, gs, ss.block.gr, mixed_, inj2_, cs);
            } else {
                fused_gr_read_multi(fa, T, xn_, cs);
            }
        }
        pscope_off();
        // ---- MoE: router, the 512 resident experts, the shared expert, the combine, the write
        // par_: the shared expert on the second stream, beside the router and the routed experts
        const bool sh_par = par_;
        if (sh_par) fork(cs);
        for (int t = 0; t < T && !routed; ++t) {
            bf16_gemv_fp32_mmvf(mixed_ + t * N, bf16("mlp.gate.weight"), logits_ + t * g.n_expert, (int) N, (int) g.n_expert, cs);
            native_router_top10(logits_ + t * g.n_expert, ids_ + t * K, w_ + t * K, cs);
        }
        moe_group_resident(ids_, (int) (T * K), (int) K, experts_, (int64_t) basalt::kernels::cpu::BLOB, grp_ptr_,
                           grp_start_, grp_counts_, hit_dst_, hit_slot_, cs);
        quantize_q8_0_scaled(mixed_, hit_xq_, hit_xs_, (int64_t) T * N, cs);
        moe_grouped_s2(grp_ptr_, grp_start_, grp_counts_, hit_dst_, hit_slot_, (int64_t) T * K, (int64_t) T * K, hit_xq_,
                       hit_xs_, hit_scratch_, parts_, cs);
        NativeSharedWeights nsw;
        nsw.gate_type = GGML_Q8_0; nsw.gate_data = q8("mlp.shared_expert.gate_proj.weight");
        nsw.up_type = GGML_Q8_0; nsw.up_data = q8("mlp.shared_expert.up_proj.weight");
        nsw.down_type = GGML_Q8_0; nsw.down_data = q8("mlp.shared_expert.down_proj.weight");
        nsw.q8_1 = xq_;
        const SForm none{};
        const bool fold_write = par_ && dec_draft_hcf();   // the mixer read applies the write
        const cudaStream_t csh = sh_par ? cs2_ : cs;
        for (int t = 0; t < T; ++t) {
            shared_expert(nullptr, nullptr, none, nullptr, nullptr, nullptr, none, nullptr, nullptr, nullptr, none,
                          nullptr, nullptr, nullptr, bf16("mlp.shared_expert_gate.weight"), sh_scratch_, shared_ + t * N,
                          N, g.n_ff, 32, csh, mixed_ + t * N, &nsw);
            if (sh_par) continue;
            native_moe_combine(parts_ + (size_t) t * K * N, w_ + t * K, shared_ + t * N, y_ + t * N, N, K, cs);
            if (!fold_write)
                gr_write(R_ + (size_t) t * HC * N, y_ + t * N, inj2_ + t * HC, gs, R_ + (size_t) t * HC * N, cs);
        }
        if (sh_par) {
            join(cs);
            for (int t = 0; t < T; ++t) {
                native_moe_combine(parts_ + (size_t) t * K * N, w_ + t * K, shared_ + t * N, y_ + t * N, N, K, cs);
                if (!fold_write)
                    gr_write(R_ + (size_t) t * HC * N, y_ + t * N, inj2_ + t * HC, gs, R_ + (size_t) t * HC * N, cs);
            }
        }
        // pre_next_: the next chain step's hidden input from this row's final residual (what the step end hands on),
        // on the second stream beside the mixer read and the head; joined by the caller
        auto fork_next_hidden = [&] {
            if (!(pre_next_ && par_ && T == 1)) return;
            fork(cs);
            record_hidden(R_, 1, xq2_, cs2_);
            fork_open_ = true;
        };
        if (!fold_write) fork_next_hidden();
        // ---- the final mixer and the main model's head
        bool mixer_q8 = false;
        if (par_ && dec_draft_hcf()) {
            pscope(false);   // after the combine
            FusedGrArgs fa[1];
            fa[0].R = R_; fa[0].R_out = R_; fa[0].apply = true; fa[0].bo_prev = y_; fa[0].inj_prev = inj2_;
            fa[0].w_norm = f32("hyper_connection_mixer.hc_norm.weight");
            fa[0].w_down = bf16("hyper_connection_mixer.input_mix_weight_down.weight");
            fa[0].w_up = bf16("hyper_connection_mixer.input_mix_weight_up.weight");
            fa[0].w_inject = nullptr; fa[0].eps = EPS; fa[0].lo = lo_; fa[0].rs = rs_; fa[0].mixed = sample_;
            if (T != 1) { err = "mtp: the one-kernel mixer read runs one row"; return false; }
            HcFusedArgs h = hcf_args(fa, 1, xn_);
            if (hc_fused_q8_rows_ok()) { h.xq = xq_; mixer_q8 = true; }
            h.next_early = pdl_ok;   // the head's blocks start with the read
            hc_fused_read(h, cs);
            pscope_off();
        } else {
            for (int t = 0; t < T; ++t)
                gr_read(R_ + (size_t) t * HC * N, f32("hyper_connection_mixer.hc_norm.weight"),
                        bf16("hyper_connection_mixer.input_mix_weight_down.weight"),
                        bf16("hyper_connection_mixer.input_mix_weight_up.weight"), nullptr, EPS, gs, ss.block.gr,
                        sample_ + t * N, dummy_inj_, cs);
        }
        if (fold_write) fork_next_hidden();
        if (mixer_q8) pscope(false);   // the head follows the read
        if (!mixer_q8) native_quantize_q8_1(sample_, xq_, (int) N, T, cs);
        const bool sub = dvocab_ != nullptr;
        // the argmax drafts over the base rows only on tier 0
        const int64_t nv = sub ? (tier_ == 0 ? n_tier1_ : n_dvocab_) : n_vocab_;
        // The head, its argmax and the argmax's probability in one launch (the probability's sum in
        // another order: last bits)
        if (T == 1 && sub && proxy_ != nullptr && nv == n_tier1_ && diag_ == nullptr &&
            (head_->type() == 13
                 ? basalt::kernels::q5k_proxy_top1(proxy_, dhead_, xq_, (int) nv, n_tier1_, dvocab_, out_ids_, probs_,
                                                   proxy_scratch_, cs)
                 : basalt::kernels::q6k_proxy_top1(proxy_, rows_mapped_ ? head_->weights() : (const void*) dhead_, xq_,
                                                   (int) nv, n_tier1_, dvocab_, out_ids_, probs_, proxy_scratch_, cs,
                                                   rows_mapped_)))
            return true;
        if (T == 1 && head_->type() == 13 && top_scratch_ != nullptr && diag_ == nullptr &&
            q5k_head_top1(sub ? dhead_ : head_->weights(), xq_, head_logits_, (int) N, (int) nv, sub ? dvocab_ : nullptr,
                          out_ids_, probs_, top_scratch_, cs))
            return true;
        if (rows_mapped_)   // the subset's rows of the main head, one row a step: the bits of the copy's rows
            native_q6_k_mmvq_rows(head_->weights(), dvocab_, xq_, head_logits_, (int) N, (int) nv, cs);
        else if (head_->type() == 13 && q5k_head_supported((int) N, T))   // the same bits
            q5k_head_mmvq(sub ? dhead_ : head_->weights(), xq_, head_logits_, (int) N, (int) nv, T, cs);
        else if (head_->type() == GGML_Q8_0 && T == 1)   // read once a step: streamed past the layer's other weights
            native_q8_0_mmvq_stream(sub ? dhead_ : head_->weights(), xq_, head_logits_, (int) N, (int) nv, cs);
        else
            native_mmvq(head_->type(), sub ? dhead_ : head_->weights(), xq_, head_logits_, (int) N, (int) nv, T, cs);
        if (sub && n_xcap_ > 0)   // rows without a token never win
            for (int t = 0; t < T; ++t) q5k_head_mask_unmapped(head_logits_ + (size_t) t * nv, dvocab_, (int) nv, cs);
        if (diag_ != nullptr && T == 1)   // the step's logits to the host, for the draft-tree study
            cudaMemcpyAsync(diag_ + (size_t) chain_j_ * (size_t) diag_stride_, head_logits_, (size_t) nv * sizeof(float),
                            cudaMemcpyDeviceToHost, cs);
        if (sample_greedy_top_prob(head_logits_, T, (int) nv, sub ? dvocab_ : nullptr, out_ids_, probs_, cs))
            return true;
        SamplerParams sp;
        sp.greedy = true;
        sp.temperature = 0.0f;
        sample_tokens(head_logits_, T, (int) nv, nullptr, 0, sp, out_ids_, cs);
        row_top_prob(head_logits_, T, (int) nv, out_ids_, probs_, cs);
        if (sub) map_ids(out_ids_, dvocab_, T, cs);
    } catch (const std::exception& e) {
        err = std::string("mtp: ") + e.what();
        return false;
    }
    return true;
}

namespace {
bool finish_capture(cudaStream_t cs, bool ok, cudaGraphExec_t& exec, const char* what, std::string& err) {
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        return false;
    }
    if (ce != cudaSuccess || cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("mtp: ") + what + " capture: " + cudaGetErrorString(ce);
        return false;
    }
    cudaGraphDestroy(graph);
    // an explicit upload: the first launch's implicit one blocked behind a device-side spin (src/core/verify/)
    cudaGraphUpload(exec, cs);
    cudaStreamSynchronize(cs);
    return true;
}
}  // namespace

bool MtpDrafter::capture_prefill(int T, std::string& err) {
    if (prefill_exec_[T]) return true;
    using namespace basalt::kernels;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "mtp: begin capture"; return false; }
    copy_i32_from_mapped(tok_, m_tok_, T, cs_);
    copy_i32_from_mapped(step_, m_step_, (int64_t) T * 4, cs_);
    copy_i32_from_mapped(pos_, m_pos_, (int64_t) T * g_->n_head, cs_);
    const bool ok = record_forward(T, -1, cs_, err);   // K/V only, rows [0, T)
    return finish_capture(cs_, ok, prefill_exec_[T], "prefill", err);
}

bool MtpDrafter::capture_prefill_dev(int T, std::string& err) {
    if (prefill_dev_exec_[T]) return true;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "mtp: begin capture"; return false; }
    const bool ok = record_forward(T, -1, cs_, err);   // K/V only, rows [0, T); tok_/step_/pos_ filled before launch
    return finish_capture(cs_, ok, prefill_dev_exec_[T], "prefill (device inputs)", err);
}

// The round: the catch-up's K/V for the window's T cells, then the full layer for row a only (its cell's K/V is written
// again, identically, unless the row is taken from the catch-up), staged by the host in step row
// 2*max_t - 1; draft 0 and its probability to the mapped outputs.  The draft chain is one graph per step
// (`capture_step`) so the host can stop it when a draft is unlikely, or conditional nodes (`capture_cond`).
bool MtpDrafter::record_round(int T, bool all_rows, std::string& err) {
    using namespace basalt::kernels;
    const int64_t HCN = g_->hc * g_->n_embd, NH = g_->n_head;
    const int64_t rows = all_rows ? 2 * (int64_t) max_t_ : 2 * (int64_t) T;
    const int ra = 2 * max_t_ - 1;
    {   // the catch-up reads rows below 2T only: the draft row's records are staged with the rest
        MtpStageSeg sg[6] = {{tok_, m_tok_, T}, {step_, m_step_, (int) (rows * 4)}, {pos_, m_pos_, (int) (rows * NH)},
                             {row_, m_row_, all_rows ? 4 : 2}, {}, {}};
        int ns = 4;
        if (!all_rows) {
            sg[ns++] = {step_ + ra * 4, m_step_ + ra * 4, 4};
            sg[ns++] = {pos_ + ra * NH, m_pos_ + ra * NH, (int) NH};
        }
        mtp_stage(sg, ns, Rin_, window_R_, (int64_t) T * HCN, cs_);
    }
    if (gstamp_ != nullptr) gpu_stamp(gstamp_, 0, cs_);
    par_ = true;
    bool ok = record_forward(T, -1, cs_, err);
    if (ok) {
        chain_j_ = 0;
        pre_next_ = par_;
        ok = record_forward(1, ra, cs_, err, true);
    }
    end_branches();
    if (ok) ok = step_end(0);
    if (ok && gstamp_ != nullptr) gpu_stamp(gstamp_, 1, cs_);
    return ok;
}

// Chain step j (1..max_t-2): one row at the cell staged in step row `max_t + j - 1`, from the previous step's
// residual and token (left in tok_[0] by the previous step); draft j and its probability to the mapped outputs.
bool MtpDrafter::record_step(int j, bool copy_records, std::string& err) {
    using namespace basalt::kernels;
    const int64_t NH = g_->n_head;
    const int row = max_t_ + j - 1;
    if (copy_records) {
        copy_i32_from_mapped(step_ + row * 4, m_step_ + row * 4, 4, cs_);
        copy_i32_from_mapped(pos_ + row * NH, m_pos_ + row * NH, NH, cs_);
    }
    chain_j_ = j;
    par_ = true;
    have_pre_ = pre_next_ = par_;   // every full row hands the next its hidden input
    bool ok = record_forward(1, row, cs_, err);
    end_branches();
    if (ok) ok = step_end(j);
    if (ok && gstamp_ != nullptr) gpu_stamp(gstamp_, 1, cs_);
    return ok;
}

// The attention projections' weights (fc_embedding, k, v, q, o) into L2 by the prefetch stream, from this point of
// `from` on, while the chain's latency-bound kernels run; joined back by end_branches.
void MtpDrafter::pf_fork(cudaStream_t from) {
    if (!par_) return;
    const ModelGeometry& g = *g_;
    const int64_t N = g.n_embd, NH = g.n_head, HD = g.head_dim, NKV = g.n_head_kv;
    auto q8_bytes = [&](int64_t rows, int64_t cols) { return (size_t) (rows * (cols / 32) * 34); };
    auto pf = [&](const void* w, size_t bytes) { basalt::kernels::dense_tc_l2_prefetch(w, bytes, cs3_); };
    cudaEventRecord(ev_pf_, from);
    cudaStreamWaitEvent(cs3_, ev_pf_, 0);
    pf(q8("fc_embedding.weight"), q8_bytes(N, N));
    pf(q8("self_attn.k_proj.weight"), q8_bytes(NKV * HD, N));
    pf(q8("self_attn.v_proj.weight"), q8_bytes(NKV * HD, N));
    pf(q8("self_attn.q_proj.weight"), q8_bytes(NH * 2 * HD, N));
    pf(q8("self_attn.o_proj.weight"), q8_bytes(N, NH * HD));
    pf_open_ = true;
}

void MtpDrafter::prefetch_end(cudaStream_t cs) {
    if (!pf_open_) return;
    cudaEventRecord(ev_pfj_, cs3_);
    cudaStreamWaitEvent(cs, ev_pfj_, 0);
    pf_open_ = false;
}

// The end of chain step j: its token and probability to the mapped outputs and the next step's token, in one launch
// that also sets the conditional handles this step ends with (the next step's input residual is the hidden input
// already computed beside the head, so the residual is not carried).
bool MtpDrafter::step_end(int j) {
    using namespace basalt::kernels;
    const CondTail* c = ctail_;
    if (!mtp_step_end(out_ids_, row_ + 1, tok_, m_out_, j, probs_, m_prob_, c ? c->h : nullptr, c ? c->n : 0, row_ + 3,
                      m_prob_, cs_))
        return false;
    ctail_done_ = c != nullptr;
    return true;
}

int MtpDrafter::diag_top(int j, int k, int32_t* ids, float* probs) const {
    if (diag_ == nullptr || j < 0 || j >= max_t_) return 0;
    const bool sub = dvocab_ != nullptr;
    const int64_t nv = sub ? (tier_ == 0 ? n_tier1_ : n_dvocab_) : n_vocab_;
    const float* l = diag_ + (size_t) j * (size_t) diag_stride_;
    std::vector<std::pair<float, int64_t>> best;
    float mx = l[0];
    for (int64_t i = 1; i < nv; ++i) mx = std::max(mx, l[i]);
    double sum = 0.0;
    for (int64_t i = 0; i < nv; ++i) {
        sum += std::exp((double) (l[i] - mx));
        if ((int) best.size() < k || l[i] > best.back().first) {
            best.emplace_back(l[i], i);
            std::sort(best.begin(), best.end(), [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
            if ((int) best.size() > k) best.pop_back();
        }
    }
    for (size_t i = 0; i < best.size(); ++i) {
        const int64_t r = best[i].second;
        ids[i] = sub && (size_t) r < diag_map_.size() ? diag_map_[(size_t) r] : (int32_t) r;
        probs[i] = (float) (std::exp((double) (best[i].first - mx)) / sum);
    }
    return (int) best.size();
}

void MtpDrafter::end_branches() {
    if (fork_open_) join(cs_);
    prefetch_end(cs_);
    fork_open_ = par_ = pre_next_ = have_pre_ = false;
}

bool MtpDrafter::capture_round(int T, std::string& err) {
    cudaGraphExec_t& exec = round_exec_[tier_][T];
    if (exec) return true;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "mtp: begin capture"; return false; }
    const bool ok = record_round(T, false, err);
    return finish_capture(cs_, ok, exec, "round", err);
}

bool MtpDrafter::capture_step(int j, std::string& err) {
    cudaGraphExec_t& exec = step_exec_[tier_][j];
    if (exec) return true;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "mtp: begin capture"; return false; }
    const bool ok = record_step(j, true, err);
    return finish_capture(cs_, ok, exec, "step", err);
}

// [round, h1 <- rule(p0)] -> IF h1 [step 1, h2 <- rule(p0, p1)] -> IF h2 [...] ...  Every record
// is copied once, by the round.  The host applies the same rule to the same probabilities, so it waits only for
// the drafts the graph makes; the device side only removes the host round trip between steps.
bool MtpDrafter::prewarm(std::string& err) {
    const OnDevice on_device(device_);
    const int max_steps = std::min(max_t_ - 1, max_drafts_);
    if (cond_off_ || max_steps < 2) return true;
    const int saved = tier_;
    bool ok = true;
    for (int tier = 0; ok && tier < (tier2_.empty() ? 1 : 2); ++tier) {
        tier_ = tier2_.empty() ? saved : tier;
        for (int T = 1; ok && T <= std::min(max_t_, basalt::kernels::kVerifyMaxT) && !cond_off_; ++T)
            ok = capture_cond(T, max_steps);
    }
    tier_ = saved;
    if (!ok && cond_off_) return true;   // the per-step graphs take over, as at a lazy failure
    if (!ok) err = "mtp: draft graph capture";
    return ok;
}

bool MtpDrafter::capture_cond(int T, int steps) {
    cudaGraphExec_t& exec = cond_exec_[tier_][T];
    if (exec && cond_steps_[tier_][T] == steps) return true;
    if (exec) {
        cudaGraphExecDestroy(exec);
        exec = nullptr;
    }
    using namespace basalt::kernels;
    std::string err;
    cudaGraph_t top = nullptr;
    cudaGraphConditionalHandle h[kMtpChainMax] = {};
    unsigned long long hv[kMtpChainMax] = {};
    bool ok = steps >= 2 && steps <= kMtpChainMax - 1 && cudaGraphCreate(&top, 0) == cudaSuccess;
    for (int j = 1; ok && j < steps; ++j) {
        // reset to 0 at every launch: a skipped step leaves the next one's handle at 0
        ok = cudaGraphConditionalHandleCreate(&h[j], top, 0, cudaGraphCondAssignDefault) == cudaSuccess;
        hv[j] = (unsigned long long) h[j];
    }
    cudaGraphNode_t prev = nullptr;
    if (ok) ok = cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
    if (ok) {
        CondTail ct{};
        for (int k = 0; k + 1 < steps; ++k) ct.h[k] = hv[k + 1];
        ct.n = steps - 1;
        ctail_ = &ct;
        ctail_done_ = false;
        bool rec = record_round(T, true, err);
        ctail_ = nullptr;
        if (rec && !ctail_done_)
            rec = mtp_draft_cond(hv + 1, steps - 1, probs_, row_ + 1, row_ + 3, cs_, m_prob_, 0);
        cudaGraph_t g0 = nullptr;
        const bool ended = cudaStreamEndCapture(cs_, &g0) == cudaSuccess;
        ok = rec && ended && g0 != nullptr && cudaGraphAddChildGraphNode(&prev, top, nullptr, 0, g0) == cudaSuccess;
        if (g0) cudaGraphDestroy(g0);
    }
    for (int j = 1; ok && j < steps; ++j) {
        cudaGraphNodeParams np{};
        np.type = cudaGraphNodeTypeConditional;
        np.conditional.handle = h[j];
        np.conditional.type = cudaGraphCondTypeIf;
        np.conditional.size = 1;
        cudaGraphNode_t node = nullptr;
        ok = cudaGraphAddNode(&node, top, &prev, nullptr, 1, &np) == cudaSuccess;
        if (!ok) break;
        cudaGraph_t body = np.conditional.phGraph_out[0];
        ok = cudaStreamBeginCaptureToGraph(cs_, body, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal) ==
             cudaSuccess;
        if (!ok) break;
        CondTail ct{};
        ct.h[0] = hv[j + 1];
        ct.n = 1;
        ctail_ = j + 1 < steps ? &ct : nullptr;
        ctail_done_ = false;
        bool rec = record_step(j, false, err);
        ctail_ = nullptr;
        if (rec && j + 1 < steps && !ctail_done_)
            rec = mtp_draft_cond(hv + j + 1, 1, probs_, row_ + 1, row_ + 3, cs_, m_prob_, j);
        cudaGraph_t captured = nullptr;   // the body itself, owned by the node
        ok = cudaStreamEndCapture(cs_, &captured) == cudaSuccess && rec;
        prev = node;
    }
    if (ok) ok = cudaGraphInstantiate(&exec, top, 0) == cudaSuccess;
    if (top) cudaGraphDestroy(top);
    if (ok) {
        cudaGraphUpload(exec, cs_);
        ok = cudaStreamSynchronize(cs_) == cudaSuccess;
    }
    if (!ok) {
        const cudaError_t e = cudaGetLastError();
        basalt::log::warn("basalt mtp", "the conditional draft graph failed (%s%s%s); one graph per "
                             "step from now on",
                     cudaGetErrorString(e), err.empty() ? "" : ", ", err.c_str());
        if (exec) cudaGraphExecDestroy(exec);
        exec = nullptr;
        cond_off_ = true;
        return false;
    }
    cond_steps_[tier_][T] = steps;
    return true;
}

void MtpDrafter::kv_restore(int64_t upto) {
    const OnDevice on_device(device_);
    if (st_.kv_mode != 2 || upto <= 0) return;
    // the ring's blocks below `upto`, from the host copy: a checkpoint resume may have left later cells in them
    const basalt::kernels::QsaShapes s = shapes_of(*g_);
    const int64_t b1 = (upto + s.page_size - 1) / s.page_size, b0 = std::max<int64_t>(0, b1 - st_.n_slots);
    basalt::kernels::kv_ring_restore(qsa_attn_pools(st_), st_.host, qsa_kv_format(st_), b0, b1, st_.n_slots, s, cs_);
    cudaStreamSynchronize(cs_);
}

bool MtpDrafter::prefill(const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0, std::string& err) {
    if (!use_main(err)) return false;
    return prefill_cur(R_rows, next_tokens, n, cell0, err);
}

bool MtpDrafter::prefill_cur(const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0, std::string& err) {
    const OnDevice on_device(device_);
    note_cells(cell0, cell0 + n);
    const Clock::time_point t0 = Clock::now();
    const int64_t HCN = g_->hc * g_->n_embd;
    // cells the window can never reach again need no K/V
    const int64_t first_needed = (window_ > 0 && prompt_len_ > 0) ? prompt_len_ - window_ - 64 : 0;
    // Every group's token / step / position records are uploaded at once; each group is then device copies and a
    // graph on the one stream, with a single sync at the end.
    const int64_t NHp = g_->n_head, per_row = 1 + 4 + NHp;
    if (n > 0) {
        if (pf_cap_ < n * per_row) {
            if (pf_dev_) cudaFree(pf_dev_);
            pf_dev_ = nullptr;
            pf_cap_ = 0;
            if (cudaMalloc((void**) &pf_dev_, (size_t) (n * per_row) * sizeof(int32_t)) != cudaSuccess) {
                err = "mtp prefill: the input records do not fit";
                return false;
            }
            pf_cap_ = n * per_row;
        }
        std::vector<int32_t> rec((size_t) (n * per_row));
        int32_t* tk = rec.data();
        int32_t* stp = tk + n;
        int32_t* ps = stp + 4 * n;
        for (int64_t i = 0; i < n; ++i) {
            const int64_t cell = cell0 + i;
            tk[i] = next_tokens[i];
            stp[i * 4 + 0] = (int32_t) cell;
            stp[i * 4 + 1] = (int32_t) (cell + 1);
            stp[i * 4 + 2] = (int32_t) ((cell + 1) / 4);
            stp[i * 4 + 3] = (int32_t) (cell + 1);
            for (int64_t h = 0; h < NHp; ++h) ps[i * NHp + h] = (int32_t) cell;
        }
        if (cudaMemcpyAsync(pf_dev_, rec.data(), rec.size() * sizeof(int32_t), cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
            err = "mtp prefill: the input records upload failed";
            return false;
        }
        const int32_t* d_tk = pf_dev_;
        const int32_t* d_stp = d_tk + n;
        const int32_t* d_ps = d_stp + 4 * n;
        int64_t blk0 = -1;   // the first row of the block in `relay_` (rows on another GPU come over in blocks)
        const int64_t batch = std::min<int64_t>(max_t_, basalt::kernels::kVerifyMaxT);
        for (int64_t c = 0; c < n; c += batch) {
            const int T = (int) std::min<int64_t>(batch, n - c);
            if (cell0 + c + T <= first_needed) continue;
            const float* rows = R_rows + (size_t) c * HCN;
            if (relay_ != nullptr && R_rows != own_R_ && R_rows != relay_) {
                if (blk0 < 0 || c + T > blk0 + kRelayRows) {
                    blk0 = c;
                    const int64_t nb = std::min<int64_t>(kRelayRows, n - c);
                    if (cudaMemcpyAsync(relay_, R_rows + (size_t) c * HCN, (size_t) nb * HCN * sizeof(float),
                                        cudaMemcpyDeviceToDevice, cs_) != cudaSuccess) {
                        err = "mtp prefill: relaying the rows failed";
                        return false;
                    }
                }
                rows = relay_ + (size_t) (c - blk0) * HCN;
            }
            if (!capture_prefill_dev(T, err)) return false;
            if (cudaMemcpyAsync(tok_, d_tk + c, (size_t) T * 4, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess ||
                cudaMemcpyAsync(step_, d_stp + c * 4, (size_t) T * 16, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess ||
                cudaMemcpyAsync(pos_, d_ps + c * NHp, (size_t) (T * NHp) * 4, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess ||
                cudaMemcpyAsync(Rin_, rows, (size_t) T * HCN * sizeof(float), cudaMemcpyDefault,   // (or pinned host rows)
                                cs_) != cudaSuccess ||
                cudaGraphLaunch(prefill_dev_exec_[T], cs_) != cudaSuccess) {
                err = std::string("mtp prefill: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
        }
        if (cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = std::string("mtp prefill: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        ms_prefill += ms_since(t0);
        return true;
    }
    ms_prefill += ms_since(t0);
    return true;
}

void MtpDrafter::ctx_note(int64_t t) {
    if (n_xcap_ == 0 || t < 0 || t >= n_vocab_ || in_sub_[(size_t) t]) return;
    if ((int64_t) xlist_.size() + (int64_t) xpend_.size() >= n_xcap_) return;
    in_sub_[(size_t) t] = 1;
    xpend_.push_back((int32_t) t);
}

void MtpDrafter::begin_request(const int64_t* prompt, int64_t n) {
    if (n_xcap_ == 0) return;
    const OnDevice on_device(device_);
    for (int32_t t : xlist_) in_sub_[(size_t) t] = 0;
    for (int32_t t : xpend_) in_sub_[(size_t) t] = 0;
    if (n_xused_ > 0) cudaMemsetAsync(dvocab_ + xbase_, 0xff, (size_t) n_xused_ * 4, cs_);   // ids -1
    if (!diag_map_.empty()) std::fill(diag_map_.begin() + xbase_, diag_map_.begin() + xbase_ + n_xcap_, -1);
    xlist_.clear();
    xpend_.clear();
    n_xused_ = 0;
    for (int64_t i = 0; i < n; ++i) ctx_note(prompt[i]);
}

// The queued tokens' head rows into the next free context rows, ahead of the draft graphs on the same stream.
void MtpDrafter::ctx_flush(cudaStream_t cs) {
    if (xpend_.empty()) return;
    const int64_t n = (int64_t) xpend_.size();
    const int64_t row_bytes = (int64_t) head_->row_bytes();
    const int64_t at = xbase_ + n_xused_;
    cudaMemcpyAsync(dxids_, xpend_.data(), (size_t) n * 4, cudaMemcpyHostToDevice, cs);
    basalt::kernels::gather_rows((const uint8_t*) head_->weights(), row_bytes, dxids_, n, dhead_ + at * row_bytes, cs);
    if (proxy_ != nullptr) basalt::kernels::q5k_proxy_pack(dhead_, at, n, proxy_, n_tier1_, cs);
    cudaMemcpyAsync(dvocab_ + at, dxids_, (size_t) n * 4, cudaMemcpyDeviceToDevice, cs);
    if (!diag_map_.empty()) std::copy(xpend_.begin(), xpend_.end(), diag_map_.begin() + at);
    xlist_.insert(xlist_.end(), xpend_.begin(), xpend_.end());
    n_xused_ += n;
    xpend_.clear();
}

bool MtpDrafter::draft(int T, const int32_t* tokens, int64_t p, int a, int32_t* drafts, std::string& err,
                       float* probs, int* n_drafts) {
    if (relay_src_ != nullptr && T >= 1 && T <= max_t_) {
        const OnDevice on_device(device_);
        if (cudaMemcpyAsync(own_R_, relay_src_, (size_t) T * (size_t) (g_->hc * g_->n_embd) * sizeof(float),
                            cudaMemcpyDeviceToDevice, cs_) != cudaSuccess) {
            err = "mtp: relaying the window rows failed";
            return false;
        }
    }
    return draft_rows(T, tokens, p, a, drafts, err, probs, n_drafts);
}

bool MtpDrafter::draft_rows(int T, const int32_t* tokens, int64_t p, int a, int32_t* drafts, std::string& err,
                            float* probs, int* n_drafts) {
    const OnDevice on_device(device_);
    if (T < 1 || T > std::min(max_t_, basalt::kernels::kVerifyMaxT) || a < 0 || a >= T) {
        err = "mtp: draft arguments out of range";
        return false;
    }
    note_cells(p, p + T + max_t_ + 1);
    if (n_xcap_ > 0) {   // the window's committed picks (rows 0..a) may hold names the subset lacks
        for (int t = 0; t <= a; ++t) ctx_note(tokens[t]);
        ctx_flush(cs_);
    }
    const int max_steps = std::min(max_t_ - 1, max_drafts_);
    if (!tier2_.empty()) {   // every row for a while after a token only the added rows hold
        for (int t = 0; t < T; ++t)
            if (tokens[t] >= 0 && tokens[t] < (int32_t) tier2_.size() && tier2_[(size_t) tokens[t]]) tier2_recent_ = 32;
        tier_ = tier2_recent_ > 0 ? 1 : 0;
        if (tier2_recent_ > 0) --tier2_recent_;
    }
    // the chain in one graph (falls back to the graphs below for good if it cannot be built)
    const bool cond = !cond_off_ && max_steps > 1 && capture_cond(T, max_steps);
    if (!cond) {
        if (!capture_round(T, err)) return false;
        for (int j = 1; j < max_steps; ++j)
            if (!capture_step(j, err)) return false;
    }
    const Clock::time_point t0 = Clock::now();
    const int64_t NH = g_->n_head;
    auto put = [&](int row, int64_t cell) {
        h_step_[row * 4 + 0] = (int32_t) cell;
        h_step_[row * 4 + 1] = (int32_t) (cell + 1);
        h_step_[row * 4 + 2] = (int32_t) ((cell + 1) / 4);
        h_step_[row * 4 + 3] = (int32_t) (cell + 1);
        for (int64_t h = 0; h < NH; ++h) h_pos_[row * NH + h] = (int32_t) cell;
    };
    for (int t = 0; t < T; ++t) {
        h_tok_[t] = tokens[t];
        put(t, p + t);
    }
    put(2 * max_t_ - 1, p + a);   // p + a: draft 0's cell
    for (int j = 1; j < max_steps; ++j)
        put(max_t_ + j - 1, p + a + j);
    for (int j = 0; j < max_steps; ++j) ((volatile int32_t*) h_out_)[j] = -1;
    h_row_[0] = a;
    h_row_[1] = 0;
    // slot 2 is reserved: the row layout is fixed by the graphs that copy it
    const float cum_th = cum_th_ >= 0.0f ? cum_th_ : draft_cum_base();
    std::memcpy(&h_row_[3], &cum_th, sizeof(float));   // the window's threshold, read by the conditional graph
    std::atomic_thread_fence(std::memory_order_seq_cst);

    SessionState* const pss = ple_ss_ ? ple_ss_ : ss_;
    const bool do_ple = pss != nullptr && pss->ple.ready() && pss->ple.table != nullptr;
    int32_t ple_prev[2] = {do_ple ? pss->ple_prev[0] : 0, do_ple ? pss->ple_prev[1] : 0};
    auto prefetch_ple = [&](int32_t tok) {
        if (!do_ple || tok < 0) return;
        uint32_t rows16[basalt::kernels::PLE_N_HEADS];
        basalt::kernels::ngram_rows(&tok, ple_prev, 1, pss->ple.consts, rows16);
        ple_prev[0] = ple_prev[1];
        ple_prev[1] = tok;
        pss->ple.table->prefetch_rows(rows16);
    };

    int n = 0;
    if (max_steps > 0) {
        decode_gaps().mark(6);
        const cudaGraphExec_t first = cond ? cond_exec_[tier_][T] : round_exec_[tier_][T];
        decode_events().record(DecodeEvents::kDraftPre, cs_);
        if (cudaGraphLaunch(first, cs_) != cudaSuccess) {
            err = std::string("mtp draft: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        decode_events().record(DecodeEvents::kDraftPost, cs_);
        (void) cudaStreamQuery(cs_);
        decode_gaps().mark(7);
        run_launch_hook();
        prefetch_ple(tokens[a]);
        auto wait_step = [&](int j) {
            uint32_t spins = 0;
            while (((volatile int32_t*) h_out_)[j] < 0) {
                basalt::platform::cpu_relax();
                if ((++spins & 1023u) == 0 && cudaStreamQuery(cs_) != cudaErrorNotReady) break;
            }
        };
        wait_step(0);
        // every draft arrived: the graph's kernels are done, only its skipped steps' conditional nodes remain
        bool arrived = cond && ((volatile int32_t*) h_out_)[0] >= 0;
        drafts[0] = ((volatile int32_t*) h_out_)[0];
        float pr[kMtpChainMax];
        float pj = ((volatile float*) h_prob_)[0];
        pr[0] = pj;
        if (probs) probs[0] = pj;
        n = 1;
        const float host_th = host_th_ >= 0.0f ? host_th_ : cum_th;
        chain_steps_ = max_steps;
        for (int j = 1; j < max_steps && draft_keep(pr, j - 1, host_th); ++j) {
            if (!cond) {   // the conditional graph runs step j on the device by the same comparison
                if (cudaGraphLaunch(step_exec_[tier_][j], cs_) != cudaSuccess) {
                    err = std::string("mtp draft step: ") + cudaGetErrorString(cudaGetLastError());
                    return false;
                }
                (void) cudaStreamQuery(cs_);
            }
            prefetch_ple(drafts[j - 1]);
            wait_step(j);
            if (cond && ((volatile int32_t*) h_out_)[j] < 0) {   // the graph ended without it: the sync says why
                arrived = false;
                break;
            }
            drafts[j] = ((volatile int32_t*) h_out_)[j];
            pj = ((volatile float*) h_prob_)[j];
            pr[j] = pj;
            if (probs) probs[j] = pj;
            ++n;
        }
        // the last draft is a window row only when it passed the cutoff; a row nobody needs would make the window's
        // PLE collect wait for its SSD read
        if (draft_keep(pr, n - 1, host_th)) prefetch_ple(drafts[n - 1]);
        decode_gaps().mark(8);
        // the skipped steps' nodes finish beside the next window; the profile stamps and the logits copy need the end
        const bool sync = !arrived || gstamp_ != nullptr || diag_ != nullptr;
        if (sync && cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = std::string("mtp draft: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
    }
    run_launch_hook();   // no graph ran: the hook still runs, once
    for (int j = n; j < max_t_ - 1; ++j) { drafts[j] = 0; if (probs) probs[j] = 0.0f; }
    if (n_drafts) *n_drafts = n;
    decode_gaps().mark(9);
    if (gstamp_ != nullptr) {   // --profile: the GPU's idle time before the round and the draft's span
        unsigned long long st[2] = {0, 0};
        cudaMemcpy(st, gstamp_, 16, cudaMemcpyDeviceToHost);
        DecodeGaps& dg = decode_gaps();
        if (dg.verify_end != 0 && st[0] > dg.verify_end && st[1] > st[0]) {
            dg.v2d += (double) (st[0] - dg.verify_end);
            dg.draft_gpu += (double) (st[1] - st[0]);
            ++dg.n_v2d;
        }
        dg.draft_end = st[1];
    }
    decode_gaps().mark(10);
    ms_draft += ms_since(t0);
    ++rounds;
    return true;
}

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
    if (!store_hi_.empty()) store_hi_[(size_t) cur_] = std::max(store_hi_[(size_t) cur_], hi);
}

bool MtpDrafter::use_store(int id, std::string& err) {
    if (id == cur_) return true;
    if (id < 0 || id > n_stores_) { err = "mtp: no such K/V store"; return false; }
    const OnDevice on_device(device_);
    KvArr lay[4];
    const int na = kv_layout(lay);
    const int64_t ps = shapes_of(*g_).page_size;
    if (dirty_hi_ > dirty_lo_) {   // the written pages go back to the sequence that held them
        const int64_t pg0 = dirty_lo_ / ps, pg1 = std::min<int64_t>(store_pages_[(size_t) cur_], (dirty_hi_ + ps - 1) / ps);
        for (int i = 0; i < na && pg1 > pg0; ++i)
            cudaMemcpyAsync((uint8_t*) stores_[(size_t) cur_][(size_t) i] + pg0 * lay[i].page_bytes,
                            (const uint8_t*) lay[i].p + pg0 * lay[i].page_bytes, (size_t) ((pg1 - pg0) * lay[i].page_bytes),
                            cudaMemcpyDeviceToDevice, cs_);
    }
    const int64_t n1 = std::min<int64_t>(store_pages_[(size_t) id], (store_hi_[(size_t) id] + ps - 1) / ps);
    for (int i = 0; i < na && n1 > 0; ++i)
        cudaMemcpyAsync(lay[i].p, stores_[(size_t) id][(size_t) i], (size_t) ((uint64_t) n1 * lay[i].page_bytes),
                        cudaMemcpyDeviceToDevice, cs_);
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
    for (int i = 0; i < na && n1 > 0; ++i)
        cudaMemcpyAsync(store_ptr(to, i, lay), store_ptr(from, i, lay), (size_t) ((uint64_t) n1 * lay[i].page_bytes),
                        cudaMemcpyDeviceToDevice, cs_);
    if (to == cur_) note_cells(0, cells);
    store_hi_[(size_t) to] = std::min<int64_t>({cells, st_.max_cells, store_pages_[(size_t) to] * ps});
    if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = "mtp: copying the K/V failed"; return false; }
    return true;
}

QsaState& MtpDrafter::kv_state_rw() {
    std::string e;
    if (!use_main(e)) basalt::log::error("basalt mtp", "%s", e.c_str());
    note_cells(0, prompt_len_ > 0 ? prompt_len_ : st_.max_cells);
    return st_;
}

const float* MtpDrafter::keep_rows(const float* R_rows, int rows, std::string& err) {
    if (rkeep_ == nullptr || rows > kKeepRows) { err = "mtp: no room to keep the window rows"; return nullptr; }
    const OnDevice on_device(device_);
    if (cudaMemcpyAsync(rkeep_, R_rows, (size_t) rows * (size_t) (g_->hc * g_->n_embd) * sizeof(float),
                        cudaMemcpyDeviceToDevice, cs_) != cudaSuccess) {
        err = "mtp: keeping the window rows failed";
        return nullptr;
    }
    return rkeep_;
}

bool MtpDrafter::draft_seq(int id, const float* R_rows, int T, const int32_t* tokens, int64_t p, int a, int32_t* drafts,
                           std::string& err, float* probs, int* n_drafts) {
    if (!flush(id, err) || !use_store(id, err)) return false;
    const OnDevice on_device(device_);
    const size_t bytes = (size_t) T * (size_t) (g_->hc * g_->n_embd) * sizeof(float);
    if (R_rows != window_R_) {   // through a copy: the rows may overlap the ones the graphs read
        if (rstage_ == nullptr) { err = "mtp: no row staging"; return false; }
        if (cudaMemcpyAsync(rstage_, R_rows, bytes, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess ||
            cudaMemcpyAsync((void*) window_R_, rstage_, bytes, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess) {
            err = "mtp: staging the window rows failed";
            return false;
        }
    }
    return draft_rows(T, tokens, p, a, drafts, err, probs, n_drafts);
}

bool MtpDrafter::flush(int id, std::string& err) {
    if (id < 0 || id >= (int) pend_tok_.size() || pend_tok_[(size_t) id].empty()) return true;
    std::vector<int32_t> tk;
    tk.swap(pend_tok_[(size_t) id]);
    if (!use_store(id, err)) return false;
    return prefill_cur(pend_R_[(size_t) id], tk.data(), (int64_t) tk.size(), pend_cell0_[(size_t) id], err);
}

bool MtpDrafter::defer_rows(int id, const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0, std::string& err) {
    if (id < 0 || id >= (int) pend_tok_.size()) { err = "mtp: no such K/V store"; return false; }
    std::vector<int32_t>& tk = pend_tok_[(size_t) id];
    if (!tk.empty() && (pend_cell0_[(size_t) id] + (int64_t) tk.size() != cell0 || (int64_t) tk.size() + n > kPendRows)) {
        if (!flush(id, err)) return false;
    }
    if (n > kPendRows) return catch_up(id, R_rows, next_tokens, n, cell0, err);
    const OnDevice on_device(device_);
    const size_t HCN = (size_t) (g_->hc * g_->n_embd);
    if (tk.empty()) pend_cell0_[(size_t) id] = cell0;
    if (cudaMemcpyAsync(pend_R_[(size_t) id] + tk.size() * HCN, R_rows, (size_t) n * HCN * sizeof(float),
                        cudaMemcpyDeviceToDevice, cs_) != cudaSuccess) {
        err = "mtp: keeping the window rows failed";
        return false;
    }
    tk.insert(tk.end(), next_tokens, next_tokens + n);
    return true;
}

bool MtpDrafter::catch_up(int id, const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0, std::string& err) {
    if (!flush(id, err) || !use_store(id, err)) return false;
    return prefill_cur(R_rows, next_tokens, n, cell0, err);
}

// The drafts the device made past the ones draft() waited for: its conditional graph runs the chain on while the host
// does other work.  Those that have arrived are taken; the host never waits here, because a window on this card may be
// holding its SMs until the host serves its rings, and the chain's kernels would wait behind it.
cudaEvent_t MtpDrafter::done_event() {
    const OnDevice on_device(device_);
    if (ev_done_ == nullptr && cudaEventCreateWithFlags(&ev_done_, cudaEventDisableTiming) != cudaSuccess) return nullptr;
    cudaEventRecord(ev_done_, cs_);
    return ev_done_;
}

int MtpDrafter::draft_more(int have, int32_t* drafts, float* probs) {
    int n = have;
    for (; n < chain_steps_ && n < kMtpChainMax; ++n) {
        if (((volatile int32_t*) h_out_)[n] < 0) break;
        drafts[n] = ((volatile int32_t*) h_out_)[n];
        if (probs) probs[n] = ((volatile float*) h_prob_)[n];
    }
    return n;
}

bool MtpDrafter::stage_rows(const float* R_rows, int n, const float** staged, std::string& err) {
    *staged = R_rows;
    if (relay_ == nullptr) return true;
    if (n < 1 || n > kRelayRows) { err = "mtp: staging rows out of range"; return false; }
    const OnDevice on_device(device_);
    if (cudaMemcpyAsync(relay_, R_rows, (size_t) n * (size_t) (g_->hc * g_->n_embd) * sizeof(float),
                        cudaMemcpyDeviceToDevice, cs_) != cudaSuccess ||
        cudaStreamSynchronize(cs_) != cudaSuccess) {
        err = "mtp: staging the window rows failed";
        return false;
    }
    *staged = relay_;
    return true;
}

bool MtpDrafter::draft_from(const float* R_rows, int T, const int32_t* tokens, int64_t p, int a, int32_t* drafts,
                            std::string& err, float* probs, int* n_drafts) {
    if (R_rows != window_R_) {
        const OnDevice on_device(device_);
        if (cudaMemcpyAsync((void*) window_R_, R_rows, (size_t) T * (size_t) (g_->hc * g_->n_embd) * sizeof(float),
                            cudaMemcpyDeviceToDevice, cs_) != cudaSuccess) {
            err = "mtp: staging the window rows failed";
            return false;
        }
    }
    return draft_rows(T, tokens, p, a, drafts, err, probs, n_drafts);
}

bool MtpDrafter::draft_first(int T, const float* R_row, int32_t token, int64_t cell, int32_t* drafts, std::string& err,
                             float* probs, int* n_drafts) {
    const OnDevice on_device(device_);
    // row 0 is the real pair; rows 1.. repeat it and only write cells the next round overwrites
    const int64_t HCN = g_->hc * g_->n_embd;
    for (int t = 0; t < T; ++t)
        if (cudaMemcpy((void*) (window_R_ + (size_t) t * HCN), R_row, (size_t) HCN * sizeof(float),
                       cudaMemcpyDeviceToDevice) != cudaSuccess) {
            err = "mtp: staging the first residual failed";
            return false;
        }
    std::vector<int32_t> toks((size_t) T, token);
    return draft_rows(T, toks.data(), cell, 0, drafts, err, probs, n_drafts);
}

}  // namespace basalt::core
