#include "state.hpp"
#include "layout.hpp"
#include "basalt/prefill/prefill.hpp"
#include "basalt/core/mtp.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/core/layout.hpp"
#include "basalt/kernels/ngram.hpp"
#include "basalt/prefill/gemm.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/log.hpp"
#include <algorithm>
#include <string>
#include <thread>
#include <vector>
#include <memory>

namespace basalt::prefill {
using namespace detail;

// src/prefill/prefill.cpp - see include/basalt/prefill/prefill.hpp.
Prefill::Prefill() : impl_(new Impl) {}
Prefill::~Prefill() { release(); }

void Prefill::reset() {
    release();
    impl_.reset(new Impl);
    stats_ = PrefillStats{};
}

void Prefill::release() {
    if (!impl_) return;
    if (impl_->warm.fut.valid()) impl_->warm.fut.wait();
    if (impl_->cs) cudaStreamSynchronize(impl_->cs);
    if (impl_->copy) cudaStreamSynchronize(impl_->copy);
    for (int i = 0; i < RING_MAX; ++i) {
        if (impl_->copied[i]) cudaEventDestroy(impl_->copied[i]);
        if (impl_->used[i]) cudaEventDestroy(impl_->used[i]);
    }
    for (int b = 0; b < 2; ++b) {
        if (impl_->hand[b]) cudaFreeHost(impl_->hand[b]);
        if (impl_->ple_copied[b]) cudaEventDestroy(impl_->ple_copied[b]);
        if (impl_->ple_emb_host[b] && impl_->ple_pageable[b].empty()) cudaFreeHost(impl_->ple_emb_host[b]);
    }
    if (impl_->copy) cudaStreamDestroy(impl_->copy);
    if (impl_->grp_host) cudaFreeHost(impl_->grp_host);
    for (float* p : {impl_->cpu_x, impl_->cpu_rows})
        if (p) cudaFreeHost(p);
    if (impl_->cpu_s) cudaStreamDestroy(impl_->cpu_s);
    for (cudaEvent_t e : {impl_->cpu_ev_in, impl_->cpu_ev_x, impl_->cpu_gev[0], impl_->cpu_gev[1], impl_->ahead_ev})
        if (e) cudaEventDestroy(e);
    if (impl_->guess_host) cudaFreeHost(impl_->guess_host);
    for (void* p : impl_->owned) cudaFree(p);
}

void Prefill::set_session(core::SessionState& ss) { impl_->ss = &ss; }

bool Prefill::init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
                   core::ExpertSource* src, const core::ExpertCache* cache, const int32_t* host_res, int64_t chunk,
                   void* stream, std::string& err, void* borrow, uint64_t borrow_bytes) {
    Impl& m = *impl_;
    m.wt = &wt; m.g = &g; m.ss = &ss; m.src = src; m.cache = cache; m.host_res = host_res;
    m.T = chunk; m.cs = (cudaStream_t) stream; m.stats = &stats_;
    if (g.n_embd != N || g.hc != HC || g.hc_lr != LR || g.n_expert < 1 || ss.k != K) {
        err = "prefill: geometry differs from the artifact's"; return false;
    }
    cudaGetDevice(&m.device);
    if (stage_le_ < 0) stage_le_ = g.n_layers;
    if (stage_lb_ < 0 || stage_lb_ >= stage_le_ || stage_le_ > g.n_layers || (stage_le_ < g.n_layers) != (next_ != nullptr)) {
        err = "prefill: the stage's layer range is wrong";
        return false;
    }
    for (int b = 0; next_ != nullptr && b < 2; ++b)
        if (!m.hand[b] && cudaHostAlloc((void**) &m.hand[b], (size_t) chunk * D * 4, cudaHostAllocPortable) != cudaSuccess) {
            err = "prefill: the layer split's hand-off buffers";
            return false;
        }
    if (next_ != nullptr) m.hand_rows = std::max(m.hand_rows, chunk);
    if (m.tok_dev == nullptr) {
        if (const cudaError_t e = cudaMalloc((void**) &m.tok_dev, (size_t) chunk * sizeof(int32_t)); e != cudaSuccess) {
            err = std::string("prefill: the token id buffer (") + cudaGetErrorString(e) + ")";
            return false;
        }
        m.owned.push_back(m.tok_dev);
        m.tok_host.resize((size_t) chunk);
    }
    if (cudaStreamCreateWithFlags(&m.copy, cudaStreamNonBlocking) != cudaSuccess) { err = "prefill: copy stream"; return false; }
    const size_t T = (size_t) chunk;
    m.T_max = chunk;
    m.borrowed = borrow != nullptr;
    bool ok = true;
    // one-time: events, the stager, the host buffers (for the largest chunk), the identity page table
    for (int i = 0; i < RING_MAX; ++i) {
        if (cudaEventCreateWithFlags(&m.copied[i], cudaEventDisableTiming) != cudaSuccess) ok = false;
        if (cudaEventCreateWithFlags(&m.used[i], cudaEventDisableTiming) != cudaSuccess) ok = false;
    }
    if (!m.stager) {
        m.stager = std::make_unique<Stager>();
        const int hw = (int) std::thread::hardware_concurrency();
        const int threads = std::max(2, std::min(4, hw / 4));
        if (!m.stager->init((size_t) MAXBLOB(), threads)) ok = false;
    }
    m.steps_host.resize(T * basalt::kernels::kStepCount);
    m.ids_host.resize(T * K); m.slot_host.resize(T * K); m.src_host.resize(T * K); m.cnt.resize(m.g->n_expert); m.off.resize(m.g->n_expert + 1);
    {
        const size_t need = 3 * T * K + (size_t) (2 * (m.g->n_expert + m.g->n_expert / MMQ_GROUP + 2));
        if (m.grp_n < need) {
            if (m.grp_host) cudaFreeHost(m.grp_host);
            m.grp_host = m.grp_dev = nullptr;
            m.grp_n = m.grp_tk = 0;
            void *h = nullptr, *d = nullptr;
            if (cudaHostAlloc(&h, need * 4, cudaHostAllocMapped | (core::peer_portable() ? cudaHostAllocPortable : 0)) == cudaSuccess &&
                cudaHostGetDevicePointer(&d, h, 0) == cudaSuccess) {
                m.grp_host = (int32_t*) h;
                m.grp_dev = (int32_t*) d;
                m.grp_n = need;
                m.grp_tk = T * K;
            } else {                              // the copies, as before
                if (h) cudaFreeHost(h);
                cudaGetLastError();
            }
        }
    }
    // the PLE rows' host buffers, for a stage that runs layer 1
    for (int b = 0; stage_lb_ <= 1 && 1 < stage_le_ && b < 2; ++b) {
        if (!m.ple_emb_host[b] &&
            cudaHostAlloc((void**) &m.ple_emb_host[b], (size_t) T * N * 4, cudaHostAllocDefault) != cudaSuccess) {
            cudaGetLastError();
            m.ple_pageable[b].resize(T * N);          // pageable: the upload is staged before it returns
            m.ple_emb_host[b] = m.ple_pageable[b].data();
        }
        if (!m.ple_copied[b] && cudaEventCreateWithFlags(&m.ple_copied[b], cudaEventDisableTiming) != cudaSuccess)
            ok = false;
        m.ple_rows[b].resize(T * basalt::kernels::PLE_N_HEADS);
    }
    if (m.guess_host == nullptr) {   // the routing guess of a chunk below stream_all_min(), outside the loan
        const int64_t rows = std::min<int64_t>(chunk, stream_all_min() - 1) * K;
        void* hd = nullptr;
        bool got = rows > 0 && cudaMalloc((void**) &m.guess_ids, (size_t) rows * 4) == cudaSuccess;
        if (got) m.owned.push_back(m.guess_ids);
        got = got && cudaMalloc((void**) &m.guess_w, (size_t) rows * 4) == cudaSuccess;
        if (got) m.owned.push_back(m.guess_w);
        got = got && cudaHostAlloc((void**) &m.guess_host, (size_t) rows * 4,
                                   cudaHostAllocMapped | (core::peer_portable() ? cudaHostAllocPortable : 0)) == cudaSuccess &&
              cudaHostGetDevicePointer(&hd, m.guess_host, 0) == cudaSuccess;
        if (got) {
            m.guess_host_dev = (int32_t*) hd;
            m.guess_cap = rows;
        } else {
            cudaGetLastError();   // no guess: no copies ahead
        }
        for (int b = 0; b < 2; ++b) m.ahead_slot[b].assign((size_t) m.g->n_expert, -1);
    }
    if (ss.qsa_states[ss.qsa_primary()].kv_mode == 1) {   // KV streaming: the staging pool's identity page table
        const int64_t pages = ss.qsa_states[ss.qsa_primary()].n_pages;
        std::vector<int32_t> ident((size_t) pages);
        for (int64_t i = 0; i < pages; ++i) ident[(size_t) i] = (int32_t) i;
        if (cudaMalloc((void**) &m.ident_table, ident.size() * 4) != cudaSuccess ||
            cudaMemcpy(m.ident_table, ident.data(), ident.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess)
            ok = false;
        else
            m.owned.push_back(m.ident_table);
    }
    if (!ok) { err = "prefill: host buffers or events for a chunk of " + std::to_string(chunk) + " tokens"; return false; }
    Alloc o;
    o.base = (uint8_t*) borrow;
    o.cap = borrow_bytes;
    o.owned = &m.owned;
    {
        uint16_t* gs = o.take<uint16_t>((size_t) GEMM_SCRATCH, ok);
        void* ws = o.take<uint8_t>(GEMM_WS, ok);
        if (!ok) { err = "prefill: GEMM scratch does not fit"; return false; }
        if (!m.gemm.init_external(stream, gs, GEMM_SCRATCH, ws, GEMM_WS, err)) return false;
    }
    if (!carve(T, &o)) {
        err = "prefill: device buffers for a chunk of " + std::to_string(chunk) + " tokens do not fit";
        return false;
    }
    return true;
}

// Every device buffer of a chunk of T tokens, from the Alloc `alloc` (after the GEMM scratch and workspace): `init`
// once, and `relayout` for a request's own chunk.  The order is `bytes_needed`'s.

}  // namespace basalt::prefill
