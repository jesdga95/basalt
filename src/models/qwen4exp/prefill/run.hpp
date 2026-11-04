#pragma once

#include "state.hpp"
#include "layout.hpp"
#include "basalt/prefill/prefill.hpp"
#include "basalt/core/mtp.hpp"
#include "basalt/core/progress.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/core/layout.hpp"
#include "basalt/kernels/ngram.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/qsa.hpp"
#include "basalt/core/layer.hpp"
#include "basalt/core/peer_experts.hpp"
#include <algorithm>
#include <atomic>
#include <string>
#include <thread>
#include <vector>
#include <future>

namespace basalt::prefill {
using namespace detail;

struct Prefill::Run {
    struct Chunk;
    Run(Prefill& owner, const int64_t* tokens, int64_t n, int64_t pos0, std::string& err)
        : owner(owner), tokens(tokens), n(n), pos0(pos0), err(err) {}
    bool execute();
    bool finish();
    const uint8_t* remote_blob(int64_t l, int32_t e);
    int64_t chunk_len(int64_t at) { return std::min(at == 0 && head > 0 ? head : m.T, n - at); }
    bool ple_gather(int64_t c0, int buf, std::string& e);

    Prefill& owner;
    const int64_t* tokens;
    int64_t n, pos0;
    std::string& err;
    Impl& m = *owner.impl_;
    const core::OnDevice on_device{m.device};
    const core::ModelGeometry& g = *m.g;
    core::SessionState& ss = *m.ss;
    const Clock::time_point t_start = Clock::now();
    const int64_t LB = owner.stage_lb_, LE = owner.stage_le_;
    // the card the ring's other-card copies come from (--peer-exclusive's peer, or set_ring_remote's card)
    const int remote_device = m.ring_peer ? m.ring_peer->device() : m.remote_dev;
    basalt::kernels::QsaShapes s = basalt::kernels::qsa_real_shapes();
    const uint64_t gdn_floats = (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
    int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
    // the first stage embeds the tokens; a later one reads the previous stage's rows (handed on, or in its own buffers)
    const bool rows_in_place = owner.rows_in_place_;
    const bool first_stage = owner.hand_in_ == nullptr && !rows_in_place;
    const bool single_chunk = first_stage ? n <= m.T : owner.single_chunk_;
    // the first chunk may be shorter (head_chunk_); the rest are m.T long
    const int64_t head = first_stage && n > m.T ? std::min(owner.head_chunk_, m.T) : 0;
    const cudaStream_t cs = (cudaStream_t) m.cs;
    const bool ple_on = ss.ple.ready() && LB <= 1 && 1 < LE;
    // the PLE block batched over the chunk: a BF16 or GGUF-native key (else token by token)
    const bool ple_batch = ple_on &&
                           (ss.ple.w.key_bf16 != nullptr || ss.ple.w.key_native_data != nullptr) &&
                           m.region_bytes / ((uint64_t) (3 * basalt::kernels::NG_HC_DIM + N + 4) * 4 + (uint64_t) N * 2 + 4096) >= 64;
    const int32_t prev0[2] = {prev[0], prev[1]};
    std::string ple_next_err;
    std::future<bool> ple_next;             // declared after everything it reads: an early return waits for it
    int ple_buf = 0;
};

struct Prefill::Run::Chunk {
    struct Experts;
    Chunk(Run& run, int64_t c0) : run(run), c0(c0), T(run.chunk_len(c0)), p0(run.pos0 + c0) {}
    bool execute();
    bool inputs();
    bool ple_land();
    bool ple_layer(int64_t l);
    bool start_stream();
    bool layer(int64_t l);
    bool gdn(const core::LayerView& v);
    bool qsa(const core::LayerView& v);
    bool moe(int64_t l, const core::LayerView& v);
    bool shared_expert(const core::LayerView& v, const core::WeightRef* wgi, const core::WeightRef* wsg,
                       const core::WeightRef* wsu, const core::WeightRef* wsd);
    int finish();
    const uint8_t* remote_blob(int64_t l, int32_t e) { return run.remote_blob(l, e); }
    void copy_ahead(int64_t L);
    // the consumer's side: entry k's copy is on the copy stream (the thread issued it), then k is given back
    void wait_issued(size_t k) {
        while (a_issued.load(std::memory_order_acquire) <= k) std::this_thread::yield();
    }
    void give_back(size_t upto) { a_consumed.store(upto, std::memory_order_release); }
    Run& run;
    decltype(Run::m)& m = run.m;
    decltype(Run::g)& g = run.g;
    decltype(Run::ss)& ss = run.ss;
    decltype(Run::t_start)& t_start = run.t_start;
    decltype(Run::LB)& LB = run.LB;
    decltype(Run::LE)& LE = run.LE;
    decltype(Run::remote_device)& remote_device = run.remote_device;
    decltype(Run::s)& s = run.s;
    decltype(Run::gdn_floats)& gdn_floats = run.gdn_floats;
    decltype(Run::prev)& prev = run.prev;
    decltype(Run::rows_in_place)& rows_in_place = run.rows_in_place;
    decltype(Run::first_stage)& first_stage = run.first_stage;
    decltype(Run::single_chunk)& single_chunk = run.single_chunk;
    decltype(Run::head)& head = run.head;
    decltype(Run::cs)& cs = run.cs;
    decltype(Run::ple_on)& ple_on = run.ple_on;
    decltype(Run::ple_batch)& ple_batch = run.ple_batch;
    decltype(Run::prev0)& prev0 = run.prev0;
    decltype(Run::ple_next_err)& ple_next_err = run.ple_next_err;
    decltype(Run::ple_next)& ple_next = run.ple_next;
    decltype(Run::ple_buf)& ple_buf = run.ple_buf;
    decltype(Run::tokens)& tokens = run.tokens;
    decltype(Run::n)& n = run.n;
    decltype(Run::pos0)& pos0 = run.pos0;
    decltype(Run::err)& err = run.err;
    decltype(Run::owner)& owner = run.owner;
    int64_t c0;
    const int64_t T, p0;
    bool ple_pending = ple_on;
    int64_t qsa_index = 0, gdn_index = 0;
    // this chunk's stream - every non-resident expert of every layer, layer by layer in id order (entry
    // k lands in ring slot k % ring); a copy is issued once the entry `ring` before it is consumed (its slot's
    // `used` event recorded), so the copy stream never waits on an event that is not queued yet
    const basalt::kernels::cpu::ExpertLayout& lay0 = basalt::kernels::cpu::expert_layout();
    const bool stream_all = m.ring > STAGE && T >= stream_all_min() && m.src != nullptr;
    // the peer takes the rows of a chunk only up to its buffers' size; a streamed chunk only up
    // to T_fp4, and then on the FP4 kernels (PeerPrefill::T_fp4)
    PeerPrefill* const ppc =
        m.pp && m.peer_share && (T <= m.pp->T_max || (stream_all && T <= m.pp->T_fp4)) ? m.pp.get() : nullptr;
    const bool peer_fp4 = ppc && stream_all && ppc->T_fp4 > 0;
    struct StreamEntry { int32_t l, e; const uint8_t* blob; int job; const uint8_t* pdev; };
    std::vector<StreamEntry> seq;
    std::vector<size_t> seq_start;
    size_t consumed = 0;
    struct StagerDone {
        Stager* st;
        ~StagerDone() { if (st) st->finish(); }
    } chunk_stager_done{nullptr};
    // the stream is issued by its own host thread, so the thread launching the layers' kernels never waits
    // behind a host copy of an unpinned blob (that wait would leave the GPU idle).  The same copies in the same order into the same slots, and a slot is refilled only once
    // the compute stream has recorded that it is done with it: the same results.
    std::atomic<size_t> a_issued{0}, a_consumed{0};
    std::atomic<bool> a_stop{false};
    double iss_ms = 0;
    int64_t iss_streamed = 0, iss_dma = 0;
    std::thread issuer;
    struct IssuerJoin {
        std::atomic<bool>* stop;
        std::thread* t;
        ~IssuerJoin() { if (t->joinable()) { stop->store(true); t->join(); } }
    } issuer_join{&a_stop, &issuer};
    const bool threaded_issue = stream_all;
    bool peer_async = false;
    const int ahead_n = !stream_all && m.host_res != nullptr && m.cache != nullptr && m.src != nullptr &&
                                m.guess_cap >= T * K
                            ? (m.ring - STAGE) / 2 : 0;
    int64_t guess_for = -1;   // the layer whose routing this layer's half guessed
    bool normed = false;   // the previous half's write already normed R for this half (grs, xn16)
};

}  // namespace basalt::prefill
