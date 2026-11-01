#include "run.hpp"
#include "basalt/prefill/prefill.hpp"
#include "basalt/core/mtp.hpp"
#include "basalt/core/progress.hpp"
#include "basalt/core/on_device.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/core/native_head.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/log.hpp"
#include <algorithm>
#include <thread>
#include <utility>
#include <vector>

namespace basalt::prefill {
using namespace detail;

bool Prefill::Run::Chunk::start_stream() {
    if (stream_all) {
        seq_start.assign((size_t) g.n_layers + 1, 0);
        std::vector<Stager::Job> js;
        for (int64_t l = LB; l < LE; ++l) {
            seq_start[(size_t) l] = seq.size();
            for (int32_t e = 0; e < m.g->n_expert; ++e) {
                if (m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0) continue;
                if (ppc && ppc->peer && ppc->peer->has(l, e)) continue;   // multi-GPU: computed on (or read from) the peer
                int job = -1;
                const uint8_t* b = nullptr;
                if (m.ring_peer && m.ring_peer->exclusive() && m.ring_peer->has(l, e)) {   // no host copy: from the peer
                    seq.push_back({(int32_t) l, e, nullptr, -1, m.ring_peer->slot_ptr(l, e)});
                    continue;
                }
                if (const uint8_t* rp = remote_blob(l, e)) {   // no pinned host copy: from the other card's cache
                    seq.push_back({(int32_t) l, e, nullptr, -1, rp});
                    continue;
                }
                {
                    b = m.src->blob(l, e);
                    if (!b) { err = "prefill: expert source has no blob"; return false; }
                    if (!m.src->pinned(l, e)) {
                        job = (int) js.size();
                        js.push_back(make_job(*m.src, l, e, b, (size_t) lay0.blob_bytes(l)));
                    }
                }
                seq.push_back({(int32_t) l, e, b, job, nullptr});
            }
        }
        for (int64_t l = LE; l <= g.n_layers; ++l) {
            seq_start[(size_t) l] = seq.size();
        }
        m.stager->start(std::move(js));
    }
    chunk_stager_done.st = stream_all ? m.stager.get() : nullptr;
    peer_async = remote_device >= 0 && m.peer_stage.setup(remote_device, (size_t) MAXBLOB());
    if (threaded_issue) {
        issuer = std::thread([this] {
            const core::OnDevice od(m.device);
            for (size_t idx = 0; idx < seq.size(); ++idx) {
                while (idx >= a_consumed.load(std::memory_order_acquire) + (size_t) m.ring) {
                    if (a_stop.load(std::memory_order_acquire)) return;
                    std::this_thread::yield();
                }
                // with the peer's share, only a few copies queue on the link at a time: the peer's result rows
                // then wait behind a few blobs, not behind the whole run-ahead into the next layers
                if (peer_fp4 && idx >= (size_t) kPeerInFlight)
                    while (cudaEventQuery(m.copied[(idx - kPeerInFlight) % (size_t) m.ring]) == cudaErrorNotReady) {
                        if (a_stop.load(std::memory_order_acquire)) return;
                        std::this_thread::yield();
                    }
                const StreamEntry& en = seq[idx];
                const int sl = (int) (idx % (size_t) m.ring);
                const auto th = Clock::now();
                const size_t bytes = (size_t) lay0.blob_bytes(en.l);
                if (m.stage_live[sl]) cudaStreamWaitEvent(m.copy, m.used[m.used_of[sl]], 0);
                if (en.pdev != nullptr) {
                    if (peer_async) m.peer_stage.copy(m.stage_dev[sl], en.pdev, bytes, m.copy);
                    else cudaMemcpyPeerAsync(m.stage_dev[sl], m.device, en.pdev, remote_device, bytes, m.copy);
                    ++iss_dma;
                } else if (en.job < 0) {
                    cudaMemcpyAsync(m.stage_dev[sl], en.blob, bytes, cudaMemcpyHostToDevice, m.copy);
                    ++iss_dma;
                } else {
                    const uint8_t* hb = m.stager->wait(en.job);
                    cudaMemcpyAsync(m.stage_dev[sl], hb, bytes, cudaMemcpyHostToDevice, m.copy);
                    m.stager->issued_one(en.job, m.copy);
                }
                cudaEventRecord(m.copied[sl], m.copy);
                m.stage_live[sl] = true;
                iss_ms += ms_since(th);
                ++iss_streamed;
                a_issued.store(idx + 1, std::memory_order_release);
            }
        });
    }
    return true;
}

}  // namespace basalt::prefill
