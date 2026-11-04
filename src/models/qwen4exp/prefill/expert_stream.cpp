#include "experts.hpp"
#include "basalt/prefill/prefill.hpp"
#include "basalt/core/mtp.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/prefill/moe_fp4.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/log.hpp"
#include <algorithm>
#include <utility>
#include <vector>

namespace basalt::prefill {
using namespace detail;

// set_ring_remote: an expert with no pinned host copy that the other card's cache holds (its blob there), or null
const uint8_t* Prefill::Run::remote_blob(int64_t l, int32_t e) {
    if (m.remote_res == nullptr || m.src == nullptr || m.src->pinned(l, e)) return nullptr;
    const int32_t s = m.remote_res[(size_t) l * m.g->n_expert + e];
    return s >= 0 ? m.remote_cache->device_slot(s) : nullptr;
}

const uint8_t* Prefill::Run::Chunk::Experts::Group::other_card(int32_t e) {
    // --peer-exclusive: an expert only the peer holds has no host copy.  It is copied from the peer's
    // memory into a staging slot; so is one set_ring_remote's card holds without a pinned host copy.
    // The blob on the other card, or null.
    if (m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0) return nullptr;
    if (m.ring_peer && m.ring_peer->exclusive() && m.ring_peer->has(l, e))
        return m.ring_peer->slot_ptr(l, e);
    return remote_blob(l, e);
}

// Stage ahead: the copy stream moves blobs host -> device while the compute stream works.
bool Prefill::Run::Chunk::Experts::Group::start_staging() {
    stage_of.assign(order.size(), -1);
    // the unpinned ones are copied to pinned buffers by the stager's threads, in this order
    job_of.assign(order.size(), -1);
    if (!stream_all) {
        std::vector<Stager::Job> js;
        for (size_t j = 0; j < order.size(); ++j) {
            const int32_t e = order[j];
            if (m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0) continue;
            if (other_card(e) != nullptr) continue;
            if (m.src->pinned(l, e)) continue;
            job_of[j] = (int) js.size();
            const uint8_t* b = m.src->blob(l, e);
            if (!b) { err = "prefill: expert source has no blob"; return false; }
            js.push_back(make_job(*m.src, l, e, b, (size_t) lay.blob_bytes(l)));
        }
        m.stager->start(std::move(js));
    }
    stager_done.st = stream_all ? nullptr : m.stager.get();
    return true;
}

bool Prefill::Run::Chunk::Experts::Group::stage_one(size_t j) {
    const int32_t e = order[j];
    const bool resident = m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0;
    if (resident) return true;
    if (ahead_n > 0 && m.ahead_layer[l & 1] == l && m.ahead_slot[l & 1][(size_t) e] >= 0) {
        stage_of[j] = m.ahead_slot[l & 1][(size_t) e];   // copied ahead of the routing
        return true;
    }
    const int sl = stage_next;
    stage_next = (stage_next + 1) % STAGE;
    const auto th = Clock::now();
    const uint8_t* via = other_card(e);
    const bool pinned = via == nullptr && m.src->pinned(l, e);
    const uint8_t* b = pinned ? m.src->blob(l, e) : nullptr;
    if (pinned && !b) { err = "prefill: expert source has no blob"; return false; }
    if (via != nullptr) {
        if (m.stage_live[sl]) cudaStreamWaitEvent(m.copy, m.used[m.used_of[sl]], 0);
        if (peer_async) m.peer_stage.copy(m.stage_dev[sl], via, (size_t) lay.blob_bytes(l), m.copy);
        else cudaMemcpyPeerAsync(m.stage_dev[sl], m.device, via, remote_device,
                                 (size_t) lay.blob_bytes(l), m.copy);
        ++owner.stats_.experts_dma;
    } else if (pinned) {
        // DMA straight from the page-locked arena: the copy stream only waits for the slot
        if (m.stage_live[sl]) cudaStreamWaitEvent(m.copy, m.used[m.used_of[sl]], 0);
        cudaMemcpyAsync(m.stage_dev[sl], b, (size_t) lay.blob_bytes(l), cudaMemcpyHostToDevice, m.copy);
        ++owner.stats_.experts_dma;
    } else {
        // copied to a pinned buffer by the stager (waits only if it is behind), then DMA
        const uint8_t* hb = m.stager->wait(job_of[j]);
        if (m.stage_live[sl]) cudaStreamWaitEvent(m.copy, m.used[m.used_of[sl]], 0);
        cudaMemcpyAsync(m.stage_dev[sl], hb, (size_t) lay.blob_bytes(l), cudaMemcpyHostToDevice, m.copy);
        m.stager->issued_one(job_of[j], m.copy);
    }
    cudaEventRecord(m.copied[sl], m.copy);
    m.stage_live[sl] = true;
    stage_of[j] = sl;
    owner.stats_.ms_experts_host += ms_since(th);
    ++owner.stats_.experts_streamed;
    return true;
}

bool Prefill::Run::Chunk::Experts::Group::walk() {
    if (cpu_maybe) cudaEventRecord(m.cpu_gev[0], m.cs);
    if (!stream_all) {
        size_t staged = 0;
        // staged this far ahead, a slot is copied into again only after the group that read it
        const size_t lookahead = STAGE - MMQ_GROUP - 1;
        for (size_t j = 0; j < order.size(); ++j) {
            while (staged < order.size() && staged <= j + lookahead) {
                if (!stage_one(staged)) return false;
                ++staged;
            }
            const int32_t e = order[j];
            if (stage_of[j] < 0) {
                ++owner.stats_.experts_resident;
                const int32_t rs = m.host_res ? m.host_res[(size_t) l * m.g->n_expert + e] : -1;
                const uint8_t* bd = rs >= 0 && m.cache ? m.cache->device_slot(rs) : m.ring_peer->slot_ptr(l, e);
                if (!compute(j, bd, -1)) return false;
            } else {
                cudaStreamWaitEvent(m.cs, m.copied[stage_of[j]], 0);
                if (!compute(j, m.stage_dev[stage_of[j]], stage_of[j])) return false;
            }
        }
    } else {
        // the streamed walk: this layer's entries [k, kend) in id order; an entry the routing did not
        // pick only gives its slot back
        size_t k = seq_start[(size_t) l];
        const size_t kend = seq_start[(size_t) l + 1];
        auto release_to = [&](int32_t e_stop) {
            while (k < kend && seq[k].e < e_stop) {
                if (gg_nslots > 0) flush();   // the open group's slots get their event first
                const int sl = (int) (k % (size_t) m.ring);
                cudaEventRecord(m.used[sl], m.cs);
                m.used_of[sl] = sl;
                consumed = ++k;
                give_back(consumed);
            }
        };
        for (size_t j = 0; j < order.size(); ++j) {
            const int32_t e = order[j];
            release_to(e);
            if (k < kend && seq[k].e == e) {
                const int sl = (int) (k % (size_t) m.ring);
                wait_issued(k);
                if (!group_gather) {
                    cudaStreamWaitEvent(m.cs, m.copied[sl], 0);
                }
                if (!compute(j, m.stage_dev[sl], sl)) return false;
                consumed = ++k;
                if (!group_gather || gg_nslots == 0) give_back(consumed);   // its group was gathered
            } else {
                const bool r0 = m.host_res && m.cache && m.host_res[(size_t) l * m.g->n_expert + e] >= 0;
                const uint8_t* bp = r0 ? m.cache->device_slot(m.host_res[(size_t) l * m.g->n_expert + e])
                                       : nullptr;
                if (bp == nullptr) { err = "prefill: an expert is neither resident, streamed nor on the peer"; return false; }
                ++owner.stats_.experts_resident;
                if (!compute(j, bp, -1)) return false;
                if (group_gather && gg_nslots == 0) give_back(consumed);
            }
        }
        release_to(m.g->n_expert);
    }
    return true;
}

}  // namespace basalt::prefill
