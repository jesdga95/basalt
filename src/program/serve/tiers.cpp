// src/program/serve/tiers.cpp - tiers operations for the persistent engine.
#include "state.hpp"

namespace basalt::program::serve {

// Seeds the adaptive tier's usage counts from the routing of the prompt just read, so the first adaptation rounds
// of a request already move the experts the new topic uses toward the GPUs.  The earlier usage decays to kSeedKeep and
// the prompt adds its routed-entry counts scaled to kSeedTokens tokens of usage.  Resident experts are always seeded;
// of the rest only the kSeedTop with the most prompt entries are (a peer-held one weighs kSeedPeerWeight, since it
// costs less to leave where it is than a CPU one).
void seed_usage_from_prompt(std::vector<float>& usage, const std::vector<uint32_t>& hist, int64_t tokens,
                            const std::vector<int32_t>& host_res, int64_t n_expert, const basalt::core::PeerExperts& peer) {
    constexpr double kSeedTokens = 64.0;
    constexpr float kSeedKeep = 0.3f, kSeedPeerWeight = 0.35f;
    constexpr size_t kSeedTop = 32;
    if (tokens <= 0 || usage.empty() || hist.size() != usage.size() || host_res.size() != usage.size()) return;
    std::vector<uint8_t> allow(hist.size(), 1);
    std::vector<std::pair<float, size_t>> rank;
    for (size_t i = 0; i < hist.size(); ++i) {
        if (hist[i] == 0 || host_res[i] >= 0) continue;
        allow[i] = 0;
        const bool on_peer = peer.valid() && peer.has((int64_t) i / n_expert, (int64_t) i % n_expert);
        rank.emplace_back((float) hist[i] * (on_peer ? kSeedPeerWeight : 1.0f), i);
    }
    const size_t k = std::min(rank.size(), kSeedTop);
    std::partial_sort(rank.begin(), rank.begin() + (ptrdiff_t) k, rank.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
    for (size_t j = 0; j < k; ++j) allow[rank[j].second] = 1;
    const float scale = (float) (kSeedTokens / (double) tokens);
    for (size_t i = 0; i < hist.size(); ++i) usage[i] = usage[i] * kSeedKeep + (allow[i] ? scale * (float) hist[i] : 0.0f);
}

// The same with free places of the RAM copy instead of exchange buffers (see FileExpertSource::set_spares): each
// evicted expert's slot is copied into a free place, which becomes its home once the copies have landed.  A swap with
// no free place waits for a later round.  `released` gets the (layer, in) pairs whose places are free once the incoming
// experts are resident on the GPU.
template <class Swap, class CopyBack, class Wait>
bool resident_spare_swaps(basalt::core::FileExpertSource& src, const std::vector<int32_t>& host_res, int64_t n_expert,
                          std::vector<Swap>& swaps, CopyBack&& copy_back, Wait&& wait,
                          std::vector<std::pair<int32_t, int32_t>>& released) {
    if (!src.complement_ready() || swaps.empty()) return true;
    struct Home { int32_t layer, out; uint64_t off; };
    std::vector<Home> homes;
    std::vector<Swap> kept;
    kept.reserve(swaps.size());
    for (const Swap& s : swaps) {
        if (!src.has_resident(s.layer, s.in) || src.has_resident(s.layer, s.out)) { kept.push_back(s); continue; }
        const int32_t slot = host_res[(size_t) s.layer * (size_t) n_expert + (size_t) s.out];
        if (slot < 0) continue;
        uint64_t off = 0;
        uint8_t* dst = src.spare_take(s.layer, off);
        if (dst == nullptr) continue;
        if (!copy_back(s.layer, slot, dst)) {
            src.spare_return(s.layer, off);
            return false;
        }
        homes.push_back({s.layer, s.out, off});
        released.emplace_back(s.layer, s.in);
        kept.push_back(s);
    }
    if (!homes.empty()) {
        if (!wait()) return false;
        for (const Home& h : homes)
            if (!src.spare_land(h.layer, h.out, h.off)) return false;
    }
    swaps.swap(kept);
    return true;
}

// a layer split's later stages keep a copy of the residency table on their devices, and swap on their own
void Runtime::res_upload() {
    if (d_res != nullptr)
        res_put(d_res);
    for (auto& st : stages) {
        const basalt::core::OnDevice on(st->dev);
        res_put(st->d_res);
    }
}

// the peer tiers take the next most-routed CPU misses: each passes over what the primary (r0) or another tier
// holds or takes in this round; with one tier r0 is its whole view
bool Runtime::adapt_peers(const std::vector<int32_t>& r0, std::string& perr) {
    const int swaps_each = o.peer_adapt_swaps >= 0 ? o.peer_adapt_swaps : o.adapt_swaps;
    if (xpeers.empty()) return peer.adapt(drive.d.usage.data(), r0.data(), swaps_each, perr);
    const size_t nt = 1 + xpeers.size();
    auto tier = [&](size_t i) -> basalt::core::PeerExperts& { return i == 0 ? peer : *xpeers[i - 1]; };
    for (size_t t = 0; t < nt; ++t) {
        if (!tier(t).valid()) continue;
        std::vector<int32_t> seen = r0;
        for (size_t u = 0; u < nt; ++u) {
            if (u == t || !tier(u).valid()) continue;
            const std::vector<int32_t>& ru = tier(u).residency();
            for (size_t i = 0; i < seen.size() && i < ru.size(); ++i)
                if (ru[i] >= 0) seen[i] = 0;
            for (size_t i : tier(u).incoming())
                if (i < seen.size()) seen[i] = 0;
        }
        if (!tier(t).adapt(drive.d.usage.data(), seen.data(), swaps_each, perr)) return false;
    }
    return true;
}

void Runtime::apply_pending(bool wait, bool exchanges) {
    if (peer.valid()) {
        if (!wait) peer.apply_pending(wait);
        else if (peer_landing) { peer.apply_pending(true); peer_landing = false; }
        else if (peer.has_pending()) { peer.issue_pending(true); peer_landing = true; }
    }
    for (size_t q = 0; q < xpeers.size(); ++q) {
        basalt::core::PeerExperts& x = *xpeers[q];
        if (!x.valid()) continue;
        if (!wait) x.apply_pending(wait);
        else if (xlanding[q]) { x.apply_pending(true); xlanding[q] = 0; }
        else if (x.has_pending()) { x.issue_pending(true); xlanding[q] = 1; }
    }
    if (pending.empty()) return;
    if (wait) cudaEventSynchronize(adapt_ev);
    else if (cudaEventQuery(adapt_ev) != cudaSuccess) return;
    for (auto& st : stages)
        if (st->adapt_live) {
            if (wait) cudaEventSynchronize(st->adapt_ev);
            else if (cudaEventQuery(st->adapt_ev) != cudaSuccess) return;
        }
    for (auto& st : stages) st->adapt_live = false;
    if (exchanges) src.commit_exchanges();   // the resident RAM mode: the evicted experts take their places in RAM
    for (const auto& [i, slot] : pending) {
        host_res[(size_t) i] = slot;
        srcp->release((int64_t) i / g.n_expert, (int64_t) i % g.n_expert);   // in VRAM now: RAM not needed
    }
    for (const auto& [l, e] : pending_spare) src.spare_release(l, e);
    pending_spare.clear();
    pending.clear();
    res_upload();
    // a promoted expert is CUDA0's now: its peer slot takes the expert CUDA0 gave up (in RAM since the commit)
    for (const auto& p : promoted) {
        std::string perr;
        if (!peer.reassign(p[0], p[1], p[0], p[2], srcp->blob(p[0], p[2]), perr))
            basalt::log::warn("basalt serve", "%s", perr.c_str());
    }
    promoted.clear();
}

void Runtime::rank_swaps(std::vector<Swap>& swaps, bool across) {
    swaps.clear();
    if (across) {
        // one ranking over every layer: a layer that routes more experts than its slots hold takes the slots of
        // the least-routed experts of any layer, so the slots follow the request's layers; with every layer's
        // residents to choose from, a missing expert is a candidate at a lower count.  An expert the peer tier
        // holds ranks lower by the promotion margin, as in the per-layer rounds, and the resident it replaces
        // takes its peer slot.
        struct Ent { float u; int32_t rank, layer, e; };
        std::vector<Ent> cand, vict;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            const float* u = drive.d.usage.data() + l * g.n_expert;
            const int32_t* r = host_res.data() + l * g.n_expert;
            for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                const int32_t rk = prof_rank[(size_t) (l * g.n_expert + e)];
                if (r[e] >= 0) vict.push_back({u[e], rk, (int32_t) l, e});
                else if (u[e] >= 1.5f && !(remote_opt && remote_opt->owns(l, e)))
                    cand.push_back({peer.valid() && peer.has(l, e) ? u[e] - promote_margin : u[e], rk, (int32_t) l, e});
            }
        }
        if (cand.empty() || vict.empty()) return;
        auto key = [](const Ent& x) { return std::make_tuple(x.rank, x.layer, x.e); };
        std::sort(cand.begin(), cand.end(), [&](const Ent& a, const Ent& b) {
            return a.u != b.u ? a.u > b.u : key(a) < key(b);
        });
        const size_t nc = std::min(cand.size(), vict.size());
        std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(), [&](const Ent& a, const Ent& b) {
            return a.u != b.u ? a.u < b.u : key(a) > key(b);
        });
        for (size_t i = 0; i < nc; ++i) {
            if (cand[i].u < vict[i].u + 1.5f) break;
            swaps.push_back({cand[i].u - vict[i].u, cand[i].layer, cand[i].e, vict[i].e, vict[i].layer});
        }
        std::stable_sort(swaps.begin(), swaps.end(), [](const Swap& a, const Swap& b) { return a.gain > b.gain; });
        return;
    }
    std::vector<std::pair<float, int32_t>> cand, vict;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        cand.clear();
        vict.clear();
        const float* u = drive.d.usage.data() + l * g.n_expert;
        const int32_t* r = host_res.data() + l * g.n_expert;
        float margin = promote_margin;
        if (promote_load > 0.0f && peer.valid()) {
            float all = 0.0f, on = 0.0f;
            for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                all += u[e];
                if (r[e] < 0 && peer.has(l, e)) on += u[e];
            }
            if (all > 0.0f && on * 10.0f > promote_load * all) margin = 0.0f;
        }
        for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
            if (r[e] < 0) {
                const bool on_peer = peer.valid() && peer.has(l, e);
                if (u[e] >= 2.0f && !(remote_opt && remote_opt->owns(l, e)))
                    cand.emplace_back(on_peer ? u[e] - margin : u[e], e);
            }
            else vict.emplace_back(u[e], e);
        }
        if (cand.empty() || vict.empty()) continue;
        std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
        const size_t nc = std::min(cand.size(), vict.size());
        std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                          [](auto& a, auto& b) { return a.first < b.first; });
        for (size_t i = 0; i < nc; ++i) {
            if (cand[i].first < vict[i].first + 1.5f) break;
            swaps.push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second, (int32_t) l});
        }
    }
    std::sort(swaps.begin(), swaps.end(), [](const Swap& a, const Swap& b) { return a.gain > b.gain; });
}

bool Runtime::adapt() {
    if (!pending.empty()) return true;   // the previous swaps are still in flight
    if (remote_opt && !remote_opt->adapt(drive.d.usage, host_res, pending, o.adapt_swaps, *srcp)) return false;
    std::vector<Swap> swaps;
    rank_swaps(swaps, false);
    if ((int) swaps.size() > o.adapt_swaps) swaps.resize((size_t) o.adapt_swaps);
    std::vector<int> staged_on;   // the devices whose adapt streams hold copies back (-1: CUDA0)
    const auto copy_back = [&](int32_t layer, int32_t slot, uint8_t* dst) -> bool {
        const int stn = multi_gpu ? stage_of(layer) : 0;
        GpuStage* gs = stn > 0 ? stages[(size_t) stn - 1].get() : nullptr;
        const basalt::core::OnDevice on(gs ? gs->dev : -1);
        staged_on.push_back(gs ? gs->dev : -1);
        return cudaMemcpyAsync(dst, gs ? gs->cache.device_slot(slot) : xcache.device_slot(slot),
                               (size_t) basalt::kernels::cpu::expert_layout().blob_bytes(layer),
                               cudaMemcpyDeviceToHost, gs ? gs->adapt_stream : adapt_stream) == cudaSuccess;
    };
    const auto wait_copies = [&]() -> bool {
        bool ok = true;
        for (const auto& st : stages)
            if (std::find(staged_on.begin(), staged_on.end(), st->dev) != staged_on.end()) {
                const basalt::core::OnDevice on(st->dev);
                ok = cudaStreamSynchronize(st->adapt_stream) == cudaSuccess && ok;
            }
        if (std::find(staged_on.begin(), staged_on.end(), -1) != staged_on.end())
            ok = cudaStreamSynchronize(adapt_stream) == cudaSuccess && ok;
        return ok;
    };
    if (!resident_spare_swaps(src, host_res, g.n_expert, swaps, copy_back, wait_copies, pending_spare)) return false;
    bool main_live = false;
    for (const Swap& s : swaps) {
        const size_t in = (size_t) s.layer * g.n_expert + s.in, out = (size_t) s.layer * g.n_expert + s.out;
        const int32_t slot = host_res[out];
        const uint8_t* b = srcp->blob(s.layer, s.in);
        const int stn = multi_gpu ? stage_of(s.layer) : 0;   // the swap stays in the layer's own cache
        GpuStage* gs = stn > 0 ? stages[(size_t) stn - 1].get() : nullptr;
        const basalt::core::OnDevice on(gs ? gs->dev : -1);
        if (slot < 0 || b == nullptr ||
            cudaMemcpyAsync(gs ? gs->cache.device_slot(slot) : xcache.device_slot(slot), b,
                            (size_t) basalt::kernels::cpu::expert_layout().blob_bytes(s.layer),
                            cudaMemcpyHostToDevice, gs ? gs->adapt_stream : adapt_stream) != cudaSuccess) {
            basalt::log::error("basalt serve", "adaptive swap copy failed (layer %d, slot %d): %s",
                         (int) s.layer, (int) slot, cudaGetErrorString(cudaGetLastError()));
            return false;
        }
        if (gs) gs->adapt_live = true;
        else main_live = true;
        host_res[out] = basalt::core::kNotResident;   // evicted now: the CPU computes it meanwhile
        srcp->prefetch(s.layer, s.out);   // a file-backed arena released its pages: read them back ahead
        pending.emplace_back((int32_t) in, slot);      // resident once the copy has landed
        if (peer.valid() && peer.has(s.layer, s.in)) promoted.push_back({s.layer, s.in, s.out});
    }
    if (!swaps.empty()) cudaEventRecord(adapt_ev, adapt_stream);
    (void) main_live;
    for (auto& st : stages)
        if (st->adapt_live) {
            const basalt::core::OnDevice on(st->dev);
            cudaEventRecord(st->adapt_ev, st->adapt_stream);
        }
    // the routing counted since the start (each count adds up to 1 / (1 - --adapt-decay) over its decays: the
    // sum is proportional to the routing itself) - only with --expert-profile-save, else `heat` is empty
    for (size_t i = 0; i < heat.size(); ++i) heat[i] += (double) drive.d.usage[i];
    if (peer.valid()) {                   // multi-GPU: the peer tiers take the next most-routed CPU misses
        std::vector<int32_t> r0 = host_res;
        for (const auto& [i, slot] : pending) r0[(size_t) i] = slot;   // swapped into the primary already
        std::string perr;
        if (!adapt_peers(r0, perr)) {
            basalt::log::error("basalt serve", "%s", perr.c_str());
            return false;
        }
    }
    for (float& v : drive.d.usage) v *= o.adapt_decay;
    return true;
}

void Runtime::xq_settle() {
    if (!xq_unsettled) return;
    cudaEventSynchronize(adapt_ev);
    for (const auto& [l, e] : xq_released) srcp->release(l, e);
    xq_released.clear();
    res_upload();
    xq_unsettled = false;
}

bool Runtime::xq_round() {
    xq_settle();
    std::vector<Swap> swaps;
    rank_swaps(swaps, xq_across);
    const size_t n_ranked = swaps.size();
    // the base share goes first, then the peer takes its own from the rest, then a saturated pipeline's extra
    // swaps take what the peer left: a large round never takes the peer's candidates
    const size_t base = (size_t) o.adapt_swaps;
    const size_t cap = xq_saturated ? base * (size_t) kSatBoost : base;
    xq_saturated = n_ranked >= base * (size_t) kSatRatio;
    if (swaps.size() > cap) swaps.resize(cap);
    xq.clear();
    size_t n_noplace = 0;
    const auto& lay = basalt::kernels::cpu::expert_layout();
    std::set<size_t> taken;   // the round's victims
    for (const Swap& s : swaps) taken.insert((size_t) s.lout * g.n_expert + s.out);
    auto plan = [&](Swap s) -> bool {
        // an expert only the peer holds is copied from its VRAM; it needs a RAM place for the one it replaces
        const bool p2p = peer_excl && peer.has(s.layer, s.in) && !src.has_resident(s.layer, s.in);
        if (p2p && src.has_resident(s.layer, s.out)) {
            // a lent slot's expert keeps its RAM place, which the incoming one would not have: the layer's
            // least-routed expert without one goes instead, if that is still a gain
            const float* u = drive.d.usage.data() + (size_t) s.layer * g.n_expert;
            int32_t best = -1;
            for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                const size_t k = (size_t) s.layer * g.n_expert + e;
                if (host_res[k] < 0 || src.has_resident(s.layer, e) || taken.count(k)) continue;
                if (best < 0 || u[e] < u[best]) best = e;
            }
            if (best < 0 || u[s.in] - promote_margin < u[best] + 1.5f) return true;
            taken.insert((size_t) s.layer * g.n_expert + best);
            s.out = best;
        }
        const int32_t slot = host_res[(size_t) s.lout * g.n_expert + s.out];
        if (slot < 0) return true;
        const bool xchg = src.complement_ready() && (p2p || src.has_resident(s.layer, s.in)) &&
                          !src.has_resident(s.lout, s.out);
        uint64_t off = 0;
        if (xchg) {
            uint8_t* dst = src.spare_take(s.lout, off);
            if (dst == nullptr) { ++n_noplace; return true; }   // no free place: a later round
            if (cudaMemcpyAsync(dst, xcache.device_slot(slot), (size_t) lay.blob_bytes(s.lout),
                                cudaMemcpyDeviceToHost, adapt_stream) != cudaSuccess) {
                basalt::log::error("basalt serve", "an eviction copy failed (layer %d, slot %d): %s",
                             (int) s.lout, (int) slot, cudaGetErrorString(cudaGetLastError()));
                return false;
            }
        }
        if (p2p) peer.hold(s.layer, s.in);
        xq.push_back({s.layer, s.in, s.out, s.lout, slot, off, xchg,
                      peer.valid() && peer.has(s.layer, s.in), p2p});
        return true;
    };
    const size_t n_base = std::min(swaps.size(), base);
    for (size_t i = 0; i < n_base; ++i)
        if (!plan(swaps[i])) return false;
    for (size_t i = 0; i < heat.size(); ++i) heat[i] += (double) drive.d.usage[i];
    std::vector<size_t> peer_takes;
    if (peer.valid()) {   // the peer tiers take the next most-routed CPU misses (not the ones moving to CUDA0)
        std::vector<int32_t> r0 = host_res;
        for (const XSwap& x : xq) r0[(size_t) x.layer * g.n_expert + x.in] = x.slot;
        std::string perr;
        if (!adapt_peers(r0, perr)) {
            basalt::log::error("basalt serve", "%s", perr.c_str());
            return false;
        }
        if (swaps.size() > n_base) {
            peer_takes = peer.incoming();
            for (const auto& xpe : xpeers)
                if (xpe->valid())
                    for (size_t i : xpe->incoming()) peer_takes.push_back(i);
        }
        std::sort(peer_takes.begin(), peer_takes.end());
    }
    for (size_t i = n_base; i < swaps.size(); ++i) {
        if (std::binary_search(peer_takes.begin(), peer_takes.end(),
                               (size_t) swaps[i].layer * g.n_expert + swaps[i].in)) continue;
        if (!plan(swaps[i])) return false;
    }
    xq_swapped += (int64_t) xq.size();
    xq_noplace += (int64_t) n_noplace;
    if (!xq.empty()) {
        cudaEventRecord(adapt_ev, adapt_stream);
        xq_stage = 1;
    }
    for (float& v : drive.d.usage) v *= fast_decay;
    return true;
}

// `wait`: step (3) waits for the copies even with the landed words (the batch windows, a drain)
bool Runtime::xq_advance(bool wait) {
    if (xq_stage == 0) return true;
    if (xq_stage == 2) ver.flush_post_launch();
    const bool nowait = xq_stage == 2 && xq_words && !wait;
    if (!nowait) cudaEventSynchronize(adapt_ev);
    if (xq_stage == 1) {
        for (const XSwap& x : xq) {
            if (x.xchg && !src.spare_land(x.lout, x.out, x.off)) {
                basalt::log::error("basalt serve", "an evicted expert has no RAM place (layer %d, expert %d)",
                             (int) x.lout, (int) x.out);
                return false;
            }
            host_res[(size_t) x.lout * g.n_expert + x.out] = basalt::core::kNotResident;
        }
        {   // the residency uploads go ahead of the incoming copies in the copy queue
            std::string perr;
            if (!ver.dm_presync(perr)) basalt::log::warn("basalt serve", "%s", perr.c_str());
            res_upload();
            ver.dm_fence(adapt_stream);
        }
        bool deferred = false, p2p = false;
        for (const XSwap& x : xq) {   // stream order keeps each behind its slot's eviction copy
            std::string perr;
            if (x.p2p) {
                if (!peer.copy_to(x.layer, x.in, xcache.device_slot(x.slot), adapt_stream, perr)) {
                    basalt::log::error("basalt serve", "%s", perr.c_str());
                    return false;
                }
                p2p = true;
                continue;
            }
            deferred = true;
        }
        if (p2p) cudaEventRecord(p2p_ev, adapt_stream);
        const uint32_t* pw = nullptr;   // this round's landed words and number
        xq_words = xq_pin != nullptr && ver.landed_ok();
        if (xq_words) {
            ++xq_no;
            uint32_t* w = xq_pin + (size_t) (xq_no % kXqPinRounds) * (size_t) (g.n_layers + 1);
            // a layer without an incoming RAM copy is in now (its copies out of the peer are ahead), the others
            // once their own copies are
            for (int64_t l = 0; l < g.n_layers; ++l) w[l] = xq_no;
            for (const XSwap& x : xq)
                if (!x.p2p) w[x.layer] = xq_no - 1;
            w[g.n_layers] = xq_no;
            if (cudaMemcpyAsync(ver.landed_words(), w, (size_t) g.n_layers * 4, cudaMemcpyHostToDevice, adapt_stream) !=
                cudaSuccess) {
                basalt::log::error("basalt serve", "the landed words' copy failed: %s", cudaGetErrorString(cudaGetLastError()));
                return false;
            }
            pw = w;
        }
        if (deferred) {   // the RAM copies are queued once the next window is launched, and run beside it
            ver.set_post_launch([&, pw] {
                const auto& lay2 = basalt::kernels::cpu::expert_layout();
                std::vector<uint32_t> ord;   // layer order: a layer's word goes up right after its own copies
                for (uint32_t i = 0; i < (uint32_t) xq.size(); ++i)
                    if (!xq[i].p2p) ord.push_back(i);
                std::stable_sort(ord.begin(), ord.end(), [&](uint32_t a, uint32_t b) { return xq[a].layer < xq[b].layer; });
                for (size_t k = 0; k < ord.size(); ++k) {
                    const XSwap& x = xq[ord[k]];
                    const uint8_t* b = srcp->blob(x.layer, x.in);
                    if (b == nullptr || cudaMemcpyAsync(xcache.device_slot(x.slot), b, (size_t) lay2.blob_bytes(x.layer),
                                                        cudaMemcpyHostToDevice, adapt_stream) != cudaSuccess)
                        basalt::log::warn("basalt serve", "adaptive swap copy failed (layer %d, slot %d): %s",
                                     (int) x.layer, (int) x.slot, cudaGetErrorString(cudaGetLastError()));
                    if (pw != nullptr && (k + 1 == ord.size() || xq[ord[k + 1]].layer != x.layer) &&
                        cudaMemcpyAsync(ver.landed_words() + x.layer, pw + g.n_layers, 4, cudaMemcpyHostToDevice,
                                        adapt_stream) != cudaSuccess)
                        basalt::log::warn("basalt serve", "the landed word's copy failed (layer %d): %s", (int) x.layer,
                                     cudaGetErrorString(cudaGetLastError()));
                }
                cudaEventRecord(adapt_ev, adapt_stream);
            });
        } else {
            cudaEventRecord(adapt_ev, adapt_stream);
        }
        xq_stage = 2;
        return true;
    }
    // with the landed words the incoming experts are resident from the next window on while their copies land:
    // its gate/up launches wait for them, and what needs them done waits in xq_settle
    if (nowait && std::any_of(xq.begin(), xq.end(), [](const XSwap& x) { return x.p2p; }))
        cudaEventSynchronize(p2p_ev);   // a promotion's peer slot is refilled below
    for (const XSwap& x : xq) {
        host_res[(size_t) x.layer * g.n_expert + x.in] = x.slot;
        if (x.xchg && !x.p2p) src.spare_release(x.layer, x.in);
        if (nowait) xq_released.emplace_back(x.layer, x.in);
        else srcp->release(x.layer, x.in);
    }
    if (nowait) {
        std::string perr;
        if (!ver.landed_need(xq_no, perr)) {
            basalt::log::error("basalt serve", "%s", perr.c_str());
            return false;
        }
        xq_unsettled = true;
    } else {
        res_upload();
    }
    for (const XSwap& x : xq) {   // a promoted expert's peer slot takes the one CUDA0 gave up, from any layer
        if (!x.promo) continue;
        std::string perr;
        if (!peer.reassign(x.layer, x.in, x.lout, x.out, srcp->blob(x.lout, x.out), perr))
            basalt::log::warn("basalt serve", "%s", perr.c_str());
    }
    xq.clear();
    xq_stage = 0;
    return true;
}

bool Runtime::xq_drain() {
    while (xq_stage != 0)
        if (!xq_advance(true)) return false;
    xq_settle();
    return true;
}

int Runtime::init_tiers() {
    all_experts_resident = !host_res.empty() &&
        std::all_of(host_res.begin(), host_res.end(), [](int32_t r) { return r >= 0; });
    if (o.adapt_every > 0 && o.adapt_swaps > 0 && !all_experts_resident)
        drive.d.usage.assign((size_t) (g.n_layers * g.n_expert), 0.0f);

    if (!o.expert_profile_save.empty() && !all_experts_resident) {
        if (drive.d.usage.empty() || host_res.empty())
            basalt::log::warn("basalt serve", "--expert-profile-save needs the adaptive tier (--adapt-every and "
                                 "--adapt-swaps above 0) and an expert profile: nothing will be saved");
        else
            heat.assign(drive.d.usage.size(), 0.0);
    }
    profile_saved_at = Clock::now();
    adapt_stream = nullptr;
    if (cudaStreamCreateWithFlags(&adapt_stream, cudaStreamNonBlocking) != cudaSuccess) {
        basalt::log::error("basalt serve", "cannot create the refill stream");
        return 1;
    }

    adapt_ev = nullptr;
    cudaEventCreateWithFlags(&adapt_ev, cudaEventDisableTiming);

    // the peer's swaps land a window after their copies are queued, so no window waits for a copy
    peer_landing = false;
    xlanding = std::vector<uint8_t>(xpeers.size(), 0);

    // the profile's rank of each pair: of equally routed experts the better-ranked one comes in and the worse-ranked
    // one leaves first
    prof_rank = std::vector<int32_t>((size_t) (g.n_layers * g.n_expert), (int32_t) profile.size());
    for (size_t i = 0; i < profile.size(); ++i) {
        const int64_t pl = profile[i].first, pe = profile[i].second;
        if (pl >= 0 && pl < g.n_layers && pe >= 0 && pe < g.n_expert && prof_rank[(size_t) (pl * g.n_expert + pe)] == (int32_t) profile.size())
            prof_rank[(size_t) (pl * g.n_expert + pe)] = (int32_t) i;
    }

    // The resident RAM mode on one GPU, with or without the peer: a round's swaps move in three steps at window
    // boundaries, with no host copy and no blocking copy on the decode path.  (1) The evicted experts' slots are
    // copied by DMA into free places of the RAM copy (they stay resident meanwhile).  (2) Those places become
    // theirs, the CPU computes them, and the incoming experts are copied into the slots.  (3) The incoming experts
    // are resident and their RAM places free.  Step (2) waits for the eviction copies; at step (3) the next
    // window's gate/up launches wait for their own layer's incoming copies (the landed words below), so the loop
    // does not, and where an expert runs depends only on the window count, as with the plain rounds.
    adapt_fast = !multi_gpu && !remote_opt;
    // Its rounds come every --adapt-every windows, so the counts decay less per round than the plain default
    // (--adapt-decay when given, else 0.85): the routing counts stay large enough for the thresholds and the peer
    // margin.
    fast_decay = o.adapt_decay_set ? o.adapt_decay : 0.85f;   // PROBE adapt sets it

    peer_excl = o.peer_exclusive && peer.valid() && src.complement_ready();
    // the rounds rank over all layers when at most one peer tier holds experts (not exclusively) and every layer's
    // blobs are one size (a slot then takes any layer's expert)
    xq_across = xpeers.empty() && !peer_excl;
    for (int64_t l = 1; xq_across && l < g.n_layers; ++l)
        xq_across = basalt::kernels::cpu::expert_layout().blob_bytes(l) == basalt::kernels::cpu::expert_layout().blob_bytes(0);

    xq_stage = 0;   // 0 idle, 1 evictions copying to RAM, 2 incoming experts copying to their slots
    xq_saturated = false;   // the last round had at least kSatRatio times --adapt-swaps misses to swap
    xq_swapped = 0;
    xq_noplace = 0;

    xq_no = 0;
    xq_words = false;   // the round in flight writes the landed words
    xq_pin = nullptr;
    if (ver.landed_words() != nullptr &&
        cudaHostAlloc((void**) &xq_pin, (size_t) kXqPinRounds * (size_t) (g.n_layers + 1) * 4, cudaHostAllocPortable) !=
            cudaSuccess) {
        cudaGetLastError();
        xq_pin = nullptr;
    }
    p2p_ev = nullptr;   // the copies out of the peer's slots, which a promotion's reassign refills
    cudaEventCreateWithFlags(&p2p_ev, cudaEventDisableTiming);
    // a round made resident before its copies were done: the bookkeeping that waits for them (the next round or
    // a drain), and the RAM pages of its experts
    xq_unsettled = false;
    return 0;
}

}  // namespace basalt::program::serve
