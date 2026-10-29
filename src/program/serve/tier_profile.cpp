// src/program/serve/tier_profile.cpp - tier profile operations.
#include "state.hpp"

namespace basalt::program::serve {

void Runtime::save_profile(const char* why) {
    if (heat.empty()) return;
    std::vector<uint8_t> resident(host_res.size(), 0);
    for (size_t i = 0; i < host_res.size(); ++i) resident[i] = host_res[i] >= 0;
    for (const auto& p : pending) resident[(size_t) p.first] = 1;
    std::string e;
    const auto ranked = basalt::core::rank_learned_profile(g.n_layers, g.n_expert, resident, heat,
                                                           profile_loaded);
    if (basalt::core::write_expert_profile(o.expert_profile_save, g.n_layers, g.n_expert, ranked, e))
        basalt::log::info("basalt serve", "expert profile saved to %s (%s)", o.expert_profile_save.c_str(),
                     why);
    else
        basalt::log::warn("basalt serve", "the expert profile was not saved: %s", e.c_str());
    profile_saved_at = Clock::now();
}

bool Runtime::probe_adapt(const std::string& cmd, std::string& e) {
    const auto t0 = Clock::now();
    if (drive.d.usage.empty() || !adapt_fast || host_res.empty() || remote_opt) {
        e = "PROBE adapt: needs the adaptive tier on the main card (--adapt-every and --adapt-swaps above 0, "
            "no layer split)";
        return false;
    }
    if (!xpeers.empty() || (peer.valid() && peer.exclusive())) {
        e = "PROBE adapt: not with more than one peer card or an exclusive peer tier";
        return false;
    }
    int every = o.adapt_every, swaps_n = o.adapt_swaps;
    float decay = fast_decay;
    std::string pfile;
    std::istringstream kvs(cmd.substr(11));
    for (std::string kv; kvs >> kv;) {
        const size_t eq = kv.find('=');
        const std::string key = eq == std::string::npos ? kv : kv.substr(0, eq);
        const std::string val = eq == std::string::npos ? std::string() : kv.substr(eq + 1);
        if (key == "every" && !val.empty()) every = std::atoi(val.c_str());
        else if (key == "swaps" && !val.empty()) swaps_n = std::atoi(val.c_str());
        else if (key == "decay" && !val.empty()) decay = (float) std::atof(val.c_str());
        else if (key == "profile" && !val.empty()) pfile = val;
        else { e = "PROBE adapt: bad setting " + kv; return false; }
    }
    if (every < 1 || swaps_n < 1 || !(decay > 0.0f && decay < 1.0f)) {
        e = "PROBE adapt: every and swaps must be at least 1, decay between 0 and 1";
        return false;
    }
    for (const PfPart& p : pf_parts)
        if (!p.lent.empty()) { e = "PROBE adapt: a prompt's loan is still out"; return false; }
    // the target: (slot, residency index) per tier
    std::vector<std::pair<int32_t, int32_t>> want_main, want_peer;
    std::vector<int32_t> rank_new = prof_rank_start;
    if (pfile.empty()) {
        for (size_t i = 0; i < res_at_start.size(); ++i)
            if (res_at_start[i] >= 0) want_main.emplace_back(res_at_start[i], (int32_t) i);
        for (size_t i = 0; i < peer_at_start.size(); ++i)
            if (peer_at_start[i] >= 0) want_peer.emplace_back(peer_at_start[i], (int32_t) i);
    } else {
        const auto& lay = basalt::kernels::cpu::expert_layout();
        for (int64_t l = 1; l < g.n_layers; ++l)
            if (lay.blob_bytes(l) != lay.blob_bytes(0)) { e = "PROBE adapt: profile= needs one blob size"; return false; }
        if (xcache.per_layer_admission()) { e = "PROBE adapt: profile= needs a cache shared by all layers"; return false; }
        std::vector<std::pair<int32_t, int32_t>> q;
        int64_t qslots = 0;
        if (!basalt::core::read_expert_profile(basalt::core::FileRange(pfile), g.n_layers, g.n_expert, q, qslots, e)) {
            e = "PROBE adapt: " + e;
            return false;
        }
        if (peer_hot_share) {
            size_t at = 0, fill = 0;
            (void) peer_hot_share_order(q, at, fill);
        }
        // the start's slots in its ranking's order take the file's ranks in order
        std::vector<int32_t> main_slots, peer_slots;
        for (const auto& pr : profile) {
            const size_t i = (size_t) pr.first * g.n_expert + pr.second;
            if (i >= res_at_start.size()) continue;
            if (res_at_start[i] >= 0) main_slots.push_back(res_at_start[i]);
            else if (i < peer_at_start.size() && peer_at_start[i] >= 0) peer_slots.push_back(peer_at_start[i]);
        }
        std::vector<uint8_t> taken(res_at_start.size(), 0);
        size_t k = 0;
        for (const auto& pr : q) {
            const size_t i = (size_t) pr.first * g.n_expert + pr.second;
            if (k >= main_slots.size()) break;
            if (i >= taken.size() || taken[i]) continue;
            taken[i] = 1;
            want_main.emplace_back(main_slots[k++], (int32_t) i);
        }
        k = 0;
        for (const auto& pr : q) {
            const size_t i = (size_t) pr.first * g.n_expert + pr.second;
            if (k >= peer_slots.size()) break;
            if (i >= taken.size() || taken[i]) continue;
            taken[i] = 1;
            want_peer.emplace_back(peer_slots[k++], (int32_t) i);
        }
        rank_new.assign(rank_new.size(), (int32_t) q.size());
        for (size_t r = 0; r < q.size(); ++r) {
            const size_t i = (size_t) q[r].first * g.n_expert + q[r].second;
            if (i < rank_new.size() && rank_new[i] == (int32_t) q.size()) rank_new[i] = (int32_t) r;
        }
    }
    if (!ver.wait_commit(e)) return false;
    apply_pending(true);
    if (!xq_drain()) { e = "PROBE adapt: an adaptive swap failed"; return false; }
    if (peer.valid()) peer.apply_pending(true);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        e = std::string("PROBE adapt: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    // the cells a longer prompt backed go back to the expert cache first: the slots it had at READY
    if (!kv_release(ss, 0, e)) { e = "PROBE adapt: " + e; return false; }
    // the main card: an expert leaving a slot without a RAM place takes a free one first (as a round's swap)
    const auto& lay = basalt::kernels::cpu::expert_layout();
    const int64_t live_slots = xcache.slots();
    std::vector<int32_t> holder((size_t) live_slots, -1);   // slot -> residency index
    for (size_t i = 0; i < host_res.size(); ++i)
        if (host_res[i] >= 0 && host_res[i] < live_slots) holder[(size_t) host_res[i]] = (int32_t) i;
    // the target is the set of experts: one already in a slot stays there, each missing one takes the slot of
    // an expert outside the set (its own target slot first)
    std::vector<std::pair<int32_t, int32_t>> todo, later;
    const auto pair_up = [](const std::vector<std::pair<int32_t, int32_t>>& want, const std::vector<int32_t>& res,
                            const std::vector<int32_t>& holder, std::vector<std::pair<int32_t, int32_t>>& out) {
        std::vector<uint8_t> in_set(res.size(), 0);
        for (const auto& w : want)
            if (w.first >= 0 && w.first < (int32_t) holder.size()) in_set[(size_t) w.second] = 1;
        std::vector<uint8_t> free_slot(holder.size(), 0);
        for (size_t sl = 0; sl < holder.size(); ++sl) free_slot[sl] = holder[sl] < 0 || !in_set[(size_t) holder[sl]];
        std::vector<int32_t> missing;
        for (const auto& w : want) {
            if (!in_set[(size_t) w.second] || res[(size_t) w.second] >= 0) continue;
            if (free_slot[(size_t) w.first]) { out.push_back(w); free_slot[(size_t) w.first] = 0; }
            else missing.push_back(w.second);
        }
        size_t sl = 0;
        for (int32_t i : missing) {
            while (sl < free_slot.size() && !free_slot[sl]) ++sl;
            if (sl == free_slot.size()) break;
            out.emplace_back((int32_t) sl, i);
            free_slot[sl] = 0;
        }
    };
    pair_up(want_main, host_res, holder, todo);
    struct Mv { int32_t slot, in, out; uint64_t off; bool xchg; };
    std::vector<Mv> mv;
    int64_t moved = 0, kept = 0;
    for (int pass = 0; pass < 2 && !todo.empty(); ++pass) {   // an expert still resident somewhere waits a pass
        later.clear();
        size_t at = 0;
        while (at < todo.size()) {
            mv.clear();
            for (; at < todo.size(); ++at) {
                const auto [slot, in] = todo[at];
                if (host_res[(size_t) in] >= 0) { later.push_back(todo[at]); continue; }
                const int32_t out = holder[(size_t) slot];
                const int64_t li = in / g.n_expert, ei = in % g.n_expert;
                const bool xchg = out >= 0 && src.complement_ready() && src.has_resident(li, ei) &&
                                  !src.has_resident(out / g.n_expert, out % g.n_expert);
                uint64_t off = 0;
                if (xchg) {
                    uint8_t* dst = src.spare_take(out / g.n_expert, off);
                    if (dst == nullptr) {
                        if (mv.empty()) { e = "PROBE adapt: no free RAM place for an evicted expert"; return false; }
                        break;
                    }
                    if (cudaMemcpyAsync(dst, xcache.device_slot(slot), (size_t) lay.blob_bytes(out / g.n_expert),
                                        cudaMemcpyDeviceToHost, adapt_stream) != cudaSuccess) {
                        e = std::string("PROBE adapt: an eviction copy failed: ") + cudaGetErrorString(cudaGetLastError());
                        return false;
                    }
                }
                mv.push_back({slot, in, out, off, xchg});
            }
            if (mv.empty()) break;
            if (cudaStreamSynchronize(adapt_stream) != cudaSuccess) {
                e = std::string("PROBE adapt: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
            for (const Mv& m : mv) {
                if (m.out < 0) continue;
                const int64_t lo = m.out / g.n_expert, eo = m.out % g.n_expert;
                if (m.xchg && !src.spare_land(lo, eo, m.off)) {
                    e = "PROBE adapt: an evicted expert has no RAM place";
                    return false;
                }
                host_res[(size_t) m.out] = basalt::core::kNotResident;
                holder[(size_t) m.slot] = -1;
                srcp->prefetch(lo, eo);
            }
            for (const Mv& m : mv) {
                const int64_t li = m.in / g.n_expert;
                const uint8_t* b = srcp->blob(li, m.in % g.n_expert);
                if (b == nullptr || cudaMemcpyAsync(xcache.device_slot(m.slot), b, (size_t) lay.blob_bytes(li),
                                                    cudaMemcpyHostToDevice, adapt_stream) != cudaSuccess) {
                    e = "PROBE adapt: an expert's copy into its slot failed";
                    return false;
                }
            }
            if (cudaStreamSynchronize(adapt_stream) != cudaSuccess) {
                e = std::string("PROBE adapt: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
            for (const Mv& m : mv) {
                const int64_t li = m.in / g.n_expert, ei = m.in % g.n_expert;
                host_res[(size_t) m.in] = m.slot;
                holder[(size_t) m.slot] = m.in;
                if (m.xchg) src.spare_release(li, ei);
                srcp->release(li, ei);
                ++moved;
            }
        }
        todo.swap(later);
    }
    kept += (int64_t) todo.size();
    res_upload();
    relend();
    int64_t peer_moved = 0;
    if (peer.valid()) {
        peer.apply_pending(true);
        const std::vector<int32_t>& pres = peer.residency();
        std::vector<int32_t> pholder((size_t) peer.cache().slots(), -1);
        for (size_t i = 0; i < pres.size(); ++i)
            if (pres[i] >= 0 && pres[i] < (int32_t) pholder.size()) pholder[(size_t) pres[i]] = (int32_t) i;
        std::vector<std::pair<int32_t, int32_t>> pwant, pw;
        for (const auto& sw : want_peer)
            if (host_res[(size_t) sw.second] < 0) pwant.push_back(sw);
            else ++kept;   // the main card holds it
        pair_up(pwant, pres, pholder, pw);
        if (!peer.place(pw, peer_moved, e)) { e = "PROBE adapt: " + e; return false; }
    }
    // the adaptation and the draft policy start again, as after a load
    std::fill(drive.d.usage.begin(), drive.d.usage.end(), 0.0f);
    std::fill(heat.begin(), heat.end(), 0.0);
    rounds = 0;
    ba_win = 0;
    xq_saturated = false;
    prof_rank = rank_new;
    (*policy) = basalt::spec::DraftPolicy(S);
    (*bpolicy) = basalt::spec::DraftPolicy(S);
    // a request after this reads its whole prompt
    ckd_finish();
    live.clear();
    live_imgs.clear();
    live_ok = false;
    checks.clear();
    for (BSlot& sl : bs)
        if (!sl.active) sl.cached = false;
    (void) conversations->take_reuse(rsp);
    while (conversations->size() > 0) (void) conversations->take(0);
    o.adapt_every = every;
    o.adapt_swaps = swaps_n;
    o.adapt_decay = decay;
    o.adapt_decay_set = true;
    fast_decay = decay;
    std::printf("PROBE end adapt every=%d swaps=%d decay=%.3f moved=%lld peer_moved=%lld kept=%lld ms=%.0f\n",
                every, swaps_n, (double) decay, (long long) moved, (long long) peer_moved, (long long) kept,
                std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    return true;
}

}  // namespace basalt::program::serve
