// src/program/serve/kv.cpp - kv operations.
#include "state.hpp"

namespace basalt::program::serve {

// the expert cache never shrinks below the slots the prompt path's smallest loan needs (a 256-token chunk, and
// the 128 slots a loan leaves), so the next prompt can always be read
int64_t Runtime::cache_floor_slots() {
    if (pf_parts.empty() || pf_parts[0].first < 0) return 0;
    const int64_t k = part_slots(pf_parts[0], 256);
    int64_t f = k + 128;
    if (o.prefill_auto) f = std::max<int64_t>(f, (k * 100 + auto_lend_pct - 1) / auto_lend_pct);
    return std::min<int64_t>(f, xcache.full_slots());
}

// The experts the decoding slots routed lately leave CUDA0's slots from `first` on, which a prompt's loan or the
// KV is about to take: each trades places with a less-routed expert of its layer below `first` that has a RAM
// copy (a slot keeps its layer's size; the experts from `first` on come back from RAM).  One blob copy per
// trade; the table is uploaded by the caller.  `why` names the taker for the log.  `lent_out`: entries whose
// slots a loan still out holds (the prompt path's buffers, no expert), so their experts are copied from RAM.
bool Runtime::move_routed_below(int64_t first, const char* why, std::string& e,
                                const std::vector<serve_loans::LentEntry>* lent_out) {

    if (drive.d.usage.empty() || srcp != &src || !src.complement_ready() || multi_gpu || first <= 0 ||
        first >= xcache.slots())
        return true;
    const auto t0 = Clock::now();
    const auto& lay = basalt::kernels::cpu::expert_layout();
    int64_t moved = 0, routed = 0, from_ram = 0;
    std::vector<uint8_t> out_of;
    if (lent_out != nullptr && !lent_out->empty()) {
        out_of.assign(host_res.size(), 0);
        for (const auto& [i, slot] : *lent_out) out_of[(size_t) i] = 1;
    }
    std::vector<uint8_t> in_ram((size_t) g.n_expert);
    std::vector<serve_loans::RoutedTrade> trades;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const float* u = drive.d.usage.data() + (size_t) (l * g.n_expert);
        int32_t* r = host_res.data() + (size_t) (l * g.n_expert);
        for (int64_t ex = 0; ex < g.n_expert; ++ex) in_ram[(size_t) ex] = src.has_resident(l, ex) ? 1 : 0;
        trades.clear();
        routed += serve_loans::plan_routed_trades(u, r, in_ram.data(),
                                                  out_of.empty() ? nullptr : out_of.data() + (size_t) (l * g.n_expert),
                                                  (int32_t) g.n_expert, first, trades);
        for (const serve_loans::RoutedTrade& t : trades) {
            const void* from = xcache.device_slot(t.from);
            cudaMemcpyKind kind = cudaMemcpyDeviceToDevice;
            if (t.from_ram) {
                if ((from = srcp->blob(l, t.hot)) == nullptr) {
                    e = "moving an expert out of the cache's tail: its RAM copy is gone";
                    return false;
                }
                kind = cudaMemcpyHostToDevice;
                ++from_ram;
            }
            if (cudaMemcpyAsync(xcache.device_slot(t.to), from, (size_t) lay.blob_bytes(l), kind, adapt_stream) !=
                cudaSuccess) {
                e = std::string("moving an expert out of the cache's tail: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
            r[t.hot] = t.to;
            r[t.cold] = t.from;
            ++moved;
        }
    }
    if (moved > 0 && cudaStreamSynchronize(adapt_stream) != cudaSuccess) {
        e = std::string("moving experts out of the cache's tail: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    const std::string ram_note = from_ram > 0 ? std::to_string(from_ram) + " from RAM, " : std::string();
    basalt::log::info("basalt batch", "%s: %lld of the %lld routed experts in the cache's last %lld slots moved "
                         "below them (%s%.1f ms)", why, (long long) moved, (long long) routed,
                 (long long) (xcache.slots() - first), ram_note.c_str(),
                 std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    return true;
}

bool Runtime::kv_drain(std::string& e) {
    if (!ver.wait_commit(e)) return false;
    apply_pending(true);
    if (!xq_drain()) { e = "elastic KV: an adaptive swap failed"; return false; }
    for (const PfPart& p : pf_parts)
        if (!p.lent.empty()) { e = "elastic KV: a prompt's loan is still out"; return false; }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        e = std::string("elastic KV: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    return true;
}

// maps the cache's tail again as far as the free VRAM above the reserve allows, and refills the slots it gets
// back: each with the expert it held, or (when that one is resident again, or has no RAM copy any more) the
// layer's most-routed expert that no GPU holds and RAM does
bool Runtime::cache_regrow(std::string& e) {
    if (!xcache.segmented() || xcache.mapped_bytes() >= xcache.full_bytes()) return true;
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    const int64_t room = (int64_t) free_b - ((int64_t) o.vram_reserve_mib << 20);
    if (room <= 0) return true;
    std::string gerr;
    (void) xcache.grow(xcache.mapped_bytes() + room, gerr);   // what it could not map stays with the driver
    const int64_t live = xcache.slots();
    auto usable = [&](int64_t j) {
        const int64_t l = j / g.n_expert, ex = j % g.n_expert;
        return host_res[(size_t) j] < 0 && (!src.complement_ready() || src.has_resident(l, ex)) &&
               !(peer.valid() && peer.has(l, ex));
    };
    std::vector<std::pair<int32_t, int32_t>> keep_out;
    std::string ferr;
    for (const auto& [i, slot] : vram_evicted) {
        if (slot >= live) { keep_out.emplace_back(i, slot); continue; }
        const int64_t layer = i / g.n_expert;
        int64_t pick = usable(i) ? i : -1;
        if (pick < 0) {
            float best = -1.0f;
            for (int64_t ex = 0; ex < g.n_expert; ++ex) {
                const int64_t j = layer * g.n_expert + ex;
                if (!usable(j)) continue;
                const float u = drive.d.usage.empty() ? 0.0f : drive.d.usage[(size_t) j];
                if (u > best) { best = u; pick = j; }
            }
        }
        if (pick < 0) continue;
        const uint8_t* b = srcp->blob(layer, pick % g.n_expert);
        if (b == nullptr || !xcache.fill_slot_queued(slot, b, ferr,
                (int64_t) basalt::kernels::cpu::expert_layout().blob_bytes(layer))) {
            e = "elastic KV: refilling the cache failed: " + ferr;
            return false;
        }
        host_res[(size_t) pick] = slot;
    }
    vram_evicted.swap(keep_out);
    if (!xcache.sync_queued(ferr)) { e = "elastic KV: " + ferr; return false; }
    res_upload();
    relend();
    return true;
}

// `shrink` false: only VRAM that is free now (a prompt read is running beside the windows, its loan in the
// cache's last slots) - false with `e` empty when that is not enough
// `min_cells`: when `cells` would take the cache below its floor, as many as leave it, down to these
bool Runtime::kv_fit(basalt::core::SessionState& s, int64_t cells, std::string& e, bool shrink, int64_t min_cells) {
    kv_short = false;
    if (kv_floor <= 0) return true;
    cells = std::min<int64_t>(std::max<int64_t>(cells, kv_floor), s.max_cells);
    int64_t add = (int64_t) basalt::core::session_kv_grow_bytes(s, cells);
    if (add <= 0) return true;
    const auto t0 = Clock::now();
    if (shrink && !kv_drain(e)) return false;
    const int64_t before = xcache.slots();
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    const int64_t room = std::max<int64_t>(0, (int64_t) free_b - ((int64_t) o.vram_reserve_mib << 20));
    if (!shrink && add > room) { e.clear(); return false; }
    if (add > room && xcache.segmented()) {
        const int64_t gran = (int64_t) std::max<uint64_t>(basalt::core::vmm::granularity(), 1);
        auto keep_for = [&](int64_t a) {
            return std::max<int64_t>(0, (xcache.mapped_bytes() - (a - room)) / gran * gran);
        };
        const int64_t floor = cache_floor_slots();
        auto leaves_floor = [&](int64_t c) {
            const int64_t a = (int64_t) basalt::core::session_kv_grow_bytes(s, c);
            return a <= room || xcache.slots_after_shrink(keep_for(a)) >= floor;
        };
        if (!leaves_floor(cells)) {
            // the most cells on the 256-cell grid that leave it, down to `min_cells`
            int64_t best = -1;
            if (min_cells > 0 && min_cells < cells && leaves_floor(min_cells)) {
                best = min_cells;
                for (int64_t lo = min_cells / 256 + 1, hi = cells / 256; lo <= hi;) {
                    const int64_t mid = lo + (hi - lo) / 2;
                    if (mid * 256 > min_cells && leaves_floor(mid * 256)) { best = mid * 256; lo = mid + 1; }
                    else hi = mid - 1;
                }
            }
            if (best < 0) {
                kv_short = true;
                const int64_t want = min_cells > 0 && min_cells < cells ? min_cells : cells;
                e = "elastic KV: " + std::to_string(want) + " cells of KV (+" +
                    std::to_string((int64_t) basalt::core::session_kv_grow_bytes(s, want) >> 20) +
                    " MiB) would take the expert cache below the " + std::to_string(floor) +
                    " slots the next prompt chunk needs (" + std::to_string(xcache.slots()) + " now, " +
                    std::to_string(room >> 20) + " MiB free above the reserve)";
                return false;
            }
            cells = best;
            add = (int64_t) basalt::core::session_kv_grow_bytes(s, cells);
        }
    }
    if (add > room && xcache.segmented()) {
        // whole tail segments: everything from the first one at or past `keep` goes
        const int64_t gran = (int64_t) std::max<uint64_t>(basalt::core::vmm::granularity(), 1);
        const int64_t keep = std::max<int64_t>(0, (xcache.mapped_bytes() - (add - room)) / gran * gran);
        if (kv_keep_routed && !move_routed_below(xcache.slots_after_shrink(keep), "elastic KV", e)) return false;
        if (!xcache.shrink(keep, e)) return false;
        const int64_t live = xcache.slots();
        for (size_t i = 0; i < host_res.size(); ++i)
            if (host_res[i] >= live) {
                vram_evicted.emplace_back((int32_t) i, host_res[i]);
                host_res[i] = basalt::core::kNotResident;
            }
        res_upload();
        relend();
    }
    if (!basalt::core::session_kv_map(s, cells, e)) { e = "elastic KV: " + e; return false; }
    ++kv_moves;
    basalt::log::info("basalt serve", "elastic KV: %lld cells backed (+%.0f MiB), the expert cache %lld -> %lld "
                         "slots, in %.1f ms", (long long) basalt::core::session_kv_cells(s),
                 (double) add / 1048576.0, (long long) before, (long long) xcache.slots(),
                 std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    return true;
}

bool Runtime::kv_release(basalt::core::SessionState& s, int64_t cells, std::string& e) {
    if (kv_floor <= 0) return true;
    cells = std::min<int64_t>(std::max<int64_t>(cells, kv_floor), s.max_cells);
    if (basalt::core::session_kv_bytes_for(s, basalt::core::session_kv_cells(s)) <=
        basalt::core::session_kv_bytes_for(s, cells)) return true;
    const auto t0 = Clock::now();
    if (!kv_drain(e)) return false;
    const int64_t before = xcache.slots(), had = basalt::core::session_kv_cells(s);
    if (!basalt::core::session_kv_unmap(s, cells, e)) { e = "elastic KV: " + e; return false; }
    if (!cache_regrow(e)) return false;
    ++kv_moves;
    basalt::log::info("basalt serve", "elastic KV: %lld -> %lld cells backed, the expert cache %lld -> %lld "
                         "slots, in %.1f ms", (long long) had, (long long) basalt::core::session_kv_cells(s),
                 (long long) before, (long long) xcache.slots(),
                 std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    return true;
}

// the elastic KV's spare cells back to the expert cache: every session keeps what it holds (the solo session
// `ss_keep` cells, a decoding slot its next window too), idle ones first; `skip` and slot `skip_slot` are left
// alone, and it stops once `enough` (if set) is true
bool Runtime::kv_give_back(int64_t ss_keep, const basalt::core::SessionState* skip, int skip_slot, const std::function<bool()>& enough, int64_t& given, std::string& e) {
    std::vector<basalt::core::SessionState*> sess;
    std::vector<basalt::program::KvHold> holds;
    auto add = [&](basalt::core::SessionState& s, int64_t keep, bool active) {
        if (&s == skip) return;
        for (size_t i = 0; i < sess.size(); ++i)
            if (sess[i] == &s) {
                holds[i].keep = std::max(holds[i].keep, keep);
                holds[i].active = holds[i].active || active;
                return;
            }
        sess.push_back(&s);
        holds.push_back({basalt::core::session_kv_cells(s), keep, active});
    };
    add(ss, ss_keep, false);
    for (size_t b = 0; !bslot_ss.empty() && b < bslot_ss[0].size() && b < bs.size(); ++b) {
        if ((int) b == skip_slot) continue;
        const BSlot& sb = bs[b];
        add(*bslot_ss[0][b], std::max<int64_t>((int64_t) sb.ids.size(), sb.active ? sb.p + kKvWindowCells : 0),
            sb.active);
    }
    for (const auto& [i, keep] : basalt::program::kv_giveback_order(holds, kv_floor)) {
        if (holds[(size_t) i].backed < 0) continue;
        if (!kv_release(*sess[(size_t) i], keep, e)) return false;
        given += holds[(size_t) i].backed - basalt::core::session_kv_cells(*sess[(size_t) i]);
        if (enough && enough()) break;
    }
    return true;
}

}  // namespace basalt::program::serve
