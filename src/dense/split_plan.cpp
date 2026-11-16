// src/dense/split_plan.cpp - see include/basalt/dense/split_plan.hpp.
#include "basalt/dense/split_plan.hpp"

#include <algorithm>
#include <cmath>

namespace basalt::dense {

namespace {
std::string mib(uint64_t b) { return std::to_string((unsigned long long) (b >> 20)) + " MiB"; }
}  // namespace

bool split_cards_alike(const SplitCard& a, const SplitCard& b) {
    if (a.mem_gbps <= 0 || b.mem_gbps <= 0) return true;
    return std::fabs(a.mem_gbps - b.mem_gbps) <= 0.1 * std::max(a.mem_gbps, b.mem_gbps);
}

SplitPlan plan_split(const SplitInputs& in) {
    SplitPlan p;
    const int n = (int) in.layer.size();
    // bytes of layers [0, c) on the main card and [c, n) on the second, with every slot's share of them
    std::vector<uint64_t> below((size_t) n + 1, 0), weights((size_t) n + 1, 0);
    for (int l = 0; l < n; ++l) {
        const uint64_t s = l < (int) in.per_slot.size() ? in.per_slot[(size_t) l] : 0;
        below[(size_t) l + 1] = below[(size_t) l] + in.layer[(size_t) l] + (uint64_t) in.slots * s;
        weights[(size_t) l + 1] = weights[(size_t) l] + in.layer[(size_t) l];
    }
    const auto main_at = [&](int c) { return in.main_fixed + below[(size_t) c]; };
    const auto peer_at = [&](int c) { return in.peer_fixed + below[(size_t) n] - below[(size_t) c]; };
    const auto fits = [&](int c) { return main_at(c) <= in.main.free && peer_at(c) <= in.peer.free; };
    if (in.forced == 0 && main_at(n) <= in.main.free) {
        p.main_bytes = main_at(n);
        return p;
    }
    if (in.forced != 0) {
        if (in.forced < 1 || in.forced >= n) {
            p.error = "a cut at layer " + std::to_string(in.forced) + " of " + std::to_string(n);
            return p;
        }
        if (!fits(in.forced)) {
            p.error = "a cut at layer " + std::to_string(in.forced) + " needs " + mib(main_at(in.forced)) + " on the main card (" +
                      mib(in.main.free) + " free) and " + mib(peer_at(in.forced)) + " on the second (" + mib(in.peer.free) + " free)";
            return p;
        }
        p.cut = in.forced;
    } else if (split_cards_alike(in.main, in.peer)) {
        // the prompt chunks pipelined over the two cards: their layers' bytes as even as fits
        double best = -1;
        for (int c = 1; c < n; ++c) {
            if (!fits(c)) continue;
            const double off = std::fabs((double) weights[(size_t) c] - (double) weights[(size_t) n] / 2);
            if (best < 0 || off < best) {
                best = off;
                p.cut = c;
            }
        }
    } else if (in.main.mem_gbps >= in.peer.mem_gbps) {
        for (int c = n - 1; c >= 1 && p.cut == 0; --c)
            if (fits(c)) p.cut = c;
    } else {
        for (int c = 1; c < n && p.cut == 0; ++c)
            if (fits(c)) p.cut = c;
    }
    if (p.cut == 0) {
        p.error = "the model and its cells need " + mib(main_at(n)) + " on one card, and no cut fits them in the two cards' " +
                  mib(in.main.free) + " and " + mib(in.peer.free) + " free";
        return p;
    }
    p.main_bytes = main_at(p.cut);
    p.peer_bytes = peer_at(p.cut);
    return p;
}

StageSplit plan_stage_split(const SplitInputs& in, const std::vector<SplitCard>& cards, uint64_t stage_bytes) {
    StageSplit out;
    const int n = (int) in.layer.size();
    const int nc = (int) cards.size();
    if (nc < 2 || n < nc) {
        out.error = "a layer split needs a card per stage and a layer per card";
        return out;
    }
    std::vector<uint64_t> below((size_t) n + 1, 0), weights((size_t) n + 1, 0);
    for (int l = 0; l < n; ++l) {
        const uint64_t s = l < (int) in.per_slot.size() ? in.per_slot[(size_t) l] : 0;
        below[(size_t) l + 1] = below[(size_t) l] + in.layer[(size_t) l] + (uint64_t) in.slots * s;
        weights[(size_t) l + 1] = weights[(size_t) l] + in.layer[(size_t) l];
    }
    const auto fixed = [&](int c) {
        if (c == 0) return in.main_fixed;
        if (c == nc - 1) return in.peer_fixed;
        return in.peer_fixed + stage_bytes;
    };
    const auto fits = [&](int c, int a, int b) { return fixed(c) + below[(size_t) b] - below[(size_t) a] <= cards[(size_t) c].free; };
    const auto gbps = [&](int c) { return cards[(size_t) c].mem_gbps > 0 ? cards[(size_t) c].mem_gbps : 1.0; };
    double best_peak = 1e300, best_sq = 1e300;
    std::vector<int> best, cur;
    const auto search = [&](auto& self, int c, int layer) -> void {
        if (c == nc - 1) {
            if (layer >= n || !fits(c, layer, n)) return;
            cur.push_back(n);
            double peak = 0, sq = 0;
            int prev = 0;
            for (int i = 0; i < nc; ++i) {
                const double load = (double) (weights[(size_t) cur[(size_t) i]] - weights[(size_t) prev]) / gbps(i);
                peak = std::max(peak, load);
                sq += load * load;
                prev = cur[(size_t) i];
            }
            if (peak < best_peak || (peak == best_peak && sq < best_sq)) {
                best_peak = peak;
                best_sq = sq;
                best = cur;
            }
            cur.pop_back();
            return;
        }
        const int latest = n - (nc - 1 - c);
        for (int end = layer + 1; end <= latest; ++end) {
            if (!fits(c, layer, end)) continue;
            const double load = (double) (weights[(size_t) end] - weights[(size_t) layer]) / gbps(c);
            if (load > best_peak) continue;
            cur.push_back(end);
            self(self, c + 1, end);
            cur.pop_back();
        }
    };
    search(search, 0, 0);
    if (best.empty()) {
        out.error = "the model and its cells need " + mib(in.main_fixed + below[(size_t) n]) + " and no split fits them across the " +
                    std::to_string(nc) + " cards' ";
        for (int c = 0; c < nc; ++c) out.error += (c ? " and " : "") + mib(cards[(size_t) c].free);
        out.error += " free";
        return out;
    }
    out.bounds.push_back(0);
    int prev = 0;
    for (int c = 0; c < nc; ++c) {
        const int end = best[(size_t) c];
        out.bounds.push_back(end);
        out.bytes.push_back(fixed(c) + below[(size_t) end] - below[(size_t) prev]);
        prev = end;
    }
    return out;
}

}  // namespace basalt::dense
