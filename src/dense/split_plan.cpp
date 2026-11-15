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

}  // namespace basalt::dense
