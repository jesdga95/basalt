// src/model/multi_gpu.cpp - see include/basalt/model/multi_gpu.hpp.
#include "basalt/model/multi_gpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace basalt::model {

namespace {

std::vector<double> normalized(const std::vector<double>& w) {
    double sum = 0;
    bool known = !w.empty();
    for (double v : w) {
        known = known && v > 0;
        sum += v;
    }
    std::vector<double> out(w.size(), w.empty() ? 0.0 : 1.0 / (double) w.size());
    if (known)
        for (size_t i = 0; i < w.size(); ++i) out[i] = w[i] / sum;
    return out;
}

std::string share_text(double v) {
    char b[16];
    std::snprintf(b, sizeof b, "%.2f", v);
    return b;
}

// "RTX 5060 Ti (nvidia-smi 1)"
std::string card_name(const GpuCards::Card& c) {
    return (c.info.name.empty() ? std::string("GPU") : c.info.name) +
           (c.info.smi >= 0 ? " (nvidia-smi " + std::to_string(c.info.smi) + ")" : std::string());
}

// the mode's name as the settings spell it (gpu_select.cpp's, which this library does not link)
const char* mode_name(program::GpuMode m) {
    return m == program::GpuMode::Peer ? "peer" : m == program::GpuMode::Split ? "split" : m == program::GpuMode::Tensor ? "tensor" : "";
}

}  // namespace

std::vector<int> GpuCards::peers() const {
    std::vector<int> out;
    for (size_t i = 1; i < cards.size(); ++i) out.push_back(cards[i].device);
    return out;
}

Shares shares_for(const GpuCards& c, ShareUse use) {
    Shares s;
    if (!c.given.empty() && c.given.size() == c.cards.size()) {
        s.of = c.given;
        s.from = "gpu-share";
        return s;
    }
    std::vector<double> w;
    for (const GpuCards::Card& k : c.cards) w.push_back(use == ShareUse::Stages ? k.info.mem_gbps : (double) k.free);
    bool known = !w.empty();
    for (double v : w) known = known && v > 0;
    s.of = normalized(w);
    s.from = !known ? "equal" : use == ShareUse::Stages ? "bandwidth" : "free VRAM";
    return s;
}

std::vector<double> peer_shares(const GpuCards& c, ShareUse use) {
    const Shares s = shares_for(c, use);
    if (s.of.size() < 2) return {};
    return normalized(std::vector<double>(s.of.begin() + 1, s.of.end()));
}

std::string gib(uint64_t bytes) {
    char b[32];
    std::snprintf(b, sizeof b, "%.1f GiB", (double) bytes / 1073741824.0);
    return b;
}

std::string cut_layers(const SplitCut& cut, size_t i, int n_layers) {
    const int a = cut.first[i], e = i + 1 < cut.first.size() ? cut.first[i + 1] : n_layers;
    return "layers " + std::to_string(a) + "-" + std::to_string(e - 1);
}

SplitCut split_cut(const GpuCards& c, const SplitSpec& s) {
    SplitCut cut;
    const size_t n = c.cards.size();
    const int L = (int) s.layer.size();
    auto fixed = [&](size_t i) { return i < s.fixed.size() ? s.fixed[i] : 0; };
    auto room = [&](size_t i) { return i < s.room.size() ? s.room[i] : 0; };
    if (n < 2 || L < (int) n) {
        cut.error = "gpu-mode split: " + std::to_string(L) + " layers over " + std::to_string(n) + " cards";
        return cut;
    }
    std::vector<uint64_t> below((size_t) L + 1, 0);
    for (int l = 0; l < L; ++l) below[(size_t) l + 1] = below[(size_t) l] + s.layer[(size_t) l];
    const Shares sh = shares_for(c, ShareUse::Split);
    cut.first.assign(n, 0);
    if (c.diag_cut > 0) {
        if (n != 2 || c.diag_cut >= L) {
            cut.error = "--diag-gpu-cut " + std::to_string(c.diag_cut) + ": expected a layer from 1 to " + std::to_string(L - 1) +
                        " with two cards";
            return cut;
        }
        cut.first[1] = c.diag_cut;
    } else {
        uint64_t total = below[(size_t) L];
        for (size_t i = 0; i < n; ++i) total += fixed(i);
        double want = 0;
        uint64_t fixed_upto = 0;
        auto nothing = [&](size_t i) {
            cut.error = "gpu-mode split: " + c.cards[i].label + " would hold nothing: its share " + share_text(sh.of[i]) + " (" +
                        sh.from + ") is less than a layer; raise its gpu-share or leave it out of gpu";
            return cut;
        };
        for (size_t i = 0; i + 1 < n; ++i) {
            // card i ends at the boundary where cards 0..i come nearest their shares of every card's bytes
            want += sh.of[i] * (double) total;
            fixed_upto += fixed(i);
            int best = cut.first[i];
            double best_off = -1;
            for (int b = cut.first[i]; b <= L; ++b) {
                const double off = std::fabs((double) (fixed_upto + below[(size_t) b]) - want);
                if (best_off < 0 || off < best_off) {
                    best_off = off;
                    best = b;
                }
            }
            if (best == cut.first[i]) return nothing(i);
            if (best == L) return nothing(i + 1);
            cut.first[i + 1] = best;
        }
    }
    for (size_t i = 0; i < n; ++i) {
        const int e = i + 1 < n ? cut.first[i + 1] : L;
        cut.need.push_back(fixed(i) + below[(size_t) e] - below[(size_t) cut.first[i]]);
    }
    for (size_t i = 0; i < n; ++i)
        if (cut.need[i] > room(i)) {
            cut.error = "gpu-mode split: " + c.cards[i].label + " takes " + cut_layers(cut, i, L) +
                        (c.diag_cut > 0 ? " at --diag-gpu-cut " + std::to_string(c.diag_cut) : " at share " + share_text(sh.of[i])) +
                        " and needs " + gib(cut.need[i]) + ", " + gib(room(i)) +
                        " free; lower its gpu-share, max-context or parallel";
            return cut;
        }
    return cut;
}

std::vector<int> deal_experts(const std::vector<uint64_t>& bytes, const std::vector<uint64_t>& room,
                              const std::vector<double>& shares) {
    std::vector<int> owner(bytes.size(), -1);
    const size_t n = room.size();
    std::vector<uint64_t> used(n, 0);
    std::vector<int64_t> taken(n, 0);
    std::vector<bool> open(n, true);
    for (size_t i = 0; i < bytes.size(); ++i) {
        // the least loaded open peer for its share takes it; one it no longer fits closes, and the next is asked
        for (;;) {
            int best = -1;
            double best_load = 0;
            for (size_t t = 0; t < n; ++t) {
                if (!open[t]) continue;
                const double share = t < shares.size() && shares[t] > 0 ? shares[t] : 1.0 / (double) n;
                const double load = (double) taken[t] / share;
                if (best < 0 || load < best_load) {
                    best = (int) t;
                    best_load = load;
                }
            }
            if (best < 0) return owner;
            const size_t b = (size_t) best;
            if (used[b] + bytes[i] > room[b]) {
                open[b] = false;
                continue;
            }
            owner[i] = best;
            used[b] += bytes[i];
            ++taken[b];
            break;
        }
    }
    return owner;
}

std::string peer_holds_nothing(const GpuCards& c, size_t card, const std::string& why) {
    const std::string label = card < c.cards.size() ? c.cards[card].label : "GPU " + std::to_string(card);
    return "gpu-mode peer: " + label + " would hold nothing: " + why +
           (c.cards.size() == 2 ? "; use gpu " + std::to_string(c.cards[0].index) : std::string("; leave it out of gpu"));
}

std::vector<int> stage_cuts(const GpuCards& c, int n_layers, int lo) {
    const size_t n = c.cards.size();
    std::vector<int> first(n, 0);
    if (n < 2) return first;
    const Shares sh = shares_for(c, ShareUse::Stages);
    double cum = 0;
    for (size_t i = 1; i < n; ++i) {
        cum += sh.of[i - 1];
        const int floor = std::max(first[i - 1] + 1, i == 1 ? std::max(lo, 1) : 0);
        const int ceil = n_layers - (int) (n - i);
        first[i] = std::clamp((int) std::lround(cum * (double) n_layers), floor, std::max(floor, ceil));
    }
    return first;
}

std::string gpu_start_line(const GpuCards& c) {
    if (c.cards.empty()) return "gpus: none";
    if (c.cards.size() == 1) return "gpus: " + card_name(c.cards[0]);
    const char* mode = mode_name(c.mode);
    Shares sh;
    if (c.mode == program::GpuMode::Peer) sh = shares_for(c, ShareUse::Stages);
    else if (c.mode == program::GpuMode::Split) sh = shares_for(c, ShareUse::Split);
    else if (!c.given.empty()) sh = shares_for(c, ShareUse::Split);
    else sh = {std::vector<double>(c.cards.size(), 1.0 / (double) c.cards.size()), "equal"};
    std::string line = std::string("gpus: ") + mode;
    for (size_t i = 0; i < c.cards.size(); ++i)
        line += "; " + card_name(c.cards[i]) + (i == 0 ? " main" : std::string(" ") + mode) + ", share " + share_text(sh.of[i]);
    return line + " (" + sh.from + ")";
}

std::string gpu_parts_line(const GpuCards& c, const std::vector<std::string>& parts) {
    std::string line;
    for (size_t i = 0; i < c.cards.size() && i < parts.size(); ++i)
        if (!parts[i].empty())
            line += (line.empty() ? "gpus: " : "; ") + (c.cards[i].info.name.empty() ? std::string("GPU") : c.cards[i].info.name) + " " +
                    parts[i];
    return line;
}

}  // namespace basalt::model
