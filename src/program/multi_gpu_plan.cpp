// src/program/multi_gpu_plan.cpp - see include/basalt/program/multi_gpu_plan.hpp.
#include "basalt/program/multi_gpu_plan.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace basalt::program {

namespace {
// Window model, fitted to cold decode windows of the daily pack on one card, on two unlike cards and on the layer split:
//   ms = gpu bytes / bandwidth  +  host tail of the experts no card holds
// The experts are ranked by use, and the share of reads the hottest `c` of all pairs serve is taken as c^kHitExponent.
constexpr double kGpuGB = 11.9;          // bytes a window moves on its GPUs at the rate the kernels reach, per window
constexpr double kMissMs = 18.1;         // host tail per unit of read mass no card holds
constexpr double kHitExponent = 1.0 / 6.0;
constexpr double kDenseShare = 0.35;     // the part of the GPU bytes that is dense weights (the rest are experts)
constexpr double kSplitFixedMs = 4.4;    // a layer split plans every layer's experts on the host and hands off per window
constexpr double kExchangeUs = 5.4;      // one mailbox exchange of 10 KiB between two cards
constexpr double kWinMargin = 0.95;      // a mechanism beyond the expert tier must beat it by 5%

double bandwidth(const GpuCard& c) { return c.mem_gbps > 0 ? c.mem_gbps : 448.0; }

double read_share(double slots, double pairs) {
    if (pairs <= 0) return 1.0;
    const double x = std::min(1.0, std::max(0.0, slots / pairs));
    return std::pow(x, kHitExponent);
}

double room_slots(const GpuCard& c, uint64_t overhead, uint64_t blob) {
    if (c.vram <= overhead || blob == 0) return 0;
    return (double) (c.vram - overhead) / (double) blob;
}

bool near(double a, double b) { return a <= 0 || b <= 0 || std::fabs(a - b) <= 0.10 * std::max(a, b); }
}  // namespace

const char* mechanism_name(Mechanism m) {
    return m == Mechanism::ExpertTier ? "expert tier" : m == Mechanism::LayerSplit ? "layer split" : "tensor parallel";
}

bool cards_alike(const GpuCard& a, const GpuCard& b) {
    return near(a.mem_gbps, b.mem_gbps) && near((double) a.vram, (double) b.vram);
}

MgPlan plan_multi_gpu(const std::vector<GpuCard>& cards, const MgModel& m) {
    MgPlan plan;
    const size_t n = cards.size();
    if (n < 2) return plan;
    const double pairs = (double) m.pairs;
    plan.equal = true;
    for (size_t i = 1; i < n; ++i) plan.equal = plan.equal && cards_alike(cards[0], cards[i]);
    std::vector<double> slots(n), bw(n);
    double total_slots = 0, total_bw = 0;
    for (size_t i = 0; i < n; ++i) {
        slots[i] = room_slots(cards[i], i == 0 ? m.main_overhead : m.peer_overhead, m.blob_bytes);
        bw[i] = bandwidth(cards[i]);
        total_slots += slots[i];
        total_bw += bw[i];
    }
    const double gpu_main = kGpuGB / bw[0] * 1000.0;
    // expert tier: the main card runs every layer; every card's experts count
    MgPrice tier{Mechanism::ExpertTier, true, gpu_main + kMissMs * (1.0 - read_share(total_slots, pairs))};
    // layer split: each later card runs k layers (k equal on all of them, or by bandwidth) and holds those layers' experts;
    // the main card runs the rest. The cheapest cut is the one the engine's split search finds.
    auto split_ms = [&](const std::vector<double>& f) {
        double gpu = 0, miss = 0;
        for (size_t i = 0; i < n; ++i) {
            gpu += f[i] * kGpuGB / bw[i] * 1000.0;
            miss += f[i] * kMissMs * (1.0 - read_share(slots[i], pairs * f[i]));
        }
        return gpu + miss + kSplitFixedMs;
    };
    std::vector<double> f(n, 0.0);
    double split_best = 1e30;
    for (size_t i = 0; i < n; ++i) f[i] = bw[i] / total_bw;
    split_best = split_ms(f);
    for (int k = 1; k * (int) (n - 1) < m.layers; ++k) {
        double rest = 1.0;
        for (size_t i = 1; i < n; ++i) { f[i] = (double) k / m.layers; rest -= f[i]; }
        f[0] = rest;
        split_best = std::min(split_best, split_ms(f));
    }
    MgPrice split{Mechanism::LayerSplit, true, split_best};
    // tensor parallel: the dense part of every layer is split across the cards (bandwidth adds up) and every layer pays its
    // exchanges between the cards; the experts stay as in the expert tier
    const double dense_saving = kDenseShare * kGpuGB * (1.0 / bw[0] - 1.0 / total_bw) * 1000.0;
    const double exchange = (double) m.layers * m.tp_exchanges_per_layer * (double) (n - 1) * kExchangeUs *
                            (m.tokens * 2560.0 * 4.0 / 10240.0) / 1000.0;
    MgPrice tp{Mechanism::TensorParallel, false, tier.ms - dense_saving + exchange};
    plan.prices = {tier, split, tp};
    plan.pick = Mechanism::ExpertTier;
    double best = tier.ms;
    if (plan.equal)
        for (const MgPrice& p : plan.prices)
            if (p.runs && p.mechanism != Mechanism::ExpertTier && p.ms < best * kWinMargin) { plan.pick = p.mechanism; best = p.ms; }
    char buf[256];
    std::snprintf(buf, sizeof buf, "multi-gpu: %zu cards %s; ms per window: expert tier %.1f, layer split %.1f, tensor parallel %.1f%s; picked %s",
                  n, plan.equal ? "alike" : "unlike", tier.ms, split.ms, tp.ms, tp.runs ? "" : " (not built)", mechanism_name(plan.pick));
    plan.line = buf;
    return plan;
}

}  // namespace basalt::program
