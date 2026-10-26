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
constexpr double kSplitHandoffMs = 0.7;  // a layer split's hand-off and the commits of its stages, per window
constexpr double kHeadMs = 1.45;         // the head on the last card, at 448 GB/s
constexpr double kDraftMs = 1.75;        // the drafter on the first card, at 448 GB/s
constexpr double kRelayMs = 0.3;         // the last stage's window rows to the drafter's card
constexpr uint64_t kHeadBytes = 600ull << 20;       // the head's copy on the last card, before its experts
constexpr uint64_t kDrafterBytes = 1000ull << 20;   // the drafter and its copy of the head on the first card
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
    // what a card running a share f of the layers holds before its experts: the part of the main card's overhead that
    // follows the layers (dense weights, session) at f, the card's own fixed part, and the drafter or the head
    const double layer_bytes = (double) m.main_overhead - (double) m.peer_overhead - (double) kDrafterBytes;
    auto split_slots = [&](size_t i, double f) {
        const double over = (double) m.peer_overhead + std::max(0.0, layer_bytes) * f + (i == 0 ? (double) kDrafterBytes : 0.0) +
                            (i + 1 == n ? (double) kHeadBytes : 0.0);
        return room_slots(cards[i], (uint64_t) over, m.blob_bytes);
    };
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
            miss += f[i] * kMissMs * (1.0 - read_share(split_slots(i, f[i]), pairs * f[i]));
        }
        return gpu + miss + kSplitHandoffMs + kRelayMs + kHeadMs * 448.0 / bw[n - 1] + kDraftMs * 448.0 / bw[0];
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

namespace {
// Prompt reads, fitted to cold reads of the pack on the main card alone and on the split.  A stage's chunk of T tokens:
//   max(T x (layers x u(pos) + the embedding card's per-token part), the chunk's expert bytes / the link's share)
//   u(pos) = kPsUsLayer x kPsRefGbps / the card's bandwidth x (1 + kPsGrowth x pos), per token and layer
// In the split every card streams its layers' experts at once: a chunk takes no less than the bytes they all stream
// for it over the host's bandwidth to all of them together (probed so; each card's gaps are the others' to fill).
constexpr double kPsUsLayer = 1.69;      // us per token and layer at the start of the context, at kPsRefGbps
constexpr double kPsRefGbps = 1792.0;
constexpr double kPsGrowth = 2.5e-6;     // per cell of context before the token (the attention's block selection)
constexpr double kPsUsToken = 7.1;       // us per token on the card that embeds it (the embedding, the PLE block)
constexpr double kPsLinkEff = 0.85;      // the share of the probed link a chunk's expert stream gets
constexpr double kPsPleUsToken = 46.0;   // the first chunk's wait for its n-gram rows, per token (either way)
constexpr double kPsDraftUsRow = 2.5;    // the draft layer's K/V of a row read back from the tier card
constexpr double kPsMargin = 0.95;       // the split reads a part only when it is at least 5% cheaper

double ps_bw(const PsCard& c, const PsCard& main) {
    return c.mem_gbps > 0 ? c.mem_gbps : main.mem_gbps > 0 ? main.mem_gbps : kPsRefGbps;
}

// A card's expert stream over layers [l0, l1) per chunk, in ms, for every range: [l0 * (layers + 1) + l1]
std::vector<double> ps_link_table(const PsCard& c, int layers) {
    std::vector<double> t((size_t) (layers + 1) * (size_t) (layers + 1), 0.0);
    for (int l0 = 0; l0 <= layers; ++l0) {
        double bytes = 0;
        for (int l1 = l0; l1 <= layers; ++l1) {
            if (l1 > l0 && l1 - 1 < (int) c.stream_bytes.size()) bytes += c.stream_bytes[(size_t) (l1 - 1)];
            t[(size_t) l0 * (size_t) (layers + 1) + (size_t) l1] =
                c.link_gbps > 0 ? bytes / (kPsLinkEff * c.link_gbps * 1e9) * 1000.0 : 0.0;
        }
    }
    return t;
}

// one stage's chunk of `T` tokens around position `pos` over layers [l0, l1), in ms; `embeds`: the card embeds it;
// `link`: its ps_link_table
double ps_stage(const PsCard& c, const PsCard& main, int64_t T, double pos, int l0, int l1, bool embeds,
                const std::vector<double>& link, int layers) {
    const double scale = kPsRefGbps / ps_bw(c, main);
    const double us = (double) (l1 - l0) * kPsUsLayer * scale * (1.0 + kPsGrowth * pos) + (embeds ? kPsUsToken * scale : 0.0);
    return std::max((double) T * us / 1000.0, link[(size_t) l0 * (size_t) (layers + 1) + (size_t) l1]);
}

// `bytes` over a card's link, in ms
double ps_link_ms(const PsCard& c, double bytes) {
    return c.link_gbps > 0 ? bytes / (kPsLinkEff * c.link_gbps * 1e9) * 1000.0 : 0.0;
}

// The split of `part` over the main card `m` and the tier cards `st` (stage 1 on, in pipeline order), cut at `cuts`
// (stage k reads [cuts[k - 1], cuts[k] or `layers`)).  The main card reads chunk c's first layers and hands it on once
// stage 1 is done with the one before; each stage hands on once the next is done with the one before.  The main card
// keeps the last chunk and reads its later layers once every stage is done.
double ps_pipeline(const PsCard& m, const std::vector<const PsCard*>& st, const std::vector<const std::vector<double>*>& links,
                   const std::vector<double>& main_link, const int* cuts, const PsPart& part, int layers) {
    const size_t ns = st.size();
    double free_at[kPsMaxStages] = {};   // stage k + 1 takes a chunk from then on
    double t = 0;
    for (size_t k = 0; k < ns; ++k) t = std::max(t, st[k]->setup_ms);
    // the main card starts a chunk no sooner than the host has streamed every card's bytes of the one before
    double host_free = 0;
    auto host_ms = [&](const PsCard& c, double link_ms) {
        return part.host_gbps > 0 ? link_ms * kPsLinkEff * c.link_gbps / part.host_gbps : 0.0;
    };
    int64_t at = part.cell0;
    const size_t n = part.split_chunks.size();
    for (size_t i = 0; i < n; ++i) {
        const int64_t T = part.split_chunks[i];
        const double pos = (double) at + 0.5 * (double) T;
        const double a = ps_stage(m, m, T, pos, 0, cuts[0], true, main_link, layers) +
                         (i == 0 ? kPsPleUsToken * (double) T / 1000.0 : 0.0);
        const double hand = (double) T * (double) part.row_bytes;
        t = std::max(t, host_free);
        if (i + 1 < n && part.host_gbps > 0) {
            double shared = host_ms(m, main_link[(size_t) cuts[0]]);
            for (size_t k = 0; k < ns; ++k) {
                const int l1 = k + 1 < ns ? cuts[k + 1] : layers;
                shared += host_ms(*st[k], (*links[k])[(size_t) cuts[k] * (size_t) (layers + 1) + (size_t) l1] +
                                              ps_link_ms(*st[k], hand));
            }
            host_free = t + shared;
        }
        if (i + 1 < n) {
            double launch = std::max(t + a + ps_link_ms(m, hand), free_at[0]);
            t = launch;
            for (size_t k = 0; k < ns; ++k) {
                const int l1 = k + 1 < ns ? cuts[k + 1] : layers;
                const double done = launch + ps_link_ms(*st[k], hand) +
                                    ps_stage(*st[k], m, T, pos, cuts[k], l1, false, *links[k], layers);
                if (k + 1 < ns) {
                    launch = std::max(done + ps_link_ms(*st[k], hand), free_at[k + 1]);
                    free_at[k] = launch;
                } else {
                    free_at[k] = done;
                }
            }
        } else {
            t = std::max(t + a, free_at[ns - 1]) + (double) part.draft_rows * kPsDraftUsRow / 1000.0 +
                ps_stage(m, m, T, pos, cuts[0], layers, false, main_link, layers);
        }
        at += T;
    }
    return t;
}
}  // namespace

PsPrice price_prompt_split(const PsCard& main_tier, const PsCard& main_split, const std::vector<PsCard>& peers,
                           const PsPart& part, int max_stages, int layers) {
    PsPrice p;
    const std::vector<double> tier_link = ps_link_table(main_tier, layers);
    int64_t at = part.cell0;
    for (size_t i = 0; i < part.tier_chunks.size(); ++i) {
        const int64_t T = part.tier_chunks[i];
        p.tier_ms += ps_stage(main_tier, main_tier, T, (double) at + 0.5 * (double) T, 0, layers, true, tier_link, layers) +
                     (i == 0 ? kPsPleUsToken * (double) T / 1000.0 : 0.0);
        at += T;
    }
    const int s_max = std::min({max_stages, (int) peers.size() + 1, kPsMaxStages});
    p.stage_ms.assign((size_t) std::max(s_max + 1, 0), 0.0);
    p.stage_cuts.assign(p.stage_ms.size(), {});
    if (part.split_chunks.size() < 2 || s_max < 2) return p;
    const std::vector<double> main_link = ps_link_table(main_split, layers);
    std::vector<std::vector<double>> peer_link;
    for (const PsCard& c : peers) peer_link.push_back(ps_link_table(c, layers));
    for (int s = 2; s <= s_max; ++s) {
        // stage k on peers[s - 1 - k]
        std::vector<const PsCard*> st;
        std::vector<const std::vector<double>*> links;
        for (int k = 1; k < s; ++k) {
            st.push_back(&peers[(size_t) (s - 1 - k)]);
            links.push_back(&peer_link[(size_t) (s - 1 - k)]);
        }
        auto lo_of = [&](int k) { return std::max(2, st[(size_t) k]->layer_lo); };   // stage k + 1's first layer, at least
        auto hi_of = [&](int k) { return st[(size_t) k]->layer_hi > 0 ? std::min(st[(size_t) k]->layer_hi, layers) : layers; };
        if (hi_of(s - 2) < layers) continue;   // the last stage ends at the last layer
        // every cut in ascending order: cut k is stage k + 1's first layer, after cut k - 1 and before both cards' ends
        int cuts[kPsMaxStages] = {};
        double best = 0;
        bool found = false;
        std::vector<int> best_cuts;
        auto rec = [&](auto&& self, int k) -> void {
            if (k == s - 1) {
                const double t = ps_pipeline(main_split, st, links, main_link, cuts, part, layers);
                if (!found || t < best) {
                    best = t;
                    found = true;
                    best_cuts.assign(cuts, cuts + (s - 1));
                }
                return;
            }
            const int from = std::max(lo_of(k), k > 0 ? cuts[k - 1] + 1 : 2);
            // stage k (the main card's at k == 0) must end inside its card's layers; stage k + 1 needs a layer
            const int to = std::min(k > 0 ? hi_of(k - 1) : layers - 1, layers - (s - 1 - k));
            for (int c = from; c <= to; ++c) {
                cuts[k] = c;
                self(self, k + 1);
            }
        };
        rec(rec, 0);
        if (!found) continue;
        p.stage_ms[(size_t) s] = best;
        p.stage_cuts[(size_t) s] = best_cuts;
        if (p.stages == 0 || best < p.split_ms) {
            p.split_ms = best;
            p.stages = s;
            p.cuts = best_cuts;
        }
    }
    p.split = p.stages > 0 && p.split_ms < kPsMargin * p.tier_ms;
    return p;
}

std::vector<size_t> prompt_split_order(const std::vector<double>& mem_gbps) {
    std::vector<size_t> order(mem_gbps.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return mem_gbps[a] > mem_gbps[b]; });
    return order;
}

std::vector<std::pair<int, int>> prompt_split_windows(const PsPrice& ref, size_t n_peers, int margin, int layers) {
    std::vector<std::pair<int, int>> w(n_peers, {0, 0});
    for (size_t s = 2; s < ref.stage_ms.size(); ++s) {
        if (ref.stage_ms[s] <= 0 || !(ref.stage_ms[s] < kPsMargin * ref.tier_ms) || ref.stage_cuts[s].size() + 1 != s) continue;
        const std::vector<int>& c = ref.stage_cuts[s];
        for (size_t k = 1; k < s && s - 1 - k < n_peers; ++k) {
            const int lo = std::max(2, c[k - 1] - margin);
            const int hi = k + 1 < s ? std::min(layers, c[k] + margin) : layers;
            std::pair<int, int>& x = w[s - 1 - k];
            x = x.second == 0 ? std::make_pair(lo, hi) : std::make_pair(std::min(x.first, lo), std::max(x.second, hi));
        }
    }
    return w;
}

int64_t prompt_split_draft_room(const std::vector<int64_t>& chunks, int64_t chunk, int64_t hand_rows) {
    int64_t room = 2 * std::max<int64_t>(0, hand_rows - chunk);
    for (size_t i = chunks.size() >= 3 ? chunks.size() - 3 : 0; i + 1 < chunks.size(); ++i) room += chunks[i];
    return room;
}

}  // namespace basalt::program
