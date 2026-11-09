// src/spec/draft_policy.cpp - see include/basalt/spec/draft_policy.hpp.
#include "basalt/spec/draft_policy.hpp"

#include <algorithm>

namespace basalt::spec {
namespace {

// The shape of a round's cost by window size, relative to one token, used only below the smallest measured size or
// while a single size is measured (two measured sizes give the cost of a row; measured round times replace both). It
// sits between the measured curves: steeper with every missed expert on the CPU, flatter with the default CPU/DMA split.
constexpr double kShape[DraftPolicy::kMaxT + 1] = {0.0, 1.0, 1.35, 1.7, 2.05, 2.45, 2.85, 3.25, 3.6};
// A round's cost is scale x base(size): the scale follows every round fast (the context grows, the tiers move, a
// request's first windows run cold), each size's base follows its own rounds slowly, so the costs of sizes not taken
// lately move with the ones taken
constexpr double kScaleAlpha = 0.2;
constexpr double kCostAlpha = 0.1;    // EMA weight of a size's new base
// A size's first rounds start from the base the other sizes imply, worth this many rounds: one cold round (a
// request's first windows run with the tiers still settling) cannot rule a size out
constexpr double kBasePrior = 3.0;
// Rounds after which a size's base is from another time (a cold start's, a past fixture's): the sizes measured since
// stand in for it until it is measured again, so a size measured dear once is not ruled out for good
constexpr long kStale = 64;
constexpr double kTokAlpha = 0.05;    // EMA weight of a new MTP window outcome
constexpr double kDecay = 0.97;       // lookup counts: older windows fade
// Before a bucket has data: the longer the match, the likelier its continuation (llama.cpp's lookup decoding gates
// on the same thing); worth 4 observations, so a few real windows override it
constexpr double kPriorQ[DraftPolicy::kBuckets] = {0.75, 0.88, 0.93, 0.96};
constexpr double kPriorN = 4.0;
constexpr int kProbes = 3;       // a lookup window size is tried this often before its guessed cost can veto it
// Stated chances: the bins' upper edges. A confidence head is best near 1, where long windows are decided, so the bins
// are finest there. Each bin starts from the stated chance, worth kChancePriorN tested drafts.
constexpr float kChanceEdge[] = {0.3f, 0.5f, 0.7f, 0.8f, 0.9f, 0.95f, 0.98f, 0.995f};
constexpr double kChanceDecay = 0.98;
constexpr double kChancePriorN = 10.0;
constexpr double kProbeChance = 0.85;   // the joint chance at which a drafter window one row past the best is probed

}  // namespace

DraftPolicy::DraftPolicy(int max_t, double margin)
    : max_t_(std::clamp(max_t, 1, kMaxT)), margin_(margin) {}

int DraftPolicy::bucket(int match) {
    return match < 6 ? 0 : match < 12 ? 1 : match < 24 ? 2 : 3;
}

// measured fewer than kProbes times since it was new, or not for a long while
bool DraftPolicy::unmeasured(int t) const { return base_n_[t] < kBasePrior + kProbes || !recent(t); }

int DraftPolicy::chance_bin(float chance) {
    int b = 0;
    while (b < kChanceBins - 1 && chance >= kChanceEdge[b]) ++b;
    return b;
}

double DraftPolicy::chance_rate(float chance) const {
    const int b = chance_bin(chance);
    const double c = std::clamp((double) chance, 0.0, 1.0);
    return (c_ok_[b] + kChancePriorN * c) / (c_n_[b] + kChancePriorN);
}

double DraftPolicy::lookup_rate(int match) const {
    const int b = bucket(match);
    return (ok_[b] + kPriorN * kPriorQ[b]) / (ok_[b] + bad_[b] + kPriorN);
}

bool DraftPolicy::recent(int t) const { return base_n_[t] > 0 && round_ - seen_[t] <= kStale; }

double DraftPolicy::base(int t) const {
    if (recent(t)) return base_[t];
    // sizes measured lately (else any measured) stand in for t: interpolated, extrapolated by the cost of a row, or
    // scaled by the prior shape
    for (const bool lately : {true, false}) {
        const double b = base_from(t, lately);
        if (b > 0) return b;
    }
    return kShape[t];
}

double DraftPolicy::base_from(int t, bool lately) const {
    // the measured sizes next to t: between two, the line through them; above the largest, its line on from the two
    // largest (the cost of a row on this machine); below the smallest, or with one, the prior shape scales it
    int lo = 0, hi = 0, below = 0;
    for (int u = 1; u <= kMaxT; ++u) {
        if (u == t || base_n_[u] <= 0 || (lately && !recent(u))) continue;
        if (u < t) { below = lo; lo = u; }
        else if (hi == 0) hi = u;
    }
    if (lo > 0 && hi > 0) return base_[lo] + (base_[hi] - base_[lo]) * (double) (t - lo) / (double) (hi - lo);
    if (lo > 0 && below > 0)
        return base_[lo] + std::max(0.0, (base_[lo] - base_[below]) / (double) (lo - below)) * (double) (t - lo);
    if (lo > 0) return base_[lo] * kShape[t] / kShape[lo];
    if (hi > 0) return base_[hi] * kShape[t] / kShape[hi];
    return -1.0;
}

double DraftPolicy::cost_ms(int t) const {
    t = std::clamp(t, 1, kMaxT);
    return (scale_ > 0 ? scale_ : 1.0) * base(t);       // before any round: the prior shape
}

double DraftPolicy::mtp_tokens(int t) const {
    if (mtp_n_[t] > 0) return mtp_tok_[t];
    return 1.0 + 0.7 * (t - 1);       // before any MTP window of this size: a typical acceptance
}

DraftPolicy::Pick DraftPolicy::choose(int t_mtp, int lookup_k, int match) const {
    Pick p;
    p.t = std::clamp(t_mtp, 1, max_t_);
    if (lookup_k <= 0) return p;
    const double base = mtp_tokens(p.t) / cost_ms(p.t);
    const double q = lookup_rate(match);
    double e = 1.0, qi = 1.0, best = 0.0;
    int best_t = 0;
    for (int k = 1; k <= std::min(lookup_k, max_t_ - 1); ++k) {
        qi *= q;
        e += qi;
        const double r = e / cost_ms(k + 1);
        if (r > best) { best = r; best_t = k + 1; }
    }
    if (best_t > 0 && best > base * (1.0 + margin_)) {
        p.lookup = true;
        p.t = best_t;
        return p;
    }
    // a guessed cost can keep the policy from ever measuring a size: the first few times a confident lookup would
    // need a size not measured yet (or not for long), it is tried (verification keeps the output; only the one
    // round's speed is at stake)
    const int t_full = std::min(lookup_k, max_t_ - 1) + 1;
    if (t_full > p.t && unmeasured(t_full) && q >= 0.85) {
        p.lookup = true;
        p.t = t_full;
    }
    return p;
}

int DraftPolicy::window(const float* chance, int n) const {
    const int most = std::clamp(n + 1, 1, max_t_);
    double e = 1.0, all = 1.0, best = 1.0 / cost_ms(1);
    int best_t = 1;
    for (int t = 2; t <= most; ++t) {
        all *= chance_rate(chance[t - 2]);
        e += all;
        const double r = e / cost_ms(t);
        if (r > best) { best = r; best_t = t; }
    }
    // as for lookup windows: a size whose cost is only guessed (or old) is tried a few times when its drafts are
    // likely accepted, so a guessed or stale cost cannot keep the window from ever growing
    if (best_t < most && unmeasured(best_t + 1)) {
        double joint = 1.0;
        for (int i = 0; i < best_t; ++i) joint *= chance_rate(chance[i]);
        if (joint >= kProbeChance) ++best_t;
    }
    return best_t;
}

void DraftPolicy::observe_window(const float* chance, int t, int accepted, double round_ms) {
    // drafts 0..accepted-1 were accepted given their prefix and draft `accepted` (if the window had it) was not; the
    // drafts after it were never tested
    const int tested = std::min(t - 1, accepted + 1);
    for (int i = 0; i < tested; ++i) {
        const int b = chance_bin(chance[i]);
        c_ok_[b] = kChanceDecay * c_ok_[b] + (i < accepted ? 1.0 : 0.0);
        c_n_[b] = kChanceDecay * c_n_[b] + 1.0;
    }
    observe(false, t, accepted, 0, round_ms);
}

void DraftPolicy::observe(bool lookup, int t, int accepted, int match, double round_ms) {
    t = std::clamp(t, 1, kMaxT);
    ++round_;
    if (round_ms > 0) {
        if (scale_ <= 0) scale_ = round_ms / kShape[t];          // the first round sets the scale
        if (!recent(t)) {                                        // new, or measured too long ago: start over from
            base_[t] = base(t);                                  // what the other sizes imply
            base_n_[t] = kBasePrior;
        }
        scale_ += kScaleAlpha * (round_ms / base_[t] - scale_);
        const double w = std::max(kCostAlpha, 1.0 / (base_n_[t] + 1.0));
        base_[t] = (1.0 - w) * base_[t] + w * round_ms / scale_;
        base_n_[t] += 1.0;
        seen_[t] = round_;
    }
    if (lookup) {
        const int b = bucket(match);
        ok_[b] = kDecay * ok_[b] + accepted;
        bad_[b] = kDecay * bad_[b] + (accepted < t - 1 ? 1.0 : 0.0);
    } else {
        const double got = accepted + 1.0;
        mtp_tok_[t] = mtp_n_[t] > 0 ? (1.0 - kTokAlpha) * mtp_tok_[t] + kTokAlpha * got : got;
        mtp_n_[t] += 1.0;
    }
}

}  // namespace basalt::spec
