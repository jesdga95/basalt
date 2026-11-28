// include/basalt/spec/draft_policy.hpp - per verify round: the MTP's window, or a lookup (suffix) window? And for a
// drafter that states each draft's chance, how many of its drafts?
//
// The suffix drafter (prompt lookup) proposes the tokens that followed an earlier repeat of the context. Taken
// whenever it proposes more than the MTP, it lost 2-8% on ordinary text: a long lookup window costs much more to
// verify than the MTP's usual 3-4 tokens, and it is only worth that when enough of it is accepted. llama.cpp's
// lookup decoding answers the same problem with confidence thresholds on its n-gram statistics; here the policy
// learns both sides online and compares expected committed tokens per millisecond:
//
//   MTP     E = the mean tokens a window of that size has committed (EMA), at the measured cost of that size
//   lookup  E(k) = 1 + q + q^2 + ... + q^k for k <= the proposal, q = the acceptance rate of lookup drafts whose
//           match was about as long (4 buckets of match length, decayed counts), at the measured cost of k + 1
//
// and takes the lookup window only when its best E/cost beats the MTP's by `margin`. Costs are the measured round
// times: a scale that follows every round times a base per window size that follows that size's rounds (sizes not
// measured lately are inferred from the ones that were, or scaled by a prior shape), so the policy adapts to the
// machine and the context length, and the costs of sizes not taken lately move with the ones taken. It only chooses which drafts to verify: the output is unchanged.
//
// A drafter that makes all its drafts in one pass and states each one's chance of acceptance (a block drafter with a
// confidence head) lets the window be chosen after the drafts are known, from the same costs:
//
//   drafter E(t) = 1 + q(c_0) + q(c_0) q(c_1) + ... (t - 1 terms), q = the acceptance rate of drafts whose stated chance
//           was about c (decayed counts per chance bin, the stated chance as the prior), at the measured cost of t
//
// and the window is the t with the most E(t) per millisecond. A row costs what it costs on this machine (more distinct
// experts on the CPU with every row where RAM holds many), so a window grows only where its drafts pay.
#pragma once

#include <array>

namespace basalt::spec {

class DraftPolicy {
public:
    static constexpr int kMaxT = 16;
    static constexpr int kBuckets = 4;
    static constexpr int kRegimes = 3;      // 0: greedy/low temp (<=0.35), 1: medium (<=0.75), 2: high-temp stochastic (>0.75)
    static constexpr int kPosGroups = 3;    // 0: draft 0, 1: draft 1, 2: draft >= 2

    /// `timed` false: round times are not read, every round costs the prior shape of its size, so the picks follow the
    /// drafts' outcomes alone (the same tokens give the same windows on every run).
    explicit DraftPolicy(int max_t, double margin = 0.03, bool timed = true);
    /// Selects the temperature/sampling calibration regime (0=greedy/low, 1=medium, 2=high-temp stochastic).
    void set_regime(int regime);
    int regime() const { return regime_; }

    struct Pick {
        bool lookup = false;
        int t = 1;                      // window size (1 + drafts)
    };
    /// `t_mtp`: the MTP's window; `lookup_k`: the lookup proposal's length (0 = none); `match`: its match length.
    Pick choose(int t_mtp, int lookup_k, int match) const;
    /// After the round: the window it used, the drafts accepted, and the round's time (verify + commit + draft).
    void observe(bool lookup, int t, int accepted, int match, double round_ms);

    /// A drafter with stated chances: the window (1 + up to `n` drafts) with the most expected tokens per ms.
    int window(const float* chance, int n) const;
    /// Whether window(chance, n) can hold draft `j` (from 0) when only the chances of drafts 0..j are known: the later
    /// ones are taken at the best rate a stated chance can have. False means every window of those drafts leaves draft j
    /// out, so a drafter that makes its drafts in turn can stop before it.
    bool may_verify(const float* chance, int j, int n) const;
    /// The same drafter's window against a lookup proposal of `lookup_k` tokens behind a match of `match`: the lookup
    /// window when its expected tokens per ms beat the drafter's best by the margin.
    Pick choose_window(const float* chance, int n, int lookup_k, int match) const;
    /// After such a window: the chances of its t - 1 drafts, the drafts accepted, and the round's time.
    void observe_window(const float* chance, int t, int accepted, double round_ms);
    /// A window whose time is not its own (a batch's): only what its drafts' chances turned out to be worth.
    void observe_chances(const float* chance, int t, int accepted);

    double lookup_rate(int match) const;   // current q for a match length
    double chance_rate(float chance, int pos = 0) const;   // current q for a draft stated at `chance` at draft position `pos`
    double cost_ms(int t) const;           // measured or scaled round time of a window of t tokens

private:
    static constexpr int kChanceBins = 9;
    static int bucket(int match);
    static int chance_bin(float chance);
    static int pos_group(int pos);
    double chance_rate_bin(int b, double c, int pos) const;
    double best_rate(int pos = 0) const;   // the highest q any stated chance gets at draft position `pos`
    bool unmeasured(int t) const;          // a size measured too few times, or too long ago, to trust its cost
    bool recent(int t) const;              // measured within the last rounds
    double base(int t) const;              // a size's cost relative to the scale, measured or inferred
    double base_from(int t, bool lately) const;   // t's base from the other (recent) sizes; < 0 without any
    double mtp_tokens(int t) const;
    Pick lookup_over(int t, double rate, int lookup_k, int match) const;   // the lookup window, if it beats `rate` at t
    double drafter_tokens(const float* chance, int t) const;               // expected tokens of a drafter window of t

    int max_t_;
    double margin_;
    bool timed_;
    int regime_ = 0;
    double scale_ = 0.0;                                   // round ms = scale x base(size)
    std::array<double, kMaxT + 1> base_{}, base_n_{};      // by window size; rounds behind it
    std::array<long, kMaxT + 1> seen_{};                   // the round each size was last measured
    long round_ = 0;
    std::array<double, kMaxT + 1> mtp_tok_{}, mtp_n_{};    // tokens committed by MTP windows of that size
    std::array<double, kBuckets> ok_{}, bad_{};            // lookup drafts accepted / windows cut short, decayed
    std::array<std::array<double, kChanceBins>, kRegimes> p_ok_{}, p_n_{};                     // pooled per (regime, bin)
    std::array<std::array<std::array<double, kChanceBins>, kPosGroups>, kRegimes> c_ok_{}, c_n_{};  // per (regime, pos_group, bin)
};

}  // namespace basalt::spec
