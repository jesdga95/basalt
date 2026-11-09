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
    static constexpr int kMaxT = 8;
    static constexpr int kBuckets = 4;

    explicit DraftPolicy(int max_t, double margin = 0.03);

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
    /// After such a window: the chances of its t - 1 drafts, the drafts accepted, and the round's time.
    void observe_window(const float* chance, int t, int accepted, double round_ms);

    double lookup_rate(int match) const;   // current q for a match length
    double chance_rate(float chance) const;   // current q for a draft stated at `chance`
    double cost_ms(int t) const;           // measured or scaled round time of a window of t tokens

private:
    static constexpr int kChanceBins = 9;
    static int bucket(int match);
    static int chance_bin(float chance);
    bool unmeasured(int t) const;          // a size measured too few times, or too long ago, to trust its cost
    bool recent(int t) const;              // measured within the last rounds
    double base(int t) const;              // a size's cost relative to the scale, measured or inferred
    double base_from(int t, bool lately) const;   // t's base from the other (recent) sizes; < 0 without any
    double mtp_tokens(int t) const;

    int max_t_;
    double margin_;
    double scale_ = 0.0;                                   // round ms = scale x base(size)
    std::array<double, kMaxT + 1> base_{}, base_n_{};      // by window size; rounds behind it
    std::array<long, kMaxT + 1> seen_{};                   // the round each size was last measured
    long round_ = 0;
    std::array<double, kMaxT + 1> mtp_tok_{}, mtp_n_{};    // tokens committed by MTP windows of that size
    std::array<double, kBuckets> ok_{}, bad_{};            // lookup drafts accepted / windows cut short, decayed
    std::array<double, kChanceBins> c_ok_{}, c_n_{};       // drafter drafts accepted / tested, by stated chance, decayed
};

}  // namespace basalt::spec
