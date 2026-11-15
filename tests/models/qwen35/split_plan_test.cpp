// tests/models/qwen35/split_plan_test.cpp - dense::plan_split on Qwen3.8-27B's bytes: the main card alone when the model
// fits it, else the cut that puts the most layers on the faster of two unlike cards or halves the layers' bytes between alike
// ones, an error when neither fits, and a forced cut taken as given when it fits.  The bytes are what the 27B pack (NVFP4 layers,
// FP8 head, block drafter) and dense::split_inputs give for each setting; the free memory is what the cards report at start.
#include "basalt/dense/split_plan.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using basalt::dense::SplitCard;
using basalt::dense::SplitInputs;
using basalt::dense::SplitPlan;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

constexpr uint64_t kGdnLayer = 216474520, kAttnLayer = 209430556;    // weights: a GDN layer, an attention layer
constexpr uint64_t kGdnSlot = 7854080;                               // a slot's recurrent state in a GDN layer
constexpr uint64_t kMainFixed = 5918443012, kMainFixed8K = 5582898692;   // head, drafter, scratch, a slot's logits (spec 16)
constexpr uint64_t kSlotRest = 20000000;                              // another slot's logits and drafter ring
constexpr uint64_t kPeerFixed = 2165045508;                           // the second card's scratch and landing buffers
constexpr uint64_t kMargin = 512ull << 20;
constexpr uint64_t k5090 = 32622837760, k16GB = 15979839488;          // free at start: RTX 5090, a 16 GB card

SplitCard card(uint64_t free, double gbps) { return {free - kMargin, gbps}; }

// every fourth layer attention; `attn_slot`: a slot's cells in an attention layer at the setting's context and KV format
SplitInputs inputs(uint64_t attn_slot, int slots, uint64_t main_fixed, SplitCard main, SplitCard peer, int forced = 0) {
    SplitInputs in;
    for (int l = 0; l < 64; ++l) {
        const bool attn = l % 4 == 3;
        in.layer.push_back(attn ? kAttnLayer : kGdnLayer);
        in.per_slot.push_back(attn ? attn_slot : kGdnSlot);
    }
    in.slots = slots;
    in.main_fixed = main_fixed + (uint64_t) (slots - 1) * kSlotRest;
    in.peer_fixed = kPeerFixed;
    in.main = main;
    in.peer = peer;
    in.forced = forced;
    return in;
}

void expect(const char* what, const SplitInputs& in, int cut) {
    const SplitPlan p = basalt::dense::plan_split(in);
    const bool fits = p.error.empty() && p.main_bytes <= in.main.free && p.peer_bytes <= in.peer.free;
    check(p.cut == cut && fits, std::string(what) + ": cut " + std::to_string(p.cut) + " (want " + std::to_string(cut) + ")" +
                                    (p.error.empty() ? "" : ", " + p.error));
}

void expect_error(const char* what, const SplitInputs& in) {
    const SplitPlan p = basalt::dense::plan_split(in);
    check(!p.error.empty() && p.cut == 0, std::string(what) + ": refused (" + p.error + ")");
}

}  // namespace

int main() {
    constexpr uint64_t fp16_8k = 33555456, fp16_262k = 1073742848, int8_128k = 276825088, int8_262k = 553649152;
    const SplitCard r5090 = card(k5090, 1792), r5060 = card(k16GB, 448), r5070 = card(k16GB, 896);
    // RTX 5090 + RTX 5060 Ti: unlike cards, the most layers on the 5090
    expect("32+16, 8K fp16 (fits the main card)", inputs(fp16_8k, 1, kMainFixed8K, r5090, r5060), 0);
    expect("32+16, 262K fp16", inputs(fp16_262k, 1, kMainFixed, r5090, r5060), 55);
    expect("32+16, 262K int8, two slots", inputs(int8_262k, 2, kMainFixed, r5090, r5060), 51);
    // a slower main card than the second (listed first): the most layers on the second card
    {
        const SplitPlan p = basalt::dense::plan_split(inputs(fp16_262k, 1, kMainFixed, card(k5090, 448), card(k5090, 1792)));
        check(p.error.empty() && p.cut > 0 && p.cut < 20, "a slower main card keeps the fewest layers that fit it: cut " + std::to_string(p.cut));
    }
    // two RTX 5090s: alike, the layers' bytes halved
    expect("32+32, 262K fp16, two slots", inputs(fp16_262k, 2, kMainFixed, r5090, r5090), 32);
    // two 16 GB cards: the weights alone do not fit one
    expect("16+16, 8K fp16", inputs(fp16_8k, 1, kMainFixed8K, r5070, r5070), 32);
    expect("two RTX 5070 Ti, 128K int8, spec 16 (#376)", inputs(int8_128k, 1, kMainFixed, r5070, r5070), 32);
    expect_error("16+16, 262K fp16 (fits neither one card nor a split)", inputs(fp16_262k, 1, kMainFixed, r5070, r5070));
    // the diagnostic cut
    expect("32+16, 8K, forced at 40", inputs(fp16_8k, 1, kMainFixed8K, r5090, r5060, 40), 40);
    expect("32+16, 8K, forced at 8", inputs(fp16_8k, 1, kMainFixed8K, r5090, r5060, 8), 8);
    expect_error("32+16, 8K, forced at 4 (the second card cannot hold the rest)", inputs(fp16_8k, 1, kMainFixed8K, r5090, r5060, 4));
    expect_error("forced at 64 (no layer left for the second card)", inputs(fp16_8k, 1, kMainFixed8K, r5090, r5060, 64));
    check(basalt::dense::split_cards_alike({0, 896}, {0, 960}) && !basalt::dense::split_cards_alike({0, 1792}, {0, 448}),
          "cards within 10% of each other's bandwidth are alike");
    std::printf("%s\n", failures ? "FAILED" : "all ok");
    return failures ? 1 : 0;
}
