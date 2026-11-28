// tests/models/qwen35/split_plan_test.cpp - a two-card split of Qwen3.8-27B by the shares (model::split_cut) on its bytes:
// the cut follows the cards' free memory, or gpu-share, or --diag-gpu-cut; a card whose part does not fit is refused with
// its numbers.  The bytes are what the 27B pack (NVFP4 layers, FP8 head, block drafter) and dense::split_spec give for each
// setting; the free memory is what the cards report at start.
#include "basalt/model/multi_gpu.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using basalt::model::GpuCards;
using basalt::model::SplitCut;
using basalt::model::SplitSpec;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

constexpr uint64_t kGdnLayer = 216476160, kAttnLayer = 209432320;    // weights: a GDN layer, an attention layer (256-byte aligned)
constexpr uint64_t kGdnSlot = 7854080;                               // a slot's recurrent state in a GDN layer
constexpr uint64_t kMainFixed = 5918443012, kMainFixed8K = 5582898692;   // head, drafter, scratch, a slot's logits (spec 16)
constexpr uint64_t kSlotRest = 20000000;                              // another slot's logits and drafter ring
constexpr uint64_t kPeerFixed = 2165045508;                           // the second card's scratch and landing buffers
constexpr uint64_t kMargin = 512ull << 20;
constexpr uint64_t k5090 = 32622837760, k16GB = 15979839488;          // free at start: RTX 5090, a 16 GB card

// two cards with this much free at start (the share's default) and the room past the margin the 27B keeps
GpuCards cards(uint64_t free0, uint64_t free1, std::vector<double> given = {}, int diag = 0) {
    GpuCards c;
    c.mode = basalt::program::GpuMode::Split;
    c.given = given;
    c.diag_cut = diag;
    for (int i = 0; i < 2; ++i) {
        GpuCards::Card k;
        k.device = i;
        k.index = i;
        k.label = "GPU " + std::to_string(i);
        k.free = i == 0 ? free0 : free1;
        c.cards.push_back(k);
    }
    return c;
}

// every fourth layer attention; `attn_slot`: a slot's cells in an attention layer at the setting's context and KV format
SplitSpec spec(uint64_t attn_slot, int slots, uint64_t main_fixed, uint64_t free0, uint64_t free1) {
    SplitSpec s;
    for (int l = 0; l < 64; ++l) {
        const bool attn = l % 4 == 3;
        s.layer.push_back((attn ? kAttnLayer : kGdnLayer) + (uint64_t) slots * (attn ? attn_slot : kGdnSlot));
    }
    s.fixed = {main_fixed + (uint64_t) (slots - 1) * kSlotRest, kPeerFixed};
    s.room = {free0 - kMargin, free1 - kMargin};
    return s;
}

void expect(const char* what, const GpuCards& c, const SplitSpec& s, int lo, int hi) {
    const SplitCut cut = basalt::model::split_cut(c, s);
    const int at = cut.first.size() > 1 ? cut.first[1] : 0;
    check(cut.error.empty() && at >= lo && at <= hi, std::string(what) + ": cut " + std::to_string(at) + " (want " + std::to_string(lo) +
                                                         "-" + std::to_string(hi) + ")" + (cut.error.empty() ? "" : ", " + cut.error));
}

void expect_error(const char* what, const GpuCards& c, const SplitSpec& s, const std::string& start) {
    const SplitCut cut = basalt::model::split_cut(c, s);
    check(cut.error.rfind(start, 0) == 0, std::string(what) + ": refused (" + cut.error + ")");
}

}  // namespace

int main() {
    constexpr uint64_t fp16_8k = 33555456, fp16_262k = 1073742848, int8_128k = 276825088, int8_262k = 553649152;
    // A big card + a small one at 8K: the share of free memory (0.67 / 0.33) gives the second card a third of the bytes
    expect("32+16, 8K fp16", cards(k5090, k16GB), spec(fp16_8k, 1, kMainFixed8K, k5090, k16GB), 40, 46);
    // gpu-share moves it: 3,1 gives the second card a quarter, 3,2 two fifths
    expect("32+16, 8K fp16, gpu-share 3,1", cards(k5090, k16GB, {0.75, 0.25}), spec(fp16_8k, 1, kMainFixed8K, k5090, k16GB), 47, 51);
    expect("32+16, 8K fp16, gpu-share 3,2", cards(k5090, k16GB, {0.6, 0.4}), spec(fp16_8k, 1, kMainFixed8K, k5090, k16GB), 34, 40);
    // 262K: the cells follow their layers, the cut by the same shares
    expect("32+16, 262K fp16", cards(k5090, k16GB), spec(fp16_262k, 1, kMainFixed, k5090, k16GB), 36, 46);
    expect("32+16, 262K int8, two slots", cards(k5090, k16GB), spec(int8_262k, 2, kMainFixed, k5090, k16GB), 36, 46);
    // two RTX 5090s: half the bytes each
    expect("32+32, 262K fp16, two slots", cards(k5090, k5090), spec(fp16_262k, 2, kMainFixed, k5090, k5090), 26, 34);
    // two 16 GB cards: the weights alone do not fit one; the main card's head and drafter count in its half
    expect("16+16, 8K fp16", cards(k16GB, k16GB), spec(fp16_8k, 1, kMainFixed8K, k16GB, k16GB), 22, 28);
    expect("two RTX 5070 Ti, 128K int8, spec 16 (#376)", cards(k16GB, k16GB), spec(int8_128k, 1, kMainFixed, k16GB, k16GB), 20, 30);
    expect_error("16+16, 262K fp16 (does not fit)", cards(k16GB, k16GB), spec(fp16_262k, 1, kMainFixed, k16GB, k16GB),
                 "gpu-mode split: GPU ");
    // the diagnostic cut, taken as given
    expect("32+16, 8K, --diag-gpu-cut 40", cards(k5090, k16GB, {}, 40), spec(fp16_8k, 1, kMainFixed8K, k5090, k16GB), 40, 40);
    expect("32+16, 8K, --diag-gpu-cut 8", cards(k5090, k16GB, {}, 8), spec(fp16_8k, 1, kMainFixed8K, k5090, k16GB), 8, 8);
    expect_error("32+16, 8K, --diag-gpu-cut 4 (the second card cannot hold the rest)", cards(k5090, k16GB, {}, 4),
                 spec(fp16_8k, 1, kMainFixed8K, k5090, k16GB), "gpu-mode split: GPU 1 takes layers 4-63 at --diag-gpu-cut 4 and needs");
    expect_error("--diag-gpu-cut 64 (no layer left for the second card)", cards(k5090, k16GB, {}, 64),
                 spec(fp16_8k, 1, kMainFixed8K, k5090, k16GB), "--diag-gpu-cut 64: expected a layer from 1 to 63");
    std::printf("%s\n", failures ? "FAILED" : "all ok");
    return failures ? 1 : 0;
}
