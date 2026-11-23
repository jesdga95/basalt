// tests/model/multi_gpu_test.cpp - the shared multi-GPU layer over fake card tables: the shares and their defaults, a split's
// cut from the shares, the expert deal and the prompt stages of a peer, each refusal's text and the start line.
#include "basalt/model/multi_gpu.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace basalt::model;
using basalt::program::GpuMode;

static int g_fail = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)
#define CHECK_EQ(a, b)                                                                                   \
    do {                                                                                                 \
        if (!((a) == (b))) {                                                                             \
            std::printf("FAIL %s:%d: %s\n  got  %s\n  want %s\n", __FILE__, __LINE__, #a, std::string(a).c_str(), \
                        std::string(b).c_str());                                                         \
            ++g_fail;                                                                                    \
        }                                                                                                \
    } while (0)

static constexpr uint64_t GiB = 1ull << 30;

// an RTX 5090 (main) and an RTX 5060 Ti, as nvidia-smi 0 and 1, with their free memory at start
static GpuCards two(GpuMode mode, uint64_t free0 = 31 * GiB, uint64_t free1 = 15 * GiB) {
    GpuCards c;
    c.mode = mode;
    const char* names[2] = {"RTX 5090", "RTX 5060 Ti"};
    const double gbps[2] = {1792, 448};
    const uint64_t free[2] = {free0, free1};
    for (int i = 0; i < 2; ++i) {
        GpuCards::Card k;
        k.device = i;
        k.index = i;
        k.info.smi = i;
        k.info.name = names[i];
        k.info.mem_gbps = gbps[i];
        k.label = std::string("GPU ") + std::to_string(i) + " (" + names[i] + ")";
        k.free = free[i];
        c.cards.push_back(k);
    }
    return c;
}

static bool near(double a, double b) { return a > b - 1e-9 && a < b + 1e-9; }

int main() {
    // ---- shares: gpu-share when given, else free VRAM for layers and experts, bandwidth for prompt stages
    {
        GpuCards c = two(GpuMode::Peer, 30 * GiB, 10 * GiB);
        Shares s = shares_for(c, ShareUse::Split);
        CHECK(s.from == "free VRAM" && near(s.of[0], 0.75) && near(s.of[1], 0.25));
        s = shares_for(c, ShareUse::Stages);
        CHECK(s.from == "bandwidth" && near(s.of[0], 0.8) && near(s.of[1], 0.2));
        c.given = {0.6, 0.4};
        s = shares_for(c, ShareUse::Stages);
        CHECK(s.from == "gpu-share" && near(s.of[1], 0.4));
        c.given.clear();
        c.cards[1].info.mem_gbps = 0;   // a figure unknown: equal
        s = shares_for(c, ShareUse::Stages);
        CHECK(s.from == "equal" && near(s.of[0], 0.5));
    }
    // the peers' shares among themselves: three cards, the main card's left out
    {
        GpuCards c = two(GpuMode::Peer, 30 * GiB, 10 * GiB);
        c.cards.push_back(c.cards[1]);
        c.cards[2].free = 30 * GiB;
        const std::vector<double> p = peer_shares(c, ShareUse::Experts);
        CHECK(p.size() == 2 && near(p[0], 0.25) && near(p[1], 0.75));
    }

    // ---- split: 48 even layers of 0.5 GiB, 2 GiB fixed on the main card and 1 GiB on the second
    SplitSpec spec;
    spec.layer.assign(48, GiB / 2);
    spec.fixed = {2 * GiB, 1 * GiB};
    spec.room = {30 * GiB, 14 * GiB};
    {
        // free VRAM 31 + 15 GiB: shares 0.674 / 0.326 of 27 GiB; the main card's 2 GiB + 32 layers is nearest 18.2 GiB
        const SplitCut cut = split_cut(two(GpuMode::Split), spec);
        CHECK(cut.error.empty());
        CHECK(cut.first.size() == 2 && cut.first[0] == 0 && cut.first[1] == 32);
        CHECK(cut.need[0] == 18 * GiB && cut.need[1] == 9 * GiB);
        CHECK_EQ(cut_layers(cut, 1, 48), "layers 32-47");
    }
    {
        // gpu-share moves it: 1,1 halves the bytes, 3,1 gives the second card a quarter
        GpuCards c = two(GpuMode::Split);
        c.given = {0.5, 0.5};
        CHECK(split_cut(c, spec).first[1] == 23);   // half of 27 GiB: the main card's 2 GiB and 23 layers
        c.given = {0.75, 0.25};
        CHECK(split_cut(c, spec).first[1] == 36);
    }
    {
        // --diag-gpu-cut takes the cut as given, and out of range is refused
        GpuCards c = two(GpuMode::Split);
        c.diag_cut = 40;
        CHECK(split_cut(c, spec).first[1] == 40);
        c.diag_cut = 48;
        CHECK_EQ(split_cut(c, spec).error, "--diag-gpu-cut 48: expected a layer from 1 to 47 with two cards");
    }
    {
        // a card whose share is less than a layer would hold nothing
        GpuCards c = two(GpuMode::Split);
        c.given = {0.99, 0.01};
        CHECK_EQ(split_cut(c, spec).error, "gpu-mode split: GPU 1 (RTX 5060 Ti) would hold nothing: its share 0.01 (gpu-share) is "
                                           "less than a layer; raise its gpu-share or leave it out of gpu");
        c.given = {0.001, 0.999};
        CHECK_EQ(split_cut(c, spec).error, "gpu-mode split: GPU 0 (RTX 5090) would hold nothing: its share 0.00 (gpu-share) is "
                                           "less than a layer; raise its gpu-share or leave it out of gpu");
    }
    {
        // a part that does not fit its card is refused with its numbers
        GpuCards c = two(GpuMode::Split);
        c.given = {0.4, 0.6};
        CHECK_EQ(split_cut(c, spec).error, "gpu-mode split: GPU 1 (RTX 5060 Ti) takes layers 18-47 at share 0.60 and needs 16.0 GiB, "
                                           "14.0 GiB free; lower its gpu-share, max-context or parallel");
    }
    {
        // the RTX 5090 + RTX 5060 Ti case that used to put one layer on the second card: with free-VRAM shares a third
        SplitSpec s = spec;
        s.layer.assign(24, GiB);
        s.fixed = {4 * GiB, 2 * GiB};
        s.room = {30 * GiB, 14 * GiB};
        const SplitCut cut = split_cut(two(GpuMode::Split), s);
        CHECK(cut.error.empty() && cut.first[1] > 12 && cut.first[1] < 23);
    }

    // ---- peer: the expert deal by share
    {
        // one peer takes what fits its room, hottest first, whatever its share
        const std::vector<uint64_t> bytes(10, 100);
        std::vector<int> owner = deal_experts(bytes, {450}, {1.0});
        CHECK(owner == std::vector<int>({0, 0, 0, 0, -1, -1, -1, -1, -1, -1}));
        // two peers at 3:1 take three for one; a full peer closes and the other keeps taking
        owner = deal_experts(bytes, {10000, 10000}, {0.75, 0.25});
        int a = 0;
        for (int i = 0; i < 8; ++i) a += owner[(size_t) i] == 0;
        CHECK(a == 6);
        owner = deal_experts(bytes, {200, 10000}, {0.75, 0.25});
        CHECK(owner[0] == 0 && owner[2] == 0 && owner[3] == 1 && owner[9] == 1);
        auto owners = [](const std::vector<int>& o) {
            std::string s;
            for (int x : o) s += x < 0 ? std::string("-") : std::to_string(x);
            return s;
        };
        const std::vector<uint64_t> eq(8, 10);
        CHECK_EQ(owners(deal_experts(eq, {40, 40}, {0.5, 0.5})), "01010101");
        CHECK_EQ(owners(deal_experts(eq, {80, 80}, {2.0 / 3, 1.0 / 3})), "01001001");
        CHECK_EQ(owners(deal_experts(eq, {20, 100}, {0.8, 0.2})), "01011111");
        CHECK_EQ(owners(deal_experts(eq, {0, 30}, {0.5, 0.5})), "111-----");
        CHECK_EQ(owners(deal_experts(eq, {0, 0, 0}, {0.3, 0.3, 0.4})), "--------");
    }
    {
        // the refusal of a peer that would hold nothing
        GpuCards c = two(GpuMode::Peer);
        CHECK_EQ(peer_holds_nothing(c, 1, "the main card holds every expert"),
                 "gpu-mode peer: GPU 1 (RTX 5060 Ti) would hold nothing: the main card holds every expert; use gpu 0");
        c.cards.push_back(c.cards[1]);
        c.cards[2].label = "GPU 2 (RTX 5060 Ti)";
        CHECK_EQ(peer_holds_nothing(c, 2, "no room for an expert"),
                 "gpu-mode peer: GPU 2 (RTX 5060 Ti) would hold nothing: no room for an expert; leave it out of gpu");
    }
    {
        // prompt stages: bandwidth 1792:448 gives the RTX 5060 Ti the last fifth of 48 layers; gpu-share moves it; a stage
        // stays within the layers its weights may hold, and every card reads one layer at least
        GpuCards c = two(GpuMode::Peer);
        CHECK(stage_cuts(c, 48, 2) == std::vector<int>({0, 38}));
        CHECK(stage_cuts(c, 64, 44) == std::vector<int>({0, 51}));
        c.given = {0.5, 0.5};
        CHECK(stage_cuts(c, 48, 2) == std::vector<int>({0, 24}));
        CHECK(stage_cuts(c, 64, 44) == std::vector<int>({0, 44}));
        c.given = {0.999, 0.001};
        CHECK(stage_cuts(c, 48, 2) == std::vector<int>({0, 47}));
        c.cards.push_back(c.cards[1]);
        c.given = {0.5, 0.25, 0.25};
        CHECK(stage_cuts(c, 48, 2) == std::vector<int>({0, 24, 36}));
    }

    // ---- the start lines
    {
        GpuCards c = two(GpuMode::Peer);
        CHECK_EQ(gpu_start_line(c), "gpus: peer; RTX 5090 (nvidia-smi 0) main, share 0.80; RTX 5060 Ti (nvidia-smi 1) peer, share 0.20 "
                                    "(bandwidth)");
        c.mode = GpuMode::Split;
        c.given = {0.75, 0.25};
        CHECK_EQ(gpu_start_line(c), "gpus: split; RTX 5090 (nvidia-smi 0) main, share 0.75; RTX 5060 Ti (nvidia-smi 1) split, share 0.25 "
                                    "(gpu-share)");
        CHECK_EQ(gpu_parts_line(c, {"layers 0-35", "layers 36-47"}), "gpus: RTX 5090 layers 0-35; RTX 5060 Ti layers 36-47");
        CHECK_EQ(gpu_parts_line(c, {"", ""}), "");
        GpuCards one = two(GpuMode::One);
        one.cards.pop_back();
        CHECK_EQ(gpu_start_line(one), "gpus: RTX 5090 (nvidia-smi 0)");
        CHECK(one.peers().empty() && c.peers() == std::vector<int>({1}));
    }
    if (g_fail) return 1;
    std::printf("multi_gpu_test: ok\n");
    return 0;
}
