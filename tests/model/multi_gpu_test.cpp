// tests/model/multi_gpu_test.cpp - the shared multi-GPU layer over fake card tables: the shares and their defaults, a split's
// cut from the shares, the expert deal and the prompt stages of a peer, the peer hooks' driver over a fake model, each
// refusal's text and the start line; the split hooks' driver the same way.
#include "basalt/model/multi_gpu.hpp"
#include "basalt/model/peer_hooks.hpp"
#include "basalt/model/split_hooks.hpp"

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

// a model's peer as the driver sees it: each peer card's room, what it divides, and what it was dealt
struct FakePeer final : PeerHooks {
    std::vector<uint64_t> rooms;   // per card (the main card's unused)
    PeerSpec spec;
    std::vector<int> opened;       // the devices peer_card opened, in order
    PeerPlacement got;
    bool placed = false;
    bool peer_card(size_t, int device, std::string&) override {
        opened.push_back(device);
        return true;
    }
    uint64_t peer_room(size_t card) const override { return rooms[card]; }
    PeerSpec peer_spec() const override { return spec; }
    bool peer_place(const PeerPlacement& p, std::string&) override {
        got = p;
        placed = true;
        return true;
    }
    std::string peer_main() const override { return "dense, KV"; }
};

// a model's split as the driver sees it: its bytes, and the cut and devices it was given
struct FakeSplit final : SplitHooks {
    SplitSpec spec;
    SplitCut got;
    std::vector<int> devs;
    bool placed = false;
    SplitSpec split_spec(const GpuCards&) const override { return spec; }
    bool split_place(const SplitCut& cut, const std::vector<int>& devices, std::string&) override {
        got = cut;
        devs = devices;
        placed = true;
        return true;
    }
    std::string split_part(size_t card) const override { return card == 0 ? "8 experts" : ""; }
};

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

    // ---- peer hooks: the deal by layer block, one by one, the stages, the refusals and each card's part
    {
        // blocks: one expert a layer of 10 layers is 100 bytes; a peer with room for 3 per layer past its reserve holds 3
        PeerSpec s;
        s.deal = PeerDeal::Blocks;
        s.unit.assign(10, 100);
        s.block_most = 8;
        s.reserve = 50;
        std::string err;
        GpuCards c = two(GpuMode::Peer);
        PeerPlacement p = deal_peers(c, s, {0, 50 + 3 * 1000 + 99}, err);
        CHECK(err.empty() && p.homes == 3 && p.first[1] == 0 && p.end[1] == 10);
        CHECK_EQ(peer_part(p, 1, 0), "30 experts, layers 0-9");
        // capped at the layer's experts
        p = deal_peers(c, s, {0, 1u << 30}, err);
        CHECK(err.empty() && p.homes == 8);
        // two peers: the blocks follow the rooms, or each peer's gpu-share of them
        c.cards.push_back(c.cards[1]);
        c.cards[2].label = "GPU 2 (RTX 5060 Ti)";
        p = deal_peers(c, s, {0, 50 + 2 * 700, 50 + 2 * 300}, err);
        CHECK(err.empty() && p.homes == 2 && p.end[1] == 7 && p.first[2] == 7 && p.end[2] == 10);
        CHECK_EQ(peer_part(p, 2, 0), "6 experts, layers 7-9");
        p = deal_peers(c, s, {0, 1450, 1450}, err);
        CHECK(err.empty() && p.homes == 2 && p.end[1] == 3 && p.first[2] == 3);
        c.given = {0.5, 0.375, 0.125};   // the peers 3:1 of their 2900 bytes: the second keeps 725
        p = deal_peers(c, s, {0, 1450, 1450}, err);
        CHECK(err.empty() && p.homes == 2 && p.end[1] == 7 && p.first[2] == 7);
        // a peer whose block cannot take one expert a layer is refused, named
        c.given.clear();
        p = deal_peers(c, s, {0, 50 + 2000, 40}, err);
        CHECK_EQ(err, "gpu-mode peer: GPU 2 (RTX 5060 Ti) would hold nothing: no room for an expert; leave it out of gpu");
        err.clear();
        s.main_holds_all = true;
        p = deal_peers(two(GpuMode::Peer), s, {0, 40}, err);
        CHECK_EQ(err, "gpu-mode peer: GPU 1 (RTX 5060 Ti) would hold nothing: the main card holds every expert; use gpu 0");
    }
    {
        // one by one: the experts outside the main card dealt within each peer's room less its reserve; a peer dealt none
        // is refused
        PeerSpec s;
        s.deal = PeerDeal::Experts;
        s.unit.assign(6, GiB);
        s.reserve = GiB / 2;
        std::string err;
        PeerPlacement p = deal_peers(two(GpuMode::Peer), s, {0, 4 * GiB}, err);
        CHECK(err.empty() && p.held[1] == 3 && p.owner == std::vector<int>({1, 1, 1, -1, -1, -1}));
        CHECK_EQ(peer_part(p, 1, 0), "3 experts (3.0 GiB)");
        s.unit.clear();
        p = deal_peers(two(GpuMode::Peer), s, {0, 4 * GiB}, err);
        CHECK_EQ(err, "gpu-mode peer: GPU 1 (RTX 5060 Ti) would hold nothing: the main card holds every expert; use gpu 0");
    }
    {
        // the driver: every peer card opened, then the rooms, the deal and the placement; each card's part after; stages
        FakePeer f;
        f.rooms = {0, 50 + 3 * 1000};
        f.spec.deal = PeerDeal::Blocks;
        f.spec.unit.assign(10, 100);
        f.spec.block_most = 8;
        f.spec.reserve = 50;
        f.spec.stage_layers = 48;
        f.spec.stage_lo = 2;
        std::string err;
        CHECK(f.peer_report(1).empty());
        CHECK(f.peer_open(two(GpuMode::Peer), err) && err.empty() && f.placed && f.opened == std::vector<int>({1}));
        CHECK(f.got.homes == 3 && f.got.stage_first == std::vector<int>({0, 38}));
        CHECK_EQ(f.peer_report(0), "dense, KV");
        CHECK_EQ(f.peer_report(1), "30 experts, layers 0-9, prompt layers 38-47");
        // one card, or another mode: nothing opens
        FakePeer one;
        GpuCards solo = two(GpuMode::One);
        solo.cards.pop_back();
        CHECK(one.peer_open(solo, err) && one.opened.empty() && !one.placed && one.peer_report(0).empty());
        // a refusal stops the open before the placement
        FakePeer none = f;
        none.placed = false;
        none.opened.clear();
        none.rooms = {0, 40};
        CHECK(!none.peer_open(two(GpuMode::Peer), err) && !none.placed &&
              err == "gpu-mode peer: GPU 1 (RTX 5060 Ti) would hold nothing: no room for an expert; use gpu 0");
    }

    // ---- the split hooks' driver: the cut by the shares, --diag-gpu-same, a refusal, the parts
    {
        FakeSplit f;
        f.spec.layer.assign(48, 1 * GiB);
        f.spec.fixed = {0, 0};
        f.spec.room = {40 * GiB, 14 * GiB};
        GpuCards c = two(GpuMode::Split);
        c.given = {0.75, 0.25};
        std::string err;
        CHECK(f.split_open(c, err) && f.placed && f.got.first == std::vector<int>({0, 36}) && f.devs == std::vector<int>({0, 1}));
        CHECK_EQ(f.split_report(0), "layers 0-35, 8 experts");
        CHECK_EQ(f.split_report(1), "layers 36-47");
        // --diag-gpu-same: the cut as given, every part on the main card, no room checked (one card holds both)
        FakeSplit same = f;
        same.placed = false;
        c.given.clear();
        c.diag_cut = 20;
        c.diag_same = true;
        CHECK(same.split_open(c, err) && same.got.first == std::vector<int>({0, 20}) && same.devs == std::vector<int>({0, 0}));
        c.diag_cut = 0;
        CHECK(!same.split_open(c, err) &&
              err == "--diag-gpu-same needs two cards and --diag-gpu-cut L, the layer the second part starts at");
        // a part that does not fit is refused before the placement
        FakeSplit big = f;
        big.placed = false;
        GpuCards d = two(GpuMode::Split);
        d.given = {0.5, 0.5};
        CHECK(!big.split_open(d, err) && !big.placed &&
              err == "gpu-mode split: GPU 1 (RTX 5060 Ti) takes layers 24-47 at share 0.50 and needs 24.0 GiB, 14.0 GiB free; lower "
                     "its gpu-share, max-context or parallel");
        // not split mode: nothing happens
        FakeSplit idle;
        idle.spec = f.spec;
        CHECK(idle.split_open(two(GpuMode::Peer), err) && !idle.placed && idle.split_report(0).empty());
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
