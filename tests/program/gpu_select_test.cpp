// tests/program/gpu_select_test.cpp - the gpu setting over fake card tables: parsing, the card choice, the roles.
#include "basalt/program/gpu_select.hpp"

#include "basalt/platform/process.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace basalt::program;

static int g_fail = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)
#define CHECK_STR(got, want)                                                                         \
    do {                                                                                             \
        const std::string g_ = (got);                                                                \
        if (g_ != (want)) { std::printf("FAIL %s:%d: got \"%s\" want \"%s\"\n", __FILE__, __LINE__, g_.c_str(), want); ++g_fail; } \
    } while (0)

static constexpr uint64_t G = 1ull << 30;

static GpuCard card(int smi, const char* name, int major, int minor, double gbps, uint64_t gib) {
    GpuCard c;
    c.token = std::to_string(smi);
    c.smi = smi;
    c.name = name;
    c.cc_major = major;
    c.cc_minor = minor;
    c.mem_gbps = gbps;
    c.vram = gib * G;
    return c;
}

static GpuChoice pick(const std::string& text, const std::vector<GpuCard>& v, bool known = true) {
    GpuSpec s;
    std::string err;
    if (!parse_gpu_spec(text, s, err)) {
        GpuChoice c;
        c.error = err;
        return c;
    }
    return choose_gpus(s, v, known);
}

static std::string cards(const GpuChoice& c) {
    std::string s;
    for (int i : c.cards) s += (s.empty() ? "" : ",") + std::to_string(i);
    return s;
}

static std::string roles(const std::vector<GpuCard>& v, const GpuChoice& c, bool peer_access, int tiers) {
    RoleInputs in;
    for (int i : c.cards) in.cards.push_back(v[(size_t) i]);
    in.peer_access = peer_access;
    in.max_expert_tiers = tiers;
    std::string s;
    for (GpuRole r : choose_roles(in)) s += (s.empty() ? "" : ",") + std::string(gpu_role_name(r));
    return s;
}

int main() {
    // Parsing.
    {
        GpuSpec s;
        std::string err;
        CHECK(parse_gpu_spec("", s, err) && s.kind == GpuSpec::Kind::Unset);
        CHECK(parse_gpu_spec("auto", s, err) && s.kind == GpuSpec::Kind::Auto);
        CHECK(parse_gpu_spec(" 1 , 0 ", s, err) && s.kind == GpuSpec::Kind::List && s.list.size() == 2 && s.list[0] == 1);
        CHECK(!parse_gpu_spec("0,0", s, err));
        CHECK_STR(err, "gpu 0,0: GPU 0 is listed twice");
        CHECK(!parse_gpu_spec("x", s, err));
        CHECK(!parse_gpu_spec("0,", s, err));
        CHECK(!parse_gpu_spec("-1", s, err));
    }
    // A 5090 and a 5060 Ti, in nvidia-smi order.
    {
        const std::vector<GpuCard> v = {card(0, "RTX 5090", 12, 0, 1792, 32), card(1, "RTX 5060 Ti", 12, 0, 448, 16)};
        GpuChoice c = pick("", v);
        CHECK_STR(cards(c), "0");
        CHECK_STR(c.warning, "GPU 1 (RTX 5060 Ti) unused; use gpu 0,1 or gpu auto");
        c = pick("0,1", v);
        CHECK_STR(cards(c), "0,1");
        CHECK(c.warning.empty() && c.error.empty());
        CHECK_STR(roles(v, c, true, 1), "main,expert tier");
        CHECK_STR(roles(v, c, false, 1), "main,idle");
        CHECK_STR(cards(pick("1,0", v)), "1,0");
        CHECK_STR(cards(pick("1", v)), "1");
        CHECK(pick("1", v).warning.empty());
        CHECK_STR(cards(pick("auto", v)), "0,1");
        CHECK_STR(pick("2", v).error, "gpu 2: 2 GPUs are visible (0-1)");
        CHECK_STR(pick("0,0", v).error, "gpu 0,0: GPU 0 is listed twice");
        CHECK_STR(gpu_plan_json(v, pick("1,0", v)),
                  "{\"devices\":\"1,0\",\"cards\":[{\"visible\":1,\"token\":\"1\",\"smi\":1,\"name\":\"RTX 5060 Ti\"},"
                  "{\"visible\":0,\"token\":\"0\",\"smi\":0,\"name\":\"RTX 5090\"}],\"notes\":[],\"warning\":\"\",\"error\":\"\"}");
    }
    // The same cards with the slower one first: auto still makes the 5090 main.
    {
        const std::vector<GpuCard> v = {card(0, "RTX 5060 Ti", 12, 0, 448, 16), card(1, "RTX 5090", 12, 0, 1792, 32)};
        CHECK_STR(cards(pick("auto", v)), "1,0");
        CHECK_STR(cards(pick("", v)), "0");
    }
    // Two equal 5090s: the lowest number is main.
    {
        const std::vector<GpuCard> v = {card(0, "RTX 5090", 12, 0, 1792, 32), card(1, "RTX 5090", 12, 0, 1792, 32)};
        const GpuChoice c = pick("auto", v);
        CHECK_STR(cards(c), "0,1");
        CHECK(c.notes.empty());
        CHECK_STR(roles(v, c, true, 1), "main,expert tier");
    }
    // Four mixed cards, one of them not Blackwell.
    {
        const std::vector<GpuCard> v = {card(0, "RTX 5070 Ti", 12, 0, 896, 16), card(1, "RTX 3090", 8, 6, 936, 24),
                                        card(2, "RTX 5090", 12, 0, 1792, 32), card(3, "RTX 5060 Ti", 12, 0, 448, 16)};
        GpuChoice c = pick("auto", v);
        CHECK_STR(cards(c), "2,0,3");
        CHECK(c.notes.size() == 1);
        if (c.notes.size() == 1) CHECK_STR(c.notes[0], "gpu auto: GPU 1 (RTX 3090) skipped, not sm_120");
        CHECK_STR(roles(v, c, true, 1), "main,expert tier,idle");
        CHECK_STR(roles(v, c, true, 3), "main,expert tier,expert tier");
        c = pick("", v);
        CHECK_STR(cards(c), "0");
        CHECK_STR(c.warning, "GPU 2 (RTX 5090), GPU 3 (RTX 5060 Ti) unused; use gpu 0,2,3 or gpu auto");
        CHECK_STR(pick("1", v).error, "gpu: GPU 1 (RTX 3090) is sm_86, not sm_120");
        CHECK_STR(cards(pick("3,2,0", v)), "3,2,0");
    }
    // A laptop's own card beside a desktop card in an enclosure: the desktop card's bandwidth makes it main; the
    // laptop card is the first visible, so it runs alone when nothing is set.
    {
        const std::vector<GpuCard> v = {card(0, "RTX 5080 Laptop GPU", 12, 0, 896, 16), card(1, "RTX 5090", 12, 0, 1792, 32)};
        CHECK_STR(cards(pick("auto", v)), "1,0");
        CHECK_STR(cards(pick("", v)), "0");
        CHECK(!pick("", v).warning.empty());
        // the same bandwidth: the larger card is main
        const std::vector<GpuCard> w = {card(0, "RTX 5070 Ti Laptop GPU", 12, 0, 896, 12), card(1, "RTX 5080", 12, 0, 896, 16)};
        CHECK_STR(cards(pick("auto", w)), "1,0");
    }
    // A first card Basalt cannot run: unset is an error that points at gpu N and auto.
    {
        const std::vector<GpuCard> v = {card(0, "RTX 4090", 8, 9, 1008, 24), card(1, "RTX 5090", 12, 0, 1792, 32)};
        CHECK(!pick("", v).error.empty());
        CHECK_STR(cards(pick("auto", v)), "1");
        CHECK(pick("auto", v).notes.size() == 1);
    }
    // No card list (no management library, no CUDA_VISIBLE_DEVICES): numbers pass, auto cannot run.
    {
        const std::vector<GpuCard> none;
        CHECK_STR(cards(pick("", none, false)), "0");
        CHECK(pick("", none, false).warning.empty());
        CHECK_STR(cards(pick("3,1", none, false)), "3,1");
        CHECK(!pick("auto", none, false).error.empty());
        CHECK_STR(pick("", none, true).error, "gpu: no GPU is visible");
    }
    // Under CUDA_VISIBLE_DEVICES the numbers count within its list, and the narrowed list keeps its tokens.
    {
        basalt::platform::set_env("CUDA_VISIBLE_DEVICES", "1,0");
        bool known = false;
        const std::vector<GpuCard> v = visible_gpus(known);
        CHECK(known && v.size() == 2);
        if (v.size() == 2) {
            CHECK_STR(v[0].token, "1");
            CHECK_STR(v[1].token, "0");
            apply_gpus(v, pick("1", v).cards);
            const char* now = std::getenv("CUDA_VISIBLE_DEVICES");
            CHECK_STR(now ? now : "", "0");
            const char* order = std::getenv("CUDA_DEVICE_ORDER");
            CHECK_STR(order ? order : "", "PCI_BUS_ID");
        }
        basalt::platform::set_env("CUDA_VISIBLE_DEVICES", "-1");
        CHECK(visible_gpus(known).empty() && known);
    }
    CHECK_STR(gpu_identity(3), "0,1,2");
    // Expert placement over several tiers: equal cards alternate; bandwidth sets the share of the hot experts; a
    // smaller card closes first and the others take the rest; a single tier takes the ranking in order.
    {
        const std::vector<uint64_t> eq(8, 10);
        auto owners = [](const std::vector<int>& o) {
            std::string s;
            for (int x : o) s += x < 0 ? std::string("-") : std::to_string(x);
            return s;
        };
        CHECK_STR(owners(spread_experts(eq, {{40, 448}, {40, 448}})), "01010101");
        CHECK_STR(owners(spread_experts(eq, {{80, 1792}, {80, 896}})), "01001001");
        CHECK_STR(owners(spread_experts(eq, {{20, 1792}, {100, 448}})), "01011111");
        CHECK_STR(owners(spread_experts(eq, {{30, 448}})), "000-----");
        CHECK_STR(owners(spread_experts(eq, {{0, 448}, {30, 448}})), "111-----");
        // three and four tiers of mixed sizes and bandwidths: every expert placed while room lasts, none twice
        const std::vector<uint64_t> mixed = {12, 12, 12, 8, 8, 8, 8, 8, 8, 8, 8, 8};
        const std::vector<int> o3 = spread_experts(mixed, {{30, 1792}, {20, 896}, {60, 448}});
        CHECK_STR(owners(o3), "012001222222");
        const std::vector<int> o4 = spread_experts(mixed, {{16, 448}, {16, 448}, {16, 448}, {200, 1792}});
        CHECK_STR(owners(o4), "012333333333");
        CHECK_STR(owners(spread_experts(eq, {{0, 1}, {0, 1}, {0, 1}})), "--------");
    }
    if (g_fail) { std::printf("gpu_select_test: %d failures\n", g_fail); return 1; }
    std::printf("gpu_select_test: ok\n");
    return 0;
}
