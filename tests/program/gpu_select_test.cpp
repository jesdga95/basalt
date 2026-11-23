// tests/program/gpu_select_test.cpp - the gpu, gpu-mode and gpu-share settings over fake card tables: parsing, every error
// text, the card choice, the model's modes, the roles.
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

static GpuChoice pick(const std::string& text, const std::vector<GpuCard>& v, const std::string& mode = "",
                      const std::string& share = "", bool known = true) {
    GpuSpec s;
    GpuMode m = GpuMode::One;
    std::string err;
    GpuChoice c;
    if (!parse_gpu_spec(text, s, err) || !parse_gpu_mode(mode, m, err)) {
        c.error = err;
        return c;
    }
    return choose_gpus(s, m, share, v, known);
}

static std::string cards(const GpuChoice& c) {
    std::string s;
    for (int i : c.cards) s += (s.empty() ? "" : ",") + std::to_string(i);
    return s;
}

int main() {
    // Parsing: a number, a list, all; any other word (the removed "auto" too) gets the one error.
    {
        const char* want = "gpu \"%s\": expected a card number like 0, a list like 0,1 (the first is the main card), or all; see docs/configuration/gpus.md";
        GpuSpec s;
        std::string err;
        CHECK(parse_gpu_spec("", s, err) && s.kind == GpuSpec::Kind::Unset);
        CHECK(parse_gpu_spec("all", s, err) && s.kind == GpuSpec::Kind::All && s.text == "all");
        CHECK(parse_gpu_spec("0", s, err) && s.kind == GpuSpec::Kind::List && s.list.size() == 1 && s.text == "0");
        CHECK(parse_gpu_spec(" 1 , 0 ", s, err) && s.kind == GpuSpec::Kind::List && s.list.size() == 2 && s.list[0] == 1 && s.text == "1,0");
        CHECK(parse_gpu_spec("1,0,2", s, err) && s.list.size() == 3);
        CHECK(!parse_gpu_spec("0,0", s, err));
        CHECK_STR(err, "gpu 0,0: GPU 0 is listed twice");
        for (const char* bad : {"auto", "x", "0,", "-1", "0,all", "ALL"}) {
            CHECK(!parse_gpu_spec(bad, s, err));
            char b[160];
            std::snprintf(b, sizeof b, want, bad);
            CHECK_STR(err, b);
        }
        GpuMode m = GpuMode::Tensor;
        CHECK(parse_gpu_mode("", m, err) && m == GpuMode::One);
        CHECK(parse_gpu_mode("peer", m, err) && m == GpuMode::Peer);
        CHECK(parse_gpu_mode(" split ", m, err) && m == GpuMode::Split);
        CHECK(parse_gpu_mode("tensor", m, err) && m == GpuMode::Tensor);
        CHECK(!parse_gpu_mode("auto", m, err));
        CHECK_STR(err, "gpu-mode \"auto\": expected peer, split or tensor; see docs/configuration/gpus.md");
        CHECK(!parse_gpu_mode("Peer", m, err));
    }
    // A 5090 and a 5060 Ti, in nvidia-smi order.
    {
        const std::vector<GpuCard> v = {card(0, "RTX 5090", 12, 0, 1792, 32), card(1, "RTX 5060 Ti", 12, 0, 448, 16)};
        GpuChoice c = pick("", v);
        CHECK_STR(cards(c), "0");
        CHECK(c.mode == GpuMode::One && c.shares.empty() && c.error.empty());
        CHECK_STR(c.warning, "GPU 1 (RTX 5060 Ti) unused; to use more cards set gpu (0,1 or all) and gpu-mode");
        c = pick("0,1", v, "peer");
        CHECK_STR(cards(c), "0,1");
        CHECK(c.warning.empty() && c.error.empty() && c.mode == GpuMode::Peer && c.shares.empty());
        CHECK_STR(cards(pick("1,0", v, "split")), "1,0");
        CHECK_STR(cards(pick("0,1", v, "tensor")), "0,1");
        CHECK_STR(cards(pick("1", v)), "1");
        CHECK(pick("1", v).warning.empty());
        c = pick("all", v, "peer");
        CHECK_STR(cards(c), "0,1");
        CHECK(c.error.empty() && c.warning.empty());
        CHECK_STR(pick("2", v).error, "gpu 2: 2 GPUs are visible (0-1)");
        CHECK_STR(pick("0,2", v, "peer").error, "gpu 2: 2 GPUs are visible (0-1)");
        CHECK_STR(pick("0,0", v).error, "gpu 0,0: GPU 0 is listed twice");
        // shares: normalized, in the order of gpu
        c = pick("0,1", v, "peer", "3,1");
        CHECK(c.error.empty() && c.shares.size() == 2 && c.shares[0] == 0.75 && c.shares[1] == 0.25);
        c = pick("1,0", v, "split", " 1 , 1 ");
        CHECK(c.error.empty() && c.shares.size() == 2 && c.shares[0] == 0.5);
        c = pick("all", v, "tensor", "2.5,7.5");
        CHECK(c.error.empty() && c.shares.size() == 2 && c.shares[0] == 0.25);
        CHECK_STR(gpu_plan_json(v, pick("1,0", v, "peer", "3,1")),
                  "{\"devices\":\"1,0\",\"cards\":[{\"visible\":1,\"token\":\"1\",\"smi\":1,\"name\":\"RTX 5060 Ti\"},"
                  "{\"visible\":0,\"token\":\"0\",\"smi\":0,\"name\":\"RTX 5090\"}],\"mode\":\"peer\",\"shares\":[0.75,0.25],"
                  "\"notes\":[],\"warning\":\"\",\"error\":\"\"}");
        CHECK_STR(gpu_plan_json(v, pick("0,1", v, "split")),
                  "{\"devices\":\"0,1\",\"cards\":[{\"visible\":0,\"token\":\"0\",\"smi\":0,\"name\":\"RTX 5090\"},"
                  "{\"visible\":1,\"token\":\"1\",\"smi\":1,\"name\":\"RTX 5060 Ti\"}],\"mode\":\"split\",\"shares\":[],"
                  "\"notes\":[],\"warning\":\"\",\"error\":\"\"}");
        CHECK_STR(gpu_plan_json(v, pick("0,1", v, "tensor", "1,1")),
                  "{\"devices\":\"0,1\",\"cards\":[{\"visible\":0,\"token\":\"0\",\"smi\":0,\"name\":\"RTX 5090\"},"
                  "{\"visible\":1,\"token\":\"1\",\"smi\":1,\"name\":\"RTX 5060 Ti\"}],\"mode\":\"tensor\",\"shares\":[0.5,0.5],"
                  "\"notes\":[],\"warning\":\"\",\"error\":\"\"}");
        CHECK_STR(gpu_plan_json(v, pick("0", v)),
                  "{\"devices\":\"0\",\"cards\":[{\"visible\":0,\"token\":\"0\",\"smi\":0,\"name\":\"RTX 5090\"}],\"mode\":\"\","
                  "\"shares\":[],\"notes\":[],\"warning\":\"\",\"error\":\"\"}");
        CHECK_STR(gpu_plan_json({}, GpuChoice{}, "qwen4exp", "", 0, 8, 8192, "Qwen3.8-Flash-Next", "--route-tail-skip"),
                  "{\"arch\":\"qwen4exp\",\"model\":\"Qwen3.8-Flash-Next\",\"community_kv\":\"\",\"quality_flags\":\"\",\"quality_window\":\"0\","
                  "\"lossy\":\"--route-tail-skip\",\"spec\":8,\"max_context\":8192,\"devices\":\"\",\"cards\":[],\"mode\":\"\",\"shares\":[],"
                  "\"notes\":[],\"warning\":\"\",\"error\":\"\"}");
        // every error text of the setting rules
        CHECK_STR(pick("0,1", v).error, "gpu 0,1 names 2 cards: set gpu-mode to peer, split or tensor; see docs/configuration/gpus.md");
        CHECK_STR(pick("1,0", v).error, "gpu 1,0 names 2 cards: set gpu-mode to peer, split or tensor; see docs/configuration/gpus.md");
        CHECK_STR(pick("all", v).error, "gpu all names 2 cards: set gpu-mode to peer, split or tensor; see docs/configuration/gpus.md");
        CHECK_STR(pick("0,1", v, "auto").error, "gpu-mode \"auto\": expected peer, split or tensor; see docs/configuration/gpus.md");
        CHECK_STR(pick("0", v, "peer").error, "gpu-mode peer: gpu names one card (gpu \"0\"); a mode needs two or more");
        CHECK_STR(pick("", v, "split").error, "gpu-mode split: gpu names one card (gpu \"0\"); a mode needs two or more");
        CHECK_STR(pick("1", v, "tensor").error, "gpu-mode tensor: gpu names one card (gpu \"1\"); a mode needs two or more");
        CHECK_STR(pick("0", v, "peer", "1").error, "gpu-mode peer: gpu names one card (gpu \"0\"); a mode needs two or more");
        CHECK_STR(pick("0", v, "", "1").error, "gpu-share: gpu names one card (gpu \"0\"); shares need two or more");
        CHECK_STR(pick("", v, "", "3,1").error, "gpu-share: gpu names one card (gpu \"0\"); shares need two or more");
        CHECK_STR(pick("0,1", v, "peer", "3").error, "gpu-share \"3\": expected one positive number per card of gpu (2 cards), like 3,1");
        CHECK_STR(pick("0,1", v, "peer", "3,1,1").error, "gpu-share \"3,1,1\": expected one positive number per card of gpu (2 cards), like 3,1");
        for (const char* bad : {"0,1", "-1,2", "a,b", "3,", ",1", "nan,1", "inf,1", "1 2"}) {
            char b[160];
            std::snprintf(b, sizeof b, "gpu-share \"%s\": expected one positive number per card of gpu (2 cards), like 3,1", bad);
            CHECK_STR(pick("0,1", v, "split", bad).error, b);
        }
        CHECK_STR(pick("all", v, "peer", "1,1,1").error, "gpu-share \"1,1,1\": expected one positive number per card of gpu (2 cards), like 3,1");
        CHECK(pick("0,1", v, "peer", "3,1").cards.size() == 2);
        CHECK(pick("0,1", v, "peer", "3,1").error.empty());
        CHECK(pick("0,2", v, "peer", "3").cards.empty());   // the card number is checked first
    }
    // one visible card: all names one card
    {
        const std::vector<GpuCard> v = {card(0, "RTX 5090", 12, 0, 1792, 32)};
        CHECK_STR(cards(pick("all", v)), "0");
        CHECK_STR(pick("all", v, "peer").error, "gpu-mode peer: gpu names one card (gpu \"all\"); a mode needs two or more");
        CHECK_STR(pick("all", v, "", "1,1").error, "gpu-share: gpu names one card (gpu \"all\"); shares need two or more");
        CHECK(pick("", v).warning.empty());
        const std::vector<GpuCard> none;
        CHECK_STR(pick("all", none).error, "gpu all: no GPU is visible");
        CHECK_STR(pick("all", none, "", "", false).error, "gpu all: GPUs not listable (no NVML)");
    }
    // The model's side: the mode it runs and the cards it takes.
    {
        const std::vector<GpuCard> v = {card(0, "RTX 5090", 12, 0, 1792, 32), card(1, "RTX 5060 Ti", 12, 0, 448, 16),
                                        card(2, "RTX 5060 Ti", 12, 0, 448, 16), card(3, "RTX 5060 Ti", 12, 0, 448, 16),
                                        card(4, "RTX 5060 Ti", 12, 0, 448, 16)};
        const GpuModes hy4{4, 0, 0}, dense{2, 2, 0}, flash{4, 8, 0};
        CHECK_STR(check_gpu_modes(pick("0,1", v, "peer"), "Hy4-flash", hy4), "");
        CHECK_STR(check_gpu_modes(pick("0,1,2,3", v, "peer"), "Hy4-flash", hy4), "");
        CHECK_STR(check_gpu_modes(pick("0", v), "Hy4-flash", hy4), "");
        CHECK_STR(check_gpu_modes(pick("0,1", v, "split"), "Hy4-flash", hy4), "gpu-mode split: Hy4-flash does not run it; it runs peer");
        CHECK_STR(check_gpu_modes(pick("0,1", v, "tensor"), "Qwen3.8-27B", dense),
                  "gpu-mode tensor: Qwen3.8-27B does not run it (not built); it runs peer or split");
        CHECK_STR(check_gpu_modes(pick("0,1", v, "tensor"), "Hy4-flash", hy4), "gpu-mode tensor: Hy4-flash does not run it (not built); it runs peer");
        CHECK_STR(check_gpu_modes(pick("0,1", v, "peer"), "Inkling-2-Mini", GpuModes{0, 2, 0}), "gpu-mode peer: Inkling-2-Mini does not run it; it runs split");
        CHECK_STR(check_gpu_modes(pick("0,1,2,3,4", v, "peer"), "Hy4-flash", hy4),
                  "gpu-mode peer: Hy4-flash takes at most 4 cards (the main card and 3 peers); gpu names 5");
        CHECK_STR(check_gpu_modes(pick("all", v, "peer"), "Hy4-flash", hy4),
                  "gpu-mode peer: Hy4-flash takes at most 4 cards (the main card and 3 peers); gpu names 5");
        CHECK_STR(check_gpu_modes(pick("0,1,2", v, "peer"), "Qwen3.8-27B", dense),
                  "gpu-mode peer: Qwen3.8-27B takes at most 2 cards (the main card and 1 peer); gpu names 3");
        CHECK_STR(check_gpu_modes(pick("0,1,2", v, "split"), "Qwen3.8-27B", dense), "gpu-mode split: Qwen3.8-27B takes at most 2 cards; gpu names 3");
        CHECK_STR(check_gpu_modes(pick("0,1,2,3,4", v, "split"), "Qwen3.8-Flash-Next", flash), "");
        CHECK_STR(check_gpu_modes(pick("0,1,2,3,4", v, "peer"), "Qwen3.8-Flash-Next", flash),
                  "gpu-mode peer: Qwen3.8-Flash-Next takes at most 4 cards (the main card and 3 peers); gpu names 5");
        // an sm_89 peer, which only a -DPEER_ADA=ON build lists: a model whose peer kernels that build does not compile refuses it
        const std::vector<GpuCard> ada = {card(0, "RTX 5090", 12, 0, 1792, 32), card(1, "RTX 4060 Ti", 8, 9, 288, 16)};
        GpuChoice two;
        two.cards = {0, 1};
        two.mode = GpuMode::Peer;
        CHECK_STR(check_gpu_modes(two, "Hy4-flash", hy4, ada), "gpu-mode peer: GPU 1 (RTX 4060 Ti) is sm_89; Hy4-flash runs peer on sm_120 cards");
        CHECK_STR(check_gpu_modes(two, "Qwen3.8-Flash-Next", GpuModes{4, 8, 0, true}, ada), "");
    }
    // The same cards with the slower one first.
    {
        const std::vector<GpuCard> v = {card(0, "RTX 5060 Ti", 12, 0, 448, 16), card(1, "RTX 5090", 12, 0, 1792, 32)};
        CHECK_STR(cards(pick("all", v, "split")), "0,1");   // the order is the user's: GPU 0 is main
        CHECK_STR(cards(pick("1,0", v, "split")), "1,0");
        CHECK_STR(cards(pick("", v)), "0");
    }
    // Four mixed cards, one of them not Blackwell: all takes them all and refuses the one that cannot run.
    {
        const std::vector<GpuCard> v = {card(0, "RTX 5070 Ti", 12, 0, 896, 16), card(1, "RTX 3090", 8, 6, 936, 24),
                                        card(2, "RTX 5090", 12, 0, 1792, 32), card(3, "RTX 5060 Ti", 12, 0, 448, 16)};
        CHECK_STR(pick("all", v, "peer").error, "gpu-mode peer: GPU 1 (RTX 3090) is sm_86; peer runs on sm_120 cards");
        CHECK_STR(pick("0,2,3", v, "peer").error, "");
        CHECK_STR(cards(pick("3,2,0", v, "split")), "3,2,0");
        CHECK_STR(pick("0,1", v, "split").error, "gpu-mode split: GPU 1 (RTX 3090) is sm_86; split and tensor run on sm_120 cards");
        CHECK_STR(pick("0,1", v, "tensor").error, "gpu-mode tensor: GPU 1 (RTX 3090) is sm_86; split and tensor run on sm_120 cards");
        GpuChoice c = pick("", v);
        CHECK_STR(cards(c), "0");
        CHECK_STR(c.warning, "GPU 2 (RTX 5090), GPU 3 (RTX 5060 Ti) unused; to use more cards set gpu (0,1 or all) and gpu-mode");
        CHECK_STR(pick("1", v).error, "gpu 1: GPU 1 (RTX 3090) is sm_86; the main card must be sm_120");
        CHECK_STR(pick("1,0", v, "peer").error, "gpu 1,0: GPU 1 (RTX 3090) is sm_86; the main card must be sm_120");
    }
    // An Ada (sm_89) card: never the main card; a peer only in a -DPEER_ADA=ON build; never in split or tensor
    {
        const std::vector<GpuCard> v = {card(0, "RTX 5080", 12, 0, 960, 16), card(1, "RTX 4060 Ti", 8, 9, 288, 16)};
        CHECK_STR(pick("1,0", v, "peer").error, "gpu 1,0: GPU 1 (RTX 4060 Ti) is sm_89; the main card must be sm_120");
        CHECK_STR(pick("1,0", v, "split").error, "gpu 1,0: GPU 1 (RTX 4060 Ti) is sm_89; the main card must be sm_120");
        CHECK_STR(pick("1", v).error, "gpu 1: GPU 1 (RTX 4060 Ti) is sm_89; the main card must be sm_120");
        CHECK_STR(pick("0,1", v, "split").error,
                  "gpu-mode split: GPU 1 (RTX 4060 Ti) is sm_89; split and tensor run on sm_120 cards (peer takes sm_89 in a -DPEER_ADA=ON build)");
        CHECK_STR(pick("all", v, "tensor").error,
                  "gpu-mode tensor: GPU 1 (RTX 4060 Ti) is sm_89; split and tensor run on sm_120 cards (peer takes sm_89 in a -DPEER_ADA=ON build)");
#if defined(BASALT_PEER_ADA)
        CHECK_STR(cards(pick("0,1", v, "peer")), "0,1");
        CHECK_STR(cards(pick("all", v, "peer")), "0,1");
#else
        CHECK_STR(pick("0,1", v, "peer").error, "gpu-mode peer: GPU 1 (RTX 4060 Ti) is sm_89; peer takes sm_89 in a -DPEER_ADA=ON build");
        CHECK_STR(pick("all", v, "peer").error, "gpu-mode peer: GPU 1 (RTX 4060 Ti) is sm_89; peer takes sm_89 in a -DPEER_ADA=ON build");
#endif
    }
    // A laptop's own card beside a desktop card in an enclosure: the order is the user's, and unset is GPU 0 alone.
    {
        const std::vector<GpuCard> v = {card(0, "RTX 5080 Laptop GPU", 12, 0, 896, 16), card(1, "RTX 5090", 12, 0, 1792, 32)};
        CHECK_STR(cards(pick("1,0", v, "peer")), "1,0");
        CHECK_STR(cards(pick("", v)), "0");
        CHECK(!pick("", v).warning.empty());
    }
    // A first card Basalt cannot run: unset is an error.
    {
        const std::vector<GpuCard> v = {card(0, "RTX 4090", 8, 9, 1008, 24), card(1, "RTX 5090", 12, 0, 1792, 32)};
        CHECK_STR(pick("", v).error, "gpu 0: GPU 0 (RTX 4090) is sm_89; the main card must be sm_120");
        CHECK_STR(pick("all", v, "peer").error, "gpu all: GPU 0 (RTX 4090) is sm_89; the main card must be sm_120");
        CHECK_STR(cards(pick("1", v)), "1");
    }
    // No card list (no management library, no CUDA_VISIBLE_DEVICES): numbers pass, all cannot run.
    {
        const std::vector<GpuCard> none;
        CHECK_STR(cards(pick("", none, "", "", false)), "0");
        CHECK(pick("", none, "", "", false).warning.empty());
        CHECK_STR(cards(pick("3,1", none, "peer", "", false)), "3,1");
        CHECK_STR(pick("3,1", none, "", "", false).error, "gpu 3,1 names 2 cards: set gpu-mode to peer, split or tensor; see docs/configuration/gpus.md");
        CHECK(!pick("all", none, "", "", false).error.empty());
        CHECK_STR(pick("", none, "", "", true).error, "gpu: no GPU is visible");
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
    if (g_fail) { std::printf("gpu_select_test: %d failures\n", g_fail); return 1; }
    std::printf("gpu_select_test: ok\n");
    return 0;
}
