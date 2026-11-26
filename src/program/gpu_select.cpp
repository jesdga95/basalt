// src/program/gpu_select.cpp - see gpu_select.hpp.
#include "basalt/program/gpu_select.hpp"

#include "basalt/core/cuda_cc.hpp"

#include "basalt/log.hpp"
#include "basalt/platform/dynlib.hpp"
#include "basalt/platform/process.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

namespace basalt::program {

namespace {

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t')) ++a;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t')) --b;
    return s.substr(a, b - a);
}

std::vector<std::string> split_commas(const std::string& s) {
    std::vector<std::string> out;
    size_t at = 0;
    while (at <= s.size()) {
        size_t end = s.find(',', at);
        if (end == std::string::npos) end = s.size();
        out.push_back(trim(s.substr(at, end - at)));
        at = end + 1;
    }
    return out;
}

bool all_digits(const std::string& s) {
    if (s.empty() || s.size() > 4) return false;
    for (char c : s)
        if (c < '0' || c > '9') return false;
    return true;
}

std::string short_name(std::string n) {
    for (const char* v : {"NVIDIA ", "GeForce "})
        if (n.rfind(v, 0) == 0) n.erase(0, std::strlen(v));
    return n;
}

std::string json_str(const std::string& s) {
    std::string o = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if ((unsigned char) c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += c;
    }
    return o + "\"";
}

// The management library's view of every card, in nvidia-smi order.
struct NvmlCards {
    std::vector<GpuCard> cards;       // token = the nvidia-smi number
    std::vector<std::string> uuids;
    bool ok = false;
};

NvmlCards read_nvml() {
    NvmlCards out;
    const basalt::platform::DynLib lib = basalt::platform::DynLib::open_nvml();
    if (!lib) return out;
    struct MemInfo { unsigned long long total, free, used; };
    const auto init = lib.sym<int (*)()>("nvmlInit_v2");
    const auto shut = lib.sym<int (*)()>("nvmlShutdown");
    const auto count = lib.sym<int (*)(unsigned*)>("nvmlDeviceGetCount_v2");
    const auto handle = lib.sym<int (*)(unsigned, void**)>("nvmlDeviceGetHandleByIndex_v2");
    const auto name = lib.sym<int (*)(void*, char*, unsigned)>("nvmlDeviceGetName");
    const auto uuid = lib.sym<int (*)(void*, char*, unsigned)>("nvmlDeviceGetUUID");
    const auto cc = lib.sym<int (*)(void*, int*, int*)>("nvmlDeviceGetCudaComputeCapability");
    const auto mem = lib.sym<int (*)(void*, MemInfo*)>("nvmlDeviceGetMemoryInfo");
    const auto bus = lib.sym<int (*)(void*, unsigned*)>("nvmlDeviceGetMemoryBusWidth");
    const auto clk = lib.sym<int (*)(void*, int, unsigned*)>("nvmlDeviceGetMaxClockInfo");
    if (!init || !count || !handle || init() != 0) return out;
    unsigned n = 0;
    if (count(&n) == 0) {
        for (unsigned i = 0; i < n; ++i) {
            GpuCard c;
            c.token = std::to_string(i);
            c.smi = (int) i;
            std::string id;
            void* h = nullptr;
            if (handle(i, &h) == 0) {
                char b[128] = {0};
                if (name && name(h, b, sizeof b) == 0) c.name = short_name(b);
                char u[96] = {0};
                if (uuid && uuid(h, u, sizeof u) == 0) id = u;
                int ma = 0, mi = 0;
                if (cc && cc(h, &ma, &mi) == 0) { c.cc_major = ma; c.cc_minor = mi; }
                MemInfo m{};
                if (mem && mem(h, &m) == 0) c.vram = m.total;
                unsigned width = 0, mhz = 0;
                // the memory clock NVML reports is half the data rate (two transfers per clock)
                if (bus && clk && bus(h, &width) == 0 && clk(h, 2 /* NVML_CLOCK_MEM */, &mhz) == 0)
                    c.mem_gbps = (double) mhz * (double) width / 4000.0;
            }
            out.cards.push_back(c);
            out.uuids.push_back(id);
        }
        out.ok = true;
    }
    if (shut) shut();
    return out;
}

}  // namespace

const char* gpu_mode_name(GpuMode m) {
    return m == GpuMode::Peer ? "peer" : m == GpuMode::Split ? "split" : m == GpuMode::Tensor ? "tensor" : "";
}

bool parse_gpu_spec(const std::string& text, GpuSpec& out, std::string& err) {
    out = GpuSpec{};
    const std::string t = trim(text);
    if (t.empty()) return true;
    if (t == "all") { out.kind = GpuSpec::Kind::All; out.text = t; return true; }
    out.kind = GpuSpec::Kind::List;
    for (const std::string& w : split_commas(t)) {
        if (!all_digits(w)) {
            err = "gpu \"" + t + "\": expected a card number like 0, a list like 0,1 (the first is the main card), or all; see docs/configuration/gpus.md";
            return false;
        }
        const int v = std::atoi(w.c_str());
        if (std::find(out.list.begin(), out.list.end(), v) != out.list.end()) {
            err = "gpu " + t + ": GPU " + w + " is listed twice";
            return false;
        }
        out.list.push_back(v);
        out.text += (out.text.empty() ? "" : ",") + std::to_string(v);
    }
    return true;
}

bool parse_gpu_mode(const std::string& text, GpuMode& out, std::string& err) {
    out = GpuMode::One;
    const std::string t = trim(text);
    if (t.empty()) return true;
    for (GpuMode m : {GpuMode::Peer, GpuMode::Split, GpuMode::Tensor})
        if (t == gpu_mode_name(m)) { out = m; return true; }
    err = "gpu-mode \"" + t + "\": expected peer, split or tensor; see docs/configuration/gpus.md";
    return false;
}

namespace {

// "3,1" as numbers, one per card; false when any is not a positive finite number or the count is not `n`.
bool parse_shares(const std::string& text, size_t n, std::vector<double>& out) {
    out.clear();
    for (const std::string& w : split_commas(trim(text))) {
        char* end = nullptr;
        const double v = w.empty() ? 0.0 : std::strtod(w.c_str(), &end);
        if (w.empty() || end != w.c_str() + w.size() || !(v > 0.0) || !std::isfinite(v)) return false;
        out.push_back(v);
    }
    return out.size() == n;
}

std::string arch_name(const GpuCard& c) { return "sm_" + std::to_string(c.cc_major) + std::to_string(c.cc_minor); }

}  // namespace

// The main GPU runs the model's decode kernels (NVFP4 tensor cores, Blackwell instructions) — this build's compute capability only.
// A peer GPU runs only the expert tier's standard CUDA kernels (Q8_1/Q2_0 rows, shared-memory rings,
// doorbells), which a build with -DPEER_ADA=ON also compiles for sm_89 (Ada).
bool gpu_main_supported(const GpuCard& c) { return c.cc_major == 0 || core::cuda_cc_is_ours(c.cc_major, c.cc_minor); }
// sm_89 (Ada) is the one older architecture the peer kernels can be compiled for (when built with PEER_ADA);
// a peer card of any other pre-Blackwell arch (or an sm_89 card without PEER_ADA) is rejected here
// rather than failing later with 'no kernel image' at the first peer launch.
bool gpu_peer_supported(const GpuCard& c) {
#if defined(BASALT_PEER_ADA)
    return c.cc_major == 0 || core::cuda_cc_is_ours(c.cc_major, c.cc_minor) || (c.cc_major == 8 && c.cc_minor == 9);
#else
    return c.cc_major == 0 || core::cuda_cc_is_ours(c.cc_major, c.cc_minor);
#endif
}

std::string gpu_label(const std::vector<GpuCard>& visible, int index) {
    std::string s = "GPU " + std::to_string(index);
    if (index >= 0 && (size_t) index < visible.size() && !visible[(size_t) index].name.empty())
        s += " (" + visible[(size_t) index].name + ")";
    return s;
}

GpuChoice choose_gpus(const GpuSpec& spec, GpuMode mode, const std::string& share_text, const std::vector<GpuCard>& visible,
                      bool known) {
    GpuChoice c;
    const int n = (int) visible.size();
    const std::string shown = spec.kind == GpuSpec::Kind::Unset ? "0" : spec.text;
    // the cards
    if (spec.kind == GpuSpec::Kind::All) {
        if (!known) { c.error = "gpu all: GPUs not listable (no NVML)"; return c; }
        if (n == 0) { c.error = "gpu all: no GPU is visible"; return c; }
        for (int i = 0; i < n; ++i) c.cards.push_back(i);
    } else if (spec.kind == GpuSpec::Kind::Unset) {
        if (known && n == 0) { c.error = "gpu: no GPU is visible"; return c; }
        c.cards.push_back(0);
    } else {
        for (const int i : spec.list) {
            if (known && i >= n) {
                c.error = "gpu " + std::to_string(i) + ": " +
                          (n == 0 ? std::string("no GPU is visible")
                                  : std::to_string(n) + (n == 1 ? " GPU is" : " GPUs are") + " visible (0-" + std::to_string(n - 1) + ")");
                c.cards.clear();
                return c;
            }
            c.cards.push_back(i);
        }
    }
    auto refuse = [&](const std::string& why) {
        c.error = why;
        c.cards.clear();
        return c;
    };
    // the main card runs the model's own kernels: this build's compute capability only
    if (known && !gpu_main_supported(visible[(size_t) c.cards[0]]))
        return refuse("gpu " + shown + ": " + gpu_label(visible, c.cards[0]) + " is " + arch_name(visible[(size_t) c.cards[0]]) +
                      "; the main card must be " + core::cuda_cc_name());
    // what the settings say about the number of cards
    const size_t count = c.cards.size();
    const std::string mode_text = gpu_mode_name(mode);
    if (count == 1) {
        if (mode != GpuMode::One)
            return refuse("gpu-mode " + mode_text + ": gpu names one card (gpu \"" + shown + "\"); a mode needs two or more");
        if (!trim(share_text).empty())
            return refuse("gpu-share: gpu names one card (gpu \"" + shown + "\"); shares need two or more");
        c.mode = mode;
        if (spec.kind == GpuSpec::Kind::Unset) {
            std::vector<int> others;
            for (int i = 1; i < n; ++i)
                if (gpu_peer_supported(visible[(size_t) i])) others.push_back(i);
            if (!others.empty()) {
                std::string names;
                for (int i : others) names += (names.empty() ? "" : ", ") + gpu_label(visible, i);
                c.warning = names + " unused; to use more cards set gpu (0,1 or all) and gpu-mode";
            }
        }
        return c;
    }
    if (mode == GpuMode::One)
        return refuse("gpu " + shown + " names " + std::to_string(count) + " cards: set gpu-mode to peer, split or tensor; see docs/configuration/gpus.md");
    c.mode = mode;
    if (!trim(share_text).empty()) {
        std::vector<double> shares;
        if (!parse_shares(share_text, count, shares))
            return refuse("gpu-share \"" + trim(share_text) + "\": expected one positive number per card of gpu (" +
                          std::to_string(count) + " cards), like 3,1");
        double sum = 0;
        for (double v : shares) sum += v;
        for (double v : shares) c.shares.push_back(v / sum);
    }
    // the cards after the main one: a peer takes this build's capability, or sm_89 in a -DPEER_ADA=ON build; split and tensor this capability only
    for (size_t idx = 1; known && idx < count; ++idx) {
        const GpuCard& g = visible[(size_t) c.cards[idx]];
        const std::string who = gpu_label(visible, c.cards[idx]);
        const bool ada = g.cc_major == 8 && g.cc_minor == 9;
        if (mode == GpuMode::Peer) {
            if (!gpu_peer_supported(g))
                return refuse("gpu-mode peer: " + who + " is " + arch_name(g) +
                              (ada ? "; peer takes sm_89 in a -DPEER_ADA=ON build" : "; peer runs on " + core::cuda_cc_name() + " cards"));
        } else if (!gpu_main_supported(g)) {
            return refuse("gpu-mode " + mode_text + ": " + who + " is " + arch_name(g) + "; split and tensor run on " + core::cuda_cc_name() + " cards" +
                          (ada ? " (peer takes sm_89 in a -DPEER_ADA=ON build)" : ""));
        }
    }
    return c;
}

std::string check_gpu_modes(const GpuChoice& c, const std::string& model, const GpuModes& m, const std::vector<GpuCard>& visible) {
    if (c.mode == GpuMode::One) return {};
    const char* name = gpu_mode_name(c.mode);
    const int most = c.mode == GpuMode::Peer ? m.peer : c.mode == GpuMode::Split ? m.split : m.tensor;
    const std::string head = std::string("gpu-mode ") + name + ": " + model;
    if (most == 0) {
        std::string runs;
        const char* names[3] = {"peer", "split", "tensor"};
        const int takes[3] = {m.peer, m.split, m.tensor};
        for (int i = 0; i < 3; ++i)
            if (takes[i] > 0) runs += (runs.empty() ? "" : " or ") + std::string(names[i]);
        return head + " does not run it" + (c.mode == GpuMode::Tensor ? " (not built)" : "") + "; it runs " + runs;
    }
    if ((int) c.cards.size() > most)
        return head + " takes at most " + std::to_string(most) + " cards" +
               (c.mode == GpuMode::Peer ? " (the main card and " + std::to_string(most - 1) + (most == 2 ? " peer)" : " peers)") : std::string()) +
               "; gpu names " + std::to_string(c.cards.size());
    // an sm_89 peer (a -DPEER_ADA=ON build) runs only the peer kernels that build compiles for it
    for (size_t i = 1; c.mode == GpuMode::Peer && !m.peer_ada && i < c.cards.size(); ++i) {
        const int k = c.cards[i];
        if (k >= 0 && (size_t) k < visible.size() && visible[(size_t) k].cc_major == 8 && visible[(size_t) k].cc_minor == 9)
            return "gpu-mode peer: " + gpu_label(visible, k) + " is " + arch_name(visible[(size_t) k]) + "; " + model + " runs peer on " + core::cuda_cc_name() + " cards";
    }
    return {};
}

std::vector<GpuCard> visible_gpus(bool& known) {
    const NvmlCards nv = read_nvml();
    const char* env = std::getenv("CUDA_VISIBLE_DEVICES");
    if (env == nullptr) {
        known = nv.ok;
        return nv.cards;
    }
    known = true;
    std::vector<GpuCard> out;
    // CUDA reads the list up to the first entry it cannot use; "-1" or an empty list hides every card
    for (const std::string& w : split_commas(env)) {
        if (w.empty() || w == "-1") break;
        GpuCard c;
        c.token = w;
        if (all_digits(w)) {
            const size_t i = (size_t) std::atoi(w.c_str());
            if (i < nv.cards.size()) c = nv.cards[i];
            else c.smi = (int) i;
            c.token = w;
        } else {
            for (size_t i = 0; i < nv.uuids.size(); ++i)
                if (!nv.uuids[i].empty() && nv.uuids[i].rfind(w, 0) == 0) {
                    c = nv.cards[i];
                    c.token = w;
                    break;
                }
        }
        out.push_back(c);
    }
    return out;
}

void apply_gpus(const std::vector<GpuCard>& visible, const std::vector<int>& cards) {
    std::string list;
    for (int i : cards) {
        if (!list.empty()) list += ',';
        list += (i >= 0 && (size_t) i < visible.size()) ? visible[(size_t) i].token : std::to_string(i);
    }
    // numbers are nvidia-smi's, which is the PCI bus order; CUDA's own default order differs
    basalt::platform::set_env("CUDA_DEVICE_ORDER", "PCI_BUS_ID");
    basalt::platform::set_env("CUDA_VISIBLE_DEVICES", list);
}

// a card named only by its index
GpuCard unknown_card(int i) {
    GpuCard c;
    c.token = std::to_string(i);
    return c;
}

std::string gpu_identity(size_t n) {
    std::string s;
    for (size_t i = 0; i < n; ++i) s += (i ? "," : "") + std::to_string(i);
    return s;
}

std::string gpu_plan_json(const std::vector<GpuCard>& visible, const GpuChoice& c, const std::string& arch,
                          const std::string& quality_flags, int quality_window, int spec, int64_t max_context,
                          const std::string& model_name, const std::string& lossy, const std::string& community_kv) {
    std::string devices, cards;
    for (int i : c.cards) {
        const GpuCard g = (size_t) i < visible.size() ? visible[(size_t) i] : unknown_card(i);
        devices += (devices.empty() ? "" : ",") + g.token;
        cards += std::string(cards.empty() ? "" : ",") + "{\"visible\":" + std::to_string(i) + ",\"token\":" +
                 json_str(g.token) + ",\"smi\":" + std::to_string(g.smi) + ",\"name\":" + json_str(g.name) + "}";
    }
    std::string notes, shares;
    for (const std::string& s : c.notes) notes += (notes.empty() ? "" : ",") + json_str(s);
    for (double v : c.shares) {
        char b[32];
        std::snprintf(b, sizeof b, "%.6g", v);
        shares += (shares.empty() ? "" : ",") + std::string(b);
    }
    return "{" + (arch.empty() ? std::string() : "\"arch\":" + json_str(arch) + ",\"model\":" + json_str(model_name) + ",\"community_kv\":" + json_str(community_kv) + ",\"quality_flags\":" + json_str(quality_flags) +
                                     ",\"quality_window\":\"" + std::to_string(quality_window) + "\",\"lossy\":" + json_str(lossy) + ",\"spec\":" +
                                     std::to_string(spec) + ",\"max_context\":" + std::to_string(max_context) + ",") + "\"devices\":" + json_str(devices) + ",\"cards\":[" + cards + "],\"mode\":" + json_str(gpu_mode_name(c.mode)) + ",\"shares\":[" + shares + "],\"notes\":[" + notes +
           "],\"warning\":" + json_str(c.warning) + ",\"error\":" + json_str(c.error) + "}";
}

bool select_gpus(const std::string& text, const std::string& mode_text, const std::string& share_text, const char* label,
                 std::string& engine_value, std::vector<GpuCard>& used, std::string& err) {
    GpuSpec spec;
    GpuMode mode = GpuMode::One;
    if (!parse_gpu_spec(text, spec, err) || !parse_gpu_mode(mode_text, mode, err)) return false;
    bool known = false;
    const std::vector<GpuCard> visible = visible_gpus(known);
    const GpuChoice c = choose_gpus(spec, mode, share_text, visible, known);
    if (!c.error.empty()) { err = c.error; return false; }
    for (const std::string& s : c.notes) basalt::log::info(label, "%s", s.c_str());
    if (!c.warning.empty()) basalt::log::warn(label, "%s", c.warning.c_str());
    apply_gpus(visible, c.cards);
    used.clear();
    for (int i : c.cards) used.push_back((size_t) i < visible.size() ? visible[(size_t) i] : unknown_card(i));
    engine_value = gpu_identity(c.cards.size());
    return true;
}

}  // namespace basalt::program
