// src/program/gpu_select.cpp - see gpu_select.hpp.
#include "basalt/program/gpu_select.hpp"

#include "basalt/log.hpp"
#include "basalt/platform/dynlib.hpp"
#include "basalt/platform/process.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
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

bool parse_gpu_spec(const std::string& text, GpuSpec& out, std::string& err) {
    out = GpuSpec{};
    const std::string t = trim(text);
    if (t.empty()) return true;
    if (t == "auto") { out.kind = GpuSpec::Kind::Auto; return true; }
    out.kind = GpuSpec::Kind::List;
    for (const std::string& w : split_commas(t)) {
        if (!all_digits(w)) {
            err = "gpu " + t + ": expected auto or GPU numbers like 0 or 0,1";
            return false;
        }
        const int v = std::atoi(w.c_str());
        if (std::find(out.list.begin(), out.list.end(), v) != out.list.end()) {
            err = "gpu " + t + ": GPU " + w + " is listed twice";
            return false;
        }
        out.list.push_back(v);
    }
    return true;
}

bool gpu_supported(const GpuCard& c) { return c.cc_major == 0 || (c.cc_major == 12 && c.cc_minor == 0); }

std::string gpu_label(const std::vector<GpuCard>& visible, int index) {
    std::string s = "GPU " + std::to_string(index);
    if (index >= 0 && (size_t) index < visible.size() && !visible[(size_t) index].name.empty())
        s += " (" + visible[(size_t) index].name + ")";
    return s;
}

GpuChoice choose_gpus(const GpuSpec& spec, const std::vector<GpuCard>& visible, bool known) {
    GpuChoice c;
    const int n = (int) visible.size();
    auto unsupported = [&](int i) {
        return "gpu: " + gpu_label(visible, i) + " is sm_" + std::to_string(visible[(size_t) i].cc_major) +
               std::to_string(visible[(size_t) i].cc_minor) + ", not sm_120";
    };
    if (spec.kind == GpuSpec::Kind::Auto) {
        if (!known) {
            c.error = "gpu auto: GPUs not listable (no NVML)";
            return c;
        }
        std::vector<int> ok;
        for (int i = 0; i < n; ++i) {
            if (gpu_supported(visible[(size_t) i])) ok.push_back(i);
            else c.notes.push_back("gpu auto: " + gpu_label(visible, i) + " skipped, not sm_120");
        }
        if (ok.empty()) {
            c.error = n == 0 ? "gpu auto: no GPU is visible" : "gpu auto: no sm_120 GPU visible";
            return c;
        }
        // the main card: the most memory bandwidth, then the most VRAM, then the lowest number
        int best = ok[0];
        for (int i : ok) {
            const GpuCard& a = visible[(size_t) i];
            const GpuCard& b = visible[(size_t) best];
            if (a.mem_gbps > b.mem_gbps || (a.mem_gbps == b.mem_gbps && a.vram > b.vram)) best = i;
        }
        c.cards.push_back(best);
        for (int i : ok)
            if (i != best) c.cards.push_back(i);
        return c;
    }
    if (spec.kind == GpuSpec::Kind::Unset) {
        if (known && n == 0) { c.error = "gpu: no GPU is visible"; return c; }
        c.cards.push_back(0);
        if (known && !gpu_supported(visible[0])) {
            c.error = unsupported(0);
            return c;
        }
        std::vector<int> others;
        for (int i = 1; i < n; ++i)
            if (gpu_supported(visible[(size_t) i])) others.push_back(i);
        if (!others.empty()) {
            std::string names, list = "0";
            for (int i : others) {
                names += (names.empty() ? "" : ", ") + gpu_label(visible, i);
                list += "," + std::to_string(i);
            }
            c.warning = names + " unused; use gpu " + list + " or gpu auto";
        }
        return c;
    }
    for (int i : spec.list) {
        if (known && i >= n) {
            c.error = "gpu " + std::to_string(i) + ": " +
                      (n == 0 ? std::string("no GPU is visible")
                              : std::to_string(n) + (n == 1 ? " GPU is" : " GPUs are") + " visible (0-" +
                                    std::to_string(n - 1) + ")");
            c.cards.clear();
            return c;
        }
        if (known && !gpu_supported(visible[(size_t) i])) {
            c.error = unsupported(i);
            c.cards.clear();
            return c;
        }
        c.cards.push_back(i);
    }
    return c;
}

const char* gpu_role_name(GpuRole r) {
    return r == GpuRole::Main ? "main" : r == GpuRole::ExpertTier ? "expert tier" : "idle";
}

std::vector<GpuRole> choose_roles(const RoleInputs& in) {
    std::vector<GpuRole> r(in.cards.size(), GpuRole::Idle);
    if (r.empty()) return r;
    r[0] = GpuRole::Main;
    if (!in.peer_access) return r;
    int tiers = 0;
    for (size_t i = 1; i < r.size() && tiers < in.max_expert_tiers; ++i, ++tiers) r[i] = GpuRole::ExpertTier;
    return r;
}

std::vector<int> spread_experts(const std::vector<uint64_t>& bytes, const std::vector<TierRoom>& tiers) {
    std::vector<int> owner(bytes.size(), -1);
    const size_t n = tiers.size();
    std::vector<uint64_t> used(n, 0);
    std::vector<int64_t> taken(n, 0);
    std::vector<bool> open(n, true);
    for (size_t i = 0; i < bytes.size(); ++i) {
        // the least loaded open tier takes it; one it no longer fits closes, and the next is asked
        for (;;) {
            int best = -1;
            double best_load = 0;
            for (size_t t = 0; t < n; ++t) {
                if (!open[t]) continue;
                const double load = (double) taken[t] / (tiers[t].gbps > 0 ? tiers[t].gbps : 1.0);
                if (best < 0 || load < best_load) { best = (int) t; best_load = load; }
            }
            if (best < 0) return owner;
            const size_t b = (size_t) best;
            if (used[b] + bytes[i] > tiers[b].bytes) { open[b] = false; continue; }
            owner[i] = best;
            used[b] += bytes[i];
            ++taken[b];
            break;
        }
    }
    return owner;
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

std::string gpu_identity(size_t n) {
    std::string s;
    for (size_t i = 0; i < n; ++i) s += (i ? "," : "") + std::to_string(i);
    return s;
}

std::string gpu_plan_json(const std::vector<GpuCard>& visible, const GpuChoice& c) {
    std::string devices, cards;
    for (int i : c.cards) {
        const GpuCard g = (size_t) i < visible.size() ? visible[(size_t) i] : GpuCard{std::to_string(i)};
        devices += (devices.empty() ? "" : ",") + g.token;
        cards += std::string(cards.empty() ? "" : ",") + "{\"visible\":" + std::to_string(i) + ",\"token\":" +
                 json_str(g.token) + ",\"smi\":" + std::to_string(g.smi) + ",\"name\":" + json_str(g.name) + "}";
    }
    std::string notes;
    for (const std::string& s : c.notes) notes += (notes.empty() ? "" : ",") + json_str(s);
    return "{\"devices\":" + json_str(devices) + ",\"cards\":[" + cards + "],\"notes\":[" + notes +
           "],\"warning\":" + json_str(c.warning) + ",\"error\":" + json_str(c.error) + "}";
}

bool select_gpus(const std::string& text, const char* label, std::string& engine_value, std::vector<GpuCard>& used,
                 std::string& err) {
    GpuSpec spec;
    if (!parse_gpu_spec(text, spec, err)) return false;
    bool known = false;
    const std::vector<GpuCard> visible = visible_gpus(known);
    const GpuChoice c = choose_gpus(spec, visible, known);
    if (!c.error.empty()) { err = c.error; return false; }
    for (const std::string& s : c.notes) basalt::log::info(label, "%s", s.c_str());
    if (!c.warning.empty()) basalt::log::warn(label, "%s", c.warning.c_str());
    apply_gpus(visible, c.cards);
    used.clear();
    for (int i : c.cards) used.push_back((size_t) i < visible.size() ? visible[(size_t) i] : GpuCard{std::to_string(i)});
    engine_value = gpu_identity(c.cards.size());
    return true;
}

}  // namespace basalt::program
