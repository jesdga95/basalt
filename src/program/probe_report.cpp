// src/program/probe_report.cpp - see probe_report.hpp.
#include "basalt/program/probe_report.hpp"

#include <algorithm>
#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <sstream>

namespace basalt::program {

const char* const kProbeDisclaimer = "Measured on this machine. Hand-tuned settings can still beat these.";

namespace {

std::string fmt(const char* f, ...) {
    va_list ap;
    va_start(ap, f);
    va_list ap2;
    va_copy(ap2, ap);
    const int n = std::vsnprintf(nullptr, 0, f, ap);
    va_end(ap);
    std::string s((size_t) std::max(n, 0), '\0');
    std::vsnprintf(s.data(), s.size() + 1, f, ap2);
    va_end(ap2);
    return s;
}

// key=value words of a line; a value in double quotes may hold spaces.
std::map<std::string, std::string> kv_words(const std::string& line) {
    std::map<std::string, std::string> m;
    size_t i = 0;
    while (i < line.size()) {
        while (i < line.size() && line[i] == ' ') ++i;
        const size_t eq = line.find('=', i);
        const size_t sp = line.find(' ', i);
        if (eq == std::string::npos || (sp != std::string::npos && sp < eq)) {
            if (sp == std::string::npos) break;
            i = sp + 1;
            continue;
        }
        const std::string key = line.substr(i, eq - i);
        i = eq + 1;
        std::string val;
        if (i < line.size() && line[i] == '"') {
            const size_t q = line.find('"', i + 1);
            val = line.substr(i + 1, q == std::string::npos ? std::string::npos : q - i - 1);
            i = q == std::string::npos ? line.size() : q + 1;
        } else {
            const size_t e = line.find(' ', i);
            val = line.substr(i, e == std::string::npos ? std::string::npos : e - i);
            i = e == std::string::npos ? line.size() : e;
        }
        m[key] = val;
    }
    return m;
}

double num(const std::map<std::string, std::string>& m, const char* k) {
    const auto it = m.find(k);
    return it == m.end() ? 0.0 : std::atof(it->second.c_str());
}

}  // namespace

std::string short_gpu_name(const std::string& name) {
    std::istringstream is(name);
    std::string w, out;
    while (is >> w) {
        if (w == "NVIDIA" || w == "GeForce" || w == "RTX") continue;
        out += (out.empty() ? "" : " ") + w;
    }
    return out.empty() ? name : out;
}

bool parse_probe_rates(const std::vector<std::string>& lines, ProbeRates& out) {
    out = ProbeRates{};
    std::map<int, std::string> names;
    std::map<int, double> mem, h2d;
    double active = 0, ram_last = 0, ram_at_active = 0;
    for (const auto& l : lines) {
        if (l.rfind("PROBE ", 0) != 0) continue;
        const size_t sp = l.find(' ', 6);
        const std::string section = l.substr(6, sp == std::string::npos ? std::string::npos : sp - 6);
        const auto kv = kv_words(l);
        const int dev = (int) num(kv, "dev");
        if (section == "gpu" && kv.count("name")) names[dev] = short_gpu_name(kv.at("name"));
        else if (section == "gpu_mem") mem[dev] = num(kv, "copy_gbps");
        else if (section == "link" && !h2d.count(dev)) h2d[dev] = num(kv, "h2d_gbps");
        else if (section == "host" && kv.count("active_workers")) {
            active = num(kv, "active_workers");
            out.active_workers = (int) active;
            out.pool_workers = (int) num(kv, "pool_workers");
        } else if (section == "cpu_expert" && kv.count("per_entry_us"))
            out.cpu_expert.push_back({(int) num(kv, "workers"), (int) num(kv, "entries"), num(kv, "per_entry_us")});
        else if (section == "host_mem") {
            ram_last = num(kv, "gbps");
            if (active > 0 && num(kv, "workers") == active) ram_at_active = ram_last;
        }
    }
    for (const auto& kvp : names) {
        ProbeRates::Gpu g;
        g.name = kvp.second;
        g.mem_gbps = mem.count(kvp.first) ? mem[kvp.first] : 0;
        out.gpus.push_back(g);
    }
    if (!names.empty() && h2d.count(names.begin()->first)) out.pcie_gbps = h2d[names.begin()->first];
    out.ram_gbps = ram_at_active > 0 ? ram_at_active : ram_last;
    return !out.gpus.empty();
}

std::string format_context(long long tokens) {
    if (tokens > 0 && tokens % 1024 == 0) return fmt("%lldk", tokens / 1024);
    return std::to_string(tokens);
}

std::string format_report(const ReportInput& in) {
    std::string o = in.command_line + "\n";
    std::string l1 = "  ";
    for (const auto& g : in.rates.gpus) l1 += fmt("%s %.0f GB/s | ", g.name.c_str(), g.mem_gbps);
    l1 += fmt("PCIe %.1f GB/s | RAM %.0f GB/s", in.rates.pcie_gbps, in.rates.ram_gbps);
    o += l1 + "\n";
    o += "  " + in.experts_line + " | KV " + in.kv + " | ctx " + format_context(in.engine_context) + "\n";

    o += "  decode cold ";
    for (const char* name : {"prose", "conv", "struct"})
        for (const auto& f : in.speeds)
            if (f.name == name) o += fmt(" %s %.0f", name, f.cold);
    o += " tok/s (measured)\n";
    if (!in.adapt_line.empty()) o += "  adapt test  " + in.adapt_line + "\n";
    for (const auto& s : in.suggested) o += "  suggested: " + s + "\n";
    o += "  config: " + in.config + "\n";
    if (in.wall_s >= 0) o += fmt("  wall  %.1f s\n\n", in.wall_s);
    o += std::string("  ") + kProbeDisclaimer + "\n";
    return o;
}

}  // namespace basalt::program
