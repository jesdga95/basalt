// src/program/community_row.cpp - appends a community benchmark row to docs/community/benchmarks.json.
#include "basalt/program/community_row.hpp"

#include <cctype>
#include <cstdio>

#include "basalt/platform/file.hpp"

namespace basalt::program {

std::string json_quote(const std::string& s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '\"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\t': o += "\\t"; break;
            case '\r': o += "\\r"; break;
            default:
                if (c < 0x20) {
                    char b[8];
                    std::snprintf(b, sizeof b, "\\u%04x", (unsigned) c);
                    o += b;
                } else {
                    o += (char) c;
                }
        }
    }
    return o + "\"";
}

namespace {
std::string num(double v, int decimals) {
    char b[64];
    std::snprintf(b, sizeof b, "%.*f", decimals, v);
    return b;
}
}  // namespace

std::string cpu_label(const std::string& raw_name, int physical_cores) {
    std::string n = raw_name;
    const auto erase_all = [&](const std::string& what) {
        for (size_t p; (p = n.find(what)) != std::string::npos;) n.erase(p, what.size());
    };
    for (const char* m : {"(R)", "(TM)", "(r)", "(tm)", " with Radeon Graphics", " Processor"}) erase_all(m);
    if (const size_t at = n.find(" @ "); at != std::string::npos) n.erase(at);   // the clock
    for (size_t p = 0; (p = n.find("-Core", p)) != std::string::npos;) {   // "16-Core"
        size_t b = p;
        while (b > 0 && std::isdigit((unsigned char) n[b - 1])) --b;
        n.erase(b, p + 5 - b);
        p = b;
    }
    if (n.size() >= 4 && n.compare(n.size() - 4, 4, " CPU") == 0) n.erase(n.size() - 4);
    std::string out;
    for (char c : n) {   // single spaces, no edge spaces
        if (std::isspace((unsigned char) c)) {
            if (!out.empty() && out.back() != ' ') out += ' ';
        } else {
            out += c;
        }
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    if (out.empty()) return out;
    if (physical_cores > 0) out += ", " + std::to_string(physical_cores) + (physical_cores == 1 ? " core" : " cores");
    return out;
}

std::string community_row_text(const CommunityRow& r) {
    std::string t = "    {\n";
    t += "      \"submitter\": " + json_quote(r.submitter) + ",\n";
    t += "      \"date\": " + json_quote(r.date) + ",\n";
    t += "      \"commit\": " + json_quote(r.commit) + ",\n";
    t += "      \"preset\": " + json_quote(r.preset) + ",\n";
    t += "      \"gpus\": [\n";
    for (size_t i = 0; i < r.gpus.size(); ++i) {
        const CommunityGpu& g = r.gpus[i];
        t += "        {\n";
        t += "          \"name\": " + json_quote(g.name) + ",\n";
        t += "          \"vram_gib\": " + num(g.vram_gib, 0) + ",\n";
        t += "          \"power_w\": " + num(g.power_w, 0);
        if (!g.pcie.empty()) t += ",\n          \"pcie\": " + json_quote(g.pcie);
        if (g.h2d_gbps > 0) t += ",\n          \"h2d_gbps\": " + num(g.h2d_gbps, 1);
        t += "\n";
        t += i + 1 < r.gpus.size() ? "        },\n" : "        }\n";
    }
    t += "      ],\n";
    t += "      \"ram_gib\": " + num(r.ram_gib, 0) + ",\n";
    if (!r.cpu.empty()) t += "      \"cpu\": " + json_quote(r.cpu) + ",\n";
    if (r.ram_gbps > 0) t += "      \"ram_gbps\": " + num(r.ram_gbps, 1) + ",\n";
    t += "      \"os\": " + json_quote(r.os) + ",\n";
    t += "      \"quant\": " + json_quote(r.quant) + ",\n";
    t += "      \"prefill_16k_tps\": " + num(r.prefill_16k_tps, 1) + ",\n";
    t += "      \"prefill_64k_tps\": " + num(r.prefill_64k_tps, 1) + ",\n";
    t += "      \"decode_tps\": " + num(r.decode_tps, 1);
    if (!r.notes.empty()) t += ",\n      \"notes\": " + json_quote(r.notes);
    t += "\n    }";
    return t;
}

bool community_add_row(const std::string& file_in, const CommunityRow& row, std::string& out, std::string& err) {
    // The canonical form is LF; a file with CRLF line endings (a checkout that translated them) is edited as LF and
    // written back as CRLF. Raw CR never occurs inside the JSON text itself, so the conversion is lossless.
    const bool crlf = file_in.find("\r\n") != std::string::npos;
    std::string file;
    if (crlf) {
        file.reserve(file_in.size());
        for (size_t i = 0; i < file_in.size(); ++i)
            if (file_in[i] != '\r' || i + 1 >= file_in.size() || file_in[i + 1] != '\n') file += file_in[i];
    } else {
        file = file_in;
    }
    static const std::string tail = "\n    }\n  ]\n}", empty = "\n  \"runs\": []\n}", close = "\n  ]\n}";
    const auto ends = [&](const std::string& s) { return file.size() >= s.size() && file.compare(file.size() - s.size(), s.size(), s) == 0; };
    if (ends(tail)) {
        out = file.substr(0, file.size() - close.size()) + ",\n" + community_row_text(row) + close;
    } else if (ends(empty)) {
        out = file.substr(0, file.size() - 4) + "[\n" + community_row_text(row) + close;
    } else {
        err = "the file does not end in the canonical runs list (run python3 tools/check_benchmarks.py --fix on it first)";
        return false;
    }
    if (crlf) {
        std::string t;
        t.reserve(out.size() + out.size() / 20);
        for (char ch : out) {
            if (ch == '\n') t += '\r';
            t += ch;
        }
        out.swap(t);
    }
    return true;
}

bool community_add_row_file(const std::string& path, const CommunityRow& row, std::string& err) {
    std::string text;
    {
        std::FILE* f = platform::open_file_utf8(path, "rb");
        if (!f) { err = "cannot open " + path; return false; }
        char buf[1 << 14];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
        const bool bad = std::ferror(f) != 0;
        std::fclose(f);
        if (bad) { err = "cannot read " + path; return false; }
    }
    std::string out;
    if (!community_add_row(text, row, out, err)) return false;
    std::FILE* f = platform::open_file_utf8(path, "wb");
    if (!f) { err = "cannot write " + path; return false; }
    const bool ok = std::fwrite(out.data(), 1, out.size(), f) == out.size();
    if (std::fclose(f) != 0 || !ok) { err = "cannot write " + path; return false; }
    return true;
}

}  // namespace basalt::program
