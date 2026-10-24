// src/program/community_row.cpp - appends a community benchmark row to docs/community/benchmarks.json.
#include "basalt/program/community_row.hpp"

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
        t += "          \"power_w\": " + num(g.power_w, 0) + "\n";
        t += i + 1 < r.gpus.size() ? "        },\n" : "        }\n";
    }
    t += "      ],\n";
    t += "      \"ram_gib\": " + num(r.ram_gib, 0) + ",\n";
    t += "      \"os\": " + json_quote(r.os) + ",\n";
    t += "      \"quant\": " + json_quote(r.quant) + ",\n";
    t += "      \"prefill_16k_tps\": " + num(r.prefill_16k_tps, 1) + ",\n";
    t += "      \"prefill_64k_tps\": " + num(r.prefill_64k_tps, 1) + ",\n";
    t += "      \"decode_tps\": " + num(r.decode_tps, 1);
    if (!r.notes.empty()) t += ",\n      \"notes\": " + json_quote(r.notes);
    t += "\n    }";
    return t;
}

bool community_add_row(const std::string& file, const CommunityRow& row, std::string& out, std::string& err) {
    static const std::string tail = "\n    }\n  ]\n}", empty = "\n  \"runs\": []\n}", close = "\n  ]\n}";
    const auto ends = [&](const std::string& s) { return file.size() >= s.size() && file.compare(file.size() - s.size(), s.size(), s) == 0; };
    if (ends(tail)) {
        out = file.substr(0, file.size() - close.size()) + ",\n" + community_row_text(row) + close;
        return true;
    }
    if (ends(empty)) {
        out = file.substr(0, file.size() - 4) + "[\n" + community_row_text(row) + close;
        return true;
    }
    err = "the file does not end in the canonical runs list (run python3 tools/check_benchmarks.py --fix on it first)";
    return false;
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
