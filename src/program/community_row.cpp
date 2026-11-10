// src/program/community_row.cpp - adds a community benchmark row to docs/community/benchmarks.json, or updates the row of the same setup.
#include "basalt/program/community_row.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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
            case '\b': o += "\\b"; break;
            case '\f': o += "\\f"; break;
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

std::string pack_quant_label(const std::string& name) {
    std::string q = name.substr(name.rfind(' ') == std::string::npos ? 0 : name.rfind(' ') + 1);
    for (const char* pre : {"NVIDIA-", "UD-"})
        if (q.rfind(pre, 0) == 0) q.erase(0, std::strlen(pre));
    return q.empty() ? "unknown" : q;
}

std::string community_row_text(const CommunityRow& r) {
    std::string t = "    {\n";
    t += "      \"submitter\": " + json_quote(r.submitter) + ",\n";
    t += "      \"date\": " + json_quote(r.date) + ",\n";
    t += "      \"commit\": " + json_quote(r.commit) + ",\n";
    t += "      \"preset\": " + json_quote(r.preset) + ",\n";
    t += "      \"model\": " + json_quote(r.model) + ",\n";
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
    t += "      \"kv\": " + json_quote(r.kv) + ",\n";
    t += "      \"prefill_16k_tps\": " + num(r.prefill_16k_tps, 1) + ",\n";
    t += "      \"prefill_64k_tps\": " + num(r.prefill_64k_tps, 1) + ",\n";
    t += "      \"decode_tps\": " + num(r.decode_tps, 1);
    for (int k = 0; k < 3; ++k)
        if (r.tests[k].decode_tps > 0) t += std::string(",\n      \"decode_") + kCommunityTestNames[k] + "_tps\": " + num(r.tests[k].decode_tps, 1);
    for (int k = 0; k < 3; ++k)
        if (r.tests[k].draft_len > 0) t += std::string(",\n      \"accept_") + kCommunityTestNames[k] + "_pct\": " + num(r.tests[k].accept_pct, 1);
    for (int k = 0; k < 3; ++k)
        if (r.tests[k].draft_len > 0) t += std::string(",\n      \"draft_len_") + kCommunityTestNames[k] + "\": " + num(r.tests[k].draft_len, 1);
    if (!r.engine_args.empty()) t += ",\n      \"engine_args\": " + json_quote(r.engine_args);
    if (!r.notes.empty()) t += ",\n      \"notes\": " + json_quote(r.notes);
    t += "\n    }";
    return t;
}

namespace {
std::string lower_ascii(std::string s) {
    for (char& c : s)
        if (c >= 'A' && c <= 'Z') c = (char) (c + 32);
    return s;
}

// A value the setup rule compares: empty or 0 is a field the row lacks, which matches anything.
bool same_text(const std::string& a, const std::string& b) { return a.empty() || b.empty() || a == b; }
bool same_num(double a, double b) { return a == 0 || b == 0 || a == b; }

void append_utf8(std::string& o, unsigned cp) {
    if (cp < 0x80) {
        o += (char) cp;
    } else if (cp < 0x800) {
        o += (char) (0xC0 | (cp >> 6));
        o += (char) (0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        o += (char) (0xE0 | (cp >> 12));
        o += (char) (0x80 | ((cp >> 6) & 0x3F));
        o += (char) (0x80 | (cp & 0x3F));
    } else {
        o += (char) (0xF0 | (cp >> 18));
        o += (char) (0x80 | ((cp >> 12) & 0x3F));
        o += (char) (0x80 | ((cp >> 6) & 0x3F));
        o += (char) (0x80 | (cp & 0x3F));
    }
}

// The text of a JSON string literal (with its quotes), as json.loads reads it. False when it is not one.
bool json_unquote(const std::string& lit, std::string& out) {
    out.clear();
    if (lit.size() < 2 || lit.front() != '"' || lit.back() != '"') return false;
    const auto hex4 = [&](size_t at, unsigned& v) {
        if (at + 4 > lit.size() - 1) return false;
        v = 0;
        for (size_t k = at; k < at + 4; ++k) {
            const char c = lit[k];
            if (!std::isxdigit((unsigned char) c)) return false;
            v = v * 16 + (unsigned) (c <= '9' ? c - '0' : (c | 32) - 'a' + 10);
        }
        return true;
    };
    for (size_t i = 1; i + 1 < lit.size(); ++i) {
        const char c = lit[i];
        if (c != '\\') { out += c; continue; }
        if (++i + 1 >= lit.size()) return false;
        switch (lit[i]) {
            case 'n': out += '\n'; break;
            case 't': out += '\t'; break;
            case 'r': out += '\r'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case '"': case '\\': case '/': out += lit[i]; break;
            case 'u': {
                unsigned v = 0, lo = 0;
                if (!hex4(i + 1, v)) return false;
                i += 4;
                if (v >= 0xD800 && v < 0xDC00 && i + 6 < lit.size() && lit[i + 1] == '\\' && lit[i + 2] == 'u' && hex4(i + 3, lo) &&
                    lo >= 0xDC00 && lo < 0xE000) {
                    v = 0x10000 + ((v - 0xD800) << 10) + (lo - 0xDC00);
                    i += 6;
                }
                append_utf8(out, v);
                break;
            }
            default: return false;
        }
    }
    return true;
}

struct ParsedRow {
    CommunityRow row;
    size_t begin = 0, end = 0;   // the row's text in the file: from its "    {" to its "    }"
};

// Reads the rows of a canonical file: a row opens with "    {", its keys sit at 6 spaces, its GPUs at 8 and theirs at 10.
bool parse_rows(const std::string& file, std::vector<ParsedRow>& rows, std::string& err) {
    rows.clear();
    static const std::string head = "\n  \"runs\": [";
    const size_t h = file.find(head);
    if (h == std::string::npos) { err = "the file has no runs list"; return false; }
    size_t pos = h + head.size();
    if (pos < file.size() && file[pos] == ']') return true;   // an empty list
    const auto bad = [&](const char* why) { err = std::string("the runs list is not in the canonical form (") + why + ")"; return false; };
    if (pos >= file.size() || file[pos] != '\n') return bad("after \"runs\": [");
    ++pos;
    bool in_row = false, in_gpus = false, in_gpu = false;
    ParsedRow cur;
    while (pos < file.size()) {
        size_t eol = file.find('\n', pos);
        if (eol == std::string::npos) eol = file.size();
        const std::string line = file.substr(pos, eol - pos);
        const size_t line_at = pos;
        pos = eol + 1;
        size_t ind = 0;
        while (ind < line.size() && line[ind] == ' ') ++ind;
        std::string body = line.substr(ind);
        if (!in_row) {
            if (line == "  ]") return true;
            if (line != "    {") return bad("a row does not open with 4 spaces and {");
            in_row = true;
            cur = ParsedRow();
            cur.begin = line_at;
            continue;
        }
        if (ind == 4) {
            if (body != "}" && body != "},") return bad("a row does not close with }");
            cur.end = line_at + 5;
            rows.push_back(cur);
            in_row = false;
            continue;
        }
        if (ind == 6 && in_gpus) {
            if (body != "]" && body != "],") return bad("gpus does not close with ]");
            in_gpus = false;
            continue;
        }
        if (ind == 8 && in_gpus) {
            if (body == "{") { in_gpu = true; cur.row.gpus.emplace_back(); continue; }
            if (body == "}" || body == "},") { in_gpu = false; continue; }
            return bad("a GPU does not open with {");
        }
        const size_t colon = body.find("\": ");
        if (body.size() < 5 || body[0] != '"' || colon == std::string::npos) return bad("a line is not a key and a value");
        const std::string key = body.substr(1, colon - 1);
        std::string val = body.substr(colon + 3);
        if (key == "gpus" && ind == 6) {
            if (val != "[") return bad("gpus does not open with [");
            in_gpus = true;
            continue;
        }
        if (!val.empty() && val.back() == ',') val.pop_back();
        std::string text;
        const bool is_text = !val.empty() && val.front() == '"';
        if (is_text && !json_unquote(val, text)) return bad("a string is not valid JSON");
        const double number = is_text ? 0.0 : std::strtod(val.c_str(), nullptr);
        if (in_gpu && ind == 10) {
            CommunityGpu& g = cur.row.gpus.back();
            if (key == "name") g.name = text;
            else if (key == "vram_gib") g.vram_gib = number;
            else if (key == "power_w") g.power_w = number;
        } else if (ind == 6) {
            CommunityRow& r = cur.row;
            if (key == "submitter") r.submitter = text;
            else if (key == "model") r.model = text;
            else if (key == "quant") r.quant = text;
            else if (key == "os") r.os = text;
            else if (key == "cpu") r.cpu = text;
            else if (key == "notes") r.notes = text;
            else if (key == "ram_gib") r.ram_gib = number;
        }
    }
    return bad("the list is not closed");
}
}  // namespace

bool community_same_setup(const CommunityRow& a, const CommunityRow& b) {
    if (!same_text(lower_ascii(a.submitter), lower_ascii(b.submitter)) || !same_text(a.model, b.model) || !same_text(a.quant, b.quant) ||
        !same_text(a.cpu, b.cpu) || !same_text(a.os, b.os) || !same_num(a.ram_gib, b.ram_gib))
        return false;
    if (a.gpus.empty() || b.gpus.empty()) return true;
    if (a.gpus.size() != b.gpus.size()) return false;
    for (size_t i = 0; i < a.gpus.size(); ++i)
        if (!same_text(a.gpus[i].name, b.gpus[i].name) || !same_num(a.gpus[i].vram_gib, b.gpus[i].vram_gib) ||
            !same_num(a.gpus[i].power_w, b.gpus[i].power_w))
            return false;
    return true;
}

bool community_parse_rows(const std::string& file, std::vector<CommunityRow>& rows, std::string& err) {
    std::vector<ParsedRow> parsed;
    if (!parse_rows(file, parsed, err)) return false;
    rows.clear();
    for (const ParsedRow& p : parsed) rows.push_back(p.row);
    return true;
}

bool community_add_row(const std::string& file_in, const CommunityRow& row_in, std::string& out, std::string& err, CommunityEdit* edit) {
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
    CommunityRow row = row_in;
    for (CommunityGpu& g : row.gpus) {   // compared as the file spells them
        g.vram_gib = std::strtod(num(g.vram_gib, 0).c_str(), nullptr);
        g.power_w = std::strtod(num(g.power_w, 0).c_str(), nullptr);
    }
    row.ram_gib = std::strtod(num(row.ram_gib, 0).c_str(), nullptr);
    if (ends(tail)) {
        std::vector<ParsedRow> old;
        if (!parse_rows(file, old, err)) return false;
        const auto same = std::find_if(old.begin(), old.end(), [&](const ParsedRow& p) { return community_same_setup(p.row, row); });
        if (same != old.end()) {
            if (row.notes.empty()) row.notes = same->row.notes;
            out = file.substr(0, same->begin) + community_row_text(row) + file.substr(same->end);
            if (edit) {
                edit->replaced = true;
                edit->index = (size_t) (same - old.begin());
                edit->notes = same->row.notes;
            }
        } else {
            out = file.substr(0, file.size() - close.size()) + ",\n" + community_row_text(row) + close;
        }
    } else if (ends(empty)) {
        out = file.substr(0, file.size() - 4) + "[\n" + community_row_text(row) + close;
    } else {
        err = "the file does not end in the canonical runs list (run python3 tools/check_benchmarks.py --fix on it first)";
        return false;
    }
    if (edit) edit->row_text = community_row_text(row);
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

bool community_add_row_file(const std::string& path, const CommunityRow& row, std::string& err, CommunityEdit* edit) {
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
    if (!community_add_row(text, row, out, err, edit)) return false;
    std::FILE* f = platform::open_file_utf8(path, "wb");
    if (!f) { err = "cannot write " + path; return false; }
    const bool ok = std::fwrite(out.data(), 1, out.size(), f) == out.size();
    if (std::fclose(f) != 0 || !ok) { err = "cannot write " + path; return false; }
    return true;
}

std::vector<int> engine_gpu_order(int peer_device, const std::vector<int>& peer_devices, const std::vector<int>& split_devices) {
    std::vector<int> out{0};
    auto add = [&out](int d) {
        if (d >= 0 && std::find(out.begin(), out.end(), d) == out.end()) out.push_back(d);
    };
    add(peer_device);
    for (int d : peer_devices) add(d);
    for (int d : split_devices) add(d);
    return out;
}

std::vector<std::string> split_gpu_ids(const std::string& csv) {
    std::vector<std::string> out;
    size_t a = 0;
    while (a <= csv.size()) {
        size_t b = csv.find(',', a);
        if (b == std::string::npos) b = csv.size();
        if (b > a) out.push_back(csv.substr(a, b - a));
        a = b + 1;
    }
    return out;
}

}  // namespace basalt::program
