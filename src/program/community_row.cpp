// src/program/community_row.cpp - one community run: the row text, the id and folder of its file, and the write of that file.
#include "basalt/program/community_row.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
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
    const auto word_ends = [&n](size_t i) { return i == n.size() || !std::isalnum((unsigned char) n[i]); };
    for (size_t p = 0; (p = n.find("-Core", p)) != std::string::npos;) {   // "16-Core"
        size_t b = p, e = p + 5;
        while (b > 0 && std::isdigit((unsigned char) n[b - 1])) --b;
        if (b < p) {   // a count tag is erased only when it ends the word, a plural "16-Cores" too
            if (e < n.size() && n[e] == 's' && word_ends(e + 1)) {
                ++e;
            } else if (!word_ends(e)) {
                p = e;
                continue;
            }
        }
        n.erase(b, e - b);
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
    size_t b = 0, e = name.size();
    while (b < e && std::isspace((unsigned char) name[b])) ++b;
    while (e > b && std::isspace((unsigned char) name[e - 1])) --e;
    const std::string t = name.substr(b, e - b);
    const size_t sp = t.rfind(' ');
    if (sp == std::string::npos) return "unknown";   // a name without a space carries no quant label
    std::string q = t.substr(sp + 1);
    const auto strip = [&q](const char* pre) {   // the prefix in any case, the rest of the label as it is written
        const size_t n = std::strlen(pre);
        if (q.size() < n) return;
        for (size_t i = 0; i < n; ++i)
            if (std::tolower((unsigned char) q[i]) != std::tolower((unsigned char) pre[i])) return;
        q.erase(0, n);
    };
    for (size_t before = 0; q.size() != before;) {   // repeatedly, in any order, until neither prefix matches
        before = q.size();
        for (const char* pre : {"NVIDIA-", "UD-"}) strip(pre);
    }
    // a quant label starts with a letter and carries a digit (NVFP4, Q4_K_M, IQ1_M, BF16); a last word without a digit
    // ("Mini") or starting with one (a size or a version: "12B", "4") is part of the model's name
    const bool digit = std::any_of(q.begin(), q.end(), [](char c) { return std::isdigit((unsigned char) c) != 0; });
    const auto all_digits = [](const std::string& s, size_t from, size_t to) {
        if (from >= to) return false;
        for (size_t i = from; i < to; ++i)
            if (std::isdigit((unsigned char) s[i]) == 0) return false;
        return true;
    };
    // an active-parameter tag ("A4B", the 26B A4B's) is a size too; a bit width ("8bit", "4bit") is a label
    const bool active = q.size() >= 3 && (q.front() == 'A' || q.front() == 'a') && (q.back() == 'B' || q.back() == 'b') &&
                        all_digits(q, 1, q.size() - 1);
    size_t k = 0;
    while (k < q.size() && std::isdigit((unsigned char) q[k]) != 0) ++k;
    const bool bits = k > 0 && q.size() == k + 3 && (q.compare(k, 3, "bit") == 0 || q.compare(k, 3, "BIT") == 0 || q.compare(k, 3, "Bit") == 0);
    const bool letter = !q.empty() && std::isalpha((unsigned char) q.front()) != 0;
    return digit && !active && (letter || bits) ? q : "unknown";
}

std::string community_row_text(const CommunityRow& r) {
    std::string t = "{\n";
    t += "  \"submitter\": " + json_quote(r.submitter) + ",\n";
    t += "  \"date\": " + json_quote(r.date) + ",\n";
    t += "  \"commit\": " + json_quote(r.commit) + ",\n";
    t += "  \"preset\": " + json_quote(r.preset) + ",\n";
    t += "  \"model\": " + json_quote(r.model) + ",\n";
    t += "  \"gpus\": [\n";
    for (size_t i = 0; i < r.gpus.size(); ++i) {
        const CommunityGpu& g = r.gpus[i];
        t += "    {\n";
        t += "      \"name\": " + json_quote(g.name) + ",\n";
        t += "      \"vram_gib\": " + num(g.vram_gib, 0) + ",\n";
        t += "      \"power_w\": " + num(g.power_w, 0);
        if (!g.pcie.empty()) t += ",\n      \"pcie\": " + json_quote(g.pcie);
        if (g.h2d_gbps > 0) t += ",\n      \"h2d_gbps\": " + num(g.h2d_gbps, 1);
        t += "\n";
        t += i + 1 < r.gpus.size() ? "    },\n" : "    }\n";
    }
    t += "  ],\n";
    t += "  \"ram_gib\": " + num(r.ram_gib, 0) + ",\n";
    if (!r.cpu.empty()) t += "  \"cpu\": " + json_quote(r.cpu) + ",\n";
    if (r.ram_gbps > 0) t += "  \"ram_gbps\": " + num(r.ram_gbps, 1) + ",\n";
    t += "  \"os\": " + json_quote(r.os) + ",\n";
    t += "  \"quant\": " + json_quote(r.quant) + ",\n";
    t += "  \"kv\": " + json_quote(r.kv) + ",\n";
    t += "  \"prefill_16k_tps\": " + num(r.prefill_16k_tps, 1) + ",\n";
    t += "  \"prefill_64k_tps\": " + num(r.prefill_64k_tps, 1) + ",\n";
    t += "  \"decode_tps\": " + num(r.decode_tps, 1);
    for (int k = 0; k < 3; ++k)
        if (r.tests[k].decode_tps > 0) t += std::string(",\n  \"decode_") + kCommunityTestNames[k] + "_tps\": " + num(r.tests[k].decode_tps, 1);
    for (int k = 0; k < 3; ++k)
        if (r.tests[k].draft_len > 0) t += std::string(",\n  \"accept_") + kCommunityTestNames[k] + "_pct\": " + num(r.tests[k].accept_pct, 1);
    for (int k = 0; k < 3; ++k)
        if (r.tests[k].draft_len > 0) t += std::string(",\n  \"draft_len_") + kCommunityTestNames[k] + "\": " + num(r.tests[k].draft_len, 1);
    if (!r.engine_args.empty()) t += ",\n  \"engine_args\": " + json_quote(r.engine_args);
    if (!r.notes.empty()) t += ",\n  \"notes\": " + json_quote(r.notes);
    t += "\n}";
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

}  // namespace

bool community_parse_row(const std::string& text, CommunityRow& row, std::string& err) {
    row = CommunityRow();
    const auto bad = [&](const char* why) { err = std::string("the row is not in the house form (") + why + ")"; return false; };
    // the house form: "{", keys at 2 spaces, the GPUs' "{" at 4 and their keys at 6, "}" at the start of the last line
    bool in_row = false, closed = false, in_gpus = false, in_gpu = false;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t eol = text.find('\n', pos);
        if (eol == std::string::npos) eol = text.size();
        std::string line = text.substr(pos, eol - pos);
        pos = eol + 1;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (closed) return bad("text after the closing }");
        if (!in_row) {
            if (line != "{") return bad("the row does not open with {");
            in_row = true;
            continue;
        }
        if (line == "}") { closed = true; continue; }
        size_t ind = 0;
        while (ind < line.size() && line[ind] == ' ') ++ind;
        const std::string body = line.substr(ind);
        if (ind == 2 && in_gpus && (body == "]" || body == "],")) { in_gpus = false; continue; }
        if (ind == 4 && in_gpus) {
            if (body == "{") { in_gpu = true; row.gpus.emplace_back(); continue; }
            if (body == "}" || body == "},") { in_gpu = false; continue; }
            return bad("a GPU does not open with {");
        }
        const size_t colon = body.find("\": ");
        if (body.size() < 5 || body[0] != '"' || colon == std::string::npos) return bad("a line is not a key and a value");
        const std::string key = body.substr(1, colon - 1);
        std::string val = body.substr(colon + 3);
        if (key == "gpus" && ind == 2) {
            if (val != "[") return bad("gpus does not open with [");
            in_gpus = true;
            continue;
        }
        if (!val.empty() && val.back() == ',') val.pop_back();
        std::string s;
        const bool is_text = !val.empty() && val.front() == '"';
        if (is_text && !json_unquote(val, s)) return bad("a string is not valid JSON");
        const double number = is_text ? 0.0 : std::strtod(val.c_str(), nullptr);
        if (in_gpu && ind == 6) {
            CommunityGpu& g = row.gpus.back();
            if (key == "name") g.name = s;
            else if (key == "vram_gib") g.vram_gib = number;
            else if (key == "power_w") g.power_w = number;
        } else if (ind == 2) {
            if (key == "submitter") row.submitter = s;
            else if (key == "model") row.model = s;
            else if (key == "quant") row.quant = s;
            else if (key == "os") row.os = s;
            else if (key == "cpu") row.cpu = s;
            else if (key == "notes") row.notes = s;
            else if (key == "ram_gib") row.ram_gib = number;
        }
    }
    return closed ? true : bad("the row is not closed");
}

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

std::string sha256_hex(const std::string& data) {
    static const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be,
        0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa,
        0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85,
        0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
        0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::string m = data;
    m += (char) 0x80;
    while (m.size() % 64 != 56) m += (char) 0;
    const uint64_t bits = (uint64_t) data.size() * 8;
    for (int i = 7; i >= 0; --i) m += (char) ((bits >> (8 * i)) & 0xff);
    const auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    for (size_t off = 0; off < m.size(); off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = ((uint32_t) (unsigned char) m[off + 4 * i] << 24) | ((uint32_t) (unsigned char) m[off + 4 * i + 1] << 16) |
                   ((uint32_t) (unsigned char) m[off + 4 * i + 2] << 8) | (uint32_t) (unsigned char) m[off + 4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t v[8];
        std::copy(h, h + 8, v);
        for (int i = 0; i < 64; ++i) {
            const uint32_t s1 = rotr(v[4], 6) ^ rotr(v[4], 11) ^ rotr(v[4], 25);
            const uint32_t t1 = v[7] + s1 + ((v[4] & v[5]) ^ (~v[4] & v[6])) + K[i] + w[i];
            const uint32_t s0 = rotr(v[0], 2) ^ rotr(v[0], 13) ^ rotr(v[0], 22);
            const uint32_t t2 = s0 + ((v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]));
            std::copy_backward(v, v + 7, v + 8);
            v[4] += t1;
            v[0] = t1 + t2;
        }
        for (int i = 0; i < 8; ++i) h[i] += v[i];
    }
    std::string out;
    char b[16];
    for (uint32_t x : h) {
        std::snprintf(b, sizeof b, "%08x", (unsigned) x);
        out += b;
    }
    return out;
}

std::string community_model_slug(const std::string& model) { return lower_ascii(model); }

std::string community_run_id(const CommunityRow& r) {
    std::string t = "[" + json_quote(lower_ascii(r.submitter)) + "," + json_quote(r.model) + "," + json_quote(r.quant) + "," + json_quote(r.cpu) +
                    "," + num(r.ram_gib, 0) + "," + json_quote(r.os) + ",[";
    for (size_t i = 0; i < r.gpus.size(); ++i)
        t += std::string(i ? "," : "") + "[" + json_quote(r.gpus[i].name) + "," + num(r.gpus[i].vram_gib, 0) + "," + num(r.gpus[i].power_w, 0) + "]";
    return sha256_hex(t + "]]").substr(0, 12);
}

bool community_write_run(const std::string& dir, const CommunityRow& row_in, std::string& err, CommunityEdit* edit) {
    CommunityRow row = row_in;
    const std::string folder = dir + "/runs/" + community_model_slug(row.model);
    const std::string path = folder + "/" + community_run_id(row) + ".json";
    if (std::FILE* f = platform::open_file_utf8(path, "rb")) {   // the earlier run of this setup: overwritten, its notes kept
        std::string text;
        char buf[1 << 14];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
        const bool bad = std::ferror(f) != 0;
        std::fclose(f);
        if (bad) { err = "cannot read " + path; return false; }
        CommunityRow old;
        if (!community_parse_row(text, old, err)) { err += " in " + path; return false; }
        if (row.notes.empty()) row.notes = old.notes;
        if (edit) {
            edit->replaced = true;
            edit->notes = old.notes;
        }
    }
    if (!platform::make_dirs(folder)) { err = "cannot create " + folder; return false; }
    const std::string text = community_row_text(row) + "\n";
    std::FILE* f = platform::open_file_utf8(path, "wb");
    if (!f) { err = "cannot write " + path; return false; }
    const bool ok = std::fwrite(text.data(), 1, text.size(), f) == text.size();
    if (std::fclose(f) != 0 || !ok) { err = "cannot write " + path; return false; }
    if (edit) {
        edit->path = path;
        edit->row_text = community_row_text(row);
    }
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
