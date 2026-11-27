// src/program/quality_suite.cpp - safetensors BF16 reference reader, Basalt score_top row parser,
// per-row KLD / top-1 / dp / PPL computation, and conversation-level bootstrap for basalt-bench --quality.
#include "quality_suite.hpp"

#include "basalt/platform/file.hpp"
#include "basalt/program/json_scan.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace basalt::program::quality {

namespace {

constexpr uint64_t kMaxHeaderBytes = 128ull * 1024ull * 1024ull;

struct FdGuard {
    int fd = -1;
    ~FdGuard() {
        if (fd >= 0) basalt::platform::close_fd(fd);
    }
};

float f16_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x03FFu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            uint32_t m = mant;
            int e = -14;
            while ((m & 0x0400u) == 0) {
                m <<= 1;
                --e;
            }
            m &= 0x03FFu;
            bits = sign | (static_cast<uint32_t>(e + 127) << 23) | (m << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

bool skip_json_value(basalt::program::json::Scan& sc);

bool skip_json_array(basalt::program::json::Scan& sc) {
    if (!sc.eat('[')) return false;
    if (sc.eat(']')) return true;
    for (;;) {
        if (!skip_json_value(sc)) return false;
        if (sc.eat(']')) return true;
        if (!sc.eat(',')) return false;
    }
}

bool skip_json_object(basalt::program::json::Scan& sc) {
    if (!sc.eat('{')) return false;
    if (sc.eat('}')) return true;
    for (;;) {
        std::string k;
        if (!sc.str(k) || !sc.eat(':') || !skip_json_value(sc)) return false;
        if (sc.eat('}')) return true;
        if (!sc.eat(',')) return false;
    }
}

bool skip_json_value(basalt::program::json::Scan& sc) {
    sc.ws();
    if (sc.i >= sc.s.size()) return false;
    const char c = sc.s[sc.i];
    if (c == '"') {
        std::string tmp;
        return sc.str(tmp);
    }
    if (c == '{') return skip_json_object(sc);
    if (c == '[') return skip_json_array(sc);
    if (c == 't' && sc.s.substr(sc.i, 4) == "true") { sc.i += 4; return true; }
    if (c == 'f' && sc.s.substr(sc.i, 5) == "false") { sc.i += 5; return true; }
    if (c == 'n' && sc.s.substr(sc.i, 4) == "null") { sc.i += 4; return true; }
    if (c == '-' || (c >= '0' && c <= '9')) {
        if (sc.s[sc.i] == '-') ++sc.i;
        const size_t start = sc.i;
        while (sc.i < sc.s.size() && sc.s[sc.i] >= '0' && sc.s[sc.i] <= '9') ++sc.i;
        if (sc.i == start) return false;
        if (sc.i < sc.s.size() && sc.s[sc.i] == '.') {
            ++sc.i;
            while (sc.i < sc.s.size() && sc.s[sc.i] >= '0' && sc.s[sc.i] <= '9') ++sc.i;
        }
        if (sc.i < sc.s.size() && (sc.s[sc.i] == 'e' || sc.s[sc.i] == 'E')) {
            ++sc.i;
            if (sc.i < sc.s.size() && (sc.s[sc.i] == '+' || sc.s[sc.i] == '-')) ++sc.i;
            const size_t es = sc.i;
            while (sc.i < sc.s.size() && sc.s[sc.i] >= '0' && sc.s[sc.i] <= '9') ++sc.i;
            if (sc.i == es) return false;
        }
        return true;
    }
    return false;
}

struct TensorInfo {
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t off_begin = 0;
    uint64_t off_end = 0;
};

bool parse_tensor_info(basalt::program::json::Scan& sc, TensorInfo& info) {
    if (!sc.eat('{')) return false;
    bool has_dtype = false, has_shape = false, has_off = false;
    if (!sc.eat('}')) {
        for (;;) {
            std::string k;
            if (!sc.str(k) || !sc.eat(':')) return false;
            if (k == "dtype") {
                if (!sc.str(info.dtype)) return false;
                has_dtype = true;
            } else if (k == "shape") {
                if (!sc.eat('[')) return false;
                info.shape.clear();
                if (!sc.eat(']')) {
                    for (;;) {
                        int64_t v = 0;
                        if (!sc.integer(v)) return false;
                        info.shape.push_back(v);
                        if (sc.eat(']')) break;
                        if (!sc.eat(',')) return false;
                    }
                }
                has_shape = true;
            } else if (k == "data_offsets") {
                int64_t b = 0, e = 0;
                if (!sc.eat('[') || !sc.integer(b) || !sc.eat(',') || !sc.integer(e) || !sc.eat(']')) return false;
                if (b < 0 || e < b) return false;
                info.off_begin = static_cast<uint64_t>(b);
                info.off_end = static_cast<uint64_t>(e);
                has_off = true;
            } else {
                if (!skip_json_value(sc)) return false;
            }
            if (sc.eat('}')) break;
            if (!sc.eat(',')) return false;
        }
    }
    return has_dtype && has_shape && has_off;
}

size_t dtype_bytes(const std::string& dt) {
    if (dt == "U8" || dt == "I8") return 1;
    if (dt == "F16" || dt == "BF16" || dt == "I16" || dt == "U16") return 2;
    if (dt == "F32" || dt == "I32" || dt == "U32") return 4;
    if (dt == "F64" || dt == "I64" || dt == "U64") return 8;
    return 0;
}

bool read_tensor_bytes(int fd, uint64_t data_base, uint64_t fsize, const std::string& name,
                       const TensorInfo& info, size_t& elem_count, std::vector<uint8_t>& raw, std::string& err) {
    const size_t ds = dtype_bytes(info.dtype);
    if (ds == 0) {
        err = "tensor '" + name + "': unsupported dtype " + info.dtype;
        return false;
    }
    uint64_t count = 1;
    for (int64_t d : info.shape) {
        if (d < 0) { err = "tensor '" + name + "': negative shape"; return false; }
        count *= static_cast<uint64_t>(d);
    }
    const uint64_t nbytes = info.off_end - info.off_begin;
    if (count * static_cast<uint64_t>(ds) != nbytes) {
        err = "tensor '" + name + "': byte length does not match shape and dtype";
        return false;
    }
    if (info.off_end > fsize || data_base > fsize - info.off_end) {
        err = "tensor '" + name + "': data offsets exceed file size";
        return false;
    }
    elem_count = static_cast<size_t>(count);
    raw.resize(static_cast<size_t>(nbytes));
    if (nbytes > 0 && !basalt::platform::pread_all(fd, raw.data(), nbytes, data_base + info.off_begin)) {
        err = "tensor '" + name + "': failed to read bytes";
        return false;
    }
    return true;
}

bool read_i64_tensor(int fd, uint64_t data_base, uint64_t fsize, const std::string& name,
                     const std::map<std::string, TensorInfo>& tensors, std::vector<int64_t>& out, std::string& err) {
    const auto it = tensors.find(name);
    if (it == tensors.end()) { err = "reference missing tensor '" + name + "'"; return false; }
    const TensorInfo& info = it->second;
    if (info.shape.size() != 1) { err = "tensor '" + name + "': expected 1D"; return false; }
    size_t n = 0;
    std::vector<uint8_t> raw;
    if (!read_tensor_bytes(fd, data_base, fsize, name, info, n, raw, err)) return false;
    out.resize(n);
    if (info.dtype == "I64") {
        if (n > 0) std::memcpy(out.data(), raw.data(), n * sizeof(int64_t));
    } else if (info.dtype == "I32") {
        for (size_t i = 0; i < n; ++i) {
            int32_t v = 0;
            std::memcpy(&v, raw.data() + i * 4, 4);
            out[i] = static_cast<int64_t>(v);
        }
    } else {
        err = "tensor '" + name + "': expected I64 or I32, got " + info.dtype;
        return false;
    }
    return true;
}

bool read_i32_tensor(int fd, uint64_t data_base, uint64_t fsize, const std::string& name,
                     const std::map<std::string, TensorInfo>& tensors, std::vector<int32_t>& out, std::string& err) {
    const auto it = tensors.find(name);
    if (it == tensors.end()) { err = "reference missing tensor '" + name + "'"; return false; }
    const TensorInfo& info = it->second;
    if (info.shape.size() != 1) { err = "tensor '" + name + "': expected 1D"; return false; }
    size_t n = 0;
    std::vector<uint8_t> raw;
    if (!read_tensor_bytes(fd, data_base, fsize, name, info, n, raw, err)) return false;
    out.resize(n);
    if (info.dtype == "I32") {
        if (n > 0) std::memcpy(out.data(), raw.data(), n * sizeof(int32_t));
    } else if (info.dtype == "I64") {
        for (size_t i = 0; i < n; ++i) {
            int64_t v = 0;
            std::memcpy(&v, raw.data() + i * 8, 8);
            out[i] = static_cast<int32_t>(v);
        }
    } else {
        err = "tensor '" + name + "': expected I32 or I64, got " + info.dtype;
        return false;
    }
    return true;
}

bool read_u8_tensor(int fd, uint64_t data_base, uint64_t fsize, const std::string& name,
                    const std::map<std::string, TensorInfo>& tensors, std::vector<uint8_t>& out, std::string& err) {
    const auto it = tensors.find(name);
    if (it == tensors.end()) { err = "reference missing tensor '" + name + "'"; return false; }
    const TensorInfo& info = it->second;
    if (info.shape.size() != 1) { err = "tensor '" + name + "': expected 1D"; return false; }
    size_t n = 0;
    std::vector<uint8_t> raw;
    if (!read_tensor_bytes(fd, data_base, fsize, name, info, n, raw, err)) return false;
    if (info.dtype == "U8") {
        out = std::move(raw);
    } else if (info.dtype == "I32") {
        out.resize(n);
        for (size_t i = 0; i < n; ++i) {
            int32_t v = 0;
            std::memcpy(&v, raw.data() + i * 4, 4);
            out[i] = static_cast<uint8_t>(v);
        }
    } else if (info.dtype == "I64") {
        out.resize(n);
        for (size_t i = 0; i < n; ++i) {
            int64_t v = 0;
            std::memcpy(&v, raw.data() + i * 8, 8);
            out[i] = static_cast<uint8_t>(v);
        }
    } else {
        err = "tensor '" + name + "': expected U8, I32 or I64, got " + info.dtype;
        return false;
    }
    return true;
}

bool read_f32_tensor(int fd, uint64_t data_base, uint64_t fsize, const std::string& name,
                     const std::map<std::string, TensorInfo>& tensors, std::vector<float>& out, std::string& err) {
    const auto it = tensors.find(name);
    if (it == tensors.end()) { err = "reference missing tensor '" + name + "'"; return false; }
    const TensorInfo& info = it->second;
    if (info.shape.size() != 1) { err = "tensor '" + name + "': expected 1D"; return false; }
    size_t n = 0;
    std::vector<uint8_t> raw;
    if (!read_tensor_bytes(fd, data_base, fsize, name, info, n, raw, err)) return false;
    out.resize(n);
    if (info.dtype == "F32") {
        if (n > 0) std::memcpy(out.data(), raw.data(), n * sizeof(float));
    } else if (info.dtype == "F16") {
        for (size_t i = 0; i < n; ++i) {
            uint16_t h = 0;
            std::memcpy(&h, raw.data() + i * 2, 2);
            out[i] = f16_to_f32(h);
        }
    } else {
        err = "tensor '" + name + "': expected F32 or F16, got " + info.dtype;
        return false;
    }
    return true;
}

double percentile_sorted(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    if (sorted.size() == 1) return sorted[0];
    const double idx = (p / 100.0) * static_cast<double>(sorted.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(idx));
    const size_t hi = std::min(sorted.size() - 1, lo + 1);
    const double frac = idx - static_cast<double>(lo);
    return sorted[lo] + frac * (sorted[hi] - sorted[lo]);
}

double sample_std(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    double mean = 0.0;
    for (double x : v) mean += x;
    mean /= static_cast<double>(v.size());
    double s2 = 0.0;
    for (double x : v) {
        const double d = x - mean;
        s2 += d * d;
    }
    return std::sqrt(s2 / static_cast<double>(v.size() - 1));
}

}  // namespace

bool parse_role_filter(const std::string& s, RoleFilter& out) {
    if (s == "assistant") { out = RoleFilter::Assistant; return true; }
    if (s == "other") { out = RoleFilter::Other; return true; }
    if (s == "all") { out = RoleFilter::All; return true; }
    return false;
}

const char* role_filter_name(RoleFilter r) {
    switch (r) {
        case RoleFilter::Assistant: return "assistant";
        case RoleFilter::Other: return "other";
        case RoleFilter::All: return "all";
    }
    return "assistant";
}

bool role_matches(uint8_t role, RoleFilter filter) {
    switch (filter) {
        case RoleFilter::Assistant: return role == 1;
        case RoleFilter::Other: return role == 2;
        case RoleFilter::All: return role > 0;
    }
    return false;
}

bool load_reference(const std::string& path, Reference& ref, std::string& err) {
    ref = Reference{};
    uint64_t fsize = 0;
    if (!basalt::platform::file_size(path, fsize) || fsize < 8) {
        err = path + ": cannot open or not a safetensors file";
        return false;
    }
    FdGuard g;
    g.fd = basalt::platform::open_fd_utf8(path);
    if (g.fd < 0) {
        err = path + ": cannot open";
        return false;
    }
    uint64_t header_len = 0;
    if (!basalt::platform::pread_all(g.fd, &header_len, 8, 0)) {
        err = path + ": truncated header length";
        return false;
    }
    if (header_len == 0 || header_len > kMaxHeaderBytes || header_len > fsize - 8) {
        err = path + ": invalid safetensors header length";
        return false;
    }
    std::string header(static_cast<size_t>(header_len), '\0');
    if (!basalt::platform::pread_all(g.fd, header.data(), header_len, 8)) {
        err = path + ": truncated header";
        return false;
    }

    basalt::program::json::Scan sc{header, 0};
    if (!sc.eat('{')) {
        err = path + ": malformed safetensors JSON header";
        return false;
    }
    std::map<std::string, TensorInfo> tensors;
    if (!sc.eat('}')) {
        for (;;) {
            std::string key;
            if (!sc.str(key) || !sc.eat(':')) {
                err = path + ": malformed safetensors JSON header";
                return false;
            }
            if (key == "__metadata__") {
                if (!sc.eat('{')) { err = path + ": malformed __metadata__"; return false; }
                if (!sc.eat('}')) {
                    for (;;) {
                        std::string mk;
                        if (!sc.str(mk) || !sc.eat(':')) { err = path + ": malformed __metadata__"; return false; }
                        if (mk == "model") {
                            if (!sc.str(ref.model_name)) { err = path + ": malformed __metadata__.model"; return false; }
                        } else {
                            if (!skip_json_value(sc)) { err = path + ": malformed __metadata__ value"; return false; }
                        }
                        if (sc.eat('}')) break;
                        if (!sc.eat(',')) { err = path + ": malformed __metadata__"; return false; }
                    }
                }
            } else {
                TensorInfo info;
                if (!parse_tensor_info(sc, info)) {
                    err = path + ": malformed tensor entry '" + key + "'";
                    return false;
                }
                tensors.emplace(std::move(key), std::move(info));
            }
            if (sc.eat('}')) break;
            if (!sc.eat(',')) {
                err = path + ": malformed safetensors JSON header";
                return false;
            }
        }
    }

    const uint64_t data_base = 8 + header_len;
    if (!read_i64_tensor(g.fd, data_base, fsize, "conv_off", tensors, ref.conv_off, err)) return false;
    if (!read_i64_tensor(g.fd, data_base, fsize, "row_off", tensors, ref.row_off, err)) return false;
    if (!read_i32_tensor(g.fd, data_base, fsize, "tokens", tensors, ref.tokens, err)) return false;
    if (!read_u8_tensor(g.fd, data_base, fsize, "roles", tensors, ref.roles, err)) return false;
    if (!read_i32_tensor(g.fd, data_base, fsize, "topk_ids", tensors, ref.topk_ids, err)) return false;
    if (!read_f32_tensor(g.fd, data_base, fsize, "topk_lp", tensors, ref.topk_lp, err)) return false;
    if (!read_f32_tensor(g.fd, data_base, fsize, "leftover", tensors, ref.leftover, err)) return false;

    const int64_t n_tok = static_cast<int64_t>(ref.tokens.size());
    if (ref.conv_off.size() < 2 || ref.conv_off.front() != 0 || ref.conv_off.back() != n_tok) {
        err = path + ": invalid conv_off bounds";
        return false;
    }
    for (size_t i = 1; i < ref.conv_off.size(); ++i) {
        if (ref.conv_off[i] < ref.conv_off[i - 1]) {
            err = path + ": conv_off is not monotonic";
            return false;
        }
    }
    if (static_cast<int64_t>(ref.roles.size()) != n_tok || static_cast<int64_t>(ref.leftover.size()) != n_tok) {
        err = path + ": roles or leftover length does not match tokens";
        return false;
    }
    if (static_cast<int64_t>(ref.row_off.size()) != n_tok + 1 || ref.row_off.front() != 0 ||
        ref.row_off.back() != static_cast<int64_t>(ref.topk_ids.size()) ||
        ref.topk_lp.size() != ref.topk_ids.size()) {
        err = path + ": invalid row_off or topk tensor sizes";
        return false;
    }
    for (size_t i = 1; i < ref.row_off.size(); ++i) {
        if (ref.row_off[i] < ref.row_off[i - 1]) {
            err = path + ": row_off is not monotonic";
            return false;
        }
    }
    return true;
}

std::vector<int64_t> select_conversations(const Reference& ref, bool smoke, int64_t max_context, int64_t& skipped) {
    std::vector<int64_t> sel;
    skipped = 0;
    const int64_t n_convs = ref.num_convs();
    for (int64_t c = 0; c < n_convs; ++c) {
        const int64_t len = ref.conv_off[static_cast<size_t>(c + 1)] - ref.conv_off[static_cast<size_t>(c)];
        if (smoke) {
            if (c % 20 != 0 || len >= 5000) continue;
        }
        if (len > max_context) {
            ++skipped;
            continue;
        }
        sel.push_back(c);
    }
    return sel;
}

int64_t first_scored_pos(const Reference& ref, int64_t conv_idx, RoleFilter role) {
    if (conv_idx < 0 || conv_idx >= ref.num_convs()) return -1;
    const int64_t a = ref.conv_off[static_cast<size_t>(conv_idx)];
    const int64_t b = ref.conv_off[static_cast<size_t>(conv_idx + 1)];
    const int64_t n_roles = static_cast<int64_t>(ref.roles.size());
    for (int64_t i = a; i < b; ++i) {
        if (ref.row_off[static_cast<size_t>(i + 1)] <= ref.row_off[static_cast<size_t>(i)]) continue;
        if (i + 1 < b && i + 1 < n_roles && role_matches(ref.roles[static_cast<size_t>(i + 1)], role)) {
            return i - a;
        }
    }
    return -1;
}

bool parse_basalt_rows(const char* data, size_t len, std::vector<ParsedRow>& out, std::string& err) {
    out.clear();
    const char* p = data;
    const char* const end = data + len;
    while (p < end) {
        const char* nl = static_cast<const char*>(std::memchr(p, '\n', static_cast<size_t>(end - p)));
        const char* line_end = nl ? nl : end;
        std::string line(p, static_cast<size_t>(line_end - p));
        p = nl ? nl + 1 : end;

        const char* s = line.c_str();
        while (*s && std::isspace(static_cast<unsigned char>(*s))) ++s;
        if (*s == '\0') continue;

        char* ep = nullptr;
        const long long pos = std::strtoll(s, &ep, 10);
        if (ep == s) { err = "malformed row line: expected pos"; return false; }
        s = ep;
        const long long target = std::strtoll(s, &ep, 10);
        if (ep == s) { err = "malformed row line: expected target"; return false; }
        s = ep;
        // skip lp_target and lse
        (void) std::strtod(s, &ep);
        if (ep == s) { err = "malformed row line: expected lp_target"; return false; }
        s = ep;
        (void) std::strtod(s, &ep);
        if (ep == s) { err = "malformed row line: expected lse"; return false; }
        s = ep;

        ParsedRow row;
        row.pos = static_cast<int64_t>(pos);
        row.target = static_cast<int32_t>(target);
        bool first = true;
        float best_lp = -std::numeric_limits<float>::infinity();
        float min_lp = std::numeric_limits<float>::infinity();

        while (*s) {
            while (*s && std::isspace(static_cast<unsigned char>(*s))) ++s;
            if (*s == '\0') break;
            const long long id = std::strtoll(s, &ep, 10);
            if (ep == s || *ep != ':') {
                err = "malformed row entry: expected id:lp";
                return false;
            }
            s = ep + 1;
            const double lpd = std::strtod(s, &ep);
            if (ep == s) {
                err = "malformed row entry: bad log-prob";
                return false;
            }
            s = ep;
            const float lp = static_cast<float>(lpd);
            const int32_t id32 = static_cast<int32_t>(id);
            if (first || lp > best_lp) {
                best_lp = lp;
                row.top1_id = id32;
            }
            if (first || lp < min_lp) {
                min_lp = lp;
            }
            first = false;
            row.entries.emplace_back(id32, lp);
        }
        if (row.entries.empty()) {
            err = "row at pos " + std::to_string(row.pos) + " has no id:lp entries";
            return false;
        }
        row.min_lp = min_lp;
        std::sort(row.entries.begin(), row.entries.end(),
                  [](const std::pair<int32_t, float>& a, const std::pair<int32_t, float>& b) {
                      return a.first < b.first;
                  });
        out.push_back(std::move(row));
    }
    return true;
}

bool Scorer::add_conversation(int64_t conv_idx, const std::vector<ParsedRow>& rows, std::string& err) {
    if (conv_idx < 0 || conv_idx >= ref_.num_convs()) {
        err = "conversation index out of range";
        return false;
    }
    const int64_t a = ref_.conv_off[static_cast<size_t>(conv_idx)];
    const int64_t b = ref_.conv_off[static_cast<size_t>(conv_idx + 1)];

    bool has_ref_rows = false;
    const int64_t n_roles = static_cast<int64_t>(ref_.roles.size());
    for (int64_t i = a; i < b; ++i) {
        if (ref_.row_off[static_cast<size_t>(i + 1)] > ref_.row_off[static_cast<size_t>(i)] &&
            i + 1 < n_roles && role_matches(ref_.roles[static_cast<size_t>(i + 1)], role_)) {
            has_ref_rows = true;
            break;
        }
    }
    if (!has_ref_rows) return true;
    if (rows.empty()) {
        err = "conv " + std::to_string(conv_idx) + ": engine wrote no rows";
        return false;
    }

    std::unordered_map<int64_t, size_t> pos_map;
    pos_map.reserve(rows.size() * 2);
    int64_t p0 = rows[0].pos;
    size_t p0_idx = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        pos_map[rows[i].pos] = i;
        if (rows[i].pos < p0) {
            p0 = rows[i].pos;
            p0_idx = i;
        }
    }

    if (shift_ < 0) {
        const int32_t tg = rows[p0_idx].target;
        const int64_t n_tok = static_cast<int64_t>(ref_.tokens.size());
        if (a + p0 + 1 >= 0 && a + p0 + 1 < n_tok && ref_.tokens[static_cast<size_t>(a + p0 + 1)] == tg) {
            shift_ = 1;
        } else if (a + p0 >= 0 && a + p0 < n_tok && ref_.tokens[static_cast<size_t>(a + p0)] == tg) {
            shift_ = 0;
        } else {
            err = "cannot align Basalt pos/target with the reference tokens";
            return false;
        }
    }

    std::vector<float> q_lp_buf;
    for (int64_t i = a; i < b; ++i) {
        const int64_t r0 = ref_.row_off[static_cast<size_t>(i)];
        const int64_t r1 = ref_.row_off[static_cast<size_t>(i + 1)];
        if (r1 <= r0) continue;
        const uint8_t row_role = (i + 1 < n_roles) ? ref_.roles[static_cast<size_t>(i + 1)] : 0;
        if (!role_matches(row_role, role_)) continue;

        const int64_t key = shift_ ? (i - a) : (i - a + 1);
        const auto it = pos_map.find(key);
        if (it == pos_map.end()) {
            err = "conv " + std::to_string(conv_idx) + ": missing Basalt row at pos " + std::to_string(key);
            return false;
        }
        const ParsedRow& prow = rows[it->second];
        const size_t k = static_cast<size_t>(r1 - r0);
        q_lp_buf.resize(k);

        double sum_q_p = 0.0;
        double kl = 0.0;
        for (size_t j = 0; j < k; ++j) {
            const int32_t id = ref_.topk_ids[static_cast<size_t>(r0) + j];
            const auto eit = std::lower_bound(
                prow.entries.begin(), prow.entries.end(), id,
                [](const std::pair<int32_t, float>& e, int32_t target_id) { return e.first < target_id; });
            float qlp = prow.min_lp;
            if (eit != prow.entries.end() && eit->first == id) {
                qlp = eit->second;
            } else {
                ++missing_ref_ids_;
            }
            q_lp_buf[j] = qlp;
            const double rlp = static_cast<double>(ref_.topk_lp[static_cast<size_t>(r0) + j]);
            const double tlp = static_cast<double>(qlp);
            const double p = std::exp(rlp);
            kl += p * (rlp - tlp);
            sum_q_p += std::exp(tlp);
        }

        const float left_q_f = static_cast<float>(std::max(0.0, 1.0 - sum_q_p));
        const double Lp = static_cast<double>(ref_.leftover[static_cast<size_t>(i)]);
        if (Lp > 0.0) {
            const double Lq = std::max(static_cast<double>(left_q_f), 1e-12);
            kl += Lp * std::log(Lp / Lq);
        }

        const int32_t top1_ref = ref_.topk_ids[static_cast<size_t>(r0)];
        const double same = (top1_ref == prow.top1_id) ? 1.0 : 0.0;
        const double dp = std::exp(static_cast<double>(q_lp_buf[0])) -
                          std::exp(static_cast<double>(ref_.topk_lp[static_cast<size_t>(r0)]));

        ScoredRow sr;
        sr.conv = conv_idx;
        sr.role = (i + 1 < static_cast<int64_t>(ref_.roles.size())) ? ref_.roles[static_cast<size_t>(i + 1)] : 0;
        sr.kld = kl;
        sr.same = same;
        sr.dp = dp;

        if (i + 1 < static_cast<int64_t>(ref_.tokens.size())) {
            const int32_t tgt = ref_.tokens[static_cast<size_t>(i + 1)];
            for (size_t j = 0; j < k; ++j) {
                if (ref_.topk_ids[static_cast<size_t>(r0) + j] == tgt) {
                    sr.has_target = true;
                    sr.nllb = -static_cast<double>(ref_.topk_lp[static_cast<size_t>(r0) + j]);
                    sr.nllq = -static_cast<double>(q_lp_buf[j]);
                    break;
                }
            }
        }
        scored_.push_back(sr);
    }
    return true;
}

bool summarize(const std::vector<ScoredRow>& scored, RoleFilter role, int boot, uint64_t seed,
               Summary& out, std::string& err) {
    auto keep_role = [&](uint8_t r) {
        switch (role) {
            case RoleFilter::Assistant: return r == 1;
            case RoleFilter::Other: return r == 2;
            case RoleFilter::All: return r > 0;
        }
        return false;
    };

    struct ConvAccum {
        double kl = 0.0;
        double same = 0.0;
        double dp = 0.0;
        double dp2 = 0.0;
        double has = 0.0;
        double dn = 0.0;
        double nq = 0.0;
        double nb = 0.0;
        double one = 0.0;
    };

    std::map<int64_t, ConvAccum> by_conv;
    std::vector<double> kld_vals;
    int64_t outside_topk = 0;

    for (const ScoredRow& r : scored) {
        if (!keep_role(r.role)) continue;
        kld_vals.push_back(r.kld);
        ConvAccum& a = by_conv[r.conv];
        a.kl += r.kld;
        a.same += r.same;
        a.dp += r.dp;
        a.dp2 += r.dp * r.dp;
        a.one += 1.0;
        if (r.has_target) {
            a.has += 1.0;
            a.dn += (r.nllq - r.nllb);
            a.nq += r.nllq;
            a.nb += r.nllb;
        } else {
            ++outside_topk;
        }
    }

    if (kld_vals.empty()) {
        err = std::string("no scored rows match role '") + role_filter_name(role) + "'";
        return false;
    }

    std::vector<ConvAccum> convs;
    convs.reserve(by_conv.size());
    ConvAccum tot;
    for (const auto& kv : by_conv) {
        convs.push_back(kv.second);
        tot.kl += kv.second.kl;
        tot.same += kv.second.same;
        tot.dp += kv.second.dp;
        tot.dp2 += kv.second.dp2;
        tot.has += kv.second.has;
        tot.dn += kv.second.dn;
        tot.nq += kv.second.nq;
        tot.nb += kv.second.nb;
        tot.one += kv.second.one;
    }

    const size_t n_rows = kld_vals.size();
    const size_t m_convs = convs.size();
    out.n_rows = static_cast<int64_t>(n_rows);
    out.n_convs = static_cast<int64_t>(m_convs);
    out.rows_target_outside_topk = outside_topk;
    out.role = role;
    out.boot = boot;

    const double nan = std::numeric_limits<double>::quiet_NaN();
    out.kld_mean = tot.kl / tot.one;
    out.top1 = tot.same / tot.one;
    out.dp_mean = tot.dp / tot.one;
    out.dp_rms = std::sqrt(tot.dp2 / tot.one);
    if (tot.has > 0.0) {
        out.ln_ppl_ratio = tot.dn / tot.has;
        out.ppl_ratio = std::exp(out.ln_ppl_ratio);
        out.ppl_q = std::exp(tot.nq / tot.has);
        out.ppl_base = std::exp(tot.nb / tot.has);
    } else {
        out.ln_ppl_ratio = nan;
        out.ppl_ratio = nan;
        out.ppl_q = nan;
        out.ppl_base = nan;
    }

    out.kld_naive_se = (n_rows >= 2) ? (sample_std(kld_vals) / std::sqrt(static_cast<double>(n_rows))) : 0.0;
    std::sort(kld_vals.begin(), kld_vals.end());
    out.kld_median = percentile_sorted(kld_vals, 50.0);
    out.kld_p90 = percentile_sorted(kld_vals, 90.0);
    out.kld_p95 = percentile_sorted(kld_vals, 95.0);
    out.kld_p99 = percentile_sorted(kld_vals, 99.0);
    out.kld_p99_9 = percentile_sorted(kld_vals, 99.9);
    out.kld_max = kld_vals.back();

    if (boot >= 2 && m_convs >= 2) {
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<size_t> dist(0, m_convs - 1);
        std::vector<double> b_kld(static_cast<size_t>(boot));
        std::vector<double> b_top1(static_cast<size_t>(boot));
        std::vector<double> b_dp(static_cast<size_t>(boot));
        std::vector<double> b_dpr(static_cast<size_t>(boot));
        std::vector<double> b_lnr(static_cast<size_t>(boot));
        std::vector<double> b_pr(static_cast<size_t>(boot));
        std::vector<double> b_pq(static_cast<size_t>(boot));
        std::vector<double> b_pb(static_cast<size_t>(boot));

        for (int bi = 0; bi < boot; ++bi) {
            ConvAccum s;
            for (size_t j = 0; j < m_convs; ++j) {
                const ConvAccum& c = convs[dist(rng)];
                s.kl += c.kl;
                s.same += c.same;
                s.dp += c.dp;
                s.dp2 += c.dp2;
                s.has += c.has;
                s.dn += c.dn;
                s.nq += c.nq;
                s.nb += c.nb;
                s.one += c.one;
            }
            const size_t u = static_cast<size_t>(bi);
            b_kld[u] = s.kl / s.one;
            b_top1[u] = s.same / s.one;
            b_dp[u] = s.dp / s.one;
            b_dpr[u] = std::sqrt(s.dp2 / s.one);
            if (s.has > 0.0) {
                b_lnr[u] = s.dn / s.has;
                b_pr[u] = std::exp(b_lnr[u]);
                b_pq[u] = std::exp(s.nq / s.has);
                b_pb[u] = std::exp(s.nb / s.has);
            } else {
                b_lnr[u] = nan;
                b_pr[u] = nan;
                b_pq[u] = nan;
                b_pb[u] = nan;
            }
        }
        out.kld_se = sample_std(b_kld);
        out.top1_se = sample_std(b_top1);
        out.dp_mean_se = sample_std(b_dp);
        out.dp_rms_se = sample_std(b_dpr);
        out.ln_ppl_ratio_se = sample_std(b_lnr);
        out.ppl_ratio_se = sample_std(b_pr);
        out.ppl_q_se = sample_std(b_pq);
        out.ppl_base_se = sample_std(b_pb);
    } else {
        out.kld_se = 0.0;
        out.top1_se = 0.0;
        out.dp_mean_se = 0.0;
        out.dp_rms_se = 0.0;
        out.ln_ppl_ratio_se = 0.0;
        out.ppl_ratio_se = 0.0;
        out.ppl_q_se = 0.0;
        out.ppl_base_se = 0.0;
    }

    out.ran = true;
    return true;
}

}  // namespace basalt::program::quality
