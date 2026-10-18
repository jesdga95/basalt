// src/core/model_file.cpp - the .basalt reader, the byte ranges the loaders share, a small JSON parser and the
// hc8 table.  Host only.  See the header.
#include "basalt/core/model_file.hpp"

#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <sys/stat.h>
#endif

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace basalt::core {

// ================================ byte ranges ================================

namespace {

bool file_size_of(const std::string& path, uint64_t& size) {
#if defined(_WIN32)
    WIN32_FILE_ATTRIBUTE_DATA d{};
    if (!GetFileAttributesExW(platform::to_wide(path).c_str(), GetFileExInfoStandard, &d)) return false;
    if (d.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) return false;   // stat's S_ISREG, in the form Windows gives
    ULARGE_INTEGER sz;
    sz.HighPart = d.nFileSizeHigh;
    sz.LowPart = d.nFileSizeLow;
    size = sz.QuadPart;
    return true;
#else
    struct stat st{};
    if (::stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 0) return false;
    size = (uint64_t) st.st_size;
    return true;
#endif
}

/// Reads `n` bytes at `off` through a descriptor, looping over the short counts one positional read answers with.
bool pread_all(int fd, void* dst, uint64_t n, uint64_t off) {
    uint8_t* p = (uint8_t*) dst;
    while (n > 0) {
        const int64_t got = platform::pread(fd, p, std::min<uint64_t>(n, 1ull << 30), off);
        if (got <= 0) {
#if defined(_WIN32)
            return false;
#else
            // A signal can interrupt a read before it has transferred anything; that one is restarted.
            if (got < 0 && errno == EINTR) continue;
            return false;
#endif
        }
        p += got;
        off += (uint64_t) got;
        n -= (uint64_t) got;
    }
    return true;
}

}  // namespace

bool FileRange::exists() const {
    uint64_t sz = 0;
    if (path.empty() || !file_size_of(path, sz)) return false;
    return offset <= sz && (bytes == 0 || bytes <= sz - offset);
}

uint64_t FileRange::size() const {
    uint64_t sz = 0;
    if (path.empty() || !file_size_of(path, sz) || offset > sz) return 0;
    return bytes != 0 ? bytes : sz - offset;
}

namespace {

// A large read without the file cache: block-aligned unbuffered reads into an aligned bounce buffer, copied out.
// False when the file system, the platform or the allocation refuses, and the caller takes the ordinary read.
bool pread_direct(const std::string& path, void* dst, uint64_t n, uint64_t off) {
    constexpr uint64_t kAlign = 4096, kChunk = 16ull << 20;
    const int fd = platform::open_unbuffered(path);
    if (fd < 0) return false;
    void* bounce = basalt::platform::alloc_aligned((size_t) (std::min<uint64_t>(n, kChunk) + 2 * kAlign), kAlign);
    if (bounce == nullptr) { platform::close_fd(fd); return false; }
    bool ok = true;
    for (uint64_t done = 0; ok && done < n;) {
        const uint64_t pos = off + done, len = std::min<uint64_t>(kChunk, n - done);
        const uint64_t a0 = pos - pos % kAlign, a1 = (pos + len + kAlign - 1) / kAlign * kAlign;
        uint64_t got = 0;
        while (got < pos + len - a0) {
            const int64_t r = platform::pread(fd, (uint8_t*) bounce + got, (size_t) (a1 - a0 - got), a0 + got);
            if (r < 0 && errno == EINTR) continue;
            if (r <= 0) break;
            got += (uint64_t) r;
        }
        ok = got >= pos + len - a0;
        if (ok) std::memcpy((uint8_t*) dst + done, (uint8_t*) bounce + (pos - a0), (size_t) len);
        done += len;
    }
    basalt::platform::free_aligned(bounce);
    platform::close_fd(fd);
    return ok;
}

}  // namespace

bool read_range_at(const FileRange& r, uint64_t off, void* dst, uint64_t n, std::string& err) {
    if (r.bytes != 0 && (off > r.bytes || n > r.bytes - off)) { err = "read past the end of a section of " + r.path; return false; }
    constexpr uint64_t kDirectMin = 256ull << 10;
    if (n >= kDirectMin && pread_direct(r.path, dst, n, r.offset + off)) return true;
    const int fd = platform::open_fd_utf8(r.path);
    if (fd < 0) { err = "cannot open " + r.path; return false; }
    const bool ok = pread_all(fd, dst, n, r.offset + off);
    platform::close_fd(fd);
    if (!ok) err = "short read from " + r.path;
    return ok;
}

bool read_range(const FileRange& r, std::vector<uint8_t>& out, std::string& err) {
    uint64_t sz = 0;
    if (r.path.empty() || !file_size_of(r.path, sz)) { err = "cannot open " + r.path; return false; }
    if (r.offset > sz || (r.bytes != 0 && r.bytes > sz - r.offset)) {
        err = r.path + ": the section runs past the end of the file";
        return false;
    }
    const uint64_t n = r.bytes != 0 ? r.bytes : sz - r.offset;
    out.resize((size_t) n);
    return n == 0 || read_range_at(r, 0, out.data(), n, err);
}

// ================================ JSON ================================

const Json* Json::get(const std::string& key) const {
    const auto it = members.find(key);
    return it == members.end() ? nullptr : &it->second;
}

double Json::num() const {
    if (type == Type::Float) return f;
    if (type == Type::Int) return neg ? -(double) u : (double) u;
    return 0.0;
}

int64_t Json::i64() const {
    if (type == Type::Float) return (int64_t) f;
    return neg ? -(int64_t) u : (int64_t) u;
}

namespace {

struct JsonParser {
    const char* p;
    const char* end;
    const char* begin;
    std::string err;

    bool fail(const char* what) {
        if (err.empty()) err = std::string(what) + " at byte " + std::to_string((size_t) (p - begin));
        return false;
    }
    void ws() { while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p; }

    static void utf8(std::string& s, uint32_t c) {
        if (c < 0x80) s += (char) c;
        else if (c < 0x800) { s += (char) (0xC0 | (c >> 6)); s += (char) (0x80 | (c & 0x3F)); }
        else if (c < 0x10000) {
            s += (char) (0xE0 | (c >> 12)); s += (char) (0x80 | ((c >> 6) & 0x3F)); s += (char) (0x80 | (c & 0x3F));
        } else {
            s += (char) (0xF0 | (c >> 18)); s += (char) (0x80 | ((c >> 12) & 0x3F));
            s += (char) (0x80 | ((c >> 6) & 0x3F)); s += (char) (0x80 | (c & 0x3F));
        }
    }
    bool hex4(uint32_t& v) {
        if (end - p < 4) return fail("bad \\u escape");
        v = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = *p++;
            v <<= 4;
            if (c >= '0' && c <= '9') v |= (uint32_t) (c - '0');
            else if (c >= 'a' && c <= 'f') v |= (uint32_t) (c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= (uint32_t) (c - 'A' + 10);
            else return fail("bad \\u escape");
        }
        return true;
    }
    bool str(std::string& out) {
        ++p;   // the opening quote
        out.clear();
        while (p < end) {
            const char c = *p++;
            if (c == '"') return true;
            if (c != '\\') { out += c; continue; }
            if (p >= end) break;
            const char e = *p++;
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t c1 = 0;
                    if (!hex4(c1)) return false;
                    if (c1 >= 0xD800 && c1 < 0xDC00 && end - p >= 6 && p[0] == '\\' && p[1] == 'u') {
                        p += 2;
                        uint32_t c2 = 0;
                        if (!hex4(c2)) return false;
                        if (c2 >= 0xDC00 && c2 < 0xE000) c1 = 0x10000 + ((c1 - 0xD800) << 10) + (c2 - 0xDC00);
                        else { utf8(out, c1); c1 = c2; }
                    }
                    utf8(out, c1);
                    break;
                }
                default: return fail("bad escape");
            }
        }
        return fail("unterminated string");
    }
    bool number(Json& v) {
        const char* a = p;
        bool neg = false;
        if (p < end && *p == '-') { neg = true; ++p; }
        const char* digits = p;
        while (p < end && *p >= '0' && *p <= '9') ++p;
        if (p == digits) return fail("bad number");
        bool is_float = false;
        if (p < end && *p == '.') { is_float = true; ++p; while (p < end && *p >= '0' && *p <= '9') ++p; }
        if (p < end && (*p == 'e' || *p == 'E')) {
            is_float = true;
            ++p;
            if (p < end && (*p == '+' || *p == '-')) ++p;
            while (p < end && *p >= '0' && *p <= '9') ++p;
        }
        const std::string tok(a, (size_t) (p - a));
        if (!is_float && (size_t) (p - digits) <= 19) {
            v.type = Json::Type::Int;
            v.neg = neg;
            v.u = std::strtoull(std::string(digits, (size_t) (p - digits)).c_str(), nullptr, 10);
            return true;
        }
        v.type = Json::Type::Float;
        v.f = std::strtod(tok.c_str(), nullptr);
        return true;
    }
    bool value(Json& v, int depth) {
        if (depth > 64) return fail("nesting too deep");
        ws();
        if (p >= end) return fail("unexpected end");
        const char c = *p;
        if (c == '{') {
            ++p;
            v.type = Json::Type::Object;
            ws();
            if (p < end && *p == '}') { ++p; return true; }
            for (;;) {
                ws();
                if (p >= end || *p != '"') return fail("expected a key");
                std::string key;
                if (!str(key)) return false;
                ws();
                if (p >= end || *p != ':') return fail("expected ':'");
                ++p;
                Json child;
                if (!value(child, depth + 1)) return false;
                v.members[std::move(key)] = std::move(child);
                ws();
                if (p < end && *p == ',') { ++p; continue; }
                if (p < end && *p == '}') { ++p; return true; }
                return fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            ++p;
            v.type = Json::Type::Array;
            ws();
            if (p < end && *p == ']') { ++p; return true; }
            for (;;) {
                Json child;
                if (!value(child, depth + 1)) return false;
                if (v.items.size() < Json::kJsonArrayKeep) v.items.push_back(std::move(child));
                ++v.count;
                ws();
                if (p < end && *p == ',') { ++p; continue; }
                if (p < end && *p == ']') { ++p; return true; }
                return fail("expected ',' or ']'");
            }
        }
        if (c == '"') { v.type = Json::Type::String; return str(v.s); }
        if (end - p >= 4 && std::memcmp(p, "true", 4) == 0) { p += 4; v.type = Json::Type::Bool; v.b = true; return true; }
        if (end - p >= 5 && std::memcmp(p, "false", 5) == 0) { p += 5; v.type = Json::Type::Bool; v.b = false; return true; }
        if (end - p >= 4 && std::memcmp(p, "null", 4) == 0) { p += 4; v.type = Json::Type::Null; return true; }
        if (c == '-' || (c >= '0' && c <= '9')) return number(v);
        return fail("unexpected character");
    }
};

}  // namespace

bool parse_json(const char* text, size_t n, Json& out, std::string& err) {
    JsonParser ps{text, text + n, text, {}};
    out = Json{};
    if (!ps.value(out, 0)) { err = "json: " + ps.err; return false; }
    ps.ws();
    if (ps.p != ps.end) { err = "json: trailing data at byte " + std::to_string((size_t) (ps.p - ps.begin)); return false; }
    return true;
}

// ================================ the model file ================================

bool ModelFile::open(const std::string& path, std::string& err) {
    path_ = path;
    sections_.clear();
    meta_ = Json{};
    meta_loaded_ = false;
    uint64_t actual = 0;
    if (!file_size_of(path, actual)) { err = "cannot open " + path; return false; }
    if (actual < kHeaderBytes) { err = path + ": not a .basalt model file (too short)"; return false; }
    std::vector<uint8_t> head((size_t) kHeaderBytes);
    {
        const int fd = platform::open_fd_utf8(path);
        if (fd < 0) { err = "cannot open " + path; return false; }
        const bool ok = pread_all(fd, head.data(), kHeaderBytes, 0);
        platform::close_fd(fd);
        if (!ok) { err = "cannot read the header of " + path; return false; }
    }
    FileHeader h;
    std::memcpy(&h, head.data(), sizeof h);
    if (std::memcmp(h.magic, kModelMagic, sizeof kModelMagic) != 0) {
        err = path + ": not a .basalt model file (bad magic)";
        return false;
    }
    if (h.version != kModelVersion) {
        err = path + ": model file version " + std::to_string(h.version) + ", this build reads " +
              std::to_string(kModelVersion);
        return false;
    }
    if (h.n_sections > kMaxSections) { err = path + ": the section table is too long"; return false; }
    if (actual < h.file_bytes) {
        err = path + ": truncated (" + std::to_string(actual) + " of " + std::to_string(h.file_bytes) + " bytes)";
        return false;
    }
    file_bytes_ = h.file_bytes;
    for (uint32_t i = 0; i < h.n_sections; ++i) {
        SectionEntry e;
        std::memcpy(&e, head.data() + sizeof(FileHeader) + (size_t) i * sizeof(SectionEntry), sizeof e);
        Section s;
        s.name.assign(e.name, strnlen(e.name, sizeof e.name));
        s.offset = e.offset;
        s.bytes = e.bytes;
        if (s.offset % kSectionAlign != 0 || s.offset < kHeaderBytes || s.offset > file_bytes_ ||
            s.bytes > file_bytes_ - s.offset) {
            err = path + ": section '" + s.name + "' lies outside the file or is misaligned";
            return false;
        }
        sections_.push_back(std::move(s));
    }
    return true;
}

const ModelFile::Section* ModelFile::find(const std::string& name) const {
    for (const Section& s : sections_)
        if (s.name == name) return &s;
    return nullptr;
}

FileRange ModelFile::range(const std::string& name) const {
    const Section* s = find(name);
    // an empty section reads as absent (a zero size would mean "to the end of the file")
    if (s == nullptr || s->bytes == 0) return {};
    return FileRange(path_, s->offset, s->bytes);
}

bool ModelFile::read(const std::string& name, std::vector<uint8_t>& out, std::string& err) const {
    const Section* s = find(name);
    if (s == nullptr) { err = path_ + " has no '" + name + "' section"; return false; }
    out.resize((size_t) s->bytes);
    if (s->bytes == 0) return true;
    return read_range_at(FileRange(path_, s->offset, s->bytes), 0, out.data(), s->bytes, err);
}

bool ModelFile::load_meta(std::string& err) {
    if (meta_loaded_) return true;
    std::vector<uint8_t> raw;
    if (!read("meta", raw, err)) return false;
    if (!parse_json((const char*) raw.data(), raw.size(), meta_, err)) { err = path_ + ": meta: " + err; return false; }
    if (!meta_.is_object()) { err = path_ + ": meta is not a JSON object"; return false; }
    meta_loaded_ = true;
    return true;
}

namespace {

MetaValue meta_value(const Json& j) {
    MetaValue v;
    switch (j.type) {
        case Json::Type::Bool: v.type = MetaType::BOOL; v.u = j.b ? 1 : 0; break;
        case Json::Type::Int:
            v.type = j.neg ? MetaType::I64 : MetaType::U64;
            v.u = j.neg ? (uint64_t) (-(int64_t) j.u) : j.u;
            if (j.neg) v.f = -(double) j.u;
            break;
        case Json::Type::Float: v.type = MetaType::F64; v.f = j.f; break;
        case Json::Type::String: v.type = MetaType::STRING; v.s = j.s; break;
        case Json::Type::Array:
            v.type = MetaType::ARRAY;
            v.count = j.count;
            for (const Json& x : j.items) v.items.push_back(meta_value(x));
            v.elem = v.items.empty() ? MetaType::U32 : v.items[0].type;
            break;
        default: v.type = MetaType::U32; break;
    }
    return v;
}

}  // namespace

bool ModelFile::register_gguf_view(std::string& err) const {
    const Json* kv = meta_.get("gguf_kv");
    const Json* tensors = meta_.get("tensors");
    const Section* gg = find("gguf");
    if (kv == nullptr || !kv->is_object() || tensors == nullptr || !tensors->is_object()) {
        err = path_ + ": meta has no gguf_kv / tensors";
        return false;
    }
    auto view = std::make_shared<VirtualGguf>();
    for (const auto& [k, v] : kv->members) view->meta.emplace(k, meta_value(v));
    for (const auto& [name, t] : tensors->members) {
        const Json* type_id = t.get("type_id");
        const Json* shape = t.get("shape");
        const Json* off = t.get("offset");
        if (gg == nullptr || type_id == nullptr || shape == nullptr || !shape->is_array() || off == nullptr ||
            shape->items.empty() || shape->items.size() > 4 || shape->count != shape->items.size()) {
            err = path_ + ": meta.tensors['" + name + "'] is malformed (or there is no gguf section)";
            return false;
        }
        TensorInfo ti;
        ti.name = name;
        for (const Json& d : shape->items) ti.shape.push_back(d.u64());
        ti.type = (uint32_t) type_id->u64();
        ti.offset = gg->offset + off->u64();
        if (const Json* nb = t.get("bytes"))
            if (off->u64() > gg->bytes || nb->u64() > gg->bytes - off->u64()) {
                err = path_ + ": tensor " + name + " runs past the end of the gguf section";
                return false;
            }
        view->tensors.push_back(std::move(ti));
    }
    gguf_register_virtual(path_, std::move(view));
    return true;
}

// ================================ the sources ================================

void model_sources(const ModelFile& m, ModelSources& s) {
    s.dense = m.range("dense");
    s.dense_index = m.range("dense.index");
    s.experts = m.range("experts");
    s.experts_layout = m.range("experts.layout");
    s.profile = m.range("profile");
    s.hc8 = m.range("hc8");
    s.mtp_dense = m.range("mtp/dense");
    s.mtp_dense_index = m.range("mtp/dense.index");
    s.mtp_experts = m.range("mtp/experts");
    s.mtp_draft_vocab = m.range("mtp/draft_vocab");
    s.model_path = m.path();
}

// ================================ the hc8 table ================================

namespace {
std::map<std::string, Hc8Copy>& hc8_table() {
    static std::map<std::string, Hc8Copy> t;
    return t;
}
}  // namespace

void hc8_set_table(std::map<std::string, Hc8Copy> table) { hc8_table() = std::move(table); }

const Hc8Copy* hc8_find(const std::string& name, int rows, int cols) {
    const auto it = hc8_table().find(name);
    if (it == hc8_table().end() || (int) it->second.rows != rows || (int) it->second.cols != cols) return nullptr;
    return &it->second;
}

bool hc8_write(const std::string& path, const std::vector<Hc8Copy>& copies, std::string& err) {
    std::vector<Hc8Entry> entries(copies.size());
    uint64_t at = (sizeof(Hc8Header) + copies.size() * sizeof(Hc8Entry) + 255) & ~255ull;
    for (size_t i = 0; i < copies.size(); ++i) {
        const Hc8Copy& c = copies[i];
        if (c.name.size() >= sizeof entries[i].name) { err = "hc8: name too long: " + c.name; return false; }
        const uint64_t want = (uint64_t) c.rows * c.cols + (uint64_t) c.rows * c.srow * 2;
        if (c.data.size() != want) { err = "hc8: wrong size for " + c.name; return false; }
        std::memset(&entries[i], 0, sizeof entries[i]);
        std::memcpy(entries[i].name, c.name.data(), c.name.size());
        entries[i].rows = c.rows;
        entries[i].cols = c.cols;
        entries[i].srow = c.srow;
        entries[i].offset = at;
        entries[i].bytes = want;
        at = (at + want + 255) & ~255ull;
    }
    Hc8Header h{};
    std::memcpy(h.magic, "BASALTH8", 8);
    h.version = 1;
    h.n_entries = (uint32_t) copies.size();
    const std::string tmp = path + ".tmp";
    std::FILE* f = platform::open_file_utf8(tmp, "wb");
    if (f == nullptr) { err = "hc8: cannot create " + tmp; return false; }
    bool ok = std::fwrite(&h, sizeof h, 1, f) == 1 &&
              (entries.empty() || std::fwrite(entries.data(), sizeof(Hc8Entry), entries.size(), f) == entries.size());
    uint64_t pos = sizeof h + entries.size() * sizeof(Hc8Entry);
    static const uint8_t zeros[256] = {};
    auto pad_to = [&](uint64_t to) {
        while (ok && pos < to) {
            const size_t n = (size_t) std::min<uint64_t>(sizeof zeros, to - pos);
            ok = std::fwrite(zeros, 1, n, f) == n;
            pos += n;
        }
    };
    for (size_t i = 0; ok && i < copies.size(); ++i) {
        pad_to(entries[i].offset);
        ok = ok && std::fwrite(copies[i].data.data(), 1, copies[i].data.size(), f) == copies[i].data.size();
        pos += copies[i].data.size();
    }
    ok = (std::fclose(f) == 0) && ok;
    if (!ok || std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        err = "hc8: could not write " + path;
        return false;
    }
    return true;
}

bool hc8_read(const FileRange& r, std::map<std::string, Hc8Copy>& out, std::string& err) {
    out.clear();
    Hc8Header h{};
    if (!read_range_at(r, 0, &h, sizeof h, err)) return false;
    if (std::memcmp(h.magic, "BASALTH8", 8) != 0 || h.version != 1) { err = "hc8: bad magic or version in " + r.path; return false; }
    const uint64_t total = r.size();
    if (h.n_entries > 4096) { err = "hc8: implausible entry count"; return false; }
    std::vector<Hc8Entry> entries(h.n_entries);
    if (h.n_entries && !read_range_at(r, sizeof h, entries.data(), entries.size() * sizeof(Hc8Entry), err)) return false;
    for (const Hc8Entry& e : entries) {
        Hc8Copy c;
        c.name.assign(e.name, strnlen(e.name, sizeof e.name));
        c.rows = e.rows;
        c.cols = e.cols;
        c.srow = e.srow;
        const uint64_t want = (uint64_t) e.rows * e.cols + (uint64_t) e.rows * e.srow * 2;
        if (e.bytes != want || e.offset > total || want > total - e.offset || e.cols % 32 != 0 ||
            e.srow < e.cols / 32) {
            err = "hc8: entry " + c.name + " is malformed";
            return false;
        }
        c.data.resize((size_t) want);
        if (!read_range_at(r, e.offset, c.data.data(), want, err)) return false;
        out[c.name] = std::move(c);
    }
    return true;
}

}  // namespace basalt::core
