// src/vision/mmproj.cpp - see basalt/vision/mmproj.hpp.
#include "basalt/vision/mmproj.hpp"

#include "basalt/platform/file.hpp"
#include "basalt/platform/memory.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace basalt::vision {

namespace {

struct Cursor {
    const uint8_t* p;
    const uint8_t* end;
    bool ok = true;
    template <class T>
    T get() {
        T v{};
        if ((size_t) (end - p) < sizeof(T)) { ok = false; return v; }
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string str() {
        const uint64_t n = get<uint64_t>();
        if (!ok || (uint64_t) (end - p) < n) { ok = false; return {}; }
        std::string s((const char*) p, n);
        p += n;
        return s;
    }
};

size_t scalar_size(uint32_t t) {
    switch (t) {
        case 0: case 1: case 7: return 1;
        case 2: case 3: return 2;
        case 4: case 5: case 6: return 4;
        case 10: case 11: case 12: return 8;
        default: return 0;
    }
}

// A metadata value: numbers are kept as doubles (one array of numbers as well); strings and other arrays skipped.
struct Value {
    bool num = false;
    double d = 0;
    std::vector<double> arr;
};

Value read_value(Cursor& c, uint32_t t, int depth = 0) {
    Value v;
    if (depth >= 8) { c.ok = false; return v; }
    if (t == 8) { c.str(); return v; }
    if (t == 9) {
        const uint32_t et = c.get<uint32_t>();
        const uint64_t n = c.get<uint64_t>();
        if (!c.ok) return v;
        const uint64_t rem = (uint64_t) (c.end - c.p);
        if (et == 8) {
            if (n > rem / 8) { c.ok = false; return v; }
        } else if (et == 9) {
            if (n > rem / 12) { c.ok = false; return v; }
        } else {
            const size_t esz = scalar_size(et);
            if (esz == 0 || n > rem / esz) { c.ok = false; return v; }
            if (n > 4096) { c.p += (size_t) (n * esz); return v; }
        }
        if (et != 8 && et != 9) v.arr.reserve((size_t) n);
        for (uint64_t i = 0; c.ok && i < n; ++i) {
            const Value e = read_value(c, et, depth + 1);
            if (e.num && n <= 4096) v.arr.push_back(e.d);
        }
        return v;
    }
    v.num = true;
    switch (t) {
        case 0: v.d = c.get<uint8_t>(); break;
        case 1: v.d = c.get<int8_t>(); break;
        case 2: v.d = c.get<uint16_t>(); break;
        case 3: v.d = c.get<int16_t>(); break;
        case 4: v.d = c.get<uint32_t>(); break;
        case 5: v.d = c.get<int32_t>(); break;
        case 6: v.d = c.get<float>(); break;
        case 7: v.d = c.get<uint8_t>(); break;
        case 10: v.d = (double) c.get<uint64_t>(); break;
        case 11: v.d = (double) c.get<int64_t>(); break;
        case 12: v.d = c.get<double>(); break;
        default: c.ok = false;
    }
    return v;
}

}  // namespace

void Mmproj::release() {
    if (map_ && !mapped_) basalt::platform::free_aligned(map_);
    mapping_.reset();
    mapped_ = false;
    map_ = nullptr;
    tensors_.clear();
}

Mmproj::~Mmproj() { release(); }

bool Mmproj::open(const core::FileRange& range, std::string& err, const char* projector) {
    release();
    hp_ = {};
    projector_.clear();
    size_ = 0;
    err.clear();
    auto fail = [&](const std::string& msg) {
        release();
        err = msg;
        return false;
    };
    const std::string& path = range.path;
    const uint64_t offset = range.offset, file_bytes = range.size();
    uint64_t whole = 0;
    if (!platform::file_size(path, whole)) return fail("cannot open " + path);
    if (file_bytes < 32 || offset > whole || file_bytes > whole - offset || (uint64_t) (size_t) file_bytes != file_bytes)
        return fail("cannot read " + path);
    size_ = (size_t) file_bytes;
    // read whole with block-aligned unbuffered reads, so the file cache neither copies nor keeps it; a file system that
    // refuses them, and a platform with no unbuffered descriptor, is mapped instead
    {
        constexpr size_t kAlign = 4096, kChunk = 16u << 20;
        const int dfd = platform::open_unbuffered(path);
        void* mem = nullptr;
        if (dfd >= 0 && offset % kAlign == 0 && size_ <= SIZE_MAX - (kAlign - 1))
            mem = basalt::platform::alloc_aligned((size_ + kAlign - 1) / kAlign * kAlign, kAlign);
        if (mem != nullptr) {
            size_t got = 0;
            while (got < size_) {
                const size_t want = std::min<size_t>(kChunk, (size_ - got + kAlign - 1) / kAlign * kAlign);
                const int64_t r = basalt::platform::pread(dfd, (uint8_t*) mem + got, want, offset + got);
                if (r <= 0) break;
                got += (size_t) r;
            }
            if (got >= size_) { map_ = mem; mapped_ = false; }
            else basalt::platform::free_aligned(mem);
        }
        if (dfd >= 0) platform::close_fd(dfd);
    }
    if (map_ == nullptr) {
        std::string why;
        platform::MapOptions window;   // only the projector's bytes: a section of a pack must not map the whole pack
        window.offset = offset;
        window.bytes = file_bytes;
        auto m = platform::MappedFile::open_ro(path, &why, window);
        if (!m) return fail("cannot map " + path);
        mapping_ = std::move(*m);
        map_ = const_cast<uint8_t*>(mapping_.data());
        mapped_ = true;
    }
    const uint8_t* base = (const uint8_t*) map_;
    Cursor c{base, base + size_};
    if (c.get<uint32_t>() != 0x46554747u) return fail(path + " is not a GGUF file");
    if (c.get<uint32_t>() != 3) return fail("unsupported GGUF version");
    const uint64_t n_tensors = c.get<uint64_t>(), n_kv = c.get<uint64_t>();
    if (!c.ok || n_kv > (uint64_t) (c.end - c.p) / 12) return fail("truncated GGUF metadata");
    std::map<std::string, Value> kv;
    std::string arch, proj_type;
    uint32_t align = 32;
    for (uint64_t i = 0; c.ok && i < n_kv; ++i) {
        const std::string key = c.str();
        const uint32_t t = c.get<uint32_t>();
        if (t == 8) {
            const std::string s = c.str();
            if (key == "general.architecture") arch = s;
            else if (key == "clip.projector_type" || key == "clip.vision.projector_type") proj_type = s;
            continue;
        }
        kv[key] = read_value(c, t);
    }
    if (!c.ok) return fail("truncated GGUF metadata");
    if (arch != "clip" || proj_type != projector)
        return fail(std::string("not a ") + projector + " projector (architecture '" + arch + "', projector '" + proj_type + "')");
    auto num = [&](const char* k, double def, bool must = true) {
        auto it = kv.find(k);
        if (it == kv.end() || !it->second.num || !std::isfinite(it->second.d)) {
            if (must && err.empty()) err = std::string("missing or invalid metadata ") + k;
            return def;
        }
        return it->second.d;
    };
    auto int_param = [&](const char* k, int def, bool must = true) {
        const double d = num(k, (double) def, must);
        if (!std::isfinite(d) || d < (double) INT_MIN || d > (double) INT_MAX) {
            if (err.empty()) err = std::string("out-of-range metadata ") + k;
            return def;
        }
        return (int) d;
    };
    hp_.image_size = int_param("clip.vision.image_size", 0);
    hp_.patch = int_param("clip.vision.patch_size", 0);
    hp_.embd = int_param("clip.vision.embedding_length", 0);
    hp_.ffn = int_param("clip.vision.feed_forward_length", 0);
    hp_.layers = int_param("clip.vision.block_count", 0);
    hp_.heads = int_param("clip.vision.attention.head_count", 0);
    hp_.merge = int_param("clip.vision.spatial_merge_size",
                          kv.count("clip.vision.projector.scale_factor") ? int_param("clip.vision.projector.scale_factor", 2, false) : 2,
                          false);
    hp_.min_pixels = int_param("clip.vision.image_min_pixels", 0, false);
    projector_ = projector;
    hp_.proj = int_param("clip.vision.projection_dim", 0);
    hp_.eps = (float) num("clip.vision.attention.layer_norm_epsilon", 1e-6, false);
    hp_.rope_theta = (float) num("clip.vision.rope.freq_base", 0, false);
    if (!err.empty()) {
        release();
        return false;
    }
    if (hp_.patch <= 0 || hp_.embd <= 0 || hp_.ffn <= 0 || hp_.layers <= 0 || hp_.heads <= 0 || hp_.merge <= 0 ||
        hp_.proj <= 0 || hp_.embd % hp_.heads != 0 || hp_.image_size < 0 || hp_.min_pixels < 0 ||
        !std::isfinite(hp_.eps) || hp_.eps < 0.f || !std::isfinite(hp_.rope_theta) || hp_.rope_theta < 0.f) {
        return fail("invalid projector hyper-parameters");
    }
    for (const char* k : {"clip.vision.image_mean", "clip.vision.image_std"}) {
        auto it = kv.find(k);
        if (it == kv.end() || it->second.arr.size() != 3) return fail(std::string("missing metadata ") + k);
        const bool is_std = k[18] != 'm';
        for (int i = 0; i < 3; ++i) {
            const float val = (float) it->second.arr[i];
            if (!std::isfinite(val) || (is_std && !(val > 0.f))) return fail(std::string("invalid metadata ") + k);
            (is_std ? hp_.std : hp_.mean)[i] = val;
        }
    }
    if (auto it = kv.find("general.alignment"); it != kv.end()) {
        const double a = it->second.d;
        if (!it->second.num || !std::isfinite(a) || a < 1.0 || a > (double) (1u << 24) || std::floor(a) != a)
            return fail("invalid general.alignment");
        align = (uint32_t) a;
        if ((align & (align - 1u)) != 0) return fail("general.alignment must be a power of two");
    }
    if (n_tensors > (uint64_t) (c.end - c.p) / 24) return fail("truncated GGUF tensor directory");
    struct Info { std::string name; uint32_t type; std::vector<int64_t> ne; uint64_t off; };
    std::vector<Info> infos;
    infos.reserve((size_t) n_tensors);
    for (uint64_t i = 0; c.ok && i < n_tensors; ++i) {
        Info t;
        t.name = c.str();
        const uint32_t nd = c.get<uint32_t>();
        if (nd > 4) return fail("bad tensor rank");
        for (uint32_t d = 0; d < nd; ++d) {
            const uint64_t dim = c.get<uint64_t>();
            if (dim == 0 || dim > (uint64_t) INT64_MAX) return fail("bad tensor dimension");
            t.ne.push_back((int64_t) dim);
        }
        t.type = c.get<uint32_t>();
        t.off = c.get<uint64_t>();
        infos.push_back(std::move(t));
    }
    if (!c.ok) return fail("truncated GGUF tensor directory");
    const size_t hdr = (size_t) (c.p - base);
    if (hdr > size_ || (size_t) (align - 1) > SIZE_MAX - hdr) return fail("GGUF header exceeds file size");
    const size_t data0 = (hdr + align - 1) / align * align;
    if (data0 > size_) return fail("GGUF tensor data starts past the end of the file");
    const uint64_t avail = (uint64_t) (size_ - data0);
    for (auto& t : infos) {
        TensorRef r;
        r.name = t.name;
        r.type = t.type;
        r.ne = t.ne;
        const size_t esz = t.type == 0 ? 4 : (t.type == 1 || t.type == 30) ? 2 : 0;
        if (esz == 0) return fail("tensor " + t.name + " has an unsupported type " + std::to_string(t.type));
        uint64_t elems = 1;
        for (int64_t d : r.ne) {
            if (d <= 0 || (uint64_t) d > UINT64_MAX / elems) return fail("tensor " + t.name + " has invalid dimensions");
            elems *= (uint64_t) d;
        }
        if (elems > avail / esz) return fail("tensor " + t.name + " runs past the end of the file");
        r.bytes = (size_t) (elems * esz);
        if (t.off > avail || (uint64_t) r.bytes > avail - t.off)
            return fail("tensor " + t.name + " runs past the end of the file");
        r.data = base + data0 + (size_t) t.off;
        tensors_[t.name] = std::move(r);
    }
    return true;
}

const TensorRef* Mmproj::find(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

const TensorRef* Mmproj::need(const std::string& name, uint32_t type, int64_t elems, std::string& err) const {
    const TensorRef* t = find(name);
    if (!t) { err = "the projector has no tensor " + name; return nullptr; }
    if (t->type != type || t->elems() != elems) {
        err = "tensor " + name + " has an unexpected type or size";
        return nullptr;
    }
    return t;
}

}  // namespace basalt::vision
