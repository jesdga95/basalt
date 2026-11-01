// Runtime adapter source resolution and base compatibility; synthetic files, no GPU initialization.
#include "basalt/core/runtime_adapters.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace basalt::core;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-78s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("basalt-runtime-adapters-test-" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        if (!fs::create_directory(path)) throw std::runtime_error("cannot create test directory");
    }
    ~TempDir() { std::error_code ignored; fs::remove_all(path, ignored); }
};

struct Section { std::string name, data; };

std::string write_container(const fs::path& path, const std::vector<Section>& sections) {
    std::vector<char> header(static_cast<size_t>(kHeaderBytes), 0);
    FileHeader h{};
    std::memcpy(h.magic, kModelMagic, sizeof h.magic);
    h.version = kModelVersion;
    h.n_sections = static_cast<uint32_t>(sections.size());
    uint64_t end = kHeaderBytes;
    std::ofstream out(path, std::ios::binary);
    for (size_t i = 0; i < sections.size(); ++i) {
        SectionEntry section{};
        std::memcpy(section.name, sections[i].name.data(), sections[i].name.size());
        section.offset = (end + kSectionAlign - 1) / kSectionAlign * kSectionAlign;
        section.bytes = sections[i].data.size();
        std::memcpy(header.data() + sizeof h + i * sizeof section, &section, sizeof section);
        out.seekp(static_cast<std::streamoff>(section.offset));
        out.write(sections[i].data.data(), static_cast<std::streamsize>(section.bytes));
        end = section.offset + section.bytes;
    }
    h.file_bytes = end;
    std::memcpy(header.data(), &h, sizeof h);
    out.seekp(0);
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    out.close();
    if (!out) throw std::runtime_error("cannot write test fixture");
    return path.string();
}

std::string pattern(size_t size, unsigned seed) {
    std::string bytes(size, '\0');
    for (size_t i = 0; i < size; ++i) bytes[i] = static_cast<char>((i * 37 + seed) & 255);
    return bytes;
}

struct Replacement {
    std::string name, shape;
    uint32_t type;
    std::string payload;
};

std::string write_adapter(const fs::path& path, const std::vector<Replacement>& replacements) {
    std::string entries, payload;
    for (size_t i = 0; i < replacements.size(); ++i) {
        const auto& r = replacements[i];
        const uint64_t offset = i * kTensorAlign;
        if (i) entries += ',';
        entries += "\"part/" + std::to_string(i) + "\":{\"target\":{\"tensor\":\"" + r.name +
                   "\",\"shape\":" + r.shape + R"(,"region":{"kind":"whole"}},"storage":{"kind":"replace","type_id":)" +
                   std::to_string(r.type) + R"(,"section":"adapter/data","offset":)" + std::to_string(offset) +
                   ",\"bytes\":" + std::to_string(r.payload.size()) + "}}";
        payload.resize(static_cast<size_t>(offset), '\0');
        payload += r.payload;
    }
    const std::string meta = R"({"format":"basalt","version":1,"kind":"adapter","adapter":{"version":1,"entries":{)" +
                             entries + "}}}";
    return write_container(path, {{"meta", meta}, {"adapter/data", payload}});
}

struct Fixture {
    ModelFile base;
    std::string base_a = pattern(24, 1), base_b = pattern(4, 2);
    std::string first_a = pattern(24, 11), last_a = pattern(24, 21), last_b = pattern(4, 22);
    std::string first, last, equal;

    explicit Fixture(const fs::path& dir) {
        const std::string meta =
            R"({"format":"basalt","version":1,"kind":"model","gguf_kv":{},"tensors":{)"
            R"("a.weight":{"type":"F32","type_id":0,"shape":[2,3],"bytes":24,"offset":0},)"
            R"("b.weight":{"type":"BF16","type_id":30,"shape":[2],"bytes":4,"offset":4096},)"
            R"("per_layer_token_embd.weight":{"type":"BF16","type_id":30,"shape":[2],"bytes":4,"offset":8192},)"
            R"("untouched.weight":{"type":"F32","type_id":0,"shape":[2],"bytes":8,"offset":12288}}})";
        std::string payload = base_a;
        payload.resize(static_cast<size_t>(kTensorAlign), '\0');
        payload += base_b;
        payload.resize(static_cast<size_t>(2 * kTensorAlign), '\0');
        payload += pattern(4, 4);
        payload.resize(static_cast<size_t>(3 * kTensorAlign), '\0');
        payload += pattern(8, 3);
        const std::string path = write_container(dir / "base.basalt", {{"meta", meta}, {"gguf", payload}});
        std::string err;
        if (!base.open(path, err) || !base.load_meta(err) || !base.register_gguf_view(err))
            throw std::runtime_error("cannot load base fixture: " + err);
        first = write_adapter(dir / "first.basalt", {{"a.weight", "[2,3]", 0, first_a}});
        last = write_adapter(dir / "last.basalt",
                             {{"a.weight", "[2,3]", 0, last_a}, {"b.weight", "[2]", 30, last_b}});
        equal = write_adapter(dir / "equal.basalt", {{"a.weight", "[2,3]", 0, last_a}});
    }
};

void check_source(const RuntimeAdapters& adapters, size_t model, const std::string& name,
                  const std::vector<uint64_t>& shape, uint32_t type, const FileRange& range,
                  const std::string& payload, const std::string& label) {
    const AdapterTensor* source = adapters.source(model, name);
    check(source && source->tensor == name && source->shape == shape && source->type_id == type &&
              source->data.path == range.path && source->data.offset == range.offset && source->data.bytes == range.bytes,
          label + ": exact encoded source range and geometry");
    std::vector<uint8_t> bytes;
    std::string err;
    check(source && read_range(source->data, bytes, err) &&
              bytes == std::vector<uint8_t>(payload.begin(), payload.end()), label + ": exact payload bytes");
}

FileRange adapter_range(const std::string& path, uint64_t offset, uint64_t bytes) {
    ModelFile file;
    std::string err;
    if (!file.open(path, err)) throw std::runtime_error("cannot open adapter fixture: " + err);
    const FileRange data = file.range("adapter/data");
    return FileRange(path, data.offset + offset, bytes);
}

void source_resolution(const Fixture& f) {
    RuntimeAdapters adapters;
    std::string err;
    const std::vector<std::vector<std::string>> groups = {
        {f.first, f.last}, {f.last, f.first}, {f.last}, {f.equal}, {}};
    const bool opened = adapters.open(f.base, groups, err);
    check(opened, "compatible ordered adapter groups open");
    if (!opened) return;
    check(adapters.model_count() == 6, "base plus every group, including an empty group, has a model index");
    check(adapters.names() == std::set<std::string>{"a.weight", "b.weight"},
          "names are the union of replacement targets, not all base tensors");

    const FileRange gguf = f.base.range("gguf");
    const FileRange base_a(f.base.path(), gguf.offset, f.base_a.size());
    const FileRange base_b(f.base.path(), gguf.offset + kTensorAlign, f.base_b.size());
    const FileRange first_a = adapter_range(f.first, 0, f.first_a.size());
    const FileRange last_a = adapter_range(f.last, 0, f.last_a.size());
    const FileRange last_b = adapter_range(f.last, kTensorAlign, f.last_b.size());
    const FileRange equal_a = adapter_range(f.equal, 0, f.last_a.size());
    struct Expected { FileRange a, b; std::string a_bytes, b_bytes; };
    const std::vector<Expected> expected = {
        {base_a, base_b, f.base_a, f.base_b},
        {last_a, last_b, f.last_a, f.last_b},
        {first_a, last_b, f.first_a, f.last_b},
        {last_a, last_b, f.last_a, f.last_b},
        {equal_a, base_b, f.last_a, f.base_b},
        {base_a, base_b, f.base_a, f.base_b},
    };
    for (size_t model = 0; model < expected.size(); ++model) {
        const auto& e = expected[model];
        const std::string label = "model " + std::to_string(model);
        check_source(adapters, model, "a.weight", {2, 3}, 0, e.a, e.a_bytes, label + " a");
        check_source(adapters, model, "b.weight", {2}, 30, e.b, e.b_bytes, label + " b");
        check(!adapters.changed(model, model, "a.weight") && !adapters.changed(model, model, "b.weight"),
              label + ": selecting the same source does not change it");
    }
    check(adapters.changed(0, 1, "a.weight") && adapters.changed(0, 1, "b.weight"),
          "base-to-adapter changes each replaced source");
    check(!adapters.changed(1, 3, "a.weight") && !adapters.changed(1, 3, "b.weight"),
          "separate groups sharing winning file ranges have identical sources");
    check(adapters.changed(1, 2, "a.weight") && !adapters.changed(1, 2, "b.weight"),
          "reversing file order changes only the multiply replaced target");
    check(adapters.changed(3, 4, "a.weight") && adapters.changed(4, 3, "a.weight"),
          "equal payload bytes in distinct files still have different source identities");
    check(adapters.changed(3, 4, "b.weight") && !adapters.changed(0, 4, "b.weight"),
          "a missing replacement restores the base source rather than inheriting another group");
    check(adapters.changed(4, 0, "a.weight") && !adapters.changed(4, 0, "b.weight") &&
              !adapters.changed(0, 5, "a.weight") && !adapters.changed(0, 5, "b.weight"),
          "restoration and an empty group resolve to the original base identities");
    check_source(adapters, 0, "a.weight", {2, 3}, 0, base_a, f.base_a, "base restoration a");
    check_source(adapters, 0, "b.weight", {2}, 30, base_b, f.base_b, "base restoration b");
    check(!adapters.source(0, "absent.weight") && !adapters.source(1, "untouched.weight"),
          "names outside the replacement union have no runtime source");
    check(!adapters.select(6, err) && !err.empty() && adapters.current() == 0,
          "selecting an out-of-range model index fails without changing the current model");
    err.clear();
    check(!adapters.select(1, err) && !err.empty() && adapters.current() == 0,
          "selecting a model with unbound replacement targets fails");
    err.clear();
    check(adapters.select(5, err) && adapters.current() == 5 &&
              adapters.select(0, err) && adapters.current() == 0,
          "selecting a model with identical sources succeeds without bound device rows");

    RuntimeAdapters base_only;
    check(base_only.open(f.base, {}, err) && base_only.model_count() == 1 && base_only.names().empty(),
          "no adapter groups retains only the base model and an empty target union");
}

void compatibility_refusals(const fs::path& dir, const Fixture& f) {
    struct Bad { const char* label; Replacement replacement; };
    const std::vector<Bad> bad = {
        {"unknown source tensor", {"absent.weight", "[2,3]", 0, pattern(24, 31)}},
        {"reordered equal-size shape", {"a.weight", "[3,2]", 0, pattern(24, 32)}},
        {"extra singleton dimension", {"a.weight", "[2,3,1]", 0, pattern(24, 33)}},
        {"different shape extent", {"a.weight", "[2,4]", 0, pattern(32, 34)}},
        {"same-width different encoded type", {"b.weight", "[2]", 1, pattern(4, 35)}},
        {"unsupported PLE table replacement", {"per_layer_token_embd.weight", "[2]", 30, pattern(4, 36)}},
    };
    for (size_t i = 0; i < bad.size(); ++i) {
        const auto& b = bad[i];
        const std::string path = write_adapter(dir / ("incompatible-" + std::to_string(i) + ".basalt"), {b.replacement});
        std::string err;
        AdapterFile standalone;
        const bool valid = standalone.open(path, err);
        check(valid, std::string(b.label) + ": replacement is independently well-formed");
        if (!valid) continue;
        RuntimeAdapters direct;
        check(!direct.open(f.base, {{path}}, err) && !err.empty(),
              std::string(b.label) + ": base compatibility refuses the adapter");
        RuntimeAdapters shadowed;
        err.clear();
        check(!shadowed.open(f.base, {{path, f.last}}, err) && !err.empty(),
              std::string(b.label) + ": later compatible replacements cannot hide invalid entries");
        RuntimeAdapters later_group;
        err.clear();
        check(!later_group.open(f.base, {{f.last}, {path}}, err) && !err.empty(),
              std::string(b.label) + ": every group is checked before publication");
    }
    const std::string malformed = write_container(dir / "malformed.basalt",
                                                  {{"meta", "{"}, {"adapter/data", pattern(24, 41)}});
    RuntimeAdapters adapters;
    std::string err;
    check(!adapters.open(f.base, {{malformed, f.last}}, err) && !err.empty(),
          "malformed adapter metadata is refused even before a valid last replacement");
}
}  // namespace

int main() {
    try {
        TempDir dir;
        Fixture fixture(dir.path);
        source_resolution(fixture);
        compatibility_refusals(dir.path, fixture);
    } catch (const std::exception& e) {
        check(false, std::string("fixture failure: ") + e.what());
    }
    std::printf("%s\n", g_fail ? "FAILED" : "all passed");
    return g_fail ? 1 : 0;
}
