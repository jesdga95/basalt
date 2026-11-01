// Standalone adapter ranges and exact replacement bytes; synthetic files only, no base model or GPU.
#include "basalt/core/model_adapter.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
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
               ("basalt-model-adapter-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
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

std::string entry(const std::string& tensor = "weight", const std::string& shape = "[2]",
                  uint32_t type = 0, uint64_t offset = 0, uint64_t bytes = 8) {
    return "{\"target\":{\"tensor\":\"" + tensor + "\",\"shape\":" + shape +
           R"(,"region":{"kind":"whole"}},"storage":{"kind":"replace","type_id":)" +
           std::to_string(type) + R"(,"section":"adapter/data","offset":)" + std::to_string(offset) +
           ",\"bytes\":" + std::to_string(bytes) + "}}";
}

std::string manifest(const std::string& entries) {
    return R"({"format":"basalt","version":1,"kind":"adapter","name":"test",)"
           R"("adapter":{"version":1,"entries":{)" + entries + "}}}";
}

std::string changed(std::string text, const std::string& from, const std::string& to) {
    const size_t at = text.find(from);
    if (at == std::string::npos) throw std::runtime_error("invalid fixture mutation: " + from);
    text.replace(at, from.size(), to);
    return text;
}

std::string pattern(size_t size, unsigned seed) {
    std::string result(size, '\0');
    for (size_t i = 0; i < size; ++i) result[i] = static_cast<char>((i * 37 + seed) & 255);
    return result;
}

void ordinary_and_quantized(const fs::path& dir) {
    struct Value {
        std::string name, shape;
        std::vector<uint64_t> dimensions;
        uint32_t type;
        std::string payload;
    };
    const std::vector<Value> values = {
        {"f32.weight", "[2]", {2}, 0, std::string("\x00\x00\xc0\x3f\x00\x00\x00\xc0", 8)},
        {"bf16.weight", "[2,2]", {2, 2}, 30, std::string("\x80\x3f\x00\xc0\x00\x00\x80\xbf", 8)},
        {"blk.0.attn_q.weight", "[256,2]", {256, 2}, 14, pattern(420, 3)},
        {"q8.weight", "[32,2]", {32, 2}, 8, pattern(68, 9)},
        {"nvfp4.weight", "[64,2]", {64, 2}, 40, pattern(72, 15)},
        {"i8.weight", "[1,2,1,1]", {1, 2, 1, 1}, 24, std::string("\x80\x7f", 2)},
    };
    std::string entries, payload;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) entries += ',';
        const uint64_t offset = i * kTensorAlign;
        entries += "\"part/" + std::to_string(i) + "\":" +
                   entry(values[i].name, values[i].shape, values[i].type, offset, values[i].payload.size());
        payload.resize(static_cast<size_t>(offset), '\0');
        payload += values[i].payload;
    }
    // Descriptive top-level fields do not alter replacement semantics.
    const std::string meta = changed(manifest(entries), "\"name\":\"test\"", R"("license":"example","description":{"text":"opaque"})");
    const std::string path = write_container(dir / "values.basalt", {{"meta", meta}, {"adapter/data", payload}});
    AdapterFile adapter;
    std::string err = "stale error";
    check(adapter.open(path, err) && err.empty(), "ordinary and quantized replacements open without a base");
    check(adapter.replacements().size() == values.size(), "every distinct replacement is retained");
    for (size_t i = 0; i < values.size(); ++i) {
        const auto& value = values[i];
        const AdapterTensor* tensor = adapter.find(value.name);
        check(tensor && tensor->tensor == value.name && tensor->shape == value.dimensions &&
                  tensor->type_id == value.type && tensor->data.path == path &&
                  tensor->data.offset == 2 * kSectionAlign + i * kTensorAlign && tensor->data.bytes == value.payload.size(),
              value.name + ": absolute range and GGUF-order shape");
        std::vector<uint8_t> bytes{99};
        err = "stale error";
        check(adapter.read_tensor(value.name, bytes, err) && err.empty() &&
                  bytes == std::vector<uint8_t>(value.payload.begin(), value.payload.end()),
              value.name + ": exact encoded replacement bytes");
    }
    ModelFile source;
    std::vector<uint8_t> after;
    check(source.open(path, err) && source.read("adapter/data", after, err) &&
              after == std::vector<uint8_t>(payload.begin(), payload.end()), "reading never mutates source payloads");
    std::vector<uint8_t> missing{1, 2, 3};
    check(!adapter.find("absent") && !adapter.read_tensor("absent", missing, err) && missing.empty() && !err.empty(),
          "an unknown tensor fails without stale output");
}

void many_entries(const fs::path& dir) {
    std::string entries, payload;
    for (unsigned i = 0; i < 97; ++i) {
        if (i) entries += ',';
        entries += "\"arbitrary-" + std::to_string(200 - i) + "\":" +
                   entry("tensor." + std::to_string(i), "[1]", 24, i * kTensorAlign, 1);
        payload.resize(i * kTensorAlign, '\0');
        payload += static_cast<char>(i);
    }
    AdapterFile adapter;
    std::string err;
    const std::string path = write_container(dir / "many.basalt", {{"meta", manifest(entries)}, {"adapter/data", payload}});
    check(adapter.open(path, err) && adapter.replacements().size() == 97, "more than 64 object entries survive JSON parsing");
    bool exact = true;
    for (unsigned i = 0; i < 97; ++i) {
        std::vector<uint8_t> bytes;
        if (!adapter.read_tensor("tensor." + std::to_string(i), bytes, err) || bytes != std::vector<uint8_t>{static_cast<uint8_t>(i)}) {
            exact = false;
        }
    }
    check(exact, "all 97 targets resolve their own encoded values regardless of entry IDs");
}

void malformed_manifests(const fs::path& dir) {
    const std::string valid = manifest("\"id\":" + entry());
    struct Bad { const char* label; const char* from; const char* to; };
    const Bad cases[] = {
        {"wrong format", "\"basalt\"", "\"gguf\""},
        {"wrong artifact kind", "\"kind\":\"adapter\"", "\"kind\":\"model\""},
        {"missing artifact kind", "\"kind\":\"adapter\",", ""},
        {"outer version", "\"version\":1", "\"version\":2"},
        {"float outer version", "\"version\":1", "\"version\":1.0"},
        {"boolean outer version", "\"version\":1", "\"version\":true"},
        {"future adapter version", "\"adapter\":{\"version\":1", "\"adapter\":{\"version\":2"},
        {"unknown adapter semantics", "\"adapter\":{", "\"adapter\":{\"scale\":1,"},
        {"unknown entry semantics", "\"id\":{", "\"id\":{\"scale\":1,"},
        {"unknown target selector", "\"tensor\":\"weight\"", "\"tensor\":\"weight\",\"layer\":0"},
        {"unsupported partial region", "\"whole\"", "\"slice\""},
        {"unknown whole-region fields", "\"kind\":\"whole\"", "\"kind\":\"whole\",\"start\":0"},
        {"unsupported additive storage", "\"replace\"", "\"add\""},
        {"unsupported LoRA storage", "\"replace\"", "\"lora\""},
        {"unknown storage semantics", "\"kind\":\"replace\"", "\"kind\":\"replace\",\"scale\":1"},
        {"wrong payload section", "\"section\":\"adapter/data\"", "\"section\":\"gguf\""},
        {"missing storage bytes", ",\"bytes\":8", ""},
        {"empty ID", "\"id\":", "\"\":"},
        {"NUL ID", "\"id\":", "\"id\\u0000\":"},
        {"empty target", "\"tensor\":\"weight\"", "\"tensor\":\"\""},
        {"NUL target", "\"weight\"", "\"weight\\u0000tail\""},
        {"nonstring target", "\"tensor\":\"weight\"", "\"tensor\":42"},
        {"NUL descriptive name", "\"name\":\"test\"", "\"name\":\"a\\u0000b\""},
        {"nonstring descriptive name", "\"name\":\"test\"", "\"name\":7"},
        {"scalar shape", "[2]", "2"},
        {"empty shape", "[2]", "[]"},
        {"too many dimensions", "[2]", "[1,1,1,1,2]"},
        {"zero extent", "[2]", "[0]"},
        {"negative extent", "[2]", "[-2]"},
        {"boolean extent", "[2]", "[true]"},
        {"float extent", "[2]", "[2.0]"},
        {"string extent", "[2]", "[\"2\"]"},
        {"shape product overflow", "[2]", "[9223372036854775808,2]"},
        {"byte product overflow", "[2]", "[4611686018427387904]"},
        {"uint64 maximum extent stays exact", "[2]", "[18446744073709551615]"},
        {"integer literal overflow", "[2]", "[18446744073709551616]"},
        {"unknown dtype", "\"type_id\":0", "\"type_id\":999"},
        {"dtype uint32 overflow", "\"type_id\":0", "\"type_id\":4294967296"},
        {"negative dtype", "\"type_id\":0", "\"type_id\":-1"},
        {"boolean dtype", "\"type_id\":0", "\"type_id\":false"},
        {"float dtype", "\"type_id\":0", "\"type_id\":0.0"},
        {"unaligned offset", "\"offset\":0", "\"offset\":1"},
        {"payload offset past end", "\"offset\":0", "\"offset\":4096"},
        {"offset plus bytes overflow", "\"offset\":0", "\"offset\":18446744073709547520"},
        {"negative offset", "\"offset\":0", "\"offset\":-4096"},
        {"boolean offset", "\"offset\":0", "\"offset\":false"},
        {"float offset", "\"offset\":0", "\"offset\":0.0"},
        {"wrong packed byte size", "\"bytes\":8", "\"bytes\":7"},
        {"zero packed bytes", "\"bytes\":8", "\"bytes\":0"},
        {"negative packed bytes", "\"bytes\":8", "\"bytes\":-8"},
        {"boolean packed bytes", "\"bytes\":8", "\"bytes\":true"},
        {"float packed bytes", "\"bytes\":8", "\"bytes\":8.0"},
        {"duplicate semantic field", "\"bytes\":8", "\"bytes\":8,\"bytes\":8"},
    };
    const auto reject = [&](const std::string& meta, const std::string& label, size_t bytes = 8) {
        const std::string path = write_container(dir / "invalid.basalt", {{"meta", meta}, {"adapter/data", pattern(bytes, 1)}});
        AdapterFile adapter;
        std::string err;
        check(!adapter.open(path, err) && !err.empty() && adapter.replacements().empty(), label);
    };
    for (const auto& bad : cases) reject(changed(valid, bad.from, bad.to), bad.label);
    reject(manifest(""), "empty entry object");
    reject(changed(valid, "\"entries\":{\"id\":" + entry() + "}", "\"entries\":[]"), "entry arrays are not supported");
    reject(manifest("\"first\":" + entry() + ",\"second\":" + entry()), "duplicate target names under distinct IDs");
    reject(manifest("\"id\":" + entry() + ",\"id\":" + entry("other")), "duplicate entry IDs");
    reject(manifest("\"id\":" + entry("q6", "[128,2]", 14, 0, 210)), "whole blocks required on fastest axis", 210);
    reject(manifest("\"id\":" + entry("nvfp4", "[64]", 40, 0, 40)), "NVFP4 rejects auxiliary routed-expert tail", 40);
    reject(valid, "payload one byte short", 7);
    std::string long_shape = "[2";
    for (unsigned i = 1; i < 65; ++i) long_shape += ",1";
    reject(changed(valid, "[2]", long_shape + "]"), "array sampling cannot hide excessive shape rank");
    reject("[]", "metadata must be an object");
    reject("{", "malformed JSON");

    AdapterFile missing;
    std::string err;
    check(!missing.open(write_container(dir / "no-data.basalt", {{"meta", valid}}), err), "missing adapter/data section");
    check(!missing.open(write_container(dir / "no-meta.basalt", {{"adapter/data", pattern(8, 1)}}), err), "missing metadata section");
}

void lifetime(const fs::path& dir) {
    const std::string first_payload = pattern(8, 77);
    const std::string first = write_container(dir / "first.basalt",
        {{"meta", manifest("\"a\":" + entry("first"))}, {"adapter/data", first_payload}});
    const std::string second = write_container(dir / "second.basalt",
        {{"meta", manifest("\"b\":" + entry("second"))}, {"adapter/data", pattern(8, 88)}});
    AdapterFile adapter;
    std::string err;
    check(adapter.open(first, err), "initial open");
    const AdapterTensor* original = adapter.find("first");
    const FileRange retained = original ? original->data : FileRange();
    check(adapter.open(second, err) && !adapter.find("first") && adapter.find("second") && adapter.replacements().size() == 1,
          "successful reopen replaces all prior state");
    std::vector<uint8_t> bytes;
    check(read_range(retained, bytes, err) && bytes == std::vector<uint8_t>(first_payload.begin(), first_payload.end()),
          "published ranges own their path beyond the temporary parser and a reopen");
    const std::string partial = write_container(dir / "partial.basalt",
        {{"meta", manifest("\"a\":" + entry("new") + ",\"z\":" + entry("bad", "[0]"))}, {"adapter/data", pattern(8, 1)}});
    check(!adapter.open(partial, err) && !err.empty() && adapter.replacements().empty() && !adapter.find("second") && !adapter.find("new"),
          "late manifest failure discards old and partially validated state");
    check(adapter.open(first, err) && err.empty(), "valid reopen recovers from validation failure");
    check(!adapter.open((dir / "absent.basalt").string(), err) && !err.empty() && adapter.replacements().empty(),
          "missing-file reopen clears published state");
    check(adapter.open(first, err), "reopen before truncation");
    fs::resize_file(first, 2 * kSectionAlign + 7);
    bytes = {1, 2, 3};
    check(!adapter.read_tensor("first", bytes, err) && !err.empty() && bytes.empty(),
          "post-open truncation is reported without stale or partial bytes");
    check(!adapter.open(first, err) && adapter.replacements().empty(), "truncated-container reopen clears published state");
    check(adapter.open(second, err), "open before source removal");
    fs::remove(second);
    bytes = {1};
    check(!adapter.read_tensor("second", bytes, err) && !err.empty() && bytes.empty(), "removed source fails exact byte read");
}
}  // namespace

int main() {
    TempDir dir;
    ordinary_and_quantized(dir.path);
    many_entries(dir.path);
    malformed_manifests(dir.path);
    lifetime(dir.path);
    std::printf("%s\n", g_fail ? "FAILED" : "all passed");
    return g_fail ? 1 : 0;
}
