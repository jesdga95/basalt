// tests/core/model_file_test.cpp - the .basalt reader (basalt/core/model_file.hpp) on a synthetic file: the table and
// its refusals, the JSON of `meta`, the GGUF view of the `gguf` section, the expert layout read from its section's
// text, and the hc8 table round trip.  No model, no GPU.
#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/core/model_file.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
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
               ("basalt-model-file-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() { std::error_code ignored; fs::remove_all(path, ignored); }
};

struct Sec { std::string name; std::string data; };

/// Writes a model file with the given sections, each at a multiple of kSectionAlign.
std::string write_model(const fs::path& path, const std::vector<Sec>& secs, bool truncate = false) {
    std::vector<char> file((size_t) kHeaderBytes, 0);
    FileHeader h{};
    std::memcpy(h.magic, kModelMagic, 8);
    h.version = kModelVersion;
    h.n_sections = (uint32_t) secs.size();
    for (size_t i = 0; i < secs.size(); ++i) {
        const uint64_t off = (file.size() + kSectionAlign - 1) / kSectionAlign * kSectionAlign;
        file.resize((size_t) off, 0);
        file.insert(file.end(), secs[i].data.begin(), secs[i].data.end());
        SectionEntry e{};
        std::memcpy(e.name, secs[i].name.data(), secs[i].name.size());
        e.offset = off;
        e.bytes = secs[i].data.size();
        std::memcpy(file.data() + sizeof h + i * sizeof e, &e, sizeof e);
    }
    h.file_bytes = file.size();
    std::memcpy(file.data(), &h, sizeof h);
    if (truncate) file.resize(file.size() - 10);
    std::ofstream(path, std::ios::binary).write(file.data(), (std::streamsize) file.size());
    return path.string();
}
}  // namespace

int main() {
    TempDir d;
    // the gguf section: two tensors at kTensorAlign
    std::string gg(2 * kTensorAlign, '\0');
    for (size_t i = 0; i < 100; ++i) gg[i] = (char) i;
    for (size_t i = 0; i < 40; ++i) gg[kTensorAlign + i] = (char) (200 + i);
    const std::string meta =
        R"({"format":"basalt","gguf_kv":{"general.architecture":"qwen4exp","qwen4exp.expert_count":512,)"
        R"("qwen4exp.rope.freq_base":10000000.0,"tok":["a","b","c"],"neg":-3,"name":"hé"},)"
        R"("tensors":{"a.weight":{"type":"F32","type_id":0,"shape":[25],"bytes":100,"offset":0},)"
        R"("b.weight":{"type":"F32","type_id":0,"shape":[5,2],"bytes":40,"offset":4096}},)"
        R"("sampling":{"temperature":1.0}})";
    basalt::core::Hc8Copy c;
    c.name = "blk.0.hc_attn_down.weight";
    c.rows = 3; c.cols = 64; c.srow = 8;
    c.data.resize(3 * 64 + 3 * 8 * 2);
    for (size_t i = 0; i < c.data.size(); ++i) c.data[i] = (uint8_t) (i * 7);
    std::string err;
    const std::string hc8 = (d.path / "copies.hc8").string();
    check(hc8_write(hc8, {c}, err), "hc8_write");
    std::string hc8_bytes;
    {
        std::ifstream f(hc8, std::ios::binary);
        hc8_bytes.assign(std::istreambuf_iterator<char>(f), {});
    }
    const std::string layout =
        "# experts v5: layer gu_type d_type offset blob_bytes (n_expert 2, total 0)\n";

    const std::string path = write_model(d.path / "m.basalt", {{"meta", meta}, {"gguf", gg}, {"hc8", hc8_bytes},
                                                                {"experts.layout", layout}});
    std::printf("1. the table\n");
    ModelFile m;
    check(m.open(path, err), "open: " + err);
    check(m.find("gguf") != nullptr && m.find("nothing") == nullptr, "find");
    check(m.range("gguf").offset % kSectionAlign == 0 && m.range("gguf").bytes == gg.size(), "a section is aligned and sized");
    check(m.range("nothing").empty(), "an absent section is an empty range");
    std::vector<uint8_t> raw;
    check(m.read("gguf", raw, err) && raw.size() == gg.size() && raw[5] == 5, "read a section");
    {
        ModelFile bad;
        std::string e2;
        const std::string trunc = write_model(d.path / "t.basalt", {{"meta", meta}}, true);
        check(!bad.open(trunc, e2) && e2.find("truncated") != std::string::npos, "a truncated file is refused: " + e2);
        std::ofstream(d.path / "n.basalt") << std::string(70000, 'x');
        check(!bad.open((d.path / "n.basalt").string(), e2), "a file without the magic is refused");
        check(!bad.open(write_model(d.path / "duplicate.basalt", {{"meta", meta}, {"meta", meta}}), e2),
              "duplicate section names cannot shadow metadata");
        check(!bad.open(write_model(d.path / "empty-name.basalt", {{"", meta}}), e2),
              "empty section names are refused");
        const std::string overlap = write_model(d.path / "overlap.basalt", {{"meta", meta}, {"gguf", gg}});
        {
            std::fstream file(overlap, std::ios::in | std::ios::out | std::ios::binary);
            SectionEntry second{};
            const auto at = static_cast<std::streamoff>(sizeof(FileHeader) + sizeof(SectionEntry));
            file.seekg(at);
            file.read(reinterpret_cast<char*>(&second), sizeof second);
            second.offset = kSectionAlign;
            file.seekp(at);
            file.write(reinterpret_cast<const char*>(&second), sizeof second);
        }
        check(!bad.open(overlap, e2), "overlapping section ranges are refused");
    }

    std::printf("2. meta and the GGUF view\n");
    check(m.load_meta(err), "load_meta: " + err);
    check(m.register_gguf_view(err), "register_gguf_view: " + err);
    {
        const basalt::GgufModel g(std::vector<std::string>{path});
        const basalt::MetaValue* ec = g.meta().get("qwen4exp.expert_count");
        const basalt::MetaValue* fb = g.meta().get("qwen4exp.rope.freq_base");
        const basalt::MetaValue* nm = g.meta().get("name");
        const basalt::MetaValue* tk = g.meta().get("tok");
        check(ec && ec->u == 512 && fb && fb->num() == 1e7 && nm && nm->s == "h\xc3\xa9" && tk && tk->count == 3,
              "kv: integer, float, escaped string, array");
        size_t shard = 9;
        const basalt::TensorInfo* t = g.find("b.weight", &shard);
        check(t && shard == 0 && t->shape.size() == 2 && t->shape[1] == 2 && t->type == 0 &&
                  g.shard(0).data_start() + t->offset == m.range("gguf").offset + 4096 &&
                  g.shard(0).tensor_data(*t)[0] == (uint8_t) 200,
              "a tensor is at the section offset + its own, and its bytes are in the mapping");
        check(g.in_bounds(*t, 0), "in_bounds");
    }
    {
        ModelFile adapter;
        const std::string adapter_path = write_model(d.path / "adapter.basalt",
            {{"meta", "{\"kind\":\"adapter\"," + meta.substr(1)}, {"gguf", gg}});
        check(adapter.open(adapter_path, err) && adapter.load_meta(err) && !adapter.register_gguf_view(err),
              "adapter metadata cannot register a model view even when it includes model fields");
        ModelFile exact;
        const std::string exact_path = write_model(d.path / "integer.basalt",
            {{"meta", R"({"limit":18446744073709551615})"}});
        check(exact.open(exact_path, err) && exact.load_meta(err) &&
                  exact.meta().get("limit")->u64() == 18446744073709551615ull,
              "metadata preserves the full unsigned 64-bit integer range");
    }

    std::printf("3. the expert layout text and the hc8 table\n");
    {
        std::vector<uint8_t> text;
        check(m.read("experts.layout", text, err), "read the layout section");
        // a header of the .basalt spelling with no rows is refused as incomplete, not for its header
        const bool ok = basalt::kernels::cpu::expert_layout_load_text(std::string(text.begin(), text.end()), 2, 2, err);
        check(!ok && err.find("missing or not contiguous") != std::string::npos,
              "\"# experts v5\" is read as a layout header: " + err);
        check(!basalt::kernels::cpu::expert_layout_load_text(std::string(), 2, 2, err), "empty text is refused");
    }
    {
        std::map<std::string, Hc8Copy> back;
        check(hc8_read(m.range("hc8"), back, err) && back.count(c.name) && back[c.name].data == c.data &&
                  back[c.name].rows == 3 && back[c.name].srow == 8,
              "the hc8 section reads back byte for byte: " + err);
        check(hc8_read(FileRange(hc8), back, err) && back.size() == 1, "and so does the file");
        hc8_set_table(back);
        check(hc8_find(c.name, 3, 64) != nullptr && hc8_find(c.name, 3, 32) == nullptr && hc8_find("x", 3, 64) == nullptr,
              "hc8_find matches name and shape");
    }
    std::printf("%s\n", g_fail ? "FAILED" : "all passed");
    return g_fail ? 1 : 0;
}
