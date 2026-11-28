// src/models/deepseek41/pack_test.cpp - CPU unit tests for DeepSeek-V4.1 open_pack validation:
// geometry bounds, tensor table bounds, experts.layout duplicate/zero/bounds checks, and Engram
// row types (IQ4_NL, Q5_K, Q6_K, MXFP4, MXFP8) and hash/section bounds. No GPU.
#include "models/deepseek41/pack.hpp"

#include "basalt/core/model_format.hpp"
#include "models/deepseek41/engram_rows.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

using namespace basalt::models::deepseek41;

namespace {

int g_fail = 0;

void check(bool ok, const char* what, const std::string& detail = {}) {
    if (!ok) {
        if (detail.empty()) std::printf("FAIL: %s\n", what);
        else std::printf("FAIL: %s (%s)\n", what, detail.c_str());
        ++g_fail;
    }
}

std::string utf8_path(const std::filesystem::path& p) {
    const std::u8string u = p.u8string();
    return std::string(reinterpret_cast<const char*>(u.data()), u.size());
}

struct TempPack {
    std::filesystem::path path;
    ~TempPack() {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
    std::string str() const { return utf8_path(path); }
};

using SectionSpec = std::pair<std::string, std::vector<uint8_t>>;

bool write_pack(const std::filesystem::path& path, const std::vector<SectionSpec>& secs) {
    using namespace basalt::core;
    std::vector<uint8_t> head((size_t) kHeaderBytes, 0);
    FileHeader h{};
    std::memcpy(h.magic, kModelMagic, sizeof kModelMagic);
    h.version = kModelVersion;
    h.n_sections = (uint32_t) secs.size();

    uint64_t off = kSectionAlign;
    std::vector<uint64_t> offs;
    offs.reserve(secs.size());
    for (size_t i = 0; i < secs.size(); ++i) {
        SectionEntry e{};
        const size_t nlen = secs[i].first.size() < sizeof e.name ? secs[i].first.size() : sizeof e.name;
        std::memcpy(e.name, secs[i].first.data(), nlen);
        e.offset = off;
        e.bytes = secs[i].second.size();
        std::memcpy(head.data() + sizeof(FileHeader) + i * sizeof(SectionEntry), &e, sizeof e);
        offs.push_back(off);
        const uint64_t end = off + e.bytes;
        off = ((end + kSectionAlign - 1) / kSectionAlign) * kSectionAlign;
        if (off == offs.back()) off += kSectionAlign;
    }
    h.file_bytes = off;
    std::memcpy(head.data(), &h, sizeof h);

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(head.data()), (std::streamsize) head.size());
    for (size_t i = 0; i < secs.size(); ++i) {
        if (secs[i].second.empty()) continue;
        f.seekp((std::streamoff) offs[i]);
        f.write(reinterpret_cast<const char*>(secs[i].second.data()), (std::streamsize) secs[i].second.size());
    }
    if (h.file_bytes > 0) {
        const char zero = 0;
        f.seekp((std::streamoff) (h.file_bytes - 1));
        f.write(&zero, 1);
    }
    return (bool) f;
}

std::vector<uint8_t> bytes_of(const std::string& s) {
    return std::vector<uint8_t>(s.begin(), s.end());
}

struct Spec {
    std::string geometry =
        R"({"block_count":2,"embedding_length":256,"expert_count":4,"expert_used_count":2,)"
        R"("expert_feed_forward_length":256,"expert_shared_count":1,"attention.compress_ratios":[0,2]})";
    std::string gguf_kv =
        R"({"tokenizer.ggml.tokens":["a","b","c","d"],"deepseek41.engram.token_map":[0,1,2,3]})";
    std::string tensors =
        R"({"token_embd.weight":{"type_id":14,"shape":[256,4],"offset":0,"bytes":64}})";
    bool include_engram_meta = true;
    std::string engram_meta =
        R"({"section":"engram","rows":5,"row_values":256,"row_bytes":144,"type_id":20,)"
        R"("ple.layers":[1],"ple.ngram_size":3,"ple.heads_per_ngram":1,"engram.pad_token_id":2,)"
        R"("ple.layer_multipliers":[7,11,13],"ple.head_vocab_sizes":[2,3],"ple.head_offsets":[0,2]})";
    std::string layout = "# experts v5\n0 19 16 0 32\n1 19 16 128 32\n";
    size_t gguf_bytes = 256;
    size_t experts_bytes = 256;
    bool include_experts_section = true;
    size_t engram_bytes = 5 * 144;
    bool include_engram_section = true;

    std::vector<SectionSpec> sections() const {
        std::string meta = "{\"arch\":\"deepseek41\",\"geometry\":" + geometry +
                           ",\"gguf_kv\":" + gguf_kv + ",\"tensors\":" + tensors;
        if (include_engram_meta) meta += ",\"engram\":" + engram_meta;
        meta += "}";
        std::vector<SectionSpec> s;
        s.push_back({"meta", bytes_of(meta)});
        s.push_back({"gguf", std::vector<uint8_t>(gguf_bytes, 0)});
        if (include_experts_section) s.push_back({"experts", std::vector<uint8_t>(experts_bytes, 0)});
        s.push_back({"experts.layout", bytes_of(layout)});
        if (include_engram_section) s.push_back({"engram", std::vector<uint8_t>(engram_bytes, 0)});
        return s;
    }
};

int g_case_id = 0;

bool try_open(const Spec& sp, Pack& p, std::string& err) {
    const std::filesystem::path dir = std::filesystem::temp_directory_path();
    TempPack tp{dir / ("ds41_pack_test_" + std::to_string(++g_case_id) + ".basalt")};
    if (!write_pack(tp.path, sp.sections())) {
        err = "write_pack failed";
        return false;
    }
    return open_pack(tp.str(), p, err);
}

void expect_reject(const Spec& sp, const char* label, const char* needle) {
    Pack p;
    std::string err;
    const bool ok = try_open(sp, p, err);
    check(!ok && err.find(needle) != std::string::npos, label, ok ? "unexpectedly accepted" : err);
    check(!p.engram.present, label, "engram.present stayed set after failure");
}

}  // namespace

int main() {
    // 1. Valid packs across all five supported Engram row types (IQ4_NL, Q5_K, Q6_K, MXFP4, MXFP8).
    struct RowTypeCase {
        int type;
        size_t row_bytes;
        const char* name;
    };
    const RowTypeCase types[] = {
        {kEngramIq4Nl, 144, "IQ4_NL"},
        {kEngramQ5K, 176, "Q5_K"},
        {kEngramQ6K, 210, "Q6_K"},
        {kEngramMxfp4, 136, "MXFP4"},
        {kEngramMxfp8, 264, "MXFP8"},
    };
    for (const RowTypeCase& tc : types) {
        Spec sp;
        sp.engram_meta =
            "{\"section\":\"engram\",\"rows\":5,\"row_values\":256,\"row_bytes\":" + std::to_string(tc.row_bytes) +
            ",\"type_id\":" + std::to_string(tc.type) +
            ",\"ple.layers\":[1],\"ple.ngram_size\":3,\"ple.heads_per_ngram\":1,\"engram.pad_token_id\":2,"
            "\"ple.layer_multipliers\":[7,11,13],\"ple.head_vocab_sizes\":[2,3],\"ple.head_offsets\":[0,2]}";
        sp.engram_bytes = 5 * tc.row_bytes;
        Pack p;
        std::string err;
        const bool ok = try_open(sp, p, err);
        check(ok && p.engram.present && p.engram.type == tc.type && p.engram.row_bytes == (uint32_t) tc.row_bytes &&
                  p.engram.rows == 5 && p.geo.n_vocab == 4,
              tc.name, err);
    }

    // 2. Valid pack without Engram and with experts.layout v7 (up == gu).
    {
        Spec sp;
        sp.include_engram_meta = false;
        sp.include_engram_section = false;
        sp.layout = "# experts v7\n0 19 16 0 32 19\n1 19 16 128 32 19\n";
        Pack p;
        std::string err;
        const bool ok = try_open(sp, p, err);
        check(ok && !p.engram.present && p.experts.size() == 2 && p.experts[1].offset == 128, "valid no-engram v7", err);
    }

    // 3. Geometry rejections.
    {
        Spec sp;
        sp.geometry =
            R"({"block_count":2,"embedding_length":300,"expert_count":4,"expert_used_count":2,)"
            R"("expert_feed_forward_length":256,"expert_shared_count":1})";
        expect_reject(sp, "n_embd not multiple of 32", "meta.geometry");
    }
    {
        Spec sp;
        sp.geometry =
            R"({"block_count":2,"embedding_length":256,"expert_count":16,"expert_used_count":9,)"
            R"("expert_feed_forward_length":256,"expert_shared_count":1})";
        expect_reject(sp, "n_used > 8", "meta.geometry");
    }
    {
        Spec sp;
        sp.geometry =
            R"({"block_count":2,"embedding_length":256,"expert_count":4,"expert_used_count":5,)"
            R"("expert_feed_forward_length":256,"expert_shared_count":1})";
        expect_reject(sp, "n_used > n_expert", "meta.geometry");
    }
    {
        Spec sp;
        sp.geometry =
            R"({"block_count":2,"embedding_length":256,"expert_count":4,"expert_used_count":2,)"
            R"("expert_feed_forward_length":100,"expert_shared_count":1})";
        expect_reject(sp, "n_ff_exp not multiple of 32", "meta.geometry");
    }
    {
        Spec sp;
        sp.geometry =
            R"({"block_count":2,"embedding_length":256,"expert_count":4,"expert_used_count":2,)"
            R"("expert_feed_forward_length":256,"expert_shared_count":-1})";
        expect_reject(sp, "negative n_shared", "meta.geometry");
    }
    {
        Spec sp;
        sp.geometry =
            R"({"block_count":2,"embedding_length":256,"expert_count":4,"expert_used_count":2,)"
            R"("expert_feed_forward_length":256,"expert_shared_count":1,"attention.compress_ratios":[0,-2]})";
        expect_reject(sp, "negative compress ratio", "compress ratio");
    }
    {
        Spec sp;
        sp.gguf_kv = R"({"deepseek41.engram.token_map":[0,1,2,3]})";
        expect_reject(sp, "missing vocabulary", "vocabulary");
    }

    // 4. Tensor table rejections.
    {
        Spec sp;
        sp.tensors = R"({"bad":123})";
        expect_reject(sp, "tensor not an object", "not an object");
    }
    {
        Spec sp;
        sp.tensors = R"({"token_embd.weight":{"type_id":14,"shape":[256,4],"bytes":64}})";
        expect_reject(sp, "tensor missing offset", "offset");
    }
    {
        Spec sp;
        sp.tensors = R"({"token_embd.weight":{"type_id":14,"shape":[256,4],"offset":200,"bytes":64}})";
        expect_reject(sp, "tensor runs past section", "past the end");
    }

    // 5. Experts section and layout rejections.
    {
        Spec sp;
        sp.include_experts_section = false;
        expect_reject(sp, "missing experts section", "missing experts section");
    }
    {
        Spec sp;
        sp.layout = "# experts v5\n0 19 16 0 0\n1 19 16 128 32\n";
        expect_reject(sp, "zero blob_bytes", "bad line");
    }
    {
        Spec sp;
        sp.layout = "# experts v5\n0 19 16 0 32\n0 19 16 128 32\n";
        expect_reject(sp, "duplicate expert layer", "duplicate layer");
    }
    {
        Spec sp;
        sp.layout = "# experts v5\n0 19 16 0 32\n1 19 16 200 32\n";
        expect_reject(sp, "expert layer past section end", "bad line");
    }

    // 6. Engram rejections.
    {
        Spec sp;
        sp.engram_meta = R"({"section":"engram","row_values":256,"row_bytes":144,"type_id":20})";
        expect_reject(sp, "engram missing rows", "missing rows");
    }
    {
        Spec sp;
        sp.engram_meta =
            R"({"section":"engram","rows":5,"row_values":256,"row_bytes":144,"type_id":12,)"
            R"("ple.layers":[1],"ple.ngram_size":3,"ple.heads_per_ngram":1,"engram.pad_token_id":2,)"
            R"("ple.layer_multipliers":[7,11,13],"ple.head_vocab_sizes":[2,3],"ple.head_offsets":[0,2]})";
        expect_reject(sp, "unsupported engram type Q4_K", "Q6_K, MXFP4 and MXFP8");
    }
    {
        Spec sp;
        sp.engram_bytes = 4 * 144;
        expect_reject(sp, "engram rows exceed section bytes", "exceed its section");
    }
    {
        Spec sp;
        sp.engram_meta =
            R"({"section":"engram","rows":5,"row_values":256,"row_bytes":144,"type_id":20,)"
            R"("ple.layers":[2],"ple.ngram_size":3,"ple.heads_per_ngram":1,"engram.pad_token_id":2,)"
            R"("ple.layer_multipliers":[7,11,13],"ple.head_vocab_sizes":[2,3],"ple.head_offsets":[0,2]})";
        expect_reject(sp, "engram layer out of range", "layer out of range");
    }
    {
        Spec sp;
        sp.engram_meta =
            R"({"section":"engram","rows":5,"row_values":256,"row_bytes":144,"type_id":20,)"
            R"("ple.layers":[1],"ple.ngram_size":10,"ple.heads_per_ngram":1,"engram.pad_token_id":2,)"
            R"("ple.layer_multipliers":[1,2,3,4,5,6,7,8,9,10],"ple.head_vocab_sizes":[1,1,1,1,1,1,1,1,1],)"
            R"("ple.head_offsets":[0,0,0,0,0,0,0,0,0]})";
        expect_reject(sp, "engram n_gram > 9", "n-gram geometry");
    }
    {
        Spec sp;
        sp.engram_meta =
            R"({"section":"engram","rows":5,"row_values":256,"row_bytes":144,"type_id":20,)"
            R"("ple.layers":[1],"ple.ngram_size":3,"ple.heads_per_ngram":1,"engram.pad_token_id":2,)"
            R"("ple.layer_multipliers":[7,11,13],"ple.head_vocab_sizes":[0,3],"ple.head_offsets":[0,2]})";
        expect_reject(sp, "engram zero head vocab size", "zero head vocab size");
    }
    {
        Spec sp;
        sp.engram_meta =
            R"({"section":"engram","rows":5,"row_values":256,"row_bytes":144,"type_id":20,)"
            R"("ple.layers":[1],"ple.ngram_size":3,"ple.heads_per_ngram":1,"engram.pad_token_id":2,)"
            R"("ple.layer_multipliers":[7,11,13],"ple.head_vocab_sizes":[2,4],"ple.head_offsets":[0,2]})";
        expect_reject(sp, "engram head range exceeds table rows", "exceeds table rows");
    }
    {
        Spec sp;
        sp.engram_meta =
            R"({"section":"engram","rows":5,"row_values":256,"row_bytes":144,"type_id":20,)"
            R"("ple.layers":[1],"ple.ngram_size":3,"ple.heads_per_ngram":1,"engram.pad_token_id":99,)"
            R"("ple.layer_multipliers":[7,11,13],"ple.head_vocab_sizes":[2,3],"ple.head_offsets":[0,2]})";
        expect_reject(sp, "engram pad_token_id out of range", "pad_token_id");
    }

    if (g_fail == 0) std::printf("ds41_pack_test: ok\n");
    return g_fail ? 1 : 0;
}
