// src/models/ernie55/pack_test.cpp - unit test for ERNIE-5.5 context policy, cache layout, BF16 check, and pack/meta validation (CPU-only).
#include "basalt/core/model_format.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/models/ernie55/cache_layout.hpp"
#include "basalt/platform/file.hpp"
#include "models/ernie55/context_policy.hpp"
#include "models/ernie55/pack.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace {

int g_fail = 0;

void expect(bool cond, const char* msg) {
    if (!cond) {
        ++g_fail;
        std::fprintf(stderr, "FAIL: %s\n", msg);
    }
}

void expect_err_has(bool ok, const std::string& err, const char* needle, const char* label) {
    if (ok || err.find(needle) == std::string::npos) {
        ++g_fail;
        std::fprintf(stderr, "FAIL: %s (ok=%d, err='%s', expected substring '%s')\n", label, (int) ok, err.c_str(),
                     needle);
    }
}

struct TempFile {
    std::string path;
    ~TempFile() {
        if (!path.empty()) {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    }
};

std::string make_temp_path(const char* name) {
    return (std::filesystem::path(basalt::platform::temp_dir()) / name).string();
}

bool write_synthetic_pack(const std::string& path, const std::string& meta_json, const std::string& exp_layout,
                          uint64_t gguf_bytes, uint64_t exp_bytes) {
    using namespace basalt::core;
    std::FILE* f = basalt::platform::open_file_utf8(path, "wb");
    if (!f) return false;

    struct Sec {
        const char* name;
        std::vector<uint8_t> data;
        uint64_t off = 0;
    };
    std::vector<Sec> secs;
    secs.push_back({"meta", std::vector<uint8_t>(meta_json.begin(), meta_json.end()), 0});
    secs.push_back({"gguf", std::vector<uint8_t>((size_t) gguf_bytes, 0), 0});
    secs.push_back({"experts", std::vector<uint8_t>((size_t) exp_bytes, 0), 0});
    secs.push_back({"experts.layout", std::vector<uint8_t>(exp_layout.begin(), exp_layout.end()), 0});

    uint64_t cur = kSectionAlign;
    for (Sec& s : secs) {
        s.off = cur;
        cur += ((s.data.size() + kSectionAlign - 1) / kSectionAlign) * kSectionAlign;
        if (s.data.empty()) cur += kSectionAlign;
    }
    const uint64_t total_bytes = cur;

    std::vector<uint8_t> hdr((size_t) kHeaderBytes, 0);
    FileHeader fh{};
    std::memcpy(fh.magic, kModelMagic, sizeof(kModelMagic));
    fh.version = kModelVersion;
    fh.n_sections = (uint32_t) secs.size();
    fh.file_bytes = total_bytes;
    std::memcpy(hdr.data(), &fh, sizeof(fh));

    for (size_t i = 0; i < secs.size(); ++i) {
        SectionEntry se{};
        const size_t nlen = std::min(std::strlen(secs[i].name), sizeof(se.name) - 1);
        std::memcpy(se.name, secs[i].name, nlen);
        se.offset = secs[i].off;
        se.bytes = secs[i].data.size();
        std::memcpy(hdr.data() + sizeof(FileHeader) + i * sizeof(SectionEntry), &se, sizeof(se));
    }

    bool ok = std::fwrite(hdr.data(), 1, hdr.size(), f) == hdr.size();
    for (const Sec& s : secs) {
        if (!ok) break;
        if (basalt::platform::fseek64(f, (int64_t) s.off, SEEK_SET) != 0) { ok = false; break; }
        if (!s.data.empty() && std::fwrite(s.data.data(), 1, s.data.size(), f) != s.data.size()) { ok = false; break; }
    }
    if (ok && total_bytes > 0) {
        const uint8_t zero = 0;
        if (basalt::platform::fseek64(f, (int64_t) (total_bytes - 1), SEEK_SET) != 0 ||
            std::fwrite(&zero, 1, 1, f) != 1)
            ok = false;
    }
    std::fclose(f);
    return ok;
}

std::map<std::string, std::string> base_kv() {
    return {
        {"ernie55.block_count", "2"},
        {"ernie55.first_k_dense_replace", "1"},
        {"ernie55.embedding_length", "4096"},
        {"ernie55.feed_forward_length", "12288"},
        {"ernie55.context_length", "1048576"},
        {"ernie55.attention.layer_norm_rms_epsilon", "1e-6"},
        {"ernie55.attention.sliding_window", "128"},
        {"ernie55.attention.compress_ratios", "[128, 0]"},
        {"ernie55.attention.head_count", "64"},
        {"ernie55.attention.head_count_kv", "1"},
        {"ernie55.attention.key_length", "512"},
        {"ernie55.attention.value_length", "512"},
        {"ernie55.attention.q_lora_rank", "1024"},
        {"ernie55.attention.kv_lora_rank", "512"},
        {"ernie55.attention.output_group_count", "8"},
        {"ernie55.attention.output_lora_rank", "1024"},
        {"ernie55.rope.dimension_count", "64"},
        {"ernie55.rope.freq_base", "8000000.0"},
        {"ernie55.attention.compress_rope_freq_base", "160000.0"},
        {"ernie55.rope.scaling.factor", "16.0"},
        {"ernie55.rope.scaling.original_context_length", "65536"},
        {"ernie55.rope.scaling.beta_fast", "32.0"},
        {"ernie55.rope.scaling.beta_slow", "1.0"},
        {"ernie55.mla.head_count", "64"},
        {"ernie55.mla.q_lora_rank", "1024"},
        {"ernie55.mla.kv_lora_rank", "512"},
        {"ernie55.mla.qk_nope_head_dim", "192"},
        {"ernie55.mla.qk_rope_head_dim", "64"},
        {"ernie55.mla.v_head_dim", "256"},
        {"ernie55.attention.indexer.head_count", "64"},
        {"ernie55.attention.indexer.key_length", "128"},
        {"ernie55.attention.indexer.top_k", "2048"},
        {"ernie55.expert_count", "512"},
        {"ernie55.expert_used_count", "10"},
        {"ernie55.expert_shared_count", "1"},
        {"ernie55.expert_feed_forward_length", "2048"},
        {"ernie55.expert_latent_length", "2048"},
        {"ernie55.expert_weights_norm", "true"},
        {"ernie55.expert_gate_softcap", "4.0"},
        {"ernie55.expert_up_softcap", "25.0"},
        {"ernie55.hyper_connection.count", "4"},
        {"ernie55.hyper_connection.sinkhorn_iterations", "20"},
        {"ernie55.hyper_connection.epsilon", "1e-6"},
        {"ernie55.has_flashindexer", "true"},
        {"ernie55.vocab_size", "256000"},
    };
}

std::string build_meta_json(const std::map<std::string, std::string>& kv) {
    std::string out = "{\"arch\":\"ernie55\",\"gguf_kv\":{";
    bool first = true;
    for (const auto& [k, v] : kv) {
        if (!first) out += ",";
        first = false;
        out += "\"" + k + "\":" + v;
    }
    out += "},\"tensors\":{\"output_norm.weight\":{\"type_id\":0,\"shape\":[4096],\"offset\":0,\"bytes\":16384}}}";
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace basalt::models::ernie55;

    // 1. Shared cache layout constants & context policy
    expect(layout::kHcaFp8RowBytes == 608, "FP8 HCA row stride is 608 B");
    expect(layout::kDsaFp8RowBytes == 656, "FP8 DSA row stride is 656 B");
    expect(layout::kIdxFp4RowBytes == 68, "MXFP4 indexer row stride is 68 B");
    expect(layout::hca_row_bytes(true) == 608 && layout::hca_row_bytes(false) == 1024, "hca_row_bytes");
    expect(layout::dsa_row_bytes(true) == 656 && layout::dsa_row_bytes(false) == 1152, "dsa_row_bytes");
    expect(layout::idx_row_bytes(true) == 68 && layout::idx_row_bytes(false) == 256, "idx_row_bytes");

    ctxp::CacheFormat fmt = ctxp::CacheFormat::Fp16;
    expect(ctxp::cache_format("", fmt) && fmt == ctxp::CacheFormat::Fp8, "default kv is fp8");
    expect(ctxp::cache_format("fp8", fmt) && fmt == ctxp::CacheFormat::Fp8, "fp8 parses");
    expect(ctxp::cache_format("fp16", fmt) && fmt == ctxp::CacheFormat::Fp16, "fp16 parses");
    expect(!ctxp::cache_format("int4", fmt), "unknown kv rejected");

    // 37 HCA (ratio 128, dim 512) + 6 MLA-DSA (512 + 64 + 128 idx)
    const uint64_t bpp_fp8 = ctxp::bytes_per_position(37, 6, 128, 512, 512, 64, 128, ctxp::CacheFormat::Fp8);
    const uint64_t bpp_fp16 = ctxp::bytes_per_position(37, 6, 128, 512, 512, 64, 128, ctxp::CacheFormat::Fp16);
    const uint64_t exp_fp8 = 37ull * ((608ull + 127ull) / 128ull) + 6ull * (656ull + 68ull);
    const uint64_t exp_fp16 = 37ull * (1024ull / 128ull) + 6ull * (1152ull + 256ull);
    expect(bpp_fp8 == exp_fp8, "fp8 bytes/pos matches layout constants (4529 B)");
    expect(bpp_fp16 == exp_fp16, "fp16 bytes/pos matches layout constants (8744 B)");

    const auto p_ok = ctxp::plan_context(262144, 1048576, 6ull << 30, 37, 6, 128, 512, 512, 64, 128, ctxp::CacheFormat::Fp8);
    expect(p_ok.error.empty() && p_ok.context == 262144, "262K fp8 fits easily in 6 GiB free VRAM");

    const auto p_oob = ctxp::plan_context(2000000, 1048576, 6ull << 30, 37, 6, 128, 512, 512, 64, 128, ctxp::CacheFormat::Fp8);
    expect(!p_oob.error.empty(), "context past 1M rejected");

    const auto p_oom = ctxp::plan_context(1048576, 1048576, 2ull << 30, 37, 6, 128, 512, 512, 64, 128, ctxp::CacheFormat::Fp8);
    expect(!p_oom.error.empty(), "1M context exceeding half of 2 GiB free VRAM rejected");

    const uint64_t sample_prompt = 800ull << 20;
    const uint64_t sample_headroom = 160ull << 20;
    const auto tr = ctxp::plan_tier_reserve(sample_prompt, sample_headroom);
    expect(tr.copies == ctxp::kCopiesReserveBytes, "copies reserve unchanged for lendable VRAM tiers");
    expect(tr.beside_homes == ctxp::kBesideHomesReserveBytes, "beside_homes reserve sized for module room");
    expect(tr.modules == ctxp::kModuleRoomBytes, "modules reserve matches CUDA/MMQ module room");
    expect(tr.homes == sample_prompt + sample_headroom + ctxp::kModuleRoomBytes,
           "homes reserve covers prompt chunk + MMQ headroom + modules");

    // 2. f32_range_is_bf16 check
    {
        TempFile tf{make_temp_path("basalt_ernie55_bf16_check.bin")};
        const std::vector<uint32_t> vals = {0x00000000u, 0x3f800000u, 0xbf000000u, 0x42c80000u, 0x3f800001u};
        std::FILE* f = basalt::platform::open_file_utf8(tf.path, "wb");
        expect(f != nullptr && std::fwrite(vals.data(), sizeof(uint32_t), vals.size(), f) == vals.size(), "write bf16 tmp");
        if (f) std::fclose(f);
        const int fd = basalt::platform::open_fd_utf8(tf.path);
        expect(fd >= 0, "open bf16 tmp");
        if (fd >= 0) {
            bool exact = false;
            std::string err;
            expect(f32_range_is_bf16(fd, 0, 4 * sizeof(uint32_t), exact, err) && exact, "exact bf16-in-f32 detected");
            expect(f32_range_is_bf16(fd, 0, 5 * sizeof(uint32_t), exact, err) && !exact, "non-zero low-16 rejected");
            basalt::platform::close_fd(fd);
        }
    }

    // 3. Bad pack path returns clean error
    {
        Pack pk;
        std::string err;
        expect(!open_pack("/nonexistent/ernie55.basalt", pk, err) && !err.empty(), "missing pack returns error");
    }

    // 4. Synthetic .basalt validation: valid pack, missing keys fail loudly, kernel dim mismatches fail loudly
    {
        TempFile tf{make_temp_path("basalt_ernie55_synth.basalt")};
        const std::string exp_layout = "1 " + std::to_string((int) GGML_TYPE_MXFP4) + " " +
                                       std::to_string((int) GGML_TYPE_MXFP4) + " 0 64\n";
        const uint64_t exp_bytes = 512ull * 64ull;

        auto kv = base_kv();
        expect(write_synthetic_pack(tf.path, build_meta_json(kv), exp_layout, 16384, exp_bytes), "write valid synth pack");
        {
            Pack pk;
            std::string err;
            expect(open_pack(tf.path, pk, err), "valid synthetic pack opens");
            expect(pk.geo.n_layers == 2 && pk.geo.n_expert == 512 && pk.geo.gate_softcap == 4.0f, "geometry parsed");
        }

        // Missing required keys must fail loudly with the missing key name
        for (const char* drop_key : {"ernie55.expert_gate_softcap", "ernie55.rope.scaling.factor",
                                     "ernie55.attention.compress_ratios", "ernie55.mla.kv_lora_rank"}) {
            auto kv_miss = base_kv();
            kv_miss.erase(drop_key);
            expect(write_synthetic_pack(tf.path, build_meta_json(kv_miss), exp_layout, 16384, exp_bytes), "write miss pack");
            Pack pk;
            std::string err;
            expect_err_has(open_pack(tf.path, pk, err), err, drop_key, "missing required key");
        }

        // Hard-coded kernel dims must be validated in open_pack
        struct BadDim {
            const char* key;
            const char* val;
            const char* needle;
        };
        const BadDim bad_dims[] = {
            {"ernie55.attention.key_length", "256", "attention.key_length"},
            {"ernie55.mla.kv_lora_rank", "256", "mla.kv_lora_rank"},
            {"ernie55.rope.dimension_count", "32", "rope.dimension_count"},
            {"ernie55.attention.indexer.key_length", "64", "attention.indexer.key_length"},
            {"ernie55.expert_count", "256", "expert_count"},
            {"ernie55.attention.compress_ratios", "[128, 64]", "non-uniform HCA compress_ratios"},
        };
        for (const BadDim& bd : bad_dims) {
            auto kv_bad = base_kv();
            kv_bad[bd.key] = bd.val;
            expect(write_synthetic_pack(tf.path, build_meta_json(kv_bad), exp_layout, 16384, exp_bytes), "write bad dim");
            Pack pk;
            std::string err;
            expect_err_has(open_pack(tf.path, pk, err), err, bd.needle, "mismatched kernel dim");
        }
    }

    // Optional real-pack check when invoked with a pack path
    if (argc > 1 && argv[1] && argv[1][0] != '\0') {
        Pack pk;
        std::string err;
        const bool ok = open_pack(argv[1], pk, err);
        if (!ok) std::fprintf(stderr, "FAIL: open_pack(%s): %s\n", argv[1], err.c_str());
        expect(ok, "open real pack passed via argv[1]");
    }

    std::printf("ernie55_pack_test: %d failures (fp8=%llu B/pos, fp16=%llu B/pos, 262K fp8=%.2f GiB)\n",
                g_fail, (unsigned long long) bpp_fp8, (unsigned long long) bpp_fp16,
                (double) p_ok.kv_bytes / (double) (1ull << 30));
    return g_fail ? 1 : 0;
}
