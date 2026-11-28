// tests/program/quality_suite_test.cpp - CPU unit tests for the quality suite safetensors reader,
// Basalt row parser, shift alignment, KLD / top-1 / dp / PPL math, and conversation bootstrap.
#include "quality_suite.hpp"

#include "basalt/platform/file.hpp"
#include "basalt/platform/process.hpp"
#include "basalt/platform/system.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace basalt::program::quality;

namespace {

int g_fail = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL line %d: %s\n", line, expr);
        ++g_fail;
    }
}

void check_near_impl(double got, double want, double tol, const char* expr, int line) {
    if (!std::isfinite(got) || std::fabs(got - want) > tol) {
        std::fprintf(stderr, "FAIL line %d: %s: got %.9g want %.9g (tol %.3g)\n", line, expr, got, want, tol);
        ++g_fail;
    }
}

#define CHECK(x) check_impl((x), #x, __LINE__)
#define CHECK_NEAR(a, b, tol) check_near_impl((a), (b), (tol), #a " ~= " #b, __LINE__)

struct TensorBlob {
    std::string name;
    std::string dtype;
    size_t count = 0;
    std::vector<uint8_t> bytes;
};

template <typename T>
TensorBlob make_blob(const std::string& name, const std::string& dtype, const std::vector<T>& v) {
    TensorBlob b;
    b.name = name;
    b.dtype = dtype;
    b.count = v.size();
    b.bytes.resize(v.size() * sizeof(T));
    if (!v.empty()) std::memcpy(b.bytes.data(), v.data(), b.bytes.size());
    return b;
}

uint16_t f32_to_f16(float x) {
    uint32_t bits = 0;
    std::memcpy(&bits, &x, 4);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int exp = static_cast<int>((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = bits & 0x007FFFFFu;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

bool write_safetensors(const std::string& path, const std::vector<TensorBlob>& blobs, const std::string& model_name) {
    std::ostringstream hdr;
    hdr << "{";
    bool first = true;
    if (!model_name.empty()) {
        hdr << "\"__metadata__\":{\"model\":\"" << model_name << "\",\"format\":\"basalt-quality-v2-reference\"}";
        first = false;
    }
    uint64_t off = 0;
    for (const auto& b : blobs) {
        if (!first) hdr << ",";
        first = false;
        const uint64_t end = off + b.bytes.size();
        hdr << "\"" << b.name << "\":{\"dtype\":\"" << b.dtype << "\",\"shape\":[" << b.count
            << "],\"data_offsets\":[" << off << "," << end << "]}";
        off = end;
    }
    hdr << "}";
    const std::string h = hdr.str();
    const uint64_t hlen = h.size();
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(&hlen), 8);
    f.write(h.data(), static_cast<std::streamsize>(h.size()));
    for (const auto& b : blobs) {
        if (!b.bytes.empty()) f.write(reinterpret_cast<const char*>(b.bytes.data()), static_cast<std::streamsize>(b.bytes.size()));
    }
    return f.good();
}

double exact_kld(double p0, double p1, double lp, double q0, double q1, double lq) {
    return p0 * std::log(p0 / q0) + p1 * std::log(p1 / q1) + lp * std::log(lp / lq);
}

void test_fixture_and_scoring(const std::string& tmpdir) {
    // Matches tools/quality_suite/test_score.py
    const std::vector<int32_t> tokens = {2, 10, 11, 12, 2, 10, 13};
    const std::vector<uint8_t> roles = {0, 1, 1, 1, 0, 1, 1};
    const std::vector<int64_t> conv_off = {0, 4, 7};
    const std::vector<int64_t> row_off = {0, 2, 4, 6, 6, 8, 10, 10};
    const std::vector<int32_t> topk_ids = {10, 11, 10, 11, 10, 11, 10, 11, 10, 11};
    const float lp10 = static_cast<float>(std::log(0.6));
    const float lp11 = static_cast<float>(std::log(0.3));
    const std::vector<float> topk_lp = {lp10, lp11, lp10, lp11, lp10, lp11, lp10, lp11, lp10, lp11};
    const std::vector<float> leftover = {0.1f, 0.1f, 0.1f, 0.0f, 0.1f, 0.1f, 0.0f};

    const std::string ref_path = (fs::path(tmpdir) / "ref.safetensors").string();
    CHECK(write_safetensors(ref_path, {
        make_blob("tokens", "I32", tokens),
        make_blob("roles", "U8", roles),
        make_blob("conv_off", "I64", conv_off),
        make_blob("row_off", "I64", row_off),
        make_blob("topk_ids", "I32", topk_ids),
        make_blob("topk_lp", "F32", topk_lp),
        make_blob("leftover", "F32", leftover),
    }, "ToyModel (BF16)"));

    Reference ref;
    std::string err;
    CHECK(load_reference(ref_path, ref, err));
    CHECK(ref.model_name == "ToyModel (BF16)");
    CHECK(ref.num_convs() == 2);
    CHECK(ref.tokens == tokens);
    CHECK(ref.roles == roles);

    // 1. Identical model (shift = 1: pos is source position)
    {
        Scorer sc(ref);
        std::vector<ParsedRow> r0, r1;
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "0 10 0 0 10:%.9f 11:%.9f 99:-9.0\n"
            "1 11 0 0 10:%.9f 11:%.9f 99:-9.0\n"
            "2 12 0 0 10:%.9f 11:%.9f 99:-9.0\n",
            std::log(0.6), std::log(0.3), std::log(0.6), std::log(0.3), std::log(0.6), std::log(0.3));
        CHECK(parse_basalt_rows(buf, std::strlen(buf), r0, err));
        CHECK(sc.add_conversation(0, r0, err));

        std::snprintf(buf, sizeof(buf),
            "0 10 0 0 10:%.9f 11:%.9f 99:-9.0\n"
            "1 13 0 0 10:%.9f 11:%.9f 99:-9.0\n",
            std::log(0.6), std::log(0.3), std::log(0.6), std::log(0.3));
        CHECK(parse_basalt_rows(buf, std::strlen(buf), r1, err));
        CHECK(sc.add_conversation(1, r1, err));
        CHECK(sc.missing_ref_ids() == 0);

        Summary sum;
        CHECK(summarize(sc.scored_rows(), RoleFilter::Assistant, 200, 0, sum, err));
        CHECK(sum.n_rows == 5);
        CHECK(sum.n_convs == 2);
        CHECK(sum.rows_target_outside_topk == 2);
        CHECK_NEAR(sum.kld_mean, 0.0, 1e-6);
        CHECK_NEAR(sum.top1, 1.0, 1e-9);
        CHECK_NEAR(sum.ppl_ratio, 1.0, 1e-6);
    }

    // 2. Known KLD, top-1, dp, PPL ratio, and shift = 0 (pos is target index)
    {
        Scorer sc(ref);
        std::vector<ParsedRow> r0, r1;
        char buf[512];
        // conv 0: q = (0.5, 0.3), row 2 has token 11 higher than 10 -> top1 mismatch on 1 of 5 rows?
        // Actually keep q=(0.5, 0.3) for rows 0..2 and (0.6, 0.25) for rows 3..4 to match test_score.py exact KLD.
        std::snprintf(buf, sizeof(buf),
            "1 10 0 0 10:%.9f 11:%.9f 99:-9.0\n"
            "2 11 0 0 10:%.9f 11:%.9f 99:-9.0\n"
            "3 12 0 0 10:%.9f 11:%.9f 99:-9.0\n",
            std::log(0.5), std::log(0.3), std::log(0.5), std::log(0.3), std::log(0.5), std::log(0.3));
        CHECK(parse_basalt_rows(buf, std::strlen(buf), r0, err));
        CHECK(sc.add_conversation(0, r0, err));

        std::snprintf(buf, sizeof(buf),
            "1 10 0 0 10:%.9f 11:%.9f 99:-9.0\n"
            "2 13 0 0 10:%.9f 11:%.9f 99:-9.0\n",
            std::log(0.6), std::log(0.25), std::log(0.6), std::log(0.25));
        CHECK(parse_basalt_rows(buf, std::strlen(buf), r1, err));
        CHECK(sc.add_conversation(1, r1, err));

        Summary sum;
        CHECK(summarize(sc.scored_rows(), RoleFilter::Assistant, 200, 0, sum, err));
        const double k0 = exact_kld(0.6, 0.3, 0.1, 0.5, 0.3, 0.2);
        const double k1 = exact_kld(0.6, 0.3, 0.1, 0.6, 0.25, 0.15);
        const double want_mean = (3.0 * k0 + 2.0 * k1) / 5.0;
        CHECK_NEAR(sum.kld_mean, want_mean, 1e-6);
        CHECK_NEAR(sum.top1, 1.0, 1e-9);
        const double want_ln = (std::log(0.6 / 0.5) + std::log(0.3 / 0.3) + std::log(0.6 / 0.6)) / 3.0;
        CHECK_NEAR(sum.ln_ppl_ratio, want_ln, 1e-6);
        CHECK_NEAR(sum.ppl_ratio, std::exp(want_ln), 1e-6);
        CHECK_NEAR(sc.scored_rows()[0].dp, 0.5 - 0.6, 1e-6);
        CHECK(sum.kld_se > 0.0);
        CHECK(sum.kld_naive_se > 0.0);
    }

    // 3. Missing reference id falls back to row min_lp and increments missing_ref_ids()
    {
        Scorer sc(ref);
        std::vector<ParsedRow> r0;
        const char* txt =
            "0 10 0 0 10:-0.5 99:-5.0\n"
            "1 11 0 0 10:-0.5 11:-1.2\n"
            "2 12 0 0 10:-0.5 11:-1.2\n";
        CHECK(parse_basalt_rows(txt, std::strlen(txt), r0, err));
        CHECK(sc.add_conversation(0, r0, err));
        CHECK(sc.missing_ref_ids() == 1);
    }

    // 4. F16 tensor loading
    {
        std::vector<uint16_t> lp_f16, left_f16;
        for (float v : topk_lp) lp_f16.push_back(f32_to_f16(v));
        for (float v : leftover) left_f16.push_back(f32_to_f16(v));
        const std::string f16_path = (fs::path(tmpdir) / "ref_f16.safetensors").string();
        CHECK(write_safetensors(f16_path, {
            make_blob("tokens", "I32", tokens),
            make_blob("roles", "U8", roles),
            make_blob("conv_off", "I64", conv_off),
            make_blob("row_off", "I64", row_off),
            make_blob("topk_ids", "I32", topk_ids),
            make_blob("topk_lp", "F16", lp_f16),
            make_blob("leftover", "F16", left_f16),
        }, ""));
        Reference ref16;
        CHECK(load_reference(f16_path, ref16, err));
        CHECK_NEAR(ref16.topk_lp[0], lp10, 1e-3);
        CHECK_NEAR(ref16.leftover[0], 0.1, 1e-3);
    }
}

void test_smoke_and_max_context() {
    Reference ref;
    ref.conv_off.push_back(0);
    for (int i = 0; i < 41; ++i) ref.conv_off.push_back(ref.conv_off.back() + 100);
    ref.conv_off.push_back(ref.conv_off.back() + 6000);   // conv 41: length 6000

    int64_t skipped = 0;
    const auto smoke = select_conversations(ref, true, 5120, skipped);
    CHECK(smoke.size() == 3);
    CHECK(smoke[0] == 0 && smoke[1] == 20 && smoke[2] == 40);
    CHECK(skipped == 0);

    const auto full = select_conversations(ref, false, 5120, skipped);
    CHECK(full.size() == 41);
    CHECK(skipped == 1);
}

void test_score_nonzero_offset(const std::string& tmpdir) {
    // Unslimmed conversation: BOS (0), user prompt tokens (role 2) at idx 1..2, assistant tokens (role 1) at idx 3..4.
    // Reference has rows at positions 0, 1, 2, 3; first assistant row is at pos 2 (predicting token idx 3).
    const std::vector<int32_t> tokens = {2, 20, 21, 10, 11};
    const std::vector<uint8_t> roles = {0, 2, 2, 1, 1};
    const std::vector<int64_t> conv_off = {0, 5};
    const std::vector<int64_t> row_off = {0, 2, 4, 6, 8, 8};
    const std::vector<int32_t> topk_ids = {20, 21, 20, 21, 10, 11, 10, 11};
    const float lp0 = static_cast<float>(std::log(0.6));
    const float lp1 = static_cast<float>(std::log(0.3));
    const std::vector<float> topk_lp = {lp0, lp1, lp0, lp1, lp0, lp1, lp0, lp1};
    const std::vector<float> leftover = {0.1f, 0.1f, 0.1f, 0.1f, 0.0f};
    const std::string path = (fs::path(tmpdir) / "ref_offset.safetensors").string();
    CHECK(write_safetensors(path, {
        make_blob("tokens", "I32", tokens),
        make_blob("roles", "U8", roles),
        make_blob("conv_off", "I64", conv_off),
        make_blob("row_off", "I64", row_off),
        make_blob("topk_ids", "I32", topk_ids),
        make_blob("topk_lp", "F32", topk_lp),
        make_blob("leftover", "F32", leftover),
    }, ""));
    Reference ref;
    std::string err;
    CHECK(load_reference(path, ref, err));
    CHECK(first_scored_pos(ref, 0, RoleFilter::Assistant) == 2);
    CHECK(first_scored_pos(ref, 0, RoleFilter::Other) == 0);
    CHECK(first_scored_pos(ref, 0, RoleFilter::All) == 0);
    // Engine ran with score=2, so it only emitted rows at pos 2 and 3.
    Scorer sc(ref, RoleFilter::Assistant);
    std::vector<ParsedRow> rows;
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "2 10 0 0 10:%.9f 11:%.9f 99:-9.0\n"
        "3 11 0 0 10:%.9f 11:%.9f 99:-9.0\n",
        std::log(0.6), std::log(0.3), std::log(0.6), std::log(0.3));
    CHECK(parse_basalt_rows(buf, std::strlen(buf), rows, err));
    CHECK(sc.add_conversation(0, rows, err));
    Summary sum;
    CHECK(summarize(sc.scored_rows(), RoleFilter::Assistant, 50, 0, sum, err));
    CHECK(sum.n_rows == 2);
    CHECK_NEAR(sum.kld_mean, 0.0, 1e-6);
    CHECK_NEAR(sum.top1, 1.0, 1e-9);
    CHECK_NEAR(sum.ppl_ratio, 1.0, 1e-6);
}

void test_role_filters() {
    std::vector<ScoredRow> rows;
    ScoredRow a; a.conv = 0; a.role = 1; a.kld = 0.1; a.same = 1.0; a.has_target = true; a.nllb = 1.0; a.nllq = 1.1;
    ScoredRow o; o.conv = 0; o.role = 2; o.kld = 0.5; o.same = 0.0; o.has_target = true; o.nllb = 2.0; o.nllq = 2.5;
    rows.push_back(a);
    rows.push_back(o);

    Summary s;
    std::string err;
    CHECK(summarize(rows, RoleFilter::Assistant, 100, 0, s, err));
    CHECK(s.n_rows == 1);
    CHECK_NEAR(s.kld_mean, 0.1, 1e-9);

    CHECK(summarize(rows, RoleFilter::Other, 100, 0, s, err));
    CHECK(s.n_rows == 1);
    CHECK_NEAR(s.kld_mean, 0.5, 1e-9);

    CHECK(summarize(rows, RoleFilter::All, 100, 0, s, err));
    CHECK(s.n_rows == 2);
    CHECK_NEAR(s.kld_mean, 0.3, 1e-9);
    CHECK_NEAR(s.top1, 0.5, 1e-9);
}

}  // namespace

int main() {
    const std::string tmpdir = basalt::platform::temp_dir() + "/basalt-qs-test-" +
                               std::to_string(static_cast<long long>(basalt::platform::process_id()));
    std::error_code ec;
    fs::create_directories(tmpdir, ec);
    test_fixture_and_scoring(tmpdir);
    test_smoke_and_max_context();
    test_role_filters();
    test_score_nonzero_offset(tmpdir);
    fs::remove_all(tmpdir, ec);
    if (g_fail != 0) {
        std::fprintf(stderr, "quality_suite_test: %d failures\n", g_fail);
        return 1;
    }
    return 0;
}
