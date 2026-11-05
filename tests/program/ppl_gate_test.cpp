// tests/program/ppl_gate_test.cpp - the quality gate over a small base file: its shape checks, and the row of a chunk it reads.
#include "basalt/platform/system.hpp"
#include "basalt/program/ppl_gate.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using basalt::program::PplGate;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++g_fail; } } while (0)

// Odd vocabulary, so a row carries its padding. Each row's log-probs are a normalized ramp that depends on the chunk and the row.
constexpr int kCtx = 8, kVocab = 7, kChunks = 2;
constexpr int kRows = kCtx - 1 - kCtx / 2;

static float base_lp(int chunk, int row, int i) {
    double sum = 0.0;
    for (int j = 0; j < kVocab; ++j) sum += std::exp(-0.5 * j);
    return (float) (-0.5 * ((i + chunk + row) % kVocab) - std::log(sum));
}

static void write_base(const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    f.write("_logits_", 8);
    const uint32_t ctx = kCtx;
    const int32_t v = kVocab, c = kChunks;
    f.write((const char*) &ctx, 4);
    f.write((const char*) &v, 4);
    f.write((const char*) &c, 4);
    for (int k = 0; k < kChunks; ++k)
        for (int i = 0; i < kCtx; ++i) {
            const int32_t t = (k * 3 + i) % kVocab;
            f.write((const char*) &t, 4);
        }
    const float scale = 0.001f, lo = -20.0f;
    for (int k = 0; k < kChunks; ++k)
        for (int r = 0; r < kRows; ++r) {
            f.write((const char*) &scale, 4);
            f.write((const char*) &lo, 4);
            for (int i = 0; i < kVocab + 1; ++i) {
                const uint16_t q = i < kVocab ? (uint16_t) std::lround((base_lp(k, r, i) - lo) / scale) : 0;
                f.write((const char*) &q, 2);
            }
        }
}

int main() {
    const auto dir = std::filesystem::temp_directory_path() / ("basalt-ppl-gate-" + std::to_string((long long) basalt::platform::process_id()));
    std::filesystem::create_directories(dir);
    const std::string base = (dir / "base.kld").string(), out = (dir / "rows.tsv").string();
    write_base(base);
    PplGate::configure(base, out);
    PplGate& g = PplGate::get();
    CHECK(g.available());
    CHECK(g.first() == kCtx / 2);
    CHECK(g.end() == kCtx - 1);

    std::vector<int64_t> prompt;
    for (int i = 0; i <= kCtx / 2; ++i) prompt.push_back((3 + i) % kVocab);   // chunk 1's first half
    std::string err;
    CHECK(!g.begin(1, prompt.data(), (int64_t) prompt.size(), kVocab + 1, err));          // another vocabulary
    CHECK(!g.begin(kChunks, prompt.data(), (int64_t) prompt.size(), kVocab, err));        // no such chunk
    CHECK(!g.begin(1, prompt.data(), (int64_t) prompt.size() - 1, kVocab, err));          // a short prompt
    CHECK(!g.begin(0, prompt.data(), (int64_t) prompt.size(), kVocab, err));              // another chunk's tokens
    CHECK(g.begin(1, prompt.data(), (int64_t) prompt.size(), kVocab, err));
    CHECK(g.token(kCtx - 1) == (3 + kCtx - 1) % kVocab);

    // The base's own row scores nothing; the row before it, read in its place, scores more.
    std::vector<float> own(kVocab), other(kVocab);
    for (int i = 0; i < kVocab; ++i) {
        own[(size_t) i] = base_lp(1, 2, i);
        other[(size_t) i] = base_lp(1, 1, i);
    }
    CHECK(g.score(g.first() + 2, std::vector<float>(own), err));
    CHECK(g.score(g.first() + 2, std::vector<float>(other), err));
    CHECK(g.finish(err));

    std::ifstream rows(out);
    std::string line;
    std::vector<std::vector<std::string>> got;
    while (std::getline(rows, line)) {
        std::istringstream in(line);
        std::vector<std::string> c;
        for (std::string x; std::getline(in, x, '\t');) c.push_back(x);
        got.push_back(c);
    }
    CHECK(got.size() == 2);
    if (got.size() == 2) {
        CHECK(got[0][0] == "1");
        CHECK(got[0][1] == std::to_string(kCtx / 2 + 2));
        CHECK(std::atof(got[0][4].c_str()) > 0.0);
        const double same = std::atof(got[0][5].c_str()), moved = std::atof(got[1][5].c_str());
        CHECK(std::fabs(same) < 2e-2);
        CHECK(moved > same + 0.01);
        CHECK(got[0][6] == "1");
    }
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    if (g_fail == 0) std::printf("ppl_gate_test OK\n");
    return g_fail == 0 ? 0 : 1;
}
