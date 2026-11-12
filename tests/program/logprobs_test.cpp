// tests/program/logprobs_test.cpp - the P line fields (logprobs.hpp) from a row of kernels::row_logprobs: the pick's
// value, the alternatives in order, the clamp to 20, a masked value's floor and a row shorter than asked.
#include "logprobs.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace basalt::program::serve;
using basalt::kernels::kLogprobStride;

static int g_fail = 0;
#define CHECK(c)                                                                         \
    do {                                                                                 \
        if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

namespace {
std::vector<float> row(float pick, const std::vector<std::pair<int32_t, float>>& alts) {
    std::vector<float> r((size_t) kLogprobStride, 0.0f);
    r[0] = pick;
    for (size_t k = 0; k < (size_t) basalt::kernels::kLogprobMaxTop; ++k) {
        const int32_t id = k < alts.size() ? alts[k].first : -1;
        std::memcpy(&r[2 + 2 * k], &id, sizeof id);
        r[3 + 2 * k] = k < alts.size() ? alts[k].second : -INFINITY;
    }
    return r;
}
}  // namespace

int main() {
    const auto r = row(-0.25f, {{7, -0.25f}, {9, -1.5f}, {3, -2.0f}});
    CHECK(logprob_fields(r.data(), 7, 0) == " 7 -0.250000");
    CHECK(logprob_fields(r.data(), 7, 2) == " 7 -0.250000 7:-0.250000 9:-1.500000");
    CHECK(logprob_fields(r.data(), 7, 5) == " 7 -0.250000 7:-0.250000 9:-1.500000 3:-2.000000");   // fewer than asked
    std::vector<std::pair<int32_t, float>> many;
    for (int i = 0; i < 25; ++i) many.push_back({i, -0.1f * (float) i});
    const auto m = row(-1.0f, many);
    const std::string all = logprob_fields(m.data(), 4, 99);
    int n = 0;
    for (char c : all) n += c == ':';
    CHECK(n == kMaxTopLogprobs);   // clamped to 20
    const auto masked = row(-INFINITY, {{1, -INFINITY}});
    CHECK(logprob_fields(masked.data(), 5, 1) == " 5 -9999.000000 1:-9999.000000");
    if (g_fail) { std::printf("%d failures\n", g_fail); return 1; }
    std::printf("logprobs_test: ok\n");
    return 0;
}
