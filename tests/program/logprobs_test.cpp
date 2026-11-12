// tests/program/logprobs_test.cpp - a pick's P line fields (logprobs.hpp) against a plain double-precision reference.
// Random rows at the real vocabulary size: the pick's log probability, the K likeliest ids in order (ties to the lower
// id), the clamp to 20 alternatives, and a masked (-inf) logit.
#include "logprobs.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using namespace basalt::program::serve;

static int g_fail = 0;
#define CHECK(c)                                                                         \
    do {                                                                                 \
        if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

namespace {

uint64_t g_rng = 7;
uint64_t next_u64() {   // splitmix64: the same sequence on every compiler
    uint64_t z = (g_rng += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
float uniform(float lo, float hi) { return lo + (hi - lo) * (float) ((next_u64() >> 11) * (1.0 / 9007199254740992.0)); }

struct Parsed { int tok = -1; double lp = 0; std::vector<int> ids; std::vector<double> lps; };

Parsed parse(const std::string& s) {
    Parsed p;
    std::istringstream in(s);
    in >> p.tok >> p.lp;
    std::string alt;
    while (in >> alt) {
        const size_t c = alt.find(':');
        p.ids.push_back(std::atoi(alt.substr(0, c).c_str()));
        p.lps.push_back(std::atof(alt.substr(c + 1).c_str()));
    }
    return p;
}

void check_row(const std::vector<float>& lg, int tok, int top) {
    const int n = (int) lg.size();
    double mx = -INFINITY, sum = 0;
    for (float v : lg) mx = std::max(mx, (double) v);
    for (float v : lg) sum += std::exp((double) v - mx);
    const double lse = mx + std::log(sum);
    std::vector<int> ord((size_t) n);
    std::iota(ord.begin(), ord.end(), 0);
    std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return lg[(size_t) a] > lg[(size_t) b]; });
    const Parsed p = parse(logprob_fields(lg.data(), n, tok, top));
    const int want = std::min(std::max(top, 0), kMaxTopLogprobs);
    CHECK(p.tok == tok);
    const double ref = std::isfinite((double) lg[(size_t) tok]) ? (double) lg[(size_t) tok] - lse : -9999.0;
    CHECK(std::fabs(p.lp - ref) < 2e-6);
    CHECK((int) p.ids.size() == want);
    for (int k = 0; k < (int) p.ids.size() && k < want; ++k) {
        CHECK(p.ids[(size_t) k] == ord[(size_t) k]);
        CHECK(std::fabs(p.lps[(size_t) k] - ((double) lg[(size_t) ord[(size_t) k]] - lse)) < 2e-6);
    }
}

}  // namespace

int main() {
    const int kVocab = 151936;   // the real head width
    std::vector<float> lg((size_t) kVocab);
    for (int trial = 0; trial < 40; ++trial) {
        for (float& v : lg) v = uniform(-20.0f, 20.0f);
        if (trial % 4 == 1)   // a peaked row: one token far above the rest
            lg[(size_t) (next_u64() % (uint64_t) kVocab)] = 60.0f;
        if (trial % 4 == 2)   // ties: the lower id comes first
            for (int i = 0; i < 30; ++i) lg[(size_t) (next_u64() % (uint64_t) kVocab)] = 25.0f;
        const int tok = (int) (next_u64() % (uint64_t) kVocab);
        check_row(lg, tok, (int) (next_u64() % 25));
    }
    // a masked pick reports the floor, and the alternatives skip nothing finite
    for (float& v : lg) v = uniform(-5.0f, 5.0f);
    lg[3] = -INFINITY;
    check_row(lg, 3, 5);
    // a small vocabulary with fewer ids than asked
    std::vector<float> small = {1.0f, 3.0f, 2.0f};
    const Parsed p = parse(logprob_fields(small.data(), 3, 1, 20));
    CHECK(p.ids.size() == 3 && p.ids[0] == 1 && p.ids[1] == 2 && p.ids[2] == 0);
    CHECK(parse(logprob_fields(small.data(), 3, 1, 0)).ids.empty());
    if (g_fail) { std::printf("%d failures\n", g_fail); return 1; }
    std::printf("logprobs_test: ok\n");
    return 0;
}
