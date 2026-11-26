#include "basalt/models/deepseek4/rope.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

namespace {

bool close(float a, float b) { return std::fabs(a - b) <= 2e-7f * std::max(1.f, std::fabs(b)); }

float source_yarn_frequency(int pair, float base, float factor, int original_context, float beta_fast,
                            float beta_slow) {
    constexpr int dim = 64;
    constexpr double pi = 3.14159265358979323846;
    const auto correction_dim = [&](float rotations) {
        return dim * std::log((double) original_context / (rotations * 2.0 * pi)) / (2.0 * std::log((double) base));
    };
    const int low = std::max(0, (int) std::floor(correction_dim(beta_fast)));
    const int high = std::min(dim - 1, (int) std::ceil(correction_dim(beta_slow)));
    float ramp_high = (float) high;
    if ((float) low == ramp_high) ramp_high += 0.001f;
    const float ramp = std::clamp(((float) pair - (float) low) / (ramp_high - (float) low), 0.f, 1.f);
    const float smooth = 1.f - ramp;
    const float raw = std::pow(base, -2.f * (float) pair / (float) dim);
    return raw / factor * (1.f - smooth) + raw * smooth;
}

}  // namespace

int main() {
    using namespace basalt::models::deepseek4;
    RopeParams g;
    g.rope_dim = 64;
    g.rope_base = 10000.f;
    g.compressed_base = 160000.f;
    g.factor = 16.f;
    g.original_context = 65536;
    g.beta_fast = 32.f;
    g.beta_slow = 1.f;

    RopeTables plain, csa, hca;
    std::string err;
    if (!make_rope_tables(g, 0, plain, err) || !make_rope_tables(g, 4, csa, err) ||
        !make_rope_tables(g, 128, hca, err)) {
        std::fprintf(stderr, "deepseek4 RoPE: %s\n", err.c_str());
        return 1;
    }
    if (plain.attention.mscale != 1.f || csa.attention.mscale != 1.f || hca.attention.mscale != 1.f ||
        plain.frequency[0] != 1.f || csa.frequency[0] != 1.f || hca.frequency[0] != 1.f) {
        std::puts("deepseek4 RoPE: phase magnitude or first frequency differs from the source contract");
        return 1;
    }
    for (int pair = 0; pair < kRopePairs; ++pair) {
        const float expected_plain = std::pow(g.rope_base, -2.f * (float) pair / (float) g.rope_dim);
        const float expected_yarn = source_yarn_frequency(pair, g.compressed_base, g.factor,
                                                          g.original_context, g.beta_fast, g.beta_slow);
        if (!close(plain.frequency[(size_t) pair], expected_plain) ||
            !close(csa.frequency[(size_t) pair], expected_yarn) ||
            csa.frequency[(size_t) pair] != hca.frequency[(size_t) pair] ||
            csa.frequency[(size_t) pair] != csa.attention.step[pair]) {
            std::fprintf(stderr, "deepseek4 RoPE: frequency mismatch at pair %d\n", pair);
            return 1;
        }
    }
    if (make_rope_tables(g, 2, plain, err) || err.empty()) {
        std::puts("deepseek4 RoPE: unsupported compression ratio was accepted");
        return 1;
    }
    g.rope_base = std::numeric_limits<float>::quiet_NaN();
    if (make_rope_tables(g, 0, plain, err) || err.empty()) {
        std::puts("deepseek4 RoPE: non-finite base was accepted");
        return 1;
    }
    std::puts("deepseek4 RoPE: sliding and compressed YaRN frequencies match the source formula");
    return 0;
}
