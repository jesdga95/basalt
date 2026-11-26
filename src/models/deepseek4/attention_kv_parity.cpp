#include "basalt/models/deepseek4/attention.hpp"
#include "parity_util.hpp"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstring>
#include <stdexcept>

using namespace ds4test;

namespace {

constexpr int D = 512;
constexpr int Rope = 64;
constexpr int QuantWidth = D - Rope;

float bf16_round(float value) { return (float) __float2bfloat16_rn(value); }

float e4m3_round(float value) { return (float) __nv_fp8_e4m3(value); }

float pow2_ceil(float value) {
    int exponent = 0;
    const float fraction = std::frexp(value, &exponent);
    return std::ldexp(1.f, fraction == 0.5f ? exponent - 1 : exponent);
}

std::vector<float> reference(const std::vector<float>& projected, const std::vector<float>& norm,
                             const std::vector<int32_t>& positions, const basalt::kernels::MqaRopeTable& rope) {
    std::vector<float> out(projected.size());
    for (size_t t = 0; t < positions.size(); ++t) {
        const size_t base = t * D;
        double square_sum = 0.0;
        for (int i = 0; i < D; ++i) {
            out[base + i] = bf16_round(projected[base + i]);
            square_sum += (double) out[base + i] * out[base + i];
        }
        const double inverse = 1.0 / std::sqrt(square_sum / D + 1e-20);
        for (int i = 0; i < D; ++i) out[base + i] = bf16_round((float) (out[base + i] * inverse * norm[i]));
        for (int pair = 0; pair < Rope / 2; ++pair) {
            const int d = QuantWidth + pair * 2;
            const double angle = (double) positions[t] * rope.step[pair];
            const double cosine = std::cos(angle) * rope.mscale;
            const double sine = std::sin(angle) * rope.mscale;
            const float a = out[base + d], b = out[base + d + 1];
            out[base + d] = bf16_round((float) (a * cosine - b * sine));
            out[base + d + 1] = bf16_round((float) (a * sine + b * cosine));
        }
        for (int group = 0; group < QuantWidth / 64; ++group) {
            float amax = 0.f;
            for (int i = group * 64; i < (group + 1) * 64; ++i) amax = std::max(amax, std::fabs(out[base + i]));
            const float scale = pow2_ceil(std::max(amax, 1e-4f) * (1.f / 448.f));
            for (int i = group * 64; i < (group + 1) * 64; ++i) {
                const float q = std::clamp(out[base + i] / scale, -448.f, 448.f);
                out[base + i] = bf16_round(e4m3_round(q) * scale);
            }
        }
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace basalt::models::deepseek4;
    constexpr int T = 5;
    constexpr float eps = 1e-20f;
    std::vector<float> projected = randn((size_t) T * D, 0x4b565052, 0.7f);
    std::vector<float> norm = randn(D, 0x4b56504e, 0.1f, 1.f);
    const std::vector<int32_t> positions = {0, 3, 127, 4096, 65535};
    basalt::kernels::MqaRopeTable rope;
    const std::vector<float> frequency = yarn_frequencies(Rope, 160000.f, 16.f, 65536, 32.f, 1.f);
    std::copy(frequency.begin(), frequency.end(), rope.step);
    Dev<float> dprojected(projected), dnorm(norm), dout((size_t) T * D);
    Dev<int32_t> dpositions(positions);
    prepare_attention_kv(dprojected.p, dnorm.p, eps, dpositions.p, rope, T, dout.p, nullptr);
    const auto got = dout.down();
    const auto ref = reference(projected, norm, positions, rope);
    const Err error = compare(got, ref);

    const std::vector<float> initial((size_t) 128 * D, -13.f);
    Dev<float> dcache(initial);
    const std::vector<int32_t> cache_positions = {1, 3, 126, 127, 128};
    Dev<int32_t> dcache_positions(cache_positions);
    cache_attention_window(dcache.p, dout.p, dcache_positions.p, T, nullptr);
    const auto cache = dcache.down();
    std::vector<float> cache_ref = initial;
    for (int t = 0; t < T; ++t) {
        const size_t slot = (size_t) cache_positions[(size_t) t] % 128;
        std::copy_n(got.data() + (size_t) t * D, D, cache_ref.data() + slot * D);
    }
    const bool cache_matches = cache == cache_ref;

    bool independent = true;
    for (int t = 0; t < T; ++t) {
        std::vector<float> row(projected.begin() + (size_t) t * D, projected.begin() + (size_t) (t + 1) * D);
        Dev<float> dx(row), dy(D);
        Dev<int32_t> dp(std::vector<int32_t>{positions[(size_t) t]});
        prepare_attention_kv(dx.p, dnorm.p, eps, dp.p, rope, 1, dy.p, nullptr);
        const auto single = dy.down();
        independent = independent && std::memcmp(single.data(), got.data() + (size_t) t * D, D * sizeof(float)) == 0;
    }

    std::printf("DeepSeek4 KV boundary D=%d T=%d: CPU-double max abs %.6g rel RMS %.6g; cache rows %s; "
                "token-width independence %s\n", D, T, error.max_abs, error.rms_rel,
                cache_matches ? "match" : "DIFFERS", independent ? "bitwise" : "DIFFERS");
    if (error.max_abs > 0.02 || error.rms_rel > 0.002 || !cache_matches || !independent) return 1;
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        const double us = bench_stream_us([&](void* stream) {
            prepare_attention_kv(dprojected.p, dnorm.p, eps, dpositions.p, rope, T, dout.p, stream);
        });
        bench_line("deepseek4 KV boundary T=5", us, (double) T * (2 * D + D + 1) * sizeof(float));
        const double cache_us = bench_stream_us([&](void* stream) {
            cache_attention_window(dcache.p, dout.p, dcache_positions.p, T, stream);
        });
        bench_line("deepseek4 KV ring write T=5", cache_us, (double) T * D * 2 * sizeof(float));
    }
    return 0;
}
