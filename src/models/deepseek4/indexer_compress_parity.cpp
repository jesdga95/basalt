#include "basalt/models/deepseek4/compress.hpp"
#include "parity_util.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <vector>

using namespace basalt::models::deepseek4;
using ds4test::Dev;

namespace {
constexpr int D = kIndexerHeadDim;
constexpr int R = 64;
constexpr int Ratio = kIndexerRatio;
constexpr int Width = 2 * D;
constexpr int StateRows = 2 * Ratio;
constexpr float eps = 1e-20f;

float bf16_rn(float value) {
    uint32_t bits = std::bit_cast<uint32_t>(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return std::bit_cast<float>(bits & 0xffff0000u);
}

float fp4_rn(float value) {
    constexpr float levels[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float magnitude = std::fabs(value);
    int selected = 7;
    for (int i = 0; i < 7; ++i) {
        const float midpoint = (levels[i] + levels[i + 1]) * .5f;
        if (magnitude < midpoint || (magnitude == midpoint && (i & 1) == 0)) {
            selected = i;
            break;
        }
    }
    return std::copysign(levels[selected], value);
}

float pow2_ceil(float value) {
    int exponent = std::ilogb(value);
    if (std::scalbn(1.f, exponent) != value) ++exponent;
    return std::scalbn(1.f, exponent);
}

void reference(int n_tokens, const std::vector<float>& kv, const std::vector<float>& score,
               const std::vector<float>& ape, const std::vector<int32_t>& pos, const std::vector<float>& norm,
               const std::vector<float>& freq, std::vector<float>& state_kv, std::vector<float>& state_score,
               std::vector<float>& out, std::vector<int32_t>& rows) {
    for (int t = 0; t < n_tokens; ++t) {
        const int p = pos[t];
        rows[t] = -1;
        if (p < 0) continue;
        const int slot = p % StateRows;
        for (int c = 0; c < Width; ++c) {
            state_kv[(size_t) slot * Width + c] = kv[(size_t) t * Width + c];
            state_score[(size_t) slot * Width + c] = score[(size_t) t * Width + c] + ape[(size_t) (p % Ratio) * Width + c];
        }
        if ((p + 1) % Ratio) continue;
        rows[t] = p / Ratio;
        const int start = p + 1 - Ratio;
        std::vector<float> values(D);
        for (int c = 0; c < D; ++c) {
            double maximum = -std::numeric_limits<double>::infinity();
            for (int j = 0; j < 2 * Ratio; ++j) {
                const int source_pos = j < Ratio ? start - Ratio + j : start + j - Ratio;
                if (source_pos < 0) continue;
                const int lane = j < Ratio ? c : D + c;
                maximum = std::max(maximum, (double) state_score[(size_t) (source_pos % StateRows) * Width + lane]);
            }
            double denominator = 0.0, numerator = 0.0;
            for (int j = 0; j < 2 * Ratio; ++j) {
                const int source_pos = j < Ratio ? start - Ratio + j : start + j - Ratio;
                if (source_pos < 0) continue;
                const int lane = j < Ratio ? c : D + c;
                const size_t index = (size_t) (source_pos % StateRows) * Width + lane;
                const double weight = std::exp((double) state_score[index] - maximum);
                denominator += weight;
                numerator += state_kv[index] * weight;
            }
            values[c] = bf16_rn((float) (numerator / denominator));
        }
        double square_sum = 0.0;
        for (float value : values) square_sum += (double) value * value;
        const double inverse = 1.0 / std::sqrt(square_sum / D + eps);
        for (int c = 0; c < D; ++c) values[c] = bf16_rn((float) (values[c] * inverse * norm[c]));
        for (int c = D - R; c < D; c += 2) {
            const int pair = (c - (D - R)) / 2;
            const double angle = (double) start * freq[pair];
            const double cosine = std::cos(angle), sine = std::sin(angle);
            const double x0 = values[c], x1 = values[c + 1];
            values[c] = bf16_rn((float) (x0 * cosine - x1 * sine));
            values[c + 1] = bf16_rn((float) (x0 * sine + x1 * cosine));
        }
        for (int stride = 1; stride < D; stride <<= 1) {
            std::vector<float> next(D);
            for (int c = 0; c < D; ++c)
                next[c] = bf16_rn((c & stride) ? values[c ^ stride] - values[c] : values[c] + values[c ^ stride]);
            values.swap(next);
        }
        for (float& value : values) value = bf16_rn(value * (1.f / std::sqrt((float) D)));
        for (int group = 0; group < D / 32; ++group) {
            float amax = 6.f * 0x1p-126f;
            for (int c = group * 32; c < (group + 1) * 32; ++c) amax = std::max(amax, std::fabs(values[c]));
            const float scale = pow2_ceil(amax * (1.f / 6.f));
            for (int c = group * 32; c < (group + 1) * 32; ++c)
                (void) fp4_rn;   // --kv f32 keeps the unrounded rotated values
                (void) scale;
        }
        for (int c = 0; c < D; ++c) out[(size_t) (p / Ratio) * D + c] = values[c];
    }
}

bool run_case(int total_tokens, int chunk, uint32_t seed) {
    auto kv = ds4test::randn((size_t) total_tokens * Width, seed, .25f);
    auto score = ds4test::randn((size_t) total_tokens * Width, seed + 1, .7f);
    auto ape = ds4test::randn((size_t) Ratio * Width, seed + 2, .2f);
    auto norm = ds4test::randn(D, seed + 3, .05f, 1.f);
    const std::vector<float> freq = ds4test::yarn_frequencies(R, 160000.f, 16.f, 65536, 32.f, 1.f);
    std::vector<int32_t> pos(total_tokens);
    for (int i = 0; i < total_tokens; ++i) pos[i] = i;
    const size_t state_count = (size_t) StateRows * Width;
    std::vector<float> ref_kv(state_count, 0.f), ref_score(state_count, -INFINITY);
    std::vector<float> got_kv(state_count), got_score(state_count);
    std::vector<float> ref_out((size_t) total_tokens / Ratio * D), got_out(ref_out.size());
    std::vector<int32_t> ref_rows(total_tokens, -1), got_rows(total_tokens, -1);
    reference(total_tokens, kv, score, ape, pos, norm, freq, ref_kv, ref_score, ref_out, ref_rows);

    Dev<float> dkv(kv), dscore(score), dape(ape), dnorm(norm), dfreq(freq);
    Dev<int32_t> dpos(pos), drow(total_tokens);
    Dev<float> dstate_kv(state_count), dstate_score(state_count), dout(got_out.size());
    reset_indexer_compress_state(dstate_kv.p, dstate_score.p, nullptr);
    for (int start = 0; start < total_tokens; start += chunk) {
        const int count = std::min(chunk, total_tokens - start);
        indexer_compress_projected(count, dkv.p + (size_t) start * Width, dscore.p + (size_t) start * Width,
                                   dape.p, dpos.p + start, dnorm.p, eps, dfreq.p, dstate_kv.p, dstate_score.p,
                                   dout.p, drow.p + start, CacheFormat::F32, nullptr);
    }
    got_kv = dstate_kv.down();
    got_score = dstate_score.down();
    got_out = dout.down();
    got_rows = drow.down();
    const auto out_err = ds4test::compare(got_out, ref_out);
    const auto kv_err = ds4test::compare(got_kv, ref_kv);
    double score_err = 0.0;
    for (size_t i = 0; i < ref_score.size(); ++i) {
        if (std::isinf(ref_score[i])) {
            if (got_score[i] != ref_score[i]) score_err = INFINITY;
        } else {
            score_err = std::max(score_err, std::fabs((double) got_score[i] - ref_score[i]));
        }
    }
    std::printf("DS4 indexer compressor T=%d chunk=%d: max abs %.3g rel RMS %.3g; state %.3g/%.3g; rows %s\n",
                total_tokens, chunk, out_err.max_abs, out_err.rms_rel, kv_err.max_abs, score_err,
                got_rows == ref_rows ? "match" : "MISMATCH");
    const double us = ds4test::bench_stream_us([&](void* stream) {
        indexer_compress_projected(std::min(chunk, 8), dkv.p, dscore.p, dape.p, dpos.p, dnorm.p, eps, dfreq.p,
                                   dstate_kv.p, dstate_score.p, dout.p, drow.p, CacheFormat::F32, stream);
    }, 100);
    const double bytes = (double) std::min(chunk, 8) * Width * 2 * sizeof(float) +
                         (double) Ratio * Width * sizeof(float) + D * 2 * sizeof(float);
    ds4test::bench_line("DS4 indexer ratio-4 compressor", us, bytes);
    return out_err.max_abs < 5e-3 && kv_err.max_abs == 0.0 && score_err == 0.0 && got_rows == ref_rows;
}
}  // namespace

int main() {
    int failures = 0;
    failures += !run_case(16, 8, 0x493458u);
    failures += !run_case(16, 4, 0x493458u);
    return failures ? 1 : 0;
}
