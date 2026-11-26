#include "basalt/models/deepseek4/compress.hpp"
#include "test_util.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <vector>

using namespace basalt::models::deepseek4;
using dstest::Dev;

namespace {
constexpr int D = 512;
constexpr int R = 64;
// Match deepseek-ai/DeepSeek-V4-Flash-Vision-Exp's configured RMS epsilon.
constexpr float eps = 1e-20f;

float bf16_rn(float value) {
    uint32_t bits = std::bit_cast<uint32_t>(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return std::bit_cast<float>(bits & 0xffff0000u);
}

void reference(CompressKind kind, int ratio, int n_tokens, const std::vector<float>& kv,
               const std::vector<float>& score, const std::vector<float>& ape, const std::vector<int32_t>& pos,
               const std::vector<float>& norm, const std::vector<float>& freq, std::vector<float>& state_kv,
               std::vector<float>& state_score, std::vector<float>& out, std::vector<int32_t>& rows) {
    const int width = kind == CompressKind::Csa ? 2 * D : D;
    const int state_rows = kind == CompressKind::Csa ? 2 * ratio : ratio;
    const int candidates = kind == CompressKind::Csa ? 2 * ratio : ratio;
    for (int t = 0; t < n_tokens; ++t) {
        const int p = pos[t];
        rows[t] = -1;
        if (p < 0) continue;
        const int slot = p % state_rows;
        for (int c = 0; c < width; ++c) {
            state_kv[(size_t) slot * width + c] = kv[(size_t) t * width + c];
            state_score[(size_t) slot * width + c] = score[(size_t) t * width + c] +
                ape[(size_t) (p % ratio) * width + c];
        }
        if ((p + 1) % ratio) continue;
        rows[t] = p / ratio;
        const int start = p + 1 - ratio;
        std::vector<double> compressed(D);
        for (int c = 0; c < D; ++c) {
            std::vector<double> s(candidates), v(candidates);
            double maximum = -std::numeric_limits<double>::infinity();
            for (int j = 0; j < candidates; ++j) {
                int source_pos, lane;
                if (kind == CompressKind::Csa) {
                    const bool previous = j < ratio;
                    source_pos = previous ? start - ratio + j : start + j - ratio;
                    lane = previous ? c : D + c;
                } else {
                    source_pos = start + j;
                    lane = c;
                }
                if (source_pos < 0) {
                    s[j] = -std::numeric_limits<double>::infinity();
                    v[j] = 0.0;
                } else {
                    const size_t index = (size_t) (source_pos % state_rows) * width + lane;
                    s[j] = state_score[index];
                    v[j] = state_kv[index];
                    maximum = std::max(maximum, s[j]);
                }
            }
            double denominator = 0.0, numerator = 0.0;
            if (std::isfinite(maximum)) {
                for (int j = 0; j < candidates; ++j) {
                    if (!std::isfinite(s[j])) continue;
                    const double weight = std::exp(s[j] - maximum);
                    denominator += weight;
                    numerator += v[j] * weight;
                }
            }
            compressed[c] = bf16_rn((float) (denominator > 0.0 ? numerator / denominator : 0.0));
        }
        double square_sum = 0.0;
        for (double value : compressed) square_sum += value * value;
        const double inverse = 1.0 / std::sqrt(square_sum / D + eps);
        std::vector<double> normalized(D);
        for (int c = 0; c < D; ++c) normalized[c] = bf16_rn((float) (compressed[c] * inverse * norm[c]));
        const int row = p / ratio;
        for (int c = D - R; c < D; c += 2) {
            const int pair = (c - (D - R)) / 2;
            const double angle = (double) start * freq[pair];
            const double cosine = std::cos(angle), sine = std::sin(angle);
            const double x0 = normalized[c], x1 = normalized[c + 1];
            normalized[c] = bf16_rn((float) (x0 * cosine - x1 * sine));
            normalized[c + 1] = bf16_rn((float) (x0 * sine + x1 * cosine));
        }
        for (int c = 0; c < D; ++c) out[(size_t) row * D + c] = (float) normalized[c];
    }
}

double max_score_error(const std::vector<float>& got, const std::vector<float>& ref) {
    double error = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (std::isinf(ref[i])) {
            if (got[i] != ref[i]) return INFINITY;
        } else {
            if (!std::isfinite(got[i])) return INFINITY;
            error = std::max(error, std::fabs((double) got[i] - ref[i]));
        }
    }
    return error;
}

bool run_case(CompressKind kind, int ratio, int total_tokens, int chunk, uint32_t seed) {
    const int width = kind == CompressKind::Csa ? 2 * D : D;
    const int state_rows = kind == CompressKind::Csa ? 2 * ratio : ratio;
    auto kv = dstest::randn((size_t) total_tokens * width, seed, 0.25f);
    auto score = dstest::randn((size_t) total_tokens * width, seed + 1, 0.7f);
    auto ape = dstest::randn((size_t) ratio * width, seed + 2, 0.2f);
    auto norm = dstest::randn(D, seed + 3, 0.05f, 1.0f);
    std::vector<float> freq(R / 2);
    for (int i = 0; i < R / 2; ++i) freq[i] = std::pow(10000.0f, -2.0f * i / R);
    std::vector<int32_t> pos(total_tokens);
    for (int i = 0; i < total_tokens; ++i) pos[i] = i;
    const size_t state_count = (size_t) state_rows * width;
    std::vector<float> ref_kv(state_count, 0.0f), ref_score(state_count, -INFINITY);
    std::vector<float> got_kv(state_count), got_score(state_count);
    std::vector<float> ref_out((size_t) total_tokens / ratio * D, 0.0f);
    std::vector<float> got_out(ref_out.size(), 0.0f);
    std::vector<int32_t> ref_rows(total_tokens, -1), got_rows(total_tokens, -1);
    reference(kind, ratio, total_tokens, kv, score, ape, pos, norm, freq, ref_kv, ref_score, ref_out, ref_rows);

    Dev<float> dkv(kv), dscore(score), dape(ape), dnorm(norm), dfreq(freq);
    Dev<int32_t> dpos(pos);
    Dev<float> dstate_kv(state_count), dstate_score(state_count), dout(got_out.size());
    Dev<int32_t> drow(total_tokens);
    dstest::check(cudaMemset(dout.p, 0, got_out.size() * sizeof(float)), "out clear");
    reset_compress_state(kind, ratio, dstate_kv.p, dstate_score.p, nullptr);
    for (int start = 0; start < total_tokens; start += chunk) {
        const int count = std::min(chunk, total_tokens - start);
        compress_projected(kind, ratio, count, dkv.p + (size_t) start * width, dscore.p + (size_t) start * width,
                           dape.p, dpos.p + start, dnorm.p, eps, dfreq.p, dstate_kv.p, dstate_score.p, dout.p,
                           drow.p + start, nullptr);
    }
    got_kv = dstate_kv.down();
    got_score = dstate_score.down();
    got_out = dout.down();
    got_rows = drow.down();
    const auto out_err = dstest::compare(got_out, ref_out);
    const auto kv_err = dstest::compare(got_kv, ref_kv);
    const double score_err = max_score_error(got_score, ref_score);
    const bool rows_ok = got_rows == ref_rows;
    std::printf("%s ratio=%d T=%d chunks=%d: output max abs %.3g rel RMS %.3g; state kv %.3g, score %.3g; row ids %s\n",
                kind == CompressKind::Csa ? "CSA" : "HCA", ratio, total_tokens, chunk, out_err.max_abs, out_err.rms_rel,
                kv_err.max_abs, score_err, rows_ok ? "match" : "MISMATCH");
    return out_err.max_abs < 3e-5 && kv_err.max_abs == 0.0 && score_err == 0.0 && rows_ok;
}

}  // namespace

int main() {
    int failures = 0;
    failures += !run_case(CompressKind::Csa, 4, 8, 8, 0x4c5341u);
    failures += !run_case(CompressKind::Csa, 4, 8, 4, 0x4c5341u);
    failures += !run_case(CompressKind::Hca, 128, 128, 8, 0x484341u);

    // One HCA group over a complete ratio-128 window, the larger real geometry.
    const int width = D;
    auto kv = dstest::randn(8 * width, 0x616263u, 0.25f);
    auto score = dstest::randn(8 * width, 0x616264u, 0.7f);
    auto ape = dstest::randn(128 * width, 0x616265u, 0.2f);
    std::vector<int32_t> pos(8);
    for (int i = 0; i < 8; ++i) pos[i] = 120 + i;
    std::vector<float> norm = dstest::randn(D, 0x616266u, 0.05f, 1.0f);
    std::vector<float> freq(R / 2);
    for (int i = 0; i < R / 2; ++i) freq[i] = std::pow(10000.0f, -2.0f * i / R);
    Dev<float> dkv(kv), dscore(score), dape(ape), dnorm(norm), dfreq(freq);
    Dev<int32_t> dpos(pos), drow(8);
    Dev<float> dstate_kv(128 * width), dstate_score(128 * width), dout(D);
    reset_compress_state(CompressKind::Hca, 128, dstate_kv.p, dstate_score.p, nullptr);
    std::vector<float> prefix_kv = dstest::randn(120 * width, 0x707265u, 0.25f);
    std::vector<float> prefix_score = dstest::randn(120 * width, 0x707273u, 0.7f);
    std::vector<float> all_kv = prefix_kv, all_score = prefix_score;
    all_kv.insert(all_kv.end(), kv.begin(), kv.end());
    all_score.insert(all_score.end(), score.begin(), score.end());
    std::vector<int32_t> all_pos(128);
    for (int i = 0; i < 128; ++i) all_pos[(size_t) i] = i;
    std::vector<float> ref_kv(128 * width), ref_score(128 * width, -INFINITY), ref_out(D);
    std::vector<int32_t> ref_rows(128, -1);
    reference(CompressKind::Hca, 128, 128, all_kv, all_score, ape, all_pos, norm, freq, ref_kv, ref_score, ref_out,
              ref_rows);
    Dev<float> dpkv(prefix_kv), dpscore(prefix_score);
    Dev<int32_t> prefix_pos(120);
    std::vector<int32_t> prefix_ids(120);
    for (int i = 0; i < 120; ++i) prefix_ids[(size_t) i] = i;
    prefix_pos.up(prefix_ids);
    for (int start = 0; start < 120; start += 8)
        compress_projected(CompressKind::Hca, 128, 8, dpkv.p + (size_t) start * width,
                           dpscore.p + (size_t) start * width, dape.p, prefix_pos.p + start, dnorm.p, eps, dfreq.p,
                           dstate_kv.p, dstate_score.p, dout.p, drow.p, nullptr);
    compress_projected(CompressKind::Hca, 128, 8, dkv.p, dscore.p, dape.p, dpos.p, dnorm.p, eps, dfreq.p,
                       dstate_kv.p, dstate_score.p, dout.p, drow.p, nullptr);
    const auto hca_out = dstest::compare(dout.down(), ref_out);
    const auto hca_kv = dstest::compare(dstate_kv.down(), ref_kv);
    const double hca_score = max_score_error(dstate_score.down(), ref_score);
    const auto hca_rows = drow.down();
    const bool rows_match = hca_rows == std::vector<int32_t>(ref_rows.end() - 8, ref_rows.end());
    std::printf("HCA continuation at token 120: output max abs %.3g rel RMS %.3g; state kv %.3g, score %.3g; row ids %s\n",
                hca_out.max_abs, hca_out.rms_rel, hca_kv.max_abs, hca_score, rows_match ? "match" : "MISMATCH");
    failures += !(hca_out.max_abs < 3e-5 && hca_kv.max_abs == 0.0 && hca_score == 0.0 && rows_match);
    const double us = dstest::bench_stream_us([&](void* stream) {
        compress_projected(CompressKind::Hca, 128, 8, dkv.p, dscore.p, dape.p, dpos.p, dnorm.p, eps, dfreq.p,
                           dstate_kv.p, dstate_score.p, dout.p, drow.p, stream);
    }, 100);
    const double bytes = (128.0 * D * 2 + 8.0 * D * 3 + D) * sizeof(float);
    dstest::bench_line("DS4 HCA compress T=8 ratio=128", us, bytes);

    return failures ? 1 : 0;
}
