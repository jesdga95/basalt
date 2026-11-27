#include "basalt/models/deepseek4/compress.hpp"
#include "parity_util.hpp"

#include <cuda_fp8.h>

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
constexpr int D = 512;
constexpr int R = 64;
// Match deepseek-ai/DeepSeek-V4-Flash-Vision-Exp's configured RMS epsilon.
constexpr float eps = 1e-20f;

float bf16_rn(float value) {
    uint32_t bits = std::bit_cast<uint32_t>(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return std::bit_cast<float>(bits & 0xffff0000u);
}

uint16_t bf16_bits(float value) {
    uint32_t bits = std::bit_cast<uint32_t>(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t) (bits >> 16);
}

float bf16_float(uint16_t value) { return std::bit_cast<float>((uint32_t) value << 16); }

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
        // --kv f32 keeps the reference's unrounded normalized row (the packed formats round it on the device).
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
    auto kv = ds4test::randn((size_t) total_tokens * width, seed, 0.25f);
    auto score = ds4test::randn((size_t) total_tokens * width, seed + 1, 0.7f);
    auto ape = ds4test::randn((size_t) ratio * width, seed + 2, 0.2f);
    auto norm = ds4test::randn(D, seed + 3, 0.05f, 1.0f);
    const std::vector<float> freq = ds4test::yarn_frequencies(R, 160000.f, 16.f, 65536, 32.f, 1.f);
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
    ds4test::check(cudaMemset(dout.p, 0, got_out.size() * sizeof(float)), "out clear");
    reset_compress_state(kind, ratio, dstate_kv.p, dstate_score.p, nullptr);
    for (int start = 0; start < total_tokens; start += chunk) {
        const int count = std::min(chunk, total_tokens - start);
        compress_projected(kind, ratio, count, dkv.p + (size_t) start * width, dscore.p + (size_t) start * width,
                           dape.p, dpos.p + start, dnorm.p, eps, dfreq.p, dstate_kv.p, dstate_score.p, dout.p,
                           drow.p + start, CacheFormat::F32, nullptr);
    }
    got_kv = dstate_kv.down();
    got_score = dstate_score.down();
    got_out = dout.down();
    got_rows = drow.down();
    const auto out_err = ds4test::compare(got_out, ref_out);
    const auto kv_err = ds4test::compare(got_kv, ref_kv);
    const double score_err = max_score_error(got_score, ref_score);
    const bool rows_ok = got_rows == ref_rows;
    std::printf("%s ratio=%d T=%d chunks=%d: output max abs %.3g rel RMS %.3g; state kv %.3g, score %.3g; row ids %s\n",
                kind == CompressKind::Csa ? "CSA" : "HCA", ratio, total_tokens, chunk, out_err.max_abs, out_err.rms_rel,
                kv_err.max_abs, score_err, rows_ok ? "match" : "MISMATCH");
    return out_err.max_abs < 3e-5 && kv_err.max_abs == 0.0 && score_err == 0.0 && rows_ok;
}

bool run_project_and_compress_case() {
    constexpr int T = 8, K = 4096, ratio = 4, width = 2 * D;
    auto x = ds4test::randn((size_t) T * K, 0x70726f6au, 0.25f);
    auto kv_weights_f = ds4test::randn((size_t) width * K, 0x6b767761u, 0.03f);
    auto score_weights_f = ds4test::randn((size_t) width * K, 0x73637761u, 0.03f);
    std::vector<uint16_t> kv_weights(kv_weights_f.size()), score_weights(score_weights_f.size());
    for (size_t i = 0; i < kv_weights.size(); ++i) {
        kv_weights[i] = bf16_bits(kv_weights_f[i]);
        score_weights[i] = bf16_bits(score_weights_f[i]);
        kv_weights_f[i] = bf16_float(kv_weights[i]);
        score_weights_f[i] = bf16_float(score_weights[i]);
    }
    std::vector<float> ape = ds4test::randn((size_t) ratio * width, 0x617065u, 0.2f);
    std::vector<float> norm = ds4test::randn(D, 0x6e6f726du, 0.05f, 1.f);
    const auto freq = ds4test::yarn_frequencies(R, 160000.f, 16.f, 65536, 32.f, 1.f);
    std::vector<int32_t> pos(T);
    for (int i = 0; i < T; ++i) pos[i] = i;

    std::vector<float> kv_ref((size_t) T * width), score_ref((size_t) T * width);
    for (int t = 0; t < T; ++t)
        for (int n = 0; n < width; ++n) {
            double kv_sum = 0.0, score_sum = 0.0;
            for (int k = 0; k < K; ++k) {
                const double activation = x[(size_t) t * K + k];
                kv_sum += activation * kv_weights_f[(size_t) n * K + k];
                score_sum += activation * score_weights_f[(size_t) n * K + k];
            }
            kv_ref[(size_t) t * width + n] = (float) kv_sum;
            score_ref[(size_t) t * width + n] = (float) score_sum;
        }

    const int state_rows = 2 * ratio;
    const size_t state_count = (size_t) state_rows * width;
    std::vector<float> ref_state_kv(state_count, 0.f), ref_state_score(state_count, -INFINITY);
    std::vector<float> ref_out((size_t) T / ratio * D);
    std::vector<int32_t> ref_rows(T, -1);
    reference(CompressKind::Csa, ratio, T, kv_ref, score_ref, ape, pos, norm, freq, ref_state_kv, ref_state_score,
              ref_out, ref_rows);

    Dev<float> dx(x), dape(ape), dnorm(norm), dfreq(freq);
    Dev<uint16_t> dkv_weight(kv_weights), dscore_weight(score_weights);
    Dev<int32_t> dpos(pos), drow(T);
    Dev<float> dkv_projected((size_t) T * width), dscore_projected((size_t) T * width);
    Dev<float> dstate_kv(state_count), dstate_score(state_count), dout(ref_out.size());
    cudaStream_t stream{};
    ds4test::check(cudaStreamCreate(&stream), "projection test stream");
    reset_compress_state(CompressKind::Csa, ratio, dstate_kv.p, dstate_score.p, stream);
    project_and_compress(CompressKind::Csa, ratio, K, T, dx.p, dkv_weight.p, dscore_weight.p, dape.p, dpos.p,
                         dnorm.p, eps, dfreq.p, dkv_projected.p, dscore_projected.p, dstate_kv.p, dstate_score.p,
                         dout.p, drow.p, CacheFormat::F32, stream);
    ds4test::check(cudaStreamSynchronize(stream), "project and compress sync");
    cudaStreamDestroy(stream);

    const auto kv_err = ds4test::compare(dkv_projected.down(), kv_ref);
    const auto score_err = ds4test::compare(dscore_projected.down(), score_ref);
    const auto out_err = ds4test::compare(dout.down(), ref_out);
    const auto state_kv_err = ds4test::compare(dstate_kv.down(), ref_state_kv);
    const auto state_score_err = max_score_error(dstate_score.down(), ref_state_score);
    const bool rows_ok = drow.down() == ref_rows;
    std::printf("BF16 projections -> CSA cache T=%d: kv rel RMS %.3g, score rel RMS %.3g, output max abs %.3g rel RMS %.3g, state kv %.3g, state score %.3g, rows %s\n",
                T, kv_err.rms_rel, score_err.rms_rel, out_err.max_abs, out_err.rms_rel, state_kv_err.max_abs,
                state_score_err, rows_ok ? "match" : "MISMATCH");
    return kv_err.rms_rel < 3e-4 && score_err.rms_rel < 3e-4 && out_err.rms_rel < 5e-4 &&
           state_kv_err.max_abs < 5e-4 && state_score_err < 5e-4 && rows_ok;
}

bool run_hca_project_and_compress_case() {
    constexpr int T = 8, K = 4096, ratio = 128, width = D, start = 120;
    auto x = ds4test::randn((size_t) T * K, 0x68636178u, 0.25f);
    auto kv_weights_f = ds4test::randn((size_t) width * K, 0x6863616bu, 0.03f);
    auto score_weights_f = ds4test::randn((size_t) width * K, 0x68636173u, 0.03f);
    std::vector<uint16_t> kv_weights(kv_weights_f.size()), score_weights(score_weights_f.size());
    for (size_t i = 0; i < kv_weights.size(); ++i) {
        kv_weights[i] = bf16_bits(kv_weights_f[i]);
        score_weights[i] = bf16_bits(score_weights_f[i]);
        kv_weights_f[i] = bf16_float(kv_weights[i]);
        score_weights_f[i] = bf16_float(score_weights[i]);
    }
    const auto ape = ds4test::randn((size_t) ratio * width, 0x68636161u, 0.2f);
    const auto norm = ds4test::randn(D, 0x6863616eu, 0.05f, 1.f);
    const auto freq = ds4test::yarn_frequencies(R, 160000.f, 16.f, 65536, 32.f, 1.f);
    std::vector<int32_t> pos(T);
    for (int i = 0; i < T; ++i) pos[i] = start + i;

    std::vector<float> kv_ref((size_t) T * width), score_ref((size_t) T * width);
    for (int t = 0; t < T; ++t)
        for (int n = 0; n < width; ++n) {
            double kv_sum = 0.0, score_sum = 0.0;
            for (int k = 0; k < K; ++k) {
                const double activation = x[(size_t) t * K + k];
                kv_sum += activation * kv_weights_f[(size_t) n * K + k];
                score_sum += activation * score_weights_f[(size_t) n * K + k];
            }
            kv_ref[(size_t) t * width + n] = (float) kv_sum;
            score_ref[(size_t) t * width + n] = (float) score_sum;
        }

    const size_t state_count = (size_t) ratio * width;
    auto prefix_kv = ds4test::randn((size_t) start * width, 0x70726b76u, 0.25f);
    auto prefix_score = ds4test::randn((size_t) start * width, 0x70727363u, 0.7f);
    std::vector<float> ref_state_kv(state_count, 0.f), ref_state_score(state_count, -INFINITY);
    for (int p = 0; p < start; ++p)
        for (int c = 0; c < width; ++c) {
            const size_t dst = (size_t) p * width + c;
            const size_t src = (size_t) p * width + c;
            ref_state_kv[dst] = prefix_kv[src];
            ref_state_score[dst] = prefix_score[src] + ape[(size_t) p * width + c];
        }
    std::vector<float> ref_out(D);
    std::vector<int32_t> ref_rows(T, -1);
    reference(CompressKind::Hca, ratio, T, kv_ref, score_ref, ape, pos, norm, freq, ref_state_kv, ref_state_score,
              ref_out, ref_rows);

    Dev<float> dx(x), dape(ape), dnorm(norm), dfreq(freq), dstate_kv(ref_state_kv), dstate_score(ref_state_score);
    Dev<uint16_t> dkv_weight(kv_weights), dscore_weight(score_weights);
    Dev<int32_t> dpos(pos), drow(T);
    Dev<float> dkv_projected((size_t) T * width), dscore_projected((size_t) T * width), dout(D);
    cudaStream_t stream{};
    ds4test::check(cudaStreamCreate(&stream), "HCA projection test stream");
    project_and_compress(CompressKind::Hca, ratio, K, T, dx.p, dkv_weight.p, dscore_weight.p, dape.p, dpos.p,
                         dnorm.p, eps, dfreq.p, dkv_projected.p, dscore_projected.p, dstate_kv.p, dstate_score.p,
                         dout.p, drow.p, CacheFormat::F32, stream);
    ds4test::check(cudaStreamSynchronize(stream), "HCA project and compress sync");
    cudaStreamDestroy(stream);

    const auto kv_err = ds4test::compare(dkv_projected.down(), kv_ref);
    const auto score_err = ds4test::compare(dscore_projected.down(), score_ref);
    const auto out_err = ds4test::compare(dout.down(), ref_out);
    const auto state_kv_err = ds4test::compare(dstate_kv.down(), ref_state_kv);
    const auto state_score_err = max_score_error(dstate_score.down(), ref_state_score);
    const bool rows_ok = drow.down() == ref_rows;
    std::printf("BF16 projections -> HCA cache at token %d: kv rel RMS %.3g, score rel RMS %.3g, output max abs %.3g rel RMS %.3g, state kv %.3g, state score %.3g, rows %s\n",
                start, kv_err.rms_rel, score_err.rms_rel, out_err.max_abs, out_err.rms_rel, state_kv_err.max_abs,
                state_score_err, rows_ok ? "match" : "MISMATCH");
    return kv_err.rms_rel < 3e-4 && score_err.rms_rel < 3e-4 && out_err.rms_rel < 5e-4 &&
           state_kv_err.max_abs < 5e-4 && state_score_err < 5e-4 && rows_ok;
}

}  // namespace

int main() {
    int failures = 0;
    failures += !run_case(CompressKind::Csa, 4, 8, 8, 0x4c5341u);
    failures += !run_case(CompressKind::Csa, 4, 8, 4, 0x4c5341u);
    failures += !run_case(CompressKind::Hca, 128, 128, 8, 0x484341u);
    failures += !run_project_and_compress_case();
    failures += !run_hca_project_and_compress_case();

    // One HCA group over a complete ratio-128 window, the larger real geometry.
    const int width = D;
    auto kv = ds4test::randn(8 * width, 0x616263u, 0.25f);
    auto score = ds4test::randn(8 * width, 0x616264u, 0.7f);
    auto ape = ds4test::randn(128 * width, 0x616265u, 0.2f);
    std::vector<int32_t> pos(8);
    for (int i = 0; i < 8; ++i) pos[i] = 120 + i;
    std::vector<float> norm = ds4test::randn(D, 0x616266u, 0.05f, 1.0f);
    const std::vector<float> freq = ds4test::yarn_frequencies(R, 160000.f, 16.f, 65536, 32.f, 1.f);
    Dev<float> dkv(kv), dscore(score), dape(ape), dnorm(norm), dfreq(freq);
    Dev<int32_t> dpos(pos), drow(8);
    Dev<float> dstate_kv(128 * width), dstate_score(128 * width), dout(D);
    reset_compress_state(CompressKind::Hca, 128, dstate_kv.p, dstate_score.p, nullptr);
    std::vector<float> prefix_kv = ds4test::randn(120 * width, 0x707265u, 0.25f);
    std::vector<float> prefix_score = ds4test::randn(120 * width, 0x707273u, 0.7f);
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
                           dstate_kv.p, dstate_score.p, dout.p, drow.p, CacheFormat::F32, nullptr);
    compress_projected(CompressKind::Hca, 128, 8, dkv.p, dscore.p, dape.p, dpos.p, dnorm.p, eps, dfreq.p,
                       dstate_kv.p, dstate_score.p, dout.p, drow.p, CacheFormat::F32, nullptr);
    const auto hca_out = ds4test::compare(dout.down(), ref_out);
    const auto hca_kv = ds4test::compare(dstate_kv.down(), ref_kv);
    const double hca_score = max_score_error(dstate_score.down(), ref_score);
    const auto hca_rows = drow.down();
    const bool rows_match = hca_rows == std::vector<int32_t>(ref_rows.end() - 8, ref_rows.end());
    std::printf("HCA continuation at token 120: output max abs %.3g rel RMS %.3g; state kv %.3g, score %.3g; row ids %s\n",
                hca_out.max_abs, hca_out.rms_rel, hca_kv.max_abs, hca_score, rows_match ? "match" : "MISMATCH");
    failures += !(hca_out.max_abs < 3e-5 && hca_kv.max_abs == 0.0 && hca_score == 0.0 && rows_match);
    const double us = ds4test::bench_stream_us([&](void* stream) {
        compress_projected(CompressKind::Hca, 128, 8, dkv.p, dscore.p, dape.p, dpos.p, dnorm.p, eps, dfreq.p,
                           dstate_kv.p, dstate_score.p, dout.p, drow.p, CacheFormat::F32, stream);
    }, 100);
    const double bytes = (128.0 * D * 2 + 8.0 * D * 3 + D) * sizeof(float);
    ds4test::bench_line("DS4 HCA compress T=8 ratio=128", us, bytes);

    return failures ? 1 : 0;
}
