#include "basalt/models/deepseek4/indexer.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/f16_bits.hpp"
#include "parity_util.hpp"

#include <cstring>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace basalt::models::deepseek4;
using ds4test::Dev;

namespace {
constexpr int T = 2;
constexpr int D = kIndexerDim;
constexpr int H = kIndexerHeads;
constexpr int R = 64;
constexpr int QR = kIndexerQueryRank;
constexpr int E = kIndexerEmbedding;

float bf16_rn(float value) {
    uint32_t bits = std::bit_cast<uint32_t>(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return std::bit_cast<float>(bits & 0xffff0000u);
}

uint16_t to_bf16(float value) { return (uint16_t) (std::bit_cast<uint32_t>(bf16_rn(value)) >> 16); }
float from_bf16(uint16_t value) { return std::bit_cast<float>((uint32_t) value << 16); }

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

void head_reference(const std::vector<uint16_t>& wp, const std::vector<float>& x, std::vector<float>& head_weights) {
    const float score_scale = (1.f / std::sqrt((float) D)) * (1.f / std::sqrt((float) H));
    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < H; ++h) {
            double sum = 0.0;
            for (int k = 0; k < E; ++k)
                sum += (double) from_bf16(wp[(size_t) h * E + k]) * bf16_rn(x[(size_t) t * E + k]);
            const float projected = bf16_rn((float) sum);
            head_weights[(size_t) t * H + h] = bf16_rn(projected * score_scale);
        }
    }
}

void query_post_reference(const std::vector<float>& raw, const std::vector<float>& angle, std::vector<float>& q_out) {
    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < H; ++h) {
            const size_t base = ((size_t) t * H + h) * D;
            std::vector<float> q(D);
            for (int c = 0; c < D; ++c) q[(size_t) c] = bf16_rn(raw[base + (size_t) c]);
            for (int c = D - R; c < D; c += 2) {
                const int pair = (c - (D - R)) / 2;
                const float theta = angle[(size_t) t * (R / 2) + pair];
                const float sine = std::sin(theta), cosine = std::cos(theta);
                const float x0 = q[(size_t) c], x1 = q[(size_t) c + 1];
                q[(size_t) c] = bf16_rn(x0 * cosine - x1 * sine);
                q[(size_t) c + 1] = bf16_rn(x0 * sine + x1 * cosine);
            }
            for (int stride = 1; stride < D; stride <<= 1) {
                std::vector<float> next(D);
                for (int c = 0; c < D; ++c)
                    next[(size_t) c] = bf16_rn((c & stride) ? q[(size_t) (c ^ stride)] - q[(size_t) c] :
                                                               q[(size_t) c] + q[(size_t) (c ^ stride)]);
                q.swap(next);
            }
            for (int c = 0; c < D; ++c) q[(size_t) c] = bf16_rn(q[(size_t) c] * (1.f / std::sqrt((float) D)));
            for (int group = 0; group < D / 32; ++group) {
                float amax = 6.f * 0x1p-126f;
                for (int c = group * 32; c < (group + 1) * 32; ++c)
                    amax = std::max(amax, std::fabs(q[(size_t) c]));
                const float scale = pow2_ceil(amax * (1.f / 6.f));
                for (int c = group * 32; c < (group + 1) * 32; ++c)
                    q_out[base + (size_t) c] = bf16_rn(fp4_rn(std::clamp(q[(size_t) c] / scale, -6.f, 6.f)) * scale);
            }
        }
    }
}

std::vector<uint8_t> q8_0_weights() {
    const size_t rows = (size_t) H * D;
    const size_t blocks = rows * QR / 32;
    std::vector<uint8_t> result(blocks * 34);
    const uint16_t scale_bits = basalt::kernels::f16_from_f32(0.002f);
    for (size_t b = 0; b < blocks; ++b) {
        std::memcpy(result.data() + b * 34, &scale_bits, sizeof(scale_bits));
        for (int i = 0; i < 32; ++i)
            result[b * 34 + 2 + (size_t) i] = (uint8_t) (int8_t) ((int) ((b * 7 + (size_t) i * 13 + 3) % 31) - 15);
    }
    return result;
}

bool q8_projection_parity(const std::vector<uint16_t>& wp, const std::vector<float>& qr, const std::vector<float>& x,
                          const std::vector<float>& angle, const std::vector<float>& expected_w) {
    const auto weights = q8_0_weights();
    Dev<uint8_t> dwq(weights);
    Dev<uint16_t> dwp(wp);
    Dev<float> dqr(qr), dx(x), dangle(angle), got_q((size_t) T * H * D), got_w((size_t) T * H);
    const size_t ws_bytes = indexer_project_workspace_bytes(T);
    Dev<uint8_t> workspace(ws_bytes);
    cudaStream_t stream = nullptr;
    ds4test::check(cudaStreamCreate(&stream), "create Q8_0 indexer stream");
    indexer_project_q8_0(dwq.p, dwp.p, dqr.p, dx.p, dangle.p, T, got_q.p, got_w.p, workspace.p, ws_bytes, stream);
    ds4test::check(cudaStreamSynchronize(stream), "synchronize Q8_0 indexer projection");

    std::vector<float> raw((size_t) T * H * D);
    const auto workspace_host = workspace.down();
    std::memcpy(raw.data(), workspace_host.data(), raw.size() * sizeof(float));
    std::vector<float> expected_q(raw.size());
    query_post_reference(raw, angle, expected_q);
    const auto got_q_host = got_q.down();
    const auto q_error = ds4test::compare(got_q_host, expected_q);
    const auto w_error = ds4test::compare(got_w.down(), expected_w);

    std::vector<float> qr_bf16(qr.size());
    for (size_t i = 0; i < qr.size(); ++i) qr_bf16[i] = bf16_rn(qr[i]);
    Dev<float> dqr_bf16(qr_bf16), direct_raw(raw.size());
    Dev<uint8_t> q8_1(basalt::kernels::native_q8_1_bytes(QR, T));
    basalt::kernels::native_quantize_q8_1(dqr_bf16.p, q8_1.p, QR, T, stream);
    basalt::kernels::native_q8_0_mmvq(dwq.p, q8_1.p, direct_raw.p, QR, H * D, T, stream);
    ds4test::check(cudaStreamSynchronize(stream), "synchronize direct Q8_0 indexer projection");
    const auto direct = direct_raw.down();
    const bool raw_exact = direct == raw;
    const auto q8_1_host = q8_1.down();
    const auto half_value = [](const uint8_t* p) {
        uint16_t bits = 0;
        std::memcpy(&bits, p, sizeof(bits));
        return (double) basalt::kernels::f32_from_f16(bits);
    };
    std::vector<float> direct_reference(direct.size());
    for (int t = 0; t < T; ++t) {
        for (int o = 0; o < H * D; ++o) {
            double sum = 0.0;
            for (int b = 0; b < QR / 32; ++b) {
                const size_t wb = ((size_t) o * (QR / 32) + b) * 34;
                const size_t xb = ((size_t) t * (QR / 32) + b) * 36;
                const double ws = half_value(weights.data() + wb);
                const double xs = half_value(q8_1_host.data() + xb);
                int dot = 0;
                for (int i = 0; i < 32; ++i)
                    dot += (int8_t) weights[wb + 2 + (size_t) i] * (int8_t) q8_1_host[xb + 4 + (size_t) i];
                sum += ws * xs * dot;
            }
            direct_reference[(size_t) t * H * D + o] = (float) sum;
        }
    }
    const auto raw_error = ds4test::compare(direct, direct_reference);

    Dev<float> one_q((size_t) H * D), one_w(H);
    const size_t one_bytes = indexer_project_workspace_bytes(1);
    Dev<uint8_t> one_workspace(one_bytes);
    bool independent = true;
    for (int t = 0; t < T; ++t) {
        indexer_project_q8_0(dwq.p, dwp.p, dqr.p + (size_t) t * QR, dx.p + (size_t) t * E,
                             dangle.p + (size_t) t * R / 2, 1, one_q.p, one_w.p,
                             one_workspace.p, one_bytes, stream);
        ds4test::check(cudaStreamSynchronize(stream), "synchronize single-row Q8_0 indexer projection");
        const auto row = one_q.down();
        independent &= std::equal(row.begin(), row.end(), got_q_host.begin() + (size_t) t * H * D);
    }

    std::printf("DS4 indexer Q8_0 path T=%d: q-post max abs %.3g rel RMS %.3g; head weights max abs %.3g; raw vs Q8_0/Q8_1 double max abs %.3g rel RMS %.3g; raw MMVQ %s; row independence %s\n",
                T, q_error.max_abs, q_error.rms_rel, w_error.max_abs, raw_error.max_abs, raw_error.rms_rel,
                raw_exact ? "bitwise" : "DIFFERS", independent ? "bitwise" : "DIFFERS");
    const double us = ds4test::bench_stream_us([&](void* bench_stream) {
        indexer_project_q8_0(dwq.p, dwp.p, dqr.p, dx.p, dangle.p, T, got_q.p, got_w.p,
                             workspace.p, ws_bytes, bench_stream);
    }, 25);
    const double bytes = (double) weights.size() + (double) H * E * 2.0 +
                         (double) T * (QR + E) * sizeof(float);
    ds4test::bench_line("DS4 indexer Q8_0 projection T=2", us, bytes);
    cudaStreamDestroy(stream);
    return q_error.max_abs < 1e-5 && w_error.max_abs < 1e-6 && raw_error.max_abs < 1e-5 &&
           raw_error.rms_rel < 1e-5 && raw_exact && independent;
}

std::vector<uint16_t> bf16_weights(size_t n, uint32_t seed, float sigma) {
    const auto values = ds4test::randn(n, seed, sigma);
    std::vector<uint16_t> result(n);
    for (size_t i = 0; i < n; ++i) result[i] = to_bf16(values[i]);
    return result;
}

bool run() {
    const auto wp = bf16_weights((size_t) H * E, 0x505257u, .02f);
    const auto qr = ds4test::randn((size_t) T * QR, 0x505251u, .2f);
    const auto x = ds4test::randn((size_t) T * E, 0x505258u, .2f);
    const std::vector<float> freq = ds4test::yarn_frequencies(R, 160000.f, 16.f, 65536, 32.f, 1.f);
    std::vector<float> angle((size_t) T * R / 2);
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < R / 2; ++i)
            angle[(size_t) t * (R / 2) + i] = (2800.f + t) * freq[(size_t) i];
    std::vector<float> expected_w((size_t) T * H);
    head_reference(wp, x, expected_w);
    return q8_projection_parity(wp, qr, x, angle, expected_w);
}

}  // namespace

int main() { return run() ? 0 : 1; }
