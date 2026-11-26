#include "basalt/models/deepseek4/indexer.hpp"
#include "parity_util.hpp"

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

void reference(const std::vector<uint16_t>& wq, const std::vector<uint16_t>& wp, const std::vector<float>& qr,
               const std::vector<float>& x, const std::vector<float>& angle, std::vector<float>& q_out,
               std::vector<float>& head_weights) {
    std::vector<float> q_raw((size_t) T * H * D), w_raw((size_t) T * H);
    for (int t = 0; t < T; ++t) {
        for (int o = 0; o < H * D; ++o) {
            double sum = 0.0;
            for (int k = 0; k < QR; ++k)
                sum += (double) from_bf16(wq[(size_t) o * QR + k]) * bf16_rn(qr[(size_t) t * QR + k]);
            q_raw[(size_t) t * H * D + o] = bf16_rn((float) sum);
        }
        for (int o = 0; o < H; ++o) {
            double sum = 0.0;
            for (int k = 0; k < E; ++k)
                sum += (double) from_bf16(wp[(size_t) o * E + k]) * bf16_rn(x[(size_t) t * E + k]);
            w_raw[(size_t) t * H + o] = bf16_rn((float) sum);
        }
    }

    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < H; ++h) {
            float* q = q_raw.data() + ((size_t) t * H + h) * D;
            for (int c = D - R; c < D; c += 2) {
                const int pair = (c - (D - R)) / 2;
                const float theta = angle[(size_t) t * (R / 2) + pair];
                const float sine = std::sin(theta), cosine = std::cos(theta);
                const float x0 = q[c], x1 = q[c + 1];
                q[c] = bf16_rn(x0 * cosine - x1 * sine);
                q[c + 1] = bf16_rn(x0 * sine + x1 * cosine);
            }
            for (int stride = 1; stride < D; stride <<= 1) {
                std::vector<float> next(D);
                for (int c = 0; c < D; ++c)
                    next[c] = bf16_rn((c & stride) ? q[c ^ stride] - q[c] : q[c] + q[c ^ stride]);
                std::copy(next.begin(), next.end(), q);
            }
            for (int c = 0; c < D; ++c) q[c] = bf16_rn(q[c] * (1.f / std::sqrt((float) D)));
            for (int group = 0; group < D / 32; ++group) {
                float amax = 6.f * 0x1p-126f;
                for (int c = group * 32; c < (group + 1) * 32; ++c) amax = std::max(amax, std::fabs(q[c]));
                const float scale = pow2_ceil(amax * (1.f / 6.f));
                for (int c = group * 32; c < (group + 1) * 32; ++c)
                    q_out[((size_t) t * H + h) * D + c] = bf16_rn(fp4_rn(std::clamp(q[c] / scale, -6.f, 6.f)) * scale);
            }
        }
        const float scale = (1.f / std::sqrt((float) D)) * (1.f / std::sqrt((float) H));
        for (int h = 0; h < H; ++h) head_weights[(size_t) t * H + h] = bf16_rn(w_raw[(size_t) t * H + h] * scale);
    }
}

std::vector<uint16_t> bf16_weights(size_t n, uint32_t seed, float sigma) {
    const auto values = ds4test::randn(n, seed, sigma);
    std::vector<uint16_t> result(n);
    for (size_t i = 0; i < n; ++i) result[i] = to_bf16(values[i]);
    return result;
}

bool run() {
    const auto wq = bf16_weights((size_t) H * D * QR, 0x505251u, .03f);
    const auto wp = bf16_weights((size_t) H * E, 0x505257u, .02f);
    const auto qr = ds4test::randn((size_t) T * QR, 0x505251u, .2f);
    const auto x = ds4test::randn((size_t) T * E, 0x505258u, .2f);
    const std::vector<float> freq = ds4test::yarn_frequencies(R, 160000.f, 16.f, 65536, 32.f, 1.f);
    std::vector<float> angle((size_t) T * R / 2);
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < R / 2; ++i)
            angle[(size_t) t * (R / 2) + i] = (2800.f + t) * freq[(size_t) i];
    std::vector<float> expected_q((size_t) T * H * D), expected_w((size_t) T * H);
    reference(wq, wp, qr, x, angle, expected_q, expected_w);

    Dev<uint16_t> dwq(wq), dwp(wp);
    Dev<float> dqr(qr), dx(x), dangle(angle), got_q(expected_q.size()), got_w(expected_w.size());
    const size_t ws_bytes = indexer_project_workspace_bytes(T);
    Dev<uint8_t> workspace(ws_bytes);
    indexer_project_bf16(dwq.p, dwp.p, dqr.p, dx.p, dangle.p, T, got_q.p, got_w.p, workspace.p, ws_bytes, nullptr);
    const auto q_error = ds4test::compare(got_q.down(), expected_q);
    const auto w_error = ds4test::compare(got_w.down(), expected_w);
    std::printf("DS4 indexer projections: q max abs %.3g rel RMS %.3g; head weights max abs %.3g rel RMS %.3g\n",
                q_error.max_abs, q_error.rms_rel, w_error.max_abs, w_error.rms_rel);

    const double us = ds4test::bench_stream_us([&](void* stream) {
        indexer_project_bf16(dwq.p, dwp.p, dqr.p, dx.p, dangle.p, T, got_q.p, got_w.p, workspace.p, ws_bytes, stream);
    }, 25);
    const double bytes = (double) H * D * (QR * 2 + 4) + (double) H * (E * 2 + 4) +
                         (double) T * (QR + E) * sizeof(float);
    ds4test::bench_line("DS4 indexer BF16 projections", us, bytes);
    return q_error.max_abs < 1e-6 && w_error.max_abs < 1e-6;
}

}  // namespace

int main() { return run() ? 0 : 1; }
