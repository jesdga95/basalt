#include "basalt/ggml/ggml.h"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/nvfp4_avx512.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace cpu = basalt::kernels::cpu;

namespace {
int check(int n, int rows, bool gate, bool edge_scales = false) {
    constexpr int T = 8;
    const size_t stride = ggml_row_size(GGML_TYPE_NVFP4, n), up = stride * rows;
    std::vector<uint8_t> weights(up * (gate ? 2 : 1));
    std::mt19937 rng(42 + n + rows);
    for (size_t i = 0; i < weights.size(); i += 36) {
        for (int k = 0; k < 4; ++k) weights[i + k] = edge_scales ? (uint8_t) ((i / 36 * 4 + k) & 0xff)
                                             : (uint8_t) (0x20 + rng() % 24);
        for (int k = 4; k < 36; ++k) weights[i + k] = (uint8_t) rng();
    }
    const auto* q = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    const auto* dot = ggml_get_type_traits_cpu(GGML_TYPE_NVFP4);
    const auto* dw = ggml_get_type_traits(GGML_TYPE_NVFP4);
    const auto* dx = ggml_get_type_traits(GGML_TYPE_Q8_0);
    std::vector<std::vector<uint8_t>> packed(T, std::vector<uint8_t>(ggml_row_size(GGML_TYPE_Q8_0, n)));
    std::vector<std::vector<float>> x(T, std::vector<float>(n));
    std::vector<std::vector<float>> ref(T, std::vector<float>(rows)), got(T, std::vector<float>(rows));
    const void* act[T];
    float* out[T];
    std::normal_distribution<float> nd(0, .25f);
    for (int t = 0; t < T; ++t) {
        for (float& v : x[t]) v = nd(rng);
        q->from_float(x[t].data(), packed[t].data(), n);
        dx->to_float(packed[t].data(), x[t].data(), n);
        act[t] = packed[t].data();
        out[t] = got[t].data();
    }
    int bad = 0;
    double max_abs = 0, max_rel = 0;
    std::vector<float> w(n), u(n);
    for (int r = 0; r < rows; ++r) {
        dw->to_float(weights.data() + stride * r, w.data(), n);
        if (gate) dw->to_float(weights.data() + up + stride * r, u.data(), n);
        for (int t = 0; t < T; ++t) {
            float a = 0, b = 0;
            dot->vec_dot(n, &a, 0, weights.data() + stride * r, 0, act[t], 0, 1);
            if (gate) dot->vec_dot(n, &b, 0, weights.data() + up + stride * r, 0, act[t], 0, 1);
            a *= .1f;
            b *= .1f;
            ref[t][r] = gate ? (a / (1.f + std::exp(-a))) * b : a;
            double da = 0, db = 0;
            for (int i = 0; i < n; ++i) {
                da += (double) w[i] * x[t][i];
                if (gate) db += (double) u[i] * x[t][i];
            }
            da *= (double) .1f;
            db *= (double) .1f;
            const double expected = gate ? da / (1 + std::exp(-da)) * db : da;
            const double err = std::abs(ref[t][r] - expected);
            max_abs = std::max(max_abs, err);
            max_rel = std::max(max_rel, err / std::max(1e-8, std::abs(expected)));
            if (!std::isfinite(ref[t][r]) || err > 1e-5 + 5e-5 * std::abs(expected)) ++bad;
        }
    }
    for (int nt = 1; nt <= T; ++nt) {
        auto run = [&](int r0, int r1) {
            if (gate) cpu::nvfp4_512_gu_rows(weights.data(), stride, up, n, .1f, .1f, act, nt, out, r0, r1);
            else cpu::nvfp4_512_down_rows(weights.data(), stride, n, .1f, act, nt, out, r0, r1);
        };
        run(0, rows);
        for (int t = 0; t < nt; ++t)
            if (std::memcmp(out[t], ref[t].data(), rows * sizeof(float))) ++bad;
        for (int t = 0; t < T; ++t) std::fill(got[t].begin(), got[t].end(), -12345.f);
        int r0 = 0;
        for (int width : {1, 2, 7, 13, rows}) {
            const int r1 = std::min(rows, r0 + width);
            run(r0, r1);
            for (int t = 0; t < nt; ++t) {
                if (std::memcmp(out[t], ref[t].data(), r1 * sizeof(float))) ++bad;
                for (int r = r1; r < rows; ++r) if (out[t][r] != -12345.f) ++bad;
            }
            for (int t = nt; t < T; ++t)
                for (float v : got[t]) if (v != -12345.f) ++bad;
            r0 = r1;
        }
    }
    std::printf("NVFP4 %s n=%d rows=%d: double reference max_abs=%.3g max_rel=%.3g; "
                "ggml identity, widths and tails: %s\n", gate ? "gate/up" : "down", n, rows,
                max_abs, max_rel, bad ? "FAIL" : "PASS");
    return bad;
}
}  // namespace

int main() {
    if (!cpu::cpu_avx512_ok()) return 77;
    int bad = 0;
    bad += check(64, 7, true);
    bad += check(64, 7, false);
    bad += check(64, 65, true, true);
    bad += check(64, 65, false, true);
    bad += check(2560, 640, true);
    bad += check(640, 2560, false);
    return bad ? 1 : 0;
}
