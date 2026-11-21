#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/ggml/ggml.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace basalt::kernels::cpu;

int main() {
    std::mt19937 rng(29);
    std::normal_distribution<float> normal(0.f, 0.5f);
    int failures = 0;
    for (int type : {8, 40}) {
        for (int width : {64, 2304}) {
            const int hidden = width == 64 ? 64 : 1152;
            NativeFmt f;
            std::string error;
            if (!native_fmt(type, type, width, hidden, f, error)) return 2;
            f.gate_softcap = 7.f;
            f.up_softcap = 4.f;
            std::vector<uint8_t> blob(f.bytes);
            std::vector<float> raw((size_t) width * hidden);
            const auto* weights = ggml_get_type_traits((ggml_type) type);
            for (size_t offset : {size_t(0), f.up_off}) {
                for (float& v : raw) v = normal(rng);
                weights->from_float_ref(raw.data(), blob.data() + offset, raw.size());
            }
            if (f.tail) {
                const float scales[4] = {1.f, 1.f, 1.f, 0.f};
                std::memcpy(blob.data() + f.bytes - f.tail, scales, sizeof(scales));
            }
            std::vector<std::vector<uint8_t>> quant(8, std::vector<uint8_t>(f.act_bytes));
            std::vector<std::vector<float>> input(8, std::vector<float>(width));
            std::vector<std::vector<float>> single(8, std::vector<float>(hidden));
            std::vector<float> gw(width), uw(width);
            const void* act[8];
            float* output[8];
            double max_abs = 0, max_rel = 0;
            for (int t = 0; t < 8; ++t) {
                for (float& v : input[t]) v = normal(rng);
                native_quant_act(f, input[t].data(), quant[t].data());
                ggml_get_type_traits((ggml_type) f.gu_act)->to_float(quant[t].data(), input[t].data(), width);
                act[t] = quant[t].data();
                output[t] = single[t].data();
                native_gu_rows(f, blob.data(), act + t, 1, output + t, 0, hidden);
                for (int r = 0; r < hidden; ++r) {
                    weights->to_float(blob.data() + (size_t) r * f.gu_row, gw.data(), width);
                    weights->to_float(blob.data() + f.up_off + (size_t) r * f.up_row, uw.data(), width);
                    double g = 0, u = 0;
                    for (int c = 0; c < width; ++c) {
                        g += (double) gw[c] * input[t][c];
                        u += (double) uw[c] * input[t][c];
                    }
                    g = f.gate_softcap * std::tanh(g / f.gate_softcap);
                    u = f.up_softcap * std::tanh(u / f.up_softcap);
                    const double ref = g / (1. + std::exp(-g)) * u;
                    const double delta = std::abs(ref - single[t][r]);
                    max_abs = std::max(max_abs, delta);
                    max_rel = std::max(max_rel, delta / std::max(1., std::abs(ref)));
                }
            }
            for (int n : {2, 4, 8}) {
                std::vector<std::vector<float>> grouped(n, std::vector<float>(hidden));
                for (int t = 0; t < n; ++t) output[t] = grouped[t].data();
                native_gu_rows(f, blob.data(), act, n, output, 0, hidden);
                for (int t = 0; t < n; ++t)
                    if (std::memcmp(grouped[t].data(), single[t].data(), hidden * sizeof(float))) ++failures;
            }
            std::printf("softcap type=%d width=%d: max_abs=%g max_rel=%g\n", type, width, max_abs, max_rel);
            // The reference accumulates in double; the vector dot accumulates 2,304 values in float.
            if (max_abs > 3e-4 || max_rel > 5e-5) ++failures;
            for (int t = 0; t < 8; ++t) output[t] = single[t].data();
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < 8; ++i) native_gu_rows(f, blob.data(), act, 4, output, 0, hidden);
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / 8;
            std::printf("  four rows: %.1f us, %.2f GFLOP/s\n", us, 16. * width * hidden / (us * 1000.));
        }
    }
    std::printf("cpu softcap: %d failures\n", failures);
    return failures ? 1 : 0;
}
