#include "basalt/ggml/ggml.h"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/nvfp4_avx2.hpp"
#include "basalt/kernels/cpu/nvfp4_avx512.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace cpu = basalt::kernels::cpu;

namespace {
enum class Mode { Avx2Base, Avx2Vnni, Avx2Disp, Avx512 };

const char* mode_name(Mode m) {
    switch (m) {
        case Mode::Avx2Base: return "AVX2-base";
        case Mode::Avx2Vnni: return "AVX2-vnni8";
        case Mode::Avx2Disp: return "AVX2";
        case Mode::Avx512: return "AVX512";
    }
    return "";
}

void run_mode(Mode m, bool gate, const uint8_t* w, size_t stride, size_t up, int n, const void* const* act, int nt,
              float* const* out, int r0, int r1) {
    if (gate) {
        switch (m) {
            case Mode::Avx2Base: cpu::nvfp4_256_gu_rows_base(w, stride, up, n, .1f, .1f, act, nt, out, r0, r1); break;
            case Mode::Avx2Vnni: cpu::nvfp4_256_gu_rows_vnni(w, stride, up, n, .1f, .1f, act, nt, out, r0, r1); break;
            case Mode::Avx2Disp: cpu::nvfp4_256_gu_rows(w, stride, up, n, .1f, .1f, act, nt, out, r0, r1); break;
            case Mode::Avx512: cpu::nvfp4_512_gu_rows(w, stride, up, n, .1f, .1f, act, nt, out, r0, r1); break;
        }
    } else {
        switch (m) {
            case Mode::Avx2Base: cpu::nvfp4_256_down_rows_base(w, stride, n, .1f, act, nt, out, r0, r1); break;
            case Mode::Avx2Vnni: cpu::nvfp4_256_down_rows_vnni(w, stride, n, .1f, act, nt, out, r0, r1); break;
            case Mode::Avx2Disp: cpu::nvfp4_256_down_rows(w, stride, n, .1f, act, nt, out, r0, r1); break;
            case Mode::Avx512: cpu::nvfp4_512_down_rows(w, stride, n, .1f, act, nt, out, r0, r1); break;
        }
    }
}

int check(int n, int rows, bool gate, bool edge_scales, Mode mode) {
    constexpr int T = 8;
    const bool wide = mode == Mode::Avx512;
    const size_t stride = ggml_row_size(GGML_TYPE_NVFP4, n), up = stride * rows;
    std::vector<uint8_t> weights(up * (gate ? 2 : 1));
    std::mt19937 rng(42 + n + rows);
    for (size_t i = 0; i < weights.size(); i += 36) {
        for (int k = 0; k < 4; ++k) weights[i + k] = edge_scales ? (uint8_t) ((i / 36 * 4 + k) & 0xff)
                                             : (uint8_t) (0x20 + rng() % 24);
        for (int k = 4; k < 36; ++k) weights[i + k] = (uint8_t) rng();
    }
    const auto traits = wide ? ggml_get_type_traits_cpu_avx512 : ggml_get_type_traits_cpu_avx2;
    const auto* q = traits(GGML_TYPE_Q8_0);
    const auto* dot = traits(GGML_TYPE_NVFP4);
    const auto* dw = ggml_get_type_traits(GGML_TYPE_NVFP4);
    const auto* dx = ggml_get_type_traits(GGML_TYPE_Q8_0);
    std::vector<std::vector<uint8_t>> packed(T, std::vector<uint8_t>(ggml_row_size(GGML_TYPE_Q8_0, n)));
    std::vector<std::vector<float>> x(T, std::vector<float>(n));
    std::vector<std::vector<float>> ref(T, std::vector<float>(rows)), got(T, std::vector<float>(rows));
    std::vector<std::vector<double>> expected(T, std::vector<double>(rows));
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
            expected[t][r] = gate ? da / (1 + std::exp(-da)) * db : da;
        }
    }
    for (int nt = 1; nt <= T; ++nt) for (int offset = 0; offset + nt <= T; ++offset) {
        auto run = [&](int r0, int r1) {
            run_mode(mode, gate, weights.data(), stride, up, n, act + offset, nt, out, r0, r1);
        };
        run(0, rows);
        for (int t = 0; t < nt; ++t) {
            if (std::memcmp(out[t], ref[t + offset].data(), rows * sizeof(float))) ++bad;
            for (int r = 0; r < rows; ++r) {
                const double want = expected[t + offset][r];
                const double err = std::abs(out[t][r] - want);
                max_abs = std::max(max_abs, err);
                max_rel = std::max(max_rel, err / std::max(1e-8, std::abs(want)));
                if (!std::isfinite(out[t][r]) || err > 1e-5 + 5e-5 * std::abs(want)) ++bad;
            }
        }
        for (int t = 0; t < T; ++t) std::fill(got[t].begin(), got[t].end(), -12345.f);
        int r0 = 0;
        for (int width : {1, 2, 7, 13, rows}) {
            const int r1 = std::min(rows, r0 + width);
            run(r0, r1);
            for (int t = 0; t < nt; ++t) {
                if (std::memcmp(out[t], ref[t + offset].data(), r1 * sizeof(float))) ++bad;
                for (int r = r1; r < rows; ++r) if (out[t][r] != -12345.f) ++bad;
            }
            for (int t = nt; t < T; ++t)
                for (float v : got[t]) if (v != -12345.f) ++bad;
            r0 = r1;
        }
    }
    std::printf("NVFP4 %s %s n=%d rows=%d: double reference max_abs=%.3g max_rel=%.3g; "
                "ggml identity, widths, positions and tails: %s\n", mode_name(mode),
                gate ? "gate/up" : "down", n, rows,
                max_abs, max_rel, bad ? "FAIL" : "PASS");
    return bad;
}

// Zero inputs and quantizer-produced activation extremes (-127..+127) against extreme NVFP4 nibbles (0x7 = +6, 0xF = -6).
int check_extremes(Mode mode) {
    constexpr int n = 128, rows = 9, T = 8;
    const bool wide = mode == Mode::Avx512;
    const auto traits = wide ? ggml_get_type_traits_cpu_avx512 : ggml_get_type_traits_cpu_avx2;
    const auto* q = traits(GGML_TYPE_Q8_0);
    const auto* dot = traits(GGML_TYPE_NVFP4);
    const size_t stride = ggml_row_size(GGML_TYPE_NVFP4, n), up = stride * rows;
    std::vector<uint8_t> weights(up * 2);
    std::mt19937 rng(999);
    for (size_t i = 0; i < weights.size(); i += 36) {
        for (int k = 0; k < 4; ++k) weights[i + k] = (uint8_t) (0x28 + (k & 7));
        const int pat = (int) ((i / 36) % 4);
        for (int k = 4; k < 36; ++k)
            weights[i + k] = pat == 0 ? 0x77 : pat == 1 ? 0xFF : pat == 2 ? 0x7F : (uint8_t) rng();
    }
    const size_t act_row = ggml_row_size(GGML_TYPE_Q8_0, n);
    std::vector<std::vector<uint8_t>> packed(T, std::vector<uint8_t>(act_row, 0));
    std::vector<float> x(n);
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < n; ++i) {
            if (t == 0) x[i] = 0.f;
            else if (t == 1) x[i] = 1.f;
            else if (t == 2) x[i] = -1.f;
            else if (t == 3) x[i] = (i & 1) ? -1.f : 1.f;
            else if (t == 4) x[i] = (i % 32 == 0) ? -4.f : 0.f;
            else if (t == 5) x[i] = (i % 32 == 0) ? 4.f : -4.f;
            else if (t == 6) x[i] = (rng() & 1) ? 2.5f : -2.5f;
            else x[i] = ((float) (int) (rng() % 255) - 127.f) / 127.f;
        }
        q->from_float(x.data(), packed[t].data(), n);
    }
    int bad = 0;
    for (bool gate : {true, false}) {
        std::vector<std::vector<float>> ref(T, std::vector<float>(rows)), got(T, std::vector<float>(rows));
        const void* act[T];
        float* out[T];
        for (int t = 0; t < T; ++t) {
            act[t] = packed[t].data();
            out[t] = got[t].data();
            for (int r = 0; r < rows; ++r) {
                float a = 0, b = 0;
                dot->vec_dot(n, &a, 0, weights.data() + stride * r, 0, act[t], 0, 1);
                if (gate) dot->vec_dot(n, &b, 0, weights.data() + up + stride * r, 0, act[t], 0, 1);
                a *= .1f;
                b *= .1f;
                ref[t][r] = gate ? (a / (1.f + std::exp(-a))) * b : a;
            }
        }
        for (int nt = 1; nt <= T; ++nt) {
            run_mode(mode, gate, weights.data(), stride, up, n, act, nt, out, 0, rows);
            for (int t = 0; t < nt; ++t)
                if (std::memcmp(out[t], ref[t].data(), rows * sizeof(float))) ++bad;
        }
    }
    return bad;
}
}  // namespace

int main() {
    if (!cpu::cpu_avx2_ok()) return 77;
    std::vector<Mode> modes = {Mode::Avx2Base, Mode::Avx2Disp};
    if (cpu::cpu_avxvnni_int8_ok()) modes.push_back(Mode::Avx2Vnni);
    if (cpu::cpu_avx512_ok()) modes.push_back(Mode::Avx512);
    int bad = 0;
    for (Mode m : modes) {
        bad += check(64, 7, true, false, m);
        bad += check(64, 7, false, false, m);
        bad += check(64, 65, true, true, m);
        bad += check(64, 65, false, true, m);
        bad += check(2560, 640, true, false, m);
        bad += check(640, 2560, false, false, m);
        bad += check_extremes(m);
    }
    return bad ? 1 : 0;
}
