// tests/kernels/cpu/mxfp4_cpu_rows_test.cpp - MXFP4 multi-token CPU expert rows on AVX2, AVX-VNNI and AVX-512
// against ggml's single-token MXFP4 dot (bitwise equal) and a double-precision reference.
//
//     build/mxfp4_cpu_rows_test [--bench]
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/mxfp4_avx2.hpp"
#include "basalt/kernels/cpu/mxfp4_avx512.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/cpu/swiglu.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace cpu = basalt::kernels::cpu;

namespace {

enum class Mode { Avx2Base, Avx2Vnni, Avx2Disp, Avx512 };

const char* mode_name(Mode m) {
    switch (m) {
        case Mode::Avx2Base: return "AVX2-base";
        case Mode::Avx2Vnni: return "AVX2-vnni";
        case Mode::Avx2Disp: return "AVX2";
        case Mode::Avx512: return "AVX512";
    }
    return "";
}

void run_mode(Mode m, bool gate, const uint8_t* w, size_t stride, size_t up, int n, const void* const* act, int nt,
              float* const* out, int r0, int r1, const cpu::SwiGlu& glu = {}) {
    if (gate) {
        switch (m) {
            case Mode::Avx2Base: cpu::mxfp4_256_gu_rows_base(w, stride, up, n, glu, act, nt, out, r0, r1); break;
            case Mode::Avx2Vnni: cpu::mxfp4_256_gu_rows_vnni(w, stride, up, n, glu, act, nt, out, r0, r1); break;
            case Mode::Avx2Disp: cpu::mxfp4_256_gu_rows(w, stride, up, n, glu, act, nt, out, r0, r1); break;
            case Mode::Avx512: cpu::mxfp4_512_gu_rows(w, stride, up, n, glu, act, nt, out, r0, r1); break;
        }
    } else {
        switch (m) {
            case Mode::Avx2Base: cpu::mxfp4_256_down_rows_base(w, stride, n, act, nt, out, r0, r1); break;
            case Mode::Avx2Vnni: cpu::mxfp4_256_down_rows_vnni(w, stride, n, act, nt, out, r0, r1); break;
            case Mode::Avx2Disp: cpu::mxfp4_256_down_rows(w, stride, n, act, nt, out, r0, r1); break;
            case Mode::Avx512: cpu::mxfp4_512_down_rows(w, stride, n, act, nt, out, r0, r1); break;
        }
    }
}

double swiglu_d(const cpu::SwiGlu& glu, double g, double u) {
    if (glu.gelu) return 0.5 * g * (1.0 + std::tanh(0.7978845608028654 * (g + 0.044715 * g * g * g))) * u;
    if (glu.gate_softcap > 0.f) {
        const double cg = (double) glu.gate_softcap, cu = (double) glu.up_softcap;
        const double sg = 1.0 / (1.0 + std::exp(-g));
        return (cg * std::tanh(g / cg)) * sg * (cu * std::tanh(u / cu));
    }
    if (glu.limit > 0.f) {
        g = std::min(g, (double) glu.limit);
        u = std::max(std::min(u, (double) glu.limit), -(double) glu.limit);
    }
    return (g / (1.0 + std::exp(-g))) * u;
}

int check(int n, int rows, bool gate, bool edge_scales, Mode mode, const cpu::SwiGlu& glu = {}) {
    constexpr int T = 8;
    const bool wide = mode == Mode::Avx512;
    const size_t stride = ggml_row_size(GGML_TYPE_MXFP4, n), up = stride * (size_t) rows;
    std::vector<uint8_t> weights(up * (gate ? 2 : 1));
    std::mt19937 rng(91 + n + rows);
    for (size_t i = 0; i < weights.size(); i += 17) {
        // E8M0 scale byte: edge_scales sweeps 0..254 (avoiding 255 / huge exponent overflow in float sums).
        weights[i] = edge_scales ? (uint8_t) ((i / 17) % 140) : (uint8_t) (118 + rng() % 14);
        for (int k = 1; k < 17; ++k) weights[i + k] = (uint8_t) rng();
    }
    const auto traits = wide ? ggml_get_type_traits_cpu_avx512 : ggml_get_type_traits_cpu_avx2;
    const auto* q = traits(GGML_TYPE_Q8_0);
    const auto* dot = traits(GGML_TYPE_MXFP4);
    const auto* dw = ggml_get_type_traits(GGML_TYPE_MXFP4);
    const auto* dx = ggml_get_type_traits(GGML_TYPE_Q8_0);
    std::vector<std::vector<uint8_t>> packed(T, std::vector<uint8_t>(ggml_row_size(GGML_TYPE_Q8_0, n)));
    std::vector<std::vector<float>> x(T, std::vector<float>(n));
    std::vector<std::vector<float>> ref(T, std::vector<float>(rows)), got(T, std::vector<float>(rows));
    std::vector<std::vector<double>> expected(T, std::vector<double>(rows));
    const void* act[T];
    float* out[T];
    std::normal_distribution<float> nd(0.f, 0.25f);
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
        dw->to_float(weights.data() + stride * (size_t) r, w.data(), n);
        if (gate) dw->to_float(weights.data() + up + stride * (size_t) r, u.data(), n);
        for (int t = 0; t < T; ++t) {
            float a = 0, b = 0;
            dot->vec_dot(n, &a, 0, weights.data() + stride * (size_t) r, 0, act[t], 0, 1);
            if (gate) dot->vec_dot(n, &b, 0, weights.data() + up + stride * (size_t) r, 0, act[t], 0, 1);
            ref[t][r] = gate ? cpu::swiglu(glu, a, b) : a;
            double da = 0, db = 0;
            for (int i = 0; i < n; ++i) {
                da += (double) w[i] * (double) x[t][i];
                if (gate) db += (double) u[i] * (double) x[t][i];
            }
            expected[t][r] = gate ? swiglu_d(glu, da, db) : da;
        }
    }
    for (int nt = 1; nt <= T; ++nt) {
        for (int offset = 0; offset + nt <= T; ++offset) {
            auto run = [&](int r0, int r1) {
                run_mode(mode, gate, weights.data(), stride, up, n, act + offset, nt, out, r0, r1, glu);
            };
            run(0, rows);
            for (int t = 0; t < nt; ++t) {
                if (std::memcmp(out[t], ref[t + offset].data(), (size_t) rows * sizeof(float)) != 0) ++bad;
                for (int r = 0; r < rows; ++r) {
                    const double want = expected[t + offset][r];
                    const double err = std::abs((double) out[t][r] - want);
                    max_abs = std::max(max_abs, err);
                    max_rel = std::max(max_rel, err / std::max(1e-8, std::abs(want)));
                    const double tol = 1e-4 + 1e-4 * std::abs(want);
                    if (!std::isfinite(out[t][r]) || err > tol) ++bad;
                }
            }
            for (int t = 0; t < T; ++t) std::fill(got[t].begin(), got[t].end(), -12345.f);
            int r0 = 0;
            for (int width : {1, 2, 7, 13, rows}) {
                const int r1 = std::min(rows, r0 + width);
                run(r0, r1);
                for (int t = 0; t < nt; ++t) {
                    if (std::memcmp(out[t], ref[t + offset].data(), (size_t) r1 * sizeof(float)) != 0) ++bad;
                    for (int r = r1; r < rows; ++r)
                        if (out[t][r] != -12345.f) ++bad;
                }
                r0 = r1;
                if (r0 == rows) break;
            }
        }
    }
    std::printf("  %-9s %-4s n=%4d rows=%3d%s%s: max_abs=%.3e max_rel=%.3e %s\n", mode_name(mode),
                gate ? "gu" : "down", n, rows, edge_scales ? " edge" : "",
                glu.gelu ? " gelu" : glu.gate_softcap > 0.f ? " softcap" : glu.limit > 0.f ? " limited" : "", max_abs,
                max_rel, bad ? "FAIL" : "ok");
    return bad;
}

int check_dispatch() {
    // Check native_gu_rows and native_down_rows dispatch at both ERNIE-5.5 (2048x2048, SiTU-GLU softcap) and
    // DeepSeek-V4.1-Flash (5120x2304, plain SwiGLU) geometries against ggml's single-token dot, bit for bit.
    int bad = 0;
    const auto* tc = ggml_get_type_traits_cpu(GGML_TYPE_MXFP4);
    std::mt19937 rng(777);
    std::normal_distribution<float> nd(0.f, 0.25f);
    for (auto cfg : {std::pair<int, int>{2048, 2048}, {5120, 2304}}) {
        const int H = cfg.first, FF = cfg.second;
        cpu::NativeFmt f{};
        std::string err;
        if (!cpu::native_fmt(GGML_TYPE_MXFP4, GGML_TYPE_MXFP4, H, FF, f, err)) {
            std::printf("native_fmt fail: %s\n", err.c_str());
            return 1;
        }
        if (H == 2048) { f.gate_softcap = 4.f; f.up_softcap = 25.f; }
        std::vector<uint8_t> blob(f.bytes);
        for (size_t i = 0; i < blob.size(); i += 17) {
            blob[i] = (uint8_t) (118 + rng() % 12);
            for (int k = 1; k < 17 && i + k < blob.size(); ++k) blob[i + k] = (uint8_t) rng();
        }
        constexpr int T = 6;
        std::vector<std::vector<uint8_t>> act(T, std::vector<uint8_t>(cpu::kNativeActBytes));
        std::vector<std::vector<uint8_t>> hq(T, std::vector<uint8_t>(cpu::kNativeHBytes));
        const void *ap[T], *hp[T];
        std::vector<float> xv(H), hv(FF);
        for (int t = 0; t < T; ++t) {
            for (float& v : xv) v = nd(rng);
            for (float& v : hv) v = nd(rng);
            cpu::native_quant_act(f, xv.data(), act[t].data());
            cpu::native_quant_h(f, hv.data(), hq[t].data());
            ap[t] = act[t].data();
            hp[t] = hq[t].data();
        }
        constexpr int kCheckRows = 64;
        std::vector<std::vector<float>> ff(T, std::vector<float>(FF)), out(T, std::vector<float>(H));
        float *ffp[T], *outp[T];
        for (int t = 0; t < T; ++t) { ffp[t] = ff[t].data(); outp[t] = out[t].data(); }
        cpu::native_gu_rows(f, blob.data(), ap, T, ffp, 0, kCheckRows);
        cpu::native_down_rows(f, blob.data(), hp, T, outp, 0, kCheckRows);
        const cpu::SwiGlu glu{f.swiglu_limit, f.gate_softcap, f.up_softcap, f.gelu};
        size_t diff_gu = 0, diff_dn = 0;
        for (int t = 0; t < T; ++t) {
            for (int r = 0; r < kCheckRows; ++r) {
                float g = 0, u = 0, d = 0;
                tc->vec_dot(H, &g, 0, blob.data() + (size_t) r * f.gu_row, 0, ap[t], 0, 1);
                tc->vec_dot(H, &u, 0, blob.data() + f.up_off + (size_t) r * f.gu_row, 0, ap[t], 0, 1);
                const float want_gu = cpu::swiglu(glu, g, u);
                diff_gu += std::memcmp(&ff[t][r], &want_gu, 4) != 0;
                tc->vec_dot(FF, &d, 0, blob.data() + f.down_off + (size_t) r * f.d_row, 0, hp[t], 0, 1);
                diff_dn += std::memcmp(&out[t][r], &d, 4) != 0;
            }
        }
        std::printf("  dispatch (%dx%d, nt=%d): gu_diff=%zu dn_diff=%zu\n", H, FF, T, diff_gu, diff_dn);
        if (diff_gu || diff_dn) ++bad;
    }
    return bad;
}

void run_bench() {
    const bool vnni = cpu::cpu_avxvnni_ok(), avx512 = cpu::cpu_avx512_ok();
    const int hw = std::max(1, (int) std::thread::hardware_concurrency());
    const int threads = std::min(24, hw);
    std::printf("\nbench (%s, threads=%d):\n", cpu::cpu_name().c_str(), threads);
    std::mt19937 rng(123);
    std::normal_distribution<float> nd(0.f, 0.25f);
    for (auto cfg : {std::pair<int, int>{2048, 2048}, {5120, 2304}}) {
        const int H = cfg.first, FF = cfg.second;
        cpu::NativeFmt f{};
        std::string err;
        cpu::native_fmt(GGML_TYPE_MXFP4, GGML_TYPE_MXFP4, H, FF, f, err);
        const int n_exp = 64;   // ~250-400 MB working set, larger than L3
        std::vector<uint8_t> arena((size_t) n_exp * f.bytes);
        for (size_t i = 0; i < arena.size(); i += 17) arena[i] = 122;
        constexpr int T = 8;
        std::vector<std::vector<uint8_t>> act(T, std::vector<uint8_t>(cpu::kNativeActBytes));
        std::vector<std::vector<uint8_t>> hq(T, std::vector<uint8_t>(cpu::kNativeHBytes));
        const void *ap[T], *hp[T];
        std::vector<float> xv(H), hv(FF);
        for (int t = 0; t < T; ++t) {
            for (float& v : xv) v = nd(rng);
            for (float& v : hv) v = nd(rng);
            cpu::native_quant_act(f, xv.data(), act[t].data());
            cpu::native_quant_h(f, hv.data(), hq[t].data());
            ap[t] = act[t].data();
            hp[t] = hq[t].data();
        }
        const cpu::SwiGlu glu{0.f, 4.f, 25.f, false};
        auto bench_one = [&](const char* label, bool gate, Mode m, int nt, int nthr) {
            const int rows = gate ? FF : H;
            const size_t bytes = gate ? 2 * f.up_off : f.d_row * (size_t) H;
            const int reps = nthr == 1 ? 6 : 4;
            const int cnt = nthr == 1 ? 1 : n_exp;
            std::vector<std::vector<float>> out((size_t) nthr * (size_t) nt, std::vector<float>(rows));
            std::atomic<int> ready{0}, go{0};
            const int per = (rows + nthr - 1) / nthr;
            std::vector<std::thread> th;
            std::vector<double> secs(nthr);
            for (int tid = 0; tid < nthr; ++tid) th.emplace_back([&, tid] {
                const int r0 = tid * per, r1 = std::min(rows, r0 + per);
                float* op[T];
                for (int t = 0; t < nt; ++t) op[t] = out[(size_t) tid * nt + t].data();
                ready.fetch_add(1);
                while (!go.load()) {}
                const auto t0 = std::chrono::steady_clock::now();
                for (int rep = 0; rep < reps; ++rep)
                    for (int e = 0; e < cnt; ++e) {
                        const uint8_t* b = arena.data() + (size_t) e * f.bytes;
                        run_mode(m, gate, gate ? b : b + f.down_off, gate ? f.gu_row : f.d_row, f.up_off,
                                 gate ? H : FF, gate ? ap : hp, nt, op, r0, r1, glu);
                    }
                secs[tid] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            });
            while (ready.load() < nthr) {}
            go.store(1);
            for (auto& x : th) x.join();
            double mx = 0;
            for (double s : secs) mx = std::max(mx, s);
            const double us_per = mx * 1e6 / ((double) cnt * reps);
            const double gbps = (double) bytes * cnt * reps / mx / 1e9;
            std::printf("  %dx%d %-4s nt=%d %2dT %-9s: %7.1f us/expert  %6.2f GB/s\n", H, FF, gate ? "gu" : "down", nt,
                        nthr, label, us_per, gbps);
        };
        for (bool gate : {true, false}) {
            for (int nt : {1, 4, 6}) {
                bench_one("AVX2-base", gate, Mode::Avx2Base, nt, 1);
                if (vnni) bench_one("AVX2-vnni", gate, Mode::Avx2Vnni, nt, 1);
                if (avx512) bench_one("AVX512", gate, Mode::Avx512, nt, 1);
                if (threads > 1) {
                    bench_one("AVX2-base", gate, Mode::Avx2Base, nt, threads);
                    if (vnni) bench_one("AVX2-vnni", gate, Mode::Avx2Vnni, nt, threads);
                    if (avx512) bench_one("AVX512", gate, Mode::Avx512, nt, threads);
                }
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool bench = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--bench") bench = true;
    if (!cpu::cpu_avx2_ok()) {
        std::printf("mxfp4_cpu_rows_test SKIPPED: this CPU lacks AVX2\n");
        return 77;
    }
    std::vector<Mode> modes = {Mode::Avx2Base, Mode::Avx2Disp};
    if (cpu::cpu_avxvnni_ok()) modes.push_back(Mode::Avx2Vnni);
    if (cpu::cpu_avx512_ok()) modes.push_back(Mode::Avx512);

    int bad = 0;
    for (Mode m : modes) {
        bad += check(256, 33, true, false, m);
        bad += check(256, 33, false, false, m);
        bad += check(256, 33, true, true, m);
        bad += check(256, 33, false, true, m);
        bad += check(256, 33, true, false, m, cpu::SwiGlu{7.f, 0.f, 0.f, false});
        bad += check(256, 33, true, false, m, cpu::SwiGlu{0.f, 4.f, 25.f, false});
        bad += check(256, 33, true, false, m, cpu::SwiGlu{0.f, 0.f, 0.f, true});
        bad += check(2048, 37, true, false, m, cpu::SwiGlu{0.f, 4.f, 25.f, false});
        bad += check(2304, 37, false, false, m);
    }
    if (cpu::cpu_avx512_ok()) {
        cpu::mxfp4_512_gu_rows(nullptr, 0, 0, 256, cpu::SwiGlu{}, nullptr, 0, nullptr, 0, 1);
        cpu::mxfp4_512_down_rows(nullptr, 0, 256, nullptr, 0, nullptr, 0, 1);
    }
    bad += check_dispatch();
    std::printf("mxfp4_cpu_rows_test: %s\n", bad ? "FAIL" : "PASS");
    if (bench) run_bench();
    return bad ? 1 : 0;
}
