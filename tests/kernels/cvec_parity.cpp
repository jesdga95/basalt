// tests/kernels/cvec_parity.cpp - the control-vector kernel (basalt/kernels/cvec.hpp) against a host reference.
//
//   1. project: h - s (h.v) v, against double precision; the steered component is gone.
//   2. add: h + d.
//   3. switched off: R unchanged.  A layer without a direction: untouched.
//   4. a row's result does not depend on the rows beside it in the window.
//   5. a row wider than the kernel covers is refused at load.
//   6. cvec_replicate puts the tables on another card, and the switch reaches it too.
//
// The FP32 path's rows are a hyper-connection model's streams; its write's bitwise equivalence with the fused read
// that folds the same write is checked by tests/models/qwen4exp/kernels/cvec_fused_write_parity.cpp.  The BF16
// path's rows are a dense model's residual, at the 27B's width.  Both references round once, to the type, the way
// the engine stores do (round to nearest even: `__float2bfloat16_rn`, `bf16_from_f32`).
//
//     build/cvec_parity        the checks
#include "basalt/kernels/bf16_bits.hpp"
#include "basalt/kernels/cvec.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = basalt::kernels;

namespace {

int g_fail = 0;

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}

template <typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T)), "malloc");
    return p;
}

template <typename T>
void up(T* d, const std::vector<T>& h) { ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "up"); }
template <typename T>
std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "down");
    return h;
}

void check(bool ok, const char* what) {
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

/// The spacing of the BF16 grid at `x`'s magnitude: what one rounding can move a value.
double bf16_step(double x) { return std::ldexp(1.0, std::ilogb(std::fabs(x) + 1e-30) - 7); }

// a dense model's shape: its layer count and two widths, so two instantiations of the kernel run
constexpr int64_t LN = 64, WIDE = 5120, NARROW = 2560;
constexpr int64_t kLayer = 7, kOff = 3;   // steered / not steered

std::vector<uint16_t> bf_rows(int64_t rows, int64_t n, uint32_t seed, float scale) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, scale);
    std::vector<uint16_t> r((size_t) (rows * n));
    for (auto& x : r) x = k::bf16_from_f32(nd(rng));
    return r;
}

/// A unit direction for layer kLayer of a row of `n` values, none anywhere else, with its scale.
void steer(std::vector<float>& dir, std::vector<float>& s, int64_t n, uint32_t seed, float scale) {
    dir.assign((size_t) (LN * n), 0.0f);
    s.assign((size_t) LN, 0.0f);
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> v((size_t) n);
    double nrm = 0.0;
    for (auto& x : v) { x = nd(rng); nrm += (double) x * x; }
    nrm = std::sqrt(nrm);
    for (int64_t j = 0; j < n; ++j) dir[(size_t) (kLayer * n + j)] = (float) (v[(size_t) j] / nrm);
    s[(size_t) kLayer] = scale;
}

/// BF16 rows against `h + d` in double, rounded once: every value within one rounding of it.
void check_add(const std::vector<uint16_t>& got, const std::vector<uint16_t>& r, const std::vector<float>& d,
               int64_t rows, int64_t n, const char* what) {
    double worst = 0.0, worst_rel = 0.0;
    int64_t beyond = 0;
    for (int64_t i = 0; i < rows * n; ++i) {
        const double h = k::f32_from_bf16(r[(size_t) i]);
        const double want = h + d[(size_t) (i % n)];
        const double g = k::f32_from_bf16(got[(size_t) i]);
        const double err = std::fabs(g - want);
        worst = std::fmax(worst, err);
        worst_rel = std::fmax(worst_rel, err / std::fmax(std::fabs(want), 1e-3));
        if (err > 1.01 * bf16_step(std::fmax(std::fabs(want), std::fabs(h))) + 1e-9) ++beyond;
    }
    std::printf("    max |err| %.3g, max rel %.3g\n", worst, worst_rel);
    check(beyond == 0, what);
}

}  // namespace

int main() {
    // ---- the FP32 path: a hyper-connection model's streams.
    constexpr int64_t L = 48, N = 2560, HC = 4, T = 5, D = HC * N;
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<float> dir((size_t) (L * N), 0.0f), s((size_t) L, 0.0f);
    {
        double nrm = 0.0;
        std::vector<float> v((size_t) N);
        for (auto& x : v) { x = nd(rng); nrm += (double) x * x; }
        nrm = std::sqrt(nrm);
        for (int64_t j = 0; j < N; ++j) dir[(size_t) (kLayer * N + j)] = (float) (v[(size_t) j] / nrm);
        s[(size_t) kLayer] = 2.0f;
    }
    std::string err;
    if (!k::cvec_upload(dir, s, /*project*/ 0, 4, 44, N, HC, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
    check(k::cvec().covers(kLayer) && !k::cvec().covers(kOff) && !k::cvec().covers(45), "covers(): steered layers only");

    std::vector<float> R((size_t) (T * D)), bo((size_t) (T * N)), inj((size_t) (T * HC));
    for (auto& x : R) x = nd(rng) * 3.0f;
    for (auto& x : bo) x = nd(rng);
    for (auto& x : inj) x = nd(rng);
    float* dR = dalloc<float>(R.size());
    float* dbo = dalloc<float>(bo.size());
    float* dinj = dalloc<float>(inj.size());
    up(dbo, bo);
    up(dinj, inj);

    std::printf("project, fp32 streams\n");
    up(dR, R);
    k::cvec_apply(dR, kLayer, T, D, nullptr, 0, nullptr, 0, false, nullptr);
    ck(cudaDeviceSynchronize(), "project");
    {
        const auto got = down(dR, R.size());
        double worst = 0.0, worst_dot = 0.0;
        for (int64_t t = 0; t < T; ++t)
            for (int64_t c = 0; c < HC; ++c) {
                const float* h = R.data() + t * D + c * N;
                const float* v = dir.data() + kLayer * N;
                double dot = 0.0;
                for (int64_t j = 0; j < N; ++j) dot += (double) h[j] * v[j];
                double after = 0.0;
                for (int64_t j = 0; j < N; ++j) {
                    const double want = h[j] - 2.0 * dot * v[j];
                    const double g = got[(size_t) (t * D + c * N + j)];
                    worst = std::fmax(worst, std::fabs(g - want));
                    after += g * v[j];
                }
                worst_dot = std::fmax(worst_dot, std::fabs(after + dot));   // s = 2 reflects: h'.v -> -h.v
            }
        std::printf("    max |err| %.3g, max |h'.v + h.v| %.3g\n", worst, worst_dot);
        check(worst < 1e-4, "project matches h - s (h.v) v (s = 2)");
        check(worst_dot < 1e-3, "the component along v is reflected (s = 2)");
    }
    // s = 1 removes it
    s[(size_t) kLayer] = 1.0f;
    if (!k::cvec_upload(dir, s, 0, 4, 44, N, HC, err)) return 2;
    up(dR, R);
    k::cvec_apply(dR, kLayer, T, D, nullptr, 0, nullptr, 0, false, nullptr);
    ck(cudaDeviceSynchronize(), "project s=1");
    {
        const auto got = down(dR, R.size());
        double worst = 0.0;
        for (int64_t t = 0; t < T; ++t)
            for (int64_t c = 0; c < HC; ++c) {
                double after = 0.0;
                for (int64_t j = 0; j < N; ++j) after += (double) got[(size_t) (t * D + c * N + j)] * dir[(size_t) (kLayer * N + j)];
                worst = std::fmax(worst, std::fabs(after));
            }
        std::printf("    max |h'.v| %.3g\n", worst);
        check(worst < 1e-3, "s = 1 projects the direction out");
    }

    up(dR, R);
    k::cvec_apply(dR, kOff, T, D, nullptr, 0, nullptr, 0, false, nullptr);
    ck(cudaDeviceSynchronize(), "off layer");
    check(std::memcmp(down(dR, R.size()).data(), R.data(), R.size() * 4) == 0, "fp32: a layer without a direction is untouched");

    std::printf("switched off\n");
    k::cvec_set_enabled(false);
    check(!k::cvec_enabled(), "cvec_enabled() follows the switch");
    up(dR, R);
    k::cvec_apply(dR, kLayer, T, D, nullptr, 0, nullptr, 0, false, nullptr);
    ck(cudaDeviceSynchronize(), "off");
    check(std::memcmp(down(dR, R.size()).data(), R.data(), R.size() * 4) == 0, "off, no pending write: R unchanged");
    k::cvec_set_enabled(true);

    std::printf("add, fp32 streams\n");
    for (int64_t j = 0; j < N; ++j) dir[(size_t) (kLayer * N + j)] *= 0.1f;   // d = 0.1 v, s = 1
    s[(size_t) kLayer] = 1.0f;
    if (!k::cvec_upload(dir, s, /*add*/ 1, 4, 44, N, HC, err)) return 2;
    up(dR, R);
    k::cvec_apply(dR, kLayer, T, D, nullptr, 0, nullptr, 0, false, nullptr);
    ck(cudaDeviceSynchronize(), "add");
    {
        const auto got = down(dR, R.size());
        bool same = true;
        for (int64_t t = 0; t < T; ++t)
            for (int64_t c = 0; c < HC; ++c)
                for (int64_t j = 0; j < N; ++j) {
                    const size_t i = (size_t) (t * D + c * N + j);
                    same = same && got[i] == R[i] + dir[(size_t) (kLayer * N + j)];
                }
        check(same, "add is h + d in every stream (bitwise)");
    }

    check(!k::cvec_upload(std::vector<float>((size_t) (L * 9000), 0.0f), s, 0, 4, 44, 9000, HC, err) && !err.empty(),
          "a row wider than the kernel covers is refused at load");

    // ---- the BF16 path: a dense model's residual rows, at the 27B's width.
    std::printf("project, bf16 rows (the 27B's width)\n");
    {
        steer(dir, s, WIDE, 99, 1.0f);   // s = 1: the direction comes out of the residual
        if (!k::cvec_upload(dir, s, /*project*/ 0, 4, LN - 1, WIDE, 1, err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 2;
        }
        check(k::cvec().covers(kLayer) && !k::cvec().covers(kOff) && !k::cvec().covers(LN),
              "covers(): a dense model's steered layers only");
        const auto r = bf_rows(T, WIDE, 4242, 3.0f);
        uint16_t* dR = dalloc<uint16_t>(r.size());
        up(dR, r);
        k::cvec_apply_bf16(dR, kLayer, T, WIDE, nullptr);
        ck(cudaDeviceSynchronize(), "bf16 project");
        const auto got = down(dR, r.size());
        double worst = 0.0, worst_rel = 0.0, worst_after = 0.0, worst_after_ref = 0.0;
        int64_t beyond = 0, leftover = 0;
        for (int64_t t = 0; t < T; ++t) {
            const uint16_t* h = r.data() + t * WIDE;
            const float* v = dir.data() + kLayer * WIDE;
            double dot = 0.0;
            for (int64_t j = 0; j < WIDE; ++j) dot += (double) k::f32_from_bf16(h[j]) * v[j];
            double after = 0.0, after_ref = 0.0;
            for (int64_t j = 0; j < WIDE; ++j) {
                const size_t i = (size_t) (t * WIDE + j);
                const double hv = k::f32_from_bf16(h[j]);
                const double want = hv - dot * v[j];   // (v is unit; s = 1)
                const double g = k::f32_from_bf16(got[i]);
                const double err = std::fabs(g - want);
                worst = std::fmax(worst, err);
                worst_rel = std::fmax(worst_rel, err / std::fmax(std::fabs(want), 1e-3));
                if (err > 1.01 * bf16_step(std::fmax(std::fabs(want), std::fabs(hv))) + 1e-9) ++beyond;
                after += g * v[j];
                after_ref += (double) k::f32_from_bf16(k::bf16_from_f32((float) want)) * v[j];
            }
            // what the row still carries along v.  A row of the residual is BF16, so storing the result leaves a
            // component of its own (after_ref): the vector is right when it leaves no more than that.
            worst_after = std::fmax(worst_after, std::fabs(after));
            worst_after_ref = std::fmax(worst_after_ref, std::fabs(after_ref));
            if (std::fabs(after) > 2.0 * std::fabs(after_ref) + 1e-3) ++leftover;
        }
        std::printf("    max |err| %.3g, max rel %.3g, largest |h'.v|: engine %.4f, from one rounding of the result alone %.4f\n",
                    worst, worst_rel, worst_after, worst_after_ref);
        check(beyond == 0, "project matches h - s (h.v) v, one rounding to BF16");
        check(leftover == 0, "s = 1 leaves nothing along v beyond what storing the row in BF16 leaves");

        up(dR, r);
        k::cvec_apply_bf16(dR, kOff, T, WIDE, nullptr);
        ck(cudaDeviceSynchronize(), "bf16 off layer");
        check(down(dR, r.size()) == r, "bf16: a layer without a direction is untouched");

        k::cvec_set_enabled(false);
        up(dR, r);
        k::cvec_apply_bf16(dR, kLayer, T, WIDE, nullptr);
        ck(cudaDeviceSynchronize(), "bf16 off");
        check(down(dR, r.size()) == r, "bf16, switched off: the rows are unchanged, bitwise");
        k::cvec_set_enabled(true);

        std::printf("window width\n");
        uint16_t* one = dalloc<uint16_t>((size_t) WIDE);
        bool same = true;
        for (int64_t t = 0; t < T && same; ++t) {
            up(one, std::vector<uint16_t>(r.begin() + t * WIDE, r.begin() + (t + 1) * WIDE));
            k::cvec_apply_bf16(one, kLayer, 1, WIDE, nullptr);
            ck(cudaDeviceSynchronize(), "bf16 one row");
            const auto row = down(one, WIDE);
            same = std::memcmp(row.data(), got.data() + (size_t) t * WIDE, (size_t) WIDE * 2) == 0;
        }
        check(same, "a row's result is the same alone as beside four others");
        ck(cudaFree(one), "free the one-row buffer");
    }

    std::printf("add, bf16 rows\n");
    {
        const int64_t rows = 3;
        steer(dir, s, NARROW, 77, 1.0f);
        for (int64_t j = 0; j < NARROW; ++j) dir[(size_t) (kLayer * NARROW + j)] *= 0.1f;   // d = 0.1 v
        if (!k::cvec_upload(dir, s, /*add*/ 1, 4, LN - 1, NARROW, 1, err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 2;
        }
        const auto r = bf_rows(rows, NARROW, 5151, 3.0f);
        uint16_t* dR = dalloc<uint16_t>(r.size());
        up(dR, r);
        k::cvec_apply_bf16(dR, kLayer, rows, NARROW, nullptr);
        ck(cudaDeviceSynchronize(), "bf16 add");
        check_add(down(dR, r.size()), r, std::vector<float>(dir.begin() + kLayer * NARROW, dir.begin() + (kLayer + 1) * NARROW),
                  rows, NARROW, "add matches h + d, one rounding to BF16");
        ck(cudaFree(dR), "free the add buffer");

        // ---- another card holds the same vector (a layer split, a prompt line).
        int devices = 0;
        ck(cudaGetDeviceCount(&devices), "device count");
        std::printf("replicate\n");
        if (devices < 2) {
            std::printf("  one device, skipped\n");
        } else {
            for (int d = 1; d < devices; ++d) {
                ck(cudaSetDevice(d), "set the stage's device");
                if (!k::cvec_replicate(err)) {
                    std::fprintf(stderr, "cvec_replicate: %s\n", err.c_str());
                    check(false, "cvec_replicate puts the tables on the stage's device");
                    break;
                }
                check(true, "cvec_replicate puts the tables on the stage's device");
                const auto r2 = bf_rows(rows, NARROW, 616, 3.0f);
                uint16_t* dr = dalloc<uint16_t>(r2.size());
                up(dr, r2);
                k::cvec_apply_bf16(dr, kLayer, rows, NARROW, nullptr);
                ck(cudaDeviceSynchronize(), "apply on the stage's device");
                check_add(down(dr, r2.size()), r2,
                          std::vector<float>(dir.begin() + kLayer * NARROW, dir.begin() + (kLayer + 1) * NARROW), rows, NARROW,
                          "the stage's device applies the same add");
                k::cvec_set_enabled(false);
                up(dr, r2);
                k::cvec_apply_bf16(dr, kLayer, rows, NARROW, nullptr);
                ck(cudaDeviceSynchronize(), "apply off on the stage's device");
                check(down(dr, r2.size()) == r2, "the switch reaches the stage's device (off: unchanged)");
                k::cvec_set_enabled(true);
                ck(cudaFree(dr), "free the stage device's buffer");
            }
            ck(cudaSetDevice(0), "back to the first device");
        }
    }

    std::printf(g_fail ? "cvec_parity: %d FAILED\n" : "cvec_parity: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
