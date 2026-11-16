// tests/models/qwen4exp/kernels/cvec_fused_write_parity.cpp - the control vector's write (basalt/kernels/cvec.hpp's
// `cvec_apply` with `write`) against Flash-Next's fused hyper-connection read doing the same write
// (basalt/models/qwen4exp/kernels/fused_gr.hpp with `apply`).
//
// The fused read folds a layer's FFN write into the next layer's read; a layer the vector follows cannot fold it,
// because the vector comes between the two, so `cvec_apply` writes it instead.  The vector being off - a request
// that did not ask for it - must then leave the engine's output BITWISE as it was without a vector at all, which is
// this kernel's write being the read's folded write, bit for bit.
//
// The vector itself, against a host reference: tests/kernels/cvec_parity.cpp.
//
//     build/cvec_fused_write_parity        the checks
#include "basalt/kernels/cvec.hpp"
#include "basalt/models/qwen4exp/kernels/fused_gr.hpp"

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

}  // namespace

int main() {
    constexpr int64_t L = 48, N = 2560, HC = 4, LR = 320, T = 5, D = HC * N;
    constexpr int64_t kLayer = 7;
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // a unit direction at kLayer with scale 1 (h -= (h.v) v); nothing elsewhere
    std::vector<float> dir((size_t) (L * N), 0.0f), s((size_t) L, 0.0f);
    {
        double nrm = 0.0;
        std::vector<float> v((size_t) N);
        for (auto& x : v) { x = nd(rng); nrm += (double) x * x; }
        nrm = std::sqrt(nrm);
        for (int64_t j = 0; j < N; ++j) dir[(size_t) (kLayer * N + j)] = (float) (v[(size_t) j] / nrm);
        s[(size_t) kLayer] = 1.0f;
    }
    std::string err;
    if (!k::cvec_upload(dir, s, /*project*/ 0, 4, 44, N, HC, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }

    std::vector<float> R((size_t) (T * D)), bo((size_t) (T * N)), inj((size_t) (T * HC));
    for (auto& x : R) x = nd(rng) * 3.0f;
    for (auto& x : bo) x = nd(rng);
    for (auto& x : inj) x = nd(rng);
    float* dR = dalloc<float>(R.size());
    float* dbo = dalloc<float>(bo.size());
    float* dinj = dalloc<float>(inj.size());
    up(dbo, bo);
    up(dinj, inj);

    // the fused read's folded write, one token at a time, as the engine runs it
    std::vector<uint16_t> wd((size_t) (LR * D)), wu((size_t) (D * LR)), wi((size_t) (HC * D));
    std::vector<float> wn((size_t) D);
    auto bf = [&](float x) { uint32_t u; std::memcpy(&u, &x, 4); return (uint16_t) (u >> 16); };
    for (auto& x : wd) x = bf(nd(rng) * 0.02f);
    for (auto& x : wu) x = bf(nd(rng) * 0.02f);
    for (auto& x : wi) x = bf(nd(rng));
    for (auto& x : wn) x = 1.0f + 0.1f * nd(rng);
    uint16_t* dwd = dalloc<uint16_t>(wd.size());
    uint16_t* dwu = dalloc<uint16_t>(wu.size());
    uint16_t* dwi = dalloc<uint16_t>(wi.size());
    float* dwn = dalloc<float>(wn.size());
    up(dwd, wd); up(dwu, wu); up(dwi, wi); up(dwn, wn);
    float* lo = dalloc<float>(LR);
    float* rs = dalloc<float>(HC);
    float* inj_out = dalloc<float>(HC);
    float* mixed = dalloc<float>(N);
    float* dF = dalloc<float>(R.size());
    up(dF, R);
    for (int64_t t = 0; t < T; ++t) {
        k::FusedGrArgs a;
        a.R = dF + t * D; a.R_out = dF + t * D; a.apply = true;
        a.bo_prev = dbo + t * N; a.inj_prev = dinj + t * HC;
        a.w_norm = dwn; a.w_down = dwd; a.w_up = dwu; a.w_inject = dwi;
        a.lo = lo; a.rs = rs; a.inject_out = inj_out; a.mixed = mixed;
        k::fused_gr_read(a, nullptr);
    }
    ck(cudaDeviceSynchronize(), "fused read");
    std::printf("the write a loaded vector leaves behind\n");
    k::cvec_set_enabled(false);
    up(dR, R);
    k::cvec_apply(dR, kLayer, T, D, dbo, N, dinj, HC, true, nullptr);
    ck(cudaDeviceSynchronize(), "write only");
    {
        const auto fused = down(dF, R.size()), mine = down(dR, R.size());
        check(std::memcmp(fused.data(), mine.data(), R.size() * 4) == 0,
              "off, pending write: bitwise the fused read's folded write");
    }
    k::cvec_set_enabled(true);
    // on: the write, then the projection, against the reference from the fused result
    up(dR, R);
    k::cvec_apply(dR, kLayer, T, D, dbo, N, dinj, HC, true, nullptr);
    ck(cudaDeviceSynchronize(), "write+project");
    {
        const auto fused = down(dF, R.size()), both = down(dR, R.size());
        double worst = 0.0;
        for (int64_t t = 0; t < T; ++t)
            for (int64_t c = 0; c < HC; ++c) {
                const float* h = fused.data() + t * D + c * N;
                const float* v = dir.data() + kLayer * N;
                double dot = 0.0;
                for (int64_t j = 0; j < N; ++j) dot += (double) h[j] * v[j];
                for (int64_t j = 0; j < N; ++j)
                    worst = std::fmax(worst, std::fabs(both[(size_t) (t * D + c * N + j)] - (h[j] - dot * v[j])));
            }
        std::printf("    write + project: max |err| %.3g\n", worst);
        check(worst < 1e-4, "on, pending write: the write, then the projection");
    }

    std::printf(g_fail ? "cvec_fused_write_parity: %d FAILED\n" : "cvec_fused_write_parity: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
