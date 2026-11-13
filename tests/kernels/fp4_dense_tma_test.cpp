// tests/kernels/fp4_dense_tma_test.cpp - the TMA dense W4A4 product (prefill::fp4_dense) against a double-precision
// reference over the same quantized codes and scales: the plain product, the Add epilogue and the SwiGLU product; a
// token's row is the same bits at 1, 1000 and 1100 tokens.  --bench: TFLOP/s at the Qwen3.8-27B MLP shapes against the
// cp.async path of prefill::fp4::run.  Exit 77 without a GPU.
#include "basalt/kernels/nvfp4_planes.hpp"
#include "basalt/prefill/fp4_dense.hpp"
#include "basalt/prefill/moe_fp4.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace fd = basalt::prefill::fp4_dense;
namespace fp4 = basalt::prefill::fp4;
namespace np = basalt::kernels::nvfp4p;

namespace {

constexpr double kE2m1[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
double code_value(int c) { return (c & 8 ? -1.0 : 1.0) * kE2m1[c & 7]; }
double ue4m3(int b) {
    const int e = (b >> 3) & 15, m = b & 7;
    if (e == 15 && m == 7) return 0;
    return e == 0 ? std::ldexp(m / 8.0, -6) : std::ldexp(1.0 + m / 8.0, e - 7);
}

struct Planes {
    std::vector<uint8_t> bytes;
    std::vector<float> dec;   // the decoded values [rows][K]
};
Planes make_planes(int64_t rows, int64_t K, std::mt19937& rng) {
    Planes p;
    p.bytes.assign(np::bytes(rows, K), 0);
    p.dec.resize((size_t) rows * K);
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t g = 0; g < K / 16; ++g) {
            const int sb = (5 << 3 | (rng() & 7)) + 8 * (int) (rng() % 5);   // exponents 5 .. 9
            p.bytes[np::scale_offset(rows, K) + np::scale_index(r, g, K)] = (uint8_t) sb;
            for (int j = 0; j < 16; ++j) {
                const int c = (int) (rng() & 15);
                const int64_t k = g * 16 + j;
                uint8_t& b = p.bytes[(size_t) r * (K / 2) + k / 2];
                b |= (uint8_t) (c << (4 * (k & 1)));
                p.dec[(size_t) r * K + k] = (float) (code_value(c) * ue4m3(sb));
            }
        }
    return p;
}

// the activations as the quantizer wrote them: decoded values (without the token scale) from the planes
std::vector<float> decode_act(const std::vector<uint8_t>& act, int64_t T, int64_t K) {
    std::vector<float> x((size_t) T * K);
    const uint8_t* scales = act.data() + (size_t) T * (K / 2);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t k = 0; k < K; ++k) {
            const int b = act[(size_t) t * (K / 2) + k / 2], c = (k & 1) ? b >> 4 : b & 15, g = (int) (k / 16);
            const int64_t tile = (t / 256) * (K / 256) + g / 16;
            const int sb = scales[tile * 4096 + (t % 256) * 16 + g % 16];
            x[(size_t) t * K + k] = (float) (code_value(c) * ue4m3(sb));
        }
    return x;
}

template <class T> T* dev_copy(const std::vector<T>& v) {
    T* d;
    cudaMalloc(&d, v.size() * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
template <class T> std::vector<T> host_copy(const T* d, size_t n) {
    std::vector<T> v(n);
    cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return v;
}

struct Ref {
    std::vector<double> sum, abs;   // [T][rows]
};
Ref reference(const std::vector<float>& w, const std::vector<float>& x, int64_t rows, int64_t K, int64_t T) {
    Ref r;
    r.sum.resize((size_t) T * rows);
    r.abs.resize((size_t) T * rows);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t i = 0; i < rows; ++i) {
            double s = 0, a = 0;
            const float* wr = &w[(size_t) i * K];
            const float* xr = &x[(size_t) t * K];
            for (int64_t k = 0; k < K; ++k) {
                const double v = (double) wr[k] * xr[k];
                s += v;
                a += std::fabs(v);
            }
            r.sum[(size_t) t * rows + i] = s;
            r.abs[(size_t) t * rows + i] = a;
        }
    return r;
}

int bench();

}  // namespace

int main(int argc, char** argv) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) return 77;
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) return bench();

    const int64_t R = 512, K = 1280, T = 1100, T2 = 1000;   // R a multiple of 256: the SwiGLU output is quantized again
    std::mt19937 rng(11);
    const Planes wg = make_planes(R, K, rng), wu = make_planes(R, K, rng);
    std::normal_distribution<float> nd;
    std::vector<float> x((size_t) T * K);
    for (size_t i = 0; i < x.size(); ++i) x[i] = nd(rng) * (1.0f + (float) (i / K % 7));
    for (int64_t k = 0; k < K; k += 16) x[5 * K + k] *= 50.0f;   // outliers inside a token
    const float sg = 0.37f, su = 1.9f;
    const float scales[2] = {sg, su};

    uint8_t *dwg = dev_copy(wg.bytes), *dwu = dev_copy(wu.bytes);
    float* dx = dev_copy(x);
    float* dsc = dev_copy(std::vector<float>(scales, scales + 2));
    void* act;
    float *gs, *y, *y2, *y1, *yh;
    cudaMalloc(&act, fd::act_bytes(T, K));
    cudaMalloc(&gs, T * 4);
    cudaMalloc(&y, (size_t) T * R * 4);
    cudaMalloc(&y2, (size_t) T * R * 4);
    cudaMalloc(&y1, (size_t) R * 4);
    cudaMalloc(&yh, (size_t) T * R * 4);

    fd::quantize(dx, T, K, K, act, gs, nullptr);
    fd::product(dwg, dsc, R, K, act, gs, T, y, R, false, nullptr);
    fd::swiglu(dwg, dwu, dsc, dsc + 1, R, K, act, gs, T, yh, R, nullptr);
    // the Add epilogue on top of the plain product's output
    std::vector<float> res((size_t) T * R);
    for (size_t i = 0; i < res.size(); ++i) res[i] = nd(rng);
    float* dres = dev_copy(res);
    fd::product(dwg, dsc, R, K, act, gs, T, dres, R, true, nullptr);
    // fewer tokens, each token quantized again, from the same buffers
    fd::quantize(dx, T2, K, K, act, gs, nullptr);
    fd::product(dwg, dsc, R, K, act, gs, T2, y2, R, false, nullptr);
    fd::quantize(dx, 1, K, K, act, gs, nullptr);
    fd::product(dwg, dsc, R, K, act, gs, 1, y1, R, false, nullptr);
    const bool launch_err = cudaDeviceSynchronize() != cudaSuccess;

    fd::quantize(dx, T, K, K, act, gs, nullptr);
    cudaDeviceSynchronize();
    const auto actb = host_copy((const uint8_t*) act, fd::act_bytes(T, K));
    const auto gsh = host_copy(gs, T);
    const auto xd = decode_act(actb, T, K);
    const Ref rg = reference(wg.dec, xd, R, K, T), ru = reference(wu.dec, xd, R, K, T);
    const auto hy = host_copy(y, (size_t) T * R), hy2 = host_copy(y2, (size_t) T2 * R), hy1 = host_copy(y1, R);
    const auto hh = host_copy(yh, (size_t) T * R), hres = host_copy(dres, (size_t) T * R);

    double worst_plain = 0, worst_add = 0, worst_sw = 0, worst_q = 0;
    for (int64_t t = 0; t < T; ++t)
        for (int64_t i = 0; i < R; ++i) {
            const size_t o = (size_t) t * R + i;
            const double g = (double) gsh[t] * sg, u = (double) gsh[t] * su;
            const double den = (double) gsh[t] * sg * rg.abs[o] + 1e-30;
            worst_plain = std::max(worst_plain, std::fabs(hy[o] - g * rg.sum[o]) / den);
            worst_add = std::max(worst_add, std::fabs(hres[o] - (res[o] + g * rg.sum[o])) / den);
            const double gv = g * rg.sum[o], uv = u * ru.sum[o];
            const double ref = gv / (1.0 + std::exp(-gv)) * uv;
            const double gb = g * rg.abs[o], ub = u * ru.abs[o];
            const double tol = 1.1 * gb * std::fabs(uv) + std::fabs(gv / (1.0 + std::exp(-gv))) * ub + 1e-30;
            worst_sw = std::max(worst_sw, std::fabs(hh[o] - ref) / tol);
        }
    // the quantizer: each token's scale is amax / (6 * 448) and the decoded row is within the FP4 grid's reach of x
    for (int64_t t = 0; t < T; ++t) {
        double amax = 0, err = 0;
        for (int64_t k = 0; k < K; ++k) {
            amax = std::max(amax, (double) std::fabs(x[(size_t) t * K + k]));
            err = std::max(err, std::fabs(x[(size_t) t * K + k] - (double) gsh[t] * xd[(size_t) t * K + k]));
        }
        worst_q = std::max(worst_q, std::fabs(gsh[t] - amax / (6.0 * 448.0)) / (amax / (6.0 * 448.0)));
        worst_q = std::max(worst_q, err / amax / 0.25);   // half a step of 2 .. 3 spacing near the top, with the scale's rounding
    }
    // the norm in the quantizer: its codes and token scales are quantize()'s of its own normed rows
    bool norm_same = false;
    {
        std::vector<float> nw(K);
        for (auto& v : nw) v = 1.0f + 0.1f * nd(rng);
        float* dnw = dev_copy(nw);
        float *dxn, *gs2;
        void* act2;
        cudaMalloc(&dxn, (size_t) T * K * 4);
        cudaMalloc(&gs2, T * 4);
        cudaMalloc(&act2, fd::act_bytes(T, K));
        fd::rms_norm_quantize(dx, dnw, 1e-6f, T, K, act, gs, dxn, nullptr);
        fd::quantize(dxn, T, K, K, act2, gs2, nullptr);
        cudaDeviceSynchronize();
        norm_same = host_copy((const uint8_t*) act, fd::act_bytes(T, K)) == host_copy((const uint8_t*) act2, fd::act_bytes(T, K)) &&
                    host_copy(gs, T) == host_copy(gs2, T);
        std::printf("fp4 dense rms_norm_quantize: codes and scales %s\n", norm_same ? "bitwise quantize's of its normed rows" : "DIFFER");
    }
    // SwiGLU's per-token amax: quantize_amax of its output gives quantize's codes and scales, reading the rows once
    bool amax_same = false;
    {
        float *hmax, *gs3, *gs4;
        void *a3, *a4;
        cudaMalloc(&hmax, T * 4);
        cudaMalloc(&gs3, T * 4);
        cudaMalloc(&gs4, T * 4);
        cudaMalloc(&a3, fd::act_bytes(T, R));
        cudaMalloc(&a4, fd::act_bytes(T, R));
        fd::quantize(dx, T, K, K, act, gs, nullptr);
        fd::swiglu(dwg, dwu, dsc, dsc + 1, R, K, act, gs, T, yh, R, nullptr, hmax);
        fd::quantize(yh, T, R, R, a3, gs3, nullptr);
        fd::quantize_amax(yh, hmax, T, R, R, a4, gs4, nullptr);
        cudaDeviceSynchronize();
        amax_same = host_copy((const uint8_t*) a3, fd::act_bytes(T, R)) == host_copy((const uint8_t*) a4, fd::act_bytes(T, R)) &&
                    host_copy(gs3, T) == host_copy(gs4, T);
        std::printf("fp4 dense swiglu amax + quantize_amax: %s\n", amax_same ? "bitwise quantize's" : "DIFFER");
    }
    const bool same = std::memcmp(hy.data(), hy2.data(), (size_t) T2 * R * 4) == 0;
    const bool same1 = std::memcmp(hy.data(), hy1.data(), (size_t) R * 4) == 0;
    const bool ok = !launch_err && worst_plain < 1e-5 && worst_add < 1e-5 && worst_sw < 1e-5 && worst_q < 1.0 && same && same1 && norm_same && amax_same;
    std::printf("fp4 dense tma [%lld x %lld] x %lld tokens: plain max err %.3g, add %.3g, swiglu %.3g (x sum|wx|), quantizer %.3g; "
                "rows bitwise at 1, 1000, 1100 tokens: %s%s\n",
                (long long) R, (long long) K, (long long) T, worst_plain, worst_add, worst_sw, worst_q,
                same && same1 ? "yes" : "NO", ok ? "" : "  FAIL");
    return ok ? 0 : 1;
}

namespace {

uint64_t xs = 88172645463325252ull;
uint8_t rnd() {
    xs ^= xs << 13; xs ^= xs >> 7; xs ^= xs << 17;
    return (uint8_t) (xs >> 24);
}

template <class F> float time_ms(F&& f, int iters = 20) {
    for (int i = 0; i < 3; ++i) f();
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) f();
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    return ms / iters;
}

std::vector<uint8_t> bench_planes(int64_t rows, int64_t K) {
    std::vector<uint8_t> w(np::bytes(rows, K));
    const size_t so = np::scale_offset(rows, K);
    for (size_t i = 0; i < so; ++i) w[i] = rnd();
    for (size_t i = so; i < w.size(); ++i) w[i] = (uint8_t) (0x30 + (rnd() & 15));
    return w;
}

int bench() {
    const int64_t gate_rows = 17408, down_rows = 5120, K1 = 5120, K2 = 17408;
    const auto bg = bench_planes(gate_rows, K1), bu = bench_planes(gate_rows, K1), bd = bench_planes(down_rows, K2);
    uint8_t *wg = dev_copy(bg), *wu = dev_copy(bu), *wd = dev_copy(bd);
    uint8_t *cg, *cu, *cd;   // the same matrices as ggml blocks for the cp.async path
    cudaMalloc(&cg, bg.size());
    cudaMalloc(&cu, bg.size());
    cudaMalloc(&cd, bd.size());
    np::to_blocks(wg, gate_rows, K1, cg, nullptr);
    np::to_blocks(wu, gate_rows, K1, cu, nullptr);
    np::to_blocks(wd, down_rows, K2, cd, nullptr);
    const float sc[4] = {0.7f, 1.3f, 1.0f, 0.0f};
    float* dsc = dev_copy(std::vector<float>(sc, sc + 4));
    std::mt19937 rng(3);
    std::normal_distribution<float> nd;
    int rc = 0;
    for (const int64_t T : {2048, 4096}) {
        std::vector<float> x1((size_t) T * K1), x2((size_t) T * K2);
        for (auto& v : x1) v = nd(rng);
        for (auto& v : x2) v = nd(rng);
        float *dx1 = dev_copy(x1), *dx2 = dev_copy(x2), *gs, *out, *out2;
        void *act1, *act2, *a4a, *a4b;
        int32_t *bounds, *toff;
        cudaMalloc(&act1, fd::act_bytes(T, K1));
        cudaMalloc(&act2, fd::act_bytes(T, K2));
        cudaMalloc(&a4a, fp4::a4_bytes(T, K1, 1));
        cudaMalloc(&a4b, fp4::a4_bytes(T, K2, 1));
        cudaMalloc(&gs, fp4::scale_count(T, 1) * 4);
        cudaMalloc(&out, (size_t) T * gate_rows * 4);
        cudaMalloc(&out2, (size_t) T * gate_rows * 4);
        cudaMalloc(&bounds, 8);
        cudaMalloc(&toff, 4 * (fp4::kMaxExperts + 1));
        const int32_t b[2] = {0, (int32_t) T};
        cudaMemcpy(bounds, b, 8, cudaMemcpyHostToDevice);
        fp4::plan(bounds, 1, toff, nullptr);

        const double flop_gu = 2.0 * T * (2 * gate_rows) * K1, flop_d = 2.0 * T * down_rows * K2;
        // the expert path's plain products (gate and up as two of them)
        auto old_product = [&](const uint8_t* w, const uint8_t* up, int64_t rows, int64_t K, void* a4, float* dst) {
            for (const uint8_t* m : {w, up}) {
                if (!m) continue;
                fp4::Product p;
                p.blob[0] = m;
                p.w_rows = rows;
                p.w_cols = K;
                p.n = 1;
                p.a4 = a4;
                p.gs = gs;
                p.bounds = bounds;
                p.tile_off = toff;
                p.dst = dst;
                p.ld_dst = rows;
                p.max_rows = T;
                fp4::run(p, nullptr);
            }
        };
        const float q1 = time_ms([&] { fd::quantize(dx1, T, K1, K1, act1, gs, nullptr); });
        const float q2 = time_ms([&] { fd::quantize(dx2, T, K2, K2, act2, gs, nullptr); });
        fd::quantize(dx1, T, K1, K1, act1, gs, nullptr);
        const float n_gu = time_ms([&] { fd::swiglu(wg, wu, dsc, dsc + 1, gate_rows, K1, act1, gs, T, out, gate_rows, nullptr); });
        fd::quantize(dx2, T, K2, K2, act2, gs, nullptr);
        const float n_d = time_ms([&] { fd::product(wd, dsc + 2, down_rows, K2, act2, gs, T, out2, down_rows, false, nullptr); });
        const float n_da = time_ms([&] { fd::product(wd, dsc + 2, down_rows, K2, act2, gs, T, out2, down_rows, true, nullptr); });
        fp4::quantize(dx1, nullptr, bounds, toff, 1, a4a, gs, K1, K1, T, nullptr);
        const float o_gu = time_ms([&] { old_product(cg, cu, gate_rows, K1, a4a, out); });
        fp4::quantize(dx2, nullptr, bounds, toff, 1, a4b, gs, K2, K2, T, nullptr);
        const float o_d = time_ms([&] { old_product(cd, nullptr, down_rows, K2, a4b, out2); });
        const float oq1 = time_ms([&] { fp4::quantize(dx1, nullptr, bounds, toff, 1, a4a, gs, K1, K1, T, nullptr); });
        const float oq2 = time_ms([&] { fp4::quantize(dx2, nullptr, bounds, toff, 1, a4b, gs, K2, K2, T, nullptr); });
        const auto tf = [](double flop, float ms) { return flop / (ms * 1e-3) / 1e12; };
        std::printf("T=%lld  gate+up swiglu [34816 x 5120]: tma %.3f ms (%.0f TFLOP/s)  cp.async %.3f ms (%.0f)\n", (long long) T, n_gu,
                    tf(flop_gu, n_gu), o_gu, tf(flop_gu, o_gu));
        std::printf("T=%lld  down [5120 x 17408]:           tma %.3f ms (%.0f TFLOP/s)  add %.3f ms (%.0f)  cp.async %.3f ms (%.0f)\n",
                    (long long) T, n_d, tf(flop_d, n_d), n_da, tf(flop_d, n_da), o_d, tf(flop_d, o_d));
        std::printf("T=%lld  quantize K=5120: %.3f ms (cp.async path %.3f)  K=17408: %.3f ms (cp.async path %.3f)\n", (long long) T, q1,
                    oq1, q2, oq2);
        if (cudaGetLastError() != cudaSuccess) rc = 1;
        for (void* p : {(void*) dx1, (void*) dx2, act1, act2, a4a, a4b, (void*) gs, (void*) out, (void*) out2, (void*) bounds, (void*) toff})
            cudaFree(p);
    }
    return rc;
}

}  // namespace
