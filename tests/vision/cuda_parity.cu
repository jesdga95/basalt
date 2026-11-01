// tests/vision/cuda_parity.cu - the encoder's kernels against host references, a microbench line for each, and the
// whole CUDA encoder against the CPU encoder on an image.
//
//   vision_cuda_parity kernels
//   vision_cuda_parity encoder <mmproj.gguf> <image>
//
// kernels: random inputs at the real geometry (1152 columns, 16 heads of 72); each result against a double
// precision host computation of the same operation, bitwise repeatability, and a row's independence of the rows
// around it; then a timing line per kernel.  encoder: per tap relative L2 against the CPU encoder run with the same
// bf16 input rounding, and the end-to-end time.  Exit status 1 when a gate fails.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include "basalt/vision/encoder_cpu.hpp"
#include "basalt/vision/encoder_cuda.hpp"
#include "vision_kernels.cuh"

using namespace basalt::vision;

namespace {

constexpr int E = 1152, H = 16, D = 72, DP = 80;
int g_fail = 0;
std::mt19937 rng(7);

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

template <class T>
T* dev(const std::vector<T>& h) {
    T* d;
    CHECK(cudaMalloc(&d, h.size() * sizeof(T)));
    CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <class T>
T* dev_empty(size_t n) {
    T* d;
    CHECK(cudaMalloc(&d, n * sizeof(T)));
    CHECK(cudaMemset(d, 0, n * sizeof(T)));
    return d;
}
template <class T>
std::vector<T> host(const T* d, size_t n) {
    std::vector<T> h(n);
    CHECK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

std::vector<float> randn(size_t n, float sd) {
    std::normal_distribution<float> nd(0.f, sd);
    std::vector<float> v(n);
    for (auto& x : v) x = nd(rng);
    return v;
}

double rel_l2(const std::vector<double>& a, const std::vector<double>& ref) {
    double d = 0, r = 0;
    for (size_t i = 0; i < a.size(); ++i) { d += (a[i] - ref[i]) * (a[i] - ref[i]); r += ref[i] * ref[i]; }
    return std::sqrt(d / std::max(r, 1e-300));
}

void verdict(const char* what, double err, double gate) {
    const bool ok = err <= gate;
    std::printf("  %-34s rel L2 %.3e (gate %.0e)%s\n", what, err, gate, ok ? "" : "  FAIL");
    if (!ok) ++g_fail;
}

// ms per call of fn over `iters` after a warm-up
template <class F>
double time_ms(F fn, int iters = 20) {
    fn();
    CHECK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) fn();
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    return ms / iters;
}

__half to_half(float f) { return __float2half_rn(f); }

void test_attention() {
    std::printf("attention (fp16 q,k,v, 16 heads x 72, full, bf16 out)\n");
    const float scale_log2 = (1.0f / std::sqrt((float) D)) * 1.4426950408889634f;
    for (int n : {1, 5, 64, 100, 1000, 4096}) {
        std::vector<__half> q((size_t) H * n * DP, to_half(0)), k = q, v = q;
        std::vector<double> qd((size_t) H * n * D), kd = qd, vd = qd;
        const std::vector<float> r = randn(3 * (size_t) H * n * D, 1.5f);
        for (int h = 0; h < H; ++h)
            for (int t = 0; t < n; ++t)
                for (int d = 0; d < D; ++d) {
                    const size_t s = ((size_t) h * n + t) * D + d, p = ((size_t) h * n + t) * DP + d;
                    q[p] = to_half(r[s]); k[p] = to_half(r[(size_t) H * n * D + s]); v[p] = to_half(r[2 * (size_t) H * n * D + s]);
                    qd[s] = __half2float(q[p]); kd[s] = __half2float(k[p]); vd[s] = __half2float(v[p]);
                }
        __half *dq = dev(q), *dk = dev(k), *dv = dev(v);
        __nv_bfloat16* dout = dev_empty<__nv_bfloat16>((size_t) n * E);
        launch_attention(dq, dk, dv, dout, n, scale_log2, 0);
        CHECK(cudaDeviceSynchronize());
        const auto o1 = host(dout, (size_t) n * E);
        launch_attention(dq, dk, dv, dout, n, scale_log2, 0);
        CHECK(cudaDeviceSynchronize());
        const bool same = std::memcmp(o1.data(), host(dout, (size_t) n * E).data(), o1.size() * 2) == 0;
        // host reference on a sample of rows (all rows up to 64, else 64 spread rows)
        std::vector<double> got, ref;
        const int rows = std::min(n, 64);
        for (int ri = 0; ri < rows; ++ri) {
            const int t = n <= 64 ? ri : (int) ((int64_t) ri * (n - 1) / (rows - 1));
            for (int h = 0; h < H; ++h) {
                std::vector<double> sc(n);
                double mx = -1e300;
                for (int u = 0; u < n; ++u) {
                    double s = 0;
                    for (int d = 0; d < D; ++d) s += qd[((size_t) h * n + t) * D + d] * kd[((size_t) h * n + u) * D + d];
                    sc[u] = s / std::sqrt((double) D);
                    mx = std::max(mx, sc[u]);
                }
                double sum = 0;
                for (int u = 0; u < n; ++u) { sc[u] = std::exp(sc[u] - mx); sum += sc[u]; }
                for (int d = 0; d < D; ++d) {
                    double a = 0;
                    for (int u = 0; u < n; ++u) a += sc[u] * vd[((size_t) h * n + u) * D + d];
                    ref.push_back(a / sum);
                    got.push_back(__bfloat162float(o1[(size_t) t * E + h * D + d]));
                }
            }
        }
        char name[64];
        std::snprintf(name, sizeof name, "n=%d%s", n, same ? "" : " (not repeatable)");
        verdict(name, rel_l2(got, ref), 4e-3);
        if (!same) ++g_fail;
        cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout);
    }
    for (int n : {1024, 4096, 16384}) {
        std::vector<__half> z((size_t) H * n * DP, to_half(0.5f));
        __half *dq = dev(z), *dk = dev(z), *dv = dev(z);
        __nv_bfloat16* dout = dev_empty<__nv_bfloat16>((size_t) n * E);
        const double ms = time_ms([&] { launch_attention(dq, dk, dv, dout, n, scale_log2, 0); }, 5);
        const double flops = 4.0 * n * (double) n * D * H;
        std::printf("  bench attention n=%-6d %9.1f us/call  %6.1f TFLOP/s (16 heads x 72)\n", n, ms * 1e3, flops / (ms * 1e-3) / 1e12);
        cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout);
    }
}

void test_layernorm() {
    std::printf("layernorm (fp32 rows of 1152 -> bf16)\n");
    const int n = 37;
    std::vector<float> x = randn((size_t) n * E, 3.f), w = randn(E, 1.f), b = randn(E, 0.5f);
    for (auto& v : w) v += 1.f;
    float *dx = dev(x), *dw = dev(w), *db = dev(b);
    __nv_bfloat16* dy = dev_empty<__nv_bfloat16>((size_t) n * E);
    launch_layernorm(dx, dw, db, dy, n, 1e-6f, 0);
    CHECK(cudaDeviceSynchronize());
    const auto y = host(dy, (size_t) n * E);
    std::vector<double> got, ref;
    for (int t = 0; t < n; ++t) {
        double m = 0, q = 0;
        for (int c = 0; c < E; ++c) m += x[(size_t) t * E + c];
        m /= E;
        for (int c = 0; c < E; ++c) q += (x[(size_t) t * E + c] - m) * (x[(size_t) t * E + c] - m);
        const double inv = 1.0 / std::sqrt(q / E + 1e-6);
        for (int c = 0; c < E; ++c) {
            ref.push_back((x[(size_t) t * E + c] - m) * inv * w[c] + b[c]);
            got.push_back(__bfloat162float(y[(size_t) t * E + c]));
        }
    }
    verdict("37 rows", rel_l2(got, ref), 4e-3);
    // a row alone gives the same bits as inside the batch
    __nv_bfloat16* d1 = dev_empty<__nv_bfloat16>(E);
    launch_layernorm(dx + (size_t) 20 * E, dw, db, d1, 1, 1e-6f, 0);
    CHECK(cudaDeviceSynchronize());
    const bool same = std::memcmp(host(d1, E).data(), &y[(size_t) 20 * E], E * 2) == 0;
    std::printf("  row 20 alone equals row 20 in the batch: %s\n", same ? "yes" : "NO");
    if (!same) ++g_fail;
    const int big = 16384;
    std::vector<float> xb((size_t) big * E, 1.f);
    float* dxb = dev(xb);
    __nv_bfloat16* dyb = dev_empty<__nv_bfloat16>((size_t) big * E);
    const double ms = time_ms([&] { launch_layernorm(dxb, dw, db, dyb, big, 1e-6f, 0); });
    const double gb = (double) big * E * (4 + 2) / 1e9;
    std::printf("  bench layernorm n=%d %9.1f us/call  %6.0f GB/s\n", big, ms * 1e3, gb / (ms * 1e-3));
    cudaFree(dxb); cudaFree(dyb); cudaFree(d1); cudaFree(dy); cudaFree(dx); cudaFree(dw); cudaFree(db);
}

void test_embed_rope_qkv() {
    std::printf("embed finish, rope table, qkv epilogue\n");
    for (auto grid : {std::pair<int, int>{8, 6}, {48, 48}, {30, 40}}) {
        const int gx = grid.first, gy = grid.second, n = gx * gy;
        std::vector<float> pos = randn(48 * 48 * (size_t) E, 1.f), bias = randn(E, 1.f), x0 = randn((size_t) n * E, 1.f);
        float *dp = dev(pos), *db = dev(bias), *dx = dev(x0);
        launch_embed_finish(dx, db, dp, n, gx, gy, 0);
        float* tab = dev_empty<float>((size_t) n * 72);
        launch_rope_table(tab, n, gx, 0);
        CHECK(cudaDeviceSynchronize());
        const auto x = host(dx, (size_t) n * E);
        const auto tb = host(tab, (size_t) n * 72);
        std::vector<double> got, ref, gt, rt;
        auto taps = [&](int out_n, int i, int& i0, int& i1, double& d) {
            float sf = (float) out_n / 48;
            if (out_n > 1) sf = (float) (out_n - 1) / 47;
            const float v = (float) i / sf;
            int a = (int) std::floor(v), b2 = a + 1;
            a = std::max(0, std::min(a, 47)); b2 = std::max(0, std::min(b2, 47));
            d = std::max(0.0f, std::min(v - (float) a, 1.0f));
            i0 = a; i1 = b2;
        };
        for (int t = 0; t < n; t += std::max(1, n / 50)) {
            const int g = t / 4, r = t % 4, per = gx / 2;
            const int py = (g / per) * 2 + r / 2, px = (g % per) * 2 + r % 2;
            int xa, xb, ya, yb;
            double dx_, dy_;
            taps(gx, px, xa, xb, dx_);
            taps(gy, py, ya, yb, dy_);
            for (int c = 0; c < E; ++c) {
                const double pe = pos[((size_t) ya * 48 + xa) * E + c] * (1 - dx_) * (1 - dy_) + pos[((size_t) ya * 48 + xb) * E + c] * dx_ * (1 - dy_) +
                                  pos[((size_t) yb * 48 + xa) * E + c] * (1 - dx_) * dy_ + pos[((size_t) yb * 48 + xb) * E + c] * dx_ * dy_;
                ref.push_back(x0[(size_t) t * E + c] + bias[c] + pe);
                got.push_back(x[(size_t) t * E + c]);
            }
            float th = (float) py;
            for (int j = 0; j < 36; ++j) {
                if (j == 18) th = (float) px;
                rt.push_back(std::cos((double) th)); rt.push_back(std::sin((double) th));
                gt.push_back(tb[((size_t) t * 36 + j) * 2]); gt.push_back(tb[((size_t) t * 36 + j) * 2 + 1]);
                th *= std::pow(10000.0f, -2.0f / 36);
            }
        }
        char name[64];
        std::snprintf(name, sizeof name, "embed %dx%d", gx, gy);
        verdict(name, rel_l2(got, ref), 1e-6);
        std::snprintf(name, sizeof name, "rope table %dx%d", gx, gy);
        verdict(name, rel_l2(gt, rt), 1e-6);
        // qkv epilogue
        std::vector<float> gm = randn((size_t) n * 3 * E, 1.f), bq = randn(3 * E, 0.3f);
        float *dg = dev(gm), *dbq = dev(bq);
        __half *q = dev_empty<__half>((size_t) H * n * DP), *k = dev_empty<__half>((size_t) H * n * DP), *v = dev_empty<__half>((size_t) H * n * DP);
        launch_qkv_epilogue(dg, dbq, tab, q, k, v, n, 0);
        CHECK(cudaDeviceSynchronize());
        const auto qh = host(q, (size_t) H * n * DP), kh = host(k, (size_t) H * n * DP), vh = host(v, (size_t) H * n * DP);
        std::vector<double> g2, r2;
        bool pad_zero = true;
        for (int t = 0; t < n; t += std::max(1, n / 20))
            for (int h = 0; h < H; ++h) {
                for (int d = 72; d < 80; ++d) pad_zero &= __half2float(qh[((size_t) h * n + t) * DP + d]) == 0.f && __half2float(vh[((size_t) h * n + t) * DP + d]) == 0.f;
                for (int j = 0; j < 36; ++j) {
                    const double c = tb[((size_t) t * 36 + j) * 2], s = tb[((size_t) t * 36 + j) * 2 + 1];
                    for (int part = 0; part < 2; ++part) {
                        const int base = part * E + h * D;
                        const double a = gm[(size_t) t * 3 * E + base + j] + bq[base + j], b = gm[(size_t) t * 3 * E + base + j + 36] + bq[base + j + 36];
                        const auto& hv = part == 0 ? qh : kh;
                        r2.push_back(a * c - b * s); g2.push_back(__half2float(hv[((size_t) h * n + t) * DP + j]));
                        r2.push_back(a * s + b * c); g2.push_back(__half2float(hv[((size_t) h * n + t) * DP + j + 36]));
                    }
                }
                for (int d = 0; d < D; ++d) {
                    r2.push_back(gm[(size_t) t * 3 * E + 2 * E + h * D + d] + bq[2 * E + h * D + d]);
                    g2.push_back(__half2float(vh[((size_t) h * n + t) * DP + d]));
                }
            }
        std::snprintf(name, sizeof name, "qkv epilogue %dx%d", gx, gy);
        verdict(name, rel_l2(g2, r2), 1e-3);
        if (!pad_zero) { std::printf("  padding columns not zero  FAIL\n"); ++g_fail; }
        if (gx == 48) {
            const double ms = time_ms([&] { launch_qkv_epilogue(dg, dbq, tab, q, k, v, n, 0); });
            std::printf("  bench qkv epilogue n=%d %9.1f us/call  %6.0f GB/s\n", n, ms * 1e3, (double) n * 3 * E * (4 + 2) / 1e9 / (ms * 1e-3));
        }
        cudaFree(dp); cudaFree(db); cudaFree(dx); cudaFree(tab); cudaFree(dg); cudaFree(dbq); cudaFree(q); cudaFree(k); cudaFree(v);
    }
}

void test_elementwise() {
    std::printf("bias + residual, bias + gelu\n");
    const int n = 301, cols = 4304;
    std::vector<float> g = randn((size_t) n * cols, 2.f), b = randn(cols, 1.f), x0 = randn((size_t) n * cols, 1.f);
    float *dg = dev(g), *db = dev(b), *dx = dev(x0);
    __nv_bfloat16* dy = dev_empty<__nv_bfloat16>((size_t) n * cols);
    launch_bias_residual(dx, dg, db, n, cols, 0);
    launch_bias_gelu(dg, db, dy, n, cols, 0);
    CHECK(cudaDeviceSynchronize());
    const auto x = host(dx, (size_t) n * cols);
    const auto y = host(dy, (size_t) n * cols);
    std::vector<double> a1, r1, a2, r2;
    for (size_t i = 0; i < x.size(); ++i) {
        a1.push_back(x[i]); r1.push_back((double) x0[i] + g[i] + b[i % cols]);
        const double v = g[i] + b[i % cols];
        r2.push_back(0.5 * v * (1.0 + std::tanh(0.7978845608028654 * v * (1.0 + 0.044715 * v * v))));
        a2.push_back(__bfloat162float(y[i]));
    }
    verdict("bias + residual", rel_l2(a1, r1), 1e-6);
    verdict("bias + gelu (bf16 out)", rel_l2(a2, r2), 4e-3);
    const int big = 16384;
    std::vector<float> gb((size_t) big * cols, 1.f);
    float* dgb = dev(gb);
    __nv_bfloat16* dyb = dev_empty<__nv_bfloat16>((size_t) big * cols);
    const double ms = time_ms([&] { launch_bias_gelu(dgb, db, dyb, big, cols, 0); });
    std::printf("  bench bias+gelu n=%d x %d %9.1f us/call  %6.0f GB/s\n", big, cols, ms * 1e3, (double) big * cols * 6 / 1e9 / (ms * 1e-3));
}

int run_encoder(const char* mmproj, const char* image) {
    std::string err;
    Mmproj mp;
    if (!mp.open(mmproj, err)) { std::printf("%s\n", err.c_str()); return 2; }
    std::ifstream f(image, std::ios::binary);
    const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    Image img;
    if (!decode_image(bytes.data(), bytes.size(), img, err)) { std::printf("%s\n", err.c_str()); return 2; }
    const Geometry geo;
    const Size sz = fit_size(img.w, img.h, geo);
    std::vector<float> planar, patches;
    normalize_planar(to_canvas(img, sz), planar);
    patchify(planar, sz, geo, patches);
    const int tokens = sz.w / 32 * (sz.h / 32);
    std::printf("encoder: %dx%d -> %d tokens\n", img.w, img.h, tokens);
    CpuEncoder cpu;
    if (!cpu.init(mp, 4, true, false, err)) { std::printf("%s\n", err.c_str()); return 2; }
    std::vector<std::pair<std::string, std::vector<float>>> ref;
    std::vector<float> ref_out;
    if (!cpu.encode(patches, sz, ref_out, [&](const std::string& n, const float* d, size_t c) { ref.push_back({n, std::vector<float>(d, d + c)}); })) return 2;
    CudaEncoder gpu;
    if (!gpu.init(mp, std::max(tokens, 8), err)) { std::printf("cuda init: %s\n", err.c_str()); return 2; }
    std::printf("  device memory: %.0f MiB\n", gpu.device_bytes() / 1048576.0);
    std::vector<float> out;
    size_t k = 0;
    auto tap = [&](const std::string& n, const float* d, size_t c) {
        if (k < ref.size() && ref[k].first == n && ref[k].second.size() == c) {
            std::vector<double> a(d, d + c), r(ref[k].second.begin(), ref[k].second.end());
            // after the first layers a bf16 rounding flip spreads; on a photo the reference's own runs differ by 1e-1 at the output
            const double gate = n == "embed" ? 1e-6 : (n == "post_ln" || n == "out") ? 1e-1 : 3e-2;
            char name[64];
            std::snprintf(name, sizeof name, "%s", n.c_str());
            if (n == "embed" || n.rfind("layer_out.", 0) == 0 && (n == "layer_out.0" || n == "layer_out.13" || n == "layer_out.26") || n == "post_ln" || n == "out")
                verdict(name, rel_l2(a, r), gate);
            else if (rel_l2(a, r) > gate) { verdict(name, rel_l2(a, r), gate); }
            ++k;
        } else {
            // the CPU taps include layer-0 FFN taps and the like that the GPU does not emit; skip to the match
            while (k < ref.size() && ref[k].first != n) ++k;
            if (k < ref.size()) { std::vector<double> a(d, d + c), r(ref[k].second.begin(), ref[k].second.end()); verdict(n.c_str(), rel_l2(a, r), 3e-2); ++k; }
        }
    };
    if (!gpu.encode(patches, sz, out, tap)) { std::printf("encode: %s\n", gpu.last_error().c_str()); return 2; }
    std::vector<double> a(out.begin(), out.end()), r(ref_out.begin(), ref_out.end());
    verdict("final embeddings vs CPU", rel_l2(a, r), 1e-1);
    double mn = 1;
    for (int t = 0; t < tokens; ++t) {
        double d = 0, na = 0, nb = 0;
        for (int c = 0; c < 2560; ++c) { d += (double) out[(size_t) t * 2560 + c] * ref_out[(size_t) t * 2560 + c]; na += (double) out[(size_t) t * 2560 + c] * out[(size_t) t * 2560 + c]; nb += (double) ref_out[(size_t) t * 2560 + c] * ref_out[(size_t) t * 2560 + c]; }
        mn = std::min(mn, d / std::sqrt(na * nb));
    }
    std::printf("  min row cosine %.5f\n", mn);
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 5; ++i) gpu.encode(patches, sz, out);
    std::printf("  encode (warm) %.1f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 5);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && !std::strcmp(argv[1], "kernels")) {
        test_layernorm();
        test_embed_rope_qkv();
        test_elementwise();
        test_attention();
    } else if (argc >= 4 && !std::strcmp(argv[1], "encoder")) {
        run_encoder(argv[2], argv[3]);
    } else {
        std::fprintf(stderr, "usage: vision_cuda_parity kernels | encoder <mmproj.gguf> <image>\n");
        return 2;
    }
    std::printf(g_fail ? "FAILED (%d)\n" : "all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
