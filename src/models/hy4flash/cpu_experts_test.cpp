// src/models/hy4flash/cpu_experts_test.cpp - the CPU experts against a double-precision reference over the dequantised
// weights, with the Q4_K gate/up and Q6_K or Q4_K down matrices of the test pack, at the model's width.
#include "basalt/ggml/ggml.h"
#include "models/hy4flash/cpu_experts.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace basalt::models::hy4flash;

int main() {
    if (ggml_get_type_traits_cpu(GGML_TYPE_Q4_K) == nullptr) {
        std::printf("no CPU dot products on this machine\n");
        return 77;
    }
    const int n = 1536, K = 5;
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.f, 1.f);
    int failures = 0;
    for (int d_type : {(int) GGML_TYPE_Q6_K, (int) GGML_TYPE_Q4_K}) {
        const size_t gu_row = ggml_row_size(GGML_TYPE_Q4_K, n), d_row = ggml_row_size((ggml_type) d_type, n);
        std::vector<std::vector<uint8_t>> blobs(K);
        std::vector<std::vector<float>> deq(K);   // gate, up, down dequantised
        for (int e = 0; e < K; ++e) {
            blobs[e].resize((size_t) n * (2 * gu_row + d_row));
            deq[e].resize((size_t) 3 * n * n);
            for (int m = 0; m < 3; ++m) {
                std::vector<float> w((size_t) n * n);
                for (auto& v : w) v = nd(rng) * 0.03f;
                const ggml_type t = m < 2 ? GGML_TYPE_Q4_K : (ggml_type) d_type;
                const size_t row = m < 2 ? gu_row : d_row, off = m < 2 ? (size_t) m * n * gu_row : (size_t) 2 * n * gu_row;
                ggml_quantize_chunk(t, w.data(), blobs[e].data() + off, 0, n, n);
                for (int r = 0; r < n; ++r)
                    ggml_get_type_traits(t)->to_float(blobs[e].data() + off + (size_t) r * row, deq[e].data() + ((size_t) m * n + r) * n, n);
            }
        }
        std::vector<float> x(n), w(K);
        for (auto& v : x) v = nd(rng);
        for (int e = 0; e < K; ++e) w[e] = 0.1f + 0.2f * e;
        std::vector<ExpertBlob> ex(K);
        for (int e = 0; e < K; ++e) ex[e] = {blobs[e].data(), (int) GGML_TYPE_Q4_K, d_type};
        std::vector<float> out(n), again(n);
        Workers pool(4);
        const float limit = 10.f;
        cpu_experts(pool, ex.data(), w.data(), K, n, limit, x.data(), out.data());
        cpu_experts(pool, ex.data(), w.data(), K, n, limit, x.data(), again.data());
        std::vector<double> ref(n, 0.0);
        for (int e = 0; e < K; ++e) {
            std::vector<double> h(n);
            for (int r = 0; r < n; ++r) {
                double g = 0, u = 0;
                for (int i = 0; i < n; ++i) {
                    g += (double) deq[e][(size_t) r * n + i] * x[i];
                    u += (double) deq[e][((size_t) n + r) * n + i] * x[i];
                }
                g = std::min(g, (double) limit);
                u = std::min(std::max(u, -(double) limit), (double) limit);
                h[r] = g / (1.0 + std::exp(-g)) * u;
            }
            for (int r = 0; r < n; ++r) {
                double d = 0;
                for (int i = 0; i < n; ++i) d += (double) deq[e][((size_t) 2 * n + r) * n + i] * h[i];
                ref[r] += w[e] * d;
            }
        }
        double max_abs = 0, max_ref = 0;
        for (int r = 0; r < n; ++r) {
            max_abs = std::max(max_abs, std::fabs(out[r] - ref[r]));
            max_ref = std::max(max_ref, std::fabs(ref[r]));
        }
        const bool same = out == again;
        const bool ok = max_abs / max_ref < 3e-2 && same;   // the activations are quantised to Q8_K, as ggml does
        std::printf("down %s: max abs %.3e rel %.3e, repeat %s  %s\n", ggml_type_name((ggml_type) d_type), max_abs, max_abs / max_ref,
                    same ? "bitwise" : "DIFFERS", ok ? "ok" : "FAIL");
        failures += !ok;
    }
    return failures ? 1 : 0;
}
