// tests/models/glm5next/kernels/head_transpose_parity.cpp - the two layouts the per-head MLA products read their
// activations in: a token-major [token][head][width] block against its head-major [head][token][width] copy, and one
// head's [token][width] block stored back into a token-major row. Both have an exact answer, so every float is
// compared bit for bit, and a token read alone must give the same bytes as the same token inside a window.
#include "models/glm5next/kernels/head_transpose.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++g_failures;
    }
}

struct Lcg {
    unsigned long long s = 88172645463325252ull;
    float operator()() {  // [-1, 1)
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        const unsigned long long r = s >> 11;
        return (float) (double) r / 9007199254740992.0f * 2.0f - 1.0f;
    }
};

float* d_alloc(size_t bytes) {
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) {
        std::printf("FAIL: cudaMalloc\n");
        std::exit(1);
    }
    return (float*) p;
}

}  // namespace

int main() {
    // the engine kernels are built for the engine's own architecture: take the first card that runs them, and skip
    // where there is none rather than fail a build that has no such card attached
    int n_gpus = 0;
    if (cudaGetDeviceCount(&n_gpus) != cudaSuccess) {
        std::printf("SKIP: no CUDA device\n");
        return 77;
    }
    bool have_device = false;
    for (int d = 0; d < n_gpus; ++d) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, d) != cudaSuccess) continue;
        if (prop.major == 12) {
            if (cudaSetDevice(d) != cudaSuccess) continue;
            have_device = true;
            break;
        }
    }
    if (!have_device) {
        std::printf("SKIP: no device the engine kernels are built for\n");
        return 77;
    }

    // The model's head widths (kv_lora 512, v_dim 256) and a narrower one, over an odd token count so a stride that
    // only holds at one window width cannot pass, and a single token as the degenerate case.
    struct Geom { int n_tok, n_head, width; };
    const Geom geoms[] = {{5, 4, 512}, {1, 4, 256}, {7, 8, 128}};
    for (const Geom& g : geoms) {
        const size_t n = (size_t) g.n_tok * g.n_head * g.width;
        Lcg rng;
        std::vector<float> x(n), want(n, 0.f);
        for (auto& v : x) v = rng();

        float* dx = d_alloc(n * sizeof(float));
        float* dxt = d_alloc(n * sizeof(float));
        cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        basalt::models::glm5next::head_major(dx, dxt, g.n_tok, g.n_head, g.width, 0);
        check(cudaGetLastError() == cudaSuccess, "head_major launches");
        check(cudaDeviceSynchronize() == cudaSuccess, "head_major runs");
        std::vector<float> got(n);
        cudaMemcpy(got.data(), dxt, n * sizeof(float), cudaMemcpyDeviceToHost);

        for (int t = 0; t < g.n_tok; ++t)
            for (int h = 0; h < g.n_head; ++h)
                for (int d = 0; d < g.width; ++d)
                    want[((size_t) h * g.n_tok + t) * g.width + d] = x[((size_t) t * g.n_head + h) * g.width + d];
        check(std::memcmp(got.data(), want.data(), n * sizeof(float)) == 0, "head_major is the exact transpose");

        // The same token read alone: a window's row must equal what the token gives on its own, at every head.
        const size_t row = (size_t) g.n_head * g.width;
        float* d_one = d_alloc(row * sizeof(float));
        float* d_one_t = d_alloc(row * sizeof(float));
        bool same_alone = true;
        for (int t = 0; t < g.n_tok; ++t) {
            cudaMemcpy(d_one, dx + (size_t) t * row, row * sizeof(float), cudaMemcpyDeviceToDevice);
            basalt::models::glm5next::head_major(d_one, d_one_t, 1, g.n_head, g.width, 0);
            if (cudaDeviceSynchronize() != cudaSuccess) {
                same_alone = false;
                break;
            }
            std::vector<float> one(row);
            cudaMemcpy(one.data(), d_one_t, row * sizeof(float), cudaMemcpyDeviceToHost);
            for (int h = 0; h < g.n_head && same_alone; ++h)
                if (std::memcmp(one.data() + (size_t) h * g.width,
                                got.data() + ((size_t) h * g.n_tok + t) * g.width,
                                g.width * sizeof(float)) != 0)
                    same_alone = false;
        }
        check(same_alone, "one token alone matches its row in the window");
        cudaFree(d_one);
        cudaFree(d_one_t);

        // head_store: one head's contiguous block back into its slice of a token-major row. The other heads' slices
        // stay as they were, which is why the destination starts at a recognisable value rather than zero.
        const float sentinel = -1234.5f;
        std::vector<float> y(n, sentinel);
        float* dy = d_alloc(n * sizeof(float));
        cudaMemcpy(dy, y.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        const size_t src_n = (size_t) g.n_tok * g.width;
        std::vector<float> src(src_n);
        for (auto& v : src) v = rng();
        float* dsrc = d_alloc(src_n * sizeof(float));
        cudaMemcpy(dsrc, src.data(), src_n * sizeof(float), cudaMemcpyHostToDevice);
        const int head = g.n_head > 1 ? 1 : 0;
        basalt::models::glm5next::head_store(dsrc, dy, g.n_tok, g.n_head, head, g.width, 0);
        check(cudaGetLastError() == cudaSuccess, "head_store launches");
        check(cudaDeviceSynchronize() == cudaSuccess, "head_store runs");
        cudaMemcpy(y.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost);
        bool stored = true, untouched = true;
        for (int t = 0; t < g.n_tok; ++t) {
            for (int d = 0; d < g.width; ++d)
                if (y[((size_t) t * g.n_head + head) * g.width + d] != src[(size_t) t * g.width + d]) stored = false;
            for (int h = 0; h < g.n_head; ++h) {
                if (h == head) continue;
                for (int d = 0; d < g.width; ++d)
                    if (y[((size_t) t * g.n_head + h) * g.width + d] != sentinel) untouched = false;
            }
        }
        check(stored, "head_store writes the head's block where the row reads it");
        check(untouched, "head_store leaves the other heads' slices alone");

        cudaFree(dx);
        cudaFree(dxt);
        cudaFree(dsrc);
        cudaFree(dy);
    }

    if (g_failures != 0) {
        std::printf("HEAD TRANSPOSE PARITY: %d FAILURES\n", g_failures);
        return 1;
    }
    std::printf("HEAD TRANSPOSE PARITY: ok\n");
    return 0;
}
