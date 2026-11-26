// tests/models/glm5next/kernels/bench.cpp - the glm5next kernels' own cost: microseconds per call and the GB/s over
// the bytes each call must move, at the model's geometry. Every kernel gets one line, decode-sized and, where the
// kernel has one, prompt-sized. The bytes counted are the buffers the call reads and writes; nothing is streamed from
// the tier here, so the MoE line counts only the weighted sum of the rows the tier already placed.
//
// Not a correctness test - the *_parity binaries are that - but the two run from the same fixtures, so a kernel that
// gets slower between two builds shows up next to the number that says it still answers the same.
#include "models/glm5next/kernels/cell_expand.hpp"
#include "models/glm5next/kernels/expert_sum.hpp"
#include "models/glm5next/kernels/head_transpose.hpp"
#include "models/glm5next/kernels/idx_score.hpp"
#include "models/glm5next/kernels/idx_select.hpp"
#include "models/glm5next/kernels/kda.hpp"
#include "models/glm5next/kernels/kda_front.hpp"
#include "models/glm5next/kernels/kda_norm.hpp"
#include "models/glm5next/kernels/kpool.hpp"
#include "models/glm5next/kernels/mla_sparse.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

namespace {

float* d_alloc(size_t bytes) {
    void* p = nullptr;
    const cudaError_t err = cudaMalloc(&p, bytes);
    if (err != cudaSuccess) {
        std::printf("FAIL: cudaMalloc %zu bytes: %s\n", bytes, cudaGetErrorString(err));
        std::exit(1);
    }
    return (float*) p;
}

/// `f` once to warm the path, then `iters` times between two events: the µs one call costs. A device error is
/// sticky and poisons everything after it, so a run that hits one stops where it is rather than print numbers
/// that are not measurements.
double microseconds_per_call(const std::function<void()>& f, int iters) {
    f();
    const cudaError_t warm = cudaDeviceSynchronize();
    if (warm != cudaSuccess) {
        std::printf("FAIL: the warm call runs: %s\n", cudaGetErrorString(warm));
        std::exit(1);
    }
    cudaEvent_t a, b;
    cudaError_t ev = cudaEventCreate(&a);
    if (ev == cudaSuccess) ev = cudaEventCreate(&b);
    if (ev != cudaSuccess) {
        std::printf("FAIL: events: %s\n", cudaGetErrorString(ev));
        std::exit(1);
    }
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) f();
    cudaEventRecord(b);
    const cudaError_t done = cudaEventSynchronize(b);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    if (done != cudaSuccess) {
        std::printf("FAIL: the timed calls run: %s\n", cudaGetErrorString(done));
        std::exit(1);
    }
    return (double) ms * 1000.0 / (double) iters;
}

void report(const char* name, const char* shape, double us, double bytes) {
    std::printf("%-16s %-22s %8.1f us/call  %7.1f GB/s  (%.2f MiB moved)\n", name, shape, us,
                bytes / (us * 1000.0), bytes / (1024.0 * 1024.0));
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
    bool have = false;
    for (int d = 0; d < n_gpus; ++d) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, d) != cudaSuccess) continue;
        if (prop.major == 12 && cudaSetDevice(d) == cudaSuccess) {
            have = true;
            break;
        }
    }
    if (!have) {
        std::printf("SKIP: no device the engine kernels are built for\n");
        return 77;
    }

    namespace g = basalt::models::glm5next;
    constexpr int kHeads = 64, kHd = 128, kC = kHeads * kHd;  // 8192 channels a KDA stream
    constexpr int kKv = 512, kKpool = 4;
    constexpr int kIdxHeads = 32, kIdxDim = 128;
    constexpr int kTopK = 2048;                 // the indexer selects this many cells
    constexpr int kTopPools = kTopK / kKpool;   // 512 pools
    constexpr int kPools = 8192;                // a 32k context holds this many
    constexpr int kCells = kPools * kKpool;
    constexpr int kE = 4096;                    // the model's width
    constexpr int kIters = 200, kItersBig = 20;

    // --- kda_step: one token of the recurrence over the whole state
    {
        const size_t state = (size_t) kHeads * kHd * kHd;
        float *s = d_alloc(state * 4), *q = d_alloc(kC * 4), *k = d_alloc(kC * 4), *v = d_alloc(kC * 4),
              *gg = d_alloc(kC * 4), *b = d_alloc(kHeads * 4), *o = d_alloc(kC * 4);
        const double us = microseconds_per_call(
            [&] { g::kda_step(s, q, k, v, gg, b, o, kHeads, kHd, kHd, 0); }, kIters);
        report("kda_step", "64x128x128, 1 tok", us,
               (2.0 * state + 4.0 * kC + kHeads) * 4.0);
        cudaFree(s); cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(gg); cudaFree(b); cudaFree(o);
    }

    // --- kda_front: the conv, the norms and the gates in front of the recurrence
    {
        float *x = d_alloc(3 * kC * 4), *hist = d_alloc(9 * kC * 4), *conv = d_alloc(12 * kC * 4),
              *a = d_alloc(kC * 4), *dt = d_alloc(kC * 4), *aneg = d_alloc(kHeads * 4), *braw = d_alloc(kHeads * 4),
              *qn = d_alloc(kC * 4), *kn = d_alloc(kC * 4), *vo = d_alloc(kC * 4), *gg = d_alloc(kC * 4),
              *be = d_alloc(kHeads * 4);
        const double us = microseconds_per_call(
            [&] { g::kda_front(x, kC, hist, conv, a, dt, aneg, braw, qn, kn, vo, gg, be, kHeads, kHd, 0); }, kIters);
        report("kda_front", "64x128, 1 tok", us, (3.0 * kC + 9.0 * kC + 12.0 * kC + 2.0 * kC + 4.0 * kC + 4 * kHeads) * 4.0);
        cudaFree(x); cudaFree(hist); cudaFree(conv); cudaFree(a); cudaFree(dt); cudaFree(aneg); cudaFree(braw);
        cudaFree(qn); cudaFree(kn); cudaFree(vo); cudaFree(gg); cudaFree(be);
    }

    // --- kda_norm: the gated output norm over the recurrence's rows
    {
        const int rows = kHeads;  // one token
        float *o = d_alloc((size_t) rows * kHd * 4), *z = d_alloc((size_t) rows * kHd * 4), *w = d_alloc(kHd * 4),
              *out = d_alloc((size_t) rows * kHd * 4);
        const double us = microseconds_per_call(
            [&] { g::kda_norm(o, z, w, out, rows, kHd, 1e-5f, 0); }, kIters);
        report("kda_norm", "64 rows x 128", us, (3.0 * rows * kHd + kHd) * 4.0);
        cudaFree(o); cudaFree(z); cudaFree(w); cudaFree(out);
    }

    // --- kpool_compress: one pool of the indexer's key history
    {
        const int n_pools = kPools;
        // gate is laid out like key, [pools][slots][dim]: the kernel reads it at the same place it reads the key
        float *key = d_alloc((size_t) n_pools * kKpool * kIdxDim * 4),
              *gate = d_alloc((size_t) n_pools * kKpool * kIdxDim * 4), *ape = d_alloc((size_t) kKpool * kIdxDim * 4),
              *out = d_alloc((size_t) n_pools * kIdxDim * 4);
        const double us = microseconds_per_call(
            [&] { g::kpool_compress(key, gate, ape, out, n_pools, kKpool, kIdxDim, 0); }, kItersBig);
        report("kpool_compress", "8192 pools x 4 x 128", us,
               (2.0 * n_pools * kKpool * kIdxDim + (double) kKpool * kIdxDim + (double) n_pools * kIdxDim) * 4.0);
        cudaFree(key); cudaFree(gate); cudaFree(ape); cudaFree(out);
    }

    // --- idx_score: every pool scored against one token
    {
        const int n_tokens = 1;
        float *iq = d_alloc((size_t) n_tokens * kIdxHeads * kIdxDim * 4), *w = d_alloc((size_t) n_tokens * kIdxHeads * 4),
              *pooled = d_alloc((size_t) kPools * kIdxDim * 4), *out = d_alloc((size_t) n_tokens * kPools * 4);
        int32_t *pos = (int32_t*) d_alloc(n_tokens * 4);
        cudaMemcpy(pos, &kCells, n_tokens * 4, cudaMemcpyHostToDevice);
        const float w_scale = 1.0f / std::sqrt((float) (kIdxDim * kIdxHeads));
        const double us = microseconds_per_call(
            [&] { g::idx_score(iq, w, pooled, pos, out, n_tokens, kPools, kIdxHeads, kIdxDim, kKpool, w_scale, 0); },
            kItersBig);
        report("idx_score", "1 tok x 8192 pools", us,
               ((double) n_tokens * kIdxHeads * kIdxDim + (double) kPools * kIdxDim + (double) n_tokens * kPools) * 4.0);
        cudaFree(iq); cudaFree(w); cudaFree(pooled); cudaFree(out); cudaFree(pos);
    }

    // --- idx_select: the top-K over those pools
    {
        const int n_tokens = 1;
        float* scores = d_alloc((size_t) n_tokens * kPools * 4);
        cudaMemset(scores, 0x3f, (size_t) n_tokens * kPools * 4);
        int32_t* ids = (int32_t*) d_alloc((size_t) n_tokens * kTopK * 4);
        int32_t* out_n = (int32_t*) d_alloc(n_tokens * 4);
        const double us = microseconds_per_call(
            [&] { g::idx_select(scores, ids, out_n, n_tokens, kPools, kTopK, 0); }, kIters);
        report("idx_select", "top-2048 of 8192", us, ((double) n_tokens * kPools + (double) n_tokens * kTopK) * 4.0);
        cudaFree(scores); cudaFree(ids); cudaFree(out_n);
    }

    // --- cell_expand: the selected pools into the cell ids the read gathers
    {
        const int n_tokens = 1;
        const int n_sel = g::cell_expand_n_sel(kKpool, kTopPools, true);
        int32_t* tp = (int32_t*) d_alloc((size_t) n_tokens * kTopPools * 4);
        int32_t* pi = (int32_t*) d_alloc((size_t) kKpool * kPools * 4);
        int32_t* ti = (int32_t*) d_alloc((size_t) (kKpool - 1) * n_tokens * 4);
        int32_t* ids = (int32_t*) d_alloc((size_t) n_tokens * n_sel * 4);
        cudaMemset(pi, 0, (size_t) kKpool * kPools * 4);
        cudaMemset(tp, 0, (size_t) n_tokens * kTopPools * 4);
        const double us = microseconds_per_call(
            [&] { g::cell_expand(tp, pi, ti, kCells, ids, n_tokens, kPools, kKpool, kTopPools, true, 0); }, kIters);
        report("cell_expand", "512 pools -> cells", us,
               ((double) n_tokens * kTopPools + (double) kKpool * kPools + (double) n_tokens * n_sel) * 4.0);
        cudaFree(tp); cudaFree(pi); cudaFree(ti); cudaFree(ids);
    }

    // --- mla_sparse: the sparse read itself, decode and a prompt window
    for (const int n_tokens : {1, 256}) {
        float* q_abs = d_alloc((size_t) n_tokens * kHeads * kKv * 4);
        float* latent = d_alloc((size_t) kCells * kKv * 4);
        float* out = d_alloc((size_t) n_tokens * kHeads * kKv * 4);
        int32_t* ids = (int32_t*) d_alloc((size_t) n_tokens * kTopK * 4);
        int32_t* cl = (int32_t*) d_alloc(n_tokens * 4);
        cudaMemset(ids, 0, (size_t) n_tokens * kTopK * 4);
        cudaMemset(cl, 0x0f, n_tokens * 4);
        const size_t ws_n = g::mla_sparse_workspace_floats(n_tokens, kHeads, kTopK);
        float* ws = d_alloc(ws_n * 4);
        const double us = microseconds_per_call(
            [&] {
                g::mla_sparse(q_abs, latent, ids, cl, out, 1.0f / std::sqrt(256.0f), n_tokens, kHeads, kTopK, ws, 0);
            },
            n_tokens == 1 ? kIters : kItersBig);
        report("mla_sparse", n_tokens == 1 ? "1 tok x 64 heads" : "256 tok x 64 heads", us,
               ((double) n_tokens * kHeads * kKv + (double) n_tokens * kTopK * kKv +
                (double) n_tokens * kHeads * kKv) * 4.0);
        cudaFree(q_abs); cudaFree(latent); cudaFree(out); cudaFree(ids); cudaFree(cl); cudaFree(ws);
    }

    // --- head_major / head_store: the layout the per-head products read
    {
        const int n_tok = 256, width = kKv;
        float* x = d_alloc((size_t) n_tok * kHeads * width * 4);
        float* xt = d_alloc((size_t) n_tok * kHeads * width * 4);
        const double us = microseconds_per_call(
            [&] { g::head_major(x, xt, n_tok, kHeads, width, 0); }, kItersBig);
        report("head_major", "256 tok x 64 x 512", us, 2.0 * (double) n_tok * kHeads * width * 4.0);
        const double us2 = microseconds_per_call(
            [&] { g::head_store(x, xt, n_tok, kHeads, 1, width, 0); }, kItersBig);
        report("head_store", "256 tok, 1 head x 512", us2, 2.0 * (double) n_tok * width * 4.0);
        cudaFree(x); cudaFree(xt);
    }

    // --- weighted_sum_block: the last launch of an expert block, over the rows the tier placed
    {
        const int n_tokens = 8, K = 8;
        g::RowPtrs rows;
        std::vector<float*> hs((size_t) n_tokens * K);
        for (auto& p : hs) {
            p = d_alloc((size_t) kE * 4);
            cudaMemset(p, 0, (size_t) kE * 4);
        }
        for (size_t e = 0; e < hs.size(); ++e) rows.p[e] = hs[e];
        float* w = d_alloc((size_t) n_tokens * K * 4);
        float* out = d_alloc((size_t) n_tokens * kE * 4);
        const double us = microseconds_per_call(
            [&] { g::weighted_sum_block(rows, w, n_tokens, K, kE, out, 0); }, kIters);
        report("weighted_sum", "8 tok x 8 experts x 4096", us,
               ((double) n_tokens * K * kE + (double) n_tokens * K + (double) n_tokens * kE) * 4.0);
        for (float* p : hs) cudaFree(p);
        cudaFree(w); cudaFree(out);
    }
    return 0;
}
