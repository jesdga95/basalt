// tests/kernels/q80_screen_parity.cpp - the screened argmax of a plain Q8_0 head (q80_screen.hpp) at the real width (3072),
// on a random head:
//   1. Planted logits (noise plus two rows' directions, the runner-up 0..4.5 units behind): the pick against the argmax of
//      native_mmvq's one-column logits (the head pass it replaces: native_mmvq + the greedy sampler) and against a CPU
//      reference (dequantize, accumulate in double); the pick's logit against native_mmvq's at that row, bitwise; the
//      same pick from a second call.
//   2. A tie: a row copied below the planted one; both the replaced path and the screen take the lower row.
//   3. Unplanted random activations (flat logits, every tile near the best): the overflow path; picks that differ from
//      the exact argmax are counted, the pick's logit is still native_mmvq's.
//   4. A head whose rows end inside a tile (4,113 rows): the rows past the end never win.
//   5. Cold microbench: the screen against native_mmvq + the greedy sampler, on a flat and on a planted activation.
//   q80_screen_parity [head rows] [trials]
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/q80_screen.hpp"
#include "basalt/kernels/sampler.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = basalt::kernels;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
constexpr int n_in = 3072, bpr = n_in / 32, blk = 34;
constexpr size_t rb = (size_t) bpr * blk;   // bytes per row

// row r of a Q8_0 head, dequantized
void dequant(const std::vector<uint8_t>& hw, int64_t r, std::vector<float>& out) {
    out.assign((size_t) n_in, 0.0f);
    for (int b = 0; b < bpr; ++b) {
        const uint8_t* bp = &hw[(size_t) r * rb + (size_t) b * blk];
        __half dh;
        std::memcpy(&dh, bp, 2);
        const float d = __half2float(dh);
        for (int l = 0; l < 32; ++l) out[(size_t) b * 32 + l] = d * (float) (int8_t) bp[2 + l];
    }
}
// the q8_1 activation as the kernels read it: per block {d as half, sum as half} then 32 codes
void dequant_x(const std::vector<uint8_t>& hx, std::vector<double>& out) {
    out.assign((size_t) n_in, 0.0);
    for (int b = 0; b < bpr; ++b) {
        const uint8_t* bp = &hx[(size_t) b * 36];
        __half dh;
        std::memcpy(&dh, bp, 2);
        const double d = __half2float(dh);
        for (int l = 0; l < 32; ++l) out[(size_t) b * 32 + l] = d * (double) (int8_t) bp[4 + l];
    }
}

struct Head {
    int rows = 0;
    std::vector<uint8_t> hw;
    void* w = nullptr;        // the head
    void* screen = nullptr;   // its copy
    void* scratch = nullptr;
};

Head make_head(int rows, std::mt19937& rng, cudaStream_t s) {
    std::normal_distribution<float> nd(0.f, 1.f);
    Head h;
    h.rows = rows;
    h.hw.resize((size_t) rows * rb);
    for (size_t i = 0; i < h.hw.size(); i += 4) {
        const uint32_t r = rng();
        std::memcpy(&h.hw[i], &r, std::min<size_t>(4, h.hw.size() - i));
    }
    for (size_t b = 0; b < (size_t) rows * bpr; ++b) {   // finite block scales, of a head's size
        const __half d = __float2half(std::fabs(nd(rng)) * 2e-4f + 1e-5f);
        std::memcpy(&h.hw[b * blk], &d, 2);
    }
    ck(cudaMalloc(&h.w, h.hw.size()), "w");
    ck(cudaMemcpy(h.w, h.hw.data(), h.hw.size(), cudaMemcpyHostToDevice), "w up");
    ck(cudaMalloc(&h.screen, k::q80_screen_bytes(n_in, rows)), "screen");
    ck(cudaMalloc(&h.scratch, k::q80_screen_scratch_bytes(rows)), "scratch");
    k::q80_screen_pack(h.w, n_in, rows, h.screen, s);
    ck(cudaStreamSynchronize(s), "pack");
    return h;
}

void free_head(Head& h) {
    cudaFree(h.w);
    cudaFree(h.screen);
    cudaFree(h.scratch);
}

struct Bufs {
    float* x = nullptr;
    void* xq = nullptr;
    float* y = nullptr;
    int32_t* id = nullptr;   // [0] the screen's pick, [1] its second call's, [2] the replaced path's
    float* lg = nullptr;     // the screen's pick logit
};

// The replaced path (native_mmvq's logits, the greedy sampler's pick) and the screen, on the activation in b.x.
void run_both(const Head& h, Bufs& b, cudaStream_t s) {
    k::native_quantize_q8_1(b.x, b.xq, n_in, 1, s);
    k::native_mmvq(8, h.w, b.xq, b.y, n_in, h.rows, 1, s);
    k::SamplerParams sp;
    sp.greedy = true;
    sp.temperature = 0.f;
    k::sample_tokens(b.y, 1, h.rows, nullptr, 0, sp, b.id + 2, s);
    k::q80_screen_top1(h.screen, h.w, b.xq, n_in, h.rows, h.scratch, b.id, s, b.lg);
    k::q80_screen_top1(h.screen, h.w, b.xq, n_in, h.rows, h.scratch, b.id + 1, s);
}

// Activation: noise of `noise` (in units of the head's logit spread) plus the directions of `planted` rows, the first
// 20 units up and the others `behind[i]` units below it.
void plant(const Head& h, const std::vector<int>& planted, const std::vector<double>& behind, float xs, float noise,
           std::mt19937& rng, std::vector<float>& hxf) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> dq;
    for (auto& v : hxf) v = noise * xs * nd(rng);
    for (size_t pl = 0; pl < planted.size(); ++pl) {
        dequant(h.hw, planted[pl], dq);
        double nn = 0;
        for (float v : dq) nn += (double) v * v;
        const double alpha = (20.0 - behind[pl]) / nn;
        for (int i = 0; i < n_in; ++i) hxf[(size_t) i] += (float) (alpha * dq[(size_t) i]);
    }
}
}  // namespace

int main(int argc, char** argv) {
    const int rows = argc > 1 ? std::atoi(argv[1]) : 120832;
    const int trials = argc > 2 ? std::atoi(argv[2]) : 300;
    if (!k::q80_screen_supported(n_in)) { std::printf("no screen at %d\n", n_in); return 1; }
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    Head h = make_head(rows, rng, s);
    Bufs b;
    ck(cudaMalloc(&b.x, (size_t) n_in * 4), "x");
    ck(cudaMalloc(&b.xq, k::native_q8_1_bytes(n_in, 1)), "xq");
    ck(cudaMalloc(&b.y, (size_t) rows * 4), "y");
    ck(cudaMalloc(&b.id, 16), "id");
    ck(cudaMalloc(&b.lg, 16), "lg");
    int fail = 0;
    std::vector<float> hxf((size_t) n_in), a((size_t) rows), dq;
    std::vector<uint8_t> hxq(k::native_q8_1_bytes(n_in, 1));
    std::vector<double> xd;

    // the head's spread on random input, to scale the planted rows
    for (auto& v : hxf) v = nd(rng);
    ck(cudaMemcpy(b.x, hxf.data(), hxf.size() * 4, cudaMemcpyHostToDevice), "x up");
    run_both(h, b, s);
    ck(cudaMemcpy(a.data(), b.y, a.size() * 4, cudaMemcpyDeviceToHost), "a");
    double m1 = 0, m2 = 0;
    for (float v : a) { m1 += v; m2 += (double) v * v; }
    const double sd = std::sqrt(m2 / rows - (m1 / rows) * (m1 / rows));
    const float xs = (float) (3.0 / sd);

    // 1. planted logits
    {
        int miss = 0, miss_cpu = 0, ties = 0, bits = 0, again = 0;
        double max_abs = 0, max_rel = 0;
        for (int tr = 0; tr < trials; ++tr) {
            const int r0 = (int) (rng() % (unsigned) rows), r1 = (int) (rng() % (unsigned) rows);
            plant(h, {r0, r1}, {0.0, (double) (rng() % 5) + 0.5 * (rng() % 2)}, xs, 0.8f, rng, hxf);
            ck(cudaMemcpy(b.x, hxf.data(), hxf.size() * 4, cudaMemcpyHostToDevice), "x up");
            run_both(h, b, s);
            ck(cudaStreamSynchronize(s), "sync");
            int32_t id[3];
            float lg;
            ck(cudaMemcpy(id, b.id, 12, cudaMemcpyDeviceToHost), "id");
            ck(cudaMemcpy(&lg, b.lg, 4, cudaMemcpyDeviceToHost), "lg");
            ck(cudaMemcpy(a.data(), b.y, a.size() * 4, cudaMemcpyDeviceToHost), "a");
            ck(cudaMemcpy(hxq.data(), b.xq, hxq.size(), cudaMemcpyDeviceToHost), "xq");
            miss += id[0] != id[2];
            again += id[0] != id[1];
            bits += std::memcmp(&lg, &a[(size_t) id[0]], 4) != 0;
            // the CPU reference over the rows near the best, and its error on them
            dequant_x(hxq, xd);
            int cbest = -1;
            double cv = -1e300, second = -1e300;
            for (int i = 0; i < rows; ++i) {
                if (a[(size_t) i] < a[(size_t) id[2]] - 1.0f) continue;
                dequant(h.hw, i, dq);
                double ref = 0, mag = 0;
                for (int j = 0; j < n_in; ++j) { ref += (double) dq[(size_t) j] * xd[(size_t) j]; mag += std::fabs((double) dq[(size_t) j] * xd[(size_t) j]); }
                const double e = std::fabs((double) a[(size_t) i] - ref);
                max_abs = std::max(max_abs, e);
                max_rel = std::max(max_rel, e / std::max(1e-30, mag));
                if (ref > cv) { second = cv; cv = ref; cbest = i; }
                else second = std::max(second, ref);
            }
            ties += cv - second < 1e-5;
            miss_cpu += cbest >= 0 && id[0] != cbest;
        }
        std::printf("planted, %d trials over %d rows: picks differing from the replaced path's %d, from the CPU reference's "
                    "%d (near-ties %d); pick logit vs native_mmvq: %s (%d differ); a second call: %s; native_mmvq vs CPU on "
                    "the near-best rows max abs %.3e, max rel (to the sum of |terms|) %.3e\n",
                    trials, rows, miss, miss_cpu, ties, bits ? "FAIL" : "bitwise", bits, again ? "DIFFERS" : "same", max_abs, max_rel);
        if (miss > trials / 100 || bits || again || max_rel > 1e-5) fail = 1;
    }
    // 2. a tie: row lo (below hi) holds hi's bytes; the planted row is hi
    {
        const int lo = 1000, hi = rows - 77;
        std::memcpy(&h.hw[(size_t) lo * rb], &h.hw[(size_t) hi * rb], rb);
        ck(cudaMemcpy((uint8_t*) h.w + (size_t) lo * rb, &h.hw[(size_t) hi * rb], rb, cudaMemcpyHostToDevice), "tie row");
        k::q80_screen_pack(h.w, n_in, rows, h.screen, s);
        plant(h, {hi}, {0.0}, xs, 0.5f, rng, hxf);
        ck(cudaMemcpy(b.x, hxf.data(), hxf.size() * 4, cudaMemcpyHostToDevice), "x up");
        run_both(h, b, s);
        int32_t id[3];
        ck(cudaMemcpy(id, b.id, 12, cudaMemcpyDeviceToHost), "id");
        const bool ok = id[0] == lo && id[2] == lo;
        std::printf("tie between rows %d and %d: screen %d, replaced path %d: %s\n", lo, hi, id[0], id[2], ok ? "ok" : "FAIL");
        if (!ok) fail = 1;
    }
    // 3. unplanted random activations
    {
        int miss = 0, bits = 0, again = 0;
        const int n3 = std::max(1, trials / 10);
        for (int tr = 0; tr < n3; ++tr) {
            for (auto& v : hxf) v = nd(rng);
            ck(cudaMemcpy(b.x, hxf.data(), hxf.size() * 4, cudaMemcpyHostToDevice), "x up");
            run_both(h, b, s);
            ck(cudaStreamSynchronize(s), "sync");
            int32_t id[3];
            float lg;
            ck(cudaMemcpy(id, b.id, 12, cudaMemcpyDeviceToHost), "id");
            ck(cudaMemcpy(&lg, b.lg, 4, cudaMemcpyDeviceToHost), "lg");
            ck(cudaMemcpy(a.data(), b.y, a.size() * 4, cudaMemcpyDeviceToHost), "a");
            miss += id[0] != id[2];
            again += id[0] != id[1];
            bits += std::memcmp(&lg, &a[(size_t) id[0]], 4) != 0;
        }
        std::printf("unplanted, %d trials (flat logits, the overflow path): picks differing from the replaced path's %d; "
                    "pick logit vs native_mmvq: %s; a second call: %s\n", n3, miss, bits ? "FAIL" : "bitwise", again ? "DIFFERS" : "same");
        if (bits || again) fail = 1;
    }
    // 4. a head that ends inside a tile
    {
        Head t = make_head(4113, rng, s);
        int miss = 0, bits = 0, past = 0;
        for (int tr = 0; tr < 40; ++tr) {
            const int r0 = 4096 + (int) (rng() % 17u);   // the last tile's rows
            plant(t, {r0}, {0.0}, xs, 0.8f, rng, hxf);
            ck(cudaMemcpy(b.x, hxf.data(), hxf.size() * 4, cudaMemcpyHostToDevice), "x up");
            run_both(t, b, s);
            int32_t id[3];
            float lg;
            ck(cudaMemcpy(id, b.id, 12, cudaMemcpyDeviceToHost), "id");
            ck(cudaMemcpy(&lg, b.lg, 4, cudaMemcpyDeviceToHost), "lg");
            ck(cudaMemcpy(a.data(), b.y, (size_t) t.rows * 4, cudaMemcpyDeviceToHost), "a");
            miss += id[0] != id[2];
            past += id[0] >= t.rows;
            bits += id[0] < t.rows && std::memcmp(&lg, &a[(size_t) id[0]], 4) != 0;
        }
        std::printf("4,113 rows, planted in the last tile: picks differing %d, past the end %d, pick logit %s\n", miss, past,
                    bits ? "FAIL" : "bitwise");
        if (miss > 1 || past || bits) fail = 1;
        free_head(t);
    }
    // 5. cold microbench: the screen against native_mmvq + the greedy sampler, on a flat activation (every tile near the
    // best: the overflow path) and on a planted one (a few rows near the best)
    {
        void* flush = nullptr;
        ck(cudaMalloc(&flush, 512u << 20), "flush");
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        k::SamplerParams sp;
        sp.greedy = true;
        sp.temperature = 0.f;
        const double mb_sc = (double) k::q80_screen_bytes(n_in, rows) / 1e6, mb_hd = (double) rows * rb / 1e6;
        for (int planted = 0; planted < 2; ++planted) {
            if (planted) plant(h, {(int) (rng() % (unsigned) rows)}, {0.0}, xs, 0.8f, rng, hxf);
            else for (auto& v : hxf) v = nd(rng);
            ck(cudaMemcpy(b.x, hxf.data(), hxf.size() * 4, cudaMemcpyHostToDevice), "x up");
            k::native_quantize_q8_1(b.x, b.xq, n_in, 1, s);
            float tot[2] = {0, 0};
            const int reps = 60;
            for (int it = 0; it < reps + 4; ++it)
                for (int v = 0; v < 2; ++v) {
                    ck(cudaMemsetAsync(flush, it, 512u << 20, s), "flush");
                    cudaEventRecord(e0, s);
                    if (v == 0) k::q80_screen_top1(h.screen, h.w, b.xq, n_in, rows, h.scratch, b.id, s);
                    else {
                        k::native_mmvq(8, h.w, b.xq, b.y, n_in, rows, 1, s);
                        k::sample_tokens(b.y, 1, rows, nullptr, 0, sp, b.id + 2, s);
                    }
                    cudaEventRecord(e1, s);
                    cudaEventSynchronize(e1);
                    float ms;
                    cudaEventElapsedTime(&ms, e0, e1);
                    if (it >= 4) tot[v] += ms;
                }
            const double u0 = 1e3 * tot[0] / reps, u1 = 1e3 * tot[1] / reps;
            std::printf("cold, %d rows of %d, %s activation: screen %.1f us (copy %.1f MB, %.0f GB/s) | native_mmvq + greedy "
                        "sampler %.1f us (%.1f MB, %.0f GB/s) | %.2fx\n",
                        rows, n_in, planted ? "planted" : "flat", u0, mb_sc, mb_sc * 1e3 / u0, u1, mb_hd, mb_hd * 1e3 / u1, u1 / u0);
        }
        cudaFree(flush);
    }
    ck(cudaGetLastError(), "end");
    free_head(h);
    std::printf("%s\n", fail ? "q80_screen_parity: FAIL" : "q80_screen_parity: OK");
    return fail;
}
