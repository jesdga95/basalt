// tests/kernels/q6k_head_parity.cpp - q6k_head_mmvq against native_mmvq (Q6_K, the exact layout), bitwise, on random
// Q6_K rows and quantized random columns, 1..8 columns; then a microbench of both at the head's shape.  Last, the draft
// head read through a row map (rows of the whole head) against the same rows gathered into a copy, bitwise: the 2-bit
// copy, the two-stage argmax, the one-column head, and the copy's 2..8-column head against the row map one column at a
// time.
//   q6k_head_parity [rows] [iters]
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/q5k_head.hpp"
#include "basalt/kernels/q6k_head.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/kernels/verify_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <random>
#include <vector>

namespace k = basalt::kernels;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
}  // namespace

int main(int argc, char** argv) {
    const int rows = argc > 1 ? std::atoi(argv[1]) : 248320;
    const int iters = argc > 2 ? std::atoi(argv[2]) : 50;
    const int n_in = 2560, MT = 8, blk = 210, bpr = n_in / 256;
    const size_t wbytes = (size_t) rows * bpr * blk;
    std::mt19937 rng(5);
    std::vector<uint8_t> hw(wbytes);
    for (size_t i = 0; i < wbytes; i += 4) {
        const uint32_t r = rng();
        std::memcpy(&hw[i], &r, 4);
    }
    std::normal_distribution<float> nd(0.f, 1.f);
    for (size_t b = 0; b < (size_t) rows * bpr; ++b) {   // finite block scales: d as a small half
        const __half d = __float2half(std::fabs(nd(rng)) * 0.01f);
        std::memcpy(&hw[b * blk + 208], &d, 2);
    }
    void* w = nullptr;
    ck(cudaMalloc(&w, wbytes), "w");
    ck(cudaMemcpy(w, hw.data(), wbytes, cudaMemcpyHostToDevice), "w up");
    std::vector<float> hx((size_t) MT * n_in);
    for (auto& v : hx) v = nd(rng);
    float* x = nullptr;
    void* xq = nullptr;
    float *y0 = nullptr, *y1 = nullptr;
    ck(cudaMalloc(&x, hx.size() * 4), "x");
    ck(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "x up");
    ck(cudaMalloc(&xq, k::native_q8_1_bytes(n_in, MT)), "xq");
    ck(cudaMalloc(&y0, (size_t) MT * rows * 4), "y0");
    ck(cudaMalloc(&y1, (size_t) MT * rows * 4), "y1");
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    k::native_quantize_q8_1(x, xq, n_in, MT, s);
    int fail = 0;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int nc = 1; nc <= MT; ++nc) {
        ck(cudaMemset(y0, 0xff, (size_t) MT * rows * 4), "m0");
        ck(cudaMemset(y1, 0xee, (size_t) MT * rows * 4), "m1");
        k::native_mmvq(14, w, xq, y0, n_in, rows, nc, s);
        k::q6k_head_mmvq(w, xq, y1, n_in, rows, nc, s);
        ck(cudaStreamSynchronize(s), "sync");
        std::vector<float> a((size_t) nc * rows), b((size_t) nc * rows);
        ck(cudaMemcpy(a.data(), y0, a.size() * 4, cudaMemcpyDeviceToHost), "a");
        ck(cudaMemcpy(b.data(), y1, b.size() * 4, cudaMemcpyDeviceToHost), "b");
        size_t bad = 0;
        for (size_t i = 0; i < a.size(); ++i) bad += std::memcmp(&a[i], &b[i], 4) != 0;
        float ms[2] = {0, 0};
        for (int rep = 0; rep < 2; ++rep) {
            cudaEventRecord(e0, s);
            for (int it = 0; it < iters; ++it) k::native_mmvq(14, w, xq, y0, n_in, rows, nc, s);
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            cudaEventElapsedTime(&ms[0], e0, e1);
            cudaEventRecord(e0, s);
            for (int it = 0; it < iters; ++it) k::q6k_head_mmvq(w, xq, y1, n_in, rows, nc, s);
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            cudaEventElapsedTime(&ms[1], e0, e1);
        }
        const double u0 = 1e3 * ms[0] / iters, u1 = 1e3 * ms[1] / iters;
        std::printf("ncols %d: %s (%zu of %zu differ) | native_mmvq %7.1f us %6.0f GB/s | q6k_head %7.1f us %6.0f GB/s | %.2fx\n",
                    nc, bad ? "FAIL" : "bitwise", bad, a.size(), u0, wbytes / u0 / 1e3, u1, wbytes / u1 / 1e3, u0 / u1);
        if (bad) fail = 1;
    }
    // the fused argmax + probability (one column) against the logits' argmax and row_top_prob's probability
    {
        void* scratch = nullptr;
        int* d_id = nullptr;
        float* d_p = nullptr;
        int* r_id = nullptr;
        float* r_p = nullptr;
        ck(cudaMalloc(&scratch, k::q6k_head_top1_scratch_bytes()), "scratch");
        ck(cudaMemset(scratch, 0, k::q6k_head_top1_scratch_bytes()), "scratch0");
        ck(cudaMalloc(&d_id, 64), "id"); ck(cudaMalloc(&d_p, 64), "p"); ck(cudaMalloc(&r_id, 64), "rid"); ck(cudaMalloc(&r_p, 64), "rp");
        double max_rel = 0;
        int id_bad = 0, logit_bad = 0;
        const int n_rows[] = {rows, 106299, 40525, 1000};
        for (int rr : n_rows) {
            if (rr > rows) continue;
            for (int col = 0; col < MT; ++col) {
                const void* xc = (const char*) xq + (size_t) col * k::native_q8_1_bytes(n_in, 1);
                k::native_mmvq(14, w, xc, y0, n_in, rr, 1, s);
                if (!k::sample_greedy_top_prob(y0, 1, rr, nullptr, r_id, r_p, s)) { std::printf("no reference top-prob\n"); break; }
                if (!k::q6k_head_top1(w, xc, y1, n_in, rr, nullptr, d_id, d_p, scratch, s)) { std::printf("top1 refused\n"); fail = 1; break; }
                ck(cudaStreamSynchronize(s), "sync");
                int a_id, b_id;
                float a_p, b_p;
                ck(cudaMemcpy(&a_id, r_id, 4, cudaMemcpyDeviceToHost), "a"); ck(cudaMemcpy(&b_id, d_id, 4, cudaMemcpyDeviceToHost), "b");
                ck(cudaMemcpy(&a_p, r_p, 4, cudaMemcpyDeviceToHost), "c"); ck(cudaMemcpy(&b_p, d_p, 4, cudaMemcpyDeviceToHost), "d");
                std::vector<float> a((size_t) rr), b((size_t) rr);
                ck(cudaMemcpy(a.data(), y0, a.size() * 4, cudaMemcpyDeviceToHost), "la");
                ck(cudaMemcpy(b.data(), y1, b.size() * 4, cudaMemcpyDeviceToHost), "lb");
                for (size_t i = 0; i < a.size(); ++i) logit_bad += std::memcmp(&a[i], &b[i], 4) != 0;
                id_bad += a_id != b_id;
                max_rel = std::max(max_rel, (double) std::fabs(a_p - b_p) / std::max(1e-30, (double) std::fabs(a_p)));
            }
            float ms[2] = {0, 0};
            for (int rep = 0; rep < 2; ++rep) {
                cudaEventRecord(e0, s);
                for (int it = 0; it < iters; ++it) {
                    k::native_mmvq(14, w, xq, y0, n_in, rr, 1, s);
                    k::sample_greedy_top_prob(y0, 1, rr, nullptr, r_id, r_p, s);
                }
                cudaEventRecord(e1, s);
                cudaEventSynchronize(e1);
                cudaEventElapsedTime(&ms[0], e0, e1);
                cudaEventRecord(e0, s);
                for (int it = 0; it < iters; ++it) k::q6k_head_top1(w, xq, y1, n_in, rr, nullptr, d_id, d_p, scratch, s);
                cudaEventRecord(e1, s);
                cudaEventSynchronize(e1);
                cudaEventElapsedTime(&ms[1], e0, e1);
            }
            std::printf("top1 rows %6d: head + top-prob %7.1f us | fused %7.1f us\n", rr, 1e3 * ms[0] / iters, 1e3 * ms[1] / iters);
        }
        std::printf("top1: logits %s (%d differ), argmax %s (%d differ), probability max rel diff %.2e\n",
                    logit_bad ? "FAIL" : "bitwise", logit_bad, id_bad ? "FAIL" : "equal", id_bad, max_rel);
        if (logit_bad || id_bad || max_rel > 1e-5) fail = 1;
    }
    // rows without a token (negative id): the fused argmax skips them, and the mask sets their logits to -inf
    {
        void* scratch = nullptr;
        int* d_id = nullptr;
        float* d_p = nullptr;
        int32_t* d_map = nullptr;
        ck(cudaMalloc(&scratch, k::q6k_head_top1_scratch_bytes()), "scratch");
        ck(cudaMemset(scratch, 0, k::q6k_head_top1_scratch_bytes()), "scratch0");
        ck(cudaMalloc(&d_id, 64), "id"); ck(cudaMalloc(&d_p, 64), "p");
        const int rr = std::min(rows, 106299);
        std::vector<int32_t> map((size_t) rr);
        for (int i = 0; i < rr; ++i) map[(size_t) i] = (i % 5 == 2 || i >= rr - 2048) ? -1 : 3 * i + 7;
        ck(cudaMalloc(&d_map, map.size() * 4), "map");
        ck(cudaMemcpy(d_map, map.data(), map.size() * 4, cudaMemcpyHostToDevice), "map up");
        int bad = 0;
        double max_rel = 0;
        for (int col = 0; col < MT; ++col) {
            const void* xc = (const char*) xq + (size_t) col * k::native_q8_1_bytes(n_in, 1);
            k::native_mmvq(14, w, xc, y0, n_in, rr, 1, s);
            k::q6k_head_top1(w, xc, y1, n_in, rr, d_map, d_id, d_p, scratch, s);
            ck(cudaStreamSynchronize(s), "sync");
            std::vector<float> a((size_t) rr), b((size_t) rr);
            ck(cudaMemcpy(a.data(), y0, a.size() * 4, cudaMemcpyDeviceToHost), "la");
            ck(cudaMemcpy(b.data(), y1, b.size() * 4, cudaMemcpyDeviceToHost), "lb");
            int best = -1;
            for (int i = 0; i < rr; ++i) {
                if (map[(size_t) i] < 0) { bad += !(std::isinf(b[(size_t) i]) && b[(size_t) i] < 0); continue; }
                bad += std::memcmp(&a[(size_t) i], &b[(size_t) i], 4) != 0;
                if (best < 0 || a[(size_t) i] > a[(size_t) best]) best = i;
            }
            double sum = 0;
            for (int i = 0; i < rr; ++i)
                if (map[(size_t) i] >= 0) sum += std::exp((double) a[(size_t) i] - (double) a[(size_t) best]);
            int id;
            float p;
            ck(cudaMemcpy(&id, d_id, 4, cudaMemcpyDeviceToHost), "id");
            ck(cudaMemcpy(&p, d_p, 4, cudaMemcpyDeviceToHost), "p");
            bad += id != map[(size_t) best];
            max_rel = std::max(max_rel, std::fabs((double) p - 1.0 / sum) * sum);
            // the mask after the plain head
            k::q6k_head_mmvq(w, xc, y1, n_in, rr, 1, s);
            k::q5k_head_mask_unmapped(y1, d_map, rr, s);
            ck(cudaStreamSynchronize(s), "sync");
            ck(cudaMemcpy(b.data(), y1, b.size() * 4, cudaMemcpyDeviceToHost), "lb2");
            for (int i = 0; i < rr; ++i)
                bad += map[(size_t) i] < 0 ? !(std::isinf(b[(size_t) i]) && b[(size_t) i] < 0)
                                           : std::memcmp(&a[(size_t) i], &b[(size_t) i], 4) != 0;
        }
        float t_map = 0, t_plain = 0;
        for (int rep = 0; rep < 2; ++rep) {
            cudaEventRecord(e0, s);
            for (int it = 0; it < iters; ++it) k::q6k_head_top1(w, xq, y1, n_in, rr, d_map, d_id, d_p, scratch, s);
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            cudaEventElapsedTime(&t_map, e0, e1);
            cudaEventRecord(e0, s);
            for (int it = 0; it < iters; ++it) k::q6k_head_top1(w, xq, y1, n_in, rr, nullptr, d_id, d_p, scratch, s);
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            cudaEventElapsedTime(&t_plain, e0, e1);
        }
        std::printf("top1 with holes, rows %d: %s (%d bad), probability max rel diff %.2e | with map %.1f us, without %.1f us\n",
                    rr, bad ? "FAIL" : "bitwise", bad, max_rel, 1e3 * t_map / iters, 1e3 * t_plain / iters);
        if (bad || max_rel > 1e-5) fail = 1;
    }
    // the two-stage argmax (2-bit copy, then exact rescoring of the near-best rows) against the exact argmax, on rows
    // scaled to logits of a few units' spread; rows without a token are skipped.  The planted rows come from a CPU
    // dequantization of the Q6_K rows, so a wrong unpacking of the copy shows as misses.
    {
        const int rr = std::min(rows, 40525);
        const int64_t cap = rr + 100;
        k::native_mmvq(14, w, xq, y0, n_in, rr, 1, s);
        ck(cudaStreamSynchronize(s), "sync");
        std::vector<float> l0((size_t) rr);
        ck(cudaMemcpy(l0.data(), y0, l0.size() * 4, cudaMemcpyDeviceToHost), "l0");
        double m1 = 0, m2 = 0;
        for (float v : l0) { m1 += v; m2 += (double) v * v; }
        const double sd = std::sqrt(m2 / rr - (m1 / rr) * (m1 / rr));
        const float f = (float) (3.0 / sd);
        std::vector<uint8_t> hw2(hw.begin(), hw.begin() + (size_t) rr * bpr * blk);
        for (size_t b = 0; b < (size_t) rr * bpr; ++b) {   // scale d so the logits' spread is 3
            __half d;
            std::memcpy(&d, &hw2[b * blk + 208], 2);
            d = __float2half(__half2float(d) * f);
            std::memcpy(&hw2[b * blk + 208], &d, 2);
        }
        auto dequant = [&](int r, std::vector<float>& out) {   // ggml's Q6_K layout
            out.assign((size_t) n_in, 0.0f);
            for (int b = 0; b < bpr; ++b) {
                const uint8_t* bp = &hw2[((size_t) r * bpr + b) * blk];
                const uint8_t* ql = bp;
                const uint8_t* qh = bp + 128;
                const int8_t* sc = (const int8_t*) (bp + 192);
                __half dh;
                std::memcpy(&dh, bp + 208, 2);
                const float d = __half2float(dh);
                for (int n = 0; n < 2; ++n)
                    for (int l = 0; l < 32; ++l) {
                        const int is = l / 16;
                        const int q1 = ((ql[64 * n + l] & 0xF) | (((qh[32 * n + l] >> 0) & 3) << 4)) - 32;
                        const int q2 = ((ql[64 * n + l + 32] & 0xF) | (((qh[32 * n + l] >> 2) & 3) << 4)) - 32;
                        const int q3 = ((ql[64 * n + l] >> 4) | (((qh[32 * n + l] >> 4) & 3) << 4)) - 32;
                        const int q4 = ((ql[64 * n + l + 32] >> 4) | (((qh[32 * n + l] >> 6) & 3) << 4)) - 32;
                        float* y = &out[(size_t) b * 256 + 128 * n];
                        y[l] = d * sc[8 * n + is] * q1;
                        y[l + 32] = d * sc[8 * n + is + 2] * q2;
                        y[l + 64] = d * sc[8 * n + is + 4] * q3;
                        y[l + 96] = d * sc[8 * n + is + 6] * q4;
                    }
            }
        };
        void* w2 = nullptr;
        ck(cudaMalloc(&w2, hw2.size()), "w2");
        ck(cudaMemcpy(w2, hw2.data(), hw2.size(), cudaMemcpyHostToDevice), "w2 up");
        void *proxy = nullptr, *scratch = nullptr, *escratch = nullptr;
        int *d_id = nullptr, *d_map_i = nullptr;
        float* d_p = nullptr;
        ck(cudaMalloc(&proxy, k::q5k_proxy_bytes(cap)), "proxy");
        ck(cudaMemset(proxy, 0, k::q5k_proxy_bytes(cap)), "proxy0");
        ck(cudaMalloc(&scratch, k::q5k_proxy_scratch_bytes(cap)), "pscratch");
        ck(cudaMalloc(&escratch, k::q6k_head_top1_scratch_bytes()), "escratch");
        ck(cudaMemset(escratch, 0, k::q6k_head_top1_scratch_bytes()), "escratch0");
        ck(cudaMalloc(&d_id, 64), "id"); ck(cudaMalloc(&d_p, 64), "p");
        std::vector<int32_t> map((size_t) rr);
        for (int i = 0; i < rr; ++i) map[(size_t) i] = (i % 7 == 3 || i >= rr - 300) ? -1 : 5 * i + 1;
        ck(cudaMalloc(&d_map_i, map.size() * 4), "map");
        ck(cudaMemcpy(d_map_i, map.data(), map.size() * 4, cudaMemcpyHostToDevice), "map up");
        k::q6k_proxy_pack(w2, 0, rr, proxy, cap, s);
        const int trials = 400;
        int miss = 0;
        double max_p_rel = 0, sum_p_rel = 0;
        float* x1 = nullptr;
        void* xq1 = nullptr;
        ck(cudaMalloc(&x1, (size_t) n_in * 4), "x1");
        ck(cudaMalloc(&xq1, k::native_q8_1_bytes(n_in, 1)), "xq1");
        std::vector<float> dq;
        for (int tr = 0; tr < trials; ++tr) {
            // noise plus two planted rows (the pick and a runner-up a few units behind), as a head's logits have
            std::vector<float> xh(n_in);
            for (auto& v : xh) v = 0.8f * nd(rng);
            for (int pl = 0; pl < 2; ++pl) {
                int r;
                do { r = (int) (rng() % (unsigned) rr); } while (map[(size_t) r] < 0);
                dequant(r, dq);
                double nn = 0;
                for (float v : dq) nn += (double) v * v;
                const double alpha = (pl == 0 ? 20.0 : 16.0 + (double) (rng() % 4)) / nn;
                for (int i = 0; i < n_in; ++i) xh[(size_t) i] += (float) (alpha * dq[(size_t) i]);
            }
            ck(cudaMemcpy(x1, xh.data(), (size_t) n_in * 4, cudaMemcpyHostToDevice), "x1 up");
            k::native_quantize_q8_1(x1, xq1, n_in, 1, s);
            k::native_mmvq(14, w2, xq1, y0, n_in, rr, 1, s);
            if (!k::q6k_proxy_top1(proxy, w2, xq1, rr, cap, d_map_i, d_id, d_p, scratch, s)) { std::printf("proxy refused\n"); fail = 1; break; }
            ck(cudaStreamSynchronize(s), "sync");
            std::vector<float> a((size_t) rr);
            ck(cudaMemcpy(a.data(), y0, a.size() * 4, cudaMemcpyDeviceToHost), "a");
            int best = -1;
            for (int i = 0; i < rr; ++i)
                if (map[(size_t) i] >= 0 && (best < 0 || a[(size_t) i] > a[(size_t) best])) best = i;
            double sum = 0;
            for (int i = 0; i < rr; ++i)
                if (map[(size_t) i] >= 0) sum += std::exp((double) a[(size_t) i] - (double) a[(size_t) best]);
            int id;
            float p;
            ck(cudaMemcpy(&id, d_id, 4, cudaMemcpyDeviceToHost), "id");
            ck(cudaMemcpy(&p, d_p, 4, cudaMemcpyDeviceToHost), "p");
            const bool hit = id == map[(size_t) best];
            miss += !hit;
            if (hit) {   // the probability of the same token
                const double rel = std::fabs((double) p * sum - 1.0);
                max_p_rel = std::max(max_p_rel, rel);
                sum_p_rel += rel;
            }
        }
        // cold time: the two-stage argmax, the fused exact head, and the plain head + top-prob the draft ran before
        float t_px = 0, t_ex = 0, t_pl = 0;
        int *r_id = nullptr;
        float* r_p = nullptr;
        ck(cudaMalloc(&r_id, 64), "rid"); ck(cudaMalloc(&r_p, 64), "rp");
        void* flush = nullptr;
        ck(cudaMalloc(&flush, 512u << 20), "flush");
        for (int rep = 0; rep < 2; ++rep) {
            ck(cudaStreamSynchronize(s), "sync");
            float tot[3] = {0, 0, 0};
            for (int it = 0; it < 40; ++it)
                for (int v = 0; v < 3; ++v) {
                    ck(cudaMemsetAsync(flush, it, 512u << 20, s), "flush");
                    cudaEventRecord(e0, s);
                    if (v == 0) k::q6k_proxy_top1(proxy, w2, xq1, rr, cap, d_map_i, d_id, d_p, scratch, s);
                    else if (v == 1) k::q6k_head_top1(w2, xq1, y1, n_in, rr, d_map_i, d_id, d_p, escratch, s);
                    else {
                        k::native_mmvq(14, w2, xq1, y1, n_in, rr, 1, s);
                        k::sample_greedy_top_prob(y1, 1, rr, d_map_i, r_id, r_p, s);
                    }
                    cudaEventRecord(e1, s);
                    cudaEventSynchronize(e1);
                    float ms;
                    cudaEventElapsedTime(&ms, e0, e1);
                    tot[v] += ms;
                }
            t_px = tot[0] / 40;
            t_ex = tot[1] / 40;
            t_pl = tot[2] / 40;
        }
        cudaFree(flush);
        std::printf("proxy top1 rows %d: %d of %d picks differ from the exact argmax; probability rel diff max %.2e mean %.2e | "
                    "cold: two-stage %.1f us, fused exact %.1f us, head + top-prob %.1f us\n",
                    rr, miss, trials, max_p_rel, sum_p_rel / trials, 1e3 * t_px, 1e3 * t_ex, 1e3 * t_pl);
        if (miss > trials / 50 || max_p_rel > 1.0) fail = 1;
    }
    // the draft head through a row map: a subset of distinct rows in shuffled order (as a draft vocabulary's base rows
    // then its added rows), read from the whole head, against the same rows gathered into a copy
    {
        const int nsub = std::min(rows / 2, 106299), cap = nsub;
        std::vector<int32_t> perm((size_t) rows);
        for (int i = 0; i < rows; ++i) perm[(size_t) i] = i;
        std::shuffle(perm.begin(), perm.end(), rng);
        std::vector<int32_t> map(perm.begin(), perm.begin() + nsub);
        int32_t* d_map = nullptr;
        void *wg = nullptr, *pa = nullptr, *pb = nullptr, *scratch = nullptr, *xqm = nullptr;
        float *ya = nullptr, *yb = nullptr, *pr = nullptr;
        int* ids = nullptr;
        const size_t rb = (size_t) bpr * blk;
        ck(cudaMalloc(&d_map, map.size() * 4), "map");
        ck(cudaMemcpy(d_map, map.data(), map.size() * 4, cudaMemcpyHostToDevice), "map up");
        ck(cudaMalloc(&wg, (size_t) nsub * rb), "wg");
        ck(cudaMalloc(&pa, k::q5k_proxy_bytes(cap)), "pa");
        ck(cudaMalloc(&pb, k::q5k_proxy_bytes(cap)), "pb");
        ck(cudaMemset(pa, 0, k::q5k_proxy_bytes(cap)), "pa0");
        ck(cudaMemset(pb, 0, k::q5k_proxy_bytes(cap)), "pb0");
        ck(cudaMalloc(&scratch, k::q5k_proxy_scratch_bytes(cap)), "scratch");
        ck(cudaMalloc(&xqm, k::native_q8_1_bytes(n_in, MT)), "xqm");
        ck(cudaMalloc(&ya, (size_t) MT * nsub * 4), "ya");
        ck(cudaMalloc(&yb, (size_t) MT * nsub * 4), "yb");
        ck(cudaMalloc(&ids, 4 * 4), "ids");
        ck(cudaMalloc(&pr, 4 * 4), "pr");
        k::gather_rows((const uint8_t*) w, (int64_t) rb, d_map, nsub, (uint8_t*) wg, s);
        k::q6k_proxy_pack(wg, 0, nsub, pa, cap, s);
        k::q6k_proxy_pack(w, 0, nsub, pb, cap, s, d_map);
        ck(cudaStreamSynchronize(s), "pack");
        std::vector<uint8_t> ha(k::q5k_proxy_bytes(cap)), hb(ha.size());
        ck(cudaMemcpy(ha.data(), pa, ha.size(), cudaMemcpyDeviceToHost), "pa");
        ck(cudaMemcpy(hb.data(), pb, hb.size(), cudaMemcpyDeviceToHost), "pb");
        const bool pack_eq = std::memcmp(ha.data(), hb.data(), ha.size()) == 0;
        int top_bad = 0, one_bad = 0, multi_bad = 0;
        std::vector<float> a((size_t) MT * nsub), b(a.size());
        for (int tr = 0; tr < 24; ++tr) {
            std::vector<float> xh((size_t) MT * n_in);
            for (auto& v : xh) v = nd(rng);
            ck(cudaMemcpy(x, xh.data(), xh.size() * 4, cudaMemcpyHostToDevice), "x up");
            k::native_quantize_q8_1(x, xqm, n_in, MT, s);
            // the two-stage argmax on the copy's base rows
            const int nb = nsub * 2 / 5;
            k::q6k_proxy_top1(pa, wg, xqm, nb, cap, d_map, ids, pr, scratch, s);
            k::q6k_proxy_top1(pb, w, xqm, nb, cap, d_map, ids + 1, pr + 1, scratch, s, true);
            int hid[2];
            float hp[2];
            ck(cudaMemcpy(hid, ids, 8, cudaMemcpyDeviceToHost), "ids");
            ck(cudaMemcpy(hp, pr, 8, cudaMemcpyDeviceToHost), "pr");
            top_bad += hid[0] != hid[1] || std::memcmp(&hp[0], &hp[1], 4) != 0;
            // one column over every subset row
            k::native_mmvq(14, wg, xqm, ya, n_in, nsub, 1, s);
            k::native_q6_k_mmvq_rows(w, d_map, xqm, yb, n_in, nsub, s);
            ck(cudaMemcpy(a.data(), ya, (size_t) nsub * 4, cudaMemcpyDeviceToHost), "ya");
            ck(cudaMemcpy(b.data(), yb, (size_t) nsub * 4, cudaMemcpyDeviceToHost), "yb");
            one_bad += std::memcmp(a.data(), b.data(), (size_t) nsub * 4) != 0;
            // 2..8 columns: the copy's head over T columns against the row map one column at a time
            const int T = 2 + tr % 7;
            k::native_mmvq(14, wg, xqm, ya, n_in, nsub, T, s);
            for (int t = 0; t < T; ++t)
                k::native_q6_k_mmvq_rows(w, d_map, (const char*) xqm + (size_t) t * k::native_q8_1_bytes(n_in, 1),
                                         yb + (size_t) t * nsub, n_in, nsub, s);
            ck(cudaMemcpy(a.data(), ya, (size_t) T * nsub * 4, cudaMemcpyDeviceToHost), "ya");
            ck(cudaMemcpy(b.data(), yb, (size_t) T * nsub * 4, cudaMemcpyDeviceToHost), "yb");
            multi_bad += std::memcmp(a.data(), b.data(), (size_t) T * nsub * 4) != 0;
        }
        // the layout of a draft vocabulary: base rows in ascending id order, then added rows in ascending id order
        std::vector<int32_t> map2 = map;
        const int nb2 = nsub == 106299 ? 40525 : nsub * 2 / 5;
        std::sort(map2.begin(), map2.begin() + nb2);
        std::sort(map2.begin() + nb2, map2.end());
        int32_t* d_map2 = nullptr;
        ck(cudaMalloc(&d_map2, map2.size() * 4), "map2");
        ck(cudaMemcpy(d_map2, map2.data(), map2.size() * 4, cudaMemcpyHostToDevice), "map2 up");
        // one column, hot: the copy against the row map (shuffled, and in a draft vocabulary's order)
        float t_v[3] = {0, 0, 0};
        auto one = [&](int v) {
            if (v == 0) k::native_mmvq(14, wg, xqm, ya, n_in, nsub, 1, s);
            else k::native_q6_k_mmvq_rows(w, v == 1 ? d_map : d_map2, xqm, yb, n_in, nsub, s);
        };
        for (int v = 0; v < 3; ++v) {
            for (int i = 0; i < 5; ++i) one(v);
            cudaEventRecord(e0, s);
            for (int i = 0; i < iters; ++i) one(v);
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms;
            cudaEventElapsedTime(&ms, e0, e1);
            t_v[v] = 1e3f * ms / iters;
        }
        const double gb = (double) nsub * rb / 1e3;
        std::printf("draft head by row map, %d of %d rows: 2-bit copy %s, two-stage argmax %d of 24 differ, one column %d of 24 "
                    "differ, 2..8 columns %d of 24 differ | one column: copy %.1f us (%.0f GB/s), row map shuffled %.1f us "
                    "(%.0f GB/s), row map in draft vocabulary order (%d + %d ascending) %.1f us (%.0f GB/s)\n",
                    nsub, rows, pack_eq ? "bitwise" : "DIFFERS", top_bad, one_bad, multi_bad, t_v[0], gb / t_v[0], t_v[1],
                    gb / t_v[1], nb2, nsub - nb2, t_v[2], gb / t_v[2]);
        if (!pack_eq || top_bad || one_bad || multi_bad) fail = 1;
        for (void* p : {(void*) d_map, (void*) d_map2, wg, pa, pb, scratch, xqm, (void*) ya, (void*) yb, (void*) ids, (void*) pr})
            cudaFree(p);
    }
    std::printf("%s\n", fail ? "q6k_head_parity: FAIL" : "q6k_head_parity: OK (bitwise)");
    return fail;
}
