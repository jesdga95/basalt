// tests/kernels/q6k_vhead_parity.cpp - the two-stage verify head (q6k_vhead) against q6k_head_mmvq and a CPU reference,
// on random Q6_K rows scaled to a head's logit spread and quantized random columns, 1..8 columns:
//   - the copy: a CPU fit of the dequantized rows gives the same levels and scales (a few boundary levels may differ
//     under fast math);
//   - the rows near each column's top are exact (the bits of q6k_head_mmvq), every other row is the CPU score of the
//     copy, and the argmax is the exact one;
// then a cold microbench of both at the head's shape.
//   q6k_vhead_parity [rows] [iters]
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/q6k_head.hpp"

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

constexpr int kNin = 2560, kBpr = 10, kBlk = 210;

// ggml's Q6_K layout, one row
void dequant_row(const uint8_t* row, float* out) {
    for (int b = 0; b < kBpr; ++b) {
        const uint8_t* bp = row + (size_t) b * kBlk;
        const uint8_t *ql = bp, *qh = bp + 128;
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
                float* y = out + (size_t) b * 256 + 128 * n;
                y[l] = d * sc[8 * n + is] * q1;
                y[l + 32] = d * sc[8 * n + is + 2] * q2;
                y[l + 64] = d * sc[8 * n + is + 4] * q3;
                y[l + 96] = d * sc[8 * n + is + 6] * q4;
            }
    }
}

int level(float w, float sh) { return std::min(3, std::max(0, (int) std::floor(w / sh + 2.0f))); }

// head_proxy.cuh's fit_group<true> on the CPU: the levels and the stored scale
float fit_group(const float* w, int* q) {
    float amax = 1e-9f;
    for (int l = 0; l < 32; ++l) amax = std::max(amax, std::fabs(w[l]));
    float best_s = amax / 1.5f, best_e = INFINITY;
    for (int t = 0; t < 6; ++t) {
        const float s = amax / 1.5f * (1.0f - 0.1f * (float) t);
        const float sh = __half2float(__float2half(s));
        float e = 0.0f;
        for (int l = 0; l < 32; ++l) {
            const float r = ((float) level(w[l], sh) - 1.5f) * sh - w[l];
            e += r * r;
        }
        if (e < best_e) { best_e = e; best_s = s; }
    }
    const float sh = __half2float(__float2half(best_s));
    float ww = 0.0f, wr = 0.0f;
    for (int l = 0; l < 32; ++l) {
        q[l] = level(w[l], sh);
        ww += w[l] * w[l];
        wr += w[l] * ((float) q[l] - 1.5f) * sh;
    }
    return wr > 0.0f ? __half2float(__float2half(sh * std::sqrt(ww / wr))) : sh;
}

// The copy's level and scale of (row, group, l) as q6k_head.cu lays them out.
struct Copy {
    const std::vector<uint32_t>& w;
    const std::vector<__half>& s;
    int lvl(int64_t row, int g, int l) const {
        const size_t tile = (size_t) (row >> 4);
        const int rr = (int) (row & 15), gg = rr & 7, hi8 = rr >> 3, q = g / 4, kk = g % 4;
        const int half = l / 16, t = (l % 16) / 4, bb = l % 4;
        const uint32_t word = w[((tile * 20 + q) * 32 + gg * 4 + t) * 4 + kk];
        return (int) ((word >> (8 * bb + 2 * (2 * half + hi8))) & 3u);
    }
    float scale(int64_t row, int g) const {
        const size_t tile = (size_t) (row >> 4);
        const int rr = (int) (row & 15), gg = rr & 7, hi8 = rr >> 3, q = g / 4, kk = g % 4;
        return __half2float(s[(((tile * 20 + q) * 8 + gg) * 4 + kk) * 2 + hi8]);
    }
};
}  // namespace

int main(int argc, char** argv) {
    const int rows = argc > 1 ? std::atoi(argv[1]) / 16 * 16 : 248320;
    const int iters = argc > 2 ? std::atoi(argv[2]) : 40;
    const int MT = 8;
    const size_t wbytes = (size_t) rows * kBpr * kBlk;
    std::mt19937 rng(11);
    std::vector<uint8_t> hw(wbytes);
    for (size_t i = 0; i < wbytes; i += 4) {
        const uint32_t r = rng();
        std::memcpy(&hw[i], &r, 4);
    }
    std::normal_distribution<float> nd(0.f, 1.f);
    for (size_t b = 0; b < (size_t) rows * kBpr; ++b) {
        const __half d = __float2half(std::fabs(nd(rng)) * 0.01f);
        std::memcpy(&hw[b * kBlk + 208], &d, 2);
    }
    void *w = nullptr, *xq = nullptr, *proxy = nullptr, *scratch = nullptr;
    float *x = nullptr, *y0 = nullptr, *y1 = nullptr;
    ck(cudaMalloc(&w, wbytes), "w");
    ck(cudaMemcpy(w, hw.data(), wbytes, cudaMemcpyHostToDevice), "w up");
    std::vector<float> hx((size_t) MT * kNin);
    for (auto& v : hx) v = nd(rng);
    ck(cudaMalloc(&x, hx.size() * 4), "x");
    ck(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "x up");
    ck(cudaMalloc(&xq, k::native_q8_1_bytes(kNin, MT)), "xq");
    ck(cudaMalloc(&y0, (size_t) MT * rows * 4), "y0");
    ck(cudaMalloc(&y1, (size_t) MT * rows * 4), "y1");
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    k::native_quantize_q8_1(x, xq, kNin, MT, s);
    // scale the block scales so a column's logits spread like a head's (sd 4)
    {
        k::q6k_head_mmvq(w, xq, y0, kNin, rows, 1, s);
        std::vector<float> l0((size_t) rows);
        ck(cudaMemcpy(l0.data(), y0, l0.size() * 4, cudaMemcpyDeviceToHost), "l0");
        double m1 = 0, m2 = 0;
        for (float v : l0) { m1 += v; m2 += (double) v * v; }
        const float f = (float) (4.0 / std::sqrt(m2 / rows - (m1 / rows) * (m1 / rows)));
        for (size_t b = 0; b < (size_t) rows * kBpr; ++b) {
            __half d;
            std::memcpy(&d, &hw[b * kBlk + 208], 2);
            d = __float2half(__half2float(d) * f);
            std::memcpy(&hw[b * kBlk + 208], &d, 2);
        }
        ck(cudaMemcpy(w, hw.data(), wbytes, cudaMemcpyHostToDevice), "w up2");
    }
    const size_t pb = k::q6k_vproxy_bytes(rows);
    ck(cudaMalloc(&proxy, pb), "proxy");
    ck(cudaMalloc(&scratch, k::q6k_vhead_scratch_bytes()), "scratch");
    ck(cudaMemset(scratch, 0, k::q6k_vhead_scratch_bytes()), "scratch0");
    k::q6k_vproxy_pack(w, rows, proxy, s);
    ck(cudaStreamSynchronize(s), "pack");
    int fail = 0;

    // the copy against the CPU fit
    const size_t lv_words = (size_t) rows / 16 * 20 * 32 * 4;
    std::vector<uint32_t> hwq(lv_words);
    std::vector<__half> hsc((size_t) rows / 16 * 20 * 8 * 8);
    ck(cudaMemcpy(hwq.data(), proxy, lv_words * 4, cudaMemcpyDeviceToHost), "copy w");
    ck(cudaMemcpy(hsc.data(), (char*) proxy + lv_words * 4, hsc.size() * 2, cudaMemcpyDeviceToHost), "copy s");
    const Copy cp{hwq, hsc};
    {
        std::vector<float> dq(kNin);
        int q[32];
        long lv_bad = 0, sc_bad = 0, n_lv = 0;
        for (int r = 0; r < rows; r += std::max(1, rows / 3000)) {
            dequant_row(&hw[(size_t) r * kBpr * kBlk], dq.data());
            for (int g = 0; g < 80; ++g) {
                const float sc = fit_group(&dq[(size_t) g * 32], q);
                sc_bad += std::fabs(sc - cp.scale(r, g)) > 1e-3f * std::fabs(sc);
                for (int l = 0; l < 32; ++l) lv_bad += q[l] != cp.lvl(r, g, l);
                n_lv += 32;
            }
        }
        std::printf("copy: %ld of %ld levels and %ld scales differ from the CPU fit\n", lv_bad, n_lv, sc_bad);
        if (lv_bad > n_lv / 1000 || sc_bad > n_lv / 32 / 100) fail = 1;
    }

    // the CPU score of the copy for every row and column
    std::vector<int8_t> hq((size_t) MT * kNin);
    std::vector<float> hd((size_t) MT * 80);
    {
        std::vector<uint8_t> raw(k::native_q8_1_bytes(kNin, MT));
        ck(cudaMemcpy(raw.data(), xq, raw.size(), cudaMemcpyDeviceToHost), "xq down");
        for (int b = 0; b < MT * 80; ++b) {
            __half dh;
            std::memcpy(&dh, &raw[(size_t) b * 36], 2);
            hd[(size_t) b] = __half2float(dh);
            std::memcpy(&hq[(size_t) b * 32], &raw[(size_t) b * 36 + 4], 32);
        }
    }
    std::vector<float> approx((size_t) MT * rows);
    {
        std::vector<int> lv(kNin);
        std::vector<float> sc(80);
        std::vector<long> sq((size_t) MT * 80, 0);
        for (int c = 0; c < MT; ++c)
            for (int i = 0; i < kNin; ++i) sq[(size_t) c * 80 + i / 32] += hq[(size_t) c * kNin + i];
        for (int r = 0; r < rows; ++r) {
            for (int g = 0; g < 80; ++g) {
                sc[(size_t) g] = cp.scale(r, g);
                for (int l = 0; l < 32; ++l) lv[(size_t) g * 32 + l] = cp.lvl(r, g, l);
            }
            for (int c = 0; c < MT; ++c) {
                double acc = 0;
                const int8_t* qc = &hq[(size_t) c * kNin];
                for (int g = 0; g < 80; ++g) {
                    long p = 0;
                    for (int l = 0; l < 32; ++l) p += (long) lv[(size_t) g * 32 + l] * qc[g * 32 + l];
                    acc += (double) sc[(size_t) g] * hd[(size_t) c * 80 + g] * ((double) p - 1.5 * (double) sq[(size_t) c * 80 + g]);
                }
                approx[(size_t) c * rows + r] = (float) acc;
            }
        }
    }

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    void* flush = nullptr;
    ck(cudaMalloc(&flush, 256u << 20), "flush");
    std::vector<double> exact_rows(MT + 1, 0);
    std::vector<float> first;   // column 0 at one column: every width gives the same bits
    for (int nc = 1; nc <= MT; ++nc) {
        k::q6k_head_mmvq(w, xq, y0, kNin, rows, nc, s);
        k::q6k_vhead(proxy, w, xq, y1, kNin, rows, nc, scratch, s);
        ck(cudaStreamSynchronize(s), "sync");
        std::vector<float> a((size_t) nc * rows), b((size_t) nc * rows);
        ck(cudaMemcpy(a.data(), y0, a.size() * 4, cudaMemcpyDeviceToHost), "a");
        ck(cudaMemcpy(b.data(), y1, b.size() * 4, cudaMemcpyDeviceToHost), "b");
        if (nc == 1) first = b;
        const long width_bad = std::memcmp(first.data(), b.data(), (size_t) rows * 4) != 0;
        long missed = 0, off_cpu = 0, n_exact = 0, argmax_bad = 0;
        double max_dev = 0;
        for (int c = 0; c < nc; ++c) {
            const float* ap = &approx[(size_t) c * rows];
            std::vector<float> srt(ap, ap + rows);
            std::nth_element(srt.begin(), srt.begin() + 63, srt.end(), std::greater<float>());
            const float must = srt[63] - 3.99f;   // the kernel's margin, less the scoring's rounding
            int am = 0, bm = 0;
            for (int r = 0; r < rows; ++r) {
                const size_t i = (size_t) c * rows + r;
                const bool exact = std::memcmp(&a[i], &b[i], 4) == 0;
                n_exact += exact;
                if (ap[r] >= must && !exact) ++missed;
                if (!exact) {
                    const double dv = std::fabs((double) b[i] - (double) ap[r]);
                    max_dev = std::max(max_dev, dv);
                    off_cpu += dv > 2e-3 * (1.0 + std::fabs((double) ap[r]));
                }
                if (a[i] > a[(size_t) c * rows + am]) am = r;
                if (b[i] > b[(size_t) c * rows + bm]) bm = r;
            }
            argmax_bad += am != bm;
        }
        exact_rows[nc] = (double) n_exact / nc;
        // cold time: each launch after a flush of the L2
        float tot[2] = {0, 0};
        for (int it = 0; it < iters; ++it)
            for (int v = 0; v < 2; ++v) {
                ck(cudaMemsetAsync(flush, it, 256u << 20, s), "flush");
                cudaEventRecord(e0, s);
                if (v == 0) k::q6k_head_mmvq(w, xq, y0, kNin, rows, nc, s);
                else k::q6k_vhead(proxy, w, xq, y1, kNin, rows, nc, scratch, s);
                cudaEventRecord(e1, s);
                cudaEventSynchronize(e1);
                float ms;
                cudaEventElapsedTime(&ms, e0, e1);
                tot[v] += ms;
            }
        const double u0 = 1e3 * tot[0] / iters, u1 = 1e3 * tot[1] / iters;
        const bool bad = missed || off_cpu || argmax_bad || width_bad;
        std::printf("ncols %d: %s (near-top rows not exact %ld, approximate rows off the CPU score %ld, max dev %.2e, "
                    "argmax differs %ld, column 0 differs from one column %ld; exact rows per column %.0f) | cold: "
                    "q6k_head %6.1f us | vhead %6.1f us | %.2fx\n",
                    nc, bad ? "FAIL" : "ok", missed, off_cpu, max_dev, argmax_bad, width_bad, exact_rows[nc], u0, u1,
                    u0 / u1);
        if (bad) fail = 1;
    }
    std::printf("%s\n", fail ? "q6k_vhead_parity: FAIL" : "q6k_vhead_parity: OK");
    return fail;
}
