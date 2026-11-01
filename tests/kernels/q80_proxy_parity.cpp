// tests/kernels/q80_proxy_parity.cpp - the Q8_0 draft head read through a row map, and its two-stage argmax.  The
// q80_* kernels read the head's bit planes (q80_vhead.hpp): the rows and the gathered copy are packed for them.
//   1. q80_mmvq_rows over a subset of a head's rows (shuffled, with rows that hold no token) against native_mmvq and
//      native_q8_0_mmvq_stream on the same rows gathered into a copy, bitwise, and against a CPU reference (double).
//   2. The 2-bit copy packed through the row map against the copy packed from the gathered rows, bitwise.
//   3. q80_proxy_top1 against the exact argmax (native_mmvq, and the CPU reference) on planted logits, through the
//      row map and from the gathered copy: picks that differ, the probability's error against the exact softmax.
//   4. Cold microbench at the draft head's shape: q80_proxy_top1 against the streamed head + mask + top-prob it
//      replaces.
//   q80_proxy_parity [head rows] [trials]
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/q5k_head.hpp"
#include "basalt/kernels/q80_proxy.hpp"
#include "basalt/kernels/q80_vhead.hpp"
#include "basalt/kernels/sampler.hpp"
#include "basalt/kernels/verify_kernels.hpp"

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
constexpr int n_in = 2560, bpr = n_in / 32, blk = 34;
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
// the q8_1 activations as the kernels read them: per block {d as half, sum as half} then 32 codes
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
}  // namespace

int main(int argc, char** argv) {
    const int rows = argc > 1 ? std::atoi(argv[1]) : 248320;
    const int trials = argc > 2 ? std::atoi(argv[2]) : 400;
    const int nbase = std::min(rows / 2, 40525), nctx = 512, nsub = nbase + nctx;
    std::mt19937 rng(8);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint8_t> hw((size_t) rows * rb);
    for (size_t i = 0; i < hw.size(); i += 4) {
        const uint32_t r = rng();
        std::memcpy(&hw[i], &r, std::min<size_t>(4, hw.size() - i));
    }
    for (size_t b = 0; b < (size_t) rows * bpr; ++b) {   // finite block scales, of a head's size
        const __half d = __float2half(std::fabs(nd(rng)) * 2e-4f + 1e-5f);
        std::memcpy(&hw[b * blk], &d, 2);
    }
    void* w = nullptr;
    ck(cudaMalloc(&w, hw.size()), "w");
    ck(cudaMemcpy(w, hw.data(), hw.size(), cudaMemcpyHostToDevice), "w up");
    // the subset: distinct rows in shuffled order, the context rows' tail holding no token (id -1) except a few
    std::vector<int32_t> perm((size_t) rows);
    for (int i = 0; i < rows; ++i) perm[(size_t) i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<int32_t> map(perm.begin(), perm.begin() + nsub);
    for (int i = nbase + 37; i < nsub; ++i) map[(size_t) i] = -1;
    for (int i = 0; i < nbase; i += 97) map[(size_t) i] = -1;   // holes among the base rows too
    std::vector<int32_t> gmap = map;   // the copy's rows: a hole gathers row 0
    for (int32_t& v : gmap) if (v < 0) v = 0;
    int32_t *d_map = nullptr, *d_gmap = nullptr;
    ck(cudaMalloc(&d_map, map.size() * 4), "map");
    ck(cudaMalloc(&d_gmap, gmap.size() * 4), "gmap");
    ck(cudaMemcpy(d_map, map.data(), map.size() * 4, cudaMemcpyHostToDevice), "map up");
    ck(cudaMemcpy(d_gmap, gmap.data(), gmap.size() * 4, cudaMemcpyHostToDevice), "gmap up");
    void* wg = nullptr;   // the gathered copy
    ck(cudaMalloc(&wg, (size_t) nsub * rb), "wg");
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    k::gather_rows((const uint8_t*) w, (int64_t) rb, d_gmap, nsub, (uint8_t*) wg, s);
    void *wp = nullptr, *wgp = nullptr;   // the head and the copy in planes
    ck(cudaMalloc(&wp, hw.size()), "wp");
    ck(cudaMalloc(&wgp, (size_t) nsub * rb), "wgp");
    ck(cudaMemcpyAsync(wp, w, hw.size(), cudaMemcpyDeviceToDevice, s), "wp copy");
    ck(cudaMemcpyAsync(wgp, wg, (size_t) nsub * rb, cudaMemcpyDeviceToDevice, s), "wgp copy");
    k::q80p_pack(wp, rows, s);
    k::q80p_pack(wgp, nsub, s);
    float* x = nullptr;
    void* xq = nullptr;
    float *y0 = nullptr, *y1 = nullptr, *y2 = nullptr;
    ck(cudaMalloc(&x, (size_t) n_in * 4), "x");
    ck(cudaMalloc(&xq, k::native_q8_1_bytes(n_in, 1)), "xq");
    ck(cudaMalloc(&y0, (size_t) nsub * 4), "y0");
    ck(cudaMalloc(&y1, (size_t) nsub * 4), "y1");
    ck(cudaMalloc(&y2, (size_t) nsub * 4), "y2");
    int fail = 0;
    std::vector<float> hxf((size_t) n_in);
    std::vector<uint8_t> hxq(k::native_q8_1_bytes(n_in, 1));
    std::vector<double> xd;
    std::vector<float> dq;

    // 1. every row through the map, bitwise against the copy, and against the CPU reference
    {
        int bad = 0;
        double max_abs = 0, max_rel = 0;
        std::vector<float> a((size_t) nsub), b(a.size()), c(a.size());
        for (int tr = 0; tr < 6; ++tr) {
            for (auto& v : hxf) v = nd(rng);
            ck(cudaMemcpy(x, hxf.data(), hxf.size() * 4, cudaMemcpyHostToDevice), "x up");
            k::native_quantize_q8_1(x, xq, n_in, 1, s);
            k::native_mmvq(8, wg, xq, y0, n_in, nsub, 1, s);
            k::native_q8_0_mmvq_stream(wg, xq, y1, n_in, nsub, s);
            k::q80_mmvq_rows(wp, d_map, xq, y2, n_in, nsub, s);
            ck(cudaStreamSynchronize(s), "sync");
            ck(cudaMemcpy(a.data(), y0, a.size() * 4, cudaMemcpyDeviceToHost), "a");
            ck(cudaMemcpy(b.data(), y1, b.size() * 4, cudaMemcpyDeviceToHost), "b");
            ck(cudaMemcpy(c.data(), y2, c.size() * 4, cudaMemcpyDeviceToHost), "c");
            ck(cudaMemcpy(hxq.data(), xq, hxq.size(), cudaMemcpyDeviceToHost), "xq");
            dequant_x(hxq, xd);
            for (int i = 0; i < nsub; ++i) {
                bad += std::memcmp(&a[(size_t) i], &c[(size_t) i], 4) != 0 || std::memcmp(&a[(size_t) i], &b[(size_t) i], 4) != 0;
                if (i % 41 != 0) continue;   // the CPU reference on a sample of rows
                dequant(hw, gmap[(size_t) i], dq);
                double ref = 0, mag = 0;
                for (int j = 0; j < n_in; ++j) { ref += (double) dq[(size_t) j] * xd[(size_t) j]; mag += std::fabs((double) dq[(size_t) j] * xd[(size_t) j]); }
                const double e = std::fabs((double) c[(size_t) i] - ref);
                max_abs = std::max(max_abs, e);
                max_rel = std::max(max_rel, e / std::max(1e-30, mag));
            }
        }
        std::printf("rows by map, %d of %d rows (%d without a token): %s against native_mmvq and the streamed kernel "
                    "(%d differ); CPU reference max abs %.3e, max rel (to the sum of |terms|) %.3e\n",
                    nsub, rows, (int) std::count(map.begin(), map.end(), -1), bad ? "FAIL" : "bitwise", bad, max_abs, max_rel);
        if (bad || max_rel > 1e-5) fail = 1;
    }
    // 2. the 2-bit copy through the map against the copy from the gathered rows (rows without a token excepted)
    void *pa = nullptr, *pb = nullptr, *scratch = nullptr;
    const size_t pbytes = k::q5k_proxy_bytes(nsub), sbytes = k::q5k_proxy_scratch_bytes(nsub);
    ck(cudaMalloc(&pa, pbytes), "pa");
    ck(cudaMalloc(&pb, pbytes), "pb");
    ck(cudaMalloc(&scratch, sbytes), "scratch");
    ck(cudaMemset(scratch, 0, sbytes), "scratch0");
    {
        // the copy packs a hole's row from row 0 in the gathered copy; the map's copy leaves it zero
        std::vector<int32_t> hmap = map;
        for (int32_t& v : hmap) if (v < 0) v = 0;
        int32_t* d_hmap = nullptr;
        ck(cudaMalloc(&d_hmap, hmap.size() * 4), "hmap");
        ck(cudaMemcpy(d_hmap, hmap.data(), hmap.size() * 4, cudaMemcpyHostToDevice), "hmap up");
        ck(cudaMemset(pa, 0, pbytes), "pa0");
        ck(cudaMemset(pb, 0, pbytes), "pb0");
        k::q80_proxy_pack(wgp, 0, nsub, pa, nsub, s);
        k::q80_proxy_pack(wp, 0, nsub, pb, nsub, s, d_hmap);
        ck(cudaStreamSynchronize(s), "pack");
        std::vector<uint8_t> ha(pbytes), hb(pbytes);
        ck(cudaMemcpy(ha.data(), pa, pbytes, cudaMemcpyDeviceToHost), "pa");
        ck(cudaMemcpy(hb.data(), pb, pbytes, cudaMemcpyDeviceToHost), "pb");
        const bool eq = std::memcmp(ha.data(), hb.data(), pbytes) == 0;
        // through the map with holes, in two parts (as context rows fill later) and at once: the same bytes
        ck(cudaMemset(pb, 0, pbytes), "pb0");
        k::q80_proxy_pack(wp, 0, nbase, pb, nsub, s, d_map);
        k::q80_proxy_pack(wp, nbase, nctx, pb, nsub, s, d_map);
        void* pc = nullptr;
        ck(cudaMalloc(&pc, pbytes), "pc");
        ck(cudaMemset(pc, 0, pbytes), "pc0");
        k::q80_proxy_pack(wp, 0, nsub, pc, nsub, s, d_map);
        ck(cudaStreamSynchronize(s), "pack2");
        std::vector<uint8_t> hc(pbytes);
        ck(cudaMemcpy(hb.data(), pb, pbytes, cudaMemcpyDeviceToHost), "pb2");
        ck(cudaMemcpy(hc.data(), pc, pbytes, cudaMemcpyDeviceToHost), "pc2");
        const bool eq2 = std::memcmp(hb.data(), hc.data(), pbytes) == 0;
        cudaFree(pc);
        std::printf("2-bit copy by row map: %s; with holes, in two parts against at once: %s\n", eq ? "bitwise" : "DIFFERS",
                    eq2 ? "bitwise" : "DIFFERS");
        if (!eq || !eq2) fail = 1;
        cudaFree(d_hmap);
    }
    // 3. the two-stage argmax on planted logits: noise plus two rows' directions (the pick and a runner-up 0..4.5
    // units behind it), scaled so the planted logits stand well above a spread of about 3
    {
        int *d_id = nullptr, *r_id = nullptr;
        float *d_p = nullptr, *r_p = nullptr;
        ck(cudaMalloc(&d_id, 16), "id"); ck(cudaMalloc(&d_p, 16), "p");
        ck(cudaMalloc(&r_id, 16), "rid"); ck(cudaMalloc(&r_p, 16), "rp");
        // the head's spread on random input, to scale the planted rows
        for (auto& v : hxf) v = nd(rng);
        ck(cudaMemcpy(x, hxf.data(), hxf.size() * 4, cudaMemcpyHostToDevice), "x up");
        k::native_quantize_q8_1(x, xq, n_in, 1, s);
        k::native_mmvq(8, wg, xq, y0, n_in, nsub, 1, s);
        std::vector<float> l0((size_t) nsub);
        ck(cudaMemcpy(l0.data(), y0, l0.size() * 4, cudaMemcpyDeviceToHost), "l0");
        double m1 = 0, m2 = 0;
        for (float v : l0) { m1 += v; m2 += (double) v * v; }
        const double sd = std::sqrt(m2 / nsub - (m1 / nsub) * (m1 / nsub));
        const float xs = (float) (3.0 / sd);
        ck(cudaMemset(pb, 0, pbytes), "pb0");
        k::q80_proxy_pack(wp, 0, nsub, pb, nsub, s, d_map);
        std::vector<float> a((size_t) nsub);
        int miss_map = 0, miss_copy = 0, miss_cpu = 0, ties = 0;
        double max_rel = 0, sum_rel = 0, sum_bias = 0;
        int n_rel = 0;
        for (int tr = 0; tr < trials; ++tr) {
            for (auto& v : hxf) v = 0.8f * xs * nd(rng);
            const int gap = (int) (rng() % 5);   // runner-up 0..4 units behind
            for (int pl = 0; pl < 2; ++pl) {
                int r;
                do { r = (int) (rng() % (unsigned) nsub); } while (map[(size_t) r] < 0);
                dequant(hw, map[(size_t) r], dq);
                double nn = 0;
                for (float v : dq) nn += (double) v * v;
                const double alpha = (pl == 0 ? 20.0 : 20.0 - gap - 0.5 * (rng() % 2)) / nn;
                for (int i = 0; i < n_in; ++i) hxf[(size_t) i] += (float) (alpha * dq[(size_t) i]);
            }
            ck(cudaMemcpy(x, hxf.data(), hxf.size() * 4, cudaMemcpyHostToDevice), "x up");
            k::native_quantize_q8_1(x, xq, n_in, 1, s);
            k::q80_mmvq_rows(wp, d_map, xq, y0, n_in, nsub, s);
            if (!k::q80_proxy_top1(pb, wp, xq, nsub, nsub, d_map, d_id, d_p, scratch, s, true) ||
                !k::q80_proxy_top1(pa, wgp, xq, nsub, nsub, d_map, d_id + 1, d_p + 1, scratch, s)) {
                std::printf("proxy refused\n");
                fail = 1;
                break;
            }
            ck(cudaStreamSynchronize(s), "sync");
            ck(cudaMemcpy(a.data(), y0, a.size() * 4, cudaMemcpyDeviceToHost), "a");
            ck(cudaMemcpy(hxq.data(), xq, hxq.size(), cudaMemcpyDeviceToHost), "xq");
            int best = -1;
            for (int i = 0; i < nsub; ++i)
                if (map[(size_t) i] >= 0 && (best < 0 || a[(size_t) i] > a[(size_t) best])) best = i;
            double sum = 0;
            for (int i = 0; i < nsub; ++i)
                if (map[(size_t) i] >= 0) sum += std::exp((double) a[(size_t) i] - (double) a[(size_t) best]);
            // the CPU reference's argmax over the near-best rows
            dequant_x(hxq, xd);
            int cbest = -1;
            double cv = -1e300, second = -1e300;
            for (int i = 0; i < nsub; ++i) {
                if (map[(size_t) i] < 0 || a[(size_t) i] < a[(size_t) best] - 1.0f) continue;
                dequant(hw, map[(size_t) i], dq);
                double ref = 0;
                for (int j = 0; j < n_in; ++j) ref += (double) dq[(size_t) j] * xd[(size_t) j];
                if (ref > cv) { second = cv; cv = ref; cbest = i; }
                else second = std::max(second, ref);
            }
            ties += cv - second < 1e-5;
            int id[2];
            float p[2];
            ck(cudaMemcpy(id, d_id, 8, cudaMemcpyDeviceToHost), "id");
            ck(cudaMemcpy(p, d_p, 8, cudaMemcpyDeviceToHost), "p");
            miss_map += id[0] != map[(size_t) best];
            miss_copy += id[1] != map[(size_t) best];
            miss_cpu += cbest >= 0 && id[0] != map[(size_t) cbest];
            if (id[0] == map[(size_t) best]) {
                const double rel = (double) p[0] * sum - 1.0;
                max_rel = std::max(max_rel, std::fabs(rel));
                sum_rel += std::fabs(rel);
                sum_bias += rel;
                ++n_rel;
            }
            // the copy and the map give the same pick; the probability's sum may run in another order
            if (id[0] != id[1] || std::fabs(p[0] - p[1]) > 1e-5f * p[0]) miss_copy += 1000;
        }
        std::printf("two-stage argmax, %d trials over %d rows: picks differing from the exact argmax: by map %d, copy "
                    "%d (by map == copy: %s); from the CPU reference's %d (near-ties %d); probability rel error to the "
                    "exact softmax: mean %.4f, bias %+.4f, max %.4f\n",
                    trials, nsub, miss_map, miss_copy % 1000, miss_copy >= 1000 ? "NO" : "yes", miss_cpu, ties,
                    sum_rel / std::max(1, n_rel), sum_bias / std::max(1, n_rel), max_rel);
        if (miss_map > trials / 100 || miss_copy >= 1000 || max_rel > 0.5) fail = 1;
        // 4. cold microbench at this shape: the two-stage argmax against the streamed head + mask + top-prob
        void* flush = nullptr;
        ck(cudaMalloc(&flush, 512u << 20), "flush");
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        float tot[3] = {0, 0, 0};
        const int reps = 60;
        for (int it = 0; it < reps + 4; ++it)
            for (int v = 0; v < 3; ++v) {
                ck(cudaMemsetAsync(flush, it, 512u << 20, s), "flush");
                cudaEventRecord(e0, s);
                if (v == 0) k::q80_proxy_top1(pb, wp, xq, nsub, nsub, d_map, d_id, d_p, scratch, s, true);
                else if (v == 1) {
                    k::native_q8_0_mmvq_stream(wg, xq, y1, n_in, nsub, s);
                    k::q5k_head_mask_unmapped(y1 + nbase, d_map + nbase, nctx, s);
                    k::sample_greedy_top_prob(y1, 1, nsub, d_map, r_id, r_p, s);
                } else k::q80_mmvq_rows(wp, d_map, xq, y2, n_in, nsub, s);
                cudaEventRecord(e1, s);
                cudaEventSynchronize(e1);
                float ms;
                cudaEventElapsedTime(&ms, e0, e1);
                if (it >= 4) tot[v] += ms;
            }
        const double gb_px = (double) pbytes / 1e3, gb_ex = (double) nsub * rb / 1e3;
        const double u0 = 1e3 * tot[0] / reps, u1 = 1e3 * tot[1] / reps, u2 = 1e3 * tot[2] / reps;
        std::printf("cold, %d rows: two-stage %.1f us (copy %.1f MB, %.0f GB/s) | streamed head + mask + top-prob %.1f us "
                    "(%.1f MB, %.0f GB/s) | rows by map %.1f us (%.0f GB/s) | %.2fx\n",
                    nsub, u0, gb_px / 1e3, gb_px / u0, u1, gb_ex / 1e3, gb_ex / u1, u2, gb_ex / u2, u1 / u0);
        cudaFree(flush);
    }
    std::printf("%s\n", fail ? "q80_proxy_parity: FAIL" : "q80_proxy_parity: OK");
    return fail;
}
