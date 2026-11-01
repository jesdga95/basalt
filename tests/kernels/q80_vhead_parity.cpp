// tests/kernels/q80_vhead_parity.cpp - the Q8_0 head in bit planes (q80p) and its two-stage verify head.
//   1. The planes: a head packed and unpacked again holds the same bytes.
//   2. q80p_mmvq on the planes against native_mmvq on the Q8_0 rows: bitwise at 1..8 columns, each column the bits of
//      its one-column call; against a CPU reference (double) on sampled rows.  Over a split's two row ranges (the first
//      rows of the head, a copy of the rest): the same bits as the whole head's rows.
//   3. q80p_vhead at K = 1, 20, 64: per column the K largest entries (by value, then lower id) are the exact head's ids
//      in the same order with the same bits, every other entry is at or above its exact logit and below the K-th, the
//      sampler draws the same tokens and probabilities (greedy, and top_k <= K with top_p, min_p, temperature), and a
//      column's entries do not depend on the window's width.  K = 0 is the exact head, bitwise.
//   4. A share of the head on another card (the last fifth of the rows on a copy of their own, screened there against
//      the first head's published thresholds, its scored entries listed and merged): the same K largest entries and
//      sampler draws as the exact head; K = 0 (batch windows, the exact pass) is the exact head, bitwise.  Two splits:
//      the last fifth, and the share the verify plan sizes for a peer of about a quarter of CUDA0's bandwidth.
//   5. A cold microbench at the head's shape (L2 flushed before each launch), with the split's two sides timed alone.
// Random Q8_0 rows (ggml's rounding of Gaussian weights) with 64 rows per column leaned towards it, so each column has
// a head's spread of near-top logits, and random columns quantized by native_quantize_q8_1.
//   q80_vhead_parity [rows] [iters]
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/q80_vhead.hpp"
#include "basalt/kernels/sampler.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace k = basalt::kernels;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
constexpr int kNin = 2560, kBpr = 80, kBlk = 34, kMaxT = 8;
constexpr size_t kRb = (size_t) kBpr * kBlk;

// ggml's Q8_0 rounding of one row
void quantize_row(const float* w, uint8_t* out) {
    for (int b = 0; b < kBpr; ++b) {
        float amax = 0.0f;
        for (int l = 0; l < 32; ++l) amax = std::max(amax, std::fabs(w[b * 32 + l]));
        const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        const __half dh = __float2half(d);
        std::memcpy(out + (size_t) b * kBlk, &dh, 2);
        for (int l = 0; l < 32; ++l) out[(size_t) b * kBlk + 2 + l] = (uint8_t) (int8_t) std::lround(w[b * 32 + l] * id);
    }
}
double dot_ref(const uint8_t* row, const uint8_t* xq) {   // dequantized row . dequantized q8_1 column
    double s = 0.0;
    for (int b = 0; b < kBpr; ++b) {
        __half dw, dx;
        std::memcpy(&dw, row + (size_t) b * kBlk, 2);
        std::memcpy(&dx, xq + (size_t) b * 36, 2);
        long long si = 0;
        for (int l = 0; l < 32; ++l) si += (long long) (int8_t) row[(size_t) b * kBlk + 2 + l] * (int8_t) xq[(size_t) b * 36 + 4 + l];
        s += (double) __half2float(dw) * (double) __half2float(dx) * (double) si;
    }
    return s;
}
// ids in the sampler's selection order (value, then lower id), the first n
std::vector<int> top_ids(const float* y, int rows, int n) {
    std::vector<int> id((size_t) rows);
    std::iota(id.begin(), id.end(), 0);
    std::partial_sort(id.begin(), id.begin() + n, id.end(), [&](int a, int b) { return y[a] > y[b] || (y[a] == y[b] && a < b); });
    id.resize((size_t) n);
    return id;
}
bool same_bits(float a, float b) { return std::memcmp(&a, &b, 4) == 0; }
}  // namespace

int main(int argc, char** argv) {
    const int rows = argc > 1 ? std::atoi(argv[1]) : 248320;
    const int iters = argc > 2 ? std::atoi(argv[2]) : 5;
    int fail = 0;
    // ---- the head and the columns
    std::mt19937_64 rng(35);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const float sw = 0.02f;
    std::vector<uint8_t> hw((size_t) rows * kRb);
    std::vector<float> xs((size_t) kMaxT * kNin);
    for (float& v : xs) v = nd(rng);
    std::vector<int> planted;
    {
        std::vector<float> wr(kNin);
        std::vector<int> pick((size_t) rows, -1);   // the column a planted row leans towards
        std::uniform_int_distribution<int> ur(0, rows - 1);
        for (int c = 0; c < kMaxT; ++c)
            for (int i = 0; i < 64 && i < rows / (2 * kMaxT); ++i) {
                int r;
                do r = ur(rng); while (pick[(size_t) r] >= 0);
                pick[(size_t) r] = c * 64 + i;
                planted.push_back(r);
            }
        std::vector<double> xn(kMaxT, 0.0);
        for (int c = 0; c < kMaxT; ++c) {
            for (int l = 0; l < kNin; ++l) xn[c] += (double) xs[(size_t) c * kNin + l] * xs[(size_t) c * kNin + l];
            xn[c] = std::sqrt(xn[c]);
        }
        for (int r = 0; r < rows; ++r) {
            for (int l = 0; l < kNin; ++l) wr[l] = sw * nd(rng);
            if (pick[(size_t) r] >= 0) {   // logit about (20 - i / 4) standard deviations above the column's mean
                const int c = pick[(size_t) r] / 64, i = pick[(size_t) r] % 64;
                const double beta = (20.0 - 0.25 * i) * sw / xn[c];
                for (int l = 0; l < kNin; ++l) wr[l] += (float) (beta * xs[(size_t) c * kNin + l]);
            }
            quantize_row(wr.data(), &hw[(size_t) r * kRb]);
        }
    }
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    void *w8 = nullptr, *wp = nullptr, *xq = nullptr, *scratch = nullptr;
    float *xf = nullptr, *y0 = nullptr, *y1 = nullptr;
    int* kd = nullptr;
    const size_t xb = k::native_q8_1_bytes(kNin, kMaxT);
    ck(cudaMalloc(&w8, hw.size()), "w8");
    ck(cudaMalloc(&wp, hw.size()), "wp");
    ck(cudaMalloc(&xq, xb), "xq");
    ck(cudaMalloc((void**) &xf, xs.size() * 4), "xf");
    ck(cudaMalloc((void**) &y0, (size_t) kMaxT * rows * 4), "y0");
    ck(cudaMalloc((void**) &y1, (size_t) kMaxT * rows * 4), "y1");
    ck(cudaMalloc((void**) &kd, sizeof(int)), "k");
    ck(cudaMalloc(&scratch, k::q80p_vhead_scratch_bytes()), "scratch");
    ck(cudaMemset(scratch, 0, k::q80p_vhead_scratch_bytes()), "scratch zero");
    ck(cudaMemcpy(w8, hw.data(), hw.size(), cudaMemcpyHostToDevice), "w8 up");
    ck(cudaMemcpy(wp, hw.data(), hw.size(), cudaMemcpyHostToDevice), "wp up");
    ck(cudaMemcpy(xf, xs.data(), xs.size() * 4, cudaMemcpyHostToDevice), "x up");
    k::native_quantize_q8_1(xf, xq, kNin, kMaxT, s);
    std::vector<uint8_t> hxq(xb);
    ck(cudaMemcpyAsync(hxq.data(), xq, xb, cudaMemcpyDeviceToHost, s), "xq down");

    // ---- 1. the planes round trip
    k::q80p_pack(wp, rows, s);
    {
        void* rt = nullptr;
        ck(cudaMalloc(&rt, hw.size()), "rt");
        ck(cudaMemcpyAsync(rt, wp, hw.size(), cudaMemcpyDeviceToDevice, s), "rt copy");
        k::q80p_unpack(rt, rows, s);
        std::vector<uint8_t> back(hw.size());
        ck(cudaMemcpyAsync(back.data(), rt, back.size(), cudaMemcpyDeviceToHost, s), "rt down");
        ck(cudaStreamSynchronize(s), "rt sync");
        const bool ok = back == hw;
        std::printf("planes: pack + unpack %s\n", ok ? "bitwise the Q8_0 rows" : "DIFFERS");
        if (!ok) fail = 1;
        cudaFree(rt);
    }

    // ---- 2. the exact pass
    std::vector<float> one((size_t) kMaxT * rows);   // each column at one column
    for (int c = 0; c < kMaxT; ++c) {
        k::native_mmvq(8, w8, (const uint8_t*) xq + (size_t) c * 80 * 36, y0, kNin, rows, 1, s);
        ck(cudaMemcpyAsync(&one[(size_t) c * rows], y0, (size_t) rows * 4, cudaMemcpyDeviceToHost, s), "one");
    }
    ck(cudaStreamSynchronize(s), "one sync");
    {
        double max_abs = 0.0, max_rel = 0.0;
        std::vector<int> sample(planted);
        for (int r = 0; r < rows; r += 97) sample.push_back(r);
        for (int c = 0; c < kMaxT; ++c)
            for (int r : sample) {
                const double ref = dot_ref(&hw[(size_t) r * kRb], &hxq[(size_t) c * 80 * 36]);
                const double e = std::fabs((double) one[(size_t) c * rows + r] - ref);
                max_abs = std::max(max_abs, e);
                max_rel = std::max(max_rel, e / std::max(1e-6, std::fabs(ref)));
            }
        std::printf("native_mmvq vs CPU reference (%zu rows x %d columns): max abs %.3e, max rel %.3e\n", sample.size(),
                    kMaxT, max_abs, max_rel);
    }
    for (int nc = 1; nc <= kMaxT; ++nc) {
        k::native_mmvq(8, w8, xq, y0, kNin, rows, nc, s);
        k::q80p_mmvq(wp, xq, y1, kNin, rows, nc, s);
        std::vector<float> a((size_t) nc * rows), b(a.size());
        ck(cudaMemcpyAsync(a.data(), y0, a.size() * 4, cudaMemcpyDeviceToHost, s), "a");
        ck(cudaMemcpyAsync(b.data(), y1, b.size() * 4, cudaMemcpyDeviceToHost, s), "b");
        ck(cudaStreamSynchronize(s), "exact sync");
        const bool vs_native = std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
        const bool vs_one = std::memcmp(b.data(), one.data(), b.size() * 4) == 0;
        std::printf("q80p_mmvq ncols %d: vs native_mmvq %s, vs one column each %s\n", nc, vs_native ? "bitwise" : "DIFFERS",
                    vs_one ? "bitwise" : "DIFFERS");
        if (!vs_native || !vs_one) fail = 1;
    }
    // a split's two row ranges, each run as a head of its own
    const int split_rows = (int) ((int64_t) rows * 201984 / 248320) / 256 * 256;
    for (const int r0 : {rows - rows / 5, split_rows}) {
        const int r1 = rows - r0;
        void* wq = nullptr;
        ck(cudaMalloc(&wq, (size_t) r1 * kRb), "range copy");
        ck(cudaMemcpyAsync(wq, (const uint8_t*) wp + (size_t) r0 * kRb, (size_t) r1 * kRb, cudaMemcpyDeviceToDevice, s),
           "range copy up");
        for (int nc = 1; nc <= kMaxT; ++nc) {
            k::q80p_mmvq(wp, xq, y1, kNin, r0, nc, s);
            k::q80p_mmvq(wq, xq, y0, kNin, r1, nc, s);
            std::vector<float> a((size_t) nc * r0), b((size_t) nc * r1);
            ck(cudaMemcpyAsync(a.data(), y1, a.size() * 4, cudaMemcpyDeviceToHost, s), "range a");
            ck(cudaMemcpyAsync(b.data(), y0, b.size() * 4, cudaMemcpyDeviceToHost, s), "range b");
            ck(cudaStreamSynchronize(s), "range sync");
            bool ok = true;
            for (int c = 0; c < nc; ++c)
                ok = ok && std::memcmp(&a[(size_t) c * r0], &one[(size_t) c * rows], (size_t) r0 * 4) == 0 &&
                     std::memcmp(&b[(size_t) c * r1], &one[(size_t) c * rows + r0], (size_t) r1 * 4) == 0;
            std::printf("q80p_mmvq rows %d + %d ncols %d: %s\n", r0, r1, nc, ok ? "bitwise the whole head's" : "DIFFERS");
            if (!ok) fail = 1;
        }
        cudaFree(wq);
    }

    // ---- 3. the two-stage head
    const int keeps[3] = {1, 20, 64};
    std::vector<double> rescored(4 * (kMaxT + 1), 0.0);
    for (int ki = 0; ki < 4; ++ki) {
        const int K = ki < 3 ? keeps[ki] : 0;
        ck(cudaMemcpyAsync(kd, &K, sizeof(int), cudaMemcpyHostToDevice, s), "k up");
        std::vector<float> first((size_t) kMaxT * rows);   // each column at one column
        for (int c = 0; c < kMaxT; ++c) {
            k::q80p_vhead(wp, (const uint8_t*) xq + (size_t) c * 80 * 36, y1, kNin, rows, 1, kd, scratch, s);
            ck(cudaMemcpyAsync(&first[(size_t) c * rows], y1, (size_t) rows * 4, cudaMemcpyDeviceToHost, s), "first");
        }
        ck(cudaStreamSynchronize(s), "first sync");
        for (int nc = 1; nc <= kMaxT; ++nc) {
            k::q80p_vhead(wp, xq, y1, kNin, rows, nc, kd, scratch, s);
            std::vector<float> b((size_t) nc * rows);
            ck(cudaMemcpyAsync(b.data(), y1, b.size() * 4, cudaMemcpyDeviceToHost, s), "vhead");
            ck(cudaStreamSynchronize(s), "vhead sync");
            long top_bad = 0, below = 0, above_kth = 0, n_exact = 0;
            const bool width_bad = std::memcmp(b.data(), first.data(), b.size() * 4) != 0;
            for (int c = 0; c < nc; ++c) {
                const float* e = &one[(size_t) c * rows];
                const float* v = &b[(size_t) c * rows];
                if (K == 0) {
                    top_bad += std::memcmp(e, v, (size_t) rows * 4) != 0;
                    continue;
                }
                const std::vector<int> te = top_ids(e, rows, K), tv = top_ids(v, rows, K);
                for (int i = 0; i < K; ++i) top_bad += te[(size_t) i] != tv[(size_t) i] || !same_bits(e[te[(size_t) i]], v[tv[(size_t) i]]);
                const float kth = e[te[(size_t) K - 1]];
                for (int r = 0; r < rows; ++r) {
                    if (same_bits(e[r], v[r])) { ++n_exact; continue; }
                    below += !(v[r] >= e[r]);   // an upper bound, or the exact value
                    above_kth += !(v[r] < kth);
                }
            }
            rescored[(size_t) ki * (kMaxT + 1) + nc] = (double) n_exact / nc;
            const bool bad = top_bad || below || above_kth || width_bad;
            if (K == 0)
                std::printf("q80p_vhead K 0 ncols %d: %s (columns not the exact head %ld, column differs from one column %d)\n",
                            nc, bad ? "FAIL" : "ok", top_bad, (int) width_bad);
            else
                std::printf("q80p_vhead K %2d ncols %d: %s (top-K entries differing %ld, entries below the exact logit %ld, "
                            "inexact entries reaching the K-th %ld, column differs from one column %d; exact entries per "
                            "column %.0f)\n", K, nc, bad ? "FAIL" : "ok", top_bad, below, above_kth, (int) width_bad,
                            rescored[(size_t) ki * (kMaxT + 1) + nc]);
            if (bad) fail = 1;
        }
        if (K == 0) continue;
        // the sampler on both: greedy, then top_k <= K draws
        ck(cudaMemcpyAsync(y0, one.data(), one.size() * 4, cudaMemcpyHostToDevice, s), "exact up");
        k::q80p_vhead(wp, xq, y1, kNin, rows, kMaxT, kd, scratch, s);
        int *o0 = nullptr, *o1 = nullptr, *qd = nullptr;
        float *p0 = nullptr, *p1 = nullptr;
        ck(cudaMalloc((void**) &o0, kMaxT * 4), "o0");
        ck(cudaMalloc((void**) &o1, kMaxT * 4), "o1");
        ck(cudaMalloc((void**) &qd, kMaxT * 4), "qd");
        ck(cudaMalloc((void**) &p0, kMaxT * 4), "p0");
        ck(cudaMalloc((void**) &p1, kMaxT * 4), "p1");
        long draws = 0, differ = 0;
        for (int trial = 0; trial < 48; ++trial) {
            k::SamplerParams sp;
            const int tk[4] = {1, 2, 5, K};
            sp.greedy = trial == 0;
            sp.temperature = trial == 0 ? 0.0f : (trial % 3 == 0 ? 0.6f : 1.0f);
            sp.top_k = std::min(K, tk[trial % 4]);
            sp.top_p = trial % 2 ? 0.95f : 1.0f;
            sp.min_p = trial % 5 == 1 ? 0.05f : 0.0f;
            sp.seed = 1000 + (uint64_t) trial;
            sp.counter = (uint64_t) trial * 8;
            std::vector<int> q(kMaxT);   // the query: a near-top token of each column
            for (int c = 0; c < kMaxT; ++c) q[(size_t) c] = top_ids(&one[(size_t) c * rows], rows, std::min(K, 3))[(size_t) std::min(K, 3) - 1];
            ck(cudaMemcpyAsync(qd, q.data(), kMaxT * 4, cudaMemcpyHostToDevice, s), "q up");
            k::sample_tokens_probe(y0, kMaxT, rows, nullptr, 0, sp, o0, qd, kMaxT, p0, s);
            k::sample_tokens_probe(y1, kMaxT, rows, nullptr, 0, sp, o1, qd, kMaxT, p1, s);
            int h0[kMaxT], h1[kMaxT];
            float f0[kMaxT], f1[kMaxT];
            ck(cudaMemcpyAsync(h0, o0, sizeof h0, cudaMemcpyDeviceToHost, s), "o0 down");
            ck(cudaMemcpyAsync(h1, o1, sizeof h1, cudaMemcpyDeviceToHost, s), "o1 down");
            ck(cudaMemcpyAsync(f0, p0, sizeof f0, cudaMemcpyDeviceToHost, s), "p0 down");
            ck(cudaMemcpyAsync(f1, p1, sizeof f1, cudaMemcpyDeviceToHost, s), "p1 down");
            ck(cudaStreamSynchronize(s), "sample sync");
            for (int c = 0; c < kMaxT; ++c) {
                ++draws;
                differ += h0[c] != h1[c] || (!sp.greedy && !same_bits(f0[c], f1[c]));
            }
        }
        std::printf("sampler K %2d: %ld draws (greedy, top_k 1..%d with top_p / min_p / temperature), %ld differ\n", K, draws,
                    K, differ);
        if (differ) fail = 1;
        cudaFree(o0); cudaFree(o1); cudaFree(qd); cudaFree(p0); cudaFree(p1);
    }

    // ---- 4. a share of the head on another card: the rows past r0 as a head of their own (its own copy, scratch and
    // threshold), its listed entries merged into the first head's column, which screened only the rows before r0
    for (const int r0 : {rows - rows / 5, split_rows}) {
        const int r1 = rows - r0;
        void *wq = nullptr, *scr1 = nullptr, *list = nullptr, *link = nullptr, *input = nullptr, *xq1 = nullptr;
        int* kd1 = nullptr;
        float* y2 = nullptr;
        ck(cudaMalloc(&wq, (size_t) r1 * kRb), "share");
        ck(cudaMemcpyAsync(wq, (const uint8_t*) wp + (size_t) r0 * kRb, (size_t) r1 * kRb, cudaMemcpyDeviceToDevice, s),
           "share copy");
        ck(cudaMalloc((void**) &y2, (size_t) kMaxT * r1 * 4), "y2");
        ck(cudaMalloc(&scr1, k::q80p_vhead_scratch_bytes()), "scratch1");
        ck(cudaMemset(scr1, 0, k::q80p_vhead_scratch_bytes()), "scratch1 zero");
        ck(cudaHostAlloc(&list, k::q80p_vhead_list_bytes(kMaxT, r1), cudaHostAllocMapped | cudaHostAllocPortable), "list");
        ck(cudaHostAlloc(&link, k::q80p_vhead_link_bytes(), cudaHostAllocMapped | cudaHostAllocPortable), "link");
        std::memset(link, 0, k::q80p_vhead_link_bytes());
        ck(cudaHostAlloc(&input, k::q80p_vhead_input_bytes(kNin, kMaxT), cudaHostAllocMapped | cudaHostAllocPortable), "in");
        ck(cudaMalloc(&xq1, xb), "xq1");
        ck(cudaMalloc((void**) &kd1, sizeof(int)), "k1");
        unsigned seq = 0;
        // one stream here, in the order the cards meet: this card posts and publishes before the share fetches and lists
        auto split_head = [&](int nc) {
            ck(cudaStreamSynchronize(s), "seq sync");
            ((volatile unsigned*) link)[0] = ++seq;
            k::q80p_vhead_post(xq, kNin, nc, kd, input, link, s);
            k::q80p_vhead(wp, xq, y1, kNin, rows, nc, kd, scratch, s, r0, (unsigned*) link);
            k::q80p_vhead_fetch(input, link, xq1, kd1, kNin, nc, scr1, s);
            k::q80p_vhead(wq, xq1, y2, kNin, r1, nc, kd1, scr1, s, -1, nullptr, (const unsigned*) link);
            k::q80p_vhead_list(y2, r1, nc, kd1, scr1, list, link, s);
            k::q80p_vhead_merge(y1, rows, r0, nc, r1, kd, list, link, s);
        };
        for (int K : {1, 20, 64, 0}) {
            ck(cudaMemcpyAsync(kd, &K, sizeof(int), cudaMemcpyHostToDevice, s), "k up");
            for (int nc = 1; nc <= kMaxT; ++nc) {
                split_head(nc);
                std::vector<float> b((size_t) nc * rows);
                ck(cudaMemcpyAsync(b.data(), y1, b.size() * 4, cudaMemcpyDeviceToHost, s), "split");
                ck(cudaStreamSynchronize(s), "split sync");
                if (k::q80p_vhead_link_failed(link)) { std::printf("split head K %d ncols %d: the link failed\n", K, nc); fail = 1; }
                long top_bad = 0, bad_entry = 0, listed = 0;
                for (int c = 0; c < nc; ++c) {
                    const float* e = &one[(size_t) c * rows];
                    const float* v = &b[(size_t) c * rows];
                    listed += ((const unsigned*) list)[c];
                    if (K == 0) {
                        top_bad += std::memcmp(e, v, (size_t) rows * 4) != 0;
                        continue;
                    }
                    const std::vector<int> te = top_ids(e, rows, K), tv = top_ids(v, rows, K);
                    for (int i = 0; i < K; ++i)
                        top_bad += te[(size_t) i] != tv[(size_t) i] || !same_bits(e[te[(size_t) i]], v[tv[(size_t) i]]);
                    const float kth = e[te[(size_t) K - 1]];
                    for (int r = 0; r < rows; ++r) {
                        if (same_bits(e[r], v[r])) continue;
                        if (r >= r0) { bad_entry += !(std::isinf(v[r]) && v[r] < 0.0f); continue; }   // dropped: -inf
                        bad_entry += !(v[r] >= e[r]) || !(v[r] < kth);
                    }
                }
                const bool bad = top_bad || bad_entry;
                std::printf("split head K %2d ncols %d (rows %d + %d): %s (top-K entries differing %ld, entries neither exact, "
                            "an upper bound below the K-th nor -inf past the split %ld; share rows listed per column %.0f)\n",
                            K, nc, r0, r1, bad ? "FAIL" : "ok", top_bad, bad_entry, (double) listed / nc);
                if (bad) fail = 1;
            }
            if (K == 0) continue;
            // the sampler on the merged head against the exact head
            ck(cudaMemcpyAsync(y0, one.data(), one.size() * 4, cudaMemcpyHostToDevice, s), "exact up");
            split_head(kMaxT);
            int *o0 = nullptr, *o1 = nullptr, *qd = nullptr;
            float *p0 = nullptr, *p1 = nullptr;
            ck(cudaMalloc((void**) &o0, kMaxT * 4), "o0");
            ck(cudaMalloc((void**) &o1, kMaxT * 4), "o1");
            ck(cudaMalloc((void**) &qd, kMaxT * 4), "qd");
            ck(cudaMalloc((void**) &p0, kMaxT * 4), "p0");
            ck(cudaMalloc((void**) &p1, kMaxT * 4), "p1");
            long draws = 0, differ = 0;
            for (int trial = 0; trial < 48; ++trial) {
                k::SamplerParams sp;
                const int tk[4] = {1, 2, 5, K};
                sp.greedy = trial == 0;
                sp.temperature = trial == 0 ? 0.0f : (trial % 3 == 0 ? 0.6f : 1.0f);
                sp.top_k = std::min(K, tk[trial % 4]);
                sp.top_p = trial % 2 ? 0.95f : 1.0f;
                sp.min_p = trial % 5 == 1 ? 0.05f : 0.0f;
                sp.seed = 2000 + (uint64_t) trial;
                sp.counter = (uint64_t) trial * 8;
                std::vector<int> q(kMaxT);
                for (int c = 0; c < kMaxT; ++c)
                    q[(size_t) c] = top_ids(&one[(size_t) c * rows], rows, std::min(K, 3))[(size_t) std::min(K, 3) - 1];
                ck(cudaMemcpyAsync(qd, q.data(), kMaxT * 4, cudaMemcpyHostToDevice, s), "q up");
                k::sample_tokens_probe(y0, kMaxT, rows, nullptr, 0, sp, o0, qd, kMaxT, p0, s);
                k::sample_tokens_probe(y1, kMaxT, rows, nullptr, 0, sp, o1, qd, kMaxT, p1, s);
                int h0[kMaxT], h1[kMaxT];
                float f0[kMaxT], f1[kMaxT];
                ck(cudaMemcpyAsync(h0, o0, sizeof h0, cudaMemcpyDeviceToHost, s), "o0 down");
                ck(cudaMemcpyAsync(h1, o1, sizeof h1, cudaMemcpyDeviceToHost, s), "o1 down");
                ck(cudaMemcpyAsync(f0, p0, sizeof f0, cudaMemcpyDeviceToHost, s), "p0 down");
                ck(cudaMemcpyAsync(f1, p1, sizeof f1, cudaMemcpyDeviceToHost, s), "p1 down");
                ck(cudaStreamSynchronize(s), "sample sync");
                for (int c = 0; c < kMaxT; ++c) {
                    ++draws;
                    differ += h0[c] != h1[c] || (!sp.greedy && !same_bits(f0[c], f1[c]));
                }
            }
            std::printf("split head sampler K %2d: %ld draws (greedy, top_k 1..%d with top_p / min_p / temperature), %ld differ\n",
                        K, draws, K, differ);
            if (differ) fail = 1;
            cudaFree(o0); cudaFree(o1); cudaFree(qd); cudaFree(p0); cudaFree(p1);
        }
        cudaFree(wq); cudaFree(y2); cudaFree(scr1); cudaFree(xq1); cudaFree(kd1); cudaFreeHost(list); cudaFreeHost(link);
        cudaFreeHost(input);
    }

    // ---- 5. cold microbench
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    void* flush = nullptr;
    ck(cudaMalloc(&flush, 256u << 20), "flush");
    const double mb8 = (double) hw.size() / 1e6, mbs = (double) rows * 2080 / 1e6;
    for (int nc = 1; nc <= kMaxT; ++nc) {
        float tot[5] = {0, 0, 0, 0, 0};
        for (int it = 0; it < iters; ++it)
            for (int v = 0; v < 5; ++v) {
                const int K = v == 2 ? 1 : v == 3 ? 20 : 64;
                if (v >= 2) ck(cudaMemcpyAsync(kd, &K, sizeof(int), cudaMemcpyHostToDevice, s), "k up");
                ck(cudaMemsetAsync(flush, it, 256u << 20, s), "flush");
                cudaEventRecord(e0, s);
                if (v == 0) {
                    k::native_mmvq(8, w8, xq, y0, kNin, rows, nc, s);
                } else if (v == 1) {
                    k::q80p_mmvq(wp, xq, y1, kNin, rows, nc, s);
                } else {
                    k::q80p_vhead(wp, xq, y1, kNin, rows, nc, kd, scratch, s);
                }
                cudaEventRecord(e1, s);
                cudaEventSynchronize(e1);
                float ms;
                cudaEventElapsedTime(&ms, e0, e1);
                tot[v] += ms;
            }
        double u[5];
        for (int v = 0; v < 5; ++v) u[v] = 1e3 * tot[v] / iters;
        std::printf("cold ncols %d: native_mmvq Q8_0 %6.1f us (%4.0f GB/s) | planes exact %6.1f us (%4.0f GB/s) | two-stage K1 "
                    "%6.1f K20 %6.1f K64 %6.1f us (screen %4.0f GB/s at K20)\n", nc, u[0], 1e3 * mb8 / u[0], u[1],
                    1e3 * mb8 / u[1], u[2], u[3], u[4], 1e3 * mbs / u[3]);
    }
    {   // the split head at K 20: the first four fifths screened here, the last fifth alone with its list
        const int r0 = rows - rows / 5, r1 = rows - r0, K = 20;
        void* list = nullptr;
        ck(cudaHostAlloc(&list, k::q80p_vhead_list_bytes(kMaxT, r1), cudaHostAllocMapped | cudaHostAllocPortable), "list");
        ck(cudaMemcpyAsync(kd, &K, sizeof(int), cudaMemcpyHostToDevice, s), "k up");
        void* wq = nullptr;
        void* scr1 = nullptr;
        ck(cudaMalloc(&wq, (size_t) r1 * kRb), "share");
        ck(cudaMemcpyAsync(wq, (const uint8_t*) wp + (size_t) r0 * kRb, (size_t) r1 * kRb, cudaMemcpyDeviceToDevice, s),
           "share copy");
        ck(cudaMalloc(&scr1, k::q80p_vhead_scratch_bytes()), "scratch1");
        ck(cudaMemset(scr1, 0, k::q80p_vhead_scratch_bytes()), "scratch1 zero");
        cudaEvent_t em;
        cudaEventCreate(&em);
        void *link = nullptr, *input = nullptr, *scr0 = nullptr, *xq1 = nullptr;
        int* kd1 = nullptr;
        ck(cudaHostAlloc(&link, k::q80p_vhead_link_bytes(), cudaHostAllocMapped | cudaHostAllocPortable), "link");
        std::memset(link, 0, k::q80p_vhead_link_bytes());
        ck(cudaHostAlloc(&input, k::q80p_vhead_input_bytes(kNin, kMaxT), cudaHostAllocMapped | cudaHostAllocPortable), "in");
        ck(cudaMalloc(&scr0, k::q80p_vhead_scratch_bytes()), "scratch0");
        ck(cudaMemset(scr0, 0, k::q80p_vhead_scratch_bytes()), "scratch0 zero");
        ck(cudaMalloc(&xq1, xb), "xq1");
        ck(cudaMalloc((void**) &kd1, sizeof(int)), "k1");
        unsigned seq = 0;
        for (int nc = 1; nc <= kMaxT; ++nc) {
            float tot[4] = {0, 0, 0, 0};
            for (int it = 0; it < iters; ++it)
                for (int v : {2, 0, 1}) {   // the share (after this card's post and publication, untimed), then this card
                    if (v == 2) {
                        ck(cudaStreamSynchronize(s), "seq sync");
                        ((volatile unsigned*) link)[0] = ++seq;
                        k::q80p_vhead_post(xq, kNin, nc, kd, input, link, s);
                        k::q80p_vhead(wp, xq, y1, kNin, rows, nc, kd, scr0, s, r0, (unsigned*) link);
                    }
                    ck(cudaMemsetAsync(flush, it, 256u << 20, s), "flush");
                    cudaEventRecord(e0, s);
                    if (v == 0) {
                        k::q80p_vhead(wp, xq, y1, kNin, rows, nc, kd, scr0, s, r0);
                    } else if (v == 1) {
                        k::q80p_vhead_merge(y1, rows, r0, nc, r1, kd, list, link, s);
                    } else {
                        k::q80p_vhead_fetch(input, link, xq1, kd1, kNin, nc, scr1, s);
                        k::q80p_vhead(wq, xq1, y0, kNin, r1, nc, kd1, scr1, s, -1, nullptr, (const unsigned*) link);
                        cudaEventRecord(em, s);
                        k::q80p_vhead_list(y0, r1, nc, kd1, scr1, list, link, s);
                    }
                    cudaEventRecord(e1, s);
                    cudaEventSynchronize(e1);
                    float ms;
                    cudaEventElapsedTime(&ms, e0, e1);
                    tot[v] += ms;
                    if (v == 2) {
                        cudaEventElapsedTime(&ms, em, e1);
                        tot[3] += ms;
                    }
                }
            const unsigned* lc = (const unsigned*) list;
            unsigned listed = 0;
            for (int c = 0; c < nc; ++c) listed += lc[c];
            std::printf("cold ncols %d split K20: first %d rows %6.1f us + merge %5.1f us | last %d rows screened + listed "
                        "%6.1f us (list %5.1f us, %.0f listed per column)\n", nc, r0, 1e3 * tot[0] / iters,
                        1e3 * tot[1] / iters, r1, 1e3 * tot[2] / iters, 1e3 * tot[3] / iters, (double) listed / nc);
        }
        cudaFree(wq);
        cudaFree(scr1);
        cudaFree(scr0);
        cudaFree(xq1);
        cudaFree(kd1);
        cudaFreeHost(link);
        cudaFreeHost(input);
        cudaFreeHost(list);
    }
    std::printf("%s\n", fail ? "q80_vhead_parity: FAIL" : "q80_vhead_parity: OK");
    return fail;
}
