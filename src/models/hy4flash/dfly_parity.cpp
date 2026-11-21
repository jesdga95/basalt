// src/models/hy4flash/dfly_parity.cpp - GPU parity of the DFly2 drafter against a double-precision CPU reference written
// from the equations in dfly.hpp: the four kernels, the tap, and the whole block pass with the sequential heads on small
// random weights (the real width E = 3072, the head screen's, with two layers, eight heads and a 300-token vocabulary) over
// random "target" caches and selections. Synthetic: no model, no checkpoint.
#include "basalt/core/loader.hpp"
#include "basalt/kernels/f16_bits.hpp"
#include "basalt/models/hy4flash/dfly_kernels.hpp"
#include "basalt/models/hy4flash/kernels.hpp"
#include "models/hy4flash/dfly.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace basalt::models::hy4flash;

static int g_fail = 0;
#define CHECK(c)                                                                      \
    do {                                                                              \
        if (!(c)) {                                                                   \
            std::fprintf(stderr, "dfly_parity: FAILED %s (line %d)\n", #c, __LINE__); \
            ++g_fail;                                                                 \
        }                                                                             \
    } while (0)

namespace {

std::mt19937 rng(1234);
float gauss(float sd) { return std::normal_distribution<float>(0.f, sd)(rng); }

uint16_t to_bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return (uint16_t) (u >> 16);
}
float from_bf16(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
uint16_t to_f16(float x) { return __half_as_ushort(__float2half_rn(x)); }
float from_f16(uint16_t b) { return __half2float(__ushort_as_half(b)); }

struct Mat {
    std::vector<uint16_t> bits;
    std::vector<float> v;   // the rounded values
    uint64_t rows = 0, cols = 0;
    std::vector<uint8_t> q8;   // a projection the drafter keeps in Q8_0: its blocks (the reference's product is then MMVQ's)
    std::vector<uint8_t> q4;   // one it keeps in Q4_0
};
Mat rand_mat(uint64_t rows, uint64_t cols, float sd) {
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.bits.resize(rows * cols);
    m.v.resize(rows * cols);
    for (size_t i = 0; i < m.bits.size(); ++i) {
        m.bits[i] = to_bf16(gauss(sd));
        m.v[i] = from_bf16(m.bits[i]);
    }
    return m;
}

template <class T>
T* upload(const std::vector<T>& h) {
    T* d = nullptr;
    cudaMalloc(&d, std::max<size_t>(h.size() * sizeof(T), 16));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
template <class T>
std::vector<T> download(const T* d, size_t n) {
    std::vector<T> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}

double max_rel(const std::vector<float>& got, const std::vector<double>& want) {
    double worst = 0, scale = 0;
    for (double w : want) scale = std::max(scale, std::fabs(w));
    for (size_t i = 0; i < want.size(); ++i) worst = std::max(worst, std::fabs(got[i] - want[i]) / (scale + 1e-12));
    return worst;
}

// ---- the kernels ---------------------------------------------------------------------------------------------------------

void test_kernels() {
    for (int H : {3, 32})
    for (int rows : {1, 3, 6, 8}) {
        const Mat kvb = rand_mat((uint64_t) H * kDflyKvbHead, kDflyRank, 0.05f);
        std::vector<float> q((size_t) rows * H * 256), olat((size_t) rows * H * kDflyRank);
        for (float& x : q) x = gauss(1.f);
        for (float& x : olat) x = gauss(1.f);
        uint16_t* dkvb = upload(kvb.bits);
        float *dq = upload(q), *dol = upload(olat), *dqa = nullptr, *dout = nullptr;
        cudaMalloc(&dqa, (size_t) rows * H * kDflyRank * 4);
        cudaMalloc(&dout, (size_t) rows * H * kDflyV * 4);
        dfly_absorb_k(dkvb, dq, H * 256, 256, rows, H, dqa, nullptr);
        dfly_expand_v(dkvb, dol, rows, H, dout, nullptr);
        std::vector<double> wa((size_t) rows * H * kDflyRank), wo((size_t) rows * H * kDflyV);
        for (int r = 0; r < rows; ++r)
            for (int h = 0; h < H; ++h) {
                for (int c = 0; c < kDflyRank; ++c) {
                    double s = 0;
                    for (int d = 0; d < kDflyNope; ++d)
                        s += (double) q[((size_t) r * H + h) * 256 + d] * kvb.v[((size_t) h * kDflyKvbHead + d) * kDflyRank + c];
                    wa[((size_t) r * H + h) * kDflyRank + c] = s;
                }
                for (int v = 0; v < kDflyV; ++v) {
                    double s = 0;
                    for (int c = 0; c < kDflyRank; ++c)
                        s += (double) olat[((size_t) r * H + h) * kDflyRank + c] * kvb.v[((size_t) h * kDflyKvbHead + kDflyNope + v) * kDflyRank + c];
                    wo[((size_t) r * H + h) * kDflyV + v] = s;
                }
            }
        const double ea = max_rel(download(dqa, wa.size()), wa), eo = max_rel(download(dout, wo.size()), wo);
        std::printf("kernels heads %d rows %d: absorb_k %.2e expand_v %.2e\n", H, rows, ea, eo);
        CHECK(ea < 1e-5 && eo < 1e-5);
        cudaFree(dkvb); cudaFree(dq); cudaFree(dol); cudaFree(dqa); cudaFree(dout);
    }
    {   // absorb_k at the drafter's geometry: microseconds a call and the key rows' bytes over that time
        const int H = 32;
        const Mat kvb = rand_mat((uint64_t) H * kDflyKvbHead, kDflyRank, 0.05f);
        std::vector<float> q((size_t) 8 * H * 256, 0.5f);
        uint16_t* dkvb = upload(kvb.bits);
        float *dq = upload(q), *dqa = nullptr;
        cudaMalloc(&dqa, (size_t) 8 * H * kDflyRank * 4);
        cudaStream_t bs = nullptr;
        cudaStreamCreate(&bs);
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        for (int rows : {1, 2, 4, 8}) {
            constexpr int kCalls = 200;
            dfly_absorb_k(dkvb, dq, H * 256, 256, rows, H, dqa, nullptr);
            cudaEventRecord(e0, bs);
            for (int i = 0; i < kCalls; ++i) dfly_absorb_k(dkvb, dq, H * 256, 256, rows, H, dqa, bs);
            cudaEventRecord(e1, bs);
            cudaEventSynchronize(e1);
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            const double us = 1e3 * ms / kCalls, bytes = (double) H * kDflyNope * kDflyRank * 2;
            std::printf("absorb_k heads %d rows %d: %.1f us a call, %.0f GB/s of key rows\n", H, rows, us, bytes / us / 1e3);
        }
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaStreamDestroy(bs);
        cudaFree(dkvb); cudaFree(dq); cudaFree(dqa);
    }
    // gather (by cells, and by position) and the gated add
    std::vector<uint16_t> cache(50 * kDflyRowWidth);
    for (uint16_t& x : cache) x = to_bf16(gauss(1.f));
    std::vector<int32_t> cells = {49, 0, 7, 7, 31};
    uint16_t *dc = upload(cache), *dout = nullptr;
    int32_t* dcells = upload(cells);
    cudaMalloc(&dout, 50 * kDflyRowWidth * 2);
    dfly_gather_rows(dc, dcells, 5, kDflyRowWidth, dout, nullptr);
    std::vector<uint16_t> got = download(dout, 5 * kDflyRowWidth);
    bool same = true;
    for (int j = 0; j < 5; ++j)
        for (int c = 0; c < kDflyRowWidth; ++c) same = same && got[(size_t) j * kDflyRowWidth + c] == cache[(size_t) cells[(size_t) j] * kDflyRowWidth + c];
    CHECK(same);
    dfly_gather_rows(dc, nullptr, 20, kDflyRowWidth, dout, nullptr);
    got = download(dout, 20 * kDflyRowWidth);
    CHECK(std::equal(got.begin(), got.end(), cache.begin()));
    {   // --kv fp8: rows written by mla_prepare, gathered and decoded to BF16 exactly
        const int n8 = 50;
        MlaArgs a;
        std::vector<float> q((size_t) n8 * a.heads * 256), qa((size_t) n8 * a.heads * kDflyRank), kv((size_t) n8 * kDflyRowWidth), nw(kDflyRank);
        for (float& x : q) x = gauss(1.f);
        for (float& x : qa) x = gauss(1.f);
        for (float& x : kv) x = gauss(3.f);
        for (float& x : nw) x = 1.f + gauss(0.5f);
        float *dq = upload(q), *dqa = upload(qa), *dkv = upload(kv), *dnw = upload(nw), *dqc = nullptr;
        cudaMalloc(&dqc, (size_t) n8 * a.heads * kDflyRowWidth * 4);
        uint8_t* d8 = nullptr;
        cudaMalloc(&d8, (size_t) n8 * kFp8RowBytes);
        mla_prepare(dq, dqa, dkv, dnw, a, 0, n8, 0, dqc, d8, nullptr);
        std::vector<uint8_t> rows8 = download(d8, (size_t) n8 * kFp8RowBytes);
        dfly_gather_rows_fp8(d8, dcells, 5, dout, nullptr);
        got = download(dout, 5 * kDflyRowWidth);
        bool exact = true;
        for (int j = 0; j < 5; ++j) {
            const uint8_t* row = &rows8[(size_t) cells[(size_t) j] * kFp8RowBytes];
            float sc[4];
            std::memcpy(sc, row + kDflyRank, sizeof sc);
            for (int c = 0; c < kDflyRowWidth; ++c) {
                uint16_t want;
                if (c < kDflyRank) {
                    const uint8_t b = row[c];
                    const int e = (b >> 3) & 15, m = b & 7;
                    double v = e ? std::ldexp(1.0 + m / 8.0, e - 7) : std::ldexp((double) m, -9);
                    v = ((b & 0x80) ? -v : v) * sc[c >> 7];
                    const float f = (float) v;
                    exact = exact && from_f16(to_f16(f)) == f;
                    want = to_f16(f);
                } else {
                    uint16_t b;
                    std::memcpy(&b, row + kDflyRank + 16 + 2 * (c - kDflyRank), 2);
                    want = to_f16(from_bf16(b));
                }
                exact = exact && got[(size_t) j * kDflyRowWidth + c] == want;
            }
        }
        std::printf("gather fp8 rows: %s\n", exact ? "exact" : "FAIL");
        CHECK(exact);
        cudaFree(dq); cudaFree(dqa); cudaFree(dkv); cudaFree(dnw); cudaFree(dqc); cudaFree(d8);
    }
    std::vector<float> gotf;
    std::vector<float> a(300), b(300), g(300);
    for (int i = 0; i < 300; ++i) { a[(size_t) i] = gauss(1.f); b[(size_t) i] = gauss(1.f); g[(size_t) i] = gauss(1.f); }
    float *da = upload(a), *db = upload(b), *dg = upload(g), *dr = nullptr;
    cudaMalloc(&dr, 300 * 4);
    dfly_gate_add(da, db, dg, 300, dr, nullptr);
    gotf = download(dr, 300);
    bool exact = true;
    for (int i = 0; i < 300; ++i) exact = exact && std::fabs(gotf[(size_t) i] - (a[(size_t) i] + g[(size_t) i] * b[(size_t) i])) < 1e-6f;   // the device fuses the multiply-add
    CHECK(exact);
    cudaFree(dc); cudaFree(dcells); cudaFree(dout); cudaFree(da); cudaFree(db); cudaFree(dg); cudaFree(dr);
}

// ---- the tap -------------------------------------------------------------------------------------------------------------

void test_tap() {
    DflyDims d;
    d.layers = 2; d.E = 16; d.idx_layers = {3, 5}; d.kv_layers = {3, 5};
    std::string err;
    auto tap = DflyTap::create(d, 32, err);
    CHECK(tap != nullptr);
    tap->begin_window(100);
    std::vector<float> h0(16), h1(16);
    for (int i = 0; i < 16; ++i) { h0[(size_t) i] = (float) i; h1[(size_t) i] = (float) -i; }
    float *dh0 = upload(h0), *dh1 = upload(h1);
    std::vector<int32_t> cells = {5, 9, 2, 40};
    int32_t* dcells = upload(cells);
    tap->hidden(0, dh0, nullptr);
    tap->hidden(1, dh1, nullptr);
    tap->selection(3, 0, dcells, 4, nullptr);
    tap->selection(4, 0, dcells, 4, nullptr);   // a layer no draft layer reads: ignored
    CHECK(tap->source_row(101) == -1);          // layer 5's selection is missing
    tap->selection(5, 0, nullptr, -1, nullptr);
    CHECK(tap->source_row(101) == 0 && tap->source_row(102) == -1 && tap->source_row(100) == -1);
    cudaDeviceSynchronize();
    CHECK(tap->count(0, 0) == 4 && tap->count(1, 0) == -1);
    CHECK(download(tap->cells(0, 0), 4) == cells);
    CHECK(download(tap->hidden_row(1), 16) == h1);
    tap->selection(3, 1, dcells, 33, nullptr);   // more cells than it keeps: not kept
    tap->selection(5, 1, nullptr, -1, nullptr);
    CHECK(tap->source_row(102) == -1);
    tap->begin_window(200);
    CHECK(tap->source_row(101) == -1);
    tap->hidden(0, dh0, nullptr);
    tap->selection(3, 0, dcells, 2, nullptr);
    tap->selection(5, 0, dcells, 3, nullptr);
    CHECK(tap->source_row(201) == 0);
    tap->invalidate();
    CHECK(tap->source_row(201) == -1);
    cudaFree(dh0); cudaFree(dh1); cudaFree(dcells);
}

// ---- the block pass ------------------------------------------------------------------------------------------------------

struct Ref {
    const DflyDims& d;
    std::map<std::string, Mat> m;                  // BF16 matrices
    std::map<std::string, std::vector<float>> f;   // FP32 vectors
    const std::vector<float>& embed;               // [vocab][E]
    const Mat& head;

    std::vector<double> mv(const Mat& w, const std::vector<double>& x) const {
        if (!w.q8.empty()) return mvq(w, x);
        if (!w.q4.empty()) return mvq4(w, x);
        std::vector<double> y(w.rows);
        for (uint64_t r = 0; r < w.rows; ++r) {
            double s = 0;
            for (uint64_t c = 0; c < w.cols; ++c) s += (double) w.v[r * w.cols + c] * x[c];
            y[r] = s;
        }
        return y;
    }
    // a Q8_0 product as MMVQ makes it: x in q8_1 blocks (q = round(x / d), d = amax / 127 kept as fp16), integer dots
    // per block, scaled by both blocks' fp16 d
    std::vector<double> mvq(const Mat& w, const std::vector<double>& x) const {
        const uint64_t nb = w.cols / 32;
        std::vector<int> qx(w.cols);
        std::vector<double> dx(nb);
        for (uint64_t b = 0; b < nb; ++b) {
            float amax = 0.f;
            for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs((float) x[b * 32 + i]));
            const float d = amax / 127.f;
            for (int i = 0; i < 32; ++i) qx[b * 32 + i] = amax == 0.f ? 0 : (int) std::round((float) x[b * 32 + i] / d);
            dx[b] = basalt::kernels::f32_from_f16(basalt::kernels::f16_from_f32(d));
        }
        std::vector<double> y(w.rows);
        for (uint64_t r = 0; r < w.rows; ++r) {
            double s = 0;
            for (uint64_t b = 0; b < nb; ++b) {
                const uint8_t* blk = w.q8.data() + (r * nb + b) * 34;
                uint16_t dh;
                std::memcpy(&dh, blk, 2);
                int dot = 0;
                for (int i = 0; i < 32; ++i) dot += (int) (int8_t) blk[2 + i] * qx[b * 32 + i];
                s += (double) basalt::kernels::f32_from_f16(dh) * dx[b] * dot;
            }
            y[r] = s;
        }
        return y;
    }
    // a Q4_0 product as MMVQ makes it: x in q8_1 blocks as above, per block d4 (d8 sum(nibble q) - 8 s), s the block's sum
    // of x kept as fp16 (q8_1's second half)
    std::vector<double> mvq4(const Mat& w, const std::vector<double>& x) const {
        const uint64_t nb = w.cols / 32;
        std::vector<int> qx(w.cols);
        std::vector<double> dx(nb), sx(nb);
        for (uint64_t b = 0; b < nb; ++b) {
            float amax = 0.f;
            double sum = 0;
            for (int i = 0; i < 32; ++i) {
                amax = std::max(amax, std::fabs((float) x[b * 32 + i]));
                sum += x[b * 32 + i];
            }
            const float d = amax / 127.f;
            for (int i = 0; i < 32; ++i) qx[b * 32 + i] = amax == 0.f ? 0 : (int) std::round((float) x[b * 32 + i] / d);
            dx[b] = basalt::kernels::f32_from_f16(basalt::kernels::f16_from_f32(d));
            sx[b] = basalt::kernels::f32_from_f16(basalt::kernels::f16_from_f32((float) sum));
        }
        std::vector<double> y(w.rows);
        for (uint64_t r = 0; r < w.rows; ++r) {
            double s = 0;
            for (uint64_t b = 0; b < nb; ++b) {
                const uint8_t* blk = w.q4.data() + (r * nb + b) * 18;
                uint16_t dh;
                std::memcpy(&dh, blk, 2);
                int dot = 0;
                for (int i = 0; i < 16; ++i) dot += (blk[2 + i] & 15) * qx[b * 32 + i] + (blk[2 + i] >> 4) * qx[b * 32 + 16 + i];
                s += (double) basalt::kernels::f32_from_f16(dh) * (dx[b] * dot - 8.0 * sx[b]);
            }
            y[r] = s;
        }
        return y;
    }
    std::vector<double> rms(const std::vector<double>& x, const std::vector<float>& w) const {
        double ss = 0;
        for (double v : x) ss += v * v;
        const double inv = 1.0 / std::sqrt(ss / (double) x.size() + d.rms_eps);
        std::vector<double> y(x.size());
        for (size_t i = 0; i < x.size(); ++i) y[i] = x[i] * inv * w[i];
        return y;
    }
    static void rot(double* x, int pos, double base) {   // pairs (2i, 2i + 1) of 64
        for (int i = 0; i < 32; ++i) {
            const double th = pos * std::pow(base, -(2.0 * i) / 64.0), c = std::cos(th), s = std::sin(th);
            const double x0 = x[2 * i], x1 = x[2 * i + 1];
            x[2 * i] = x0 * c - x1 * s;
            x[2 * i + 1] = x0 * s + x1 * c;
        }
    }
};

void test_block() {
    DflyDims d;
    d.layers = 2; d.E = 3072; d.q_lora = 128; d.heads = 8; d.ffn = 192; d.hc_ffn = 96; d.vocab = 300; d.mask_id = 299;
    d.topk = 64; d.rms_eps = 1e-5f; d.rope_base = 1e7f;
    d.kv_layers = {4, 6};
    d.idx_layers = {4, 6};
    const int E = d.E, H = d.heads, ctx_rows = 200, sel_cap = 256, pos = 120;

    // weights
    std::map<std::string, Mat> rm;
    std::map<std::string, std::vector<float>> rf;
    DflyWeights w;
    std::string err;
    CHECK(w.allocate(d, err));
    for (const DflySpec& s : dfly_specs(d)) {
        if (s.f32) {
            std::vector<float> v((size_t) s.elems());
            const bool sink = s.name.find("sink") != std::string::npos, gate = s.name.find("gate") != std::string::npos || s.name.find("confidence") != std::string::npos;
            for (float& x : v) x = sink ? gauss(1.f) : gate ? gauss(0.3f) : 1.f + gauss(0.1f);
            cudaMemcpy(w.slot(s.name), v.data(), v.size() * 4, cudaMemcpyHostToDevice);
            rf[s.name] = v;
        } else {
            Mat mat = rand_mat(s.shape[0], s.shape[1], 1.0f / std::sqrt((float) s.shape[1]));
            if (s.q8) {   // what the loader makes of it (core::Conv::Bf16ToQ8_0)
                mat.q8.resize(s.bytes());
                basalt::core::quantize_bf16_q8_0(mat.bits.data(), mat.bits.size(), mat.q8.data());
                cudaMemcpy(w.slot(s.name), mat.q8.data(), mat.q8.size(), cudaMemcpyHostToDevice);
            } else if (s.q4) {   // core::Conv::Bf16ToQ4_0
                mat.q4.resize(s.bytes());
                basalt::core::quantize_bf16_q4_0(mat.bits.data(), mat.bits.size(), mat.q4.data());
                cudaMemcpy(w.slot(s.name), mat.q4.data(), mat.q4.size(), cudaMemcpyHostToDevice);
            } else {
                cudaMemcpy(w.slot(s.name), mat.bits.data(), mat.bits.size() * 2, cudaMemcpyHostToDevice);
            }
            rm[s.name] = std::move(mat);
        }
    }
    Mat head = rand_mat((uint64_t) d.vocab, (uint64_t) E, 2.0f / std::sqrt((float) E));
    head.q8.resize((size_t) d.vocab * E / 32 * 34);   // the drafts' head, as the loader rounds it (core::Conv::Bf16ToQ8_0)
    basalt::core::quantize_bf16_q8_0(head.bits.data(), head.bits.size(), head.q8.data());
    void* dhead_q8 = upload(head.q8);
    w.head_q8 = dhead_q8;
    CHECK(w.pack_screen(nullptr, err));
    std::vector<float> embed((size_t) d.vocab * E);
    for (float& x : embed) x = gauss(1.f);
    Ref r2{d, std::move(rm), std::move(rf), embed, head};

    // the target: caches, the tap's row, selections (a list for layer 0, every position for layer 1)
    std::vector<std::vector<uint16_t>> cache(2, std::vector<uint16_t>((size_t) ctx_rows * kDflyRowWidth));
    for (auto& c : cache)
        for (uint16_t& x : c) x = to_f16(gauss(1.f));
    std::vector<uint16_t*> dcache = {upload(cache[0]), upload(cache[1])};
    std::vector<int32_t> cells;
    for (int i = 0; i < 37; ++i) cells.push_back(std::uniform_int_distribution<int>(0, pos - 1)(rng));
    cells.push_back(pos - 1);
    std::vector<float> hid(E);
    for (float& x : hid) x = gauss(1.f);
    float* dhid = upload(hid);
    int32_t* dcells = upload(cells);
    auto tap = DflyTap::create(d, sel_cap, err);
    tap->begin_window(pos - 1);
    tap->hidden(0, dhid, nullptr);
    tap->selection(4, 0, dcells, (int) cells.size(), nullptr);
    tap->selection(6, 0, nullptr, -1, nullptr);
    CHECK(tap->source_row(pos) == 0);

    DflyRunner::Env env;
    env.w = &w;
    env.sel_cap = sel_cap;
    env.embed = [&](int32_t t, float* o) { std::memcpy(o, embed.data() + (size_t) t * E, (size_t) E * 4); return t >= 0 && t < d.vocab; };
    auto run = DflyRunner::create(env, err);
    cudaStream_t st = nullptr;   // the block pass's MMVQ runs on an explicit stream, as the session's does
    CHECK(cudaStreamCreate(&st) == cudaSuccess);
    CHECK(run != nullptr);
    if (!run) return;

    for (int n : {1, 3, 5}) {
        const int32_t anchor = 17;
        std::vector<const void*> caches = {dcache[0], dcache[1]};
        const bool began = run->begin(anchor, pos, n, *tap, 0, caches.data(), st, err);
        if (!began) std::printf("begin: %s\n", err.c_str());
        CHECK(began);
        int32_t ids[5];
        float conf[5];
        int made = 0;
        CHECK(run->finish(ids, conf, made, st, err) && made == n);
        // the same block, ended before draft n - 1 by `need` (asked with each draft's chance before its head): the drafts
        // before it and every chance asked are the same bits
        if (n > 1) {
            CHECK(run->begin(anchor, pos, n, *tap, 0, caches.data(), st, err));
            int32_t ids2[5];
            float conf2[5];
            int asked = 0, made2 = 0;
            const auto need = [&](const float* c, int j) {
                CHECK(c == conf2 && j == asked);
                ++asked;
                return j < n - 1;
            };
            CHECK(run->finish(ids2, conf2, made2, st, err, nullptr, need) && made2 == n - 1 && asked == n);
            CHECK(std::equal(ids, ids + n - 1, ids2) && std::memcmp(conf, conf2, sizeof(float) * (size_t) n) == 0);
        }

        // the reference: the same equations in double
        const int R = n + 1;
        std::vector<std::vector<double>> X((size_t) R, std::vector<double>((size_t) E));
        {
            std::vector<double> e(E), hn(E), cat;
            for (int i = 0; i < E; ++i) e[(size_t) i] = embed[(size_t) anchor * E + i];
            std::vector<double> hh(hid.begin(), hid.end());
            const auto en = r2.rms(e, r2.f["mtp_fusion.enorm.weight"]), hnv = r2.rms(hh, r2.f["mtp_fusion.hnorm.weight"]);
            cat = en;
            cat.insert(cat.end(), hnv.begin(), hnv.end());
            const auto fused = r2.mv(r2.m["mtp_fusion.eh_proj.weight"], cat);
            for (int i = 0; i < E; ++i) X[0][(size_t) i] = e[(size_t) i] + r2.f["mtp_fusion.gate"][(size_t) i] * fused[(size_t) i];
            for (int r = 1; r < R; ++r)
                for (int i = 0; i < E; ++i) X[(size_t) r][(size_t) i] = embed[(size_t) d.mask_id * E + i];
        }
        for (int l = 0; l < d.layers; ++l) {
            const std::string p = std::to_string(l) + ".";
            std::vector<int> rowsel;   // the target rows this layer reads
            if (l == 0) rowsel.assign(cells.begin(), cells.end());
            else for (int i = 0; i < pos; ++i) rowsel.push_back(i);
            std::vector<std::vector<double>> q((size_t) R), kvr((size_t) R), xn((size_t) R);
            for (int r = 0; r < R; ++r) {
                xn[(size_t) r] = r2.rms(X[(size_t) r], r2.f[p + "input_layernorm.weight"]);
                q[(size_t) r] = r2.mv(r2.m[p + "self_attn.q_b_proj.weight"], r2.rms(r2.mv(r2.m[p + "self_attn.q_a_proj.weight"], xn[(size_t) r]), r2.f[p + "self_attn.q_a_layernorm.weight"]));
                kvr[(size_t) r] = r2.mv(r2.m[p + "self_attn.kv_a_proj_with_mqa.weight"], xn[(size_t) r]);
            }
            // the block rows' K = [normed latent | rotated rope]
            std::vector<std::vector<double>> krow((size_t) R);
            for (int r = 0; r < R; ++r) {
                std::vector<double> lat(kvr[(size_t) r].begin(), kvr[(size_t) r].begin() + 512);
                krow[(size_t) r] = r2.rms(lat, r2.f[p + "self_attn.kv_a_layernorm.weight"]);
                std::vector<double> rp(kvr[(size_t) r].begin() + 512, kvr[(size_t) r].end());
                Ref::rot(rp.data(), pos + r, d.rope_base);
                krow[(size_t) r].insert(krow[(size_t) r].end(), rp.begin(), rp.end());
                for (double& x : krow[(size_t) r]) x = from_f16(to_f16((float) x));   // the block's rows are held as fp16, as the target's cache
            }
            const Mat& kvb = r2.m[p + "self_attn.kv_b_proj.weight"];
            std::vector<std::vector<double>> vout((size_t) R, std::vector<double>((size_t) H * 256));
            for (int r = 0; r < R; ++r)
                for (int h = 0; h < H; ++h) {
                    std::vector<double> qc(576);
                    for (int c = 0; c < 512; ++c) {
                        double s = 0;
                        for (int dd = 0; dd < 192; ++dd) s += q[(size_t) r][(size_t) h * 256 + dd] * kvb.v[((size_t) h * 448 + dd) * 512 + c];
                        qc[(size_t) c] = s;
                    }
                    std::vector<double> qp(q[(size_t) r].begin() + h * 256 + 192, q[(size_t) r].begin() + h * 256 + 256);
                    Ref::rot(qp.data(), pos + r, d.rope_base);
                    std::copy(qp.begin(), qp.end(), qc.begin() + 512);
                    std::vector<const double*> rowsp;
                    std::vector<std::vector<double>> tmp;
                    std::vector<double> sc;
                    std::vector<std::vector<double>> rows;
                    for (int c : rowsel) {
                        std::vector<double> rw(576);
                        for (int k = 0; k < 576; ++k) rw[(size_t) k] = from_f16(cache[(size_t) l][(size_t) c * 576 + k]);
                        rows.push_back(rw);
                    }
                    for (int b = 0; b <= r; ++b) rows.push_back(krow[(size_t) b]);
                    const double sink = r2.f[p + "self_attn.learnable_sink_param"][(size_t) h];
                    double m = sink;
                    for (auto& rw : rows) {
                        double s = 0;
                        for (int c = 0; c < 576; ++c) s += qc[(size_t) c] * rw[(size_t) c];
                        sc.push_back(s / 16.0);
                        m = std::max(m, sc.back());
                    }
                    double den = std::exp(sink - m);
                    for (double& s : sc) { s = std::exp(s - m); den += s; }
                    std::vector<double> ol(512, 0.0);
                    for (size_t j = 0; j < rows.size(); ++j)
                        for (int c = 0; c < 512; ++c) ol[(size_t) c] += sc[j] / den * rows[j][(size_t) c];
                    for (int v = 0; v < 256; ++v) {
                        double s = 0;
                        for (int c = 0; c < 512; ++c) s += ol[(size_t) c] * kvb.v[((size_t) h * 448 + 192 + v) * 512 + c];
                        vout[(size_t) r][(size_t) h * 256 + v] = s;
                    }
                }
            for (int r = 0; r < R; ++r) {
                const auto g = r2.mv(r2.m[p + "self_attn.linear_gate.weight"], xn[(size_t) r]);
                for (size_t i = 0; i < g.size(); ++i) vout[(size_t) r][i] *= 1.0 / (1.0 + std::exp(-g[i]));
                const auto o = r2.rms(r2.mv(r2.m[p + "self_attn.o_proj.weight"], vout[(size_t) r]), r2.f[p + "self_attn.post_proj_norm.weight"]);
                std::vector<double> x1(E);
                for (int i = 0; i < E; ++i) x1[(size_t) i] = X[(size_t) r][(size_t) i] + o[(size_t) i];
                const auto xn2 = r2.rms(x1, r2.f[p + "post_attention_layernorm.weight"]);
                const auto gg = r2.mv(r2.m[p + "mlp.gate_proj.weight"], xn2), uu = r2.mv(r2.m[p + "mlp.up_proj.weight"], xn2);
                std::vector<double> hh(gg.size());
                for (size_t i = 0; i < gg.size(); ++i) hh[i] = gg[i] / (1.0 + std::exp(-gg[i])) * uu[i];
                const auto dn = r2.mv(r2.m[p + "mlp.down_proj.weight"], hh);
                for (int i = 0; i < E; ++i) X[(size_t) r][(size_t) i] = x1[(size_t) i] + dn[(size_t) i];
            }
        }
        // the heads, along the device's own tokens
        int32_t prev = anchor;
        double worst_conf = 0;
        int checked = 0;
        for (int j = 0; j < n; ++j) {
            const auto z = r2.rms(X[(size_t) j + 1], r2.f["final_norm.weight"]);
            std::vector<double> pe(E);
            for (int i = 0; i < E; ++i) pe[(size_t) i] = embed[(size_t) prev * E + i];
            auto cat = r2.rms(z, r2.f["hidden_correction.hidden_norm.weight"]);
            const auto pn = r2.rms(pe, r2.f["hidden_correction.embed_norm.weight"]);
            cat.insert(cat.end(), pn.begin(), pn.end());
            const auto gg = r2.mv(r2.m["hidden_correction.gate_proj.weight"], cat), uu = r2.mv(r2.m["hidden_correction.up_proj.weight"], cat);
            std::vector<double> hh(gg.size());
            for (size_t i = 0; i < gg.size(); ++i) hh[i] = gg[i] / (1.0 + std::exp(-gg[i])) * uu[i];
            const auto dn = r2.mv(r2.m["hidden_correction.down_proj.weight"], hh);
            std::vector<double> hc(E);
            for (int i = 0; i < E; ++i) hc[(size_t) i] = z[(size_t) i] + dn[(size_t) i];
            const auto lg = r2.mv(head, hc);
            int best = 0, second = -1;
            for (int v = 1; v < d.vocab; ++v)
                if (lg[(size_t) v] > lg[(size_t) best]) { second = best; best = v; }
                else if (second < 0 || lg[(size_t) v] > lg[(size_t) second]) second = v;
            if (second < 0) second = best == 0 ? 1 : 0;
            if (lg[(size_t) best] - lg[(size_t) second] > 1e-3) {
                ++checked;
                CHECK(ids[j] == best);
            }
            double craw = r2.f["confidence_head.proj.bias"][0];
            for (int i = 0; i < E; ++i) craw += (double) r2.f["confidence_head.proj.weight"][(size_t) i] * hc[(size_t) i];
            worst_conf = std::max(worst_conf, std::fabs(conf[j] - 1.0 / (1.0 + std::exp(-craw))));
            prev = ids[j];
        }
        std::printf("block n=%d: %d of %d tokens decisive and equal, confidence differs by %.2e\n", n, checked, n, worst_conf);
        // the products are MMVQ's exactly (one row agrees to 1e-7), but the reference's activations are doubles where the
        // card's are floats, so now and then a value on a q8_1 rounding boundary takes the next code; such flips compound
        // over the block's later rows (7e-4 at three, 1.2e-2 at five). The tokens must still agree: the target verifies.
        CHECK(worst_conf < 3e-2);
        CHECK(checked >= n - 1);
    }
    cudaStreamDestroy(st);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError() == cudaSuccess);
}

}  // namespace

int main() {
    test_kernels();
    test_tap();
    test_block();
    if (g_fail) return 1;
    std::printf("dfly_parity: ok\n");
    return 0;
}
