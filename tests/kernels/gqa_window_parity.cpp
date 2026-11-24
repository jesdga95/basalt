// tests/kernels/gqa_window_parity.cpp - gqa_window_attn (basalt/kernels/gqa_attn.hpp) at Inkling 2's DSpark geometry
// (24 x 128 query heads, 8 KV heads: a block of 8 rows is 24 rows of a KV group, two 16-row tiles; 32 rows are six)
// against a CPU reference in double precision, over block and per-row windows, a full and a wrapped sliding cache,
// widths 1..32 and odd depths, short calls (one tile a block) and deep ones (two).
//
// Checks: max abs and rel error to the reference; a row's bits are the same at every width of a window with the same
// end, and in a one-row window (another place in another tile); the outputs' bits against a frozen hash (a change that
// must not move a bit leaves it as it is).  The inputs come from mt19937's raw output alone, so they are the same on
// every platform.
// --bench: the full-attention layer's call (a window of 1..32 rows over every cell) at 4K..160K cells, us per call and
// the K/V bytes read once per call over that time, and the sliding layer's call.  Exit 77 without a GPU.
#include "basalt/kernels/f16_bits.hpp"
#include "basalt/kernels/gqa_attn.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int NH = 24, NKV = 8, HD = 128;
constexpr int BW = 32;   // the widest window checked and hashed (fixed, so the hash does not move with kGqaMaxWindow)
static_assert(BW <= basalt::kernels::kGqaMaxWindow, "gqa_window_parity: windows of BW rows");

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

template <class T> T* dev(size_t n) {
    void* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc");
    return (T*) p;
}

float h2f(uint16_t h) { return basalt::kernels::f32_from_f16(h); }

// a unit-variance draw from four 24-bit uniforms of the generator's raw output (Irwin-Hall), exact up to the last rounding
float draw(std::mt19937& g) {
    double s = 0;
    for (int i = 0; i < 4; ++i) s += (double) (g() >> 8) * (1.0 / 16777216.0);
    return (float) ((s - 2.0) * 1.7320508075688772);
}

struct Cache {
    int64_t cap = 0;
    std::vector<uint16_t> k, v;
    uint16_t *dk = nullptr, *dv = nullptr;
};

Cache make_cache(int64_t cap, std::mt19937& rng) {
    Cache c;
    c.cap = cap;
    c.k.resize((size_t) cap * NKV * HD);
    c.v.resize(c.k.size());
    for (auto& x : c.k) x = basalt::kernels::f16_from_f32(draw(rng));
    for (auto& x : c.v) x = basalt::kernels::f16_from_f32(draw(rng));
    c.dk = dev<uint16_t>(c.k.size());
    c.dv = dev<uint16_t>(c.v.size());
    ck(cudaMemcpy(c.dk, c.k.data(), c.k.size() * 2, cudaMemcpyHostToDevice), "copy");
    ck(cudaMemcpy(c.dv, c.v.data(), c.v.size() * 2, cudaMemcpyHostToDevice), "copy");
    return c;
}

// one query row: positions [lo, hi), cell p % cap
void ref_row(const float* q, const Cache& c, int64_t lo, int64_t hi, double* out) {
    for (int h = 0; h < NH; ++h) {
        const int kvh = h / (NH / NKV);
        std::vector<double> s;
        double mx = -1e300;
        for (int64_t p = lo; p < hi; ++p) {
            const size_t cell = (size_t) (p % c.cap);
            double d = 0;
            for (int i = 0; i < HD; ++i) d += (double) q[h * HD + i] * h2f(c.k[(cell * NKV + kvh) * HD + i]);
            s.push_back(d / std::sqrt((double) HD));
            mx = std::max(mx, s.back());
        }
        double l = 0;
        for (auto& x : s) { x = std::exp(x - mx); l += x; }
        for (int i = 0; i < HD; ++i) {
            double a = 0;
            for (size_t j = 0; j < s.size(); ++j) a += s[j] * h2f(c.v[(((size_t) ((lo + (int64_t) j) % c.cap)) * NKV + kvh) * HD + i]);
            out[h * HD + i] = a / l;
        }
    }
}

struct Case {
    int64_t pos, window;   // window <= 0: every position (the full layer's pos + BW)
    bool block, ring;
};

}  // namespace

int main(int argc, char** argv) {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("no GPU\n");
        return 77;
    }
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    int fails = 0;
    std::mt19937 rng(23);
    cudaStream_t S;
    ck(cudaStreamCreate(&S), "stream");

    // a full cache (no wrap) and a sliding layer's ring (its window, a window's rows and 48 more), the DSpark's
    const int64_t full_cap = 13000, ring_window = 1025, ring_cap = ring_window + 64;
    Cache full = make_cache(full_cap, rng), ring = make_cache(ring_cap, rng);
    std::vector<float> hq((size_t) BW * NH * HD);
    for (auto& x : hq) x = 2.0f * draw(rng);   // peaked rows: scores of a few units
    float* dq = dev<float>(hq.size());
    float* dout = dev<float>(hq.size());
    float* dwide = dev<float>(hq.size());
    ck(cudaMemcpy(dq, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice), "copy");
    float* dscr = dev<float>(basalt::kernels::gqa_decode_scratch_floats(full_cap, BW, NH, HD));
    std::vector<float> got(hq.size()), wide(hq.size());
    unsigned long long hash = 1469598103934665603ull;
    const auto fold = [&](const std::vector<float>& v, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            unsigned u;
            std::memcpy(&u, &v[i], 4);
            hash = (hash ^ u) * 1099511628211ull;
        }
    };

    // a split of a deep window holds 4+ chunks (two tiles a block); 12345 / 9018 starts its rows' windows on a chunk
    const Case cases[] = {
        {0, 0, true, false},      {1, 0, true, false},       {31, 0, true, false},      {33, 0, true, false},
        {1000, 0, true, false},   {4097, 0, true, false},    {9001, 0, true, false},    {12345, 0, true, false},
        {777, 0, false, false},   {777, 500, false, false},  {12345, 9018, false, false},
        {3000, ring_window, true, true}, {5001, ring_window, true, true}, {4321, ring_window, false, true},
    };
    double worst = 0, worst_rel = 0;
    int width_diff = 0, checked = 0, place_diff = 0, placed = 0;
    for (const Case& cs : cases) {
        const Cache& c = cs.ring ? ring : full;
        const int64_t hi = cs.pos + BW;   // every width ends where the widest window does: a row's range is the same at any T
        const int64_t window = cs.window > 0 ? cs.window : cs.pos + BW;
        const int64_t lo_all = std::max<int64_t>(0, cs.pos - window + 1);
        const int64_t scells = std::min<int64_t>(c.cap, hi - lo_all + 64);
        basalt::kernels::gqa_window_attn(dq, c.dk, c.dv, NH, NKV, HD, cs.pos, BW, hi, window, c.cap, dscr, scells, dwide, S, cs.block);
        ck(cudaStreamSynchronize(S), "window attention");
        ck(cudaMemcpy(wide.data(), dwide, wide.size() * 4, cudaMemcpyDeviceToHost), "copy");
        std::vector<double> want((size_t) NH * HD);
        for (int t = 0; t < BW; ++t) {
            const int64_t lo = std::max<int64_t>(0, cs.pos + (cs.block ? 0 : t) - window + 1);
            ref_row(&hq[(size_t) t * NH * HD], c, lo, hi, want.data());
            for (int j = 0; j < NH * HD; ++j) {
                const double e = std::fabs(wide[(size_t) t * NH * HD + j] - want[(size_t) j]);
                worst = std::max(worst, e);
                worst_rel = std::max(worst_rel, e / std::max(std::fabs(want[(size_t) j]), 1e-2));
            }
        }
        for (int T = 1; T <= BW; ++T) {
            basalt::kernels::gqa_window_attn(dq, c.dk, c.dv, NH, NKV, HD, cs.pos, T, hi, window, c.cap, dscr, scells, dout, S, cs.block);
            ck(cudaStreamSynchronize(S), "window attention");
            ck(cudaMemcpy(got.data(), dout, (size_t) T * NH * HD * 4, cudaMemcpyDeviceToHost), "copy");
            fold(got, (size_t) T * NH * HD);
            if (std::memcmp(got.data(), wide.data(), (size_t) T * NH * HD * 4)) ++width_diff;
            ++checked;
        }
        // each row of a deep window alone: one tile from its own first position, its heads at other places of it; the
        // same range and the same chunk splits (the window's rows start within one chunk)
        if (!cs.ring && scells >= 4 * 64 * 32 && (cs.block || (lo_all % 32 == 0 && cs.window > 0))) {
            for (int t = 0; t < BW; ++t) {
                basalt::kernels::gqa_window_attn(dq + (size_t) t * NH * HD, c.dk, c.dv, NH, NKV, HD, cs.pos + t, 1, hi, window, c.cap,
                                                 dscr, scells, dout, S, cs.block);
                ck(cudaStreamSynchronize(S), "window attention");
                ck(cudaMemcpy(got.data(), dout, (size_t) NH * HD * 4, cudaMemcpyDeviceToHost), "copy");
                if (std::memcmp(got.data(), &wide[(size_t) t * NH * HD], (size_t) NH * HD * 4)) ++place_diff;
                ++placed;
            }
        }
    }
    // fp16 q, K, V and p on the tensor cores, the queries twice gqa_attn_parity's
    std::printf("gqa_window_attn: max abs %.3g rel %.3g vs double over %zu windows of %d rows%s\n", worst, worst_rel,
                sizeof(cases) / sizeof(cases[0]), BW, worst > 2e-3 ? "  FAIL" : "");
    if (worst > 2e-3) ++fails;
    std::printf("gqa_window_attn: rows bitwise the same at every width 1-%d: %s (%d of %d differ)\n", BW, width_diff ? "NO" : "yes",
                width_diff, checked);
    if (width_diff) ++fails;
    std::printf("gqa_window_attn: a deep window's rows bitwise their one-row windows: %s (%d of %d differ)\n",
                place_diff || !placed ? "NO" : "yes", place_diff, placed);
    if (place_diff || !placed) ++fails;
    const unsigned long long frozen = 0x966f4088b1f5f753ull;   // the one-tile kernel's
    std::printf("gqa_window_attn hash: %016llx (bits against the frozen hash: %s)\n", hash, hash == frozen ? "identical" : "DIFFERENT");
    if (hash != frozen) ++fails;
    cudaFree(full.dk); cudaFree(full.dv); cudaFree(ring.dk); cudaFree(ring.dv); cudaFree(dscr); cudaFree(dwide);

    if (bench) {
        // the drafter's full-attention layer: every row of the block sees [0, pos + 8); the splits from the cells it reads
        const int64_t depth_max = 160 * 1024;
        const int64_t cap = depth_max + 128;
        uint16_t* bk = dev<uint16_t>((size_t) cap * NKV * HD);
        uint16_t* bv = dev<uint16_t>((size_t) cap * NKV * HD);
        ck(cudaMemset(bk, 0x2c, (size_t) cap * NKV * HD * 2), "memset");   // finite fp16 (0x2c2c = 0.0659)
        ck(cudaMemset(bv, 0x2c, (size_t) cap * NKV * HD * 2), "memset");
        float* bs = dev<float>(basalt::kernels::gqa_decode_scratch_floats(cap, BW, NH, HD));
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        for (int64_t depth : {4096LL, 8192LL, 16384LL, 32768LL, 65536LL, 102400LL, 163840LL - 64}) {
            for (int T : {1, 5, 8, 16, 32}) {
                const int64_t pos = depth - T, hi = pos + T, window = pos + T;
                const int64_t cells = hi + 64;
                const auto call = [&] {
                    basalt::kernels::gqa_window_attn(dq, bk, bv, NH, NKV, HD, pos, T, hi, window, cap, bs, cells, dout, S, true);
                };
                for (int i = 0; i < 3; ++i) call();
                const int iters = depth > 65536 ? 20 : 50;
                cudaEventRecord(e0, S);
                for (int i = 0; i < iters; ++i) call();
                cudaEventRecord(e1, S);
                ck(cudaEventSynchronize(e1), "bench");
                float ms = 0;
                cudaEventElapsedTime(&ms, e0, e1);
                const double us = 1e3 * ms / iters, bytes = (double) hi * NKV * HD * 2 * 2;
                std::printf("bench: window attention %d rows x %lld cells (%d x 128 heads, %d KV): %.1f us, %.0f GB/s of K/V read once\n",
                            T, (long long) hi, NH, NKV, us, bytes / us / 1e3);
            }
        }
        // the sliding-window layers' call: the window before the block and the block, in a ring of window + 63 cells
        const int64_t rw = 1025, rcap = 1088;
        for (int T : {1, 5, 8}) {
            const int64_t pos = 100000, hi = pos + T;
            const auto call = [&] {
                basalt::kernels::gqa_window_attn(dq, bk, bv, NH, NKV, HD, pos, T, hi, rw, rcap, bs, hi - (pos - rw + 1) + 64, dout, S, true);
            };
            for (int i = 0; i < 3; ++i) call();
            cudaEventRecord(e0, S);
            for (int i = 0; i < 200; ++i) call();
            cudaEventRecord(e1, S);
            ck(cudaEventSynchronize(e1), "bench");
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            std::printf("bench: window attention %d rows x a %lld-position window in a %lld-cell ring: %.1f us\n", T, (long long) rw,
                        (long long) rcap, 1e3 * ms / 200);
        }
        cudaFree(bk); cudaFree(bv); cudaFree(bs);
    }
    std::printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
