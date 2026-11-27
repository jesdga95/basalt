// tests/kernels/gqa_window_parity.cpp - gqa_window_attn (basalt/kernels/gqa_attn.hpp) at Inkling 2's DSpark geometry
// (24 x 128 query heads, 8 KV heads: a block of 8 rows is 24 rows of a KV group, two 16-row tiles; 32 rows are six)
// against a CPU reference in double precision, over block and per-row windows, a full and a wrapped sliding cache,
// widths 1..32 and odd depths, short calls (one tile a block) and deep ones (two).
//
// Checks: max abs and rel error to the reference; a row's bits are the same at every width of a window with the same
// end, and in a one-row window (another place in another tile); the outputs' bits against a frozen hash (a change that
// must not move a bit leaves it as it is).  The inputs come from mt19937's raw output alone, so they are the same on
// every platform.
// The int8 cache (KvInt8, the drafter's full layer with --kv int8): codes and per-64 scales by the cache's rule; every
// output's bits against the fp16 path on the cache's decoded values (the same fp16 the kernel decodes to), and against the
// CPU reference on those values; rows bitwise the same at every width.
// The segmented reader (gqa_window_attn_segs): int8 and fp16 segments over distinct caches and depths, each bitwise its
// own solo call; segment 1 read with segment 0's scales must differ.
// --bench: the full-attention layer's call (a window of 1..32 rows over every cell) at 4K..160K cells, fp16 and int8, us per
// call and the K/V bytes read once per call over that time, and the sliding layer's call.  Exit 77 without a GPU.
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
#include <type_traits>
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

// `c`'s values in the int8 cache (per 64 values of a cell's head: scale amax / 127 rounded to fp16, codes rounded to nearest
// against it), and `deq` the fp16 cache of the decoded values code x scale, rounded to fp16 as the kernel decodes them
struct Cache8 {
    std::vector<int8_t> k, v;
    std::vector<uint16_t> ks, vs;
    basalt::kernels::KvInt8 d;
};
void quantize(const std::vector<uint16_t>& src, std::vector<int8_t>& codes, std::vector<uint16_t>& scales, std::vector<uint16_t>& deq) {
    codes.resize(src.size());
    scales.resize(src.size() / 64);
    deq.resize(src.size());
    for (size_t g = 0; g < scales.size(); ++g) {
        float amax = 0;
        for (int i = 0; i < 64; ++i) amax = std::max(amax, std::fabs(h2f(src[g * 64 + i])));
        scales[g] = basalt::kernels::f16_from_f32(amax / 127.0f);
        const float sc = h2f(scales[g]);
        for (int i = 0; i < 64; ++i) {
            const float q = sc > 0 ? std::rint(h2f(src[g * 64 + i]) / sc) : 0.0f;
            codes[g * 64 + i] = (int8_t) std::max(-127.0f, std::min(127.0f, q));
            deq[g * 64 + i] = basalt::kernels::f16_from_f32((float) codes[g * 64 + i] * sc);
        }
    }
}

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
    // one launch of 64 segments of 5 rows (a batched drafter's blocks at 64 slots, past the 32 segments and 64 rows a
    // launch once took) against each segment alone, bitwise; half in the ring, half in the full cache's windows
    {
        constexpr int NSEG = basalt::kernels::kWindowSegsMax, T = 5;
        std::vector<float> sq((size_t) NSEG * T * NH * HD);
        for (auto& x : sq) x = 2.0f * draw(rng);
        float* dsq = dev<float>(sq.size());
        float* dso = dev<float>(sq.size());
        float* done = dev<float>((size_t) T * NH * HD);
        ck(cudaMemcpy(dsq, sq.data(), sq.size() * 4, cudaMemcpyHostToDevice), "copy");
        float* sscr = dev<float>(basalt::kernels::gqa_decode_scratch_floats(ring_cap, NSEG * T, NH, HD));
        basalt::kernels::GqaWindowSeg segs[NSEG];
        for (int k = 0; k < NSEG; ++k) {
            const bool in_ring = k % 2;
            const Cache& c = in_ring ? ring : full;
            const int64_t pos = 2000 + 37 * k;
            segs[k] = {c.dk, c.dv, pos, T, pos + T, ring_window, c.cap, true};
        }
        const int64_t scells = ring_window + T + 64;
        basalt::kernels::gqa_window_attn_segs(dsq, NH, NKV, HD, segs, NSEG, sscr, scells, dso, S);
        ck(cudaStreamSynchronize(S), "segments attention");
        std::vector<float> all(sq.size()), one((size_t) T * NH * HD);
        ck(cudaMemcpy(all.data(), dso, all.size() * 4, cudaMemcpyDeviceToHost), "copy");
        int seg_diff = 0;
        for (int k = 0; k < NSEG; ++k) {
            const basalt::kernels::GqaWindowSeg& g = segs[k];
            basalt::kernels::gqa_window_attn(dsq + (size_t) k * T * NH * HD, g.k_cache, g.v_cache, NH, NKV, HD, g.pos0, T, g.hi, g.window,
                                             g.cap, sscr, scells, done, S, true);
            ck(cudaStreamSynchronize(S), "window attention");
            ck(cudaMemcpy(one.data(), done, one.size() * 4, cudaMemcpyDeviceToHost), "copy");
            if (std::memcmp(one.data(), &all[(size_t) k * T * NH * HD], one.size() * 4)) ++seg_diff;
        }
        std::printf("gqa_window_attn_segs: %d segments of %d rows in one launch = each alone: %s (%d differ)\n", NSEG, T,
                    seg_diff ? "NO" : "yes", seg_diff);
        if (seg_diff) ++fails;
        if (bench) {
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            cudaEventRecord(e0, S);
            for (int i = 0; i < 100; ++i)
                basalt::kernels::gqa_window_attn_segs(dsq, NH, NKV, HD, segs, NSEG, sscr, scells, dso, S);
            cudaEventRecord(e1, S);
            ck(cudaEventSynchronize(e1), "bench");
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            std::printf("bench: window attention, %d segments x %d rows in one launch: %.1f us\n", NSEG, T, 1e3 * ms / 100);
        }
        cudaFree(dsq); cudaFree(dso); cudaFree(done); cudaFree(sscr);
    }

    // the int8 cache: the same cases on the full and ring caches quantized; each output bitwise the fp16 path's on the
    // decoded cache, near the CPU reference on it, and the same at every width
    {
        int bit_diff = 0, bits_checked = 0, wdiff = 0, wchecked = 0;
        double worst8 = 0;
        const auto to_dev = [](const auto& v) {
            using T = typename std::decay_t<decltype(v)>::value_type;
            T* p = dev<T>(v.size());
            ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "copy");
            return p;
        };
        Cache8 q8[2];
        Cache deq[2];
        for (int i = 0; i < 2; ++i) {
            const Cache& c = i ? ring : full;
            quantize(c.k, q8[i].k, q8[i].ks, deq[i].k);
            quantize(c.v, q8[i].v, q8[i].vs, deq[i].v);
            deq[i].cap = c.cap;
            deq[i].dk = to_dev(deq[i].k);
            deq[i].dv = to_dev(deq[i].v);
            q8[i].d.k = to_dev(q8[i].k);
            q8[i].d.v = to_dev(q8[i].v);
            q8[i].d.ks = to_dev(q8[i].ks);
            q8[i].d.vs = to_dev(q8[i].vs);
        }
        std::vector<float> ref16(hq.size());
        for (const Case& cs : cases) {
            const int i = cs.ring ? 1 : 0;
            const Cache& c = deq[i];
            const int64_t hi = cs.pos + BW;
            const int64_t window = cs.window > 0 ? cs.window : cs.pos + BW;
            const int64_t lo_all = std::max<int64_t>(0, cs.pos - window + 1);
            const int64_t scells = std::min<int64_t>(c.cap, hi - lo_all + 64);
            basalt::kernels::gqa_window_attn(dq, c.dk, c.dv, NH, NKV, HD, cs.pos, BW, hi, window, c.cap, dscr, scells, dwide, S, cs.block);
            ck(cudaStreamSynchronize(S), "window attention");
            ck(cudaMemcpy(ref16.data(), dwide, ref16.size() * 4, cudaMemcpyDeviceToHost), "copy");
            basalt::kernels::gqa_window_attn(dq, nullptr, nullptr, NH, NKV, HD, cs.pos, BW, hi, window, c.cap, dscr, scells, dwide, S,
                                             cs.block, q8[i].d);
            ck(cudaStreamSynchronize(S), "window attention int8");
            ck(cudaMemcpy(wide.data(), dwide, wide.size() * 4, cudaMemcpyDeviceToHost), "copy");
            ++bits_checked;
            if (std::memcmp(wide.data(), ref16.data(), wide.size() * 4)) ++bit_diff;
            std::vector<double> want((size_t) NH * HD);
            for (int t = 0; t < BW; ++t) {
                const int64_t lo = std::max<int64_t>(0, cs.pos + (cs.block ? 0 : t) - window + 1);
                ref_row(&hq[(size_t) t * NH * HD], c, lo, hi, want.data());
                for (int j = 0; j < NH * HD; ++j)
                    worst8 = std::max(worst8, std::fabs(wide[(size_t) t * NH * HD + j] - want[(size_t) j]));
            }
            for (int T : {1, 3, 8, 17, BW}) {
                basalt::kernels::gqa_window_attn(dq, nullptr, nullptr, NH, NKV, HD, cs.pos, T, hi, window, c.cap, dscr, scells, dout, S,
                                                 cs.block, q8[i].d);
                ck(cudaStreamSynchronize(S), "window attention int8");
                ck(cudaMemcpy(got.data(), dout, (size_t) T * NH * HD * 4, cudaMemcpyDeviceToHost), "copy");
                if (std::memcmp(got.data(), wide.data(), (size_t) T * NH * HD * 4)) ++wdiff;
                ++wchecked;
            }
        }
        std::printf("gqa_window_attn int8: bitwise the fp16 path on the decoded cache: %s (%d of %d windows differ)\n",
                    bit_diff ? "NO" : "yes", bit_diff, bits_checked);
        if (bit_diff) ++fails;
        std::printf("gqa_window_attn int8: max abs %.3g vs double on the decoded cache%s\n", worst8, worst8 > 2e-3 ? "  FAIL" : "");
        if (worst8 > 2e-3) ++fails;
        std::printf("gqa_window_attn int8: rows bitwise the same at widths 1, 3, 8, 17, %d: %s (%d of %d differ)\n", BW,
                    wdiff ? "NO" : "yes", wdiff, wchecked);
        if (wdiff) ++fails;

        // the segmented reader (a batch window's sessions in one call): segments over distinct caches at distinct depths,
        // each segment's rows bitwise its own solo call with the same scratch.  Int8: two full caches and a sliding window
        // of the first; fp16: two rings.  A copy of the int8 call with segment 1's scales swapped for segment 0's must differ.
        {
            Cache full2 = make_cache(full_cap, rng), ring2 = make_cache(ring_cap, rng);
            Cache8 q82;
            Cache deq2;
            quantize(full2.k, q82.k, q82.ks, deq2.k);
            quantize(full2.v, q82.v, q82.vs, deq2.v);
            q82.d.k = to_dev(q82.k);
            q82.d.v = to_dev(q82.v);
            q82.d.ks = to_dev(q82.ks);
            q82.d.vs = to_dev(q82.vs);
            const auto seg = [](const Cache* c, basalt::kernels::KvInt8 q, int64_t pos, int T, int64_t window, int64_t cap, bool block) {
                basalt::kernels::GqaWindowSeg g;
                if (c != nullptr) { g.k_cache = c->dk; g.v_cache = c->dv; }
                g.q8 = q;
                g.pos0 = pos;
                g.T = T;
                g.hi = pos + T;
                g.window = window > 0 ? window : pos + T;
                g.cap = cap;
                g.block = block;
                return g;
            };
            const auto run_segs = [&](const std::vector<basalt::kernels::GqaWindowSeg>& sg, std::vector<float>& outv) {
                int64_t scells = 0, rows = 0;
                for (const auto& g : sg) {
                    const int64_t lo = std::max<int64_t>(0, g.pos0 - g.window + 1);
                    scells = std::max(scells, std::min<int64_t>(g.cap, g.hi - lo + 64));
                    rows += g.T;
                }
                basalt::kernels::gqa_window_attn_segs(dq, NH, NKV, HD, sg.data(), (int) sg.size(), dscr, scells, dout, S);
                ck(cudaStreamSynchronize(S), "window attention segments");
                outv.resize((size_t) rows * NH * HD);
                ck(cudaMemcpy(outv.data(), dout, outv.size() * 4, cudaMemcpyDeviceToHost), "copy");
                return scells;
            };
            // segment k's rows against its solo call (its queries at its rows' offset), bitwise
            const auto check_solo = [&](const std::vector<basalt::kernels::GqaWindowSeg>& sg, const std::vector<float>& segv,
                                        int64_t scells, int& diff, int& n) {
                size_t row0 = 0;
                for (const auto& g : sg) {
                    basalt::kernels::gqa_window_attn(dq + row0 * NH * HD, g.k_cache, g.v_cache, NH, NKV, HD, g.pos0, g.T, g.hi,
                                                     g.window, g.cap, dscr, scells, dwide, S, g.block, g.q8);
                    ck(cudaStreamSynchronize(S), "window attention");
                    ck(cudaMemcpy(got.data(), dwide, (size_t) g.T * NH * HD * 4, cudaMemcpyDeviceToHost), "copy");
                    if (std::memcmp(got.data(), &segv[row0 * NH * HD], (size_t) g.T * NH * HD * 4)) ++diff;
                    ++n;
                    row0 += (size_t) g.T;
                }
            };
            int sdiff = 0, sn = 0;
            const std::vector<basalt::kernels::GqaWindowSeg> s8 = {
                seg(nullptr, q8[0].d, 12345, 8, 0, full_cap, true),
                seg(nullptr, q82.d, 4097, 5, 0, full_cap, true),
                seg(nullptr, q8[0].d, 777, 3, 500, full_cap, false),
            };
            std::vector<float> seg8, seg16, swapped;
            int64_t sc = run_segs(s8, seg8);
            check_solo(s8, seg8, sc, sdiff, sn);
            const std::vector<basalt::kernels::GqaWindowSeg> s16 = {
                seg(&ring, {}, 5001, 8, ring_window, ring_cap, true),
                seg(&ring2, {}, 3000, 5, ring_window, ring_cap, true),
            };
            sc = run_segs(s16, seg16);
            check_solo(s16, seg16, sc, sdiff, sn);
            std::printf("gqa_window_attn_segs: segments bitwise their solo calls (int8 x 3, fp16 rings x 2): %s (%d of %d differ)\n",
                        sdiff ? "NO" : "yes", sdiff, sn);
            if (sdiff) ++fails;
            std::vector<basalt::kernels::GqaWindowSeg> bad = s8;
            bad[1].q8.ks = q8[0].d.ks;
            bad[1].q8.vs = q8[0].d.vs;
            run_segs(bad, swapped);
            const size_t r1 = (size_t) s8[0].T * NH * HD, n1 = (size_t) s8[1].T * NH * HD;
            const bool seen = std::memcmp(&swapped[r1], &seg8[r1], n1 * 4) != 0;
            std::printf("gqa_window_attn_segs: segment 1 read with segment 0's scales differs: %s\n", seen ? "yes" : "NO");
            if (!seen) ++fails;
            cudaFree(full2.dk); cudaFree(full2.dv); cudaFree(ring2.dk); cudaFree(ring2.dv);
            cudaFree(q82.d.k); cudaFree(q82.d.v); cudaFree(q82.d.ks); cudaFree(q82.d.vs);
        }
        for (int i = 0; i < 2; ++i) {
            cudaFree(deq[i].dk); cudaFree(deq[i].dv);
            cudaFree(q8[i].d.k); cudaFree(q8[i].d.v); cudaFree(q8[i].d.ks); cudaFree(q8[i].d.vs);
        }
    }
    cudaFree(full.dk); cudaFree(full.dv); cudaFree(ring.dk); cudaFree(ring.dv); cudaFree(dscr); cudaFree(dwide);

    if (bench) {
        // the drafter's full-attention layer: every row of the block sees [0, pos + 8); the splits from the cells it reads
        const int64_t depth_max = 160 * 1024;
        const int64_t cap = depth_max + 128;
        uint16_t* bk = dev<uint16_t>((size_t) cap * NKV * HD);
        uint16_t* bv = dev<uint16_t>((size_t) cap * NKV * HD);
        ck(cudaMemset(bk, 0x2c, (size_t) cap * NKV * HD * 2), "memset");   // finite fp16 (0x2c2c = 0.0659)
        ck(cudaMemset(bv, 0x2c, (size_t) cap * NKV * HD * 2), "memset");
        basalt::kernels::KvInt8 b8;   // the int8 cache of the same cells: codes 0x2c, scales 0x2c2c
        b8.k = dev<int8_t>((size_t) cap * NKV * HD);
        b8.v = dev<int8_t>((size_t) cap * NKV * HD);
        b8.ks = dev<uint16_t>((size_t) cap * NKV * (HD / 64));
        b8.vs = dev<uint16_t>((size_t) cap * NKV * (HD / 64));
        ck(cudaMemset(b8.k, 0x2c, (size_t) cap * NKV * HD), "memset");
        ck(cudaMemset(b8.v, 0x2c, (size_t) cap * NKV * HD), "memset");
        ck(cudaMemset(b8.ks, 0x2c, (size_t) cap * NKV * (HD / 64) * 2), "memset");
        ck(cudaMemset(b8.vs, 0x2c, (size_t) cap * NKV * (HD / 64) * 2), "memset");
        float* bs = dev<float>(basalt::kernels::gqa_decode_scratch_floats(cap, BW, NH, HD));
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        for (int64_t depth : {4096LL, 8192LL, 16384LL, 32768LL, 65536LL, 102400LL, 131072LL, 163840LL - 64}) {
            for (int T : {1, 5, 8, 16, 32})
                for (const bool i8 : {false, true}) {
                    const int64_t pos = depth - T, hi = pos + T, window = pos + T;
                    const int64_t cells = hi + 64;
                    const auto call = [&] {
                        if (i8) basalt::kernels::gqa_window_attn(dq, nullptr, nullptr, NH, NKV, HD, pos, T, hi, window, cap, bs, cells, dout,
                                                                 S, true, b8);
                        else basalt::kernels::gqa_window_attn(dq, bk, bv, NH, NKV, HD, pos, T, hi, window, cap, bs, cells, dout, S, true);
                    };
                    for (int i = 0; i < 3; ++i) call();
                    const int iters = depth > 65536 ? 20 : 50;
                    cudaEventRecord(e0, S);
                    for (int i = 0; i < iters; ++i) call();
                    cudaEventRecord(e1, S);
                    ck(cudaEventSynchronize(e1), "bench");
                    float ms = 0;
                    cudaEventElapsedTime(&ms, e0, e1);
                    // fp16: 2 bytes a value; int8: a byte a value and a 2-byte scale per 64
                    const double row = i8 ? HD + (HD / 64) * 2 : HD * 2;
                    const double us = 1e3 * ms / iters, bytes = (double) hi * NKV * row * 2;
                    std::printf("bench: window attention %s %d rows x %lld cells (%d x 128 heads, %d KV): %.1f us, %.0f GB/s of K/V read once\n",
                                i8 ? "int8" : "fp16", T, (long long) hi, NH, NKV, us, bytes / us / 1e3);
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
        cudaFree(bk); cudaFree(bv); cudaFree(bs); cudaFree(b8.k); cudaFree(b8.v); cudaFree(b8.ks); cudaFree(b8.vs);
    }
    std::printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
