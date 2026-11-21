// src/models/inkling2/kernels/dspark_parity.cpp - the DSpark drafter's kernels against plain CPU references (double
// accumulation) at the drafter's geometry: the target features' mixes (and a row's bits alone), the block's inputs, the
// Markov recurrence, the screened head over the whole vocabulary (against its exact rows, its tie rule), the head folds and
// the recurrence's split. Each is timed. Exit 77 without a GPU.
#include "basalt/models/inkling2/dspark_kernels.hpp"
#include "basalt/models/inkling2/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

using namespace basalt::models::inkling2;

namespace {

constexpr int E = kWidth;
constexpr int kVocab = 200058;   // Inkling 2 Mini's real vocabulary
constexpr int kBlock = 8;        // the drafter's block: the anchor and seven masks
std::mt19937 rng(29);
int failures = 0;

std::vector<float> randn(size_t n, float s = 1.f, float mean = 0.f) {
    std::normal_distribution<float> d(mean, s);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}
template <class T> T* dev(const std::vector<T>& h) {
    T* p = nullptr;
    cudaMalloc(&p, h.size() * sizeof(T) + 16);
    cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}
template <class T> std::vector<T> host(const T* d, size_t n) {
    std::vector<T> h(n);
    cudaDeviceSynchronize();
    cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}
template <class T> T* zeros(size_t n) {
    T* p = nullptr;
    cudaMalloc(&p, n * sizeof(T) + 16);
    cudaMemset(p, 0, n * sizeof(T) + 16);
    return p;
}
uint16_t to_bf16(float f) {   // round to nearest even
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
float from_bf16(uint16_t h) {
    const uint32_t u = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
std::vector<uint16_t> bf16s(const std::vector<float>& v) {
    std::vector<uint16_t> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = to_bf16(v[i]);
    return o;
}
double q8_value(const uint8_t* rows, int n, int64_t r, int i) {
    const uint8_t* blk = rows + ((size_t) r * (n / 32) + i / 32) * 34;
    __half hd;
    std::memcpy(&hd, blk, 2);
    return (double) __half2float(hd) * (double) (int8_t) blk[2 + i % 32];
}

void report(const char* what, const std::vector<float>& got, const std::vector<double>& ref, double tol) {
    double max_abs = 0, max_ref = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        max_abs = std::max(max_abs, std::fabs(got[i] - ref[i]));
        max_ref = std::max(max_ref, std::fabs(ref[i]));
    }
    const double rel = max_abs / std::max(max_ref, 1e-30);
    const bool ok = rel <= tol && std::isfinite(max_abs);
    std::printf("%-34s max abs %.3e  max rel %.3e  %s\n", what, max_abs, rel, ok ? "ok" : "FAIL");
    failures += !ok;
}
void check(const char* what, bool ok) {
    std::printf("%-34s %s\n", what, ok ? "ok" : "FAIL");
    failures += !ok;
}
double time_us(const std::function<void()>& f, int n = 50) {
    f();
    cudaDeviceSynchronize();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) f();
    cudaDeviceSynchronize();
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / n;
}
void timing(const char* what, double us, double bytes) {
    std::printf("%-34s %8.2f us  %7.1f GB/s\n", what, us, bytes / us * 1e-3);
}

void test_mix() {
    const int R = 8, cap = 16;
    const auto tap = randn((size_t) kDsTargets * R * E);
    std::vector<float> alpha((size_t) kDsMixes * kDsTargets);
    for (int j = 0; j < kDsMixes; ++j) {   // softmaxed rows
        const auto raw = randn(kDsTargets);
        double z = 0;
        for (float v : raw) z += std::exp(v);
        for (int k = 0; k < kDsTargets; ++k) alpha[j * kDsTargets + k] = (float) (std::exp(raw[k]) / z);
    }
    std::vector<std::vector<float>> norm(kDsMixes);
    DsMixArgs a;
    a.alpha = dev(alpha);
    for (int j = 0; j < kDsMixes; ++j) {
        norm[j] = randn(E, 0.2f, 1.f);
        a.norm[j] = dev(norm[j]);
    }
    std::vector<int32_t> pos(R);
    for (int r = 0; r < R; ++r) pos[r] = 1000 + 3 * r;
    float* d_tap = dev(tap);
    int32_t* d_pos = dev(pos);
    float* d_out = zeros<float>((size_t) 3 * R * E);
    float* d_fr = zeros<float>((size_t) cap * E);
    int32_t* d_pos3 = zeros<int32_t>((size_t) R * 3);
    dspark_mix(d_tap, (int64_t) R * E, R, a, d_out, (int64_t) R * E, d_pos, d_fr, cap, d_pos3, nullptr);
    const auto out = host(d_out, (size_t) 3 * R * E);
    const auto fr = host(d_fr, (size_t) cap * E);
    const auto pos3 = host(d_pos3, (size_t) R * 3);
    std::vector<float> got_m, got_f;
    std::vector<double> ref_m, ref_f;
    for (int j = 0; j < kDsMixes; ++j)
        for (int r = 0; r < R; ++r) {
            std::vector<double> m(E);
            double ss = 0;
            for (int c = 0; c < E; ++c) {
                for (int k = 0; k < kDsTargets; ++k) m[c] += (double) alpha[j * kDsTargets + k] * tap[((size_t) k * R + r) * E + c];
                ss += m[c] * m[c];
            }
            const double inv = 1.0 / std::sqrt(ss / E + 1e-6);
            for (int c = 0; c < E; ++c) {
                const double y = m[c] * inv * norm[j][c];
                if (j < 3) {
                    got_m.push_back(out[((size_t) j * R + r) * E + c]);
                    ref_m.push_back(y);
                } else {
                    got_f.push_back(fr[(size_t) (pos[r] % cap) * E + c]);
                    ref_f.push_back(y);
                }
            }
        }
    report("mix (layer contexts)", got_m, ref_m, 2e-6);
    report("mix (Markov context ring)", got_f, ref_f, 2e-6);
    bool pos_ok = true;
    for (int r = 0; r < R; ++r)
        for (int i = 0; i < 3; ++i) pos_ok = pos_ok && pos3[r * 3 + i] == pos[r];
    check("mix positions", pos_ok);
    // row 3 alone gives the same bits
    float* d_out1 = zeros<float>((size_t) 3 * R * E);
    dspark_mix(d_tap + 3 * E, (int64_t) R * E, 1, a, d_out1, (int64_t) R * E, d_pos + 3, d_fr, cap, d_pos3, nullptr);
    const auto out1 = host(d_out1, (size_t) 3 * R * E);
    bool same = true;
    for (int j = 0; j < 3; ++j) same = same && std::memcmp(&out1[(size_t) j * R * E], &out[((size_t) j * R + 3) * E], E * 4) == 0;
    check("mix row 3 alone: same bits", same);
    const double us = time_us([&] { dspark_mix(d_tap, (int64_t) R * E, R, a, d_out, (int64_t) R * E, d_pos, d_fr, cap, d_pos3, nullptr); });
    timing("mix, 8 rows", us, (double) kDsTargets * R * E * 4 + (double) kDsMixes * R * E * 4);
}

void test_block_inputs() {
    const int V = 64, anchor = 37;
    const int64_t pos0 = 500;
    const auto emb = randn((size_t) V * E, 0.05f);
    const auto embb = bf16s(emb);
    const auto norm = randn(E, 0.2f, 1.f);
    const auto mask = randn(E, 0.3f);
    uint16_t* d_emb = dev(embb);
    float* d_norm = dev(norm);
    float* d_mask = dev(mask);
    float* d_x = zeros<float>((size_t) kBlock * E);
    int32_t* d_pos3 = zeros<int32_t>(kBlock * 3);
    int32_t* d_prev = zeros<int32_t>(1);
    float* d_state = dev(randn(kDsRank));
    const auto fr = randn((size_t) 16 * E);
    float* d_fr = dev(fr);
    float* d_frc = zeros<float>(E);
    int32_t* d_meta = dev(std::vector<int32_t>{anchor, (int32_t) pos0});
    const auto w1 = bf16s(randn((size_t) V * kDsRank, 0.5f));
    uint16_t* d_w1 = dev(w1);
    uint16_t* d_w1r = zeros<uint16_t>(kDsRank);
    auto run = [&] {
        dspark_block_inputs(d_emb, d_meta, d_norm, 1e-6f, d_mask, kBlock, d_x, d_pos3, d_prev, d_state, d_fr, 16, d_frc, d_w1, d_w1r, nullptr);
    };
    run();
    const auto x = host(d_x, (size_t) kBlock * E);
    std::vector<float> got;
    std::vector<double> ref;
    double ss = 0;
    for (int c = 0; c < E; ++c) ss += (double) from_bf16(embb[(size_t) anchor * E + c]) * from_bf16(embb[(size_t) anchor * E + c]);
    const double inv = 1.0 / std::sqrt(ss / E + 1e-6);
    for (int c = 0; c < E; ++c) {
        got.push_back(x[c]);
        ref.push_back(from_bf16(embb[(size_t) anchor * E + c]) * inv * norm[c]);
    }
    report("block inputs (anchor row)", got, ref, 2e-6);
    bool ok = std::memcmp(x.data() + E, mask.data(), E * 4) == 0 && std::memcmp(x.data() + 7 * E, mask.data(), E * 4) == 0;
    const auto pos3 = host(d_pos3, kBlock * 3);
    for (int t = 0; t < kBlock; ++t) ok = ok && pos3[t * 3] == pos0 + t && pos3[t * 3 + 2] == pos0 + t;
    const auto prev = host(d_prev, 1);
    const auto state = host(d_state, kDsRank);
    ok = ok && prev[0] == anchor;
    for (float s : state) ok = ok && s == 0.f;
    ok = ok && std::memcmp(host(d_frc, E).data(), fr.data() + (size_t) ((pos0 - 1) % 16) * E, E * 4) == 0;
    ok = ok && std::memcmp(host(d_w1r, kDsRank).data(), w1.data() + (size_t) anchor * kDsRank, kDsRank * 2) == 0;
    check("block inputs (masks, positions, start)", ok);
    const double us = time_us(run);
    timing("block inputs", us, (double) kBlock * E * 4 + E * 6);
}

void test_markov_step() {
    const int V = 64, prev = 11;
    const auto wsp = bf16s(randn((size_t) kDsRnn * 2 * kDsRank, 0.05f));
    const auto w1 = bf16s(randn((size_t) V * kDsRank, 0.5f));
    auto state = randn(kDsRank, 0.5f);
    for (auto& s : state) s = std::tanh(s);
    const auto ch = randn(kDsRnn), cf = randn(kDsRnn), b = randn(kDsRnn, 0.3f);
    uint16_t *d_wsp = dev(wsp), *d_w1 = dev(w1);
    float *d_ch = dev(ch), *d_cf = dev(cf), *d_b = dev(b), *d_state = dev(state);
    int32_t* d_prev = dev(std::vector<int32_t>{prev});
    float* d_out = zeros<float>(kDsRank);
    float* d_lat = zeros<float>(kDsRank);
    dspark_markov_step(d_ch, d_cf, d_b, d_wsp, d_w1 + (size_t) prev * kDsRank, d_state, d_out, d_lat, nullptr);
    const auto out = host(d_out, kDsRank), lat = host(d_lat, kDsRank);
    std::vector<double> x(2 * kDsRank);
    for (int i = 0; i < kDsRank; ++i) {
        x[i] = state[i];
        x[kDsRank + i] = from_bf16(w1[(size_t) prev * kDsRank + i]);
    }
    std::vector<double> raw(kDsRnn);
    for (int o = 0; o < kDsRnn; ++o) {
        double acc = (double) ch[o] + cf[o] + b[o];
        for (int k = 0; k < 2 * kDsRank; ++k) acc += (double) from_bf16(wsp[(size_t) o * 2 * kDsRank + k]) * x[k];
        raw[o] = acc;
    }
    std::vector<double> ref_s(kDsRank), ref_l(kDsRank);
    for (int j = 0; j < kDsRank; ++j) {
        const double g = 1.0 / (1.0 + std::exp(-raw[j]));
        ref_s[j] = g * state[j] + (1.0 - g) * std::tanh(raw[kDsRank + j]);
        ref_l[j] = std::tanh(raw[2 * kDsRank + j]);
    }
    report("Markov step (state)", out, ref_s, 2e-5);
    report("Markov step (latent)", lat, ref_l, 2e-5);
    const double us = time_us([&] { dspark_markov_step(d_ch, d_cf, d_b, d_wsp, d_w1 + (size_t) prev * kDsRank, d_state, d_out, d_lat, nullptr); });
    timing("Markov step", us, (double) kDsRnn * 2 * kDsRank * 2 + 3.0 * kDsRnn * 4);
}

// q8_1 blocks of a row (ggml's quantize_row_q8_1: d = amax / 127, the codes rounded, then d and d * sum of the codes)
std::vector<uint8_t> q8_1(const std::vector<float>& x) {
    std::vector<uint8_t> o(x.size() / 32 * 36);
    for (size_t b = 0; b < x.size() / 32; ++b) {
        float amax = 0;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[b * 32 + i]));
        const float d = amax / 127.f, id = d ? 1.f / d : 0.f;
        int sum = 0;
        for (int i = 0; i < 32; ++i) {
            const int q = (int) std::lround(x[b * 32 + i] * id);
            o[b * 36 + 4 + i] = (uint8_t) (int8_t) q;
            sum += q;
        }
        const __half hd = __float2half(d), hs = __float2half(d * (float) sum);
        std::memcpy(&o[b * 36], &hd, 2);
        std::memcpy(&o[b * 36 + 2], &hs, 2);
    }
    return o;
}
double q8_1_value(const uint8_t* x, int i) {
    __half hd;
    std::memcpy(&hd, x + (i / 32) * 36, 2);
    return (double) __half2float(hd) * (double) (int8_t) x[(i / 32) * 36 + 4 + i % 32];
}
// Q8_0 rows with random codes and scales of `scale` (1 to 2 times)
std::vector<uint8_t> q8_0_rows(int64_t rows, int n, float scale, uint32_t seed) {
    std::vector<uint8_t> w((size_t) rows * (n / 32) * 34);
    uint32_t s = seed;
    for (size_t blk = 0; blk < w.size() / 34; ++blk) {
        const __half hd = __float2half(scale * (1.f + (float) (blk % 13) / 13.f));
        std::memcpy(&w[blk * 34], &hd, 2);
        for (int i = 0; i < 32; ++i) {
            s = s * 1664525u + 1013904223u;
            w[blk * 34 + 2 + i] = (uint8_t) (s >> 24);
        }
    }
    return w;
}

// The screened head over the real vocabulary: planted drafts (the activation leans toward a chosen row, so the best leads by a
// few units, as a confident draft's does) against the exact head's first argmax (dspark_head_exact, itself checked against a
// CPU reference), the pick's probability against the exact softmax, a tie, and unplanted (flat) drafts counted.
void test_head() {
    const int64_t V = kVocab;
    const int R = kDsDrafts;
    const auto head = q8_0_rows(V, E, 1.5e-4f, 777u), fold = q8_0_rows(V, kDsRank, 2e-3f, 4242u);
    const auto fb = randn((size_t) V, 0.5f);
    uint8_t *d_head = dev(head), *d_fold = dev(fold);
    float* d_fb = dev(fb);
    void* d_hsc = zeros<uint8_t>(dspark_screen_bytes(E, V));
    void* d_fsc = zeros<uint8_t>(dspark_screen_bytes(kDsRank, V));
    dspark_screen_pack(d_head, E, V, d_hsc, nullptr);
    dspark_screen_pack(d_fold, kDsRank, V, d_fsc, nullptr);
    DsHead h;
    h.head = d_head;
    h.fold = d_fold;
    h.fold_b = d_fb;
    h.head_sc = d_hsc;
    h.fold_sc = d_fsc;
    h.n_vocab = V;
    h.cap = 30.f;
    float* d_approx = zeros<float>((size_t) R * dspark_screen_rows(V));
    float* d_fbp = zeros<float>((size_t) dspark_screen_rows(V));
    dspark_screen_order(d_fb, V, d_fbp, nullptr);
    h.fold_bp = d_fbp;
    float* d_logits = zeros<float>((size_t) V);
    void* scr = zeros<uint8_t>(dspark_head_scratch_bytes(V));
    int32_t* d_prev = zeros<int32_t>(1);
    int32_t* d_ids = zeros<int32_t>(8);
    float* d_conf = zeros<float>(8);
    std::vector<uint16_t> w1((size_t) V * kDsRank);   // the Markov head's token rows: a pattern that names its row
    for (size_t k = 0; k < w1.size(); ++k) w1[k] = (uint16_t) ((k / kDsRank) * 131 + k % kDsRank);
    uint16_t* d_w1 = dev(w1);
    uint16_t* d_w1r = zeros<uint16_t>(kDsRank);
    bool rows_ok = true;
    const size_t hrow = (size_t) E / 32 * 36, lrow = (size_t) kDsRank / 32 * 36;
    std::uniform_int_distribution<int64_t> pick_row(0, V - 1);
    std::uniform_real_distribution<float> lean(0.f, 1.f);
    // one block of R drafts: the rows (planted toward plant[i] when given), the latents; the screened picks and the exact
    // head's argmax and probability for each
    int differ = 0, prob_bad = 0, drafts = 0;
    double worst_prob = 0;
    const int64_t cluster[6] = {8, 11, 12, 13, 25, 30};
    auto block = [&](const std::vector<int64_t>& plant, float strength, bool count_probs, bool clustered = false) {
        std::vector<uint8_t> hq, lq;
        for (int i = 0; i < R; ++i) {
            auto x = randn(E, 5.f);
            if (clustered) {   // six rows of one tile in token order lead the rest, within a unit of each other
                for (int k = 0; k < 6; ++k) {
                    const int64_t t = cluster[k];
                    double n2 = 0;
                    for (int e = 0; e < E; ++e) n2 += q8_value(head.data(), E, t, e) * q8_value(head.data(), E, t, e);
                    const float a = strength * (1.f - 0.04f * lean(rng));
                    for (int e = 0; e < E; ++e) x[e] += (float) (a * q8_value(head.data(), E, t, e) / n2);
                }
            } else if (!plant.empty()) {
                double n2 = 0;
                for (int e = 0; e < E; ++e) n2 += q8_value(head.data(), E, plant[(size_t) i], e) * q8_value(head.data(), E, plant[(size_t) i], e);
                for (int e = 0; e < E; ++e) x[e] += (float) (strength * q8_value(head.data(), E, plant[(size_t) i], e) / n2);
            }
            const auto q = q8_1(x);
            hq.insert(hq.end(), q.begin(), q.end());
            auto lat = randn(kDsRank, 0.5f);
            for (auto& v : lat) v = std::tanh(v);
            const auto ql = q8_1(lat);
            lq.insert(lq.end(), ql.begin(), ql.end());
        }
        uint8_t *d_hq = dev(hq), *d_lq = dev(lq);
        dspark_head_base(h, d_hq, R, d_approx, nullptr);
        for (int i = 0; i < R; ++i) dspark_head_pick(h, d_hq, d_lq + i * lrow, d_approx, i, scr, d_prev, d_ids, d_conf, d_w1, d_w1r, nullptr);
        const auto ids = host(d_ids, 8);
        rows_ok = rows_ok && std::memcmp(host(d_w1r, kDsRank).data(), w1.data() + (size_t) ids[R - 1] * kDsRank, kDsRank * 2) == 0;
        const auto cf = host(d_conf, 8);
        for (int i = 0; i < R; ++i) {
            dspark_head_exact(h, d_hq, d_lq + i * lrow, i, d_logits, nullptr);
            const auto l = host(d_logits, (size_t) V);
            const int64_t best = std::max_element(l.begin(), l.end()) - l.begin();
            double z = 0;
            for (int64_t v = 0; v < V; ++v) z += std::exp((double) l[v] - l[best]);
            ++drafts;
            differ += best != ids[i];
            if (count_probs) {
                const double rel = std::fabs(cf[i] - 1.0 / z) / (1.0 / z);
                worst_prob = std::max(worst_prob, std::fabs(cf[i] - 1.0 / z));
                prob_bad += std::fabs(cf[i] - 1.0 / z) > 0.05 && rel > 0.1;
            }
        }
        cudaFree(d_hq);
        cudaFree(d_lq);
        return ids;
    };
    // the exact logits against a CPU reference on a few rows
    {
        std::vector<int64_t> plant(R);
        for (auto& p : plant) p = pick_row(rng);
        std::vector<uint8_t> hq = q8_1(randn((size_t) E * R, 5.f)), lq = q8_1(randn((size_t) kDsRank * R, 0.5f));
        uint8_t *d_hq = dev(hq), *d_lq = dev(lq);
        dspark_head_exact(h, d_hq, d_lq + 3 * lrow, 3, d_logits, nullptr);
        const auto l = host(d_logits, (size_t) V);
        std::vector<float> got;
        std::vector<double> ref;
        for (int k = 0; k < 64; ++k) {
            const int64_t v = k * (V - 1) / 63;
            double a = 0, b = 0;
            for (int e = 0; e < E; ++e) a += q8_value(head.data(), E, v, e) * q8_1_value(hq.data() + 3 * hrow, e);
            for (int e = 0; e < kDsRank; ++e) b += q8_value(fold.data(), kDsRank, v, e) * q8_1_value(lq.data() + 3 * lrow, e);
            got.push_back(l[(size_t) v]);
            ref.push_back(30.0 * std::tanh((a + b + fb[(size_t) v]) / 30.0));
        }
        report("head: exact logits", got, ref, 1e-5);
        cudaFree(d_hq);
        cudaFree(d_lq);
    }
    for (int t = 0; t < 12; ++t) {
        std::vector<int64_t> plant(R);
        for (auto& p : plant) p = pick_row(rng);
        block(plant, 22.f + 12.f * lean(rng), true);
    }
    std::printf("%-34s %d of %d picks differ from the exact head; probability worst abs error %.3e\n", "head: planted", differ,
                drafts, worst_prob);
    check("head: planted picks are the exact head's", differ == 0);
    check("head: planted probabilities", prob_bad == 0);
    differ = drafts = 0;
    for (int t = 0; t < 6; ++t) block({}, 26.f, false, true);
    std::printf("%-34s %d of %d picks differ from the exact head\n", "head: six near-best rows together", differ, drafts);
    check("head: near-best rows of one token tile", differ == 0);
    // a tie: row 150000 copied to row 777 (head, fold and bias), the drafts planted at 150000 take 777
    {
        std::vector<uint8_t> h2 = head, f2 = fold;
        std::vector<float> fb2 = fb;
        std::memcpy(&h2[(size_t) 777 * (E / 32) * 34], &head[(size_t) 150000 * (E / 32) * 34], (size_t) (E / 32) * 34);
        std::memcpy(&f2[(size_t) 777 * (kDsRank / 32) * 34], &fold[(size_t) 150000 * (kDsRank / 32) * 34], (size_t) (kDsRank / 32) * 34);
        fb2[777] = fb[150000];
        cudaMemcpy(d_head, h2.data(), h2.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_fold, f2.data(), f2.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_fb, fb2.data(), fb2.size() * 4, cudaMemcpyHostToDevice);
        dspark_screen_pack(d_head, E, V, d_hsc, nullptr);
        dspark_screen_pack(d_fold, kDsRank, V, d_fsc, nullptr);
        dspark_screen_order(d_fb, V, d_fbp, nullptr);
        differ = drafts = 0;
        const auto ids = block(std::vector<int64_t>(R, 150000), 30.f, false);
        bool ok = differ == 0;
        for (int i = 0; i < R; ++i) ok = ok && ids[i] == 777;
        check("head: a tie takes the lower row", ok);
        cudaMemcpy(d_head, head.data(), head.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_fold, fold.data(), fold.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_fb, fb.data(), fb.size() * 4, cudaMemcpyHostToDevice);
        dspark_screen_pack(d_head, E, V, d_hsc, nullptr);
        dspark_screen_pack(d_fold, kDsRank, V, d_fsc, nullptr);
        dspark_screen_order(d_fb, V, d_fbp, nullptr);
    }
    // flat logits (no plant): every tile near the best, the overflow path; counted, not failed
    differ = drafts = 0;
    for (int t = 0; t < 4; ++t) block({}, 0.f, false);
    std::printf("%-34s %d of %d picks differ from the exact head (flat logits, the overflow path)\n", "head: unplanted", differ, drafts);
    check("head: the pick's w1 row copied out", rows_ok);
    // timing: the head term of seven rows once, then a draft's pick, on flat and on planted rows
    std::vector<float> xp = randn((size_t) E * R, 5.f);
    for (int i = 0; i < R; ++i) {
        const int64_t t = pick_row(rng);
        double n2 = 0;
        for (int e = 0; e < E; ++e) n2 += q8_value(head.data(), E, t, e) * q8_value(head.data(), E, t, e);
        for (int e = 0; e < E; ++e) xp[(size_t) i * E + e] += (float) (28.0 * q8_value(head.data(), E, t, e) / n2);
    }
    std::vector<uint8_t> hq = q8_1(randn((size_t) E * R, 5.f)), hp = q8_1(xp), lq = q8_1(randn((size_t) kDsRank * R, 0.5f));
    uint8_t *d_hq = dev(hq), *d_hp = dev(hp), *d_lq = dev(lq);
    double us = time_us([&] { dspark_head_base(h, d_hq, R, d_approx, nullptr); });
    timing("head base, 7 rows", us, (double) dspark_screen_bytes(E, V) + (double) R * V * 4);
    us = time_us([&] { dspark_head_pick(h, d_hq, d_lq, d_approx, 0, scr, d_prev, d_ids, d_conf, d_w1, d_w1r, nullptr); });
    timing("head pick, one draft (flat)", us, (double) dspark_screen_bytes(kDsRank, V) + 2.0 * V * 4);
    dspark_head_base(h, d_hp, R, d_approx, nullptr);
    us = time_us([&] { dspark_head_pick(h, d_hp, d_lq, d_approx, 0, scr, d_prev, d_ids, d_conf, d_w1, d_w1r, nullptr); });
    timing("head pick, one draft (planted)", us, (double) dspark_screen_bytes(kDsRank, V) + 2.0 * V * 4);
    d_hq = d_hp;
    us = time_us([&] {
        dspark_head_base(h, d_hq, R, d_approx, nullptr);
        for (int i = 0; i < R; ++i) dspark_head_pick(h, d_hq, d_lq + i * lrow, d_approx, i, scr, d_prev, d_ids, d_conf, d_w1, d_w1r, nullptr);
    });
    timing("head, a block of 7 drafts", us, (double) dspark_screen_bytes(E, V) + R * (double) dspark_screen_bytes(kDsRank, V));
}

void test_fold() {
    const int64_t V = kVocab, check_rows = 512;
    // the head's Q8_0 blocks made directly: random codes, scales near the real head's
    std::vector<uint8_t> head((size_t) V * (E / 32) * 34);
    uint32_t s = 12345u;
    for (size_t blk = 0; blk < head.size() / 34; ++blk) {
        const __half hd = __float2half(1.5e-4f * (1.f + (float) (blk % 13) / 13.f));
        std::memcpy(&head[blk * 34], &hd, 2);
        for (int i = 0; i < 32; ++i) {
            s = s * 1664525u + 1013904223u;
            head[blk * 34 + 2 + i] = (uint8_t) (s >> 24);
        }
    }
    const auto w = bf16s(randn((size_t) E * kDsRank, 0.02f));
    const auto b = randn(E, 0.05f);
    uint8_t* d_head = dev(head);
    uint16_t* d_w = dev(w);
    float* d_b = dev(b);
    uint8_t* d_fold = zeros<uint8_t>((size_t) V * (kDsRank / 32) * 34);
    float* d_fb = zeros<float>((size_t) V);
    dspark_fold(d_head, V, d_w, d_b, d_fold, d_fb, nullptr);
    const auto fold = host(d_fold, (size_t) V * (kDsRank / 32) * 34);
    const auto fbv = host(d_fb, (size_t) V);
    // rows spread over the vocabulary: each block of 32 outputs is within half a Q8_0 step of the exact product
    double worst = 0;
    std::vector<float> got_b;
    std::vector<double> ref_b;
    for (int64_t i = 0; i < check_rows; ++i) {
        const int64_t v = i * (V - 1) / (check_rows - 1);
        std::vector<double> ex(kDsRank, 0.0);
        double eb = 0;
        for (int e = 0; e < E; ++e) {
            const double h = q8_value(head.data(), E, v, e);
            eb += h * b[e];
            for (int r = 0; r < kDsRank; ++r) ex[r] += h * from_bf16(w[(size_t) e * kDsRank + r]);
        }
        for (int blk = 0; blk < kDsRank / 32; ++blk) {
            double amax = 0;
            for (int r = blk * 32; r < blk * 32 + 32; ++r) amax = std::max(amax, std::fabs(ex[r]));
            for (int r = blk * 32; r < blk * 32 + 32; ++r)
                worst = std::max(worst, std::fabs(q8_value(fold.data(), kDsRank, v, r) - ex[r]) / std::max(amax, 1e-30));
        }
        got_b.push_back(fbv[v]);
        ref_b.push_back(eb);
    }
    // half a step of amax / 127, the fp16 rounding of the step (2^-11 of it, on codes up to 127), and a little for the sums
    const bool ok = worst <= 0.5 / 127.0 + 1.0 / 2048.0 + 1e-5;
    std::printf("%-34s worst error / block max %.3e  %s\n", "fold (Q8_0 rows)", worst, ok ? "ok" : "FAIL");
    failures += !ok;
    report("fold bias", got_b, ref_b, 2e-5);
    const double us = time_us([&] { dspark_fold(d_head, V, d_w, d_b, d_fold, d_fb, nullptr); }, 3);
    timing("folds (once at load)", us, (double) head.size() * 2 + (double) fold.size());
}

void test_split() {
    constexpr int kIn = 2 * kDsRank + 2 * E;
    std::vector<uint16_t> w((size_t) kDsRnn * kIn);
    for (size_t i = 0; i < w.size(); ++i) w[i] = (uint16_t) (i * 2654435761u >> 7);
    uint16_t* d_w = dev(w);
    uint16_t* d_sp = zeros<uint16_t>((size_t) kDsRnn * 2 * kDsRank);
    uint16_t* d_h = zeros<uint16_t>((size_t) kDsRnn * E);
    uint16_t* d_f = zeros<uint16_t>((size_t) kDsRnn * E);
    dspark_split_rnn(d_w, d_sp, d_h, d_f, nullptr);
    const auto sp = host(d_sp, (size_t) kDsRnn * 2 * kDsRank), h = host(d_h, (size_t) kDsRnn * E), f = host(d_f, (size_t) kDsRnn * E);
    bool ok = true;
    for (int o = 0; o < kDsRnn && ok; ++o) {
        ok = std::memcmp(&sp[(size_t) o * 2 * kDsRank], &w[(size_t) o * kIn], 2 * kDsRank * 2) == 0 &&
             std::memcmp(&h[(size_t) o * E], &w[(size_t) o * kIn + 2 * kDsRank], E * 2) == 0 &&
             std::memcmp(&f[(size_t) o * E], &w[(size_t) o * kIn + 2 * kDsRank + E], E * 2) == 0;
    }
    check("recurrence split", ok);
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("no GPU\n");
        return 77;
    }
    test_mix();
    test_block_inputs();
    test_markov_step();
    test_head();
    test_fold();
    test_split();
    std::printf("inkling2_dspark_parity: %d failures\n", failures);
    return failures ? 1 : 0;
}
