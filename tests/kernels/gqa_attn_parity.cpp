// tests/kernels/gqa_attn_parity.cpp - gqa_decode_attn and gqa_prompt_attn (basalt/kernels/gqa_attn.hpp) and
// attn_qk_prepare (dense_ops.hpp) against a CPU reference in double precision, at the Qwen3.5-27B attention geometry
// (24 query heads, 4 KV heads, 256 wide, 64 rotated dims, interleaved M-RoPE sections 11/11/10).
//
// Checks: each kernel's max absolute and relative error to the reference (the int8 prompt kernel's narrow, split and
// 8-warp launches); a decode row's output is bitwise the same
// whatever the window width or the row's place in it; the prompt kernel's rows agree with decode's within fp16 rounding.
// Exit 77 without a GPU.  Prints one line per check and a microbench line for the decode and prompt kernels.
#include "basalt/kernels/dense_ops.hpp"
#include "basalt/kernels/f16_bits.hpp"
#include "basalt/kernels/gqa_attn.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int NH = 24, NKV = 4, HD = 256, ROT = 64;

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

// out[h][d] for one query row at `visible` cells
void ref_attn(const float* q, const std::vector<uint16_t>& kc, const std::vector<uint16_t>& vc, int64_t visible, double* out) {
    for (int h = 0; h < NH; ++h) {
        const int kvh = h / (NH / NKV);
        std::vector<double> s((size_t) visible);
        double mx = -1e300;
        for (int64_t c = 0; c < visible; ++c) {
            double d = 0;
            for (int i = 0; i < HD; ++i) d += (double) q[h * HD + i] * h2f(kc[((size_t) c * NKV + kvh) * HD + i]);
            s[(size_t) c] = d / 16.0;
            mx = std::max(mx, s[(size_t) c]);
        }
        double l = 0;
        for (auto& x : s) { x = std::exp(x - mx); l += x; }
        for (int i = 0; i < HD; ++i) {
            double a = 0;
            for (int64_t c = 0; c < visible; ++c) a += s[(size_t) c] * h2f(vc[((size_t) c * NKV + kvh) * HD + i]);
            out[h * HD + i] = a / l;
        }
    }
}

struct Err {
    double abs = 0, rel = 0;
    void add(double got, double want) {
        abs = std::max(abs, std::fabs(got - want));
        rel = std::max(rel, std::fabs(got - want) / std::max(std::fabs(want), 1e-2));
    }
};

}  // namespace

int main(int argc, char** argv) {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("no GPU\n");
        return 77;
    }
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    int fails = 0;

    // ---- attn_qk_prepare: q/k norm, partial rope with M-RoPE axes, cache write
    {
        const int T = 3;
        std::vector<float> qf((size_t) T * NH * 2 * HD), k((size_t) T * NKV * HD), v(k.size()), qn(HD), kn(HD);
        for (auto& x : qf) x = nd(rng);
        for (auto& x : k) x = nd(rng);
        for (auto& x : v) x = nd(rng);
        for (auto& x : qn) x = 1.0f + 0.1f * nd(rng);
        for (auto& x : kn) x = 1.0f + 0.1f * nd(rng);
        std::vector<int32_t> pos = {5, 5, 5, 6, 9, 11, 7, 7, 7};   // a text row, an image row (t, h, w), a text row
        basalt::kernels::RopeSpec rope;
        rope.rot = ROT;
        rope.theta = 1e7f;
        for (int i = 0; i < ROT / 2; ++i) rope.axis[i] = (int8_t) (i % 3 == 1 && i < 33 ? 1 : i % 3 == 2 && i < 30 ? 2 : 0);
        float *d_qf = dev<float>(qf.size()), *d_k = dev<float>(k.size()), *d_v = dev<float>(v.size()), *d_qn = dev<float>(HD),
              *d_kn = dev<float>(HD), *d_qo = dev<float>((size_t) T * NH * HD);
        int32_t* d_pos = dev<int32_t>(pos.size());
        const int64_t cell0 = 10;
        uint16_t *d_kc = dev<uint16_t>((size_t) (cell0 + T) * NKV * HD), *d_vc = dev<uint16_t>((size_t) (cell0 + T) * NKV * HD);
        ck(cudaMemcpy(d_qf, qf.data(), qf.size() * 4, cudaMemcpyHostToDevice), "copy");
        ck(cudaMemcpy(d_k, k.data(), k.size() * 4, cudaMemcpyHostToDevice), "copy");
        ck(cudaMemcpy(d_v, v.data(), v.size() * 4, cudaMemcpyHostToDevice), "copy");
        ck(cudaMemcpy(d_qn, qn.data(), HD * 4, cudaMemcpyHostToDevice), "copy");
        ck(cudaMemcpy(d_kn, kn.data(), HD * 4, cudaMemcpyHostToDevice), "copy");
        ck(cudaMemcpy(d_pos, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice), "copy");
        basalt::kernels::attn_qk_prepare(d_qf, d_k, d_v, d_qn, d_kn, 1e-6f, rope, d_pos, T, NH, NKV, d_qo, d_kc, d_vc, cell0, nullptr);
        std::vector<float> qo((size_t) T * NH * HD);
        std::vector<uint16_t> kc((size_t) (cell0 + T) * NKV * HD);
        ck(cudaMemcpy(qo.data(), d_qo, qo.size() * 4, cudaMemcpyDeviceToHost), "copy");
        ck(cudaMemcpy(kc.data(), d_kc, kc.size() * 2, cudaMemcpyDeviceToHost), "copy");
        Err eq, ek;
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < NH + NKV; ++j) {
                const bool isq = j < NH;
                const float* x = isq ? &qf[((size_t) t * NH + j) * 2 * HD] : &k[((size_t) t * NKV + (j - NH)) * HD];
                const float* w = isq ? qn.data() : kn.data();
                double ss = 0;
                for (int i = 0; i < HD; ++i) ss += (double) x[i] * x[i];
                const double inv = 1.0 / std::sqrt(ss / HD + 1e-6);
                std::vector<double> y(HD);
                for (int i = 0; i < HD; ++i) y[(size_t) i] = x[i] * inv * w[i];
                std::vector<double> r = y;
                for (int i = 0; i < ROT / 2; ++i) {
                    const double f = std::pow(1e7, -2.0 * i / ROT) * pos[(size_t) t * 3 + rope.axis[i]];
                    r[(size_t) i] = y[(size_t) i] * std::cos(f) - y[(size_t) i + ROT / 2] * std::sin(f);
                    r[(size_t) i + ROT / 2] = y[(size_t) i + ROT / 2] * std::cos(f) + y[(size_t) i] * std::sin(f);
                }
                for (int i = 0; i < HD; ++i) {
                    if (isq) eq.add(qo[((size_t) t * NH + j) * HD + i], r[(size_t) i]);
                    else ek.add(h2f(kc[(((size_t) cell0 + t) * NKV + (j - NH)) * HD + i]), r[(size_t) i]);
                }
            }
        std::printf("attn_qk_prepare: q max abs %.3g rel %.3g; k (fp16 cache) max abs %.3g rel %.3g\n", eq.abs, eq.rel, ek.abs, ek.rel);
        if (eq.abs > 1e-3 || ek.rel > 2e-3) ++fails;
    }

    // ---- attention: a cache of L cells, decode windows and a prompt chunk
    const int64_t L = 3000, max_cells = 4096;
    std::vector<uint16_t> kc((size_t) max_cells * NKV * HD), vc(kc.size());
    for (auto& x : kc) x = basalt::kernels::f16_from_f32(nd(rng));
    for (auto& x : vc) x = basalt::kernels::f16_from_f32(nd(rng));
    const int TP = 300;   // the prompt chunk: rows at cells L - TP .. L - 1
    std::vector<float> q((size_t) TP * NH * HD);
    for (auto& x : q) x = 2.0f * nd(rng);
    uint16_t *d_kc = dev<uint16_t>(kc.size()), *d_vc = dev<uint16_t>(vc.size());
    float *d_q = dev<float>(q.size()), *d_out = dev<float>(q.size());
    float* d_scr = dev<float>(basalt::kernels::gqa_decode_scratch_floats(max_cells, basalt::kernels::kGqaMaxWindow, NH));
    ck(cudaMemcpy(d_kc, kc.data(), kc.size() * 2, cudaMemcpyHostToDevice), "copy");
    ck(cudaMemcpy(d_vc, vc.data(), vc.size() * 2, cudaMemcpyHostToDevice), "copy");
    ck(cudaMemcpy(d_q, q.data(), q.size() * 4, cudaMemcpyHostToDevice), "copy");
    const int64_t p0 = L - TP;
    // reference for the last 8 rows and a few early ones
    const std::vector<int> rows = {0, 1, 57, TP - 8, TP - 7, TP - 6, TP - 5, TP - 4, TP - 3, TP - 2, TP - 1};
    std::vector<std::vector<double>> want(rows.size(), std::vector<double>((size_t) NH * HD));
    for (size_t i = 0; i < rows.size(); ++i) ref_attn(&q[(size_t) rows[i] * NH * HD], kc, vc, p0 + rows[i] + 1, want[i].data());
    // decode: the last 8 rows as windows of 1..8 rows; each row bitwise equal across widths
    constexpr int TW = basalt::kernels::kGqaMaxWindow;
    std::vector<float> single((size_t) TW * NH * HD), got((size_t) TW * NH * HD);
    Err ed;
    bool bitwise = true;
    for (int T = 1; T <= TW; ++T) {
        const int r0 = TP - T;
        basalt::kernels::gqa_decode_attn(d_q + (size_t) r0 * NH * HD, d_kc, d_vc, NH, NKV, p0 + r0, T, d_scr, max_cells, d_out, nullptr);
        ck(cudaMemcpy(got.data(), d_out, (size_t) T * NH * HD * 4, cudaMemcpyDeviceToHost), "copy");
        // row TP-1 is the window's last row: compare with the single-row window
        const float* last = &got[(size_t) (T - 1) * NH * HD];
        if (T == 1) std::memcpy(single.data(), last, (size_t) NH * HD * 4);
        else if (std::memcmp(single.data(), last, (size_t) NH * HD * 4) != 0) bitwise = false;
        for (int t = 0; t < T; ++t) {
            const int row = r0 + t;
            for (size_t i = 0; i < rows.size(); ++i)
                if (rows[i] == row)
                    for (int j = 0; j < NH * HD; ++j) ed.add(got[(size_t) t * NH * HD + j], want[i][(size_t) j]);
        }
    }
    std::printf("gqa_decode_attn: max abs %.3g rel %.3g vs double; the last row bitwise equal across widths 1-%d: %s\n", ed.abs,
                ed.rel, TW, bitwise ? "yes" : "NO");
    if (ed.abs > 2e-4 || !bitwise) ++fails;
    {
        // a captured window reads its first cell on the device and loops its blocks over the chunks in use: bitwise the
        // host-position launch, past the grid's first round of chunks
        const int64_t L2 = 12000, cap2 = 65536, at = L2 - 8;
        std::vector<uint16_t> k2((size_t) L2 * NKV * HD), v2(k2.size());
        for (auto& x : k2) x = basalt::kernels::f16_from_f32(nd(rng));
        for (auto& x : v2) x = basalt::kernels::f16_from_f32(nd(rng));
        uint16_t *d_k2 = dev<uint16_t>((size_t) cap2 * NKV * HD), *d_v2 = dev<uint16_t>((size_t) cap2 * NKV * HD);
        float* d_s2 = dev<float>(basalt::kernels::gqa_decode_scratch_floats(cap2, 8, NH));
        int32_t* d_at = dev<int32_t>(2);
        const int32_t at2[2] = {(int32_t) at, 0};
        ck(cudaMemcpy(d_k2, k2.data(), k2.size() * 2, cudaMemcpyHostToDevice), "copy");
        ck(cudaMemcpy(d_v2, v2.data(), v2.size() * 2, cudaMemcpyHostToDevice), "copy");
        ck(cudaMemcpy(d_at, at2, sizeof at2, cudaMemcpyHostToDevice), "copy");
        std::vector<float> h1((size_t) 8 * NH * HD), h2(h1.size());
        basalt::kernels::gqa_decode_attn(d_q, d_k2, d_v2, NH, NKV, at, 8, d_s2, cap2, d_out, nullptr);
        ck(cudaMemcpy(h1.data(), d_out, h1.size() * 4, cudaMemcpyDeviceToHost), "copy");
        basalt::kernels::gqa_decode_attn(d_q, d_k2, d_v2, NH, NKV, at, 8, d_s2, cap2, d_out, nullptr, d_at);
        ck(cudaMemcpy(h2.data(), d_out, h2.size() * 4, cudaMemcpyDeviceToHost), "copy");
        const bool same = std::memcmp(h1.data(), h2.data(), h1.size() * 4) == 0;
        std::printf("gqa_decode_attn: device first cell at %lld of %lld cells bitwise the host launch: %s\n", (long long) at,
                    (long long) cap2, same ? "yes" : "NO");
        if (!same) ++fails;
        // its cost at a short context in a large capacity, against the host launch's grid of the chunks in use
        const int32_t near[2] = {2999, 0};
        ck(cudaMemcpy(d_at, near, sizeof near, cudaMemcpyHostToDevice), "copy");
        double us[2];
        for (int dv = 0; dv < 2; ++dv) {
            for (int i = 0; i < 5; ++i)
                basalt::kernels::gqa_decode_attn(d_q, d_k2, d_v2, NH, NKV, 2999, 1, d_s2, cap2, d_out, nullptr, dv ? d_at : nullptr);
            cudaDeviceSynchronize();
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 200; ++i)
                basalt::kernels::gqa_decode_attn(d_q, d_k2, d_v2, NH, NKV, 2999, 1, d_s2, cap2, d_out, nullptr, dv ? d_at : nullptr);
            cudaDeviceSynchronize();
            us[dv] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 200;
        }
        std::printf("bench: decode 1 row x 3000 cells in %lld: %.1f us host first cell, %.1f us device first cell\n",
                    (long long) cap2, us[0], us[1]);
        cudaFree(d_k2); cudaFree(d_v2); cudaFree(d_s2); cudaFree(d_at);
    }
    // prompt chunk
    basalt::kernels::gqa_prompt_attn(d_q, d_kc, d_vc, NH, NKV, p0, TP, d_out, nullptr);
    std::vector<float> pr(q.size());
    ck(cudaMemcpy(pr.data(), d_out, pr.size() * 4, cudaMemcpyDeviceToHost), "copy");
    Err ep;
    for (size_t i = 0; i < rows.size(); ++i)
        for (int j = 0; j < NH * HD; ++j) ep.add(pr[(size_t) rows[i] * NH * HD + j], want[i][(size_t) j]);
    std::printf("gqa_prompt_attn: max abs %.3g rel %.3g vs double (fp16 q and p)\n", ep.abs, ep.rel);
    if (ep.abs > 5e-3) ++fails;

    // ---- int8 cache: the same cells as codes and per-64 scales (the write kernel's rule), against the reference on the
    // dequantized values
    {
        const size_t rows = (size_t) max_cells * NKV;
        std::vector<int8_t> k8(rows * HD), v8(rows * HD);
        std::vector<uint16_t> ks(rows * HD / 64), vs(rows * HD / 64), kd(kc.size()), vd(vc.size());
        auto quant = [&](const std::vector<uint16_t>& src, std::vector<int8_t>& q, std::vector<uint16_t>& sc, std::vector<uint16_t>& deq) {
            for (size_t g = 0; g < rows * HD / 64; ++g) {
                float amax = 0;
                for (int i = 0; i < 64; ++i) amax = std::max(amax, std::fabs(h2f(src[g * 64 + i])));
                const uint16_t h = basalt::kernels::f16_from_f32(amax / 127.0f);
                const float s16 = h2f(h);
                sc[g] = h;
                for (int i = 0; i < 64; ++i) {
                    const float c = s16 > 0 ? std::rint(h2f(src[g * 64 + i]) / s16) : 0.0f;
                    q[g * 64 + i] = (int8_t) std::max(-127.0f, std::min(127.0f, c));
                    deq[g * 64 + i] = basalt::kernels::f16_from_f32(q[g * 64 + i] * s16);
                }
            }
        };
        quant(kc, k8, ks, kd);
        quant(vc, v8, vs, vd);
        basalt::kernels::KvInt8 q8;
        ck(cudaMalloc(&q8.k, k8.size()), "malloc");
        ck(cudaMalloc(&q8.v, v8.size()), "malloc");
        ck(cudaMalloc(&q8.ks, ks.size() * 2), "malloc");
        ck(cudaMalloc(&q8.vs, vs.size() * 2), "malloc");
        cudaMemcpy(q8.k, k8.data(), k8.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(q8.v, v8.data(), v8.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(q8.ks, ks.data(), ks.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(q8.vs, vs.data(), vs.size() * 2, cudaMemcpyHostToDevice);
        // the prompt kernel rounds the queries to int8 per 64 dims (amax / 127) for its int8 QK: references on the
        // queries as given and on those
        auto ref8 = [&](const float* qrow, int64_t visible, double* w, double* wq) {
            ref_attn(qrow, kd, vd, visible, w);
            std::vector<float> qq(qrow, qrow + (size_t) NH * HD);
            for (size_t g0 = 0; g0 < qq.size(); g0 += 64) {
                float amax = 0;
                for (int i = 0; i < 64; ++i) amax = std::max(amax, std::fabs(qq[g0 + i]));
                for (int i = 0; i < 64; ++i)
                    qq[g0 + i] = amax > 0 ? (float) std::nearbyint(qq[g0 + i] * (127.0f / amax)) * (amax / 127.0f) : 0.0f;
            }
            ref_attn(qq.data(), kd, vd, visible, wq);
        };
        std::vector<double> w8((size_t) NH * HD), w8q((size_t) NH * HD);
        ref8(&q[(size_t) (TP - 1) * NH * HD], p0 + TP, w8.data(), w8q.data());
        basalt::kernels::gqa_decode_attn(d_q + (size_t) (TP - 1) * NH * HD, nullptr, nullptr, NH, NKV, p0 + TP - 1, 1, d_scr,
                                         max_cells, d_out, nullptr, nullptr, q8);
        std::vector<float> g8((size_t) NH * HD);
        ck(cudaMemcpy(g8.data(), d_out, g8.size() * 4, cudaMemcpyDeviceToHost), "copy");
        Err e8;
        for (int j = 0; j < NH * HD; ++j) e8.add(g8[(size_t) j], w8[(size_t) j]);
        std::printf("int8 cache: decode max abs %.3g rel %.3g vs double on the dequantized cells\n", e8.abs, e8.rel);
        // decode reads code x scale in fp32, the reference its fp16 rounding
        if (e8.abs > 1e-3) ++fails;

        // prompt chunks at the end of the L cells: 300 rows (narrow: 4-warp blocks), the same with split scratch (the
        // cells split over more blocks and merged), and 2048 rows (8-warp blocks), each at rows on and around the block
        // and key-tile edges
        const int TB = 2048;
        std::vector<float> qb((size_t) TB * NH * HD);
        for (auto& x : qb) x = 2.0f * nd(rng);
        float *d_qb = dev<float>(qb.size()), *d_ob = dev<float>(qb.size());
        ck(cudaMemcpy(d_qb, qb.data(), qb.size() * 4, cudaMemcpyHostToDevice), "copy");
        const size_t split_floats = basalt::kernels::gqa_prompt_split_floats(TP, NH, 8);
        float* d_split = dev<float>(split_floats);
        for (int cfg = 0; cfg < 3; ++cfg) {
            const int T = cfg < 2 ? TP : TB;
            const int64_t pp = L - T;
            const float* qh = cfg < 2 ? q.data() : qb.data();
            basalt::kernels::gqa_prompt_attn(cfg < 2 ? d_q : d_qb, nullptr, nullptr, NH, NKV, pp, T, d_ob, nullptr, q8,
                                             cfg == 1 ? d_split : nullptr, cfg == 1 ? split_floats : 0);
            std::vector<float> po((size_t) T * NH * HD);
            ck(cudaMemcpy(po.data(), d_ob, po.size() * 4, cudaMemcpyDeviceToHost), "copy");
            Err p8, p8q;
            for (const int r : {0, 1, 57, 63, 64, 65, 127, 128, 255, T - 2, T - 1, 1000, 2047}) {
                if (r >= T) continue;
                ref8(&qh[(size_t) r * NH * HD], pp + r + 1, w8.data(), w8q.data());
                for (int j = 0; j < NH * HD; ++j) {
                    p8.add(po[(size_t) r * NH * HD + j], w8[(size_t) j]);
                    p8q.add(po[(size_t) r * NH * HD + j], w8q[(size_t) j]);
                }
            }
            static const char* what[3] = {"300 rows", "300 rows split", "2048 rows"};
            std::printf("int8 cache: prompt %s max abs %.3g (%.3g against int8-rounded queries) vs double on the dequantized "
                        "cells\n", what[cfg], p8.abs, p8q.abs);
            // the prompt's QK reads int8 queries, its P V fp16 probabilities summed in fp16 per 64-cell tile
            if (p8q.abs > 5e-3 || p8.abs > 3e-2) ++fails;
        }
        cudaFree(d_qb);
        cudaFree(d_ob);
        cudaFree(d_split);
    }

    // ---- microbench: one decode row at 3000 cells, and the 300-row chunk
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    const int reps = 50;
    cudaEventRecord(a);
    for (int i = 0; i < reps; ++i)
        basalt::kernels::gqa_decode_attn(d_q, d_kc, d_vc, NH, NKV, L - 1, 1, d_scr, max_cells, d_out, nullptr);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    const double us_dec = 1000.0 * ms / reps, bytes = (double) L * NKV * HD * 2 * 2;
    cudaEventRecord(a);
    for (int i = 0; i < 10; ++i) basalt::kernels::gqa_prompt_attn(d_q, d_kc, d_vc, NH, NKV, p0, TP, d_out, nullptr);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    cudaEventElapsedTime(&ms, a, b);
    const double us_pr = 1000.0 * ms / 10;
    double flops = 0;
    for (int t = 0; t < TP; ++t) flops += 4.0 * NH * HD * (double) (p0 + t + 1);
    std::printf("bench: decode 1 row x %lld cells %.1f us (%.0f GB/s of K/V); prompt %d rows %.1f us (%.1f TFLOP/s)\n",
                (long long) L, us_dec, bytes / us_dec / 1e3, TP, us_pr, flops / us_pr / 1e6);
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {   // a 2048-row chunk at the end of 4K and 16K caches, fp16 and int8
        for (const int64_t cells : {4096, 16384}) {
            const int64_t T2 = 2048, pp = cells - T2;
            uint16_t* bk = dev<uint16_t>((size_t) cells * NKV * HD);
            uint16_t* bv = dev<uint16_t>((size_t) cells * NKV * HD);
            float* bq = dev<float>((size_t) T2 * NH * HD);
            float* bo = dev<float>((size_t) T2 * NH * HD);
            basalt::kernels::KvInt8 b8;
            cudaMalloc(&b8.k, (size_t) cells * NKV * HD);
            cudaMalloc(&b8.v, (size_t) cells * NKV * HD);
            cudaMalloc(&b8.ks, (size_t) cells * NKV * HD / 64 * 2);
            cudaMalloc(&b8.vs, (size_t) cells * NKV * HD / 64 * 2);
            cudaMemset(bk, 0x2A, (size_t) cells * NKV * HD * 2);
            cudaMemset(bv, 0x2A, (size_t) cells * NKV * HD * 2);
            cudaMemset(bq, 0x3C, (size_t) T2 * NH * HD * 4);
            cudaMemset(b8.k, 0x11, (size_t) cells * NKV * HD);
            cudaMemset(b8.v, 0x11, (size_t) cells * NKV * HD);
            cudaMemset(b8.ks, 0x20, (size_t) cells * NKV * HD / 64 * 2);
            cudaMemset(b8.vs, 0x20, (size_t) cells * NKV * HD / 64 * 2);
            double fl = 0;
            for (int64_t t = 0; t < T2; ++t) fl += 4.0 * NH * HD * (double) (pp + t + 1);
            // fp16, int8, int8 with scratch for three key splits
            const size_t sf = basalt::kernels::gqa_prompt_split_floats(T2, NH, 3);
            float* bs = dev<float>(sf);
            for (int run = 0; run < 3; ++run) {
                const basalt::kernels::KvInt8 kv = run ? b8 : basalt::kernels::KvInt8{};
                float* scr = run == 2 ? bs : nullptr;
                basalt::kernels::gqa_prompt_attn(bq, bk, bv, NH, NKV, pp, T2, bo, nullptr, kv, scr, sf);
                cudaEventRecord(a);
                for (int i = 0; i < 5; ++i) basalt::kernels::gqa_prompt_attn(bq, bk, bv, NH, NKV, pp, T2, bo, nullptr, kv, scr, sf);
                cudaEventRecord(b);
                cudaEventSynchronize(b);
                cudaEventElapsedTime(&ms, a, b);
                std::printf("bench: prompt chunk 2048 rows at %lld cells, %s cache: %.2f ms (%.1f TFLOP/s)\n", (long long) cells,
                            run == 0 ? "fp16" : run == 1 ? "int8" : "int8 split scratch", ms / 5, fl / (ms / 5) / 1e9);
            }
            cudaFree(bs);
            cudaFree(bk); cudaFree(bv); cudaFree(bq); cudaFree(bo); cudaFree(b8.k); cudaFree(b8.v); cudaFree(b8.ks); cudaFree(b8.vs);
        }
    }
    std::printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
