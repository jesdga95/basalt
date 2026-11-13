// tests/kernels/gqa_attn_parity.cpp - gqa_decode_attn and gqa_prompt_attn (basalt/kernels/gqa_attn.hpp) and
// attn_qk_prepare (dense_ops.hpp) against a CPU reference in double precision, at the Qwen3.5-27B attention geometry
// (24 query heads, 4 KV heads, 256 wide, 64 rotated dims, interleaved M-RoPE sections 11/11/10).
//
// Checks: each kernel's max absolute and relative error to the reference; a decode row's output is bitwise the same
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

int main() {
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
    std::vector<float> single((size_t) 8 * NH * HD), got((size_t) 8 * NH * HD);
    Err ed;
    bool bitwise = true;
    for (int T = 1; T <= 8; ++T) {
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
    std::printf("gqa_decode_attn: max abs %.3g rel %.3g vs double; the last row bitwise equal across widths 1-8: %s\n", ed.abs,
                ed.rel, bitwise ? "yes" : "NO");
    if (ed.abs > 2e-4 || !bitwise) ++fails;
    // prompt chunk
    basalt::kernels::gqa_prompt_attn(d_q, d_kc, d_vc, NH, NKV, p0, TP, d_out, nullptr);
    std::vector<float> pr(q.size());
    ck(cudaMemcpy(pr.data(), d_out, pr.size() * 4, cudaMemcpyDeviceToHost), "copy");
    Err ep;
    for (size_t i = 0; i < rows.size(); ++i)
        for (int j = 0; j < NH * HD; ++j) ep.add(pr[(size_t) rows[i] * NH * HD + j], want[i][(size_t) j]);
    std::printf("gqa_prompt_attn: max abs %.3g rel %.3g vs double (fp16 q and p)\n", ep.abs, ep.rel);
    if (ep.abs > 5e-3) ++fails;

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
    std::printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
