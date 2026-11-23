// tests/kernels/gqa_h128_parity.cpp - gqa_decode_attn and gqa_prompt_attn (basalt/kernels/gqa_attn.hpp) at head width
// 128 with 16 query heads per KV head (Nemotron-H Super: 32 query heads, 2 KV heads, no positional encoding) over an
// fp16 cache, against a CPU reference in double precision.
//
// Checks: a decode row's output is bitwise the same whatever the window width or the row's place in it; decode and
// the fp16 prompt kernel against the reference; the prompt kernel's rows against decode's within fp16 rounding.  Exit
// 77 without a GPU.  Prints a microbench line for each.
#include "basalt/kernels/f16_bits.hpp"
#include "basalt/kernels/gqa_attn.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int NH = 32, NKV = 2, HD = 128;
constexpr int64_t CELLS = 700;

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

// out[h][d] for one query row seeing cells [0, visible)
void ref_attn(const float* q, const std::vector<uint16_t>& kc, const std::vector<uint16_t>& vc, int64_t visible, double* out) {
    const double scale = 1.0 / std::sqrt((double) HD);
    for (int h = 0; h < NH; ++h) {
        const int kvh = h / (NH / NKV);
        std::vector<double> s((size_t) visible);
        double mx = -1e300;
        for (int64_t c = 0; c < visible; ++c) {
            double d = 0;
            for (int i = 0; i < HD; ++i) d += (double) q[h * HD + i] * h2f(kc[((size_t) c * NKV + kvh) * HD + i]);
            s[(size_t) c] = d * scale;
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
    void add(const float* got, const double* ref, int n) {
        double num = 0, den = 0;
        for (int i = 0; i < n; ++i) {
            const double e = std::fabs(got[i] - ref[i]);
            abs = std::max(abs, e);
            num += e;
            den += std::fabs(ref[i]);
        }
        rel = std::max(rel, num / (den + 1e-30));
    }
};

}  // namespace

int main() {
    namespace K = basalt::kernels;
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
    std::mt19937 rng(41);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> kc((size_t) CELLS * NKV * HD), vc(kc.size());
    for (auto& v : kc) v = K::f16_from_f32(nd(rng));
    for (auto& v : vc) v = K::f16_from_f32(nd(rng));
    std::vector<float> q((size_t) CELLS * NH * HD);   // a query row per position
    for (auto& v : q) v = 1.5f * nd(rng);
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    uint16_t* dk = dev<uint16_t>(kc.size());
    uint16_t* dv = dev<uint16_t>(vc.size());
    float* dq = dev<float>(q.size());
    float* dout = dev<float>(q.size());
    ck(cudaMemcpy(dk, kc.data(), kc.size() * 2, cudaMemcpyHostToDevice), "k");
    ck(cudaMemcpy(dv, vc.data(), vc.size() * 2, cudaMemcpyHostToDevice), "v");
    ck(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice), "q");
    float* scratch = dev<float>(K::gqa_decode_scratch_floats(CELLS, K::kGqaMaxWindow, NH, HD));
    const int rowf = NH * HD;
    int failures = 0;

    // decode: every position alone (T = 1), then windows of 4, 8 and 16 rows at several starts, bitwise the rows alone
    std::vector<float> alone((size_t) CELLS * rowf), win((size_t) K::kGqaMaxWindow * rowf);
    for (int64_t p = 0; p < CELLS; ++p)
        K::gqa_decode_attn(dq + p * rowf, dk, dv, NH, NKV, p, 1, scratch, CELLS, dout + p * rowf, s, nullptr, {}, nullptr,
                           nullptr, {}, HD);
    ck(cudaStreamSynchronize(s), "decode");
    ck(cudaMemcpy(alone.data(), dout, alone.size() * 4, cudaMemcpyDeviceToHost), "alone");
    bool window_same = true;
    for (int T : {4, 8, 16})
        for (int64_t p0 : {(int64_t) 0, (int64_t) 31, (int64_t) 250, CELLS - T}) {
            K::gqa_decode_attn(dq + p0 * rowf, dk, dv, NH, NKV, p0, T, scratch, CELLS, dout, s, nullptr, {}, nullptr, nullptr,
                               {}, HD);
            ck(cudaMemcpy(win.data(), dout, (size_t) T * rowf * 4, cudaMemcpyDeviceToHost), "window");
            if (std::memcmp(win.data(), alone.data() + p0 * rowf, (size_t) T * rowf * 4)) window_same = false;
        }
    Err ed;
    std::vector<double> ref(rowf);
    for (int64_t p : {(int64_t) 0, (int64_t) 1, (int64_t) 40, (int64_t) 333, CELLS - 1}) {
        ref_attn(q.data() + p * rowf, kc, vc, p + 1, ref.data());
        ed.add(alone.data() + p * rowf, ref.data(), rowf);
    }
    {
        const bool ok = window_same && ed.rel < 2e-3;
        std::printf("decode h128 G16: windows of 4/8/16 rows = rows alone %s; vs double max abs %.2e rel %.2e  %s\n",
                    window_same ? "bitwise" : "DIFFER", ed.abs, ed.rel, ok ? "ok" : "FAIL");
        failures += !ok;
    }

    // the fp16 prompt kernel: a chunk of 600 rows after 100 cached cells, against the reference and decode
    const int64_t P0 = 100, PT = CELLS - P0;
    K::gqa_prompt_attn(dq + P0 * rowf, dk, dv, NH, NKV, P0, PT, dout, s, {}, nullptr, 0, {}, HD);
    ck(cudaStreamSynchronize(s), "prompt");
    std::vector<float> pr((size_t) PT * rowf);
    ck(cudaMemcpy(pr.data(), dout, pr.size() * 4, cudaMemcpyDeviceToHost), "prompt out");
    Err ep, epd;
    for (int64_t t : {(int64_t) 0, (int64_t) 5, (int64_t) 97, (int64_t) 400, PT - 1}) {
        ref_attn(q.data() + (P0 + t) * rowf, kc, vc, P0 + t + 1, ref.data());
        ep.add(pr.data() + t * rowf, ref.data(), rowf);
    }
    for (int64_t t = 0; t < PT; ++t) {
        std::vector<double> d(alone.begin() + (P0 + t) * rowf, alone.begin() + (P0 + t + 1) * rowf);
        epd.add(pr.data() + t * rowf, d.data(), rowf);
    }
    {
        const bool ok = ep.rel < 5e-3 && epd.rel < 5e-3 && std::isfinite(ep.rel);
        std::printf("prompt fp16 h128 G16 (%lld rows after %lld cells): vs double max abs %.2e rel %.2e; vs decode rel %.2e  %s\n",
                    (long long) PT, (long long) P0, ep.abs, ep.rel, epd.rel, ok ? "ok" : "FAIL");
        failures += !ok;
    }

    // the microbench
    {
        const int64_t big = 8192;
        uint16_t* bk = dev<uint16_t>((size_t) big * NKV * HD);
        uint16_t* bv = dev<uint16_t>((size_t) big * NKV * HD);
        float* bq = dev<float>((size_t) 2048 * rowf);
        float* bo = dev<float>((size_t) 2048 * rowf);
        float* bs = dev<float>(K::gqa_decode_scratch_floats(big, K::kGqaMaxWindow, NH, HD));
        cudaMemset(bk, 0, (size_t) big * NKV * HD * 2);
        cudaMemset(bv, 0, (size_t) big * NKV * HD * 2);
        cudaMemset(bq, 0, (size_t) 2048 * rowf * 4);
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        auto time_us = [&](int R, auto&& fn) {
            for (int i = 0; i < 3; ++i) fn();
            cudaEventRecord(a, s);
            for (int i = 0; i < R; ++i) fn();
            cudaEventRecord(b, s);
            cudaEventSynchronize(b);
            float ms = 0;
            cudaEventElapsedTime(&ms, a, b);
            return 1000.0 * ms / R;
        };
        const double kv_mb = (double) big * NKV * HD * 2 * 2 / 1e6;
        for (int T : {1, 8}) {
            const double us = time_us(100, [&] {
                K::gqa_decode_attn(bq, bk, bv, NH, NKV, big - T, T, bs, big, bo, s, nullptr, {}, nullptr, nullptr, {}, HD);
            });
            std::printf("bench decode h128 T=%d at %lld cells: %.1f us (%.0f GB/s of K/V)\n", T, (long long) big, us, kv_mb * 1e3 / us);
        }
        const double us = time_us(5, [&] { K::gqa_prompt_attn(bq, bk, bv, NH, NKV, 0, 2048, bo, s, {}, nullptr, 0, {}, HD); });
        const double flops = 4.0 * NH * HD * 2048.0 * 2048.0 / 2;   // QK^T and P V over the causal half
        std::printf("bench prompt fp16 h128 2048 rows from 0: %.0f us (%.1f TFLOP/s)\n", us, flops / us / 1e6);
        for (void* p : {(void*) bk, (void*) bv, (void*) bq, (void*) bo, (void*) bs}) cudaFree(p);
    }
    std::printf("%s\n", failures ? "FAILED" : "all ok");
    return failures ? 1 : 0;
}
