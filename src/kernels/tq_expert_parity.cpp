// src/kernels/tq_expert_parity.cpp - the APR trellis experts (TQ2_T .. TQP7) and their Hadamard fold, every path.
//
//     build/tq_expert_parity            random trellis blobs at the model's geometry, plain and Hadamard-folded
//     build/tq_expert_parity --bench    the grouped decode kernels: us per call and weight GB/s
//
// (a) float reference: the kernels' own decode to float (tq_dequant_row, the fp16 codebook times the block scale),
// the rotation by hadamard_rotate, a SwiGLU expert in double.  (b) the CPU path (native_quant_act / native_gu_rows /
// native_quant_h / native_down_rows on Q8_0 activations), every ISA bit for bit.  (c) the GPU path
// (native_expert_grouped, q8_1 activations, the int8 codebook): the decode-once kernels against the per-entry ones
// bit for bit, and a token alone against the same token in a group bit for bit.  (b) and (c) differ from (a) by their
// activation and codebook rounding only.  (d) the rotation kernels against the CPU's, and the prompt path's fp16
// dequantizer against tq_dequant_row.
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/hadamard.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/cpu/tq_cpu.hpp"
#include "basalt/kernels/iq_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = basalt::kernels::cpu;
namespace k = basalt::kernels;

namespace {
constexpr int NT = 3;
constexpr int64_t H = 2560, FF = 640;

double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) { n += std::fabs((double) a[i] - b[i]); d += std::fabs((double) b[i]); }
    return n / (d + 1e-30);
}

const char* type_name(int t) {
    static const char* names[] = { "TQ2_T", "TQK6", "TQK7", "TQP5", "TQP6", "TQP7" };
    return t >= 144 && t <= 149 ? names[t - 144] : "?";
}

uint16_t f16_bits(float v) {
    const __half h = __float2half(v);
    uint16_t b;
    std::memcpy(&b, &h, 2);
    return b;
}

// random trellis rows: any stream is a valid trellis; the block scales are small positive fp16 values
void random_rows(int type, uint8_t* dst, int64_t rows, int64_t n, std::mt19937& rng) {
    const size_t row = k::iq_row_bytes(type, n), bsz = row / (size_t) (n / 128);
    std::uniform_real_distribution<float> sd(0.004f, 0.03f);
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t b = 0; b < n / 128; ++b) {
            uint8_t* blk = dst + (size_t) r * row + (size_t) b * bsz;
            const uint16_t d = f16_bits(sd(rng));
            std::memcpy(blk, &d, 2);
            for (size_t i = 2; i < bsz; ++i) blk[i] = (uint8_t) rng();
        }
}

#define CK(x) do { if ((x) != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(cudaGetLastError())); std::exit(1); } } while (0)

struct Gpu {
    uint8_t* blob = nullptr;
    void *xq = nullptr, *scr = nullptr;
    float *x = nullptr, *out = nullptr;
    unsigned long long* ptr = nullptr;
    int32_t *start = nullptr, *count = nullptr, *dst = nullptr, *tok = nullptr;
};

// the GPU expert on `nt` tokens of x (one group), the result in `out` (nt x H)
void gpu_expert(const Gpu& g, const cpu::NativeFmt& f, const float* x, int nt, std::vector<float>& out, cudaStream_t s) {
    CK(cudaMemcpyAsync(g.x, x, (size_t) nt * H * 4, cudaMemcpyHostToDevice, s));
    const int32_t start[2] = { 0, nt }, one = 1;
    int32_t idx[NT];
    for (int i = 0; i < nt; ++i) idx[i] = i;
    CK(cudaMemcpyAsync(g.start, start, 8, cudaMemcpyHostToDevice, s));
    CK(cudaMemcpyAsync(g.count, &one, 4, cudaMemcpyHostToDevice, s));
    CK(cudaMemcpyAsync(g.dst, idx, (size_t) nt * 4, cudaMemcpyHostToDevice, s));
    CK(cudaMemcpyAsync(g.tok, idx, (size_t) nt * 4, cudaMemcpyHostToDevice, s));
    if (f.had_gu) k::hadamard_q8_1_rows(g.x, nt, H, f.had_block, k::native_expert_gu_signs(f), g.xq, s);
    else k::quantize_q8_1_rows(g.x, nt, H, g.xq, s);
    k::native_expert_grouped(k::native_expert_layout(f), g.ptr, g.start, g.count, g.dst, g.tok, 1, nt, g.xq, g.scr,
                             g.out, s);
    out.assign((size_t) nt * H, 0.f);
    CK(cudaMemcpyAsync(out.data(), g.out, out.size() * 4, cudaMemcpyDeviceToHost, s));
    CK(cudaStreamSynchronize(s));
}

int check_pair(int gu, int dn, bool had, const Gpu& g, cudaStream_t s) {
    int failures = 0;
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(gu, dn, H, FF, f, err)) { std::printf("  native_fmt: %s\n", err.c_str()); return 1; }
    // one sign vector per width for the process, as the engine's spec: hadamard_device_signs keeps a device copy per
    // host vector
    static std::vector<float> sx((size_t) H), sh((size_t) FF);
    static const bool signed_once = [] {
        std::mt19937 r(5);
        for (auto& v : sx) v = (r() & 1) ? 1.f : -1.f;
        for (auto& v : sh) v = (r() & 1) ? 1.f : -1.f;
        return true;
    }();
    (void) signed_once;
    std::mt19937 rng(7 + gu * 13 + dn);
    if (had) {
        f.had_block = 128;
        f.had_gu = f.had_d = true;
        f.gu_signs = sx.data();
        f.d_signs = sh.data();
    }
    std::vector<uint8_t> blob(f.bytes);
    random_rows(gu, blob.data(), 2 * FF, H, rng);
    random_rows(dn, blob.data() + f.down_off, H, FF, rng);
    std::printf("%s/%s%s:\n", type_name(gu), type_name(dn), had ? " Hadamard-folded" : "");

    // (a) the float reference
    std::vector<float> G((size_t) FF * H), U((size_t) FF * H), D((size_t) H * FF);
    for (int64_t r = 0; r < FF; ++r) {
        cpu::tq_dequant_row(gu, blob.data() + r * f.gu_row, (int) H, G.data() + r * H);
        cpu::tq_dequant_row(gu, blob.data() + f.up_off + r * f.gu_row, (int) H, U.data() + r * H);
    }
    for (int64_t r = 0; r < H; ++r) cpu::tq_dequant_row(dn, blob.data() + f.down_off + r * f.d_row, (int) FF, D.data() + r * FF);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) NT * H), ref((size_t) NT * H);
    for (auto& v : x) v = nd(rng);
    for (int t = 0; t < NT; ++t) {
        std::vector<float> xr(x.begin() + t * H, x.begin() + (t + 1) * H), h((size_t) FF);
        if (had) cpu::hadamard_rotate(xr.data(), (int) H, 128, sx.data());
        for (int64_t r = 0; r < FF; ++r) {
            double a = 0, b = 0;
            for (int64_t i = 0; i < H; ++i) { a += (double) G[r * H + i] * xr[i]; b += (double) U[r * H + i] * xr[i]; }
            h[r] = (float) (a / (1.0 + std::exp(-a)) * b);
        }
        if (had) cpu::hadamard_rotate(h.data(), (int) FF, 128, sh.data());
        for (int64_t r = 0; r < H; ++r) {
            double o = 0;
            for (int64_t i = 0; i < FF; ++i) o += (double) D[r * FF + i] * h[i];
            ref[t * H + r] = (float) o;
        }
    }

    // (b) the CPU path, then each ISA's rows against the scalar kernel's
    {
        std::vector<std::vector<uint8_t>> act(NT, std::vector<uint8_t>(cpu::kNativeActBytes)),
            hq(NT, std::vector<uint8_t>(cpu::kNativeHBytes));
        std::vector<float> ff((size_t) NT * FF), out((size_t) NT * H);
        const void* a[NT];
        const void* hp[NT];
        float* fp[NT];
        float* op[NT];
        for (int t = 0; t < NT; ++t) {
            cpu::native_quant_act(f, x.data() + t * H, act[(size_t) t].data());
            a[t] = act[(size_t) t].data();
            fp[t] = ff.data() + t * FF;
            op[t] = out.data() + t * H;
        }
        cpu::native_gu_rows(f, blob.data(), a, NT, fp, 0, (int) FF);
        for (int t = 0; t < NT; ++t) { cpu::native_quant_h(f, fp[t], hq[(size_t) t].data()); hp[t] = hq[(size_t) t].data(); }
        cpu::native_down_rows(f, blob.data(), hp, NT, op, 0, (int) H);
        const double rc = rel(out, ref);
        std::printf("  cpu (%s) vs float reference: rel %.2e\n", cpu::tq_isa_name(cpu::tq_isa()), rc);
        if (!(rc < 3e-2)) ++failures;
        std::vector<float> want((size_t) NT * H), got((size_t) NT * H);
        float* wp[NT];
        float* gp[NT];
        for (int t = 0; t < NT; ++t) { wp[t] = want.data() + t * H; gp[t] = got.data() + t * H; }
        cpu::tq_rows(cpu::TqIsa::scalar, dn, blob.data() + f.down_off, f.d_row, (int) FF, hp, NT, wp, 0, (int) H);
        for (cpu::TqIsa isa : { cpu::TqIsa::avx2, cpu::TqIsa::avx512 }) {
            if (!cpu::tq_isa_ok(isa)) continue;
            cpu::tq_rows(isa, dn, blob.data() + f.down_off, f.d_row, (int) FF, hp, NT, gp, 0, (int) H);
            const bool same = std::memcmp(want.data(), got.data(), want.size() * 4) == 0;
            std::printf("  cpu %s down rows vs scalar: %s\n", cpu::tq_isa_name(isa), same ? "bitwise equal" : "DIFFER");
            if (!same) ++failures;
        }
        // one token alone gives the same bits as in the group
        float* g1[1] = { got.data() };
        const void* h1[1] = { hp[NT - 1] };
        cpu::tq_rows(cpu::tq_isa(), dn, blob.data() + f.down_off, f.d_row, (int) FF, h1, 1, g1, 0, (int) H);
        cpu::tq_rows(cpu::tq_isa(), dn, blob.data() + f.down_off, f.d_row, (int) FF, hp, NT, wp, 0, (int) H);
        const bool alone = std::memcmp(got.data(), want.data() + (NT - 1) * H, (size_t) H * 4) == 0;
        std::printf("  cpu token alone vs in a group of %d: %s\n", NT, alone ? "bitwise equal" : "DIFFER");
        if (!alone) ++failures;
    }

    // (c) the GPU path
    {
        CK(cudaMemcpy(g.blob, blob.data(), blob.size(), cudaMemcpyHostToDevice));
        const unsigned long long p = (unsigned long long) g.blob;
        CK(cudaMemcpy(g.ptr, &p, 8, cudaMemcpyHostToDevice));
        std::vector<float> got, old_k, one;
        gpu_expert(g, f, x.data(), NT, got, s);
        const double rg = rel(got, ref);
        std::printf("  gpu vs float reference: rel %.2e\n", rg);
        if (!(rg < 3e-2)) ++failures;
        k::iq_set_old_kernels(true);
        gpu_expert(g, f, x.data(), NT, old_k, s);
        k::iq_set_old_kernels(false);
        const bool same = std::memcmp(got.data(), old_k.data(), got.size() * 4) == 0;
        std::printf("  gpu decode-once vs per-entry kernels: %s\n", same ? "bitwise equal" : "DIFFER");
        if (!same) ++failures;
        gpu_expert(g, f, x.data() + (NT - 1) * H, 1, one, s);
        const bool alone = std::memcmp(one.data(), got.data() + (NT - 1) * H, (size_t) H * 4) == 0;
        std::printf("  gpu token alone vs in a group of %d: %s\n", NT, alone ? "bitwise equal" : "DIFFER");
        if (!alone) ++failures;
    }

    // (d) the rotations against the CPU's, and the prompt path's dequantizer
    if (had) {
        std::vector<float> xr(x);
        for (int t = 0; t < NT; ++t) cpu::hadamard_rotate(xr.data() + t * H, (int) H, 128, sx.data());
        const float* dsx = k::native_expert_gu_signs(f);
        CK(cudaMemcpy(g.x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        k::hadamard_rows(g.x, g.out, NT, H, 128, dsx, s);
        std::vector<float> gx(x.size());
        CK(cudaMemcpyAsync(gx.data(), g.out, gx.size() * 4, cudaMemcpyDeviceToHost, s));
        CK(cudaStreamSynchronize(s));
        const bool same = std::memcmp(gx.data(), xr.data(), gx.size() * 4) == 0;
        std::printf("  gpu hadamard_rows vs cpu hadamard_rotate: %s (rel %.2e)\n", same ? "bitwise equal" : "differ",
                    rel(gx, xr));
        if (!(rel(gx, xr) < 1e-6)) ++failures;
        // the fused rotation + q8_1 against the two kernels
        const size_t qb = (size_t) NT * (H / 32) * 36;
        std::vector<uint8_t> q1(qb), q2(qb);
        k::hadamard_q8_1_rows(g.x, NT, H, 128, dsx, g.xq, s);
        CK(cudaMemcpyAsync(q1.data(), g.xq, qb, cudaMemcpyDeviceToHost, s));
        k::quantize_q8_1_rows(g.out, NT, H, g.xq, s);
        CK(cudaMemcpyAsync(q2.data(), g.xq, qb, cudaMemcpyDeviceToHost, s));
        CK(cudaStreamSynchronize(s));
        const bool qsame = std::memcmp(q1.data(), q2.data(), qb) == 0;
        std::printf("  hadamard_q8_1_rows vs hadamard_rows + quantize_q8_1_rows: %s\n", qsame ? "bitwise equal" : "DIFFER");
        if (!qsame) ++failures;
        // the f16 form (prompt path): within fp16 rounding of the float rotation
        std::vector<uint16_t> xh(x.size()), xo(x.size());
        for (size_t i = 0; i < x.size(); ++i) xh[i] = f16_bits(x[i]);
        CK(cudaMemcpy(g.out, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice));
        k::hadamard_rows_f16((uint16_t*) g.out, NT, H, 128, dsx, s);
        CK(cudaMemcpyAsync(xo.data(), g.out, xo.size() * 2, cudaMemcpyDeviceToHost, s));
        CK(cudaStreamSynchronize(s));
        std::vector<float> xf(x.size());
        for (size_t i = 0; i < xf.size(); ++i) { __half_raw r; r.x = xo[i]; xf[i] = __half2float(__half(r)); }
        std::printf("  gpu hadamard_rows_f16 vs cpu float rotation: rel %.2e\n", rel(xf, xr));
        if (!(rel(xf, xr) < 2e-3)) ++failures;
    }
    {
        const int64_t n = FF * H;   // the down matrix
        std::vector<uint16_t> dq((size_t) n);
        uint16_t* d16 = (uint16_t*) g.out;
        k::iq_dequant_f16(dn, g.blob + f.down_off, n, d16, s);
        CK(cudaMemcpyAsync(dq.data(), d16, dq.size() * 2, cudaMemcpyDeviceToHost, s));
        CK(cudaStreamSynchronize(s));
        size_t bad = 0;
        for (size_t i = 0; i < dq.size(); ++i) bad += dq[i] != f16_bits(D[i]);
        std::printf("  gpu fp16 dequantizer vs tq_dequant_row: %s\n", bad ? "DIFFER" : "fp16 of the same floats");
        if (bad) ++failures;
    }
    return failures;
}

// the grouped decode kernels as the verify window runs them: `groups` experts of `per` entries each, distinct blobs
void bench(int gu, int dn, bool had, int groups, int per, cudaStream_t s) {
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(gu, dn, H, FF, f, err)) return;
    static std::vector<float> sx((size_t) H, 1.f), sh((size_t) FF, 1.f);
    if (had) { f.had_block = 128; f.had_gu = f.had_d = true; f.gu_signs = sx.data(); f.d_signs = sh.data(); }
    const int ne = groups * per;
    uint8_t* blobs = nullptr;
    void *xq = nullptr, *scr = nullptr;
    float *x = nullptr, *out = nullptr;
    unsigned long long* ptr = nullptr;
    int32_t *start = nullptr, *count = nullptr, *idx = nullptr, *tok = nullptr;
    CK(cudaMalloc(&blobs, f.bytes * groups));
    CK(cudaMalloc(&x, (size_t) ne * H * 4));
    CK(cudaMalloc(&xq, (size_t) ne * (H / 32) * 36));
    CK(cudaMalloc(&scr, k::native_expert_scratch_bytes(ne, FF)));
    CK(cudaMalloc(&out, (size_t) ne * H * 4));
    CK(cudaMalloc(&ptr, (size_t) groups * 8));
    CK(cudaMalloc(&start, (size_t) (groups + 1) * 4));
    CK(cudaMalloc(&count, 4));
    CK(cudaMalloc(&idx, (size_t) ne * 4));
    CK(cudaMalloc(&tok, (size_t) ne * 4));
    std::mt19937 rng(3);
    std::vector<uint8_t> blob(f.bytes);
    for (int e = 0; e < groups; ++e) {
        random_rows(gu, blob.data(), 2 * FF, H, rng);
        random_rows(dn, blob.data() + f.down_off, H, FF, rng);
        CK(cudaMemcpy(blobs + f.bytes * e, blob.data(), f.bytes, cudaMemcpyHostToDevice));
    }
    std::vector<unsigned long long> hp((size_t) groups);
    std::vector<int32_t> hs((size_t) groups + 1), hi((size_t) ne), ht((size_t) ne);
    for (int e = 0; e < groups; ++e) { hp[(size_t) e] = (unsigned long long) (blobs + f.bytes * e); hs[(size_t) e] = e * per; }
    hs[(size_t) groups] = ne;
    for (int i = 0; i < ne; ++i) { hi[(size_t) i] = i; ht[(size_t) i] = i % per; }
    CK(cudaMemcpy(ptr, hp.data(), hp.size() * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(start, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(count, &groups, 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(idx, hi.data(), hi.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(tok, ht.data(), ht.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(x, 0, (size_t) ne * H * 4));
    const auto L = k::native_expert_layout(f);
    const float* dsx = k::native_expert_gu_signs(f);
    auto run = [&] {
        if (had) k::hadamard_q8_1_rows(x, per, H, 128, dsx, xq, s);
        else k::quantize_q8_1_rows(x, per, H, xq, s);
        k::native_expert_grouped(L, ptr, start, count, idx, tok, groups, ne, xq, scr, out, s);
    };
    for (int i = 0; i < 20; ++i) run();
    CK(cudaStreamSynchronize(s));
    const int it = 200;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < it; ++i) run();
    CK(cudaStreamSynchronize(s));
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / it;
    std::printf("bench %s/%s%s: %d groups x %d entries: %.1f us per call, %.0f GB/s of weights\n", type_name(gu),
                type_name(dn), had ? " folded" : "", groups, per, us, (double) f.bytes * groups / (us * 1e3));
    cudaFree(blobs); cudaFree(x); cudaFree(xq); cudaFree(scr); cudaFree(out); cudaFree(ptr); cudaFree(start);
    cudaFree(count); cudaFree(idx); cudaFree(tok);
}
}  // namespace

int main(int argc, char** argv) {
    const bool do_bench = argc > 1 && std::string(argv[1]) == "--bench";
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    Gpu g;
    CK(cudaMalloc(&g.blob, 4 << 20));
    CK(cudaMalloc(&g.x, (size_t) NT * H * 4));
    CK(cudaMalloc(&g.xq, (size_t) NT * (H / 32) * 36));
    CK(cudaMalloc(&g.scr, k::native_expert_scratch_bytes(NT, FF)));
    CK(cudaMalloc(&g.out, (size_t) FF * H * 2 > (size_t) NT * H * 4 ? (size_t) FF * H * 2 : (size_t) NT * H * 4));
    CK(cudaMalloc(&g.ptr, 8));
    CK(cudaMalloc(&g.start, 8));
    CK(cudaMalloc(&g.count, 4));
    CK(cudaMalloc(&g.dst, NT * 4));
    CK(cudaMalloc(&g.tok, NT * 4));
    static const int pairs[][2] = { { 147, 149 }, { 148, 148 }, { 149, 147 }, { 144, 144 }, { 145, 146 } };
    if (do_bench) {
        for (const auto& p : pairs)
            for (int per : { 1, 2, 4 }) bench(p[0], p[1], true, 64, per, s);
        return 0;
    }
    int failures = 0;
    for (const auto& p : pairs)
        for (bool had : { false, true }) failures += check_pair(p[0], p[1], had, g, s);
    std::printf("%s (%d failed check%s)\n", failures ? "FAIL" : "PASS", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
