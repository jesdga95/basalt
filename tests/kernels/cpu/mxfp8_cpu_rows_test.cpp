// tests/kernels/cpu/mxfp8_cpu_rows_test.cpp - the MXFP8 (type 46) CPU dot on each instruction set, bit for bit against
// the scalar row, on random blocks that include subnormals, the largest code, zero, both signs and E4M3's NaN code;
// and against a double reference.  `--bench [n_embd] [n_ff]` times one expert's three matrices per row set against
// the Q8_0 row.
#define GGML_COMMON_DECL_CPP
#include "basalt/ggml/ggml_common.h"
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/cpu/swiglu.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>


namespace cpu = basalt::kernels::cpu;

namespace {

struct Isa {
    const char* name;
    const ggml_type_traits_cpu* (*traits)(ggml_type);
    void (*scalar)(int, float*, const void*, const void*);
    void (*prep)(int, const void*, float*);
    void (*rows)(int, const void*, size_t, const float* const*, int, float* const*, int, int);
    bool ok;
};

// E4M3 codes worth hitting on purpose: zero, the smallest subnormal, the largest subnormal, the smallest normal, one,
// the largest, the NaN code, each with both signs
const uint8_t kSpecial[] = {0x00, 0x01, 0x07, 0x08, 0x38, 0x7E, 0x7F, 0x80, 0x81, 0x87, 0x88, 0xB8, 0xFE, 0xFF};

void fill(std::mt19937& rng, std::vector<block_mxfp8>& w, std::vector<block_q8_0>& a, int mode) {
    for (auto& b : w) {
        for (auto& q : b.qs) {
            const uint32_t r = rng();
            q = mode == 0 ? (uint8_t) r : (r % 4 == 0 ? kSpecial[(r >> 8) % sizeof(kSpecial)] : (uint8_t) (r >> 8));
        }
        for (auto& d : b.d) {
            const uint32_t r = rng();
            // mode 0: scales around 1; mode 1: the whole UE8M0 range with 0 and 255 over-represented
            d = mode == 0 ? (uint8_t) (110 + r % 40) : (r % 8 == 0 ? (uint8_t) (r % 2 ? 0xFF : 0x00) : (uint8_t) (r >> 8));
        }
    }
    for (auto& b : a) {
        for (auto& q : b.qs) {
            const uint32_t r = rng();
            q = r % 16 == 0 ? (int8_t) ((r >> 8) % 2 ? 127 : -128) : (int8_t) (r >> 8);
        }
        const float d = 0.001f + (rng() % 20000) * 1e-4f;
        b.d = ggml_fp32_to_fp16(d);
    }
}

bool same(float x, float y) {
    if (std::isnan(x) && std::isnan(y)) return true;
    return std::memcmp(&x, &y, 4) == 0;
}

double ref_dot(int n, const block_mxfp8* w, const block_q8_0* a, double* mag) {
    double s = 0;
    *mag = 0;
    for (int ib = 0; ib < n / 128; ++ib)
        for (int sb = 0; sb < 4; ++sb) {
            const auto& yb = a[4 * ib + sb];
            double d = 0, dm = 0;
            for (int j = 0; j < 32; ++j) {
                const uint8_t c = w[ib].qs[sb * 32 + j];
                const int e = (c >> 3) & 15, m = c & 7;
                double v = (e == 15 && m == 7) ? 0.0 : e == 0 ? std::ldexp((double) m, -9) : std::ldexp(1.0 + m / 8.0, e - 7);
                d += (c & 0x80 ? -v : v) * yb.qs[j];
                dm += v * std::abs((int) yb.qs[j]);
            }
            const uint8_t u = w[ib].d[sb];
            const double sc = (double) ggml_fp16_to_fp32(yb.d) * (u == 0xFF ? 0.0 : std::ldexp(1.0, (u == 0 ? -127 : u - 127)));
            s += sc * d;
            *mag += sc * dm;
        }
    return s;
}

int check(const Isa& isa, int n, int mode, int rows) {
    const auto dot = isa.traits(GGML_TYPE_MXFP8)->vec_dot;
    std::mt19937 rng(1234 + n + mode);
    const int nb = n / 128;
    int bad = 0, worst_n = 0;
    double worst = 0;
    for (int r = 0; r < rows; ++r) {
        std::vector<block_mxfp8> w(nb);
        std::vector<block_q8_0> a(nb * 4);
        fill(rng, w, a, mode);
        float s = 0, t = 0;
        isa.scalar(n, &s, w.data(), a.data());
        dot(n, &t, 0, w.data(), 0, a.data(), 0, 1);
        // the multi-token rows kernel, every token count, each token's result the scalar row's bits
        for (int nt = 1; nt <= 8; ++nt) {
            std::vector<std::vector<block_q8_0>> acts(nt, std::vector<block_q8_0>(nb * 4));
            std::vector<std::vector<float>> prep(nt, std::vector<float>(nb * 144));
            std::vector<float> outv(nt);
            const float* pp[8];
            float* op[8];
            for (int k = 0; k < nt; ++k) {
                if (k == 0) acts[0] = a;
                else { std::vector<block_mxfp8> junk(nb); fill(rng, junk, acts[k], mode); }
                isa.prep(n, acts[k].data(), prep[k].data());
                pp[k] = prep[k].data();
                op[k] = &outv[k];
            }
            isa.rows(n, w.data(), 0, pp, nt, op, 0, 1);
            for (int k = 0; k < nt; ++k) {
                float sk = 0;
                isa.scalar(n, &sk, w.data(), acts[k].data());
                if (!same(sk, outv[k]) && bad++ < 5)
                    std::printf("  %s n=%d mode=%d nt=%d token %d: scalar %a, rows %a\n", isa.name, n, mode, nt, k, sk, outv[k]);
            }
        }
        if (!same(s, t)) {
            if (bad++ < 5) std::printf("  %s n=%d mode=%d row %d: scalar %a, vector %a\n", isa.name, n, mode, r, s, t);
        }
        if (mode == 0) {
            double mag = 0;   // the sum of the absolute terms bounds the fp32 sum's error
            const double e = ref_dot(n, w.data(), a.data(), &mag);
            const double err = std::fabs((double) t - e) / (mag + 1e-30);
            if (err > worst) { worst = err; worst_n = r; }
        }
    }
    std::printf("%-7s n=%-5d mode=%d: %d rows, %d not bit-identical\n", isa.name, n, mode, rows, bad);
    if (mode == 0 && worst > 1e-5) {
        std::printf("  error vs double, relative to the absolute terms, %.3g (row %d)\n", worst, worst_n);
        ++bad;
    }
    return bad;
}

double ns_per(int reps, void (*f)(void*), void* ctx) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) f(ctx);
    return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / reps;
}

struct BenchCtx {
    ggml_vec_dot_t dot;
    int n, rows;
    const uint8_t* w;
    size_t stride;
    const void* a;
    float* out;
};
void run_rows(void* p) {
    auto* c = (BenchCtx*) p;
    for (int r = 0; r < c->rows; ++r) c->dot(c->n, c->out + r, 0, c->w + (size_t) r * c->stride, 0, c->a, 0, 1);
}

int bench(const Isa& isa, int n_embd, int n_ff) {
    // one expert: gate and up [n_ff][n_embd], down [n_embd][n_ff]; cache-resident rows (a slice repeated) and the
    // 1 MiB-spaced working set of many experts is the benchmark of the pool, not of the dot
    const int rows = 256;
    std::mt19937 rng(7);
    const ggml_type types[2] = {GGML_TYPE_MXFP8, GGML_TYPE_Q8_0};
    for (const ggml_type ty : types) {
        const auto* tr = isa.traits(ty);
        std::vector<uint8_t> act_q(ggml_row_size(GGML_TYPE_Q8_0, std::max(n_embd, n_ff)));
        for (auto& b : act_q) b = (uint8_t) rng();
        // sane Q8_0 scales
        for (size_t i = 0; i + 34 <= act_q.size(); i += 34) {
            const ggml_fp16_t d = ggml_fp32_to_fp16(0.01f);
            std::memcpy(&act_q[i], &d, 2);
        }
        double total = 0, bytes = 0;
        for (int m = 0; m < 3; ++m) {
            const int n = m < 2 ? n_embd : n_ff, out_rows = m < 2 ? n_ff : n_embd;
            const size_t stride = ggml_row_size(ty, n);
            const int use = std::min(rows, out_rows);
            std::vector<uint8_t> w(stride * use);
            for (auto& b : w) b = (uint8_t) rng();
            if (ty == GGML_TYPE_MXFP8)
                for (size_t i = 0; i < w.size(); i += 132) for (int k = 0; k < 4; ++k) w[i + k] = (uint8_t) (115 + rng() % 10);
            if (ty == GGML_TYPE_Q8_0)
                for (size_t i = 0; i + 34 <= w.size(); i += 34) { const ggml_fp16_t d = ggml_fp32_to_fp16(0.01f); std::memcpy(&w[i], &d, 2); }
            std::vector<float> out(use);
            BenchCtx c{tr->vec_dot, n, use, w.data(), stride, act_q.data(), out.data()};
            ns_per(20, run_rows, &c);
            double best = 1e30;
            for (int k = 0; k < 5; ++k) best = std::min(best, ns_per(200, run_rows, &c));
            // scale to the matrix's rows
            total += best * out_rows / use;
            bytes += (double) stride * out_rows;
        }
        std::printf("%-7s %-6s one expert (%dx%d, 3 matrices): %8.1f us, %5.1f GB/s of weights\n", isa.name, ggml_type_name(ty),
                    n_embd, n_ff, total / 1e3, bytes / total);
        if (ty != GGML_TYPE_MXFP8) continue;
        {   // the table-lookup scalar row the vector rows equal
            total = 0;
            for (int m = 0; m < 3; ++m) {
                const int n = m < 2 ? n_embd : n_ff, out_rows = m < 2 ? n_ff : n_embd;
                const size_t stride = ggml_row_size(ty, n);
                const int use = std::min(32, out_rows);   // the scalar row is slow: a smaller slice
                std::vector<uint8_t> w(stride * use);
                for (auto& b : w) b = (uint8_t) rng();
                for (size_t i = 0; i < w.size(); i += 132) for (int k = 0; k < 4; ++k) w[i + k] = (uint8_t) (115 + rng() % 10);
                std::vector<float> out(use);
                auto once = [&] { for (int r = 0; r < use; ++r) isa.scalar(n, &out[r], w.data() + (size_t) r * stride, act_q.data()); };
                for (int k = 0; k < 3; ++k) once();
                double best = 1e30;
                for (int k = 0; k < 5; ++k) {
                    const auto t0 = std::chrono::steady_clock::now();
                    for (int i = 0; i < 20; ++i) once();
                    best = std::min(best, std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / 20);
                }
                total += best * out_rows / use;
            }
            std::printf("%-7s mxfp8 scalar row, one expert: %8.1f us\n", isa.name, total / 1e3);
        }
        for (const int nt : {1, 2, 4, 8}) {
            total = 0;
            for (int m = 0; m < 3; ++m) {
                const int n = m < 2 ? n_embd : n_ff, out_rows = m < 2 ? n_ff : n_embd;
                const size_t stride = ggml_row_size(ty, n);
                const int use = std::min(rows, out_rows);
                std::vector<uint8_t> w(stride * use);
                for (auto& b : w) b = (uint8_t) rng();
                for (size_t i = 0; i < w.size(); i += 132) for (int k = 0; k < 4; ++k) w[i + k] = (uint8_t) (115 + rng() % 10);
                std::vector<std::vector<float>> prep(nt, std::vector<float>(n / 128 * 144)), out(nt, std::vector<float>(use));
                const float* pp[8];
                float* op[8];
                for (int k = 0; k < nt; ++k) { isa.prep(n, act_q.data(), prep[k].data()); pp[k] = prep[k].data(); op[k] = out[k].data(); }
                auto once = [&] { isa.rows(n, w.data(), stride, pp, nt, op, 0, use); };
                for (int k = 0; k < 20; ++k) once();
                double best = 1e30;
                for (int k = 0; k < 5; ++k) {
                    const auto t0 = std::chrono::steady_clock::now();
                    for (int i = 0; i < 200; ++i) once();
                    best = std::min(best, std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / 200);
                }
                total += best * out_rows / use;
            }
            std::printf("%-7s mxfp8 rows kernel, %d token%s: %8.1f us per expert, %6.1f us per token\n", isa.name, nt, nt > 1 ? "s" : "", total / 1e3, total / 1e3 / nt);
        }
    }
    return 0;
}


// the native expert's gate/up/SwiGLU and down rows (what the decode's CPU tier calls) against the single-token dot of the
// table plus the layer's SwiGLU, for token counts around the kernel's chunk of 8 and a row range that is not whole
int check_native(int n_embd, int n_ff, float limit) {
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(GGML_TYPE_MXFP8, GGML_TYPE_MXFP8, n_embd, n_ff, f, err, limit)) {
        std::printf("native_fmt: %s\n", err.c_str());
        return 1;
    }
    std::mt19937 rng(99 + n_embd);
    std::vector<uint8_t> blob(f.bytes);
    {
        std::vector<block_mxfp8> w(f.bytes / sizeof(block_mxfp8));
        std::vector<block_q8_0> dummy(1);
        fill(rng, w, dummy, 0);
        std::memcpy(blob.data(), w.data(), blob.size());
    }
    const auto dot = ggml_get_type_traits_cpu(GGML_TYPE_MXFP8)->vec_dot;
    int bad = 0;
    for (const int nt : {1, 2, 3, 5, 8, 11}) {
        std::vector<std::vector<uint8_t>> act(nt, std::vector<uint8_t>(f.act_bytes)), hq(nt, std::vector<uint8_t>(f.h_bytes));
        std::vector<std::vector<float>> ff(nt, std::vector<float>(n_ff)), want(nt, std::vector<float>(n_ff)),
            out(nt, std::vector<float>(n_embd)), wout(nt, std::vector<float>(n_embd));
        const void* ap[16];
        const void* hp[16];
        float* fp[16];
        float* op[16];
        std::normal_distribution<float> nd(0, 1.f);
        for (int k = 0; k < nt; ++k) {
            std::vector<float> x(n_embd), h(n_ff);
            for (auto& v : x) v = nd(rng);
            for (auto& v : h) v = nd(rng);
            cpu::native_quant_act(f, x.data(), act[k].data());
            cpu::native_quant_h(f, h.data(), hq[k].data());
            ap[k] = act[k].data();
            hp[k] = hq[k].data();
            fp[k] = ff[k].data();
            op[k] = out[k].data();
        }
        const int r0 = 3, r1 = n_ff - 5;
        cpu::native_gu_rows(f, blob.data(), ap, nt, fp, r0, r1);
        const cpu::SwiGlu glu{limit, 0.f, 0.f};
        for (int k = 0; k < nt; ++k)
            for (int r = r0; r < r1; ++r) {
                float g = 0, u = 0;
                dot(n_embd, &g, 0, blob.data() + (size_t) r * f.gu_row, 0, ap[k], 0, 1);
                dot(n_embd, &u, 0, blob.data() + f.up_off + (size_t) r * f.up_row, 0, ap[k], 0, 1);
                if (!same(cpu::swiglu(glu, g, u), ff[k][r]) && bad++ < 5) std::printf("  gate/up nt=%d token %d row %d differs\n", nt, k, r);
            }
        cpu::native_down_rows(f, blob.data(), hp, nt, op, 1, n_embd - 2);
        for (int k = 0; k < nt; ++k)
            for (int r = 1; r < n_embd - 2; ++r) {
                float s = 0;
                dot(n_ff, &s, 0, blob.data() + f.down_off + (size_t) r * f.d_row, 0, hp[k], 0, 1);
                if (!same(s, out[k][r]) && bad++ < 5) std::printf("  down nt=%d token %d row %d differs\n", nt, k, r);
            }
    }
    std::printf("native expert %dx%d limit %g: %d differ from the single-token dot\n", n_embd, n_ff, limit, bad);
    return bad;
}

}  // namespace

int main(int argc, char** argv) {
    const Isa isas[2] = {
        {"AVX2", ggml_get_type_traits_cpu_avx2, ggml_vec_dot_mxfp8_q8_0_scalar_avx2, ggml_mxfp8_q8_0_prep_avx2, ggml_mxfp8_q8_0_rows_avx2, cpu::cpu_avx2_ok()},
        {"AVX512", ggml_get_type_traits_cpu_avx512, ggml_vec_dot_mxfp8_q8_0_scalar_avx512, ggml_mxfp8_q8_0_prep_avx512, ggml_mxfp8_q8_0_rows_avx512, cpu::cpu_avx512_ok()},
    };
    if (!isas[0].ok) {
        std::printf("no AVX2 on this host, skipped\n");
        return 77;
    }
    if (argc > 1 && !std::strcmp(argv[1], "--bench")) {
        const int ne = argc > 2 ? std::atoi(argv[2]) : 3072, nf = argc > 3 ? std::atoi(argv[3]) : 1024;
        for (const auto& isa : isas) if (isa.ok) bench(isa, ne, nf);
        return 0;
    }
    int bad = 0;
    for (const auto& isa : isas) {
        if (!isa.ok) {
            std::printf("%s: not on this host, skipped\n", isa.name);
            continue;
        }
        for (const int n : {128, 256, 640, 3072, 4096})
            for (const int mode : {0, 1}) bad += check(isa, n, mode, 400);
    }
    bad += check_native(256, 384, 0.f);
    bad += check_native(640, 256, 7.f);
    std::printf(bad ? "FAIL\n" : "OK\n");
    return bad ? 1 : 0;
}
