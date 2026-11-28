// tests/kernels/cpu/iq_avx512_test.cpp - the AVX-512 i-quant rows against their frozen output hashes and a plain
// double-precision reference.
//
//     build/iq_avx512_test [--print] [--bench]
//
// Gate/up rows (`iq512_gu_rows`) and plain dot rows (`iq512_rows`) of IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S and IQ2_S at the
// model's row geometry, 1 to 8 tokens, on random weights, on weights pinned to their extreme codes (every grid index and
// sign bit set, and every one clear) and with saturating activations.  Checks: (1) every dot-row word hashes to the value
// the kernels gave when the hash was frozen (the decode may change, the bits may not); (2) a token's rows do not depend on
// how many tokens share the call; (3) the dot rows and the gate/up rows against a double reference (dequantize, then sum in
// double), whose error is reported.  The dot rows are hashed: their integer decode and float reduction are the same on
// every compiler.  The gate/up rows carry the silu, whose expf comes from the C library and does not agree between MSVC
// and glibc, so they are checked against the reference instead of a frozen bit pattern.  The fixture uses only the raw
// generator output, so it is the same on every standard library.
// `--print` prints the hashes, `--bench` one in-cache thread's rate per format and token count.
#include "basalt/models/qwen4exp/geometry.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/iq_avx512.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace cpu = basalt::kernels::cpu;

namespace {

constexpr int H = cpu::H, FF = cpu::FF, MAXT = 8, MODES = 4;

struct Frozen {
    int type;
    uint64_t hash;
};
// FNV-1a over every dot-row word of every mode and token count, in the order main() computes them
constexpr Frozen kFrozen[] = {
    {16, 0x5863fdcf90fa4725ull}, {17, 0x3acf759ebce5f2a1ull}, {18, 0x5861282d89bf5a41ull},
    {21, 0x931a4e9ae9822c6aull}, {22, 0xacaaeba7bf448c1eull},
};

// mode 0 random bytes, 1 every code bit set, 2 random, 3 every code bit clear; a valid small fp16 scale at the start of
// each block (IQ2_XS / IQ3_S / IQ2_S scale nibbles stay in the code bytes)
void fill(std::vector<uint8_t>& v, int type, uint64_t seed, int mode) {
    std::mt19937_64 rng(seed);
    for (size_t i = 0; i + 8 <= v.size(); i += 8) {
        const uint64_t x = mode == 1 ? ~0ull : mode == 3 ? 0ull : rng();
        std::memcpy(v.data() + i, &x, 8);
    }
    const size_t bs = ggml_type_size((ggml_type) type);
    for (size_t b = 0; b + bs <= v.size(); b += bs) {
        const uint16_t d = (uint16_t) (0x2800 + (rng() & 0xff));
        std::memcpy(&v[b], &d, 2);
    }
}

// block_q8_K: float d, int8 qs[256], int16 bsums[16]
void dequant_q8_K(const uint8_t* p, float* y, int n) {
    for (int b = 0; b < n / 256; ++b, p += 292) {
        float d;
        std::memcpy(&d, p, 4);
        for (int i = 0; i < 256; ++i) *y++ = d * (float) (int8_t) p[4 + i];
    }
}

void fnv(uint64_t& h, const float* p, size_t n) {
    const auto* b = (const unsigned char*) p;
    for (size_t i = 0; i < 4 * n; ++i) h = (h ^ b[i]) * 1099511628211ull;
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char** argv) {
    bool print = false, bench = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        print |= a == "--print";
        bench |= a == "--bench";
    }
    if (!cpu::cpu_avx512_ok()) {
        std::printf("iq_avx512_test SKIPPED: this CPU lacks the AVX-512 set the kernels need\n");
        return 0;
    }
    std::printf("iq_avx512_test: %s\n", cpu::cpu_name().c_str());
    int bad = 0;
    for (const Frozen& fz : kFrozen) {
        const int type = fz.type;
        cpu::NativeFmt f;
        std::string err;
        if (!cpu::native_fmt(type, 20, H, FF, f, err)) { std::printf("native_fmt(%d): %s\n", type, err.c_str()); return 1; }
        const ggml_type_traits* tr = ggml_get_type_traits((ggml_type) type);
        uint64_t h = 1469598103934665603ull;
        double max_rel = 0, max_gu = 0;
        size_t tok_diff = 0;
        std::mt19937 rng(11 + (unsigned) type);
        for (int mode = 0; mode < MODES; ++mode) {
            std::vector<uint8_t> blob(2 * f.up_off);
            fill(blob, type, 100 + (uint64_t) type * 7 + (uint64_t) mode, mode);
            std::vector<std::vector<uint8_t>> act(MAXT, std::vector<uint8_t>(cpu::kNativeActBytes));
            std::vector<std::vector<float>> actf(MAXT, std::vector<float>(H));
            std::vector<float> x(H);
            for (int t = 0; t < MAXT; ++t) {
                for (auto& v : x) v = mode == 2 ? ((rng() & 1) ? 1.f : -1.f) : (float) ((int) (rng() % 2001) - 1000) / 400.f;
                cpu::native_quant_act(f, x.data(), act[(size_t) t].data());
                dequant_q8_K(act[(size_t) t].data(), actf[(size_t) t].data(), H);
            }
            // each token alone: the rows every wider call must reproduce
            std::vector<std::vector<float>> one_ff(MAXT, std::vector<float>(FF)), one_d(MAXT, std::vector<float>(FF));
            for (int t = 0; t < MAXT; ++t) {
                const void* a1[1] = {act[(size_t) t].data()};
                float* f1[1] = {one_ff[(size_t) t].data()};
                float* d1[1] = {one_d[(size_t) t].data()};
                cpu::iq512_gu_rows(type, blob.data(), f.gu_row, f.up_off, H, a1, 1, f1, 0, FF);
                cpu::iq512_rows(type, blob.data(), f.gu_row, H, a1, 1, d1, 0, FF);
            }
            for (int nt = 1; nt <= MAXT; ++nt) {
                const void* ap[MAXT];
                std::vector<std::vector<float>> ff(nt, std::vector<float>(FF)), dd(nt, std::vector<float>(FF));
                float *fp[MAXT], *dp[MAXT];
                for (int t = 0; t < nt; ++t) {
                    ap[t] = act[(size_t) t].data();
                    fp[t] = ff[(size_t) t].data();
                    dp[t] = dd[(size_t) t].data();
                }
                cpu::iq512_gu_rows(type, blob.data(), f.gu_row, f.up_off, H, ap, nt, fp, 0, FF);
                cpu::iq512_rows(type, blob.data(), f.gu_row, H, ap, nt, dp, 0, FF);
                for (int t = 0; t < nt; ++t) {
                    fnv(h, dp[t], FF);
                    for (int r = 0; r < FF; ++r)
                        tok_diff += std::memcmp(&fp[t][r], &one_ff[(size_t) t][(size_t) r], 4) != 0 ||
                                    std::memcmp(&dp[t][r], &one_d[(size_t) t][(size_t) r], 4) != 0;
                }
                std::vector<float> wrow(H), wup(H);
                for (int r : {0, 1, 97, FF - 1}) {
                    tr->to_float(blob.data() + (size_t) r * f.gu_row, wrow.data(), H);
                    tr->to_float(blob.data() + f.up_off + (size_t) r * f.gu_row, wup.data(), H);
                    for (int t = 0; t < nt; ++t) {
                        double want = 0, up = 0;
                        for (int i = 0; i < H; ++i) {
                            const double a = (double) actf[(size_t) t][(size_t) i];
                            want += (double) wrow[(size_t) i] * a;
                            up += (double) wup[(size_t) i] * a;
                        }
                        max_rel = std::max(max_rel, std::fabs(dp[t][r] - want) / std::max(1e-3, std::fabs(want)));
                        const double gu = (want / (1.0 + std::exp(-want))) * up;   // silu(gate_r . a) * (up_r . a)
                        max_gu = std::max(max_gu, std::fabs(fp[t][r] - gu) / std::max(1e-3, std::fabs(gu)));
                    }
                }
            }
        }
        const bool hash_ok = h == fz.hash;
        std::printf("  type %2d: hash %016llx %s, token-count dependence %zu words, dot rows vs double: max rel %.3g, "
                    "gate/up rows vs double: max rel %.3g\n",
                    type, (unsigned long long) h, hash_ok ? "(frozen)" : "(DIFFERS from the frozen hash)", tok_diff, max_rel,
                    max_gu);
        if (print) std::printf("    {%d, 0x%016llxull},\n", type, (unsigned long long) h);
        bad += !hash_ok;
        bad += tok_diff != 0;
        if (max_rel > 1e-3) { std::printf("REFERENCE ERROR too large for type %d (dot rows)\n", type); ++bad; }
        if (max_gu > 1e-4) { std::printf("REFERENCE ERROR too large for type %d (gate/up rows)\n", type); ++bad; }
    }
    // Q8_K hidden activation quantization (`native_quant_h`) and exact AVX-512 multi-token IQ down rows
    // (`iq_exact_512_rows` / `iq1s_512_rows` / `native_down_rows`) bit for bit against ggml's single-token dot.
    {
        constexpr int kRows = 256, kCols = 2304;
        std::mt19937 rng(701);
        std::normal_distribution<float> nd(0.f, 1.f);
        const auto* qa = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
        std::vector<float> h(kCols);
        std::vector<uint8_t> hgot(cpu::kNativeHBytes), href(cpu::kNativeHBytes);
        cpu::NativeFmt qf{};
        std::string qerr;
        if (!cpu::native_fmt(16, 16, kRows, kCols, qf, qerr)) {
            std::printf("native_fmt(16, 16): %s\n", qerr.c_str());
            return 1;
        }
        size_t qdiff = 0;
        for (int trial = 0; trial < 16; ++trial) {
            for (auto& v : h) v = trial == 0 ? 0.f : nd(rng);
            cpu::native_quant_h(qf, h.data(), hgot.data());
            qa->from_float(h.data(), href.data(), kCols);
            qdiff += std::memcmp(hgot.data(), href.data(), qf.h_bytes) != 0;
        }
        if (qdiff) { std::printf("FAIL: native_quant_h Q8_K differs from ggml ref (%zu trials)\n", qdiff); ++bad; }

        for (int type : {16, 17, 18, 19, 21, 22, 29}) {
            cpu::NativeFmt f{};
            std::string err;
            if (!cpu::native_fmt(type, type, kRows, kCols, f, err)) {
                std::printf("native_fmt(%d, %d): %s\n", type, type, err.c_str());
                return 1;
            }
            std::vector<uint8_t> blob(f.bytes);
            fill(blob, type, 900 + (uint64_t) type, 0);
            if (type == GGML_TYPE_IQ1_M) {
                const size_t bs = ggml_type_size(GGML_TYPE_IQ1_M);
                for (size_t b = 0; b + bs <= blob.size(); b += bs) {
                    // IQ1_M packs its fp16 scale into the top nibbles of the four u16 words in scales[8] at +48.
                    blob[b + 55] = (uint8_t) ((blob[b + 55] & 0x0f) | 0x20);
                }
            }
            std::vector<std::vector<uint8_t>> hq(MAXT, std::vector<uint8_t>(cpu::kNativeHBytes));
            const void* hp[MAXT];
            for (int t = 0; t < MAXT; ++t) {
                for (auto& v : h) v = nd(rng);
                cpu::native_quant_h(f, h.data(), hq[(size_t) t].data());
                hp[t] = hq[(size_t) t].data();
            }
            const auto* tc = ggml_get_type_traits_cpu((ggml_type) type);
            std::vector<std::vector<float>> ref(MAXT, std::vector<float>(kRows));
            for (int t = 0; t < MAXT; ++t)
                for (int r = 0; r < kRows; ++r)
                    tc->vec_dot(kCols, &ref[(size_t) t][(size_t) r], 0,
                                blob.data() + f.down_off + (size_t) r * f.d_row, 0, hp[t], 0, 1);

            size_t exact_diff = 0;
            constexpr float kGuard = -777.f;
            constexpr int r0 = 7, r1 = kRows - 11;
            for (int nt = 1; nt <= MAXT; ++nt) {
                std::vector<std::vector<float>> out(nt, std::vector<float>(kRows, kGuard));
                float* op[MAXT];
                for (int t = 0; t < nt; ++t) op[t] = out[(size_t) t].data();
                cpu::native_down_rows(f, blob.data(), hp, nt, op, r0, r1);
                for (int t = 0; t < nt; ++t)
                    for (int r = 0; r < kRows; ++r) {
                        const float want = (r >= r0 && r < r1) ? ref[(size_t) t][(size_t) r] : kGuard;
                        exact_diff += std::memcmp(&out[(size_t) t][(size_t) r], &want, 4) != 0;
                    }
            }
            if (exact_diff) {
                std::printf("FAIL: type %d exact AVX-512 down rows differ from ggml (%zu values)\n", type, exact_diff);
                ++bad;
            }
            // Also verify clamped-SwiGLU exact gate/up rows across all 7 types.
            cpu::NativeFmt fg = f;
            fg.swiglu_limit = 7.f;
            std::vector<float> xv(kRows);
            std::vector<std::vector<uint8_t>> xq(MAXT, std::vector<uint8_t>(cpu::kNativeActBytes));
            const void* xp[MAXT];
            for (int t = 0; t < MAXT; ++t) {
                for (auto& v : xv) v = nd(rng);
                cpu::native_quant_act(fg, xv.data(), xq[(size_t) t].data());
                xp[t] = xq[(size_t) t].data();
            }
            constexpr int gr0 = 5, gr1 = 61;
            std::vector<std::vector<float>> gref(MAXT, std::vector<float>(kCols));
            for (int t = 0; t < MAXT; ++t)
                for (int r = gr0; r < gr1; ++r) {
                    float g = 0.f, u = 0.f;
                    tc->vec_dot(kRows, &g, 0, blob.data() + (size_t) r * fg.gu_row, 0, xp[t], 0, 1);
                    tc->vec_dot(kRows, &u, 0, blob.data() + fg.up_off + (size_t) r * fg.gu_row, 0, xp[t], 0, 1);
                    g = std::fmin(g, fg.swiglu_limit);
                    u = std::fmax(std::fmin(u, fg.swiglu_limit), -fg.swiglu_limit);
                    gref[(size_t) t][(size_t) r] = (g / (1.f + std::exp(-g))) * u;
                }
            size_t gu_diff = 0;
            for (int nt = 1; nt <= MAXT; ++nt) {
                std::vector<std::vector<float>> ffo(nt, std::vector<float>(kCols, kGuard));
                float* fp[MAXT];
                for (int t = 0; t < nt; ++t) fp[t] = ffo[(size_t) t].data();
                cpu::native_gu_rows(fg, blob.data(), xp, nt, fp, gr0, gr1);
                for (int t = 0; t < nt; ++t)
                    for (int r = 0; r < gr1 + 4; ++r) {
                        const float want = (r >= gr0 && r < gr1) ? gref[(size_t) t][(size_t) r] : kGuard;
                        gu_diff += std::memcmp(&ffo[(size_t) t][(size_t) r], &want, 4) != 0;
                    }
            }
            if (gu_diff) {
                std::printf("FAIL: type %d exact AVX-512 limited gate/up rows differ from ggml (%zu values)\n", type,
                            gu_diff);
                ++bad;
            }
        }
        std::printf("  exact AVX-512 IQ gu/down rows & Q8_K quant_h (cols=%d): %s\n", kCols, bad ? "FAIL" : "bit-exact ok");
    }
    // ---- multi-token bounds (nt <= 0 no-op, nt = 9, 13 chunking bitwise vs nt = 1) and Q8_K zero/NaN edge cases
    {
        constexpr int kBigT = 13, kRows = 16;
        std::mt19937 rng(913);
        for (const Frozen& fz : kFrozen) {
            const int type = fz.type;
            cpu::NativeFmt f;
            std::string err;
            if (!cpu::native_fmt(type, 20, H, FF, f, err)) return 1;
            std::vector<uint8_t> blob(2 * f.up_off);
            fill(blob, type, 500 + (uint64_t) type, 0);
            for (int nt : {0, -1}) {
                cpu::iq512_gu_rows(type, blob.data(), f.gu_row, f.up_off, H, nullptr, nt, nullptr, 0, kRows);
                cpu::iq512_rows(type, blob.data(), f.gu_row, H, nullptr, nt, nullptr, 0, kRows);
            }
            std::vector<std::vector<uint8_t>> act(kBigT, std::vector<uint8_t>(cpu::kNativeActBytes));
            std::vector<float> x(H);
            const void* ap[kBigT];
            for (int t = 0; t < kBigT; ++t) {
                for (float& v : x) v = (float) ((int) (rng() % 2001) - 1000) / 400.f;
                cpu::native_quant_act(f, x.data(), act[(size_t) t].data());
                ap[t] = act[(size_t) t].data();
            }
            std::vector<std::vector<float>> ref_gu(kBigT, std::vector<float>(kRows)), ref_d(kBigT, std::vector<float>(kRows));
            for (int t = 0; t < kBigT; ++t) {
                float* fg[1] = {ref_gu[(size_t) t].data()};
                float* fd[1] = {ref_d[(size_t) t].data()};
                cpu::iq512_gu_rows(type, blob.data(), f.gu_row, f.up_off, H, ap + t, 1, fg, 0, kRows);
                cpu::iq512_rows(type, blob.data(), f.gu_row, H, ap + t, 1, fd, 0, kRows);
            }
            for (int nt : {9, 13}) {
                std::vector<std::vector<float>> g(nt, std::vector<float>(kRows, -123.f)), d(nt, std::vector<float>(kRows, -123.f));
                float *gp[kBigT], *dp[kBigT];
                for (int t = 0; t < nt; ++t) { gp[t] = g[(size_t) t].data(); dp[t] = d[(size_t) t].data(); }
                cpu::iq512_gu_rows(type, blob.data(), f.gu_row, f.up_off, H, ap, nt, gp, 0, kRows);
                cpu::iq512_rows(type, blob.data(), f.gu_row, H, ap, nt, dp, 0, kRows);
                for (int t = 0; t < nt; ++t) {
                    if (std::memcmp(gp[t], ref_gu[(size_t) t].data(), kRows * 4) != 0 ||
                        std::memcmp(dp[t], ref_d[(size_t) t].data(), kRows * 4) != 0) {
                        std::printf("MISMATCH avx512 type %d nt %d token %d\n", type, nt, t);
                        ++bad;
                    }
                }
            }
        }
        uint8_t blk[292];
        float zeros[256] = {};
        std::memset(blk, 0xA5, sizeof blk);
        cpu::quantize_q8_K_avx512(zeros, blk, 256);
        uint8_t zero_blk[292] = {};
        if (std::memcmp(blk, zero_blk, sizeof blk) != 0) {
            std::printf("Q8_K avx512 all-zero block left non-zero bytes\n");
            ++bad;
        }
        for (int lane : {0, 1, 15, 16, 127, 128, 255}) {
            for (float fill_v : {0.0f, 1.0f}) {
                float x[256];
                for (float& v : x) v = fill_v;
                x[lane] = std::numeric_limits<float>::quiet_NaN();
                std::memset(blk, 0xA5, sizeof blk);
                cpu::quantize_q8_K_avx512(x, blk, 256);
                float d = 0.0f;
                std::memcpy(&d, blk, 4);
                if (!std::isnan(d)) {
                    std::printf("Q8_K avx512 NaN lane %d (fill %g) produced d=%g\n", lane, fill_v, d);
                    ++bad;
                }
            }
        }
    }
    std::printf("iq_avx512_test: %s\n", bad ? "FAIL" : "PASS");

    if (bench) {   // 200 gate/up rows (the pool's task) per call, the weights cache-resident: one thread's decode rate
        std::mt19937 rng(5);
        for (const Frozen& fz : kFrozen) {
            const int type = fz.type;
            cpu::NativeFmt f;
            std::string err;
            cpu::native_fmt(type, 20, H, FF, f, err);
            std::vector<uint8_t> blob(2 * f.up_off);
            fill(blob, type, 5, 0);
            std::vector<uint8_t> act(cpu::kNativeActBytes);
            std::vector<float> x(H);
            for (auto& v : x) v = (float) ((int) (rng() % 2001) - 1000) / 400.f;
            cpu::native_quant_act(f, x.data(), act.data());
            for (int nt : {1, 2, 4}) {
                const void* ap[MAXT];
                std::vector<std::vector<float>> ff(nt, std::vector<float>(FF));
                float* fp[MAXT];
                for (int t = 0; t < nt; ++t) { ap[t] = act.data(); fp[t] = ff[(size_t) t].data(); }
                const int reps = 400;
                double best = 1e30;
                for (int k = 0; k < 5; ++k) {
                    const double t0 = now_ms();
                    for (int r = 0; r < reps; ++r) cpu::iq512_gu_rows(type, blob.data(), f.gu_row, f.up_off, H, ap, nt, fp, 0, 200);
                    best = std::min(best, now_ms() - t0);
                }
                std::printf("  bench type %2d nt %d: %.2f us per 200 rows, %.2f GB/s (in cache, one thread)\n", type, nt,
                            best * 1e3 / reps, (double) reps * 400.0 * (double) f.gu_row / (best * 1e6));
            }
        }
        // Hidden Q8_K quantization and exact IQ down rows at DeepSeek-V4.1 expert geometry (5120 x 2304).
        constexpr int kDownRows = 5120, kDownCols = 2304;
        std::vector<float> h(kDownCols);
        for (auto& v : h) v = (float) ((int) (rng() % 2001) - 1000) / 400.f;
        std::vector<uint8_t> hq_buf(cpu::kNativeHBytes);
        const auto* qa = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
        cpu::NativeFmt qf{};
        std::string qerr;
        cpu::native_fmt(16, 16, kDownRows, kDownCols, qf, qerr);
        auto bench_q = [&](auto&& fn) {
            double best = 1e30;
            constexpr int kReps = 5000;
            for (int k = 0; k < 5; ++k) {
                const double t0 = now_ms();
                for (int r = 0; r < kReps; ++r) fn();
                best = std::min(best, now_ms() - t0);
            }
            return best * 1e3 / kReps;
        };
        const double q_ref = bench_q([&] { qa->from_float(h.data(), hq_buf.data(), kDownCols); });
        const double q_vec = bench_q([&] { cpu::native_quant_h(qf, h.data(), hq_buf.data()); });
        std::printf("  bench quant_h Q8_K (cols=%d): ggml %.2f us, native %.2f us (%.2fx)\n", kDownCols, q_ref, q_vec,
                    q_ref / q_vec);
        for (int type : {16, 17, 18, 19, 21, 22, 29}) {
            cpu::NativeFmt f{};
            std::string err;
            cpu::native_fmt(type, type, kDownRows, kDownCols, f, err);
            std::vector<uint8_t> blob(f.bytes);
            fill(blob, type, 77 + (uint64_t) type, 0);
            if (type == GGML_TYPE_IQ1_M) {
                const size_t bs = ggml_type_size(GGML_TYPE_IQ1_M);
                for (size_t b = f.down_off; b + bs <= blob.size(); b += bs)
                    blob[b + 55] = (uint8_t) ((blob[b + 55] & 0x0f) | 0x20);
            }
            cpu::native_quant_h(f, h.data(), hq_buf.data());
            const auto* tc = ggml_get_type_traits_cpu((ggml_type) type);
            for (int nt : {1, 4, 6}) {
                const void* hp[MAXT];
                std::vector<std::vector<float>> out(nt, std::vector<float>(kDownRows));
                float* op[MAXT];
                for (int t = 0; t < nt; ++t) { hp[t] = hq_buf.data(); op[t] = out[(size_t) t].data(); }
                constexpr int kReps = 5;
                auto bench_d = [&](auto&& fn) {
                    double best = 1e30;
                    for (int k = 0; k < 3; ++k) {
                        const double t0 = now_ms();
                        for (int r = 0; r < kReps; ++r) fn();
                        best = std::min(best, now_ms() - t0);
                    }
                    return best * 1e3 / kReps;
                };
                const double t_ref = bench_d([&] {
                    for (int r = 0; r < kDownRows; ++r) {
                        const uint8_t* dr = blob.data() + f.down_off + (size_t) r * f.d_row;
                        for (int t = 0; t < nt; ++t) tc->vec_dot(kDownCols, &op[t][r], 0, dr, 0, hp[t], 0, 1);
                    }
                });
                const double t_nat = bench_d([&] { cpu::native_down_rows(f, blob.data(), hp, nt, op, 0, kDownRows); });
                const double bytes = (double) kDownRows * (double) f.d_row;
                std::printf("  bench down type %2d (%dx%d) nt=%d: ggml %7.1f us, native %7.1f us (%.2fx, %.2f GB/s)\n",
                            type, kDownRows, kDownCols, nt, t_ref, t_nat, t_ref / t_nat, bytes / (t_nat * 1e3));
            }
        }
    }
    return bad == 0 ? 0 : 1;
}
