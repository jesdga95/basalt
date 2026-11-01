// tests/kernels/cpu/iq_avx512_test.cpp - the AVX-512 i-quant rows against their frozen output hashes and a plain
// double-precision reference.
//
//     build/iq_avx512_test [--print] [--bench]
//
// Gate/up rows (`iq512_gu_rows`) and plain dot rows (`iq512_rows`) of IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S and IQ2_S at the
// model's row geometry, 1 to 8 tokens, on random weights, on weights pinned to their extreme codes (every grid index and
// sign bit set, and every one clear) and with saturating activations.  Checks: (1) every output word hashes to the value
// the kernels gave when the hash was frozen (the decode may change, the bits may not); (2) a token's rows do not depend on
// how many tokens share the call; (3) the dot rows against a double reference (dequantize, then sum in double), whose
// error is reported.  The fixture uses only the raw generator output, so it is the same on every standard library.
// `--print` prints the hashes, `--bench` one in-cache thread's rate per format and token count.
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
// FNV-1a over every output word of every mode, token count and kernel, in the order main() computes them
constexpr Frozen kFrozen[] = {
    {16, 0x9fdbeb21f694fd2dull}, {17, 0xf352818aa1e11197ull}, {18, 0xd6afdf7513d359a7ull},
    {21, 0x0b05f3ee4879334bull}, {22, 0x397461e6abe5db6full},
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
        double max_rel = 0;
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
                    fnv(h, fp[t], FF);
                    fnv(h, dp[t], FF);
                    for (int r = 0; r < FF; ++r)
                        tok_diff += std::memcmp(&fp[t][r], &one_ff[(size_t) t][(size_t) r], 4) != 0 ||
                                    std::memcmp(&dp[t][r], &one_d[(size_t) t][(size_t) r], 4) != 0;
                }
                std::vector<float> wrow(H);
                for (int r : {0, 1, 97, FF - 1}) {
                    tr->to_float(blob.data() + (size_t) r * f.gu_row, wrow.data(), H);
                    for (int t = 0; t < nt; ++t) {
                        double want = 0;
                        for (int i = 0; i < H; ++i) want += (double) wrow[(size_t) i] * (double) actf[(size_t) t][(size_t) i];
                        max_rel = std::max(max_rel, std::fabs(dp[t][r] - want) / std::max(1e-3, std::fabs(want)));
                    }
                }
            }
        }
        const bool hash_ok = h == fz.hash;
        std::printf("  type %2d: hash %016llx %s, token-count dependence %zu words, dot rows vs double: max rel %.3g\n", type,
                    (unsigned long long) h, hash_ok ? "(frozen)" : "(DIFFERS from the frozen hash)", tok_diff, max_rel);
        if (print) std::printf("    {%d, 0x%016llxull},\n", type, (unsigned long long) h);
        bad += !hash_ok;
        bad += tok_diff != 0;
        if (max_rel > 1e-3) { std::printf("REFERENCE ERROR too large for type %d\n", type); ++bad; }
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
    }
    return bad == 0 ? 0 : 1;
}
