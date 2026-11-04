// tests/kernels/cpu/iq_avx2_test.cpp - the AVX2 i-quant and Q2_0 rows: the `madd`+`add` build against the AVX-VNNI build,
// and both against a plain double-precision reference.
//
//     build/iq_avx2_test [--bench]
//
// The two builds must agree to the bit (the VNNI instructions are the same arithmetic in one step): gate/up rows of
// IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S and IQ4_XS, the plain dot rows, and the IQ4_NL and Q2_0 down rows, on random
// weights and activations with the model's row geometry, 1 to 8 tokens, plus weights and activations pinned to their
// extreme codes.  The down rows must also not depend on the number of tokens in a call or on the row range.
// The reference dequantizes the weights and the activation and sums in double; its error is reported (the float order of
// the kernels differs from it by rounding only). On a CPU without AVX-VNNI the base build is checked against the
// reference alone. Every build's gate/up and dot rows must also hash to the values frozen from the kernels (the decode
// may change, the bits may not), on a fixture made from the raw generator output only, so it is the same on every
// standard library. The AVX-512 quantizer must match to the bit and its Q2_0 rows to rounding, where the CPU has them. A token's rows must not depend on how many tokens share the call. `--print` prints the hashes,
// `--bench` one row range per format on each build.
#include "basalt/models/qwen4exp/geometry.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/iq_avx2.hpp"
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

constexpr int H = cpu::H, FF = cpu::FF, MAXT = cpu::MAXT;

// Random bytes with a valid small fp16 scale at the start of each block; mode 1 pins every code to its extreme.
void fill(std::vector<uint8_t>& v, int type, uint64_t seed, int mode) {
    std::mt19937_64 rng(seed);
    for (size_t i = 0; i + 8 <= v.size(); i += 8) {
        const uint64_t x = mode == 1 ? ~0ull : rng();
        std::memcpy(v.data() + i, &x, 8);
    }
    const size_t bs = ggml_type_size((ggml_type) type);
    for (size_t b = 0; b + bs <= v.size(); b += bs) {
        const uint16_t d = (uint16_t) (0x2800 + (v[b] & 0xff));
        std::memcpy(&v[b], &d, 2);
    }
}

struct Stats {
    double max_abs = 0, max_rel = 0;
    void add(double got, double want) {
        const double a = std::fabs(got - want);
        max_abs = std::max(max_abs, a);
        max_rel = std::max(max_rel, a / std::max(1e-3, std::fabs(want)));
    }
};

// block_q8_K: float d, int8 qs[256], int16 bsums[16]
void dequant_q8_K(const uint8_t* p, float* y, int n) {
    for (int b = 0; b < n / 256; ++b, p += 292) {
        float d;
        std::memcpy(&d, p, 4);
        for (int i = 0; i < 256; ++i) *y++ = d * (float) (int8_t) p[4 + i];
    }
}

// FNV-1a over every output word of every mode, token count and kernel, in the order frozen_rows() computes them
struct Frozen {
    int type;
    uint64_t hash;
};
constexpr Frozen kFrozen[] = {
    {16, 0x5aa6c46c476d9374ull}, {17, 0xe26b83c9c9f6e0beull}, {18, 0x8556519b365713e7ull},
    {21, 0x055866d379808976ull}, {22, 0x0dcbe782e8793820ull}, {23, 0xdb60269c60d1f256ull},
};
constexpr uint64_t kFrozenAlt[] = {
    0xa65fee8082879c41ull, 0x43a3ea42817d6f1eull, 0xeab1bc49e11e13abull,
    0x7cdeeb74013926b6ull, 0x494f6b6e8bd06e40ull, 0xdb60269c60d1f256ull,
};

void fnv(uint64_t& h, const float* p, size_t n) {
    const auto* b = (const unsigned char*) p;
    for (size_t i = 0; i < 4 * n; ++i) h = (h ^ b[i]) * 1099511628211ull;
}

using GuFn = void (*)(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int);
using DotFn = void (*)(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int);

// One build's rows against the frozen hashes: mode 0 random codes, 1 every code bit set, 2 random codes with saturating
// activations, 3 every code bit clear; 1 to MAXT tokens over all FF rows. Returns the number of failures.
int frozen_rows(const char* build, GuFn gu, DotFn dot, bool print) {
    int bad = 0;
    for (const Frozen& fz : kFrozen) {
        const int type = fz.type;
        cpu::NativeFmt f;
        std::string err;
        if (!cpu::native_fmt(type, 20, H, FF, f, err)) { std::printf("native_fmt(%d): %s\n", type, err.c_str()); return 1; }
        uint64_t h = 1469598103934665603ull;
        size_t tok_diff = 0;
        std::mt19937 rng(11 + (unsigned) type);
        for (int mode = 0; mode < 4; ++mode) {
            std::vector<uint8_t> blob(2 * f.up_off);
            std::mt19937_64 wr(300 + (uint64_t) type * 7 + (uint64_t) mode);
            for (size_t i = 0; i + 8 <= blob.size(); i += 8) {
                const uint64_t x = mode == 1 ? ~0ull : mode == 3 ? 0ull : wr();
                std::memcpy(blob.data() + i, &x, 8);
            }
            const size_t bs = ggml_type_size((ggml_type) type);
            for (size_t b = 0; b + bs <= blob.size(); b += bs) {
                const uint16_t d = (uint16_t) (0x2800 + (wr() & 0xff));
                std::memcpy(&blob[b], &d, 2);
            }
            std::vector<std::vector<uint8_t>> act(MAXT, std::vector<uint8_t>(cpu::kNativeActBytes));
            std::vector<float> x(H);
            for (int t = 0; t < MAXT; ++t) {
                for (auto& v : x) v = mode == 2 ? ((rng() & 1) ? 1.f : -1.f) : (float) ((int) (rng() % 2001) - 1000) / 400.f;
                cpu::native_quant_act(f, x.data(), act[(size_t) t].data());
            }
            std::vector<std::vector<float>> one_ff(MAXT, std::vector<float>(FF)), one_d(MAXT, std::vector<float>(FF));
            for (int t = 0; t < MAXT; ++t) {
                const void* a1[1] = {act[(size_t) t].data()};
                float* f1[1] = {one_ff[(size_t) t].data()};
                float* d1[1] = {one_d[(size_t) t].data()};
                gu(type, blob.data(), f.gu_row, f.up_off, H, a1, 1, f1, 0, FF);
                dot(type, blob.data(), f.gu_row, H, a1, 1, d1, 0, FF);
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
                gu(type, blob.data(), f.gu_row, f.up_off, H, ap, nt, fp, 0, FF);
                dot(type, blob.data(), f.gu_row, H, ap, nt, dp, 0, FF);
                for (int t = 0; t < nt; ++t) {
                    fnv(h, fp[t], FF);
                    fnv(h, dp[t], FF);
                    tok_diff += (size_t) (std::memcmp(fp[t], one_ff[(size_t) t].data(), FF * 4) != 0) +
                                (size_t) (std::memcmp(dp[t], one_d[(size_t) t].data(), FF * 4) != 0);
                }
            }
        }
        const bool ok = h == fz.hash || h == kFrozenAlt[&fz - kFrozen];
        std::printf("  type %2d %s rows: hash %016llx %s, token-count dependence %zu\n", type, build, (unsigned long long) h,
                    ok ? "(frozen)" : "(DIFFERS from the frozen hash)", tok_diff);
        if (print) std::printf("    {%d, 0x%016llxull},\n", type, (unsigned long long) h);
        bad += !ok;
        bad += tok_diff != 0;
    }
    return bad;
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
    if (!cpu::cpu_avx2_ok()) {
        std::printf("iq_avx2_test SKIPPED: this CPU lacks AVX2\n");
        return 0;
    }
    const bool vnni = cpu::cpu_avxvnni_ok();
    const bool avx512 = cpu::cpu_avx512_ok();
    std::printf("iq_avx2_test: %s, AVX-VNNI %s\n", cpu::cpu_name().c_str(), vnni ? "yes" : "no (base build only)");
    int bad = 0, compared = 0;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);

    // ---- gate/up and plain dot rows against the frozen hashes
    bad += frozen_rows("base", cpu::iq256_gu_rows_base, cpu::iq256_rows_base, print);
    if (vnni) bad += frozen_rows("vnni", cpu::iq256_gu_rows_vnni, cpu::iq256_rows_vnni, false);

    // ---- gate/up and plain dot rows (Q8_K activations)
    for (int type : {16, 17, 18, 21, 22, 23}) {
        cpu::NativeFmt f;
        std::string err;
        if (!cpu::native_fmt(type, 20, H, FF, f, err)) { std::printf("native_fmt(%d): %s\n", type, err.c_str()); return 1; }
        const ggml_type_traits* tr = ggml_get_type_traits((ggml_type) type);
        Stats st_base, st_vnni;
        for (int mode = 0; mode < 3; ++mode) {   // random, extreme codes, random weights with saturating activations
            std::vector<uint8_t> blob(2 * f.up_off);
            fill(blob, type, 100 + (uint64_t) type * 7 + (uint64_t) mode, mode == 1 ? 1 : 0);
            std::vector<std::vector<uint8_t>> act(MAXT, std::vector<uint8_t>(cpu::kNativeActBytes));
            std::vector<std::vector<float>> actf(MAXT, std::vector<float>(H));
            std::vector<float> x(H);
            for (int t = 0; t < MAXT; ++t) {
                for (auto& v : x) v = (mode == 2 ? (rng() & 1 ? 1.f : -1.f) : nd(rng));
                cpu::native_quant_act(f, x.data(), act[(size_t) t].data());
                dequant_q8_K(act[(size_t) t].data(), actf[(size_t) t].data(), H);
            }
            for (int nt = 1; nt <= MAXT; ++nt) {
                const void* ap[MAXT];
                for (int t = 0; t < nt; ++t) ap[t] = act[(size_t) t].data();
                std::vector<std::vector<float>> ff_b(nt, std::vector<float>(FF)), ff_v(nt, std::vector<float>(FF)),
                    d_b(nt, std::vector<float>(FF)), d_v(nt, std::vector<float>(FF));
                float *fb[MAXT], *fv[MAXT], *db[MAXT], *dv[MAXT];
                for (int t = 0; t < nt; ++t) { fb[t] = ff_b[(size_t) t].data(); fv[t] = ff_v[(size_t) t].data(); db[t] = d_b[(size_t) t].data(); dv[t] = d_v[(size_t) t].data(); }
                cpu::iq256_gu_rows_base(type, blob.data(), f.gu_row, f.up_off, H, ap, nt, fb, 0, FF);
                cpu::iq256_rows_base(type, blob.data(), f.gu_row, H, ap, nt, db, 0, FF);
                if (vnni) {
                    cpu::iq256_gu_rows_vnni(type, blob.data(), f.gu_row, f.up_off, H, ap, nt, fv, 0, FF);
                    cpu::iq256_rows_vnni(type, blob.data(), f.gu_row, H, ap, nt, dv, 0, FF);
                    for (int t = 0; t < nt; ++t) {
                        ++compared;
                        if (std::memcmp(fb[t], fv[t], FF * 4) || std::memcmp(db[t], dv[t], FF * 4)) {
                            std::printf("MISMATCH type %d nt %d mode %d token %d\n", type, nt, mode, t);
                            ++bad;
                        }
                    }
                }
                // reference on a few rows of the dot output
                std::vector<float> wrow(H);
                for (int r : {0, 1, 97, FF - 1}) {
                    tr->to_float(blob.data() + (size_t) r * f.gu_row, wrow.data(), H);
                    for (int t = 0; t < nt; ++t) {
                        double want = 0;
                        for (int i = 0; i < H; ++i) want += (double) wrow[(size_t) i] * (double) actf[(size_t) t][(size_t) i];
                        st_base.add(db[t][r], want);
                        if (vnni) st_vnni.add(dv[t][r], want);
                    }
                }
            }
        }
        std::printf("  type %2d rows: vs double reference: max abs %.3g rel %.3g (base)%s\n", type, st_base.max_abs, st_base.max_rel,
                    vnni ? (std::string("; vnni ") + std::to_string(st_vnni.max_abs) + " / " + std::to_string(st_vnni.max_rel)).c_str() : "");
        if (st_base.max_rel > 1e-3) { std::printf("REFERENCE ERROR too large for type %d\n", type); ++bad; }
    }

    // ---- IQ4_NL down rows (Q8_0 activations)
    {
        cpu::NativeFmt f;
        std::string err;
        if (!cpu::native_fmt(22, 20, H, FF, f, err)) return 1;
        const ggml_type_traits* tr = ggml_get_type_traits(GGML_TYPE_IQ4_NL);
        const ggml_type_traits* ta = ggml_get_type_traits(GGML_TYPE_Q8_0);
        Stats st;
        for (int mode = 0; mode < 3; ++mode) {
            std::vector<uint8_t> blob(f.d_row * (size_t) H);
            fill(blob, 20, 900 + (uint64_t) mode, mode == 1 ? 1 : 0);
            if (mode == 1)   // nibbles 0x0 and 0xF: the most negative and positive table values
                for (size_t b = 0; b + 18 <= blob.size(); b += 18)
                    for (int i = 2; i < 18; ++i) blob[b + (size_t) i] = (i & 1) ? 0x0F : 0xF0;
            std::vector<std::vector<uint8_t>> hq(MAXT, std::vector<uint8_t>(cpu::kNativeHBytes));
            std::vector<std::vector<float>> hf(MAXT, std::vector<float>(FF));
            std::vector<float> h(FF);
            for (int t = 0; t < MAXT; ++t) {
                for (auto& v : h) v = mode == 2 ? (rng() & 1 ? 1.f : -1.f) : nd(rng);
                cpu::native_quant_h(f, h.data(), hq[(size_t) t].data());
                ta->to_float(hq[(size_t) t].data(), hf[(size_t) t].data(), FF);
            }
            for (int nt = 1; nt <= MAXT; ++nt) {
                const void* hp[MAXT];
                for (int t = 0; t < nt; ++t) hp[t] = hq[(size_t) t].data();
                std::vector<std::vector<float>> ob(nt, std::vector<float>(H)), ov(nt, std::vector<float>(H));
                float *pb[MAXT], *pv[MAXT];
                for (int t = 0; t < nt; ++t) { pb[t] = ob[(size_t) t].data(); pv[t] = ov[(size_t) t].data(); }
                cpu::iq4nl256_down_rows_base(blob.data(), f.d_row, FF, hp, nt, pb, 0, H);
                if (vnni) {
                    cpu::iq4nl256_down_rows_vnni(blob.data(), f.d_row, FF, hp, nt, pv, 0, H);
                    for (int t = 0; t < nt; ++t) {
                        ++compared;
                        if (std::memcmp(pb[t], pv[t], H * 4)) { std::printf("MISMATCH IQ4_NL down nt %d mode %d token %d\n", nt, mode, t); ++bad; }
                    }
                }
                // width invariance: each token alone gives the bits it gets in the group; row ranges of any length
                // and start give the bits of the full range
                for (int v = 0; v < (vnni ? 2 : 1); ++v) {
                    auto fn = v ? cpu::iq4nl256_down_rows_vnni : cpu::iq4nl256_down_rows_base;
                    float* const* full = v ? pv : pb;
                    std::vector<float> one(H, 0.f);
                    for (int t = 0; t < nt; ++t) {
                        float* op[1] = {one.data()};
                        fn(blob.data(), f.d_row, FF, hp + t, 1, op, 0, H);
                        ++compared;
                        if (std::memcmp(op[0], full[t], H * 4)) { std::printf("MISMATCH IQ4_NL down %s token %d alone vs nt %d\n", v ? "vnni" : "base", t, nt); ++bad; }
                    }
                    std::vector<std::vector<float>> part(nt, std::vector<float>(H, 0.f));
                    float* pp[MAXT];
                    for (int t = 0; t < nt; ++t) pp[t] = part[(size_t) t].data();
                    for (int r = 0; r < H;) {   // ranges of 1..13 rows: every pass width and remainder
                        const int r1 = std::min(H, r + 1 + (r * 7 + nt) % 13);
                        fn(blob.data(), f.d_row, FF, hp, nt, pp, r, r1);
                        r = r1;
                    }
                    for (int t = 0; t < nt; ++t) {
                        ++compared;
                        if (std::memcmp(pp[t], full[t], H * 4)) { std::printf("MISMATCH IQ4_NL down %s row ranges nt %d token %d\n", v ? "vnni" : "base", nt, t); ++bad; }
                    }
                }
                std::vector<float> wrow(FF);
                for (int r : {0, 5, 1234, H - 1}) {
                    tr->to_float(blob.data() + (size_t) r * f.d_row, wrow.data(), FF);
                    for (int t = 0; t < nt; ++t) {
                        double want = 0;
                        for (int i = 0; i < FF; ++i) want += (double) wrow[(size_t) i] * (double) hf[(size_t) t][(size_t) i];
                        st.add(pb[t][r], want);
                    }
                }
            }
        }
        std::printf("  IQ4_NL down rows: vs double reference: max abs %.3g rel %.3g\n", st.max_abs, st.max_rel);
        if (st.max_rel > 1e-3) { std::printf("REFERENCE ERROR too large for IQ4_NL down\n"); ++bad; }
    }
    // ---- IQ4_NL down rows, the extreme grid: weight nibbles and activation codes pinned to their largest magnitudes, so
    // the four-wise sums the VNNI build takes in one step are the ones a saturating pair sum would clip first.  Codes
    // -127..127 (what the Q8_0 quantizer writes in a scaled block) give the same bits in both builds; -128 is exact in
    // the VNNI build (checked against the double reference) and wraps in the madd build's sign step, as in ggml's dot.
    if (vnni) {
        cpu::NativeFmt f;
        std::string err;
        if (!cpu::native_fmt(22, 20, H, FF, f, err)) return 1;
        const ggml_type_traits* tr = ggml_get_type_traits(GGML_TYPE_IQ4_NL);
        const ggml_type_traits* ta = ggml_get_type_traits(GGML_TYPE_Q8_0);
        int grid_bad = 0, grid_n = 0;
        double worst128 = 0;
        for (int wmode = 0; wmode < 5; ++wmode) {        // random, all nibbles 0, 15, and both mixed
            std::vector<uint8_t> blob(f.d_row * (size_t) H);
            fill(blob, 20, 1700 + (uint64_t) wmode, 0);
            if (wmode > 0) {
                const uint8_t pat = wmode == 1 ? 0x00 : wmode == 2 ? 0xFF : wmode == 3 ? 0x0F : 0xF0;
                for (size_t b = 0; b + 18 <= blob.size(); b += 18)
                    for (int i = 2; i < 18; ++i) blob[b + (size_t) i] = pat;
            }
            for (int amode = 0; amode < 6; ++amode) {    // random, all 127, all -127, random 127 / -127, all -128, random 127 / -128
                std::vector<std::vector<uint8_t>> hq(MAXT, std::vector<uint8_t>(cpu::kNativeHBytes, 0));
                for (int t = 0; t < MAXT; ++t)
                    for (int b = 0; b < FF / 32; ++b) {   // block_q8_0: fp16 scale, 32 int8 codes
                        uint8_t* blk = hq[(size_t) t].data() + (size_t) b * 34;
                        const uint16_t d = (uint16_t) (0x2800 + t);
                        std::memcpy(blk, &d, 2);
                        for (int i = 0; i < 32; ++i) {
                            int8_t q = 0;
                            switch (amode) {
                                case 0: q = (int8_t) ((int) (rng() % 255) - 127); break;
                                case 1: q = 127; break;
                                case 2: q = -127; break;
                                case 3: q = (rng() & 1) ? 127 : -127; break;
                                case 4: q = -128; break;
                                default: q = (rng() & 1) ? 127 : -128; break;
                            }
                            blk[2 + i] = (uint8_t) q;
                        }
                    }
                for (int nt = 1; nt <= MAXT; ++nt) {
                    const void* hp[MAXT];
                    for (int t = 0; t < nt; ++t) hp[t] = hq[(size_t) t].data();
                    for (auto range : {std::pair<int, int>{0, H}, {1, 200}, {2047, H}}) {
                        std::vector<std::vector<float>> ob(nt, std::vector<float>(H, 0.f)), ov(nt, std::vector<float>(H, 0.f));
                        float *pb[MAXT], *pv[MAXT];
                        for (int t = 0; t < nt; ++t) { pb[t] = ob[(size_t) t].data(); pv[t] = ov[(size_t) t].data(); }
                        cpu::iq4nl256_down_rows_base(blob.data(), f.d_row, FF, hp, nt, pb, range.first, range.second);
                        cpu::iq4nl256_down_rows_vnni(blob.data(), f.d_row, FF, hp, nt, pv, range.first, range.second);
                        if (amode < 4) {
                            for (int t = 0; t < nt; ++t) {
                                ++grid_n;
                                if (std::memcmp(pb[t], pv[t], H * 4)) { ++grid_bad; std::printf("MISMATCH IQ4_NL extreme grid w %d a %d nt %d\n", wmode, amode, nt); }
                            }
                            continue;
                        }
                        // -128: the VNNI rows against the double reference, relative to the sum of the terms' magnitudes
                        std::vector<float> wrow(FF), hf(FF);
                        for (int r : {range.first, (range.first + range.second) / 2, range.second - 1}) {
                            tr->to_float(blob.data() + (size_t) r * f.d_row, wrow.data(), FF);
                            for (int t = 0; t < nt; ++t) {
                                ta->to_float(hq[(size_t) t].data(), hf.data(), FF);
                                double want = 0, mag = 0;
                                for (int i = 0; i < FF; ++i) {
                                    want += (double) wrow[(size_t) i] * (double) hf[(size_t) i];
                                    mag += std::fabs((double) wrow[(size_t) i] * (double) hf[(size_t) i]);
                                }
                                const double e = std::fabs((double) pv[t][r] - want) / std::max(mag, 1e-30);
                                worst128 = std::max(worst128, e);
                                ++grid_n;
                                if (e > 1e-5) { ++grid_bad; std::printf("REFERENCE ERROR IQ4_NL -128 codes w %d a %d nt %d row %d: %.3g\n", wmode, amode, nt, r, e); }
                            }
                        }
                    }
                }
            }
        }
        std::printf("  IQ4_NL down, extreme weights x extreme activations: %d comparisons, %d mismatches; -128 codes (vnni) vs "
                    "double reference: worst %.3g of the terms' magnitude\n", grid_n, grid_bad, worst128);
        compared += grid_n;
        bad += grid_bad;
    }
    // ---- the down activation's Q8_0 codes: the quantizer writes -128 only in a block whose fp16 scale is 0 (a block too
    // small for 127 / amax to be finite), where every product is multiplied by that 0: the builds' different -128
    // products cannot reach an output bit.  Magnitudes at, just under and far from powers of two, both signs at amax.
    {
        cpu::NativeFmt f;
        std::string err;
        if (!cpu::native_fmt(22, 20, H, FF, f, err)) return 1;
        int codes128 = 0, scaled128 = 0;
        std::vector<float> h(FF);
        std::vector<uint8_t> hq(cpu::kNativeHBytes);
        for (int mode = 0; mode < 9; ++mode)
            for (int rep = 0; rep < 64; ++rep) {
                const float scale = mode == 0 ? 1.f : mode == 1 ? std::nextafter(2.f, 0.f) : mode == 2 ? 3e38f
                                  : mode == 3 ? 1e-30f : mode == 4 ? 1e-40f : mode == 5 ? std::ldexp(1.f, -149 + 4 * rep)
                                  : mode == 6 ? 65504.f : mode == 7 ? 1e-7f : std::nextafter(std::ldexp(1.f, rep - 32), 0.f);
                for (int i = 0; i < FF; ++i) {
                    const int k = (int) (rng() % 4);
                    h[(size_t) i] = scale * (k == 0 ? 1.f : k == 1 ? -1.f : (float) ((int) (rng() % 2001) - 1000) / 1000.f);
                }
                cpu::native_quant_h(f, h.data(), hq.data());
                for (int b = 0; b < FF / 32; ++b) {
                    const uint8_t* blk = hq.data() + (size_t) b * 34;
                    uint16_t d;
                    std::memcpy(&d, blk, 2);
                    for (int i = 0; i < 32; ++i)
                        if ((int8_t) blk[2 + i] == -128) {
                            ++codes128;
                            if ((d & 0x7fff) != 0) ++scaled128;
                        }
                }
            }
        std::printf("  Q8_0 down activations: %d codes -128, %d of them in a block with a nonzero scale\n", codes128, scaled128);
        if (scaled128 != 0) { std::printf("QUANTIZER wrote -128 in a block with a nonzero scale\n"); ++bad; }
    }
    std::printf("iq_avx2_test: %d base/vnni comparisons, %d failures\n", compared, bad);

    // ---- Q2_0 down rows (the pool's ActQ activations): 64-value blocks of 18 bytes, value i in byte i/4 bits 2*(i%4)
    {
        constexpr int NB = FF / 64;
        constexpr size_t ROW = (size_t) NB * 18;
        Stats st, st512;
        int q2_512_diff = 0;
        for (int mode = 0; mode < 3; ++mode) {   // random, all codes 3, random codes with activations at +-127
            std::vector<uint8_t> blob(ROW * (size_t) H);
            std::mt19937_64 wr(2300 + (uint64_t) mode);
            for (auto& b : blob) b = mode == 1 ? 0xFF : (uint8_t) wr();
            for (size_t b = 0; b + 18 <= blob.size(); b += 18) {
                const uint16_t d = (uint16_t) (0x2800 + (blob[b] & 0xff));
                std::memcpy(&blob[b], &d, 2);
            }
            std::vector<cpu::ActQ> a(MAXT);
            std::vector<float> h(FF);
            for (int t = 0; t < MAXT; ++t) {
                for (auto& v : h) v = mode == 2 ? (rng() & 1 ? 1.f : -1.f) : nd(rng);
                cpu::act_quant_q8_1_avx2(h.data(), FF, a[(size_t) t]);
                if (avx512) {   // the AVX-512 quantizer is the same rule
                    cpu::ActQ a512;
                    cpu::act_quant_q8_1(h.data(), FF, a512);
                    ++compared;
                    const cpu::ActQ& a2 = a[(size_t) t];
                    bool same = a512.nchunks == a2.nchunks && !std::memcmp(a512.q, a2.q, FF);
                    for (int c = 0; same && c < a2.nchunks; ++c)
                        same = !std::memcmp(&a512.scale[c], &a2.scale[c], 4) && a512.sum[c] == a2.sum[c];
                    if (!same) {
                        std::printf("MISMATCH act_quant avx512 mode %d token %d\n", mode, t);
                        ++bad;
                    }
                }
            }
            const cpu::ActQ* ap[MAXT];
            for (int t = 0; t < MAXT; ++t) ap[t] = &a[(size_t) t];
            for (int nt = 1; nt <= MAXT; ++nt) {
                std::vector<std::vector<float>> ob(nt, std::vector<float>(H)), ov(nt, std::vector<float>(H));
                float *pb[MAXT], *pv[MAXT];
                for (int t = 0; t < nt; ++t) { pb[t] = ob[(size_t) t].data(); pv[t] = ov[(size_t) t].data(); }
                cpu::q2_0_gguf_rows_multi_avx2_base(blob.data(), ROW, NB, ap, nt, pb, 0, H);
                if (avx512) {   // the AVX-512 rows give the same bits
                    std::vector<std::vector<float>> oa(nt, std::vector<float>(H));
                    float* pa[MAXT];
                    for (int t = 0; t < nt; ++t) pa[t] = oa[(size_t) t].data();
                    cpu::q2_0_gguf_rows_multi(blob.data(), ROW, NB, ap, nt, pa, 0, H);
                    for (int t = 0; t < nt; ++t) {
                        ++compared;
                        if (std::memcmp(pa[t], pb[t], H * 4)) ++q2_512_diff;
                        for (int r = 0; r < H; ++r) st512.add(pa[t][r], pb[t][r]);
                    }
                }
                if (vnni) {
                    cpu::q2_0_gguf_rows_multi_avx2_vnni(blob.data(), ROW, NB, ap, nt, pv, 0, H);
                    for (int t = 0; t < nt; ++t) {
                        ++compared;
                        if (std::memcmp(pb[t], pv[t], H * 4)) { std::printf("MISMATCH Q2_0 down nt %d mode %d token %d\n", nt, mode, t); ++bad; }
                    }
                }
                for (int v = 0; v < (vnni ? 2 : 1); ++v) {
                    auto fn = v ? cpu::q2_0_gguf_rows_multi_avx2_vnni : cpu::q2_0_gguf_rows_multi_avx2_base;
                    float* const* full = v ? pv : pb;
                    std::vector<float> one(H, 0.f);
                    for (int t = 0; t < nt; ++t) {
                        float* op[1] = {one.data()};
                        fn(blob.data(), ROW, NB, ap + t, 1, op, 0, H);
                        ++compared;
                        if (std::memcmp(op[0], full[t], H * 4)) { std::printf("MISMATCH Q2_0 down %s token %d alone vs nt %d\n", v ? "vnni" : "base", t, nt); ++bad; }
                    }
                    std::vector<std::vector<float>> part(nt, std::vector<float>(H, 0.f));
                    float* pp[MAXT];
                    for (int t = 0; t < nt; ++t) pp[t] = part[(size_t) t].data();
                    for (int r = 0; r < H;) {
                        const int r1 = std::min(H, r + 1 + (r * 7 + nt) % 13);
                        fn(blob.data(), ROW, NB, ap, nt, pp, r, r1);
                        r = r1;
                    }
                    for (int t = 0; t < nt; ++t) {
                        ++compared;
                        if (std::memcmp(pp[t], full[t], H * 4)) { std::printf("MISMATCH Q2_0 down %s row ranges nt %d token %d\n", v ? "vnni" : "base", nt, t); ++bad; }
                    }
                }
                for (int r : {0, 5, 1234, H - 1}) {   // (code - 1) * d_w * q * scale, in double
                    const uint8_t* row = blob.data() + (size_t) r * ROW;
                    for (int t = 0; t < nt; ++t) {
                        double want = 0;
                        for (int b = 0; b < NB; ++b) {
                            uint16_t dh;
                            std::memcpy(&dh, row + b * 18, 2);
                            const double dw = ggml_fp16_to_fp32(dh);
                            for (int i = 0; i < 64; ++i) {
                                const int code = (row[b * 18 + 2 + i / 4] >> (2 * (i % 4))) & 3;
                                want += dw * (code - 1) * (double) a[(size_t) t].q[b * 64 + i] * (double) a[(size_t) t].scale[(b * 64 + i) / 32];
                            }
                        }
                        st.add(pb[t][r], want);
                    }
                }
            }
        }
        std::printf("  Q2_0 down rows: vs double reference: max abs %.3g rel %.3g\n", st.max_abs, st.max_rel);
        if (st.max_rel > 1e-3) { std::printf("REFERENCE ERROR too large for Q2_0 down\n"); ++bad; }
        if (avx512) {
            std::printf("  Q2_0 down rows: avx512 vs avx2: %d tokens differ, max abs %.3g rel %.3g\n", q2_512_diff, st512.max_abs, st512.max_rel);
            if (st512.max_rel > 1e-3) { std::printf("AVX-512 Q2_0 down rows too far from AVX2\n"); ++bad; }
        }
    }
    std::printf("iq_avx2_test: %d base/vnni comparisons, %d failures\n", compared, bad);

    if (bench) {   // 200 gate/up rows (the pool's task) per call, the arena far larger than the caches
        for (int type : {22, 18, 21}) {
            cpu::NativeFmt f;
            std::string err;
            cpu::native_fmt(type, 20, H, FF, f, err);
            std::vector<uint8_t> blob(2 * f.up_off);
            fill(blob, type, 5, 0);
            std::vector<uint8_t> act(cpu::kNativeActBytes);
            std::vector<float> x(H);
            for (auto& v : x) v = nd(rng);
            cpu::native_quant_act(f, x.data(), act.data());
            for (int nt : {1, 2, 4}) {
                const void* ap[MAXT];
                for (int t = 0; t < nt; ++t) ap[t] = act.data();
                std::vector<std::vector<float>> ff(nt, std::vector<float>(FF));
                float* fp[MAXT];
                for (int t = 0; t < nt; ++t) fp[t] = ff[(size_t) t].data();
                for (int v = 0; v < (vnni ? 2 : 1); ++v) {
                    auto fn = v ? cpu::iq256_gu_rows_vnni : cpu::iq256_gu_rows_base;
                    const int reps = 400;
                    const double t0 = now_ms();
                    for (int r = 0; r < reps; ++r) fn(type, blob.data(), f.gu_row, f.up_off, H, ap, nt, fp, 0, 200);
                    const double ms = now_ms() - t0;
                    std::printf("  type %2d nt %d %s: %.2f GB/s (in cache, one thread)\n", type, nt, v ? "vnni" : "base",
                                (double) reps * 400.0 * (double) f.gu_row / (ms * 1e6));
                }
            }
        }
    }
    if (bench) {   // one expert's down rows (2560) per call, one thread, the expert in cache
        cpu::NativeFmt f;
        std::string err;
        cpu::native_fmt(22, 20, H, FF, f, err);
        std::vector<uint8_t> blob(f.d_row * (size_t) H), q2blob((size_t) FF / 64 * 18 * H);
        fill(blob, 20, 6, 0);
        std::mt19937_64 wr(7);
        for (auto& b : q2blob) b = (uint8_t) wr();
        for (size_t b = 0; b + 18 <= q2blob.size(); b += 18) { const uint16_t d = 0x2800; std::memcpy(&q2blob[b], &d, 2); }
        std::vector<uint8_t> hq(cpu::kNativeHBytes);
        cpu::ActQ a;
        std::vector<float> h(FF);
        for (auto& v : h) v = nd(rng);
        cpu::native_quant_h(f, h.data(), hq.data());
        cpu::act_quant_q8_1_avx2(h.data(), FF, a);
        for (int nt : {1, 2, 4}) {
            const void* hp[MAXT];
            const cpu::ActQ* ap[MAXT];
            for (int t = 0; t < nt; ++t) { hp[t] = hq.data(); ap[t] = &a; }
            std::vector<std::vector<float>> o(nt, std::vector<float>(H));
            float* op[MAXT];
            for (int t = 0; t < nt; ++t) op[t] = o[(size_t) t].data();
            for (int v = 0; v < (vnni ? 2 : 1); ++v) {
                auto fi = v ? cpu::iq4nl256_down_rows_vnni : cpu::iq4nl256_down_rows_base;
                auto fq = v ? cpu::q2_0_gguf_rows_multi_avx2_vnni : cpu::q2_0_gguf_rows_multi_avx2_base;
                const int reps = 200;
                for (int r = 0; r < 5; ++r) fi(blob.data(), f.d_row, FF, hp, nt, op, 0, H);
                double t0 = now_ms();
                for (int r = 0; r < reps; ++r) fi(blob.data(), f.d_row, FF, hp, nt, op, 0, H);
                const double ui = (now_ms() - t0) * 1e3 / reps;
                for (int r = 0; r < 5; ++r) fq(q2blob.data(), (size_t) FF / 64 * 18, FF / 64, ap, nt, op, 0, H);
                t0 = now_ms();
                for (int r = 0; r < reps; ++r) fq(q2blob.data(), (size_t) FF / 64 * 18, FF / 64, ap, nt, op, 0, H);
                const double uq = (now_ms() - t0) * 1e3 / reps;
                std::printf("  down nt %d %s: IQ4_NL %.1f us (%.2f GB/s), Q2_0 %.1f us (%.2f GB/s) per expert (in cache, one thread)\n",
                            nt, v ? "vnni" : "base", ui, (double) blob.size() / ui / 1e3, uq, (double) q2blob.size() / uq / 1e3);
            }
        }
    }
    return bad == 0 ? 0 : 1;
}
