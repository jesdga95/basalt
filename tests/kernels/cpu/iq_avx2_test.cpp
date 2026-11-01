// tests/kernels/cpu/iq_avx2_test.cpp - the AVX2 i-quant rows: the `madd`+`add` build against the AVX-VNNI build, and both
// against a plain double-precision reference.
//
//     build/iq_avx2_test [--bench]
//
// The two builds must agree to the bit (the VNNI instructions are the same arithmetic in one step): gate/up rows of
// IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S and IQ4_XS, the plain dot rows, and the IQ4_NL down rows, on random weights and
// activations with the model's row geometry, 1 to 8 tokens, plus weights and activations pinned to their extreme codes.
// The reference dequantizes the weights and the activation and sums in double; its error is reported (the float order of
// the kernels differs from it by rounding only). On a CPU without AVX-VNNI the base build is checked against the
// reference alone. `--bench` prints one row range per format on each build.
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/iq_avx2.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
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

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    if (!cpu::cpu_avx2_ok()) {
        std::printf("iq_avx2_test SKIPPED: this CPU lacks AVX2\n");
        return 0;
    }
    const bool vnni = cpu::cpu_avxvnni_ok();
    std::printf("iq_avx2_test: %s, AVX-VNNI %s\n", cpu::cpu_name().c_str(), vnni ? "yes" : "no (base build only)");
    int bad = 0, compared = 0;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);

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
    // ---- IQ4_NL down rows, the extreme grid: weight nibbles and activation codes pinned to their largest magnitudes, so the
    // four-wise sums the VNNI build takes in one step are the ones a saturating pair sum would clip first
    if (vnni) {
        cpu::NativeFmt f;
        std::string err;
        if (!cpu::native_fmt(22, 20, H, FF, f, err)) return 1;
        int grid_bad = 0, grid_n = 0;
        for (int wmode = 0; wmode < 5; ++wmode) {        // random, all nibbles 0, 15, and both mixed
            std::vector<uint8_t> blob(f.d_row * (size_t) H);
            fill(blob, 20, 1700 + (uint64_t) wmode, 0);
            if (wmode > 0) {
                const uint8_t pat = wmode == 1 ? 0x00 : wmode == 2 ? 0xFF : wmode == 3 ? 0x0F : 0xF0;
                for (size_t b = 0; b + 18 <= blob.size(); b += 18)
                    for (int i = 2; i < 18; ++i) blob[b + (size_t) i] = pat;
            }
            for (int amode = 0; amode < 5; ++amode) {    // random, all 127, all -128, all -127, random 127 / -128
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
                                case 2: q = -128; break;
                                case 3: q = -127; break;
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
                        for (int t = 0; t < nt; ++t) {
                            ++grid_n;
                            if (std::memcmp(pb[t], pv[t], H * 4)) { ++grid_bad; std::printf("MISMATCH IQ4_NL extreme grid w %d a %d nt %d\n", wmode, amode, nt); }
                        }
                    }
                }
            }
        }
        std::printf("  IQ4_NL down, extreme weights x extreme activations: %d comparisons, %d mismatches\n", grid_n, grid_bad);
        compared += grid_n;
        bad += grid_bad;
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
    return bad == 0 ? 0 : 1;
}
