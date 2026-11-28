// tests/kernels/cpu/pool_native_test.cpp - the pool's multi-token native path against a serial reference.
//
//     build/pool_native_test [--bench]
//
// `run_split_multi_native` cuts a layer's experts into row ranges and hands them to the workers; whatever the
// cut, every output row must be bitwise what one thread computes alone (the gate/up rows, the intermediate's
// quantization, the down rows).  The check runs the formats of the i-quant and NVFP4 packs on random blobs, 1 to 130
// experts (more than the pool's buffers too), 1 to 8 tokens per expert, at several active worker counts, with a
// pause longer than the workers' spin in the middle, and fails on the first differing word or untouched output.
// `--bench` prints the time and rate of one layer of 15 experts.
#include "basalt/models/qwen4exp/geometry.hpp"
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/kernels/cpu/swiglu.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace cpu = basalt::kernels::cpu;

namespace {

constexpr int H = cpu::H, FF = cpu::FF, MAXT = cpu::MAXT;
constexpr int kBlobs = 48;   // distinct blobs per format; experts index into them

// NVFP4: rows quantized from random values with a spread of block scales, and the blob's FP32 scale tail.
void fill_nvfp4_blob(uint8_t* p, const cpu::NativeFmt& f, uint64_t seed) {
    std::mt19937 rng((uint32_t) seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ex(-6.f, 2.f), ts(0.004f, 0.03f);
    const auto* tr = ggml_get_type_traits(GGML_TYPE_NVFP4);
    std::vector<float> row((size_t) std::max(f.n_embd, f.n_ff));
    auto fill = [&](uint8_t* dst, int rows, int cols, size_t row_bytes) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; c += 16) {
                const float m = std::exp2(ex(rng));
                for (int j = 0; j < 16; ++j) row[(size_t) (c + j)] = nd(rng) * m;
            }
            tr->from_float_ref(row.data(), dst + (size_t) r * row_bytes, cols);
        }
    };
    fill(p, 2 * FF, (int) f.n_embd, f.gu_row);
    fill(p + f.down_off, H, (int) f.n_ff, f.d_row);
    const float tail[4] = {ts(rng), ts(rng), ts(rng), 0.f};
    std::memcpy(p + f.bytes - cpu::kNvfp4Tail, tail, sizeof tail);
}

// Random bytes with a valid small fp16 scale at the start of every block of the gate/up and down parts.
void fill_blob(uint8_t* p, const cpu::NativeFmt& f, uint64_t seed) {
    if (f.gu_type == GGML_TYPE_NVFP4) { fill_nvfp4_blob(p, f, seed); return; }
    std::mt19937_64 rng(seed);
    for (size_t i = 0; i + 8 <= f.bytes; i += 8) {
        const uint64_t v = rng();
        std::memcpy(p + i, &v, 8);
    }
    auto scales = [&](size_t off, size_t len, int type) {
        const size_t bs = ggml_type_size((ggml_type) type);
        for (size_t b = 0; b + bs <= len; b += bs) {
            const uint16_t d = (uint16_t) (0x2800 + (p[off + b] & 0xff));
            if (type == GGML_TYPE_IQ1_M) {
                uint16_t sc[4];
                std::memcpy(sc, p + off + b + bs - 8, 8);
                for (int k = 0; k < 4; ++k) sc[k] = (uint16_t) ((sc[k] & 0x0fffu) | (((d >> (4 * k)) & 0xfu) << 12));
                std::memcpy(p + off + b + bs - 8, sc, 8);
            } else {
                std::memcpy(p + off + b, &d, 2);
            }
        }
    };
    scales(0, 2 * f.up_off, f.gu_type);
    scales(f.down_off, f.bytes - f.down_off, f.d_type);
}

struct Format {
    int gu, d;
    cpu::NativeFmt f;
    std::vector<std::vector<uint8_t>> blobs;
    std::vector<std::vector<uint8_t>> nact;   // one gate/up activation per token slot
    std::vector<cpu::ActQ> actq;              // the same inputs for a Q2_0 gate/up
};

// One expert for `nt` tokens on a single thread: the definition of the right answer.
void serial_expert(const Format& fm, const uint8_t* blob, int nt, float* const* out) {
    const cpu::NativeFmt& f = fm.f;
    alignas(64) float ff[MAXT][FF];
    float* ffp[MAXT];
    const void* act[MAXT];
    for (int t = 0; t < nt; ++t) { ffp[t] = ff[t]; act[t] = fm.nact[(size_t) t].data(); }
    if (cpu::q2_native_kernels(f.gu_type)) {
        alignas(64) float gate[MAXT][FF], up[MAXT][FF];
        float *gp[MAXT], *upp[MAXT];
        const cpu::ActQ* ap[MAXT];
        for (int t = 0; t < nt; ++t) { gp[t] = gate[t]; upp[t] = up[t]; ap[t] = &fm.actq[(size_t) t]; }
        const int nbk = (int) (f.n_embd / 64);
        cpu::q2_rows_any(blob, f.gu_row, nbk, ap, nt, gp, 0, FF);
        cpu::q2_rows_any(blob + f.up_off, f.gu_row, nbk, ap, nt, upp, 0, FF);
        for (int t = 0; t < nt; ++t)
            for (int r = 0; r < FF; ++r) ff[t][r] = (gate[t][r] / (1.f + std::exp(-gate[t][r]))) * up[t][r];
    } else {
        cpu::native_gu_rows(f, blob, act, nt, ffp, 0, FF);
    }
    if (cpu::q2_native_kernels(f.d_type)) {
        cpu::ActQ a2[MAXT];
        const cpu::ActQ* ap[MAXT];
        for (int t = 0; t < nt; ++t) { cpu::act_quant_any(ff[t], FF, a2[t]); ap[t] = &a2[t]; }
        cpu::q2_rows_any(blob + f.down_off, f.d_row, (int) (f.n_ff / 64), ap, nt, out, 0, H);
    } else {
        alignas(64) uint8_t hq[MAXT][cpu::kNativeHBytes];
        const void* hp[MAXT];
        for (int t = 0; t < nt; ++t) { cpu::native_quant_h(f, ff[t], hq[t]); hp[t] = hq[t]; }
        cpu::native_down_rows(f, blob, hp, nt, out, 0, H);
    }
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    if (!cpu::cpu_avx2_ok()) {
        std::printf("pool_native_test SKIPPED: this CPU lacks AVX2\n");
        return 0;
    }
    std::vector<Format> formats;
    for (auto g : {std::pair<int, int>{22, 20}, {18, 20}, {21, 20}, {23, 20}, {22, 42}, {21, 42}, {42, 20}, {42, 42}, {40, 40}}) {
        Format fm;
        fm.gu = g.first;
        fm.d = g.second;
        std::string err;
        if (!cpu::native_fmt(fm.gu, fm.d, H, FF, fm.f, err)) {
            std::printf("native_fmt(%d, %d): %s\n", fm.gu, fm.d, err.c_str());
            return 1;
        }
        fm.blobs.resize(kBlobs);
        for (int b = 0; b < kBlobs; ++b) {
            fm.blobs[(size_t) b].resize(fm.f.bytes);
            fill_blob(fm.blobs[(size_t) b].data(), fm.f, (uint64_t) fm.gu * 1000 + (uint64_t) fm.d * 100 + (uint64_t) b);
        }
        std::mt19937 rng((uint32_t) (fm.gu * 31 + fm.d));
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> x(H);
        fm.nact.assign(MAXT, std::vector<uint8_t>(cpu::kNativeActBytes));
        fm.actq.resize(MAXT);
        for (int t = 0; t < MAXT; ++t) {
            for (auto& v : x) v = nd(rng);
            cpu::native_quant_act(fm.f, x.data(), fm.nact[(size_t) t].data());
            cpu::act_quant_any(x.data(), H, fm.actq[(size_t) t]);
        }
        formats.push_back(std::move(fm));
    }

    const int hw = (int) std::thread::hardware_concurrency();
    const int W = std::max(3, std::min(15, hw - 1));
    cpu::ExpertPool pool(W);
    const float kSentinel = -1.2345e30f;
    const int ns[] = {1, 2, 3, 7, 15, 16, 33, 64, 96, 97, 130};
    int checked = 0, bad = 0;
    std::mt19937 rng(5);
    for (const Format& fm : formats) {
        for (int n : ns) {
            std::vector<int> nts((size_t) n), idx((size_t) n);
            for (int e = 0; e < n; ++e) { nts[(size_t) e] = 1 + (int) (rng() % MAXT); idx[(size_t) e] = (int) (rng() % kBlobs); }
            std::vector<std::vector<float>> want((size_t) n * MAXT, std::vector<float>(H)), got((size_t) n * MAXT, std::vector<float>(H));
            for (int e = 0; e < n; ++e) {
                float* o[MAXT];
                for (int t = 0; t < nts[(size_t) e]; ++t) o[t] = want[(size_t) e * MAXT + (size_t) t].data();
                serial_expert(fm, fm.blobs[(size_t) idx[(size_t) e]].data(), nts[(size_t) e], o);
            }
            std::vector<cpu::ExpertJobMulti> jobs((size_t) n);
            for (int e = 0; e < n; ++e) {
                cpu::ExpertJobMulti& j = jobs[(size_t) e];
                j.blob = fm.blobs[(size_t) idx[(size_t) e]].data();
                j.nt = nts[(size_t) e];
                for (int t = 0; t < j.nt; ++t) {
                    j.nact[t] = fm.nact[(size_t) t].data();
                    j.act[t] = &fm.actq[(size_t) t];
                    j.out[t] = got[(size_t) e * MAXT + (size_t) t].data();
                }
            }
            for (int k : {W, 1, 2, std::max(1, W / 2), W, W}) {
                pool.set_active_workers(k);
                for (auto& v : got) std::fill(v.begin(), v.end(), kSentinel);
                pool.run_split_multi_native(fm.f, jobs.data(), n);
                ++checked;
                bool ok = true;
                for (int e = 0; e < n && ok; ++e)
                    for (int t = 0; t < nts[(size_t) e]; ++t)
                        if (std::memcmp(want[(size_t) e * MAXT + (size_t) t].data(), got[(size_t) e * MAXT + (size_t) t].data(), H * sizeof(float)) != 0) {
                            std::printf("MISMATCH gu %d down %d: %d experts, active workers %d, expert %d token %d\n", fm.gu, fm.d, n, k, e, t);
                            ok = false;
                            break;
                        }
                if (!ok) ++bad;
            }
            pool.set_active_workers(W);
        }
        // workers that went to sleep between batches wake into the next one
        {
            std::this_thread::sleep_for(cpu::ExpertPool::kSpinBeforeSleep + std::chrono::milliseconds(15));
            const int n = 3;
            std::vector<cpu::ExpertJobMulti> jobs((size_t) n);
            std::vector<std::vector<float>> want((size_t) n * MAXT, std::vector<float>(H)), got((size_t) n * MAXT, std::vector<float>(H));
            for (int e = 0; e < n; ++e) {
                cpu::ExpertJobMulti& j = jobs[(size_t) e];
                j.blob = fm.blobs[(size_t) e].data();
                j.nt = 2;
                float* o[MAXT];
                for (int t = 0; t < 2; ++t) {
                    j.nact[t] = fm.nact[(size_t) t].data();
                    j.act[t] = &fm.actq[(size_t) t];
                    j.out[t] = got[(size_t) e * MAXT + (size_t) t].data();
                    o[t] = want[(size_t) e * MAXT + (size_t) t].data();
                }
                serial_expert(fm, j.blob, 2, o);
            }
            pool.run_split_multi_native(fm.f, jobs.data(), n);
            ++checked;
            for (int e = 0; e < n; ++e)
                for (int t = 0; t < 2; ++t)
                    if (std::memcmp(want[(size_t) e * MAXT + (size_t) t].data(), got[(size_t) e * MAXT + (size_t) t].data(), H * sizeof(float)) != 0) {
                        std::printf("MISMATCH after sleep, gu %d down %d expert %d token %d\n", fm.gu, fm.d, e, t);
                        ++bad;
                        e = n;
                        break;
                    }
        }
    }
    std::printf("pool_native_test: %d runs over %zu formats, %d mismatches\n", checked, formats.size(), bad);

    // Bounds checks in native_fmt and expert_layout_load_text, Q2_0 null-act fallback, and Q2_0 swiglu_limit parity.
    {
        cpu::NativeFmt bad_f;
        std::string ferr;
        if (cpu::native_fmt(GGML_TYPE_IQ2_XS, GGML_TYPE_IQ4_NL, H, cpu::kNativeMaxFF + 256, bad_f, ferr)) {
            std::printf("FAIL: native_fmt accepted n_ff > kNativeMaxFF\n");
            ++bad;
        }
        if (cpu::native_fmt(GGML_TYPE_Q2_0, GGML_TYPE_Q2_0, cpu::kActMaxWidth + 256, FF, bad_f, ferr)) {
            std::printf("FAIL: native_fmt accepted Q2_0 n_embd > kActMaxWidth\n");
            ++bad;
        }
        if (cpu::native_fmt(GGML_TYPE_Q4_K, GGML_TYPE_Q8_0, 0, FF, bad_f, ferr) ||
            cpu::native_fmt(GGML_TYPE_Q4_K, GGML_TYPE_Q8_0, H, 0, bad_f, ferr)) {
            std::printf("FAIL: native_fmt accepted zero geometry\n");
            ++bad;
        }
        if (cpu::expert_layout_load_text("# experts v6 (n_expert 0)\n0 20 20 0 100\n", 1, 4, ferr, H, FF) ||
            cpu::expert_layout_load_text("# experts v6\n0 20 20 0 100\n0 20 20 0 100\n", 1, 4, ferr, H, FF) ||
            cpu::expert_layout_load_text("# experts v6\n", 0, 4, ferr, H, FF)) {
            std::printf("FAIL: expert_layout_load_text accepted invalid layout\n");
            ++bad;
        }

        // Q2_0 job with j.act == nullptr must fall back to j.nact cleanly without dereferencing null.
        for (const Format& fm : formats) {
            if (fm.gu != GGML_TYPE_Q2_0 || fm.d != GGML_TYPE_Q2_0) continue;
            cpu::ExpertJobMulti j;
            j.blob = fm.blobs[0].data();
            j.nt = 2;
            std::vector<float> got0(H, kSentinel), got1(H, kSentinel);
            j.nact[0] = fm.nact[0].data();
            j.nact[1] = fm.nact[1].data();
            j.out[0] = got0.data();
            j.out[1] = got1.data();
            pool.run_split_multi_native(fm.f, &j, 1);
            if (got0[0] == kSentinel || got1[0] == kSentinel || !std::isfinite(got0[0]) || !std::isfinite(got1[0])) {
                std::printf("FAIL: Q2_0 null-act fallback produced invalid output\n");
                ++bad;
            }
            // Q2_0 with swiglu_limit > 0 must clamp gate/up via SwiGlu, matching the scalar swiglu() reference.
            cpu::NativeFmt lim_f = fm.f;
            lim_f.swiglu_limit = 0.05f;
            cpu::ExpertJobMulti lj;
            lj.blob = fm.blobs[0].data();
            lj.nt = 2;
            std::vector<float> lout0(H), lout1(H), ref0(H), ref1(H);
            lj.act[0] = &fm.actq[0];
            lj.act[1] = &fm.actq[1];
            lj.nact[0] = fm.nact[0].data();
            lj.nact[1] = fm.nact[1].data();
            lj.out[0] = lout0.data();
            lj.out[1] = lout1.data();
            pool.run_split_multi_native(lim_f, &lj, 1);
            // Serial reference with swiglu_limit
            alignas(64) float g[MAXT][FF], u[MAXT][FF], ff[MAXT][FF];
            float *gp[2] = {g[0], g[1]}, *up[2] = {u[0], u[1]};
            const cpu::ActQ* ap[2] = {&fm.actq[0], &fm.actq[1]};
            cpu::q2_rows_any(lj.blob, lim_f.gu_row, (int) (lim_f.n_embd / 64), ap, 2, gp, 0, FF);
            cpu::q2_rows_any(lj.blob + lim_f.up_off, lim_f.gu_row, (int) (lim_f.n_embd / 64), ap, 2, up, 0, FF);
            const cpu::SwiGlu glu{lim_f.swiglu_limit, 0.f, 0.f};
            for (int t = 0; t < 2; ++t)
                for (int r = 0; r < FF; ++r) ff[t][r] = cpu::swiglu(glu, g[t][r], u[t][r]);
            cpu::ActQ a2[2];
            const cpu::ActQ* a2p[2] = {&a2[0], &a2[1]};
            for (int t = 0; t < 2; ++t) cpu::act_quant_any(ff[t], FF, a2[t]);
            float* rp[2] = {ref0.data(), ref1.data()};
            cpu::q2_rows_any(lj.blob + lim_f.down_off, lim_f.d_row, (int) (lim_f.n_ff / 64), a2p, 2, rp, 0, H);
            if (std::memcmp(lout0.data(), ref0.data(), H * sizeof(float)) != 0 ||
                std::memcmp(lout1.data(), ref1.data(), H * sizeof(float)) != 0) {
                std::printf("FAIL: Q2_0 pool did not apply swiglu_limit\n");
                ++bad;
            }
            break;
        }
        // IQ1_M gate/up and IQ2_XXS/IQ2_XS/IQ1_M down with swiglu_limit == 0 must match ggml single-token bitwise.
        constexpr int kFFK = 512;   // QK_K-aligned (256) intermediate width for IQ2/IQ1 down projections
        std::mt19937 krng(0x2026u);
        std::normal_distribution<float> knd(0.f, 1.f);
        std::vector<float> kx0(H), kx1(H);
        for (int i = 0; i < H; ++i) { kx0[(size_t) i] = knd(krng); kx1[(size_t) i] = knd(krng); }
        for (auto pair : {std::pair<int, int>{GGML_TYPE_IQ1_M, GGML_TYPE_IQ2_XXS},
                          std::pair<int, int>{GGML_TYPE_IQ1_M, GGML_TYPE_IQ2_XS},
                          std::pair<int, int>{GGML_TYPE_IQ1_M, GGML_TYPE_IQ1_M}}) {
            cpu::NativeFmt ef;
            if (!cpu::native_fmt(pair.first, pair.second, H, kFFK, ef, ferr, 0.f)) {
                std::printf("FAIL: native_fmt(%d, %d): %s\n", pair.first, pair.second, ferr.c_str());
                ++bad;
                continue;
            }
            std::vector<uint8_t> blob(ef.bytes);
            fill_blob(blob.data(), ef, (uint64_t) pair.first * 100 + (uint64_t) pair.second);
            std::vector<uint8_t> xq0(ef.act_bytes), xq1(ef.act_bytes);
            cpu::native_quant_act(ef, kx0.data(), xq0.data());
            cpu::native_quant_act(ef, kx1.data(), xq1.data());
            std::vector<float> ff_got[2] = {std::vector<float>(kFFK), std::vector<float>(kFFK)};
            std::vector<float> ff_ref[2] = {std::vector<float>(kFFK), std::vector<float>(kFFK)};
            float* fp[2] = {ff_got[0].data(), ff_got[1].data()};
            const void* act[2] = {xq0.data(), xq1.data()};
            cpu::native_gu_rows(ef, blob.data(), act, 2, fp, 0, kFFK);
            const auto* tg = ggml_get_type_traits_cpu((ggml_type) pair.first);
            for (int r = 0; r < kFFK; ++r)
                for (int t = 0; t < 2; ++t) {
                    float g = 0.f, u = 0.f;
                    tg->vec_dot(H, &g, 0, blob.data() + (size_t) r * ef.gu_row, 0, act[t], 0, 1);
                    tg->vec_dot(H, &u, 0, blob.data() + ef.up_off + (size_t) r * ef.up_row, 0, act[t], 0, 1);
                    ff_ref[t][(size_t) r] = (g / (1.f + std::exp(-g))) * u;
                }
            if (std::memcmp(ff_got[0].data(), ff_ref[0].data(), kFFK * sizeof(float)) != 0 ||
                std::memcmp(ff_got[1].data(), ff_ref[1].data(), kFFK * sizeof(float)) != 0) {
                std::printf("FAIL: IQ1_M native_gu_rows mismatch vs ggml\n");
                ++bad;
            }
            std::vector<uint8_t> hq0(ef.h_bytes), hq1(ef.h_bytes);
            cpu::native_quant_h(ef, ff_ref[0].data(), hq0.data());
            cpu::native_quant_h(ef, ff_ref[1].data(), hq1.data());
            const void* hqp[2] = {hq0.data(), hq1.data()};
            std::vector<float> d_got[2] = {std::vector<float>(H), std::vector<float>(H)};
            std::vector<float> d_ref[2] = {std::vector<float>(H), std::vector<float>(H)};
            float* dp[2] = {d_got[0].data(), d_got[1].data()};
            cpu::native_down_rows(ef, blob.data(), hqp, 2, dp, 0, H);
            const auto* td = ggml_get_type_traits_cpu((ggml_type) pair.second);
            for (int r = 0; r < H; ++r)
                for (int t = 0; t < 2; ++t) {
                    float s = 0.f;
                    td->vec_dot(kFFK, &s, 0, blob.data() + ef.down_off + (size_t) r * ef.d_row, 0, hqp[t], 0, 1);
                    d_ref[t][(size_t) r] = s;
                }
            if (std::memcmp(d_got[0].data(), d_ref[0].data(), H * sizeof(float)) != 0 ||
                std::memcmp(d_got[1].data(), d_ref[1].data(), H * sizeof(float)) != 0) {
                std::printf("FAIL: type %d native_down_rows mismatch vs ggml\n", pair.second);
                ++bad;
            }
        }
    }

    if (bench) {   // a decode layer (15 experts x 2 tokens) and a prompt-sized one (96 x 8), experts far apart in memory
        for (const Format& fm : formats) {
            for (auto shape : {std::pair<int, int>{15, 2}, {96, 8}}) {
                const int n = shape.first, nt = shape.second, reps = n > 32 ? 40 : 200;
                std::vector<std::vector<float>> got((size_t) n * (size_t) nt, std::vector<float>(H));
                std::vector<cpu::ExpertJobMulti> jobs((size_t) n);
                std::vector<double> us;
                for (int r = 0; r < reps; ++r) {
                    for (int e = 0; e < n; ++e) {
                        cpu::ExpertJobMulti& j = jobs[(size_t) e];
                        j.blob = fm.blobs[(size_t) ((r * n + e) % kBlobs)].data();
                        j.nt = nt;
                        for (int t = 0; t < nt; ++t) {
                            j.nact[t] = fm.nact[(size_t) t].data();
                            j.act[t] = &fm.actq[(size_t) t];
                            j.out[t] = got[(size_t) e * (size_t) nt + (size_t) t].data();
                        }
                    }
                    const double t0 = now_ms();
                    pool.run_split_multi_native(fm.f, jobs.data(), n);
                    us.push_back((now_ms() - t0) * 1e3);
                }
                std::sort(us.begin(), us.end());
                const double med = us[us.size() / 2];
                std::printf("  gu %2d down %2d: %2d experts x %d tokens, %d workers + host: median %.1f us (%.1f GB/s)\n", fm.gu, fm.d, n, nt,
                            pool.workers(), med, (double) n * (double) fm.f.bytes / (med * 1e3));
            }
        }
    }
    return bad == 0 ? 0 : 1;
}
