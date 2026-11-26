// tests/models/inkling2/native_expert_cpu_parity.cpp - Inkling 2's routed experts in the GSQ-RCO GGUF types on the CPU.
//
//     build/inkling2_native_expert_cpu_parity
//
// ISTA-DASLab's GSQ-RCO Inkling 2 Mini GGUFs keep each layer's experts in its own types: gate/up IQ1_S, IQ1_M,
// IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS or IQ3_S over a Q2_0 or IQ4_NL down. The engine's CPU experts (cpu_experts.cpp)
// run them through ExpertPool::run_split_multi_native with the softcapped SwiGLU. For every such pair at the model's
// geometry (2304 x 1152, gate softcap 4, up softcap 25) the pool's output is compared with the plain reference: the
// blob dequantized by ggml's to_float, the products and the SwiGLU in double, in two stages that must agree to float
// rounding (kExact, relative to the largest reference value of the token):
//   gate/up  the SwiGLU of the rows against the activation as the kernels see it (Q8_K, dequantized)
//   down     the pool's output against the down rows times the intermediate as the kernels see it (Q8_0 for IQ4_NL,
//            the engine's 32-value int8 chunks for Q2_0, dequantized)
// and end to end against the unquantized activations (reported: the activations' 8-bit rounding, as llama.cpp's CPU
// backend computes). A token's output must also be bitwise the same alone and at any position of a batch of four.
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/cpu/pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace cpu = basalt::kernels::cpu;

namespace {

constexpr int E = 2304, F = 1152, kTok = 4;
constexpr float kGateCap = 4.f, kUpCap = 25.f;
constexpr double kExact = 1e-4;   // float accumulation over 2304 / 1152 terms against double

// Random blocks with a small finite fp16 scale in every block (IQ1_M: in the top nibbles of its four scale words).
void fill(uint8_t* p, size_t len, int type, std::mt19937_64& rng) {
    for (size_t i = 0; i < len; ++i) p[i] = (uint8_t) rng();
    const size_t bs = ggml_type_size((ggml_type) type);
    for (size_t b = 0; b + bs <= len; b += bs) {
        uint8_t* blk = p + b;
        if (type == GGML_TYPE_IQ1_M) {   // scale fp16 0x2800 (2^-5) from the 4 top nibbles of scales[0..3] (bytes 48..55)
            uint16_t sc[4];
            std::memcpy(sc, blk + 48, sizeof sc);
            const uint16_t nib[4] = {0x0, 0x0, 0x8, 0x2};
            for (int i = 0; i < 4; ++i) sc[i] = (uint16_t) ((sc[i] & 0x0FFF) | (nib[i] << 12));
            std::memcpy(blk + 48, sc, sizeof sc);
        } else {
            const uint16_t d = (uint16_t) (0x2800 + (blk[0] & 0xff));
            std::memcpy(blk, &d, 2);
        }
    }
}

std::vector<float> rows_f32(int type, const uint8_t* p, int rows, int cols) {
    std::vector<float> out((size_t) rows * cols);
    ggml_get_type_traits((ggml_type) type)->to_float(p, out.data(), (int64_t) rows * cols);
    return out;
}

double swiglu_ref(double g, double u) {
    return kGateCap * std::tanh(g / kGateCap) * (1.0 / (1.0 + std::exp(-g))) * (kUpCap * std::tanh(u / kUpCap));
}

// The activations as the dot products see them.
std::vector<double> q8_k_values(const uint8_t* p, int n) {   // block_q8_K: float d, int8 qs[256], int16 bsums[16]
    std::vector<double> v((size_t) n);
    for (int b = 0; b < n / 256; ++b) {
        const uint8_t* blk = p + (size_t) b * 292;
        float d;
        std::memcpy(&d, blk, 4);
        for (int i = 0; i < 256; ++i) v[(size_t) b * 256 + i] = (double) d * (int8_t) blk[4 + i];
    }
    return v;
}
std::vector<double> q8_0_values(const uint8_t* p, int n) {   // block_q8_0: fp16 d, int8 qs[32]
    std::vector<double> v((size_t) n);
    for (int b = 0; b < n / 32; ++b) {
        const uint8_t* blk = p + (size_t) b * 34;
        uint16_t h;
        std::memcpy(&h, blk, 2);
        const double d = ggml_fp16_to_fp32(h);
        for (int i = 0; i < 32; ++i) v[(size_t) b * 32 + i] = d * (int8_t) blk[2 + i];
    }
    return v;
}
std::vector<double> actq_values(const cpu::ActQ& a, int n) {   // the engine's chunks: q[i] * scale[i / 32]
    std::vector<double> v((size_t) n);
    for (int i = 0; i < n; ++i) v[(size_t) i] = (double) a.q[i] * a.scale[i / cpu::QKA];
    return v;
}

struct Err {
    double abs = 0.0, rel = 0.0;
    void add(const std::vector<double>& ref, const float* got) {
        double amax = 0.0, e = 0.0;
        for (size_t i = 0; i < ref.size(); ++i) {
            amax = std::max(amax, std::fabs(ref[i]));
            e = std::max(e, std::fabs(ref[i] - (double) got[i]));
        }
        abs = std::max(abs, e);
        rel = std::max(rel, amax > 0 ? e / amax : e);
    }
};

std::vector<double> matvec(const std::vector<float>& w, int rows, int cols, const std::vector<double>& x) {
    std::vector<double> y((size_t) rows);
    for (int r = 0; r < rows; ++r) {
        double s = 0.0;
        for (int c = 0; c < cols; ++c) s += (double) w[(size_t) r * cols + c] * x[(size_t) c];
        y[(size_t) r] = s;
    }
    return y;
}

}  // namespace

int main() {
    if (!cpu::cpu_kernels_ok()) {
        std::printf("inkling2_native_expert_cpu_parity SKIPPED: no AVX2 / NEON\n");
        return 77;
    }
    const int gus[] = {GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,
                       GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S};
    const int downs[] = {GGML_TYPE_Q2_0, GGML_TYPE_IQ4_NL};
    cpu::ExpertPool pool(3);
    int bad = 0, pairs = 0;
    double worst_rel = 0.0;
    for (int gu : gus)
        for (int dt : downs) {
            ++pairs;
            cpu::NativeFmt f;
            std::string err;
            if (!cpu::native_fmt(gu, dt, E, F, f, err)) {
                std::printf("gu %-7s down %-6s native_fmt: %s\n", ggml_type_name((ggml_type) gu), ggml_type_name((ggml_type) dt), err.c_str());
                ++bad;
                continue;
            }
            f.gate_softcap = kGateCap;
            f.up_softcap = kUpCap;
            std::mt19937_64 rng((uint64_t) gu * 100 + (uint64_t) dt);
            std::vector<uint8_t> blob(f.bytes);
            fill(blob.data(), f.down_off, gu, rng);
            fill(blob.data() + f.down_off, f.bytes - f.down_off, dt, rng);
            const std::vector<float> wg = rows_f32(gu, blob.data(), F, E), wu = rows_f32(gu, blob.data() + f.up_off, F, E),
                                     wd = rows_f32(dt, blob.data() + f.down_off, E, F);

            std::normal_distribution<float> nd(0.f, 1.f);
            std::vector<std::vector<float>> x(kTok, std::vector<float>(E));
            std::vector<std::vector<uint8_t>> act(kTok, std::vector<uint8_t>(cpu::kNativeActBytes));
            for (int t = 0; t < kTok; ++t) {
                for (float& v : x[(size_t) t]) v = nd(rng);
                cpu::native_quant_act(f, x[(size_t) t].data(), act[(size_t) t].data());
            }
            auto run = [&](const std::vector<int>& toks, std::vector<std::vector<float>>& out) {
                out.assign(toks.size(), std::vector<float>(E, -1.2345e30f));
                cpu::ExpertJobMulti j{};
                j.blob = blob.data();
                j.nt = (int) toks.size();
                for (size_t i = 0; i < toks.size(); ++i) {
                    j.nact[i] = act[(size_t) toks[i]].data();   // no Q2_0 gate/up here: `act` stays unread
                    j.out[i] = out[i].data();
                }
                pool.run_split_multi_native(f, &j, 1);
            };
            std::vector<std::vector<float>> batch, alone;
            run({0, 1, 2, 3}, batch);
            bool same = true;
            for (int t = 0; t < kTok; ++t) {
                run({t}, alone);
                same &= std::memcmp(alone[0].data(), batch[(size_t) t].data(), E * sizeof(float)) == 0;
                std::vector<std::vector<float>> shifted;
                run({(t + 1) % kTok, (t + 2) % kTok, (t + 3) % kTok, t}, shifted);
                same &= std::memcmp(shifted[3].data(), batch[(size_t) t].data(), E * sizeof(float)) == 0;
            }
            if (f.gu_act != GGML_TYPE_Q8_K || (!cpu::q2_native_kernels(dt) && f.d_act != GGML_TYPE_Q8_0)) {
                std::printf("gu %d down %d: activation types %d / %d, not the Q8_K / Q8_0 this test reads\n", gu, dt, f.gu_act, f.d_act);
                ++bad;
                continue;
            }
            Err gu_err, down_err, e2e;
            std::vector<float> ff(F);
            std::vector<uint8_t> hbuf(cpu::kNativeHBytes);
            static cpu::ActQ a2;
            auto swiglu_rows = [&](const std::vector<double>& xv) {
                const std::vector<double> g = matvec(wg, F, E, xv), u = matvec(wu, F, E, xv);
                std::vector<double> h((size_t) F);
                for (int r = 0; r < F; ++r) h[(size_t) r] = swiglu_ref(g[(size_t) r], u[(size_t) r]);
                return h;
            };
            for (int t = 0; t < kTok; ++t) {
                // gate/up: the rows against the activation as the kernels see it
                const void* a = act[(size_t) t].data();
                float* ffp = ff.data();
                cpu::native_gu_rows(f, blob.data(), &a, 1, &ffp, 0, F);
                gu_err.add(swiglu_rows(q8_k_values(act[(size_t) t].data(), E)), ff.data());
                // down: the pool's output against the intermediate as the kernels see it (pool.cpp native_quantize)
                std::vector<double> hq;
                if (cpu::q2_native_kernels(dt)) {
                    cpu::act_quant_any(ff.data(), F, a2);
                    hq = actq_values(a2, F);
                } else {
                    cpu::native_quant_h(f, ff.data(), hbuf.data());
                    hq = q8_0_values(hbuf.data(), F);
                }
                down_err.add(matvec(wd, E, F, hq), batch[(size_t) t].data());
                // end to end, unquantized activations
                const std::vector<double> xd(x[(size_t) t].begin(), x[(size_t) t].end());
                e2e.add(matvec(wd, E, F, swiglu_rows(xd)), batch[(size_t) t].data());
            }
            worst_rel = std::max({worst_rel, gu_err.rel, down_err.rel});
            const bool ok = same && gu_err.rel <= kExact && down_err.rel <= kExact && std::isfinite(e2e.abs);
            bad += !ok;
            std::printf("gu %-7s down %-6s gate/up rel %.2g  down rel %.2g (abs %.3g)  end-to-end abs %.4g rel %.3g  "
                        "batch-independent %s  %s\n",
                        ggml_type_name((ggml_type) gu), ggml_type_name((ggml_type) dt), gu_err.rel, down_err.rel,
                        down_err.abs, e2e.abs, e2e.rel, same ? "yes" : "NO", ok ? "ok" : "FAIL");
        }
    std::printf("inkling2_native_expert_cpu_parity: %d pairs, worst stage rel %.3g (limit %.0e), %d failed\n", pairs,
                worst_rel, kExact, bad);
    return bad ? 1 : 0;
}
