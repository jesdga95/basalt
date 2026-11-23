// tests/kernels/gqa_attn_parity.cpp - gqa_decode_attn and gqa_prompt_attn (basalt/kernels/gqa_attn.hpp) and
// attn_qk_prepare (dense_ops.hpp) against a CPU reference in double precision, at the Qwen3.5-27B attention geometry
// (24 query heads, 4 KV heads, 256 wide, 64 rotated dims, interleaved M-RoPE sections 11/11/10), unscaled and under YaRN.
//
// Checks: each kernel's max absolute and relative error to the reference (the int8 prompt kernel's narrow, split and
// 8-warp launches; writes to and reads from int8 and NVFP4 caches); a decode row's output is bitwise the same
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
constexpr double kPi = 3.14159265358979323846;

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

// NVFP4 cells as the cache stores them (KvInt8::fp4): an E4M3 scale byte per 16 values, E2M1 codes two to a byte (value 2i
// in byte i's low nibble)
float e4m3_value(uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    return e ? std::ldexp(1.0f + m / 8.0f, e - 7) : std::ldexp(m / 8.0f, -6);
}

// the nearest finite positive E4M3 code, ties to the even code, saturated at 448
uint8_t e4m3_code(float v) {
    uint8_t best = 0;
    float bd = INFINITY;
    for (int b = 0; b <= 0x7e; ++b) {
        const float d = std::fabs(v - e4m3_value((uint8_t) b));
        if (d < bd || (d == bd && !(b & 1))) {
            bd = d;
            best = (uint8_t) b;
        }
    }
    return best;
}

const float kE2m1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

// the nearest E2M1 code, ties to even, bit 3 the sign
uint8_t e2m1_code(float v) {
    const float a = std::fabs(v);
    const int m = a <= 0.25f ? 0 : a < 0.75f ? 1 : a <= 1.25f ? 2 : a < 1.75f ? 3 : a <= 2.5f ? 4 : a < 3.5f ? 5 : a <= 5.0f ? 6 : 7;
    return (uint8_t) (m | (std::signbit(v) ? 8 : 0));
}

float e2m1_value(uint8_t c) { return (c & 8 ? -1.0f : 1.0f) * kE2m1[c & 7]; }

// The write rule: per 16 values the scale amax / 6 (floored so an all-zero group's is finite), each value the nearest
// code against the decoded scale.  `deq`: each value as the decode kernel reads it (code x scale, rounded to fp16).
void quant4(const std::vector<float>& x, std::vector<uint8_t>& codes, std::vector<uint8_t>& scales, std::vector<uint16_t>& deq) {
    codes.assign(x.size() / 2, 0);
    scales.assign(x.size() / 16, 0);
    deq.assign(x.size(), 0);
    for (size_t g = 0; g < x.size() / 16; ++g) {
        float amax = 0;
        for (int i = 0; i < 16; ++i) amax = std::max(amax, std::fabs(x[g * 16 + i]));
        scales[g] = e4m3_code(std::max(amax, 6.0f * 0.001953125f) / 6.0f);
        const float s = e4m3_value(scales[g]);
        for (int i = 0; i < 16; ++i) {
            const size_t at = g * 16 + i;
            const uint8_t c = e2m1_code(std::min(std::max(x[at] / s, -6.0f), 6.0f));
            codes[at / 2] |= (uint8_t) (c << (4 * (at & 1)));
            deq[at] = basalt::kernels::f16_from_f32(e2m1_value(c) * s);
        }
    }
}

uint8_t code_at(const std::vector<uint8_t>& codes, size_t at) { return (uint8_t) ((codes[at / 2] >> (4 * (at & 1))) & 15); }

// device NVFP4 buffers for `values` cache values (K and V each)
basalt::kernels::KvInt8 alloc4(size_t values) {
    basalt::kernels::KvInt8 f;
    f.fp4 = true;
    ck(cudaMalloc(&f.k, values / 2), "malloc");
    ck(cudaMalloc(&f.v, values / 2), "malloc");
    ck(cudaMalloc(&f.ks, values / 16), "malloc");
    ck(cudaMalloc(&f.vs, values / 16), "malloc");
    return f;
}

void free4(const basalt::kernels::KvInt8& f) {
    cudaFree(f.k);
    cudaFree(f.v);
    cudaFree(f.ks);
    cudaFree(f.vs);
}

// the normalized 256-point Hadamard transform (Sylvester order) of one head's values
void hadamard256(const double* y, double* out) {
    for (int i = 0; i < 256; ++i) {
        double a = 0;
        for (int j = 0; j < 256; ++j) {
            int bits = 0;
            for (unsigned m = (unsigned) (i & j); m; m &= m - 1) ++bits;
            a += (bits & 1) ? -y[j] : y[j];
        }
        out[i] = a / 16.0;
    }
}

uint16_t bf16_from_f32(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}

float f32_from_bf16(uint16_t h) {
    const uint32_t u = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

}  // namespace

int main(int argc, char** argv) {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("no GPU\n");
        return 77;
    }
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    int fails = 0;

    // ---- attn_qk_prepare: q/k norm, partial rope with M-RoPE axes, cache write; unscaled, then under YaRN x2 past the
    // trained 262,144 against transformers' YaRN (_compute_yarn_parameters: the frequency blend by pair index, cos and sin
    // times 0.1 ln(factor) + 1), the angle rounded as the kernel takes it (fp32 position times fp32 frequency)
    for (const bool yarn : {false, true}) {
        const int T = 3;
        std::vector<float> qf((size_t) T * NH * 2 * HD), k((size_t) T * NKV * HD), v(k.size()), qn(HD), kn(HD);
        for (auto& x : qf) x = nd(rng);
        for (auto& x : k) x = nd(rng);
        for (auto& x : v) x = nd(rng);
        for (auto& x : qn) x = 1.0f + 0.1f * nd(rng);
        for (auto& x : kn) x = 1.0f + 0.1f * nd(rng);
        std::vector<int32_t> pos = {5, 5, 5, 6, 9, 11, 7, 7, 7};   // a text row, an image row (t, h, w), a text row
        if (yarn) pos = {300001, 300001, 300001, 400000, 400003, 400005, 524287, 524287, 524287};
        basalt::kernels::RopeSpec rope;
        rope.rot = ROT;
        rope.theta = 1e7f;
        for (int i = 0; i < ROT / 2; ++i) rope.axis[i] = (int8_t) (i % 3 == 1 && i < 33 ? 1 : i % 3 == 2 && i < 30 ? 2 : 0);
        const double factor = 2.0, orig = 262144.0, base = 1e7;
        std::vector<double> inv_freq(ROT / 2);
        double mscale = 1.0;
        for (int i = 0; i < ROT / 2; ++i) inv_freq[(size_t) i] = std::pow(base, -2.0 * i / ROT);
        if (yarn) {
            rope.scaling.type = basalt::kernels::RopeScalingType::YaRN;
            rope.scaling.factor = factor;
            rope.scaling.orig_ctx = orig;
            rope.scaling.freq_base = base;
            rope.scaling.ext_factor = 1.0;
            // transformers: find_correction_range(beta_fast 32, beta_slow 1, dim, base, orig) with truncation, a linear
            // ramp over dim / 2 pairs, inv = interp (1 - extrap_factor) + extrap extrap_factor, extrap_factor = 1 - ramp
            const auto corr = [&](double rot) { return ROT * std::log(orig / (rot * 2 * kPi)) / (2 * std::log(base)); };
            const double lo = std::max(0.0, std::floor(corr(32))), hi0 = std::min((double) ROT - 1, std::ceil(corr(1)));
            const double hi = lo == hi0 ? hi0 + 0.001 : hi0;
            for (int i = 0; i < ROT / 2; ++i) {
                const double ramp = std::min(1.0, std::max(0.0, (i - lo) / (hi - lo)));
                const double ex = 1.0 - ramp, extrap = inv_freq[(size_t) i];
                inv_freq[(size_t) i] = extrap / factor * (1.0 - ex) + extrap * ex;
            }
            mscale = 0.1 * std::log(factor) + 1.0;
        }
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
                    const int32_t p = pos[(size_t) t * 3 + rope.axis[i]];
                    const double f = yarn ? (double) ((float) p * (float) inv_freq[(size_t) i]) : inv_freq[(size_t) i] * p;
                    const double c = std::cos(f) * mscale, sn = std::sin(f) * mscale;
                    r[(size_t) i] = y[(size_t) i] * c - y[(size_t) i + ROT / 2] * sn;
                    r[(size_t) i + ROT / 2] = y[(size_t) i + ROT / 2] * c + y[(size_t) i] * sn;
                }
                for (int i = 0; i < HD; ++i) {
                    if (isq) eq.add(qo[((size_t) t * NH + j) * HD + i], r[(size_t) i]);
                    else ek.add(h2f(kc[(((size_t) cell0 + t) * NKV + (j - NH)) * HD + i]), r[(size_t) i]);
                }
            }
        std::printf("attn_qk_prepare%s: q max abs %.3g rel %.3g; k (fp16 cache) max abs %.3g rel %.3g\n", yarn ? " (YaRN x2)" : "",
                    eq.abs, eq.rel, ek.abs, ek.rel);
        if (eq.abs > 1e-3 || ek.rel > 2e-3) ++fails;
    }

    // ---- attn_qk_prepare into an NVFP4 cache: the queries and keys turned by the Hadamard transform, the keys' and values'
    // codes and scales by the write rule; the BF16 form writes the same cells bit for bit
    {
        std::mt19937 wr(11);   // its own draws, so the checks after it see the same inputs as before it
        const int T = 3;
        const int64_t cell0 = 10;
        const size_t cellv = (size_t) (cell0 + T) * NKV * HD;
        std::vector<float> qf((size_t) T * NH * 2 * HD), k((size_t) T * NKV * HD), v(k.size()), qn(HD), kn(HD);
        // bf16-representable inputs, so the two forms read the same values
        const auto bfr = [](float x) { return f32_from_bf16(bf16_from_f32(x)); };
        for (auto& x : qf) x = bfr(nd(wr));
        for (auto& x : k) x = bfr(nd(wr));
        for (auto& x : v) x = bfr(nd(wr));
        for (auto& x : qn) x = 1.0f + 0.1f * nd(wr);
        for (auto& x : kn) x = 1.0f + 0.1f * nd(wr);
        const std::vector<int32_t> pos = {5, 5, 5, 6, 9, 11, 7, 7, 7};
        basalt::kernels::RopeSpec rope;
        rope.rot = ROT;
        rope.theta = 1e7f;
        for (int i = 0; i < ROT / 2; ++i) rope.axis[i] = (int8_t) (i % 3 == 1 && i < 33 ? 1 : i % 3 == 2 && i < 30 ? 2 : 0);
        float *d_qf = dev<float>(qf.size()), *d_k = dev<float>(k.size()), *d_v = dev<float>(v.size()), *d_qn = dev<float>(HD),
              *d_kn = dev<float>(HD), *d_qo = dev<float>((size_t) T * NH * HD);
        int32_t* d_pos = dev<int32_t>(pos.size());
        ck(cudaMemcpy(d_qf, qf.data(), qf.size() * 4, cudaMemcpyHostToDevice), "copy");
        ck(cudaMemcpy(d_k, k.data(), k.size() * 4, cudaMemcpyHostToDevice), "copy");
        ck(cudaMemcpy(d_v, v.data(), v.size() * 4, cudaMemcpyHostToDevice), "copy");
        ck(cudaMemcpy(d_qn, qn.data(), HD * 4, cudaMemcpyHostToDevice), "copy");
        ck(cudaMemcpy(d_kn, kn.data(), HD * 4, cudaMemcpyHostToDevice), "copy");
        ck(cudaMemcpy(d_pos, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice), "copy");
        const basalt::kernels::KvInt8 f4 = alloc4(cellv), b4 = alloc4(cellv);
        for (const auto& f : {f4, b4}) {
            ck(cudaMemset(f.k, 0, cellv / 2), "memset");
            ck(cudaMemset(f.v, 0, cellv / 2), "memset");
            ck(cudaMemset(f.ks, 0, cellv / 16), "memset");
            ck(cudaMemset(f.vs, 0, cellv / 16), "memset");
        }
        basalt::kernels::attn_qk_prepare(d_qf, d_k, d_v, d_qn, d_kn, 1e-6f, rope, d_pos, T, NH, NKV, d_qo, nullptr, nullptr, cell0,
                                         nullptr, 256, true, 0, nullptr, f4);
        std::vector<float> qo((size_t) T * NH * HD);
        std::vector<uint8_t> kc4(cellv / 2), vc4(cellv / 2), ks4(cellv / 16), vs4(cellv / 16);
        ck(cudaMemcpy(qo.data(), d_qo, qo.size() * 4, cudaMemcpyDeviceToHost), "copy");
        ck(cudaMemcpy(kc4.data(), f4.k, kc4.size(), cudaMemcpyDeviceToHost), "copy");
        ck(cudaMemcpy(vc4.data(), f4.v, vc4.size(), cudaMemcpyDeviceToHost), "copy");
        ck(cudaMemcpy(ks4.data(), f4.ks, ks4.size(), cudaMemcpyDeviceToHost), "copy");
        ck(cudaMemcpy(vs4.data(), f4.vs, vs4.size(), cudaMemcpyDeviceToHost), "copy");
        std::vector<double> inv_freq(ROT / 2);
        for (int i = 0; i < ROT / 2; ++i) inv_freq[(size_t) i] = std::pow(1e7, -2.0 * i / ROT);
        Err eq;
        size_t k_scale_off = 0, k_code_off = 0, k_far = 0, v_diff = 0;
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < NH + NKV; ++j) {
                const bool isq = j < NH;
                const float* x = isq ? &qf[((size_t) t * NH + j) * 2 * HD] : &k[((size_t) t * NKV + (j - NH)) * HD];
                const float* w = isq ? qn.data() : kn.data();
                double ss = 0;
                for (int i = 0; i < HD; ++i) ss += (double) x[i] * x[i];
                const double inv = 1.0 / std::sqrt(ss / HD + 1e-6);
                std::vector<double> y(HD), r(HD), h(HD);
                for (int i = 0; i < HD; ++i) y[(size_t) i] = x[i] * inv * w[i];
                r = y;
                for (int i = 0; i < ROT / 2; ++i) {
                    const double f = inv_freq[(size_t) i] * pos[(size_t) t * 3 + rope.axis[i]];
                    r[(size_t) i] = y[(size_t) i] * std::cos(f) - y[(size_t) i + ROT / 2] * std::sin(f);
                    r[(size_t) i + ROT / 2] = y[(size_t) i + ROT / 2] * std::cos(f) + y[(size_t) i] * std::sin(f);
                }
                hadamard256(r.data(), h.data());
                if (isq) {
                    for (int i = 0; i < HD; ++i) eq.add(qo[((size_t) t * NH + j) * HD + i], h[(size_t) i]);
                    continue;
                }
                // the kernel rounds its own fp32 key, which can sit across a rounding edge from the double reference: the
                // codes are counted against the rule and held to the bound the format allows
                const size_t row = ((size_t) cell0 + t) * NKV + (size_t) (j - NH);
                for (int gi = 0; gi < HD / 16; ++gi) {
                    float amax = 0;
                    for (int i = 0; i < 16; ++i) amax = std::max(amax, (float) std::fabs(h[(size_t) gi * 16 + i]));
                    const uint8_t want = e4m3_code(std::max(amax, 6.0f * 0.001953125f) / 6.0f), got = ks4[row * 16 + (size_t) gi];
                    k_scale_off += want != got;
                    const float s = e4m3_value(got);
                    for (int i = 0; i < 16; ++i) {
                        const size_t d = (size_t) gi * 16 + i;
                        const uint8_t c = code_at(kc4, row * HD + d);
                        k_code_off += c != e2m1_code(std::min(std::max((float) h[d] / s, -6.0f), 6.0f));
                        const double a = std::fabs(h[d]) / s, half = a < 2 ? 0.25 : a < 4 ? 0.5 : 1.0;
                        if (std::fabs(e2m1_value(c) * s - h[d]) > (std::max(half, a - 6.0) + 1e-4) * s) ++k_far;
                    }
                }
            }
        // values: no transform, so the rule on the same fp32 inputs, bit for bit
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < NKV; ++j) {
                const float* x0 = &v[((size_t) t * NKV + j) * HD];
                const std::vector<float> x(x0, x0 + HD);
                std::vector<uint8_t> c, sc;
                std::vector<uint16_t> dq;
                quant4(x, c, sc, dq);
                const size_t row = ((size_t) cell0 + t) * NKV + (size_t) j;
                v_diff += std::memcmp(c.data(), &vc4[row * HD / 2], HD / 2) != 0;
                v_diff += std::memcmp(sc.data(), &vs4[row * HD / 16], HD / 16) != 0;
            }
        std::printf("attn_qk_prepare (nvfp4 cache): q (Hadamard-turned) max abs %.3g rel %.3g; k: %zu of %d scales and %zu of %d "
                    "codes off the rule on the double reference, %zu codes past the format's bound; v rows off the rule: %zu\n",
                    eq.abs, eq.rel, k_scale_off, T * NKV * HD / 16, k_code_off, T * NKV * HD, k_far, v_diff);
        if (eq.abs > 1e-3 || k_far || v_diff) ++fails;
        // a row's turned queries, codes and scales are its own: every window [s, s + w) of the rows, written into a fresh
        // cache from cell s (not the full launch's cells), gives each row the full launch's bits, alone, first or last
        {
            const basalt::kernels::KvInt8 w4 = alloc4(cellv);
            float* d_qw = dev<float>((size_t) T * NH * HD);
            const size_t qrow = (size_t) NH * HD, crow = (size_t) NKV * HD;
            std::vector<float> qw((size_t) T * NH * HD);
            std::vector<uint8_t> kw(cellv / 2), vw(cellv / 2), ksw(cellv / 16), vsw(cellv / 16);
            bool rows_same = true;
            int windows = 0;
            for (int s = 0; s < T; ++s)
                for (int w = 1; s + w <= T; ++w) {
                    if (s == 0 && w == T) continue;   // the full launch itself
                    ck(cudaMemset(w4.k, 0xab, cellv / 2), "memset");   // a cell the launch left unwritten cannot match
                    ck(cudaMemset(w4.v, 0xab, cellv / 2), "memset");
                    ck(cudaMemset(w4.ks, 0xab, cellv / 16), "memset");
                    ck(cudaMemset(w4.vs, 0xab, cellv / 16), "memset");
                    basalt::kernels::attn_qk_prepare(d_qf + (size_t) s * NH * 2 * HD, d_k + (size_t) s * crow, d_v + (size_t) s * crow,
                                                     d_qn, d_kn, 1e-6f, rope, d_pos + (size_t) s * 3, w, NH, NKV, d_qw, nullptr,
                                                     nullptr, s, nullptr, 256, true, 0, nullptr, w4);
                    ck(cudaMemcpy(qw.data(), d_qw, (size_t) w * qrow * 4, cudaMemcpyDeviceToHost), "copy");
                    ck(cudaMemcpy(kw.data(), w4.k, kw.size(), cudaMemcpyDeviceToHost), "copy");
                    ck(cudaMemcpy(vw.data(), w4.v, vw.size(), cudaMemcpyDeviceToHost), "copy");
                    ck(cudaMemcpy(ksw.data(), w4.ks, ksw.size(), cudaMemcpyDeviceToHost), "copy");
                    ck(cudaMemcpy(vsw.data(), w4.vs, vsw.size(), cudaMemcpyDeviceToHost), "copy");
                    const size_t at = (size_t) s * crow, full = ((size_t) cell0 + s) * crow, n = (size_t) w * crow;
                    rows_same = rows_same && std::memcmp(qw.data(), &qo[(size_t) s * qrow], (size_t) w * qrow * 4) == 0 &&
                                std::memcmp(&kw[at / 2], &kc4[full / 2], n / 2) == 0 &&
                                std::memcmp(&vw[at / 2], &vc4[full / 2], n / 2) == 0 &&
                                std::memcmp(&ksw[at / 16], &ks4[full / 16], n / 16) == 0 &&
                                std::memcmp(&vsw[at / 16], &vs4[full / 16], n / 16) == 0;
                    ++windows;
                }
            std::printf("attn_qk_prepare (nvfp4 cache): each row's q, codes and scales bitwise the full launch's in %d windows "
                        "of 1-%d rows at other cells: %s\n", windows, T - 1, rows_same ? "yes" : "NO");
            if (!rows_same) ++fails;
            free4(w4);
            cudaFree(d_qw);
        }
        // the BF16 form on the same values
        std::vector<uint16_t> qb(qf.size()), kb(k.size()), vb(v.size());
        for (size_t i = 0; i < qf.size(); ++i) qb[i] = bf16_from_f32(qf[i]);
        for (size_t i = 0; i < k.size(); ++i) kb[i] = bf16_from_f32(k[i]);
        for (size_t i = 0; i < v.size(); ++i) vb[i] = bf16_from_f32(v[i]);
        uint16_t *d_qb = dev<uint16_t>(qb.size()), *d_kb = dev<uint16_t>(kb.size()), *d_vb = dev<uint16_t>(vb.size()),
                 *d_qob = dev<uint16_t>((size_t) T * NH * HD);
        ck(cudaMemcpy(d_qb, qb.data(), qb.size() * 2, cudaMemcpyHostToDevice), "copy");
        ck(cudaMemcpy(d_kb, kb.data(), kb.size() * 2, cudaMemcpyHostToDevice), "copy");
        ck(cudaMemcpy(d_vb, vb.data(), vb.size() * 2, cudaMemcpyHostToDevice), "copy");
        basalt::kernels::attn_qk_prepare_bf16(d_qb, d_kb, d_vb, d_qn, d_kn, 1e-6f, rope, d_pos, T, NH, NKV, d_qob, cell0, nullptr, b4,
                                              nullptr, nullptr);
        std::vector<uint8_t> a(cellv / 2 + cellv / 16), bb(a.size());
        bool same = true;
        for (int t = 0; t < 2; ++t) {
            ck(cudaMemcpy(a.data(), t ? f4.v : f4.k, cellv / 2, cudaMemcpyDeviceToHost), "copy");
            ck(cudaMemcpy(a.data() + cellv / 2, t ? f4.vs : f4.ks, cellv / 16, cudaMemcpyDeviceToHost), "copy");
            ck(cudaMemcpy(bb.data(), t ? b4.v : b4.k, cellv / 2, cudaMemcpyDeviceToHost), "copy");
            ck(cudaMemcpy(bb.data() + cellv / 2, t ? b4.vs : b4.ks, cellv / 16, cudaMemcpyDeviceToHost), "copy");
            same = same && a == bb;
        }
        std::printf("attn_qk_prepare_bf16 (nvfp4 cache): the same codes and scales as the fp32 form: %s\n", same ? "yes" : "NO");
        if (!same) ++fails;
        free4(f4);
        free4(b4);
        cudaFree(d_qf); cudaFree(d_k); cudaFree(d_v); cudaFree(d_qn); cudaFree(d_kn); cudaFree(d_qo); cudaFree(d_pos);
        cudaFree(d_qb); cudaFree(d_kb); cudaFree(d_vb); cudaFree(d_qob);
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
    constexpr int TW = basalt::kernels::kGqaMaxWindow;
    std::vector<float> single((size_t) TW * NH * HD), got((size_t) TW * NH * HD);
    Err ed;
    bool bitwise = true;
    for (int T = 1; T <= TW; ++T) {
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
    std::printf("gqa_decode_attn: max abs %.3g rel %.3g vs double; the last row bitwise equal across widths 1-%d: %s\n", ed.abs,
                ed.rel, TW, bitwise ? "yes" : "NO");
    if (ed.abs > 1e-3 || !bitwise) ++fails;   // fp16 q, K, V and p on the tensor cores
    {
        // a captured window reads its first cell on the device and loops its blocks over the chunks in use: bitwise the
        // host-position launch, past the grid's first round of chunks
        const int64_t L2 = 12000, cap2 = 65536, at = L2 - 8;
        std::vector<uint16_t> k2((size_t) L2 * NKV * HD), v2(k2.size());
        for (auto& x : k2) x = basalt::kernels::f16_from_f32(nd(rng));
        for (auto& x : v2) x = basalt::kernels::f16_from_f32(nd(rng));
        uint16_t *d_k2 = dev<uint16_t>((size_t) cap2 * NKV * HD), *d_v2 = dev<uint16_t>((size_t) cap2 * NKV * HD);
        float* d_s2 = dev<float>(basalt::kernels::gqa_decode_scratch_floats(cap2, 16, NH));
        int32_t* d_at = dev<int32_t>(2);
        const int32_t at2[2] = {(int32_t) at, 0};
        ck(cudaMemcpy(d_k2, k2.data(), k2.size() * 2, cudaMemcpyHostToDevice), "copy");
        ck(cudaMemcpy(d_v2, v2.data(), v2.size() * 2, cudaMemcpyHostToDevice), "copy");
        ck(cudaMemcpy(d_at, at2, sizeof at2, cudaMemcpyHostToDevice), "copy");
        std::vector<float> h1((size_t) 8 * NH * HD), h2(h1.size());
        basalt::kernels::gqa_decode_attn(d_q, d_k2, d_v2, NH, NKV, at, 8, d_s2, cap2, d_out, nullptr);
        ck(cudaMemcpy(h1.data(), d_out, h1.size() * 4, cudaMemcpyDeviceToHost), "copy");
        basalt::kernels::gqa_decode_attn(d_q, d_k2, d_v2, NH, NKV, at, 8, d_s2, cap2, d_out, nullptr, d_at);
        ck(cudaMemcpy(h2.data(), d_out, h2.size() * 4, cudaMemcpyDeviceToHost), "copy");
        const bool same = std::memcmp(h1.data(), h2.data(), h1.size() * 4) == 0;
        std::printf("gqa_decode_attn: device first cell at %lld of %lld cells bitwise the host launch: %s\n", (long long) at,
                    (long long) cap2, same ? "yes" : "NO");
        if (!same) ++fails;
        {   // two sequences' windows in one launch (a batch window's segments), each bitwise its own launch
            const int TA = 5, TB = 7;
            std::vector<float> solo((size_t) (TA + TB) * NH * HD), seg(solo.size());
            basalt::kernels::gqa_decode_attn(d_q, d_k2, d_v2, NH, NKV, 3000, TA, d_s2, cap2, d_out, nullptr);
            ck(cudaMemcpy(solo.data(), d_out, (size_t) TA * NH * HD * 4, cudaMemcpyDeviceToHost), "copy");
            basalt::kernels::gqa_decode_attn(d_q + (size_t) TA * NH * HD, d_k2, d_v2, NH, NKV, at - TB, TB, d_s2, cap2, d_out, nullptr);
            ck(cudaMemcpy(solo.data() + (size_t) TA * NH * HD, d_out, (size_t) TB * NH * HD * 4, cudaMemcpyDeviceToHost), "copy");
            const int32_t atb[2] = {(int32_t) (at - TB), 0};
            ck(cudaMemcpy(d_at, atb, sizeof atb, cudaMemcpyHostToDevice), "copy");
            basalt::kernels::GqaSeg sg[2];
            sg[0].T = TA; sg[0].k_cache = d_k2; sg[0].v_cache = d_v2; sg[0].pos0 = 3000;
            sg[1].T = TB; sg[1].k_cache = d_k2; sg[1].v_cache = d_v2; sg[1].pos0 = at - TB; sg[1].pos_dev = d_at;
            basalt::kernels::gqa_decode_attn_segs(d_q, NH, NKV, sg, 2, d_s2, cap2, d_out, nullptr);
            ck(cudaMemcpy(seg.data(), d_out, seg.size() * 4, cudaMemcpyDeviceToHost), "copy");
            const bool same_seg = std::memcmp(solo.data(), seg.data(), seg.size() * 4) == 0;
            std::printf("gqa_decode_attn_segs: windows of %d and %d rows in one launch bitwise their own launches: %s\n", TA, TB,
                        same_seg ? "yes" : "NO");
            if (!same_seg) ++fails;
            ck(cudaMemcpy(d_at, at2, sizeof at2, cudaMemcpyHostToDevice), "copy");
        }
        // its cost at a short context in a large capacity, against the host launch's grid of the chunks in use
        const int32_t near_at[2] = {2999, 0};
        ck(cudaMemcpy(d_at, near_at, sizeof near_at, cudaMemcpyHostToDevice), "copy");
        double us[2];
        for (int dv = 0; dv < 2; ++dv) {
            for (int i = 0; i < 5; ++i)
                basalt::kernels::gqa_decode_attn(d_q, d_k2, d_v2, NH, NKV, 2999, 1, d_s2, cap2, d_out, nullptr, dv ? d_at : nullptr);
            cudaDeviceSynchronize();
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 200; ++i)
                basalt::kernels::gqa_decode_attn(d_q, d_k2, d_v2, NH, NKV, 2999, 1, d_s2, cap2, d_out, nullptr, dv ? d_at : nullptr);
            cudaDeviceSynchronize();
            us[dv] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 200;
        }
        std::printf("bench: decode 1 row x 3000 cells in %lld: %.1f us host first cell, %.1f us device first cell\n",
                    (long long) cap2, us[0], us[1]);
        for (int T : {1, 8, 16}) {   // a window of T rows near 12K cells
            for (int i = 0; i < 5; ++i) basalt::kernels::gqa_decode_attn(d_q, d_k2, d_v2, NH, NKV, at - 8, T, d_s2, cap2, d_out, nullptr);
            cudaDeviceSynchronize();
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 100; ++i) basalt::kernels::gqa_decode_attn(d_q, d_k2, d_v2, NH, NKV, at - 8, T, d_s2, cap2, d_out, nullptr);
            cudaDeviceSynchronize();
            std::printf("bench: decode %d rows x %lld cells: %.1f us\n", T, (long long) at, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 100);
        }
        cudaFree(d_k2); cudaFree(d_v2); cudaFree(d_s2); cudaFree(d_at);
    }
    // prompt chunk
    basalt::kernels::gqa_prompt_attn(d_q, d_kc, d_vc, NH, NKV, p0, TP, d_out, nullptr);
    std::vector<float> pr(q.size());
    ck(cudaMemcpy(pr.data(), d_out, pr.size() * 4, cudaMemcpyDeviceToHost), "copy");
    Err ep;
    for (size_t i = 0; i < rows.size(); ++i)
        for (int j = 0; j < NH * HD; ++j) ep.add(pr[(size_t) rows[i] * NH * HD + j], want[i][(size_t) j]);
    std::printf("gqa_prompt_attn: max abs %.3g rel %.3g vs double (fp16 q and p)\n", ep.abs, ep.rel);
    if (ep.abs > 5e-3) ++fails;

    // ---- int8 cache: the same cells as codes and per-64 scales (the write kernel's rule), against the reference on the
    // dequantized values
    {
        const size_t rows = (size_t) max_cells * NKV;
        std::vector<int8_t> k8(rows * HD), v8(rows * HD);
        std::vector<uint16_t> ks(rows * HD / 64), vs(rows * HD / 64), kd(kc.size()), vd(vc.size());
        auto quant = [&](const std::vector<uint16_t>& src, std::vector<int8_t>& q, std::vector<uint16_t>& sc, std::vector<uint16_t>& deq) {
            for (size_t g = 0; g < rows * HD / 64; ++g) {
                float amax = 0;
                for (int i = 0; i < 64; ++i) amax = std::max(amax, std::fabs(h2f(src[g * 64 + i])));
                const uint16_t h = basalt::kernels::f16_from_f32(amax / 127.0f);
                const float s16 = h2f(h);
                sc[g] = h;
                for (int i = 0; i < 64; ++i) {
                    const float c = s16 > 0 ? std::rint(h2f(src[g * 64 + i]) / s16) : 0.0f;
                    q[g * 64 + i] = (int8_t) std::max(-127.0f, std::min(127.0f, c));
                    deq[g * 64 + i] = basalt::kernels::f16_from_f32(q[g * 64 + i] * s16);
                }
            }
        };
        quant(kc, k8, ks, kd);
        quant(vc, v8, vs, vd);
        basalt::kernels::KvInt8 q8;
        ck(cudaMalloc(&q8.k, k8.size()), "malloc");
        ck(cudaMalloc(&q8.v, v8.size()), "malloc");
        ck(cudaMalloc(&q8.ks, ks.size() * 2), "malloc");
        ck(cudaMalloc(&q8.vs, vs.size() * 2), "malloc");
        cudaMemcpy(q8.k, k8.data(), k8.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(q8.v, v8.data(), v8.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(q8.ks, ks.data(), ks.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(q8.vs, vs.data(), vs.size() * 2, cudaMemcpyHostToDevice);
        // the prompt kernel rounds the queries to int8 per 64 dims (amax / 127) for its int8 QK: references on the
        // queries as given and on those
        auto ref8 = [&](const float* qrow, int64_t visible, double* w, double* wq) {
            ref_attn(qrow, kd, vd, visible, w);
            std::vector<float> qq(qrow, qrow + (size_t) NH * HD);
            for (size_t g0 = 0; g0 < qq.size(); g0 += 64) {
                float amax = 0;
                for (int i = 0; i < 64; ++i) amax = std::max(amax, std::fabs(qq[g0 + i]));
                for (int i = 0; i < 64; ++i)
                    qq[g0 + i] = amax > 0 ? (float) std::nearbyint(qq[g0 + i] * (127.0f / amax)) * (amax / 127.0f) : 0.0f;
            }
            ref_attn(qq.data(), kd, vd, visible, wq);
        };
        std::vector<double> w8((size_t) NH * HD), w8q((size_t) NH * HD);
        ref8(&q[(size_t) (TP - 1) * NH * HD], p0 + TP, w8.data(), w8q.data());
        basalt::kernels::gqa_decode_attn(d_q + (size_t) (TP - 1) * NH * HD, nullptr, nullptr, NH, NKV, p0 + TP - 1, 1, d_scr,
                                         max_cells, d_out, nullptr, nullptr, q8);
        std::vector<float> g8((size_t) NH * HD);
        ck(cudaMemcpy(g8.data(), d_out, g8.size() * 4, cudaMemcpyDeviceToHost), "copy");
        Err e8;
        for (int j = 0; j < NH * HD; ++j) e8.add(g8[(size_t) j], w8[(size_t) j]);
        std::printf("int8 cache: decode max abs %.3g rel %.3g vs double on the dequantized cells\n", e8.abs, e8.rel);
        // decode reads code x scale in fp32, the reference its fp16 rounding
        if (e8.abs > 1e-3) ++fails;

        // prompt chunks at the end of the L cells: 300 rows (narrow: 4-warp blocks), the same with split scratch (the
        // cells split over more blocks and merged), and 2048 rows (8-warp blocks), each at rows on and around the block
        // and key-tile edges
        const int TB = 2048;
        std::vector<float> qb((size_t) TB * NH * HD);
        for (auto& x : qb) x = 2.0f * nd(rng);
        float *d_qb = dev<float>(qb.size()), *d_ob = dev<float>(qb.size());
        ck(cudaMemcpy(d_qb, qb.data(), qb.size() * 4, cudaMemcpyHostToDevice), "copy");
        const size_t split_floats = basalt::kernels::gqa_prompt_split_floats(TP, NH, 8);
        float* d_split = dev<float>(split_floats);
        for (int cfg = 0; cfg < 3; ++cfg) {
            const int T = cfg < 2 ? TP : TB;
            const int64_t pp = L - T;
            const float* qh = cfg < 2 ? q.data() : qb.data();
            basalt::kernels::gqa_prompt_attn(cfg < 2 ? d_q : d_qb, nullptr, nullptr, NH, NKV, pp, T, d_ob, nullptr, q8,
                                             cfg == 1 ? d_split : nullptr, cfg == 1 ? split_floats : 0);
            std::vector<float> po((size_t) T * NH * HD);
            ck(cudaMemcpy(po.data(), d_ob, po.size() * 4, cudaMemcpyDeviceToHost), "copy");
            Err p8, p8q;
            for (const int r : {0, 1, 57, 63, 64, 65, 127, 128, 255, T - 2, T - 1, 1000, 2047}) {
                if (r >= T) continue;
                ref8(&qh[(size_t) r * NH * HD], pp + r + 1, w8.data(), w8q.data());
                for (int j = 0; j < NH * HD; ++j) {
                    p8.add(po[(size_t) r * NH * HD + j], w8[(size_t) j]);
                    p8q.add(po[(size_t) r * NH * HD + j], w8q[(size_t) j]);
                }
            }
            static const char* what[3] = {"300 rows", "300 rows split", "2048 rows"};
            std::printf("int8 cache: prompt %s max abs %.3g (%.3g against int8-rounded queries) vs double on the dequantized "
                        "cells\n", what[cfg], p8.abs, p8q.abs);
            // the prompt's QK reads int8 queries; its P V rounds each probability times its cell's scale to u8 per row per
            // 64-cell tile (half a step of 1/255 of the tile's largest product each), summed exactly in int32
            if (p8q.abs > 2e-2 || p8.abs > 3e-2) ++fails;
        }
        cudaFree(d_qb);
        cudaFree(d_ob);
        cudaFree(d_split);
    }

    // ---- NVFP4 cache: the same cells as codes and per-16 scales (the write rule), the decode kernel against the reference
    // on the decoded cells, a row's bits the same at every window width and beside another window in one launch; a prompt
    // chunk through the int8 copy (kv4_to_int8) and the int8 kernel
    {
        const size_t n = (size_t) max_cells * NKV * HD;
        std::vector<uint8_t> c4[2], s4[2];
        std::vector<uint16_t> d4[2];
        for (int t = 0; t < 2; ++t) {
            std::vector<float> x(n);
            for (size_t i = 0; i < n; ++i) x[i] = h2f((t ? vc : kc)[i]);
            quant4(x, c4[t], s4[t], d4[t]);
        }
        const basalt::kernels::KvInt8 f4 = alloc4(n);
        ck(cudaMemcpy(f4.k, c4[0].data(), n / 2, cudaMemcpyHostToDevice), "copy");
        ck(cudaMemcpy(f4.v, c4[1].data(), n / 2, cudaMemcpyHostToDevice), "copy");
        ck(cudaMemcpy(f4.ks, s4[0].data(), n / 16, cudaMemcpyHostToDevice), "copy");
        ck(cudaMemcpy(f4.vs, s4[1].data(), n / 16, cudaMemcpyHostToDevice), "copy");
        std::vector<double> w4((size_t) NH * HD);
        ref_attn(&q[(size_t) (TP - 1) * NH * HD], d4[0], d4[1], p0 + TP, w4.data());
        std::vector<float> one((size_t) NH * HD), win((size_t) TW * NH * HD);
        bool same = true;
        Err e4;
        for (int T = 1; T <= TW; ++T) {
            basalt::kernels::gqa_decode_attn(d_q + (size_t) (TP - T) * NH * HD, nullptr, nullptr, NH, NKV, p0 + TP - T, T, d_scr,
                                             max_cells, d_out, nullptr, nullptr, f4);
            ck(cudaMemcpy(win.data(), d_out, (size_t) T * NH * HD * 4, cudaMemcpyDeviceToHost), "copy");
            const float* last = &win[(size_t) (T - 1) * NH * HD];
            if (T == 1) {
                std::memcpy(one.data(), last, one.size() * 4);
                for (int j = 0; j < NH * HD; ++j) e4.add(one[(size_t) j], w4[(size_t) j]);
            } else if (std::memcmp(one.data(), last, one.size() * 4) != 0) {
                same = false;
            }
        }
        std::printf("nvfp4 cache: decode max abs %.3g rel %.3g vs double on the decoded cells; the last row bitwise equal across "
                    "widths 1-%d: %s\n", e4.abs, e4.rel, TW, same ? "yes" : "NO");
        if (e4.abs > 1e-3 || !same) ++fails;
        {
            const int TA = 5, TB = 7;
            std::vector<float> solo((size_t) (TA + TB) * NH * HD), seg(solo.size());
            basalt::kernels::gqa_decode_attn(d_q, nullptr, nullptr, NH, NKV, 1000, TA, d_scr, max_cells, d_out, nullptr, nullptr, f4);
            ck(cudaMemcpy(solo.data(), d_out, (size_t) TA * NH * HD * 4, cudaMemcpyDeviceToHost), "copy");
            basalt::kernels::gqa_decode_attn(d_q + (size_t) TA * NH * HD, nullptr, nullptr, NH, NKV, 2500, TB, d_scr, max_cells, d_out,
                                             nullptr, nullptr, f4);
            ck(cudaMemcpy(solo.data() + (size_t) TA * NH * HD, d_out, (size_t) TB * NH * HD * 4, cudaMemcpyDeviceToHost), "copy");
            basalt::kernels::GqaSeg sg[2];
            sg[0].T = TA;
            sg[0].pos0 = 1000;
            sg[0].q8 = f4;
            sg[1] = sg[0];
            sg[1].T = TB;
            sg[1].pos0 = 2500;
            basalt::kernels::gqa_decode_attn_segs(d_q, NH, NKV, sg, 2, d_scr, max_cells, d_out, nullptr);
            ck(cudaMemcpy(seg.data(), d_out, seg.size() * 4, cudaMemcpyDeviceToHost), "copy");
            const bool same_seg = std::memcmp(solo.data(), seg.data(), seg.size() * 4) == 0;
            std::printf("nvfp4 cache: windows of %d and %d rows in one launch bitwise their own launches: %s\n", TA, TB,
                        same_seg ? "yes" : "NO");
            if (!same_seg) ++fails;
        }
        {
            basalt::kernels::KvInt8 c8;
            ck(cudaMalloc(&c8.k, n), "malloc");
            ck(cudaMalloc(&c8.v, n), "malloc");
            ck(cudaMalloc(&c8.ks, n / 64 * 2), "malloc");
            ck(cudaMalloc(&c8.vs, n / 64 * 2), "malloc");
            basalt::kernels::kv4_to_int8(f4, c8, L, NKV, nullptr);
            basalt::kernels::gqa_prompt_attn(d_q, nullptr, nullptr, NH, NKV, p0, TP, d_out, nullptr, c8);
            std::vector<float> po(q.size());
            ck(cudaMemcpy(po.data(), d_out, po.size() * 4, cudaMemcpyDeviceToHost), "copy");
            Err p4;
            std::vector<double> wr((size_t) NH * HD);
            for (const int r : {0, 1, 57, 63, 64, 65, 127, 128, 255, TP - 2, TP - 1}) {
                ref_attn(&q[(size_t) r * NH * HD], d4[0], d4[1], p0 + r + 1, wr.data());
                for (int j = 0; j < NH * HD; ++j) p4.add(po[(size_t) r * NH * HD + j], wr[(size_t) j]);
            }
            std::printf("nvfp4 cache: prompt 300 rows through the int8 copy max abs %.3g rel %.3g vs double on the decoded cells\n",
                        p4.abs, p4.rel);
            // the int8 prompt kernel's own rounding (int8 queries, u8 probabilities) on top of the copy's int8 rounding
            if (p4.abs > 3e-2) ++fails;
            cudaFree(c8.k); cudaFree(c8.v); cudaFree(c8.ks); cudaFree(c8.vs);
        }
        free4(f4);
    }

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
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {   // a 2048-row chunk at the end of 4K and 16K caches, fp16 and int8
        for (const int64_t cells : {4096, 16384}) {
            const int64_t T2 = 2048, pp = cells - T2;
            uint16_t* bk = dev<uint16_t>((size_t) cells * NKV * HD);
            uint16_t* bv = dev<uint16_t>((size_t) cells * NKV * HD);
            float* bq = dev<float>((size_t) T2 * NH * HD);
            float* bo = dev<float>((size_t) T2 * NH * HD);
            basalt::kernels::KvInt8 b8;
            cudaMalloc(&b8.k, (size_t) cells * NKV * HD);
            cudaMalloc(&b8.v, (size_t) cells * NKV * HD);
            cudaMalloc(&b8.ks, (size_t) cells * NKV * HD / 64 * 2);
            cudaMalloc(&b8.vs, (size_t) cells * NKV * HD / 64 * 2);
            cudaMemset(bk, 0x2A, (size_t) cells * NKV * HD * 2);
            cudaMemset(bv, 0x2A, (size_t) cells * NKV * HD * 2);
            cudaMemset(bq, 0x3C, (size_t) T2 * NH * HD * 4);
            cudaMemset(b8.k, 0x11, (size_t) cells * NKV * HD);
            cudaMemset(b8.v, 0x11, (size_t) cells * NKV * HD);
            cudaMemset(b8.ks, 0x20, (size_t) cells * NKV * HD / 64 * 2);
            cudaMemset(b8.vs, 0x20, (size_t) cells * NKV * HD / 64 * 2);
            double fl = 0;
            for (int64_t t = 0; t < T2; ++t) fl += 4.0 * NH * HD * (double) (pp + t + 1);
            // fp16, int8, int8 with scratch for three key splits
            const size_t sf = basalt::kernels::gqa_prompt_split_floats(T2, NH, 3);
            float* bs = dev<float>(sf);
            for (int run = 0; run < 3; ++run) {
                const basalt::kernels::KvInt8 kv = run ? b8 : basalt::kernels::KvInt8{};
                float* scr = run == 2 ? bs : nullptr;
                basalt::kernels::gqa_prompt_attn(bq, bk, bv, NH, NKV, pp, T2, bo, nullptr, kv, scr, sf);
                cudaEventRecord(a);
                for (int i = 0; i < 5; ++i) basalt::kernels::gqa_prompt_attn(bq, bk, bv, NH, NKV, pp, T2, bo, nullptr, kv, scr, sf);
                cudaEventRecord(b);
                cudaEventSynchronize(b);
                cudaEventElapsedTime(&ms, a, b);
                std::printf("bench: prompt chunk 2048 rows at %lld cells, %s cache: %.2f ms (%.1f TFLOP/s)\n", (long long) cells,
                            run == 0 ? "fp16" : run == 1 ? "int8" : "int8 split scratch", ms / 5, fl / (ms / 5) / 1e9);
            }
            cudaFree(bs);
            cudaFree(bk); cudaFree(bv); cudaFree(bq); cudaFree(bo); cudaFree(b8.k); cudaFree(b8.v); cudaFree(b8.ks); cudaFree(b8.vs);
        }
    }
    // ---- an NVFP4 cache's int8 copy (kv4_to_int8): random codes and scales, every code and scale bit for bit the int8
    // write rule on the decoded values
    {
        std::mt19937 fr(5);
        const int64_t cells = 1000;
        const size_t n = (size_t) cells * NKV * HD;
        std::vector<uint8_t> c4[2] = {std::vector<uint8_t>(n / 2), std::vector<uint8_t>(n / 2)};
        std::vector<uint8_t> s4[2] = {std::vector<uint8_t>(n / 16), std::vector<uint8_t>(n / 16)};
        for (int t = 0; t < 2; ++t) {
            for (auto& x : c4[t]) x = (uint8_t) fr();
            for (auto& x : s4[t]) x = (uint8_t) (fr() % 0x7f);   // finite E4M3, sign clear, subnormals and zero included
        }
        const auto e4m3 = [](uint8_t b) {
            const int e = (b >> 3) & 15, m = b & 7;
            return e ? std::ldexp(1.0f + m / 8.0f, e - 7) : std::ldexp(m / 8.0f, -6);
        };
        static const float e2m1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
        basalt::kernels::KvInt8 f4, o8;
        f4.fp4 = true;
        ck(cudaMalloc(&f4.k, n / 2), "malloc");
        ck(cudaMalloc(&f4.v, n / 2), "malloc");
        ck(cudaMalloc(&f4.ks, n / 16), "malloc");
        ck(cudaMalloc(&f4.vs, n / 16), "malloc");
        ck(cudaMalloc(&o8.k, n), "malloc");
        ck(cudaMalloc(&o8.v, n), "malloc");
        ck(cudaMalloc(&o8.ks, n / 64 * 2), "malloc");
        ck(cudaMalloc(&o8.vs, n / 64 * 2), "malloc");
        ck(cudaMemcpy(f4.k, c4[0].data(), n / 2, cudaMemcpyHostToDevice), "copy");
        ck(cudaMemcpy(f4.v, c4[1].data(), n / 2, cudaMemcpyHostToDevice), "copy");
        ck(cudaMemcpy(f4.ks, s4[0].data(), n / 16, cudaMemcpyHostToDevice), "copy");
        ck(cudaMemcpy(f4.vs, s4[1].data(), n / 16, cudaMemcpyHostToDevice), "copy");
        basalt::kernels::kv4_to_int8(f4, o8, cells, NKV, nullptr);
        std::vector<int8_t> g8(n);
        std::vector<uint16_t> gs(n / 64);
        size_t bad = 0;
        for (int t = 0; t < 2; ++t) {
            ck(cudaMemcpy(g8.data(), t ? o8.v : o8.k, n, cudaMemcpyDeviceToHost), "copy");
            ck(cudaMemcpy(gs.data(), t ? o8.vs : o8.ks, n / 64 * 2, cudaMemcpyDeviceToHost), "copy");
            for (size_t g = 0; g < n / 64; ++g) {
                float x[64], amax = 0;
                for (int i = 0; i < 64; ++i) {
                    const size_t at = g * 64 + i;
                    const uint8_t nib = (uint8_t) ((c4[t][at / 2] >> (4 * (at & 1))) & 15);
                    x[i] = (nib & 8 ? -1.0f : 1.0f) * e2m1[nib & 7] * e4m3(s4[t][at / 16]);
                    amax = std::max(amax, std::fabs(x[i]));
                }
                const uint16_t h = basalt::kernels::f16_from_f32(amax / 127.0f);
                const float sc = h2f(h);
                bad += gs[g] != h;
                for (int i = 0; i < 64; ++i) {
                    const float r = sc > 0 ? std::max(-127.0f, std::min(127.0f, std::nearbyint(x[i] / sc))) : 0.0f;
                    bad += g8[g * 64 + i] != (int8_t) r;
                }
            }
        }
        std::printf("kv4_to_int8: %zu of %zu codes and scales differ from the int8 rule on the decoded values\n", bad,
                    2 * (n + n / 64));
        if (bad) ++fails;
        if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {   // a 172K-cell layer's copy
            const int64_t big = 172 * 1024;
            basalt::kernels::KvInt8 b4, b8;
            b4.fp4 = true;
            const size_t nb = (size_t) big * NKV * HD;
            cudaMalloc(&b4.k, nb / 2); cudaMalloc(&b4.v, nb / 2); cudaMalloc(&b4.ks, nb / 16); cudaMalloc(&b4.vs, nb / 16);
            cudaMalloc(&b8.k, nb); cudaMalloc(&b8.v, nb); cudaMalloc(&b8.ks, nb / 32); cudaMalloc(&b8.vs, nb / 32);
            cudaMemset(b4.k, 0x35, nb / 2); cudaMemset(b4.v, 0x35, nb / 2); cudaMemset(b4.ks, 0x30, nb / 16); cudaMemset(b4.vs, 0x30, nb / 16);
            basalt::kernels::kv4_to_int8(b4, b8, big, NKV, nullptr);
            cudaEvent_t a2, b2;
            cudaEventCreate(&a2);
            cudaEventCreate(&b2);
            cudaEventRecord(a2);
            for (int i = 0; i < 10; ++i) basalt::kernels::kv4_to_int8(b4, b8, big, NKV, nullptr);
            cudaEventRecord(b2);
            cudaEventSynchronize(b2);
            float ms2 = 0;
            cudaEventElapsedTime(&ms2, a2, b2);
            const double bytes2 = (double) nb * 2 * (0.5 + 1.0 / 16 + 1.0 + 1.0 / 32);
            std::printf("bench: kv4_to_int8 %lld cells: %.3f ms (%.0f GB/s)\n", (long long) big, ms2 / 10, bytes2 / (ms2 / 10) / 1e6);
            // decode windows at the end of the same cells, int8 against NVFP4
            constexpr int TWB = basalt::kernels::kGqaMaxWindow;
            float* dsb = dev<float>(basalt::kernels::gqa_decode_scratch_floats(big, TWB, NH));
            float *dqb = dev<float>((size_t) TWB * NH * HD), *dob = dev<float>((size_t) TWB * NH * HD);
            cudaMemset(dqb, 0x3C, (size_t) TWB * NH * HD * 4);
            for (int T : {1, 16})
                for (int fmt = 0; fmt < 2; ++fmt) {
                    const basalt::kernels::KvInt8 kv = fmt ? b4 : b8;
                    const auto run = [&] {
                        basalt::kernels::gqa_decode_attn(dqb, nullptr, nullptr, NH, NKV, big - T, T, dsb, big, dob, nullptr, nullptr, kv);
                    };
                    for (int i = 0; i < 3; ++i) run();
                    cudaEventRecord(a2);
                    for (int i = 0; i < 50; ++i) run();
                    cudaEventRecord(b2);
                    cudaEventSynchronize(b2);
                    cudaEventElapsedTime(&ms2, a2, b2);
                    const double kvb = (double) big * NKV * HD * 2 * (fmt ? 0.5 + 1.0 / 16 : 1.0 + 1.0 / 32);
                    std::printf("bench: decode %d rows x %lld cells, %s cache: %.1f us (%.0f GB/s of K/V)\n", T, (long long) big,
                                fmt ? "nvfp4" : "int8", 1000.0 * ms2 / 50, kvb / (ms2 / 50) / 1e6);
                }
            cudaFree(dsb); cudaFree(dqb); cudaFree(dob);
            cudaFree(b4.k); cudaFree(b4.v); cudaFree(b4.ks); cudaFree(b4.vs);
            cudaFree(b8.k); cudaFree(b8.v); cudaFree(b8.ks); cudaFree(b8.vs);
        }
        cudaFree(f4.k); cudaFree(f4.v); cudaFree(f4.ks); cudaFree(f4.vs);
        cudaFree(o8.k); cudaFree(o8.v); cudaFree(o8.ks); cudaFree(o8.vs);
    }
    // ---- the decode kernel's bits are frozen: random fp16 and int8 caches, many (pos0, T), a device pos0, two windows in one launch.
    // The hashes below are those of the kernel before its next-chunk fetch; a change that must not move a bit leaves them as they are.
    {
        std::mt19937 hr(99);
        std::normal_distribution<float> hn(0.f, 1.f);
        const int64_t cap = 4096;
        const size_t n = (size_t) cap * NKV * HD;
        std::vector<uint16_t> hk(n), hv(n);
        for (size_t i = 0; i < n; ++i) {
            hk[i] = basalt::kernels::f16_from_f32(hn(hr));
            hv[i] = basalt::kernels::f16_from_f32(hn(hr));
        }
        std::vector<int8_t> k8(n), v8(n);
        std::vector<uint16_t> ks(n / 64), vs(n / 64);
        const auto quant = [&](const std::vector<uint16_t>& src, std::vector<int8_t>& q8v, std::vector<uint16_t>& sc) {
            for (size_t g = 0; g < n / 64; ++g) {
                float amax = 0;
                for (int i = 0; i < 64; ++i) amax = std::max(amax, std::fabs(h2f(src[g * 64 + i])));
                sc[g] = basalt::kernels::f16_from_f32(amax / 127.0f);
                const float s16 = h2f(sc[g]);
                for (int i = 0; i < 64; ++i) q8v[g * 64 + i] = (int8_t) (s16 > 0 ? std::lrint(h2f(src[g * 64 + i]) / s16) : 0);
            }
        };
        quant(hk, k8, ks);
        quant(hv, v8, vs);
        uint16_t *dk = dev<uint16_t>(n), *dv = dev<uint16_t>(n);
        ck(cudaMemcpy(dk, hk.data(), n * 2, cudaMemcpyHostToDevice), "copy");
        ck(cudaMemcpy(dv, hv.data(), n * 2, cudaMemcpyHostToDevice), "copy");
        basalt::kernels::KvInt8 c8;
        ck(cudaMalloc(&c8.k, n), "malloc");
        ck(cudaMalloc(&c8.v, n), "malloc");
        ck(cudaMalloc(&c8.ks, ks.size() * 2), "malloc");
        ck(cudaMalloc(&c8.vs, vs.size() * 2), "malloc");
        ck(cudaMemcpy(c8.k, k8.data(), n, cudaMemcpyHostToDevice), "copy");
        ck(cudaMemcpy(c8.v, v8.data(), n, cudaMemcpyHostToDevice), "copy");
        ck(cudaMemcpy(c8.ks, ks.data(), ks.size() * 2, cudaMemcpyHostToDevice), "copy");
        ck(cudaMemcpy(c8.vs, vs.data(), vs.size() * 2, cudaMemcpyHostToDevice), "copy");
        constexpr int TW2 = basalt::kernels::kGqaMaxWindow;
        std::vector<float> hq((size_t) TW2 * NH * HD);
        for (auto& x : hq) x = 2.0f * hn(hr);
        float *dq = dev<float>(hq.size()), *dout = dev<float>(hq.size());
        float* dscr = dev<float>(basalt::kernels::gqa_decode_scratch_floats(cap, TW2, NH));
        int32_t* dpos = dev<int32_t>(1);
        ck(cudaMemcpy(dq, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice), "copy");
        std::vector<float> got(hq.size());
        unsigned long long hash[4] = {1469598103934665603ull, 1469598103934665603ull, 1469598103934665603ull, 1469598103934665603ull};
        const auto fold = [&](unsigned long long& h, size_t floats) {
            ck(cudaMemcpy(got.data(), dout, floats * 4, cudaMemcpyDeviceToHost), "copy");
            for (size_t i = 0; i < floats; ++i) {
                unsigned u;
                std::memcpy(&u, &got[i], 4);
                h = (h ^ u) * 1099511628211ull;
            }
        };
        const int64_t starts[] = {0, 1, 31, 32, 33, 500, 1999, 2047, 3000, 4000};
        for (int cache = 0; cache < 2; ++cache) {
            const uint16_t* kk = cache ? nullptr : dk;
            const uint16_t* vv = cache ? nullptr : dv;
            const basalt::kernels::KvInt8 q8 = cache ? c8 : basalt::kernels::KvInt8{};
            for (int T : {1, 2, 5, 8, 12, 16})
                for (int64_t p0 : starts) {
                    if (p0 + T > cap) continue;
                    basalt::kernels::gqa_decode_attn(dq, kk, vv, NH, NKV, p0, T, dscr, cap, dout, nullptr, nullptr, q8);
                    fold(hash[cache], (size_t) T * NH * HD);
                }
            for (int p0 : {1500, 3000}) {
                ck(cudaMemcpy(dpos, &p0, 4, cudaMemcpyHostToDevice), "copy");
                basalt::kernels::gqa_decode_attn(dq, kk, vv, NH, NKV, 0, 3, dscr, cap, dout, nullptr, dpos, q8);
                fold(hash[2], (size_t) 3 * NH * HD);
            }
            basalt::kernels::GqaSeg sg[2];
            sg[0].T = 5;
            sg[0].k_cache = kk;
            sg[0].v_cache = vv;
            sg[0].pos0 = 1000;
            sg[0].q8 = q8;
            sg[1] = sg[0];
            sg[1].T = 7;
            sg[1].pos0 = 3333;
            basalt::kernels::gqa_decode_attn_segs(dq, NH, NKV, sg, 2, dscr, cap, dout, nullptr);
            fold(hash[3], (size_t) 12 * NH * HD);
        }
        const unsigned long long frozen[4] = {0x430a8888287f57faull, 0xb6471ecdf9c79b73ull, 0x1c64ae81d108b6b5ull, 0x24e2ead0c780fabeull};
        std::printf("decode hashes: %016llx %016llx %016llx %016llx\n", hash[0], hash[1], hash[2], hash[3]);
        bool frozen_ok = true;
        for (int i = 0; i < 4; ++i) frozen_ok = frozen_ok && hash[i] == frozen[i];
        std::printf("decode kernel bits against the frozen hashes: %s\n", frozen_ok ? "identical" : "DIFFERENT");
        if (!frozen_ok) ++fails;
    }
    std::printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
