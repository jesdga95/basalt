// CPU reference of the cache-row roundings and layouts for the parity test (the model's act_quant / fp4_act_quant).
#pragma once

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace ds4ref {

inline void check(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return;
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
    std::exit(1);
}

template <class T>
struct Dev {
    T* p = nullptr;
    size_t n = 0;
    Dev() = default;
    explicit Dev(size_t count) : n(count) { check(cudaMalloc((void**) &p, count * sizeof(T)), "cudaMalloc"); }
    explicit Dev(const std::vector<T>& h) : Dev(h.size()) { up(h); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    ~Dev() { if (p) cudaFree(p); }
    void up(const std::vector<T>& h) { check(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "up"); }
    std::vector<T> down() const {
        std::vector<T> h(n);
        if (p) check(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "down");
        return h;
    }
};

inline std::vector<float> randn(size_t n, uint32_t seed, float sigma = 1.f, float mean = 0.f) {
    std::mt19937 g(seed);
    std::normal_distribution<float> d(mean, sigma);
    std::vector<float> v(n);
    for (auto& x : v) x = d(g);
    return v;
}

/// The CPU reference of the model's cache roundings over groups of consecutive values: Fp8Pow2 E4M3 with a
/// power-of-two scale per 32 (window rows), Fp4E4m3 E2M1 with an E4M3 scale per 16 (compressed rows), Fp4Pow2 E2M1 with
/// a power-of-two scale per 32 (index keys and queries). Each value is first taken to fp32, as the kernels hold it.
enum class Round { Fp8Pow2, Fp4E4m3, Fp4Pow2 };
inline int group_of(Round k) { return k == Round::Fp4E4m3 ? 16 : 32; }
inline float pow2_ceil(float x) {
    int e;
    const float m = std::frexp(x, &e);   // x = m 2^e, m in [0.5, 1)
    return m == 0.5f ? x : std::ldexp(1.0f, e);
}
inline uint8_t e4m3_code(float v) { return (uint8_t) __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3); }
inline float e4m3_value(uint8_t q) { return __half2float(__half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t) q, __NV_E4M3))); }
inline float e4m3_round(float v) { return e4m3_value(e4m3_code(v)); }
inline const float* e2m1_table() {
    static const float t[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    return t;
}
inline float e2m1_round(float v) {
    const float a = std::fabs(v);
    const float r = a <= 0.25f ? 0.0f : a < 0.75f ? 0.5f : a <= 1.25f ? 1.0f : a < 1.75f ? 1.5f : a <= 2.5f ? 2.0f
                  : a < 3.5f ? 3.0f : a <= 5.0f ? 4.0f : 6.0f;
    return std::copysign(r, v);
}
inline uint8_t e2m1_code(float r) {   // r an E2M1 value
    for (int i = 0; i < 16; ++i)
        if (e2m1_table()[i] == r && std::signbit(e2m1_table()[i]) == std::signbit(r)) return (uint8_t) i;
    return 0xff;
}
// the group scale of `n` values at v
inline float ref_scale(const float* v, size_t n, Round k) {
    float a = 0.f;
    for (size_t i = 0; i < n; ++i) a = std::fmax(a, std::fabs(v[i]));
    if (k == Round::Fp8Pow2) return pow2_ceil(std::fmax(a, 1e-4f) * (1.0f / 448.0f));
    if (k == Round::Fp4E4m3) return e4m3_round(std::fmax(a, 6.0f * 0.001953125f) / 6.0f);
    return pow2_ceil(std::fmax(a, 6.0f * 1.1754943508222875e-38f) * (1.0f / 6.0f));
}
inline float ref_code_value(float x, float s, Round k) {
    return k == Round::Fp8Pow2 ? e4m3_round(std::fmin(std::fmax(x / s, -448.0f), 448.0f))
                               : e2m1_round(std::fmin(std::fmax(x / s, -6.0f), 6.0f));
}
template <class T>
void ref_round(T* v, size_t n, Round k) {
    const size_t g = (size_t) group_of(k);
    for (size_t g0 = 0; g0 < n; g0 += g) {
        float f[32];
        const size_t m = std::min(g, n - g0);
        for (size_t i = 0; i < m; ++i) f[i] = (float) v[g0 + i];
        const float s = ref_scale(f, m, k);
        for (size_t i = 0; i < m; ++i) v[g0 + i] = (T) (ref_code_value(f[i], s, k) * s);
    }
}

/// The packed row of d values (attn_index.hpp's layouts): codes first, then the group scales, `row_bytes` in all
/// (the padding zero).
inline std::vector<uint8_t> ref_pack(const float* v, int d, Round k, int row_bytes) {
    std::vector<uint8_t> r((size_t) row_bytes, 0);
    const int g = group_of(k), codes = k == Round::Fp8Pow2 ? d : d / 2;
    for (int g0 = 0; g0 < d; g0 += g) {
        const float s = ref_scale(v + g0, (size_t) g, k);
        for (int i = g0; i < g0 + g; ++i) {
            const float q = ref_code_value(v[i], s, k);
            if (k == Round::Fp8Pow2) r[(size_t) i] = e4m3_code(std::fmin(std::fmax(v[i] / s, -448.0f), 448.0f));
            else r[(size_t) i / 2] |= (uint8_t) (e2m1_code(q) << (4 * (i & 1)));
        }
        int e;
        std::frexp(s, &e);
        r[(size_t) (codes + g0 / g)] = k == Round::Fp4E4m3 ? e4m3_code(s) : (uint8_t) (e - 1 + 127);
    }
    return r;
}

/// The values of a packed row.
inline std::vector<float> ref_unpack(const uint8_t* r, int d, Round k) {
    std::vector<float> v((size_t) d);
    const int g = group_of(k), codes = k == Round::Fp8Pow2 ? d : d / 2;
    for (int i = 0; i < d; ++i) {
        const uint8_t sb = r[codes + i / g];
        const float s = k == Round::Fp4E4m3 ? e4m3_value(sb) : std::ldexp(1.0f, (int) sb - 127);
        const float q = k == Round::Fp8Pow2 ? e4m3_value(r[i]) : e2m1_table()[(r[i / 2] >> (4 * (i & 1))) & 15];
        v[(size_t) i] = q * s;
    }
    return v;
}

/// n packed rows of random normal values, rounded as `k`.
inline std::vector<uint8_t> packed_rows(int n, int d, int rb, Round k, uint32_t seed) {
    const auto f = randn((size_t) n * d, seed, 1.f);
    std::vector<uint8_t> b;
    for (int r = 0; r < n; ++r) {
        const auto row = ref_pack(&f[(size_t) r * d], d, k, rb);
        b.insert(b.end(), row.begin(), row.end());
    }
    return b;
}

/// Fraction of got[i] that differ from ref[i] (the rounded reference): values on the other side of a rounding boundary.
template <class A, class B>
double flips(const std::vector<A>& got, const std::vector<B>& ref) {
    size_t n = 0;
    for (size_t i = 0; i < ref.size(); ++i) n += std::fabs((double) got[i] - (double) ref[i]) > 1e-6 * std::fmax(1e-3, std::fabs((double) ref[i]));
    return ref.empty() ? 0.0 : (double) n / (double) ref.size();
}


}  // namespace ds4ref
