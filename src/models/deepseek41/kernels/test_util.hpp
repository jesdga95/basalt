// Shared helpers of the DeepSeek-V4.1 kernel parity tests: error statistics, raw dumps, timing.
#pragma once

#include "basalt/log.hpp"
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace dstest {

inline void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        basalt::log::error("basalt deepseek41", "%s: %s", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

template <class T>
struct Dev {
    T* p = nullptr;
    size_t n = 0;
    Dev() = default;
    explicit Dev(size_t count) : n(count) { check(cudaMalloc(&p, count * sizeof(T)), "cudaMalloc"); }
    Dev(const std::vector<T>& h) : Dev(h.size()) { up(h); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    ~Dev() { cudaFree(p); }
    void up(const std::vector<T>& h) { check(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "up"); }
    std::vector<T> down() const {
        std::vector<T> h(n);
        check(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "down");
        return h;
    }
};

struct Err {
    double max_abs = 0, max_rel = 0, rms_rel = 0;
    size_t worst = 0;
};

/// Compares `got` with `ref`: max abs error, max error relative to max(|ref|, floor), and sqrt(sum err^2 / sum ref^2).
template <class A, class B>
Err compare(const std::vector<A>& got, const std::vector<B>& ref, double floor = 1e-3) {
    Err r;
    double se = 0, sr = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = std::fabs((double) got[i] - (double) ref[i]);
        const double rel = d / std::fmax(std::fabs((double) ref[i]), floor);
        if (d > r.max_abs) { r.max_abs = d; r.worst = i; }
        r.max_rel = std::fmax(r.max_rel, rel);
        se += d * d;
        sr += (double) ref[i] * (double) ref[i];
    }
    r.rms_rel = std::sqrt(se / std::fmax(sr, 1e-300));
    return r;
}

inline bool read_file(const std::string& path, std::vector<char>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    out.resize((size_t) f.tellg());
    f.seekg(0);
    f.read(out.data(), (std::streamsize) out.size());
    return (bool) f;
}

template <class T>
bool read_vec(const std::string& path, std::vector<T>& v) {
    std::vector<char> raw;
    if (!read_file(path, raw)) return false;
    v.resize(raw.size() / sizeof(T));
    std::memcpy(v.data(), raw.data(), v.size() * sizeof(T));
    return true;
}

inline std::vector<float> randn(size_t n, uint32_t seed, float sigma = 1.f, float mean = 0.f) {
    std::mt19937 g(seed);
    std::normal_distribution<float> d(mean, sigma);
    std::vector<float> v(n);
    for (auto& x : v) x = d(g);
    return v;
}

/// Mean microseconds per call of `fn` (stream-less launches, warm-up first).
inline double bench_us(const std::function<void()>& fn, int iters = 300) {
    for (int i = 0; i < 20; ++i) fn();
    check(cudaDeviceSynchronize(), "warmup");
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a, nullptr);
    for (int i = 0; i < iters; ++i) fn();
    cudaEventRecord(b, nullptr);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return 1000.0 * ms / iters;
}

/// Mean microseconds per call of `fn(stream)` on a private stream, so a call does not synchronise.
inline double bench_stream_us(const std::function<void(void*)>& fn, int iters = 500) {
    cudaStream_t st;
    check(cudaStreamCreate(&st), "stream");
    for (int i = 0; i < 20; ++i) fn(st);
    check(cudaStreamSynchronize(st), "warmup");
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a, st);
    for (int i = 0; i < iters; ++i) fn(st);
    cudaEventRecord(b, st);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    cudaStreamDestroy(st);
    return 1000.0 * ms / iters;
}

inline void bench_line(const char* name, double us, double bytes) {
    std::printf("bench %-34s %8.2f us/call  %7.1f GB/s (%.2f MB moved)\n", name, us, bytes / (us * 1e-6) / 1e9,
                bytes / 1e6);
}

struct Verdict {
    int failures = 0;
    void expect(bool ok, const char* what) {
        if (!ok) { std::printf("FAIL: %s\n", what); ++failures; }
    }
};

}  // namespace dstest
