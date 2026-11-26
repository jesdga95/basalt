#pragma once

#include "basalt/core/device_copy.hpp"
#include "basalt/log.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

namespace ds4test {

inline void check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        basalt::log::error("basalt deepseek4 parity", "%s: %s", what, cudaGetErrorString(error));
        std::exit(1);
    }
}

template <class T>
struct Dev {
    T* p = nullptr;
    size_t n = 0;
    Dev() = default;
    explicit Dev(size_t count) : n(count) { check(cudaMalloc(&p, count * sizeof(T)), "cudaMalloc"); }
    Dev(const std::vector<T>& host) : Dev(host.size()) { up(host); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    ~Dev() { cudaFree(p); }
    void up(const std::vector<T>& host) {
        check(basalt::core::device_copy(p, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), "copy to device");
    }
    std::vector<T> down() const {
        std::vector<T> host(n);
        check(basalt::core::device_copy(host.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "copy from device");
        return host;
    }
};

struct Err { double max_abs = 0, rms_rel = 0; };

template <class A, class B>
Err compare(const std::vector<A>& got, const std::vector<B>& ref) {
    Err result;
    double square_error = 0.0, square_ref = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double error = std::fabs((double) got[i] - (double) ref[i]);
        result.max_abs = std::max(result.max_abs, error);
        square_error += error * error;
        square_ref += (double) ref[i] * (double) ref[i];
    }
    result.rms_rel = std::sqrt(square_error / std::max(square_ref, 1e-300));
    return result;
}

inline std::vector<float> randn(size_t n, uint32_t seed, float sigma = 1.f, float mean = 0.f) {
    std::mt19937 generator(seed);
    std::normal_distribution<float> distribution(mean, sigma);
    std::vector<float> values(n);
    for (float& value : values) value = distribution(generator);
    return values;
}

inline double bench_stream_us(const std::function<void(void*)>& fn, int iters = 100) {
    cudaStream_t stream;
    check(cudaStreamCreate(&stream), "create benchmark stream");
    for (int i = 0; i < 20; ++i) fn(stream);
    check(cudaStreamSynchronize(stream), "warm up benchmark stream");
    cudaEvent_t begin, end;
    check(cudaEventCreate(&begin), "create benchmark event");
    check(cudaEventCreate(&end), "create benchmark event");
    check(cudaEventRecord(begin, stream), "record benchmark start");
    for (int i = 0; i < iters; ++i) fn(stream);
    check(cudaEventRecord(end, stream), "record benchmark end");
    check(cudaEventSynchronize(end), "wait for benchmark");
    float milliseconds = 0.f;
    check(cudaEventElapsedTime(&milliseconds, begin, end), "measure benchmark");
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaStreamDestroy(stream);
    return 1000.0 * milliseconds / iters;
}

inline void bench_line(const char* name, double us, double bytes) {
    std::printf("bench %-34s %8.2f us/call  %7.1f GB/s (%.2f MB moved)\n", name, us,
                bytes / (us * 1e-6) / 1e9, bytes / 1e6);
}

}  // namespace ds4test
