// src/models/deepseek41/ops_parity.cpp - the layer loop's copy kernels across the bus: copy_rows (the routed experts'
// rows to mapped host memory and back) and copy_words (their ids), against the source bytes, with nothing written past
// the end, at the decode geometry (5120-wide rows, windows of 1 to 8 rows, 6 ids per row) and short counts. Prints
// microbench lines.
#include "models/deepseek41/kernels/test_util.hpp"
#include "models/deepseek41/ops.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace basalt::models::deepseek41;
using dstest::check;
using dstest::Dev;

namespace {

constexpr int kN = 5120, kMaxT = 8, kMaxWords = 64;

template <class T>
T* mapped(size_t n, T** dev) {
    T* h = nullptr;
    check(cudaHostAlloc((void**) &h, n * sizeof(T), cudaHostAllocPortable | cudaHostAllocMapped), "cudaHostAlloc");
    check(cudaHostGetDevicePointer((void**) dev, h, 0), "cudaHostGetDevicePointer");
    return h;
}

}  // namespace

int main() {
    dstest::Verdict v;
    cudaStream_t st;
    check(cudaStreamCreate(&st), "stream");
    float* dxh = nullptr;
    float* xh = mapped<float>((size_t) kMaxT * kN, &dxh);
    int32_t* dwh = nullptr;
    int32_t* wh = mapped<int32_t>(kMaxWords, &dwh);

    for (int T : {1, 2, 3, 5, 8}) {
        const std::vector<float> x = dstest::randn((size_t) T * kN, 7 + T);
        Dev<float> dx(x), back((size_t) kMaxT * kN);
        std::fill(xh, xh + (size_t) kMaxT * kN, -1.f);
        copy_rows(dx.p, dxh, T * kN, st);
        check(cudaStreamSynchronize(st), "sync");
        const bool same = std::memcmp(xh, x.data(), x.size() * sizeof(float)) == 0;
        const bool tail = std::all_of(xh + (size_t) T * kN, xh + (size_t) kMaxT * kN, [](float f) { return f == -1.f; });
        check(cudaMemsetAsync(back.p, 0, back.n * sizeof(float), st), "memset");
        copy_rows(dxh, back.p, T * kN, st);
        const std::vector<float> b = back.down();
        const bool same_back = std::memcmp(b.data(), x.data(), x.size() * sizeof(float)) == 0;
        std::printf("copy_rows T=%d: to host %s, past the end %s, back %s\n", T, same ? "same" : "DIFFER",
                    tail ? "untouched" : "WRITTEN", same_back ? "same" : "DIFFER");
        v.expect(same && tail && same_back, "copy_rows moves the rows' bytes and nothing else");
    }

    std::mt19937 g(11);
    for (int n : {1, 3, 5, 6, 18, 30, 48}) {
        std::vector<int32_t> w((size_t) n);
        for (int32_t& e : w) e = (int32_t) (g() & 0x7fffffff);
        Dev<int32_t> dw(w), back(kMaxWords);
        std::fill(wh, wh + kMaxWords, -1);
        copy_words(dw.p, dwh, n, st);
        check(cudaStreamSynchronize(st), "sync");
        const bool same = std::memcmp(wh, w.data(), w.size() * sizeof(int32_t)) == 0;
        const bool tail = std::all_of(wh + n, wh + kMaxWords, [](int32_t e) { return e == -1; });
        check(cudaMemsetAsync(back.p, 0, back.n * sizeof(int32_t), st), "memset");
        copy_words(dwh, back.p, n, st);
        const std::vector<int32_t> b = back.down();
        const bool same_back = std::memcmp(b.data(), w.data(), w.size() * sizeof(int32_t)) == 0 &&
                               std::all_of(b.begin() + n, b.end(), [](int32_t e) { return e == 0; });
        std::printf("copy_words %3d: to host %s, past the end %s, back %s\n", n, same ? "same" : "DIFFER",
                    tail ? "untouched" : "WRITTEN", same_back ? "same" : "DIFFER");
        v.expect(same && tail && same_back, "copy_words moves the words and nothing else");
    }

    Dev<float> dx((size_t) kMaxT * kN);
    for (int T : {1, 3}) {
        char name[64];
        std::snprintf(name, sizeof name, "copy_rows T=%d to host", T);
        dstest::bench_line(name, dstest::bench_stream_us([&](void* s) { copy_rows(dx.p, dxh, T * kN, s); }),
                           (double) T * kN * 4);
    }
    Dev<int32_t> dw(kMaxWords);
    dstest::bench_line("copy_words 18 to host", dstest::bench_stream_us([&](void* s) { copy_words(dw.p, dwh, 18, s); }),
                       18.0 * 4);
    cudaFreeHost(xh);
    cudaFreeHost(wh);
    cudaStreamDestroy(st);
    std::printf(v.failures ? "FAILED\n" : "PASS\n");
    return v.failures ? 1 : 0;
}
