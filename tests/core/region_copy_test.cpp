// tests/core/region_copy_test.cpp - a running state's regions packed into one device buffer and back (regions_gather,
// regions_scatter) and a checkpoint restored through the deferral's spare (CheckpointDeferral::restore).
//
// Regions of odd sizes and alignments beside 16-byte ones, more than one launch takes: the packed bytes are the regions'
// back to back, and the scatter puts every byte back (the bytes between the regions untouched). A restore through the
// spare gives the regions a direct copy in gives. --bench: GB/s of the gather and the scatter over a model's ~290
// regions of a checkpoint (window rings and convolution rings).
#include "basalt/core/checkpoint_transfer.hpp"
#include "basalt/core/region_copy.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace basalt::core;

namespace {
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", description); std::exit(1); }
}
void cuda_check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "FAIL: cuda: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

// One device arena the regions sit in at the given offsets (odd ones included), filled with a pattern.
struct Arena {
    uint8_t* base = nullptr;
    size_t bytes = 0;
    std::vector<DeviceRegion> regions;
    explicit Arena(const std::vector<std::pair<size_t, size_t>>& at_size) {
        for (const auto& [at, n] : at_size) bytes = std::max(bytes, at + n);
        bytes += 64;
        cuda_check(cudaMalloc(&base, bytes));
        for (const auto& [at, n] : at_size) regions.push_back({base + at, n});
    }
    ~Arena() { cudaFree(base); }
    std::vector<uint8_t> get() const {
        std::vector<uint8_t> h(bytes);
        cuda_check(cudaMemcpy(h.data(), base, bytes, cudaMemcpyDeviceToHost));
        return h;
    }
    void put(const std::vector<uint8_t>& h) { cuda_check(cudaMemcpy(base, h.data(), bytes, cudaMemcpyHostToDevice)); }
};

std::vector<std::pair<size_t, size_t>> layout(std::mt19937& rng, int n) {
    std::vector<std::pair<size_t, size_t>> v;
    size_t at = 0;
    std::uniform_int_distribution<int> kind(0, 3), odd(1, 999), big(1, 64);
    for (int i = 0; i < n; ++i) {
        const int k = kind(rng);
        const size_t size = k == 0 ? (size_t) odd(rng) : k == 1 ? (size_t) big(rng) * 4096 : k == 2 ? (size_t) big(rng) * 16 + 4 : 16;
        at += k == 2 ? 3 : 0;   // some regions start off 16-byte alignment
        v.push_back({at, size});
        at += size + (size_t) odd(rng) % 32;
    }
    return v;
}

void test_round_trip(int n) {
    std::mt19937 rng(7 + n);
    const auto lay = layout(rng, n);
    Arena a(lay);
    std::vector<uint8_t> h(a.bytes);
    for (auto& b : h) b = (uint8_t) rng();
    a.put(h);
    size_t total = 0;
    for (const DeviceRegion& r : a.regions) total += r.bytes;
    uint8_t* packed = nullptr;
    cuda_check(cudaMalloc(&packed, total + 16));
    cuda_check(regions_gather(a.regions, packed, nullptr));
    cuda_check(cudaDeviceSynchronize());
    std::vector<uint8_t> p(total);
    cuda_check(cudaMemcpy(p.data(), packed, total, cudaMemcpyDeviceToHost));
    std::vector<uint8_t> want;
    for (const auto& [at, sz] : lay) want.insert(want.end(), h.begin() + (long) at, h.begin() + (long) (at + sz));
    check(p == want, "the gather packs the regions back to back");
    // scatter into an arena of zeros: the regions' bytes come back, the bytes between them stay zero
    a.put(std::vector<uint8_t>(a.bytes, 0));
    cuda_check(regions_scatter(a.regions, packed, nullptr));
    cuda_check(cudaDeviceSynchronize());
    const std::vector<uint8_t> back = a.get();
    std::vector<uint8_t> expect(a.bytes, 0);
    for (const auto& [at, sz] : lay) std::copy(h.begin() + (long) at, h.begin() + (long) (at + sz), expect.begin() + (long) at);
    check(back == expect, "the scatter puts every region back, nothing between them");
    cudaFree(packed);
    std::printf("regions_gather / regions_scatter, %d regions: bitwise round trip\n", n);
}

void test_restore() {
    std::mt19937 rng(11);
    const auto lay = layout(rng, 300);
    Arena a(lay), b(lay);
    size_t total = 0;
    for (const DeviceRegion& r : a.regions) total += r.bytes;
    std::vector<uint8_t> state(total);
    for (auto& x : state) x = (uint8_t) rng();
    std::string err;
    CheckpointDeferral d;
    check(d.reserve(total, err), "the spare");
    check(d.restore({{state.data(), a.regions}}, nullptr, err), "a restore through the spare");
    check(checkpoint_copy_in({{state.data(), b.regions}}, nullptr, err), "a direct copy in");
    std::vector<uint8_t> ha = a.get(), hb = b.get();
    bool same = true;
    for (const auto& [at, sz] : lay) same = same && !std::memcmp(ha.data() + at, hb.data() + at, sz);
    check(same, "a restore through the spare = a direct copy in");
    std::printf("CheckpointDeferral::restore, 300 regions: = checkpoint_copy_in\n");
}

void bench() {
    // a checkpoint of 48 visits: 36 window rings' keys and values (~1 MiB each at a short position) and 4 convolution
    // rings a visit (two of 76 KiB, two of 171 KiB)
    std::vector<std::pair<size_t, size_t>> lay;
    size_t at = 0;
    const auto add = [&](size_t n) {
        lay.push_back({at, n});
        at += (n + 255) / 256 * 256;
    };
    for (int v = 0; v < 48; ++v) {
        if (v % 4) {
            add(8 * 640 * 128);
            add(8 * 640 * 128);
        }
        add(19 * 1024 * 4);
        add(19 * 1024 * 4);
        add(19 * 2304 * 4);
        add(19 * 2304 * 4);
    }
    Arena a(lay);
    size_t total = 0;
    for (const DeviceRegion& r : a.regions) total += r.bytes;
    uint8_t* packed = nullptr;
    cuda_check(cudaMalloc(&packed, total));
    for (int pass = 0; pass < 2; ++pass) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 20; ++i) cuda_check(pass ? regions_scatter(a.regions, packed, nullptr) : regions_gather(a.regions, packed, nullptr));
        cuda_check(cudaDeviceSynchronize());
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 20;
        std::printf("bench %s: %zu regions, %.1f MiB, %.1f us, %.0f GB/s\n", pass ? "regions_scatter" : "regions_gather",
                    a.regions.size(), (double) total / 1048576.0, us, 2.0 * (double) total / (us * 1e3));
    }
    cudaFree(packed);
}
}  // namespace

int main(int argc, char** argv) {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
    if (argc > 1 && std::string(argv[1]) == "--bench") {
        bench();
        return 0;
    }
    test_round_trip(5);
    test_round_trip(290);
    test_round_trip(700);
    test_restore();
    std::printf("region_copy_test: %d checks passed\n", checks);
    return 0;
}
