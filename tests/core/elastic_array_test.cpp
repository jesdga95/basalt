// core::ElasticArray: growing by steps keeps every byte already backed at its address, zeroes the new bytes, and `growth`
// names exactly what a grow maps; a trim followed by a grow zeroes again. Needs a CUDA device; skips (77) without one.
#include "basalt/core/elastic_array.hpp"
#include "basalt/core/vmm.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {
int fails = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++fails;
    }
}

// the bytes [from, to) of the array, read back
std::vector<uint8_t> read(const basalt::core::ElasticArray& a, uint64_t from, uint64_t to) {
    std::vector<uint8_t> h(to - from);
    if (cudaMemcpy(h.data(), a.data() + from, h.size(), cudaMemcpyDeviceToHost) != cudaSuccess) h.clear();
    return h;
}

uint8_t pattern(uint64_t i) { return (uint8_t) (i * 131u + 7u); }
}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0 || !basalt::core::vmm::supported()) {
        cudaGetLastError();
        std::puts("no CUDA device with virtual memory management: skipped");
        return 77;
    }
    using basalt::core::ElasticArray;
    const uint64_t gran = basalt::core::vmm::granularity();
    const uint64_t step = gran * 4;
    ElasticArray a;
    std::string err;
    check(a.open(step * 12 + 12345, err), "open reserves the range");
    check(a.backed() == 0 && a.capacity() >= step * 12 + 12345, "nothing is backed after open");
    check(a.growth(1) == step && a.growth(step) == step && a.growth(step + 1) == 2 * step, "growth rounds up to whole steps");
    check(a.growth(a.capacity() + 1) == 0, "growth of more than the range is 0 (ensure refuses it)");
    check(a.ensure(step / 3, 0, err) && a.backed() == step, "the first ensure backs one step");
    check(a.growth(step / 3) == 0, "growth of what is backed is 0");
    uint8_t* const base = a.data();
    std::vector<uint8_t> want(step);
    for (uint64_t i = 0; i < step; ++i) want[i] = pattern(i);
    check(cudaMemcpy(base, want.data(), step, cudaMemcpyHostToDevice) == cudaSuccess, "write a pattern into the backed bytes");
    // grow past each step boundary: the old bytes stay where they were, the new ones are zero, the address does not move
    for (int k = 2; k <= 9; ++k) {
        const uint64_t ask = step * (uint64_t) (k - 1) + 1;
        const uint64_t g = a.growth(ask), before = a.backed();
        check(a.ensure(ask, 0, err), "ensure past a step boundary");
        check(a.backed() == before + g && a.backed() == step * (uint64_t) k, "growth named what the ensure mapped");
        check(a.data() == base, "the address does not move");
        check(read(a, 0, before) == want, "the bytes backed before the grow are unchanged");
        const std::vector<uint8_t> fresh = read(a, before, a.backed());
        bool zero = !fresh.empty();
        for (uint8_t b : fresh) zero = zero && b == 0;
        check(zero, "the bytes a grow adds are zero");
        want.resize(a.backed(), (uint8_t) k);
        check(cudaMemcpy(base + before, want.data() + before, step, cudaMemcpyHostToDevice) == cudaSuccess, "write the new step");
    }
    // trim: whole steps past `bytes` go, the range stays; a grow again gives zeroed bytes at the same addresses
    a.trim(step * 3);
    want.resize(step * 3);
    check(a.backed() == step * 3 && a.data() == base, "trim keeps the steps up to the bytes asked");
    check(read(a, 0, step * 3) == want, "trim leaves the kept steps as they were");
    check(a.ensure(step * 6, 0, err) && a.backed() == step * 6, "a grow after a trim backs the range again");
    check(read(a, 0, step * 3) == want, "regrowth leaves the retained steps unchanged");
    const std::vector<uint8_t> again = read(a, step * 3, step * 6);
    bool zero = !again.empty();
    for (uint8_t b : again) zero = zero && b == 0;
    check(zero, "bytes backed again after a trim are zero");
    check(!a.ensure(a.capacity() + 1, 0, err), "an ensure past the range is refused");
    check(cudaGetLastError() == cudaSuccess, "no sticky error");
    std::printf(fails == 0 ? "elastic_array_test: ok\n" : "elastic_array_test: %d failures\n", fails);
    return fails == 0 ? 0 : 1;
}
