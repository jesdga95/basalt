// Registration ownership, independent mapped aliases and host endpoints on every visible card.
#include "basalt/core/ram_load.hpp"
#include "basalt/core/vmm.hpp"
#include "basalt/platform/memory.hpp"
#include "basalt/kernels/elementwise.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <utility>

namespace {
void require(bool ok, const char* what) {
    if (!ok) { std::printf("FAIL: %s\n", what); std::exit(1); }
}
void checked(cudaError_t e, int line) {
    if (e != cudaSuccess) { std::printf("CUDA at line %d: %s\n", line, cudaGetErrorString(e)); std::exit(1); }
}
#define ck(call) checked((call), __LINE__)
void roundtrip(uint8_t* host, size_t bytes, int cards) {
    std::vector<uint8_t> expected(host, host + bytes);
    for (int d = 0; d < cards; ++d) {
        ck(cudaSetDevice(d));
        uint8_t* gpu = nullptr;
        cudaStream_t s{};
        ck(cudaMalloc(&gpu, bytes));
        ck(cudaStreamCreate(&s));
        ck(cudaMemcpyAsync(gpu, host, bytes, cudaMemcpyHostToDevice, s));
        ck(cudaStreamSynchronize(s));
        std::memset(host, 0, bytes);
        ck(cudaMemcpyAsync(host, gpu, bytes, cudaMemcpyDeviceToHost, s));
        ck(cudaStreamSynchronize(s));
        require(std::memcmp(host, expected.data(), bytes) == 0, "host endpoint roundtrip changed bytes");
        ck(cudaStreamDestroy(s));
        ck(cudaFree(gpu));
    }
    ck(cudaSetDevice(0));
}
}

int main() {
    using namespace basalt::core;
    using namespace basalt::platform;
    require(ram_pin_slices({80, 0, 32, 16, 48, 64, 32}, 80, 16, 32) ==
            std::vector<uint64_t>({0, 32, 64, 80}), "slice cuts must be sorted and page aligned");
    require(ram_pin_slices({0, 12, 24, 36, 48, 60, 72}, 72, 16, 32) ==
            std::vector<uint64_t>({0, 48, 72}), "an object straddles a registration");
    require(ram_pin_slices({}, 0, 4096, 32).size() == 1, "empty registration");
    uint8_t a[64]{}, b[64]{};
    require(ram_pin_alias({0, 32}, {a, b}, 40) == b + 8, "second slice reused the first alias");
    require(ram_pin_alias({0, 32}, {a}, 40) == nullptr, "incomplete alias table accepted");
    require(pin_reserve(0) == (1ull << 30) && pin_reserve(100ull << 30) == (5ull << 30), "budget reserve");

    int cards = 0;
    if (cudaGetDeviceCount(&cards) != cudaSuccess || cards == 0) return 77;
    ck(cudaSetDevice(0));
    uint64_t allowance = 0;
    std::string why;
    const bool known = host_pin_budget(allowance, why);
    int current = -1;
    ck(cudaGetDevice(&current));
    require(current == 0, "budget probe changed the current card");
    std::printf("portable allowance: %s %llu MiB %s\n", known ? "known" : "driver",
                (unsigned long long) (allowance >> 20), why.c_str());
    if (cards > 1) {
        ck(cudaSetDevice(1));
        (void) host_pin_budget(allowance, why);
        ck(cudaGetDevice(&current));
        require(current == 1, "budget probe switched away from the helper card");
        ck(cudaSetDevice(0));
    }

    constexpr uint64_t chunk = 2ull << 20;
    auto* host = alloc_ram(3 * chunk);
    require(host != nullptr, "allocate registration fixture");
    std::memset(host, 0x6d, 3 * chunk);
    const uint8_t* device = nullptr;
    RamPin whole = pin_ram(host, 3 * chunk, 3 * chunk, {}, true, &device);
    require(whole.registered_bytes == 3 * chunk && device, "whole registration has no mapped alias");
    std::printf("whole range roundtrip\n");
    roundtrip(host, 3 * chunk, cards);
    unpin_ram(host, whole);
    ck(cudaGetLastError());

    if constexpr (kBoundHostRegistration) {
        // The last range belongs to another registration. Whole registration must fail without taking ownership of it.
        ck(cudaHostRegister(host + 2 * chunk, chunk, cudaHostRegisterPortable | cudaHostRegisterMapped));
        RamPin sliced = pin_ram(host, 3 * chunk, 0, {0, chunk, 2 * chunk, 3 * chunk}, true, &device);
        require(sliced.sliced && sliced.slices == 2 && sliced.registered_bytes == 2 * chunk, "partial registration ownership");
        require(sliced.slice_devices.size() == 2, "mapped aliases were lost");
        for (int i = 0; i < 2; ++i) {
            void* alias = nullptr;
            ck(cudaHostGetDevicePointer(&alias, host + i * chunk, 0));
            require(ram_pin_alias(sliced.slice_starts, sliced.slice_devices, i * chunk + 19) ==
                    (const uint8_t*) alias + 19, "wrong device alias in a slice");
            uint8_t from_alias[32]{};
            ck(cudaMemcpy(from_alias, (const uint8_t*) alias + 19, sizeof from_alias, cudaMemcpyDeviceToHost));
            require(std::memcmp(from_alias, host + i * chunk + 19, sizeof from_alias) == 0, "mapped alias changed bytes");
        }
        for (int i = 0; i < 3; ++i) roundtrip(host + i * chunk, chunk, cards);
        unpin_ram(host, sliced);
        ck(cudaGetLastError());
        ck(cudaHostUnregister(host + 2 * chunk));  // remains owned by the fixture
        ck(cudaHostRegister(host, 3 * chunk, cudaHostRegisterPortable));
        ck(cudaHostUnregister(host));             // every range taken by pin_ram was released
    }
    RamPin none = pin_ram(host, 0, chunk, {}, false);
    require(none.registered_bytes == 0 && none.slices == 0, "zero-byte registration was attempted");
    unpin_ram(host, none);
    basalt::platform::free_aligned(host);

    // Both tier layouts use the same owner, including ordinary registration and move/cleanup of a loaded span.
    std::vector<RamAllocation> allocations(2);
    require(allocations[0].allocate(chunk + 1), "allocate shared expert span");
    require(allocations[1].allocate(chunk + 1, false), "allocate ordinary fallback span");
    require(!allocations[1].allocation_pinned(), "pageable allocation became allocation-pinned");
    for (auto& span : allocations) std::memset(span.data(), 0x71, (size_t) span.bytes());
    auto pins = pin_ram_spans(allocations, false);
    for (size_t i = 0; i < allocations.size(); ++i) {
        require(pins[i].registered_bytes >= allocations[i].bytes(), "shared pin did not cover its span");
        require(pins[i].allocation_backed == allocations[i].allocation_pinned(), "wrong pin ownership");
        roundtrip(allocations[i].data(), (size_t) allocations[i].bytes(), cards);
        unpin_ram(allocations[i].data(), pins[i]);
    }
    const auto* at = allocations[0].data();
    RamAllocation moved = std::move(allocations[0]);
    require(!allocations[0].data() && moved.data() == at, "allocation move copied or lost the span");
    const uint8_t* alias = nullptr;
    const RamPin mapped_pin = pin_ram(moved, {}, true, &alias);
    require(alias && mapped_pin.registered_bytes >= moved.bytes(), "shared mapped alias");
    int32_t* gpu_values = nullptr;
    ck(cudaMalloc(&gpu_values, 128));
    basalt::kernels::copy_i32_from_mapped(gpu_values, (const int32_t*) alias, 32, nullptr);
    ck(cudaGetLastError());
    int32_t values[32]{};
    ck(cudaMemcpy(values, gpu_values, sizeof values, cudaMemcpyDeviceToHost));
    require(std::memcmp(values, moved.data(), sizeof values) == 0, "GPU kernel read the wrong mapped bytes");
    ck(cudaFree(gpu_values));
    unpin_ram(moved.data(), mapped_pin);
    moved.reset();
    require(moved.data() == nullptr && moved.capacity() == 0, "allocation reset retained ownership");
    require(!moved.allocate(~uint64_t(0)) && !moved.data(), "allocation overflow");
    allocations.clear();
    ck(cudaGetLastError());

    if constexpr (kPreferHostVmm) {
        const bool host_vmm = vmm::host_granularity() > 0;
        if (!host_vmm) std::printf("host VMM unavailable; ordinary registration tested\n");
        if (host_vmm) {
            // One expert can straddle the boundary between two physical backing allocations.
            constexpr uint64_t boundary = 2ull << 30;
            RamAllocation spanning;
            require(spanning.allocate(boundary + chunk) && spanning.allocation_pinned(), "multi-piece host mapping");
            std::memset(spanning.data() + boundary - chunk, 0x2b, 2 * chunk);
            roundtrip(spanning.data() + boundary - chunk, 2 * chunk, cards);
            int32_t* gpu = nullptr;
            ck(cudaMalloc(&gpu, 128));
            const auto* src = spanning.data() + boundary - 64;
            basalt::kernels::copy_i32_from_mapped(gpu, (const int32_t*) src, 32, nullptr);
            int32_t back[32]{};
            ck(cudaMemcpy(back, gpu, sizeof back, cudaMemcpyDeviceToHost));
            require(std::memcmp(back, src, sizeof back) == 0, "GPU read across physical backing boundary");
            ck(cudaFree(gpu));
        }
        for (int i = 0; host_vmm && i < 3; ++i) {
            uint64_t mapped = 0;
            uint8_t* p = vmm::alloc_host(chunk + 1, mapped);
            require(p && mapped >= chunk + 1 && mapped % vmm::host_granularity() == 0, "host mapping extent");
            std::memset(p, 0xa7, chunk + 1);
            std::printf("host VMM roundtrip %d\n", i);
            roundtrip(p, chunk + 1, cards);
            vmm::free_host(p, mapped);
        }
        uint64_t mapped = 99;
        require(vmm::alloc_host(0, mapped) == nullptr && mapped == 0, "empty host mapping");
        require(vmm::alloc_host(~uint64_t(0), mapped) == nullptr && mapped == 0, "host mapping overflow");
    }
    ck(cudaGetLastError());
    std::printf("PASS: host registration and endpoints on %d cards\n", cards);
}
