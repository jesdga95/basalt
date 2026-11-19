// core::device_memory and devices_allocate: a card the runtime cannot serve is reported with the call, the error and the
// card, and leaves no error behind. core::device_mem_info: free is the smaller of the device's own memory and the
// commit room, and where the OS charges device memory to its commit limit an allocation is charged when it is made.
// Needs a CUDA device; skips (77) without one.
#include "basalt/core/device.hpp"
#include "basalt/core/device_memory.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>

namespace {
int fails = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++fails;
    }
}
}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        cudaGetLastError();
        std::puts("no CUDA device: skipped");
        return 77;
    }
    std::string why;
    uint64_t f = 0, t = 0;
    check(!basalt::core::device_memory(99, f, t, why), "a card that does not exist has no memory to read");
    check(why.rfind("cuda", 0) == 0 && why.find("(code ") != std::string::npos && why.find("device 99 of ") != std::string::npos,
          "the failure names the call, the error code and the card");
    why.clear();
    check(!basalt::core::devices_allocate({99}, why) && why.find("device 99 of ") != std::string::npos, "an unserved card fails the allocation check");
    check(cudaGetLastError() == cudaSuccess, "the checks leave no sticky error");
    check(basalt::core::devices_allocate({}, why), "no cards, nothing to fail");
    std::string why2;
    check(basalt::core::devices_allocate({-1}, why) == basalt::core::device_memory(-1, f, t, why2),
          "the current card (-1) is served exactly when it answers a query");
    // the two bounds behind device_mem_info (readings moments apart: within 64 MiB)
    {
        using namespace basalt::core;
        constexpr uint64_t kSlack = 64ull << 20, kGiB = 1ull << 30;
        const DeviceMemoryModel model = device_memory_model();
        DeviceMemoryReading r;
        size_t free_b = 0, total_b = 0;
        check(device_mem_reading(r) == cudaSuccess && device_mem_info(&free_b, &total_b) == cudaSuccess,
              "device_mem_reading and device_mem_info read the current device");
        const uint64_t lo = std::min(r.device_free, r.commit_room);
        check((uint64_t) free_b + kSlack >= lo && lo + kSlack >= (uint64_t) free_b && total_b == r.total,
              "device_mem_info's free is the smaller of the device's own and the commit room");
        check(model.commit_bound == (r.commit_room != UINT64_MAX), "a commit room exactly where the OS charges one");
        check(!model.commit_bound || model.integrated, "only an integrated GPU's memory is charged to the commit");
        if (model.commit_bound && r.commit_room > 2 * kGiB) {
            void* p = nullptr;
            check(cudaMalloc(&p, kGiB) == cudaSuccess, "1 GiB allocates");
            DeviceMemoryReading after;
            (void) device_mem_reading(after);
            check(after.commit_room + kGiB <= r.commit_room + kSlack,
                  "an allocation is charged to the commit when it is made, before it is written");
            (void) cudaFree(p);
        }
        std::printf("device memory: %s, own %.2f of %.2f GiB free, commit room %.2f GiB\n",
                    model.integrated ? "integrated" : "discrete", (double) r.device_free / (double) kGiB,
                    (double) r.total / (double) kGiB,
                    r.commit_room == UINT64_MAX ? -1.0 : (double) r.commit_room / (double) kGiB);
    }
    std::printf(fails == 0 ? "device_memory_test: ok\n" : "device_memory_test: %d failures\n", fails);
    return fails == 0 ? 0 : 1;
}
