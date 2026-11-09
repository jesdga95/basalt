// core::device_memory and devices_allocate: a card the runtime cannot serve is reported with the call, the error and the
// card, and leaves no error behind. Needs a CUDA device; skips (77) without one.
#include "basalt/core/device.hpp"

#include <cuda_runtime.h>

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
    std::printf(fails == 0 ? "device_memory_test: ok\n" : "device_memory_test: %d failures\n", fails);
    return fails == 0 ? 0 : 1;
}
