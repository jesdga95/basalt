#include "basalt/core/pinned.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "%s\n", message); ++failures; }
}
void clean_error(const char* message) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", message, cudaGetErrorString(error));
        ++failures;
    }
}
}  // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    const int device = count - 1;
    if (cudaSetDevice(device) != cudaSuccess) return 1;
    constexpr uint64_t slice = 2ull << 20;
    {
        // A cap smaller than the unsliced arena deliberately prevents CUDA registration.
        basalt::core::PinnedArena arena(slice, std::vector<uint64_t>{}, 1);
        check(arena.valid() && arena.registered_bytes == 0, "capped unsliced arena was registered");
        clean_error("unregistered arena construction left a CUDA error");
    }
    clean_error("unregistered arena destruction left a CUDA error");
    {
        basalt::core::PinnedArena arena(2 * slice, std::vector<uint64_t>{0, slice, 2 * slice}, slice);
        check(arena.valid() && arena.registered_bytes == slice && arena.registered_slices == 1,
              "partial registration did not respect the whole-slice cap");
        check(arena.slice_starts == std::vector<uint64_t>{0}, "registered slice ownership differs");
    }
    clean_error("sliced arena destruction left a CUDA error");
    {
        basalt::core::PinnedArena arena(slice);
        check(arena.valid() && arena.registered_bytes == slice && arena.registered_slices == 0,
              "small uncapped arena was not registered as one range");
    }
    clean_error("whole arena destruction left a CUDA error");
    int current = -1;
    check(cudaGetDevice(&current) == cudaSuccess && current == device, "registration changed the caller's device");
    std::puts(failures ? "pinned_arena_test: FAIL" : "pinned_arena_test: ok");
    return failures ? 1 : 0;
}
