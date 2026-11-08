#include "basalt/core/host_pool.hpp"
#include "basalt/core/conversation_buffer.hpp"
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    auto check = [](bool ok, const char* what) {
        if (!ok) std::fprintf(stderr, "FAIL: %s\n", what);
        return ok;
    };
    std::string error;
    if (!check(!basalt::core::open_host_pool(32u << 20, 16u << 20, error) && !basalt::core::host_pool_enabled(),
               "insufficient backing is refused before startup")) return 1;
    if (!check(basalt::core::open_host_pool(8u << 20, 16u << 20, error), error.c_str())) return 1;
    if (!check(basalt::core::discard_host_pool_free(), "unused prefaulted backing can be returned")) return 1;
    void *a = nullptr, *b = nullptr, *extra = nullptr;
    if (!check(basalt::core::allocate_pinned_host(&a, 4u << 20, cudaHostAllocMapped) == cudaSuccess &&
               basalt::core::allocate_pinned_host(&b, 4u << 20, cudaHostAllocPortable) == cudaSuccess,
               "two allocations consume the same preallocated backing")) return 1;
    if (!check(basalt::core::allocate_pinned_host(&extra, 1, 0) == cudaErrorMemoryAllocation && !extra,
               "exhaustion does not fall back to another CUDA host allocation")) return 1;
    if (!check(!basalt::core::allocate_plain_host(1),
               "plain host allocations cannot bypass the exhausted pool")) return 1;
    unsigned flags = 0; void* alias = nullptr;
    if (!check(cudaHostGetFlags(&flags, a) == cudaSuccess && cudaHostGetDevicePointer(&alias, a, 0) == cudaSuccess && alias,
               "suballocations retain the registered device alias")) return 1;
    std::memset(b, 0x19, 4u << 20);
    void* device = nullptr;
    if (!check(cudaMalloc(&device, 4096) == cudaSuccess && cudaMemset(device, 0x37, 4096) == cudaSuccess &&
               cudaMemcpy(a, device, 4096, cudaMemcpyDeviceToHost) == cudaSuccess, "DMA into a pool suballocation")) return 1;
    for (size_t i = 0; i < 4096; ++i) if (!check(static_cast<uint8_t*>(a)[i] == 0x37, "DMA payload")) return 1;
    if (!check(basalt::core::register_host_memory(a, 4096, cudaHostRegisterMapped) == cudaSuccess &&
               basalt::core::unregister_host_memory(a) == cudaSuccess, "registration of an owned subrange is reused")) return 1;
    if (!check(basalt::core::release_pinned_host(a) == cudaSuccess &&
               basalt::core::discard_host_pool_free(1u << 20) &&
               basalt::core::discard_host_pool_free(),
               "released backing is discarded beside a live CUDA lease")) return 1;
    for (size_t i = 0; i < (4u << 20); ++i)
        if (!check(static_cast<uint8_t*>(b)[i] == 0x19, "a live lease survives neighboring page discard")) return 1;
    if (!check(basalt::core::release_pinned_host(b) == cudaSuccess,
               "free returns bytes to the same backing")) return 1;
    {
        basalt::core::ConversationBuffer bytes; bytes.resize(7u << 20, 91);
        std::vector<uint8_t> sample(512);
        if (!check(bytes.read(sample.data(), 6u << 20, sample.size()), "snapshot reads reused pool bytes")) return 1;
        for (uint8_t x : sample) if (!check(x == 91, "snapshot payload")) return 1;
    }
    const auto stats = basalt::core::host_pool_stats();
    cudaFree(device);
    if (!check(stats.capacity == (8u << 20) && stats.used == 0 && stats.peak <= stats.capacity && stats.failures == 2,
               "one fixed backing survived DMA, exhaustion and reuse")) return 1;
    std::printf("host_pool_gpu_test: prefault, pinning, DMA, exhaustion and reuse passed\n");
}
