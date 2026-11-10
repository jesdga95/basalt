// GPU test: a 1 MiB state round-trips gpu -> host -> disk -> host -> gpu bit-identically through the CUDA DeviceCopy.
// Skips (77) when the device has no room: it never competes with a running engine for more than a moment.
#include "basalt/context/cuda_device_copy.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <vector>

using namespace basalt::context;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); std::exit(1); } } while (0)

int main() {
    constexpr size_t PB = 4096, N = 512, BYTES = 1 << 20;
    CudaDeviceCopy dev(8 << 20);
    std::string e;
    if (!dev.allocate("k", BYTES, e)) { std::printf("skip: %s\n", e.c_str()); return 77; }
    std::vector<uint8_t> pattern(BYTES), back(BYTES);
    for (size_t i = 0; i < BYTES; ++i) pattern[i] = (uint8_t) (i * 131 + 17);
    CHECK(dev.upload("k", 0, pattern.data(), BYTES, e));

    char tmpl[] = "/tmp/cdc_XXXXXX";
    CHECK(mkdtemp(tmpl));
    std::vector<uint8_t> pool(PB * N + PB);
    void* base = (void*) (((uintptr_t) pool.data() + PB - 1) & ~(uintptr_t) (PB - 1));
    PageStore store;
    CHECK(store.open(base, PB * N, PB));
    DeviceTierMover dm(store, dev);
    FileTierMover fm(store, tmpl);
    CompositeTierMover mover(dm, fm);
    std::atomic<bool> no{false};
    Digest a{}, b{}, c{}, d{};
    CHECK(mover.copy({"k", Tier::gpu, Tier::host, 1, BYTES, false, {}}, no, a, e) == TierMover::Outcome::done);
    CHECK(mover.copy({"k", Tier::host, Tier::disk, 1, BYTES, true, a}, no, b, e) == TierMover::Outcome::done);
    mover.release("k", Tier::gpu);
    mover.release("k", Tier::host);
    CHECK(dev.owned_bytes() == 0);
    CHECK(mover.copy({"k", Tier::disk, Tier::host, 1, BYTES, true, a}, no, c, e) == TierMover::Outcome::done);
    CHECK(mover.copy({"k", Tier::host, Tier::gpu, 1, BYTES, true, a}, no, d, e) == TierMover::Outcome::done);
    CHECK(dev.download("k", 0, back.data(), BYTES, e));
    CHECK(back == pattern && a == b && b == c && c == d);
    // ranges are bounds-checked, bind() is non-owning
    CHECK(!dev.download("k", BYTES - 8, back.data(), 16, e));
    void* mine = nullptr;
    CHECK(cudaMalloc(&mine, 4096) == cudaSuccess);
    CHECK(dev.bind("b", mine, 4096, e) && !dev.bind("b", mine, 4096, e));
    dev.free("b");
    CHECK(cudaMemset(mine, 0, 4096) == cudaSuccess);   // still valid: bind never owned it
    cudaFree(mine);
    mover.release("k", Tier::gpu);
    mover.release("k", Tier::host);
    mover.release("k", Tier::disk);
    std::puts("cuda_device_copy_test OK");
    std::string rm = std::string("rm -rf ") + tmpl; (void) !system(rm.c_str());
    return 0;
}
