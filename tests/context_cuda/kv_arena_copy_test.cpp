// GPU test: a session's KV prefix in a real KvArena leaves for host and disk and comes back bit-identical.
// Small geometry (4 arrays, a few pages); skips (77) without a device, VMM support or room.
#include "basalt/context/kv_arena_copy.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <vector>

using namespace basalt::context;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); std::exit(1); } } while (0)

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) { std::puts("no CUDA device: skipped"); return 77; }
    constexpr int64_t kPage = 64, kPages = 4096;
    const std::vector<uint64_t> page_bytes = {kPage * 4 * 128, kPage * 4 * 128, kPage * 4 * 2 * 2, kPage * 4 * 2 * 2};
    basalt::core::KvArena arena;
    std::string e;
    if (!arena.open(page_bytes, kPages, kPage, e)) { std::printf("skip: %s\n", e.c_str()); return 77; }
    if (!arena.map(kPage * 3, e)) { std::printf("skip: %s\n", e.c_str()); return 77; }   // three pages, rounded to a granule
    KvArenaCopy dev(arena, page_bytes);

    const uint64_t bytes = dev.adopt("kv1", e);
    CHECK(bytes > 0 && bytes % dev.bytes_per_page() == 0);
    // fill each array's prefix with its own pattern, straight into the arena
    const uint64_t pages = bytes / dev.bytes_per_page();
    std::vector<uint8_t> pattern(bytes);
    uint64_t at = 0;
    for (size_t i = 0; i < page_bytes.size(); ++i) {
        const uint64_t len = pages * page_bytes[i];
        for (uint64_t j = 0; j < len; ++j) pattern[at + j] = (uint8_t) (j * 31 + i * 7 + 3);
        CHECK(cudaMemcpy(arena.array(i), pattern.data() + at, len, cudaMemcpyHostToDevice) == cudaSuccess);
        at += len;
    }
    // a second context cannot take the arena while kv1 is resident
    CHECK(!dev.allocate("kv2", dev.bytes_per_page(), e) && dev.adopt("kv2", e) == 0);

    char tmpl[] = "/tmp/kac_XXXXXX";
    CHECK(mkdtemp(tmpl));
    constexpr size_t PB = 4096;
    std::vector<uint8_t> pool(PB * 4096 + PB);
    void* base = (void*) (((uintptr_t) pool.data() + PB - 1) & ~(uintptr_t) (PB - 1));
    PageStore store;
    CHECK(store.open(base, PB * 4096, PB));
    DeviceTierMover dm(store, dev);
    FileTierMover fm(store, tmpl);
    CompositeTierMover mover(dm, fm);
    std::atomic<bool> no{false};
    Digest a{}, b{}, c{}, d{};
    CHECK(mover.copy({"kv1", Tier::gpu, Tier::host, 1, bytes, false, {}}, no, a, e) == TierMover::Outcome::done);
    CHECK(mover.copy({"kv1", Tier::host, Tier::disk, 1, bytes, true, a}, no, b, e) == TierMover::Outcome::done);
    mover.release("kv1", Tier::gpu);            // unmaps: the VRAM is back
    mover.release("kv1", Tier::host);
    CHECK(arena.cells() == 0 && arena.mapped_bytes() == 0);
    CHECK(mover.copy({"kv1", Tier::disk, Tier::host, 1, bytes, true, a}, no, c, e) == TierMover::Outcome::done);
    CHECK(mover.copy({"kv1", Tier::host, Tier::gpu, 1, bytes, true, a}, no, d, e) == TierMover::Outcome::done);
    CHECK(a == b && b == c && c == d);
    // read the restored arrays directly: every byte of every array's prefix matches
    at = 0;
    for (size_t i = 0; i < page_bytes.size(); ++i) {
        const uint64_t len = pages * page_bytes[i];
        std::vector<uint8_t> back(len);
        CHECK(cudaMemcpy(back.data(), arena.array(i), len, cudaMemcpyDeviceToHost) == cudaSuccess);
        CHECK(std::equal(back.begin(), back.end(), pattern.begin() + at));
        at += len;
    }
    // reads spanning an array boundary, and past the context
    std::vector<uint8_t> span(256);
    const uint64_t edge = pages * page_bytes[0];
    CHECK(dev.download("kv1", edge - 128, span.data(), 256, e));
    CHECK(std::equal(span.begin(), span.end(), pattern.begin() + (edge - 128)));
    CHECK(!dev.download("kv1", bytes - 8, span.data(), 16, e));
    mover.release("kv1", Tier::gpu);
    mover.release("kv1", Tier::host);
    mover.release("kv1", Tier::disk);
    CHECK(arena.cells() == 0);
    std::puts("kv_arena_copy_test OK");
    std::string rm = std::string("rm -rf ") + tmpl; (void) !system(rm.c_str());
    return 0;
}
