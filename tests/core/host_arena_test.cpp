#include "basalt/core/host_arena.hpp"
#include "basalt/core/conversation_buffer.hpp"
#include "basalt/core/model_file.hpp"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

namespace {
basalt::core::HostArena arena;
void check(bool condition, const char* message) {
    if (!condition) { std::fprintf(stderr, "FAIL: %s\n", message); std::abort(); }
}
void* allocate(size_t bytes, size_t alignment) { return arena.allocate(bytes, alignment); }
bool release(void* p) { return arena.owns(p) && arena.release(p); }
uint64_t available() { const auto s = arena.stats(); return s.capacity - s.used; }
const basalt::platform::HostAllocatorHooks hooks{allocate, release, available};
}

int main() {
    constexpr size_t bytes = 8 << 20;
    std::vector<uint8_t> backing(bytes);
    check(arena.reset(backing.data(), backing.size()), "open fixed backing");
    void* p = arena.allocate(bytes, 1);
    check(arena.allocation_bytes(p) == bytes && arena.allocation_bytes(static_cast<uint8_t*>(p) + 1) == 0,
          "only the live allocation base has a release extent");
    check(p == backing.data() && !arena.allocate(1), "exhaustion cannot grow backing");
    check(!arena.reset(nullptr, 0), "live backing cannot be replaced");
    check(!arena.release(static_cast<uint8_t*>(p) + 1), "an interior pointer cannot be freed");
    check(arena.release(p) && !arena.release(p), "free exactly once");
    size_t free_bytes = 0;
    arena.visit_free([&](void*, size_t n) { free_bytes += n; });
    check(free_bytes == bytes && arena.allocation_bytes(p) == 0, "free backing is enumerable after release");
    for (size_t align : {1, 16, 256, 4096, 65536}) {
        p = arena.allocate(12345, align);
        check(p && reinterpret_cast<uintptr_t>(p) % align == 0, "alignment");
        check(arena.release(p), "aligned free");
    }
    std::vector<std::thread> threads;
    std::atomic<uint64_t> operations{0};
    for (unsigned t = 0; t < 8; ++t) threads.emplace_back([t, &operations] {
        std::mt19937 random(431 + t);
        struct Block { void* p; size_t n; uint8_t value; };
        std::vector<Block> held;
        for (int i = 0; i < 12000; ++i) {
            if (!held.empty() && (held.size() >= 12 || random() % 2)) {
                const size_t at = random() % held.size(); const auto b = held[at];
                for (size_t j = 0; j < b.n; ++j) check(static_cast<uint8_t*>(b.p)[j] == b.value, "live data survives other allocations");
                check(arena.release(b.p), "random free"); held.erase(held.begin() + at);
            } else {
                const size_t n = 1 + random() % 32768;
                if (void* b = arena.allocate(n, 1u << (random() % 13))) {
                    const uint8_t value = static_cast<uint8_t>(t + i);
                    std::memset(b, value, n); held.push_back({b, n, value});
                }
            }
            ++operations;
        }
        for (const auto b : held) check(arena.release(b.p), "drain thread");
    });
    for (auto& thread : threads) thread.join();
    check(arena.stats().used == 0 && arena.stats().allocations == 0, "all fragments coalesce");
    p = arena.allocate(bytes, 1); check(p == backing.data(), "whole backing reusable after fragmentation"); arena.release(p);
    basalt::platform::host_allocator_hooks.store(&hooks);
    {
        basalt::core::ConversationBuffer first, second;
        first.resize(6 << 20, 37);
        bool refused = false;
        try { second.resize(3 << 20, 19); } catch (const std::bad_alloc&) { refused = true; }
        check(refused && arena.stats().used <= bytes, "snapshot cannot bypass the pool on exhaustion");
        first.resize(0); second.resize(0);
        second.resize(7 << 20, 54);
        std::vector<uint8_t> sample(512); check(second.read(sample.data(), (6 << 20), sample.size()), "reused snapshot readable");
        for (uint8_t value : sample) check(value == 54, "snapshot bytes preserved");
        second.resize(0);
        basalt::core::Hc8Copy weights;
        weights.data.resize(2 << 20, 83);
        check(arena.owns_range(weights.data.data(), weights.data.size()), "host weight copies use the reserved pool");
    }
    basalt::platform::host_allocator_hooks.store(nullptr);
    check(arena.stats().used == 0 && arena.reset(nullptr, 0), "pool closes after snapshots release");
    std::printf("host_arena_test: %llu concurrent operations and bounded snapshot reuse passed\n", (unsigned long long) operations.load());
}
