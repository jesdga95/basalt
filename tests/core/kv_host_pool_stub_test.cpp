// Run the pool's registration transactions with a CPU registrar and real virtual-memory ranges.
#include "basalt/core/kv_host_pool.hpp"
#include "basalt/core/session.hpp"
#include "basalt/platform/memory.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

namespace {
struct Pin { uint8_t* p; size_t n; };
std::vector<Pin> pins;
int attempts = 0, fail_on = 0, alias_on = 0, identity = 1, allocated = 0, unregisters = 0, restore_fail_on = 0;
uint64_t peak_pinned = 0;

bool same_pins(const std::vector<Pin>& expected) {
    if (expected.size() != pins.size()) return false;
    for (const auto& wanted : expected)
        if (std::none_of(pins.begin(), pins.end(), [&](const Pin& pin) { return pin.p == wanted.p && pin.n == wanted.n; }))
            return false;
    return true;
}

cudaError_t stub_register(void* p, size_t n, unsigned flags) {
    ++attempts;
    if (attempts == fail_on || attempts == restore_fail_on) return cudaErrorMemoryAllocation;
    const size_t page = (size_t) basalt::platform::vm_page_size();
    if ((uintptr_t) p % page || n % page || flags != (cudaHostRegisterMapped | cudaHostRegisterPortable))
        return cudaErrorInvalidValue;
    for (const auto& pin : pins)
        if ((uintptr_t) p < (uintptr_t) pin.p + pin.n && (uintptr_t) pin.p < (uintptr_t) p + n)
            return cudaErrorHostMemoryAlreadyRegistered;
    pins.push_back({(uint8_t*) p, n});
    uint64_t total = 0;
    for (const auto& pin : pins) total += pin.n;
    peak_pinned = std::max(peak_pinned, total);
    return cudaSuccess;
}
cudaError_t stub_unregister(void* p) {
    auto it = std::find_if(pins.begin(), pins.end(), [p](const Pin& pin) { return pin.p == p; });
    if (it == pins.end()) return cudaErrorInvalidValue;
    pins.erase(it);
    ++unregisters;
    return cudaSuccess;
}
cudaError_t stub_alias(void** d, void* h, unsigned) {
    *d = attempts == alias_on && alias_on > 0 ? (uint8_t*) h + 4096 : h;
    return cudaSuccess;
}
cudaError_t stub_alloc(void** p, size_t n, unsigned) {
    *p = basalt::platform::alloc_aligned(n, (size_t) basalt::platform::vm_page_size());
    if (*p == nullptr) return cudaErrorMemoryAllocation;
    ++allocated;
    return cudaSuccess;
}
cudaError_t stub_free(void* p) { basalt::platform::free_aligned(p); --allocated; return cudaSuccess; }
cudaError_t stub_device(int* d) { *d = 0; return cudaSuccess; }
cudaError_t stub_attribute(int* v, cudaDeviceAttr, int) { *v = identity; return cudaSuccess; }
cudaError_t stub_error() { return cudaSuccess; }
}  // namespace

// Substitute only this translation unit's calls; no CUDA runtime entry point runs in this test.
#define cudaHostRegister stub_register
#define cudaHostUnregister stub_unregister
#define cudaHostGetDevicePointer stub_alias
#define cudaHostAlloc stub_alloc
#define cudaFreeHost stub_free
#define cudaGetDevice stub_device
#define cudaDeviceGetAttribute stub_attribute
#define cudaGetLastError stub_error
#include "../../src/core/kv_host_pool.cpp"
#undef cudaHostRegister
#undef cudaHostUnregister
#undef cudaHostGetDevicePointer
#undef cudaHostAlloc
#undef cudaFreeHost
#undef cudaGetDevice
#undef cudaDeviceGetAttribute
#undef cudaGetLastError

namespace c = basalt::core;
namespace k = basalt::kernels;
int bad = 0;
#define CHECK(x) do { if (!(x)) { std::printf("FAIL line %d: %s\n", __LINE__, #x); ++bad; } } while (0)

void run(bool int8, bool q4) {
    c::QsaState states[5];
    for (auto& st : states) { st.kv_int8 = int8; st.kv_q4 = q4; }
    c::SessionState ss;
    ss.qsa_states = states;
    ss.qsa_ord0 = 1;
    ss.qsa_alloc = 3;
    ss.max_cells = 16387;
    const c::ModelGeometry g{};
    std::string err;
    c::KvHostPool pool;
    attempts = fail_on = alias_on = 0;
    identity = 1;
    CHECK(pool.reserve(ss, g, 1001, err));
    CHECK(pool.grows_in_place() && pool.cells() == 1004);
    if (!pool.grows_in_place()) return;
    const auto arrays = c::arrays_of(pool.layer(0));
    const auto runs = c::page_runs(states[1], g);
    const uint64_t old_bytes = pool.bytes();
    const auto old_pins = pins;
    uint64_t pinned = 0;
    for (const auto& pin : pins) pinned += pin.n;
    CHECK(pinned == old_bytes);
    for (int64_t j = 0; j < pool.layers(); ++j) {
        const auto a = c::arrays_of(pool.layer(j));
        for (size_t i = 0; i < a.size(); ++i) std::memset(a[i], (int) (17 + j + i), (size_t) (251 * runs[i]));
    }
    fail_on = attempts + 5;
    CHECK(!pool.grow(12000, err));
    CHECK(pool.cells() == 1004 && pool.bytes() == old_bytes && same_pins(old_pins));
    fail_on = 0;
    alias_on = attempts + 3;
    CHECK(!pool.grow(12000, err));
    CHECK(pool.cells() == 1004 && pool.bytes() == old_bytes && same_pins(old_pins));
    alias_on = 0;
    const uint64_t delta = pool.growth_bytes(12000);
    peak_pinned = old_bytes;
    const int unregistered_before = unregisters;
    CHECK(pool.grow(12000, err));
    CHECK(pins.size() == old_pins.size());
    CHECK(unregisters - unregistered_before == (int) old_pins.size());
    CHECK(peak_pinned <= old_bytes + delta);
    for (const auto& old : old_pins) {
        const auto it = std::find_if(pins.begin(), pins.end(), [&](const Pin& pin) { return pin.p == old.p; });
        CHECK(it != pins.end() && it->n > old.n);
    }
    CHECK(pool.bytes() == old_bytes + delta && pool.cells() == 12000);
    CHECK(c::arrays_of(pool.layer(0)) == arrays);
    for (int64_t j = 0; j < pool.layers(); ++j) {
        const auto a = c::arrays_of(pool.layer(j));
        for (size_t i = 0; i < a.size(); ++i)
            for (uint64_t b = 0; b < 251 * runs[i]; ++b)
                if (a[i][b] != (uint8_t) (17 + j + i)) { CHECK(false); break; }
    }
    CHECK(pool.grow(16387, err) && pool.cells() == 16388);
    CHECK(!pool.grow(16389, err) && pool.cells() == 16388);
    const int calls = attempts;
    CHECK(pool.grow(12000, err) && attempts == calls);
    // A legacy allocation copies the same prefixes from a registered layout.
    identity = 0;
    c::KvHostPool copied;
    CHECK(copied.reserve(ss, g, 16387, err) && !copied.grows_in_place());
    CHECK(copied.copy_from(pool, 1001, err));
    for (int64_t j = 0; j < pool.layers(); ++j) {
        const auto a = c::arrays_of(pool.layer(j)), b = c::arrays_of(copied.layer(j));
        for (size_t i = 0; i < a.size(); ++i) CHECK(std::memcmp(a[i], b[i], (size_t) (251 * runs[i])) == 0);
    }
    c::KvHostPool moved(std::move(pool));
    CHECK(!pool.valid() && !pool.grows_in_place() && moved.grows_in_place());
    pool = std::move(moved);
    CHECK(!moved.valid() && pool.cells() == 16388);
    pool.release();
    copied.release();
    CHECK(pins.empty() && allocated == 0);
    // Initial registration or alias failure falls back after releasing the entire reservation.
    identity = 1;
    fail_on = attempts + 2;
    CHECK(pool.reserve(ss, g, 1001, err) && !pool.grows_in_place() && pins.empty());
    pool.release();
    fail_on = 0;
    alias_on = attempts + 2;
    CHECK(pool.reserve(ss, g, 1001, err) && !pool.grows_in_place() && pins.empty());
    pool.release();
    alias_on = 0;
    CHECK(allocated == 0);
    // A prefault writes the layout's pages and registers nothing; the take registers that layout in place.
    c::KvHostPool ahead;
    CHECK(!ahead.prefaulted_for(ss));
    CHECK(ahead.prefault(ss, g, 1001, err) && ahead.prefaulting() && !ahead.valid() && pins.empty());
    (void) ahead.prefault_wait();
    const uint64_t written = ahead.prefaulted_bytes();
    CHECK(written == ahead.prefault_bytes() && written > 0);   // the writer reached the committed total
    const auto laid = c::arrays_of(ahead.layer(0));
    c::KvHostPool taken(std::move(ahead));
    CHECK(!ahead.prefaulting() && taken.prefaulting());
    CHECK(taken.reserve(ss, g, 1001, err) && taken.valid() && !taken.prefaulting() && taken.cells() == 1004);
    CHECK(taken.grows_in_place() && c::arrays_of(taken.layer(0)) == laid && written == taken.bytes());
    pinned = 0;
    for (const auto& pin : pins) pinned += pin.n;
    CHECK(pinned == taken.bytes() && pins.size() == (size_t) taken.layers() * laid.size());
    taken.release();
    CHECK(pins.empty());
    // A prefault of other layers is laid out again; a refused pin falls back; a release stops the writer.
    CHECK(ahead.prefault(ss, g, 1001, err));
    c::SessionState other = ss;
    other.qsa_alloc = 2;
    CHECK(ahead.prefaulted_for(ss) && !ahead.prefaulted_for(other));
    CHECK(ahead.reserve(other, g, 1001, err) && ahead.valid() && ahead.layers() == 2 && !ahead.prefaulting());
    ahead.release();
    CHECK(ahead.prefault(ss, g, 1001, err));
    fail_on = attempts + 1;
    CHECK(ahead.reserve(ss, g, 1001, err) && !ahead.grows_in_place() && pins.empty() && allocated == 1);
    fail_on = 0;
    ahead.release();
    CHECK(ahead.prefault(ss, g, 16387, err));
    CHECK(ahead.reserve(ss, g, 1001, err) && ahead.cells() == 1004 && ahead.grow(12000, err) && ahead.cells() == 12000);
    ahead.release();
    CHECK(ahead.prefault(ss, g, 16387, err));
    ahead.release();
    CHECK(!ahead.prefaulting() && !ahead.valid() && pins.empty() && ahead.prefault_bytes() == 0);
    identity = 0;
    CHECK(!ahead.prefault(ss, g, 1001, err) && !ahead.prefaulting());
    identity = 1;
    CHECK(allocated == 0);
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--restore-failure") {
        c::QsaState st;
        st.kv_int8 = true;
        c::SessionState ss;
        ss.qsa_states = &st;
        ss.qsa_alloc = 1;
        ss.max_cells = 16384;
        c::KvHostPool pool;
        std::string err;
        if (!pool.reserve(ss, c::ModelGeometry{}, 1024, err) || !pool.grows_in_place()) return 0;
        fail_on = attempts + 2;
        restore_fail_on = attempts + 3;
        (void) pool.grow(12000, err);
        return 0;   // The failed restoration must stop the engine before this return.
    }
    if (argc != 1) return 1;
    run(true, false);
    run(false, true);
    run(false, false);
    std::printf("kv_host_pool_stub_test: %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
