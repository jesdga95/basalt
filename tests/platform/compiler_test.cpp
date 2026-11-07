// tests/platform/compiler_test.cpp - the compiler-facing helpers in basalt/platform/compiler.hpp, and the
// std::atomic_ref max-update the verifier uses for its flags. CPU only.
#include "basalt/platform/compiler.hpp"

#include <atomic>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("compiler %-58s -> %s\n", what.c_str(), ok ? "ok" : "FAILED   <-- WRONG");
    failures |= !ok;
}
}  // namespace

int main() {
    using namespace basalt::platform;
    unsigned r[4] = {};
    cpuid(0, 0, r);
    if constexpr (kX86) {
        char vendor[13] = {};
        std::memcpy(vendor, &r[1], 4);
        std::memcpy(vendor + 4, &r[3], 4);
        std::memcpy(vendor + 8, &r[2], 4);
        std::printf("compiler: cpuid vendor %s, highest leaf %u\n", vendor, r[0]);
        check(r[0] >= 1 && vendor[0] != 0, "cpuid leaf 0 names a vendor");
        cpuid(1, 0, r);
        if (((r[2] >> 27) & 1u) != 0) check((xgetbv0() & 0x1) != 0, "XCR0 reports x87 state when OSXSAVE is set");
    } else {
        check(r[0] == 0 && r[1] == 0 && r[2] == 0 && r[3] == 0 && xgetbv0() == 0, "cpuid and xgetbv read as no x86 features");
        check(!cpu_has_avx512_bf16(), "no AVX512_BF16 off x86");
    }

    check(ctz32(1) == 0 && ctz32(0x80000000u) == 31 && ctz32(0x00f0u) == 4 && ctz32(0xffffffffu) == 0, "ctz32");
    for (int i = 0; i < 4; ++i) cpu_relax();
    check(true, "cpu_relax returns");
    store_fence();
    check(true, "store_fence returns");

    check(f16_round(1.0f) == 1.0f && f16_round(-2.5f) == -2.5f && f16_round(0.0f) == 0.0f, "f16_round exact values");
    check(f16_round(0.1f) == 0.0999755859375f, "f16_round rounds 0.1 to the nearest half");
    check(f16_round(65519.0f) == 65504.0f, "f16_round keeps the largest finite half");
    check(f16_round(1.00048828125f) == 1.0f && f16_round(1.0014648f) == 1.0019531f, "f16_round ties to even");

    (void) cpu_has_avx512_bf16();
    check(true, "cpu_has_avx512_bf16 answers");

    check(std::bit_cast<float>(0x3f800000u) == 1.0f && std::bit_cast<uint32_t>(2.0f) == 0x40000000u, "std::bit_cast");

    // the flag only rises: concurrent raisers leave the largest value
    uint32_t flag = 0;
    std::vector<std::thread> ts;
    for (uint32_t t = 0; t < 4; ++t)
        ts.emplace_back([&flag, t] {
            for (uint32_t v = 1; v <= 20000; ++v) {
                std::atomic_ref<uint32_t> f(flag);
                uint32_t cur = f.load();
                const uint32_t value = v * 4 + t;
                while (cur < value && !f.compare_exchange_strong(cur, value)) {}
            }
        });
    for (auto& t : ts) t.join();
    check(flag == 20000u * 4 + 3, "an atomic_ref max-update ends at the largest value");
    return failures ? 1 : 0;
}
