#include "basalt/core/conversation_memory.hpp"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <sstream>


using namespace basalt::core;
namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
}

int main() {
    for (const char* text : {"", "MemFree: 100 kB\n", "MemAvailable: -1 kB\n",
                            "MemAvailable: +1 kB\n", "MemAvailable: 1 MB\n",
                            "MemAvailable: 1\n", "MemAvailable: 1x kB\n",
                            "MemAvailable: 18446744073709551615 kB\n",
                            "MemAvailable: 18446744073709551616 kB\n",
                            "MemAvailable: 1 kB trailing\n",
                            "MemAvailable: 1 kB\nMemAvailable: 2 kB\n"}) {
        std::istringstream input(text);
        check(!conversation_mem_available(input), "missing/malformed telemetry is unknown");
    }
    std::istringstream normal("MemTotal: 999999 kB\nMemAvailable:    12345 kB\nSwapFree: 777 kB\n");
    check(conversation_mem_available(normal) == 12345ULL * 1024, "only MemAvailable is counted");
    std::istringstream zero("MemAvailable: 0 kB");
    check(conversation_mem_available(zero) == 0, "zero available memory is known");
    std::istringstream broken("MemAvailable: 123 kB\n");
    broken.setstate(std::ios::badbit);
    check(!conversation_mem_available(broken), "I/O failure declines admission");
    check(!conversation_memory_admit({}, 0, 0), "unknown fails closed even with zero floor");
    check(conversation_memory_admit(100, 40, 60), "exact allocation plus floor fits");
    check(!conversation_memory_admit(99, 40, 60), "one byte below required memory rejected");
    check(!conversation_memory_admit(59, 0, 60), "floor subtraction cannot underflow");
    check(conversation_memory_admit(60, 0, 60), "post-capture floor check");
    check(!conversation_memory_admit(100, std::numeric_limits<uint64_t>::max(), 1), "allocation arithmetic cannot overflow");
    check(conversation_memory_admit(std::numeric_limits<uint64_t>::max(),
                                   std::numeric_limits<uint64_t>::max(), 0), "maximal exact bound");
    check(!conversation_memory_admit(std::numeric_limits<uint64_t>::max(),
                                    std::numeric_limits<uint64_t>::max(), 1), "maximal sum overflow rejected");
    // Test the real provider without assuming any particular amount of free RAM.
    const auto available = conversation_available_memory();
    check(!available || conversation_memory_admit(available, 0, 0), "provider returns bytes or unknown");
    // The provider must yield a reading wherever the test runs: an unknown reading disables
    // parking, so a probe dead on this OS fails here rather than silently in a serve run.
    check(available.has_value(), "provider returns a reading on this OS");
    // Admit reads only the bytes it is handed, so the same values decide the same on both OSes.
    // Push this OS's own reading through the park arithmetic at its exact bounds; those bounds
    // checks read the value, so they run only when the provider gave one.
    if (available) {
        const uint64_t live = *available;
        check(conversation_memory_admit(live, live, 0), "a park as large as the reading fits");
        check(!conversation_memory_admit(live, live, 1), "one byte of floor excludes that park");
    }
    constexpr uint64_t avail = 10ull << 30, park = 8ull << 30, floor = 2ull << 30;
    check(conversation_memory_admit(avail, park, floor), "fixed numbers admit park plus floor");
    check(!conversation_memory_admit(avail - 1, park, floor), "the same numbers reject one byte lower");
    {   // the verdict names the check that failed, with the numbers that decided
        constexpr uint64_t MiB = 1ull << 20;
        const Admission none = conversation_admission(true, 4096 * MiB, {}, 100 * MiB, 4096 * MiB);
        check(none.check == AdmitCheck::NoTelemetry && !none &&
              conversation_refusal(none) == "host RAM telemetry unavailable", "no telemetry says so");
        const Admission guard = conversation_admission(false, 4096 * MiB, 3000 * MiB, 100 * MiB, 4096 * MiB);
        check(guard.check == AdmitCheck::BelowGuard &&
              conversation_refusal(guard) == "host free 3000 MiB is below the RAM guard's 4096 MiB floor", "guard below its floor");
        const Admission shortage = conversation_admission(true, 4096 * MiB, 8000 * MiB, 6492 * MiB, 4096 * MiB);
        check(shortage.check == AdmitCheck::Short && shortage.available == 8000 * MiB &&
              conversation_refusal(shortage) == "host free 8000 MiB < required 10588 MiB (need 6492 MiB + floor 4096 MiB)",
              "short of need plus floor names the free RAM and the required");
        check((bool) conversation_admission(true, 4096 * MiB, 10588 * MiB, 6492 * MiB, 4096 * MiB), "exactly enough admits");
    }
    {   // one episode per closed stretch of admission; the total survives it
        RefusalEpisodes e;
        check(e.admitted() == 0, "no episode open, nothing to end");
        check(e.refused() && !e.refused() && !e.refused(), "only the first refusal opens the episode");
        check(e.admitted() == 3 && e.admitted() == 0, "the episode ends with its refusal count, once");
        check(e.refused() && e.total() == 4, "a new episode opens; the total counts all");
    }
    std::printf("conversation_memory_test: %d checks passed\n", checks);
}
