#include "basalt/core/conversation_memory.hpp"

#include "basalt/platform/memory.hpp"
#include "basalt/platform/system.hpp"

#include <charconv>
#include <limits>
#include <sstream>
#include <string>


namespace basalt::core {

std::optional<uint64_t> conversation_mem_available(std::istream& meminfo) {
    std::optional<uint64_t> result;
    std::string line;
    while (std::getline(meminfo, line)) {
        std::istringstream fields(line);
        std::string key, value, unit, extra;
        if (!(fields >> key) || key != "MemAvailable:") continue;
        if (result || !(fields >> value >> unit) || unit != "kB" || (fields >> extra)) return {};
        uint64_t kb = 0;
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), kb);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
            kb > std::numeric_limits<uint64_t>::max() / 1024) return {};
        result = kb * 1024;
    }
    if (meminfo.bad() || (meminfo.fail() && !meminfo.eof())) return {};
    return result;
}

std::optional<uint64_t> conversation_available_memory() {
    basalt::platform::HostMemory m;
    if (basalt::platform::host_available_memory(m)) return m.available;
    const std::string path = basalt::platform::meminfo_path();
    if (path.empty()) {
        // No memory accounting file here: take the platform probe's available-RAM reading, in bytes. It reports 0
        // when the query gives no reading, which stays unknown so admission fails closed.
        const uint64_t avail = basalt::platform::available_memory();
        if (avail == 0) return {};
        return avail;
    }
    return {};
}

Admission conversation_admission(bool guard_ok, uint64_t guard_floor, std::optional<uint64_t> available,
                                 uint64_t need, uint64_t floor) {
    Admission a;
    a.available = available.value_or(0);
    a.need = need;
    a.floor = floor;
    a.guard_floor = guard_floor;
    if (!available) a.check = AdmitCheck::NoTelemetry;
    else if (!guard_ok) a.check = AdmitCheck::BelowGuard;
    else if (!conversation_memory_admit(available, need, floor)) a.check = AdmitCheck::Short;
    return a;
}

std::string conversation_refusal(const Admission& a) {
    const auto mib = [](uint64_t b) { return std::to_string(b >> 20); };
    switch (a.check) {
    case AdmitCheck::NoTelemetry:
        return "host RAM telemetry unavailable";
    case AdmitCheck::BelowGuard:
        return "host free " + mib(a.available) + " MiB is below the RAM guard's " + mib(a.guard_floor) + " MiB floor";
    case AdmitCheck::Short:
        return "host free " + mib(a.available) + " MiB < required " + mib(a.need + a.floor) + " MiB (need " +
               mib(a.need) + " MiB + floor " + mib(a.floor) + " MiB)";
    case AdmitCheck::Admitted:
        break;
    }
    return "admitted";
}

} // namespace basalt::core
