#include "basalt/core/conversation_memory.hpp"

#include "basalt/platform/memory.hpp"

#include <charconv>
#include <fstream>
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
#if defined(_WIN32)
    // No /proc/meminfo here: take the platform probe's physical-RAM reading, in bytes. It reports
    // 0 when the query gives no reading, which stays unknown so admission fails closed.
    const uint64_t avail = basalt::platform::available_memory();
    if (avail == 0) return {};
    return avail;
#else
    std::ifstream meminfo("/proc/meminfo");
    if (!meminfo) return {};
    return conversation_mem_available(meminfo);
#endif
}

} // namespace basalt::core
