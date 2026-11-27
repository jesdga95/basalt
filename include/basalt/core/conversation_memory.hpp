#pragma once

#include <cstdint>
#include <istream>
#include <mutex>
#include <optional>
#include <string>

namespace basalt::core {

// Available RAM in bytes, capped by any visible Linux cgroup memory limit (matching platform::available_memory()).
// Unknown telemetry is deliberately distinct from a measured zero.
std::optional<uint64_t> conversation_available_memory();
std::optional<uint64_t> conversation_mem_available(std::istream& meminfo);

inline bool conversation_memory_admit(std::optional<uint64_t> available,
                                      uint64_t allocation, uint64_t floor) {
    return available && *available >= floor && allocation <= *available - floor;
}

/// Which check refused a capture of `need` bytes, with the host RAM available when it was judged.
enum class AdmitCheck { Admitted, NoTelemetry, BelowGuard, Short };

struct Admission {
    AdmitCheck check = AdmitCheck::Admitted;
    uint64_t available = 0;    // host RAM available at the decision (0 with no telemetry)
    uint64_t need = 0;         // the capture's bytes
    uint64_t floor = 0;        // what stays free after it
    uint64_t guard_floor = 0;  // the RAM guard's floor
    explicit operator bool() const { return check == AdmitCheck::Admitted; }
};

/// The verdict on a capture: no telemetry, the RAM guard below its floor (`guard_ok` false), or less free than need + floor.
Admission conversation_admission(bool guard_ok, uint64_t guard_floor, std::optional<uint64_t> available,
                                 uint64_t need, uint64_t floor);

/// The refused verdict in words, with the number that decided: "host free 1228 MiB < required 8192 MiB (need
/// 4096 MiB + floor 4096 MiB)".
std::string conversation_refusal(const Admission& a);

/// Counts the refusals of RAM admission and groups them into episodes: one that opens when admission closes and
/// ends when a capture is admitted again, so a log says each change once.
class RefusalEpisodes {
public:
    /// A refusal; true when it opens an episode.
    bool refused() {
        std::lock_guard<std::mutex> lock(mu_);
        ++total_;
        return ++in_episode_ == 1;
    }
    /// A capture was admitted; the refusals of the episode it ends, 0 when none was open.
    uint64_t admitted() {
        std::lock_guard<std::mutex> lock(mu_);
        const uint64_t n = in_episode_;
        in_episode_ = 0;
        return n;
    }
    /// The refusals since the engine started.
    uint64_t total() const {
        std::lock_guard<std::mutex> lock(mu_);
        return total_;
    }

private:
    mutable std::mutex mu_;
    uint64_t total_ = 0;
    uint64_t in_episode_ = 0;
};

} // namespace basalt::core
