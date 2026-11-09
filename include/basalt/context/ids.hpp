// Shared vocabulary for agent context sharing. Every other basalt/context header builds on these types.
// Contract: docs/architecture/agent-context-contract.md. Host-only; no CUDA headers.
#pragma once
#include <array>
#include <cstdint>
#include <string>

namespace basalt::context {

using Digest = std::array<uint8_t, 32>;          // SHA-256 (domain-separated by the producer)
using ContextId = std::string;                   // opaque, [A-Za-z0-9_-]{1,128}; never a slot number
using Generation = uint64_t;                     // fencing token; strictly increases per engine epoch / writer lease

enum class Tier : uint8_t { gpu = 0, host = 1, disk = 2, missing = 3 };

// Mirrors the PostgreSQL context FSM: RECORDED -> MATERIALIZING -> HOT -> COOLING -> WARM -> COLD,
// RESTORING back to HOT, FAILED terminal-recoverable (transcript is never lost).
enum class ContextState : uint8_t { recorded, materializing, hot, cooling, warm, cold, restoring, failed };

inline bool valid_context_id(const std::string& s) {
    if (s.empty() || s.size() > 128) return false;
    for (unsigned char c : s)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_'))
            return false;
    return true;
}
inline const char* tier_name(Tier t) {
    switch (t) { case Tier::gpu: return "GPU"; case Tier::host: return "HOST"; case Tier::disk: return "DISK"; default: return "MISSING"; }
}
}  // namespace basalt::context
