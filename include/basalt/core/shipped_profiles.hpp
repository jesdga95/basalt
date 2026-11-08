// include/basalt/core/shipped_profiles.hpp - the expert profiles the engine ships beside its binary (data/profiles).
//
// A model names the profiles its pack has (a model whose pack has none names nothing); the start calls this before any
// weights load, so a binary copied away from its data/ folder stops with the path instead of loading with the
// ranking-less fallback (every expert, no ranking: more VRAM taken, experts pushed to the file).
#pragma once

#include "basalt/platform/file.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace basalt::core {

/// The error for the first of `names` (file names under `profile_dir`) that is missing or empty; empty when all are there.
inline std::string missing_shipped_profile(const std::string& profile_dir, const std::vector<std::string>& names) {
    for (const std::string& name : names) {
        const std::string path = profile_dir + "/" + name;
        uint64_t bytes = 0;
        if (!basalt::platform::file_size(path, bytes) || bytes == 0) {
            return "expert profile " + (profile_dir.empty() ? name : path) + " is missing: data/ must sit next to bin/ "
                   "(the engine reads data/profiles from the directory above its own executable), so run the binary "
                   "from its install folder, or pass --expert-profile";
        }
    }
    return {};
}

}  // namespace basalt::core
