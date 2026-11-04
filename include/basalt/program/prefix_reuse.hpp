#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace basalt::program::prefix_reuse {

// How many leading prompt tokens a session that holds `held` can keep. All of `held` must be a prefix of `toks`, and
// at least one token must remain: the last prompt token is decoded to produce the first answer token. Anything else
// starts from an empty sequence, because the compressor state cannot rewind to an arbitrary earlier position.
inline size_t reusable(const std::vector<int32_t>& held, const std::vector<int32_t>& toks) {
    if (held.empty() || held.size() >= toks.size()) return 0;
    for (size_t i = 0; i < held.size(); ++i)
        if (held[i] != toks[i]) return 0;
    return held.size();
}

}  // namespace basalt::program::prefix_reuse
