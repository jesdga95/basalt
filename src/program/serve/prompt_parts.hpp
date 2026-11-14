// src/program/serve/prompt_parts.hpp - where a prompt's read stops for a checkpoint (header-only, no GPU).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace basalt::program::serve {

/// The chain's root of a prompt read from token 0: its first turn boundary below `below` (the end of the system prompt,
/// with the tools), when at least `min_tokens` precede it; -1 for none.  A boundary whose role is `reply_role` opens
/// the model's own reply: that is the new turn's header (a prompt with no system turn), which no other chat shares, so
/// it is no root and the read is not cut there.  `reply_role` -1: the template's turn token opens no reply.
inline int64_t root_boundary(const std::vector<int64_t>& ids, int64_t below, int64_t turn_token, int64_t reply_role,
                             int64_t min_tokens) {
    if (turn_token < 0 || min_tokens <= 0) return -1;
    for (int64_t i = 1; i < below && i < (int64_t) ids.size(); ++i)
        if (ids[(size_t) i] == turn_token) {
            const bool reply = reply_role >= 0 && i + 1 < (int64_t) ids.size() && ids[(size_t) i + 1] == reply_role;
            return i >= min_tokens && !reply ? i : -1;
        }
    return -1;
}

}  // namespace basalt::program::serve
