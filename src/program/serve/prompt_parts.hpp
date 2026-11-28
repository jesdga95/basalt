// src/program/serve/prompt_parts.hpp - where a prompt's read stops for a checkpoint (header-only, no GPU).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
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

/// The checkpoints a batch slot keeps at admission, moved out of `checks`: the conversation's shortest (the system prompt,
/// which other requests may share) and longest (its next turn).  Includes a checkpoint whose length equals `live.size()`,
/// which is where `in_slot` admissions leave `live` when the turn checkpoint sits at `n - 1` before any decode step runs.
/// Moved, not copied: a checkpoint is the model's whole running state, and the admission waits for its copy.
template <class Checkpoint, class IdsEqual>
void select_slot_checkpoints(std::vector<Checkpoint>& checks, const std::vector<int32_t>& live, IdsEqual ids_equal,
                             std::vector<Checkpoint>& out) {
    size_t best = checks.size(), root = checks.size();
    for (size_t i = 0; i < checks.size(); ++i) {
        const Checkpoint& c = checks[i];
        if (!c.ids.empty() && c.ids.size() <= live.size() && ids_equal(c.ids.data(), c.ids.size(), live.data())) {
            if (best == checks.size() || c.ids.size() > checks[best].ids.size()) best = i;
            if (root == checks.size() || c.ids.size() < checks[root].ids.size()) root = i;
        }
    }
    if (best == checks.size()) return;
    if (root != best) out.push_back(std::move(checks[root]));
    out.push_back(std::move(checks[best]));
    checks.erase(checks.begin() + (std::ptrdiff_t) std::max(root, best));   // the later one first: the earlier stays put
    if (root != best) checks.erase(checks.begin() + (std::ptrdiff_t) std::min(root, best));
}

/// Moves an idle slot's checkpoints back into the solo chain without duplicating checkpoints already held there.
template <class Checkpoint, class IdsEqual>
void merge_slot_checkpoints(std::vector<Checkpoint>& src, std::vector<Checkpoint>& dst, IdsEqual ids_equal) {
    for (Checkpoint& c : src) {
        const bool have = std::any_of(dst.begin(), dst.end(), [&](const Checkpoint& k) {
            return k.ids.size() == c.ids.size() && k.imgs == c.imgs &&
                   ids_equal(k.ids.data(), k.ids.size(), c.ids.data());
        });
        if (!have) dst.push_back(std::move(c));
    }
    src.clear();
}

}  // namespace basalt::program::serve
