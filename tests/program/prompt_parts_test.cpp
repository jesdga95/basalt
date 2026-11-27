// tests/program/prompt_parts_test.cpp - where a prompt read from token 0 places the chain's root (header-only, CPU).
#include "program/serve/prompt_parts.hpp"

#include <cstdio>
#include <vector>

using basalt::program::serve::root_boundary;

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}
constexpr int64_t kTurn = 9, kSystem = 1, kUser = 2, kReply = 3, kText = 5;
// a turn: the turn token, its role, `len` tokens of text
void turn(std::vector<int64_t>& ids, int64_t role, int len) {
    ids.push_back(kTurn);
    ids.push_back(role);
    ids.insert(ids.end(), (size_t) len, kText);
}
}  // namespace

int main() {
    // Qwen: the turn token opens every turn, the reply's too
    std::vector<int64_t> chat;
    turn(chat, kSystem, 3000);
    const int64_t user_at = (int64_t) chat.size();
    turn(chat, kUser, 500);
    const int64_t reply_at = (int64_t) chat.size();
    turn(chat, kReply, 4);
    const int64_t n = (int64_t) chat.size();
    check(root_boundary(chat, n - 1, kTurn, kReply, 2048) == user_at, "a system prompt: the root at its end, read from the last token");
    check(root_boundary(chat, reply_at, kTurn, kReply, 2048) == user_at, "a system prompt: the root at its end, read to the reply header");
    check(root_boundary(chat, n - 1, kTurn, kReply, 4096) == -1, "a system prompt shorter than the minimum: no root");
    check(root_boundary(chat, n - 1, kTurn, kReply, 0) == -1, "a minimum of 0: never a root");

    std::vector<int64_t> lone;   // a user turn and the reply's header, no system turn
    turn(lone, kUser, 5000);
    turn(lone, kReply, 4);
    check(root_boundary(lone, (int64_t) lone.size() - 1, kTurn, kReply, 2048) == -1,
          "no system turn: the reply's header is no root");
    check(root_boundary(lone, (int64_t) lone.size() - 1, kTurn, -1, 2048) == 5002,
          "a template whose turn token opens no reply: the first boundary is the root");

    // DeepSeek: the turn token opens the user's turn only, so a first chat's root is the system prompt's end
    std::vector<int64_t> ds(3000, kText);
    ds.push_back(kTurn);
    ds.insert(ds.end(), 400, kText);
    ds.push_back(kReply);
    check(root_boundary(ds, (int64_t) ds.size() - 1, kTurn, -1, 2048) == 3000, "a user-turn token: the root at the system prompt's end");

    // Batch-slot checkpoint admission: an in_slot admission leaves live at n - 1 (no decode step yet), which equals
    // the turn checkpoint taken at n - 1; the slot must keep that checkpoint so a later thinking-budget wrap-up matches.
    struct FakeCk {
        std::vector<int32_t> ids;
        std::vector<int32_t> imgs;
    };
    auto eq = [](const int32_t* a, size_t n_ids, const int32_t* b) {
        return std::equal(a, a + n_ids, b);
    };
    {
        const std::vector<int32_t> live = {1, 2, 3, 4, 5};
        std::vector<FakeCk> checks = {
            {{1, 2}, {}},
            {{1, 2, 3, 4, 5}, {}},        // turn checkpoint at live.size() == n - 1
            {{1, 2, 3, 4, 5, 6}, {}},     // longer than live: ignored
            {{1, 9, 3}, {}},              // different branch: ignored
        };
        std::vector<FakeCk> kept;
        basalt::program::serve::select_slot_checkpoints(checks, live, eq, kept);
        check(kept.size() == 2 && kept[0].ids.size() == 2 && kept[1].ids.size() == 5,
              "in_slot admission keeps both root and turn checkpoint at live.size()");
        check(checks.size() == 2 && checks[0].ids.size() == 6 && checks[1].ids.size() == 3,
              "the kept checkpoints leave the chain (moved, not copied); the others stay in order");
    }
    {
        const std::vector<int32_t> live = {1, 2, 3, 4, 5};
        std::vector<FakeCk> checks = {{{1, 2, 3, 4, 5}, {}}};
        std::vector<FakeCk> kept;
        basalt::program::serve::select_slot_checkpoints(checks, live, eq, kept);
        check(kept.size() == 1 && kept[0].ids.size() == 5,
              "single turn checkpoint at live.size() is kept once without root duplication");
        check(checks.empty(), "a lone kept checkpoint leaves the chain once");
    }
    {
        std::vector<FakeCk> dst = {{{1, 2}, {}}, {{1, 2, 3, 4}, {}}};
        std::vector<FakeCk> src = {{{1, 2}, {}}, {{1, 2, 3, 4}, {7}}, {{1, 2, 3, 4, 5}, {}}};
        basalt::program::serve::merge_slot_checkpoints(src, dst, eq);
        check(src.empty() && dst.size() == 4 && dst[2].imgs == std::vector<int32_t>{7} && dst[3].ids.size() == 5,
              "merging idle slot 0 checkpoints back skips exact duplicates and preserves distinct checkpoints");
    }
    std::printf("%s\n", failures ? "FAILED" : "all ok");
    return failures ? 1 : 0;
}
