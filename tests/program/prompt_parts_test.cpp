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
    std::printf("%s\n", failures ? "FAILED" : "all ok");
    return failures ? 1 : 0;
}
