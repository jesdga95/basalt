// Host-only test for NativeThreadTapes::fork / shared_prefix.
#include <cstdio>
#include <cstdlib>
#include "../../src/program/serve/thread_tape.hpp"

using basalt::program::serve::NativeThreadTapes;

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); std::exit(1); } } while (0)

int main() {
    NativeThreadTapes t(4);
    CHECK(t.attach("p", "h0", {1, 2, 3, 4, 5}, 100, 1000).ok);
    // whole-tape fork shares the full prefix
    CHECK(t.fork("p", "h0", "c1", "k0", 0).ok);
    CHECK(t.shared_prefix("p", "c1") == 5);
    // idempotent retry, conflicting retry
    CHECK(t.fork("p", "h0", "c1", "k0", 0).ok);
    CHECK(!t.fork("p", "h0", "c1", "k0", 3).ok);
    // fork at a point; child diverges after append
    CHECK(t.fork("p", "h0", "c2", "m0", 3).ok);
    CHECK(t.append("c2", "m0", "m1", {9, 9}, 100, 1000).ok);
    CHECK(t.shared_prefix("p", "c2") == 3);
    // parent untouched by child writes
    CHECK(t.get("p", "h0") && t.get("p", "h0")->size() == 5);
    // stale parent head, bad point, unknown parent, self fork, capacity
    CHECK(!t.fork("p", "stale", "c3", "n0", 0).ok);
    CHECK(!t.fork("p", "h0", "c3", "n0", 6).ok);
    CHECK(!t.fork("nope", "h0", "c3", "n0", 0).ok);
    CHECK(!t.fork("p", "h0", "p", "n0", 0).ok);
    CHECK(t.fork("p", "h0", "c3", "n0", 0).ok);
    CHECK(!t.fork("p", "h0", "c4", "n0", 0).ok);  // 4 tapes = capacity
    // releasing the parent leaves children valid
    CHECK(t.release("p"));
    CHECK(t.get("c1", "k0") && t.get("c1", "k0")->size() == 5);
    std::puts("thread_fork_test OK");
    return 0;
}
