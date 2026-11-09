// No CUDA or model pack required. Compile with -std=c++20.
#include "../../src/program/serve/thread_tape.hpp"
#include <cassert>
#include <vector>
using basalt::program::serve::NativeThreadTapes;
int main() {
    NativeThreadTapes store(2);
    const std::vector<int32_t> prefix{1,2,3};
    assert(store.attach("thread-a","head-1",prefix,100,1000).ok);
    assert(store.attach("thread-a","head-1",prefix,100,1000).ok);
    assert(!store.attach("thread-a","wrong",prefix,100,1000).ok);
    assert(!store.append("thread-a","stale","head-2",{4},100,1000).ok);
    assert(store.append("thread-a","head-1","head-2",{4,5},100,1000).ok);
    const auto* ids=store.get("thread-a","head-2");
    assert(ids!=nullptr && *ids==std::vector<int32_t>({1,2,3,4,5}));
    assert(store.get("thread-a","head-1")==nullptr);
    assert(!store.append("thread-a","head-2","head-3",{1000},100,1000).ok);
    assert(!store.append("thread-a","head-2","head-3",{6},6,1000).ok);
    assert(store.size()==1);
    assert(store.attach("thread-b","head-1",{1},100,1000).ok);
    assert(!store.attach("thread-c","head-1",{1},100,1000).ok);
    assert(store.release("thread-a"));
    assert(!store.release("thread-a"));
    assert(store.attach("thread-c","head-1",{1},100,1000).ok);
    assert(!NativeThreadTapes::valid_name("x/y"));
    return 0;
}
