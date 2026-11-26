// tests/model/batch_window_test.cpp - several sessions' windows as one window of segments: rows, positions, the device
// segment table and a row's segment, the refusals, and the composition graphs' least recently used eviction (CPU).
#include "basalt/model/batch_window.hpp"

#include <cstdio>
#include <memory>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}
}  // namespace

int main() {
    using basalt::model::BatchWindow;
    using basalt::model::DecodeRequest;
    namespace bk = basalt::kernels;
    const int32_t a[3] = {11, 12, 13}, b[1] = {21}, c[4] = {31, 32, 33, 34};
    DecodeRequest r[3];
    r[0].tokens = a, r[0].n_tokens = 3, r[0].pos = 100;
    r[1].tokens = b, r[1].n_tokens = 1, r[1].pos = 7;
    r[2].tokens = c, r[2].n_tokens = 4, r[2].pos = 101;
    BatchWindow w;
    std::string err;
    check(w.build(r, 3, 8, 4096, err), "three windows of 8 rows in all build");
    check(w.rows == 8 && w.segs.size() == 3 && w.segs[1].row0 == 3 && w.segs[2].row0 == 4 && w.segs[2].pos0 == 101, "segments in order");
    check(w.tokens == std::vector<int32_t>({11, 12, 13, 21, 31, 32, 33, 34}), "tokens in row order");
    check(w.positions == std::vector<int32_t>({100, 101, 102, 7, 101, 102, 103, 104}), "each row its own position");
    check(w.max_pos == 104, "the last position");
    const bk::WindowSegs s = w.device_segs();
    check(s.n == 3 && s.row0[0] == 0 && s.row0[1] == 3 && s.row0[2] == 4 && s.row0[3] == 8, "the device table");
    const int seg[8] = {0, 0, 0, 1, 2, 2, 2, 2}, first[8] = {0, 0, 0, 3, 4, 4, 4, 4};
    bool ok = true;
    for (int i = 0; i < 8; ++i) ok = ok && bk::seg_of(s, i) == seg[i] && bk::seg_first(s, i) == first[i];
    check(ok, "each row's segment and first row");
    const bk::WindowSegs solo;
    check(bk::seg_of(solo, 5) == 0 && bk::seg_first(solo, 5) == 0, "no table: one conversation");
    check(!w.build(r, 3, 7, 4096, err), "more rows than the cap refused");
    check(!w.build(r, 3, 8, 104, err), "a window past the context refused");
    r[1].n_tokens = 0;
    check(!w.build(r, 3, 8, 4096, err), "an empty window refused");

    basalt::model::GraphLru<std::unique_ptr<int>> lru(2);
    lru.insert({1}, std::make_unique<int>(1));
    lru.insert({2}, std::make_unique<int>(2));
    check(lru.find({1}) && **lru.find({1}) == 1, "a composition found");
    lru.insert({3}, std::make_unique<int>(3));   // {2} was used longest ago
    check(lru.size() == 2 && lru.find({2}) == nullptr && lru.find({1}) && lru.find({3}), "the least recently used goes");
    return failures ? 1 : 0;
}
