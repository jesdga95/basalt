// src/models/inkling2/fill_drain_test.cpp - a call's end and a card's projection fill (fill_drain.hpp): a joined fill needs
// nothing, whatever the stream's capture status; a pending one is waited for; a pending one on a capturing stream, or one
// whose capture status could not be read, is unconfirmed, never skipped.
#include "models/inkling2/fill_drain.hpp"

#include <cstdio>
#include <initializer_list>

using namespace basalt::models::inkling2;

namespace {
int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++failures;
    }
}
}  // namespace

int main() {
    for (const bool query_ok : {false, true})
        for (const bool capturing : {false, true}) expect(fill_drain(false, query_ok, capturing) == FillDrain::Done, "a joined fill");
    expect(fill_drain(true, true, false) == FillDrain::Wait, "a pending fill is waited for");
    expect(fill_drain(true, true, true) == FillDrain::Unconfirmed, "a pending fill under a capture");
    expect(fill_drain(true, false, false) == FillDrain::Unconfirmed, "a pending fill whose capture status failed");
    expect(fill_drain(true, false, true) == FillDrain::Unconfirmed, "a pending fill, both");
    if (failures) return 1;
    std::printf("inkling2_fill_drain_test: ok\n");
    return 0;
}
