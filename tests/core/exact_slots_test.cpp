// tests/core/exact_slots_test.cpp - the sizing of an exact --expert-cache: per-blob slot counts and the refusal text
// (no GPU).
#include "basalt/core/exact_slots.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-84s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
}  // namespace

int main() {
    using basalt::core::blobs_within;
    // blobs of 1000 B take a 1024 B stride, blobs of 2560 B take 2560 B
    const std::vector<int64_t> mixed = {1000, 2560, 1000, 1000, 2560};
    check(blobs_within(mixed, 0) == 0, "no room: no slot");
    check(blobs_within(mixed, 1023) == 0, "room one byte short of the first padded blob: no slot");
    check(blobs_within(mixed, 1024) == 1, "room for exactly the first padded blob: one slot");
    check(blobs_within(mixed, 1024 + 2560 + 1024) == 3, "room for the first three padded blobs: three slots");
    check(blobs_within(mixed, 1ull << 40) == 5, "more room than blobs: every blob");
    check(blobs_within({}, 1ull << 20) == 0, "no blobs: no slot");

    // an exact count against the room: the room holds fewer than asked, so the start is refused and the text says why
    const int64_t fit = blobs_within(mixed, 1024 + 2560 + 1024);
    check(fit < 5, "five asked, the room holds fewer: refused, not shrunk");
    const std::string m = basalt::core::exact_slots_refusal(5, fit, "", 3ull << 30, 1024);
    check(m == "--expert-cache 5: only 3 fit (3.00 GiB free, 1024 MiB reserve); use a smaller value or auto",
          "the refusal is one line: the value, what fits, the free VRAM, the reserve, the way out");
    check(basalt::core::exact_slots_refusal(5, 2, " per layer", 0, 0).find("only 2 per layer fit") != std::string::npos,
          "a per-layer count says so");
    check(basalt::core::exact_slots_refusal(5, -2, "", 0, 0).find("only 0 fit") != std::string::npos,
          "a negative fit prints as 0");

    std::printf(g_fail == 0 ? "exact_slots_test: all ok\n" : "exact_slots_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
