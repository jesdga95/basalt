// tests/spec/draft_cum_test.cpp - the cumulative draft-length rule (include/basalt/core/draft_cum.hpp).
//
// Checks that
//   1. the threshold is kDraftCumTh unless --exp draft_th=V sets another level, and V <= 0 leaves the default;
//   2. a draft stays while the bin-table product of it and every draft before it is at least the threshold, and the
//      products on either side of one half fall on the right side (float, as the host and the device compute them);
//   3. the window width a run of drafts gets does not depend on anything but those drafts (no per-request state).
#include "basalt/core/draft_cum.hpp"

#include <cstdio>

using namespace basalt::core;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-70s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
int width(const float* probs, int max_t, float th) {   // the serve loop's window choice
    int T = 1;
    while (T < max_t && draft_keep(probs, T - 1, th)) ++T;
    return T;
}
}  // namespace

int main() {
    std::printf("draft_cum_test\n");
    check(draft_cum_th() == kDraftCumTh && kDraftCumTh == 0.50f, "the threshold is the constant one half");
    diagnostics().exp["draft_th"] = 0.40;
    check(draft_cum_th() == 0.40f, "--exp draft_th=0.40 sets the level");
    diagnostics().exp["draft_th"] = 0.0;
    check(draft_cum_th() == kDraftCumTh, "--exp draft_th=0 leaves the constant");
    diagnostics().exp.erase("draft_th");

    const float th = draft_cum_th();
    {
        const float p[] = {0.75f};   // 0.51
        check(draft_keep(p, 0, th), "a first draft in [0.70, 0.80) is proposed (0.51)");
    }
    {
        const float p[] = {0.65f};   // 0.43
        check(!draft_keep(p, 0, th), "a first draft in [0.50, 0.70) is not (0.43)");
    }
    {
        const float p[] = {0.995f, 0.75f};   // 0.96 * 0.51 = 0.4896
        check(draft_keep(p, 0, th) && !draft_keep(p, 1, th), "0.96 then 0.51: the second draft stops (0.4896)");
    }
    {
        const float p[] = {0.97f, 0.92f, 0.995f};   // 0.79 * 0.66 * 0.96 = 0.50054
        check(draft_keep(p, 2, th), "0.79 * 0.66 * 0.96 stays (0.50054)");
    }
    {
        const float p[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};   // 0.96^j stays above one half through j = 7
        check(width(p, 8, th) == 8, "seven near-certain drafts fill a window of 8");
    }
    {
        const float p[] = {0.995f, 0.995f, 0.85f, 0.995f, 0.1f, 0.1f, 0.1f};   // 0.96, 0.9216, 0.5437, 0.522, 0.063
        const int a = width(p, 8, th), b = width(p, 8, th);
        check(a == 5 && a == b, "the width is a function of the drafts alone");
    }
    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
