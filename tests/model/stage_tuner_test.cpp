// tests/model/stage_tuner_test.cpp - the stage cut that follows a read: a read the stage card is the long pole of moves the cut
// toward the main card, a balanced read and matched cards keep it, the cut never gives the stage more than the shares did,
// a short read or a given share changes nothing.
#include "basalt/model/stage_tuner.hpp"

#include <algorithm>
#include <cstdio>

using basalt::model::StageTuner;

static int g_fail = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

// what a read of `chunks` chunks measures when the main card costs `a` and the stage card `b` ms per layer and chunk
static void measure(const StageTuner& t, int cut, int end, double a, double b, int chunks, double& total, double& stage) {
    total = t.read_ms(end - cut, a, b, chunks);
    stage = (double) chunks * (double) (end - cut) * b;
}

int main() {
    const int end = 48, chunks = 8;
    // a stage card 4x slower per layer: the balanced cut leaves it about a fifth of the layers
    {
        StageTuner t(end, 32, true);   // the shares gave it 16 layers
        double total, stage;
        measure(t, t.cut(), end, 0.07, 0.28, chunks, total, stage);
        CHECK(t.observe(t.cut(), chunks, total, stage));
        CHECK(t.cut() >= 36 && t.cut() <= 40);
        const int first = t.cut();
        measure(t, t.cut(), end, 0.07, 0.28, chunks, total, stage);
        t.observe(t.cut(), chunks, total, stage);
        CHECK(std::abs(t.cut() - first) <= 1);   // a second read of the same cards agrees
        // and the pipeline is shorter than at the shares' cut
        CHECK(t.read_ms(end - t.cut(), 0.07, 0.28, chunks) < 0.9 * t.read_ms(16, 0.07, 0.28, chunks));
    }
    // the shares' cut is already the best one: it stays
    {
        StageTuner t(end, 38, true);
        double total, stage;
        measure(t, 38, end, 0.0706, 0.26, chunks, total, stage);
        CHECK(!t.observe(38, chunks, total, stage));
        CHECK(t.cut() == 38);
    }
    // the stage card is faster than the shares thought: it never reads more than the shares gave it
    {
        StageTuner t(end, 38, true);
        double total, stage;
        measure(t, 38, end, 0.07, 0.05, chunks, total, stage);
        t.observe(38, chunks, total, stage);
        CHECK(t.cut() >= 38);
    }
    // matched cards at the half cut keep it
    {
        StageTuner t(end, 24, true);
        double total, stage;
        measure(t, 24, end, 0.07, 0.07, chunks, total, stage);
        t.observe(24, chunks, total, stage);
        CHECK(t.cut() == 24);
    }
    // a card far slower than any share: the cut ends with the stage a layer or a few, and the read is no longer than it
    {
        StageTuner t(end, 32, true);
        double total, stage;
        for (int i = 0; i < 3; ++i) {
            measure(t, t.cut(), end, 0.05, 0.5, chunks, total, stage);
            t.observe(t.cut(), chunks, total, stage);
        }
        CHECK(t.cut() >= 42);
        CHECK(t.read_ms(end - t.cut(), 0.05, 0.5, chunks) <= t.read_ms(16, 0.05, 0.5, chunks));
    }
    // too few chunks, or a share the user gave: nothing moves
    {
        StageTuner t(end, 32, true);
        CHECK(!t.observe(32, 3, 9000, 8000));
        CHECK(t.cut() == 32);
        StageTuner given(end, 32, false);
        double total, stage;
        measure(given, 32, end, 0.07, 0.28, chunks, total, stage);
        CHECK(!given.observe(32, chunks, total, stage));
        CHECK(given.cut() == 32);
    }
    if (g_fail == 0) std::printf("stage_tuner_test: ok\n");
    return g_fail == 0 ? 0 : 1;
}
