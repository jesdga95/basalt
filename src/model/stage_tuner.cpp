// src/model/stage_tuner.cpp - see include/basalt/model/stage_tuner.hpp.
#include "basalt/model/stage_tuner.hpp"

#include <algorithm>

namespace basalt::model {

namespace {
constexpr int kMinChunks = 4;        // fewer chunks are mostly fill and drain
constexpr double kMinGain = 0.03;    // a move needs the pipeline this much shorter
}  // namespace

double StageTuner::read_ms(int n1, double a, double b, int chunks) const {
    const double n0 = (double) (end_ - n1);
    double done = 0;
    for (int j = 1; j <= chunks; ++j) done = std::max((double) j * n0 * a, done) + (double) n1 * b;
    return done + (double) n1 * a;
}

bool StageTuner::observe(int cut, int chunks, double total_ms, double stage_ms) {
    const int n1 = end_ - cut;
    if (!active_ || chunks < kMinChunks || n1 < 1 || cut < 1 || total_ms <= 0 || stage_ms <= 0) return false;
    // the stage's cost per layer and chunk is what it did; the main card's is the one that makes the pipeline take
    // as long as the read did (the read is longer for a larger cost, so the cost is found by bisection)
    const double b = stage_ms / ((double) chunks * (double) n1);
    double lo = 0, hi = total_ms / (double) cut;
    for (int i = 0; i < 48; ++i) {
        const double mid = 0.5 * (lo + hi);
        (read_ms(n1, mid, b, chunks) < total_ms ? lo : hi) = mid;
    }
    const double a = 0.5 * (lo + hi);
    int best = end_ - cut_;
    double best_ms = read_ms(best, a, b, chunks);
    const double keep_ms = best_ms;
    for (int n = 1; n <= end_ - widest_; ++n) {
        const double ms = read_ms(n, a, b, chunks);
        if (ms < best_ms) {
            best_ms = ms;
            best = n;
        }
    }
    if (best == end_ - cut_ || !(best_ms < (1.0 - kMinGain) * keep_ms)) return false;
    cut_ = end_ - best;
    return true;
}

}  // namespace basalt::model
