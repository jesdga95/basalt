#include "basalt/core/decision_schedule.hpp"
#include <cassert>
#include <random>
#include <vector>

int main() {
    using namespace basalt::core;
    std::mt19937 rng(73);
    for (int capacity = 2; capacity <= 8; ++capacity)
        for (int lanes = 2; lanes <= capacity; ++lanes)
            for (int trial = 0; trial < 100; ++trial) {
                std::vector<int> remaining(lanes), original(lanes), visited(lanes);
                for (int& n : remaining) n = 1 + rng() % 257;
                original = remaining;
                int total = 0;
                for (int n : remaining) total += n;
                int windows = 0;
                for (;;) {
                    const auto plan = decision_window(remaining, capacity);
                    if (!plan.size) break;
                    assert(plan.rows >= plan.size && plan.rows <= capacity);
                    int previous = -1, rows = 0;
                    for (int j = 0; j < plan.size; ++j) {
                        const int i = plan.slots[j], n = plan.lengths[j];
                        assert(i > previous && n > 0 && n <= remaining[i]);
                        previous = i; rows += n; visited[i] += n; remaining[i] -= n;
                    }
                    assert(rows == plan.rows);
                    assert(++windows <= 257);
                }
                assert(visited == original);
                assert(windows == (total + capacity - 1) / capacity);
            }
    assert(decision_window(std::array{0, 0}, 2).size == 0);
    assert(decision_window(std::array{1, 1, 1}, 2).size == 0);
    const auto short_wave = decision_window(std::array{7, 1}, 8);
    assert(short_wave.rows == 8 && short_wave.lengths[0] == 7 && short_wave.lengths[1] == 1);
    assert(decision_draft_prefix(-1, 100, false) == 100);
    assert(decision_draft_prefix(100, 200, false) == 100);
    assert(decision_draft_prefix(100, 60, false) == 60);
    assert(decision_draft_prefix(100, 0, false) == 0);
    assert(decision_draft_prefix(0, 200, true) == 200);
}
