// src/program/serve/prompt_window.hpp - scheduling batch windows between synchronized prompt chunks.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <deque>
#include <string>
#include <vector>

namespace basalt::program::serve {

/// Apply leading stops for active slots. Any other command keeps its place before the next window.
template <class Slot>
bool prompt_window_queue(std::deque<std::string>& lines, std::vector<Slot>& slots) {
    while (!lines.empty()) {
        const std::string& cmd = lines.front();
        if (cmd.rfind("BSTOP ", 0) != 0) return false;
        const int b = std::atoi(cmd.c_str() + 6);
        if (b < 0 || b >= (int) slots.size() || !slots[(size_t) b].active) return false;
        slots[(size_t) b].stop = true;
        lines.pop_front();
    }
    return true;
}

/// Between two prompt chunks, run at least one ready window. Start another while less than half the chunk time has elapsed.
template <class Clock, class Ready, class Next>
bool prompt_decode_windows(std::chrono::time_point<Clock> chunk_start, Ready ready, Next next) {
    const auto start = Clock::now();
    const auto budget = (start - chunk_start) / 2;
    do {
        if (!ready()) break;
        if (!next()) return false;
    } while (Clock::now() - start < budget);
    return true;
}

}  // namespace basalt::program::serve
