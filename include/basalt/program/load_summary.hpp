// include/basalt/program/load_summary.hpp - the engine's `basalt load:` stderr lines (format in
// basalt/core/load_progress.hpp) condensed into the lines a benchmark program prints while the engine starts.
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace basalt::program {

/// Fed the engine's log lines with a clock in seconds; each call returns the text to write to stderr (maybe empty).
/// Off a terminal: a header line, one line per finished phase and one about every `period` seconds.
/// On a terminal: one status line rewritten in place (at most every `redraw` seconds), left on screen at the end.
class LoadSummary {
public:
    LoadSummary(std::string label, bool tty, double now, double period = 10.0, double redraw = 0.5)
        : label_(std::move(label)), tty_(tty), t0_(now), last_out_(now), period_(period), redraw_(redraw) {}

    /// One log line. Lines that are neither load nor error lines are ignored.
    std::string feed(const std::string& line, double now);
    /// Time passing with no new line: the periodic update.
    std::string tick(double now);
    /// The start ended: READY (ready = true) or a failure. Closes the status line.
    std::string finish(double now, bool ready);

private:
    struct Target {
        std::string name;
        bool copy = false, reported = false;
        unsigned long long plan = 0, done = 0, last = 0, seen = 0;   // done: finished phases; last: current phase
        double start = -1;
        unsigned long long have() const { return done + last; }
        unsigned long long size() const { return plan > have() ? plan : have(); }
    };
    Target& target(const std::string& name, bool copy);
    std::string running() const;                    // "t d / p GiB" for the targets still loading, with percent
    std::string all_targets(double now) const;      // the terminal status line (without the label)
    std::string line(const std::string& text) const { return label_ + ": " + text + "\n"; }
    std::string redraw(double now);

    std::string label_;
    bool tty_;
    double t0_, last_out_, period_, redraw_;
    bool drawn_ = false, header_ = false;
    unsigned long long total_ = 0;
    std::vector<Target> targets_;   // in order of first mention
};

}  // namespace basalt::program
