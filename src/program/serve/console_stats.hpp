// src/program/serve/console_stats.hpp - what the server's console shows per slot and of the experts, once it asks
// (`STATS`): each slot's decode windows, its drafts by position and the expert lookups of its windows, and the
// expert map's sample.  Counted on the host from figures the windows already have; written at most twice a second
// while windows run, and when a request ends.
//
//   XSLOT <slot> <windows> <tokens> <hits> <lookups> <offloaded> <files> <accepted>/<offered>,...
//       the counts of the engine run of a request in that slot (the current one, or the last), from its start; the
//       last field has one entry per draft position (1, 2, ...) up to the last one offered, or "-" before any draft
//   XMAP <total> <homes> <routed>,...
//       the map's sample: `homes` one digit per sampled expert (core::ExpertHome), `routed` its lookups since the last
//       XMAP line
//
// The expert counts are those of DONE: `lookups` the main card or the CPU served, `hits` the main card's share of
// them, `offloaded` what a bus copy or another card served, `files` the lookups read from the pack file.  A window
// several slots share (a batch round) counts its lookups for each of them.
#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace basalt::program::serve {

/// The model's expert counts so far (ModelSession::ExpertCounts' fields).
struct LookupCounts {
    int64_t hits = 0, lookups = 0, offloaded = 0, files = 0;
};

struct SlotTally {
    static constexpr int kPositions = 16;
    int64_t windows = 0, tokens = 0;
    LookupCounts look;
    int64_t offered[kPositions] = {}, accepted[kPositions] = {};
    int positions = 0;   ///< draft positions offered so far
    bool changed = false;   ///< moved since its last XSLOT line
};

class ConsoleStats {
public:
    using Clock = std::chrono::steady_clock;
    static constexpr double kEverySeconds = 0.5;

    bool on = false;

    /// A request's engine run starts in `slot`: its counts start from zero.
    void start(int slot) {
        if (slot < 0) slot = 0;
        SlotTally& t = at(slot);
        t = SlotTally{};
        t.changed = true;
    }
    /// One decode window of `slot`: `offered` drafts, the first `accepted` of them kept, `tokens` sent.
    void window(int slot, int offered, int accepted, int tokens) {
        if (slot < 0) slot = 0;
        SlotTally& t = at(slot);
        ++t.windows;
        t.tokens += tokens;
        if (offered > SlotTally::kPositions) offered = SlotTally::kPositions;
        if (accepted > offered) accepted = offered;
        for (int j = 0; j < offered; ++j) ++t.offered[j];
        for (int j = 0; j < accepted; ++j) ++t.accepted[j];
        if (offered > t.positions) t.positions = offered;
        t.changed = true;
        if (!in_round(slot)) round_.push_back(slot);
    }
    /// A window or round starts: the counts its lookups are measured from.
    void lookups_begin(const LookupCounts& c) {
        look0_ = c;
        round_.clear();
    }
    /// It ended: the lookups since lookups_begin go to every slot that had a window in it.
    void lookups_end(const LookupCounts& c) {
        const LookupCounts d{c.hits - look0_.hits, c.lookups - look0_.lookups, c.offloaded - look0_.offloaded,
                             c.files - look0_.files};
        for (const int b : round_) {
            LookupCounts& l = at(b).look;
            l.hits += d.hits;
            l.lookups += d.lookups;
            l.offloaded += d.offloaded;
            l.files += d.files;
        }
        round_.clear();
        look0_ = c;
    }
    /// Whether the lines are due now (half a second since the last ones), or `force`d.
    bool due(Clock::time_point now, bool force) {
        if (!on) return false;
        if (!force && std::chrono::duration<double>(now - last_).count() < kEverySeconds) return false;
        last_ = now;
        return true;
    }
    /// The XSLOT lines of the slots whose counts moved since the last call.
    std::string slot_lines() {
        std::string out;
        for (size_t b = 0; b < tally_.size(); ++b) {
            if (!tally_[b].changed) continue;
            tally_[b].changed = false;
            out += slot_line((int) b);
        }
        return out;
    }
    std::string slot_line(int slot) const {
        const SlotTally& t = tally_[(size_t) slot];
        std::string s = "XSLOT " + std::to_string(slot) + " " + std::to_string(t.windows) + " " + std::to_string(t.tokens) +
                        " " + std::to_string(t.look.hits) + " " + std::to_string(t.look.lookups) + " " +
                        std::to_string(t.look.offloaded) + " " + std::to_string(t.look.files) + " ";
        if (t.positions == 0) s += "-";
        for (int j = 0; j < t.positions; ++j)
            s += (j ? "," : "") + std::to_string(t.accepted[j]) + "/" + std::to_string(t.offered[j]);
        return s + "\n";
    }
    const SlotTally* slot(int b) const { return b >= 0 && (size_t) b < tally_.size() ? &tally_[(size_t) b] : nullptr; }

    static std::string map_line(int64_t total, const std::vector<uint8_t>& homes, const std::vector<uint32_t>& routed) {
        std::string s = "XMAP " + std::to_string(total) + " ";
        for (const uint8_t h : homes) s += (char) ('0' + (h > 9 ? 9 : h));
        s += " ";
        for (size_t i = 0; i < routed.size(); ++i) s += (i ? "," : "") + std::to_string(routed[i]);
        return s + "\n";
    }

private:
    SlotTally& at(int slot) {
        if (slot < 0) slot = 0;
        if ((size_t) slot >= tally_.size()) tally_.resize((size_t) slot + 1);
        return tally_[(size_t) slot];
    }
    bool in_round(int slot) const {
        for (const int b : round_) if (b == slot) return true;
        return false;
    }
    std::vector<SlotTally> tally_;
    std::vector<int> round_;
    LookupCounts look0_;
    Clock::time_point last_{};
};

}  // namespace basalt::program::serve
