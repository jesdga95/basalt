#pragma once
// Progress lines for the startup load, one per line on stderr:
//
//   basalt load: planning                      the start is deciding where everything goes (nothing placed yet)
//   basalt load: total <bytes>                 the bytes the whole start places, once, before any is placed
//   basalt load: weights <done> <total>        every byte placed so far, wherever it went (a card, a second card, RAM)
//   basalt load: stages <target>...            where the pass will place, in order: each card (`cuda<N>`, the main
//                                              card first, one card after the other), then `ram`; once, before the
//                                              first `into`
//   basalt load: into <target>                 where the pass places now, one of the stages: a label of the one bar,
//                                              not a phase of its own
//   basalt load: ngram <done> <total>          the n-gram table locked in RAM: a phase of its own, outside the total
//   basalt load: kv [slot|draft ]<target> <bytes> <tokens> <max_bytes>
//                                              the session state (KV and recurrent state) allocated on a device
//                                              (`cuda<N>`) for <tokens> of context; an allocation, not a load.
//                                              No kind: the solo session; `slot`: one batch slot's session; `draft`:
//                                              the draft layer's K/V.  <bytes> is what is backed now, <max_bytes> what
//                                              the state takes at <tokens> (larger only when the K/V is elastic).
//
// A start plans everything first and places it in one pass (place_start in loader.hpp): the total is what that pass
// places, announced once before its first byte, and never changes.  The loader counts every item it places; the one
// counter reads 0 to the total over the whole load.  A line is printed when the counter has advanced by 1% of the total
// or 250 ms have passed, and once with done == total at complete().  A start that places more or less than it announced
// is a miscount: it is reported once as a warning (and stops a debug build) and the total stays as announced.
// Thread safe.
#include <cassert>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include "basalt/log.hpp"

namespace basalt::core {

class LoadProgress {
public:
    /// A counted phase of its own, outside the weights (the n-gram table lock): `target` is its name in the lines.
    LoadProgress(std::string target, uint64_t total) : target_(std::move(target)), total_(total) {
        last_ = std::chrono::steady_clock::now();
        emit(0);
    }
    void add(uint64_t bytes) {
        std::lock_guard<std::mutex> lk(mu_);
        done_ += bytes;
        if (done_ > total_) done_ = total_;
        if (!finished_ && due(done_, last_done_, total_, last_)) emit_locked(std::chrono::steady_clock::now());
    }
    void finish() {
        std::lock_guard<std::mutex> lk(mu_);
        if (finished_) return;
        finished_ = true;
        done_ = total_;
        emit_locked(std::chrono::steady_clock::now());
    }

    /// Announces the bytes the whole start places (before the first is placed).
    static void announce_total(uint64_t bytes) {
        std::lock_guard<std::mutex> lk(weights_mu());
        W& w = weights();
        w.total = bytes;
        w.announced = true;
        basalt::log::info("basalt load", "total %llu", (unsigned long long) w.total);
        print_weights_locked();
    }
    /// Bytes landed, from any thread; outside a start (no total announced) they are not a start's.
    static void placed(uint64_t bytes) {
        std::lock_guard<std::mutex> lk(weights_mu());
        W& w = weights();
        if (!w.announced) return;
        w.placed += bytes;
        if (w.placed > w.total) miscount_locked("more than");
        w.done = w.placed < w.total ? w.placed : w.total;
        if (due(w.done, w.last_done, w.total, w.last)) print_weights_locked();
    }
    /// Where the pass will place, in order (`cuda<N>`..., `ram`): announced once, before the first `into`.
    static void stages(const std::vector<std::string>& targets) {
        std::lock_guard<std::mutex> lk(weights_mu());
        if (!weights().announced || targets.empty()) return;
        std::string line;
        for (const std::string& t : targets) line += " " + t;
        basalt::log::info("basalt load", "stages%s", line.c_str());
    }
    /// Where the pass places from now on (`cuda<N>` or `ram`), once per change; the bar keeps its one total.
    static void into(const std::string& target) {
        std::lock_guard<std::mutex> lk(weights_mu());
        W& w = weights();
        if (!w.announced || w.into == target) return;
        w.into = target;
        basalt::log::info("basalt load", "into %s", target.c_str());
    }
    /// The start begins planning: the first line, before the total.
    static void planning() { basalt::log::info("basalt load", "planning"); }
    /// The load is over: the final line, done == total.
    static void complete() {
        std::lock_guard<std::mutex> lk(weights_mu());
        W& w = weights();
        if (w.placed < w.total) miscount_locked("less than");
        w.done = w.total;
        print_weights_locked();
    }
    /// Reports the session state allocated on a device (not part of the weights).
    /// `kind` is "" (the solo session), "slot" or "draft".  `max_bytes` 0: the state is fully backed.
    static void kv(const std::string& target, uint64_t bytes, int64_t tokens, const char* kind = "",
                   uint64_t max_bytes = 0) {
        std::lock_guard<std::mutex> lk(weights_mu());
        basalt::log::info("basalt load", "kv %s%s%s %llu %lld %llu", kind, kind[0] ? " " : "", target.c_str(),
                     (unsigned long long) bytes, (long long) tokens,
                     (unsigned long long) (max_bytes > bytes ? max_bytes : bytes));
    }
    /// The target of a device ordinal: "cuda<N>".
    static std::string cuda(int device) { return "cuda" + std::to_string(device); }

private:
    using Clock = std::chrono::steady_clock;
    struct W {
        uint64_t total = 0, done = 0, placed = 0, last_done = ~0ull;
        bool announced = false, miscounted = false;
        std::string into;
        Clock::time_point last{};
    };
    static std::mutex& weights_mu() { static std::mutex m; return m; }
    static W& weights() { static W w; return w; }
    static bool due(uint64_t done, uint64_t last_done, uint64_t total, Clock::time_point last) {
        if (done == last_done) return false;
        const bool by_size = total > 0 && last_done != ~0ull && (done - last_done) * 100 >= total;
        const bool by_time = std::chrono::duration<double, std::milli>(Clock::now() - last).count() >= 250.0;
        return last_done == ~0ull || by_size || by_time;
    }
    static void miscount_locked(const char* how) {
        W& w = weights();
        if (w.miscounted) return;
        w.miscounted = true;
        basalt::log::warn("basalt load", "placed %s the announced total (%llu of %llu bytes so far)", how,
                          (unsigned long long) w.placed, (unsigned long long) w.total);
        assert(!"the start placed other bytes than it announced");
    }
    static void print_weights_locked() {
        W& w = weights();
        basalt::log::info("basalt load", "weights %llu %llu", (unsigned long long) w.done, (unsigned long long) w.total);
        w.last_done = w.done;
        w.last = Clock::now();
    }
    void emit(uint64_t done) {
        std::lock_guard<std::mutex> lk(mu_);
        done_ = done;
        emit_locked(Clock::now());
    }
    void emit_locked(Clock::time_point now) {
        basalt::log::info("basalt load", "%s %llu %llu", target_.c_str(), (unsigned long long) done_,
                     (unsigned long long) total_);
        last_done_ = done_;
        last_ = now;
    }

    std::string target_;
    uint64_t total_;
    uint64_t done_ = 0, last_done_ = ~0ull;
    bool finished_ = false;
    std::mutex mu_;
    Clock::time_point last_;
};

}  // namespace basalt::core
