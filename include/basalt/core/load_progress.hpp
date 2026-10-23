#pragma once
// Progress lines for the startup loads, one per line on stderr:
//
//   basalt load: total <bytes>                        the bytes the whole start places, once, before any phase
//   basalt load: plan [copy ]<target> <bytes>         what one target's phases will report in all, as soon as known
//   basalt load: <target> <done_bytes> <total_bytes>  a phase that places model bytes
//   basalt load: copy <target> <done> <total>         a phase that copies bytes a counted phase already placed
//
// `<target>` is `cuda<N>` (bytes written into that device) or `ram` (experts copied or locked into system RAM).
//   basalt load: kv [slot|draft ]<target> <bytes> <tokens> <max_bytes>
//                                                     the session state (KV and recurrent state) allocated on a
//                                                     device for <tokens> of context; an allocation, not a load phase.
//                                                     No kind: the solo session; `slot`: one batch slot's session;
//                                                     `draft`: the draft layer's K/V (working copy and slot stores).
//                                                     <bytes> is what is backed now, <max_bytes> what the state
//                                                     takes at <tokens> (larger only when the K/V is elastic).
// A plan is the sum of the totals of the phases the target will run (a later plan line replaces an earlier one).
// The total is announced once, from the model file's tables, and counts every byte once: the sum of the phase totals
// that are not copies.  If a part cannot be placed (the RAM is short and the experts are read from the file instead),
// skipped() lowers it and the line is printed again.  A copy is a second home for bytes already counted, such as an
// expert tier on another card that is also held in RAM.
// A load phase makes a LoadProgress with its total (known before copying), calls add() as bytes land and finish()
// at the end.  A line is printed when the phase starts (done 0), whenever done has advanced by 1% of the total or
// 250 ms have passed, and once with done == total.  Safe to call from several threads.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <utility>
#include "basalt/log.hpp"

namespace basalt::core {

class LoadProgress {
public:
    LoadProgress(std::string target, uint64_t total, bool copy = false)
        : target_(std::move(target)), total_(total), copy_(copy) {
        last_ = std::chrono::steady_clock::now();
        emit(0);
    }
    /// Announces the bytes the whole start places (before the first phase).
    static void announce_total(uint64_t bytes) {
        std::lock_guard<std::mutex> lk(total_mu());
        announced() = true;
        total_value() = bytes;
        print_total_locked();
    }
    /// Announces what a target's phases will add up to (a copy target: bytes another tier already holds).
    static void plan(const std::string& target, uint64_t bytes, bool copy = false) {
        std::lock_guard<std::mutex> lk(total_mu());
        basalt::log::info("basalt load", "plan %s%s %llu", copy ? "copy " : "", target.c_str(),
                     (unsigned long long) bytes);
    }
    /// Reports the session state allocated on a device (not part of any load total).
    /// `kind` is "" (the solo session), "slot" or "draft".  `max_bytes` 0: the state is fully backed.
    static void kv(const std::string& target, uint64_t bytes, int64_t tokens, const char* kind = "",
                   uint64_t max_bytes = 0) {
        std::lock_guard<std::mutex> lk(total_mu());
        basalt::log::info("basalt load", "kv %s%s%s %llu %lld %llu", kind, kind[0] ? " " : "", target.c_str(),
                     (unsigned long long) bytes, (long long) tokens,
                     (unsigned long long) (max_bytes > bytes ? max_bytes : bytes));
    }
    /// Lowers the announced total by bytes that will not be placed after all; a no-op without a total.
    static void skipped(uint64_t bytes) {
        std::lock_guard<std::mutex> lk(total_mu());
        if (!announced()) return;
        total_value() = bytes < total_value() ? total_value() - bytes : 0;
        print_total_locked();
    }
    /// The target of a device ordinal: "cuda<N>".
    static std::string cuda(int device) { return "cuda" + std::to_string(device); }

    void add(uint64_t bytes) {
        std::lock_guard<std::mutex> lk(mu_);
        done_ += bytes;
        if (done_ > total_) done_ = total_;
        const auto now = std::chrono::steady_clock::now();
        const bool by_size = total_ > 0 && (done_ - last_done_) * 100 >= total_;
        const bool by_time = std::chrono::duration<double, std::milli>(now - last_).count() >= 250.0;
        if (!finished_ && (by_size || by_time) && done_ != last_done_) emit_locked(now);
    }
    /// The final line (done == total).
    void finish() {
        std::lock_guard<std::mutex> lk(mu_);
        if (finished_) return;
        finished_ = true;
        done_ = total_;
        emit_locked(std::chrono::steady_clock::now());
    }

private:
    void emit(uint64_t done) {
        std::lock_guard<std::mutex> lk(mu_);
        done_ = done;
        emit_locked(std::chrono::steady_clock::now());
    }
    void emit_locked(std::chrono::steady_clock::time_point now) {
        basalt::log::info("basalt load", "%s%s %llu %llu", copy_ ? "copy " : "", target_.c_str(),
                     (unsigned long long) done_, (unsigned long long) total_);
        last_done_ = done_;
        last_ = now;
    }

    static std::mutex& total_mu() { static std::mutex m; return m; }
    static bool& announced() { static bool a = false; return a; }
    static uint64_t& total_value() { static uint64_t t = 0; return t; }
    static void print_total_locked() {
        basalt::log::info("basalt load", "total %llu", (unsigned long long) total_value());
    }

    std::string target_;
    uint64_t total_;
    bool copy_;
    uint64_t done_ = 0, last_done_ = 0;
    bool finished_ = false;
    std::mutex mu_;
    std::chrono::steady_clock::time_point last_;
};

}  // namespace basalt::core
