// src/models/inkling2/wide_chunk.hpp - a long prompt read's wider chunk (prompt.cpp): the room taken from the card's expert
// copies, the rest of the arena backed and the drafter's wider tap, in that order; a step that fails undoes the ones before
// it, and the read's end undoes all of them, so the card holds the copies it held before. Memory whose release could not
// be confirmed is never counted free again.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace basalt::models::inkling2 {

/// Taking the room: taken; refused with nothing moved; or failed, with the copies not as they were.
enum class Take { Taken, Refused, Failed };
/// Backing the arena's rest, or allocating the tap: done; short (the card lacked the memory and nothing was allocated or
/// mapped); or broken (a failure that may have left something allocated or mapped).
enum class WideStep { Done, Short, Broken };
/// A read's start: wide; narrow with everything as it was; or failed (the read fails: the copies are not as they were, or
/// memory could not be released).
enum class WideOpen { Wide, Narrow, Failed };
/// A read's end: done, the copies as they were; the memory came back but not every copy; or stuck (the work may still run,
/// or the memory could not be released: it stays taken).
enum class WideClose { Done, RestoreFailed, Stuck };

/// Ops: `take(bytes)` -> Take; `give(bytes)` -> whether every copy take sent back returned, confirmed; `map()` -> WideStep
/// and `unmap()` (the arena's rest; false: not confirmed); `tap_on()` -> WideStep and `tap_off()` (the drafter's wider tap;
/// false: its release not confirmed). Only a Short step is undone and refunded; a Broken one keeps the room taken.
template <class Ops>
class WideChunk {
public:
    explicit WideChunk(Ops& ops) : ops_(ops) {}
    /// Before the first chunk.
    WideOpen open(uint64_t bytes, bool tap) {
        if (state_ == State::Stuck) return WideOpen::Failed;
        if (state_ == State::Open) return WideOpen::Narrow;
        switch (ops_.take(bytes)) {
            case Take::Refused: return WideOpen::Narrow;
            case Take::Failed: return WideOpen::Failed;
            case Take::Taken: break;
        }
        bytes_ = bytes;
        switch (ops_.map()) {
            case WideStep::Done: break;
            case WideStep::Short: return refund() ? WideOpen::Narrow : WideOpen::Failed;
            case WideStep::Broken:   // it may have left a mapping: the room stays taken
                state_ = State::Stuck;
                return WideOpen::Failed;
        }
        if (tap) {
            const WideStep t = ops_.tap_on();
            if (t != WideStep::Done) {
                // what may be allocated or mapped stays, and so does its room
                if (t == WideStep::Broken || !ops_.unmap()) {
                    state_ = State::Stuck;
                    return WideOpen::Failed;
                }
                return refund() ? WideOpen::Narrow : WideOpen::Failed;
            }
        }
        tap_ = tap;
        state_ = State::Open;
        return WideOpen::Wide;
    }
    /// After the read. `drained`: its streams are confirmed idle; when not, nothing is released and the memory stays taken.
    /// Never throws.
    WideClose close(bool drained) noexcept {
        if (state_ == State::Idle) return WideClose::Done;
        if (state_ == State::Stuck || !drained) {
            state_ = State::Stuck;
            return WideClose::Stuck;
        }
        try {
            if (tap_ && !ops_.tap_off()) {
                state_ = State::Stuck;
                return WideClose::Stuck;
            }
            tap_ = false;
            if (!ops_.unmap()) {
                state_ = State::Stuck;
                return WideClose::Stuck;
            }
            state_ = State::Idle;
            return ops_.give(bytes_) ? WideClose::Done : WideClose::RestoreFailed;
        } catch (...) {
            state_ = State::Stuck;
            return WideClose::Stuck;
        }
    }
    bool wide() const { return state_ == State::Open; }
    bool stuck() const { return state_ == State::Stuck; }

private:
    enum class State { Idle, Open, Stuck };
    bool refund() noexcept {
        try {
            return ops_.give(bytes_);
        } catch (...) {
            return false;
        }
    }
    Ops& ops_;
    uint64_t bytes_ = 0;
    bool tap_ = false;
    State state_ = State::Idle;
};

/// The pairs of `before` (in its order) that `after` no longer holds, into `out`. `kept` holds a zero for every pair and
/// `out` the capacity for all of `before`, so nothing allocates: the caller sizes both before any copy moves.
inline void left_out(const std::vector<size_t>& before, const std::vector<size_t>& after, std::vector<uint8_t>& kept,
                     std::vector<size_t>& out) {
    out.clear();
    for (const size_t pair : after) kept[pair] = 1;
    for (const size_t pair : before)
        if (!kept[pair]) out.push_back(pair);
    for (const size_t pair : after) kept[pair] = 0;
}

/// Whether the copies a take sent back are confirmed back: every pair of `pairs` has a copy again (`has(pair)`) and the
/// step that put them there finished (`fenced`: its last wait for the card succeeded).
template <class Has>
bool restored(const std::vector<size_t>& pairs, Has&& has, bool fenced) {
    if (!fenced) return false;
    for (const size_t pair : pairs)
        if (!has(pair)) return false;
    return true;
}

}  // namespace basalt::models::inkling2
