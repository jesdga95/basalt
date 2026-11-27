// src/models/inkling2/fill_drain.hpp - what a call's end does with a card's projection fill (session.cpp): nothing when
// its join was queued, a wait for its stream otherwise, and an unconfirmed completion when the card's window stream is
// capturing or its capture status cannot be read (a host wait is not allowed during a capture, and a failed query can be
// an earlier error of that work).
#pragma once

namespace basalt::models::inkling2 {

enum class FillDrain { Done, Wait, Unconfirmed };

/// `pending`: a fill was forked on the card and its join was not queued; `query_ok`: the window stream's capture status was
/// read; `capturing`: that stream is capturing (active or invalidated).
inline FillDrain fill_drain(bool pending, bool query_ok, bool capturing) {
    if (!pending) return FillDrain::Done;
    if (!query_ok || capturing) return FillDrain::Unconfirmed;
    return FillDrain::Wait;
}

}  // namespace basalt::models::inkling2
