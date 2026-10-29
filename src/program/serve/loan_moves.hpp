// basalt/program/serve/loan_moves.hpp - the residency-table side of a prompt loan beside decoding slots.
//
// A loan that does not cover the next one gives back the entries the next loan leaves alone and keeps the rest; before
// a loan (or the KV) takes the expert cache's last slots, the experts there the decoding slots routed lately trade
// places with less-routed ones below.  A slot a loan still holds has the prompt path's buffers in it, not its expert,
// so a trade never reads it: that expert comes from its RAM copy.  Plain tables in, decisions out; no engine state.
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace basalt::program::serve {

/// A residency-table entry the prompt path borrowed: (layer * n_expert + expert, cache slot).
using LentEntry = std::pair<int32_t, int32_t>;

/// A loan that does not cover the next one, laid out from slot `first`: `again` holds the entries the next loan takes
/// again (their slots still hold the prompt path's buffers), `back` the ones it gives back (refilled now).
struct LoanSplit {
    std::vector<LentEntry> again, back;
};
LoanSplit split_loan(const std::vector<LentEntry>& lent, int64_t first);

/// One trade before a loan or the KV takes the cache's slots from `first` on: `hot`, an expert there that the decoding
/// slots routed lately, takes slot `to` below `first` from `cold`, a less-routed expert of its layer with a RAM copy,
/// which takes `hot`'s slot `from`.  `from_ram`: `from` holds no expert (a loan still out laid its buffers there), so
/// `hot`'s bytes come from its RAM copy.
struct RoutedTrade {
    int32_t hot = 0, cold = 0, from = 0, to = 0;
    bool from_ram = false;
};

/// The trades of one layer of `n` experts: `use` their recent routing, `slot` their cache slots (negative: none),
/// `in_ram` whether each has a RAM copy, `lent_out` (null: none) whether each one's slot belongs to a loan still out.
/// The routed experts from `first` on (one whose slot is lent out only with a RAM copy) trade, most-routed first, with
/// the least-routed RAM-backed experts below `first` while the hot one is routed more.  Appends to `out`; returns the
/// routed experts from `first` on.
int64_t plan_routed_trades(const float* use, const int32_t* slot, const uint8_t* in_ram, const uint8_t* lent_out,
                           int32_t n, int64_t first, std::vector<RoutedTrade>& out);

}  // namespace basalt::program::serve
