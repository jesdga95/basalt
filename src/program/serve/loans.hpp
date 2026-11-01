// basalt/program/serve/loans.hpp - the prompt path's chunk sizes and the expert-cache slots it borrows.
//
// The prompt path lends the tail of the expert cache to its own buffers, so a chunk is sized by what the cache can
// spare: `bytes_needed` for a chunk of T tokens, rounded up to whole blobs, against the slots left.  The cache, the
// geometry and the session arrive as arguments; nothing here reads or holds engine state.
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "basalt/core/expert_cache.hpp"   // ExpertCache
#include "basalt/core/layout.hpp"         // ModelGeometry
#include "basalt/core/session.hpp"        // SessionState

namespace basalt::program::serve {

/// How many of `xc`'s slots `need` bytes take, measured from the end (sized slots) or in whole blobs.
int64_t cache_slots_for(const basalt::core::ExpertCache& xc, uint64_t need);

/// The slots a prompt chunk of `cells` tokens needs from `xc`'s tail.
int64_t lend_slots(const basalt::core::ExpertCache& xc, const basalt::core::ModelGeometry& g,
                   const basalt::core::SessionState& ss, int64_t cells);

/// One card's chunk request rounded up to the 256-token grid the prompt path works on, capped at `max_chunk`.
int64_t request_chunk(int64_t tokens, int64_t max_chunk);

/// How one card reads a part of `tokens` in chunks of at most `cap`: equal chunks for the fewest chunks, and with
/// `head_ok` a first chunk of its own (`auto_fit` is the operator's --prefill auto-with-fit).  (chunk, head chunk).
std::pair<int64_t, int64_t> one_card_chunks(int64_t tokens, int64_t cap, bool head_ok, bool auto_fit);

/// The sizes of the chunks the prompt path reads a part in, for a chunk and head chunk.
std::vector<int64_t> chunk_list(int64_t tokens, int64_t chunk, int64_t head);

/// The largest chunk the lend can fund (with `prefill_auto`, the largest on the 256-token grid under the caps),
/// written to `chunk`; returns the slots it needs, 0 when nothing fits.
int64_t plan_lend(int64_t& chunk, const basalt::core::ExpertCache& xc, const basalt::core::ModelGeometry& g,
                  const basalt::core::SessionState& ss, bool prefill_auto, int64_t auto_ceiling, int auto_lend_pct);

}  // namespace basalt::program::serve
