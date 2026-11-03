// include/basalt/model/batch_check.hpp - the batch conformance check every batching model passes (docs/design/batching.md):
// through the session interface only, so it names no model. `basalt --model FILE --batch N --tokens-file F
// --diag-batch-check` runs it.
#pragma once

#include "basalt/model/adapter.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace basalt::model {

struct BatchCheck {
    int compared = 0;     ///< windows compared, row by row and their last logits
    int differ = 0;       ///< of them, those whose bits are not the solo window's
    std::string report;   ///< one line per batch shape tried
};

/// Sessions ss[0..n) (n >= 2) of one batching model each read `prompt` from a different start (session i skips its first
/// i * 7 tokens). Then for `steps` rounds, in several compositions (all sessions in order, in reverse order: another
/// leader and every segment elsewhere in the call, and each pair), each session's window of 1, 2 or 3 rows (its last
/// token and drafts that are mostly rejected) runs alone (decode_window) and then as one batch at the same positions;
/// every row's pick and each window's last logits must be bitwise the solo window's: a row's bits do not depend on the
/// other rows of its call nor on its place among them. Each session then keeps its first pick. Error: a session failed.
Result check_batch(ModelSession* const* ss, int n, const std::vector<int32_t>& prompt, int steps, int64_t n_vocab, BatchCheck& out);

}  // namespace basalt::model
