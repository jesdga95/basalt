// include/basalt/kernels/expert_plan.hpp - a decode window's routed (row, slot) pairs grouped by expert for
// native_expert_grouped (iq_kernels.hpp): one group per expert on the card, its entries in row order.
#pragma once

#include <cstdint>

namespace basalt::kernels {

constexpr int kPlanAhead = 8;   ///< the groups of a call moe_plan records (rec_blobs, rec_n) for a read-ahead

/// The grouped expert kernel's plan from the router's choices: group g is an expert on the card (its blob at blobs[e], a
/// device address; 0 when it is not on the card) and its entries are the (row, slot) pairs that chose it, in row order;
/// an entry writes row r * k + slot of the output. A pair whose expert is not on the card is left out (the caller
/// supplies its row). ptr [k rows], start [k rows + 2], n_groups [1], dst and tok [k rows]. rows * k <= 1024, n_expert <= 256.
/// With `tally` [n_expert], each expert on the card adds its pairs to its entry (the console's expert map); with
/// `rec_blobs` [kPlanAhead] and `rec_n`, the first groups' blobs are recorded. A programmatic dependent (pdl.hpp).
void moe_plan(const int32_t* ids, int rows, int k, int n_expert, const unsigned long long* blobs, unsigned long long* ptr,
              int32_t* start, int32_t* n_groups, int32_t* dst, int32_t* tok, void* stream, uint32_t* tally = nullptr,
              unsigned long long* rec_blobs = nullptr, int32_t* rec_n = nullptr);

}  // namespace basalt::kernels
