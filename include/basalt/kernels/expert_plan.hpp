// include/basalt/kernels/expert_plan.hpp - routed (row, slot) pairs grouped by expert: a decode window's for
// native_expert_grouped (iq_kernels.hpp), one group per expert on the card, its entries in row order; a prompt chunk's
// as a stable sort by expert for the prompt products.
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::kernels {

constexpr int kPlanAhead = 8;   ///< the groups of a call moe_plan records (rec_blobs, rec_n) for a read-ahead
constexpr int kPlanMaxPairs = 4096;   ///< the (row, slot) pairs one moe_plan call takes at most

/// The grouped expert kernel's plan from the router's choices: group g is an expert on the card (its blob at blobs[e], a
/// device address; 0 when it is not on the card) and its entries are the (row, slot) pairs that chose it, in row order;
/// an entry writes row r * k + slot of the output. A pair whose expert is not on the card is left out (the caller
/// supplies its row). ptr [k rows], start [k rows + 2], n_groups [1], dst and tok [k rows]. rows * k <= kPlanMaxPairs,
/// n_expert <= 256.
/// With `tally` [n_expert], each expert on the card adds its pairs to its entry (the console's expert map); with
/// `rec_blobs` [kPlanAhead] and `rec_n`, the first groups' blobs are recorded. A programmatic dependent (pdl.hpp).
void moe_plan(const int32_t* ids, int rows, int k, int n_expert, const unsigned long long* blobs, unsigned long long* ptr,
              int32_t* start, int32_t* n_groups, int32_t* dst, int32_t* tok, void* stream, uint32_t* tally = nullptr,
              unsigned long long* rec_blobs = nullptr, int32_t* rec_n = nullptr);

/// Bytes of scratch moe_sort needs for `pairs` (row, slot) pairs.
size_t moe_sort_scratch(int pairs, int n_expert);
/// The pairs of `rows` rows' top-k choices ids [rows][k] sorted by expert, in pair order within an expert (a negative id
/// is a pair another card computes: it is left out, and bounds[n_expert] counts the rest): bounds
/// [n_expert + 1] (device), and for sorted entry i its token row tok[i], its pair dst[i] = row * k + slot and its expert
/// exp[i]. The same order at any launch shape. n_expert <= 512.
void moe_sort(const int32_t* ids, int rows, int k, int n_expert, void* scratch, int32_t* bounds, int32_t* tok, int32_t* dst,
              int32_t* exp, void* stream);

}  // namespace basalt::kernels
