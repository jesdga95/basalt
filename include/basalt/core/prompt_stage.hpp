// include/basalt/core/prompt_stage.hpp - a prompt split's stages, the contract every model with one shares.
//
// A prompt split reads a long prompt's chunks across cards: the main card reads a chunk's first layers while each stage
// card reads an earlier chunk's later layers.  How a model runs it is its own (a dense model's resident copy on its second
// card, an expert model's image for its tier cards); what a stage is, how its weights count in the start's total and the
// start's lines about it are the same.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace basalt::core {

/// Where a prompt split's stage keeps its weights between reads.
enum class StageHome {
    Card,        ///< a resident copy on the stage's card, which has room beside the model (a dense model's second card)
    PinnedRam,   ///< a page-locked image, taken to the card's lent expert slots for each split read (a card full of experts)
};

/// A stage of a prompt split: a card that reads layers [lo, hi) of long prompts' chunks beside the main card, and where
/// its weights live.  A start decides its stages in its plan and reads their weights in its one pass; they count in the
/// start's total as the bytes read from the file for them (none when they are copied from weights the pass places on the
/// main card).
struct PromptStage {
    int device = -1;
    int lo = 0, hi = 0;
    StageHome home = StageHome::Card;
    uint64_t bytes = 0;   ///< its weights' bytes in that home
};

/// The start's line of a split that is on, the same for every model:
///   prompt split: layers 44-63 on CUDA1 (2.10 GiB on the card)[, 24-35 on CUDA2 (1.40 GiB in page-locked RAM) ...]
void log_prompt_split(const char* area, const std::vector<PromptStage>& stages);
/// The start's line of a split that is off: `prompt split: off (<why>)`; `lost` (a failure took it away, not a decision):
/// as a warning.
void log_prompt_split_off(const char* area, const std::string& why, bool lost = false);

}  // namespace basalt::core
