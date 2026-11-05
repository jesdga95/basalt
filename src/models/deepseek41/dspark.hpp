// src/models/deepseek41/dspark.hpp - the DSpark drafter: three DeepSeek-V4.1 stages (window-only attention, 128 routed
// MXFP4 experts with top-3 and a shared expert, single-pass hyper-connections), a block-parallel pass over
// [anchor, mask x4], and the Markov and confidence heads. It shares the main model's token embedding and output matrix.
//
// The main model hands it, for every committed position, the mean over the four hyper-connection streams of the inputs
// of its target layers; `inject` turns them into each stage's window latent. `draft_begin` runs the block for the next
// position on a stream of its own and `draft_end` waits for it, so host work can overlap the block.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace basalt::models::deepseek41 {

class DsModel;

constexpr int kDsparkBlock = 5;     ///< rows of a block: the anchor and four mask tokens
constexpr int kDsparkWindow = 128;  ///< positions before the block a stage attends

struct DsparkDraft {
    int32_t token[kDsparkBlock];   ///< token[i] proposes the position i + 1 after the anchor
    float conf[kDsparkBlock];      ///< the confidence head's estimate that token[i] is accepted
};

class DsparkDrafter {
public:
    ~DsparkDrafter();

    /// Loads the pack's dspark/gguf section onto `device` (weights, a window ring per stage, scratch). Null with `err` set
    /// on failure.
    static std::unique_ptr<DsparkDrafter> create(const DsModel& main, int device, std::string& err);

    /// Slot (0..n_target_layers) the main model's layer `il` fills in the feature rows, or -1.
    int feature_slot(int il) const;
    int feature_floats() const;   ///< n_target_layers * n_embd
    int device() const;
    uint64_t device_bytes() const;   ///< what it took on its card

    /// Window latents of positions pos0.. from `n_rows` rows of target features (host, feature_floats() each).
    bool inject(int64_t pos0, int n_rows, const float* features, std::string& err);
    /// Starts the block for the anchor token `anchor` at position `pos`.
    bool draft_begin(int32_t anchor, int64_t pos, std::string& err);
    /// Waits for the block.
    bool draft_end(DsparkDraft& out, std::string& err);
    /// Forgets the window rows at or above `pos`: another conversation's rows are not to be read.
    void forget_from(int64_t pos);
    /// Forgets the whole window.
    void reset();
    /// One line of what the blocks took so far.
    void report() const;

private:
    struct Impl;
    explicit DsparkDrafter(std::unique_ptr<Impl> i);
    std::unique_ptr<Impl> p_;
};

}  // namespace basalt::models::deepseek41
