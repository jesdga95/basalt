// src/models/deepseek41/dspark.hpp - the DSpark drafter: three DeepSeek-V4.1 stages (window-only attention, 128 routed
// MXFP4 experts with top-3 and a shared expert, single-pass hyper-connections), a block-parallel pass over
// [anchor, mask x4], and the Markov and confidence heads. It shares the main model's token embedding and output matrix.
//
// The main model hands it, for every committed position, the mean over the four hyper-connection streams of the inputs
// of its target layers; `inject` turns them into each stage's window latent. `draft_begin` runs the block for the next
// position on a stream of its own and `draft_end` waits for it, so host work can overlap the block.
//
// One drafter serves every session: each holds a window of its own (`open_window`), the rows of its conversation.
// The stages read one set of rings; a call for another window first moves the rings out to their owner's store and
// that window's rows in.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace basalt::models::deepseek41 {

class DsModel;

constexpr int kDsparkBlock = 5;     ///< rows of a block: the anchor and four mask tokens
constexpr int kDsparkWindow = 128;  ///< positions before the block a stage attends

struct DsparkDraft {
    int n = 0;                     ///< the drafts the block made
    int32_t token[kDsparkBlock];   ///< token[i] proposes the position i + 1 after the anchor
    float conf[kDsparkBlock];      ///< the confidence head's estimate that token[i] is accepted
};

/// The experts each of `stages` stages keeps: its first `keep` in the ranked (stage, expert) pairs, topped up in index order
/// when the ranking names fewer, in ascending order.
std::vector<std::vector<int>> dspark_kept_experts(const std::vector<std::pair<int32_t, int32_t>>& ranked, int stages, int experts,
                                                  int keep);

class DsparkDrafter {
public:
    ~DsparkDrafter();

    /// Loads the pack's dspark/gguf section onto `device` (weights, a window ring per stage, scratch). Null with `err` set
    /// on failure.
    /// `ranking` (the format of data/profiles, a pair per stage and expert, most routed first): each stage keeps its `keep`
    /// most routed experts (0: all) and the router picks among them; without the file every expert loads.
    /// `windows`: the sessions that will draft, each with a window of its own.
    static std::unique_ptr<DsparkDrafter> create(const DsModel& main, int device, const std::string& ranking, int keep,
                                                 int windows, std::string& err);

    /// The next of the `windows` the drafter was made for; -1 when all are taken.
    int open_window();

    /// Slot (0..n_target_layers) the main model's layer `il` fills in the feature rows, or -1.
    int feature_slot(int il) const;
    int feature_floats() const;   ///< n_target_layers * n_embd
    int device() const;
    uint64_t device_bytes() const;   ///< what it took on its card
    const std::string& kept() const;   ///< how many experts each stage kept, and why all when that was not asked

    /// Window `w`'s latents of positions pos0.. from `n_rows` rows of target features (host, feature_floats() each).
    bool inject(int w, int64_t pos0, int n_rows, const float* features, std::string& err);
    /// Starts the block over window `w` for the anchor token `anchor` at position `pos`; its first `n_out` rows become drafts.
    bool draft_begin(int w, int32_t anchor, int64_t pos, int n_out, std::string& err);
    /// Waits for the block.
    bool draft_end(DsparkDraft& out, std::string& err);
    /// Forgets window `w`'s rows at or above `pos`: another conversation's rows are not to be read.
    void forget_from(int w, int64_t pos);
    /// Forgets the whole of window `w`.
    void reset(int w);

private:
    struct Impl;
    explicit DsparkDrafter(std::unique_ptr<Impl> i);
    std::unique_ptr<Impl> p_;
};

}  // namespace basalt::models::deepseek41
