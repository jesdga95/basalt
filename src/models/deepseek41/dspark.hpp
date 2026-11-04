// src/models/deepseek41/dspark.hpp - the DSpark drafter on a second GPU: three DeepSeek-V4.1 stages (window-only
// attention, 128 routed MXFP4 experts with top-3 and a shared expert, single-pass hyper-connections), a block-parallel
// pass over [anchor, noise x4], and the Markov and confidence heads. It shares the main model's token embedding and
// output matrix.
//
// The main model hands it, for every committed position, the mean over the four hyper-connection streams of the inputs
// of its target layers (15360 floats); `inject` turns them into each stage's window latent. `draft` runs the block for
// the next position.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace basalt::models::deepseek41 {

class DsModel;

constexpr int kDsparkBlock = 5;   ///< rows of a block: the anchor and four noise tokens

struct DsparkDraft {
    int32_t token[kDsparkBlock];   ///< token[i] proposes the position i + 1 after the anchor
    float conf[kDsparkBlock];      ///< the confidence head's estimate that token[i] is accepted
    float draft_ms = 0;            ///< device time of the block on the drafter's GPU
};

class DsparkDrafter {
public:
    ~DsparkDrafter();

    /// Loads the GGUF at `path` onto `device` (weights, a window ring per stage, scratch). The GGUF is the whole file,
    /// or `bytes` bytes at `offset` when `bytes` is set (a pack's dspark section). Null with `err` set on failure.
    static std::unique_ptr<DsparkDrafter> create(const DsModel& main, const std::string& path, uint64_t offset,
                                                 uint64_t bytes, int device, std::string& err);

    int n_target_layers() const;
    /// Slot (0..n_target_layers) the main model's layer `il` fills in the feature vector, or -1.
    int feature_slot(int il) const;
    int feature_floats() const;   ///< n_target_layers * n_embd

    /// Window latents of position `pos` from the target features (host, feature_floats()).
    bool inject(int64_t pos, const float* features, std::string& err);
    /// The block for the anchor token `id_last` at position `pos`.
    bool draft(int32_t id_last, int64_t pos, DsparkDraft& out, std::string& err);
    /// Forgets the window (a new sequence).
    void reset();

private:
    struct Impl;
    explicit DsparkDrafter(std::unique_ptr<Impl> i);
    std::unique_ptr<Impl> p_;
};

}  // namespace basalt::models::deepseek41
