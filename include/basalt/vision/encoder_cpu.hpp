// basalt/vision/encoder_cpu.hpp - the vision encoder on the CPU: fp32 activations, bf16 weights widened in the GEMM.
//
// Also the reference every GPU stage is checked against.  `round_bf16` rounds each GEMM input to bf16 (what the
// batched GPU path does); off, the activations stay fp32.  `f16_quirks` reproduces the reference CPU run's fp16
// patch input and fp16 GELU table, for stage-by-stage comparison with it only.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "basalt/vision/mmproj.hpp"
#include "basalt/vision/preprocess.hpp"

namespace basalt::vision {

/// Receives named intermediate tensors: "embed" (patch + position embedding), "layer_out.N", "post_ln", "out".
using TapFn = std::function<void(const std::string& name, const float* data, size_t n)>;

class CpuEncoder {
public:
    /// `mp` must outlive the encoder.  threads <= 0: all cores.
    bool init(const Mmproj& mp, int threads, bool round_bf16, bool f16_quirks, std::string& err);

    /// patches: [n_patches][768] in merge order (patchify), or [n_patches][1536] for a video group (patchify_video:
    /// the two frames' taps apart); canvas: the picture size in pixels. out: [n_patches / 4][proj] floats.
    bool encode(const std::vector<float>& patches, Size canvas, std::vector<float>& out, const TapFn& tap = {});

private:
    struct Layer {
        const float *ln1w, *ln1b, *ln2w, *ln2b, *bqkv, *bo, *bup, *bdown;
        const uint16_t *wqkv, *wo, *wup, *wdown;
    };
    const Mmproj* mp_ = nullptr;
    int threads_ = 1;
    bool round_ = false, f16_ = false;
    std::vector<Layer> layers_;
    std::vector<float> wpatch_;               // [embd][768]: the two temporal taps summed, for a still picture
    std::vector<float> wpatch2_;              // [embd][1536]: per channel tap 0 then tap 1, for a video group
    const float *bpatch_ = nullptr, *pos_ = nullptr, *postw_ = nullptr, *postb_ = nullptr;
    const float *bm0_ = nullptr, *bm2_ = nullptr;
    const uint16_t *wm0_ = nullptr, *wm2_ = nullptr;
};

}  // namespace basalt::vision
