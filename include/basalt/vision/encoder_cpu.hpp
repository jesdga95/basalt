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

/// Receives named intermediate tensors: "embed" (patch + position embedding), "layer_out.N", "post_ln", "out"; the
/// class-token tower gives "pre_ln" (its sequence after the pre-norm) instead of "post_ln", its rows the class token first.
using TapFn = std::function<void(const std::string& name, const float* data, size_t n)>;

class CpuEncoder {
public:
    /// `mp` must outlive the encoder.  threads <= 0: all cores.
    bool init(const Mmproj& mp, int threads, bool round_bf16, bool f16_quirks, std::string& err);

    /// patches: [n_patches][768] in merge order (patchify), or [n_patches][1536] for a video group (patchify_video:
    /// the two frames' taps apart); canvas: the picture size in pixels. out: [n_patches / 4][proj] floats; for projector
    /// hy_v4_flash the merged rows in lines of canvas.w / 32, each followed by the newline row.
    bool encode(const std::vector<float>& patches, Size canvas, std::vector<float>& out, const TapFn& tap = {});

    /// The class-token tower (a projector file with v.class_embd): the picture at its own size, at the top left of a canvas
    /// of whole cells (patch x merge pixels a side; patchify_padded). out: [cells][proj] floats, the cells in raster order.
    bool encode_padded(const Image& img, std::vector<float>& out, const TapFn& tap = {});

private:
    struct Layer {
        const float *ln1w, *ln1b, *ln2w, *ln2b, *bqkv, *bo, *bup, *bdown;
        const uint16_t *wqkv, *wo, *wup, *wdown;
        const float *ls1 = nullptr, *ls2 = nullptr;   ///< the class-token tower's layer scales
    };
    bool init_cls(const Mmproj& mp, std::string& err);
    const Mmproj* mp_ = nullptr;
    int threads_ = 1;
    bool round_ = false, f16_ = false;
    std::vector<Layer> layers_;
    std::vector<float> wpatch_;               // [embd][768]: the two temporal taps summed, for a still picture
    std::vector<float> wpatch2_;              // [embd][1536]: per channel tap 0 then tap 1, for a video group
    const float *bpatch_ = nullptr, *pos_ = nullptr, *postw_ = nullptr, *postb_ = nullptr;
    const float *bm0_ = nullptr, *bm2_ = nullptr;
    // projector hy_v4_flash: a single patch tap, an edge_ x edge_ position table resampled per grid, no rotation, an RMS norm
    // (postw_) before an erf-GELU 2x2 merge product, and a newline row after each merged row
    bool hy4_ = false;
    int edge_ = 48;
    const float* newline_ = nullptr;
    // the class-token tower: the class token (cls_, its position summed in), a pre-norm (postw_, postb_), layer scale, 2D rotary
    // positions on interleaved pairs, and the merged rows through two products without bias
    bool cls_ = false;
    std::vector<float> cls_row_;
    // the matrices that have to be made from the file's (hy4: q, k and v as one, the convolutions as bf16 products)
    std::vector<std::vector<uint16_t>> own_w_;
    std::vector<std::vector<float>> own_b_;
    const uint16_t *wm0_ = nullptr, *wm2_ = nullptr;
};

}  // namespace basalt::vision
