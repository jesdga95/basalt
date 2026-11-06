// include/basalt/models/deepseek41/vision.hpp - DeepSeek-V4.1 image input: the resize plan, the patch rows, and the
// vision encoder (a 32-layer ViT with RMSNorm, SwiGLU and 2D RoPE, a 3x3 patch merge and a two-layer GELU aligner).
//
// An image becomes one block of embeddings in the text model's width:
//   [START] ([merged patch] * lw  [NEWLINE]) * lh [END]      (lh * (lw + 1) + 2 rows)
// START, NEWLINE and END are learned rows of the projector file; the block replaces the placeholder tokens in the prompt.
#pragma once

#include "basalt/vision/mmproj.hpp"
#include "basalt/vision/preprocess.hpp"

#include <memory>
#include <string>
#include <vector>

namespace basalt::models::deepseek41 {

constexpr int kVisionPatch = 14;
constexpr int kVisionPatchValues = 3 * kVisionPatch * kVisionPatch;   ///< one patch row: channel, y, x
constexpr int kVisionMaxTokens = 1024;                                ///< merged tokens of the largest image

struct VisionPlan {
    int width = 0, height = 0;   ///< canvas in pixels (multiples of the patch size)
    int gx = 0, gy = 0;          ///< patch grid
    int lw = 0, lh = 0;          ///< merged grid, rounded up
    int merge = 3;
    int n_patches() const { return gx * gy; }
    int n_tokens() const { return lh * (lw + 1) + 2; }
};

/// Canvas for a w x h picture: at least `min_pixels` area, at most `max_tokens` block rows, aspect preserved.
VisionPlan plan_image(int w, int h, int merge, int min_pixels, int max_tokens = kVisionMaxTokens);

/// Patch rows [n_patches][588] in raster order: the picture resized to fit the canvas, centred on grey 127, scaled to
/// [-1, 1] and rounded to bf16.
void image_patches(const basalt::vision::Image& src, const VisionPlan& plan, std::vector<float>& rows);

class VisionEncoder {
public:
    VisionEncoder();
    ~VisionEncoder();
    VisionEncoder(const VisionEncoder&) = delete;
    VisionEncoder& operator=(const VisionEncoder&) = delete;

    /// Uploads the projector `mp` (a deepseek41v file) to the current device; buffers grow on demand up to `max_patches`.
    bool init(const basalt::vision::Mmproj& mp, int max_patches, std::string& err);
    /// `out` = [plan.n_tokens()][width] floats, the embedding block.
    bool encode(const std::vector<float>& rows, const VisionPlan& plan, std::vector<float>& out, std::string& err);
    int width() const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace basalt::models::deepseek41
