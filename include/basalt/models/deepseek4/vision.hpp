#pragma once

#include "basalt/vision/preprocess.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace basalt::models::deepseek4 {

inline constexpr int kVisionPatch = 14;
inline constexpr int kVisionMerge = 3;
inline constexpr int kVisionMaxTokens = 384;
inline constexpr int kVisionMinPixels = 147456;
inline constexpr int kVisionMaxWhRatio = 8;
inline constexpr int kVisionCompressPad = 4;
inline constexpr int kVisionPatchValues = 3 * kVisionPatch * kVisionPatch;

enum class ImageToken : int { Patch = 2, Newline = 3, Pad = 4, Start = 0, End = 1 };

struct VisionPlan {
    int width = 0, height = 0;  // resized canvas, in pixels
    int n_vit_h = 0, n_vit_w = 0;
    int n_llm_h = 0, n_llm_w = 0;
    int n_tokens = 0;           // rows after N-layout padding, before prompt alignment padding
    bool stretch = false;       // extreme panorama branch in the reference preprocessor
};

struct ImageBlock {
    int alignment_pad = 0;
    int padded_rows = 0;
    int last_row_pad = 0;
    std::vector<ImageToken> types;
    std::vector<int> patch_order;
};

/// Match DeepSeek-V4 image_processor.py's resize and token-budget decisions.
bool plan_image(int width, int height, VisionPlan& plan, std::string& err);

/// RGB -> BF16-rounded patch rows [n_vit_h*n_vit_w][3][14][14], channel first.
bool image_patches(const basalt::vision::Image& image, const VisionPlan& plan,
                   std::vector<float>& patches, std::string& err);

/// Build the N-layout image token types and patch permutation at prompt position `start_pos`.
bool build_image_block(const VisionPlan& plan, int64_t start_pos, ImageBlock& block, std::string& err);

/// Assemble projected grid rows and learned special rows into the model's final image embedding sequence.
bool assemble_image_embeddings(const ImageBlock& block, const std::vector<float>& grid_rows, int width,
                               const float* start, const float* end, const float* pad, const float* newline,
                               std::vector<float>& rows, std::string& err);

}  // namespace basalt::models::deepseek4
