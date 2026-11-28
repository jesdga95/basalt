// src/models/gemma4/vision.hpp - Gemma 4 image input: the size a picture is read at and the picture the encoder reads.
//
// The processor resizes a picture (aspect kept, bicubic) to the largest size of whole cells (k x k patches of 16 pixels,
// k the pooling kernel) whose patches fit the soft-token budget times k^2, up or down. Each cell becomes one row of
// embeddings in the text model's width, the cells in raster order; the prompt holds <|image> (boi), one <|image|> per row,
// <image|> (eoi), and the rows replace the <|image|> tokens' embeddings as they are (not scaled). The text model's window
// layers let a picture's rows attend to each other both ways.
#pragma once

#include "basalt/vision/preprocess.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace basalt::models::gemma4 {

/// The projector's name in the pack's `vision/mmproj` section.
inline constexpr const char* kProjector = "gemma4v";

/// A picture as the encoder reads it.
struct ImagePlan {
    int w = 0, h = 0;          ///< the size it is read at (whole cells)
    int cols = 0, rows = 0;    ///< cells across and down
    bool resized = false;      ///< the size differs from the picture's own
    int cells() const { return cols * rows; }
};

/// The plan of a w x h picture for cells of `k` x `k` patches of `patch` pixels and at most `max_cells` cells: the
/// processor's get_aspect_ratio_preserving_size (in double, as Python computes it). False with `err` for an empty picture
/// or one the processor refuses.
bool plan_image(int w, int h, int patch, int k, int max_cells, ImagePlan& p, std::string& err);

/// The picture the encoder reads: `src` with its transparency (`alpha`, w * h values; empty when it has none) over white,
/// as the processor's RGB conversion composites it, resized to the plan's size when it says so (Pillow-style bicubic).
vision::Image encoder_input(const vision::Image& src, const std::vector<uint8_t>& alpha, const ImagePlan& p);

}  // namespace basalt::models::gemma4
