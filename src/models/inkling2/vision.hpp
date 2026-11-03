// src/models/inkling2/vision.hpp - Inkling 2 image input: a picture's cells and the picture the encoder reads.
//
// A picture is read at its own size: the encoder places it at the top left of a canvas of whole 40 x 40 pixel cells and
// pads the rest. Each cell becomes one row of embeddings in the text model's width, the cells in raster order, no
// separator rows; the prompt holds one pad token per row after <|content_image|>, and the rows replace the pad tokens'
// embeddings as they are (no embedding norm). The text model has no rotary positions, so the rows need no grid.
#pragma once

#include "basalt/vision/preprocess.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace basalt::models::inkling2 {

/// The projector's name in the pack's `vision/mmproj` section.
inline constexpr const char* kProjector = "inkling2v";

/// Pixels of a cell side: 20-pixel patches merged 2 x 2.
inline constexpr int kCell = 40;

/// The most cells a picture takes: the model card's largest picture, 4096 pixels a side (103 x 103 cells). A picture with
/// more is scaled down to fit (aspect kept, to whole cells), the work one request can ask of the encoder bounded.
inline constexpr int kMaxCells = 103 * 103;

/// A picture as the encoder reads it.
struct ImagePlan {
    int w = 0, h = 0;          ///< the picture's size as encoded (scaled when the original was over the budget)
    int cols = 0, rows = 0;    ///< cells across and down
    bool scaled = false;
    int cells() const { return cols * rows; }
};

/// The plan of a w x h picture: at its own size when its cells fit kMaxCells, else scaled to the largest whole cells that
/// do (smart_resize's arithmetic). False with `err` for an empty picture.
bool plan_image(int w, int h, ImagePlan& p, std::string& err);

/// The picture the encoder reads: `src` with its transparency (`alpha`, w * h values; empty when it has none) over the
/// reference processor's checkerboard (8-pixel squares of 204 and 153), scaled to the plan's size when it says so
/// (Pillow-style bicubic).
vision::Image encoder_input(const vision::Image& src, const std::vector<uint8_t>& alpha, const ImagePlan& p);

}  // namespace basalt::models::inkling2
