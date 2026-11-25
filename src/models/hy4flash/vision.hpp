// src/models/hy4flash/vision.hpp - Hy4-flash image input: what the processor asks of the shared preprocessing (canvas bounds,
// bicubic), the token layout of a picture, its rotary positions, and the encoder wrapper the session drives.
//
// A picture becomes one block of embedding rows in the text model's width:
//   <image_start> ( [merged patch] * ow  [newline] ) * oh <image_end>
// The prompt holds the pad token once per row of the block (oh * (ow + 1) of them, the newline slots included) between the
// start and end tokens; the rows replace the pad tokens' embeddings. Positions are 3-D (t, h, w) with the interleaved
// sections [12, 10, 10] over the 32 rotated pairs of a head: a picture's rows take (p, p + row, p + column) with the
// newline in column ow, and the text after it continues at p + max(oh, ow + 1).
#pragma once

#include "basalt/core/model_file.hpp"
#include "basalt/vision/preprocess.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace basalt::models::hy4flash {

/// 16-pixel patches merged 2 x 2; the processor's min_pixels 4,096 and max_pixels 4,194,304 are 4 and 4,096 merged tokens.
inline vision::Geometry vision_geometry() {
    vision::Geometry g;
    g.patch = 16;
    g.merge = 2;
    g.min_tokens = 4;
    g.max_tokens = 4096;
    return g;
}

/// The projector's name in the pack's `vision/mmproj` section.
inline constexpr const char* kProjector = "hy_v4_flash";

/// The processor's smart_resize refuses a picture more than 200 times longer than wide.
inline constexpr int kMaxAspect = 200;

/// A picture's canvas and the block of rows it becomes.
struct ImagePlan {
    vision::Size canvas;      ///< pixels, multiples of 32
    int gx = 0, gy = 0;       ///< the patch grid
    int ow = 0, oh = 0;       ///< merged tokens across and down
    int n_rows() const { return oh * (ow + 1); }   ///< the block: a newline row after each line
    int n_merged() const { return ow * oh; }
};

/// The canvas of a w x h picture (smart_resize: within [4, 4096] merged tokens, aspect kept to the 32-pixel grid) and its
/// grid. False with `err` for an empty picture or an aspect ratio over kMaxAspect.
bool plan_image(int w, int h, ImagePlan& p, std::string& err);

/// The patch rows [n_patches][768] in merge order: the picture resized to the canvas with Pillow's bicubic (transformers'
/// processor and the original it follows leave the configured `resample: 1`, Lanczos, unused), scaled to [-1, 1], cut into
/// 16 x 16 patches.
void image_patches(const vision::Image& src, const ImagePlan& plan, std::vector<float>& patches);

/// A token's rotary position on the three axes.
struct RopePos {
    int32_t t = 0, h = 0, w = 0;
    int32_t last() const { return t > h ? (t > w ? t : w) : (h > w ? h : w); }
};

/// The rotary section of the model: pair i of a head turns by the h axis when i % 3 == 1 and i < 3 * kMropeH, by w when
/// i % 3 == 2 and i < 3 * kMropeW, by t otherwise (mrope_section [12, 10, 10], interleaved).
inline constexpr int kMropeH = 10, kMropeW = 10;

/// The rotary positions of a session's tokens by cell. A text token follows the last cell's largest position; a picture's
/// rows are placed from the next text position by their grid. A cell rewound past is forgotten with its position.
class RopeTrack {
public:
    /// The next text token's position: one past the largest position of the cell before it.
    int32_t next() const { return cells_.empty() ? 0 : cells_.back().last() + 1; }
    /// Forgets the cells from `pos` on.
    void rewind(int64_t pos) {
        if (pos < 0) pos = 0;
        if ((size_t) pos < cells_.size()) cells_.resize((size_t) pos);
        if (first_image_ >= 0 && first_image_ >= (int64_t) cells_.size()) first_image_ = -1;
    }
    int64_t size() const { return (int64_t) cells_.size(); }
    bool has_image() const { return first_image_ >= 0; }
    const RopePos& at(int64_t cell) const { return cells_[(size_t) cell]; }
    /// Places a text token at the next cell.
    RopePos push_text() { const int32_t p = next(); cells_.push_back({p, p, p}); return cells_.back(); }
    /// Places row `j` of a picture block of `nx` columns (the newline column included) starting at position `start` at the
    /// next cell.
    RopePos push_row(int32_t start, int nx, int j) {
        if (first_image_ < 0) first_image_ = (int64_t) cells_.size();
        cells_.push_back({start, start + j / nx, start + j % nx});
        return cells_.back();
    }
    void clear() {
        cells_.clear();
        first_image_ = -1;
    }

private:
    std::vector<RopePos> cells_;
    int64_t first_image_ = -1;
};

/// The tokens of the pack's vocabulary the layout uses (the pack's meta.hf_config).
struct VisionTokens {
    int32_t image_start = -1, image_pad = -1, image_end = -1, image_newline = -1;
};
/// Reads them from the pack's `meta`; false with `err` when it names none (a pack made without the HF config).
bool read_vision_tokens(const core::Json& meta, VisionTokens& t, std::string& err);

}  // namespace basalt::models::hy4flash
