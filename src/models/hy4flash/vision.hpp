// src/models/hy4flash/vision.hpp - what Hy4-flash's processor asks of the shared image preprocessing: the canvas bounds and
// the resize filter.
#pragma once

#include "basalt/vision/preprocess.hpp"

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

/// The processor's `resample: 1`.
inline constexpr vision::Filter kVisionFilter = vision::Filter::Lanczos;

}  // namespace basalt::models::hy4flash
