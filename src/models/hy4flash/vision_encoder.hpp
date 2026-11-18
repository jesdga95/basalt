// src/models/hy4flash/vision_encoder.hpp - the image encoder on the model's card, as the session drives it.
#pragma once

#include "basalt/core/model_file.hpp"
#include "basalt/model/adapter.hpp"
#include "basalt/vision/encoder_cuda.hpp"
#include "basalt/vision/preprocess.hpp"
#include "models/hy4flash/vision.hpp"

#include <memory>
#include <string>
#include <vector>

namespace basalt::models::hy4flash {

/// The merged grid as the engine passes it with the rows: the newline column counts as a column.
inline model::ImageGrid image_grid(const ImagePlan& p) { return {p.ow + 1, p.oh}; }

/// Opened from the pack's `vision/mmproj` section for the pictures of a request and
/// closed by `release` (the weights and work buffers are freed; the pack is read again for the next request).
class ImageEncoder {
public:
    ImageEncoder();
    ~ImageEncoder();
    ImageEncoder(const ImageEncoder&) = delete;
    ImageEncoder& operator=(const ImageEncoder&) = delete;

    /// A picture's block of rows [n_rows][n_embd] and its grid. `section` is the pack's vision/mmproj range.
    bool encode(const core::FileRange& section, const vision::Image& img, std::vector<float>& rows, int& n_rows,
                model::ImageGrid& grid, std::string& err);
    void release();

private:
    std::unique_ptr<vision::CudaEncoder> enc_;
    int tokens_ = 0;   ///< merged tokens the open encoder is sized for
};

}  // namespace basalt::models::hy4flash
