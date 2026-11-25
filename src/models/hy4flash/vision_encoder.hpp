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

/// --vision (dynamic): the projector of the pack's `vision/mmproj` section staged once in page-locked RAM, as Flash-Next's
/// in-engine encoder. It holds no VRAM at rest: a request's pictures get a device region (the weights and the work buffers),
/// which the session makes room for by lending the expert tier's slots, and `release` gives it back.
class ImageEncoder {
public:
    ImageEncoder();
    ~ImageEncoder();
    ImageEncoder(const ImageEncoder&) = delete;
    ImageEncoder& operator=(const ImageEncoder&) = delete;

    /// Stages the projector, makes the stream and product handle and encodes one full-size picture in a temporary region
    /// (the product library loads its kernels on first use): call before the expert tier is sized.
    bool stage(const core::FileRange& section, int n_embd, std::string& err);
    bool staged() const { return enc_ != nullptr; }
    size_t weights_bytes() const;
    /// Whether `plan` needs a new device region (none open, or the open one is too small).
    bool needs_region(const ImagePlan& plan) const { return tokens_ < plan.n_merged(); }
    /// Opens a region for `plan` (and a typical picture or more, so a request's pictures share it); false with `err` when the
    /// card has no room for it, and the session makes room.
    bool reserve(const ImagePlan& plan, std::string& err);
    /// The device bytes `reserve` takes for `plan`.
    size_t region_bytes(const ImagePlan& plan) const;

    /// A picture's block of rows [n_rows][n_embd] and its grid; reserves the region when needs_region.
    bool encode(const vision::Image& img, const ImagePlan& plan, std::vector<float>& rows, int& n_rows,
                model::ImageGrid& grid, std::string& err);
    /// The region goes back to the driver.
    void release();

private:
    std::unique_ptr<vision::CudaEncoder> enc_;
    void* region_ = nullptr;
    int tokens_ = 0;   ///< merged tokens the open region is sized for
};

}  // namespace basalt::models::hy4flash
