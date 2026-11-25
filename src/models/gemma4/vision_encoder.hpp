// src/models/gemma4/vision_encoder.hpp - the image encoder on the model's card, as the session drives it.
#pragma once

#include "basalt/core/model_file.hpp"
#include "basalt/vision/encoder_cuda.hpp"
#include "models/gemma4/vision.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace basalt::models::gemma4 {

/// --vision (dynamic): the projector of the pack's `vision/mmproj` section staged once in page-locked RAM. It holds no VRAM
/// at rest: a request's pictures get one device region (the weights and the work buffers for the largest picture), from the
/// room the model keeps free for it, and `release` gives it back.
class ImageEncoder {
public:
    ImageEncoder();
    ~ImageEncoder();
    ImageEncoder(const ImageEncoder&) = delete;
    ImageEncoder& operator=(const ImageEncoder&) = delete;

    /// Stages the projector for pictures of at most `max_cells` cells, makes the stream and product handle, and encodes one
    /// picture of the largest size in a temporary region (the product library loads its kernels on first use): call before
    /// the caches are sized.
    bool stage(const core::FileRange& section, int n_embd, int max_cells, std::string& err);
    bool staged() const { return enc_ != nullptr; }
    size_t weights_bytes() const;
    /// The device bytes of the region a request's pictures take.
    size_t region_bytes() const;

    /// A picture file's rows [cells][n_embd] (the cells in raster order); opens the region when none is open.
    bool encode(const uint8_t* data, size_t bytes, std::vector<float>& rows, int& n_rows, std::string& err);
    /// The region goes back to the driver.
    void release();

private:
    bool run(const vision::Image& src, const std::vector<uint8_t>& alpha, const ImagePlan& plan, std::vector<float>& rows,
             std::string& err);

    std::unique_ptr<vision::CudaEncoder> enc_;
    void* region_ = nullptr;
    int patch_ = 0, k_ = 0, max_cells_ = 0;
};

}  // namespace basalt::models::gemma4
