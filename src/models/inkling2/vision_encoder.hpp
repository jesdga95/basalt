// src/models/inkling2/vision_encoder.hpp - the image encoder on the model's card, as the session drives it.
#pragma once

#include "basalt/core/model_file.hpp"
#include "basalt/model/adapter.hpp"
#include "basalt/vision/encoder_cuda.hpp"
#include "models/inkling2/vision.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace basalt::models::inkling2 {

/// --vision (dynamic): the projector of the pack's `vision/mmproj` section staged once in page-locked RAM. It holds no VRAM
/// at rest: a request's pictures get a device region (the weights and the work buffers), and `release` gives it back. The
/// model keeps region_bytes() free beside its caches, so a region always fits.
class ImageEncoder {
public:
    ImageEncoder();
    ~ImageEncoder();
    ImageEncoder(const ImageEncoder&) = delete;
    ImageEncoder& operator=(const ImageEncoder&) = delete;

    /// Stages the projector, makes the stream and product handle, and encodes one picture of the largest size in a temporary
    /// region (the product library loads its kernels on first use): call before the caches are sized.
    bool stage(const core::FileRange& section, int n_embd, std::string& err);
    bool staged() const { return enc_ != nullptr; }
    size_t weights_bytes() const;
    /// The device bytes of the region for the largest picture (kMaxCells).
    size_t region_bytes() const;

    /// A picture file's rows [cells][n_embd] and its grid (cells across and down); opens the region when the open one is too
    /// small.
    bool encode(const uint8_t* data, size_t bytes, std::vector<float>& rows, int& n_rows, model::ImageGrid& grid,
                std::string& err);
    /// The region goes back to the driver.
    void release();

private:
    bool reserve(int cells, std::string& err);
    bool run(const vision::Image& src, const std::vector<uint8_t>& alpha, const ImagePlan& plan, std::vector<float>& rows,
             std::string& err);

    std::unique_ptr<vision::CudaEncoder> enc_;
    void* region_ = nullptr;
    int cells_ = 0;   ///< cells the open region is sized for
};

}  // namespace basalt::models::inkling2
