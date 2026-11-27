// src/models/gemma4/vision_encoder.hpp - the image encoder on the model's card, as the session drives it.
#pragma once

#include "basalt/core/model_file.hpp"
#include "basalt/vision/encoder_cuda.hpp"
#include "basalt/vision/encoder_region.hpp"
#include "models/gemma4/vision.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace basalt::models::gemma4 {

/// --vision (dynamic): the projector of the pack's `vision/mmproj` section staged once in page-locked RAM. It holds no VRAM
/// at rest: a request's pictures get one device region (the weights before the tower's layers, two layers streamed through
/// it and the work buffers for the largest picture), its room lent by the model (the lender, vision::RegionLender), and
/// `release` gives the region back and the lender its room.
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
    /// The model's lender, set once after `stage` (the warm-up's region takes free memory, before the caches are sized).
    void set_lender(vision::RegionLender lender) { region_.set_lender(std::move(lender)); }
    size_t weights_bytes() const;
    /// The device bytes of the region a request's pictures take.
    size_t region_bytes() const;

    /// A picture file's rows [cells][n_embd] (the cells in raster order); opens the region when none is open.
    bool encode(const uint8_t* data, size_t bytes, std::vector<float>& rows, int& n_rows, std::string& err);
    /// The region goes back to the driver and its room to the lender.
    void release() { region_.release(); }

private:
    bool run(const vision::Image& src, const std::vector<uint8_t>& alpha, const ImagePlan& plan, std::vector<float>& rows,
             std::string& err);

    std::unique_ptr<vision::CudaEncoder> enc_;
    vision::EncoderRegion region_;   ///< after enc_: dropped before it
    int patch_ = 0, k_ = 0, max_cells_ = 0;
};

}  // namespace basalt::models::gemma4
