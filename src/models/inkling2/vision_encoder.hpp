// src/models/inkling2/vision_encoder.hpp - the image encoder on the model's card, as the session drives it.
#pragma once

#include "basalt/core/model_file.hpp"
#include "basalt/model/adapter.hpp"
#include "basalt/vision/encoder_cuda.hpp"
#include "models/inkling2/vision.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace basalt::models::inkling2 {

/// --vision (dynamic): the projector of the pack's `vision/mmproj` section staged once in page-locked RAM. It holds no VRAM
/// at rest: a request's pictures get a device region (the weights and the work buffers), its room lent by the model (the
/// lender), and `release` gives the region back and the lender its room.
class ImageEncoder {
public:
    /// Where a region's room comes from: `room(bytes, err)` leaves at least `bytes` free on the card (false: it cannot, with
    /// nothing lent); `give_back()` runs once the region is gone, after the last picture of a request.
    struct Lender {
        std::function<bool(uint64_t, std::string&)> room;
        std::function<void()> give_back;
    };

    ImageEncoder();
    ~ImageEncoder();
    ImageEncoder(const ImageEncoder&) = delete;
    ImageEncoder& operator=(const ImageEncoder&) = delete;

    /// Stages the projector, makes the stream and product handle, and encodes one picture of the largest size in a temporary
    /// region (the product library loads its kernels on first use): call before the caches are sized.
    bool stage(const core::FileRange& section, int n_embd, std::string& err);
    bool staged() const { return enc_ != nullptr; }
    /// The model's lender, set once after `stage` (the warm-up's region takes free memory, before the caches are sized).
    void set_lender(Lender lender) { lender_ = std::move(lender); }
    size_t weights_bytes() const;
    /// The device bytes of the region for the largest picture (kMaxCells).
    size_t region_bytes() const;

    /// A picture file's rows [cells][n_embd] and its grid (cells across and down); opens the region when the open one is too
    /// small.
    bool encode(const uint8_t* data, size_t bytes, std::vector<float>& rows, int& n_rows, model::ImageGrid& grid,
                std::string& err);
    /// The region goes back to the driver and its room to the lender.
    void release();

private:
    bool reserve(int cells, std::string& err);
    void drop();   ///< the region back to the driver
    bool run(const vision::Image& src, const std::vector<uint8_t>& alpha, const ImagePlan& plan, std::vector<float>& rows,
             std::string& err);

    std::unique_ptr<vision::CudaEncoder> enc_;
    Lender lender_;
    void* region_ = nullptr;
    int cells_ = 0;        ///< cells the open region is sized for
    bool lent_ = false;    ///< the lender's room is out
};

}  // namespace basalt::models::inkling2
