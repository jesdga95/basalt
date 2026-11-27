// include/basalt/vision/encoder_region.hpp - the device region a request's pictures take, its room lent by the model.
#pragma once

#include "basalt/vision/encoder_cuda.hpp"

#include <cstdint>
#include <functional>
#include <string>

namespace basalt::vision {

/// Where a region's room comes from: `room(bytes, err)` leaves at least `bytes` free on the card (false: it cannot, with
/// nothing lent); `give_back()` runs once the region is gone, after the last picture of a request.
struct RegionLender {
    std::function<bool(uint64_t, std::string&)> room;
    std::function<void()> give_back;
};

/// A staged CudaEncoder's region: none at rest. `open` sizes one for `tokens` merged tokens, from the lender's room when a
/// lender is set, and attaches the encoder to it; `drop` gives the region back to the driver; `release` drops it and gives
/// the lender its room back.
class EncoderRegion {
public:
    EncoderRegion() = default;
    ~EncoderRegion() { drop(); }
    EncoderRegion(const EncoderRegion&) = delete;
    EncoderRegion& operator=(const EncoderRegion&) = delete;

    void set_lender(RegionLender lender) { lender_ = std::move(lender); }
    /// Replaces the open region (a larger picture after a smaller one: the lender adds to what it lent).
    bool open(CudaEncoder& enc, int tokens, std::string& err);
    void drop();
    void release();
    /// The merged tokens the open region is sized for (0: none open).
    int tokens() const { return tokens_; }

private:
    CudaEncoder* enc_ = nullptr;
    RegionLender lender_;
    void* region_ = nullptr;
    int tokens_ = 0;
    bool lent_ = false;   ///< the lender's room is out
};

}  // namespace basalt::vision
