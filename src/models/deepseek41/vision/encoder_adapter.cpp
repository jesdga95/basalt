#include "basalt/models/deepseek41/vision.hpp"

namespace basalt::models::deepseek41 {

VisionEncoder::VisionEncoder() = default;
VisionEncoder::~VisionEncoder() = default;
int VisionEncoder::width() const { return impl_.width(); }

bool VisionEncoder::init(const basalt::vision::Mmproj& mp, VisionBlas& blas, std::string& err) {
    return impl_.init(mp, blas, err);
}

bool VisionEncoder::reserve(const VisionPlan& plan, std::string& err) {
    const basalt::vision::TransformerGrid grid{plan.gx, plan.gy, plan.lw, plan.lh};
    return impl_.reserve(grid, err);
}

bool VisionEncoder::encode(const std::vector<float>& rows, const VisionPlan& plan,
                           std::vector<float>& out, std::string& err) {
    const basalt::vision::TransformerGrid grid{plan.gx, plan.gy, plan.lw, plan.lh};
    return impl_.encode_block(rows, grid, out, err);
}

}  // namespace basalt::models::deepseek41
