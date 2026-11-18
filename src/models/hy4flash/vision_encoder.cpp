// src/models/hy4flash/vision_encoder.cpp - see vision_encoder.hpp.
#include "models/hy4flash/vision_encoder.hpp"

#include "basalt/vision/mmproj.hpp"

#include <algorithm>

namespace basalt::models::hy4flash {

ImageEncoder::ImageEncoder() = default;
ImageEncoder::~ImageEncoder() = default;

void ImageEncoder::release() {
    enc_.reset();
    tokens_ = 0;
}

bool ImageEncoder::encode(const core::FileRange& section, const vision::Image& img, std::vector<float>& rows, int& n_rows,
                          model::ImageGrid& grid, std::string& err) {
    ImagePlan plan;
    if (!plan_image(img.w, img.h, plan, err)) return false;
    if (!enc_ || tokens_ < plan.n_merged()) {
        if (section.empty() || section.bytes == 0) { err = "this pack carries no vision projector (convert with --mmproj)"; return false; }
        enc_.reset();
        vision::Mmproj mp;
        if (!mp.open(section, err, kProjector)) return false;
        // sized for this picture and no smaller than a typical one, so a few pictures of a request share one encoder
        const int tokens = std::min(std::max(plan.n_merged(), 1024), vision_geometry().max_tokens);
        auto e = std::make_unique<vision::CudaEncoder>();
        if (!e->init(mp, tokens, err)) return false;
        enc_ = std::move(e);
        tokens_ = tokens;
    }
    std::vector<float> patches;
    image_patches(img, plan, patches);
    if (!enc_->encode(patches, plan.canvas, rows, vision::TapFn{})) {
        err = "the vision encoder failed: " + enc_->last_error();
        return false;
    }
    n_rows = plan.n_rows();
    grid = image_grid(plan);
    if ((int) (rows.size() / std::max<size_t>(1, (size_t) n_rows)) <= 0 || rows.size() % (size_t) n_rows != 0) {
        err = "the vision encoder returned an unexpected number of rows";
        return false;
    }
    return true;
}

}  // namespace basalt::models::hy4flash
