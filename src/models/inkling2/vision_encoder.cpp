// src/models/inkling2/vision_encoder.cpp - see vision_encoder.hpp.
#include "models/inkling2/vision_encoder.hpp"

#include "basalt/vision/mmproj.hpp"

#include <algorithm>

namespace basalt::models::inkling2 {

namespace {
// sized for this picture and no smaller than a typical one, so a few pictures of a request share one region
int region_cells(const ImagePlan& plan) { return std::min(std::max(plan.cells(), 1024), kMaxCells); }
}  // namespace

ImageEncoder::ImageEncoder() = default;
ImageEncoder::~ImageEncoder() = default;

size_t ImageEncoder::weights_bytes() const { return enc_ ? enc_->weights_bytes() : 0; }
size_t ImageEncoder::region_bytes() const { return enc_ ? enc_->arena_bytes(kMaxCells) : 0; }

bool ImageEncoder::stage(const core::FileRange& section, int n_embd, std::string& err) {
    if (section.empty() || section.bytes == 0) { err = "--vision needs the pack's vision/mmproj section"; return false; }
    vision::Mmproj mp;
    auto e = std::make_unique<vision::CudaEncoder>();
    if (!mp.open(section, err, kProjector) || !e->stage(mp, true, err) || !e->create(err)) return false;
    if (mp.hp().proj != n_embd) { err = "the vision projector's width differs from the model's"; return false; }
    mp.release();
    enc_ = std::move(e);
    // the warm-up: a gray picture of a typical size in a region of its own, freed before the caches are sized
    vision::Image gray;
    gray.w = gray.h = 32 * kCell;
    gray.rgb.assign((size_t) gray.w * gray.h * 3, 128);
    ImagePlan plan;
    std::vector<float> rows;
    const bool ok = plan_image(gray.w, gray.h, plan, err) && run(gray, {}, plan, rows, err);
    region_.drop();
    if (!ok) enc_.reset();
    return ok;
}

bool ImageEncoder::run(const vision::Image& src, const std::vector<uint8_t>& alpha, const ImagePlan& plan,
                       std::vector<float>& rows, std::string& err) {
    if (region_.tokens() < plan.cells() && !region_.open(*enc_, region_cells(plan), err)) return false;
    vision::Image prepared;   // the decoded picture is what the encoder reads unless it has alpha or is scaled: no copy then
    if (!alpha.empty() || plan.scaled) prepared = encoder_input(src, alpha, plan);
    if (!enc_->encode_padded(prepared.rgb.empty() ? src : prepared, rows)) {
        err = "the vision encoder failed: " + enc_->last_error();
        return false;
    }
    return true;
}

bool ImageEncoder::encode(const uint8_t* data, size_t bytes, std::vector<float>& rows, int& n_rows, model::ImageGrid& grid,
                          std::string& err) {
    if (!enc_) { err = "started without --vision"; return false; }
    vision::Image img;
    std::vector<uint8_t> alpha;
    ImagePlan plan;
    if (!vision::decode_image(data, bytes, img, err, vision::kMaxImagePixels, &alpha) || !plan_image(img.w, img.h, plan, err) ||
        !run(img, alpha, plan, rows, err))
        return false;
    n_rows = plan.cells();
    grid = {plan.cols, plan.rows};
    if (rows.size() % (size_t) n_rows != 0) {
        err = "the vision encoder returned an unexpected number of rows";
        return false;
    }
    return true;
}

}  // namespace basalt::models::inkling2
