// src/models/hy4flash/vision_encoder.cpp - see vision_encoder.hpp.
#include "models/hy4flash/vision_encoder.hpp"

#include "basalt/vision/mmproj.hpp"

#include <cuda_runtime.h>

#include <algorithm>

namespace basalt::models::hy4flash {

ImageEncoder::ImageEncoder() = default;
ImageEncoder::~ImageEncoder() { release(); }

size_t ImageEncoder::weights_bytes() const { return enc_ ? enc_->weights_bytes() : 0; }

bool ImageEncoder::stage(const core::FileRange& section, int n_embd, std::string& err) {
    if (section.empty() || section.bytes == 0) { err = "--vision needs the pack's vision/mmproj section (convert with --mmproj)"; return false; }
    vision::Mmproj mp;
    auto e = std::make_unique<vision::CudaEncoder>();
    if (!mp.open(section, err, kProjector) || !e->stage(mp, true, err) || !e->create(err)) return false;
    if (mp.hp().proj != n_embd) { err = "the vision projector's width differs from the model's"; return false; }
    mp.release();
    enc_ = std::move(e);
    // the warm-up: a full-size gray picture in a region of its own, freed before the tier is sized
    const vision::Image gray{2048, 2048, std::vector<uint8_t>((size_t) 2048 * 2048 * 3, 128)};
    ImagePlan plan;
    std::vector<float> rows;
    int n = 0;
    model::ImageGrid grid;
    const bool ok = plan_image(gray.w, gray.h, plan, err) && encode(gray, plan, rows, n, grid, err);
    release();
    if (!ok) enc_.reset();
    return ok;
}

// sized for this picture and no smaller than a typical one, so a few pictures of a request share one region
static int region_tokens(const ImagePlan& plan) { return std::min(std::max(plan.n_merged(), 1024), vision_geometry().max_tokens); }

size_t ImageEncoder::region_bytes(const ImagePlan& plan) const { return enc_ ? enc_->arena_bytes(region_tokens(plan)) : 0; }

bool ImageEncoder::reserve(const ImagePlan& plan, std::string& err) {
    release();
    const int tokens = region_tokens(plan);
    const size_t bytes = enc_->arena_bytes(tokens);
    if (cudaMalloc(&region_, bytes) != cudaSuccess) {
        cudaGetLastError();
        region_ = nullptr;
        err = "no device memory for the image encoder (" + std::to_string((unsigned long long) (bytes >> 20)) + " MiB)";
        return false;
    }
    if (!enc_->attach(region_, bytes, tokens, err)) {
        release();
        return false;
    }
    tokens_ = tokens;
    return true;
}

void ImageEncoder::release() {
    if (!region_) return;
    enc_->detach();
    cudaFree(region_);
    region_ = nullptr;
    tokens_ = 0;
}

bool ImageEncoder::encode(const vision::Image& img, const ImagePlan& plan, std::vector<float>& rows, int& n_rows,
                          model::ImageGrid& grid, std::string& err) {
    if (!enc_) { err = "started without --vision"; return false; }
    if (needs_region(plan) && !reserve(plan, err)) return false;
    std::vector<float> patches;
    image_patches(img, plan, patches);
    if (!enc_->encode(patches, plan.canvas, rows, vision::TapFn{})) {
        err = "the vision encoder failed: " + enc_->last_error();
        return false;
    }
    n_rows = plan.n_rows();
    grid = image_grid(plan);
    if (n_rows <= 0 || rows.size() % (size_t) n_rows != 0) {
        err = "the vision encoder returned an unexpected number of rows";
        return false;
    }
    return true;
}

}  // namespace basalt::models::hy4flash
