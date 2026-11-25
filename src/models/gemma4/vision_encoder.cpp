// src/models/gemma4/vision_encoder.cpp - see vision_encoder.hpp.
#include "models/gemma4/vision_encoder.hpp"

#include "basalt/vision/mmproj.hpp"

#include <cuda_runtime.h>

namespace basalt::models::gemma4 {

ImageEncoder::ImageEncoder() = default;
ImageEncoder::~ImageEncoder() { release(); }

size_t ImageEncoder::weights_bytes() const { return enc_ ? enc_->weights_bytes() : 0; }
size_t ImageEncoder::region_bytes() const { return enc_ ? enc_->arena_bytes(max_cells_) : 0; }

bool ImageEncoder::stage(const core::FileRange& section, int n_embd, int max_cells, std::string& err) {
    if (section.empty() || section.bytes == 0) { err = "--vision needs the pack's vision/mmproj section"; return false; }
    if (max_cells < 1) { err = "the pack names no soft-token budget for a picture"; return false; }
    vision::Mmproj mp;
    auto e = std::make_unique<vision::CudaEncoder>();
    if (!mp.open(section, err, kProjector) || !e->stage(mp, true, err) || !e->create(err)) return false;
    if (mp.hp().proj != n_embd) { err = "the vision projector's width differs from the model's"; return false; }
    patch_ = mp.hp().patch;
    k_ = mp.hp().merge;
    max_cells_ = max_cells;
    mp.release();
    enc_ = std::move(e);
    // the warm-up: a gray picture of the largest square size in a region of its own, freed before the caches are sized
    vision::Image gray;
    gray.w = gray.h = 1024;
    gray.rgb.assign((size_t) gray.w * gray.h * 3, 128);
    ImagePlan plan;
    std::vector<float> rows;
    const bool ok = plan_image(gray.w, gray.h, patch_, k_, max_cells_, plan, err) && run(gray, {}, plan, rows, err);
    release();
    if (!ok) enc_.reset();
    return ok;
}

void ImageEncoder::release() {
    if (!region_) return;
    enc_->detach();
    cudaFree(region_);
    region_ = nullptr;
}

bool ImageEncoder::run(const vision::Image& src, const std::vector<uint8_t>& alpha, const ImagePlan& plan,
                       std::vector<float>& rows, std::string& err) {
    if (!region_) {
        const size_t bytes = enc_->arena_bytes(max_cells_);
        if (cudaMalloc(&region_, bytes) != cudaSuccess) {
            cudaGetLastError();
            region_ = nullptr;
            err = "no device memory for the image encoder (" + std::to_string((unsigned long long) (bytes >> 20)) + " MiB)";
            return false;
        }
        if (!enc_->attach(region_, bytes, max_cells_, err)) {
            release();
            return false;
        }
    }
    vision::Image prepared;   // the decoded picture is what the encoder reads unless it has alpha or is resized: no copy then
    if (!alpha.empty() || plan.resized) prepared = encoder_input(src, alpha, plan);
    if (!enc_->encode_padded(prepared.rgb.empty() ? src : prepared, rows)) {
        err = "the vision encoder failed: " + enc_->last_error();
        return false;
    }
    return true;
}

bool ImageEncoder::encode(const uint8_t* data, size_t bytes, std::vector<float>& rows, int& n_rows, std::string& err) {
    if (!enc_) { err = "started without --vision"; return false; }
    vision::Image img;
    std::vector<uint8_t> alpha;
    ImagePlan plan;
    if (!vision::decode_image(data, bytes, img, err, vision::kMaxImagePixels, &alpha) ||
        !plan_image(img.w, img.h, patch_, k_, max_cells_, plan, err) || !run(img, alpha, plan, rows, err))
        return false;
    n_rows = plan.cells();
    if (rows.empty() || rows.size() % (size_t) n_rows != 0) {
        err = "the vision encoder returned an unexpected number of rows";
        return false;
    }
    return true;
}

}  // namespace basalt::models::gemma4
