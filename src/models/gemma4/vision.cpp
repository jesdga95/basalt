// src/models/gemma4/vision.cpp - see vision.hpp.
#include "models/gemma4/vision.hpp"

#include <algorithm>
#include <cmath>

namespace basalt::models::gemma4 {

bool plan_image(int w, int h, int patch, int k, int max_cells, ImagePlan& p, std::string& err) {
    if (w <= 0 || h <= 0) {
        err = "an empty image";
        return false;
    }
    p = {};
    const int64_t max_patches = (int64_t) max_cells * k * k, side = (int64_t) k * patch;
    const double target_px = (double) max_patches * patch * patch;
    const double factor = std::sqrt(target_px / ((double) h * (double) w));
    int64_t th = (int64_t) std::floor(factor * h / (double) side) * side;
    int64_t tw = (int64_t) std::floor(factor * w / (double) side) * side;
    if (th == 0 && tw == 0) {
        err = "an image too small to resize";
        return false;
    }
    const int64_t max_side = (max_patches / ((int64_t) k * k)) * side;
    if (th == 0) {
        th = side;
        tw = std::min((int64_t) std::floor((double) w / (double) h) * side, max_side);
    } else if (tw == 0) {
        tw = side;
        th = std::min((int64_t) std::floor((double) h / (double) w) * side, max_side);
    }
    if ((double) th * (double) tw > target_px) {
        err = "an image whose aspect ratio leaves it more patches than the budget";
        return false;
    }
    p.w = (int) tw;
    p.h = (int) th;
    p.cols = (int) (tw / side);
    p.rows = (int) (th / side);
    p.resized = p.w != w || p.h != h;
    return true;
}

vision::Image encoder_input(const vision::Image& src, const std::vector<uint8_t>& alpha, const ImagePlan& p) {
    vision::Image img;
    img.w = src.w;
    img.h = src.h;
    if (alpha.empty()) {
        img.rgb = src.rgb;
    } else {
        // the processor's RGB conversion: alpha-composited over opaque white, rounded to nearest
        img.rgb.resize(src.rgb.size());
        for (size_t i = 0; i < (size_t) src.w * src.h; ++i) {
            const int a = alpha[i];
            for (int c = 0; c < 3; ++c) img.rgb[i * 3 + c] = (uint8_t) ((src.rgb[i * 3 + c] * a + 255 * (255 - a) + 127) / 255);
        }
    }
    if (p.resized) img = vision::resize_bicubic_to(img, p.w, p.h);
    return img;
}

}  // namespace basalt::models::gemma4
