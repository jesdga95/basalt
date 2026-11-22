// src/models/inkling2/vision.cpp - see vision.hpp.
#include "models/inkling2/vision.hpp"

namespace basalt::models::inkling2 {

bool plan_image(int w, int h, ImagePlan& p, std::string& err) {
    if (w <= 0 || h <= 0) {
        err = "an empty image";
        return false;
    }
    p = {};
    p.w = w;
    p.h = h;
    p.cols = (w + kCell - 1) / kCell;
    p.rows = (h + kCell - 1) / kCell;
    if ((int64_t) p.cols * p.rows <= kMaxCells) return true;
    vision::Geometry g;
    g.patch = kCell / 2;
    g.merge = 2;
    g.min_tokens = 1;
    g.max_tokens = kMaxCells;
    const vision::Size s = vision::fit_size(w, h, g);
    p.w = s.w;
    p.h = s.h;
    p.cols = s.w / kCell;
    p.rows = s.h / kCell;
    p.scaled = true;
    return true;
}

vision::Image encoder_input(const vision::Image& src, const std::vector<uint8_t>& alpha, const ImagePlan& p) {
    vision::Image img;
    img.w = src.w;
    img.h = src.h;
    if (alpha.empty()) {
        img.rgb = src.rgb;
    } else {
        // the reference processor's RGBA conversion: over 8 x 8 squares of 204 and 153, rounded to nearest
        img.rgb.resize(src.rgb.size());
        for (int y = 0; y < src.h; ++y)
            for (int x = 0; x < src.w; ++x) {
                const size_t i = (size_t) y * src.w + x;
                const int a = alpha[i], bg = ((y / 8 + x / 8) % 2 == 0) ? 204 : 153;
                for (int c = 0; c < 3; ++c) img.rgb[i * 3 + c] = (uint8_t) ((src.rgb[i * 3 + c] * a + bg * (255 - a) + 127) / 255);
            }
    }
    if (p.scaled) img = vision::resize_bicubic_to(img, p.w, p.h);
    return img;
}

}  // namespace basalt::models::inkling2
