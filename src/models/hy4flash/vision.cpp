// src/models/hy4flash/vision.cpp - see vision.hpp.
#include "models/hy4flash/vision.hpp"

#include <algorithm>

namespace basalt::models::hy4flash {

bool plan_image(int w, int h, ImagePlan& p, std::string& err) {
    if (w <= 0 || h <= 0) { err = "an empty image"; return false; }
    if (std::max(w, h) / (double) std::min(w, h) > kMaxAspect) {
        err = "the image is " + std::to_string(w) + "x" + std::to_string(h) + "; the aspect ratio must stay under " +
              std::to_string(kMaxAspect);
        return false;
    }
    const vision::Geometry g = vision_geometry();
    p.canvas = vision::fit_size(w, h, g);
    const int f = g.patch * g.merge;
    p.gx = p.canvas.w / g.patch;
    p.gy = p.canvas.h / g.patch;
    p.ow = p.canvas.w / f;
    p.oh = p.canvas.h / f;
    return true;
}

void image_patches(const vision::Image& src, const ImagePlan& plan, std::vector<float>& patches) {
    std::vector<float> planar;
    vision::normalize_planar(vision::to_canvas(src, plan.canvas), planar);
    vision::patchify(planar, plan.canvas, vision_geometry(), patches);
}

bool read_vision_tokens(const core::Json& meta, VisionTokens& t, std::string& err) {
    const core::Json* c = meta.get("hf_config");
    if (!c || !c->is_object()) { err = "the pack's meta has no hf_config (convert with --hf-dir)"; return false; }
    auto id = [&](const char* k, int32_t& out) {
        const core::Json* j = c->get(k);
        if (!j || !j->is_number()) { err = std::string("hf_config has no ") + k; return false; }
        out = (int32_t) j->num();
        return true;
    };
    return id("image_start_token_id", t.image_start) && id("image_token_id", t.image_pad) &&
           id("image_end_token_id", t.image_end) && id("image_newline_token_id", t.image_newline);
}

}  // namespace basalt::models::hy4flash
