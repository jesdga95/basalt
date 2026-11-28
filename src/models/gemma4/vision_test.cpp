// src/models/gemma4/vision_test.cpp - Gemma 4's image plan against the processor's get_aspect_ratio_preserving_size (the
// sizes below are what it gives for 16-pixel patches pooled 3 x 3 and 280 soft tokens), and the picture the encoder reads:
// transparency over white, the resize only when the plan says so.
//
//   gemma4_vision_test
#include "models/gemma4/vision.hpp"

#include <cstdio>
#include <string>

using namespace basalt;
using namespace basalt::models::gemma4;

namespace {

int g_fail = 0;

void expect(bool ok, const std::string& what) {
    std::printf("%-64s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    g_fail += !ok;
}

}  // namespace

int main() {
    struct Case { int w, h, pw, ph; };
    // up or down to the largest whole cells in the budget; a long side stays within it; a picture already on whole cells
    // that fits is read as it is
    const Case cases[] = {{1024, 1024, 768, 768}, {640, 480, 912, 672},   {1920, 1080, 1056, 576}, {48, 48, 768, 768},
                          {10, 10, 768, 768},     {4000, 100, 5040, 96},  {100, 4000, 96, 5040},   {10000, 20, 13440, 48},
                          {20, 10000, 48, 13440}, {333, 777, 480, 1200},  {3000, 2000, 960, 624},  {816, 768, 816, 768}};
    for (const Case& c : cases) {
        ImagePlan p;
        std::string err;
        const bool ok = plan_image(c.w, c.h, 16, 3, 280, p, err);
        expect(ok && p.w == c.pw && p.h == c.ph && p.cols * 48 == p.w && p.rows * 48 == p.h && p.cells() <= 280 &&
                   p.resized == (c.w != c.pw || c.h != c.ph),
               std::to_string(c.w) + "x" + std::to_string(c.h) + " -> " + std::to_string(c.pw) + "x" + std::to_string(c.ph) + " (got " +
                   std::to_string(p.w) + "x" + std::to_string(p.h) + (ok ? "" : ", " + err) + ")");
    }
    ImagePlan p;
    std::string err;
    expect(!plan_image(0, 10, 16, 3, 280, p, err), "an empty picture is refused");

    // transparency over white, rounded to nearest; no resize for a plan at the picture's size
    vision::Image src;
    src.w = 2;
    src.h = 1;
    src.rgb = {0, 100, 200, 10, 20, 30};
    const std::vector<uint8_t> alpha = {255, 0};
    ImagePlan same;
    same.w = 2, same.h = 1;
    const vision::Image a = encoder_input(src, alpha, same);
    expect(a.w == 2 && a.h == 1 && a.rgb == std::vector<uint8_t>({0, 100, 200, 255, 255, 255}), "transparency composited over white");
    const vision::Image o = encoder_input(src, {}, same);
    expect(o.rgb == src.rgb, "an opaque picture at its plan's size is read as it is");
    ImagePlan big;
    big.w = 96, big.h = 48, big.resized = true;
    const vision::Image r = encoder_input(src, {}, big);
    expect(r.w == 96 && r.h == 48 && r.rgb.size() == (size_t) 96 * 48 * 3, "resized to the plan's size");
    std::printf("%s\n", g_fail ? "FAILED" : "all ok");
    return g_fail ? 1 : 0;
}
