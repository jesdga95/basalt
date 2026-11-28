// src/models/deepseek41/vision/plan_test.cpp - CPU unit tests for DeepSeek-V4.1 vision plan_image and image_patches
// bounds clamping (merge <= 0, max_tokens < 4, extreme aspect ratios, degenerate inputs).
#include "basalt/models/deepseek41/vision.hpp"

#include <cmath>
#include <cstdio>
#include <tuple>
#include <vector>

using namespace basalt::models::deepseek41;

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++g_fail;
    }
}

void check_valid_plan(const VisionPlan& pl, const char* what) {
    check(pl.width >= kVisionPatch && pl.height >= kVisionPatch &&
              pl.width % kVisionPatch == 0 && pl.height % kVisionPatch == 0 &&
              pl.gx == pl.width / kVisionPatch && pl.gy == pl.height / kVisionPatch &&
              pl.gx >= 1 && pl.gy >= 1 && pl.merge >= 1 &&
              pl.lw == (pl.gx + pl.merge - 1) / pl.merge &&
              pl.lh == (pl.gy + pl.merge - 1) / pl.merge &&
              pl.lw >= 1 && pl.lh >= 1 && pl.n_patches() >= 1 && pl.n_tokens() >= 4,
          what);
}

}  // namespace

int main() {
    // 1. Standard images with merge=3, min_pixels=3136, max_tokens=1024.
    {
        const VisionPlan sq = plan_image(224, 224, 3, 3136, kVisionMaxTokens);
        check_valid_plan(sq, "224x224 standard");
        check(sq.width == 224 && sq.height == 224 && sq.gx == 16 && sq.gy == 16 && sq.lw == 6 && sq.lh == 6,
              "224x224 exact grid");

        const VisionPlan tiny = plan_image(10, 10, 3, 3136, kVisionMaxTokens);
        check_valid_plan(tiny, "10x10 min_pixels upscale");
        check(tiny.width == 56 && tiny.height == 56, "10x10 scales to 56x56");

        const VisionPlan big = plan_image(4000, 3000, 3, 3136, kVisionMaxTokens);
        check_valid_plan(big, "4000x3000 downscale");
        check(big.n_tokens() <= kVisionMaxTokens, "4000x3000 within max_tokens");

        const VisionPlan tall = plan_image(100, 10000, 3, 3136, kVisionMaxTokens);
        check_valid_plan(tall, "100x10000 extreme portrait");
        check(tall.n_tokens() <= kVisionMaxTokens, "100x10000 within max_tokens");

        const VisionPlan wide = plan_image(10000, 100, 3, 3136, kVisionMaxTokens);
        check_valid_plan(wide, "10000x100 extreme landscape");
        check(wide.n_tokens() <= kVisionMaxTokens, "10000x100 within max_tokens");
    }

    // 2. Degenerate merge <= 0 is clamped to 1 (no division by zero).
    for (int m : {0, -1, -7}) {
        const VisionPlan pl = plan_image(128, 96, m, 3136, 1024);
        check_valid_plan(pl, "merge <= 0 clamped");
        check(pl.merge == 1, "merge clamped to 1");
    }

    // 3. Degenerate max_tokens < 4 is clamped to 4 (no zero/negative canvas).
    for (int mt : {-10, 0, 1, 2, 3}) {
        const VisionPlan pl = plan_image(1024, 768, 3, 0, mt);
        check_valid_plan(pl, "max_tokens < 4 clamped");
        check(pl.n_tokens() == 4, "max_tokens < 4 produces 1x1 merged block (4 tokens)");
    }

    // 4. Else-branch floor-to-zero edge case (merge=1, aspect ratio where floor(h*beta/p) == 0).
    {
        const VisionPlan pl = plan_image(10000, 5100, 1, 0, 6);
        check_valid_plan(pl, "else-branch floor-to-zero clamped to >= patch");
    }

    // 5. Non-positive w / h (including both negative so w*h > 0).
    for (auto [w, h, want_w, want_h] : {std::tuple<int, int, int, int>{0, 0, kVisionPatch, kVisionPatch},
                                        {-10, -10, kVisionPatch, kVisionPatch},
                                        {-5, 20, kVisionPatch, 2 * kVisionPatch},
                                        {20, -5, 2 * kVisionPatch, kVisionPatch}}) {
        const VisionPlan pl = plan_image(w, h, 3, 3136, 1024);
        check_valid_plan(pl, "non-positive w/h clamped safely");
        check(pl.width == want_w && pl.height == want_h, "non-positive w/h expected clamped dimensions");
    }

    // 6. image_patches on a valid image and on degenerate inputs.
    {
        basalt::vision::Image img;
        img.w = 64;
        img.h = 48;
        img.rgb.assign((size_t) img.w * (size_t) img.h * 3, 127);
        const VisionPlan pl = plan_image(img.w, img.h, 3, 3136, 1024);
        std::vector<float> rows;
        image_patches(img, pl, rows);
        bool all_finite = rows.size() == (size_t) pl.n_patches() * (size_t) kVisionPatchValues;
        for (float v : rows) all_finite = all_finite && std::isfinite(v) && v >= -1.0f && v <= 1.0f;
        check(all_finite, "image_patches produces finite [-1, 1] patch rows");

        // Degenerate image / plan clears rows safely.
        basalt::vision::Image empty;
        image_patches(empty, pl, rows);
        check(rows.empty(), "empty image clears rows");

        rows = {1.0f};
        image_patches(img, VisionPlan{}, rows);
        check(rows.empty(), "zero plan clears rows");

        VisionPlan bad_pl = pl;
        bad_pl.gx = pl.width;   // gx * kVisionPatch > width
        rows = {1.0f};
        image_patches(img, bad_pl, rows);
        check(rows.empty(), "inconsistent plan grid clears rows");
    }

    if (g_fail == 0) std::printf("ds41_vision_plan_test: ok\n");
    return g_fail ? 1 : 0;
}
