#include "basalt/models/deepseek4/vision.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace basalt::models::deepseek4 {
namespace {

int ceil_div(int a, int b) { return (a + b - 1) / b; }

void grid_tokens(int height, int width, int& grid_h, int& grid_w, int& tokens) {
    grid_h = ceil_div(height / kVisionPatch, kVisionMerge);
    grid_w = ceil_div(width / kVisionPatch, kVisionMerge);
    tokens = grid_h * (grid_w + 1) + 2;
    if (grid_h % 2) tokens += grid_w + 1;
    tokens += (grid_h + 1) / 2 * (grid_w + 1) % 2 * 2;
}

bool solve_resize(int height, int width, int budget, int& best_h, int& best_w) {
    if (height <= 0 || width <= 0 || budget <= 3) return false;
    const double ratio = (double) height / width;
    const double max_w_float = std::sqrt((budget - 2) / ratio + 0.25) - 0.5;
    const double max_h_float = max_w_float * ratio;
    int max_w = 0, max_h = 0;
    if (max_w_float < 1.0) {
        max_w = 1;
        max_h = (budget - 2) / (max_w + 1);
        if (max_h % 2) --max_h;
        best_w = max_w * kVisionPatch * kVisionMerge;
        best_h = max_h * kVisionPatch * kVisionMerge;
    } else if (max_h_float < 2.0) {
        max_h = 2;
        max_w = (budget - 2) / max_h - 1;
        if (max_w <= 1) return false;
        best_w = max_w * kVisionPatch * kVisionMerge;
        best_h = max_h * kVisionPatch * kVisionMerge;
    } else {
        max_w = (int) std::floor(max_w_float);
        max_h = (int) std::floor(max_h_float);
        if (max_h % 2) --max_h;
        const double scale = std::min((double) (max_w * kVisionPatch * kVisionMerge) / width,
                                      (double) (max_h * kVisionPatch * kVisionMerge) / height);
        best_w = (int) std::floor(width * scale / kVisionPatch) * kVisionPatch;
        best_h = (int) std::floor(height * scale / kVisionPatch) * kVisionPatch;
    }
    return best_h >= kVisionPatch && best_w >= kVisionPatch;
}

float bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    bits &= 0xffff0000u;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

}  // namespace

bool plan_image(int width, int height, VisionPlan& plan, std::string& err) {
    err.clear();
    plan = {};
    if (width <= 0 || height <= 0) { err = "image dimensions must be positive"; return false; }
    int fit_w = width, fit_h = height;
    if (fit_w > (int64_t) fit_h * kVisionMaxWhRatio) fit_w = fit_h * kVisionMaxWhRatio;
    if ((int64_t) fit_w * fit_h < kVisionMinPixels) {
        const double scale = std::sqrt((double) kVisionMinPixels / ((double) fit_w * fit_h));
        fit_w = std::max(1, (int) (fit_w * scale));
        fit_h = std::max(1, (int) (fit_h * scale));
    }
    int best_w = ceil_div(fit_w, kVisionPatch) * kVisionPatch;
    int best_h = ceil_div(fit_h, kVisionPatch) * kVisionPatch;
    int grid_h = 0, grid_w = 0, tokens = 0;
    grid_tokens(best_h, best_w, grid_h, grid_w, tokens);
    int budget = kVisionMaxTokens - (kVisionCompressPad - 1);
    while (tokens > budget) {
        if (!solve_resize(fit_h, fit_w, budget, best_h, best_w)) {
            err = "image dimensions cannot fit the DeepSeek-V4 vision token budget";
            return false;
        }
        grid_tokens(best_h, best_w, grid_h, grid_w, tokens);
        if (tokens <= budget) break;
        if (--budget <= 3) { err = "image resize did not converge within the vision token budget"; return false; }
    }
    plan.width = best_w;
    plan.height = best_h;
    plan.n_vit_h = best_h / kVisionPatch;
    plan.n_vit_w = best_w / kVisionPatch;
    plan.n_llm_h = grid_h;
    plan.n_llm_w = grid_w;
    plan.n_tokens = tokens;
    plan.stretch = (int64_t) width >= (int64_t) height * kVisionMaxWhRatio;
    return true;
}

bool image_patches(const basalt::vision::Image& image, const VisionPlan& plan,
                   std::vector<float>& patches, std::string& err) {
    err.clear();
    if (image.w <= 0 || image.h <= 0 || image.rgb.size() != (size_t) image.w * image.h * 3 ||
        plan.width <= 0 || plan.height <= 0 || plan.n_vit_w != plan.width / kVisionPatch ||
        plan.n_vit_h != plan.height / kVisionPatch) {
        err = "invalid image or DeepSeek-V4 vision plan";
        return false;
    }
    int resized_w = plan.width, resized_h = plan.height;
    if (!plan.stretch) {
        const double scale = std::min((double) plan.width / image.w, (double) plan.height / image.h);
        resized_w = std::max(1, std::min(plan.width, (int) std::nearbyint(image.w * scale)));
        resized_h = std::max(1, std::min(plan.height, (int) std::nearbyint(image.h * scale)));
    }
    const auto resized = basalt::vision::resize_bicubic_to(image, resized_w, resized_h);
    std::vector<uint8_t> canvas((size_t) plan.width * plan.height * 3, 127);
    const int ox = plan.stretch ? 0 : (plan.width - resized_w) / 2;
    const int oy = plan.stretch ? 0 : (plan.height - resized_h) / 2;
    for (int y = 0; y < resized_h; ++y)
        std::memcpy(canvas.data() + ((size_t) (y + oy) * plan.width + ox) * 3,
                    resized.rgb.data() + (size_t) y * resized_w * 3, (size_t) resized_w * 3);
    patches.resize((size_t) plan.n_vit_h * plan.n_vit_w * kVisionPatchValues);
    for (int py = 0; py < plan.n_vit_h; ++py)
        for (int px = 0; px < plan.n_vit_w; ++px)
            for (int c = 0; c < 3; ++c)
                for (int y = 0; y < kVisionPatch; ++y)
                    for (int x = 0; x < kVisionPatch; ++x) {
                        const uint8_t v = canvas[((size_t) (py * kVisionPatch + y) * plan.width +
                                                  px * kVisionPatch + x) * 3 + c];
                        const size_t at = ((size_t) py * plan.n_vit_w + px) * kVisionPatchValues +
                                          (size_t) c * kVisionPatch * kVisionPatch + (size_t) y * kVisionPatch + x;
                        patches[at] = bf16((float(v) / 255.f - 0.5f) / 0.5f);
                    }
    return true;
}

bool build_image_block(const VisionPlan& plan, int64_t start_pos, ImageBlock& block, std::string& err) {
    err.clear();
    block = {};
    if (start_pos < 0 || plan.n_llm_h <= 0 || plan.n_llm_w <= 0 || plan.n_tokens <= 0) {
        err = "invalid DeepSeek-V4 image block geometry or prompt position";
        return false;
    }
    const int h = plan.n_llm_h, w = plan.n_llm_w, row_len = w + 1;
    const int rows = h + (h & 1);
    const int last_pad = (rows / 2 * row_len) % 2 * 2;
    const int expected = h * row_len + (h & 1) * row_len + last_pad + 2;
    if (expected != plan.n_tokens) { err = "image block token count differs from the resize plan"; return false; }
    block.alignment_pad = kVisionCompressPad - 1 - (int) (start_pos % kVisionCompressPad);
    block.padded_rows = rows;
    block.last_row_pad = last_pad;
    block.types.assign((size_t) block.alignment_pad, ImageToken::Pad);
    block.types.push_back(ImageToken::Start);
    // Reorder the type grid and patch rows together into the reference N-layout.
    for (int pair = 0; pair < rows / 2; ++pair)
        for (int x = 0; x <= w; ++x)
            for (int dy = 0; dy < 2; ++dy) {
                const int y = pair * 2 + dy;
                if (y >= h) block.types.push_back(ImageToken::Pad);
                else if (x == w) block.types.push_back(ImageToken::Newline);
                else {
                    block.types.push_back(ImageToken::Patch);
                    block.patch_order.push_back(y * w + x);
                }
            }
    for (int i = 0; i < last_pad; ++i) block.types.push_back(ImageToken::Pad);
    block.types.push_back(ImageToken::End);

    if ((int) block.patch_order.size() != h * w || (int) block.types.size() != block.alignment_pad + plan.n_tokens) {
        err = "DeepSeek-V4 N-layout permutation has inconsistent geometry";
        return false;
    }
    return true;
}

}  // namespace basalt::models::deepseek4
