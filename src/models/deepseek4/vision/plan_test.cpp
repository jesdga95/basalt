#include "basalt/models/deepseek4/vision.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace basalt::models::deepseek4;

namespace {
bool expect(bool ok, const char* what) {
    if (!ok) std::fprintf(stderr, "deepseek4 vision plan: %s\n", what);
    return ok;
}
}

int main() {
    bool ok = true;
    VisionPlan square;
    std::string err;
    ok &= expect(plan_image(224, 224, square, err), "224x224 plan succeeds");
    ok &= expect(square.width == 392 && square.height == 392 && square.n_vit_h == 28 && square.n_vit_w == 28 &&
                 square.n_llm_h == 10 && square.n_llm_w == 10 && square.n_tokens == 114,
                 "224x224 reference resize and N-layout token count");

    VisionPlan landscape;
    ok &= expect(plan_image(1280, 720, landscape, err), "1280x720 plan succeeds");
    ok &= expect(landscape.width == 1036 && landscape.height == 588 && landscape.n_llm_h == 14 &&
                 landscape.n_llm_w == 25 && landscape.n_tokens == 366,
                 "1280x720 reference resize stays within 384 token budget");

    VisionPlan panorama;
    ok &= expect(plan_image(8000, 1000, panorama, err) && panorama.stretch,
                 "8:1 panorama uses the source stretch path");
    ok &= expect(panorama.n_tokens + kVisionCompressPad - 1 <= kVisionMaxTokens,
                 "image rows plus maximum alignment padding fit token budget");

    ImageBlock block;
    ok &= expect(build_image_block(square, 0, block, err), "N-layout block builds at position zero");
    ok &= expect(block.alignment_pad == 3 && block.types.size() == 117 && block.patch_order.size() == 100,
                 "prompt alignment padding, token kinds and patch count");
    ok &= expect(block.patch_order.size() >= 4 && block.patch_order[0] == 0 && block.patch_order[1] == 10 &&
                 block.patch_order[2] == 1 && block.patch_order[3] == 11,
                 "N-layout interleaves two rows by columns");
    ok &= expect(block.types[0] == ImageToken::Pad && block.types[2] == ImageToken::Pad &&
                 block.types[3] == ImageToken::Start && block.types[4] == ImageToken::Patch,
                 "alignment pads precede image start and interleaved content");
    std::vector<float> grid(100 * 2), assembled;
    for (int i = 0; i < 100; ++i) { grid[2 * i] = (float) i; grid[2 * i + 1] = (float) -i; }
    const float start[] = {1000.f, 1001.f}, end[] = {2000.f, 2001.f};
    const float pad[] = {3000.f, 3001.f}, newline[] = {4000.f, 4001.f};
    ok &= expect(assemble_image_embeddings(block, grid, 2, start, end, pad, newline, assembled, err),
                 "N-layout image embedding assembly succeeds");
    ok &= expect(assembled.size() == block.types.size() * 2 && assembled[0] == 3000.f &&
                 assembled[6] == 1000.f && assembled[8] == 0.f && assembled[10] == 10.f &&
                 assembled[12] == 1.f && assembled[14] == 11.f && assembled[48] == 4000.f,
                 "learned special rows and interleaved encoded rows follow the image token types");

    VisionPlan odd_grid;
    ok &= expect(plan_image(150, 203, odd_grid, err) && odd_grid.width == 336 && odd_grid.height == 448 &&
                 odd_grid.n_llm_h == 11 && odd_grid.n_llm_w == 8 && odd_grid.n_tokens == 110,
                 "minimum-area upscale preserves an odd merged-grid height");
    ImageBlock odd_block;
    ok &= expect(build_image_block(odd_grid, 2, odd_block, err) && odd_block.alignment_pad == 1 &&
                 odd_block.padded_rows == 12 && odd_block.patch_order.size() == 88,
                 "odd merged-grid height pads and permutes the final N-layout row");

    basalt::vision::Image image;
    image.w = image.h = 1;
    image.rgb = {127, 127, 127};
    std::vector<float> patches;
    ok &= expect(image_patches(image, square, patches, err), "constant image produces patch rows");
    ok &= expect(patches.size() == (size_t) square.n_vit_h * square.n_vit_w * kVisionPatchValues &&
                 std::all_of(patches.begin(), patches.end(), [](float x) { return std::isfinite(x); }),
                 "patch rows have the expected shape and finite BF16-rounded values");

    err.clear();
    ok &= expect(!plan_image(0, 224, square, err) && !err.empty(), "invalid dimensions are rejected");
    return ok ? 0 : 1;
}
