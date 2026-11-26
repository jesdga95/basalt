#include "basalt/log.hpp"
#include "models/deepseek4/pack.hpp"
#include "basalt/models/deepseek4/vision.hpp"
#include "basalt/vision/mmproj.hpp"
#include "basalt/vision/transformer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::puts("usage: ds4_vision_tower_test PACK.basalt [reference.bin]");
        if (argc < 2) std::puts("skip: pass a DeepSeek-V4 Vision Basalt pack to exercise its projector");
        return 77;
    }
    basalt::models::deepseek4::Pack pack;
    std::string err;
    if (!basalt::models::deepseek4::open_pack(argv[1], pack, err)) {
        basalt::log::error("deepseek4 vision tower test", "%s", err.c_str());
        return 1;
    }
    basalt::vision::Mmproj mmproj;
    if (!mmproj.open(pack.vision, err, "deepseek4v")) {
        basalt::log::error("deepseek4 vision tower test", "%s", err.c_str());
        return 1;
    }
    basalt::vision::TransformerBlas blas;
    basalt::vision::TransformerEncoder encoder;
    if (!encoder.init(mmproj, blas, err)) {
        basalt::log::error("deepseek4 vision tower test", "encoder init: %s", err.c_str());
        return 1;
    }

    // A 3x3 patch grid is the smallest complete aligner cell and exercises the actual ViT, 2D RoPE and GELU projector.
    constexpr int gx = 3, gy = 3, width = 4096;
    const basalt::vision::TransformerGrid grid{gx, gy, 1, 1};
    std::vector<float> patches((size_t) gx * gy * basalt::models::deepseek4::kVisionPatchValues);
    for (size_t i = 0; i < patches.size(); ++i) patches[i] = (float) ((int) (i % 17) - 8) / 8.f;
    std::vector<float> cells;
    if (!encoder.encode_grid(patches, grid, cells, err)) {
        basalt::log::error("deepseek4 vision tower test", "encode grid: %s", err.c_str());
        return 1;
    }
    if (cells.size() != width || !std::all_of(cells.begin(), cells.end(), [](float x) { return std::isfinite(x); })) {
        basalt::log::error("deepseek4 vision tower test", "projected grid has an invalid shape or non-finite value");
        return 1;
    }
    if (argc == 3) {
        std::ifstream ref_file(argv[2], std::ios::binary);
        int32_t header[2] = {};
        ref_file.read(reinterpret_cast<char*>(header), sizeof(header));
        if (!ref_file || header[0] != 1 || header[1] != width) {
            basalt::log::error("deepseek4 vision tower test", "reference header must be [1, 4096]");
            return 1;
        }
        std::vector<float> reference((size_t) header[0] * header[1]);
        ref_file.read(reinterpret_cast<char*>(reference.data()), (std::streamsize) (reference.size() * sizeof(float)));
        if (!ref_file || ref_file.peek() != std::ifstream::traits_type::eof() ||
            !std::all_of(reference.begin(), reference.end(), [](float x) { return std::isfinite(x); })) {
            basalt::log::error("deepseek4 vision tower test", "reference payload is truncated, oversized, or non-finite");
            return 1;
        }
        double diff = 0.0, norm = 0.0;
        for (size_t i = 0; i < cells.size(); ++i) {
            const double d = (double) cells[i] - reference[i];
            diff += d * d;
            norm += (double) reference[i] * reference[i];
        }
        const double relative_rms = std::sqrt(diff / (norm + 1e-30));
        std::printf("projected-grid relative RMS %.3e\n", relative_rms);
        if (relative_rms >= 3e-2) {
            basalt::log::error("deepseek4 vision tower test", "projected grid differs from the source reference");
            return 1;
        }
    }

    basalt::models::deepseek4::VisionPlan plan;
    plan.n_llm_h = plan.n_llm_w = 1;
    plan.n_tokens = 6;
    basalt::models::deepseek4::ImageBlock block;
    if (!basalt::models::deepseek4::build_image_block(plan, 0, block, err)) {
        basalt::log::error("deepseek4 vision tower test", "N-layout: %s", err.c_str());
        return 1;
    }
    const auto row = [&](const char* name) -> const float* {
        const auto* t = mmproj.find(name);
        return t && t->type == 0 && t->elems() == width ? static_cast<const float*>(t->data) : nullptr;
    };
    std::vector<float> output;
    if (!basalt::models::deepseek4::assemble_image_embeddings(block, cells, width,
            row("v.token_embd.img_start"), row("v.token_embd.img_end"), row("v.token_embd.img_pad"),
            row("v.image_newline"), output, err)) {
        basalt::log::error("deepseek4 vision tower test", "assemble image block: %s", err.c_str());
        return 1;
    }
    if (output.size() != block.types.size() * (size_t) width ||
        !std::all_of(output.begin(), output.end(), [](float x) { return std::isfinite(x); })) {
        basalt::log::error("deepseek4 vision tower test", "assembled N-layout block has an invalid shape or non-finite value");
        return 1;
    }
    std::printf("deepseek4 vision tower: 32-layer projector encoded %dx%d patches into %zu V4 N-layout rows\n",
                gx, gy, block.types.size());
    return 0;
}
