// Encodes one image with the DeepSeek-V4.1 vision encoder and compares the embedding block with a reference dump
// ([int32 n_tokens][int32 width][float data], as written by llama.cpp's mtmd debug option).
//   ds41_vision_parity PROJECTOR.gguf IMAGE REF.bin [OUT.bin]
#include "basalt/models/deepseek41/vision.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>

using namespace basalt::models::deepseek41;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s PROJECTOR.gguf IMAGE REF.bin [OUT.bin]\n", argv[0]);
        return 2;
    }
    std::string err;
    basalt::vision::Mmproj mp;
    if (!mp.open(basalt::core::FileRange(argv[1]), err, "deepseek41v")) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    std::ifstream f(argv[2], std::ios::binary);
    const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    basalt::vision::Image img;
    if (!basalt::vision::decode_image(bytes.data(), bytes.size(), img, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    const VisionPlan pl = plan_image(img.w, img.h, mp.hp().merge, mp.hp().min_pixels);
    std::vector<float> rows;
    image_patches(img, pl, rows);
    std::printf("image %dx%d -> canvas %dx%d, %d patches, %d block rows\n", img.w, img.h, pl.width, pl.height, pl.n_patches(), pl.n_tokens());
    VisionBlas blas;
    VisionEncoder enc;
    if (!enc.init(mp, pl.n_patches(), blas, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    std::vector<float> out;
    for (int rep = 0; rep < 2; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!enc.encode(rows, pl, out, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("encode %s: %.1f ms\n", rep ? "warm" : "cold", ms);
    }
    if (argc > 4) {
        std::ofstream o(argv[4], std::ios::binary);
        const int32_t hdr[2] = {pl.n_tokens(), enc.width()};
        o.write((const char*) hdr, sizeof hdr);
        o.write((const char*) out.data(), out.size() * sizeof(float));
    }
    std::ifstream r(argv[3], std::ios::binary);
    int32_t hdr[2] = {0, 0};
    r.read((char*) hdr, sizeof hdr);
    std::vector<float> ref((size_t) hdr[0] * hdr[1]);
    r.read((char*) ref.data(), ref.size() * sizeof(float));
    if (hdr[0] != pl.n_tokens() || hdr[1] != enc.width()) {
        std::printf("FAIL shape: ref %dx%d, ours %dx%d\n", hdr[0], hdr[1], pl.n_tokens(), enc.width());
        return 1;
    }
    double num = 0, den = 0, worst_row = 0;
    for (int t = 0; t < hdr[0]; ++t) {
        double rn = 0, rd = 0;
        for (int c = 0; c < hdr[1]; ++c) {
            const double d = (double) out[(size_t) t * hdr[1] + c] - ref[(size_t) t * hdr[1] + c];
            rn += d * d;
            rd += (double) ref[(size_t) t * hdr[1] + c] * ref[(size_t) t * hdr[1] + c];
        }
        num += rn;
        den += rd;
        worst_row = std::max(worst_row, std::sqrt(rn / (rd + 1e-30)));
    }
    const double rel = std::sqrt(num / den);
    std::printf("embedding rel RMS %.3e, worst row %.3e\n", rel, worst_row);
    std::printf("%s\n", rel < 3e-2 ? "PASS" : "FAIL");
    return rel < 3e-2 ? 0 : 1;
}
