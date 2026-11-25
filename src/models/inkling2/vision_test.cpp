// src/models/inkling2/vision_test.cpp - Inkling 2's picture plan and encoder input, no GPU.
//
//   inkling2_vision_test <picture with transparency.png> <opaque picture.png>
//
// The cells of a picture at its own size and of one past the budget (scaled with smart_resize's arithmetic, whole cells);
// transparency over the reference processor's checkerboard against its formula; a file's alpha kept apart by the decoder.
#include "models/inkling2/vision.hpp"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace basalt;
using namespace basalt::models::inkling2;

namespace {

int g_fail = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("FAIL %s\n", what.c_str());
        ++g_fail;
    }
}

void check_plan(int w, int h, int cols, int rows, bool scaled, int sw = 0, int sh = 0) {
    ImagePlan p;
    std::string err;
    const bool ok = plan_image(w, h, p, err);
    const std::string tag = std::to_string(w) + "x" + std::to_string(h);
    expect(ok && p.cols == cols && p.rows == rows && p.scaled == scaled,
           "plan " + tag + ": " + std::to_string(p.cols) + " x " + std::to_string(p.rows) + (p.scaled ? " scaled" : ""));
    if (scaled) expect(p.w == sw && p.h == sh && p.w == cols * kCell && p.h == rows * kCell, "scaled size " + tag);
    else expect(p.w == w && p.h == h, "own size " + tag);
    expect(p.cells() <= kMaxCells, "budget " + tag);
}

std::vector<uint8_t> read_file(const char* path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: inkling2_vision_test <picture with transparency.png> <opaque picture.png>\n");
        return 2;
    }
    // at its own size: whole cells, the last ones padded
    check_plan(40, 40, 1, 1, false);
    check_plan(1, 1, 1, 1, false);
    check_plan(57, 23, 2, 1, false);
    check_plan(640, 488, 16, 13, false);
    check_plan(4096, 4096, 103, 103, false);       // the model card's largest picture
    check_plan(4120, 4120, 103, 103, false);       // the budget exactly
    check_plan(16000, 40, 400, 1, false);          // long and thin: few cells
    // past the budget: scaled to the largest whole cells that fit, the aspect kept (smart_resize: floor after the scale)
    check_plan(4121, 4120, 103, 103, true, 4120, 4120);
    check_plan(7260, 5052, 123, 85, true, 4920, 3400);
    check_plan(8064, 6048, 118, 89, true, 4720, 3560);   // a 48-megapixel phone photo
    {
        ImagePlan p;
        std::string err;
        expect(!plan_image(0, 10, p, err) && !err.empty(), "an empty picture is refused");
    }

    // transparency over the checkerboard: (rgb * a + bg * (255 - a) + 127) / 255, bg 204 or 153 by 8-pixel squares
    {
        vision::Image src;
        src.w = 20;
        src.h = 11;
        src.rgb.resize((size_t) src.w * src.h * 3);
        std::vector<uint8_t> alpha((size_t) src.w * src.h);
        for (size_t i = 0; i < alpha.size(); ++i) {
            alpha[i] = (uint8_t) ((i * 53) % 256);
            for (int c = 0; c < 3; ++c) src.rgb[i * 3 + c] = (uint8_t) ((i * 7 + c * 91) % 256);
        }
        alpha[0] = 0;
        alpha[1] = 255;
        ImagePlan p;
        std::string err;
        plan_image(src.w, src.h, p, err);
        const vision::Image got = encoder_input(src, alpha, p);
        bool same = got.w == src.w && got.h == src.h && got.rgb.size() == src.rgb.size();
        for (int y = 0; same && y < src.h; ++y)
            for (int x = 0; x < src.w; ++x) {
                const size_t i = (size_t) y * src.w + x;
                const int bg = (x / 8 + y / 8) % 2 == 0 ? 204 : 153;
                for (int c = 0; c < 3; ++c)
                    same = same && got.rgb[i * 3 + c] == (src.rgb[i * 3 + c] * alpha[i] + bg * (255 - alpha[i]) + 127) / 255;
            }
        expect(same, "transparency over the checkerboard");
        expect(got.rgb[0] == 204 && got.rgb[1] == 204 && got.rgb[2] == 204, "a clear pixel is the checkerboard's");
        expect(got.rgb[3] == src.rgb[3] && got.rgb[5] == src.rgb[5], "an opaque pixel is its own");
        const vision::Image opaque = encoder_input(src, {}, p);
        expect(opaque.rgb == src.rgb, "no alpha: the picture as it is");
    }

    // the decoder keeps a file's alpha apart, and gives none for an opaque file
    {
        const std::vector<uint8_t> t = read_file(argv[1]), o = read_file(argv[2]);
        vision::Image img;
        std::vector<uint8_t> alpha;
        std::string err;
        const bool ok = !t.empty() && vision::decode_image(t.data(), t.size(), img, err, vision::kMaxImagePixels, &alpha);
        expect(ok && alpha.size() == (size_t) img.w * img.h && img.rgb.size() == alpha.size() * 3, "alpha kept apart: " + err);
        size_t clear = 0;
        for (uint8_t a : alpha) clear += a == 0;
        expect(clear > 0, "the transparent picture has clear pixels");
        vision::Image same;
        expect(vision::decode_image(t.data(), t.size(), same, err) && same.rgb == img.rgb, "the colours do not depend on keeping alpha");
        expect(!o.empty() && vision::decode_image(o.data(), o.size(), img, err, vision::kMaxImagePixels, &alpha) && alpha.empty(),
               "an opaque picture has no alpha");
    }
    std::printf("%s\n", g_fail ? "FAILED" : "all ok");
    return g_fail ? 1 : 0;
}
