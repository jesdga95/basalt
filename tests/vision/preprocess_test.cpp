// The image grid and canvas against the reference processor: sizes from transformers v4.57.1 smart_resize (Python
// round() ties to even, double arithmetic, no 32-pixel floor before the area check) and pixels from Pillow 13.0.0
// Image.resize(BICUBIC) of the same source, straight to the canvas with no padding.
#include "basalt/vision/preprocess.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace basalt::vision;

int main() {
    int failures = 0;
    struct Case { int w, h, ow, oh; };
    const Case cases[] = {
        {1280, 720, 1280, 704},
        {720, 1280, 704, 1280},
        {784, 784, 768, 768},
        {752, 752, 768, 768},
        {400, 400, 384, 384},
        {1920, 1080, 1920, 1088},
        {3840, 2160, 2720, 1536},
        {67, 47, 128, 96},
        {1000, 10, 928, 32},
        {2872, 16, 1216, 32},
        {2721, 1595, 2656, 1536},
        {90, 5, 384, 32},
    };
    const Geometry g{16, 2, 8, 4096};
    for (const Case& c : cases) {
        const Size got = fit_size(c.w, c.h, g);
        if (got.w != c.ow || got.h != c.oh) {
            std::fprintf(stderr, "FAIL fit_size %dx%d (%d..%d tokens): %dx%d, want %dx%d\n", c.w, c.h, g.min_tokens,
                         g.max_tokens, got.w, got.h, c.ow, c.oh);
            ++failures;
        }
    }

    // a source that is neither constant nor separable, one side down and one up; one canvas row per line
    Image src{13, 7, std::vector<uint8_t>(13 * 7 * 3)};
    for (int y = 0; y < src.h; ++y)
        for (int x = 0; x < src.w; ++x)
            for (int ch = 0; ch < 3; ++ch)
                src.rgb[((size_t) y * src.w + x) * 3 + ch] =
                    (uint8_t) ((x * 7 + y * 13 + ch * 50 + (x * y) % 31) & 255);
    const std::vector<uint8_t> want = {
          3,  53, 103,  18,  68, 118,  33,  83, 133,  49,  99, 149,  64, 114, 164,  79, 129, 179,
         11,  61, 111,  27,  77, 127,  42,  92, 142,  59, 109, 159,  75, 125, 175,  91, 141, 191,
         22,  72, 122,  39,  89, 139,  57, 107, 157,  75, 125, 175,  93, 143, 193, 111, 161, 211,
         31,  81, 131,  49,  99, 149,  69, 119, 169,  89, 139, 189, 109, 159, 209, 127, 177, 227,
         40,  90, 140,  60, 110, 160,  82, 132, 182, 104, 154, 204, 126, 176, 226, 126, 176, 226,
         50, 100, 150,  72, 122, 172,  95, 145, 195, 114, 164, 214, 129, 179, 229, 131, 181, 240,
         60, 110, 160,  83, 133, 183, 108, 158, 208, 120, 170, 220, 122, 172, 222, 147, 197, 241,
         70, 120, 170,  94, 144, 194, 121, 171, 221, 124, 174, 226, 139, 189, 244, 166, 216,  91,
         80, 130, 180, 107, 157, 207, 131, 181, 231, 132, 182, 234, 157, 207, 151, 171, 221,  15,
         87, 137, 187, 115, 165, 215, 136, 186, 236, 139, 189, 240, 168, 218,  43, 168, 218,   2,
    };
    const Image got = to_canvas(src, {6, 10});
    if (got.w != 6 || got.h != 10 || got.rgb != want) {
        std::fprintf(stderr, "FAIL to_canvas 13x7 -> 6x10: %dx%d, %zu bytes, not Pillow's pixels\n", got.w, got.h,
                     got.rgb.size());
        ++failures;
    }

    if (failures) {
        std::fprintf(stderr, "preprocess_test: %d failures\n", failures);
        return 1;
    }
    std::puts("preprocess_test: passed");
}
