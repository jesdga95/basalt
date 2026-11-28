// src/models/hy4flash/vision_test.cpp - the bicubic resize against Pillow on a real picture, Hy4-flash's canvas grid against
// the processor's smart_resize, and the rotary positions of a prompt with pictures against the reference's get_rope_index.
// Host only. The reference values come from transformers' hy_v4_vl (image_processing_pil_hy_v4_vl.smart_resize and
// modeling_hy_v4_vl.get_rope_index, run on CPU on the same token runs; 900 is a picture's pad, 5 text) and Pillow 12.3.0
// (Image.resize of assets/basalt-lockup-light.png converted to RGB). usage: hy4flash_vision_test [picture.png]
#include "models/hy4flash/vision.hpp"

#include "basalt/vision/mrope_positions.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>

namespace hy4 = basalt::models::hy4flash;
using namespace basalt;

namespace {

uint64_t fnv1a(const std::vector<uint8_t>& b) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint8_t x : b) h = (h ^ x) * 0x100000001b3ull;
    return h;
}

// One run of ids: 900 a picture's pad, 5 text; `grids` the pictures' patch grids (rows, columns) as the processor reports
// them. The positions as (t, h, w) per token.
struct RopeCase {
    const char* name;
    std::vector<int64_t> ids;
    std::vector<int32_t> want;
    std::vector<std::pair<int, int>> grids;
};

}  // namespace

int main(int argc, char** argv) {
    int failures = 0;
    auto check = [&](bool ok, const char* what) {
        std::printf("%-44s %s\n", what, ok ? "ok" : "FAIL");
        failures += !ok;
    };
    // a flat picture stays flat, up and down
    vision::Image flat;
    flat.w = 64;
    flat.h = 48;
    flat.rgb.assign((size_t) 64 * 48 * 3, 117);
    for (auto [w, h] : {std::pair<int, int>{32, 24}, {200, 150}}) {
        const vision::Image r = vision::resize_bicubic_to(flat, w, h);
        bool ok = r.w == w && r.h == h;
        for (uint8_t v : r.rgb) ok = ok && v == 117;
        check(ok, "flat picture stays flat");
    }
    // the processor's bounds: a tiny picture grows to 4 tokens, a huge one shrinks to at most 4,096
    const vision::Geometry g = hy4::vision_geometry();
    const vision::Size small = vision::fit_size(10, 10, g), big = vision::fit_size(8000, 6000, g);
    check(small.w * small.h >= 4 * 32 * 32 && small.w % 32 == 0 && small.h % 32 == 0, "small picture: at least 4 tokens");
    check((int64_t) big.w * big.h <= 4096ll * 32 * 32 && big.w % 32 == 0 && big.h % 32 == 0, "large picture: at most 4096 tokens");

    // the processor's canvas (smart_resize, factor 32) for pictures of many shapes; a ratio over 200 is refused
    struct Canvas { int w, h, ow, oh; };
    const Canvas canvases[] = {
        {1280, 720, 1280, 704}, {720, 1280, 704, 1280}, {1920, 1080, 1920, 1088}, {3840, 2160, 2720, 1536},
        {4096, 4096, 2048, 2048}, {2048, 2048, 2048, 2048}, {2049, 2047, 2048, 2048}, {10, 10, 64, 64}, {1, 1, 64, 64},
        {5, 3000, 0, 0}, {15, 3000, 32, 928}, {3000, 15, 928, 32}, {16, 16, 64, 64}, {31, 33, 64, 96}, {48, 48, 64, 64},
        {100, 100, 96, 96}, {512, 512, 512, 512}, {7680, 4320, 2720, 1536}, {8000, 6000, 2336, 1760}, {640, 480, 640, 480},
        {333, 555, 320, 544}, {1000, 10, 640, 32}, {5000, 25, 4992, 32}, {33, 6600, 32, 6592}, {63, 63, 64, 64},
        {64, 63, 64, 64}, {45, 45, 64, 64}, {1024, 1, 0, 0}, {8192, 41, 8192, 32}, {8192, 40, 0, 0}, {5110, 2100, 3168, 1312},
        {2945, 5664, 1472, 2816}, {5349, 4350, 2240, 1824}, {245, 3822, 256, 3808}, {2048, 5324, 1248, 3296},
        {432, 1292, 448, 1280}, {935, 3053, 928, 3040}, {3850, 2027, 2816, 1472}, {3127, 4462, 1696, 2432},
        {843, 4709, 832, 4704}, {2050, 115, 2048, 128}, {5997, 1783, 3744, 1088}, {3351, 2297, 2464, 1664},
        {1499, 3198, 1376, 2976},
    };
    for (const Canvas& c : canvases) {
        hy4::ImagePlan p;
        std::string err;
        const bool ok = hy4::plan_image(c.w, c.h, p, err);
        const bool want_ok = c.ow > 0;
        const bool same = ok == want_ok && (!ok || (p.canvas.w == c.ow && p.canvas.h == c.oh && p.gx == c.ow / 16 &&
                                                     p.gy == c.oh / 16 && p.n_rows() == (c.oh / 32) * (c.ow / 32 + 1)));
        if (!same) std::printf("  %d x %d -> %d x %d, the processor says %d x %d\n", c.w, c.h, ok ? p.canvas.w : 0, ok ? p.canvas.h : 0, c.ow, c.oh);
        failures += !same;
    }
    std::printf("%-44s %s\n", "canvas grid against smart_resize", failures ? "FAIL" : "ok");

    // rotary positions: the reference's get_rope_index, the shared prompt rule, and the session's cell-by-cell track
    const std::vector<RopeCase> ropes = {
        {"1 image 2x3", {5,5,5,900,900,900,900,900,900,900,900,5,5,5,5},
         {0,0,0,1,1,1,2,2,2,3,3,3,3,3,4,3,3,5,3,3,6,3,4,3,3,4,4,3,4,5,3,4,6,7,7,7,8,8,8,9,9,9,10,10,10}, {{4, 6}}},
        {"2 images, 3x2 then 1x4", {5,5,900,900,900,900,900,900,900,900,900,5,5,5,900,900,900,900,900,5,5},
         {0,0,0,1,1,1,2,2,2,2,2,3,2,2,4,2,3,2,2,3,3,2,3,4,2,4,2,2,4,3,2,4,4,5,5,5,6,6,6,7,7,7,8,8,8,8,8,9,8,8,10,8,8,11,8,8,12,13,13,13,14,14,14},
         {{6, 4}, {2, 8}}},
        {"wide 1x6", {5,900,900,900,900,900,900,900,5,5},
         {0,0,0,1,1,1,1,1,2,1,1,3,1,1,4,1,1,5,1,1,6,1,1,7,8,8,8,9,9,9}, {{2, 12}}},
        {"tall 6x1", {5,900,900,900,900,900,900,900,900,900,900,900,900,5,5},
         {0,0,0,1,1,1,1,1,2,1,2,1,1,2,2,1,3,1,1,3,2,1,4,1,1,4,2,1,5,1,1,5,2,1,6,1,1,6,2,7,7,7,8,8,8}, {{12, 2}}},
        {"image first", {900,900,900,900,900,900,5,5,5},
         {0,0,0,0,0,1,0,0,2,0,1,0,0,1,1,0,1,2,3,3,3,4,4,4,5,5,5}, {{4, 4}}},
    };
    for (const RopeCase& c : ropes) {
        const int64_t n = (int64_t) c.ids.size();
        // as the engine's prompt rule: grids of (ow + 1) columns, oh rows
        std::vector<vision::MediaGrid> grids;
        for (auto [gh, gw] : c.grids) grids.push_back({(int64_t) (gh / 2) * (gw / 2 + 1), gw / 2 + 1, gh / 2});
        std::vector<int32_t> pos((size_t) n * 3);
        std::vector<int64_t> runs;
        const std::string err = vision::mrope_positions(c.ids.data(), n, {900}, grids, n, pos.data(), runs);
        check(err.empty() && pos == c.want, (std::string("prompt rule: ") + c.name).c_str());
        // the session's track: a cell at a time
        hy4::RopeTrack track;
        std::vector<int32_t> got;
        size_t k = 0;
        for (int64_t i = 0; i < n;) {
            if (c.ids[(size_t) i] != 900) { const hy4::RopePos r = track.push_text(); got.insert(got.end(), {r.t, r.h, r.w}); ++i; continue; }
            const int32_t start = track.next();
            const int nx = (int) grids[k].nx;
            for (int j = 0; j < (int) grids[k].rows; ++j) { const hy4::RopePos r = track.push_row(start, nx, j); got.insert(got.end(), {r.t, r.h, r.w}); }
            i += grids[k++].rows;
        }
        check(got == c.want, (std::string("track: ") + c.name).c_str());
    }
    // a rewind drops the cells after it, a rewind into the text before a picture restores the text position and clears has_image
    {
        hy4::RopeTrack t;
        check(!t.has_image(), "track: empty track has no image");
        for (int i = 0; i < 3; ++i) t.push_text();
        check(!t.has_image(), "track: text-only track has no image");
        const int32_t start = t.next();
        for (int j = 0; j < 8; ++j) t.push_row(start, 4, j);
        check(t.next() == 7 && t.has_image(), "track: the text after a 2x3 picture");
        t.push_text();
        t.rewind(11);
        check(t.has_image(), "track: rewind after image keeps has_image");
        t.rewind(4);
        check(t.has_image(), "track: rewind inside image keeps has_image");
        t.rewind(3);
        check(t.size() == 3 && t.next() == 3 && !t.has_image(), "track: a rewind to before the picture clears has_image");
        t.push_row(3, 2, 0);
        check(t.has_image(), "track: pushing an image after rewind sets has_image again");
        t.clear();
        check(t.size() == 0 && !t.has_image(), "track: clear resets has_image");
    }

    // bicubic against Pillow on a real picture, up, down and mixed
    if (argc > 1) {
        std::ifstream f(argv[1], std::ios::binary);
        const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        vision::Image pic;
        std::string err;
        check(vision::decode_image(bytes.data(), bytes.size(), pic, err) && pic.w == 930 && pic.h == 262, "the picture decodes");
        struct Pillow { int w, h; uint64_t hash; };
        const Pillow cases[] = {
            {465, 131, 0x825c025a110a154full}, {1860, 524, 0xd748896413c7fabaull}, {640, 288, 0xdbbb9561afe16a22ull},
            {224, 512, 0x7d55f6a23839be02ull}, {928, 256, 0x10ecca7e5fc76a0full},
        };
        for (const Pillow& c : cases) {
            const vision::Image r = vision::resize_bicubic_to(pic, c.w, c.h);
            char what[96];
            std::snprintf(what, sizeof what, "pillow bicubic %dx%d -> %dx%d", pic.w, pic.h, c.w, c.h);
            check(fnv1a(r.rgb) == c.hash, what);
        }
    }
    return failures ? 1 : 0;
}
