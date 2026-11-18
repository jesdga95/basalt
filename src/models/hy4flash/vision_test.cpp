// src/models/hy4flash/vision_test.cpp - the Lanczos resize and Hy4-flash's canvas bounds.
#include "models/hy4flash/vision.hpp"

#include <cmath>
#include <cstdio>

using namespace basalt;

int main() {
    int failures = 0;
    auto check = [&](bool ok, const char* what) {
        std::printf("%-44s %s\n", what, ok ? "ok" : "FAIL");
        failures += !ok;
    };
    // a flat picture stays flat under both filters, up and down
    vision::Image flat;
    flat.w = 64;
    flat.h = 48;
    flat.rgb.assign((size_t) 64 * 48 * 3, 117);
    for (vision::Filter f : {vision::Filter::Bicubic, vision::Filter::Lanczos})
        for (auto [w, h] : {std::pair<int, int>{32, 24}, {200, 150}}) {
            const vision::Image r = vision::resize_to(flat, w, h, f);
            bool ok = r.w == w && r.h == h;
            for (uint8_t v : r.rgb) ok = ok && v == 117;
            check(ok, "flat picture stays flat");
        }
    // bicubic through the old entry point is the same function
    vision::Image ramp;
    ramp.w = 40;
    ramp.h = 40;
    for (int y = 0; y < 40; ++y)
        for (int x = 0; x < 40; ++x)
            for (int c = 0; c < 3; ++c) ramp.rgb.push_back((uint8_t) (x * 6 + y + c));
    check(vision::resize_bicubic_to(ramp, 17, 23).rgb == vision::resize_to(ramp, 17, 23, vision::Filter::Bicubic).rgb,
          "bicubic entry points agree");
    // a vertical step edge stays one step: monotone along the row, with the flat sides intact
    vision::Image step;
    step.w = 64;
    step.h = 8;
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 64; ++x)
            for (int c = 0; c < 3; ++c) step.rgb.push_back(x < 32 ? 20 : 220);
    const vision::Image dn = vision::resize_to(step, 32, 8, vision::Filter::Lanczos);
    bool ok = dn.rgb[0] == 20 && dn.rgb[((size_t) 31) * 3] == 220;
    for (int x = 1; x < 32; ++x) ok = ok && dn.rgb[(size_t) x * 3] >= dn.rgb[((size_t) x - 1) * 3] - 20;   // overshoot at the edge only
    check(ok, "lanczos step edge");
    // the processor's bounds: a tiny picture grows to 4 tokens, a huge one shrinks to at most 4,096
    const vision::Geometry g = basalt::models::hy4flash::vision_geometry();
    const vision::Size small = vision::fit_size(10, 10, g), big = vision::fit_size(8000, 6000, g);
    check(small.w * small.h >= 4 * 32 * 32 && small.w % 32 == 0 && small.h % 32 == 0, "small picture: at least 4 tokens");
    check((int64_t) big.w * big.h <= 4096ll * 32 * 32 && big.w % 32 == 0 && big.h % 32 == 0, "large picture: at most 4096 tokens");
    return failures ? 1 : 0;
}
