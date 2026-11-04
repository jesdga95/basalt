// DeepSeek-V4.1 image plan and patch rows; the grid arithmetic follows llama.cpp's mtmd (MIT) deepseek41v preprocessor.
#include "basalt/models/deepseek41/vision.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace basalt::models::deepseek41 {

namespace {

int align_up(int v, int a) { return (v + a - 1) / a * a; }

int block_tokens(int bh, int bw, int p, int r) {
    const int lh = ((bh / p) + r - 1) / r, lw = ((bw / p) + r - 1) / r;
    return lh * (lw + 1) + 2;
}

float bf16_round(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    u &= 0xffff0000u;
    std::memcpy(&v, &u, 4);
    return v;
}

}  // namespace

VisionPlan plan_image(int w, int h, int merge, int min_pixels, int max_tokens) {
    const int p = kVisionPatch, r = merge;
    if (min_pixels > 0 && (int64_t) w * h > 0 && (int64_t) w * h < min_pixels) {
        const double ratio = std::sqrt((double) min_pixels / ((double) w * (double) h));
        w = (int) (w * ratio);
        h = (int) (h * ratio);
    }
    w = std::max(w, 1);
    h = std::max(h, 1);
    int bw = align_up(w, p), bh = align_up(h, p);
    if (block_tokens(bh, bw, p, r) > max_tokens) {
        const double ratio = (double) h / w;
        const double max_w = std::sqrt((max_tokens - 2) / ratio + 0.25) - 0.5;
        const double max_h = max_w * ratio;
        const int cell = p * r;
        if (max_w < 1.0) {
            bh = (max_tokens - 2) / 2 * cell;
            bw = cell;
        } else if (max_h < 1.0) {
            bh = cell;
            bw = (max_tokens - 3) * cell;
        } else {
            const double beta = std::min(std::floor(max_w) * cell / w, std::floor(max_h) * cell / h);
            bh = (int) std::floor(h * beta / p) * p;
            bw = (int) std::floor(w * beta / p) * p;
        }
    }
    VisionPlan pl;
    pl.width = bw;
    pl.height = bh;
    pl.gx = bw / p;
    pl.gy = bh / p;
    pl.merge = r;
    pl.lw = (pl.gx + r - 1) / r;
    pl.lh = (pl.gy + r - 1) / r;
    return pl;
}

void image_patches(const basalt::vision::Image& src, const VisionPlan& plan, std::vector<float>& rows) {
    const int bw = plan.width, bh = plan.height;
    // aspect-preserving fit, centred on grey (python round: half to even)
    int nw = bw, nh = bh;
    const double im = (double) src.w / src.h, dest = (double) bw / bh;
    if (im != dest) {
        if (im > dest) nh = (int) std::nearbyint((double) src.h / src.w * bw);
        else nw = (int) std::nearbyint((double) src.w / src.h * bh);
    }
    nw = std::max(1, std::min(nw, bw));
    nh = std::max(1, std::min(nh, bh));
    const basalt::vision::Image fit = basalt::vision::resize_bicubic_to(src, nw, nh);
    std::vector<uint8_t> canvas((size_t) bw * bh * 3, 127);
    const int ox = (int) std::nearbyint((bw - nw) * 0.5), oy = (int) std::nearbyint((bh - nh) * 0.5);
    for (int y = 0; y < nh; ++y)
        std::memcpy(&canvas[((size_t) (y + oy) * bw + ox) * 3], &fit.rgb[(size_t) y * nw * 3], (size_t) nw * 3);

    const int p = kVisionPatch;
    rows.assign((size_t) plan.n_patches() * kVisionPatchValues, 0.f);
    for (int py = 0; py < plan.gy; ++py)
        for (int px = 0; px < plan.gx; ++px) {
            float* dst = &rows[((size_t) py * plan.gx + px) * kVisionPatchValues];
            for (int c = 0; c < 3; ++c)
                for (int y = 0; y < p; ++y)
                    for (int x = 0; x < p; ++x) {
                        const uint8_t v = canvas[((size_t) (py * p + y) * bw + px * p + x) * 3 + c];
                        dst[c * p * p + y * p + x] = bf16_round((v / 255.0f - 0.5f) / 0.5f);
                    }
        }
}

}  // namespace basalt::models::deepseek41
