// src/vision/preprocess.cpp - see basalt/vision/preprocess.hpp.
#include "basalt/vision/preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_GIF
#define STBI_NO_STDIO
#include "stb_image.h"

namespace basalt::vision {

bool decode_image(const uint8_t* data, size_t len, Image& out, std::string& err, int64_t max_pixels) {
    if (data == nullptr || len == 0) { err = "empty image"; return false; }
    const int n = (int) std::min<size_t>(len, 0x7fffffff);
    int w = 0, h = 0, c = 0;
    // The header first: a file that claims more pixels than the guard allows is refused before the decode allocates
    // anything.  stb's own size checks still apply afterwards.
    if (!stbi_info_from_memory(data, n, &w, &h, &c)) {
        err = stbi_failure_reason() ? stbi_failure_reason() : "unreadable image";
        return false;
    }
    if ((int64_t) w * (int64_t) h > max_pixels) {
        const std::string limit = max_pixels >= (1 << 20) ? std::to_string(max_pixels >> 20) + " megapixels"
                                                          : std::to_string(max_pixels) + " pixels";
        err = "the image is " + std::to_string(w) + "x" + std::to_string(h) + " pixels; the limit is " + limit;
        return false;
    }
    uint8_t* px = stbi_load_from_memory(data, n, &w, &h, &c, 3);
    if (!px) {
        err = stbi_failure_reason() ? stbi_failure_reason() : "unreadable image";
        return false;
    }
    out.w = w;
    out.h = h;
    out.rgb.assign(px, px + (size_t) w * h * 3);
    stbi_image_free(px);
    return true;
}

Size fit_size(int w, int h, const Geometry& g) {
    if (w <= 0 || h <= 0) return {0, 0};
    const int f = g.patch * g.merge;
    const int64_t min_px = (int64_t) g.min_tokens * f * f, max_px = (int64_t) g.max_tokens * f * f;
    auto round_f = [f](float x) { return (int) std::round(x / (float) f) * f; };
    auto ceil_f = [f](float x) { return (int) std::ceil(x / (float) f) * f; };
    auto floor_f = [f](float x) { return (int) std::floor(x / (float) f) * f; };
    int wb = std::max(f, round_f((float) w)), hb = std::max(f, round_f((float) h));
    if ((int64_t) hb * wb > max_px) {
        const float beta = std::sqrt((float) h * (float) w / (float) max_px);
        hb = std::max(f, floor_f((float) h / beta));
        wb = std::max(f, floor_f((float) w / beta));
    } else if ((int64_t) hb * wb < min_px) {
        const float beta = std::sqrt((float) min_px / ((float) h * (float) w));
        hb = ceil_f((float) h * beta);
        wb = ceil_f((float) w * beta);
    }
    return {wb, hb};
}

namespace {

// Taps of a separable bicubic resample (a = -0.5) from `in` to `out` samples, 22-bit fixed point.
struct Taps {
    int ksize = 0;
    std::vector<int> start, count;
    std::vector<int32_t> w;
};

double cubic(double x) {
    constexpr double a = -0.5;
    x = std::fabs(x);
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
    return 0.0;
}

Taps make_taps(int in, int out) {
    constexpr int kBits = 22;
    const double scale = (double) in / out;
    const double fscale = std::max(1.0, scale);
    const double support = 2.0 * fscale;
    Taps t;
    t.ksize = (int) std::ceil(support) * 2 + 1;
    t.start.resize(out);
    t.count.resize(out);
    t.w.assign((size_t) out * t.ksize, 0);
    std::vector<double> tmp(t.ksize);
    for (int o = 0; o < out; ++o) {
        const double center = (o + 0.5) * scale;
        int lo = (int) (center - support + 0.5);
        int hi = (int) (center + support + 0.5);
        lo = std::max(lo, 0);
        hi = std::min(hi, in) - lo;
        const double ss = 1.0 / fscale;
        double sum = 0.0;
        for (int i = 0; i < hi; ++i) {
            tmp[i] = cubic((i + lo - center + 0.5) * ss);
            sum += tmp[i];
        }
        for (int i = 0; i < hi; ++i) {
            const double v = sum != 0.0 ? tmp[i] / sum : tmp[i];
            const double s = v * (double) (1 << kBits);
            t.w[(size_t) o * t.ksize + i] = (int32_t) (s + (v < 0 ? -0.5 : 0.5));
        }
        t.start[o] = lo;
        t.count[o] = hi;
    }
    return t;
}

uint8_t clip8(int v) { return (uint8_t) (v < 0 ? 0 : v > 255 ? 255 : v); }

std::vector<uint8_t> resample_x(const uint8_t* src, int iw, int ih, int ow, const Taps& t) {
    constexpr int kBits = 22;
    std::vector<uint8_t> dst((size_t) ow * ih * 3);
    for (int y = 0; y < ih; ++y) {
        const uint8_t* row = src + (size_t) y * iw * 3;
        uint8_t* d = dst.data() + (size_t) y * ow * 3;
        for (int x = 0; x < ow; ++x) {
            const int32_t* k = &t.w[(size_t) x * t.ksize];
            const uint8_t* p = row + (size_t) t.start[x] * 3;
            int32_t s0 = 1 << (kBits - 1), s1 = s0, s2 = s0;
            for (int i = 0; i < t.count[x]; ++i, p += 3) {
                s0 += p[0] * k[i];
                s1 += p[1] * k[i];
                s2 += p[2] * k[i];
            }
            d[x * 3] = clip8(s0 >> kBits);
            d[x * 3 + 1] = clip8(s1 >> kBits);
            d[x * 3 + 2] = clip8(s2 >> kBits);
        }
    }
    return dst;
}

std::vector<uint8_t> resample_y(const uint8_t* src, int iw, int oh, const Taps& t) {
    constexpr int kBits = 22;
    const size_t row = (size_t) iw * 3;
    std::vector<uint8_t> dst(row * oh);
    std::vector<int32_t> acc(row);
    for (int y = 0; y < oh; ++y) {
        std::fill(acc.begin(), acc.end(), 1 << (kBits - 1));
        const int32_t* k = &t.w[(size_t) y * t.ksize];
        for (int i = 0; i < t.count[y]; ++i) {
            const uint8_t* s = src + (size_t) (t.start[y] + i) * row;
            for (size_t j = 0; j < row; ++j) acc[j] += s[j] * k[i];
        }
        for (size_t j = 0; j < row; ++j) dst[(size_t) y * row + j] = clip8(acc[j] >> kBits);
    }
    return dst;
}

Image resize_bicubic(const Image& s, int ow, int oh) {
    Image d;
    d.w = ow;
    d.h = oh;
    const bool hx = ow != s.w, hy = oh != s.h;
    if (hx && hy) {
        const auto mid = resample_x(s.rgb.data(), s.w, s.h, ow, make_taps(s.w, ow));
        d.rgb = resample_y(mid.data(), ow, oh, make_taps(s.h, oh));
    } else if (hx) {
        d.rgb = resample_x(s.rgb.data(), s.w, s.h, ow, make_taps(s.w, ow));
    } else if (hy) {
        d.rgb = resample_y(s.rgb.data(), s.w, oh, make_taps(s.h, oh));
    } else {
        d.rgb = s.rgb;
    }
    return d;
}

}  // namespace

Image to_canvas(const Image& src, Size target) {
    if (src.w == target.w && src.h == target.h) return src;
    const float sw = (float) target.w / src.w, sh = (float) target.h / src.h;
    const float scale = std::min(sw, sh);
    const int nw = std::min((int) std::ceil(src.w * scale), target.w);
    const int nh = std::min((int) std::ceil(src.h * scale), target.h);
    const Image r = resize_bicubic(src, nw, nh);
    Image out;
    out.w = target.w;
    out.h = target.h;
    out.rgb.assign((size_t) target.w * target.h * 3, 0);
    const int ox = (target.w - nw) / 2, oy = (target.h - nh) / 2;
    for (int y = 0; y < nh; ++y) {
        if (y + oy < 0 || y + oy >= target.h) continue;
        for (int x = 0; x < nw; ++x) {
            if (x + ox < 0 || x + ox >= target.w) continue;
            std::copy_n(&r.rgb[((size_t) y * nw + x) * 3], 3, &out.rgb[((size_t) (y + oy) * target.w + x + ox) * 3]);
        }
    }
    return out;
}

void normalize_planar(const Image& c, std::vector<float>& out) {
    const size_t n = (size_t) c.w * c.h;
    out.resize(n * 3);
    for (size_t i = 0; i < n; ++i)
        for (int ch = 0; ch < 3; ++ch) out[ch * n + i] = ((float) c.rgb[i * 3 + ch] / 255.0f - 0.5f) / 0.5f;
}

void patchify(const std::vector<float>& planar, Size canvas, const Geometry& g, std::vector<float>& out) {
    const int P = g.patch, gx = canvas.w / P, gy = canvas.h / P, m = g.merge;
    const size_t plane = (size_t) canvas.w * canvas.h, row = (size_t) 3 * P * P;
    out.resize((size_t) gx * gy * row);
    size_t n = 0;
    for (int yb = 0; yb < gy; yb += m)
        for (int xb = 0; xb < gx; xb += m)
            for (int dy = 0; dy < m; ++dy)
                for (int dx = 0; dx < m; ++dx, ++n) {
                    float* d = &out[n * row];
                    for (int c = 0; c < 3; ++c)
                        for (int ky = 0; ky < P; ++ky)
                            std::copy_n(&planar[c * plane + (size_t) ((yb + dy) * P + ky) * canvas.w + (xb + dx) * P], P,
                                        d + ((size_t) c * P + ky) * P);
                }
}

}  // namespace basalt::vision
