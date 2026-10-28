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

bool decode_image(const uint8_t* data, size_t len, Image& out, std::string& err) {
    int w = 0, h = 0, c = 0;
    uint8_t* px = stbi_load_from_memory(data, (int) std::min<size_t>(len, 0x7fffffff), &w, &h, &c, 3);
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
    // the reference smart_resize's arithmetic: doubles, and Python round() (nearbyint rounds half to even in the
    // default rounding mode)
    auto round_f = [f](double x) { return (int) std::nearbyint(x / (double) f) * f; };
    auto ceil_f = [f](double x) { return (int) std::ceil(x / (double) f) * f; };
    auto floor_f = [f](double x) { return (int) std::floor(x / (double) f) * f; };
    int wb = round_f((double) w), hb = round_f((double) h);
    if ((int64_t) hb * wb > max_px) {
        const double beta = std::sqrt((double) h * (double) w / (double) max_px);
        hb = std::max(f, floor_f((double) h / beta));
        wb = std::max(f, floor_f((double) w / beta));
    } else if ((int64_t) hb * wb < min_px) {
        const double beta = std::sqrt((double) min_px / ((double) h * (double) w));
        hb = ceil_f((double) h * beta);
        wb = ceil_f((double) w * beta);
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
    return resize_bicubic(src, target.w, target.h);
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
