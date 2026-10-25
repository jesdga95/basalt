// basalt/vision/preprocess.hpp - an image to the encoder's input: decode, fit to the token budget, normalise.
//
// The picture is resized (Pillow-style bicubic) straight to a multiple of 32 pixels on each side inside the token
// budget, normalised to [-1, 1] and cut into 16x16 patches ordered by 2x2 merge group.  Every step is deterministic
// integer or floating-point math so the result is reproducible bit for bit.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace basalt::vision {

struct Image {
    int w = 0, h = 0;
    std::vector<uint8_t> rgb;   ///< interleaved, w * h * 3
};

struct Geometry {
    int patch = 16;          ///< pixels per patch side
    int merge = 2;           ///< patches per merged token side
    int min_tokens = 8;      ///< merged tokens
    int max_tokens = 4096;
};

struct Size {
    int w = 0, h = 0;
    bool operator==(const Size& o) const { return w == o.w && h == o.h; }
};

/// JPEG, PNG, BMP or GIF (first frame) from memory, three channels (alpha dropped).  False on a bad image.
bool decode_image(const uint8_t* data, size_t len, Image& out, std::string& err);

/// The canvas size for a w x h picture: a multiple of patch * merge, within [min_tokens, max_tokens] merged tokens.
Size fit_size(int w, int h, const Geometry& g);

/// The picture resized to the `target` canvas, each side on its own (no padding).
Image to_canvas(const Image& src, Size target);

/// The canvas as planar float [3][h][w], (v / 255 - 0.5) / 0.5.
void normalize_planar(const Image& canvas, std::vector<float>& out);

/// Patch rows [n_patches][3 * patch * patch] (channel, row, column order), the patches in merge order: for each
/// 2x2 group of patches in raster order, its (dy, dx) patches.  `planar` as from normalize_planar.
void patchify(const std::vector<float>& planar, Size canvas, const Geometry& g, std::vector<float>& out);

}  // namespace basalt::vision
