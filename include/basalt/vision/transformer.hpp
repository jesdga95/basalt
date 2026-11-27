#pragma once

#include "basalt/vision/mmproj.hpp"

#include <memory>
#include <string>
#include <vector>

namespace basalt::vision {

inline constexpr int kTransformerPatch = 14;
inline constexpr int kTransformerMerge = 3;
inline constexpr int kTransformerPatchValues = 3 * kTransformerPatch * kTransformerPatch;

struct TransformerGrid {
    int gx = 0, gy = 0;
    int lw = 0, lh = 0;
    int n_patches() const { return gx * gy; }
    int n_cells() const { return lw * lh; }
    int n_block_rows() const { return lh * (lw + 1) + 2; }
};

/// Reusable 1024-wide BF16 ViT with 2D RoPE and a 3x3 GELU aligner, loaded from an embedded vision projector.
class TransformerBlas {
public:
    TransformerBlas();
    ~TransformerBlas();
    TransformerBlas(const TransformerBlas&) = delete;
    TransformerBlas& operator=(const TransformerBlas&) = delete;
    bool open(std::string& err);

    struct Impl;

private:
    friend class TransformerEncoder;
    std::unique_ptr<Impl> impl_;
};

class TransformerEncoder {
public:
    TransformerEncoder();
    ~TransformerEncoder();
    TransformerEncoder(const TransformerEncoder&) = delete;
    TransformerEncoder& operator=(const TransformerEncoder&) = delete;

    bool init(const Mmproj& projector, TransformerBlas& blas, std::string& err);
    size_t work_bytes(const TransformerGrid& grid) const;
    bool reserve(const TransformerGrid& grid, std::string& err);
    /// Returns [ceil(grid_height/3)*ceil(grid_width/3)][projector width] projected grid rows.
    bool encode_grid(const std::vector<float>& patches, const TransformerGrid& grid,
                     std::vector<float>& rows, std::string& err);
    /// V4.1-compatible START/NEWLINE/END block. V4 should use encode_grid and assemble its own N-layout.
    bool encode_block(const std::vector<float>& patches, const TransformerGrid& grid,
                      std::vector<float>& rows, std::string& err);
    int width() const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace basalt::vision
