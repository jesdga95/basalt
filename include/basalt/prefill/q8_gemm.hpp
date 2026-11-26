// include/basalt/prefill/q8_gemm.hpp - a prompt chunk's products against Q8_0 matrices in planes on Blackwell int8
// tensor cores (mma m16n8k32 s8). Weights use wgemm::q8_planes int8 codes [rows][K] alongside coalesced transposed
// fp16 block scales [K / 32][scale_plane_ld(rows)]. Activations are quantized per 32-value block to int8 codes
// [rows][K] with coalesced fp32 block scales [K / 32][act_scale_ld(rows)]. Each K=32 integer dot product is scaled
// by a_scale * w_scale and accumulated in fp32 in K order, so a row's bits do not depend on chunk length or tile
// placement.
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::prefill::q8 {

/// Row stride (in elements) of coalesced activation scales [cols / 32][act_scale_ld(rows)].
inline int64_t act_scale_ld(int64_t rows) { return (rows + 127) & ~int64_t(127); }

/// Row stride (in elements) of coalesced weight scales [cols / 32][scale_plane_ld(rows)].
inline int64_t scale_plane_ld(int64_t rows) { return (rows + 127) & ~int64_t(127); }

/// Bytes of `rows` activation rows of `cols` values (cols a multiple of 64): int8 codes [rows][cols], then fp32
/// block scales [cols / 32][act_scale_ld(rows)] (16-byte aligned).
size_t act_bytes(int64_t rows, int64_t cols);

/// The fp32 block scales [cols / 32][act_scale_ld(rows)] inside `act`, for a producer that writes activations itself.
float* act_scales(void* act, int64_t rows, int64_t cols);

/// Bytes of transposed fp16 weight block scales [cols / 32][scale_plane_ld(rows)] for a [rows x cols] Q8_0 matrix.
size_t scale_plane_bytes(int64_t rows, int64_t cols);

/// Transposes row-major wgemm::q8_planes fp16 scales [rows][cols / 32] into coalesced [cols / 32][scale_plane_ld(rows)]
/// with zero-padded tail rows. `cols` must be a multiple of 32.
void transpose_scales(const uint16_t* row_major, int64_t rows, int64_t cols, uint16_t* col_major, void* stream);

/// Quantizes rows of x (`ld` floats apart) to Q8 planes: per 32-value block, d = amax / 127 (fp32), q = round(x / d).
void quantize(const float* x, int64_t rows, int64_t cols, int64_t ld, void* act, void* stream);

enum class Epilogue { Store, Add };

/// One weight matrix of a segmented product: int8 codes `w` [rows][w_cols] and coalesced fp16 scales `w_scales_t`
/// [w_cols / 32][scale_plane_ld(rows)] built by `transpose_scales`.
struct Segment {
    const int8_t* w;
    int64_t rows;
    const uint16_t* w_scales_t;
    float* dst;
    int64_t ld_dst;
};

/// One product over up to three Q8 weight matrices that share `act` (e.g. attention Q/K/V): every segment gets what a
/// gemm of it alone would give (bitwise), in one launch. w_cols a multiple of 64; each segment's rows even and
/// ld_dst >= rows (ld_dst even, dst 8-byte aligned).
void gemm_segmented(const Segment* seg, int nseg, int64_t w_cols, const void* act, int64_t rows, Epilogue epi,
                    void* stream);

void gemm(const int8_t* w, const uint16_t* w_scales_t, int64_t w_rows, int64_t w_cols, const void* act, int64_t rows,
          float* dst, int64_t ld_dst, Epilogue epi, void* stream);

}  // namespace basalt::prefill::q8
