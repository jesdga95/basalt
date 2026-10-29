// include/basalt/prefill/gemm_fp8.hpp - E4M3 products for the prompt path's dense projections.
//
// Rows of FP16 values become E4M3 (FP8, 3 mantissa bits) with power-of-two scales, and cuBLASLt multiplies them on the
// FP8 tensor cores with FP32 accumulation and FP32 output.  A power-of-two scale changes no mantissa, so a value's E4M3
// rounding is the same under any scale that keeps it in the normal range.  Scale granularities:
//   Row:     one FP32 scale per row (cuBLASLt's outer-vector scaling): a token's rounding never depends on other tokens.
//   Block32: one UE8M0 scale per 32 values of a row, in cuBLASLt's tiled block-scale layout (MXFP8).
//   Tensor:  one FP32 scale for the whole matrix (a token's scale follows the other tokens of the chunk).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace basalt::prefill::fp8 {

enum class Scale { Row, Block32, Tensor };
const char* scale_name(Scale s);

/// Bytes of the scales of `rows` rows of `cols` values (`cols` a multiple of 32).
size_t scale_bytes(Scale s, int64_t rows, int64_t cols);

/// in [rows, cols] (FP16 bits, row stride `ld` values, 16-byte aligned rows) -> q [rows, cols] E4M3 (row stride
/// `cols`) and its scales, the values rounded to nearest and saturated at 448:
///   Row:     scales[row0 + r] (FP32) = the power of two that maps row r's largest magnitude into (224, 448]
///   Block32: one UE8M0 byte per 32 values, at row row0 + r of the tiled layout of a `total_rows`-row matrix
///   Tensor:  every value divided by *tensor_scale (from `tensor_scale`); `scales` unused
/// `row0` lets a caller convert a matrix in row slices.
void to_e4m3(const uint16_t* in, int64_t rows, int64_t cols, int64_t ld, uint8_t* q, void* scales, int64_t row0,
             int64_t total_rows, Scale s, const float* tensor_scale, void* stream);

/// *out = the power of two that maps the largest magnitude of in[0, n) (FP16 bits) into (224, 448].
void tensor_scale(const uint16_t* in, int64_t n, float* out, void* stream);

/// cuBLASLt for Y[T, N] (FP32, row stride ldy) = X[T, K] . W[N, K]^T with X and W as E4M3 rows and their scales.
class Lt {
public:
    Lt() = default;
    ~Lt();
    Lt(const Lt&) = delete;
    Lt& operator=(const Lt&) = delete;

    bool init(std::string& err);
    bool ready() const { return state_ != nullptr; }
    /// The algorithms cuBLASLt offers for the shape (0: the shape or the scale mode is not supported here).  The
    /// scale pointers (device memory; a block-scaled heuristic may want them set) are replaced by those of each run.
    int algos(Scale s, int64_t T, int64_t N, int64_t K, int64_t ldy, size_t ws_bytes, const void* w_scales,
              const void* x_scales, bool fast_accum = false);
    /// `algo`: which of the offered algorithms (the first is cuBLASLt's pick).  false: none offered, or the call failed.
    bool run(Scale s, const uint8_t* W, const void* w_scales, const uint8_t* X, const void* x_scales, float* Y,
             int64_t T, int64_t N, int64_t K, int64_t ldy, void* workspace, size_t ws_bytes, void* stream,
             bool fast_accum = false, int algo = 0);

private:
    struct State;
    State* state_ = nullptr;
};

}  // namespace basalt::prefill::fp8
