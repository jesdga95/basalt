// include/basalt/prefill/gemm.hpp - the batched projections of prompt processing.
//
// Every projection of a chunk of T tokens is Y[T, N] = X[T, K] . W[N, K]^T with W row-major (the GGUF / pack layout)
// and FP32 outputs.  Weights are BF16 on the device - either already (the pack's BF16 tensors) or dequantized from
// their native GGUF blocks into a reusable scratch (`dequant_bf16`) right before the product - and activations are
// rounded to BF16, which is also what llama.cpp's batched CUDA path does.  Tensor-core GEMM through cuBLAS.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace basalt::core { class WeightTable; struct ModelGeometry; }

namespace basalt::prefill {

class GemmPlanError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class Gemm {
public:
    static constexpr int64_t kPrefillScratchElems = 32ll << 20;
    struct Shape {
        bool bf16;
        int64_t n, k, ldy;
        int align;
    };
    Gemm() = default;
    ~Gemm();
    Gemm(const Gemm&) = delete;
    Gemm& operator=(const Gemm&) = delete;

    /// Prepares the current device's cuBLAS and cuBLASLt handles for the first Gemm there. A server calls this
    /// before sizing the expert cache so their device memory is already out of the free figure.
    static bool prepare_handle(std::string& err);
    static std::vector<Shape> model_shapes(const core::WeightTable& wt, const core::ModelGeometry& g,
                                           int64_t scratch_elems);
    static bool prepare_model(const core::WeightTable& wt, const core::ModelGeometry& g, int64_t scratch_elems,
                              std::string& err);
    static bool prepare_shapes(const std::vector<Shape>& shapes, std::string& err);

    /// `scratch_elems`: BF16 elements of the dequantization scratch (the largest weight dequantized at once).
    bool init(void* stream, int64_t scratch_elems, std::string& err);
    /// The same with caller-owned device buffers (the prompt path borrowing expert-cache slots).
    bool init_external(void* stream, uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes,
                       std::string& err);

    /// Y[T, N] (fp32, row stride ldy) = X[T, K] (bf16, row-major) . W[N, K]^T (bf16, row-major).  `beta` = 1 adds.
    void bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy = 0,
              float beta = 0.0f);

    /// `bf16`, or, when W is an 8-bit copy standing for a BF16 matrix that is not
    /// in VRAM (hc_fused.hpp), the same product with W's row slices dequantized to BF16 in the scratch.
    void bf16_w(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy = 0);

    /// W [N, K] as BF16 rows for a kernel of the caller's own, the values `bf16_w` multiplies by: W itself, or its 8-bit
    /// copy dequantized into the scratch (valid until the next product that uses the scratch).  nullptr: the
    /// scratch cannot hold the N rows at once.
    const uint16_t* bf16_rows(const uint16_t* W, int64_t N, int64_t K);

    /// Y = X . W^T with both in FP16 (bits). Prepared zero-beta shapes use one algorithm across widths;
    /// unprepared one-row calls and nonzero-beta calls may fall back to cuBLAS.
    void f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy = 0,
             float beta = 0.0f);

    /// W given as native GGUF blocks of `ggml_type`, dequantized to FP16 in the scratch, X in FP16.
    void native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                int64_t ldy = 0, float beta = 0.0f);

    /// Caller-owned buffers only: the scratch and workspace moved (the prompt path laid its buffers out again).
    void rebind(uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes);

    uint16_t* scratch() const { return scratch_; }
    int64_t scratch_elems() const { return scratch_elems_; }
    void* stream() const { return stream_; }

private:
    struct Plans;
    static std::shared_ptr<Plans> prepared_plans_[16];
    static bool prepare_lt_handle(std::string& err);
    /// Y = X . W^T on the shape's pinned cuBLASLt algorithm (`type`: a cudaDataType_t); false: the shape has none.
    bool pinned(int type, const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                float beta);
    void free_plans();

    void* handle_ = nullptr;
    std::shared_ptr<Plans> plans_;
    void* stream_ = nullptr;
    uint16_t* scratch_ = nullptr;
    int64_t scratch_elems_ = 0;
    void* workspace_ = nullptr;
    size_t ws_bytes_ = 0;
    bool external_ = false;
    void* hipblaslt_state_ = nullptr;
};



}  // namespace basalt::prefill
