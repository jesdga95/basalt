// src/prefill/gemm.cu - see include/basalt/prefill/gemm.hpp.
#include "basalt/prefill/gemm.hpp"
#include "basalt/kernels/dequant_bf16.hpp"
#include "basalt/kernels/hc_fused.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>


#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include "basalt/log.hpp"


namespace basalt::prefill {
namespace {

void ck(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        basalt::log::error("basalt", "prefill gemm: %s: cuBLAS status %d", what, (int) s);
        std::exit(1);
    }
}

// A setup call whose failure the engine survives (the handle keeps its defaults), but says so.
void note(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) basalt::log::warn("basalt", "prefill gemm: %s: cuBLAS status %d (continuing)", what, (int) s);
}


// One handle made ahead of time per device, taken by the first Gemm that starts there.
cublasHandle_t g_prepared[16] = {};

}  // namespace

bool Gemm::prepare_handle(std::string& err) {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 16 || g_prepared[dev] != nullptr) return true;
    if (const cublasStatus_t s = cublasCreate(&g_prepared[dev]); s != CUBLAS_STATUS_SUCCESS) {
        g_prepared[dev] = nullptr;
        err = "prefill gemm: cublasCreate: cuBLAS status " + std::to_string((int) s);
        return false;
    }
    return true;
}

Gemm::~Gemm() {
    delete lt_;
    if (handle_) cublasDestroy((cublasHandle_t) handle_);
    if (!external_) {
        if (scratch_) cudaFree(scratch_);
        if (workspace_) cudaFree(workspace_);
    }
}

bool Gemm::init_external(void* stream, uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes,
                         std::string& err) {
    cublasHandle_t h = nullptr;
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev >= 0 && dev < 16 && g_prepared[dev] != nullptr) {
        h = g_prepared[dev];
        g_prepared[dev] = nullptr;
    } else if (const cublasStatus_t s = cublasCreate(&h); s != CUBLAS_STATUS_SUCCESS) {
        err = "prefill gemm: cublasCreate: cuBLAS status " + std::to_string((int) s);
        return false;
    }
    handle_ = h;
    stream_ = stream;
    external_ = true;
    note(cublasSetStream(h, (cudaStream_t) stream), "cublasSetStream");
    workspace_ = workspace;
    ws_bytes_ = ws_bytes;
    note(cublasSetWorkspace(h, workspace_, ws_bytes), "cublasSetWorkspace");
    note(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    return true;
}

void Gemm::rebind(uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes) {
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    workspace_ = workspace;
    ws_bytes_ = ws_bytes;
    cublasSetWorkspace((cublasHandle_t) handle_, workspace_, ws_bytes);
}

bool Gemm::init(void* stream, int64_t scratch_elems, std::string& err) {
    // every failure names the call and the real status, so "no VRAM" can be told from a broken install
    cublasHandle_t h = nullptr;
    if (const cublasStatus_t s = cublasCreate(&h); s != CUBLAS_STATUS_SUCCESS) {
        err = "prefill gemm: cublasCreate: cuBLAS status " + std::to_string((int) s);
        return false;
    }
    handle_ = h;
    stream_ = stream;
    note(cublasSetStream(h, (cudaStream_t) stream), "cublasSetStream");
    // A fixed workspace so the handle never allocates on the way (and graphs could capture it later).
    const size_t ws = 32u << 20;
    if (const cudaError_t e = cudaMalloc(&workspace_, ws); e != cudaSuccess) {
        err = std::string("prefill gemm: workspace of 32 MiB: ") + cudaGetErrorString(e);
        return false;
    }
    ws_bytes_ = ws;
    note(cublasSetWorkspace(h, workspace_, ws), "cublasSetWorkspace");
    note(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
    if (scratch_elems > 0) {
        if (const cudaError_t e = cudaMalloc((void**) &scratch_, (size_t) scratch_elems * 2); e != cudaSuccess) {
            err = "prefill gemm: dequant scratch of " + std::to_string(scratch_elems * 2 >> 20) + " MiB: " +
                  cudaGetErrorString(e);
            return false;
        }
    }
    scratch_elems_ = scratch_elems;
    return true;
}

void Gemm::bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    const float alpha = 1.0f;
    // Column-major view: Y^T[N, T] = W[N, K] (stored K x N col-major, transposed) . X^T[K, T].
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    CUDA_R_16BF, (int) K, X, CUDA_R_16BF, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx");
}

void Gemm::bf16_w(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy) {
    basalt::kernels::Hc8View v;
    if (basalt::kernels::hc8_view(W, v) && v.owned) {
        if (v.rows != N || v.cols != K) {
            basalt::log::error("basalt", "prefill gemm: an 8-bit copy is %d x %d, the product wants %lld x %lld", v.rows,
                         v.cols, (long long) N, (long long) K);
            std::exit(1);
        }
        if (ldy <= 0) ldy = N;
        const int64_t rows = std::min<int64_t>(N, scratch_elems_ / K);
        if (rows <= 0 || scratch_ == nullptr) {
            basalt::log::error("basalt", "prefill gemm: no scratch for an 8-bit matrix (K=%lld)", (long long) K);
            std::exit(1);
        }
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = std::min<int64_t>(rows, N - r0);
            basalt::kernels::hc8_dequant_bf16(v, r0, n, scratch_, stream_);
            bf16(X, scratch_, Y + r0, T, n, K, ldy);
        }
        return;
    }
    bf16(X, W, Y, T, N, K, ldy);
}

const uint16_t* Gemm::bf16_rows(const uint16_t* W, int64_t N, int64_t K) {
    basalt::kernels::Hc8View v;
    if (!basalt::kernels::hc8_view(W, v) || !v.owned) return W;
    if (v.rows != N || v.cols != K || scratch_ == nullptr || N * K > scratch_elems_) return nullptr;
    basalt::kernels::hc8_dequant_bf16(v, 0, N, scratch_, stream_);
    return scratch_;
}

void Gemm::f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
               float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    const float alpha = 1.0f;
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    CUDA_R_16F, (int) K, X, CUDA_R_16F, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx f16");
}

void Gemm::native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                  int64_t ldy, float beta) {
    if (N * K > scratch_elems_) {
        // Too large for the scratch at once: in row slices.
        const int64_t rows = scratch_elems_ / K;
        if (rows <= 0) { basalt::log::error("basalt", "prefill gemm: scratch too small for K=%lld", (long long) K); std::exit(1); }
        if (ldy <= 0) ldy = N;
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = (N - r0 < rows) ? N - r0 : rows;
            basalt::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, scratch_, stream_);
            f16(X, scratch_, Y + r0, T, n, K, ldy, beta);
        }
        return;
    }
    basalt::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, scratch_, stream_);
    f16(X, scratch_, Y, T, N, K, ldy, beta);
}

namespace {
size_t align256(size_t v) { return (v + 255) & ~(size_t) 255; }
}  // namespace

size_t Gemm::act8_bytes(int64_t rows, int64_t cols) {
    return align256((size_t) rows * (size_t) cols) +
           align256(std::max(fp8::scale_bytes(fp8::Scale::Row, rows, cols), fp8::scale_bytes(fp8::Scale::Block32, rows, cols)));
}

bool Gemm::fp8_ready() {
    if (fp8_state_ != 0) return fp8_state_ > 0;
    fp8_state_ = -1;
    std::string err;
    lt_ = new fp8::Lt;
    if (!lt_->init(err)) {
        basalt::log::warn("basalt", "prefill gemm: FP8 products off: %s", err.c_str());
        return false;
    }
    // a prompt-sized shape; the workspace stands in for the scale vectors (the heuristic does not read them)
    for (fp8::Scale s : {fp8::Scale::Row, fp8::Scale::Block32}) {
        if (lt_->algos(s, 4096, 2560, 2560, 2560, ws_bytes_, workspace_, workspace_) > 0) {
            fp8_scale_ = s;
            fp8_state_ = 1;
            basalt::log::info("basalt", "prefill gemm: dense products in FP8 (E4M3, %s scales)", fp8::scale_name(s));
            return true;
        }
    }
    basalt::log::warn("basalt", "prefill gemm: FP8 products off: cuBLASLt offers no E4M3 algorithm with per-row or "
                                "per-block scales");
    return false;
}

Gemm::Rows8 Gemm::act8(const uint16_t* X, int64_t rows, int64_t cols, void* buf) {
    Rows8 r;
    uint8_t* q = static_cast<uint8_t*>(buf);
    void* sc = q + align256((size_t) rows * (size_t) cols);
    fp8::to_e4m3(X, rows, cols, cols, q, sc, 0, rows, fp8_scale_, nullptr, stream_);
    r.q = q;
    r.scales = sc;
    r.rows = rows;
    r.cols = cols;
    return r;
}

bool Gemm::native8(const Rows8& X, int ggml_type, const void* W_blocks, float* Y, int64_t N, int64_t ldy) {
    const int64_t T = X.rows, K = X.cols;
    if (fp8_state_ <= 0 || T <= 0 || N <= 0 || K % 32 != 0) return false;
    if (ldy <= 0) ldy = N;
    // the scratch: the E4M3 matrix, its scales, then FP16 row slices on their way to E4M3
    const size_t total = (size_t) scratch_elems_ * 2, wq = align256((size_t) N * (size_t) K),
                 ws = align256(fp8::scale_bytes(fp8_scale_, N, K));
    if (scratch_ == nullptr || wq + ws + (size_t) K * 2 * 16 > total) return false;
    uint8_t* q = reinterpret_cast<uint8_t*>(scratch_);
    void* sc = q + wq;
    uint16_t* stage = reinterpret_cast<uint16_t*>(q + wq + ws);
    if (lt_->algos(fp8_scale_, T, N, K, ldy, ws_bytes_, sc, X.scales) == 0) return false;
    const int64_t rows = std::min<int64_t>(N, (int64_t) ((total - wq - ws) / ((size_t) K * 2)));
    for (int64_t r0 = 0; r0 < N; r0 += rows) {
        const int64_t n = std::min<int64_t>(rows, N - r0);
        basalt::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, stage, stream_);
        fp8::to_e4m3(stage, n, K, K, q + (size_t) r0 * (size_t) K, sc, r0, N, fp8_scale_, nullptr, stream_);
    }
    if (!lt_->run(fp8_scale_, q, sc, X.q, X.scales, Y, T, N, K, ldy, workspace_, ws_bytes_, stream_)) {
        basalt::log::error("basalt", "prefill gemm: cublasLtMatmul (E4M3, %lld x %lld x %lld) failed", (long long) T,
                           (long long) N, (long long) K);
        std::exit(1);
    }
    return true;
}

}  // namespace basalt::prefill
