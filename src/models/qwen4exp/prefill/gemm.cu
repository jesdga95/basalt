// src/models/qwen4exp/prefill/gemm.cu - see include/basalt/models/qwen4exp/prefill/gemm.hpp.
#include "basalt/models/qwen4exp/prefill/gemm.hpp"
#include "basalt/kernels/dequant_bf16.hpp"
#include "basalt/models/qwen4exp/kernels/hc_fused.hpp"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>


#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include "basalt/core/diag.hpp"
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

// Under --diag-main-card-bits, the main card's algorithm for a product, as its cublasLt heuristic picks it within the workspace: the one
// cublasGemmEx runs there (checked bit for bit against it on an RTX 5090, and on an RTX 5060 Ti running the 5090's
// algorithm).  Shared by every stage's Gemm, made once per product shape.
struct MainAlgo {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulAlgo_t algo{};
    bool ok = false;
};
std::mutex g_main_mu;
cublasLtHandle_t g_main_lt = nullptr;   // made on the main card (CUDA 0): its heuristic
std::map<std::tuple<int, int64_t, int64_t, int64_t, int64_t, size_t>, MainAlgo> g_main_algos;

const MainAlgo& main_algo(int type, int64_t T, int64_t N, int64_t K, int64_t ldy, size_t ws) {
    std::lock_guard<std::mutex> lk(g_main_mu);
    MainAlgo& m = g_main_algos[std::make_tuple(type, T, N, K, ldy, ws)];
    if (m.op != nullptr) return m;
    int cur = 0;
    cudaGetDevice(&cur);
    cudaSetDevice(0);
    if (g_main_lt == nullptr && cublasLtCreate(&g_main_lt) != CUBLAS_STATUS_SUCCESS) g_main_lt = nullptr;
    const cudaDataType_t dt = (cudaDataType_t) type;
    const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    bool ok = cublasLtMatmulDescCreate(&m.op, CUBLAS_COMPUTE_32F, CUDA_R_32F) == CUBLAS_STATUS_SUCCESS &&
              cublasLtMatmulDescSetAttribute(m.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta) == CUBLAS_STATUS_SUCCESS &&
              cublasLtMatmulDescSetAttribute(m.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb) == CUBLAS_STATUS_SUCCESS &&
              cublasLtMatrixLayoutCreate(&m.a, dt, (uint64_t) K, (uint64_t) N, K) == CUBLAS_STATUS_SUCCESS &&
              cublasLtMatrixLayoutCreate(&m.b, dt, (uint64_t) K, (uint64_t) T, K) == CUBLAS_STATUS_SUCCESS &&
              cublasLtMatrixLayoutCreate(&m.c, CUDA_R_32F, (uint64_t) N, (uint64_t) T, ldy) == CUBLAS_STATUS_SUCCESS;
    cublasLtMatmulPreference_t pref = nullptr;
    ok = ok && g_main_lt != nullptr && cublasLtMatmulPreferenceCreate(&pref) == CUBLAS_STATUS_SUCCESS &&
         cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws) ==
             CUBLAS_STATUS_SUCCESS;
    cublasLtMatmulHeuristicResult_t r{};
    int got = 0;
    ok = ok && cublasLtMatmulAlgoGetHeuristic(g_main_lt, m.op, m.a, m.b, m.c, m.c, pref, 1, &r, &got) == CUBLAS_STATUS_SUCCESS &&
         got > 0 && r.state == CUBLAS_STATUS_SUCCESS;
    if (pref != nullptr) cublasLtMatmulPreferenceDestroy(pref);
    if (ok) m.algo = r.algo;
    m.ok = ok;
    if (!ok) basalt::log::warn("basalt", "prefill gemm: no main-card algorithm for a %lld x %lld x %lld product: this card's "
                               "own (its bits may differ from the main card's)", (long long) T, (long long) N, (long long) K);
    cudaSetDevice(cur);
    (void) cudaGetLastError();
    return m;
}

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
    if (handle_) cublasDestroy((cublasHandle_t) handle_);
    if (lt_) cublasLtDestroy((cublasLtHandle_t) lt_);
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
    if (dev != 0 && basalt::core::diagnostics().main_card_bits) note(cublasLtCreate((cublasLtHandle_t*) &lt_), "cublasLtCreate");
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
    note(cublasSetWorkspace(h, workspace_, ws), "cublasSetWorkspace");
    ws_bytes_ = ws;
    if (int dev = 0; cudaGetDevice(&dev) == cudaSuccess && dev != 0 && basalt::core::diagnostics().main_card_bits) note(cublasLtCreate((cublasLtHandle_t*) &lt_), "cublasLtCreate");
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
    product(CUDA_R_16BF, X, W, Y, T, N, K, ldy, beta, "cublasGemmEx");
}

void Gemm::product(int type, const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                   float beta, const char* what) {
    const float alpha = 1.0f;
    if (lt_ != nullptr) {   // --diag-main-card-bits, off the main card: the main card's algorithm
        const MainAlgo& m = main_algo(type, T, N, K, ldy, ws_bytes_);
        if (m.ok) {
            ck(cublasLtMatmul((cublasLtHandle_t) lt_, m.op, &alpha, W, m.a, X, m.b, &beta, Y, m.c, Y, m.c, &m.algo, workspace_,
                              ws_bytes_, (cudaStream_t) stream_),
               "cublasLtMatmul");
            return;
        }
    }
    // Column-major view: Y^T[N, T] = W[N, K] (stored K x N col-major, transposed) . X^T[K, T].
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    (cudaDataType_t) type, (int) K, X, (cudaDataType_t) type, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       what);
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
    product(CUDA_R_16F, X, W, Y, T, N, K, ldy, beta, "cublasGemmEx f16");
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

}  // namespace basalt::prefill
