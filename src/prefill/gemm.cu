// src/prefill/gemm.cu - see include/basalt/prefill/gemm.hpp.
#include "basalt/prefill/gemm.hpp"
#include "basalt/kernels/dequant_bf16.hpp"
#include "basalt/kernels/hc_fused.hpp"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>


#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <tuple>
#include <vector>
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

// ---- FP16 / BF16 products on one cuBLASLt algorithm per weight shape
//
// cublasGemmEx picks a kernel, a tile and a split of K per call, and the call's row count T is part of the shape, so a
// token's products would depend on the chunk that carries it.  Each shape (input type, N, K, ldy, the operands'
// alignment) instead takes one algorithm, chosen once and accepted only if it runs every row count of a spread: K in
// one piece, or in a fixed number of parts summed by a separate reduction.  The candidates come from the heuristic's
// offers at a range of chunk sizes, never from a timing: first the no-split 64x64-tile kernel (algorithm 21, 32x6
// stages) when it is offered, as its tile keeps short chunks busy and long ones efficient, then the others by the sum
// of their ranks across the range.  The configuration does not say everything a kernel does with the row count, so
// the shape's first call of two or more rows checks a candidate on its own rows: row 0 again in narrower calls, and a
// later row again at place 0, must give the same bits, else the next candidate runs.  Each output element is
// then the same arithmetic in the same order at every T.  A shape without such an algorithm, or whose algorithm a
// call refuses, runs cublasGemmEx from then on.
namespace {
constexpr int64_t kPlanRows[] = {64, 256, 512, 1024, 2048, 4096, 16384, 32768};   // the heuristic's offers
constexpr int64_t kPlanCheckRows[] = {1, 2, 3, 5, 8, 17, 37, 64, 300, 1000, 4096, 16384, 32768};
constexpr int kPlanOffers = 32;
constexpr size_t kPlanWs = 32u << 20;

int alignment(const void* p) {
    const uintptr_t v = (uintptr_t) p;
    int a = 256;
    while (a > 1 && v % (uintptr_t) a != 0) a >>= 1;
    return a;
}

// An algorithm's configuration: what identifies it across the heuristic's answers.
struct Config {
    int32_t id = 0, splits = 0;
    uint32_t tile = 0, stages = 0, reduction = 0, swizzle = 0, custom = 0;
    explicit Config(const cublasLtMatmulAlgo_t& a) {
        size_t got = 0;
        cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_ID, &id, sizeof id, &got);
        cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof tile, &got);
        cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof stages, &got);
        cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splits, sizeof splits, &got);
        cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reduction, sizeof reduction, &got);
        cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof swizzle, &got);
        cublasLtMatmulAlgoConfigGetAttribute(&a, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &custom, sizeof custom, &got);
    }
    auto tie() const { return std::tie(id, tile, stages, splits, reduction, swizzle, custom); }
    bool operator<(const Config& o) const { return tie() < o.tie(); }
    // K in one piece, or in a fixed number of parts reduced out of place (a negative count is a split chosen per call)
    bool fixed_order() const {
        if (splits == 0 || splits == 1) return true;
        return splits > 1 && (reduction == CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE ||
                              reduction == CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE);
    }
    bool preferred() const {
        return id == 21 && tile == CUBLASLT_MATMUL_TILE_64x64 && stages == CUBLASLT_MATMUL_STAGES_32x6 && splits <= 1;
    }
};
}  // namespace

struct Gemm::Plans {
    cublasLtHandle_t h = nullptr;
    struct Plan {
        cublasLtMatmulDesc_t op = nullptr;
        cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;   // b and c hold `rows` rows
        int64_t rows = -1;
        cublasLtMatmulAlgo_t algo{};
        size_t ws = 0;
        bool ok = false;                               // `algo` is checked and runs every call
        std::vector<cublasLtMatmulAlgo_t> candidates;  // in order of preference, before the check
        size_t next = 0;
    };
    using Key = std::tuple<int, int64_t, int64_t, int64_t, int>;
    std::map<Key, Plan> plans;

    ~Plans() {
        for (auto& kv : plans) {
            Plan& p = kv.second;
            if (p.op) cublasLtMatmulDescDestroy(p.op);
            for (cublasLtMatrixLayout_t l : {p.a, p.b, p.c})
                if (l) cublasLtMatrixLayoutDestroy(l);
        }
        if (h) cublasLtDestroy(h);
    }

    // The B (X^T, K x T) and C (Y^T, N x T) layouts for T rows.
    bool rows(Plan& p, cudaDataType_t ty, int64_t T, int64_t N, int64_t K, int64_t ldy) {
        if (p.rows == T) return true;
        if (p.b) cublasLtMatrixLayoutDestroy(p.b);
        if (p.c) cublasLtMatrixLayoutDestroy(p.c);
        p.b = p.c = nullptr;
        p.rows = -1;
        if (cublasLtMatrixLayoutCreate(&p.b, ty, (uint64_t) K, (uint64_t) T, K) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatrixLayoutCreate(&p.c, CUDA_R_32F, (uint64_t) N, (uint64_t) T, ldy) != CUBLAS_STATUS_SUCCESS)
            return false;
        p.rows = T;
        return true;
    }

    // Whether `algo` runs every row count of the spread within kPlanWs; `ws` the most it needs.
    bool every_width(Plan& p, cublasLtMatmulAlgo_t& algo, cudaDataType_t ty, int64_t N, int64_t K, int64_t ldy,
                     size_t& ws) {
        ws = 0;
        for (const int64_t T : kPlanCheckRows) {
            cublasLtMatmulHeuristicResult_t chk{};
            if (!rows(p, ty, T, N, K, ldy) ||
                cublasLtMatmulAlgoCheck(h, p.op, p.a, p.b, p.c, p.c, &algo, &chk) != CUBLAS_STATUS_SUCCESS ||
                chk.workspaceSize > kPlanWs)
                return false;
            ws = std::max(ws, chk.workspaceSize);
        }
        return true;
    }

    Plan& plan(cudaDataType_t ty, int64_t N, int64_t K, int64_t ldy, int align) {
        const Key key{(int) ty, N, K, ldy, align};
        auto it = plans.find(key);
        if (it != plans.end()) return it->second;
        Plan& p = plans[key];
        // column-major view: Y^T [N, T] = op(W) [N, K] . X^T [K, T], with W stored as K x N (rows of K values)
        const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
        const bool ok =
            cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F) == CUBLAS_STATUS_SUCCESS &&
            cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta) == CUBLAS_STATUS_SUCCESS &&
            cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb) == CUBLAS_STATUS_SUCCESS &&
            cublasLtMatrixLayoutCreate(&p.a, ty, (uint64_t) K, (uint64_t) N, K) == CUBLAS_STATUS_SUCCESS;
        cublasLtMatmulPreference_t pref = nullptr;
        if (!ok || cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) return p;
        const uint64_t wsb = kPlanWs;
        const uint32_t al = (uint32_t) align;
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsb, sizeof wsb);
        for (cublasLtMatmulPreferenceAttributes_t a :
             {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
              CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES})
            cublasLtMatmulPreferenceSetAttribute(pref, a, &al, sizeof al);
        // every fixed-order algorithm offered at any of the sizes, with the sum of its ranks from 256 rows up (an
        // absent offer counts as the last rank)
        struct Offer {
            cublasLtMatmulAlgo_t algo;
            int score = 0;
        };
        std::map<Config, Offer> offers;
        std::vector<std::map<Config, int>> ranks;
        for (const int64_t T : kPlanRows) {
            cublasLtMatmulHeuristicResult_t res[kPlanOffers] = {};
            int n = 0;
            if (!rows(p, ty, T, N, K, ldy) ||
                cublasLtMatmulAlgoGetHeuristic(h, p.op, p.a, p.b, p.c, p.c, pref, kPlanOffers, res, &n) !=
                    CUBLAS_STATUS_SUCCESS)
                n = 0;
            ranks.emplace_back();
            for (int i = 0; i < n; ++i) {
                if (res[i].state != CUBLAS_STATUS_SUCCESS) continue;
                const Config c(res[i].algo);
                if (!c.fixed_order()) continue;
                offers.emplace(c, Offer{res[i].algo, 0});
                ranks.back().emplace(c, i);
            }
        }
        cublasLtMatmulPreferenceDestroy(pref);
        std::vector<std::pair<std::tuple<int, int, Config>, cublasLtMatmulAlgo_t>> order;
        for (auto& [c, o] : offers) {
            int score = 0;
            for (size_t t = 1; t < ranks.size(); ++t) {
                const auto r = ranks[t].find(c);
                score += r != ranks[t].end() ? r->second : kPlanOffers;
            }
            order.push_back({{c.preferred() ? 0 : 1, score, c}, o.algo});
        }
        std::sort(order.begin(), order.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
        for (auto& [rank, algo] : order) p.candidates.push_back(algo);
        return p;
    }

    bool run(Plan& p, const cublasLtMatmulAlgo_t& algo, cudaDataType_t ty, const uint16_t* X, const uint16_t* W,
             float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy, void* ws, size_t ws_bytes, void* stream) {
        const float alpha = 1.0f, beta = 0.0f;
        return rows(p, ty, T, N, K, ldy) &&
               cublasLtMatmul(h, p.op, &alpha, W, p.a, X, p.b, &beta, Y, p.c, Y, p.c, &algo, ws, ws_bytes,
                              (cudaStream_t) stream) == CUBLAS_STATUS_SUCCESS;
    }

    // Y = X . W^T (T >= 2 rows) with the first candidate that runs every row count of the spread, whose calls of 1, 3,
    // 7, 37 and 300 rows (those below T) give their rows the bits of the full call, and whose call of the rows from m
    // on does too; that candidate becomes the plan's.
    bool settle(Plan& p, cudaDataType_t ty, const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N,
                int64_t K, int64_t ldy, int align, void* ws, size_t ws_bytes, void* stream) {
        int64_t m = 1;   // a later row whose X and Y rows keep the call's alignment
        while (m < T && (alignment(X + m * K) < align || alignment(Y + m * ldy) < align)) ++m;
        const int64_t R = std::min<int64_t>(T, 301);   // the rows compared
        std::vector<float> full((size_t) R * N), part((size_t) R * N);
        // rows [r0, r1) of Y to `host` (N values a row)
        auto rows_back = [&](int64_t r0, int64_t r1, float* host) {
            return cudaMemcpy2DAsync(host, (size_t) N * 4, Y + r0 * ldy, (size_t) ldy * 4, (size_t) N * 4,
                                     (size_t) (r1 - r0), cudaMemcpyDeviceToHost, (cudaStream_t) stream) == cudaSuccess &&
                   cudaStreamSynchronize((cudaStream_t) stream) == cudaSuccess;
        };
        auto same = [&](int64_t r0, int64_t r1) {
            return rows_back(r0, r1, part.data()) &&
                   std::memcmp(part.data(), full.data() + r0 * N, (size_t) (r1 - r0) * N * 4) == 0;
        };
        for (; p.next < p.candidates.size(); ++p.next) {
            cublasLtMatmulAlgo_t& algo = p.candidates[p.next];
            size_t need = 0;
            if (!every_width(p, algo, ty, N, K, ldy, need) || need > ws_bytes) continue;
            if (!run(p, algo, ty, X, W, Y, T, N, K, ldy, ws, ws_bytes, stream) || !rows_back(0, R, full.data())) continue;
            bool ok = true;
            for (const int64_t w : {(int64_t) 1, (int64_t) 3, (int64_t) 7, (int64_t) 37, (int64_t) 300})
                if (ok && w < T) ok = run(p, algo, ty, X, W, Y, w, N, K, ldy, ws, ws_bytes, stream) && same(0, w);
            if (ok && m < R)
                ok = run(p, algo, ty, X + m * K, W, Y + m * ldy, T - m, N, K, ldy, ws, ws_bytes, stream) && same(m, R);
            if (!ok) continue;
            p.algo = algo;
            p.ws = need;
            p.ok = true;
            p.candidates.clear();
            return true;   // Y holds the full product: the reruns wrote the same bits
        }
        p.candidates.clear();
        basalt::log::warn("basalt", "prefill gemm: no fixed cuBLASLt algorithm for %s %lld x %lld: cublasGemmEx, "
                                    "whose rounding follows the chunk size",
                          ty == CUDA_R_16BF ? "BF16" : "FP16", (long long) N, (long long) K);
        return false;
    }
};

bool Gemm::pinned(int type, const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K,
                  int64_t ldy, float beta) {
    if (plans_ == nullptr) {
        plans_ = new Plans;
        if (cublasLtCreate(&plans_->h) != CUBLAS_STATUS_SUCCESS) {
            plans_->h = nullptr;
            basalt::log::warn("basalt", "prefill gemm: cublasLtCreate failed: cublasGemmEx, whose rounding follows the "
                                        "chunk size");
        }
    }
    if (plans_->h == nullptr) return false;
    const cudaDataType_t ty = (cudaDataType_t) type;
    const int align = std::min(alignment(X), std::min(alignment(W), alignment(Y)));
    Plans::Plan& p = plans_->plan(ty, N, K, ldy, align);
    if (!p.ok) {
        // the candidates are checked on a call of two or more rows that overwrites Y; until then cublasGemmEx
        if (p.candidates.empty() || T < 2 || beta != 0.0f) return false;
        return plans_->settle(p, ty, X, W, Y, T, N, K, ldy, align, workspace_, ws_bytes_, stream_);
    }
    if (p.ws > ws_bytes_ || !plans_->rows(p, ty, T, N, K, ldy)) return false;
    const float alpha = 1.0f;
    if (cublasLtMatmul(plans_->h, p.op, &alpha, W, p.a, X, p.b, &beta, Y, p.c, Y, p.c, &p.algo, workspace_, ws_bytes_,
                       (cudaStream_t) stream_) != CUBLAS_STATUS_SUCCESS) {
        p.ok = false;
        basalt::log::warn("basalt", "prefill gemm: cublasLtMatmul refused %s %lld x %lld x %lld: cublasGemmEx for this "
                                    "shape from here",
                          ty == CUDA_R_16BF ? "BF16" : "FP16", (long long) T, (long long) N, (long long) K);
        return false;
    }
    return true;
}

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
    delete plans_;
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
    if (pinned(CUDA_R_16BF, X, W, Y, T, N, K, ldy, beta)) return;
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
    if (pinned(CUDA_R_16F, X, W, Y, T, N, K, ldy, beta)) return;
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

}  // namespace basalt::prefill
