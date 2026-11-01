// src/prefill/gemm_plans.cpp - the FP16 / BF16 products' per-shape cuBLASLt algorithms (Gemm::pinned). Host code
// only, so the standard containers it keeps are compiled by the host compiler alone.
#include "basalt/prefill/gemm.hpp"
#include "basalt/core/layout.hpp"
#include "basalt/kernels/ngram.hpp"

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>
#include "basalt/log.hpp"

namespace basalt::prefill {

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
cublasLtHandle_t g_prepared_lt[16] = {};
std::vector<Gemm::Shape> g_model_shapes[16];

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

std::vector<Gemm::Shape> Gemm::model_shapes(const core::WeightTable& wt, const core::ModelGeometry& g,
                                            int64_t scratch_elems) {
    std::vector<Shape> result;
    auto add = [&](bool bf16, int64_t n, int64_t k, int64_t ld, bool sliced) {
        if (n <= 0 || k <= 0 || ld < n) return;
        int64_t rows = n;
        if (sliced && scratch_elems > 0) rows = std::min(n, scratch_elems / k);
        if (rows <= 0) return;
        for (int64_t off = 0; off < n; off += rows) {
            const int64_t nr = std::min(rows, n - off);
            for (int a = 2; a <= 256; a *= 2) result.push_back({bf16, nr, k, ld, a});
        }
    };
    const char* native[] = {"attn_qkv.weight", "attn_gate.weight", "ssm_out.weight", "attn_q.weight",
                            "attn_k.weight", "attn_v.weight", "attn_output.weight", "ffn_gate_shexp.weight",
                            "ffn_up_shexp.weight", "ffn_down_shexp.weight"};
    const char* bf16[] = {"hc_attn_down.weight", "hc_attn_up.weight", "hc_attn_inject.weight",
                          "hc_ffn_down.weight", "hc_ffn_up.weight", "hc_ffn_inject.weight",
                          "indexer.k_proj.weight", "indexer.q_proj.weight", "ffn_gate_inp.weight"};
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const core::LayerView v(wt, l);
        for (const char* name : native) if (const auto* w = v.get(name))
            add(false, w->ne1, w->ne0, w->ne1, true);
        for (const char* name : bf16) if (const auto* w = v.get(name)) {
            const int64_t n = std::max<int64_t>(1, w->ne1);
            add(true, n, w->ne0, n, false);
            add(true, n, w->ne0, n, true);
        }
        for (const char* name : {"ssm_alpha.weight", "ssm_beta.weight"}) if (const auto* w = v.get(name)) {
            add(true, std::max<int64_t>(1, w->ne1), w->ne0, 2 * g.ssm_v_heads, false);
            add(true, std::max<int64_t>(1, w->ne1), w->ne0, 2 * g.ssm_v_heads, true);
        }
    }
    add(true, 1, g.n_embd, 1, false);
    add(false, 2 * g.n_ff, g.n_embd, 2 * g.n_ff, false);
    add(false, g.n_embd, g.n_ff, g.n_embd, false);
    add(true, g.hc_lr, g.hc_dim(), g.hc_lr, true);
    add(true, g.hc_dim(), g.hc_lr, g.hc_dim(), true);
    add(false, g.n_embd, g.n_embd, g.n_embd, true);
    add(false, g.n_head_kv * g.head_dim, g.n_embd, g.n_head_kv * g.head_dim, true);
    add(true, basalt::kernels::NG_HC_DIM, g.n_embd, basalt::kernels::NG_HC_DIM, false);
    add(false, basalt::kernels::NG_HC_DIM, g.n_embd, basalt::kernels::NG_HC_DIM, true);
    add(true, g.n_embd, g.n_embd, g.n_embd, false);
    auto key = [](const Shape& s) { return std::tie(s.bf16, s.n, s.k, s.ldy, s.align); };
    std::sort(result.begin(), result.end(), [&](const Shape& a, const Shape& b) { return key(a) < key(b); });
    result.erase(std::unique(result.begin(), result.end(), [&](const Shape& a, const Shape& b) {
        return key(a) == key(b);
    }), result.end());
    return result;
}

bool Gemm::prepare_lt_handle(std::string& err) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) return true;
    if (g_prepared_lt[dev]) return true;
    const cublasStatus_t s = cublasLtCreate(&g_prepared_lt[dev]);
    if (s != CUBLAS_STATUS_SUCCESS) {
        g_prepared_lt[dev] = nullptr;
        err = "prefill gemm: cublasLtCreate: cuBLAS status " + std::to_string((int) s);
        return false;
    }
    return true;
}

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
    // 7, 37 and 300 rows (those below T) give their rows the bits of the full call, and whose call of rows [m, R) does
    // too; that candidate becomes the plan's.
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
                ok = run(p, algo, ty, X + m * K, W, Y + m * ldy, R - m, N, K, ldy, ws, ws_bytes, stream) && same(m, R);
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

Gemm::Plans* Gemm::prepared_plans_[16] = {};

bool Gemm::prepare_model(const core::WeightTable& wt, const core::ModelGeometry& g, int64_t scratch_elems,
                         std::string& err) {
    if (!prepare_handle(err)) return false;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) return true;
    g_model_shapes[dev] = model_shapes(wt, g, scratch_elems);
    std::unique_ptr<Plans> plans(new Plans);
    plans->h = g_prepared_lt[dev];
    g_prepared_lt[dev] = nullptr;
    struct DeviceBuffer {
        void* p = nullptr;
        ~DeviceBuffer() { if (p) cudaFree(p); }
        bool allocate(size_t bytes) { return cudaMalloc(&p, bytes) == cudaSuccess; }
    } ws;
    if (!plans->h || !ws.allocate(kPlanWs)) {
        err = "prefill gemm: startup plan workspace";
        return false;
    }
    constexpr int64_t rows = 301;
    const auto& shapes = g_model_shapes[dev];
    for (size_t i = 0; i < shapes.size();) {
        const Shape& shape = shapes[i];
        size_t end = i + 1;
        while (end < shapes.size() && shapes[end].bf16 == shape.bf16 && shapes[end].n == shape.n &&
               shapes[end].k == shape.k && shapes[end].ldy == shape.ldy) ++end;
        const uint64_t nw = (uint64_t) shape.n * (uint64_t) shape.k;
        const uint64_t nx = (uint64_t) rows * (uint64_t) shape.k;
        const uint64_t ny = (uint64_t) rows * (uint64_t) shape.ldy;
        if (nw > (256ull << 20) / 2 || nx > (256ull << 20) / 2 || ny > (256ull << 20) / 4) {
            err = "prefill gemm: startup plan shape exceeds temporary buffer limit";
            return false;
        }
        DeviceBuffer w, x, y;
        if (!w.allocate((size_t) nw * 2 + 256) || !x.allocate((size_t) nx * 2 + 256) ||
            !y.allocate((size_t) ny * 4 + 256)) {
            err = "prefill gemm: startup plan buffers";
            return false;
        }
        std::vector<uint16_t> data((size_t) std::max(nw, nx + 128));
        const uint16_t base = shape.bf16 ? 0x3e00 : 0x3800;
        for (size_t j = 0; j < data.size(); ++j)
            data[j] = (uint16_t) (base | ((j * 73 + j / 17) & 0x7f) | ((j & 1) ? 0x8000 : 0));
        if (cudaMemcpy(w.p, data.data(), (size_t) nw * 2, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(x.p, data.data(), (size_t) nx * 2 + 256, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "prefill gemm: startup plan input upload";
            return false;
        }
        const auto ty = shape.bf16 ? CUDA_R_16BF : CUDA_R_16F;
        for (size_t j = i; j < end; ++j) {
            const int a = shapes[j].align;
            auto* xp = (const uint16_t*) ((const char*) x.p + (a == 256 ? 0 : a));
            auto& p = plans->plan(ty, shape.n, shape.k, shape.ldy, a);
            if (!p.candidates.empty())
                plans->settle(p, ty, xp, (const uint16_t*) w.p, (float*) y.p, rows, shape.n, shape.k, shape.ldy,
                              a, ws.p, kPlanWs, nullptr);
        }
        i = end;
    }
    delete prepared_plans_[dev];
    prepared_plans_[dev] = plans.release();
    return true;
}

bool Gemm::pinned(int type, const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K,
                  int64_t ldy, float beta) {
    if (plans_ == nullptr) {
        int dev = 0;
        if (cudaGetDevice(&dev) == cudaSuccess && dev >= 0 && dev < 16 && prepared_plans_[dev]) {
            plans_ = prepared_plans_[dev];
            prepared_plans_[dev] = nullptr;
        } else {
            plans_ = new Plans;
        }
        if (!plans_->h && dev >= 0 && dev < 16 && g_prepared_lt[dev]) {
            plans_->h = g_prepared_lt[dev];
            g_prepared_lt[dev] = nullptr;
        } else if (!plans_->h && cublasLtCreate(&plans_->h) != CUBLAS_STATUS_SUCCESS) {
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

void Gemm::free_plans() {
    delete plans_;
    plans_ = nullptr;
}

}  // namespace basalt::prefill
