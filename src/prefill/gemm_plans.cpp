// src/prefill/gemm_plans.cpp - the FP16 / BF16 products' per-shape cuBLASLt algorithms (Gemm::pinned). Host code
// only, so the standard containers it keeps are compiled by the host compiler alone.
#include "basalt/prefill/gemm.hpp"
#include "basalt/core/layout.hpp"
#include "basalt/kernels/ngram.hpp"
#include "basalt/kernels/hc_fused.hpp"

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <utility>
#include <vector>
#include "basalt/log.hpp"

namespace basalt::prefill {

// A shape keeps one fixed-order algorithm across row counts. Startup checks representative rows and times the
// accepted candidates; a shape without a prepared plan is checked on its first eligible call.
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
    auto add = [&](bool bf16, int64_t n, int64_t k, int64_t ld, bool sliced, int base_align = 256) {
        if (n <= 0 || k <= 0 || ld < n) return;
        int64_t rows = n;
        if (sliced && scratch_elems > 0) rows = std::min(n, scratch_elems / k);
        if (rows <= 0) return;
        for (int64_t off = 0; off < n; off += rows) {
            const int64_t nr = std::min(rows, n - off);
            const uintptr_t byte_offset = (uintptr_t) off * sizeof(float);
            int out_align = 256;
            while (out_align > 1 && byte_offset % (uintptr_t) out_align != 0) out_align >>= 1;
            result.push_back({bf16, nr, k, ld, std::min(base_align, out_align)});
        }
    };
    const char* native[] = {"attn_qkv.weight", "attn_gate.weight", "ssm_out.weight", "attn_q.weight",
                            "attn_k.weight", "attn_v.weight", "attn_output.weight", "ffn_gate_shexp.weight",
                            "ffn_up_shexp.weight", "ffn_down_shexp.weight"};
    const char* bf16[] = {"hc_attn_down.weight", "hc_attn_up.weight", "hc_attn_inject.weight",
                          "hc_ffn_down.weight", "hc_ffn_up.weight", "hc_ffn_inject.weight",
                          "indexer.k_proj.weight", "indexer.q_proj.weight", "ffn_gate_inp.weight",
                          "ffn_gate_inp_shexp.weight"};
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const core::LayerView v(wt, l);
        for (const char* name : native) if (const auto* w = v.get(name); w && w->native_data)
            add(false, w->ne1, w->ne0, w->ne1, true);
        for (const char* name : bf16) if (const auto* w = v.get(name); w && w->data) {
            const int64_t n = std::max<int64_t>(1, w->ne1);
            basalt::kernels::Hc8View view;
            if (w->data && basalt::kernels::hc8_view(w->data, view) && view.owned)
                add(true, n, w->ne0, n, true);
            else
                add(true, n, w->ne0, n, false, alignment(w->data));
        }
        for (const char* name : {"ssm_alpha.weight", "ssm_beta.weight"}) if (const auto* w = v.get(name); w && w->data) {
            const int out_align = std::strcmp(name, "ssm_beta.weight") == 0
                ? alignment((const void*) (uintptr_t) (g.ssm_v_heads * 4)) : 256;
            basalt::kernels::Hc8View view;
            if (w->data && basalt::kernels::hc8_view(w->data, view) && view.owned)
                add(true, std::max<int64_t>(1, w->ne1), w->ne0, 2 * g.ssm_v_heads, true, out_align);
            else
                add(true, std::max<int64_t>(1, w->ne1), w->ne0, 2 * g.ssm_v_heads, false,
                    std::min(alignment(w->data), out_align));
        }
    }
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
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) {
        err = "prefill gemm: invalid current device";
        return false;
    }
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
    bool prepared = false;
    std::mutex mutex;
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
                int64_t K, int64_t ldy, int align, void* ws, size_t ws_bytes, void* stream, bool tune = false,
                int64_t timing_rows = 4096) {
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
        struct Timed { cublasLtMatmulAlgo_t algo; size_t ws; float ms[3]; };
        std::vector<Timed> timed;
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
            if (!tune) {
                p.algo = algo;
                p.ws = need;
                p.ok = true;
                p.candidates.clear();
                return true;
            }
            Timed entry{algo, need, {}};
            cudaEvent_t start = nullptr, stop = nullptr;
            bool measured = cudaEventCreate(&start) == cudaSuccess && cudaEventCreate(&stop) == cudaSuccess;
            const int64_t widths[] = {64, std::min<int64_t>(512, timing_rows), timing_rows};
            for (int q = 0; measured && q < 3; ++q) {
                const auto s = (cudaStream_t) stream;
                float samples[3] = {};
                measured = run(p, algo, ty, X, W, Y, widths[q], N, K, ldy, ws, ws_bytes, stream);
                for (int rep = 0; measured && rep < 3; ++rep)
                    measured = cudaEventRecord(start, s) == cudaSuccess &&
                               run(p, algo, ty, X, W, Y, widths[q], N, K, ldy, ws, ws_bytes, stream) &&
                               cudaEventRecord(stop, s) == cudaSuccess && cudaEventSynchronize(stop) == cudaSuccess &&
                               cudaEventElapsedTime(&samples[rep], start, stop) == cudaSuccess && samples[rep] > 0;
                if (measured) {
                    std::sort(samples, samples + 3);
                    entry.ms[q] = samples[1];
                }
            }
            if (start) cudaEventDestroy(start);
            if (stop) cudaEventDestroy(stop);
            if (measured) timed.push_back(entry);
            if (timed.size() == 8) break;
        }
        if (!timed.empty()) {
            float best[3] = {timed[0].ms[0], timed[0].ms[1], timed[0].ms[2]};
            for (const auto& e : timed) for (int q = 0; q < 3; ++q) best[q] = std::min(best[q], e.ms[q]);
            size_t selected = 0;
            auto score = [&](const Timed& e) {
                return e.ms[0] / best[0] + e.ms[1] / best[1] + e.ms[2] / best[2];
            };
            for (size_t q = 1; q < timed.size(); ++q)
                if (score(timed[q]) < score(timed[selected])) selected = q;
            p.algo = timed[selected].algo;
            p.ws = timed[selected].ws;
            p.ok = true;
            p.candidates.clear();
            return true;
        }
        p.candidates.clear();
        basalt::log::warn("basalt", "prefill gemm: no fixed cuBLASLt algorithm for %s %lld x %lld",
                          ty == CUDA_R_16BF ? "BF16" : "FP16", (long long) N, (long long) K);
        return false;
    }
};

std::shared_ptr<Gemm::Plans> Gemm::prepared_plans_[16];

bool Gemm::prepare_model(const core::WeightTable& wt, const core::ModelGeometry& g, int64_t scratch_elems,
                         std::string& err) {
    return prepare_shapes(model_shapes(wt, g, scratch_elems), err);
}

bool Gemm::prepare_shapes(const std::vector<Shape>& declared, std::string& err) {
    if (!prepare_handle(err)) return false;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) {
        err = "prefill gemm: invalid current device";
        return false;
    }
    g_model_shapes[dev] = declared;
    auto key = [](const Shape& s) { return std::tie(s.bf16, s.n, s.k, s.ldy, s.align); };
    for (const auto& s : declared) {
        if (s.n <= 0 || s.k <= 0 || s.ldy < s.n || s.align < 2 || s.align > 256 ||
            (s.align & (s.align - 1)) != 0) {
            err = "prefill gemm: invalid startup plan shape";
            return false;
        }
    }
    auto& catalog = g_model_shapes[dev];
    std::sort(catalog.begin(), catalog.end(), [&](const Shape& a, const Shape& b) { return key(a) < key(b); });
    catalog.erase(std::unique(catalog.begin(), catalog.end(), [&](const Shape& a, const Shape& b) {
        return key(a) == key(b);
    }), catalog.end());
    std::shared_ptr<Plans> plans = std::make_shared<Plans>();
    plans->h = g_prepared_lt[dev];
    g_prepared_lt[dev] = nullptr;
    struct DeviceBuffer {
        void* p = nullptr;
        ~DeviceBuffer() { if (p) cudaFree(p); }
        bool allocate(size_t bytes) { return cudaMalloc(&p, bytes) == cudaSuccess; }
    } mem;
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        err = "prefill gemm: startup memory query failed";
        return false;
    }
    const uint64_t temp_budget = std::min<uint64_t>(128ull << 20, free_bytes / 4) & ~255ull;
    if (!plans->h || temp_budget <= kPlanWs + 1536 || !mem.allocate((size_t) temp_budget)) {
        err = "prefill gemm: startup plan workspace";
        return false;
    }
    void* const ws = mem.p;
    constexpr int64_t rows = 301;
    const auto& shapes = g_model_shapes[dev];
    for (size_t i = 0; i < shapes.size();) {
        const Shape& shape = shapes[i];
        size_t end = i + 1;
        while (end < shapes.size() && shapes[end].bf16 == shape.bf16 && shapes[end].n == shape.n &&
               shapes[end].k == shape.k && shapes[end].ldy == shape.ldy) ++end;
        const int base_align = shapes[end - 1].align;
        if ((uint64_t) shape.k > (temp_budget - kPlanWs - 1536) / 2 / (uint64_t) shape.n ||
            (uint64_t) shape.k > (128ull << 20) || (uint64_t) shape.ldy > (128ull << 20)) {
            err = "prefill gemm: startup plan shape exceeds temporary buffer limit";
            return false;
        }
        const uint64_t nw = (uint64_t) shape.n * (uint64_t) shape.k;
        const uint64_t fixed_bytes = kPlanWs + nw * 2 + 1536;
        const uint64_t row_bytes = (uint64_t) shape.k * 2 + (uint64_t) shape.ldy * 4;
        const int64_t buffer_rows = fixed_bytes < temp_budget
            ? (int64_t) std::min<uint64_t>(4096, (temp_budget - fixed_bytes) / row_bytes) : 0;
        if (buffer_rows < rows) {
            err = "prefill gemm: insufficient temporary memory for startup plan " + std::to_string(shape.n) +
                  " x " + std::to_string(shape.k);
            return false;
        }
        const uint64_t nx = (uint64_t) buffer_rows * (uint64_t) shape.k;
        auto round256 = [](uint64_t v) { return (v + 255) & ~255ull; };
        auto* w = (char*) mem.p + kPlanWs;
        auto* x = w + round256(nw * 2 + 256);
        auto* y = x + round256(nx * 2 + 256);
        std::vector<uint16_t> data((size_t) std::max(nw, nx + 128));
        const uint16_t base = shape.bf16 ? 0x3e00 : 0x3800;
        for (size_t j = 0; j < data.size(); ++j)
            data[j] = (uint16_t) (base | ((j * 73 + j / 17) & 0x7f) | ((j & 1) ? 0x8000 : 0));
        if (cudaMemcpy(w, data.data(), (size_t) nw * 2, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(x, data.data(), (size_t) nx * 2 + 256, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "prefill gemm: startup plan input upload";
            return false;
        }
        const auto ty = shape.bf16 ? CUDA_R_16BF : CUDA_R_16F;
        std::vector<float> reference((size_t) rows * shape.n), current(reference.size());
        for (size_t pos = i; pos < end; ++pos) {
            const size_t j = pos == i ? end - 1 : pos - 1;
            const int a = shapes[j].align;
            auto* xp = (const uint16_t*) (x + (a == 256 ? 0 : a));
            const size_t input_bytes = (size_t) (a == base_align ? nx : (uint64_t) rows * shape.k) * 2;
            if (cudaMemcpy((void*) xp, data.data(), input_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
                err = "prefill gemm: startup plan alignment input upload";
                return false;
            }
            auto& p = plans->plan(ty, shape.n, shape.k, shape.ldy, a);
            auto& base_plan = plans->plan(ty, shape.n, shape.k, shape.ldy, base_align);
            if (a != base_align && base_plan.ok) {
                p.candidates.assign(1, base_plan.algo);
                p.next = 0;
            }
            if (p.candidates.empty() ||
                !plans->settle(p, ty, xp, (const uint16_t*) w, (float*) y, rows, shape.n, shape.k, shape.ldy,
                               a, ws, kPlanWs, nullptr, a == base_align, buffer_rows)) {
                err = "prefill gemm: no checked startup plan for " + std::to_string(shape.n) + " x " +
                      std::to_string(shape.k) + " stride " + std::to_string(shape.ldy) + " alignment " +
                      std::to_string(a);
                return false;
            }
            if (!plans->run(p, p.algo, ty, xp, (const uint16_t*) w, (float*) y, rows, shape.n, shape.k,
                            shape.ldy, ws, kPlanWs, nullptr) ||
                cudaMemcpy2D(current.data(), (size_t) shape.n * 4, y, (size_t) shape.ldy * 4,
                             (size_t) shape.n * 4, (size_t) rows, cudaMemcpyDeviceToHost) != cudaSuccess) {
                err = "prefill gemm: startup plan alignment comparison failed";
                return false;
            }
            if (pos == i) reference = current;
            else if (std::memcmp(reference.data(), current.data(), current.size() * sizeof(float)) != 0) {
                err = "prefill gemm: startup plan output differs across alignments";
                return false;
            }
        }
        i = end;
    }
    plans->prepared = true;
    prepared_plans_[dev] = std::move(plans);
    return true;
}

bool Gemm::pinned(int type, const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K,
                  int64_t ldy, float beta) {
    if (!plans_) {
        int dev = 0;
        if (cudaGetDevice(&dev) == cudaSuccess && dev >= 0 && dev < 16 && prepared_plans_[dev]) {
            plans_ = prepared_plans_[dev];
        } else {
            plans_ = std::make_shared<Plans>();
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
    std::lock_guard<std::mutex> lock(plans_->mutex);
    const cudaDataType_t ty = (cudaDataType_t) type;
    const int align = std::min(alignment(X), std::min(alignment(W), alignment(Y)));
    const Plans::Key key{type, N, K, ldy, align};
    const auto found = plans_->plans.find(key);
    if (plans_->prepared && (found == plans_->plans.end() || !found->second.ok) && T >= 2 && beta == 0.0f) {
        throw GemmPlanError("prefill gemm: no prepared plan for " + std::to_string(N) + " x " +
                            std::to_string(K) + " stride " + std::to_string(ldy) + " alignment " +
                            std::to_string(align));
    }
    Plans::Plan& p = plans_->plan(ty, N, K, ldy, align);
    if (!p.ok) {
        // the candidates are checked on a call of two or more rows that overwrites Y; until then cublasGemmEx
        if (p.candidates.empty() || T < 2 || beta != 0.0f) return false;
        return plans_->settle(p, ty, X, W, Y, T, N, K, ldy, align, workspace_, ws_bytes_, stream_);
    }
    if (p.ws > ws_bytes_ || !plans_->rows(p, ty, T, N, K, ldy)) {
        if (plans_->prepared && T >= 2 && beta == 0.0f) {
            throw GemmPlanError("prefill gemm: prepared plan workspace or layout unavailable");
        }
        return false;
    }
    const float alpha = 1.0f;
    if (cublasLtMatmul(plans_->h, p.op, &alpha, W, p.a, X, p.b, &beta, Y, p.c, Y, p.c, &p.algo, workspace_, ws_bytes_,
                       (cudaStream_t) stream_) != CUBLAS_STATUS_SUCCESS) {
        p.ok = false;
        if (plans_->prepared && T >= 2 && beta == 0.0f) {
            throw GemmPlanError("prefill gemm: prepared plan refused");
        }
        basalt::log::warn("basalt", "prefill gemm: cublasLtMatmul refused %s %lld x %lld x %lld: cublasGemmEx for this "
                                    "shape from here",
                          ty == CUDA_R_16BF ? "BF16" : "FP16", (long long) T, (long long) N, (long long) K);
        return false;
    }
    return true;
}

void Gemm::free_plans() {
    plans_.reset();
}

}  // namespace basalt::prefill
