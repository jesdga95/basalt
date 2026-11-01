// prefill_gemm_width_test - the prompt path's FP16 and BF16 products (Gemm::native, Gemm::bf16) at the model's dense
// geometries: a row's products must be the same bits whatever the call's row count and the row's place in it (rows of a
// 512-row call again in calls of 1, 7, 37, 64 and 300 rows, row r at (r + 7) % w), checked against
//   - the cublasGemmEx call (CUBLAS_GEMM_DEFAULT) these products ran on before: max |diff| / max |y|, and its own count
//     of rows that differ between widths;
//   - a double-precision reference over the same FP16 / BF16 operands on sampled rows: relative RMS within 2e-9 x K and
//     the worst row within 4e-9 x K (one fp32 loop over K per output).
// --bench [T...]: microseconds per product (weights already in FP16 / BF16) at each T (default 32768 4096 512 64) for
// Gemm and for cublasGemmEx, and the sum per chunk over the model's layers.  Exit 77 without a device.
#include "basalt/kernels/dequant_bf16.hpp"
#include "basalt/prefill/gemm.hpp"
#include "basalt/core/layout.hpp"

#include "basalt/ggml/ggml.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace basalt::core {
struct WeightTableFixture {
    static void put(WeightTable& wt, const std::string& name, int64_t n, int64_t k) {
        WeightRef r;
        r.ne0 = k;
        r.ne1 = n;
        wt.table_[name] = r;
    }
};
}

namespace {
using basalt::prefill::Gemm;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
void ckb(cublasStatus_t e, const char* what) {
    if (e != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(what) + ": cuBLAS status " + std::to_string((int) e));
}

struct Dev {
    void* p = nullptr;
    explicit Dev(size_t n) { ck(cudaMalloc(&p, std::max<size_t>(n, 256)), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    template <typename T> T* as() const { return (T*) p; }
};

constexpr ggml_type kBF16 = (ggml_type) -1;   // a BF16 tensor (this test's marker, not a GGUF type)

struct Shape {
    const char* name;
    ggml_type t;       // kBF16: a BF16 tensor (Gemm::bf16), else GGUF blocks (Gemm::native)
    int64_t n, k, ldy;
    int per_chunk;     // products of this shape per chunk over the model's 48 layers (benchmark weighting)
};
const Shape kShapes[] = {
    {"attn_qkv Q6_K", GGML_TYPE_Q6_K, 10240, 2560, 10240, 36},
    {"attn_gate Q6_K", GGML_TYPE_Q6_K, 6144, 2560, 6144, 36},
    {"ssm_out Q6_K", GGML_TYPE_Q6_K, 2560, 6144, 2560, 36},
    {"attn_output Q4_K", GGML_TYPE_Q4_K, 2560, 6144, 2560, 12},
    {"attn_q Q6_K", GGML_TYPE_Q6_K, 12288, 2560, 12288, 12},
    {"attn_k Q6_K", GGML_TYPE_Q6_K, 512, 2560, 512, 12},
    {"attn_v Q5_K", GGML_TYPE_Q5_K, 512, 2560, 512, 12},
    {"shexp gate Q4_K", GGML_TYPE_Q4_K, 640, 2560, 640, 48},
    {"shexp up IQ4_XS", GGML_TYPE_IQ4_XS, 640, 2560, 640, 48},
    {"shexp down IQ4_NL", GGML_TYPE_IQ4_NL, 2560, 640, 2560, 48},
    {"hc down BF16", kBF16, 320, 10240, 320, 96},
    {"hc inject BF16", kBF16, 4, 10240, 4, 96},
    {"hc up BF16", kBF16, 10240, 320, 10240, 0},
    {"ssm_alpha BF16", kBF16, 48, 2560, 96, 72},
    {"indexer k BF16", kBF16, 128, 2560, 128, 12},
    {"indexer q BF16", kBF16, 512, 2560, 512, 12},
    {"router BF16", kBF16, 512, 2560, 512, 95},
    {"shexp gate_inp BF16", kBF16, 1, 2560, 1, 48},
    {"ple value BF16", kBF16, 2560, 2560, 2560, 1},
    {"ple key BF16", kBF16, 10240, 2560, 10240, 1},
};

uint16_t f2bf(float f) {   // round to nearest even
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
float bf2f(uint16_t h) {
    const uint32_t u = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// random blocks with sane fp16 scales (the weights a few hundredths, centred on zero)
void fill_blocks(uint8_t* m, ggml_type t, int64_t rows, int64_t cols, std::mt19937& rng) {
    const size_t bs = ggml_type_size(t), nb = (size_t) rows * (size_t) (cols / ggml_blck_size(t));
    std::uniform_int_distribution<int> byte(0, 255);
    for (size_t i = 0; i < nb * bs; ++i) m[i] = (uint8_t) byte(rng);
    float lo = 0.0008f, mid = 0.0f;
    switch (t) {
        case GGML_TYPE_Q4_K: lo = 0.0002f; mid = 7.5f; break;
        case GGML_TYPE_Q5_K: lo = 0.0001f; mid = 15.5f; break;
        case GGML_TYPE_IQ4_XS: lo = 0.0002f; break;
        case GGML_TYPE_IQ4_NL: lo = 0.0004f; break;
        default: break;
    }
    std::uniform_real_distribution<float> sc(lo, 4.0f * lo);
    if (t == GGML_TYPE_Q6_K) {   // {ql, qh, int8 scales, d}
        std::uniform_int_distribution<int> s8(-100, 100);
        std::uniform_real_distribution<float> sd(0.00002f, 0.00008f);
        for (size_t b = 0; b < nb; ++b) {
            for (int j = 0; j < 16; ++j) m[b * bs + 192 + j] = (uint8_t) (int8_t) s8(rng);
            const ggml_fp16_t h = ggml_fp32_to_fp16(sd(rng));
            std::memcpy(m + b * bs + 208, &h, 2);
        }
        return;
    }
    for (size_t b = 0; b < nb; ++b) {
        const ggml_fp16_t h = ggml_fp32_to_fp16(sc(rng));
        std::memcpy(m + b * bs, &h, 2);
        if (mid != 0.0f) {
            const ggml_fp16_t hm = ggml_fp32_to_fp16(mid * ggml_fp16_to_fp32(h));
            std::memcpy(m + b * bs + 2, &hm, 2);
        }
    }
}

template <typename T> std::vector<T> download(const void* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}

// One shape's operands on the device: W as FP16 / BF16 rows (and its GGUF blocks), X as FP16 / BF16 rows.
struct Operands {
    const Shape& s;
    bool bf;
    Dev wq, w16, x16, y;
    std::vector<uint16_t> w_h, x_h;
    int64_t T;
    Operands(const Shape& sh, int64_t rows, std::mt19937& rng)
        : s(sh), bf(sh.t == kBF16), wq(bf ? 256 : ggml_row_size(sh.t, sh.k) * (size_t) sh.n),
          w16((size_t) sh.n * sh.k * 2), x16((size_t) rows * sh.k * 2), y((size_t) rows * sh.ldy * 4), T(rows) {
        std::normal_distribution<float> nd(0.0f, 1.0f);
        if (bf) {
            w_h.resize((size_t) sh.n * sh.k);
            for (auto& v : w_h) v = f2bf(0.02f * nd(rng));
            ck(cudaMemcpy(w16.p, w_h.data(), w_h.size() * 2, cudaMemcpyHostToDevice), "w");
        } else {
            std::vector<uint8_t> blocks(ggml_row_size(sh.t, sh.k) * (size_t) sh.n);
            fill_blocks(blocks.data(), sh.t, sh.n, sh.k, rng);
            ck(cudaMemcpy(wq.p, blocks.data(), blocks.size(), cudaMemcpyHostToDevice), "wq");
            basalt::kernels::dequant_f16(sh.t, wq.p, 0, sh.n, sh.k, w16.as<uint16_t>(), nullptr);
            ck(cudaDeviceSynchronize(), "dequant");
            w_h = download<uint16_t>(w16.p, (size_t) sh.n * sh.k);
        }
        // 512 distinct rows, repeated for longer calls
        const int64_t uniq = std::min<int64_t>(rows, 512);
        x_h.resize((size_t) uniq * sh.k);
        for (auto& v : x_h) v = bf ? f2bf(nd(rng)) : ggml_fp32_to_fp16(nd(rng));
        for (int64_t r0 = 0; r0 < rows; r0 += uniq)
            ck(cudaMemcpy((char*) x16.p + (size_t) r0 * sh.k * 2, x_h.data(),
                          (size_t) std::min(uniq, rows - r0) * sh.k * 2, cudaMemcpyHostToDevice), "x");
        ck(cudaMemset(y.p, 0, (size_t) rows * sh.ldy * 4), "y");
        ck(cudaDeviceSynchronize(), "upload");
    }
    float w(size_t i) const { return bf ? bf2f(w_h[i]) : ggml_fp16_to_fp32(w_h[i]); }
    float x(size_t i) const { return bf ? bf2f(x_h[i]) : ggml_fp16_to_fp32(x_h[i]); }
};

// The product as the prompt path runs it (GGUF blocks through Gemm::native, BF16 through Gemm::bf16).
void run_gemm(Gemm& g, const Operands& o, const void* x, float* y, int64_t rows) {
    if (o.bf) g.bf16((const uint16_t*) x, o.w16.as<uint16_t>(), y, rows, o.s.n, o.s.k, o.s.ldy);
    else g.native((const uint16_t*) x, o.s.t, o.wq.p, y, rows, o.s.n, o.s.k, o.s.ldy);
}
// The same product with W already in FP16 / BF16 (the benchmark: no dequantization).
void run_product(Gemm& g, const Operands& o, const void* x, float* y, int64_t rows) {
    if (o.bf) g.bf16((const uint16_t*) x, o.w16.as<uint16_t>(), y, rows, o.s.n, o.s.k, o.s.ldy);
    else g.f16((const uint16_t*) x, o.w16.as<uint16_t>(), y, rows, o.s.n, o.s.k, o.s.ldy);
}
// The call these products ran on before.
void run_default(cublasHandle_t h, const Operands& o, const void* x, float* y, int64_t rows) {
    const float alpha = 1.0f, beta = 0.0f;
    const cudaDataType_t ty = o.bf ? CUDA_R_16BF : CUDA_R_16F;
    ckb(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, (int) o.s.n, (int) rows, (int) o.s.k, &alpha, o.w16.p, ty,
                     (int) o.s.k, x, ty, (int) o.s.k, &beta, y, CUDA_R_32F, (int) o.s.ldy, CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT),
        "cublasGemmEx");
}

// returns the failures
int check_shape(const Shape& s, cudaStream_t st, Gemm& g, cublasHandle_t h) {
    const int64_t T = 512;
    std::mt19937 rng(11 + (unsigned) s.t * 131 + (unsigned) s.n * 7 + (unsigned) s.k);
    Operands o(s, T, rng);
    Dev xs((size_t) T * s.k * 2), ys((size_t) T * s.ldy * 4);
    auto rows_of = [&](const Dev& y, int64_t rows) {   // the n products of each row
        const std::vector<float> all = download<float>(y.p, (size_t) rows * s.ldy);
        std::vector<float> r((size_t) rows * s.n);
        for (int64_t t = 0; t < rows; ++t) std::memcpy(&r[(size_t) t * s.n], &all[(size_t) t * s.ldy], (size_t) s.n * 4);
        return r;
    };
    // the shape's first call chooses and checks its algorithm: its wall time against the next call's
    const auto t0 = std::chrono::steady_clock::now();
    run_gemm(g, o, o.x16.p, o.y.as<float>(), T);
    ck(cudaStreamSynchronize(st), "gemm");
    const auto t1 = std::chrono::steady_clock::now();
    run_gemm(g, o, o.x16.p, o.y.as<float>(), T);
    ck(cudaStreamSynchronize(st), "gemm again");
    const auto t2 = std::chrono::steady_clock::now();
    const double ms_first = std::chrono::duration<double, std::milli>(t1 - t0).count(),
                 ms_next = std::chrono::duration<double, std::milli>(t2 - t1).count();
    const std::vector<float> y_new = rows_of(o.y, T);
    run_default(h, o, o.x16.p, o.y.as<float>(), T);
    ck(cudaStreamSynchronize(st), "default");
    const std::vector<float> y_old = rows_of(o.y, T);

    // width: rows [0, w) again as a call of w rows, row r at (r + 7) % w
    long diff_new = 0, diff_old = 0, n_rows = 0;
    for (const int64_t w : {(int64_t) 1, (int64_t) 7, (int64_t) 37, (int64_t) 64, (int64_t) 300}) {
        std::vector<uint16_t> xw((size_t) w * s.k);
        for (int64_t r = 0; r < w; ++r)
            std::memcpy(&xw[(size_t) ((r + 7) % w) * s.k], &o.x_h[(size_t) r * s.k], (size_t) s.k * 2);
        ck(cudaMemcpy(xs.p, xw.data(), xw.size() * 2, cudaMemcpyHostToDevice), "xw");
        ck(cudaDeviceSynchronize(), "upload xw");
        run_gemm(g, o, xs.p, ys.as<float>(), w);
        ck(cudaStreamSynchronize(st), "gemm w");
        const std::vector<float> yn = rows_of(ys, w);
        run_default(h, o, xs.p, ys.as<float>(), w);
        ck(cudaStreamSynchronize(st), "default w");
        const std::vector<float> yo = rows_of(ys, w);
        for (int64_t r = 0; r < w; ++r) {
            const size_t at = (size_t) ((r + 7) % w);
            diff_new += std::memcmp(&y_new[(size_t) r * s.n], &yn[at * s.n], (size_t) s.n * 4) != 0;
            diff_old += std::memcmp(&y_old[(size_t) r * s.n], &yo[at * s.n], (size_t) s.n * 4) != 0;
            ++n_rows;
        }
    }

    // the new products against the old, and both against the double reference on sampled rows
    double dmax = 0, ymax = 0;
    for (size_t i = 0; i < y_new.size(); ++i) {
        dmax = std::max(dmax, (double) std::fabs(y_new[i] - y_old[i]));
        ymax = std::max(ymax, (double) std::fabs(y_old[i]));
    }
    const int64_t sample[] = {0, 1, 5, 63, 64, 200, 377, 511};
    double e2n = 0, e2o = 0, r2 = 0, worst_n = 0, worst_o = 0;
    for (const int64_t r : sample) {
        double rm = 0, en = 0, eo = 0;
        for (int64_t c = 0; c < s.n; ++c) {
            double acc = 0;
            for (int64_t j = 0; j < s.k; ++j) acc += (double) o.x((size_t) r * s.k + j) * (double) o.w((size_t) c * s.k + j);
            const double dn = (double) y_new[(size_t) r * s.n + c] - acc, d0 = (double) y_old[(size_t) r * s.n + c] - acc;
            e2n += dn * dn; e2o += d0 * d0; r2 += acc * acc;
            rm = std::max(rm, std::fabs(acc)); en = std::max(en, std::fabs(dn)); eo = std::max(eo, std::fabs(d0));
        }
        if (rm > 0) { worst_n = std::max(worst_n, en / rm); worst_o = std::max(worst_o, eo / rm); }
    }
    const double rms_n = r2 > 0 ? std::sqrt(e2n / r2) : 0, rms_o = r2 > 0 ? std::sqrt(e2o / r2) : 0;
    // one fp32 K loop per output: the rounding error grows with K
    const bool bad = diff_new != 0 || rms_n > 2e-9 * (double) s.k || worst_n > 4e-9 * (double) s.k;
    std::printf("%-20s %5lld x %5lld | width (%ld rows): %ld differ (cublasGemmEx: %ld) | vs cublasGemmEx max %.1e of max "
                "|y| | vs double rel RMS %.1e worst %.1e (cublasGemmEx %.1e / %.1e) | first call %.2f ms, next %.2f ms %s\n",
                s.name, (long long) s.n, (long long) s.k, n_rows, diff_new, diff_old, ymax > 0 ? dmax / ymax : 0.0,
                rms_n, worst_n, rms_o, worst_o, ms_first, ms_next, bad ? "FAIL" : "ok");
    return bad ? 1 : 0;
}

template <typename F> double time_us(F&& f, cudaStream_t st, int reps) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    f();
    f();
    ck(cudaStreamSynchronize(st), "warm-up");
    std::vector<float> v;
    for (int i = 0; i < reps; ++i) {
        cudaEventRecord(a, st);
        f();
        cudaEventRecord(b, st);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        v.push_back(ms);
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    std::sort(v.begin(), v.end());
    return 1e3 * v[v.size() / 2];
}
}  // namespace

int main(int argc, char** argv) {
    try {
        {
            basalt::core::WeightTable wt;
            basalt::core::ModelGeometry geom;
            geom.n_layers = 1;
            basalt::core::WeightTableFixture::put(wt, "blk.0.attn_qkv.weight", 10, 200);
            basalt::core::WeightTableFixture::put(wt, "blk.0.ssm_alpha.weight", 48, 2560);
            const auto shapes = Gemm::model_shapes(wt, geom, 600);
            auto has = [&](bool bf16, int64_t n, int64_t k, int64_t ld, int align) {
                return std::any_of(shapes.begin(), shapes.end(), [&](const Gemm::Shape& s) {
                    return s.bf16 == bf16 && s.n == n && s.k == k && s.ldy == ld && s.align == align;
                });
            };
            if (!has(false, 3, 200, 10, 256) || !has(false, 1, 200, 10, 2) ||
                !has(true, 48, 2560, 96, 64))
                throw std::runtime_error("model shape catalog missed a split tail or strided projection");
        }
        int n_dev = 0;
        if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
            std::printf("no CUDA device - skipped\n");
            return 77;
        }
        bool bench = false;
        std::vector<int64_t> bench_t;
        for (int i = 1; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--bench")) bench = true;
            else if (bench) bench_t.push_back(std::atoll(argv[i]));
        }
        if (bench && bench_t.empty()) bench_t = {32768, 4096, 512, 64};
        cudaStream_t st;
        ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
        Gemm g;
        std::string err;
        if (!g.init(st, 32ll << 20, err)) throw std::runtime_error(err);
        cublasHandle_t h = nullptr;
        ckb(cublasCreate(&h), "cublasCreate");
        ckb(cublasSetStream(h, st), "cublasSetStream");
        Dev ws(32u << 20);
        ckb(cublasSetWorkspace(h, ws.p, 32u << 20), "cublasSetWorkspace");
        ckb(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
        if (bench) {
            for (const int64_t T : bench_t) {
                double sum_new = 0, sum_old = 0;
                for (const Shape& s : kShapes) {
                    std::mt19937 rng(3);
                    Operands o(s, T, rng);
                    const int reps = T >= 8192 ? 7 : 25;
                    const double tn = time_us([&] { run_product(g, o, o.x16.p, o.y.as<float>(), T); }, st, reps);
                    const double to = time_us([&] { run_default(h, o, o.x16.p, o.y.as<float>(), T); }, st, reps);
                    const double tf = 2.0 * (double) T * s.n * s.k / 1e6;
                    std::printf("T %6lld %-20s Gemm %9.1f us %6.1f TFLOP/s | cublasGemmEx %9.1f us %6.1f TFLOP/s | %+6.1f%%\n",
                                (long long) T, s.name, tn, tf / tn, to, tf / to, 100.0 * (tn / to - 1.0));
                    sum_new += s.per_chunk * tn;
                    sum_old += s.per_chunk * to;
                }
                std::printf("T %6lld per chunk, 48 layers: Gemm %.2f ms, cublasGemmEx %.2f ms (%+.2f%%)\n", (long long) T,
                            sum_new / 1e3, sum_old / 1e3, 100.0 * (sum_new / sum_old - 1.0));
            }
            cublasDestroy(h);
            return 0;
        }
        int fails = 0;
        for (const Shape& s : kShapes) fails += check_shape(s, st, g, h);
        cublasDestroy(h);
        std::printf("prefill_gemm_width_test: %d failure(s)\n", fails);
        return fails ? 1 : 0;
    } catch (const std::exception& e) {
        std::printf("prefill_gemm_width_test: %s\n", e.what());
        return 1;
    }
}
