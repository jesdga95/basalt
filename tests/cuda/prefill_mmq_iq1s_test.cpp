// prefill_mmq_iq1s_test - llama.cpp's MMQ for the DeepSeek-V4.1 expert formats (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S) on random blocks at the
// model's geometry (5120 -> 2 x 2304 and 2304 -> 5120), against a double-precision reference: the weights dequantized by
// ggml's own to_float times the FP32 activations. MMQ rounds the activations to int8 per 32 values, so the check is a
// relative RMS bound (2e-2), not equality. Three experts with 200, 0 and 57 rows in one launch. Prints a microbench line
// (microseconds per product, achieved int8 TOP/s). Exit 77 without a CUDA device.
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/prefill/moe_mmq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace mmq = basalt::prefill::mmq;
namespace kg = basalt::kernels;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Dev {
    void* p = nullptr;
    explicit Dev(size_t n) { ck(cudaMalloc(&p, std::max<size_t>(n, 16)), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
};

// random codes are valid blocks in both formats; the fp16 scale at the block's start is set to a sane value
std::vector<uint8_t> make_weights(ggml_type t, int64_t rows, int64_t cols, int n_exp, std::mt19937& rng) {
    const size_t bs = ggml_type_size(t), per_exp = (size_t) rows * (size_t) (cols / ggml_blck_size(t)) * bs;
    std::vector<uint8_t> w(per_exp * (size_t) n_exp);
    std::uniform_int_distribution<int> byte(0, 255);
    for (uint8_t& v : w) v = (uint8_t) byte(rng);
    std::uniform_real_distribution<float> sc(0.002f, 0.008f);
    for (size_t b = 0; b < w.size() / bs; ++b) {
        const ggml_fp16_t h = ggml_fp32_to_fp16(sc(rng));
        if (t == GGML_TYPE_IQ1_M) {   // its fp16 sits in the top nibbles of the block's last four int16
            for (int i = 0; i < 4; ++i) {
                uint16_t v;
                std::memcpy(&v, w.data() + b * bs + bs - 8 + 2 * (size_t) i, 2);
                v = (uint16_t) ((v & 0x0FFF) | (((h >> (4 * i)) & 0xF) << 12));
                std::memcpy(w.data() + b * bs + bs - 8 + 2 * (size_t) i, &v, 2);
            }
            continue;
        }
        std::memcpy(w.data() + b * bs, &h, 2);
    }
    return w;
}

struct Result {
    double rel_rms = 0, worst_row = 0, us = 0, tops = 0;
};

Result run_case(ggml_type t, int64_t w_rows, int64_t w_cols, const std::vector<int>& counts, std::mt19937& rng) {
    const int n = (int) counts.size();
    int64_t rows = 0, max_rows = 0;
    std::vector<int32_t> bounds{0};
    for (int c : counts) {
        rows += c;
        max_rows = std::max<int64_t>(max_rows, c);
        bounds.push_back((int32_t) rows);
    }
    const size_t per_exp = (size_t) w_rows * (size_t) (w_cols / ggml_blck_size(t)) * ggml_type_size(t);
    const std::vector<uint8_t> w = make_weights(t, w_rows, w_cols, n, rng);
    std::vector<float> x((size_t) rows * (size_t) w_cols);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (float& v : x) v = nd(rng);

    Dev dw(w.size()), dx(x.size() * 4), dxq(mmq::q8_bytes(rows, w_cols)), dids((size_t) rows * 4),
        db(bounds.size() * 4), dy((size_t) rows * (size_t) w_rows * 4);
    ck(cudaMemcpy(dw.p, w.data(), w.size(), cudaMemcpyHostToDevice), "w");
    ck(cudaMemcpy(dx.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(db.p, bounds.data(), bounds.size() * 4, cudaMemcpyHostToDevice), "bounds");
    mmq::iota((int32_t*) dids.p, rows, nullptr);
    mmq::quantize((const float*) dx.p, nullptr, dxq.p, (int) t, w_cols, w_cols, rows, nullptr);

    mmq::Product pr;
    pr.w = dw.p;
    pr.type = (int) t;
    pr.w_rows = w_rows;
    pr.w_cols = w_cols;
    pr.expert_bytes = per_exp;
    pr.n = n;
    pr.xq = dxq.p;
    pr.bounds = (const int32_t*) db.p;
    pr.ids = (const int32_t*) dids.p;
    pr.total_rows = rows;
    pr.max_rows = max_rows;
    pr.dst = (float*) dy.p;
    pr.ld_dst = w_rows;
    mmq::Context ctx;
    ctx.run(pr, nullptr);
    ck(cudaDeviceSynchronize(), "run");

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    const int reps = 20;
    cudaEventRecord(e0);
    for (int i = 0; i < reps; ++i) ctx.run(pr, nullptr);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);

    std::vector<float> y((size_t) rows * (size_t) w_rows);
    ck(cudaMemcpy(y.data(), dy.p, y.size() * 4, cudaMemcpyDeviceToHost), "y");
    const auto* tr = ggml_get_type_traits(t);
    const size_t row_bytes = ggml_row_size(t, w_cols);
    std::vector<float> wf((size_t) w_cols);
    double num = 0, den = 0, worst = 0;
    std::vector<double> rn((size_t) rows, 0.0), rd((size_t) rows, 0.0);
    for (int e = 0; e < n; ++e) {
        for (int64_t o = 0; o < w_rows; ++o) {
            tr->to_float(w.data() + (size_t) e * per_exp + (size_t) o * row_bytes, wf.data(), w_cols);
            for (int32_t r = bounds[(size_t) e]; r < bounds[(size_t) e + 1]; ++r) {
                double ref = 0;
                for (int64_t k = 0; k < w_cols; ++k) ref += (double) wf[(size_t) k] * x[(size_t) r * (size_t) w_cols + (size_t) k];
                const double d = y[(size_t) r * (size_t) w_rows + (size_t) o] - ref;
                num += d * d;
                den += ref * ref;
                rn[(size_t) r] += d * d;
                rd[(size_t) r] += ref * ref;
            }
        }
    }
    for (int64_t r = 0; r < rows; ++r) worst = std::max(worst, std::sqrt(rn[(size_t) r] / std::max(rd[(size_t) r], 1e-30)));
    Result res;
    res.rel_rms = std::sqrt(num / den);
    res.worst_row = worst;
    res.us = 1e3 * ms / reps;
    res.tops = 2.0 * (double) rows * (double) w_rows * (double) w_cols / (res.us * 1e-6) / 1e12;
    return res;
}

// A format against the dp4a matvec (the one IQ1_M replaces in the prompt path), and a row's value whatever the launch it
// rides in: two experts in one launch, each alone, and the rows in reverse order.
bool mmq_checks(ggml_type t, int64_t w_rows, int64_t w_cols, std::mt19937& rng) {
    const int counts[2] = {40, 24};
    const int rows = counts[0] + counts[1];
    const size_t per_exp = (size_t) w_rows * (size_t) (w_cols / ggml_blck_size(t)) * ggml_type_size(t);
    const std::vector<uint8_t> w = make_weights(t, w_rows, w_cols, 2, rng);
    std::vector<float> x((size_t) rows * (size_t) w_cols);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (float& v : x) v = nd(rng);
    Dev dw(w.size()), dx(x.size() * 4), dxq(mmq::q8_bytes(rows, w_cols)), dids((size_t) rows * 4), db(3 * 4),
        dy((size_t) rows * (size_t) w_rows * 4);
    ck(cudaMemcpy(dw.p, w.data(), w.size(), cudaMemcpyHostToDevice), "w");
    ck(cudaMemcpy(dx.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    mmq::iota((int32_t*) dids.p, rows, nullptr);
    mmq::Context ctx;
    auto launch = [&](const uint8_t* wp, int n, const std::vector<int32_t>& bounds, const std::vector<int32_t>& xrows, int32_t dst0) {
        std::vector<int32_t> dst;
        for (size_t i = 0; i < xrows.size(); ++i) dst.push_back(dst0 + (int32_t) i);
        Dev dxr(xrows.size() * 4), ddst(dst.size() * 4);
        ck(cudaMemcpy(dxr.p, xrows.data(), xrows.size() * 4, cudaMemcpyHostToDevice), "rows");
        ck(cudaMemcpy(ddst.p, dst.data(), dst.size() * 4, cudaMemcpyHostToDevice), "dst");
        ck(cudaMemcpy(db.p, bounds.data(), bounds.size() * 4, cudaMemcpyHostToDevice), "bounds");
        mmq::quantize((const float*) dx.p, (const int32_t*) dxr.p, dxq.p, (int) t, w_cols, w_cols, (int64_t) xrows.size(), nullptr);
        mmq::Product pr;
        pr.w = wp;
        pr.type = (int) t;
        pr.w_rows = w_rows;
        pr.w_cols = w_cols;
        pr.expert_bytes = per_exp;
        pr.n = n;
        pr.xq = dxq.p;
        pr.bounds = (const int32_t*) db.p;
        pr.ids = (const int32_t*) ddst.p;
        pr.total_rows = (int64_t) xrows.size();
        pr.max_rows = *std::max_element(counts, counts + 2);
        pr.dst = (float*) dy.p;
        pr.ld_dst = w_rows;
        ctx.run(pr, nullptr);
        ck(cudaDeviceSynchronize(), "run");
    };
    std::vector<int32_t> all(rows);
    for (int i = 0; i < rows; ++i) all[(size_t) i] = i;
    auto read = [&]() {
        std::vector<float> y((size_t) rows * (size_t) w_rows);
        ck(cudaMemcpy(y.data(), dy.p, y.size() * 4, cudaMemcpyDeviceToHost), "y");
        return y;
    };
    launch((const uint8_t*) dw.p, 2, {0, counts[0], rows}, all, 0);
    const std::vector<float> together = read();
    launch((const uint8_t*) dw.p, 1, {0, counts[0]}, std::vector<int32_t>(all.begin(), all.begin() + counts[0]), 0);
    launch((const uint8_t*) dw.p + per_exp, 1, {0, counts[1]}, std::vector<int32_t>(all.begin() + counts[0], all.end()), counts[0]);
    auto diff = [&](const std::vector<float>& y, const std::vector<int32_t>& order, int n) {
        double num = 0, den = 0;
        for (int i = 0; i < n; ++i)
            for (int64_t o = 0; o < w_rows; ++o) {
                const double a = y[(size_t) i * (size_t) w_rows + (size_t) o], b = together[(size_t) order[(size_t) i] * (size_t) w_rows + (size_t) o];
                num += (a - b) * (a - b);
                den += b * b;
            }
        return std::sqrt(num / den);
    };
    const std::vector<float> apart = read();
    const bool alone = apart == together;
    const double alone_rel = diff(apart, all, rows);
    std::vector<int32_t> rev(all.rbegin(), all.rbegin() + counts[1]);
    rev.insert(rev.end(), all.rbegin() + counts[1], all.rend());
    launch((const uint8_t*) dw.p + per_exp, 1, {0, counts[1]}, std::vector<int32_t>(rev.begin(), rev.begin() + counts[1]), 0);
    std::vector<float> y = read();
    bool reversed = true;
    for (int i = 0; i < counts[1]; ++i)
        reversed = reversed && !std::memcmp(&y[(size_t) i * (size_t) w_rows], &together[(size_t) rev[(size_t) i] * (size_t) w_rows], (size_t) w_rows * 4);
    const double rev_rel = diff(y, rev, counts[1]);

    // the matvec the prompt path ran before, one column at a time on the same activations
    Dev dq1(kg::native_q8_1_bytes((int) w_cols, 1)), dv((size_t) w_rows * 4);
    cudaStream_t st = nullptr;
    ck(cudaStreamCreate(&st), "stream");
    double num = 0, den = 0;
    for (int r = 0; r < rows; ++r) {
        kg::native_quantize_q8_1((const float*) dx.p + (size_t) r * (size_t) w_cols, dq1.p, (int) w_cols, 1, st);
        kg::native_mmvq((int) t, (const uint8_t*) dw.p + (r < counts[0] ? 0 : per_exp), dq1.p, (float*) dv.p, (int) w_cols, (int) w_rows, 1, st);
        ck(cudaStreamSynchronize(st), "matvec");
        std::vector<float> v((size_t) w_rows);
        ck(cudaMemcpy(v.data(), dv.p, v.size() * 4, cudaMemcpyDeviceToHost), "v");
        for (int64_t o = 0; o < w_rows; ++o) {
            const double d = (double) together[(size_t) r * (size_t) w_rows + (size_t) o] - (double) v[(size_t) o];
            num += d * d;
            den += (double) v[(size_t) o] * (double) v[(size_t) o];
        }
    }
    cudaStreamDestroy(st);
    const double rel = std::sqrt(num / den);
    const bool pass = alone_rel < 1e-5 && rev_rel < 1e-5 && rel < 2e-2;
    std::printf("%s %lld x %lld: launch-independent %s (rel RMS %.2e), row order %s (%.2e), against the dp4a matvec rel RMS %.2e  %s\n", ggml_type_name(t), (long long) w_rows,
                (long long) w_cols, alone ? "bitwise" : "no", alone_rel, reversed ? "bitwise" : "no", rev_rel, rel, pass ? "ok" : "FAIL");
    return pass;
}

}  // namespace

int main() {
    int dev = 0;
    if (cudaGetDeviceCount(&dev) != cudaSuccess || dev == 0) return 77;
    if (!mmq::built() || !mmq::supported(GGML_TYPE_IQ1_S) || !mmq::supported(GGML_TYPE_IQ1_M) || !mmq::supported(GGML_TYPE_IQ2_XXS)) {
        std::printf("MMQ does not cover IQ1_S / IQ1_M / IQ2_XXS in this build\n");
        return 1;
    }
    std::mt19937 rng(7);
    bool ok = true;
    struct Case {
        ggml_type t;
        int64_t rows, cols;
        const char* what;
    };
    for (const Case& c : {Case{GGML_TYPE_IQ1_S, 4608, 5120, "IQ1_S gate+up 5120 -> 4608"},
                          Case{GGML_TYPE_IQ1_M, 4608, 5120, "IQ1_M gate+up 5120 -> 4608"},
                          Case{GGML_TYPE_IQ1_M, 5120, 2304, "IQ1_M down 2304 -> 5120"},
                          Case{GGML_TYPE_IQ2_XXS, 5120, 2304, "IQ2_XXS down 2304 -> 5120"},
                          Case{GGML_TYPE_IQ2_XXS, 4608, 5120, "IQ2_XXS gate+up 5120 -> 4608"},
                          Case{GGML_TYPE_IQ2_XS, 4608, 5120, "IQ2_XS gate+up 5120 -> 4608"},
                          Case{GGML_TYPE_IQ2_XS, 5120, 2304, "IQ2_XS down 2304 -> 5120"},
                          Case{GGML_TYPE_IQ2_S, 5120, 2304, "IQ2_S down 2304 -> 5120"}}) {
        const Result r = run_case(c.t, c.rows, c.cols, {200, 0, 57}, rng);
        const bool pass = r.rel_rms < 2e-2 && r.worst_row < 0.1;
        ok = ok && pass;
        std::printf("%s: rel RMS %.2e, worst row %.2e, %.0f us/product (257 rows), %.1f int8 TOP/s  %s\n", c.what, r.rel_rms,
                    r.worst_row, r.us, r.tops, pass ? "ok" : "FAIL");
    }
    for (ggml_type t : {GGML_TYPE_IQ1_M, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS}) {
        ok = mmq_checks(t, 4608, 5120, rng) && ok;
        ok = mmq_checks(t, 5120, 2304, rng) && ok;
    }
    return ok ? 0 : 1;
}
