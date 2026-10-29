// prefill_fp8_dense_test - the prompt path's dense projections (Y[T, N] = X[T, K] . W[N, K]^T, W in its native GGUF
// blocks) on the FP8 tensor cores (gemm_fp8.hpp, Gemm::act8 / Gemm::native8), against
//   - a double-precision reference: W dequantized by ggml's own to_float, times the FP32 activations, and
//   - the FP16 path (W dequantized to FP16 in a scratch, X rounded to FP16, cuBLAS), as Gemm::native runs it.
// Checks, per real geometry and weight type, plain and with 1-in-200 12x outliers:
//   - the E4M3 conversion (rows of FP16 values, per-row and per-32-block power-of-two scales) against a CPU encoder,
//     bit for bit, scales included (the block scales at their place in cuBLASLt's tiled layout);
//   - each scale granularity cuBLASLt offers here against the reference (E4M3 keeps 3 mantissa bits: the bounds are
//     relative RMS 6e-2, worst row 0.25), and the path the prompt runs (Gemm::native8) the same;
//   - an all-zero row gives zeros;
//   - width invariance: rows run in a batch of 64 at shifted places give the same E4M3 bytes as in the full batch,
//     and products within 1e-4 of the max |y| (cuBLASLt may pick another algorithm for another width).
// --bench [T]: microseconds per product and TFLOP/s at T rows (default 32768): the FP16 path, each FP8 scale
// granularity (conversions, cuBLASLt's pick, the best of the algorithms it offers, fast accumulation), the prompt
// path's FP8 product, and llama.cpp's MMQ on Q8_0 weights for comparison.
// --vram: the device memory the prompt path's FP8 products take on first use (cuBLASLt's lazily loaded kernels), after
// the FP16 products of the same shapes ran.  Exit 77 without a device.
#include "basalt/kernels/dequant_bf16.hpp"
#include "basalt/prefill/gemm.hpp"
#include "basalt/prefill/gemm_fp8.hpp"
#include "basalt/prefill/moe_mmq.hpp"

#include "basalt/ggml/ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
namespace fp8 = basalt::prefill::fp8;
namespace mmq = basalt::prefill::mmq;
using basalt::prefill::Gemm;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Dev {
    void* p = nullptr;
    explicit Dev(size_t n) { ck(cudaMalloc(&p, std::max<size_t>(n, 256)), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    template <typename T> T* as() const { return (T*) p; }
};

struct Geo {
    ggml_type t;
    int64_t n, k;   // W [n][k]
    const char* name;
    int layers;     // how many of the model's 48 layers run it (benchmark weighting; one type per geometry counted)
};
const Geo kGeos[] = {
    {GGML_TYPE_Q6_K, 10240, 2560, "attn_qkv Q6_K", 36},   {GGML_TYPE_Q5_K, 10240, 2560, "attn_qkv Q5_K", 0},
    {GGML_TYPE_IQ4_XS, 10240, 2560, "attn_qkv IQ4_XS", 0}, {GGML_TYPE_Q6_K, 6144, 2560, "attn_gate Q6_K", 36},
    {GGML_TYPE_Q4_K, 6144, 2560, "attn_gate Q4_K", 0},     {GGML_TYPE_IQ4_XS, 6144, 2560, "attn_gate IQ4_XS", 0},
    {GGML_TYPE_Q6_K, 2560, 6144, "ssm_out Q6_K", 36},      {GGML_TYPE_Q5_K, 2560, 6144, "ssm_out Q5_K", 0},
    {GGML_TYPE_Q4_K, 2560, 6144, "attn_output Q4_K", 12},  {GGML_TYPE_IQ4_XS, 12288, 2560, "attn_q IQ4_XS", 0},
    {GGML_TYPE_Q6_K, 12288, 2560, "attn_q Q6_K", 12},      {GGML_TYPE_Q6_K, 512, 2560, "attn_k Q6_K", 12},
    {GGML_TYPE_Q5_K, 512, 2560, "attn_v Q5_K", 12},        {GGML_TYPE_Q4_K, 640, 2560, "shexp gate Q4_K", 48},
    {GGML_TYPE_IQ4_XS, 640, 2560, "shexp up IQ4_XS", 48},  {GGML_TYPE_Q6_K, 640, 2560, "shexp gate Q6_K", 0},
    {GGML_TYPE_IQ4_NL, 2560, 640, "shexp down IQ4_NL", 48},
};
const fp8::Scale kScales[] = {fp8::Scale::Row, fp8::Scale::Block32, fp8::Scale::Tensor};

// random blocks with sane fp16 scales, sized so the weights are a few hundredths and centred on zero
void fill_matrix(uint8_t* m, ggml_type t, int64_t rows, int64_t cols, std::mt19937& rng) {
    const size_t bs = ggml_type_size(t), nb = (size_t) rows * (size_t) (cols / ggml_blck_size(t));
    std::uniform_int_distribution<int> byte(0, 255);
    for (size_t i = 0; i < nb * bs; ++i) m[i] = (uint8_t) byte(rng);
    float lo = 0.0008f, mid = 0.0f;
    switch (t) {
        case GGML_TYPE_Q4_K: lo = 0.0002f; mid = 7.5f; break;
        case GGML_TYPE_Q5_K: lo = 0.0001f; mid = 15.5f; break;
        case GGML_TYPE_IQ4_XS: lo = 0.0002f; break;
        case GGML_TYPE_IQ4_NL: lo = 0.0004f; break;
        case GGML_TYPE_Q8_0: lo = 0.0002f; break;
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

// T rows of k values: `uniq` distinct random rows repeated (rows [0, uniq) are distinct), outliers 1 in 200 at 12x
std::vector<float> make_x(int64_t T, int64_t k, int64_t uniq, bool outliers, std::mt19937& rng) {
    std::vector<float> x((size_t) T * k);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> pick(0, 199);
    uniq = std::min(uniq, T);
    for (size_t i = 0; i < (size_t) uniq * k; ++i) x[i] = nd(rng) * (outliers && pick(rng) == 0 ? 12.0f : 1.0f);
    for (int64_t r = uniq; r < T; ++r) std::memcpy(&x[(size_t) r * k], &x[(size_t) (r % uniq) * k], (size_t) k * 4);
    return x;
}

std::vector<uint16_t> to_f16(const std::vector<float>& x) {
    std::vector<uint16_t> h(x.size());
    const int nth = std::max(1, std::min(16, (int) std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (int w = 0; w < nth; ++w)
        th.emplace_back([&, w] {
            for (size_t i = (size_t) w; i < x.size(); i += (size_t) nth) h[i] = ggml_fp32_to_fp16(x[i]);
        });
    for (auto& t : th) t.join();
    return h;
}

// ---- the CPU E4M3 encoder: the 127 non-negative finite codes in order, round to nearest (ties to the even code),
// saturated at 448
struct E4M3 {
    float v[127];
    E4M3() {
        for (int c = 0; c < 127; ++c) {
            const int e = c >> 3, m = c & 7;
            v[c] = e == 0 ? std::ldexp((float) m, -9) : std::ldexp(1.0f + (float) m / 8.0f, e - 7);
        }
    }
    uint8_t enc(float f) const {
        if (std::isnan(f)) return 0x7f;
        const uint8_t sign = std::signbit(f) ? 0x80 : 0;
        const float a = std::fabs(f);
        if (a >= v[126]) return sign | 126;
        const int hi = (int) (std::upper_bound(v, v + 127, a) - v);   // v[hi - 1] <= a < v[hi]
        const int lo = hi - 1;
        const float dl = a - v[lo], dh = v[hi] - a;
        const int c = dl < dh ? lo : dh < dl ? hi : ((lo & 1) == 0 ? lo : hi);
        return sign | (uint8_t) c;
    }
    float dec(uint8_t b) const {
        const float a = (b & 0x7f) == 0x7f ? NAN : v[b & 0x7f];
        return (b & 0x80) ? -a : a;
    }
};
const E4M3 kE4M3;

int pow2_exp_ref(float amax) {
    if (!(amax > 0.0f) || !std::isfinite(amax)) return 0;
    int e = 0;
    const float m = std::frexp(amax * (1.0f / 448.0f), &e);
    return m == 0.5f ? e - 1 : e;
}
int64_t sf_offset_ref(int64_t r, int64_t b, int64_t blocks_pad) {
    return ((r / 128) * (blocks_pad / 4) + b / 4) * 512 + (r % 32) * 16 + (r % 128) / 32 * 4 + (b % 4);
}

// The CPU's E4M3 rows and scales for `s` (Row: FP32 per row; Block32: the tiled UE8M0 bytes; Tensor: one FP32).
void e4m3_ref(const std::vector<uint16_t>& x16, int64_t rows, int64_t cols, fp8::Scale s, std::vector<uint8_t>& q,
              std::vector<uint8_t>& sc) {
    q.assign((size_t) rows * cols, 0);
    sc.assign(fp8::scale_bytes(s, rows, cols), 0);
    float tmax = 0.0f;
    if (s == fp8::Scale::Tensor)
        for (uint16_t h : x16) tmax = std::max(tmax, std::fabs(ggml_fp16_to_fp32(h)));
    const int et = pow2_exp_ref(tmax);
    if (s == fp8::Scale::Tensor) {
        const float f = std::ldexp(1.0f, et);
        std::memcpy(sc.data(), &f, 4);
    }
    const int64_t bpad = (cols / 32 + 3) / 4 * 4;
    for (int64_t r = 0; r < rows; ++r) {
        const uint16_t* xr = &x16[(size_t) r * cols];
        if (s == fp8::Scale::Block32) {
            for (int64_t b = 0; b < cols / 32; ++b) {
                float a = 0.0f;
                for (int j = 0; j < 32; ++j) a = std::max(a, std::fabs(ggml_fp16_to_fp32(xr[b * 32 + j])));
                const int e = pow2_exp_ref(a);
                sc[(size_t) sf_offset_ref(r, b, bpad)] = (uint8_t) (e + 127);
                for (int j = 0; j < 32; ++j)
                    q[(size_t) r * cols + b * 32 + j] = kE4M3.enc(std::ldexp(ggml_fp16_to_fp32(xr[b * 32 + j]), -e));
            }
            continue;
        }
        int e = et;
        if (s == fp8::Scale::Row) {
            float a = 0.0f;
            for (int64_t j = 0; j < cols; ++j) a = std::max(a, std::fabs(ggml_fp16_to_fp32(xr[j])));
            e = pow2_exp_ref(a);
            const float f = std::ldexp(1.0f, e);
            std::memcpy(&sc[(size_t) r * 4], &f, 4);
        }
        for (int64_t j = 0; j < cols; ++j) q[(size_t) r * cols + j] = kE4M3.enc(std::ldexp(ggml_fp16_to_fp32(xr[j]), -e));
    }
}

template <typename F> double time_ms(F f, cudaStream_t s, int reps) {
    cudaEvent_t a, b;
    ck(cudaEventCreate(&a), "event");
    ck(cudaEventCreate(&b), "event");
    f();
    ck(cudaStreamSynchronize(s), "warm");
    std::vector<double> v;
    for (int i = 0; i < reps; ++i) {
        ck(cudaEventRecord(a, s), "record");
        f();
        ck(cudaEventRecord(b, s), "record");
        ck(cudaEventSynchronize(b), "sync");
        float ms = 0;
        ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
        v.push_back(ms);
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct Err {
    double rms = 0, worst = 0;
};
// rows `rows` of y (n per row) against ref (rows.size() x n)
Err compare(const std::vector<float>& y, const std::vector<double>& ref, const std::vector<int64_t>& rows, int64_t n) {
    Err e;
    double e2 = 0, r2 = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        double rm = 0, em = 0;
        for (int64_t c = 0; c < n; ++c) {
            const double r = ref[i * n + c], d = (double) y[(size_t) rows[i] * n + c] - r;
            e2 += d * d; r2 += r * r;
            rm = std::max(rm, std::fabs(r)); em = std::max(em, std::fabs(d));
        }
        if (rm > 0) e.worst = std::max(e.worst, em / rm);
    }
    e.rms = r2 > 0 ? std::sqrt(e2 / r2) : 0;
    return e;
}
// relative RMS of a - b over all values
double rel_rms(const std::vector<float>& a, const std::vector<float>& b) {
    double e2 = 0, r2 = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = (double) a[i] - b[i];
        e2 += d * d; r2 += (double) b[i] * b[i];
    }
    return r2 > 0 ? std::sqrt(e2 / r2) : 0;
}

template <typename T> std::vector<T> download(const void* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}

// The E4M3 matrix of W for scale `s` in `w8` / `ws`: dequantized to FP16 in `w16` first (the Tensor scale in `ts`).
void weights_e4m3(const Geo& g, const void* w, fp8::Scale s, uint16_t* w16, uint8_t* w8, void* ws, float* ts,
                  cudaStream_t st) {
    basalt::kernels::dequant_f16(g.t, w, 0, g.n, g.k, w16, st);
    if (s == fp8::Scale::Tensor) fp8::tensor_scale(w16, g.n * g.k, ts, st);
    fp8::to_e4m3(w16, g.n, g.k, g.k, w8, s == fp8::Scale::Tensor ? (void*) ts : ws, 0, g.n, s, ts, st);
}
void acts_e4m3(const uint16_t* x16, int64_t T, int64_t k, fp8::Scale s, uint8_t* x8, void* xs, float* ts,
               cudaStream_t st) {
    if (s == fp8::Scale::Tensor) fp8::tensor_scale(x16, T * k, ts, st);
    fp8::to_e4m3(x16, T, k, k, x8, s == fp8::Scale::Tensor ? (void*) ts : xs, 0, T, s, ts, st);
}

constexpr size_t kWs = 32u << 20;

// returns the failures
int check_geo(const Geo& g, bool outliers, cudaStream_t s, Gemm& gemm, fp8::Lt& lt, void* ws, const bool* offered) {
    const int64_t T = 512, small = 64;
    std::mt19937 rng(7 + (unsigned) g.t * 131 + (unsigned) g.n);
    const size_t wb = ggml_row_size(g.t, g.k) * (size_t) g.n;
    std::vector<uint8_t> w(wb);
    fill_matrix(w.data(), g.t, g.n, g.k, rng);
    std::vector<float> x = make_x(T, g.k, T, outliers, rng);
    std::fill(x.begin() + 5 * g.k, x.begin() + 6 * g.k, 0.0f);   // an all-zero row
    const std::vector<uint16_t> x16 = to_f16(x);
    Dev w_dev(wb), x16_dev(x16.size() * 2), y_dev((size_t) T * g.n * 4), w16((size_t) g.n * g.k * 2),
        w8((size_t) g.n * g.k), wsc(fp8::scale_bytes(fp8::Scale::Block32, g.n, g.k) + 256), x8((size_t) T * g.k),
        xsc(fp8::scale_bytes(fp8::Scale::Block32, T, g.k) + 256), tsw(256), tsx(256), buf(Gemm::act8_bytes(T, g.k));
    ck(cudaMemcpy(w_dev.p, w.data(), wb, cudaMemcpyHostToDevice), "w");
    ck(cudaMemcpy(x16_dev.p, x16.data(), x16.size() * 2, cudaMemcpyHostToDevice), "x16");
    // a pageable upload can still be landing when cudaMemcpy returns; the work below runs on a non-blocking stream
    ck(cudaDeviceSynchronize(), "upload");

    int fail = 0;
    // the conversion against the CPU encoder, bit for bit (rows of X; W's FP16 rows through the same kernel)
    for (fp8::Scale sm : kScales) {
        std::vector<uint8_t> q, sc;
        e4m3_ref(x16, T, g.k, sm, q, sc);
        ck(cudaMemset(xsc.p, 0, fp8::scale_bytes(fp8::Scale::Block32, T, g.k)), "xsc");
        acts_e4m3(x16_dev.as<uint16_t>(), T, g.k, sm, x8.as<uint8_t>(), xsc.p, tsx.as<float>(), s);
        ck(cudaStreamSynchronize(s), "e4m3");
        const auto gq = download<uint8_t>(x8.p, q.size());
        const auto gs = download<uint8_t>(sm == fp8::Scale::Tensor ? tsx.p : xsc.p, sc.size());
        if (gq != q || gs != sc) {
            size_t bad = 0;
            for (size_t i = 0; i < q.size(); ++i) bad += gq[i] != q[i];
            std::printf("%-18s E4M3 %s conversion differs from the CPU encoder: %zu bytes, scales %s FAIL\n", g.name,
                        fp8::scale_name(sm), bad, gs == sc ? "equal" : "differ");
            ++fail;
        }
    }

    // the reference on sampled rows (the zero row among them)
    std::vector<int64_t> rows = {0, 1, 5, 63, 64, 200, 377, 511};
    std::vector<float> wf((size_t) g.n * g.k);
    const auto* tt = ggml_get_type_traits(g.t);
    const size_t rb = ggml_row_size(g.t, g.k);
    for (int64_t r = 0; r < g.n; ++r) tt->to_float(w.data() + (size_t) r * rb, wf.data() + (size_t) r * g.k, g.k);
    std::vector<double> ref(rows.size() * (size_t) g.n);
    {
        std::vector<std::thread> th;
        const int nth = std::max(1, std::min(16, (int) std::thread::hardware_concurrency()));
        for (int wk = 0; wk < nth; ++wk)
            th.emplace_back([&, wk] {
                for (size_t i = 0; i < rows.size(); ++i)
                    for (int64_t c = wk; c < g.n; c += nth) {
                        double acc = 0;
                        const float* xr = &x[(size_t) rows[i] * g.k];
                        const float* wr = &wf[(size_t) c * g.k];
                        for (int64_t j = 0; j < g.k; ++j) acc += (double) xr[j] * (double) wr[j];
                        ref[i * g.n + c] = acc;
                    }
            });
        for (auto& t : th) t.join();
    }
    gemm.native(x16_dev.as<uint16_t>(), g.t, w_dev.p, y_dev.as<float>(), T, g.n, g.k);
    ck(cudaStreamSynchronize(s), "f16");
    const std::vector<float> y_f16 = download<float>(y_dev.p, (size_t) T * g.n);
    const Err ef = compare(y_f16, ref, rows, g.n);
    const double lim_rms = 6e-2, lim_worst = 0.25;
    auto zero_row = [&](const std::vector<float>& y) {
        for (int64_t c = 0; c < g.n; ++c)
            if (y[(size_t) 5 * g.n + c] != 0.0f) return false;
        return true;
    };

    // every scale granularity cuBLASLt offers, through the Lt wrapper
    std::string line;
    for (int i = 0; i < 3; ++i) {
        const fp8::Scale sm = kScales[i];
        if (!offered[i]) continue;
        void* wsp = sm == fp8::Scale::Tensor ? tsw.p : wsc.p;
        void* xsp = sm == fp8::Scale::Tensor ? tsx.p : xsc.p;
        weights_e4m3(g, w_dev.p, sm, w16.as<uint16_t>(), w8.as<uint8_t>(), wsc.p, tsw.as<float>(), s);
        acts_e4m3(x16_dev.as<uint16_t>(), T, g.k, sm, x8.as<uint8_t>(), xsc.p, tsx.as<float>(), s);
        if (!lt.run(sm, w8.as<uint8_t>(), wsp, x8.as<uint8_t>(), xsp, y_dev.as<float>(), T, g.n, g.k, g.n, ws, kWs, s)) {
            line += std::string(" | ") + fp8::scale_name(sm) + ": no algorithm at T 512";
            ++fail;
            continue;
        }
        ck(cudaStreamSynchronize(s), "fp8");
        const std::vector<float> y = download<float>(y_dev.p, (size_t) T * g.n);
        const Err e = compare(y, ref, rows, g.n);
        const bool bad = e.rms > lim_rms || e.worst > lim_worst || !zero_row(y);
        fail += bad;
        char tmp[160];
        std::snprintf(tmp, sizeof tmp, " | %s %.2e / %.2e (vs FP16 %.2e)%s", fp8::scale_name(sm), e.rms, e.worst,
                      rel_rms(y, y_f16), bad ? " FAIL" : "");
        line += tmp;
    }

    // the prompt path's product (Gemm::act8 + Gemm::native8), and its width invariance
    const Gemm::Rows8 xr = gemm.act8(x16_dev.as<uint16_t>(), T, g.k, buf.p);
    if (!gemm.native8(xr, g.t, w_dev.p, y_dev.as<float>(), g.n)) {
        std::printf("%-18s Gemm::native8 did not run FAIL\n", g.name);
        return fail + 1;
    }
    ck(cudaStreamSynchronize(s), "native8");
    const std::vector<float> y8 = download<float>(y_dev.p, (size_t) T * g.n);
    const std::vector<uint8_t> q_full = download<uint8_t>(xr.q, (size_t) T * g.k);
    const Err e8 = compare(y8, ref, rows, g.n);
    if (e8.rms > lim_rms || e8.worst > lim_worst || !zero_row(y8)) ++fail;

    std::vector<uint16_t> xs((size_t) small * g.k);
    for (int64_t r = 0; r < small; ++r)
        std::memcpy(&xs[(size_t) ((r + 7) % small) * g.k], &x16[(size_t) r * g.k], (size_t) g.k * 2);
    ck(cudaMemcpy(x16_dev.p, xs.data(), xs.size() * 2, cudaMemcpyHostToDevice), "xs");
    ck(cudaDeviceSynchronize(), "upload xs");
    const Gemm::Rows8 xsr = gemm.act8(x16_dev.as<uint16_t>(), small, g.k, buf.p);
    if (!gemm.native8(xsr, g.t, w_dev.p, y_dev.as<float>(), g.n)) {
        std::printf("%-18s Gemm::native8 did not run at 64 rows FAIL\n", g.name);
        return fail + 1;
    }
    ck(cudaStreamSynchronize(s), "native8 small");
    const std::vector<float> y_small = download<float>(y_dev.p, (size_t) small * g.n);
    const std::vector<uint8_t> q_small = download<uint8_t>(xsr.q, (size_t) small * g.k);
    long qdiff = 0;
    for (int64_t r = 0; r < small; ++r)
        qdiff += std::memcmp(&q_full[(size_t) r * g.k], &q_small[(size_t) ((r + 7) % small) * g.k], (size_t) g.k) != 0;
    double wmax = 0, rmax = 0;
    long nbits = 0;
    for (int64_t r = 0; r < small; ++r)
        for (int64_t c = 0; c < g.n; ++c) {
            const float a = y8[(size_t) r * g.n + c], b = y_small[(size_t) ((r + 7) % small) * g.n + c];
            if (std::memcmp(&a, &b, 4) != 0) ++nbits;
            wmax = std::max(wmax, (double) std::fabs(a - b));
            rmax = std::max(rmax, (double) std::fabs(a));
        }
    const double wrel = rmax > 0 ? wmax / rmax : 0;
    if (qdiff != 0 || wrel > 1e-4) ++fail;
    std::printf("%-18s %-8s FP16 %.2e / %.2e%s | path %.2e / %.2e (vs FP16 %.2e) | width: %ld E4M3 rows differ, %ld "
                "values differ, max %.2e of the max |y| %s\n",
                g.name, outliers ? "outliers" : "plain", ef.rms, ef.worst, line.c_str(), e8.rms, e8.worst,
                rel_rms(y8, y_f16), qdiff, nbits, wrel, fail ? "FAIL" : "ok");
    return fail;
}

struct BenchSum {
    double f16 = 0, path = 0, mm[3] = {}, best[3] = {}, conv[3] = {};
};

void bench_geo(const Geo& g, int64_t T, cudaStream_t s, Gemm& gemm, fp8::Lt& lt, void* ws, const bool* offered,
               mmq::Context& mctx, Dev& x_dev, Dev& x16_dev, Dev& y_dev, Dev& x8, Dev& xsc, Dev& tsx, Dev& abuf,
               Dev& xq, Dev& bounds, Dev& ident, BenchSum& sum) {
    std::mt19937 rng(3 + (unsigned) g.t);
    const size_t wb = ggml_row_size(g.t, g.k) * (size_t) g.n;
    std::vector<uint8_t> w(wb);
    fill_matrix(w.data(), g.t, g.n, g.k, rng);
    Dev w_dev(wb), w16((size_t) g.n * g.k * 2), w8((size_t) g.n * g.k),
        wsc(fp8::scale_bytes(fp8::Scale::Block32, g.n, g.k) + 256), tsw(256);
    ck(cudaMemcpy(w_dev.p, w.data(), wb, cudaMemcpyHostToDevice), "w");
    ck(cudaDeviceSynchronize(), "upload w");
    const int reps = 7;
    const double fl = 2.0 * (double) T * g.n * g.k;
    auto tf = [&](double ms) { return fl / (ms * 1e9); };
    const double t_f16 =
        time_ms([&] { gemm.native(x16_dev.as<uint16_t>(), g.t, w_dev.p, y_dev.as<float>(), T, g.n, g.k); }, s, reps);
    const double t_dq = time_ms([&] { basalt::kernels::dequant_f16(g.t, w_dev.p, 0, g.n, g.k, gemm.scratch(), s); }, s,
                                reps);
    std::printf("%-18s T %6lld | FP16 %8.1f us (dequant %6.1f) %6.1f TFLOP/s\n", g.name, (long long) T, 1e3 * t_f16,
                1e3 * t_dq, tf(t_f16));
    for (int i = 0; i < 3; ++i) {
        const fp8::Scale sm = kScales[i];
        if (!offered[i]) continue;
        void* wsp = sm == fp8::Scale::Tensor ? tsw.p : wsc.p;
        void* xsp = sm == fp8::Scale::Tensor ? tsx.p : xsc.p;
        const double t_w = time_ms([&] {
            weights_e4m3(g, w_dev.p, sm, w16.as<uint16_t>(), w8.as<uint8_t>(), wsc.p, tsw.as<float>(), s);
        }, s, reps);
        const double t_x = time_ms([&] {
            acts_e4m3(x16_dev.as<uint16_t>(), T, g.k, sm, x8.as<uint8_t>(), xsc.p, tsx.as<float>(), s);
        }, s, reps);
        const int na = lt.algos(sm, T, g.n, g.k, g.n, kWs, wsp, xsp);
        if (na == 0) {
            std::printf("    %-13s no algorithm at T %lld\n", fp8::scale_name(sm), (long long) T);
            continue;
        }
        auto run = [&](int a, bool fast) {
            if (!lt.run(sm, w8.as<uint8_t>(), wsp, x8.as<uint8_t>(), xsp, y_dev.as<float>(), T, g.n, g.k, g.n, ws, kWs,
                        s, fast, a))
                throw std::runtime_error("cublasLtMatmul failed");
        };
        const double t0 = time_ms([&] { run(0, false); }, s, reps);
        double tb = t0;
        int ib = 0;
        for (int a = 1; a < na; ++a) {
            const double t = time_ms([&] { run(a, false); }, s, 3);
            if (t < tb) { tb = t; ib = a; }
        }
        double tfast = -1;
        if (lt.algos(sm, T, g.n, g.k, g.n, kWs, wsp, xsp, true) > 0) tfast = time_ms([&] {
            if (!lt.run(sm, w8.as<uint8_t>(), wsp, x8.as<uint8_t>(), xsp, y_dev.as<float>(), T, g.n, g.k, g.n, ws, kWs,
                        s, true, 0))
                throw std::runtime_error("cublasLtMatmul (fast accumulation) failed");
        }, s, reps);
        std::printf("    %-13s W->E4M3 %6.1f us, X->E4M3 %6.1f us | product %8.1f us %6.1f TFLOP/s (%.2fx FP16) | best of "
                    "%2d: #%d %8.1f us %6.1f TFLOP/s | fast accumulation %8.1f us\n",
                    fp8::scale_name(sm), 1e3 * t_w, 1e3 * t_x, 1e3 * t0, tf(t0), t_f16 / t0, na, ib, 1e3 * tb, tf(tb),
                    1e3 * tfast);
        sum.mm[i] += g.layers * t0;
        sum.best[i] += g.layers * tb;
        sum.conv[i] += g.layers * (t_w + t_x);
    }
    if (gemm.fp8_ready()) {
        const double t_path = time_ms([&] {
            const Gemm::Rows8 xr = gemm.act8(x16_dev.as<uint16_t>(), T, g.k, abuf.p);
            if (!gemm.native8(xr, g.t, w_dev.p, y_dev.as<float>(), g.n)) throw std::runtime_error("native8 did not run");
        }, s, reps);
        std::printf("    prompt path   act8 + native8 %8.1f us %6.1f TFLOP/s (%.2fx FP16)\n", 1e3 * t_path, tf(t_path),
                    t_f16 / t_path);
        sum.path += g.layers * t_path;
    }
    sum.f16 += g.layers * t_f16;
    // llama.cpp's MMQ on Q8_0 weights of the same shape (product only; X quantized to q8_1 beforehand)
    if (mmq::fits(GGML_TYPE_Q8_0, g.n)) {
        const size_t qb = ggml_row_size(GGML_TYPE_Q8_0, g.k) * (size_t) g.n;
        std::vector<uint8_t> wq(qb);
        fill_matrix(wq.data(), GGML_TYPE_Q8_0, g.n, g.k, rng);
        Dev wq_dev(qb);
        ck(cudaMemcpy(wq_dev.p, wq.data(), qb, cudaMemcpyHostToDevice), "wq");
        const int32_t b[2] = {0, (int32_t) T};
        ck(cudaMemcpy(bounds.p, b, sizeof b, cudaMemcpyHostToDevice), "bounds");
        ck(cudaDeviceSynchronize(), "upload wq");
        mmq::quantize(x_dev.as<float>(), nullptr, xq.p, GGML_TYPE_Q8_0, g.k, g.k, T, s);
        mmq::Product p;
        p.w = wq_dev.p; p.type = GGML_TYPE_Q8_0; p.w_rows = g.n; p.w_cols = g.k; p.expert_bytes = qb;
        p.n = 1; p.xq = xq.p; p.bounds = bounds.as<int32_t>(); p.ids = ident.as<int32_t>(); p.total_rows = T;
        p.max_rows = T; p.dst = y_dev.as<float>(); p.ld_dst = g.n;
        const double t_mmq = time_ms([&] { mctx.run(p, s); }, s, reps);
        const double t_q = time_ms([&] { mmq::quantize(x_dev.as<float>(), nullptr, xq.p, GGML_TYPE_Q8_0, g.k, g.k, T, s); },
                                   s, reps);
        std::printf("    MMQ Q8_0      product %8.1f us %6.1f TFLOP/s (%.2fx FP16), quantize %6.1f us\n", 1e3 * t_mmq,
                    tf(t_mmq), t_f16 / t_mmq, 1e3 * t_q);
    }
}
}  // namespace

int main(int argc, char** argv) {
    try {
        int n_dev = 0;
        if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
            std::printf("no CUDA device - skipped\n");
            return 77;
        }
        bool bench = false, vram = false;
        int64_t bench_t = 32768;
        for (int i = 1; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--vram")) vram = true;
            if (!std::strcmp(argv[i], "--bench")) {
                bench = true;
                if (i + 1 < argc && argv[i + 1][0] != '-') bench_t = std::atoll(argv[++i]);
            }
        }
        cudaStream_t s;
        ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        Gemm gemm;
        std::string err;
        if (!gemm.init(s, 32ll << 20, err)) throw std::runtime_error(err);
        int64_t kmax = 0, nmax = 0;
        for (const Geo& g : kGeos) {
            kmax = std::max(kmax, g.k);
            nmax = std::max(nmax, g.n);
        }
        if (vram) {
            // every geometry at a short and a long chunk: FP16 first, then cuBLASLt set up and the same shapes in FP8
            const int64_t ts[4] = {512, 2048, 8192, 32768};
            Dev x16((size_t) ts[3] * kmax * 2), y((size_t) ts[3] * nmax * 4), abuf(Gemm::act8_bytes(ts[3], kmax));
            ck(cudaMemset(x16.p, 0, (size_t) ts[3] * kmax * 2), "x16");
            std::vector<Dev*> ws_;
            std::mt19937 rng(5);
            for (const Geo& g : kGeos) {
                const size_t wb = ggml_row_size(g.t, g.k) * (size_t) g.n;
                std::vector<uint8_t> w(wb);
                fill_matrix(w.data(), g.t, g.n, g.k, rng);
                ws_.push_back(new Dev(wb));
                ck(cudaMemcpy(ws_.back()->p, w.data(), wb, cudaMemcpyHostToDevice), "w");
            }
            ck(cudaDeviceSynchronize(), "upload");
            auto free_mib = [] {
                size_t f = 0, t = 0;
                ck(cudaMemGetInfo(&f, &t), "cudaMemGetInfo");
                return (double) f / (1 << 20);
            };
            for (int64_t T : ts)
                for (size_t i = 0; i < std::size(kGeos); ++i)
                    gemm.native(x16.as<uint16_t>(), kGeos[i].t, ws_[i]->p, y.as<float>(), T, kGeos[i].n, kGeos[i].k);
            ck(cudaStreamSynchronize(s), "fp16");
            const double f0 = free_mib();
            if (!gemm.fp8_ready()) throw std::runtime_error("no FP8 products here");
            for (int64_t T : ts)
                for (size_t i = 0; i < std::size(kGeos); ++i) {
                    const Gemm::Rows8 xr = gemm.act8(x16.as<uint16_t>(), T, kGeos[i].k, abuf.p);
                    if (!gemm.native8(xr, kGeos[i].t, ws_[i]->p, y.as<float>(), kGeos[i].n))
                        throw std::runtime_error("native8 did not run");
                }
            ck(cudaStreamSynchronize(s), "fp8");
            const double f1 = free_mib();
            for (Dev* d : ws_) delete d;
            std::printf("VRAM: the FP8 products' first use took %.1f MiB (free %.1f -> %.1f MiB)\n", f0 - f1, f0, f1);
            return 0;
        }
        fp8::Lt lt;
        if (!lt.init(err)) throw std::runtime_error(err);
        Dev ws(kWs);
        // which scale granularities cuBLASLt offers on this device (a prompt-sized shape)
        bool offered[3] = {};
        for (int i = 0; i < 3; ++i) {
            offered[i] = lt.algos(kScales[i], 4096, 2560, 2560, 2560, kWs, ws.p, ws.p) > 0;
            std::printf("E4M3 %-13s scales: %s\n", fp8::scale_name(kScales[i]),
                        offered[i] ? "offered by cuBLASLt" : "NOT offered by cuBLASLt");
        }
        if (!gemm.fp8_ready()) {
            std::printf("prefill_fp8_dense_test: the prompt path's FP8 products are not available here\n");
            return 1;
        }
        if (bench) {
            mmq::Context mctx;
            std::mt19937 rng(1);
            const std::vector<float> x = make_x(bench_t, kmax, 2048, false, rng);
            Dev x_dev(x.size() * 4), x16_dev(x.size() * 2), y_dev((size_t) bench_t * nmax * 4), x8((size_t) bench_t * kmax),
                xsc(fp8::scale_bytes(fp8::Scale::Block32, bench_t, kmax) + 256), tsx(256),
                abuf(Gemm::act8_bytes(bench_t, kmax)), xq(mmq::q8_bytes(bench_t, kmax)), bounds(8),
                ident((size_t) bench_t * 4);
            mmq::iota(ident.as<int32_t>(), bench_t, nullptr);
            ck(cudaMemcpy(x_dev.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
            const std::vector<uint16_t> x16 = to_f16(x);
            ck(cudaMemcpy(x16_dev.p, x16.data(), x16.size() * 2, cudaMemcpyHostToDevice), "x16");
            ck(cudaDeviceSynchronize(), "setup");
            BenchSum sum;
            for (const Geo& g : kGeos) bench_geo(g, bench_t, s, gemm, lt, ws.p, offered, mctx, x_dev, x16_dev, y_dev, x8,
                                                 xsc, tsx, abuf, xq, bounds, ident, sum);
            std::printf("per chunk of %lld tokens, 48 layers (one type per geometry): FP16 %.1f ms | prompt path FP8 %.1f "
                        "ms (%.2fx)\n",
                        (long long) bench_t, sum.f16, sum.path, sum.path > 0 ? sum.f16 / sum.path : 0.0);
            for (int i = 0; i < 3; ++i)
                if (offered[i])
                    std::printf("    %-13s products %.1f ms (best algorithms %.1f ms) + conversions %.1f ms\n",
                                fp8::scale_name(kScales[i]), sum.mm[i], sum.best[i], sum.conv[i]);
            return 0;
        }
        int fails = 0;
        for (const Geo& g : kGeos)
            for (int o = 0; o < 2; ++o) fails += check_geo(g, o != 0, s, gemm, lt, ws.p, offered);
        std::printf("prefill_fp8_dense_test: %d failure(s)\n", fails);
        return fails ? 1 : 0;
    } catch (const std::exception& e) {
        std::printf("prefill_fp8_dense_test: %s\n", e.what());
        return 1;
    }
}
