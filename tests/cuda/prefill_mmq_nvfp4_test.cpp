// prefill_mmq_nvfp4_test - the prompt path's NVFP4 experts through MMQ (moe_mmq.hpp): the NVFP4 tile loader with the
// per-16-scale int8 dot, the scale tail gathered with each expert and applied in mmq::swiglu, at the real geometry
// (gate/up 1280 x 2560, down 2560 x 640) on synthetic NVFP4 experts (sub-block scales over 13 binades, UE4M3
// subnormals included; tails {s_gate, s_up, s_down} in 0.004..0.03), in expert_mmq.cpp's sequence (16-expert groups,
// q8_1 of the activations, gate/up, SwiGLU, q8_1 of H, down).  Checked against three CPU references:
//   - `same`: double-precision math on MMQ's own roundings (x and H rounded to int8 per 32 values the way the
//     quantizer does).  The gate/up product (before the tail and SwiGLU) isolates the loader and the dot: only fp32
//     accumulation order differs.  Tolerance: RMS 1e-5, worst row 1e-4.  After the down product an H value whose
//     x/d lands within fp32 rounding of a .5 tie takes the neighbouring int8 code (one step is 1/127 of its 32-value
//     block's max), so the tolerance there is RMS 2e-4, worst row 1e-2.
//   - `double`: the unrounded activations.  MMQ's error here is the int8 activation rounding every MMQ format has
//     (the decode kernels measure 1.2e-2 on the same rounding).  Tolerance: RMS 2.5e-2.
//   - `fp16`: the FP16 fallback this path replaces (weights times their tail rounded to fp16, activations and H in
//     fp16, fp32 accumulation).  Tolerance: MMQ vs fp16 RMS 2e-2.
// The per-expert gather and the grouped gather must give bitwise equal outputs.  An all-zero token stays exactly 0.
// Then a microbench line: one layer at a 2048-token chunk (512 experts, top 10) - gathers plus products - in us and
// TFLOP/s.  --no-bench.  Exit 77 without a CUDA device.
#include "basalt/prefill/moe_mmq.hpp"

#include "basalt/ggml/ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
namespace mmq = basalt::prefill::mmq;

constexpr int N = 2560, FF = 640, K = 10, GROUP = 16;
constexpr size_t GU_ROW = (size_t) N / 64 * 36, D_ROW = (size_t) FF / 64 * 36;
constexpr size_t UP_OFF = GU_ROW * FF, DOWN_OFF = 2 * UP_OFF, TAIL_OFF = DOWN_OFF + D_ROW * N, BYTES = TAIL_OFF + 16;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Dev {
    void* p = nullptr;
    explicit Dev(size_t n) { ck(cudaMalloc(&p, n), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    template <typename T> T* as() const { return (T*) p; }
};

template <typename F> void parallel(int n, F f) {
    const int nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> th;
    for (int t = 0; t < nt; ++t)
        th.emplace_back([&, t] { for (int i = t; i < n; i += nt) f(i); });
    for (auto& x : th) x.join();
}

// one NVFP4 expert blob [gate | up | down | tail]: each 16 values N(0, 1) times 2^U(-10, 3), so the UE4M3 sub-block
// scales span normals and subnormals
std::vector<uint8_t> make_blob(std::mt19937& rng, float tail[3]) {
    const auto* tr = ggml_get_type_traits(GGML_TYPE_NVFP4);
    std::vector<uint8_t> b(BYTES);
    std::vector<float> row(N);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ex(-10.0f, 3.0f), ts(0.004f, 0.03f);
    auto fill = [&](uint8_t* dst, int rows, int cols, size_t row_bytes) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; c += 16) {
                const float m = std::exp2(ex(rng));
                for (int j = 0; j < 16; ++j) row[(size_t) (c + j)] = nd(rng) * m;
            }
            tr->from_float_ref(row.data(), dst + (size_t) r * row_bytes, cols);
        }
    };
    fill(b.data(), 2 * FF, N, GU_ROW);
    fill(b.data() + DOWN_OFF, N, FF, D_ROW);
    for (int i = 0; i < 3; ++i) tail[i] = ts(rng);
    const float t4[4] = {tail[0], tail[1], tail[2], 0.0f};
    std::memcpy(b.data() + TAIL_OFF, t4, 16);
    return b;
}

// MMQ's activation rounding (quantize.cu): per 32 values d = 1 / (127 / amax), q = round(x * 127 / amax)
void q8_round(const float* x, float* y, int n) {
    for (int b = 0; b < n; b += 32) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[b + j]));
        const float d_inv = 127.0f / amax, d = 1.0f / d_inv;
        for (int j = 0; j < 32; ++j) y[b + j] = amax == 0.0f ? 0.0f : std::round(x[b + j] * d_inv) * d;
    }
}
float f16(float v) { return ggml_fp16_to_fp32(ggml_fp32_to_fp16(v)); }

struct Err {
    double rms = 0, worst = 0;
};
Err compare(const std::vector<float>& a, const std::vector<float>& ref, size_t rows) {
    double e2 = 0, r2 = 0;
    Err out;
    for (size_t i = 0; i < rows; ++i) {
        double me = 0, mr = 0;
        for (int o = 0; o < N; ++o) {
            const float va = a[i * N + o], vr = ref[i * N + o];
            if (!std::isfinite(va)) throw std::runtime_error("a non-finite output");
            const double d = (double) va - vr;
            e2 += d * d;
            r2 += (double) vr * vr;
            me = std::max(me, std::fabs(d));
            mr = std::max(mr, (double) std::fabs(vr));
        }
        if (mr > 0) out.worst = std::max(out.worst, me / mr);
        else if (me > 0) out.worst = 1e30;
    }
    out.rms = r2 > 0 ? std::sqrt(e2 / r2) : 0;
    return out;
}

struct Layer {
    int T, E;
    std::vector<int32_t> cnt, off, src, order;   // per expert; sorted row -> token; experts with rows
    std::vector<int32_t> bh;                     // bounds: absolute per expert in order, then per group relative
};
Layer route(int T, int E, std::mt19937& rng, int hot) {
    Layer L{T, E, {}, {}, {}, {}, {}};
    std::vector<int32_t> ids((size_t) T * K);
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < K; ++k) {
            int e;
            bool dup;
            do {
                e = (hot > 0 && k < 3 && rng() % 4 != 0) ? (int) (rng() % (unsigned) hot) : (int) (rng() % (unsigned) E);
                dup = false;
                for (int j = 0; j < k; ++j) dup |= ids[(size_t) t * K + j] == e;
            } while (dup);
            ids[(size_t) t * K + k] = e;
        }
    L.cnt.assign((size_t) E, 0);
    for (int32_t e : ids) ++L.cnt[(size_t) e];
    L.off.assign((size_t) E + 1, 0);
    for (int e = 0; e < E; ++e) L.off[(size_t) e + 1] = L.off[(size_t) e] + L.cnt[(size_t) e];
    std::vector<int32_t> fill(L.off.begin(), L.off.end() - 1);
    L.src.resize(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) L.src[(size_t) fill[(size_t) ids[i]]++] = (int32_t) (i / K);
    for (int e = 0; e < E; ++e) if (L.cnt[(size_t) e] > 0) L.order.push_back(e);
    const size_t n = L.order.size(), ng = (n + GROUP - 1) / GROUP;
    L.bh.assign(n + 1 + ng * (GROUP + 1), 0);
    for (size_t j = 0; j < n; ++j) L.bh[j] = L.off[(size_t) L.order[j]];
    L.bh[n] = (int32_t) ids.size();
    for (size_t g = 0; g < ng; ++g)
        for (size_t i = 0; i <= GROUP; ++i)
            L.bh[n + 1 + g * (GROUP + 1) + i] = L.bh[std::min(n, g * GROUP + i)] - L.bh[g * GROUP];
    return L;
}

constexpr size_t TAIL = 4096;
struct Bufs {
    Dev xq, gu, h, hq, dm, ident, bounds, grp_gu, grp_d;
    Bufs(int64_t rows, size_t nb)
        : xq(mmq::q8_bytes(rows, N)), gu((size_t) rows * 1280 * 4), h((size_t) rows * FF * 4),
          hq(mmq::q8_bytes(rows, FF)), dm((size_t) rows * N * 4), ident((size_t) rows * 4), bounds(nb * 4),
          grp_gu(GROUP * mmq::matrix_bytes(GGML_TYPE_NVFP4, 1280, N) + TAIL),
          grp_d(GROUP * mmq::matrix_bytes(GGML_TYPE_NVFP4, N, FF) + TAIL + GROUP * 16) {}
    float* scales() const { return (float*) (grp_d.as<uint8_t>() + GROUP * mmq::matrix_bytes(GGML_TYPE_NVFP4, N, FF) + TAIL); }
};

// expert_mmq.cpp's sequence for one layer; `grouped`: one gather launch per group, else one per expert
void run_mmq(mmq::Context& ctx, Bufs& b, const Layer& L, const float* x_dev, const int32_t* src_dev,
             const std::vector<const uint8_t*>& blob, bool grouped, cudaStream_t s) {
    const size_t gub = mmq::matrix_bytes(GGML_TYPE_NVFP4, 1280, N), db = mmq::matrix_bytes(GGML_TYPE_NVFP4, N, FF);
    const int64_t rows = (int64_t) L.T * K;
    const size_t n = L.order.size();
    ck(cudaMemcpyAsync(b.bounds.p, L.bh.data(), L.bh.size() * 4, cudaMemcpyHostToDevice, s), "bounds");
    mmq::iota(b.ident.as<int32_t>(), rows, s);
    mmq::quantize(x_dev, src_dev, b.xq.p, GGML_TYPE_NVFP4, N, N, rows, s);
    mmq::GatherGroup gg;
    for (size_t j = 0; j < n; ++j) {
        const size_t q = j % GROUP;
        const uint8_t* bl = blob[(size_t) L.order[j]];
        if (grouped) {
            gg.blob[q] = bl;
            gg.n = (int) q + 1;
        } else {
            mmq::gather_native(bl, bl + UP_OFF, gub / 2, bl + DOWN_OFF, db, b.grp_gu.as<uint8_t>() + q * gub,
                               b.grp_d.as<uint8_t>() + q * db, s, bl + TAIL_OFF, b.scales() + 4 * q);
        }
        if (q + 1 < GROUP && j + 1 < n) continue;
        if (grouped) {
            if (!mmq::gather_native_group(gg, UP_OFF, gub / 2, DOWN_OFF, db, b.grp_gu.p, gub, b.grp_d.p, db, s, TAIL_OFF,
                                          b.scales()))
                throw std::runtime_error("gather_native_group refused aligned blobs");
            gg = mmq::GatherGroup{};
        }
        const size_t j0 = j - q, g = j0 / GROUP;
        const int ngx = (int) (q + 1);
        const int64_t r0 = L.bh[j0], nr = L.bh[j + 1] - r0;
        int64_t maxr = 0;
        for (size_t i = j0; i <= j; ++i) maxr = std::max<int64_t>(maxr, L.cnt[(size_t) L.order[i]]);
        ck(cudaMemsetAsync(b.grp_gu.as<uint8_t>() + (size_t) ngx * gub, 0, TAIL, s), "tail");
        ck(cudaMemsetAsync(b.grp_d.as<uint8_t>() + (size_t) ngx * db, 0, TAIL, s), "tail");
        const int32_t* rel = b.bounds.as<int32_t>() + n + 1 + g * (GROUP + 1);
        mmq::Product gu;
        gu.w = b.grp_gu.p; gu.type = GGML_TYPE_NVFP4; gu.w_rows = 1280; gu.w_cols = N; gu.expert_bytes = gub;
        gu.n = ngx; gu.xq = b.xq.p; gu.bounds = b.bounds.as<int32_t>() + j0; gu.ids = b.ident.as<int32_t>();
        gu.total_rows = rows; gu.max_rows = maxr; gu.dst = b.gu.as<float>(); gu.ld_dst = 1280;
        ctx.run(gu, s);
        mmq::swiglu(b.gu.as<float>() + r0 * 1280, b.h.as<float>() + r0 * FF, nr, FF, false, s, b.scales(), rel, ngx);
        mmq::quantize(b.h.as<float>() + r0 * FF, nullptr, b.hq.p, GGML_TYPE_NVFP4, FF, FF, nr, s);
        mmq::Product dn;
        dn.w = b.grp_d.p; dn.type = GGML_TYPE_NVFP4; dn.w_rows = N; dn.w_cols = FF; dn.expert_bytes = db;
        dn.n = ngx; dn.xq = b.hq.p; dn.bounds = rel; dn.ids = b.ident.as<int32_t>(); dn.total_rows = nr;
        dn.max_rows = maxr; dn.dst = b.dm.as<float>() + r0 * N; dn.ld_dst = N;
        ctx.run(dn, s);
    }
}

std::vector<float> download(const Dev& d, size_t n) {
    std::vector<float> h(n);
    ck(cudaMemcpy(h.data(), d.p, n * 4, cudaMemcpyDeviceToHost), "download");
    return h;
}

int parity(cudaStream_t s) {
    constexpr int T = 160, E = 40, ZERO = 33;
    std::mt19937 rng(4040);
    std::vector<std::vector<uint8_t>> host((size_t) E);
    std::vector<std::array<float, 3>> tails((size_t) E);
    for (int e = 0; e < E; ++e) host[(size_t) e] = make_blob(rng, tails[(size_t) e].data());
    std::vector<std::unique_ptr<Dev>> dev;
    std::vector<const uint8_t*> blob((size_t) E);
    for (int e = 0; e < E; ++e) {
        dev.push_back(std::make_unique<Dev>(BYTES));
        ck(cudaMemcpy(dev.back()->p, host[(size_t) e].data(), BYTES, cudaMemcpyHostToDevice), "blob");
        blob[(size_t) e] = dev.back()->as<uint8_t>();
    }
    std::vector<float> x((size_t) T * N);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (float& v : x) v = nd(rng) * (rng() % 200 == 0 ? 12.0f : 1.0f);
    std::fill(x.begin() + (size_t) ZERO * N, x.begin() + (size_t) (ZERO + 1) * N, 0.0f);
    const Layer L = route(T, E, rng, 2);   // experts 0, 1 hot: several 64-row tiles
    const int64_t rows = (int64_t) T * K;
    std::printf("parity: %d tokens, %d experts, %zu with rows, max rows per expert %d\n", T, E, L.order.size(),
                *std::max_element(L.cnt.begin(), L.cnt.end()));

    Dev x_dev(x.size() * 4), src_dev((size_t) rows * 4);
    ck(cudaMemcpy(x_dev.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(src_dev.p, L.src.data(), L.src.size() * 4, cudaMemcpyHostToDevice), "src");
    ck(cudaDeviceSynchronize(), "uploads");
    mmq::Context ctx;
    Bufs b(rows, L.bh.size());
    run_mmq(ctx, b, L, x_dev.as<float>(), src_dev.as<int32_t>(), blob, false, s);
    ck(cudaStreamSynchronize(s), "sync");
    const std::vector<float> y1 = download(b.dm, (size_t) rows * N);
    ck(cudaMemset(b.dm.p, 0xff, (size_t) rows * N * 4), "sentinel");
    ck(cudaMemset(b.scales(), 0xff, GROUP * 16), "sentinel");
    ck(cudaDeviceSynchronize(), "sentinel");
    run_mmq(ctx, b, L, x_dev.as<float>(), src_dev.as<int32_t>(), blob, true, s);
    ck(cudaStreamSynchronize(s), "sync");
    const std::vector<float> y = download(b.dm, (size_t) rows * N), ygu = download(b.gu, (size_t) rows * 1280);
    const bool same_gather = std::memcmp(y.data(), y1.data(), y.size() * 4) == 0;

    // the references, an expert at a time on all cores
    std::vector<float> r_same((size_t) rows * N), r_dbl((size_t) rows * N), r_f16((size_t) rows * N),
                       r_gu((size_t) rows * 1280);
    const auto* tr = ggml_get_type_traits(GGML_TYPE_NVFP4);
    parallel(E, [&](int e) {
        if (L.cnt[(size_t) e] == 0) return;
        const float sg = tails[(size_t) e][0], su = tails[(size_t) e][1], sd = tails[(size_t) e][2];
        std::vector<float> wgu((size_t) 2 * FF * N), wd((size_t) N * FF);
        for (int r = 0; r < 2 * FF; ++r) tr->to_float(host[(size_t) e].data() + (size_t) r * GU_ROW, wgu.data() + (size_t) r * N, N);
        for (int r = 0; r < N; ++r) tr->to_float(host[(size_t) e].data() + DOWN_OFF + (size_t) r * D_ROW, wd.data() + (size_t) r * FF, FF);
        std::vector<float> wgu16(wgu.size()), wd16(wd.size());
        for (size_t i = 0; i < wgu.size(); ++i) wgu16[i] = f16(wgu[i] * (i < (size_t) FF * N ? sg : su));
        for (size_t i = 0; i < wd.size(); ++i) wd16[i] = f16(wd[i] * sd);
        std::vector<float> xr(N), x16(N), h(FF), hr(FF), hd(FF), h16(FF);
        for (int32_t p = L.off[(size_t) e]; p < L.off[(size_t) e + 1]; ++p) {
            const float* xt = x.data() + (size_t) L.src[(size_t) p] * N;
            q8_round(xt, xr.data(), N);
            for (int c = 0; c < N; ++c) x16[(size_t) c] = f16(xt[c]);
            for (int k = 0; k < FF; ++k) {
                const float* wg = wgu.data() + (size_t) k * N;
                const float* wu = wgu.data() + (size_t) (FF + k) * N;
                const float* wg16 = wgu16.data() + (size_t) k * N;
                const float* wu16 = wgu16.data() + (size_t) (FF + k) * N;
                double g = 0, u = 0, gd = 0, ud = 0, g16 = 0, u16 = 0;
                for (int c = 0; c < N; ++c) {
                    g += (double) wg[c] * xr[(size_t) c];
                    u += (double) wu[c] * xr[(size_t) c];
                    gd += (double) wg[c] * xt[c];
                    ud += (double) wu[c] * xt[c];
                    g16 += (double) wg16[c] * x16[(size_t) c];
                    u16 += (double) wu16[c] * x16[(size_t) c];
                }
                r_gu[(size_t) p * 1280 + k] = (float) g;
                r_gu[(size_t) p * 1280 + FF + k] = (float) u;
                g *= sg; u *= su; gd *= sg; ud *= su;
                h[(size_t) k] = (float) (g / (1.0 + std::exp(-g)) * u * sd);   // MMQ's H carries s_down
                hd[(size_t) k] = (float) (gd / (1.0 + std::exp(-gd)) * ud);
                h16[(size_t) k] = f16((float) (g16 / (1.0 + std::exp(-g16)) * u16));
            }
            q8_round(h.data(), hr.data(), FF);
            float* ys = r_same.data() + (size_t) p * N;
            float* yd = r_dbl.data() + (size_t) p * N;
            float* yf = r_f16.data() + (size_t) p * N;
            for (int o = 0; o < N; ++o) {
                const float* w = wd.data() + (size_t) o * FF;
                const float* w16 = wd16.data() + (size_t) o * FF;
                double a = 0, ad = 0, af = 0;
                for (int k = 0; k < FF; ++k) {
                    a += (double) w[k] * hr[(size_t) k];
                    ad += (double) w[k] * hd[(size_t) k];
                    af += (double) w16[k] * h16[(size_t) k];
                }
                ys[o] = (float) a;
                yd[o] = (float) (ad * sd);
                yf[o] = (float) af;
            }
        }
    });
    const Err es = compare(y, r_same, (size_t) rows), ed = compare(y, r_dbl, (size_t) rows),
              ef = compare(y, r_f16, (size_t) rows), fd = compare(r_f16, r_dbl, (size_t) rows);
    double ge2 = 0, gr2 = 0, gworst = 0;
    for (int64_t p = 0; p < rows; ++p) {
        double me = 0, mr = 0;
        for (int c = 0; c < 1280; ++c) {
            const double d = (double) ygu[(size_t) p * 1280 + c] - r_gu[(size_t) p * 1280 + c];
            ge2 += d * d;
            gr2 += (double) r_gu[(size_t) p * 1280 + c] * r_gu[(size_t) p * 1280 + c];
            me = std::max(me, std::fabs(d));
            mr = std::max(mr, (double) std::fabs(r_gu[(size_t) p * 1280 + c]));
        }
        if (mr > 0) gworst = std::max(gworst, me / mr);
    }
    const double grms = gr2 > 0 ? std::sqrt(ge2 / gr2) : 0;
    bool zero_ok = true;
    for (int64_t p = 0; p < rows; ++p)
        if (L.src[(size_t) p] == ZERO)
            for (int o = 0; o < N; ++o) zero_ok &= y[(size_t) p * N + o] == 0.0f;
    std::printf("  gate/up vs same-rounding reference: rms %.3e  worst row %.3e   (tol 1e-5, 1e-4)\n", grms, gworst);
    std::printf("  MMQ vs same-rounding reference: rms %.3e  worst row %.3e   (tol 2e-4, 1e-2)\n", es.rms, es.worst);
    std::printf("  MMQ vs double reference:        rms %.3e  worst row %.3e   (tol rms 2.5e-2)\n", ed.rms, ed.worst);
    std::printf("  MMQ vs FP16 fallback:           rms %.3e  worst row %.3e   (tol rms 2e-2)\n", ef.rms, ef.worst);
    std::printf("  FP16 fallback vs double:        rms %.3e  worst row %.3e\n", fd.rms, fd.worst);
    std::printf("  grouped gather == per-expert gather (bitwise): %s; zero token exactly 0: %s\n",
                same_gather ? "yes" : "NO", zero_ok ? "yes" : "NO");
    const bool ok = grms <= 1e-5 && gworst <= 1e-4 && es.rms <= 2e-4 && es.worst <= 1e-2 && ed.rms <= 2.5e-2 &&
                    ef.rms <= 2e-2 && same_gather && zero_ok;
    std::printf("parity: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

void bench(cudaStream_t s) {
    constexpr int T = 2048, E = 512;
    std::mt19937 rng(77);
    // random code bytes with sane sub-block scales (timing only): one blob copied to every slot
    std::vector<uint8_t> h(BYTES);
    for (auto& v : h) v = (uint8_t) (rng() & 0x77);
    const float t4[4] = {0.01f, 0.01f, 0.01f, 0.0f};
    std::memcpy(h.data() + TAIL_OFF, t4, 16);
    const size_t stride = (BYTES + 255) / 256 * 256;
    Dev all(stride * E);
    for (int e = 0; e < E; ++e) ck(cudaMemcpy(all.as<uint8_t>() + stride * e, h.data(), BYTES, cudaMemcpyHostToDevice), "blob");
    std::vector<const uint8_t*> blob((size_t) E);
    for (int e = 0; e < E; ++e) blob[(size_t) e] = all.as<uint8_t>() + stride * e;
    std::vector<float> x((size_t) T * N);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (float& v : x) v = nd(rng);
    const Layer L = route(T, E, rng, 0);
    const int64_t rows = (int64_t) T * K;
    Dev x_dev(x.size() * 4), src_dev((size_t) rows * 4);
    ck(cudaMemcpy(x_dev.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(src_dev.p, L.src.data(), L.src.size() * 4, cudaMemcpyHostToDevice), "src");
    ck(cudaDeviceSynchronize(), "uploads");
    mmq::Context ctx;
    Bufs b(rows, L.bh.size());
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int w = 0; w < 2; ++w) run_mmq(ctx, b, L, x_dev.as<float>(), src_dev.as<int32_t>(), blob, true, s);
    constexpr int R = 10;
    cudaEventRecord(e0, s);
    for (int r = 0; r < R; ++r) run_mmq(ctx, b, L, x_dev.as<float>(), src_dev.as<int32_t>(), blob, true, s);
    cudaEventRecord(e1, s);
    ck(cudaEventSynchronize(e1), "bench");
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    const double us = 1e3 * ms / R, flop = 2.0 * (double) rows * (1280.0 * N + (double) N * FF);
    std::printf("bench: NVFP4 MMQ layer, %d tokens x top %d over %d experts (%zu with rows): %.1f us, %.1f TFLOP/s "
                "(gathers + q8_1 + gate/up + SwiGLU + down)\n", T, K, E, L.order.size(), us, flop / us * 1e-6);
}

}  // namespace

int main(int argc, char** argv) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
    bool do_bench = true;
    for (int i = 1; i < argc; ++i) if (std::strcmp(argv[i], "--no-bench") == 0) do_bench = false;
    try {
        if (!mmq::fits(GGML_TYPE_NVFP4, 1280) || !mmq::fits(GGML_TYPE_NVFP4, N)) {
            std::printf("MMQ has no NVFP4 tile on this device\n");
            return 1;
        }
        cudaStream_t s;
        ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        const int rc = parity(s);
        if (do_bench) bench(s);
        cudaStreamDestroy(s);
        return rc;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "prefill_mmq_nvfp4_test: %s\n", ex.what());
        return 1;
    }
}
