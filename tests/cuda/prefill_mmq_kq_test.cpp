// prefill_mmq_kq_test - llama.cpp's MMQ for the K-quant expert formats (gate/up Q4_K, Q5_K or Q6_K, down Q5_1 or Q8_0: the
// UD-Q4_K_XL pack's layers) on random native expert blobs at the model's geometry (2560 x 640), against
//   - a double-precision reference: the weights dequantized by ggml's own to_float, times the FP32 activations, and
//   - the path these formats took before: each expert dequantized to FP16 (iq_dequant_gu_f16 / iq_dequant_f16), FP16
//     activations, two cuBLAS products and the FP16 SwiGLU, as expert_mmq.cpp runs it.
// MMQ rounds the activations and H to int8 per 32 values, so this is not a bitwise test: its error against the
// reference has to stay within the bounds the i-quant MMQ path meets (relative RMS 2e-2 with plain activations, 3e-2
// with outliers; worst row 0.1 / 0.15).  Part 1, per pair: 256 tokens over 64 experts, skewed routing (experts with no
// rows, with one, with several tiles), an all-zero token, each blob in its own allocation; then eight tokens
// again, each alone, and alone at another place in a batch: a row's output may differ from its row in the full batch only
// by rounding (MMQ's stream-k split and tile width follow the launch's size, for every format; H's int8 rounding can
// carry that to one int8 step of an H value), at most a tenth of MMQ's worst-row error against the reference.  Part 2 (timing, --bench): one group of 16 experts at 64-640 rows each, the FP16 path against MMQ
// (with its gathers, and its products alone), microseconds per expert and the achieved int8/fp16 TFLOP/s.
// --no-ref, --bench.  Exit 77 without a CUDA device.
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/prefill/gemm.hpp"
#include "basalt/prefill/kernels.hpp"
#include "basalt/prefill/moe_mmq.hpp"

#include "basalt/ggml/ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Dev {
    void* p = nullptr;
    explicit Dev(size_t n) { ck(cudaMalloc(&p, std::max<size_t>(n, 16)), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    template <typename T> T* as() const { return (T*) p; }
};

struct Pair {
    ggml_type gu, d;
    const char* name;
};

struct Geo {
    size_t gu_row, d_row, up_off, down_off, bytes;
    explicit Geo(const Pair& p) {
        gu_row = ggml_row_size(p.gu, N);
        d_row = ggml_row_size(p.d, FF);
        up_off = gu_row * FF;
        down_off = 2 * up_off;
        bytes = down_off + d_row * N;
    }
};

// random blocks with sane fp16 scales: Q6_K (scales and d set below), Q4_K/Q5_K {d, dmin} at the block's start, Q5_1 {d, m}, Q8_0 {d}; sized so
// the weights' typical magnitude is that of a 4-5 bit expert (a few hundredths), and with the minimum at about the
// codes' mean times the scale, so the weights are centred on zero as a trained matrix's are
void fill_matrix(uint8_t* m, ggml_type t, int64_t rows, int64_t cols, std::mt19937& rng) {
    const size_t bs = ggml_type_size(t), nb = (size_t) rows * (size_t) (cols / ggml_blck_size(t));
    std::uniform_int_distribution<int> byte(0, 255);
    for (size_t i = 0; i < nb * bs; ++i) m[i] = (uint8_t) byte(rng);
    float lo = 0.0008f, mid = 0.0f;   // mid: the minimum's size in units of d
    switch (t) {
        case GGML_TYPE_Q4_K: lo = 0.0002f; mid = 7.5f; break;    // times 6-bit sub-scales and sub-minima
        case GGML_TYPE_Q5_K: lo = 0.0001f; mid = 15.5f; break;
        case GGML_TYPE_Q5_1: lo = 0.002f; mid = -15.5f; break;   // the minimum is added: negative
        default: break;
    }
    std::uniform_real_distribution<float> sc(lo, 4.0f * lo);
    if (t == GGML_TYPE_Q6_K) {   // {ql, qh, int8 scales, d}: scales of both signs, d at the block's end
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
        const float d = sc(rng);
        const ggml_fp16_t h = ggml_fp32_to_fp16(d);
        std::memcpy(m + b * bs, &h, 2);
        if (mid != 0.0f) {
            const ggml_fp16_t hm = ggml_fp32_to_fp16(mid * d);
            std::memcpy(m + b * bs + 2, &hm, 2);
        }
    }
}
std::vector<uint8_t> make_blob(const Pair& p, const Geo& g, std::mt19937& rng) {
    std::vector<uint8_t> b(g.bytes);
    fill_matrix(b.data(), p.gu, 2 * FF, N, rng);
    fill_matrix(b.data() + g.down_off, p.d, N, FF, rng);
    return b;
}

void dequant(const Pair& p, const Geo& g, const std::vector<uint8_t>& b, std::vector<float>& gu, std::vector<float>& dn) {
    const auto* tg = ggml_get_type_traits(p.gu);
    const auto* td = ggml_get_type_traits(p.d);
    gu.resize((size_t) 2 * FF * N);
    dn.resize((size_t) N * FF);
    for (int r = 0; r < 2 * FF; ++r) tg->to_float(b.data() + (size_t) r * g.gu_row, gu.data() + (size_t) r * N, N);
    for (int r = 0; r < N; ++r) td->to_float(b.data() + g.down_off + (size_t) r * g.d_row, dn.data() + (size_t) r * FF, FF);
}

std::vector<float> make_x(int T, std::mt19937& rng, int zero, bool outliers) {
    std::vector<float> x((size_t) T * N);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> pick(0, 199);
    for (float& v : x) v = nd(rng) * (outliers && pick(rng) == 0 ? 12.0f : 1.0f);
    if (zero >= 0) std::fill(x.begin() + (size_t) zero * N, x.begin() + (size_t) (zero + 1) * N, 0.0f);
    return x;
}

// K distinct experts per token; the first choices skewed towards the lowest `hot` ids (tiles of several rows)
std::vector<int32_t> make_ids(int T, int E, std::mt19937& rng, int hot, int unused) {
    std::vector<int32_t> ids((size_t) T * K);
    std::uniform_int_distribution<int> any(0, E - 1 - unused);
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < K; ++k) {
            int e;
            bool dup;
            do {
                e = (hot > 0 && k < 3 && (int) (rng() % 4) != 0) ? (int) (rng() % (unsigned) hot) : any(rng);
                dup = false;
                for (int j = 0; j < k; ++j) dup |= ids[(size_t) t * K + j] == e;
            } while (dup);
            ids[(size_t) t * K + k] = e;
        }
    return ids;
}

struct Routing {
    std::vector<int32_t> cnt, off, src;   // per expert; sorted row -> token
    std::vector<int32_t> row_of;          // pair -> sorted row
};
Routing sort_rows(const std::vector<int32_t>& ids, int E) {
    Routing r;
    r.cnt.assign((size_t) E, 0);
    for (int32_t e : ids) ++r.cnt[(size_t) e];
    r.off.assign((size_t) E + 1, 0);
    for (int e = 0; e < E; ++e) r.off[(size_t) e + 1] = r.off[(size_t) e] + r.cnt[(size_t) e];
    std::vector<int32_t> fill(r.off.begin(), r.off.end() - 1);
    r.src.resize(ids.size());
    r.row_of.resize(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        const int32_t p = fill[(size_t) ids[i]]++;
        r.row_of[i] = p;
        r.src[(size_t) p] = (int32_t) (i / K);
    }
    return r;
}

// MMQ, as expert_mmq.cpp runs a native layer: per group of 16 routed experts a gather, gate/up, SwiGLU (gate rows, then
// up rows), q8_1 of H, down.  Dm [rows][N] in sorted-row order.
struct MmqBufs {
    Dev xq, gu, h, hq, dm, ident, bounds, grp_gu, grp_d;
    MmqBufs(const Pair& p, int64_t rows, int E)
        : xq(mmq::q8_bytes(rows, N)), gu((size_t) rows * 1280 * 4), h((size_t) rows * FF * 4),
          hq(mmq::q8_bytes(rows, FF)), dm((size_t) rows * N * 4), ident((size_t) rows * 4),
          bounds((size_t) (2 * (E + E / GROUP + 2)) * 4), grp_gu(GROUP * mmq::matrix_bytes(p.gu, 1280, N) + 4096),
          grp_d(GROUP * mmq::matrix_bytes(p.d, N, FF) + 4096) {}
};
// `direct`: no gathers - the products read the blobs where they are (a fixed stride apart, every expert with rows)
void run_mmq(const Pair& p, const Geo& geo, mmq::Context& ctx, MmqBufs& b, const Routing& r, const float* x_dev,
             const int32_t* src_dev, const std::vector<const uint8_t*>& blob, int64_t rows, cudaStream_t s,
             bool direct = false, size_t stride = 0) {
    const size_t gub = mmq::matrix_bytes(p.gu, 1280, N), db = mmq::matrix_bytes(p.d, N, FF);
    const int E = (int) r.cnt.size();
    std::vector<int32_t> order;
    for (int e = 0; e < E; ++e) if (r.cnt[(size_t) e] > 0) order.push_back(e);
    const size_t n = order.size(), ng = (n + GROUP - 1) / GROUP;
    std::vector<int32_t> bh(n + 1 + ng * (GROUP + 1));
    for (size_t j = 0; j < n; ++j) bh[j] = r.off[(size_t) order[j]];
    bh[n] = (int32_t) rows;
    for (size_t g = 0; g < ng; ++g)
        for (size_t i = 0; i <= GROUP; ++i)
            bh[n + 1 + g * (GROUP + 1) + i] = bh[std::min(n, g * GROUP + i)] - bh[g * GROUP];
    ck(cudaMemcpyAsync(b.bounds.p, bh.data(), bh.size() * 4, cudaMemcpyHostToDevice, s), "bounds");
    mmq::iota(b.ident.as<int32_t>(), rows, s);
    mmq::quantize(x_dev, src_dev, b.xq.p, (int) p.gu, N, N, rows, s);
    for (size_t j = 0; j < n; ++j) {
        const size_t q = j % GROUP;
        const uint8_t* bl = blob[(size_t) order[j]];
        if (!direct)
            mmq::gather_native(bl, bl + geo.up_off, gub / 2, bl + geo.down_off, db, b.grp_gu.as<uint8_t>() + q * gub,
                               b.grp_d.as<uint8_t>() + q * db, s);
        if (q + 1 < GROUP && j + 1 < n) continue;
        const size_t j0 = j - q, g = j0 / GROUP;
        const int ngx = (int) (q + 1);
        const int64_t r0 = bh[j0], nr = bh[j + 1] - r0;
        int64_t maxr = 0;
        for (size_t i = j0; i <= j; ++i) maxr = std::max<int64_t>(maxr, r.cnt[(size_t) order[i]]);
        if (!direct) {
            ck(cudaMemsetAsync(b.grp_gu.as<uint8_t>() + (size_t) ngx * gub, 0, 4096, s), "tail");
            ck(cudaMemsetAsync(b.grp_d.as<uint8_t>() + (size_t) ngx * db, 0, 4096, s), "tail");
        }
        const uint8_t* b0 = blob[(size_t) order[j0]];
        mmq::Product gu;
        gu.w = direct ? (const void*) b0 : b.grp_gu.p; gu.type = (int) p.gu; gu.w_rows = 1280; gu.w_cols = N;
        gu.expert_bytes = direct ? stride : gub;
        gu.n = ngx; gu.xq = b.xq.p; gu.bounds = b.bounds.as<int32_t>() + j0; gu.ids = b.ident.as<int32_t>();
        gu.total_rows = rows; gu.max_rows = maxr; gu.dst = b.gu.as<float>(); gu.ld_dst = 1280;
        ctx.run(gu, s);
        mmq::swiglu(b.gu.as<float>() + r0 * 1280, b.h.as<float>() + r0 * FF, nr, FF, false, s);
        mmq::quantize(b.h.as<float>() + r0 * FF, nullptr, b.hq.p, (int) p.d, FF, FF, nr, s);
        mmq::Product dn;
        dn.w = direct ? (const void*) (b0 + geo.down_off) : b.grp_d.p; dn.type = (int) p.d; dn.w_rows = N; dn.w_cols = FF;
        dn.expert_bytes = direct ? stride : db;
        dn.n = ngx; dn.xq = b.hq.p; dn.bounds = b.bounds.as<int32_t>() + n + 1 + g * (GROUP + 1);
        dn.ids = b.ident.as<int32_t>(); dn.total_rows = nr; dn.max_rows = maxr; dn.dst = b.dm.as<float>() + r0 * N;
        dn.ld_dst = N;
        ctx.run(dn, s);
    }
}

// The FP16 path: per routed expert, gate/up dequantized interleaved and down to FP16, X (FP16, sorted rows) times
// them through cuBLAS, the FP16 SwiGLU between.  Dm [rows][N] in sorted-row order.
struct F16Bufs {
    Dev xs, gu, hh, dm, dq_gu, dq_d;
    explicit F16Bufs(int64_t rows)
        : xs((size_t) rows * N * 2), gu((size_t) rows * 1280 * 4), hh((size_t) rows * FF * 2), dm((size_t) rows * N * 4),
          dq_gu((size_t) 1280 * N * 2), dq_d((size_t) N * FF * 2) {}
};
void run_f16(const Pair& p, const Geo& geo, basalt::prefill::Gemm& gemm, F16Bufs& b, const Routing& r,
             const std::vector<const uint8_t*>& blob, cudaStream_t s) {
    const int E = (int) r.cnt.size();
    for (int e = 0; e < E; ++e) {
        const int64_t ne = r.cnt[(size_t) e], o0 = r.off[(size_t) e];
        if (ne == 0) continue;
        const uint8_t* bl = blob[(size_t) e];
        basalt::kernels::iq_dequant_gu_f16((int) p.gu, bl, bl + geo.up_off, FF, N, b.dq_gu.as<uint16_t>(), s);
        basalt::kernels::iq_dequant_f16((int) p.d, bl + geo.down_off, (int64_t) N * FF, b.dq_d.as<uint16_t>(), s);
        ck(cudaGetLastError(), "dequant");
        gemm.f16(b.xs.as<uint16_t>() + o0 * N, b.dq_gu.as<uint16_t>(), b.gu.as<float>() + o0 * 1280, ne, 1280, N);
        basalt::prefill::swiglu_interleaved(b.gu.as<float>() + o0 * 1280, b.hh.as<uint16_t>() + o0 * FF, ne, s);
        ck(cudaGetLastError(), "swiglu");
        gemm.f16(b.hh.as<uint16_t>() + o0 * FF, b.dq_d.as<uint16_t>(), b.dm.as<float>() + o0 * N, ne, N, FF);
    }
}
// X in sorted-row order, FP16 (the prompt path's Xs)
void upload_xs(const std::vector<float>& x, const Routing& r, F16Bufs& b) {
    std::vector<ggml_fp16_t> h(r.src.size() * N);
    for (size_t i = 0; i < r.src.size(); ++i)
        for (int k = 0; k < N; ++k) h[i * N + k] = ggml_fp32_to_fp16(x[(size_t) r.src[i] * N + k]);
    ck(cudaMemcpy(b.xs.p, h.data(), h.size() * 2, cudaMemcpyHostToDevice), "xs");
}

struct Err {
    double rms = 0, worst = 0;   // RMS error / RMS of the reference; the worst row's max |error| / its max |ref|
    double amax = 0, rmax = 0;   // max |error|; max |error| / |ref| over elements with |ref| above 1% of the row's max
};
template <typename RA, typename RB>
Err compare(const std::vector<float>& a, RA row_a, const std::vector<float>& b, RB row_b, size_t pairs) {
    double e2 = 0, r2 = 0;
    Err out;
    for (size_t i = 0; i < pairs; ++i) {
        const float* ya = a.data() + (size_t) row_a(i) * N;
        const float* yb = b.data() + (size_t) row_b(i) * N;
        double me = 0, mr = 0;
        for (int o = 0; o < N; ++o) mr = std::max(mr, (double) std::fabs(yb[o]));
        for (int o = 0; o < N; ++o) {
            if (!std::isfinite(ya[o])) throw std::runtime_error("a non-finite output");
            const double d = (double) ya[o] - yb[o];
            e2 += d * d;
            r2 += (double) yb[o] * yb[o];
            me = std::max(me, std::fabs(d));
            if (std::fabs(yb[o]) > 0.01 * mr) out.rmax = std::max(out.rmax, std::fabs(d) / std::fabs(yb[o]));
        }
        out.amax = std::max(out.amax, me);
        if (mr > 0) out.worst = std::max(out.worst, me / mr);
        else if (me > 0) out.worst = 1e30;   // a zero row must stay exactly zero
    }
    out.rms = r2 > 0 ? std::sqrt(e2 / r2) : 0;
    return out;
}

std::vector<float> download(const Dev& d, size_t n) {
    std::vector<float> h(n);
    ck(cudaMemcpy(h.data(), d.p, n * 4, cudaMemcpyDeviceToHost), "download");
    return h;
}

void print_err(const char* what, const Err& e) {
    std::printf("  %-22s rel RMS %.3e  worst row %.3e  max abs %.3e  max rel %.3e\n", what, e.rms, e.worst, e.amax,
                e.rmax);
}

// Q6_K gate/up has no dequant-to-FP16 path (MMQ is its only prefill path), so it is checked against the CPU reference alone
bool has_f16(const Pair& p) { return p.gu != GGML_TYPE_Q6_K; }

// part 1
void run_pair(const Pair& p, bool outliers, cudaStream_t s) {
    constexpr int T = 256, E = 64;
    const Geo geo(p);
    std::mt19937 rng(11 + (unsigned) p.gu * 7 + (unsigned) p.d + (outliers ? 1000u : 0u));
    std::vector<std::vector<uint8_t>> host((size_t) E);
    std::vector<std::unique_ptr<Dev>> dev;
    std::vector<const uint8_t*> blob((size_t) E);
    for (int e = 0; e < E; ++e) {
        host[(size_t) e] = make_blob(p, geo, rng);
        dev.push_back(std::make_unique<Dev>(geo.bytes + 64));
        uint8_t* q = dev.back()->as<uint8_t>();   // the FP16 path's dequantizers need aligned blobs
        ck(cudaMemcpy(q, host[(size_t) e].data(), geo.bytes, cudaMemcpyHostToDevice), "blob");
        blob[(size_t) e] = q;
    }
    const std::vector<float> x = make_x(T, rng, 5, outliers);
    const std::vector<int32_t> ids = make_ids(T, E, rng, 6, 3);
    const Routing r = sort_rows(ids, E);
    const int64_t rows = (int64_t) T * K;
    int used = 0, maxr = 0;
    for (int c : r.cnt) { used += c > 0; maxr = std::max(maxr, c); }
    std::printf("%s, %s activations: %d tokens, %d routed experts, up to %d rows each\n", p.name,
                outliers ? "with outliers" : "plain", T, used, maxr);

    Dev x_dev(x.size() * 4), src_dev((size_t) rows * 4);
    ck(cudaMemcpy(x_dev.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(src_dev.p, r.src.data(), r.src.size() * 4, cudaMemcpyHostToDevice), "src");
    mmq::Context ctx;
    MmqBufs mb(p, rows, E);
    run_mmq(p, geo, ctx, mb, r, x_dev.as<float>(), src_dev.as<int32_t>(), blob, rows, s);
    ck(cudaStreamSynchronize(s), "mmq");
    const std::vector<float> y_mmq = download(mb.dm, (size_t) rows * N);

    const bool f16 = has_f16(p);
    basalt::prefill::Gemm gemm;
    std::vector<float> y_f16;
    if (f16) {
        std::string err;
        if (!gemm.init(s, 0, err)) throw std::runtime_error(err);
        F16Bufs fb(rows);
        upload_xs(x, r, fb);
        run_f16(p, geo, gemm, fb, r, blob, s);
        ck(cudaStreamSynchronize(s), "f16");
        y_f16 = download(fb.dm, (size_t) rows * N);
    }

    // the reference
    std::vector<float> ref((size_t) rows * N);
    std::vector<std::thread> th;
    const int nth = std::max(1, std::min(32, (int) std::thread::hardware_concurrency()));
    for (int w = 0; w < nth; ++w)
        th.emplace_back([&, w] {
            std::vector<float> gu, dn, h(FF);
            for (int e = w; e < E; e += nth) {
                if (r.cnt[(size_t) e] == 0) continue;
                dequant(p, geo, host[(size_t) e], gu, dn);
                for (int64_t i = 0; i < rows; ++i) {
                    if (ids[(size_t) i] != e) continue;
                    const float* xr = x.data() + (size_t) (i / K) * N;
                    for (int ff = 0; ff < FF; ++ff) {
                        double gt = 0, up = 0;
                        const float *wg = gu.data() + (size_t) ff * N, *wu = gu.data() + (size_t) (FF + ff) * N;
                        for (int k = 0; k < N; ++k) { gt += (double) wg[k] * xr[k]; up += (double) wu[k] * xr[k]; }
                        h[(size_t) ff] = (float) (gt / (1.0 + std::exp(-gt)) * up);
                    }
                    for (int o = 0; o < N; ++o) {
                        double a = 0;
                        const float* wd = dn.data() + (size_t) o * FF;
                        for (int ff = 0; ff < FF; ++ff) a += (double) wd[ff] * h[(size_t) ff];
                        ref[(size_t) i * N + o] = (float) a;
                    }
                }
            }
        });
    for (auto& t : th) t.join();

    auto pairi = [](size_t i) { return (int64_t) i; };
    auto srow = [&](size_t i) { return (int64_t) r.row_of[i]; };
    const Err em = compare(y_mmq, srow, ref, pairi, (size_t) rows);
    print_err("MMQ vs reference:", em);
    if (f16) {
        print_err("FP16 path vs reference:", compare(y_f16, srow, ref, pairi, (size_t) rows));
        print_err("MMQ vs FP16 path:", compare(y_mmq, srow, y_f16, srow, (size_t) rows));
    }
    const double lim_rms = outliers ? 3e-2 : 2e-2, lim_worst = outliers ? 0.15 : 0.1;
    if (em.rms > lim_rms || em.worst > lim_worst)
        throw std::runtime_error(std::string(p.name) + ": MMQ's error is above the i-quant path's bounds");

    int differ = 0;
    double worst_alone = 0;
    // batch independence: tokens 0..7 again, each alone, and each alone as the last token of a 3-token batch whose
    // other two tokens route elsewhere
    for (int t = 0; t < 8; ++t) {
        for (int variant = 0; variant < 2; ++variant) {
            const int Tn = variant == 0 ? 1 : 3, at = Tn - 1;
            std::vector<int32_t> ids1((size_t) Tn * K);
            std::vector<float> x1((size_t) Tn * N);
            for (int u = 0; u < Tn; ++u) {
                if (u == at) {
                    std::copy(ids.begin() + (size_t) t * K, ids.begin() + (size_t) (t + 1) * K, ids1.begin() + (size_t) u * K);
                    std::copy(x.begin() + (size_t) t * N, x.begin() + (size_t) (t + 1) * N, x1.begin() + (size_t) u * N);
                } else {
                    // another token's activations and experts not routed for token t
                    int filled = 0;
                    for (int e = E - 1; filled < K; --e) {
                        bool mine = false;
                        for (int k = 0; k < K; ++k) mine |= ids[(size_t) t * K + k] == e;
                        bool taken = false;
                        for (int k = 0; k < filled; ++k) taken |= ids1[(size_t) u * K + k] == e;
                        if (!mine && !taken && (e + u) % 2 == 0) ids1[(size_t) u * K + filled++] = e;
                    }
                    std::copy(x.begin() + (size_t) (100 + u) * N, x.begin() + (size_t) (101 + u) * N, x1.begin() + (size_t) u * N);
                }
            }
            const Routing r1 = sort_rows(ids1, E);
            const int64_t rows1 = (int64_t) Tn * K;
            Dev x1d(x1.size() * 4), s1d((size_t) rows1 * 4);
            ck(cudaMemcpy(x1d.p, x1.data(), x1.size() * 4, cudaMemcpyHostToDevice), "x1");
            ck(cudaMemcpy(s1d.p, r1.src.data(), r1.src.size() * 4, cudaMemcpyHostToDevice), "src1");
            MmqBufs mb1(p, rows1, E);
            run_mmq(p, geo, ctx, mb1, r1, x1d.as<float>(), s1d.as<int32_t>(), blob, rows1, s);
            ck(cudaStreamSynchronize(s), "mmq alone");
            const std::vector<float> y1 = download(mb1.dm, (size_t) rows1 * N);
            for (int k = 0; k < K; ++k) {
                const float* a = y1.data() + (size_t) r1.row_of[(size_t) at * K + k] * N;
                const float* b = y_mmq.data() + (size_t) r.row_of[(size_t) t * K + k] * N;
                double md = 0, mr = 0;
                for (int o = 0; o < N; ++o) {
                    md = std::max(md, (double) std::fabs(a[o] - b[o]));
                    mr = std::max(mr, (double) std::fabs(b[o]));
                }
                if (md > 0) ++differ;
                worst_alone = std::max(worst_alone, mr > 0 ? md / mr : md);
            }
        }
    }
    // MMQ's stream-k split of the K loop and its tile width follow the launch's size, so a row's sum can be rounded in
    // another order (every MMQ format does this), and H's int8 rounding can turn that into one int8 step of an H value:
    // a few 1e-4 of the row's largest output.  It has to stay an order of magnitude below MMQ's own error against the
    // reference; an indexing error would be of the order of the outputs themselves.
    std::printf("  rows alone and in another batch: %d of 160 rows differ, worst max |diff| / row max %.3e\n", differ,
                worst_alone);
    if (worst_alone > 0.1 * em.worst)
        throw std::runtime_error(std::string(p.name) + ": a row's output depends on the batch beyond rounding");
}

// part 2: one group of 16 experts at R rows each
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

void bench_pair(const Pair& p, cudaStream_t s) {
    constexpr int E = GROUP;
    const Geo geo(p);
    std::mt19937 rng(5 + (unsigned) p.gu);
    // the experts at a fixed stride (the "direct" layout), each blob 256-byte aligned
    const size_t stride = (geo.bytes + 255) / 256 * 256;
    Dev blobs(stride * E + 4096);
    ck(cudaMemset(blobs.p, 0, stride * E + 4096), "zero");
    std::vector<const uint8_t*> blob((size_t) E);
    for (int e = 0; e < E; ++e) {
        const std::vector<uint8_t> h = make_blob(p, geo, rng);
        ck(cudaMemcpy(blobs.as<uint8_t>() + (size_t) e * stride, h.data(), geo.bytes, cudaMemcpyHostToDevice), "blob");
        blob[(size_t) e] = blobs.as<uint8_t>() + (size_t) e * stride;
    }
    const bool f16 = has_f16(p);
    basalt::prefill::Gemm gemm;
    std::string err;
    if (f16 && !gemm.init(s, 0, err)) throw std::runtime_error(err);
    mmq::Context ctx;
    const double flop_e = 2.0 * (1280.0 * N + (double) N * FF);   // per row
    const double wbytes = (double) mmq::matrix_bytes(p.gu, 1280, N) + (double) mmq::matrix_bytes(p.d, N, FF);
    std::printf("%s: one group of %d experts (weights %.2f MB each)\n", p.name, E, wbytes / 1e6);
    std::printf("  %5s  %12s  %12s  %12s  %8s  %10s  %10s\n", "rows", "FP16 us/exp", "MMQ us/exp", "MMQ-k us/exp",
                "speedup", "MMQ TFLOP/s", "FP16 TFLOP/s");
    for (int R : {64, 100, 150, 200, 300, 450, 600}) {
        const int T = R * E / K;   // each token on 10 of the 16 experts: every expert gets R rows
        std::vector<int32_t> ids((size_t) T * K);
        for (int t = 0; t < T; ++t)
            for (int k = 0; k < K; ++k) ids[(size_t) t * K + k] = (t + k) % E;
        const Routing r = sort_rows(ids, E);
        const int64_t rows = (int64_t) T * K;
        std::vector<float> x = make_x(T, rng, -1, false);
        Dev x_dev(x.size() * 4), src_dev((size_t) rows * 4);
        ck(cudaMemcpy(x_dev.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
        ck(cudaMemcpy(src_dev.p, r.src.data(), r.src.size() * 4, cudaMemcpyHostToDevice), "src");
        MmqBufs mb(p, rows, E);
        F16Bufs fb(f16 ? rows : 1);
        if (f16) upload_xs(x, r, fb);
        const int reps = 21;
        const double t_f16 = f16 ? time_ms([&] { run_f16(p, geo, gemm, fb, r, blob, s); }, s, reps) : 0.0;
        const double t_mmq = time_ms([&] {
            run_mmq(p, geo, ctx, mb, r, x_dev.as<float>(), src_dev.as<int32_t>(), blob, rows, s);
        }, s, reps);
        const double t_dir = time_ms([&] {
            run_mmq(p, geo, ctx, mb, r, x_dev.as<float>(), src_dev.as<int32_t>(), blob, rows, s, true, stride);
        }, s, reps);
        const double fl = flop_e * rows;
        if (!f16) {
            std::printf("  %5d  %12s  %12.1f  %12.1f  %8s  %10.1f  %10s\n", R, "-", 1000 * t_mmq / E, 1000 * t_dir / E,
                        "-", fl / (t_mmq * 1e9), "-");
            continue;
        }
        std::printf("  %5d  %12.1f  %12.1f  %12.1f  %7.2fx  %10.1f  %10.1f\n", R, 1000 * t_f16 / E, 1000 * t_mmq / E,
                    1000 * t_dir / E, t_f16 / t_mmq, fl / (t_mmq * 1e9), fl / (t_f16 * 1e9));
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
        bool ref = true, bench = false;
        for (int i = 1; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--no-ref")) ref = false;
            else if (!std::strcmp(argv[i], "--bench")) bench = true;
            else throw std::runtime_error(std::string("unknown argument ") + argv[i]);
        }
        const Pair pairs[] = {{GGML_TYPE_Q4_K, GGML_TYPE_Q5_1, "Q4_K / Q5_1"},
                              {GGML_TYPE_Q4_K, GGML_TYPE_Q8_0, "Q4_K / Q8_0"},
                              {GGML_TYPE_Q5_K, GGML_TYPE_Q8_0, "Q5_K / Q8_0"},
                              {GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, "Q6_K / Q8_0"}};
        for (const Pair& p : pairs)
            if (!mmq::fits((int) p.gu, 1280) || !mmq::fits((int) p.d, N))
                throw std::runtime_error(std::string(p.name) + ": MMQ does not take this pair");
        cudaStream_t s = nullptr;
        ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        if (ref)
            for (const Pair& p : pairs)
                for (bool outliers : {false, true}) run_pair(p, outliers, s);
        if (bench)
            for (const Pair& p : pairs) bench_pair(p, s);
        ck(cudaStreamDestroy(s), "destroy");
        std::printf("prefill K-quant MMQ parity passed\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "prefill K-quant MMQ parity failed: %s\n", e.what());
        return 1;
    }
}
