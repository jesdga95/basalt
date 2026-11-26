// prefill_mmq_mxfp4_test - the prompt path's MXFP4 experts through MMQ (moe_mmq.hpp): the MXFP4 tile loader with the
// E8M0-scaled int8 dot and the IQ4_NL-layout Q8_0 tile, at the real deepseek41 geometry (gate/up 4608 x 5120,
// down 5120 x 2304, top 6) on synthetic experts (IQ2 random codes with small fp16 scales; MXFP4 random codes with
// E8M0 scale bytes 118..123), in expert_mmq.cpp's sequence (16-expert groups, q8_1 of the activations, gate/up,
// SwiGLU, q8_1 of H, down).  The parity runs once per gate/up format of the UD file's MXFP4 layers: IQ2_XXS,
// IQ2_XS and IQ2_S.  Checked against two CPU references:
//   - `same`: double-precision math on MMQ's own roundings (x and H rounded to int8 per 32 values the way the
//     quantizer does).  The gate/up product isolates the loader and the dot: only fp32 accumulation order differs.
//     Tolerance: RMS 1e-5, worst row 1e-4.  After the down product an H value whose x/d lands within fp32 rounding
//     of a .5 tie takes the neighbouring int8 code, so the tolerance there is RMS 2e-4, worst row 1e-2.
//   - `double`: the unrounded activations - the int8 activation rounding every MMQ format has.  Tolerance: RMS 2.5e-2.
// The per-expert gather and the grouped gather must give bitwise equal outputs, and a token's output rows must be
// bitwise equal when run in a half-width chunk with the tokens at different positions.  An all-zero token stays
// exactly 0.  Every error line reports the max absolute error and the max absolute reference next to the rms and
// the worst-row relative error.
// Then a microbench line: one layer at a 512-token chunk (128 experts, top 6) - gathers plus products - in us and
// TFLOP/s.  --no-bench.  Exit 77 without a CUDA device.
#include "basalt/prefill/moe_mmq.hpp"

#include "basalt/ggml/ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
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

constexpr int N = 5120, FF = 2304, K = 6, GROUP = 16;
const size_t D_ROW = ggml_row_size(GGML_TYPE_MXFP4, FF);

// One parity/bench run's gate/up format and its blob layout: the expert is [gate | up | down] with gate/up rows of
// `type` (an IQ2 format) and MXFP4 down rows.
struct Gate {
    int type;
    size_t row, up_off, down_off, bytes;
    explicit Gate(int t) : type(t), row(ggml_row_size((ggml_type) t, N)) {
        const size_t gu = row * FF;
        up_off = gu, down_off = 2 * gu, bytes = down_off + D_ROW * N;
    }
    const char* name() const { return ggml_type_name((ggml_type) type); }
};

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

// one [gate | up | down] expert blob: gate/up rows in the gate's IQ2 format (random code bytes, fp16 block scales
// in 2^-13 .. 2^-6) and MXFP4 down rows (random codes, E8M0 scale bytes 118..123)
std::vector<uint8_t> make_blob(const Gate& g, std::mt19937& rng) {
    std::vector<uint8_t> b(g.bytes);
    for (auto& v : b) v = (uint8_t) rng();
    std::uniform_int_distribution<int> ex(3, 10), man(0, 1023), sgn(0, 3);
    for (size_t o = 0; o < g.down_off; o += g.row / (N / 256)) {   // IQ2: the fp16 d at the head of every block
        const uint16_t h = (uint16_t) ((sgn(rng) == 0 ? 0x8000 : 0) | (ex(rng) << 10) | man(rng));
        std::memcpy(b.data() + o, &h, 2);
    }
    for (size_t o = g.down_off; o < g.bytes; o += 17) b[o] = (uint8_t) (118 + rng() % 6);
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

struct Err {
    double rms = 0, worst = 0, maxabs = 0, maxref = 0;
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
        out.maxabs = std::max(out.maxabs, me);
        out.maxref = std::max(out.maxref, mr);
    }
    out.rms = r2 > 0 ? std::sqrt(e2 / r2) : 0;
    return out;
}

struct Layer {
    int T, E;
    std::vector<int32_t> ids;                    // token x top-k expert id
    std::vector<int32_t> cnt, off, src, order;   // per expert; sorted row -> token; experts with rows
    std::vector<int32_t> bh;                     // bounds: absolute per expert in order, then per group relative
};
Layer route_ids(int T, int E, const std::vector<int32_t>& ids) {
    Layer L{T, E, ids, {}, {}, {}, {}, {}};
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
Layer route(int T, int E, std::mt19937& rng, int hot) {
    std::vector<int32_t> ids((size_t) T * K);
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < K; ++k) {
            int e;
            bool dup;
            do {
                e = (hot > 0 && k < 2 && rng() % 4 != 0) ? (int) (rng() % (unsigned) hot) : (int) (rng() % (unsigned) E);
                dup = false;
                for (int j = 0; j < k; ++j) dup |= ids[(size_t) t * K + j] == e;
            } while (dup);
            ids[(size_t) t * K + k] = e;
        }
    return route_ids(T, E, ids);
}
// row of every assignment index (token x k) in a layer: rows fill each expert block in ascending assignment order
std::vector<int32_t> assign_rows(const Layer& L) {
    std::vector<int32_t> fill(L.off.begin(), L.off.end() - 1), r(L.ids.size());
    for (size_t i = 0; i < L.ids.size(); ++i) r[i] = fill[(size_t) L.ids[i]]++;
    return r;
}

struct Bufs {
    Dev xq, gu, h, hq, dm, ident, bounds, grp_gu, grp_d;
    Bufs(int64_t rows, size_t nb, const Gate& gate)
        : xq(mmq::q8_bytes(rows, N)), gu((size_t) rows * 2 * FF * 4), h((size_t) rows * FF * 4),
          hq(mmq::q8_bytes(rows, FF)), dm((size_t) rows * N * 4), ident((size_t) rows * 4), bounds(nb * 4),
          grp_gu(GROUP * mmq::matrix_bytes(gate.type, 2 * FF, N)),
          grp_d(GROUP * mmq::matrix_bytes(GGML_TYPE_MXFP4, N, FF)) {}
};

// expert_mmq.cpp's sequence for one layer; `grouped`: one gather launch per group, else one per expert
void run_mmq(mmq::Context& ctx, Bufs& b, const Gate& gate, const Layer& L, const float* x_dev,
             const int32_t* src_dev, const std::vector<const uint8_t*>& blob, bool grouped, cudaStream_t s) {
    const size_t gub = mmq::matrix_bytes(gate.type, 2 * FF, N), db = mmq::matrix_bytes(GGML_TYPE_MXFP4, N, FF);
    const int64_t rows = (int64_t) L.T * K;
    const size_t n = L.order.size();
    ck(cudaMemcpyAsync(b.bounds.p, L.bh.data(), L.bh.size() * 4, cudaMemcpyHostToDevice, s), "bounds");
    mmq::iota(b.ident.as<int32_t>(), rows, s);
    mmq::quantize(x_dev, src_dev, b.xq.p, gate.type, N, N, rows, s);
    mmq::GatherGroup gg;
    for (size_t j = 0; j < n; ++j) {
        const size_t q = j % GROUP;
        const uint8_t* bl = blob[(size_t) L.order[j]];
        if (grouped) {
            gg.blob[q] = bl;
            gg.n = (int) q + 1;
        } else {
            mmq::gather_native(bl, bl + gate.up_off, gub / 2, bl + gate.down_off, db,
                               b.grp_gu.as<uint8_t>() + q * gub, b.grp_d.as<uint8_t>() + q * db, s);
        }
        if (q + 1 < GROUP && j + 1 < n) continue;
        if (grouped) {
            if (!mmq::gather_native_group(gg, gate.up_off, gub / 2, gate.down_off, db, b.grp_gu.p, gub, b.grp_d.p, db, s))
                throw std::runtime_error("gather_native_group refused aligned blobs");
            gg = mmq::GatherGroup{};
        }
        const size_t j0 = j - q, g = j0 / GROUP;
        const int ngx = (int) (q + 1);
        const int64_t r0 = L.bh[j0], nr = L.bh[j + 1] - r0;
        int64_t maxr = 0;
        for (size_t i = j0; i <= j; ++i) maxr = std::max<int64_t>(maxr, L.cnt[(size_t) L.order[i]]);
        const int32_t* rel = b.bounds.as<int32_t>() + n + 1 + g * (GROUP + 1);
        mmq::Product gu;
        gu.w = b.grp_gu.p; gu.type = gate.type; gu.w_rows = 2 * FF; gu.w_cols = N; gu.expert_bytes = gub;
        gu.n = ngx; gu.xq = b.xq.p; gu.bounds = b.bounds.as<int32_t>() + j0; gu.ids = b.ident.as<int32_t>();
        gu.total_rows = rows; gu.max_rows = maxr; gu.dst = b.gu.as<float>(); gu.ld_dst = 2 * FF;
        ctx.run(gu, s);
        mmq::swiglu(b.gu.as<float>() + r0 * 2 * FF, b.h.as<float>() + r0 * FF, nr, FF, false, s);
        mmq::quantize(b.h.as<float>() + r0 * FF, nullptr, b.hq.p, GGML_TYPE_MXFP4, FF, FF, nr, s);
        mmq::Product dn;
        dn.w = b.grp_d.p; dn.type = GGML_TYPE_MXFP4; dn.w_rows = N; dn.w_cols = FF; dn.expert_bytes = db;
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

int parity(cudaStream_t s, const Gate& gate) {
    constexpr int T = 96, E = 32, ZERO = 33;
    std::mt19937 rng(4041);
    std::vector<std::vector<uint8_t>> host((size_t) E);
    for (int e = 0; e < E; ++e) host[(size_t) e] = make_blob(gate, rng);
    std::vector<std::unique_ptr<Dev>> dev;
    std::vector<const uint8_t*> blob((size_t) E);
    for (int e = 0; e < E; ++e) {
        dev.push_back(std::make_unique<Dev>(gate.bytes));
        ck(cudaMemcpy(dev.back()->p, host[(size_t) e].data(), gate.bytes, cudaMemcpyHostToDevice), "blob");
        blob[(size_t) e] = dev.back()->as<uint8_t>();
    }
    std::vector<float> x((size_t) T * N);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (float& v : x) v = nd(rng) * (rng() % 200 == 0 ? 12.0f : 1.0f);
    std::fill(x.begin() + (size_t) ZERO * N, x.begin() + (size_t) (ZERO + 1) * N, 0.0f);
    const Layer L = route(T, E, rng, 2);   // experts 0, 1 hot: several 64-row tiles
    const int64_t rows = (int64_t) L.T * K;
    std::printf("parity (%s gate/up, MXFP4 down): %d tokens, %d experts, %zu with rows, max rows per expert %d\n",
                gate.name(), T, E, L.order.size(), *std::max_element(L.cnt.begin(), L.cnt.end()));

    Dev x_dev(x.size() * 4), src_dev((size_t) rows * 4);
    ck(cudaMemcpy(x_dev.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(src_dev.p, L.src.data(), L.src.size() * 4, cudaMemcpyHostToDevice), "src");
    ck(cudaDeviceSynchronize(), "uploads");
    mmq::Context ctx;
    Bufs b(rows, L.bh.size(), gate);
    run_mmq(ctx, b, gate, L, x_dev.as<float>(), src_dev.as<int32_t>(), blob, false, s);
    ck(cudaStreamSynchronize(s), "sync");
    const std::vector<float> y1 = download(b.dm, (size_t) rows * N);
    ck(cudaMemset(b.dm.p, 0xff, (size_t) rows * N * 4), "sentinel");
    ck(cudaDeviceSynchronize(), "sentinel");
    run_mmq(ctx, b, gate, L, x_dev.as<float>(), src_dev.as<int32_t>(), blob, true, s);
    ck(cudaStreamSynchronize(s), "sync");
    const std::vector<float> y = download(b.dm, (size_t) rows * N), ygu = download(b.gu, (size_t) rows * 2 * FF);
    const bool same_gather = std::memcmp(y.data(), y1.data(), y.size() * 4) == 0;

    // batch-width and position invariance: the last half of the tokens as a half-width chunk, every token now at
    // new row positions; each token's rows must come out bitwise identical to the full run
    constexpr int T2 = T / 2;
    std::vector<int32_t> ids2((size_t) T2 * K);
    for (int t = 0; t < T2; ++t)
        for (int k = 0; k < K; ++k) ids2[(size_t) t * K + k] = L.ids[(size_t) (T - T2 + t) * K + k];
    const Layer L2 = route_ids(T2, E, ids2);
    const int64_t rows2 = (int64_t) T2 * K;
    Dev x2_dev((size_t) T2 * N * 4), src2_dev((size_t) rows2 * 4);
    ck(cudaMemcpy(x2_dev.p, x.data() + (size_t) (T - T2) * N, (size_t) T2 * N * 4, cudaMemcpyHostToDevice), "x2");
    ck(cudaMemcpy(src2_dev.p, L2.src.data(), L2.src.size() * 4, cudaMemcpyHostToDevice), "src2");
    ck(cudaDeviceSynchronize(), "uploads2");
    Bufs b2(rows2, L2.bh.size(), gate);
    run_mmq(ctx, b2, gate, L2, x2_dev.as<float>(), src2_dev.as<int32_t>(), blob, true, s);
    ck(cudaStreamSynchronize(s), "sync2");
    const std::vector<float> y2 = download(b2.dm, (size_t) rows2 * N);
    const std::vector<int32_t> pos = assign_rows(L), pos2 = assign_rows(L2);
    bool width_ok = true;
    for (int t = T - T2; t < T && width_ok; ++t)
        for (int k = 0; k < K; ++k) {
            const size_t ia = (size_t) t * K + k, ib = (size_t) (t - (T - T2)) * K + k;
            width_ok &= std::memcmp(y.data() + (size_t) pos[ia] * N, y2.data() + (size_t) pos2[ib] * N,
                                    (size_t) N * 4) == 0;
        }

    // the references, an expert at a time on all cores
    std::vector<float> r_same((size_t) rows * N), r_dbl((size_t) rows * N), r_gu((size_t) rows * 2 * FF);
    const auto* tr_gu = ggml_get_type_traits((ggml_type) gate.type);
    const auto* tr_d = ggml_get_type_traits(GGML_TYPE_MXFP4);
    parallel(E, [&](int e) {
        if (L.cnt[(size_t) e] == 0) return;
        std::vector<float> wgu((size_t) 2 * FF * N), wd((size_t) N * FF);
        for (int r = 0; r < 2 * FF; ++r)
            tr_gu->to_float(host[(size_t) e].data() + (size_t) r * gate.row, wgu.data() + (size_t) r * N, N);
        for (int r = 0; r < N; ++r)
            tr_d->to_float(host[(size_t) e].data() + gate.down_off + (size_t) r * D_ROW,
                           wd.data() + (size_t) r * FF, FF);
        std::vector<float> xr(N), h(FF), hr(FF), hd(FF);
        for (int32_t p = L.off[(size_t) e]; p < L.off[(size_t) e + 1]; ++p) {
            const float* xt = x.data() + (size_t) L.src[(size_t) p] * N;
            q8_round(xt, xr.data(), N);
            for (int k = 0; k < FF; ++k) {
                const float* wg = wgu.data() + (size_t) k * N;
                const float* wu = wgu.data() + (size_t) (FF + k) * N;
                double g = 0, u = 0, gd = 0, ud = 0;
                for (int c = 0; c < N; ++c) {
                    g += (double) wg[c] * xr[(size_t) c];
                    u += (double) wu[c] * xr[(size_t) c];
                    gd += (double) wg[c] * xt[c];
                    ud += (double) wu[c] * xt[c];
                }
                r_gu[(size_t) p * 2 * FF + k] = (float) g;
                r_gu[(size_t) p * 2 * FF + FF + k] = (float) u;
                h[(size_t) k] = (float) (g / (1.0 + std::exp(-g)) * u);    // H on MMQ's rounded x
                hd[(size_t) k] = (float) (gd / (1.0 + std::exp(-gd)) * ud);   // H on the unrounded x
            }
            q8_round(h.data(), hr.data(), FF);   // MMQ rounds H itself before the down product
            float* ys = r_same.data() + (size_t) p * N;
            float* yd = r_dbl.data() + (size_t) p * N;
            for (int o = 0; o < N; ++o) {
                const float* w = wd.data() + (size_t) o * FF;
                double a = 0, ad = 0;
                for (int kk = 0; kk < FF; ++kk) {
                    a += (double) w[kk] * hr[(size_t) kk];
                    ad += (double) w[kk] * hd[(size_t) kk];
                }
                ys[o] = (float) a;
                yd[o] = (float) ad;
            }
        }
    });
    const Err es = compare(y, r_same, (size_t) rows), ed = compare(y, r_dbl, (size_t) rows);
    double ge2 = 0, gr2 = 0, gworst = 0, gmax = 0, gmaxref = 0;
    for (int64_t p = 0; p < rows; ++p) {
        double me = 0, mr = 0;
        for (int c = 0; c < 2 * FF; ++c) {
            const double d = (double) ygu[(size_t) p * 2 * FF + c] - r_gu[(size_t) p * 2 * FF + c];
            ge2 += d * d;
            gr2 += (double) r_gu[(size_t) p * 2 * FF + c] * r_gu[(size_t) p * 2 * FF + c];
            me = std::max(me, std::fabs(d));
            mr = std::max(mr, (double) std::fabs(r_gu[(size_t) p * 2 * FF + c]));
        }
        if (mr > 0) gworst = std::max(gworst, me / mr);
        gmax = std::max(gmax, me);
        gmaxref = std::max(gmaxref, mr);
    }
    const double grms = gr2 > 0 ? std::sqrt(ge2 / gr2) : 0;
    bool zero_ok = true;
    for (int64_t p = 0; p < rows; ++p)
        if (L.src[(size_t) p] == ZERO)
            for (int o = 0; o < N; ++o) zero_ok &= y[(size_t) p * N + o] == 0.0f;
    std::printf("  gate/up vs same-rounding reference: max abs %.3e (max |ref| %.3e)  "
                "rms %.3e  worst row %.3e   (tol 1e-5, 1e-4)\n",
                gmax, gmaxref, grms, gworst);
    std::printf("  MMQ vs same-rounding reference: max abs %.3e (max |ref| %.3e)  "
                "rms %.3e  worst row %.3e   (tol 2e-4, 1e-2)\n",
                es.maxabs, es.maxref, es.rms, es.worst);
    std::printf("  MMQ vs double reference:        max abs %.3e (max |ref| %.3e)  "
                "rms %.3e  worst row %.3e   (tol rms 2.5e-2)\n",
                ed.maxabs, ed.maxref, ed.rms, ed.worst);
    std::printf("  grouped gather == per-expert gather (bitwise): %s; zero token exactly 0: %s\n",
                same_gather ? "yes" : "NO", zero_ok ? "yes" : "NO");
    std::printf("  width/position invariance (bitwise): %s\n", width_ok ? "yes" : "NO");
    const bool ok = grms <= 1e-5 && gworst <= 1e-4 && es.rms <= 2e-4 && es.worst <= 1e-2 && ed.rms <= 2.5e-2 &&
                    same_gather && zero_ok && width_ok;
    std::printf("parity (%s): %s\n", gate.name(), ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

void bench(cudaStream_t s, const Gate& gate) {
    constexpr int T = 512, E = 128;
    std::mt19937 rng(78);
    // one synthetic blob copied to every slot (timing only)
    const std::vector<uint8_t> h = make_blob(gate, rng);
    const size_t stride = (gate.bytes + 255) / 256 * 256;
    Dev all(stride * E);
    for (int e = 0; e < E; ++e)
        ck(cudaMemcpy(all.as<uint8_t>() + stride * e, h.data(), gate.bytes, cudaMemcpyHostToDevice), "blob");
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
    Bufs b(rows, L.bh.size(), gate);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int w = 0; w < 2; ++w) run_mmq(ctx, b, gate, L, x_dev.as<float>(), src_dev.as<int32_t>(), blob, true, s);
    constexpr int R = 10;
    cudaEventRecord(e0, s);
    for (int r = 0; r < R; ++r) run_mmq(ctx, b, gate, L, x_dev.as<float>(), src_dev.as<int32_t>(), blob, true, s);
    cudaEventRecord(e1, s);
    ck(cudaEventSynchronize(e1), "bench");
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    const double us = 1e3 * ms / R, flop = 2.0 * (double) rows * (2.0 * FF * N + (double) N * FF);
    std::printf("bench: %s/MXFP4 MMQ layer, %d tokens x top %d over %d experts (%zu with rows): %.1f us, "
                "%.1f TFLOP/s (gathers + q8_1 + gate/up + SwiGLU + down)\n", gate.name(), T, K, E, L.order.size(), us,
                flop / us * 1e-6);
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
        if (!mmq::fits(GGML_TYPE_IQ2_XXS, 2 * FF) || !mmq::fits(GGML_TYPE_MXFP4, N)) {
            std::printf("MMQ has no IQ2_XXS / MXFP4 tile on this device\n");
            return 1;
        }
        cudaStream_t s;
        ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        int rc = 0;
        // the three IQ2 gate/up formats the UD file's MXFP4 layers pair with MXFP4 down
        for (const ggml_type gt : {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S}) {
            const Gate gate(gt);
            if (!mmq::fits(gate.type, 2 * FF)) {   // IQ2_XXS failed above; a card without the other tiles skips them
                std::printf("%s gate/up: MMQ has no tile on this device: skipped\n", gate.name());
                continue;
            }
            rc |= parity(s, gate);
        }
        if (do_bench) bench(s, Gate(GGML_TYPE_IQ2_XXS));
        cudaStreamDestroy(s);
        return rc;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "prefill_mmq_mxfp4_test: %s\n", ex.what());
        return 1;
    }
}
