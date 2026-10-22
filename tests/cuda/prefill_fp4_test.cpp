// prefill_fp4_test - the prompt path's NVFP4 experts on the FP4 block-scaled tensor cores (moe_fp4.hpp) next to the MMQ
// path they replace, at the real geometry (gate/up 1280 x 2560, down 2560 x 640) on synthetic NVFP4 experts (sub-block
// scales over 13 binades, UE4M3 subnormals included; tails {s_gate, s_up, s_down} in 0.004..0.03), in prefill.cpp's
// sequence (16-expert groups, activations to codes, gate/up, SwiGLU, H to codes, down).  Checked against CPU references:
//   - FP4: the same A4 roundings (token scale amax / (6 * 448), UE4M3 per 16, E2M1 codes) in double: gate/up RMS 1e-5,
//     down RMS 1e-4; and an output that does not depend on the token's tile or neighbours (bitwise).  The unrounded
//     activations in double: layer output RMS 0.2 (the W4A4 rounding error itself).  The MMQ output: RMS 0.2.
//   - MMQ: `same`, double-precision math on MMQ's own int8 roundings (gate/up RMS 1e-5, worst row 1e-4; after the down
//     product RMS 2e-4, worst row 1e-2); `double` on the unrounded activations (RMS 2.5e-2); the FP16 fallback it
//     replaced (RMS 2e-2).  The per-expert gather and the grouped gather give bitwise equal outputs; a zero token stays 0.
// The production gate/up reads each row's token from token planes (fp4::quantize_tokens) and leaves the tail-scaled
// SwiGLU from its epilogue: the planes must hold the bytes quantize() puts in the tiles, H the bits of the plain
// product followed by mmq::swiglu (and the A4 reference in double to RMS 1e-5).  The group path (`fp4::run_group`, what
// prefill.cpp runs) with H holding only a run's rows must equal the absolute-row sequence of the plain kernels bitwise,
// and so must every split of a group into runs (a row cap from one expert to none), on two routings with hot experts,
// with nothing written past the run's rows.  Edge activations: zero sub-blocks
// next to non-zero ones, tokens scaled by 1e-30 and 1e30, one huge sub-block in a tiny token: finite, zero stays zero,
// and the gate/up product follows the A4 reference.  Then a microbench line (--no-bench skips it).  With a second
// device, the peer card's share of a streamed chunk (its experts' rows in pieces of a row cap, each group planned on
// its own copy of the activations) must equal the primary's whole-layer rows bit for bit, at 512 experts and top 10,
// with a timing line for the peer.  Exit 77 without a CUDA device.
#include "basalt/prefill/moe_fp4.hpp"
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
#include <utility>
#include <vector>

namespace {
namespace mmq = basalt::prefill::mmq;
namespace fp4 = basalt::prefill::fp4;

constexpr int N = 2560, FF = 640, K = 10, GROUP = 16;
constexpr size_t GU_ROW = (size_t) N / 64 * 36, D_ROW = (size_t) FF / 64 * 36;
constexpr size_t UP_OFF = GU_ROW * FF, DOWN_OFF = 2 * UP_OFF, TAIL_OFF = DOWN_OFF + D_ROW * N, BYTES = TAIL_OFF + 16;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Dev {
    void* p = nullptr;
    size_t bytes = 0;
    explicit Dev(size_t n) : bytes(n) { ck(cudaMalloc(&p, n), "cudaMalloc"); }
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

// the FP4 activation rounding (quantize_kernel in moe_fp4.cu): token scale g = amax / (6 * 448), per 16 values a UE4M3
// scale of block amax / (6 g), codes rounded to nearest even on x / (scale * g)
float e4m3_round(float v) {
    static float tab[127];
    static bool init = false;
    if (!init) {
        for (int c = 0; c < 127; ++c) {
            const int e = c >> 3, m = c & 7;
            tab[c] = e == 0 ? std::ldexp((float) m, -9) : std::ldexp(1.0f + m / 8.0f, e - 7);
        }
        init = true;
    }
    if (!(v > 0.0f)) return 0.0f;
    if (v >= 448.0f) return 448.0f;
    int lo = 0, hi = 126;   // the largest code at or below v
    while (hi > lo) {
        const int mid = (lo + hi + 1) >> 1;
        if (tab[mid] <= v) lo = mid; else hi = mid - 1;
    }
    if (lo == 126) return tab[lo];
    const float a = tab[lo], b = tab[lo + 1], mid = 0.5f * (a + b);
    return v < mid ? a : v > mid ? b : (lo & 1 ? b : a);
}
float e2m1_round(float v) {
    static const float val[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float a = std::fabs(v);
    int c = 7;
    for (int i = 0; i < 7; ++i) {
        const float mid = 0.5f * (val[i] + val[i + 1]);
        if (a < mid || (a == mid && (i & 1) == 0)) { c = i; break; }   // a tie goes to the even code
    }
    return v < 0 ? -val[c] : val[c];
}
void a4_round(const float* x, float* y, int n) {
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) amax = std::max(amax, std::fabs(x[i]));
    const float g = amax * (1.0f / (6.0f * 448.0f)), inv_g = amax > 0.0f ? 1.0f / g : 0.0f;
    for (int b = 0; b < n; b += 16) {
        float bm = 0.0f;
        for (int j = 0; j < 16; ++j) bm = std::max(bm, std::fabs(x[b + j]));
        const float sd = e4m3_round(bm * inv_g * (1.0f / 6.0f));
        const float inv = sd > 0.0f ? 1.0f / (sd * g) : 0.0f;
        for (int j = 0; j < 16; ++j) y[b + j] = e2m1_round(x[b + j] * inv) * sd * g;
    }
}

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

// prefill.cpp's sequence for one layer; `grouped`: one gather launch per group, else one per expert
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
// FNV-1a over a float array's bytes: a build's outputs on this layer, to compare builds bit for bit
uint64_t fnv(const std::vector<float>& v) {
    uint64_t h = 1469598103934665603ull;
    const auto* b = (const unsigned char*) v.data();
    for (size_t i = 0; i < v.size() * 4; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

// the same layer through the FP4 tensor-core path
struct F4 {
    Dev a4x, gsx, tox, a4h, gsh, toh;
    F4(int64_t rows, size_t n)
        : a4x(fp4::a4_bytes(rows, N, (int) n)), gsx(fp4::scale_count(rows, (int) n) * 4), tox((n + 1) * 4),
          a4h(fp4::a4_bytes(rows, FF, GROUP)), gsh(fp4::scale_count(rows, GROUP) * 4), toh((GROUP + 1) * 4) {}
};
void run_fp4(F4& f, Bufs& b, const Layer& L, const float* x_dev, const int32_t* src_dev,
             const std::vector<const uint8_t*>& blob, cudaStream_t s) {
    const int64_t rows = (int64_t) L.T * K;
    const size_t n = L.order.size();
    ck(cudaMemcpyAsync(b.bounds.p, L.bh.data(), L.bh.size() * 4, cudaMemcpyHostToDevice, s), "bounds");
    fp4::plan(b.bounds.as<int32_t>(), (int) n, f.tox.as<int32_t>(), s);
    fp4::quantize(x_dev, src_dev, b.bounds.as<int32_t>(), f.tox.as<int32_t>(), (int) n, f.a4x.p, f.gsx.as<float>(), N, N,
                  rows, s);
    for (size_t j0 = 0; j0 < n; j0 += GROUP) {
        const size_t j1 = std::min(n, j0 + GROUP), g = j0 / GROUP;
        const int ngx = (int) (j1 - j0);
        const int64_t r0 = L.bh[j0], nr = L.bh[j1] - r0;
        int64_t maxr = 0;
        for (size_t i = j0; i < j1; ++i) maxr = std::max<int64_t>(maxr, L.cnt[(size_t) L.order[i]]);
        const int32_t* rel = b.bounds.as<int32_t>() + n + 1 + g * (GROUP + 1);
        fp4::Product gu;
        for (int q = 0; q < ngx; ++q) gu.blob[q] = blob[(size_t) L.order[j0 + (size_t) q]];
        gu.w_off = 0; gu.w_rows = 1280; gu.w_cols = N; gu.n = ngx; gu.a4 = f.a4x.p; gu.gs = f.gsx.as<float>();
        gu.bounds = b.bounds.as<int32_t>() + j0; gu.tile_off = f.tox.as<int32_t>() + j0; gu.dst = b.gu.as<float>();
        gu.ld_dst = 1280; gu.max_rows = maxr;
        fp4::run(gu, s);
        fp4::gather_tails(gu, TAIL_OFF, b.scales(), s);
        mmq::swiglu(b.gu.as<float>() + r0 * 1280, b.h.as<float>() + r0 * FF, nr, FF, false, s, b.scales(), rel, ngx);
        fp4::plan(rel, ngx, f.toh.as<int32_t>(), s);
        fp4::quantize(b.h.as<float>() + r0 * FF, nullptr, rel, f.toh.as<int32_t>(), ngx, f.a4h.p, f.gsh.as<float>(), FF, FF,
                      nr, s);
        fp4::Product dn = gu;
        dn.w_off = DOWN_OFF; dn.w_rows = N; dn.w_cols = FF; dn.a4 = f.a4h.p; dn.gs = f.gsh.as<float>(); dn.bounds = rel;
        dn.tile_off = f.toh.as<int32_t>(); dn.dst = b.dm.as<float>() + r0 * N; dn.ld_dst = N;
        fp4::run(dn, s);
    }
}

// The token planes against the tiles: every row's codes, sub-block scales and token scale must be the bytes quantize()
// put in its tile (the tile's 16-byte code chunks swizzled by row).
bool token_planes_match(const F4& f, const Layer& L, const float* x_dev, const int32_t* src_dev, cudaStream_t s) {
    const int64_t rows = (int64_t) L.T * K;
    const size_t n = L.order.size();
    constexpr int NKT = N / 128, ST = 4608;
    Dev a4t(fp4::token_bytes(L.T, N)), gst((size_t) L.T * 4), bounds((n + 1) * 4);
    ck(cudaMemcpyAsync(bounds.p, L.bh.data(), (n + 1) * 4, cudaMemcpyHostToDevice, s), "bounds");
    fp4::plan(bounds.as<int32_t>(), (int) n, f.tox.as<int32_t>(), s);
    fp4::quantize(x_dev, src_dev, bounds.as<int32_t>(), f.tox.as<int32_t>(), (int) n, f.a4x.p, f.gsx.as<float>(), N, N,
                  rows, s);
    fp4::quantize_tokens(x_dev, L.T, N, N, a4t.p, gst.as<float>(), s);
    ck(cudaStreamSynchronize(s), "sync planes");
    std::vector<uint8_t> tiles(f.a4x.bytes), tok(a4t.bytes);
    std::vector<float> gs(f.gsx.bytes / 4), gt((size_t) L.T);
    std::vector<int32_t> tox(n + 1);
    ck(cudaMemcpy(tiles.data(), f.a4x.p, tiles.size(), cudaMemcpyDeviceToHost), "d");
    ck(cudaMemcpy(tok.data(), a4t.p, tok.size(), cudaMemcpyDeviceToHost), "d");
    ck(cudaMemcpy(gs.data(), f.gsx.p, gs.size() * 4, cudaMemcpyDeviceToHost), "d");
    ck(cudaMemcpy(gt.data(), gst.p, gt.size() * 4, cudaMemcpyDeviceToHost), "d");
    ck(cudaMemcpy(tox.data(), f.tox.p, tox.size() * 4, cudaMemcpyDeviceToHost), "d");
    const uint8_t* scales = tok.data() + (size_t) L.T * (N / 2);
    int64_t bad = 0;
    for (size_t j = 0; j < n; ++j)
        for (int32_t r = L.bh[j]; r < L.bh[j + 1]; ++r) {
            const int64_t pos = (int64_t) tox[j] * 64 + (r - L.bh[j]), tile = pos / 64;
            const int rr = (int) (pos % 64);
            const int32_t t = L.src[(size_t) r];
            bad += std::memcmp(&gs[(size_t) pos], &gt[(size_t) t], 4) != 0;
            for (int kt = 0; kt < NKT; ++kt) {
                const uint8_t* tp = tiles.data() + (size_t) (tile * NKT + kt) * ST;
                for (int q = 0; q < 4; ++q)
                    bad += std::memcmp(tp + rr * 64 + ((q ^ ((rr >> 1) & 3)) << 4),
                                       tok.data() + (size_t) t * (N / 2) + kt * 64 + 16 * q, 16) != 0;
                bad += std::memcmp(tp + 4096 + rr * 8, scales + (size_t) t * (N / 16) + kt * 8, 8) != 0;
            }
        }
    std::printf("  token planes == the tiles' codes, scales and token scales (bitwise, %lld rows): %s\n", (long long) rows,
                bad == 0 ? "yes" : "NO");
    return bad == 0;
}

// The gate/up product from the token planes with the SwiGLU in its epilogue, every group at once into absolute H rows.
void run_fused_h(const F4& f, Bufs& b, const Layer& L, const float* x_dev, const int32_t* src_dev,
                 const std::vector<const uint8_t*>& blob, float* h_abs, cudaStream_t s) {
    const size_t n = L.order.size();
    ck(cudaMemcpyAsync(b.bounds.p, L.bh.data(), L.bh.size() * 4, cudaMemcpyHostToDevice, s), "bounds");
    fp4::quantize_tokens(x_dev, L.T, N, N, f.a4x.p, f.gsx.as<float>(), s);
    for (size_t j0 = 0; j0 < n; j0 += GROUP) {
        const size_t j1 = std::min(n, j0 + GROUP);
        int64_t maxr = 0;
        for (size_t i = j0; i < j1; ++i) maxr = std::max<int64_t>(maxr, L.cnt[(size_t) L.order[i]]);
        fp4::Product gu;
        for (size_t q = j0; q < j1; ++q) gu.blob[q - j0] = blob[(size_t) L.order[q]];
        gu.w_off = 0; gu.w_rows = 1280; gu.w_cols = N; gu.n = (int) (j1 - j0); gu.a4 = f.a4x.p; gu.gs = f.gsx.as<float>();
        gu.bounds = b.bounds.as<int32_t>() + j0; gu.ids = src_dev; gu.tokens = L.T; gu.swiglu = true; gu.tail_off = TAIL_OFF;
        gu.dst = h_abs; gu.ld_dst = FF; gu.max_rows = maxr;
        fp4::run(gu, s);
    }
}

// The production sequence: `fp4::run_group` over each group of up to 16 experts, GU and H holding only a run's rows,
// in runs of at most `row_cap` rows.  Buffers past the run rows carry guard bytes that must stay untouched.
constexpr size_t kGuardRows = 64, kGuardBytes = 4096;
struct F4Run {
    int64_t cap_rows;
    Dev h, a4h, gsh, th;
    F4Run(int64_t rows)
        : cap_rows(rows), h((size_t) (rows + kGuardRows) * FF * 4), a4h(fp4::a4_bytes(rows, FF, GROUP) + kGuardBytes),
          gsh(fp4::scale_count(rows, GROUP) * 4 + kGuardBytes), th(2 * (GROUP + 1) * 4) {}
};
bool guard_clear(const void* base, size_t from, size_t to) {
    std::vector<uint8_t> v(to - from);
    ck(cudaMemcpy(v.data(), (const uint8_t*) base + from, v.size(), cudaMemcpyDeviceToHost), "guard");
    for (uint8_t c : v) if (c != 0xff) return false;
    return true;
}
// prefill.cpp's FP4 sequence for one layer: the token planes, then each group in runs of at most `row_cap` rows
int enqueue_fp4_group(F4& f, F4Run& r, Bufs& b, const Layer& L, const float* x_dev, const int32_t* src_dev,
                      const std::vector<const uint8_t*>& blob, int64_t row_cap, cudaStream_t s) {
    const size_t n = L.order.size();
    ck(cudaMemcpyAsync(b.bounds.p, L.bh.data(), L.bh.size() * 4, cudaMemcpyHostToDevice, s), "bounds");
    fp4::quantize_tokens(x_dev, L.T, N, N, f.a4x.p, f.gsx.as<float>(), s);
    int runs = 0;
    for (size_t j0 = 0; j0 < n; j0 += GROUP) {
        const int ngx = (int) std::min<size_t>(GROUP, n - j0);
        fp4::Group g;
        g.n = ngx;
        for (int q = 0; q < ngx; ++q) g.blob[q] = blob[(size_t) L.order[j0 + (size_t) q]];
        g.bounds_host = L.bh.data() + j0;
        g.bounds_dev = b.bounds.as<int32_t>() + j0;
        g.ids = src_dev; g.a4x = f.a4x.p; g.gsx = f.gsx.as<float>(); g.tokens = L.T;
        g.down_off = DOWN_OFF; g.tail_off = TAIL_OFF;
        g.h = r.h.as<float>(); g.dm = b.dm.as<float>();
        g.a4h = r.a4h.p; g.gsh = r.gsh.as<float>(); g.toh = r.th.as<int32_t>(); g.rel = g.toh + (GROUP + 1);
        runs += fp4::run_group(g, row_cap, s);
    }
    return runs;
}
// returns the number of runs; `guards` is cleared when anything wrote past a run's rows
int run_fp4_group(F4& f, F4Run& r, Bufs& b, const Layer& L, const float* x_dev, const int32_t* src_dev,
                  const std::vector<const uint8_t*>& blob, int64_t row_cap, bool& guards, cudaStream_t s) {
    ck(cudaMemsetAsync(r.h.p, 0xff, r.h.bytes, s), "guard");
    ck(cudaMemsetAsync(r.a4h.p, 0xff, r.a4h.bytes, s), "guard");
    ck(cudaMemsetAsync(r.gsh.p, 0xff, r.gsh.bytes, s), "guard");
    const int runs = enqueue_fp4_group(f, r, b, L, x_dev, src_dev, blob, row_cap, s);
    ck(cudaStreamSynchronize(s), "sync group");
    guards = guard_clear(r.h.p, (size_t) r.cap_rows * FF * 4, r.h.bytes) &&
             guard_clear(r.a4h.p, fp4::a4_bytes(r.cap_rows, FF, GROUP), r.a4h.bytes) &&
             guard_clear(r.gsh.p, fp4::scale_count(r.cap_rows, GROUP) * 4, r.gsh.bytes);
    return runs;
}

// Every split of a group into runs must give the absolute-row sequence's output bit for bit.
bool split_equivalence(const char* name, const Layer& L, const float* x_dev, const std::vector<const uint8_t*>& blob,
                       cudaStream_t s) {
    const int64_t rows = (int64_t) L.T * K;
    const size_t n = L.order.size(), groups = (n + GROUP - 1) / GROUP;
    Dev src_dev((size_t) rows * 4);
    ck(cudaMemcpy(src_dev.p, L.src.data(), L.src.size() * 4, cudaMemcpyHostToDevice), "src");
    Bufs b(rows, L.bh.size());
    F4 f(rows, n);
    ck(cudaMemset(b.dm.p, 0xff, (size_t) rows * N * 4), "clr");
    run_fp4(f, b, L, x_dev, src_dev.as<int32_t>(), blob, s);
    ck(cudaStreamSynchronize(s), "sync abs");
    const std::vector<float> ref = download(b.dm, (size_t) rows * N);
    int64_t maxcnt = 0;
    for (int32_t c : L.cnt) maxcnt = std::max<int64_t>(maxcnt, c);
    bool ok = true;
    int runs_unsplit = 0, runs_small = 0;
    for (int64_t cap : {(int64_t) 1 << 40, (int64_t) 2 * 64 + 1, (int64_t) 65, maxcnt, (int64_t) 1}) {
        F4Run r(std::min<int64_t>(rows, std::max(cap, maxcnt)));
        ck(cudaMemset(b.dm.p, 0xff, (size_t) rows * N * 4), "clr");
        bool guards = true;
        const int runs = run_fp4_group(f, r, b, L, x_dev, src_dev.as<int32_t>(), blob, cap, guards, s);
        const std::vector<float> y = download(b.dm, (size_t) rows * N);
        const bool same = std::memcmp(y.data(), ref.data(), ref.size() * 4) == 0;
        if (cap > rows) runs_unsplit = runs;
        if (cap == 65) runs_small = runs;
        std::printf("  %s: cap %lld rows: %d runs over %zu groups, bitwise equal to the absolute-row sequence: %s, guards: %s\n",
                    name, (long long) std::min<int64_t>(cap, rows), runs, groups, same ? "yes" : "NO", guards ? "clear" : "WRITTEN");
        ok &= same && guards;
        if (cap == 1) ok &= runs == (int) n;
    }
    ok &= runs_unsplit == (int) groups && runs_small > (int) groups;   // the split really happened, and only when capped
    return ok;
}

// Edge activations through the quantizer and the gate/up product (the A4 reference in double), and the whole group path
// on those without the 1e30 tokens (SwiGLU of a 1e30 gate and up overflows by itself).
bool edge_activations(const std::vector<std::vector<uint8_t>>& host, const std::vector<const uint8_t*>& blob,
                      cudaStream_t s) {
    constexpr int R = 8;   // tokens: 0 zero, 1 zero sub-blocks between non-zero, 2 first half zero, 3 x 1e-30, 4 x 1e30,
                           // 5 one 1e30 sub-block in a 1e-30 token, 6 normal, 7 a lone 1e-30 value in a zero token
    std::mt19937 rng(99);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t) R * N);
    for (float& v : x) v = nd(rng);
    for (int c = 0; c < N; ++c) {
        float* t0 = x.data();
        float* t1 = x.data() + (size_t) N;
        float* t2 = x.data() + (size_t) 2 * N;
        float* t3 = x.data() + (size_t) 3 * N;
        float* t4 = x.data() + (size_t) 4 * N;
        float* t5 = x.data() + (size_t) 5 * N;
        float* t7 = x.data() + (size_t) 7 * N;
        t0[c] = 0.0f;
        if ((c / 16) % 2 == 1) t1[c] = 0.0f;
        if (c < N / 2) t2[c] = 0.0f;
        t3[c] *= 1e-30f;
        t4[c] *= 1e30f;
        t5[c] *= (c / 16 == 7) ? 1e30f : 1e-30f;
        t7[c] = c == 1234 ? 1e-30f : 0.0f;
    }
    Dev x_dev((size_t) R * N * 4);
    ck(cudaMemcpy(x_dev.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    const int32_t bounds[2] = {0, R};
    Dev bd(8);
    ck(cudaMemcpy(bd.p, bounds, 8, cudaMemcpyHostToDevice), "bounds");
    F4 f(R, 1);
    fp4::plan(bd.as<int32_t>(), 1, f.tox.as<int32_t>(), s);
    fp4::quantize(x_dev.as<float>(), nullptr, bd.as<int32_t>(), f.tox.as<int32_t>(), 1, f.a4x.p, f.gsx.as<float>(), N, N, R, s);
    Dev gu((size_t) R * 1280 * 4);
    ck(cudaMemset(gu.p, 0xff, gu.bytes), "clr");
    fp4::Product p;
    p.blob[0] = blob[0];
    p.w_off = 0; p.w_rows = 1280; p.w_cols = N; p.n = 1;
    p.a4 = f.a4x.p; p.gs = f.gsx.as<float>(); p.bounds = bd.as<int32_t>(); p.tile_off = f.tox.as<int32_t>();
    p.dst = gu.as<float>(); p.ld_dst = 1280; p.max_rows = R;
    fp4::run(p, s);
    ck(cudaStreamSynchronize(s), "sync edge");
    const std::vector<float> y = download(gu, (size_t) R * 1280);
    std::vector<float> w((size_t) 2 * FF * N);
    const auto* tr = ggml_get_type_traits(GGML_TYPE_NVFP4);
    for (int r = 0; r < 2 * FF; ++r) tr->to_float(host[0].data() + (size_t) r * GU_ROW, w.data() + (size_t) r * N, N);
    bool ok = true;
    std::vector<float> x4(N);
    for (int t = 0; t < R; ++t) {
        a4_round(x.data() + (size_t) t * N, x4.data(), N);
        double e2 = 0, r2 = 0;
        bool finite = true, zero = true;
        for (int o = 0; o < 1280; ++o) {
            double ref = 0;
            for (int c = 0; c < N; ++c) ref += (double) w[(size_t) o * N + c] * x4[(size_t) c];
            const float v = y[(size_t) t * 1280 + o];
            finite &= std::isfinite(v);
            zero &= v == 0.0f;
            e2 += ((double) v - ref) * ((double) v - ref);
            r2 += ref * ref;
        }
        const double rel = r2 > 0 ? std::sqrt(e2 / r2) : (e2 > 0 ? 1e30 : 0.0);
        // a token whose A4 reference is exactly zero must give exactly zero
        const bool ref_zero = r2 == 0;
        const bool good = finite && rel <= 1e-4 && (!ref_zero || zero);
        std::printf("  edge token %d: gate/up vs A4 reference rel rms %.3e, finite %s%s -> %s\n", t, rel, finite ? "yes" : "NO",
                    ref_zero ? ", reference exactly zero" : "", good ? "ok" : "FAIL");
        ok &= good;
    }
    // the whole path on the tokens without the huge values
    {
        const int keep[6] = {0, 1, 2, 3, 6, 7};
        constexpr int RK = 6;
        std::vector<float> xk((size_t) RK * N);
        for (int i = 0; i < RK; ++i) std::memcpy(xk.data() + (size_t) i * N, x.data() + (size_t) keep[i] * N, N * 4);
        Dev xk_dev(xk.size() * 4);
        ck(cudaMemcpy(xk_dev.p, xk.data(), xk.size() * 4, cudaMemcpyHostToDevice), "x");
        Layer L;
        L.T = RK;
        L.E = 1;
        L.cnt = {RK};
        L.off = {0, RK};
        L.src.resize(RK);
        for (int i = 0; i < RK; ++i) L.src[(size_t) i] = i;
        L.order = {0};
        L.bh = {0, RK, 0, RK};
        // rows = T * K in the helpers: this layer has RK rows, so describe it with K = 1 rows by hand
        Dev src_dev((size_t) RK * 4), bdev(L.bh.size() * 4), dm((size_t) RK * N * 4);
        ck(cudaMemcpy(src_dev.p, L.src.data(), RK * 4, cudaMemcpyHostToDevice), "src");
        ck(cudaMemcpy(bdev.p, L.bh.data(), L.bh.size() * 4, cudaMemcpyHostToDevice), "bounds");
        F4 fk(RK, 1);
        fp4::quantize_tokens(xk_dev.as<float>(), RK, N, N, fk.a4x.p, fk.gsx.as<float>(), s);
        F4Run r(RK);
        ck(cudaMemset(dm.p, 0xff, dm.bytes), "clr");
        fp4::Group g;
        g.n = 1;
        g.blob[0] = blob[0];
        g.bounds_host = L.bh.data();
        g.bounds_dev = bdev.as<int32_t>();
        g.ids = src_dev.as<int32_t>(); g.a4x = fk.a4x.p; g.gsx = fk.gsx.as<float>(); g.tokens = RK;
        g.down_off = DOWN_OFF; g.tail_off = TAIL_OFF;
        g.h = r.h.as<float>(); g.dm = dm.as<float>();
        g.a4h = r.a4h.p; g.gsh = r.gsh.as<float>(); g.toh = r.th.as<int32_t>(); g.rel = g.toh + (GROUP + 1);
        fp4::run_group(g, 1 << 20, s);
        ck(cudaStreamSynchronize(s), "sync edge path");
        const std::vector<float> o = download(dm, (size_t) RK * N);
        for (int i = 0; i < RK; ++i) {
            bool finite = true, zero = true;
            double mx = 0;
            for (int c = 0; c < N; ++c) {
                const float v = o[(size_t) i * N + c];
                finite &= std::isfinite(v);
                zero &= v == 0.0f;
                mx = std::max(mx, (double) std::fabs(v));
            }
            const bool must_zero = keep[i] == 0;   // the all-zero token
            const bool good = finite && (!must_zero || zero) && (keep[i] != 3 || mx < 1e-20);
            std::printf("  edge token %d through the whole path: finite %s, max |out| %.3e%s -> %s\n", keep[i],
                        finite ? "yes" : "NO", mx, must_zero ? " (must be exactly 0)" : "", good ? "ok" : "FAIL");
            ok &= good;
        }
    }
    return ok;
}

// The streamed walk's FP4 sequence on the current device: the experts `order` (absolute row bounds `bh`, n + 1, rows
// laid out by `src`: row -> token) over the `T` tokens' activations rounded once, then run in groups of GROUP experts.
std::vector<float> primary_rows(const std::vector<int32_t>& order, const std::vector<int32_t>& bh, int64_t rows,
                                const float* x_dev, int64_t T, const std::vector<int32_t>& src,
                                const std::vector<const uint8_t*>& blob, int64_t row_cap, cudaStream_t s) {
    const int n = (int) order.size();
    int64_t maxcnt = 0;
    for (int j = 0; j < n; ++j) maxcnt = std::max<int64_t>(maxcnt, bh[(size_t) j + 1] - bh[(size_t) j]);
    const int64_t gr = std::min(rows, std::max(row_cap, maxcnt));   // GU and H hold a run's rows
    Dev src_dev(src.size() * 4 + 4), bd((size_t) (n + 1) * 4), a4x(fp4::token_bytes(T, N)),
        gsx((size_t) T * 4),
        h((size_t) gr * FF * 4), a4h(fp4::a4_bytes(gr, FF, GROUP)), gsh(fp4::scale_count(gr, GROUP) * 4),
        th(2 * (GROUP + 1) * 4), dm((size_t) rows * N * 4);
    ck(cudaMemcpy(src_dev.p, src.data(), src.size() * 4, cudaMemcpyHostToDevice), "src");
    ck(cudaMemcpy(bd.p, bh.data(), (size_t) (n + 1) * 4, cudaMemcpyHostToDevice), "bounds");
    ck(cudaMemset(dm.p, 0xff, dm.bytes), "clr");
    fp4::quantize_tokens(x_dev, T, N, N, a4x.p, gsx.as<float>(), s);
    for (int j0 = 0; j0 < n; j0 += GROUP) {
        fp4::Group g;
        g.n = std::min(GROUP, n - j0);
        for (int q = 0; q < g.n; ++q) g.blob[q] = blob[(size_t) order[(size_t) (j0 + q)]];
        g.bounds_host = bh.data() + j0;
        g.bounds_dev = bd.as<int32_t>() + j0;
        g.ids = src_dev.as<int32_t>(); g.a4x = a4x.p; g.gsx = gsx.as<float>(); g.tokens = T;
        g.down_off = DOWN_OFF; g.tail_off = TAIL_OFF;
        g.h = h.as<float>(); g.dm = dm.as<float>();
        g.a4h = a4h.p; g.gsh = gsh.as<float>(); g.toh = th.as<int32_t>(); g.rel = g.toh + (GROUP + 1);
        fp4::run_group(g, row_cap, s);
    }
    ck(cudaStreamSynchronize(s), "sync primary rows");
    return download(dm, (size_t) rows * N);
}

// The peer card's sequence (prefill.cpp, no direct access between the cards): its experts `order` (bounds `bh`
// relative to its block, rows laid out by `src`) in pieces of at most `cap` rows (fp4::cut_pieces), from the `T`
// tokens' activations rounded once, into buffers of `cap` rows.  `ms`: the compute alone, timed again.
std::vector<float> peer_rows(const std::vector<int32_t>& order, const std::vector<int32_t>& bh, int64_t rows,
                             const float* x_dev, int64_t T, const std::vector<int32_t>& src,
                             const std::vector<const uint8_t*>& blob, int64_t cap, cudaStream_t s, float& ms,
                             int& npieces, int& ngroups) {
    fp4::Pieces pc;
    fp4::cut_pieces(bh.data(), (int) order.size(), cap, pc);
    ngroups = (int) pc.group.size() - 1;
    npieces = (int) pc.expert.size();
    std::vector<int32_t> rel;
    std::vector<size_t> rel_at;
    for (int g = 0; g < ngroups; ++g) {
        rel_at.push_back(rel.size());
        for (int i = pc.group[(size_t) g]; i <= pc.group[(size_t) g + 1]; ++i)
            rel.push_back(pc.bounds[(size_t) i] - pc.bounds[(size_t) pc.group[(size_t) g]]);
    }
    Dev src_dev(src.size() * 4 + 4), rd(rel.size() * 4 + 4), a4x(fp4::token_bytes(T, N)), gsx((size_t) T * 4),
        h((size_t) cap * FF * 4), a4h(fp4::a4_bytes(cap, FF, GROUP)), gsh(fp4::scale_count(cap, GROUP) * 4),
        th(2 * (GROUP + 1) * 4), dm((size_t) cap * N * 4);
    ck(cudaMemcpy(src_dev.p, src.data(), src.size() * 4, cudaMemcpyHostToDevice), "src");
    ck(cudaMemcpy(rd.p, rel.data(), rel.size() * 4, cudaMemcpyHostToDevice), "bounds");
    std::vector<float> out((size_t) rows * N);
    fp4::quantize_tokens(x_dev, T, N, N, a4x.p, gsx.as<float>(), s);
    auto group = [&](int g) {
        const int p0 = pc.group[(size_t) g], ns = pc.group[(size_t) g + 1] - p0;
        const int64_t r0 = pc.bounds[(size_t) p0], nr = pc.bounds[(size_t) p0 + ns] - r0;
        const int32_t* rb = rd.as<int32_t>() + rel_at[(size_t) g];
        fp4::Group fg;
        fg.n = ns;
        for (int i = 0; i < ns; ++i) fg.blob[i] = blob[(size_t) order[(size_t) pc.expert[(size_t) (p0 + i)]]];
        fg.bounds_host = rel.data() + rel_at[(size_t) g];
        fg.bounds_dev = rb;
        fg.ids = src_dev.as<int32_t>() + r0; fg.a4x = a4x.p; fg.gsx = gsx.as<float>(); fg.tokens = T;
        fg.down_off = DOWN_OFF; fg.tail_off = TAIL_OFF;
        fg.h = h.as<float>(); fg.dm = dm.as<float>();
        fg.a4h = a4h.p; fg.gsh = gsh.as<float>(); fg.toh = th.as<int32_t>(); fg.rel = fg.toh + (GROUP + 1);
        if (fp4::run_group(fg, cap, s) != 1) throw std::runtime_error("a peer group took more than one run");
        return std::make_pair(r0, nr);
    };
    for (int g = 0; g < ngroups; ++g) {
        ck(cudaMemsetAsync(dm.p, 0xff, dm.bytes, s), "clr");
        const auto [r0, nr] = group(g);
        ck(cudaMemcpyAsync(out.data() + (size_t) r0 * N, dm.p, (size_t) nr * N * 4, cudaMemcpyDeviceToHost, s), "rows");
        ck(cudaStreamSynchronize(s), "sync peer group");
    }
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "event");
    ck(cudaEventCreate(&e1), "event");
    ck(cudaEventRecord(e0, s), "record");
    for (int g = 0; g < ngroups; ++g) group(g);
    ck(cudaEventRecord(e1, s), "record");
    ck(cudaEventSynchronize(e1), "sync timing");
    ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return out;
}

// The peer card's share of a streamed chunk: every row of the experts it holds, computed there in pieces, and every
// row of the primary's own experts, computed without them, must equal bit for bit the row the primary computes when it
// runs the whole layer (the streamed walk without the peer).  Real geometry: 512 experts, top 10.
struct PeerShape { int T, hot; int64_t cap; };
bool peer_share(int d0, int d1, const PeerShape& sh, const std::vector<const uint8_t*>& blob0,
                const std::vector<const uint8_t*>& blob1, std::mt19937& rng) {
    constexpr int E = 512;
    const int T = sh.T;
    const Layer L = route(T, E, rng, sh.hot);
    const int64_t rows = (int64_t) T * K;
    std::vector<float> x((size_t) T * N);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (float& v : x) v = nd(rng) * (rng() % 200 == 0 ? 12.0f : 1.0f);
    // the peer's experts: every fourth, and expert 0 (hot when the routing has hot experts)
    std::vector<char> on_peer((size_t) E, 0);
    for (int e = 0; e < E; ++e) on_peer[(size_t) e] = e % 4 == 1 || e == 0;
    // prefill.cpp's layout with a peer: the primary's experts first, then the peer's block, each in id order
    std::vector<int32_t> off2((size_t) E, 0), order_l, order_p, bh_l, bh_p;
    int32_t r = 0;
    for (int pass = 0; pass < 2; ++pass)
        for (int e = 0; e < E; ++e) {
            if ((bool) on_peer[(size_t) e] != (pass == 1)) continue;
            off2[(size_t) e] = r;
            if (L.cnt[(size_t) e] > 0) (pass ? order_p : order_l).push_back(e);
            r += L.cnt[(size_t) e];
        }
    const int64_t rows_local = [&] {
        int64_t n = 0;
        for (int e = 0; e < E; ++e) if (!on_peer[(size_t) e]) n += L.cnt[(size_t) e];
        return n;
    }();
    std::vector<int32_t> src2((size_t) rows);
    for (int e = 0; e < E; ++e)
        for (int q = 0; q < L.cnt[(size_t) e]; ++q) src2[(size_t) (off2[(size_t) e] + q)] = L.src[(size_t) (L.off[(size_t) e] + q)];
    for (int32_t e : order_l) bh_l.push_back(off2[(size_t) e]);
    bh_l.push_back((int32_t) rows_local);
    for (int32_t e : order_p) bh_p.push_back(off2[(size_t) e] - (int32_t) rows_local);
    bh_p.push_back((int32_t) (rows - rows_local));
    std::vector<int32_t> bh_all;
    for (int32_t e : L.order) bh_all.push_back(L.off[(size_t) e]);
    bh_all.push_back((int32_t) rows);
    const int64_t row_cap = std::min<int64_t>(rows, std::max<int64_t>(T, (int64_t) 1023 * K));   // group_rows_cap
    // the primary: the whole layer, then its own experts alone
    ck(cudaSetDevice(d0), "device 0");
    cudaStream_t s0;
    ck(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking), "stream");
    std::vector<float> ref, loc;
    {
        Dev x0(x.size() * 4);
        ck(cudaMemcpy(x0.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
        ref = primary_rows(L.order, bh_all, rows, x0.as<float>(), sh.T, L.src, blob0, row_cap, s0);
        const std::vector<int32_t> src_l(src2.begin(), src2.begin() + rows_local);
        loc = primary_rows(order_l, bh_l, rows_local, x0.as<float>(), sh.T, src_l, blob0, row_cap, s0);
    }
    cudaStreamDestroy(s0);
    // the peer: its own copy of the activations (the same bytes, through the host)
    ck(cudaSetDevice(d1), "device 1");
    cudaStream_t s1;
    ck(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking), "stream");
    std::vector<float> per;
    float ms = 0;
    int npieces = 0, ngroups = 0;
    {
        Dev x1(x.size() * 4);
        ck(cudaMemcpy(x1.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
        const std::vector<int32_t> src_p(src2.begin() + rows_local, src2.end());
        per = peer_rows(order_p, bh_p, rows - rows_local, x1.as<float>(), sh.T, src_p, blob1, sh.cap, s1, ms, npieces, ngroups);
    }
    cudaStreamDestroy(s1);
    ck(cudaSetDevice(d0), "device 0");
    int64_t bad = 0, maxcnt_p = 0;
    for (int32_t e : order_p) maxcnt_p = std::max<int64_t>(maxcnt_p, L.cnt[(size_t) e]);
    for (int e = 0; e < E; ++e)
        for (int q = 0; q < L.cnt[(size_t) e]; ++q) {
            const float* a = ref.data() + (size_t) (L.off[(size_t) e] + q) * N;
            const int64_t r2 = off2[(size_t) e] + q;
            const float* b = on_peer[(size_t) e] ? per.data() + (size_t) (r2 - rows_local) * N : loc.data() + (size_t) r2 * N;
            bad += std::memcmp(a, b, (size_t) N * 4) != 0;
        }
    const int64_t rows_p = rows - rows_local;
    const double flop = 2.0 * (double) rows_p * (1280.0 * N + (double) N * FF);
    std::printf("  peer share, %d tokens (%s routing), cap %lld: the peer's %zu experts hold %lld of %lld rows (%.1f%%, "
                "largest %lld) in %d pieces / %d groups; bitwise equal to the primary's whole layer: %s (%lld rows differ)\n",
                T, sh.hot ? "hot" : "even", (long long) sh.cap, order_p.size(), (long long) rows_p, (long long) rows,
                100.0 * (double) rows_p / (double) rows, (long long) maxcnt_p, npieces, ngroups, bad == 0 ? "yes" : "NO",
                (long long) bad);
    std::printf("  bench: peer share on device %d: %lld rows in %.2f ms (%.1f TFLOP/s)\n", d1, (long long) rows_p, ms,
                flop / (ms * 1e-3) * 1e-12);
    return bad == 0;
}

int peer_parity(int d0, int d1) {
    std::printf("peer share: device %d computes its experts' rows for device %d\n", d1, d0);
    std::mt19937 rng(5150);
    constexpr int NB = 16;   // distinct blobs, expert e takes blob e % NB
    std::vector<std::vector<uint8_t>> host((size_t) NB);
    float t3[3];
    for (int i = 0; i < NB; ++i) host[(size_t) i] = make_blob(rng, t3);
    std::vector<std::unique_ptr<Dev>> keep;
    std::vector<const uint8_t*> b0(512), b1(512);
    for (int d : {d0, d1}) {
        ck(cudaSetDevice(d), "device");
        std::vector<const uint8_t*> dist((size_t) NB);
        for (int i = 0; i < NB; ++i) {
            keep.push_back(std::make_unique<Dev>(BYTES));
            ck(cudaMemcpy(keep.back()->p, host[(size_t) i].data(), BYTES, cudaMemcpyHostToDevice), "blob");
            dist[(size_t) i] = keep.back()->as<uint8_t>();
        }
        for (int e = 0; e < 512; ++e) (d == d0 ? b0 : b1)[(size_t) e] = dist[(size_t) (e % NB)];
    }
    bool ok = true;
    for (const PeerShape& sh : {PeerShape{2600, 0, 2559}, PeerShape{4806, 2, 2559}, PeerShape{4806, 0, 1000},
                                PeerShape{1100, 5, 64}})
        ok &= peer_share(d0, d1, sh, b0, b1, rng);
    // the blobs are freed on their own devices
    for (size_t i = 0; i < keep.size(); ++i) {
        ck(cudaSetDevice(i < (size_t) NB ? d0 : d1), "device");
        keep[i].reset();
    }
    ck(cudaSetDevice(d0), "device 0");
    std::printf("peer share: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
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
    F4 f4(rows, L.order.size());
    ck(cudaMemset(b.dm.p, 0, (size_t) rows * N * 4), "clr");
    ck(cudaMemset(b.gu.p, 0, (size_t) rows * 1280 * 4), "clr");
    run_fp4(f4, b, L, x_dev.as<float>(), src_dev.as<int32_t>(), blob, s);
    ck(cudaStreamSynchronize(s), "sync fp4");
    const std::vector<float> fy = download(b.dm, (size_t) rows * N), fgu = download(b.gu, (size_t) rows * 1280),
                             fh = download(b.h, (size_t) rows * FF);
    const bool same_gather = std::memcmp(y.data(), y1.data(), y.size() * 4) == 0;
    // the production gate/up: token planes, SwiGLU in the epilogue; H must be the plain product + mmq::swiglu's bits
    const bool planes_ok = token_planes_match(f4, L, x_dev.as<float>(), src_dev.as<int32_t>(), s);
    Dev hf_dev((size_t) rows * FF * 4);
    ck(cudaMemset(hf_dev.p, 0xff, hf_dev.bytes), "clr");
    run_fused_h(f4, b, L, x_dev.as<float>(), src_dev.as<int32_t>(), blob, hf_dev.as<float>(), s);
    ck(cudaStreamSynchronize(s), "sync fused");
    const std::vector<float> hf = download(hf_dev, (size_t) rows * FF);
    const bool fused_same = std::memcmp(hf.data(), fh.data(), hf.size() * 4) == 0;
    std::printf("  fused SwiGLU H == plain gate/up product + mmq::swiglu (bitwise, %lld x %d): %s\n", (long long) rows, FF,
                fused_same ? "yes" : "NO");

    // the references, an expert at a time on all cores
    std::vector<float> r_same((size_t) rows * N), r_dbl((size_t) rows * N), r_f16((size_t) rows * N),
                       r_gu((size_t) rows * 1280), r_gu4((size_t) rows * 1280), r_dn4((size_t) rows * N),
                       r_h4((size_t) rows * FF);
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
        std::vector<float> xr(N), x16(N), h(FF), hr(FF), hd(FF), h16(FF), x4(N), h4(FF), h4r(FF);
        for (int32_t p = L.off[(size_t) e]; p < L.off[(size_t) e + 1]; ++p) {
            const float* xt = x.data() + (size_t) L.src[(size_t) p] * N;
            q8_round(xt, xr.data(), N);
            a4_round(xt, x4.data(), N);
            for (int c = 0; c < N; ++c) x16[(size_t) c] = f16(xt[c]);
            for (int k = 0; k < FF; ++k) {
                const float* wg = wgu.data() + (size_t) k * N;
                const float* wu = wgu.data() + (size_t) (FF + k) * N;
                const float* wg16 = wgu16.data() + (size_t) k * N;
                const float* wu16 = wgu16.data() + (size_t) (FF + k) * N;
                double g = 0, u = 0, gd = 0, ud = 0, g16 = 0, u16 = 0, g4 = 0, u4 = 0;
                for (int c = 0; c < N; ++c) {
                    g += (double) wg[c] * xr[(size_t) c];
                    u += (double) wu[c] * xr[(size_t) c];
                    gd += (double) wg[c] * xt[c];
                    ud += (double) wu[c] * xt[c];
                    g16 += (double) wg16[c] * x16[(size_t) c];
                    u16 += (double) wu16[c] * x16[(size_t) c];
                    g4 += (double) wg[c] * x4[(size_t) c];
                    u4 += (double) wu[c] * x4[(size_t) c];
                }
                r_gu[(size_t) p * 1280 + k] = (float) g;
                r_gu[(size_t) p * 1280 + FF + k] = (float) u;
                r_gu4[(size_t) p * 1280 + k] = (float) g4;
                r_gu4[(size_t) p * 1280 + FF + k] = (float) u4;
                g4 *= sg; u4 *= su;
                h4[(size_t) k] = (float) (g4 / (1.0 + std::exp(-g4)) * u4 * sd);
                r_h4[(size_t) p * FF + k] = h4[(size_t) k];
                g *= sg; u *= su; gd *= sg; ud *= su;
                h[(size_t) k] = (float) (g / (1.0 + std::exp(-g)) * u * sd);   // MMQ's H carries s_down
                hd[(size_t) k] = (float) (gd / (1.0 + std::exp(-gd)) * ud);
                h16[(size_t) k] = f16((float) (g16 / (1.0 + std::exp(-g16)) * u16));
            }
            q8_round(h.data(), hr.data(), FF);
            a4_round(h4.data(), h4r.data(), FF);
            for (int o = 0; o < N; ++o) {
                const float* w = wd.data() + (size_t) o * FF;
                double a = 0;
                for (int k = 0; k < FF; ++k) a += (double) w[k] * h4r[(size_t) k];
                r_dn4[(size_t) p * N + o] = (float) a;
            }
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
    const Err fe = compare(fy, r_dbl, (size_t) rows), fm = compare(fy, y, (size_t) rows);
    {
        double e2 = 0, r2 = 0, w = 0;
        for (int64_t p = 0; p < rows; ++p) {
            double me = 0, mr = 0;
            for (int c = 0; c < 1280; ++c) {
                const double d = (double) fgu[(size_t) p * 1280 + c] - r_gu[(size_t) p * 1280 + c];
                e2 += d * d; r2 += (double) r_gu[(size_t) p * 1280 + c] * r_gu[(size_t) p * 1280 + c];
                me = std::max(me, std::fabs(d)); mr = std::max(mr, (double) std::fabs(r_gu[(size_t) p * 1280 + c]));
            }
            if (mr > 0) w = std::max(w, me / mr);
        }
        std::printf("  FP4 gate/up vs MMQ-rounding reference: rms %.3e worst row %.3e;  FP4 vs double: rms %.3e worst row %.3e\n",
                    std::sqrt(e2 / r2), w, fe.rms, fe.worst);
    }
    // the FP4 path against the same A4 roundings in double (only fp32 accumulation order and rounding ties differ),
    // and its output for a token must not depend on the token's tile or its neighbours: every expert's rows reversed
    double a2 = 0, b2 = 0;
    for (size_t i = 0; i < (size_t) rows * 1280; ++i) {
        const double d = (double) fgu[i] - r_gu4[i];
        a2 += d * d; b2 += (double) r_gu4[i] * r_gu4[i];
    }
    const double f4gu = b2 > 0 ? std::sqrt(a2 / b2) : 0;
    const Err f4dn = compare(fy, r_dn4, (size_t) rows);
    bool indep = true;
    {
        Layer L2 = L;
        for (int e = 0; e < E; ++e) std::reverse(L2.src.begin() + L.off[(size_t) e], L2.src.begin() + L.off[(size_t) e + 1]);
        Dev src2((size_t) rows * 4);
        ck(cudaMemcpy(src2.p, L2.src.data(), L2.src.size() * 4, cudaMemcpyHostToDevice), "src2");
        ck(cudaMemset(b.dm.p, 0xff, (size_t) rows * N * 4), "clr");
        run_fp4(f4, b, L2, x_dev.as<float>(), src2.as<int32_t>(), blob, s);
        ck(cudaStreamSynchronize(s), "sync fp4 reversed");
        const std::vector<float> ry = download(b.dm, (size_t) rows * N);
        for (int e = 0; e < E; ++e)
            for (int i = 0; i < L.cnt[(size_t) e]; ++i)
                indep &= std::memcmp(fy.data() + (size_t) (L.off[(size_t) e] + i) * N,
                                     ry.data() + (size_t) (L.off[(size_t) e] + L.cnt[(size_t) e] - 1 - i) * N, N * 4) == 0;
    }
    std::printf("  FP4 layer output vs double:  rms %.3e worst row %.3e (tol rms 0.2);  vs MMQ output: rms %.3e (tol 0.2)\n",
                fe.rms, fe.worst, fm.rms);
    std::printf("  FP4 gate/up vs A4-rounding reference: rms %.3e (tol 1e-5)\n", f4gu);
    std::printf("  FP4 down vs A4-rounding reference:    rms %.3e worst row %.3e (tol rms 1e-4)\n", f4dn.rms, f4dn.worst);
    std::printf("  FP4 output independent of tile position and neighbours (bitwise): %s\n", indep ? "yes" : "NO");
    std::printf("  FP4 output hashes (FNV-1a): H %016llx, layer %016llx\n", (unsigned long long) fnv(fh),
                (unsigned long long) fnv(fy));
    double h2 = 0, hr2 = 0, hw = 0;
    for (int64_t p = 0; p < rows; ++p) {
        double me = 0, mr = 0;
        for (int k = 0; k < FF; ++k) {
            const double d = (double) hf[(size_t) p * FF + k] - r_h4[(size_t) p * FF + k];
            h2 += d * d; hr2 += (double) r_h4[(size_t) p * FF + k] * r_h4[(size_t) p * FF + k];
            me = std::max(me, std::fabs(d)); mr = std::max(mr, (double) std::fabs(r_h4[(size_t) p * FF + k]));
        }
        if (mr > 0) hw = std::max(hw, me / mr);
    }
    const double hrms = hr2 > 0 ? std::sqrt(h2 / hr2) : 0;
    std::printf("  fused SwiGLU H vs A4-rounding reference: rms %.3e worst row %.3e (tol rms 1e-5)\n", hrms, hw);
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
    // the group path, split every way, on this routing and on one with more hot experts
    Layer Lb = route(T, E, rng, 5);
    bool split_ok = split_equivalence("routing A (2 hot experts)", L, x_dev.as<float>(), blob, s);
    split_ok &= split_equivalence("routing B (5 hot experts)", Lb, x_dev.as<float>(), blob, s);
    const bool edge_ok = edge_activations(host, blob, s);
    const bool ok = split_ok && edge_ok && fe.rms <= 0.2 && fm.rms <= 0.2 && grms <= 1e-5 && gworst <= 1e-4 && es.rms <= 2e-4 && es.worst <= 1e-2 && ed.rms <= 2.5e-2 &&
                    ef.rms <= 2e-2 && same_gather && zero_ok &&
                    f4gu <= 1e-5 && f4dn.rms <= 1e-4 && indep && planes_ok && fused_same && hrms <= 1e-5;
    std::printf("parity: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

void bench(cudaStream_t s) {
    constexpr int T = 16384, E = 512;
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
    const size_t cmp_rows = 4096;
    std::vector<float> m_gu(cmp_rows * 1280), m_dm(cmp_rows * N);
    ck(cudaMemcpy(m_gu.data(), b.gu.p, m_gu.size() * 4, cudaMemcpyDeviceToHost), "d");
    ck(cudaMemcpy(m_dm.data(), b.dm.p, m_dm.size() * 4, cudaMemcpyDeviceToHost), "d");
    F4 f(rows, L.order.size());
    for (int w = 0; w < 2; ++w) run_fp4(f, b, L, x_dev.as<float>(), src_dev.as<int32_t>(), blob, s);
    cudaEventRecord(e0, s);
    for (int r = 0; r < R; ++r) run_fp4(f, b, L, x_dev.as<float>(), src_dev.as<int32_t>(), blob, s);
    cudaEventRecord(e1, s);
    ck(cudaEventSynchronize(e1), "bench fp4");
    float ms4 = 0;
    cudaEventElapsedTime(&ms4, e0, e1);
    std::vector<float> f_gu(cmp_rows * 1280), f_dm(cmp_rows * N);
    ck(cudaMemcpy(f_gu.data(), b.gu.p, f_gu.size() * 4, cudaMemcpyDeviceToHost), "d");
    ck(cudaMemcpy(f_dm.data(), b.dm.p, f_dm.size() * 4, cudaMemcpyDeviceToHost), "d");
    auto rel = [](const std::vector<float>& a, const std::vector<float>& r) {
        double e = 0, q = 0;
        for (size_t i = 0; i < a.size(); ++i) { e += (double) (a[i] - r[i]) * (a[i] - r[i]); q += (double) r[i] * r[i]; }
        return std::sqrt(e / q);
    };
    std::printf("bench: FP4 layer %.1f us (x%.2f vs MMQ); rel rms fp4 vs mmq: gate/up %.3e (tail-free), down %.3e\n",
                1e3 * ms4 / R, ms / ms4, rel(f_gu, m_gu), rel(f_dm, m_dm));
    // the production path: token planes, the gate/up product with the SwiGLU in its epilogue, a group per run
    F4Run run(rows);
    auto prod = [&]() { enqueue_fp4_group(f, run, b, L, x_dev.as<float>(), src_dev.as<int32_t>(), blob, (int64_t) 1 << 40, s); };
    for (int w = 0; w < 2; ++w) prod();
    cudaEventRecord(e0, s);
    for (int r = 0; r < R; ++r) prod();
    cudaEventRecord(e1, s);
    ck(cudaEventSynchronize(e1), "bench production");
    float msp = 0;
    cudaEventElapsedTime(&msp, e0, e1);
    const double flop4 = 2.0 * (double) rows * (1280.0 * N + (double) N * FF);
    std::printf("bench: FP4 expert path, %d tokens x top %d over %d experts: quantize per row + plain gate/up + SwiGLU "
                "%.1f us (%.1f TFLOP/s), token planes + fused SwiGLU %.1f us (%.1f TFLOP/s), x%.2f\n", T, K, E,
                1e3 * ms4 / R, flop4 / (1e3 * ms4 / R) * 1e-6, 1e3 * msp / R, flop4 / (1e3 * msp / R) * 1e-6, ms4 / msp);
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
        int rc = parity(s);
        if (do_bench) bench(s);
        cudaStreamDestroy(s);
        // the GEMM launch's shared-memory opt-in is per device: every other device must run it too
        for (int d = 1; d < ndev && rc == 0; ++d) {
            ck(cudaSetDevice(d), "set device");
            std::printf("device %d\n", d);
            cudaStream_t s2;
            ck(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking), "stream");
            rc = parity(s2);
            cudaStreamDestroy(s2);
        }
        ck(cudaSetDevice(0), "set device");
        if (rc == 0 && ndev >= 2) rc = peer_parity(0, 1);
        else if (ndev < 2) std::printf("peer share: one device, skipped\n");
        return rc;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "prefill_mmq_nvfp4_test: %s\n", ex.what());
        return 1;
    }
}
