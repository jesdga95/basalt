// prefill_cpu_share_test - a small prompt chunk's experts on the CPU pool (cpu_experts.cpp's CPU share) against the GPU's
// MMQ path (gather, q8_1 activations, gate/up, SwiGLU, q8_1 of H, down) and a double-precision reference (the weights
// dequantized by ggml's own to_float, times the FP32 activations), on random native expert blobs at the model's
// geometry (2560 x 640) and random routing in which every expert has at most MAXT rows - the experts the share takes.
// The CPU rounds the activations per 256 values (Q8_K) and H per its down type's format, MMQ per 32, so this is not a
// bitwise test: the CPU path's error against the reference has to be at most 1.5x MMQ's RMS and worst row (3x when
// the activations carry a few large values, which cost the coarser Q8_K blocks more).  A
// token's output must not depend on how many tokens share its expert's job: every row is also run alone and must match
// bitwise.  Prints the pool's time per expert and the bytes it read per second.  Exit 77 without a CUDA device.
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/prefill/moe_mmq.hpp"

#include "basalt/ggml/ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
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
    explicit Dev(size_t n) { ck(cudaMalloc(&p, n), "cudaMalloc"); }
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

// random blocks with a sane fp16 scale `d` at each block's start (every format here begins with it): scales chosen so
// that the weights' typical size is that of a 2-3 bit expert (a few hundredths)
void fill_matrix(uint8_t* m, ggml_type t, int64_t rows, int64_t cols, std::mt19937& rng) {
    const size_t bs = ggml_type_size(t), nb = (size_t) rows * (size_t) (cols / ggml_blck_size(t));
    std::uniform_int_distribution<int> byte(0, 255);
    const float lo = t == GGML_TYPE_IQ3_S ? 0.0004f : t == GGML_TYPE_IQ4_XS ? 0.00005f
                   : t == GGML_TYPE_IQ4_NL ? 0.0002f : t == GGML_TYPE_Q2_0 ? 0.004f : 0.0008f;
    std::uniform_real_distribution<float> sc(lo, 6.0f * lo);
    for (size_t i = 0; i < nb * bs; ++i) m[i] = (uint8_t) byte(rng);
    for (size_t b = 0; b < nb; ++b) {
        const ggml_fp16_t h = ggml_fp32_to_fp16(sc(rng));
        std::memcpy(m + b * bs, &h, 2);
    }
}
std::vector<uint8_t> make_blob(const Pair& p, const Geo& g, std::mt19937& rng) {
    std::vector<uint8_t> b(g.bytes);
    fill_matrix(b.data(), p.gu, 2 * FF, N, rng);
    fill_matrix(b.data() + g.down_off, p.d, N, FF, rng);
    return b;
}

// ggml's dequantized weights: gate [640][2560], up [640][2560], down [2560][640]
void dequant(const Pair& p, const Geo& g, const std::vector<uint8_t>& b, std::vector<float>& gu, std::vector<float>& dn) {
    const auto* tg = ggml_get_type_traits(p.gu);
    const auto* td = ggml_get_type_traits(p.d);
    gu.resize((size_t) 2 * FF * N);
    dn.resize((size_t) N * FF);
    for (int r = 0; r < 2 * FF; ++r) tg->to_float(b.data() + (size_t) r * g.gu_row, gu.data() + (size_t) r * N, N);
    for (int r = 0; r < N; ++r) td->to_float(b.data() + g.down_off + (size_t) r * g.d_row, dn.data() + (size_t) r * FF, FF);
}

// activations: N(0, 1) with a few large values (the hidden state has outliers); token `zero` all zero
std::vector<float> make_x(int T, std::mt19937& rng, int zero, bool outliers) {
    std::vector<float> x((size_t) T * N);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> pick(0, 199);
    for (float& v : x) v = nd(rng) * (outliers && pick(rng) == 0 ? 12.0f : 1.0f);
    if (zero >= 0) std::fill(x.begin() + (size_t) zero * N, x.begin() + (size_t) (zero + 1) * N, 0.0f);
    return x;
}

// K distinct experts per token; `hot` > 0 skews the first choices towards the lowest ids
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
    std::vector<int32_t> cnt, off, src;   // per expert; sorted row -> token (rows in pair order within an expert)
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
// up rows), q8_1 of H, down.  Returns Dm [rows][N] in sorted-row order.
struct MmqBufs {
    Dev xq, gu, h, hq, dm, ident, bounds, grp_gu, grp_d;
    MmqBufs(const Pair& p, int64_t rows, int E)
        : xq(mmq::q8_bytes(rows, N)), gu((size_t) rows * 1280 * 4), h((size_t) rows * FF * 4),
          hq(mmq::q8_bytes(rows, FF)), dm((size_t) rows * N * 4), ident((size_t) rows * 4),
          bounds((size_t) (2 * (E + E / GROUP + 2)) * 4), grp_gu(GROUP * mmq::matrix_bytes(p.gu, 1280, N) + 4096),
          grp_d(GROUP * mmq::matrix_bytes(p.d, N, FF) + 4096) {}
};
// `direct`: no gathers - the products read the blobs where they are (they must lie at a fixed stride, every expert
// with rows: the timing part's layout), the cost of MMQ's own kernels alone
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

struct Err {
    double rms = 0, worst = 0;   // RMS error / RMS of the reference; the worst row's max |error| / its max |ref|
};
// y_a[row_a(i)] against y_b[row_b(i)] for every pair i (rows of N values)
template <typename RA, typename RB>
Err compare(const std::vector<float>& a, RA row_a, const std::vector<float>& b, RB row_b, size_t pairs) {
    double e2 = 0, r2 = 0;
    Err out;
    for (size_t i = 0; i < pairs; ++i) {
        const float* ya = a.data() + (size_t) row_a(i) * N;
        const float* yb = b.data() + (size_t) row_b(i) * N;
        double me = 0, mr = 0;
        for (int o = 0; o < N; ++o) {
            if (!std::isfinite(ya[o])) throw std::runtime_error("a non-finite output");
            const double d = (double) ya[o] - yb[o];
            e2 += d * d;
            r2 += (double) yb[o] * yb[o];
            me = std::max(me, std::fabs(d));
            mr = std::max(mr, (double) std::fabs(yb[o]));
        }
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


// part 1: the CPU pool and MMQ against the reference
void run_pair(const Pair& p, bool outliers, basalt::kernels::cpu::ExpertPool& pool, cudaStream_t s) {
    namespace cpu = basalt::kernels::cpu;
    constexpr int T = 12, E = 64;
    const Geo geo(p);
    std::mt19937 rng(7 + (unsigned) p.gu * 5 + (unsigned) p.d);
    std::vector<std::vector<uint8_t>> host((size_t) E);
    std::vector<std::unique_ptr<Dev>> dev;
    std::vector<const uint8_t*> blob((size_t) E);
    for (int e = 0; e < E; ++e) {
        host[(size_t) e] = make_blob(p, geo, rng);
        dev.push_back(std::make_unique<Dev>(geo.bytes + 64));
        uint8_t* q = dev.back()->as<uint8_t>() + 2 * (size_t) (e % 2);
        ck(cudaMemcpy(q, host[(size_t) e].data(), geo.bytes, cudaMemcpyHostToDevice), "blob");
        blob[(size_t) e] = q;
    }
    const std::vector<float> x = make_x(T, rng, 5, outliers);
    std::vector<int32_t> ids;
    Routing r;
    do {   // every expert at most MAXT rows, some with several
        ids = make_ids(T, E, rng, 6, 0);
        r = sort_rows(ids, E);
    } while (*std::max_element(r.cnt.begin(), r.cnt.end()) > cpu::MAXT);
    const int64_t rows = (int64_t) T * K;
    int used = 0, maxr = 0;
    for (int c : r.cnt) { used += c > 0; maxr = std::max(maxr, c); }
    std::printf("%s, %s activations: %d tokens, %d routed experts, up to %d rows each\n", p.name,
                outliers ? "with outliers" : "plain", T, used, maxr);

    Dev x_dev(x.size() * 4), src_dev((size_t) rows * 4);
    ck(cudaMemcpy(x_dev.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(src_dev.p, r.src.data(), r.src.size() * 4, cudaMemcpyHostToDevice), "src");
    ck(cudaDeviceSynchronize(), "uploads");
    mmq::Context ctx;
    MmqBufs mb(p, rows, E);
    run_mmq(p, geo, ctx, mb, r, x_dev.as<float>(), src_dev.as<int32_t>(), blob, rows, s);
    ck(cudaStreamSynchronize(s), "sync");
    const std::vector<float> y_mmq = download(mb.dm, (size_t) rows * N);

    // the CPU: one job per routed expert, its rows in sorted-row order
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt((int) p.gu, (int) p.d, N, FF, f, err)) throw std::runtime_error(err);
    std::vector<uint8_t> act((size_t) T * cpu::kNativeActBytes);
    for (int t = 0; t < T; ++t) cpu::native_quant_act(f, x.data() + (size_t) t * N, act.data() + (size_t) t * cpu::kNativeActBytes);
    std::vector<float> y_cpu((size_t) rows * N, std::nanf(""));
    std::vector<cpu::ExpertJobMulti> jobs;
    for (int e = 0; e < E; ++e) {
        if (r.cnt[(size_t) e] == 0) continue;
        cpu::ExpertJobMulti j;
        j.blob = host[(size_t) e].data();
        j.nt = r.cnt[(size_t) e];
        for (int i = 0; i < j.nt; ++i) {
            const int32_t row = r.off[(size_t) e] + i;
            j.nact[i] = act.data() + (size_t) r.src[(size_t) row] * cpu::kNativeActBytes;
            j.out[i] = y_cpu.data() + (size_t) row * N;
        }
        jobs.push_back(j);
    }
    pool.run_split_multi_native(f, jobs.data(), (int) jobs.size());   // warm
    const auto t0 = std::chrono::steady_clock::now();
    constexpr int REP = 20;
    for (int i = 0; i < REP; ++i) pool.run_split_multi_native(f, jobs.data(), (int) jobs.size());
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / REP;

    // every row alone: the same bits
    std::vector<float> y_one((size_t) rows * N, std::nanf(""));
    std::vector<cpu::ExpertJobMulti> ones;
    for (int e = 0; e < E; ++e)
        for (int i = 0; i < r.cnt[(size_t) e]; ++i) {
            cpu::ExpertJobMulti j;
            j.blob = host[(size_t) e].data();
            j.nt = 1;
            const int32_t row = r.off[(size_t) e] + i;
            j.nact[0] = act.data() + (size_t) r.src[(size_t) row] * cpu::kNativeActBytes;
            j.out[0] = y_one.data() + (size_t) row * N;
            ones.push_back(j);
        }
    pool.run_split_multi_native(f, ones.data(), (int) ones.size());
    if (std::memcmp(y_one.data(), y_cpu.data(), y_cpu.size() * 4) != 0)
        throw std::runtime_error(std::string(p.name) + ": a row's output depends on the tokens sharing its expert");

    // the reference
    std::vector<float> ref((size_t) rows * N);
    std::vector<std::thread> th;
    const int nth = std::max(1, std::min(16, (int) std::thread::hardware_concurrency()));
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
    const Err ec = compare(y_cpu, srow, ref, pairi, (size_t) rows);
    const Err ecm = compare(y_cpu, srow, y_mmq, srow, (size_t) rows);
    double amax = 0;
    for (int64_t i = 0; i < rows; ++i)
        for (int o = 0; o < N; ++o)
            amax = std::max(amax, (double) std::fabs(y_cpu[(size_t) r.row_of[(size_t) i] * N + o] - y_mmq[(size_t) r.row_of[(size_t) i] * N + o]));
    std::printf("  MMQ vs FP32 reference: rel RMS %.3e  worst row max rel %.3e\n", em.rms, em.worst);
    std::printf("  CPU vs FP32 reference: rel RMS %.3e  worst row max rel %.3e\n", ec.rms, ec.worst);
    std::printf("  CPU vs MMQ           : rel RMS %.3e  worst row max rel %.3e  max abs %.3e\n", ecm.rms, ecm.worst, amax);
    std::printf("  pool: %.3f ms per layer call, %.1f us per expert, %.1f GB/s of expert bytes read\n", ms,
                1000.0 * ms / used, (double) used * (double) f.bytes / (ms * 1e6));
    // the CPU rounds the activations per 256 values against MMQ's 32, so a few large values cost it more; plain
    // activations must be comparable
    const double lim = outliers ? 3.0 : 1.5;
    if (ec.rms > lim * em.rms || ec.worst > lim * em.worst)
        throw std::runtime_error(std::string(p.name) + ": the CPU path's error is not comparable to MMQ's");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        int n_dev = 0;
        if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
            std::printf("no CUDA device - skipped\n");
            return 77;
        }
        (void) argc; (void) argv;
        const Pair pairs[] = {{GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, "IQ3_S / IQ4_NL"},
                              {GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL, "IQ4_XS / IQ4_NL"},
                              {GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL, "IQ3_XXS / IQ4_NL"},
                              {GGML_TYPE_IQ4_XS, GGML_TYPE_Q2_0, "IQ4_XS / Q2_0"}};
        cudaStream_t s = nullptr;
        ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        basalt::kernels::cpu::ExpertPool pool((std::max)(1, (int) basalt::kernels::cpu::physical_cores(true).size()),
                                                    basalt::kernels::cpu::PoolAffinity::All);
        for (const Pair& p : pairs)
            for (bool outliers : {false, true}) run_pair(p, outliers, pool, s);
        ck(cudaStreamDestroy(s), "destroy");
        std::printf("prefill CPU share parity passed\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "prefill CPU share parity failed: %s\n", e.what());
        return 1;
    }
}
