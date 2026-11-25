// tests/kernels/q8_gemm_test.cpp - parity, invariance, and microbench for prefill::q8 (2-CTA TMA block-32 Q8_0 GEMM).
#include "basalt/prefill/q8_gemm.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "basalt/prefill/moe_mmq.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/window_gemm.hpp"

namespace {

using namespace basalt::prefill;
struct alignas(2) block_q8_0 { uint16_t d; int8_t qs[32]; };
static_assert(sizeof(block_q8_0) == 34);

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::printf("CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

void make_q8_0(const std::vector<float>& w_f32, int64_t rows, int64_t cols, std::vector<block_q8_0>& out) {
    const int64_t nb = cols / 32;
    out.resize((size_t) (rows * nb));
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t b = 0; b < nb; ++b) {
            const float* p = &w_f32[(size_t) (r * cols + b * 32)];
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(p[i]));
            const float d = amax / 127.0f;
            const __half dh = __float2half(d);
            const float d_eff = __half2float(dh);
            const float id = d_eff > 0.0f ? (1.0f / d_eff) : 0.0f;
            block_q8_0 blk{};
            std::memcpy(&blk.d, &dh, sizeof(blk.d));
            for (int i = 0; i < 32; ++i) {
                int q = (int) std::lrintf(p[i] * id);
                q = std::max(-127, std::min(127, q));
                blk.qs[i] = (int8_t) q;
            }
            out[(size_t) (r * nb + b)] = blk;
        }
    }
}

void ref_q8_double(const std::vector<float>& x, int64_t ntok, int64_t cols, const std::vector<block_q8_0>& w,
                   int64_t rows, std::vector<float>& out, bool add) {
    const int64_t nb = cols / 32;
    // Quantize activations per block-32 matching q8::quantize, then double-precision accumulate.
    std::vector<int8_t> aq((size_t) (ntok * cols));
    std::vector<float> ad((size_t) (ntok * nb));
    for (int64_t t = 0; t < ntok; ++t) {
        for (int64_t b = 0; b < nb; ++b) {
            const float* p = &x[(size_t) (t * cols + b * 32)];
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(p[i]));
            const float d = amax * (1.0f / 127.0f);
            const float id = amax > 0.0f ? (127.0f / amax) : 0.0f;
            ad[(size_t) (t * nb + b)] = d;
            for (int i = 0; i < 32; ++i) aq[(size_t) (t * cols + b * 32 + i)] = (int8_t) std::lrintf(p[i] * id);
        }
    }
    if (!add) out.assign((size_t) (ntok * rows), 0.0f);
    for (int64_t t = 0; t < ntok; ++t) {
        for (int64_t r = 0; r < rows; ++r) {
            double s = 0.0;
            for (int64_t b = 0; b < nb; ++b) {
                const block_q8_0& blk = w[(size_t) (r * nb + b)];
                const __half dh = __ushort_as_half(blk.d);
                int idot = 0;
                for (int i = 0; i < 32; ++i) idot += (int) aq[(size_t) (t * cols + b * 32 + i)] * (int) blk.qs[i];
                s += (double) idot * ((double) ad[(size_t) (t * nb + b)] * (double) __half2float(dh));
            }
            if (add) out[(size_t) (t * rows + r)] += (float) s;
            else out[(size_t) (t * rows + r)] = (float) s;
        }
    }
}

struct GpuWeight {
    void* planes = nullptr;
    uint16_t* scales_t = nullptr;
    int64_t rows = 0;
    int64_t cols = 0;

    void init(const std::vector<block_q8_0>& blocks, int64_t r, int64_t c) {
        rows = r;
        cols = c;
        void* d_raw = nullptr;
        ck(cudaMalloc(&d_raw, blocks.size() * sizeof(block_q8_0)), "malloc raw");
        ck(cudaMemcpy(d_raw, blocks.data(), blocks.size() * sizeof(block_q8_0), cudaMemcpyHostToDevice), "cpy raw");
        ck(cudaMalloc(&planes, (size_t)(r * (c / 32) * 34)), "malloc planes");
        basalt::kernels::wgemm::q8_planes(d_raw, r, c, planes, nullptr);
        ck(cudaFree(d_raw), "free raw");
        ck(cudaMalloc(&scales_t, q8::scale_plane_bytes(r, c)), "malloc scales_t");
        const uint16_t* row_sc = reinterpret_cast<const uint16_t*>(static_cast<const uint8_t*>(planes) + r * c);
        q8::transpose_scales(row_sc, r, c, scales_t, nullptr);
        ck(cudaDeviceSynchronize(), "sync init");
    }

    const int8_t* codes() const { return static_cast<const int8_t*>(planes); }

    void destroy() {
        if (planes) cudaFree(planes);
        if (scales_t) cudaFree(scales_t);
        planes = nullptr;
        scales_t = nullptr;
    }
};

void test_case(std::mt19937& rng, int64_t ntok, int64_t rows, int64_t cols, bool add, int64_t ld_pad = 0) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x((size_t) (ntok * cols));
    std::vector<float> wf((size_t) (rows * cols));
    for (float& v : x) v = dist(rng);
    for (float& v : wf) v = dist(rng);

    std::vector<block_q8_0> wb;
    make_q8_0(wf, rows, cols, wb);

    const int64_t ld = rows + ld_pad;
    std::vector<float> base((size_t) (ntok * rows), 0.0f);
    if (add) {
        for (float& v : base) v = dist(rng) * 3.0f;
    }
    std::vector<float> ref = base;
    ref_q8_double(x, ntok, cols, wb, rows, ref, add);

    GpuWeight gw;
    gw.init(wb, rows, cols);

    float* d_x = nullptr;
    void* d_act = nullptr;
    float* d_y = nullptr;
    ck(cudaMalloc(&d_x, x.size() * sizeof(float)), "malloc x");
    ck(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "cpy x");
    ck(cudaMalloc(&d_act, q8::act_bytes(ntok, cols)), "malloc act");
    ck(cudaMalloc(&d_y, (size_t) (ntok * ld) * sizeof(float)), "malloc y");
    std::vector<float> y_init((size_t) (ntok * ld), 123.45f);
    if (add) {
        for (int64_t t = 0; t < ntok; ++t)
            for (int64_t r = 0; r < rows; ++r) y_init[(size_t) (t * ld + r)] = base[(size_t) (t * rows + r)];
    }
    ck(cudaMemcpy(d_y, y_init.data(), y_init.size() * sizeof(float), cudaMemcpyHostToDevice), "cpy y");

    q8::quantize(d_x, ntok, cols, cols, d_act, nullptr);
    q8::gemm(gw.codes(), gw.scales_t, rows, cols, d_act, ntok, d_y, ld,
             add ? q8::Epilogue::Add : q8::Epilogue::Store, nullptr);
    ck(cudaDeviceSynchronize(), "sync case");

    std::vector<float> got((size_t) (ntok * ld));
    ck(cudaMemcpy(got.data(), d_y, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "d2h y");

    double max_abs = 0.0, max_ref = 0.0;
    for (int64_t t = 0; t < ntok; ++t) {
        for (int64_t r = 0; r < rows; ++r) {
            const float g = got[(size_t) (t * ld + r)];
            const float e = ref[(size_t) (t * rows + r)];
            max_abs = std::max(max_abs, (double) std::fabs(g - e));
            max_ref = std::max(max_ref, (double) std::fabs(e));
        }
        for (int64_t r = rows; r < ld; ++r) {
            if (got[(size_t) (t * ld + r)] != 123.45f) {
                std::printf("FAIL: OOB write past rows at t=%lld r=%lld\n", (long long) t, (long long) r);
                std::exit(1);
            }
        }
    }
    const double rel = max_ref > 0.0 ? max_abs / max_ref : max_abs;
    if (rel > 2e-5) {
        std::printf("FAIL [%lldx%lldx%lld add=%d ld_pad=%lld]: max_abs=%.3e rel=%.3e\n", (long long) ntok,
                    (long long) rows, (long long) cols, (int) add, (long long) ld_pad, max_abs, rel);
        std::exit(1);
    }
    std::printf("PASS [%4lld x %4lld x %4lld add=%d pad=%lld]: max_abs=%.2e rel=%.2e\n", (long long) ntok,
                (long long) rows, (long long) cols, (int) add, (long long) ld_pad, max_abs, rel);

    cudaFree(d_x);
    cudaFree(d_act);
    cudaFree(d_y);
    gw.destroy();
}

void test_invariance_and_segmented(std::mt19937& rng) {
    // 1) Token-count invariance: row 0..16 bits in ntok=17 must equal row 0..16 bits in ntok=257.
    // 2) Segmented bitwise equality: 3 segments in one launch must equal 3 separate launches bitwise.
    const int64_t cols = 2304;
    const int64_t r0 = 6552, r1 = 1056, r2 = 1056;
    const int64_t n_small = 17, n_big = 257;
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x((size_t) (n_big * cols));
    for (float& v : x) v = dist(rng);

    auto mk_w = [&](int64_t r) {
        std::vector<float> wf((size_t) (r * cols));
        for (float& v : wf) v = dist(rng);
        std::vector<block_q8_0> wb;
        make_q8_0(wf, r, cols, wb);
        GpuWeight gw;
        gw.init(wb, r, cols);
        return gw;
    };
    GpuWeight w0 = mk_w(r0), w1 = mk_w(r1), w2 = mk_w(r2);

    float* d_x = nullptr;
    ck(cudaMalloc(&d_x, x.size() * sizeof(float)), "malloc x");
    ck(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "cpy x");

    void *d_act_s = nullptr, *d_act_b = nullptr;
    ck(cudaMalloc(&d_act_s, q8::act_bytes(n_small, cols)), "act s");
    ck(cudaMalloc(&d_act_b, q8::act_bytes(n_big, cols)), "act b");
    q8::quantize(d_x, n_small, cols, cols, d_act_s, nullptr);
    q8::quantize(d_x, n_big, cols, cols, d_act_b, nullptr);

    float *y0s, *y0b, *y1b, *y2b, *y0seg, *y1seg, *y2seg;
    ck(cudaMalloc(&y0s, (size_t) (n_small * r0) * sizeof(float)), "y0s");
    ck(cudaMalloc(&y0b, (size_t) (n_big * r0) * sizeof(float)), "y0b");
    ck(cudaMalloc(&y1b, (size_t) (n_big * r1) * sizeof(float)), "y1b");
    ck(cudaMalloc(&y2b, (size_t) (n_big * r2) * sizeof(float)), "y2b");
    ck(cudaMalloc(&y0seg, (size_t) (n_big * r0) * sizeof(float)), "y0seg");
    ck(cudaMalloc(&y1seg, (size_t) (n_big * r1) * sizeof(float)), "y1seg");
    ck(cudaMalloc(&y2seg, (size_t) (n_big * r2) * sizeof(float)), "y2seg");

    q8::gemm(w0.codes(), w0.scales_t, r0, cols, d_act_s, n_small, y0s, r0, q8::Epilogue::Store, nullptr);
    q8::gemm(w0.codes(), w0.scales_t, r0, cols, d_act_b, n_big, y0b, r0, q8::Epilogue::Store, nullptr);
    q8::gemm(w1.codes(), w1.scales_t, r1, cols, d_act_b, n_big, y1b, r1, q8::Epilogue::Store, nullptr);
    q8::gemm(w2.codes(), w2.scales_t, r2, cols, d_act_b, n_big, y2b, r2, q8::Epilogue::Store, nullptr);

    const q8::Segment segs[3] = {
        {w0.codes(), r0, w0.scales_t, y0seg, r0},
        {w1.codes(), r1, w1.scales_t, y1seg, r1},
        {w2.codes(), r2, w2.scales_t, y2seg, r2},
    };
    q8::gemm_segmented(segs, 3, cols, d_act_b, n_big, q8::Epilogue::Store, nullptr);
    ck(cudaDeviceSynchronize(), "sync inv");

    auto check_eq = [](const float* da, const float* db, size_t n, const char* name) {
        std::vector<uint32_t> a(n), b(n);
        ck(cudaMemcpy(a.data(), da, n * 4, cudaMemcpyDeviceToHost), "d2h a");
        ck(cudaMemcpy(b.data(), db, n * 4, cudaMemcpyDeviceToHost), "d2h b");
        size_t diff = 0;
        for (size_t i = 0; i < n; ++i)
            if (a[i] != b[i]) ++diff;
        if (diff) {
            std::printf("FAIL %s: %zu/%zu bits differ\n", name, diff, n);
            std::exit(1);
        }
        std::printf("PASS %s: bitwise equal (%zu floats)\n", name, n);
    };
    check_eq(y0s, y0b, (size_t) (n_small * r0), "row-count invariance (17 vs 257)");
    check_eq(y0b, y0seg, (size_t) (n_big * r0), "segmented[0] (Q 6552) vs standalone");
    check_eq(y1b, y1seg, (size_t) (n_big * r1), "segmented[1] (K 1056) vs standalone");
    check_eq(y2b, y2seg, (size_t) (n_big * r2), "segmented[2] (V 1056) vs standalone");

    cudaFree(d_x);
    cudaFree(d_act_s);
    cudaFree(d_act_b);
    cudaFree(y0s);
    cudaFree(y0b);
    cudaFree(y1b);
    cudaFree(y2b);
    cudaFree(y0seg);
    cudaFree(y1seg);
    cudaFree(y2seg);
    w0.destroy();
    w1.destroy();
    w2.destroy();
}

void test_q8_planes_mmvq_parity(std::mt19937& rng) {
    namespace bk = basalt::kernels;
    cudaStream_t st = nullptr;
    ck(cudaStreamCreate(&st), "stream");
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    struct Sh { const char* name; int M, K; };
    const Sh shs[] = {{"w.proj [8664x2304]", 8664, 2304}, {"w.out [2304x3072]", 2304, 3072}, {"lm_head [65536x2304]", 65536, 2304}};
    constexpr int TMAX = 64;
    for (const auto& s : shs) {
        std::vector<float> wf((size_t) s.M * s.K);
        for (float& v : wf) v = dist(rng);
        std::vector<block_q8_0> wb;
        make_q8_0(wf, s.M, s.K, wb);
        const size_t bytes = wb.size() * sizeof(block_q8_0);
        std::vector<float> hx((size_t) TMAX * s.K);
        for (float& v : hx) v = dist(rng);
        void *draw = nullptr, *dpl = nullptr, *dxq = nullptr;
        float *dx = nullptr, *y0 = nullptr, *y1 = nullptr;
        ck(cudaMalloc(&draw, bytes), "raw");
        ck(cudaMemcpy(draw, wb.data(), bytes, cudaMemcpyHostToDevice), "cpy raw");
        ck(cudaMalloc(&dpl, bytes), "pl");
        bk::wgemm::q8_planes(draw, s.M, s.K, dpl, st);
        ck(cudaMalloc(&dx, hx.size() * sizeof(float)), "dx");
        ck(cudaMemcpy(dx, hx.data(), hx.size() * sizeof(float), cudaMemcpyHostToDevice), "cpy dx");
        ck(cudaMalloc(&dxq, bk::native_q8_1_bytes(s.K, TMAX)), "dxq");
        bk::native_quantize_q8_1(dx, dxq, s.K, TMAX, st);
        ck(cudaMalloc(&y0, (size_t) 8 * s.M * sizeof(float)), "y0");
        ck(cudaMalloc(&y1, (size_t) TMAX * s.M * sizeof(float)), "y1");
        const size_t q81_col_bytes = bk::native_q8_1_bytes(s.K, 1);
        std::vector<float> single_ref((size_t) TMAX * s.M);
        for (int c = 0; c < TMAX; ++c) {
            bk::native_q8_planes_mmvq(dpl, static_cast<const char*>(dxq) + (size_t) c * q81_col_bytes, y1 + (size_t) c * s.M, s.K, s.M, 1, st);
        }
        ck(cudaStreamSynchronize(st), "sync single ref");
        ck(cudaMemcpy(single_ref.data(), y1, single_ref.size() * 4, cudaMemcpyDeviceToHost), "d2h single ref");
        for (int T = 1; T <= TMAX; ++T) {
            if (T > 8 && T != 9 && T != 15 && T != 16 && T != 17 && T != 24 && T != 25 && T != 31 && T != 32 && T != 33 && T != 48 && T != 49 && T != 63 && T != 64) continue;
            bk::native_q8_planes_mmvq(dpl, dxq, y1, s.K, s.M, T, st);
            if (T <= 8) bk::native_q8_0_mmvq(draw, dxq, y0, s.K, s.M, T, st);
            ck(cudaStreamSynchronize(st), "sync mmvq");
            std::vector<float> b((size_t) T * s.M);
            ck(cudaMemcpy(b.data(), y1, b.size() * 4, cudaMemcpyDeviceToHost), "d2h b");
            if (std::memcmp(single_ref.data(), b.data(), b.size() * sizeof(float)) != 0) {
                std::printf("FAIL native_q8_planes_mmvq row-invariance %s T=%d (not bitwise identical to T=1)\n", s.name, T);
                std::exit(1);
            }
            if (T <= 8) {
                std::vector<float> a((size_t) T * s.M);
                ck(cudaMemcpy(a.data(), y0, a.size() * 4, cudaMemcpyDeviceToHost), "d2h a");
                double max_abs = 0.0, max_ref = 0.0;
                for (size_t i = 0; i < a.size(); ++i) {
                    max_abs = std::max(max_abs, std::abs((double) a[i] - (double) b[i]));
                    max_ref = std::max(max_ref, std::abs((double) a[i]));
                }
                const double rel = max_ref > 0.0 ? max_abs / max_ref : max_abs;
                if (rel > 1e-5) {
                    std::printf("FAIL native_q8_planes_mmvq %s T=%d max_abs=%.3e rel=%.3e\n", s.name, T, max_abs, rel);
                    std::exit(1);
                }
            }
        }
        std::printf("PASS native_q8_planes_mmvq %s T=1..64: bitwise row-invariant & matches native_q8_0_mmvq (rel <= 1e-5)\n", s.name);
        if (s.M == 8664) {
            bk::Q80Segments sg0, sg1;
            sg0.n = sg1.n = 3;
            sg0.row0[0] = sg1.row0[0] = 0;    sg0.n_out[0] = sg1.n_out[0] = 6552;
            sg0.row0[1] = sg1.row0[1] = 6552; sg0.n_out[1] = sg1.n_out[1] = 1056;
            sg0.row0[2] = sg1.row0[2] = 7608; sg0.n_out[2] = sg1.n_out[2] = 1056;
            for (int k = 0; k < 3; ++k) sg0.x[k] = sg1.x[k] = dxq;
            sg0.y[0] = y0; sg0.y[1] = y0 + 8 * 6552; sg0.y[2] = y0 + 8 * 7608;
            sg1.y[0] = y1; sg1.y[1] = y1 + TMAX * 6552; sg1.y[2] = y1 + TMAX * 7608;
            std::vector<float> seg_ref((size_t) TMAX * s.M);
            for (int c = 0; c < TMAX; ++c) {
                bk::Q80Segments sgc = sg1;
                for (int k = 0; k < 3; ++k) sgc.x[k] = static_cast<const char*>(dxq) + (size_t) c * q81_col_bytes;
                sgc.y[0] = y1 + (size_t) c * 6552;
                sgc.y[1] = y1 + TMAX * 6552 + (size_t) c * 1056;
                sgc.y[2] = y1 + TMAX * 7608 + (size_t) c * 1056;
                bk::native_q8_planes_mmvq_segments(dpl, s.M, sgc, s.K, 1, st);
            }
            ck(cudaStreamSynchronize(st), "sync seg ref");
            ck(cudaMemcpy(seg_ref.data(), y1, seg_ref.size() * 4, cudaMemcpyDeviceToHost), "d2h seg ref");
            for (int T = 1; T <= TMAX; ++T) {
                if (T > 8 && T != 9 && T != 15 && T != 16 && T != 17 && T != 24 && T != 25 && T != 31 && T != 32 && T != 33 && T != 48 && T != 49 && T != 63 && T != 64) continue;
                if (T <= 8) bk::native_q8_0_mmvq_segments(draw, sg0, s.K, T, st);
                bk::native_q8_planes_mmvq_segments(dpl, s.M, sg1, s.K, T, st);
                ck(cudaStreamSynchronize(st), "sync seg");
                std::vector<float> b((size_t) TMAX * s.M);
                ck(cudaMemcpy(b.data(), y1, b.size() * 4, cudaMemcpyDeviceToHost), "d2h sb");
                std::vector<float> a;
                if (T <= 8) {
                    a.resize((size_t) 8 * s.M);
                    ck(cudaMemcpy(a.data(), y0, a.size() * 4, cudaMemcpyDeviceToHost), "d2h sa");
                }
                auto check_range = [&](size_t off_ref, size_t off_a, size_t cnt, const char* tag) {
                    if (std::memcmp(seg_ref.data() + off_ref, b.data() + off_ref, cnt * sizeof(float)) != 0) {
                        std::printf("FAIL seg row-invariance %s T=%d (not bitwise identical to T=1)\n", tag, T);
                        std::exit(1);
                    }
                    if (T <= 8) {
                        double ma = 0.0, mr = 0.0;
                        for (size_t i = 0; i < cnt; ++i) {
                            ma = std::max(ma, std::abs((double) a[off_a + i] - (double) b[off_ref + i]));
                            mr = std::max(mr, std::abs((double) a[off_a + i]));
                        }
                        if ((mr > 0.0 ? ma / mr : ma) > 1e-5) {
                            std::printf("FAIL seg %s T=%d rel=%.3e\n", tag, T, mr > 0.0 ? ma / mr : ma);
                            std::exit(1);
                        }
                    }
                };
                check_range(0, 0, (size_t) T * 6552, "Q");
                check_range(TMAX * 6552, 8 * 6552, (size_t) T * 1056, "K");
                check_range(TMAX * 7608, 8 * 7608, (size_t) T * 1056, "V");
            }
            std::printf("PASS native_q8_planes_mmvq_segments [6552,1056,1056]x2304 T=1..64: bitwise row-invariant & matches (rel <= 1e-5)\n");
        }
        cudaFree(draw); cudaFree(dpl); cudaFree(dx); cudaFree(dxq); cudaFree(y0); cudaFree(y1);
    }
    cudaStreamDestroy(st);
}

void run_bench(std::mt19937& rng) {
    struct Shape {
        const char* name;
        int64_t ntok, rows, cols;
    };
    const Shape shapes[] = {
        {"w.proj fused [8664x2304] T=4096", 4096, 8664, 2304},
        {"w.proj Q     [6552x2304] T=4096", 4096, 6552, 2304},
        {"w.proj K/V   [1056x2304] T=4096", 4096, 1056, 2304},
        {"w.out        [2304x3072] T=4096", 4096, 2304, 3072},
        {"w.proj fused [8664x2304] T=2048", 2048, 8664, 2304},
        {"w.out        [2304x3072] T=2048", 2048, 2304, 3072},
    };
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "e0");
    ck(cudaEventCreate(&e1), "e1");

    for (const auto& s : shapes) {
        std::vector<float> x((size_t) (s.ntok * s.cols), 0.25f);
        std::vector<float> wf((size_t) (s.rows * s.cols));
        for (float& v : wf) v = dist(rng);
        std::vector<block_q8_0> wb;
        make_q8_0(wf, s.rows, s.cols, wb);
        GpuWeight gw;
        gw.init(wb, s.rows, s.cols);

        void* d_raw = nullptr;
        const size_t pad = basalt::prefill::mmq::read_past_bytes(8, s.cols);
        ck(cudaMalloc(&d_raw, wb.size() * sizeof(block_q8_0) + pad), "raw");
        ck(cudaMemset(d_raw, 0, wb.size() * sizeof(block_q8_0) + pad), "zero raw");
        ck(cudaMemcpy(d_raw, wb.data(), wb.size() * sizeof(block_q8_0), cudaMemcpyHostToDevice), "cpy raw");

        float *d_x = nullptr, *d_y = nullptr;
        void *d_act = nullptr, *d_q81 = nullptr;
        int32_t *d_bounds = nullptr, *d_ids = nullptr;
        ck(cudaMalloc(&d_x, x.size() * sizeof(float)), "x");
        ck(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "cpy x");
        ck(cudaMalloc(&d_act, q8::act_bytes(s.ntok, s.cols)), "act");
        ck(cudaMalloc(&d_q81, basalt::prefill::mmq::q8_bytes(s.ntok, s.cols)), "q81");
        ck(cudaMalloc(&d_y, (size_t) (s.ntok * s.rows) * sizeof(float)), "y");
        ck(cudaMalloc(&d_bounds, 2 * sizeof(int32_t)), "bounds");
        ck(cudaMalloc(&d_ids, (size_t) s.ntok * sizeof(int32_t)), "ids");
        const int32_t hb[2] = {0, (int32_t) s.ntok};
        ck(cudaMemcpy(d_bounds, hb, sizeof(hb), cudaMemcpyHostToDevice), "cpy bounds");
        basalt::prefill::mmq::iota(d_ids, s.ntok, nullptr);

        q8::quantize(d_x, s.ntok, s.cols, s.cols, d_act, nullptr);
        basalt::prefill::mmq::quantize(d_x, nullptr, d_q81, 8, s.cols, s.cols, s.ntok, nullptr);
        basalt::prefill::mmq::Context mmq_ctx;
        basalt::prefill::mmq::Product prod{};
        prod.w = d_raw;
        prod.type = 8;
        prod.w_rows = s.rows;
        prod.w_cols = s.cols;
        prod.expert_bytes = wb.size() * sizeof(block_q8_0);
        prod.n = 1;
        prod.xq = d_q81;
        prod.bounds = d_bounds;
        prod.ids = d_ids;
        prod.total_rows = s.ntok;
        prod.max_rows = s.ntok;
        prod.dst = d_y;
        prod.ld_dst = s.rows;

        const int warm = 10, iters = 40;
        for (int i = 0; i < warm; ++i)
            q8::gemm(gw.codes(), gw.scales_t, s.rows, s.cols, d_act, s.ntok, d_y, s.rows, q8::Epilogue::Store, nullptr);
        ck(cudaEventRecord(e0), "r0");
        for (int i = 0; i < iters; ++i)
            q8::gemm(gw.codes(), gw.scales_t, s.rows, s.cols, d_act, s.ntok, d_y, s.rows, q8::Epilogue::Store, nullptr);
        ck(cudaEventRecord(e1), "r1");
        ck(cudaEventSynchronize(e1), "sync");
        float ms_q8 = 0.0f;
        ck(cudaEventElapsedTime(&ms_q8, e0, e1), "el");
        ms_q8 /= iters;

        for (int i = 0; i < warm; ++i) mmq_ctx.run(prod, nullptr);
        ck(cudaEventRecord(e0), "r0");
        for (int i = 0; i < iters; ++i) mmq_ctx.run(prod, nullptr);
        ck(cudaEventRecord(e1), "r1");
        ck(cudaEventSynchronize(e1), "sync");
        float ms_mmq = 0.0f;
        ck(cudaEventElapsedTime(&ms_mmq, e0, e1), "el");
        ms_mmq /= iters;

        // Quantize cost
        for (int i = 0; i < warm; ++i) q8::quantize(d_x, s.ntok, s.cols, s.cols, d_act, nullptr);
        ck(cudaEventRecord(e0), "r0");
        for (int i = 0; i < iters; ++i) q8::quantize(d_x, s.ntok, s.cols, s.cols, d_act, nullptr);
        ck(cudaEventRecord(e1), "r1");
        ck(cudaEventSynchronize(e1), "sync");
        float ms_quant = 0.0f;
        ck(cudaEventElapsedTime(&ms_quant, e0, e1), "el");
        ms_quant /= iters;

        const double ops = 2.0 * (double) s.ntok * (double) s.rows * (double) s.cols;
        const double tops_q8 = ops / (ms_q8 * 1e9);
        const double tops_mmq = ops / (ms_mmq * 1e9);
        std::printf("%-34s  q8=%6.1f us (%5.1f TOP/s)  mmq=%6.1f us (%5.1f TOP/s)  speedup=%.2fx  quant=%.1f us\n",
                    s.name, ms_q8 * 1000.0, tops_q8, ms_mmq * 1000.0, tops_mmq, ms_mmq / ms_q8, ms_quant * 1000.0);

        cudaFree(d_raw);
        cudaFree(d_x);
        cudaFree(d_act);
        cudaFree(d_q81);
        cudaFree(d_y);
        cudaFree(d_bounds);
        cudaFree(d_ids);
        gw.destroy();
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
}

}  // namespace

int main(int argc, char** argv) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess || prop.major < 12) {
        std::printf("SKIP: requires Blackwell (sm_120+)\n");
        return 77;
    }
    std::printf("GPU: %s (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    std::mt19937 rng(42);
    test_case(rng, 1, 18, 64, false);
    test_case(rng, 7, 66, 192, false);
    test_case(rng, 63, 64, 128, true, 4);
    test_case(rng, 65, 130, 320, true, 2);
    test_case(rng, 130, 1056, 2304, false);
    test_case(rng, 130, 2304, 3072, true);
    test_case(rng, 257, 6552, 2304, false);
    test_invariance_and_segmented(rng);
    test_q8_planes_mmvq_parity(rng);
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0) run_bench(rng);
    }
    std::printf("ALL Q8 GEMM TESTS PASSED\n");
    return 0;
}
