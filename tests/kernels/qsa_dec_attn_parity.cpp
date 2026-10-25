// tests/kernels/qsa_dec_attn_parity.cpp - qsa_dec_attn against qsa_decode_attn_batch (+ native_qsa_gate_apply), bitwise,
// on random INT8 and FP16 pools through a permuted page table with masked pages; then a microbench of both.
//   qsa_dec_attn_parity [iters]
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/native_qsa.hpp"
#include "basalt/kernels/qsa.hpp"
#include "basalt/kernels/qsa_dec_attn.hpp"
#include "basalt/kernels/qsa_decode_attn.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = basalt::kernels;

namespace {
int g_fail = 0;
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 256), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 256), "memset");
    return p;
}
template <typename T> void up(T* d, const std::vector<T>& h) { ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "up"); }
template <typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "down");
    return h;
}
uint16_t f16(float x) { __half h = __float2half(x); uint16_t b; std::memcpy(&b, &h, 2); return b; }

struct Pools {
    int8_t *kq = nullptr, *vq = nullptr;
    uint16_t *ks = nullptr, *vs = nullptr, *kp = nullptr, *vp = nullptr;
    int32_t* table = nullptr;
};
}  // namespace

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 200;
    const k::QsaShapes s = k::qsa_real_shapes();
    const int H = (int) s.n_head_kv, D = (int) s.head_dim, NH = (int) s.n_head, P = (int) s.page_size;
    const int max_cells = 40000, pages = max_cells / P, MAXQ = 8;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.002f, 0.05f);
    const size_t rows = (size_t) pages * H * P;
    Pools p;
    p.kq = dalloc<int8_t>(rows * D); p.vq = dalloc<int8_t>(rows * D);
    p.ks = dalloc<uint16_t>(rows * D / 64); p.vs = dalloc<uint16_t>(rows * D / 64);
    p.kp = dalloc<uint16_t>(rows * D); p.vp = dalloc<uint16_t>(rows * D);
    p.table = dalloc<int32_t>(pages);
    {
        std::vector<int8_t> c(rows * D);
        std::vector<uint16_t> sc(rows * D / 64), hp(rows * D);
        for (int which = 0; which < 2; ++which) {
            for (auto& x : c) x = (int8_t) ((int) (rng() % 255) - 127);
            for (auto& x : sc) x = f16(ud(rng));
            for (auto& x : hp) x = f16(nd(rng) * 0.7f);
            up(which ? p.vq : p.kq, c);
            up(which ? p.vs : p.ks, sc);
            up(which ? p.vp : p.kp, hp);
        }
    }
    std::vector<int32_t> table(pages);
    for (int i = 0; i < pages; ++i) table[i] = i;
    std::shuffle(table.begin(), table.end(), rng);
    float* q = dalloc<float>((size_t) MAXQ * NH * D);
    float* qfull = dalloc<float>((size_t) MAXQ * NH * 2 * D);
    {
        std::vector<float> h((size_t) MAXQ * NH * D), hf((size_t) MAXQ * NH * 2 * D);
        for (auto& x : h) x = nd(rng) * 2.0f;   // scores of a few units: many probabilities far below the max
        for (auto& x : hf) x = nd(rng) * 3.0f;
        up(q, h);
        up(qfull, hf);
    }
    int32_t* steps = dalloc<int32_t>((size_t) MAXQ * k::kStepCount);
    float* out_ref = dalloc<float>((size_t) MAXQ * NH * D);
    float* out_new = dalloc<float>((size_t) MAXQ * NH * D);
    float* gated_ref = dalloc<float>((size_t) MAXQ * NH * D);
    float* gated_new = dalloc<float>((size_t) MAXQ * NH * D);
    float* out_fm = dalloc<float>((size_t) MAXQ * NH * D);
    float* gated_fm = dalloc<float>((size_t) MAXQ * NH * D);
    const size_t q8b = k::native_q8_1_bytes(NH * D, MAXQ);
    uint8_t* xq_ref = dalloc<uint8_t>(q8b);
    uint8_t* xq_fm = dalloc<uint8_t>(q8b);
    cudaStream_t st;
    ck(cudaStreamCreate(&st), "stream");

    auto pools_of = [&](int mode) {
        k::QsaAttnPools a;
        a.page_table = p.table;
        if (mode == 1) { a.k_q = p.kq; a.v_q = p.vq; a.k_scale = p.ks; a.v_scale = p.vs; }
        else { a.k_pool = p.kp; a.v_pool = p.vp; }
        return a;
    };
    // one case: n_q queries of widths w[i] over cap; masked: some pages -1; ident: the identity selection
    auto run_case = [&](int mode, int cap, int n_q, const std::vector<int>& w, bool masked, bool ident, bool bench,
                        const char* label) {
        std::vector<int32_t> tb = table;
        if (masked) for (int i = 0; i < pages; i += 7) tb[i] = -1;
        up(p.table, tb);
        std::vector<int32_t> hs((size_t) n_q * k::kStepCount), hid((size_t) n_q * cap, 0);
        for (int i = 0; i < n_q; ++i) {
            const int n_kv = ident ? w[i] : std::min(max_cells, w[i] * 3 + 11);
            k::qsa_step_fill(hs.data() + i * k::kStepCount, n_kv - 1, s);
            hs[i * k::kStepCount + k::kStepWidth] = w[i];
            std::vector<int32_t> cells(n_kv);
            for (int c = 0; c < n_kv; ++c) cells[c] = c;
            if (!ident) { std::shuffle(cells.begin(), cells.end(), rng); cells.resize(w[i]); std::sort(cells.begin(), cells.end()); }
            for (int c = 0; c < w[i]; ++c) hid[(size_t) i * cap + c] = cells[c];
        }
        int32_t* ids = dalloc<int32_t>((size_t) n_q * cap);
        up(ids, hid);
        up(steps, hs);
        const size_t sf = k::qsa_decode_attn_scratch_floats(cap, s);
        float* scr_ref = dalloc<float>(sf * n_q);
        float* scr_new = dalloc<float>(sf * n_q);
        const k::QsaAttnPools a = pools_of(mode);
        k::qsa_decode_attn_batch(q, a, ids, steps, cap, s, scr_ref, out_ref, n_q, st);
        k::native_qsa_gate_apply(out_ref, qfull, gated_ref, NH * n_q, D, st);
        k::qsa_dec_attn(q, a, ids, steps, cap, s, scr_new, out_new, n_q, st, qfull, gated_new);
        k::native_quantize_q8_1(gated_ref, xq_ref, NH * D, (int) n_q, st);
        k::qsa_dec_attn(q, a, ids, steps, cap, s, scr_new, out_fm, n_q, st, qfull, gated_fm, xq_fm);
        ck(cudaStreamSynchronize(st), "sync");
        {   // the merge writing the q8_1 rows too
            const size_t nn = (size_t) n_q * NH * D, nb = k::native_q8_1_bytes(NH * D, (int) n_q);
            const auto r0 = down(out_ref, nn), x0 = down(out_fm, nn), g0 = down(gated_ref, nn), g1 = down(gated_fm, nn);
            const auto q0 = down(xq_ref, nb), q1 = down(xq_fm, nb);
            size_t b0 = 0, b1 = 0, b2 = 0;
            for (size_t i = 0; i < nn; ++i) {
                b0 += std::memcmp(&r0[i], &x0[i], 4) != 0;
                b1 += std::memcmp(&g0[i], &g1[i], 4) != 0;
            }
            for (size_t i = 0; i < nb; ++i) b2 += q0[i] != q1[i];
            if (b0 || b1 || b2) ++g_fail;
            std::printf("%-34s merge + q8_1 rows: %s (attn %zu / gated %zu / q8_1 bytes %zu differ)\n", label,
                        b0 || b1 || b2 ? "FAIL" : "bitwise", b0, b1, b2);
        }
        const size_t n = (size_t) n_q * NH * D;
        const auto r = down(out_ref, n), x = down(out_new, n), gr = down(gated_ref, n), gx = down(gated_new, n);
        size_t bad = 0, gbad = 0;
        double maxd = 0;
        for (size_t i = 0; i < n; ++i) {
            if (std::memcmp(&r[i], &x[i], 4) != 0) { ++bad; maxd = std::max(maxd, (double) std::fabs(r[i] - x[i])); }
            if (std::memcmp(&gr[i], &gx[i], 4) != 0) ++gbad;
        }
        if (bad || gbad) ++g_fail;
        std::printf("%-34s mode %d cap %5d nq %d: %s (attn %zu / gated %zu of %zu differ, max |d| %.3g)\n", label, mode, cap,
                    n_q, bad || gbad ? "FAIL" : "bitwise", bad, gbad, n, maxd);
        if (bench) {
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            double bytes = 0;
            for (int i = 0; i < n_q; ++i)
                bytes += (double) w[i] * H * (mode == 1 ? (2.0 * D + 2.0 * (D / 64) * 2) : 4.0 * D) + 4.0 * w[i];
            float ms_ref = 0, ms_new = 0;
            for (int rep = 0; rep < 2; ++rep) {
                cudaEventRecord(e0, st);
                for (int it = 0; it < iters; ++it) {
                    k::qsa_decode_attn_batch(q, a, ids, steps, cap, s, scr_ref, out_ref, n_q, st);
                    k::native_qsa_gate_apply(out_ref, qfull, gated_ref, NH * n_q, D, st);
                }
                cudaEventRecord(e1, st);
                cudaEventSynchronize(e1);
                cudaEventElapsedTime(&ms_ref, e0, e1);
                cudaEventRecord(e0, st);
                for (int it = 0; it < iters; ++it)
                    k::qsa_dec_attn(q, a, ids, steps, cap, s, scr_new, out_new, n_q, st, qfull, gated_new);
                cudaEventRecord(e1, st);
                cudaEventSynchronize(e1);
                cudaEventElapsedTime(&ms_new, e0, e1);
            }
            float ms_fm = 0;
            cudaEventRecord(e0, st);
            for (int it = 0; it < iters; ++it)
                k::qsa_dec_attn(q, a, ids, steps, cap, s, scr_new, out_fm, n_q, st, qfull, gated_fm, xq_fm);
            cudaEventRecord(e1, st);
            cudaEventSynchronize(e1);
            cudaEventElapsedTime(&ms_fm, e0, e1);
            std::printf("   bench: with the q8_1 rows %7.2f us\n", 1e3 * ms_fm / iters);
            const double ur = 1e3 * ms_ref / iters, un = 1e3 * ms_new / iters;
            std::printf("   bench: ref (attn+merge+gate) %7.2f us %6.1f GB/s | new %7.2f us %6.1f GB/s | %.2fx\n", ur,
                        bytes / ur / 1e3, un, bytes / un / 1e3, ur / un);
        }
        cudaFree(ids);
        cudaFree(scr_ref);
        cudaFree(scr_new);
    };
    const int cap = (int) k::qsa_selection_width(k::kTopkMaxCells, s);
    for (int mode = 1; mode >= 0; --mode) {
        run_case(mode, cap, 1, {1}, false, true, false, "first cell");
        run_case(mode, cap, 3, {63, 64, 65}, false, true, false, "chunk edges");
        run_case(mode, cap, 3, {1250, 1251, 1252}, false, true, true, "prose window (identity)");
        run_case(mode, cap, 2, {cap, cap}, false, false, true, "full selection");
        run_case(mode, cap, 5, {cap, 1700, 900, 300, 2000}, true, false, false, "masked pages");
        run_case(mode, cap, 8, {cap, cap, cap, cap, cap, cap, cap, cap}, false, false, true, "T=8 full");
        run_case(mode, 4096, 1, {4096}, false, true, true, "draft window 4096");
        run_case(mode, 4096, 1, {1250}, false, true, true, "draft window, 1250 cells");
    }

    // ---- the draft layer's wide window (qsa_dec_attn takes tensor-core chunks of up to 256 cells there, not bitwise):
    // against the reference batch kernel and a double-precision CPU reference
    {
        const auto hkq = down(p.kq, (size_t) rows * D), hvq = down(p.vq, (size_t) rows * D);
        const auto hks = down(p.ks, (size_t) rows * D / 64), hvs = down(p.vs, (size_t) rows * D / 64);
        const auto hq = down(q, (size_t) NH * D);
        auto h2f = [](uint16_t b) { __half h; std::memcpy(&h, &b, 2); return (double) __half2float(h); };
        const int cap = 32768;
        const size_t sf = k::qsa_decode_attn_scratch_floats(cap, s);
        float* scr = dalloc<float>(sf);
        int32_t* ids = dalloc<int32_t>(cap);
        std::vector<int32_t> hid(cap);
        const k::QsaAttnPools a = pools_of(1);
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        char* flush = nullptr;
        const size_t fn = 256u << 20;
        cudaMalloc(&flush, fn);
        for (int masked = 0; masked < 2; ++masked)
            for (const int w : {32768, 20000, 1250, 257, 64, 63, 1}) {
                std::vector<int32_t> tb = table;
                if (masked) for (int i = 0; i < pages; i += 7) tb[i] = -1;
                up(p.table, tb);
                // the drafter's window: the last w of a longer context, cells ascending
                const int n_kv = std::min(max_cells, w + 777), start = n_kv - w;
                std::vector<int32_t> hs(k::kStepCount);
                k::qsa_step_fill(hs.data(), n_kv - 1, s);
                hs[k::kStepWidth] = w;
                for (int c = 0; c < w; ++c) hid[c] = start + c;
                up(ids, hid);
                up(steps, hs);
                std::vector<float> out[2];
                k::qsa_decode_attn_batch(q, a, ids, steps, cap, s, scr, out_ref, 1, st);
                k::qsa_dec_attn(q, a, ids, steps, cap, s, scr, out_new, 1, st);
                ck(cudaStreamSynchronize(st), "sync");
                out[0] = down(out_ref, (size_t) NH * D);
                out[1] = down(out_new, (size_t) NH * D);
                // CPU double reference
                double max_ref = 0, e_old = 0, e_new = 0, e_vs_old = 0;
                for (int h = 0; h < NH; ++h) {
                    const int kvh = h / 12;
                    std::vector<double> sc;
                    std::vector<int> rowv;
                    for (int c = 0; c < w; ++c) {
                        const int cell = start + c;
                        const int page = tb[cell / P];
                        if (page < 0) continue;
                        const size_t r = ((size_t) page * H + kvh) * P + cell % P;
                        double dot = 0;
                        for (int d = 0; d < D; ++d)
                            dot += (double) hq[(size_t) h * D + d] * (double) hkq[r * D + d] * h2f(hks[r * (D / 64) + d / 64]);
                        sc.push_back(dot / 16.0);
                        rowv.push_back((int) r);
                    }
                    double mx = -1e300;
                    for (double x : sc) mx = std::max(mx, x);
                    double l = 0;
                    std::vector<double> acc(D, 0.0);
                    for (size_t i = 0; i < sc.size(); ++i) {
                        const double pw = std::exp(sc[i] - mx);
                        l += pw;
                        const size_t r = (size_t) rowv[i];
                        for (int d = 0; d < D; ++d) acc[d] += pw * (double) hvq[r * D + d] * h2f(hvs[r * (D / 64) + d / 64]);
                    }
                    for (int d = 0; d < D; ++d) {
                        const double ref = l > 0 ? acc[d] / l : 0.0;
                        max_ref = std::max(max_ref, std::fabs(ref));
                        e_old = std::max(e_old, std::fabs(ref - out[0][(size_t) h * D + d]));
                        e_new = std::max(e_new, std::fabs(ref - out[1][(size_t) h * D + d]));
                        e_vs_old = std::max(e_vs_old, (double) std::fabs(out[0][(size_t) h * D + d] - out[1][(size_t) h * D + d]));
                    }
                }
                // the same call again and again: the same bits every time
                size_t unstable = 0;
                for (int rep = 0; rep < 200; ++rep) {
                    k::qsa_dec_attn(q, a, ids, steps, cap, s, scr, out_ref, 1, st);
                    ck(cudaStreamSynchronize(st), "sync");
                    const auto again = down(out_ref, (size_t) NH * D);
                    if (std::memcmp(again.data(), out[1].data(), again.size() * 4) != 0) ++unstable;
                }
                if (unstable) std::printf("draft attention w=%d: %zu of 200 repeats differ\n", w, unstable);
                if (unstable) ++g_fail;
                const bool ok = e_new <= 4e-3 * std::max(max_ref, 1e-6);
                if (!ok) ++g_fail;
                std::printf("draft attention w=%5d%s: max|ref| %.3g | reference kernel err %.3g | tensor-core err %.3g (%.2g of max) | new vs old %.3g  %s\n",
                            w, masked ? " masked" : "       ", max_ref, e_old, e_new, e_new / std::max(max_ref, 1e-30), e_vs_old, ok ? "ok" : "FAIL");
            }
        // microbench: one row, window 32768 and 1250, keys warm / flushed
        {
            std::vector<int32_t> tb = table;
            up(p.table, tb);
            for (const int w : {32768, 1250}) {
                const int n_kv = w;
                std::vector<int32_t> hs(k::kStepCount);
                k::qsa_step_fill(hs.data(), n_kv - 1, s);
                hs[k::kStepWidth] = w;
                for (int c = 0; c < w; ++c) hid[c] = c;
                up(ids, hid);
                up(steps, hs);
                for (int on = 0; on < 2; ++on) {
                    for (int cold = 0; cold < 2; ++cold) {
                        double tot = 0;
                        const int it_n = cold ? 50 : iters;
                        auto call = [&] {
                            if (on) k::qsa_dec_attn(q, a, ids, steps, cap, s, scr, out_new, 1, st, qfull, gated_new);
                            else {
                                k::qsa_decode_attn_batch(q, a, ids, steps, cap, s, scr, out_ref, 1, st);
                                k::native_qsa_gate_apply(out_ref, qfull, gated_ref, NH, D, st);
                            }
                        };
                        if (!cold) cudaEventRecord(e0, st);
                        for (int it = 0; it < it_n; ++it) {
                            if (cold) {
                                cudaMemsetAsync(flush, it, fn, st);
                                cudaEventRecord(e0, st);
                            }
                            call();
                            if (cold) {
                                cudaEventRecord(e1, st);
                                cudaEventSynchronize(e1);
                                float ms = 0;
                                cudaEventElapsedTime(&ms, e0, e1);
                                tot += ms;
                            }
                        }
                        if (!cold) {
                            cudaEventRecord(e1, st);
                            cudaEventSynchronize(e1);
                            float ms = 0;
                            cudaEventElapsedTime(&ms, e0, e1);
                            tot = ms;
                        }
                        std::printf("   bench draft window %5d, %-18s %s: %7.2f us/call\n", w, on ? "tensor cores" : "reference kernel",
                                    cold ? "cold (L2 flushed, event-timed)" : "warm (back to back)      ", tot / it_n * 1000.0);
                    }
                }
            }
        }
        cudaFree(flush);
    }
    {   // microbench: the verify window at long context - 3 rows each selecting cap cells as random 4-cell blocks out of
        // the pools (a block's rows are contiguous), the pools' lines evicted before every call
        up(p.table, table);
        const int n_q = 3, n_kv = max_cells;
        std::vector<int32_t> hs((size_t) n_q * k::kStepCount), hid((size_t) n_q * cap, 0);
        for (int i = 0; i < n_q; ++i) {
            k::qsa_step_fill(hs.data() + i * k::kStepCount, n_kv - 1 - i, s);
            hs[i * k::kStepCount + k::kStepWidth] = cap;
            std::vector<int32_t> blocks(n_kv / 4);
            for (int b = 0; b < (int) blocks.size(); ++b) blocks[b] = b;
            std::shuffle(blocks.begin(), blocks.end(), rng);
            blocks.resize((cap + 3) / 4);
            std::sort(blocks.begin(), blocks.end());
            for (int c = 0; c < cap; ++c) hid[(size_t) i * cap + c] = blocks[c / 4] * 4 + c % 4;
        }
        int32_t* ids = dalloc<int32_t>((size_t) n_q * cap);
        up(ids, hid);
        up(steps, hs);
        float* scr = dalloc<float>(k::qsa_decode_attn_scratch_floats(cap, s) * n_q);
        const size_t fn = 256u << 20;
        char* flush = dalloc<char>(fn);
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        for (int mode = 1; mode >= 0; --mode) {
            const k::QsaAttnPools a = pools_of(mode);
            for (int cold = 0; cold < 2; ++cold) {
                double tot = 0;
                for (int it = 0; it < iters; ++it) {
                    if (cold) cudaMemsetAsync(flush, it, fn, st);
                    cudaEventRecord(e0, st);
                    k::qsa_dec_attn(q, a, ids, steps, cap, s, scr, out_new, n_q, st, qfull, gated_new, xq_fm);
                    cudaEventRecord(e1, st);
                    cudaEventSynchronize(e1);
                    float ms = 0;
                    cudaEventElapsedTime(&ms, e0, e1);
                    tot += ms;
                }
                std::printf("   bench verify window 3 x %d cells in blocks, mode %d, pools %-6s: %7.2f us/call (chunk + merge)\n", cap,
                            mode, cold ? "cold" : "warm", tot / iters * 1000.0);
            }
        }
    }
    std::printf("%s\n", g_fail ? "qsa_dec_attn_parity: FAIL" : "qsa_dec_attn_parity: OK (bitwise)");
    return g_fail ? 1 : 0;
}
