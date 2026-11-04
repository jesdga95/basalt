// tests/models/qwen4exp/kernels/ple_key_rows_parity.cpp - the verify window's PLE for every key form the packs use (BF16, native Q8_0,
// native Q2_0) at the real geometry: ple_key_rows against the per-row projection inside ple_block and a CPU reference;
// then the window (ple_key_rows, the multi-row value MMVF, native_ple_window, plain and segmented) against the
// per-row chain (native_gr_post, ple_block, ple_history_advance, the history copy), bitwise, for windows of 1..8 rows;
// then both forms captured as graphs and timed.
//   ple_key_rows_parity [iters]
#include "basalt/models/qwen4exp/kernels/native_gr_postops.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/models/qwen4exp/kernels/native_ple_postops.hpp"
#include "basalt/models/qwen4exp/kernels/ngram.hpp"
#include "basalt/models/qwen4exp/kernels/ple.hpp"
#include "basalt/kernels/bf16_gemv.hpp"
#include "basalt/models/qwen4exp/kernels/verify_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = basalt::kernels;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 256), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 256), "memset");
    return p;
}
// a pageable copy may return before it lands and the non-blocking test stream does not wait for it: finish it here
template <typename T> void up(T* d, const std::vector<T>& h) {
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "up");
    ck(cudaDeviceSynchronize(), "up sync");
}
template <typename T> std::vector<T> down(const T* d, size_t n) {
    ck(cudaDeviceSynchronize(), "down sync");
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "down");
    return h;
}
size_t ndiff(const std::vector<float>& a, const std::vector<float>& b) {
    size_t c = 0;
    for (size_t i = 0; i < a.size(); ++i) c += std::memcmp(&a[i], &b[i], 4) != 0;
    return c;
}
float bf16f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
uint16_t f2bf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t(u >> 16); }
float half_f(uint16_t h) { return __half2float(__ushort_as_half(h)); }
uint16_t f_half(float f) { return __half_as_ushort(__float2half(f)); }
}  // namespace

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 200;
    constexpr int N = k::NG_N_EMBD, H = k::NG_HC, D = k::NG_HC_DIM, HS = k::NG_HIST * k::NG_HC_DIM, MT = 8;
    std::mt19937 rng(196);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto randv = [&](size_t n, float sc, float off) { std::vector<float> v(n); for (auto& x : v) x = off + sc * nd(rng); return v; };

    // ---- the weights: a BF16, a Q8_0 and a Q2_0 key ([10240 rows][2560]), a BF16 value, the post-op weights
    std::vector<uint16_t> h_kb((size_t) D * N), h_vb((size_t) N * N);
    for (auto& x : h_kb) x = f2bf16(0.02f * nd(rng));
    for (auto& x : h_vb) x = f2bf16(0.02f * nd(rng));
    std::vector<uint8_t> h_k8((size_t) D * (N / 32) * 34), h_k2((size_t) D * (N / 64) * 18);
    for (size_t b = 0; b < h_k8.size() / 34; ++b) {
        const uint16_t d = f_half(0.0004f * (1.0f + 0.5f * std::fabs(nd(rng))));
        std::memcpy(&h_k8[b * 34], &d, 2);
        for (int i = 0; i < 32; ++i) h_k8[b * 34 + 2 + i] = (uint8_t) (int8_t) ((int) (rng() % 255) - 127);
    }
    for (size_t b = 0; b < h_k2.size() / 18; ++b) {
        const uint16_t d = f_half(0.01f * (1.0f + 0.5f * std::fabs(nd(rng))));
        std::memcpy(&h_k2[b * 18], &d, 2);
        for (int i = 0; i < 16; ++i) h_k2[b * 18 + 2 + i] = (uint8_t) rng();
    }
    std::vector<uint16_t> conv(4 * (size_t) D);
    for (auto& w : conv) w = f_half(0.3f * nd(rng));
    const auto h_R = randv((size_t) MT * D, 1.0f, 0.0f), h_bo = randv((size_t) MT * N, 0.5f, 0.0f),
               h_inj = randv((size_t) MT * H, 2.0f, 0.0f), h_emb = randv((size_t) MT * N, 1.0f, 0.0f),
               h_hist = randv((size_t) MT * HS, 1.0f, 0.0f), h_nk = randv(D, 0.1f, 1.0f),
               h_nq = randv(D, 0.1f, 1.0f), h_nc = randv(D, 0.1f, 1.0f);
    uint16_t *kb = dalloc<uint16_t>(h_kb.size()), *vb = dalloc<uint16_t>(h_vb.size()), *cw = dalloc<uint16_t>(conv.size());
    uint8_t *k8 = dalloc<uint8_t>(h_k8.size()), *k2 = dalloc<uint8_t>(h_k2.size());
    float *R = dalloc<float>((size_t) MT * D), *bo = dalloc<float>((size_t) MT * N), *inj = dalloc<float>((size_t) MT * H),
          *emb = dalloc<float>((size_t) MT * N), *hist = dalloc<float>((size_t) MT * HS),
          *snap = dalloc<float>((size_t) MT * HS), *nk = dalloc<float>(D), *nq = dalloc<float>(D), *nc = dalloc<float>(D),
          *key = dalloc<float>((size_t) MT * D), *val = dalloc<float>((size_t) MT * N), *key1 = dalloc<float>((size_t) MT * D),
          *val1 = dalloc<float>((size_t) MT * N), *win = dalloc<float>(2 * (size_t) MT * D);
    void* q8 = dalloc<uint8_t>(k::native_q8_1_bytes(N, MT));
    void* q8one = dalloc<uint8_t>(k::native_q8_1_bytes(N));
    k::SegRow* dseg = dalloc<k::SegRow>(MT);
    up(kb, h_kb); up(vb, h_vb); up(cw, conv); up(k8, h_k8); up(k2, h_k2); up(bo, h_bo); up(inj, h_inj); up(emb, h_emb);
    up(nk, h_nk); up(nq, h_nq); up(nc, h_nc);
    void* scratch = nullptr;
    ck(cudaMalloc(&scratch, k::ple_block_scratch_bytes() + (size_t) D * 4 + 256), "scratch");
    float* normalized = (float*) ((uint8_t*) scratch + k::ple_block_scratch_bytes());
    cudaStream_t st;
    ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");

    struct Form { const char* name; k::PleWeights w; };
    std::vector<Form> forms(3);
    for (auto& f : forms) {
        f.w.value_bf16 = vb; f.w.norm_key = nk; f.w.norm_query = nq; f.w.norm_conv = nc; f.w.conv1d_f16 = cw;
    }
    forms[0].name = "BF16"; forms[0].w.key_bf16 = kb;
    forms[1].name = "Q8_0"; forms[1].w.key_native_data = k8; forms[1].w.key_native_type = 8; forms[1].w.key_native_q8_1 = q8one;
    forms[2].name = "Q2_0"; forms[2].w.key_native_data = k2; forms[2].w.key_native_type = 42; forms[2].w.key_native_q8_1 = q8one;

    // the CPU reference of a key row: BF16 in double; Q8_0 as llama.cpp's q8_1 quantization of the row (scale amax/127,
    // stored as FP16) times the integer dot and the two scales
    auto cpu_key = [&](int form, const float* x, std::vector<double>& y) {
        y.assign(D, 0.0);
        std::vector<int> qx(N);
        std::vector<float> dx(N / 32);
        for (int b = 0; b < N / 32; ++b) {
            float amax = 0;
            for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(x[b * 32 + i]));
            const float d = amax / 127.0f;
            for (int i = 0; i < 32; ++i) qx[b * 32 + i] = amax == 0.0f ? 0 : (int) std::roundf(x[b * 32 + i] / d);
            dx[b] = half_f(f_half(d));
        }
        for (int o = 0; o < D; ++o) {
            double s = 0;
            if (form == 0) {
                for (int i = 0; i < N; ++i) s += (double) bf16f(h_kb[(size_t) o * N + i]) * x[i];
            } else {
                for (int b = 0; b < N / 32; ++b) {
                    const uint8_t* blk = &h_k8[((size_t) o * (N / 32) + b) * 34];
                    uint16_t dh;
                    std::memcpy(&dh, blk, 2);
                    int si = 0;
                    for (int i = 0; i < 32; ++i) si += (int) (int8_t) blk[2 + i] * qx[b * 32 + i];
                    s += (double) half_f(dh) * dx[b] * si;
                }
            }
            y[o] = s;
        }
    };

    int fail = 0;
    for (int f = 0; f < 3; ++f) {
        const k::PleWeights& w = forms[f].w;
        // ---- the projections: ple_key_rows and the multi-row value against ple_block's per-row calls
        for (int t = 0; t < MT; ++t) {
            if (f == 0) {
                k::bf16_gemv_fp32_mmvf(emb + (size_t) t * N, kb, key1 + (size_t) t * D, N, D, st);
            } else {
                k::native_quantize_q8_1(emb + (size_t) t * N, q8one, N, 1, st);
                k::native_mmvq(w.key_native_type, w.key_native_data, q8one, key1 + (size_t) t * D, N, D, 1, st);
            }
            k::bf16_gemv_fp32_mmvf(emb + (size_t) t * N, vb, val1 + (size_t) t * N, N, N, st);
        }
        const auto rk = down(key1, (size_t) MT * D), rv = down(val1, (size_t) MT * N);
        for (int T = 1; T <= MT; ++T) {
            ck(cudaMemsetAsync(key, 0xa5, (size_t) MT * D * 4, st), "poison");
            k::ple_key_rows(emb, T, w, key, q8, st);
            k::bf16_gemv_fp32_mmvf_multi(emb, N, vb, val, N, N, N, T, st);
            const auto gk = down(key, (size_t) T * D), gv = down(val, (size_t) T * N);
            const size_t dk = ndiff(gk, std::vector<float>(rk.begin(), rk.begin() + (long) gk.size())),
                         dv = ndiff(gv, std::vector<float>(rv.begin(), rv.begin() + (long) gv.size()));
            if (dk + dv != 0 || T == 1 || T == MT)
                std::printf("%s projections T=%d  key %zu  value %zu  differing values%s\n", forms[f].name, T, dk, dv,
                            dk + dv ? "  DIFF" : "");
            fail += dk + dv != 0;
        }
        // the CPU reference (Q2_0's arithmetic is native_mmvq_parity's)
        if (f < 2) {
            double worst = 0;
            for (int t = 0; t < MT; t += 7) {
                std::vector<double> ref;
                cpu_key(f, &h_emb[(size_t) t * N], ref);
                double amax = 0;
                for (double v : ref) amax = std::fmax(amax, std::fabs(v));
                for (int o = 0; o < D; ++o) worst = std::fmax(worst, std::fabs(ref[o] - rk[(size_t) t * D + o]) / amax);
            }
            const bool ok = worst < 1e-4;
            std::printf("%s key vs CPU reference: max error %.3e of the row's largest%s\n", forms[f].name, worst,
                        ok ? "" : "  FAIL");
            fail += !ok;
        }

        // ---- the window against the per-row chain the verify window ran for a key without the window kernels
        auto chain = [&](int r0, int n, float* h, float* sn) {
            for (int t = r0; t < r0 + n; ++t) {
                float* Rt = R + (size_t) t * D;
                k::native_gr_post(Rt, bo + (size_t) t * N, inj + (size_t) t * H, Rt, N, H, st);
                k::PleOut po;
                po.normalized = normalized;
                po.result = Rt;
                k::ple_block(emb + (size_t) t * N, Rt, h, w, po, scratch, st);
                k::ple_history_advance(h, normalized, st);
                ck(cudaMemcpyAsync(sn + (size_t) t * HS, h, (size_t) HS * 4, cudaMemcpyDeviceToDevice, st), "snap");
            }
        };
        auto window = [&](int T, const k::SegRow* segs) {
            k::ple_key_rows(emb, T, w, key, q8, st);
            k::bf16_gemv_fp32_mmvf_multi(emb, N, vb, val, N, N, N, T, st);
            k::PleWindowArgs a;
            a.R = R; a.block_out = bo; a.inject = inj; a.key = key; a.value = val; a.history = hist; a.hist_snap = snap;
            a.scratch = win; a.T = T; a.segs = segs;
            k::native_ple_window(a, w, st);
        };
        auto reset = [&] {
            ck(cudaDeviceSynchronize(), "sync");
            up(R, h_R); up(hist, h_hist);
            ck(cudaMemset(snap, 0, (size_t) MT * HS * 4), "memset");
            ck(cudaDeviceSynchronize(), "sync");
        };
        for (int T = 1; T <= MT; ++T) {
            for (int trial = 0; trial < 6; ++trial) {
                // trial 0: one sequence (a plain window); then segments of different sequences, one-row ones included
                std::vector<int> lens;
                for (int left = T; left > 0;) {
                    const int n = trial == 0 ? T : trial == 1 ? 1 : 1 + (int) (rng() % (unsigned) left);
                    lens.push_back(n);
                    left -= n;
                }
                std::vector<k::SegRow> tab((size_t) T);
                for (size_t s2 = 0, r0 = 0; s2 < lens.size(); r0 += (size_t) lens[s2], ++s2)
                    for (int j = 0; j < lens[s2]; ++j) {
                        k::SegRow& e = tab[r0 + (size_t) j];
                        std::memset(&e, 0, sizeof e);
                        e.row0 = (int32_t) r0; e.j = j; e.n = lens[s2]; e.n_keep = lens[s2];
                    }
                up(dseg, tab);
                std::vector<float> out[2][3];
                for (int v = 0; v < 2; ++v) {
                    reset();
                    if (v == 0) {
                        for (size_t s2 = 0, r0 = 0; s2 < lens.size(); r0 += (size_t) lens[s2], ++s2)
                            chain((int) r0, lens[s2], hist + (trial == 0 ? 0 : r0 * HS), snap);
                    } else {
                        window(T, trial == 0 ? nullptr : dseg);
                    }
                    out[v][0] = down(R, (size_t) T * D);
                    std::vector<float> hh = down(hist, (size_t) T * HS), keep;
                    for (size_t s2 = 0, r0 = 0; s2 < lens.size(); r0 += (size_t) lens[s2], ++s2)
                        keep.insert(keep.end(), hh.begin() + (long) (r0 * HS), hh.begin() + (long) ((r0 + 1) * HS));
                    out[v][1] = keep;
                    out[v][2] = down(snap, (size_t) T * HS);
                }
                const size_t d0 = ndiff(out[0][0], out[1][0]), d1 = ndiff(out[0][1], out[1][1]),
                             d2 = ndiff(out[0][2], out[1][2]);
                if (d0 + d1 + d2 != 0 || ((T == 1 || T == MT) && trial < 3)) {
                    std::printf("%s window T=%d [", forms[f].name, T);
                    for (int n : lens) std::printf(" %d", n);
                    std::printf(" ]  result %zu  history %zu  snapshots %zu  differing values%s\n", d0, d1, d2,
                                d0 + d1 + d2 ? "  DIFF" : "");
                }
                fail += d0 + d1 + d2 != 0;
            }
        }
        // ---- microbench: each form captured once per T (one sequence), replayed
        for (int T : {1, 4, 8}) {
            float ms[2];
            for (int v = 0; v < 2; ++v) {
                reset();
                cudaGraph_t g;
                cudaGraphExec_t ge;
                ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "capture");
                if (v == 0) chain(0, T, hist, snap); else window(T, nullptr);
                ck(cudaStreamEndCapture(st, &g), "end capture");
                ck(cudaGraphInstantiate(&ge, g, 0), "instantiate");
                for (int i = 0; i < 20; ++i) cudaGraphLaunch(ge, st);
                cudaEvent_t e0, e1;
                cudaEventCreate(&e0); cudaEventCreate(&e1);
                cudaEventRecord(e0, st);
                for (int i = 0; i < iters; ++i) cudaGraphLaunch(ge, st);
                cudaEventRecord(e1, st);
                ck(cudaEventSynchronize(e1), "bench");
                cudaEventElapsedTime(&ms[v], e0, e1);
                cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
                cudaEventDestroy(e0); cudaEventDestroy(e1);
            }
            std::printf("bench: %s key T=%d  per-row chain %.2f us  window %.2f us\n", forms[f].name, T,
                        1000.0f * ms[0] / iters, 1000.0f * ms[1] / iters);
        }
    }
    std::printf("%s\n", fail ? "PLE KEY ROWS PARITY FAILED" : "PLE KEY ROWS PARITY OK (bitwise)");
    return fail ? 1 : 0;
}
