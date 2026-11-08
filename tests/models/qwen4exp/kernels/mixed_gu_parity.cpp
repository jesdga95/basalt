// tests/models/qwen4exp/kernels/mixed_gu_parity.cpp - experts whose gate and up rows differ in type (UD-IQ1_S's layer
// of IQ2_S gate, IQ2_XXS up, Q2_0 down) on the decode kernels and the CPU tier, synthetic blobs at the model's shape.
//
//     build/mixed_gu_parity
//
// GPU (skipped without a device), windows of T = 1, 3, 6, 8 tokens over 9 experts:
//   halves     native_expert_grouped's gate rows BITWISE those of an all-IQ2_S expert with the same gate rows, its up
//              rows those of an all-IQ2_XXS expert with the same up rows (the kernels the uniform layers run)
//   peer math  the persistent kernel's items (dec_moe_peer_math_test, with and without the two-rows-a-warp gate/up)
//              BITWISE native_expert_grouped's rows
//   width      a token's rows the same alone in its group as among the window's
//   reference  the rows against a double-precision reference (ggml's dequantized weights and the q8_1 activations
//              as the kernels see them), next to the same error of the all-IQ2_S and all-IQ2_XXS experts
// CPU: native_gu_rows on a mixed layer (the AVX-512 or AVX2 row kernels, as the CPU has them) BITWISE the uniform
// row kernels of each half with native_gu_rows' SwiGLU, 1 to 8 tokens, every token as alone; and against the double
// reference.
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/iq_avx2.hpp"
#include "basalt/kernels/cpu/iq_avx512.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/models/qwen4exp/kernels/dec_moe.hpp"
#include "basalt/models/qwen4exp/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace k = basalt::kernels;
namespace cpu = basalt::kernels::cpu;

namespace {

constexpr int64_t H = 2560, FF = 640;
constexpr int TG = 22, TU = 16, TD = 42, NB = 9;
int g_fail = 0;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}
void check(bool ok, const std::string& what) {
    std::printf("  %-100s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
template<typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 256), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 256), "memset");
    return p;
}
template<typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}

// `rows` rows of `n` values: random codes, a finite fp16 block scale (2^-13 .. 2^-6) at every block's start
std::vector<uint8_t> random_rows(int t, int64_t rows, int64_t n, std::mt19937& rng) {
    const size_t rb = ggml_row_size((ggml_type) t, n), bs = ggml_type_size((ggml_type) t);
    std::vector<uint8_t> w((size_t) rows * rb);
    std::uniform_int_distribution<int> byte(0, 255), ex(2, 8), man(0, 1023), sgn(0, 1);
    for (auto& b : w) b = (uint8_t) byte(rng);
    for (size_t o = 0; o < w.size(); o += bs) {
        const uint16_t h = (uint16_t) ((sgn(rng) ? 0x8000 : 0) | (ex(rng) << 10) | man(rng));
        std::memcpy(&w[o], &h, 2);
    }
    return w;
}
std::vector<uint8_t> cat(std::initializer_list<const std::vector<uint8_t>*> parts) {
    std::vector<uint8_t> b;
    for (const auto* p : parts) b.insert(b.end(), p->begin(), p->end());
    return b;
}
std::vector<float> deq(int t, const uint8_t* w, int64_t rows, int64_t n) {
    std::vector<float> f((size_t) (rows * n));
    const auto* tr = ggml_get_type_traits((ggml_type) t);
    const size_t rb = ggml_row_size((ggml_type) t, n);
    for (int64_t r = 0; r < rows; ++r) tr->to_float(w + (size_t) r * rb, f.data() + (size_t) (r * n), n);
    return f;
}
double silu(double g) { return g / (1.0 + std::exp(-g)); }
// out[o] = down . (silu(gate . x) * (up . x)) in double; w*: dequantized rows
void reference(const std::vector<float>& wg, const std::vector<float>& wu, const std::vector<float>& wd,
               const float* x, double* out) {
    std::vector<double> h((size_t) FF);
    for (int64_t f = 0; f < FF; ++f) {
        double g = 0, u = 0;
        for (int64_t c = 0; c < H; ++c) {
            g += (double) wg[(size_t) (f * H + c)] * x[c];
            u += (double) wu[(size_t) (f * H + c)] * x[c];
        }
        h[(size_t) f] = silu(g) * u;
    }
    for (int64_t o = 0; o < H; ++o) {
        double a = 0;
        for (int64_t f = 0; f < FF; ++f) a += (double) wd[(size_t) (o * FF + f)] * h[(size_t) f];
        out[o] = a;
    }
}

struct Expert {
    std::vector<uint8_t> g, u, d;            // IQ2_S gate rows, IQ2_XXS up rows, Q2_0 down rows
    std::vector<float> wg, wu, wd;           // dequantized
};

// ------------------------------------------------------------------------------------------------ GPU
struct Err {
    double max_abs = 0, max_rel = 0;         // the worst |got - ref| and that over the row's max |ref|
};
void add_err(Err& e, const float* got, const double* ref) {
    double me = 0, mr = 0;
    for (int64_t o = 0; o < H; ++o) {
        me = std::max(me, std::fabs((double) got[o] - ref[o]));
        mr = std::max(mr, std::fabs(ref[o]));
    }
    e.max_abs = std::max(e.max_abs, me);
    if (mr > 0) e.max_rel = std::max(e.max_rel, me / mr);
}

void gpu_part(const std::vector<Expert>& ex, std::mt19937& rng) {
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    const k::NativeExpertLayout LM = k::native_expert_layout(TG, TD, H, FF, 0.0f, TU), LA = k::native_expert_layout(TG, TD, H, FF),
                                LB = k::native_expert_layout(TU, TD, H, FF);
    check(k::native_expert_supported(TG, TD, H, FF, TU) && k::dec_moe_formats_ok(TG, TD, H, FF, TU) &&
              k::dec_moe_glue_ok(LM) && !k::native_expert_supported(TG, TD, H, FF, 18) &&
              !k::dec_moe_formats_ok(TG, 20, H, FF, TU),
          "IQ2_S + IQ2_XXS / Q2_0 is covered (grouped, persistent, glue); other mixes are not");
    // the three layouts' blobs, one slot each per expert
    const size_t slot = 3 << 20;
    uint8_t* blobs = dalloc<uint8_t>(slot * 3 * NB);
    for (int b = 0; b < NB; ++b) {
        const Expert& e = ex[(size_t) b];
        const auto m = cat({&e.g, &e.u, &e.d}), a = cat({&e.g, &e.g, &e.d}), c = cat({&e.u, &e.u, &e.d});
        check(m.size() == LM.bytes && a.size() == LA.bytes && c.size() == LB.bytes, "blob sizes as the layouts say");
        ck(cudaMemcpy(blobs + slot * (size_t) (3 * b), m.data(), m.size(), cudaMemcpyHostToDevice), "blob");
        ck(cudaMemcpy(blobs + slot * (size_t) (3 * b + 1), a.data(), a.size(), cudaMemcpyHostToDevice), "blob");
        ck(cudaMemcpy(blobs + slot * (size_t) (3 * b + 2), c.data(), c.size(), cudaMemcpyHostToDevice), "blob");
    }
    const int MT = k::kVerifyMaxT, cap = k::kDmCap;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) (MT * H));
    for (auto& v : x) v = nd(rng);
    float* dx = dalloc<float>(x.size());
    ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    uint8_t* xq = dalloc<uint8_t>((size_t) (MT * (H / 32) * 36));
    k::quantize_q8_1_rows(dx, MT, H, xq, s);
    ck(cudaStreamSynchronize(s), "quantize");
    // the activations as the kernels see them: q8_1 dequantized
    std::vector<float> xs((size_t) (MT * H));
    {
        const auto q = down(xq, (size_t) (MT * (H / 32) * 36));
        for (int64_t b = 0; b < MT * H / 32; ++b) {
            const uint8_t* blk = q.data() + b * 36;
            const float d = ggml_fp16_to_fp32(*(const ggml_fp16_t*) blk);
            for (int i = 0; i < 32; ++i) xs[(size_t) (b * 32 + i)] = d * (float) (int8_t) blk[4 + i];
        }
    }
    const size_t scr_bytes = std::max(k::native_expert_scratch_bytes(cap, FF), k::dec_moe_peer_scratch_bytes((int) H, (int) FF));
    uint8_t* scr = dalloc<uint8_t>(scr_bytes);
    uint8_t* pscr = dalloc<uint8_t>(scr_bytes);
    float* out = dalloc<float>((size_t) cap * H);
    unsigned long long* ptr = dalloc<unsigned long long>(cap);
    int32_t *start = dalloc<int32_t>(cap + 1), *n = dalloc<int32_t>(1), *dst = dalloc<int32_t>(cap), *tok = dalloc<int32_t>(cap);
    const size_t fa = ((size_t) cap * FF * 4 + 255) & ~(size_t) 255;   // native_expert_grouped's scratch: gate, up
    Err em, ea, eb;
    for (const int T : {1, 3, 6, 8}) {
        // groups of 1..T entries on distinct experts, scattered destinations, random tokens
        std::vector<int> order(NB);
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);
        std::vector<int32_t> st(1, 0), tk, ds, gexp;
        for (int g = 0; g < NB && st.back() < cap; ++g) {
            const int m = std::min<int>(1 + (int) (rng() % (unsigned) T), cap - st.back());
            gexp.push_back(order[(size_t) g]);
            st.push_back(st.back() + m);
        }
        const int ng = (int) gexp.size(), ne = st.back();
        for (int e = 0; e < ne; ++e) tk.push_back((int32_t) (rng() % (unsigned) T));
        ds.resize((size_t) ne);
        std::iota(ds.begin(), ds.end(), 0);
        std::shuffle(ds.begin(), ds.end(), rng);
        ck(cudaMemcpy(start, st.data(), st.size() * 4, cudaMemcpyHostToDevice), "start");
        ck(cudaMemcpy(n, &ng, 4, cudaMemcpyHostToDevice), "n");
        ck(cudaMemcpy(dst, ds.data(), ds.size() * 4, cudaMemcpyHostToDevice), "dst");
        ck(cudaMemcpy(tok, tk.data(), tk.size() * 4, cudaMemcpyHostToDevice), "tok");
        // which: 0 mixed, 1 all IQ2_S, 2 all IQ2_XXS
        auto run = [&](int which, std::vector<float>& o, std::vector<float>& gate, std::vector<float>& up) {
            std::vector<unsigned long long> p;
            for (int e : gexp) p.push_back((unsigned long long) (blobs + slot * (size_t) (3 * e + which)));
            ck(cudaMemcpy(ptr, p.data(), p.size() * 8, cudaMemcpyHostToDevice), "ptr");
            ck(cudaMemset(out, 0xff, (size_t) cap * H * 4), "memset");
            k::native_expert_grouped(which == 0 ? LM : which == 1 ? LA : LB, ptr, start, n, dst, tok, cap, cap, xq, scr,
                                     out, s);
            ck(cudaStreamSynchronize(s), "grouped");
            o = down(out, (size_t) cap * H);
            gate = down((const float*) scr, (size_t) ne * FF);
            up = down((const float*) (scr + fa), (size_t) ne * FF);
        };
        std::vector<float> om, gm, um, oa, ga, ua, ob, gb, ub;
        run(1, oa, ga, ua);
        run(2, ob, gb, ub);
        run(0, om, gm, um);
        check(std::memcmp(gm.data(), ga.data(), gm.size() * 4) == 0 && std::memcmp(um.data(), ub.data(), um.size() * 4) == 0,
              "T " + std::to_string(T) + ", " + std::to_string(ng) + " groups: grouped gate rows = all-IQ2_S's, up rows = all-IQ2_XXS's");
        // the persistent kernel's items on the same plan (its pointers are the mixed blobs now)
        for (const bool g2 : {false, true}) {
            float* out2 = dalloc<float>((size_t) cap * H);
            ck(cudaMemset(out2, 0xff, (size_t) cap * H * 4), "memset");
            ck(cudaMemset(pscr, 0, scr_bytes), "memset");
            k::dec_moe_peer_math_test(LM, ptr, start, ng, tok, dst, xq, pscr, out2, s, g2);
            ck(cudaStreamSynchronize(s), "peer math");
            const auto o2 = down(out2, (size_t) cap * H);
            size_t diff = 0;
            for (int e = 0; e < ne; ++e)
                diff += std::memcmp(&o2[(size_t) ds[(size_t) e] * H], &om[(size_t) ds[(size_t) e] * H], H * 4) != 0;
            check(diff == 0, std::string("T ") + std::to_string(T) + ": peer math" + (g2 ? " (gate/up two rows a warp)" : "") +
                                 " rows = grouped rows (" + std::to_string(diff) + " differ)");
            cudaFree(out2);
        }
        // each entry alone in a one-entry group: the same rows
        {
            size_t diff = 0;
            std::vector<float> o1, g1, u1;
            for (int gi = 0; gi < ng; ++gi)
                for (int e = st[(size_t) gi]; e < st[(size_t) gi + 1]; ++e) {
                    const int32_t one_st[2] = {0, 1}, one_n = 1, one_tok = tk[(size_t) e], one_dst = 0;
                    const unsigned long long p = (unsigned long long) (blobs + slot * (size_t) (3 * gexp[(size_t) gi]));
                    ck(cudaMemcpy(ptr, &p, 8, cudaMemcpyHostToDevice), "ptr");
                    ck(cudaMemcpy(start, one_st, 8, cudaMemcpyHostToDevice), "start");
                    ck(cudaMemcpy(n, &one_n, 4, cudaMemcpyHostToDevice), "n");
                    ck(cudaMemcpy(tok, &one_tok, 4, cudaMemcpyHostToDevice), "tok");
                    ck(cudaMemcpy(dst, &one_dst, 4, cudaMemcpyHostToDevice), "dst");
                    k::native_expert_grouped(LM, ptr, start, n, dst, tok, cap, cap, xq, scr, out, s);
                    ck(cudaStreamSynchronize(s), "grouped");
                    const auto r = down(out, (size_t) H);
                    diff += std::memcmp(r.data(), &om[(size_t) ds[(size_t) e] * H], H * 4) != 0;
                }
            check(diff == 0, "T " + std::to_string(T) + ": every entry alone = in its group (" + std::to_string(diff) + " differ)");
        }
        // the double reference
        std::vector<double> ref((size_t) H);
        for (int gi = 0; gi < ng; ++gi) {
            const Expert& e = ex[(size_t) gexp[(size_t) gi]];
            for (int j = st[(size_t) gi]; j < st[(size_t) gi + 1]; ++j) {
                const float* xr = xs.data() + (size_t) tk[(size_t) j] * H;
                const size_t row = (size_t) ds[(size_t) j] * H;
                reference(e.wg, e.wu, e.wd, xr, ref.data());
                add_err(em, &om[row], ref.data());
                reference(e.wg, e.wg, e.wd, xr, ref.data());
                add_err(ea, &oa[row], ref.data());
                reference(e.wu, e.wu, e.wd, xr, ref.data());
                add_err(eb, &ob[row], ref.data());
            }
        }
    }
    std::printf("  vs the double reference (max abs / worst row's max rel): mixed %.3e / %.3e, all-IQ2_S %.3e / %.3e, "
                "all-IQ2_XXS %.3e / %.3e\n", em.max_abs, em.max_rel, ea.max_abs, ea.max_rel, eb.max_abs, eb.max_rel);
    check(em.max_rel <= 2.0 * std::max(ea.max_rel, eb.max_rel), "the mixed rows' error is that of the uniform layers");
    cudaFree(blobs); cudaFree(dx); cudaFree(xq); cudaFree(scr); cudaFree(pscr); cudaFree(out);
    cudaFree(ptr); cudaFree(start); cudaFree(n); cudaFree(dst); cudaFree(tok);
    cudaStreamDestroy(s);
}

// ------------------------------------------------------------------------------------------------ CPU
void cpu_part(const std::vector<Expert>& ex, std::mt19937& rng) {
    const bool avx512 = cpu::cpu_avx512_ok(), avx2 = cpu::cpu_avx2_ok();
    if (!avx2) {
        std::printf("  CPU: no AVX2 - skipped\n");
        return;
    }
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(TG, TD, H, FF, f, err, 0.f, TU)) {
        check(false, "native_fmt: " + err);
        return;
    }
    // the uniform row kernels of each half, as native_gu_rows picks them
    auto rows = [&](int t, const uint8_t* w, size_t rb, const void* const* a, int nt, float* const* o) {
        if (avx512) cpu::iq512_rows(t, w, rb, (int) H, a, nt, o, 0, (int) FF);
        else cpu::iq256_rows(t, w, rb, (int) H, a, nt, o, 0, (int) FF);
    };
    constexpr int MAXT = 8;
    const Expert& e = ex[0];
    const auto blob = cat({&e.g, &e.u, &e.d});
    std::vector<std::vector<uint8_t>> act(MAXT, std::vector<uint8_t>(cpu::kNativeActBytes));
    std::vector<std::vector<float>> xf(MAXT, std::vector<float>((size_t) H));
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int t = 0; t < MAXT; ++t) {
        for (auto& v : xf[(size_t) t]) v = nd(rng);
        cpu::native_quant_act(f, xf[(size_t) t].data(), act[(size_t) t].data());
    }
    size_t diff = 0, tok_diff = 0;
    std::vector<std::vector<float>> one(MAXT, std::vector<float>((size_t) FF));
    for (int t = 0; t < MAXT; ++t) {
        const void* a1[1] = {act[(size_t) t].data()};
        float* o1[1] = {one[(size_t) t].data()};
        cpu::native_gu_rows(f, blob.data(), a1, 1, o1, 0, (int) FF);
    }
    double max_rel = 0;
    for (int nt = 1; nt <= MAXT; ++nt) {
        const void* ap[MAXT];
        std::vector<std::vector<float>> ff(nt, std::vector<float>((size_t) FF)), g(nt, std::vector<float>((size_t) FF)),
            u(nt, std::vector<float>((size_t) FF));
        float *fp[MAXT], *gp[MAXT], *up[MAXT];
        for (int t = 0; t < nt; ++t) {
            ap[t] = act[(size_t) t].data();
            fp[t] = ff[(size_t) t].data();
            gp[t] = g[(size_t) t].data();
            up[t] = u[(size_t) t].data();
        }
        cpu::native_gu_rows(f, blob.data(), ap, nt, fp, 0, (int) FF);
        rows(TG, blob.data(), f.gu_row, ap, nt, gp);
        rows(TU, blob.data() + f.up_off, f.up_row, ap, nt, up);
        for (int t = 0; t < nt; ++t)
            for (int r = 0; r < FF; ++r) {
                const float gg = gp[t][r], uu = up[t][r], want = (gg / (1.f + std::exp(-gg))) * uu;
                diff += std::memcmp(&want, &fp[t][r], 4) != 0;
                tok_diff += std::memcmp(&fp[t][r], &one[(size_t) t][(size_t) r], 4) != 0;
            }
        if (nt == MAXT) {   // the double reference of the SwiGLU rows, against the activations as quantized
            std::vector<float> xq((size_t) H);
            for (int t = 0; t < nt; ++t) {
                // Q8_K blocks: fp32 d, 256 int8 codes, 16 int16 sums (ggml has no to_float for it)
                const uint8_t* q8 = act[(size_t) t].data();
                const size_t qb = 4 + 256 + 32;
                for (int64_t b = 0; b < H / 256; ++b) {
                    float d;
                    std::memcpy(&d, q8 + b * qb, 4);
                    for (int i = 0; i < 256; ++i) xq[(size_t) (b * 256 + i)] = d * (float) (int8_t) q8[b * qb + 4 + i];
                }
                double mr = 0, me = 0;
                for (int r = 0; r < FF; ++r) {
                    double gd = 0, ud = 0;
                    for (int64_t c = 0; c < H; ++c) {
                        gd += (double) e.wg[(size_t) (r * H + c)] * xq[(size_t) c];
                        ud += (double) e.wu[(size_t) (r * H + c)] * xq[(size_t) c];
                    }
                    const double ref = silu(gd) * ud;
                    me = std::max(me, std::fabs(ref - fp[t][r]));
                    mr = std::max(mr, std::fabs(ref));
                }
                if (mr > 0) max_rel = std::max(max_rel, me / mr);
            }
        }
    }
    std::printf("  CPU (%s): SwiGLU rows vs the double reference, worst row's max rel %.3e\n", avx512 ? "AVX-512" : "AVX2", max_rel);
    check(diff == 0, "CPU native_gu_rows = the uniform row kernels of each half (" + std::to_string(diff) + " differ)");
    check(tok_diff == 0, "CPU: a token's rows the same alone as among 1..8 (" + std::to_string(tok_diff) + " differ)");
    check(max_rel < 1e-4, "CPU: within float rounding of the reference");
}

}  // namespace

int main() {
    std::printf("mixed_gu_parity: IQ2_S gate + IQ2_XXS up / Q2_0 down, %lld x %lld\n", (long long) H, (long long) FF);
    std::mt19937 rng(1817);
    std::vector<Expert> ex((size_t) NB);
    for (auto& e : ex) {
        e.g = random_rows(TG, FF, H, rng);
        e.u = random_rows(TU, FF, H, rng);
        e.d = random_rows(TD, H, FF, rng);
        e.wg = deq(TG, e.g.data(), FF, H);
        e.wu = deq(TU, e.u.data(), FF, H);
        e.wd = deq(TD, e.d.data(), H, FF);
    }
    cpu_part(ex, rng);
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("  no CUDA device: the GPU part is skipped\n");
    } else {
        gpu_part(ex, rng);
    }
    std::printf(g_fail ? "mixed_gu_parity: %d FAILED\n" : "mixed_gu_parity: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
