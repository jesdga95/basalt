// tests/kernels/native_expert_parity.cpp - one native expert three ways.
//
//     build/native_expert_parity <shard.gguf> [layer ...]     real rows (a split model's shards are found by name)
//     build/native_expert_parity --synthetic GU/DOWN ...      random weights quantized by ggml (e.g. q4_K/q5_1)
//     build/native_expert_parity --q5_1-min                   the Q5_1 min term on crafted activations
//     build/native_expert_parity --synthetic nvfp4/nvfp4      NVFP4: the CPU tier and the GPU's generic kernels
//     build/native_expert_parity --experts FILE [layer ...]   NVFP4 experts of a pack's experts.bin or a .basalt
//
// (a) float reference: ggml's own dequantizer (`to_float`) and a float SwiGLU expert, (b) the CPU path
// (ggml-cpu vec_dot with its quantized activations), (c) the GPU path (`native_expert_grouped`, q8_1
// activations).  (b) and (c) each differ from (a) by their activation rounding only (a few 1e-3 relative).  (c) runs
// twice, with the kernels that decode a weight part once for all entries and with the per-entry ones: bitwise equal.
// (d) the GPU dequantizers of the prompt path and the embedding against `to_float` (Q8_0: bit for bit).
#include "basalt/artifact/gguf_reader.hpp"
#include "basalt/core/native_head.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/kernels/cpu/iq_avx512.hpp"
#include "basalt/kernels/cpu/iq_avx2.hpp"
#include "basalt/kernels/cpu/nvfp4_avx2.hpp"
#include "basalt/kernels/cpu/kq_avx2.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/iq_kernels.hpp"

#include "basalt/ggml/ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace cpu = basalt::kernels::cpu;

static double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) { n += std::fabs((double) a[i] - b[i]); d += std::fabs((double) b[i]); }
    return n / (d + 1e-30);
}


namespace {
constexpr int NT = 3, E = 7;
constexpr int64_t H = 2560, FF = 640;

// The three ways and the dequantizers for one expert blob; returns the number of failed checks.
int check_blob(const cpu::NativeFmt& f, const std::vector<uint8_t>& blob, int seed, const std::string& label,
               cudaStream_t s) {
    int failures = 0;
    // (a) the float reference
    const auto* tg = ggml_get_type_traits((ggml_type) f.gu_type);
    const auto* td = ggml_get_type_traits((ggml_type) f.d_type);
    std::vector<float> G((size_t) FF * H), U((size_t) FF * H), D((size_t) H * FF);
    for (int64_t r = 0; r < FF; ++r) {
        tg->to_float(blob.data() + r * f.gu_row, G.data() + r * H, H);
        tg->to_float(blob.data() + f.up_off + r * f.gu_row, U.data() + r * H, H);
    }
    for (int64_t r = 0; r < H; ++r) td->to_float(blob.data() + f.down_off + r * f.d_row, D.data() + r * FF, FF);
    std::mt19937 rng(11 + seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) NT * H);
    for (auto& v : x) v = nd(rng);
    std::vector<float> ref((size_t) NT * H), got_c((size_t) NT * H), got_g((size_t) NT * H);
    for (int k = 0; k < NT; ++k) {
        std::vector<float> h(FF);
        for (int64_t r = 0; r < FF; ++r) {
            double g = 0, u = 0;
            for (int64_t i = 0; i < H; ++i) { g += (double) G[r * H + i] * x[k * H + i]; u += (double) U[r * H + i] * x[k * H + i]; }
            h[r] = (float) (g / (1.0 + std::exp(-g)) * u);
        }
        for (int64_t r = 0; r < H; ++r) {
            double o = 0;
            for (int64_t i = 0; i < FF; ++i) o += (double) D[r * FF + i] * h[i];
            ref[k * H + r] = (float) o;
        }
    }
    // (b) the CPU
    {
        std::vector<std::vector<uint8_t>> act(NT, std::vector<uint8_t>(cpu::kNativeActBytes));
        std::vector<std::vector<uint8_t>> hq(NT, std::vector<uint8_t>(cpu::kNativeHBytes));
        std::vector<std::vector<float>> ff(NT, std::vector<float>(FF));
        const void* a[NT];
        float* ffp[NT];
        const void* hp[NT];
        float* op[NT];
        for (int k = 0; k < NT; ++k) {
            cpu::native_quant_act(f, x.data() + k * H, act[k].data());
            a[k] = act[k].data();
            ffp[k] = ff[k].data();
        }
        cpu::native_gu_rows(f, blob.data(), a, NT, ffp, 0, (int) FF);
        // either multi-token kernel: IQ4_XS has an AVX-2 one and no AVX-512 one, so the gate cannot be
        // iq512_supported alone - that would leave the format untested on every CPU.
        if (cpu::iq512_supported(f.gu_type) || cpu::iq256_supported(f.gu_type)) {
            // ggml's own vec_dot, same Q8_K activations: the reference for both multi-token kernels
            // (float-order differences only)
            const auto* tc = ggml_get_type_traits_cpu((ggml_type) f.gu_type);
            std::vector<float> gref((size_t) NT * FF);
            for (int k = 0; k < NT; ++k)
                for (int64_t r = 0; r < FF; ++r)
                    tc->vec_dot((int) H, &gref[k * FF + r], 0, blob.data() + r * f.gu_row, 0, a[k], 0, 1);
            double usg = 0.0;
            {   // ggml single-token throughput, the weights cache-resident (compute bound)
                const int it = 50;
                auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < it; ++i)
                    for (int64_t r = 0; r < FF; ++r) {
                        float gg, uu;
                        tc->vec_dot((int) H, &gg, 0, blob.data() + r * f.gu_row, 0, a[0], 0, 1);
                        tc->vec_dot((int) H, &uu, 0, blob.data() + f.up_off + r * f.gu_row, 0, a[0], 0, 1);
                    }
                auto t1 = std::chrono::steady_clock::now();
                usg = std::chrono::duration<double, std::micro>(t1 - t0).count() / it;
            }
            // one multi-token kernel: parity against ggml's vec_dot, then one-token and NT-token throughput
            auto check = [&](const char* tag, bool is512) {
                std::vector<float> g((size_t) NT * FF);
                float* gp[NT];
                for (int k = 0; k < NT; ++k) gp[k] = g.data() + k * FF;
                (is512 ? cpu::iq512_rows : cpu::iq256_rows)(f.gu_type, blob.data(), f.gu_row, (int) H, a, NT, gp, 0, (int) FF);
                const double rg = rel(g, gref);
                std::printf("          %s %s gate rows vs ggml vec_dot: rel %.2e\n",
                            ggml_type_name((ggml_type) f.gu_type), tag, rg);
                if (rg > 1e-5) {   // ggml's own vec_dot is the reference: above 1e-5 it is a real mismatch,
                    std::printf("          %s %s gate rows MISMATCH (rel %.2e > 1e-5)\n",
                                ggml_type_name((ggml_type) f.gu_type), tag, rg);   // not rounding
                    ++failures;
                }
                float* f1[1] = {ffp[0]};
                const int it = 50;
                const auto gu = is512 ? cpu::iq512_gu_rows : cpu::iq256_gu_rows;
                auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < it; ++i) gu(f.gu_type, blob.data(), f.gu_row, f.up_off, (int) H, a, 1, f1, 0, (int) FF);
                auto t1 = std::chrono::steady_clock::now();
                for (int i = 0; i < it; ++i) gu(f.gu_type, blob.data(), f.gu_row, f.up_off, (int) H, a, NT, ffp, 0, (int) FF);
                auto t2 = std::chrono::steady_clock::now();
                const double us1 = std::chrono::duration<double, std::micro>(t1 - t0).count() / it;
                const double usn = std::chrono::duration<double, std::micro>(t2 - t1).count() / it;
                std::printf("          gate+up one token one thread: %s %.0f us (%.2f GB/s), ggml %.0f us (%.2f GB/s)\n",
                            tag, us1, 2.0 * f.up_off / us1 / 1e3, usg, 2.0 * f.up_off / usg / 1e3);
                std::printf("          gate+up %d tokens one thread: %s %.0f us vs ggml %d x %.0f us\n", NT, tag, usn, NT, usg);
            };
            // guarded: the binary runs on AVX-2 CPUs too, and IQ4_XS has no AVX-512 kernel (an empty switch)
            if (cpu::cpu_avx512_ok() && cpu::iq512_supported(f.gu_type)) check("avx512", true);
            if (cpu::cpu_avx2_ok() && cpu::iq256_supported(f.gu_type)) check("avx2", false);   // no AVX2: ggml-cpu only
        }
        for (int k = 0; k < NT; ++k) {
            cpu::native_quant_h(f, ff[k].data(), hq[k].data());
            hp[k] = hq[k].data();
            op[k] = got_c.data() + k * H;
        }
        cpu::native_down_rows(f, blob.data(), hp, NT, op, 0, (int) H);
        {   // a token's rows must not depend on how many tokens share the call - each token alone (nt = 1)
            // against its row of the NT-token call above, bit for bit, gate/up and down
            size_t gu_diff = 0, dn_diff = 0;
            std::vector<float> one_ff(FF), one_out(H);
            // the NT-token gate/up rows again: the throughput loops above overwrote ff with other kernels' rows
            for (int k = 0; k < NT; ++k) ffp[k] = ff[k].data();
            cpu::native_gu_rows(f, blob.data(), a, NT, ffp, 0, (int) FF);
            for (int k = 0; k < NT; ++k) {
                const void* a1[1] = {a[k]};
                float* f1[1] = {one_ff.data()};
                cpu::native_gu_rows(f, blob.data(), a1, 1, f1, 0, (int) FF);
                for (int64_t r = 0; r < FF; ++r) gu_diff += std::memcmp(&one_ff[r], &ff[k][r], 4) != 0;
                const void* h1[1] = {hp[k]};
                float* o1[1] = {one_out.data()};
                cpu::native_down_rows(f, blob.data(), h1, 1, o1, 0, (int) H);
                for (int64_t r = 0; r < H; ++r) dn_diff += std::memcmp(&one_out[r], &got_c[k * H + r], 4) != 0;
            }
            std::printf("          width invariance (1 vs %d tokens): gate/up %zu, down %zu rows differ\n", NT,
                        gu_diff, dn_diff);
            if (gu_diff || dn_diff) ++failures;
        }
        // UD-Q4_K_XL's formats: the multi-token AVX2 kernels (kq_avx2.cpp) against ggml-cpu's vec_dot per token,
        // BIT FOR BIT, for 1..8 tokens; then the time of 4 tokens (a verify window) both ways
        for (int role = 0; role < 2; ++role) {
            const int type = role == 0 ? f.gu_type : f.d_type;
            if (!cpu::cpu_avx2_ok() || !cpu::kq256_supported(type) || (role == 0 && type != 12)) continue;
            const int n = role == 0 ? (int) H : (int) FF, rows = role == 0 ? (int) FF : (int) H;
            const size_t rb = role == 0 ? f.gu_row : f.d_row;
            const uint8_t* w = blob.data() + (role == 0 ? 0 : f.down_off);
            const auto* tc = ggml_get_type_traits_cpu((ggml_type) type);
            const ggml_type at = tc->vec_dot_type;
            std::vector<std::vector<uint8_t>> acts(8, std::vector<uint8_t>(ggml_row_size(at, n)));
            std::mt19937 arng(77 + seed);
            std::normal_distribution<float> and_(0.f, 1.f);
            std::vector<float> xs((size_t) n);
            const void* ap[8];
            for (int t = 0; t < 8; ++t) {
                for (auto& v : xs) v = and_(arng);
                ggml_get_type_traits_cpu(at)->from_float(xs.data(), acts[t].data(), n);
                ap[t] = acts[t].data();
            }
            std::vector<float> ref((size_t) 8 * rows), got((size_t) 8 * rows);
            for (int t = 0; t < 8; ++t)
                for (int r = 0; r < rows; ++r) tc->vec_dot(n, &ref[(size_t) t * rows + r], 0, w + (size_t) r * rb, 0, ap[t], 0, 1);
            size_t differ = 0;
            for (int nt = 1; nt <= 8; ++nt) {
                float* op[8];
                for (int t = 0; t < nt; ++t) op[t] = got.data() + (size_t) t * rows;
                cpu::kq256_rows(type, w, rb, n, ap, nt, op, 0, rows);
                for (int t = 0; t < nt; ++t)
                    differ += std::memcmp(op[t], ref.data() + (size_t) t * rows, (size_t) rows * 4) != 0;
            }
            const int it = 30;
            float* op4[4] = {got.data(), got.data() + rows, got.data() + 2 * rows, got.data() + 3 * rows};
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < it; ++i) cpu::kq256_rows(type, w, rb, n, ap, 4, op4, 0, rows);
            auto t1 = std::chrono::steady_clock::now();
            for (int i = 0; i < it; ++i)
                for (int r = 0; r < rows; ++r)
                    for (int t = 0; t < 4; ++t) tc->vec_dot(n, op4[t] + r, 0, w + (size_t) r * rb, 0, ap[t], 0, 1);
            auto t2 = std::chrono::steady_clock::now();
            const double us_k = std::chrono::duration<double, std::micro>(t1 - t0).count() / it;
            const double us_g = std::chrono::duration<double, std::micro>(t2 - t1).count() / it;
            std::printf("          %s %s rows, AVX2 multi-token vs ggml vec_dot: %zu of 36 token-sets differ in any bit; "
                        "4 tokens one thread %.0f us vs ggml %.0f us (%.2fx)\n", ggml_type_name((ggml_type) type),
                        role == 0 ? "gate" : "down", differ, us_k, us_g, us_g / us_k);
            if (differ) ++failures;
        }
        if (cpu::q2_native_kernels(f.d_type)) {
            // (b2) the GGUF-layout Q2_0 kernel the pool uses for Q2_0 down projections - the AVX-512 one
            // where the CPU has it, the AVX-2 one (q2_avx2.cpp) where it does not.  Calling the AVX-512
            // kernel unconditionally faults on a Zen 2/3 CPU.
            std::vector<cpu::ActQ> a2(NT);
            const cpu::ActQ* ap[NT];
            std::vector<float> alt((size_t) NT * H);
            float* altp[NT];
            for (int k = 0; k < NT; ++k) {
                cpu::act_quant_any(ff[k].data(), (int) FF, a2[k]);   // gated: AVX-512 or AVX-2 per CPU
                ap[k] = &a2[k];
                altp[k] = alt.data() + k * H;
            }
            cpu::q2_rows_any(blob.data() + f.down_off, f.d_row, (int) (FF / 64), ap, NT, altp, 0, (int) H);
            std::printf("          q2_0 %s down vs ggml down: rel %.2e\n",
                        cpu::cpu_avx512_ok() ? "AVX-512" : "AVX-2", rel(alt, got_c));
        }
        if (f.d_type == 20 && cpu::cpu_avx2_ok()) {
            // (b3) the IQ4_NL multi-token AVX-2 kernel the pool now uses for IQ4_NL down projections,
            // against ggml-cpu's single-token vec_dot on the SAME Q8_0 activations (h), plus timing.
            std::vector<float> alt((size_t) NT * H), refd((size_t) NT * H);
            float* altp[NT];
            for (int k = 0; k < NT; ++k) altp[k] = alt.data() + k * H;
            cpu::iq4nl256_down_rows(blob.data() + f.down_off, f.d_row, (int) FF, hp, NT, altp, 0, (int) H);
            const auto* tdc = ggml_get_type_traits_cpu((ggml_type) f.d_type);
            for (int k = 0; k < NT; ++k)
                for (int64_t r = 0; r < H; ++r)
                    tdc->vec_dot((int) FF, refd.data() + k * H + r, 0,
                                 blob.data() + f.down_off + (size_t) r * f.d_row, 0, hp[k], 0, 1);
            const double rdn = rel(alt, refd);
            std::printf("          iq4_nl AVX-2 multi-token down vs ggml down: rel %.2e\n", rdn);
            if (rdn > 1e-5) {   // same reasoning as the gate/up rows above
                std::printf("          iq4_nl down MISMATCH (rel %.2e > 1e-5)\n", rdn);
                ++failures;
            }
            const int it = 50;
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < it; ++i)
                cpu::iq4nl256_down_rows(blob.data() + f.down_off, f.d_row, (int) FF, hp, NT, altp, 0, (int) H);
            const auto t1 = std::chrono::steady_clock::now();
            for (int i = 0; i < it; ++i)
                for (int k = 0; k < NT; ++k)
                    for (int64_t r = 0; r < H; ++r)
                        tdc->vec_dot((int) FF, refd.data() + k * H + r, 0,
                                    blob.data() + f.down_off + (size_t) r * f.d_row, 0, hp[k], 0, 1);
            const auto t2 = std::chrono::steady_clock::now();
            const double usn = std::chrono::duration<double, std::micro>(t1 - t0).count() / it;
            const double usg = std::chrono::duration<double, std::micro>(t2 - t1).count() / it;
            std::printf("          down %d tokens one thread: avx2 %.0f us (%.2f GB/s) vs ggml %.0f us (%.2f GB/s)\n",
                        NT, usn, (double) (f.bytes - f.down_off) / usn / 1e3, usg,
                        (double) (f.bytes - f.down_off) / usg / 1e3);
        }
    }
    // (c) the GPU: one group holding the NT entries
    {
        const auto L = basalt::kernels::native_expert_layout(f.gu_type, f.d_type, H, FF);
        void *dblob, *dx, *dxq, *dscr;
        float* dout;
        unsigned long long* dptr;
        int32_t *dstart, *dn, *ddst, *dtok;
        cudaMalloc(&dblob, blob.size());
        cudaMalloc(&dx, x.size() * 4);
        cudaMalloc(&dxq, (size_t) NT * H / 32 * 36);
        cudaMalloc(&dscr, basalt::kernels::native_expert_scratch_bytes(NT, FF));
        cudaMalloc((void**) &dout, (size_t) NT * H * 4);
        cudaMalloc((void**) &dptr, 8);
        cudaMalloc((void**) &dstart, 8);
        cudaMalloc((void**) &dn, 4);
        cudaMalloc((void**) &ddst, NT * 4);
        cudaMalloc((void**) &dtok, NT * 4);
        cudaMemcpy(dblob, blob.data(), blob.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        const unsigned long long p = (unsigned long long) dblob;
        const int32_t st[2] = {0, NT}, one = 1, idx[NT] = {0, 1, 2};
        cudaMemcpy(dptr, &p, 8, cudaMemcpyHostToDevice);
        cudaMemcpy(dstart, st, 8, cudaMemcpyHostToDevice);
        cudaMemcpy(dn, &one, 4, cudaMemcpyHostToDevice);
        cudaMemcpy(ddst, idx, NT * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dtok, idx, NT * 4, cudaMemcpyHostToDevice);
        basalt::kernels::quantize_q8_1_rows((const float*) dx, NT, H, dxq, s);
        basalt::kernels::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, 1, NT, dxq, dscr, dout, s);
        cudaStreamSynchronize(s);
        cudaMemcpy(got_g.data(), dout, got_g.size() * 4, cudaMemcpyDeviceToHost);
        {   // the kernels that decode a weight part once for the NT entries, bitwise vs the per-entry ones (
            // the formats without a decode-once split take the per-entry kernels either way)
            std::vector<float> old_g(got_g.size());
            const bool was = basalt::kernels::iq_old_kernels();
            basalt::kernels::iq_set_old_kernels(!was);
            basalt::kernels::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, 1, NT, dxq, dscr, dout, s);
            basalt::kernels::iq_set_old_kernels(was);
            cudaStreamSynchronize(s);
            cudaMemcpy(old_g.data(), dout, old_g.size() * 4, cudaMemcpyDeviceToHost);
            const bool same = std::memcmp(old_g.data(), got_g.data(), got_g.size() * 4) == 0;
            std::printf("          gpu decode-once vs per-entry kernels: %s\n", same ? "bitwise equal" : "MISMATCH");
            if (!same) ++failures;
        }
        {   // (d) the GPU dequantizers (prompt path: iq_dequant_f32/_f16; embedding: iq_embed_rows) against to_float
            float* dq = nullptr;
            uint16_t* dh = nullptr;
            int32_t* dtk = nullptr;
            cudaMalloc((void**) &dq, (size_t) FF * H * 4);
            cudaMalloc((void**) &dh, (size_t) FF * H * 2);
            cudaMalloc((void**) &dtk, (size_t) FF * 4);
            std::vector<float> got_dq((size_t) FF * H);
            std::vector<uint16_t> got_h((size_t) FF * H);
            std::vector<int32_t> tk((size_t) FF);
            for (int64_t r = 0; r < FF; ++r) tk[(size_t) r] = (int32_t) (FF - 1 - r);   // rows in reverse order
            cudaMemcpy(dtk, tk.data(), tk.size() * 4, cudaMemcpyHostToDevice);
            double dq_err = 0.0;
            size_t differ = 0, h_differ = 0;
            const struct { const uint8_t* src; int type; const std::vector<float>* want; } mats[3] = {
                {(const uint8_t*) dblob, f.gu_type, &G}, {(const uint8_t*) dblob + f.up_off, f.gu_type, &U},
                {(const uint8_t*) dblob + f.down_off, f.d_type, &D}};
            for (const auto& m : mats) {
                basalt::kernels::iq_dequant_f32(m.type, m.src, FF * H, dq, s);
                basalt::kernels::iq_dequant_f16(m.type, m.src, FF * H, dh, s);
                cudaStreamSynchronize(s);
                cudaMemcpy(got_dq.data(), dq, got_dq.size() * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(got_h.data(), dh, got_h.size() * 2, cudaMemcpyDeviceToHost);
                dq_err = (std::max)(dq_err, rel(got_dq, *m.want));
                for (size_t i = 0; i < got_dq.size(); ++i) {
                    differ += std::memcmp(&got_dq[i], &(*m.want)[i], 4) != 0;
                    h_differ += got_h[i] != ggml_fp32_to_fp16(got_dq[i]);
                }
            }
            // the gate matrix as an embedding table of FF rows of H values
            basalt::kernels::iq_embed_rows(f.gu_type, dblob, f.gu_row, dtk, FF, H, dq, s);
            cudaStreamSynchronize(s);
            cudaMemcpy(got_dq.data(), dq, got_dq.size() * 4, cudaMemcpyDeviceToHost);
            size_t emb_differ = 0;
            for (int64_t r = 0; r < FF; ++r)
                emb_differ += std::memcmp(got_dq.data() + r * H, G.data() + (FF - 1 - r) * H, (size_t) H * 4) != 0;
            cudaFree(dq); cudaFree(dh); cudaFree(dtk);
            std::printf("          GPU dequant vs ggml to_float: rel %.2e, %zu of %lld values differ in any bit; fp16 of "
                        "it: %zu differ; embedding rows: %zu of %lld differ\n", dq_err, differ, (long long) (3 * FF * H),
                        h_differ, emb_differ, (long long) FF);
            // exact where there is one product (Q8_0), a rounding at most elsewhere (fused multiply-adds)
            const bool exact = f.gu_type == 8 && f.d_type == 8;
            if (!(dq_err < 1e-6) || h_differ || (exact && (differ || emb_differ))) ++failures;
        }
        cudaFree(dblob); cudaFree(dx); cudaFree(dxq); cudaFree(dscr); cudaFree(dout); cudaFree(dptr);
        cudaFree(dstart); cudaFree(dn); cudaFree(ddst); cudaFree(dtok);
    }
    const double ec = rel(got_c, ref), eg = rel(got_g, ref), ecg = rel(got_c, got_g);
    const bool ok = ec < 3e-2 && eg < 3e-2 && std::isfinite(ec) && std::isfinite(eg);
    std::printf("%-9s %-8s/%-7s blob %8zu  cpu rel %.2e  gpu rel %.2e  cpu-gpu %.2e  %s\n", label.c_str(),
                ggml_type_name((ggml_type) f.gu_type), ggml_type_name((ggml_type) f.d_type), f.bytes, ec, eg, ecg,
                ok ? "ok" : "FAIL");
    return failures + (ok ? 0 : 1);
}

int type_of(std::string name) {
    for (auto& c : name) c = (char) std::tolower((unsigned char) c);
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char* n = ggml_type_name((ggml_type) t);
        if (n == nullptr) continue;
        std::string m(n);
        for (auto& c : m) c = (char) std::tolower((unsigned char) c);
        if (m == name) return t;
    }
    return -1;
}

// Random weights (rows of different magnitudes) quantized by ggml into the blob layout.
std::vector<uint8_t> synthetic_blob(const cpu::NativeFmt& f, int seed) {
    std::vector<uint8_t> blob(f.bytes);
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto quant = [&](int type, int64_t rows, int64_t cols, uint8_t* dst) {
        std::vector<float> w((size_t) (rows * cols));
        for (int64_t r = 0; r < rows; ++r) {
            const float scale = 0.02f * (0.5f + (float) (r % 7) / 7.0f);
            for (int64_t c = 0; c < cols; ++c) w[(size_t) (r * cols + c)] = scale * nd(rng);
        }
        ggml_quantize_chunk((ggml_type) type, w.data(), dst, 0, rows, cols);
    };
    quant(f.gu_type, FF, H, blob.data());
    quant(f.gu_type, FF, H, blob.data() + f.up_off);
    quant(f.d_type, H, FF, blob.data() + f.down_off);
    return blob;
}

// NVFP4 random weights with the pack's magnitudes: block values in the thousands, FP32 per-expert scales in the
// blob's tail bring them back ({s_gate, s_up, s_down, 0}, s_gate == s_up).
std::vector<uint8_t> synthetic_nvfp4_blob(const cpu::NativeFmt& f, int seed) {
    std::vector<uint8_t> blob(f.bytes);
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    const float sgu = 4e-5f * (1.f + 0.1f * (float) (seed % 5)), sd = 9e-5f;
    auto quant = [&](float scale, int64_t rows, int64_t cols, uint8_t* dst) {
        std::vector<float> w((size_t) (rows * cols));
        for (int64_t r = 0; r < rows; ++r) {
            const float rs = 0.02f * (0.5f + (float) (r % 7) / 7.0f) / scale;
            for (int64_t c = 0; c < cols; ++c) w[(size_t) (r * cols + c)] = rs * nd(rng);
        }
        ggml_quantize_chunk(GGML_TYPE_NVFP4, w.data(), dst, 0, rows, cols);
    };
    quant(sgu, FF, H, blob.data());
    quant(sgu, FF, H, blob.data() + f.up_off);
    quant(sd, H, FF, blob.data() + f.down_off);
    const float tail[4] = {sgu, sgu, sd, 0.f};
    std::memcpy(blob.data() + f.bytes - f.tail, tail, sizeof tail);
    return blob;
}

// NVFP4 on the CPU tier: (1) the AVX2 and AVX-512 vec_dot against the plain-C one on the expert's own rows and Q8_0
// activations (the same integer sums: float order only); (2) the CPU expert (native_gu_rows / native_down_rows, the
// tail's scales applied) against a double-precision expert on ggml's to_float weights with s_gate/s_up before SwiGLU
// and s_down on the output; (3) a single-thread microbench line; (4) the GPU (gpu_nvfp4).
int gpu_nvfp4(const cpu::NativeFmt& f, const std::vector<uint8_t>& blob, const std::vector<float>& x,
              const std::vector<float>& G, const std::vector<float>& U, const std::vector<float>& D, const float* sc,
              const std::vector<float>& ref, const std::vector<float>& got_c, cudaStream_t s);

// NVFP4 multi-token AVX-512 kernel against the single-token ggml dot this CPU would otherwise run (gate/up with s_gate,
// s_up and SwiGLU, down with s_down): bitwise equal for every token count 1..9 (9 crosses the 8-token pass) and for
// a row range that splits a down row pair; then a single-thread microbench, ggml dot vs the kernel, 1/2/4/8 tokens.
int check_nvfp4_multi(const cpu::NativeFmt& f, const std::vector<uint8_t>& blob, int seed, bool avx2) {
    const char* isa = avx2 ? "avx2" : "avx512";
    if (!avx2 && !cpu::cpu_avx512_ok()) { std::printf("          nvfp4 multi-token avx512: no AVX-512 on this CPU\n"); return 0; }
    if (avx2 && !cpu::cpu_avx2_ok()) { std::printf("          nvfp4 multi-token avx2: no AVX2 on this CPU\n"); return 0; }
    constexpr int MT = 9;
    float sc[4];
    std::memcpy(sc, blob.data() + f.bytes - f.tail, sizeof sc);
    const ggml_vec_dot_t dot = (avx2 ? ggml_get_type_traits_cpu_avx2(GGML_TYPE_NVFP4)
                                      : ggml_get_type_traits_cpu_avx512(GGML_TYPE_NVFP4))->vec_dot;
    std::mt19937 rng(23 + seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<std::vector<uint8_t>> act(MT, std::vector<uint8_t>(f.act_bytes)), hq(MT, std::vector<uint8_t>(f.h_bytes));
    std::vector<float> x(H), h(FF);
    const void* a[MT];
    const void* hp[MT];
    const void* const* a_ = a;
    const void* const* hp_ = hp;
    // the kernel under test: the AVX2 one is called directly, so an AVX-512 CPU checks it too
    auto new_gu = [&](int nt, float* const* ff, int r0, int r1) {
        if (avx2) cpu::nvfp4_256_gu_rows(blob.data(), f.gu_row, f.up_off, (int) H, sc[0], sc[1], a_, nt, ff, r0, r1);
        else cpu::native_gu_rows(f, blob.data(), a_, nt, ff, r0, r1);
    };
    auto new_down = [&](int nt, float* const* out, int r0, int r1) {
        if (avx2) cpu::nvfp4_256_down_rows(blob.data() + f.down_off, f.d_row, (int) FF, sc[2], hp_, nt, out, r0, r1);
        else cpu::native_down_rows(f, blob.data(), hp_, nt, out, r0, r1);
    };
    for (int t = 0; t < MT; ++t) {
        for (auto& v : x) v = nd(rng) * (t == 4 ? 30.f : 1.f);
        for (auto& v : h) v = 0.05f * nd(rng);
        cpu::native_quant_act(f, x.data(), act[t].data());
        cpu::native_quant_h(f, h.data(), hq[t].data());
        a[t] = act[t].data();
        hp[t] = hq[t].data();
    }
    // the old path: one ggml dot per row and token
    auto old_gu = [&](int nt, float* const* ff, int r0, int r1) {
        for (int r = r0; r < r1; ++r)
            for (int t = 0; t < nt; ++t) {
                float g = 0.f, u = 0.f;
                dot((int) H, &g, 0, blob.data() + r * f.gu_row, 0, a[t], 0, 1);
                dot((int) H, &u, 0, blob.data() + f.up_off + r * f.gu_row, 0, a[t], 0, 1);
                g *= sc[0];
                u *= sc[1];
                ff[t][r] = (g / (1.f + std::exp(-g))) * u;
            }
    };
    auto old_down = [&](int nt, float* const* out, int r0, int r1) {
        for (int r = r0; r < r1; ++r)
            for (int t = 0; t < nt; ++t) {
                float v = 0.f;
                dot((int) FF, &v, 0, blob.data() + f.down_off + r * f.d_row, 0, hp[t], 0, 1);
                out[t][r] = v * sc[2];
            }
    };
    std::vector<std::vector<float>> f0(MT, std::vector<float>(FF)), f1(MT, std::vector<float>(FF));
    std::vector<std::vector<float>> o0(MT, std::vector<float>(H)), o1(MT, std::vector<float>(H));
    float* pf0[MT]; float* pf1[MT]; float* po0[MT]; float* po1[MT];
    for (int t = 0; t < MT; ++t) { pf0[t] = f0[t].data(); pf1[t] = f1[t].data(); po0[t] = o0[t].data(); po1[t] = o1[t].data(); }
    size_t diff = 0, total = 0;
    old_gu(MT, pf0, 0, (int) FF);
    old_down(MT, po0, 0, (int) H);
    for (int nt = 1; nt <= MT; ++nt) {
        for (int t = 0; t < MT; ++t) { std::fill(f1[t].begin(), f1[t].end(), -1.f); std::fill(o1[t].begin(), o1[t].end(), -1.f); }
        new_gu(nt, pf1, 0, 7);   // two calls: a split row range
        new_gu(nt, pf1, 7, (int) FF);
        new_down(nt, po1, 0, 5);
        new_down(nt, po1, 5, (int) H);
        for (int t = 0; t < nt; ++t) {
            diff += std::memcmp(f0[t].data(), f1[t].data(), FF * 4) != 0;
            diff += std::memcmp(o0[t].data(), o1[t].data(), H * 4) != 0;
            total += 2;
        }
    }
    std::printf("          nvfp4 %s multi-token vs ggml dot: %zu token-rowsets, %zu differ  %s\n", isa, total, diff,
                diff == 0 ? "bitwise ok" : "FAIL");
    // microbench: one thread, the blob cache-resident across passes
    const int it = 30;
    std::string line = std::string("          bench: nvfp4 ") + isa + " expert 1 thread, us per expert (ggml dot / multi-token):";
    for (int nt : {1, 2, 4, 8}) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < it; ++i) { old_gu(nt, pf0, 0, (int) FF); old_down(nt, po0, 0, (int) H); }
        const double us0 = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / it;
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < it; ++i) {
            new_gu(nt, pf1, 0, (int) FF);
            new_down(nt, po1, 0, (int) H);
        }
        const double us1 = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / it;
        char b[128];
        std::snprintf(b, sizeof b, "  %dt %.0f/%.0f (x%.2f, %.1f GB/s)", nt, us0, us1, us0 / us1,
                      (double) f.bytes / us1 * 1e-3);
        line += b;
    }
    std::printf("%s\n", line.c_str());
    return diff == 0 ? 0 : 1;
}

int check_nvfp4_cpu(const cpu::NativeFmt& f, const std::vector<uint8_t>& blob, int seed, const std::string& label,
                    cudaStream_t s) {
    int failures = 0;
    float sc[4];
    std::memcpy(sc, blob.data() + f.bytes - f.tail, sizeof sc);
    std::mt19937 rng(11 + seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) NT * H);
    for (auto& v : x) v = nd(rng);
    // (1) the dots
    {
        const ggml_type_traits_cpu* isa[2] = {ggml_get_type_traits_cpu_avx2(GGML_TYPE_NVFP4),
                                              cpu::cpu_avx512_ok() ? ggml_get_type_traits_cpu_avx512(GGML_TYPE_NVFP4)
                                                                   : nullptr};
        const char* name[2] = {"avx2", "avx512"};
        std::vector<uint8_t> xq(ggml_row_size(GGML_TYPE_Q8_0, H)), hq(ggml_row_size(GGML_TYPE_Q8_0, FF));
        std::vector<float> h(FF);
        for (auto& v : h) v = 0.05f * nd(rng);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, x.data(), xq.data(), 0, 1, H);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, h.data(), hq.data(), 0, 1, FF);
        for (int i = 0; i < 2; ++i) {
            if (isa[i] == nullptr) { std::printf("          nvfp4 %s dot: not on this CPU\n", name[i]); continue; }
            std::vector<float> want, got;
            for (int64_t r = 0; r < FF; ++r) {
                float a = 0.f, b = 0.f;
                ggml_vec_dot_nvfp4_q8_0_ref((int) H, &a, blob.data() + r * f.gu_row, xq.data());
                isa[i]->vec_dot((int) H, &b, 0, blob.data() + r * f.gu_row, 0, xq.data(), 0, 1);
                want.push_back(a); got.push_back(b);
            }
            for (int64_t r = 0; r < H; ++r) {
                float a = 0.f, b = 0.f;
                ggml_vec_dot_nvfp4_q8_0_ref((int) FF, &a, blob.data() + f.down_off + r * f.d_row, hq.data());
                isa[i]->vec_dot((int) FF, &b, 0, blob.data() + f.down_off + r * f.d_row, 0, hq.data(), 0, 1);
                want.push_back(a); got.push_back(b);
            }
            const double e = rel(got, want);
            std::printf("          nvfp4 %-6s dot vs plain C: %zu rows, rel %.2e  %s\n", name[i], want.size(), e,
                        e < 1e-6 ? "ok" : "FAIL");
            if (!(e < 1e-6)) ++failures;
        }
    }
    // (2) the expert
    const auto* tt = ggml_get_type_traits(GGML_TYPE_NVFP4);
    std::vector<float> G((size_t) FF * H), U((size_t) FF * H), D((size_t) H * FF);
    for (int64_t r = 0; r < FF; ++r) {
        tt->to_float(blob.data() + r * f.gu_row, G.data() + r * H, H);
        tt->to_float(blob.data() + f.up_off + r * f.gu_row, U.data() + r * H, H);
    }
    for (int64_t r = 0; r < H; ++r) tt->to_float(blob.data() + f.down_off + r * f.d_row, D.data() + r * FF, FF);
    std::vector<float> ref((size_t) NT * H), got((size_t) NT * H);
    for (int k = 0; k < NT; ++k) {
        std::vector<double> h(FF);
        for (int64_t r = 0; r < FF; ++r) {
            double g = 0, u = 0;
            for (int64_t i = 0; i < H; ++i) { g += (double) G[r * H + i] * x[k * H + i]; u += (double) U[r * H + i] * x[k * H + i]; }
            g *= sc[0];
            u *= sc[1];
            h[r] = g / (1.0 + std::exp(-g)) * u;
        }
        for (int64_t r = 0; r < H; ++r) {
            double o = 0;
            for (int64_t i = 0; i < FF; ++i) o += (double) D[r * FF + i] * h[i];
            ref[k * H + r] = (float) (o * sc[2]);
        }
    }
    std::vector<std::vector<uint8_t>> act(NT, std::vector<uint8_t>(cpu::kNativeActBytes));
    std::vector<std::vector<uint8_t>> hqv(NT, std::vector<uint8_t>(cpu::kNativeHBytes));
    std::vector<std::vector<float>> ff(NT, std::vector<float>(FF));
    const void* a[NT];
    const void* hp[NT];
    float* ffp[NT];
    float* op[NT];
    for (int k = 0; k < NT; ++k) {
        cpu::native_quant_act(f, x.data() + k * H, act[k].data());
        a[k] = act[k].data();
        ffp[k] = ff[k].data();
        op[k] = got.data() + k * H;
        hp[k] = hqv[k].data();
    }
    cpu::native_gu_rows(f, blob.data(), a, NT, ffp, 0, (int) FF);
    for (int k = 0; k < NT; ++k) cpu::native_quant_h(f, ff[k].data(), hqv[k].data());
    cpu::native_down_rows(f, blob.data(), hp, NT, op, 0, (int) H);
    // (3) one token, one thread: gate/up rows and down rows (the blob stays cache-resident across passes)
    const int it = 20;
    float* f1[1] = {ffp[0]};
    float* o1[1] = {op[0]};
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < it; ++i) {
        cpu::native_gu_rows(f, blob.data(), a, 1, f1, 0, (int) FF);
        cpu::native_down_rows(f, blob.data(), hp, 1, o1, 0, (int) H);
    }
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / it;
    for (int k = 0; k < NT; ++k) {   // the timing loop overwrote token 0
        cpu::native_gu_rows(f, blob.data(), a, 1, ffp + k, 0, (int) FF);
        cpu::native_down_rows(f, blob.data(), hp + k, 1, op + k, 0, (int) H);
    }
    // the same double expert on the CPU's own Q8_0 activations (x and h as the CPU rounded them): what is left is
    // float order, so the scales sit where the reference puts them
    std::vector<float> ref_q((size_t) NT * H);
    {
        const auto* t8 = ggml_get_type_traits(GGML_TYPE_Q8_0);
        std::vector<float> xd(H), hd(FF);
        for (int k = 0; k < NT; ++k) {
            t8->to_float(act[k].data(), xd.data(), H);
            t8->to_float(hqv[k].data(), hd.data(), FF);
            for (int64_t r = 0; r < H; ++r) {
                double o = 0;
                for (int64_t i = 0; i < FF; ++i) o += (double) D[r * FF + i] * hd[i];
                ref_q[k * H + r] = (float) (o * sc[2]);
            }
        }
    }
    const double e = rel(got, ref), eq = rel(got, ref_q);
    const bool ok = e < 3e-2 && std::isfinite(e) && eq < 1e-5;
    std::printf("%-9s nvfp4/nvfp4 blob %8zu  scales g %.3g u %.3g d %.3g  cpu rel %.2e (vs its own q8_0 h: %.2e)  %s\n",
                label.c_str(), f.bytes, sc[0], sc[1], sc[2], e, eq, ok ? "ok" : "FAIL");
    std::printf("          bench: cpu expert 1 token 1 thread %.0f us (%.2f GB/s of weights)\n", us,
                (double) f.bytes / us * 1e-3);
    failures += check_nvfp4_multi(f, blob, seed, false);
    failures += check_nvfp4_multi(f, blob, seed, true);
    failures += gpu_nvfp4(f, blob, x, G, U, D, sc, ref, got, s);
    return failures + (ok ? 0 : 1);
}

// NVFP4 on the GPU's generic grouped kernels (native_expert_grouped: Fmt<40> dp4a dot, the tail's scales applied in
// the gate/up and down kernels).  Checks, on one group of NT entries:
//  - gate, up (scaled, read back from the scratch) and the output against a double-precision expert on the GPU's
//    OWN q8_1 activations (x and h as the kernels rounded them, from the scratch): the same integer sums, so only
//    float order and the fp32 products are left -> rel < 1e-5 (measured ~1e-7);
//  - the output against the double expert on the unrounded x (rel < 3e-2, as every format: the q8_1 rounding of x
//    and h) and against the CPU tier (printed; the two round their activations differently, q8_0 vs q8_1);
//  - the dequantizers against ggml's to_float: fp32 bit for bit (E2M1 x UE4M3 is exact in fp32), and the scaled fp16
//    gate/up/down of the prompt fallback equal to fp16(to_float * s) in every value;
//  - a microbench line: one token, the blob L2-resident.
int gpu_nvfp4(const cpu::NativeFmt& f, const std::vector<uint8_t>& blob, const std::vector<float>& x,
              const std::vector<float>& G, const std::vector<float>& U, const std::vector<float>& D, const float* sc,
              const std::vector<float>& ref, const std::vector<float>& got_c, cudaStream_t s) {
    namespace K = basalt::kernels;
    int failures = 0;
    if (!K::native_expert_supported(f.gu_type, f.d_type, H, FF)) {
        std::printf("          gpu: NVFP4 is not a pair the GPU expert kernels take  FAIL\n");
        return 1;
    }
    const auto L = K::native_expert_layout(f.gu_type, f.d_type, H, FF);
    if (L.bytes != f.bytes || L.up_off != f.up_off || L.down_off != f.down_off) {
        std::printf("          gpu layout %zu B differs from the CPU's %zu B  FAIL\n", L.bytes, f.bytes);
        return 1;
    }
    const size_t scr_bytes = K::native_expert_scratch_bytes(NT, FF);
    uint8_t *dblob, *dxq, *dscr;
    float *dx, *dout;
    unsigned long long* dptr;
    int32_t *dstart, *dn, *didx;
    cudaMalloc((void**) &dblob, blob.size());
    cudaMalloc((void**) &dx, x.size() * 4);
    cudaMalloc((void**) &dxq, (size_t) NT * H / 32 * 36);
    cudaMalloc((void**) &dscr, scr_bytes);
    cudaMalloc((void**) &dout, (size_t) NT * H * 4);
    cudaMalloc((void**) &dptr, 8);
    cudaMalloc((void**) &dstart, 8);
    cudaMalloc((void**) &dn, 4);
    cudaMalloc((void**) &didx, NT * 4);
    cudaMemcpy(dblob, blob.data(), blob.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    const unsigned long long p = (unsigned long long) dblob;
    const int32_t st[2] = {0, NT}, one = 1, idx[NT] = {0, 1, 2};
    cudaMemcpy(dptr, &p, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dstart, st, 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dn, &one, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(didx, idx, NT * 4, cudaMemcpyHostToDevice);
    cudaMemset(dscr, 0, scr_bytes);
    K::quantize_q8_1_rows(dx, NT, H, dxq, s);
    K::native_expert_grouped(L, dptr, dstart, dn, didx, didx, 1, NT, dxq, dscr, dout, s);
    cudaStreamSynchronize(s);
    std::vector<float> got((size_t) NT * H), gate((size_t) NT * FF), up((size_t) NT * FF);
    std::vector<uint8_t> xq((size_t) NT * H / 32 * 36), hq((size_t) NT * FF / 32 * 36);
    // the scratch: gate, up, h (fp32, cap x n_ff each, 256-aligned), then h as q8_1
    const size_t fa = (((size_t) NT * FF * 4) + 255) & ~(size_t) 255;
    cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(gate.data(), dscr, gate.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(up.data(), dscr + fa, up.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hq.data(), dscr + 3 * fa, hq.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(xq.data(), dxq, xq.size(), cudaMemcpyDeviceToHost);
    const cudaError_t ce = cudaGetLastError();
    // q8_1 blocks: {half d, half s} then 32 int8
    auto deq8_1 = [](const uint8_t* b, int64_t n, std::vector<double>& o) {
        o.resize((size_t) n);
        for (int64_t i = 0; i < n; ++i) {
            const uint8_t* blk = b + (i / 32) * 36;
            uint16_t dh;
            std::memcpy(&dh, blk, 2);
            o[(size_t) i] = (double) ggml_fp16_to_fp32(dh) * (double) (int8_t) blk[4 + i % 32];
        }
    };
    std::vector<float> rg((size_t) NT * FF), ru((size_t) NT * FF), ro((size_t) NT * H);
    for (int k = 0; k < NT; ++k) {
        std::vector<double> xd, hd;
        deq8_1(xq.data() + (size_t) k * H / 32 * 36, H, xd);
        deq8_1(hq.data() + (size_t) k * FF / 32 * 36, FF, hd);
        for (int64_t r = 0; r < FF; ++r) {
            double g = 0, u = 0;
            for (int64_t i = 0; i < H; ++i) { g += (double) G[r * H + i] * xd[i]; u += (double) U[r * H + i] * xd[i]; }
            rg[k * FF + r] = (float) (g * sc[0]);
            ru[k * FF + r] = (float) (u * sc[1]);
        }
        for (int64_t r = 0; r < H; ++r) {
            double o = 0;
            for (int64_t i = 0; i < FF; ++i) o += (double) D[r * FF + i] * hd[i];
            ro[k * H + r] = (float) (o * sc[2]);
        }
    }
    const double eg = rel(gate, rg), eu = rel(up, ru), eo = rel(got, ro), e = rel(got, ref), ecg = rel(got, got_c);
    const bool ok = ce == cudaSuccess && eg < 1e-5 && eu < 1e-5 && eo < 1e-5 && e < 3e-2 && std::isfinite(e);
    std::printf("          gpu: gate %.2e up %.2e down %.2e vs double on its own q8_1 x/h; vs double ref %.2e; "
                "cpu-gpu %.2e  %s\n", eg, eu, eo, e, ecg, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
    {   // the dequantizers: fp32 unscaled vs to_float bitwise; fp16 scaled (the prompt fallback) vs fp16(to_float * s)
        float* dq = nullptr;
        uint16_t* dh = nullptr;
        cudaMalloc((void**) &dq, (size_t) FF * H * 4);
        cudaMalloc((void**) &dh, (size_t) 2 * FF * H * 2);
        std::vector<float> got_dq((size_t) FF * H);
        std::vector<uint16_t> got_h((size_t) 2 * FF * H);
        const float* dtail = (const float*) (dblob + f.bytes - f.tail);
        size_t differ = 0, h_differ = 0;
        const struct { const uint8_t* src; const std::vector<float>* want; } mats[3] = {
            {dblob, &G}, {dblob + f.up_off, &U}, {dblob + f.down_off, &D}};
        for (const auto& m : mats) {
            K::iq_dequant_f32(GGML_TYPE_NVFP4, m.src, FF * H, dq, s);
            cudaStreamSynchronize(s);
            cudaMemcpy(got_dq.data(), dq, got_dq.size() * 4, cudaMemcpyDeviceToHost);
            for (size_t i = 0; i < got_dq.size(); ++i) differ += std::memcmp(&got_dq[i], &(*m.want)[i], 4) != 0;
        }
        K::iq_dequant_gu_f16(GGML_TYPE_NVFP4, dblob, dblob + f.up_off, FF, H, dh, s, dtail);
        cudaStreamSynchronize(s);
        cudaMemcpy(got_h.data(), dh, got_h.size() * 2, cudaMemcpyDeviceToHost);
        for (int64_t r = 0; r < FF; ++r)
            for (int64_t c = 0; c < H; ++c) {
                h_differ += got_h[(size_t) (2 * r) * H + c] != ggml_fp32_to_fp16(G[r * H + c] * sc[0]);
                h_differ += got_h[(size_t) (2 * r + 1) * H + c] != ggml_fp32_to_fp16(U[r * H + c] * sc[1]);
            }
        K::iq_dequant_f16(GGML_TYPE_NVFP4, dblob + f.down_off, FF * H, dh, s, dtail + 2);
        cudaStreamSynchronize(s);
        cudaMemcpy(got_h.data(), dh, (size_t) FF * H * 2, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < (size_t) FF * H; ++i) h_differ += got_h[i] != ggml_fp32_to_fp16(D[i] * sc[2]);
        cudaFree(dq); cudaFree(dh);
        std::printf("          gpu dequant: fp32 vs to_float %zu of %lld differ in any bit; scaled fp16 (prompt "
                    "fallback) vs fp16(to_float * s): %zu differ  %s\n", differ, (long long) (3 * FF * H), h_differ,
                    differ || h_differ ? "FAIL" : "ok");
        if (differ || h_differ) ++failures;
    }
    {   // microbench: one entry, one group, the blob L2-resident (repeated launches); the generic per-entry kernels
        const int32_t st1[2] = {0, 1};
        cudaMemcpy(dstart, st1, 8, cudaMemcpyHostToDevice);
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        for (int i = 0; i < 20; ++i) K::native_expert_grouped(L, dptr, dstart, dn, didx, didx, 1, 1, dxq, dscr, dout, s);
        const int it = 200;
        cudaEventRecord(a, s);
        for (int i = 0; i < it; ++i) K::native_expert_grouped(L, dptr, dstart, dn, didx, didx, 1, 1, dxq, dscr, dout, s);
        cudaEventRecord(b, s);
        cudaEventSynchronize(b);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a, b);
        const double us = 1e3 * ms / it;
        std::printf("          bench: gpu expert (native_expert_grouped, 1 entry) %.1f us (%.0f GB/s of weights)\n", us,
                    (double) f.bytes / us * 1e-3);
        cudaEventDestroy(a);
        cudaEventDestroy(b);
    }
    cudaFree(dblob); cudaFree(dx); cudaFree(dxq); cudaFree(dscr); cudaFree(dout); cudaFree(dptr);
    cudaFree(dstart); cudaFree(dn); cudaFree(didx);
    return failures;
}

// Expert `e` of `layer` from an NVFP4 pack: a native experts.bin (its native_experts.txt beside it) or a .basalt
// (its `experts` section and `experts.layout`).  Blobs are layer-major, `blob` bytes each.
bool read_pack_expert(const std::string& path, int layer, int e, const cpu::NativeFmt& f, std::vector<uint8_t>& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    uint64_t base = 0, n_expert = 512;
    char magic[8] = {};
    in.read(magic, 8);
    if (std::memcmp(magic, "BASALTMF", 8) == 0) {   // header (64 B) then 128-byte entries: name[64], off, size, ...
        uint32_t ver = 0, n = 0;
        in.read((char*) &ver, 4);
        in.read((char*) &n, 4);
        for (uint32_t i = 0; i < n; ++i) {
            char ent[128];
            in.seekg(64 + (std::streamoff) i * 128);
            in.read(ent, 128);
            uint64_t off, size;
            std::memcpy(&off, ent + 64, 8);
            std::memcpy(&size, ent + 72, 8);
            if (std::strncmp(ent, "experts", 64) == 0) base = off;
            if (std::strncmp(ent, "experts.layout", 64) == 0) {
                std::string text(size, '\0');
                in.seekg((std::streamoff) off);
                in.read(text.data(), (std::streamsize) size);
                const size_t at = text.find("(n_expert ");
                if (at != std::string::npos) n_expert = std::strtoull(text.c_str() + at + 10, nullptr, 10);
            }
        }
        if (base == 0) return false;
    }
    out.assign(f.bytes, 0);
    in.clear();
    in.seekg((std::streamoff) (base + ((uint64_t) layer * n_expert + (uint64_t) e) * f.bytes));
    in.read((char*) out.data(), (std::streamsize) f.bytes);
    return (size_t) in.gcount() == f.bytes;
}

// The Q5_1 min term.  Activations x = d * (q + 0.45) round to q, so each q8_1 block's sum of the ORIGINAL values
// (`ds.y`, what llama.cpp CUDA multiplies the min by) exceeds d * sum(q) by 0.45 * 32 * d.  The GPU dot must take
// the quantized sum, as ggml-cpu's vec_dot does: equal to ggml-cpu on the same blocks with s = d * sum(q), and far
// from the ds.y variant.  Weights with large mins make the difference visible.
int check_q5_1_min(cudaStream_t s) {
    constexpr int n = 640, rows = 64;
    std::mt19937 rng(5);
    std::uniform_int_distribution<int> qd(-100, 100);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> w((size_t) rows * n), x(n);
    for (auto& v : w) v = 0.3f + 0.05f * nd(rng);        // positive weights: a large min per block
    const float d = 0.01f;
    for (auto& v : x) v = d * ((float) qd(rng) + 0.45f);
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q5_1, n);
    std::vector<uint8_t> wq((size_t) rows * row_bytes);
    ggml_quantize_chunk(GGML_TYPE_Q5_1, w.data(), wq.data(), 0, rows, n);
    // the GPU: its own q8_1 quantizer, then iq_mmvq on Q5_1
    void *dw, *dx, *dxq;
    float* dy;
    cudaMalloc(&dw, wq.size());
    cudaMalloc(&dx, n * 4);
    cudaMalloc(&dxq, (size_t) n / 32 * 36);
    cudaMalloc((void**) &dy, rows * 4);
    cudaMemcpy(dw, wq.data(), wq.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(), n * 4, cudaMemcpyHostToDevice);
    basalt::kernels::quantize_q8_1_rows((const float*) dx, 1, n, dxq, s);
    basalt::kernels::iq_mmvq(GGML_TYPE_Q5_1, dw, dxq, dy, n, rows, 1, s);
    cudaStreamSynchronize(s);
    std::vector<float> gpu(rows);
    std::vector<uint8_t> xq((size_t) n / 32 * 36);
    cudaMemcpy(gpu.data(), dy, rows * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(xq.data(), dxq, xq.size(), cudaMemcpyDeviceToHost);
    cudaFree(dw); cudaFree(dx); cudaFree(dxq); cudaFree(dy);
    // ggml-cpu on the GPU's own blocks: once with s = d * sum(q) (the quantized sum), once with s = ds.y as stored
    struct Q81 { ggml_fp16_t d, s; int8_t qs[32]; };   // ggml's block_q8_1: {d, s} halves, 32 codes
    static_assert(sizeof(Q81) == 36, "block_q8_1 layout");
    auto* blocks = (Q81*) xq.data();
    std::vector<Q81> qsum(blocks, blocks + n / 32);
    double shift = 0.0;
    for (auto& b : qsum) {
        int sq = 0;
        for (int i = 0; i < 32; ++i) sq += b.qs[i];
        const float dd = ggml_fp16_to_fp32(b.d);
        shift = (std::max)(shift, (double) std::fabs(ggml_fp16_to_fp32(b.s) - dd * sq));
        b.s = ggml_fp32_to_fp16(dd * (float) sq);
    }
    const auto* tc = ggml_get_type_traits_cpu(GGML_TYPE_Q5_1);
    std::vector<float> want(rows), stored(rows);
    for (int r = 0; r < rows; ++r) {
        const uint8_t* row = wq.data() + (size_t) r * row_bytes;
        tc->vec_dot(n, &want[r], 0, row, 0, qsum.data(), 0, 1);
        tc->vec_dot(n, &stored[r], 0, row, 0, blocks, 0, 1);
    }
    const double e_q = rel(gpu, want), e_s = rel(gpu, stored);
    // the quantized sum goes through fp16 in ggml-cpu's block (s is a half), so 1e-3 rather than bitwise
    const bool ok = e_q < 1e-3 && e_s > 10 * e_q && shift > 0.0;
    std::printf("q5_1 min   row 0: GPU %.6g, ggml-cpu (quantized sum) %.6g, (stored sum) %.6g\n", gpu[0], want[0],
                stored[0]);
    std::printf("q5_1 min   GPU vs ggml-cpu with s = d*sum(q): rel %.2e; vs s = sum(x) (llama.cpp CUDA's): rel %.2e "
                "(largest per-block sum shift %.3g)  %s\n", e_q, e_s, shift, ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}
// the BF16 token embedding - iq_embed_rows and iq_dequant_f32 on a random BF16 table against
// the exact widening (bits << 16), every bit; rows gathered out of order, with repeats
int check_bf16_embd(cudaStream_t s) {
    constexpr int kBf16 = 30;
    const int64_t H = 2560, V = 61, NT = 97;
    int failures = 0;
    if (!basalt::kernels::embed_type_supported(kBf16) || basalt::kernels::iq_supported(kBf16) ||
        basalt::kernels::iq_row_bytes(kBf16, H) != (size_t) H * 2) {
        std::printf("bf16 embedding: type support / row bytes wrong\n");
        return 1;
    }
    std::mt19937 rng(290);
    std::vector<uint16_t> table((size_t) (V * H));
    for (auto& v : table) {   // any bit pattern but NaN (a NaN's payload is not what the test is about)
        do v = (uint16_t) (rng() & 0xffff); while ((v & 0x7f80) == 0x7f80 && (v & 0x7f));
    }
    std::vector<int32_t> tok((size_t) NT);
    for (auto& t : tok) t = (int32_t) (rng() % V);
    void* dt = nullptr;
    int32_t* dtok = nullptr;
    float* dout = nullptr;
    cudaMalloc(&dt, table.size() * 2);
    cudaMalloc((void**) &dtok, tok.size() * 4);
    const int64_t out_rows = NT > V ? NT : V;   // the gathered rows (NT) and the whole table (V) share the buffer
    cudaMalloc((void**) &dout, (size_t) (out_rows * H) * 4);
    cudaMemcpy(dt, table.data(), table.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dtok, tok.data(), tok.size() * 4, cudaMemcpyHostToDevice);
    auto widen = [](uint16_t b) { const uint32_t u = (uint32_t) b << 16; float f; std::memcpy(&f, &u, 4); return f; };
    std::vector<float> got((size_t) (out_rows * H));
    basalt::kernels::iq_embed_rows(kBf16, dt, (size_t) H * 2, dtok, NT, H, dout, s);
    cudaStreamSynchronize(s);
    cudaMemcpy(got.data(), dout, (size_t) (NT * H) * 4, cudaMemcpyDeviceToHost);
    size_t rows_differ = 0;
    for (int64_t r = 0; r < NT; ++r)
        for (int64_t d = 0; d < H; ++d) {
            const float want = widen(table[(size_t) (tok[(size_t) r] * H + d)]);
            if (std::memcmp(&got[(size_t) (r * H + d)], &want, 4) != 0) { ++rows_differ; break; }
        }
    basalt::kernels::iq_dequant_f32(kBf16, dt, V * H, dout, s);
    cudaStreamSynchronize(s);
    cudaMemcpy(got.data(), dout, table.size() * 4, cudaMemcpyDeviceToHost);
    size_t values_differ = 0;
    for (size_t i = 0; i < table.size(); ++i) {
        const float want = widen(table[i]);
        values_differ += std::memcmp(&got[i], &want, 4) != 0;
    }
    cudaFree(dt); cudaFree(dtok); cudaFree(dout);
    std::printf("bf16 embedding: %lld gathered rows, %zu differ; dequant of %lld values, %zu differ in any bit\n",
                (long long) NT, rows_differ, (long long) (V * H), values_differ);
    if (rows_differ || values_differ) ++failures;
    return failures;
}

// The CPU tier alone on random blobs of one pair (no GPU): each ISA build of the ggml dot against a double
// reference over the dequantized weights and activations, the multi-row expert against the float expert, and the
// time of one thread's gate/up rows.
extern "C" {
const ggml_type_traits_cpu* ggml_get_type_traits_cpu_avx2(ggml_type type);
const ggml_type_traits_cpu* ggml_get_type_traits_cpu_avx512(ggml_type type);
}

int check_cpu_pair(int gu, int dn, int seed) {
    int failures = 0;
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(gu, dn, H, FF, f, err)) {
        std::printf("%s/%s: %s\n", ggml_type_name((ggml_type) gu), ggml_type_name((ggml_type) dn), err.c_str());
        return 1;
    }
    const std::vector<uint8_t> blob = synthetic_blob(f, seed);
    const auto* tg = ggml_get_type_traits((ggml_type) gu);
    const auto* td = ggml_get_type_traits((ggml_type) dn);
    std::mt19937 rng(11 + seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) NT * H);
    for (auto& v : x) v = nd(rng);
    std::vector<std::vector<uint8_t>> act(NT, std::vector<uint8_t>(cpu::kNativeActBytes));
    const void* a[NT];
    for (int k = 0; k < NT; ++k) {
        cpu::native_quant_act(f, x.data() + k * H, act[k].data());
        a[k] = act[k].data();
    }
    // the activations the dot sees (Q8_K), dequantized
    std::vector<float> xd((size_t) NT * H);
    for (int k = 0; k < NT; ++k) {
        if (f.gu_act != GGML_TYPE_Q8_K) {
            std::printf("  the pair's activation is not Q8_K\n");
            return 1;
        }
        // block_q8_K: float d, int8 qs[256], int16 bsums[16]
        for (int64_t b = 0; b < H / 256; ++b) {
            const uint8_t* blk = act[k].data() + b * 292;
            float d;
            std::memcpy(&d, blk, 4);
            for (int i = 0; i < 256; ++i) xd[k * H + b * 256 + i] = d * (float) (int8_t) blk[4 + i];
        }
    }
    std::vector<float> G((size_t) FF * H), U((size_t) FF * H), D((size_t) H * FF);
    for (int64_t r = 0; r < FF; ++r) {
        tg->to_float(blob.data() + r * f.gu_row, G.data() + r * H, H);
        tg->to_float(blob.data() + f.up_off + r * f.gu_row, U.data() + r * H, H);
    }
    for (int64_t r = 0; r < H; ++r) td->to_float(blob.data() + f.down_off + r * f.d_row, D.data() + r * FF, FF);
    const struct { const char* name; const ggml_type_traits_cpu* (*get)(ggml_type); bool ok; } isas[] = {
        {"avx2", ggml_get_type_traits_cpu_avx2, cpu::cpu_avx2_ok()},
        {"avx512", ggml_get_type_traits_cpu_avx512, cpu::cpu_avx512_ok()}};
    for (const auto& isa : isas) {
        if (!isa.ok) continue;
        const auto* tc = isa.get((ggml_type) gu);
        if (tc == nullptr || tc->vec_dot == nullptr) {
            std::printf("          %s: no dot for %s\n", isa.name, ggml_type_name((ggml_type) gu));
            ++failures;
            continue;
        }
        double worst = 0.0;
        for (int k = 0; k < NT; ++k)
            for (int64_t r = 0; r < FF; ++r) {
                float got = 0.f;
                tc->vec_dot((int) H, &got, 0, blob.data() + r * f.gu_row, 0, a[k], 0, 1);
                double want = 0.0, mag = 0.0;
                for (int64_t i = 0; i < H; ++i) {
                    const double t = (double) G[r * H + i] * xd[k * H + i];
                    want += t;
                    mag += std::fabs(t);
                }
                worst = std::max(worst, std::fabs(got - want) / (mag + 1e-30));
            }
        std::printf("          %s %s dot vs double reference: worst error / sum|terms| %.2e\n", isa.name,
                    ggml_type_name((ggml_type) gu), worst);
        if (worst > 1e-5) {
            std::printf("          %s %s dot MISMATCH\n", isa.name, ggml_type_name((ggml_type) gu));
            ++failures;
        }
        const int it = 50;
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < it; ++i)
            for (int64_t r = 0; r < FF; ++r) {
                float gg, uu;
                tc->vec_dot((int) H, &gg, 0, blob.data() + r * f.gu_row, 0, a[0], 0, 1);
                tc->vec_dot((int) H, &uu, 0, blob.data() + f.up_off + r * f.gu_row, 0, a[0], 0, 1);
            }
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / it;
        std::printf("          gate+up one token one thread: %s %s %.0f us (%.2f GB/s, weights cache-resident)\n",
                    isa.name, ggml_type_name((ggml_type) gu), us, 2.0 * f.up_off / us / 1e3);
    }
    // the expert through the tier's own entry points against the float expert
    std::vector<std::vector<uint8_t>> hq(NT, std::vector<uint8_t>(cpu::kNativeHBytes));
    std::vector<std::vector<float>> ff(NT, std::vector<float>(FF));
    std::vector<float> got((size_t) NT * H), ref((size_t) NT * H);
    float* ffp[NT];
    const void* hp[NT];
    float* op[NT];
    for (int k = 0; k < NT; ++k) ffp[k] = ff[k].data();
    cpu::native_gu_rows(f, blob.data(), a, NT, ffp, 0, (int) FF);
    for (int k = 0; k < NT; ++k) {
        cpu::native_quant_h(f, ff[k].data(), hq[k].data());
        hp[k] = hq[k].data();
        op[k] = got.data() + k * H;
    }
    cpu::native_down_rows(f, blob.data(), hp, NT, op, 0, (int) H);
    for (int k = 0; k < NT; ++k) {
        std::vector<float> h(FF);
        for (int64_t r = 0; r < FF; ++r) {
            double g = 0, u = 0;
            for (int64_t i = 0; i < H; ++i) { g += (double) G[r * H + i] * x[k * H + i]; u += (double) U[r * H + i] * x[k * H + i]; }
            h[r] = (float) (g / (1.0 + std::exp(-g)) * u);
        }
        for (int64_t r = 0; r < H; ++r) {
            double o = 0;
            for (int64_t i = 0; i < FF; ++i) o += (double) D[r * FF + i] * h[i];
            ref[k * H + r] = (float) o;
        }
    }
    const double ec = rel(got, ref);
    std::printf("  %s/%s cpu expert vs float reference: rel %.2e\n", ggml_type_name((ggml_type) gu),
                ggml_type_name((ggml_type) dn), ec);
    if (ec > 3e-2) ++failures;
    return failures;
}

// the Q6_K token embedding - iq_embed_rows and iq_dequant_f32 on random blocks against ggml's dequantize_row_q6_K
int check_q6k_embd(cudaStream_t s) {
    constexpr int kQ6K = 14;
    const int64_t H = 2560, V = 61, NT = 97;
    if (!basalt::kernels::embed_type_supported(kQ6K) ||
        basalt::kernels::iq_row_bytes(kQ6K, H) != (size_t) (H / 256) * 210) {
        std::printf("q6_K embedding: type support / row bytes wrong\n");
        return 1;
    }
    const size_t rb = basalt::kernels::iq_row_bytes(kQ6K, H);
    std::mt19937 rng(614);
    std::vector<uint8_t> table(V * rb);
    for (auto& b : table) b = (uint8_t) (rng() & 0xff);
    for (size_t blk = 0; blk < table.size() / 210; ++blk) {   // d (the last two bytes of a block): a finite half
        const ggml_fp16_t d = ggml_fp32_to_fp16(((float) (rng() % 2000) - 1000.0f) / 4096.0f);
        std::memcpy(&table[blk * 210 + 208], &d, 2);
    }
    std::vector<float> ref((size_t) (V * H));
    const auto* tg = ggml_get_type_traits(GGML_TYPE_Q6_K);
    tg->to_float(table.data(), ref.data(), V * H);
    std::vector<int32_t> tok((size_t) NT);
    for (auto& t : tok) t = (int32_t) (rng() % V);
    void* dt = nullptr;
    int32_t* dtok = nullptr;
    float* dout = nullptr;
    cudaMalloc(&dt, table.size());
    cudaMalloc((void**) &dtok, tok.size() * 4);
    cudaMalloc((void**) &dout, (size_t) ((NT > V ? NT : V) * H) * 4);
    cudaMemcpy(dt, table.data(), table.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dtok, tok.data(), tok.size() * 4, cudaMemcpyHostToDevice);
    std::vector<float> got((size_t) ((NT > V ? NT : V) * H));
    basalt::kernels::iq_embed_rows(kQ6K, dt, rb, dtok, NT, H, dout, s);
    cudaStreamSynchronize(s);
    cudaMemcpy(got.data(), dout, (size_t) (NT * H) * 4, cudaMemcpyDeviceToHost);
    size_t gather_differ = 0;
    double max_diff = 0.0;
    for (int64_t r = 0; r < NT; ++r)
        for (int64_t d = 0; d < H; ++d) {
            const float a = got[(size_t) (r * H + d)], b = ref[(size_t) (tok[(size_t) r] * H + d)];
            gather_differ += std::memcmp(&a, &b, 4) != 0;
            max_diff = (std::max)(max_diff, (double) std::fabs(a - b));
        }
    basalt::kernels::iq_dequant_f32(kQ6K, dt, V * H, dout, s);
    cudaStreamSynchronize(s);
    cudaMemcpy(got.data(), dout, ref.size() * 4, cudaMemcpyDeviceToHost);
    size_t flat_differ = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        flat_differ += std::memcmp(&got[i], &ref[i], 4) != 0;
        max_diff = (std::max)(max_diff, (double) std::fabs(got[i] - ref[i]));
    }
    cudaFree(dt); cudaFree(dtok); cudaFree(dout);
    std::printf("q6_K embedding: %lld gathered rows, %zu values differ; dequant of %lld values, %zu differ; max abs diff %.3g\n",
                (long long) NT, gather_differ, (long long) (V * H), flat_differ, max_diff);
    return gather_differ || flat_differ ? 1 : 0;
}

// The Q8_0 token embedding at the model's geometry (2560 x 248,320) in mapped host memory with its first
// NativeEmbed::kHotRows rows also in VRAM: iq_embed_rows with the hot copy against the same gather without it and
// against ggml's dequantize_row_q8_0, bit for bit, on ids across the boundary; then the per-call time of each.
int check_q8_embd_hot(cudaStream_t s) {
    constexpr int kQ80 = 8;
    const int64_t H = 2560, V = 248320, NHOT = basalt::core::NativeEmbed::kHotRows, NT = 512;
    const size_t rb = basalt::kernels::iq_row_bytes(kQ80, H);
    if (rb != (size_t) (H / 32) * 34) { std::printf("q8_0 embedding: row bytes wrong\n"); return 1; }
    void* host = nullptr;
    if (cudaHostAlloc(&host, rb * V, cudaHostAllocMapped) != cudaSuccess) { std::printf("q8_0 embedding: no pinned table\n"); return 1; }
    auto* table = (uint8_t*) host;
    std::mt19937 rng(808);
    for (size_t i = 0; i < rb * V; i += 4) { const uint32_t r = rng(); std::memcpy(table + i, &r, std::min<size_t>(4, rb * V - i)); }
    for (size_t blk = 0; blk < rb * V / 34; ++blk) {   // d (the first two bytes of a block): a finite half
        const ggml_fp16_t d = ggml_fp32_to_fp16(((float) (rng() % 2000) - 1000.0f) / 8192.0f);
        std::memcpy(table + blk * 34, &d, 2);
    }
    void* mapped = nullptr;
    cudaHostGetDevicePointer(&mapped, host, 0);
    void* hot = nullptr;
    cudaMalloc(&hot, rb * NHOT);
    cudaMemcpy(hot, host, rb * NHOT, cudaMemcpyHostToDevice);
    std::vector<int32_t> tok((size_t) NT);
    const int32_t edge[] = {0, 1, (int32_t) NHOT - 1, (int32_t) NHOT, (int32_t) NHOT + 1, (int32_t) V - 1};
    for (int64_t i = 0; i < NT; ++i)
        tok[(size_t) i] = i < 6 ? edge[i] : (int32_t) (i % 2 ? rng() % NHOT : rng() % V);
    int32_t* dtok = nullptr;
    float *da = nullptr, *db = nullptr;
    cudaMalloc((void**) &dtok, tok.size() * 4);
    cudaMalloc((void**) &da, (size_t) (NT * H) * 4);
    cudaMalloc((void**) &db, (size_t) (NT * H) * 4);
    cudaMemcpy(dtok, tok.data(), tok.size() * 4, cudaMemcpyHostToDevice);
    basalt::kernels::iq_embed_rows(kQ80, mapped, rb, dtok, NT, H, da, s, hot, NHOT);
    basalt::kernels::iq_embed_rows(kQ80, mapped, rb, dtok, NT, H, db, s);
    cudaStreamSynchronize(s);
    std::vector<float> a((size_t) (NT * H)), b(a.size()), ref((size_t) H);
    cudaMemcpy(a.data(), da, a.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), db, b.size() * 4, cudaMemcpyDeviceToHost);
    const auto* tq = ggml_get_type_traits(GGML_TYPE_Q8_0);
    size_t vs_mapped = 0, vs_ref = 0;
    for (int64_t r = 0; r < NT; ++r) {
        tq->to_float(table + (size_t) tok[(size_t) r] * rb, ref.data(), H);
        vs_mapped += std::memcmp(&a[(size_t) (r * H)], &b[(size_t) (r * H)], (size_t) H * 4) != 0;
        vs_ref += std::memcmp(&a[(size_t) (r * H)], ref.data(), (size_t) H * 4) != 0;
    }
    std::printf("q8_0 embedding, %lld hot rows: %lld gathered rows, %zu differ from the mapped-only gather, %zu from ggml\n",
                (long long) NHOT, (long long) NT, vs_mapped, vs_ref);
    // per call: fresh ids each call (a hit in L2 would hide the PCIe read), rows 1 and 3
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    const int iters = 400;
    for (int T : {1, 3}) {
        for (int mode = 0; mode < 3; ++mode) {   // 0 mapped only, 1 hot copy with hot ids, 2 hot copy with cold ids
            std::vector<int32_t> ids((size_t) (iters * T));
            for (auto& v : ids) v = (int32_t) (mode == 1 ? rng() % NHOT : NHOT + rng() % (V - NHOT));
            int32_t* dids = nullptr;
            cudaMalloc((void**) &dids, ids.size() * 4);
            cudaMemcpy(dids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice);
            const void* h = mode == 0 ? nullptr : hot;
            for (int it = 0; it < 20; ++it) basalt::kernels::iq_embed_rows(kQ80, mapped, rb, dids + it * T, T, H, da, s, h, NHOT);
            cudaEventRecord(e0, s);
            for (int it = 0; it < iters; ++it)
                basalt::kernels::iq_embed_rows(kQ80, mapped, rb, dids + it * T, T, H, da, s, h, NHOT);
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, e0, e1);
            const double us = 1000.0 * ms / iters;
            std::printf("  bench: embed rows %d, %-24s %6.2f us per call, %5.2f GB/s\n", T,
                        mode == 0 ? "mapped" : mode == 1 ? "hot copy, hot ids" : "hot copy, cold ids", us,
                        (double) (T * rb) / (us * 1e3));
            cudaFree(dids);
        }
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(hot); cudaFree(dtok); cudaFree(da); cudaFree(db);
    cudaFreeHost(host);
    return vs_mapped || vs_ref ? 1 : 0;
}

// The AVX-512 Q8_K activation quantizer against ggml's reference quantizer: every byte of every block equal, on
// random vectors of the model's width (dense, sparse, tied maxima, all-zero blocks, wide dynamic range).
// One quantizer against ggml's reference: bitwise on random, sparse, tied, wide-range and all-zero-block vectors, and a
// per-vector timing line.
int check_q8k_one(const char* tag, void (*quant)(const float*, void*, int64_t)) {
    const auto* tr = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    const size_t bytes = ggml_row_size(GGML_TYPE_Q8_K, H);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> pick(0, 9);
    std::vector<float> x((size_t) H);
    std::vector<uint8_t> a(bytes), b(bytes);
    int bad = 0;
    const int N = 200000;
    for (int it = 0; it < N; ++it) {
        const int kind = it % 6;
        for (int64_t i = 0; i < H; ++i) {
            float v = nd(rng);
            if (kind == 1 && pick(rng) != 0) v = 0.f;                          // sparse
            if (kind == 2) v = (float) (pick(rng) - 5);                        // ties and equal +-max
            if (kind == 3) v *= std::exp((float) pick(rng) * 2.f - 9.f);       // wide dynamic range
            if (kind == 4 && (i / 256) % 3 == 0) v = 0.f;                      // all-zero blocks
            x[(size_t) i] = v;
        }
        std::fill(a.begin(), a.end(), 0xA5);
        std::fill(b.begin(), b.end(), 0xA5);
        tr->from_float(x.data(), a.data(), H);
        quant(x.data(), b.data(), H);
        // ggml leaves the bsums of an all-zero block unwritten; they are zero there
        const size_t blk = bytes / (H / 256);
        for (int64_t k = 0; k < H / 256; ++k) {
            const auto* ya = a.data() + (size_t) k * blk;
            float d;
            std::memcpy(&d, ya, 4);
            if (d == 0.f) std::memset(a.data() + (size_t) k * blk + 4 + 256, 0, 32);
        }
        if (std::memcmp(a.data(), b.data(), bytes) != 0) ++bad;
    }
    std::vector<uint8_t> o(bytes);
    for (int i = 0; i < 1000; ++i) {
        for (int64_t j = 0; j < H; ++j) x[(size_t) j] = nd(rng);
        quant(x.data(), o.data(), H);
    }
    auto t0 = std::chrono::steady_clock::now();
    const int R = 200000;
    for (int i = 0; i < R; ++i) tr->from_float(x.data(), a.data(), H);
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < R; ++i) quant(x.data(), b.data(), H);
    auto t2 = std::chrono::steady_clock::now();
    const double ur = std::chrono::duration<double, std::micro>(t1 - t0).count() / R;
    const double uf = std::chrono::duration<double, std::micro>(t2 - t1).count() / R;
    std::printf("q8_K quantizer %s: %d of %d vectors differ from ggml's reference (bitwise)\n", tag, bad, N);
    std::printf("q8_K quantizer microbench, %lld floats: ggml reference %.3f us, %s %.3f us (%.1fx)\n", (long long) H, ur, tag, uf, ur / uf);
    return bad != 0 ? 1 : 0;
}

int check_q8k_quant() {
    int failures = 0;
    if (cpu::cpu_avx512_ok()) failures += check_q8k_one("AVX-512", cpu::quantize_q8_K_avx512);
    else std::printf("q8_K quantizer: no AVX-512 on this CPU, skipped\n");
    if (cpu::cpu_avx2_ok()) failures += check_q8k_one("AVX-2", cpu::quantize_q8_K_avx2);
    return failures;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);   // keep the trail on a crash
    if (argc < 2) {
        std::fprintf(stderr, "usage: native_expert_parity <shard.gguf> [layer ...]\n"
                             "       native_expert_parity --synthetic GU/DOWN ...   (ggml type names, e.g. q4_K/q5_1)\n"
                             "       native_expert_parity --cpu-synthetic GU/DOWN ...   (CPU tier only, e.g. q6_K/q8_0)\n"
                             "       native_expert_parity --experts experts.bin|MODEL.basalt [layer ...]   (NVFP4)\n"
                             "       native_expert_parity --q5_1-min\n"
                             "       native_expert_parity --q8k-quant\n"
                             "       native_expert_parity --bf16-embd\n"
                             "       native_expert_parity --q6k-embd\n"
                             "       native_expert_parity --q8-embd-hot\n");
        return 2;
    }
    int failures = 0;
    cudaStream_t s = nullptr;   // created on first use
    auto gpu = [&]() {
        if (s == nullptr) cudaStreamCreate(&s);
        return s;
    };
    const std::string mode = argv[1];
    if (mode == "--q8k-quant") {
        failures += check_q8k_quant();
    } else if (mode == "--q5_1-min") {
        failures += check_q5_1_min(gpu());
    } else if (mode == "--bf16-embd") {
        failures += check_bf16_embd(gpu());
    } else if (mode == "--q6k-embd") {
        failures += check_q6k_embd(gpu());
    } else if (mode == "--q8-embd-hot") {
        failures += check_q8_embd_hot(gpu());
    } else if (mode == "--cpu-synthetic") {
        for (int i = 2; i < argc; ++i) {
            const std::string arg = argv[i];
            const size_t slash = arg.find('/');
            const int gu = slash == std::string::npos ? -1 : type_of(arg.substr(0, slash));
            const int dn = slash == std::string::npos ? -1 : type_of(arg.substr(slash + 1));
            if (gu < 0 || dn < 0) { std::printf("%s: not GU/DOWN type names\n", arg.c_str()); ++failures; continue; }
            failures += check_cpu_pair(gu, dn, i);
        }
    } else if (mode == "--synthetic") {
        for (int i = 2; i < argc; ++i) {
            const std::string arg = argv[i];
            const size_t slash = arg.find('/');
            const int gu = slash == std::string::npos ? -1 : type_of(arg.substr(0, slash));
            const int dn = slash == std::string::npos ? -1 : type_of(arg.substr(slash + 1));
            cpu::NativeFmt f;
            std::string err;
            if (gu == GGML_TYPE_NVFP4 && dn == GGML_TYPE_NVFP4 && cpu::native_fmt(gu, dn, H, FF, f, err)) {
                failures += check_nvfp4_cpu(f, synthetic_nvfp4_blob(f, i), i, "synthetic", gpu());
                continue;
            }
            if (gu < 0 || dn < 0 || !cpu::native_fmt(gu, dn, H, FF, f, err) ||
                !basalt::kernels::native_expert_supported(gu, dn, H, FF)) {
                std::printf("%s: %s\n", arg.c_str(), err.empty() ? "not a pair the GPU expert kernels take" : err.c_str());
                ++failures;
                continue;
            }
            failures += check_blob(f, synthetic_blob(f, i), i, "synthetic", gpu());
        }
    } else if (mode == "--experts") {
        if (argc < 3) { std::fprintf(stderr, "--experts FILE [layer ...]\n"); return 2; }
        cpu::NativeFmt f;
        std::string err;
        if (!cpu::native_fmt(GGML_TYPE_NVFP4, GGML_TYPE_NVFP4, H, FF, f, err)) {
            std::printf("%s\n", err.c_str());
            return 1;
        }
        std::vector<int> layers;
        for (int i = 3; i < argc; ++i) layers.push_back(std::atoi(argv[i]));
        if (layers.empty()) layers = {0, 1, 2, 3, 20, 47};
        for (int l : layers) {
            std::vector<uint8_t> blob;
            if (!read_pack_expert(argv[2], l, E, f, blob)) {
                std::printf("layer %d: cannot read expert %d from %s\n", l, E, argv[2]);
                ++failures;
                continue;
            }
            failures += check_nvfp4_cpu(f, blob, l, "layer " + std::to_string(l), gpu());
        }
    } else {
        const basalt::GgufModel model(basalt::gguf_split_paths(argv[1]));
        std::vector<int> layers;
        for (int i = 2; i < argc; ++i) layers.push_back(std::atoi(argv[i]));
        if (layers.empty()) layers = {0, 1, 2, 3, 20, 47};
        for (int l : layers) {
            const basalt::TensorInfo* t[3] = {};
            const uint8_t* data[3] = {};
            const char* roles[3] = {"gate", "up", "down"};
            for (int r = 0; r < 3; ++r) {   // each role from the shard that holds it
                size_t at = 0;
                t[r] = model.find("blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight", &at);
                if (t[r]) data[r] = model.shard(at).tensor_data(*t[r]);
            }
            if (!t[0] || !t[1] || !t[2]) { std::printf("layer %d: no expert tensors\n", l); ++failures; continue; }
            cpu::NativeFmt f;
            std::string err;
            if (!cpu::native_fmt((int) t[0]->type, (int) t[2]->type, H, FF, f, err)) {
                std::printf("layer %d: %s\n", l, err.c_str()); ++failures; continue;
            }
            std::vector<uint8_t> blob(f.bytes);
            std::memcpy(blob.data(), data[0] + (size_t) E * f.up_off, f.up_off);
            std::memcpy(blob.data() + f.up_off, data[1] + (size_t) E * f.up_off, f.up_off);
            std::memcpy(blob.data() + f.down_off, data[2] + (size_t) E * (f.bytes - f.down_off),
                        f.bytes - f.down_off);
            failures += check_blob(f, blob, l, "layer " + std::to_string(l), gpu());
        }
    }
    std::printf("native_expert_parity: %d failures\n", failures);
    return failures ? 1 : 0;
}
