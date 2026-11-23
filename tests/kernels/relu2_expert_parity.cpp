// tests/kernels/relu2_expert_parity.cpp - squared-ReLU NVFP4 experts (no gate rows) at Nemotron-H Super's geometry
// (latent 1024, intermediate 2688), three ways.
//
//     build/relu2_expert_parity          the CPU tier and the GPU grouped kernels
//     build/relu2_expert_parity --cpu    the CPU tier only
//
// (a) the reference: ggml's own dequantizer (`to_float`), h = relu(in . x * s_in)^2, out = down . h * s_down, in
// double; (b) the CPU tier (native_gu_rows / native_down_rows with a relu2 NativeFmt, Q8_0 activations); (c) the GPU
// (native_expert_grouped with a relu2 layout, q8_1 activations).  (b) and (c) differ from (a) by their activation
// rounding; against a double product over their own rounded activations they agree to float order.  A token's rows do
// not depend on the entries beside it: (b) every token alone and in a batch, (c) every entry alone, in one batch, and
// with the group stride, all bitwise.  Then a microbench of a decode window's grouped call.
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/kernels/iq_kernels.hpp"

#include "basalt/ggml/ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = basalt::kernels::cpu;
namespace K = basalt::kernels;

namespace {
constexpr int64_t LAT = 1024, FF = 2688;
constexpr int NE = 6;                 // experts
constexpr int NT = 8;                 // tokens (a decode window's rows)

double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) { n += std::fabs((double) a[i] - b[i]); d += std::fabs((double) b[i]); }
    return n / (d + 1e-30);
}

// An NVFP4 relu2 blob with the checkpoint's magnitudes: block values in the thousands, the FP32 scales in the tail
// bring them back ({s_in, 0, s_down, 0}).
std::vector<uint8_t> make_blob(const cpu::NativeFmt& f, int seed, float& s_in, float& s_d) {
    std::vector<uint8_t> blob(f.bytes);
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    s_in = 3e-5f * (1.f + 0.1f * (float) (seed % 5));
    s_d = 7e-5f;
    auto quant = [&](float scale, int64_t rows, int64_t cols, uint8_t* dst) {
        std::vector<float> w((size_t) (rows * cols));
        for (int64_t r = 0; r < rows; ++r) {
            const float rs = 0.03f * (0.5f + (float) (r % 7) / 7.0f) / scale;
            for (int64_t c = 0; c < cols; ++c) w[(size_t) (r * cols + c)] = rs * nd(rng);
        }
        ggml_quantize_chunk(GGML_TYPE_NVFP4, w.data(), dst, 0, rows, cols);
    };
    quant(s_in, FF, LAT, blob.data());
    quant(s_d, LAT, FF, blob.data() + f.down_off);
    const float tail[4] = {s_in, 0.f, s_d, 0.f};
    std::memcpy(blob.data() + f.bytes - f.tail, tail, sizeof tail);
    return blob;
}

struct Expert {
    std::vector<uint8_t> blob;
    std::vector<float> in, down;   // dequantized, unscaled
    float s_in = 0, s_d = 0;
};

// out = down . relu(in . x * s_in)^2 * s_d in double, from the dequantized weights and the given activations
void reference(const Expert& e, const std::vector<double>& x, std::vector<double>& h, std::vector<float>& out) {
    h.assign(FF, 0.0);
    for (int64_t r = 0; r < FF; ++r) {
        double a = 0;
        for (int64_t i = 0; i < LAT; ++i) a += (double) e.in[(size_t) (r * LAT + i)] * x[(size_t) i];
        a = std::max(a * e.s_in, 0.0);
        h[(size_t) r] = a * a;
    }
    out.assign(LAT, 0.f);
    for (int64_t r = 0; r < LAT; ++r) {
        double o = 0;
        for (int64_t i = 0; i < FF; ++i) o += (double) e.down[(size_t) (r * FF + i)] * h[(size_t) i];
        out[(size_t) r] = (float) (o * e.s_d);
    }
}

// q8_0 / q8_1 blocks back to doubles (q8_0: half d then 32 int8; q8_1: half d, half s, then 32 int8)
std::vector<double> deq(const uint8_t* b, int64_t n, bool q8_1) {
    const int64_t bs = q8_1 ? 36 : 34, off = q8_1 ? 4 : 2;
    std::vector<double> o((size_t) n);
    for (int64_t i = 0; i < n; ++i) {
        const uint8_t* blk = b + (i / 32) * bs;
        uint16_t dh;
        std::memcpy(&dh, blk, 2);
        o[(size_t) i] = (double) ggml_fp16_to_fp32(dh) * (double) (int8_t) blk[off + i % 32];
    }
    return o;
}

int check_cpu(const cpu::NativeFmt& f, const std::vector<Expert>& ex, const std::vector<std::vector<float>>& x) {
    int failures = 0;
    std::vector<std::vector<uint8_t>> act(NT, std::vector<uint8_t>(f.act_bytes)), hq(NT, std::vector<uint8_t>(f.h_bytes));
    for (int t = 0; t < NT; ++t) cpu::native_quant_act(f, x[(size_t) t].data(), act[(size_t) t].data());
    const void* a[NT];
    for (int t = 0; t < NT; ++t) a[t] = act[(size_t) t].data();
    double worst_ref = 0, worst_own = 0;
    bool alone_same = true;
    for (int e = 0; e < NE; ++e) {
        const Expert& E = ex[(size_t) e];
        std::vector<std::vector<float>> ffb(NT, std::vector<float>(FF)), outb(NT, std::vector<float>(LAT));
        float* ffp[NT];
        float* outp[NT];
        for (int t = 0; t < NT; ++t) { ffp[t] = ffb[(size_t) t].data(); outp[t] = outb[(size_t) t].data(); }
        // a batch of every token, its rows in two ranges that split the matrix unevenly
        cpu::native_gu_rows(f, E.blob.data(), a, NT, ffp, 0, 1000);
        cpu::native_gu_rows(f, E.blob.data(), a, NT, ffp, 1000, (int) FF);
        const void* hp[NT];
        for (int t = 0; t < NT; ++t) {
            cpu::native_quant_h(f, ffb[(size_t) t].data(), hq[(size_t) t].data());
            hp[t] = hq[(size_t) t].data();
        }
        cpu::native_down_rows(f, E.blob.data(), hp, NT, outp, 0, (int) LAT);
        for (int t = 0; t < NT; ++t) {
            // each token alone: bitwise the batch's rows
            std::vector<float> ff1(FF), out1(LAT);
            float* f1 = ff1.data();
            float* o1 = out1.data();
            const void* a1 = a[t];
            cpu::native_gu_rows(f, E.blob.data(), &a1, 1, &f1, 0, (int) FF);
            const void* h1 = hp[t];
            cpu::native_down_rows(f, E.blob.data(), &h1, 1, &o1, 0, (int) LAT);
            if (std::memcmp(ff1.data(), ffb[(size_t) t].data(), FF * 4) || std::memcmp(out1.data(), outb[(size_t) t].data(), LAT * 4))
                alone_same = false;
            // against double over the true x, and over the CPU's own rounded activations
            std::vector<double> xd(x[(size_t) t].begin(), x[(size_t) t].end()), h;
            std::vector<float> ref;
            reference(E, xd, h, ref);
            worst_ref = std::max(worst_ref, rel(outb[(size_t) t], ref));
            const std::vector<double> xq = deq(act[(size_t) t].data(), LAT, false);
            reference(E, xq, h, ref);
            std::vector<float> hf(h.begin(), h.end());
            worst_own = std::max(worst_own, rel(ffb[(size_t) t], hf));
        }
    }
    const bool ok = worst_ref < 3e-2 && worst_own < 1e-5 && alone_same;
    std::printf("cpu   relu2 nvfp4 %lldx%lld: vs double ref %.2e, h vs double on its own q8_0 x %.2e, token alone = batch %s  %s\n",
                (long long) LAT, (long long) FF, worst_ref, worst_own, alone_same ? "yes" : "NO", ok ? "ok" : "FAIL");
    if (!ok) ++failures;
    return failures;
}

int check_gpu(const cpu::NativeFmt& f, const std::vector<Expert>& ex, const std::vector<std::vector<float>>& x) {
    int failures = 0;
    const K::NativeExpertLayout L = K::native_expert_layout_relu2(40, 40, LAT, FF);
    if (!L.relu2 || L.bytes != f.bytes || L.up_off != f.up_off || L.down_off != f.down_off || L.up_row != 0) {
        std::printf("gpu   relu2 layout %zu B / down at %zu differs from the CPU's %zu / %zu  FAIL\n", L.bytes, L.down_off,
                    f.bytes, f.down_off);
        return 1;
    }
    if (!K::native_expert_supported(40, 40, LAT, FF)) {
        std::printf("gpu   NVFP4 %lldx%lld is not a pair the expert kernels take  FAIL\n", (long long) LAT, (long long) FF);
        return 1;
    }
    cudaStream_t s;
    cudaStreamCreate(&s);
    // the routing of a window: expert e takes the tokens with (t + e) % 3 != 0, so groups have 5 or 6 entries
    std::vector<int32_t> ent_tok, start{0};
    std::vector<unsigned long long> ptr;
    uint8_t* dblobs = nullptr;
    cudaMalloc((void**) &dblobs, (size_t) NE * f.bytes);
    for (int e = 0; e < NE; ++e) {
        cudaMemcpy(dblobs + (size_t) e * f.bytes, ex[(size_t) e].blob.data(), f.bytes, cudaMemcpyHostToDevice);
        ptr.push_back((unsigned long long) (dblobs + (size_t) e * f.bytes));
        for (int t = 0; t < NT; ++t)
            if ((t + e) % 3 != 0) ent_tok.push_back(t);
        start.push_back((int32_t) ent_tok.size());
    }
    const int n_ent = (int) ent_tok.size();
    std::vector<int32_t> ent_dst(n_ent);
    for (int i = 0; i < n_ent; ++i) ent_dst[(size_t) i] = i;
    const size_t scr = K::native_expert_scratch_bytes(n_ent, FF);
    float *dx, *dout;
    uint8_t *dxq, *dscr;
    unsigned long long* dptr;
    int32_t *dstart, *dn, *dtok, *ddst;
    cudaMalloc((void**) &dx, (size_t) NT * LAT * 4);
    cudaMalloc((void**) &dxq, (size_t) NT * LAT / 32 * 36);
    cudaMalloc((void**) &dscr, scr);
    cudaMalloc((void**) &dout, (size_t) n_ent * LAT * 4);
    cudaMalloc((void**) &dptr, NE * 8);
    cudaMalloc((void**) &dstart, (NE + 1) * 4);
    cudaMalloc((void**) &dn, 4);
    cudaMalloc((void**) &dtok, (size_t) n_ent * 4);
    cudaMalloc((void**) &ddst, (size_t) n_ent * 4);
    for (int t = 0; t < NT; ++t) cudaMemcpy(dx + (size_t) t * LAT, x[(size_t) t].data(), LAT * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dptr, ptr.data(), NE * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dstart, start.data(), (NE + 1) * 4, cudaMemcpyHostToDevice);
    const int32_t ng = NE;
    cudaMemcpy(dn, &ng, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dtok, ent_tok.data(), (size_t) n_ent * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(ddst, ent_dst.data(), (size_t) n_ent * 4, cudaMemcpyHostToDevice);
    cudaMemset(dscr, 0, scr);
    K::quantize_q8_1_rows(dx, NT, LAT, dxq, s);
    K::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, NE, n_ent, dxq, dscr, dout, s);
    cudaStreamSynchronize(s);
    std::vector<float> got((size_t) n_ent * LAT), pre((size_t) n_ent * FF);
    std::vector<uint8_t> xq((size_t) NT * LAT / 32 * 36), hq((size_t) n_ent * FF / 32 * 36);
    const size_t fa = (((size_t) n_ent * FF * 4) + 255) & ~(size_t) 255;   // scratch: gate, up, h (fp32), then h q8_1
    cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(pre.data(), dscr, pre.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hq.data(), dscr + 3 * fa, hq.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(xq.data(), dxq, xq.size(), cudaMemcpyDeviceToHost);
    const cudaError_t ce = cudaGetLastError();
    // per entry: the input rows (pre-activation, scaled) and the output against double on the GPU's own q8_1 x and h,
    // and the output against the double reference on the true x
    double e_pre = 0, e_out = 0, e_ref = 0;
    for (int e = 0; e < NE; ++e) {
        const Expert& E = ex[(size_t) e];
        for (int i = start[(size_t) e]; i < start[(size_t) e + 1]; ++i) {
            const int t = ent_tok[(size_t) i];
            const std::vector<double> xd = deq(xq.data() + (size_t) t * LAT / 32 * 36, LAT, true);
            const std::vector<double> hd = deq(hq.data() + (size_t) i * FF / 32 * 36, FF, true);
            std::vector<float> rp(FF), ro(LAT);
            for (int64_t r = 0; r < FF; ++r) {
                double a = 0;
                for (int64_t k = 0; k < LAT; ++k) a += (double) E.in[(size_t) (r * LAT + k)] * xd[(size_t) k];
                rp[(size_t) r] = (float) (a * E.s_in);
            }
            for (int64_t r = 0; r < LAT; ++r) {
                double o = 0;
                for (int64_t k = 0; k < FF; ++k) o += (double) E.down[(size_t) (r * FF + k)] * hd[(size_t) k];
                ro[(size_t) r] = (float) (o * E.s_d);
            }
            const std::vector<float> gp(pre.begin() + (ptrdiff_t) i * FF, pre.begin() + (ptrdiff_t) (i + 1) * FF);
            const std::vector<float> go(got.begin() + (ptrdiff_t) i * LAT, got.begin() + (ptrdiff_t) (i + 1) * LAT);
            e_pre = std::max(e_pre, rel(gp, rp));
            e_out = std::max(e_out, rel(go, ro));
            std::vector<double> xt(x[(size_t) t].begin(), x[(size_t) t].end()), h;
            std::vector<float> ref;
            reference(E, xt, h, ref);
            e_ref = std::max(e_ref, rel(go, ref));
        }
    }
    bool ok = ce == cudaSuccess && e_pre < 1e-5 && e_out < 1e-5 && e_ref < 3e-2;
    std::printf("gpu   relu2 nvfp4 %lldx%lld, %d entries in %d groups: rows %.2e down %.2e vs double on its own q8_1 x/h; "
                "vs double ref %.2e  %s\n", (long long) LAT, (long long) FF, n_ent, NE, e_pre, e_out, e_ref, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
    // every entry alone (its own one-group call), and the batch with the group stride: bitwise the batch's rows
    {
        bool alone = true, strided = true;
        std::vector<float> one(LAT), again((size_t) n_ent * LAT);
        for (int e = 0; e < NE; ++e)
            for (int i = start[(size_t) e]; i < start[(size_t) e + 1]; ++i) {
                const int32_t st1[2] = {i, i + 1}, n1 = 1, dst0 = 0;
                cudaMemcpy(dptr, &ptr[(size_t) e], 8, cudaMemcpyHostToDevice);
                cudaMemcpy(dstart, st1, 8, cudaMemcpyHostToDevice);
                cudaMemcpy(dn, &n1, 4, cudaMemcpyHostToDevice);
                cudaMemcpy(ddst + i, &dst0, 4, cudaMemcpyHostToDevice);
                K::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, 1, n_ent, dxq, dscr, dout, s);
                cudaMemcpy(one.data(), dout, LAT * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(ddst + i, &ent_dst[(size_t) i], 4, cudaMemcpyHostToDevice);
                if (std::memcmp(one.data(), got.data() + (size_t) i * LAT, LAT * 4)) alone = false;
            }
        cudaMemcpy(dptr, ptr.data(), NE * 8, cudaMemcpyHostToDevice);
        cudaMemcpy(dstart, start.data(), (NE + 1) * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dn, &ng, 4, cudaMemcpyHostToDevice);
        K::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, NE, n_ent, dxq, dscr, dout, s, 2);
        cudaMemcpy(again.data(), dout, again.size() * 4, cudaMemcpyDeviceToHost);
        if (std::memcmp(again.data(), got.data(), again.size() * 4)) strided = false;
        const bool ok2 = alone && strided && cudaGetLastError() == cudaSuccess;
        std::printf("gpu   an entry alone = in the batch %s, group stride 2 = one row per group %s  %s\n", alone ? "yes" : "NO",
                    strided ? "yes" : "NO", ok2 ? "ok" : "FAIL");
        if (!ok2) ++failures;
    }
    // CPU against GPU: the two activation roundings
    {
        std::vector<std::vector<uint8_t>> act(NT, std::vector<uint8_t>(f.act_bytes));
        double worst = 0;
        for (int e = 0; e < NE; ++e)
            for (int i = start[(size_t) e]; i < start[(size_t) e + 1]; ++i) {
                const int t = ent_tok[(size_t) i];
                cpu::native_quant_act(f, x[(size_t) t].data(), act[(size_t) t].data());
                std::vector<float> ff(FF), o(LAT);
                std::vector<uint8_t> h(f.h_bytes);
                float* fp = ff.data();
                float* op = o.data();
                const void* ap = act[(size_t) t].data();
                cpu::native_gu_rows(f, ex[(size_t) e].blob.data(), &ap, 1, &fp, 0, (int) FF);
                cpu::native_quant_h(f, ff.data(), h.data());
                const void* hp = h.data();
                cpu::native_down_rows(f, ex[(size_t) e].blob.data(), &hp, 1, &op, 0, (int) LAT);
                worst = std::max(worst, rel(std::vector<float>(got.begin() + (ptrdiff_t) i * LAT, got.begin() + (ptrdiff_t) (i + 1) * LAT), o));
            }
        const bool ok3 = worst < 3e-2;
        std::printf("gpu   cpu vs gpu %.2e (q8_0 vs q8_1 activations)  %s\n", worst, ok3 ? "ok" : "FAIL");
        if (!ok3) ++failures;
    }
    // the microbench: a decode window's call (8 tokens, 6 experts, the entries above), the blobs' bytes it must read
    {
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        for (int i = 0; i < 20; ++i) K::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, NE, n_ent, dxq, dscr, dout, s);
        constexpr int R = 200;
        cudaEventRecord(a, s);
        for (int i = 0; i < R; ++i) K::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, NE, n_ent, dxq, dscr, dout, s);
        cudaEventRecord(b, s);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        const double us = 1000.0 * ms / R, gbs = (double) NE * (double) f.bytes / (us * 1e3);
        std::printf("bench relu2 grouped: %d experts x %.2f MB, %d entries: %.1f us per call, %.0f GB/s of expert weights\n",
                    NE, f.bytes / 1e6, n_ent, us, gbs);
        // the SwiGLU kernels at the same width (the input rows twice, as gate and up) for comparison
        const K::NativeExpertLayout G = K::native_expert_layout(40, 40, LAT, FF);
        uint8_t* gblobs = nullptr;
        cudaMalloc((void**) &gblobs, (size_t) NE * G.bytes);
        std::vector<unsigned long long> gptr;
        for (int e = 0; e < NE; ++e) {
            uint8_t* gb = gblobs + (size_t) e * G.bytes;
            const std::vector<uint8_t>& src = ex[(size_t) e].blob;
            cudaMemcpy(gb, src.data(), L.up_off, cudaMemcpyHostToDevice);
            cudaMemcpy(gb + G.up_off, src.data(), L.up_off, cudaMemcpyHostToDevice);
            cudaMemcpy(gb + G.down_off, src.data() + L.down_off, f.bytes - L.down_off, cudaMemcpyHostToDevice);
            gptr.push_back((unsigned long long) gb);
        }
        cudaMemcpy(dptr, gptr.data(), NE * 8, cudaMemcpyHostToDevice);
        uint8_t* gscr = nullptr;
        cudaMalloc((void**) &gscr, K::native_expert_scratch_bytes(n_ent, FF));
        for (int i = 0; i < 20; ++i) K::native_expert_grouped(G, dptr, dstart, dn, ddst, dtok, NE, n_ent, dxq, gscr, dout, s);
        cudaEventRecord(a, s);
        for (int i = 0; i < R; ++i) K::native_expert_grouped(G, dptr, dstart, dn, ddst, dtok, NE, n_ent, dxq, gscr, dout, s);
        cudaEventRecord(b, s);
        cudaEventSynchronize(b);
        cudaEventElapsedTime(&ms, a, b);
        const double gus = 1000.0 * ms / R;
        std::printf("bench swiglu grouped (same width, for comparison): %.2f MB, %.1f us per call, %.0f GB/s\n", G.bytes / 1e6,
                    gus, (double) NE * (double) G.bytes / (gus * 1e3));
        cudaMemcpy(dptr, ptr.data(), NE * 8, cudaMemcpyHostToDevice);
        cudaFree(gscr);
        cudaFree(gblobs);
        cudaEventDestroy(a);
        cudaEventDestroy(b);
    }
    for (void* p : {(void*) dblobs, (void*) dx, (void*) dout, (void*) dxq, (void*) dscr, (void*) dptr, (void*) dstart, (void*) dn,
                    (void*) dtok, (void*) ddst})
        cudaFree(p);
    cudaStreamDestroy(s);
    return failures;
}

}  // namespace

int main(int argc, char** argv) {
    const bool cpu_only = argc > 1 && std::string(argv[1]) == "--cpu";
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt_relu2(40, 40, LAT, FF, f, err)) {
        std::printf("native_fmt_relu2: %s  FAIL\n", err.c_str());
        return 1;
    }
    if (!f.relu2 || f.up_row != 0 || f.down_off != f.up_off || f.bytes != f.down_off + f.d_row * LAT + 16) {
        std::printf("the relu2 format's offsets are wrong  FAIL\n");
        return 1;
    }
    {   // a wider intermediate than the pool's buffers is refused
        cpu::NativeFmt g;
        std::string e2;
        if (cpu::native_fmt_relu2(40, 40, LAT, cpu::kNativeMaxFF + 64, g, e2)) {
            std::printf("an intermediate of %d was taken past the pool's %d  FAIL\n", cpu::kNativeMaxFF + 64, cpu::kNativeMaxFF);
            return 1;
        }
    }
    const auto* tt = ggml_get_type_traits(GGML_TYPE_NVFP4);
    std::vector<Expert> ex(NE);
    for (int e = 0; e < NE; ++e) {
        Expert& E = ex[(size_t) e];
        E.blob = make_blob(f, 100 + e, E.s_in, E.s_d);
        E.in.resize((size_t) (FF * LAT));
        E.down.resize((size_t) (LAT * FF));
        for (int64_t r = 0; r < FF; ++r) tt->to_float(E.blob.data() + r * f.gu_row, E.in.data() + r * LAT, LAT);
        for (int64_t r = 0; r < LAT; ++r) tt->to_float(E.blob.data() + f.down_off + r * f.d_row, E.down.data() + r * FF, FF);
    }
    std::mt19937 rng(9);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<std::vector<float>> x(NT, std::vector<float>(LAT));
    for (int t = 0; t < NT; ++t)
        for (auto& v : x[(size_t) t]) v = nd(rng) * (t == 5 ? 20.f : 1.f);
    int failures = check_cpu(f, ex, x);
    if (!cpu_only) {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
            std::printf("gpu   no CUDA device  FAIL\n");
            return 1;
        }
        failures += check_gpu(f, ex, x);
    }
    std::printf("%s\n", failures ? "FAILED" : "all ok");
    return failures ? 1 : 0;
}
