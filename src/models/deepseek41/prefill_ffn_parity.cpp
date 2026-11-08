// src/models/deepseek41/prefill_ffn_parity.cpp - the prompt path's routed experts in NVFP4 at the real geometry (hidden
// 5120, FF 2304, SwiGLU limit 10): experts staged staged_stride apart as prefill_ffn stages them, the MMQ gate/up product,
// swiglu_split_clamped with the experts' scale tails, the MMQ down product. Every row against a CPU double reference
// (dequantized weights times the tail, the unrounded activation) and against the decode path's grouped kernel on the
// same blobs; one expert of the launch takes no rows. Without tails swiglu_split_clamped is the plain clamped SwiGLU
// (checked against a CPU float reference). Prints microbench lines.
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/prefill/moe_mmq.hpp"
#include "models/deepseek41/kernels/test_util.hpp"
#include "models/deepseek41/pack.hpp"
#include "models/deepseek41/prefill_ffn.hpp"
#include "models/deepseek41/prompt_host.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace basalt::models::deepseek41;
using dstest::check;
using dstest::Dev;
namespace mmq = basalt::prefill::mmq;
namespace kg = basalt::kernels;

namespace {

constexpr int H = 5120, FF = 2304, T = 8, NB = 5;
constexpr float kLimit = 10.0f;

struct Expert {
    std::vector<uint8_t> blob;
    std::vector<float> wg, wu, wd;   // dequantized, times the tail's scales
};

std::vector<uint8_t> nvfp4_rows(int64_t rows, int64_t n, float sd, std::mt19937& rng) {
    const size_t rb = ggml_row_size(GGML_TYPE_NVFP4, n);
    std::vector<uint8_t> w((size_t) rows * rb);
    std::vector<float> x((size_t) n);
    std::normal_distribution<float> nd(0.f, sd);
    for (int64_t r = 0; r < rows; ++r) {
        for (auto& v : x) v = nd(rng);
        ggml_get_type_traits(GGML_TYPE_NVFP4)->from_float_ref(x.data(), w.data() + (size_t) r * rb, n);
    }
    return w;
}

std::vector<float> deq(const uint8_t* w, int64_t rows, int64_t n, float s) {
    std::vector<float> o((size_t) rows * n);
    const size_t rb = ggml_row_size(GGML_TYPE_NVFP4, n);
    for (int64_t r = 0; r < rows; ++r) ggml_get_type_traits(GGML_TYPE_NVFP4)->to_float(w + (size_t) r * rb, o.data() + (size_t) r * n, n);
    for (float& v : o) v *= s;
    return o;
}

Expert make_expert(std::mt19937& rng) {
    std::uniform_real_distribution<float> ts(0.9f, 1.6f);
    const float t4[4] = {ts(rng), ts(rng), 0.02f * ts(rng), 0.f};
    auto g = nvfp4_rows(FF, H, 0.1f, rng), u = nvfp4_rows(FF, H, 0.1f, rng), d = nvfp4_rows(H, FF, 0.1f, rng);
    Expert e;
    e.wg = deq(g.data(), FF, H, t4[0]);
    e.wu = deq(u.data(), FF, H, t4[1]);
    e.wd = deq(d.data(), H, FF, t4[2]);
    e.blob = g;
    e.blob.insert(e.blob.end(), u.begin(), u.end());
    e.blob.insert(e.blob.end(), d.begin(), d.end());
    e.blob.insert(e.blob.end(), (const uint8_t*) t4, (const uint8_t*) t4 + 16);
    return e;
}

std::vector<double> ref_expert(const Expert& e, const float* x) {
    std::vector<double> h(FF), out(H);
    for (int r = 0; r < FF; ++r) {
        double g = 0, u = 0;
        for (int k = 0; k < H; ++k) {
            g += (double) e.wg[(size_t) r * H + k] * x[k];
            u += (double) e.wu[(size_t) r * H + k] * x[k];
        }
        g = std::min(g, (double) kLimit);
        u = std::max(std::min(u, (double) kLimit), -(double) kLimit);
        h[r] = g / (1.0 + std::exp(-g)) * u;
    }
    for (int r = 0; r < H; ++r) {
        double s = 0;
        for (int k = 0; k < FF; ++k) s += (double) e.wd[(size_t) r * FF + k] * h[k];
        out[r] = s;
    }
    return out;
}

}  // namespace

int main() {
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) return 77;
    dstest::Verdict v;
    cudaStream_t st;
    check(cudaStreamCreate(&st), "stream");
    std::mt19937 rng(41);

    ExpertLayer el;
    el.gu_type = el.d_type = GGML_TYPE_NVFP4;
    const kg::NativeExpertLayout L = kg::native_expert_layout(el.gu_type, el.d_type, H, FF);
    el.blob_bytes = L.bytes;
    const size_t stride = staged_stride(el);
    v.expect(stride % 36 == 0 && stride >= L.bytes, "the staging stride steps in whole NVFP4 blocks");
    std::vector<Expert> ex;
    for (int b = 0; b < NB; ++b) ex.push_back(make_expert(rng));
    std::vector<uint8_t> staged(stride * NB, 0);
    for (int b = 0; b < NB; ++b) std::memcpy(staged.data() + stride * b, ex[(size_t) b].blob.data(), L.bytes);
    Dev<uint8_t> blobs(staged);

    // rows per expert (expert 2 takes none), tokens in a mixed order
    const int rows_of[NB] = {3, 5, 0, 1, 7};
    std::vector<int32_t> tok, bounds{0};
    for (int b = 0; b < NB; ++b) {
        for (int i = 0; i < rows_of[b]; ++i) tok.push_back((int32_t) ((b * 3 + i * 5) % T));
        bounds.push_back((int32_t) tok.size());
    }
    const int rows = (int) tok.size();
    const int max_rows = *std::max_element(rows_of, rows_of + NB);
    const std::vector<float> x = dstest::randn((size_t) T * H, 7);
    Dev<float> dx(x), gu((size_t) rows * 2 * FF), h((size_t) rows * FF), out((size_t) rows * H);
    Dev<int32_t> dtok(tok), dbounds(bounds);
    std::vector<int32_t> ident_h((size_t) rows);
    for (int i = 0; i < rows; ++i) ident_h[(size_t) i] = i;
    Dev<int32_t> ident(ident_h);
    Dev<uint8_t> xq1(mmq::q8_bytes(rows, H)), xq2(mmq::q8_bytes(rows, FF));
    mmq::Context ctx;
    mmq::load(el.gu_type, 2 * FF);
    mmq::load(el.d_type, H);

    // prefill_ffn's sequence
    auto chain = [&](void* s) {
        mmq::quantize(dx.p, dtok.p, xq1.p, el.gu_type, H, H, rows, s);
        mmq::Product p;
        p.w = blobs.p;
        p.type = el.gu_type;
        p.w_rows = 2 * FF;
        p.w_cols = H;
        p.expert_bytes = stride;
        p.n = NB;
        p.xq = xq1.p;
        p.bounds = dbounds.p;
        p.ids = ident.p;
        p.total_rows = rows;
        p.max_rows = max_rows;
        p.dst = gu.p;
        p.ld_dst = p.w_rows;
        ctx.run(p, s);
        ExpertTails tails;
        tails.blobs = blobs.p;
        tails.stride = stride;
        tails.tail_off = L.bytes - 16;
        tails.bounds = dbounds.p;
        tails.n = NB;
        swiglu_split_clamped(gu.p, h.p, rows, FF, kLimit, s, &tails);
        mmq::quantize(h.p, nullptr, xq2.p, el.d_type, FF, FF, rows, s);
        mmq::Product d = p;
        d.w = blobs.p + L.down_off;
        d.type = el.d_type;
        d.w_rows = H;
        d.w_cols = FF;
        d.xq = xq2.p;
        d.dst = out.p;
        d.ld_dst = H;
        ctx.run(d, s);
    };
    chain(st);
    check(cudaStreamSynchronize(st), "chain");
    const std::vector<float> got = out.down();

    // the decode path's grouped kernel on the same blobs and tokens
    std::vector<unsigned long long> gptr;
    std::vector<int32_t> gstart{0}, gdst, gtok;
    for (int b = 0; b < NB; ++b) {
        if (!rows_of[b]) continue;
        gptr.push_back((unsigned long long) (blobs.p + stride * b));
        for (int i = bounds[(size_t) b]; i < bounds[(size_t) b + 1]; ++i) {
            gdst.push_back(i);
            gtok.push_back(tok[(size_t) i]);
        }
        gstart.push_back((int32_t) gdst.size());
    }
    const int32_t ng = (int32_t) gptr.size();
    Dev<unsigned long long> dptr(gptr);
    Dev<int32_t> dstart(gstart), ddst(gdst), dgtok(gtok), dng(std::vector<int32_t>{ng});
    Dev<uint8_t> xq(kg::native_q8_1_bytes(H, T)), scr(kg::native_expert_scratch_bytes(rows, FF));
    Dev<float> dout((size_t) rows * H);
    kg::quantize_q8_1_rows(dx.p, T, H, xq.p, st);
    kg::native_expert_grouped_clamped(L, kLimit, dptr.p, dstart.p, dng.p, ddst.p, dgtok.p, ng, rows, xq.p, scr.p, dout.p, st);
    check(cudaStreamSynchronize(st), "decode");
    const std::vector<float> dec = dout.down();

    std::vector<double> ref((size_t) rows * H);
    for (int b = 0; b < NB; ++b)
        for (int i = bounds[(size_t) b]; i < bounds[(size_t) b + 1]; ++i) {
            const std::vector<double> r = ref_expert(ex[(size_t) b], x.data() + (size_t) tok[(size_t) i] * H);
            std::copy(r.begin(), r.end(), ref.begin() + (size_t) i * H);
        }
    const dstest::Err e_ref = dstest::compare(got, ref), e_dec = dstest::compare(got, dec), d_ref = dstest::compare(dec, ref);
    std::printf("NVFP4 prompt experts (%d rows, %d experts, one without rows) vs double: rel RMS %.2e max abs %.2e max rel %.2e\n",
                rows, NB, e_ref.rms_rel, e_ref.max_abs, e_ref.max_rel);
    std::printf("NVFP4 prompt experts vs the decode grouped kernel: rel RMS %.2e max abs %.2e; decode vs double: rel RMS %.2e\n",
                e_dec.rms_rel, e_dec.max_abs, d_ref.rms_rel);
    v.expect(e_ref.rms_rel < 2.5e-2, "prompt NVFP4 experts against the double reference");
    v.expect(e_dec.rms_rel < 3e-2, "prompt NVFP4 experts against the decode path");
    v.expect(std::all_of(got.begin(), got.end(), [](float f) { return std::isfinite(f); }), "finite outputs");

    // the limit binds: the tail-scaled gate/up of some rows pass it
    {
        const std::vector<float> g = gu.down();
        size_t over = 0;
        for (int b = 0; b < NB; ++b) {
            float sc[2];
            std::memcpy(sc, ex[(size_t) b].blob.data() + L.bytes - 16, 8);
            for (int i = bounds[(size_t) b]; i < bounds[(size_t) b + 1]; ++i)
                for (int k = 0; k < FF; ++k)
                    over += g[(size_t) i * 2 * FF + k] * sc[0] > kLimit || std::fabs(g[(size_t) i * 2 * FF + FF + k] * sc[1]) > kLimit;
        }
        std::printf("values past the SwiGLU limit: %zu of %d\n", over, rows * FF);
        v.expect(over > 0, "the SwiGLU limit binds in the test");
    }

    // without tails: the plain clamped SwiGLU, against a float CPU reference
    {
        const std::vector<float> gin = dstest::randn((size_t) rows * 2 * FF, 9, 6.f);
        Dev<float> dgu(gin), dh((size_t) rows * FF);
        swiglu_split_clamped(dgu.p, dh.p, rows, FF, kLimit, st);
        check(cudaStreamSynchronize(st), "swiglu");
        const std::vector<float> hg = dh.down();
        std::vector<double> hr((size_t) rows * FF);
        for (int r = 0; r < rows; ++r)
            for (int k = 0; k < FF; ++k) {
                const double g = std::min((double) gin[(size_t) r * 2 * FF + k], (double) kLimit);
                const double u = std::max(std::min((double) gin[(size_t) r * 2 * FF + FF + k], (double) kLimit), -(double) kLimit);
                hr[(size_t) r * FF + k] = g / (1.0 + std::exp(-g)) * u;
            }
        const dstest::Err e = dstest::compare(hg, hr);
        std::printf("clamped SwiGLU without tails vs double: max rel %.2e\n", e.max_rel);
        v.expect(e.max_rel < 1e-5, "clamped SwiGLU without tails");
    }

    dstest::bench_line("NVFP4 prompt chain, 16 rows", dstest::bench_stream_us(chain, 50),
                       (double) (NB - 1) * (double) L.bytes);
    {
        ExpertTails tails;
        tails.blobs = blobs.p;
        tails.stride = stride;
        tails.tail_off = L.bytes - 16;
        tails.bounds = dbounds.p;
        tails.n = NB;
        dstest::bench_line("swiglu with tails, 16 rows", dstest::bench_stream_us([&](void* s) {
            swiglu_split_clamped(gu.p, h.p, rows, FF, kLimit, s, &tails);
        }), (double) rows * 3 * FF * 4);
    }
    cudaStreamDestroy(st);
    std::printf(v.failures ? "FAILED\n" : "PASS\n");
    return v.failures ? 1 : 0;
}
