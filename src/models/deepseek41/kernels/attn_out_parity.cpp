// Grouped output parity: attn_out's wo_a stage as one launch against one native single-column MMVQ launch per
// (token, group) (bitwise, with an output hash) and against a double-precision CPU reference on dequantised weights.
// Also the token's independence of the call's width. Prints a microbench line.
#include "basalt/artifact/dequant.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/models/deepseek41/attn.hpp"
#include "attn_out_grouped.hpp"
#include "test_util.hpp"

#include <cuda_fp16.h>

using namespace basalt::models::deepseek41;
using dstest::Dev;
namespace K = basalt::kernels;

namespace {

constexpr int kRows = kAttnGroups * kAttnOutRank, kBlocks = kAttnGroupIn / 256;

void put_half(uint8_t* p, float v) {
    const __half h = __float2half(v);
    std::memcpy(p, &h, 2);
}

uint64_t fnv(const std::vector<float>& v) {
    uint64_t h = 1469598103934665603ull;
    for (float f : v) { uint32_t u; std::memcpy(&u, &f, 4); h = (h ^ u) * 1099511628211ull; }
    return h;
}

cudaStream_t g_stream;

int run(int type, int T, bool bench, dstest::Verdict& v) {
    const size_t bsz = type == 14 ? 210 : 176;
    std::vector<uint8_t> w((size_t) kRows * kBlocks * bsz);
    std::mt19937 g(1234 + type);
    for (auto& b : w) b = (uint8_t) g();
    for (size_t i = 0; i < (size_t) kRows * kBlocks; ++i) {
        uint8_t* b = &w[i * bsz];
        if (type == 14) put_half(b + 208, 0.003f + 0.001f * (float) (g() % 7));
        else { put_half(b, 0.004f); put_half(b + 2, 0.002f); }
    }
    const std::vector<float> attn = dstest::randn((size_t) T * kAttnHeads * kAttnHeadDim, 77 + T);
    Dev<uint8_t> dw(w.size());
    dw.up(w);
    Dev<float> da(attn);
    const size_t xq_bytes = attn_out_scratch_bytes(T);
    Dev<uint8_t> xq(xq_bytes);
    Dev<float> oa_new((size_t) T * kAttnGroups * kAttnOutRank), oa_ref((size_t) T * kAttnGroups * kAttnOutRank);
    K::native_quantize_q8_1(da.p, xq.p, kAttnHeads * kAttnHeadDim, T, g_stream);
    if (!wo_a_grouped(type, dw.p, xq.p, oa_new.p, T, nullptr)) { std::printf("FAIL: type unsupported\n"); return 1; }
    const size_t col_q8 = K::native_q8_1_bytes(kAttnGroupIn, 1);
    const size_t group_w = K::native_mmvq_weight_bytes(type, kAttnGroupIn, kAttnOutRank);
    for (int t = 0; t < T; ++t)
        for (int gi = 0; gi < kAttnGroups; ++gi)
            K::native_mmvq(type, (const char*) dw.p + (size_t) gi * group_w, (const char*) xq.p + (size_t) (t * kAttnGroups + gi) * col_q8,
                           oa_ref.p + (size_t) t * kAttnGroups * kAttnOutRank + (size_t) gi * kAttnOutRank, kAttnGroupIn, kAttnOutRank, 1, g_stream);
    dstest::check(cudaStreamSynchronize(g_stream), "sync");
    const auto got = oa_new.down(), ref = oa_ref.down();
    const bool same = std::memcmp(got.data(), ref.data(), got.size() * 4) == 0;
    std::printf("type %d T=%d: bitwise vs native mmvq %s, hash %016llx\n", type, T, same ? "yes" : "NO", (unsigned long long) fnv(got));
    v.expect(same, "grouped wo_a equals native mmvq bitwise");
    // CPU reference on a row subset, unquantised activations
    double se = 0, sr = 0;
    std::vector<float> wrow(256);
    for (int t = 0; t < T; ++t)
        for (int gi = 0; gi < kAttnGroups; ++gi)
            for (int r = 0; r < kAttnOutRank; r += 61) {
                double acc = 0;
                for (int kb = 0; kb < kBlocks; ++kb) {
                    const uint8_t* b = &w[((size_t) (gi * kAttnOutRank + r) * kBlocks + kb) * bsz];
                    if (type == 14) basalt::dequantize_q6_K(b, wrow.data()); else basalt::dequantize_q5_K(b, wrow.data());
                    for (int e = 0; e < 256; ++e) acc += (double) wrow[e] * attn[(size_t) (t * kAttnGroups + gi) * kAttnGroupIn + kb * 256 + e];
                }
                const double o = got[(size_t) t * kAttnGroups * kAttnOutRank + (size_t) gi * kAttnOutRank + r];
                se += (o - acc) * (o - acc);
                sr += acc * acc;
            }
    const double rel = std::sqrt(se / sr);
    std::printf("  CPU reference rel RMS %.4f (q8_1 activation)\n", rel);
    v.expect(rel < 0.03, "CPU reference");
    if (bench && T == 1) {
        const double us_new = dstest::bench_stream_us([&](void* s) { wo_a_grouped(type, dw.p, xq.p, oa_new.p, T, s); });
        const double us_old = dstest::bench_stream_us([&](void* s) {
            for (int gi = 0; gi < kAttnGroups; ++gi)
                K::native_mmvq(type, (const char*) dw.p + (size_t) gi * group_w, (const char*) xq.p + (size_t) gi * col_q8,
                               oa_ref.p + (size_t) gi * kAttnOutRank, kAttnGroupIn, kAttnOutRank, 1, s);
        });
        const double bytes = (double) w.size();
        dstest::bench_line(type == 14 ? "wo_a grouped q6_k T=1" : "wo_a grouped q5_k T=1", us_new, bytes);
        dstest::bench_line(type == 14 ? "wo_a 8x native q6_k T=1" : "wo_a 8x native q5_k T=1", us_old, bytes);
    }
    return 0;
}

}  // namespace

int main() {
    dstest::check(cudaStreamCreate(&g_stream), "stream");
    dstest::Verdict v;
    for (int type : {14, 13})
        for (int T : {1, 2, 4}) run(type, T, true, v);
    std::printf(v.failures ? "FAILED\n" : "PASS\n");
    return v.failures ? 1 : 0;
}
