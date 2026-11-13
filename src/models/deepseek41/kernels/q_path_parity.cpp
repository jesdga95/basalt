// Query-path parity: q_latent's normed latent against a double-precision CPU norm of the same projection, its fused
// q8_1 rows against native_quantize_q8_1 of the latent (byte for byte), each token independent of the call's width.
// Prints a microbench line.
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/models/deepseek41/attn.hpp"
#include "test_util.hpp"

#include <cuda_fp16.h>

#include <random>

using namespace basalt::models::deepseek41;
using dstest::Dev;
namespace K = basalt::kernels;

namespace {

constexpr int kType = 14, kBsz = 210, kBlocks = kAttnEmbd / 256;   // Q6_K rows of wq_a

void put_half(uint8_t* p, float v) {
    const __half h = __float2half(v);
    std::memcpy(p, &h, 2);
}

}  // namespace

int main() {
    dstest::Verdict v;
    cudaStream_t st;
    dstest::check(cudaStreamCreate(&st), "stream");
    std::vector<uint8_t> w((size_t) kAttnQRank * kBlocks * kBsz);
    std::mt19937 g(99);
    for (auto& b : w) b = (uint8_t) g();
    for (size_t i = 0; i < (size_t) kAttnQRank * kBlocks; ++i) put_half(&w[i * kBsz + 208], 0.003f + 0.001f * (float) (g() % 7));
    std::vector<float> norm = dstest::randn(kAttnQRank, 5, 0.3f, 1.f);
    Dev<uint8_t> dw(w.size());
    dw.up(w);
    Dev<float> dn(norm);
    const int TM = kAttnMaxTokens;
    const std::vector<float> x = dstest::randn((size_t) TM * kAttnEmbd, 17, 2.f);
    Dev<float> dx(x), qr((size_t) TM * kAttnQRank), raw((size_t) TM * kAttnQRank), qr1((size_t) TM * kAttnQRank);
    const size_t qb = q_path_scratch_bytes(1);
    Dev<uint8_t> xq(K::native_q8_1_bytes(kAttnEmbd, TM)), q8(q_path_scratch_bytes(TM)), q8_ref(q_path_scratch_bytes(TM)), q81(q_path_scratch_bytes(TM));
    K::native_quantize_q8_1(dx.p, xq.p, kAttnEmbd, TM, st);
    QPathWeights qw{dw.p, kType, dn.p, nullptr, 0, 1e-6f};
    q_latent(xq.p, qw, TM, qr.p, q8.p, st);
    K::native_mmvq(kType, dw.p, xq.p, raw.p, kAttnEmbd, kAttnQRank, TM, st);
    K::native_quantize_q8_1(qr.p, q8_ref.p, kAttnQRank, TM, st);
    const size_t xb = K::native_q8_1_bytes(kAttnEmbd, 1);
    for (int t = 0; t < TM; ++t)
        q_latent(xq.p + t * xb, qw, 1, qr1.p + (size_t) t * kAttnQRank, q81.p + t * qb, st);
    dstest::check(cudaStreamSynchronize(st), "q_latent");
    const auto got = qr.down(), r = raw.down(), got1 = qr1.down();
    const auto b = q8.down(), bref = q8_ref.down(), b1 = q81.down();
    double se = 0, sr = 0, worst = 0;
    for (int t = 0; t < TM; ++t) {
        double ss = 0;
        for (int i = 0; i < kAttnQRank; ++i) ss += (double) r[(size_t) t * kAttnQRank + i] * r[(size_t) t * kAttnQRank + i];
        const double s = 1.0 / std::sqrt(ss / kAttnQRank + 1e-6);
        for (int i = 0; i < kAttnQRank; ++i) {
            const double want = r[(size_t) t * kAttnQRank + i] * s * norm[i], d = got[(size_t) t * kAttnQRank + i] - want;
            se += d * d;
            sr += want * want;
            worst = std::max(worst, std::fabs(d));
        }
    }
    const bool q_same = b == bref, width = got == got1 && b == b1;
    std::printf("q_latent T=%d: latent vs CPU norm rel RMS %.3g max abs %.3g; q8_1 rows vs native_quantize_q8_1 %s; "
                "independent of the call's width %s\n", TM, std::sqrt(se / sr), worst, q_same ? "bitwise" : "DIFFER",
                width ? "yes" : "NO");
    v.expect(std::sqrt(se / sr) < 1e-6, "latent vs CPU norm");
    v.expect(q_same, "q8_1 rows equal native_quantize_q8_1 bitwise");
    v.expect(width, "token independent of the call's width");
    for (int T : {1, 3}) {
        const double us = dstest::bench_stream_us([&](void* s) { q_latent(xq.p, qw, T, qr.p, q8.p, s); }, 300);
        char name[64];
        std::snprintf(name, sizeof name, "q_latent T=%d", T);
        dstest::bench_line(name, us, (double) kAttnQRank * kBlocks * kBsz + (double) T * kAttnQRank * 4 * 2);
    }
    std::printf(v.failures ? "FAILED (%d)\n" : "PASS\n", v.failures);
    return v.failures ? 1 : 0;
}
