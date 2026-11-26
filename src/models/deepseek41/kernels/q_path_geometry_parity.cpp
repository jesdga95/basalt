// Query-path geometry parity at DeepSeek-V4 Flash's 4096-wide, rank-1024 attention shape.
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/models/deepseek41/attn.hpp"
#include "test_util.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>

using namespace basalt::models::deepseek41;
using dstest::Dev;
namespace K = basalt::kernels;

namespace {

constexpr int kType = 14, kBsz = 210;

void put_half(uint8_t* p, float v) {
    const __half h = __float2half(v);
    std::memcpy(p, &h, 2);
}

bool check_geometry(int n_embd, int q_rank) {
    dstest::Verdict v;
    cudaStream_t st;
    dstest::check(cudaStreamCreate(&st), "stream");
    const int blocks = n_embd / 256, T = kAttnMaxTokens;
    std::vector<uint8_t> w((size_t) q_rank * blocks * kBsz);
    std::mt19937 g(99 + n_embd);
    for (auto& b : w) b = (uint8_t) g();
    for (size_t i = 0; i < (size_t) q_rank * blocks; ++i)
        put_half(&w[i * kBsz + 208], 0.003f + 0.001f * (float) (g() % 7));
    const std::vector<float> norm = dstest::randn(q_rank, 5, 0.3f, 1.f);
    Dev<uint8_t> dw(w.size());
    dw.up(w);
    Dev<float> dn(norm);
    const std::vector<float> x = dstest::randn((size_t) T * n_embd, 17, 2.f);
    Dev<float> dx(x), qr((size_t) T * q_rank), raw((size_t) T * q_rank), qr1((size_t) T * q_rank);
    const size_t qb = q_path_scratch_bytes(1, q_rank);
    Dev<uint8_t> xq(K::native_q8_1_bytes(n_embd, T)), q8(q_path_scratch_bytes(T, q_rank)),
        q8_ref(q_path_scratch_bytes(T, q_rank)), q81(q_path_scratch_bytes(T, q_rank));
    K::native_quantize_q8_1(dx.p, xq.p, n_embd, T, st);
    QPathWeights qw{dw.p, kType, dn.p, nullptr, 0, 1e-6f};
    qw.n_embd = n_embd;
    qw.q_rank = q_rank;
    q_latent(xq.p, qw, T, qr.p, q8.p, st);
    K::native_mmvq(kType, dw.p, xq.p, raw.p, n_embd, q_rank, T, st);
    K::native_quantize_q8_1(qr.p, q8_ref.p, q_rank, T, st);
    const size_t xb = K::native_q8_1_bytes(n_embd, 1);
    for (int t = 0; t < T; ++t)
        q_latent(xq.p + t * xb, qw, 1, qr1.p + (size_t) t * q_rank, q81.p + t * qb, st);
    dstest::check(cudaStreamSynchronize(st), "q_latent");
    const auto got = qr.down(), r = raw.down(), got1 = qr1.down();
    const auto b = q8.down(), bref = q8_ref.down(), b1 = q81.down();
    double se = 0, sr = 0, worst = 0;
    for (int t = 0; t < T; ++t) {
        double ss = 0;
        for (int i = 0; i < q_rank; ++i) ss += (double) r[(size_t) t * q_rank + i] * r[(size_t) t * q_rank + i];
        const double scale = 1.0 / std::sqrt(ss / q_rank + 1e-6);
        for (int i = 0; i < q_rank; ++i) {
            const double want = r[(size_t) t * q_rank + i] * scale * norm[i];
            const double d = got[(size_t) t * q_rank + i] - want;
            se += d * d;
            sr += want * want;
            worst = std::max(worst, std::fabs(d));
        }
    }
    const bool width = got == got1 && b == b1;
    std::printf("q_latent geometry=%d/%d: CPU norm rel RMS %.3g max abs %.3g; q8_1 parity %s; token-width independent %s\n",
                n_embd, q_rank, std::sqrt(se / sr), worst, b == bref ? "bitwise" : "DIFFER",
                width ? "yes" : "NO");
    v.expect(std::sqrt(se / sr) < 1e-6, "latent vs CPU norm");
    v.expect(b == bref, "q8_1 rows equal native_quantize_q8_1");
    v.expect(width, "token independent of call width");
    cudaStreamDestroy(st);
    return v.failures == 0;
}

}  // namespace

int main() { return check_geometry(4096, 1024) ? 0 : 1; }
