// Optional per-head query RMS normalization and forward RoPE against a CPU double reference.
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/sparse_mqa.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

template <class T> struct Device {
    T* p = nullptr;
    explicit Device(size_t n) { if (n) check(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc"); }
    ~Device() { if (p) cudaFree(p); }
    void upload(const std::vector<T>& v) { check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload"); }
    std::vector<T> download(size_t n) const {
        std::vector<T> v(n);
        check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
        return v;
    }
};

struct Q80 { __half d; int8_t qs[32]; };
static_assert(sizeof(Q80) == 34);

std::vector<uint8_t> weights_q8_0(int rows, int cols) {
    std::mt19937 rng(0x51484541u);
    const size_t blocks = (size_t) rows * cols / 32;
    std::vector<uint8_t> w(blocks * sizeof(Q80));
    const __half d = __float2half(0.002f);
    for (size_t i = 0; i < blocks; ++i) {
        uint8_t* b = w.data() + i * sizeof(Q80);
        std::memcpy(b, &d, sizeof(d));
        for (int j = 0; j < 32; ++j) b[sizeof(d) + j] = (uint8_t) (int8_t) ((int) (rng() % 31) - 15);
    }
    return w;
}

}  // namespace

int main() {
    using namespace basalt::kernels;
    constexpr int T = 3, rank = 1024, n_out = kSparseMqaHeads * kSparseMqaDim, type = 8;
    constexpr float eps = 1e-20f;
    const size_t wbytes = native_mmvq_weight_bytes(type, rank, n_out);
    auto weights = weights_q8_0(n_out, rank);
    if (weights.size() != wbytes) throw std::runtime_error("Q8_0 weight byte count");
    std::mt19937 rng(0x51524f50u);
    std::vector<float> qr((size_t) T * rank);
    for (float& v : qr) v = ((int) (rng() % 2001) - 1000) * 0.001f;
    std::vector<int32_t> pos{0, 17, 129};
    MqaRopeTable rope{};
    for (int i = 0; i < kSparseMqaRopeDim / 2; ++i) rope.step[i] = 0.015f / (i + 1.f);
    rope.mscale = 1.f;

    Device<uint8_t> dw(weights.size()), dq8(native_q8_1_bytes(rank, T));
    Device<float> dqr(qr.size()), got((size_t) T * n_out), raw((size_t) T * n_out), single((size_t) T * n_out);
    Device<int32_t> dpos(pos.size());
    dw.upload(weights); dqr.upload(qr); dpos.upload(pos);
    cudaStream_t stream{};
    check(cudaStreamCreate(&stream), "stream");
    native_quantize_q8_1(dqr.p, dq8.p, rank, T, stream);
    SparseMqaQWeights qweights{};
    qweights.wq_b = dw.p;
    qweights.wq_b_type = type;
    qweights.q_rank = rank;
    qweights.head_rms_eps = eps;

    const bool exact = native_mmvq_multi_exact();
    native_mmvq_set_multi_exact(true);
    sparse_mqa_q_heads(qweights, T, dpos.p, rope, dq8.p, got.p, stream);
    native_mmvq(type, dw.p, dq8.p, raw.p, rank, n_out, T, stream);
    const size_t one_q8 = native_q8_1_bytes(rank, 1);
    for (int t = 0; t < T; ++t)
        sparse_mqa_q_heads(qweights, 1, dpos.p + t, rope, (const uint8_t*) dq8.p + (size_t) t * one_q8,
                           single.p + (size_t) t * n_out, stream);
    native_mmvq_set_multi_exact(exact);
    check(cudaStreamSynchronize(stream), "sync");

    const auto qraw = raw.download((size_t) T * n_out);
    const auto qgot = got.download((size_t) T * n_out);
    const auto qsingle = single.download((size_t) T * n_out);
    double se = 0.0, sr = 0.0;
    double max_abs = 0.0;
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < kSparseMqaHeads; ++h) {
            const size_t base = ((size_t) t * kSparseMqaHeads + h) * kSparseMqaDim;
            double square = 0.0;
            for (int d = 0; d < kSparseMqaDim; ++d) {
                const double v = qraw[base + d];
                square += v * v;
            }
            const double scale = 1.0 / std::sqrt(square / kSparseMqaDim + eps);
            for (int d = 0; d < kSparseMqaDim; d += 2) {
                double a = qraw[base + d] * scale, b = qraw[base + d + 1] * scale;
                if (d >= kSparseMqaDim - kSparseMqaRopeDim) {
                    const int pair = (d - (kSparseMqaDim - kSparseMqaRopeDim)) / 2;
                    const double angle = (double) pos[t] * rope.step[pair];
                    const double c = std::cos(angle), s = std::sin(angle);
                    const double x = a * c - b * s;
                    b = a * s + b * c;
                    a = x;
                }
                for (int j = 0; j < 2; ++j) {
                    const double want = j ? b : a;
                    const double diff = qgot[base + d + j] - want;
                    se += diff * diff;
                    sr += want * want;
                    max_abs = std::max(max_abs, std::fabs(diff));
                }
            }
        }
    const double rel = std::sqrt(se / std::max(sr, 1e-30));
    const bool independent = qgot == qsingle;
    std::printf("per-head RMS + RoPE T=%d: CPU double rel RMS %.3g max abs %.3g; token-width independent %s\n",
                T, rel, max_abs, independent ? "yes" : "NO");
    check(cudaStreamDestroy(stream), "destroy stream");
    return rel < 2e-5 && independent ? 0 : 1;
}
