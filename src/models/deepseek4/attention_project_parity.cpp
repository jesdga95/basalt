#include "basalt/models/deepseek4/attention.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "parity_util.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace ds4test;

namespace {

constexpr int K = 4096;
constexpr int N = 512;
constexpr int Blocks = K / 32;
constexpr int Q8_0Bytes = 34;
constexpr int Q8_1Bytes = 36;

uint16_t half_bits(float value) {
    const __half h = __float2half_rn(value);
    uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

float half_value(const uint8_t* bytes) {
    __half h;
    std::memcpy(&h, bytes, sizeof(h));
    return __half2float(h);
}

std::vector<uint8_t> q8_weights() {
    std::vector<uint8_t> weights((size_t) N * Blocks * Q8_0Bytes);
    const uint16_t scale = half_bits(0.0125f);
    for (int row = 0; row < N; ++row) {
        for (int block = 0; block < Blocks; ++block) {
            uint8_t* dst = weights.data() + ((size_t) row * Blocks + block) * Q8_0Bytes;
            std::memcpy(dst, &scale, sizeof(scale));
            for (int i = 0; i < 32; ++i) {
                const int value = ((row * 17 + block * 7 + i * 13) % 255) - 127;
                dst[2 + i] = (uint8_t) (int8_t) value;
            }
        }
    }
    return weights;
}

std::vector<float> cpu_projection(const std::vector<uint8_t>& weights, const std::vector<uint8_t>& q8_1, int tokens) {
    std::vector<float> out((size_t) tokens * N);
    for (int t = 0; t < tokens; ++t) {
        for (int row = 0; row < N; ++row) {
            double sum = 0.0;
            for (int b = 0; b < Blocks; ++b) {
                const uint8_t* wb = weights.data() + ((size_t) row * Blocks + b) * Q8_0Bytes;
                const uint8_t* xb = q8_1.data() + ((size_t) t * Blocks + b) * Q8_1Bytes;
                int dot = 0;
                for (int i = 0; i < 32; ++i)
                    dot += (int) (int8_t) wb[2 + i] * (int) (int8_t) xb[4 + i];
                sum += (double) half_value(wb) * half_value(xb) * dot;
            }
            out[(size_t) t * N + row] = (float) sum;
        }
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace basalt::models::deepseek4;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    constexpr int T = 3;
    const auto weights = q8_weights();
    const auto x = randn((size_t) T * K, 0x41544b56, 0.7f);
    const auto norm = randn(N, 0x41544e4d, 0.1f, 1.f);
    const std::vector<int32_t> positions = {0, 127, 4096};
    basalt::kernels::MqaRopeTable rope;
    const auto frequency = yarn_frequencies(64, 160000.f, 16.f, 65536, 32.f, 1.f);
    std::copy(frequency.begin(), frequency.end(), rope.step);

    Dev<uint8_t> dw(weights), scratch(attention_kv_scratch_bytes(T));
    Dev<float> dx(x), dn(norm), projected((size_t) T * N), out((size_t) T * N);
    Dev<int32_t> dp(positions);
    cudaStream_t stream = nullptr;
    check(cudaStreamCreate(&stream), "create projection stream");
    project_attention_kv(dx.p, 8, dw.p, dn.p, 1e-20f, dp.p, rope, T, scratch.p, projected.p, out.p, stream);
    check(cudaStreamSynchronize(stream), "synchronize projection stream");

    const auto q8_1 = scratch.down();
    const auto projected_cpu = cpu_projection(weights, q8_1, T);
    const auto projected_gpu = projected.down();
    const Err projection_error = compare(projected_gpu, projected_cpu);

    Dev<float> direct((size_t) T * N);
    prepare_attention_kv(projected.p, dn.p, 1e-20f, dp.p, rope, T, direct.p, stream);
    check(cudaStreamSynchronize(stream), "synchronize direct KV boundary");
    const auto composed = out.down();
    const auto direct_rows = direct.down();
    const bool same_pipeline = std::memcmp(composed.data(), direct_rows.data(), composed.size() * sizeof(float)) == 0;

    bool independent = true;
    for (int t = 0; t < T; ++t) {
        Dev<float> sx(std::vector<float>(x.begin() + (size_t) t * K, x.begin() + (size_t) (t + 1) * K));
        Dev<int32_t> sp(std::vector<int32_t>{positions[(size_t) t]});
        Dev<uint8_t> ss(attention_kv_scratch_bytes(1));
        Dev<float> sr(N), so(N);
        project_attention_kv(sx.p, 8, dw.p, dn.p, 1e-20f, sp.p, rope, 1, ss.p, sr.p, so.p, stream);
        check(cudaStreamSynchronize(stream), "synchronize single-row projection");
        const auto single = so.down();
        independent = independent && std::memcmp(single.data(), composed.data() + (size_t) t * N, N * sizeof(float)) == 0;
    }

    std::printf("DeepSeek4 KV projection K=%d N=%d T=%d: Q8_0 CPU-double max abs %.6g rel RMS %.6g; "
                "composition %s; token-width independence %s\n", K, N, T, projection_error.max_abs,
                projection_error.rms_rel, same_pipeline ? "bitwise" : "DIFFERS", independent ? "bitwise" : "DIFFERS");
    if (projection_error.max_abs > 2e-4 || projection_error.rms_rel > 2e-5 || !same_pipeline || !independent) return 1;

    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        const double us = bench_stream_us([&](void* bench_stream) {
            project_attention_kv(dx.p, 8, dw.p, dn.p, 1e-20f, dp.p, rope, 1, scratch.p, projected.p, out.p, bench_stream);
        });
        const double bytes = (double) N * Blocks * Q8_0Bytes + (double) K * sizeof(float) +
                             (double) N * sizeof(float) * 2 + attention_kv_scratch_bytes(1);
        bench_line("deepseek4 KV projection T=1", us, bytes);
    }
    cudaStreamDestroy(stream);
    return 0;
}
