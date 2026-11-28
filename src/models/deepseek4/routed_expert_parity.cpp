#include "basalt/kernels/iq_kernels.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/models/deepseek4/layer.hpp"
#include "models/deepseek4/routed_expert.hpp"
#include "parity_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace ds4test;

namespace {

constexpr int D = 4096;
constexpr int F = 2048;
constexpr int K = 6;
constexpr int E = 6;

struct Q8Block {
    uint16_t d;
    int8_t q[32];
};
static_assert(sizeof(Q8Block) == 34);

struct Expert {
    std::vector<uint8_t> gate, up, down, blob;
};

std::vector<uint8_t> matrix(int rows, int cols, int seed) {
    std::vector<uint8_t> bytes((size_t) rows * cols / 32 * sizeof(Q8Block));
    auto* blocks = reinterpret_cast<Q8Block*>(bytes.data());
    const int blocks_per_row = cols / 32;
    for (int row = 0; row < rows; ++row) {
        for (int b = 0; b < blocks_per_row; ++b) {
            Q8Block& block = blocks[(size_t) row * blocks_per_row + b];
            block.d = 0x3800;  // fp16 0.5
            for (int i = 0; i < 32; ++i) {
                const int col = b * 32 + i;
                block.q[i] = (int8_t) (((row * 7 + col * 11 + seed * 3) % 5) - 2);
            }
        }
    }
    return bytes;
}

void cpu_reference(const std::vector<float>& x, const std::vector<Expert>& experts, const std::vector<float>& weights,
                   int tokens, std::vector<float>& output) {
    output.assign((size_t) tokens * D, 0.f);
    for (int t = 0; t < tokens; ++t) {
        for (int slot = 0; slot < K; ++slot) {
            const int e = (slot - (t % E) + E) % E;
            const auto* gate = reinterpret_cast<const Q8Block*>(experts[(size_t) e].gate.data());
            const auto* up = reinterpret_cast<const Q8Block*>(experts[(size_t) e].up.data());
            const auto* down = reinterpret_cast<const Q8Block*>(experts[(size_t) e].down.data());
            double hidden[F];
            for (int r = 0; r < F; ++r) {
                double g = 0.0, u = 0.0;
                for (int c = 0; c < D; ++c) {
                    g += (double) x[(size_t) t * D + c] * 0.5 * gate[(size_t) r * (D / 32) + c / 32].q[c % 32];
                    u += (double) x[(size_t) t * D + c] * 0.5 * up[(size_t) r * (D / 32) + c / 32].q[c % 32];
                }
                g = std::min(g, 10.0);
                u = std::clamp(u, -10.0, 10.0);
                hidden[r] = g / (1.0 + std::exp(-g)) * u;
            }
            for (int r = 0; r < D; ++r) {
                double v = 0.0;
                for (int c = 0; c < F; ++c)
                    v += hidden[c] * 0.5 * down[(size_t) r * (F / 32) + c / 32].q[c % 32];
                output[(size_t) t * D + r] += (float) (weights[(size_t) t * K + slot] * v);
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    using namespace basalt::kernels;
    using namespace basalt::models::deepseek4;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;

    constexpr int kSwiRows = 13;
    auto split = randn((size_t) kSwiRows * 2 * F, 0x44533453, 0.2f);
    Dev<float> d_split(split), d_hidden((size_t) kSwiRows * F);
    cudaStream_t swiglu_stream = nullptr;
    check(cudaStreamCreate(&swiglu_stream), "create split SwiGLU stream");
    swiglu_split_clamped(d_split.p, kSwiRows, F, 10.f, d_hidden.p, swiglu_stream);
    check(cudaStreamSynchronize(swiglu_stream), "wait for split SwiGLU");
    const auto split_got = d_hidden.down();
    std::vector<float> split_expected((size_t) kSwiRows * F);
    for (int row = 0; row < kSwiRows; ++row)
        for (int col = 0; col < F; ++col) {
            const float g = std::min(split[(size_t) row * 2 * F + col], 10.f);
            const float u = std::clamp(split[(size_t) row * 2 * F + F + col], -10.f, 10.f);
            split_expected[(size_t) row * F + col] = g / (1.f + std::exp(-g)) * u;
        }
    const Err split_error = compare(split_got, split_expected);
    cudaStreamDestroy(swiglu_stream);
    if (split_error.max_abs > 2e-5) {
        std::fprintf(stderr, "DeepSeek4 split SwiGLU parity failed: max_abs=%.6g\n", split_error.max_abs);
        return 1;
    }

    std::vector<Expert> experts(E);
    std::vector<uint8_t> storage;
    for (int e = 0; e < E; ++e) {
        Expert& expert = experts[(size_t) e];
        expert.gate = matrix(F, D, 10 + e);
        expert.up = matrix(F, D, 20 + e);
        expert.down = matrix(D, F, 30 + e);
        expert.blob.reserve(expert.gate.size() + expert.up.size() + expert.down.size());
        expert.blob.insert(expert.blob.end(), expert.gate.begin(), expert.gate.end());
        expert.blob.insert(expert.blob.end(), expert.up.begin(), expert.up.end());
        expert.blob.insert(expert.blob.end(), expert.down.begin(), expert.down.end());
        storage.insert(storage.end(), expert.blob.begin(), expert.blob.end());
    }
    const size_t expert_bytes = experts.front().blob.size();
    auto input = randn(8u * D, 0x44533452, 0.05f);
    std::vector<float> route_weights(8u * K, 1.f / K);
    std::vector<int32_t> starts(E + 1), counts = {E}, dst(8u * K), tok(8u * K);
    for (int e = 0; e < E; ++e) {
        starts[(size_t) e] = e * 8;
        for (int t = 0; t < 8; ++t) {
            const int entry = e * 8 + t;
            dst[(size_t) entry] = t * K + ((e + t) % K);
            tok[(size_t) entry] = t;
        }
    }
    starts[E] = 8 * K;

    Dev<uint8_t> d_storage(storage), d_xq(native_q8_1_bytes(D, 8));
    Dev<float> d_x(input), d_weights(route_weights), d_rows(8u * K * D), d_output(8u * D);
    Dev<uint8_t> d_expert_scratch(native_expert_scratch_bytes(8 * K, F));
    std::vector<unsigned long long> ptrs(E);
    for (int e = 0; e < E; ++e)
        ptrs[(size_t) e] = reinterpret_cast<unsigned long long>(d_storage.p + (size_t) e * expert_bytes);
    Dev<unsigned long long> d_ptrs(ptrs);
    Dev<int32_t> d_starts(starts), d_counts(counts), d_dst(dst), d_tok(tok);

    cudaStream_t stream = nullptr;
    check(cudaStreamCreate(&stream), "create routed expert stream");
    RoutedExpertWeights weights{8, 8, D, F, 10.f};
    RoutedExpertPlan plan{d_ptrs.p, d_starts.p, d_counts.p, d_dst.p, d_tok.p, E, 8 * K};
    RoutedExpertScratch scratch{d_xq.p, d_expert_scratch.p, d_rows.p};
    routed_expert_step(d_x.p, d_weights.p, 8, weights, plan, scratch, d_output.p, stream);
    check(cudaStreamSynchronize(stream), "wait for grouped routed experts");
    const auto got = d_output.down();
    std::vector<float> expected;
    cpu_reference(input, experts, route_weights, 8, expected);
    const Err error = compare(got, expected);
    double max_rel = 0.0, ref_peak = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        ref_peak = std::max(ref_peak, std::fabs((double) expected[i]));
        if (std::fabs((double) expected[i]) >= 1.0)
            max_rel = std::max(max_rel, std::fabs((double) got[i] - expected[i]) / std::fabs((double) expected[i]));
    }

    std::vector<int32_t> one_starts(E + 1), one_counts = {E}, one_dst(E), one_tok(E, 0);
    for (int e = 0; e < E; ++e) {
        one_starts[(size_t) e] = e;
        one_dst[(size_t) e] = e;
    }
    one_starts[E] = E;
    Dev<int32_t> d_one_starts(one_starts), d_one_counts(one_counts), d_one_dst(one_dst), d_one_tok(one_tok);
    Dev<float> d_one_weights(std::vector<float>(route_weights.begin(), route_weights.begin() + K));
    Dev<float> d_one_rows(K * D), d_one_output(D);
    Dev<uint8_t> d_one_xq(native_q8_1_bytes(D)), d_one_scratch(native_expert_scratch_bytes(K, F));
    RoutedExpertPlan one_plan{d_ptrs.p, d_one_starts.p, d_one_counts.p, d_one_dst.p, d_one_tok.p, E, K};
    RoutedExpertScratch one_scratch{d_one_xq.p, d_one_scratch.p, d_one_rows.p};
    routed_expert_step(d_x.p, d_one_weights.p, 1, weights, one_plan, one_scratch, d_one_output.p, stream);
    check(cudaStreamSynchronize(stream), "wait for one-token routed experts");
    const auto got_one = d_one_output.down();
    bool independent = true;
    for (int i = 0; i < D; ++i) independent &= got_one[(size_t) i] == got[(size_t) i];

    std::printf("DeepSeek4 grouped routed experts T=8 CPU-double ref: max_abs=%.6g "
                "max_rel(|ref|>=1)=%.6g rms_rel=%.6g ref_peak=%.6g; T=1/8 slot independence %s\n",
                error.max_abs, max_rel, error.rms_rel, ref_peak, independent ? "bitwise" : "DIFFERS");
    if (argc > 1 && std::string(argv[1]) == "--bench") {
        const double us = bench_stream_us([&](void* s) {
            routed_expert_step(d_x.p, d_weights.p, 8, weights, plan, scratch, d_output.p, s);
        });
        bench_line("deepseek4 routed experts T=8", us, E * expert_bytes);
    }
    {
        constexpr int T = 2, H = 64, K = 6;
        std::vector<float> h_rows(T * K * H), h_w(T * K), h_shared(T * H), h_out(T * H);
        for (size_t i = 0; i < h_rows.size(); ++i) h_rows[i] = std::sin((float) i * 0.13f);
        for (size_t i = 0; i < h_w.size(); ++i) h_w[i] = 0.1f + 0.05f * (float) i;
        for (size_t i = 0; i < h_shared.size(); ++i) h_shared[i] = std::cos((float) i * 0.07f);
        float *d_rows = nullptr, *d_w = nullptr, *d_shared = nullptr, *d_out = nullptr;
        if (cudaMalloc(&d_rows, h_rows.size() * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&d_w, h_w.size() * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&d_shared, h_shared.size() * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&d_out, h_out.size() * sizeof(float)) != cudaSuccess) return 1;
        cudaMemcpy(d_rows, h_rows.data(), h_rows.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_w, h_w.data(), h_w.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_shared, h_shared.data(), h_shared.size() * sizeof(float), cudaMemcpyHostToDevice);
        basalt::models::deepseek4::WindowRowPtrs ptrs;
        for (int r = 0; r < T * K; ++r) ptrs.p[r] = d_rows + (size_t) r * H;
        basalt::models::deepseek4::weighted_route_sum_ptrs(ptrs, d_w, d_shared, T, H, d_out, stream);
        cudaStreamSynchronize(stream);
        cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost);
        double max_err = 0.0;
        for (int t = 0; t < T; ++t) {
            for (int c = 0; c < H; ++c) {
                double ref = h_shared[(size_t) t * H + c];
                for (int k = 0; k < K; ++k) ref += (double) h_w[(size_t) t * K + k] * h_rows[((size_t) t * K + k) * H + c];
                max_err = std::max(max_err, std::abs((double) h_out[(size_t) t * H + c] - ref));
            }
        }
        cudaFree(d_rows); cudaFree(d_w); cudaFree(d_shared); cudaFree(d_out);
        if (max_err > 1e-5) {
            std::printf("weighted_route_sum_ptrs max_err=%.3e\n", max_err);
            cudaStreamDestroy(stream);
            return 1;
        }
    }
    cudaStreamDestroy(stream);
    return error.rms_rel <= 0.03 && independent ? 0 : 1;
}
