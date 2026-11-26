#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/models/deepseek4/layer.hpp"
#include "parity_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace ds4test;

namespace {

constexpr int Embd = 4096;
constexpr int Ff = 2048;
constexpr uint16_t HalfPointFive = 0x3800;

struct Q8Block {
    uint16_t d;
    int8_t q[32];
};
static_assert(sizeof(Q8Block) == 34);

std::vector<uint8_t> make_weights(int rows, int cols, int seed) {
    std::vector<uint8_t> bytes((size_t) rows * cols / 32 * sizeof(Q8Block));
    auto* blocks = reinterpret_cast<Q8Block*>(bytes.data());
    const int blocks_per_row = cols / 32;
    for (int row = 0; row < rows; ++row) {
        for (int b = 0; b < blocks_per_row; ++b) {
            Q8Block& block = blocks[(size_t) row * blocks_per_row + b];
            block.d = HalfPointFive;
            for (int i = 0; i < 32; ++i) {
                const int col = b * 32 + i;
                const int value = (row * 7 + col * 11 + seed * 3) % 5 - 2;
                block.q[i] = static_cast<int8_t>(value);
            }
        }
    }
    return bytes;
}

void reference(const std::vector<float>& x, int tokens, const std::vector<uint8_t>& gate,
               const std::vector<uint8_t>& up, const std::vector<uint8_t>& down, std::vector<float>& y) {
    const auto* g = reinterpret_cast<const Q8Block*>(gate.data());
    const auto* u = reinterpret_cast<const Q8Block*>(up.data());
    const auto* d = reinterpret_cast<const Q8Block*>(down.data());
    y.resize((size_t) tokens * Embd);
    for (int t = 0; t < tokens; ++t) {
        double hidden[Ff];
        for (int r = 0; r < Ff; ++r) {
            double gv = 0.0, uv = 0.0;
            for (int c = 0; c < Embd; ++c) {
                gv += (double) x[(size_t) t * Embd + c] * 0.5 * g[(size_t) r * (Embd / 32) + c / 32].q[c % 32];
                uv += (double) x[(size_t) t * Embd + c] * 0.5 * u[(size_t) r * (Embd / 32) + c / 32].q[c % 32];
            }
            gv = std::min(gv, 10.0);
            uv = std::clamp(uv, -10.0, 10.0);
            hidden[r] = gv / (1.0 + std::exp(-gv)) * uv;
        }
        for (int r = 0; r < Embd; ++r) {
            double value = 0.0;
            for (int c = 0; c < Ff; ++c)
                value += hidden[c] * 0.5 * d[(size_t) r * (Ff / 32) + c / 32].q[c % 32];
            y[(size_t) t * Embd + r] = static_cast<float>(value);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    using namespace basalt::models::deepseek4;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;

    const auto gate = make_weights(Ff, Embd, 1);
    const auto up = make_weights(Ff, Embd, 2);
    const auto down = make_weights(Embd, Ff, 3);
    auto input = randn(8u * Embd, 0x44533453, 0.015f);
    Dev<uint8_t> d_gate(gate), d_up(up), d_down(down);
    Dev<float> d_input(input), d_gate_rows(8u * Ff), d_up_rows(8u * Ff), d_hidden(8u * Ff), d_output(8u * Embd);
    Dev<uint8_t> d_input_q8(basalt::kernels::native_q8_1_bytes(Embd, 8));
    Dev<uint8_t> d_hidden_q8(basalt::kernels::native_q8_1_bytes(Ff, 8));
    cudaStream_t stream = nullptr;
    check(cudaStreamCreate(&stream), "create shared expert stream");

    SharedExpertWeights weights{8, 8, 8, d_gate.p, d_up.p, d_down.p, Embd, Ff, 10.f};
    SharedExpertScratch scratch{d_input_q8.p, d_hidden_q8.p, d_gate_rows.p, d_up_rows.p, d_hidden.p};
    shared_expert_step(d_input.p, 8, weights, scratch, d_output.p, stream);
    check(cudaStreamSynchronize(stream), "wait for batched shared expert");
    const auto got = d_output.down();
    std::vector<float> expected;
    reference(input, 8, gate, up, down, expected);
    const Err error = compare(got, expected);
    double max_rel = 0.0, ref_peak = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        ref_peak = std::max(ref_peak, std::fabs((double) expected[i]));
        if (std::fabs((double) expected[i]) >= 1.0)
            max_rel = std::max(max_rel, std::fabs((double) got[i] - expected[i]) / std::fabs((double) expected[i]));
    }
    bool independent = true;
    Dev<float> d_single_input(std::vector<float>(input.begin(), input.begin() + Embd));
    Dev<float> d_single_gate(Ff), d_single_up(Ff), d_single_hidden(Ff), d_single_output(Embd);
    Dev<uint8_t> d_single_input_q8(basalt::kernels::native_q8_1_bytes(Embd));
    Dev<uint8_t> d_single_hidden_q8(basalt::kernels::native_q8_1_bytes(Ff));
    SharedExpertScratch single_scratch{d_single_input_q8.p, d_single_hidden_q8.p, d_single_gate.p, d_single_up.p,
                                       d_single_hidden.p};
    shared_expert_step(d_single_input.p, 1, weights, single_scratch, d_single_output.p, stream);
    check(cudaStreamSynchronize(stream), "wait for single-token shared expert");
    const auto single = d_single_output.down();
    for (int i = 0; i < Embd; ++i) independent &= single[(size_t) i] == got[(size_t) i];

    std::printf("DeepSeek4 shared expert T=8 CPU-double ref: max_abs=%.6g max_rel(|ref|>=1)=%.6g "
                "rms_rel=%.6g ref_peak=%.6g; T=1/8 slot independence %s\n",
                error.max_abs, max_rel, error.rms_rel, ref_peak, independent ? "bitwise" : "DIFFERS");
    if (argc > 1 && std::string(argv[1]) == "--bench") {
        const double us = bench_stream_us([&](void* s) {
            shared_expert_step(d_input.p, 8, weights, scratch, d_output.p, s);
        });
        bench_line("deepseek4 shared expert T=8", us, gate.size() + up.size() + down.size());
    }
    cudaStreamDestroy(stream);
    return error.rms_rel <= 0.03 && independent ? 0 : 1;
}
