#include "basalt/core/device_copy.hpp"
#include "basalt/core/model_file.hpp"
#include "basalt/models/deepseek4/head.hpp"
#include "models/deepseek4/pack.hpp"
#include "parity_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace ds4test;

namespace {

uint16_t to_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t) (bits >> 16);
}

float from_bf16(uint16_t value) {
    const uint32_t bits = (uint32_t) value << 16;
    float result = 0.f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint32_t read_u32le(const uint8_t* p) {
    return (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
}

bool small_parity(bool bench) {
    using namespace basalt::models::deepseek4;
    constexpr int D = 64, V = 97, T = 3;
    constexpr float Eps = 1e-6f;
    auto hidden = randn((size_t) T * D, 0x44533448, 0.4f);
    auto norm = randn(D, 0x44533449, 0.05f, 1.f);
    std::vector<uint16_t> weights((size_t) V * D);
    for (size_t i = 0; i < weights.size(); ++i)
        weights[i] = to_bf16((float) ((int) (i % 19) - 9) * 0.002f);

    Dev<float> d_hidden(hidden), d_norm(norm), d_normalized((size_t) T * D), d_logits((size_t) T * V);
    Dev<uint16_t> d_weights(weights);
    cudaStream_t stream = nullptr;
    check(cudaStreamCreate(&stream), "create logits parity stream");
    logits_step(d_hidden.p, d_norm.p, d_weights.p, T, D, V, Eps, d_normalized.p, d_logits.p, stream);
    auto expected = std::vector<float>((size_t) T * V);
    for (int t = 0; t < T; ++t) {
        double square = 0.0;
        for (int k = 0; k < D; ++k) square += (double) hidden[(size_t) t * D + k] * hidden[(size_t) t * D + k];
        const double factor = 1.0 / std::sqrt(square / D + Eps);
        std::vector<double> normalized(D);
        for (int k = 0; k < D; ++k) normalized[(size_t) k] = hidden[(size_t) t * D + k] * factor * norm[(size_t) k];
        for (int v = 0; v < V; ++v) {
            double sum = 0.0;
            for (int k = 0; k < D; ++k) sum += normalized[(size_t) k] * from_bf16(weights[(size_t) v * D + k]);
            expected[(size_t) t * V + v] = (float) sum;
        }
    }

    Dev<float> d_single_norm(D), d_single_logits((size_t) T * V);
    for (int t = 0; t < T; ++t)
        logits_step(d_hidden.p + (size_t) t * D, d_norm.p, d_weights.p, 1, D, V, Eps,
                    d_single_norm.p, d_single_logits.p + (size_t) t * V, stream);
    check(cudaStreamSynchronize(stream), "synchronize logits parity");
    const auto got = d_logits.down();
    const auto singles = d_single_logits.down();
    const Err err = compare(got, expected);
    double max_rel = 0.0;
    bool independent = true;
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::fabs((double) expected[i]) >= 0.1)
            max_rel = std::max(max_rel, std::fabs((double) got[i] - expected[i]) / std::fabs((double) expected[i]));
        independent &= got[i] == singles[i];
    }
    std::printf("DeepSeek4 logits T=3 vs CPU-double: max_abs=%.6g max_rel(|ref|>=0.1)=%.6g rms_rel=%.6g; row independence %s\n",
                err.max_abs, max_rel, err.rms_rel, independent ? "bitwise" : "DIFFERS");
    if (bench) {
        const double us = bench_stream_us([&](void* s) {
            logits_step(d_hidden.p, d_norm.p, d_weights.p, T, D, V, Eps, d_normalized.p, d_logits.p, s);
        });
        bench_line("deepseek4 logits T=3 small geometry", us, weights.size() * sizeof(uint16_t));
    }
    cudaStreamDestroy(stream);
    return err.rms_rel < 2e-5 && max_rel < 2e-4 && independent;
}

bool real_pack_check(const char* path, bool bench) {
    using namespace basalt::models::deepseek4;
    Pack pack;
    std::string err;
    if (!open_pack(path, pack, err)) {
        basalt::log::error("basalt deepseek4 logits", "open pack: %s", err.c_str());
        return false;
    }
    const TensorRef* wt = pack.find("output.weight");
    const TensorRef* nt = pack.find("output_norm.weight");
    const int D = pack.geo.n_embd, V = pack.geo.n_vocab;
    const basalt::core::FileRange gguf = pack.file.range("gguf");
    if (!wt || !nt || wt->type != 30 || nt->type != 0 || D != 4096 || V != 129280 ||
        wt->bytes != (uint64_t) D * V * 2 || nt->bytes != (uint64_t) D * 4 ||
        wt->offset < gguf.offset || nt->offset < gguf.offset || wt->offset - gguf.offset > gguf.bytes ||
        wt->bytes > gguf.bytes - (wt->offset - gguf.offset) || nt->offset - gguf.offset > gguf.bytes ||
        nt->bytes > gguf.bytes - (nt->offset - gguf.offset)) {
        basalt::log::error("basalt deepseek4 logits", "pack output head has invalid type, geometry, or bounds");
        return false;
    }
    std::vector<uint8_t> raw_weights((size_t) wt->bytes), raw_norm((size_t) nt->bytes);
    if (!basalt::core::read_range_at(gguf, wt->offset - gguf.offset, raw_weights.data(), wt->bytes, err) ||
        !basalt::core::read_range_at(gguf, nt->offset - gguf.offset, raw_norm.data(), nt->bytes, err)) {
        basalt::log::error("basalt deepseek4 logits", "read output head: %s", err.c_str());
        return false;
    }
    std::vector<uint16_t> weights((size_t) D * V);
    for (size_t i = 0; i < weights.size(); ++i)
        weights[i] = (uint16_t) raw_weights[i * 2] | ((uint16_t) raw_weights[i * 2 + 1] << 8);
    std::vector<float> norm(D);
    for (int i = 0; i < D; ++i) {
        const uint32_t bits = read_u32le(raw_norm.data() + (size_t) i * 4);
        std::memcpy(&norm[(size_t) i], &bits, sizeof(bits));
    }
    const auto hidden = randn(D, 0x4453344a, 0.15f);
    Dev<uint16_t> d_weights(weights);
    Dev<float> d_norm(norm), d_hidden(hidden), d_normalized(D), d_logits(V);
    cudaStream_t stream = nullptr;
    check(cudaStreamCreate(&stream), "create real-pack logits stream");
    logits_step(d_hidden.p, d_norm.p, d_weights.p, 1, D, V, pack.geo.rms_eps,
                d_normalized.p, d_logits.p, stream);
    check(cudaStreamSynchronize(stream), "synchronize real-pack logits");
    const auto normalized = d_normalized.down();
    const auto logits = d_logits.down();
    double max_abs = 0.0, max_rel = 0.0;
    constexpr int sample[] = {0, 1, 17, 1024, 32768, 65535, 100000, 129279};
    for (const int token : sample) {
        double ref = 0.0;
        for (int k = 0; k < D; ++k) ref += (double) normalized[(size_t) k] * from_bf16(weights[(size_t) token * D + k]);
        const double delta = std::fabs((double) logits[(size_t) token] - ref);
        max_abs = std::max(max_abs, delta);
        if (std::fabs(ref) > 0.01) max_rel = std::max(max_rel, delta / std::fabs(ref));
    }
    std::printf("DeepSeek4 real output head (8 sampled logits): max_abs=%.6g max_rel(|ref|>0.01)=%.6g\n",
                max_abs, max_rel);
    if (bench) {
        const double us = bench_stream_us([&](void* s) {
            logits_step(d_hidden.p, d_norm.p, d_weights.p, 1, D, V, pack.geo.rms_eps,
                        d_normalized.p, d_logits.p, s);
        });
        bench_line("deepseek4 output head T=1, real BF16", us, wt->bytes);
    }
    cudaStreamDestroy(stream);
    return max_abs < 1e-2 && max_rel < 1e-2;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 3) return 2;
    const bool bench = (argc >= 2 && std::string(argv[argc - 1]) == "--bench");
    if (!small_parity(bench)) return 1;
    const char* pack = argc >= 2 && std::string(argv[1]) != "--bench" ? argv[1] : nullptr;
    if (pack && !real_pack_check(pack, bench)) return 1;
    return 0;
}
