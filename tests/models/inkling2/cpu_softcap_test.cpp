#include "basalt/ggml/ggml.h"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/cpu/pool.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = basalt::kernels::cpu;

namespace {

uint64_t output_hash(uint64_t hash, const float& value) {
    const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
    for (size_t i = 0; i < sizeof(value); ++i) hash = (hash ^ bytes[i]) * 1099511628211ull;
    return hash;
}

int check(ggml_type type, int width, std::mt19937& rng, std::normal_distribution<float>& normal, bool bench) {
    int failures = 0;
    const int hidden = width == 64 ? 64 : 1152;
    cpu::NativeFmt f;
    std::string error;
    if (!cpu::native_fmt(type, type, width, hidden, f, error)) {
        std::printf("softcap type=%s width=%d: %s\n", ggml_type_name(type), width, error.c_str());
        return 1;
    }
    f.gate_softcap = 7.f;
    f.up_softcap = 4.f;
    std::vector<uint8_t> blob(f.bytes);
    std::vector<float> raw((size_t) width * hidden);
    const auto* weights = ggml_get_type_traits((ggml_type) type);
    for (size_t offset : {size_t(0), f.up_off}) {
        for (float& v : raw) v = normal(rng);
        weights->from_float_ref(raw.data(), blob.data() + offset, raw.size());
    }
    for (float& v : raw) v = normal(rng);
    weights->from_float_ref(raw.data(), blob.data() + f.down_off, raw.size());
    if (f.tail) {
        const float scales[4] = {1.25f, 0.75f, 1.f, 0.f};
        std::memcpy(blob.data() + f.bytes - f.tail, scales, sizeof(scales));
    }
    std::vector<std::vector<uint8_t>> quant(8, std::vector<uint8_t>(f.act_bytes));
    std::vector<std::vector<float>> input(8, std::vector<float>(width));
    std::vector<std::vector<float>> single(8, std::vector<float>(hidden));
    std::vector<float> gw(width), uw(width);
    const void* act[8];
    float* output[8];
    double max_abs = 0, max_rel = 0, max_scaled = 0, previous_abs = 0, previous_rel = 0;
    uint64_t actual_hash = 14695981039346656037ull, previous_hash = actual_hash;
    for (int t = 0; t < 8; ++t) {
        for (float& v : input[t]) v = normal(rng);
        cpu::native_quant_act(f, input[t].data(), quant[t].data());
        ggml_get_type_traits((ggml_type) f.gu_act)->to_float(quant[t].data(), input[t].data(), width);
        act[t] = quant[t].data();
        output[t] = single[t].data();
        cpu::native_gu_rows(f, blob.data(), act + t, 1, output + t, 0, hidden);
        for (int r = 0; r < hidden; ++r) {
            float old_g = 0, old_u = 0;
            const auto dot = ggml_get_type_traits_cpu((ggml_type) type)->vec_dot;
            dot(width, &old_g, 0, blob.data() + (size_t) r * f.gu_row, 0, act[t], 0, 1);
            dot(width, &old_u, 0, blob.data() + f.up_off + (size_t) r * f.up_row, 0, act[t], 0, 1);
            if (f.tail) { old_g *= 1.25f; old_u *= 0.75f; }
            old_g = f.gate_softcap * std::tanh(old_g / f.gate_softcap);
            old_u = f.up_softcap * std::tanh(old_u / f.up_softcap);
            const float previous = (old_g / (1.f + std::exp(-old_g))) * old_u;
            const double previous_delta = std::abs((double) previous - single[t][r]);
            previous_abs = std::max(previous_abs, previous_delta);
            previous_rel = std::max(previous_rel, previous_delta / std::max(1e-8, std::abs((double) previous)));
            actual_hash = output_hash(actual_hash, single[t][r]);
            previous_hash = output_hash(previous_hash, previous);
            if (std::memcmp(&previous, &single[t][r], sizeof(float))) ++failures;
            weights->to_float(blob.data() + (size_t) r * f.gu_row, gw.data(), width);
            weights->to_float(blob.data() + f.up_off + (size_t) r * f.up_row, uw.data(), width);
            double g = 0, u = 0;
            for (int c = 0; c < width; ++c) {
                g += (double) gw[c] * input[t][c];
                u += (double) uw[c] * input[t][c];
            }
            if (f.tail) { g *= 1.25; u *= 0.75; }
            g = f.gate_softcap * std::tanh(g / f.gate_softcap);
            u = f.up_softcap * std::tanh(u / f.up_softcap);
            const double ref = g / (1. + std::exp(-g)) * u;
            const double delta = std::abs(ref - single[t][r]);
            max_abs = std::max(max_abs, delta);
            max_rel = std::max(max_rel, delta / std::max(1e-8, std::abs(ref)));
            max_scaled = std::max(max_scaled, delta / std::max(1., std::abs(ref)));
        }
    }
    for (int n = 1; n <= 8; ++n) for (int offset = 0; offset + n <= 8; ++offset) {
        std::vector<std::vector<float>> grouped(8, std::vector<float>(hidden, -12345.f));
        for (int t = 0; t < 8; ++t) output[t] = grouped[t].data();
        cpu::native_gu_rows(f, blob.data(), act + offset, n, output, 0, hidden);
        bool same = true;
        for (int t = 0; t < n; ++t)
            same &= std::memcmp(grouped[t].data(), single[offset + t].data(), hidden * sizeof(float)) == 0;
        const void* reversed[8];
        for (int t = 0; t < n; ++t) {
            reversed[t] = act[7 - offset - t];
            std::fill(grouped[t].begin(), grouped[t].end(), -12345.f);
        }
        cpu::native_gu_rows(f, blob.data(), reversed, n, output, 0, hidden / 2);
        for (int t = 0; t < n; ++t) {
            same &= std::memcmp(grouped[t].data(), single[7 - offset - t].data(), hidden / 2 * sizeof(float)) == 0;
            for (int r = hidden / 2; r < hidden; ++r) same &= grouped[t][r] == -12345.f;
            std::fill(grouped[t].begin(), grouped[t].begin() + hidden / 2, -12345.f);
        }
        cpu::native_gu_rows(f, blob.data(), reversed, n, output, hidden / 2, hidden);
        for (int t = 0; t < n; ++t) {
            for (int r = 0; r < hidden / 2; ++r) same &= grouped[t][r] == -12345.f;
            same &= std::memcmp(grouped[t].data() + hidden / 2, single[7 - offset - t].data() + hidden / 2,
                                hidden / 2 * sizeof(float)) == 0;
        }
        for (int t = n; t < 8; ++t)
            for (float v : grouped[t]) same &= v == -12345.f;
        if (!same) {
            std::printf("softcap type=%s width=%d nt=%d offset=%d: batching/slice FAIL\n",
                        ggml_type_name(type), width, n, offset);
            ++failures;
        }
    }
    std::vector<std::vector<float>> serial(8, std::vector<float>(width));
    std::vector<uint8_t> hq(f.h_bytes);
    for (int t = 0; t < 8; ++t) {
        cpu::native_quant_h(f, single[t].data(), hq.data());
        const void* a[] = {hq.data()};
        float* y[] = {serial[t].data()};
        cpu::native_down_rows(f, blob.data(), a, 1, y, 0, width);
    }
    for (int workers : {3, 12}) {
        cpu::ExpertPool pool(workers);
        for (int count : {1, 8, 32}) {
            std::vector<cpu::ExpertJobMulti> jobs(count);
            std::vector<std::vector<float>> result(count * 8, std::vector<float>(width));
            for (int e = 0; e < count; ++e) {
                jobs[e].blob = blob.data();
                jobs[e].nt = 1 + e % 8;
                for (int t = 0; t < jobs[e].nt; ++t) {
                    jobs[e].nact[t] = quant[(e + t) % 8].data();
                    jobs[e].out[t] = result[e * 8 + t].data();
                }
            }
            pool.run_split_multi_native(f, jobs.data(), count);
            for (int e = 0; e < count; ++e)
                for (int t = 0; t < jobs[e].nt; ++t)
                    if (std::memcmp(result[e * 8 + t].data(), serial[(e + t) % 8].data(), width * sizeof(float))) ++failures;
        }
    }
    std::printf("softcap type=%d width=%d: double max_abs=%g max_rel=%g max_scaled=%g; previous max_abs=%g max_rel=%g\n",
                type, width, max_abs, max_rel, max_scaled, previous_abs, previous_rel);
    std::printf("  output hashes: actual=%016llx previous=%016llx\n",
                (unsigned long long) actual_hash, (unsigned long long) previous_hash);
    // The reference accumulates in double; the row kernels accumulate in float.
    if (max_abs > 3e-4 || max_scaled > 5e-5 || actual_hash != previous_hash) ++failures;
    if (bench) {
        for (int t = 0; t < 8; ++t) output[t] = single[t].data();
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 8; ++i) cpu::native_gu_rows(f, blob.data(), act, 4, output, 0, hidden);
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / 8;
        std::printf("  four rows: %.1f us, %.2f GFLOP/s\n", us, 16. * width * hidden / (us * 1000.));
    }
    return failures;
}
}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    std::mt19937 rng(29);
    std::normal_distribution<float> normal(0.f, 0.5f);
    int failures = 0;
    for (ggml_type type : {GGML_TYPE_Q8_0, GGML_TYPE_NVFP4})
        for (int width : {64, 2304}) failures += check(type, width, rng, normal, bench);
    std::printf("cpu softcap: %d failures\n", failures);
    return failures ? 1 : 0;
}
