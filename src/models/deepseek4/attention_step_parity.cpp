#include "basalt/models/deepseek4/attention.hpp"
#include "basalt/models/deepseek4/layer.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/kernels/row_norm.hpp"
#include "parity_util.hpp"

#include <cuda_fp16.h>

#include <cstdint>
#include <cstring>
#include <vector>

using namespace ds4test;

namespace {

constexpr int Embd = 4096;
constexpr int Rank = 1024;
constexpr int HeadsDim = 64 * 512;
constexpr int Q8Block = 34;

uint16_t half_bits(float value) {
    const __half h = __float2half_rn(value);
    uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

std::vector<uint8_t> q8_matrix(int rows, int cols, uint32_t seed) {
    constexpr int block = 32;
    const int blocks = cols / block;
    std::vector<uint8_t> weights((size_t) rows * blocks * Q8Block);
    const uint16_t scale = half_bits(0.001f);
    uint32_t state = seed;
    for (int row = 0; row < rows; ++row) {
        for (int b = 0; b < blocks; ++b) {
            uint8_t* dst = weights.data() + ((size_t) row * blocks + b) * Q8Block;
            std::memcpy(dst, &scale, sizeof(scale));
            for (int i = 0; i < block; ++i) {
                state ^= state << 13; state ^= state >> 17; state ^= state << 5;
                const int q = (int) (state % 15) - 7;
                dst[2 + i] = (uint8_t) (int8_t) q;
            }
        }
    }
    return weights;
}

struct TestBuffers {
    Dev<float> x{Embd}, q_norm{Rank}, kv_norm{512}, sinks{64};
    Dev<float> q_latent{Rank}, q_heads{HeadsDim}, kv_projected{512}, kv_row{512};
    Dev<float> mqa_output{HeadsDim}, output_latent{8192}, output{Embd};
    Dev<float> mqa_workspace{basalt::kernels::sparse_mqa_workspace_floats(1, 0)};
    Dev<uint8_t> input_q8{basalt::kernels::native_q8_1_bytes(Embd, 1)};
    Dev<uint8_t> q_latent_q8{basalt::kernels::sparse_mqa_q_scratch_bytes(1, Rank)};
    Dev<uint8_t> output_q8{basalt::kernels::sparse_mqa_output_scratch_bytes(1)};
    Dev<int32_t> pos{1}, win_start{1}, win_n{1};
    Dev<float> ring{(size_t) 128 * 512};
};

void set_ring(TestBuffers& b, const std::vector<float>& ring, int32_t pos, int32_t start, int32_t count) {
    b.ring.up(ring);
    b.pos.up(std::vector<int32_t>{pos});
    b.win_start.up(std::vector<int32_t>{start});
    b.win_n.up(std::vector<int32_t>{count});
}

}  // namespace

int main() {
    using namespace basalt::kernels;
    using namespace basalt::models::deepseek4;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;

    const auto q_a = q8_matrix(Rank, Embd, 0x41545331);
    const auto q_b = q8_matrix(HeadsDim, Rank, 0x41545332);
    const auto kv = q8_matrix(512, Embd, 0x41545333);
    const auto output_a = q8_matrix(8 * 1024, Embd, 0x41545334);
    const auto output_b = q8_matrix(Embd, 8 * 1024, 0x41545335);
    Dev<uint8_t> d_q_a(q_a), d_q_b(q_b), d_kv(kv), d_output_a(output_a), d_output_b(output_b);

    TestBuffers actual, direct;
    const auto x = randn(Embd, 0x41545336, 0.15f);
    const auto ring = randn((size_t) 128 * 512, 0x41545337, 0.05f);
    const auto q_norm = std::vector<float>(Rank, 1.f);
    const auto kv_norm = std::vector<float>(512, 1.f);
    const auto sinks = std::vector<float>(64, -0.25f);
    actual.x.up(x); direct.x.up(x);
    actual.q_norm.up(q_norm); direct.q_norm.up(q_norm);
    actual.kv_norm.up(kv_norm); direct.kv_norm.up(kv_norm);
    actual.sinks.up(sinks); direct.sinks.up(sinks);
    set_ring(actual, ring, 130, 3, 128);
    set_ring(direct, ring, 130, 3, 128);

    AttentionStepWeights weights;
    weights.q = {d_q_a.p, 8, actual.q_norm.p, d_q_b.p, 8, 1e-20f, Embd, Rank, 1e-20f};
    weights.kv_type = 8; weights.kv = d_kv.p; weights.kv_norm = actual.kv_norm.p; weights.sinks = actual.sinks.p;
    weights.output_a_type = 8; weights.output_a = d_output_a.p;
    weights.output_b_type = 8; weights.output_b = d_output_b.p;

    MqaRopeTable rope;
    const auto frequency = yarn_frequencies(64, 160000.f, 16.f, 65536, 32.f, 1.f);
    std::copy(frequency.begin(), frequency.end(), rope.step);
    SparseMqaArgs selection;
    selection.n_tokens = 1; selection.n_sel = 0;
    selection.pos = actual.pos.p; selection.win_start = actual.win_start.p; selection.win_n = actual.win_n.p;
    selection.scale = 1.f / std::sqrt(512.f);

    cudaStream_t stream = nullptr;
    check(cudaStreamCreate(&stream), "create attention-step stream");
    AttentionStepScratch scratch{actual.input_q8.p, actual.q_latent.p, actual.q_latent_q8.p, actual.q_heads.p,
                                 actual.kv_projected.p, actual.kv_row.p, actual.mqa_output.p,
                                 actual.mqa_workspace.p, actual.output_q8.p, actual.output_latent.p, actual.output.p};
    attention_step(actual.x.p, weights, rope, actual.ring.p, selection, scratch, stream);

    native_quantize_q8_1(direct.x.p, direct.input_q8.p, Embd, 1, stream);
    SparseMqaQWeights q_weights = weights.q;
    sparse_mqa_q_latent(direct.input_q8.p, q_weights, 1, direct.q_latent.p, direct.q_latent_q8.p, stream);
    sparse_mqa_q_heads(q_weights, 1, direct.pos.p, rope, direct.q_latent_q8.p, direct.q_heads.p, stream);
    project_attention_kv(direct.x.p, 8, d_kv.p, direct.kv_norm.p, 1e-20f, direct.pos.p, rope, 1,
                         direct.input_q8.p, direct.kv_projected.p, direct.kv_row.p, stream);
    cache_attention_window(direct.ring.p, direct.kv_row.p, direct.pos.p, 1, stream);
    SparseMqaArgs direct_args = selection;
    direct_args.pos = direct.pos.p; direct_args.win_start = direct.win_start.p; direct_args.win_n = direct.win_n.p;
    direct_args.q = direct.q_heads.p; direct_args.win = direct.ring.p; direct_args.win_type = MqaRowType::F32;
    direct_args.win_cap = 128; direct_args.sinks = direct.sinks.p; direct_args.rope = rope;
    direct_args.round_output_bf16 = true; direct_args.out = direct.mqa_output.p;
    sparse_mqa(direct_args, direct.mqa_workspace.p, stream);
    sparse_mqa_output(direct.mqa_output.p, 8, d_output_a.p, 8, d_output_b.p, Embd, 1,
                      direct.output_q8.p, direct.output_latent.p, direct.output.p, stream);
    check(cudaStreamSynchronize(stream), "synchronize attention-step parity");

    const auto actual_out = actual.output.down();
    const auto direct_out = direct.output.down();
    const auto actual_ring = actual.ring.down();
    const auto direct_ring = direct.ring.down();
    const bool output_equal = actual_out == direct_out;
    const bool ring_equal = actual_ring == direct_ring;

    const auto streams_host = randn((size_t) 4 * Embd, 0x41545338, 0.2f);
    const auto hc_fn_host = randn((size_t) kHcCoef * 4 * Embd, 0x41545339, 0.0002f);
    const std::vector<float> hc_scale_host = {0.7f, 0.9f, 0.6f};
    const auto hc_base_host = randn(kHcCoef, 0x4154533a, 0.15f);
    const std::vector<float> attn_norm_host(Embd, 1.f);
    Dev<float> streams((size_t) 4 * Embd), next_streams((size_t) 4 * Embd), expected_streams((size_t) 4 * Embd);
    Dev<float> hc_fn(hc_fn_host), hc_scale(hc_scale_host), hc_base(hc_base_host), attn_norm(attn_norm_host);
    Dev<float> hc_workspace(hc_mixes_workspace_floats(1)), hc_coefficients(kHcCoef), hc_collapsed(Embd), hc_normalized(Embd);
    streams.up(streams_host);
    set_ring(actual, ring, 130, 3, 128);
    set_ring(direct, ring, 130, 3, 128);

    AttentionLayerWeights layer_weights;
    layer_weights.hc = {Embd, 1e-20f, 1e-6f, 20};
    layer_weights.hc_fn = hc_fn.p;
    layer_weights.hc_scale = hc_scale.p;
    layer_weights.hc_base = hc_base.p;
    layer_weights.norm = attn_norm.p;
    layer_weights.attention = weights;
    AttentionLayerScratch layer_scratch{hc_workspace.p, hc_coefficients.p, hc_collapsed.p, hc_normalized.p, scratch};
    attention_layer_step(streams.p, layer_weights, rope, actual.ring.p, selection, layer_scratch, next_streams.p, stream);

    HcArgs hc_args = layer_weights.hc;
    Dev<float> direct_hc_workspace(hc_mixes_workspace_floats(1)), direct_hc_coefficients(kHcCoef);
    Dev<float> direct_collapsed(Embd), direct_normalized(Embd);
    hc_mixes(streams.p, hc_fn.p, hc_scale.p, hc_base.p, 1, hc_args, direct_hc_workspace.p,
             direct_hc_coefficients.p, stream);
    hc_pre_rms_norm(streams.p, direct_hc_coefficients.p, kHcCoef, attn_norm.p, 1, Embd, hc_args.rms_eps,
                    direct_collapsed.p, direct_normalized.p, stream);
    SparseMqaArgs direct_selection = selection;
    direct_selection.pos = direct.pos.p;
    direct_selection.win_start = direct.win_start.p;
    direct_selection.win_n = direct.win_n.p;
    AttentionStepScratch direct_scratch{direct.input_q8.p, direct.q_latent.p, direct.q_latent_q8.p, direct.q_heads.p,
                                        direct.kv_projected.p, direct.kv_row.p, direct.mqa_output.p,
                                        direct.mqa_workspace.p, direct.output_q8.p, direct.output_latent.p, direct.output.p};
    attention_step(direct_normalized.p, weights, rope, direct.ring.p, direct_selection, direct_scratch, stream);
    hc_post(direct.output.p, streams.p, direct_hc_coefficients.p, kHcCoef, 1, Embd, expected_streams.p, stream);
    check(cudaStreamSynchronize(stream), "synchronize attention-layer composition");
    const bool layer_output_equal = next_streams.down() == expected_streams.down();
    const bool layer_ring_equal = actual.ring.down() == direct.ring.down();
    std::printf("DeepSeek4 attention sublayer composition: streams %s; cache %s; mHC and attention primitives composed\n",
                layer_output_equal ? "bitwise" : "DIFFERS", layer_ring_equal ? "bitwise" : "DIFFERS");
    std::printf("DeepSeek4 one-token attention step: output %s; ring state %s; Q8_0 checkpoint geometry %d/%d/%d\n",
                output_equal ? "bitwise" : "DIFFERS", ring_equal ? "bitwise" : "DIFFERS", Embd, Rank, HeadsDim);
    cudaStreamDestroy(stream);
    return output_equal && ring_equal && layer_output_equal && layer_ring_equal ? 0 : 1;
}
