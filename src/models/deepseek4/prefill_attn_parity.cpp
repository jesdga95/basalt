#include "basalt/models/deepseek4/prefill_attn.hpp"
#include "basalt/models/deepseek4/attention.hpp"
#include "basalt/kernels/sparse_mqa.hpp"
#include "parity_util.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace basalt::models::deepseek4;
using ds4test::Dev;

namespace {

bool run() {
    constexpr int T = 4;
    constexpr int W = 16;
    constexpr int S = 8;
    constexpr int H = 64;
    constexpr int D = 512;

    auto q = ds4test::randn((size_t) T * H * D, 0x50415431u, 0.2f);
    auto sinks = ds4test::randn(H, 0x50415432u, 0.05f);
    auto win_f32 = ds4test::randn((size_t) W * D, 0x50415433u, 0.2f);
    auto comp_f32 = ds4test::randn((size_t) S * D, 0x50415434u, 0.2f);
    std::vector<float> rope(128, 1.0f);
    for (int i = 0; i < 64; ++i) {
        rope[2 * i] = std::cos(0.01f * i);
        rope[2 * i + 1] = std::sin(0.01f * i);
    }

    std::vector<int32_t> pos{10, 11, 12, 13};
    std::vector<int32_t> wstart{0, 0, 0, 0};
    std::vector<int32_t> wn{11, 12, 13, 14};
    std::vector<int32_t> clim{4, 4, 4, 4};
    std::vector<int32_t> sel((size_t) T * S);
    for (int t = 0; t < T; ++t)
        for (int s = 0; s < S; ++s) sel[(size_t) t * S + s] = s;

    Dev<float> d_q(q), d_sinks(sinks), d_rope(rope);
    Dev<float> d_out_ref((size_t) T * H * D), d_out_tc((size_t) T * H * D);
    Dev<int32_t> d_pos(pos), d_ws(wstart), d_wn(wn), d_clim(clim), d_sel(sel);
    Dev<uint16_t> d_win_fp16((size_t) W * D);
    Dev<uint16_t> d_comp_fp16((size_t) S * D);
    std::vector<float> zero_ws((size_t) T * H * 64 * (D + 2), 0.f);
    Dev<float> d_ws_ref(basalt::kernels::sparse_mqa_workspace_floats(T, S));
    Dev<float> d_ws_tc(zero_ws);

    std::vector<uint16_t> win_h(win_f32.size()), comp_h(comp_f32.size());
    for (size_t i = 0; i < win_f32.size(); ++i) {
        const __half h = __float2half_rn(win_f32[i]);
        std::memcpy(&win_h[i], &h, sizeof(uint16_t));
    }
    for (size_t i = 0; i < comp_f32.size(); ++i) {
        const __half h = __float2half_rn(comp_f32[i]);
        std::memcpy(&comp_h[i], &h, sizeof(uint16_t));
    }
    d_win_fp16.up(win_h);
    d_comp_fp16.up(comp_h);

    basalt::kernels::SparseMqaArgs ref{};
    ref.n_tokens = T;
    ref.win = d_win_fp16.p;
    ref.win_type = basalt::kernels::MqaRowType::F16;
    ref.win_cap = W;
    ref.win_start = d_ws.p;
    ref.win_n = d_wn.p;
    ref.pos = d_pos.p;
    ref.rope = {};
    ref.n_sel = S;
    ref.sel = d_sel.p;
    ref.comp = d_comp_fp16.p;
    ref.comp_type = basalt::kernels::MqaRowType::F16;
    ref.comp_limit = d_clim.p;
    ref.q = d_q.p;
    ref.sinks = d_sinks.p;
    ref.round_output_bf16 = true;
    ref.out = d_out_ref.p;
    basalt::kernels::sparse_mqa(ref, d_ws_ref.p, nullptr);

    PrefillMqaArgs tc{};
    tc.n_tokens = T;
    tc.win = d_win_fp16.p;
    tc.win_type = CacheFormat::Fp16;
    tc.win_cap = W;
    tc.win_start = d_ws.p;
    tc.win_n = d_wn.p;
    tc.pos = d_pos.p;
    tc.rope = {};
    tc.n_sel = S;
    tc.sel = d_sel.p;
    tc.comp = d_comp_fp16.p;
    tc.comp_type = CacheFormat::Fp16;
    tc.comp_limit = d_clim.p;
    tc.q = d_q.p;
    tc.sinks = d_sinks.p;
    tc.round_output_bf16 = true;
    tc.out = d_out_tc.p;
    sparse_mqa_tc(tc, d_ws_tc.p, nullptr);

    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "ds4_prefill_attn_parity: CUDA sync failed\n");
        return false;
    }

    const auto got_ref = d_out_ref.down();
    const auto got_tc = d_out_tc.down();
    const auto e = ds4test::compare(got_tc, got_ref);
    std::printf("ds4_prefill_attn_parity (Fp16, T=%d): max_abs=%.3e\n", T, e.max_abs);
    if (e.max_abs > 1e-2) {
        std::fprintf(stderr, "ds4_prefill_attn_parity failed: max_abs=%.6g\n", e.max_abs);
        return false;
    }
    return true;
}

}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    return run() ? 0 : 1;
}
