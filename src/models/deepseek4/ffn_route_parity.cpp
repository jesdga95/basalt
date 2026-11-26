#include "basalt/models/deepseek4/layer.hpp"
#include "basalt/kernels/row_norm.hpp"
#include "parity_util.hpp"

#include <cstring>
#include <string>
#include <vector>

using namespace ds4test;

int main() {
    using namespace basalt::kernels;
    using namespace basalt::models::deepseek4;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;

    constexpr int Embd = 4096, Experts = 256, Tokens = 8, VocabRows = 4;
    const auto streams = randn((size_t) Tokens * 4 * Embd, 0x46524f31, 0.25f);
    const auto hc_fn = randn((size_t) kHcCoef * 4 * Embd, 0x46524f32, 0.0003f);
    const std::vector<float> hc_scale = {0.75f, 0.9f, 0.65f};
    const auto hc_base = randn(kHcCoef, 0x46524f33, 0.2f);
    const std::vector<float> norm((size_t) Embd, 1.f);
    const auto gate = randn((size_t) Experts * Embd, 0x46524f34, 0.0005f);
    const auto bias = randn(Experts, 0x46524f35, 0.1f);
    const auto bias_vl = randn(Experts, 0x46524f36, 0.1f);

    std::vector<int32_t> tokens(Tokens);
    for (int t = 0; t < Tokens; ++t) tokens[(size_t) t] = t;
    RouteRows routes;
    std::string err;
    if (!build_route_rows(tokens.data(), Tokens, VocabRows, 2, 43, 3, routes, err)) {
        basalt::log::error("basalt deepseek4 FFN route parity", "%s", err.c_str());
        return 1;
    }
    std::vector<int32_t> table((size_t) VocabRows * 6);
    for (int row = 0; row < VocabRows; ++row)
        for (int k = 0; k < 6; ++k) table[(size_t) row * 6 + k] = (row * 17 + k * 43) % Experts;

    Dev<float> d_streams(streams), d_hc_fn(hc_fn), d_hc_scale(hc_scale), d_hc_base(hc_base), d_norm(norm), d_gate(gate);
    Dev<float> d_bias(bias), d_bias_vl(bias_vl), d_hc_ws(hc_mixes_workspace_floats(Tokens));
    Dev<float> d_coef((size_t) Tokens * kHcCoef), d_collapsed((size_t) Tokens * Embd), d_normalized((size_t) Tokens * Embd);
    Dev<float> d_logits((size_t) Tokens * Experts), d_weights((size_t) Tokens * 6);
    Dev<int32_t> d_ids((size_t) Tokens * 6), d_table(table), d_fixed_rows(routes.fixed_rows);
    Dev<uint8_t> d_fixed_mask(routes.fixed_mask), d_alternate_mask(routes.alternate_mask);

    FfnRouteWeights weights;
    weights.hc = {Embd, 1e-20f, 1e-6f, 20};
    weights.hc_fn = d_hc_fn.p;
    weights.hc_scale = d_hc_scale.p;
    weights.hc_base = d_hc_base.p;
    weights.norm = d_norm.p;
    weights.gate = d_gate.p;
    weights.routing = router_mixed_args(Experts, VocabRows, d_table.p, d_fixed_rows.p,
                                        d_fixed_mask.p, d_alternate_mask.p, d_bias.p, d_bias_vl.p);
    FfnRouteScratch scratch{d_hc_ws.p, d_coef.p, d_collapsed.p, d_normalized.p, d_logits.p, d_ids.p, d_weights.p};
    prepare_ffn_routes(d_streams.p, Tokens, weights, scratch, nullptr);

    Dev<float> ref_hc_ws(hc_mixes_workspace_floats(Tokens)), ref_coef((size_t) Tokens * kHcCoef);
    Dev<float> ref_collapsed((size_t) Tokens * Embd), ref_normalized((size_t) Tokens * Embd);
    Dev<float> ref_logits((size_t) Tokens * Experts), ref_weights((size_t) Tokens * 6);
    Dev<int32_t> ref_ids((size_t) Tokens * 6);
    hc_mixes(d_streams.p, d_hc_fn.p, d_hc_scale.p, d_hc_base.p, Tokens, weights.hc, ref_hc_ws.p, ref_coef.p, nullptr);
    hc_pre_rms_norm(d_streams.p, ref_coef.p, kHcCoef, d_norm.p, Tokens, Embd, weights.hc.rms_eps,
                    ref_collapsed.p, ref_normalized.p, nullptr);
    router_step(ref_normalized.p, d_gate.p, Tokens, Embd, weights.routing, ref_logits.p, ref_ids.p, ref_weights.p, nullptr);

    const bool norm_equal = d_normalized.down() == ref_normalized.down();
    const bool ids_equal = d_ids.down() == ref_ids.down();
    const auto got_weights = d_weights.down();
    const auto expected_weights = ref_weights.down();
    const bool weights_equal = got_weights == expected_weights;
    std::printf("DeepSeek4 FFN route prep T=%d: normalized rows %s, expert ids %s, route weights %s\n",
                Tokens, norm_equal ? "bitwise" : "DIFFERS", ids_equal ? "bitwise" : "DIFFERS",
                weights_equal ? "bitwise" : "DIFFERS");
    return norm_equal && ids_equal && weights_equal ? 0 : 1;
}
