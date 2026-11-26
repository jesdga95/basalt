#include "basalt/models/deepseek4/rope.hpp"

#include "basalt/kernels/rope_scaling.hpp"

#include <cmath>

namespace basalt::models::deepseek4 {

bool make_rope_tables(const RopeParams& p, int compress_ratio, RopeTables& tables, std::string& err) {
    if ((compress_ratio != 0 && compress_ratio != 4 && compress_ratio != 128) || p.rope_dim != 64 ||
        !std::isfinite(p.rope_base) || p.rope_base <= 1.f || !std::isfinite(p.compressed_base) ||
        p.compressed_base <= 1.f || !std::isfinite(p.factor) || p.factor < 1.f || p.original_context <= 0 ||
        !std::isfinite(p.beta_fast) || p.beta_fast <= 0.f || !std::isfinite(p.beta_slow) || p.beta_slow <= 0.f) {
        err = "deepseek4 RoPE: invalid geometry or compression ratio";
        return false;
    }

    const bool yarn = compress_ratio != 0;
    const double base = yarn ? (double) p.compressed_base : (double) p.rope_base;
    kernels::RopeScaling scaling;
    scaling.type = yarn ? kernels::RopeScalingType::YaRN : kernels::RopeScalingType::None;
    scaling.freq_base = base;
    scaling.factor = yarn ? (double) p.factor : 1.0;
    scaling.orig_ctx = yarn ? (double) p.original_context : 1.0;
    scaling.ext_factor = yarn ? 1.0 : 0.0;
    scaling.attn_factor = 1.0;
    scaling.beta_fast = (double) p.beta_fast;
    scaling.beta_slow = (double) p.beta_slow;
    if (const char* why = kernels::rope_scaling_invalid(scaling)) {
        err = std::string("deepseek4 RoPE: ") + why;
        return false;
    }
    const kernels::RopeKernelArgs args = scaling.kernel_args(p.rope_dim);

    RopeTables result;
    for (int pair = 0; pair < kRopePairs; ++pair) {
        const float inverse_frequency = std::pow((float) base, -2.f * (float) pair / (float) p.rope_dim);
        float frequency_scale = 1.f;
        if (yarn) {
            const float correction = kernels::rope_yarn_ramp(args.corr_low, args.corr_high, pair) * args.ext_factor;
            frequency_scale = args.freq_scale * (1.f - correction) + correction;
        }
        result.frequency[(size_t) pair] = inverse_frequency * frequency_scale;
        result.attention.step[pair] = result.frequency[(size_t) pair];
    }
    // DeepSeek's reference multiplies by unit-magnitude complex phases and has no YaRN magnitude correction.
    result.attention.mscale = 1.f;
    tables = result;
    err.clear();
    return true;
}

}  // namespace basalt::models::deepseek4
