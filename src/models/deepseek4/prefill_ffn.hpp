#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace basalt::core { class ExpertTier; }

namespace basalt::models::deepseek4 {

class ModelWeights;

/// Chunk-sized MMQ products for DeepSeek4's shared and routed experts.
class PrefillFfn {
public:
    virtual ~PrefillFfn() = default;
    virtual bool routed(int layer, int rows, const float* x, const int32_t* ids, const float* weights,
                        float* out, void* stream, std::string& err) = 0;
    virtual bool shared(int layer, int rows, const float* x, float* out, void* stream, std::string& err) = 0;
};

std::unique_ptr<PrefillFfn> make_prefill_ffn(ModelWeights& weights, core::ExpertTier& tier, int row_cap,
                                              std::string& err);

}  // namespace basalt::models::deepseek4
