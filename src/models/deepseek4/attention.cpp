#include "basalt/models/deepseek4/attention.hpp"

#include "basalt/kernels/native_mmvq.hpp"

#include <stdexcept>

namespace basalt::models::deepseek4 {

size_t attention_kv_scratch_bytes(int n_tokens) {
    if (n_tokens < 1 || n_tokens > 8) throw std::invalid_argument("deepseek4 attention_kv_scratch_bytes: token count");
    return kernels::native_q8_1_bytes(4096, n_tokens);
}

void project_attention_kv(const float* x, int wkv_type, const void* wkv, const float* norm, float rms_eps,
                          const int32_t* positions, const kernels::MqaRopeTable& rope, int n_tokens,
                          void* q8_scratch, float* projected, float* out, void* stream) {
    attention_kv_scratch_bytes(n_tokens);
    if (!x || !wkv || !norm || !positions || !q8_scratch || !projected || !out || !stream ||
        !kernels::native_mmvq_supported(wkv_type))
        throw std::invalid_argument("deepseek4 project_attention_kv: invalid arguments or weight type");
    (void) kernels::native_mmvq_weight_bytes(wkv_type, 4096, 512);
    kernels::native_quantize_q8_1(x, q8_scratch, 4096, n_tokens, stream);
    kernels::native_mmvq(wkv_type, wkv, q8_scratch, projected, 4096, 512, n_tokens, stream);
    prepare_attention_kv(projected, norm, rms_eps, positions, rope, n_tokens, out, stream);
}

}  // namespace basalt::models::deepseek4
