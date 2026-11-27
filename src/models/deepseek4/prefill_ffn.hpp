#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace basalt::core { class ExpertTier; }

namespace basalt::models::deepseek4 {

class ModelWeights;

// The prompt read's shape: a few tokens are decode steps; a short prompt reads beside a tier's copies in short chunks;
// a long one lends the tier's VRAM and reads up to a full chunk at a time.
constexpr int kStepPrompt = 12;
constexpr int kShortPrompt = 2048;
constexpr int kShortChunk = 1024;
constexpr int kMaxChunk = 16384;
constexpr int kMinChunk = 1024;
constexpr uint64_t kShortMargin = 256ull << 20;
constexpr uint64_t kLongMargin = 1ull << 30;

/// Chunk-sized MMQ products for DeepSeek4's shared and routed experts.
class PrefillFfn {
public:
    virtual ~PrefillFfn() = default;
    virtual bool routed(int layer, int rows, const float* x, const int32_t* ids, const float* weights,
                        float* out, void* stream, std::string& err) = 0;
    virtual bool shared(int layer, int rows, const float* x, float* out, void* stream, std::string& err) = 0;
    virtual bool dense_mmq(const float* x, int ld_x, int w_type, const void* w, int n_in, int n_out,
                           int rows, float* out, int ld_out, void* stream, std::string& err) = 0;
};

std::unique_ptr<PrefillFfn> make_prefill_ffn(ModelWeights& weights, core::ExpertTier& tier, int row_cap,
                                              std::string& err);

/// The device bytes make_prefill_ffn's buffers take for `row_cap` rows.
size_t prefill_ffn_bytes(const ModelWeights& weights, int row_cap);

/// The device bytes a prompt chunk of `cap` rows takes in all (its buffers and the expert staging beside them), for a
/// session whose context is `context`: the room a tier must leave free beside its homes and copies.
uint64_t prompt_room_bytes(const ModelWeights& weights, int64_t context, int cap);

}  // namespace basalt::models::deepseek4
