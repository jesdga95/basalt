// src/models/deepseek41/prefill_ffn.hpp - the routed and shared experts over a chunk of prompt tokens.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace basalt::models::deepseek41 {

class DsModel;
class LayerOps;

/// h[r][k] = silu(min(gate, limit)) * clamp(up, -limit, limit) over gu[r][2 n_ff] (gate rows, then up rows).
void swiglu_split_clamped(const float* gu, float* h, int64_t rows, int n_ff, float limit, void* stream);

/// out[t][i] = sum_k w[t][k] * rows[map[t][k]][i], summed in k order.
void weighted_sum_tokens(const float* rows, const int32_t* map, const float* w, int n_tokens, int n_rows, int n, float* out,
                         void* stream);

class PrefillFfn {
public:
    virtual ~PrefillFfn() = default;
    /// out[t] = sum_k weights[t][k] * expert(ids[t][k])(x[t]) for T tokens (x, out: [T][n_embd] device; ids, weights:
    /// [T][k] device). Each expert the chunk routes to is staged once, from the ops' expert tier when it has one.
    virtual bool routed(int layer, int n_tokens, const float* x, const int32_t* ids, const float* weights, float* out,
                        void* stream, std::string& err) = 0;
    /// The layers with routed experts in the chunk about to start, 0.. in order: the tokens each one takes.
    virtual void chunk_layers(const std::vector<int>& tokens) = 0;
    /// The shared expert over T tokens.
    virtual bool shared(int layer, int n_tokens, const float* x, float* out, void* stream, std::string& err) = 0;
};

std::unique_ptr<PrefillFfn> make_prefill_ffn(const DsModel& model, LayerOps& ops, int max_tokens, std::string& err);

}  // namespace basalt::models::deepseek41
