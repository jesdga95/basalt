// src/models/deepseek41/layer_ops.cpp - the default layer ops and the factory that picks the implementation.
#include "models/deepseek41/layer_ops.hpp"


namespace basalt::models::deepseek41 {

bool LayerOps::attention_batch(int layer, int64_t pos0, int n_tokens, const int32_t* tokens, const float* x, float* out,
                               int n_embd, void* stream, std::string& err) {
    for (int t = 0; t < n_tokens; ++t) {
        begin_token(pos0 + t);
        if (!attention(layer, pos0 + t, tokens[t], x + (size_t) t * n_embd, out + (size_t) t * n_embd, stream, err))
            return false;
    }
    return true;
}

bool LayerOps::routed_and_shared_staged(int, const float*, const int32_t*, const float*, float*, float*, void*,
                                        std::string& err) {
    err = "deepseek41: this layer ops implementation has no overlapped shared expert";
    return false;
}

bool LayerOps::routed_experts_batch(int layer, int64_t pos0, int n_tokens, const float* x, const int32_t* ids,
                                    const float* weights, int k, float* out, int n_embd, void* stream,
                                    std::string& err) {
    for (int t = 0; t < n_tokens; ++t) {
        begin_token(pos0 + t);
        if (!routed_experts(layer, x + (size_t) t * n_embd, ids + (size_t) t * k, weights + (size_t) t * k,
                            out + (size_t) t * n_embd, stream, err))
            return false;
    }
    return true;
}

bool LayerOps::shared_expert_batch(int layer, int64_t pos0, int n_tokens, const float* x, float* out, int n_embd,
                                   void* stream, std::string& err) {
    for (int t = 0; t < n_tokens; ++t) {
        begin_token(pos0 + t);
        if (!shared_expert(layer, x + (size_t) t * n_embd, out + (size_t) t * n_embd, stream, err)) return false;
    }
    return true;
}

namespace {

class UnavailableOps final : public LayerOps {
public:
    bool attention(int, int64_t, int32_t, const float*, float*, void*, std::string& err) override {
        err = "deepseek41: the attention kernels are not wired yet";
        return false;
    }
    bool routed_experts(int, const float*, const int32_t*, const float*, float*, void*, std::string& err) override {
        err = "deepseek41: the routed expert kernels are not wired yet";
        return false;
    }
    bool shared_expert(int, const float*, float*, void*, std::string& err) override {
        err = "deepseek41: the shared expert kernel is not wired yet";
        return false;
    }
};

}  // namespace

std::unique_ptr<LayerOps> make_unavailable_ops() { return std::make_unique<UnavailableOps>(); }

std::unique_ptr<LayerOps> make_attention_ops(const DsModel& model, std::unique_ptr<LayerOps> experts, std::string& err);

std::unique_ptr<LayerOps> make_kernel_ops(const DsModel& model, const std::string& fixture, std::string& err) {
    std::unique_ptr<LayerOps> experts;
    if (!fixture.empty()) {
        experts = make_fixture_ops(model, fixture, err);
        if (!experts) return nullptr;
    }
    return make_attention_ops(model, std::move(experts), err);
}


}  // namespace basalt::models::deepseek41
