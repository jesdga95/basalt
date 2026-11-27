#pragma once

#include "models/deepseek4/pack.hpp"
#include "basalt/model/adapter.hpp"
#include "basalt/models/deepseek4/layer.hpp"
#include "basalt/models/deepseek4/head.hpp"

#include <map>
#include <string>
#include <vector>

namespace basalt::core {
class LoadPlan;
struct ExpertTierSpec;
}

namespace basalt::models::deepseek4 {

struct DeviceTensor {
    void* ptr = nullptr;
    int type = 0;
    uint64_t bytes = 0;
    std::vector<uint64_t> shape;
};

/// Owns DeepSeek4's validated pack and the dense tensors queued for the start's single LoadPlan pass.
class ModelWeights {
public:
    ModelWeights() = default;
    ~ModelWeights();
    ModelWeights(const ModelWeights&) = delete;
    ModelWeights& operator=(const ModelWeights&) = delete;

    bool plan(const std::string& path, int device, core::LoadPlan& reads, std::string& err);
    const Pack& pack() const { return pack_; }
    const Geometry& geo() const { return pack_.geo; }
    int device() const { return device_; }
    const DeviceTensor* tensor(const std::string& name) const;
    const DeviceTensor* tensor(int layer, const char* suffix) const;
    bool attention_layer_weights(int layer, AttentionLayerWeights& weights, std::string& err) const;
    bool ffn_layer_weights(int layer, FfnRouteWeights& route, SharedExpertWeights& shared, std::string& err) const;
    bool head_weights(HeadWeights& weights, std::string& err) const;
    bool embedding_row(int32_t token, std::vector<float>& row, std::string& err) const;
    /// Ask the OS to read the token table in: no first-token page fault once every expert is resident in RAM.
    void warm_token_table() const;
    bool hash_route_table(int layer, std::vector<int32_t>& ids, std::string& err) const;
    /// Read one validated expert blob from the pack into caller-owned staging memory.
    bool read_expert_blob(int layer, int expert, void* dst, size_t bytes, std::string& err) const;
    bool expert_tier_spec(const model::LoadRequest& request, core::ExpertTierSpec& spec, std::string& err) const;

private:
    void reset();

    Pack pack_;
    std::map<std::string, DeviceTensor> tensors_;
    std::string path_;
    int fd_ = -1;
    int device_ = -1;
};

}  // namespace basalt::models::deepseek4
