#pragma once

#include "basalt/core/expert_tier.hpp"
#include "basalt/model/adapter.hpp"
#include "basalt/model/batch_driver.hpp"
#include "basalt/vision/transformer.hpp"
#include "models/deepseek4/model_weights.hpp"

#include <memory>
#include <string>

namespace basalt::models::deepseek4 {

class DsparkDrafter;

class DsModel final : public model::LoadedModel {
public:
    ~DsModel() override;

    const model::ModelDesc& desc() const override { return desc_; }
    int64_t context() const override { return context_; }
    model::Result plan(const model::LoadRequest& request, core::LoadPlan& reads,
                       std::vector<std::unique_ptr<model::ModelSession>>& sessions) override;
    model::Result attach(const model::LoadRequest& request,
                         std::vector<std::unique_ptr<model::ModelSession>>& sessions) override;

    ModelWeights& weights() { return weights_; }
    const ModelWeights& weights() const { return weights_; }
    core::ExpertTier* tier() const { return tier_.get(); }
    DsparkDrafter* dspark() const { return dspark_.get(); }
    /// The image encoders' cuBLAS handle, opened by the expert tier at load when images are on.
    basalt::vision::TransformerBlas& vision_blas() const { return vision_blas_; }
    int spec() const { return request_.spec; }
    bool vision_enabled() const { return request_.vision; }
    int diag_rows() const { return request_.diag_rows; }
    model::Result ensure_tier();
    model::BatchShared& batch_shared() { return batch_shared_; }

private:
    ModelWeights weights_;
    model::ModelDesc desc_;
    model::LoadRequest request_;
    model::BatchShared batch_shared_;
    core::LoadPlan* planning_ = nullptr;
    std::shared_ptr<core::ExpertTier> tier_;
    mutable basalt::vision::TransformerBlas vision_blas_;
    std::unique_ptr<DsparkDrafter> dspark_;
    float* h_expert_input_ = nullptr;
    float* h_expert_rows_ = nullptr;
    int device_ = -1;
    int64_t context_ = 0;
};

std::unique_ptr<model::ModelSession> make_session_impl(DsModel& model, std::string& err);

}  // namespace basalt::models::deepseek4
