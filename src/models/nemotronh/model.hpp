// src/models/nemotronh/model.hpp - a loaded Nemotron-H pack: the dense tensors on the main card, the token embedding in
// mapped host memory, the routed experts in the expert tier (VRAM slots, the pinned RAM tier, the CPU pool).
#pragma once

#include "basalt/core/expert_tier.hpp"
#include "basalt/model/adapter.hpp"
#include "models/nemotronh/pack.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace basalt::models::nemotronh {

constexpr int64_t kDefaultContext = 8192;
constexpr int kTypeF32 = 0, kTypeBf16 = 30, kTypeF8 = 44, kTypeNvfp4Planes = 45;

/// One projection on the card: `rows` outputs of `cols` inputs in `type`'s layout (F8_E4M3 codes, NVFP4_PLANES or
/// BF16), its FP32 tensor scale (device, null for 1).
struct Mat {
    int type = -1;
    const uint8_t* w = nullptr;
    int64_t rows = 0, cols = 0;
    const float* scale = nullptr;
};

struct Layer {
    Kind kind = Kind::Mamba;
    const float* norm = nullptr;
    // Mamba-2: in_proj rows [z | xBC | dt], the convolution [conv_dim][4] and its bias, dt bias, A = -exp(A_log), D, the
    // gated group norm's weight
    Mat in, out;
    const float *conv_w = nullptr, *conv_b = nullptr, *dt_bias = nullptr, *A = nullptr, *D = nullptr, *ssm_norm = nullptr;
    // attention
    Mat q, k, v, o;
    // MoE: the router (F32 [n_expert][n_embd]) and its selection bias, the latent projections, the shared expert
    const float *router = nullptr, *router_bias = nullptr;
    Mat lat_down, lat_up, sh_up, sh_down;
    int tier_layer = -1;   ///< the expert tier's layer of a MoE layer
};

/// The MTP head's own tensors: eh_proj over [enorm(embedding) | hnorm(hidden)] (BF16 [n_embd][2 n_embd]), the two norms
/// and the norm before the output head; its attention and MoE layers are layer(n_layers) and layer(n_layers + 1).
struct MtpHead {
    Mat eh;
    const float *enorm = nullptr, *hnorm = nullptr, *final_norm = nullptr;
};
/// The most second cards whose expert tiers a start uses (gpu-mode peer takes the main card and up to these).
constexpr int kMaxSecondCards = 3;
/// The piece the card's expert slots are opened in, so a long prompt borrows only what its expert ring needs.
constexpr uint64_t kExpertLendSegment = 256ull << 20;

class NemotronModel final : public model::LoadedModel {
public:
    NemotronModel();
    ~NemotronModel() override;
    model::Result plan(const model::LoadRequest& req, core::LoadPlan& reads,
                       std::vector<std::unique_ptr<model::ModelSession>>& sessions) override;
    model::Result attach(const model::LoadRequest& req, std::vector<std::unique_ptr<model::ModelSession>>& sessions) override;
    const model::ModelDesc& desc() const override { return desc_; }
    int64_t context() const override { return ctx_; }
    /// gpu-mode peer: the second cards hold blocks of the expert tier's layers (model/peer_hooks.hpp).
    model::PeerHooks* peer() const override;

    const Pack& pack() const { return pack_; }
    const Geometry& geo() const { return pack_.geo; }
    const Layer& layer(int l) const { return layers_[(size_t) l]; }   ///< l < geo().all_layers()
    bool has_mtp() const { return !pack_.geo.mtp_kinds.empty(); }
    const MtpHead& mtp() const { return mtp_; }
    const uint16_t* embedding() const { return embd_dev_; }   ///< BF16 [n_vocab][n_embd], mapped host memory
    const float* out_norm() const { return out_norm_; }
    const Mat& head() const { return head_; }
    int device() const { return device_; }
    int fd() const { return fd_; }   ///< the pack, open: the prompt path reads experts only the file holds
    int sessions() const { return flags_.sessions > 0 ? flags_.sessions : 1; }
    const model::LoadRequest& flags() const { return flags_; }

    /// The tier's descriptor: the MoE layers only (the MTP head's last), over the latent the experts run in.
    core::ExpertTierSpec tier_spec() const;
    /// The expert tier and the pinned rows the CPU pool reads and writes: one per model, whatever the sessions.
    struct SharedTier {
        std::shared_ptr<core::ExpertTier> tier;
        float* h_x = nullptr;      ///< [rows][n_latent] the latent rows of a group, pinned and mapped
        float* h_rows = nullptr;   ///< [rows * n_used][n_latent] the CPU's expert rows, pinned and mapped
    };
    SharedTier& shared_tier() const { return shared_tier_; }
    /// The start's reads while it plans (null after): what the first session's tier queues into.
    core::LoadPlan* planning() const { return planning_; }
    /// Bytes between two staged experts of the prompt path: the blob and what MMQ reads past it, in whole NVFP4 blocks.
    size_t expert_stride() const;

private:
    Pack pack_;
    model::ModelDesc desc_;
    std::vector<Layer> layers_;
    std::vector<void*> dev_;   ///< every device allocation, freed with the model
    uint16_t* embd_host_ = nullptr;
    uint16_t* embd_dev_ = nullptr;
    const float* out_norm_ = nullptr;
    Mat head_;
    MtpHead mtp_;
    model::LoadRequest flags_;
    std::string path_;
    int device_ = 0, fd_ = -1;
    int64_t ctx_ = kDefaultContext;
    core::LoadPlan* planning_ = nullptr;
    mutable SharedTier shared_tier_;
};

/// A session over `model`; null with `err` set.
std::unique_ptr<model::ModelSession> make_session(NemotronModel& model, std::string& err);

}  // namespace basalt::models::nemotronh
