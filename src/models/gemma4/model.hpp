// src/models/gemma4/model.hpp - a loaded Gemma 4 pack: every weight on the main card (the projections FP8 E4M3 with a
// scale per matrix, or the MLP's NVFP4 planes, the experts of a layer in one run of NVFP4 blobs, the head FP8 rows with a
// scale per row), the token embedding in mapped host memory.
#pragma once

#include "basalt/core/expert_tier.hpp"
#include "basalt/model/adapter.hpp"
#include "basalt/kernels/window_gemm.hpp"
#include "basalt/model/batch_driver.hpp"
#include "basalt/models/gemma4/kernels.hpp"
#include "models/gemma4/pack.hpp"
#include "models/gemma4/vision_encoder.hpp"   // [vision]

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace basalt::models::gemma4 {

constexpr int64_t kDefaultContext = 8192;
/// Rows of a prompt chunk (prompt.hpp): a window layer's ring holds its window and a chunk.
constexpr int kChunk = 2048;

/// A projection [rows][K]: FP8 E4M3 codes, or NVFP4 planes (kernels/nvfp4_planes.hpp), and its device FP32 scale (per
/// matrix, or per row for the FP8 head).
struct Mat {
    const uint8_t* w = nullptr;
    const float* scale = nullptr;
    int64_t rows = 0, K = 0;
    basalt::kernels::wgemm::Format fmt = basalt::kernels::wgemm::Format::F8;
};

struct LayerWeights {
    LayerShape sh;
    RopeTab rope;                         ///< the layer kind's rotary tables
    const float *attn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr, *post_attn = nullptr;
    const float *ffn_norm = nullptr, *post_ffw = nullptr, *scalar = nullptr;
    Mat q, k, v, o, gate, up, down;       ///< v empty on a layer whose keys are its values; the MLP's three FP8 or NVFP4
    // the experts (a layer with experts; core::ExpertTier keeps their blobs, experts.hpp their tables)
    const float *post_ffw1 = nullptr, *post_ffw2 = nullptr, *pre_ffw2 = nullptr;
    const float* router = nullptr;        ///< F32 [n_expert][n_embd], its input scale folded in
    size_t blob_bytes = 0;
    int gu_type = 0, d_type = 0;
};

/// The room of the tier's slots: one in every layer, and the largest (256-byte aligned, as the tier lays them out).
struct ExpertRoom {
    uint64_t one_each = 0, slot = 0;
};

/// One layer of the assistant drafter: its queries read the keys and values of the model's last layer of its kind
/// (`source`), at the anchor's position.
struct DrafterLayer {
    LayerShape sh;
    int source = 0;
    RopeTab rope;
    const float *attn_norm = nullptr, *q_norm = nullptr, *post_attn = nullptr, *ffn_norm = nullptr, *post_ffw = nullptr;
    const float* scalar = nullptr;
    Mat q, o, gate, up, down;
};

/// Google's assistant drafter (gemma4.assistant.*): [the anchor's embedding | the model's last normed state] projected
/// to its width, its layers, its head over the vocabulary, and the state it hands its next step.
struct DrafterWeights {
    int n_embd = 0;
    int64_t n_ff_pad = 0;
    Mat pre, post, head;
    const float* out_norm = nullptr;
    std::vector<DrafterLayer> layers;
};

class PromptPath;
class TierExperts;

class Gemma4Model final : public model::LoadedModel {
public:
    Gemma4Model();
    ~Gemma4Model() override;
    model::Result plan(const model::LoadRequest& req, core::LoadPlan& reads,
                       std::vector<std::unique_ptr<model::ModelSession>>& sessions) override;
    model::Result attach(const model::LoadRequest& req, std::vector<std::unique_ptr<model::ModelSession>>& sessions) override;
    const model::ModelDesc& desc() const override { return desc_; }
    int64_t context() const override { return ctx_; }

    const Pack& pack() const { return pack_; }
    const Geometry& geo() const { return pack_.geo; }
    bool mxfp8_experts() const { return mxfp8_experts_; }   ///< some layer's experts are MXFP8
    const LayerWeights& layer(int l) const { return layers_[(size_t) l]; }
    const uint16_t* embedding() const { return embd_dev_; }   ///< BF16 [n_vocab][n_embd], mapped host memory
    const Mat& head() const { return head_; }               ///< rows n_vocab, a scale per row
    const float* output_norm() const { return output_norm_; }
    const float* ones() const { return ones_; }               ///< [n_embd] of 1: the router input's unweighted norm
    int64_t n_ff_pad() const { return n_ff_pad_; }            ///< the MLP's inner width as the products read it
    bool mlp_nvfp4() const { return mlp_nvfp4_; }             ///< the MLP's projections are NVFP4 planes (else FP8)
    KvType kv_type() const { return kv_; }
    int device() const { return device_; }
    int sessions() const { return flags_.sessions > 0 ? flags_.sessions : 1; }
    const model::LoadRequest& flags() const { return flags_; }
    /// The assistant drafter; null without one in the pack or with --spec below 2.
    const DrafterWeights* drafter() const { return drafter_.get(); }
    /// What the sessions share (session.cpp's Shared): made by the first, gone with the last.
    std::weak_ptr<void>& shared() { return shared_; }
    /// [vision] The image encoder: staged with --vision, else it refuses pictures. One per model; ENC runs between requests.
    ImageEncoder& encoder() const { return vis_; }
    /// [vision] The sessions' prompt path (session.cpp's shared scratch registers it, null once it is gone): its buffers are
    /// the room the image encoder borrows.
    void set_prompt(PromptPath* p) const { prompt_ = p; }
    /// The routed experts' placement (null for a dense model) and their tables and hand-off (experts.hpp).
    core::ExpertTier* tier() const { return tier_.get(); }
    TierExperts* experts() const { return experts_.get(); }
    /// A prompt's staging of the experts off the card: the experts a layer's buffer holds (0: every expert is on the card)
    /// and the bytes of each.
    int stage_experts() const { return stage_experts_; }
    uint64_t stage_slot() const { return stage_slot_; }

private:
    bool mxfp8_experts_ = false;
    // [vision] the image encoder's lender: the prompt path's buffers go back to the driver while pictures are read
    bool lend_room(uint64_t bytes, std::string& err);
    void refill();
    mutable PromptPath* prompt_ = nullptr;
    bool parked_ = false;
    uint64_t card_spare_ = 0;   ///< the free VRAM the plan keeps beside the caches
    ExpertRoom expert_room() const;
    bool open_tier(const model::LoadRequest& req, uint64_t need, uint64_t free_b, core::LoadPlan& reads, std::string& err);
    std::unique_ptr<core::ExpertTier> tier_;
    std::unique_ptr<TierExperts> experts_;
    float* tier_rows_ = nullptr;   ///< mapped [kMoePairs][n_embd]: the host's expert rows
    int stage_experts_ = 0;
    uint64_t stage_slot_ = 0;
    float* small(const std::string& name, uint64_t floats, core::LoadPlan& reads, int file, std::string& err);
    /// An FP8 matrix, or with `nvfp4` an NVFP4-planes one; its bytes added to `bytes`.
    bool mat(const std::string& name, int64_t rows, int64_t K, core::LoadPlan& reads, int file, Mat& m, std::string& err,
             bool per_row = false, bool nvfp4 = false);
    void* device_buf(uint64_t bytes);

    Pack pack_;
    model::ModelDesc desc_;
    model::LoadRequest flags_;
    std::vector<LayerWeights> layers_;
    std::vector<void*> dev_;                 ///< every device allocation, freed with the model
    uint16_t* embd_host_ = nullptr;          ///< pinned, mapped
    uint16_t* embd_dev_ = nullptr;
    Mat head_;
    float* output_norm_ = nullptr;
    float* ones_ = nullptr;
    RopeTab rope_swa_, rope_full_;
    int64_t n_ff_pad_ = 0;
    bool mlp_nvfp4_ = false;
    uint64_t weight_bytes_ = 0;              ///< the card's projection and head bytes
    KvType kv_ = KvType::F16;
    int device_ = 0, fd_ = -1;
    int64_t ctx_ = kDefaultContext;
    std::weak_ptr<void> shared_;
    std::unique_ptr<DrafterWeights> drafter_;
    mutable ImageEncoder vis_;   // [vision]
};

/// A session over `model`; null with `err` set.
std::unique_ptr<model::ModelSession> make_session(Gemma4Model& model, std::string& err);

/// The device bytes a session takes for `context` positions: its caches (a window layer's ring of the window and a
/// chunk's rows, a full layer's cells), its logits and graphs.
uint64_t session_bytes(const Geometry& g, int64_t context, KvType kv);
/// The device bytes every session shares once on the card: the forward's scratch and the prompt path's.
uint64_t shared_bytes(const Geometry& g, int64_t n_ff_pad, bool mlp_nvfp4, int64_t context, KvType kv, bool mxfp8);
/// The split-K workspace of the decode window's NVFP4 MLP products (floats of partial sums, tickets); 0 for FP8.
void mlp_workspace(const Geometry& g, int64_t n_ff_pad, bool mlp_nvfp4, size_t& partial, size_t& tickets);

}  // namespace basalt::models::gemma4
