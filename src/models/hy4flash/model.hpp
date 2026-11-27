// src/models/hy4flash/model.hpp - a loaded Hy4-flash pack: the dense tensors on the main card, the token embedding in
// host memory, every routed expert in a RAM tier run by the CPU, and the Engram table read from the file by row.
#pragma once

#include "basalt/core/expert_tier.hpp"
#include "basalt/model/adapter.hpp"
#include "basalt/model/batch_driver.hpp"
#include "basalt/ngram/ple_reader.hpp"
#include "models/hy4flash/dfly.hpp"   // DFLY HOOK
#include "models/hy4flash/pack.hpp"
#include "models/hy4flash/vision_encoder.hpp"   // [vision]

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace basalt::models::hy4flash {

struct DevTensor {
    void* ptr = nullptr;
    int type = 0;
    uint64_t bytes = 0;
    std::vector<uint64_t> shape;
};

/// The most second cards whose expert tiers a start uses (gpu-mode peer takes no more cards than the main card and these).
constexpr int kMaxSecondCards = 3;
constexpr int kMaxWindow = 8;   // rows of a decode window, and of one expert group
constexpr int64_t kDefaultContext = 8192;
constexpr uint64_t kExpertLendSegment = 256ull << 20;

class Hy4Model final : public model::LoadedModel {
public:
    Hy4Model();
    ~Hy4Model() override;
    model::Result plan(const model::LoadRequest& req, core::LoadPlan& reads,
                       std::vector<std::unique_ptr<model::ModelSession>>& sessions) override;
    model::Result attach(const model::LoadRequest& req, std::vector<std::unique_ptr<model::ModelSession>>& sessions) override;
    const model::ModelDesc& desc() const override { return desc_; }
    int64_t context() const override { return ctx_; }
    /// The second cards are the expert tier's (model/peer_hooks.hpp).
    model::PeerHooks* peer() const override;

    const Pack& pack() const { return pack_; }
    const Geometry& geo() const { return pack_.geo; }
    /// Device tensor `blk.<layer>.<suffix>` or a global name; null when absent.
    const DevTensor* tensor(const std::string& name) const;
    const DevTensor* tensor(int layer, const char* suffix) const { return tensor("blk." + std::to_string(layer) + "." + suffix); }
    /// One token-embedding row, dequantised into `out` (n_embd floats).
    bool embedding_row(int32_t token, float* out) const;
    ngram::PleReader& engram_reader() const { return *engram_; }
    /// The tier's descriptor: the MoE layers only (the tier's layer 0 is the first of them), over the 1536-wide latent.
    core::ExpertTierSpec tier_spec() const;
    /// Bytes between two staged experts of the prompt path: the largest blob rounded to every matrix type's block size, as MMQ
    /// steps from expert to expert in whole blocks (the main card's launches and the second card's share alike).
    size_t expert_stride() const;
    /// The start's reads while it plans (null after): what the first session's tier queues into.
    core::LoadPlan* planning() const { return planning_; }
    /// The expert tier and the pinned rows it shares with a second card: one per model, whatever the sessions.
    struct SharedTier {
        std::shared_ptr<core::ExpertTier> tier;
        float* h_x = nullptr;
        float* h_rows = nullptr;
    };
    SharedTier& shared_tier() const { return shared_tier_; }
    /// What every session's batch windows share (model/batch_driver.hpp).
    model::BatchShared& batch_shared() const { return batch_; }
    /// The forward scratch every session shares (session.cpp's Scratch, allocated by the first session): row buffers, expert
    /// staging and workspaces. Sessions hold only their conversation's state.
    std::shared_ptr<void>& forward_scratch() const { return scratch_; }
    const model::LoadRequest& flags() const { return flags_; }
    const std::string& path() const { return path_; }
    int sessions() const { return flags_.sessions > 0 ? flags_.sessions : 1; }
    int device() const { return device_; }
    /// --kv fp8: the sparse layers' latent rows as FP8 (default fp16).
    bool cache_fp8() const { return fp8_; }
    /// --kv f32 (diagnostic): every latent row as fp32, the reference's own precision.
    bool cache_f32() const { return f32_; }
    /// --diag-rows: the rows of one forward call of a prompt (0: the largest the session takes).
    int diag_rows() const { return diag_rows_; }
    /// DFLY HOOK: the drafter's weights (null: none in the pack, or no drafting) and the verify window --spec asked for.
    const DflyWeights* dfly() const { return dfly_.get(); }
    int spec() const { return spec_; }
    const std::string& diag_dump() const { return diag_dump_; }
    /// [vision] The image encoder: staged with --vision, else it refuses pictures. One per model; ENC runs between requests.
    ImageEncoder& encoder() const { return vis_; }

private:
    Pack pack_;
    model::ModelDesc desc_;
    std::map<std::string, DevTensor> dev_;
    core::LoadPlan* planning_ = nullptr;
    mutable SharedTier shared_tier_;
    mutable model::BatchShared batch_;
    mutable std::shared_ptr<void> scratch_;
    model::GpuCards gpus_;                    ///< the cards the load received
    model::LoadRequest flags_;
    std::string path_;
    std::vector<uint8_t> embd_;
    uint32_t embd_type_ = 0;
    std::string diag_dump_;
    std::unique_ptr<ngram::PleReader> engram_;
    int device_ = 0, fd_ = -1;
    bool fp8_ = false, f32_ = false;
    int diag_rows_ = 0;
    mutable ImageEncoder vis_;   // [vision]
    std::unique_ptr<DflyWeights> dfly_;   // DFLY HOOK
    int spec_ = 0;
    int64_t ctx_ = kDefaultContext;
};

/// A session over `model`; null with `err` set. `secondary` starts dormant (0 physical VRAM for rings/KV/drafter until used).
std::unique_ptr<model::ModelSession> make_session(Hy4Model& model, std::string& err, bool secondary = false);

}  // namespace basalt::models::hy4flash
