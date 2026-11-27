// src/models/glm5next/model.hpp - a loaded glm5-next pack: the dense tensors on the GPU, the routed experts in the
// shared expert tier (core/expert_tier.hpp). Loads through open_pack + the shared loader.
#pragma once

#include "basalt/model/adapter.hpp"
#include "basalt/kernels/iq_kernels.hpp"   // NativeExpertLayout, the packed expert rows the card computes
#include "models/glm5next/pack.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace basalt::core {
class ExpertTier;
struct ExpertTierSpec;
}  // namespace basalt::core

namespace basalt::models::glm5next {

/// Tokens one expert block covers: the routed experts of a window are looked up, fetched, computed and summed this
/// many tokens at a time, which is the tier's own window (core::ExpertTier::kMaxWindow) and keeps the pinned rows the
/// CPU pool writes (kExpertRows x n_used x n_embd floats) small whatever the prompt chunk's width.
inline constexpr int kExpertRows = 8;

struct DevTensor {
    std::string name;   ///< the pack's tensor name, so a diagnostic can say which one it means
    void* ptr = nullptr;
    int type = 0;
    uint64_t bytes = 0;
    std::vector<uint64_t> shape;
};

class GlmModel final : public model::LoadedModel {
public:
    ~GlmModel() override;
    /// Opens the pack and allocates every dense tensor on the device, their reads queued in `reads`. The sessions, caches
    /// and expert tiers are not yet sized (they arrive with the session layer).
    model::Result plan(const model::LoadRequest& req, core::LoadPlan& reads,
                       std::vector<std::unique_ptr<model::ModelSession>>& sessions) override;
    /// Nothing to connect once the dense tensors have landed (no tier or prompt stage yet).
    model::Result attach(const model::LoadRequest& req,
                         std::vector<std::unique_ptr<model::ModelSession>>& sessions) override;

    const model::ModelDesc& desc() const override { return desc_; }
    const Pack& pack() const { return pack_; }
    const Geometry& geo() const { return pack_.geo; }
    /// Device tensor `blk.<layer>.<suffix>` or a global name; null when absent.
    const DevTensor* tensor(const std::string& name) const;
    /// Every dense tensor the loader put on the device, by name (the forward pass checks their types against what it
    /// can multiply before it runs).
    const std::map<std::string, DevTensor>& dev_tensors() const { return dev_; }
    const DevTensor* tensor(int layer, const char* suffix) const {
        return tensor("blk." + std::to_string(layer) + "." + suffix);
    }
    /// The CUDA device the dense tensors live on.
    int device() const { return device_; }
    /// What the forward pass holds on the card outside a window (see `forward_footprint_bytes`), measured from this
    /// pack. The window cap spends the free memory after it, and the expert tier's reserve keeps the card's share of it.
    uint64_t footprint_bytes() const { return footprint_; }
    const std::string& path() const { return path_; }
    /// The open file descriptor for reading tensors that stay on disk (token_embd.weight).
    int read_fd() const { return read_fd_; }
    /// The tier the routed experts live in: pinned RAM places read from the pack's experts section, their rows computed
    /// by the CPU pool. Null until the start's plan opens it (plan() does, so a session always finds one).
    core::ExpertTier* tier() const { return tier_.get(); }
    /// The drafter's block's own tier, for the same reason it has its own experts section: its rows are a different
    /// format from the model's, and the tier plans a placement per layer of its own. Null when the block cannot run.
    core::ExpertTier* draft_tier() const { return draft_tier_.get(); }
    /// The model's peer driver: the expert tier is one (core::ExpertTier derives model::PeerHooks), so a second card
    /// held a block of layers' experts and computes their rows in every expert block.
    model::PeerHooks* peer() const override;
    /// The pinned rows the CPU pool writes one expert block's outputs into: `kExpertRows * n_used` rows of `n_embd`
    /// floats, row `t * n_used + k` of block token t's k-th pick.
    float* host_rows() const { return h_rows_; }
    /// The same rows as the card sees them (the allocation is mapped), for the sum that reads them beside the rows the
    /// card's own experts wrote.
    float* host_rows_dev() const { return h_rows_dev_; }
    /// One expert block's activations in pinned memory, `kExpertRows` rows of `n_embd` floats: what a second card
    /// quantizes and multiplies for the experts it holds (core::PeerExperts::arm reads them where they are).
    float* host_act() const { return h_act_; }
    /// One layer's packed expert rows as the card's grouped kernel reads them (blob_bytes checked against it at the
    /// open). Valid for every layer `has_experts` names.
    const kernels::NativeExpertLayout& expert_layout(int layer) const { return elayouts_[(size_t) layer]; }
    /// Whether the layer routes to experts at all (the dense lead layers route to none).
    bool has_experts(int layer) const {
        return layer >= 0 && layer < (int) pack_.experts.size() && pack_.experts[layer].blob_bytes != 0;
    }
    /// The drafter block's own packed expert rows, from its own section and layout: empty unless the card has a grouped
    /// kernel for its types and its blob is exactly those rows (the load says which when it is empty).
    const kernels::NativeExpertLayout& draft_expert_layout() const { return draft_elayout_; }
    bool has_draft_experts() const { return draft_elayout_.bytes != 0; }

    /// Whether the start asked for windows wider than one row (`--spec`): the pack's next-n block fills the extra one,
    /// or an oracle's tokens do when the start brings `--spec-oracle`.
    bool wide_windows_asked() const { return req_.spec >= 2; }
    /// Whether the engine means to draft with the block's own experts: only then is it worth loading GiB of them.
    bool drafting_asked() const { return req_.draft; }

private:
    /// Opens the pack and allocates every dense tensor, their reads queued in `reads`.
    bool load(const std::string& path, core::LoadPlan& reads, std::string& err);
    /// What the tier needs to know about this pack and the start's flags (§ core/expert_tier.hpp).
    core::ExpertTierSpec tier_spec() const;
    /// The tier, opened while the start's plan is live so the reads of its places go into `reads`.
    bool ensure_tier(core::LoadPlan& reads, std::string& err);
    /// The drafter's tier: its spec, and the tier, which is nothing at all when the pack carries no next-n block.
    core::ExpertTierSpec draft_spec() const;
    bool ensure_draft_tier(core::LoadPlan& reads, std::string& err);
    /// Every routed layer's expert layout, checked against the pack's blob sizes and the grouped kernels' types.
    bool check_expert_layouts(std::string& err);
    /// The drafter's block, checked weight by weight against the geometry, so a pack that carries half a block says so
    /// when it is loaded rather than when something first tries to draft.
    void check_draft_weights() const;

    Pack pack_;
    model::ModelDesc desc_;
    std::map<std::string, DevTensor> dev_;
    std::string path_;
    int device_ = 0;
    int read_fd_ = -1;   ///< kept open for the loader's reads of the dense tensors
    model::LoadRequest req_;   ///< the cards, the tier's flags and the session count the start asked for
    uint64_t cache_bytes_ = 0;  ///< one session's caches, which the tier leaves room for beside its places
    uint64_t footprint_ = 0;    ///< what the forward pass holds on the card outside a window, this pack's measure
    std::shared_ptr<core::ExpertTier> tier_;
    std::shared_ptr<core::ExpertTier> draft_tier_;   ///< the drafter block's own, null when it cannot run
    float* h_rows_ = nullptr;   ///< pinned: the CPU pool's expert rows, kExpertRows x n_used x n_embd
    float* h_rows_dev_ = nullptr;   ///< the same, as the card reads them
    float* h_act_ = nullptr;   ///< pinned, portable: one expert block's activations, as a second card reads them
    std::vector<kernels::NativeExpertLayout> elayouts_;   ///< per routed layer, in pack_.experts' order
    kernels::NativeExpertLayout draft_elayout_{};         ///< the drafter block's own, empty when it cannot be used
};

}  // namespace basalt::models::glm5next
