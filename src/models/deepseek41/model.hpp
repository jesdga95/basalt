// src/models/deepseek41/model.hpp - a loaded DeepSeek-V4.1 pack: dense tensors on the GPU, the experts section and the
// Engram table as file ranges.
#pragma once

#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/model/adapter.hpp"
#include "basalt/ngram/ple_reader.hpp"
#include "basalt/vision/mmproj.hpp"
#include "models/deepseek41/pack.hpp"
#include "models/deepseek41/prefill_plan.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace basalt::models::deepseek41 {

struct DevTensor {
    void* ptr = nullptr;
    int type = 0;
    uint64_t bytes = 0;
    std::vector<uint64_t> shape;
};

/// y[c][r] = W[r] . x[c] for `ncols` float columns, quantized through `scratch_q8_1`, whatever W's type.
inline void dense_f32(const DevTensor& w, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out, int ncols,
                      void* stream) {
    kernels::native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    kernels::native_mmvq(w.type, w.ptr, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

class ExpertTier;

class DsModel final : public model::LoadedModel {
public:
    ~DsModel() override;
    /// Reads the pack and uploads every dense tensor except the token embedding (read per token from the file).
    bool load(const std::string& path, std::string& err);

    const model::ModelDesc& desc() const override { return desc_; }
    int64_t context() const override { return ctx_ ? ctx_ : ctxp_default(); }
    /// Fixes the context from --max-context (`set`) or the model's maximum, bounded by the free device memory. Call
    /// once after load(), before a session is made. False with `err` when the request cannot be honoured.
    bool resolve_context(int64_t requested, std::string& err);
    model::Result make_session(std::unique_ptr<model::ModelSession>& out) override;

    const Pack& pack() const { return pack_; }
    const Geometry& geo() const { return pack_.geo; }
    /// Device tensor `blk.<layer>.<suffix>` or a global name; null when absent.
    const DevTensor* tensor(const std::string& name) const;
    const DevTensor* tensor(int layer, const char* suffix) const { return tensor("blk." + std::to_string(layer) + "." + suffix); }
    /// One token-embedding row, dequantised into `out` (n_embd floats).
    bool embedding_row(int32_t token, float* out, std::string& err) const;
    ngram::PleReader& engram_reader() const { return *engram_; }
    int experts_fd() const { return experts_fd_; }
    const std::string& path() const { return path_; }
    /// Diagnostics (off by default): replay recorded expert outputs; dump per-layer tensors.
    const std::string& diag_dump() const { return diag_dump_; }
    void set_diag_dump(std::string dump) { diag_dump_ = std::move(dump); }
    /// Fixed expert-tier sizes per layer for repeatable runs (-1 = sized from free memory).
    int diag_tier_ram() const { return diag_tier_ram_; }
    int diag_tier_vram() const { return diag_tier_vram_; }
    void set_diag_tier(int ram, int vram) { diag_tier_ram_ = ram; diag_tier_vram_ = vram; }
    /// The expert tier and the pinned rows it shares with a second card: one per model, whatever the sessions.
    struct SharedTier {
        std::shared_ptr<ExpertTier> tier;
        float* h_x = nullptr;
        float* h_rows = nullptr;
    };
    SharedTier& shared_tier() const { return shared_tier_; }
    /// The sessions that will hold a context each.
    int sessions() const { return tiers_.sessions > 0 ? tiers_.sessions : 1; }
    /// What the flags ask of the expert tiers.
    const model::LoadRequest& tiers() const { return tiers_; }
    void set_tiers(const model::LoadRequest& req) { tiers_ = req; }
    /// The CUDA device of the second card's expert tier, or -1.
    int peer_device() const { return peer_device_; }
    void set_peer_device(int d) { peer_device_ = d; }
    /// The CUDA device the dense tensors live on.
    int device() const { return device_; }
    /// The sizes the prompt path's workspaces depend on.
    prefillp::Dims prompt_dims() const;
    /// Device bytes of the compressed caches of a full context.
    uint64_t context_bytes() const;
    /// Opens the image projector: the diagnostic standalone file when set, else the pack's vision section.
    /// False with `err` set when neither exists.
    bool open_mmproj(vision::Mmproj& mp, std::string& err) const;

private:
    Pack pack_;
    model::ModelDesc desc_;
    std::map<std::string, DevTensor> dev_;
    std::string diag_dump_, path_;
    int diag_tier_ram_ = -1, diag_tier_vram_ = -1;
    int peer_device_ = -1;
    int device_ = 0;
    model::LoadRequest tiers_;
    mutable SharedTier shared_tier_;
    int64_t ctx_ = 0;
    static int64_t ctxp_default();
    int embd_fd_ = -1;
    int experts_fd_ = -1;
    std::unique_ptr<ngram::PleReader> engram_;
};

}  // namespace basalt::models::deepseek41
