// src/models/deepseek41/model.hpp - a loaded DeepSeek-V4.1 pack: dense tensors on the GPU, the experts section and the
// Engram table as file ranges.
#pragma once

#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/model/adapter.hpp"
#include "basalt/ngram/ple_reader.hpp"
#include "basalt/vision/mmproj.hpp"
#include "models/deepseek41/pack.hpp"

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

class DsModel final : public model::LoadedModel {
public:
    ~DsModel() override;
    /// Reads the pack and uploads every dense tensor except the token embedding (read per token from the file).
    bool load(const std::string& path, std::string& err);

    const model::ModelDesc& desc() const override { return desc_; }
    int64_t context() const override { return ctx_ ? ctx_ : ctxp_default(); }
    /// Fixes the context from --max-context (`set`) or the model's maximum, bounded by the free device memory. Call
    /// once after load(), before a session is made. False with `err` when the request cannot be honoured.
    bool resolve_context(int64_t requested, bool set, std::string& err);
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
    const std::string& diag_fixture() const { return diag_fixture_; }
    const std::string& diag_dump() const { return diag_dump_; }
    void set_diag(std::string fixture, std::string dump) { diag_fixture_ = std::move(fixture); diag_dump_ = std::move(dump); }
    void set_diag_mmproj(std::string path) { diag_mmproj_ = std::move(path); }
    /// Fixed expert-tier sizes per layer for repeatable runs (-1 = sized from free memory).
    int diag_tier_ram() const { return diag_tier_ram_; }
    int diag_tier_vram() const { return diag_tier_vram_; }
    void set_diag_tier(int ram, int vram) { diag_tier_ram_ = ram; diag_tier_vram_ = vram; }
    /// Drafter GGUF run on the second GPU beside decode (empty = off).
    const std::string& diag_dspark() const { return diag_dspark_; }
    void set_diag_dspark(std::string path) { diag_dspark_ = std::move(path); }
    /// --spec: 2 or more loads the drafter (the pack's dspark section, or the diagnostic file) and drafts spec - 1 tokens.
    int spec() const { return spec_; }
    void set_spec(int n) { spec_ = n; }
    /// Opens the image projector: the diagnostic standalone file when set, else the pack's vision section.
    /// False with `err` set when neither exists.
    bool open_mmproj(vision::Mmproj& mp, std::string& err) const;
    /// The pack's drafter section (empty when it carries none).
    const core::FileRange& drafter_range() const { return pack_.dspark; }

private:
    Pack pack_;
    model::ModelDesc desc_;
    std::map<std::string, DevTensor> dev_;
    std::string diag_fixture_, diag_dump_, diag_mmproj_, diag_dspark_, path_;
    int diag_tier_ram_ = -1, diag_tier_vram_ = -1;
    int spec_ = 0;
    int64_t ctx_ = 0;
    static int64_t ctxp_default();
    int embd_fd_ = -1;
    int experts_fd_ = -1;
    std::unique_ptr<ngram::PleReader> engram_;
};

}  // namespace basalt::models::deepseek41
