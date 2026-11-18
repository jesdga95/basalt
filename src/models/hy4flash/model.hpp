// src/models/hy4flash/model.hpp - a loaded Hy4-flash pack: the dense tensors on the main card, the token embedding in
// host memory, every routed expert in a RAM tier run by the CPU, and the Engram table read from the file by row.
#pragma once

#include "basalt/core/ram_load.hpp"
#include "basalt/model/adapter.hpp"
#include "basalt/ngram/ple_reader.hpp"
#include "models/hy4flash/cpu_experts.hpp"
#include "models/hy4flash/pack.hpp"

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

/// The longest context this build keeps (fp32 latent rows).
constexpr int64_t kMaxContext = 131072;
constexpr int64_t kDefaultContext = 8192;

class Hy4Model final : public model::LoadedModel {
public:
    Hy4Model();
    ~Hy4Model() override;
    model::Result plan(const model::LoadRequest& req, core::LoadPlan& reads,
                       std::vector<std::unique_ptr<model::ModelSession>>& sessions) override;
    model::Result attach(const model::LoadRequest& req, std::vector<std::unique_ptr<model::ModelSession>>& sessions) override;
    const model::ModelDesc& desc() const override { return desc_; }
    int64_t context() const override { return ctx_; }

    const Pack& pack() const { return pack_; }
    const Geometry& geo() const { return pack_.geo; }
    /// Device tensor `blk.<layer>.<suffix>` or a global name; null when absent.
    const DevTensor* tensor(const std::string& name) const;
    const DevTensor* tensor(int layer, const char* suffix) const { return tensor("blk." + std::to_string(layer) + "." + suffix); }
    /// One token-embedding row, dequantised into `out` (n_embd floats).
    bool embedding_row(int32_t token, float* out) const;
    /// The blob of routed expert `e` of `layer`.
    ExpertBlob expert(int layer, int e) const;
    ngram::PleReader& engram_reader() const { return *engram_; }
    Workers& workers() const { return *workers_; }
    int device() const { return device_; }
    const std::string& diag_dump() const { return diag_dump_; }

private:
    Pack pack_;
    model::ModelDesc desc_;
    std::map<std::string, DevTensor> dev_;
    std::vector<core::RamAllocation> ram_;      ///< per layer with experts
    std::vector<int> ram_index_;                ///< layer -> index in ram_, or -1
    std::vector<uint8_t> embd_;
    std::string diag_dump_;
    std::unique_ptr<ngram::PleReader> engram_;
    std::unique_ptr<Workers> workers_;
    int device_ = 0, fd_ = -1;
    int64_t ctx_ = kDefaultContext;
};

/// A session over `model`; null with `err` set.
std::unique_ptr<model::ModelSession> make_session(Hy4Model& model, std::string& err);

}  // namespace basalt::models::hy4flash
