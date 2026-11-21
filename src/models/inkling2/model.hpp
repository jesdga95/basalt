// src/models/inkling2/model.hpp - a loaded Inkling 2 pack: every weight on the main card (the projections of a layer fused
// into one Q8_0 matrix, the experts of a layer in one run of blobs at a stride of whole blocks, the head rounded to Q8_0 as
// it loads), the token embedding in mapped host memory.
#pragma once

#include "basalt/model/adapter.hpp"
#include "models/inkling2/pack.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace basalt::models::inkling2 {

constexpr int64_t kDefaultContext = 8192;

/// The four loop controllers of a core layer: the query, key, value and MLP inputs' messages.
constexpr int kStreams = 4;

struct LayerWeights {
    uint8_t* proj = nullptr;                 ///< Q8_0 [kProj][n_embd]: q, r, gate, sink, k, k taps, v, v taps
    uint8_t* out = nullptr;                  ///< Q8_0 [n_embd][3072]
    const float* attn_norm = nullptr;        ///< sliding-window layers
    const float* q_norm = nullptr, *k_norm = nullptr, *v_norm = nullptr;   ///< full layers
    const float* q_clip = nullptr, *r_clip = nullptr, *rel_table = nullptr;
    const float* k_conv = nullptr, *v_conv = nullptr, *attn_post_conv = nullptr, *ffn_post_conv = nullptr;
    const float* attn_query = nullptr;       ///< [1 or 3][n_embd]; null on layer 0
    const float* ffn_query = nullptr;
    const float* ffn_norm = nullptr;
    const float* router = nullptr;           ///< F32 [n_expert][n_embd]
    const float* router_bias = nullptr;
    uint8_t* experts = nullptr;              ///< n_expert blobs, expert e's at experts + e * expert_stride
    int gu_type = 0, d_type = 0;
    size_t blob_bytes = 0;
    size_t expert_stride = 0;                ///< blob_bytes rounded up to whole blocks of its type and 16 bytes
    const float* msg_query = nullptr;        ///< core layers: [kStreams][n_embd]
    const uint16_t* ctrl_down[kStreams] = {};   ///< BF16 [gate_rank][2 n_embd]
    const uint16_t* ctrl_gf[kStreams] = {};     ///< BF16 [2 n_embd][gate_rank]: the gain rows, then the forget rows
};

class InklingModel final : public model::LoadedModel {
public:
    InklingModel();
    ~InklingModel() override;
    model::Result plan(const model::LoadRequest& req, core::LoadPlan& reads,
                       std::vector<std::unique_ptr<model::ModelSession>>& sessions) override;
    model::Result attach(const model::LoadRequest& req, std::vector<std::unique_ptr<model::ModelSession>>& sessions) override;
    const model::ModelDesc& desc() const override { return desc_; }
    int64_t context() const override { return ctx_; }

    const Pack& pack() const { return pack_; }
    const Geometry& geo() const { return pack_.geo; }
    const LayerWeights& layer(int l) const { return layers_[(size_t) l]; }
    const uint8_t* head() const { return head_; }              ///< Q8_0 [n_vocab][n_embd]
    const uint16_t* embedding() const { return embd_dev_; }    ///< BF16 [n_vocab][n_embd], mapped host memory
    const float* embed_norm() const { return embed_norm_; }
    const float* output_norm() const { return output_norm_; }
    const float* output_query() const { return output_query_; }
    int device() const { return device_; }
    int sessions() const { return flags_.sessions > 0 ? flags_.sessions : 1; }
    const model::LoadRequest& flags() const { return flags_; }
    bool cache_fp8() const { return fp8_; }

private:
    float* small(const std::string& name, uint64_t floats, core::LoadPlan& reads, int file, std::string& err);
    uint16_t* small_bf16(const std::string& name, uint64_t values, core::LoadPlan& reads, int file, std::string& err,
                         uint16_t* dst = nullptr);

    Pack pack_;
    model::ModelDesc desc_;
    std::vector<LayerWeights> layers_;
    std::vector<void*> dev_;                 ///< every device allocation, freed with the model
    uint8_t* head_ = nullptr;
    uint16_t* embd_host_ = nullptr;          ///< pinned, mapped
    uint16_t* embd_dev_ = nullptr;           ///< its device address
    const float* embed_norm_ = nullptr;
    const float* output_norm_ = nullptr;
    const float* output_query_ = nullptr;
    model::LoadRequest flags_;
    int device_ = 0, fd_ = -1;
    bool fp8_ = false;
    int64_t ctx_ = kDefaultContext;
};

/// A session over `model`; null with `err` set.
std::unique_ptr<model::ModelSession> make_session(InklingModel& model, std::string& err);

/// The device bytes a session takes for `context` positions (fp8: one byte a cached value).
uint64_t session_bytes(const Geometry& g, int64_t context, bool fp8);

}  // namespace basalt::models::inkling2
