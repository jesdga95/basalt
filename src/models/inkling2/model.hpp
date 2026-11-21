// src/models/inkling2/model.hpp - a loaded Inkling 2 pack: every weight on the main card (the projections of a layer fused
// into one Q8_0 matrix, the experts of a layer in one run of blobs at a stride of whole blocks, the head rounded to Q8_0 as
// it loads), the token embedding in mapped host memory.
#pragma once

#include "basalt/model/adapter.hpp"
#include "models/inkling2/dspark.hpp"   // DSPARK HOOK
#include "models/inkling2/pack.hpp"
#include "models/inkling2/spill.hpp"
#include "models/inkling2/vision_encoder.hpp"   // [vision]

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
    unsigned long long* blob_table = nullptr;   ///< device [n_expert]: each expert's blob address (its RAM copy's when it spilled)
    int n_spill = 0;                         ///< experts in RAM (spill.hpp): the layer's last slots
    const int32_t* slot_map = nullptr;       ///< device [n_expert]: each expert's slot in `experts`; null: the expert's own
    const uint8_t* spill_host = nullptr;     ///< the RAM experts' blobs in slot order (pinned, mapped), null without a spill
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
    /// [core layers][kStreams][n_embd]: every core layer's message queries, layer loop_start first
    const float* msg_queries() const { return msg_queries_; }
    int device() const { return device_; }
    int sessions() const { return flags_.sessions > 0 ? flags_.sessions : 1; }
    const model::LoadRequest& flags() const { return flags_; }
    bool cache_fp8() const { return fp8_; }
    /// [vision] The image encoder: staged with --vision, else it refuses pictures. One per model; ENC runs between requests.
    ImageEncoder& encoder() const { return vis_; }
    const DsparkWeights* dspark() const { return dspark_.get(); }   // DSPARK HOOK: null without a drafter or with --spec 1
    /// Whether expert e of layer l is on the card (every one, unless the context asked for made some spill to RAM).
    bool on_card(int l, int e) const {
        return spill_.slot.empty() || spill_.slot[(size_t) l].empty() ||
               spill_.slot[(size_t) l][(size_t) e] < geo().n_expert - spill_.n_spill[(size_t) l] ||
               (!copy_.empty() && copy_[(size_t) l * geo().n_expert + e]);
    }
    /// The elastic spill: the experts the asked context sent to RAM come back to the card as copies while its caches back
    /// only the positions in use. On with a spill the context made (not --expert-cache) and a context above the elastic
    /// KV's base (core/elastic_kv.hpp).
    bool elastic() const { return elastic_; }
    /// Copies RAM experts to the card, the most routed first, while `keep_free` bytes stay free; their blob table entries
    /// point at the copies (decode reads them there; a prompt chunk's staging is unchanged). Bytes copied.
    uint64_t promote(uint64_t keep_free);
    /// Sends copies back, the least routed first, until at least `bytes` are freed: their entries point at RAM again (on
    /// `stream`, which is waited for before the copies are freed). Bytes freed.
    uint64_t demote(uint64_t bytes, void* stream);
    int64_t promoted() const { return n_copy_; }
    /// A prompt chunk's staging of layer l's RAM experts from their card copies, when every one has a copy (false: none
    /// issued; the caller copies them from RAM).
    bool stage_from_card(int l, void* stream) const;
    const SpillPlan& spill() const { return spill_; }

private:
    float* small(const std::string& name, uint64_t floats, core::LoadPlan& reads, int file, std::string& err,
                 float* dst = nullptr);
    uint16_t* small_bf16(const std::string& name, uint64_t values, core::LoadPlan& reads, int file, std::string& err,
                         uint16_t* dst = nullptr);
    /// Every layer's experts, the ones `spill_` sends to RAM there; their bytes on the card into `experts`.
    model::Result load_experts(core::LoadPlan& reads, int file, uint64_t& experts);

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
    float* msg_queries_ = nullptr;
    model::LoadRequest flags_;
    int device_ = 0, fd_ = -1;
    bool fp8_ = false;
    int64_t ctx_ = kDefaultContext;
    mutable ImageEncoder vis_;   // [vision]
    std::unique_ptr<DsparkWeights> dspark_;   // DSPARK HOOK
    // the spill (spill.hpp): each spilled layer's reserved range and its own backing, the staging they share, the RAM copies
    struct Range {
        uint8_t* va = nullptr;
        uint64_t bytes = 0, own = 0;
        unsigned long long handle = 0;
    };
    SpillPlan spill_;
    std::vector<Range> ranges_;
    std::vector<unsigned long long> staging_;   ///< granule-sized pieces (the driver maps whole allocations only)
    uint8_t* spill_host_ = nullptr;
    // the elastic spill: the RAM experts the most routed first, each one's card copy (null: none) and mapped RAM address
    bool elastic_ = false;
    std::vector<std::pair<int32_t, int32_t>> spill_order_;
    std::vector<void*> copy_;
    std::vector<unsigned long long> host_addr_;
    int64_t n_copy_ = 0;
    std::vector<int> layer_copies_;   ///< [layer] RAM experts with a card copy
};

/// A session over `model`; null with `err` set.
std::unique_ptr<model::ModelSession> make_session(InklingModel& model, std::string& err);

/// The device bytes a session takes for `context` positions (fp8: one byte a cached value).
uint64_t session_bytes(const Geometry& g, int64_t context, bool fp8);

}  // namespace basalt::models::inkling2
