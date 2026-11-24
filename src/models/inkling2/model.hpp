// src/models/inkling2/model.hpp - a loaded Inkling 2 pack: every weight on the main card (the projections of a layer fused
// into one Q8_0 matrix, the experts of a layer in one run of blobs at a stride of whole blocks in a reserved address range,
// the head rounded to Q8_0 as it loads), the token embedding in mapped host memory. With the layer split the second card
// holds the later layers' weights, experts and caches instead (cards(), card_of()).
#pragma once

#include "basalt/core/peer_experts.hpp"
#include "basalt/model/adapter.hpp"
#include "basalt/model/peer_hooks.hpp"
#include "basalt/model/split_hooks.hpp"
#include "models/inkling2/dspark.hpp"   // DSPARK HOOK
#include "models/inkling2/pack.hpp"
#include "models/inkling2/spill.hpp"
#include "models/inkling2/vision_encoder.hpp"   // [vision]

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace basalt::models::inkling2 {

constexpr int64_t kDefaultContext = 8192;

/// The four loop controllers of a core layer: the query, key, value and MLP inputs' messages.
constexpr int kStreams = 4;

struct LayerWeights {
    uint8_t* proj = nullptr;                 ///< Q8_0 planes [kProj][n_embd] codes then fp16 scales (wgemm::q8_planes)
    uint8_t* out = nullptr;                  ///< Q8_0 planes [n_embd][3072] codes then fp16 scales (wgemm::q8_planes)
    uint16_t* proj_q_sc_t = nullptr;         ///< coalesced [n_embd/32][ld(kProjQ)] fp16 scales for prefill::q8
    uint16_t* proj_k_sc_t = nullptr;         ///< coalesced [n_embd/32][ld(kProjKV)] fp16 scales for prefill::q8
    uint16_t* proj_v_sc_t = nullptr;         ///< coalesced [n_embd/32][ld(kProjKV)] fp16 scales for prefill::q8
    uint16_t* out_sc_t = nullptr;            ///< coalesced [3072/32][ld(n_embd)] fp16 scales for prefill::q8
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
    unsigned long long* blob_table = nullptr;   ///< device [n_expert]: each expert's blob address on the card (its copy's), 0: in RAM
    int n_spill = 0;                         ///< experts in RAM (spill.hpp): the layer's last slots
    const int32_t* slot_map = nullptr;       ///< device [n_expert]: each expert's slot in `experts`; null: the expert's own
    const int32_t* slot_full = nullptr;      ///< gpu-mode peer: slot_map with the second card's experts in their slots too (slot_map has -1 for them)
    const uint8_t* spill_host = nullptr;     ///< the RAM experts' blobs in slot order (pinned), null without a spill
    const float* msg_query = nullptr;        ///< core layers: [kStreams][n_embd]
    const uint16_t* ctrl_down[kStreams] = {};   ///< BF16 [gate_rank][2 n_embd]
    const uint16_t* ctrl_gf[kStreams] = {};     ///< BF16 [2 n_embd][gate_rank]: the gain rows, then the forget rows
};

class InklingModel final : public model::LoadedModel, private model::SplitHooks, private model::PeerHooks {
public:
    InklingModel();
    ~InklingModel() override;
    model::Result plan(const model::LoadRequest& req, core::LoadPlan& reads,
                       std::vector<std::unique_ptr<model::ModelSession>>& sessions) override;
    model::Result attach(const model::LoadRequest& req, std::vector<std::unique_ptr<model::ModelSession>>& sessions) override;
    const model::ModelDesc& desc() const override { return desc_; }
    int64_t context() const override { return ctx_; }
    std::string gpu_report(int card) const override;
    model::PeerHooks* peer() const override { return peer_ ? const_cast<InklingModel*>(this) : nullptr; }

    const Pack& pack() const { return pack_; }
    const Geometry& geo() const { return pack_.geo; }
    const LayerWeights& layer(int l) const { return layers_[(size_t) l]; }
    const uint8_t* head() const { return head_; }              ///< Q8_0 [n_vocab][n_embd]
    const uint16_t* embedding() const { return embd_dev_; }    ///< BF16 [n_vocab][n_embd], mapped host memory
    const float* embed_norm() const { return embed_norm_; }
    const float* output_norm() const { return output_norm_; }
    const float* output_query() const { return output_query_; }
    /// [core layers][kStreams][n_embd]: every core layer's message queries, layer loop_start first, on card `card` (null on
    /// a card that runs no core layer)
    const float* msg_queries(int card = 0) const { return msg_queries_[card]; }
    int device() const { return device_; }
    /// The layer split (docs/configuration/gpus.md): with two cards the second runs layers [cut, n_layers) with their
    /// weights, experts and caches; the first runs the rest, the embedding, the head and the drafter. One card: card 0.
    int cards() const { return cut_ > 0 ? 2 : 1; }
    int card_of(int layer) const { return cut_ > 0 && layer >= cut_ ? 1 : 0; }
    int card_device(int card) const { return card == 0 ? device_ : second_; }
    int split_cut() const { return cut_; }
    int sessions() const { return flags_.sessions > 0 ? flags_.sessions : 1; }
    const model::LoadRequest& flags() const { return flags_; }
    bool cache_fp8() const { return fp8_; }
    bool cache_int8() const { return int8_; }
    /// [vision] The image encoder: staged with --vision, else it refuses pictures. One per model; ENC runs between requests
    /// and windows, so nothing reads the experts while it holds their memory (lend_room).
    ImageEncoder& encoder() const { return vis_; }
    /// What the sessions share (session.cpp's Shared): made by the first, gone with the last.
    std::weak_ptr<void>& shared() { return shared_; }
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
    /// Sends copies back, the least routed first, until at least `bytes` are freed: their entries go back to 0, the CPU
    /// computes them again (on `stream`, which is waited for before the copies are freed). Bytes freed.
    uint64_t demote(uint64_t bytes, void* stream);
    int64_t promoted() const { return n_copy_; }
    /// The free VRAM the session holds the card at from now on (VRAM / POST /v1/vram; 0: its own spare only): a grow of
    /// the caches and a promote keep at least this much free. `lend(true)` gives every card copy back for a sleep and holds them off (nothing promotes)
    /// until `lend(false)`.
    uint64_t reserve() const { return reserve_; }
    bool lent() const { return lent_; }
    uint64_t held_free(uint64_t base) const { return std::max(base, reserve_); }
    /// Resizes the card's copies so that `free_bytes` are free: copies go back to RAM when short, return while there is
    /// room. False when even no copy at all leaves that much.
    bool resize_reserve(uint64_t free_bytes, void* stream);
    void lend(bool low, void* stream);
    /// A prompt chunk's staging of layer l's RAM experts from their card copies, when every one has a copy (false: none
    /// issued; the caller copies them from RAM). `with_peer`: the second card's experts too (all_copied).
    bool stage_from_card(int l, void* stream, bool with_peer) const;
    /// Every expert of layer l that left the card has a copy on it again (the elastic spill): the layer runs on the main card alone.
    bool all_copied(int l) const { return !layer_copies_.empty() && layers_[(size_t) l].n_spill > 0 && layer_copies_[(size_t) l] >= layers_[(size_t) l].n_spill; }
    /// A prompt chunk sends layer l's rows to the second card: it holds experts of the layer that the main card has no copy of.
    bool peer_runs(int l) const { return peer_n(l) > 0 && !all_copied(l); }
    const SpillPlan& spill() const { return spill_; }
    /// gpu-mode peer (docs/design/multi-gpu.md): the experts that spilled from the main card and the second card holds in
    /// its VRAM, the hottest first as far as its room goes. The main card runs every layer; a decode window's pairs on
    /// these experts go to the card from the CPU experts' doorbell thread (cpu_experts.hpp), a prompt chunk's rows go to
    /// it layer by layer (prompt.cpp) and its partial sums come back. They have no RAM copy and no staging.
    core::PeerExperts* peer_experts() const { return peer_.get(); }
    int peer_device() const { return peer_ ? peer_->device() : -1; }
    bool peer_held(int l, int e) const { return !peer_slot_.empty() && peer_slot_[(size_t) l * geo().n_expert + e] >= 0; }
    /// The second card is the one to compute expert e of layer l now: it holds it and the main card has no copy (the elastic
    /// spill copies the second card's experts to the main card too, while it has room).
    bool peer_serves(int l, int e) const { return peer_held(l, e) && !copy_[(size_t) l * geo().n_expert + e]; }
    /// Experts of layer l on the second card; their blobs are at peer_base(l) + j * peer_stride(l), j in expert order,
    /// peer_map(l) (a device array on that card) giving each expert's j or -1.
    int peer_n(int l) const { return peer_n_.empty() ? 0 : peer_n_[(size_t) l]; }
    uint8_t* peer_base(int l) const { return peer_->slot_ptr(peer_first_[(size_t) l]); }
    uint64_t peer_stride(int l) const { return (layers_[(size_t) l].expert_stride + 255) / 256 * 256; }
    const int32_t* peer_map(int l) const { return peer_map_[(size_t) l]; }
    /// The layer's experts in RAM that a prompt chunk stages and the CPU computes (the spilled ones less the second card's).
    int ram_n(int l) const { return layers_[(size_t) l].n_spill - peer_n(l); }
    /// Expert e of layer l's blob in RAM when the CPU computes it (it spilled and has no card copy; its blob table entry is
    /// 0), else null (cpu_experts.hpp).
    const uint8_t* ram_blob(int l, int e) const;
    /// The layers with an expert the CPU computes, bit l for layer l: what a window's launches depend on.
    uint64_t cpu_layers() const;
    /// [vision] Leaves at least `bytes` free above the card's spare for the image encoder's region: the experts' card copies
    /// go back to RAM first (demote), then whole layers' expert memory goes to RAM, the last layers first, its addresses kept
    /// (the range stays reserved). Every stream is idle on return. False with `err` when even all of it is short (nothing
    /// lent then).
    bool lend_room(uint64_t bytes, std::string& err);
    /// [vision] The lent layers' memory comes back at the same addresses with the same bytes, then the copies (promote).
    void refill();

private:
    float* small(const std::string& name, uint64_t floats, core::LoadPlan& reads, int file, std::string& err,
                 float* dst = nullptr);
    uint16_t* small_bf16(const std::string& name, uint64_t values, core::LoadPlan& reads, int file, std::string& err,
                         uint16_t* dst = nullptr);
    /// Every layer's experts, the ones `spill_` sends to RAM there; their bytes on the card into `experts`.
    model::Result load_experts(core::LoadPlan& reads, int file, uint64_t& experts);
    /// Two cards (gpu-mode split): the layer split's cut by the shares (SplitHooks: split_spec and split_place below).
    model::Result plan_cards(const model::LoadRequest& req);
    // gpu-mode peer (PeerHooks): one second card holding spilled experts
    bool peer_card(size_t card, int device, std::string& err) override;
    uint64_t peer_room(size_t card) const override;
    model::PeerSpec peer_spec() const override;
    bool peer_place(const model::PeerPlacement& p, std::string& err) override;
    std::string peer_main() const override;
    model::SplitSpec split_spec(const model::GpuCards& c) const override;
    bool split_place(const model::SplitCut& cut, const std::vector<int>& devices, std::string& err) override;
    std::string split_part(size_t card) const override;

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
    float* msg_queries_[2] = {};
    const model::LoadRequest* req_ = nullptr;   ///< what split_spec reads while plan_cards runs
    int cut_ = 0, second_ = -1;              ///< the layer split's cut (0: none) and the second card's device
    std::vector<uint64_t> split_need_;       ///< the split's bytes on each card, as the cut counted them
    model::LoadRequest flags_;
    int device_ = 0, fd_ = -1;
    int ldev_ = 0;                           ///< the card the weights being planned load to
    bool fp8_ = false, int8_ = false;
    int64_t ctx_ = kDefaultContext;
    mutable ImageEncoder vis_;   // [vision]
    std::unique_ptr<DsparkWeights> dspark_;   // DSPARK HOOK
    std::weak_ptr<void> shared_;
    // each layer's experts in a reserved range with its own backing (a spilled layer's RAM slots over the staging they share,
    // spill.hpp; without virtual memory management a layer that keeps every expert has a plain allocation), the RAM copies
    struct Range {
        uint8_t* va = nullptr;
        uint64_t bytes = 0, own = 0;
        unsigned long long handle = 0;   ///< the own backing; 0 while lent (lend_room)
    };
    SpillPlan spill_;
    std::vector<Range> ranges_;
    std::vector<unsigned long long> staging_;   ///< granule-sized pieces (the driver maps whole allocations only)
    uint8_t* spill_host_ = nullptr;
    // the elastic spill: the RAM experts the most routed first, each one's card copy (null: none)
    bool elastic_ = false;
    uint64_t reserve_ = 0;
    bool lent_ = false;
    std::vector<std::pair<int32_t, int32_t>> spill_order_;
    std::vector<void*> copy_;
    int64_t n_copy_ = 0;
    std::vector<int> layer_copies_;   ///< [layer] RAM experts with a card copy
    std::vector<int> layer_ram_copies_;   ///< [layer] those the second card does not hold (a prompt chunk stages the others' copies)
    // [vision] the experts' memory lent to the image encoder: each lent range's bytes, in RAM until refill
    struct LentRange {
        size_t range = 0;
        std::unique_ptr<uint8_t[]> bytes;
    };
    std::vector<LentRange> lent_ranges_;
    int64_t lend_copies_ = 0;   ///< card copies demoted for the encoder
    // gpu-mode peer: the second card's experts, layer by layer in expert order in one cache (peer_first_ the layer's first slot)
    std::unique_ptr<core::PeerExperts> peer_;
    std::vector<int32_t> peer_slot_;         ///< [layer * n_expert + e] the expert's slot there, -1: not there
    std::vector<int> peer_n_;                ///< [layer] experts there
    std::vector<int32_t> peer_first_;        ///< [layer] first slot
    std::vector<int32_t*> peer_map_;         ///< [layer] device array on that card: expert -> its index in the layer's run, or -1
    std::vector<std::pair<int, void*>> peer_bufs_;   ///< (device, allocation) of the maps
    uint64_t peer_bytes_ = 0;
    double lend_ms_ = 0;
};

/// A session over `model`; null with `err` set.
std::unique_ptr<model::ModelSession> make_session(InklingModel& model, std::string& err);

/// The device bytes a session takes for `context` positions in the selected KV cache format, its drafter aside, on card
/// `card` of a split at `cut` (0: one card, every visit).
uint64_t session_bytes(const Geometry& g, int64_t context, bool fp8, bool int8, int cut = 0, int card = 0);
/// The device bytes every session shares once on a card: the forward's scratch, the prompt path, the drafter's tap (the
/// first card's).
uint64_t shared_bytes(const Geometry& g, bool drafter);

}  // namespace basalt::models::inkling2
