// src/models/inkling2/model.hpp - a loaded Inkling 2 pack: every weight on the main card (the projections of a layer fused
// into one Q8_0 matrix, the experts of a layer in one run of blobs at a stride of whole blocks in a reserved address range,
// the head rounded to Q8_0 as it loads), the token embedding in mapped host memory. With the layer split the second card
// holds the later layers' weights, experts and caches instead (cards(), card_of()).
#pragma once

#include "basalt/core/progress.hpp"
#include "basalt/model/adapter.hpp"
#include "models/inkling2/copy_heat.hpp"
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
    const uint8_t* spill_host = nullptr;     ///< the RAM experts' blobs in slot order (pinned), null without a spill
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
    /// [vision] The image encoder: staged with --vision, else it refuses pictures. One per model; ENC runs between requests
    /// and windows, so nothing reads the experts while it holds their memory (lend_room).
    ImageEncoder& encoder() const { return vis_; }
    /// What the sessions share (session.cpp's Shared): made by the first, gone with the last.
    std::weak_ptr<void>& shared() { return shared_; }
    const DsparkWeights* dspark() const { return dspark_.get(); }   // DSPARK HOOK: null without a drafter or with --spec 1
    /// Whether expert e of layer l is on the card (every one, unless the context asked for made some spill to RAM), a
    /// loaned copy aside.
    bool on_card(int l, int e) const {
        return spill_.slot.empty() || spill_.slot[(size_t) l].empty() ||
               spill_.slot[(size_t) l][(size_t) e] < geo().n_expert - spill_.n_spill[(size_t) l] ||
               (!copy_.empty() && copy_[(size_t) l * geo().n_expert + e] && !on_loan(l, e));
    }
    bool on_loan(int l, int e) const { return !loan_flag_.empty() && loan_flag_[(size_t) l * geo().n_expert + e]; }
    const std::vector<size_t>& loaned_pairs() const { return loan_list_; }
    /// The elastic spill: the experts the asked context sent to RAM come back to the card as copies while its caches back
    /// only the positions in use. On with a spill the context made (not --expert-cache) and a context above the elastic
    /// KV's base (core/elastic_kv.hpp).
    bool has_spill() const { return has_spill_; }
    bool elastic() const { return elastic_; }
    /// Whether the RAM experts come back to the card as copies while it has room beyond its spare: any spill the context
    /// made (not --expert-cache), static or elastic.
    bool copies() const { return has_spill_ && flags_.expert_cache <= 0; }
    /// Copies RAM experts to the card, the highest ranked first (copy_heat.hpp), while `keep_free` bytes stay free; their
    /// blob table entries point at the copies. Bytes copied.
    uint64_t promote(uint64_t keep_free);
    /// Sends the lowest-ranked copies back until at least `bytes` are freed (their entries go back to 0, the CPU computes
    /// them again), packs the rest to the front and returns whole granules. `freed`: the bytes given back. False when a
    /// table or copy update could not be confirmed: the placement is then wedged.
    bool demote(uint64_t bytes, uint64_t& freed);
    /// Adds a sample of lookups (each pair's, since the last) that covers `rows` kept rows to the copies' ranking.
    void learn(const std::vector<uint32_t>& lookups, int64_t rows);
    /// Swaps up to `max_swaps` copies for hotter RAM experts of the same size, between windows; joins every reader of the
    /// tables first. The swaps made.
    int rebalance(int max_swaps);
    int64_t promoted() const { return (int64_t) copy_slots_.size(); }
    int64_t card_copies() const { return promoted() + loaned(); }
    /// One line on the copies after a placement event (paid and loaned counts, free VRAM, the paid range's backing).
    void log_copy_change(const char* event, int64_t before, uint64_t to_map = 0, bool always = false, bool warning = false) const;
    /// Advances when an address a session saved from the blob tables (the decode read-ahead's) may no longer hold that expert.
    uint64_t placement_epoch() const { return epoch_; }
    /// The free VRAM the session holds the card at from now on (VRAM / POST /v1/vram; 0: its own spare only): a grow of
    /// the caches and a promote keep at least this much free. `held_free(base)`: the free VRAM a step keeps, `base` its own
    /// default unless --vram-reserve-mib set the card's spare. `lend(true)` gives every card copy back for a sleep and holds them off (nothing promotes)
    /// until `lend(false)`.
    uint64_t reserve() const { return reserve_; }
    bool lent() const { return lent_; }
    /// Device work or placement that could not be confirmed: every later request is refused and the engine ends without
    /// the CUDA teardown.
    void wedge() {
        wedged_ = true;
        core::gpu_left_running().store(true);
    }
    bool wedged() const { return wedged_; }
    uint64_t held_free(uint64_t base) const { return std::max(spare_ ? spare_ : base, reserve_); }
    /// The card's free bytes as placement counts them: read when the copies are first placed, at a VRAM resize and at a
    /// wake, and moved by every map and release of the copies and the sessions' caches (account). What else allocates
    /// (kernels, graphs) is left to the spare and the late room. Placement reads this, never a fresh query, so it follows
    /// the requests alone.
    uint64_t room() const { return room_ > 0 ? (uint64_t) room_ : 0; }
    void account(int64_t mapped) { room_ -= mapped; }
    void sync_room();
    /// Resizes the card's copies so that `free_bytes` are free: copies go back to RAM when short, return while there is
    /// room. False, with nothing moved, when even no copy at all leaves that much; false when the copies could not go back.
    bool resize_reserve(uint64_t free_bytes);
    /// False when the copies could not all go back (a low `lend` then does not hold them off).
    bool lend(bool low);
    /// The prompt loan: with an elastic spill, one card and one session, the prompt path's idle device memory (its chunk
    /// arena and the experts' staging) holds copies of the next highest-ranked RAM experts while decode runs. They count
    /// for decode but take no slot or page of the paid copies.
    /// Null `base`: the arena goes; false when a loan upload into it could not be confirmed finished (its memory must not be
    /// freed then).
    bool set_prompt_arena(uint8_t* base, uint64_t bytes);
    /// After a decode window: starts the loan's uploads when one is due (`publish_at`: the position its copies become
    /// visible at) and queues their next bounded part. Never makes a copy visible.
    bool submit_loan(int64_t publish_at);
    /// Whether uploads wait to be published, and the position they are published at.
    bool loan_pending() const { return loan_publish_at_ >= 0; }
    int64_t loan_publish_at() const { return loan_publish_at_; }
    /// Queues what is left of the loan, waits for all of it and points the table entries at the copies (on `stream`).
    bool publish_loan(void* stream);
    /// Waits for the uploads, sets the loaned entries back to 0 (the CPU computes them again) and frees the regions.
    bool recall_loan(void* stream);
    int64_t loaned() const { return (int64_t) loan_list_.size(); }
    /// A prompt chunk's staging of layer l's RAM experts from their card copies, when every one has a copy (false: none
    /// issued; the caller copies them from RAM).
    bool stage_from_card(int l, void* stream) const;
    const SpillPlan& spill() const { return spill_; }
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
    /// Two cards: prices what they do together and sets the layer split's cut (0: the first card runs alone).
    model::Result plan_cards(const model::LoadRequest& req);

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
    int cut_ = 0, second_ = -1;              ///< the layer split's cut (0: none) and the second card's device
    model::LoadRequest flags_;
    int device_ = 0, fd_ = -1;
    int ldev_ = 0;                           ///< the card the weights being planned load to
    bool fp8_ = false;
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
    bool has_spill_ = false;
    bool elastic_ = false;
    uint64_t reserve_ = 0;
    uint64_t spare_ = 0;   ///< --vram-reserve-mib: the free VRAM the copies leave (0: each step's own default)
    bool lent_ = false;
    bool wedged_ = false;
    std::vector<std::pair<int32_t, int32_t>> spill_order_;
    std::vector<void*> copy_;
    CopyHeat heat_;
    std::vector<size_t> copy_slots_;   ///< the paid copies' pairs in range order
    std::vector<size_t> copy_candidates() const;
    bool join_placement();
    uint64_t copy_stride(size_t pair) const;
    const uint8_t* copy_source(size_t pair) const;
    bool publish_copy(size_t pair, const void* ptr);
    // the copies back to back in one reserved range (spill.hpp CopyLayout); its granules are mapped while a copy uses them,
    // one handle each from the range's start
    bool copy_range();
    bool trim_copies();
    /// With --vram-reserve-mib: kLateRoom, the room the copies leave for what loads after them (room() does not see it
    /// load); 0 without it.
    uint64_t late_room() const;
    int64_t room_ = 0;   ///< room(): may run below 0 when the caches grow past what placement counted
    uint8_t* copies_va_ = nullptr;
    CopyLayout copy_lay_;
    uint64_t copy_reserved_ = 0;
    std::vector<unsigned long long> copy_maps_;
    uint64_t epoch_ = 0;   ///< address invalidations
    bool copies_retired_ = false;   ///< a granule of the range could not be unmapped: no promote copies into it again
    std::vector<int> layer_copies_;   ///< [layer] RAM experts with a card copy (paid or loaned)
    // the prompt loan: the idle regions, the loaned (layer, expert) pairs in loan order (flag per pair, copy_ holds the address)
    bool loan_ready();
    bool queue_loan(uint64_t budget);
    bool drain_loan_upload();
    struct LoanUpload { size_t pair; uint8_t* ptr; };
    void* loan_stream_ = nullptr;
    void* loan_event_ = nullptr;
    std::vector<LoanUpload> loan_plan_;   ///< the loan fixed at its start: each pair and its place, in rank order
    size_t loan_next_ = 0;                ///< the plan's uploads queued so far
    bool loan_inflight_ = false;          ///< uploads queued since the copy stream was last confirmed idle
    bool loan_stuck_ = false;             ///< an upload could not be confirmed finished: its memory is never released
    uint8_t* loan_arena_ = nullptr;
    uint64_t loan_arena_bytes_ = 0;
    uint8_t* loan_stage_ = nullptr;   ///< the shared staging's second address range
    uint64_t loan_stage_bytes_ = 0;
    unsigned long long* loan_addr_ = nullptr;   ///< pinned: the table entries a loan writes
    std::vector<uint8_t> loan_flag_;
    std::vector<size_t> loan_list_;
    int loan_state_ = 0;              ///< 0: not set up, 1: ready, -1: not available
    bool loan_due_ = true;            ///< the next decode window starts a new ranked fill
    int64_t loan_publish_at_ = -1;    ///< the position the queued uploads are published at (-1: none queued)
    // [vision] the experts' memory lent to the image encoder: each lent range's bytes, in RAM until refill
    struct LentRange {
        size_t range = 0;
        std::unique_ptr<uint8_t[]> bytes;
    };
    std::vector<LentRange> lent_ranges_;
    int64_t lend_copies_ = 0;   ///< card copies demoted for the encoder
    double lend_ms_ = 0;
};

/// A session over `model`; null with `err` set.
std::unique_ptr<model::ModelSession> make_session(InklingModel& model, std::string& err);

/// The device bytes a session takes for `context` positions (fp8: one byte a cached value), its drafter aside, on card
/// `card` of a split at `cut` (0: one card, every visit).
uint64_t session_bytes(const Geometry& g, int64_t context, bool fp8, int cut = 0, int card = 0);
/// The device bytes every session shares once on a card: the forward's scratch, the prompt path, the drafter's tap (the
/// first card's).
uint64_t shared_bytes(const Geometry& g, bool drafter);

}  // namespace basalt::models::inkling2
