// src/models/inkling2/model.hpp - a loaded Inkling 2 pack: every weight on the main card (the projections of a layer fused
// into one Q8_0 matrix, the experts of a layer in one run of blobs at a stride of whole blocks in a reserved address range,
// the head rounded to Q8_0 as it loads), the token embedding in mapped host memory. With the layer split the second card
// holds the later layers' weights, experts and caches instead (cards(), card_of()).
#pragma once

#include "basalt/core/kv_pool.hpp"
#include "basalt/core/progress.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/model/adapter.hpp"
#include "models/inkling2/copy_heat.hpp"
#include "basalt/model/peer_hooks.hpp"
#include "basalt/model/split_hooks.hpp"
#include "models/inkling2/dspark.hpp"   // DSPARK HOOK
#include "models/inkling2/pack.hpp"
#include "models/inkling2/spill.hpp"
#include "models/inkling2/vision_encoder.hpp"   // [vision]
#include "models/inkling2/wide_chunk.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
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
    /// Card `card`'s KV pool: the full visits' cells of every session on it (the drafter's full layers on the main card).
    core::KvPool& kv_pool(int card) { return kv_pool_[card]; }
    const core::KvPool& kv_pool(int card) const { return kv_pool_[card]; }
    bool cache_int8() const { return int8_; }
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
    /// blob table entries point at the copies. A map the card refuses reads the card again, and copies go back, the lowest
    /// ranked first, until `keep_free` is free. Bytes copied.
    uint64_t promote(uint64_t keep_free);
    /// One warning when the last promote found less free memory than counted: placement then follows a fresh reading of
    /// the card, not the requests alone.
    void note_shortage(const std::string& where);
    /// A cache grow the card refused although the pool counted room: true for the first refusal of an episode (the caller
    /// warns once), false while the episode lasts; a grow that maps without a refusal ends it.
    bool grow_short() { return !std::exchange(grow_short_, true); }
    void grow_mapped() { grow_short_ = false; }
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
    /// the requests alone unless a map lacks the memory: then it reads the card again and a warning says so.
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
    /// `at`: the position it is published at (the log line names it).
    bool publish_loan(void* stream, int64_t at);
    /// Waits for the uploads, sets the loaned entries back to 0 (the CPU computes them again) and frees the regions.
    /// `event`: the log line's event.
    bool recall_loan(void* stream, const std::string& event = "prompt loan recall");
    int64_t loaned() const { return (int64_t) loan_list_.size(); }
    /// A prompt chunk's staging of layer l's RAM experts into their slots: each from its card copy while the elastic spill
    /// holds one, from RAM otherwise, a run of adjacent slots with no copy in one copy (spill.hpp stage_plan). `want`
    /// [slot]: only the slots set (null: every one). `with_peer`: the second card's experts' slots too, from their card
    /// copies (all_copied). False when a copy could not be queued.
    bool stage(int l, const uint8_t* want, void* stream, bool with_peer) const;
    /// Every expert of layer l that left the card has a copy on it again (the elastic spill): the layer runs on the main card alone.
    bool all_copied(int l) const { return !layer_copies_.empty() && layers_[(size_t) l].n_spill > 0 && layer_copies_[(size_t) l] >= layers_[(size_t) l].n_spill; }
    /// A prompt chunk sends layer l's rows to the second card: it holds experts of the layer that the main card has no copy of.
    bool peer_runs(int l) const { return peer_n(l) > 0 && !all_copied(l); }
    const SpillPlan& spill() const { return spill_; }
    /// gpu-mode peer (docs/architecture/multi-gpu.md): the experts that spilled from the main card and the second card holds in
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
    /// A long prompt read's wider chunk (wide_chunk.hpp): frees `bytes` more, the lowest-ranked card copies going back to RAM
    /// first, and counts them as taken. Refused, with nothing moved, when the copies hold less (by the counted room, never a
    /// fresh reading of the card); Failed when the copies could not be put back as they were.
    Take take_room(uint64_t bytes);
    /// The wider chunk's `bytes` are free again: the copies take_room sent back return, so the card holds the set it held.
    /// False when not every one of them came back.
    bool give_room(uint64_t bytes) noexcept;

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
    core::KvPool kv_pool_[2];
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
    /// Copies `pairs` to the card in that order while `keep_free` bytes stay free (promote's body).
    uint64_t promote_pairs(const std::vector<size_t>& pairs, uint64_t keep_free);
    int64_t room_ = 0;   ///< room(): may run below 0 when the caches grow past what placement counted
    bool map_short_ = false;   ///< a promote could not map a page it counted (note_shortage)
    bool grow_short_ = false;  ///< a cache grow found the card short and none has mapped cleanly since (grow_short)
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
    bool loan_stuck_ = false;             ///< an upload or a reader of the loan could not be confirmed finished: its memory is never released
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
    std::vector<size_t> wide_out_;   ///< the copies a long read's wider chunk sent back (take_room), until give_room
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

/// The device bytes a session holds of its own for `context` positions in the selected KV cache format, its drafter aside,
/// on card `card` of a split at `cut` (0: one card, every visit): the sliding-window visits' rings and its fixed buffers.
uint64_t session_bytes(const Geometry& g, int64_t context, bool fp8, bool int8, int cut = 0, int card = 0);
/// The KV pool bytes of card `card` the full visits' caches of one session take at `positions` positions, the drafter aside,
/// each cache's allocation granules of `gran` bytes counted whole.
uint64_t pool_bytes(const Geometry& g, int64_t positions, bool fp8, bool int8, uint64_t gran, int cut = 0, int card = 0);
/// The device bytes every session shares once on a card: the forward's scratch, the prompt path, the drafter's tap (the
/// first card's) and, on the card that draws (`head`), the sampler's list for a request without a top-k limit.
uint64_t shared_bytes(const Geometry& g, bool drafter, bool head);

}  // namespace basalt::models::inkling2
