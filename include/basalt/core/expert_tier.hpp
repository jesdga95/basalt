// include/basalt/core/expert_tier.hpp - where the routed experts of a pack live while a token runs.
//
// An expert's home is a slot of the main card, a slot of a second card, the pinned RAM tier or the pack file mapped
// read-only (core::policy::Homes); the routing order deals them out at the start. With experts left in the file, the RAM tier
// is also a cache of it: a lookup reads the file experts it routes to into the places of the layer's least recently used
// pinned ones (FileReads), and the CPU computes them from there. The main card's other slots (shared
// core::ExpertCache storage) hold copies of pinned experts; its homes and copies share a per-layer LRU, and a home that
// loses its slot, on either card, goes to RAM, into the place of the expert taking the slot. The GPUs compute the experts they
// hold; the CPU pool computes the rest from RAM or the mapped file while the GPUs work, and the pinned experts it
// computed are copied into the main card's LRU on a side stream so the next token finds them there.
#pragma once

#include "basalt/core/expert_cache.hpp"
#include "basalt/core/expert_map.hpp"
#include "basalt/core/model_file.hpp"
#include "basalt/core/loader.hpp"
#include "basalt/core/ram_load.hpp"
#include "basalt/program/expert_spill.hpp"
#include "basalt/program/probe_rates.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/kernels/route_lossy.hpp"
#include "basalt/prefill/peer_share.hpp"
#include "basalt/platform/mapped_file.hpp"
#include "basalt/core/file_reads.hpp"
#include "basalt/core/tier_policy.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace basalt::core {

/// What a model tells the tier about its routed experts and the flags that size it.
struct ExpertTierSpec {
    /// One layer's experts: a blob is [gate | up | down] and the layers' blobs are laid out one expert after another from
    /// `offset` in the experts section.
    struct Layer {
        int gu_type = 0, d_type = 0;
        uint64_t offset = 0, blob_bytes = 0;
        float clamp = 0;   ///< the SwiGLU clamp of the layer's experts (0: none)
    };
    std::string area;   ///< log area, "basalt <model>"
    std::string name;   ///< the model's name in refusals
    int n_layers = 0, n_expert = 0, top_k = 0;
    int n_embd = 0;     ///< width of an expert's input (the layer's input, or the latent a model projects it to)
    int n_ff = 0;       ///< width of an expert's hidden layer
    std::vector<Layer> layers;
    std::string path;           ///< the pack
    int fd = -1;                ///< a descriptor of it
    FileRange range;            ///< the experts section
    std::vector<int> peer_devices;   ///< the second cards, in block order
    int diag_tier_ram = -1, diag_tier_vram = -1;   ///< fixed experts per layer (-1: sized from free memory)
    bool diag_tier_frozen = false;
    int vram_reserve_mib = 0, peer_reserve_mib = 600;
    int expert_cache = 0;
    uint64_t resident_budget = 0;
    int pool_workers = 0;
    kernels::cpu::PoolAffinity pool_affinity = kernels::cpu::PoolAffinity::All;
    bool pool_affinity_given = false;
    std::string expert_profile;   ///< the routing order of the experts (empty: the file's own)
    /// CPU uses that earn a slot (policy::count_use) for every placement; 0: the repeat-miss rule with RAM holding all
    /// experts and kPromoteUses with the file's, the way the first model ran them.
    int admit_uses = 0;
    /// Copies (bus fetches and promotions) a window may claim per row; 0: 128 a window whatever its rows, the way the first
    /// model ran them. A window's rows route to mostly different experts, so a per-row budget keeps their admissions.
    int claims_per_row = 0;
    /// Tokens after which the lookup counts that pick the main card's victims halve; 0: the slot used longest ago gives way,
    /// the way the first model ran them. With N, the slot whose expert was looked up least lately gives way (every lookup
    /// counts, hits too, and the counts halve every N tokens), the one used longest ago among equals: an expert that recurs
    /// over hundreds of tokens keeps its slot through a run of experts that come once.
    int victim_half_life = 0;
    /// The main card's copies per layer in these proportions (one weight per layer; core::policy::layer_slots spreads the
    /// same total), for a model whose layers miss at different rates. Empty: every layer the same, the way the first model
    /// ran them. Main card homes are always the same in every layer.
    std::vector<float> layer_weights;
    /// Bytes of each segment the base arena of copies is mapped in, so a prompt or the caches can borrow part of it
    /// (ExpertTier::lend) and the rest keep their experts; 0: one allocation, which only lend_all gives up.
    uint64_t lend_segment = 0;
    /// The prompt path's host buffers, which are allocated before the RAM tier is sized beside them. Each is optional.
    struct Host {
        std::function<bool(std::string&)> open;                 ///< before the RAM tier is sized
        std::function<uint64_t()> bytes;                        ///< what `open` took
        std::function<bool(int, std::string&)> set_ring;        ///< the pinned ring of file reads, `ring_pieces` experts
        std::function<void()> ram_only;                         ///< no expert is left in the file
    } host;
    int ring_pieces = 0;
    uint64_t share_stride = 0;   ///< bytes between experts in a second card's prompt staging (the largest layer's)
    int share_max_tokens = 0;    ///< most tokens of a prompt chunk
    /// --route-tail-skip R and --route-prior LAMBDA (basalt/model/route_flags.hpp; 0 = off): the tier keeps the device
    /// table of the experts the GPUs hold that the model's decode windows route with (ExpertTier::publish_held).
    int route_tail_skip = 0;
    float route_prior = 0.0f;
};

class ExpertTier {
public:
    static constexpr int kMaxK = 128;     ///< distinct experts one lookup holds (a window of up to 8 rows, 16 each)
    static constexpr int kMaxWindow = 8;

    /// The tier lookup of one layer's routed experts.
    struct Lookup {
        int k = 0;
        const uint8_t* dev[kMaxK] = {};    ///< device blob of an expert the VRAM tier holds, else null
        const uint8_t* peer[kMaxK] = {};   ///< device blob of an expert the second card holds or fetched, else null
        const uint8_t* host[kMaxK] = {};   ///< host blob of the others (and of those the second card fetched)
        int experts[kMaxK] = {};
        bool again[kMaxK] = {};            ///< a miss that repeats one from the last kPromoteWindow tokens
        bool fetched[kMaxK] = {};          ///< `dev` or `peer` is being copied in by `fetch`
        bool reading[kMaxK] = {};          ///<  is a RAM place a file read is filling (await_reads)
        bool dropped[kMaxK] = {};          ///< id -1: --route-tail-skip dropped the entry; nothing computes it
    };

    /// The window rows that use one looked-up expert: row i of token tok[i] is `row[i]` of the output rows.
    struct Use {
        int nt = 0;
        int tok[kMaxWindow] = {};
        int row[kMaxWindow] = {};
    };

    explicit ExpertTier(ExpertTierSpec spec);
    ~ExpertTier();
    ExpertTier(const ExpertTier&) = delete;
    ExpertTier& operator=(const ExpertTier&) = delete;

    /// Main card bytes each arrangement leaves free.
    struct Reserve {
        uint64_t copies = 0;       ///< beside slots that are all copies, which a long prompt borrows
        uint64_t homes = 0;        ///< beside homes, which stay through a prompt at the full context
        uint64_t beside_homes = 0; ///< beside homes and the copies next to them, where a short prompt's workspaces fit
        uint64_t modules = 0;      ///< ... until a first token has loaded the kernel modules
    };
    /// Sizes the tiers and queues their experts' reads in `reads` (the start's one pass places them). `h_rows`: the mapped
    /// rows a second card shares with the CPU pool.
    bool open(const Reserve& reserve, float* h_rows, core::LoadPlan& reads, std::string& err);
    /// Once the experts have landed: the RAM tier pinned, the main card's copies seeded, and the slots beside the base arena
    /// planned once (an exact --expert-cache that does not fit stops the start here) and opened.
    bool attach(std::string& err);

    /// The tier's size is planned once, at the start (`attach`); nothing reads the free VRAM afterwards. A prompt
    /// borrows slots inside that total and gives the same number back.
    ///
    /// Gives the VRAM slots that the prompt path's workspaces occupy back to the driver (call before they are
    /// allocated); `grow` opens the planned slots beside the base arena again once they are free, with the experts a
    /// shrink or lend took off the card (most recently used first) and then the profile's ranked ones.
    void shrink();
    void grow();
    /// A long prompt streams every expert to the main card and needs the card for its chunks: the slots that are
    /// copies go back whole, and reclaim() opens the same number again (an error if the card cannot hold them any
    /// more): empty, or with `restore` holding the experts the lend took off the card, the most recently used first,
    /// copied before it returns. Homes stay.
    void lend_all();
    bool reclaim(std::string& err, bool restore = false);
    /// Whether lend() can give up part of the base arena (ExpertTierSpec::lend_segment, copies, a segmented arena).
    bool partial_lend() const { return !live_.empty(); }
    /// Leaves `bytes` device bytes free above the floor by giving up the fewest slots: the ones beside the base arena,
    /// then each layer's last base slots in proportion to its share (the arena's tail holds every layer's last ones).
    /// First the most recently used experts of each layer move into the slots it keeps. False, with nothing given up,
    /// when all of them would not free that much (or without partial_lend). reclaim() maps them again (refilled with restore).
    bool lend(uint64_t bytes);
    /// An explicit new --vram-reserve-mib (`mib` < 0: the start's): the one time the size is planned again after the
    /// start. The slots that fit above the new reserve are opened; the response reports them.
    bool replan(int64_t mib, std::string& err);
    /// Main card bytes the tier, the caches and the prompt path leave free: --vram-reserve-mib, or `mib` once set
    /// (`mib` < 0: back to the start's). The arena is not resized here.
    void set_free_floor(int64_t mib);
    uint64_t free_floor() const { return free_floor_; }

    /// Called before each token or window. The copies queued during the previous one (promotions, second card exchanges)
    /// land layer by layer: each before its layer's next lookup (`pump`), so what a lookup finds depends only on the
    /// routing history, as if every copy had landed at the window's start.
    void begin_token(int n_tokens = 1);
    /// A prompt starts, which may change the topic: the use counts that pick the main card's victims start again (with
    /// ExpertTierSpec::victim_half_life; the slots keep their experts).
    void forget_use();
    /// Looks up `ids[0..k)` of `layer` and stamps them as used; `layer`'s queued copies land first. The file experts
    /// among them start reading into the RAM tier (`reading`). An id of -1 (route_tail_skip dropped it) is `dropped`: no
    /// source, no stamp, and fetch, promote and the CPU's share pass it by.
    bool lookup(int layer, const int32_t* ids, int k, Lookup& lk, std::string& err);
    /// The reads the last lookup started have landed.
    bool await_reads(std::string& err);
    /// The CPU's share: the experts of `lk` without a device blob, over the activation `x` (n_embd floats, host).
    /// Row i of `rows` (n_embd floats each) gets expert i's output. Started here, it runs on the pool's workers while
    /// the caller launches the GPUs' shares; finish_cpu joins it. `lk` and `rows` stay as they are until then.
    /// `reading` picks the experts being read instead of the others: they start once finish_cpu and await_reads returned.
    void start_cpu(int layer, const float* x, const Lookup& lk, float* rows, bool reading = false);
    /// The same over a window: `x` holds a row per token and each expert i runs over the rows `use[i]` names, into those
    /// rows of `rows`.
    void start_cpu_window(int layer, const float* x, const Lookup& lk, const Use* use, float* rows, bool reading = false);
    void finish_cpu();
    /// The promotions' copies, and the second card's exchanges, wait for the bus to idle: mark_gap notes the point of
    /// `main` (a layer's sum) they start behind, and pump issues up to `pieces` promotion pieces there (and at most
    /// core::policy::kPeerGapPieces exchange pieces on the second card's bus), after the next launches went out (the next
    /// layer's head; at the window's end, its output), so they do not take the host memory's bandwidth from the CPU's
    /// share and the bus copies of a layer's misses. A copy queued after them on another stream (a layer's bus fetch)
    /// waits for one piece at most. With `layer` >= 0 (called before the layer's lookup) that layer's queued copies first
    /// go out whole and land.
    void mark_gap(cudaStream_t main);
    void pump(int layer, int pieces);
    static constexpr int kAllPieces = 1 << 20;
    /// Starts the bus copy of the pinned misses of `lk` beyond the CPU's half (core::policy::bus_count) into the VRAM tier; those
    /// experts get a device blob in `lk`, marked `fetched`, so the GPU computes them while the CPU does the rest. With a
    /// second card, core::policy::peer_bus_count of them cross its own link into its scratch slots instead (`peer`, marked
    /// `fetched`); their VRAM tier copies follow off the token path. How many the main card's link started; `await_fetch`
    /// makes `main` wait for them.
    int fetch(int layer, Lookup& lk);
    void await_fetch(cudaStream_t main);
    /// Queues the pinned misses the CPU computed that missed before within kPromoteWindow tokens for a copy into the
    /// VRAM tier (side stream, off the token path, in pieces).
    void promote(int layer, const Lookup& lk);

    /// The copies under way land: every expert is in its place. A prompt reads the places, so it comes first.
    void settle();
    /// Where a resident copy of (layer, e) lives: a main card slot, a second card slot or the pinned RAM tier; all null
    /// when only the file has it. Read-only: no stamp, no promotion.
    struct Source {
        const uint8_t* dev = nullptr;
        const uint8_t* peer = nullptr;
        const uint8_t* pinned = nullptr;
    };
    Source locate(int layer, int e) const;
    /// The first second card (null without any): for "is there a second card" questions; per-layer work takes peer_for.
    core::PeerExperts* peer() const { return peers_.empty() ? nullptr : peers_[0].get(); }
    /// The second card that holds `layer`'s second-card homes (each card owns a block of layers), or null.
    core::PeerExperts* peer_for(int layer) const {
        return peers_.empty() ? nullptr : peers_[(size_t) layer_card_[(size_t) layer]].get();
    }
    /// Every second card, in block order (for work each card does once, such as loading kernel modules).
    const std::vector<std::unique_ptr<core::PeerExperts>>& peers() const { return peers_; }
    /// The first second card's share of a prompt chunk's experts; null without a second card.
    prefill::PeerShare* peer_share() const { return shares_.empty() ? nullptr : shares_[0].get(); }
    /// The share of `layer`'s prompt experts, opened on peer_for(layer)'s card; null without a second card.
    prefill::PeerShare* share_for(int layer) const {
        return shares_.empty() ? nullptr : shares_[(size_t) layer_card_[(size_t) layer]].get();
    }
    /// What the split of a layer's experts between the cards prices them at.
    const prefill::ShareRates& share_rates(int layer) const { return rates_[(size_t) layer]; }
    kernels::cpu::ExpertPool* pool() const { return pool_.get(); }
    const kernels::cpu::NativeFmt& fmt(int layer) const { return fmt_[(size_t) layer]; }

    /// The VRAM slots open now, every layer's.
    int64_t slots() const { return live_base() + (int64_t) n_layers_ * ext_per_layer_; }
    /// The experts the GPUs hold now: the main card's (the caller's device) and each second card's slots and the VRAM their
    /// slots take; and the slots' sum, what the engine reports as its expert slots.
    struct CardSlots { int device = 0; int64_t slots = 0; uint64_t bytes = 0; };
    std::vector<CardSlots> slots_by_card() const;
    int64_t gpu_slots() const;
    /// The bytes of the pinned RAM tier's places, every layer's.
    uint64_t ram_bytes() const;

    /// The lossy routing flags (ExpertTierSpec::route_*) this tier applies: 0 when off, or when every expert is held by a
    /// GPU (attach switches them off then, as nothing would be skipped and a bonus for every expert changes a biased choice).
    int route_tail_skip() const { return route_tail_skip_; }
    float route_prior() const { return route_prior_; }
    bool route_lossy() const { return route_tail_skip_ > 0 || route_prior_ > 0.0f; }
    /// A decode window starts: the device table of the experts the GPUs hold now (a main card or second card slot) goes
    /// out on `stream` for the window's routers, the layers that changed since the last window. Copies that land during
    /// the window count from the next one. Only with route_lossy().
    void publish_held(cudaStream_t stream);
    /// `layer`'s held table as the routing kernels read it (empty without route_lossy()).
    kernels::ResidentTabs held_tabs(int layer) const;
    /// The tail skip's counters (basalt/kernels/route_lossy.hpp), device; null without route_lossy().
    unsigned long long* route_counts() const { return route_counts_; }
    /// n_embd zeros on the main card: the output row of a dropped entry.
    const float* zero_row() const { return zero_row_; }

    void report(const char* tag) const;
    /// The routed experts so far: hits of the main card's slots, those plus the CPU's, a bus copy's or the second card's.
    int64_t hits() const { return hits_; }
    int64_t lookups() const { return hits_ + cpu_; }
    int64_t offloaded() const { return fetched_ + peer_hits_; }
    /// Of the CPU's, those read from the file.
    int64_t file_lookups() const { return file_hits_; }
    /// The experts only the file holds.
    program::ExpertSpill spill() const;
    /// The server's expert map: the sample's pairs of all, each one's home and its lookups since the last call (the first
    /// call starts counting them).
    void expert_map(int64_t& total, std::vector<uint8_t>& homes, std::vector<uint32_t>& routed);
    /// What `PROBE rates` measures: this tier's pool, RAM blobs, formats and second card.
    void probe_env(program::ProbeEnv& env);

private:
    bool plan_ext(std::string& err);
    bool reopen_base(int64_t n, std::string& err);
    /// A slot's copy on its way. `out` >= 0: the home it replaces is on its way to RAM place `place`.
    struct Pending {
        int layer, expert, slot, event;
        int out = -1, place = -1;
    };
    /// A promotion whose copies wait for an idle bus: the home's bytes out to `home_to` (on the out stream), then `src`
    /// into `dst` once they are out.
    struct Queued {
        Pending p;
        uint8_t* dst;
        const uint8_t* src;
        uint8_t* home_to;
        uint64_t bytes, off = 0, home_off = 0;
    };
    /// `in` on its way into the second card's `slot`, its pieces waiting for an idle bus; `out`, which held it, on its
    /// way to RAM place `place`: the slot's bytes out to `to` first, then `from` in.
    struct PeerMove {
        int layer, in, out, slot, place;
        uint8_t* to;
        const uint8_t* from;
        uint64_t bytes, out_off = 0, in_off = 0;
        int64_t in_mark = -1;   ///< the second card's mark behind its last piece
    };
    /// Issues what is left of `layer`'s queued copies (>= 0), then up to `pieces` pieces of the others, oldest first:
    /// the main card's and the second card's.
    void issue(int layer, int pieces);
    void issue_peer(int layer, int pieces);
    /// The copies queued for `layer` (or every layer) land: the experts take their places.
    void land(int layer);
    void land_all();
    void apply(const Pending& p);
    void apply(const PeerMove& m);
    bool load_ram(core::LoadPlan& reads, std::string& err);
    bool land_ram(std::string& err);
    bool open_share(size_t card, std::string& err);
    void fill_share_rates();
    int ranked(int layer, int r) const;
    bool load_rank(std::string& err);
    /// `n` copies in each layer, or with `base_shares` each layer's base arena share (base_n).
    void seed_copies(int64_t n, bool base_shares = false);
    /// [layer] the experts a shrink, lend or lend_all took off the card that no restore has put back, the most recently used
    /// first; `recall` adds the owners of slots [lo, lo + n) of `layer` (call before they are cleared).
    std::vector<std::vector<int32_t>> recall_;
    void recall(int layer, int64_t lo, int64_t n);
    bool recalled() const;
    /// Fills each layer's empty slots with its recall_ experts, then with `ranked_too` the profile's order, and waits for
    /// the copies (a placement the next lookup sees whatever the timing).
    void restore(bool ranked_too);
    bool pinned(int layer, int e) const { return ram_idx_[(size_t) layer * n_expert_ + e] >= 0; }
    /// Gives the file experts of `ids` the RAM places of the layer's least recently used pinned experts outside `ids`
    /// and starts reading them there; `reading[i]` marks those of `ids`.
    bool read_misses(int layer, const int32_t* ids, int k, bool* reading, std::string& err);
    const uint8_t* blob_of(int layer, int e) const;
    const uint8_t* slot_ptr(int32_t slot) const;
    uint64_t slot_total() const { return (uint64_t) (base_slots() + (int64_t) n_layers_ * ext_per_layer_); }   ///< slot numbers in use
    /// Copies (layer, e) from `src` into the least recently used free slot of the layer; the slot, or -1. With `homes_too`
    /// that slot may be a home's, when the layer has a free RAM place for it. `whole`: one copy, for a layer that waits
    /// on it; else pieces, which let a waited copy pass. With `later` the pieces wait in the queue for pump.
    int32_t claim(int layer, int e, const uint8_t* src, cudaStream_t stream, bool homes_too = false, bool whole = false,
                  bool later = false);
    /// The free slot of the layer's main card share that gives way first (the least recently used; with victim_half_life the
    /// least looked up, skipping experts this very lookup uses), or -1; `best` gets its last use (the current tick when
    /// every slot is in use by this lookup).
    int32_t oldest_slot(int layer, bool homes_too, uint64_t& best) const;
    /// Moves (layer, e) into the second card when the expert it replaces there was last used core::policy::kPeerMoveAge
    /// lookups before the main card's oldest.
    bool move_to_peer(int layer, int e, const uint8_t* src);

    const ExpertTierSpec s_;
    core::ExpertCache cache_;
    int n_layers_ = 0, n_expert_ = 0;
    int64_t per_layer_ = 0;                 ///< VRAM slots per layer in the base arena (their mean, with layer_weights)
    std::vector<int64_t> base_lo_;          ///< [layer] its first base arena slot; [n_layers] the arena's size
    /// The base arena's layout for `per_layer` slots a layer on average: the layers' shares (set_base), each layer's, all.
    void set_base(int64_t per_layer);
    int64_t base_n(int layer) const { return base_lo_[(size_t) layer + 1] - base_lo_[(size_t) layer]; }
    int64_t base_slots() const { return base_lo_.empty() ? 0 : base_lo_.back(); }
    /// Opens the base arena for `per_layer` slots a layer on average (set_base), segmented with ExpertTierSpec::lend_segment.
    bool open_base(int64_t per_layer, std::string& err);
    /// With a segmented arena: [base slot] its place in the arena, each layer's k-th slot at depth (k + 1/2) / base_n, so
    /// cutting the arena's tail leaves every layer a prefix of its slots in proportion to its share. Empty: the slot's own.
    std::vector<int32_t> phys_;
    /// With a segmented arena: [layer] its base slots mapped now (a prefix); empty without one (all of base_n).
    std::vector<int64_t> live_;
    int64_t live_n(int layer) const { return live_.empty() ? base_n(layer) : live_[(size_t) layer]; }
    int64_t live_base() const;
    void count_live();
    /// Moves the most recently used experts of each layer into its first keep[layer] base slots and clears the rest.
    void keep_hottest(const std::vector<int64_t>& keep);
    core::ExpertCache ext_;                 ///< slots beside the base arena, open only while no prompt workspace is
    int64_t ext_per_layer_ = 0;             ///< slots each layer owns in `ext_`
    bool ram_queued_ = false;               ///< the RAM tier's reads are in the start's pass
    std::chrono::steady_clock::time_point ram_t0_{};
    int64_t ext_n_ = 0;                     ///< the slots per layer `grow` opens in `ext_`: planned at the start (and by replan)
    int64_t exact_per_layer_ = 0;           ///< an exact --expert-cache, per layer; 0 when auto
    int64_t lent_per_layer_ = 0;            ///< base slots given up by lend_all(), 0 when none
    int64_t planned_per_layer_ = 0;         ///< the base slots a reclaim opens again when they fit (the largest lent)
    std::vector<int32_t> res_;              ///< [layer * n_expert + e] VRAM slot or -1
    std::vector<uint64_t> stamp_;           ///< [layer * n_expert + e] last use
    std::vector<uint16_t> looks_;           ///< [layer * n_expert + e] lookups, halved every victim_half_life tokens (empty when 0)
    std::vector<int32_t> owner_;            ///< [slot] expert it holds or is receiving, or -1
    std::vector<uint8_t> busy_;             ///< [slot] a promotion is copying into it
    std::vector<Pending> pending_;          ///< issued copies not landed yet
    std::vector<Queued> queue_;             ///< promotions not yet fully issued, oldest first
    std::vector<PeerMove> peer_moves_;      ///< second card exchanges not landed yet, oldest first
    std::vector<int32_t> peer_owner_;       ///< [second card slot] expert it holds or is receiving
    std::vector<cudaEvent_t> events_;       ///< [event] behind a copy's last piece in
    std::vector<cudaEvent_t> out_events_;   ///< [event] behind the last piece out of the home a promotion replaces
    std::vector<int> free_events_;
    int claims_ = 0;                        ///< copies claimed since the last begin_token (at most claim_cap_)
    int claim_cap_ = 0;                     ///< what this window may claim
    cudaStream_t copy_stream_ = nullptr;    ///< promotions, off the token path
    cudaStream_t out_stream_ = nullptr;     ///< the homes they replace, out to RAM beside them
    cudaStream_t fetch_stream_ = nullptr;   ///< copies a layer waits for (high priority)
    cudaEvent_t fetch_ev_ = nullptr;
    cudaEvent_t gap_ev_ = nullptr;          ///< the main stream's point the pumped pieces start behind
    bool gap_marked_ = false;               ///< gap_ev_ was recorded for the next pump
    uint64_t tick_ = 0;
    uint64_t one_each_ = 0, reserve_ = 0, modules_ = 0;
    uint64_t start_floor_ = 0, free_floor_ = 0;
    int64_t cap_per_layer_ = -1;            ///< --expert-cache or the diagnostic's slots per layer, -1 when neither

    std::vector<core::RamAllocation> ram_allocations_;
    std::vector<core::RamPin> ram_pins_;    ///< [layer] what each span's pin got
    std::vector<uint8_t*> ram_;             ///< [layer] cover_ + 1 pinned expert slots (the last starts as the spare)
    core::policy::Homes homes_;
    std::vector<int32_t> rank_;             ///< [layer * n_expert + rank] expert; empty: the file's order
    int cover_ = 0;                         ///< experts per layer in the RAM tier
    std::vector<int32_t> ram_idx_;          ///< [layer * n_expert + e] pinned slot or -1
    std::vector<int32_t> spare_;            ///< [layer] pinned slot free for a home leaving the main card, or -1
    std::vector<uint64_t> last_miss_;       ///< [layer * n_expert + e] token + 1 of the last miss
    uint64_t token_ = 0;
    platform::MappedFile file_;             ///< the experts section
    core::FileReads reads_;                       ///< open when the file holds experts and the RAM tier caches them
    std::vector<core::FileReads::Blob> blobs_;    ///< the reads of the last lookup
    std::vector<uint16_t> uses_;            ///< [layer * n_expert + e] CPU uses toward a promotion, with reads_ open
    int64_t read_ = 0;                      ///< experts read into the RAM tier
    uint64_t host_bytes_ = 0;               ///< what the model's host buffers took
    std::vector<std::unique_ptr<core::PeerExperts>> peers_;   ///< [card] the second cards, each holding blocks_'s layers
    std::vector<std::unique_ptr<prefill::PeerShare>> shares_; ///< [card] the prompt share on the same card
    core::policy::PeerBlocks blocks_;             ///< card c holds the second-card homes of layers [lo[c], hi[c])
    std::vector<int> layer_card_;           ///< [layer] its card in peers_ / shares_
    std::vector<prefill::ShareRates> rates_;    ///< [layer]
    std::vector<kernels::cpu::NativeFmt> fmt_;   ///< [layer]
    std::unique_ptr<kernels::cpu::ExpertPool> pool_;
    uint8_t* act_ = nullptr;                ///< the quantized activation the CPU jobs share
    kernels::cpu::ExpertJobMulti cpu_jobs_[kMaxK];   ///< the CPU share started and not yet joined

    int route_tail_skip_ = 0;
    float route_prior_ = 0.0f;
    bool open_route(std::string& err);
    bool held(int layer, int e) const;
    int32_t* held_dev_ = nullptr;            ///< [layer * n_expert + e] 0 when a GPU holds it, else -1 (device)
    int32_t* held_pin_ = nullptr;            ///< its staging (pinned host)
    std::vector<int32_t> held_pub_;          ///< what was published last
    cudaEvent_t held_ev_ = nullptr;          ///< behind the last publish's copies
    unsigned long long* route_counts_ = nullptr;
    float* zero_row_ = nullptr;

    core::ExpertMapTally map_;              ///< the expert map's sample, counted from the first expert_map()
    int64_t hits_ = 0, cpu_ = 0, ram_hits_ = 0, promoted_ = 0, fetched_ = 0, dropped_ = 0, peer_hits_ = 0, file_hits_ = 0;
};

}  // namespace basalt::core
