// src/models/deepseek41/tier.hpp - where the routed experts of a DeepSeek-V4.1 pack live while a token runs.
//
// An expert's home is a slot of the main card, a slot of a second card, the pinned RAM tier or the pack file mapped
// read-only (policy::Homes); the routing order deals them out at the start. The main card's other slots (shared
// core::ExpertCache storage) hold copies of pinned experts; its homes and copies share a per-layer LRU, and a home that
// loses its slot, on either card, goes to RAM, into the place of the expert taking the slot. The GPUs compute the experts they
// hold; the CPU pool computes the rest from RAM or the mapped file while the GPUs work, and the pinned experts it
// computed are copied into the main card's LRU on a side stream so the next token finds them there.
#pragma once

#include "basalt/core/expert_cache.hpp"
#include "basalt/core/ram_load.hpp"
#include "basalt/program/expert_spill.hpp"
#include "basalt/program/probe_rates.hpp"
#include "basalt/core/peer_experts.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/platform/mapped_file.hpp"
#include "models/deepseek41/tier_policy.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace basalt::models::deepseek41 {

class DsModel;

class ExpertTier {
public:
    static constexpr int kMaxK = 48;      ///< distinct experts one lookup holds (a window of up to 8 rows, 6 each)
    static constexpr int kMaxWindow = 8;

    /// The tier lookup of one layer's routed experts.
    struct Lookup {
        int k = 0;
        const uint8_t* dev[kMaxK] = {};    ///< device blob of an expert the VRAM tier holds, else null
        const uint8_t* peer[kMaxK] = {};   ///< device blob of an expert the second card holds, else null
        const uint8_t* host[kMaxK] = {};   ///< host blob of the others
        int experts[kMaxK] = {};
        bool again[kMaxK] = {};            ///< a miss that repeats one from the last kPromoteWindow tokens
        bool fetched[kMaxK] = {};          ///< `dev` is being copied in by `fetch`
    };

    /// The window rows that use one looked-up expert: row i of token tok[i] is `row[i]` of the output rows.
    struct Use {
        int nt = 0;
        int tok[kMaxWindow] = {};
        int row[kMaxWindow] = {};
    };

    explicit ExpertTier(const DsModel& model);
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
    /// Sizes the tiers and loads them. `h_rows`: the mapped rows a second card shares with the CPU pool.
    bool open(const Reserve& reserve, float* h_rows, std::string& err);

    /// Gives the VRAM slots that the prompt path's workspaces occupy back to the driver (call before they are
    /// allocated); `grow` takes the memory freed after them as extra slots. Residency in those slots is dropped.
    void shrink();
    void grow();
    /// A long prompt streams every expert to the main card and needs the card for its chunks: the slots that are
    /// copies go back whole, and reclaim() opens them again empty. Homes stay.
    void lend_all();
    bool reclaim(std::string& err);
    /// Main card bytes the tier, the caches and the prompt path leave free: --vram-reserve-mib, or `mib` once set
    /// (`mib` < 0: back to the start's). The arena is not resized here.
    void set_free_floor(int64_t mib);
    uint64_t free_floor() const { return free_floor_; }

    /// Called before each token or window: the promotions queued during the previous one finish and become resident, so
    /// what a lookup finds depends only on the routing history.
    void begin_token(int n_tokens = 1);
    /// Looks up `ids[0..k)` of `layer` and stamps them as used.
    bool lookup(int layer, const int32_t* ids, int k, Lookup& lk, std::string& err);
    /// The CPU's share: the experts of `lk` without a device blob, over the activation `x` (n_embd floats, host).
    /// Row i of `rows` (n_embd floats each) gets expert i's output.
    bool run_cpu(int layer, const float* x, const Lookup& lk, float* rows, std::string& err);
    /// The same over a window: `x` holds a row per token and each expert i runs over the rows `use[i]` names, into those
    /// rows of `rows`.
    bool run_cpu_window(int layer, const float* x, const Lookup& lk, const Use* use, float* rows, std::string& err);
    /// Starts the bus copy of the pinned misses of `lk` beyond the CPU's half (policy::bus_count) into the VRAM tier; those
    /// experts get a device blob in `lk`, marked `fetched`, so the GPU computes them while the CPU does the rest. How many
    /// started; `await_fetch` makes `main` wait for them.
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
    core::PeerExperts* peer() const { return peer_.get(); }
    kernels::cpu::ExpertPool* pool() const { return pool_.get(); }
    const kernels::cpu::NativeFmt& fmt(int layer) const { return fmt_[(size_t) layer]; }
    /// The main card holds homes: they stay through a prompt, and a short prompt keeps the copies beside them.
    bool has_homes() const { return homes_.main > 0; }

    /// The VRAM slots open now, every layer's.
    int64_t slots() const { return (int64_t) n_layers_ * (per_layer_ + ext_per_layer_); }

    void report(const char* tag) const;
    /// The routed experts so far: hits of the main card's slots, those plus the CPU's, a bus copy's or the second card's.
    int64_t hits() const { return hits_; }
    int64_t lookups() const { return hits_ + cpu_; }
    int64_t offloaded() const { return fetched_ + peer_hits_; }
    /// Of the CPU's, those read from the file.
    int64_t file_lookups() const { return file_hits_; }
    /// The experts only the file holds.
    program::ExpertSpill spill() const;
    /// What `PROBE rates` measures: this tier's pool, RAM blobs, formats and second card.
    void probe_env(program::ProbeEnv& env);

private:
    /// A slot's copy on its way. `out` >= 0: the home it replaces is on its way to RAM place `place`.
    struct Pending {
        int layer, expert, slot, event;
        int out = -1, place = -1;
    };
    /// `in` on its way into the second card's `slot`; `out`, which held it, on its way to RAM place `place`.
    struct PeerMove {
        int layer, in, out, slot, place;
    };
    /// A cold pinned expert (victim) replaced by a hot file-backed one (expert), read into the layer's spare slot.
    struct Swap {
        int layer, expert, victim;
    };
    bool load_ram(std::string& err);
    int ranked(int layer, int r) const;
    bool load_rank(std::string& err);
    void seed_copies(int first_rank, int64_t n);
    void preread();
    bool pinned(int layer, int e) const { return ram_idx_[(size_t) layer * n_expert_ + e] >= 0; }
    const uint8_t* blob_of(int layer, int e) const;
    const uint8_t* slot_ptr(int32_t slot) const;
    uint64_t slot_total() const { return (uint64_t) n_layers_ * (uint64_t) (per_layer_ + ext_per_layer_); }
    void drain();
    void finish_swaps();
    void start_swaps();
    void swap_worker();
    /// Copies (layer, e) from `src` into the least recently used free slot of the layer; the slot, or -1. With `homes_too`
    /// that slot may be a home's, when the layer has a free RAM place for it.
    int32_t claim(int layer, int e, const uint8_t* src, cudaStream_t stream, bool homes_too = false);
    /// The least recently used free slot of the layer's main card share, or -1; `best` gets its last use.
    int32_t oldest_slot(int layer, bool homes_too, uint64_t& best) const;
    /// Moves (layer, e) into the second card when the expert it replaces there was used before any on the main card.
    bool move_to_peer(int layer, int e, const uint8_t* src);

    const DsModel& m_;
    core::ExpertCache cache_;
    int n_layers_ = 0, n_expert_ = 0;
    int64_t per_layer_ = 0;                 ///< VRAM slots each layer owns in the base arena
    core::ExpertCache ext_;                 ///< slots beside the base arena, open only while no prompt workspace is
    int64_t ext_per_layer_ = 0;             ///< slots each layer owns in `ext_`
    int64_t lent_per_layer_ = 0;            ///< base slots given up by lend_all(), 0 when none
    std::vector<int32_t> res_;              ///< [layer * n_expert + e] VRAM slot or -1
    std::vector<uint64_t> stamp_;           ///< [layer * n_expert + e] last use
    std::vector<int32_t> owner_;            ///< [slot] expert it holds or is receiving, or -1
    std::vector<uint8_t> busy_;             ///< [slot] a promotion is copying into it
    std::vector<Pending> pending_;
    std::vector<PeerMove> peer_moves_;
    std::vector<int32_t> peer_owner_;       ///< [second card slot] expert it holds or is receiving
    std::vector<cudaEvent_t> events_;
    std::vector<int> free_events_;
    cudaStream_t copy_stream_ = nullptr;    ///< promotions, off the token path
    cudaStream_t fetch_stream_ = nullptr;   ///< copies a layer waits for (high priority)
    cudaEvent_t fetch_ev_ = nullptr;
    uint64_t tick_ = 0;
    uint64_t one_each_ = 0, reserve_ = 0, modules_ = 0;
    uint64_t start_floor_ = 0, free_floor_ = 0;
    int64_t cap_per_layer_ = -1;            ///< --expert-cache or the diagnostic's slots per layer, -1 when neither

    std::vector<core::RamPin> ram_pins_;    ///< [layer] what each span's pin got
    std::vector<uint8_t*> ram_;             ///< [layer] cover_ + 1 pinned expert slots (the last starts as the spare)
    policy::Homes homes_;
    std::vector<int32_t> rank_;             ///< [layer * n_expert + rank] expert; empty: the file's order
    int cover_ = 0;                         ///< experts per layer in the RAM tier
    std::vector<int32_t> ram_idx_;          ///< [layer * n_expert + e] pinned slot or -1
    std::vector<int32_t> spare_;            ///< [layer] pinned slot free for a swap, or -1
    std::vector<uint64_t> spare_since_;     ///< [layer] token at which the spare became free
    std::vector<uint64_t> last_miss_;       ///< [layer * n_expert + e] token + 1 of the last miss
    std::vector<std::pair<int, int>> want_; ///< file-backed experts that missed twice, in lookup order
    std::vector<Swap> flight_;              ///< swaps started at the previous begin_token
    std::thread worker_;
    std::thread preread_;
    std::atomic<bool> preread_quit_{false};
    std::mutex mu_;
    std::condition_variable cv_;
    std::vector<Swap> jobs_;
    size_t jobs_done_ = 0;
    bool quit_ = false;
    uint64_t token_ = 0;
    int64_t swapped_ = 0;
    platform::MappedFile file_;             ///< the experts section
    std::unique_ptr<core::PeerExperts> peer_;
    std::vector<kernels::cpu::NativeFmt> fmt_;   ///< [layer]
    std::unique_ptr<kernels::cpu::ExpertPool> pool_;
    uint8_t* act_ = nullptr;                ///< the quantized activation the CPU jobs share

    int64_t hits_ = 0, cpu_ = 0, ram_hits_ = 0, promoted_ = 0, fetched_ = 0, dropped_ = 0, peer_hits_ = 0, file_hits_ = 0;
};

}  // namespace basalt::models::deepseek41
