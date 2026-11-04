// src/models/deepseek41/tier.hpp - where the routed experts of a DeepSeek-V4.1 pack live while a token runs.
//
// Three places hold an expert: a slot of the VRAM tier (shared core::ExpertCache storage, per-layer LRU), the pinned
// RAM tier (a prefix of every layer's experts), and the pack file mapped read-only. The GPU computes the experts that
// sit in VRAM; the CPU pool computes the rest from RAM or the mapped file while the GPU works, and the experts it
// computed are copied into the VRAM tier on a side stream so the next token finds them there.
#pragma once

#include "basalt/core/expert_cache.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/platform/mapped_file.hpp"
#include "models/deepseek41/tier_policy.hpp"

#include <cuda_runtime.h>

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
        const uint8_t* host[kMaxK] = {};   ///< host blob of the others
        int experts[kMaxK] = {};
        bool again[kMaxK] = {};            ///< a miss that repeats one from the last kPromoteWindow tokens
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

    /// Sizes the VRAM tier from the free memory less `reserve_bytes`, loads the RAM tier and starts the CPU pool.
    bool open(uint64_t reserve_bytes, std::string& err);

    /// Gives the VRAM slots that the prompt path's workspaces occupy back to the driver (call before they are
    /// allocated); `grow` takes the memory freed after them as extra slots. Residency in those slots is dropped.
    void shrink();
    void grow();

    /// Called before each token: the promotions queued during the previous token finish and become resident, so what a
    /// lookup finds depends only on the routing history.
    void begin_token();
    /// Looks up `ids[0..k)` of `layer` and stamps them as used.
    bool lookup(int layer, const int32_t* ids, int k, Lookup& lk, std::string& err);
    /// The CPU's share: the experts of `lk` without a device blob, over the activation `x` (n_embd floats, host).
    /// Row i of `rows` (n_embd floats each) gets expert i's output.
    bool run_cpu(int layer, const float* x, const Lookup& lk, float* rows, std::string& err);
    /// run_cpu for a window: `x` holds one activation row per token (n_embd floats each, host), `use[i]` says which
    /// tokens read expert i of `lk` and where each one's output row goes in `rows`.
    bool run_cpu_window(int layer, const float* x, const Lookup& lk, const Use* use, float* rows, std::string& err);
    /// Starts the bus copy of the pinned misses of `lk` beyond the CPU's half (policy::bus_count) into the VRAM tier and makes `main` wait for the copies;
    /// those experts get a device blob in `lk`, so the GPU computes them with the hits while the CPU does the rest.
    void fetch(int layer, Lookup& lk, cudaStream_t main);
    /// Queues the pinned misses the CPU computed that missed before within kPromoteWindow tokens for a copy into the
    /// VRAM tier (side stream, off the token path, in pieces).
    void promote(int layer, const Lookup& lk);

    /// Where a resident copy of (layer, e) lives: a VRAM tier slot, or the pinned RAM tier; both null when only the
    /// file has it. Read-only: no stamp, no promotion.
    struct Source {
        const uint8_t* dev = nullptr;
        const uint8_t* pinned = nullptr;
    };
    Source locate(int layer, int e) const;

    void report(const char* tag) const;

private:
    struct Pending {
        int layer, expert, slot, event;
    };
    /// A cold pinned expert (victim) replaced by a hot file-backed one (expert), read into the layer's spare slot.
    struct Swap {
        int layer, expert, victim;
    };
    bool load_ram(uint64_t bytes, std::string& err);
    bool pinned(int layer, int e) const { return ram_idx_[(size_t) layer * n_expert_ + e] >= 0; }
    const uint8_t* blob_of(int layer, int e) const;
    const uint8_t* slot_ptr(int32_t slot) const;
    uint64_t slot_total() const { return (uint64_t) n_layers_ * (uint64_t) (per_layer_ + ext_per_layer_); }
    void drain();
    void finish_swaps();
    void start_swaps();
    void swap_worker();
    /// Copies (layer, e) from `src` into the least recently used free slot of the layer; the slot, or -1.
    int32_t claim(int layer, int e, const uint8_t* src, cudaStream_t stream);

    const DsModel& m_;
    core::ExpertCache cache_;
    int n_layers_ = 0, n_expert_ = 0;
    int64_t per_layer_ = 0;                 ///< VRAM slots each layer owns in the base arena
    core::ExpertCache ext_;                 ///< slots beside the base arena, open only while no prompt workspace is
    int64_t ext_per_layer_ = 0;             ///< slots each layer owns in `ext_`
    std::vector<int32_t> res_;              ///< [layer * n_expert + e] VRAM slot or -1
    std::vector<uint64_t> stamp_;           ///< [layer * n_expert + e] last use
    std::vector<int32_t> owner_;            ///< [slot] expert it holds or is receiving, or -1
    std::vector<uint8_t> busy_;             ///< [slot] a promotion is copying into it
    std::vector<Pending> pending_;
    std::vector<cudaEvent_t> events_;
    std::vector<int> free_events_;
    cudaStream_t copy_stream_ = nullptr;    ///< promotions, off the token path
    cudaStream_t fetch_stream_ = nullptr;   ///< copies a layer waits for (high priority)
    cudaEvent_t fetch_ev_ = nullptr;
    uint64_t tick_ = 0;
    uint64_t one_each_ = 0, reserve_ = 0;

    std::vector<uint8_t*> ram_;             ///< [layer] cover_ + 1 pinned expert slots (the last starts as the spare)
    int cover_ = 0;                         ///< experts per layer in the RAM tier
    std::vector<int32_t> ram_idx_;          ///< [layer * n_expert + e] pinned slot or -1
    std::vector<int32_t> spare_;            ///< [layer] pinned slot free for a swap, or -1
    std::vector<uint64_t> spare_since_;     ///< [layer] token at which the spare became free
    std::vector<uint64_t> last_miss_;       ///< [layer * n_expert + e] token + 1 of the last miss
    std::vector<std::pair<int, int>> want_; ///< file-backed experts that missed twice, in lookup order
    std::vector<Swap> flight_;              ///< swaps started at the previous begin_token
    std::thread worker_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::vector<Swap> jobs_;
    size_t jobs_done_ = 0;
    bool quit_ = false;
    uint64_t token_ = 0;
    int64_t swapped_ = 0;
    platform::MappedFile file_;             ///< the experts section
    std::vector<kernels::cpu::NativeFmt> fmt_;   ///< [layer]
    std::unique_ptr<kernels::cpu::ExpertPool> pool_;
    uint8_t* act_ = nullptr;                ///< the quantized activation the CPU jobs share

    int64_t hits_ = 0, cpu_ = 0, ram_hits_ = 0, promoted_ = 0, fetched_ = 0, dropped_ = 0;
};

}  // namespace basalt::models::deepseek41
